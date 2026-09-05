from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Literal

SpeechService = Literal["stt", "tts", "llm"]
SpeechOutcome = Literal["success", "error", "cancelled"]
SERVICE_LATENCY_HEADER = "x-speech-service-latency-ms"
logger = logging.getLogger("s2s-endpoint")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stats(samples: list[SpeechLatencySample], field: str) -> dict[str, float | int]:
    values = [float(value) for sample in samples if (value := getattr(sample, field)) is not None]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(mean(values), 3),
        "min": round(min(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p90": round(_percentile(values, 0.90), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
        "max": round(max(values), 3),
    }


@dataclass(frozen=True)
class SpeechLatencySample:
    timestamp_s: float
    request_id: str
    outcome: SpeechOutcome
    backend: str | None
    attempts: int
    total_ms: float
    proxy_application_ms: float
    backend_round_trip_ms: float
    backend_service_ms: float | None
    backend_transport_ms: float | None
    proxy_path_overhead_ms: float | None
    model: str | None = None
    provider: str | None = None
    pool: str | None = None
    revision: str | None = None


class SpeechProxyMetrics:
    def __init__(
        self,
        service: SpeechService,
        *,
        max_samples: int = 50_000,
        time_fn=time.time,
        route: dict[str, str] | None = None,
        parent: SpeechProxyMetrics | None = None,
    ) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        self.service = service
        self.route = route or {}
        self.parent = parent
        self.max_samples = max_samples
        self._time_fn = time_fn
        self._samples: deque[SpeechLatencySample] = deque(maxlen=max_samples)
        self._lifetime_outcomes: Counter[str] = Counter()
        self._lock = asyncio.Lock()

    async def record(self, sample: SpeechLatencySample) -> None:
        async with self._lock:
            self._samples.append(sample)
            self._lifetime_outcomes[sample.outcome] += 1
        if self.parent is not None:
            await self.parent.record(sample)

    async def snapshot(self, window_s: float) -> dict[str, object]:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        now = self._time_fn()
        cutoff = now - window_s
        async with self._lock:
            retained = list(self._samples)
            lifetime = dict(self._lifetime_outcomes)
        samples = [sample for sample in retained if sample.timestamp_s >= cutoff]
        outcomes = Counter(sample.outcome for sample in samples)
        reported = sum(sample.backend_service_ms is not None for sample in samples)
        eligible = sum(sample.backend_round_trip_ms is not None for sample in samples)
        by_backend: dict[str, dict[str, object]] = {}
        for pool, backend in sorted({(sample.pool or "", sample.backend) for sample in samples if sample.backend}):
            backend_samples = [
                sample for sample in samples if sample.backend == backend and (sample.pool or "") == pool
            ]
            key = f"{pool}/{backend}" if pool else backend
            by_backend[key] = self._latency_summary(backend_samples)

        return {
            "status": "ok",
            "service": self.service,
            **self.route,
            "phase": {
                "stt": "transcription",
                "tts": "first_audio",
                "llm": "first_token",
            }[self.service],
            "generated_at_s": now,
            "window_s": window_s,
            "retained_samples": len(retained),
            "max_samples": self.max_samples,
            "requests": {
                "window": len(samples),
                "successes": outcomes["success"],
                "errors": outcomes["error"],
                "cancellations": outcomes["cancelled"],
                "lifetime": lifetime,
            },
            "service_timing_coverage": {
                "reported": reported,
                "eligible": eligible,
                "ratio": round(reported / eligible, 4) if eligible else 0.0,
            },
            "latency_ms": self._latency_summary(samples),
            "backends": by_backend,
        }

    @staticmethod
    def _latency_summary(samples: list[SpeechLatencySample]) -> dict[str, object]:
        return {
            "total": _stats(samples, "total_ms"),
            "proxy_application": _stats(samples, "proxy_application_ms"),
            "backend_round_trip": _stats(samples, "backend_round_trip_ms"),
            "backend_service": _stats(samples, "backend_service_ms"),
            "backend_transport": _stats(samples, "backend_transport_ms"),
            "proxy_path_overhead": _stats(samples, "proxy_path_overhead_ms"),
        }


class SpeechRequestTrace:
    """Shared STT/TTS request trace and latency decomposition."""

    def __init__(
        self,
        metrics: SpeechProxyMetrics,
        *,
        request_id: str | None = None,
        monotonic_fn=time.monotonic,
        wall_time_fn=time.time,
    ) -> None:
        self.metrics = metrics
        self.request_id = request_id or uuid.uuid4().hex
        self._monotonic_fn = monotonic_fn
        self._wall_time_fn = wall_time_fn
        self._started = monotonic_fn()
        self._upstream_started: float | None = None
        self._upstream_ms = 0.0
        self._service_ms = 0.0
        self._reported_attempts = 0
        self._attempts = 0
        self._backend: str | None = None
        self._sample: SpeechLatencySample | None = None

    def start_upstream(self, backend: str) -> None:
        if self._upstream_started is not None:
            raise RuntimeError("upstream attempt is already running")
        self._backend = backend
        self._attempts += 1
        self._upstream_started = self._monotonic_fn()

    def finish_upstream(self, service_latency_header: str | None = None) -> None:
        if self._upstream_started is None:
            return
        self._upstream_ms += max((self._monotonic_fn() - self._upstream_started) * 1000.0, 0.0)
        self._upstream_started = None
        if service_latency_header is None:
            return
        try:
            service_ms = float(service_latency_header)
        except (TypeError, ValueError):
            return
        if service_ms < 0:
            return
        self._service_ms += service_ms
        self._reported_attempts += 1

    async def record(self, outcome: SpeechOutcome) -> SpeechLatencySample:
        if self._sample is not None:
            return self._sample
        self.finish_upstream()
        total_ms = max((self._monotonic_fn() - self._started) * 1000.0, 0.0)
        proxy_ms = max(total_ms - self._upstream_ms, 0.0)
        service_ms = self._service_ms if self._attempts and self._reported_attempts == self._attempts else None
        transport_ms = max(self._upstream_ms - service_ms, 0.0) if service_ms is not None else None
        path_overhead_ms = max(total_ms - service_ms, 0.0) if service_ms is not None else None
        self._sample = SpeechLatencySample(
            timestamp_s=self._wall_time_fn(),
            request_id=self.request_id,
            outcome=outcome,
            backend=self._backend,
            attempts=self._attempts,
            total_ms=total_ms,
            proxy_application_ms=proxy_ms,
            backend_round_trip_ms=self._upstream_ms,
            backend_service_ms=service_ms,
            backend_transport_ms=transport_ms,
            proxy_path_overhead_ms=path_overhead_ms,
            **self.metrics.route,
        )
        await self.metrics.record(self._sample)
        logger.info(
            "Speech proxy result service=%s request_id=%s outcome=%s backend=%s total_ms=%.3f "
            "overhead_ms=%s proxy_ms=%.3f backend_ms=%.3f service_ms=%s transport_ms=%s attempts=%s "
            "model=%s provider=%s pool=%s revision=%s",
            self.metrics.service,
            self._sample.request_id,
            self._sample.outcome,
            self._sample.backend,
            self._sample.total_ms,
            self._sample.proxy_path_overhead_ms,
            self._sample.proxy_application_ms,
            self._sample.backend_round_trip_ms,
            self._sample.backend_service_ms,
            self._sample.backend_transport_ms,
            self._sample.attempts,
            self._sample.model,
            self._sample.provider,
            self._sample.pool,
            self._sample.revision,
        )
        return self._sample

    @property
    def sample(self) -> SpeechLatencySample | None:
        return self._sample


def sample_headers(sample: SpeechLatencySample) -> dict[str, str]:
    timings = [
        f"speech-total;dur={sample.total_ms:.3f}",
        f"speech-proxy;dur={sample.proxy_application_ms:.3f}",
        f"speech-backend;dur={sample.backend_round_trip_ms:.3f}",
    ]
    headers = {
        "X-Speech-Request-Id": sample.request_id,
        "X-Speech-Total-Latency-Ms": f"{sample.total_ms:.3f}",
        "X-Speech-Proxy-Latency-Ms": f"{sample.proxy_application_ms:.3f}",
        "X-Speech-Backend-Latency-Ms": f"{sample.backend_round_trip_ms:.3f}",
    }
    if sample.backend:
        headers["X-Speech-Backend"] = sample.backend
    if sample.backend_service_ms is not None and sample.backend_transport_ms is not None:
        timings.extend(
            (
                f"speech-service;dur={sample.backend_service_ms:.3f}",
                f"speech-transport;dur={sample.backend_transport_ms:.3f}",
            )
        )
        headers["X-Speech-Service-Latency-Ms"] = f"{sample.backend_service_ms:.3f}"
        headers["X-Speech-Transport-Latency-Ms"] = f"{sample.backend_transport_ms:.3f}"
        if sample.proxy_path_overhead_ms is not None:
            timings.append(f"speech-overhead;dur={sample.proxy_path_overhead_ms:.3f}")
            headers["X-Speech-Overhead-Latency-Ms"] = f"{sample.proxy_path_overhead_ms:.3f}"
    headers["Server-Timing"] = ", ".join(timings)
    return headers


def sample_as_dict(sample: SpeechLatencySample) -> dict[str, object]:
    return asdict(sample)

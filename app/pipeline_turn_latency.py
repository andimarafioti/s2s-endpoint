from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Literal

TURN_LATENCY_METADATA_KEY = "speech_to_speech.turn_latency"
TURN_LATENCY_EVENT = "turn_latency"
TurnLatencyStatus = Literal["completed", "cancelled", "failed", "incomplete"]
_STATUSES = {"completed", "cancelled", "failed", "incomplete"}
_LATENCY_FIELDS = ("stt_s", "llm_ttft_s", "llm_s", "tts_ttfa_s", "e2e_s", "mlx_lock_wait_s")


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"{key} must be a non-empty string of at most 256 characters")
    return value


def _optional_seconds(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a non-negative number or null")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{key} must be a finite non-negative number or null")
    return seconds


@dataclass(frozen=True)
class PipelineTurnLatency:
    version: int
    turn_id: str
    turn_revision: int
    response_key: str
    status: TurnLatencyStatus
    stt_s: float | None
    llm_ttft_s: float | None
    llm_s: float | None
    tts_ttfa_s: float | None
    e2e_s: float | None
    mlx_lock_wait_s: float | None

    @classmethod
    def from_payload(cls, payload: object) -> "PipelineTurnLatency":
        if not isinstance(payload, dict):
            raise ValueError("turn latency must be a JSON object")
        if payload.get("version") != 1:
            raise ValueError("turn latency version must be 1")

        revision = payload.get("turn_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("turn_revision must be a non-negative integer")
        status = payload.get("status")
        if status not in _STATUSES:
            raise ValueError("turn latency status is invalid")

        return cls(
            version=1,
            turn_id=_required_text(payload, "turn_id"),
            turn_revision=revision,
            response_key=_required_text(payload, "response_key"),
            status=status,
            **{field: _optional_seconds(payload, field) for field in _LATENCY_FIELDS},
        )

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def extract_pipeline_turn_latency(message: str) -> PipelineTurnLatency | None:
    """Extract the server-owned measurement from a terminal Realtime event."""
    try:
        event = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(event, dict) or event.get("type") != "response.done":
        return None
    response = event.get("response")
    if not isinstance(response, dict):
        return None
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        return None
    encoded = metadata.get(TURN_LATENCY_METADATA_KEY)
    if not isinstance(encoded, str):
        return None
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("turn latency metadata is not valid JSON") from exc
    return PipelineTurnLatency.from_payload(payload)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stats(samples: list["PipelineTurnLatencySample"], field: str) -> dict[str, float | int]:
    values = [float(value) * 1000.0 for sample in samples if (value := getattr(sample.latency, field)) is not None]
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
class PipelineTurnLatencySample:
    timestamp_s: float
    session_id: str
    latency: PipelineTurnLatency


class PipelineTurnLatencyMetrics:
    """Bounded, deduplicated measurements reported by active pipeline sessions."""

    def __init__(self, *, max_samples: int = 50_000, time_fn=time.time) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        self.max_samples = max_samples
        self._time_fn = time_fn
        self._samples: deque[PipelineTurnLatencySample] = deque()
        self._keys: set[tuple[str, str]] = set()
        self._duplicates = 0
        self._lock = asyncio.Lock()

    async def record(self, session_id: str, latency: PipelineTurnLatency) -> bool:
        key = (session_id, latency.response_key)
        async with self._lock:
            if key in self._keys:
                self._duplicates += 1
                return False
            if len(self._samples) >= self.max_samples:
                removed = self._samples.popleft()
                self._keys.discard((removed.session_id, removed.latency.response_key))
            self._samples.append(PipelineTurnLatencySample(self._time_fn(), session_id, latency))
            self._keys.add(key)
            return True

    async def snapshot(self, window_s: float) -> dict[str, object]:
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        cutoff = self._time_fn() - window_s
        async with self._lock:
            retained = list(self._samples)
            duplicates = self._duplicates
        samples = [sample for sample in retained if sample.timestamp_s >= cutoff]
        statuses = Counter(sample.latency.status for sample in samples)
        return {
            "status": "ok",
            "phase": "terminal_response",
            "window_s": window_s,
            "retained_samples": len(retained),
            "max_samples": self.max_samples,
            "responses": {
                "window": len(samples),
                "completed": statuses["completed"],
                "cancelled": statuses["cancelled"],
                "failed": statuses["failed"],
                "incomplete": statuses["incomplete"],
                "duplicates_ignored_lifetime": duplicates,
            },
            "latency_ms": {field.removesuffix("_s"): _stats(samples, field) for field in _LATENCY_FIELDS},
        }

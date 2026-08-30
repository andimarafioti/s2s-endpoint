from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Literal

import httpx

from app.app_utils import cancel_and_await

SpeechService = Literal["stt", "tts"]


class NoSpeechBackendAvailable(RuntimeError):
    """Raised when no ready backend can accept the requested work."""


@dataclass(frozen=True)
class SpeechBackendConfig:
    name: str
    url: str


@dataclass
class SpeechBackendState:
    config: SpeechBackendConfig
    ready: bool = False
    draining: bool = False
    active_requests: int = 0
    active_work: float = 0.0
    ewma_latency: float | None = None
    requests: int = 0
    successes: int = 0
    errors: int = 0
    cancellations: int = 0
    consecutive_failures: int = 0
    last_selected: int = 0
    last_health_error: str | None = None
    last_health_at: float | None = None


@dataclass(frozen=True)
class SpeechBackendSnapshot:
    name: str
    url: str
    ready: bool
    draining: bool
    active_requests: int
    active_work: float
    target_work: float
    ewma_latency: float | None
    requests: int
    successes: int
    errors: int
    cancellations: int
    consecutive_failures: int
    last_health_error: str | None
    last_health_at: float | None


class SpeechBackendLease:
    def __init__(
        self,
        pool: SpeechBackendPool,
        backend_name: str,
        backend_url: str,
        work: float,
    ) -> None:
        self._pool = pool
        self.backend_name = backend_name
        self.backend_url = backend_url
        self.work = work
        self.started_at = time.monotonic()
        self._released = False
        self._release_lock = asyncio.Lock()

    async def release(
        self,
        *,
        success: bool,
        cancelled: bool = False,
        latency: float | None = None,
        retryable_failure: bool = False,
        error: str | None = None,
    ) -> None:
        async with self._release_lock:
            if self._released:
                return
            self._released = True
            if latency is None:
                latency = time.monotonic() - self.started_at
            await self._pool.release(
                self.backend_name,
                work=self.work,
                success=success,
                cancelled=cancelled,
                latency=latency,
                retryable_failure=retryable_failure,
                error=error,
            )


@dataclass(frozen=True)
class SpeechBackendPoolSettings:
    service: SpeechService
    target_work: float
    latency_target: float
    latency_weight: float = 0.25
    ewma_alpha: float = 0.2
    failure_threshold: int = 2
    health_path: str = "/health"
    health_interval_s: float = 10.0
    health_timeout_s: float = 5.0
    backend_api_key: str | None = None
    tts_warmup_enabled: bool = True
    tts_warmup_timeout_s: float = 120.0
    tts_warmup_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_warmup_voice: str = "aiden"
    tts_warmup_language: str = "English"

    def __post_init__(self) -> None:
        if self.service not in {"stt", "tts"}:
            raise ValueError("service must be 'stt' or 'tts'")
        if self.target_work <= 0:
            raise ValueError("target_work must be > 0")
        if self.latency_target <= 0:
            raise ValueError("latency_target must be > 0")
        if self.latency_weight < 0:
            raise ValueError("latency_weight must be >= 0")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.health_interval_s <= 0 or self.health_timeout_s <= 0:
            raise ValueError("health intervals and timeouts must be > 0")


class SpeechBackendPool:
    """Atomic, request-level routing across a fleet of speech workers."""

    def __init__(
        self,
        backends: tuple[SpeechBackendConfig, ...],
        settings: SpeechBackendPoolSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not backends:
            raise ValueError("at least one speech backend is required")
        names = [backend.name for backend in backends]
        if len(names) != len(set(names)):
            raise ValueError("speech backend names must be unique")
        urls = [backend.url for backend in backends]
        if len(urls) != len(set(urls)):
            raise ValueError("speech backend URLs must be unique")
        self.settings = settings
        self._states = {backend.name: SpeechBackendState(config=backend) for backend in backends}
        self._lock = asyncio.Lock()
        self._selection_counter = 0
        self._health_task: asyncio.Task | None = None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(follow_redirects=True)

    async def start(self) -> None:
        await self.refresh_health()
        self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        await cancel_and_await(self._health_task)
        self._health_task = None
        if self._owns_client:
            await self._client.aclose()

    async def reserve(
        self,
        work: float,
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> SpeechBackendLease:
        if work <= 0:
            raise ValueError("work must be > 0")
        async with self._lock:
            candidates = [
                state
                for state in self._states.values()
                if state.ready and not state.draining and state.config.name not in exclude
            ]
            if not candidates:
                raise NoSpeechBackendAvailable("no ready speech backend is available")
            state = min(candidates, key=self._score)
            self._selection_counter += 1
            state.last_selected = self._selection_counter
            state.active_requests += 1
            state.active_work += work
            state.requests += 1
            return SpeechBackendLease(
                self,
                state.config.name,
                state.config.url,
                work,
            )

    async def release(
        self,
        backend_name: str,
        *,
        work: float,
        success: bool,
        cancelled: bool,
        latency: float,
        retryable_failure: bool,
        error: str | None,
    ) -> None:
        async with self._lock:
            state = self._states[backend_name]
            state.active_requests = max(state.active_requests - 1, 0)
            state.active_work = max(state.active_work - work, 0.0)
            if cancelled:
                state.cancellations += 1
                state.consecutive_failures = 0
                return
            if success:
                state.successes += 1
                state.consecutive_failures = 0
                state.last_health_error = None
                state.ewma_latency = self._ewma(state.ewma_latency, latency)
                return
            state.errors += 1
            if error:
                state.last_health_error = error
            if retryable_failure:
                state.consecutive_failures += 1
                if state.consecutive_failures >= self.settings.failure_threshold:
                    state.ready = False

    async def set_draining(self, backend_name: str, draining: bool) -> None:
        async with self._lock:
            self._states[backend_name].draining = draining

    async def refresh_health(self) -> None:
        await asyncio.gather(*(self._refresh_one(name) for name in self._states))

    async def snapshots(self) -> tuple[SpeechBackendSnapshot, ...]:
        async with self._lock:
            return tuple(
                SpeechBackendSnapshot(
                    name=state.config.name,
                    url=state.config.url,
                    ready=state.ready,
                    draining=state.draining,
                    active_requests=state.active_requests,
                    active_work=round(state.active_work, 3),
                    target_work=self.settings.target_work,
                    ewma_latency=state.ewma_latency,
                    requests=state.requests,
                    successes=state.successes,
                    errors=state.errors,
                    cancellations=state.cancellations,
                    consecutive_failures=state.consecutive_failures,
                    last_health_error=state.last_health_error,
                    last_health_at=state.last_health_at,
                )
                for state in self._states.values()
            )

    async def healthy(self) -> bool:
        async with self._lock:
            return any(state.ready and not state.draining for state in self._states.values())

    def _score(self, state: SpeechBackendState) -> tuple[float, int]:
        work_score = state.active_work / self.settings.target_work
        latency_score = state.ewma_latency / self.settings.latency_target if state.ewma_latency is not None else 0.0
        return work_score + self.settings.latency_weight * latency_score, state.last_selected

    def _ewma(self, previous: float | None, current: float) -> float:
        if previous is None:
            return current
        alpha = self.settings.ewma_alpha
        return alpha * current + (1 - alpha) * previous

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.health_interval_s)
            await self.refresh_health()

    async def _refresh_one(self, backend_name: str) -> None:
        async with self._lock:
            state = self._states[backend_name]
            url = state.config.url
            was_ready = state.ready
        error: str | None = None
        ready = False
        headers = self._authorization_headers()
        try:
            response = await self._client.get(
                f"{url.rstrip('/')}/{self.settings.health_path.lstrip('/')}",
                headers=headers,
                timeout=self.settings.health_timeout_s,
            )
            response.raise_for_status()
            if self.settings.service == "tts" and self.settings.tts_warmup_enabled and not was_ready:
                await self._warm_tts(url, headers)
            ready = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        async with self._lock:
            state = self._states[backend_name]
            state.last_health_at = time.time()
            state.ready = ready
            state.last_health_error = error
            if ready:
                state.consecutive_failures = 0

    async def _warm_tts(self, url: str, headers: dict[str, str]) -> None:
        response = await self._client.post(
            f"{url.rstrip('/')}/v1/audio/speech",
            headers=headers,
            json={
                "model": self.settings.tts_warmup_model,
                "voice": self.settings.tts_warmup_voice,
                "language": self.settings.tts_warmup_language,
                "input": "Ready.",
                "response_format": "pcm",
                "stream": True,
            },
            timeout=self.settings.tts_warmup_timeout_s,
        )
        response.raise_for_status()
        if not response.content:
            raise RuntimeError("TTS warmup returned an empty response")

    def _authorization_headers(self) -> dict[str, str]:
        if not self.settings.backend_api_key:
            return {}
        return {"Authorization": f"Bearer {self.settings.backend_api_key}"}

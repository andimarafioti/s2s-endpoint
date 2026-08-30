from __future__ import annotations

import asyncio
import io
import time
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import UploadFile

from app.app_utils import build_lifespan, env_bool, env_optional, env_text, setup_logging
from app.speech_proxy_router import (
    NoSpeechBackendAvailable,
    SpeechBackendConfig,
    SpeechBackendLease,
    SpeechBackendPool,
    SpeechBackendPoolSettings,
    SpeechService,
)

logger = setup_logging()
APP_ROLE = "speech_proxy"
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
RESPONSE_HEADERS = frozenset({"content-type", "content-disposition", "x-request-id"})
RETRYABLE_BAD_REQUEST_MARKERS = (b"paused", b"scaled to zero", b"unavailable")


def _positive_float(name: str, value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0")
    return parsed


def _positive_int(name: str, value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be >= 1")
    return parsed


def parse_backends(value: str) -> tuple[SpeechBackendConfig, ...]:
    backends: list[SpeechBackendConfig] = []
    for index, raw_entry in enumerate(value.split(","), start=1):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name, url = (part.strip() for part in entry.split("=", 1))
        else:
            name, url = f"backend-{index:02d}", entry
        if not name or not url:
            raise ValueError(f"Invalid SPEECH_BACKENDS entry: {entry!r}")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Speech backend URL must use http or https: {url!r}")
        backends.append(SpeechBackendConfig(name=name, url=url.rstrip("/")))
    if not backends:
        raise ValueError("SPEECH_BACKENDS must contain at least one backend")
    return tuple(backends)


@dataclass(frozen=True)
class SpeechProxySettings:
    service: SpeechService
    backends: tuple[SpeechBackendConfig, ...]
    backend_api_key: str | None = None
    target_work: float = 8.0
    max_work: float = 16.0
    latency_target: float = 0.5
    latency_weight: float = 0.25
    ewma_alpha: float = 0.2
    failure_threshold: int = 2
    health_path: str = "/health"
    health_interval_s: float = 10.0
    health_timeout_s: float = 5.0
    request_timeout_s: float = 120.0
    max_attempts: int = 2
    stt_audio_equivalent_s: float = 5.0
    tts_warmup_enabled: bool = True
    tts_warmup_timeout_s: float = 120.0
    tts_warmup_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_warmup_voice: str = "aiden"
    tts_warmup_language: str = "English"

    def __post_init__(self) -> None:
        SpeechBackendPoolSettings(
            service=self.service,
            target_work=self.target_work,
            max_work=self.max_work,
            latency_target=self.latency_target,
            latency_weight=self.latency_weight,
            ewma_alpha=self.ewma_alpha,
            failure_threshold=self.failure_threshold,
            health_path=self.health_path,
            health_interval_s=self.health_interval_s,
            health_timeout_s=self.health_timeout_s,
            backend_api_key=self.backend_api_key,
            tts_warmup_enabled=self.tts_warmup_enabled,
            tts_warmup_timeout_s=self.tts_warmup_timeout_s,
            tts_warmup_model=self.tts_warmup_model,
            tts_warmup_voice=self.tts_warmup_voice,
            tts_warmup_language=self.tts_warmup_language,
        )
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be > 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.stt_audio_equivalent_s <= 0:
            raise ValueError("stt_audio_equivalent_s must be > 0")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SpeechProxySettings:
        service = env_text("SPEECH_PROXY_SERVICE", environ=environ).lower()
        if service not in {"stt", "tts"}:
            raise ValueError("SPEECH_PROXY_SERVICE must be 'stt' or 'tts'")
        defaults = {
            "stt": {"target_work": "96", "max_work": "128", "latency_target": "0.1"},
            "tts": {"target_work": "8", "max_work": "16", "latency_target": "0.5"},
        }[service]
        backend_api_key = env_optional("SPEECH_BACKEND_API_KEY", environ=environ)
        if backend_api_key is None:
            backend_api_key = env_optional("HF_TOKEN", environ=environ)
        return cls(
            service=service,  # type: ignore[arg-type]
            backends=parse_backends(env_text("SPEECH_BACKENDS", environ=environ)),
            backend_api_key=backend_api_key,
            target_work=_positive_float(
                "SPEECH_TARGET_WORK",
                env_text("SPEECH_TARGET_WORK", defaults["target_work"], environ=environ),
            ),
            max_work=_positive_float(
                "SPEECH_MAX_WORK",
                env_text("SPEECH_MAX_WORK", defaults["max_work"], environ=environ),
            ),
            latency_target=_positive_float(
                "SPEECH_LATENCY_TARGET",
                env_text("SPEECH_LATENCY_TARGET", defaults["latency_target"], environ=environ),
            ),
            latency_weight=float(env_text("SPEECH_LATENCY_WEIGHT", "0.25", environ=environ)),
            ewma_alpha=float(env_text("SPEECH_EWMA_ALPHA", "0.2", environ=environ)),
            failure_threshold=_positive_int(
                "SPEECH_FAILURE_THRESHOLD",
                env_text("SPEECH_FAILURE_THRESHOLD", "2", environ=environ),
            ),
            health_path=env_text("SPEECH_HEALTH_PATH", "/health", environ=environ),
            health_interval_s=_positive_float(
                "SPEECH_HEALTH_INTERVAL_S",
                env_text("SPEECH_HEALTH_INTERVAL_S", "10", environ=environ),
            ),
            health_timeout_s=_positive_float(
                "SPEECH_HEALTH_TIMEOUT_S",
                env_text("SPEECH_HEALTH_TIMEOUT_S", "5", environ=environ),
            ),
            request_timeout_s=_positive_float(
                "SPEECH_REQUEST_TIMEOUT_S",
                env_text("SPEECH_REQUEST_TIMEOUT_S", "120", environ=environ),
            ),
            max_attempts=_positive_int(
                "SPEECH_MAX_ATTEMPTS",
                env_text("SPEECH_MAX_ATTEMPTS", "2", environ=environ),
            ),
            stt_audio_equivalent_s=_positive_float(
                "STT_AUDIO_EQUIVALENT_S",
                env_text("STT_AUDIO_EQUIVALENT_S", "5", environ=environ),
            ),
            tts_warmup_enabled=env_bool("TTS_WARMUP_ENABLED", True, environ=environ),
            tts_warmup_timeout_s=_positive_float(
                "TTS_WARMUP_TIMEOUT_S",
                env_text("TTS_WARMUP_TIMEOUT_S", "120", environ=environ),
            ),
            tts_warmup_model=env_text(
                "TTS_WARMUP_MODEL",
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                environ=environ,
            ),
            tts_warmup_voice=env_text("TTS_WARMUP_VOICE", "aiden", environ=environ),
            tts_warmup_language=env_text("TTS_WARMUP_LANGUAGE", "English", environ=environ),
        )

    def pool_settings(self) -> SpeechBackendPoolSettings:
        return SpeechBackendPoolSettings(
            service=self.service,
            target_work=self.target_work,
            max_work=self.max_work,
            latency_target=self.latency_target,
            latency_weight=self.latency_weight,
            ewma_alpha=self.ewma_alpha,
            failure_threshold=self.failure_threshold,
            health_path=self.health_path,
            health_interval_s=self.health_interval_s,
            health_timeout_s=self.health_timeout_s,
            backend_api_key=self.backend_api_key,
            tts_warmup_enabled=self.tts_warmup_enabled,
            tts_warmup_timeout_s=self.tts_warmup_timeout_s,
            tts_warmup_model=self.tts_warmup_model,
            tts_warmup_voice=self.tts_warmup_voice,
            tts_warmup_language=self.tts_warmup_language,
        )


@dataclass
class SpeechProxyDependencies:
    pool: SpeechBackendPool
    client: httpx.AsyncClient
    owns_client: bool = True

    async def start(self) -> None:
        await self.pool.start()

    async def stop(self) -> None:
        await self.pool.stop()
        if self.owns_client:
            await self.client.aclose()


def create_dependencies(settings: SpeechProxySettings) -> SpeechProxyDependencies:
    timeout = httpx.Timeout(settings.request_timeout_s, connect=min(settings.request_timeout_s, 10.0))
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    pool = SpeechBackendPool(settings.backends, settings.pool_settings(), client=client)
    return SpeechProxyDependencies(pool=pool, client=client)


def _backend_headers(settings: SpeechProxySettings) -> dict[str, str]:
    headers = {"Accept": "*/*"}
    if settings.backend_api_key:
        headers["Authorization"] = f"Bearer {settings.backend_api_key}"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {name: value for name, value in response.headers.items() if name.lower() in RESPONSE_HEADERS}


def _wav_duration(content: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            rate = audio.getframerate()
            if rate <= 0:
                return None
            return audio.getnframes() / rate
    except (EOFError, wave.Error):
        return None


async def _stt_form(
    request: Request,
) -> tuple[list[tuple[str, tuple[str | None, bytes | str, str | None]]], float]:
    form = await request.form()
    multipart: list[tuple[str, tuple[str | None, bytes | str, str | None]]] = []
    duration_s: float | None = None
    for name, value in form.multi_items():
        if isinstance(value, UploadFile):
            try:
                content = await value.read()
            finally:
                await value.close()
            multipart.append(
                (
                    name,
                    (
                        value.filename or "upload",
                        content,
                        value.content_type or "application/octet-stream",
                    ),
                )
            )
            if name == "file":
                duration_s = _wav_duration(content)
        else:
            multipart.append((name, (None, str(value), None)))
    if not any(filename is not None for _, (filename, _, _) in multipart):
        raise HTTPException(status_code=422, detail="STT request must include an uploaded file")
    return multipart, duration_s or 0.0


def _upstream_error(status_code: int, body: bytes, headers: dict[str, str]) -> Response:
    return Response(content=body, status_code=status_code, headers=headers)


def _retryable_response(status_code: int, body: bytes) -> bool:
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    lowered = body.lower()
    return status_code == 400 and any(marker in lowered for marker in RETRYABLE_BAD_REQUEST_MARKERS)


def _unavailable_response(service: SpeechService) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": f"No {service.upper()} backend is currently available",
                "type": "service_unavailable",
                "code": "speech_backend_unavailable",
                "param": None,
            }
        },
        status_code=503,
    )


async def _proxy_stt(
    request: Request,
    settings: SpeechProxySettings,
    dependencies: SpeechProxyDependencies,
) -> Response:
    multipart, duration_s = await _stt_form(request)
    work = max(duration_s / settings.stt_audio_equivalent_s, 1.0)
    excluded: set[str] = set()
    last_error = "no backend attempt was made"
    for _ in range(settings.max_attempts):
        try:
            lease = await dependencies.pool.reserve(work, exclude=frozenset(excluded))
        except NoSpeechBackendAvailable:
            break
        excluded.add(lease.backend_name)
        started = time.monotonic()
        try:
            response = await dependencies.client.post(
                f"{lease.backend_url}/v1/audio/transcriptions",
                headers=_backend_headers(settings),
                files=multipart,
                timeout=settings.request_timeout_s,
            )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("STT backend %s transport failed: %s", lease.backend_name, last_error)
            await lease.release(success=False, retryable_failure=True, error=last_error)
            continue
        elapsed = time.monotonic() - started
        success = 200 <= response.status_code < 300
        retryable = _retryable_response(response.status_code, response.content)
        latency_metric = elapsed / duration_s if duration_s > 0 else elapsed
        await lease.release(
            success=success,
            latency=latency_metric,
            retryable_failure=retryable,
            error=None if success else f"HTTP {response.status_code}",
        )
        if retryable:
            last_error = f"backend returned HTTP {response.status_code}"
            logger.warning("STT backend %s %s", lease.backend_name, last_error)
            continue
        return _upstream_error(response.status_code, response.content, _response_headers(response))
    logger.error("No STT backend completed the request: %s", last_error)
    return _unavailable_response("stt")


async def _read_first_chunk(response: httpx.Response) -> tuple[bytes, Any]:
    iterator = response.aiter_raw()
    while True:
        chunk = await anext(iterator)
        if chunk:
            return chunk, iterator


async def _tts_stream(
    first_chunk: bytes,
    iterator: Any,
    response: httpx.Response,
    lease: SpeechBackendLease,
    first_audio_latency: float,
):
    completed = False
    try:
        yield first_chunk
        async for chunk in iterator:
            if chunk:
                yield chunk
        completed = True
    except asyncio.CancelledError:
        await lease.release(success=False, cancelled=True, latency=first_audio_latency)
        raise
    except Exception as exc:
        await lease.release(
            success=False,
            latency=first_audio_latency,
            retryable_failure=True,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        await response.aclose()
        if completed:
            await lease.release(success=True, latency=first_audio_latency)
        else:
            await lease.release(success=False, cancelled=True, latency=first_audio_latency)


async def _proxy_tts(
    request: Request,
    settings: SpeechProxySettings,
    dependencies: SpeechProxyDependencies,
) -> Response:
    body = await request.body()
    excluded: set[str] = set()
    last_error = "no backend attempt was made"
    for _ in range(settings.max_attempts):
        try:
            lease = await dependencies.pool.reserve(1.0, exclude=frozenset(excluded))
        except NoSpeechBackendAvailable:
            break
        excluded.add(lease.backend_name)
        started = time.monotonic()
        upstream_request = dependencies.client.build_request(
            "POST",
            f"{lease.backend_url}/v1/audio/speech",
            headers={
                **_backend_headers(settings),
                "Content-Type": request.headers.get("content-type", "application/json"),
            },
            content=body,
            timeout=settings.request_timeout_s,
        )
        try:
            response = await dependencies.client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("TTS backend %s transport failed: %s", lease.backend_name, last_error)
            await lease.release(success=False, retryable_failure=True, error=last_error)
            continue
        if not 200 <= response.status_code < 300:
            response_body = await response.aread()
            headers = _response_headers(response)
            retryable = _retryable_response(response.status_code, response_body)
            await response.aclose()
            await lease.release(
                success=False,
                retryable_failure=retryable,
                error=f"HTTP {response.status_code}",
            )
            if retryable:
                last_error = f"backend returned HTTP {response.status_code}"
                logger.warning("TTS backend %s %s", lease.backend_name, last_error)
                continue
            return _upstream_error(response.status_code, response_body, headers)
        try:
            first_chunk, iterator = await _read_first_chunk(response)
        except (httpx.HTTPError, httpx.StreamError, StopAsyncIteration) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("TTS backend %s failed before first audio: %s", lease.backend_name, last_error)
            await response.aclose()
            await lease.release(success=False, retryable_failure=True, error=last_error)
            continue
        first_audio_latency = time.monotonic() - started
        return StreamingResponse(
            _tts_stream(first_chunk, iterator, response, lease, first_audio_latency),
            status_code=response.status_code,
            headers=_response_headers(response),
        )
    logger.error("No TTS backend started a response: %s", last_error)
    return _unavailable_response("tts")


def create_app(
    settings: SpeechProxySettings,
    dependencies: SpeechProxyDependencies | None = None,
) -> FastAPI:
    dependencies = dependencies or create_dependencies(settings)

    app = FastAPI(
        title=f"S2S {settings.service.upper()} proxy",
        lifespan=build_lifespan(dependencies),
    )
    app.state.settings = settings
    app.state.dependencies = dependencies

    @app.get("/")
    async def root():
        return {
            "message": f"s2s {settings.service} proxy is up",
            "role": APP_ROLE,
            "service": settings.service,
            "health": "/health",
        }

    @app.get("/health")
    async def health():
        snapshots = await dependencies.pool.snapshots()
        payload = {
            "status": "ok" if await dependencies.pool.healthy() else "unhealthy",
            "role": APP_ROLE,
            "service": settings.service,
            "ready_backends": sum(snapshot.ready and not snapshot.draining for snapshot in snapshots),
            "backends": [asdict(snapshot) for snapshot in snapshots],
        }
        if payload["ready_backends"] == 0:
            return JSONResponse(payload, status_code=503)
        return payload

    if settings.service == "stt":

        @app.post("/v1/audio/transcriptions")
        async def transcriptions(request: Request):
            return await _proxy_stt(request, settings, dependencies)

    else:

        @app.post("/v1/audio/speech")
        async def speech(request: Request):
            return await _proxy_tts(request, settings, dependencies)

    return app

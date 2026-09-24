from __future__ import annotations

import asyncio
import hmac
import io
import json
import math
import time
import uuid
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import UploadFile

from app.app_utils import build_lifespan, cancel_and_await, env_bool, env_optional, env_text, setup_logging
from app.endpoint_pool_router import HuggingFaceEndpointController
from app.speech_proxy_metrics import (
    SERVICE_LATENCY_HEADER,
    SpeechProxyMetrics,
    SpeechRequestTrace,
    sample_headers,
)
from app.speech_proxy_router import (
    NoSpeechBackendAvailable,
    SpeechBackendConfig,
    SpeechBackendLease,
    SpeechBackendPool,
    SpeechBackendPoolSettings,
    SpeechPoolCapacityExceeded,
    SpeechService,
)
from app.speech_route_catalog import SessionDemandUpdate, SpeechRoute, SpeechRouteCatalog
from app.speech_worker_lifecycle import SpeechWorkerLifecycle, WorkerLifecycleSettings

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


def parse_backends(value: str, *, managed: bool = False) -> tuple[SpeechBackendConfig, ...]:
    backends: list[SpeechBackendConfig] = []
    for index, raw_entry in enumerate(value.split(","), start=1):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name, url = (part.strip() for part in entry.split("=", 1))
        elif managed and not entry.startswith(("http://", "https://")):
            name, url = entry, ""
        else:
            name, url = f"backend-{index:02d}", entry
        if not name or (not url and not managed):
            raise ValueError(f"Invalid SPEECH_BACKENDS entry: {entry!r}")
        if url and not url.startswith(("http://", "https://")):
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
    max_work: float | None = None
    latency_target: float = 0.5
    latency_weight: float = 0.25
    ewma_alpha: float = 0.2
    failure_threshold: int = 2
    health_path: str = "/health"
    health_interval_s: float = 10.0
    health_timeout_s: float = 5.0
    request_timeout_s: float = 120.0
    max_connections: int = 1024
    max_keepalive_connections: int = 256
    max_attempts: int = 2
    stt_audio_equivalent_s: float = 5.0
    tts_warmup_enabled: bool = True
    tts_warmup_timeout_s: float = 120.0
    tts_warmup_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    tts_warmup_voice: str = "aiden"
    tts_warmup_language: str = "English"
    llm_warmup_enabled: bool = True
    llm_warmup_timeout_s: float = 120.0
    llm_warmup_model: str = "nvidia/Gemma-4-26B-A4B-NVFP4"
    lifecycle: WorkerLifecycleSettings | None = None
    control_token: str | None = None
    endpoint_namespace: str | None = None
    catalog: SpeechRouteCatalog | None = None
    routes: tuple[SpeechProxySettings, ...] = ()
    route: SpeechRoute | None = None
    access_api_key: str | None = field(default=None, repr=False)
    capacity_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        SpeechBackendPoolSettings(
            service=self.service,
            target_work=self.target_work,
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
            llm_warmup_enabled=self.llm_warmup_enabled,
            llm_warmup_timeout_s=self.llm_warmup_timeout_s,
            llm_warmup_model=self.llm_warmup_model,
        )
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be > 0")
        if self.max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if not 0 <= self.max_keepalive_connections <= self.max_connections:
            raise ValueError("max_keepalive_connections must be between 0 and max_connections")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.stt_audio_equivalent_s <= 0:
            raise ValueError("stt_audio_equivalent_s must be > 0")
        if self.lifecycle is not None and (not self.control_token or not self.endpoint_namespace):
            raise ValueError("autoscaling requires HF_CONTROL_TOKEN and HF_ENDPOINT_NAMESPACE")
        if self.lifecycle is not None and self.lifecycle.max_workers > len(self.backends):
            raise ValueError("max_workers cannot exceed the explicit worker inventory")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SpeechProxySettings:
        service = env_text("SPEECH_PROXY_SERVICE", environ=environ).lower()
        if service not in {"stt", "tts", "llm"}:
            raise ValueError("SPEECH_PROXY_SERVICE must be 'stt', 'tts', or 'llm'")
        defaults = {
            "stt": {"target_work": "96", "latency_target": "0.1"},
            "tts": {"target_work": "8", "latency_target": "0.5"},
            "llm": {"target_work": "64", "latency_target": "0.5"},
        }[service]
        backend_api_key = env_optional("SPEECH_BACKEND_API_KEY", environ=environ)
        if backend_api_key is None:
            backend_api_key = env_optional("HF_TOKEN", environ=environ)
        autoscale = env_bool("SPEECH_AUTOSCALE_ENABLED", False, environ=environ)
        raw_catalog = env_text("SPEECH_ROUTE_CATALOG", environ=environ)
        catalog = SpeechRouteCatalog.model_validate_json(raw_catalog) if raw_catalog else None
        if catalog and autoscale:
            raise ValueError("catalog mode requires lifecycle settings per pool, not SPEECH_AUTOSCALE_ENABLED")
        backends = () if catalog else parse_backends(env_text("SPEECH_BACKENDS", environ=environ), managed=autoscale)
        lifecycle = None
        if autoscale:
            lifecycle = WorkerLifecycleSettings(
                **{
                    name: type(default)(env_text(f"SPEECH_WORKER_{name.upper()}", str(default), environ=environ))
                    for name, default in asdict(WorkerLifecycleSettings(max_workers=len(backends))).items()
                }
            )
        settings = cls(
            capacity_api_key=env_optional("SPEECH_CAPACITY_API_KEY", environ=environ),
            service=service,  # type: ignore[arg-type]
            backends=backends,
            lifecycle=lifecycle,
            control_token=env_optional("HF_CONTROL_TOKEN", environ=environ),
            endpoint_namespace=env_optional("HF_ENDPOINT_NAMESPACE", environ=environ),
            backend_api_key=backend_api_key,
            target_work=_positive_float(
                "SPEECH_TARGET_WORK",
                env_text("SPEECH_TARGET_WORK", defaults["target_work"], environ=environ),
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
            max_connections=_positive_int(
                "SPEECH_MAX_CONNECTIONS",
                env_text("SPEECH_MAX_CONNECTIONS", "1024", environ=environ),
            ),
            max_keepalive_connections=int(env_text("SPEECH_MAX_KEEPALIVE_CONNECTIONS", "256", environ=environ)),
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
            llm_warmup_enabled=env_bool("LLM_WARMUP_ENABLED", True, environ=environ),
            llm_warmup_timeout_s=_positive_float(
                "LLM_WARMUP_TIMEOUT_S",
                env_text("LLM_WARMUP_TIMEOUT_S", "120", environ=environ),
            ),
            llm_warmup_model=env_text(
                "LLM_WARMUP_MODEL",
                "nvidia/Gemma-4-26B-A4B-NVFP4",
                environ=environ,
            ),
        )
        if catalog is None:
            return settings
        catalog.validate_service(service)

        def secret(name: str | None) -> str | None:
            if name is None:
                return None
            value = env_optional(name, environ=environ)
            if not value:
                raise ValueError(f"Missing required route secret: {name}")
            return value

        routes = []
        for route in catalog.pools:
            policy = route.policy.model_dump(exclude_unset=True)
            if service == "tts":
                policy.setdefault("tts_warmup_voice", route.capabilities.voices[0])
                if policy["tts_warmup_voice"] not in route.capabilities.voices:
                    raise ValueError("tts_warmup_voice must be supported by its route")
            if route.kind == "external":
                policy.update(tts_warmup_enabled=False, llm_warmup_enabled=False)
                policy.setdefault("health_path", "/v1/models")
            routes.append(
                replace(
                    settings,
                    route=route,
                    backends=tuple(SpeechBackendConfig(b.name, b.url.rstrip("/")) for b in route.backends),
                    backend_api_key=secret(route.credential_env),
                    access_api_key=secret(route.access_key_env),
                    lifecycle=WorkerLifecycleSettings(**route.lifecycle) if route.lifecycle is not None else None,
                    control_token=secret(route.control_token_env),
                    endpoint_namespace=route.namespace,
                    llm_warmup_model=route.upstream_model,
                    tts_warmup_model=route.upstream_model,
                    **policy,
                )
            )
        return replace(settings, catalog=catalog, routes=tuple(routes))

    def pool_settings(self) -> SpeechBackendPoolSettings:
        return SpeechBackendPoolSettings(
            service=self.service,
            target_work=self.target_work,
            max_work=self.max_work,
            session_work=self.route.session_workload.work_per_session
            if self.route and self.route.session_workload
            else None,
            session_rpm=self.route.session_workload.requests_per_minute
            if self.route and self.route.session_workload
            else 1,
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
            tts_warmup_format=self.route.capabilities.audio_formats[0]
            if self.route and self.service == "tts"
            else "pcm",
            llm_warmup_enabled=self.llm_warmup_enabled,
            llm_warmup_timeout_s=self.llm_warmup_timeout_s,
            llm_warmup_model=self.llm_warmup_model,
            external=self.route is not None and self.route.kind == "external",
            llm_warmup_api="responses"
            if self.route and "responses" in self.route.protocols and "chat_completions" not in self.route.protocols
            else "chat_completions",
            max_concurrency=self.route.capacity.max_concurrency if self.route and self.route.capacity else None,
            requests_per_minute=self.route.capacity.requests_per_minute if self.route and self.route.capacity else None,
        )


@dataclass
class SpeechProxyDependencies:
    pool: SpeechBackendPool | None = None
    client: httpx.AsyncClient | None = None
    metrics: SpeechProxyMetrics | None = None
    owns_client: bool = True
    lifecycle: SpeechWorkerLifecycle | None = None
    routes: dict[str, SpeechProxyDependencies] = field(default_factory=dict)
    startup_tasks: list[asyncio.Task] = field(default_factory=list)

    async def start(self) -> None:
        if self.routes:
            # Optional cold/unavailable models must not delay healthy routes' admission.
            self.startup_tasks = [
                asyncio.create_task(self._start_route(name, route)) for name, route in self.routes.items()
            ]
            return
        if self.lifecycle is not None:
            await self.lifecycle.start()
        await self.pool.start()

    async def _start_route(self, name: str, route: SpeechProxyDependencies) -> None:
        try:
            await route.start()
        except Exception as exc:
            logger.error("Pool startup failed pool=%s error_type=%s", name, type(exc).__name__)

    async def stop(self) -> None:
        if self.routes:
            for task in self.startup_tasks:
                await cancel_and_await(task)
            await asyncio.gather(*(route.stop() for route in self.routes.values()))
            return
        if self.lifecycle is not None:
            await self.lifecycle.stop()
        await self.pool.stop()
        if self.owns_client:
            await self.client.aclose()


def create_dependencies(settings: SpeechProxySettings) -> SpeechProxyDependencies:
    if settings.catalog:
        dependencies = SpeechProxyDependencies(
            routes={route.route.pool: create_dependencies(route) for route in settings.routes},
            metrics=SpeechProxyMetrics(settings.service),
        )
        for route in dependencies.routes.values():
            route.metrics.parent = dependencies.metrics
        return dependencies
    timeout = httpx.Timeout(settings.request_timeout_s, connect=min(settings.request_timeout_s, 10.0))
    limits = httpx.Limits(
        max_connections=settings.max_connections,
        max_keepalive_connections=settings.max_keepalive_connections,
    )
    client = httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True)
    pool = SpeechBackendPool(settings.backends, settings.pool_settings(), client=client)
    lifecycle = None
    if settings.lifecycle is not None:
        controller = HuggingFaceEndpointController(
            namespace=settings.endpoint_namespace,
            token=settings.control_token,
            park_strategy="pause",
            http_timeout_s=10,
        )
        lifecycle = SpeechWorkerLifecycle(pool, controller, settings.lifecycle)
    return SpeechProxyDependencies(
        pool=pool,
        client=client,
        metrics=SpeechProxyMetrics(settings.service, route=settings.route.labels() if settings.route else None),
        lifecycle=lifecycle,
    )


def _backend_headers(settings: SpeechProxySettings) -> dict[str, str]:
    headers = {"Accept": "*/*"}
    if settings.backend_api_key:
        headers["Authorization"] = f"Bearer {settings.backend_api_key}"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {name: value for name, value in response.headers.items() if name.lower() in RESPONSE_HEADERS}


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-speech-request-id", "").strip()
    supplied = "".join(character for character in supplied if character.isalnum() or character in "-_.")[:128]
    return supplied or uuid.uuid4().hex


def _traced_response_headers(response: httpx.Response, trace: SpeechRequestTrace) -> dict[str, str]:
    headers = _response_headers(response)
    if trace.metrics.route and "retry-after" in response.headers:
        headers["Retry-After"] = response.headers["retry-after"]
    if trace.sample is not None:
        headers.update(sample_headers(trace.sample))
    return headers


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


def _capacity_response(exc: SpeechPoolCapacityExceeded) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": str(exc), "type": "rate_limit_error", "code": "pool_capacity_exceeded"}},
        status_code=429,
        headers={"Retry-After": str(max(1, math.ceil(exc.retry_after)))},
    )


async def _proxy_stt(
    request: Request,
    settings: SpeechProxySettings,
    dependencies: SpeechProxyDependencies,
    trace: SpeechRequestTrace,
    prepared_form: tuple[list, float] | None = None,
) -> Response:
    try:
        multipart, duration_s = prepared_form if prepared_form is not None else await _stt_form(request)
        work = max(duration_s / settings.stt_audio_equivalent_s, 1.0)
        excluded: set[str] = set()
        last_error = "no backend attempt was made"
        for _ in range(settings.max_attempts):
            try:
                lease = await dependencies.pool.reserve(work, exclude=frozenset(excluded))
            except SpeechPoolCapacityExceeded as exc:
                await trace.record("error")
                return _capacity_response(exc)
            except NoSpeechBackendAvailable:
                break
            if not dependencies.pool.settings.external:
                excluded.add(lease.backend_name)
            started = time.monotonic()
            trace.start_upstream(lease.backend_name)
            try:
                response = await dependencies.client.post(
                    f"{lease.backend_url}/v1/audio/transcriptions",
                    headers={
                        **_backend_headers(settings),
                        "X-Speech-Request-Id": trace.request_id,
                    },
                    files=multipart,
                    timeout=settings.request_timeout_s,
                )
            except asyncio.CancelledError:
                trace.finish_upstream()
                await lease.release(success=False, cancelled=True)
                raise
            except httpx.HTTPError as exc:
                trace.finish_upstream()
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("STT backend %s transport failed: %s", lease.backend_name, last_error)
                await lease.release(success=False, retryable_failure=True, error=last_error)
                continue
            trace.finish_upstream(response.headers.get(SERVICE_LATENCY_HEADER))
            elapsed = time.monotonic() - started
            success = 200 <= response.status_code < 300
            retryable = _retryable_response(response.status_code, response.content)
            if dependencies.pool.settings.external and (
                response.status_code == 429 or (response.status_code == 503 and "retry-after" in response.headers)
            ):
                await dependencies.pool.rate_limited(response.headers.get("retry-after"))
                retryable = False
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
            await trace.record("success" if success else "error")
            return _upstream_error(
                response.status_code,
                response.content,
                _traced_response_headers(response, trace),
            )
        logger.error("No STT backend completed the request: %s", last_error)
        sample = await trace.record("error")
        response = _unavailable_response("stt")
        response.headers.update(sample_headers(sample))
        return response
    except asyncio.CancelledError:
        await trace.record("cancelled")
        raise
    except Exception:
        await trace.record("error")
        raise


async def _read_first_chunk(response: httpx.Response) -> tuple[bytes, Any]:
    # The downstream header allowlist omits Content-Encoding/Content-Length,
    # so forward decoded bytes just as the buffered STT/error paths do.
    iterator = response.aiter_bytes()
    while True:
        chunk = await anext(iterator)
        if chunk:
            return chunk, iterator


async def _proxy_stream(
    first_chunk: bytes,
    iterator: Any,
    response: httpx.Response,
    lease: SpeechBackendLease,
    routing_latency: float,
):
    completed = False
    try:
        yield first_chunk
        async for chunk in iterator:
            if chunk:
                yield chunk
        completed = True
    except asyncio.CancelledError:
        await lease.release(success=False, cancelled=True, latency=routing_latency)
        raise
    except Exception as exc:
        await lease.release(
            success=False,
            latency=routing_latency,
            retryable_failure=True,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        await response.aclose()
        if completed:
            await lease.release(success=True, latency=routing_latency)
        else:
            await lease.release(success=False, cancelled=True, latency=routing_latency)


# Retain the original test/helper name while sharing the lifecycle implementation
# with streamed LLM responses.
_tts_stream = _proxy_stream


async def _proxy_streaming_json(
    request: Request,
    path: str,
    settings: SpeechProxySettings,
    dependencies: SpeechProxyDependencies,
    trace: SpeechRequestTrace,
    body: bytes | None = None,
) -> Response:
    service = settings.service
    first_result = "audio" if service == "tts" else "token"
    try:
        body = await request.body() if body is None else body
        excluded: set[str] = set()
        last_error = "no backend attempt was made"
        for _ in range(settings.max_attempts):
            try:
                lease = await dependencies.pool.reserve(1.0, exclude=frozenset(excluded))
            except SpeechPoolCapacityExceeded as exc:
                await trace.record("error")
                return _capacity_response(exc)
            except NoSpeechBackendAvailable:
                break
            if not dependencies.pool.settings.external:
                excluded.add(lease.backend_name)
            started = time.monotonic()
            trace.start_upstream(lease.backend_name)
            response: httpx.Response | None = None
            stream_owns_response = False
            try:
                upstream_request = dependencies.client.build_request(
                    "POST",
                    f"{lease.backend_url}{path}",
                    headers={
                        **_backend_headers(settings),
                        "Content-Type": request.headers.get("content-type", "application/json"),
                        "X-Speech-Request-Id": trace.request_id,
                    },
                    content=body,
                    timeout=settings.request_timeout_s,
                )
                response = await dependencies.client.send(upstream_request, stream=True)
                if not 200 <= response.status_code < 300:
                    provider_limited = dependencies.pool.settings.external and (
                        response.status_code == 429
                        or (response.status_code == 503 and "retry-after" in response.headers)
                    )
                    if provider_limited:
                        await dependencies.pool.rate_limited(response.headers.get("retry-after"))
                    response_body = await response.aread()
                    trace.finish_upstream(response.headers.get(SERVICE_LATENCY_HEADER))
                    retryable = _retryable_response(response.status_code, response_body)
                    if provider_limited:
                        retryable = False
                    await lease.release(
                        success=False,
                        retryable_failure=retryable,
                        error=f"HTTP {response.status_code}",
                    )
                    if retryable:
                        last_error = f"backend returned HTTP {response.status_code}"
                        logger.warning("%s backend %s %s", service.upper(), lease.backend_name, last_error)
                        continue
                    await trace.record("error")
                    return _upstream_error(
                        response.status_code,
                        response_body,
                        _traced_response_headers(response, trace),
                    )
                first_chunk, iterator = await _read_first_chunk(response)
                trace.finish_upstream(response.headers.get(SERVICE_LATENCY_HEADER))
                first_result_latency = time.monotonic() - started
                await trace.record("success")
                downstream_response = StreamingResponse(
                    _proxy_stream(first_chunk, iterator, response, lease, first_result_latency),
                    status_code=response.status_code,
                    headers=_traced_response_headers(response, trace),
                )
                stream_owns_response = True
                return downstream_response
            except asyncio.CancelledError:
                trace.finish_upstream(response.headers.get(SERVICE_LATENCY_HEADER) if response is not None else None)
                await lease.release(
                    success=False,
                    cancelled=True,
                    latency=time.monotonic() - started,
                )
                raise
            except (httpx.HTTPError, httpx.StreamError, StopAsyncIteration) as exc:
                trace.finish_upstream(response.headers.get(SERVICE_LATENCY_HEADER) if response is not None else None)
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "%s backend %s failed before first %s: %s",
                    service.upper(),
                    lease.backend_name,
                    first_result,
                    last_error,
                )
                await lease.release(success=False, retryable_failure=True, error=last_error)
                continue
            except Exception as exc:
                trace.finish_upstream(response.headers.get(SERVICE_LATENCY_HEADER) if response is not None else None)
                await lease.release(success=False, error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                # Only the downstream stream may keep a response/lease beyond this
                # attempt. Failed reads, cancellation, and passthrough errors all
                # release above, before closing can itself fail.
                if response is not None and not stream_owns_response:
                    await response.aclose()
        logger.error("No %s backend started a response: %s", service.upper(), last_error)
        sample = await trace.record("error")
        response = _unavailable_response(service)
        response.headers.update(sample_headers(sample))
        return response
    except asyncio.CancelledError:
        await trace.record("cancelled")
        raise
    except Exception:
        await trace.record("error")
        raise


async def _dispatch_request(
    request: Request, path: str, settings: SpeechProxySettings, dependencies: SpeechProxyDependencies
) -> Response:
    trace = SpeechRequestTrace(dependencies.metrics, request_id=_request_id(request))
    prepared_form = None
    body = None
    if settings.catalog:
        try:
            if settings.service == "stt":
                prepared_form = await _stt_form(request)
                fields = [(name, value) for name, (filename, value, _) in prepared_form[0] if filename is None]
                if len([1 for name, _ in fields if name in {"model", "provider"}]) != len(
                    {name for name, _ in fields if name in {"model", "provider"}}
                ):
                    raise ValueError("model and provider must not be repeated")
                payload = dict(fields)
            else:
                payload = await request.json()
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
            model = payload.get("model")
            provider = request.headers.get("X-Speech-Provider", payload.get("provider"))
            if not isinstance(model, str) or not model or (provider is not None and not isinstance(provider, str)):
                raise ValueError("model and optional provider must be nonempty strings")
            if "provider" in payload and "X-Speech-Provider" in request.headers and payload["provider"] != provider:
                raise ValueError("provider field and header disagree")
            route = settings.catalog.resolve(model, provider)
            selected = next(item for item in settings.routes if item.route.pool == route.pool)
            if selected.access_api_key:
                credential = request.headers.get("X-Speech-Authorization", request.headers.get("Authorization", ""))
                if not hmac.compare_digest(credential.encode(), f"Bearer {selected.access_api_key}".encode()):
                    raise HTTPException(status_code=401, detail="route authorization required")
            route.validate_request(path, payload)
            required_context = request.headers.get("X-Speech-Required-Context-Tokens")
            if required_context is not None:
                required = int(required_context)
                if (
                    required < 1
                    or route.capabilities.context_window is None
                    or required > route.capabilities.context_window
                ):
                    raise ValueError("the selected route cannot satisfy the required context window")
            payload["model"] = route.upstream_model
            payload.pop("provider", None)
            if prepared_form is not None:
                multipart, duration = prepared_form
                prepared_form = (
                    [
                        (name, (None, route.upstream_model, None) if name == "model" else value)
                        for name, value in multipart
                        if name != "provider"
                    ],
                    duration,
                )
            else:
                body = json.dumps(payload).encode()
        except (ValueError, HTTPException) as exc:
            await trace.record("error")
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except asyncio.CancelledError:
            await trace.record("cancelled")
            raise
        except Exception:
            await trace.record("error")
            raise
        settings = selected
        dependencies = dependencies.routes[route.pool]
        trace.metrics = dependencies.metrics
    if settings.service == "stt":
        response = await _proxy_stt(request, settings, dependencies, trace, prepared_form)
    else:
        response = await _proxy_streaming_json(request, path, settings, dependencies, trace, body)
    if settings.route:
        response.headers.update({f"X-Speech-{key}": value for key, value in settings.route.labels().items()})
    return response


def create_app(
    settings: SpeechProxySettings,
    dependencies: SpeechProxyDependencies | None = None,
) -> FastAPI:
    dependencies = dependencies or create_dependencies(settings)
    if dependencies.metrics is None:
        dependencies.metrics = SpeechProxyMetrics(settings.service)

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
            "metrics": "/metrics",
        }

    @app.get("/health")
    async def health():
        if settings.catalog:
            pools = {}
            backends = []
            for selected in settings.routes:
                route = selected.route
                deps = dependencies.routes[route.pool]
                snapshots = await deps.pool.snapshots()
                ready = sum(snapshot.ready and not snapshot.draining for snapshot in snapshots)
                pools[route.pool] = {
                    **route.labels(),
                    "ready_backends": ready,
                    "status": "ok" if ready else "unhealthy",
                    "backends": [asdict(snapshot) for snapshot in snapshots],
                    "capacity": await deps.pool.capacity_snapshot(),
                    "lifecycle": await deps.lifecycle.snapshot() if deps.lifecycle else {"enabled": False},
                }
                backends.extend({**asdict(snapshot), **route.labels()} for snapshot in snapshots)
            ready = sum(pool["ready_backends"] for pool in pools.values())
            return JSONResponse(
                {
                    "status": "ok" if ready else "unhealthy",
                    "role": APP_ROLE,
                    "service": settings.service,
                    "ready_backends": ready,
                    "backends": backends,
                    "pools": pools,
                },
                status_code=200 if ready else 503,
            )
        snapshots = await dependencies.pool.snapshots()
        payload = {
            "status": "ok" if await dependencies.pool.healthy() else "unhealthy",
            "role": APP_ROLE,
            "service": settings.service,
            "ready_backends": sum(snapshot.ready and not snapshot.draining for snapshot in snapshots),
            "backends": [asdict(snapshot) for snapshot in snapshots],
            "lifecycle": await dependencies.lifecycle.snapshot() if dependencies.lifecycle else {"enabled": False},
        }
        if payload["ready_backends"] == 0:
            return JSONResponse(payload, status_code=503)
        return payload

    @app.get("/metrics")
    async def metrics(window_s: float = 300.0):
        try:
            payload = await dependencies.metrics.snapshot(window_s)
            if settings.catalog:
                payload["pools"] = {}
                for pool, deps in dependencies.routes.items():
                    snapshot = await deps.metrics.snapshot(window_s)
                    if deps.lifecycle:
                        snapshot["lifecycle"] = await deps.lifecycle.snapshot()
                    payload["pools"][pool] = snapshot
            if dependencies.lifecycle is not None:
                payload["lifecycle"] = await dependencies.lifecycle.snapshot()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return payload

    @app.post("/internal/capacity")
    async def session_capacity(request: Request):
        if not settings.catalog or not settings.capacity_api_key:
            raise HTTPException(status_code=404, detail="session capacity is not configured")
        credential = request.headers.get("X-Speech-Capacity-Authorization", request.headers.get("Authorization", ""))
        if not hmac.compare_digest(credential.encode(), f"Bearer {settings.capacity_api_key}".encode()):
            raise HTTPException(status_code=401, detail="capacity controller authorization required")
        try:
            update = SessionDemandUpdate.model_validate_json(await request.body())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid session demand") from exc
        selected = {s.route.pool: s for s in settings.routes}
        if any(pool not in selected or not selected[pool].route.session_workload for pool in update.session_counts):
            raise HTTPException(status_code=400, detail="unknown pool or missing workload profile")
        results = {}
        for pool, count in update.session_counts.items():
            route = selected[pool].route
            deps = dependencies.routes[pool]
            await deps.pool.set_session_demand(count, reserve_sessions=update.reserve_sessions)
            results[pool] = {
                **route.labels(),
                "request_model": route.aliases[0] if route.aliases else route.model,
                "aliases": route.aliases,
                "default_model_route": settings.catalog.defaults.get(route.model) == pool,
                "capabilities": route.capabilities.model_dump(mode="json"),
                "protocols": route.protocols,
                "voices": route.capabilities.voices,
                "profile": route.session_workload.profile,
                **await deps.pool.capacity_snapshot(),
            }
        return {"pools": results}

    if settings.service == "stt":

        @app.post("/v1/audio/transcriptions")
        async def transcriptions(request: Request):
            return await _dispatch_request(request, "/v1/audio/transcriptions", settings, dependencies)

    elif settings.service == "tts":

        @app.post("/v1/audio/speech")
        async def speech(request: Request):
            return await _dispatch_request(request, "/v1/audio/speech", settings, dependencies)

    else:

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await _dispatch_request(request, "/v1/chat/completions", settings, dependencies)

        @app.post("/v1/responses")
        async def responses(request: Request):
            return await _dispatch_request(request, "/v1/responses", settings, dependencies)

    return app

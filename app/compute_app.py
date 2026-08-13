import asyncio
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Mapping, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.app_utils import build_lifespan, env_bool, env_text, setup_logging
from app.requester_identity import bearer_token, client_address, is_validatable_hf_token
from app.session_router import SessionRouter
from app.session_tokens import llm_token_fingerprint, verify_session_token, websocket_host_matches
from app.ws_proxy import proxy_websocket

logger = setup_logging()
APP_ROLE = "compute"

INTERNAL_SLOT_WS_PATH = "/v1/realtime"
PUBLIC_WS_PATH = "/v1/realtime"
INTERNAL_USAGE_PATH = "/v1/usage"
INTERNAL_POOL_PATH = "/v1/pool"
LLM_PROXY_CONNECT_TIMEOUT_S = 10.0
LLM_PROXY_ACCOUNTING_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class ComputeSettings:
    internal_ws_host: str = "127.0.0.1"
    internal_ws_base_port: int = 9000
    s2s_repo_dir: str = "/opt/speech-to-speech"
    num_pipelines: str = "1"
    language: str = "en"
    chat_size: str = "30"
    stt: str = "parakeet-tdt"
    llm: str = "chat-completions"
    tts: str = "qwen3"
    enable_live_transcription: bool = True
    live_transcription_update_interval: str = ""
    enable_smart_turn: bool = True
    smart_turn_model_path: str = ""
    enable_llm_proxy: bool = False
    model_name: str = ""
    init_chat_prompt: str = "gpt-5.4"
    responses_api_base_url: str = ""
    responses_api_api_key: str = ""
    responses_api_reasoning_effort: str = ""
    responses_api_stream: bool = True
    extra_s2s_args: str = ""
    session_shared_secret: str = ""
    lb_callback_auth_token: str = ""
    lb_callback_retry_attempts: int = 5
    lb_callback_retry_backoff_s: float = 1.0
    llm_proxy_accounting_callback_url: str = ""
    llm_proxy_trust_proxy_headers: bool = True
    llm_proxy_requests_per_minute: int = 20

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ComputeSettings":
        responses_api_api_key = env_text("RESPONSES_API_API_KEY", environ=environ)
        if not responses_api_api_key:
            responses_api_api_key = env_text("HF_TOKEN", environ=environ)
        return cls(
            internal_ws_host=env_text("INTERNAL_WS_HOST", "127.0.0.1", environ=environ, strip=False),
            internal_ws_base_port=int(env_text("INTERNAL_WS_PORT", "9000", environ=environ, strip=False)),
            s2s_repo_dir=env_text("S2S_REPO_DIR", "/opt/speech-to-speech", environ=environ, strip=False),
            num_pipelines=env_text("NUM_PIPELINES", "1", environ=environ),
            language=env_text("LANGUAGE", "en", environ=environ),
            chat_size=env_text("CHAT_SIZE", "30", environ=environ),
            stt=env_text("STT", "parakeet-tdt", environ=environ),
            llm=env_text("LLM", "chat-completions", environ=environ),
            tts=env_text("TTS", "qwen3", environ=environ),
            enable_live_transcription=env_bool("ENABLE_LIVE_TRANSCRIPTION", True, environ=environ),
            live_transcription_update_interval=env_text("LIVE_TRANSCRIPTION_UPDATE_INTERVAL", environ=environ),
            enable_smart_turn=env_bool("ENABLE_SMART_TURN", True, environ=environ),
            smart_turn_model_path=env_text("SMART_TURN_MODEL_PATH", environ=environ),
            enable_llm_proxy=env_bool("ENABLE_LLM_PROXY", False, environ=environ),
            model_name=env_text("MODEL_NAME", environ=environ),
            init_chat_prompt=env_text("INIT_CHAT_PROMPT", "gpt-5.4", environ=environ),
            responses_api_base_url=env_text("RESPONSES_API_BASE_URL", environ=environ),
            responses_api_api_key=responses_api_api_key,
            responses_api_reasoning_effort=env_text("RESPONSES_API_REASONING_EFFORT", environ=environ),
            responses_api_stream=env_bool("RESPONSES_API_STREAM", True, environ=environ),
            extra_s2s_args=env_text("EXTRA_S2S_ARGS", environ=environ),
            session_shared_secret=env_text("SESSION_SHARED_SECRET", environ=environ),
            lb_callback_auth_token=env_text("LB_CALLBACK_AUTH_TOKEN", environ=environ),
            lb_callback_retry_attempts=max(
                int(env_text("LB_CALLBACK_RETRY_ATTEMPTS", "5", environ=environ)),
                1,
            ),
            lb_callback_retry_backoff_s=max(
                float(env_text("LB_CALLBACK_RETRY_BACKOFF_S", "1.0", environ=environ)),
                0.0,
            ),
            llm_proxy_accounting_callback_url=env_text(
                "LLM_PROXY_ACCOUNTING_CALLBACK_URL",
                environ=environ,
            ),
            llm_proxy_trust_proxy_headers=env_bool(
                "LLM_PROXY_TRUST_PROXY_HEADERS",
                True,
                environ=environ,
            ),
            llm_proxy_requests_per_minute=int(env_text("LLM_PROXY_REQUESTS_PER_MINUTE", "20", environ=environ)),
        )

    @property
    def internal_ws_url(self) -> str:
        return f"ws://{self.internal_ws_host}:{self.internal_ws_base_port}{INTERNAL_SLOT_WS_PATH}"

    @property
    def internal_usage_url(self) -> str:
        return f"http://{self.internal_ws_host}:{self.internal_ws_base_port}{INTERNAL_USAGE_PATH}"

    @property
    def internal_pool_url(self) -> str:
        return f"http://{self.internal_ws_host}:{self.internal_ws_base_port}{INTERNAL_POOL_PATH}"

    @property
    def internal_http_base_url(self) -> str:
        return f"http://{self.internal_ws_host}:{self.internal_ws_base_port}"


def _add_bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def _add_str_flag(cmd: list[str], value: str, flag: str) -> None:
    if value:
        cmd.extend([flag, value])


def build_s2s_command(
    host: str,
    port: int,
    settings: ComputeSettings,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--no-dev",
        "--no-sync",
        "--directory",
        settings.s2s_repo_dir,
        "speech-to-speech",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        "cuda",
        "--language",
        settings.language,
        "--chat_size",
        settings.chat_size,
        "--stt",
        settings.stt,
        "--llm_backend",
        settings.llm,
        "--tts",
        settings.tts,
    ]

    _add_str_flag(cmd, settings.num_pipelines, "--num_pipelines")
    _add_bool_flag(cmd, settings.enable_llm_proxy, "--enable_llm_proxy")
    _add_bool_flag(cmd, settings.enable_live_transcription, "--enable_live_transcription")
    _add_str_flag(cmd, settings.live_transcription_update_interval, "--live_transcription_update_interval")
    if settings.enable_smart_turn:
        _add_str_flag(cmd, settings.smart_turn_model_path, "--smart_turn_model_path")
    else:
        cmd.append("--no_smart_turn")
    _add_str_flag(cmd, settings.model_name, "--model_name")
    _add_str_flag(cmd, settings.init_chat_prompt, "--init_chat_prompt")

    if settings.llm in {"responses-api", "chat-completions"}:
        if settings.responses_api_base_url:
            _add_str_flag(cmd, settings.responses_api_base_url, "--responses_api_base_url")
        if settings.responses_api_api_key:
            _add_str_flag(cmd, settings.responses_api_api_key, "--responses_api_api_key")
        if settings.responses_api_reasoning_effort:
            _add_str_flag(cmd, settings.responses_api_reasoning_effort, "--responses_api_reasoning_effort")
        _add_bool_flag(cmd, settings.responses_api_stream, "--responses_api_stream")

    if settings.extra_s2s_args:
        cmd.extend(settings.extra_s2s_args.split())

    return cmd


async def wait_for_internal_server(
    host: str,
    port: int,
    process: Optional[subprocess.Popen],
    timeout_s: float = 900.0,
    *,
    http_get_json: Callable[[str], dict[str, object]] | None = None,
) -> None:
    get_json = http_get_json or _http_get_json
    http_url = f"http://{host}:{port}{INTERNAL_USAGE_PATH}"
    start = asyncio.get_running_loop().time()
    last_error = None

    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"speech-to-speech process exited early with code {process.returncode}")

        try:
            await asyncio.to_thread(get_json, http_url)
            logger.info("Internal speech-to-speech listener is ready at %s", http_url)
            return
        except Exception as exc:
            last_error = exc

        if asyncio.get_running_loop().time() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for internal realtime server at {http_url}. Last error: {last_error}"
            )

        await asyncio.sleep(2.0)


async def root(settings: ComputeSettings):
    return {
        "message": "s2s compute endpoint is up",
        "role": APP_ROLE,
        "health": "/health",
        "websocket": PUBLIC_WS_PATH,
        "internal_ws": settings.internal_ws_url,
        "internal_usage": settings.internal_usage_url,
        "config": {
            "stt": settings.stt,
            "llm": settings.llm,
            "tts": settings.tts,
            "language": settings.language,
        },
    }


async def health(settings: ComputeSettings, dependencies: "ComputeDependencies"):
    healthy, detail, snapshot = await dependencies.session_router.healthcheck()
    if not healthy:
        raise HTTPException(status_code=503, detail=detail or "compute router is not ready")

    return JSONResponse(
        {
            "status": "ok",
            "role": APP_ROLE,
            "internal_ws_base": settings.internal_ws_url,
            "internal_usage_url": settings.internal_usage_url,
            "public_websocket": PUBLIC_WS_PATH,
            "stt": settings.stt,
            "llm": settings.llm,
            "tts": settings.tts,
            "router": snapshot,
        }
    )


async def pool(settings: ComputeSettings, dependencies: "ComputeDependencies"):
    try:
        data = await asyncio.to_thread(
            dependencies.http_get_json,
            settings.internal_pool_url,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return JSONResponse(_redact_pool_payload(data))


def _redact_pool_payload(data: dict[str, object]) -> dict[str, object]:
    """Strip session ids from the pool payload before it leaves the replica.

    Session ids are private to their holders; nothing on this surface needs
    them, so they don't leave the replica. The load balancer's stuck-unit
    recovery reads only unit states and durations, which pass through
    untouched.
    """
    units = data.get("units")
    if isinstance(units, list):
        for unit in units:
            if isinstance(unit, dict):
                unit.pop("session_id", None)
    return data


class _ConnectedFingerprintRegistry:
    """HF token fingerprints with a currently connected realtime session.

    Membership is the LLM proxy access window: a fingerprint is added when
    its session's websocket connects and removed when it disconnects.
    Refcounted because one token may hold several concurrent sessions.
    Single event loop, so no locking.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def add(self, fingerprint: str) -> None:
        self._counts[fingerprint] = self._counts.get(fingerprint, 0) + 1

    def remove(self, fingerprint: str) -> None:
        count = self._counts.get(fingerprint, 0) - 1
        if count > 0:
            self._counts[fingerprint] = count
        else:
            self._counts.pop(fingerprint, None)

    def __contains__(self, fingerprint: str) -> bool:
        return fingerprint in self._counts


def _now() -> float:
    # Module-level so tests can monkeypatch the clock instead of sleeping.
    return time.monotonic()


_RATE_LIMIT_WINDOW_S = 60.0


class _FingerprintRateLimiter:
    """In-memory sliding window per HF token fingerprint, replica-local."""

    def __init__(self, limit_rpm: int, *, time_fn: Callable[[], float] = _now):
        self.limit_rpm = limit_rpm
        self._time_fn = time_fn
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, fingerprint: str) -> bool:
        if self.limit_rpm <= 0:
            return False
        now = self._time_fn()
        hits = self._hits[fingerprint]
        while hits and now - hits[0] >= _RATE_LIMIT_WINDOW_S:
            hits.popleft()
        if len(hits) >= self.limit_rpm:
            return False
        hits.append(now)
        # Sessions are short-lived relative to the process; drop fingerprints
        # whose newest hit fell out of the window so dead ones don't
        # accumulate forever (their deques are only ever popped above, on
        # their own requests).
        if len(self._hits) > 1024:
            self._hits = defaultdict(
                deque,
                {k: v for k, v in self._hits.items() if v and now - v[-1] < _RATE_LIMIT_WINDOW_S},
            )
        return True


def _llm_proxy_error(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": error_type}}, status_code=status_code)


def _llm_proxy_token(request: Request) -> Optional[str]:
    token = bearer_token(request.headers.get("x-reachy-mini-authorization"))
    if token is None:
        token = bearer_token(request.headers.get("authorization"))
    return token


def _llm_proxy_gate(
    token: Optional[str],
    settings: ComputeSettings,
    dependencies: "ComputeDependencies",
) -> tuple[Optional[JSONResponse], str]:
    """Return the local gate response and its canonical accounting reason.

    The api key must be the HF token the session was created with, checked by
    fingerprint against the sessions whose websocket is currently connected.
    Without a shared secret the replica cannot verify anything, so the paths
    fail closed. Checked once at request start: an answer already streaming
    when its session disconnects finishes undisturbed.
    """
    if not settings.enable_llm_proxy:
        return None, "proxy_disabled"

    fingerprint = None
    if settings.session_shared_secret and token is not None:
        candidate = llm_token_fingerprint(settings.session_shared_secret, token)
        if candidate in dependencies.connected_llm_fingerprints:
            fingerprint = candidate
    if fingerprint is None:
        denial = _llm_proxy_error(
            401,
            "Invalid API key: pass the HF token this session was created with, "
            "while the session's realtime websocket is connected.",
            "invalid_api_key",
        )
        return denial, "missing_token" if token is None else "no_active_session_match"
    if not dependencies.llm_rate_limiter.allow(fingerprint):
        denial = _llm_proxy_error(
            429,
            f"Rate limit exceeded: {dependencies.llm_rate_limiter.limit_rpm} requests per minute "
            "per user. Back off and retry.",
            "rate_limit_exceeded",
        )
        return denial, "rate_limited"
    return None, "accepted"


async def _report_llm_proxy_request(
    request: Request,
    settings: ComputeSettings,
    dependencies: "ComputeDependencies",
    *,
    reason: str,
    token: Optional[str],
) -> None:
    if not settings.llm_proxy_accounting_callback_url:
        logger.warning("LLM proxy accounting skipped: no fleet callback URL is configured")
        return

    payload: dict[str, object] = {"reason": reason}
    if token is not None and is_validatable_hf_token(token):
        payload["token"] = token
    address = client_address(request, trust_proxy_headers=settings.llm_proxy_trust_proxy_headers)
    if address is not None:
        payload["client_ip"] = address
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                _post_json,
                settings.llm_proxy_accounting_callback_url,
                payload,
                callback_auth_token=settings.lb_callback_auth_token,
                timeout_s=LLM_PROXY_ACCOUNTING_TIMEOUT_S,
            ),
            timeout=LLM_PROXY_ACCOUNTING_TIMEOUT_S,
        )
    except Exception as exc:
        logger.warning("Failed to record LLM proxy request reason=%s: %s", reason, exc)


async def llm_proxy_chat_completions(
    request: Request,
    settings: ComputeSettings,
    dependencies: "ComputeDependencies",
) -> Response:
    return await _proxy_llm_request(
        request,
        "/v1/chat/completions",
        settings,
        dependencies,
    )


async def llm_proxy_responses(
    request: Request,
    settings: ComputeSettings,
    dependencies: "ComputeDependencies",
) -> Response:
    return await _proxy_llm_request(
        request,
        "/v1/responses",
        settings,
        dependencies,
    )


async def _proxy_llm_request(
    request: Request,
    path: str,
    settings: ComputeSettings,
    dependencies: "ComputeDependencies",
) -> Response:
    """Pass an authorized LLM proxy request through to the internal pipeline.

    The replica owns access control (see ``_llm_proxy_denial``); the pipeline
    behind it owns the OpenAI contract itself (501 reasons, upstream errors,
    the upstream provider key). An authorized request is forwarded with its
    method and body unchanged — the api key is a user's HF token, so it is
    dropped here and never travels further — and the answer streams back
    frame by frame, whatever its status. The replica only synthesizes 404
    (feature disabled), 401 and 429 (denials), and 502 (internal pipeline
    unreachable).
    """
    token = _llm_proxy_token(request)
    denial, reason = _llm_proxy_gate(token, settings, dependencies)
    await _report_llm_proxy_request(
        request,
        settings,
        dependencies,
        reason=reason,
        token=token,
    )
    if reason == "proxy_disabled":
        raise HTTPException(status_code=404)
    if denial is not None:
        return denial

    headers = {}
    content_type = request.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    # Connects are loopback so a short timeout is safe; reads get none at
    # all, because a proxied generation can legitimately take minutes.
    timeout = httpx.Timeout(None, connect=LLM_PROXY_CONNECT_TIMEOUT_S)
    client = httpx.AsyncClient(timeout=timeout)
    try:
        upstream_request = client.build_request(
            "POST",
            settings.internal_http_base_url + path,
            content=await request.body(),
            headers=headers,
        )
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        logger.warning("LLM proxy: internal pipeline unreachable: %s", exc)
        return JSONResponse(
            {
                "error": {
                    "message": "The internal pipeline is unreachable.",
                    "type": "upstream_unreachable",
                }
            },
            status_code=502,
        )
    except Exception:
        await client.aclose()
        raise

    async def _cleanup() -> None:
        await upstream.aclose()
        await client.aclose()

    async def _stream_and_cleanup():
        # The generator owns the cleanup: Starlette only runs background
        # tasks after a successful send, so a pipeline crash mid-stream
        # would otherwise leak the httpx client and its connection. The
        # finally runs on normal exhaustion, on upstream errors, and on
        # client-disconnect cancellation alike; the BackgroundTask below is
        # a harmless second aclose on the successful path.
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await _cleanup()

    # The content type rides along as a raw header: a media_type would get a
    # charset appended by Starlette, and the answer must stay verbatim.
    response_headers = {}
    upstream_content_type = upstream.headers.get("content-type")
    if upstream_content_type:
        response_headers["content-type"] = upstream_content_type

    return StreamingResponse(
        _stream_and_cleanup(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(_cleanup),
    )


async def websocket_proxy(
    client_ws: WebSocket,
    settings: ComputeSettings,
    dependencies: "ComputeDependencies",
):
    session_payload = _get_session_payload(client_ws, settings)

    if settings.session_shared_secret and session_payload is None:
        await client_ws.close(code=1008, reason="Missing or invalid session token")
        return

    llm_fingerprint: Optional[str] = None
    if session_payload is not None:
        claim = session_payload.get("llmf")
        if isinstance(claim, str) and claim:
            llm_fingerprint = claim
    llm_fingerprint_registered = False

    async def _notify_connected() -> None:
        # Runs only after a pipeline slot is actually secured. Notifying the
        # LB before acquiring capacity meant a rejected connection produced a
        # connected/disconnected pair milliseconds apart, which the dashboard
        # counted as a completed conversation while live users stayed at zero.
        if session_payload is None:
            return
        await dependencies.notify_lb_session_event(
            session_payload["callback_url"],
            session_payload["session_token"],
            "connected",
        )
        # The LLM proxy access window opens with the connection; it closes in
        # the finally below, alongside the disconnected notification.
        nonlocal llm_fingerprint_registered
        if llm_fingerprint is not None:
            dependencies.connected_llm_fingerprints.add(llm_fingerprint)
            llm_fingerprint_registered = True

    try:
        await dependencies.proxy_websocket(
            client_ws,
            acquire_lease=lambda _: dependencies.session_router.acquire(),
            release_lease=dependencies.session_router.release,
            describe_lease=lambda slot: f"slot {slot.slot_id} at {slot.ws_url}",
            no_capacity_reason="No pipeline capacity available",
            no_capacity_log="Failed to allocate speech-to-speech slot",
            on_lease_acquired=_notify_connected,
        )
    except Exception as exc:
        logger.warning("Rejected websocket session: %s", exc)
        try:
            await client_ws.close(code=1013, reason="Failed to establish reserved session")
        except Exception:
            pass
    finally:
        if llm_fingerprint_registered and llm_fingerprint is not None:
            dependencies.connected_llm_fingerprints.remove(llm_fingerprint)
        if session_payload is not None:
            # Always tell the LB the session is over. For a normal session this
            # completes the conversation; for a capacity rejection it releases
            # the pending lease immediately instead of holding the slot until
            # the pending reaper fires. The LB treats a disconnect for an
            # unknown or never-connected session as a no-op release.
            try:
                await dependencies.notify_lb_session_event(
                    session_payload["callback_url"],
                    session_payload["session_token"],
                    "disconnected",
                    attempts=settings.lb_callback_retry_attempts,
                )
            except Exception:
                logger.exception("Failed to notify LB that session ended")


def _get_session_payload(
    client_ws: WebSocket,
    settings: ComputeSettings,
) -> Optional[dict[str, object]]:
    if not settings.session_shared_secret:
        return None

    session_token = _extract_session_token(client_ws)
    if not session_token:
        return None

    try:
        payload = verify_session_token(session_token, settings.session_shared_secret)
    except ValueError:
        logger.warning("Rejected websocket with invalid session token")
        return None

    request_host = client_ws.headers.get("x-forwarded-host") or client_ws.headers.get("host")
    if not websocket_host_matches(str(payload["ws_url"]), request_host):
        logger.warning("Rejected websocket for mismatched compute endpoint host %s", request_host)
        return None

    payload["session_token"] = session_token
    return payload


def _extract_session_token(client_ws: WebSocket) -> Optional[str]:
    query_token = client_ws.query_params.get("session_token", "").strip()
    if query_token:
        return query_token

    auth_header = client_ws.headers.get("authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()

    return None


async def _notify_lb_session_event(
    callback_url: str,
    session_token: str,
    event: str,
    *,
    post_json: Callable[[str, dict[str, str]], None],
    default_backoff_s: float,
    attempts: int = 1,
    backoff_s: Optional[float] = None,
) -> None:
    """Post a session lifecycle event to the LB callback URL.

    Retries with exponential backoff when attempts > 1. The LB endpoint is
    idempotent for our purposes: a disconnect for an unknown or already
    released session returns 200 already_released, so repeating a request
    whose response was lost is safe.
    """
    payload = {
        "session_token": session_token,
        "event": event,
    }
    if backoff_s is None:
        backoff_s = default_backoff_s
    attempts = max(attempts, 1)
    delay = backoff_s
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.to_thread(post_json, callback_url, payload)
            return
        except Exception as exc:
            if attempt >= attempts:
                raise
            logger.warning(
                "LB '%s' notification failed (attempt %d/%d), retrying in %.1fs: %s",
                event,
                attempt,
                attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
            delay *= 3


def _http_get_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status_code = getattr(response, "status", 200)
            if status_code >= 400:
                raise RuntimeError(f"HTTP GET failed with status {status_code}")
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP GET failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HTTP GET failed: {exc.reason}") from exc

    try:
        return json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("HTTP GET returned invalid JSON") from exc


def _post_json(
    url: str,
    payload: dict[str, object],
    *,
    callback_auth_token: str,
    timeout_s: float = 10.0,
) -> None:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if callback_auth_token:
        headers["Authorization"] = f"Bearer {callback_auth_token}"

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status_code = getattr(response, "status", 200)
            if status_code >= 400:
                raise RuntimeError(f"LB callback failed with HTTP {status_code}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LB callback failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LB callback failed: {exc.reason}") from exc


@dataclass(frozen=True)
class ComputeDependencies:
    session_router: SessionRouter
    connected_llm_fingerprints: _ConnectedFingerprintRegistry
    llm_rate_limiter: _FingerprintRateLimiter
    http_get_json: Callable[[str], dict[str, object]]
    notify_lb_session_event: Callable[..., Awaitable[None]]
    proxy_websocket: Callable[..., Awaitable[None]]


@dataclass(frozen=True)
class ComputeRuntime:
    settings: ComputeSettings
    dependencies: ComputeDependencies


def build_compute_dependencies(settings: ComputeSettings) -> ComputeDependencies:
    async def wait_for_ready(
        host: str,
        port: int,
        process: Optional[subprocess.Popen],
        timeout_s: float,
    ) -> None:
        await wait_for_internal_server(
            host,
            port,
            process,
            timeout_s,
            http_get_json=_http_get_json,
        )

    def post_json(url: str, payload: dict[str, object]) -> None:
        _post_json(
            url,
            payload,
            callback_auth_token=settings.lb_callback_auth_token,
        )

    async def notify_lb_session_event(
        callback_url: str,
        session_token: str,
        event: str,
        *,
        attempts: int = 1,
        backoff_s: Optional[float] = None,
    ) -> None:
        await _notify_lb_session_event(
            callback_url,
            session_token,
            event,
            post_json=post_json,
            default_backoff_s=settings.lb_callback_retry_backoff_s,
            attempts=attempts,
            backoff_s=backoff_s,
        )

    router = SessionRouter(
        host=settings.internal_ws_host,
        base_port=settings.internal_ws_base_port,
        ws_path=INTERNAL_SLOT_WS_PATH,
        repo_dir=settings.s2s_repo_dir,
        build_command=lambda host, port: build_s2s_command(host, port, settings),
        wait_for_ready=wait_for_ready,
        max_sessions=int(settings.num_pipelines),
    )
    return ComputeDependencies(
        session_router=router,
        connected_llm_fingerprints=_ConnectedFingerprintRegistry(),
        llm_rate_limiter=_FingerprintRateLimiter(settings.llm_proxy_requests_per_minute),
        http_get_json=_http_get_json,
        notify_lb_session_event=notify_lb_session_event,
        proxy_websocket=proxy_websocket,
    )


def create_app(
    settings: ComputeSettings,
    dependencies: ComputeDependencies | None = None,
) -> FastAPI:
    """Create a compute application from explicit configuration."""
    resolved_dependencies = dependencies or build_compute_dependencies(settings)
    if not settings.session_shared_secret:
        logger.warning(
            "SESSION_SHARED_SECRET is unset; the LLM proxy paths fail closed and answer 401 for every request"
        )
    runtime = ComputeRuntime(settings, resolved_dependencies)
    application = FastAPI(lifespan=build_lifespan(resolved_dependencies.session_router))
    application.state.runtime = runtime
    application.state.settings = settings
    application.state.dependencies = resolved_dependencies

    async def root_route():
        return await root(settings)

    async def health_route():
        return await health(settings, resolved_dependencies)

    async def pool_route():
        return await pool(settings, resolved_dependencies)

    async def chat_completions_route(request: Request):
        return await llm_proxy_chat_completions(request, settings, resolved_dependencies)

    async def responses_route(request: Request):
        return await llm_proxy_responses(request, settings, resolved_dependencies)

    async def websocket_route(client_ws: WebSocket):
        await websocket_proxy(client_ws, settings, resolved_dependencies)

    application.add_api_route("/", root_route, methods=["GET"], name="root")
    application.add_api_route("/health", health_route, methods=["GET"], name="health")
    application.add_api_route("/v1/pool", pool_route, methods=["GET"], name="pool")
    application.add_api_route(
        "/v1/chat/completions",
        chat_completions_route,
        methods=["POST"],
        name="llm_proxy_chat_completions",
    )
    application.add_api_route(
        "/v1/responses",
        responses_route,
        methods=["POST"],
        name="llm_proxy_responses",
    )
    application.add_api_websocket_route(
        "/v1/realtime",
        websocket_route,
        name="websocket_proxy",
    )
    return application

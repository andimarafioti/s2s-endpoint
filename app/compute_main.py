import asyncio
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from app.app_utils import build_lifespan, setup_logging
from app.requester_identity import bearer_token
from app.session_router import SessionRouter
from app.session_tokens import llm_token_fingerprint, verify_session_token, websocket_host_matches
from app.ws_proxy import proxy_websocket

logger = setup_logging()
APP_ROLE = "compute"

INTERNAL_WS_HOST = os.getenv("INTERNAL_WS_HOST", "127.0.0.1")
INTERNAL_WS_BASE_PORT = int(os.getenv("INTERNAL_WS_PORT", "9000"))

S2S_REPO_DIR = os.getenv("S2S_REPO_DIR", "/opt/speech-to-speech")
NUM_PIPELINES = os.getenv("NUM_PIPELINES", "1").strip()

# Core pipeline selection
LANGUAGE = os.getenv("LANGUAGE", "en").strip()
CHAT_SIZE = os.getenv("CHAT_SIZE", "30").strip()

STT = os.getenv("STT", "parakeet-tdt").strip()
LLM = os.getenv("LLM", "chat-completions").strip()
TTS = os.getenv("TTS", "qwen3").strip()

# General module flags
ENABLE_LIVE_TRANSCRIPTION = os.getenv("ENABLE_LIVE_TRANSCRIPTION", "1").strip().lower() in {"1", "true", "yes"}
LIVE_TRANSCRIPTION_UPDATE_INTERVAL = os.getenv("LIVE_TRANSCRIPTION_UPDATE_INTERVAL", "").strip()
ENABLE_SMART_TURN = os.getenv("ENABLE_SMART_TURN", "1").strip().lower() in {"1", "true", "yes"}
SMART_TURN_DEVICE = os.getenv("SMART_TURN_DEVICE", "cpu").strip()
SMART_TURN_MODEL_PATH = os.getenv("SMART_TURN_MODEL_PATH", "").strip()
SMART_TURN_THRESHOLD = os.getenv("SMART_TURN_THRESHOLD", "0.5").strip()
SMART_TURN_MAX_WAIT_MS = os.getenv("SMART_TURN_MAX_WAIT_MS", "2000").strip()
SMART_TURN_CPU_COUNT = os.getenv("SMART_TURN_CPU_COUNT", "").strip()
# Master switch for the LLM proxy feature: passes --enable_llm_proxy to the
# internal speech-to-speech server (which defaults the routes off) and opens
# the replica's /v1/chat/completions and /v1/responses proxy paths. When off,
# those paths answer 404, indistinguishable from a build without the feature.
ENABLE_LLM_PROXY = os.getenv("ENABLE_LLM_PROXY", "0").strip().lower() in {"1", "true", "yes"}

# Responses API / HF router
MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
INIT_CHAT_PROMPT = os.getenv("INIT_CHAT_PROMPT", "gpt-5.4").strip()
RESPONSES_API_BASE_URL = os.getenv("RESPONSES_API_BASE_URL", "").strip()
RESPONSES_API_API_KEY = os.getenv("RESPONSES_API_API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()
RESPONSES_API_REASONING_EFFORT = os.getenv("RESPONSES_API_REASONING_EFFORT", "").strip()
RESPONSES_API_STREAM = os.getenv("RESPONSES_API_STREAM", "1").strip().lower() in {"1", "true", "yes"}

# Optional generic extras for power users
EXTRA_S2S_ARGS = os.getenv("EXTRA_S2S_ARGS", "").strip()

SESSION_SHARED_SECRET = os.getenv("SESSION_SHARED_SECRET", "").strip()
LB_CALLBACK_AUTH_TOKEN = os.getenv("LB_CALLBACK_AUTH_TOKEN", "").strip()
# Disconnect notifications are retried so a transient LB timeout or 503 does
# not leave a session counted as connected forever (connected sessions are
# never reaped by the pending-session reaper). Defaults give backoff waits of
# 1s/3s/9s/27s, roughly 40s of coverage: enough to ride out an LB redeploy or
# brief network partition, which is precisely the case sync gating cannot
# cover (the LB stayed up, so its in-memory session survives).
LB_CALLBACK_RETRY_ATTEMPTS = max(int(os.getenv("LB_CALLBACK_RETRY_ATTEMPTS", "5")), 1)
LB_CALLBACK_RETRY_BACKOFF_S = max(float(os.getenv("LB_CALLBACK_RETRY_BACKOFF_S", "1.0")), 0.0)

INTERNAL_SLOT_WS_PATH = "/v1/realtime"
PUBLIC_WS_PATH = "/v1/realtime"
INTERNAL_WS_URL = f"ws://{INTERNAL_WS_HOST}:{INTERNAL_WS_BASE_PORT}{INTERNAL_SLOT_WS_PATH}"
INTERNAL_USAGE_PATH = "/v1/usage"
INTERNAL_USAGE_URL = f"http://{INTERNAL_WS_HOST}:{INTERNAL_WS_BASE_PORT}{INTERNAL_USAGE_PATH}"
INTERNAL_POOL_PATH = "/v1/pool"
INTERNAL_POOL_URL = f"http://{INTERNAL_WS_HOST}:{INTERNAL_WS_BASE_PORT}{INTERNAL_POOL_PATH}"
INTERNAL_HTTP_BASE_URL = f"http://{INTERNAL_WS_HOST}:{INTERNAL_WS_BASE_PORT}"
LLM_PROXY_CONNECT_TIMEOUT_S = 10.0
# Sliding-window ceiling per HF token fingerprint. Zero or negative closes
# the LLM proxy paths entirely (every request answers 429).
LLM_PROXY_REQUESTS_PER_MINUTE = int(os.getenv("LLM_PROXY_REQUESTS_PER_MINUTE", "20"))

if not SESSION_SHARED_SECRET:
    logger.warning("SESSION_SHARED_SECRET is unset; the LLM proxy paths fail closed and answer 401 for every request")


def _add_bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def _add_str_flag(cmd: list[str], value: str, flag: str) -> None:
    if value:
        cmd.extend([flag, value])


def build_s2s_command(host: str, port: int) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--no-dev",
        "--no-sync",
        "--directory",
        S2S_REPO_DIR,
        "speech-to-speech",
        "--mode",
        "realtime",
        "--ws_host",
        host,
        "--ws_port",
        str(port),
        "--device",
        "cuda",
        "--language",
        LANGUAGE,
        "--chat_size",
        CHAT_SIZE,
        "--stt",
        STT,
        "--llm_backend",
        LLM,
        "--tts",
        TTS,
    ]

    _add_str_flag(cmd, NUM_PIPELINES, "--num_pipelines")
    _add_bool_flag(cmd, ENABLE_LLM_PROXY, "--enable_llm_proxy")
    _add_bool_flag(cmd, ENABLE_LIVE_TRANSCRIPTION, "--enable_live_transcription")
    _add_str_flag(cmd, LIVE_TRANSCRIPTION_UPDATE_INTERVAL, "--live_transcription_update_interval")
    if ENABLE_SMART_TURN:
        cmd.append("--smart_turn")
        _add_str_flag(cmd, SMART_TURN_DEVICE, "--smart_turn_device")
        _add_str_flag(cmd, SMART_TURN_MODEL_PATH, "--smart_turn_model_path")
        _add_str_flag(cmd, SMART_TURN_THRESHOLD, "--smart_turn_threshold")
        _add_str_flag(cmd, SMART_TURN_MAX_WAIT_MS, "--smart_turn_max_wait_ms")
        _add_str_flag(cmd, SMART_TURN_CPU_COUNT, "--smart_turn_cpu_count")
    else:
        cmd.append("--no_smart_turn")
    _add_str_flag(cmd, MODEL_NAME, "--model_name")
    _add_str_flag(cmd, INIT_CHAT_PROMPT, "--init_chat_prompt")

    if LLM in {"responses-api", "chat-completions"}:
        if RESPONSES_API_BASE_URL:
            _add_str_flag(cmd, RESPONSES_API_BASE_URL, "--responses_api_base_url")
        if RESPONSES_API_API_KEY:
            _add_str_flag(cmd, RESPONSES_API_API_KEY, "--responses_api_api_key")
        if RESPONSES_API_REASONING_EFFORT:
            _add_str_flag(cmd, RESPONSES_API_REASONING_EFFORT, "--responses_api_reasoning_effort")
        _add_bool_flag(cmd, RESPONSES_API_STREAM, "--responses_api_stream")

    if EXTRA_S2S_ARGS:
        cmd.extend(EXTRA_S2S_ARGS.split())

    return cmd


def _smart_turn_config() -> dict[str, object]:
    return {
        "enabled": ENABLE_SMART_TURN,
        "device": SMART_TURN_DEVICE,
        "model_path": SMART_TURN_MODEL_PATH or None,
        "threshold": SMART_TURN_THRESHOLD,
        "max_wait_ms": SMART_TURN_MAX_WAIT_MS,
        "cpu_count": SMART_TURN_CPU_COUNT or None,
    }


async def wait_for_internal_server(
    host: str,
    port: int,
    process: Optional[subprocess.Popen],
    timeout_s: float = 900.0,
) -> None:
    http_url = f"http://{host}:{port}{INTERNAL_USAGE_PATH}"
    start = asyncio.get_running_loop().time()
    last_error = None

    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"speech-to-speech process exited early with code {process.returncode}")

        try:
            await asyncio.to_thread(_http_get_json, http_url)
            logger.info("Internal speech-to-speech listener is ready at %s", http_url)
            return
        except Exception as exc:
            last_error = exc

        if asyncio.get_running_loop().time() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for internal realtime server at {http_url}. Last error: {last_error}"
            )

        await asyncio.sleep(2.0)


session_router = SessionRouter(
    host=INTERNAL_WS_HOST,
    base_port=INTERNAL_WS_BASE_PORT,
    ws_path=INTERNAL_SLOT_WS_PATH,
    repo_dir=S2S_REPO_DIR,
    build_command=build_s2s_command,
    wait_for_ready=wait_for_internal_server,
    max_sessions=int(NUM_PIPELINES),
)

app = FastAPI(lifespan=build_lifespan(session_router))


@app.get("/")
async def root():
    return {
        "message": "s2s compute endpoint is up",
        "role": APP_ROLE,
        "health": "/health",
        "websocket": PUBLIC_WS_PATH,
        "internal_ws": INTERNAL_WS_URL,
        "internal_usage": INTERNAL_USAGE_URL,
        "config": {
            "stt": STT,
            "llm": LLM,
            "tts": TTS,
            "language": LANGUAGE,
            "smart_turn": _smart_turn_config(),
        },
    }


@app.get("/health")
async def health():
    healthy, detail, snapshot = await session_router.healthcheck()
    if not healthy:
        raise HTTPException(status_code=503, detail=detail or "compute router is not ready")

    return JSONResponse(
        {
            "status": "ok",
            "role": APP_ROLE,
            "internal_ws_base": INTERNAL_WS_URL,
            "internal_usage_url": INTERNAL_USAGE_URL,
            "public_websocket": PUBLIC_WS_PATH,
            "stt": STT,
            "llm": LLM,
            "tts": TTS,
            "smart_turn": _smart_turn_config(),
            "router": snapshot,
        }
    )


@app.get("/v1/pool")
async def pool():
    try:
        data = await asyncio.to_thread(_http_get_json, INTERNAL_POOL_URL)
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

    def __init__(self, limit_rpm: int):
        self.limit_rpm = limit_rpm
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, fingerprint: str) -> bool:
        if self.limit_rpm <= 0:
            return False
        now = _now()
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


_connected_llm_fingerprints = _ConnectedFingerprintRegistry()
_llm_rate_limiter = _FingerprintRateLimiter(LLM_PROXY_REQUESTS_PER_MINUTE)


def _llm_proxy_error(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": error_type}}, status_code=status_code)


def _llm_proxy_denial(request: Request) -> Optional[JSONResponse]:
    """Access check for the LLM proxy paths; None means forward the request.

    The api key must be the HF token the session was created with, checked by
    fingerprint against the sessions whose websocket is currently connected.
    Without a shared secret the replica cannot verify anything, so the paths
    fail closed. Checked once at request start: an answer already streaming
    when its session disconnects finishes undisturbed.

    The key is read from ``x-reachy-mini-authorization`` first, then from
    ``Authorization`` — the same precedence the load balancer uses at session
    creation, and for the same reason: the HF Inference Endpoints ingress
    consumes the standard Authorization header before it reaches the app, so
    SDK clients on that infrastructure carry the token in the custom header
    (``default_headers``) alongside their normal ``api_key``.
    """
    if SESSION_SHARED_SECRET:
        token = bearer_token(request.headers.get("x-reachy-mini-authorization"))
        if token is None:
            token = bearer_token(request.headers.get("authorization"))
        if token is not None:
            fingerprint = llm_token_fingerprint(SESSION_SHARED_SECRET, token)
            if fingerprint in _connected_llm_fingerprints:
                if _llm_rate_limiter.allow(fingerprint):
                    return None
                return _llm_proxy_error(
                    429,
                    f"Rate limit exceeded: {_llm_rate_limiter.limit_rpm} requests per minute "
                    "per user. Back off and retry.",
                    "rate_limit_exceeded",
                )
    return _llm_proxy_error(
        401,
        "Invalid API key: pass the HF token this session was created with, "
        "while the session's realtime websocket is connected.",
        "invalid_api_key",
    )


@app.post("/v1/chat/completions")
async def llm_proxy_chat_completions(request: Request) -> Response:
    return await _proxy_llm_request(request, "/v1/chat/completions")


@app.post("/v1/responses")
async def llm_proxy_responses(request: Request) -> Response:
    return await _proxy_llm_request(request, "/v1/responses")


async def _proxy_llm_request(request: Request, path: str) -> Response:
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
    if not ENABLE_LLM_PROXY:
        # Checked before auth: a disabled replica reveals nothing, answering
        # exactly like an app where these routes were never registered.
        raise HTTPException(status_code=404)

    denial = _llm_proxy_denial(request)
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
            INTERNAL_HTTP_BASE_URL + path,
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


@app.websocket("/v1/realtime")
async def websocket_proxy(client_ws: WebSocket):
    session_payload = _get_session_payload(client_ws)

    if SESSION_SHARED_SECRET and session_payload is None:
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
        await _notify_lb_session_event(
            session_payload["callback_url"],
            session_payload["session_token"],
            "connected",
        )
        # The LLM proxy access window opens with the connection; it closes in
        # the finally below, alongside the disconnected notification.
        nonlocal llm_fingerprint_registered
        if llm_fingerprint is not None:
            _connected_llm_fingerprints.add(llm_fingerprint)
            llm_fingerprint_registered = True

    try:
        await proxy_websocket(
            client_ws,
            acquire_lease=lambda _: session_router.acquire(),
            release_lease=session_router.release,
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
            _connected_llm_fingerprints.remove(llm_fingerprint)
        if session_payload is not None:
            # Always tell the LB the session is over. For a normal session this
            # completes the conversation; for a capacity rejection it releases
            # the pending lease immediately instead of holding the slot until
            # the pending reaper fires. The LB treats a disconnect for an
            # unknown or never-connected session as a no-op release.
            try:
                await _notify_lb_session_event(
                    session_payload["callback_url"],
                    session_payload["session_token"],
                    "disconnected",
                    attempts=LB_CALLBACK_RETRY_ATTEMPTS,
                )
            except Exception:
                logger.exception("Failed to notify LB that session ended")


def _get_session_payload(client_ws: WebSocket) -> Optional[dict[str, str]]:
    if not SESSION_SHARED_SECRET:
        return None

    session_token = _extract_session_token(client_ws)
    if not session_token:
        return None

    try:
        payload = verify_session_token(session_token, SESSION_SHARED_SECRET)
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
        backoff_s = LB_CALLBACK_RETRY_BACKOFF_S
    attempts = max(attempts, 1)
    delay = backoff_s
    for attempt in range(1, attempts + 1):
        try:
            await asyncio.to_thread(_post_json, callback_url, payload)
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


def _post_json(url: str, payload: dict[str, str]) -> None:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if LB_CALLBACK_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {LB_CALLBACK_AUTH_TOKEN}"

    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status_code = getattr(response, "status", 200)
            if status_code >= 400:
                raise RuntimeError(f"LB callback failed with HTTP {status_code}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LB callback failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LB callback failed: {exc.reason}") from exc

import asyncio
import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse

from app.app_utils import build_lifespan, setup_logging
from app.session_router import SessionRouter
from app.session_tokens import verify_session_token, websocket_host_matches
from app.ws_proxy import proxy_websocket

logger = setup_logging()
APP_ROLE = "compute"

INTERNAL_WS_HOST = os.getenv("INTERNAL_WS_HOST", "127.0.0.1")
INTERNAL_WS_BASE_PORT = int(os.getenv("INTERNAL_WS_PORT", "9000"))

S2S_REPO_DIR = os.getenv("S2S_REPO_DIR", "/opt/speech-to-speech")
PIPELINE_MAX_INSTANCES = int(os.getenv("PIPELINE_MAX_INSTANCES", "1"))
PIPELINE_MIN_IDLE_INSTANCES = int(os.getenv("PIPELINE_MIN_IDLE_INSTANCES", "1"))

# Core pipeline selection
LANGUAGE = os.getenv("LANGUAGE", "en").strip()
CHAT_SIZE = os.getenv("CHAT_SIZE", "10").strip()

STT = os.getenv("STT", "parakeet-tdt").strip()
LLM = os.getenv("LLM", "open_api").strip()
TTS = os.getenv("TTS", "qwen3").strip()

# General module flags
ENABLE_LIVE_TRANSCRIPTION = os.getenv("ENABLE_LIVE_TRANSCRIPTION", "1").strip().lower() in {"1", "true", "yes"}
LIVE_TRANSCRIPTION_UPDATE_INTERVAL = os.getenv("LIVE_TRANSCRIPTION_UPDATE_INTERVAL", "").strip()

# Open API / HF router
OPEN_API_MODEL_NAME = os.getenv("OPEN_API_MODEL_NAME", "Qwen/Qwen3.5-9B:together").strip()
OPEN_API_BASE_URL = os.getenv("OPEN_API_BASE_URL", "https://router.huggingface.co/v1").strip()
OPEN_API_API_KEY = os.getenv("OPEN_API_API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()
OPEN_API_STREAM = os.getenv("OPEN_API_STREAM", "1").strip().lower() in {"1", "true", "yes"}
OPEN_API_INIT_CHAT_PROMPT = os.getenv("OPEN_API_INIT_CHAT_PROMPT", "").strip()
OPEN_API_IMAGE_PATHS = os.getenv("OPEN_API_IMAGE_PATHS", "").strip()

# Optional generic extras for power users
EXTRA_S2S_ARGS = os.getenv("EXTRA_S2S_ARGS", "").strip()

SESSION_SHARED_SECRET = os.getenv("SESSION_SHARED_SECRET", "").strip()
LB_CALLBACK_AUTH_TOKEN = os.getenv("LB_CALLBACK_AUTH_TOKEN", "").strip()

INTERNAL_SLOT_WS_PATH = "/v1/realtime"
PUBLIC_WS_PATH = "/v1/realtime"
INTERNAL_WS_URL = f"ws://{INTERNAL_WS_HOST}:{INTERNAL_WS_BASE_PORT}{INTERNAL_SLOT_WS_PATH}"
INTERNAL_USAGE_PATH = "/v1/usage"
INTERNAL_USAGE_URL = f"http://{INTERNAL_WS_HOST}:{INTERNAL_WS_BASE_PORT}{INTERNAL_USAGE_PATH}"


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
        "--directory",
        S2S_REPO_DIR,
        "python",
        "s2s_pipeline.py",
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
        "--llm",
        LLM,
        "--tts",
        TTS,
    ]

    _add_bool_flag(cmd, ENABLE_LIVE_TRANSCRIPTION, "--enable_live_transcription")
    _add_str_flag(cmd, LIVE_TRANSCRIPTION_UPDATE_INTERVAL, "--live_transcription_update_interval")

    if LLM == "open_api":
        _add_str_flag(cmd, OPEN_API_MODEL_NAME, "--open_api_model_name")
        _add_str_flag(cmd, OPEN_API_BASE_URL, "--open_api_base_url")
        _add_str_flag(cmd, OPEN_API_API_KEY, "--open_api_api_key")
        _add_bool_flag(cmd, OPEN_API_STREAM, "--open_api_stream")
        _add_str_flag(cmd, OPEN_API_INIT_CHAT_PROMPT, "--open_api_init_chat_prompt")
        _add_str_flag(cmd, OPEN_API_IMAGE_PATHS, "--open_api_image_paths")

    if EXTRA_S2S_ARGS:
        cmd.extend(EXTRA_S2S_ARGS.split())

    return cmd


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
            raise RuntimeError(
                f"speech-to-speech process exited early with code {process.returncode}"
            )

        try:
            await asyncio.to_thread(_http_get_json, http_url)
            logger.info("Internal speech-to-speech listener is ready at %s", http_url)
            return
        except Exception as exc:
            last_error = exc

        if asyncio.get_running_loop().time() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for internal realtime server at {http_url}. "
                f"Last error: {last_error}"
            )

        await asyncio.sleep(2.0)


session_router = SessionRouter(
    host=INTERNAL_WS_HOST,
    base_port=INTERNAL_WS_BASE_PORT,
    ws_path=INTERNAL_SLOT_WS_PATH,
    repo_dir=S2S_REPO_DIR,
    min_idle_instances=PIPELINE_MIN_IDLE_INSTANCES,
    max_instances=PIPELINE_MAX_INSTANCES,
    build_command=build_s2s_command,
    wait_for_ready=wait_for_internal_server,
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
            "router": snapshot,
        }
    )


@app.websocket("/v1/realtime")
async def websocket_proxy(client_ws: WebSocket):
    session_payload = _get_session_payload(client_ws)
    connected_notified = False

    if SESSION_SHARED_SECRET and session_payload is None:
        await client_ws.close(code=1008, reason="Missing or invalid session token")
        return

    try:
        if session_payload is not None:
            await _notify_lb_session_event(
                session_payload["callback_url"],
                session_payload["session_token"],
                "connected",
            )
            connected_notified = True

        await proxy_websocket(
            client_ws,
            acquire_lease=lambda timeout_s: session_router.acquire(timeout_s=timeout_s),
            release_lease=session_router.release,
            describe_lease=lambda slot: f"slot {slot.slot_id} at {slot.ws_url}",
            no_capacity_reason="No pipeline capacity available",
            no_capacity_log="Failed to allocate speech-to-speech slot",
        )
    except Exception as exc:
        logger.warning("Rejected websocket session: %s", exc)
        try:
            await client_ws.close(code=1013, reason="Failed to establish reserved session")
        except Exception:
            pass
    finally:
        if connected_notified and session_payload is not None:
            try:
                await _notify_lb_session_event(
                    session_payload["callback_url"],
                    session_payload["session_token"],
                    "disconnected",
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


async def _notify_lb_session_event(callback_url: str, session_token: str, event: str) -> None:
    payload = {
        "session_token": session_token,
        "event": event,
    }
    await asyncio.to_thread(_post_json, callback_url, payload)


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

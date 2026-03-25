import asyncio
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import websockets
from websockets.exceptions import ConnectionClosed

from app.endpoint_pool_router import EndpointPoolRouter, HuggingFaceEndpointController
from app.session_router import SessionRouter

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("s2s-endpoint")


class SuppressHealthcheckAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "GET /health " not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(SuppressHealthcheckAccessFilter())

PORT = int(os.getenv("PORT", "7860"))
APP_ROLE = os.getenv("APP_ROLE", "compute").strip().lower()

INTERNAL_WS_HOST = os.getenv("INTERNAL_WS_HOST", "127.0.0.1")
INTERNAL_WS_BASE_PORT = int(os.getenv("INTERNAL_WS_PORT", "9000"))
INTERNAL_WS_URL = f"ws://{INTERNAL_WS_HOST}:{INTERNAL_WS_BASE_PORT}"

S2S_REPO_DIR = os.getenv("S2S_REPO_DIR", "/opt/speech-to-speech")
PIPELINE_MAX_INSTANCES = int(os.getenv("PIPELINE_MAX_INSTANCES", "1"))
PIPELINE_MIN_IDLE_INSTANCES = int(os.getenv("PIPELINE_MIN_IDLE_INSTANCES", "1"))

HF_ENDPOINT_NAMESPACE = os.getenv("HF_ENDPOINT_NAMESPACE", "").strip() or None
COMPUTE_ENDPOINT_NAMES = [
    name.strip() for name in os.getenv("COMPUTE_ENDPOINT_NAMES", "").split(",") if name.strip()
]
COMPUTE_ENDPOINT_SLOTS = int(os.getenv("COMPUTE_ENDPOINT_SLOTS", "1"))
COMPUTE_ENDPOINT_MIN_WARM = int(os.getenv("COMPUTE_ENDPOINT_MIN_WARM", "1"))
COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS = int(
    os.getenv("COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS", str(COMPUTE_ENDPOINT_SLOTS))
)
COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S = float(os.getenv("COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S", "300"))
COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S = float(os.getenv("COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S", "10"))
COMPUTE_ENDPOINT_WAIT_TIMEOUT_S = int(os.getenv("COMPUTE_ENDPOINT_WAIT_TIMEOUT_S", "900"))
COMPUTE_ENDPOINT_PARK_STRATEGY = os.getenv("COMPUTE_ENDPOINT_PARK_STRATEGY", "pause").strip().lower()
HF_CONTROL_TOKEN = os.getenv("HF_CONTROL_TOKEN", "").strip() or os.getenv("HF_TOKEN", "").strip() or None
DOWNSTREAM_ENDPOINT_TOKEN = os.getenv("DOWNSTREAM_ENDPOINT_TOKEN", "").strip() or HF_CONTROL_TOKEN

# Core pipeline selection
DEVICE = os.getenv("DEVICE", "cuda").strip()
LANGUAGE = os.getenv("LANGUAGE", "en").strip()
CHAT_SIZE = os.getenv("CHAT_SIZE", "10").strip()

STT = os.getenv("STT", "parakeet-tdt").strip()
LLM = os.getenv("LLM", "open_api").strip()
TTS = os.getenv("TTS", "qwen3").strip()

# General module flags
ENABLE_LIVE_TRANSCRIPTION = os.getenv("ENABLE_LIVE_TRANSCRIPTION", "1").strip().lower() in {"1", "true", "yes"}
LIVE_TRANSCRIPTION_UPDATE_INTERVAL = os.getenv("LIVE_TRANSCRIPTION_UPDATE_INTERVAL", "").strip()

# Whisper/faster-whisper only
STT_COMPILE_MODE = os.getenv("STT_COMPILE_MODE", "").strip()

# Open API / HF router
OPEN_API_MODEL_NAME = os.getenv("OPEN_API_MODEL_NAME", "Qwen/Qwen3.5-9B:together").strip()
OPEN_API_BASE_URL = os.getenv("OPEN_API_BASE_URL", "https://router.huggingface.co/v1").strip()
OPEN_API_API_KEY = os.getenv("OPEN_API_API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()
OPEN_API_STREAM = os.getenv("OPEN_API_STREAM", "1").strip().lower() in {"1", "true", "yes"}
OPEN_API_INIT_CHAT_PROMPT = os.getenv("OPEN_API_INIT_CHAT_PROMPT", "").strip()
OPEN_API_IMAGE_PATHS = os.getenv("OPEN_API_IMAGE_PATHS", "").strip()

# Optional generic extras for power users
EXTRA_S2S_ARGS = os.getenv("EXTRA_S2S_ARGS", "").strip()


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
        "websocket",
        "--ws_host",
        host,
        "--ws_port",
        str(port),
        "--device",
        DEVICE,
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

    # Live transcription is especially relevant for parakeet-tdt.
    _add_bool_flag(cmd, ENABLE_LIVE_TRANSCRIPTION, "--enable_live_transcription")
    _add_str_flag(cmd, LIVE_TRANSCRIPTION_UPDATE_INTERVAL, "--live_transcription_update_interval")

    # Whisper compile path only makes sense for whisper-family backends.
    if STT_COMPILE_MODE and STT in {"whisper", "faster-whisper"}:
        cmd.extend(["--stt_compile_mode", STT_COMPILE_MODE])

    # Open API / HF router params
    if LLM == "open_api":
        _add_str_flag(cmd, OPEN_API_MODEL_NAME, "--open_api_model_name")
        _add_str_flag(cmd, OPEN_API_BASE_URL, "--open_api_base_url")
        _add_str_flag(cmd, OPEN_API_API_KEY, "--open_api_api_key")
        _add_bool_flag(cmd, OPEN_API_STREAM, "--open_api_stream")
        _add_str_flag(cmd, OPEN_API_INIT_CHAT_PROMPT, "--open_api_init_chat_prompt")

        # Pass through exactly as a single argument string, matching your current usage.
        _add_str_flag(cmd, OPEN_API_IMAGE_PATHS, "--open_api_image_paths")

    if EXTRA_S2S_ARGS:
        cmd.extend(EXTRA_S2S_ARGS.split())

    return cmd


async def wait_for_internal_ws(
    host: str,
    port: int,
    process: Optional[subprocess.Popen],
    timeout_s: float = 900.0,
) -> None:
    ws_url = f"ws://{host}:{port}"
    start = asyncio.get_event_loop().time()
    last_error = None

    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"speech-to-speech process exited early with code {process.returncode}"
            )

        try:
            # Probe with a real websocket open / close sequence so the upstream
            # listener doesn't log invalid HTTP handshake errors for readiness checks.
            async with websockets.connect(
                ws_url,
                open_timeout=5,
                ping_interval=None,
                max_size=None,
            ):
                logger.info("Internal speech-to-speech listener is ready at %s", ws_url)
            return
        except Exception as exc:
            last_error = exc

        if asyncio.get_event_loop().time() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for internal websocket server at {ws_url}. "
                f"Last error: {last_error}"
            )

        await asyncio.sleep(2.0)


session_router = SessionRouter(
    host=INTERNAL_WS_HOST,
    base_port=INTERNAL_WS_BASE_PORT,
    repo_dir=S2S_REPO_DIR,
    min_idle_instances=PIPELINE_MIN_IDLE_INSTANCES,
    max_instances=PIPELINE_MAX_INSTANCES,
    build_command=build_s2s_command,
    wait_for_ready=wait_for_internal_ws,
)


def build_lb_router() -> EndpointPoolRouter:
    if not COMPUTE_ENDPOINT_NAMES:
        raise RuntimeError("COMPUTE_ENDPOINT_NAMES must be set when APP_ROLE=load_balancer")

    controller = HuggingFaceEndpointController(
        namespace=HF_ENDPOINT_NAMESPACE,
        token=HF_CONTROL_TOKEN,
        wait_timeout_s=COMPUTE_ENDPOINT_WAIT_TIMEOUT_S,
        active_min_replica=1,
        active_max_replica=1,
        park_strategy=COMPUTE_ENDPOINT_PARK_STRATEGY,
    )

    return EndpointPoolRouter(
        endpoint_names=COMPUTE_ENDPOINT_NAMES,
        endpoint_slots=COMPUTE_ENDPOINT_SLOTS,
        min_warm_endpoints=COMPUTE_ENDPOINT_MIN_WARM,
        wake_threshold_slots=COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS,
        idle_park_timeout_s=COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S,
        reconcile_interval_s=COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S,
        controller=controller,
    )


if APP_ROLE == "compute":
    ws_router = session_router
elif APP_ROLE == "load_balancer":
    ws_router = build_lb_router()
else:
    raise RuntimeError("APP_ROLE must be either 'compute' or 'load_balancer'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ws_router.start()
    try:
        yield
    finally:
        await ws_router.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    payload = {
        "message": "s2s endpoint is up",
        "health": "/health",
        "websocket": "/ws",
        "role": APP_ROLE,
        "config": {
            "stt": STT,
            "llm": LLM,
            "tts": TTS,
            "device": DEVICE,
            "language": LANGUAGE,
        },
    }
    if APP_ROLE == "compute":
        payload["internal_ws"] = INTERNAL_WS_URL
    else:
        payload["compute_endpoints"] = COMPUTE_ENDPOINT_NAMES
    return payload


@app.get("/health")
async def health():
    healthy, detail, snapshot = await ws_router.healthcheck()
    if not healthy:
        raise HTTPException(status_code=503, detail=detail or "session router is not ready")

    payload = {
        "status": "ok",
        "role": APP_ROLE,
        "stt": STT,
        "llm": LLM,
        "tts": TTS,
        "router": snapshot,
    }
    if APP_ROLE == "compute":
        payload["internal_ws_base"] = INTERNAL_WS_URL
    else:
        payload["compute_endpoints"] = COMPUTE_ENDPOINT_NAMES
    return JSONResponse(payload)


def build_upstream_headers() -> Optional[list[tuple[str, str]]]:
    if APP_ROLE != "load_balancer" or not DOWNSTREAM_ENDPOINT_TOKEN:
        return None
    return [("Authorization", f"Bearer {DOWNSTREAM_ENDPOINT_TOKEN}")]


@app.websocket("/ws")
async def websocket_proxy(client_ws: WebSocket):
    slot = None

    try:
        slot = await ws_router.acquire(timeout_s=900.0)
    except Exception as exc:
        await client_ws.close(code=1013, reason="No pipeline capacity available")
        logger.warning("Failed to allocate speech-to-speech slot: %s", exc)
        return

    await client_ws.accept()
    logger.info("Client websocket connected to slot %s at %s", slot.slot_id, slot.ws_url)

    try:
        async with websockets.connect(
            slot.ws_url,
            additional_headers=build_upstream_headers(),
            open_timeout=30,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as upstream_ws:

            async def client_to_upstream():
                while True:
                    message = await client_ws.receive()

                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()

                    if "bytes" in message and message["bytes"] is not None:
                        await upstream_ws.send(message["bytes"])
                    elif "text" in message and message["text"] is not None:
                        await upstream_ws.send(message["text"])

            async def upstream_to_client():
                while True:
                    msg = await upstream_ws.recv()
                    if isinstance(msg, bytes):
                        await client_ws.send_bytes(msg)
                    else:
                        await client_ws.send_text(msg)

            await asyncio.gather(client_to_upstream(), upstream_to_client())

    except WebSocketDisconnect:
        logger.info("Client websocket disconnected")
    except ConnectionClosed:
        logger.info("Upstream websocket disconnected")
        try:
            await client_ws.close()
        except Exception:
            pass
    except Exception:
        logger.exception("Websocket proxy failed")
        try:
            await client_ws.close(code=1011, reason="Proxy failure")
        except Exception:
            pass
    finally:
        if slot is not None:
            await ws_router.release(slot.slot_id)

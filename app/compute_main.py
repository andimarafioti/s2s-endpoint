import asyncio
import os
import subprocess
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse
import websockets

from app.app_utils import build_lifespan, setup_logging
from app.session_router import SessionRouter
from app.ws_proxy import proxy_websocket

logger = setup_logging()
APP_ROLE = "compute"

INTERNAL_WS_HOST = os.getenv("INTERNAL_WS_HOST", "127.0.0.1")
INTERNAL_WS_BASE_PORT = int(os.getenv("INTERNAL_WS_PORT", "9000"))
INTERNAL_WS_URL = f"ws://{INTERNAL_WS_HOST}:{INTERNAL_WS_BASE_PORT}"

S2S_REPO_DIR = os.getenv("S2S_REPO_DIR", "/opt/speech-to-speech")
PIPELINE_MAX_INSTANCES = int(os.getenv("PIPELINE_MAX_INSTANCES", "1"))
PIPELINE_MIN_IDLE_INSTANCES = int(os.getenv("PIPELINE_MIN_IDLE_INSTANCES", "1"))

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

    _add_bool_flag(cmd, ENABLE_LIVE_TRANSCRIPTION, "--enable_live_transcription")
    _add_str_flag(cmd, LIVE_TRANSCRIPTION_UPDATE_INTERVAL, "--live_transcription_update_interval")

    if STT_COMPILE_MODE and STT in {"whisper", "faster-whisper"}:
        cmd.extend(["--stt_compile_mode", STT_COMPILE_MODE])

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

app = FastAPI(lifespan=build_lifespan(session_router))


@app.get("/")
async def root():
    return {
        "message": "s2s compute endpoint is up",
        "role": APP_ROLE,
        "health": "/health",
        "websocket": "/ws",
        "internal_ws": INTERNAL_WS_URL,
        "config": {
            "stt": STT,
            "llm": LLM,
            "tts": TTS,
            "device": DEVICE,
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
            "stt": STT,
            "llm": LLM,
            "tts": TTS,
            "router": snapshot,
        }
    )


@app.websocket("/ws")
async def websocket_proxy(client_ws: WebSocket):
    await proxy_websocket(
        client_ws,
        acquire_lease=lambda timeout_s: session_router.acquire(timeout_s=timeout_s),
        release_lease=session_router.release,
        describe_lease=lambda slot: f"slot {slot.slot_id} at {slot.ws_url}",
        no_capacity_reason="No pipeline capacity available",
        no_capacity_log="Failed to allocate speech-to-speech slot",
    )

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    await session_router.start()
    try:
        yield
    finally:
        await session_router.stop()


app = FastAPI(lifespan=lifespan)


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
    slot = None

    try:
        slot = await session_router.acquire(timeout_s=900.0)
    except Exception as exc:
        await client_ws.close(code=1013, reason="No pipeline capacity available")
        logger.warning("Failed to allocate speech-to-speech slot: %s", exc)
        return

    await client_ws.accept()
    logger.info("Client websocket connected to slot %s at %s", slot.slot_id, slot.ws_url)

    try:
        async with websockets.connect(
            slot.ws_url,
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
            await session_router.release(slot.slot_id)

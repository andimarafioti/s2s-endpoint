import asyncio
import logging
import os
import signal
import subprocess
from contextlib import asynccontextmanager, suppress
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import websockets
from websockets.exceptions import ConnectionClosed

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

INTERNAL_WS_HOST = os.getenv("INTERNAL_WS_HOST", "127.0.0.1")
INTERNAL_WS_PORT = int(os.getenv("INTERNAL_WS_PORT", "9000"))
INTERNAL_WS_URL = f"ws://{INTERNAL_WS_HOST}:{INTERNAL_WS_PORT}"

S2S_REPO_DIR = os.getenv("S2S_REPO_DIR", "/opt/speech-to-speech")

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

pipeline_process: Optional[subprocess.Popen] = None
internal_ws_ready = False
internal_ws_error: Optional[str] = None
internal_ws_monitor_task: Optional[asyncio.Task] = None


def _add_bool_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def _add_str_flag(cmd: list[str], value: str, flag: str) -> None:
    if value:
        cmd.extend([flag, value])


def build_s2s_command() -> list[str]:
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
        INTERNAL_WS_HOST,
        "--ws_port",
        str(INTERNAL_WS_PORT),
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


async def wait_for_internal_ws(timeout_s: float = 900.0) -> None:
    start = asyncio.get_event_loop().time()
    last_error = None

    while True:
        if pipeline_process is not None and pipeline_process.poll() is not None:
            raise RuntimeError(
                f"speech-to-speech process exited early with code {pipeline_process.returncode}"
            )

        try:
            # Probe with a real websocket open / close sequence so the upstream
            # listener doesn't log invalid HTTP handshake errors for readiness checks.
            async with websockets.connect(
                INTERNAL_WS_URL,
                open_timeout=5,
                ping_interval=None,
                max_size=None,
            ):
                logger.info("Internal speech-to-speech listener is ready at %s", INTERNAL_WS_URL)
            return
        except Exception as exc:
            last_error = exc

        if asyncio.get_event_loop().time() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for internal websocket server at {INTERNAL_WS_URL}. "
                f"Last error: {last_error}"
            )

        await asyncio.sleep(2.0)


async def monitor_internal_ws_readiness() -> None:
    global internal_ws_ready, internal_ws_error

    try:
        await wait_for_internal_ws(timeout_s=900.0)
        internal_ws_ready = True
        internal_ws_error = None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        internal_ws_ready = False
        internal_ws_error = str(exc)
        logger.error("Internal websocket did not become ready: %s", exc)


def start_pipeline() -> None:
    global pipeline_process

    if pipeline_process is not None and pipeline_process.poll() is None:
        logger.info("speech-to-speech process already running")
        return

    cmd = build_s2s_command()
    logger.info("Starting speech-to-speech subprocess:\n%s", " ".join(cmd))

    env = os.environ.copy()

    pipeline_process = subprocess.Popen(
        cmd,
        cwd=S2S_REPO_DIR,
        env=env,
        stdout=None,
        stderr=None,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )


def stop_pipeline() -> None:
    global pipeline_process

    if pipeline_process is None:
        return

    if pipeline_process.poll() is not None:
        logger.info("speech-to-speech process already stopped")
        return

    logger.info("Stopping speech-to-speech subprocess")

    try:
        if os.name != "nt":
            os.killpg(os.getpgid(pipeline_process.pid), signal.SIGTERM)
        else:
            pipeline_process.terminate()
        pipeline_process.wait(timeout=20)
    except Exception:
        logger.exception("Graceful shutdown failed, killing subprocess")
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(pipeline_process.pid), signal.SIGKILL)
            else:
                pipeline_process.kill()
        except Exception:
            logger.exception("Failed to kill subprocess")
    finally:
        pipeline_process = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global internal_ws_ready, internal_ws_error, internal_ws_monitor_task

    internal_ws_ready = False
    internal_ws_error = None
    start_pipeline()
    internal_ws_monitor_task = asyncio.create_task(monitor_internal_ws_readiness())
    try:
        yield
    finally:
        if internal_ws_monitor_task is not None:
            internal_ws_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await internal_ws_monitor_task
            internal_ws_monitor_task = None
        internal_ws_ready = False
        internal_ws_error = None
        stop_pipeline()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "message": "s2s endpoint is up",
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
    if pipeline_process is None:
        raise HTTPException(status_code=503, detail="speech-to-speech process not started")

    if pipeline_process.poll() is not None:
        raise HTTPException(
            status_code=503,
            detail=f"speech-to-speech process exited with code {pipeline_process.returncode}",
        )

    if not internal_ws_ready:
        detail = internal_ws_error or f"internal websocket not ready at {INTERNAL_WS_URL}"
        raise HTTPException(status_code=503, detail=detail)

    return JSONResponse(
        {
            "status": "ok",
            "internal_ws": INTERNAL_WS_URL,
            "stt": STT,
            "llm": LLM,
            "tts": TTS,
        }
    )


@app.websocket("/ws")
async def websocket_proxy(client_ws: WebSocket):
    await client_ws.accept()
    logger.info("Client websocket connected")

    try:
        async with websockets.connect(
            INTERNAL_WS_URL,
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

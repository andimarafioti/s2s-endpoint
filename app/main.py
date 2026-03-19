import asyncio
import logging
import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
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

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "7860"))

INTERNAL_WS_HOST = os.getenv("INTERNAL_WS_HOST", "127.0.0.1")
INTERNAL_WS_PORT = int(os.getenv("INTERNAL_WS_PORT", "9000"))
INTERNAL_WS_URL = f"ws://{INTERNAL_WS_HOST}:{INTERNAL_WS_PORT}"

S2S_REPO_DIR = os.getenv("S2S_REPO_DIR", "/opt/speech-to-speech")

# Baseline model choices. Keep them simple for a first deployment.
# You can override any of these in the endpoint env vars.
LM_MODEL_NAME = os.getenv("LM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
TTS = os.getenv("TTS", "pocket")
POCKET_TTS_VOICE = os.getenv("POCKET_TTS_VOICE", "jean")
DEVICE = os.getenv("DEVICE", "cuda")
LANGUAGE = os.getenv("LANGUAGE", "en")
CHAT_SIZE = os.getenv("CHAT_SIZE", "10")
STT_COMPILE_MODE = os.getenv("STT_COMPILE_MODE", "reduce-overhead")

# Optional extra CLI args for speech-to-speech, space-separated.
# Example:
#   EXTRA_S2S_ARGS="--stt_model_name large-v3 --temperature 0.7"
EXTRA_S2S_ARGS = os.getenv("EXTRA_S2S_ARGS", "").strip()

# If you later want to use an OpenAI-compatible API-backed LLM instead of a local LM,
# set USE_OPENAI_API_LLM=1 and configure the related env vars.
USE_OPENAI_API_LLM = os.getenv("USE_OPENAI_API_LLM", "0") == "1"
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "")

pipeline_process: Optional[subprocess.Popen] = None

def build_s2s_command() -> list[str]:
    cmd = [
        "uv",
        "run",
        "--directory",
        S2S_REPO_DIR,
        "python",
        "s2s_pipeline.py",
        "--mode", "websocket",
        "--ws_host", INTERNAL_WS_HOST,
        "--ws_port", str(INTERNAL_WS_PORT),
        "--device", DEVICE,
        "--language", LANGUAGE,
        "--chat_size", CHAT_SIZE,
        "--tts", TTS,
    ]

    if STT_COMPILE_MODE:
        cmd += ["--stt_compile_mode", STT_COMPILE_MODE]

    if TTS == "pocket" and POCKET_TTS_VOICE:
        cmd += ["--pocket_tts_voice", POCKET_TTS_VOICE]

    if USE_OPENAI_API_LLM:
        cmd += [
            "--llm", "open-api",
            "--open_api_base_url", OPENAI_API_BASE,
            "--open_api_key", OPENAI_API_KEY,
            "--open_api_model_name", OPENAI_API_MODEL,
        ]
    else:
        cmd += ["--lm_model_name", LM_MODEL_NAME]

    if EXTRA_S2S_ARGS:
        cmd += EXTRA_S2S_ARGS.split()

    return cmd


async def wait_for_internal_ws(timeout_s: float = 900.0) -> None:
    """
    Wait until the internal speech-to-speech websocket server accepts connections.
    First model load can take a while on endpoint startup.
    """
    start = asyncio.get_event_loop().time()
    last_error = None

    while True:
        if pipeline_process is not None and pipeline_process.poll() is not None:
            raise RuntimeError(
                f"speech-to-speech process exited early with code {pipeline_process.returncode}"
            )

        try:
            async with websockets.connect(
                INTERNAL_WS_URL,
                open_timeout=5,
                ping_interval=None,
                max_size=None,
            ):
                logger.info("Internal speech-to-speech websocket is ready at %s", INTERNAL_WS_URL)
                return
        except Exception as exc:
            last_error = exc

        if asyncio.get_event_loop().time() - start > timeout_s:
            raise RuntimeError(
                f"Timed out waiting for internal websocket server at {INTERNAL_WS_URL}. "
                f"Last error: {last_error}"
            )

        await asyncio.sleep(2.0)


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
        stdout=sys.stdout,
        stderr=sys.stderr,
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
    start_pipeline()
    try:
        yield
    finally:
        stop_pipeline()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "message": "s2s endpoint is up",
        "health": "/health",
        "websocket": "/ws",
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

    try:
        await asyncio.wait_for(wait_for_internal_ws(timeout_s=5), timeout=6)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"internal websocket not ready: {exc}") from exc

    return JSONResponse({"status": "ok", "internal_ws": INTERNAL_WS_URL})


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

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import websockets
from websockets.exceptions import ConnectionClosed

from app.endpoint_pool_router import EndpointPoolRouter, HuggingFaceEndpointController

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
APP_ROLE = "load_balancer"

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


def build_lb_router() -> EndpointPoolRouter:
    if not COMPUTE_ENDPOINT_NAMES:
        raise RuntimeError("COMPUTE_ENDPOINT_NAMES must be set for the load-balancer app")

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


endpoint_router = build_lb_router()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await endpoint_router.start()
    try:
        yield
    finally:
        await endpoint_router.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "message": "s2s load balancer endpoint is up",
        "role": APP_ROLE,
        "health": "/health",
        "websocket": "/ws",
        "compute_endpoints": COMPUTE_ENDPOINT_NAMES,
    }


@app.get("/health")
async def health():
    healthy, detail, snapshot = await endpoint_router.healthcheck()
    if not healthy:
        raise HTTPException(status_code=503, detail=detail or "endpoint router is not ready")

    return JSONResponse(
        {
            "status": "ok",
            "role": APP_ROLE,
            "compute_endpoints": COMPUTE_ENDPOINT_NAMES,
            "router": snapshot,
        }
    )


def build_upstream_headers() -> Optional[list[tuple[str, str]]]:
    if not DOWNSTREAM_ENDPOINT_TOKEN:
        return None
    return [("Authorization", f"Bearer {DOWNSTREAM_ENDPOINT_TOKEN}")]


@app.websocket("/ws")
async def websocket_proxy(client_ws: WebSocket):
    slot = None

    try:
        slot = await endpoint_router.acquire(timeout_s=900.0)
    except Exception as exc:
        await client_ws.close(code=1013, reason="No compute endpoint capacity available")
        logger.warning("Failed to allocate compute endpoint slot: %s", exc)
        return

    await client_ws.accept()
    logger.info("Client websocket connected to endpoint %s at %s", slot.endpoint_name, slot.ws_url)

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
            await endpoint_router.release(slot.slot_id)

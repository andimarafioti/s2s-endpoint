import os
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import JSONResponse

from app.app_utils import build_lifespan, setup_logging
from app.endpoint_pool_router import EndpointPoolRouter, HuggingFaceEndpointController
from app.ws_proxy import proxy_websocket

setup_logging()
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

app = FastAPI(lifespan=build_lifespan(endpoint_router))


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
    await proxy_websocket(
        client_ws,
        acquire_lease=lambda timeout_s: endpoint_router.acquire(timeout_s=timeout_s),
        release_lease=endpoint_router.release,
        describe_lease=lambda slot: f"endpoint {slot.endpoint_name} at {slot.ws_url}",
        no_capacity_reason="No compute endpoint capacity available",
        no_capacity_log="Failed to allocate compute endpoint slot",
        additional_headers=build_upstream_headers(),
    )

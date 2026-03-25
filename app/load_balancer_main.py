import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse

from app.app_utils import build_lifespan, public_base_url, setup_logging
from app.direct_session_manager import DirectSessionManager
from app.endpoint_pool_router import EndpointPoolRouter, HuggingFaceEndpointController

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
COMPUTE_ENDPOINT_WAKING_CAPACITY_TIMEOUT_S = float(
    os.getenv("COMPUTE_ENDPOINT_WAKING_CAPACITY_TIMEOUT_S", "300")
)
COMPUTE_ENDPOINT_WAIT_TIMEOUT_S = int(os.getenv("COMPUTE_ENDPOINT_WAIT_TIMEOUT_S", "900"))
COMPUTE_ENDPOINT_PARK_STRATEGY = os.getenv("COMPUTE_ENDPOINT_PARK_STRATEGY", "pause").strip().lower()
HF_CONTROL_TOKEN = os.getenv("HF_CONTROL_TOKEN", "").strip() or os.getenv("HF_TOKEN", "").strip() or None

SESSION_SHARED_SECRET = os.getenv("SESSION_SHARED_SECRET", "").strip()
SESSION_PENDING_TIMEOUT_S = float(os.getenv("SESSION_PENDING_TIMEOUT_S", "60"))
SESSION_TOKEN_TTL_S = float(os.getenv("SESSION_TOKEN_TTL_S", "86400"))
SESSION_REAP_INTERVAL_S = float(os.getenv("SESSION_REAP_INTERVAL_S", "5"))


def build_endpoint_router() -> EndpointPoolRouter:
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
        waking_capacity_timeout_s=COMPUTE_ENDPOINT_WAKING_CAPACITY_TIMEOUT_S,
        controller=controller,
    )


session_manager = DirectSessionManager(
    endpoint_router=build_endpoint_router(),
    session_shared_secret=SESSION_SHARED_SECRET,
    pending_timeout_s=SESSION_PENDING_TIMEOUT_S,
    session_token_ttl_s=SESSION_TOKEN_TTL_S,
    reap_interval_s=SESSION_REAP_INTERVAL_S,
)

app = FastAPI(lifespan=build_lifespan(session_manager))


@app.get("/")
async def root():
    return {
        "message": "s2s load balancer endpoint is up",
        "role": APP_ROLE,
        "health": "/health",
        "session": "/session",
        "compute_endpoints": COMPUTE_ENDPOINT_NAMES,
    }


@app.get("/health")
async def health():
    healthy, detail, snapshot = await session_manager.healthcheck()
    if not healthy:
        raise HTTPException(status_code=503, detail=detail or "endpoint router is not ready")

    return JSONResponse(
        {
            "status": "ok",
            "role": APP_ROLE,
            "compute_endpoints": COMPUTE_ENDPOINT_NAMES,
            "sessions": snapshot,
        }
    )


@app.post("/session")
async def create_session(request: Request):
    try:
        allocation = await session_manager.allocate(public_base_url(request))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to allocate compute endpoint: {exc}") from exc

    return JSONResponse(allocation)


@app.post("/internal/sessions/{session_id}/event")
async def session_event(session_id: str, payload: dict[str, Any]):
    session_token = str(payload.get("session_token", "")).strip()
    event = str(payload.get("event", "")).strip()
    if not session_token:
        raise HTTPException(status_code=400, detail="session_token is required")
    if not event:
        raise HTTPException(status_code=400, detail="event is required")

    try:
        result = await session_manager.handle_event(session_id, session_token, event)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown session id") from None
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    return JSONResponse(result)


@app.websocket("/ws")
async def deprecated_websocket_route(client_ws: WebSocket):
    await client_ws.close(code=1008, reason="Use POST /session and connect directly to the returned compute websocket URL")

import logging
import os
import secrets
from time import monotonic
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse

from app.app_utils import build_lifespan, elapsed_ms, public_base_url, setup_logging
from app.dashboard_history_store import HuggingFaceBucketHistoryStore, ReadOnlyDashboardHistoryStore
from app.dashboard_preview import DashboardPreviewSessionManager
from app.direct_session_manager import DirectSessionManager
from app.endpoint_pool_router import (
    EndpointCapacityTimeoutError,
    EndpointPoolRouter,
    HuggingFaceEndpointController,
    fetch_compute_active_sessions,
)
from app.swarm_dashboard import SwarmDashboard

logger = setup_logging()
APP_ROLE = "load_balancer"

HF_ENDPOINT_NAMESPACE = os.getenv("HF_ENDPOINT_NAMESPACE", "").strip() or None
COMPUTE_ENDPOINT_NAMES_ENV = os.getenv("COMPUTE_ENDPOINT_NAMES", "").strip()
COMPUTE_ENDPOINT_NAMES = [name.strip() for name in COMPUTE_ENDPOINT_NAMES_ENV.split(",") if name.strip()]
COMPUTE_ENDPOINT_SLOTS = int(os.getenv("COMPUTE_ENDPOINT_SLOTS", "1"))
COMPUTE_ENDPOINT_MIN_WARM = int(os.getenv("COMPUTE_ENDPOINT_MIN_WARM", "1"))
COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS = int(
    os.getenv("COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS", str(COMPUTE_ENDPOINT_SLOTS))
)
COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S = float(os.getenv("COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S", "600"))
COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S = float(os.getenv("COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S", "10"))
COMPUTE_ENDPOINT_WAKING_CAPACITY_TIMEOUT_S = float(
    os.getenv("COMPUTE_ENDPOINT_WAKING_CAPACITY_TIMEOUT_S", "300")
)
COMPUTE_ENDPOINT_PARK_COOLDOWN_S = float(os.getenv("COMPUTE_ENDPOINT_PARK_COOLDOWN_S", "180"))
COMPUTE_ENDPOINT_WAIT_TIMEOUT_S = int(os.getenv("COMPUTE_ENDPOINT_WAIT_TIMEOUT_S", "900"))
COMPUTE_ENDPOINT_PARK_STRATEGY = os.getenv("COMPUTE_ENDPOINT_PARK_STRATEGY", "pause").strip().lower()
COMPUTE_ENDPOINT_AUTO_RESTART = os.getenv("COMPUTE_ENDPOINT_AUTO_RESTART", "true").strip().lower() in {"true", "1", "yes"}
COMPUTE_ENDPOINT_MAX_RESTART_ATTEMPTS = int(os.getenv("COMPUTE_ENDPOINT_MAX_RESTART_ATTEMPTS", "3"))
COMPUTE_ENDPOINT_RESTART_BACKOFF_S = float(os.getenv("COMPUTE_ENDPOINT_RESTART_BACKOFF_S", "30"))
COMPUTE_ENDPOINT_RESTART_BACKOFF_MAX_S = float(os.getenv("COMPUTE_ENDPOINT_RESTART_BACKOFF_MAX_S", "300"))
COMPUTE_ENDPOINT_RESTART_STABLE_RUNNING_S = float(os.getenv("COMPUTE_ENDPOINT_RESTART_STABLE_RUNNING_S", "120"))
COMPUTE_ENDPOINT_DRAIN_RESTART_TIMEOUT_S = float(os.getenv("COMPUTE_ENDPOINT_DRAIN_RESTART_TIMEOUT_S", "600"))
HF_CONTROL_TOKEN = os.getenv("HF_CONTROL_TOKEN", "").strip() or os.getenv("HF_TOKEN", "").strip() or None
LB_ADMIN_AUTH_TOKEN = os.getenv("LB_ADMIN_AUTH_TOKEN", "").strip() or HF_CONTROL_TOKEN

SESSION_SHARED_SECRET = os.getenv("SESSION_SHARED_SECRET", "").strip()
SESSION_PENDING_TIMEOUT_S = float(os.getenv("SESSION_PENDING_TIMEOUT_S", "60"))
SESSION_TOKEN_TTL_S = float(os.getenv("SESSION_TOKEN_TTL_S", "86400"))
SESSION_REAP_INTERVAL_S = float(os.getenv("SESSION_REAP_INTERVAL_S", "5"))
DASHBOARD_SAMPLE_INTERVAL_S = float(os.getenv("DASHBOARD_SAMPLE_INTERVAL_S", "15"))
DASHBOARD_RETENTION_MINUTES = int(os.getenv("DASHBOARD_RETENTION_MINUTES", str(28 * 24 * 60)))
DASHBOARD_BUCKET_ID = os.getenv("DASHBOARD_BUCKET_ID", "").strip() or None
DASHBOARD_BUCKET_PREFIX = os.getenv("DASHBOARD_BUCKET_PREFIX", "s2s-endpoint/swarm-dashboard").strip()
DASHBOARD_BUCKET_TOKEN = os.getenv("DASHBOARD_BUCKET_TOKEN", "").strip() or HF_CONTROL_TOKEN
DASHBOARD_PREVIEW_MODE = os.getenv("DASHBOARD_PREVIEW_MODE", "").strip().lower() in {"true", "1", "yes"}
DASHBOARD_PREVIEW_SENTINELS = {"test", "preview", "dashboard_preview"}
if len(COMPUTE_ENDPOINT_NAMES) == 1 and COMPUTE_ENDPOINT_NAMES[0].strip().lower() in DASHBOARD_PREVIEW_SENTINELS:
    DASHBOARD_PREVIEW_MODE = True
    COMPUTE_ENDPOINT_NAMES = []
if DASHBOARD_PREVIEW_MODE and not COMPUTE_ENDPOINT_NAMES:
    COMPUTE_ENDPOINT_NAMES = ["preview-compute-01", "preview-compute-02", "preview-compute-03", "preview-compute-04"]


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
        park_cooldown_s=COMPUTE_ENDPOINT_PARK_COOLDOWN_S,
        controller=controller,
        auto_restart=COMPUTE_ENDPOINT_AUTO_RESTART,
        max_restart_attempts=COMPUTE_ENDPOINT_MAX_RESTART_ATTEMPTS,
        restart_backoff_s=COMPUTE_ENDPOINT_RESTART_BACKOFF_S,
        restart_backoff_max_s=COMPUTE_ENDPOINT_RESTART_BACKOFF_MAX_S,
        restart_stable_running_s=COMPUTE_ENDPOINT_RESTART_STABLE_RUNNING_S,
        drain_restart_timeout_s=COMPUTE_ENDPOINT_DRAIN_RESTART_TIMEOUT_S,
        compute_usage_fetcher=fetch_compute_active_sessions,
        # How long a previously observed usage count stays trusted when
        # health polls fail transiently. Must be comfortably above the
        # reconcile interval (10s): the default 60s means roughly six
        # consecutive failed polls before a synced node loses capacity.
        # Setting it below the reconcile interval revokes on a single blip.
        usage_sync_stale_ttl_s=float(os.getenv("COMPUTE_USAGE_STALE_TTL_S", "60")),
    )


if DASHBOARD_PREVIEW_MODE:
    session_manager = DashboardPreviewSessionManager(endpoint_slots=COMPUTE_ENDPOINT_SLOTS)
else:
    session_manager = DirectSessionManager(
        endpoint_router=build_endpoint_router(),
        session_shared_secret=SESSION_SHARED_SECRET,
        pending_timeout_s=SESSION_PENDING_TIMEOUT_S,
        session_token_ttl_s=SESSION_TOKEN_TTL_S,
        reap_interval_s=SESSION_REAP_INTERVAL_S,
    )

dashboard_history_store = None
if DASHBOARD_BUCKET_ID:
    dashboard_history_store = HuggingFaceBucketHistoryStore(
        bucket_id=DASHBOARD_BUCKET_ID,
        prefix=DASHBOARD_BUCKET_PREFIX,
        token=DASHBOARD_BUCKET_TOKEN,
    )
    if DASHBOARD_PREVIEW_MODE:
        dashboard_history_store = ReadOnlyDashboardHistoryStore(dashboard_history_store)

dashboard = SwarmDashboard(
    snapshot_provider=session_manager.healthcheck,
    sample_interval_s=DASHBOARD_SAMPLE_INTERVAL_S,
    retention_minutes=DASHBOARD_RETENTION_MINUTES,
    history_store=dashboard_history_store,
    restore_history_in_background=True,
)


async def record_abnormal_session_disconnect(result: dict[str, object]) -> None:
    await dashboard.record_session_event(
        "disconnected",
        conversation_duration_s=result.get("conversation_duration_s"),
        conversation_counted=bool(result.get("conversation_counted")),
    )


class LoadBalancerRuntime:
    async def start(self) -> None:
        # Dashboard preview mode uses a synthetic manager with no real sessions.
        if hasattr(session_manager, "set_abnormal_disconnect_handler"):
            session_manager.set_abnormal_disconnect_handler(record_abnormal_session_disconnect)
        await session_manager.start()
        await dashboard.start()

    async def stop(self) -> None:
        await dashboard.stop()
        await session_manager.stop()


app = FastAPI(lifespan=build_lifespan(LoadBalancerRuntime()))


def _log_session_allocation_outcome(
    outcome: str,
    *,
    allocation: dict[str, object] | None,
    allocation_wait_ms: int | None,
    allocation_total_ms: int,
    level: int,
    error: str | None = None,
) -> None:
    allocation = allocation or {}
    session_id = allocation.get("session_id")
    endpoint_name = allocation.get("endpoint_name")
    slot_id = allocation.get("slot_id")
    waited_for_capacity = allocation.get("waited_for_capacity")
    extra = {
        "session_id": session_id,
        "endpoint_name": endpoint_name,
        "slot_id": slot_id,
        "allocation_wait_ms": allocation_wait_ms,
        "allocation_total_ms": allocation_total_ms,
        "outcome": outcome,
        "waited_for_capacity": waited_for_capacity,
        "allocation_error": error,
        "http_route": "POST /session",
    }
    message = (
        "Session allocation outcome outcome=%s session_id=%s endpoint_name=%s "
        "slot_id=%s allocation_wait_ms=%s allocation_total_ms=%d "
        "waited_for_capacity=%s"
    )
    args: list[object] = [
        outcome,
        session_id,
        endpoint_name,
        slot_id,
        allocation_wait_ms,
        allocation_total_ms,
        waited_for_capacity,
    ]
    if error is not None:
        message += " error=%s"
        args.append(error)

    logger.log(level, message, *args, extra=extra)


def _allocation_wait_ms(allocation: dict[str, object], *, fallback_ms: int) -> int:
    value = allocation.get("allocation_wait_ms")
    if value is None:
        return fallback_ms
    return max(int(value), 0)


def _public_session_allocation(allocation: dict[str, object]) -> dict[str, object]:
    return {
        key: allocation[key]
        for key in (
            "session_id",
            "websocket_url",
            "connect_url",
            "session_token",
            "pending_timeout_s",
        )
        if key in allocation
    }


@app.get("/")
async def root():
    return {
        "message": "s2s load balancer endpoint is up",
        "role": APP_ROLE,
        "ready": "/ready",
        "health": "/health",
        "session": "/session",
        "dashboard": "/dashboard",
        "dashboard_data": "/dashboard/data",
        "compute_endpoints": COMPUTE_ENDPOINT_NAMES,
        "dashboard_preview_mode": DASHBOARD_PREVIEW_MODE,
    }


@app.get("/ready")
async def ready():
    return JSONResponse(
        {
            "status": "ok",
            "role": APP_ROLE,
        }
    )


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
            "dashboard_preview_mode": DASHBOARD_PREVIEW_MODE,
            "sessions": snapshot,
        }
    )


@app.post("/session")
async def create_session(request: Request):
    await dashboard.record_session_request()
    allocation_started_at = monotonic()
    try:
        allocation = await session_manager.allocate(public_base_url(request))
    except Exception as exc:
        allocation_total_ms = elapsed_ms(allocation_started_at, monotonic())
        waited_for_capacity = isinstance(exc, EndpointCapacityTimeoutError)
        failure_allocation = {"waited_for_capacity": waited_for_capacity}
        _log_session_allocation_outcome(
            "allocation_failed",
            allocation=failure_allocation,
            allocation_wait_ms=allocation_total_ms if waited_for_capacity else None,
            allocation_total_ms=allocation_total_ms,
            level=logging.WARNING,
            error=str(exc),
        )
        await dashboard.record_session_allocation_failure()
        raise HTTPException(status_code=503, detail=f"Failed to allocate compute endpoint: {exc}") from exc

    allocation_total_ms = elapsed_ms(allocation_started_at, monotonic())
    allocation_wait_ms = _allocation_wait_ms(allocation, fallback_ms=allocation_total_ms)
    allocation.setdefault("allocation_wait_ms", allocation_wait_ms)

    if await request.is_disconnected():
        session_id = allocation.get("session_id")
        if session_id and hasattr(session_manager, "cancel_pending_session"):
            await session_manager.cancel_pending_session(session_id)
        _log_session_allocation_outcome(
            "client_disconnected",
            allocation=allocation,
            allocation_wait_ms=allocation_wait_ms,
            allocation_total_ms=allocation_total_ms,
            level=logging.WARNING,
        )
        raise HTTPException(status_code=503, detail="Client disconnected before session could be delivered")

    await dashboard.record_session_allocation_success()
    _log_session_allocation_outcome(
        "success",
        allocation=allocation,
        allocation_wait_ms=allocation_wait_ms,
        allocation_total_ms=allocation_total_ms,
        level=logging.INFO,
    )
    return JSONResponse(_public_session_allocation(allocation))


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
        if event == "disconnected":
            return JSONResponse({"status": "ok", "session_id": session_id, "state": "already_released"})
        raise HTTPException(status_code=404, detail="Unknown session id") from None
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    await dashboard.record_session_event(
        event,
        conversation_duration_s=result.get("conversation_duration_s"),
        conversation_counted=bool(result.get("conversation_counted")),
    )
    return JSONResponse(result)


@app.get("/internal/endpoints/{endpoint_name}")
async def endpoint_status(endpoint_name: str, request: Request):
    require_admin_auth(request)

    endpoint_snapshot = await get_endpoint_snapshot(endpoint_name)
    return JSONResponse(
        {
            "status": "ok",
            "endpoint_name": endpoint_name,
            "endpoint": endpoint_snapshot,
        }
    )


@app.post("/internal/endpoints/{endpoint_name}/drain")
async def endpoint_drain(endpoint_name: str, request: Request, payload: dict[str, Any]):
    require_admin_auth(request)

    endpoint_router = getattr(session_manager, "endpoint_router", None)
    if endpoint_router is None:
        raise HTTPException(status_code=404, detail="Endpoint draining is not available")

    draining = bool(payload.get("draining", True))
    try:
        await endpoint_router.set_draining(endpoint_name, draining)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown endpoint") from None

    endpoint_snapshot = await get_endpoint_snapshot(endpoint_name)

    return JSONResponse(
        {
            "status": "ok",
            "endpoint_name": endpoint_name,
            "draining": draining,
            "endpoint": endpoint_snapshot,
        }
    )


async def get_endpoint_snapshot(endpoint_name: str) -> dict[str, object]:
    endpoint_router = getattr(session_manager, "endpoint_router", None)
    if endpoint_router is None:
        raise HTTPException(status_code=404, detail="Endpoint status is not available")

    _, _, snapshot = await session_manager.healthcheck()
    router_snapshot = snapshot.get("router", {})
    endpoints = router_snapshot.get("endpoints", []) if isinstance(router_snapshot, dict) else []
    endpoint_snapshot = next(
        (
            endpoint
            for endpoint in endpoints
            if isinstance(endpoint, dict) and endpoint.get("name") == endpoint_name
        ),
        None,
    )
    if endpoint_snapshot is None:
        raise HTTPException(status_code=404, detail="Unknown endpoint")
    return endpoint_snapshot


def require_admin_auth(request: Request) -> None:
    if not LB_ADMIN_AUTH_TOKEN:
        raise HTTPException(status_code=503, detail="LB admin auth token is not configured")

    token = _bearer_token(request.headers.get("authorization"))
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Missing admin bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token, LB_ADMIN_AUTH_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid admin authorization")


def _bearer_token(authorization: str | None) -> str | None:
    scheme, separator, token = (authorization or "").partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None

    token = token.strip()
    if not token:
        return None
    return token


@app.websocket("/ws")
async def deprecated_websocket_route(client_ws: WebSocket):
    await client_ws.close(code=1008, reason="Use POST /session and connect directly to the returned compute websocket URL")


@app.get("/dashboard")
async def dashboard_page():
    return HTMLResponse(dashboard.html())


@app.get("/dashboard/data")
async def dashboard_data(window: str = "6h", resolution: str = ""):
    try:
        payload = await dashboard.data(window=window, resolution=resolution or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(payload)

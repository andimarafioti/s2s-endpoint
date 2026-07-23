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
from app.direct_session_manager import DirectSessionManager, QueueAtCapacityError
from app.endpoint_pool_router import (
    EndpointCapacityTimeoutError,
    EndpointDrainLeaseConflictError,
    EndpointPoolRouter,
    EndpointTransitionConflictError,
    HuggingFaceEndpointController,
    fetch_compute_active_sessions,
)
from app.requester_identity import (
    RequesterIdentity,
    RequesterIdentityResolver,
    bearer_token,
)
from app.requester_rate_limiter import (
    RateLimitDecision,
    RequesterRateLimitConfig,
    RequesterRateLimiter,
)
from app.session_request_metadata import reported_hardware_id
from app.session_requester_tracker import SessionRequesterTracker
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
COMPUTE_ENDPOINT_DRAIN_LEASE_TTL_S = float(os.getenv("COMPUTE_ENDPOINT_DRAIN_LEASE_TTL_S", "3600"))
COMPUTE_ENDPOINT_DRAIN_WARNING_AFTER_S = float(os.getenv("COMPUTE_ENDPOINT_DRAIN_WARNING_AFTER_S", "600"))
COMPUTE_ENDPOINT_DRAIN_WARNING_INTERVAL_S = float(os.getenv("COMPUTE_ENDPOINT_DRAIN_WARNING_INTERVAL_S", "300"))
HF_CONTROL_TOKEN = os.getenv("HF_CONTROL_TOKEN", "").strip() or os.getenv("HF_TOKEN", "").strip() or None
LB_ADMIN_AUTH_TOKEN = os.getenv("LB_ADMIN_AUTH_TOKEN", "").strip() or None

SESSION_SHARED_SECRET = os.getenv("SESSION_SHARED_SECRET", "").strip()
SESSION_PENDING_TIMEOUT_S = float(os.getenv("SESSION_PENDING_TIMEOUT_S", "60"))
SESSION_TOKEN_TTL_S = float(os.getenv("SESSION_TOKEN_TTL_S", "86400"))
SESSION_REAP_INTERVAL_S = float(os.getenv("SESSION_REAP_INTERVAL_S", "5"))
# Waiting queue: when every slot is busy a caller gets a ticket and polls
# GET /queue/{id} until the head of the line claims a freed slot. Waiting never
# reserves compute or usage time; an un-polled ticket is reaped after its TTL.
#
# The queue ships dark: with SESSION_QUEUE_ENABLED unset, /session behaves
# exactly as before the queue existed — the request blocks until a slot frees
# (up to COMPUTE_ENDPOINT_WAIT_TIMEOUT_S) and /queue/* returns 404. Set
# SESSION_QUEUE_ENABLED=true on an instance only once its clients understand
# {"state": "queued"} responses and poll GET /queue/{id}; a pre-queue client
# on a queueing instance would receive a 200 without a connect_url and fail.
SESSION_QUEUE_ENABLED = os.getenv("SESSION_QUEUE_ENABLED", "false").strip().lower() in {"true", "1", "yes"}
QUEUE_MAX_DEPTH = int(os.getenv("QUEUE_MAX_DEPTH", "100"))
QUEUE_TICKET_TTL_S = float(os.getenv("QUEUE_TICKET_TTL_S", "8"))
QUEUE_POLL_INTERVAL_S = float(os.getenv("QUEUE_POLL_INTERVAL_S", "2"))
QUEUE_REAP_INTERVAL_S = float(os.getenv("QUEUE_REAP_INTERVAL_S", "2"))
REQUEST_USAGE_HASH_SECRET = (
    os.getenv("REQUEST_USAGE_HASH_SECRET", "").strip()
    or SESSION_SHARED_SECRET
    or None
)
REQUEST_USAGE_TRUST_PROXY_HEADERS = os.getenv(
    "REQUEST_USAGE_TRUST_PROXY_HEADERS",
    "true",
).strip().lower() in {"true", "1", "yes"}
REQUEST_USAGE_MAX_ACTORS_PER_MINUTE = int(os.getenv("REQUEST_USAGE_MAX_ACTORS_PER_MINUTE", "1000"))
REQUEST_USAGE_MAX_RETAINED_RECORDS = int(os.getenv("REQUEST_USAGE_MAX_RETAINED_RECORDS", "50000"))
REQUEST_USAGE_MAX_PENDING_VALIDATIONS = int(os.getenv("REQUEST_USAGE_MAX_PENDING_VALIDATIONS", "128"))
REQUEST_USAGE_VALIDATION_CONCURRENCY = int(os.getenv("REQUEST_USAGE_VALIDATION_CONCURRENCY", "4"))
REQUEST_USAGE_HIGH_REQUESTS = int(os.getenv("REQUEST_USAGE_HIGH_REQUESTS", "100"))
REQUEST_USAGE_BURST_PER_MINUTE = int(os.getenv("REQUEST_USAGE_BURST_PER_MINUTE", "20"))
REQUEST_USAGE_MANY_NETWORKS = int(os.getenv("REQUEST_USAGE_MANY_NETWORKS", "5"))
REQUEST_RATE_LIMIT_ENABLED = os.getenv(
    "REQUEST_RATE_LIMIT_ENABLED",
    "true",
).strip().lower() in {"true", "1", "yes"}
REQUEST_RATE_LIMIT_WINDOW_S = float(os.getenv("REQUEST_RATE_LIMIT_WINDOW_S", "60"))
REQUEST_RATE_LIMIT_REQUESTS_PER_WINDOW = int(
    os.getenv("REQUEST_RATE_LIMIT_REQUESTS_PER_WINDOW", "20")
)
REQUEST_RATE_LIMIT_MAX_PARALLEL = int(os.getenv("REQUEST_RATE_LIMIT_MAX_PARALLEL", "10"))
REQUEST_RATE_LIMIT_NO_CONNECTS = int(os.getenv("REQUEST_RATE_LIMIT_NO_CONNECTS", "3"))
REQUEST_RATE_LIMIT_SHORT_SESSION_S = float(
    os.getenv("REQUEST_RATE_LIMIT_SHORT_SESSION_S", "10")
)
REQUEST_RATE_LIMIT_SHORT_SESSIONS = int(os.getenv("REQUEST_RATE_LIMIT_SHORT_SESSIONS", "8"))
REQUEST_RATE_LIMIT_COOLDOWN_S = float(os.getenv("REQUEST_RATE_LIMIT_COOLDOWN_S", "900"))
REQUEST_RATE_LIMIT_ACTOR_RETENTION_S = float(
    os.getenv("REQUEST_RATE_LIMIT_ACTOR_RETENTION_S", "3600")
)
REQUEST_RATE_LIMIT_MAX_ACTORS = int(os.getenv("REQUEST_RATE_LIMIT_MAX_ACTORS", "10000"))
DASHBOARD_SAMPLE_INTERVAL_S = float(os.getenv("DASHBOARD_SAMPLE_INTERVAL_S", "15"))
DASHBOARD_RETENTION_MINUTES = int(os.getenv("DASHBOARD_RETENTION_MINUTES", str(28 * 24 * 60)))
DASHBOARD_FLUSH_BATCH_SIZE = int(os.getenv("DASHBOARD_FLUSH_BATCH_SIZE", "100"))
DASHBOARD_FLUSH_TIMEOUT_S = float(os.getenv("DASHBOARD_FLUSH_TIMEOUT_S", "60"))
DASHBOARD_DIRTY_BUCKET_WARNING_AGE_S = float(os.getenv("DASHBOARD_DIRTY_BUCKET_WARNING_AGE_S", "300"))
DASHBOARD_STARTUP_MERGE_DELAY_S = float(os.getenv("DASHBOARD_STARTUP_MERGE_DELAY_S", "60"))
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
        drain_lease_ttl_s=COMPUTE_ENDPOINT_DRAIN_LEASE_TTL_S,
        drain_warning_after_s=COMPUTE_ENDPOINT_DRAIN_WARNING_AFTER_S,
        drain_warning_interval_s=COMPUTE_ENDPOINT_DRAIN_WARNING_INTERVAL_S,
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
    if SESSION_QUEUE_ENABLED:
        logger.warning("SESSION_QUEUE_ENABLED is ignored in dashboard preview mode")
else:
    session_manager = DirectSessionManager(
        endpoint_router=build_endpoint_router(),
        session_shared_secret=SESSION_SHARED_SECRET,
        pending_timeout_s=SESSION_PENDING_TIMEOUT_S,
        session_token_ttl_s=SESSION_TOKEN_TTL_S,
        reap_interval_s=SESSION_REAP_INTERVAL_S,
        queue_enabled=SESSION_QUEUE_ENABLED,
        queue_max_depth=QUEUE_MAX_DEPTH,
        queue_ticket_ttl_s=QUEUE_TICKET_TTL_S,
        queue_poll_interval_s=QUEUE_POLL_INTERVAL_S,
        queue_reap_interval_s=QUEUE_REAP_INTERVAL_S,
    )

dashboard_history_store = None
if DASHBOARD_BUCKET_ID:
    dashboard_history_store = HuggingFaceBucketHistoryStore(
        bucket_id=DASHBOARD_BUCKET_ID,
        prefix=DASHBOARD_BUCKET_PREFIX,
        token=DASHBOARD_BUCKET_TOKEN,
        request_timeout_s=DASHBOARD_FLUSH_TIMEOUT_S,
    )
    if DASHBOARD_PREVIEW_MODE:
        dashboard_history_store = ReadOnlyDashboardHistoryStore(dashboard_history_store)

dashboard = SwarmDashboard(
    snapshot_provider=session_manager.healthcheck,
    sample_interval_s=DASHBOARD_SAMPLE_INTERVAL_S,
    retention_minutes=DASHBOARD_RETENTION_MINUTES,
    history_store=dashboard_history_store,
    restore_history_in_background=True,
    flush_batch_size=DASHBOARD_FLUSH_BATCH_SIZE,
    flush_timeout_s=DASHBOARD_FLUSH_TIMEOUT_S,
    dirty_bucket_warning_age_s=DASHBOARD_DIRTY_BUCKET_WARNING_AGE_S,
    startup_merge_delay_s=DASHBOARD_STARTUP_MERGE_DELAY_S,
    max_requesters_per_bucket=REQUEST_USAGE_MAX_ACTORS_PER_MINUTE,
    max_requester_records=REQUEST_USAGE_MAX_RETAINED_RECORDS,
    requester_high_volume_threshold=REQUEST_USAGE_HIGH_REQUESTS,
    requester_burst_threshold_per_minute=REQUEST_USAGE_BURST_PER_MINUTE,
    requester_many_networks_threshold=REQUEST_USAGE_MANY_NETWORKS,
)
requester_identity_resolver = RequesterIdentityResolver(
    hash_secret=REQUEST_USAGE_HASH_SECRET,
    on_identity_update=dashboard.update_requester_identity,
    trust_proxy_headers=REQUEST_USAGE_TRUST_PROXY_HEADERS,
    max_pending_validations=REQUEST_USAGE_MAX_PENDING_VALIDATIONS,
    validation_concurrency=REQUEST_USAGE_VALIDATION_CONCURRENCY,
)
requester_rate_limiter = RequesterRateLimiter(
    config=RequesterRateLimitConfig(
        enabled=REQUEST_RATE_LIMIT_ENABLED,
        request_window_s=REQUEST_RATE_LIMIT_WINDOW_S,
        max_requests_per_window=REQUEST_RATE_LIMIT_REQUESTS_PER_WINDOW,
        max_parallel_allocations=REQUEST_RATE_LIMIT_MAX_PARALLEL,
        max_consecutive_no_connects=REQUEST_RATE_LIMIT_NO_CONNECTS,
        short_session_threshold_s=REQUEST_RATE_LIMIT_SHORT_SESSION_S,
        max_consecutive_short_sessions=REQUEST_RATE_LIMIT_SHORT_SESSIONS,
        cooldown_s=REQUEST_RATE_LIMIT_COOLDOWN_S,
        actor_retention_s=REQUEST_RATE_LIMIT_ACTOR_RETENTION_S,
        max_actor_states=REQUEST_RATE_LIMIT_MAX_ACTORS,
    )
)
session_requester_tracker = SessionRequesterTracker(
    retention_s=SESSION_PENDING_TIMEOUT_S + max(2 * SESSION_REAP_INTERVAL_S, 30.0),
)
# Queue polls are bodyless GETs, so the hardware-id identity resolved from the
# original POST /session can't be re-derived at claim time — carry it across the
# wait keyed by ticket. Refreshed on every poll; retention only has to outlive
# the ticket itself (un-polled tickets die at QUEUE_TICKET_TTL_S).
queue_requester_tracker = SessionRequesterTracker(
    retention_s=QUEUE_TICKET_TTL_S + max(2 * QUEUE_REAP_INTERVAL_S, 10.0),
)


async def record_abnormal_session_disconnect(result: dict[str, object]) -> None:
    session_id = str(result.get("session_id") or "")
    if session_id:
        outcome = requester_rate_limiter.record_disconnected(
            session_id,
            duration_s=_optional_float(result.get("conversation_duration_s")),
            penalize=False,
        )
        if outcome is not None and outcome.connected and outcome.duration_s is not None:
            await dashboard.record_requester_session_disconnected(
                outcome.requester,
                duration_s=outcome.duration_s,
                short_session=False,
            )
    await dashboard.record_session_event(
        "disconnected",
        conversation_duration_s=result.get("conversation_duration_s"),
        conversation_counted=bool(result.get("conversation_counted")),
    )


async def record_expired_queue_ticket(ticket_id: str) -> None:
    """Terminal outcome for a queued request whose ticket the reaper dropped:
    the waiter stopped polling, which is the queue's version of abandoning."""
    requester = queue_requester_tracker.take(ticket_id)
    if requester is not None:
        requester_rate_limiter.record_allocation_abandoned(requester)
    await dashboard.record_session_request_abandoned(requester)


class LoadBalancerRuntime:
    async def start(self) -> None:
        # Dashboard preview mode uses a synthetic manager with no real sessions.
        if hasattr(session_manager, "set_abnormal_disconnect_handler"):
            session_manager.set_abnormal_disconnect_handler(record_abnormal_session_disconnect)
        if hasattr(session_manager, "set_ticket_expired_handler"):
            session_manager.set_ticket_expired_handler(record_expired_queue_ticket)
        logger.info(
            "Session queue %s",
            "enabled" if session_manager.queue_enabled else "disabled",
        )
        await session_manager.start()
        await dashboard.start()

    async def stop(self) -> None:
        await requester_identity_resolver.stop()
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
    requester: RequesterIdentity | None = None,
    error: str | None = None,
    http_route: str = "POST /session",
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
        "http_route": http_route,
        "requester_id": requester.actor_id if requester is not None else None,
        "requester_kind": requester.kind if requester is not None else None,
        "requester_verification": requester.verification if requester is not None else None,
        "requester_network_id": requester.network_id if requester is not None else None,
        "requester_reported_robot_id": (
            requester.reported_robot_id if requester is not None else None
        ),
        "requester_client_kind": requester.client_kind if requester is not None else None,
    }
    message = (
        "Session allocation outcome outcome=%s session_id=%s endpoint_name=%s "
        "slot_id=%s allocation_wait_ms=%s allocation_total_ms=%d "
        "waited_for_capacity=%s requester_id=%s requester_kind=%s "
        "reported_robot_id=%s client_kind=%s"
    )
    args: list[object] = [
        outcome,
        session_id,
        endpoint_name,
        slot_id,
        allocation_wait_ms,
        allocation_total_ms,
        waited_for_capacity,
        requester.actor_id if requester is not None else None,
        requester.kind if requester is not None else None,
        requester.reported_robot_id if requester is not None else None,
        requester.client_kind if requester is not None else None,
    ]
    if error is not None:
        message += " error=%s"
        args.append(error)

    logger.log(level, message, *args, extra=extra)


def _log_rate_limit_rejection(
    decision: RateLimitDecision,
    *,
    requester: RequesterIdentity,
) -> None:
    extra = {
        "outcome": "rate_limited",
        "http_route": "POST /session",
        "rate_limit_reason": decision.reason,
        "retry_after_s": decision.retry_after_s,
        "recent_requests": decision.recent_requests,
        "active_allocations": decision.active_allocations,
        "consecutive_no_connects": decision.consecutive_no_connects,
        "consecutive_short_sessions": decision.consecutive_short_sessions,
        "requester_id": requester.actor_id,
        "requester_kind": requester.kind,
        "requester_verification": requester.verification,
        "requester_network_id": requester.network_id,
        "requester_reported_robot_id": requester.reported_robot_id,
        "requester_client_kind": requester.client_kind,
    }
    logger.warning(
        "Session request rate limited requester_id=%s requester_kind=%s "
        "reported_robot_id=%s client_kind=%s reason=%s retry_after_s=%s "
        "recent_requests=%d active_allocations=%d consecutive_no_connects=%d "
        "consecutive_short_sessions=%d",
        requester.actor_id,
        requester.kind,
        requester.reported_robot_id,
        requester.client_kind,
        decision.reason,
        decision.retry_after_s,
        decision.recent_requests,
        decision.active_allocations,
        decision.consecutive_no_connects,
        decision.consecutive_short_sessions,
        extra=extra,
    )


def _allocation_wait_ms(allocation: dict[str, object], *, fallback_ms: int) -> int:
    value = allocation.get("allocation_wait_ms")
    if value is None:
        return fallback_ms
    return max(int(value), 0)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _public_session_allocation(allocation: dict[str, object]) -> dict[str, object]:
    return {
        key: allocation[key]
        for key in (
            "session_id",
            "websocket_url",
            "connect_url",
            "session_token",
            "pending_timeout_s",
            "state",
        )
        if key in allocation
    }


async def _refresh_requester_identity(requester: RequesterIdentity) -> RequesterIdentity:
    latest = requester_identity_resolver.latest_identity(requester)
    if latest != requester:
        await dashboard.update_requester_identity(latest)
    return latest


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

    requester_tracking = requester_identity_resolver.status()
    requester_tracking["pending_session_attributions"] = session_requester_tracker.count()
    requester_tracking["rate_limit"] = requester_rate_limiter.status()
    return JSONResponse(
        {
            "status": "ok",
            "role": APP_ROLE,
            "compute_endpoints": COMPUTE_ENDPOINT_NAMES,
            "dashboard_preview_mode": DASHBOARD_PREVIEW_MODE,
            "dashboard_history": dashboard.persistence_status(),
            "requester_tracking": requester_tracking,
            "sessions": snapshot,
        }
    )


@app.post("/session")
async def create_session(request: Request):
    """Grant a session if a slot is free and the line is empty, otherwise return a
    queue ticket the caller polls via GET /queue/{id}. 503 with {state:"at_capacity"}
    when the queue itself is full; 503 otherwise when the pool can't allocate."""
    hardware_id = await reported_hardware_id(request)
    requester = requester_identity_resolver.identify(request, hardware_id=hardware_id)
    await dashboard.record_session_request(requester)
    requester = await _refresh_requester_identity(requester)
    rate_limit_decision = requester_rate_limiter.acquire(requester)
    if not rate_limit_decision.allowed:
        _log_rate_limit_rejection(rate_limit_decision, requester=requester)
        await dashboard.record_session_rate_limited(requester)
        retry_after_s = rate_limit_decision.retry_after_s or 1
        raise HTTPException(
            status_code=429,
            detail={
                "code": "requester_rate_limited",
                "reason": rate_limit_decision.reason,
                "retry_after_s": retry_after_s,
            },
            headers={"Retry-After": str(retry_after_s)},
        )
    allocation_started_at = monotonic()
    try:
        allocation = await session_manager.allocate(public_base_url(request))
    except QueueAtCapacityError as exc:
        requester_rate_limiter.record_allocation_failure(requester)
        requester = await _refresh_requester_identity(requester)
        allocation_total_ms = elapsed_ms(allocation_started_at, monotonic())
        _log_session_allocation_outcome(
            "queue_at_capacity",
            allocation=None,
            allocation_wait_ms=None,
            allocation_total_ms=allocation_total_ms,
            level=logging.WARNING,
            requester=requester,
            error=str(exc),
        )
        await dashboard.record_session_allocation_failure(requester)
        return JSONResponse({"state": "at_capacity", "detail": str(exc)}, status_code=503)
    except BaseException as exc:
        requester_rate_limiter.record_allocation_failure(requester)
        if not isinstance(exc, Exception):
            raise
        requester = await _refresh_requester_identity(requester)
        allocation_total_ms = elapsed_ms(allocation_started_at, monotonic())
        waited_for_capacity = isinstance(exc, EndpointCapacityTimeoutError)
        failure_allocation = {"waited_for_capacity": waited_for_capacity}
        _log_session_allocation_outcome(
            "allocation_failed",
            allocation=failure_allocation,
            allocation_wait_ms=allocation_total_ms if waited_for_capacity else None,
            allocation_total_ms=allocation_total_ms,
            level=logging.WARNING,
            requester=requester,
            error=str(exc),
        )
        await dashboard.record_session_allocation_failure(requester)
        raise HTTPException(status_code=503, detail=f"Failed to allocate compute endpoint: {exc}") from exc

    # No slot free (and/or others waiting): the caller joined the queue. Keep the
    # requester identity for the claim — queue polls are bodyless GETs that can't
    # re-derive it.
    if allocation.get("state") == "queued":
        queue_requester_tracker.remember(str(allocation["queue_id"]), requester)
        return JSONResponse(allocation)

    return await _deliver_grant(request, allocation, allocation_started_at, requester)


@app.get("/queue/{queue_id}")
async def queue_status(queue_id: str, request: Request):
    """Advance a waiting ticket: report position, or — for the head of the line —
    hand back a session grant once a slot frees. 404 for an unknown/expired ticket.
    404 for everything when the queue is disabled — indistinguishable from main,
    where these routes don't exist."""
    if not session_manager.queue_enabled:
        raise HTTPException(status_code=404, detail="Not found.")

    poll_started_at = monotonic()
    try:
        result = await session_manager.poll(queue_id, public_base_url(request))
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown or expired ticket.") from None
    except Exception as exc:
        # Same contract as POST /session: allocation-time failures are 503s, not
        # 500s. The manager re-queues the ticket at the head on a failed claim,
        # so the caller keeps its place and simply polls again.
        raise HTTPException(status_code=503, detail=f"Failed to claim session: {exc}") from exc

    requester = queue_requester_tracker.take(queue_id)
    if result.get("state") == "queued":
        if requester is not None:
            queue_requester_tracker.remember(queue_id, requester)  # refresh retention
        return JSONResponse(result)

    # Head of line claimed a slot — same delivery path as a fast-path grant. The
    # requester was resolved at ticket creation; falling back to the poll request
    # (bodyless, so IP-only) only happens if the tracker entry expired.
    if requester is None:
        hardware_id = await reported_hardware_id(request)
        requester = requester_identity_resolver.identify(request, hardware_id=hardware_id)
    return await _deliver_grant(
        request, result, poll_started_at, requester, http_route="GET /queue/{queue_id}"
    )


@app.delete("/queue/{queue_id}")
async def queue_leave(queue_id: str):
    """Leave the queue early (explicit button / teardown beacon). Idempotent."""
    if not session_manager.queue_enabled:
        raise HTTPException(status_code=404, detail="Not found.")
    left = await session_manager.leave(queue_id)
    requester = queue_requester_tracker.take(queue_id)
    if left:
        # Terminal outcome for the queued request: leaving the line is the queue's
        # version of abandoning before delivery.
        if requester is not None:
            requester_rate_limiter.record_allocation_abandoned(requester)
        await dashboard.record_session_request_abandoned(requester)
    return JSONResponse({"status": "ok", "state": "left", "removed": left})


async def _deliver_grant(
    request: Request,
    allocation: dict[str, object],
    started_at: float,
    requester: RequesterIdentity,
    *,
    http_route: str = "POST /session",
) -> JSONResponse:
    """Shared tail for a granted session (fast path or queue claim): guard against a
    client that vanished, record the success, and return the public grant fields."""
    allocation_total_ms = elapsed_ms(started_at, monotonic())
    allocation_wait_ms = _allocation_wait_ms(allocation, fallback_ms=allocation_total_ms)
    allocation.setdefault("allocation_wait_ms", allocation_wait_ms)
    session_id = str(allocation.get("session_id") or "")
    if session_id:
        requester_rate_limiter.record_allocation(
            session_id,
            requester,
            pending_timeout_s=float(
                allocation.get("pending_timeout_s") or SESSION_PENDING_TIMEOUT_S
            ),
        )
    else:
        requester_rate_limiter.record_allocation_failure(requester)

    if await request.is_disconnected():
        requester = await _refresh_requester_identity(requester)
        if session_id and hasattr(session_manager, "cancel_pending_session"):
            await session_manager.cancel_pending_session(session_id)
        if session_id:
            requester_rate_limiter.record_disconnected(session_id)
        _log_session_allocation_outcome(
            "client_disconnected",
            allocation=allocation,
            allocation_wait_ms=allocation_wait_ms,
            allocation_total_ms=allocation_total_ms,
            level=logging.WARNING,
            requester=requester,
            http_route=http_route,
        )
        await dashboard.record_session_request_abandoned(requester)
        raise HTTPException(status_code=503, detail="Client disconnected before session could be delivered")

    requester = await _refresh_requester_identity(requester)
    if session_id:
        session_requester_tracker.remember(session_id, requester)
    await dashboard.record_session_allocation_success(requester)
    _log_session_allocation_outcome(
        "success",
        allocation=allocation,
        allocation_wait_ms=allocation_wait_ms,
        allocation_total_ms=allocation_total_ms,
        level=logging.INFO,
        requester=requester,
        http_route=http_route,
    )
    # "state": "granted" rides along from the manager's grant dict through the
    # public-field whitelist — single-sourced there, not re-asserted here.
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
            requester_rate_limiter.record_disconnected(session_id)
            session_requester_tracker.discard(session_id)
            return JSONResponse({"status": "ok", "session_id": session_id, "state": "already_released"})
        raise HTTPException(status_code=404, detail="Unknown session id") from None
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    await dashboard.record_session_event(
        event,
        conversation_duration_s=result.get("conversation_duration_s"),
        conversation_counted=bool(result.get("conversation_counted")),
    )
    if event == "connected":
        requester_rate_limiter.record_connected(session_id)
        requester = session_requester_tracker.take(session_id)
        if requester is not None:
            requester = await _refresh_requester_identity(requester)
            await dashboard.record_requester_session_connected(requester)
    elif event == "disconnected":
        outcome = requester_rate_limiter.record_disconnected(
            session_id,
            duration_s=_optional_float(result.get("conversation_duration_s")),
            penalize=(
                bool(result.get("conversation_counted"))
                and result.get("release_reason") != "endpoint_unavailable"
            ),
        )
        if outcome is not None and outcome.connected and outcome.duration_s is not None:
            await dashboard.record_requester_session_disconnected(
                outcome.requester,
                duration_s=outcome.duration_s,
                short_session=outcome.short_session,
            )
        session_requester_tracker.discard(session_id)
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
        raise HTTPException(status_code=503, detail="Endpoint draining is not available")

    endpoint_snapshot = await get_endpoint_snapshot(endpoint_name)
    draining = payload.get("draining", True)
    if type(draining) is not bool:
        raise HTTPException(status_code=422, detail="draining must be a boolean")
    lease_ttl_s = payload.get("lease_ttl_s")
    if lease_ttl_s is not None and (
        type(lease_ttl_s) not in (int, float) or lease_ttl_s <= 0
    ):
        raise HTTPException(status_code=422, detail="lease_ttl_s must be a positive number")
    lease_id = payload.get("lease_id")
    force = payload.get("force", False)
    if type(force) is not bool:
        raise HTTPException(status_code=422, detail="force must be a boolean")
    if draining and force:
        raise HTTPException(status_code=422, detail="force is only valid when clearing a drain")
    if not force and (not isinstance(lease_id, str) or not lease_id.strip()):
        raise HTTPException(
            status_code=422,
            detail="lease_id is required unless force-clearing a drain",
        )
    try:
        await endpoint_router.set_draining(
            endpoint_name,
            draining,
            lease_ttl_s=float(lease_ttl_s) if lease_ttl_s is not None else None,
            lease_id=lease_id.strip() if isinstance(lease_id, str) else None,
            force=force,
        )
    except KeyError:
        raise HTTPException(status_code=503, detail="Endpoint became unavailable") from None
    except EndpointTransitionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EndpointDrainLeaseConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    endpoint_snapshot = await get_endpoint_snapshot(endpoint_name)

    return JSONResponse(
        {
            "status": "ok",
            "endpoint_name": endpoint_name,
            "draining": endpoint_snapshot.get("draining"),
            "endpoint": endpoint_snapshot,
        }
    )


async def get_endpoint_snapshot(endpoint_name: str) -> dict[str, object]:
    endpoint_router = getattr(session_manager, "endpoint_router", None)
    if endpoint_router is None:
        raise HTTPException(status_code=503, detail="Endpoint status is not available")

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
    return bearer_token(authorization)


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

import json
import logging
import secrets
from dataclasses import dataclass, field, replace
from inspect import cleandoc
from time import monotonic
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse

from app.app_utils import (
    build_lifespan,
    elapsed_ms,
    env_bool,
    env_optional,
    env_text,
    public_base_url,
    setup_logging,
)
from app.dashboard_history_store import HuggingFaceBucketHistoryStore, ReadOnlyDashboardHistoryStore
from app.dashboard_preview import DashboardPreviewSessionManager
from app.direct_session_manager import DirectSessionManager, QueueAtCapacityError
from app.endpoint_pool_router import (
    EndpointCapacityTimeoutError,
    EndpointDrainLeaseConflictError,
    EndpointPoolRouter,
    EndpointTransitionConflictError,
    HuggingFaceEndpointController,
    fetch_compute_usage,
)
from app.llm_proxy_usage import (
    LLM_PROXY_CALLBACK_AUTH_HEADER,
    LLM_PROXY_CALLBACK_BODY_MAX_BYTES,
    LLM_PROXY_CLIENT_IP_MAX_LENGTH,
    LLM_PROXY_REASONS,
)
from app.pipeline_capacity import PipelineCapacity, PipelineCapacityConfig
from app.requester_identity import (
    RequesterIdentity,
    RequesterIdentityResolver,
    bearer_token,
    is_validatable_hf_token,
    normalize_hardware_id,
)
from app.requester_rate_limiter import (
    RateLimitDecision,
    RequesterRateLimitConfig,
    RequesterRateLimiter,
)
from app.session_manager import SessionManager
from app.session_request_metadata import reported_hardware_id, session_metadata
from app.session_requester_tracker import SessionRequesterTracker
from app.session_tokens import llm_token_fingerprint
from app.speech_proxy_telemetry import SpeechProxyTelemetryClient, SpeechProxyTelemetryTarget
from app.swarm_dashboard import SwarmDashboard
from app.verification_admission_limiter import (
    VerificationAdmissionConfig,
    VerificationAdmissionLimiter,
)

logger = setup_logging()
APP_ROLE = "load_balancer"
DASHBOARD_PREVIEW_SENTINELS = {"test", "preview", "dashboard_preview"}
LB_ADMIN_AUTH_HEADER = "X-Reachy-Mini-Admin-Authorization"


@dataclass(frozen=True)
class LoadBalancerSettings:
    hf_endpoint_namespace: str | None = None
    compute_endpoint_names: tuple[str, ...] = ()
    compute_endpoint_min_warm: int = 1
    compute_endpoint_wake_threshold_slots: int = 1
    compute_endpoint_idle_park_timeout_s: float = 600.0
    compute_endpoint_reconcile_interval_s: float = 10.0
    compute_endpoint_waking_capacity_timeout_s: float = 300.0
    compute_endpoint_control_fetch_timeout_s: float = 30.0
    compute_endpoint_http_timeout_s: float = 10.0
    compute_endpoint_reconcile_stale_after_s: float | None = None
    compute_endpoint_park_cooldown_s: float = 180.0
    compute_endpoint_wait_timeout_s: int = 900
    compute_endpoint_control_operation_timeout_s: float | None = None
    compute_endpoint_park_strategy: str = "pause"
    compute_endpoint_auto_restart: bool = True
    compute_endpoint_max_restart_attempts: int = 3
    compute_endpoint_restart_backoff_s: float = 30.0
    compute_endpoint_restart_backoff_max_s: float = 300.0
    compute_endpoint_restart_stable_running_s: float = 120.0
    compute_endpoint_drain_restart_timeout_s: float = 600.0
    compute_endpoint_drain_lease_ttl_s: float = 3600.0
    compute_endpoint_drain_warning_after_s: float = 600.0
    compute_endpoint_drain_warning_interval_s: float = 300.0
    compute_usage_stale_ttl_s: float = 60.0
    hf_control_token: str | None = None
    lb_admin_auth_token: str | None = None
    lb_callback_auth_token: str | None = None
    session_shared_secret: str = ""
    session_require_verified_hf_token: bool = False
    session_hf_token_verify_timeout_s: float = 5.0
    session_hf_token_max_verified_age_s: float = 1800.0
    llm_proxy_claim_verify_timeout_s: float = 5.0
    session_pending_timeout_s: float = 60.0
    session_token_ttl_s: float = 86400.0
    session_reap_interval_s: float = 5.0
    session_queue_enabled: bool = False
    queue_max_depth: int = 100
    queue_ticket_ttl_s: float = 8.0
    queue_poll_interval_s: float = 2.0
    queue_reap_interval_s: float = 2.0
    request_usage_hash_secret: str | None = None
    request_usage_trust_proxy_headers: bool = True
    request_usage_max_actors_per_minute: int = 1000
    request_usage_max_retained_records: int = 50000
    request_usage_max_pending_validations: int = 128
    request_usage_validation_concurrency: int = 4
    session_hf_token_verify_max_pending: int = 64
    session_hf_token_verify_max_pending_per_network: int = 4
    request_usage_high_requests: int = 100
    request_usage_burst_per_minute: int = 20
    request_usage_many_networks: int = 5
    request_rate_limit_enabled: bool = True
    request_rate_limit_window_s: float = 60.0
    request_rate_limit_requests_per_window: int = 20
    request_rate_limit_max_parallel: int = 10
    request_rate_limit_no_connects: int = 5
    request_rate_limit_short_session_s: float = 10.0
    request_rate_limit_short_sessions: int = 8
    request_rate_limit_cooldown_s: float = 900.0
    request_rate_limit_actor_retention_s: float = 3600.0
    request_rate_limit_max_actors: int = 10000
    dashboard_sample_interval_s: float = 15.0
    dashboard_retention_minutes: int = 28 * 24 * 60
    dashboard_flush_batch_size: int = 100
    dashboard_flush_timeout_s: float = 60.0
    dashboard_dirty_bucket_warning_age_s: float = 300.0
    dashboard_startup_merge_delay_s: float = 60.0
    dashboard_bucket_id: str | None = None
    dashboard_bucket_prefix: str = "s2s-endpoint/swarm-dashboard"
    dashboard_bucket_token: str | None = None
    dashboard_preview_mode: bool = False
    speech_stt_proxy_url: str | None = None
    speech_tts_proxy_url: str | None = None
    speech_llm_proxy_url: str | None = None
    speech_proxy_api_key: str | None = None
    speech_proxy_metrics_timeout_s: float = 5.0
    pipeline_capacity: PipelineCapacityConfig | None = None
    speech_capacity_api_key: str | None = field(default=None, repr=False)
    speech_capacity_ingress_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.session_hf_token_verify_timeout_s <= 0:
            raise ValueError("SESSION_HF_TOKEN_VERIFY_TIMEOUT_S must be > 0")
        if self.session_hf_token_max_verified_age_s <= 0:
            raise ValueError("SESSION_HF_TOKEN_MAX_VERIFIED_AGE_S must be > 0")
        if self.dashboard_preview_mode and not self.compute_endpoint_names:
            object.__setattr__(
                self,
                "compute_endpoint_names",
                tuple(f"preview-compute-{index:02d}" for index in range(1, 5)),
            )
        if self.compute_endpoint_control_operation_timeout_s is None:
            object.__setattr__(
                self,
                "compute_endpoint_control_operation_timeout_s",
                float(self.compute_endpoint_wait_timeout_s),
            )
        if self.compute_endpoint_reconcile_stale_after_s is None:
            object.__setattr__(
                self,
                "compute_endpoint_reconcile_stale_after_s",
                max(
                    self.compute_endpoint_reconcile_interval_s * 3,
                    self.compute_endpoint_control_fetch_timeout_s * 2,
                ),
            )
        if self.request_usage_hash_secret is None and self.session_shared_secret:
            object.__setattr__(self, "request_usage_hash_secret", self.session_shared_secret)
        if self.dashboard_bucket_token is None and self.hf_control_token:
            object.__setattr__(self, "dashboard_bucket_token", self.hf_control_token)
        if self.speech_proxy_api_key is None and self.hf_control_token:
            object.__setattr__(self, "speech_proxy_api_key", self.hf_control_token)
        if self.speech_proxy_metrics_timeout_s <= 0:
            raise ValueError("SPEECH_PROXY_METRICS_TIMEOUT_S must be > 0")
        if self.pipeline_capacity is not None:
            if not self.session_queue_enabled:
                raise ValueError("PIPELINE_CAPACITY requires SESSION_QUEUE_ENABLED")
            if not all(
                (
                    self.speech_stt_proxy_url,
                    self.speech_llm_proxy_url,
                    self.speech_tts_proxy_url,
                    self.speech_capacity_api_key,
                )
            ):
                raise ValueError("PIPELINE_CAPACITY requires all gateway URLs and SPEECH_CAPACITY_API_KEY")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "LoadBalancerSettings":
        names = tuple(
            name.strip() for name in env_text("COMPUTE_ENDPOINT_NAMES", environ=environ).split(",") if name.strip()
        )
        preview_mode = env_bool("DASHBOARD_PREVIEW_MODE", False, environ=environ)
        if len(names) == 1 and names[0].lower() in DASHBOARD_PREVIEW_SENTINELS:
            preview_mode = True
            names = ()
        reconcile_interval_s = float(env_text("COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S", "10", environ=environ))
        control_fetch_timeout_s = float(env_text("COMPUTE_ENDPOINT_CONTROL_FETCH_TIMEOUT_S", "30", environ=environ))
        wait_timeout_s = int(env_text("COMPUTE_ENDPOINT_WAIT_TIMEOUT_S", "900", environ=environ))
        hf_control_token = env_optional("HF_CONTROL_TOKEN", environ=environ)
        if hf_control_token is None:
            hf_control_token = env_optional("HF_TOKEN", environ=environ)
        session_shared_secret = env_text("SESSION_SHARED_SECRET", environ=environ)
        reconcile_stale_after = env_optional("COMPUTE_ENDPOINT_RECONCILE_STALE_AFTER_S", environ=environ)
        control_operation_timeout = env_optional(
            "COMPUTE_ENDPOINT_CONTROL_OPERATION_TIMEOUT_S",
            environ=environ,
        )

        return cls(
            hf_endpoint_namespace=env_optional("HF_ENDPOINT_NAMESPACE", environ=environ),
            compute_endpoint_names=names,
            compute_endpoint_min_warm=int(env_text("COMPUTE_ENDPOINT_MIN_WARM", "1", environ=environ)),
            compute_endpoint_wake_threshold_slots=int(
                env_text("COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS", "1", environ=environ)
            ),
            compute_endpoint_idle_park_timeout_s=float(
                env_text("COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S", "600", environ=environ)
            ),
            compute_endpoint_reconcile_interval_s=reconcile_interval_s,
            compute_endpoint_waking_capacity_timeout_s=float(
                env_text("COMPUTE_ENDPOINT_WAKING_CAPACITY_TIMEOUT_S", "300", environ=environ)
            ),
            compute_endpoint_control_fetch_timeout_s=control_fetch_timeout_s,
            compute_endpoint_http_timeout_s=float(env_text("COMPUTE_ENDPOINT_HTTP_TIMEOUT_S", "10", environ=environ)),
            compute_endpoint_reconcile_stale_after_s=(
                float(reconcile_stale_after) if reconcile_stale_after is not None else None
            ),
            compute_endpoint_park_cooldown_s=float(
                env_text("COMPUTE_ENDPOINT_PARK_COOLDOWN_S", "180", environ=environ)
            ),
            compute_endpoint_wait_timeout_s=wait_timeout_s,
            compute_endpoint_control_operation_timeout_s=(
                float(control_operation_timeout) if control_operation_timeout is not None else None
            ),
            compute_endpoint_park_strategy=env_text("COMPUTE_ENDPOINT_PARK_STRATEGY", "pause", environ=environ).lower(),
            compute_endpoint_auto_restart=env_bool("COMPUTE_ENDPOINT_AUTO_RESTART", True, environ=environ),
            compute_endpoint_max_restart_attempts=int(
                env_text("COMPUTE_ENDPOINT_MAX_RESTART_ATTEMPTS", "3", environ=environ)
            ),
            compute_endpoint_restart_backoff_s=float(
                env_text("COMPUTE_ENDPOINT_RESTART_BACKOFF_S", "30", environ=environ)
            ),
            compute_endpoint_restart_backoff_max_s=float(
                env_text("COMPUTE_ENDPOINT_RESTART_BACKOFF_MAX_S", "300", environ=environ)
            ),
            compute_endpoint_restart_stable_running_s=float(
                env_text("COMPUTE_ENDPOINT_RESTART_STABLE_RUNNING_S", "120", environ=environ)
            ),
            compute_endpoint_drain_restart_timeout_s=float(
                env_text("COMPUTE_ENDPOINT_DRAIN_RESTART_TIMEOUT_S", "600", environ=environ)
            ),
            compute_endpoint_drain_lease_ttl_s=float(
                env_text("COMPUTE_ENDPOINT_DRAIN_LEASE_TTL_S", "3600", environ=environ)
            ),
            compute_endpoint_drain_warning_after_s=float(
                env_text("COMPUTE_ENDPOINT_DRAIN_WARNING_AFTER_S", "600", environ=environ)
            ),
            compute_endpoint_drain_warning_interval_s=float(
                env_text("COMPUTE_ENDPOINT_DRAIN_WARNING_INTERVAL_S", "300", environ=environ)
            ),
            compute_usage_stale_ttl_s=float(env_text("COMPUTE_USAGE_STALE_TTL_S", "60", environ=environ)),
            hf_control_token=hf_control_token,
            lb_admin_auth_token=env_optional("LB_ADMIN_AUTH_TOKEN", environ=environ),
            lb_callback_auth_token=env_optional("LB_CALLBACK_AUTH_TOKEN", environ=environ),
            session_shared_secret=session_shared_secret,
            session_require_verified_hf_token=env_bool("SESSION_REQUIRE_VERIFIED_HF_TOKEN", False, environ=environ),
            session_hf_token_verify_timeout_s=float(
                env_text("SESSION_HF_TOKEN_VERIFY_TIMEOUT_S", "5", environ=environ)
            ),
            session_hf_token_max_verified_age_s=float(
                env_text("SESSION_HF_TOKEN_MAX_VERIFIED_AGE_S", "1800", environ=environ)
            ),
            llm_proxy_claim_verify_timeout_s=float(env_text("LLM_PROXY_CLAIM_VERIFY_TIMEOUT_S", "5", environ=environ)),
            session_pending_timeout_s=float(env_text("SESSION_PENDING_TIMEOUT_S", "60", environ=environ)),
            session_token_ttl_s=float(env_text("SESSION_TOKEN_TTL_S", "86400", environ=environ)),
            session_reap_interval_s=float(env_text("SESSION_REAP_INTERVAL_S", "5", environ=environ)),
            session_queue_enabled=env_bool("SESSION_QUEUE_ENABLED", False, environ=environ),
            queue_max_depth=int(env_text("QUEUE_MAX_DEPTH", "100", environ=environ)),
            queue_ticket_ttl_s=float(env_text("QUEUE_TICKET_TTL_S", "8", environ=environ)),
            queue_poll_interval_s=float(env_text("QUEUE_POLL_INTERVAL_S", "2", environ=environ)),
            queue_reap_interval_s=float(env_text("QUEUE_REAP_INTERVAL_S", "2", environ=environ)),
            request_usage_hash_secret=env_optional("REQUEST_USAGE_HASH_SECRET", environ=environ),
            request_usage_trust_proxy_headers=env_bool("REQUEST_USAGE_TRUST_PROXY_HEADERS", True, environ=environ),
            request_usage_max_actors_per_minute=int(
                env_text("REQUEST_USAGE_MAX_ACTORS_PER_MINUTE", "1000", environ=environ)
            ),
            request_usage_max_retained_records=int(
                env_text("REQUEST_USAGE_MAX_RETAINED_RECORDS", "50000", environ=environ)
            ),
            request_usage_max_pending_validations=int(
                env_text("REQUEST_USAGE_MAX_PENDING_VALIDATIONS", "128", environ=environ)
            ),
            request_usage_validation_concurrency=int(
                env_text("REQUEST_USAGE_VALIDATION_CONCURRENCY", "4", environ=environ)
            ),
            session_hf_token_verify_max_pending=int(
                env_text("SESSION_HF_TOKEN_VERIFY_MAX_PENDING", "64", environ=environ)
            ),
            session_hf_token_verify_max_pending_per_network=int(
                env_text("SESSION_HF_TOKEN_VERIFY_MAX_PENDING_PER_NETWORK", "4", environ=environ)
            ),
            request_usage_high_requests=int(env_text("REQUEST_USAGE_HIGH_REQUESTS", "100", environ=environ)),
            request_usage_burst_per_minute=int(env_text("REQUEST_USAGE_BURST_PER_MINUTE", "20", environ=environ)),
            request_usage_many_networks=int(env_text("REQUEST_USAGE_MANY_NETWORKS", "5", environ=environ)),
            request_rate_limit_enabled=env_bool("REQUEST_RATE_LIMIT_ENABLED", True, environ=environ),
            request_rate_limit_window_s=float(env_text("REQUEST_RATE_LIMIT_WINDOW_S", "60", environ=environ)),
            request_rate_limit_requests_per_window=int(
                env_text("REQUEST_RATE_LIMIT_REQUESTS_PER_WINDOW", "20", environ=environ)
            ),
            request_rate_limit_max_parallel=int(env_text("REQUEST_RATE_LIMIT_MAX_PARALLEL", "10", environ=environ)),
            request_rate_limit_no_connects=int(env_text("REQUEST_RATE_LIMIT_NO_CONNECTS", "5", environ=environ)),
            request_rate_limit_short_session_s=float(
                env_text("REQUEST_RATE_LIMIT_SHORT_SESSION_S", "10", environ=environ)
            ),
            request_rate_limit_short_sessions=int(env_text("REQUEST_RATE_LIMIT_SHORT_SESSIONS", "8", environ=environ)),
            request_rate_limit_cooldown_s=float(env_text("REQUEST_RATE_LIMIT_COOLDOWN_S", "900", environ=environ)),
            request_rate_limit_actor_retention_s=float(
                env_text("REQUEST_RATE_LIMIT_ACTOR_RETENTION_S", "3600", environ=environ)
            ),
            request_rate_limit_max_actors=int(env_text("REQUEST_RATE_LIMIT_MAX_ACTORS", "10000", environ=environ)),
            dashboard_sample_interval_s=float(env_text("DASHBOARD_SAMPLE_INTERVAL_S", "15", environ=environ)),
            dashboard_retention_minutes=int(
                env_text("DASHBOARD_RETENTION_MINUTES", str(28 * 24 * 60), environ=environ)
            ),
            dashboard_flush_batch_size=int(env_text("DASHBOARD_FLUSH_BATCH_SIZE", "100", environ=environ)),
            dashboard_flush_timeout_s=float(env_text("DASHBOARD_FLUSH_TIMEOUT_S", "60", environ=environ)),
            dashboard_dirty_bucket_warning_age_s=float(
                env_text("DASHBOARD_DIRTY_BUCKET_WARNING_AGE_S", "300", environ=environ)
            ),
            dashboard_startup_merge_delay_s=float(env_text("DASHBOARD_STARTUP_MERGE_DELAY_S", "60", environ=environ)),
            dashboard_bucket_id=env_optional("DASHBOARD_BUCKET_ID", environ=environ),
            dashboard_bucket_prefix=env_text(
                "DASHBOARD_BUCKET_PREFIX", "s2s-endpoint/swarm-dashboard", environ=environ
            ),
            dashboard_bucket_token=env_optional("DASHBOARD_BUCKET_TOKEN", environ=environ),
            dashboard_preview_mode=preview_mode,
            speech_stt_proxy_url=env_optional("SPEECH_STT_PROXY_URL", environ=environ),
            speech_tts_proxy_url=env_optional("SPEECH_TTS_PROXY_URL", environ=environ),
            speech_llm_proxy_url=env_optional("SPEECH_LLM_PROXY_URL", environ=environ),
            speech_proxy_api_key=env_optional("SPEECH_PROXY_API_KEY", environ=environ),
            speech_proxy_metrics_timeout_s=float(env_text("SPEECH_PROXY_METRICS_TIMEOUT_S", "5", environ=environ)),
            pipeline_capacity=PipelineCapacityConfig.model_validate_json(env_text("PIPELINE_CAPACITY", environ=environ))
            if env_optional("PIPELINE_CAPACITY", environ=environ)
            else None,
            speech_capacity_api_key=env_optional("SPEECH_CAPACITY_API_KEY", environ=environ),
            speech_capacity_ingress_api_key=env_optional("SPEECH_CAPACITY_INGRESS_API_KEY", environ=environ),
        )


def build_endpoint_router(settings: LoadBalancerSettings) -> EndpointPoolRouter:
    if not settings.compute_endpoint_names:
        raise RuntimeError("COMPUTE_ENDPOINT_NAMES must be set for the load-balancer app")

    controller = HuggingFaceEndpointController(
        namespace=settings.hf_endpoint_namespace,
        token=settings.hf_control_token,
        wait_timeout_s=settings.compute_endpoint_control_operation_timeout_s,
        active_min_replica=1,
        active_max_replica=1,
        park_strategy=settings.compute_endpoint_park_strategy,
        http_timeout_s=settings.compute_endpoint_http_timeout_s,
    )

    return EndpointPoolRouter(
        endpoint_names=settings.compute_endpoint_names,
        min_warm_endpoints=settings.compute_endpoint_min_warm,
        wake_threshold_slots=settings.compute_endpoint_wake_threshold_slots,
        idle_park_timeout_s=settings.compute_endpoint_idle_park_timeout_s,
        reconcile_interval_s=settings.compute_endpoint_reconcile_interval_s,
        waking_capacity_timeout_s=settings.compute_endpoint_waking_capacity_timeout_s,
        park_cooldown_s=settings.compute_endpoint_park_cooldown_s,
        controller=controller,
        auto_restart=settings.compute_endpoint_auto_restart,
        max_restart_attempts=settings.compute_endpoint_max_restart_attempts,
        restart_backoff_s=settings.compute_endpoint_restart_backoff_s,
        restart_backoff_max_s=settings.compute_endpoint_restart_backoff_max_s,
        restart_stable_running_s=settings.compute_endpoint_restart_stable_running_s,
        drain_restart_timeout_s=settings.compute_endpoint_drain_restart_timeout_s,
        drain_lease_ttl_s=settings.compute_endpoint_drain_lease_ttl_s,
        drain_warning_after_s=settings.compute_endpoint_drain_warning_after_s,
        drain_warning_interval_s=settings.compute_endpoint_drain_warning_interval_s,
        compute_usage_fetcher=lambda url: fetch_compute_usage(url, api_key=settings.hf_control_token),
        # How long a previously observed usage count stays trusted when
        # health polls fail transiently. Must be comfortably above the
        # reconcile interval (10s): the default 60s means roughly six
        # consecutive failed polls before a synced node loses capacity.
        # Setting it below the reconcile interval revokes on a single blip.
        usage_sync_stale_ttl_s=settings.compute_usage_stale_ttl_s,
        control_fetch_timeout_s=settings.compute_endpoint_control_fetch_timeout_s,
        reconcile_stale_after_s=settings.compute_endpoint_reconcile_stale_after_s,
        pipeline_capacity=PipelineCapacity(
            settings.pipeline_capacity,
            {
                "stt": settings.speech_stt_proxy_url,
                "llm": settings.speech_llm_proxy_url,
                "tts": settings.speech_tts_proxy_url,
            },
            settings.speech_capacity_api_key,
            ingress_api_key=settings.speech_capacity_ingress_api_key,
        )
        if settings.pipeline_capacity
        else None,
    )


@dataclass
class LoadBalancerDependencies:
    session_manager: SessionManager
    dashboard_history_store: Any | None
    dashboard: SwarmDashboard
    requester_identity_resolver: RequesterIdentityResolver
    session_verification_limiter: VerificationAdmissionLimiter
    requester_rate_limiter: RequesterRateLimiter
    session_requester_tracker: SessionRequesterTracker
    queue_requester_tracker: SessionRequesterTracker
    speech_proxy_telemetry: SpeechProxyTelemetryClient | None = None


def build_load_balancer_dependencies(settings: LoadBalancerSettings) -> LoadBalancerDependencies:
    if settings.dashboard_preview_mode:
        session_manager = DashboardPreviewSessionManager()
        if settings.session_queue_enabled:
            logger.warning("SESSION_QUEUE_ENABLED is ignored in dashboard preview mode")
    else:
        session_manager = DirectSessionManager(
            endpoint_router=build_endpoint_router(settings),
            session_shared_secret=settings.session_shared_secret,
            pending_timeout_s=settings.session_pending_timeout_s,
            session_token_ttl_s=settings.session_token_ttl_s,
            reap_interval_s=settings.session_reap_interval_s,
            queue_enabled=settings.session_queue_enabled,
            queue_max_depth=settings.queue_max_depth,
            queue_ticket_ttl_s=settings.queue_ticket_ttl_s,
            queue_poll_interval_s=settings.queue_poll_interval_s,
            queue_reap_interval_s=settings.queue_reap_interval_s,
        )

    dashboard_history_store = None
    if settings.dashboard_bucket_id:
        dashboard_history_store = HuggingFaceBucketHistoryStore(
            bucket_id=settings.dashboard_bucket_id,
            prefix=settings.dashboard_bucket_prefix,
            token=settings.dashboard_bucket_token,
            request_timeout_s=settings.dashboard_flush_timeout_s,
        )
        if settings.dashboard_preview_mode:
            dashboard_history_store = ReadOnlyDashboardHistoryStore(dashboard_history_store)

    speech_targets = tuple(
        SpeechProxyTelemetryTarget(service=service, url=url.rstrip("/"))
        for service, url in (
            ("stt", settings.speech_stt_proxy_url),
            ("tts", settings.speech_tts_proxy_url),
            ("llm", settings.speech_llm_proxy_url),
        )
        if url
    )
    speech_proxy_telemetry = (
        SpeechProxyTelemetryClient(
            speech_targets,
            api_key=settings.speech_proxy_api_key,
            timeout_s=settings.speech_proxy_metrics_timeout_s,
        )
        if speech_targets
        else None
    )

    dashboard = SwarmDashboard(
        snapshot_provider=session_manager.healthcheck,
        speech_telemetry_provider=(speech_proxy_telemetry.snapshot if speech_proxy_telemetry else None),
        sample_interval_s=settings.dashboard_sample_interval_s,
        retention_minutes=settings.dashboard_retention_minutes,
        history_store=dashboard_history_store,
        restore_history_in_background=True,
        flush_batch_size=settings.dashboard_flush_batch_size,
        flush_timeout_s=settings.dashboard_flush_timeout_s,
        dirty_bucket_warning_age_s=settings.dashboard_dirty_bucket_warning_age_s,
        startup_merge_delay_s=settings.dashboard_startup_merge_delay_s,
        max_requesters_per_bucket=settings.request_usage_max_actors_per_minute,
        max_requester_records=settings.request_usage_max_retained_records,
        requester_high_volume_threshold=settings.request_usage_high_requests,
        requester_burst_threshold_per_minute=settings.request_usage_burst_per_minute,
        requester_many_networks_threshold=settings.request_usage_many_networks,
    )
    requester_identity_resolver = RequesterIdentityResolver(
        hash_secret=settings.request_usage_hash_secret,
        on_identity_update=dashboard.update_requester_identity,
        trust_proxy_headers=settings.request_usage_trust_proxy_headers,
        max_pending_validations=settings.request_usage_max_pending_validations,
        validation_concurrency=settings.request_usage_validation_concurrency,
    )
    session_verification_limiter = VerificationAdmissionLimiter(
        config=VerificationAdmissionConfig(
            max_global_pending=settings.session_hf_token_verify_max_pending,
            max_network_pending=settings.session_hf_token_verify_max_pending_per_network,
        )
    )
    requester_rate_limiter = RequesterRateLimiter(
        config=RequesterRateLimitConfig(
            enabled=settings.request_rate_limit_enabled,
            request_window_s=settings.request_rate_limit_window_s,
            max_requests_per_window=settings.request_rate_limit_requests_per_window,
            max_parallel_allocations=settings.request_rate_limit_max_parallel,
            max_consecutive_no_connects=settings.request_rate_limit_no_connects,
            short_session_threshold_s=settings.request_rate_limit_short_session_s,
            max_consecutive_short_sessions=settings.request_rate_limit_short_sessions,
            cooldown_s=settings.request_rate_limit_cooldown_s,
            actor_retention_s=settings.request_rate_limit_actor_retention_s,
            max_actor_states=settings.request_rate_limit_max_actors,
        )
    )
    session_requester_tracker = SessionRequesterTracker(
        retention_s=settings.session_pending_timeout_s + max(2 * settings.session_reap_interval_s, 30.0),
    )
    queue_requester_tracker = SessionRequesterTracker(
        retention_s=settings.queue_ticket_ttl_s + max(2 * settings.queue_reap_interval_s, 10.0),
    )
    return LoadBalancerDependencies(
        session_manager=session_manager,
        dashboard_history_store=dashboard_history_store,
        dashboard=dashboard,
        requester_identity_resolver=requester_identity_resolver,
        session_verification_limiter=session_verification_limiter,
        requester_rate_limiter=requester_rate_limiter,
        session_requester_tracker=session_requester_tracker,
        queue_requester_tracker=queue_requester_tracker,
        speech_proxy_telemetry=speech_proxy_telemetry,
    )


async def record_abnormal_session_disconnect(
    runtime: "LoadBalancerRuntime",
    result: dict[str, object],
) -> None:
    dependencies = runtime.dependencies
    session_id = str(result.get("session_id") or "")
    if session_id:
        outcome = dependencies.requester_rate_limiter.record_disconnected(
            session_id,
            duration_s=_optional_float(result.get("conversation_duration_s")),
            penalize=False,
        )
        if outcome is not None and outcome.connected and outcome.duration_s is not None:
            requester = await _refresh_requester_identity(runtime, outcome.requester)
            await dependencies.dashboard.record_requester_session_disconnected(
                requester,
                duration_s=outcome.duration_s,
                short_session=False,
            )
    await dependencies.dashboard.record_session_event(
        "disconnected",
        conversation_duration_s=result.get("conversation_duration_s"),
        conversation_counted=bool(result.get("conversation_counted")),
    )


async def record_expired_queue_ticket(runtime: "LoadBalancerRuntime", ticket_id: str) -> None:
    """Terminal outcome for a queued request whose ticket the reaper dropped:
    the waiter stopped polling, which is the queue's version of abandoning."""
    dependencies = runtime.dependencies
    requester, _ = dependencies.queue_requester_tracker.take_with_expiry(ticket_id)
    if requester is not None:
        dependencies.requester_rate_limiter.record_allocation_abandoned(requester)
    await dependencies.dashboard.record_session_request_abandoned(requester)


class LoadBalancerRuntime:
    def __init__(self, settings: LoadBalancerSettings, dependencies: LoadBalancerDependencies):
        self.settings = settings
        self.dependencies = dependencies

    async def start(self) -> None:
        dependencies = self.dependencies

        async def abnormal_disconnect(result: dict[str, object]) -> None:
            await record_abnormal_session_disconnect(self, result)

        async def expired_ticket(ticket_id: str) -> None:
            await record_expired_queue_ticket(self, ticket_id)

        dependencies.session_manager.set_abnormal_disconnect_handler(abnormal_disconnect)
        dependencies.session_manager.set_ticket_expired_handler(expired_ticket)
        logger.info(
            "Session queue %s",
            "enabled" if dependencies.session_manager.queue_enabled else "disabled",
        )
        await dependencies.session_manager.start()
        await dependencies.dashboard.start()

    async def stop(self) -> None:
        await self.dependencies.requester_identity_resolver.stop()
        await self.dependencies.dashboard.stop()
        if self.dependencies.speech_proxy_telemetry is not None:
            await self.dependencies.speech_proxy_telemetry.close()
        await self.dependencies.session_manager.stop()


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
    no_connect_penalty_excluded: bool | None = None,
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
        "no_connect_penalty_excluded": no_connect_penalty_excluded,
        "requester_id": requester.actor_id if requester is not None else None,
        "requester_kind": requester.kind if requester is not None else None,
        "requester_verification": requester.verification if requester is not None else None,
        "requester_network_id": requester.network_id if requester is not None else None,
        "requester_reported_robot_id": (requester.reported_robot_id if requester is not None else None),
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
    if no_connect_penalty_excluded is not None:
        message += " no_connect_penalty_excluded=%s"
        args.append(no_connect_penalty_excluded)

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


async def _require_verified_session_requester(
    runtime: LoadBalancerRuntime,
    request: Request,
    requester: RequesterIdentity,
    *,
    http_route: str,
    stage: str,
    wait_for_pending: bool,
) -> RequesterIdentity:
    settings = runtime.settings
    dependencies = runtime.dependencies
    pending_timed_out = False
    if requester.verification == "verified" and not _session_verification_is_fresh(runtime, requester):
        requester = await _start_session_verification(runtime, request, requester, force=True)
    elif (
        requester.verification == "pending"
        and dependencies.requester_identity_resolver.validation_task(requester) is None
    ):
        requester = await _start_session_verification(runtime, request, requester, force=False)
    if wait_for_pending and requester.verification == "pending":
        requester = await dependencies.requester_identity_resolver.wait_for_verification(
            requester,
            timeout_s=settings.session_hf_token_verify_timeout_s,
        )
        pending_timed_out = requester.verification == "pending"
    else:
        requester = await _refresh_requester_identity(runtime, requester)

    if _session_verification_is_fresh(runtime, requester):
        return requester

    await _raise_session_auth_rejection(
        runtime,
        requester,
        reason=_session_auth_rejection_reason(requester, pending_timed_out=pending_timed_out),
        http_route=http_route,
        stage=stage,
        status_code=_session_auth_rejection_status(requester),
    )
    raise AssertionError("authentication rejection helper returned")


async def _start_session_verification(
    runtime: LoadBalancerRuntime,
    request: Request,
    requester: RequesterIdentity,
    *,
    force: bool,
) -> RequesterIdentity:
    dependencies = runtime.dependencies
    token = _request_hf_token(request)
    if token is None:
        return requester

    decision, permit = dependencies.session_verification_limiter.acquire(requester.network_id)
    if not decision.allowed or permit is None:
        await _raise_session_auth_rejection(
            runtime,
            requester,
            reason=f"verification_{decision.reason or 'quota'}",
            http_route="POST /session",
            stage="verification_admission",
            status_code=503,
        )
        raise AssertionError("verification quota rejection helper returned")

    requester, task, started = dependencies.requester_identity_resolver.start_verification(
        token,
        requester,
        force=force,
    )
    if task is not None and started:
        permit.release_when_done(task)
    else:
        permit.release()
    return requester


async def _raise_session_auth_rejection(
    runtime: LoadBalancerRuntime,
    requester: RequesterIdentity | None,
    *,
    reason: str,
    http_route: str,
    stage: str,
    status_code: int,
) -> None:
    dependencies = runtime.dependencies
    await dependencies.dashboard.record_session_auth_rejected(requester)
    extra = {
        "outcome": "auth_rejected",
        "auth_rejection_reason": reason,
        "auth_stage": stage,
        "http_route": http_route,
        "requester_id": requester.actor_id if requester is not None else None,
        "requester_kind": requester.kind if requester is not None else None,
        "requester_verification": requester.verification if requester is not None else None,
        "requester_network_id": requester.network_id if requester is not None else None,
        "requester_reported_robot_id": requester.reported_robot_id if requester is not None else None,
        "requester_client_kind": requester.client_kind if requester is not None else None,
    }
    logger.warning(
        "Session authentication rejected route=%s stage=%s reason=%s "
        "requester_id=%s requester_kind=%s requester_verification=%s",
        http_route,
        stage,
        reason,
        requester.actor_id if requester is not None else None,
        requester.kind if requester is not None else None,
        requester.verification if requester is not None else None,
        extra=extra,
    )

    if status_code == 503:
        retry_after_s = (
            dependencies.requester_identity_resolver.verification_retry_after_s(requester)
            if requester is not None
            else 1
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "hf_token_verification_unavailable",
                "reason": reason,
                "retry_after_s": retry_after_s,
            },
            headers={"Retry-After": str(retry_after_s)},
        )
    raise HTTPException(
        status_code=401,
        detail={"code": "verified_hf_token_required", "reason": reason},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _session_auth_rejection_status(requester: RequesterIdentity) -> int:
    return 503 if requester.verification in {"pending", "unavailable"} else 401


def _session_auth_rejection_reason(
    requester: RequesterIdentity,
    *,
    pending_timed_out: bool,
) -> str:
    if requester.verification == "not_provided":
        return "token_not_provided"
    if requester.verification == "unrecognized":
        return "token_unrecognized"
    if requester.verification == "invalid":
        return "token_invalid"
    if requester.verification == "unavailable":
        return "verification_unavailable"
    if requester.verification == "pending" and pending_timed_out:
        return "verification_timeout"
    if requester.verification == "verified":
        return "verification_stale"
    return "identity_not_verified"


def _session_verification_is_fresh(
    runtime: LoadBalancerRuntime,
    requester: RequesterIdentity,
) -> bool:
    return runtime.dependencies.requester_identity_resolver.verification_is_fresh(
        requester,
        max_age_s=runtime.settings.session_hf_token_max_verified_age_s,
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
            "http_base_url",
            "connect_url",
            "session_token",
            "pending_timeout_s",
            "state",
        )
        if key in allocation
    }


async def _refresh_requester_identity(
    runtime: LoadBalancerRuntime,
    requester: RequesterIdentity,
) -> RequesterIdentity:
    dependencies = runtime.dependencies
    latest = dependencies.requester_identity_resolver.latest_identity(requester)
    if latest != requester:
        await dependencies.dashboard.update_requester_identity(latest)
    return latest


async def _llm_proxy_fingerprint(
    runtime: LoadBalancerRuntime,
    request: Request,
    requester: RequesterIdentity,
) -> str | None:
    """Fingerprint of the verified HF token a session is being created with, or None.

    Embedded as a claim in the signed session token; the compute replica
    opens its LLM proxy paths to api keys matching it while the session's
    websocket is connected. The claim is minted only for tokens HF whoami
    has actually accepted (requester verification "verified"): any invented
    bearer value would otherwise get proxy access, and each rotation a fresh
    per-fingerprint rate-limit identity. A first-seen token's background
    validation is awaited briefly; sessions whose token cannot be verified
    in time get no claim, so their holders get 401 from the LLM paths for
    the session's lifetime. The raw token is never stored or forwarded.
    """
    settings = runtime.settings
    if not settings.session_shared_secret:
        return None
    token = _request_hf_token(request)
    if token is None or not is_validatable_hf_token(token):
        return None
    if requester.verification == "pending":
        requester = await runtime.dependencies.requester_identity_resolver.wait_for_verification(
            requester,
            timeout_s=settings.llm_proxy_claim_verify_timeout_s,
        )
    if requester.verification != "verified":
        return None
    return llm_token_fingerprint(settings.session_shared_secret, token)


def _request_hf_token(request: Request) -> str | None:
    token = bearer_token(request.headers.get("x-reachy-mini-authorization"))
    if token is None:
        token = bearer_token(request.headers.get("authorization"))
    return token


async def root(runtime: LoadBalancerRuntime):
    settings = runtime.settings
    return {
        "message": "s2s load balancer endpoint is up",
        "role": APP_ROLE,
        "ready": "/ready",
        "health": "/health",
        "session": "/session",
        "dashboard": "/dashboard",
        "dashboard_data": "/dashboard/data",
        "compute_endpoints": list(settings.compute_endpoint_names),
        "dashboard_preview_mode": settings.dashboard_preview_mode,
    }


async def ready():
    return JSONResponse(
        {
            "status": "ok",
            "role": APP_ROLE,
        }
    )


async def health(runtime: LoadBalancerRuntime):
    settings = runtime.settings
    dependencies = runtime.dependencies
    healthy, detail, snapshot = await dependencies.session_manager.healthcheck()
    requester_tracking = dependencies.requester_identity_resolver.status()
    requester_tracking["require_verified_hf_token"] = settings.session_require_verified_hf_token
    requester_tracking["session_verification_timeout_s"] = settings.session_hf_token_verify_timeout_s
    requester_tracking["session_max_verified_age_s"] = settings.session_hf_token_max_verified_age_s
    requester_tracking["session_verification_limit"] = dependencies.session_verification_limiter.status()
    requester_tracking["pending_session_attributions"] = dependencies.session_requester_tracker.count()
    requester_tracking["rate_limit"] = dependencies.requester_rate_limiter.status()
    payload = {
        "status": "ok" if healthy else "unhealthy",
        "role": APP_ROLE,
        "compute_endpoints": list(settings.compute_endpoint_names),
        "dashboard_preview_mode": settings.dashboard_preview_mode,
        "dashboard_history": dependencies.dashboard.persistence_status(),
        "requester_tracking": requester_tracking,
        "sessions": snapshot,
    }
    if not healthy:
        payload["detail"] = detail or "endpoint router is not ready"
    return JSONResponse(payload, status_code=200 if healthy else 503)


async def create_session(runtime: LoadBalancerRuntime, request: Request):
    """Grant a session if a slot is free and the line is empty, otherwise return a
    queue ticket the caller polls via GET /queue/{id}. 503 with {state:"at_capacity"}
    when the queue itself is full; 503 otherwise when the pool can't allocate."""
    settings = runtime.settings
    dependencies = runtime.dependencies
    metadata = await session_metadata(request, strict=settings.pipeline_capacity is not None)
    hardware_id = normalize_hardware_id(metadata.get("hardware_id"))
    pipeline = metadata.get("pipeline")
    if pipeline is not None and (
        not isinstance(pipeline, str)
        or settings.pipeline_capacity is None
        or pipeline not in settings.pipeline_capacity.routes
    ):
        raise HTTPException(status_code=400, detail="unknown pipeline route")
    requester = dependencies.requester_identity_resolver.identify(
        request,
        hardware_id=hardware_id,
        schedule_validation=not settings.session_require_verified_hf_token,
    )
    await dependencies.dashboard.record_session_request(requester)
    if settings.session_require_verified_hf_token:
        requester = await _require_verified_session_requester(
            runtime,
            request,
            requester,
            http_route="POST /session",
            stage="admission",
            wait_for_pending=True,
        )
    else:
        requester = await _refresh_requester_identity(runtime, requester)
    rate_limit_decision = dependencies.requester_rate_limiter.acquire(requester)
    if not rate_limit_decision.allowed:
        _log_rate_limit_rejection(rate_limit_decision, requester=requester)
        await dependencies.dashboard.record_session_rate_limited(requester)
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
        allocation = await dependencies.session_manager.allocate(
            public_base_url(request),
            llm_fingerprint=await _llm_proxy_fingerprint(runtime, request, requester),
            **({"pipeline": pipeline} if pipeline is not None else {}),
        )
    except QueueAtCapacityError as exc:
        dependencies.requester_rate_limiter.record_allocation_failure(requester)
        requester = await _refresh_requester_identity(runtime, requester)
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
        await dependencies.dashboard.record_session_allocation_failure(requester)
        return JSONResponse({"state": "at_capacity", "detail": str(exc)}, status_code=503)
    except BaseException as exc:
        dependencies.requester_rate_limiter.record_allocation_failure(requester)
        if not isinstance(exc, Exception):
            raise
        requester = await _refresh_requester_identity(runtime, requester)
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
        await dependencies.dashboard.record_session_allocation_failure(requester)
        raise HTTPException(status_code=503, detail=f"Failed to allocate compute endpoint: {exc}") from exc

    # No slot free (and/or others waiting): the caller joined the queue. Keep the
    # requester identity for the claim — queue polls are bodyless GETs that can't
    # re-derive it.
    if allocation.get("state") == "queued":
        dependencies.queue_requester_tracker.remember(str(allocation["queue_id"]), requester)
        return JSONResponse(allocation)

    return await _deliver_grant(runtime, request, allocation, allocation_started_at, requester)


async def queue_status(runtime: LoadBalancerRuntime, queue_id: str, request: Request):
    """Advance a waiting ticket: report position, or — for the head of the line —
    hand back a session grant once a slot frees. 404 for an unknown/expired ticket.
    404 for everything when the queue is disabled — indistinguishable from main,
    where these routes don't exist."""
    settings = runtime.settings
    dependencies = runtime.dependencies
    if not dependencies.session_manager.queue_enabled:
        raise HTTPException(status_code=404, detail="Not found.")

    requester: RequesterIdentity | None = None
    if settings.session_require_verified_hf_token:
        requester, requester_expired = dependencies.queue_requester_tracker.get_with_expiry(queue_id)
        if requester is None or requester_expired:
            # Do not let the manager claim a free slot when the authorization
            # context associated with this ticket has disappeared. Removing the
            # ticket also prevents repeated attempts from reaching allocation.
            left = await dependencies.session_manager.leave(queue_id)
            if not left:
                raise HTTPException(status_code=404, detail="Unknown or expired ticket.")
            dependencies.queue_requester_tracker.discard(queue_id)
            if requester is not None:
                dependencies.requester_rate_limiter.record_allocation_auth_rejection(requester)
            await _raise_session_auth_rejection(
                runtime,
                requester,
                reason="queue_identity_expired" if requester_expired else "queue_identity_missing",
                http_route="GET /queue/{queue_id}",
                stage="queue_claim",
                status_code=401,
            )
        requester = await _refresh_requester_identity(runtime, requester)
        if not _session_verification_is_fresh(runtime, requester):
            left = await dependencies.session_manager.leave(queue_id)
            if not left:
                raise HTTPException(status_code=404, detail="Unknown or expired ticket.")
            dependencies.queue_requester_tracker.discard(queue_id)
            dependencies.requester_rate_limiter.record_allocation_auth_rejection(requester)
            await _raise_session_auth_rejection(
                runtime,
                requester,
                reason=_session_auth_rejection_reason(requester, pending_timed_out=True),
                http_route="GET /queue/{queue_id}",
                stage="queue_claim",
                status_code=_session_auth_rejection_status(requester),
            )

    poll_started_at = monotonic()
    try:
        result = await dependencies.session_manager.poll(queue_id, public_base_url(request))
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown or expired ticket.") from None
    except Exception as exc:
        # Same contract as POST /session: allocation-time failures are 503s, not
        # 500s. The manager re-queues the ticket at the head on a failed claim,
        # so the caller keeps its place and simply polls again.
        if requester is not None:
            dependencies.queue_requester_tracker.remember(queue_id, requester)
        raise HTTPException(status_code=503, detail=f"Failed to claim session: {exc}") from exc

    tracked_requester = dependencies.queue_requester_tracker.take(queue_id)
    if tracked_requester is not None:
        requester = tracked_requester
    if result.get("state") == "queued":
        if requester is not None:
            dependencies.queue_requester_tracker.remember(queue_id, requester)  # refresh retention
        return JSONResponse(result)
    if result.get("state") == "timed_out":
        if requester is not None:
            dependencies.requester_rate_limiter.record_allocation_failure(requester)
        await dependencies.dashboard.record_session_allocation_failure(requester)
        return JSONResponse(result, status_code=503, headers={"Retry-After": str(result["retry_after_s"])})

    # Head of line claimed a slot — same delivery path as a fast-path grant. The
    # requester was resolved at ticket creation; falling back to the poll request
    # (bodyless, so IP-only) only happens if the tracker entry expired.
    if requester is None:
        hardware_id = await reported_hardware_id(request)
        requester = dependencies.requester_identity_resolver.identify(request, hardware_id=hardware_id)
    return await _deliver_grant(
        runtime,
        request,
        result,
        poll_started_at,
        requester,
        http_route="GET /queue/{queue_id}",
    )


async def queue_leave(runtime: LoadBalancerRuntime, queue_id: str):
    """Leave the queue early (explicit button / teardown beacon). Idempotent."""
    dependencies = runtime.dependencies
    if not dependencies.session_manager.queue_enabled:
        raise HTTPException(status_code=404, detail="Not found.")
    left = await dependencies.session_manager.leave(queue_id)
    requester, _ = dependencies.queue_requester_tracker.take_with_expiry(queue_id)
    if left:
        # Terminal outcome for the queued request: leaving the line is the queue's
        # version of abandoning before delivery.
        if requester is not None:
            dependencies.requester_rate_limiter.record_allocation_abandoned(requester)
        await dependencies.dashboard.record_session_request_abandoned(requester)
    return JSONResponse({"status": "ok", "state": "left", "removed": left})


async def _deliver_grant(
    runtime: LoadBalancerRuntime,
    request: Request,
    allocation: dict[str, object],
    started_at: float,
    requester: RequesterIdentity,
    *,
    http_route: str = "POST /session",
) -> JSONResponse:
    """Shared tail for a granted session (fast path or queue claim): guard against a
    client that vanished, record the success, and return the public grant fields."""
    settings = runtime.settings
    dependencies = runtime.dependencies
    allocation_total_ms = elapsed_ms(started_at, monotonic())
    allocation_wait_ms = _allocation_wait_ms(allocation, fallback_ms=allocation_total_ms)
    allocation.setdefault("allocation_wait_ms", allocation_wait_ms)
    session_id = str(allocation.get("session_id") or "")

    if settings.session_require_verified_hf_token:
        requester = await _refresh_requester_identity(runtime, requester)
        if not _session_verification_is_fresh(runtime, requester):
            if session_id:
                await dependencies.session_manager.cancel_pending_session(session_id)
            dependencies.requester_rate_limiter.record_allocation_auth_rejection(requester)
            await _raise_session_auth_rejection(
                runtime,
                requester,
                reason=_session_auth_rejection_reason(requester, pending_timed_out=True),
                http_route=http_route,
                stage="grant_delivery",
                status_code=_session_auth_rejection_status(requester),
            )

    if session_id:
        dependencies.requester_rate_limiter.record_allocation(
            session_id,
            requester,
            pending_timeout_s=float(allocation.get("pending_timeout_s") or settings.session_pending_timeout_s),
        )
    else:
        dependencies.requester_rate_limiter.record_allocation_failure(requester)

    if await request.is_disconnected():
        requester = await _refresh_requester_identity(runtime, requester)
        # Queue grants are delayed by definition; waited_for_capacity also covers
        # the queue-disabled path that blocked until a compute slot became free.
        no_connect_penalty_excluded = bool(allocation.get("waited_for_capacity")) or (
            http_route == "GET /queue/{queue_id}"
        )
        if session_id:
            await dependencies.session_manager.cancel_pending_session(session_id)
        if session_id:
            dependencies.requester_rate_limiter.record_disconnected(
                session_id,
                penalize=not no_connect_penalty_excluded,
            )
        _log_session_allocation_outcome(
            "client_disconnected",
            allocation=allocation,
            allocation_wait_ms=allocation_wait_ms,
            allocation_total_ms=allocation_total_ms,
            level=logging.WARNING,
            requester=requester,
            http_route=http_route,
            no_connect_penalty_excluded=no_connect_penalty_excluded,
        )
        await dependencies.dashboard.record_session_request_abandoned(requester)
        raise HTTPException(status_code=503, detail="Client disconnected before session could be delivered")

    requester = await _refresh_requester_identity(runtime, requester)
    if session_id:
        dependencies.session_requester_tracker.remember(session_id, requester)
    await dependencies.dashboard.record_session_allocation_success(requester)
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


async def session_event(
    runtime: LoadBalancerRuntime,
    session_id: str,
    payload: dict[str, Any],
):
    dependencies = runtime.dependencies
    session_token = str(payload.get("session_token", "")).strip()
    event = str(payload.get("event", "")).strip()
    if not session_token:
        raise HTTPException(status_code=400, detail="session_token is required")
    if not event:
        raise HTTPException(status_code=400, detail="event is required")

    try:
        result = await dependencies.session_manager.handle_event(session_id, session_token, event)
    except KeyError:
        if event == "disconnected":
            dependencies.requester_rate_limiter.record_disconnected(session_id)
            dependencies.session_requester_tracker.discard(session_id)
            return JSONResponse({"status": "ok", "session_id": session_id, "state": "already_released"})
        raise HTTPException(status_code=404, detail="Unknown session id") from None
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    await dependencies.dashboard.record_session_event(
        event,
        conversation_duration_s=result.get("conversation_duration_s"),
        conversation_counted=bool(result.get("conversation_counted")),
    )
    if event == "connected":
        dependencies.requester_rate_limiter.record_connected(session_id)
        requester = dependencies.session_requester_tracker.take(session_id)
        if requester is not None:
            requester = await _refresh_requester_identity(runtime, requester)
            await dependencies.dashboard.record_requester_session_connected(requester)
    elif event == "disconnected":
        outcome = dependencies.requester_rate_limiter.record_disconnected(
            session_id,
            duration_s=_optional_float(result.get("conversation_duration_s")),
            penalize=(
                bool(result.get("conversation_counted")) and result.get("release_reason") != "endpoint_unavailable"
            ),
        )
        if outcome is not None and outcome.connected and outcome.duration_s is not None:
            requester = await _refresh_requester_identity(runtime, outcome.requester)
            await dependencies.dashboard.record_requester_session_disconnected(
                requester,
                duration_s=outcome.duration_s,
                short_session=outcome.short_session,
            )
        dependencies.session_requester_tracker.discard(session_id)
    return JSONResponse(result)


async def llm_proxy_usage(
    runtime: LoadBalancerRuntime,
    payload: dict[str, Any],
):
    """Record one compute-reported LLM proxy gate decision."""
    if payload.keys() - {"reason", "token", "client_ip"}:
        raise HTTPException(status_code=400, detail="callback payload contains unknown fields")
    reason = payload.get("reason")
    if reason not in LLM_PROXY_REASONS:
        raise HTTPException(status_code=400, detail="invalid LLM proxy reason")

    token = payload.get("token")
    if token is not None and (not isinstance(token, str) or not is_validatable_hf_token(token)):
        raise HTTPException(status_code=400, detail="token is malformed or too long")

    client_ip = payload.get("client_ip")
    if client_ip is not None:
        if not isinstance(client_ip, str) or not client_ip.strip() or len(client_ip) > LLM_PROXY_CLIENT_IP_MAX_LENGTH:
            raise HTTPException(status_code=400, detail="client_ip is malformed or too long")

    resolver = runtime.dependencies.requester_identity_resolver
    requester = resolver.identify_values(token=None, address=client_ip)
    if token is not None:
        token_requester = await resolver.wait_for_verification(
            resolver.identify_values(token=token, address=client_ip),
            timeout_s=runtime.settings.llm_proxy_claim_verify_timeout_s,
        )
        requester = (
            token_requester
            if token_requester.verification == "verified"
            else replace(requester, verification=token_requester.verification)
        )
    elif reason == "no_active_session_match":
        requester = replace(requester, verification="unrecognized")

    await runtime.dependencies.dashboard.record_llm_proxy_request(
        reason=reason,
        actor_id=requester.actor_id,
        metadata=requester.history_metadata(),
    )
    return JSONResponse({"status": "ok", "state": "recorded"})


async def _llm_proxy_usage_payload(request: Request) -> dict[str, Any]:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > LLM_PROXY_CALLBACK_BODY_MAX_BYTES:
            raise HTTPException(status_code=413, detail="callback payload is too large")
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="callback payload must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="callback payload must be an object")
    return payload


async def endpoint_status(runtime: LoadBalancerRuntime, endpoint_name: str, request: Request):
    require_admin_auth(runtime, request)

    endpoint_snapshot = await get_endpoint_snapshot(runtime, endpoint_name)
    return JSONResponse(
        {
            "status": "ok",
            "endpoint_name": endpoint_name,
            "endpoint": endpoint_snapshot,
        }
    )


async def endpoint_drain(
    runtime: LoadBalancerRuntime,
    endpoint_name: str,
    request: Request,
    payload: dict[str, Any],
):
    require_admin_auth(runtime, request)

    endpoint_router = getattr(runtime.dependencies.session_manager, "endpoint_router", None)
    if endpoint_router is None:
        raise HTTPException(status_code=503, detail="Endpoint draining is not available")

    endpoint_snapshot = await get_endpoint_snapshot(runtime, endpoint_name)
    draining = payload.get("draining", True)
    if type(draining) is not bool:
        raise HTTPException(status_code=422, detail="draining must be a boolean")
    lease_ttl_s = payload.get("lease_ttl_s")
    if lease_ttl_s is not None and (type(lease_ttl_s) not in (int, float) or lease_ttl_s <= 0):
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
    endpoint_snapshot = await get_endpoint_snapshot(runtime, endpoint_name)

    return JSONResponse(
        {
            "status": "ok",
            "endpoint_name": endpoint_name,
            "draining": endpoint_snapshot.get("draining"),
            "endpoint": endpoint_snapshot,
        }
    )


async def get_endpoint_snapshot(
    runtime: LoadBalancerRuntime,
    endpoint_name: str,
) -> dict[str, object]:
    session_manager = runtime.dependencies.session_manager
    endpoint_router = getattr(session_manager, "endpoint_router", None)
    if endpoint_router is None:
        raise HTTPException(status_code=503, detail="Endpoint status is not available")

    _, _, snapshot = await session_manager.healthcheck()
    router_snapshot = snapshot.get("router", {})
    endpoints = router_snapshot.get("endpoints", []) if isinstance(router_snapshot, dict) else []
    endpoint_snapshot = next(
        (endpoint for endpoint in endpoints if isinstance(endpoint, dict) and endpoint.get("name") == endpoint_name),
        None,
    )
    if endpoint_snapshot is None:
        raise HTTPException(status_code=404, detail="Unknown endpoint")
    return endpoint_snapshot


def require_callback_auth(runtime: LoadBalancerRuntime, request: Request) -> None:
    authorization = request.headers.get(LLM_PROXY_CALLBACK_AUTH_HEADER)
    if authorization is None:
        authorization = request.headers.get("authorization")
    _require_bearer_auth(authorization, runtime.settings.lb_callback_auth_token, "callback")


def require_admin_auth(runtime: LoadBalancerRuntime, request: Request) -> None:
    authorization = request.headers.get(LB_ADMIN_AUTH_HEADER)
    if authorization is None:
        authorization = request.headers.get("authorization")
    _require_bearer_auth(authorization, runtime.settings.lb_admin_auth_token, "admin")


def _require_bearer_auth(authorization: str | None, expected_token: str | None, label: str) -> None:
    if not expected_token:
        raise HTTPException(status_code=503, detail=f"LB {label} auth token is not configured")
    token = _bearer_token(authorization)
    if token is None:
        raise HTTPException(
            status_code=401,
            detail=f"Missing {label} bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(token, expected_token):
        raise HTTPException(status_code=403, detail=f"Invalid {label} authorization")


def _bearer_token(authorization: str | None) -> str | None:
    return bearer_token(authorization)


async def deprecated_websocket_route(client_ws: WebSocket):
    await client_ws.close(
        code=1008, reason="Use POST /session and connect directly to the returned compute websocket URL"
    )


async def dashboard_page(runtime: LoadBalancerRuntime):
    return HTMLResponse(runtime.dependencies.dashboard.html())


async def dashboard_data(
    runtime: LoadBalancerRuntime,
    window: str = "6h",
    resolution: str = "",
):
    try:
        payload = await runtime.dependencies.dashboard.data(
            window=window,
            resolution=resolution or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(payload)


def create_app(
    settings: LoadBalancerSettings,
    dependencies: LoadBalancerDependencies | None = None,
) -> FastAPI:
    """Create a load-balancer application from explicit configuration."""
    resolved_dependencies = dependencies or build_load_balancer_dependencies(settings)
    runtime = LoadBalancerRuntime(settings, resolved_dependencies)
    application = FastAPI(lifespan=build_lifespan(runtime))
    application.state.runtime = runtime
    application.state.settings = settings
    application.state.dependencies = resolved_dependencies

    async def root_route():
        return await root(runtime)

    async def health_route():
        return await health(runtime)

    async def create_session_route(request: Request):
        return await create_session(runtime, request)

    async def queue_status_route(queue_id: str, request: Request):
        return await queue_status(runtime, queue_id, request)

    async def queue_leave_route(queue_id: str):
        return await queue_leave(runtime, queue_id)

    async def session_event_route(session_id: str, payload: dict[str, Any]):
        return await session_event(runtime, session_id, payload)

    async def llm_proxy_usage_route(request: Request):
        require_callback_auth(runtime, request)
        return await llm_proxy_usage(runtime, await _llm_proxy_usage_payload(request))

    async def endpoint_status_route(endpoint_name: str, request: Request):
        return await endpoint_status(runtime, endpoint_name, request)

    async def endpoint_drain_route(
        endpoint_name: str,
        request: Request,
        payload: dict[str, Any],
    ):
        return await endpoint_drain(runtime, endpoint_name, request, payload)

    async def dashboard_page_route():
        return await dashboard_page(runtime)

    async def dashboard_data_route(window: str = "6h", resolution: str = ""):
        return await dashboard_data(runtime, window, resolution)

    application.add_api_route("/", root_route, methods=["GET"], name="root")
    application.add_api_route("/ready", ready, methods=["GET"], name="ready")
    application.add_api_route("/health", health_route, methods=["GET"], name="health")
    application.add_api_route(
        "/session",
        create_session_route,
        methods=["POST"],
        name="create_session",
        description=cleandoc(create_session.__doc__ or ""),
    )
    application.add_api_route(
        "/queue/{queue_id}",
        queue_status_route,
        methods=["GET"],
        name="queue_status",
        description=cleandoc(queue_status.__doc__ or ""),
    )
    application.add_api_route(
        "/queue/{queue_id}",
        queue_leave_route,
        methods=["DELETE"],
        name="queue_leave",
        description=cleandoc(queue_leave.__doc__ or ""),
    )
    application.add_api_route(
        "/internal/sessions/{session_id}/event",
        session_event_route,
        methods=["POST"],
        name="session_event",
    )
    application.add_api_route(
        "/internal/llm-proxy-usage",
        llm_proxy_usage_route,
        methods=["POST"],
        name="llm_proxy_usage",
    )
    application.add_api_route(
        "/internal/endpoints/{endpoint_name}",
        endpoint_status_route,
        methods=["GET"],
        name="endpoint_status",
    )
    application.add_api_route(
        "/internal/endpoints/{endpoint_name}/drain",
        endpoint_drain_route,
        methods=["POST"],
        name="endpoint_drain",
    )
    application.add_api_websocket_route(
        "/ws",
        deprecated_websocket_route,
        name="deprecated_websocket_route",
    )
    application.add_api_route(
        "/dashboard",
        dashboard_page_route,
        methods=["GET"],
        name="dashboard_page",
    )
    application.add_api_route(
        "/dashboard/data",
        dashboard_data_route,
        methods=["GET"],
        name="dashboard_data",
    )
    return application

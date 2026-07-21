import asyncio
import copy
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol


logger = logging.getLogger("s2s-endpoint")
DAY_MINUTES = 24 * 60
DAY_SECONDS = DAY_MINUTES * 60
HISTORY_MERGE_COMPARISON_CHUNK_SIZE = 256
PERSISTENCE_WORKER_RETRY_DELAY_S = 1.0
FLUSH_RETRY_INITIAL_DELAY_S = 15.0
FLUSH_RETRY_MAX_DELAY_S = 300.0
STARTUP_MERGE_PASS_COUNT = 2


def _normalize_status(status: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(status).lower())


def _bucket_start_epoch_s(epoch_s: float, bucket_minutes: int) -> int:
    bucket_seconds = bucket_minutes * 60
    return int(epoch_s // bucket_seconds) * bucket_seconds


def _day_start_epoch_s(epoch_s: float) -> int:
    return _bucket_start_epoch_s(epoch_s, DAY_MINUTES)


def _startup_merge_retention_minutes(now_epoch_s: float) -> int:
    current_bucket_start_s = _bucket_start_epoch_s(now_epoch_s, 1)
    previous_day_start_s = _day_start_epoch_s(now_epoch_s) - DAY_SECONDS
    return ((current_bucket_start_s - previous_day_start_s) // 60) + 1


def _day_key(epoch_s: int | float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y-%m-%d")


def _isoformat(epoch_s: int | float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat().replace("+00:00", "Z")


class DashboardHistoryStore(Protocol):
    def load_recent(self, *, retention_minutes: int, now_epoch_s: float) -> list["SwarmHistoryBucket"]:
        ...

    def write_buckets(self, buckets: list["SwarmHistoryBucket"]) -> None:
        ...

    def write_day_buckets(self, *, day_start_s: int, buckets: list["SwarmHistoryBucket"]) -> Optional[str]:
        ...


@dataclass
class SwarmStateSample:
    captured_at_s: float
    healthy: bool
    detail: Optional[str]
    total_endpoints: int
    running_endpoints: int
    warming_endpoints: int
    transitioning_endpoints: int
    parked_endpoints: int
    connected_sessions: int
    pending_sessions: int
    free_slots: int
    effective_free_slots: int
    router_active_sessions: int
    errors_count: int
    endpoints: list[dict[str, object]]

    @classmethod
    def from_health_snapshot(
        cls,
        *,
        healthy: bool,
        detail: Optional[str],
        snapshot: dict[str, object],
        captured_at_s: float,
    ) -> "SwarmStateSample":
        router = snapshot.get("router") or {}
        endpoints = list(router.get("endpoints") or [])
        status_counts: dict[str, int] = {}
        for endpoint in endpoints:
            status = _normalize_status(endpoint.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1

        paused_endpoints = status_counts.get("paused", 0)
        scaled_to_zero_endpoints = status_counts.get("scaledtozero", 0) + status_counts.get("scaledto0", 0)
        transitioning_endpoints = status_counts.get("initializing", 0) + status_counts.get("updating", 0)

        return cls(
            captured_at_s=captured_at_s,
            healthy=healthy,
            detail=detail,
            total_endpoints=len(endpoints),
            running_endpoints=int(router.get("running_endpoints", 0)),
            warming_endpoints=int(router.get("waking_endpoints", 0)),
            transitioning_endpoints=transitioning_endpoints,
            parked_endpoints=paused_endpoints + scaled_to_zero_endpoints,
            connected_sessions=int(snapshot.get("connected_sessions", 0)),
            pending_sessions=int(snapshot.get("pending_sessions", 0)),
            free_slots=int(router.get("free_slots", 0)),
            effective_free_slots=int(router.get("effective_free_slots", 0)),
            router_active_sessions=int(router.get("active_sessions", 0)),
            errors_count=len(list(router.get("errors") or [])),
            endpoints=endpoints,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "captured_at": _isoformat(self.captured_at_s),
            "healthy": self.healthy,
            "detail": self.detail,
            "total_endpoints": self.total_endpoints,
            "running_endpoints": self.running_endpoints,
            "warming_endpoints": self.warming_endpoints,
            "transitioning_endpoints": self.transitioning_endpoints,
            "parked_endpoints": self.parked_endpoints,
            "connected_sessions": self.connected_sessions,
            "pending_sessions": self.pending_sessions,
            "free_slots": self.free_slots,
            "effective_free_slots": self.effective_free_slots,
            "router_active_sessions": self.router_active_sessions,
            "errors_count": self.errors_count,
            "endpoints": self.endpoints,
        }


@dataclass
class SwarmHistoryBucket:
    bucket_start_s: int
    sample_count: int = 0
    running_endpoints_last: int = 0
    running_endpoints_sum: float = 0.0
    warming_endpoints_last: int = 0
    warming_endpoints_sum: float = 0.0
    transitioning_endpoints_last: int = 0
    transitioning_endpoints_sum: float = 0.0
    parked_endpoints_last: int = 0
    parked_endpoints_sum: float = 0.0
    connected_sessions_last: int = 0
    connected_sessions_sum: float = 0.0
    connected_sessions_max: int = 0
    pending_sessions_last: int = 0
    pending_sessions_sum: float = 0.0
    free_slots_last: int = 0
    free_slots_sum: float = 0.0
    effective_free_slots_last: int = 0
    effective_free_slots_sum: float = 0.0
    router_active_sessions_last: int = 0
    router_active_sessions_sum: float = 0.0
    healthy_last: bool = False
    healthy_samples: int = 0
    errors_count_last: int = 0
    errors_count_sum: float = 0.0
    session_requests: int = 0
    session_allocation_successes: int = 0
    session_allocation_failures: int = 0
    session_connected_events: int = 0
    session_disconnected_events: int = 0
    completed_conversations: int = 0
    completed_conversation_duration_total_s: float = 0.0
    completed_conversation_duration_max_s: float = 0.0
    completed_conversation_duration_samples_s: list[float] = field(default_factory=list)
    requester_usage: dict[str, dict[str, object]] = field(default_factory=dict)

    def record_sample(self, sample: SwarmStateSample) -> None:
        self.sample_count += 1
        self.running_endpoints_last = sample.running_endpoints
        self.running_endpoints_sum += sample.running_endpoints
        self.warming_endpoints_last = sample.warming_endpoints
        self.warming_endpoints_sum += sample.warming_endpoints
        self.transitioning_endpoints_last = sample.transitioning_endpoints
        self.transitioning_endpoints_sum += sample.transitioning_endpoints
        self.parked_endpoints_last = sample.parked_endpoints
        self.parked_endpoints_sum += sample.parked_endpoints
        self.connected_sessions_last = sample.connected_sessions
        self.connected_sessions_sum += sample.connected_sessions
        self.connected_sessions_max = max(self.connected_sessions_max, sample.connected_sessions)
        self.pending_sessions_last = sample.pending_sessions
        self.pending_sessions_sum += sample.pending_sessions
        self.free_slots_last = sample.free_slots
        self.free_slots_sum += sample.free_slots
        self.effective_free_slots_last = sample.effective_free_slots
        self.effective_free_slots_sum += sample.effective_free_slots
        self.router_active_sessions_last = sample.router_active_sessions
        self.router_active_sessions_sum += sample.router_active_sessions
        self.healthy_last = sample.healthy
        self.healthy_samples += int(sample.healthy)
        self.errors_count_last = sample.errors_count
        self.errors_count_sum += sample.errors_count

    def as_minute_point(self) -> dict[str, object]:
        return {
            "timestamp": _isoformat(self.bucket_start_s),
            "running_endpoints": self.running_endpoints_last,
            "warming_endpoints": self.warming_endpoints_last,
            "transitioning_endpoints": self.transitioning_endpoints_last,
            "parked_endpoints": self.parked_endpoints_last,
            "connected_sessions": self.connected_sessions_last,
            "pending_sessions": self.pending_sessions_last,
            "free_slots": self.free_slots_last,
            "effective_free_slots": self.effective_free_slots_last,
            "router_active_sessions": self.router_active_sessions_last,
            "errors_count": self.errors_count_last,
            "healthy": self.healthy_last,
            "session_requests": self.session_requests,
            "session_allocation_successes": self.session_allocation_successes,
            "session_allocation_failures": self.session_allocation_failures,
            "session_connected_events": self.session_connected_events,
            "session_disconnected_events": self.session_disconnected_events,
            "completed_conversations": self.completed_conversations,
            "avg_conversation_duration_s": (
                round(self.completed_conversation_duration_total_s / self.completed_conversations, 2)
                if self.completed_conversations
                else 0.0
            ),
            "avg_conversation_duration_min": (
                round((self.completed_conversation_duration_total_s / self.completed_conversations) / 60.0, 2)
                if self.completed_conversations
                else 0.0
            ),
            "max_conversation_duration_s": round(self.completed_conversation_duration_max_s, 2),
            "max_conversation_duration_min": round(self.completed_conversation_duration_max_s / 60.0, 2),
        }

    def to_dict(self) -> dict[str, object]:
        payload = {}
        for bucket_field in dataclass_fields(self):
            value = getattr(self, bucket_field.name)
            payload[bucket_field.name] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SwarmHistoryBucket":
        values = {
            bucket_field.name: _coerce_history_bucket_field(bucket_field.name, payload)
            for bucket_field in dataclass_fields(cls)
        }
        return cls(**values)

    def completed_duration_samples(self) -> list[float]:
        if self.completed_conversations <= 0:
            return []
        if self.completed_conversation_duration_samples_s:
            return list(self.completed_conversation_duration_samples_s)
        avg_duration_s = self.completed_conversation_duration_total_s / self.completed_conversations
        return [avg_duration_s] * self.completed_conversations


_HISTORY_BUCKET_INT_FIELDS = {
    "bucket_start_s",
    "sample_count",
    "running_endpoints_last",
    "warming_endpoints_last",
    "transitioning_endpoints_last",
    "parked_endpoints_last",
    "connected_sessions_last",
    "connected_sessions_max",
    "pending_sessions_last",
    "free_slots_last",
    "effective_free_slots_last",
    "router_active_sessions_last",
    "healthy_samples",
    "errors_count_last",
    "session_requests",
    "session_allocation_successes",
    "session_allocation_failures",
    "session_connected_events",
    "session_disconnected_events",
    "completed_conversations",
}
_HISTORY_BUCKET_FLOAT_FIELDS = {
    "running_endpoints_sum",
    "warming_endpoints_sum",
    "transitioning_endpoints_sum",
    "parked_endpoints_sum",
    "connected_sessions_sum",
    "pending_sessions_sum",
    "free_slots_sum",
    "effective_free_slots_sum",
    "router_active_sessions_sum",
    "errors_count_sum",
    "completed_conversation_duration_total_s",
    "completed_conversation_duration_max_s",
}


def _coerce_history_bucket_field(name: str, payload: dict[str, object]) -> object:
    if name == "connected_sessions_max":
        return max(
            int(payload.get("connected_sessions_max", 0)),
            int(payload.get("connected_sessions_last", 0)),
        )
    if name in _HISTORY_BUCKET_INT_FIELDS:
        return int(payload.get(name, 0))
    if name in _HISTORY_BUCKET_FLOAT_FIELDS:
        return float(payload.get(name, 0.0))
    if name == "healthy_last":
        return bool(payload.get(name, False))
    if name == "completed_conversation_duration_samples_s":
        return [max(float(value), 0.0) for value in list(payload.get(name) or [])]
    if name == "requester_usage":
        return _coerce_requester_usage(payload.get(name))
    raise KeyError(f"Unknown SwarmHistoryBucket field: {name}")


def _coerce_requester_usage(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}

    usage: dict[str, dict[str, object]] = {}
    for raw_actor_id, raw_record in value.items():
        if not isinstance(raw_record, dict):
            continue
        actor_id = str(raw_actor_id)[:128]
        network_ids = [str(item)[:128] for item in list(raw_record.get("network_ids") or [])[:32]]
        raw_client_kinds = raw_record.get("client_kinds") or {}
        client_kinds = (
            {
                str(kind)[:80]: max(int(count), 0)
                for kind, count in raw_client_kinds.items()
            }
            if isinstance(raw_client_kinds, dict)
            else {}
        )
        usage[actor_id] = {
            "label": str(raw_record.get("label") or "Unknown requester")[:160],
            "kind": str(raw_record.get("kind") or "unknown")[:40],
            "verification": str(raw_record.get("verification") or "unknown")[:40],
            "fingerprint": str(raw_record.get("fingerprint") or "")[:40],
            "account_name": (
                str(raw_record["account_name"])[:80]
                if raw_record.get("account_name") is not None
                else None
            ),
            "requests": max(int(raw_record.get("requests", 0)), 0),
            "successes": max(int(raw_record.get("successes", 0)), 0),
            "failures": max(int(raw_record.get("failures", 0)), 0),
            "abandoned": max(int(raw_record.get("abandoned", 0)), 0),
            "network_ids": network_ids,
            "network_ids_overflow": bool(raw_record.get("network_ids_overflow", False)),
            "client_kinds": client_kinds,
        }
    return usage


_VERIFICATION_RANK = {
    "unknown": 0,
    "not_provided": 1,
    "not_applicable": 1,
    "unrecognized": 2,
    "pending": 3,
    "unavailable": 4,
    "invalid": 5,
    "verified": 6,
}


def _new_requester_usage_record(metadata: dict[str, object]) -> dict[str, object]:
    return {
        "label": str(metadata.get("label") or "Unknown requester")[:160],
        "kind": str(metadata.get("kind") or "unknown")[:40],
        "verification": str(metadata.get("verification") or "unknown")[:40],
        "fingerprint": str(metadata.get("fingerprint") or "")[:40],
        "account_name": metadata.get("account_name"),
        "requests": 0,
        "successes": 0,
        "failures": 0,
        "abandoned": 0,
        "network_ids": [],
        "network_ids_overflow": False,
        "client_kinds": {},
    }


def _merge_requester_identity(record: dict[str, object], metadata: dict[str, object]) -> None:
    current_verification = str(record.get("verification") or "unknown")
    new_verification = str(metadata.get("verification") or "unknown")
    if _VERIFICATION_RANK.get(new_verification, 0) < _VERIFICATION_RANK.get(current_verification, 0):
        return

    record["label"] = str(metadata.get("label") or record.get("label") or "Unknown requester")[:160]
    record["kind"] = str(metadata.get("kind") or record.get("kind") or "unknown")[:40]
    record["verification"] = new_verification[:40]
    record["fingerprint"] = str(metadata.get("fingerprint") or record.get("fingerprint") or "")[:40]
    account_name = metadata.get("account_name")
    if account_name is not None:
        record["account_name"] = str(account_name)[:80]


def _record_request_context(record: dict[str, object], metadata: dict[str, object]) -> None:
    network_id = metadata.get("network_id")
    network_ids = record.setdefault("network_ids", [])
    if network_id and isinstance(network_ids, list) and network_id not in network_ids:
        if len(network_ids) < 32:
            network_ids.append(str(network_id)[:128])
        else:
            record["network_ids_overflow"] = True

    client_kind = str(metadata.get("client_kind") or "unknown")[:80]
    client_kinds = record.setdefault("client_kinds", {})
    if isinstance(client_kinds, dict):
        client_kinds[client_kind] = int(client_kinds.get(client_kind, 0)) + 1


class DashboardHistory:
    def __init__(
        self,
        *,
        retention_minutes: int,
        history_store: Optional[DashboardHistoryStore] = None,
        restore_history_in_background: bool = False,
        flush_batch_size: int = 100,
        flush_timeout_s: float = 60.0,
        dirty_bucket_warning_age_s: float = 300.0,
        startup_merge_delay_s: float = 60.0,
        max_requesters_per_bucket: int = 1000,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        if retention_minutes < 60:
            raise ValueError("retention_minutes must be >= 60")
        if flush_batch_size < 1:
            raise ValueError("flush_batch_size must be >= 1")
        if flush_timeout_s <= 0:
            raise ValueError("flush_timeout_s must be > 0")
        if dirty_bucket_warning_age_s <= 0:
            raise ValueError("dirty_bucket_warning_age_s must be > 0")
        if startup_merge_delay_s < 0:
            raise ValueError("startup_merge_delay_s must be >= 0")
        if max_requesters_per_bucket < 1:
            raise ValueError("max_requesters_per_bucket must be >= 1")

        self.retention_minutes = retention_minutes
        self.history_store = history_store
        self.restore_history_in_background = restore_history_in_background
        self.flush_batch_size = flush_batch_size
        self.flush_timeout_s = flush_timeout_s
        self.shutdown_flush_budget_s = 2 * flush_timeout_s
        self.dirty_bucket_warning_age_s = dirty_bucket_warning_age_s
        self.startup_merge_delay_s = startup_merge_delay_s
        self.max_requesters_per_bucket = max_requesters_per_bucket
        self._time_fn = time_fn
        self._lock = asyncio.Lock()
        self._history: "OrderedDict[int, SwarmHistoryBucket]" = OrderedDict()
        self._restore_task: Optional[asyncio.Task] = None
        self._startup_merge_task: Optional[asyncio.Task] = None
        self._persistence_task: Optional[asyncio.Task] = None
        self._flush_write_task: Optional[asyncio.Task] = None
        self._persistence_wakeup = asyncio.Event()
        self._persistence_stop_requested = False
        self._dirty_bucket_starts: set[int] = set()
        self._locally_sampled_bucket_starts: set[int] = set()
        self._flush_started_at_monotonic_s: Optional[float] = None
        self._last_flush_started_at_s: Optional[float] = None
        self._last_flush_finished_at_s: Optional[float] = None
        self._last_flush_error: Optional[str] = None
        self._flush_stalled_started_at_s: Optional[float] = None
        self._flush_stalled_started_at_monotonic_s: Optional[float] = None
        self._flush_retry_failure_count = 0
        self._flush_retry_delay_s: Optional[float] = None
        self._flush_retry_next_at_s: Optional[float] = None
        self._flush_retry_not_before_monotonic_s: Optional[float] = None
        self._last_dirty_bucket_warning_at_monotonic_s: Optional[float] = None
        self._day_rollover_cursor_s: Optional[int] = None
        self._history_restore_status = "disabled" if history_store is None else "pending"
        self._history_restore_detail: Optional[str] = None
        self._history_restore_started_at_s: Optional[float] = None
        self._history_restore_finished_at_s: Optional[float] = None
        self._history_restore_bucket_count = 0
        self._startup_merge_status = (
            "disabled"
            if history_store is None or startup_merge_delay_s == 0
            else "pending"
        )
        self._startup_merge_scheduled_passes = (
            0
            if history_store is None or startup_merge_delay_s == 0
            else STARTUP_MERGE_PASS_COUNT
        )
        self._startup_merge_attempted_passes = 0
        self._startup_merge_completed_passes = 0
        self._startup_merge_failed_passes = 0
        self._startup_merge_bucket_count = 0
        self._startup_merge_updated_bucket_count = 0
        self._startup_merge_last_error: Optional[str] = None

    async def start(self) -> None:
        if self.history_store is not None:
            if self.restore_history_in_background:
                logger.info("Restoring dashboard history in the background")
                self._history_restore_status = "running"
                self._history_restore_detail = "Loading persisted dashboard history"
                self._restore_task = asyncio.create_task(self._restore_history())
            else:
                await self._restore_history()
            if self._startup_merge_scheduled_passes > 0:
                self._startup_merge_status = "waiting"
                self._startup_merge_attempted_passes = 0
                self._startup_merge_completed_passes = 0
                self._startup_merge_failed_passes = 0
                self._startup_merge_bucket_count = 0
                self._startup_merge_updated_bucket_count = 0
                self._startup_merge_last_error = None
                self._startup_merge_task = asyncio.create_task(self._delayed_startup_history_merge())
        if self._history_store_day_writer() is not None:
            self._day_rollover_cursor_s = _day_start_epoch_s(self._time_fn())
        if self._history_store_is_writable():
            self._persistence_stop_requested = False
            self._persistence_task = asyncio.create_task(self._persistence_loop())

    async def stop(self) -> None:
        if self._startup_merge_task is not None and not self._startup_merge_task.done():
            self._startup_merge_task.cancel()
            try:
                await self._startup_merge_task
            except asyncio.CancelledError:
                pass
            self._startup_merge_task = None

        if self._restore_task is not None and not self._restore_task.done():
            self._restore_task.cancel()
            try:
                await self._restore_task
            except asyncio.CancelledError:
                pass
            self._restore_task = None

        if not self._history_store_is_writable():
            return

        self._persistence_stop_requested = True
        self._persistence_wakeup.set()
        if self._persistence_task is None or self._persistence_task.done():
            self._persistence_task = asyncio.create_task(self._persistence_loop())
        try:
            await asyncio.wait_for(
                asyncio.shield(self._persistence_task),
                timeout=self.shutdown_flush_budget_s,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Dashboard shutdown persistence exceeded its %.1fs budget; abandoning %s dirty buckets "
                "and continuing shutdown",
                self.shutdown_flush_budget_s,
                len(self._dirty_bucket_starts),
            )
        else:
            self._persistence_task = None

    async def record_sample(self, sample: SwarmStateSample) -> None:
        async with self._lock:
            bucket = self._get_bucket_unlocked(sample.captured_at_s)
            bucket.record_sample(sample)
            self._mark_bucket_dirty_unlocked(bucket.bucket_start_s)
            self._prune_unlocked(sample.captured_at_s)
            self._wake_persistence_unlocked()

    async def snapshot(self) -> list[SwarmHistoryBucket]:
        async with self._lock:
            return list(self._history.values())

    def history_restore_status(self) -> dict[str, object]:
        now_s = time.monotonic()
        started_at_s = self._history_restore_started_at_s
        finished_at_s = self._history_restore_finished_at_s
        return {
            "status": self._history_restore_status,
            "detail": self._history_restore_detail,
            "bucket_count": self._history_restore_bucket_count,
            "background": self.restore_history_in_background,
            "elapsed_s": (
                round((finished_at_s or now_s) - started_at_s, 2)
                if started_at_s is not None
                else None
            ),
        }

    def persistence_status(self) -> dict[str, object]:
        now_epoch_s = self._time_fn()
        oldest_dirty_bucket_start_s = min(self._dirty_bucket_starts, default=None)
        worker_alive = self._persistence_task is not None and not self._persistence_task.done()
        flush_write_in_flight = self._flush_write_task is not None and not self._flush_write_task.done()
        flush_task_age_s = None
        if flush_write_in_flight and self._flush_started_at_monotonic_s is not None:
            flush_task_age_s = round(
                max(time.monotonic() - self._flush_started_at_monotonic_s, 0.0),
                2,
            )
        flush_stalled_age_s = None
        if self._flush_stalled_started_at_monotonic_s is not None:
            flush_stalled_age_s = round(
                max(time.monotonic() - self._flush_stalled_started_at_monotonic_s, 0.0),
                2,
            )
        flush_retry_delay_remaining_s = self._flush_retry_delay_remaining_s()
        return {
            "enabled": self.history_store is not None,
            "read_only": bool(getattr(self.history_store, "read_only", False)),
            "worker_alive": worker_alive,
            "dirty_bucket_count": len(self._dirty_bucket_starts),
            "oldest_dirty_bucket_age_s": (
                round(max(now_epoch_s - oldest_dirty_bucket_start_s, 0.0), 2)
                if oldest_dirty_bucket_start_s is not None
                else None
            ),
            "last_flush_started_at": (
                _isoformat(self._last_flush_started_at_s)
                if self._last_flush_started_at_s is not None
                else None
            ),
            "last_flush_finished_at": (
                _isoformat(self._last_flush_finished_at_s)
                if self._last_flush_finished_at_s is not None
                else None
            ),
            "last_flush_error": self._last_flush_error,
            "flush_task_age_s": flush_task_age_s,
            "flush_write_in_flight": flush_write_in_flight,
            "flush_stalled": self._flush_stalled_started_at_monotonic_s is not None,
            "flush_stalled_since_at": (
                _isoformat(self._flush_stalled_started_at_s)
                if self._flush_stalled_started_at_s is not None
                else None
            ),
            "flush_stalled_age_s": flush_stalled_age_s,
            "flush_retry_backoff": self._flush_retry_not_before_monotonic_s is not None,
            "flush_retry_failure_count": self._flush_retry_failure_count,
            "flush_retry_delay_s": self._flush_retry_delay_s,
            "flush_retry_next_at": (
                _isoformat(self._flush_retry_next_at_s)
                if self._flush_retry_next_at_s is not None
                else None
            ),
            "flush_retry_delay_remaining_s": (
                round(flush_retry_delay_remaining_s, 2)
                if flush_retry_delay_remaining_s is not None
                else None
            ),
            "flush_retry_initial_delay_s": FLUSH_RETRY_INITIAL_DELAY_S,
            "flush_retry_max_delay_s": FLUSH_RETRY_MAX_DELAY_S,
            "flush_batch_size": self.flush_batch_size,
            "flush_timeout_s": self.flush_timeout_s,
            "dirty_bucket_warning_age_s": self.dirty_bucket_warning_age_s,
            "startup_merge": self.startup_merge_status(),
        }

    def startup_merge_status(self) -> dict[str, object]:
        return {
            "status": self._startup_merge_status,
            "delay_s": self.startup_merge_delay_s,
            "scheduled_passes": self._startup_merge_scheduled_passes,
            "attempted_passes": self._startup_merge_attempted_passes,
            "completed_passes": self._startup_merge_completed_passes,
            "failed_passes": self._startup_merge_failed_passes,
            "all_passes_completed": (
                self._startup_merge_scheduled_passes > 0
                and self._startup_merge_completed_passes == self._startup_merge_scheduled_passes
            ),
            "bucket_count": self._startup_merge_bucket_count,
            "updated_bucket_count": self._startup_merge_updated_bucket_count,
            "last_error": self._startup_merge_last_error,
        }


    async def _persistence_loop(self) -> None:
        while True:
            await self._wait_for_persistence_work()
            self._persistence_wakeup.clear()
            try:
                self._warn_if_dirty_buckets_stale_unlocked()
                await self._flush_dirty_buckets(
                    include_open_bucket=self._persistence_stop_requested,
                )
                await self._rollover_completed_days(flush_first=False)
                if self._persistence_stop_requested:
                    await self._flush_dirty_buckets(include_open_bucket=True)
                    await self._rollover_completed_days(flush_first=False)
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Dashboard persistence worker failed; retrying")
                await asyncio.sleep(PERSISTENCE_WORKER_RETRY_DELAY_S)
                self._persistence_wakeup.set()

    async def _wait_for_persistence_work(self) -> None:
        while True:
            retry_delay_s = self._flush_retry_delay_remaining_s()
            if retry_delay_s is None or self._persistence_stop_requested:
                await self._persistence_wakeup.wait()
                return
            if retry_delay_s <= 0:
                return

            self._persistence_wakeup.clear()
            try:
                await asyncio.wait_for(
                    self._persistence_wakeup.wait(),
                    timeout=retry_delay_s,
                )
            except asyncio.TimeoutError:
                return
            if self._persistence_stop_requested:
                return

    async def increment_counter(self, field_name: str) -> None:
        now = self._time_fn()
        async with self._lock:
            bucket = self._get_bucket_unlocked(now)
            setattr(bucket, field_name, getattr(bucket, field_name) + 1)
            self._mark_bucket_dirty_unlocked(bucket.bucket_start_s)
            self._prune_unlocked(now)
            self._wake_persistence_unlocked()

    async def record_requester_event(
        self,
        event: str,
        *,
        actor_id: str | None,
        metadata: dict[str, object] | None,
    ) -> None:
        counter_fields = {
            "request": ("session_requests", "requests"),
            "success": ("session_allocation_successes", "successes"),
            "failure": ("session_allocation_failures", "failures"),
            "abandoned": (None, "abandoned"),
        }
        if event not in counter_fields:
            raise ValueError(f"Unknown requester event: {event}")

        now = self._time_fn()
        async with self._lock:
            bucket = self._get_bucket_unlocked(now)
            global_field, requester_field = counter_fields[event]
            if global_field is not None:
                setattr(bucket, global_field, getattr(bucket, global_field) + 1)
            if actor_id and metadata is not None:
                resolved_actor_id = actor_id
                resolved_metadata = metadata
                if (
                    resolved_actor_id not in bucket.requester_usage
                    and len(bucket.requester_usage) >= self.max_requesters_per_bucket
                ):
                    resolved_actor_id = "overflow"
                    resolved_metadata = {
                        "label": "Other requesters (cardinality limit)",
                        "kind": "overflow",
                        "verification": "not_applicable",
                        "fingerprint": "",
                    }
                record = bucket.requester_usage.setdefault(
                    resolved_actor_id,
                    _new_requester_usage_record(resolved_metadata),
                )
                _merge_requester_identity(record, resolved_metadata)
                record[requester_field] = int(record.get(requester_field, 0)) + 1
                if event == "request" and resolved_actor_id != "overflow":
                    _record_request_context(record, resolved_metadata)
            self._mark_bucket_dirty_unlocked(bucket.bucket_start_s)
            self._prune_unlocked(now)
            self._wake_persistence_unlocked()

    async def update_requester_identity(
        self,
        actor_id: str,
        metadata: dict[str, object],
    ) -> None:
        async with self._lock:
            updated_bucket_starts: list[int] = []
            for bucket in self._history.values():
                record = bucket.requester_usage.get(actor_id)
                if record is None:
                    continue
                before = copy.deepcopy(record)
                _merge_requester_identity(record, metadata)
                if record != before:
                    updated_bucket_starts.append(bucket.bucket_start_s)
            for bucket_start_s in updated_bucket_starts:
                self._mark_bucket_dirty_unlocked(bucket_start_s)
            if updated_bucket_starts:
                self._wake_persistence_unlocked()

    async def record_completed_conversation(self, duration_s: float) -> None:
        now = self._time_fn()
        async with self._lock:
            bucket = self._get_bucket_unlocked(now)
            bucket.completed_conversations += 1
            bucket.completed_conversation_duration_total_s += duration_s
            bucket.completed_conversation_duration_max_s = max(
                bucket.completed_conversation_duration_max_s,
                duration_s,
            )
            bucket.completed_conversation_duration_samples_s.append(duration_s)
            self._mark_bucket_dirty_unlocked(bucket.bucket_start_s)
            self._prune_unlocked(now)
            self._wake_persistence_unlocked()

    def _get_bucket_unlocked(self, epoch_s: float) -> SwarmHistoryBucket:
        bucket_start_s = _bucket_start_epoch_s(epoch_s, 1)
        bucket = self._history.get(bucket_start_s)
        if bucket is None:
            bucket = SwarmHistoryBucket(bucket_start_s=bucket_start_s)
            self._history[bucket_start_s] = bucket
        return bucket

    def _mark_bucket_dirty_unlocked(self, bucket_start_s: int) -> None:
        self._locally_sampled_bucket_starts.add(bucket_start_s)
        if self._history_store_is_writable():
            self._dirty_bucket_starts.add(bucket_start_s)

    def _prune_unlocked(self, epoch_s: float) -> None:
        min_allowed_bucket = _bucket_start_epoch_s(epoch_s, 1) - (self.retention_minutes - 1) * 60
        while self._history:
            oldest_key = next(iter(self._history))
            if oldest_key >= min_allowed_bucket:
                break
            self._history.popitem(last=False)
            self._dirty_bucket_starts.discard(oldest_key)
            self._locally_sampled_bucket_starts.discard(oldest_key)

    async def _restore_history(self) -> None:
        if self.history_store is None:
            self._history_restore_status = "disabled"
            return

        started_s = time.monotonic()
        self._history_restore_started_at_s = started_s
        self._history_restore_finished_at_s = None
        self._history_restore_bucket_count = 0
        self._history_restore_status = "running"
        self._history_restore_detail = "Loading persisted dashboard history"
        try:
            buckets = await self._load_persisted_history()
        except asyncio.CancelledError:
            self._history_restore_status = "cancelled"
            self._history_restore_detail = "Dashboard history restore was cancelled"
            self._history_restore_finished_at_s = time.monotonic()
            raise
        except Exception as exc:
            self._history_restore_status = "failed"
            self._history_restore_detail = str(exc)
            self._history_restore_finished_at_s = time.monotonic()
            logger.warning("Failed to restore dashboard history from bucket store: %s", exc)
            return

        if not buckets:
            self._history_restore_status = "empty"
            self._history_restore_detail = "No persisted dashboard history found"
            self._history_restore_finished_at_s = time.monotonic()
            logger.info(
                "No persisted dashboard history restored after %.2fs",
                time.monotonic() - started_s,
            )
            return

        await self._merge_persisted_history_buckets(buckets)

        self._history_restore_status = "complete"
        self._history_restore_detail = "Persisted dashboard history restored"
        self._history_restore_bucket_count = len(buckets)
        self._history_restore_finished_at_s = time.monotonic()
        logger.info(
            "Restored %s persisted dashboard minute buckets in %.2fs",
            len(buckets),
            time.monotonic() - started_s,
        )

    async def _delayed_startup_history_merge(self) -> None:
        try:
            initial_restore_task = self._restore_task
            if self.history_store is None:
                self._startup_merge_status = "disabled"
                return

            for pass_number in range(1, self._startup_merge_scheduled_passes + 1):
                self._startup_merge_status = "waiting"
                await asyncio.sleep(self.startup_merge_delay_s)
                if (
                    pass_number == 1
                    and initial_restore_task is not None
                    and initial_restore_task is not asyncio.current_task()
                    and not initial_restore_task.done()
                ):
                    await initial_restore_task

                started_s = time.monotonic()
                self._startup_merge_status = "running"
                self._startup_merge_attempted_passes += 1
                try:
                    now_epoch_s = self._time_fn()
                    buckets = await self._load_persisted_history(
                        retention_minutes=min(
                            self.retention_minutes,
                            _startup_merge_retention_minutes(now_epoch_s),
                        ),
                        now_epoch_s=now_epoch_s,
                    )
                    updated_bucket_count = await self._merge_persisted_history_buckets(buckets)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._startup_merge_failed_passes += 1
                    self._startup_merge_last_error = str(exc)
                    logger.warning(
                        "Dashboard startup history merge pass %s/%s failed: %s",
                        pass_number,
                        self._startup_merge_scheduled_passes,
                        exc,
                    )
                    continue

                self._startup_merge_completed_passes += 1
                self._startup_merge_bucket_count += len(buckets)
                self._startup_merge_updated_bucket_count += updated_bucket_count
                logger.info(
                    "Completed dashboard startup history merge pass %s/%s: "
                    "loaded %s buckets, updated %s in %.2fs",
                    pass_number,
                    self._startup_merge_scheduled_passes,
                    len(buckets),
                    updated_bucket_count,
                    time.monotonic() - started_s,
                )
        except asyncio.CancelledError:
            self._startup_merge_status = "cancelled"
            raise

        if self._startup_merge_completed_passes == self._startup_merge_scheduled_passes:
            self._startup_merge_status = "complete"
        elif self._startup_merge_completed_passes > 0:
            self._startup_merge_status = "partial"
        else:
            self._startup_merge_status = "failed"

    async def _load_persisted_history(
        self,
        *,
        retention_minutes: Optional[int] = None,
        now_epoch_s: Optional[float] = None,
    ) -> list[SwarmHistoryBucket]:
        if self.history_store is None:
            return []
        return await asyncio.to_thread(
            self.history_store.load_recent,
            retention_minutes=(
                self.retention_minutes
                if retention_minutes is None
                else retention_minutes
            ),
            now_epoch_s=self._time_fn() if now_epoch_s is None else now_epoch_s,
        )

    async def _merge_persisted_history_buckets(self, buckets: list[SwarmHistoryBucket]) -> int:
        merge_candidates: list[tuple[SwarmHistoryBucket, Optional[SwarmHistoryBucket]]] = []
        sorted_buckets = sorted(buckets, key=lambda item: item.bucket_start_s)
        for chunk_start in range(0, len(sorted_buckets), HISTORY_MERGE_COMPARISON_CHUNK_SIZE):
            chunk = sorted_buckets[chunk_start:chunk_start + HISTORY_MERGE_COMPARISON_CHUNK_SIZE]
            serialized_buckets = [
                (bucket, bucket.to_dict()) for bucket in chunk
            ]
            async with self._lock:
                for bucket, serialized_bucket in serialized_buckets:
                    if (
                        bucket.bucket_start_s in self._dirty_bucket_starts
                        or bucket.bucket_start_s in self._locally_sampled_bucket_starts
                    ):
                        continue
                    current_bucket = self._history.get(bucket.bucket_start_s)
                    if current_bucket is None or current_bucket.to_dict() != serialized_bucket:
                        merge_candidates.append((bucket, current_bucket))
            await asyncio.sleep(0)

        updated_bucket_count = 0
        async with self._lock:
            for bucket, compared_bucket in merge_candidates:
                if (
                    bucket.bucket_start_s in self._dirty_bucket_starts
                    or bucket.bucket_start_s in self._locally_sampled_bucket_starts
                ):
                    continue
                current_bucket = self._history.get(bucket.bucket_start_s)
                if current_bucket is not compared_bucket:
                    if current_bucket is not None and current_bucket.to_dict() == bucket.to_dict():
                        continue
                self._history[bucket.bucket_start_s] = bucket
                updated_bucket_count += 1
            self._history = OrderedDict(sorted(self._history.items()))
            self._prune_unlocked(self._time_fn())
        return updated_bucket_count

    def _wake_persistence_unlocked(self) -> None:
        if not self._history_store_is_writable():
            return
        self._warn_if_dirty_buckets_stale_unlocked()
        self._persistence_wakeup.set()

    async def _flush_dirty_buckets(self, *, include_open_bucket: bool) -> None:
        if self.history_store is None:
            return
        if self._flush_retry_is_pending() and not self._persistence_stop_requested:
            return
        await self._wait_for_inflight_write()
        if self._flush_retry_is_pending() and not self._persistence_stop_requested:
            return

        flush_started = False
        try:
            while True:
                async with self._lock:
                    buckets = self._collect_dirty_buckets_unlocked(include_open_bucket=include_open_bucket)
                if not buckets:
                    if flush_started:
                        self._last_flush_finished_at_s = self._time_fn()
                    return

                if not flush_started:
                    flush_started = True
                    self._flush_retry_next_at_s = None
                    self._flush_retry_not_before_monotonic_s = None
                    self._last_flush_started_at_s = self._time_fn()
                    self._last_flush_finished_at_s = None
                    self._last_flush_error = None
                    self._flush_started_at_monotonic_s = time.monotonic()

                if not await self._write_bucket_batch(buckets):
                    return

                async with self._lock:
                    for bucket in buckets:
                        current_bucket = self._history.get(bucket.bucket_start_s)
                        if current_bucket is None or current_bucket.to_dict() == bucket.to_dict():
                            self._dirty_bucket_starts.discard(bucket.bucket_start_s)
        finally:
            self._flush_started_at_monotonic_s = None

    async def _wait_for_inflight_write(self) -> None:
        write_task = self._flush_write_task
        if write_task is None:
            return
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_flush_failure(exc)
            logger.warning("Failed dashboard history write finished before persistence resumed: %s", exc)
        finally:
            if self._flush_write_task is write_task and write_task.done():
                self._flush_write_task = None
                self._flush_stalled_started_at_s = None
                self._flush_stalled_started_at_monotonic_s = None

    async def _write_bucket_batch(self, buckets: list[SwarmHistoryBucket]) -> bool:
        if self.history_store is None:
            return False

        write_task = asyncio.create_task(
            asyncio.to_thread(self.history_store.write_buckets, buckets)
        )
        self._flush_write_task = write_task
        timed_out = False
        try:
            done, _ = await asyncio.wait({write_task}, timeout=self.flush_timeout_s)
            if not done:
                timed_out = True
                self._last_flush_error = (
                    f"Dashboard history flush stalled after {self.flush_timeout_s:g}s"
                )
                self._flush_stalled_started_at_s = self._time_fn()
                self._flush_stalled_started_at_monotonic_s = time.monotonic()
                logger.error(
                    "%s while persisting %s buckets; the single writer remains in flight",
                    self._last_flush_error,
                    len(buckets),
                )
                await asyncio.shield(write_task)
            else:
                write_task.result()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_flush_failure(exc)
            logger.warning("Failed to persist dashboard history to bucket store: %s", exc)
            return False
        finally:
            if self._flush_write_task is write_task and write_task.done():
                self._flush_write_task = None
            self._flush_stalled_started_at_s = None
            self._flush_stalled_started_at_monotonic_s = None

        if timed_out:
            logger.info("Stalled dashboard history flush completed; persistence can resume")
        self._reset_flush_retry_backoff()
        self._last_flush_error = None
        return True

    def _flush_retry_delay_remaining_s(self) -> Optional[float]:
        if self._flush_retry_not_before_monotonic_s is None:
            return None
        return max(self._flush_retry_not_before_monotonic_s - time.monotonic(), 0.0)

    def _flush_retry_is_pending(self) -> bool:
        retry_delay_s = self._flush_retry_delay_remaining_s()
        return retry_delay_s is not None and retry_delay_s > 0

    def _record_flush_failure(self, exc: Exception) -> None:
        self._flush_retry_failure_count += 1
        if self._flush_retry_delay_s is None:
            retry_delay_s = FLUSH_RETRY_INITIAL_DELAY_S
        else:
            retry_delay_s = min(self._flush_retry_delay_s * 2, FLUSH_RETRY_MAX_DELAY_S)
        self._flush_retry_delay_s = retry_delay_s
        self._flush_retry_next_at_s = self._time_fn() + retry_delay_s
        self._flush_retry_not_before_monotonic_s = time.monotonic() + retry_delay_s
        self._last_flush_error = str(exc)
        self._last_flush_finished_at_s = self._time_fn()

    def _reset_flush_retry_backoff(self) -> None:
        self._flush_retry_failure_count = 0
        self._flush_retry_delay_s = None
        self._flush_retry_next_at_s = None
        self._flush_retry_not_before_monotonic_s = None

    def _warn_if_dirty_buckets_stale_unlocked(self) -> None:
        oldest_dirty_bucket_start_s = min(self._dirty_bucket_starts, default=None)
        if oldest_dirty_bucket_start_s is None:
            return
        oldest_dirty_bucket_age_s = max(self._time_fn() - oldest_dirty_bucket_start_s, 0.0)
        if oldest_dirty_bucket_age_s < self.dirty_bucket_warning_age_s:
            return

        now_monotonic_s = time.monotonic()
        warning_interval_s = max(self.dirty_bucket_warning_age_s, 60.0)
        if (
            self._last_dirty_bucket_warning_at_monotonic_s is not None
            and now_monotonic_s - self._last_dirty_bucket_warning_at_monotonic_s < warning_interval_s
        ):
            return
        self._last_dirty_bucket_warning_at_monotonic_s = now_monotonic_s
        logger.warning(
            "Dashboard history persistence has %s dirty buckets; oldest dirty bucket age is %.1fs, "
            "flush task age is %s, last flush error is %s",
            len(self._dirty_bucket_starts),
            oldest_dirty_bucket_age_s,
            self.persistence_status()["flush_task_age_s"],
            self._last_flush_error,
        )

    def _history_store_is_writable(self) -> bool:
        return self.history_store is not None and not bool(getattr(self.history_store, "read_only", False))

    def _history_store_day_writer(self) -> Optional[Callable[..., Optional[str]]]:
        if not self._history_store_is_writable():
            return None
        writer = getattr(self.history_store, "write_day_buckets", None)
        return writer if callable(writer) else None

    async def _rollover_completed_days(self, *, flush_first: bool = True) -> None:
        writer = self._history_store_day_writer()
        if writer is None:
            return

        if flush_first:
            await self._flush_dirty_buckets(include_open_bucket=False)

        async with self._lock:
            current_day_start_s = _day_start_epoch_s(self._time_fn())
            if self._day_rollover_cursor_s is None:
                self._day_rollover_cursor_s = current_day_start_s
                return
            if self._day_rollover_cursor_s >= current_day_start_s:
                return

            day_starts = list(range(self._day_rollover_cursor_s, current_day_start_s, DAY_SECONDS))
            buckets_by_day: dict[int, list[SwarmHistoryBucket]] = {day_start: [] for day_start in day_starts}
            for bucket in self._history.values():
                day_start = _day_start_epoch_s(bucket.bucket_start_s)
                if day_start in buckets_by_day:
                    buckets_by_day[day_start].append(bucket)

        next_cursor_s = current_day_start_s
        for day_start in day_starts:
            day_buckets = sorted(buckets_by_day.get(day_start, []), key=lambda bucket: bucket.bucket_start_s)
            if not day_buckets:
                logger.info("Skipping live dashboard day rollover for %s because no minute buckets are in memory", _day_key(day_start))
                continue
            if len(day_buckets) < DAY_MINUTES:
                logger.warning(
                    "Finalizing partial live dashboard day rollover for %s with only %s of 1440 minute buckets",
                    _day_key(day_start),
                    len(day_buckets),
                )

            try:
                path = await asyncio.to_thread(writer, day_start_s=day_start, buckets=day_buckets)
            except Exception as exc:
                logger.warning("Failed to cache live dashboard history day %s: %s", _day_key(day_start), exc)
                next_cursor_s = day_start
                break
            if path:
                logger.info("Cached live dashboard history day %s at %s", _day_key(day_start), path)

        async with self._lock:
            if self._day_rollover_cursor_s is None or self._day_rollover_cursor_s < next_cursor_s:
                self._day_rollover_cursor_s = next_cursor_s

    def _collect_dirty_buckets_unlocked(self, *, include_open_bucket: bool) -> list[SwarmHistoryBucket]:
        current_bucket_start_s = _bucket_start_epoch_s(self._time_fn(), 1)
        return [
            SwarmHistoryBucket.from_dict(self._history[bucket_start_s].to_dict())
            for bucket_start_s in sorted(self._dirty_bucket_starts)
            if bucket_start_s in self._history
            and (include_open_bucket or bucket_start_s < current_bucket_start_s)
        ][: self.flush_batch_size]

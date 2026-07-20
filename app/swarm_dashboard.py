import asyncio
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Awaitable, Callable, Iterable, Optional, Protocol


SnapshotProvider = Callable[[], Awaitable[tuple[bool, Optional[str], dict[str, object]]]]
logger = logging.getLogger("s2s-endpoint")
ROLLING_VIEW_WINDOWS: tuple[tuple[str, int], ...] = (
    ("1h", 60),
    ("6h", 6 * 60),
    ("24h", 24 * 60),
)
DAY_MINUTES = 24 * 60
DAY_SECONDS = DAY_MINUTES * 60


def _normalize_status(status: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(status).lower())


def _bucket_start_epoch_s(epoch_s: float, bucket_minutes: int) -> int:
    bucket_seconds = bucket_minutes * 60
    return int(epoch_s // bucket_seconds) * bucket_seconds


def _day_start_epoch_s(epoch_s: float) -> int:
    return _bucket_start_epoch_s(epoch_s, DAY_MINUTES)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def _day_key(epoch_s: int | float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y-%m-%d")


def _isoformat(epoch_s: int | float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_window_minutes(window: str | None) -> int:
    value = (window or "6h").strip().lower()
    match = re.fullmatch(r"(\d+)(m|h|d)", value)
    if not match:
        raise ValueError("window must look like 60m, 6h, or 7d")

    amount = int(match.group(1))
    unit = match.group(2)
    if amount < 1:
        raise ValueError("window amount must be >= 1")

    if unit == "m":
        return amount
    if unit == "h":
        return amount * 60
    return amount * 24 * 60


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
            payload[bucket_field.name] = list(value) if isinstance(value, list) else value
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
    raise KeyError(f"Unknown SwarmHistoryBucket field: {name}")


@dataclass
class SwarmBucketAggregate:
    sample_count: int = 0
    running_endpoints_sum: float = 0.0
    warming_endpoints_sum: float = 0.0
    transitioning_endpoints_sum: float = 0.0
    parked_endpoints_sum: float = 0.0
    connected_sessions_sum: float = 0.0
    connected_sessions_max: int = 0
    pending_sessions_sum: float = 0.0
    free_slots_sum: float = 0.0
    effective_free_slots_sum: float = 0.0
    router_active_sessions_sum: float = 0.0
    healthy_samples: int = 0
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
    peak_running_endpoints: int = 0
    active_conversation_minutes: float = 0.0

    @classmethod
    def from_buckets(cls, buckets: Iterable[SwarmHistoryBucket]) -> "SwarmBucketAggregate":
        aggregate = cls()
        for bucket in buckets:
            aggregate.sample_count += bucket.sample_count
            aggregate.running_endpoints_sum += bucket.running_endpoints_sum
            aggregate.warming_endpoints_sum += bucket.warming_endpoints_sum
            aggregate.transitioning_endpoints_sum += bucket.transitioning_endpoints_sum
            aggregate.parked_endpoints_sum += bucket.parked_endpoints_sum
            aggregate.connected_sessions_sum += bucket.connected_sessions_sum
            aggregate.connected_sessions_max = max(aggregate.connected_sessions_max, bucket.connected_sessions_max)
            aggregate.pending_sessions_sum += bucket.pending_sessions_sum
            aggregate.free_slots_sum += bucket.free_slots_sum
            aggregate.effective_free_slots_sum += bucket.effective_free_slots_sum
            aggregate.router_active_sessions_sum += bucket.router_active_sessions_sum
            aggregate.healthy_samples += bucket.healthy_samples
            aggregate.errors_count_sum += bucket.errors_count_sum
            aggregate.session_requests += bucket.session_requests
            aggregate.session_allocation_successes += bucket.session_allocation_successes
            aggregate.session_allocation_failures += bucket.session_allocation_failures
            aggregate.session_connected_events += bucket.session_connected_events
            aggregate.session_disconnected_events += bucket.session_disconnected_events
            aggregate.completed_conversations += bucket.completed_conversations
            aggregate.completed_conversation_duration_total_s += bucket.completed_conversation_duration_total_s
            aggregate.completed_conversation_duration_max_s = max(
                aggregate.completed_conversation_duration_max_s,
                bucket.completed_conversation_duration_max_s,
            )
            aggregate.completed_conversation_duration_samples_s.extend(bucket.completed_duration_samples())
            aggregate.peak_running_endpoints = max(aggregate.peak_running_endpoints, bucket.running_endpoints_last)
            if bucket.sample_count:
                aggregate.active_conversation_minutes += bucket.connected_sessions_sum / bucket.sample_count
        return aggregate

    def sample_average(self, value_sum: float) -> float:
        return round(value_sum / self.sample_count, 2) if self.sample_count else 0.0

    @property
    def avg_conversation_duration_s(self) -> float:
        if not self.completed_conversations:
            return 0.0
        return round(self.completed_conversation_duration_total_s / self.completed_conversations, 2)

    @property
    def median_conversation_duration_s(self) -> float:
        return round(_median(self.completed_conversation_duration_samples_s), 2)

    def as_summary_dict(self) -> dict[str, object]:
        return {
            "session_requests": self.session_requests,
            "session_allocation_successes": self.session_allocation_successes,
            "session_allocation_failures": self.session_allocation_failures,
            "session_connected_events": self.session_connected_events,
            "session_disconnected_events": self.session_disconnected_events,
            "completed_conversations": self.completed_conversations,
            "active_conversation_minutes": round(self.active_conversation_minutes, 2),
            "active_conversation_hours": round(self.active_conversation_minutes / 60.0, 2),
            "active_conversation_days": round(self.active_conversation_minutes / (24.0 * 60.0), 3),
            "avg_conversation_duration_s": self.avg_conversation_duration_s,
            "max_conversation_duration_s": round(self.completed_conversation_duration_max_s, 2),
            "peak_connected_sessions": self.connected_sessions_max,
            "peak_running_endpoints": self.peak_running_endpoints,
        }

    def as_hourly_point(self, timestamp_s: int) -> dict[str, object]:
        return {
            "timestamp": _isoformat(timestamp_s),
            "running_endpoints": self.sample_average(self.running_endpoints_sum),
            "warming_endpoints": self.sample_average(self.warming_endpoints_sum),
            "transitioning_endpoints": self.sample_average(self.transitioning_endpoints_sum),
            "parked_endpoints": self.sample_average(self.parked_endpoints_sum),
            "connected_sessions": self.sample_average(self.connected_sessions_sum),
            "pending_sessions": self.sample_average(self.pending_sessions_sum),
            "free_slots": self.sample_average(self.free_slots_sum),
            "effective_free_slots": self.sample_average(self.effective_free_slots_sum),
            "router_active_sessions": self.sample_average(self.router_active_sessions_sum),
            "errors_count": self.sample_average(self.errors_count_sum),
            "healthy": (self.healthy_samples / self.sample_count) >= 0.5 if self.sample_count else False,
            "session_requests": self.session_requests,
            "session_allocation_successes": self.session_allocation_successes,
            "session_allocation_failures": self.session_allocation_failures,
            "session_connected_events": self.session_connected_events,
            "session_disconnected_events": self.session_disconnected_events,
            "completed_conversations": self.completed_conversations,
            "avg_conversation_duration_s": self.avg_conversation_duration_s,
            "avg_conversation_duration_min": round(self.avg_conversation_duration_s / 60.0, 2),
            "max_conversation_duration_s": round(self.completed_conversation_duration_max_s, 2),
            "max_conversation_duration_min": round(self.completed_conversation_duration_max_s / 60.0, 2),
        }

    def as_rolling_fields(self, label: str) -> dict[str, object]:
        return {
            f"completed_conversations_{label}": self.completed_conversations,
            f"active_conversation_minutes_{label}": round(self.active_conversation_minutes, 2),
            f"active_conversation_hours_{label}": round(self.active_conversation_minutes / 60.0, 2),
            f"active_conversation_days_{label}": round(self.active_conversation_minutes / (24.0 * 60.0), 3),
            f"avg_conversation_duration_s_{label}": self.avg_conversation_duration_s,
            f"avg_conversation_duration_min_{label}": round(self.avg_conversation_duration_s / 60.0, 2),
            f"connected_sessions_avg_{label}": self.sample_average(self.connected_sessions_sum),
            f"connected_sessions_max_{label}": self.connected_sessions_max,
            f"median_conversation_duration_s_{label}": self.median_conversation_duration_s,
            f"median_conversation_duration_min_{label}": round(self.median_conversation_duration_s / 60.0, 2),
        }



class SwarmDashboard:
    def __init__(
        self,
        *,
        snapshot_provider: SnapshotProvider,
        sample_interval_s: float = 15.0,
        retention_minutes: int = 28 * 24 * 60,
        history_store: Optional[DashboardHistoryStore] = None,
        restore_history_in_background: bool = False,
        flush_batch_size: int = 100,
        flush_timeout_s: float = 60.0,
        dirty_bucket_warning_age_s: float = 300.0,
        startup_merge_delay_s: float = 60.0,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
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

        self.snapshot_provider = snapshot_provider
        self.sample_interval_s = sample_interval_s
        self.retention_minutes = retention_minutes
        self.history_store = history_store
        self.restore_history_in_background = restore_history_in_background
        self.flush_batch_size = flush_batch_size
        self.flush_timeout_s = flush_timeout_s
        self.shutdown_flush_budget_s = 2 * flush_timeout_s
        self.dirty_bucket_warning_age_s = dirty_bucket_warning_age_s
        self.startup_merge_delay_s = startup_merge_delay_s
        self._time_fn = time_fn
        self._lock = asyncio.Lock()
        self._history: "OrderedDict[int, SwarmHistoryBucket]" = OrderedDict()
        self._latest_sample: Optional[SwarmStateSample] = None
        self._sample_task: Optional[asyncio.Task] = None
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
        self._startup_merge_bucket_count = 0
        self._startup_merge_updated_bucket_count = 0

    async def start(self) -> None:
        await self.capture_sample()
        if self.history_store is not None:
            if self.restore_history_in_background:
                logger.info("Restoring dashboard history in the background")
                self._history_restore_status = "running"
                self._history_restore_detail = "Loading persisted dashboard history"
                self._restore_task = asyncio.create_task(self._restore_history())
            else:
                await self._restore_history()
            if self.startup_merge_delay_s > 0:
                self._startup_merge_status = "waiting"
                self._startup_merge_task = asyncio.create_task(self._delayed_startup_history_merge())
        if self._history_store_day_writer() is not None:
            self._day_rollover_cursor_s = _day_start_epoch_s(self._time_fn())
        if self._history_store_is_writable():
            self._persistence_stop_requested = False
            self._persistence_task = asyncio.create_task(self._persistence_loop())
        self._sample_task = asyncio.create_task(self._sample_loop())

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

        if self._sample_task is not None:
            self._sample_task.cancel()
            try:
                await self._sample_task
            except asyncio.CancelledError:
                pass
            self._sample_task = None

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

    async def capture_sample(self) -> SwarmStateSample:
        healthy, detail, snapshot = await self.snapshot_provider()
        sample = SwarmStateSample.from_health_snapshot(
            healthy=healthy,
            detail=detail,
            snapshot=snapshot,
            captured_at_s=self._time_fn(),
        )
        await self.record_sample(sample)
        return sample

    async def record_sample(self, sample: SwarmStateSample) -> None:
        async with self._lock:
            bucket = self._get_bucket_unlocked(sample.captured_at_s)
            bucket.record_sample(sample)
            self._latest_sample = sample
            self._mark_bucket_dirty_unlocked(bucket.bucket_start_s)
            self._prune_unlocked(sample.captured_at_s)
            self._wake_persistence_unlocked()

    async def record_session_request(self) -> None:
        await self._increment_counter("session_requests")

    async def record_session_allocation_success(self) -> None:
        await self._increment_counter("session_allocation_successes")

    async def record_session_allocation_failure(self) -> None:
        await self._increment_counter("session_allocation_failures")

    async def record_session_event(
        self,
        event: str,
        *,
        conversation_duration_s: Optional[float] = None,
        conversation_counted: bool = False,
    ) -> None:
        if event == "connected":
            await self._increment_counter("session_connected_events")
        elif event == "disconnected":
            await self._increment_counter("session_disconnected_events")
            if conversation_counted:
                await self._record_completed_conversation(max(float(conversation_duration_s or 0.0), 0.0))

    async def live_sample(self) -> SwarmStateSample:
        return await self.capture_sample()

    async def data(self, *, window: str | None, resolution: str | None) -> dict[str, object]:
        window_minutes = _parse_window_minutes(window)
        resolved_resolution = (resolution or "").strip().lower() or (
            "minute" if window_minutes <= 24 * 60 else "hour"
        )
        if resolved_resolution not in {"minute", "hour"}:
            raise ValueError("resolution must be 'minute' or 'hour'")

        current = await self.live_sample()
        series = await self.series(window_minutes=window_minutes, resolution=resolved_resolution)
        rolling_series = await self.rolling_series(window_minutes=window_minutes, resolution=resolved_resolution)
        summary = await self.summary(window_minutes=window_minutes, requested_window=window or "6h")

        return {
            "generated_at": _isoformat(self._time_fn()),
            "window": {
                "requested": window or "6h",
                "minutes": window_minutes,
                "resolution": resolved_resolution,
            },
            "current": current.to_dict(),
            "summary": summary,
            "series": series,
            "rolling_windows": [
                {"label": label, "minutes": minutes}
                for label, minutes in ROLLING_VIEW_WINDOWS
            ],
            "rolling_series": rolling_series,
            "retention_minutes": self.retention_minutes,
            "history_restore": self.history_restore_status(),
            "history_persistence": self.persistence_status(),
        }

    async def summary(self, *, window_minutes: int, requested_window: str) -> dict[str, object]:
        async with self._lock:
            latest = self._latest_sample
            minute_buckets = list(self._history.values())

        selected = self._aggregate_recent(minute_buckets, window_minutes=window_minutes)

        return {
            "current": latest.to_dict() if latest is not None else None,
            "window_label": requested_window,
            "window_minutes": window_minutes,
            "session_requests_window": selected["session_requests"],
            "session_failures_window": selected["session_allocation_failures"],
            "session_successes_window": selected["session_allocation_successes"],
            "session_connects_window": selected["session_connected_events"],
            "session_disconnects_window": selected["session_disconnected_events"],
            "conversations_started_window": selected["session_connected_events"],
            "conversations_completed_window": selected["completed_conversations"],
            "active_conversation_minutes_window": selected["active_conversation_minutes"],
            "active_conversation_hours_window": selected["active_conversation_hours"],
            "active_conversation_days_window": selected["active_conversation_days"],
            "avg_conversation_duration_window_s": selected["avg_conversation_duration_s"],
            "avg_conversation_duration_window_min": round(selected["avg_conversation_duration_s"] / 60.0, 2),
            "max_conversation_duration_window_s": selected["max_conversation_duration_s"],
            "max_conversation_duration_window_min": round(selected["max_conversation_duration_s"] / 60.0, 2),
            "peak_connected_sessions_window": selected["peak_connected_sessions"],
            "peak_running_endpoints_window": selected["peak_running_endpoints"],
        }

    async def series(self, *, window_minutes: int, resolution: str) -> list[dict[str, object]]:
        async with self._lock:
            minute_buckets = list(self._history.values())

        now = self._time_fn()
        end_bucket = _bucket_start_epoch_s(now, 1)
        start_bucket = end_bucket - (window_minutes - 1) * 60
        minute_map = {
            bucket.bucket_start_s: bucket
            for bucket in minute_buckets
            if bucket.bucket_start_s >= start_bucket
        }

        if resolution == "minute":
            points = []
            for bucket_start in range(start_bucket, end_bucket + 1, 60):
                bucket = minute_map.get(bucket_start)
                if bucket is None:
                    points.append(
                        {
                            "timestamp": _isoformat(bucket_start),
                            "running_endpoints": 0,
                            "warming_endpoints": 0,
                            "transitioning_endpoints": 0,
                            "parked_endpoints": 0,
                            "connected_sessions": 0,
                            "pending_sessions": 0,
                            "free_slots": 0,
                            "effective_free_slots": 0,
                            "router_active_sessions": 0,
                            "errors_count": 0,
                            "healthy": False,
                            "session_requests": 0,
                            "session_allocation_successes": 0,
                            "session_allocation_failures": 0,
                            "session_connected_events": 0,
                            "session_disconnected_events": 0,
                            "completed_conversations": 0,
                            "avg_conversation_duration_s": 0.0,
                            "avg_conversation_duration_min": 0.0,
                            "max_conversation_duration_s": 0.0,
                            "max_conversation_duration_min": 0.0,
                        }
                    )
                else:
                    points.append(bucket.as_minute_point())
            return points

        return self._aggregate_hourly(minute_map, start_bucket, end_bucket)

    async def rolling_series(self, *, window_minutes: int, resolution: str) -> list[dict[str, object]]:
        async with self._lock:
            minute_buckets = list(self._history.values())

        now = self._time_fn()
        end_bucket = _bucket_start_epoch_s(now, 1)
        start_bucket = end_bucket - (window_minutes - 1) * 60
        max_window_minutes = max(minutes for _, minutes in ROLLING_VIEW_WINDOWS)
        context_start_bucket = start_bucket - (max_window_minutes - 1) * 60
        minute_map = {
            bucket.bucket_start_s: bucket
            for bucket in minute_buckets
            if context_start_bucket <= bucket.bucket_start_s <= end_bucket
        }

        minute_starts = list(range(context_start_bucket, end_bucket + 1, 60))
        bucket_sequence = [minute_map.get(bucket_start_s) for bucket_start_s in minute_starts]

        def range_indices(start_s: int, end_s: int) -> tuple[int, int] | None:
            if start_s > end_bucket or end_s < context_start_bucket:
                return None
            bounded_start_s = max(start_s, context_start_bucket)
            bounded_end_s = min(end_s, end_bucket)
            return (
                int((bounded_start_s - context_start_bucket) // 60),
                int((bounded_end_s - context_start_bucket) // 60),
            )

        def range_buckets(start_s: int, end_s: int) -> list[SwarmHistoryBucket]:
            indices = range_indices(start_s, end_s)
            if indices is None:
                return []
            start_index, end_index = indices
            return [bucket for bucket in bucket_sequence[start_index : end_index + 1] if bucket is not None]

        step_s = 60 if resolution == "minute" else 3600
        first_point_s = start_bucket if resolution == "minute" else _bucket_start_epoch_s(start_bucket, 60)
        points = []
        for point_start_s in range(first_point_s, end_bucket + 1, step_s):
            point_end_s = point_start_s if resolution == "minute" else min(point_start_s + 59 * 60, end_bucket)
            point: dict[str, object] = {"timestamp": _isoformat(point_start_s)}
            for label, minutes in ROLLING_VIEW_WINDOWS:
                window_start_s = point_end_s - (minutes - 1) * 60
                aggregate = SwarmBucketAggregate.from_buckets(range_buckets(window_start_s, point_end_s))
                point.update(aggregate.as_rolling_fields(label))
            points.append(point)

        return points

    def html(self) -> str:
        return _dashboard_html(history_persisted=self.history_store is not None)

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
        return {
            "enabled": self.history_store is not None,
            "read_only": bool(getattr(self.history_store, "read_only", False)),
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
            "flush_batch_size": self.flush_batch_size,
            "flush_timeout_s": self.flush_timeout_s,
            "dirty_bucket_warning_age_s": self.dirty_bucket_warning_age_s,
            "startup_merge": self.startup_merge_status(),
        }

    def startup_merge_status(self) -> dict[str, object]:
        return {
            "status": self._startup_merge_status,
            "delay_s": self.startup_merge_delay_s,
            "bucket_count": self._startup_merge_bucket_count,
            "updated_bucket_count": self._startup_merge_updated_bucket_count,
        }

    async def _sample_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.sample_interval_s)
                await self.capture_sample()
        except asyncio.CancelledError:
            raise

    async def _persistence_loop(self) -> None:
        while True:
            await self._persistence_wakeup.wait()
            self._persistence_wakeup.clear()
            self._warn_if_dirty_buckets_stale_unlocked()
            await self._flush_dirty_buckets(
                include_open_bucket=self._persistence_stop_requested,
            )
            await self._rollover_completed_days(flush_first=False)
            if self._persistence_stop_requested:
                return

    async def _increment_counter(self, field_name: str) -> None:
        now = self._time_fn()
        async with self._lock:
            bucket = self._get_bucket_unlocked(now)
            setattr(bucket, field_name, getattr(bucket, field_name) + 1)
            self._mark_bucket_dirty_unlocked(bucket.bucket_start_s)
            self._prune_unlocked(now)
            self._wake_persistence_unlocked()

    async def _record_completed_conversation(self, duration_s: float) -> None:
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

    def _aggregate_recent(self, minute_buckets: list[SwarmHistoryBucket], *, window_minutes: int) -> dict[str, object]:
        now = self._time_fn()
        min_bucket = _bucket_start_epoch_s(now, 1) - (window_minutes - 1) * 60
        selected = [bucket for bucket in minute_buckets if bucket.bucket_start_s >= min_bucket]
        return SwarmBucketAggregate.from_buckets(selected).as_summary_dict()

    def _aggregate_hourly(
        self,
        minute_map: dict[int, SwarmHistoryBucket],
        start_bucket_s: int,
        end_bucket_s: int,
    ) -> list[dict[str, object]]:
        points = []
        current_hour_s = _bucket_start_epoch_s(start_bucket_s, 60)
        end_hour_s = _bucket_start_epoch_s(end_bucket_s, 60)

        while current_hour_s <= end_hour_s:
            minute_buckets = [
                minute_map.get(minute_bucket_s)
                for minute_bucket_s in range(current_hour_s, current_hour_s + 3600, 60)
                if start_bucket_s <= minute_bucket_s <= end_bucket_s
            ]
            minute_buckets = [bucket for bucket in minute_buckets if bucket is not None]

            points.append(SwarmBucketAggregate.from_buckets(minute_buckets).as_hourly_point(current_hour_s))
            current_hour_s += 3600

        return points

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
            await asyncio.sleep(self.startup_merge_delay_s)
            initial_restore_task = self._restore_task
            if (
                initial_restore_task is not None
                and initial_restore_task is not asyncio.current_task()
                and not initial_restore_task.done()
            ):
                await initial_restore_task

            if self.history_store is None:
                self._startup_merge_status = "disabled"
                return

            started_s = time.monotonic()
            self._startup_merge_bucket_count = 0
            self._startup_merge_updated_bucket_count = 0
            self._startup_merge_status = "running"
            buckets = await self._load_persisted_history()
            updated_bucket_count = await self._merge_persisted_history_buckets(buckets)
        except asyncio.CancelledError:
            self._startup_merge_status = "cancelled"
            raise
        except Exception as exc:
            self._startup_merge_status = "failed"
            logger.warning("Failed to merge delayed dashboard history from bucket store: %s", exc)
            return

        self._startup_merge_bucket_count = len(buckets)
        self._startup_merge_updated_bucket_count = updated_bucket_count
        self._startup_merge_status = "complete" if buckets else "empty"
        logger.info(
            "Merged delayed dashboard history: loaded %s buckets, updated %s in %.2fs",
            len(buckets),
            updated_bucket_count,
            time.monotonic() - started_s,
        )

    async def _load_persisted_history(self) -> list[SwarmHistoryBucket]:
        if self.history_store is None:
            return []
        return await asyncio.to_thread(
            self.history_store.load_recent,
            retention_minutes=self.retention_minutes,
            now_epoch_s=self._time_fn(),
        )

    async def _merge_persisted_history_buckets(self, buckets: list[SwarmHistoryBucket]) -> int:
        updated_bucket_count = 0
        async with self._lock:
            for bucket in sorted(buckets, key=lambda item: item.bucket_start_s):
                if (
                    bucket.bucket_start_s in self._dirty_bucket_starts
                    or bucket.bucket_start_s in self._locally_sampled_bucket_starts
                ):
                    continue
                current_bucket = self._history.get(bucket.bucket_start_s)
                if current_bucket is None or current_bucket.to_dict() != bucket.to_dict():
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
        await self._wait_for_inflight_write()

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
            self._last_flush_error = str(exc)
            self._last_flush_finished_at_s = self._time_fn()
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
            self._last_flush_error = str(exc)
            self._last_flush_finished_at_s = self._time_fn()
            logger.warning("Failed to persist dashboard history to bucket store: %s", exc)
            return False
        finally:
            if self._flush_write_task is write_task and write_task.done():
                self._flush_write_task = None
            self._flush_stalled_started_at_s = None
            self._flush_stalled_started_at_monotonic_s = None

        if timed_out:
            logger.info("Stalled dashboard history flush completed; persistence can resume")
        self._last_flush_error = None
        return True

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


def _dashboard_html(*, history_persisted: bool = False) -> str:
    history_note = (
        "Minute history persists to bucket storage across restarts."
        if history_persisted
        else "Single-replica LB in-memory history; resets on restart."
    )
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>s2s Swarm Dashboard</title>
  <style>
    :root {
      --bg: #f4f1e8;
      --surface: rgba(255, 255, 255, 0.82);
      --surface-strong: rgba(255, 255, 255, 0.94);
      --ink: #182125;
      --muted: #5f6d73;
      --line: rgba(24, 33, 37, 0.14);
      --good: #117a65;
      --accent: #0b5cab;
      --warm: #d9822b;
      --danger: #bb2d3b;
      --shadow: 0 18px 40px rgba(24, 33, 37, 0.10);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(11, 92, 171, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(217, 130, 43, 0.14), transparent 22%),
        linear-gradient(180deg, #f8f5ed 0%, #ece6d9 100%);
    }

    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }

    .hero {
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }

    .panel {
      background: var(--surface);
      backdrop-filter: blur(12px);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .hero-copy {
      padding: 26px 28px 22px;
      min-height: 180px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .eyebrow, .label {
      font-family: "Menlo", "Consolas", monospace;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }

    h1 {
      margin: 10px 0 14px;
      font-size: clamp(32px, 5vw, 56px);
      line-height: 0.95;
      letter-spacing: -0.04em;
      max-width: 12ch;
    }

    .hero-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
    }

    .hero-stat-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1px;
      background: var(--line);
    }

    .hero-stat {
      background: var(--surface-strong);
      padding: 20px 22px;
      min-height: 92px;
    }

    .hero-stat strong {
      display: block;
      margin-top: 6px;
      font-size: 28px;
      line-height: 1;
      letter-spacing: -0.04em;
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
    }

    .toggle {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.72);
      padding: 8px 12px;
      font: inherit;
      color: var(--ink);
      cursor: pointer;
      transition: 120ms ease;
    }

    .toggle.active {
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }

    .toggle:disabled {
      cursor: not-allowed;
      opacity: 0.42;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
    }

    .span-4 { grid-column: span 4; }
    .span-6 { grid-column: span 6; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }

    .card {
      padding: 20px 22px 18px;
    }

    .card h2 {
      margin: 6px 0 10px;
      font-size: 24px;
      letter-spacing: -0.04em;
    }

    .kpis {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }

    .rolling-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }

    .kpi strong {
      display: block;
      margin-top: 12px;
      font-size: 34px;
      letter-spacing: -0.05em;
    }

    .kpi small {
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .chart-shell {
      padding-top: 4px;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin: 10px 0 12px;
      color: var(--muted);
      font-size: 13px;
    }

    .legend span::before,
    .legend-toggle::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 8px;
      background: currentColor;
      vertical-align: middle;
    }

    .legend-toggle {
      display: inline-flex;
      align-items: center;
      border: 0;
      background: transparent;
      padding: 0;
      font: inherit;
      color: inherit;
      cursor: pointer;
    }

    .legend-toggle.hidden {
      opacity: 0.36;
      text-decoration: line-through;
    }

    .legend-toggle:focus-visible {
      outline: 2px solid currentColor;
      outline-offset: 3px;
      border-radius: 4px;
    }

    canvas {
      width: 100%;
      height: 260px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.36));
      border: 1px solid rgba(24, 33, 37, 0.08);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }

    th, td {
      text-align: left;
      padding: 12px 10px;
      border-top: 1px solid var(--line);
      vertical-align: top;
    }

    th {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--muted);
      border-top: 0;
      padding-top: 0;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      background: rgba(24, 33, 37, 0.06);
    }

    .status-pill.good { color: var(--good); background: rgba(17, 122, 101, 0.10); }
    .status-pill.warm { color: var(--warm); background: rgba(217, 130, 43, 0.12); }
    .status-pill.bad { color: var(--danger); background: rgba(187, 45, 59, 0.10); }

    .fleet-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 4px 0 14px;
    }

    .fleet-wall {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 14px;
    }

    .endpoint-tile {
      position: relative;
      border: 1px solid rgba(24, 33, 37, 0.10);
      border-radius: 18px;
      padding: 16px 16px 14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,255,255,0.64));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.4);
    }

    .endpoint-tile::before {
      content: "";
      position: absolute;
      inset: 0 auto 0 0;
      width: 6px;
      border-radius: 18px 0 0 18px;
      background: rgba(24, 33, 37, 0.12);
    }

    .endpoint-tile.state-running::before { background: var(--good); }
    .endpoint-tile.state-warm::before { background: var(--warm); }
    .endpoint-tile.state-parked::before { background: #8b969b; }
    .endpoint-tile.state-error::before { background: var(--danger); }

    .endpoint-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
    }

    .endpoint-name {
      font-family: "Menlo", "Consolas", monospace;
      font-size: 13px;
      line-height: 1.35;
      word-break: break-word;
    }

    .endpoint-badges {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 6px;
    }

    .tiny-pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 600;
      background: rgba(24, 33, 37, 0.06);
      color: var(--muted);
    }

    .endpoint-stats {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }

    .endpoint-stat-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }

    .endpoint-stat-value {
      margin-top: 4px;
      font-size: 24px;
      line-height: 1;
      letter-spacing: -0.05em;
    }

    .endpoint-meter {
      margin-top: 14px;
      height: 10px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(24, 33, 37, 0.08);
    }

    .endpoint-meter-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #0b5cab, #117a65);
    }

    .endpoint-flags {
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
      min-height: 16px;
    }

    .mini-chart-shell {
      margin-top: 16px;
      display: grid;
      grid-template-columns: minmax(0, 200px) 1fr;
      gap: 14px;
      align-items: center;
    }

    .mini-chart-shell canvas {
      height: 200px;
      background: transparent;
      border: 0;
    }

    .mini-legend {
      display: grid;
      gap: 8px;
      align-content: center;
    }

    .mini-legend-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
    }

    .mini-legend-label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
    }

    .mini-legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }

    .muted { color: var(--muted); }
    .mono { font-family: "Menlo", "Consolas", monospace; }
    .footer-note { margin-top: 14px; color: var(--muted); font-size: 13px; }

    @media (max-width: 1100px) {
      .hero, .kpis, .rolling-grid { grid-template-columns: 1fr; }
      .span-4, .span-6, .span-8 { grid-column: span 12; }
    }

    @media (max-width: 720px) {
      .mini-chart-shell {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel hero-copy">
        <div>
          <div class="eyebrow">s2s / swarm control room</div>
          <h1>Realtime view of the live endpoint fleet.</h1>
          <div class="hero-meta">
            <span id="generated-at">Loading...</span>
            <span>Minute history rolls up into hourly trends automatically.</span>
          </div>
        </div>
        <div class="hero-meta">
          <span class="mono">/dashboard</span>
          <span>__HISTORY_NOTE__</span>
        </div>
      </div>
      <div class="panel hero-stat-grid" id="hero-stats"></div>
    </section>

    <div class="controls">
      <button class="toggle" data-window="1h">1h</button>
      <button class="toggle active" data-window="6h">6h</button>
      <button class="toggle" data-window="24h">24h</button>
      <button class="toggle" data-window="7d">7d</button>
      <button class="toggle" data-window="14d">14d</button>
      <button class="toggle" data-window="28d">28d</button>
      <button class="toggle" data-resolution="minute">Minute</button>
      <button class="toggle" data-resolution="hour">Hour</button>
    </div>

    <section class="rolling-grid" id="rolling-charts"></section>

    <section class="kpis" id="kpis"></section>

    <section class="grid">
      <div class="panel card span-6">
        <div class="label">Chart / Capacity</div>
        <h2>Endpoint States</h2>
        <div class="legend">
          <span style="color: #117a65">Running</span>
          <span style="color: #d9822b">Transitioning</span>
          <span style="color: #7f8c8d">Parked</span>
          <span style="color: #0b5cab">Connected Users</span>
        </div>
        <div class="chart-shell"><canvas id="endpoints-chart" width="900" height="260"></canvas></div>
      </div>

      <div class="panel card span-6">
        <div class="label">Chart / Flow</div>
        <h2>Sessions And Spare Capacity</h2>
        <div class="legend">
          <span style="color: #0b5cab">Connected</span>
          <span style="color: #d9822b">Pending</span>
          <span style="color: #117a65">Effective Free Slots</span>
          <span style="color: #bb2d3b">Errors</span>
        </div>
        <div class="chart-shell"><canvas id="sessions-chart" width="900" height="260"></canvas></div>
      </div>

      <div class="panel card span-8">
        <div class="label">Chart / Demand</div>
        <h2>Requests And Session Events</h2>
        <div class="legend">
          <span style="color: #182125">Session Requests</span>
          <span style="color: #117a65">Allocations</span>
          <span style="color: #bb2d3b">Allocation Failures</span>
          <span style="color: #0b5cab">Session Connects</span>
          <span style="color: #d9822b">Session Disconnects</span>
        </div>
        <div class="chart-shell"><canvas id="requests-chart" width="1200" height="260"></canvas></div>
      </div>

      <div class="panel card span-4">
        <div class="label">Now</div>
        <h2>Swarm Health</h2>
        <div id="health-badges"></div>
        <div class="mini-chart-shell">
          <canvas id="health-pie-chart" width="220" height="220"></canvas>
          <div class="mini-legend" id="health-pie-legend"></div>
        </div>
        <div class="footer-note" id="health-detail"></div>
      </div>

      <div class="panel card span-12">
        <div class="label">Chart / Conversations</div>
        <h2>Conversation Volume And Duration</h2>
        <div class="legend">
          <span style="color: #0b5cab">Starts</span>
          <span style="color: #117a65">Completed</span>
          <span style="color: #d9822b">Avg Duration (min)</span>
          <span style="color: #bb2d3b">Max Duration (min)</span>
        </div>
        <div class="chart-shell"><canvas id="conversations-chart" width="1200" height="260"></canvas></div>
      </div>

      <div class="panel card span-12">
        <div class="label">Current Fleet</div>
        <h2>Endpoint Wall</h2>
        <div class="fleet-summary" id="fleet-summary"></div>
        <div class="fleet-wall" id="endpoint-wall"></div>
      </div>
    </section>
  </div>

  <script>
    const windowButtons = [...document.querySelectorAll('[data-window]')];
    const resolutionButtons = [...document.querySelectorAll('[data-resolution]')];
    let selectedWindow = '6h';
    let selectedResolution = '';
    let refreshHandle = null;
    let latestDashboardPayload = null;
    const hiddenSeries = new Set();
    const rollingCharts = [
      {
        label: 'Rolling / Conversations',
        title: 'Completed Conversations',
        canvasId: 'rolling-conversations-chart',
        series: [
          { key: 'completed_conversations_1h', label: '1h total', color: '#0b5cab' },
          { key: 'completed_conversations_6h', label: '6h total', color: '#117a65' },
          { key: 'completed_conversations_24h', label: '24h total', color: '#d9822b' },
        ],
      },
      {
        label: 'Rolling / Workload',
        title: 'Conversation Days Served',
        canvasId: 'rolling-active-days-chart',
        series: [
          { key: 'active_conversation_days_1h', label: '1h sum', color: '#0b5cab' },
          { key: 'active_conversation_days_6h', label: '6h sum', color: '#117a65' },
          { key: 'active_conversation_days_24h', label: '24h sum', color: '#d9822b' },
        ],
      },
      {
        label: 'Rolling / Duration',
        title: 'Average Duration',
        canvasId: 'rolling-duration-chart',
        series: [
          { key: 'avg_conversation_duration_min_1h', label: '1h avg', color: '#0b5cab' },
          { key: 'avg_conversation_duration_min_6h', label: '6h avg', color: '#117a65' },
          { key: 'avg_conversation_duration_min_24h', label: '24h avg', color: '#d9822b' },
        ],
      },
      {
        label: 'Rolling / Users',
        title: 'Connected Users',
        canvasId: 'rolling-users-chart',
        series: [
          { key: 'connected_sessions_avg_1h', label: '1h avg', color: '#0b5cab' },
          { key: 'connected_sessions_avg_6h', label: '6h avg', color: '#117a65' },
          { key: 'connected_sessions_avg_24h', label: '24h avg', color: '#d9822b' },
        ],
      },
      {
        label: 'Rolling / Users',
        title: 'Maximum Connected Users',
        canvasId: 'rolling-max-users-chart',
        series: [
          { key: 'connected_sessions_max_1h', label: '1h max', color: '#0b5cab' },
          { key: 'connected_sessions_max_6h', label: '6h max', color: '#117a65' },
          { key: 'connected_sessions_max_24h', label: '24h max', color: '#d9822b' },
        ],
      },
      {
        label: 'Rolling / Duration',
        title: 'Median Duration',
        canvasId: 'rolling-median-duration-chart',
        series: [
          { key: 'median_conversation_duration_min_1h', label: '1h median', color: '#0b5cab' },
          { key: 'median_conversation_duration_min_6h', label: '6h median', color: '#117a65' },
          { key: 'median_conversation_duration_min_24h', label: '24h median', color: '#d9822b' },
        ],
      },
    ];

    function prettyNumber(value) {
      const numeric = Number(value || 0);
      if (Math.abs(numeric - Math.round(numeric)) < 0.01) {
        return String(Math.round(numeric));
      }
      return numeric.toFixed(1);
    }

    function formatDuration(seconds) {
      const totalSeconds = Math.max(0, Math.round(Number(seconds || 0)));
      const hours = Math.floor(totalSeconds / 3600);
      const minutes = Math.floor((totalSeconds % 3600) / 60);
      const secs = totalSeconds % 60;
      if (hours > 0) {
        return `${hours}h ${String(minutes).padStart(2, '0')}m`;
      }
      if (minutes > 0) {
        return `${minutes}m ${String(secs).padStart(2, '0')}s`;
      }
      return `${secs}s`;
    }

    function formatHoursServed(hoursValue) {
      const totalMinutes = Math.max(0, Math.round(Number(hoursValue || 0) * 60));
      const days = Math.floor(totalMinutes / (24 * 60));
      const hours = Math.floor((totalMinutes % (24 * 60)) / 60);
      const minutes = totalMinutes % 60;
      if (days > 0) {
        return `${days}d ${hours}h ${minutes}m`;
      }
      if (hours > 0) {
        return `${hours}h ${String(minutes).padStart(2, '0')}m`;
      }
      return `${minutes}m`;
    }

    function statusClass(healthy, errors) {
      if (!healthy || errors > 0) return 'bad';
      return 'good';
    }

    function setActiveButtons() {
      const longWindow = isLongWindow(selectedWindow);
      if (longWindow) {
        selectedResolution = 'hour';
      }
      for (const button of windowButtons) {
        button.classList.toggle('active', button.dataset.window === selectedWindow);
      }
      for (const button of resolutionButtons) {
        button.disabled = longWindow && button.dataset.resolution === 'minute';
        button.classList.toggle('active', button.dataset.resolution === selectedResolution);
      }
    }

    function isLongWindow(windowValue) {
      const match = String(windowValue || '').match(/^(\\d+)d$/);
      return Boolean(match && Number(match[1]) >= 7);
    }

    function activeSeries(seriesConfig) {
      return seriesConfig.filter((series) => !hiddenSeries.has(series.key));
    }

    function renderRollingChartCards() {
      document.getElementById('rolling-charts').innerHTML = rollingCharts.map((chart) => `
        <div class="panel card">
          <div class="label">${htmlEscape(chart.label)}</div>
          <h2>${htmlEscape(chart.title)}</h2>
          <div class="legend">
            ${chart.series.map((series) => `
              <button
                class="legend-toggle"
                type="button"
                style="color: ${series.color}"
                data-series-toggle="${htmlEscape(series.key)}"
                aria-pressed="true"
              >${htmlEscape(series.label)}</button>
            `).join('')}
          </div>
          <div class="chart-shell"><canvas id="${htmlEscape(chart.canvasId)}" width="720" height="260"></canvas></div>
        </div>
      `).join('');
    }

    function updateLegendToggles() {
      for (const button of document.querySelectorAll('[data-series-toggle]')) {
        const hidden = hiddenSeries.has(button.dataset.seriesToggle);
        button.classList.toggle('hidden', hidden);
        button.setAttribute('aria-pressed', hidden ? 'false' : 'true');
      }
    }

    function drawRollingCharts(rollingSeries) {
      for (const chart of rollingCharts) {
        drawChart(
          document.getElementById(chart.canvasId),
          rollingSeries,
          activeSeries(chart.series),
        );
      }
    }

    function kpiCard(label, value, detail) {
      return `<div class="panel card kpi"><div class="label">${label}</div><strong>${htmlEscape(value)}</strong><small>${htmlEscape(detail)}</small></div>`;
    }

    function htmlEscape(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function renderHeroStats(current, summary) {
      const windowLabel = summary.window_label || '6h';
      const stats = [
        ['Running', current.running_endpoints],
        ['Connected', current.connected_sessions],
        [`Time served / ${windowLabel}`, formatHoursServed(summary.active_conversation_hours_window)],
        [`Avg Duration / ${windowLabel}`, formatDuration(summary.avg_conversation_duration_window_s)],
      ];
      document.getElementById('hero-stats').innerHTML = stats.map(([label, value]) => `
        <div class="hero-stat">
          <div class="label">${htmlEscape(label)}</div>
          <strong>${htmlEscape(typeof value === 'number' ? prettyNumber(value) : value)}</strong>
        </div>
      `).join('');
    }

    function renderKpis(current, summary) {
      const windowLabel = summary.window_label || '6h';
      document.getElementById('kpis').innerHTML = [
        kpiCard('Live conversations', prettyNumber(current.connected_sessions), 'Current live websocket sessions'),
        kpiCard('Pending joins', prettyNumber(current.pending_sessions), 'Reserved sessions waiting to connect'),
        kpiCard(`Requests / ${windowLabel}`, prettyNumber(summary.session_requests_window), `POST /session requests in the last ${windowLabel}`),
        kpiCard(`Failures / ${windowLabel}`, prettyNumber(summary.session_failures_window), `Allocation failures in the last ${windowLabel}`),
        kpiCard(`Started / ${windowLabel}`, prettyNumber(summary.conversations_started_window), `Conversation starts recorded in the last ${windowLabel}`),
        kpiCard(`Completed / ${windowLabel}`, prettyNumber(summary.conversations_completed_window), `Conversation ends recorded in the last ${windowLabel}`),
        kpiCard(`Avg duration / ${windowLabel}`, formatDuration(summary.avg_conversation_duration_window_s), `Average completed conversation duration in the last ${windowLabel}`),
        kpiCard(`Max duration / ${windowLabel}`, formatDuration(summary.max_conversation_duration_window_s), `Longest completed conversation in the last ${windowLabel}`),
        kpiCard(`Peak users / ${windowLabel}`, prettyNumber(summary.peak_connected_sessions_window), `Highest concurrent connected sessions in the last ${windowLabel}`),
        kpiCard(`Peak running / ${windowLabel}`, prettyNumber(summary.peak_running_endpoints_window), `Highest active compute endpoint count in the last ${windowLabel}`),
        kpiCard('Free slots', prettyNumber(current.free_slots), 'Currently running free slots'),
        kpiCard('Errors', prettyNumber(current.errors_count), current.healthy ? 'No active router errors' : (current.detail || 'Swarm is degraded')),
      ].join('');
    }

    function countEndpointStates(endpoints) {
      const counts = { running: 0, warm: 0, parked: 0, error: 0 };
      for (const endpoint of endpoints) {
        counts[endpointVisualState(endpoint)] += 1;
      }
      return counts;
    }

    function renderHealth(current) {
      const counts = countEndpointStates(current.endpoints || []);
      const badges = [
        `<span class="status-pill good">healthy ${htmlEscape(prettyNumber(counts.running))}</span>`,
        `<span class="status-pill ${counts.warm ? 'warm' : 'good'}">changing ${htmlEscape(prettyNumber(counts.warm))}</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(counts.parked))} parked</span>`,
        `<span class="status-pill ${counts.error || !current.healthy ? 'bad' : 'good'}">error ${htmlEscape(prettyNumber(counts.error))}</span>`,
      ];
      document.getElementById('health-badges').innerHTML = badges.join(' ');
      document.getElementById('health-detail').textContent =
        current.detail || (current.healthy ? 'Swarm snapshot is current.' : 'The load balancer reports degraded routing health.');
      drawStatePieChart(document.getElementById('health-pie-chart'), counts);
      renderStateLegend(document.getElementById('health-pie-legend'), counts);
    }

    function endpointVisualState(endpoint) {
      const rawStatus = String(endpoint.status || 'unknown').toLowerCase();
      if (endpoint.last_error) return 'error';
      if (endpoint.waking || rawStatus.includes('init') || rawStatus.includes('updat')) return 'warm';
      if (rawStatus.includes('run')) return 'running';
      if (rawStatus.includes('pause') || rawStatus.includes('scale')) return 'parked';
      return 'error';
    }

    function endpointSortWeight(endpoint) {
      const state = endpointVisualState(endpoint);
      if (state === 'error') return 0;
      if (Number(endpoint.active_sessions || 0) > 0) return 1;
      if (state === 'warm') return 2;
      if (state === 'running') return 3;
      return 4;
    }

    function renderEndpointWall(endpoints) {
      const sorted = [...endpoints].sort((left, right) => {
        const weightDiff = endpointSortWeight(left) - endpointSortWeight(right);
        if (weightDiff !== 0) return weightDiff;
        const activeDiff = Number(right.active_sessions || 0) - Number(left.active_sessions || 0);
        if (activeDiff !== 0) return activeDiff;
        return String(left.name || '').localeCompare(String(right.name || ''));
      });

      const counts = countEndpointStates(sorted);

      document.getElementById('fleet-summary').innerHTML = [
        `<span class="status-pill good">${htmlEscape(prettyNumber(counts.running))} running</span>`,
        `<span class="status-pill warm">${htmlEscape(prettyNumber(counts.warm))} changing</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(counts.parked))} parked</span>`,
        `<span class="status-pill bad">${htmlEscape(prettyNumber(counts.error))} error</span>`,
      ].join('');

      const cards = sorted.map((endpoint) => {
        const state = endpointVisualState(endpoint);
        const activeSessions = Number(endpoint.active_sessions || 0);
        const freeSlots = Number(endpoint.free_slots || 0);
        const totalSlots = Math.max(activeSessions + freeSlots, 1);
        const utilizationPct = Math.max(0, Math.min(100, (activeSessions / totalSlots) * 100));
        const statusLabel =
          state === 'running' ? 'good' :
          state === 'warm' ? 'warm' :
          state === 'parked' ? '' : 'bad';
        const flags = [
          endpoint.waking ? 'waking' : null,
          endpoint.parking ? 'parking' : null,
          endpoint.last_error ? 'error' : null,
        ].filter(Boolean);

        return `
          <article class="endpoint-tile state-${htmlEscape(state)}">
            <div class="endpoint-top">
              <div>
                <div class="endpoint-name">${htmlEscape(endpoint.name || 'unknown')}</div>
                <div class="muted" style="margin-top:6px;">${htmlEscape(endpoint.status || 'unknown')}</div>
              </div>
              <div class="endpoint-badges">
                <span class="status-pill ${statusLabel}">${htmlEscape(state)}</span>
              </div>
            </div>
            <div class="endpoint-stats">
              <div>
                <div class="endpoint-stat-label">Users</div>
                <div class="endpoint-stat-value">${htmlEscape(prettyNumber(activeSessions))}</div>
              </div>
              <div>
                <div class="endpoint-stat-label">Free</div>
                <div class="endpoint-stat-value">${htmlEscape(prettyNumber(freeSlots))}</div>
              </div>
              <div>
                <div class="endpoint-stat-label">Use</div>
                <div class="endpoint-stat-value">${htmlEscape(prettyNumber(utilizationPct))}%</div>
              </div>
            </div>
            <div class="endpoint-meter">
              <div class="endpoint-meter-fill" style="width:${utilizationPct}%;"></div>
            </div>
            <div class="endpoint-flags">${htmlEscape(flags.join(' • ') || 'stable')}</div>
          </article>
        `;
      }).join('');

      document.getElementById('endpoint-wall').innerHTML = cards;
    }

    function prepareCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const cssWidth = Math.max(1, Math.round(rect.width || canvas.clientWidth || canvas.width));
      const cssHeight = Math.max(1, Math.round(rect.height || canvas.clientHeight || canvas.height));
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const pixelWidth = Math.round(cssWidth * dpr);
      const pixelHeight = Math.round(cssHeight * dpr);
      if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
        canvas.width = pixelWidth;
        canvas.height = pixelHeight;
      }
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssWidth, cssHeight);
      return { ctx, width: cssWidth, height: cssHeight };
    }

    function drawStatePieChart(canvas, counts) {
      const { ctx, width, height } = prepareCanvas(canvas);
      const radius = Math.min(width, height) * 0.34;
      const centerX = width / 2;
      const centerY = height / 2;
      const total = counts.running + counts.warm + counts.parked + counts.error;

      const segments = [
        { value: counts.running, color: '#117a65', label: 'Healthy' },
        { value: counts.warm, color: '#d9822b', label: 'Changing' },
        { value: counts.parked, color: '#8b969b', label: 'Parked' },
        { value: counts.error, color: '#bb2d3b', label: 'Error' },
      ];

      if (total <= 0) {
        ctx.fillStyle = 'rgba(24, 33, 37, 0.08)';
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.fill();
      } else {
        let startAngle = -Math.PI / 2;
        for (const segment of segments) {
          if (!segment.value) continue;
          const endAngle = startAngle + (segment.value / total) * Math.PI * 2;
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.arc(centerX, centerY, radius, startAngle, endAngle);
          ctx.closePath();
          ctx.fillStyle = segment.color;
          ctx.fill();
          startAngle = endAngle;
        }
      }

      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * 0.54, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
      ctx.fill();

      ctx.fillStyle = '#182125';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '12px Menlo, Consolas, monospace';
      ctx.fillText('endpoints', centerX, centerY - 14);
      ctx.font = 'bold 34px "Avenir Next", "Segoe UI", sans-serif';
      ctx.fillText(String(total), centerX, centerY + 10);
    }

    function renderStateLegend(container, counts) {
      const items = [
        { key: 'running', label: 'Healthy', color: '#117a65' },
        { key: 'warm', label: 'Changing', color: '#d9822b' },
        { key: 'parked', label: 'Parked', color: '#8b969b' },
        { key: 'error', label: 'Error', color: '#bb2d3b' },
      ];
      container.innerHTML = items.map((item) => `
        <div class="mini-legend-row">
          <span class="mini-legend-label">
            <span class="mini-legend-dot" style="background:${item.color};"></span>
            ${htmlEscape(item.label)}
          </span>
          <strong>${htmlEscape(prettyNumber(counts[item.key]))}</strong>
        </div>
      `).join('');
    }

    function niceTickSize(value) {
      if (value <= 0) return 1;
      const exponent = Math.floor(Math.log10(value));
      const magnitude = 10 ** exponent;
      const fraction = value / magnitude;
      if (fraction <= 1) return magnitude;
      if (fraction <= 2) return 2 * magnitude;
      if (fraction <= 5) return 5 * magnitude;
      return 10 * magnitude;
    }

    function chartTickValues(maxValue) {
      const tickSize = niceTickSize(maxValue / 4);
      const niceMax = Math.max(tickSize, Math.ceil(maxValue / tickSize) * tickSize);
      const ticks = [];
      for (let value = 0; value <= niceMax + tickSize * 0.5; value += tickSize) {
        ticks.push(Number(value.toFixed(10)));
      }
      return ticks;
    }

    function formatAxisNumber(value) {
      const numeric = Number(value || 0);
      if (numeric >= 1000) {
        return Intl.NumberFormat([], { notation: 'compact', maximumFractionDigits: 1 }).format(numeric);
      }
      if (Math.abs(numeric - Math.round(numeric)) < 0.01) {
        return String(Math.round(numeric));
      }
      if (numeric >= 10) return numeric.toFixed(1);
      return numeric.toFixed(2).replace(/\\.?0+$/, '');
    }

    function xAxisLabel(timestamp, includeDate) {
      const date = new Date(timestamp);
      if (!includeDate) {
        return {
          primary: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          secondary: '',
        };
      }
      return {
        primary: date.toLocaleDateString([], { month: 'short', day: 'numeric' }),
        secondary: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
    }

    function drawXAxisLabel(ctx, label, x, y, align) {
      ctx.textAlign = align;
      ctx.fillText(label.primary, x, y);
      if (label.secondary) {
        ctx.fillText(label.secondary, x, y + 14);
      }
    }

    function drawChart(canvas, points, seriesConfig) {
      const { ctx, width, height } = prepareCanvas(canvas);
      const first = points.length ? new Date(points[0].timestamp) : null;
      const last = points.length ? new Date(points[points.length - 1].timestamp) : null;
      const spanMs = first && last ? last.getTime() - first.getTime() : 0;
      const includeDate = spanMs >= 36 * 3600 * 1000;

      const padding = { top: 18, right: 22, bottom: includeDate ? 52 : 36, left: 50 };
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const values = points
        .flatMap((point) => seriesConfig.map((series) => Number(point[series.key] || 0)))
        .filter((value) => Number.isFinite(value));
      const maxValue = Math.max(1, ...values);
      const ticks = chartTickValues(maxValue);
      const chartMax = ticks[ticks.length - 1] || 1;
      const xStep = points.length > 1 ? plotWidth / (points.length - 1) : 0;

      ctx.strokeStyle = 'rgba(24, 33, 37, 0.12)';
      ctx.lineWidth = 1;
      for (const value of ticks) {
        const y = padding.top + plotHeight - (plotHeight * value) / chartMax;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(24, 33, 37, 0.74)';
      ctx.font = '12px Menlo, Consolas, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const value of ticks) {
        const y = padding.top + plotHeight - (plotHeight * value) / chartMax;
        ctx.fillText(formatAxisNumber(value), padding.left - 10, y);
      }

      seriesConfig.forEach((series) => {
        ctx.strokeStyle = series.color;
        ctx.lineWidth = series.width || 2.5;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        points.forEach((point, index) => {
          const x = padding.left + xStep * index;
          const y = padding.top + plotHeight - (plotHeight * Number(point[series.key] || 0)) / chartMax;
          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
      });

      if (points.length > 1) {
        ctx.fillStyle = 'rgba(24, 33, 37, 0.74)';
        ctx.font = '12px Menlo, Consolas, monospace';
        ctx.textBaseline = 'alphabetic';
        const middleIndex = Math.floor((points.length - 1) / 2);
        const middleX = padding.left + xStep * middleIndex;
        const labelY = includeDate ? height - 24 : height - 12;
        drawXAxisLabel(ctx, xAxisLabel(points[0].timestamp, includeDate), padding.left, labelY, 'left');
        drawXAxisLabel(ctx, xAxisLabel(points[middleIndex].timestamp, includeDate), middleX, labelY, 'center');
        drawXAxisLabel(ctx, xAxisLabel(points[points.length - 1].timestamp, includeDate), width - padding.right, labelY, 'right');
      }
    }

    async function loadDashboard() {
      const params = new URLSearchParams({ window: selectedWindow });
      if (selectedResolution) {
        params.set('resolution', selectedResolution);
      }
      const response = await fetch(`/dashboard/data?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`Dashboard request failed: ${response.status}`);
      }
      const payload = await response.json();
      latestDashboardPayload = payload;
      const current = payload.current;
      const summary = payload.summary;
      const series = payload.series;
      const rollingSeries = payload.rolling_series || [];

      document.getElementById('generated-at').textContent =
        `Updated ${new Date(payload.generated_at).toLocaleString()} • ${payload.window.resolution} timeline • last ${payload.window.requested}`;

      renderHeroStats(current, summary);
      renderKpis(current, summary);
      renderHealth(current);
      renderEndpointWall(current.endpoints || []);

      drawChart(document.getElementById('endpoints-chart'), series, [
        { key: 'running_endpoints', color: '#117a65' },
        { key: 'transitioning_endpoints', color: '#d9822b' },
        { key: 'parked_endpoints', color: '#7f8c8d' },
        { key: 'connected_sessions', color: '#0b5cab', width: 2.2 },
      ]);

      drawChart(document.getElementById('sessions-chart'), series, [
        { key: 'connected_sessions', color: '#0b5cab' },
        { key: 'pending_sessions', color: '#d9822b' },
        { key: 'effective_free_slots', color: '#117a65' },
        { key: 'errors_count', color: '#bb2d3b', width: 2.2 },
      ]);

      drawChart(document.getElementById('requests-chart'), series, [
        { key: 'session_requests', color: '#182125' },
        { key: 'session_allocation_successes', color: '#117a65' },
        { key: 'session_allocation_failures', color: '#bb2d3b' },
        { key: 'session_connected_events', color: '#0b5cab' },
        { key: 'session_disconnected_events', color: '#d9822b' },
      ]);

      drawChart(document.getElementById('conversations-chart'), series, [
        { key: 'session_connected_events', color: '#0b5cab' },
        { key: 'completed_conversations', color: '#117a65' },
        { key: 'avg_conversation_duration_min', color: '#d9822b' },
        { key: 'max_conversation_duration_min', color: '#bb2d3b' },
      ]);

      drawRollingCharts(rollingSeries);
    }

    function scheduleRefresh() {
      if (refreshHandle) {
        clearInterval(refreshHandle);
      }
      refreshHandle = setInterval(() => {
        loadDashboard().catch((error) => console.error(error));
      }, 15000);
    }

    windowButtons.forEach((button) => {
      button.addEventListener('click', () => {
        selectedWindow = button.dataset.window;
        if (isLongWindow(selectedWindow)) {
          selectedResolution = 'hour';
        }
        setActiveButtons();
        loadDashboard().catch((error) => console.error(error));
      });
    });

    resolutionButtons.forEach((button) => {
      button.addEventListener('click', () => {
        if (button.disabled) {
          return;
        }
        selectedResolution = button.dataset.resolution === selectedResolution ? '' : button.dataset.resolution;
        setActiveButtons();
        loadDashboard().catch((error) => console.error(error));
      });
    });

    renderRollingChartCards();

    document.querySelectorAll('[data-series-toggle]').forEach((button) => {
      button.addEventListener('click', () => {
        const key = button.dataset.seriesToggle;
        if (hiddenSeries.has(key)) {
          hiddenSeries.delete(key);
        } else {
          hiddenSeries.add(key);
        }
        updateLegendToggles();
        if (latestDashboardPayload) {
          drawRollingCharts(latestDashboardPayload.rolling_series || []);
        }
      });
    });

    setActiveButtons();
    updateLegendToggles();
    loadDashboard().catch((error) => console.error(error));
    scheduleRefresh();
  </script>
</body>
</html>
""".replace("__HISTORY_NOTE__", history_note)

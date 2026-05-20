import asyncio
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional, Protocol


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
        return {
            "bucket_start_s": self.bucket_start_s,
            "sample_count": self.sample_count,
            "running_endpoints_last": self.running_endpoints_last,
            "running_endpoints_sum": self.running_endpoints_sum,
            "warming_endpoints_last": self.warming_endpoints_last,
            "warming_endpoints_sum": self.warming_endpoints_sum,
            "transitioning_endpoints_last": self.transitioning_endpoints_last,
            "transitioning_endpoints_sum": self.transitioning_endpoints_sum,
            "parked_endpoints_last": self.parked_endpoints_last,
            "parked_endpoints_sum": self.parked_endpoints_sum,
            "connected_sessions_last": self.connected_sessions_last,
            "connected_sessions_sum": self.connected_sessions_sum,
            "pending_sessions_last": self.pending_sessions_last,
            "pending_sessions_sum": self.pending_sessions_sum,
            "free_slots_last": self.free_slots_last,
            "free_slots_sum": self.free_slots_sum,
            "effective_free_slots_last": self.effective_free_slots_last,
            "effective_free_slots_sum": self.effective_free_slots_sum,
            "router_active_sessions_last": self.router_active_sessions_last,
            "router_active_sessions_sum": self.router_active_sessions_sum,
            "healthy_last": self.healthy_last,
            "healthy_samples": self.healthy_samples,
            "errors_count_last": self.errors_count_last,
            "errors_count_sum": self.errors_count_sum,
            "session_requests": self.session_requests,
            "session_allocation_successes": self.session_allocation_successes,
            "session_allocation_failures": self.session_allocation_failures,
            "session_connected_events": self.session_connected_events,
            "session_disconnected_events": self.session_disconnected_events,
            "completed_conversations": self.completed_conversations,
            "completed_conversation_duration_total_s": self.completed_conversation_duration_total_s,
            "completed_conversation_duration_max_s": self.completed_conversation_duration_max_s,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SwarmHistoryBucket":
        return cls(
            bucket_start_s=int(payload.get("bucket_start_s", 0)),
            sample_count=int(payload.get("sample_count", 0)),
            running_endpoints_last=int(payload.get("running_endpoints_last", 0)),
            running_endpoints_sum=float(payload.get("running_endpoints_sum", 0.0)),
            warming_endpoints_last=int(payload.get("warming_endpoints_last", 0)),
            warming_endpoints_sum=float(payload.get("warming_endpoints_sum", 0.0)),
            transitioning_endpoints_last=int(payload.get("transitioning_endpoints_last", 0)),
            transitioning_endpoints_sum=float(payload.get("transitioning_endpoints_sum", 0.0)),
            parked_endpoints_last=int(payload.get("parked_endpoints_last", 0)),
            parked_endpoints_sum=float(payload.get("parked_endpoints_sum", 0.0)),
            connected_sessions_last=int(payload.get("connected_sessions_last", 0)),
            connected_sessions_sum=float(payload.get("connected_sessions_sum", 0.0)),
            pending_sessions_last=int(payload.get("pending_sessions_last", 0)),
            pending_sessions_sum=float(payload.get("pending_sessions_sum", 0.0)),
            free_slots_last=int(payload.get("free_slots_last", 0)),
            free_slots_sum=float(payload.get("free_slots_sum", 0.0)),
            effective_free_slots_last=int(payload.get("effective_free_slots_last", 0)),
            effective_free_slots_sum=float(payload.get("effective_free_slots_sum", 0.0)),
            router_active_sessions_last=int(payload.get("router_active_sessions_last", 0)),
            router_active_sessions_sum=float(payload.get("router_active_sessions_sum", 0.0)),
            healthy_last=bool(payload.get("healthy_last", False)),
            healthy_samples=int(payload.get("healthy_samples", 0)),
            errors_count_last=int(payload.get("errors_count_last", 0)),
            errors_count_sum=float(payload.get("errors_count_sum", 0.0)),
            session_requests=int(payload.get("session_requests", 0)),
            session_allocation_successes=int(payload.get("session_allocation_successes", 0)),
            session_allocation_failures=int(payload.get("session_allocation_failures", 0)),
            session_connected_events=int(payload.get("session_connected_events", 0)),
            session_disconnected_events=int(payload.get("session_disconnected_events", 0)),
            completed_conversations=int(payload.get("completed_conversations", 0)),
            completed_conversation_duration_total_s=float(
                payload.get("completed_conversation_duration_total_s", 0.0)
            ),
            completed_conversation_duration_max_s=float(
                payload.get("completed_conversation_duration_max_s", 0.0)
            ),
        )



class SwarmDashboard:
    def __init__(
        self,
        *,
        snapshot_provider: SnapshotProvider,
        sample_interval_s: float = 15.0,
        retention_minutes: int = 28 * 24 * 60,
        history_store: Optional[DashboardHistoryStore] = None,
        restore_history_in_background: bool = False,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
        if retention_minutes < 60:
            raise ValueError("retention_minutes must be >= 60")

        self.snapshot_provider = snapshot_provider
        self.sample_interval_s = sample_interval_s
        self.retention_minutes = retention_minutes
        self.history_store = history_store
        self.restore_history_in_background = restore_history_in_background
        self._time_fn = time_fn
        self._lock = asyncio.Lock()
        self._history: "OrderedDict[int, SwarmHistoryBucket]" = OrderedDict()
        self._latest_sample: Optional[SwarmStateSample] = None
        self._sample_task: Optional[asyncio.Task] = None
        self._restore_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._day_rollover_task: Optional[asyncio.Task] = None
        self._dirty_bucket_starts: set[int] = set()
        self._day_rollover_cursor_s: Optional[int] = None
        self._history_restore_status = "disabled" if history_store is None else "pending"
        self._history_restore_detail: Optional[str] = None
        self._history_restore_started_at_s: Optional[float] = None
        self._history_restore_finished_at_s: Optional[float] = None
        self._history_restore_bucket_count = 0

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
        if self._history_store_day_writer() is not None:
            self._day_rollover_cursor_s = _day_start_epoch_s(self._time_fn())
            self._day_rollover_task = asyncio.create_task(self._day_rollover_loop())
        self._sample_task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> None:
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

        if self._day_rollover_task is not None:
            self._day_rollover_task.cancel()
            try:
                await self._day_rollover_task
            except asyncio.CancelledError:
                pass
            self._day_rollover_task = None

        if self._flush_task is not None:
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        await self._flush_dirty_buckets(include_open_bucket=True)
        await self._rollover_completed_days()

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
            self._dirty_bucket_starts.add(bucket.bucket_start_s)
            self._prune_unlocked(sample.captured_at_s)
            self._schedule_flush_unlocked(include_open_bucket=False)

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
        completed_prefix = [0]
        duration_prefix = [0.0]
        connected_sum_prefix = [0.0]
        sample_count_prefix = [0]
        active_conversation_minutes_prefix = [0.0]
        for bucket_start_s in minute_starts:
            bucket = minute_map.get(bucket_start_s)
            completed_prefix.append(
                completed_prefix[-1] + (bucket.completed_conversations if bucket is not None else 0)
            )
            duration_prefix.append(
                duration_prefix[-1]
                + (bucket.completed_conversation_duration_total_s if bucket is not None else 0.0)
            )
            connected_sum_prefix.append(
                connected_sum_prefix[-1] + (bucket.connected_sessions_sum if bucket is not None else 0.0)
            )
            sample_count_prefix.append(
                sample_count_prefix[-1] + (bucket.sample_count if bucket is not None else 0)
            )
            active_conversation_minutes_prefix.append(
                active_conversation_minutes_prefix[-1]
                + (
                    bucket.connected_sessions_sum / bucket.sample_count
                    if bucket is not None and bucket.sample_count
                    else 0.0
                )
            )

        def range_sum(prefix: list[int] | list[float], start_s: int, end_s: int) -> int | float:
            if start_s > end_bucket or end_s < context_start_bucket:
                return 0
            bounded_start_s = max(start_s, context_start_bucket)
            bounded_end_s = min(end_s, end_bucket)
            start_index = (bounded_start_s - context_start_bucket) // 60
            end_index = (bounded_end_s - context_start_bucket) // 60
            return prefix[end_index + 1] - prefix[start_index]

        step_s = 60 if resolution == "minute" else 3600
        first_point_s = start_bucket if resolution == "minute" else _bucket_start_epoch_s(start_bucket, 60)
        points = []
        for point_start_s in range(first_point_s, end_bucket + 1, step_s):
            point_end_s = point_start_s if resolution == "minute" else min(point_start_s + 59 * 60, end_bucket)
            point: dict[str, object] = {"timestamp": _isoformat(point_start_s)}
            for label, minutes in ROLLING_VIEW_WINDOWS:
                window_start_s = point_end_s - (minutes - 1) * 60
                completed = int(range_sum(completed_prefix, window_start_s, point_end_s))
                duration_total_s = float(range_sum(duration_prefix, window_start_s, point_end_s))
                connected_sum = float(range_sum(connected_sum_prefix, window_start_s, point_end_s))
                sample_count = int(range_sum(sample_count_prefix, window_start_s, point_end_s))
                active_conversation_minutes = float(
                    range_sum(active_conversation_minutes_prefix, window_start_s, point_end_s)
                )

                point[f"completed_conversations_{label}"] = completed
                point[f"active_conversation_minutes_{label}"] = round(active_conversation_minutes, 2)
                point[f"active_conversation_hours_{label}"] = round(active_conversation_minutes / 60.0, 2)
                point[f"active_conversation_days_{label}"] = round(active_conversation_minutes / (24.0 * 60.0), 3)
                point[f"avg_conversation_duration_s_{label}"] = (
                    round(duration_total_s / completed, 2) if completed else 0.0
                )
                point[f"avg_conversation_duration_min_{label}"] = (
                    round((duration_total_s / completed) / 60.0, 2) if completed else 0.0
                )
                point[f"connected_sessions_avg_{label}"] = (
                    round(connected_sum / sample_count, 2) if sample_count else 0.0
                )
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

    async def _sample_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.sample_interval_s)
                await self.capture_sample()
        except asyncio.CancelledError:
            raise

    async def _day_rollover_loop(self) -> None:
        try:
            while True:
                now_s = self._time_fn()
                next_day_start_s = _day_start_epoch_s(now_s) + DAY_SECONDS
                await asyncio.sleep(max(1.0, next_day_start_s + 90 - now_s))
                await self._rollover_completed_days()
        except asyncio.CancelledError:
            raise

    async def _increment_counter(self, field_name: str) -> None:
        now = self._time_fn()
        async with self._lock:
            bucket = self._get_bucket_unlocked(now)
            setattr(bucket, field_name, getattr(bucket, field_name) + 1)
            self._dirty_bucket_starts.add(bucket.bucket_start_s)
            self._prune_unlocked(now)
            self._schedule_flush_unlocked(include_open_bucket=False)

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
            self._dirty_bucket_starts.add(bucket.bucket_start_s)
            self._prune_unlocked(now)
            self._schedule_flush_unlocked(include_open_bucket=False)

    def _get_bucket_unlocked(self, epoch_s: float) -> SwarmHistoryBucket:
        bucket_start_s = _bucket_start_epoch_s(epoch_s, 1)
        bucket = self._history.get(bucket_start_s)
        if bucket is None:
            bucket = SwarmHistoryBucket(bucket_start_s=bucket_start_s)
            self._history[bucket_start_s] = bucket
        return bucket

    def _prune_unlocked(self, epoch_s: float) -> None:
        min_allowed_bucket = _bucket_start_epoch_s(epoch_s, 1) - (self.retention_minutes - 1) * 60
        while self._history:
            oldest_key = next(iter(self._history))
            if oldest_key >= min_allowed_bucket:
                break
            self._history.popitem(last=False)
            self._dirty_bucket_starts.discard(oldest_key)

    def _aggregate_recent(self, minute_buckets: list[SwarmHistoryBucket], *, window_minutes: int) -> dict[str, object]:
        now = self._time_fn()
        min_bucket = _bucket_start_epoch_s(now, 1) - (window_minutes - 1) * 60
        selected = [bucket for bucket in minute_buckets if bucket.bucket_start_s >= min_bucket]
        completed_conversations = sum(bucket.completed_conversations for bucket in selected)
        completed_conversation_duration_total_s = sum(
            bucket.completed_conversation_duration_total_s for bucket in selected
        )
        active_conversation_minutes = sum(
            bucket.connected_sessions_sum / bucket.sample_count
            for bucket in selected
            if bucket.sample_count
        )
        return {
            "session_requests": sum(bucket.session_requests for bucket in selected),
            "session_allocation_successes": sum(bucket.session_allocation_successes for bucket in selected),
            "session_allocation_failures": sum(bucket.session_allocation_failures for bucket in selected),
            "session_connected_events": sum(bucket.session_connected_events for bucket in selected),
            "session_disconnected_events": sum(bucket.session_disconnected_events for bucket in selected),
            "completed_conversations": completed_conversations,
            "active_conversation_minutes": round(active_conversation_minutes, 2),
            "active_conversation_hours": round(active_conversation_minutes / 60.0, 2),
            "active_conversation_days": round(active_conversation_minutes / (24.0 * 60.0), 3),
            "avg_conversation_duration_s": (
                round(completed_conversation_duration_total_s / completed_conversations, 2)
                if completed_conversations
                else 0.0
            ),
            "max_conversation_duration_s": round(
                max((bucket.completed_conversation_duration_max_s for bucket in selected), default=0.0),
                2,
            ),
            "peak_connected_sessions": max((bucket.connected_sessions_last for bucket in selected), default=0),
            "peak_running_endpoints": max((bucket.running_endpoints_last for bucket in selected), default=0),
        }

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

            sample_count = sum(bucket.sample_count for bucket in minute_buckets)
            if sample_count == 0:
                point = {
                    "timestamp": _isoformat(current_hour_s),
                    "running_endpoints": 0.0,
                    "warming_endpoints": 0.0,
                    "transitioning_endpoints": 0.0,
                    "parked_endpoints": 0.0,
                    "connected_sessions": 0.0,
                    "pending_sessions": 0.0,
                    "free_slots": 0.0,
                    "effective_free_slots": 0.0,
                    "router_active_sessions": 0.0,
                    "errors_count": 0.0,
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
            else:
                completed_conversations = sum(bucket.completed_conversations for bucket in minute_buckets)
                completed_conversation_duration_total_s = sum(
                    bucket.completed_conversation_duration_total_s for bucket in minute_buckets
                )
                point = {
                    "timestamp": _isoformat(current_hour_s),
                    "running_endpoints": round(
                        sum(bucket.running_endpoints_sum for bucket in minute_buckets) / sample_count, 2
                    ),
                    "warming_endpoints": round(
                        sum(bucket.warming_endpoints_sum for bucket in minute_buckets) / sample_count, 2
                    ),
                    "transitioning_endpoints": round(
                        sum(bucket.transitioning_endpoints_sum for bucket in minute_buckets) / sample_count, 2
                    ),
                    "parked_endpoints": round(
                        sum(bucket.parked_endpoints_sum for bucket in minute_buckets) / sample_count, 2
                    ),
                    "connected_sessions": round(
                        sum(bucket.connected_sessions_sum for bucket in minute_buckets) / sample_count, 2
                    ),
                    "pending_sessions": round(
                        sum(bucket.pending_sessions_sum for bucket in minute_buckets) / sample_count, 2
                    ),
                    "free_slots": round(
                        sum(bucket.free_slots_sum for bucket in minute_buckets) / sample_count,
                        2,
                    ),
                    "effective_free_slots": round(
                        sum(bucket.effective_free_slots_sum for bucket in minute_buckets) / sample_count,
                        2,
                    ),
                    "router_active_sessions": round(
                        sum(bucket.router_active_sessions_sum for bucket in minute_buckets) / sample_count,
                        2,
                    ),
                    "errors_count": round(
                        sum(bucket.errors_count_sum for bucket in minute_buckets) / sample_count,
                        2,
                    ),
                    "healthy": (sum(bucket.healthy_samples for bucket in minute_buckets) / sample_count) >= 0.5,
                    "session_requests": sum(bucket.session_requests for bucket in minute_buckets),
                    "session_allocation_successes": sum(
                        bucket.session_allocation_successes for bucket in minute_buckets
                    ),
                    "session_allocation_failures": sum(
                        bucket.session_allocation_failures for bucket in minute_buckets
                    ),
                    "session_connected_events": sum(bucket.session_connected_events for bucket in minute_buckets),
                    "session_disconnected_events": sum(
                        bucket.session_disconnected_events for bucket in minute_buckets
                    ),
                    "completed_conversations": completed_conversations,
                    "avg_conversation_duration_s": round(
                        completed_conversation_duration_total_s / completed_conversations,
                        2,
                    ) if completed_conversations else 0.0,
                    "avg_conversation_duration_min": round(
                        (completed_conversation_duration_total_s / completed_conversations) / 60.0,
                        2,
                    ) if completed_conversations else 0.0,
                    "max_conversation_duration_s": round(
                        max((bucket.completed_conversation_duration_max_s for bucket in minute_buckets), default=0.0),
                        2,
                    ),
                    "max_conversation_duration_min": round(
                        max((bucket.completed_conversation_duration_max_s for bucket in minute_buckets), default=0.0)
                        / 60.0,
                        2,
                    ),
                }
            points.append(point)
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
            buckets = await asyncio.to_thread(
                self.history_store.load_recent,
                retention_minutes=self.retention_minutes,
                now_epoch_s=self._time_fn(),
            )
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

        async with self._lock:
            for bucket in sorted(buckets, key=lambda item: item.bucket_start_s):
                if bucket.bucket_start_s not in self._dirty_bucket_starts:
                    self._history[bucket.bucket_start_s] = bucket
            self._prune_unlocked(self._time_fn())

        self._history_restore_status = "complete"
        self._history_restore_detail = "Persisted dashboard history restored"
        self._history_restore_bucket_count = len(buckets)
        self._history_restore_finished_at_s = time.monotonic()
        logger.info(
            "Restored %s persisted dashboard minute buckets in %.2fs",
            len(buckets),
            time.monotonic() - started_s,
        )

    def _schedule_flush_unlocked(self, *, include_open_bucket: bool) -> None:
        if self.history_store is None:
            return
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._flush_task = asyncio.create_task(self._flush_dirty_buckets(include_open_bucket=include_open_bucket))

    async def _flush_dirty_buckets(self, *, include_open_bucket: bool) -> None:
        if self.history_store is None:
            return

        while True:
            async with self._lock:
                buckets = self._collect_dirty_buckets_unlocked(include_open_bucket=include_open_bucket)
            if not buckets:
                return

            try:
                await asyncio.to_thread(self.history_store.write_buckets, buckets)
            except Exception as exc:
                logger.warning("Failed to persist dashboard history to bucket store: %s", exc)
                return

            async with self._lock:
                for bucket in buckets:
                    self._dirty_bucket_starts.discard(bucket.bucket_start_s)

    def _history_store_day_writer(self) -> Optional[Callable[..., Optional[str]]]:
        if self.history_store is None:
            return None
        if getattr(self.history_store, "read_only", False):
            return None
        writer = getattr(self.history_store, "write_day_buckets", None)
        return writer if callable(writer) else None

    async def _rollover_completed_days(self) -> None:
        writer = self._history_store_day_writer()
        if writer is None:
            return

        current_task = asyncio.current_task()
        if self._flush_task is not None and self._flush_task is not current_task and not self._flush_task.done():
            try:
                await self._flush_task
            except asyncio.CancelledError:
                raise

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
            if len(day_buckets) != DAY_MINUTES:
                logger.info(
                    "Skipping live dashboard day rollover for %s because only %s of 1440 minute buckets are in memory",
                    _day_key(day_start),
                    len(day_buckets),
                )
                continue

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
            self._history[bucket_start_s]
            for bucket_start_s in sorted(self._dirty_bucket_starts)
            if bucket_start_s in self._history
            and (include_open_bucket or bucket_start_s < current_bucket_start_s)
        ]


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

    <section class="rolling-grid">
      <div class="panel card">
        <div class="label">Rolling / Conversations</div>
        <h2>Completed Conversations</h2>
        <div class="legend">
          <button class="legend-toggle" type="button" style="color: #0b5cab" data-series-toggle="completed_conversations_1h" aria-pressed="true">1h total</button>
          <button class="legend-toggle" type="button" style="color: #117a65" data-series-toggle="completed_conversations_6h" aria-pressed="true">6h total</button>
          <button class="legend-toggle" type="button" style="color: #d9822b" data-series-toggle="completed_conversations_24h" aria-pressed="true">24h total</button>
        </div>
        <div class="chart-shell"><canvas id="rolling-conversations-chart" width="720" height="260"></canvas></div>
      </div>

      <div class="panel card">
        <div class="label">Rolling / Workload</div>
        <h2>Conversation Days Served</h2>
        <div class="legend">
          <button class="legend-toggle" type="button" style="color: #0b5cab" data-series-toggle="active_conversation_days_1h" aria-pressed="true">1h sum</button>
          <button class="legend-toggle" type="button" style="color: #117a65" data-series-toggle="active_conversation_days_6h" aria-pressed="true">6h sum</button>
          <button class="legend-toggle" type="button" style="color: #d9822b" data-series-toggle="active_conversation_days_24h" aria-pressed="true">24h sum</button>
        </div>
        <div class="chart-shell"><canvas id="rolling-active-days-chart" width="720" height="260"></canvas></div>
      </div>

      <div class="panel card">
        <div class="label">Rolling / Duration</div>
        <h2>Average Duration</h2>
        <div class="legend">
          <button class="legend-toggle" type="button" style="color: #0b5cab" data-series-toggle="avg_conversation_duration_min_1h" aria-pressed="true">1h avg</button>
          <button class="legend-toggle" type="button" style="color: #117a65" data-series-toggle="avg_conversation_duration_min_6h" aria-pressed="true">6h avg</button>
          <button class="legend-toggle" type="button" style="color: #d9822b" data-series-toggle="avg_conversation_duration_min_24h" aria-pressed="true">24h avg</button>
        </div>
        <div class="chart-shell"><canvas id="rolling-duration-chart" width="720" height="260"></canvas></div>
      </div>

      <div class="panel card">
        <div class="label">Rolling / Users</div>
        <h2>Connected Users</h2>
        <div class="legend">
          <button class="legend-toggle" type="button" style="color: #0b5cab" data-series-toggle="connected_sessions_avg_1h" aria-pressed="true">1h avg</button>
          <button class="legend-toggle" type="button" style="color: #117a65" data-series-toggle="connected_sessions_avg_6h" aria-pressed="true">6h avg</button>
          <button class="legend-toggle" type="button" style="color: #d9822b" data-series-toggle="connected_sessions_avg_24h" aria-pressed="true">24h avg</button>
        </div>
        <div class="chart-shell"><canvas id="rolling-users-chart" width="720" height="260"></canvas></div>
      </div>
    </section>

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
    const rollingChartConfigs = {
      conversations: [
        { key: 'completed_conversations_1h', color: '#0b5cab' },
        { key: 'completed_conversations_6h', color: '#117a65' },
        { key: 'completed_conversations_24h', color: '#d9822b' },
      ],
      activeDays: [
        { key: 'active_conversation_days_1h', color: '#0b5cab' },
        { key: 'active_conversation_days_6h', color: '#117a65' },
        { key: 'active_conversation_days_24h', color: '#d9822b' },
      ],
      duration: [
        { key: 'avg_conversation_duration_min_1h', color: '#0b5cab' },
        { key: 'avg_conversation_duration_min_6h', color: '#117a65' },
        { key: 'avg_conversation_duration_min_24h', color: '#d9822b' },
      ],
      users: [
        { key: 'connected_sessions_avg_1h', color: '#0b5cab' },
        { key: 'connected_sessions_avg_6h', color: '#117a65' },
        { key: 'connected_sessions_avg_24h', color: '#d9822b' },
      ],
    };

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

    function updateLegendToggles() {
      for (const button of document.querySelectorAll('[data-series-toggle]')) {
        const hidden = hiddenSeries.has(button.dataset.seriesToggle);
        button.classList.toggle('hidden', hidden);
        button.setAttribute('aria-pressed', hidden ? 'false' : 'true');
      }
    }

    function drawRollingCharts(rollingSeries) {
      drawChart(
        document.getElementById('rolling-conversations-chart'),
        rollingSeries,
        activeSeries(rollingChartConfigs.conversations),
      );
      drawChart(
        document.getElementById('rolling-active-days-chart'),
        rollingSeries,
        activeSeries(rollingChartConfigs.activeDays),
      );
      drawChart(
        document.getElementById('rolling-duration-chart'),
        rollingSeries,
        activeSeries(rollingChartConfigs.duration),
      );
      drawChart(
        document.getElementById('rolling-users-chart'),
        rollingSeries,
        activeSeries(rollingChartConfigs.users),
      );
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

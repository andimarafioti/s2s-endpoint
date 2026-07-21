import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Optional

from app.dashboard_history import (
    DashboardHistory,
    DashboardHistoryStore,
    SwarmHistoryBucket,
    SwarmStateSample,
    _bucket_start_epoch_s,
    _isoformat,
)
from app.requester_identity import RequesterIdentity


SnapshotProvider = Callable[[], Awaitable[tuple[bool, Optional[str], dict[str, object]]]]
ROLLING_VIEW_WINDOWS: tuple[tuple[str, int], ...] = (
    ("1h", 60),
    ("6h", 6 * 60),
    ("24h", 24 * 60),
)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


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


_REQUESTER_VERIFICATION_RANK = {
    "unknown": 0,
    "not_provided": 1,
    "not_applicable": 1,
    "unrecognized": 2,
    "pending": 3,
    "unavailable": 4,
    "invalid": 5,
    "verified": 6,
}


def _aggregate_requester_usage(
    buckets: Iterable[SwarmHistoryBucket],
    *,
    window_minutes: int,
    total_session_requests: int,
    high_volume_threshold: int,
    burst_threshold_per_minute: int,
    many_networks_threshold: int,
) -> dict[str, object]:
    actors: dict[str, dict[str, object]] = {}
    for bucket in buckets:
        for actor_id, record in bucket.requester_usage.items():
            actor = actors.setdefault(
                actor_id,
                {
                    "actor_id": actor_id,
                    "label": str(record.get("label") or "Unknown requester"),
                    "kind": str(record.get("kind") or "unknown"),
                    "verification": str(record.get("verification") or "unknown"),
                    "fingerprint": str(record.get("fingerprint") or ""),
                    "account_name": record.get("account_name"),
                    "requests": 0,
                    "successes": 0,
                    "failures": 0,
                    "abandoned": 0,
                    "peak_requests_per_minute": 0,
                    "network_ids": set(),
                    "network_ids_overflow": False,
                    "client_kinds": {},
                    "first_seen_s": bucket.bucket_start_s,
                    "last_seen_s": bucket.bucket_start_s,
                },
            )

            current_verification = str(actor.get("verification") or "unknown")
            record_verification = str(record.get("verification") or "unknown")
            if _REQUESTER_VERIFICATION_RANK.get(record_verification, 0) >= _REQUESTER_VERIFICATION_RANK.get(
                current_verification,
                0,
            ):
                actor["label"] = str(record.get("label") or actor["label"])
                actor["kind"] = str(record.get("kind") or actor["kind"])
                actor["verification"] = record_verification
                actor["fingerprint"] = str(record.get("fingerprint") or actor["fingerprint"])
                if record.get("account_name") is not None:
                    actor["account_name"] = str(record["account_name"])

            requests = max(int(record.get("requests", 0)), 0)
            actor["requests"] = int(actor["requests"]) + requests
            actor["successes"] = int(actor["successes"]) + max(int(record.get("successes", 0)), 0)
            actor["failures"] = int(actor["failures"]) + max(int(record.get("failures", 0)), 0)
            actor["abandoned"] = int(actor["abandoned"]) + max(int(record.get("abandoned", 0)), 0)
            actor["peak_requests_per_minute"] = max(int(actor["peak_requests_per_minute"]), requests)
            actor["first_seen_s"] = min(int(actor["first_seen_s"]), bucket.bucket_start_s)
            actor["last_seen_s"] = max(int(actor["last_seen_s"]), bucket.bucket_start_s)

            network_ids = actor["network_ids"]
            if isinstance(network_ids, set):
                network_ids.update(str(item) for item in list(record.get("network_ids") or []))
            actor["network_ids_overflow"] = bool(
                actor["network_ids_overflow"] or record.get("network_ids_overflow", False)
            )
            client_kinds = actor["client_kinds"]
            if isinstance(client_kinds, dict):
                for kind, count in dict(record.get("client_kinds") or {}).items():
                    client_kinds[str(kind)] = int(client_kinds.get(str(kind), 0)) + max(int(count), 0)

    tracked_requests = sum(int(actor["requests"]) for actor in actors.values())
    peer_request_counts = [
        int(actor["requests"])
        for actor_id, actor in actors.items()
        if actor_id != "overflow" and int(actor["requests"]) > 0
    ]
    median_peer_requests = _median([float(value) for value in peer_request_counts])
    relative_threshold = max(20, int(median_peer_requests * 5))
    window_hours = max(window_minutes / 60.0, 1.0 / 60.0)

    rows: list[dict[str, object]] = []
    authenticated_accounts: set[str] = set()
    token_actors: set[str] = set()
    anonymous_actors: set[str] = set()
    authenticated_requests = 0
    anonymous_requests = 0
    invalid_token_requests = 0

    for actor_id, actor in actors.items():
        requests = int(actor["requests"])
        successes = int(actor["successes"])
        failures = int(actor["failures"])
        abandoned = int(actor["abandoned"])
        kind = str(actor["kind"])
        verification = str(actor["verification"])
        account_name = actor.get("account_name")
        network_ids = actor["network_ids"] if isinstance(actor["network_ids"], set) else set()
        client_kinds = dict(actor["client_kinds"]) if isinstance(actor["client_kinds"], dict) else {}
        automated_requests = sum(
            count for client_kind, count in client_kinds.items() if client_kind.startswith("automation:")
        )
        traffic_share_pct = round((requests / total_session_requests) * 100.0, 1) if total_session_requests else 0.0

        if requests > 0 and actor_id != "overflow":
            if actor_id.startswith("token:"):
                token_actors.add(actor_id)
            if kind == "authenticated":
                authenticated_accounts.add(str(account_name or actor_id))
                authenticated_requests += requests
            elif kind == "anonymous":
                if actor_id != "anonymous:unknown":
                    anonymous_actors.add(actor_id)
                anonymous_requests += requests
            elif kind == "invalid_token":
                invalid_token_requests += requests

        signals: list[str] = []
        if requests >= high_volume_threshold:
            signals.append(f"high volume: {requests:,} requests")
        elif requests >= relative_threshold and len(peer_request_counts) >= 2:
            signals.append(f"unusual vs peers: {requests:,} requests")
        peak_requests_per_minute = int(actor["peak_requests_per_minute"])
        if peak_requests_per_minute >= burst_threshold_per_minute:
            signals.append(f"burst: {peak_requests_per_minute:,}/min")
        if requests >= 20 and traffic_share_pct >= 50.0:
            signals.append(f"dominant traffic share: {traffic_share_pct:g}%")
        if len(network_ids) >= many_networks_threshold or bool(actor["network_ids_overflow"]):
            suffix = "+" if actor["network_ids_overflow"] else ""
            signals.append(f"many networks: {len(network_ids)}{suffix}")
        if requests >= 5 and automated_requests / max(requests, 1) >= 0.8:
            signals.append("mostly automation-like clients")
        if verification == "invalid":
            signals.append("invalid HF token")

        high_risk = any(
            signal.startswith(("high volume", "burst", "dominant traffic share"))
            for signal in signals
        )
        risk = "high" if high_risk else ("watch" if signals else "normal")
        rows.append(
            {
                "actor_id": actor_id,
                "label": actor["label"],
                "kind": kind,
                "verification": verification,
                "fingerprint": actor["fingerprint"],
                "account_name": account_name,
                "requests": requests,
                "successes": successes,
                "failures": failures,
                "abandoned": abandoned,
                "success_rate_pct": round((successes / requests) * 100.0, 1) if requests else 0.0,
                "traffic_share_pct": traffic_share_pct,
                "requests_per_hour": round(requests / window_hours, 2),
                "peak_requests_per_minute": peak_requests_per_minute,
                "network_count": len(network_ids),
                "network_count_overflow": bool(actor["network_ids_overflow"]),
                "client_kinds": dict(sorted(client_kinds.items(), key=lambda item: (-item[1], item[0]))),
                "automated_requests": automated_requests,
                "first_seen": _isoformat(int(actor["first_seen_s"])),
                "last_seen": _isoformat(int(actor["last_seen_s"])),
                "risk": risk,
                "signals": signals,
            }
        )

    rows.sort(key=lambda row: (-int(row["requests"]), str(row["label"])))
    unusual_requesters = sum(1 for row in rows if row["risk"] != "normal")
    unattributed_requests = max(total_session_requests - tracked_requests, 0)
    unique_requesters = sum(
        1
        for row in rows
        if row["actor_id"] != "overflow" and int(row["requests"]) > 0
    )

    return {
        "summary": {
            "unique_requesters_window": unique_requesters,
            "authenticated_users_window": len(authenticated_accounts),
            "tokens_window": len(token_actors),
            "anonymous_ips_window": len(anonymous_actors),
            "token_requests_window": sum(
                int(row["requests"])
                for row in rows
                if str(row["actor_id"]).startswith("token:")
            ),
            "authenticated_requests_window": authenticated_requests,
            "anonymous_requests_window": anonymous_requests,
            "invalid_token_requests_window": invalid_token_requests,
            "unattributed_requests_window": unattributed_requests,
            "unusual_requesters_window": unusual_requesters,
        },
        "tracked_requests": tracked_requests,
        "unattributed_requests": unattributed_requests,
        "median_requests_per_requester": median_peer_requests,
        "thresholds": {
            "high_volume_requests": high_volume_threshold,
            "burst_requests_per_minute": burst_threshold_per_minute,
            "many_networks": many_networks_threshold,
        },
        "leaderboard": rows[:20],
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
        max_requesters_per_bucket: int = 1000,
        requester_high_volume_threshold: int = 100,
        requester_burst_threshold_per_minute: int = 20,
        requester_many_networks_threshold: int = 5,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")

        self.snapshot_provider = snapshot_provider
        self.sample_interval_s = sample_interval_s
        self.retention_minutes = retention_minutes
        self.history_store = history_store
        self._time_fn = time_fn
        self.requester_high_volume_threshold = requester_high_volume_threshold
        self.requester_burst_threshold_per_minute = requester_burst_threshold_per_minute
        self.requester_many_networks_threshold = requester_many_networks_threshold
        self.history = DashboardHistory(
            retention_minutes=retention_minutes,
            history_store=history_store,
            restore_history_in_background=restore_history_in_background,
            flush_batch_size=flush_batch_size,
            flush_timeout_s=flush_timeout_s,
            dirty_bucket_warning_age_s=dirty_bucket_warning_age_s,
            startup_merge_delay_s=startup_merge_delay_s,
            max_requesters_per_bucket=max_requesters_per_bucket,
            time_fn=time_fn,
        )
        self._latest_sample: Optional[SwarmStateSample] = None
        self._sample_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self.capture_sample()
        await self.history.start()
        self._sample_task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> None:
        if self._sample_task is not None:
            self._sample_task.cancel()
            try:
                await self._sample_task
            except asyncio.CancelledError:
                pass
            self._sample_task = None
        await self.history.stop()

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
        await self.history.record_sample(sample)
        self._latest_sample = sample

    async def record_session_request(self, requester: RequesterIdentity | None = None) -> None:
        await self.history.record_requester_event(
            "request",
            actor_id=requester.actor_id if requester is not None else None,
            metadata=requester.history_metadata() if requester is not None else None,
        )

    async def record_session_allocation_success(self, requester: RequesterIdentity | None = None) -> None:
        await self.history.record_requester_event(
            "success",
            actor_id=requester.actor_id if requester is not None else None,
            metadata=requester.history_metadata() if requester is not None else None,
        )

    async def record_session_allocation_failure(self, requester: RequesterIdentity | None = None) -> None:
        await self.history.record_requester_event(
            "failure",
            actor_id=requester.actor_id if requester is not None else None,
            metadata=requester.history_metadata() if requester is not None else None,
        )

    async def record_session_request_abandoned(self, requester: RequesterIdentity | None = None) -> None:
        await self.history.record_requester_event(
            "abandoned",
            actor_id=requester.actor_id if requester is not None else None,
            metadata=requester.history_metadata() if requester is not None else None,
        )

    async def update_requester_identity(self, requester: RequesterIdentity) -> None:
        await self.history.update_requester_identity(
            requester.actor_id,
            requester.history_metadata(),
        )

    async def record_session_event(
        self,
        event: str,
        *,
        conversation_duration_s: Optional[float] = None,
        conversation_counted: bool = False,
    ) -> None:
        if event == "connected":
            await self.history.increment_counter("session_connected_events")
        elif event == "disconnected":
            await self.history.increment_counter("session_disconnected_events")
            if conversation_counted:
                await self.history.record_completed_conversation(max(float(conversation_duration_s or 0.0), 0.0))

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
        requesters = await self.requester_usage(window_minutes=window_minutes)
        summary.update(requesters["summary"])

        return {
            "generated_at": _isoformat(self._time_fn()),
            "window": {
                "requested": window or "6h",
                "minutes": window_minutes,
                "resolution": resolved_resolution,
            },
            "current": current.to_dict(),
            "summary": summary,
            "requesters": requesters,
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
        minute_buckets = await self.history.snapshot()
        selected = self._aggregate_recent(minute_buckets, window_minutes=window_minutes)

        return {
            "current": self._latest_sample.to_dict() if self._latest_sample is not None else None,
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
        minute_buckets = await self.history.snapshot()
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

    async def requester_usage(self, *, window_minutes: int) -> dict[str, object]:
        minute_buckets = await self.history.snapshot()
        min_bucket = _bucket_start_epoch_s(self._time_fn(), 1) - (window_minutes - 1) * 60
        selected = [bucket for bucket in minute_buckets if bucket.bucket_start_s >= min_bucket]
        total_session_requests = sum(bucket.session_requests for bucket in selected)
        return _aggregate_requester_usage(
            selected,
            window_minutes=window_minutes,
            total_session_requests=total_session_requests,
            high_volume_threshold=self.requester_high_volume_threshold,
            burst_threshold_per_minute=self.requester_burst_threshold_per_minute,
            many_networks_threshold=self.requester_many_networks_threshold,
        )

    async def rolling_series(self, *, window_minutes: int, resolution: str) -> list[dict[str, object]]:
        minute_buckets = await self.history.snapshot()
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
        return self.history.history_restore_status()

    def persistence_status(self) -> dict[str, object]:
        return self.history.persistence_status()

    def startup_merge_status(self) -> dict[str, object]:
        return self.history.startup_merge_status()

    async def _sample_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.sample_interval_s)
                await self.capture_sample()
        except asyncio.CancelledError:
            raise


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
    .table-scroll { overflow-x: auto; }
    .requester-label { min-width: 210px; }
    .requester-signals { min-width: 220px; color: var(--muted); font-size: 12px; }
    .requester-client-mix { min-width: 150px; color: var(--muted); font-size: 12px; }
    .risk-pill.normal { color: var(--good); background: rgba(17, 122, 101, 0.10); }
    .risk-pill.watch { color: var(--warm); background: rgba(217, 130, 43, 0.12); }
    .risk-pill.high { color: var(--danger); background: rgba(187, 45, 59, 0.10); }

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
        <div class="label">Traffic Attribution</div>
        <h2>Requester Usage</h2>
        <p class="muted">HF accounts are resolved asynchronously. Token and network identifiers are one-way fingerprints; raw tokens and IP addresses are never stored.</p>
        <div class="fleet-summary" id="requester-summary"></div>
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Requester</th>
                <th>Status</th>
                <th>Requests</th>
                <th>Allocated</th>
                <th>Traffic</th>
                <th>Peak</th>
                <th>Networks</th>
                <th>Clients</th>
                <th>Signals</th>
              </tr>
            </thead>
            <tbody id="requester-leaderboard"></tbody>
          </table>
        </div>
        <div class="footer-note" id="requester-detail"></div>
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
        kpiCard(`HF users / ${windowLabel}`, prettyNumber(summary.authenticated_users_window || 0), `Distinct verified Hugging Face accounts in the last ${windowLabel}`),
        kpiCard(`Anonymous IPs / ${windowLabel}`, prettyNumber(summary.anonymous_ips_window || 0), `Distinct privacy-safe network fingerprints without tokens`),
        kpiCard(`Flagged / ${windowLabel}`, prettyNumber(summary.unusual_requesters_window || 0), `Requesters with volume, burst, network, token, or automation signals`),
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

    function requesterStatusClass(row) {
      if (row.risk === 'high') return 'bad';
      if (row.risk === 'watch' || row.verification === 'pending' || row.verification === 'unavailable') return 'warm';
      return 'good';
    }

    function requesterStatusLabel(row) {
      const labels = {
        verified: 'verified HF',
        pending: 'verifying',
        unavailable: 'lookup unavailable',
        invalid: 'invalid token',
        unrecognized: 'unrecognized token',
        not_provided: 'no token',
        not_applicable: 'other',
      };
      return labels[row.verification] || row.kind || 'unknown';
    }

    function requesterClientMix(clientKinds) {
      const entries = Object.entries(clientKinds || {}).slice(0, 3);
      if (!entries.length) return 'unknown';
      return entries.map(([kind, count]) => `${kind.replace('automation:', '')}: ${prettyNumber(count)}`).join(' · ');
    }

    function renderRequesterUsage(requesters, summary) {
      const rows = requesters?.leaderboard || [];
      const windowLabel = summary.window_label || '6h';
      document.getElementById('requester-summary').innerHTML = [
        `<span class="status-pill good">${htmlEscape(prettyNumber(summary.authenticated_users_window || 0))} HF users</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(summary.tokens_window || 0))} tokens</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(summary.anonymous_ips_window || 0))} anonymous IPs</span>`,
        `<span class="status-pill ${summary.unusual_requesters_window ? 'bad' : 'good'}">${htmlEscape(prettyNumber(summary.unusual_requesters_window || 0))} flagged</span>`,
      ].join('');

      document.getElementById('requester-leaderboard').innerHTML = rows.length ? rows.map((row) => {
        const statusClass = requesterStatusClass(row);
        const networks = `${prettyNumber(row.network_count || 0)}${row.network_count_overflow ? '+' : ''}`;
        const signals = (row.signals || []).join(' · ') || 'No unusual signal';
        return `
          <tr>
            <td class="requester-label">
              <div><strong>${htmlEscape(row.label || 'Unknown requester')}</strong></div>
              <div class="muted mono" style="margin-top:4px;">${htmlEscape(row.actor_id || '')}</div>
            </td>
            <td>
              <span class="status-pill ${statusClass}">${htmlEscape(requesterStatusLabel(row))}</span>
              <div style="margin-top:6px;"><span class="tiny-pill risk-pill ${htmlEscape(row.risk || 'normal')}">${htmlEscape(row.risk || 'normal')}</span></div>
            </td>
            <td><strong>${htmlEscape(prettyNumber(row.requests || 0))}</strong><div class="muted">${htmlEscape(prettyNumber(row.requests_per_hour || 0))}/h</div></td>
            <td>${htmlEscape(prettyNumber(row.successes || 0))}<div class="muted">${htmlEscape(row.success_rate_pct || 0)}%</div></td>
            <td>${htmlEscape(row.traffic_share_pct || 0)}%</td>
            <td>${htmlEscape(prettyNumber(row.peak_requests_per_minute || 0))}/min</td>
            <td>${htmlEscape(networks)}</td>
            <td class="requester-client-mix">${htmlEscape(requesterClientMix(row.client_kinds))}</td>
            <td class="requester-signals">${htmlEscape(signals)}</td>
          </tr>
        `;
      }).join('') : '<tr><td colspan="9" class="muted">No attributed session requests in this window yet.</td></tr>';

      const unattributed = Number(requesters?.unattributed_requests || 0);
      document.getElementById('requester-detail').textContent = unattributed
        ? `${prettyNumber(unattributed)} request(s) in the last ${windowLabel} predate attribution or could not be attributed.`
        : `All recorded session requests in the last ${windowLabel} have requester attribution.`;
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
      renderRequesterUsage(payload.requesters || {}, summary);
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

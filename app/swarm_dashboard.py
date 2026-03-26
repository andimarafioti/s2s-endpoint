import asyncio
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional


SnapshotProvider = Callable[[], Awaitable[tuple[bool, Optional[str], dict[str, object]]]]


def _normalize_status(status: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(status).lower())


def _bucket_start_epoch_s(epoch_s: float, bucket_minutes: int) -> int:
    bucket_seconds = bucket_minutes * 60
    return int(epoch_s // bucket_seconds) * bucket_seconds


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
        }


class SwarmDashboard:
    def __init__(
        self,
        *,
        snapshot_provider: SnapshotProvider,
        sample_interval_s: float = 15.0,
        retention_minutes: int = 7 * 24 * 60,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        if sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be > 0")
        if retention_minutes < 60:
            raise ValueError("retention_minutes must be >= 60")

        self.snapshot_provider = snapshot_provider
        self.sample_interval_s = sample_interval_s
        self.retention_minutes = retention_minutes
        self._time_fn = time_fn
        self._lock = asyncio.Lock()
        self._history: "OrderedDict[int, SwarmHistoryBucket]" = OrderedDict()
        self._latest_sample: Optional[SwarmStateSample] = None
        self._sample_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self.capture_sample()
        self._sample_task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> None:
        if self._sample_task is not None:
            self._sample_task.cancel()
            try:
                await self._sample_task
            except asyncio.CancelledError:
                pass
            self._sample_task = None

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
            self._prune_unlocked(sample.captured_at_s)

    async def record_session_request(self) -> None:
        await self._increment_counter("session_requests")

    async def record_session_allocation_success(self) -> None:
        await self._increment_counter("session_allocation_successes")

    async def record_session_allocation_failure(self) -> None:
        await self._increment_counter("session_allocation_failures")

    async def record_session_event(self, event: str) -> None:
        if event == "connected":
            await self._increment_counter("session_connected_events")
        elif event == "disconnected":
            await self._increment_counter("session_disconnected_events")

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
        summary = await self.summary()

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
            "retention_minutes": self.retention_minutes,
        }

    async def summary(self) -> dict[str, object]:
        async with self._lock:
            latest = self._latest_sample
            minute_buckets = list(self._history.values())

        one_hour = self._aggregate_recent(minute_buckets, window_minutes=60)
        one_day = self._aggregate_recent(minute_buckets, window_minutes=24 * 60)

        return {
            "current": latest.to_dict() if latest is not None else None,
            "session_requests_last_hour": one_hour["session_requests"],
            "session_failures_last_hour": one_hour["session_allocation_failures"],
            "session_successes_last_hour": one_hour["session_allocation_successes"],
            "session_connects_last_hour": one_hour["session_connected_events"],
            "session_disconnects_last_hour": one_hour["session_disconnected_events"],
            "peak_connected_sessions_24h": one_day["peak_connected_sessions"],
            "peak_running_endpoints_24h": one_day["peak_running_endpoints"],
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
                        }
                    )
                else:
                    points.append(bucket.as_minute_point())
            return points

        return self._aggregate_hourly(minute_map, start_bucket, end_bucket)

    def html(self) -> str:
        return _dashboard_html()

    async def _sample_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.sample_interval_s)
                await self.capture_sample()
        except asyncio.CancelledError:
            raise

    async def _increment_counter(self, field_name: str) -> None:
        now = self._time_fn()
        async with self._lock:
            bucket = self._get_bucket_unlocked(now)
            setattr(bucket, field_name, getattr(bucket, field_name) + 1)
            self._prune_unlocked(now)

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

    def _aggregate_recent(self, minute_buckets: list[SwarmHistoryBucket], *, window_minutes: int) -> dict[str, int]:
        now = self._time_fn()
        min_bucket = _bucket_start_epoch_s(now, 1) - (window_minutes - 1) * 60
        selected = [bucket for bucket in minute_buckets if bucket.bucket_start_s >= min_bucket]
        return {
            "session_requests": sum(bucket.session_requests for bucket in selected),
            "session_allocation_successes": sum(bucket.session_allocation_successes for bucket in selected),
            "session_allocation_failures": sum(bucket.session_allocation_failures for bucket in selected),
            "session_connected_events": sum(bucket.session_connected_events for bucket in selected),
            "session_disconnected_events": sum(bucket.session_disconnected_events for bucket in selected),
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
                }
            else:
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
                }
            points.append(point)
            current_hour_s += 3600

        return points


def _dashboard_html() -> str:
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

    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 8px;
      background: currentColor;
      vertical-align: middle;
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

    .muted { color: var(--muted); }
    .mono { font-family: "Menlo", "Consolas", monospace; }
    .footer-note { margin-top: 14px; color: var(--muted); font-size: 13px; }

    @media (max-width: 1100px) {
      .hero, .kpis { grid-template-columns: 1fr; }
      .span-4, .span-6, .span-8 { grid-column: span 12; }
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
          <span>Single-replica LB in-memory history; resets on restart.</span>
        </div>
      </div>
      <div class="panel hero-stat-grid" id="hero-stats"></div>
    </section>

    <div class="controls">
      <button class="toggle active" data-window="1h">1h</button>
      <button class="toggle" data-window="6h">6h</button>
      <button class="toggle" data-window="24h">24h</button>
      <button class="toggle" data-window="7d">7d</button>
      <button class="toggle" data-resolution="minute">Minute</button>
      <button class="toggle" data-resolution="hour">Hour</button>
    </div>

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
        <div class="footer-note" id="health-detail"></div>
      </div>

      <div class="panel card span-12">
        <div class="label">Current Fleet</div>
        <h2>Endpoint Table</h2>
        <div style="overflow:auto;">
          <table>
            <thead>
              <tr>
                <th>Endpoint</th>
                <th>Status</th>
                <th>Active Sessions</th>
                <th>Free Slots</th>
                <th>Flags</th>
              </tr>
            </thead>
            <tbody id="endpoint-table"></tbody>
          </table>
        </div>
      </div>
    </section>
  </div>

  <script>
    const windowButtons = [...document.querySelectorAll('[data-window]')];
    const resolutionButtons = [...document.querySelectorAll('[data-resolution]')];
    let selectedWindow = '6h';
    let selectedResolution = '';
    let refreshHandle = null;

    function prettyNumber(value) {
      const numeric = Number(value || 0);
      if (Math.abs(numeric - Math.round(numeric)) < 0.01) {
        return String(Math.round(numeric));
      }
      return numeric.toFixed(1);
    }

    function statusClass(healthy, errors) {
      if (!healthy || errors > 0) return 'bad';
      return 'good';
    }

    function setActiveButtons() {
      for (const button of windowButtons) {
        button.classList.toggle('active', button.dataset.window === selectedWindow);
      }
      for (const button of resolutionButtons) {
        button.classList.toggle('active', button.dataset.resolution === selectedResolution);
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
      const stats = [
        ['Running', current.running_endpoints],
        ['Connected', current.connected_sessions],
        ['Spare Slots', current.effective_free_slots],
        ['Req / Last Hour', summary.session_requests_last_hour],
      ];
      document.getElementById('hero-stats').innerHTML = stats.map(([label, value]) => `
        <div class="hero-stat">
          <div class="label">${htmlEscape(label)}</div>
          <strong>${htmlEscape(prettyNumber(value))}</strong>
        </div>
      `).join('');
    }

    function renderKpis(current, summary) {
      document.getElementById('kpis').innerHTML = [
        kpiCard('Connected users', prettyNumber(current.connected_sessions), 'Current live websocket sessions'),
        kpiCard('Pending joins', prettyNumber(current.pending_sessions), 'Reserved sessions waiting to connect'),
        kpiCard('Requests / 60m', prettyNumber(summary.session_requests_last_hour), 'POST /session requests in the last hour'),
        kpiCard('Failures / 60m', prettyNumber(summary.session_failures_last_hour), 'Allocation failures in the last hour'),
        kpiCard('Peak users / 24h', prettyNumber(summary.peak_connected_sessions_24h), 'Highest concurrent connected sessions'),
        kpiCard('Peak running / 24h', prettyNumber(summary.peak_running_endpoints_24h), 'Highest active compute endpoint count'),
        kpiCard('Free slots', prettyNumber(current.free_slots), 'Currently running free slots'),
        kpiCard('Errors', prettyNumber(current.errors_count), current.healthy ? 'No active router errors' : (current.detail || 'Swarm is degraded')),
      ].join('');
    }

    function renderHealth(current) {
      const badges = [
        `<span class="status-pill ${statusClass(current.healthy, current.errors_count)}">${current.healthy ? 'Healthy' : 'Degraded'}</span>`,
        `<span class="status-pill ${current.warming_endpoints || current.transitioning_endpoints ? 'warm' : 'good'}">warming ${htmlEscape(prettyNumber(current.warming_endpoints + current.transitioning_endpoints))}</span>`,
        `<span class="status-pill">${htmlEscape(prettyNumber(current.parked_endpoints))} parked</span>`,
      ];
      document.getElementById('health-badges').innerHTML = badges.join(' ');
      document.getElementById('health-detail').textContent = current.detail || 'Swarm snapshot is current.';
    }

    function renderEndpointTable(endpoints) {
      const rows = endpoints.map((endpoint) => {
        const rawStatus = endpoint.status || 'unknown';
        const normalized = String(rawStatus).toLowerCase();
        const klass = normalized.includes('running') ? 'good' : (normalized.includes('init') || normalized.includes('updat') || endpoint.waking ? 'warm' : 'bad');
        const flags = [
          endpoint.waking ? 'waking' : null,
          endpoint.parking ? 'parking' : null,
          endpoint.last_error ? 'error' : null,
        ].filter(Boolean).join(', ') || '—';
        return `
          <tr>
            <td class="mono">${htmlEscape(endpoint.name)}</td>
            <td><span class="status-pill ${klass}">${htmlEscape(rawStatus)}</span></td>
            <td>${htmlEscape(prettyNumber(endpoint.active_sessions || 0))}</td>
            <td>${htmlEscape(prettyNumber(endpoint.free_slots || 0))}</td>
            <td class="muted">${htmlEscape(flags)}</td>
          </tr>
        `;
      }).join('');
      document.getElementById('endpoint-table').innerHTML = rows;
    }

    function drawChart(canvas, points, seriesConfig) {
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const padding = { top: 18, right: 18, bottom: 28, left: 34 };
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const values = points.flatMap((point) => seriesConfig.map((series) => Number(point[series.key] || 0)));
      const maxValue = Math.max(1, ...values);
      const xStep = points.length > 1 ? plotWidth / (points.length - 1) : 0;

      ctx.strokeStyle = 'rgba(24, 33, 37, 0.12)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i += 1) {
        const y = padding.top + (plotHeight * i) / 4;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(24, 33, 37, 0.55)';
      ctx.font = '12px Menlo, Consolas, monospace';
      ctx.textAlign = 'right';
      for (let i = 0; i <= 4; i += 1) {
        const value = maxValue - (maxValue * i) / 4;
        const y = padding.top + (plotHeight * i) / 4 + 4;
        ctx.fillText(prettyNumber(value), padding.left - 8, y);
      }

      seriesConfig.forEach((series) => {
        ctx.strokeStyle = series.color;
        ctx.lineWidth = series.width || 2.5;
        ctx.beginPath();
        points.forEach((point, index) => {
          const x = padding.left + xStep * index;
          const y = padding.top + plotHeight - (plotHeight * Number(point[series.key] || 0)) / maxValue;
          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
      });

      if (points.length > 1) {
        ctx.fillStyle = 'rgba(24, 33, 37, 0.55)';
        ctx.textAlign = 'center';
        const first = new Date(points[0].timestamp);
        const last = new Date(points[points.length - 1].timestamp);
        ctx.fillText(first.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), padding.left, height - 8);
        ctx.fillText(last.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), width - padding.right, height - 8);
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
      const current = payload.current;
      const summary = payload.summary;
      const series = payload.series;

      document.getElementById('generated-at').textContent =
        `Updated ${new Date(payload.generated_at).toLocaleString()} • ${payload.window.resolution} timeline • last ${payload.window.requested}`;

      renderHeroStats(current, summary);
      renderKpis(current, summary);
      renderHealth(current);
      renderEndpointTable(current.endpoints || []);

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
        if (selectedWindow === '7d' && !selectedResolution) {
          selectedResolution = 'hour';
        }
        setActiveButtons();
        loadDashboard().catch((error) => console.error(error));
      });
    });

    resolutionButtons.forEach((button) => {
      button.addEventListener('click', () => {
        selectedResolution = button.dataset.resolution === selectedResolution ? '' : button.dataset.resolution;
        setActiveButtons();
        loadDashboard().catch((error) => console.error(error));
      });
    });

    setActiveButtons();
    loadDashboard().catch((error) => console.error(error));
    scheduleRefresh();
  </script>
</body>
</html>
"""

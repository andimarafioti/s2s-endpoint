import asyncio
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.dashboard_history_store import HuggingFaceBucketHistoryStore, ReadOnlyDashboardHistoryStore
from app.swarm_dashboard import (
    SwarmDashboard,
    SwarmHistoryBucket,
    SwarmStateSample,
)


class FakeClock:
    def __init__(self, now: float):
        self._now = now

    def now(self) -> float:
        return self._now

    def set(self, now: float) -> None:
        self._now = now


class FakeSnapshotProvider:
    def __init__(self, payload):
        self.payload = payload

    async def __call__(self):
        return self.payload


class FakeHistoryStore:
    def __init__(self, initial_buckets=None):
        self.saved = {
            bucket.bucket_start_s: bucket.to_dict()
            for bucket in (initial_buckets or [])
        }
        self.write_calls = []
        self.load_calls = 0

    def load_recent(self, *, retention_minutes: int, now_epoch_s: float):
        self.load_calls += 1
        min_bucket = int(now_epoch_s // 60) * 60 - (retention_minutes - 1) * 60
        return [
            SwarmHistoryBucket.from_dict(payload)
            for bucket_start_s, payload in sorted(self.saved.items())
            if bucket_start_s >= min_bucket
        ]

    def write_buckets(self, buckets):
        self.write_calls.append([bucket.bucket_start_s for bucket in buckets])
        for bucket in buckets:
            self.saved[bucket.bucket_start_s] = bucket.to_dict()


class DayRolloverHistoryStore(FakeHistoryStore):
    def __init__(self, initial_buckets=None):
        super().__init__(initial_buckets=initial_buckets)
        self.day_write_calls = []

    def write_day_buckets(self, *, day_start_s: int, buckets):
        self.day_write_calls.append((day_start_s, [bucket.bucket_start_s for bucket in buckets]))
        return f"days/{day_start_s}.json"


class SlowHistoryStore(FakeHistoryStore):
    def __init__(self, *, delay_s: float, initial_buckets=None):
        super().__init__(initial_buckets=initial_buckets)
        self.delay_s = delay_s

    def load_recent(self, *, retention_minutes: int, now_epoch_s: float):
        time.sleep(self.delay_s)
        return super().load_recent(retention_minutes=retention_minutes, now_epoch_s=now_epoch_s)


class BlockingFirstWriteHistoryStore(FakeHistoryStore):
    def __init__(self):
        super().__init__()
        self.first_write_started = threading.Event()
        self.release_first_write = threading.Event()
        self.write_attempts = 0

    def write_buckets(self, buckets):
        self.write_attempts += 1
        if self.write_attempts == 1:
            self.first_write_started.set()
            self.release_first_write.wait(timeout=1.0)
        super().write_buckets(buckets)


class FakeBucketItem:
    def __init__(self, path: str, *, xet_hash: str | None = None):
        self.path = path
        self.xet_hash = xet_hash


class FakeBucketApi:
    def __init__(self, files: dict[str, dict[str, object]], *, missing_xet_hash_paths: set[str] | None = None):
        self.files = files
        self.missing_xet_hash_paths = set(missing_xet_hash_paths or [])
        self.list_prefixes = []
        self.downloads = []
        self.batch_adds = []
        self.batch_deletes = []
        self.batch_calls = []

    def list_bucket_tree(self, bucket_id, *, prefix=None, recursive=None, token=None):
        self.list_prefixes.append(prefix)
        return [
            FakeBucketItem(path, xet_hash=None if path in self.missing_xet_hash_paths else path)
            for path in sorted(self.files)
            if prefix is None or path.startswith(prefix)
        ]

    def download_bucket_files(self, bucket_id, *, files, raise_on_missing_files, token=None):
        self.downloads.extend(files)
        for remote_path, local_path in files:
            payload = self.files.get(remote_path)
            if payload is None:
                continue
            local_path.write_text(json.dumps(payload))

    def batch_bucket_files(self, bucket_id, *, add=None, copy=None, delete=None, token=None):
        if add:
            self.batch_calls.append(list(add))
            self.batch_adds.extend(add)
            for source, path in add:
                if isinstance(source, bytes):
                    payload = json.loads(source.decode("utf-8"))
                else:
                    payload = json.loads(Path(source).read_text())
                self.files[path] = payload
        if copy:
            self.batch_calls.append(list(copy))
            for source_repo_type, source_repo_id, xet_hash, path in copy:
                if source_repo_type != "bucket" or source_repo_id != bucket_id:
                    continue
                payload = self.files.get(xet_hash)
                if payload is not None:
                    self.files[path] = payload
        if delete:
            self.batch_deletes.extend(delete)
            for path in delete:
                self.files.pop(path, None)


def _health_snapshot(*, connected: int, pending: int, running: int, waking: int, free_slots: int, effective_free_slots: int):
    endpoints = [
        {
            "name": "reachy-s2s-01",
            "status": "running",
            "waking": False,
            "parking": False,
            "active_sessions": connected,
            "free_slots": max(free_slots - 1, 0),
            "url": "https://reachy-s2s-01.example",
            "last_error": None,
        },
        {
            "name": "reachy-s2s-02",
            "status": "running",
            "waking": False,
            "parking": False,
            "active_sessions": 0,
            "free_slots": 1,
            "url": "https://reachy-s2s-02.example",
            "last_error": None,
        },
        {
            "name": "reachy-s2s-03",
            "status": "initializing",
            "waking": True,
            "parking": False,
            "active_sessions": 0,
            "free_slots": 0,
            "url": None,
            "last_error": None,
        },
        {
            "name": "reachy-s2s-04",
            "status": "paused",
            "waking": False,
            "parking": False,
            "active_sessions": 0,
            "free_slots": 0,
            "url": None,
            "last_error": None,
        },
    ]
    return (
        True,
        None,
        {
            "pending_sessions": pending,
            "connected_sessions": connected,
            "router": {
                "running_endpoints": running,
                "waking_endpoints": waking,
                "active_sessions": connected,
                "free_slots": free_slots,
                "effective_free_slots": effective_free_slots,
                "errors": [],
                "endpoints": endpoints,
            },
        },
    )


def _day_start(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def _bucket_store(api: FakeBucketApi) -> HuggingFaceBucketHistoryStore:
    store = HuggingFaceBucketHistoryStore(bucket_id="org/dashboard", prefix="reachy-s2s-lb")
    store._list_bucket_tree = api.list_bucket_tree
    store._download_bucket_files = api.download_bucket_files
    store._batch_bucket_files = api.batch_bucket_files
    return store


class SwarmDashboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_data_exposes_minute_series_and_event_counters(self):
        clock = FakeClock(2 * 3600)
        provider = FakeSnapshotProvider(
            _health_snapshot(
                connected=2,
                pending=1,
                running=2,
                waking=1,
                free_slots=2,
                effective_free_slots=3,
            )
        )
        dashboard = SwarmDashboard(
            snapshot_provider=provider,
            sample_interval_s=15,
            retention_minutes=24 * 60,
            time_fn=clock.now,
        )

        await dashboard.capture_sample()
        await dashboard.record_session_request()
        await dashboard.record_session_allocation_success()
        await dashboard.record_session_event("connected")
        await dashboard.record_session_event(
            "disconnected",
            conversation_duration_s=150.0,
            conversation_counted=True,
        )

        payload = await dashboard.data(window="60m", resolution="minute")
        point = payload["series"][-1]
        current = payload["current"]

        self.assertEqual(current["running_endpoints"], 2)
        self.assertEqual(current["transitioning_endpoints"], 1)
        self.assertEqual(current["parked_endpoints"], 1)
        self.assertEqual(point["session_requests"], 1)
        self.assertEqual(point["session_allocation_successes"], 1)
        self.assertEqual(point["session_connected_events"], 1)
        self.assertEqual(point["completed_conversations"], 1)
        self.assertEqual(point["avg_conversation_duration_s"], 150.0)
        self.assertEqual(point["max_conversation_duration_min"], 2.5)
        self.assertEqual(payload["summary"]["window_label"], "60m")
        self.assertEqual(payload["summary"]["session_requests_window"], 1)
        self.assertEqual(payload["summary"]["conversations_completed_window"], 1)
        self.assertEqual(payload["summary"]["active_conversation_minutes_window"], 2.0)
        self.assertEqual(payload["summary"]["active_conversation_hours_window"], 0.03)
        self.assertEqual(payload["summary"]["active_conversation_days_window"], 0.001)
        self.assertEqual(payload["summary"]["avg_conversation_duration_window_s"], 150.0)
        self.assertEqual(payload["summary"]["peak_connected_sessions_window"], 2)
        self.assertFalse(payload["history_persistence"]["enabled"])
        self.assertEqual(payload["history_persistence"]["dirty_bucket_count"], 0)

    async def test_hourly_series_averages_state_metrics_and_sums_events(self):
        clock = FakeClock(3 * 3600)
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=0,
                waking=0,
                free_slots=0,
                effective_free_slots=0,
            )),
            sample_interval_s=15,
            retention_minutes=24 * 60,
            time_fn=clock.now,
        )

        await dashboard.record_sample(
            SwarmStateSample(
                captured_at_s=3 * 3600 + 5 * 60,
                healthy=True,
                detail=None,
                total_endpoints=4,
                running_endpoints=2,
                warming_endpoints=1,
                transitioning_endpoints=1,
                parked_endpoints=1,
                connected_sessions=1,
                pending_sessions=0,
                free_slots=2,
                effective_free_slots=3,
                router_active_sessions=1,
                errors_count=0,
                endpoints=[],
            )
        )
        clock.set(3 * 3600 + 10 * 60)
        await dashboard.record_session_request()
        await dashboard.record_session_allocation_success()

        await dashboard.record_sample(
            SwarmStateSample(
                captured_at_s=3 * 3600 + 35 * 60,
                healthy=True,
                detail=None,
                total_endpoints=4,
                running_endpoints=4,
                warming_endpoints=0,
                transitioning_endpoints=0,
                parked_endpoints=0,
                connected_sessions=3,
                pending_sessions=1,
                free_slots=1,
                effective_free_slots=1,
                router_active_sessions=3,
                errors_count=1,
                endpoints=[],
            )
        )
        clock.set(3 * 3600 + 40 * 60)
        await dashboard.record_session_request()
        await dashboard.record_session_allocation_failure()
        await dashboard.record_session_event(
            "disconnected",
            conversation_duration_s=600.0,
            conversation_counted=True,
        )
        clock.set(3 * 3600 + 59 * 60)

        points = await dashboard.series(window_minutes=60, resolution="hour")
        point = points[-1]

        self.assertEqual(point["running_endpoints"], 3.0)
        self.assertEqual(point["connected_sessions"], 2.0)
        self.assertEqual(point["session_requests"], 2)
        self.assertEqual(point["session_allocation_successes"], 1)
        self.assertEqual(point["session_allocation_failures"], 1)
        self.assertEqual(point["session_disconnected_events"], 1)
        self.assertEqual(point["completed_conversations"], 1)
        self.assertEqual(point["avg_conversation_duration_s"], 600.0)
        self.assertEqual(point["max_conversation_duration_min"], 10.0)

    def test_dashboard_html_generates_rolling_charts_from_descriptors(self):
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=0,
                waking=0,
                free_slots=0,
                effective_free_slots=0,
            )),
        )

        html = dashboard.html()

        self.assertIn('id="rolling-charts"', html)
        self.assertIn("const rollingCharts = [", html)
        self.assertIn("Maximum Connected Users", html)
        self.assertIn("Median Duration", html)
        self.assertIn("renderRollingChartCards();", html)

    async def test_summary_peak_connected_sessions_uses_bucket_max(self):
        now_s = 6 * 3600
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=0,
                waking=0,
                free_slots=0,
                effective_free_slots=0,
            )),
            time_fn=FakeClock(now_s).now,
        )
        bucket = SwarmHistoryBucket(bucket_start_s=now_s)
        bucket.sample_count = 2
        bucket.connected_sessions_last = 1
        bucket.connected_sessions_sum = 6
        bucket.connected_sessions_max = 5
        dashboard._history[bucket.bucket_start_s] = bucket

        summary = await dashboard.summary(window_minutes=60, requested_window="1h")

        self.assertEqual(summary["peak_connected_sessions_window"], 5)

    async def test_rolling_series_reports_conversation_and_connected_user_windows(self):
        now_s = 48 * 3600
        clock = FakeClock(now_s)
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=0,
                waking=0,
                free_slots=0,
                effective_free_slots=0,
            )),
            sample_interval_s=15,
            retention_minutes=48 * 60,
            time_fn=clock.now,
        )

        def add_bucket(
            *,
            bucket_start_s: int,
            completed_conversations: int,
            duration_total_s: float,
            duration_samples_s: list[float],
            connected_sessions_sum: float,
            connected_sessions_max: int,
            sample_count: int,
        ) -> None:
            bucket = SwarmHistoryBucket(bucket_start_s=bucket_start_s)
            bucket.completed_conversations = completed_conversations
            bucket.completed_conversation_duration_total_s = duration_total_s
            bucket.completed_conversation_duration_samples_s = duration_samples_s
            bucket.connected_sessions_sum = connected_sessions_sum
            bucket.connected_sessions_max = connected_sessions_max
            bucket.sample_count = sample_count
            dashboard._history[bucket_start_s] = bucket

        add_bucket(
            bucket_start_s=now_s - 23 * 3600,
            completed_conversations=1,
            duration_total_s=60.0,
            duration_samples_s=[60.0],
            connected_sessions_sum=2.0,
            connected_sessions_max=11,
            sample_count=1,
        )
        add_bucket(
            bucket_start_s=now_s - 5 * 3600,
            completed_conversations=2,
            duration_total_s=600.0,
            duration_samples_s=[120.0, 480.0],
            connected_sessions_sum=10.0,
            connected_sessions_max=7,
            sample_count=2,
        )
        add_bucket(
            bucket_start_s=now_s - 30 * 60,
            completed_conversations=3,
            duration_total_s=1800.0,
            duration_samples_s=[300.0, 600.0, 900.0],
            connected_sessions_sum=8.0,
            connected_sessions_max=9,
            sample_count=1,
        )

        points = await dashboard.rolling_series(window_minutes=24 * 60, resolution="hour")
        point = points[-1]

        self.assertEqual(point["completed_conversations_1h"], 3)
        self.assertEqual(point["completed_conversations_6h"], 5)
        self.assertEqual(point["completed_conversations_24h"], 6)
        self.assertEqual(point["active_conversation_minutes_1h"], 8.0)
        self.assertEqual(point["active_conversation_minutes_6h"], 13.0)
        self.assertEqual(point["active_conversation_minutes_24h"], 15.0)
        self.assertEqual(point["active_conversation_hours_1h"], 0.13)
        self.assertEqual(point["active_conversation_hours_6h"], 0.22)
        self.assertEqual(point["active_conversation_hours_24h"], 0.25)
        self.assertEqual(point["active_conversation_days_1h"], 0.006)
        self.assertEqual(point["active_conversation_days_6h"], 0.009)
        self.assertEqual(point["active_conversation_days_24h"], 0.01)
        self.assertEqual(point["avg_conversation_duration_min_1h"], 10.0)
        self.assertEqual(point["avg_conversation_duration_min_6h"], 8.0)
        self.assertEqual(point["avg_conversation_duration_min_24h"], 6.83)
        self.assertEqual(point["median_conversation_duration_min_1h"], 10.0)
        self.assertEqual(point["median_conversation_duration_min_6h"], 8.0)
        self.assertEqual(point["median_conversation_duration_min_24h"], 6.5)
        self.assertEqual(point["connected_sessions_avg_1h"], 8.0)
        self.assertEqual(point["connected_sessions_avg_6h"], 6.0)
        self.assertEqual(point["connected_sessions_avg_24h"], 5.0)
        self.assertEqual(point["connected_sessions_max_1h"], 9)
        self.assertEqual(point["connected_sessions_max_6h"], 9)
        self.assertEqual(point["connected_sessions_max_24h"], 11)

    async def test_prunes_buckets_older_than_retention_window(self):
        clock = FakeClock(10 * 3600)
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            sample_interval_s=15,
            retention_minutes=60,
            time_fn=clock.now,
        )

        await dashboard.record_sample(
            SwarmStateSample(
                captured_at_s=10 * 3600,
                healthy=True,
                detail=None,
                total_endpoints=1,
                running_endpoints=1,
                warming_endpoints=0,
                transitioning_endpoints=0,
                parked_endpoints=0,
                connected_sessions=0,
                pending_sessions=0,
                free_slots=1,
                effective_free_slots=1,
                router_active_sessions=0,
                errors_count=0,
                endpoints=[],
            )
        )

        clock.set(11 * 3600 + 1 * 60)
        await dashboard.record_sample(
            SwarmStateSample(
                captured_at_s=clock.now(),
                healthy=True,
                detail=None,
                total_endpoints=1,
                running_endpoints=1,
                warming_endpoints=0,
                transitioning_endpoints=0,
                parked_endpoints=0,
                connected_sessions=0,
                pending_sessions=0,
                free_slots=1,
                effective_free_slots=1,
                router_active_sessions=0,
                errors_count=0,
                endpoints=[],
            )
        )

        self.assertEqual(len(dashboard._history), 1)

    async def test_persists_closed_buckets_to_history_store(self):
        clock = FakeClock(2 * 3600)
        store = FakeHistoryStore()
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=1,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            sample_interval_s=15,
            retention_minutes=24 * 60,
            history_store=store,
            time_fn=clock.now,
        )

        await dashboard.record_sample(
            SwarmStateSample(
                captured_at_s=clock.now(),
                healthy=True,
                detail=None,
                total_endpoints=1,
                running_endpoints=1,
                warming_endpoints=0,
                transitioning_endpoints=0,
                parked_endpoints=0,
                connected_sessions=1,
                pending_sessions=0,
                free_slots=1,
                effective_free_slots=1,
                router_active_sessions=1,
                errors_count=0,
                endpoints=[],
            )
        )
        await dashboard.record_session_request()
        await dashboard.record_session_event(
            "disconnected",
            conversation_duration_s=90.0,
            conversation_counted=True,
        )
        clock.set(clock.now() + 60)
        await dashboard.record_sample(
            SwarmStateSample(
                captured_at_s=clock.now(),
                healthy=True,
                detail=None,
                total_endpoints=1,
                running_endpoints=1,
                warming_endpoints=0,
                transitioning_endpoints=0,
                parked_endpoints=0,
                connected_sessions=0,
                pending_sessions=0,
                free_slots=1,
                effective_free_slots=1,
                router_active_sessions=0,
                errors_count=0,
                endpoints=[],
            )
        )
        if dashboard._flush_task is not None:
            await dashboard._flush_task

        persisted = SwarmHistoryBucket.from_dict(store.saved[2 * 3600])
        self.assertEqual(persisted.session_requests, 1)
        self.assertEqual(persisted.connected_sessions_last, 1)
        self.assertEqual(persisted.connected_sessions_max, 1)
        self.assertEqual(persisted.completed_conversations, 1)
        self.assertEqual(persisted.completed_conversation_duration_total_s, 90.0)
        self.assertEqual(persisted.completed_conversation_duration_samples_s, [90.0])

    async def test_flushes_dirty_buckets_in_bounded_batches(self):
        clock = FakeClock(5 * 60)
        store = FakeHistoryStore()
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            retention_minutes=60,
            history_store=store,
            flush_batch_size=2,
            time_fn=clock.now,
        )
        for bucket_start_s in range(0, 5 * 60, 60):
            dashboard._history[bucket_start_s] = SwarmHistoryBucket(bucket_start_s=bucket_start_s)
            dashboard._dirty_bucket_starts.add(bucket_start_s)

        await dashboard._flush_dirty_buckets(include_open_bucket=False)

        self.assertEqual(store.write_calls, [[0, 60], [120, 180], [240]])
        status = dashboard.persistence_status()
        self.assertEqual(status["dirty_bucket_count"], 0)
        self.assertIsNotNone(status["last_flush_started_at"])
        self.assertIsNotNone(status["last_flush_finished_at"])
        self.assertIsNone(status["last_flush_error"])

    async def test_stalled_flush_remains_single_flight_and_writes_newest_snapshot_last(self):
        clock = FakeClock(2 * 60)
        store = BlockingFirstWriteHistoryStore()
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            retention_minutes=60,
            history_store=store,
            flush_timeout_s=0.01,
            time_fn=clock.now,
        )
        dashboard._history[0] = SwarmHistoryBucket(bucket_start_s=0, session_requests=1)
        dashboard._dirty_bucket_starts.add(0)

        with self.assertLogs("s2s-endpoint", level="ERROR"):
            dashboard._schedule_flush_unlocked(include_open_bucket=False)
            while not store.first_write_started.is_set():
                await asyncio.sleep(0.001)
            await asyncio.sleep(0.02)

            flush_task = dashboard._flush_task
            status = dashboard.persistence_status()
            dashboard._history[0].session_requests = 2
            dashboard._dirty_bucket_starts.add(0)
            dashboard._schedule_flush_unlocked(include_open_bucket=False)

        self.assertTrue(status["flush_write_in_flight"])
        self.assertTrue(status["flush_stalled"])
        self.assertIsNotNone(status["flush_stalled_since_at"])
        self.assertIsNotNone(status["flush_stalled_age_s"])
        self.assertIs(dashboard._flush_task, flush_task)
        self.assertEqual(store.write_attempts, 1)

        store.release_first_write.set()
        await flush_task

        self.assertEqual(store.write_attempts, 2)
        self.assertEqual(store.saved[0]["session_requests"], 2)
        self.assertEqual(dashboard.persistence_status()["dirty_bucket_count"], 0)
        self.assertFalse(dashboard.persistence_status()["flush_write_in_flight"])
        self.assertFalse(dashboard.persistence_status()["flush_stalled"])
        self.assertIsNone(dashboard.persistence_status()["last_flush_error"])

    async def test_warns_when_dirty_dashboard_buckets_are_stale(self):
        clock = FakeClock(10 * 60)
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            retention_minutes=60,
            history_store=FakeHistoryStore(),
            dirty_bucket_warning_age_s=300,
            time_fn=clock.now,
        )
        dashboard._history[0] = SwarmHistoryBucket(bucket_start_s=0)
        dashboard._dirty_bucket_starts.add(0)
        dashboard._flush_task_started_at_monotonic_s = time.monotonic()
        dashboard._flush_task = asyncio.create_task(asyncio.sleep(1))
        try:
            with self.assertLogs("s2s-endpoint", level="WARNING") as logs:
                dashboard._schedule_flush_unlocked(include_open_bucket=False)
            status = dashboard.persistence_status()
        finally:
            dashboard._flush_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await dashboard._flush_task

        self.assertIn("oldest dirty bucket age is 600.0s", logs.output[0])
        self.assertEqual(status["oldest_dirty_bucket_age_s"], 600.0)
        self.assertIsNotNone(status["flush_task_age_s"])

    async def test_restores_persisted_history_from_store_on_start(self):
        bucket = SwarmHistoryBucket(bucket_start_s=4 * 3600)
        bucket.running_endpoints_last = 2
        bucket.running_endpoints_sum = 2
        bucket.connected_sessions_last = 1
        bucket.connected_sessions_sum = 1
        bucket.sample_count = 1

        clock = FakeClock(4 * 3600 + 60)
        store = FakeHistoryStore(initial_buckets=[bucket])
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=2,
                pending=0,
                running=2,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            sample_interval_s=3600,
            retention_minutes=24 * 60,
            history_store=store,
            time_fn=clock.now,
        )

        await dashboard.start()
        try:
            series = await dashboard.series(window_minutes=2, resolution="minute")
        finally:
            await dashboard.stop()

        self.assertEqual(store.load_calls, 1)
        self.assertEqual(series[0]["running_endpoints"], 2)

    async def test_live_day_rollover_writes_complete_past_day_from_memory(self):
        day_start = _day_start(2026, 5, 18)
        clock = FakeClock(day_start + 24 * 60 * 60 + 120)
        store = DayRolloverHistoryStore()
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            sample_interval_s=3600,
            retention_minutes=2 * 24 * 60,
            history_store=store,
            time_fn=clock.now,
        )
        dashboard._day_rollover_cursor_s = day_start
        for minute in range(24 * 60):
            bucket_start_s = day_start + minute * 60
            bucket = SwarmHistoryBucket(bucket_start_s=bucket_start_s)
            bucket.sample_count = 1
            dashboard._history[bucket_start_s] = bucket

        await dashboard._rollover_completed_days()

        self.assertEqual(len(store.day_write_calls), 1)
        self.assertEqual(store.day_write_calls[0][0], day_start)
        self.assertEqual(len(store.day_write_calls[0][1]), 24 * 60)
        self.assertEqual(dashboard._day_rollover_cursor_s, day_start + 24 * 60 * 60)

    async def test_live_day_rollover_writes_incomplete_past_day(self):
        day_start = _day_start(2026, 5, 18)
        clock = FakeClock(day_start + 24 * 60 * 60 + 120)
        store = DayRolloverHistoryStore()
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            sample_interval_s=3600,
            retention_minutes=2 * 24 * 60,
            history_store=store,
            time_fn=clock.now,
        )
        dashboard._day_rollover_cursor_s = day_start
        for minute in range((24 * 60) - 1):
            bucket_start_s = day_start + minute * 60
            dashboard._history[bucket_start_s] = SwarmHistoryBucket(bucket_start_s=bucket_start_s)

        await dashboard._rollover_completed_days()

        self.assertEqual(len(store.day_write_calls), 1)
        self.assertEqual(store.day_write_calls[0][0], day_start)
        self.assertEqual(len(store.day_write_calls[0][1]), (24 * 60) - 1)
        self.assertEqual(dashboard._day_rollover_cursor_s, day_start + 24 * 60 * 60)

    async def test_background_restore_does_not_block_start(self):
        bucket = SwarmHistoryBucket(bucket_start_s=4 * 3600)
        bucket.running_endpoints_last = 2
        bucket.running_endpoints_sum = 2
        bucket.sample_count = 1

        clock = FakeClock(4 * 3600 + 60)
        store = SlowHistoryStore(delay_s=0.15, initial_buckets=[bucket])
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            sample_interval_s=3600,
            retention_minutes=24 * 60,
            history_store=store,
            restore_history_in_background=True,
            time_fn=clock.now,
        )

        started = time.monotonic()
        await dashboard.start()
        elapsed = time.monotonic() - started
        try:
            self.assertLess(elapsed, 0.1)
            self.assertEqual(dashboard.history_restore_status()["status"], "running")
            await dashboard._restore_task
            self.assertEqual(dashboard.history_restore_status()["status"], "complete")
            series = await dashboard.series(window_minutes=2, resolution="minute")
            self.assertEqual(series[0]["running_endpoints"], 2)
        finally:
            await dashboard.stop()

    async def test_delayed_startup_merge_recovers_late_history_without_overwriting_live_bucket(self):
        late_bucket = SwarmHistoryBucket(bucket_start_s=4 * 3600)
        late_bucket.running_endpoints_last = 3
        late_bucket.running_endpoints_sum = 3
        late_bucket.sample_count = 1
        stale_live_bucket = SwarmHistoryBucket(bucket_start_s=4 * 3600 + 60)
        stale_live_bucket.running_endpoints_last = 99
        stale_live_bucket.running_endpoints_sum = 99
        stale_live_bucket.sample_count = 1

        clock = FakeClock(4 * 3600 + 60)
        store = FakeHistoryStore()
        dashboard = SwarmDashboard(
            snapshot_provider=FakeSnapshotProvider(_health_snapshot(
                connected=0,
                pending=0,
                running=1,
                waking=0,
                free_slots=1,
                effective_free_slots=1,
            )),
            sample_interval_s=3600,
            retention_minutes=24 * 60,
            history_store=store,
            restore_history_in_background=True,
            startup_merge_delay_s=0.01,
            time_fn=clock.now,
        )

        await dashboard.start()
        try:
            await dashboard._restore_task
            store.saved[late_bucket.bucket_start_s] = late_bucket.to_dict()
            store.saved[stale_live_bucket.bucket_start_s] = stale_live_bucket.to_dict()
            await dashboard._startup_merge_task
            series = await dashboard.series(window_minutes=2, resolution="minute")
            status = dashboard.startup_merge_status()
        finally:
            await dashboard.stop()

        self.assertEqual(store.load_calls, 2)
        self.assertEqual([point["running_endpoints"] for point in series], [3, 1])
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["bucket_count"], 2)
        self.assertEqual(status["updated_bucket_count"], 1)


class HuggingFaceBucketHistoryStoreTests(unittest.TestCase):
    def test_load_recent_prefers_day_files_over_minute_files(self):
        day_start = _day_start(2026, 5, 18)
        day_buckets = []
        for minute in range(24 * 60):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.running_endpoints_last = 3
            day_buckets.append(bucket)
        api = FakeBucketApi(
            {
                "reachy-s2s-lb/days/2026-05-18.json": {
                    "version": 1,
                    "day_start_s": day_start,
                    "day": "2026-05-18",
                    "complete": True,
                    "minute_bucket_count": 24 * 60,
                    "expected_minute_bucket_count": 24 * 60,
                    "buckets": [bucket.to_dict() for bucket in day_buckets],
                },
                "reachy-s2s-lb/minutes/999999.json": {
                    "version": 1,
                    "bucket": SwarmHistoryBucket(bucket_start_s=999999).to_dict(),
                },
            }
        )
        store = _bucket_store(api)

        loaded = store.load_recent(retention_minutes=1, now_epoch_s=day_start)

        self.assertEqual([bucket.bucket_start_s for bucket in loaded], [day_start])
        self.assertEqual(loaded[0].running_endpoints_last, 3)
        self.assertEqual(api.list_prefixes, ["reachy-s2s-lb/days"])

    def test_load_recent_merges_partial_day_file_with_minute_files(self):
        day_start = _day_start(2026, 5, 18)
        first_bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        first_bucket.running_endpoints_last = 2
        second_bucket = SwarmHistoryBucket(bucket_start_s=day_start + 60)
        second_bucket.running_endpoints_last = 4
        api = FakeBucketApi(
            {
                "reachy-s2s-lb/days/2026-05-18.json": {
                    "version": 1,
                    "day_start_s": day_start,
                    "day": "2026-05-18",
                    "complete": False,
                    "minute_bucket_count": 1,
                    "expected_minute_bucket_count": 24 * 60,
                    "buckets": [first_bucket.to_dict()],
                },
                f"reachy-s2s-lb/minutes/2026-05-18/{day_start + 60}.json": {
                    "version": 1,
                    "bucket": second_bucket.to_dict(),
                },
            }
        )
        store = _bucket_store(api)

        loaded = store.load_recent(retention_minutes=24 * 60 + 3, now_epoch_s=day_start + 24 * 60 * 60 + 2 * 60)

        self.assertEqual([bucket.bucket_start_s for bucket in loaded], [day_start, day_start + 60])
        self.assertEqual([bucket.running_endpoints_last for bucket in loaded], [2, 4])
        self.assertIn("reachy-s2s-lb/minutes/2026-05-18", api.list_prefixes)
        self.assertEqual([remote_path for remote_path, _ in api.downloads], [
            "reachy-s2s-lb/days/2026-05-18.json",
            f"reachy-s2s-lb/minutes/2026-05-18/{day_start + 60}.json",
        ])
        self.assertEqual(api.batch_adds[0][1], "reachy-s2s-lb/days/2026-05-18.json")
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["finalized"], True)
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["minute_bucket_count"], 2)

    def test_load_recent_merges_late_minutes_into_finalized_partial_day(self):
        day_start = _day_start(2026, 5, 18)
        first_bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        first_bucket.running_endpoints_last = 2
        second_bucket = SwarmHistoryBucket(bucket_start_s=day_start + 60)
        second_bucket.running_endpoints_last = 4
        api = FakeBucketApi(
            {
                "reachy-s2s-lb/days/2026-05-18.json": {
                    "version": 1,
                    "day_start_s": day_start,
                    "day": "2026-05-18",
                    "complete": False,
                    "finalized": True,
                    "minute_bucket_count": 1,
                    "expected_minute_bucket_count": 24 * 60,
                    "missing_minute_bucket_count": (24 * 60) - 1,
                    "incomplete_reason": "missing_minute_buckets",
                    "buckets": [first_bucket.to_dict()],
                },
                f"reachy-s2s-lb/minutes/2026-05-18/{day_start + 60}.json": {
                    "version": 1,
                    "bucket": second_bucket.to_dict(),
                },
            }
        )
        store = _bucket_store(api)

        loaded = store.load_recent(
            retention_minutes=24 * 60 + 3,
            now_epoch_s=day_start + 24 * 60 * 60 + 2 * 60,
        )

        self.assertEqual([bucket.bucket_start_s for bucket in loaded], [day_start, day_start + 60])
        self.assertEqual([bucket.running_endpoints_last for bucket in loaded], [2, 4])
        self.assertIn("reachy-s2s-lb/minutes/2026-05-18", api.list_prefixes)
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["minute_bucket_count"], 2)

    def test_load_recent_backfills_complete_day_file_from_minutes(self):
        day_start = _day_start(2026, 5, 18)
        files = {}
        for minute in range(24 * 60):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            files[f"reachy-s2s-lb/minutes/{bucket.bucket_start_s}.json"] = {
                "version": 1,
                "bucket": bucket.to_dict(),
            }
        api = FakeBucketApi(files)
        store = _bucket_store(api)

        loaded = store.load_recent(retention_minutes=24 * 60 + 2, now_epoch_s=day_start + 24 * 60 * 60 + 60)

        self.assertEqual(len(loaded), 24 * 60)
        self.assertEqual(len(api.batch_adds), 1)
        _, path = api.batch_adds[0]
        self.assertEqual(path, "reachy-s2s-lb/days/2026-05-18.json")

    def test_read_only_store_does_not_backfill_day_files(self):
        day_start = _day_start(2026, 5, 18)
        files = {}
        for minute in range(24 * 60):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            files[f"reachy-s2s-lb/minutes/{bucket.bucket_start_s}.json"] = {
                "version": 1,
                "bucket": bucket.to_dict(),
            }
        api = FakeBucketApi(files)
        store = _bucket_store(api)
        readonly = ReadOnlyDashboardHistoryStore(store)

        loaded = readonly.load_recent(retention_minutes=24 * 60 + 2, now_epoch_s=day_start + 24 * 60 * 60 + 60)

        self.assertEqual(len(loaded), 24 * 60)
        self.assertEqual(api.batch_adds, [])

    def test_write_buckets_uses_date_sharded_minute_path(self):
        day_start = _day_start(2026, 5, 18)
        bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        api = FakeBucketApi({})
        store = _bucket_store(api)

        store.write_buckets([bucket])

        self.assertEqual(api.batch_adds[0][1], f"reachy-s2s-lb/minutes/2026-05-18/{day_start}.json")

    def test_write_day_buckets_uses_day_path(self):
        day_start = _day_start(2026, 5, 18)
        api = FakeBucketApi({})
        store = _bucket_store(api)
        buckets = []
        for minute in range(24 * 60):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            buckets.append(bucket)

        path = store.write_day_buckets(day_start_s=day_start, buckets=buckets)

        self.assertEqual(path, "reachy-s2s-lb/days/2026-05-18.json")
        self.assertEqual(api.batch_adds[0][1], "reachy-s2s-lb/days/2026-05-18.json")
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["complete"], True)
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["finalized"], True)
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["minute_bucket_count"], 24 * 60)

    def test_write_day_buckets_finalizes_incomplete_day_path(self):
        day_start = _day_start(2026, 5, 18)
        api = FakeBucketApi({})
        store = _bucket_store(api)
        buckets = []
        for minute in range((24 * 60) - 1):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            buckets.append(bucket)

        path = store.write_day_buckets(day_start_s=day_start, buckets=buckets)

        payload = api.files["reachy-s2s-lb/days/2026-05-18.json"]
        self.assertEqual(path, "reachy-s2s-lb/days/2026-05-18.json")
        self.assertEqual(payload["complete"], False)
        self.assertEqual(payload["finalized"], True)
        self.assertEqual(payload["minute_bucket_count"], (24 * 60) - 1)
        self.assertEqual(payload["missing_minute_bucket_count"], 1)
        self.assertEqual(payload["incomplete_reason"], "missing_minute_buckets")

    def test_migrate_legacy_minute_files_moves_flat_files_to_day_folder(self):
        day_start = _day_start(2026, 5, 18)
        first_bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        second_bucket = SwarmHistoryBucket(bucket_start_s=day_start + 60)
        api = FakeBucketApi(
            {
                f"reachy-s2s-lb/minutes/{day_start}.json": {
                    "version": 1,
                    "bucket": first_bucket.to_dict(),
                },
                f"reachy-s2s-lb/minutes/{day_start + 60}.json": {
                    "version": 1,
                    "bucket": second_bucket.to_dict(),
                },
            }
        )
        store = _bucket_store(api)

        result = store.migrate_legacy_minute_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start + 60,
        )

        self.assertEqual(result["moved_minute_files"], 2)
        self.assertEqual(result["moved_days"], [{"day": "2026-05-18", "count": 2}])
        self.assertNotIn(f"reachy-s2s-lb/minutes/{day_start}.json", api.files)
        self.assertNotIn(f"reachy-s2s-lb/minutes/{day_start + 60}.json", api.files)
        self.assertIn(f"reachy-s2s-lb/minutes/2026-05-18/{day_start}.json", api.files)
        self.assertIn(f"reachy-s2s-lb/minutes/2026-05-18/{day_start + 60}.json", api.files)
        self.assertEqual(api.downloads, [])
        self.assertEqual(api.batch_adds, [])
        self.assertEqual(
            api.batch_deletes,
            [
                f"reachy-s2s-lb/minutes/{day_start}.json",
                f"reachy-s2s-lb/minutes/{day_start + 60}.json",
            ],
        )

    def test_migrate_legacy_minute_files_uploads_files_without_xet_hash(self):
        day_start = _day_start(2026, 5, 18)
        legacy_path = f"reachy-s2s-lb/minutes/{day_start}.json"
        target_path = f"reachy-s2s-lb/minutes/2026-05-18/{day_start}.json"
        bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        api = FakeBucketApi(
            {
                legacy_path: {
                    "version": 1,
                    "bucket": bucket.to_dict(),
                },
            },
            missing_xet_hash_paths={legacy_path},
        )
        store = _bucket_store(api)

        result = store.migrate_legacy_minute_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start,
        )

        self.assertEqual(result["moved_minute_files"], 1)
        self.assertEqual([remote_path for remote_path, _ in api.downloads], [legacy_path])
        self.assertEqual(api.batch_adds[0][1], target_path)
        self.assertEqual(api.batch_deletes, [legacy_path])
        self.assertNotIn(legacy_path, api.files)
        self.assertIn(target_path, api.files)

    def test_migrate_legacy_minute_files_deletes_existing_sharded_duplicates(self):
        day_start = _day_start(2026, 5, 18)
        bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        api = FakeBucketApi(
            {
                f"reachy-s2s-lb/minutes/{day_start}.json": {
                    "version": 1,
                    "bucket": bucket.to_dict(),
                },
                f"reachy-s2s-lb/minutes/2026-05-18/{day_start}.json": {
                    "version": 1,
                    "bucket": bucket.to_dict(),
                },
            }
        )
        store = _bucket_store(api)

        result = store.migrate_legacy_minute_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start,
        )

        self.assertEqual(result["moved_minute_files"], 0)
        self.assertEqual(result["deleted_legacy_duplicate_files"], 1)
        self.assertEqual(result["deleted_legacy_duplicate_days"], [{"day": "2026-05-18", "count": 1}])
        self.assertNotIn(f"reachy-s2s-lb/minutes/{day_start}.json", api.files)
        self.assertIn(f"reachy-s2s-lb/minutes/2026-05-18/{day_start}.json", api.files)
        self.assertEqual(api.batch_deletes, [f"reachy-s2s-lb/minutes/{day_start}.json"])

    def test_backfill_day_files_uses_same_day_cache_behavior(self):
        day_start = _day_start(2026, 5, 18)
        files = {}
        for minute in range(24 * 60):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            files[f"reachy-s2s-lb/minutes/{bucket.bucket_start_s}.json"] = {
                "version": 1,
                "bucket": bucket.to_dict(),
            }
        api = FakeBucketApi(files)
        store = _bucket_store(api)

        result = store.backfill_day_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start,
            now_epoch_s=day_start + 24 * 60 * 60,
        )

        self.assertEqual(result["created_days"], ["2026-05-18"])
        self.assertEqual(result["created_paths"], ["reachy-s2s-lb/days/2026-05-18.json"])
        self.assertEqual(result["minute_buckets_loaded"], 24 * 60)
        self.assertEqual(result["incomplete_days"], [])
        self.assertEqual(len(api.batch_adds), 1)

    def test_backfill_day_files_reports_incomplete_days(self):
        day_start = _day_start(2026, 5, 18)
        files = {}
        for minute in range(12):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            files[f"reachy-s2s-lb/minutes/{bucket.bucket_start_s}.json"] = {
                "version": 1,
                "bucket": bucket.to_dict(),
            }
        api = FakeBucketApi(files)
        store = _bucket_store(api)

        result = store.backfill_day_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start,
            now_epoch_s=day_start + 24 * 60 * 60,
        )

        self.assertEqual(result["created_days"], [])
        self.assertEqual(result["incomplete_days"], [{"day": "2026-05-18", "minute_buckets_found": 12}])
        self.assertEqual(api.batch_adds, [])

    def test_backfill_day_files_can_create_partial_past_day_file(self):
        day_start = _day_start(2026, 5, 18)
        files = {}
        for minute in range(12):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            files[f"reachy-s2s-lb/minutes/{bucket.bucket_start_s}.json"] = {
                "version": 1,
                "bucket": bucket.to_dict(),
            }
        api = FakeBucketApi(files)
        store = _bucket_store(api)

        result = store.backfill_day_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start,
            now_epoch_s=day_start + 24 * 60 * 60,
            allow_partial_days=True,
        )

        self.assertEqual(result["created_days"], [])
        self.assertEqual(result["created_partial_days"], ["2026-05-18"])
        self.assertEqual(result["created_paths"], ["reachy-s2s-lb/days/2026-05-18.json"])
        self.assertEqual(result["incomplete_days"], [])
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["complete"], False)
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["finalized"], True)
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["missing_minute_bucket_count"], 1428)
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["minute_bucket_count"], 12)

    def test_backfill_day_files_replaces_partial_day_file_when_minutes_are_complete(self):
        day_start = _day_start(2026, 5, 18)
        partial_bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        files = {
            "reachy-s2s-lb/days/2026-05-18.json": {
                "version": 1,
                "day_start_s": day_start,
                "day": "2026-05-18",
                "complete": False,
                "minute_bucket_count": 1,
                "expected_minute_bucket_count": 24 * 60,
                "buckets": [partial_bucket.to_dict()],
            },
        }
        for minute in range(24 * 60):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            files[f"reachy-s2s-lb/minutes/2026-05-18/{bucket.bucket_start_s}.json"] = {
                "version": 1,
                "bucket": bucket.to_dict(),
            }
        api = FakeBucketApi(files)
        store = _bucket_store(api)

        result = store.backfill_day_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start,
            now_epoch_s=day_start + 24 * 60 * 60,
        )

        self.assertEqual(result["existing_days"], [])
        self.assertEqual(result["existing_partial_days"], ["2026-05-18"])
        self.assertEqual(result["created_days"], ["2026-05-18"])
        self.assertEqual(result["created_partial_days"], [])
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["complete"], True)
        self.assertEqual(api.files["reachy-s2s-lb/days/2026-05-18.json"]["minute_bucket_count"], 24 * 60)

    def test_backfill_day_files_finalizes_open_partial_without_redownloading_existing_minutes(self):
        day_start = _day_start(2026, 5, 18)
        partial_bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        api = FakeBucketApi(
            {
                "reachy-s2s-lb/days/2026-05-18.json": {
                    "version": 1,
                    "day_start_s": day_start,
                    "day": "2026-05-18",
                    "complete": False,
                    "minute_bucket_count": 1,
                    "expected_minute_bucket_count": 24 * 60,
                    "buckets": [partial_bucket.to_dict()],
                },
            }
        )
        store = _bucket_store(api)

        result = store.backfill_day_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start,
            now_epoch_s=day_start + 24 * 60 * 60,
            allow_partial_days=True,
        )

        payload = api.files["reachy-s2s-lb/days/2026-05-18.json"]
        self.assertEqual(result["existing_open_partial_days"], ["2026-05-18"])
        self.assertEqual(result["created_partial_days"], ["2026-05-18"])
        self.assertEqual(result["minute_buckets_loaded"], 0)
        self.assertEqual([remote_path for remote_path, _ in api.downloads], ["reachy-s2s-lb/days/2026-05-18.json"])
        self.assertEqual(payload["complete"], False)
        self.assertEqual(payload["finalized"], True)
        self.assertEqual(payload["missing_minute_bucket_count"], (24 * 60) - 1)

    def test_download_minute_bucket_candidates_reuses_local_cache(self):
        day_start = _day_start(2026, 5, 18)
        first_bucket = SwarmHistoryBucket(bucket_start_s=day_start)
        second_bucket = SwarmHistoryBucket(bucket_start_s=day_start + 60)
        first_path = f"reachy-s2s-lb/minutes/2026-05-18/{day_start}.json"
        second_path = f"reachy-s2s-lb/minutes/2026-05-18/{day_start + 60}.json"
        api = FakeBucketApi(
            {
                first_path: {
                    "version": 1,
                    "bucket": first_bucket.to_dict(),
                },
                second_path: {
                    "version": 1,
                    "bucket": second_bucket.to_dict(),
                },
            }
        )
        store = _bucket_store(api)

        with tempfile.TemporaryDirectory() as tmpdir:
            store.local_download_cache_dir = Path(tmpdir)
            cached_path = Path(tmpdir) / store.bucket_id / first_path
            cached_path.parent.mkdir(parents=True)
            cached_path.write_text(json.dumps(api.files[first_path]))

            loaded = store._download_minute_bucket_candidates(
                [
                    (day_start, first_path),
                    (day_start + 60, second_path),
                ]
            )

        self.assertEqual([bucket.bucket_start_s for bucket in loaded], [day_start, day_start + 60])
        self.assertEqual([remote_path for remote_path, _ in api.downloads], [second_path])

    def test_backfill_day_files_writes_each_day_incrementally(self):
        first_day = _day_start(2026, 5, 17)
        second_day = _day_start(2026, 5, 18)
        files = {}
        for day_start in [first_day, second_day]:
            for minute in range(24 * 60):
                bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
                bucket.sample_count = 1
                files[f"reachy-s2s-lb/minutes/{bucket.bucket_start_s}.json"] = {
                    "version": 1,
                    "bucket": bucket.to_dict(),
                }
        api = FakeBucketApi(files)
        store = _bucket_store(api)

        result = store.backfill_day_files(
            start_epoch_s=first_day,
            end_epoch_s=second_day,
            now_epoch_s=second_day + 24 * 60 * 60,
        )

        self.assertEqual(result["created_days"], ["2026-05-17", "2026-05-18"])
        self.assertEqual(len(api.batch_calls), 2)
        self.assertEqual(api.batch_calls[0][0][1], "reachy-s2s-lb/days/2026-05-17.json")
        self.assertEqual(api.batch_calls[1][0][1], "reachy-s2s-lb/days/2026-05-18.json")

    def test_backfill_day_files_reports_would_create_when_read_only(self):
        day_start = _day_start(2026, 5, 18)
        files = {}
        for minute in range(24 * 60):
            bucket = SwarmHistoryBucket(bucket_start_s=day_start + minute * 60)
            bucket.sample_count = 1
            files[f"reachy-s2s-lb/minutes/{bucket.bucket_start_s}.json"] = {
                "version": 1,
                "bucket": bucket.to_dict(),
            }
        api = FakeBucketApi(files)
        store = _bucket_store(api)
        store.read_only = True

        result = store.backfill_day_files(
            start_epoch_s=day_start,
            end_epoch_s=day_start,
            now_epoch_s=day_start + 24 * 60 * 60,
        )

        self.assertEqual(result["created_days"], [])
        self.assertEqual(result["would_create_days"], ["2026-05-18"])
        self.assertEqual(result["incomplete_days"], [])
        self.assertEqual(api.batch_adds, [])


if __name__ == "__main__":
    unittest.main()

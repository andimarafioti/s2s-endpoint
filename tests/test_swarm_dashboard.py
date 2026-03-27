import unittest

from app.swarm_dashboard import SwarmDashboard, SwarmHistoryBucket, SwarmStateSample


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
        self.assertEqual(payload["summary"]["session_requests_last_hour"], 1)
        self.assertEqual(payload["summary"]["conversations_completed_last_hour"], 1)
        self.assertEqual(payload["summary"]["avg_conversation_duration_last_hour_s"], 150.0)

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
        self.assertEqual(persisted.completed_conversations, 1)
        self.assertEqual(persisted.completed_conversation_duration_total_s, 90.0)

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


if __name__ == "__main__":
    unittest.main()

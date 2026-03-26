import unittest

from app.swarm_dashboard import SwarmDashboard, SwarmStateSample


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

        payload = await dashboard.data(window="60m", resolution="minute")
        point = payload["series"][-1]
        current = payload["current"]

        self.assertEqual(current["running_endpoints"], 2)
        self.assertEqual(current["transitioning_endpoints"], 1)
        self.assertEqual(current["parked_endpoints"], 1)
        self.assertEqual(point["session_requests"], 1)
        self.assertEqual(point["session_allocation_successes"], 1)
        self.assertEqual(point["session_connected_events"], 1)
        self.assertEqual(payload["summary"]["session_requests_last_hour"], 1)

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
        await dashboard.record_session_event("disconnected")
        clock.set(3 * 3600 + 59 * 60)

        points = await dashboard.series(window_minutes=60, resolution="hour")
        point = points[-1]

        self.assertEqual(point["running_endpoints"], 3.0)
        self.assertEqual(point["connected_sessions"], 2.0)
        self.assertEqual(point["session_requests"], 2)
        self.assertEqual(point["session_allocation_successes"], 1)
        self.assertEqual(point["session_allocation_failures"], 1)
        self.assertEqual(point["session_disconnected_events"], 1)

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


if __name__ == "__main__":
    unittest.main()

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from app.dashboard_history_store import ReadOnlyDashboardHistoryStore
from app.dashboard_preview import DashboardPreviewSessionManager
from app.swarm_dashboard import SwarmHistoryBucket


class FakeClock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeHistoryStore:
    def __init__(self, buckets=None):
        self.buckets = list(buckets or [])
        self.write_calls = []

    def load_recent(self, *, retention_minutes: int, now_epoch_s: float):
        return list(self.buckets)

    def write_buckets(self, buckets):
        self.write_calls.append(list(buckets))


class ReadOnlyDashboardHistoryStoreTests(unittest.TestCase):
    def test_loads_from_wrapped_store_without_writing(self):
        bucket = SwarmHistoryBucket(bucket_start_s=120)
        wrapped = FakeHistoryStore(buckets=[bucket])
        store = ReadOnlyDashboardHistoryStore(wrapped)

        self.assertEqual(store.load_recent(retention_minutes=60, now_epoch_s=180), [bucket])
        store.write_buckets([SwarmHistoryBucket(bucket_start_s=180)])

        self.assertEqual(wrapped.write_calls, [])


class DashboardPreviewSessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_healthcheck_returns_synthetic_dashboard_snapshot(self):
        clock = FakeClock(1000.0)
        manager = DashboardPreviewSessionManager(endpoint_slots=2, time_fn=clock)
        await manager.start()

        healthy, detail, snapshot = await manager.healthcheck()

        self.assertTrue(healthy)
        self.assertIn("preview mode", detail)
        self.assertTrue(snapshot["preview_mode"])
        self.assertEqual(len(snapshot["router"]["endpoints"]), 4)
        self.assertGreaterEqual(snapshot["router"]["running_endpoints"], 2)
        self.assertGreaterEqual(snapshot["router"]["effective_free_slots"], snapshot["router"]["free_slots"])

    async def test_allocation_is_disabled(self):
        manager = DashboardPreviewSessionManager(endpoint_slots=1)

        with self.assertRaisesRegex(RuntimeError, "preview mode"):
            await manager.allocate("https://lb.example")


class LoadBalancerPreviewModeTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("app.load_balancer_main", None)

    def test_compute_endpoint_names_test_enables_preview_without_session_secret(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
            }
        )

        self.assertTrue(module.DASHBOARD_PREVIEW_MODE)
        self.assertIsInstance(module.session_manager, DashboardPreviewSessionManager)
        self.assertEqual(module.COMPUTE_ENDPOINT_NAMES[0], "preview-compute-01")

    @patch("app.dashboard_history_store.HuggingFaceBucketHistoryStore.__init__", return_value=None)
    def test_preview_mode_uses_dashboard_bucket_persistence_read_only(self, init_store):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "DASHBOARD_BUCKET_ID": "HuggingFaceM4/some-dashboard-bucket",
                "SESSION_SHARED_SECRET": "",
            }
        )

        self.assertTrue(module.DASHBOARD_PREVIEW_MODE)
        self.assertIsInstance(module.dashboard_history_store, ReadOnlyDashboardHistoryStore)
        self.assertIs(module.dashboard.history_store, module.dashboard_history_store)
        init_store.assert_called_once()

    def _import_load_balancer(self, env):
        sys.modules.pop("app.load_balancer_main", None)
        with patch.dict(
            "os.environ",
            {
                "DASHBOARD_BUCKET_ID": "",
                "DASHBOARD_PREVIEW_MODE": "",
                **env,
            },
            clear=False,
        ):
            return importlib.import_module("app.load_balancer_main")


class LoadBalancerSessionHandlerTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        sys.modules.pop("app.load_balancer_main", None)

    async def test_disconnected_session_allocation_releases_pending_session(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dashboard = fake_dashboard
        module.session_manager = fake_session_manager

        request = FakeDisconnectedRequest()

        with (
            patch.object(module.time, "monotonic", side_effect=[20.0, 21.5]),
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await module.create_session(request)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(fake_session_manager.cancelled_session_ids, ["session-123"])
        self.assertEqual(fake_dashboard.calls, ["request"])
        record = logs.records[0]
        self.assertEqual(record.outcome, "client_disconnected")
        self.assertEqual(record.session_id, "session-123")
        self.assertEqual(record.endpoint_name, "endpoint-a")
        self.assertEqual(record.slot_id, "endpoint-a")
        self.assertEqual(record.allocation_wait_ms, 1500)
        self.assertTrue(record.waited_for_capacity)
        self.assertIn("outcome=client_disconnected", record.getMessage())
        self.assertIn("endpoint_name=endpoint-a", record.getMessage())
        self.assertIn("allocation_wait_ms=1500", record.getMessage())

    async def test_successful_session_allocation_logs_outcome(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeSessionManager()

        with (
            patch.object(module.time, "monotonic", side_effect=[20.0, 20.05]),
            self.assertLogs("s2s-endpoint", level="INFO") as logs,
        ):
            response = await module.create_session(FakeConnectedRequest())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_dashboard.calls, ["request", "success"])
        record = logs.records[0]
        self.assertEqual(record.outcome, "success")
        self.assertEqual(record.session_id, "session-123")
        self.assertEqual(record.endpoint_name, "endpoint-a")
        self.assertEqual(record.slot_id, "endpoint-a")
        self.assertEqual(record.allocation_wait_ms, 50)
        self.assertTrue(record.waited_for_capacity)

    async def test_failed_session_allocation_logs_outcome(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeFailingSessionManager()

        with (
            patch.object(module.time, "monotonic", side_effect=[20.0, 20.25]),
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await module.create_session(FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(fake_dashboard.calls, ["request", "failure"])
        record = logs.records[0]
        self.assertEqual(record.outcome, "allocation_failed")
        self.assertIsNone(record.session_id)
        self.assertIsNone(record.endpoint_name)
        self.assertIsNone(record.slot_id)
        self.assertEqual(record.allocation_wait_ms, 250)
        self.assertEqual(record.allocation_error, "no capacity")
        self.assertIn("outcome=allocation_failed", record.getMessage())
        self.assertIn("error=no capacity", record.getMessage())

    def _import_load_balancer(self):
        sys.modules.pop("app.load_balancer_main", None)
        with patch.dict(
            "os.environ",
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "DASHBOARD_BUCKET_ID": "",
                "DASHBOARD_PREVIEW_MODE": "",
                "SESSION_SHARED_SECRET": "",
            },
            clear=False,
        ):
            return importlib.import_module("app.load_balancer_main")


class FakeDashboard:
    def __init__(self):
        self.calls = []

    async def record_session_request(self):
        self.calls.append("request")

    async def record_session_allocation_failure(self):
        self.calls.append("failure")

    async def record_session_allocation_success(self):
        self.calls.append("success")


class FakeSessionManager:
    def __init__(self):
        self.cancelled_session_ids = []

    async def allocate(self, lb_base_url):
        return {
            "session_id": "session-123",
            "connect_url": f"{lb_base_url}ws?session=session-123",
            "endpoint_name": "endpoint-a",
            "slot_id": "endpoint-a",
            "waited_for_capacity": True,
        }

    async def cancel_pending_session(self, session_id):
        self.cancelled_session_ids.append(session_id)


class FakeFailingSessionManager:
    async def allocate(self, lb_base_url):
        raise RuntimeError("no capacity")


class FakeDisconnectedRequest:
    headers = {
        "x-forwarded-proto": "https",
        "x-forwarded-host": "lb.example",
    }
    url = SimpleNamespace(scheme="http", netloc="internal.local")

    async def is_disconnected(self):
        return True


class FakeConnectedRequest(FakeDisconnectedRequest):
    async def is_disconnected(self):
        return False


if __name__ == "__main__":
    unittest.main()

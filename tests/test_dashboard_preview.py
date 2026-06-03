import importlib
import sys
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()

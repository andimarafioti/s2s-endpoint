import json
import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.dashboard_history_store import ReadOnlyDashboardHistoryStore
from app.dashboard_preview import DashboardPreviewSessionManager
from app.endpoint_pool_router import EndpointCapacityTimeoutError, EndpointTransitionConflictError
from app.swarm_dashboard import SwarmHistoryBucket
from tests.helpers import monotonic_sequence


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

    def test_drain_route_requires_admin_authorization(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
                "HF_CONTROL_TOKEN": "",
                "HF_TOKEN": "",
                "LB_ADMIN_AUTH_TOKEN": "admin-secret",
            }
        )
        client = TestClient(module.app)

        missing_auth = client.post(
            "/internal/endpoints/preview-compute-01/drain",
            json={"draining": True},
        )
        wrong_auth = client.post(
            "/internal/endpoints/preview-compute-01/drain",
            headers={"Authorization": "Bearer wrong-secret"},
            json={"draining": True},
        )
        correct_auth = client.post(
            "/internal/endpoints/preview-compute-01/drain",
            headers={"Authorization": "Bearer admin-secret"},
            json={"draining": True},
        )

        self.assertEqual(missing_auth.status_code, 401)
        self.assertEqual(missing_auth.headers["www-authenticate"], "Bearer")
        self.assertEqual(wrong_auth.status_code, 403)
        self.assertEqual(correct_auth.status_code, 503)
        self.assertEqual(correct_auth.json()["detail"], "Endpoint draining is not available")

        status = client.get(
            "/internal/endpoints/preview-compute-01",
            headers={"Authorization": "Bearer admin-secret"},
        )
        self.assertEqual(status.status_code, 503)
        self.assertEqual(status.json()["detail"], "Endpoint status is not available")

    def test_drain_route_validates_endpoint_before_mutating(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
                "HF_CONTROL_TOKEN": "",
                "HF_TOKEN": "",
                "LB_ADMIN_AUTH_TOKEN": "admin-secret",
            }
        )

        class RecordingRouter:
            def __init__(self):
                self.calls = []

            async def set_draining(self, endpoint_name, draining):
                self.calls.append((endpoint_name, draining))

        class MissingEndpointSessionManager:
            def __init__(self):
                self.endpoint_router = RecordingRouter()

            async def healthcheck(self):
                return True, None, {"router": {"endpoints": []}}

        session_manager = MissingEndpointSessionManager()
        module.session_manager = session_manager
        client = TestClient(module.app)

        response = client.post(
            "/internal/endpoints/unknown/drain",
            headers={"Authorization": "Bearer admin-secret"},
            json={"draining": True},
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Unknown endpoint")
        self.assertEqual(session_manager.endpoint_router.calls, [])

    def test_drain_route_reports_transition_conflict(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
                "HF_CONTROL_TOKEN": "",
                "HF_TOKEN": "",
                "LB_ADMIN_AUTH_TOKEN": "admin-secret",
            }
        )

        class ConflictingRouter:
            async def set_draining(self, endpoint_name, draining):
                raise EndpointTransitionConflictError(
                    f"Endpoint {endpoint_name} has an active control-plane transition: parking"
                )

        class ConflictingSessionManager:
            endpoint_router = ConflictingRouter()

            async def healthcheck(self):
                return True, None, {
                    "router": {
                        "endpoints": [
                            {
                                "name": "reachy-s2s-01",
                                "status": "running",
                                "draining": False,
                            }
                        ]
                    }
                }

        module.session_manager = ConflictingSessionManager()
        client = TestClient(module.app)

        response = client.post(
            "/internal/endpoints/reachy-s2s-01/drain",
            headers={"Authorization": "Bearer admin-secret"},
            json={"draining": True},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("parking", response.json()["detail"])

    def test_endpoint_status_returns_snapshot_when_health_is_unready(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
                "HF_CONTROL_TOKEN": "",
                "HF_TOKEN": "",
                "LB_ADMIN_AUTH_TOKEN": "admin-secret",
            }
        )

        class UnhealthySessionManager:
            endpoint_router = object()

            async def healthcheck(self):
                return (
                    False,
                    "no running endpoint has synced usage",
                    {
                        "router": {
                            "endpoints": [
                                {
                                    "name": "reachy-s2s-01",
                                    "active_sessions": 0,
                                    "draining": True,
                                    "require_usage_sync": True,
                                    "usage_synced": False,
                                }
                            ]
                        }
                    },
                )

        module.session_manager = UnhealthySessionManager()
        client = TestClient(module.app)

        health = client.get("/health")
        endpoint_status = client.get(
            "/internal/endpoints/reachy-s2s-01",
            headers={"Authorization": "Bearer admin-secret"},
        )

        self.assertEqual(health.status_code, 503)
        self.assertEqual(endpoint_status.status_code, 200)
        endpoint = endpoint_status.json()["endpoint"]
        self.assertEqual(endpoint["name"], "reachy-s2s-01")
        self.assertFalse(endpoint["usage_synced"])

    def test_admin_routes_do_not_fall_back_to_hf_control_token(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
                "HF_CONTROL_TOKEN": "hf-control-token",
                "HF_TOKEN": "",
                "LB_ADMIN_AUTH_TOKEN": "",
            }
        )
        client = TestClient(module.app)

        response = client.post(
            "/internal/endpoints/preview-compute-01/drain",
            headers={"Authorization": "Bearer hf-control-token"},
            json={"draining": True},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "LB admin auth token is not configured")

    def _import_load_balancer(self, env):
        sys.modules.pop("app.load_balancer_main", None)
        with patch.dict(
            "os.environ",
            {
                "DASHBOARD_BUCKET_ID": "",
                "DASHBOARD_PREVIEW_MODE": "",
                "LB_ADMIN_AUTH_TOKEN": "",
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
            patch.object(module, "monotonic", new=monotonic_sequence(20.0, 21.5)),
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
        self.assertEqual(record.allocation_wait_ms, 1200)
        self.assertEqual(record.allocation_total_ms, 1500)
        self.assertTrue(record.waited_for_capacity)
        self.assertIn("outcome=client_disconnected", record.getMessage())
        self.assertIn("endpoint_name=endpoint-a", record.getMessage())
        self.assertIn("allocation_wait_ms=1200", record.getMessage())
        self.assertIn("allocation_total_ms=1500", record.getMessage())

    async def test_successful_session_allocation_logs_outcome(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with (
            patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.05)),
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
        self.assertEqual(record.allocation_wait_ms, 40)
        self.assertEqual(record.allocation_total_ms, 50)
        self.assertTrue(record.waited_for_capacity)
        payload = json.loads(response.body)
        self.assertEqual(
            payload,
            {
                "session_id": "session-123",
                "websocket_url": "wss://endpoint-a.example/ws",
                "connect_url": "https://lb.example/ws?session=session-123",
                "session_token": "session-token",
                "pending_timeout_s": 60,
            },
        )

    async def test_failed_session_allocation_logs_outcome(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeFailingSessionManager()

        with (
            patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.25)),
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
        self.assertIsNone(record.allocation_wait_ms)
        self.assertEqual(record.allocation_total_ms, 250)
        self.assertFalse(record.waited_for_capacity)
        self.assertEqual(record.allocation_error, "no capacity")
        self.assertIn("outcome=allocation_failed", record.getMessage())
        self.assertIn("error=no capacity", record.getMessage())

    async def test_capacity_timeout_session_allocation_logs_waited_for_capacity(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeFailingSessionManager(
            EndpointCapacityTimeoutError("timed out waiting for an available compute endpoint")
        )

        with (
            patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.25)),
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await module.create_session(FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(fake_dashboard.calls, ["request", "failure"])
        record = logs.records[0]
        self.assertEqual(record.outcome, "allocation_failed")
        self.assertEqual(record.allocation_wait_ms, 250)
        self.assertEqual(record.allocation_total_ms, 250)
        self.assertTrue(record.waited_for_capacity)

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
    def __init__(self, *, allocation_wait_ms: int = 1200):
        self.cancelled_session_ids = []
        self.allocation_wait_ms = allocation_wait_ms

    async def allocate(self, lb_base_url):
        return {
            "session_id": "session-123",
            "websocket_url": "wss://endpoint-a.example/ws",
            "connect_url": f"{lb_base_url}ws?session=session-123",
            "session_token": "session-token",
            "pending_timeout_s": 60,
            "endpoint_name": "endpoint-a",
            "slot_id": "endpoint-a",
            "allocation_wait_ms": self.allocation_wait_ms,
            "waited_for_capacity": True,
        }

    async def cancel_pending_session(self, session_id):
        self.cancelled_session_ids.append(session_id)


class FakeFailingSessionManager:
    def __init__(self, exc=None):
        self.exc = exc or RuntimeError("no capacity")

    async def allocate(self, lb_base_url):
        raise self.exc


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

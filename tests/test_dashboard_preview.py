import json
import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.dashboard_history import SwarmHistoryBucket
from app.dashboard_history_store import ReadOnlyDashboardHistoryStore
from app.dashboard_preview import DashboardPreviewSessionManager
from app.endpoint_pool_router import EndpointCapacityTimeoutError, EndpointTransitionConflictError
from app.requester_rate_limiter import RequesterRateLimitConfig, RequesterRateLimiter
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

    def test_health_exposes_dashboard_persistence_status(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
            }
        )

        response = TestClient(module.app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["dashboard_history"]["enabled"])
        self.assertTrue(response.json()["requester_tracking"]["rate_limit"]["enabled"])

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

            async def set_draining(
                self,
                endpoint_name,
                draining,
                *,
                lease_ttl_s=None,
                lease_id=None,
                force=False,
            ):
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
            async def set_draining(
                self,
                endpoint_name,
                draining,
                *,
                lease_ttl_s=None,
                lease_id=None,
                force=False,
            ):
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
            json={"draining": True, "lease_id": "rollout-a"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("parking", response.json()["detail"])

    def test_drain_route_returns_fresh_post_mutation_snapshot(self):
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
                self.draining = False
                self.lease_ttl_s = None
                self.calls = []

            async def set_draining(
                self,
                endpoint_name,
                draining,
                *,
                lease_ttl_s=None,
                lease_id=None,
                force=False,
            ):
                self.draining = draining
                self.lease_ttl_s = lease_ttl_s
                self.calls.append((draining, lease_id, force))

        class RecordingSessionManager:
            def __init__(self):
                self.endpoint_router = RecordingRouter()

            async def healthcheck(self):
                return True, None, {
                    "router": {
                        "endpoints": [
                            {
                                "name": "reachy-s2s-01",
                                "status": "running",
                                "draining": self.endpoint_router.draining,
                                "drain_lease_remaining_s": self.endpoint_router.lease_ttl_s,
                            }
                        ]
                    }
                }

        session_manager = RecordingSessionManager()
        module.session_manager = session_manager
        client = TestClient(module.app)

        response = client.post(
            "/internal/endpoints/reachy-s2s-01/drain",
            headers={"Authorization": "Bearer admin-secret"},
            json={
                "draining": True,
                "lease_ttl_s": 900,
                "lease_id": "rollout-a",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["endpoint"]["draining"])
        self.assertEqual(response.json()["endpoint"]["drain_lease_remaining_s"], 900.0)

        force_clear = client.post(
            "/internal/endpoints/reachy-s2s-01/drain",
            headers={"Authorization": "Bearer admin-secret"},
            json={"draining": False, "force": True},
        )
        self.assertEqual(force_clear.status_code, 200)
        self.assertFalse(force_clear.json()["endpoint"]["draining"])
        self.assertEqual(
            session_manager.endpoint_router.calls,
            [(True, "rollout-a", False), (False, None, True)],
        )

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
        self.assertEqual(fake_dashboard.calls, ["request", "abandoned"])
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
        self.assertEqual(record.requester_id, "anonymous:unknown")
        self.assertEqual(record.requester_kind, "anonymous")
        self.assertIn("requester_id=anonymous:unknown", record.getMessage())
        payload = json.loads(response.body)
        self.assertEqual(
            payload,
            {
                "state": "granted",
                "session_id": "session-123",
                "websocket_url": "wss://endpoint-a.example/ws",
                "connect_url": "https://lb.example/ws?session=session-123",
                "session_token": "session-token",
                "pending_timeout_s": 60,
            },
        )

    async def test_rate_limit_rejects_before_compute_allocation(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager(allocation_wait_ms=40)
        module.dashboard = fake_dashboard
        module.session_manager = fake_session_manager
        module.requester_rate_limiter = RequesterRateLimiter(
            config=RequesterRateLimitConfig(max_parallel_allocations=1)
        )

        with patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.05)):
            await module.create_session(FakeConnectedRequest())

        with (
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await module.create_session(FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.headers["Retry-After"], "60")
        self.assertEqual(raised.exception.detail["code"], "requester_rate_limited")
        self.assertEqual(raised.exception.detail["reason"], "parallel_allocations")
        self.assertEqual(fake_session_manager.allocation_calls, 1)
        self.assertEqual(
            fake_dashboard.calls,
            ["request", "success", "request", "rate_limited"],
        )
        record = logs.records[0]
        self.assertEqual(record.outcome, "rate_limited")
        self.assertEqual(record.rate_limit_reason, "parallel_allocations")
        self.assertEqual(record.requester_client_kind, "missing-user-agent")

    async def test_leaving_queue_releases_parallel_allocation_permit(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeQueuedSessionManager()

        response = await module.create_session(FakeConnectedRequest())

        self.assertEqual(json.loads(response.body)["state"], "queued")
        self.assertEqual(module.requester_rate_limiter.status()["active_allocations"], 1)

        await module.queue_leave("queue-123")

        status = module.requester_rate_limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["allocation_abandonments"], 1)
        self.assertEqual(fake_dashboard.calls, ["request", "abandoned"])

    async def test_expired_queue_ticket_releases_parallel_allocation_permit(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeQueuedSessionManager()

        await module.create_session(FakeConnectedRequest())
        await module.record_expired_queue_ticket("queue-123")

        status = module.requester_rate_limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["allocation_abandonments"], 1)
        self.assertEqual(fake_dashboard.calls, ["request", "abandoned"])

    async def test_session_allocation_tracks_reported_hardware_id_as_fingerprint(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeSessionManager(allocation_wait_ms=40)
        raw_hardware_id = "ABCDEF0123456789"

        with patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.05)):
            response = await module.create_session(FakeJsonRequest({"hardware_id": raw_hardware_id}))

        self.assertEqual(response.status_code, 200)
        requester = fake_dashboard.requesters[0]
        self.assertTrue(requester.reported_robot_id.startswith("robot:"))
        self.assertNotIn(raw_hardware_id.lower(), requester.reported_robot_id)

    async def test_session_allocation_ignores_invalid_reported_hardware_id(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.05)):
            response = await module.create_session(FakeJsonRequest({"hardware_id": "invalid"}))

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(fake_dashboard.requesters[0].reported_robot_id)

    async def test_connected_callback_is_attributed_to_requester_once(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.05)):
            await module.create_session(FakeConnectedRequest())

        self.assertEqual(module.session_requester_tracker.count(), 1)
        payload = {"session_token": "session-token", "event": "connected"}
        first = await module.session_event("session-123", payload)
        second = await module.session_event("session-123", payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(module.session_requester_tracker.count(), 0)
        self.assertEqual(fake_dashboard.session_events, ["connected", "connected"])
        self.assertEqual(fake_dashboard.connected_requesters, [fake_dashboard.requesters[0]])

    async def test_disconnected_callback_records_requester_duration_after_connect(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.05)):
            await module.create_session(FakeConnectedRequest())

        await module.session_event(
            "session-123",
            {"session_token": "session-token", "event": "connected"},
        )
        await module.session_event(
            "session-123",
            {"session_token": "session-token", "event": "disconnected"},
        )

        self.assertEqual(
            fake_dashboard.disconnected_requesters,
            [(fake_dashboard.requesters[0], 6.0, True)],
        )

    async def test_pre_connect_compute_rejection_does_not_penalize_requester(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dashboard = fake_dashboard
        module.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with patch.object(module, "monotonic", new=monotonic_sequence(20.0, 20.05)):
            await module.create_session(FakeConnectedRequest())

        await module.session_event(
            "session-123",
            {"session_token": "session-token", "event": "disconnected"},
        )

        totals = module.requester_rate_limiter.status()["totals"]
        self.assertNotIn("no_connects", totals)
        self.assertEqual(module.requester_rate_limiter.status()["active_allocations"], 0)

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
        self.requesters = []
        self.session_events = []
        self.connected_requesters = []
        self.disconnected_requesters = []

    async def record_session_request(self, requester=None):
        self.calls.append("request")
        self.requesters.append(requester)

    async def record_session_allocation_failure(self, requester=None):
        self.calls.append("failure")

    async def record_session_allocation_success(self, requester=None):
        self.calls.append("success")

    async def record_session_request_abandoned(self, requester=None):
        self.calls.append("abandoned")

    async def record_session_rate_limited(self, requester=None):
        self.calls.append("rate_limited")

    async def record_session_event(self, event, **kwargs):
        self.session_events.append(event)

    async def record_requester_session_connected(self, requester):
        self.connected_requesters.append(requester)

    async def record_requester_session_disconnected(
        self,
        requester,
        *,
        duration_s,
        short_session,
    ):
        self.disconnected_requesters.append((requester, duration_s, short_session))


class FakeSessionManager:
    def __init__(self, *, allocation_wait_ms: int = 1200):
        self.cancelled_session_ids = []
        self.allocation_wait_ms = allocation_wait_ms
        self.allocation_calls = 0
        self.connected_session_ids = set()

    async def allocate(self, lb_base_url):
        # Mirrors DirectSessionManager._grant_from_lease, which stamps
        # "state": "granted" on every grant it returns.
        self.allocation_calls += 1
        return {
            "state": "granted",
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

    async def handle_event(self, session_id, session_token, event):
        was_connected = session_id in self.connected_session_ids
        if event == "connected":
            self.connected_session_ids.add(session_id)
        elif event == "disconnected":
            self.connected_session_ids.discard(session_id)
        return {
            "status": "ok",
            "session_id": session_id,
            "state": "connected" if event == "connected" else "released",
            "release_reason": "client_disconnected" if event == "disconnected" else None,
            "conversation_counted": event == "disconnected" and was_connected,
            "conversation_duration_s": (
                6.0 if event == "disconnected" and was_connected else None
            ),
        }


class FakeFailingSessionManager:
    def __init__(self, exc=None):
        self.exc = exc or RuntimeError("no capacity")

    async def allocate(self, lb_base_url):
        raise self.exc


class FakeQueuedSessionManager:
    queue_enabled = True

    def __init__(self):
        self.left = False

    async def allocate(self, lb_base_url):
        return {
            "state": "queued",
            "queue_id": "queue-123",
            "position": 1,
            "poll_interval_s": 1,
        }

    async def leave(self, queue_id):
        if queue_id != "queue-123" or self.left:
            return False
        self.left = True
        return True


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


class FakeJsonRequest(FakeConnectedRequest):
    def __init__(self, payload):
        self.payload = payload
        self.headers = {
            **self.headers,
            "content-type": "application/json",
        }

    async def stream(self):
        yield json.dumps(self.payload).encode("utf-8")


if __name__ == "__main__":
    unittest.main()

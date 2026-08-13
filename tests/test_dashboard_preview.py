import asyncio
import json
import unittest
from time import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.dashboard_history import SwarmHistoryBucket
from app.dashboard_history_store import ReadOnlyDashboardHistoryStore
from app.dashboard_preview import DashboardPreviewSessionManager
from app.endpoint_pool_router import EndpointCapacityTimeoutError, EndpointTransitionConflictError
from app.load_balancer_app import (
    create_session,
    llm_proxy_usage,
    queue_leave,
    queue_status,
    record_expired_queue_ticket,
    session_event,
)
from app.requester_identity import RequesterIdentity
from app.requester_rate_limiter import RequesterRateLimitConfig, RequesterRateLimiter
from app.session_requester_tracker import SessionRequesterTracker
from app.verification_admission_limiter import (
    VerificationAdmissionConfig,
    VerificationAdmissionLimiter,
)
from tests.helpers import load_balancer_fixture, monotonic_sequence


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
        manager = DashboardPreviewSessionManager(time_fn=clock)
        await manager.start()

        healthy, detail, snapshot = await manager.healthcheck()

        self.assertTrue(healthy)
        self.assertIn("preview mode", detail)
        self.assertTrue(snapshot["preview_mode"])
        self.assertEqual(len(snapshot["router"]["endpoints"]), 4)
        self.assertGreaterEqual(snapshot["router"]["running_endpoints"], 2)
        self.assertGreaterEqual(snapshot["router"]["effective_free_slots"], snapshot["router"]["free_slots"])

    async def test_allocation_is_disabled(self):
        manager = DashboardPreviewSessionManager()

        with self.assertRaisesRegex(RuntimeError, "preview mode"):
            await manager.allocate("https://lb.example")

    async def test_supports_common_session_manager_interface(self):
        manager = DashboardPreviewSessionManager()

        manager.set_abnormal_disconnect_handler(None)
        manager.set_ticket_expired_handler(None)
        await manager.cancel_pending_session("unknown-session")

        self.assertFalse(await manager.leave("unknown-ticket"))
        with self.assertRaisesRegex(RuntimeError, "queue is disabled"):
            await manager.poll("unknown-ticket", "https://lb.example")

    async def test_healthcheck_uses_each_endpoints_last_known_capacity(self):
        clock = FakeClock(1000.0)
        manager = DashboardPreviewSessionManager(
            last_known_capacities={
                "preview-compute-01": 1,
                "preview-compute-02": 3,
                "preview-compute-03": 4,
                "preview-compute-04": 2,
            },
            time_fn=clock,
        )

        _, _, snapshot = await manager.healthcheck()

        endpoints = {endpoint["name"]: endpoint for endpoint in snapshot["router"]["endpoints"]}
        self.assertEqual(endpoints["preview-compute-01"]["max_sessions"], 1)
        self.assertEqual(endpoints["preview-compute-02"]["max_sessions"], 3)
        self.assertEqual(endpoints["preview-compute-03"]["max_sessions"], 4)
        self.assertEqual(endpoints["preview-compute-04"]["max_sessions"], 2)
        self.assertEqual(snapshot["router"]["warming_slots"], 4)
        self.assertEqual(
            snapshot["router"]["effective_free_slots"],
            snapshot["router"]["free_slots"] + 4,
        )


class LoadBalancerPreviewModeTests(unittest.TestCase):
    def test_compute_endpoint_names_test_enables_preview_without_session_secret(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
            }
        )

        self.assertTrue(module.settings.dashboard_preview_mode)
        self.assertIsInstance(module.dependencies.session_manager, DashboardPreviewSessionManager)
        self.assertEqual(module.settings.compute_endpoint_names[0], "preview-compute-01")

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

        self.assertTrue(module.settings.dashboard_preview_mode)
        self.assertIsInstance(module.dependencies.dashboard_history_store, ReadOnlyDashboardHistoryStore)
        self.assertIs(module.dependencies.dashboard.history_store, module.dependencies.dashboard_history_store)
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
        module.dependencies.session_manager = session_manager
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
                return (
                    True,
                    None,
                    {
                        "router": {
                            "endpoints": [
                                {
                                    "name": "reachy-s2s-01",
                                    "status": "running",
                                    "draining": False,
                                }
                            ]
                        }
                    },
                )

        module.dependencies.session_manager = ConflictingSessionManager()
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
                return (
                    True,
                    None,
                    {
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
                    },
                )

        session_manager = RecordingSessionManager()
        module.dependencies.session_manager = session_manager
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

        module.dependencies.session_manager = UnhealthySessionManager()
        client = TestClient(module.app)

        health = client.get("/health")
        endpoint_status = client.get(
            "/internal/endpoints/reachy-s2s-01",
            headers={"Authorization": "Bearer admin-secret"},
        )

        self.assertEqual(health.status_code, 503)
        self.assertEqual(health.json()["status"], "unhealthy")
        self.assertEqual(
            health.json()["sessions"]["router"]["endpoints"][0]["name"],
            "reachy-s2s-01",
        )
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

    def test_llm_proxy_callback_authenticates_before_bounded_body_read(self):
        module = self._import_load_balancer(
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "SESSION_SHARED_SECRET": "",
                "LB_CALLBACK_AUTH_TOKEN": "callback-secret",
            }
        )
        client = TestClient(module.app)
        oversized_body = b"x" * 8193

        unauthenticated = client.post("/internal/llm-proxy-usage", content=oversized_body)
        authenticated = client.post(
            "/internal/llm-proxy-usage",
            headers={"Authorization": "Bearer callback-secret"},
            content=oversized_body,
        )

        self.assertEqual(unauthenticated.status_code, 401)
        self.assertEqual(authenticated.status_code, 413)

    def _import_load_balancer(self, env):
        return load_balancer_fixture({"LB_ADMIN_AUTH_TOKEN": "", **env})


class LoadBalancerSessionHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_delayed_disconnected_session_allocation_skips_no_connect_penalty(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager

        request = FakeDisconnectedRequest()

        with (
            patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 21.5)),
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, request)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(fake_session_manager.cancelled_session_ids, ["session-123"])
        self.assertEqual(fake_dashboard.calls, ["request", "abandoned"])
        status = module.dependencies.requester_rate_limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertNotIn("no_connects", status["totals"])
        record = logs.records[0]
        self.assertEqual(record.outcome, "client_disconnected")
        self.assertEqual(record.session_id, "session-123")
        self.assertEqual(record.endpoint_name, "endpoint-a")
        self.assertEqual(record.slot_id, "endpoint-a")
        self.assertEqual(record.allocation_wait_ms, 1200)
        self.assertEqual(record.allocation_total_ms, 1500)
        self.assertTrue(record.waited_for_capacity)
        self.assertTrue(record.no_connect_penalty_excluded)
        self.assertIn("outcome=client_disconnected", record.getMessage())
        self.assertIn("endpoint_name=endpoint-a", record.getMessage())
        self.assertIn("allocation_wait_ms=1200", record.getMessage())
        self.assertIn("allocation_total_ms=1500", record.getMessage())
        self.assertIn("no_connect_penalty_excluded=True", record.getMessage())

    async def test_disconnected_queue_grant_releases_permit_without_no_connect_penalty(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeQueuedGrantSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager

        response = await create_session(module.runtime, FakeConnectedRequest())
        self.assertEqual(json.loads(response.body)["state"], "queued")
        self.assertEqual(module.dependencies.requester_rate_limiter.status()["active_allocations"], 1)

        with (
            patch("app.load_balancer_app.monotonic", new=monotonic_sequence(30.0, 30.2)),
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await queue_status(module.runtime, "queue-123", FakeDisconnectedRequest())

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(fake_session_manager.cancelled_session_ids, ["session-123"])
        self.assertEqual(fake_dashboard.calls, ["request", "abandoned"])
        status = module.dependencies.requester_rate_limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["allocations"], 1)
        self.assertNotIn("no_connects", status["totals"])
        decision = module.dependencies.requester_rate_limiter.acquire(fake_dashboard.requesters[0])
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.consecutive_no_connects, 0)
        record = logs.records[0]
        self.assertEqual(record.http_route, "GET /queue/{queue_id}")
        self.assertEqual(record.allocation_wait_ms, 12_000)
        self.assertTrue(record.no_connect_penalty_excluded)

    async def test_promptly_delivered_allocation_without_join_is_penalized(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        clock = FakeClock(0.0)
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeSessionManager(
            allocation_wait_ms=40,
            waited_for_capacity=False,
        )
        module.dependencies.requester_rate_limiter = RequesterRateLimiter(
            config=RequesterRateLimitConfig(max_consecutive_no_connects=1),
            time_fn=clock,
        )

        with patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)):
            response = await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(module.dependencies.requester_rate_limiter.status()["active_allocations"], 1)
        clock.now = 60.0
        decision = module.dependencies.requester_rate_limiter.acquire(fake_dashboard.requesters[0])

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "behavior_cooldown")
        self.assertEqual(decision.consecutive_no_connects, 1)
        status = module.dependencies.requester_rate_limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["no_connects"], 1)

    async def test_successful_session_allocation_logs_outcome(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with (
            patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)),
            self.assertLogs("s2s-endpoint", level="INFO") as logs,
        ):
            response = await create_session(module.runtime, FakeConnectedRequest())

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
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        module.dependencies.requester_rate_limiter = RequesterRateLimiter(
            config=RequesterRateLimitConfig(max_parallel_allocations=1)
        )

        with patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)):
            await create_session(module.runtime, FakeConnectedRequest())

        with (
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, FakeConnectedRequest())

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

    async def test_required_token_rejects_anonymous_request_before_rate_limit_or_allocation(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager

        with (
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.headers["WWW-Authenticate"], "Bearer")
        self.assertEqual(raised.exception.detail["reason"], "token_not_provided")
        self.assertEqual(fake_session_manager.allocation_calls, 0)
        self.assertEqual(fake_dashboard.calls, ["request", "auth_rejected"])
        self.assertEqual(module.dependencies.requester_rate_limiter.status()["active_allocations"], 0)
        self.assertEqual(logs.records[0].outcome, "auth_rejected")
        self.assertEqual(logs.records[0].auth_stage, "admission")

    async def test_required_token_rejects_malformed_and_unrecognized_credentials(self):
        cases = (
            ("malformed", "Basic not-a-bearer", "token_not_provided"),
            ("unrecognized", f"Bearer {'x' * 4097}", "token_unrecognized"),
        )
        for label, authorization, reason in cases:
            with self.subTest(label=label):
                module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
                fake_dashboard = FakeDashboard()
                fake_session_manager = FakeSessionManager()
                module.dependencies.dashboard = fake_dashboard
                module.dependencies.session_manager = fake_session_manager

                with self.assertRaises(HTTPException) as raised:
                    await create_session(module.runtime, FakeHeaderRequest({"authorization": authorization}))

                self.assertEqual(raised.exception.status_code, 401)
                self.assertEqual(raised.exception.headers["WWW-Authenticate"], "Bearer")
                self.assertEqual(raised.exception.detail["reason"], reason)
                self.assertEqual(fake_session_manager.allocation_calls, 0)
                self.assertEqual(fake_dashboard.calls, ["request", "auth_rejected"])

    async def test_required_token_uses_cached_failure_lifetime_for_retry_after(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        unavailable = _requester_identity(verification="unavailable")

        with (
            patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=unavailable),
            patch.object(
                module.dependencies.requester_identity_resolver,
                "verification_retry_after_s",
                return_value=60,
            ) as verification_retry_after_s,
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.headers["Retry-After"], "60")
        self.assertEqual(raised.exception.detail["retry_after_s"], 60)
        self.assertEqual(raised.exception.detail["reason"], "verification_unavailable")
        self.assertEqual(fake_session_manager.allocation_calls, 0)
        self.assertEqual(fake_dashboard.calls, ["request", "auth_rejected"])
        verification_retry_after_s.assert_called_once_with(unavailable)

    async def test_required_token_returns_retryable_503_when_verification_times_out(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        pending = _requester_identity(verification="pending")

        with (
            patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=pending),
            patch.object(
                module.dependencies.requester_identity_resolver,
                "wait_for_verification",
                new=AsyncMock(return_value=pending),
            ) as wait_for_verification,
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.headers["Retry-After"], "1")
        self.assertEqual(raised.exception.detail["reason"], "verification_timeout")
        self.assertEqual(fake_session_manager.allocation_calls, 0)
        self.assertEqual(fake_dashboard.calls, ["request", "auth_rejected"])
        wait_for_verification.assert_awaited_once_with(
            pending,
            timeout_s=module.settings.session_hf_token_verify_timeout_s,
        )

    async def test_pre_verification_network_quota_blocks_distinct_tokens_before_resolver_queue(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        module.dependencies.session_verification_limiter = VerificationAdmissionLimiter(
            config=VerificationAdmissionConfig(max_global_pending=2, max_network_pending=1)
        )
        first = RequesterIdentity(
            **{
                **_requester_identity(verification="pending").__dict__,
                "actor_id": "token:first",
                "fingerprint": "first",
                "network_id": "net:same",
            }
        )
        second = RequesterIdentity(
            **{
                **_requester_identity(verification="pending").__dict__,
                "actor_id": "token:second",
                "fingerprint": "second",
                "network_id": "net:same",
            }
        )
        release_validation = asyncio.Event()
        validation_task = asyncio.create_task(release_validation.wait())

        with (
            patch.object(module.dependencies.requester_identity_resolver, "identify", side_effect=[first, second]),
            patch.object(
                module.dependencies.requester_identity_resolver,
                "start_verification",
                return_value=(first, validation_task, True),
            ) as start_verification,
            patch.object(
                module.dependencies.requester_identity_resolver,
                "wait_for_verification",
                new=AsyncMock(return_value=first),
            ),
        ):
            with self.assertRaises(HTTPException):
                await create_session(
                    module.runtime, FakeHeaderRequest({"authorization": "Bearer hf_first_fabricated_token"})
                )
            with self.assertRaises(HTTPException) as blocked:
                await create_session(
                    module.runtime, FakeHeaderRequest({"authorization": "Bearer hf_second_fabricated_token"})
                )

        self.assertEqual(blocked.exception.status_code, 503)
        self.assertEqual(blocked.exception.detail["reason"], "verification_network_quota")
        self.assertEqual(start_verification.call_count, 1)
        self.assertEqual(fake_session_manager.allocation_calls, 0)
        self.assertEqual(module.dependencies.session_verification_limiter.status()["pending"], 1)

        release_validation.set()
        await validation_task
        await asyncio.sleep(0)
        self.assertEqual(module.dependencies.session_verification_limiter.status()["pending"], 0)

    async def test_required_token_rejects_invalid_identity_without_allocation(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        invalid = _requester_identity(verification="invalid", kind="invalid_token")

        with (
            patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=invalid),
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.detail["reason"], "token_invalid")
        self.assertEqual(fake_session_manager.allocation_calls, 0)
        self.assertEqual(fake_dashboard.calls, ["request", "auth_rejected"])

    async def test_required_verified_token_can_receive_immediate_grant(self):
        module = self._import_load_balancer(
            {
                "SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true",
                "SESSION_SHARED_SECRET": "shared-secret",
            }
        )
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        verified = _requester_identity(verification="verified", kind="authenticated")

        with patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=verified):
            response = await create_session(
                module.runtime,
                FakeHeaderRequest({"authorization": "Bearer hf_faketesttoken1234"}),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_session_manager.allocation_calls, 1)
        self.assertIsNotNone(fake_session_manager.allocation_arguments[0]["llm_fingerprint"])
        self.assertEqual(fake_dashboard.calls, ["request", "success"])

    async def test_default_verification_age_allows_the_full_default_allocation_wait(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        clock = FakeClock(0.0)
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        module.dependencies.requester_identity_resolver._time_fn = clock
        verified = RequesterIdentity(
            **{
                **_requester_identity(verification="verified", kind="authenticated").__dict__,
                "verified_at_s": 0.0,
            }
        )
        allocate = fake_session_manager.allocate

        async def allocate_after_default_wait(*args, **kwargs):
            result = await allocate(*args, **kwargs)
            clock.now = 900.0
            return result

        fake_session_manager.allocate = allocate_after_default_wait

        with patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=verified):
            response = await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(module.settings.session_hf_token_max_verified_age_s, 1800.0)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_session_manager.cancelled_session_ids, [])
        self.assertEqual(fake_dashboard.calls, ["request", "success"])

    async def test_stale_cached_verification_is_refreshed_before_admission(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        fresh = _requester_identity(verification="verified", kind="authenticated")
        stale = RequesterIdentity(**{**fresh.__dict__, "verified_at_s": 0.0})
        pending = _requester_identity(verification="pending")
        raw_token = "hf_fresh_authentication_proof"

        with (
            patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=stale),
            patch.object(
                module.dependencies.requester_identity_resolver,
                "start_verification",
                return_value=(pending, None, True),
            ) as start_verification,
            patch.object(
                module.dependencies.requester_identity_resolver,
                "wait_for_verification",
                new=AsyncMock(return_value=fresh),
            ),
        ):
            response = await create_session(module.runtime, FakeHeaderRequest({"authorization": f"Bearer {raw_token}"}))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(fake_session_manager.allocation_calls, 1)
        start_verification.assert_called_once_with(raw_token, stale, force=True)

    async def test_required_verified_token_is_preserved_through_queue_grant(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeQueuedGrantSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        verified = _requester_identity(verification="verified", kind="authenticated")

        with patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=verified):
            queued = await create_session(module.runtime, FakeConnectedRequest())
            granted = await queue_status(module.runtime, "queue-123", FakeConnectedRequest())

        self.assertEqual(json.loads(queued.body)["state"], "queued")
        self.assertEqual(json.loads(granted.body)["state"], "granted")
        self.assertEqual(fake_session_manager.poll_calls, 1)
        self.assertEqual(fake_dashboard.calls, ["request", "success"])

    async def test_default_verification_age_preserves_queue_ticket_past_60_seconds(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeQueuedGrantSessionManager()
        clock = FakeClock(0.0)
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        module.dependencies.requester_identity_resolver._time_fn = clock
        verified = RequesterIdentity(
            **{
                **_requester_identity(verification="verified", kind="authenticated").__dict__,
                "verified_at_s": 0.0,
            }
        )

        with patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=verified):
            queued = await create_session(module.runtime, FakeConnectedRequest())
            clock.now = 61.0
            granted = await queue_status(module.runtime, "queue-123", FakeConnectedRequest())

        self.assertEqual(module.settings.session_hf_token_max_verified_age_s, 1800.0)
        self.assertEqual(json.loads(queued.body)["state"], "queued")
        self.assertEqual(json.loads(granted.body)["state"], "granted")
        self.assertFalse(fake_session_manager.left)
        self.assertEqual(fake_session_manager.poll_calls, 1)
        self.assertEqual(fake_dashboard.calls, ["request", "success"])

    async def test_expired_queue_identity_is_rejected_before_claiming_capacity(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeQueuedGrantSessionManager()
        clock = FakeClock(0.0)
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        module.dependencies.queue_requester_tracker = SessionRequesterTracker(retention_s=1, time_fn=clock)
        verified = _requester_identity(verification="verified", kind="authenticated")

        with patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=verified):
            await create_session(module.runtime, FakeConnectedRequest())
            clock.now = 2.0
            with self.assertRaises(HTTPException) as raised:
                await queue_status(module.runtime, "queue-123", FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.detail["reason"], "queue_identity_expired")
        self.assertEqual(fake_session_manager.poll_calls, 0)
        self.assertTrue(fake_session_manager.left)
        self.assertEqual(module.dependencies.requester_rate_limiter.status()["active_allocations"], 0)
        self.assertEqual(fake_dashboard.calls, ["request", "auth_rejected"])

    async def test_queue_identity_must_still_be_verified_before_claiming_capacity(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeQueuedGrantSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        verified = _requester_identity(verification="verified", kind="authenticated")
        invalid = _requester_identity(verification="invalid", kind="invalid_token")

        with (
            patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=verified),
            patch.object(
                module.dependencies.requester_identity_resolver,
                "latest_identity",
                side_effect=[verified, invalid],
            ),
        ):
            await create_session(module.runtime, FakeConnectedRequest())
            with self.assertRaises(HTTPException) as raised:
                await queue_status(module.runtime, "queue-123", FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(raised.exception.detail["reason"], "token_invalid")
        self.assertEqual(fake_session_manager.poll_calls, 0)
        self.assertTrue(fake_session_manager.left)
        self.assertEqual(module.dependencies.requester_rate_limiter.status()["active_allocations"], 0)
        self.assertEqual(fake_dashboard.calls, ["request", "auth_rejected"])

    async def test_final_grant_guard_releases_session_if_identity_is_no_longer_verified(self):
        module = self._import_load_balancer({"SESSION_REQUIRE_VERIFIED_HF_TOKEN": "true"})
        fake_dashboard = FakeDashboard()
        fake_session_manager = FakeSessionManager()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = fake_session_manager
        verified = _requester_identity(verification="verified", kind="authenticated")
        invalid = _requester_identity(verification="invalid", kind="invalid_token")

        with (
            patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=verified),
            patch.object(
                module.dependencies.requester_identity_resolver,
                "latest_identity",
                side_effect=[verified, invalid],
            ),
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(fake_session_manager.cancelled_session_ids, ["session-123"])
        self.assertEqual(fake_dashboard.calls, ["request", "auth_rejected"])
        status = module.dependencies.requester_rate_limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["allocation_auth_rejections"], 1)

    async def test_leaving_queue_releases_parallel_allocation_permit(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeQueuedSessionManager()

        response = await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(json.loads(response.body)["state"], "queued")
        self.assertEqual(module.dependencies.requester_rate_limiter.status()["active_allocations"], 1)

        await queue_leave(module.runtime, "queue-123")

        status = module.dependencies.requester_rate_limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["allocation_abandonments"], 1)
        self.assertEqual(fake_dashboard.calls, ["request", "abandoned"])

    async def test_expired_queue_ticket_releases_parallel_allocation_permit(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeQueuedSessionManager()

        await create_session(module.runtime, FakeConnectedRequest())
        await record_expired_queue_ticket(module.runtime, "queue-123")

        status = module.dependencies.requester_rate_limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["allocation_abandonments"], 1)
        self.assertEqual(fake_dashboard.calls, ["request", "abandoned"])

    async def test_session_allocation_tracks_reported_hardware_id_as_fingerprint(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeSessionManager(allocation_wait_ms=40)
        raw_hardware_id = "ABCDEF0123456789"

        with patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)):
            response = await create_session(module.runtime, FakeJsonRequest({"hardware_id": raw_hardware_id}))

        self.assertEqual(response.status_code, 200)
        requester = fake_dashboard.requesters[0]
        self.assertTrue(requester.reported_robot_id.startswith("robot:"))
        self.assertNotIn(raw_hardware_id.lower(), requester.reported_robot_id)

    async def test_session_allocation_ignores_invalid_reported_hardware_id(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)):
            response = await create_session(module.runtime, FakeJsonRequest({"hardware_id": "invalid"}))

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(fake_dashboard.requesters[0].reported_robot_id)

    async def test_connected_callback_is_attributed_to_requester_once(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)):
            await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(module.dependencies.session_requester_tracker.count(), 1)
        payload = {"session_token": "session-token", "event": "connected"}
        first = await session_event(module.runtime, "session-123", payload)
        second = await session_event(module.runtime, "session-123", payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(module.dependencies.session_requester_tracker.count(), 0)
        self.assertEqual(fake_dashboard.session_events, ["connected", "connected"])
        self.assertEqual(fake_dashboard.connected_requesters, [fake_dashboard.requesters[0]])

    async def test_valid_token_is_grouped_by_token_identity_with_or_without_session_match(self):
        module = self._import_load_balancer({"LB_CALLBACK_AUTH_TOKEN": "callback-secret"})
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        requester = RequesterIdentity(
            actor_id="token:abc123",
            label="@reachy-user · token •abc123",
            kind="authenticated",
            verification="verified",
            fingerprint="abc123",
            account_name="reachy-user",
            network_id="net:network123",
        )
        callback_request = SimpleNamespace(headers={"authorization": "Bearer callback-secret"})
        with patch.object(module.dependencies.requester_identity_resolver, "identify_values", return_value=requester):
            accepted = await llm_proxy_usage(
                module.runtime,
                callback_request,
                {
                    "outcome": "accepted",
                    "reason": "accepted",
                    "session_matched": True,
                    "credential_present": True,
                    "token": "hf_valid_token",
                    "client_ip": "203.0.113.8",
                },
            )
            rejected = await llm_proxy_usage(
                module.runtime,
                callback_request,
                {
                    "outcome": "rejected",
                    "reason": "no_active_session_match",
                    "session_matched": False,
                    "credential_present": True,
                    "token": "hf_valid_token",
                    "client_ip": "203.0.113.8",
                },
            )

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 200)
        self.assertEqual(
            fake_dashboard.llm_proxy_requests,
            [
                ("accepted", "accepted", requester.actor_id, requester.history_metadata()),
                (
                    "rejected",
                    "no_active_session_match",
                    requester.actor_id,
                    requester.history_metadata(),
                ),
            ],
        )
        persisted_record = json.dumps(fake_dashboard.llm_proxy_requests)
        self.assertNotIn("hf_valid_token", persisted_record)
        self.assertNotIn("203.0.113.8", persisted_record)

    async def test_llm_proxy_missing_token_falls_back_to_privacy_safe_ip(self):
        module = self._import_load_balancer({"LB_CALLBACK_AUTH_TOKEN": "callback-secret"})
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        network_requester = RequesterIdentity(
            actor_id="anonymous:network123",
            label="Anonymous IP •network1",
            kind="anonymous",
            verification="not_provided",
            fingerprint="network123",
            network_id="net:network123",
        )

        with patch.object(
            module.dependencies.requester_identity_resolver,
            "identify_values",
            return_value=network_requester,
        ):
            response = await llm_proxy_usage(
                module.runtime,
                SimpleNamespace(headers={"authorization": "Bearer callback-secret"}),
                {
                    "outcome": "rejected",
                    "reason": "missing_token",
                    "session_matched": False,
                    "credential_present": False,
                    "client_ip": "203.0.113.8",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            fake_dashboard.llm_proxy_requests,
            [
                (
                    "rejected",
                    "missing_token",
                    network_requester.actor_id,
                    network_requester.history_metadata(),
                )
            ],
        )
        self.assertNotIn("203.0.113.8", json.dumps(fake_dashboard.llm_proxy_requests))

    async def test_llm_proxy_callback_requires_compute_authentication(self):
        module = self._import_load_balancer({"LB_CALLBACK_AUTH_TOKEN": "callback-secret"})
        payload = {
            "outcome": "rejected",
            "reason": "missing_token",
            "session_matched": False,
            "credential_present": False,
            "client_ip": "203.0.113.8",
        }

        with self.assertRaises(HTTPException) as missing_auth:
            await llm_proxy_usage(
                module.runtime,
                SimpleNamespace(headers={}),
                payload,
            )
        with self.assertRaises(HTTPException) as user_token_auth:
            await llm_proxy_usage(
                module.runtime,
                SimpleNamespace(headers={"authorization": "Bearer hf_valid_token"}),
                payload,
            )

        self.assertEqual(missing_auth.exception.status_code, 401)
        self.assertEqual(user_token_auth.exception.status_code, 403)

    async def test_llm_proxy_invalid_token_falls_back_to_privacy_safe_ip(self):
        module = self._import_load_balancer({"LB_CALLBACK_AUTH_TOKEN": "callback-secret"})
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        token_requester = RequesterIdentity(
            actor_id="token:abc123",
            label="Invalid token •abc123",
            kind="invalid_token",
            verification="invalid",
            fingerprint="abc123",
        )
        network_requester = RequesterIdentity(
            actor_id="anonymous:network123",
            label="Anonymous IP •network1",
            kind="anonymous",
            verification="not_provided",
            fingerprint="network123",
            network_id="net:network123",
        )

        with (
            patch.object(
                module.dependencies.requester_identity_resolver,
                "identify_values",
                side_effect=[token_requester, network_requester],
            ),
            patch.object(
                module.dependencies.requester_identity_resolver,
                "latest_identity",
                return_value=token_requester,
            ),
        ):
            response = await llm_proxy_usage(
                module.runtime,
                SimpleNamespace(headers={"authorization": "Bearer callback-secret"}),
                {
                    "outcome": "rejected",
                    "reason": "no_active_session_match",
                    "session_matched": False,
                    "credential_present": True,
                    "token": "hf_invalid_token",
                    "client_ip": "203.0.113.8",
                },
            )

        self.assertEqual(response.status_code, 200)
        _, reason, actor_id, metadata = fake_dashboard.llm_proxy_requests[0]
        self.assertEqual(reason, "no_active_session_match")
        self.assertEqual(actor_id, network_requester.actor_id)
        self.assertEqual(metadata["verification"], "invalid")
        self.assertNotIn("hf_invalid_token", str(metadata))
        self.assertNotIn("203.0.113.8", str(metadata))

    async def test_disconnected_callback_records_requester_duration_after_connect(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)):
            await create_session(module.runtime, FakeConnectedRequest())

        await session_event(
            module.runtime,
            "session-123",
            {"session_token": "session-token", "event": "connected"},
        )
        await session_event(
            module.runtime,
            "session-123",
            {"session_token": "session-token", "event": "disconnected"},
        )

        self.assertEqual(
            fake_dashboard.disconnected_requesters,
            [(fake_dashboard.requesters[0], 6.0, True)],
        )

    async def test_disconnected_callback_refreshes_stale_allocation_identity(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeSessionManager(allocation_wait_ms=40)
        pending = RequesterIdentity(
            actor_id="token:abc123",
            label="HF token •abc123",
            kind="unverified_token",
            verification="pending",
            fingerprint="abc123",
        )
        verified = RequesterIdentity(
            actor_id="token:abc123",
            label="@reachy-user · token •abc123",
            kind="authenticated",
            verification="verified",
            fingerprint="abc123",
            account_name="reachy-user",
        )

        with (
            patch.object(module.dependencies.requester_identity_resolver, "identify", return_value=pending),
            patch.object(
                module.dependencies.requester_identity_resolver,
                "latest_identity",
                side_effect=[pending, verified, verified, verified],
            ),
            patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)),
        ):
            await create_session(module.runtime, FakeConnectedRequest())
            await session_event(
                module.runtime,
                "session-123",
                {"session_token": "session-token", "event": "connected"},
            )
            await session_event(
                module.runtime,
                "session-123",
                {"session_token": "session-token", "event": "disconnected"},
            )

        self.assertEqual(fake_dashboard.connected_requesters, [verified])
        self.assertEqual(
            fake_dashboard.disconnected_requesters,
            [(verified, 6.0, True)],
        )

    async def test_pre_connect_compute_rejection_does_not_penalize_requester(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeSessionManager(allocation_wait_ms=40)

        with patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.05)):
            await create_session(module.runtime, FakeConnectedRequest())

        await session_event(
            module.runtime,
            "session-123",
            {"session_token": "session-token", "event": "disconnected"},
        )

        totals = module.dependencies.requester_rate_limiter.status()["totals"]
        self.assertNotIn("no_connects", totals)
        self.assertEqual(module.dependencies.requester_rate_limiter.status()["active_allocations"], 0)

    async def test_failed_session_allocation_logs_outcome(self):
        module = self._import_load_balancer()
        fake_dashboard = FakeDashboard()
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeFailingSessionManager()

        with (
            patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.25)),
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, FakeConnectedRequest())

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
        module.dependencies.dashboard = fake_dashboard
        module.dependencies.session_manager = FakeFailingSessionManager(
            EndpointCapacityTimeoutError("timed out waiting for an available compute endpoint")
        )

        with (
            patch("app.load_balancer_app.monotonic", new=monotonic_sequence(20.0, 20.25)),
            self.assertLogs("s2s-endpoint", level="WARNING") as logs,
            self.assertRaises(HTTPException) as raised,
        ):
            await create_session(module.runtime, FakeConnectedRequest())

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(fake_dashboard.calls, ["request", "failure"])
        record = logs.records[0]
        self.assertEqual(record.outcome, "allocation_failed")
        self.assertEqual(record.allocation_wait_ms, 250)
        self.assertEqual(record.allocation_total_ms, 250)
        self.assertTrue(record.waited_for_capacity)

    def _import_load_balancer(self, env=None):
        return load_balancer_fixture(env)


class FakeDashboard:
    def __init__(self):
        self.calls = []
        self.requesters = []
        self.session_events = []
        self.connected_requesters = []
        self.disconnected_requesters = []
        self.identity_updates = []
        self.llm_proxy_requests = []

    async def record_session_request(self, requester=None):
        self.calls.append("request")
        self.requesters.append(requester)

    async def record_session_allocation_failure(self, requester=None):
        self.calls.append("failure")

    async def record_session_allocation_success(self, requester=None):
        self.calls.append("success")

    async def record_session_auth_rejected(self, requester=None):
        self.calls.append("auth_rejected")

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

    async def update_requester_identity(self, requester):
        self.identity_updates.append(requester)

    async def record_llm_proxy_request(self, outcome, *, reason, actor_id, metadata):
        self.llm_proxy_requests.append((outcome, reason, actor_id, metadata))


class FakeSessionManager:
    def __init__(self, *, allocation_wait_ms: int = 1200, waited_for_capacity: bool = True):
        self.cancelled_session_ids = []
        self.allocation_wait_ms = allocation_wait_ms
        self.waited_for_capacity = waited_for_capacity
        self.allocation_calls = 0
        self.allocation_arguments = []
        self.connected_session_ids = set()

    async def allocate(self, lb_base_url, *, llm_fingerprint=None):
        # Mirrors DirectSessionManager._grant_from_lease, which stamps
        # "state": "granted" on every grant it returns.
        self.allocation_calls += 1
        self.allocation_arguments.append(
            {
                "llm_fingerprint": llm_fingerprint,
            }
        )
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
            "waited_for_capacity": self.waited_for_capacity,
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
            "conversation_duration_s": (6.0 if event == "disconnected" and was_connected else None),
        }


class FakeFailingSessionManager:
    def __init__(self, exc=None):
        self.exc = exc or RuntimeError("no capacity")

    async def allocate(self, lb_base_url, *, llm_fingerprint=None):
        raise self.exc


class FakeQueuedSessionManager:
    queue_enabled = True

    def __init__(self):
        self.left = False

    async def allocate(self, lb_base_url, *, llm_fingerprint=None):
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


class FakeQueuedGrantSessionManager(FakeQueuedSessionManager):
    def __init__(self):
        super().__init__()
        self.cancelled_session_ids = []
        self.poll_calls = 0

    async def poll(self, queue_id, lb_base_url):
        self.poll_calls += 1
        if queue_id != "queue-123":
            raise KeyError(queue_id)
        return {
            "state": "granted",
            "session_id": "session-123",
            "websocket_url": "wss://endpoint-a.example/ws",
            "connect_url": f"{lb_base_url}ws?session=session-123",
            "session_token": "session-token",
            "pending_timeout_s": 60,
            "endpoint_name": "endpoint-a",
            "slot_id": "endpoint-a",
            "allocation_wait_ms": 12_000,
            "waited_for_capacity": True,
        }

    async def cancel_pending_session(self, session_id):
        self.cancelled_session_ids.append(session_id)


def _requester_identity(*, verification, kind="unverified_token"):
    return RequesterIdentity(
        actor_id="token:abc123",
        label="HF token •abc123",
        kind=kind,
        verification=verification,
        fingerprint="abc123",
        account_name="reachy-user" if verification == "verified" else None,
        verified_at_s=time() if verification == "verified" else None,
    )


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


class FakeHeaderRequest(FakeConnectedRequest):
    def __init__(self, headers):
        self.headers = {**self.headers, **headers}


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

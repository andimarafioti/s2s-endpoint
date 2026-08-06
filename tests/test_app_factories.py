import asyncio
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError

from fastapi.testclient import TestClient

from app import compute_app, load_balancer_app


class _ComputeRouter:
    def __init__(self, label: str):
        self.label = label
        self.starts = 0
        self.stops = 0

    async def healthcheck(self):
        return True, "", {"label": self.label}

    async def start(self):
        self.starts += 1

    async def stop(self):
        self.stops += 1


def _compute_dependencies(label: str) -> compute_app.ComputeDependencies:
    async def notify(*args, **kwargs):
        pass

    async def proxy(*args, **kwargs):
        pass

    return compute_app.ComputeDependencies(
        session_router=_ComputeRouter(label),
        connected_llm_fingerprints=compute_app._ConnectedFingerprintRegistry(),
        llm_rate_limiter=compute_app._FingerprintRateLimiter(10),
        http_get_json=lambda url: {},
        notify_lb_session_event=notify,
        proxy_websocket=proxy,
    )


class _LoadBalancerManager:
    queue_enabled = False

    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.abnormal_disconnect_handler = None
        self.ticket_expired_handler = None

    def set_abnormal_disconnect_handler(self, handler):
        self.abnormal_disconnect_handler = handler

    def set_ticket_expired_handler(self, handler):
        self.ticket_expired_handler = handler

    async def start(self):
        self.starts += 1

    async def stop(self):
        self.stops += 1


class _LoadBalancerDashboard:
    def __init__(self):
        self.starts = 0
        self.stops = 0
        self.session_events = []
        self.abandoned_requesters = []

    async def start(self):
        self.starts += 1

    async def stop(self):
        self.stops += 1

    async def record_session_event(self, event, **details):
        self.session_events.append((event, details))

    async def record_session_request_abandoned(self, requester):
        self.abandoned_requesters.append(requester)


class _Resolver:
    def __init__(self):
        self.stops = 0

    async def stop(self):
        self.stops += 1


class _EmptyTracker:
    def take_with_expiry(self, key):
        return None, False


def _load_balancer_dependencies():
    return load_balancer_app.LoadBalancerDependencies(
        session_manager=_LoadBalancerManager(),
        dashboard_history_store=None,
        dashboard=_LoadBalancerDashboard(),
        requester_identity_resolver=_Resolver(),
        session_verification_limiter=object(),
        requester_rate_limiter=object(),
        session_requester_tracker=object(),
        queue_requester_tracker=_EmptyTracker(),
    )


class ComputeSettingsTests(unittest.TestCase):
    def test_from_env_preserves_boolean_and_api_key_precedence(self):
        settings = compute_app.ComputeSettings.from_env(
            {
                "ENABLE_LLM_PROXY": "yes",
                "ENABLE_SMART_TURN": "0",
                "RESPONSES_API_API_KEY": " explicit-key ",
                "HF_TOKEN": "fallback-key",
            }
        )

        self.assertTrue(settings.enable_llm_proxy)
        self.assertFalse(settings.enable_smart_turn)
        self.assertEqual(settings.responses_api_api_key, "explicit-key")

    def test_settings_are_frozen(self):
        settings = compute_app.ComputeSettings()

        with self.assertRaises(FrozenInstanceError):
            settings.stt = "other"  # type: ignore[misc]


class ComputeApplicationFactoryTests(unittest.TestCase):
    def test_apps_keep_settings_and_dependencies_isolated(self):
        first = compute_app.create_app(
            compute_app.ComputeSettings(stt="first-stt", internal_ws_base_port=9101),
            _compute_dependencies("first-router"),
        )
        second = compute_app.create_app(
            compute_app.ComputeSettings(stt="second-stt", internal_ws_base_port=9102),
            _compute_dependencies("second-router"),
        )

        first_root = TestClient(first).get("/").json()
        second_health = TestClient(second).get("/health").json()
        first_health = TestClient(first).get("/health").json()

        self.assertEqual(first_root["config"]["stt"], "first-stt")
        self.assertEqual(first_root["internal_ws"], "ws://127.0.0.1:9101/v1/realtime")
        self.assertEqual(second_health["stt"], "second-stt")
        self.assertEqual(second_health["router"]["label"], "second-router")
        self.assertEqual(first_health["router"]["label"], "first-router")
        self.assertIsNot(first.state.dependencies, second.state.dependencies)

    def test_two_app_lifespans_own_their_routers(self):
        first_dependencies = _compute_dependencies("first-router")
        second_dependencies = _compute_dependencies("second-router")
        first = compute_app.create_app(compute_app.ComputeSettings(), first_dependencies)
        second = compute_app.create_app(compute_app.ComputeSettings(), second_dependencies)

        with TestClient(first), TestClient(second):
            self.assertEqual(first_dependencies.session_router.starts, 1)
            self.assertEqual(second_dependencies.session_router.starts, 1)
            self.assertEqual(first_dependencies.session_router.stops, 0)
            self.assertEqual(second_dependencies.session_router.stops, 0)

        self.assertEqual(first_dependencies.session_router.stops, 1)
        self.assertEqual(second_dependencies.session_router.stops, 1)


class LoadBalancerSettingsTests(unittest.TestCase):
    def test_from_env_preserves_preview_and_secret_fallbacks(self):
        settings = load_balancer_app.LoadBalancerSettings.from_env(
            {
                "COMPUTE_ENDPOINT_NAMES": " TEST ",
                "HF_TOKEN": " hf-control ",
                "SESSION_SHARED_SECRET": " session-secret ",
            }
        )

        self.assertTrue(settings.dashboard_preview_mode)
        self.assertEqual(
            settings.compute_endpoint_names,
            tuple(f"preview-compute-{index:02d}" for index in range(1, 5)),
        )
        self.assertEqual(settings.hf_control_token, "hf-control")
        self.assertEqual(settings.dashboard_bucket_token, "hf-control")
        self.assertEqual(settings.request_usage_hash_secret, "session-secret")

    def test_settings_are_frozen(self):
        settings = load_balancer_app.LoadBalancerSettings()

        with self.assertRaises(FrozenInstanceError):
            settings.dashboard_preview_mode = True  # type: ignore[misc]

    def test_direct_construction_resolves_dependent_defaults(self):
        settings = load_balancer_app.LoadBalancerSettings(
            dashboard_preview_mode=True,
            session_shared_secret="shared",
            hf_control_token="hf-control",
            compute_endpoint_wait_timeout_s=12,
            compute_endpoint_reconcile_interval_s=30,
        )

        self.assertEqual(settings.compute_endpoint_names[0], "preview-compute-01")
        self.assertEqual(settings.request_usage_hash_secret, "shared")
        self.assertEqual(settings.dashboard_bucket_token, "hf-control")
        self.assertEqual(settings.compute_endpoint_control_operation_timeout_s, 12.0)
        self.assertEqual(settings.compute_endpoint_reconcile_stale_after_s, 90.0)


class LoadBalancerApplicationFactoryTests(unittest.TestCase):
    def test_apps_can_be_created_with_distinct_preview_settings(self):
        first_settings = load_balancer_app.LoadBalancerSettings(
            compute_endpoint_names=("first-compute",),
            dashboard_preview_mode=True,
            request_usage_hash_secret="first-secret",
        )
        second_settings = load_balancer_app.LoadBalancerSettings(
            compute_endpoint_names=("second-compute",),
            dashboard_preview_mode=True,
            request_usage_hash_secret="second-secret",
        )

        first = load_balancer_app.create_app(first_settings)
        second = load_balancer_app.create_app(second_settings)

        first_root = TestClient(first).get("/").json()
        second_root = TestClient(second).get("/").json()
        first_health = TestClient(first).get("/health").json()

        self.assertEqual(first_root["compute_endpoints"], ["first-compute"])
        self.assertEqual(second_root["compute_endpoints"], ["second-compute"])
        self.assertTrue(first_health["dashboard_preview_mode"])
        self.assertIsNot(first.state.dependencies, second.state.dependencies)

    def test_lifespans_and_background_callbacks_stay_with_their_app(self):
        first_dependencies = _load_balancer_dependencies()
        second_dependencies = _load_balancer_dependencies()
        settings = load_balancer_app.LoadBalancerSettings(dashboard_preview_mode=True)
        first = load_balancer_app.create_app(settings, first_dependencies)
        second = load_balancer_app.create_app(settings, second_dependencies)

        with TestClient(first), TestClient(second):
            self.assertEqual(first_dependencies.session_manager.starts, 1)
            self.assertEqual(second_dependencies.session_manager.starts, 1)
            self.assertIsNot(
                first_dependencies.session_manager.abnormal_disconnect_handler,
                second_dependencies.session_manager.abnormal_disconnect_handler,
            )
            asyncio.run(first_dependencies.session_manager.abnormal_disconnect_handler({"conversation_counted": False}))
            asyncio.run(second_dependencies.session_manager.ticket_expired_handler("ticket-2"))

        self.assertEqual(len(first_dependencies.dashboard.session_events), 1)
        self.assertEqual(second_dependencies.dashboard.session_events, [])
        self.assertEqual(first_dependencies.dashboard.abandoned_requesters, [])
        self.assertEqual(second_dependencies.dashboard.abandoned_requesters, [None])
        self.assertEqual(first_dependencies.session_manager.stops, 1)
        self.assertEqual(second_dependencies.session_manager.stops, 1)
        self.assertEqual(first_dependencies.requester_identity_resolver.stops, 1)
        self.assertEqual(second_dependencies.requester_identity_resolver.stops, 1)

    def test_factory_module_import_does_not_require_deployment_environment(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from app.load_balancer_app import LoadBalancerSettings, create_app",
            ],
            cwd=str(load_balancer_app.__file__).removesuffix("/app/load_balancer_app.py"),
            env={},
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()

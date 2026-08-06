import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import compute_main

with patch.dict(
    "os.environ",
    {
        "COMPUTE_ENDPOINT_NAMES": "TEST",
        "DASHBOARD_BUCKET_ID": "",
        "SESSION_SHARED_SECRET": "",
    },
    clear=False,
):
    from app import load_balancer_main


class _ComputeRouter:
    def __init__(self, label: str):
        self.label = label

    async def healthcheck(self):
        return True, "", {"label": self.label}

    async def start(self):
        pass

    async def stop(self):
        pass


def _compute_dependencies(label: str) -> compute_main.ComputeDependencies:
    async def notify(*args, **kwargs):
        pass

    async def proxy(*args, **kwargs):
        pass

    return compute_main.ComputeDependencies(
        session_router=_ComputeRouter(label),
        connected_llm_fingerprints=compute_main._ConnectedFingerprintRegistry(),
        llm_rate_limiter=compute_main._FingerprintRateLimiter(10),
        http_get_json=lambda url: {},
        post_json=lambda url, payload: None,
        notify_lb_session_event=notify,
        proxy_websocket=proxy,
    )


class ComputeSettingsTests(unittest.TestCase):
    def test_from_env_preserves_boolean_and_api_key_precedence(self):
        settings = compute_main.ComputeSettings.from_env(
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
        settings = compute_main.ComputeSettings()

        with self.assertRaises(FrozenInstanceError):
            settings.stt = "other"  # type: ignore[misc]


class ComputeApplicationFactoryTests(unittest.TestCase):
    def test_apps_keep_settings_and_dependencies_isolated(self):
        first = compute_main.create_app(
            compute_main.ComputeSettings(stt="first-stt", internal_ws_base_port=9101),
            _compute_dependencies("first-router"),
        )
        second = compute_main.create_app(
            compute_main.ComputeSettings(stt="second-stt", internal_ws_base_port=9102),
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


class LoadBalancerSettingsTests(unittest.TestCase):
    def test_from_env_preserves_preview_and_secret_fallbacks(self):
        settings = load_balancer_main.LoadBalancerSettings.from_env(
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
        settings = load_balancer_main.LoadBalancerSettings()

        with self.assertRaises(FrozenInstanceError):
            settings.dashboard_preview_mode = True  # type: ignore[misc]


class LoadBalancerApplicationFactoryTests(unittest.TestCase):
    def test_apps_can_be_created_with_distinct_preview_settings(self):
        first_settings = load_balancer_main.LoadBalancerSettings(
            compute_endpoint_names=("first-compute",),
            dashboard_preview_mode=True,
            request_usage_hash_secret="first-secret",
        )
        second_settings = load_balancer_main.LoadBalancerSettings(
            compute_endpoint_names=("second-compute",),
            dashboard_preview_mode=True,
            request_usage_hash_secret="second-secret",
        )

        first = load_balancer_main.create_app(first_settings)
        second = load_balancer_main.create_app(second_settings)

        first_root = TestClient(first).get("/").json()
        second_root = TestClient(second).get("/").json()
        first_health = TestClient(first).get("/health").json()

        self.assertEqual(first_root["compute_endpoints"], ["first-compute"])
        self.assertEqual(second_root["compute_endpoints"], ["second-compute"])
        self.assertTrue(first_health["dashboard_preview_mode"])
        self.assertIsNot(first.state.dependencies, second.state.dependencies)


if __name__ == "__main__":
    unittest.main()

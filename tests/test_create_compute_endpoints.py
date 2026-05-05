import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from create_compute_endpoints import build_endpoint_env, resolve_names_to_create  # noqa: E402


class FakeResponse:
    status_code = 404


class FakeNotFound(Exception):
    response = FakeResponse()


class FakeApi:
    def __init__(self, existing_names: list[str]):
        self.existing_names = set(existing_names)
        self.calls: list[str] = []

    def get_inference_endpoint(self, name: str, namespace: str | None = None):
        self.calls.append(name)
        if name not in self.existing_names:
            raise FakeNotFound(name)
        return object()


class CreateComputeEndpointsTests(unittest.TestCase):
    def test_target_total_creates_only_missing_tail(self):
        api = FakeApi(
            [
                "reachy-s2s-01",
                "reachy-s2s-02",
                "reachy-s2s-03",
                "reachy-s2s-04",
                "reachy-s2s-05",
                "reachy-s2s-06",
                "reachy-s2s-07",
                "reachy-s2s-08",
            ]
        )

        names = resolve_names_to_create(
            api=api,
            namespace="HuggingFaceM4",
            prefix="reachy-s2s",
            count=None,
            target_total=12,
            names=[],
        )

        self.assertEqual(
            names,
            ["reachy-s2s-09", "reachy-s2s-10", "reachy-s2s-11", "reachy-s2s-12"],
        )
        self.assertEqual(
            api.calls,
            [
                "reachy-s2s-01",
                "reachy-s2s-02",
                "reachy-s2s-03",
                "reachy-s2s-04",
                "reachy-s2s-05",
                "reachy-s2s-06",
                "reachy-s2s-07",
                "reachy-s2s-08",
                "reachy-s2s-09",
            ],
        )

    def test_target_total_returns_empty_when_pool_is_already_large_enough(self):
        api = FakeApi(["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"])

        names = resolve_names_to_create(
            api=api,
            namespace="HuggingFaceM4",
            prefix="reachy-s2s",
            count=None,
            target_total=3,
            names=[],
        )

        self.assertEqual(names, [])

    def test_build_endpoint_env_preserves_copied_values_without_overrides(self):
        env = build_endpoint_env(
            base_env={
                "SESSION_SHARED_SECRET": "shared",
                "PIPELINE_MAX_INSTANCES": "2",
                "PIPELINE_MIN_IDLE_INSTANCES": "2",
                "OPEN_API_MODEL_NAME": "template-model",
            },
            session_shared_secret=None,
            pipeline_max_instances=None,
            pipeline_min_idle_instances=None,
            lb_callback_auth_token=None,
        )

        self.assertEqual(
            env,
            {
                "SESSION_SHARED_SECRET": "shared",
                "PIPELINE_MAX_INSTANCES": "2",
                "PIPELINE_MIN_IDLE_INSTANCES": "2",
                "OPEN_API_MODEL_NAME": "template-model",
            },
        )

    def test_build_endpoint_env_requires_session_secret_when_not_copied(self):
        with self.assertRaisesRegex(ValueError, "--session-shared-secret is required"):
            build_endpoint_env(
                base_env={},
                session_shared_secret=None,
                pipeline_max_instances=None,
                pipeline_min_idle_instances=None,
                lb_callback_auth_token=None,
            )


if __name__ == "__main__":
    unittest.main()

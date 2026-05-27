import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from create_compute_endpoints import build_endpoint_env, main, resolve_names_to_create  # noqa: E402


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
                "OPEN_API_MODEL_NAME": "template-model",
            },
            session_shared_secret=None,
            num_pipelines=None,
            lb_callback_auth_token=None,
        )

        self.assertEqual(
            env,
            {
                "SESSION_SHARED_SECRET": "shared",
                "NUM_PIPELINES": "1",
                "OPEN_API_MODEL_NAME": "template-model",
            },
        )

    def test_build_endpoint_env_requires_session_secret_when_not_copied(self):
        with self.assertRaisesRegex(ValueError, "--session-shared-secret is required"):
            build_endpoint_env(
                base_env={},
                session_shared_secret=None,
                num_pipelines=None,
                lb_callback_auth_token=None,
            )


class CreateEndpointMainTests(unittest.TestCase):
    def test_main_passes_built_env_to_create_inference_endpoint(self):
        mock_endpoint = MagicMock()
        mock_endpoint.name = "test-endpoint"
        mock_endpoint.status = "pending"
        mock_endpoint.url = None

        mock_api = MagicMock()
        mock_api.create_inference_endpoint.return_value = mock_endpoint

        argv = [
            "create_compute_endpoints",
            "--names", "test-endpoint",
            "--vendor", "aws",
            "--region", "us-east-1",
            "--instance-size", "x1",
            "--instance-type", "nvidia-a10g",
            "--image-url", "registry/myimage:latest",
            "--num-pipelines", "4",
            "--session-shared-secret", "my-secret",
        ]

        with (
            patch("sys.argv", argv),
            patch("sys.stdout", io.StringIO()),
            patch("sys.stderr", io.StringIO()),
            patch("create_compute_endpoints.HfApi", return_value=mock_api),
        ):
            main()

        mock_api.create_inference_endpoint.assert_called_once()
        env = mock_api.create_inference_endpoint.call_args.kwargs["env"]
        self.assertEqual(env["NUM_PIPELINES"], "4")
        self.assertEqual(env["SESSION_SHARED_SECRET"], "my-secret")


if __name__ == "__main__":
    unittest.main()

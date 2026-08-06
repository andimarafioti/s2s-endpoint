import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.errors import InferenceEndpointError, InferenceEndpointTimeoutError

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _endpoint_helpers import (  # noqa: E402
    DEFAULT_LOAD_BALANCER_HEALTH_ROUTE,
    build_names,
    current_custom_image,
    current_model_env,
    expected_target_status,
    merge_env_updates,
    run_ordered_batch,
    wait_for_endpoint_update,
)


class FakeEndpoint:
    def __init__(self, name: str, status: str, fetch_statuses: list[str] | None = None):
        self.name = name
        self.status = status
        self.fetch_statuses = list(fetch_statuses or [])
        self.wait_calls: list[tuple[float | None, float]] = []
        self.fetch_count = 0

    def wait(self, timeout=None, refresh_every=None):
        self.wait_calls.append((timeout, refresh_every))
        self.status = "running"

    def fetch(self):
        self.fetch_count += 1
        if self.fetch_statuses:
            self.status = self.fetch_statuses.pop(0)
        return self


class EndpointHelpersTests(unittest.TestCase):
    def test_default_load_balancer_health_route_is_ready(self):
        self.assertEqual(DEFAULT_LOAD_BALANCER_HEALTH_ROUTE, "/ready")

    def test_build_names_expands_prefix_count(self):
        self.assertEqual(
            build_names("reachy-s2s", 3, []),
            ["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"],
        )

    def test_current_model_env_returns_empty_dict_when_missing(self):
        self.assertEqual(current_model_env({}), {})
        self.assertEqual(current_model_env({"model": {}}), {})

    def test_current_model_env_stringifies_values(self):
        raw = {"model": {"env": {"NUM_PIPELINES": 2, "DEBUG": True}}}
        self.assertEqual(
            current_model_env(raw),
            {
                "NUM_PIPELINES": "2",
                "DEBUG": "True",
            },
        )

    def test_merge_env_updates_overwrites_and_unsets_keys(self):
        current_env = {
            "RESPONSES_API_MODEL_NAME": "Qwen/Qwen3.5-9B:together",
            "SESSION_SHARED_SECRET": "secret",
            "OLD_FLAG": "1",
        }
        updated = merge_env_updates(
            current_env,
            updates={"RESPONSES_API_MODEL_NAME": "Qwen/Qwen3.5-72B:together"},
            unset_keys=["OLD_FLAG"],
        )
        self.assertEqual(
            updated,
            {
                "RESPONSES_API_MODEL_NAME": "Qwen/Qwen3.5-72B:together",
                "SESSION_SHARED_SECRET": "secret",
            },
        )

    def test_current_custom_image_preserves_url_health_route_and_port(self):
        raw = {
            "model": {
                "image": {
                    "custom": {
                        "url": "andito/s2s-compute:v0.3",
                        "healthRoute": "/healthz",
                        "port": 9000,
                    }
                }
            }
        }
        self.assertEqual(
            current_custom_image(raw),
            {
                "url": "andito/s2s-compute:v0.3",
                "health_route": "/healthz",
                "port": 9000,
            },
        )

    def test_expected_target_status_preserves_parked_endpoints(self):
        self.assertEqual(expected_target_status("paused"), "parked")
        self.assertEqual(expected_target_status("scaledToZero"), "parked")
        self.assertEqual(expected_target_status("running"), "running")

    def test_wait_for_running_endpoint_uses_hugging_face_wait(self):
        endpoint = FakeEndpoint("reachy-s2s-01", "updating")

        result = wait_for_endpoint_update(
            endpoint,
            target_status="running",
            timeout=60,
            refresh_every=5,
        )

        self.assertIs(result, endpoint)
        self.assertEqual(endpoint.wait_calls, [(60, 5)])
        self.assertEqual(endpoint.fetch_count, 1)

    def test_wait_for_parked_endpoint_accepts_any_parked_status(self):
        endpoint = FakeEndpoint("reachy-s2s-01", "updating", ["scaledToZero"])

        with patch("_endpoint_helpers.time.sleep"):
            result = wait_for_endpoint_update(
                endpoint,
                target_status="parked",
                timeout=60,
                refresh_every=5,
            )

        self.assertIs(result, endpoint)
        self.assertEqual(endpoint.status, "scaledToZero")
        self.assertEqual(endpoint.fetch_count, 2)

    def test_wait_for_endpoint_update_raises_on_failure_and_timeout(self):
        failed = FakeEndpoint("reachy-s2s-01", "updateFailed")
        with self.assertRaises(InferenceEndpointError):
            wait_for_endpoint_update(
                failed,
                target_status="parked",
                timeout=60,
                refresh_every=5,
            )

        timed_out = FakeEndpoint("reachy-s2s-02", "updating")
        with (
            patch("_endpoint_helpers.time.time", side_effect=[0, 2]),
            self.assertRaises(InferenceEndpointTimeoutError),
        ):
            wait_for_endpoint_update(
                timed_out,
                target_status="parked",
                timeout=1,
                refresh_every=5,
            )

    def test_run_ordered_batch_preserves_sequential_progress_and_order(self):
        names = ["reachy-s2s-02", "reachy-s2s-01"]
        calls: list[str] = []
        progress: list[str] = []

        def worker(name: str) -> dict[str, str]:
            calls.append(name)
            return {"name": name, "status": "done"}

        results = run_ordered_batch(
            names=names,
            worker=worker,
            parallelism=1,
            progress=progress.append,
            parallel_start_message="Parallel {total} with {max_workers}",
            sequential_start_message="[{index}/{total}] Starting {name}",
            parallel_submit_message="[{index}/{total}] Submitting {name}",
            completed_message=lambda result: result["status"],
        )

        self.assertEqual(calls, names)
        self.assertEqual([result["name"] for result in results], names)
        self.assertEqual(
            progress,
            [
                "[1/2] Starting reachy-s2s-02",
                "[1/2] reachy-s2s-02: done",
                "[2/2] Starting reachy-s2s-01",
                "[2/2] reachy-s2s-01: done",
            ],
        )

    def test_run_ordered_batch_returns_parallel_results_in_requested_order(self):
        names = ["reachy-s2s-03", "reachy-s2s-01", "reachy-s2s-02"]
        barrier = threading.Barrier(len(names))
        progress: list[str] = []

        def worker(name: str) -> str:
            barrier.wait(timeout=1)
            return name

        results = run_ordered_batch(
            names=names,
            worker=worker,
            parallelism=0,
            progress=progress.append,
            parallel_start_message="Parallel {total} with {max_workers}",
            sequential_start_message="[{index}/{total}] Starting {name}",
            parallel_submit_message="[{index}/{total}] Submitting {name}",
            completed_message=lambda result: f"completed {result}",
        )

        self.assertEqual(results, names)
        self.assertEqual(
            progress[:4],
            [
                "Parallel 3 with 3",
                "[1/3] Submitting reachy-s2s-03",
                "[2/3] Submitting reachy-s2s-01",
                "[3/3] Submitting reachy-s2s-02",
            ],
        )
        self.assertCountEqual(
            progress[4:],
            [
                "[1/3] reachy-s2s-03: completed reachy-s2s-03",
                "[2/3] reachy-s2s-01: completed reachy-s2s-01",
                "[3/3] reachy-s2s-02: completed reachy-s2s-02",
            ],
        )


if __name__ == "__main__":
    unittest.main()

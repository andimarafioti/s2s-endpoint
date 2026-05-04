import sys
import threading
import time
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from update_compute_endpoints_env import (  # noqa: E402
    expected_target_status,
    update_many,
    update_one,
    wait_for_endpoint_update,
)


class FakeEndpoint:
    def __init__(self, name: str, status: str, fetch_statuses: list[str]):
        self.name = name
        self.status = status
        self.fetch_statuses = list(fetch_statuses)
        self.wait_called = False
        self.fetch_count = 0

    def wait(self, timeout=None, refresh_every=None):
        self.wait_called = True
        self.status = "running"

    def fetch(self):
        self.fetch_count += 1
        if self.fetch_statuses:
            self.status = self.fetch_statuses.pop(0)
        return self


class ParallelTracker:
    def __init__(self):
        self.active_waiters = 0
        self.max_active_waiters = 0
        self.lock = threading.Lock()

    def wait_started(self):
        with self.lock:
            self.active_waiters += 1
            self.max_active_waiters = max(self.max_active_waiters, self.active_waiters)

    def wait_finished(self):
        with self.lock:
            self.active_waiters -= 1


class FakeManagedEndpoint:
    def __init__(self, name: str, env: dict[str, str], tracker: ParallelTracker):
        self.name = name
        self.status = "running"
        self.url = f"https://example.test/{name}"
        self._env = dict(env)
        self._tracker = tracker

    @property
    def raw(self):
        return {"model": {"env": self._env}}

    def wait(self, timeout=None, refresh_every=None):
        self._tracker.wait_started()
        try:
            time.sleep(0.05)
            self.status = "running"
        finally:
            self._tracker.wait_finished()

    def fetch(self):
        return self


class FakeParallelUpdateApi:
    def __init__(self, names: list[str]):
        self.tracker = ParallelTracker()
        self.endpoints = {
            name: FakeManagedEndpoint(name, {"OPEN_API_MODEL_NAME": "old-model"}, self.tracker) for name in names
        }

    def get_inference_endpoint(self, name: str, namespace: str | None = None):
        return self.endpoints[name]

    def update_inference_endpoint(self, name: str, namespace: str | None = None, env=None, secrets=None):
        endpoint = self.endpoints[name]
        endpoint.status = "updating"
        endpoint._env = dict(env or {})
        return endpoint


class UpdateComputeEndpointsEnvTests(unittest.TestCase):
    def test_expected_target_status_treats_paused_and_scaled_to_zero_as_parked(self):
        self.assertEqual(expected_target_status("paused"), "parked")
        self.assertEqual(expected_target_status("scaledToZero"), "parked")
        self.assertEqual(expected_target_status("running"), "running")

    def test_wait_for_endpoint_update_accepts_any_parked_status(self):
        endpoint = FakeEndpoint("reachy-s2s-01", "updating", ["scaledToZero"])

        wait_for_endpoint_update(
            endpoint,
            target_status="parked",
            timeout=1,
            refresh_every=0.001,
        )

        self.assertEqual(endpoint.status, "scaledToZero")
        self.assertFalse(endpoint.wait_called)
        self.assertGreaterEqual(endpoint.fetch_count, 1)

    def test_update_many_runs_compute_env_updates_in_parallel(self):
        names = ["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"]
        api = FakeParallelUpdateApi(names)

        results = update_many(
            api=api,
            namespace="HuggingFaceM4",
            names=names,
            env_updates={"OPEN_API_MODEL_NAME": "new-model"},
            unset_env=[],
            secret_updates={},
            wait=True,
            wait_timeout_s=60,
            wait_refresh_every_s=1,
            dry_run=False,
            parallelism=0,
        )

        self.assertEqual([result["name"] for result in results], names)
        self.assertGreaterEqual(api.tracker.max_active_waiters, 2)

    def test_update_one_can_replace_env_with_template_env(self):
        api = FakeParallelUpdateApi(["reachy-s2s-09"])
        api.endpoints["reachy-s2s-09"]._env = {
            "OPEN_API_MODEL_NAME": "wrong-model",
            "EXTRA_FLAG": "remove-me",
        }

        result = update_one(
            api=api,
            namespace="HuggingFaceM4",
            name="reachy-s2s-09",
            env_updates={
                "OPEN_API_MODEL_NAME": "template-model",
                "SESSION_SHARED_SECRET": "shared",
            },
            unset_env=[],
            secret_updates={},
            wait=False,
            wait_timeout_s=60,
            wait_refresh_every_s=1,
            dry_run=False,
            replace_env=True,
        )

        self.assertEqual(
            api.endpoints["reachy-s2s-09"]._env,
            {
                "OPEN_API_MODEL_NAME": "template-model",
                "SESSION_SHARED_SECRET": "shared",
            },
        )
        self.assertEqual(result["removed"], ["EXTRA_FLAG"])


if __name__ == "__main__":
    unittest.main()

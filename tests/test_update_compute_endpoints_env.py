import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from update_compute_endpoints_env import (  # noqa: E402
    expected_target_status,
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


if __name__ == "__main__":
    unittest.main()

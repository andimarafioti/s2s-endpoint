import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from update_load_balancer_endpoint_env import add_compute_pool_updates  # noqa: E402


class UpdateLoadBalancerEndpointEnvTests(unittest.TestCase):
    def test_add_compute_pool_updates_expands_prefix_count(self):
        env_updates: dict[str, str] = {}

        add_compute_pool_updates(
            env_updates,
            prefix="reachy-s2s",
            count=4,
            min_warm=3,
            wake_threshold_slots=3,
        )

        self.assertEqual(
            env_updates,
            {
                "COMPUTE_ENDPOINT_NAMES": "reachy-s2s-01,reachy-s2s-02,reachy-s2s-03,reachy-s2s-04",
                "COMPUTE_ENDPOINT_MIN_WARM": "3",
                "COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS": "3",
            },
        )


if __name__ == "__main__":
    unittest.main()

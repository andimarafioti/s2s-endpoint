import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from create_load_balancer_endpoint import resolve_compute_endpoint_names  # noqa: E402


class CreateLoadBalancerEndpointTests(unittest.TestCase):
    def test_resolve_compute_endpoint_names_accepts_prefix_count(self):
        self.assertEqual(
            resolve_compute_endpoint_names(
                names_csv=None,
                prefix="reachy-s2s",
                count=3,
            ),
            ["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"],
        )

    def test_resolve_compute_endpoint_names_accepts_csv_names(self):
        self.assertEqual(
            resolve_compute_endpoint_names(
                names_csv=" reachy-s2s-01,reachy-s2s-02 ",
                prefix=None,
                count=None,
            ),
            ["reachy-s2s-01", "reachy-s2s-02"],
        )


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import httpx
from huggingface_hub.errors import HfHubHTTPError


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from update_endpoints_images import (  # noqa: E402
    build_compute_summary,
    default_compute_prefix,
    discover_sequential_compute_names,
)


class FakeApi:
    def __init__(self, existing_names: list[str]):
        self.existing_names = set(existing_names)
        self.calls: list[str] = []

    def get_inference_endpoint(self, name: str, namespace: str | None = None):
        self.calls.append(name)
        if name in self.existing_names:
            return object()
        request = httpx.Request("GET", f"https://example.test/{namespace or 'default'}/{name}")
        response = httpx.Response(404, request=request)
        raise HfHubHTTPError("not found", response=response)


class UpdateEndpointImagesTests(unittest.TestCase):
    def test_default_compute_prefix_strips_lb_suffix(self):
        self.assertEqual(default_compute_prefix("reachy-s2s-lb"), "reachy-s2s")
        self.assertEqual(default_compute_prefix("custom-router"), "custom-router")

    def test_discover_sequential_compute_names_stops_on_first_missing_index(self):
        api = FakeApi(["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"])

        names, selection = discover_sequential_compute_names(
            api=api,
            namespace="HuggingFaceM4",
            prefix="reachy-s2s",
        )

        self.assertEqual(names, ["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"])
        self.assertEqual(
            api.calls,
            ["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03", "reachy-s2s-04"],
        )
        self.assertEqual(
            selection,
            {
                "discovery": "sequential_scan",
                "prefix": "reachy-s2s",
                "start_index": 1,
                "end_index": 3,
            },
        )
        self.assertEqual(build_compute_summary(names, selection), "updated endpoints 1 through 3")

    def test_discover_sequential_compute_names_raises_when_first_index_is_missing(self):
        api = FakeApi([])

        with self.assertRaisesRegex(ValueError, r"Expected to find 'reachy-s2s-01' first"):
            discover_sequential_compute_names(
                api=api,
                namespace="HuggingFaceM4",
                prefix="reachy-s2s",
            )


if __name__ == "__main__":
    unittest.main()

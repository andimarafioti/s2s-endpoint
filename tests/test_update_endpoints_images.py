import sys
import threading
import time
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
    update_one,
    update_many,
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
    def __init__(self, name: str, image_url: str, tracker: ParallelTracker):
        self.name = name
        self.status = "running"
        self.url = f"https://example.test/{name}"
        self._image_url = image_url
        self._tracker = tracker

    @property
    def raw(self):
        return {
            "model": {
                "image": {
                    "custom": {
                        "url": self._image_url,
                        "healthRoute": "/health",
                        "port": 7860,
                    }
                }
            }
        }

    def wait(self, timeout=None, refresh_every=None):
        self._tracker.wait_started()
        try:
            time.sleep(0.05)
            self.status = "running"
        finally:
            self._tracker.wait_finished()

    def fetch(self):
        return self


class FakeTransitionEndpoint(FakeManagedEndpoint):
    def __init__(self, name: str, image_url: str, tracker: ParallelTracker, status: str, fetch_statuses: list[str]):
        super().__init__(name, image_url, tracker)
        self.status = status
        self.fetch_statuses = list(fetch_statuses)
        self.wait_called = False
        self.fetch_count = 0

    def wait(self, timeout=None, refresh_every=None):
        self.wait_called = True
        super().wait(timeout=timeout, refresh_every=refresh_every)

    def fetch(self):
        self.fetch_count += 1
        if self.fetch_statuses:
            self.status = self.fetch_statuses.pop(0)
        return self


class FakeParallelUpdateApi:
    def __init__(self, names: list[str]):
        self.tracker = ParallelTracker()
        self.endpoints = {
            name: FakeManagedEndpoint(name, "andito/s2s-compute:old", self.tracker) for name in names
        }

    def get_inference_endpoint(self, name: str, namespace: str | None = None):
        return self.endpoints[name]

    def update_inference_endpoint(self, name: str, namespace: str | None = None, custom_image=None):
        endpoint = self.endpoints[name]
        endpoint.status = "updating"
        endpoint._image_url = custom_image["url"]
        return endpoint


class FakeTransitionUpdateApi:
    def __init__(self, endpoint: FakeTransitionEndpoint):
        self.endpoint = endpoint

    def get_inference_endpoint(self, name: str, namespace: str | None = None):
        return self.endpoint

    def update_inference_endpoint(self, name: str, namespace: str | None = None, custom_image=None):
        self.endpoint.status = "updating"
        self.endpoint._image_url = custom_image["url"]
        return self.endpoint


class FakeHealthRouteUpdateApi:
    def __init__(self):
        self.endpoint = FakeManagedEndpoint("reachy-s2s-lb", "andito/s2s-load_balancer:v1", ParallelTracker())
        self.last_custom_image = None

    def get_inference_endpoint(self, name: str, namespace: str | None = None):
        return self.endpoint

    def update_inference_endpoint(self, name: str, namespace: str | None = None, custom_image=None):
        self.last_custom_image = dict(custom_image or {})
        self.endpoint.status = "running"
        return self.endpoint


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

    def test_update_many_runs_compute_updates_in_parallel(self):
        names = ["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"]
        api = FakeParallelUpdateApi(names)

        results = update_many(
            api=api,
            namespace="HuggingFaceM4",
            names=names,
            image_url="andito/s2s-compute:new",
            wait=True,
            wait_timeout_s=60,
            wait_refresh_every_s=1,
            dry_run=False,
            parallelism=0,
        )

        self.assertEqual([result["name"] for result in results], names)
        self.assertGreaterEqual(api.tracker.max_active_waiters, 2)

    def test_update_one_waits_for_paused_endpoint_to_return_to_any_parked_state(self):
        tracker = ParallelTracker()
        endpoint = FakeTransitionEndpoint(
            "reachy-s2s-01",
            "andito/s2s-compute:old",
            tracker,
            status="paused",
            fetch_statuses=["scaledToZero"],
        )
        api = FakeTransitionUpdateApi(endpoint)

        result = update_one(
            api=api,
            namespace="HuggingFaceM4",
            name="reachy-s2s-01",
            image_url="andito/s2s-compute:new",
            wait=True,
            wait_timeout_s=1,
            wait_refresh_every_s=0.001,
            dry_run=False,
        )

        self.assertEqual(result["status_before"], "paused")
        self.assertEqual(result["expected_status_after"], "parked")
        self.assertEqual(result["status_after"], "scaledToZero")
        self.assertFalse(endpoint.wait_called)
        self.assertGreaterEqual(endpoint.fetch_count, 1)

    def test_update_one_applies_load_balancer_ready_health_route_even_when_image_matches(self):
        api = FakeHealthRouteUpdateApi()

        result = update_one(
            api=api,
            namespace="HuggingFaceM4",
            name="reachy-s2s-lb",
            image_url="andito/s2s-load_balancer:v1",
            health_route_override="/ready",
            wait=False,
            wait_timeout_s=1,
            wait_refresh_every_s=0.001,
            dry_run=False,
        )

        self.assertFalse(result["skipped"])
        self.assertEqual(result["health_route"], "/ready")
        self.assertEqual(api.last_custom_image["health_route"], "/ready")


if __name__ == "__main__":
    unittest.main()

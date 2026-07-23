"""The session creation response carries ``http_base_url`` — the HTTP origin of
the compute replica that owns the session — so LLM proxy clients can address
the right replica without deriving it from the websocket URL."""

import importlib
import sys
import unittest
from unittest.mock import patch

from app.app_utils import http_base_url_from_ws_url
from app.direct_session_manager import DirectSessionManager
from app.endpoint_pool_router import EndpointLease


class HttpBaseUrlFromWsUrlTests(unittest.TestCase):
    def test_wss_becomes_https_and_the_path_is_dropped(self):
        self.assertEqual(
            http_base_url_from_ws_url("wss://compute-01.example.endpoints.huggingface.cloud/v1/realtime"),
            "https://compute-01.example.endpoints.huggingface.cloud",
        )

    def test_ws_becomes_http_and_the_port_survives(self):
        self.assertEqual(
            http_base_url_from_ws_url("ws://127.0.0.1:9000/v1/realtime"),
            "http://127.0.0.1:9000",
        )


class GrantCarriesHttpBaseUrlTests(unittest.IsolatedAsyncioTestCase):
    async def test_grant_includes_http_base_url_for_the_leased_endpoint(self):
        manager = DirectSessionManager(
            endpoint_router=_SingleLeaseRouter(),
            session_shared_secret="shared-secret",
            queue_enabled=False,
        )

        allocation = await manager.allocate("https://lb.example")

        self.assertEqual(allocation["state"], "granted")
        self.assertEqual(
            allocation["http_base_url"],
            "https://compute-01.example.endpoints.huggingface.cloud",
        )


class PublicSessionAllocationTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("app.load_balancer_main", None)

    def test_http_base_url_survives_the_public_projection(self):
        module = self._import_load_balancer()

        public = module._public_session_allocation(
            {
                "state": "granted",
                "session_id": "sid",
                "websocket_url": "wss://compute-01.example/v1/realtime",
                "connect_url": "wss://compute-01.example/v1/realtime?session_token=tok",
                "session_token": "tok",
                "pending_timeout_s": 60.0,
                "http_base_url": "https://compute-01.example",
                "endpoint_name": "internal-only",
            }
        )

        self.assertEqual(public["http_base_url"], "https://compute-01.example")
        self.assertNotIn("endpoint_name", public)

    def _import_load_balancer(self):
        sys.modules.pop("app.load_balancer_main", None)
        with patch.dict(
            "os.environ",
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "DASHBOARD_BUCKET_ID": "",
                "DASHBOARD_PREVIEW_MODE": "",
                "SESSION_SHARED_SECRET": "",
            },
            clear=False,
        ):
            return importlib.import_module("app.load_balancer_main")


class _SingleLeaseRouter:
    LEASE = EndpointLease(
        slot_id="compute-01",
        endpoint_name="compute-01",
        ws_url="wss://compute-01.example.endpoints.huggingface.cloud/v1/realtime",
        waited_for_capacity=False,
    )

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def acquire(self, timeout_s: float = 900.0) -> EndpointLease:
        return self.LEASE

    async def release(self, slot_id, *, connected: bool = False) -> None:
        pass


if __name__ == "__main__":
    unittest.main()

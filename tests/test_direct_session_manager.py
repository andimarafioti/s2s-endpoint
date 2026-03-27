import asyncio
import unittest
from unittest.mock import patch

from app.direct_session_manager import DirectSessionManager
from app.endpoint_pool_router import EndpointLease
from app.session_tokens import verify_session_token, websocket_host_matches


class FakeLeaseRouter:
    def __init__(self):
        self.acquire_calls = 0
        self.release_calls = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def acquire(self, timeout_s: float = 900.0) -> EndpointLease:
        self.acquire_calls += 1
        endpoint_name = f"endpoint-{self.acquire_calls}"
        return EndpointLease(
            slot_id=endpoint_name,
            endpoint_name=endpoint_name,
            ws_url=f"wss://{endpoint_name}.example.endpoints.huggingface.cloud/ws",
        )

    async def release(self, slot_id: str) -> None:
        self.release_calls.append(slot_id)

    async def healthcheck(self):
        return True, None, {"running_endpoints": 1}


class DirectSessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        manager = getattr(self, "manager", None)
        if manager is not None:
            await manager.stop()

    async def test_allocate_then_connect_then_disconnect(self):
        router = FakeLeaseRouter()
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=60,
            session_token_ttl_s=3600,
            reap_interval_s=60,
        )
        await self.manager.start()

        allocation = await self.manager.allocate("https://lb.example")
        payload = verify_session_token(allocation["session_token"], "shared-secret")

        self.assertEqual(allocation["session_id"], payload["sid"])
        self.assertEqual(allocation["websocket_url"], payload["ws_url"])
        self.assertTrue(allocation["connect_url"].startswith(allocation["websocket_url"]))
        self.assertTrue(
            websocket_host_matches(payload["ws_url"], "endpoint-1.example.endpoints.huggingface.cloud")
        )

        with patch("app.direct_session_manager.time.monotonic", side_effect=[100.0, 105.0, 112.5]):
            connected = await self.manager.handle_event(
                allocation["session_id"],
                allocation["session_token"],
                "connected",
            )
            self.assertEqual(connected["state"], "connected")

            snapshot = await self.manager.snapshot()
            self.assertEqual(snapshot["pending_sessions"], 0)
            self.assertEqual(snapshot["connected_sessions"], 1)
            self.assertAlmostEqual(snapshot["sessions"][0]["connected_duration_s"], 5.0, places=3)

            released = await self.manager.handle_event(
                allocation["session_id"],
                allocation["session_token"],
                "disconnected",
            )

        self.assertEqual(released["state"], "released")
        self.assertTrue(released["conversation_counted"])
        self.assertAlmostEqual(released["conversation_duration_s"], 12.5, places=3)
        self.assertEqual(router.release_calls, ["endpoint-1"])

    async def test_pending_session_is_released_if_client_never_connects(self):
        router = FakeLeaseRouter()
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=0.01,
            session_token_ttl_s=3600,
            reap_interval_s=0.01,
        )
        await self.manager.start()

        allocation = await self.manager.allocate("https://lb.example")
        self.assertEqual(allocation["session_id"][:1] != "", True)

        await asyncio.sleep(0.05)

        snapshot = await self.manager.snapshot()
        self.assertEqual(snapshot["pending_sessions"], 0)
        self.assertEqual(snapshot["connected_sessions"], 0)
        self.assertEqual(router.release_calls, ["endpoint-1"])


if __name__ == "__main__":
    unittest.main()

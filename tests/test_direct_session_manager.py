import asyncio
import unittest
from unittest.mock import patch

from app.direct_session_manager import DirectSessionManager, QueueAtCapacityError
from app.endpoint_pool_router import EndpointLease
from app.session_tokens import verify_session_token, websocket_host_matches
from tests.helpers import monotonic_sequence


class FakeLeaseRouter:
    def __init__(
        self,
        health_snapshot: dict[str, object] | None = None,
        *,
        waited_for_capacity: bool = False,
    ):
        self.acquire_calls = 0
        self.release_calls = []
        self.release_connected_calls = []
        self.mark_connected_calls = []
        self.started = False
        self.stopped = False
        self.health_snapshot = health_snapshot or {"running_endpoints": 1}
        self.waited_for_capacity = waited_for_capacity

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
            waited_for_capacity=self.waited_for_capacity,
        )

    async def try_acquire(self) -> EndpointLease:
        # The queue owns the waiting now; the manager only ever asks for an
        # immediately-available slot. The fake always has one.
        return await self.acquire()

    async def mark_connected(self, slot_id: str) -> None:
        self.mark_connected_calls.append(slot_id)

    async def release(self, slot_id: str, *, connected: bool = False) -> None:
        self.release_calls.append(slot_id)
        self.release_connected_calls.append(connected)

    async def healthcheck(self):
        return True, None, dict(self.health_snapshot)


class ToggleCapacityRouter(FakeLeaseRouter):
    """Fake router whose free capacity can be flipped on/off to drive the queue."""

    def __init__(self, *, has_capacity: bool = False):
        super().__init__()
        self.has_capacity = has_capacity
        self.try_acquire_calls = 0

    async def try_acquire(self):
        self.try_acquire_calls += 1
        if not self.has_capacity:
            return None
        return await self.acquire()


class SessionQueueTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        manager = getattr(self, "manager", None)
        if manager is not None:
            await manager.stop()

    def _make(self, router, **kwargs):
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=60,
            session_token_ttl_s=3600,
            reap_interval_s=3600,
            queue_reap_interval_s=3600,  # reap only when we call it explicitly
            **kwargs,
        )
        return self.manager

    async def test_busy_pool_queues_and_head_claims_when_capacity_returns(self):
        router = ToggleCapacityRouter(has_capacity=False)
        manager = self._make(router, queue_max_depth=5)
        await manager.start()

        first = await manager.allocate("https://lb.example")
        second = await manager.allocate("https://lb.example")
        self.assertEqual(first["state"], "queued")
        self.assertEqual(first["position"], 1)
        self.assertEqual(second["position"], 2)
        self.assertEqual((await manager.snapshot())["queued_sessions"], 2)

        # Still no capacity: the head just gets its position back, no grant.
        again = await manager.poll(first["queue_id"], "https://lb.example")
        self.assertEqual(again["state"], "queued")
        self.assertEqual(again["position"], 1)

        # A non-head ticket never claims, even if capacity exists.
        router.has_capacity = True
        held = await manager.poll(second["queue_id"], "https://lb.example")
        self.assertEqual(held["state"], "queued")
        self.assertEqual(held["position"], 2)

        # The head claims the freed slot and leaves the queue.
        granted = await manager.poll(first["queue_id"], "https://lb.example")
        self.assertEqual(granted["state"], "granted")
        self.assertTrue(granted["waited_for_capacity"])
        self.assertTrue(granted["connect_url"].startswith(granted["websocket_url"]))

        snap = await manager.snapshot()
        self.assertEqual(snap["queued_sessions"], 1)  # only `second` remains
        self.assertEqual(snap["pending_sessions"], 1)  # the granted, not-yet-connected one

        # `second` is now the head and can claim.
        promoted = await manager.poll(second["queue_id"], "https://lb.example")
        self.assertEqual(promoted["state"], "granted")

    async def test_unknown_ticket_raises_keyerror(self):
        router = ToggleCapacityRouter(has_capacity=True)
        manager = self._make(router)
        await manager.start()
        with self.assertRaises(KeyError):
            await manager.poll("no-such-ticket", "https://lb.example")

    async def test_leave_removes_ticket(self):
        router = ToggleCapacityRouter(has_capacity=False)
        manager = self._make(router)
        await manager.start()

        ticket = await manager.allocate("https://lb.example")
        self.assertTrue(await manager.leave(ticket["queue_id"]))
        self.assertEqual((await manager.snapshot())["queued_sessions"], 0)
        self.assertFalse(await manager.leave(ticket["queue_id"]))  # idempotent
        with self.assertRaises(KeyError):
            await manager.poll(ticket["queue_id"], "https://lb.example")

    async def test_queue_at_capacity_raises(self):
        router = ToggleCapacityRouter(has_capacity=False)
        manager = self._make(router, queue_max_depth=2)
        await manager.start()

        await manager.allocate("https://lb.example")
        await manager.allocate("https://lb.example")
        with self.assertRaises(QueueAtCapacityError):
            await manager.allocate("https://lb.example")

    async def test_stale_ticket_is_reaped(self):
        router = ToggleCapacityRouter(has_capacity=False)
        manager = self._make(router, queue_ticket_ttl_s=0.0)
        await manager.start()

        ticket = await manager.allocate("https://lb.example")
        await manager._reap_stale_tickets()  # TTL 0 => immediately stale
        self.assertEqual((await manager.snapshot())["queued_sessions"], 0)
        with self.assertRaises(KeyError):
            await manager.poll(ticket["queue_id"], "https://lb.example")


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
        self.assertEqual(allocation["endpoint_name"], "endpoint-1")
        self.assertEqual(allocation["slot_id"], "endpoint-1")
        self.assertIsInstance(allocation["allocation_wait_ms"], int)
        self.assertFalse(allocation["waited_for_capacity"])
        self.assertTrue(allocation["connect_url"].startswith(allocation["websocket_url"]))
        self.assertTrue(
            websocket_host_matches(payload["ws_url"], "endpoint-1.example.endpoints.huggingface.cloud")
        )

        with patch("app.direct_session_manager.monotonic", new=monotonic_sequence(100.0, 105.0, 112.5)):
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
        self.assertEqual(router.mark_connected_calls, ["endpoint-1"])
        self.assertEqual(router.release_calls, ["endpoint-1"])
        self.assertEqual(router.release_connected_calls, [True])

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
        self.assertEqual(router.release_connected_calls, [False])

    async def test_healthcheck_counts_observed_router_sessions_without_pending(self):
        router = FakeLeaseRouter({"running_endpoints": 1, "active_sessions": 2})
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=60,
            session_token_ttl_s=3600,
            reap_interval_s=60,
        )
        await self.manager.start()

        await self.manager.allocate("https://lb.example")
        healthy, detail, snapshot = await self.manager.healthcheck()

        self.assertTrue(healthy)
        self.assertIsNone(detail)
        self.assertEqual(snapshot["pending_sessions"], 1)
        self.assertEqual(snapshot["connected_sessions"], 1)
        self.assertEqual(snapshot["router"], {"running_endpoints": 1, "active_sessions": 2})


    async def test_cancel_pending_session_releases_slot_immediately(self):
        router = FakeLeaseRouter()
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=3600,
            session_token_ttl_s=3600,
            reap_interval_s=3600,
        )
        await self.manager.start()

        allocation = await self.manager.allocate("https://lb.example")
        session_id = allocation["session_id"]

        self.assertEqual(router.release_calls, [])

        await self.manager.cancel_pending_session(session_id)

        self.assertEqual(router.release_calls, ["endpoint-1"])
        self.assertEqual(router.release_connected_calls, [False])

        snapshot = await self.manager.snapshot()
        self.assertEqual(snapshot["pending_sessions"], 0)

    async def test_cancel_pending_session_logs_endpoint_and_allocation_wait(self):
        router = FakeLeaseRouter(waited_for_capacity=True)
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=3600,
            session_token_ttl_s=3600,
            reap_interval_s=3600,
        )
        await self.manager.start()

        with patch("app.direct_session_manager.monotonic", new=monotonic_sequence(10.0, 11.25)):
            allocation = await self.manager.allocate("https://lb.example")

        with self.assertLogs("s2s-endpoint", level="INFO") as logs:
            await self.manager.cancel_pending_session(allocation["session_id"])

        record = logs.records[0]
        self.assertEqual(record.session_id, allocation["session_id"])
        self.assertEqual(record.endpoint_name, "endpoint-1")
        self.assertEqual(record.slot_id, "endpoint-1")
        self.assertEqual(record.allocation_wait_ms, 1250)
        self.assertEqual(record.outcome, "pending_released")
        self.assertTrue(record.waited_for_capacity)
        self.assertIn("endpoint endpoint-1", record.getMessage())
        self.assertIn("allocation_wait_ms=1250", record.getMessage())
        self.assertIn("waited_for_capacity=True", record.getMessage())

    async def test_cancel_pending_session_ignores_unknown_session_id(self):
        router = FakeLeaseRouter()
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=3600,
            session_token_ttl_s=3600,
            reap_interval_s=3600,
        )
        await self.manager.start()

        await self.manager.cancel_pending_session("nonexistent-session-id")

        self.assertEqual(router.release_calls, [])

    async def test_cancel_pending_session_ignores_already_connected_session(self):
        router = FakeLeaseRouter()
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=3600,
            session_token_ttl_s=3600,
            reap_interval_s=3600,
        )
        await self.manager.start()

        allocation = await self.manager.allocate("https://lb.example")
        await self.manager.handle_event(allocation["session_id"], allocation["session_token"], "connected")

        await self.manager.cancel_pending_session(allocation["session_id"])

        self.assertEqual(router.release_calls, [])
        snapshot = await self.manager.snapshot()
        self.assertEqual(snapshot["connected_sessions"], 1)

    async def test_allocate_releases_lease_if_interrupted_after_acquire(self):
        router = FakeLeaseRouter()
        self.manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret="shared-secret",
            pending_timeout_s=3600,
            session_token_ttl_s=3600,
            reap_interval_s=3600,
        )
        await self.manager.start()

        original_lock = self.manager._lock

        async def failing_lock_acquire():
            raise RuntimeError("simulated failure after acquire")

        class BrokenLock:
            async def __aenter__(self):
                raise RuntimeError("simulated failure after acquire")
            async def __aexit__(self, *args):
                pass

        self.manager._lock = BrokenLock()
        with self.assertRaises(RuntimeError):
            await self.manager.allocate("https://lb.example")
        self.manager._lock = original_lock

        self.assertEqual(router.release_calls, ["endpoint-1"])
        self.assertEqual(router.release_connected_calls, [False])


if __name__ == "__main__":
    unittest.main()

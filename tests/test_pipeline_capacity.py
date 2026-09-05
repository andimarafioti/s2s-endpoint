import asyncio
import unittest

import httpx

from app.speech_proxy_router import SpeechBackendPool, SpeechBackendPoolSettings, SpeechPoolCapacityExceeded
from app.speech_worker_lifecycle import SpeechWorkerLifecycle, WorkerLifecycleSettings
from tests.test_speech_proxy_router import _backends
from tests.test_speech_worker_lifecycle import Controller


class PoolSessionCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def make_pool(self, *, external=False):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
        self.addAsyncCleanup(client.aclose)
        pool = SpeechBackendPool(
            _backends(1 if external else 3),
            SpeechBackendPoolSettings(
                service="stt", target_work=2, max_work=4, latency_target=1,
                session_work=0.5, session_rpm=2, external=external,
                max_concurrency=4 if external else None, requests_per_minute=12 if external else None,
            ),
            client=client,
        )
        await pool.refresh_health()
        return pool

    async def test_pending_sessions_reduce_headroom_without_holding_inference_leases(self):
        pool = await self.make_pool()
        await pool.set_available("backend-2", False)
        await pool.set_available("backend-3", False)
        await pool.set_session_demand(2, reserve_sessions=5)
        snapshot = await pool.capacity_snapshot()
        self.assertEqual(snapshot["available_sessions"], 2)
        self.assertEqual(snapshot["admissible_sessions"], 6)
        self.assertEqual(snapshot["session_work"], 1)
        self.assertEqual(snapshot["reserve_work"], 2.5)
        self.assertEqual(sum(b.active_requests for b in await pool.snapshots()), 0)
        # Normal target exhaustion allows bounded degradation up to the hard limit.
        leases = [await pool.reserve(1) for _ in range(4)]
        with self.assertRaises(SpeechPoolCapacityExceeded):
            await pool.reserve(1)
        await asyncio.gather(*(lease.release(success=True, latency=0.1) for lease in leases))

    async def test_reserve_wakes_workers_without_advertising_them_as_ready(self):
        pool = await self.make_pool()
        controller = Controller(("running", "paused", "paused"))
        lifecycle = SpeechWorkerLifecycle(pool, controller, WorkerLifecycleSettings(max_workers=3))
        await lifecycle.start()
        self.addAsyncCleanup(lifecycle.stop)
        await pool.refresh_health()
        await pool.set_session_demand(2, reserve_sessions=5)
        await lifecycle.reconcile()
        await asyncio.gather(*lifecycle._operations.values())
        self.assertEqual(controller.calls, [("wake", "backend-2")])
        await lifecycle.reconcile()
        self.assertEqual(controller.calls, [("wake", "backend-2")])
        self.assertEqual((await pool.capacity_snapshot())["available_sessions"], 2)

    async def test_external_headroom_respects_shared_rpm_budget(self):
        pool = await self.make_pool(external=True)
        await pool.set_session_demand(5, reserve_sessions=5)
        self.assertEqual((await pool.capacity_snapshot())["admissible_sessions"], 1)
        await pool.rate_limited("10")
        self.assertEqual((await pool.capacity_snapshot())["admissible_sessions"], 0)

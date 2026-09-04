import asyncio
import unittest

import httpx

from app.endpoint_pool_router import EndpointSnapshot
from app.speech_proxy_app import SpeechProxySettings
from app.speech_proxy_router import NoSpeechBackendAvailable, SpeechBackendPool, SpeechBackendPoolSettings
from app.speech_worker_lifecycle import SpeechWorkerLifecycle, WorkerLifecycleSettings
from tests.test_speech_proxy_router import _backends


class Controller:
    def __init__(self, statuses):
        self.statuses = {f"backend-{i}": s for i, s in enumerate(statuses, 1)}
        self.calls = []
        self.fail_fetch = False
        self.fail_park = False

    def fetch(self, name):
        if self.fail_fetch:
            raise TimeoutError()
        status = self.statuses[name]
        return EndpointSnapshot(name, status, status, f"https://{name}.example")

    def begin_wake(self, name):
        self.calls.append(("wake", name))
        self.statuses[name] = "pending"
        return self.fetch(name)

    def force_restart(self, name):
        self.calls.append(("restart", name))
        self.statuses[name] = "pending"
        return self.fetch(name)

    def park(self, name):
        self.calls.append(("park", name))
        if self.fail_park:
            raise TimeoutError()
        self.statuses[name] = "paused"
        return self.fetch(name)

    def close(self):
        pass


class WorkerLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def make_fleet(self, statuses=("running", "paused", "paused"), **overrides):
        self.now = 1000.0
        self.probes = []

        async def handler(request):
            self.probes.append(request.url.host)
            return httpx.Response(200, content=b"audio")

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(self.client.aclose)
        self.pool = SpeechBackendPool(
            _backends(len(statuses)),
            SpeechBackendPoolSettings(service="tts", target_work=8, latency_target=0.5, tts_warmup_enabled=False),
            client=self.client,
        )
        self.controller = Controller(statuses)
        config = dict(max_workers=len(statuses), reconcile_interval_s=1000)
        config.update(overrides)
        self.lifecycle = SpeechWorkerLifecycle(
            self.pool, self.controller, WorkerLifecycleSettings(**config), time_fn=lambda: self.now
        )
        await self.lifecycle.start()
        self.addAsyncCleanup(self.lifecycle.stop)
        await self.pool.refresh_health()
        await self.tick()

    async def tick(self):
        await self.lifecycle.reconcile()
        await asyncio.gather(*self.lifecycle._operations.values())

    async def load(self, count):
        return await asyncio.gather(*(self.pool.reserve(1) for _ in range(count)))

    async def test_parked_workers_are_not_http_probed(self):
        await self.make_fleet()
        self.assertEqual(self.probes, ["backend-1.example"])
        self.assertEqual(self.controller.calls, [])

    async def test_wakes_at_soft_target_and_counts_pending_capacity(self):
        await self.make_fleet()
        leases = await self.load(7)
        await asyncio.gather(*(self.lifecycle.reconcile() for _ in range(8)))
        await self.tick()
        self.assertEqual(self.controller.calls, [("wake", "backend-2")])
        # An overloaded sole healthy worker still accepts new work.
        leases += await self.load(3)
        self.assertTrue(all(lease.backend_name == "backend-1" for lease in leases))
        self.now += 31
        await self.tick()
        self.assertEqual(len(self.controller.calls), 1)
        self.controller.statuses["backend-2"] = "running"
        await self.tick()
        self.assertFalse((await self.pool.snapshots())[1].ready)
        await self.pool.refresh_health()
        await self.tick()
        self.assertTrue((await self.pool.snapshots())[1].ready)
        self.assertEqual((await self.pool.reserve(1)).backend_name, "backend-2")

    async def test_short_stt_style_burst_is_not_lost_between_polls(self):
        await self.make_fleet()
        leases = await self.load(7)
        await asyncio.gather(*(lease.release(success=True, latency=0.1) for lease in leases))
        await self.tick()
        self.assertEqual(self.controller.calls, [("wake", "backend-2")])

    async def test_maximum_worker_count_bounds_wakes(self):
        await self.make_fleet(max_workers=2)
        await self.load(50)
        await self.tick()
        self.now += 31
        await self.tick()
        self.assertEqual(self.controller.calls, [("wake", "backend-2")])

    async def test_only_fresh_fleetwide_sustained_latency_scales_up(self):
        await self.make_fleet(("running", "running", "paused"))
        leases = await self.load(2)
        await leases[0].release(success=True, latency=2)
        await leases[1].release(success=True, latency=0.1)
        await self.load(1)
        await self.tick()
        self.now += 60
        await self.tick()
        self.assertEqual(self.controller.calls, [])
        self.pool._states["backend-2"].ewma_latency = 2
        await self.tick()
        self.now += 31
        await self.tick()
        self.assertEqual(self.controller.calls, [("wake", "backend-3")])

    async def test_idle_or_stale_slow_samples_do_not_wake_workers(self):
        await self.make_fleet()
        lease = (await self.load(1))[0]
        await lease.release(success=True, latency=2)
        self.pool._states["backend-1"].last_latency_at -= 100
        await self.load(1)
        await self.tick()
        self.now += 100
        await self.tick()
        self.assertEqual(self.controller.calls, [])

    async def test_idle_scale_down_preserves_warm_floor(self):
        await self.make_fleet(("running", "running", "running"))
        self.now += 1000
        for state in self.pool._states.values():
            state.last_used_at -= 1000
        self.pool._states["backend-1"].last_used_at -= 100
        await self.tick()
        self.assertEqual(self.controller.calls, [("park", "backend-1")])
        self.assertTrue((await self.pool.snapshots())[0].draining)
        self.assertNotEqual((await self.pool.reserve(1)).backend_name, "backend-1")
        await self.tick()
        self.assertEqual(len(self.controller.calls), 1)
        self.now += 601
        await self.tick()
        self.assertEqual(len(self.controller.calls), 2)

    async def test_low_continuous_load_eventually_consolidates(self):
        await self.make_fleet(("running", "running"), idle_timeout_s=600)
        for _ in range(11):
            lease = (await self.load(1))[0]
            await lease.release(success=True, latency=0.1)
            self.now += 60
            await self.tick()
        self.assertEqual(len([call for call in self.controller.calls if call[0] == "park"]), 1)

    async def test_burst_cancels_drain_before_remote_pause(self):
        await self.make_fleet(("running", "running"), idle_timeout_s=600)
        held = await self.load(2)
        self.now += 601
        await self.tick()
        draining = next(w for w in self.lifecycle._workers.values() if w.park_requested)
        self.assertEqual(self.controller.calls, [])
        await self.load(8)
        await self.tick()
        self.assertFalse(draining.park_requested)
        self.assertEqual(self.controller.calls, [])
        await asyncio.gather(*(lease.release(success=True) for lease in held))

    async def test_active_stream_blocks_parking_and_recovery(self):
        await self.make_fleet(("running", "running"))
        await self.load(2)
        self.now += 1000
        for state in self.pool._states.values():
            state.last_used_at -= 1000
        await self.tick()
        self.assertEqual(self.controller.calls, [])
        self.pool._states["backend-1"].ready = False
        await self.tick()
        self.now += 200
        await self.tick()
        self.assertEqual(self.controller.calls, [])

    async def test_failed_park_stays_quarantined_until_confirmed_stopped(self):
        await self.make_fleet(("running", "running"))
        self.now += 1000
        self.pool._states["backend-1"].last_used_at -= 1000
        self.controller.fail_park = True
        await self.tick()
        await self.tick()
        await self.pool.refresh_health()
        self.assertEqual((await self.pool.reserve(1)).backend_name, "backend-2")
        await self.load(10)
        await self.tick()
        self.assertTrue(self.lifecycle._workers["backend-1"].park_requested)
        self.assertFalse((await self.pool.snapshots())[0].ready)
        self.controller.fail_park = False
        self.now += 31
        await self.tick()
        await self.tick()
        self.assertEqual(self.controller.statuses["backend-1"], "paused")

    async def test_control_outage_keeps_healthy_routing_but_prevents_parking(self):
        await self.make_fleet(("running", "running"))
        self.controller.fail_fetch = True
        self.now += 1000
        for state in self.pool._states.values():
            state.last_used_at -= 1000
        await self.tick()
        self.assertEqual(self.controller.calls, [])
        self.assertIsNotNone(await self.pool.reserve(1))

    async def test_failed_workers_recover_with_bounded_attempts(self):
        await self.make_fleet(("running", "failed"), max_restart_attempts=1)
        self.assertEqual(self.controller.calls, [("restart", "backend-2")])
        self.controller.statuses["backend-2"] = "failed"
        self.now += 1000
        await self.tick()
        self.assertEqual(len(self.controller.calls), 1)

    async def test_late_health_result_cannot_revive_parked_worker(self):
        await self.make_fleet(("running",))
        started, complete = asyncio.Event(), asyncio.Event()

        async def handler(request):
            started.set()
            await complete.wait()
            return httpx.Response(200)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        self.pool._client = client
        task = asyncio.create_task(self.pool.refresh_health())
        await started.wait()
        await self.pool.set_available("backend-1", False)
        complete.set()
        await task
        with self.assertRaises(NoSpeechBackendAvailable):
            await self.pool.reserve(1)

    def test_autoscale_is_opt_in_and_requires_control_credentials(self):
        env = {"SPEECH_PROXY_SERVICE": "tts", "SPEECH_BACKENDS": "a=https://a.example"}
        self.assertIsNone(SpeechProxySettings.from_env(env).lifecycle)
        env["SPEECH_AUTOSCALE_ENABLED"] = "true"
        with self.assertRaisesRegex(ValueError, "HF_CONTROL_TOKEN"):
            SpeechProxySettings.from_env(env)
        env.update(HF_CONTROL_TOKEN="secret", HF_ENDPOINT_NAMESPACE="org", SPEECH_WORKER_IDLE_TIMEOUT_S="700")
        self.assertEqual(SpeechProxySettings.from_env(env).lifecycle.idle_timeout_s, 700)

    def test_managed_inventory_can_resolve_parked_urls_later(self):
        settings = SpeechProxySettings.from_env(
            {
                "SPEECH_PROXY_SERVICE": "stt",
                "SPEECH_BACKENDS": "worker-1,worker-2",
                "SPEECH_AUTOSCALE_ENABLED": "true",
                "HF_CONTROL_TOKEN": "control",
                "HF_ENDPOINT_NAMESPACE": "org",
            }
        )
        self.assertEqual([b.url for b in settings.backends], ["", ""])
        self.assertEqual(settings.lifecycle.max_workers, 2)
        with self.assertRaises(ValueError):
            SpeechProxySettings.from_env({"SPEECH_PROXY_SERVICE": "stt", "SPEECH_BACKENDS": "worker-1,worker-2"})

    async def test_url_change_requires_new_readiness_and_updates_routing(self):
        await self.make_fleet(("running",))
        await self.pool.set_available("backend-1", True, url="https://replacement.example")
        with self.assertRaises(NoSpeechBackendAvailable):
            await self.pool.reserve(1)
        await self.pool.refresh_health()
        self.assertEqual((await self.pool.reserve(1)).backend_url, "https://replacement.example")

    async def test_quarantine_is_atomic_with_reservations(self):
        await self.make_fleet(("running",))
        lease = await self.pool.reserve(1)
        self.assertFalse(await self.pool.quarantine_if_idle("backend-1"))
        await lease.release(success=True)
        self.assertTrue(await self.pool.quarantine_if_idle("backend-1"))
        with self.assertRaises(NoSpeechBackendAvailable):
            await self.pool.reserve(1)


if __name__ == "__main__":
    unittest.main()

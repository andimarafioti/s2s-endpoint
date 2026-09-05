import asyncio
import unittest
from unittest.mock import patch

import httpx

from app.direct_session_manager import DirectSessionManager
from app.endpoint_pool_router import ComputeUsage
from app.pipeline_capacity import PipelineCapacity, PipelineCapacityConfig
from app.session_tokens import verify_session_token
from app.speech_proxy_router import (
    NoSpeechBackendAvailable,
    SpeechBackendPool,
    SpeechBackendPoolSettings,
    SpeechPoolCapacityExceeded,
)
from app.speech_worker_lifecycle import SpeechWorkerLifecycle, WorkerLifecycleSettings
from tests.test_endpoint_pool_router import FakeEndpointController, _make_test_router
from tests.test_speech_proxy_router import _backends
from tests.test_speech_worker_lifecycle import Controller


class PoolSessionCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def make_pool(self, *, external=False):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
        self.addAsyncCleanup(client.aclose)
        pool = SpeechBackendPool(
            _backends(1 if external else 3),
            SpeechBackendPoolSettings(
                service="stt",
                target_work=2,
                max_work=None if external else 4,
                latency_target=1,
                session_work=0.5,
                session_rpm=2,
                external=external,
                max_concurrency=4 if external else None,
                requests_per_minute=12 if external else None,
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
        with self.assertRaises(NoSpeechBackendAvailable) as raised:
            await pool.reserve(1, exclude=frozenset({"backend-1"}))
        self.assertNotIsInstance(raised.exception, SpeechPoolCapacityExceeded)

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


class PipelineAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = 100.0
        self.requests = []
        self.broken = None
        config = PipelineCapacityConfig.model_validate(
            {
                "routes": {
                    "qwen": {"stt": "qwen", "llm": "shared", "tts": "shared"},
                    "openai": {"stt": "openai", "llm": "shared", "tts": "shared"},
                },
                "default": "qwen",
            }
        )

        def respond(request):
            import json

            service = request.url.host
            body = json.loads(request.content)
            self.requests.append((service, body, request.headers))
            if service == self.broken:
                return httpx.Response(503)
            return httpx.Response(
                200,
                json={
                    "pools": {
                        pool: {
                            "model": f"{service}-{pool}",
                            "provider": pool,
                            "profile": "small recorded workload",
                            "protocols": {"stt": ["transcriptions"], "llm": ["chat_completions"], "tts": ["speech"]}[
                                service
                            ],
                            "voices": ["alloy"] if service == "tts" else [],
                            "available_sessions": max(0, 5 - count),
                            "admissible_sessions": max(0, 7 - count),
                            "demand_stale": False,
                        }
                        for pool, count in body["session_counts"].items()
                    }
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(client.aclose)
        self.capacity = PipelineCapacity(
            config,
            {s: f"https://{s}" for s in ("stt", "llm", "tts")},
            "app-secret",
            ingress_api_key="ingress-secret",
            client=client,
            time_fn=lambda: self.now,
        )
        await self.capacity.refresh({})

    async def test_alternatives_share_demand_and_spend_headroom_immediately(self):
        self.assertEqual(self.capacity.snapshot({}, 10)["available_pipelines"], 5)
        sessions = {"qwen": 2, "openai": 3}
        self.assertEqual(
            self.capacity.pool_counts(sessions),
            {("stt", "qwen"): 2, ("stt", "openai"): 3, ("llm", "shared"): 5, ("tts", "shared"): 5},
        )
        self.assertEqual(self.capacity.snapshot(sessions, 10)["available_pipelines"], 0)
        self.assertTrue(self.capacity.can_admit("qwen", sessions))  # reserve is a target
        self.assertFalse(self.capacity.can_admit("openai", {"qwen": 4, "openai": 3}))
        await self.capacity.refresh(sessions)
        llm_request = [r for r in self.requests if r[0] == "llm"][-1]
        self.assertEqual(llm_request[1], {"session_counts": {"shared": 5}, "reserve_sessions": 5})
        self.assertEqual(llm_request[2]["authorization"], "Bearer ingress-secret")
        self.assertEqual(llm_request[2]["x-speech-capacity-authorization"], "Bearer app-secret")
        self.assertEqual(self.capacity.snapshot({}, 10)["available_pipelines"], 0)  # releases await a new observation

    async def test_unknown_surviving_sessions_and_stale_or_failed_gateway_are_conservative(self):
        self.assertFalse(self.capacity.can_admit("qwen", {"old-configuration": 7}))
        self.now += 15
        self.assertFalse(self.capacity.can_admit("qwen", {}))
        await self.capacity.refresh({})
        self.assertTrue(self.capacity.can_admit("qwen", {}))
        self.broken = "tts"
        await self.capacity.refresh({})
        self.assertFalse(self.capacity.can_admit("openai", {}))
        self.assertEqual(self.capacity.snapshot({}, 10)["routes"]["qwen"]["limiting_stage"], "tts")

    async def test_pending_grants_are_atomic_and_signed_routing_survives_queue(self):
        controller = FakeEndpointController([("cpu", "running", "https://cpu.example")])
        router = _make_test_router(
            endpoint_names=["cpu"],
            endpoint_slots=20,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=10,
            waking_capacity_timeout_s=60,
            park_cooldown_s=60,
            controller=controller,
            pipeline_capacity=self.capacity,
        )
        endpoint = router._endpoints["cpu"]
        endpoint.apply_snapshot(controller.fetch("cpu"))
        manager = DirectSessionManager(endpoint_router=router, session_shared_secret="secret", queue_enabled=True)
        self.addAsyncCleanup(manager.stop)
        allocations = await asyncio.gather(
            *(manager.allocate("https://allocator.example", pipeline="qwen" if i % 2 else "openai") for i in range(8))
        )
        granted = [a for a in allocations if a["state"] == "granted"]
        self.assertEqual(len(granted), 7)
        for allocation in granted:
            claims = verify_session_token(allocation["session_token"], "secret")
            self.assertEqual(claims["routing"], allocation["routing"])
            self.assertEqual(claims["routing"]["routes"]["llm"]["model"], "llm-shared")
        self.assertEqual(sum(router._pipeline_counts_unlocked().values()), 7)
        await manager.handle_event(granted[0]["session_id"], granted[0]["session_token"], "connected")
        self.assertEqual(sum(router._pipeline_counts_unlocked().values()), 7)
        await manager.cancel_pending_session(granted[1]["session_id"])
        self.assertEqual(sum(router._pipeline_counts_unlocked().values()), 6)
        queued = [a for a in allocations if a["state"] == "queued"][0]
        claimed = await manager.poll(queued["queue_id"], "https://allocator.example")
        self.assertEqual(claimed["state"], "granted")
        self.assertEqual(claimed["routing"]["pipeline"], "qwen")
        await manager.cancel_pending_session(claimed["session_id"])
        with patch.object(self.capacity, "routing", side_effect=ValueError("invalid handoff")):
            with self.assertRaises(ValueError):
                await manager.allocate("https://allocator.example")
        self.assertEqual(sum(router._pipeline_counts_unlocked().values()), 6)

    async def test_restart_counts_observed_routes_and_unclassified_connections(self):
        router = _make_test_router(
            endpoint_names=["cpu"],
            endpoint_slots=20,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=10,
            waking_capacity_timeout_s=60,
            park_cooldown_s=60,
            controller=FakeEndpointController([]),
            pipeline_capacity=self.capacity,
        )
        self.addAsyncCleanup(router.stop)
        endpoint = router._endpoints["cpu"]
        endpoint.apply_usage_sync(
            ComputeUsage(active_sessions=4, max_sessions=20, route_sessions={"qwen": 2}),
            synced_at=100,
            drain_generation=0,
        )
        counts = router._pipeline_counts_unlocked()
        self.assertEqual(counts, {"qwen": 2, "__unknown__": 2})
        self.assertEqual(self.capacity.pool_counts(counts)[("llm", "shared")], 4)

    async def test_cpu_idle_reserve_wakes_once_and_prevents_premature_consolidation(self):
        controller = FakeEndpointController(
            [("cpu1", "running", "https://cpu1.example"), ("cpu2", "paused", None), ("cpu3", "paused", None)]
        )
        router = _make_test_router(
            endpoint_names=list(controller.states),
            endpoint_slots=4,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=1,
            reconcile_interval_s=10,
            waking_capacity_timeout_s=60,
            park_cooldown_s=0,
            controller=controller,
            pipeline_capacity=self.capacity,
        )
        self.addAsyncCleanup(router.stop)
        for endpoint in router._endpoints.values():
            endpoint.apply_snapshot(controller.fetch(endpoint.name))
        self.assertEqual(router._mark_endpoints_to_wake_unlocked(), ["cpu2"])
        self.assertEqual(router._mark_endpoints_to_wake_unlocked(), [])
        self.assertEqual((await router.snapshot())["pipeline_capacity"]["available_pipelines"], 4)
        for name in ("cpu2", "cpu3"):
            endpoint = router._endpoints[name]
            endpoint.waking = False
            endpoint.status = "running"
            endpoint.url = f"https://{name}.example"
            endpoint.last_used_at = 0
            endpoint.last_usage_sync_at = __import__("time").monotonic()
        first = router._mark_endpoints_to_park_unlocked()
        self.assertEqual(len(first), 1)
        self.assertEqual(router._mark_endpoints_to_park_unlocked(), [])

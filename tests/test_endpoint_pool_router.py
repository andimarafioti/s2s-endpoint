import asyncio
import time
import unittest

from app.endpoint_pool_router import EndpointPoolRouter, EndpointSnapshot, _to_ws_url


class FakeEndpointController:
    def __init__(self, initial_states):
        self.states = {
            name: {"status": status, "url": url}
            for name, status, url in initial_states
        }
        self.wake_calls = []
        self.park_calls = []

    def fetch(self, name: str) -> EndpointSnapshot:
        state = self.states[name]
        return EndpointSnapshot(
            name=name,
            status=state["status"],
            raw_status=state["status"],
            url=state["url"],
        )

    def wake(self, name: str) -> EndpointSnapshot:
        self.wake_calls.append(name)
        self.states[name] = {
            "status": "running",
            "url": f"https://{name}.example.endpoints.huggingface.cloud",
        }
        return self.fetch(name)

    def park(self, name: str) -> EndpointSnapshot:
        self.park_calls.append(name)
        self.states[name] = {
            "status": "paused",
            "url": None,
        }
        return self.fetch(name)


class EndpointPoolRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        router = getattr(self, "router", None)
        if router is not None:
            await router.stop()

    async def test_start_ensures_minimum_warm_endpoints(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "paused", None),
                ("endpoint-b", "paused", None),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a", "endpoint-b"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        await self.router.start()

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 1)
        self.assertEqual(len(controller.wake_calls), 1)

    async def test_acquire_wakes_another_endpoint_when_free_slots_are_critical(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
                ("endpoint-b", "paused", None),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a", "endpoint-b"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        await self.router.start()
        lease = await self.router.acquire(timeout_s=0.2)
        self.assertEqual(lease.endpoint_name, "endpoint-a")

        await asyncio.sleep(0.05)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 2)
        self.assertIn("endpoint-b", controller.wake_calls)

    async def test_reconcile_parks_idle_endpoints_above_warm_floor(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
                ("endpoint-b", "running", "https://endpoint-b.example.endpoints.huggingface.cloud"),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a", "endpoint-b"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=0,
            reconcile_interval_s=0.05,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        await self.router.start()
        await asyncio.sleep(0.15)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 1)
        self.assertEqual(len(controller.park_calls), 1)

    async def test_reconcile_does_not_wake_endpoints_without_load_when_min_warm_is_zero(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "paused", None),
                ("endpoint-b", "paused", None),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a", "endpoint-b"],
            endpoint_slots=1,
            min_warm_endpoints=0,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=0.05,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        await self.router.start()
        await asyncio.sleep(0.15)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 0)
        self.assertEqual(controller.wake_calls, [])

    async def test_reconcile_does_not_wake_extra_endpoints_while_idle_at_warm_floor(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
                ("endpoint-b", "paused", None),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a", "endpoint-b"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=0.05,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        await self.router.start()
        await asyncio.sleep(0.15)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 1)
        self.assertEqual(controller.wake_calls, [])

    async def test_warming_endpoint_counts_as_capacity_until_timeout(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
                ("endpoint-b", "paused", None),
                ("endpoint-c", "paused", None),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a", "endpoint-b", "endpoint-c"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        now = time.monotonic()
        async with self.router._condition:
            endpoint_a = self.router._endpoints["endpoint-a"]
            endpoint_a.status = "running"
            endpoint_a.raw_status = "running"
            endpoint_a.url = "https://endpoint-a.example.endpoints.huggingface.cloud"
            endpoint_a.active_sessions = 1

            endpoint_b = self.router._endpoints["endpoint-b"]
            endpoint_b.status = "paused"
            endpoint_b.raw_status = "paused"
            endpoint_b.waking = True
            endpoint_b.wake_capacity_until = now + 300

            endpoint_c = self.router._endpoints["endpoint-c"]
            endpoint_c.status = "paused"
            endpoint_c.raw_status = "paused"

            self.assertEqual(self.router._mark_endpoints_to_wake_unlocked(), [])

            endpoint_b.wake_capacity_until = now - 1
            self.assertEqual(self.router._mark_endpoints_to_wake_unlocked(), ["endpoint-c"])

    async def test_reconcile_parks_one_endpoint_per_cycle_with_cooldown(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
                ("endpoint-b", "running", "https://endpoint-b.example.endpoints.huggingface.cloud"),
                ("endpoint-c", "running", "https://endpoint-c.example.endpoints.huggingface.cloud"),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a", "endpoint-b", "endpoint-c"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=0,
            reconcile_interval_s=0.05,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0.2,
            controller=controller,
        )

        await self.router.start()
        await asyncio.sleep(0.12)
        self.assertEqual(len(controller.park_calls), 1)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 2)
        self.assertGreater(snapshot["park_cooldown_remaining_s"], 0.0)

        await asyncio.sleep(0.25)
        self.assertEqual(len(controller.park_calls), 2)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 1)


class EndpointUrlTests(unittest.TestCase):
    def test_to_ws_url_uses_ws_route(self):
        self.assertEqual(
            _to_ws_url("https://abc.endpoints.huggingface.cloud"),
            "wss://abc.endpoints.huggingface.cloud/ws",
        )
        self.assertEqual(
            _to_ws_url("https://abc.endpoints.huggingface.cloud/base"),
            "wss://abc.endpoints.huggingface.cloud/base/ws",
        )


if __name__ == "__main__":
    unittest.main()

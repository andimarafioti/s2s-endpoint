import asyncio
import time
import unittest

from app.endpoint_pool_router import EndpointPoolRouter, EndpointSnapshot, ManagedEndpoint, _to_health_url, _to_ws_url


class FakeEndpointController:
    def __init__(self, initial_states):
        self.states = {
            name: {"status": status, "url": url}
            for name, status, url in initial_states
        }
        self.wake_calls = []
        self.park_calls = []
        self.force_restart_calls = []

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

    def restart(self, name: str) -> EndpointSnapshot:
        return self.wake(name)

    def force_restart(self, name: str) -> EndpointSnapshot:
        self.force_restart_calls.append(name)
        self.park(name)
        return self.wake(name)


class FakeComputeUsageFetcher:
    def __init__(self, busy_by_url):
        self.busy_by_url = dict(busy_by_url)
        self.calls = []

    def __call__(self, url: str) -> int:
        self.calls.append(url)
        return self.busy_by_url.get(url, 0)


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
        await asyncio.sleep(0.05)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 1)
        self.assertEqual(len(controller.wake_calls), 1)

    async def test_start_does_not_block_on_initial_warmup(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "paused", None),
            ]
        )

        original_wake = controller.wake

        def delayed_wake(name: str):
            time.sleep(0.1)
            return original_wake(name)

        controller.wake = delayed_wake
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        started = time.monotonic()
        await self.router.start()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.1)
        await asyncio.sleep(0.15)
        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 1)

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

    async def test_start_syncs_running_compute_usage(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
        usage_fetcher = FakeComputeUsageFetcher({endpoint_url: 1})
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", endpoint_url),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            compute_usage_fetcher=usage_fetcher,
        )

        await self.router.start()

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["active_sessions"], 1)
        self.assertEqual(snapshot["observed_active_sessions"], 1)
        self.assertEqual(snapshot["free_slots"], 0)
        self.assertEqual(snapshot["endpoints"][0]["active_sessions"], 1)
        self.assertEqual(snapshot["endpoints"][0]["observed_active_sessions"], 1)

    async def test_observed_usage_and_local_pending_do_not_oversubscribe_multi_slot_endpoint(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
        usage_fetcher = FakeComputeUsageFetcher({endpoint_url: 1})
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", endpoint_url),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=2,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            compute_usage_fetcher=usage_fetcher,
        )

        await self.router.start()
        lease = await self.router.acquire(timeout_s=0.2)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["active_sessions"], 2)
        self.assertEqual(snapshot["local_active_sessions"], 1)
        self.assertEqual(snapshot["local_pending_sessions"], 1)
        self.assertEqual(snapshot["observed_active_sessions"], 1)
        self.assertEqual(snapshot["free_slots"], 0)

        with self.assertRaisesRegex(RuntimeError, "timed out waiting"):
            await self.router.acquire(timeout_s=0.01)

        await self.router.release(lease.slot_id, connected=False)

    async def test_connected_local_session_frees_capacity_after_compute_usage_clears(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
        usage_fetcher = FakeComputeUsageFetcher({endpoint_url: 1})
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", endpoint_url),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=2,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            compute_usage_fetcher=usage_fetcher,
        )

        await self.router.start()
        lease = await self.router.acquire(timeout_s=0.2)
        await self.router.mark_connected(lease.slot_id)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["active_sessions"], 2)
        self.assertEqual(snapshot["local_pending_sessions"], 0)
        self.assertEqual(snapshot["unobserved_connected_sessions"], 1)
        self.assertEqual(snapshot["free_slots"], 0)

        usage_fetcher.busy_by_url[endpoint_url] = 2
        await self.router.refresh()
        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["active_sessions"], 2)
        self.assertEqual(snapshot["unobserved_connected_sessions"], 0)
        self.assertEqual(snapshot["free_slots"], 0)

        await self.router.release(lease.slot_id, connected=True)
        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["active_sessions"], 2)
        self.assertEqual(snapshot["local_active_sessions"], 0)
        self.assertEqual(snapshot["free_slots"], 0)

        usage_fetcher.busy_by_url[endpoint_url] = 1
        await self.router.refresh()
        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["active_sessions"], 1)
        self.assertEqual(snapshot["observed_active_sessions"], 1)
        self.assertEqual(snapshot["free_slots"], 1)

    async def test_refresh_marks_compute_available_after_observed_usage_clears(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
        usage_fetcher = FakeComputeUsageFetcher({endpoint_url: 1})
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", endpoint_url),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            compute_usage_fetcher=usage_fetcher,
        )

        await self.router.start()
        usage_fetcher.busy_by_url[endpoint_url] = 0
        await self.router.refresh()

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["active_sessions"], 0)
        self.assertEqual(snapshot["observed_active_sessions"], 0)
        self.assertEqual(snapshot["free_slots"], 1)

    async def test_reconcile_does_not_park_observed_busy_endpoint(self):
        endpoint_a_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
        endpoint_b_url = "https://endpoint-b.example.endpoints.huggingface.cloud"
        usage_fetcher = FakeComputeUsageFetcher({endpoint_a_url: 1, endpoint_b_url: 0})
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", endpoint_a_url),
                ("endpoint-b", "running", endpoint_b_url),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a", "endpoint-b"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=0,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            compute_usage_fetcher=usage_fetcher,
        )

        await self.router.start()
        await self.router._schedule_parks_if_needed()
        await asyncio.sleep(0.05)

        self.assertEqual(controller.park_calls, ["endpoint-b"])
        snapshot = await self.router.snapshot()
        endpoints = {endpoint["name"]: endpoint for endpoint in snapshot["endpoints"]}
        self.assertEqual(endpoints["endpoint-a"]["active_sessions"], 1)
        self.assertEqual(endpoints["endpoint-a"]["observed_active_sessions"], 1)

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

        self.router._fetch_pool_units = lambda url: []
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


class ManagedEndpointPropertyTests(unittest.TestCase):
    def test_drain_restarting_zeroes_free_slots(self):
        ep = ManagedEndpoint(name="ep", slots=2)
        ep.status = "running"
        ep.url = "https://ep.example.endpoints.huggingface.cloud"

        self.assertEqual(ep.free_slots, 2)

        ep.drain_restarting = True
        self.assertEqual(ep.free_slots, 0)


def _make_drain_router(controller, *, drain_restart_timeout_s=600, endpoint_slots=1, **extra):
    defaults = dict(
        endpoint_slots=endpoint_slots,
        min_warm_endpoints=1,
        wake_threshold_slots=0,
        idle_park_timeout_s=60,
        reconcile_interval_s=60,
        waking_capacity_timeout_s=300,
        park_cooldown_s=0,
        drain_restart_timeout_s=drain_restart_timeout_s,
        controller=controller,
    )
    defaults.update(extra)
    return EndpointPoolRouter(**defaults)


class DrainRestartTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        router = getattr(self, "router", None)
        if router is not None:
            await router.stop()

    async def test_drain_restart_triggers_when_unit_stuck_above_threshold(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = _make_drain_router(
            controller,
            endpoint_names=["endpoint-a"],
            drain_restart_timeout_s=600,
        )
        self.router._fetch_pool_units = lambda url: [
            {"state": "active"},
            {"state": "draining", "draining_for_s": 700},
        ]

        await self.router.start()
        await self.router._check_drain_restarts()
        await asyncio.sleep(0.05)

        self.assertIn("endpoint-a", controller.force_restart_calls)

    async def test_drain_restart_not_triggered_when_all_units_draining_below_threshold(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = _make_drain_router(
            controller,
            endpoint_names=["endpoint-a"],
            drain_restart_timeout_s=600,
        )
        # All units draining but well below the 600s threshold — normal session cleanup,
        # should NOT trigger a force restart.
        self.router._fetch_pool_units = lambda url: [
            {"state": "draining", "draining_for_s": 5},
            {"state": "draining", "draining_for_s": 3},
        ]

        await self.router.start()
        await self.router._check_drain_restarts()
        await asyncio.sleep(0.05)

        self.assertEqual(controller.force_restart_calls, [])

    async def test_drain_restart_triggers_when_all_units_stuck_above_threshold(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = _make_drain_router(
            controller,
            endpoint_names=["endpoint-a"],
            drain_restart_timeout_s=600,
        )
        # All units stuck draining above threshold — wedged pool, should trigger force restart.
        self.router._fetch_pool_units = lambda url: [
            {"state": "draining", "draining_for_s": 700},
            {"state": "draining", "draining_for_s": 650},
        ]

        await self.router.start()
        await self.router._check_drain_restarts()
        await asyncio.sleep(0.05)

        self.assertIn("endpoint-a", controller.force_restart_calls)

    async def test_drain_restart_not_triggered_below_threshold(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = _make_drain_router(
            controller,
            endpoint_names=["endpoint-a"],
            drain_restart_timeout_s=600,
        )
        self.router._fetch_pool_units = lambda url: [
            {"state": "active"},
            {"state": "draining", "draining_for_s": 100},
        ]

        await self.router.start()
        await self.router._check_drain_restarts()
        await asyncio.sleep(0.05)

        self.assertEqual(controller.force_restart_calls, [])

    async def test_drain_restart_skips_already_drain_restarting_endpoint(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = _make_drain_router(
            controller,
            endpoint_names=["endpoint-a"],
            drain_restart_timeout_s=0,
        )
        self.router._fetch_pool_units = lambda url: [{"state": "draining", "draining_for_s": 999}]

        await self.router.start()
        async with self.router._condition:
            self.router._endpoints["endpoint-a"].drain_restarting = True

        await self.router._check_drain_restarts()
        await asyncio.sleep(0.05)

        self.assertEqual(controller.force_restart_calls, [])

    async def test_drain_restarting_cleared_after_force_restart_completes(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = _make_drain_router(
            controller,
            endpoint_names=["endpoint-a"],
            drain_restart_timeout_s=600,
        )
        self.router._fetch_pool_units = lambda url: [
            {"state": "active"},
            {"state": "draining", "draining_for_s": 700},
        ]

        await self.router.start()
        await self.router._check_drain_restarts()
        await asyncio.sleep(0.05)

        async with self.router._condition:
            ep = self.router._endpoints["endpoint-a"]
        self.assertFalse(ep.drain_restarting)


class EndpointUrlTests(unittest.TestCase):
    def test_to_ws_url_uses_realtime_route(self):
        self.assertEqual(
            _to_ws_url("https://abc.endpoints.huggingface.cloud"),
            "wss://abc.endpoints.huggingface.cloud/v1/realtime",
        )
        self.assertEqual(
            _to_ws_url("https://abc.endpoints.huggingface.cloud/base"),
            "wss://abc.endpoints.huggingface.cloud/base/v1/realtime",
        )

    def test_to_ws_url_supports_custom_route(self):
        self.assertEqual(
            _to_ws_url("https://abc.endpoints.huggingface.cloud", "/custom"),
            "wss://abc.endpoints.huggingface.cloud/custom",
        )

    def test_to_health_url_uses_health_route(self):
        self.assertEqual(
            _to_health_url("https://abc.endpoints.huggingface.cloud"),
            "https://abc.endpoints.huggingface.cloud/health",
        )
        self.assertEqual(
            _to_health_url("https://abc.endpoints.huggingface.cloud/base"),
            "https://abc.endpoints.huggingface.cloud/base/health",
        )


if __name__ == "__main__":
    unittest.main()

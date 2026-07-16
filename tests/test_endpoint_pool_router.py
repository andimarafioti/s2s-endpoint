import asyncio
import json
import time
import unittest
from unittest.mock import patch

from app.endpoint_pool_router import (
    ComputeUsageSchemaError,
    EndpointDrainLeaseConflictError,
    EndpointPoolRouter,
    EndpointSnapshot,
    EndpointTransitionConflictError,
    ManagedEndpoint,
    drain_lease_owner_fingerprint,
    _to_health_url,
    _to_ws_url,
    fetch_compute_active_sessions,
)


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


class FakeUrlopenResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class EndpointPoolRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        router = getattr(self, "router", None)
        if router is not None:
            await router.stop()

    def test_fetch_compute_active_sessions_reads_active_sessions_from_health(self):
        with patch(
            "app.endpoint_pool_router.urllib.request.urlopen",
            return_value=FakeUrlopenResponse({"router": {"active_sessions": 2}}),
        ):
            active_sessions = fetch_compute_active_sessions("https://compute.example")

        self.assertEqual(active_sessions, 2)

    def test_fetch_compute_active_sessions_supports_legacy_ready_busy(self):
        with patch(
            "app.endpoint_pool_router.urllib.request.urlopen",
            return_value=FakeUrlopenResponse({"router": {"ready_busy": 1}}),
        ):
            active_sessions = fetch_compute_active_sessions("https://compute.example")

        self.assertEqual(active_sessions, 1)

    def test_fetch_compute_active_sessions_raises_on_missing_session_count(self):
        # Regression guard for the 2026-06-07 incident: a schema drift in the
        # compute health payload must fail loudly instead of silently reading
        # 0 sessions and letting the LB treat busy nodes as free.
        with patch(
            "app.endpoint_pool_router.urllib.request.urlopen",
            return_value=FakeUrlopenResponse({"router": {"ready": True, "starting": False}}),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not include a session count"):
                fetch_compute_active_sessions("https://compute.example")

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
        self.assertFalse(lease.waited_for_capacity)

        await asyncio.sleep(0.05)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 2)
        self.assertIn("endpoint-b", controller.wake_calls)

    async def test_draining_endpoint_is_not_selected_for_new_sessions(self):
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
            wake_threshold_slots=1,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        await self.router.start()
        await self.router.set_draining("endpoint-a", True, lease_id="rollout-a")

        lease = await self.router.acquire(timeout_s=0.2)
        snapshot = await self.router.snapshot()
        endpoint_a = next(endpoint for endpoint in snapshot["endpoints"] if endpoint["name"] == "endpoint-a")

        self.assertEqual(lease.endpoint_name, "endpoint-b")
        self.assertTrue(endpoint_a["draining"])
        self.assertEqual(endpoint_a["free_slots"], 0)

    async def test_set_draining_rejects_existing_drain_restart(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )
        await self.router.start()
        async with self.router._condition:
            self.router._endpoints["endpoint-a"].drain_restarting = True

        with self.assertRaisesRegex(EndpointTransitionConflictError, "drain_restarting"):
            await self.router.set_draining("endpoint-a", True, lease_id="rollout-a")

        snapshot = await self.router.snapshot()
        self.assertFalse(snapshot["endpoints"][0]["draining"])

    async def test_set_draining_rejects_existing_parking_transition(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )
        await self.router.start()
        async with self.router._condition:
            self.router._endpoints["endpoint-a"].parking = True

        with self.assertRaisesRegex(EndpointTransitionConflictError, "parking"):
            await self.router.set_draining("endpoint-a", True, lease_id="rollout-a")

        snapshot = await self.router.snapshot()
        self.assertFalse(snapshot["endpoints"][0]["draining"])

    async def test_drain_lease_owner_fences_other_rollout_clients(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )
        await self.router.start()

        await self.router.set_draining(
            "endpoint-a",
            True,
            lease_ttl_s=60,
            lease_id="rollout-a-secret",
        )
        await self.router.set_draining(
            "endpoint-a",
            True,
            lease_ttl_s=120,
            lease_id="rollout-a-secret",
        )

        with self.assertRaisesRegex(EndpointDrainLeaseConflictError, "another rollout"):
            await self.router.set_draining(
                "endpoint-a",
                True,
                lease_ttl_s=120,
                lease_id="rollout-b-secret",
            )
        with self.assertRaisesRegex(EndpointDrainLeaseConflictError, "another rollout"):
            await self.router.set_draining(
                "endpoint-a",
                False,
                lease_id="rollout-b-secret",
            )

        snapshot = await self.router.snapshot()
        endpoint = snapshot["endpoints"][0]
        self.assertTrue(endpoint["draining"])
        self.assertEqual(
            endpoint["drain_lease_owner"],
            drain_lease_owner_fingerprint("rollout-a-secret"),
        )
        self.assertNotIn("rollout-a-secret", str(endpoint))

        await self.router.set_draining(
            "endpoint-a",
            False,
            lease_id="rollout-a-secret",
        )
        snapshot = await self.router.snapshot()
        self.assertFalse(snapshot["endpoints"][0]["draining"])

    async def test_drain_requires_usage_request_started_after_acquisition(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
        controller = FakeEndpointController(
            [("endpoint-a", "running", endpoint_url)]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            compute_usage_fetcher=lambda url: 0,
        )
        await self.router.start()

        async def acquire_drain_after_request_started(function, *args):
            await self.router.set_draining("endpoint-a", True, lease_id="rollout-a")
            return function(*args)

        # _sync_compute_usage captures the current drain generation before it
        # starts the request. Acquiring the drain while that request is in
        # flight must leave its result ineligible for rollout safety.
        with patch(
            "app.endpoint_pool_router.asyncio.to_thread",
            new=acquire_drain_after_request_started,
        ):
            await self.router._sync_compute_usage()

        snapshot = await self.router.snapshot()
        endpoint = snapshot["endpoints"][0]
        self.assertTrue(endpoint["usage_synced"])
        self.assertFalse(endpoint["usage_synced_after_drain"])

        await self.router._sync_compute_usage()
        snapshot = await self.router.snapshot()
        self.assertTrue(snapshot["endpoints"][0]["usage_synced_after_drain"])

        # Retrying an idempotent drain request after a lost response must not
        # reset the generation and force another usage synchronization.
        async with self.router._condition:
            managed_endpoint = self.router._endpoints["endpoint-a"]
            drain_generation = managed_endpoint.drain_generation
            draining_since = managed_endpoint.draining_since
            managed_endpoint.drain_expires_at = time.monotonic() + 1
            previous_expiry = managed_endpoint.drain_expires_at
        await self.router.set_draining(
            "endpoint-a",
            True,
            lease_ttl_s=60,
            lease_id="rollout-a",
        )
        snapshot = await self.router.snapshot()
        self.assertTrue(snapshot["endpoints"][0]["usage_synced_after_drain"])
        async with self.router._condition:
            managed_endpoint = self.router._endpoints["endpoint-a"]
            self.assertEqual(managed_endpoint.drain_generation, drain_generation)
            self.assertEqual(managed_endpoint.draining_since, draining_since)
            self.assertGreater(managed_endpoint.drain_expires_at, previous_expiry)

        await self.router.set_draining("endpoint-a", False, lease_id="rollout-a")
        snapshot = await self.router.snapshot()
        self.assertFalse(snapshot["endpoints"][0]["usage_synced_after_drain"])
        self.assertIsNone(snapshot["endpoints"][0]["drain_lease_remaining_s"])

    async def test_expired_drain_lease_clears_automatically(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )
        await self.router.start()
        await self.router.set_draining(
            "endpoint-a",
            True,
            lease_ttl_s=60,
            lease_id="rollout-a",
        )
        async with self.router._condition:
            endpoint = self.router._endpoints["endpoint-a"]
            endpoint.draining_since = time.monotonic() - 120
            endpoint.drain_expires_at = time.monotonic() - 1

        with self.assertLogs("s2s-endpoint", level="ERROR") as logs:
            await self.router._maintain_drain_leases()

        snapshot = await self.router.snapshot()
        self.assertFalse(snapshot["endpoints"][0]["draining"])
        self.assertTrue(any("lease expired" in line for line in logs.output))

    async def test_long_running_drain_logs_recurring_warning(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            drain_warning_after_s=60,
            drain_warning_interval_s=300,
        )
        await self.router.start()
        await self.router.set_draining(
            "endpoint-a",
            True,
            lease_ttl_s=3600,
            lease_id="rollout-a",
        )
        async with self.router._condition:
            endpoint = self.router._endpoints["endpoint-a"]
            endpoint.draining_since = time.monotonic() - 120

        with self.assertLogs("s2s-endpoint", level="WARNING") as first_warning:
            await self.router._maintain_drain_leases()
        self.assertTrue(any("allocator-drained" in line for line in first_warning.output))

        async with self.router._condition:
            self.router._endpoints["endpoint-a"].last_drain_warning_at = (
                time.monotonic() - 301
            )
        with self.assertLogs("s2s-endpoint", level="WARNING") as recurring_warning:
            await self.router._maintain_drain_leases()
        self.assertTrue(any("allocator-drained" in line for line in recurring_warning.output))

    async def test_acquire_marks_lease_when_it_waited_for_capacity(self):
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=1,
            min_warm_endpoints=1,
            wake_threshold_slots=0,
            idle_park_timeout_s=60,
            reconcile_interval_s=60,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
        )

        await self.router.start()
        first_lease = await self.router.acquire(timeout_s=0.2)
        second_acquire = asyncio.create_task(self.router.acquire(timeout_s=0.2))
        await asyncio.sleep(0.01)

        await self.router.release(first_lease.slot_id, connected=False)
        second_lease = await second_acquire

        self.assertEqual(second_lease.endpoint_name, "endpoint-a")
        self.assertTrue(second_lease.waited_for_capacity)

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

    async def test_refresh_after_idle_park_with_no_sessions_does_not_report_endpoint_down(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
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
        )
        downed_endpoints = []

        async def record_endpoint_down(endpoint_name: str) -> None:
            downed_endpoints.append(endpoint_name)

        await self.router.start()
        self.router._on_endpoint_down = record_endpoint_down

        async with self.router._condition:
            self.router._endpoints["endpoint-a"].parking = True

        controller.states["endpoint-a"] = {"status": "paused", "url": None}
        await self.router.refresh()

        self.assertEqual(downed_endpoints, [])
        snapshot = await self.router.snapshot()
        endpoint = snapshot["endpoints"][0]
        self.assertEqual(snapshot["active_sessions"], 0)
        self.assertFalse(endpoint["parking"])

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

    async def test_running_endpoint_offers_no_capacity_until_usage_synced(self):
        # An LB restart must not treat a running compute node as free before
        # its true session count has been observed at least once.
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"

        class AlwaysFailingFetcher:
            def __init__(self):
                self.calls = 0

            def __call__(self, url: str) -> int:
                self.calls += 1
                raise RuntimeError("health unreachable")

        usage_fetcher = AlwaysFailingFetcher()
        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", endpoint_url),
            ]
        )
        self.router = EndpointPoolRouter(
            endpoint_names=["endpoint-a"],
            endpoint_slots=4,
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
        self.assertEqual(snapshot["running_endpoints"], 1)
        self.assertEqual(snapshot["free_slots"], 0)
        self.assertFalse(snapshot["endpoints"][0]["usage_synced"])
        # Initial sync plus the startup retry.
        self.assertEqual(usage_fetcher.calls, 2)

        with self.assertRaisesRegex(RuntimeError, "timed out waiting"):
            await self.router.acquire(timeout_s=0.05)

        # Health must not report green while allocation is impossible for
        # lack of usage knowledge (as opposed to known-full, which is fine).
        healthy, detail, health_snapshot = await self.router.healthcheck()
        self.assertFalse(healthy)
        self.assertIn("have not synced usage", detail)
        self.assertEqual(health_snapshot["unsynced_running_endpoints"], 1)

    async def test_capacity_unlocks_once_usage_sync_succeeds(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"

        class FlakyFetcher:
            def __init__(self, fail_times: int, value: int):
                self.remaining_failures = fail_times
                self.value = value
                self.calls = 0

            def __call__(self, url: str) -> int:
                self.calls += 1
                if self.remaining_failures > 0:
                    self.remaining_failures -= 1
                    raise RuntimeError("health unreachable")
                return self.value

        usage_fetcher = FlakyFetcher(fail_times=2, value=1)
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
        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["free_slots"], 0)

        await self.router.refresh()

        snapshot = await self.router.snapshot()
        self.assertTrue(snapshot["endpoints"][0]["usage_synced"])
        self.assertEqual(snapshot["observed_active_sessions"], 1)
        self.assertEqual(snapshot["free_slots"], 1)

        lease = await self.router.acquire(timeout_s=0.2)
        self.assertEqual(lease.endpoint_name, "endpoint-a")
        await self.router.release(lease.slot_id, connected=False)

        healthy, detail, _ = await self.router.healthcheck()
        self.assertTrue(healthy)
        self.assertIsNone(detail)

    async def test_acquire_wakes_parked_node_when_running_nodes_are_unsynced(self):
        # Regression: acquire() used to mark parked endpoints as waking, then
        # suspend on the condition BEFORE spawning the wake task. Nothing else
        # re-selects an endpoint already marked waking, so the wake was
        # stranded until an unrelated notify (next reconcile) resumed the
        # waiter. With unsynced-running being a normal post-restart state,
        # acquire must wake parked capacity on its own, without the reconcile
        # loop's help.
        class AlwaysFailingFetcher:
            def __call__(self, url: str) -> int:
                raise RuntimeError("health unreachable")

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
            reconcile_interval_s=3600,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            compute_usage_fetcher=AlwaysFailingFetcher(),
        )

        await self.router.start()

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["free_slots"], 0)
        self.assertFalse(
            next(ep for ep in snapshot["endpoints"] if ep["name"] == "endpoint-a")["usage_synced"]
        )

        # The reconcile loop cannot help within this timeout (interval 3600s):
        # acquire itself must spawn the wake before suspending.
        lease = await self.router.acquire(timeout_s=2.0)
        self.assertEqual(lease.endpoint_name, "endpoint-b")
        self.assertTrue(lease.waited_for_capacity)
        await self.router.release(lease.slot_id, connected=False)

    async def test_schema_error_revokes_capacity_after_successful_sync(self):
        # A node that synced once must go conservative again when the health
        # schema breaks at runtime, instead of allocating on stale data.
        class SchemaBreakingFetcher:
            def __init__(self):
                self.calls = 0

            def __call__(self, url: str) -> int:
                self.calls += 1
                if self.calls <= 1:
                    return 0
                raise ComputeUsageSchemaError("no session count in payload")

        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
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
            compute_usage_fetcher=SchemaBreakingFetcher(),
            usage_sync_stale_ttl_s=3600,
        )

        await self.router.start()
        snapshot = await self.router.snapshot()
        self.assertTrue(snapshot["endpoints"][0]["usage_synced"])
        self.assertEqual(snapshot["free_slots"], 2)

        await self.router.refresh()

        # The generous staleness TTL must not shield a schema error.
        snapshot = await self.router.snapshot()
        self.assertFalse(snapshot["endpoints"][0]["usage_synced"])
        self.assertEqual(snapshot["free_slots"], 0)

        healthy, detail, _ = await self.router.healthcheck()
        self.assertFalse(healthy)
        self.assertIn("have not synced usage", detail)

    async def test_transient_sync_failure_keeps_capacity_within_ttl(self):
        class FailAfterFirstFetcher:
            def __init__(self):
                self.calls = 0

            def __call__(self, url: str) -> int:
                self.calls += 1
                if self.calls <= 1:
                    return 1
                raise RuntimeError("read timeout")

        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
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
            compute_usage_fetcher=FailAfterFirstFetcher(),
            usage_sync_stale_ttl_s=3600,
        )

        await self.router.start()
        await self.router.refresh()

        # Within the TTL a transient failure keeps the last observation.
        snapshot = await self.router.snapshot()
        self.assertTrue(snapshot["endpoints"][0]["usage_synced"])
        self.assertEqual(snapshot["observed_active_sessions"], 1)
        self.assertEqual(snapshot["free_slots"], 1)

    async def test_transient_sync_failures_revoke_capacity_after_ttl(self):
        class FailAfterFirstFetcher:
            def __init__(self):
                self.calls = 0

            def __call__(self, url: str) -> int:
                self.calls += 1
                if self.calls <= 1:
                    return 0
                raise RuntimeError("read timeout")

        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
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
            compute_usage_fetcher=FailAfterFirstFetcher(),
            usage_sync_stale_ttl_s=0.05,
        )

        await self.router.start()
        await asyncio.sleep(0.1)
        await self.router.refresh()

        snapshot = await self.router.snapshot()
        self.assertFalse(snapshot["endpoints"][0]["usage_synced"])
        self.assertEqual(snapshot["free_slots"], 0)

    async def test_unsynced_endpoint_is_never_parked(self):
        # Blocking review finding: busy_sessions == 0 is meaningless for an
        # unsynced node. After an LB restart with a broken sync, a node
        # mid-conversation looks idle, and parking it would kill the live
        # conversation. Worse than the incident it replays.
        class AlwaysFailingFetcher:
            def __call__(self, url: str) -> int:
                raise RuntimeError("health unreachable")

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
            wake_threshold_slots=1,
            idle_park_timeout_s=0,
            reconcile_interval_s=3600,
            waking_capacity_timeout_s=300,
            park_cooldown_s=0,
            controller=controller,
            compute_usage_fetcher=AlwaysFailingFetcher(),
        )

        await self.router.start()
        # Both nodes are idle-past-timeout and above the warm floor; the only
        # thing protecting them is the sync gate.
        await self.router.refresh()
        await asyncio.sleep(0.05)

        self.assertEqual(controller.park_calls, [])
        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 2)

    async def test_schema_error_logs_even_when_never_synced(self):
        # Review finding: gating the error log on a usage_synced transition
        # meant a freshly restarted LB (nodes start unsynced) logged nothing,
        # on every poll, forever. This is the incident-replay case and must
        # be loud.
        class SchemaFailingFetcher:
            def __call__(self, url: str) -> int:
                raise ComputeUsageSchemaError("no session count in payload")

        controller = FakeEndpointController(
            [
                ("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud"),
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
            compute_usage_fetcher=SchemaFailingFetcher(),
        )

        with self.assertLogs("s2s-endpoint", level="ERROR") as first:
            await self.router.start()
        self.assertTrue(any("schema error" in line for line in first.output))

        # Still loud on subsequent polls, not just on a state transition.
        with self.assertLogs("s2s-endpoint", level="ERROR") as second:
            await self.router.refresh()
        self.assertTrue(any("schema error" in line for line in second.output))

    async def test_fully_busy_but_synced_pool_stays_healthy(self):
        # Known-full is healthy; only unknown capacity is degraded.
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
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
            compute_usage_fetcher=lambda url: 1,
        )

        await self.router.start()

        snapshot = await self.router.snapshot()
        self.assertTrue(snapshot["endpoints"][0]["usage_synced"])
        self.assertEqual(snapshot["free_slots"], 0)

        healthy, detail, _ = await self.router.healthcheck()
        self.assertTrue(healthy)
        self.assertIsNone(detail)

    async def test_start_retries_failed_initial_usage_sync(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"

        class FailOnceFetcher:
            def __init__(self, value: int):
                self.value = value
                self.calls = 0

            def __call__(self, url: str) -> int:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient health failure")
                return self.value

        usage_fetcher = FailOnceFetcher(value=1)
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

        snapshot = await self.router.snapshot()
        self.assertEqual(usage_fetcher.calls, 2)
        self.assertTrue(snapshot["endpoints"][0]["usage_synced"])
        self.assertEqual(snapshot["observed_active_sessions"], 1)
        self.assertEqual(snapshot["free_slots"], 1)

    async def test_woken_endpoint_counts_as_synced_without_health_poll(self):
        # A node the LB wakes itself is a fresh process with zero sessions,
        # so it must offer capacity even if its /health is not yet reachable.
        class AlwaysFailingFetcher:
            def __call__(self, url: str) -> int:
                raise RuntimeError("health unreachable")

        controller = FakeEndpointController(
            [
                ("endpoint-a", "paused", None),
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
            compute_usage_fetcher=AlwaysFailingFetcher(),
        )

        await self.router.start()
        await asyncio.sleep(0.1)

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["running_endpoints"], 1)
        self.assertTrue(snapshot["endpoints"][0]["usage_synced"])
        self.assertEqual(snapshot["free_slots"], 1)

        lease = await self.router.acquire(timeout_s=0.2)
        await self.router.release(lease.slot_id, connected=False)

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

    async def test_refresh_reports_busy_endpoint_when_it_becomes_paused(self):
        endpoint_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
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
        )
        downed_endpoints = []

        async def record_endpoint_down(endpoint_name: str) -> None:
            downed_endpoints.append(endpoint_name)

        await self.router.start()
        lease = await self.router.acquire(timeout_s=0.2)
        await self.router.mark_connected(lease.slot_id)
        self.router._on_endpoint_down = record_endpoint_down

        controller.states["endpoint-a"] = {"status": "paused", "url": None}
        await self.router.refresh()

        self.assertEqual(downed_endpoints, ["endpoint-a"])
        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["active_sessions"], 0)
        self.assertEqual(snapshot["local_active_sessions"], 0)
        self.assertEqual(snapshot["local_connected_sessions"], 0)

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

    async def test_reconcile_does_not_park_local_connected_endpoint(self):
        endpoint_a_url = "https://endpoint-a.example.endpoints.huggingface.cloud"
        endpoint_b_url = "https://endpoint-b.example.endpoints.huggingface.cloud"
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
        )

        await self.router.start()
        lease = await self.router.acquire(timeout_s=0.2)
        await self.router.mark_connected(lease.slot_id)

        await self.router._schedule_parks_if_needed()
        await asyncio.sleep(0.05)

        self.assertEqual(lease.endpoint_name, "endpoint-a")
        self.assertEqual(controller.park_calls, ["endpoint-b"])
        snapshot = await self.router.snapshot()
        endpoints = {endpoint["name"]: endpoint for endpoint in snapshot["endpoints"]}
        self.assertEqual(endpoints["endpoint-a"]["local_connected_sessions"], 1)
        self.assertEqual(endpoints["endpoint-a"]["active_sessions"], 1)

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

    async def test_drain_restart_skips_allocator_draining_endpoint(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = _make_drain_router(
            controller,
            endpoint_names=["endpoint-a"],
            drain_restart_timeout_s=0,
        )
        self.router._fetch_pool_units = lambda url: [
            {"state": "draining", "draining_for_s": 999}
        ]

        await self.router.start()
        await self.router.set_draining("endpoint-a", True, lease_id="rollout-a")
        await self.router._check_drain_restarts()
        await asyncio.sleep(0.05)

        self.assertEqual(controller.force_restart_calls, [])

    async def test_drain_restart_rechecks_allocator_drain_after_pool_poll(self):
        controller = FakeEndpointController(
            [("endpoint-a", "running", "https://endpoint-a.example.endpoints.huggingface.cloud")]
        )
        self.router = _make_drain_router(
            controller,
            endpoint_names=["endpoint-a"],
            drain_restart_timeout_s=0,
        )
        self.router._fetch_pool_units = lambda url: [
            {"state": "draining", "draining_for_s": 999}
        ]
        await self.router.start()

        async def mark_draining_during_poll(function, *args):
            await self.router.set_draining("endpoint-a", True, lease_id="rollout-a")
            return function(*args)

        with patch("app.endpoint_pool_router.asyncio.to_thread", new=mark_draining_during_poll):
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

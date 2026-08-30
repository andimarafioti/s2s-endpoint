import asyncio
import unittest

import httpx

from app.speech_proxy_router import (
    NoSpeechBackendAvailable,
    SpeechBackendConfig,
    SpeechBackendPool,
    SpeechBackendPoolSettings,
)


def _backends(count: int = 2):
    return tuple(
        SpeechBackendConfig(name=f"backend-{index}", url=f"https://backend-{index}.example")
        for index in range(1, count + 1)
    )


class SpeechBackendPoolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def handler(request: httpx.Request):
            return httpx.Response(200, json={"status": "ok"})

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def asyncTearDown(self):
        await self.client.aclose()

    def settings(self, **overrides):
        values = {
            "service": "tts",
            "target_work": 2,
            "max_work": 2,
            "latency_target": 0.5,
            "tts_warmup_enabled": False,
        }
        values.update(overrides)
        return SpeechBackendPoolSettings(**values)

    async def test_atomic_reservations_balance_and_honor_hard_capacity(self):
        pool = SpeechBackendPool(_backends(), self.settings(), client=self.client)
        await pool.refresh_health()

        leases = await asyncio.gather(*(pool.reserve(1) for _ in range(4)))
        snapshots = {snapshot.name: snapshot for snapshot in await pool.snapshots()}

        self.assertEqual([lease.backend_name for lease in leases], ["backend-1", "backend-2"] * 2)
        self.assertEqual(snapshots["backend-1"].active_work, 2)
        self.assertEqual(snapshots["backend-2"].active_work, 2)
        with self.assertRaises(NoSpeechBackendAvailable):
            await pool.reserve(1)

        await asyncio.gather(*(lease.release(success=True, latency=0.2) for lease in leases))
        snapshots = {snapshot.name: snapshot for snapshot in await pool.snapshots()}
        self.assertEqual(snapshots["backend-1"].active_work, 0)
        self.assertEqual(snapshots["backend-1"].successes, 2)
        self.assertAlmostEqual(snapshots["backend-1"].ewma_latency or 0, 0.2)

    async def test_latency_penalty_routes_around_a_slow_idle_backend(self):
        pool = SpeechBackendPool(
            _backends(),
            self.settings(max_work=4, latency_weight=1),
            client=self.client,
        )
        await pool.refresh_health()
        first = await pool.reserve(1)
        await first.release(success=True, latency=2.0)
        second = await pool.reserve(1)

        self.assertEqual(first.backend_name, "backend-1")
        self.assertEqual(second.backend_name, "backend-2")
        await second.release(success=True, latency=0.1)

    async def test_draining_backend_finishes_existing_work_but_gets_no_new_work(self):
        pool = SpeechBackendPool(_backends(), self.settings(), client=self.client)
        await pool.refresh_health()
        existing = await pool.reserve(1)
        await pool.set_draining(existing.backend_name, True)

        replacement = await pool.reserve(1)

        self.assertNotEqual(existing.backend_name, replacement.backend_name)
        await existing.release(success=True, latency=0.1)
        await replacement.release(success=True, latency=0.1)

    async def test_retryable_failures_make_backend_unready_at_threshold(self):
        pool = SpeechBackendPool(
            _backends(1),
            self.settings(failure_threshold=2),
            client=self.client,
        )
        await pool.refresh_health()

        for _ in range(2):
            lease = await pool.reserve(1)
            await lease.release(
                success=False,
                retryable_failure=True,
                error="transport failed",
            )

        snapshot = (await pool.snapshots())[0]
        self.assertFalse(snapshot.ready)
        self.assertEqual(snapshot.errors, 2)
        with self.assertRaises(NoSpeechBackendAvailable):
            await pool.reserve(1)

    async def test_tts_transition_to_ready_requires_real_warmup_audio(self):
        requests: list[tuple[str, str]] = []

        async def handler(request: httpx.Request):
            requests.append((request.method, request.url.path))
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(200, content=b"pcm-audio")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        pool = SpeechBackendPool(
            _backends(1),
            self.settings(tts_warmup_enabled=True),
            client=client,
        )

        await pool.refresh_health()

        self.assertTrue((await pool.snapshots())[0].ready)
        self.assertEqual(requests, [("GET", "/health"), ("POST", "/v1/audio/speech")])

    async def test_empty_tts_warmup_keeps_backend_unready(self):
        async def handler(request: httpx.Request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(200, content=b"")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        pool = SpeechBackendPool(
            _backends(1),
            self.settings(tts_warmup_enabled=True),
            client=client,
        )

        await pool.refresh_health()

        snapshot = (await pool.snapshots())[0]
        self.assertFalse(snapshot.ready)
        self.assertIn("empty response", snapshot.last_health_error or "")


if __name__ == "__main__":
    unittest.main()

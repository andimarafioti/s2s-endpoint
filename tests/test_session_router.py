import asyncio
import unittest

from app.session_router import SessionRouter


async def fake_wait_for_ready(host, port, process, timeout_s):
    return None


def fake_build_command(host, port):
    return ["echo", f"{host}:{port}"]


class SessionRouterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.router = SessionRouter(
            host="127.0.0.1",
            base_port=9000,
            repo_dir=".",
            build_command=fake_build_command,
            wait_for_ready=fake_wait_for_ready,
        )

    async def _fake_start(self):
        self.router._ready = True
        self.router._starting = False
        self.router._last_error = None

    async def test_start_makes_pipeline_ready(self):
        self.router._start_process = lambda timeout_s: self._fake_start()
        await self.router.start()

        snapshot = await self.router.snapshot()
        self.assertTrue(snapshot["ready"])

    async def test_acquire_returns_slot_when_ready(self):
        self.router._ready = True
        slot = await self.router.acquire()
        self.assertEqual(slot.ws_url, "ws://127.0.0.1:9000")

    async def test_acquire_raises_when_not_ready(self):
        with self.assertRaises(RuntimeError):
            await self.router.acquire()

    async def test_acquire_concurrent_up_to_max_sessions(self):
        router = SessionRouter(
            host="127.0.0.1",
            base_port=9000,
            repo_dir=".",
            build_command=fake_build_command,
            wait_for_ready=fake_wait_for_ready,
            max_sessions=2,
        )
        router._ready = True

        slot1 = await router.acquire()
        slot2 = await router.acquire()
        self.assertEqual(slot1.slot_id, slot2.slot_id)

        with self.assertRaises(RuntimeError):
            await router.acquire()

    async def test_release_frees_session(self):
        self.router._ready = True
        slot = await self.router.acquire()

        with self.assertRaises(RuntimeError):
            await self.router.acquire()

        await self.router.release(slot.slot_id)
        slot_again = await self.router.acquire()
        self.assertEqual(slot.slot_id, slot_again.slot_id)

    async def test_snapshot_reports_active_sessions(self):
        router = SessionRouter(
            host="127.0.0.1",
            base_port=9000,
            repo_dir=".",
            build_command=fake_build_command,
            wait_for_ready=fake_wait_for_ready,
            max_sessions=3,
        )
        router._ready = True

        await router.acquire()
        await router.acquire()
        snapshot = await router.snapshot()
        self.assertEqual(snapshot["max_sessions"], 3)
        self.assertEqual(snapshot["active_sessions"], 2)
        self.assertEqual(snapshot["free_sessions"], 1)

    async def test_healthcheck_healthy_when_ready(self):
        self.router._ready = True
        healthy, detail, snapshot = await self.router.healthcheck()
        self.assertTrue(healthy)
        self.assertIsNone(detail)

    async def test_healthcheck_unhealthy_when_starting(self):
        self.router._starting = True
        healthy, detail, _ = await self.router.healthcheck()
        self.assertFalse(healthy)
        self.assertIn("starting", detail)

    async def test_healthcheck_unhealthy_with_error(self):
        self.router._last_error = "process crashed"
        healthy, detail, _ = await self.router.healthcheck()
        self.assertFalse(healthy)
        self.assertEqual(detail, "process crashed")


if __name__ == "__main__":
    unittest.main()

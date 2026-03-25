import asyncio
import unittest

from app.session_router import SessionRouter


async def fake_wait_for_ready(host, port, process, timeout_s):
    return None


def fake_build_command(host, port):
    return ["echo", f"{host}:{port}"]


class SessionRouterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.router = SessionRouter(
            host="127.0.0.1",
            base_port=9000,
            repo_dir=".",
            min_idle_instances=1,
            max_instances=2,
            build_command=fake_build_command,
            wait_for_ready=fake_wait_for_ready,
        )

        async def fake_start_slot(slot, timeout_s):
            slot.starting = False
            slot.ready = True
            slot.last_error = None

        self.router._start_slot = fake_start_slot  # type: ignore[method-assign]
        self.router._stop_slot_process = lambda slot: None  # type: ignore[method-assign]

    async def test_start_ensures_min_idle_slot(self):
        await self.router.start()

        snapshot = await self.router.snapshot()
        self.assertEqual(snapshot["total_slots"], 1)
        self.assertEqual(snapshot["ready_idle"], 1)
        self.assertEqual(snapshot["ready_busy"], 0)

    async def test_acquire_waits_until_slot_is_released(self):
        self.router.max_instances = 1
        await self.router.start()

        slot = await self.router.acquire(timeout_s=0.2)

        async def acquire_again():
            return await self.router.acquire(timeout_s=0.2)

        waiter = asyncio.create_task(acquire_again())
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())

        await self.router.release(slot.slot_id)
        reused_slot = await waiter

        self.assertEqual(reused_slot.slot_id, slot.slot_id)
        self.assertTrue(reused_slot.busy)


if __name__ == "__main__":
    unittest.main()

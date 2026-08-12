import asyncio
import unittest

from app.app_utils import cancel_and_await


class CancelAndAwaitTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancels_and_awaits_running_task(self):
        started = asyncio.Event()

        async def run_until_cancelled():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(run_until_cancelled())
        await started.wait()

        await cancel_and_await(task)

        self.assertTrue(task.cancelled())

    async def test_propagates_non_cancellation_exceptions(self):
        async def fail():
            raise RuntimeError("task failed")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)

        with self.assertRaisesRegex(RuntimeError, "task failed"):
            await cancel_and_await(task)


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest

from app.verification_admission_limiter import (
    VerificationAdmissionConfig,
    VerificationAdmissionLimiter,
)


class VerificationAdmissionLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_enforces_per_network_pending_limit_until_task_finishes(self):
        limiter = VerificationAdmissionLimiter(
            config=VerificationAdmissionConfig(max_global_pending=4, max_network_pending=1)
        )
        _, permit = limiter.acquire("net:first")
        self.assertIsNotNone(permit)
        blocked, blocked_permit = limiter.acquire("net:first")

        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, "network_quota")
        self.assertIsNone(blocked_permit)

        task = asyncio.create_task(asyncio.sleep(0))
        permit.release_when_done(task)
        await task

        admitted, replacement = limiter.acquire("net:first")
        self.assertTrue(admitted.allowed)
        replacement.release()

    async def test_enforces_global_pending_limit_across_networks(self):
        limiter = VerificationAdmissionLimiter(
            config=VerificationAdmissionConfig(max_global_pending=2, max_network_pending=2)
        )
        _, first = limiter.acquire("net:first")
        _, second = limiter.acquire("net:second")
        blocked, blocked_permit = limiter.acquire("net:third")

        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, "global_quota")
        self.assertIsNone(blocked_permit)
        self.assertEqual(limiter.status()["pending"], 2)

        first.release()
        second.release()
        self.assertEqual(limiter.status()["pending"], 0)

    async def test_permit_release_is_idempotent(self):
        limiter = VerificationAdmissionLimiter(
            config=VerificationAdmissionConfig(max_global_pending=1, max_network_pending=1)
        )
        _, permit = limiter.acquire(None)

        permit.release()
        permit.release()

        self.assertEqual(limiter.status()["pending"], 0)


if __name__ == "__main__":
    unittest.main()

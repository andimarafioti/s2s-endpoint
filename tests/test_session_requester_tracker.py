import unittest

from app.requester_identity import RequesterIdentity
from app.session_requester_tracker import SessionRequesterTracker


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


class SessionRequesterTrackerTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.tracker = SessionRequesterTracker(retention_s=60, time_fn=self.clock)
        self.requester = RequesterIdentity(
            actor_id="token:abc123",
            label="HF token •abc123",
            kind="unverified_token",
            verification="pending",
            fingerprint="abc123",
        )

    def test_returns_requester_only_once(self):
        self.tracker.remember("session-1", self.requester)

        self.assertEqual(self.tracker.take("session-1"), self.requester)
        self.assertIsNone(self.tracker.take("session-1"))

    def test_discards_disconnected_pending_session(self):
        self.tracker.remember("session-1", self.requester)

        self.tracker.discard("session-1")

        self.assertIsNone(self.tracker.take("session-1"))

    def test_expires_session_that_never_connects(self):
        self.tracker.remember("session-1", self.requester)
        self.clock.now = 60

        self.assertEqual(self.tracker.count(), 0)
        self.assertIsNone(self.tracker.take("session-1"))


if __name__ == "__main__":
    unittest.main()

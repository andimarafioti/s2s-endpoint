import unittest

from app.requester_identity import RequesterIdentity
from app.requester_rate_limiter import (
    RequesterRateLimitConfig,
    RequesterRateLimiter,
)


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RequesterRateLimiterTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.config = RequesterRateLimitConfig(
            request_window_s=60,
            max_requests_per_window=100,
            max_parallel_allocations=2,
            max_consecutive_no_connects=3,
            short_session_threshold_s=10,
            max_consecutive_short_sessions=3,
            cooldown_s=120,
            actor_retention_s=300,
            max_actor_states=10,
        )
        self.limiter = RequesterRateLimiter(config=self.config, time_fn=self.clock)
        self.requester = RequesterIdentity(
            actor_id="anonymous:abc123",
            label="Anonymous IP •abc123",
            kind="anonymous",
            verification="not_provided",
            fingerprint="abc123",
            network_id="net:abc123",
            client_kind="automation:python-requests",
        )

    def allocate(self, session_id: str, *, pending_timeout_s: float = 60) -> None:
        decision = self.limiter.acquire(self.requester)
        self.assertTrue(decision.allowed)
        self.limiter.record_allocation(
            session_id,
            self.requester,
            pending_timeout_s=pending_timeout_s,
        )

    def test_rejects_parallel_allocations_before_more_capacity_is_reserved(self):
        self.allocate("session-1")
        self.allocate("session-2")

        rejected = self.limiter.acquire(self.requester)

        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.reason, "parallel_allocations")
        self.assertEqual(rejected.active_allocations, 2)
        self.assertEqual(rejected.retry_after_s, 60)

    def test_allocation_failure_releases_in_flight_permit_without_behavior_penalty(self):
        decision = self.limiter.acquire(self.requester)
        self.assertTrue(decision.allowed)

        self.limiter.record_allocation_failure(self.requester)

        status = self.limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["allocation_failures"], 1)
        self.assertNotIn("no_connects", status["totals"])

    def test_queued_abandonment_releases_in_flight_permit_without_failure(self):
        decision = self.limiter.acquire(self.requester)
        self.assertTrue(decision.allowed)

        self.limiter.record_allocation_abandoned(self.requester)

        status = self.limiter.status()
        self.assertEqual(status["active_allocations"], 0)
        self.assertEqual(status["totals"]["allocation_abandonments"], 1)
        self.assertNotIn("allocation_failures", status["totals"])
        self.assertNotIn("no_connects", status["totals"])

    def test_zero_length_pending_timeout_is_tracked_as_immediately_expired(self):
        decision = self.limiter.acquire(self.requester)
        self.assertTrue(decision.allowed)
        self.limiter.record_allocation(
            "session-1",
            self.requester,
            pending_timeout_s=0,
        )

        self.limiter.status()

        self.assertEqual(self.limiter.status()["totals"]["no_connects"], 1)

    def test_rejects_request_burst_and_recovers_when_window_moves(self):
        limiter = RequesterRateLimiter(
            config=RequesterRateLimitConfig(
                request_window_s=60,
                max_requests_per_window=3,
                max_parallel_allocations=2,
            ),
            time_fn=self.clock,
        )
        for _ in range(3):
            decision = limiter.acquire(self.requester)
            self.assertTrue(decision.allowed)
            limiter.record_allocation_failure(self.requester)

        rejected = limiter.acquire(self.requester)
        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.reason, "request_rate")
        self.assertEqual(rejected.retry_after_s, 60)

        self.clock.advance(60)
        recovered = limiter.acquire(self.requester)
        self.assertTrue(recovered.allowed)

    def test_repeated_allocations_that_never_connect_trigger_cooldown(self):
        for index in range(3):
            self.allocate(f"session-{index}", pending_timeout_s=10)
            self.clock.advance(10)

        rejected = self.limiter.acquire(self.requester)

        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.reason, "behavior_cooldown")
        self.assertEqual(rejected.consecutive_no_connects, 3)
        self.assertEqual(rejected.retry_after_s, 120)
        self.assertEqual(self.limiter.status()["totals"]["no_connects"], 3)

    def test_successful_join_resets_no_connect_streak(self):
        for index in range(2):
            self.allocate(f"miss-{index}", pending_timeout_s=10)
            self.clock.advance(10)

        self.allocate("joined")
        self.assertEqual(self.limiter.record_connected("joined"), self.requester)
        self.clock.advance(20)
        outcome = self.limiter.record_disconnected("joined")

        self.assertIsNotNone(outcome)
        self.assertFalse(outcome.short_session)
        status = self.limiter.status()
        self.assertEqual(status["blocked_actors"], 0)

    def test_repeated_short_connected_sessions_trigger_cooldown(self):
        for index in range(3):
            self.allocate(f"short-{index}")
            self.limiter.record_connected(f"short-{index}")
            self.clock.advance(6)
            outcome = self.limiter.record_disconnected(
                f"short-{index}",
                duration_s=6,
            )
            self.assertTrue(outcome.short_session)
            self.clock.advance(5)

        rejected = self.limiter.acquire(self.requester)

        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.reason, "behavior_cooldown")
        self.assertEqual(rejected.consecutive_short_sessions, 3)
        self.assertEqual(self.limiter.status()["totals"]["short_sessions"], 3)

    def test_meaningful_session_breaks_short_reconnect_streak(self):
        for index in range(2):
            self.allocate(f"short-{index}")
            self.limiter.record_connected(f"short-{index}")
            self.limiter.record_disconnected(f"short-{index}", duration_s=6)

        self.allocate("meaningful")
        self.limiter.record_connected("meaningful")
        outcome = self.limiter.record_disconnected("meaningful", duration_s=45)
        self.assertFalse(outcome.short_session)

        for index in range(2, 4):
            self.allocate(f"short-{index}")
            self.limiter.record_connected(f"short-{index}")
            self.limiter.record_disconnected(f"short-{index}", duration_s=6)

        decision = self.limiter.acquire(self.requester)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.consecutive_short_sessions, 2)

    def test_system_release_does_not_penalize_requester(self):
        self.allocate("session-1")
        self.limiter.record_connected("session-1")

        outcome = self.limiter.record_disconnected(
            "session-1",
            duration_s=1,
            penalize=False,
        )

        self.assertTrue(outcome.short_session)
        decision = self.limiter.acquire(self.requester)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.consecutive_short_sessions, 0)

    def test_disabled_mode_tracks_but_does_not_enforce(self):
        limiter = RequesterRateLimiter(
            config=RequesterRateLimitConfig(
                enabled=False,
                max_parallel_allocations=1,
                max_requests_per_window=1,
            ),
            time_fn=self.clock,
        )
        first = limiter.acquire(self.requester)
        limiter.record_allocation("session-1", self.requester, pending_timeout_s=60)
        second = limiter.acquire(self.requester)

        self.assertTrue(first.allowed)
        self.assertTrue(second.allowed)
        self.assertFalse(limiter.status()["enabled"])

    def test_actor_state_is_bounded_and_active_actor_is_not_evicted(self):
        limiter = RequesterRateLimiter(
            config=RequesterRateLimitConfig(max_actor_states=1),
            time_fn=self.clock,
        )
        first = limiter.acquire(self.requester)
        self.assertTrue(first.allowed)
        limiter.record_allocation("active", self.requester, pending_timeout_s=60)
        other = RequesterIdentity(
            actor_id="anonymous:other",
            label="other",
            kind="anonymous",
            verification="not_provided",
            fingerprint="other",
        )

        rejected = limiter.acquire(other)

        self.assertFalse(rejected.allowed)
        self.assertEqual(rejected.reason, "tracker_capacity")


class RequesterRateLimitConfigTests(unittest.TestCase):
    def test_rejects_non_positive_thresholds(self):
        with self.assertRaisesRegex(ValueError, "cooldown_s must be > 0"):
            RequesterRateLimitConfig(cooldown_s=0)


if __name__ == "__main__":
    unittest.main()

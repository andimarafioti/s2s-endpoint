import unittest

from app.dashboard_history import DashboardHistory, SwarmHistoryBucket
from app.requester_dashboard_ui import inject_requester_dashboard
from app.requester_identity import RequesterIdentity
from app.requester_usage import RequesterUsageService, RequesterUsageThresholds


class FakeClock:
    def __init__(self, now: float):
        self._now = now

    def now(self) -> float:
        return self._now

    def set(self, now: float) -> None:
        self._now = now


class RequesterUsageServiceTests(unittest.IsolatedAsyncioTestCase):
    def _service(
        self,
        clock: FakeClock,
        *,
        thresholds: RequesterUsageThresholds | None = None,
        max_requester_records: int = 50_000,
    ) -> RequesterUsageService:
        history = DashboardHistory(
            retention_minutes=24 * 60,
            max_requester_records=max_requester_records,
            time_fn=clock.now,
        )
        return RequesterUsageService(
            history=history,
            thresholds=thresholds or RequesterUsageThresholds(),
            time_fn=clock.now,
        )

    async def test_counts_hf_users_anonymous_ips_and_anomalies(self):
        service = self._service(
            FakeClock(2 * 3600),
            thresholds=RequesterUsageThresholds(
                high_volume_requests=3,
                burst_requests_per_minute=3,
                many_networks=2,
            ),
        )
        token_requester = RequesterIdentity(
            actor_id="token:abc123",
            label="@reachy-user · token •abc123",
            kind="authenticated",
            verification="verified",
            fingerprint="abc123",
            account_name="reachy-user",
            network_id="net:first",
            reported_robot_id="robot:first",
            client_kind="automation:httpx",
        )
        second_network = RequesterIdentity(
            **{
                **token_requester.__dict__,
                "network_id": "net:second",
                "reported_robot_id": "robot:second",
            }
        )
        anonymous = RequesterIdentity(
            actor_id="anonymous:ip123",
            label="Anonymous IP •ip123",
            kind="anonymous",
            verification="not_provided",
            fingerprint="ip123",
            network_id="net:ip123",
            client_kind="browser",
        )

        await service.record("request", token_requester)
        await service.record("request", second_network)
        await service.record("request", token_requester)
        await service.record("success", token_requester)
        await service.record("failure", token_requester)
        await service.record("request", anonymous)

        payload = await service.data(window_minutes=60)
        summary = payload["summary"]
        leaderboard = payload["leaderboard"]

        self.assertEqual(summary["unique_requesters_window"], 2)
        self.assertEqual(summary["authenticated_users_window"], 1)
        self.assertEqual(summary["tokens_window"], 1)
        self.assertEqual(summary["anonymous_ips_window"], 1)
        self.assertEqual(summary["reported_robots_window"], 2)
        self.assertEqual(summary["reported_robot_requests_window"], 3)
        self.assertEqual(summary["authenticated_requests_window"], 3)
        self.assertEqual(summary["anonymous_requests_window"], 1)
        self.assertEqual(summary["unattributed_requests_window"], 0)
        self.assertEqual(leaderboard[0]["label"], "@reachy-user · token •abc123")
        self.assertEqual(leaderboard[0]["requests"], 3)
        self.assertEqual(leaderboard[0]["network_count"], 2)
        self.assertEqual(leaderboard[0]["reported_robot_count"], 2)
        self.assertEqual(leaderboard[0]["reported_robot_requests"], 3)
        self.assertEqual(
            leaderboard[0]["reported_robot_ids"],
            ["robot:first", "robot:second"],
        )
        self.assertEqual(leaderboard[0]["automated_requests"], 3)
        self.assertEqual(leaderboard[0]["risk"], "high")
        self.assertIn("high volume: 3 requests", leaderboard[0]["signals"])
        self.assertIn("burst: 3/min", leaderboard[0]["signals"])
        self.assertIn("many networks: 2", leaderboard[0]["signals"])

    async def test_resolved_identity_updates_and_round_trips_existing_history(self):
        service = self._service(FakeClock(2 * 3600))
        pending = RequesterIdentity(
            actor_id="token:abc123",
            label="HF token •abc123",
            kind="unverified_token",
            verification="pending",
            fingerprint="abc123",
            network_id="net:first",
            reported_robot_id="robot:first",
            client_kind="browser",
        )
        resolved = RequesterIdentity(
            actor_id="token:abc123",
            label="@reachy-user · token •abc123",
            kind="authenticated",
            verification="verified",
            fingerprint="abc123",
            account_name="reachy-user",
        )

        await service.record("request", pending)
        await service.update_identity(resolved)

        bucket = (await service.history.snapshot())[-1]
        restored = SwarmHistoryBucket.from_dict(bucket.to_dict())
        record = restored.requester_usage["token:abc123"]
        self.assertEqual(record["label"], "@reachy-user · token •abc123")
        self.assertEqual(record["kind"], "authenticated")
        self.assertEqual(record["verification"], "verified")
        self.assertEqual(record["account_name"], "reachy-user")
        self.assertEqual(record["network_ids"], ["net:first"])
        self.assertEqual(record["reported_robot_requests"], 1)
        self.assertEqual(record["reported_robot_ids"], ["robot:first"])
        self.assertFalse(record["reported_robot_ids_overflow"])
        self.assertEqual(record["client_kinds"], {"browser": 1})

    async def test_compacts_oldest_requester_details_at_retention_wide_limit(self):
        clock = FakeClock(2 * 3600)
        service = self._service(clock, max_requester_records=2)

        for index in range(3):
            clock.set(2 * 3600 + index * 60)
            await service.record(
                "request",
                RequesterIdentity(
                    actor_id=f"token:{index}",
                    label=f"Token {index}",
                    kind="authenticated",
                    verification="verified",
                    fingerprint=str(index),
                    account_name=f"user-{index}",
                ),
            )

        buckets = await service.history.snapshot()
        payload = await service.data(window_minutes=60)

        self.assertEqual(sum(bucket.session_requests for bucket in buckets), 3)
        self.assertEqual(sum(len(bucket.requester_usage) for bucket in buckets), 2)
        self.assertEqual(buckets[0].requester_usage, {})
        self.assertEqual(payload["tracked_requests"], 2)
        self.assertEqual(payload["unattributed_requests"], 1)
        self.assertEqual(
            {row["actor_id"] for row in payload["leaderboard"]},
            {"token:1", "token:2"},
        )
        self.assertEqual(service.history.persistence_status()["requester_record_count"], 2)
        self.assertEqual(service.history.persistence_status()["max_requester_records"], 2)

    async def test_bounds_requester_details_restored_from_persistence(self):
        clock = FakeClock(2 * 3600)
        service = self._service(clock, max_requester_records=2)
        buckets = []
        for index in range(3):
            bucket = SwarmHistoryBucket(bucket_start_s=2 * 3600 + index * 60)
            bucket.requester_usage[f"token:{index}"] = {
                "label": f"Token {index}",
                "kind": "authenticated",
                "verification": "verified",
                "fingerprint": str(index),
                "account_name": f"user-{index}",
                "requests": 1,
                "successes": 0,
                "failures": 0,
                "abandoned": 0,
                "network_ids": [],
                "network_ids_overflow": False,
                "client_kinds": {},
            }
            buckets.append(bucket)

        await service.history._merge_persisted_history_buckets(buckets)
        restored = await service.history.snapshot()

        self.assertEqual(sum(len(bucket.requester_usage) for bucket in restored), 2)
        self.assertEqual(restored[0].requester_usage, {})
        self.assertEqual(service.history.persistence_status()["requester_record_count"], 2)

    async def test_counts_each_bucket_using_its_own_token_state(self):
        clock = FakeClock(2 * 3600)
        service = self._service(clock)
        verified = RequesterIdentity(
            actor_id="token:changing",
            label="@reachy-user · token •changing",
            kind="authenticated",
            verification="verified",
            fingerprint="changing",
            account_name="reachy-user",
        )
        invalid = RequesterIdentity(
            actor_id="token:changing",
            label="Invalid token •changing",
            kind="invalid_token",
            verification="invalid",
            fingerprint="changing",
        )

        for _ in range(10):
            await service.record("request", verified)
        clock.set(2 * 3600 + 60)
        for _ in range(5):
            await service.record("request", invalid)

        payload = await service.data(window_minutes=60)
        summary = payload["summary"]
        row = payload["leaderboard"][0]

        self.assertEqual(summary["authenticated_requests_window"], 10)
        self.assertEqual(summary["invalid_token_requests_window"], 5)
        self.assertEqual(row["requests"], 15)
        self.assertEqual(row["kind"], "invalid_token")
        self.assertEqual(row["verification"], "invalid")
        self.assertEqual(row["invalid_token_requests"], 5)
        self.assertIn("invalid HF token", row["signals"])


class RequesterDashboardUiTests(unittest.TestCase):
    def test_injects_requester_dashboard_fragments(self):
        template = """
        <style>__REQUESTER_DASHBOARD_STYLES__</style>
        <main>__REQUESTER_DASHBOARD_MARKUP__</main>
        <script>
          const cards = [__REQUESTER_DASHBOARD_KPI_CARDS__];
          __REQUESTER_DASHBOARD_SCRIPT__
        </script>
        """

        html = inject_requester_dashboard(template)

        self.assertNotIn("__REQUESTER_DASHBOARD_", html)
        self.assertIn('id="requester-leaderboard"', html)
        self.assertIn("Requester Usage", html)
        self.assertIn("Reported robots", html)
        self.assertIn("not hardware attestation", html)
        self.assertIn("function renderRequesterUsage(requesters, summary)", html)

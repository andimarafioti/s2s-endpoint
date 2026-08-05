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
        await service.record("connected", token_requester)
        await service.record("failure", token_requester)
        await service.record("auth_rejected", token_requester)
        await service.record("rate_limited", token_requester)
        await service.record_session_outcome(
            token_requester,
            duration_s=6,
            short_session=True,
        )
        await service.record_session_outcome(
            token_requester,
            duration_s=6,
            short_session=True,
        )
        await service.record_session_outcome(
            token_requester,
            duration_s=45,
            short_session=False,
        )
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
        self.assertEqual(summary["allocated_requesters_window"], 1)
        self.assertEqual(summary["connected_requesters_window"], 1)
        self.assertEqual(summary["authenticated_users_connected_window"], 1)
        self.assertEqual(summary["attributed_connections_window"], 1)
        self.assertEqual(summary["authenticated_requests_window"], 3)
        self.assertEqual(summary["anonymous_requests_window"], 1)
        self.assertEqual(summary["auth_rejected_requests_window"], 1)
        self.assertEqual(summary["rate_limited_requests_window"], 1)
        self.assertEqual(summary["unattributed_requests_window"], 0)
        self.assertEqual(leaderboard[0]["actor_id"], "hf:reachy-user")
        self.assertEqual(leaderboard[0]["label"], "@reachy-user")
        self.assertEqual(leaderboard[0]["token_count"], 1)
        self.assertEqual(leaderboard[0]["token_fingerprints"], ["abc123"])
        self.assertEqual(leaderboard[0]["requests"], 3)
        self.assertEqual(leaderboard[0]["auth_rejected"], 1)
        self.assertEqual(leaderboard[0]["network_count"], 2)
        self.assertEqual(leaderboard[0]["reported_robot_count"], 2)
        self.assertEqual(leaderboard[0]["reported_robot_requests"], 3)
        self.assertEqual(
            leaderboard[0]["reported_robot_ids"],
            ["robot:first", "robot:second"],
        )
        self.assertEqual(leaderboard[0]["automated_requests"], 3)
        self.assertEqual(leaderboard[0]["connections"], 1)
        self.assertEqual(leaderboard[0]["completed_sessions"], 3)
        self.assertEqual(leaderboard[0]["short_sessions"], 2)
        self.assertEqual(leaderboard[0]["avg_connected_duration_s"], 19.0)
        self.assertEqual(leaderboard[0]["max_connected_duration_s"], 45.0)
        self.assertEqual(leaderboard[0]["rate_limited"], 1)
        self.assertNotIn("connection_rate_pct", leaderboard[0])
        self.assertEqual(leaderboard[0]["risk"], "high")
        self.assertIn("high volume: 3 requests", leaderboard[0]["signals"])
        self.assertIn("burst: 3/min", leaderboard[0]["signals"])
        self.assertIn("many networks: 2", leaderboard[0]["signals"])
        self.assertIn("rate limited: 1 request", leaderboard[0]["signals"])
        self.assertFalse(any(signal.startswith("mostly short sessions") for signal in leaderboard[0]["signals"]))

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
        await service.record("connected", resolved)

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
        self.assertEqual(record["connections"], 1)
        self.assertEqual(record["client_kinds"], {"browser": 1})

    async def test_stale_disconnect_cannot_downgrade_verified_identity(self):
        clock = FakeClock(2 * 3600)
        service = self._service(clock)
        pending = RequesterIdentity(
            actor_id="token:abc123",
            label="HF token •abc123",
            kind="unverified_token",
            verification="pending",
            fingerprint="abc123",
        )
        verified = RequesterIdentity(
            actor_id="token:abc123",
            label="@reachy-user · token •abc123",
            kind="authenticated",
            verification="verified",
            fingerprint="abc123",
            account_name="reachy-user",
        )

        await service.record("request", pending)
        await service.update_identity(verified)
        await service.record("connected", verified)
        clock.set(2 * 3600 + 60)
        await service.record_session_outcome(
            pending,
            duration_s=297,
            short_session=False,
        )

        payload = await service.data(window_minutes=60)
        row = payload["leaderboard"][0]

        self.assertEqual(payload["summary"]["authenticated_users_window"], 1)
        self.assertEqual(payload["summary"]["authenticated_users_connected_window"], 1)
        self.assertEqual(row["actor_id"], "hf:reachy-user")
        self.assertEqual(row["label"], "@reachy-user")
        self.assertEqual(row["kind"], "authenticated")
        self.assertEqual(row["verification"], "verified")
        self.assertEqual(row["account_name"], "reachy-user")

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
            {"hf:user-1", "hf:user-2"},
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

    async def test_counts_each_bucket_without_downgrading_stronger_identity(self):
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
        self.assertEqual(row["kind"], "authenticated")
        self.assertEqual(row["verification"], "verified")
        self.assertEqual(row["account_name"], "reachy-user")
        self.assertEqual(row["invalid_token_requests"], 5)
        self.assertIn("invalid HF token", row["signals"])

    async def test_merges_verified_tokens_for_same_account_across_persisted_buckets(self):
        clock = FakeClock(2 * 3600)
        service = self._service(
            clock,
            thresholds=RequesterUsageThresholds(
                high_volume_requests=3,
                burst_requests_per_minute=2,
                many_networks=2,
            ),
        )
        first_token = RequesterIdentity(
            actor_id="token:first",
            label="@Andito · token •first",
            kind="authenticated",
            verification="verified",
            fingerprint="first",
            account_name="Andito",
            network_id="net:first",
            reported_robot_id="robot:first",
            client_kind="browser",
        )
        second_token = RequesterIdentity(
            actor_id="token:second",
            label="@andito · token •second",
            kind="authenticated",
            verification="verified",
            fingerprint="second",
            account_name="andito",
            network_id="net:second",
            reported_robot_id="robot:second",
            client_kind="automation:httpx",
        )

        await service.record("request", first_token)
        await service.record("success", first_token)
        await service.record("failure", first_token)
        await service.record("connected", first_token)
        await service.record_session_outcome(first_token, duration_s=10, short_session=True)
        clock.set(2 * 3600 + 60)
        await service.record("request", first_token)
        await service.record("request", second_token)
        await service.record("success", second_token)
        await service.record("rate_limited", second_token)
        await service.record("abandoned", second_token)
        await service.record("connected", second_token)
        await service.record_session_outcome(second_token, duration_s=30, short_session=False)

        persisted = [SwarmHistoryBucket.from_dict(bucket.to_dict()) for bucket in await service.history.snapshot()]
        restored = self._service(clock, thresholds=service.thresholds)
        await restored.history._merge_persisted_history_buckets(persisted)

        payload = await restored.data(window_minutes=60)
        summary = payload["summary"]
        self.assertEqual(len(payload["leaderboard"]), 1)
        row = payload["leaderboard"][0]

        self.assertEqual(row["actor_id"], "hf:andito")
        self.assertEqual(row["label"], "@andito")
        self.assertEqual(row["account_name"], "andito")
        self.assertEqual(row["token_count"], 2)
        self.assertEqual(row["token_fingerprints"], ["first", "second"])
        self.assertEqual(row["requests"], 3)
        self.assertEqual(row["successes"], 2)
        self.assertEqual(row["failures"], 1)
        self.assertEqual(row["rate_limited"], 1)
        self.assertEqual(row["abandoned"], 1)
        self.assertEqual(row["connections"], 2)
        self.assertEqual(row["completed_sessions"], 2)
        self.assertEqual(row["short_sessions"], 1)
        self.assertEqual(row["avg_connected_duration_s"], 20.0)
        self.assertEqual(row["max_connected_duration_s"], 30.0)
        self.assertEqual(row["peak_requests_per_minute"], 2)
        self.assertEqual(row["network_count"], 2)
        self.assertEqual(row["reported_robot_count"], 2)
        self.assertEqual(row["reported_robot_requests"], 3)
        self.assertEqual(
            row["client_kinds"],
            {"browser": 2, "automation:httpx": 1},
        )
        self.assertEqual(row["risk"], "high")
        self.assertIn("high volume: 3 requests", row["signals"])
        self.assertIn("burst: 2/min", row["signals"])
        self.assertIn("many networks: 2", row["signals"])
        self.assertIn("rate limited: 1 request", row["signals"])
        self.assertEqual(summary["unique_requesters_window"], 1)
        self.assertEqual(summary["authenticated_users_window"], 1)
        self.assertEqual(summary["tokens_window"], 2)
        self.assertEqual(summary["token_requests_window"], 3)
        self.assertEqual(summary["allocated_requesters_window"], 1)
        self.assertEqual(summary["connected_requesters_window"], 1)
        self.assertEqual(summary["authenticated_users_connected_window"], 1)

    async def test_keeps_verified_accounts_separate(self):
        service = self._service(FakeClock(2 * 3600))
        for fingerprint, account_name in (
            ("first", "andito"),
            ("second", "reachy-user"),
        ):
            await service.record(
                "request",
                RequesterIdentity(
                    actor_id=f"token:{fingerprint}",
                    label=f"@{account_name} · token •{fingerprint}",
                    kind="authenticated",
                    verification="verified",
                    fingerprint=fingerprint,
                    account_name=account_name,
                ),
            )

        payload = await service.data(window_minutes=60)

        self.assertEqual(
            {row["actor_id"] for row in payload["leaderboard"]},
            {"hf:andito", "hf:reachy-user"},
        )
        self.assertEqual(payload["summary"]["authenticated_users_window"], 2)
        self.assertEqual(payload["summary"]["tokens_window"], 2)

    async def test_keeps_unverified_and_invalid_tokens_as_distinct_actors(self):
        service = self._service(FakeClock(2 * 3600))
        requesters = (
            RequesterIdentity(
                actor_id="token:verified",
                label="@andito · token •verified",
                kind="authenticated",
                verification="verified",
                fingerprint="verified",
                account_name="andito",
            ),
            RequesterIdentity(
                actor_id="token:pending",
                label="HF token •pending",
                kind="unverified_token",
                verification="pending",
                fingerprint="pending",
            ),
            RequesterIdentity(
                actor_id="token:invalid",
                label="Invalid token •invalid",
                kind="invalid_token",
                verification="invalid",
                fingerprint="invalid",
            ),
        )
        for requester in requesters:
            await service.record("request", requester)

        payload = await service.data(window_minutes=60)

        self.assertEqual(
            {row["actor_id"] for row in payload["leaderboard"]},
            {"hf:andito", "token:pending", "token:invalid"},
        )
        self.assertTrue(all(row["token_count"] == 1 for row in payload["leaderboard"]))
        self.assertEqual(payload["summary"]["authenticated_users_window"], 1)
        self.assertEqual(payload["summary"]["tokens_window"], 3)
        self.assertEqual(payload["summary"]["invalid_token_requests_window"], 1)


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
        self.assertIn("Connected requesters", html)
        self.assertIn("first compute websocket callback", html)
        self.assertIn("not hardware attestation", html)
        self.assertIn("function requesterCredentialSummary(row)", html)
        self.assertIn("row.token_fingerprints", html)
        self.assertIn("function renderRequesterUsage(requesters, summary)", html)

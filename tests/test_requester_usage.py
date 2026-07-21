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


class RequesterUsageServiceTests(unittest.IsolatedAsyncioTestCase):
    def _service(
        self,
        clock: FakeClock,
        *,
        thresholds: RequesterUsageThresholds | None = None,
    ) -> RequesterUsageService:
        history = DashboardHistory(
            retention_minutes=24 * 60,
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
            client_kind="automation:httpx",
        )
        second_network = RequesterIdentity(
            **{
                **token_requester.__dict__,
                "network_id": "net:second",
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
        self.assertEqual(summary["authenticated_requests_window"], 3)
        self.assertEqual(summary["anonymous_requests_window"], 1)
        self.assertEqual(summary["unattributed_requests_window"], 0)
        self.assertEqual(leaderboard[0]["label"], "@reachy-user · token •abc123")
        self.assertEqual(leaderboard[0]["requests"], 3)
        self.assertEqual(leaderboard[0]["network_count"], 2)
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
        self.assertEqual(record["client_kinds"], {"browser": 1})


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
        self.assertIn("function renderRequesterUsage(requesters, summary)", html)

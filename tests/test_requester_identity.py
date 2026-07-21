import asyncio
import unittest
from types import SimpleNamespace

from app.requester_identity import RequesterIdentityResolver, bearer_token, client_address


class FakeRequest:
    def __init__(self, *, headers=None, client_host="10.0.0.1"):
        self.headers = headers or {}
        self.client = SimpleNamespace(host=client_host) if client_host is not None else None


class FakeHubError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"HF status {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class RequesterIdentityResolverTests(unittest.IsolatedAsyncioTestCase):
    async def test_anonymous_requests_are_grouped_by_hashed_forwarded_ip(self):
        resolver = RequesterIdentityResolver(hash_secret="stable-secret", whoami_fn=lambda token: {})
        first = resolver.identify(
            FakeRequest(headers={"x-forwarded-for": "203.0.113.8, 10.0.0.4"})
        )
        second = resolver.identify(
            FakeRequest(headers={"x-forwarded-for": "203.0.113.8"}, client_host="10.0.0.9")
        )

        self.assertEqual(first.actor_id, second.actor_id)
        self.assertEqual(first.kind, "anonymous")
        self.assertTrue(first.label.startswith("Anonymous IP"))
        self.assertNotIn("203.0.113.8", first.actor_id)
        self.assertNotIn("203.0.113.8", first.label)

        await resolver.stop()

    async def test_proxy_headers_can_be_ignored(self):
        request = FakeRequest(
            headers={"x-forwarded-for": "203.0.113.8"},
            client_host="10.0.0.9",
        )

        self.assertEqual(client_address(request, trust_proxy_headers=True), "203.0.113.8")
        self.assertEqual(client_address(request, trust_proxy_headers=False), "10.0.0.9")

    async def test_valid_hf_token_is_resolved_and_cached_without_exposing_token(self):
        calls = []
        updates = []
        updated = asyncio.Event()

        def whoami(token):
            calls.append(token)
            return {"name": "reachy-user"}

        async def on_update(identity):
            updates.append(identity)
            updated.set()

        resolver = RequesterIdentityResolver(
            hash_secret="stable-secret",
            whoami_fn=whoami,
            on_identity_update=on_update,
        )
        raw_token = "hf_a_secret_token_value"
        request = FakeRequest(
            headers={
                "authorization": f"Bearer {raw_token}",
                "x-forwarded-for": "203.0.113.10",
                "user-agent": "python-httpx/0.27",
            }
        )

        pending = resolver.identify(request)
        self.assertEqual(pending.verification, "pending")
        self.assertNotIn(raw_token, pending.actor_id)
        self.assertNotIn(raw_token, pending.label)

        await asyncio.wait_for(updated.wait(), timeout=1)
        resolved = resolver.identify(request)

        self.assertEqual(calls, [raw_token])
        self.assertEqual(resolved.kind, "authenticated")
        self.assertEqual(resolved.verification, "verified")
        self.assertEqual(resolved.account_name, "reachy-user")
        self.assertIn("@reachy-user", resolved.label)
        self.assertEqual(resolved.client_kind, "automation:httpx")
        self.assertEqual(updates[0].actor_id, pending.actor_id)

        await resolver.stop()

    async def test_invalid_token_is_classified_but_request_remains_identifiable(self):
        updated = asyncio.Event()
        updates = []

        async def on_update(identity):
            updates.append(identity)
            updated.set()

        resolver = RequesterIdentityResolver(
            hash_secret="stable-secret",
            whoami_fn=lambda token: (_ for _ in ()).throw(FakeHubError(401)),
            on_identity_update=on_update,
        )
        request = FakeRequest(headers={"authorization": "Bearer hf_invalid_but_well_formed"})

        pending = resolver.identify(request)
        await asyncio.wait_for(updated.wait(), timeout=1)
        invalid = resolver.identify(request)

        self.assertEqual(invalid.actor_id, pending.actor_id)
        self.assertEqual(invalid.kind, "invalid_token")
        self.assertEqual(invalid.verification, "invalid")
        self.assertTrue(invalid.label.startswith("Invalid token"))
        self.assertEqual(updates[0].kind, "invalid_token")

        await resolver.stop()

    async def test_non_hf_bearer_value_is_not_sent_to_whoami(self):
        calls = []
        resolver = RequesterIdentityResolver(
            hash_secret="stable-secret",
            whoami_fn=lambda token: calls.append(token) or {},
        )

        identity = resolver.identify(
            FakeRequest(headers={"authorization": "Bearer not-an-hf-token"})
        )
        await asyncio.sleep(0)

        self.assertEqual(identity.kind, "unverified_token")
        self.assertEqual(identity.verification, "unrecognized")
        self.assertEqual(calls, [])

        await resolver.stop()


class BearerTokenTests(unittest.TestCase):
    def test_extracts_case_insensitive_bearer_token(self):
        self.assertEqual(bearer_token("bEaReR hf_value"), "hf_value")

    def test_rejects_missing_or_other_auth_schemes(self):
        self.assertIsNone(bearer_token(None))
        self.assertIsNone(bearer_token("Basic value"))
        self.assertIsNone(bearer_token("Bearer   "))


if __name__ == "__main__":
    unittest.main()

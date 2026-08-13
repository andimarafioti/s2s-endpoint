"""The session token carries an LLM proxy claim derived from the HF token.

At session creation the load balancer fingerprints the caller's HF token
(HMAC keyed by the session shared secret) and embeds it in the signed session
token, for the compute replica to gate its LLM proxy paths against. The claim
is minted only for tokens HF whoami has verified: sessions created without a
token, or with one that cannot be verified, carry no claim. Raw tokens are
never stored anywhere.
"""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.direct_session_manager import DirectSessionManager
from app.endpoint_pool_router import EndpointLease
from app.load_balancer_app import _llm_proxy_fingerprint
from app.requester_identity import RequesterIdentity, RequesterIdentityResolver
from app.session_tokens import (
    create_session_token,
    llm_token_fingerprint,
    verify_session_token,
)
from tests.helpers import load_balancer_fixture

SECRET = "shared-secret"
HF_TOKEN = "hf_faketesttoken1234"
REQUESTER_CONTEXT = {
    "actor_id": "token:abcdef0123456789",
    "metadata": {
        "label": "@reachy-user · token •abcdef01",
        "kind": "authenticated",
        "verification": "verified",
        "fingerprint": "abcdef0123456789",
        "account_name": "reachy-user",
        "network_id": "net:network123",
        "reported_robot_id": "robot:robot123",
        "client_kind": "robot:httpx",
    },
}


class LlmTokenFingerprintTests(unittest.TestCase):
    def test_same_secret_and_token_agree_across_calls(self):
        # The LB computes it at session creation, the replica on every proxy
        # request; the whole design rests on the two agreeing.
        self.assertEqual(
            llm_token_fingerprint(SECRET, HF_TOKEN),
            llm_token_fingerprint(SECRET, HF_TOKEN),
        )

    def test_fingerprint_depends_on_the_secret(self):
        self.assertNotEqual(
            llm_token_fingerprint(SECRET, HF_TOKEN),
            llm_token_fingerprint("another-secret", HF_TOKEN),
        )

    def test_fingerprint_depends_on_the_token(self):
        self.assertNotEqual(
            llm_token_fingerprint(SECRET, HF_TOKEN),
            llm_token_fingerprint(SECRET, "hf_othertoken"),
        )

    def test_fingerprint_does_not_contain_the_token(self):
        self.assertNotIn(HF_TOKEN, llm_token_fingerprint(SECRET, HF_TOKEN))


class SessionTokenClaimTests(unittest.TestCase):
    def _mint(self, **kwargs):
        return create_session_token(
            SECRET,
            session_id="session-1",
            websocket_url="wss://compute-01.example/v1/realtime",
            callback_url="https://lb.example/internal/sessions/session-1/event",
            ttl_s=60.0,
            **kwargs,
        )

    def test_claim_survives_the_signed_round_trip(self):
        fingerprint = llm_token_fingerprint(SECRET, HF_TOKEN)
        payload = verify_session_token(
            self._mint(llm_fingerprint=fingerprint, llm_requester=REQUESTER_CONTEXT),
            SECRET,
        )
        self.assertEqual(payload["llmf"], fingerprint)
        self.assertEqual(payload["llmr"], REQUESTER_CONTEXT)
        self.assertNotIn(HF_TOKEN, str(payload))

    def test_token_without_claim_has_no_llmf_key(self):
        payload = verify_session_token(self._mint(), SECRET)
        self.assertNotIn("llmf", payload)


class _SingleLeaseRouter:
    LEASE = EndpointLease(
        slot_id="compute-01",
        endpoint_name="compute-01",
        ws_url="wss://compute-01.example/v1/realtime",
        waited_for_capacity=False,
    )

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def acquire(self, timeout_s: float = 900.0) -> EndpointLease:
        return self.LEASE

    async def try_acquire(self) -> EndpointLease:
        return self.LEASE

    async def release(self, slot_id, *, connected: bool = False) -> None:
        pass


class _ToggleRouter(_SingleLeaseRouter):
    def __init__(self, *, has_capacity: bool):
        self.has_capacity = has_capacity

    async def try_acquire(self):
        if not self.has_capacity:
            return None
        return self.LEASE


class GrantEmbedsClaimTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_path_grant_embeds_the_fingerprint(self):
        manager = DirectSessionManager(
            endpoint_router=_SingleLeaseRouter(),
            session_shared_secret=SECRET,
            queue_enabled=False,
        )
        fingerprint = llm_token_fingerprint(SECRET, HF_TOKEN)

        allocation = await manager.allocate("https://lb.example", llm_fingerprint=fingerprint)

        payload = verify_session_token(str(allocation["session_token"]), SECRET)
        self.assertEqual(payload["llmf"], fingerprint)

    async def test_fast_path_grant_embeds_the_requester_context(self):
        manager = DirectSessionManager(
            endpoint_router=_SingleLeaseRouter(),
            session_shared_secret=SECRET,
            queue_enabled=False,
        )
        fingerprint = llm_token_fingerprint(SECRET, HF_TOKEN)

        allocation = await manager.allocate(
            "https://lb.example",
            llm_fingerprint=fingerprint,
            llm_requester=REQUESTER_CONTEXT,
        )

        payload = verify_session_token(str(allocation["session_token"]), SECRET)
        self.assertEqual(payload["llmr"], REQUESTER_CONTEXT)

    async def test_grant_without_fingerprint_carries_no_claim(self):
        manager = DirectSessionManager(
            endpoint_router=_SingleLeaseRouter(),
            session_shared_secret=SECRET,
            queue_enabled=False,
        )

        allocation = await manager.allocate("https://lb.example")

        payload = verify_session_token(str(allocation["session_token"]), SECRET)
        self.assertNotIn("llmf", payload)

    async def test_queue_ticket_carries_the_fingerprint_to_the_claimed_grant(self):
        # Queue polls are bodyless GETs with no Authorization header, so the
        # fingerprint computed at ticket creation must ride the ticket.
        router = _ToggleRouter(has_capacity=False)
        manager = DirectSessionManager(
            endpoint_router=router,
            session_shared_secret=SECRET,
            queue_enabled=True,
            queue_reap_interval_s=3600,
        )
        fingerprint = llm_token_fingerprint(SECRET, HF_TOKEN)
        try:
            ticket = await manager.allocate(
                "https://lb.example",
                llm_fingerprint=fingerprint,
                llm_requester=REQUESTER_CONTEXT,
            )
            self.assertEqual(ticket["state"], "queued")

            router.has_capacity = True
            grant = await manager.poll(str(ticket["queue_id"]), "https://lb.example")

            self.assertEqual(grant["state"], "granted")
            payload = verify_session_token(str(grant["session_token"]), SECRET)
            self.assertEqual(payload["llmf"], fingerprint)
            self.assertEqual(payload["llmr"], REQUESTER_CONTEXT)
        finally:
            await manager.stop()


class _FakeHubError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"HF status {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class LoadBalancerFingerprintTests(unittest.IsolatedAsyncioTestCase):
    def _import_load_balancer(self, secret: str = SECRET, **environ):
        return load_balancer_fixture({"SESSION_SHARED_SECRET": secret, **environ})

    def _request(self, headers: dict[str, str]) -> SimpleNamespace:
        return SimpleNamespace(headers=headers)

    def _requester(self, verification: str) -> RequesterIdentity:
        return RequesterIdentity(
            actor_id="token:abcdef0123456789",
            label="HF token •abcdef01",
            kind="authenticated" if verification == "verified" else "unverified_token",
            verification=verification,
            fingerprint="abcdef0123456789",
        )

    async def test_verified_hf_bearer_yields_the_shared_fingerprint(self):
        module = self._import_load_balancer()
        result = await _llm_proxy_fingerprint(
            module.runtime,
            self._request({"authorization": f"Bearer {HF_TOKEN}"}),
            self._requester("verified"),
        )
        self.assertEqual(result, llm_token_fingerprint(SECRET, HF_TOKEN))

    async def test_reachy_authorization_header_wins_over_authorization(self):
        module = self._import_load_balancer()
        result = await _llm_proxy_fingerprint(
            module.runtime,
            self._request(
                {
                    "x-reachy-mini-authorization": f"Bearer {HF_TOKEN}",
                    "authorization": "Bearer hf_other",
                }
            ),
            self._requester("verified"),
        )
        self.assertEqual(result, llm_token_fingerprint(SECRET, HF_TOKEN))

    async def test_missing_bearer_yields_no_claim(self):
        module = self._import_load_balancer()
        self.assertIsNone(await _llm_proxy_fingerprint(module.runtime, self._request({}), self._requester("verified")))

    async def test_unvalidatable_token_yields_no_claim(self):
        module = self._import_load_balancer()
        self.assertIsNone(
            await _llm_proxy_fingerprint(
                module.runtime,
                self._request({"authorization": "Bearer bad token with spaces"}),
                self._requester("verified"),
            )
        )

    async def test_no_shared_secret_yields_no_claim(self):
        module = self._import_load_balancer(secret="")
        self.assertIsNone(
            await _llm_proxy_fingerprint(
                module.runtime,
                self._request({"authorization": f"Bearer {HF_TOKEN}"}),
                self._requester("verified"),
            )
        )

    async def test_unverified_token_yields_no_claim(self):
        # The gate the whole claim rests on: an invented-but-plausible bearer
        # must not mint a claim, or it would get LLM proxy access and, when
        # rotated, a fresh per-fingerprint rate-limit identity each time.
        module = self._import_load_balancer()
        for verification in ("pending", "invalid", "unavailable", "unrecognized"):
            with self.subTest(verification=verification):
                self.assertIsNone(
                    await _llm_proxy_fingerprint(
                        module.runtime,
                        self._request({"authorization": f"Bearer {HF_TOKEN}"}),
                        self._requester(verification),
                    )
                )

    async def test_pending_token_waits_for_whoami_and_mints_the_claim(self):
        # First-seen tokens are still validating when the session is created;
        # the claim decision waits for the verdict instead of failing closed.
        module = self._import_load_balancer()
        resolver = RequesterIdentityResolver(hash_secret=SECRET, whoami_fn=lambda token: {"name": "reachy-user"})
        request = self._request({"authorization": f"Bearer {HF_TOKEN}"})
        try:
            with patch.object(module.dependencies, "requester_identity_resolver", resolver):
                pending = resolver.identify(request)
                self.assertEqual(pending.verification, "pending")
                result = await _llm_proxy_fingerprint(module.runtime, request, pending)
            self.assertEqual(result, llm_token_fingerprint(SECRET, HF_TOKEN))
        finally:
            await resolver.stop()

    async def test_pending_token_rejected_by_whoami_yields_no_claim(self):
        def whoami(token):
            raise _FakeHubError(401)

        module = self._import_load_balancer()
        resolver = RequesterIdentityResolver(hash_secret=SECRET, whoami_fn=whoami)
        request = self._request({"authorization": f"Bearer {HF_TOKEN}"})
        try:
            with patch.object(module.dependencies, "requester_identity_resolver", resolver):
                pending = resolver.identify(request)
                self.assertIsNone(await _llm_proxy_fingerprint(module.runtime, request, pending))
        finally:
            await resolver.stop()

    async def test_slow_whoami_fails_closed_without_blocking_session_creation(self):
        def whoami(token):
            time.sleep(0.5)
            return {"name": "reachy-user"}

        module = self._import_load_balancer(
            LLM_PROXY_CLAIM_VERIFY_TIMEOUT_S="0.05",
        )
        resolver = RequesterIdentityResolver(hash_secret=SECRET, whoami_fn=whoami)
        request = self._request({"authorization": f"Bearer {HF_TOKEN}"})
        try:
            with patch.object(module.dependencies, "requester_identity_resolver", resolver):
                pending = resolver.identify(request)
                started = asyncio.get_running_loop().time()
                result = await _llm_proxy_fingerprint(module.runtime, request, pending)
                waited_s = asyncio.get_running_loop().time() - started
            self.assertIsNone(result)
            self.assertLess(waited_s, 0.4)
        finally:
            await resolver.stop()


if __name__ == "__main__":
    unittest.main()

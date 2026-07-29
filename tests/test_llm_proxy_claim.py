"""The session token carries an LLM proxy claim derived from the HF token.

At session creation the load balancer fingerprints the caller's HF token
(HMAC keyed by the session shared secret) and embeds it in the signed session
token, for the compute replica to gate its LLM proxy paths against. Sessions
created without a plausible HF token carry no claim. Raw tokens are never
stored anywhere.
"""

import importlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.direct_session_manager import DirectSessionManager
from app.endpoint_pool_router import EndpointLease
from app.session_tokens import (
    create_session_token,
    llm_token_fingerprint,
    verify_session_token,
)

SECRET = "shared-secret"
HF_TOKEN = "hf_faketesttoken1234"


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
        payload = verify_session_token(self._mint(llm_fingerprint=fingerprint), SECRET)
        self.assertEqual(payload["llmf"], fingerprint)

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
            ticket = await manager.allocate("https://lb.example", llm_fingerprint=fingerprint)
            self.assertEqual(ticket["state"], "queued")

            router.has_capacity = True
            grant = await manager.poll(str(ticket["queue_id"]), "https://lb.example")

            self.assertEqual(grant["state"], "granted")
            payload = verify_session_token(str(grant["session_token"]), SECRET)
            self.assertEqual(payload["llmf"], fingerprint)
        finally:
            await manager.stop()


class LoadBalancerFingerprintTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("app.load_balancer_main", None)

    def _import_load_balancer(self, secret: str = SECRET):
        sys.modules.pop("app.load_balancer_main", None)
        with patch.dict(
            "os.environ",
            {
                "COMPUTE_ENDPOINT_NAMES": "TEST",
                "DASHBOARD_BUCKET_ID": "",
                "DASHBOARD_PREVIEW_MODE": "",
                "SESSION_SHARED_SECRET": secret,
            },
            clear=False,
        ):
            return importlib.import_module("app.load_balancer_main")

    def _request(self, headers: dict[str, str]) -> SimpleNamespace:
        return SimpleNamespace(headers=headers)

    def test_hf_shaped_bearer_yields_the_shared_fingerprint(self):
        module = self._import_load_balancer()
        result = module._llm_proxy_fingerprint(self._request({"authorization": f"Bearer {HF_TOKEN}"}))
        self.assertEqual(result, llm_token_fingerprint(SECRET, HF_TOKEN))

    def test_reachy_authorization_header_wins_over_authorization(self):
        module = self._import_load_balancer()
        result = module._llm_proxy_fingerprint(
            self._request(
                {
                    "x-reachy-mini-authorization": f"Bearer {HF_TOKEN}",
                    "authorization": "Bearer hf_other",
                }
            )
        )
        self.assertEqual(result, llm_token_fingerprint(SECRET, HF_TOKEN))

    def test_missing_bearer_yields_no_claim(self):
        module = self._import_load_balancer()
        self.assertIsNone(module._llm_proxy_fingerprint(self._request({})))

    def test_unvalidatable_token_yields_no_claim(self):
        module = self._import_load_balancer()
        self.assertIsNone(
            module._llm_proxy_fingerprint(self._request({"authorization": "Bearer bad token with spaces"}))
        )

    def test_no_shared_secret_yields_no_claim(self):
        module = self._import_load_balancer(secret="")
        self.assertIsNone(module._llm_proxy_fingerprint(self._request({"authorization": f"Bearer {HF_TOKEN}"})))


if __name__ == "__main__":
    unittest.main()

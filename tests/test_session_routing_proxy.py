import json
import unittest

import httpx

from app.session_routing_proxy import SessionRoutingProxy
from tests.test_ws_proxy import FakeClientWS


class RoutingProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_hold_and_ack_are_private_and_correlated(self):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append((body, request.headers))
            if body["action"] == "prepare":
                return httpx.Response(200, json={"hold": "@old+new::", "routing": {"pipeline": "@new::", "routes": {}, "updates_enabled": True}})
            return httpx.Response(200, json={"pipeline": "@new::"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(client.aclose)
        ws = FakeClientWS()
        counts = {"old": 1}
        proxy = SessionRoutingProxy(ws, {"sid": "session", "session_token": "grant", "callback_url": "https://lb/internal/sessions/session/event", "routing": {"pipeline": "old"}}, "callback-key", counts, client=client)
        raw = {"type": "session.update", "event_id": "client-event", "session": {"models": {"stt": None}}, "_session_routing": {"routing": "forged"}}
        forwarded = json.loads(await proxy.client_message(json.dumps(raw)))
        update_id = forwarded["_session_routing"]["update_id"]
        self.assertNotEqual(update_id, "client-event")
        self.assertEqual(forwarded["_session_routing"]["routing"]["id"], "session")
        self.assertEqual(counts, {"@old+new::": 1})
        result = json.loads(await proxy.server_message(json.dumps({"type": "session.updated", "session": {}, "_session_routing": update_id})))
        await proxy.after_send()
        self.assertNotIn("_session_routing", result)
        self.assertEqual(counts, {"@new::": 1})
        self.assertEqual(requests[-1][0]["action"], "commit")
        self.assertEqual(requests[0][1]["x-reachy-mini-callback-authorization"], "Bearer callback-key")

    async def test_capacity_rejection_does_not_forward_or_change_selection(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(409)))
        self.addAsyncCleanup(client.aclose)
        ws = FakeClientWS()
        counts = {"old": 1}
        proxy = SessionRoutingProxy(ws, {"sid": "session", "session_token": "grant", "callback_url": "https://lb/internal/sessions/session/event", "routing": {"pipeline": "old"}}, "callback-key", counts, client=client)
        result = await proxy.client_message(json.dumps({"type": "session.update", "event_id": "event", "session": {"model": "new", "instructions": "changed"}}))
        self.assertIsNone(result)
        error = json.loads(ws.sent[0])
        self.assertEqual(error["error"]["event_id"], "event")
        self.assertEqual(counts, {"old": 1})

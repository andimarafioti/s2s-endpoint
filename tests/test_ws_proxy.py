import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from websockets.exceptions import ConnectionClosed

from app import compute_app as compute_main
from app.ws_proxy import proxy_websocket


class FakeClientWS:
    def __init__(self, events=None, headers=None):
        self.events = events if events is not None else []
        self.headers = headers if headers is not None else {}
        self.sent = []
        self.close_calls = []
        self.accepted_subprotocols = []

    async def accept(self, subprotocol=None):
        self.events.append("accept")
        self.accepted_subprotocols.append(subprotocol)

    async def send_text(self, text):
        self.sent.append(text)

    async def send_bytes(self, data):
        self.sent.append(data)

    async def close(self, code=1000, reason=""):
        self.close_calls.append((code, reason))

    async def receive(self):
        return {"type": "websocket.disconnect"}


class FakeUpstreamWS:
    async def recv(self):
        raise ConnectionClosed(None, None)

    async def send(self, data):
        pass


class _FakeConnectCtx:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return FakeUpstreamWS()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _lease(slot_id=0, ws_url="ws://127.0.0.1:9999"):
    return SimpleNamespace(slot_id=slot_id, ws_url=ws_url)


def _dependencies(router, notify):
    return compute_main.ComputeDependencies(
        session_router=router,
        connected_llm_fingerprints=compute_main._ConnectedFingerprintRegistry(),
        llm_rate_limiter=compute_main._FingerprintRateLimiter(100),
        http_get_json=lambda url: {},
        notify_lb_session_event=notify,
        proxy_websocket=proxy_websocket,
    )


class ProxyWebsocketLeaseHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_on_lease_acquired_runs_after_acquire_and_before_accept(self):
        events = []
        client = FakeClientWS(events=events)
        released = []

        async def acquire(_timeout):
            events.append("acquire")
            return _lease()

        async def release(slot_id):
            released.append(slot_id)

        async def on_lease_acquired():
            events.append("callback")

        with patch("app.ws_proxy.websockets.connect", _FakeConnectCtx):
            acquired = await proxy_websocket(
                client,
                acquire_lease=acquire,
                release_lease=release,
                describe_lease=lambda lease: str(lease.slot_id),
                no_capacity_reason="No pipeline capacity available",
                no_capacity_log="no capacity",
                on_lease_acquired=on_lease_acquired,
            )

        self.assertTrue(acquired)
        self.assertEqual(events, ["acquire", "callback", "accept"])
        self.assertEqual(released, [0])

    async def test_capacity_rejection_skips_lease_callback(self):
        client = FakeClientWS()
        callback = AsyncMock()

        async def acquire(_timeout):
            raise RuntimeError("all 3 pipeline session(s) are in use")

        async def release(slot_id):
            raise AssertionError("release must not run when no lease was acquired")

        acquired = await proxy_websocket(
            client,
            acquire_lease=acquire,
            release_lease=release,
            describe_lease=lambda lease: str(lease.slot_id),
            no_capacity_reason="No pipeline capacity available",
            no_capacity_log="no capacity",
            on_lease_acquired=callback,
        )

        self.assertFalse(acquired)
        callback.assert_not_awaited()
        self.assertTrue(any("session_limit_reached" in str(item) for item in client.sent))
        self.assertEqual(client.close_calls[-1][0], 1013)

    async def test_accept_failure_releases_lease(self):
        # accept() can raise when the client drops before the handshake
        # completes; the compute lease must be released or the pipeline slot
        # is permanently lost until the process restarts.
        class AcceptFailsWS(FakeClientWS):
            async def accept(self, subprotocol=None):
                raise RuntimeError("client went away before accept")

        client = AcceptFailsWS()
        released = []

        async def acquire(_timeout):
            return _lease(slot_id=3)

        async def release(slot_id):
            released.append(slot_id)

        async def on_lease_acquired():
            pass

        with self.assertRaisesRegex(RuntimeError, "client went away"):
            await proxy_websocket(
                client,
                acquire_lease=acquire,
                release_lease=release,
                describe_lease=lambda lease: str(lease.slot_id),
                no_capacity_reason="No pipeline capacity available",
                no_capacity_log="no capacity",
                on_lease_acquired=on_lease_acquired,
            )

        self.assertEqual(released, [3])

    async def test_lease_callback_failure_releases_lease_and_closes_client(self):
        client = FakeClientWS()
        released = []

        async def acquire(_timeout):
            return _lease(slot_id=7)

        async def release(slot_id):
            released.append(slot_id)

        async def on_lease_acquired():
            raise RuntimeError("LB unreachable")

        acquired = await proxy_websocket(
            client,
            acquire_lease=acquire,
            release_lease=release,
            describe_lease=lambda lease: str(lease.slot_id),
            no_capacity_reason="No pipeline capacity available",
            no_capacity_log="no capacity",
            on_lease_acquired=on_lease_acquired,
        )

        self.assertTrue(acquired)
        self.assertEqual(released, [7])
        self.assertEqual(client.close_calls[-1][0], 1011)


class ProxyWebsocketSubprotocolTests(unittest.IsolatedAsyncioTestCase):
    async def _proxy(self, client, *, capacity_available=True):
        async def acquire(_timeout):
            if not capacity_available:
                raise RuntimeError("all pipeline sessions are in use")
            return _lease()

        async def release(_slot_id):
            pass

        with patch("app.ws_proxy.websockets.connect", _FakeConnectCtx):
            return await proxy_websocket(
                client,
                acquire_lease=acquire,
                release_lease=release,
                describe_lease=lambda lease: str(lease.slot_id),
                no_capacity_reason="No pipeline capacity available",
                no_capacity_log="no capacity",
            )

    async def test_selects_realtime_from_offered_subprotocols(self):
        client = FakeClientWS(
            headers={
                "sec-websocket-protocol": ("realtime, openai-insecure-api-key.s2s-local, openai-agents-sdk.0.14.3")
            }
        )

        acquired = await self._proxy(client)

        self.assertTrue(acquired)
        self.assertEqual(client.accepted_subprotocols, ["realtime"])

    async def test_capacity_rejection_selects_realtime(self):
        client = FakeClientWS(headers={"sec-websocket-protocol": "realtime, client-metadata"})

        acquired = await self._proxy(client, capacity_available=False)

        self.assertFalse(acquired)
        self.assertEqual(client.accepted_subprotocols, ["realtime"])

    async def test_does_not_echo_unknown_subprotocol(self):
        client = FakeClientWS(headers={"sec-websocket-protocol": "client-auth, client-metadata"})

        await self._proxy(client)

        self.assertEqual(client.accepted_subprotocols, [None])

    async def test_accepts_clients_without_subprotocols(self):
        client = FakeClientWS()

        await self._proxy(client)

        self.assertEqual(client.accepted_subprotocols, [None])


class ComputeSessionEventOrderingTests(unittest.IsolatedAsyncioTestCase):
    """Regression tests for the 2026-06-07 incident dashboard artifact.

    A capacity-rejected websocket used to post connected then disconnected
    milliseconds apart, spiking completed conversations while live users
    stayed at zero.
    """

    def _payload(self):
        return {
            "callback_url": "https://lb.example/internal/sessions/abc/event",
            "session_token": "token-abc",
        }

    def _settings(self):
        return compute_main.ComputeSettings(session_shared_secret="test-secret")

    async def test_rejected_session_posts_only_disconnected(self):
        client = FakeClientWS()
        notify = AsyncMock()

        async def failing_acquire():
            raise RuntimeError("all 3 pipeline session(s) are in use")

        router = SimpleNamespace(acquire=failing_acquire, release=AsyncMock())
        with patch.object(compute_main, "_get_session_payload", return_value=self._payload()):
            await compute_main.websocket_proxy(client, self._settings(), _dependencies(router, notify))

        events = [call.args[2] for call in notify.await_args_list]
        self.assertEqual(events, ["disconnected"])
        self.assertTrue(any("session_limit_reached" in str(item) for item in client.sent))

    async def test_successful_session_posts_connected_then_disconnected(self):
        client = FakeClientWS()
        notify = AsyncMock()

        async def acquire():
            return _lease(slot_id=0, ws_url="ws://127.0.0.1:9999")

        async def release(slot_id):
            pass

        router = SimpleNamespace(acquire=acquire, release=release)
        with (
            patch.object(compute_main, "_get_session_payload", return_value=self._payload()),
            patch("app.ws_proxy.websockets.connect", _FakeConnectCtx),
        ):
            await compute_main.websocket_proxy(client, self._settings(), _dependencies(router, notify))

        events = [call.args[2] for call in notify.await_args_list]
        self.assertEqual(events, ["connected", "disconnected"])

    async def test_connected_notify_failure_still_attempts_release_notification(self):
        # If the connected callback failed, the LB may or may not have
        # registered the session; a best-effort disconnected keeps a possibly
        # half-registered session from leaking as connected forever.
        client = FakeClientWS()
        calls = []

        async def notify(callback_url, token, event, **kwargs):
            calls.append(event)
            if event == "connected":
                raise RuntimeError("LB unreachable")

        async def acquire():
            return _lease(slot_id=0, ws_url="ws://127.0.0.1:9999")

        async def release(slot_id):
            pass

        router = SimpleNamespace(acquire=acquire, release=release)
        with patch.object(compute_main, "_get_session_payload", return_value=self._payload()):
            await compute_main.websocket_proxy(client, self._settings(), _dependencies(router, notify))

        self.assertEqual(calls, ["connected", "disconnected"])
        # The session must not proxy after a failed connected notification.
        self.assertEqual(client.close_calls[-1][0], 1011)

    async def test_accept_failure_after_connected_releases_slot_and_notifies_lb(self):
        # The exact reviewer scenario: LB already told 'connected', then
        # accept() fails. The pipeline slot must be released and the LB must
        # still receive 'disconnected' so the session does not leak.
        class AcceptFailsWS(FakeClientWS):
            async def accept(self, subprotocol=None):
                raise RuntimeError("client went away before accept")

        client = AcceptFailsWS()
        notify = AsyncMock()
        released = []

        async def acquire():
            return _lease(slot_id=0, ws_url="ws://127.0.0.1:9999")

        async def release(slot_id):
            released.append(slot_id)

        router = SimpleNamespace(acquire=acquire, release=release)
        with patch.object(compute_main, "_get_session_payload", return_value=self._payload()):
            await compute_main.websocket_proxy(client, self._settings(), _dependencies(router, notify))

        events = [call.args[2] for call in notify.await_args_list]
        self.assertEqual(events, ["connected", "disconnected"])
        self.assertEqual(released, [0])


class NotifyLbRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_until_success(self):
        calls = []

        def post(url, payload):
            calls.append(payload["event"])
            if len(calls) < 3:
                raise RuntimeError("LB callback failed with HTTP 503")

        await compute_main._notify_lb_session_event(
            "https://lb.example/internal/sessions/abc/event",
            "token-abc",
            "disconnected",
            post_json=post,
            default_backoff_s=1.0,
            attempts=3,
            backoff_s=0,
        )

        self.assertEqual(calls, ["disconnected"] * 3)

    async def test_raises_after_exhausting_attempts(self):
        calls = []

        def post(url, payload):
            calls.append(payload["event"])
            raise RuntimeError("LB callback failed: read timeout")

        with self.assertRaisesRegex(RuntimeError, "read timeout"):
            await compute_main._notify_lb_session_event(
                "https://lb.example/internal/sessions/abc/event",
                "token-abc",
                "disconnected",
                post_json=post,
                default_backoff_s=1.0,
                attempts=3,
                backoff_s=0,
            )

        self.assertEqual(len(calls), 3)

    async def test_disconnect_survives_transient_lb_failure_end_to_end(self):
        # The demo swarm stale-session scenario: the first disconnected POST
        # times out, but the retry lands and the LB releases the session.
        client = FakeClientWS()
        posted = []

        def post(url, payload):
            posted.append(payload["event"])
            if payload["event"] == "disconnected" and posted.count("disconnected") == 1:
                raise RuntimeError("LB callback failed: read timeout")

        async def acquire():
            return _lease(slot_id=0, ws_url="ws://127.0.0.1:9999")

        async def release(slot_id):
            pass

        payload = {
            "callback_url": "https://lb.example/internal/sessions/abc/event",
            "session_token": "token-abc",
        }

        async def notify(callback_url, token, event, *, attempts=1, backoff_s=None):
            await compute_main._notify_lb_session_event(
                callback_url,
                token,
                event,
                post_json=post,
                default_backoff_s=0.0,
                attempts=attempts,
                backoff_s=backoff_s,
            )

        router = SimpleNamespace(acquire=acquire, release=release)
        settings = compute_main.ComputeSettings(
            session_shared_secret="test-secret",
            lb_callback_retry_attempts=5,
            lb_callback_retry_backoff_s=0.0,
        )
        with (
            patch.object(compute_main, "_get_session_payload", return_value=payload),
            patch("app.ws_proxy.websockets.connect", _FakeConnectCtx),
        ):
            await compute_main.websocket_proxy(client, settings, _dependencies(router, notify))

        self.assertEqual(posted, ["connected", "disconnected", "disconnected"])


if __name__ == "__main__":
    unittest.main()

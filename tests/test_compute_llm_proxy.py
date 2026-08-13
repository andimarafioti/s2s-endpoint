"""App-level tests for the compute replica's LLM proxy passthrough and gate.

The replica owns access control for POST /v1/chat/completions and
POST /v1/responses: the api key must be the HF token a currently connected
realtime session was created with (checked by HMAC fingerprint against the
session token's claim), throttled per fingerprint. Authorized requests are
forwarded to the internal speech-to-speech pipeline, which owns the OpenAI
contract itself. These tests drive compute_main's app with a stub internal
pipeline HTTP server on an ephemeral port and a faked pipeline websocket for
the realtime connections that open the access window.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi.testclient import TestClient

from app import compute_app as compute_main
from app.session_tokens import create_session_token, llm_token_fingerprint

SECRET = "compute-test-secret"
HF_TOKEN = "hf_faketesttoken1234"
OTHER_HF_TOKEN = "hf_otherfaketoken5678"


def _auth(token: str = HF_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


class StubInternalPipeline:
    """Stand-in for the internal speech-to-speech HTTP listener.

    Records every request (method, path, headers, body) and answers with
    whatever ``responder`` returns: ``(status, dict)`` for a JSON body, or
    ``(status, content_type, frames)`` where ``frames`` is a list of
    ``(delay_s, bytes)`` chunks flushed one at a time for streaming answers.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responder: Callable[[str], Any] = lambda path: (200, {"ok": True})

        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                self.do_POST()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                stub.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {key.lower(): value for key, value in self.headers.items()},
                        "body": body,
                    }
                )
                answer = stub.responder(self.path)
                if len(answer) == 2:
                    status, payload = answer
                    content = json.dumps(payload).encode("utf-8")
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    status, content_type, frames = answer
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for delay_s, frame in frames:
                        time.sleep(delay_s)
                        chunk = f"{len(frame):x}\r\n".encode("ascii") + frame + b"\r\n"
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _IdleUpstreamWS:
    """Fake pipeline websocket that stays open until the client disconnects."""

    async def recv(self):
        await asyncio.Event().wait()

    async def send(self, data) -> None:
        pass


class _FakeUpstreamConnect:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _IdleUpstreamWS:
        return _IdleUpstreamWS()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _mint_session_token(
    *,
    host: str = "testserver",
    hf_token: str | None = HF_TOKEN,
    secret: str = SECRET,
    session_id: str = "session-under-test",
) -> str:
    return create_session_token(
        secret,
        session_id=session_id,
        websocket_url=f"ws://{host}/v1/realtime",
        callback_url=f"http://lb.internal/internal/sessions/{session_id}/event",
        ttl_s=600.0,
        llm_fingerprint=llm_token_fingerprint(secret, hf_token) if hf_token else None,
    )


@contextlib.contextmanager
def _connected_session(client: TestClient, **mint_kwargs: Any) -> Iterator[None]:
    token = _mint_session_token(**mint_kwargs)
    with client.websocket_connect(f"/v1/realtime?session_token={token}"):
        yield


class ComputeLlmProxyTestCase(unittest.TestCase):
    """Shared harness: stub pipeline plus the gate armed on compute_main's app.

    ``gated_client`` patches the session router to hand out fake leases and
    the pipeline websocket to an idle fake, so TestClient websocket
    connections exercise the real route — session token verification,
    fingerprint registration, LB notifications (stubbed) — without spawning
    a speech-to-speech process. The TestClient is never entered as a context
    manager: the app lifespan would spawn a real speech-to-speech process.
    """

    def setUp(self) -> None:
        self.stub = StubInternalPipeline()
        self.addCleanup(self.stub.close)

    def gated_client(
        self,
        *,
        rate_limit_rpm: int = 100,
        enable_llm_proxy: bool = True,
        secret: str = SECRET,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> TestClient:
        self.lb_events: list[tuple[str, dict[str, Any]]] = []
        self.fail_usage_callback = False

        async def _fake_acquire():
            return SimpleNamespace(slot_id=0, ws_url="ws://127.0.0.1:1/v1/realtime")

        async def _fake_release(slot_id) -> None:
            pass

        async def _fake_healthcheck():
            return True, None, {"active_sessions": 1, "max_sessions": 1}

        async def _no_lb_callback(callback_url, session_token, event, **kwargs: Any) -> None:
            if event == "llm_proxy_request" and self.fail_usage_callback:
                raise RuntimeError("LB unavailable")
            self.lb_events.append((event, kwargs.get("extra_payload") or {}))

        self.enterContext(patch("app.ws_proxy.websockets.connect", _FakeUpstreamConnect))
        stub_url = urlsplit(self.stub.base_url)
        settings = compute_main.ComputeSettings(
            internal_ws_host=str(stub_url.hostname),
            internal_ws_base_port=int(stub_url.port),
            enable_llm_proxy=enable_llm_proxy,
            session_shared_secret=secret,
        )
        dependencies = compute_main.ComputeDependencies(
            session_router=SimpleNamespace(
                acquire=_fake_acquire,
                release=_fake_release,
                healthcheck=_fake_healthcheck,
            ),
            connected_llm_fingerprints=compute_main._ConnectedFingerprintRegistry(),
            llm_rate_limiter=compute_main._FingerprintRateLimiter(rate_limit_rpm, time_fn=time_fn),
            http_get_json=compute_main._http_get_json,
            notify_lb_session_event=_no_lb_callback,
            proxy_websocket=compute_main.proxy_websocket,
        )
        return TestClient(compute_main.create_app(settings, dependencies))

    def post_chat(self, client: TestClient, headers: dict[str, str] | None = None):
        if headers is None:
            headers = _auth()
        return client.post("/v1/chat/completions", content=b"{}", headers=headers)

    def post_until_401(self, client: TestClient, deadline_s: float = 5.0):
        """Post until the access window closes: the websocket route's teardown
        runs asynchronously after the client-side close, so the 401 is
        eventual."""
        deadline = time.monotonic() + deadline_s
        while True:
            response = self.post_chat(client)
            if response.status_code == 401 or time.monotonic() > deadline:
                return response
            time.sleep(0.05)

    def wait_for_usage_events(self, count: int = 1, deadline_s: float = 5.0) -> list[dict[str, Any]]:
        deadline = time.monotonic() + deadline_s
        while True:
            events = [payload for event, payload in self.lb_events if event == "llm_proxy_request"]
            if len(events) >= count or time.monotonic() > deadline:
                return events
            time.sleep(0.01)


class AuthorizedPassthroughTests(ComputeLlmProxyTestCase):
    def test_chat_completions_reaches_internal_pipeline_verbatim(self) -> None:
        self.stub.responder = lambda path: (
            200,
            {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]},
        )
        client = self.gated_client()
        with _connected_session(client):
            response = client.post(
                "/v1/chat/completions",
                content=b'{"messages":[{"role":"user","content":"hi"}],"custom_field":1}',
                headers=_auth(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]})
        self.assertEqual(len(self.stub.requests), 1)
        seen = self.stub.requests[0]
        self.assertEqual(seen["method"], "POST")
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["body"], b'{"messages":[{"role":"user","content":"hi"}],"custom_field":1}')

    def test_api_key_is_never_forwarded_to_the_pipeline(self) -> None:
        """The api key is a user's HF token — a real credential — so it must
        stop at the replica."""
        client = self.gated_client()
        with _connected_session(client):
            self.post_chat(client)

        seen = self.stub.requests[0]
        self.assertNotIn("authorization", seen["headers"])
        self.assertNotIn(HF_TOKEN, json.dumps(seen["headers"]))

    def test_responses_path_forwards_to_internal_responses(self) -> None:
        self.stub.responder = lambda path: (200, {"id": "resp_1", "output": []})
        client = self.gated_client()
        with _connected_session(client):
            response = client.post("/v1/responses", content=b'{"input":"hello"}', headers=_auth())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"id": "resp_1", "output": []})
        self.assertEqual(self.stub.requests[0]["path"], "/v1/responses")

    def test_pipeline_answers_pass_through_unchanged(self) -> None:
        """Beyond the gate, the contract clients see is the pipeline's own:
        the replica must not reinterpret its answers."""
        cases = [(400, "invalid_request_error"), (501, "not_implemented"), (500, "server_error")]
        for status, error_type in cases:
            with self.subTest(status=status):
                payload = {"error": {"message": "from the pipeline", "type": error_type}}
                self.stub.responder = lambda path, status=status, payload=payload: (status, payload)
                client = self.gated_client()
                with _connected_session(client):
                    response = self.post_chat(client)

                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json(), payload)

    def test_sse_stream_passes_through_verbatim(self) -> None:
        frames = [
            (0.0, b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'),
            (0.0, b'data: {"choices":[{"delta":{"content":"two"}}]}\n\n'),
            (0.0, b"data: [DONE]\n\n"),
        ]
        self.stub.responder = lambda path: (200, "text/event-stream", frames)
        client = self.gated_client()
        with _connected_session(client):
            response = client.post("/v1/chat/completions", content=b'{"stream":true}', headers=_auth())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream")
        self.assertEqual(response.content, b"".join(frame for _, frame in frames))

    def test_streamed_frames_arrive_as_produced(self) -> None:
        """Frames separated upstream by a delay must reach the client
        separated too — the replica forwards the stream, it does not buffer
        it. TestClient buffers whole ASGI responses, so this one runs over a
        real server, with the access window opened directly on the registry
        (the websocket wiring is covered by the TestClient tests)."""
        frames = [
            (0.0, b"data: first\n\n"),
            (0.4, b"data: second\n\n"),
        ]
        self.stub.responder = lambda path: (200, "text/event-stream", frames)
        configured_client = self.gated_client()
        registry = configured_client.app.state.dependencies.connected_llm_fingerprints
        registry.add(
            llm_token_fingerprint(SECRET, HF_TOKEN),
            "https://lb.example/event",
            "session-token",
            "session-under-test",
        )
        live = LiveApp(configured_client.app)
        self.addCleanup(live.close)

        arrivals: list[tuple[float, bytes]] = []
        with httpx.Client(timeout=10.0) as client:
            with client.stream(
                "POST",
                f"{live.base_url}/v1/chat/completions",
                content=b'{"stream":true}',
                headers=_auth(),
            ) as response:
                self.assertEqual(response.status_code, 200)
                for chunk in response.iter_raw():
                    arrivals.append((time.monotonic(), chunk))

        self.assertEqual(b"".join(chunk for _, chunk in arrivals), b"".join(frame for _, frame in frames))
        first_at = arrivals[0][0]
        last_at = arrivals[-1][0]
        self.assertGreater(last_at - first_at, 0.2, "frames arrived together: the stream was buffered")

    def test_unreachable_internal_pipeline_answers_502(self) -> None:
        self.stub.close()  # a port that was just bound and freed: connection refused
        client = self.gated_client()
        with _connected_session(client):
            response = self.post_chat(client)

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["type"], "upstream_unreachable")

        usage_events = self.wait_for_usage_events()
        self.assertEqual(len(usage_events), 1)

    def test_authorized_alternate_route_is_counted(self) -> None:
        client = self.gated_client()
        with _connected_session(client):
            response = client.post("/v1/responses", content=b"{}", headers=_auth())

        self.assertEqual(response.status_code, 200)
        usage_events = self.wait_for_usage_events()
        self.assertEqual(len(usage_events), 1)

    def test_accounting_retries_until_acknowledged(self) -> None:
        async def exercise() -> int:
            calls = 0

            async def notify(*args: Any, **kwargs: Any) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("LB unavailable")

            async def no_sleep(delay: float) -> None:
                pass

            usage = compute_main._LLMProxyUsage()
            with patch("app.compute_app.asyncio.sleep", new=no_sleep):
                usage.record(
                    "https://lb.example/event",
                    "session-token",
                    "session-1",
                    SECRET,
                    notify,
                    attempts=1,
                )
                await usage.stop()
            return calls

        self.assertEqual(asyncio.run(exercise()), 2)

    def test_accounting_worker_accepts_events_after_becoming_idle(self) -> None:
        async def exercise() -> list[int]:
            sequences: list[int] = []

            async def notify(*args: Any, **kwargs: Any) -> None:
                sequences.append(kwargs["extra_payload"]["sequence"])

            usage = compute_main._LLMProxyUsage()
            usage.record(
                "https://lb.example/event",
                "session-token",
                "session-1",
                SECRET,
                notify,
                attempts=1,
            )
            assert usage._worker is not None
            await usage._worker
            usage.record(
                "https://lb.example/event",
                "session-token",
                "session-1",
                SECRET,
                notify,
                attempts=1,
            )
            await usage.stop()
            return sequences

        self.assertEqual(asyncio.run(exercise()), [1, 2])

    def test_accounting_failure_does_not_block_proxying(self) -> None:
        client = self.gated_client()
        self.fail_usage_callback = True
        with _connected_session(client):
            started = time.monotonic()
            response = self.post_chat(client)
            elapsed = time.monotonic() - started
            self.fail_usage_callback = False

        self.assertEqual(response.status_code, 200)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(self.stub.requests), 1)

    def test_shutdown_retains_unavailable_usage_for_restart(self) -> None:
        async def exercise() -> None:
            outbox_path = str(temporary_directory / "usage.sqlite3")

            async def notify(*args: Any, **kwargs: Any) -> None:
                raise RuntimeError("LB unavailable")

            usage = compute_main._LLMProxyUsage(outbox_path=outbox_path)
            usage.record(
                "https://lb.example/event",
                "session-token",
                "session-1",
                SECRET,
                notify,
                attempts=1,
            )
            with self.assertLogs("s2s-endpoint", level="ERROR") as logs:
                await usage.stop(timeout_s=0.01)
            self.assertIn("Retaining 1 undelivered LLM usage events", logs.output[-1])

            delivered: list[int] = []

            async def recovered_notify(*args: Any, **kwargs: Any) -> None:
                delivered.append(kwargs["extra_payload"]["sequence"])

            recovered = compute_main._LLMProxyUsage(
                notify=recovered_notify,
                outbox_path=outbox_path,
            )
            instance_id = recovered.instance_id
            await recovered.start()
            recovered.record(
                "https://lb.example/event",
                "session-token",
                "session-1",
                SECRET,
                recovered_notify,
                attempts=1,
            )
            await recovered.stop()
            self.assertEqual(delivered, [1, 2])
            self.assertEqual(recovered.instance_id, instance_id)

        with tempfile.TemporaryDirectory() as directory:
            temporary_directory = Path(directory)
            asyncio.run(exercise())

    def test_only_post_is_exposed_on_proxy_paths(self) -> None:
        client = self.gated_client()
        for path in ("/v1/chat/completions", "/v1/responses"):
            self.assertEqual(client.get(path).status_code, 405)

        self.assertEqual(self.stub.requests, [])


class HfTokenGateTests(ComputeLlmProxyTestCase):
    def test_missing_api_key_is_401(self) -> None:
        client = self.gated_client()
        with _connected_session(client):
            response = self.post_chat(client, headers={"Content-Type": "application/json"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["type"], "invalid_api_key")
        self.assertEqual(self.stub.requests, [])

    def test_wrong_api_key_is_401(self) -> None:
        client = self.gated_client()
        with _connected_session(client):
            response = self.post_chat(client, headers=_auth("hf_not_the_sessions_token"))

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.stub.requests, [])

    def test_api_key_in_reachy_authorization_header_opens_the_gate(self) -> None:
        """The HF Inference Endpoints ingress consumes the standard
        Authorization header, so SDK clients on that infrastructure carry the
        token in x-reachy-mini-authorization instead."""
        client = self.gated_client()
        with _connected_session(client):
            response = self.post_chat(
                client,
                headers={
                    "x-reachy-mini-authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.stub.requests), 1)

    def test_reachy_authorization_header_wins_over_authorization(self) -> None:
        # Same precedence as the load balancer at session creation: the custom
        # header is the one that survives the ingress, so it is the one read.
        client = self.gated_client()
        with _connected_session(client):
            response = self.post_chat(
                client,
                headers={
                    "x-reachy-mini-authorization": f"Bearer {HF_TOKEN}",
                    "Authorization": "Bearer hf_not_the_sessions_token",
                    "Content-Type": "application/json",
                },
            )

        self.assertEqual(response.status_code, 200)

    def test_api_key_without_a_connected_session_is_401(self) -> None:
        client = self.gated_client()
        response = self.post_chat(client)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.stub.requests, [])

    def test_session_created_without_hf_token_carries_no_access(self) -> None:
        """A session whose token has no claim (created anonymously or with a
        junk token) never opens the LLM paths, even to the key its holder
        guesses."""
        client = self.gated_client()
        with _connected_session(client, hf_token=None):
            response = self.post_chat(client)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.stub.requests, [])

    def test_disconnect_closes_the_access_window(self) -> None:
        client = self.gated_client()
        with _connected_session(client):
            self.assertEqual(self.post_chat(client).status_code, 200)

        self.assertEqual(self.post_until_401(client).status_code, 401)

    def test_same_token_with_two_sessions_keeps_the_window_open(self) -> None:
        """The registry refcounts: closing one of a token's sessions must not
        cut off the other."""
        client = self.gated_client()
        with _connected_session(client, session_id="session-a"):
            with _connected_session(client, session_id="session-b"):
                pass
            # session-b closed; session-a still holds the window open. The
            # teardown is asynchronous, so give it a beat to run first.
            time.sleep(0.2)
            self.assertEqual(self.post_chat(client).status_code, 200)

        self.assertEqual(self.post_until_401(client).status_code, 401)

    def test_missing_shared_secret_fails_closed(self) -> None:
        client = self.gated_client(secret="")

        response = self.post_chat(client)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.stub.requests, [])


class DisabledLlmProxyTests(ComputeLlmProxyTestCase):
    def test_disabled_proxy_answers_404_even_for_a_connected_session(self) -> None:
        # An authorized key changes nothing: the check runs before auth, so a
        # disabled replica never touches the gate or the pipeline.
        client = self.gated_client(enable_llm_proxy=False)
        with _connected_session(client):
            for path in ("/v1/chat/completions", "/v1/responses"):
                response = client.post(path, content=b"{}", headers=_auth())
                self.assertEqual(response.status_code, 404)

        self.assertEqual(self.stub.requests, [])

    def test_disabled_proxy_404_is_indistinguishable_from_an_unknown_route(self) -> None:
        client = self.gated_client(enable_llm_proxy=False)

        disabled = client.post("/v1/chat/completions", content=b"{}", headers=_auth())
        unknown = client.post("/v1/does-not-exist", content=b"{}", headers=_auth())

        self.assertEqual(disabled.status_code, unknown.status_code)
        self.assertEqual(disabled.json(), unknown.json())


class FingerprintRateLimitTests(ComputeLlmProxyTestCase):
    def test_rate_limit_answers_429_and_other_users_are_unaffected(self) -> None:
        client = self.gated_client(rate_limit_rpm=2)
        with _connected_session(client, session_id="session-a"):
            with _connected_session(client, session_id="session-b", hf_token=OTHER_HF_TOKEN):
                self.assertEqual(self.post_chat(client).status_code, 200)
                self.assertEqual(self.post_chat(client).status_code, 200)
                throttled = self.post_chat(client)
                self.assertEqual(throttled.status_code, 429)
                self.assertEqual(throttled.json()["error"]["type"], "rate_limit_exceeded")
                # The other user still has their own budget.
                self.assertEqual(self.post_chat(client, headers=_auth(OTHER_HF_TOKEN)).status_code, 200)

        # The throttled request never reached the pipeline.
        self.assertEqual(len(self.stub.requests), 3)

    def test_rate_limit_window_slides_and_recovers(self) -> None:
        clock = {"now": 1000.0}
        client = self.gated_client(rate_limit_rpm=2, time_fn=lambda: clock["now"])
        with _connected_session(client):
            self.assertEqual(self.post_chat(client).status_code, 200)
            clock["now"] += 30.0
            self.assertEqual(self.post_chat(client).status_code, 200)
            self.assertEqual(self.post_chat(client).status_code, 429)
            # 61s after the first hit, one slot has slid out of the window.
            clock["now"] += 31.0
            self.assertEqual(self.post_chat(client).status_code, 200)
            self.assertEqual(self.post_chat(client).status_code, 429)

    def test_zero_rate_limit_closes_the_route(self) -> None:
        client = self.gated_client(rate_limit_rpm=0)
        with _connected_session(client):
            response = self.post_chat(client)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.stub.requests, [])

    def test_denied_requests_consume_no_budget(self) -> None:
        client = self.gated_client(rate_limit_rpm=1)
        with _connected_session(client):
            for _ in range(3):
                self.assertEqual(self.post_chat(client, headers=_auth("hf_unknown")).status_code, 401)
            self.assertEqual(self.post_chat(client).status_code, 200)


class PoolPassthroughTests(ComputeLlmProxyTestCase):
    def test_pool_passthrough_redacts_session_ids(self) -> None:
        """Session ids are private to their holders, so the replica's pool
        passthrough strips them. The LB's stuck-unit recovery only reads unit
        states and durations, which stay intact."""
        pool_payload = {
            "size": 2,
            "in_use": 2,
            "units": [
                {"index": 0, "state": "active", "session_id": "session_secret_bearer"},
                {
                    "index": 1,
                    "state": "stuck",
                    "session_id": "session_other",
                    "draining_for_s": 12.5,
                    "stuck_for_s": 3.0,
                },
            ],
        }
        self.stub.responder = lambda path: (200, pool_payload)
        client = self.gated_client()

        response = client.get("/v1/pool")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["size"], 2)
        self.assertEqual(payload["units"][0], {"index": 0, "state": "active"})
        self.assertEqual(
            payload["units"][1],
            {"index": 1, "state": "stuck", "draining_for_s": 12.5, "stuck_for_s": 3.0},
        )
        self.assertNotIn("session_secret_bearer", response.text)


class LiveApp:
    """A configured compute app on a real uvicorn server, lifespan off (the real
    lifespan would spawn a speech-to-speech process)."""

    def __init__(self, application) -> None:
        config = uvicorn.Config(
            application,
            host="127.0.0.1",
            port=0,
            log_level="error",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn test server did not start")
            time.sleep(0.01)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"

    def close(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


if __name__ == "__main__":
    unittest.main()

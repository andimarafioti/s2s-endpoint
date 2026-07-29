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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import patch

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from app import compute_main
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
def _gated_client(
    monkeypatch: Any,
    stub: StubInternalPipeline | None,
    *,
    rate_limit_rpm: int = 100,
) -> Iterator[TestClient]:
    """compute_main's app with the gate armed and the realtime path faked.

    The session router hands out fake leases and the pipeline websocket is an
    idle fake, so TestClient websocket connections exercise the real route —
    session token verification, fingerprint registration, LB notifications
    (stubbed) — without spawning a speech-to-speech process.
    """
    monkeypatch.setattr(compute_main, "SESSION_SHARED_SECRET", SECRET)
    if stub is not None:
        monkeypatch.setattr(compute_main, "INTERNAL_HTTP_BASE_URL", stub.base_url)
    monkeypatch.setattr(
        compute_main, "_connected_llm_fingerprints", compute_main._ConnectedFingerprintRegistry()
    )
    monkeypatch.setattr(
        compute_main, "_llm_rate_limiter", compute_main._FingerprintRateLimiter(rate_limit_rpm)
    )

    async def _fake_acquire():
        return SimpleNamespace(slot_id=0, ws_url="ws://127.0.0.1:1/v1/realtime")

    async def _fake_release(slot_id) -> None:
        pass

    async def _no_lb_callback(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(compute_main.session_router, "acquire", _fake_acquire)
    monkeypatch.setattr(compute_main.session_router, "release", _fake_release)
    monkeypatch.setattr(compute_main, "_notify_lb_session_event", _no_lb_callback)
    with patch("app.ws_proxy.websockets.connect", _FakeUpstreamConnect):
        # Never entered as a context manager: the app lifespan would spawn a
        # real speech-to-speech process.
        yield TestClient(compute_main.app)


@contextlib.contextmanager
def _connected_session(client: TestClient, **mint_kwargs: Any) -> Iterator[None]:
    token = _mint_session_token(**mint_kwargs)
    with client.websocket_connect(f"/v1/realtime?session_token={token}"):
        yield


def _post_until_401(client: TestClient, deadline_s: float = 5.0) -> Any:
    """Post until the access window closes: the websocket route's teardown runs
    asynchronously after the client-side close, so the 401 is eventual."""
    deadline = time.monotonic() + deadline_s
    while True:
        response = client.post("/v1/chat/completions", content=b"{}", headers=_auth())
        if response.status_code == 401 or time.monotonic() > deadline:
            return response
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Authorized passthrough
# ---------------------------------------------------------------------------


def test_chat_completions_reaches_internal_pipeline_verbatim(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        stub.responder = lambda path: (
            200,
            {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]},
        )
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client):
                response = client.post(
                    "/v1/chat/completions",
                    content=b'{"messages":[{"role":"user","content":"hi"}],"custom_field":1}',
                    headers=_auth(),
                )

        assert response.status_code == 200
        assert response.json() == {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]}
        assert len(stub.requests) == 1
        seen = stub.requests[0]
        assert seen["method"] == "POST"
        assert seen["path"] == "/v1/chat/completions"
        assert seen["body"] == b'{"messages":[{"role":"user","content":"hi"}],"custom_field":1}'
    finally:
        stub.close()


def test_api_key_is_never_forwarded_to_the_pipeline(monkeypatch: Any) -> None:
    """The api key is a user's HF token — a real credential — so it must stop
    at the replica."""
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client):
                client.post("/v1/chat/completions", content=b"{}", headers=_auth())

        seen = stub.requests[0]
        assert "authorization" not in seen["headers"]
        assert HF_TOKEN not in json.dumps(seen["headers"])
    finally:
        stub.close()


def test_responses_path_forwards_to_internal_responses(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        stub.responder = lambda path: (200, {"id": "resp_1", "output": []})
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client):
                response = client.post(
                    "/v1/responses",
                    content=b'{"input":"hello"}',
                    headers=_auth(),
                )

        assert response.status_code == 200
        assert response.json() == {"id": "resp_1", "output": []}
        assert stub.requests[0]["path"] == "/v1/responses"
    finally:
        stub.close()


@pytest.mark.parametrize(
    "status, error_type",
    [(400, "invalid_request_error"), (501, "not_implemented"), (500, "server_error")],
)
def test_pipeline_answers_pass_through_unchanged(monkeypatch: Any, status: int, error_type: str) -> None:
    """Beyond the gate, the contract clients see is the pipeline's own: the
    replica must not reinterpret its answers."""
    stub = StubInternalPipeline()
    try:
        payload = {"error": {"message": "from the pipeline", "type": error_type}}
        stub.responder = lambda path: (status, payload)
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client):
                response = client.post("/v1/chat/completions", content=b"{}", headers=_auth())

        assert response.status_code == status
        assert response.json() == payload
    finally:
        stub.close()


def test_sse_stream_passes_through_verbatim(monkeypatch: Any) -> None:
    frames = [
        (0.0, b'data: {"choices":[{"delta":{"content":"one"}}]}\n\n'),
        (0.0, b'data: {"choices":[{"delta":{"content":"two"}}]}\n\n'),
        (0.0, b"data: [DONE]\n\n"),
    ]
    stub = StubInternalPipeline()
    try:
        stub.responder = lambda path: (200, "text/event-stream", frames)
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client):
                response = client.post(
                    "/v1/chat/completions",
                    content=b'{"stream":true}',
                    headers=_auth(),
                )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        assert response.content == b"".join(frame for _, frame in frames)
    finally:
        stub.close()


def test_streamed_frames_arrive_as_produced(monkeypatch: Any) -> None:
    """Frames separated upstream by a delay must reach the client separated
    too — the replica forwards the stream, it does not buffer it. TestClient
    buffers whole ASGI responses, so this one runs over a real server, with
    the access window opened directly on the registry (the websocket wiring
    is covered by the TestClient tests)."""
    frames = [
        (0.0, b"data: first\n\n"),
        (0.4, b"data: second\n\n"),
    ]
    stub = StubInternalPipeline()
    live = LiveApp()
    try:
        stub.responder = lambda path: (200, "text/event-stream", frames)
        monkeypatch.setattr(compute_main, "INTERNAL_HTTP_BASE_URL", stub.base_url)
        monkeypatch.setattr(compute_main, "SESSION_SHARED_SECRET", SECRET)
        registry = compute_main._ConnectedFingerprintRegistry()
        registry.add(llm_token_fingerprint(SECRET, HF_TOKEN))
        monkeypatch.setattr(compute_main, "_connected_llm_fingerprints", registry)
        monkeypatch.setattr(compute_main, "_llm_rate_limiter", compute_main._FingerprintRateLimiter(100))

        arrivals: list[tuple[float, bytes]] = []
        with httpx.Client(timeout=10.0) as client:
            with client.stream(
                "POST",
                f"{live.base_url}/v1/chat/completions",
                content=b'{"stream":true}',
                headers=_auth(),
            ) as response:
                assert response.status_code == 200
                for chunk in response.iter_raw():
                    arrivals.append((time.monotonic(), chunk))

        assert b"".join(chunk for _, chunk in arrivals) == b"".join(frame for _, frame in frames)
        first_at = arrivals[0][0]
        last_at = arrivals[-1][0]
        assert last_at - first_at > 0.2, "frames arrived together: the stream was buffered"
    finally:
        live.close()
        stub.close()


def test_unreachable_internal_pipeline_answers_502(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    stub.close()  # a port that was just bound and freed: connection refused
    with _gated_client(monkeypatch, stub) as client:
        with _connected_session(client):
            response = client.post("/v1/chat/completions", content=b"{}", headers=_auth())

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unreachable"


def test_only_post_is_exposed_on_proxy_paths(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub) as client:
            for path in ("/v1/chat/completions", "/v1/responses"):
                assert client.get(path).status_code == 405

        assert stub.requests == []
    finally:
        stub.close()


# ---------------------------------------------------------------------------
# The HF token gate
# ---------------------------------------------------------------------------


def test_missing_api_key_is_401(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client):
                response = client.post(
                    "/v1/chat/completions",
                    content=b"{}",
                    headers={"Content-Type": "application/json"},
                )

        assert response.status_code == 401
        assert response.json()["error"]["type"] == "invalid_api_key"
        assert stub.requests == []
    finally:
        stub.close()


def test_wrong_api_key_is_401(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client):
                response = client.post(
                    "/v1/chat/completions",
                    content=b"{}",
                    headers=_auth("hf_not_the_sessions_token"),
                )

        assert response.status_code == 401
        assert stub.requests == []
    finally:
        stub.close()


def test_api_key_without_a_connected_session_is_401(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub) as client:
            response = client.post("/v1/chat/completions", content=b"{}", headers=_auth())

        assert response.status_code == 401
        assert stub.requests == []
    finally:
        stub.close()


def test_session_created_without_hf_token_carries_no_access(monkeypatch: Any) -> None:
    """A session whose token has no claim (created anonymously or with a junk
    token) never opens the LLM paths, even to the key its holder guesses."""
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client, hf_token=None):
                response = client.post("/v1/chat/completions", content=b"{}", headers=_auth())

        assert response.status_code == 401
        assert stub.requests == []
    finally:
        stub.close()


def test_disconnect_closes_the_access_window(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client):
                assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 200

            assert _post_until_401(client).status_code == 401
    finally:
        stub.close()


def test_same_token_with_two_sessions_keeps_the_window_open(monkeypatch: Any) -> None:
    """The registry refcounts: closing one of a token's sessions must not cut
    off the other."""
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub) as client:
            with _connected_session(client, session_id="session-a"):
                with _connected_session(client, session_id="session-b"):
                    pass
                # session-b closed; session-a still holds the window open. The
                # teardown is asynchronous, so give it a beat to run first.
                time.sleep(0.2)
                response = client.post("/v1/chat/completions", content=b"{}", headers=_auth())
                assert response.status_code == 200

            assert _post_until_401(client).status_code == 401
    finally:
        stub.close()


def test_missing_shared_secret_fails_closed(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        monkeypatch.setattr(compute_main, "SESSION_SHARED_SECRET", "")
        monkeypatch.setattr(compute_main, "INTERNAL_HTTP_BASE_URL", stub.base_url)
        client = TestClient(compute_main.app)

        response = client.post("/v1/chat/completions", content=b"{}", headers=_auth())

        assert response.status_code == 401
        assert stub.requests == []
    finally:
        stub.close()


# ---------------------------------------------------------------------------
# Rate limiting per fingerprint
# ---------------------------------------------------------------------------


def test_rate_limit_answers_429_and_other_users_are_unaffected(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub, rate_limit_rpm=2) as client:
            with _connected_session(client, session_id="session-a"):
                with _connected_session(client, session_id="session-b", hf_token=OTHER_HF_TOKEN):
                    assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 200
                    assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 200
                    throttled = client.post("/v1/chat/completions", content=b"{}", headers=_auth())
                    assert throttled.status_code == 429
                    assert throttled.json()["error"]["type"] == "rate_limit_exceeded"
                    # The other user still has their own budget.
                    other = client.post(
                        "/v1/chat/completions", content=b"{}", headers=_auth(OTHER_HF_TOKEN)
                    )
                    assert other.status_code == 200

        # The throttled request never reached the pipeline.
        assert len(stub.requests) == 3
    finally:
        stub.close()


def test_rate_limit_window_slides_and_recovers(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        clock = {"now": 1000.0}
        monkeypatch.setattr(compute_main, "_now", lambda: clock["now"])
        with _gated_client(monkeypatch, stub, rate_limit_rpm=2) as client:
            with _connected_session(client):
                assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 200
                clock["now"] += 30.0
                assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 200
                assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 429
                # 61s after the first hit, one slot has slid out of the window.
                clock["now"] += 31.0
                assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 200
                assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 429
    finally:
        stub.close()


def test_zero_rate_limit_closes_the_route(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub, rate_limit_rpm=0) as client:
            with _connected_session(client):
                response = client.post("/v1/chat/completions", content=b"{}", headers=_auth())

        assert response.status_code == 429
        assert stub.requests == []
    finally:
        stub.close()


def test_denied_requests_consume_no_budget(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        with _gated_client(monkeypatch, stub, rate_limit_rpm=1) as client:
            with _connected_session(client):
                for _ in range(3):
                    assert (
                        client.post(
                            "/v1/chat/completions", content=b"{}", headers=_auth("hf_unknown")
                        ).status_code
                        == 401
                    )
                assert client.post("/v1/chat/completions", content=b"{}", headers=_auth()).status_code == 200
    finally:
        stub.close()


# ---------------------------------------------------------------------------
# Pool passthrough
# ---------------------------------------------------------------------------


def test_pool_passthrough_redacts_session_ids(monkeypatch: Any) -> None:
    """Session ids are private to their holders, so the replica's pool
    passthrough strips them. The LB's stuck-unit recovery only reads unit
    states and durations, which stay intact."""
    stub = StubInternalPipeline()
    try:
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
        stub.responder = lambda path: (200, pool_payload)
        monkeypatch.setattr(compute_main, "INTERNAL_POOL_URL", f"{stub.base_url}/v1/pool")
        client = TestClient(compute_main.app)

        response = client.get("/v1/pool")

        assert response.status_code == 200
        payload = response.json()
        assert payload["size"] == 2
        assert payload["units"][0] == {"index": 0, "state": "active"}
        assert payload["units"][1] == {
            "index": 1,
            "state": "stuck",
            "draining_for_s": 12.5,
            "stuck_for_s": 3.0,
        }
        assert "session_secret_bearer" not in response.text
    finally:
        stub.close()


class LiveApp:
    """compute_main.app on a real uvicorn server, lifespan off (the real
    lifespan would spawn a speech-to-speech process)."""

    def __init__(self) -> None:
        config = uvicorn.Config(
            compute_main.app,
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

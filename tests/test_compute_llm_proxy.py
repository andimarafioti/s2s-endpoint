"""App-level tests for the compute replica's LLM proxy passthrough.

The replica forwards POST /v1/chat/completions and POST /v1/responses to its
internal speech-to-speech pipeline, which owns the whole contract (session
gating, rate limits, 501 reasons). These tests drive compute_main's app with
a stub internal pipeline HTTP server on an ephemeral port.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from app import compute_main


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


def _client_against(monkeypatch: Any, stub: StubInternalPipeline) -> TestClient:
    monkeypatch.setattr(compute_main, "INTERNAL_HTTP_BASE_URL", stub.base_url)
    return TestClient(compute_main.app)


def test_chat_completions_reaches_internal_pipeline_verbatim(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        stub.responder = lambda path: (
            200,
            {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]},
        )
        client = _client_against(monkeypatch, stub)

        response = client.post(
            "/v1/chat/completions",
            content=b'{"messages":[{"role":"user","content":"hi"}],"custom_field":1}',
            headers={
                "Authorization": "Bearer realtime-session-id",
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 200
        assert response.json() == {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]}
        assert len(stub.requests) == 1
        seen = stub.requests[0]
        assert seen["method"] == "POST"
        assert seen["path"] == "/v1/chat/completions"
        assert seen["headers"]["authorization"] == "Bearer realtime-session-id"
        assert seen["body"] == b'{"messages":[{"role":"user","content":"hi"}],"custom_field":1}'
    finally:
        stub.close()


def test_responses_path_forwards_to_internal_responses(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        stub.responder = lambda path: (200, {"id": "resp_1", "output": []})
        client = _client_against(monkeypatch, stub)

        response = client.post(
            "/v1/responses",
            content=b'{"input":"hello"}',
            headers={"Authorization": "Bearer sid", "Content-Type": "application/json"},
        )

        assert response.status_code == 200
        assert response.json() == {"id": "resp_1", "output": []}
        assert stub.requests[0]["path"] == "/v1/responses"
    finally:
        stub.close()


@pytest.mark.parametrize(
    "status, error_type",
    [(401, "invalid_session"), (429, "rate_limit_exceeded"), (501, "not_implemented")],
)
def test_pipeline_answers_pass_through_unchanged(monkeypatch: Any, status: int, error_type: str) -> None:
    """The contract clients see is the pipeline's own: the replica must not
    reinterpret 401/429/501 answers."""
    stub = StubInternalPipeline()
    try:
        payload = {"error": {"message": "from the pipeline", "type": error_type}}
        stub.responder = lambda path: (status, payload)
        client = _client_against(monkeypatch, stub)

        response = client.post(
            "/v1/chat/completions",
            content=b"{}",
            headers={"Authorization": "Bearer sid", "Content-Type": "application/json"},
        )

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
        client = _client_against(monkeypatch, stub)

        response = client.post(
            "/v1/chat/completions",
            content=b'{"stream":true}',
            headers={"Authorization": "Bearer sid", "Content-Type": "application/json"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        assert response.content == b"".join(frame for _, frame in frames)
    finally:
        stub.close()


def test_streamed_frames_arrive_as_produced(monkeypatch: Any) -> None:
    """Frames separated upstream by a delay must reach the client separated
    too — the replica forwards the stream, it does not buffer it. TestClient
    buffers whole ASGI responses, so this one runs over a real server."""
    frames = [
        (0.0, b"data: first\n\n"),
        (0.4, b"data: second\n\n"),
    ]
    stub = StubInternalPipeline()
    live = LiveApp()
    try:
        stub.responder = lambda path: (200, "text/event-stream", frames)
        monkeypatch.setattr(compute_main, "INTERNAL_HTTP_BASE_URL", stub.base_url)

        arrivals: list[tuple[float, bytes]] = []
        with httpx.Client(timeout=10.0) as client:
            with client.stream(
                "POST",
                f"{live.base_url}/v1/chat/completions",
                content=b'{"stream":true}',
                headers={"Authorization": "Bearer sid", "Content-Type": "application/json"},
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
    monkeypatch.setattr(compute_main, "INTERNAL_HTTP_BASE_URL", stub.base_url)
    client = TestClient(compute_main.app)

    response = client.post(
        "/v1/chat/completions",
        content=b"{}",
        headers={"Authorization": "Bearer sid", "Content-Type": "application/json"},
    )

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unreachable"


def test_only_post_is_exposed_on_proxy_paths(monkeypatch: Any) -> None:
    stub = StubInternalPipeline()
    try:
        client = _client_against(monkeypatch, stub)

        for path in ("/v1/chat/completions", "/v1/responses"):
            assert client.get(path).status_code == 405

        assert stub.requests == []
    finally:
        stub.close()


def test_pool_passthrough_redacts_session_ids(monkeypatch: Any) -> None:
    """Session ids double as bearer tokens for the LLM proxy paths, so the
    replica's pool passthrough must strip them. The LB's stuck-unit recovery
    only reads unit states and durations, which stay intact."""
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

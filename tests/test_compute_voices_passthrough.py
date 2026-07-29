import importlib
import json
import os
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import compute_main


class _StubRealtimeHandler(BaseHTTPRequestHandler):
    """Stands in for the internal speech-to-speech HTTP listener."""

    def do_GET(self):
        self._record_and_respond(body=b"")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self._record_and_respond(body=self.rfile.read(length))

    def _record_and_respond(self, body: bytes) -> None:
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "content_type": self.headers.get("Content-Type", ""),
                "body": body,
            }
        )
        status, response_body, content_type = self.server.response
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        pass


class _StubRealtimeServer(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _StubRealtimeHandler)
        self.requests: list[dict[str, object]] = []
        self.response: tuple[int, bytes, str] = (200, b"{}", "application/json")


class ComputeVoicesPassthroughTests(unittest.TestCase):
    def setUp(self):
        self.internal = _StubRealtimeServer()
        self._server_thread = threading.Thread(target=self.internal.serve_forever, daemon=True)
        self._server_thread.start()
        self.addCleanup(self._stop_internal_server)

        self.module = self._reload_with_internal_port(self.internal.server_address[1])
        self.client = TestClient(self.module.app)

    def _stop_internal_server(self):
        self.internal.shutdown()
        self.internal.server_close()
        self._server_thread.join(timeout=5)

    def _reload_with_internal_port(self, port: int):
        with patch.dict(os.environ, {"INTERNAL_WS_PORT": str(port)}, clear=True):
            module = importlib.reload(compute_main)
        self.addCleanup(importlib.reload, compute_main)
        return module

    def test_get_relays_internal_voice_list(self):
        payload = json.dumps({"voices": [{"voice_id": "ab12", "name": "Amir", "created_at": "now"}]}).encode()
        self.internal.response = (200, payload, "application/json")

        response = self.client.get("/v1/realtime/sessions/sess_abc/voices")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["voices"][0]["voice_id"], "ab12")
        self.assertEqual(self.internal.requests[0]["path"], "/v1/realtime/sessions/sess_abc/voices")

    def test_get_relays_internal_error_statuses_verbatim(self):
        for status, code in ((404, "unknown_session"), (409, "voice_cloning_unsupported")):
            with self.subTest(status=status):
                self.internal.response = (
                    status,
                    json.dumps({"error": {"message": "nope", "code": code}}).encode(),
                    "application/json",
                )

                response = self.client.get("/v1/realtime/sessions/sess_abc/voices")

                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"]["code"], code)

    def test_post_forwards_body_and_content_type(self):
        self.internal.response = (201, json.dumps({"voice_id": "ab12"}).encode(), "application/json")
        body = b"--xyz\r\nfake multipart payload\r\n--xyz--\r\n"

        response = self.client.post(
            "/v1/realtime/sessions/sess_abc/voices",
            content=body,
            headers={"Content-Type": "multipart/form-data; boundary=xyz"},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["voice_id"], "ab12")
        recorded = self.internal.requests[0]
        self.assertEqual(recorded["method"], "POST")
        self.assertEqual(recorded["body"], body)
        self.assertEqual(recorded["content_type"], "multipart/form-data; boundary=xyz")

    def test_post_rejects_oversized_upload_without_forwarding(self):
        response = self.client.post(
            "/v1/realtime/sessions/sess_abc/voices",
            content=b"0" * (self.module.VOICES_MAX_UPLOAD_BYTES + 1),
            headers={"Content-Type": "multipart/form-data; boundary=xyz"},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error"]["code"], "upload_too_large")
        self.assertEqual(self.internal.requests, [])

    def test_session_id_is_url_quoted_before_forwarding(self):
        self.internal.response = (404, b'{"error": {"message": "nope", "code": "unknown_session"}}', "application/json")

        response = self.client.get("/v1/realtime/sessions/a%20b/voices")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.internal.requests[0]["path"], "/v1/realtime/sessions/a%20b/voices")


class ComputeVoicesUnreachableInternalTests(unittest.TestCase):
    def test_unreachable_internal_server_answers_503(self):
        placeholder = socket.socket()
        placeholder.bind(("127.0.0.1", 0))
        free_port = placeholder.getsockname()[1]
        placeholder.close()

        with patch.dict(os.environ, {"INTERNAL_WS_PORT": str(free_port)}, clear=True):
            module = importlib.reload(compute_main)
        self.addCleanup(importlib.reload, compute_main)

        response = TestClient(module.app).get("/v1/realtime/sessions/sess_abc/voices")

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()

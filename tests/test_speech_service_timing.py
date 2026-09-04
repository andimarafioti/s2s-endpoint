import asyncio
import unittest

from app.speech_service_timing import SpeechServiceTimingMiddleware


class SpeechServiceTimingMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_initial_chunks_do_not_precede_response_headers(self):
        for path in ("/v1/audio/speech", "/v1/audio/transcriptions"):
            for result in (b"result", b""):
                with self.subTest(path=path, result=result):
                    sent = []

                    async def app(scope, receive, send):
                        await send({"type": "http.response.start", "status": 200, "headers": []})
                        await send({"type": "http.response.body", "body": b"", "more_body": True})
                        await send({"type": "http.response.body", "more_body": True})
                        self.assertEqual(sent, [])
                        await send({"type": "http.response.body", "body": result, "more_body": False})

                    async def send(message):
                        if message["type"] == "http.response.body":
                            self.assertTrue(sent, "body must follow response headers")
                            self.assertEqual(sent[0]["type"], "http.response.start")
                        sent.append(message)

                    middleware = SpeechServiceTimingMiddleware(app)
                    await middleware({"type": "http", "path": path, "headers": []}, lambda: None, send)

                    self.assertEqual([message["type"] for message in sent], ["http.response.start", "http.response.body"])
                    self.assertEqual(sent[1]["body"], result)
                    self.assertFalse(sent[1]["more_body"])
                    headers = dict(sent[0]["headers"])
                    self.assertIn(b"x-speech-service-latency-ms", headers)

    async def test_holds_headers_until_first_audio_and_reports_service_latency(self):
        sent = []
        first_body_ready = asyncio.Event()

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"audio/pcm")]})
            self.assertEqual(sent, [])
            first_body_ready.set()
            await send({"type": "http.response.body", "body": b"pcm", "more_body": True})
            await send({"type": "http.response.body", "body": b"audio", "more_body": False})

        middleware = SpeechServiceTimingMiddleware(app)
        scope = {
            "type": "http",
            "path": "/v1/audio/speech",
            "headers": [(b"x-speech-request-id", b"trace-123")],
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        await middleware(scope, receive, send)

        self.assertTrue(first_body_ready.is_set())
        self.assertEqual(
            [message["type"] for message in sent], ["http.response.start", "http.response.body", "http.response.body"]
        )
        headers = dict(sent[0]["headers"])
        self.assertEqual(headers[b"x-speech-request-id"], b"trace-123")
        self.assertIn(b"speech-service;dur=", headers[b"server-timing"])
        self.assertGreaterEqual(float(headers[b"x-speech-service-latency-ms"]), 0)

    async def test_ignores_untracked_health_route(self):
        sent = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        async def send(message):
            sent.append(message)

        middleware = SpeechServiceTimingMiddleware(app)
        await middleware({"type": "http", "path": "/health", "headers": []}, lambda: None, send)

        self.assertEqual(sent[0]["headers"], [])


if __name__ == "__main__":
    unittest.main()

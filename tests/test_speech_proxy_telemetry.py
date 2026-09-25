import unittest

import httpx

from app.speech_proxy_telemetry import SpeechProxyTelemetryClient, SpeechProxyTelemetryTarget


class SpeechProxyTelemetryClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_all_services_with_shared_window_and_auth(self):
        seen = []

        async def handler(request: httpx.Request):
            seen.append(request)
            service = (request.url.host or "").split(".")[0]
            return httpx.Response(200, json={"status": "ok", "service": service, "latency_ms": {}})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http_client.aclose)
        client = SpeechProxyTelemetryClient(
            (
                SpeechProxyTelemetryTarget("stt", "https://stt.example"),
                SpeechProxyTelemetryTarget("tts", "https://tts.example/"),
                SpeechProxyTelemetryTarget("llm", "https://llm.example/"),
            ),
            api_key="secret",
            client=http_client,
        )

        snapshot = await client.snapshot(3600)

        self.assertTrue(snapshot["configured"])
        self.assertEqual(set(snapshot["services"]), {"stt", "tts", "llm"})
        self.assertTrue(snapshot["services"]["stt"]["reachable"])
        self.assertEqual({request.url.path for request in seen}, {"/metrics"})
        self.assertEqual({request.url.params["window_s"] for request in seen}, {"3600"})
        self.assertEqual({request.headers["authorization"] for request in seen}, {"Bearer secret"})

    async def test_reports_one_unreachable_proxy_without_hiding_the_other(self):
        async def handler(request: httpx.Request):
            if request.url.host == "stt.example":
                raise httpx.ConnectError("offline", request=request)
            return httpx.Response(200, json={"status": "ok", "service": "tts"})

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http_client.aclose)
        client = SpeechProxyTelemetryClient(
            (
                SpeechProxyTelemetryTarget("stt", "https://stt.example"),
                SpeechProxyTelemetryTarget("tts", "https://tts.example"),
            ),
            api_key=None,
            client=http_client,
        )

        snapshot = await client.snapshot(300)

        self.assertFalse(snapshot["services"]["stt"]["reachable"])
        self.assertTrue(snapshot["services"]["tts"]["reachable"])


if __name__ == "__main__":
    unittest.main()

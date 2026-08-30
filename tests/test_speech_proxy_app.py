import asyncio
import io
import unittest
import wave

import httpx
from fastapi.testclient import TestClient

from app.speech_proxy_app import (
    SpeechProxyDependencies,
    SpeechProxySettings,
    _tts_stream,
    create_app,
)
from app.speech_proxy_router import SpeechBackendConfig, SpeechBackendPool


class AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def wav_bytes(duration_s: float = 5.0) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\0\0" * int(16_000 * duration_s))
    return output.getvalue()


def settings(service: str, count: int = 1, **overrides) -> SpeechProxySettings:
    defaults = {
        "service": service,
        "backends": tuple(
            SpeechBackendConfig(
                name=f"backend-{index}",
                url=f"https://backend-{index}.example",
            )
            for index in range(1, count + 1)
        ),
        "backend_api_key": "backend-secret",
        "target_work": 8 if service == "tts" else 96,
        "latency_target": 0.5 if service == "tts" else 0.1,
        "tts_warmup_enabled": False,
    }
    defaults.update(overrides)
    return SpeechProxySettings(**defaults)


def dependencies(proxy_settings: SpeechProxySettings, handler) -> SpeechProxyDependencies:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    pool = SpeechBackendPool(
        proxy_settings.backends,
        proxy_settings.pool_settings(),
        client=client,
    )
    return SpeechProxyDependencies(pool=pool, client=client)


class SpeechProxySettingsTests(unittest.TestCase):
    def test_service_defaults_are_independent(self):
        common = {"SPEECH_BACKENDS": "one=https://one.example", "HF_TOKEN": "hf-secret"}

        stt = SpeechProxySettings.from_env({**common, "SPEECH_PROXY_SERVICE": "stt"})
        tts = SpeechProxySettings.from_env({**common, "SPEECH_PROXY_SERVICE": "tts"})

        self.assertEqual(stt.target_work, 96)
        self.assertEqual(tts.target_work, 8)
        self.assertEqual(stt.backend_api_key, "hf-secret")
        self.assertEqual(stt.max_connections, 1024)
        self.assertEqual(stt.max_keepalive_connections, 256)

    def test_invalid_service_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be 'stt' or 'tts'"):
            SpeechProxySettings.from_env(
                {
                    "SPEECH_PROXY_SERVICE": "other",
                    "SPEECH_BACKENDS": "https://one.example",
                }
            )


class SpeechProxyApplicationTests(unittest.TestCase):
    def test_stt_forwards_openai_multipart_and_tracks_duration_weight(self):
        seen: list[tuple[str, str, str, bytes]] = []
        active_work: list[float] = []
        deps_holder: dict[str, SpeechProxyDependencies] = {}

        async def handler(request: httpx.Request):
            body = await request.aread()
            seen.append(
                (
                    request.method,
                    request.url.path,
                    request.headers.get("authorization", ""),
                    body,
                )
            )
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            snapshot = (await deps_holder["value"].pool.snapshots())[0]
            active_work.append(snapshot.active_work)
            return httpx.Response(
                200,
                json={"text": "hello"},
                headers={"x-speech-service-latency-ms": "0.010"},
            )

        proxy_settings = settings("stt")
        deps = dependencies(proxy_settings, handler)
        deps_holder["value"] = deps
        app = create_app(proxy_settings, deps)

        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                headers={"X-Speech-Request-Id": "trace-stt"},
                data={"model": "asr-model", "response_format": "json"},
                files={"file": ("audio.wav", wav_bytes(20), "audio/wav")},
            )
            health = client.get("/health").json()
            metrics = client.get("/metrics", params={"window_s": 60}).json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "hello"})
        self.assertEqual(response.headers["x-speech-request-id"], "trace-stt")
        self.assertEqual(float(response.headers["x-speech-service-latency-ms"]), 0.010)
        self.assertIn("speech-proxy;dur=", response.headers["server-timing"])
        upstream = next(item for item in seen if item[1] == "/v1/audio/transcriptions")
        self.assertEqual(upstream[2], "Bearer backend-secret")
        self.assertIn(b"asr-model", upstream[3])
        self.assertIn(b"audio.wav", upstream[3])
        self.assertEqual(active_work, [4.0])
        backend = health["backends"][0]
        self.assertEqual(backend["active_work"], 0)
        self.assertEqual(backend["successes"], 1)
        self.assertEqual(backend["requests"], 1)
        self.assertEqual(metrics["requests"]["successes"], 1)
        self.assertEqual(metrics["latency_ms"]["backend_service"]["n"], 1)
        self.assertEqual(metrics["service_timing_coverage"]["ratio"], 1)

    def test_tts_retries_another_backend_before_streaming_begins(self):
        attempts: list[str] = []

        async def handler(request: httpx.Request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            attempts.append(request.url.host or "")
            if request.url.host == "backend-1.example":
                return httpx.Response(400, json={"error": "endpoint is paused"})
            return httpx.Response(
                200,
                stream=AsyncBytes(b"pcm-", b"audio"),
                headers={"content-type": "audio/pcm"},
            )

        proxy_settings = settings("tts", count=2)
        app = create_app(proxy_settings, dependencies(proxy_settings, handler))

        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={"model": "tts-model", "voice": "aiden", "input": "Hello"},
            )
            health = client.get("/health").json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"pcm-audio")
        self.assertEqual(attempts, ["backend-1.example", "backend-2.example"])
        snapshots = {backend["name"]: backend for backend in health["backends"]}
        self.assertEqual(snapshots["backend-1"]["errors"], 1)
        self.assertEqual(snapshots["backend-2"]["successes"], 1)

    def test_tts_does_not_retry_a_client_error(self):
        attempts: list[str] = []

        async def handler(request: httpx.Request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            attempts.append(request.url.host or "")
            return httpx.Response(400, json={"error": "invalid voice"})

        proxy_settings = settings("tts", count=2)
        app = create_app(proxy_settings, dependencies(proxy_settings, handler))

        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/speech",
                json={"model": "tts-model", "voice": "missing", "input": "Hello"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(attempts, ["backend-1.example"])

    def test_app_exposes_only_the_configured_openai_route(self):
        async def handler(request: httpx.Request):
            return httpx.Response(200, json={"status": "ok"})

        stt_settings = settings("stt")
        tts_settings = settings("tts")
        stt_app = create_app(stt_settings, dependencies(stt_settings, handler))
        tts_app = create_app(tts_settings, dependencies(tts_settings, handler))

        self.assertIn("/v1/audio/transcriptions", stt_app.openapi()["paths"])
        self.assertNotIn("/v1/audio/speech", stt_app.openapi()["paths"])
        self.assertIn("/v1/audio/speech", tts_app.openapi()["paths"])
        self.assertNotIn("/v1/audio/transcriptions", tts_app.openapi()["paths"])


class _FakeResponse:
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class _FakeLease:
    def __init__(self):
        self.releases = []

    async def release(self, **kwargs):
        if not self.releases:
            self.releases.append(kwargs)


class TTSStreamLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_closing_downstream_stream_releases_backend_as_cancelled(self):
        async def rest():
            yield b"second"
            yield b"third"

        response = _FakeResponse()
        lease = _FakeLease()
        stream = _tts_stream(b"first", rest(), response, lease, 0.2)

        self.assertEqual(await anext(stream), b"first")
        await stream.aclose()

        self.assertTrue(response.closed)
        self.assertEqual(len(lease.releases), 1)
        self.assertTrue(lease.releases[0]["cancelled"])

    async def test_failure_after_first_audio_is_not_reported_as_cancellation(self):
        async def rest():
            raise httpx.ReadError("stream broke")
            yield b"unreachable"

        response = _FakeResponse()
        lease = _FakeLease()
        stream = _tts_stream(b"first", rest(), response, lease, 0.2)

        self.assertEqual(await anext(stream), b"first")
        with self.assertRaises(httpx.ReadError):
            await anext(stream)

        self.assertTrue(response.closed)
        self.assertEqual(len(lease.releases), 1)
        self.assertFalse(lease.releases[0]["success"])
        self.assertTrue(lease.releases[0]["retryable_failure"])


class ProxyCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def _cancel_inflight_request(self, service: str) -> dict:
        request_started = asyncio.Event()

        async def handler(request: httpx.Request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            request_started.set()
            await asyncio.Event().wait()

        proxy_settings = settings(service)
        deps = dependencies(proxy_settings, handler)
        await deps.pool.refresh_health()
        app = create_app(proxy_settings, deps)
        proxy_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://proxy.test",
        )
        self.addAsyncCleanup(proxy_client.aclose)
        self.addAsyncCleanup(deps.stop)

        if service == "tts":
            task = asyncio.create_task(
                proxy_client.post(
                    "/v1/audio/speech",
                    json={"model": "tts-model", "voice": "aiden", "input": "Hello"},
                )
            )
        else:
            task = asyncio.create_task(
                proxy_client.post(
                    "/v1/audio/transcriptions",
                    data={"model": "asr-model"},
                    files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
                )
            )
        await asyncio.wait_for(request_started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        return (await deps.pool.snapshots())[0].__dict__

    async def test_tts_cancellation_before_first_audio_releases_capacity(self):
        snapshot = await self._cancel_inflight_request("tts")

        self.assertEqual(snapshot["active_work"], 0)
        self.assertEqual(snapshot["cancellations"], 1)

    async def test_stt_cancellation_releases_duration_weighted_capacity(self):
        snapshot = await self._cancel_inflight_request("stt")

        self.assertEqual(snapshot["active_work"], 0)
        self.assertEqual(snapshot["cancellations"], 1)


if __name__ == "__main__":
    unittest.main()

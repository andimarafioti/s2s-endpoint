import asyncio
import gzip
import io
import json
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
        "target_work": {"stt": 96, "tts": 8, "llm": 64}[service],
        "latency_target": {"stt": 0.1, "tts": 0.5, "llm": 0.5}[service],
        "tts_warmup_enabled": False,
        "llm_warmup_enabled": False,
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
        llm = SpeechProxySettings.from_env({**common, "SPEECH_PROXY_SERVICE": "llm"})

        self.assertEqual(stt.target_work, 96)
        self.assertEqual(tts.target_work, 8)
        self.assertEqual(llm.target_work, 64)
        self.assertEqual(llm.latency_target, 0.5)
        self.assertEqual(llm.llm_warmup_model, "nvidia/Gemma-4-26B-A4B-NVFP4")
        self.assertEqual(stt.backend_api_key, "hf-secret")
        self.assertEqual(stt.max_connections, 1024)
        self.assertEqual(stt.max_keepalive_connections, 256)

    def test_invalid_service_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be 'stt', 'tts', or 'llm'"):
            SpeechProxySettings.from_env(
                {
                    "SPEECH_PROXY_SERVICE": "other",
                    "SPEECH_BACKENDS": "https://one.example",
                }
            )


class SpeechProxyApplicationTests(unittest.TestCase):
    def test_proxy_decodes_compressed_response_bodies(self):
        for service, path, content_type, content in (
            ("tts", "/v1/audio/speech", "audio/pcm", b"\x00\x01\x02\x03" * 32),
            (
                "llm",
                "/v1/chat/completions",
                "text/event-stream",
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n',
            ),
            ("llm", "/v1/responses", "application/json", b'{"id":"resp_1","output":[]}'),
        ):
            with self.subTest(service=service, path=path):
                encoded = gzip.compress(content)

                async def handler(request: httpx.Request):
                    if request.url.path == "/health":
                        return httpx.Response(200)
                    self.assertIn("gzip", request.headers["accept-encoding"])
                    return httpx.Response(
                        200,
                        # A gzip header contains no result bytes. Decoding must
                        # skip it and continue consuming subsequent chunks.
                        stream=AsyncBytes(encoded[:10], encoded[10:20], encoded[20:]),
                        headers={
                            "content-type": content_type,
                            "content-encoding": "gzip",
                            "content-length": str(len(encoded)),
                        },
                    )

                proxy_settings = settings(service)
                app = create_app(proxy_settings, dependencies(proxy_settings, handler))
                with TestClient(app) as client:
                    response = client.post(path, json={"model": "test"})
                    health = client.get("/health").json()

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, content)
                self.assertEqual(response.headers["content-type"], content_type)
                self.assertNotIn("content-encoding", response.headers)
                self.assertNotIn("content-length", response.headers)
                self.assertEqual(health["backends"][0]["active_work"], 0)
                self.assertEqual(health["backends"][0]["successes"], 1)

    def test_broken_error_body_releases_capacity_and_retries(self):
        class BrokenBody(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b'{"error":'
                raise httpx.ReadError("upstream disconnected")

            async def aclose(self):
                self.closed = True

        for service, path in (
            ("tts", "/v1/audio/speech"),
            ("llm", "/v1/chat/completions"),
            ("llm", "/v1/responses"),
        ):
            for backend_count in (1, 2):
                with self.subTest(service=service, path=path, backend_count=backend_count):
                    broken_body = BrokenBody()
                    attempts = []

                    async def handler(request: httpx.Request):
                        if request.url.path == "/health":
                            return httpx.Response(200)
                        attempts.append(request.url.host)
                        if request.url.host == "backend-1.example":
                            return httpx.Response(503, stream=broken_body)
                        return httpx.Response(200, stream=AsyncBytes(b"result"))

                    proxy_settings = settings(service, count=backend_count)
                    app = create_app(proxy_settings, dependencies(proxy_settings, handler))
                    with TestClient(app) as client:
                        response = client.post(path, json={"model": "test"})
                        health = client.get("/health").json()

                    self.assertTrue(broken_body.closed)
                    self.assertEqual(response.status_code, 200 if backend_count == 2 else 503)
                    self.assertEqual(len(attempts), backend_count)
                    failed_backend = health["backends"][0]
                    self.assertEqual(failed_backend["active_requests"], 0)
                    self.assertEqual(failed_backend["active_work"], 0)
                    self.assertEqual(failed_backend["errors"], 1)
                    if backend_count == 2:
                        self.assertEqual(response.content, b"result")
                        self.assertEqual(health["backends"][1]["successes"], 1)

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

    def test_llm_streams_tool_calls_without_changing_the_body(self):
        seen: list[tuple[str, str, str, bytes]] = []

        async def handler(request: httpx.Request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            body = await request.aread()
            seen.append(
                (
                    request.method,
                    request.url.path,
                    request.headers.get("authorization", ""),
                    body,
                )
            )
            return httpx.Response(
                200,
                stream=AsyncBytes(
                    b'data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"set_fan"}}]}}]}\n\n',
                    b"data: [DONE]\n\n",
                ),
                headers={"content-type": "text/event-stream"},
            )

        proxy_settings = settings("llm")
        app = create_app(proxy_settings, dependencies(proxy_settings, handler))
        request_body = {
            "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
            "messages": [{"role": "user", "content": "Turn on the fan"}],
            "tools": [{"type": "function", "function": {"name": "set_fan"}}],
            "stream": True,
        }

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"X-Speech-Request-Id": "trace-llm"},
                json=request_body,
            )
            metrics = client.get("/metrics", params={"window_s": 60}).json()
            health = client.get("/health").json()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'"tool_calls"', response.content)
        self.assertTrue(response.content.endswith(b"data: [DONE]\n\n"))
        self.assertEqual(response.headers["x-speech-request-id"], "trace-llm")
        self.assertEqual(seen[0][0:3], ("POST", "/v1/chat/completions", "Bearer backend-secret"))
        self.assertEqual(json.loads(seen[0][3]), request_body)
        self.assertEqual(metrics["service"], "llm")
        self.assertEqual(metrics["phase"], "first_token")
        self.assertEqual(metrics["requests"]["successes"], 1)
        self.assertEqual(health["backends"][0]["active_work"], 0)
        self.assertEqual(health["backends"][0]["successes"], 1)

    def test_llm_responses_route_is_forwarded(self):
        paths: list[str] = []

        async def handler(request: httpx.Request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "ok"})
            paths.append(request.url.path)
            return httpx.Response(
                200,
                stream=AsyncBytes(b'{"id":"resp_1","output":[]}'),
                headers={"content-type": "application/json"},
            )

        proxy_settings = settings("llm")
        app = create_app(proxy_settings, dependencies(proxy_settings, handler))

        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "nvidia/Gemma-4-26B-A4B-NVFP4", "input": "Hello"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"id": "resp_1", "output": []})
        self.assertEqual(paths, ["/v1/responses"])

    def test_app_exposes_only_the_configured_openai_route(self):
        async def handler(request: httpx.Request):
            return httpx.Response(200, json={"status": "ok"})

        stt_settings = settings("stt")
        tts_settings = settings("tts")
        llm_settings = settings("llm")
        stt_app = create_app(stt_settings, dependencies(stt_settings, handler))
        tts_app = create_app(tts_settings, dependencies(tts_settings, handler))
        llm_app = create_app(llm_settings, dependencies(llm_settings, handler))

        self.assertIn("/v1/audio/transcriptions", stt_app.openapi()["paths"])
        self.assertNotIn("/v1/audio/speech", stt_app.openapi()["paths"])
        self.assertIn("/v1/audio/speech", tts_app.openapi()["paths"])
        self.assertNotIn("/v1/audio/transcriptions", tts_app.openapi()["paths"])
        self.assertIn("/v1/chat/completions", llm_app.openapi()["paths"])
        self.assertIn("/v1/responses", llm_app.openapi()["paths"])
        self.assertNotIn("/v1/audio/speech", llm_app.openapi()["paths"])


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
    async def test_cancellation_while_reading_error_body_releases_capacity(self):
        for service, path in (
            ("tts", "/v1/audio/speech"),
            ("llm", "/v1/chat/completions"),
            ("llm", "/v1/responses"),
        ):
            with self.subTest(service=service, path=path):
                body_started = asyncio.Event()

                class StalledBody(httpx.AsyncByteStream):
                    closed = False

                    async def __aiter__(self):
                        body_started.set()
                        await asyncio.Event().wait()
                        yield b"unreachable"

                    async def aclose(self):
                        self.closed = True

                body = StalledBody()

                async def handler(request: httpx.Request):
                    if request.url.path == "/health":
                        return httpx.Response(200)
                    return httpx.Response(503, stream=body)

                proxy_settings = settings(service)
                deps = dependencies(proxy_settings, handler)
                self.addAsyncCleanup(deps.stop)
                await deps.pool.refresh_health()
                app = create_app(proxy_settings, deps)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
                ) as client:
                    task = asyncio.create_task(client.post(path, json={"model": "test"}))
                    await asyncio.wait_for(body_started.wait(), timeout=1)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

                snapshot = (await deps.pool.snapshots())[0]
                self.assertTrue(body.closed)
                self.assertEqual(snapshot.active_requests, 0)
                self.assertEqual(snapshot.active_work, 0)
                self.assertEqual(snapshot.cancellations, 1)
                self.assertEqual(snapshot.errors, 0)

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
        elif service == "stt":
            task = asyncio.create_task(
                proxy_client.post(
                    "/v1/audio/transcriptions",
                    data={"model": "asr-model"},
                    files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
                )
            )
        else:
            task = asyncio.create_task(
                proxy_client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                    },
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

    async def test_llm_cancellation_before_first_token_releases_capacity(self):
        snapshot = await self._cancel_inflight_request("llm")

        self.assertEqual(snapshot["active_work"], 0)
        self.assertEqual(snapshot["cancellations"], 1)


if __name__ == "__main__":
    unittest.main()

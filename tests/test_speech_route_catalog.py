import asyncio
import copy
import json
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.speech_proxy_app import SpeechProxySettings, create_app
from app.speech_proxy_router import SpeechPoolCapacityExceeded
from tests.test_speech_proxy_app import AsyncBytes, wav_bytes
from tests.test_speech_worker_lifecycle import Controller


def route(pool, model, *, provider="hf", kind="self_hosted", **overrides):
    value = {
        "pool": pool,
        "model": model,
        "provider": provider,
        "kind": kind,
        "revision": "test-revision",
        "upstream_model": f"deployed-{model}",
        "protocols": ["chat_completions"],
        "capabilities": {"context_window": 8192},
        "credential_env": f"KEY_{pool.upper()}",
        "backends": [{"name": pool, "url": f"https://{pool}.example"}],
        "policy": {"llm_warmup_enabled": False},
    }
    if kind == "external":
        value["capacity"] = {"max_concurrency": 2, "requests_per_minute": 60}
    value.update(overrides)
    return value


def environment(*routes, defaults=None):
    return {
        "SPEECH_PROXY_SERVICE": "llm",
        "SPEECH_ROUTE_CATALOG": json.dumps({"pools": list(routes), "defaults": defaults or {}}),
        **{r["credential_env"]: f"secret-{r['pool']}" for r in routes},
    }


def make_app(env, handler):
    real_client = httpx.AsyncClient

    def httpx_client(**kwargs):
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    with patch("app.speech_proxy_app.httpx.AsyncClient", side_effect=httpx_client):
        return create_app(SpeechProxySettings.from_env(env))


@contextmanager
def catalog_client(env, handler):
    app = make_app(env, handler)
    with TestClient(app) as client:
        yield client


class CatalogApplicationTests(unittest.TestCase):
    def test_external_models_share_api_url_with_independent_request_budgets(self):
        seen = []

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "deployed-a"}, {"id": "deployed-b"}]})
            seen.append((json.loads(request.content)["model"], request.headers["authorization"]))
            return httpx.Response(200, content=b"result")

        routes = [
            route(
                model,
                model,
                kind="external",
                provider="api",
                backends=[{"name": "api", "url": "https://api.example"}],
                capacity={"max_concurrency": 1, "requests_per_minute": 1},
            )
            for model in ("a", "b")
        ]
        with catalog_client(environment(*routes), backend) as client:
            for model, status in (("a", 200), ("a", 429), ("b", 200)):
                response = client.post("/v1/chat/completions", json={"model": model, "messages": []})
                self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(seen, [("deployed-a", "Bearer secret-a"), ("deployed-b", "Bearer secret-b")])

    def test_declared_tools_images_and_context_are_forwarded(self):
        seen = []

        async def backend(request):
            if request.method == "POST":
                seen.append(json.loads(request.content))
                return httpx.Response(200, stream=AsyncBytes(b'data: {"choices":[{"delta":{"tool_calls":[]}}]}\n\n'))
            return httpx.Response(200)

        configured = route("one", "model", capabilities={"context_window": 8192, "tools": True, "images": True})
        payload = {
            "model": "model",
            "tools": [{"type": "function", "function": {"name": "camera"}}],
            "messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]}
            ],
            "stream": True,
        }
        with catalog_client(environment(configured), backend) as client:
            response = client.post(
                "/v1/chat/completions", json=payload, headers={"X-Speech-Required-Context-Tokens": "8192"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"tool_calls", response.content)
        self.assertEqual(seen, [{**payload, "model": "deployed-model"}])

    def test_self_hosted_responses_only_route_uses_compatible_warmup(self):
        paths = []

        async def backend(request):
            if request.method == "POST":
                paths.append(request.url.path)
                self.assertEqual(json.loads(request.content)["model"], "deployed-model")
            return httpx.Response(200, content=b"result")

        configured = route("one", "model", protocols=["responses"], policy={"llm_warmup_enabled": True})
        with catalog_client(environment(configured), backend) as client:
            self.assertEqual(client.post("/v1/responses", json={"model": "model", "input": "hello"}).status_code, 200)
        self.assertEqual(paths, ["/v1/responses", "/v1/responses"])

    def test_tts_warmup_uses_declared_voice_and_audio_format(self):
        seen = []

        async def backend(request):
            if request.method == "POST":
                payload = json.loads(request.content)
                seen.append(payload)
                if payload["voice"] != "speaker" or payload["response_format"] != "wav":
                    return httpx.Response(400, content=b"unsupported voice or format")
            return httpx.Response(200, content=b"audio")

        env = environment(
            route("tts", "model", protocols=["speech"], capabilities={"voices": ["speaker"], "audio_formats": ["wav"]})
        )
        env["SPEECH_PROXY_SERVICE"] = "tts"
        with catalog_client(env, backend) as client:
            response = client.post(
                "/v1/audio/speech", json={"model": "model", "voice": "speaker", "response_format": "wav", "input": "hi"}
            )
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["model"], "deployed-model")

    def test_routes_two_models_and_external_provider_to_compatible_pools(self):
        seen = []

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "deployed-qwen"}]})
            seen.append((request.url.host, json.loads(request.content), request.headers["authorization"]))
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

        routes = (
            route("gemma", "gemma"),
            route("qwen", "qwen"),
            route("external", "qwen", provider="example-api", kind="external"),
        )
        with (
            patch("app.speech_proxy_app.HuggingFaceEndpointController") as controller,
            catalog_client(environment(*routes, defaults={"qwen": "qwen"}), backend) as client,
        ):
            for model, provider in (("gemma", None), ("qwen", "hf"), ("qwen", "example-api")):
                headers = {"X-Speech-Provider": provider} if provider else {}
                response = client.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": "Hello"}]},
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["x-speech-provider"], provider or "hf")
            controller.assert_not_called()
            metrics = client.get("/metrics").json()
            self.assertEqual(metrics["requests"]["successes"], 3)
            for pool in ("gemma", "qwen", "external"):
                self.assertEqual(metrics["pools"][pool]["requests"]["successes"], 1)
                self.assertEqual(metrics["pools"][pool]["pool"], pool)
            health = client.get("/health").json()
            self.assertEqual(health["pools"]["external"]["lifecycle"], {"enabled": False})
            self.assertNotIn("secret-", json.dumps(health) + json.dumps(metrics))
            self.assertNotIn("Hello", json.dumps(health) + json.dumps(metrics))
        self.assertEqual([entry[0] for entry in seen], ["gemma.example", "qwen.example", "external.example"])
        self.assertEqual([entry[1]["model"] for entry in seen], ["deployed-gemma", "deployed-qwen", "deployed-qwen"])
        self.assertEqual(
            [entry[2] for entry in seen], ["Bearer secret-gemma", "Bearer secret-qwen", "Bearer secret-external"]
        )

    def test_rejected_selections_and_capabilities_never_reach_inference(self):
        requests = []

        async def backend(request):
            if request.method == "POST":
                requests.append(request)
            return httpx.Response(200)

        env = environment(route("one", "model", protocols=["chat_completions", "responses"]))
        rejected = [
            ({"model": "model"}, {"X-Speech-Required-Context-Tokens": "8193"}),
            ({"model": "model"}, {"X-Speech-Required-Context-Tokens": "invalid"}),
            ({"model": "unknown"}, {}),
            ({"model": "model", "provider": "wrong"}, {}),
            ({"model": "model", "provider": "hf"}, {"X-Speech-Provider": "wrong"}),
            ({"model": "model", "tools": [{"type": "function"}]}, {}),
            ({"model": "model", "messages": [{"role": "tool", "content": "result"}]}, {}),
            (
                {
                    "model": "model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}],
                        }
                    ],
                },
                {},
            ),
            ({"model": "model", "input": [{"type": "input_image", "image_url": "x"}]}, {}),
            ({"model": "model", "input": [{"type": "input_audio"}]}, {}),
            ({"model": "model", "max_tokens": 8193}, {}),
            ({"model": "model", "max_output_tokens": True}, {}),
            ({"model": "model", "previous_response_id": "resp_prior"}, {}),
            ({"model": "model", "input": [{"type": "item_reference", "id": "prior"}]}, {}),
            ({"model": "model", "conversation": "conv_prior"}, {}),
            ({"model": "model", "prompt_cache_retention": "24h"}, {}),
            ({"model": "model", "background": True}, {}),
        ]
        with catalog_client(env, backend) as client:
            for payload, headers in rejected:
                with self.subTest(payload=payload):
                    response = client.post("/v1/responses", json=payload, headers=headers)
                    self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual(requests, [])
            self.assertEqual(client.get("/health").json()["backends"][0]["requests"], 0)

    def test_explicit_provider_failure_does_not_fall_back(self):
        seen = []

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "deployed-model"}]})
            seen.append(request.url.host)
            return httpx.Response(503 if request.url.host == "external.example" else 200, content=b"result")

        env = environment(
            route("one", "model"),
            route("external", "model", provider="api", kind="external"),
            defaults={"model": "one"},
        )
        with catalog_client(env, backend) as client:
            self.assertEqual(
                client.post("/v1/chat/completions", json={"model": "model", "provider": "api"}).status_code, 503
            )
            self.assertEqual(seen, ["external.example", "external.example"])
            self.assertEqual(client.post("/v1/chat/completions", json={"model": "model"}).status_code, 200)

    def test_application_auth_custom_header_takes_precedence_and_is_not_forwarded(self):
        seen = []

        async def backend(request):
            if request.method == "POST":
                seen.append(request)
            return httpx.Response(200, content=b"ok")

        env = {**environment(route("one", "model", access_key_env="APP_KEY")), "APP_KEY": "application-secret"}
        with catalog_client(env, backend) as client:
            for headers, expected in (
                ({}, 401),
                ({"Authorization": "Bearer application-secret"}, 200),
                ({"Authorization": "Bearer ingress", "X-Speech-Authorization": "Bearer application-secret"}, 200),
                ({"Authorization": "Bearer application-secret", "X-Speech-Authorization": "Bearer wrong"}, 401),
            ):
                response = client.post("/v1/chat/completions", json={"model": "model"}, headers=headers)
                self.assertEqual(response.status_code, expected)
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(r.headers["authorization"] == "Bearer secret-one" for r in seen))
        self.assertTrue(all("x-speech-authorization" not in r.headers for r in seen))

    def test_external_continuation_and_cache_controls_are_declared_and_forwarded(self):
        seen = []

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "deployed-model"}]})
            seen.append(json.loads(request.content))
            return httpx.Response(200, stream=AsyncBytes(b'data: {"type":"response.completed"}\n\n'))

        env = environment(
            route(
                "external",
                "model",
                kind="external",
                provider="api",
                protocols=["responses"],
                capabilities={
                    "context_window": 8192,
                    "continuation": "provider_state",
                    "cache_controls": ["prompt_cache_key"],
                },
            )
        )
        payload = {
            "model": "model",
            "provider": "api",
            "previous_response_id": "resp_prior",
            "prompt_cache_key": "cache-key",
            "input": "next turn",
            "stream": True,
        }
        with catalog_client(env, backend) as client:
            response = client.post("/v1/responses", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"response.completed", response.content)
            self.assertEqual(client.post("/v1/chat/completions", json={"model": "model"}).status_code, 400)
        self.assertEqual(seen, [{**{k: v for k, v in payload.items() if k != "provider"}, "model": "deployed-model"}])

    def test_external_429_sets_pool_cooldown_across_aliases(self):
        attempts = []

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "deployed-model"}]})
            attempts.append(request)
            return httpx.Response(429, json={"error": "quota"}, headers={"Retry-After": "30"})

        env = environment(route("external", "model", kind="external", provider="api", aliases=["canary"]))
        with catalog_client(env, backend) as client:
            first = client.post("/v1/chat/completions", json={"model": "model"})
            second = client.post("/v1/chat/completions", json={"model": "canary"})
            self.assertEqual(first.status_code, 429)
            self.assertEqual(first.headers["retry-after"], "30")
            self.assertEqual(second.status_code, 429)
            self.assertGreater(int(second.headers["retry-after"]), 0)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(client.get("/health").json()["backends"][0]["active_requests"], 0)

    def test_stt_multipart_and_tts_voice_format_use_selected_route(self):
        seen = []

        async def backend(request):
            if request.method == "POST":
                seen.append((request.url.host, request.content))
            return httpx.Response(200, content=b"result")

        stt = environment(route("stt", "transcriber", protocols=["transcriptions"], capabilities={}))
        stt["SPEECH_PROXY_SERVICE"] = "stt"
        with catalog_client(stt, backend) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                data={"model": "transcriber", "provider": "hf"},
                files={"file": ("audio.wav", wav_bytes(), "audio/wav")},
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"deployed-transcriber", seen[-1][1])
            self.assertNotIn(b'name="provider"', seen[-1][1])
        tts = environment(
            route(
                "tts",
                "speaker",
                protocols=["speech"],
                capabilities={"voices": ["aiden"], "audio_formats": ["pcm"]},
                policy={"tts_warmup_enabled": False},
            )
        )
        tts["SPEECH_PROXY_SERVICE"] = "tts"
        with catalog_client(tts, backend) as client:
            payload = {"model": "speaker", "voice": "aiden", "response_format": "pcm", "input": "hello"}
            self.assertEqual(client.post("/v1/audio/speech", json=payload).status_code, 200)
            self.assertEqual(json.loads(seen[-1][1])["model"], "deployed-speaker")
            self.assertEqual(client.post("/v1/audio/speech", json={**payload, "voice": "other"}).status_code, 400)
            self.assertEqual(
                client.post("/v1/audio/speech", json={**payload, "response_format": "mp3"}).status_code, 400
            )


class CatalogValidationTests(unittest.TestCase):
    def test_ambiguous_selections_require_defaults_or_explicit_aliases(self):
        routes = (
            route("old", "model", aliases=["stable"]),
            route("new", "model", aliases=["canary"], revision="new-revision"),
        )
        catalog = SpeechProxySettings.from_env(environment(*routes)).catalog
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            catalog.resolve("model", "hf")
        self.assertEqual(catalog.resolve("canary", None).pool, "new")
        with self.assertRaisesRegex(ValueError, "unknown"):
            catalog.resolve("canary", "wrong")
        catalog = SpeechProxySettings.from_env(environment(*routes, defaults={"model": "old"})).catalog
        self.assertEqual(catalog.resolve("model", "hf").pool, "old")

    def test_invalid_catalogs_fail_at_configuration_time(self):
        valid = route("one", "model")
        invalid = [
            [{**valid, "typo": True}],
            [{**valid, "policy": {"target_work": 0}}],
            [{**valid, "policy": {"request_timeout_s": float("nan")}}],
            [{**valid, "kind": "external"}],
            [{**valid, "capabilities": {"context_window": 8192, "continuation": "provider_state"}}],
            [{**valid, "backends": [{"name": "one", "url": "https://user:password@backend.example"}]}],
            [{**valid, "backends": [{"name": "one", "url": "https://backend.example?key=secret"}]}],
            [{**valid, "backends": [{"name": "one"}]}],
            [{**valid, "lifecycle": {"max_workers": 2}, "namespace": "test", "control_token_env": "CTRL"}],
            [valid, valid],
            [valid, {**valid, "pool": "two"}],
            [{**valid, "aliases": ["model"]}],
            [{**valid, "protocols": ["speech"]}],
            [{**valid, "capabilities": {}}],
            [route("external", "model", kind="external", lifecycle={}, provider="api")],
        ]
        for pools in invalid:
            with self.subTest(pools=pools), self.assertRaises(ValueError):
                SpeechProxySettings.from_env(environment(*pools))

    def test_duplicate_managed_endpoint_ownership_is_rejected_without_urls(self):
        a = route(
            "a",
            "model-a",
            namespace="org",
            control_token_env="CTRL",
            lifecycle={"max_workers": 1},
            backends=[{"name": "shared"}],
        )
        b = {**a, "pool": "b", "model": "model-b"}
        with self.assertRaisesRegex(ValueError, "multiple pool controllers"):
            SpeechProxySettings.from_env(environment(a, b))

    def test_missing_secret_does_not_fall_back_to_ingress_credentials(self):
        env = environment(route("one", "model"))
        del env["KEY_ONE"]
        env["HF_TOKEN"] = "ingress-secret"
        with self.assertRaisesRegex(ValueError, "KEY_ONE") as error:
            SpeechProxySettings.from_env(env)
        self.assertNotIn("ingress-secret", str(error.exception))


class CatalogIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_optional_startup_does_not_block_healthy_requests(self):
        blocked = asyncio.Event()

        async def backend(request):
            if request.url.host == "optional.example":
                blocked.set()
                await asyncio.Event().wait()
            return httpx.Response(200, content=b"result")

        app = make_app(environment(route("healthy", "a"), route("optional", "b")), backend)
        deps = app.state.dependencies
        self.addAsyncCleanup(deps.stop)
        await asyncio.wait_for(deps.start(), 1)
        await asyncio.wait_for(blocked.wait(), 1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as client:
            response = await client.post("http://proxy/v1/chat/completions", json={"model": "a"})
            self.assertEqual(response.status_code, 200)
            health = await client.get("http://proxy/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["pools"]["optional"]["ready_backends"], 0)
            self.assertEqual(
                (await client.post("http://proxy/v1/chat/completions", json={"model": "b"})).status_code, 503
            )

    async def test_concurrency_is_pool_wide_and_cancellation_releases_the_slot(self):
        entered = asyncio.Event()

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "deployed-model"}]})
            if request.url.host == "external.example":
                entered.set()
                await asyncio.Event().wait()
            return httpx.Response(200, content=b"result")

        env = environment(
            route(
                "external",
                "model",
                provider="api",
                kind="external",
                aliases=["alias"],
                capacity={"max_concurrency": 1, "requests_per_minute": 10},
            ),
            route("other", "other"),
        )
        app = make_app(env, backend)
        deps = app.state.dependencies
        self.addAsyncCleanup(deps.stop)
        await asyncio.gather(*(dep.pool.refresh_health() for dep in deps.routes.values()))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as client:
            active = asyncio.create_task(client.post("http://proxy/v1/chat/completions", json={"model": "model"}))
            await asyncio.wait_for(entered.wait(), 1)
            try:
                limited = await client.post("http://proxy/v1/chat/completions", json={"model": "alias"})
                self.assertEqual(limited.status_code, 429)
                self.assertEqual(
                    (await client.post("http://proxy/v1/chat/completions", json={"model": "other"})).status_code, 200
                )
            finally:
                active.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await active
        state = (await deps.routes["external"].pool.snapshots())[0]
        self.assertEqual(state.active_requests, 0)
        self.assertEqual(state.cancellations, 1)
        lease = await deps.routes["external"].pool.reserve(1)
        await lease.release(success=True)

    async def test_request_quota_counts_attempts_after_reservations_are_released(self):
        async def backend(request):
            return httpx.Response(200, json={"data": [{"id": "deployed-model"}]})

        app = make_app(
            environment(
                route(
                    "external",
                    "model",
                    provider="api",
                    kind="external",
                    capacity={"max_concurrency": 2, "requests_per_minute": 1},
                )
            ),
            backend,
        )
        deps = app.state.dependencies
        self.addAsyncCleanup(deps.stop)
        pool = deps.routes["external"].pool
        await pool.refresh_health()
        lease = await pool.reserve(1)
        await lease.release(success=False, cancelled=True)
        with self.assertRaises(SpeechPoolCapacityExceeded):
            await pool.reserve(1)
        pool._request_times[0] -= 61
        lease = await pool.reserve(1)
        await lease.release(success=True)

    async def test_partial_external_stream_is_not_replayed_and_releases_capacity(self):
        attempts = []

        class BrokenStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                raise httpx.ReadError("stream disconnected")

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "deployed-model"}]})
            attempts.append(request)
            return httpx.Response(200, stream=BrokenStream())

        app = make_app(environment(route("external", "model", provider="api", kind="external")), backend)
        deps = app.state.dependencies
        self.addAsyncCleanup(deps.stop)
        await deps.routes["external"].pool.refresh_health()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as client:
            with self.assertRaises(httpx.ReadError):
                await client.post("http://proxy/v1/chat/completions", json={"model": "model", "stream": True})
        self.assertEqual(len(attempts), 1)
        self.assertEqual((await deps.routes["external"].pool.snapshots())[0].active_requests, 0)

    async def test_hf_controllers_scale_and_drain_only_their_own_pool(self):
        controllers = []

        def controller(**kwargs):
            value = Controller(())
            value.statuses = {"backend-1": "running", "backend-2": "paused"}
            controllers.append(value)
            return value

        async def backend(request):
            return httpx.Response(200, content=b"result")

        a = route(
            "a",
            "model-a",
            namespace="org-a",
            control_token_env="CTRL_A",
            lifecycle={"max_workers": 2},
            policy={"target_work": 1, "llm_warmup_enabled": False},
            backends=[{"name": "backend-1"}, {"name": "backend-2"}],
        )
        b = copy.deepcopy(a)
        b.update(pool="b", model="model-b", namespace="org-b", control_token_env="CTRL_B", credential_env="KEY_B")
        b["policy"]["target_work"] = 10
        env = {**environment(a, b), "CTRL_A": "control-a", "CTRL_B": "control-b"}
        with patch("app.speech_proxy_app.HuggingFaceEndpointController", side_effect=controller):
            app = make_app(env, backend)
        deps = app.state.dependencies
        self.addAsyncCleanup(deps.stop)
        for dep in deps.routes.values():
            await dep.lifecycle.start()
            await dep.pool.refresh_health()
        self.assertEqual(len(controllers), 2)
        self.assertIsNot(deps.routes["a"].pool, deps.routes["b"].pool)
        lease = await deps.routes["a"].pool.reserve(1)
        for dep in deps.routes.values():
            await dep.lifecycle.reconcile()
            await asyncio.gather(*dep.lifecycle._operations.values())
        self.assertEqual(controllers[0].calls, [("wake", "backend-2")])
        self.assertEqual(controllers[1].calls, [])
        self.assertEqual((await deps.routes["b"].pool.snapshots())[0].active_requests, 0)
        await deps.routes["a"].pool.set_draining("backend-1", True)
        self.assertFalse((await deps.routes["b"].pool.snapshots())[0].draining)
        self.assertEqual((await deps.routes["a"].pool.snapshots())[0].active_requests, 1)
        await lease.release(success=True)

import json
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.speech_proxy_app import SpeechProxySettings, create_app


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


class CatalogApplicationTests(unittest.TestCase):
    def test_routes_two_models_and_external_provider_to_compatible_pools(self):
        seen = []

        async def backend(request):
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{"id": "deployed-qwen"}]})
            seen.append((request.url.host, json.loads(request.content), request.headers["authorization"]))
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

        httpx_client = lambda **kwargs: real_client(transport=httpx.MockTransport(backend), **kwargs)
        real_client = httpx.AsyncClient
        routes = (
            route("gemma", "gemma"),
            route("qwen", "qwen"),
            route("external", "qwen", provider="example-api", kind="external"),
        )
        settings = SpeechProxySettings.from_env(environment(*routes, defaults={"qwen": "qwen"}))
        with patch("app.speech_proxy_app.httpx.AsyncClient", side_effect=lambda **kw: httpx_client(**kw)):
            app = create_app(settings)
        with TestClient(app) as client:
            for model, provider in (("gemma", None), ("qwen", "hf"), ("qwen", "example-api")):
                headers = {"X-Speech-Provider": provider} if provider else {}
                response = client.post(
                    "/v1/chat/completions",
                    headers=headers,
                    json={"model": model, "messages": [{"role": "user", "content": "Hello"}]},
                )
                self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([entry[0] for entry in seen], ["gemma.example", "qwen.example", "external.example"])
        self.assertEqual([entry[1]["model"] for entry in seen], ["deployed-gemma", "deployed-qwen", "deployed-qwen"])
        self.assertEqual([entry[2] for entry in seen], ["Bearer secret-gemma", "Bearer secret-qwen", "Bearer secret-external"])



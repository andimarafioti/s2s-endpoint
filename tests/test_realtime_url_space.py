from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient


SPACE_APP_PATH = Path(__file__).resolve().parents[1] / "spaces" / "reachy-mini-realtime-url" / "app.py"


def _load_space_app_module() -> Any:
    spec = importlib.util.spec_from_file_location("reachy_mini_realtime_url_space_app", SPACE_APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Space app from {SPACE_APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_health_does_not_require_upstream_url(monkeypatch: Any) -> None:
    module = _load_space_app_module()
    for env_name in module.UPSTREAM_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)

    client = TestClient(module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["configured"] is False


def test_ready_requires_upstream_url(monkeypatch: Any) -> None:
    module = _load_space_app_module()
    for env_name in module.UPSTREAM_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)

    client = TestClient(module.app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["error"] == "upstream_session_url_not_configured"


def test_session_url_returns_configured_upstream(monkeypatch: Any) -> None:
    module = _load_space_app_module()
    monkeypatch.setenv("UPSTREAM_SESSION_URL", "https://allocator.example.test/session")

    client = TestClient(module.app)

    response = client.get("/session-url")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {"session_url": "https://allocator.example.test/session"}


def test_session_rejects_invalid_upstream_url(monkeypatch: Any) -> None:
    module = _load_space_app_module()
    monkeypatch.setenv("UPSTREAM_SESSION_URL", "ws://allocator.example.test/session")

    client = TestClient(module.app)

    response = client.post("/session")

    assert response.status_code == 503
    assert response.json()["error"] == "invalid_upstream_session_url"


def test_session_proxies_upstream_response(monkeypatch: Any) -> None:
    module = _load_space_app_module()
    monkeypatch.setenv("UPSTREAM_SESSION_URL", "https://allocator.example.test/session")
    captured_request: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            captured_request["timeout"] = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
            captured_request["url"] = url
            captured_request["content"] = content
            captured_request["headers"] = headers
            return httpx.Response(
                200,
                content=b'{"connect_url":"wss://compute.example.test/v1/realtime?session_token=abc"}',
                headers={"content-type": "application/json"},
            )

    monkeypatch.setattr(module.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(module.app)

    response = client.post(
        "/session",
        content=b'{"client":"reachy-mini"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json() == {"connect_url": "wss://compute.example.test/v1/realtime?session_token=abc"}
    assert captured_request["timeout"] == module.DEFAULT_TIMEOUT_SECONDS
    assert captured_request["url"] == "https://allocator.example.test/session"
    assert captured_request["content"] == b'{"client":"reachy-mini"}'
    assert captured_request["headers"]["Content-Type"] == "application/json"
    assert captured_request["headers"]["X-Reachy-Mini-Realtime-URL"] == "1"

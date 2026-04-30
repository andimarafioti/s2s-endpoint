from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


SERVICE_NAME = "reachy-mini-realtime-url"
UPSTREAM_ENV_NAMES = ("UPSTREAM_SESSION_URL", "REALTIME_SESSION_URL", "HF_REALTIME_SESSION_URL")
DEFAULT_TIMEOUT_SECONDS = 10.0

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="Reachy Mini Realtime URL", version="0.1.0")


@dataclass(frozen=True)
class UpstreamConfig:
    session_url: str
    timeout_s: float


def _timeout_seconds() -> float:
    raw_timeout = os.getenv("REQUEST_TIMEOUT_SECONDS", "").strip()
    if not raw_timeout:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except ValueError:
        logger.warning("Invalid REQUEST_TIMEOUT_SECONDS=%r; using default %.1f", raw_timeout, DEFAULT_TIMEOUT_SECONDS)
        return DEFAULT_TIMEOUT_SECONDS
    return max(0.1, timeout)


def _configured_session_url() -> str | None:
    for env_name in UPSTREAM_ENV_NAMES:
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def _validate_session_url(session_url: str) -> str:
    parsed = urlsplit(session_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("upstream session URL must be an absolute http(s) URL")
    return session_url


def _upstream_config() -> UpstreamConfig | None:
    session_url = _configured_session_url()
    if session_url is None:
        return None
    return UpstreamConfig(session_url=_validate_session_url(session_url), timeout_s=_timeout_seconds())


def _no_store_headers() -> dict[str, str]:
    return {"Cache-Control": "no-store"}


def _not_configured_response() -> JSONResponse:
    return JSONResponse(
        {
            "ok": False,
            "error": "upstream_session_url_not_configured",
            "expected_env": list(UPSTREAM_ENV_NAMES),
        },
        status_code=503,
        headers=_no_store_headers(),
    )


def _invalid_config_response(exc: ValueError) -> JSONResponse:
    return JSONResponse(
        {
            "ok": False,
            "error": "invalid_upstream_session_url",
            "detail": str(exc),
        },
        status_code=503,
        headers=_no_store_headers(),
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": SERVICE_NAME,
            "health": "/health",
            "ready": "/ready",
            "session": "/session",
            "session_url": "/session-url",
        },
        headers=_no_store_headers(),
    )


@app.get("/health")
async def health() -> JSONResponse:
    try:
        configured = _upstream_config() is not None
    except ValueError:
        configured = False
    return JSONResponse({"ok": True, "service": SERVICE_NAME, "configured": configured}, headers=_no_store_headers())


@app.get("/ready")
async def ready() -> JSONResponse:
    try:
        config = _upstream_config()
    except ValueError as exc:
        return _invalid_config_response(exc)
    if config is None:
        return _not_configured_response()
    return JSONResponse({"ok": True, "configured": True}, headers=_no_store_headers())


@app.get("/session-url")
async def session_url() -> JSONResponse:
    try:
        config = _upstream_config()
    except ValueError as exc:
        return _invalid_config_response(exc)
    if config is None:
        return _not_configured_response()
    return JSONResponse({"session_url": config.session_url}, headers=_no_store_headers())


@app.get("/config")
async def config() -> JSONResponse:
    return await session_url()


@app.post("/session")
async def create_session(request: Request) -> Response:
    try:
        config = _upstream_config()
    except ValueError as exc:
        return _invalid_config_response(exc)
    if config is None:
        return _not_configured_response()

    body = await request.body()
    headers = {"X-Reachy-Mini-Realtime-URL": "1"}
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type
    accept = request.headers.get("accept")
    if accept:
        headers["Accept"] = accept

    try:
        async with httpx.AsyncClient(timeout=config.timeout_s) as client:
            upstream_response = await client.post(config.session_url, content=body, headers=headers)
    except httpx.TimeoutException:
        logger.warning("Timed out while calling upstream session URL")
        return JSONResponse(
            {"ok": False, "error": "upstream_timeout"},
            status_code=504,
            headers=_no_store_headers(),
        )
    except httpx.HTTPError as exc:
        logger.warning("Failed to call upstream session URL: %s", exc)
        return JSONResponse(
            {"ok": False, "error": "upstream_unreachable"},
            status_code=502,
            headers=_no_store_headers(),
        )

    media_type = upstream_response.headers.get("content-type") or "application/json"
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=media_type,
        headers=_no_store_headers(),
    )

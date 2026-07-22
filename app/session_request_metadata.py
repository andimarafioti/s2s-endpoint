from __future__ import annotations

import asyncio
import json

from starlette.requests import ClientDisconnect, Request

from app.requester_identity import normalize_hardware_id


MAX_SESSION_METADATA_BODY_BYTES = 4096
SESSION_METADATA_READ_TIMEOUT_S = 1.0


async def reported_hardware_id(
    request: Request,
    *,
    max_body_bytes: int = MAX_SESSION_METADATA_BODY_BYTES,
    read_timeout_s: float = SESSION_METADATA_READ_TIMEOUT_S,
) -> str | None:
    content_type = str(request.headers.get("content-type") or "").lower()
    if content_type.partition(";")[0].strip() != "application/json":
        return None

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            return None
        if declared_size < 0 or declared_size > max_body_bytes:
            return None

    try:
        body = await asyncio.wait_for(
            _read_bounded_body(request, max_body_bytes=max_body_bytes),
            timeout=read_timeout_s,
        )
    except (TimeoutError, ClientDisconnect, RuntimeError):
        return None
    if body is None:
        return None

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    return normalize_hardware_id(payload.get("hardware_id"))


async def _read_bounded_body(request: Request, *, max_body_bytes: int) -> bytes | None:
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_body_bytes:
            return None
        body.extend(chunk)
    return bytes(body)

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

logger = logging.getLogger("speech-service-timing")
TRACKED_PATHS = frozenset({"/v1/audio/transcriptions", "/v1/audio/speech"})
REQUEST_ID_HEADER = b"x-speech-request-id"
SERVICE_LATENCY_HEADER = b"x-speech-service-latency-ms"
SERVER_TIMING_HEADER = b"server-timing"

ASGIApp = Callable[
    [
        MutableMapping[str, Any],
        Callable[[], Awaitable[MutableMapping[str, Any]]],
        Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ],
    Awaitable[None],
]


def _request_id(scope: MutableMapping[str, Any]) -> str:
    for name, value in scope.get("headers", []):
        if name.lower() == REQUEST_ID_HEADER:
            decoded = value.decode("ascii", errors="ignore").strip()
            decoded = "".join(character for character in decoded if character.isalnum() or character in "-_.")[:128]
            if decoded:
                return decoded
    return uuid.uuid4().hex


def _append_header(headers: list[tuple[bytes, bytes]], name: bytes, value: bytes) -> None:
    for index, (existing_name, existing_value) in enumerate(headers):
        if existing_name.lower() == name:
            separator = b", " if existing_value else b""
            headers[index] = (existing_name, existing_value + separator + value)
            return
    headers.append((name, value))


class SpeechServiceTimingMiddleware:
    """Report ASGI arrival-to-first-result latency from the GPU service.

    Response headers are held until the first non-empty body chunk so TTS can
    report time to first audio rather than the earlier time to HTTP headers.
    STT reports time to its JSON transcription body. The first body is sent
    immediately after the enriched headers, so this does not delay the result.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http" or scope.get("path") not in TRACKED_PATHS:
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        request_id = _request_id(scope)
        response_start: MutableMapping[str, Any] | None = None
        start_sent = False

        async def timed_send(message: MutableMapping[str, Any]) -> None:
            nonlocal response_start, start_sent
            message_type = message.get("type")
            if message_type == "http.response.start":
                response_start = dict(message)
                response_start["headers"] = list(message.get("headers", []))
                return

            if message_type == "http.response.body" and response_start is not None and not start_sent:
                body = message.get("body", b"")
                if not body and message.get("more_body", False):
                    # Empty intermediate chunks carry no result and cannot be
                    # forwarded while http.response.start is still held.
                    return
                latency_ms = max((time.perf_counter() - started) * 1000.0, 0.0)
                headers = response_start["headers"]
                _append_header(headers, REQUEST_ID_HEADER, request_id.encode("ascii"))
                _append_header(headers, SERVICE_LATENCY_HEADER, f"{latency_ms:.3f}".encode("ascii"))
                _append_header(
                    headers,
                    SERVER_TIMING_HEADER,
                    f"speech-service;dur={latency_ms:.3f}".encode("ascii"),
                )
                await send(response_start)
                start_sent = True
                logger.info(
                    "Speech service result request_id=%s path=%s status=%s service_latency_ms=%.3f",
                    request_id,
                    scope.get("path"),
                    response_start.get("status"),
                    latency_ms,
                )

            await send(message)

        await self.app(scope, receive, timed_send)

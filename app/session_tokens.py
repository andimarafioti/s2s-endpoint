import base64
import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def create_session_token(
    secret: str,
    *,
    session_id: str,
    websocket_url: str,
    callback_url: str,
    ttl_s: float,
) -> str:
    expires_at = int(time.time() + ttl_s)
    payload = {
        "callback_url": callback_url,
        "exp": expires_at,
        "sid": session_id,
        "ws_url": websocket_url,
    }
    encoded_payload = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = _sign(secret, encoded_payload)
    return f"{encoded_payload}.{signature}"


def verify_session_token(token: str, secret: str) -> dict[str, Any]:
    try:
        encoded_payload, signature = token.split(".", 1)
    except ValueError as exc:
        raise ValueError("malformed session token") from exc

    expected_signature = _sign(secret, encoded_payload)
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("invalid session token signature")

    try:
        payload = json.loads(_b64decode(encoded_payload))
    except Exception as exc:
        raise ValueError("invalid session token payload") from exc

    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("session token expired")

    return payload


def attach_session_token(websocket_url: str, session_token: str) -> str:
    parsed = urlparse(websocket_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["session_token"] = session_token
    return urlunparse(parsed._replace(query=urlencode(query)))


def websocket_host_matches(expected_websocket_url: str, actual_host: str | None) -> bool:
    if not actual_host:
        return False
    return urlparse(expected_websocket_url).netloc == actual_host.strip().lower()


def _sign(secret: str, encoded_payload: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), encoded_payload.encode("utf-8"), hashlib.sha256).digest()
    return _b64encode(digest)


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)

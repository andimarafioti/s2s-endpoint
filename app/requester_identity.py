from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Optional


logger = logging.getLogger("s2s-endpoint")
IdentityUpdateHandler = Callable[["RequesterIdentity"], Awaitable[None]]
WhoAmIFunction = Callable[[str], dict[str, object]]


@dataclass(frozen=True)
class RequesterIdentity:
    actor_id: str
    label: str
    kind: str
    verification: str
    fingerprint: str
    account_name: Optional[str] = None
    network_id: Optional[str] = None
    client_kind: str = "unknown"

    def with_request_context(self, *, network_id: Optional[str], client_kind: str) -> "RequesterIdentity":
        return replace(self, network_id=network_id, client_kind=client_kind)

    def history_metadata(self) -> dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "verification": self.verification,
            "fingerprint": self.fingerprint,
            "account_name": self.account_name,
            "network_id": self.network_id,
            "client_kind": self.client_kind,
        }


@dataclass(frozen=True)
class _CachedIdentity:
    identity: RequesterIdentity
    expires_at_s: float


class RequesterIdentityResolver:
    """Build privacy-preserving requester identities and resolve HF accounts in the background."""

    def __init__(
        self,
        *,
        hash_secret: str | None,
        on_identity_update: Optional[IdentityUpdateHandler] = None,
        whoami_fn: Optional[WhoAmIFunction] = None,
        trust_proxy_headers: bool = True,
        max_pending_validations: int = 128,
        validation_concurrency: int = 4,
        cache_size: int = 4096,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        if max_pending_validations < 1:
            raise ValueError("max_pending_validations must be >= 1")
        if validation_concurrency < 1:
            raise ValueError("validation_concurrency must be >= 1")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")

        configured_secret = (hash_secret or "").strip()
        self.stable_fingerprints = bool(configured_secret)
        self._hash_key = (
            configured_secret.encode("utf-8")
            if configured_secret
            else secrets.token_bytes(32)
        )
        if not self.stable_fingerprints:
            logger.warning(
                "REQUEST_USAGE_HASH_SECRET and SESSION_SHARED_SECRET are unset; "
                "requester fingerprints will change when the load balancer restarts"
            )

        self._on_identity_update = on_identity_update
        self._whoami_fn = whoami_fn or _default_whoami
        self._trust_proxy_headers = trust_proxy_headers
        self._max_pending_validations = max_pending_validations
        self._validation_semaphore = asyncio.Semaphore(validation_concurrency)
        self._cache_size = cache_size
        self._time_fn = time_fn
        self._cache: "OrderedDict[str, _CachedIdentity]" = OrderedDict()
        self._validation_tasks: dict[str, asyncio.Task[None]] = {}

    def set_identity_update_handler(self, handler: IdentityUpdateHandler) -> None:
        self._on_identity_update = handler

    def identify(self, request: object) -> RequesterIdentity:
        network_id = self._network_id(request)
        client_kind = _client_kind(_header(request, "user-agent"))
        token = bearer_token(_header(request, "authorization"))

        if token is None:
            fingerprint = network_id.removeprefix("net:") if network_id else "unknown"
            identity = RequesterIdentity(
                actor_id=f"anonymous:{fingerprint}",
                label=(
                    f"Anonymous IP •{fingerprint[:8]}"
                    if fingerprint != "unknown"
                    else "Anonymous / IP unavailable"
                ),
                kind="anonymous",
                verification="not_provided",
                fingerprint=fingerprint,
            )
            return identity.with_request_context(network_id=network_id, client_kind=client_kind)

        fingerprint = self._fingerprint("token", token)
        actor_id = f"token:{fingerprint}"
        cached = self._cached_identity(actor_id)
        if cached is not None:
            return cached.with_request_context(network_id=network_id, client_kind=client_kind)

        identity = RequesterIdentity(
            actor_id=actor_id,
            label=f"HF token •{fingerprint[:8]}",
            kind="unverified_token",
            verification="pending" if _looks_like_hf_token(token) else "unrecognized",
            fingerprint=fingerprint,
        )
        if _looks_like_hf_token(token) and not self._schedule_validation(token, identity):
            identity = replace(identity, verification="unavailable")
        return identity.with_request_context(network_id=network_id, client_kind=client_kind)

    def latest_identity(self, identity: RequesterIdentity) -> RequesterIdentity:
        cached = self._cached_identity(identity.actor_id)
        if cached is None:
            return identity
        return cached.with_request_context(
            network_id=identity.network_id,
            client_kind=identity.client_kind,
        )

    async def stop(self) -> None:
        tasks = list(self._validation_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._validation_tasks.clear()

    def status(self) -> dict[str, object]:
        return {
            "stable_fingerprints": self.stable_fingerprints,
            "cached_tokens": len(self._cache),
            "pending_token_validations": len(self._validation_tasks),
            "trust_proxy_headers": self._trust_proxy_headers,
        }

    def _network_id(self, request: object) -> Optional[str]:
        address = client_address(request, trust_proxy_headers=self._trust_proxy_headers)
        if not address:
            return None
        return f"net:{self._fingerprint('network', address)}"

    def _fingerprint(self, namespace: str, value: str) -> str:
        digest = hmac.new(
            self._hash_key,
            f"{namespace}\0{value}".encode("utf-8", errors="replace"),
            hashlib.sha256,
        ).hexdigest()
        return digest[:16]

    def _cached_identity(self, actor_id: str) -> Optional[RequesterIdentity]:
        cached = self._cache.get(actor_id)
        if cached is None:
            return None
        if cached.expires_at_s <= self._time_fn():
            self._cache.pop(actor_id, None)
            return None
        self._cache.move_to_end(actor_id)
        return cached.identity

    def _schedule_validation(self, token: str, identity: RequesterIdentity) -> bool:
        if identity.actor_id in self._validation_tasks:
            return True
        if len(self._validation_tasks) >= self._max_pending_validations:
            return False

        task = asyncio.create_task(self._validate_token(token, identity))
        self._validation_tasks[identity.actor_id] = task
        task.add_done_callback(
            lambda completed, actor_id=identity.actor_id: self._validation_finished(actor_id, completed)
        )
        return True

    def _validation_finished(self, actor_id: str, task: asyncio.Task[None]) -> None:
        if self._validation_tasks.get(actor_id) is task:
            self._validation_tasks.pop(actor_id, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception:
            logger.exception("Unexpected requester token validation failure actor_id=%s", actor_id)

    async def _validate_token(self, token: str, identity: RequesterIdentity) -> None:
        try:
            async with self._validation_semaphore:
                whoami = await asyncio.to_thread(self._whoami_fn, token)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {401, 403}:
                resolved = replace(
                    identity,
                    label=f"Invalid token •{identity.fingerprint[:8]}",
                    kind="invalid_token",
                    verification="invalid",
                )
                ttl_s = 15 * 60
            else:
                resolved = replace(identity, verification="unavailable")
                ttl_s = 60
                logger.warning(
                    "HF requester identity lookup unavailable actor_id=%s status_code=%s error_type=%s",
                    identity.actor_id,
                    status_code,
                    type(exc).__name__,
                )
        else:
            account_name = _safe_text(whoami.get("name"), max_length=80)
            if account_name:
                label = f"@{account_name} · token •{identity.fingerprint[:8]}"
            else:
                label = f"Verified HF token •{identity.fingerprint[:8]}"
            resolved = replace(
                identity,
                label=label,
                kind="authenticated",
                verification="verified",
                account_name=account_name,
            )
            ttl_s = 24 * 60 * 60

        self._cache_identity(resolved, ttl_s=ttl_s)
        if self._on_identity_update is not None:
            await self._on_identity_update(resolved)

    def _cache_identity(self, identity: RequesterIdentity, *, ttl_s: float) -> None:
        self._cache[identity.actor_id] = _CachedIdentity(
            identity=identity,
            expires_at_s=self._time_fn() + ttl_s,
        )
        self._cache.move_to_end(identity.actor_id)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def bearer_token(authorization: str | None) -> str | None:
    scheme, separator, token = (authorization or "").partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def client_address(request: object, *, trust_proxy_headers: bool) -> Optional[str]:
    candidate: Optional[str] = None
    if trust_proxy_headers:
        forwarded_for = _header(request, "x-forwarded-for")
        if forwarded_for:
            candidate = forwarded_for.split(",", 1)[0].strip().strip('"')
        if not candidate:
            candidate = (_header(request, "x-real-ip") or "").strip().strip('"') or None

    if not candidate:
        client = getattr(request, "client", None)
        candidate = str(getattr(client, "host", "") or "").strip() or None
    if not candidate:
        return None
    return _normalize_address(candidate)


def _normalize_address(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and "]" in value:
        value = value[1 : value.index("]")]
    else:
        host, separator, port = value.rpartition(":")
        if separator and port.isdigit() and host.count(":") == 0:
            value = host
    try:
        return ipaddress.ip_address(value).compressed
    except ValueError:
        return value.lower()[:128]


def _header(request: object, name: str) -> Optional[str]:
    headers = getattr(request, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value) if value is not None else None


def _looks_like_hf_token(token: str) -> bool:
    return token.startswith("hf_") and 8 <= len(token) <= 512


def _safe_text(value: object, *, max_length: int) -> Optional[str]:
    if value is None:
        return None
    sanitized = "".join(character for character in str(value).strip() if character.isprintable())
    return sanitized[:max_length] or None


def _client_kind(user_agent: str | None) -> str:
    value = (user_agent or "").strip().lower()
    if not value:
        return "missing-user-agent"

    automation_markers = (
        ("python-httpx", "automation:httpx"),
        ("python-requests", "automation:python-requests"),
        ("aiohttp", "automation:aiohttp"),
        ("curl/", "automation:curl"),
        ("wget/", "automation:wget"),
        ("postmanruntime", "automation:postman"),
        ("insomnia/", "automation:insomnia"),
        ("headless", "automation:headless"),
        ("bot", "automation:bot"),
        ("crawler", "automation:crawler"),
        ("spider", "automation:spider"),
    )
    for marker, label in automation_markers:
        if marker in value:
            return label
    if "mozilla/" in value:
        return "browser"
    if "okhttp/" in value or "dart/" in value:
        return "mobile-app"
    return "other"


def _default_whoami(token: str) -> dict[str, object]:
    from huggingface_hub import HfApi

    return HfApi().whoami(token=token, cache=False)

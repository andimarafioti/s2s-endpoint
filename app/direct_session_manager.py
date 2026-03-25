import asyncio
import contextlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from app.endpoint_pool_router import EndpointLease, EndpointPoolRouter
from app.session_tokens import attach_session_token, create_session_token, verify_session_token


logger = logging.getLogger("s2s-endpoint")


@dataclass
class DirectSession:
    session_id: str
    lease: EndpointLease
    session_token: str
    pending_expires_at: Optional[float]
    connected: bool = False


class DirectSessionManager:
    def __init__(
        self,
        *,
        endpoint_router: EndpointPoolRouter,
        session_shared_secret: str,
        pending_timeout_s: float = 60.0,
        session_token_ttl_s: float = 86400.0,
        reap_interval_s: float = 5.0,
        allocate_timeout_s: float = 900.0,
    ) -> None:
        if not session_shared_secret:
            raise ValueError("session_shared_secret must be set")

        self.endpoint_router = endpoint_router
        self.session_shared_secret = session_shared_secret
        self.pending_timeout_s = pending_timeout_s
        self.session_token_ttl_s = session_token_ttl_s
        self.reap_interval_s = reap_interval_s
        self.allocate_timeout_s = allocate_timeout_s

        self._lock = asyncio.Lock()
        self._sessions: dict[str, DirectSession] = {}
        self._reaper_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self.endpoint_router.start()
        self._reaper_task = asyncio.create_task(self._reap_loop())

    async def stop(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None

        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            await self.endpoint_router.release(session.lease.slot_id)

        await self.endpoint_router.stop()

    async def allocate(self, lb_base_url: str) -> dict[str, object]:
        lease = await self.endpoint_router.acquire(timeout_s=self.allocate_timeout_s)
        session_id = secrets.token_urlsafe(18)
        callback_url = _build_callback_url(lb_base_url, session_id)
        session_token = create_session_token(
            self.session_shared_secret,
            session_id=session_id,
            websocket_url=lease.ws_url,
            callback_url=callback_url,
            ttl_s=self.session_token_ttl_s,
        )
        pending_expires_at = time.monotonic() + self.pending_timeout_s

        session = DirectSession(
            session_id=session_id,
            lease=lease,
            session_token=session_token,
            pending_expires_at=pending_expires_at,
        )

        async with self._lock:
            self._sessions[session_id] = session

        return {
            "session_id": session_id,
            "websocket_url": lease.ws_url,
            "connect_url": attach_session_token(lease.ws_url, session_token),
            "session_token": session_token,
            "pending_timeout_s": self.pending_timeout_s,
        }

    async def handle_event(self, session_id: str, session_token: str, event: str) -> dict[str, object]:
        payload = verify_session_token(session_token, self.session_shared_secret)
        if payload.get("sid") != session_id:
            raise ValueError("session token does not match session id")

        if event not in {"connected", "disconnected"}:
            raise ValueError("event must be 'connected' or 'disconnected'")

        session_to_release: Optional[DirectSession] = None

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("unknown session id")
            if session.lease.ws_url != payload.get("ws_url"):
                raise ValueError("session token does not match reserved endpoint")

            if event == "connected":
                session.connected = True
                session.pending_expires_at = None
                return {
                    "status": "ok",
                    "session_id": session_id,
                    "state": "connected",
                }

            session_to_release = self._sessions.pop(session_id)

        await self.endpoint_router.release(session_to_release.lease.slot_id)
        return {
            "status": "ok",
            "session_id": session_id,
            "state": "released",
        }

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            sessions = list(self._sessions.values())

        pending_sessions = sum(1 for session in sessions if not session.connected)
        connected_sessions = sum(1 for session in sessions if session.connected)

        return {
            "pending_sessions": pending_sessions,
            "connected_sessions": connected_sessions,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "endpoint_name": session.lease.endpoint_name,
                    "connected": session.connected,
                    "pending_expires_at_monotonic": session.pending_expires_at,
                }
                for session in sorted(sessions, key=lambda item: item.session_id)
            ],
        }

    async def healthcheck(self) -> tuple[bool, Optional[str], dict[str, object]]:
        healthy, detail, router_snapshot = await self.endpoint_router.healthcheck()
        snapshot = await self.snapshot()
        snapshot["router"] = router_snapshot
        return healthy, detail, snapshot

    async def _reap_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.reap_interval_s)
                await self._release_expired_pending_sessions()
        except asyncio.CancelledError:
            raise

    async def _release_expired_pending_sessions(self) -> None:
        now = time.monotonic()
        expired: list[DirectSession] = []

        async with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.connected or session.pending_expires_at is None:
                    continue
                if session.pending_expires_at > now:
                    continue
                expired.append(self._sessions.pop(session_id))

        for session in expired:
            await self.endpoint_router.release(session.lease.slot_id)
            logger.info(
                "Released expired pending session %s for endpoint %s",
                session.session_id,
                session.lease.endpoint_name,
            )


def _build_callback_url(lb_base_url: str, session_id: str) -> str:
    return f"{lb_base_url.rstrip('/')}/internal/sessions/{session_id}/event"

import asyncio
import contextlib
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from app.endpoint_pool_router import EndpointLease, EndpointPoolRouter
from app.session_tokens import attach_session_token, create_session_token, verify_session_token


logger = logging.getLogger("s2s-endpoint")
SessionReleaseHandler = Callable[[dict[str, object]], Awaitable[None]]


@dataclass
class DirectSession:
    session_id: str
    lease: EndpointLease
    session_token: str
    pending_expires_at: Optional[float]
    allocated_at_monotonic: float
    connected: bool = False
    connected_at_monotonic: Optional[float] = None


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
        self.endpoint_router._on_endpoint_down = self._release_sessions_for_endpoint
        self.session_shared_secret = session_shared_secret
        self.pending_timeout_s = pending_timeout_s
        self.session_token_ttl_s = session_token_ttl_s
        self.reap_interval_s = reap_interval_s
        self.allocate_timeout_s = allocate_timeout_s

        self._lock = asyncio.Lock()
        self._sessions: dict[str, DirectSession] = {}
        self._reaper_task: Optional[asyncio.Task] = None
        self._abnormal_disconnect_handler: Optional[SessionReleaseHandler] = None

    def set_abnormal_disconnect_handler(self, handler: Optional[SessionReleaseHandler]) -> None:
        self._abnormal_disconnect_handler = handler

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
            await self.endpoint_router.release(session.lease.slot_id, connected=session.connected)

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
            allocated_at_monotonic=time.monotonic(),
        )

        try:
            async with self._lock:
                self._sessions[session_id] = session
        except BaseException:
            await self.endpoint_router.release(lease.slot_id, connected=False)
            raise

        return {
            "session_id": session_id,
            "websocket_url": lease.ws_url,
            "connect_url": attach_session_token(lease.ws_url, session_token),
            "session_token": session_token,
            "pending_timeout_s": self.pending_timeout_s,
        }

    async def cancel_pending_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.connected:
                return
            self._sessions.pop(session_id)
        await self.endpoint_router.release(session.lease.slot_id, connected=False)
        logger.info("Released abandoned pending session %s (client disconnected before response)", session_id)

    async def handle_event(self, session_id: str, session_token: str, event: str) -> dict[str, object]:
        payload = verify_session_token(session_token, self.session_shared_secret)
        if payload.get("sid") != session_id:
            raise ValueError("session token does not match session id")

        if event not in {"connected", "disconnected"}:
            raise ValueError("event must be 'connected' or 'disconnected'")

        session_to_release: Optional[DirectSession] = None
        connected_session: Optional[DirectSession] = None

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("unknown session id")
            if session.lease.ws_url != payload.get("ws_url"):
                raise ValueError("session token does not match reserved endpoint")

            if event == "connected":
                was_connected = session.connected
                session.connected = True
                session.pending_expires_at = None
                if session.connected_at_monotonic is None:
                    session.connected_at_monotonic = time.monotonic()
                if not was_connected:
                    connected_session = session
            else:
                session_to_release = self._sessions.pop(session_id)

        if connected_session is not None:
            await self.endpoint_router.mark_connected(connected_session.lease.slot_id)
            return {
                "status": "ok",
                "session_id": session_id,
                "state": "connected",
            }

        if event == "connected":
            return {
                "status": "ok",
                "session_id": session_id,
                "state": "connected",
            }

        assert session_to_release is not None
        await self.endpoint_router.release(
            session_to_release.lease.slot_id,
            connected=session_to_release.connected,
        )
        return self._release_result(session_to_release, release_reason="client_disconnected")

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
                    "connected_at_monotonic": session.connected_at_monotonic,
                    "connected_duration_s": (
                        max(time.monotonic() - session.connected_at_monotonic, 0.0)
                        if session.connected_at_monotonic is not None
                        else None
                    ),
                }
                for session in sorted(sessions, key=lambda item: item.session_id)
            ],
        }

    async def healthcheck(self) -> tuple[bool, Optional[str], dict[str, object]]:
        healthy, detail, router_snapshot = await self.endpoint_router.healthcheck()
        snapshot = await self.snapshot()
        router_active_sessions = int(router_snapshot.get("active_sessions", 0))
        pending_sessions = int(snapshot.get("pending_sessions", 0))
        observed_connected_sessions = max(router_active_sessions - pending_sessions, 0)
        snapshot["connected_sessions"] = max(
            int(snapshot.get("connected_sessions", 0)),
            observed_connected_sessions,
        )
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
            await self.endpoint_router.release(session.lease.slot_id, connected=False)
            logger.info(
                "Released expired pending session %s for endpoint %s",
                session.session_id,
                session.lease.endpoint_name,
            )

    async def _release_sessions_for_endpoint(self, endpoint_name: str) -> None:
        to_release: list[DirectSession] = []

        async with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.lease.endpoint_name == endpoint_name:
                    to_release.append(self._sessions.pop(session_id))

        for session in to_release:
            await self.endpoint_router.release(session.lease.slot_id, connected=session.connected)
            result = self._release_result(session, release_reason="endpoint_unavailable")
            logger.info(
                "Released session %s for downed endpoint %s (connected=%s)",
                session.session_id,
                endpoint_name,
                session.connected,
            )
            if session.connected:
                await self._record_abnormal_disconnect(result)

    def _release_result(self, session: DirectSession, *, release_reason: str) -> dict[str, object]:
        conversation_duration_s = 0.0
        if session.connected_at_monotonic is not None:
            conversation_duration_s = max(
                time.monotonic() - session.connected_at_monotonic,
                0.0,
            )
        return {
            "status": "ok",
            "session_id": session.session_id,
            "state": "released",
            "event": "disconnected",
            "release_reason": release_reason,
            "conversation_counted": session.connected_at_monotonic is not None,
            "conversation_duration_s": conversation_duration_s,
        }

    async def _record_abnormal_disconnect(self, result: dict[str, object]) -> None:
        if self._abnormal_disconnect_handler is None:
            return
        try:
            await self._abnormal_disconnect_handler(result)
        except Exception:
            logger.exception(
                "Failed to record abnormal disconnect for session %s",
                result.get("session_id"),
            )


def _build_callback_url(lb_base_url: str, session_id: str) -> str:
    return f"{lb_base_url.rstrip('/')}/internal/sessions/{session_id}/event"

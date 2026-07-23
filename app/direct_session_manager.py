import asyncio
import contextlib
import logging
import secrets
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import Awaitable, Callable, Optional

from app.app_utils import elapsed_ms, http_base_url_from_ws_url
from app.endpoint_pool_router import EndpointLease, EndpointPoolRouter
from app.session_tokens import attach_session_token, create_session_token, verify_session_token


logger = logging.getLogger("s2s-endpoint")
SessionReleaseHandler = Callable[[dict[str, object]], Awaitable[None]]
TicketExpiredHandler = Callable[[str], Awaitable[None]]


class QueueAtCapacityError(RuntimeError):
    """Raised when the waiting queue is full, so a new caller can't even join."""


@dataclass
class QueueTicket:
    """A held place in the waiting line. Not a session — only a promise of a spot.

    ``last_seen`` is refreshed on every poll; a ticket that goes un-polled past
    the TTL is how we detect a caller who abandoned the queue."""
    ticket_id: str
    created_at: float
    last_seen: float


@dataclass
class DirectSession:
    session_id: str
    lease: EndpointLease
    session_token: str
    pending_expires_at: Optional[float]
    allocated_at_monotonic: float
    allocation_wait_ms: int
    waited_for_capacity: bool
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
        # No default on purpose: every construction site must consciously pick a
        # /session contract — queueless blocking (as before the queue existed) or
        # ticket-and-poll. The env default in load_balancer_main is off.
        queue_enabled: bool,
        queue_max_depth: int = 100,
        queue_ticket_ttl_s: float = 8.0,
        queue_poll_interval_s: float = 2.0,
        queue_reap_interval_s: float = 2.0,
    ) -> None:
        if not session_shared_secret:
            raise ValueError("session_shared_secret must be set")
        if queue_max_depth < 0:
            raise ValueError("queue_max_depth must be >= 0")

        self.endpoint_router = endpoint_router
        self.endpoint_router._on_endpoint_down = self._release_sessions_for_endpoint
        self.session_shared_secret = session_shared_secret
        self.pending_timeout_s = pending_timeout_s
        self.session_token_ttl_s = session_token_ttl_s
        self.reap_interval_s = reap_interval_s
        self.allocate_timeout_s = allocate_timeout_s
        self.queue_enabled = queue_enabled
        self.queue_max_depth = queue_max_depth
        self.queue_ticket_ttl_s = queue_ticket_ttl_s
        self.queue_poll_interval_s = queue_poll_interval_s
        self.queue_reap_interval_s = queue_reap_interval_s

        self._lock = asyncio.Lock()
        self._sessions: dict[str, DirectSession] = {}
        # Insertion order == arrival order == FIFO admission order.
        self._queue: "OrderedDict[str, QueueTicket]" = OrderedDict()
        self._reaper_task: Optional[asyncio.Task] = None
        self._ticket_reaper_task: Optional[asyncio.Task] = None
        self._abnormal_disconnect_handler: Optional[SessionReleaseHandler] = None
        self._ticket_expired_handler: Optional[TicketExpiredHandler] = None

    def set_abnormal_disconnect_handler(self, handler: Optional[SessionReleaseHandler]) -> None:
        self._abnormal_disconnect_handler = handler

    def set_ticket_expired_handler(self, handler: Optional[TicketExpiredHandler]) -> None:
        """Called with each ticket_id the reaper drops for going un-polled past the
        TTL — the caller's chance to record the abandoned request."""
        self._ticket_expired_handler = handler

    async def start(self) -> None:
        await self.endpoint_router.start()
        self._reaper_task = asyncio.create_task(self._reap_loop())
        if self.queue_enabled:
            self._ticket_reaper_task = asyncio.create_task(self._ticket_reap_loop())

    async def stop(self) -> None:
        for task_attr in ("_reaper_task", "_ticket_reaper_task"):
            task = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                setattr(self, task_attr, None)

        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._queue.clear()

        for session in sessions:
            await self.endpoint_router.release(session.lease.slot_id, connected=session.connected)

        await self.endpoint_router.stop()

    async def allocate(self, lb_base_url: str) -> dict[str, object]:
        """Grant a session if a slot is free *and* nobody is waiting; otherwise
        mint a queue ticket. Never blocks — the waiting lives in the queue, polled
        via ``poll``. Raises ``QueueAtCapacityError`` when the queue itself is full.

        With the queue disabled this is the pre-queue contract instead: block until
        a slot frees (up to ``allocate_timeout_s``, then
        ``EndpointCapacityTimeoutError``) and never return a ticket."""
        started_at = monotonic()
        if not self.queue_enabled:
            lease = await self.endpoint_router.acquire(timeout_s=self.allocate_timeout_s)
        else:
            lease = None
            # The empty-line check and the slot grab must be one atomic step, or two
            # concurrent callers could both see an empty queue and both fast-path into
            # the same freed capacity, jumping the line. Holding the lock across
            # ``try_acquire`` is safe: it only touches the router's own lock and never
            # calls back into this manager, and it never waits for capacity.
            async with self._lock:
                # FIFO: only fast-path a grant when the line is empty. If anyone is
                # already waiting, a fresh caller joins the back — no queue-jumping.
                if not self._queue:
                    lease = await self.endpoint_router.try_acquire()
                if lease is None:
                    # No slot free (or someone already waiting): join the queue. A
                    # depth of 0 disables the waiting room entirely — every caller
                    # that can't be granted immediately is turned away at capacity.
                    if len(self._queue) >= self.queue_max_depth:
                        raise QueueAtCapacityError(
                            f"queue is full ({self.queue_max_depth} waiting)"
                        )
                    now = monotonic()
                    ticket_id = secrets.token_urlsafe(18)
                    self._queue[ticket_id] = QueueTicket(ticket_id, created_at=now, last_seen=now)
                    position = len(self._queue)  # just appended, so it's last in line

            if lease is None:
                logger.info(
                    "Queued session request ticket_id=%s position=%d queue_depth=%d",
                    ticket_id,
                    position,
                    position,
                    extra={"ticket_id": ticket_id, "position": position, "outcome": "queued"},
                )
                return self._ticket_view(ticket_id, position)

        granted_at = monotonic()
        return await self._grant_from_lease(
            lease,
            lb_base_url,
            allocated_at=granted_at,
            allocation_wait_ms=elapsed_ms(started_at, granted_at),
            waited_for_capacity=lease.waited_for_capacity,
        )

    async def poll(self, ticket_id: str, lb_base_url: str) -> dict[str, object]:
        """Advance a waiting ticket. Refreshes its last-seen, reports position, and
        — only for the head of the line — claims a free slot if one is available,
        returning a grant. Raises ``KeyError`` for an unknown/expired ticket."""
        now = monotonic()
        lease: Optional[EndpointLease] = None
        # Hold the lock across the whole claim — the head check, the slot grab, and
        # the pop — so two overlapping polls for the same head ticket can't each
        # grant a session. The loser finds the ticket already gone and 404s. Safe
        # to await ``try_acquire`` here: it only takes the router's own lock.
        async with self._lock:
            ticket = self._queue.get(ticket_id)
            if ticket is None:
                raise KeyError("unknown or expired ticket")
            ticket.last_seen = now
            created_at = ticket.created_at
            position = list(self._queue).index(ticket_id) + 1

            if position == 1:
                lease = await self.endpoint_router.try_acquire()
                if lease is not None:
                    self._queue.pop(ticket_id, None)

        if lease is not None:
            logger.info(
                "Claimed slot for ticket_id=%s wait_ms=%d",
                ticket_id,
                elapsed_ms(created_at, now),
                extra={"ticket_id": ticket_id, "outcome": "claimed"},
            )
            try:
                return await self._grant_from_lease(
                    lease,
                    lb_base_url,
                    allocated_at=now,
                    allocation_wait_ms=elapsed_ms(created_at, now),
                    waited_for_capacity=True,
                )
            except BaseException:
                # The ticket was popped optimistically under the lock; a grant
                # failure must not evict the head of the line after its whole
                # wait. Put it back in front so the next poll retries.
                async with self._lock:
                    if ticket_id not in self._queue:
                        self._queue[ticket_id] = ticket
                        self._queue.move_to_end(ticket_id, last=False)
                raise

        return self._ticket_view(ticket_id, position)

    async def leave(self, ticket_id: str) -> bool:
        """Drop a waiting ticket (explicit "leave the queue" / teardown beacon).
        Returns True if a ticket was actually removed."""
        async with self._lock:
            removed = self._queue.pop(ticket_id, None) is not None
        if removed:
            logger.info(
                "Ticket %s left the queue",
                ticket_id,
                extra={"ticket_id": ticket_id, "outcome": "left"},
            )
        return removed

    def _ticket_view(self, ticket_id: str, position: int) -> dict[str, object]:
        return {
            "state": "queued",
            "queue_id": ticket_id,
            "position": position,
            "poll_interval_s": self.queue_poll_interval_s,
            "ticket_ttl_s": self.queue_ticket_ttl_s,
        }

    async def _grant_from_lease(
        self,
        lease: EndpointLease,
        lb_base_url: str,
        *,
        allocated_at: float,
        allocation_wait_ms: int = 0,
        waited_for_capacity: bool = False,
    ) -> dict[str, object]:
        session_id = secrets.token_urlsafe(18)
        callback_url = _build_callback_url(lb_base_url, session_id)
        session_token = create_session_token(
            self.session_shared_secret,
            session_id=session_id,
            websocket_url=lease.ws_url,
            callback_url=callback_url,
            ttl_s=self.session_token_ttl_s,
        )
        pending_expires_at = allocated_at + self.pending_timeout_s

        session = DirectSession(
            session_id=session_id,
            lease=lease,
            session_token=session_token,
            pending_expires_at=pending_expires_at,
            allocated_at_monotonic=allocated_at,
            allocation_wait_ms=allocation_wait_ms,
            waited_for_capacity=waited_for_capacity,
        )

        try:
            async with self._lock:
                self._sessions[session_id] = session
        except BaseException:
            await self.endpoint_router.release(lease.slot_id, connected=False)
            raise

        return {
            "state": "granted",
            "session_id": session_id,
            "websocket_url": lease.ws_url,
            # HTTP origin of the replica that owns the session, for the LLM
            # proxy paths it serves next to the websocket.
            "http_base_url": http_base_url_from_ws_url(lease.ws_url),
            "connect_url": attach_session_token(lease.ws_url, session_token),
            "session_token": session_token,
            "pending_timeout_s": self.pending_timeout_s,
            "endpoint_name": lease.endpoint_name,
            "slot_id": lease.slot_id,
            "allocation_wait_ms": allocation_wait_ms,
            "waited_for_capacity": waited_for_capacity,
        }

    async def cancel_pending_session(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.connected:
                return
            self._sessions.pop(session_id)
        await self.endpoint_router.release(session.lease.slot_id, connected=False)
        logger.info(
            "Released abandoned pending session %s for endpoint %s slot_id=%s "
            "allocation_wait_ms=%d waited_for_capacity=%s "
            "(client disconnected before response)",
            session_id,
            session.lease.endpoint_name,
            session.lease.slot_id,
            session.allocation_wait_ms,
            session.waited_for_capacity,
            extra=_session_log_extra(session, outcome="pending_released"),
        )

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
                    session.connected_at_monotonic = monotonic()
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
            queued_sessions = len(self._queue)

        pending_sessions = sum(1 for session in sessions if not session.connected)
        connected_sessions = sum(1 for session in sessions if session.connected)

        return {
            "pending_sessions": pending_sessions,
            "connected_sessions": connected_sessions,
            "queued_sessions": queued_sessions,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "endpoint_name": session.lease.endpoint_name,
                    "connected": session.connected,
                    "pending_expires_at_monotonic": session.pending_expires_at,
                    "connected_at_monotonic": session.connected_at_monotonic,
                    "connected_duration_s": (
                        max(monotonic() - session.connected_at_monotonic, 0.0)
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

    async def _ticket_reap_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.queue_reap_interval_s)
                await self._reap_stale_tickets()
        except asyncio.CancelledError:
            raise

    async def _reap_stale_tickets(self) -> None:
        """Drop tickets that haven't been polled within the TTL — the signal that a
        waiter abandoned the queue. Removing a ticket shifts everyone behind it up,
        and lets the head-of-line self-heal if the leader vanished."""
        now = monotonic()
        dropped: list[str] = []
        async with self._lock:
            for ticket_id, ticket in list(self._queue.items()):
                if now - ticket.last_seen > self.queue_ticket_ttl_s:
                    self._queue.pop(ticket_id, None)
                    dropped.append(ticket_id)

        for ticket_id in dropped:
            if self._ticket_expired_handler is not None:
                try:
                    await self._ticket_expired_handler(ticket_id)
                except Exception:
                    logger.exception(
                        "Ticket-expired handler failed for ticket %s", ticket_id
                    )
            logger.info(
                "Dropped abandoned queue ticket %s (no poll within %.0fs TTL)",
                ticket_id,
                self.queue_ticket_ttl_s,
                extra={"ticket_id": ticket_id, "outcome": "ticket_expired"},
            )

    async def _release_expired_pending_sessions(self) -> None:
        now = monotonic()
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
                "Released expired pending session %s for endpoint %s slot_id=%s "
                "allocation_wait_ms=%d waited_for_capacity=%s",
                session.session_id,
                session.lease.endpoint_name,
                session.lease.slot_id,
                session.allocation_wait_ms,
                session.waited_for_capacity,
                extra=_session_log_extra(session, outcome="pending_expired"),
            )

    async def _release_sessions_for_endpoint(self, endpoint_name: str) -> None:
        to_release: list[DirectSession] = []

        async with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.lease.endpoint_name == endpoint_name:
                    to_release.append(self._sessions.pop(session_id))

        for session in to_release:
            # Keep slot cleanup independent from best-effort dashboard accounting.
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
                monotonic() - session.connected_at_monotonic,
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


def _session_log_extra(session: DirectSession, *, outcome: str) -> dict[str, object]:
    return {
        "session_id": session.session_id,
        "endpoint_name": session.lease.endpoint_name,
        "slot_id": session.lease.slot_id,
        "allocation_wait_ms": session.allocation_wait_ms,
        "outcome": outcome,
        "waited_for_capacity": session.waited_for_capacity,
    }

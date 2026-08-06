from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from app.requester_identity import RequesterIdentity


@dataclass(frozen=True)
class _SessionRequesterEntry:
    requester: RequesterIdentity
    expires_at_s: float


class SessionRequesterTracker:
    """Temporarily associate an allocated session with its privacy-safe requester."""

    def __init__(
        self,
        *,
        retention_s: float,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if retention_s <= 0:
            raise ValueError("retention_s must be > 0")
        self.retention_s = retention_s
        self._time_fn = time_fn
        self._entries: dict[str, _SessionRequesterEntry] = {}

    def remember(self, session_id: str, requester: RequesterIdentity) -> None:
        now = self._time_fn()
        self._prune(now)
        self._entries[session_id] = _SessionRequesterEntry(
            requester=requester,
            expires_at_s=now + self.retention_s,
        )

    def take(self, session_id: str) -> RequesterIdentity | None:
        requester, expired = self.take_with_expiry(session_id)
        return None if expired else requester

    def get_with_expiry(self, session_id: str) -> tuple[RequesterIdentity | None, bool]:
        """Inspect an entry without consuming a still-valid authorization context."""
        now = self._time_fn()
        entry = self._entries.get(session_id)
        self._prune(now)
        if entry is None:
            return None, False
        return entry.requester, entry.expires_at_s <= now

    def take_with_expiry(self, session_id: str) -> tuple[RequesterIdentity | None, bool]:
        """Remove an entry while still exposing whether it had expired.

        Security-sensitive callers must not use an expired identity to authorize
        work, but may still need it to release accounting state associated with
        the original request. Pop the requested entry before pruning the rest so
        that cleanup remains possible without extending its authorization life.
        """
        now = self._time_fn()
        entry = self._entries.pop(session_id, None)
        self._prune(now)
        if entry is None:
            return None, False
        return entry.requester, entry.expires_at_s <= now

    def discard(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def count(self) -> int:
        self._prune(self._time_fn())
        return len(self._entries)

    def _prune(self, now_s: float) -> None:
        expired = [session_id for session_id, entry in self._entries.items() if entry.expires_at_s <= now_s]
        for session_id in expired:
            self._entries.pop(session_id, None)

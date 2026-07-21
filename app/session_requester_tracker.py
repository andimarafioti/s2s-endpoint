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
        self._prune(self._time_fn())
        entry = self._entries.pop(session_id, None)
        return entry.requester if entry is not None else None

    def discard(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def count(self) -> int:
        self._prune(self._time_fn())
        return len(self._entries)

    def _prune(self, now_s: float) -> None:
        expired = [
            session_id
            for session_id, entry in self._entries.items()
            if entry.expires_at_s <= now_s
        ]
        for session_id in expired:
            self._entries.pop(session_id, None)

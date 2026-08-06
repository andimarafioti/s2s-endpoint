from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationAdmissionConfig:
    max_global_pending: int = 64
    max_network_pending: int = 4

    def __post_init__(self) -> None:
        if self.max_global_pending < 1:
            raise ValueError("max_global_pending must be >= 1")
        if self.max_network_pending < 1:
            raise ValueError("max_network_pending must be >= 1")


@dataclass(frozen=True)
class VerificationAdmissionDecision:
    allowed: bool
    reason: str | None
    retry_after_s: int


class VerificationAdmissionPermit:
    def __init__(self, limiter: "VerificationAdmissionLimiter", network_id: str) -> None:
        self._limiter = limiter
        self._network_id = network_id
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._limiter._release(self._network_id)

    def release_when_done(self, task: asyncio.Task[None]) -> None:
        task.add_done_callback(lambda _completed: self.release())


class VerificationAdmissionLimiter:
    """Bound new remote token checks before they enter the resolver's work queue."""

    def __init__(self, *, config: VerificationAdmissionConfig) -> None:
        self.config = config
        self._global_pending = 0
        self._network_pending: Counter[str] = Counter()
        self._totals: Counter[str] = Counter()

    def acquire(
        self,
        network_id: str | None,
    ) -> tuple[VerificationAdmissionDecision, VerificationAdmissionPermit | None]:
        resolved_network_id = network_id or "network:unknown"
        if self._network_pending[resolved_network_id] >= self.config.max_network_pending:
            self._totals["network_rejections"] += 1
            return VerificationAdmissionDecision(False, "network_quota", 1), None
        if self._global_pending >= self.config.max_global_pending:
            self._totals["global_rejections"] += 1
            return VerificationAdmissionDecision(False, "global_quota", 1), None

        self._global_pending += 1
        self._network_pending[resolved_network_id] += 1
        self._totals["admitted"] += 1
        return (
            VerificationAdmissionDecision(True, None, 0),
            VerificationAdmissionPermit(self, resolved_network_id),
        )

    def status(self) -> dict[str, object]:
        return {
            "pending": self._global_pending,
            "pending_networks": len(self._network_pending),
            "limits": {
                "max_global_pending": self.config.max_global_pending,
                "max_network_pending": self.config.max_network_pending,
            },
            "totals": dict(sorted(self._totals.items())),
        }

    def _release(self, network_id: str) -> None:
        self._global_pending = max(self._global_pending - 1, 0)
        pending = self._network_pending.get(network_id, 0)
        if pending <= 1:
            self._network_pending.pop(network_id, None)
        else:
            self._network_pending[network_id] = pending - 1

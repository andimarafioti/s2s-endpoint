from __future__ import annotations

import math
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque

from app.requester_identity import RequesterIdentity


@dataclass(frozen=True)
class RequesterRateLimitConfig:
    enabled: bool = True
    request_window_s: float = 60.0
    max_requests_per_window: int = 20
    max_parallel_allocations: int = 2
    max_consecutive_no_connects: int = 3
    short_session_threshold_s: float = 10.0
    max_consecutive_short_sessions: int = 8
    cooldown_s: float = 15 * 60.0
    actor_retention_s: float = 60 * 60.0
    max_actor_states: int = 10_000

    def __post_init__(self) -> None:
        positive_values = {
            "request_window_s": self.request_window_s,
            "max_requests_per_window": self.max_requests_per_window,
            "max_parallel_allocations": self.max_parallel_allocations,
            "max_consecutive_no_connects": self.max_consecutive_no_connects,
            "short_session_threshold_s": self.short_session_threshold_s,
            "max_consecutive_short_sessions": self.max_consecutive_short_sessions,
            "cooldown_s": self.cooldown_s,
            "actor_retention_s": self.actor_retention_s,
            "max_actor_states": self.max_actor_states,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    reason: str | None
    retry_after_s: int | None
    actor_id: str
    recent_requests: int
    active_allocations: int
    consecutive_no_connects: int
    consecutive_short_sessions: int


@dataclass(frozen=True)
class RequesterSessionOutcome:
    requester: RequesterIdentity
    connected: bool
    duration_s: float | None
    short_session: bool
    no_connect: bool


@dataclass
class _TrackedAllocation:
    requester: RequesterIdentity
    expires_at_s: float
    connected_at_s: float | None = None


@dataclass
class _ActorState:
    requester: RequesterIdentity
    last_seen_s: float
    request_times_s: Deque[float] = field(default_factory=deque)
    in_flight_allocations: int = 0
    allocations: dict[str, _TrackedAllocation] = field(default_factory=dict)
    consecutive_no_connects: int = 0
    consecutive_short_sessions: int = 0
    blocked_until_s: float = 0.0


class RequesterRateLimiter:
    """Bound requester capacity using allocation outcomes instead of User-Agent bans.

    Calls are synchronous so a check and its in-flight reservation happen without
    another asyncio task interleaving between them. The load balancer must release
    that reservation through ``record_allocation`` or ``record_allocation_failure``.
    """

    def __init__(
        self,
        *,
        config: RequesterRateLimitConfig | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or RequesterRateLimitConfig()
        self._time_fn = time_fn
        self._actors: "OrderedDict[str, _ActorState]" = OrderedDict()
        self._session_actors: dict[str, str] = {}
        self._totals: Counter[str] = Counter()
        self._rejection_reasons: Counter[str] = Counter()
        self._last_prune_s = self._time_fn()

    def acquire(self, requester: RequesterIdentity) -> RateLimitDecision:
        """Record a request and reserve one allocation permit when it is allowed."""

        now_s = self._time_fn()
        self._totals["requests"] += 1
        self._prune_if_due(now_s)
        state = self._state_for(requester, now_s)
        if state is None:
            if self.config.enabled:
                return self._reject_without_state(requester, reason="tracker_capacity")
            self._totals["allowed"] += 1
            return RateLimitDecision(
                allowed=True,
                reason=None,
                retry_after_s=None,
                actor_id=requester.actor_id,
                recent_requests=0,
                active_allocations=0,
                consecutive_no_connects=0,
                consecutive_short_sessions=0,
            )

        self._expire_pending_allocations(state, now_s)
        self._prune_request_window(state, now_s)
        state.request_times_s.append(now_s)
        state.requester = requester
        state.last_seen_s = now_s
        self._actors.move_to_end(requester.actor_id)

        reason: str | None = None
        retry_after_s: int | None = None
        if state.blocked_until_s > now_s:
            reason = "behavior_cooldown"
            retry_after_s = _retry_after(state.blocked_until_s - now_s)
        elif len(state.request_times_s) > self.config.max_requests_per_window:
            reason = "request_rate"
            retry_after_s = _retry_after(
                state.request_times_s[0] + self.config.request_window_s - now_s
            )
        elif self._active_allocations(state) >= self.config.max_parallel_allocations:
            reason = "parallel_allocations"
            retry_after_s = self._parallel_retry_after(state, now_s)

        if reason is not None and self.config.enabled:
            return self._reject(state, reason=reason, retry_after_s=retry_after_s)

        state.in_flight_allocations += 1
        self._totals["allowed"] += 1
        return self._decision(state, allowed=True)

    def record_allocation(
        self,
        session_id: str,
        requester: RequesterIdentity,
        *,
        pending_timeout_s: float,
    ) -> None:
        if pending_timeout_s <= 0:
            raise ValueError("pending_timeout_s must be > 0")
        now_s = self._time_fn()
        state = self._actors.get(requester.actor_id)
        if state is None:
            return
        state.in_flight_allocations = max(state.in_flight_allocations - 1, 0)
        state.requester = requester
        state.last_seen_s = now_s
        previous_actor_id = self._session_actors.get(session_id)
        if previous_actor_id is not None and previous_actor_id != requester.actor_id:
            previous_state = self._actors.get(previous_actor_id)
            if previous_state is not None:
                previous_state.allocations.pop(session_id, None)
        state.allocations[session_id] = _TrackedAllocation(
            requester=requester,
            expires_at_s=now_s + pending_timeout_s,
        )
        self._session_actors[session_id] = requester.actor_id
        self._actors.move_to_end(requester.actor_id)
        self._totals["allocations"] += 1

    def record_allocation_failure(self, requester: RequesterIdentity) -> None:
        state = self._actors.get(requester.actor_id)
        if state is None:
            return
        state.in_flight_allocations = max(state.in_flight_allocations - 1, 0)
        state.last_seen_s = self._time_fn()
        self._totals["allocation_failures"] += 1

    def record_connected(self, session_id: str) -> RequesterIdentity | None:
        state, allocation = self._allocation(session_id)
        if state is None or allocation is None:
            return None
        now_s = self._time_fn()
        if allocation.connected_at_s is None:
            allocation.connected_at_s = now_s
            allocation.expires_at_s = math.inf
            state.consecutive_no_connects = 0
            state.last_seen_s = now_s
            self._totals["connections"] += 1
        return allocation.requester

    def record_disconnected(
        self,
        session_id: str,
        *,
        duration_s: float | None = None,
        penalize: bool = True,
    ) -> RequesterSessionOutcome | None:
        state, allocation = self._pop_allocation(session_id)
        if state is None or allocation is None:
            return None

        now_s = self._time_fn()
        state.last_seen_s = now_s
        if allocation.connected_at_s is None:
            if penalize:
                self._record_no_connect(state, now_s)
            return RequesterSessionOutcome(
                requester=allocation.requester,
                connected=False,
                duration_s=None,
                short_session=False,
                no_connect=penalize,
            )

        resolved_duration_s = max(
            float(duration_s)
            if duration_s is not None
            else now_s - allocation.connected_at_s,
            0.0,
        )
        short_session = resolved_duration_s <= self.config.short_session_threshold_s
        if penalize and short_session:
            state.consecutive_short_sessions += 1
            self._totals["short_sessions"] += 1
            if (
                state.consecutive_short_sessions
                >= self.config.max_consecutive_short_sessions
            ):
                self._activate_cooldown(state, now_s)
        elif penalize:
            state.consecutive_short_sessions = 0
        self._totals["completed_sessions"] += 1
        return RequesterSessionOutcome(
            requester=allocation.requester,
            connected=True,
            duration_s=resolved_duration_s,
            short_session=short_session,
            no_connect=False,
        )

    def status(self) -> dict[str, object]:
        now_s = self._time_fn()
        self._prune_all(now_s)
        active_allocations = sum(self._active_allocations(state) for state in self._actors.values())
        return {
            "enabled": self.config.enabled,
            "tracked_actors": len(self._actors),
            "blocked_actors": sum(
                1 for state in self._actors.values() if state.blocked_until_s > now_s
            ),
            "active_allocations": active_allocations,
            "totals": dict(sorted(self._totals.items())),
            "rejection_reasons": dict(sorted(self._rejection_reasons.items())),
            "limits": {
                "request_window_s": self.config.request_window_s,
                "max_requests_per_window": self.config.max_requests_per_window,
                "max_parallel_allocations": self.config.max_parallel_allocations,
                "max_consecutive_no_connects": self.config.max_consecutive_no_connects,
                "short_session_threshold_s": self.config.short_session_threshold_s,
                "max_consecutive_short_sessions": self.config.max_consecutive_short_sessions,
                "cooldown_s": self.config.cooldown_s,
            },
        }

    def _state_for(
        self,
        requester: RequesterIdentity,
        now_s: float,
    ) -> _ActorState | None:
        state = self._actors.get(requester.actor_id)
        if state is not None:
            return state
        self._make_actor_capacity(now_s)
        if len(self._actors) >= self.config.max_actor_states:
            return None
        state = _ActorState(requester=requester, last_seen_s=now_s)
        self._actors[requester.actor_id] = state
        return state

    def _make_actor_capacity(self, now_s: float) -> None:
        if len(self._actors) < self.config.max_actor_states:
            return
        self._prune_all(now_s)
        if len(self._actors) < self.config.max_actor_states:
            return
        for actor_id, state in list(self._actors.items()):
            if self._active_allocations(state) == 0 and state.blocked_until_s <= now_s:
                self._actors.pop(actor_id, None)
                return

    def _expire_pending_allocations(self, state: _ActorState, now_s: float) -> None:
        expired_session_ids = [
            session_id
            for session_id, allocation in state.allocations.items()
            if allocation.connected_at_s is None and allocation.expires_at_s <= now_s
        ]
        for session_id in expired_session_ids:
            state.allocations.pop(session_id, None)
            self._session_actors.pop(session_id, None)
            self._record_no_connect(state, now_s)

    def _record_no_connect(self, state: _ActorState, now_s: float) -> None:
        state.consecutive_no_connects += 1
        self._totals["no_connects"] += 1
        if state.consecutive_no_connects >= self.config.max_consecutive_no_connects:
            self._activate_cooldown(state, now_s)

    def _activate_cooldown(self, state: _ActorState, now_s: float) -> None:
        state.blocked_until_s = max(
            state.blocked_until_s,
            now_s + self.config.cooldown_s,
        )

    def _allocation(
        self,
        session_id: str,
    ) -> tuple[_ActorState | None, _TrackedAllocation | None]:
        actor_id = self._session_actors.get(session_id)
        if actor_id is None:
            return None, None
        state = self._actors.get(actor_id)
        if state is None:
            self._session_actors.pop(session_id, None)
            return None, None
        return state, state.allocations.get(session_id)

    def _pop_allocation(
        self,
        session_id: str,
    ) -> tuple[_ActorState | None, _TrackedAllocation | None]:
        actor_id = self._session_actors.pop(session_id, None)
        if actor_id is None:
            return None, None
        state = self._actors.get(actor_id)
        if state is None:
            return None, None
        return state, state.allocations.pop(session_id, None)

    def _prune_request_window(self, state: _ActorState, now_s: float) -> None:
        cutoff_s = now_s - self.config.request_window_s
        while state.request_times_s and state.request_times_s[0] <= cutoff_s:
            state.request_times_s.popleft()

    def _prune_if_due(self, now_s: float) -> None:
        prune_interval_s = min(self.config.request_window_s, 60.0)
        if now_s - self._last_prune_s >= prune_interval_s:
            self._prune_all(now_s)

    def _prune_all(self, now_s: float) -> None:
        for state in list(self._actors.values()):
            self._expire_pending_allocations(state, now_s)
            self._prune_request_window(state, now_s)
        retention_cutoff_s = now_s - self.config.actor_retention_s
        for actor_id, state in list(self._actors.items()):
            if (
                state.last_seen_s <= retention_cutoff_s
                and self._active_allocations(state) == 0
                and state.blocked_until_s <= now_s
            ):
                self._actors.pop(actor_id, None)
        self._last_prune_s = now_s

    def _parallel_retry_after(self, state: _ActorState, now_s: float) -> int:
        pending_expirations = [
            allocation.expires_at_s
            for allocation in state.allocations.values()
            if allocation.connected_at_s is None
        ]
        if pending_expirations:
            return _retry_after(min(pending_expirations) - now_s)
        return 5

    def _active_allocations(self, state: _ActorState) -> int:
        return state.in_flight_allocations + len(state.allocations)

    def _reject(
        self,
        state: _ActorState,
        *,
        reason: str,
        retry_after_s: int | None,
    ) -> RateLimitDecision:
        self._totals["rejected"] += 1
        self._rejection_reasons[reason] += 1
        decision = self._decision(state, allowed=False, reason=reason)
        return RateLimitDecision(
            **{
                **decision.__dict__,
                "retry_after_s": retry_after_s or 1,
            }
        )

    def _reject_without_state(
        self,
        requester: RequesterIdentity,
        *,
        reason: str,
    ) -> RateLimitDecision:
        self._totals["rejected"] += 1
        self._rejection_reasons[reason] += 1
        return RateLimitDecision(
            allowed=False,
            reason=reason,
            retry_after_s=5,
            actor_id=requester.actor_id,
            recent_requests=0,
            active_allocations=0,
            consecutive_no_connects=0,
            consecutive_short_sessions=0,
        )

    def _decision(
        self,
        state: _ActorState,
        *,
        allowed: bool,
        reason: str | None = None,
    ) -> RateLimitDecision:
        return RateLimitDecision(
            allowed=allowed,
            reason=reason,
            retry_after_s=None,
            actor_id=state.requester.actor_id,
            recent_requests=len(state.request_times_s),
            active_allocations=self._active_allocations(state),
            consecutive_no_connects=state.consecutive_no_connects,
            consecutive_short_sessions=state.consecutive_short_sessions,
        )


def _retry_after(delay_s: float) -> int:
    return max(int(math.ceil(delay_s)), 1)

"""Single-owner GPU lifecycle control, colocated with atomic request routing.

Only explicit endpoint names are managed. Provisioning is a separate operation;
each endpoint has one replica so its work accounting matches one GPU worker.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from app.endpoint_pool_router import EndpointSnapshot
from app.speech_proxy_router import SpeechBackendPool, SpeechBackendSnapshot

logger = logging.getLogger("s2s-endpoint")
PARKED = {"paused", "scaledtozero", "scaledto0"}
FAILED = {"failed", "updatefailed"}


class WorkerController(Protocol):
    def fetch(self, name: str) -> EndpointSnapshot: ...

    def begin_wake(self, name: str) -> EndpointSnapshot: ...

    def park(self, name: str) -> EndpointSnapshot: ...

    def force_restart(self, name: str) -> EndpointSnapshot: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class WorkerLifecycleSettings:
    min_warm: int = 1
    max_workers: int = 1
    scale_up_utilization: float = 0.85
    scale_down_utilization: float = 0.5
    reconcile_interval_s: float = 5.0
    scale_up_cooldown_s: float = 30.0
    scale_down_cooldown_s: float = 180.0
    idle_timeout_s: float = 600.0
    min_uptime_s: float = 300.0
    latency_breach_s: float = 30.0
    latency_max_age_s: float = 60.0
    startup_timeout_s: float = 900.0
    unhealthy_restart_s: float = 120.0
    retry_backoff_s: float = 30.0
    max_restart_attempts: int = 3
    stable_reset_s: float = 300.0

    def __post_init__(self):
        if not 1 <= self.min_warm <= self.max_workers:
            raise ValueError("worker limits require 1 <= min_warm <= max_workers")
        if not 0 < self.scale_down_utilization < self.scale_up_utilization <= 1:
            raise ValueError("utilization requires 0 < scale_down < scale_up <= 1")
        if self.max_restart_attempts < 0:
            raise ValueError("max_restart_attempts must be >= 0")
        for name, value in asdict(self).items():
            if name.endswith("_s") and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and > 0")


@dataclass
class WorkerState:
    name: str
    status: str = "unknown"
    phase: str = "unknown"
    action: str | None = None
    reason: str | None = None
    last_error: str | None = None
    started_at: float | None = None
    ready_since: float | None = None
    unhealthy_since: float | None = None
    ever_ready: bool = False
    retry_at: float = 0.0
    restarts: int = 0
    # A failed/ambiguous park stays quarantined until control confirms stopped.
    park_requested: bool = False
    pause_started: bool = False


class SpeechWorkerLifecycle:
    def __init__(
        self,
        pool: SpeechBackendPool,
        controller: WorkerController,
        settings: WorkerLifecycleSettings,
        *,
        time_fn=time.monotonic,
    ):
        self.pool = pool
        self.controller = controller
        self.settings = settings
        self._time = time_fn
        self._workers: dict[str, WorkerState] = {}
        self._operations: dict[str, asyncio.Task] = {}
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._reconcile_lock = asyncio.Lock()
        self._latency_breach_at: float | None = None
        self._low_load_since: float | None = None
        self._next_up_at = 0.0
        self._next_down_at = 0.0
        self._last_reconcile_at: float | None = None
        self._last_error: str | None = None

    async def start(self):
        for backend in await self.pool.snapshots():
            self._workers[backend.name] = WorkerState(backend.name)
            await self.pool.set_available(backend.name, False)
        if self.settings.max_workers > len(self._workers):
            raise ValueError("max_workers cannot exceed the explicit worker inventory")
        await self.reconcile()
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        # Control calls only initiate changes and have bounded HTTP timeouts.
        # Do not cancel their thread and accidentally overlap remote mutations.
        self._stopped.set()
        if self._task is not None:
            await self._task
        await asyncio.gather(*self._operations.values(), return_exceptions=True)
        self.controller.close()

    async def _loop(self):
        while not self._stopped.is_set():
            try:
                await asyncio.wait_for(self._stopped.wait(), self.settings.reconcile_interval_s)
            except asyncio.TimeoutError:
                try:
                    await self.reconcile()
                except Exception as exc:
                    self._last_error = type(exc).__name__
                    logger.exception("Speech worker reconciliation failed service=%s", self.pool.settings.service)

    async def snapshot(self) -> dict:
        backends = {b.name: b for b in await self.pool.snapshots()}
        return {
            "enabled": True,
            "settings": asdict(self.settings),
            "last_reconcile_age_s": (
                self._time() - self._last_reconcile_at if self._last_reconcile_at is not None else None
            ),
            "last_error": self._last_error,
            "workers": [
                {
                    "name": w.name,
                    "status": w.status,
                    "phase": w.phase,
                    "action": w.action,
                    "reason": w.reason,
                    "last_error": w.last_error,
                    "restart_attempts": w.restarts,
                    "active_requests": backends[w.name].active_requests,
                    "active_work": backends[w.name].active_work,
                    "target_work": backends[w.name].target_work,
                    "ewma_latency": backends[w.name].ewma_latency,
                }
                for w in self._workers.values()
            ],
        }

    async def reconcile(self):
        async with self._reconcile_lock:
            await asyncio.gather(*(self._refresh(w) for w in self._workers.values()))
            backends = {b.name: b for b in await self.pool.snapshots()}
            now = self._time()
            for w in self._workers.values():
                self._phase(w, backends[w.name], now)

            ready = [b for b in backends.values() if b.ready and not b.draining]
            warming = [w for w in self._workers.values() if w.phase == "warming"]
            peak = max(await self.pool.take_work_peak(), sum(b.active_work for b in backends.values()))
            capacity = await self.pool.capacity_snapshot()
            demand = max(peak, capacity.get("session_work", 0)) + capacity.get("reserve_work", 0)
            wanted = max(
                self.settings.min_warm,
                math.ceil(demand / (self.pool.settings.target_work * self.settings.scale_up_utilization)),
            )
            # One slow worker is not fleet saturation. Unknown or stale latency
            # never votes to add GPUs, and idle history cannot trigger scale-up.
            all_slow = (
                bool(ready)
                and peak > 0
                and all(
                    b.latency_age_s is not None
                    and b.latency_age_s <= self.settings.latency_max_age_s
                    and b.ewma_latency is not None
                    and b.ewma_latency > self.pool.settings.latency_target
                    for b in ready
                )
            )
            if not all_slow:
                self._latency_breach_at = None
            elif self._latency_breach_at is None:
                self._latency_breach_at = now
            if self._latency_breach_at is not None and now - self._latency_breach_at >= self.settings.latency_breach_s:
                wanted = max(wanted, len(ready) + 1)
            wanted = min(wanted, self.settings.max_workers)
            pending_capacity = len(ready) + len(warming)
            deficit = wanted - pending_capacity
            # A returning burst can cancel a drain until the actual pause starts.
            for w in self._workers.values():
                if deficit > 0 and w.park_requested and not w.pause_started and not w.action and w.status == "running":
                    w.park_requested = False
                    w.ever_ready = False
                    w.started_at = now
                    w.phase = "warming"
                    w.reason = "cancel drain: capacity needed"
                    await self.pool.set_draining(w.name, False)
                    await self.pool.set_available(w.name, True)
                    deficit -= 1
                    pending_capacity += 1
            # Recover minimum availability immediately; load growth uses a cooldown.
            running = sum(w.status not in PARKED | FAILED or w.action is not None for w in self._workers.values())
            started = 0
            if deficit > 0 and (now >= self._next_up_at or pending_capacity < self.settings.min_warm):
                candidates = [
                    w
                    for w in self._workers.values()
                    if w.status in PARKED and not w.action and not w.park_requested and now >= w.retry_at
                ]
                for w in candidates[: min(1, deficit, max(self.settings.max_workers - running, 0))]:
                    await self._schedule(w, "begin_wake", "fleet headroom or latency")
                    started += 1
                    running += 1
                    deficit -= 1
                    self._next_up_at = now + self.settings.scale_up_cooldown_s
                    self._next_down_at = now + self.settings.scale_down_cooldown_s

            for w in self._workers.values():
                b = backends[w.name]
                if w.action or w.status == "unknown" or now < w.retry_at:
                    continue
                if w.status in FAILED and (deficit <= 0 or started or running >= self.settings.max_workers):
                    continue
                if w.park_requested and not b.active_requests:
                    await self._schedule(w, "park", "complete quarantined park")
                elif (
                    not b.active_requests
                    and w.restarts < self.settings.max_restart_attempts
                    and (
                        w.status in FAILED
                        or (
                            w.status == "running"
                            and w.phase == "unavailable"
                            and w.unhealthy_since is not None
                            and now - w.unhealthy_since >= self.settings.unhealthy_restart_s
                        )
                    )
                ):
                    if await self.pool.quarantine_if_idle(w.name):
                        if w.status in FAILED:
                            started += 1
                            running += 1
                            deficit -= 1
                        w.restarts += 1
                        await self._schedule(w, "force_restart", "recover unhealthy worker")

            low_load = (
                not warming
                and not capacity.get("demand_stale", False)
                and wanted < len(ready)
                and not all_slow
                and all(w.status != "unknown" and not w.action and not w.park_requested for w in self._workers.values())
                and demand <= (len(ready) - 1) * self.pool.settings.target_work * self.settings.scale_down_utilization
            )
            if not low_load:
                self._low_load_since = None
            elif self._low_load_since is None:
                self._low_load_since = now
            if (
                self._low_load_since is not None
                and now - self._low_load_since >= self.settings.idle_timeout_s
                and now >= self._next_down_at
            ):
                for b in sorted(ready, key=lambda b: (b.active_work, -b.idle_for_s)):
                    w = self._workers[b.name]
                    if (
                        w.action
                        or w.phase != "ready"
                        or w.started_at is None
                        or now - w.started_at < self.settings.min_uptime_s
                    ):
                        continue
                    if await self.pool.drain_if_surplus(
                        b.name,
                        min_ready=self.settings.min_warm,
                        utilization=self.settings.scale_down_utilization,
                    ):
                        w.park_requested = True
                        w.phase = "draining"
                        w.reason = "sustained surplus capacity"
                        # Admission may have changed after the policy snapshot
                        # but before the atomic drain. Re-read once fenced; now
                        # reservations can only decrease, never increase.
                        drained = next(s for s in await self.pool.snapshots() if s.name == b.name)
                        if not drained.active_requests:
                            await self._schedule(w, "park", w.reason)
                        self._next_down_at = now + self.settings.scale_down_cooldown_s
                        self._low_load_since = None
                        break
            self._last_reconcile_at = now
            self._last_error = None

    async def _refresh(self, w: WorkerState):
        if w.action:
            return
        try:
            endpoint = await asyncio.to_thread(self.controller.fetch, w.name)
        except Exception as exc:
            w.last_error = f"control fetch: {type(exc).__name__}"
            # Keep data-plane health independently useful during control outages,
            # but do not make ANY lifecycle decision from stale control state.
            w.phase = "unknown"
            w.status = "unknown"
            return
        w.status = endpoint.status
        if endpoint.status in PARKED:
            w.park_requested = False
            w.pause_started = False
            w.started_at = None
            w.ever_ready = False
            await self.pool.set_draining(w.name, False)
        elif w.started_at is None:
            w.started_at = self._time()
        available = endpoint.status == "running" and endpoint.url is not None and not w.park_requested
        await self.pool.set_available(w.name, available, url=endpoint.url)
        w.last_error = None

    def _phase(self, w: WorkerState, b: SpeechBackendSnapshot, now: float):
        if w.park_requested:
            w.phase = "draining"
        elif w.action or (w.status not in PARKED | FAILED | {"running", "unknown"}):
            w.phase = "warming"
        elif w.status in PARKED:
            w.phase = "parked"
        elif w.status == "unknown":
            w.phase = "unknown"
        elif b.ready:
            w.phase = "ready"
            w.ever_ready = True
            w.unhealthy_since = None
            if w.ready_since is None:
                w.ready_since = now
            if now - w.ready_since >= self.settings.stable_reset_s:
                w.restarts = 0
        elif (
            not w.ever_ready
            and w.status == "running"
            and w.started_at is not None
            and now - w.started_at < self.settings.startup_timeout_s
        ):
            w.phase = "warming"
        else:
            w.phase = "unavailable"
        if w.phase != "ready":
            w.ready_since = None
            if w.unhealthy_since is None:
                w.unhealthy_since = now
        if w.phase == "warming" and w.started_at is not None and now - w.started_at > self.settings.startup_timeout_s:
            w.phase = "unavailable"
            w.last_error = "worker startup timed out"

    async def _schedule(self, w: WorkerState, action: str, reason: str):
        if w.action or self._stopped.is_set():
            return
        w.action, w.reason = action, reason
        if action == "park":
            w.park_requested = True
            w.pause_started = True
            w.phase = "draining"
        else:
            w.phase = "warming"
            w.started_at = self._time()
            w.ever_ready = False
        await self.pool.set_available(w.name, False)
        logger.info(
            "Speech worker action service=%s worker=%s action=%s reason=%s",
            self.pool.settings.service,
            w.name,
            action,
            reason,
        )
        self._operations[w.name] = asyncio.create_task(self._operate(w, action))

    async def _operate(self, w: WorkerState, action: str):
        try:
            result = await asyncio.to_thread(getattr(self.controller, action), w.name)
            w.status = result.status
        except Exception as exc:
            w.last_error = f"{action}: {type(exc).__name__}"
            logger.warning("Speech worker action failed worker=%s action=%s error=%s", w.name, action, w.last_error)
        finally:
            w.action = None
            w.retry_at = self._time() + min(self.settings.retry_backoff_s * 2**w.restarts, 300.0)

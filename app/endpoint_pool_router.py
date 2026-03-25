import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol
from urllib.parse import urlparse, urlunparse


logger = logging.getLogger("s2s-endpoint")


def _normalize_status(status: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(status).lower())


def _to_ws_url(base_url: str, path: str = "/ws") -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    route_path = (parsed.path.rstrip("/") + path) if parsed.path else path
    return urlunparse(parsed._replace(scheme=scheme, path=route_path))


def _is_running_status(status: str) -> bool:
    return status == "running"


def _is_parked_status(status: str) -> bool:
    return status in {"paused", "scaledtozero", "scaledto0"}


@dataclass
class EndpointLease:
    slot_id: str
    endpoint_name: str
    ws_url: str


@dataclass
class EndpointSnapshot:
    name: str
    status: str
    raw_status: str
    url: Optional[str]


class EndpointController(Protocol):
    def fetch(self, name: str) -> EndpointSnapshot:
        ...

    def wake(self, name: str) -> EndpointSnapshot:
        ...

    def park(self, name: str) -> EndpointSnapshot:
        ...


class HuggingFaceEndpointController:
    def __init__(
        self,
        *,
        namespace: Optional[str],
        token: Optional[str],
        wait_timeout_s: int = 900,
        wait_refresh_every_s: int = 5,
        active_min_replica: int = 1,
        active_max_replica: int = 1,
        park_strategy: str = "pause",
    ) -> None:
        from huggingface_hub import get_inference_endpoint

        if park_strategy not in {"pause", "scale_to_zero"}:
            raise ValueError("park_strategy must be 'pause' or 'scale_to_zero'")

        self.namespace = namespace
        self.token = token
        self.wait_timeout_s = wait_timeout_s
        self.wait_refresh_every_s = wait_refresh_every_s
        self.active_min_replica = active_min_replica
        self.active_max_replica = active_max_replica
        self.park_strategy = park_strategy
        self._get_inference_endpoint = get_inference_endpoint

    def fetch(self, name: str) -> EndpointSnapshot:
        endpoint = self._get(name)
        endpoint.fetch()
        return self._snapshot(name, endpoint)

    def wake(self, name: str) -> EndpointSnapshot:
        endpoint = self._get(name)
        status = _normalize_status(getattr(endpoint, "status", ""))

        if status == "paused":
            endpoint.resume(running_ok=True)
        elif _is_parked_status(status):
            endpoint.update(
                min_replica=self.active_min_replica,
                max_replica=self.active_max_replica,
            )
        elif not _is_running_status(status):
            endpoint.wait(timeout=self.wait_timeout_s, refresh_every=self.wait_refresh_every_s)
            endpoint.fetch()
            return self._snapshot(name, endpoint)

        endpoint.wait(timeout=self.wait_timeout_s, refresh_every=self.wait_refresh_every_s)
        endpoint.fetch()
        return self._snapshot(name, endpoint)

    def park(self, name: str) -> EndpointSnapshot:
        endpoint = self._get(name)
        status = _normalize_status(getattr(endpoint, "status", ""))

        if self.park_strategy == "pause":
            if status != "paused":
                endpoint.pause()
        else:
            if not _is_parked_status(status):
                endpoint.scale_to_zero()

        endpoint.fetch()
        return self._snapshot(name, endpoint)

    def _get(self, name: str):
        return self._get_inference_endpoint(
            name,
            namespace=self.namespace,
            token=self.token or None,
        )

    def _snapshot(self, name: str, endpoint) -> EndpointSnapshot:
        raw_status = str(getattr(endpoint, "status", "unknown"))
        return EndpointSnapshot(
            name=name,
            status=_normalize_status(raw_status),
            raw_status=raw_status,
            url=getattr(endpoint, "url", None),
        )


@dataclass
class ManagedEndpoint:
    name: str
    slots: int
    status: str = "unknown"
    raw_status: str = "unknown"
    url: Optional[str] = None
    active_sessions: int = 0
    waking: bool = False
    parking: bool = False
    last_error: Optional[str] = None
    last_used_at: float = field(default_factory=time.monotonic)
    wake_capacity_until: Optional[float] = None

    @property
    def running(self) -> bool:
        return _is_running_status(self.status) and self.url is not None

    @property
    def free_slots(self) -> int:
        if not self.running or self.parking:
            return 0
        return max(self.slots - self.active_sessions, 0)

    @property
    def ws_url(self) -> Optional[str]:
        if self.url is None:
            return None
        return _to_ws_url(self.url)


class EndpointPoolRouter:
    def __init__(
        self,
        *,
        endpoint_names: list[str],
        endpoint_slots: int,
        min_warm_endpoints: int,
        wake_threshold_slots: int,
        idle_park_timeout_s: float,
        reconcile_interval_s: float,
        waking_capacity_timeout_s: float,
        controller: EndpointController,
    ) -> None:
        names = [name.strip() for name in endpoint_names if name.strip()]
        if not names:
            raise ValueError("endpoint_names must not be empty")
        if endpoint_slots < 1:
            raise ValueError("endpoint_slots must be >= 1")
        if min_warm_endpoints < 0:
            raise ValueError("min_warm_endpoints must be >= 0")
        if min_warm_endpoints > len(names):
            raise ValueError("min_warm_endpoints cannot exceed number of endpoints")
        if wake_threshold_slots < 0:
            raise ValueError("wake_threshold_slots must be >= 0")
        if waking_capacity_timeout_s < 0:
            raise ValueError("waking_capacity_timeout_s must be >= 0")

        self.endpoint_slots = endpoint_slots
        self.min_warm_endpoints = min_warm_endpoints
        self.wake_threshold_slots = wake_threshold_slots
        self.idle_park_timeout_s = idle_park_timeout_s
        self.reconcile_interval_s = reconcile_interval_s
        self.waking_capacity_timeout_s = waking_capacity_timeout_s
        self.controller = controller

        self._endpoints = {name: ManagedEndpoint(name=name, slots=endpoint_slots) for name in names}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._closed = False
        self._reconcile_task: Optional[asyncio.Task] = None
        self._last_error: Optional[str] = None

    async def start(self) -> None:
        await self.refresh()
        await self.ensure_min_warm()
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())

    async def stop(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconcile_task
            self._reconcile_task = None

    async def acquire(self, timeout_s: float = 900.0) -> EndpointLease:
        deadline = asyncio.get_event_loop().time() + timeout_s

        while True:
            async with self._condition:
                self._raise_if_closed()

                endpoint = self._select_endpoint_unlocked()
                if endpoint is not None and endpoint.ws_url is not None:
                    endpoint.active_sessions += 1
                    endpoint.last_used_at = time.monotonic()
                    wake_names = self._mark_endpoints_to_wake_unlocked()
                    lease = EndpointLease(
                        slot_id=endpoint.name,
                        endpoint_name=endpoint.name,
                        ws_url=endpoint.ws_url,
                    )
                    break

                wake_names = self._mark_endpoints_to_wake_unlocked(force=True)
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise RuntimeError("timed out waiting for an available compute endpoint")
                await asyncio.wait_for(self._condition.wait(), timeout=remaining)

            self._spawn_wake_tasks(wake_names)

        self._spawn_wake_tasks(wake_names)
        return lease

    async def release(self, slot_id: str) -> None:
        async with self._condition:
            endpoint = self._endpoints.get(slot_id)
            if endpoint is None:
                return

            endpoint.active_sessions = max(endpoint.active_sessions - 1, 0)
            endpoint.last_used_at = time.monotonic()

            self._condition.notify_all()

    async def healthcheck(self) -> tuple[bool, Optional[str], dict[str, object]]:
        snapshot = await self.snapshot()
        if snapshot["running_endpoints"]:
            return True, None, snapshot
        if snapshot["waking_endpoints"]:
            return False, "compute endpoints are still waking", snapshot
        errors = snapshot["errors"]
        if errors:
            return False, str(errors[0]["error"]), snapshot
        return False, "no compute endpoint is ready", snapshot

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            endpoints = list(self._endpoints.values())

        running = sum(1 for endpoint in endpoints if endpoint.running)
        waking = sum(1 for endpoint in endpoints if endpoint.waking)
        parking = sum(1 for endpoint in endpoints if endpoint.parking)
        free_slots = sum(endpoint.free_slots for endpoint in endpoints)
        warming_slots = 0
        now = time.monotonic()
        for endpoint in endpoints:
            if self._counts_as_warming_capacity(endpoint, now):
                warming_slots += endpoint.slots
        active_sessions = sum(endpoint.active_sessions for endpoint in endpoints)
        errors = [
            {"endpoint": endpoint.name, "error": endpoint.last_error}
            for endpoint in endpoints
            if endpoint.last_error
        ]
        if self._last_error and not errors:
            errors.append({"endpoint": None, "error": self._last_error})

        return {
            "min_warm_endpoints": self.min_warm_endpoints,
            "wake_threshold_slots": self.wake_threshold_slots,
            "idle_park_timeout_s": self.idle_park_timeout_s,
            "waking_capacity_timeout_s": self.waking_capacity_timeout_s,
            "running_endpoints": running,
            "waking_endpoints": waking,
            "parking_endpoints": parking,
            "active_sessions": active_sessions,
            "free_slots": free_slots,
            "warming_slots": warming_slots,
            "effective_free_slots": free_slots + warming_slots,
            "endpoints": [
                {
                    "name": endpoint.name,
                    "status": endpoint.raw_status,
                    "running": endpoint.running,
                    "waking": endpoint.waking,
                    "parking": endpoint.parking,
                    "active_sessions": endpoint.active_sessions,
                    "free_slots": endpoint.free_slots,
                    "warming_capacity_counted": self._counts_as_warming_capacity(endpoint, now),
                    "url": endpoint.url,
                    "last_error": endpoint.last_error,
                }
                for endpoint in sorted(endpoints, key=lambda item: item.name)
            ],
            "errors": errors,
        }

    async def refresh(self) -> None:
        tasks = [
            asyncio.to_thread(self.controller.fetch, endpoint.name)
            for endpoint in self._endpoints.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        async with self._condition:
            for endpoint, result in zip(self._endpoints.values(), results):
                if isinstance(result, Exception):
                    endpoint.last_error = str(result)
                    self._last_error = endpoint.last_error
                    continue

                endpoint.status = result.status
                endpoint.raw_status = result.raw_status
                endpoint.url = result.url
                if endpoint.running:
                    endpoint.waking = False
                    endpoint.wake_capacity_until = None
                if _is_parked_status(endpoint.status):
                    endpoint.parking = False
                endpoint.last_error = None
                self._last_error = None

            self._condition.notify_all()

    async def ensure_min_warm(self) -> None:
        while True:
            async with self._condition:
                self._raise_if_closed()
                deficit = self.min_warm_endpoints - self._running_or_waking_count_unlocked()
                if deficit <= 0:
                    return

                wake_names = self._mark_endpoints_to_wake_unlocked(target_count=deficit)
                if not wake_names:
                    return

            await asyncio.gather(*(self._wake_endpoint(name) for name in wake_names))

    async def _wake_endpoint(self, name: str) -> None:
        try:
            snapshot = await asyncio.to_thread(self.controller.wake, name)
        except Exception as exc:
            async with self._condition:
                endpoint = self._endpoints[name]
                endpoint.waking = False
                endpoint.wake_capacity_until = None
                endpoint.last_error = str(exc)
                self._last_error = endpoint.last_error
                self._condition.notify_all()
            logger.error("Failed to wake endpoint %s: %s", name, exc)
            return

        async with self._condition:
            endpoint = self._endpoints[name]
            endpoint.status = snapshot.status
            endpoint.raw_status = snapshot.raw_status
            endpoint.url = snapshot.url
            endpoint.waking = False
            endpoint.wake_capacity_until = None
            endpoint.parking = False
            endpoint.last_error = None
            self._last_error = None
            self._condition.notify_all()

        logger.info("Endpoint %s is ready at %s", name, snapshot.url)

    async def _park_endpoint(self, name: str) -> None:
        try:
            snapshot = await asyncio.to_thread(self.controller.park, name)
        except Exception as exc:
            async with self._condition:
                endpoint = self._endpoints[name]
                endpoint.parking = False
                endpoint.wake_capacity_until = None
                endpoint.last_error = str(exc)
                self._last_error = endpoint.last_error
                self._condition.notify_all()
            logger.error("Failed to park endpoint %s: %s", name, exc)
            return

        async with self._condition:
            endpoint = self._endpoints[name]
            endpoint.status = snapshot.status
            endpoint.raw_status = snapshot.raw_status
            endpoint.url = snapshot.url
            endpoint.parking = False
            endpoint.wake_capacity_until = None
            endpoint.last_error = None
            self._last_error = None
            self._condition.notify_all()

    async def _reconcile_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.reconcile_interval_s)
                await self.refresh()
                await self.ensure_min_warm()
                await self._schedule_wakes_if_needed()
                await self._schedule_parks_if_needed()
        except asyncio.CancelledError:
            raise

    async def _schedule_wakes_if_needed(self) -> None:
        async with self._condition:
            wake_names = self._mark_endpoints_to_wake_unlocked()
        self._spawn_wake_tasks(wake_names)

    async def _schedule_parks_if_needed(self) -> None:
        async with self._condition:
            park_names = self._mark_endpoints_to_park_unlocked()
        self._spawn_park_tasks(park_names)

    def _mark_endpoints_to_wake_unlocked(
        self,
        *,
        force: bool = False,
        target_count: Optional[int] = None,
    ) -> list[str]:
        effective_free_slots = self._effective_free_slots_unlocked()
        if not force and target_count is None:
            # Keep the warm floor, but don't spin up extra endpoints while the
            # system is idle. Proactive wakes should only happen once there is
            # actual allocated session pressure.
            if self._active_sessions_unlocked() == 0:
                return []
            if effective_free_slots >= self.wake_threshold_slots:
                return []
        elif force and target_count is None and effective_free_slots > 0:
            return []

        if target_count is None:
            target_count = 1

        candidates = [
            endpoint
            for endpoint in self._endpoints.values()
            if _is_parked_status(endpoint.status) and not endpoint.waking and not endpoint.parking
        ]
        candidates.sort(key=lambda item: (item.active_sessions, item.name))

        selected = []
        for endpoint in candidates[:target_count]:
            endpoint.waking = True
            endpoint.wake_capacity_until = time.monotonic() + self.waking_capacity_timeout_s
            endpoint.last_error = None
            selected.append(endpoint.name)
        return selected

    def _mark_endpoints_to_park_unlocked(self) -> list[str]:
        eligible = [
            endpoint
            for endpoint in self._endpoints.values()
            if self._should_park_endpoint_unlocked(endpoint)
        ]
        eligible.sort(key=lambda item: item.last_used_at)

        selected = []
        running_count = self._running_count_unlocked()
        for endpoint in eligible:
            if running_count - len(selected) <= self.min_warm_endpoints:
                break
            endpoint.parking = True
            selected.append(endpoint.name)
        return selected

    def _should_park_endpoint_unlocked(self, endpoint: ManagedEndpoint) -> bool:
        if not endpoint.running or endpoint.waking or endpoint.parking:
            return False
        if endpoint.active_sessions != 0:
            return False
        if self._running_count_unlocked() <= self.min_warm_endpoints:
            return False
        idle_for = time.monotonic() - endpoint.last_used_at
        return idle_for >= self.idle_park_timeout_s

    def _select_endpoint_unlocked(self) -> Optional[ManagedEndpoint]:
        candidates = [
            endpoint
            for endpoint in self._endpoints.values()
            if endpoint.free_slots > 0
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: (
                item.active_sessions / item.slots,
                item.active_sessions,
                item.name,
            ),
        )

    def _running_count_unlocked(self) -> int:
        return sum(1 for endpoint in self._endpoints.values() if endpoint.running and not endpoint.parking)

    def _running_or_waking_count_unlocked(self) -> int:
        now = time.monotonic()
        return sum(
            1
            for endpoint in self._endpoints.values()
            if (endpoint.running and not endpoint.parking) or self._counts_as_warming_capacity(endpoint, now)
        )

    def _free_slots_unlocked(self) -> int:
        return sum(endpoint.free_slots for endpoint in self._endpoints.values())

    def _effective_free_slots_unlocked(self) -> int:
        now = time.monotonic()
        return sum(endpoint.free_slots for endpoint in self._endpoints.values()) + sum(
            endpoint.slots
            for endpoint in self._endpoints.values()
            if self._counts_as_warming_capacity(endpoint, now)
        )

    def _active_sessions_unlocked(self) -> int:
        return sum(endpoint.active_sessions for endpoint in self._endpoints.values())

    def _counts_as_warming_capacity(self, endpoint: ManagedEndpoint, now: float) -> bool:
        return (
            endpoint.waking
            and not endpoint.parking
            and not endpoint.running
            and endpoint.wake_capacity_until is not None
            and now < endpoint.wake_capacity_until
        )

    def _spawn_wake_tasks(self, names: list[str]) -> None:
        for name in names:
            asyncio.create_task(self._wake_endpoint(name))

    def _spawn_park_tasks(self, names: list[str]) -> None:
        for name in names:
            asyncio.create_task(self._park_endpoint(name))

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("endpoint pool router is shutting down")

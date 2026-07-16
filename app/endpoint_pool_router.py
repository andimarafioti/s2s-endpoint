import asyncio
import contextlib
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional, Protocol
from urllib.parse import urlparse, urlunparse


logger = logging.getLogger("s2s-endpoint")
ComputeUsageFetcher = Callable[[str], int]


class EndpointCapacityTimeoutError(RuntimeError):
    pass


class EndpointTransitionConflictError(RuntimeError):
    pass


def _normalize_status(status: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(status).lower())


def _to_ws_url(base_url: str, path: str = "/v1/realtime") -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    route_path = (parsed.path.rstrip("/") + path) if parsed.path else path
    return urlunparse(parsed._replace(scheme=scheme, path=route_path))


def _to_health_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    route_path = (parsed.path.rstrip("/") + "/health") if parsed.path else "/health"
    return urlunparse(parsed._replace(path=route_path))


class ComputeUsageSchemaError(RuntimeError):
    """The compute health payload parsed but did not match the expected schema.

    Distinct from transient network failures: a schema mismatch means every
    future poll of this node will also fail, so the caller should revoke the
    node's capacity immediately rather than trusting a stale observation.
    """


def fetch_compute_active_sessions(base_url: str) -> int:
    request = urllib.request.Request(
        _to_health_url(base_url),
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"compute health returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"compute health request failed: {exc.reason}") from exc

    router = payload.get("router") if isinstance(payload, dict) else None
    if not isinstance(router, dict):
        raise ComputeUsageSchemaError("compute health response did not include router usage")
    if "active_sessions" in router:
        return max(int(router.get("active_sessions", 0)), 0)
    if "ready_busy" in router:
        return max(int(router.get("ready_busy", 0)), 0)
    # Fail loud on schema drift. Silently defaulting to 0 here previously made
    # the load balancer treat busy compute nodes as free after a restart
    # (2026-06-07 incident): the compute snapshot had renamed its session count
    # key and this function kept returning 0 without any error or log line.
    raise ComputeUsageSchemaError(
        "compute health response did not include a session count "
        "(expected 'active_sessions' or 'ready_busy' in router payload); "
        "refusing to report 0 to avoid treating a busy node as free"
    )


def _is_running_status(status: str) -> bool:
    return status == "running"


def _is_parked_status(status: str) -> bool:
    return status in {"paused", "scaledtozero", "scaledto0"}


def _is_failed_status(status: str) -> bool:
    return status in {"failed", "updatefailed"}


@dataclass
class EndpointLease:
    slot_id: str
    endpoint_name: str
    ws_url: str
    waited_for_capacity: bool = False


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

    def restart(self, name: str) -> EndpointSnapshot:
        ...

    def force_restart(self, name: str) -> EndpointSnapshot:
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

    def restart(self, name: str) -> EndpointSnapshot:
        endpoint = self._get(name)
        status = _normalize_status(getattr(endpoint, "status", ""))

        if _is_running_status(status):
            endpoint.fetch()
            return self._snapshot(name, endpoint)

        try:
            endpoint.pause()
        except Exception:
            pass
        endpoint.resume(running_ok=True)

        endpoint.fetch()
        return self._snapshot(name, endpoint)

    def force_restart(self, name: str) -> EndpointSnapshot:
        """Pause then resume regardless of current status (used for drain recovery)."""
        endpoint = self._get(name)
        try:
            endpoint.pause()
        except Exception:
            pass
        endpoint.resume(running_ok=True)
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
    ws_path: str = "/v1/realtime"
    status: str = "unknown"
    raw_status: str = "unknown"
    url: Optional[str] = None
    active_sessions: int = 0
    connected_sessions: int = 0
    observed_active_sessions: int = 0
    unobserved_connected_sessions: int = 0
    waking: bool = False
    parking: bool = False
    restarting: bool = False
    draining: bool = False
    last_error: Optional[str] = None
    last_used_at: float = field(default_factory=time.monotonic)
    wake_capacity_until: Optional[float] = None
    restart_attempts: int = 0
    last_restart_at: Optional[float] = None
    running_since: Optional[float] = None
    drain_restarting: bool = False
    # When require_usage_sync is set (a compute_usage_fetcher is configured),
    # a running endpoint offers no capacity until its true session count has
    # been observed at least once. This protects a freshly restarted load
    # balancer from routing sessions to nodes that are still busy with
    # conversations that survived the restart.
    require_usage_sync: bool = False
    usage_synced: bool = False
    last_usage_sync_at: Optional[float] = None
    # A successful usage request records the drain generation that was active
    # when the request started. This prevents a request already in flight when
    # a drain is acquired from being mistaken for a post-drain observation.
    drain_generation: int = 0
    usage_sync_drain_generation: Optional[int] = None
    draining_since: Optional[float] = None
    drain_expires_at: Optional[float] = None
    last_drain_warning_at: Optional[float] = None
    last_sync_failure_log_at: Optional[float] = None

    @property
    def running(self) -> bool:
        return _is_running_status(self.status) and self.url is not None

    @property
    def free_slots(self) -> int:
        if not self.running or self.parking or self.draining or self.drain_restarting:
            return 0
        if self.require_usage_sync and not self.usage_synced:
            return 0
        return max(self.slots - self.busy_sessions, 0)

    @property
    def usage_synced_after_drain(self) -> bool:
        return (
            self.draining
            and self.usage_synced
            and self.usage_sync_drain_generation == self.drain_generation
        )

    @property
    def busy_sessions(self) -> int:
        return min(
            self.observed_active_sessions
            + self.pending_sessions
            + self.unobserved_connected_sessions,
            self.slots,
        )

    @property
    def pending_sessions(self) -> int:
        return max(self.active_sessions - self.connected_sessions, 0)

    @property
    def ws_url(self) -> Optional[str]:
        if self.url is None:
            return None
        return _to_ws_url(self.url, self.ws_path)


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
        park_cooldown_s: float,
        controller: EndpointController,
        endpoint_ws_path: str = "/v1/realtime",
        auto_restart: bool = True,
        max_restart_attempts: int = 3,
        restart_backoff_s: float = 30.0,
        restart_backoff_max_s: float = 300.0,
        restart_stable_running_s: float = 120.0,
        drain_restart_timeout_s: float = 600.0,
        drain_lease_ttl_s: float = 3600.0,
        drain_warning_after_s: float = 600.0,
        drain_warning_interval_s: float = 300.0,
        compute_usage_fetcher: Optional[ComputeUsageFetcher] = None,
        usage_sync_stale_ttl_s: float = 60.0,
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
        if park_cooldown_s < 0:
            raise ValueError("park_cooldown_s must be >= 0")
        if not endpoint_ws_path.startswith("/"):
            raise ValueError("endpoint_ws_path must start with '/'")
        if drain_lease_ttl_s <= 0:
            raise ValueError("drain_lease_ttl_s must be > 0")

        self.endpoint_slots = endpoint_slots
        self.endpoint_ws_path = endpoint_ws_path
        self.min_warm_endpoints = min_warm_endpoints
        self.wake_threshold_slots = wake_threshold_slots
        self.idle_park_timeout_s = idle_park_timeout_s
        self.reconcile_interval_s = reconcile_interval_s
        self.waking_capacity_timeout_s = waking_capacity_timeout_s
        self.park_cooldown_s = park_cooldown_s
        self.controller = controller
        self.auto_restart = auto_restart
        self.max_restart_attempts = max_restart_attempts
        self.restart_backoff_s = restart_backoff_s
        self.restart_backoff_max_s = restart_backoff_max_s
        self.restart_stable_running_s = restart_stable_running_s
        self.drain_restart_timeout_s = drain_restart_timeout_s
        self.drain_lease_ttl_s = drain_lease_ttl_s
        self.drain_warning_after_s = max(drain_warning_after_s, 0.0)
        self.drain_warning_interval_s = max(drain_warning_interval_s, 0.0)
        self.compute_usage_fetcher = compute_usage_fetcher
        self.usage_sync_stale_ttl_s = max(usage_sync_stale_ttl_s, 0.0)

        self._on_endpoint_down: Optional[Callable[[str], Awaitable[None]]] = None
        self._endpoints = {
            name: ManagedEndpoint(
                name=name,
                slots=endpoint_slots,
                ws_path=endpoint_ws_path,
                require_usage_sync=compute_usage_fetcher is not None,
            )
            for name in names
        }
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._closed = False
        self._reconcile_task: Optional[asyncio.Task] = None
        self._initial_warm_task: Optional[asyncio.Task] = None
        self._last_error: Optional[str] = None
        self._next_park_allowed_at = 0.0

    async def start(self) -> None:
        await self.refresh()
        await self._retry_initial_usage_sync()
        self._initial_warm_task = asyncio.create_task(self.ensure_min_warm())
        self._reconcile_task = asyncio.create_task(self._reconcile_loop())

    async def _retry_initial_usage_sync(self) -> None:
        """Give unsynced running endpoints a second chance at startup.

        A node whose /health call failed during the initial refresh would
        otherwise offer zero capacity until the first reconcile tick. One
        immediate retry covers transient failures without delaying startup.
        """
        if self.compute_usage_fetcher is None:
            return
        async with self._lock:
            needs_retry = any(
                endpoint.running and not endpoint.usage_synced
                for endpoint in self._endpoints.values()
            )
        if needs_retry:
            await self._sync_compute_usage()

    async def stop(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()

        if self._initial_warm_task is not None:
            self._initial_warm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._initial_warm_task
            self._initial_warm_task = None

        if self._reconcile_task is not None:
            self._reconcile_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconcile_task
            self._reconcile_task = None

    async def acquire(self, timeout_s: float = 900.0) -> EndpointLease:
        deadline = asyncio.get_event_loop().time() + timeout_s
        waited_for_capacity = False

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
                        waited_for_capacity=waited_for_capacity,
                    )
                    break

                wake_names = self._mark_endpoints_to_wake_unlocked(force=True)
                # Spawn the wake tasks BEFORE suspending on the condition.
                # The endpoints are already marked waking, so no other path
                # will pick them up; waiting first would strand them until an
                # unrelated notify (e.g. the next reconcile sync) resumed this
                # coroutine, adding up to a full reconcile interval of delay
                # or a hard timeout. create_task does not await, and
                # Condition.wait() releases the lock, so the wake task can
                # make progress while we wait.
                self._spawn_wake_tasks(wake_names)
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise EndpointCapacityTimeoutError("timed out waiting for an available compute endpoint")
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise EndpointCapacityTimeoutError(
                        "timed out waiting for an available compute endpoint"
                    ) from exc
                waited_for_capacity = True

        # Only the success path reaches here: forced wake names are spawned
        # in-loop before waiting, and wake_names was reassigned by the
        # non-forced mark on break, so this serves the proactive top-up when
        # remaining capacity dipped below the wake threshold. No double spawn.
        self._spawn_wake_tasks(wake_names)
        return lease

    async def mark_connected(self, slot_id: str) -> None:
        async with self._condition:
            endpoint = self._endpoints.get(slot_id)
            if endpoint is None:
                return
            if endpoint.connected_sessions >= endpoint.active_sessions:
                return

            endpoint.connected_sessions += 1
            endpoint.unobserved_connected_sessions += 1
            endpoint.last_used_at = time.monotonic()

            self._condition.notify_all()

    async def release(self, slot_id: str, *, connected: bool = False) -> None:
        async with self._condition:
            endpoint = self._endpoints.get(slot_id)
            if endpoint is None:
                return

            endpoint.active_sessions = max(endpoint.active_sessions - 1, 0)
            if connected:
                endpoint.connected_sessions = max(endpoint.connected_sessions - 1, 0)
                endpoint.unobserved_connected_sessions = max(
                    endpoint.unobserved_connected_sessions - 1,
                    0,
                )
            endpoint.connected_sessions = min(endpoint.connected_sessions, endpoint.active_sessions)
            endpoint.unobserved_connected_sessions = min(
                endpoint.unobserved_connected_sessions,
                endpoint.connected_sessions,
            )
            endpoint.last_used_at = time.monotonic()

            self._condition.notify_all()

    async def set_draining(
        self,
        name: str,
        draining: bool,
        *,
        lease_ttl_s: Optional[float] = None,
    ) -> None:
        if draining and lease_ttl_s is not None and lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be > 0")

        wake_names: list[str] = []
        async with self._condition:
            endpoint = self._endpoints.get(name)
            if endpoint is None:
                raise KeyError(name)

            if draining:
                active_transitions = [
                    flag
                    for flag in ("waking", "parking", "restarting", "drain_restarting")
                    if getattr(endpoint, flag)
                ]
                if active_transitions:
                    raise EndpointTransitionConflictError(
                        f"Endpoint {name} has an active control-plane transition: "
                        f"{', '.join(active_transitions)}"
                    )

            now = time.monotonic()
            if draining:
                if not endpoint.draining:
                    endpoint.drain_generation += 1
                    endpoint.draining_since = now
                    endpoint.last_drain_warning_at = None
                endpoint.draining = True
                endpoint.drain_expires_at = now + (
                    lease_ttl_s if lease_ttl_s is not None else self.drain_lease_ttl_s
                )
            else:
                endpoint.draining = False
                endpoint.draining_since = None
                endpoint.drain_expires_at = None
                endpoint.last_drain_warning_at = None
            if draining:
                deficit = self.min_warm_endpoints - self._running_or_waking_count_unlocked()
                if deficit > 0:
                    wake_names = self._mark_endpoints_to_wake_unlocked(target_count=deficit)

            self._condition.notify_all()

        self._spawn_wake_tasks(wake_names)

    async def healthcheck(self) -> tuple[bool, Optional[str], dict[str, object]]:
        snapshot = await self.snapshot()
        if snapshot["running_endpoints"]:
            # A running endpoint is not necessarily usable: with a usage
            # fetcher configured it offers zero capacity until its true
            # session count has been observed. A pool where every running
            # node is unsynced cannot allocate (schema drift or a compute
            # health outage), which is different from a pool that is merely
            # full: known-full is healthy, unknown capacity is not.
            if snapshot["unsynced_running_endpoints"] == snapshot["running_endpoints"]:
                return (
                    False,
                    "running compute endpoints have not synced usage yet",
                    snapshot,
                )
            return True, None, snapshot
        if snapshot["waking_endpoints"]:
            return False, "compute endpoints are still waking", snapshot
        if snapshot["restarting_endpoints"]:
            return False, "compute endpoints are restarting after failure", snapshot
        errors = snapshot["errors"]
        if errors:
            return False, str(errors[0]["error"]), snapshot
        return False, "no compute endpoint is ready", snapshot

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            endpoints = list(self._endpoints.values())

        running = sum(1 for endpoint in endpoints if endpoint.running)
        unsynced_running = sum(
            1
            for endpoint in endpoints
            if endpoint.running and endpoint.require_usage_sync and not endpoint.usage_synced
        )
        waking = sum(1 for endpoint in endpoints if endpoint.waking)
        parking = sum(1 for endpoint in endpoints if endpoint.parking)
        restarting = sum(1 for endpoint in endpoints if endpoint.restarting)
        free_slots = sum(endpoint.free_slots for endpoint in endpoints)
        warming_slots = 0
        now = time.monotonic()
        for endpoint in endpoints:
            if self._counts_as_warming_capacity(endpoint, now):
                warming_slots += endpoint.slots
        active_sessions = sum(endpoint.active_sessions for endpoint in endpoints)
        connected_sessions = sum(endpoint.connected_sessions for endpoint in endpoints)
        pending_sessions = sum(endpoint.pending_sessions for endpoint in endpoints)
        observed_active_sessions = sum(endpoint.observed_active_sessions for endpoint in endpoints)
        unobserved_connected_sessions = sum(
            endpoint.unobserved_connected_sessions for endpoint in endpoints
        )
        busy_sessions = sum(endpoint.busy_sessions for endpoint in endpoints)
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
            "park_cooldown_s": self.park_cooldown_s,
            "park_cooldown_remaining_s": max(self._next_park_allowed_at - now, 0.0),
            "auto_restart": self.auto_restart,
            "max_restart_attempts": self.max_restart_attempts,
            "running_endpoints": running,
            "unsynced_running_endpoints": unsynced_running,
            "waking_endpoints": waking,
            "parking_endpoints": parking,
            "restarting_endpoints": restarting,
            "active_sessions": busy_sessions,
            "local_active_sessions": active_sessions,
            "local_connected_sessions": connected_sessions,
            "local_pending_sessions": pending_sessions,
            "observed_active_sessions": observed_active_sessions,
            "unobserved_connected_sessions": unobserved_connected_sessions,
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
                    "restarting": endpoint.restarting,
                    "draining": endpoint.draining,
                    "draining_for_s": (
                        max(now - endpoint.draining_since, 0.0)
                        if endpoint.draining and endpoint.draining_since is not None
                        else None
                    ),
                    "drain_lease_remaining_s": (
                        max(endpoint.drain_expires_at - now, 0.0)
                        if endpoint.draining and endpoint.drain_expires_at is not None
                        else None
                    ),
                    "drain_restarting": endpoint.drain_restarting,
                    "restart_attempts": endpoint.restart_attempts,
                    "active_sessions": endpoint.busy_sessions,
                    "local_active_sessions": endpoint.active_sessions,
                    "local_connected_sessions": endpoint.connected_sessions,
                    "local_pending_sessions": endpoint.pending_sessions,
                    "observed_active_sessions": endpoint.observed_active_sessions,
                    "unobserved_connected_sessions": endpoint.unobserved_connected_sessions,
                    "usage_synced": endpoint.usage_synced,
                    "usage_synced_after_drain": endpoint.usage_synced_after_drain,
                    "require_usage_sync": endpoint.require_usage_sync,
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

        now = time.monotonic()
        downed_endpoints: list[str] = []

        async with self._condition:
            for endpoint, result in zip(self._endpoints.values(), results):
                if isinstance(result, Exception):
                    endpoint.last_error = str(result)
                    self._last_error = endpoint.last_error
                    continue

                was_running = endpoint.running

                endpoint.status = result.status
                endpoint.raw_status = result.raw_status
                endpoint.url = result.url
                if endpoint.running:
                    endpoint.waking = False
                    endpoint.wake_capacity_until = None
                    if not was_running:
                        endpoint.running_since = now
                        endpoint.drain_restarting = False
                    if (
                        endpoint.restart_attempts > 0
                        and endpoint.running_since is not None
                        and now - endpoint.running_since >= self.restart_stable_running_s
                    ):
                        endpoint.restart_attempts = 0
                        endpoint.last_restart_at = None
                else:
                    endpoint.running_since = None

                if _is_parked_status(endpoint.status):
                    endpoint.parking = False

                local_active_sessions = endpoint.active_sessions
                if was_running and not endpoint.running:
                    endpoint.active_sessions = 0
                    endpoint.connected_sessions = 0
                    endpoint.observed_active_sessions = 0
                    endpoint.unobserved_connected_sessions = 0
                    endpoint.usage_synced = False
                    endpoint.last_usage_sync_at = None
                    endpoint.usage_sync_drain_generation = None
                    if local_active_sessions > 0:
                        downed_endpoints.append(endpoint.name)
                elif not endpoint.running:
                    endpoint.observed_active_sessions = 0
                    endpoint.unobserved_connected_sessions = 0
                    endpoint.usage_synced = False
                    endpoint.last_usage_sync_at = None
                    endpoint.usage_sync_drain_generation = None

                if (
                    _is_failed_status(endpoint.status)
                    and self.auto_restart
                    and endpoint.restart_attempts >= self.max_restart_attempts
                ):
                    endpoint.last_error = (
                        f"endpoint failed, {endpoint.restart_attempts} restart attempt(s) exhausted"
                    )
                else:
                    endpoint.last_error = None
                self._last_error = None

            self._condition.notify_all()

        if self._on_endpoint_down is not None:
            for name in downed_endpoints:
                try:
                    await self._on_endpoint_down(name)
                except Exception:
                    logger.exception("on_endpoint_down callback failed for %s", name)

        await self._sync_compute_usage()

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

    async def _sync_compute_usage(self) -> None:
        if self.compute_usage_fetcher is None:
            return

        async with self._condition:
            targets = [
                (endpoint.name, endpoint.url, endpoint.drain_generation)
                for endpoint in self._endpoints.values()
                if endpoint.running and endpoint.url is not None
            ]

        if not targets:
            return

        results = await asyncio.gather(
            *(
                asyncio.to_thread(self.compute_usage_fetcher, url)
                for _, url, _ in targets
            ),
            return_exceptions=True,
        )

        now = time.monotonic()
        async with self._condition:
            for (name, url, drain_generation), result in zip(targets, results):
                endpoint = self._endpoints.get(name)
                if endpoint is None or endpoint.url != url or not endpoint.running:
                    continue
                if isinstance(result, Exception):
                    if isinstance(result, ComputeUsageSchemaError):
                        # Every future poll will fail the same way; stale
                        # observations must not keep granting capacity. Log
                        # unconditionally: on a freshly restarted LB the node
                        # was never synced, and gating the log on a state
                        # transition would replay the original incident's
                        # silence.
                        logger.error(
                            "Compute usage schema error for %s, endpoint offers no capacity until sync recovers: %s",
                            name,
                            result,
                        )
                        endpoint.usage_synced = False
                        endpoint.last_usage_sync_at = None
                        endpoint.usage_sync_drain_generation = None
                    else:
                        stale_for = (
                            None
                            if endpoint.last_usage_sync_at is None
                            else now - endpoint.last_usage_sync_at
                        )
                        if stale_for is None or stale_for > self.usage_sync_stale_ttl_s:
                            # Rate-limit to one error line per node per minute
                            # so a prolonged outage stays visible without
                            # flooding, and a never-synced node is not silent.
                            if (
                                endpoint.last_sync_failure_log_at is None
                                or now - endpoint.last_sync_failure_log_at >= 60.0
                            ):
                                logger.error(
                                    "Compute usage for %s unavailable (last success: %s, TTL %.1fs), "
                                    "endpoint offers no capacity until sync recovers: %s",
                                    name,
                                    f"{stale_for:.1f}s ago" if stale_for is not None else "never",
                                    self.usage_sync_stale_ttl_s,
                                    result,
                                )
                                endpoint.last_sync_failure_log_at = now
                            endpoint.usage_synced = False
                            endpoint.usage_sync_drain_generation = None
                        else:
                            logger.warning(
                                "Failed to sync compute usage for %s (last success %.1fs ago, TTL %.1fs): %s",
                                name,
                                stale_for,
                                self.usage_sync_stale_ttl_s,
                                result,
                            )
                    continue

                previous_observed_active_sessions = endpoint.observed_active_sessions
                observed_active_sessions = min(max(int(result), 0), endpoint.slots)
                observed_increase = max(
                    observed_active_sessions - previous_observed_active_sessions,
                    0,
                )
                if observed_active_sessions != endpoint.observed_active_sessions:
                    logger.info(
                        "Synced compute usage for %s: observed active sessions %s -> %s",
                        name,
                        endpoint.observed_active_sessions,
                        observed_active_sessions,
                    )
                endpoint.observed_active_sessions = observed_active_sessions
                endpoint.usage_synced = True
                endpoint.last_usage_sync_at = now
                endpoint.usage_sync_drain_generation = drain_generation
                endpoint.last_sync_failure_log_at = None
                if observed_increase:
                    endpoint.unobserved_connected_sessions = max(
                        endpoint.unobserved_connected_sessions - observed_increase,
                        0,
                    )
                if observed_active_sessions:
                    endpoint.last_used_at = now

            self._condition.notify_all()

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
            # A wake spawns a fresh compute process with zero sessions, so the
            # usage of this endpoint is known without polling /health.
            endpoint.observed_active_sessions = 0
            endpoint.usage_synced = True
            endpoint.last_usage_sync_at = time.monotonic()
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
                await self._maintain_drain_leases()
                await self._schedule_restarts_if_needed()
                await self._check_drain_restarts()
                await self.ensure_min_warm()
                await self._schedule_wakes_if_needed()
                await self._schedule_parks_if_needed()
        except asyncio.CancelledError:
            raise

    async def _maintain_drain_leases(self) -> None:
        now = time.monotonic()
        expired: list[tuple[str, float]] = []
        warnings: list[tuple[str, float, float]] = []

        async with self._condition:
            for endpoint in self._endpoints.values():
                if not endpoint.draining or endpoint.draining_since is None:
                    continue

                draining_for_s = max(now - endpoint.draining_since, 0.0)
                remaining_s = (
                    max(endpoint.drain_expires_at - now, 0.0)
                    if endpoint.drain_expires_at is not None
                    else 0.0
                )
                if endpoint.drain_expires_at is None or now >= endpoint.drain_expires_at:
                    endpoint.draining = False
                    endpoint.draining_since = None
                    endpoint.drain_expires_at = None
                    endpoint.last_drain_warning_at = None
                    expired.append((endpoint.name, draining_for_s))
                    continue

                warning_due = (
                    draining_for_s >= self.drain_warning_after_s
                    and (
                        endpoint.last_drain_warning_at is None
                        or now - endpoint.last_drain_warning_at
                        >= self.drain_warning_interval_s
                    )
                )
                if warning_due:
                    endpoint.last_drain_warning_at = now
                    warnings.append((endpoint.name, draining_for_s, remaining_s))

            if expired:
                self._condition.notify_all()

        for name, draining_for_s, remaining_s in warnings:
            logger.warning(
                "Endpoint %s has been allocator-drained for %.0fs; lease expires in %.0fs",
                name,
                draining_for_s,
                remaining_s,
            )
        for name, draining_for_s in expired:
            logger.error(
                "Allocator drain lease expired for endpoint %s after %.0fs; clearing drain automatically",
                name,
                draining_for_s,
            )

    async def _schedule_restarts_if_needed(self) -> None:
        if not self.auto_restart:
            return

        now = time.monotonic()
        restart_names: list[str] = []

        async with self._condition:
            for endpoint in self._endpoints.values():
                if not _is_failed_status(endpoint.status):
                    continue
                if endpoint.restarting or endpoint.waking or endpoint.parking:
                    continue
                if endpoint.draining:
                    continue
                if endpoint.restart_attempts >= self.max_restart_attempts:
                    continue
                if endpoint.last_restart_at is not None:
                    backoff = min(
                        self.restart_backoff_s * (2 ** (endpoint.restart_attempts - 1)),
                        self.restart_backoff_max_s,
                    )
                    if now - endpoint.last_restart_at < backoff:
                        continue

                endpoint.restarting = True
                endpoint.restart_attempts += 1
                endpoint.last_restart_at = now
                restart_names.append(endpoint.name)

        for name in restart_names:
            logger.info(
                "Scheduling restart for failed endpoint %s (attempt %d/%d)",
                name,
                self._endpoints[name].restart_attempts,
                self.max_restart_attempts,
            )
            asyncio.create_task(self._restart_endpoint(name))

    async def _restart_endpoint(self, name: str) -> None:
        try:
            snapshot = await asyncio.to_thread(self.controller.restart, name)
        except Exception as exc:
            async with self._condition:
                endpoint = self._endpoints[name]
                endpoint.restarting = False
                endpoint.last_error = str(exc)
                self._last_error = endpoint.last_error
                self._condition.notify_all()
            logger.error("Failed to restart endpoint %s: %s", name, exc)
            return

        async with self._condition:
            endpoint = self._endpoints[name]
            endpoint.status = snapshot.status
            endpoint.raw_status = snapshot.raw_status
            endpoint.url = snapshot.url
            endpoint.restarting = False
            endpoint.last_error = None
            self._last_error = None
            if endpoint.running:
                endpoint.running_since = time.monotonic()
                endpoint.observed_active_sessions = 0
                endpoint.usage_synced = True
                endpoint.last_usage_sync_at = time.monotonic()
            self._condition.notify_all()

        logger.info(
            "Restart action completed for endpoint %s (status: %s)",
            name,
            snapshot.raw_status,
        )

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
            if _is_parked_status(endpoint.status)
            and not endpoint.draining
            and not endpoint.waking
            and not endpoint.parking
            and not endpoint.restarting
            and not endpoint.drain_restarting
        ]
        candidates.sort(key=lambda item: (item.busy_sessions, item.name))

        selected = []
        for endpoint in candidates[:target_count]:
            endpoint.waking = True
            endpoint.wake_capacity_until = time.monotonic() + self.waking_capacity_timeout_s
            endpoint.last_error = None
            selected.append(endpoint.name)
        return selected

    def _mark_endpoints_to_park_unlocked(self) -> list[str]:
        now = time.monotonic()
        if now < self._next_park_allowed_at:
            return []

        eligible = [
            endpoint
            for endpoint in self._endpoints.values()
            if self._should_park_endpoint_unlocked(endpoint)
        ]
        eligible.sort(key=lambda item: item.last_used_at)

        running_count = self._running_count_unlocked()
        for endpoint in eligible:
            if running_count - 1 < self.min_warm_endpoints:
                return []
            endpoint.parking = True
            self._next_park_allowed_at = now + self.park_cooldown_s
            return [endpoint.name]
        return []

    def _should_park_endpoint_unlocked(self, endpoint: ManagedEndpoint) -> bool:
        if (
            not endpoint.running
            or endpoint.waking
            or endpoint.parking
            or endpoint.restarting
            or endpoint.draining
            or endpoint.drain_restarting
        ):
            return False
        if endpoint.busy_sessions != 0:
            return False
        if endpoint.require_usage_sync and not endpoint.usage_synced:
            # busy_sessions cannot be trusted for an unsynced endpoint: after
            # an LB restart with a broken sync, a node mid-conversation has no
            # local leases and observed_active_sessions = 0, so it looks idle
            # here. Parking it would kill live conversations, a strictly worse
            # outcome than the session rejections this gating exists to
            # prevent. The cost is that a pool stuck unsynced never scales
            # down, which is acceptable because healthcheck already reports
            # that state as unhealthy.
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
                item.busy_sessions / item.slots,
                item.busy_sessions,
                item.name,
            ),
        )

    def _running_count_unlocked(self) -> int:
        return sum(
            1
            for endpoint in self._endpoints.values()
            if endpoint.running
            and not endpoint.parking
            and not endpoint.draining
            and not endpoint.drain_restarting
        )

    def _running_or_waking_count_unlocked(self) -> int:
        now = time.monotonic()
        return sum(
            1
            for endpoint in self._endpoints.values()
            if (
                endpoint.running
                and not endpoint.parking
                and not endpoint.draining
                and not endpoint.drain_restarting
            )
            or self._counts_as_warming_capacity(endpoint, now)
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
        return sum(endpoint.busy_sessions for endpoint in self._endpoints.values())

    def _counts_as_warming_capacity(self, endpoint: ManagedEndpoint, now: float) -> bool:
        return (
            endpoint.waking
            and not endpoint.parking
            and not endpoint.draining
            and not endpoint.drain_restarting
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

    @staticmethod
    def _fetch_pool_units(url: str) -> Optional[list[dict]]:
        """GET {url}/v1/pool and return the units list, or None if unavailable."""
        pool_url = url.rstrip("/") + "/v1/pool"
        req = urllib.request.Request(pool_url, headers={"Accept": "application/json"}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                body = response.read()
            data = json.loads(body.decode("utf-8"))
            units = data.get("units")
            return units if isinstance(units, list) else None
        except Exception:
            return None

    async def _check_drain_restarts(self) -> None:
        """Poll /v1/pool on each running endpoint and force-restart (pause → resume) when:
        - any unit has been draining for >= drain_restart_timeout_s, OR
        - all units are simultaneously draining (pool fully wedged — restart immediately).
        Uses draining_for_s from the pool response as the authoritative drain duration.
        """
        async with self._lock:
            to_poll = [
                (ep.name, ep.url)
                for ep in self._endpoints.values()
                if ep.running
                and not ep.waking
                and not ep.parking
                and not ep.restarting
                and not ep.drain_restarting
                and not ep.draining
                and ep.url is not None
            ]

        if not to_poll:
            return

        # Poll all endpoints concurrently so a slow/unreachable node doesn't stall the reconcile loop.
        poll_results = await asyncio.gather(
            *(asyncio.to_thread(self._fetch_pool_units, url) for _, url in to_poll),
            return_exceptions=True,
        )

        to_restart: list[tuple[str, str]] = []  # (name, reason)

        for (name, _), units in zip(to_poll, poll_results):
            if isinstance(units, BaseException) or not units:
                continue

            draining = [u for u in units if u.get("state") == "draining"]
            if not draining:
                continue

            all_draining = len(draining) == len(units)
            max_draining_s = max(float(u.get("draining_for_s", 0)) for u in draining)

            if max_draining_s >= self.drain_restart_timeout_s:
                if all_draining:
                    reason = f"all {len(units)} pipeline unit(s) stuck draining for {max_draining_s:.0f}s"
                else:
                    reason = f"{len(draining)}/{len(units)} unit(s) stuck draining for {max_draining_s:.0f}s"
            else:
                logger.warning(
                    "Endpoint %s: %d/%d pipeline unit(s) draining, max draining_for_s=%.0f "
                    "(restart threshold %.0fs)",
                    name, len(draining), len(units), max_draining_s, self.drain_restart_timeout_s,
                )
                continue

            async with self._condition:
                ep = self._endpoints.get(name)
                if (
                    ep is None
                    or ep.drain_restarting
                    or ep.restarting
                    or ep.parking
                    or ep.waking
                    or ep.draining
                ):
                    continue
                ep.drain_restarting = True
                self._condition.notify_all()

            to_restart.append((name, reason))

        for name, reason in to_restart:
            logger.error(
                "Endpoint %s triggering force restart (pause → resume): %s",
                name, reason,
            )
            asyncio.create_task(self._drain_restart_endpoint(name))

    async def _drain_restart_endpoint(self, name: str) -> None:
        try:
            snapshot = await asyncio.to_thread(self.controller.force_restart, name)
        except Exception as exc:
            async with self._condition:
                ep = self._endpoints[name]
                ep.drain_restarting = False
                ep.last_error = str(exc)
                self._last_error = ep.last_error
                self._condition.notify_all()
            logger.error("Drain restart failed for endpoint %s: %s", name, exc)
            return

        async with self._condition:
            ep = self._endpoints[name]
            ep.status = snapshot.status
            ep.raw_status = snapshot.raw_status
            ep.url = snapshot.url
            ep.drain_restarting = False
            ep.last_error = None
            self._last_error = None
            if ep.running:
                ep.running_since = time.monotonic()
                ep.observed_active_sessions = 0
                ep.usage_synced = True
                ep.last_usage_sync_at = time.monotonic()
            self._condition.notify_all()

        logger.info("Drain restart completed for endpoint %s (status: %s)", name, snapshot.raw_status)

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("endpoint pool router is shutting down")

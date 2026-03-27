import asyncio
import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional


logger = logging.getLogger("s2s-endpoint")


WaitForReady = Callable[[str, int, Optional[subprocess.Popen], float], Awaitable[None]]
BuildCommand = Callable[[str, int], list[str]]


@dataclass
class PipelineSlot:
    slot_id: int
    host: str
    port: int
    ws_path: str = ""
    process: Optional[subprocess.Popen] = None
    ready: bool = False
    busy: bool = False
    starting: bool = False
    last_error: Optional[str] = None
    ws_url: str = field(init=False)

    def __post_init__(self) -> None:
        self.ws_url = f"ws://{self.host}:{self.port}{self.ws_path}"


class SessionRouter:
    def __init__(
        self,
        *,
        host: str,
        base_port: int,
        repo_dir: str,
        min_idle_instances: int,
        max_instances: int,
        build_command: BuildCommand,
        wait_for_ready: WaitForReady,
        ws_path: str = "",
    ) -> None:
        if max_instances < 1:
            raise ValueError("max_instances must be >= 1")
        if min_idle_instances < 0:
            raise ValueError("min_idle_instances must be >= 0")
        if min_idle_instances > max_instances:
            raise ValueError("min_idle_instances cannot exceed max_instances")

        self.host = host
        self.base_port = base_port
        self.ws_path = ws_path
        self.repo_dir = repo_dir
        self.min_idle_instances = min_idle_instances
        self.max_instances = max_instances
        self.build_command = build_command
        self.wait_for_ready = wait_for_ready

        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._slots: dict[int, PipelineSlot] = {}
        self._next_slot_id = 0
        self._closed = False
        self._last_error: Optional[str] = None

    async def start(self) -> None:
        await self.ensure_min_idle()

    async def stop(self) -> None:
        async with self._lock:
            self._closed = True
            slots = list(self._slots.values())
            self._slots.clear()
            self._condition.notify_all()

        for slot in slots:
            await asyncio.to_thread(self._stop_slot_process, slot)

    async def acquire(self, timeout_s: float = 900.0) -> PipelineSlot:
        starter: Optional[PipelineSlot] = None
        deadline = asyncio.get_event_loop().time() + timeout_s

        while True:
            async with self._condition:
                self._raise_if_closed()

                idle_slot = self._find_idle_slot_unlocked()
                if idle_slot is not None:
                    idle_slot.busy = True
                    slot = idle_slot
                    break

                if len(self._slots) < self.max_instances:
                    starter = self._create_slot_unlocked(busy=True)
                    slot = starter
                    break

                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise RuntimeError("timed out waiting for an available speech-to-speech pipeline slot")
                await asyncio.wait_for(self._condition.wait(), timeout=remaining)

        if starter is not None:
            await self._start_slot(starter, timeout_s=timeout_s)

        asyncio.create_task(self.ensure_min_idle())
        return slot

    async def release(self, slot_id: int) -> None:
        needs_reconcile = False

        async with self._condition:
            slot = self._slots.get(slot_id)
            if slot is None:
                return

            slot.busy = False

            if slot.process is not None and slot.process.poll() is not None:
                slot.ready = False
                slot.last_error = f"speech-to-speech process exited with code {slot.process.returncode}"
                self._last_error = slot.last_error
                self._slots.pop(slot_id, None)
                needs_reconcile = True

            self._condition.notify_all()

        if needs_reconcile:
            await asyncio.to_thread(self._stop_slot_process, slot)

        asyncio.create_task(self.ensure_min_idle())

    async def ensure_min_idle(self) -> None:
        while True:
            async with self._lock:
                self._raise_if_closed()
                idle_or_starting = sum(
                    1 for slot in self._slots.values() if (slot.ready and not slot.busy) or slot.starting
                )
                if idle_or_starting >= self.min_idle_instances or len(self._slots) >= self.max_instances:
                    return

                slot = self._create_slot_unlocked(busy=False)

            try:
                await self._start_slot(slot, timeout_s=900.0)
            except Exception:
                return

    async def snapshot(self) -> dict[str, object]:
        async with self._lock:
            slots = list(self._slots.values())

        ready_idle = sum(1 for slot in slots if slot.ready and not slot.busy)
        ready_busy = sum(1 for slot in slots if slot.ready and slot.busy)
        starting = sum(1 for slot in slots if slot.starting)
        errors = [
            {"slot_id": slot.slot_id, "ws_url": slot.ws_url, "error": slot.last_error}
            for slot in slots
            if slot.last_error
        ]
        if self._last_error and not errors:
            errors.append({"slot_id": None, "ws_url": None, "error": self._last_error})

        return {
            "max_instances": self.max_instances,
            "min_idle_instances": self.min_idle_instances,
            "total_slots": len(slots),
            "ready_idle": ready_idle,
            "ready_busy": ready_busy,
            "starting": starting,
            "errors": errors,
        }

    async def healthcheck(self) -> tuple[bool, Optional[str], dict[str, object]]:
        snapshot = await self.snapshot()

        if snapshot["ready_idle"] or snapshot["ready_busy"]:
            return True, None, snapshot

        if snapshot["starting"]:
            return False, "speech-to-speech pipelines are still starting", snapshot

        errors = snapshot["errors"]
        if errors:
            first_error = errors[0]["error"]
            return False, str(first_error), snapshot

        if self.max_instances == 0:
            return False, "session router has no capacity configured", snapshot

        return False, "no speech-to-speech pipeline is ready", snapshot

    def _create_slot_unlocked(self, *, busy: bool) -> PipelineSlot:
        slot_id = self._next_slot_id
        self._next_slot_id += 1

        slot = PipelineSlot(
            slot_id=slot_id,
            host=self.host,
            port=self.base_port + slot_id,
            ws_path=self.ws_path,
            busy=busy,
            starting=True,
        )
        self._slots[slot_id] = slot
        return slot

    def _find_idle_slot_unlocked(self) -> Optional[PipelineSlot]:
        for slot in self._slots.values():
            if slot.ready and not slot.busy:
                return slot
        return None

    async def _start_slot(self, slot: PipelineSlot, timeout_s: float) -> None:
        cmd = self.build_command(slot.host, slot.port)
        logger.info("Starting speech-to-speech slot %s at %s", slot.slot_id, slot.ws_url)

        env = os.environ.copy()
        slot.process = subprocess.Popen(
            cmd,
            cwd=self.repo_dir,
            env=env,
            stdout=None,
            stderr=None,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        try:
            await self.wait_for_ready(slot.host, slot.port, slot.process, timeout_s)
        except Exception as exc:
            async with self._condition:
                slot.ready = False
                slot.starting = False
                slot.last_error = str(exc)
                self._last_error = slot.last_error
                self._slots.pop(slot.slot_id, None)
                self._condition.notify_all()

            await asyncio.to_thread(self._stop_slot_process, slot)
            logger.error("speech-to-speech slot %s failed to become ready: %s", slot.slot_id, exc)
            raise

        async with self._condition:
            if self._closed:
                self._condition.notify_all()
                await asyncio.to_thread(self._stop_slot_process, slot)
                return

            slot.ready = True
            slot.starting = False
            slot.last_error = None
            self._last_error = None
            self._condition.notify_all()

        logger.info("speech-to-speech slot %s is ready at %s", slot.slot_id, slot.ws_url)

    def _stop_slot_process(self, slot: PipelineSlot) -> None:
        if slot.process is None:
            return

        if slot.process.poll() is not None:
            return

        try:
            if os.name != "nt":
                os.killpg(os.getpgid(slot.process.pid), signal.SIGTERM)
            else:
                slot.process.terminate()
            slot.process.wait(timeout=20)
        except Exception:
            logger.exception("Graceful shutdown failed for slot %s, killing subprocess", slot.slot_id)
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(slot.process.pid), signal.SIGKILL)
                else:
                    slot.process.kill()
            except Exception:
                logger.exception("Failed to kill subprocess for slot %s", slot.slot_id)
        finally:
            slot.process = None
            slot.ready = False
            slot.starting = False

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("session router is shutting down")

import asyncio
import contextlib
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
    ws_url: str = field(init=False)

    def __post_init__(self) -> None:
        self.ws_url = f"ws://{self.host}:{self.port}{self.ws_path}"


class SessionRouter:
    """Manages a single speech-to-speech process.

    The process itself handles concurrent sessions internally via
    ``--num_pipelines``.  This router only starts/stops the process and
    exposes a stable ``ws_url`` that callers can connect to repeatedly.
    """

    def __init__(
        self,
        *,
        host: str,
        base_port: int,
        repo_dir: str,
        build_command: BuildCommand,
        wait_for_ready: WaitForReady,
        ws_path: str = "",
        max_sessions: int = 1,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")

        self.host = host
        self.base_port = base_port
        self.ws_path = ws_path
        self.repo_dir = repo_dir
        self.build_command = build_command
        self.wait_for_ready = wait_for_ready
        self.max_sessions = max_sessions

        self._slot = PipelineSlot(slot_id=0, host=host, port=base_port, ws_path=ws_path)
        self._process: Optional[subprocess.Popen] = None
        self._ready = False
        self._starting = False
        self._closed = False
        self._active_sessions = 0
        self._last_error: Optional[str] = None
        self._watchdog_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self._start_process(timeout_s=900.0)

    async def stop(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._stop_process)
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watchdog_task
            self._watchdog_task = None

    async def acquire(self) -> PipelineSlot:
        if self._closed:
            raise RuntimeError("session router is shutting down")
        self._refresh_process_state()
        if not self._ready:
            raise RuntimeError(self._last_error or "speech-to-speech pipeline is not ready")
        if self._active_sessions >= self.max_sessions:
            raise RuntimeError(f"all {self.max_sessions} pipeline session(s) are in use")
        self._active_sessions += 1
        return self._slot

    async def release(self, slot_id: int) -> None:
        self._active_sessions = max(self._active_sessions - 1, 0)

    async def snapshot(self) -> dict[str, object]:
        self._refresh_process_state()
        return {
            "ws_url": self._slot.ws_url,
            "ready": self._ready,
            "starting": self._starting,
            "max_sessions": self.max_sessions,
            "active_sessions": self._active_sessions,
            "free_sessions": max(self.max_sessions - self._active_sessions, 0) if self._ready else 0,
            "errors": [self._last_error] if self._last_error else [],
        }

    async def healthcheck(self) -> tuple[bool, Optional[str], dict[str, object]]:
        snap = await self.snapshot()

        if self._ready:
            return True, None, snap

        if self._starting:
            return False, "speech-to-speech pipeline is still starting", snap

        if self._last_error:
            return False, self._last_error, snap

        return False, "speech-to-speech pipeline is not ready", snap

    async def _start_process(self, timeout_s: float) -> None:
        cmd = self.build_command(self._slot.host, self._slot.port)
        logger.info("Starting speech-to-speech process at %s", self._slot.ws_url)
        self._starting = True

        env = os.environ.copy()
        self._process = subprocess.Popen(
            cmd,
            cwd=self.repo_dir,
            env=env,
            stdout=None,
            stderr=None,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        try:
            await self.wait_for_ready(self._slot.host, self._slot.port, self._process, timeout_s)
            if self._process.poll() is not None:
                raise RuntimeError(f"speech-to-speech process exited with code {self._process.returncode}")
        except Exception as exc:
            self._ready = False
            self._starting = False
            self._last_error = str(exc)
            await asyncio.to_thread(self._stop_process)
            logger.error("speech-to-speech process failed to become ready: %s", exc)
            raise

        if self._closed:
            await asyncio.to_thread(self._stop_process)
            return

        self._ready = True
        self._starting = False
        self._last_error = None
        logger.info("speech-to-speech process is ready at %s", self._slot.ws_url)
        self._watchdog_task = asyncio.create_task(
            self._watch_process(self._process),
            name="speech-to-speech-process-watchdog",
        )

    async def _watch_process(self, process: subprocess.Popen) -> None:
        try:
            returncode = await asyncio.to_thread(process.wait)
            self._record_process_exit(process, returncode)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._process is process and not self._closed:
                self._ready = False
                self._starting = False
                self._last_error = f"failed to monitor speech-to-speech process: {exc}"
                logger.exception(self._last_error)
        finally:
            if self._watchdog_task is asyncio.current_task():
                self._watchdog_task = None

    def _refresh_process_state(self) -> None:
        process = self._process
        if process is None:
            return

        returncode = process.poll()
        if returncode is not None:
            self._record_process_exit(process, returncode)

    def _record_process_exit(self, process: subprocess.Popen, returncode: int) -> None:
        if self._process is not process or self._closed:
            return

        error = f"speech-to-speech process exited with code {returncode}"
        already_reported = not self._ready and self._last_error == error
        self._ready = False
        self._starting = False
        self._last_error = error
        if not already_reported:
            logger.error(error)

    def _stop_process(self) -> None:
        if self._process is None:
            return

        if self._process.poll() is not None:
            self._process = None
            self._ready = False
            return

        try:
            if os.name != "nt":
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()
            self._process.wait(timeout=20)
        except Exception:
            logger.exception("Graceful shutdown failed, killing subprocess")
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                else:
                    self._process.kill()
            except Exception:
                logger.exception("Failed to kill subprocess")
        finally:
            self._process = None
            self._ready = False
            self._starting = False

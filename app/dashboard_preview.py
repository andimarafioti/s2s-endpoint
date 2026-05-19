import math
import time
from typing import Optional


class DashboardPreviewSessionManager:
    """Synthetic session manager used to preview dashboard UI without HF endpoints."""

    def __init__(self, *, endpoint_slots: int = 2, time_fn=time.time) -> None:
        self.endpoint_slots = max(int(endpoint_slots), 1)
        self._time_fn = time_fn
        self._started_at_s = self._time_fn()
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def allocate(self, lb_base_url: str) -> dict[str, object]:
        raise RuntimeError("session allocation is disabled in dashboard preview mode")

    async def handle_event(self, session_id: str, session_token: str, event: str) -> dict[str, object]:
        raise KeyError("dashboard preview mode does not track real sessions")

    async def healthcheck(self) -> tuple[bool, Optional[str], dict[str, object]]:
        elapsed_s = max(self._time_fn() - self._started_at_s, 0.0)
        wave = (math.sin(elapsed_s / 24.0) + 1.0) / 2.0
        connected_sessions = 1 + int(round(wave * 4))
        pending_sessions = int((elapsed_s // 18) % 3)
        phase = int(elapsed_s // 30) % 4

        endpoint_states = [
            ("preview-compute-01", "running", False, False),
            ("preview-compute-02", "running", False, False),
            ("preview-compute-03", "initializing" if phase in {0, 1} else "running", phase in {0, 1}, False),
            ("preview-compute-04", "paused" if phase != 3 else "updating", False, phase == 3),
        ]
        running_endpoint_count = sum(1 for _, status, _, _ in endpoint_states if status == "running")
        waking_endpoint_count = sum(1 for _, _, waking, _ in endpoint_states if waking)
        active_remaining = connected_sessions
        endpoints = []

        for name, status, waking, parking in endpoint_states:
            is_running = status == "running"
            active_sessions = min(active_remaining, self.endpoint_slots) if is_running else 0
            active_remaining -= active_sessions
            free_slots = max(self.endpoint_slots - active_sessions, 0) if is_running else 0
            endpoints.append(
                {
                    "name": name,
                    "status": status,
                    "waking": waking,
                    "parking": parking,
                    "active_sessions": active_sessions,
                    "free_slots": free_slots,
                    "url": f"https://{name}.preview.local" if is_running else None,
                    "last_error": None,
                }
            )

        free_slots = sum(int(endpoint["free_slots"]) for endpoint in endpoints)
        effective_free_slots = free_slots + waking_endpoint_count * self.endpoint_slots
        snapshot = {
            "pending_sessions": pending_sessions,
            "connected_sessions": connected_sessions,
            "preview_mode": True,
            "router": {
                "running_endpoints": running_endpoint_count,
                "waking_endpoints": waking_endpoint_count,
                "active_sessions": connected_sessions,
                "free_slots": free_slots,
                "effective_free_slots": effective_free_slots,
                "errors": [],
                "endpoints": endpoints,
            },
        }
        return True, "Dashboard preview mode uses synthetic endpoint data.", snapshot

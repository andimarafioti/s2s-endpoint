from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class SpeechProxyTelemetryTarget:
    service: str
    url: str

    def __post_init__(self) -> None:
        if self.service not in {"stt", "tts"}:
            raise ValueError("service must be 'stt' or 'tts'")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("speech proxy URL must use http or https")


class SpeechProxyTelemetryClient:
    """Fetch correlated STT/TTS metrics for the load-balancer dashboard."""

    def __init__(
        self,
        targets: tuple[SpeechProxyTelemetryTarget, ...],
        *,
        api_key: str | None,
        timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self.targets = targets
        self.api_key = api_key
        self.timeout_s = timeout_s
        self._client = client or httpx.AsyncClient(timeout=timeout_s, follow_redirects=True)
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def snapshot(self, window_s: float) -> dict[str, object]:
        results = await asyncio.gather(
            *(self._fetch(target, window_s) for target in self.targets),
        )
        return {
            "configured": bool(self.targets),
            "window_s": window_s,
            "services": {target.service: result for target, result in zip(self.targets, results, strict=True)},
        }

    async def _fetch(self, target: SpeechProxyTelemetryTarget, window_s: float) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = await self._client.get(
                f"{target.url.rstrip('/')}/metrics",
                headers=headers,
                params={"window_s": window_s},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("metrics response must be a JSON object")
            payload["reachable"] = True
            return payload
        except httpx.HTTPStatusError as exc:
            error = f"HTTP {exc.response.status_code}"
        except Exception as exc:
            error = type(exc).__name__
        return {
            "status": "unavailable",
            "service": target.service,
            "reachable": False,
            "error": error,
        }

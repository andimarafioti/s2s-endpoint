"""Compose gateway headroom for the allocator's single admission owner."""

from __future__ import annotations

import asyncio
import time
from collections import Counter

import httpx
from pydantic import Field, model_validator

from app.speech_route_catalog import CatalogModel, PoolId

SERVICES = ("stt", "llm", "tts")


class PipelineRoute(CatalogModel):
    stt: PoolId
    llm: PoolId
    tts: PoolId


class PipelineCapacityConfig(CatalogModel):
    routes: dict[PoolId, PipelineRoute] = Field(min_length=1)
    default: PoolId
    reserve_sessions: int = Field(default=5, ge=0, le=300)
    refresh_interval_s: float = Field(default=5, gt=0, le=30)
    snapshot_max_age_s: float = Field(default=15, gt=0, le=60)
    llm_protocol: str = "chat_completions"

    @model_validator(mode="after")
    def validate_routes(self):
        if self.default not in self.routes:
            raise ValueError("pipeline default must name a configured route")
        if self.llm_protocol not in {"chat_completions", "responses"}:
            raise ValueError("llm_protocol must match the CPU adapter")
        if self.snapshot_max_age_s <= self.refresh_interval_s:
            raise ValueError("snapshot_max_age_s must exceed the refresh interval")
        return self


class PipelineCapacity:
    def __init__(
        self,
        config: PipelineCapacityConfig,
        urls: dict[str, str],
        api_key: str,
        *,
        ingress_api_key: str | None = None,
        client=None,
        time_fn=time.monotonic,
    ):
        if set(urls) != set(SERVICES) or not api_key:
            raise ValueError("pipeline capacity requires all three gateway URLs and a capacity API key")
        self.config = config
        self.urls = {service: url.rstrip("/") for service, url in urls.items()}
        self._client = client or httpx.AsyncClient(timeout=5)
        self._owns_client = client is None
        self._headers = {"X-Speech-Capacity-Authorization": f"Bearer {api_key}"}
        if ingress_api_key:
            self._headers["Authorization"] = f"Bearer {ingress_api_key}"
        self._time = time_fn
        self._views: dict[tuple[str, str], dict] = {}
        self._seen: dict[str, float] = {}
        self._counts: dict[tuple[str, str], int] = {}

    async def close(self):
        if self._owns_client:
            await self._client.aclose()

    def resolve(self, name: str | None) -> str:
        name = self.config.default if name is None else name
        if name not in self.config.routes:
            raise ValueError("unknown pipeline route")
        return name

    def pool_counts(self, sessions: dict[str, int]) -> dict[tuple[str, str], int]:
        counts = Counter({(s, getattr(r, s)): 0 for r in self.config.routes.values() for s in SERVICES})
        for name, count in sessions.items():
            route = self.config.routes.get(name)
            if route is None:
                # A surviving connection from an older allocator/configuration
                # must not disappear from capacity estimates after a restart.
                for key in counts:
                    counts[key] += count
            else:
                for service in SERVICES:
                    counts[(service, getattr(route, service))] += count
        return dict(counts)

    async def refresh(self, sessions: dict[str, int]):
        counts = self.pool_counts(sessions)

        async def fetch(service):
            selected = {pool: count for (stage, pool), count in counts.items() if stage == service}
            try:
                response = await self._client.post(
                    f"{self.urls[service]}/internal/capacity",
                    headers=self._headers,
                    json={"session_counts": selected, "reserve_sessions": self.config.reserve_sessions},
                    timeout=5,
                )
                response.raise_for_status()
                views = response.json()["pools"]
                for pool in selected:
                    view = views[pool]
                    protocol = {"stt": "transcriptions", "tts": "speech", "llm": self.config.llm_protocol}[service]
                    if protocol not in view["protocols"]:
                        raise ValueError("pool protocol does not match the CPU adapter")
                    if service == "tts" and not view["voices"]:
                        raise ValueError("TTS pool must declare a voice")
                    if any(
                        type(view[field]) is not int or view[field] < 0
                        for field in ("available_sessions", "admissible_sessions")
                    ):
                        raise ValueError("invalid gateway capacity")
                    if view.get("demand_stale", True):
                        raise ValueError("gateway demand is stale")
                    if not all(
                        isinstance(view.get(field), str) and view[field]
                        for field in ("model", "request_model", "provider", "profile")
                    ):
                        raise ValueError("missing route identity or workload profile")
                for pool, count in selected.items():
                    self._views[(service, pool)] = views[pool]
                    self._counts[(service, pool)] = count
                self._seen[service] = self._time()
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                self._seen.pop(service, None)

        await asyncio.gather(*(fetch(s) for s in SERVICES))

    def _remaining(self, key, counts, field):
        service, _ = key
        if service not in self._seen or self._time() - self._seen[service] >= self.config.snapshot_max_age_s:
            return 0
        # Claims since this snapshot immediately spend the shared pool's
        # estimate. Releases never create headroom from an older observation.
        return max(0, self._views[key][field] - max(0, counts[key] - self._counts[key]))

    def can_admit(self, name: str, sessions: dict[str, int]) -> bool:
        route = self.config.routes[name]
        counts = self.pool_counts(sessions)
        return all(self._remaining((s, getattr(route, s)), counts, "admissible_sessions") >= 1 for s in SERVICES)

    def routing(self, name: str) -> dict:
        route = self.config.routes[name]
        routes = {}
        for service in SERVICES:
            view = self._views[(service, getattr(route, service))]
            routes[service] = {
                "model": view["request_model"],
                "provider": view["provider"],
                "protocol": {"stt": "transcriptions", "tts": "speech", "llm": self.config.llm_protocol}[service],
            }
            if service == "tts":
                routes[service]["voice"] = view["voices"][0]
        return {"pipeline": name, "routes": routes}

    def snapshot(self, sessions: dict[str, int], cpu_slots: int) -> dict:
        counts = self.pool_counts(sessions)
        routes = {}
        for name, route in self.config.routes.items():
            stages = {s: self._remaining((s, getattr(route, s)), counts, "available_sessions") for s in SERVICES}
            stages["cpu"] = cpu_slots
            limiting = min(stages, key=stages.get)
            routes[name] = {"available_pipelines": stages[limiting], "limiting_stage": limiting, "stages": stages}
        return {
            "reserve_target": self.config.reserve_sessions,
            "available_pipelines": min(r["available_pipelines"] for r in routes.values()),
            "routes": routes,
            "pools": {
                f"{s}/{p}": {"sessions": count, **self._views.get((s, p), {})} for (s, p), count in counts.items()
            },
        }

"""Compose gateway headroom for the allocator's single admission owner."""

from __future__ import annotations

import asyncio
import time
from collections import Counter

import httpx
from pydantic import Field, model_validator

from app.speech_route_catalog import CatalogModel, Identifier, PoolId

SERVICES = ("stt", "llm", "tts")


class PipelineRoute(CatalogModel):
    stt: PoolId | None
    llm: PoolId | None
    tts: PoolId | None


class ModelChoice(CatalogModel):
    model: Identifier
    provider: Identifier | None = None


class PipelineCapacityConfig(CatalogModel):
    routes: dict[PoolId, PipelineRoute] = Field(min_length=1)
    default: PoolId
    reserve_sessions: int = Field(default=5, ge=0, le=300)
    refresh_interval_s: float = Field(default=5, gt=0, le=30)
    snapshot_max_age_s: float = Field(default=15, gt=0, le=60)
    llm_protocol: str = "chat_completions"
    session_updates_enabled: bool = False

    @model_validator(mode="after")
    def validate_routes(self):
        if self.default not in self.routes:
            raise ValueError("pipeline default must name a configured route")
        if self.llm_protocol not in {"chat_completions", "responses"}:
            raise ValueError("llm_protocol must match the CPU adapter")
        if self.snapshot_max_age_s <= self.refresh_interval_s:
            raise ValueError("snapshot_max_age_s must exceed the refresh interval")
        if not self.session_updates_enabled and any(
            getattr(r, s) is None for r in self.routes.values() for s in SERVICES
        ):
            raise ValueError("partial pipeline routes require session_updates_enabled")
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
        if name is None:
            name = "@::" if self.config.session_updates_enabled else self.config.default
        self._pools(name)
        return name

    def _pools(self, name: str) -> dict[str, tuple[str, ...]]:
        if name in self.config.routes:
            route = self.config.routes[name]
            return {s: (getattr(route, s),) if getattr(route, s) is not None else () for s in SERVICES}
        # A compact, self-describing accounting key lets compute health retain
        # partial selections and an old/new union through allocator restarts.
        if self.config.session_updates_enabled and name.startswith("@") and len(name) <= 800:
            parts = name[1:].split(":")
            if len(parts) == len(SERVICES):
                pools = {s: tuple(part.split("+")) if part else () for s, part in zip(SERVICES, parts)}
                if all(
                    len(values) <= 2
                    and len(set(values)) == len(values)
                    and all(p in {getattr(r, s) for r in self.config.routes.values()} for p in values)
                    for s, values in pools.items()
                ):
                    return pools
        raise ValueError("unknown pipeline route")

    @staticmethod
    def _key(pools: dict[str, tuple[str, ...]]) -> str:
        return "@" + ":".join("+".join(sorted(pools[s])) for s in SERVICES)

    def select_models(self, current: str | None, models: dict) -> str:
        if not self.config.session_updates_enabled:
            raise ValueError("session model selection is disabled")
        if not isinstance(models, dict) or set(models) - set(SERVICES):
            raise ValueError("models must contain only stt, llm and tts")
        pools = self._pools(self.resolve(current))
        for service, raw in models.items():
            if raw is None:
                pools[service] = ()
                continue
            choice = ModelChoice.model_validate({"model": raw} if isinstance(raw, str) else raw)
            candidates = [
                pool
                for (stage, pool), view in self._views.items()
                if stage == service
                and choice.model in {view["model"], view["request_model"], *view.get("aliases", [])}
                and (choice.provider is None or choice.provider == view["provider"])
            ]
            if len(candidates) > 1:
                candidates = [p for p in candidates if self._views[(service, p)].get("default_model_route")]
            if len(candidates) != 1:
                raise ValueError("unknown or ambiguous model/provider selection")
            pools[service] = (candidates[0],)
        return self._key(pools)

    def hold_selection(self, current: str, proposed: str) -> str:
        old, new = self._pools(current), self._pools(proposed)
        return self._key({s: tuple(set(old[s]) | set(new[s])) for s in SERVICES})

    def can_switch(self, current: str, proposed: str, sessions: dict[str, int]) -> bool:
        old, new = self._pools(current), self._pools(proposed)
        counts = self.pool_counts(sessions)
        return all(
            self._remaining((s, p), counts, "admissible_sessions") >= 1
            for s in SERVICES
            for p in set(new[s]) - set(old[s])
        )

    def pool_counts(self, sessions: dict[str, int]) -> dict[tuple[str, str], int]:
        counts = Counter(
            {(s, getattr(r, s)): 0 for r in self.config.routes.values() for s in SERVICES if getattr(r, s) is not None}
        )
        for name, count in sessions.items():
            try:
                pools = self._pools(name)
            except ValueError:
                # A surviving connection from an older allocator/configuration
                # must not disappear from capacity estimates after a restart.
                for key in counts:
                    counts[key] += count
            else:
                for service, selected in pools.items():
                    for pool in selected:
                        counts[(service, pool)] += count
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
        counts = self.pool_counts(sessions)
        return all(
            self._remaining((s, p), counts, "admissible_sessions") >= 1
            for s, pools in self._pools(name).items()
            for p in pools
        )

    def routing(self, name: str) -> dict:
        routes = {}
        for service, pools in self._pools(name).items():
            if not pools:
                routes[service] = None
                continue
            if len(pools) != 1:
                raise ValueError("an accounting hold is not an inference selection")
            view = self._views[(service, pools[0])]
            routes[service] = {
                "model": view["request_model"],
                "provider": view["provider"],
                "protocol": {"stt": "transcriptions", "tts": "speech", "llm": self.config.llm_protocol}[service],
            }
            if service == "tts":
                routes[service]["voice"] = view["voices"][0]
                if self.config.session_updates_enabled:
                    routes[service]["voices"] = view["voices"]
            if self.config.session_updates_enabled and service == "llm":
                routes[service]["capabilities"] = {
                    key: value
                    for key, value in view.get("capabilities", {}).items()
                    if key in {"tools", "images", "audio_input", "context_window", "continuation"}
                }
        return {
            "pipeline": name,
            "routes": routes,
            **({"updates_enabled": True} if self.config.session_updates_enabled else {}),
        }

    def snapshot(self, sessions: dict[str, int], cpu_slots: int) -> dict:
        counts = self.pool_counts(sessions)
        routes = {}
        for name in self.config.routes:
            stages = {
                s: self._remaining((s, p), counts, "available_sessions")
                for s, pools in self._pools(name).items()
                for p in pools
            }
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

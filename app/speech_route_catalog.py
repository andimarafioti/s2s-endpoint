"""Validated, deployment-owned routes. Each pool has one compatible identity and owner."""

from __future__ import annotations

from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.speech_worker_lifecycle import WorkerLifecycleSettings

Identifier = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/@:-]*$")]
EnvName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
PoolId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Protocol = Literal["transcriptions", "speech", "chat_completions", "responses"]
PROTOCOL_PATHS = {
    "/v1/audio/transcriptions": "transcriptions",
    "/v1/audio/speech": "speech",
    "/v1/chat/completions": "chat_completions",
    "/v1/responses": "responses",
}


class CatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False, hide_input_in_errors=True)


class RouteBackend(CatalogModel):
    name: PoolId
    url: str = ""

    @model_validator(mode="after")
    def validate_url(self):
        if self.url:
            parsed = urlsplit(self.url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("backend URLs require http(s), a host, and no embedded credentials")
            if parsed.query or parsed.fragment:
                raise ValueError("backend URLs cannot contain query strings or fragments")
        return self


class RouteCapabilities(CatalogModel):
    tools: bool = False
    images: bool = False
    audio_input: bool = False
    context_window: int | None = Field(default=None, gt=0)
    voices: tuple[str, ...] = ()
    audio_formats: tuple[str, ...] = ()
    continuation: Literal["full_context", "provider_state"] = "full_context"
    cache_controls: tuple[Literal["prompt_cache_key", "prompt_cache_retention"], ...] = ()


class RoutePolicy(CatalogModel):
    target_work: float = Field(default=8, gt=0)
    max_work: float | None = Field(default=None, gt=0)
    latency_target: float = Field(default=0.5, gt=0)
    latency_weight: float = Field(default=0.25, ge=0)
    ewma_alpha: float = Field(default=0.2, gt=0, le=1)
    failure_threshold: int = Field(default=2, ge=1)
    health_path: str = "/health"
    health_interval_s: float = Field(default=10, gt=0)
    health_timeout_s: float = Field(default=5, gt=0)
    request_timeout_s: float = Field(default=120, gt=0)
    max_attempts: int = Field(default=2, ge=1)
    stt_audio_equivalent_s: float = Field(default=5, gt=0)
    tts_warmup_enabled: bool = True
    llm_warmup_enabled: bool = True
    tts_warmup_voice: str = "aiden"
    tts_warmup_language: str = "English"


class ExternalCapacity(CatalogModel):
    max_concurrency: int = Field(gt=0)
    requests_per_minute: int = Field(gt=0)


class SessionWorkload(CatalogModel):
    profile: str = Field(min_length=1, max_length=2000)
    work_per_session: float = Field(gt=0)
    requests_per_minute: float = Field(default=1, gt=0)


class SessionDemandUpdate(CatalogModel):
    session_counts: dict[PoolId, Annotated[int, Field(ge=0)]]
    reserve_sessions: int = Field(ge=0, le=300)


class SpeechRoute(CatalogModel):
    pool: PoolId
    model: Identifier
    provider: Identifier
    kind: Literal["self_hosted", "external"]
    revision: Identifier
    upstream_model: Identifier
    protocols: tuple[Protocol, ...] = Field(min_length=1)
    capabilities: RouteCapabilities
    credential_env: EnvName
    access_key_env: EnvName | None = None
    aliases: tuple[Identifier, ...] = ()
    backends: tuple[RouteBackend, ...] = Field(min_length=1)
    policy: RoutePolicy = Field(default_factory=RoutePolicy)
    capacity: ExternalCapacity | None = None
    session_workload: SessionWorkload | None = None
    lifecycle: dict[str, int | float] | None = None
    namespace: str | None = None
    control_token_env: EnvName | None = None

    @model_validator(mode="after")
    def validate_pool(self):
        if self.policy.max_work is not None and self.policy.max_work < self.policy.target_work:
            raise ValueError("max_work must be >= target_work")
        if self.session_workload and self.kind == "self_hosted" and self.policy.max_work is None:
            raise ValueError("self-hosted session headroom requires max_work")
        if self.kind == "external":
            if self.policy.max_work is not None:
                raise ValueError("external routes use capacity.max_concurrency rather than max_work")
            if self.lifecycle is not None or self.namespace or self.control_token_env:
                raise ValueError("external routes cannot configure HF lifecycle operations")
            if self.capacity is None or len(self.backends) != 1:
                raise ValueError("external routes require capacity limits and one provider API backend")
        elif self.capacity is not None:
            raise ValueError("hard provider capacity is only supported for external routes")
        if self.kind == "self_hosted" and self.capabilities.continuation != "full_context":
            raise ValueError("self-hosted pools require full_context continuation")
        if self.lifecycle is not None:
            if not self.namespace or not self.control_token_env:
                raise ValueError("managed pools require namespace and control_token_env")
            for name in ("min_warm", "max_workers", "max_restart_attempts"):
                if name in self.lifecycle and type(self.lifecycle[name]) is not int:
                    raise ValueError(f"{name} must be an integer")
            try:
                lifecycle = WorkerLifecycleSettings(**self.lifecycle)
            except TypeError as exc:
                raise ValueError("invalid lifecycle settings") from exc
            if lifecycle.max_workers > len(self.backends):
                raise ValueError("max_workers exceeds pool inventory")
        elif any(not backend.url for backend in self.backends):
            raise ValueError("unmanaged backends require URLs")
        names = [backend.name for backend in self.backends]
        if len(names) != len(set(names)):
            raise ValueError("backend names must be unique within a pool")
        return self

    def labels(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in ("model", "provider", "pool", "revision")}

    def validate_request(self, path: str, payload: dict[str, Any]) -> None:
        if PROTOCOL_PATHS[path] not in self.protocols:
            raise ValueError("the selected route does not support this API")
        caps = self.capabilities
        if payload.get("background"):
            raise ValueError("background generations are not supported by request-scoped pools")
        if not caps.tools and (
            payload.get("tools") or payload.get("functions") or payload.get("tool_choice") not in (None, "none")
        ):
            raise ValueError("the selected route does not support tools")
        if caps.continuation == "full_context" and (payload.get("previous_response_id") or payload.get("conversation")):
            raise ValueError("the selected route requires complete context; backend-local continuation is unsupported")
        for key in ("prompt_cache_key", "prompt_cache_retention"):
            if payload.get(key) is not None and key not in caps.cache_controls:
                raise ValueError(f"the selected route does not support {key}")
        for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            if payload.get(key) is not None:
                value = payload[key]
                if type(value) is not int or value < 1 or (caps.context_window and value > caps.context_window):
                    raise ValueError(f"{key} must fit the selected route's context window")
        if path == "/v1/audio/speech":
            if payload.get("voice") not in caps.voices:
                raise ValueError("the selected route does not support this voice")
            if payload.get("response_format", "mp3") not in caps.audio_formats:
                raise ValueError("the selected route does not support this audio format")
        # Inspect input content and tool results, not tool JSON schemas.
        pending = [payload.get("messages", payload.get("input", []))]
        while pending:
            item = pending.pop()
            if isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if (kind in {"image_url", "input_image"} or "image_url" in item) and not caps.images:
                    raise ValueError("the selected route does not support images")
                if kind in {"input_audio", "audio"} and not caps.audio_input:
                    raise ValueError("the selected route does not support audio input")
                if kind == "item_reference" and caps.continuation == "full_context":
                    raise ValueError("the selected route requires complete input items")
                if not caps.tools and (
                    kind in {"function_call", "function_call_output"}
                    or item.get("tool_calls")
                    or item.get("function_call")
                    or item.get("role") in {"tool", "function"}
                ):
                    raise ValueError("the selected route does not support tool history")
                if "content" in item:
                    pending.append(item["content"])
                if kind == "function_call_output":
                    pending.append(item.get("output"))


class SpeechRouteCatalog(CatalogModel):
    pools: tuple[SpeechRoute, ...] = Field(min_length=1)
    defaults: dict[str, str] = Field(default_factory=dict)

    def secret_names(self) -> list[str]:
        return sorted(
            {
                name
                for route in self.pools
                for name in (route.credential_env, route.access_key_env, route.control_token_env)
                if name
            }
        )

    @model_validator(mode="after")
    def validate_catalog(self):
        pools = {route.pool: route for route in self.pools}
        if len(pools) != len(self.pools):
            raise ValueError("pool identities must be unique")
        aliases = [alias for route in self.pools for alias in route.aliases]
        if len(aliases) != len(set(aliases)) or set(aliases) & {route.model for route in self.pools}:
            raise ValueError("route aliases must be unique and distinct from logical models")
        urls: dict[str, str] = {}
        workers: set[tuple[str, str]] = set()
        for route in self.pools:
            if (
                route.session_workload
                and not route.aliases
                and sum(other.model == route.model and other.provider == route.provider for other in self.pools) > 1
            ):
                raise ValueError("session capacity requires a unique alias for each pool sharing a model/provider")
            for backend in route.backends:
                url = backend.url.rstrip("/")
                if url in urls and (route.kind != "external" or urls[url] != "external"):
                    raise ValueError("self-hosted backend URLs cannot belong to multiple pools")
                if url:
                    urls[url] = route.kind
                if route.lifecycle is not None:
                    key = (route.namespace, backend.name)
                    if key in workers:
                        raise ValueError("an HF endpoint cannot have multiple pool controllers")
                    workers.add(key)
        for model, pool in self.defaults.items():
            if pool not in pools or pools[pool].model != model:
                raise ValueError("each default must reference a pool for that logical model")
        return self

    def validate_service(self, service: str) -> None:
        allowed = {"stt": {"transcriptions"}, "tts": {"speech"}, "llm": {"chat_completions", "responses"}}[service]
        for route in self.pools:
            if not set(route.protocols) <= allowed:
                raise ValueError("route protocols must match SPEECH_PROXY_SERVICE")
            if service == "llm" and route.capabilities.context_window is None:
                raise ValueError("LLM routes must declare a context_window")
            if service == "tts" and (not route.capabilities.voices or not route.capabilities.audio_formats):
                raise ValueError("TTS routes must declare voices and audio_formats")

    def resolve(self, model: str, provider: str | None) -> SpeechRoute:
        candidates = [route for route in self.pools if model == route.model or model in route.aliases]
        if provider is not None:
            candidates = [route for route in candidates if route.provider == provider]
        if len(candidates) == 1:
            return candidates[0]
        for route in candidates:
            if self.defaults.get(model) == route.pool:
                return route
        if not candidates:
            raise ValueError("unknown model/provider combination")
        raise ValueError("ambiguous model/provider selection; configure a default or request a route alias")

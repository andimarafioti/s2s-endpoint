# Model/provider routes and compatible pools

The speech proxy can resolve a logical `model` and optional provider before it
reserves an inference worker. Each configured pool has its own router, health
checks, request accounting, latency targets, and optional HF lifecycle controller.
Retries stay inside that pool. A busy or failed model never selects a different
model/provider as a fallback.

Leave `SPEECH_ROUTE_CATALOG` unset to retain the existing `SPEECH_BACKENDS`
single-model deployment and its defaults. Catalog mode is opt-in and supports the
same separately deployable `stt`, `tts`, and `llm` gateways.

## Configure routes

Set `SPEECH_ROUTE_CATALOG` to a JSON catalog, or pass its file to the deployment
helper below. The following is an illustrative LLM catalog: replace the model
IDs, revisions, endpoint names, API URL, capabilities, and budgets with values
verified for your deployment. These examples do not assert that a particular
model is offered by an external provider.

```json
{
  "defaults": {"model-a": "model-a-r1"},
  "pools": [
    {
      "pool": "model-a-r1",
      "model": "model-a",
      "provider": "hf",
      "kind": "self_hosted",
      "revision": "verified-model-and-runtime-r1",
      "upstream_model": "org/model-a",
      "aliases": ["model-a-hf"],
      "protocols": ["chat_completions", "responses"],
      "capabilities": {"tools": true, "context_window": 32768},
      "credential_env": "HF_INFERENCE_TOKEN",
      "namespace": "your-namespace",
      "control_token_env": "HF_CONTROL_TOKEN",
      "backends": [{"name": "model-a-01"}, {"name": "model-a-02"}],
      "policy": {"target_work": 32, "latency_target": 0.5},
      "lifecycle": {"min_warm": 1, "max_workers": 2}
    },
    {
      "pool": "model-b-r1",
      "model": "model-b",
      "provider": "hf",
      "kind": "self_hosted",
      "revision": "verified-model-and-runtime-r1",
      "upstream_model": "org/model-b",
      "protocols": ["chat_completions"],
      "capabilities": {"context_window": 16384},
      "credential_env": "HF_INFERENCE_TOKEN",
      "backends": [{"name": "model-b-01", "url": "https://model-b.example"}],
      "policy": {"target_work": 16, "latency_target": 0.7}
    },
    {
      "pool": "model-a-api",
      "model": "model-a",
      "provider": "example-api",
      "kind": "external",
      "revision": "verified-provider-deployment-r1",
      "upstream_model": "provider-model-a",
      "aliases": ["model-a-external"],
      "protocols": ["chat_completions"],
      "capabilities": {"context_window": 16384},
      "credential_env": "EXTERNAL_API_KEY",
      "backends": [{"name": "external-api", "url": "https://api.example.com"}],
      "policy": {"request_timeout_s": 60, "max_attempts": 2},
      "capacity": {"max_concurrency": 8, "requests_per_minute": 120}
    }
  ]
}
```

Each pool declares one compatible model/provider/deployment identity. Use aliases
to expose additional selections for the same pool, without duplicating controllers
or quota accounting. Duplicate pool IDs, aliases, self-hosted backend URLs, or
managed HF endpoint ownership are rejected. External models may share a provider
API URL with separately allocated per-pool budgets. Model/provider pairs with multiple revisions
require a matching entry in `defaults` or an explicit alias. An alias plus a
conflicting provider is rejected.

The catalog is an operator assertion about compatibility. Pin and verify actual
model/runtime revisions on the workers; the `revision` label does not change a
worker's deployed model. Managed worker URLs are discovered through the HF API.
Unmanaged workers require explicit API root URLs, without embedded credentials,
query strings, or fragments. Inference paths include `/v1`; do not append it to
the backend root URL.

Pool `policy` overrides the existing service defaults. `lifecycle` accepts the
existing `WorkerLifecycleSettings` fields, including warm floors, activation
budgets, cooldowns, idle windows, and latency thresholds. Remove the service-wide
`SPEECH_AUTOSCALE_ENABLED` setting when enabling a catalog: lifecycle ownership is
declared per pool. HF operating targets remain soft while additional capacity
warms. External routes cannot declare HF lifecycle settings.

## Select and authorize a route

Send the logical name in the existing JSON or STT multipart `model` field.
Optionally send `X-Speech-Provider`, or a gateway-only `provider` field; if both are
present, they must agree. The gateway replaces `model` with `upstream_model` and
removes `provider` before forwarding. An explicit provider choice is never
overridden by `defaults` or load balancing.

Existing CPU pipelines can use `MODEL_NAME=model-a-external` and the LLM proxy's
`LLM_BASE_URL` ending in `/v1`. `pipeline_entrypoint.py` already propagates the
model string through the selected adapter, so it requires no new worker-address
or provider-catalog logic. STT/TTS use the existing `STT_MODEL`/`TTS_MODEL` and
logical gateway URLs. The protocol selected by `LLM_BACKEND` must be declared by
the route; this gateway does not translate Responses into Chat Completions.

Keep gateways behind authenticated ingress. In a protected HF deployment,
ingress authorizes callers for the gateway and the catalog allowlists the routes
available to them. For additional route restrictions, set `access_key_env` to a
secret environment-variable name. Callers then supply
`X-Speech-Authorization: Bearer <application-key>`; it takes precedence over the
standard `Authorization` fallback, even when invalid. Reserve `Authorization`
for HF ingress credentials. The deployment helper requires `access_key_env` for
every route when creating a public catalog gateway.

Only deployment-side `credential_env` values become upstream bearer credentials;
caller credentials are never forwarded to inference providers. Current pinned
CPU adapters support the protected-ingress route path; forwarding an additional
application-auth header from those adapters requires the upstream companion work.

Capabilities are checked before reservation: API, tools/tool history, images,
LLM audio input, TTS voices/formats, output token budgets, cache controls, and
continuation mode. LLM routes must declare `context_window`. A caller with a
known context requirement can send `X-Speech-Required-Context-Tokens`; a route
with a smaller window is rejected. Exact input-plus-output token counting stays
with the inference runtime; the gateway does not tokenize or silently truncate
history. For TTS, declare `voices` and `audio_formats` and use protocol `speech`;
for STT use `transcriptions`. Fields not covered by these checks retain the
selected upstream API's validation semantics.

Self-hosted pools require complete-context requests and reject backend-local
`previous_response_id`, `conversation`, and input item references. External
routes may explicitly declare `continuation: "provider_state"` only when their
provider guarantees it across requests. Background generations are unsupported
because reservations account for active requests and streams. Provider-supported
`prompt_cache_key`/`prompt_cache_retention` controls can be allowlisted in
`capabilities.cache_controls` and are forwarded unchanged. No provider replica
affinity or cache-hit guarantee is implied.

## External API capacity and health

External mode adapts bearer-authenticated, OpenAI-compatible API endpoints. It
uses one configured provider API backend, disables GPU warmup, and defaults to
`/v1/models` for health. LLM health requires the configured `upstream_model` in
the returned `data[].id` list; `policy.health_path` can select an equivalent
provider endpoint. This is API availability evidence, not an inference benchmark.

Concurrency limits apply until response completion, failure, or cancellation.
Every inference attempt, including a retry, consumes one request in a rolling
60-second window. Local exhaustion returns 429 with `Retry-After`; provider 429s
are returned to the caller and put the entire pool into a cooldown using the
provider's `Retry-After` (seconds or HTTP date; one second if missing/invalid).
External 503 responses carrying `Retry-After` also set this cooldown and are
returned with the original status and retry header, without immediate replay.
Other retryable failures use the bounded attempt count before downstream bytes
begin. A partially streamed response is never replayed.

Allocate per-pool budgets within the provider account's shared limits. Token,
daily, and other account quotas remain enforced by the provider and feed back
through 429 handling; these limits are not estimated locally. Multiple routes
may share a CPU gateway, but independent pool budgets do not create additional
provider account quota. Verify the provider's actual model capabilities and
parameter restrictions; a compatible URL is insufficient. For example, see
[Cerebras compatibility](https://inference-docs.cerebras.ai/resources/openai) and
[rate-limit semantics](https://inference-docs.cerebras.ai/support/rate-limits).

## Deploy, roll out, and remove pools

Deploy one service at a time, using secret values supplied through the local
environment. Dry-run output contains only the catalog and required secret names:

```bash
uv run --with-requirements requirements.txt python scripts/create_speech_proxy_endpoints.py \
  --services llm --route-catalog routes.json \
  --llm-proxy-name your-llm-gateway \
  --image-url ghcr.io/your-org/s2s-speech-proxy:sha-YOUR_COMMIT --dry-run
```

Remove `--dry-run` to create the endpoint. This helper keeps one gateway replica
and does not provision GPU workers. Provision/pin the inventory separately using
the existing deployment helpers. A managed pool must have exactly one gateway
process/controller across the deployment; replicas and overlapping catalogs on
different gateways require coordination that is not implemented here.

1. **Add a route:** provision its compatible inventory, supply its credential
   references, and validate the catalog with a dry run. Deploy with the existing
   default preserved. Check the new pool's health and exercise its explicit alias.
2. **Change a revision:** provision a separate pool with a new pool ID, revision,
   and canary alias. Retain the old pool; do not relabel mixed worker revisions as
   compatible. Deploy catalog changes with graceful shutdown of the old gateway
   before the replacement takes lifecycle ownership.
3. **Gradual rollout:** point a chosen subset of CPU pipelines/callers at the new
   alias, keeping the default on the old pool. Compare per-pool errors, active
   work, and latency. Increase that subset, then move `defaults`. Roll back by
   choosing the old alias/default; no automatic cross-revision fallback occurs.
4. **Remove safely:** move callers away, account for old aliases, and allow active
   requests/streams to finish. Gracefully stop the owning gateway without
   overlapping controllers, remove the unused pool, and restart with the remaining
   catalog. Explicitly pause/retire the old HF inventory after verifying it has no
   active work; removing configuration does not delete or automatically park it.

Catalog changes require restart. Preserve both pools until caller migration is
complete. Graceful gateway restarts drain active inference streams; they do not
provide durable reservations or transfer caches between processes.

## Observability and follow-ups

`/health` and `/metrics` include `pools`, labeled with logical model, provider,
pool ID, and revision, plus backend accounting. Responses carry corresponding
`X-Speech-*` labels. Health succeeds when at least one pool is ready; slow or
unavailable optional routes do not block startup of healthy routes. Inspect the
specific pool for route readiness. Capacity and lifecycle decisions stay local
to each pool. Metrics contain no credential values, prompts, or conversation IDs.

Conversation affinity and gradual cache-aware draining remain
[#112](https://github.com/andimarafioti/s2s-endpoint/issues/112); their bindings
must live inside the selected pool. Cache/continuation verification remains
[#113](https://github.com/andimarafioti/s2s-endpoint/issues/113). Optional Realtime
`session.update` model switching and conversation-identifier propagation remain
[speech-to-speech #547](https://github.com/huggingface/speech-to-speech/issues/547).
This gateway's per-request route resolution supplies the pool boundary for those
changes; it does not implement mid-session switching or affinity itself.
## Session admission and five-pipeline reserve

Capacity composition is opt-in on the allocator with `PIPELINE_CAPACITY`. Its
named choices reference pool IDs on the three configured gateway URLs:

```json
{
  "default": "qwen-gemma-qwen",
  "routes": {
    "qwen-gemma-qwen": {"stt": "qwen-asr", "llm": "gemma", "tts": "qwen-tts"},
    "openai-gemma-qwen": {"stt": "openai-asr", "llm": "gemma", "tts": "qwen-tts"}
  },
  "reserve_sessions": 5,
  "llm_protocol": "chat_completions"
}
```

Set `SESSION_QUEUE_ENABLED=true`, `SPEECH_STT_PROXY_URL`, `SPEECH_LLM_PROXY_URL`,
`SPEECH_TTS_PROXY_URL`, and a shared `SPEECH_CAPACITY_API_KEY` on this allocator
and its gateways. For protected HF gateways, set
`SPEECH_CAPACITY_INGRESS_API_KEY` separately: it uses `Authorization`, while
the control key uses `X-Speech-Capacity-Authorization`. The latter takes
precedence over the standard-header fallback. Keep one admission owner and
one lifecycle controller per managed pool; do not run duplicate allocators
against the same pool's capacity control endpoint.

Each referenced pool needs a measured `session_workload`, for example:

```json
{
  "session_workload": {
    "profile": "Calibration reference, audio/context sizes, turn rate and concurrency assumptions",
    "work_per_session": 0.5,
    "requests_per_minute": 2
  },
  "policy": {"target_work": 8, "max_work": 12}
}
```

These numbers illustrate the schema, not a calibrated production setting.
`work_per_session` uses the pool's existing request-work units: STT equivalent
audio work, TTS/LLM requests. For external APIs it is concurrent requests;
the additional request-rate estimate must fit the provider's RPM budget.
Self-hosted admission pools require `max_work >= target_work` as a separate
hard per-worker ceiling. External pools retain their concurrency/RPM ceilings.

The allocator posts pending-plus-connected demand to `/internal/capacity` and
uses ready soft headroom to advertise available pipelines. It accounts for
both STT alternatives independently and shared stages once in aggregate.
Claims immediately spend shared headroom before another caller can allocate.
Compute health exports route counts for surviving connections after allocator
restart; unclassified connections count conservatively against all choices.
Returning activity remains in the workload forecast; observed in-flight work
can increase demand above it. Inference leases still last only for requests.

Admission requires one complete route within hard ceilings, even if the
five-pipeline reserve or normal target is depleted. Warming workers deduplicate
wakes but contribute no ready reserve. Inference workers scale from demand plus
reserve; CPU workers replenish free slots using their existing wake logic.
Cooldowns and sustained low use gate consolidation, and stale demand cannot
authorize an inference-worker drain. Stale gateway observations stop new
admissions; they do not disconnect healthy existing conversations. Health
reports `pipeline_capacity`, with per-choice stage headroom and the limiting
stage. Alternative capacities are never added together.

Clients may include `{"pipeline":"qwen-gemma-qwen"}` in `POST /session`; omission
uses the configured default. Unknown choices fail with 400. The selected
model/provider/protocol identities are signed into the grant and passed by the
compute wrapper as `X-Speech-Session-Routing` to the private upstream listener.
Client copies of that header are not forwarded. Enable `SESSION_ROUTING_ENABLED`
on those CPU workers and use the companion speech-to-speech revision supporting
the [initial routing handoff](https://github.com/huggingface/speech-to-speech/blob/feature/session-routing-handoff/docs/session-routing-handoff.md).
All advertised choices must match the worker's LLM adapter and audio settings.

Provider-specific request options must also be compatible across those choices.
`RESPONSES_API_DISABLE_THINKING=false` disables the automatic vLLM chat-template
extension. `TTS_STREAM=false` omits vLLM's `stream` flag while retaining the HTTP
stream reader and PCM `stream_format=audio`; `TTS_LANGUAGE=` omits its language
extension. Existing defaults stay unchanged. Do not assume an OpenAI endpoint
accepts vLLM-only fields. Tune and measure a compatible deployment before
advertising its workload-derived reserve.

Connected sessions are not expired because their ticket, token or backend cache
ages. Queue tickets have the separate absolute 300-second waiting deadline.
Cache retention/affinity and full mid-session model switching remain #112,
#113 and upstream #547 work; the initial admitted model is immutable here.

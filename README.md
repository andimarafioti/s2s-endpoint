---
library_name: none
tags:
- speech
- audio
- inference-endpoint
---

# s2s-endpoint

Speech-to-speech endpoint project.

## Deployment Split

This repo builds separate images for session compute and routing:

- compute image: `Dockerfile.compute`
  Starts `app.compute_main:app` on a GPU instance, runs local `speech-to-speech serve` subprocesses, and serves `/v1/realtime` directly.
- load-balancer image: `Dockerfile.load_balancer`
  Starts `app.load_balancer_main:app` on a CPU instance, tracks a configured set of pre-created compute endpoints, keeps a warm pool, wakes parked endpoints when free session capacity gets tight, and allocates direct compute sessions for clients.

For the split speech-service experiment it also builds two OpenAI-compatible
GPU service images:

- STT image: `Dockerfile.stt`
  Runs Qwen3-ASR through vLLM on port 8000. The image adds the audio extras that
  the official vLLM image intentionally omits.
- TTS image: `Dockerfile.tts`
  Runs Qwen3-TTS CustomVoice through vLLM-Omni on port 8091. Its bundled deploy
  config keeps async chunking and multi-request batching enabled.
- pipeline image: `Dockerfile.pipeline`
  Runs only VAD, Smart Turn, and `speech-to-speech serve` on CPU. It sends STT
  and TTS requests to dedicated GPU endpoints and can send LLM requests either
  to OpenAI or to the dedicated LLM proxy.
- service proxy image: `Dockerfile.speech_proxy`
  Runs as small CPU endpoints that provide stable STT, TTS, and LLM addresses.
  Each request is independently assigned to a ready inference worker.

This is intended for a deployment with:

- one load-balancer endpoint
- multiple compute endpoints
- one compute endpoint per unit of isolated capacity

The load balancer keeps session counts in memory, so it should run as a single
replica unless you add shared state outside this repo.

## Build Images

Build the compute image:

```bash
docker build --platform linux/amd64 -f Dockerfile.compute -t your-registry/s2s-endpoint-compute:latest .
```

`Dockerfile.compute` defaults
`S2S_REPO_URL=https://github.com/huggingface/speech-to-speech.git` and
`S2S_REF=main`. Override either value explicitly with:

```bash
docker build --platform linux/amd64 -f Dockerfile.compute \
  --build-arg S2S_REPO_URL=https://github.com/huggingface/speech-to-speech.git \
  --build-arg S2S_REF=main \
  -t your-registry/s2s-endpoint-compute:realtime .
```

The compute image uses CUDA 12.8 and Qwen3-TTS defaults to the GGML backend in
`speech-to-speech`. `Dockerfile.compute` downloads the CUDA 12.8
`qwentts-cpp-python` Hugging Face wheel into a local wheelhouse before running
`uv sync`, because the PyPI wheel may require a newer manylinux tag than the
base image exposes. For a different CUDA target and a compatible base image,
override the wheel URL and filename:

```bash
docker build --platform linux/amd64 -f Dockerfile.compute \
  --build-arg QWENTTS_WHEEL_URL=https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/resolve/main/whl/cu130/qwentts_cpp_python-0.3.1%2Bcu130-py3-none-manylinux_2_39_x86_64.whl \
  --build-arg QWENTTS_WHEEL_FILENAME=qwentts_cpp_python-0.3.1+cu130-py3-none-manylinux_2_39_x86_64.whl \
  -t your-registry/s2s-endpoint-compute:cuda13 .
```

To build against the temporary llama.cpp compatibility fix before it lands upstream, use:

```bash
docker build --platform linux/amd64 -f Dockerfile.compute \
  --build-arg S2S_REPO_URL=https://github.com/andimarafioti/speech-to-speech.git \
  --build-arg S2S_REF=fix/openai-responses-history-serialization \
  -t your-registry/s2s-endpoint-compute:llamacpp-fix .
```

Build the load-balancer image:

```bash
docker build --platform linux/amd64 -f Dockerfile.load_balancer -t your-registry/s2s-endpoint-lb:latest .
```

Build the custom vLLM image that embeds a Qwen non-thinking chat template:

```bash
docker build --platform linux/amd64 -f Dockerfile.vllm -t your-registry/s2s-endpoint-vllm:latest .
```

When deploying that image on a Hugging Face vLLM endpoint, use container arguments like:

```text
--max-model-len 32768 --reasoning-parser qwen3 --chat-template /app/qwen3_nonthinking.jinja
```

Build the dedicated speech-service images:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.stt \
  -t your-registry/s2s-stt:sha-YOUR_FULL_COMMIT_SHA --push .
docker buildx build --platform linux/amd64 -f Dockerfile.tts \
  -t your-registry/s2s-tts:sha-YOUR_FULL_COMMIT_SHA --push .
```

On every relevant `main` change, the `Publish speech service images` workflow
publishes Linux AMD64 images as
`ghcr.io/andimarafioti/s2s-{stt,tts}:sha-<full-commit-sha>`. A manual workflow
run can additionally promote an explicit version alias such as `v0.3`; normal
builds never overwrite a version alias. These images use the CUDA 12 vLLM
runtime required by the current A10G endpoint hosts.

Create one protected, warm A10G endpoint for each service in the production
region. The script resolves and pins the current model revisions before it
creates anything, and refuses to reuse an existing endpoint name:

```bash
uv run --with-requirements requirements.txt python scripts/create_speech_service_endpoints.py \
  --namespace HuggingFaceM4 \
  --stt-image-url ghcr.io/andimarafioti/s2s-stt:sha-YOUR_FULL_COMMIT_SHA \
  --tts-image-url ghcr.io/andimarafioti/s2s-tts:sha-YOUR_FULL_COMMIT_SHA \
  --dry-run

uv run --with-requirements requirements.txt python scripts/create_speech_service_endpoints.py \
  --namespace HuggingFaceM4 \
  --stt-image-url ghcr.io/andimarafioti/s2s-stt:sha-YOUR_FULL_COMMIT_SHA \
  --tts-image-url ghcr.io/andimarafioti/s2s-tts:sha-YOUR_FULL_COMMIT_SHA \
  --wait
```

The resulting OpenAI-compatible base URLs are the endpoint URL plus `/v1`.
Because the endpoints are protected, pass a Hugging Face token as the
`--openai_stt_api_key` or `--openai_tts_api_key` value when configuring the
speech-to-speech pipeline. This deployment does not change
`stream_batch_sentences`; the experiment should preserve the pipeline's current
sentence batching while comparing service placement.

Build and deploy the request-level speech proxies after the GPU workers exist:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.speech_proxy \
  -t your-registry/s2s-speech-proxy:sha-YOUR_FULL_COMMIT_SHA --push .

export HF_TOKEN=...
uv run --with-requirements requirements.txt python scripts/create_speech_proxy_endpoints.py \
  --namespace HuggingFaceM4 \
  --image-url your-registry/s2s-speech-proxy:sha-YOUR_FULL_COMMIT_SHA \
  --stt-backends reachy-s2s-stt-01 \
  --tts-backends reachy-s2s-tts-01 \
  --dry-run
uv run --with-requirements requirements.txt python scripts/create_speech_proxy_endpoints.py \
  --namespace HuggingFaceM4 \
  --image-url your-registry/s2s-speech-proxy:sha-YOUR_FULL_COMMIT_SHA \
  --stt-backends reachy-s2s-stt-01 \
  --tts-backends reachy-s2s-tts-01 \
  --wait
```

The same proxy image is configured as STT, TTS, or LLM by environment. STT
accounts for work in five-second audio equivalents; TTS and LLM account for
concurrent calls. The initial operating targets are 96 STT work units, 8 TTS
calls, and 64 LLM generations per worker. The LLM latency target is 500 ms to
first token, based on the Gemma 4 26B-A4B NVFP4 RTX PRO 6000 curve. These are
soft routing targets and do not reject excess work. When every healthy worker
is above target, new calls still go to the best available worker. Routing
combines current work with an EWMA latency penalty. TTS and LLM readiness each
include a real short inference. Retries move to another worker only before the
first audio/token reaches the caller, and cancellation closes the upstream
response and releases its reservation.

Create the LLM proxy explicitly after the Gemma workers exist. The tested RTX
PRO 6000 endpoint is in AWS `us-east-2`, but Hugging Face currently offers no
CPU endpoint shapes in that region. The inexpensive managed placement is
therefore AWS `us-east-1`; measure its cross-region overhead rather than
assuming the shared `us-east` prefix means colocation. Exact colocation would
require running this CPU proxy outside Hugging Face Endpoints or consuming a
second RTX PRO 6000 just for the proxy.

```bash
uv run --with-requirements requirements.txt python scripts/create_speech_proxy_endpoints.py \
  --namespace HuggingFaceM4 \
  --services llm \
  --image-url your-registry/s2s-speech-proxy:sha-YOUR_FULL_COMMIT_SHA \
  --llm-backends reachy-s2s-llm-01 \
  --llm-model nvidia/Gemma-4-26B-A4B-NVFP4 \
  --region us-east-1 \
  --wait
```

The resulting base URL supports both `/v1/chat/completions` and
`/v1/responses`. Request JSON and streamed SSE bytes pass through unchanged, so
tool calls and multimodal request bodies use the same vLLM behavior as direct
requests.

To point a deployed CPU pipeline at this proxy, add the following arguments to
`create_pipeline_endpoint.py`. Chat Completions is the recommended Gemma path
because that is the API used for the tool-call and vision validation.

```bash
  --llm-base-url https://YOUR-LLM-PROXY.us-east-2.aws.endpoints.huggingface.cloud/v1 \
  --llm-backend chat-completions \
  --model-name nvidia/Gemma-4-26B-A4B-NVFP4
```

In this mode the deployment reuses `HF_TOKEN` for the protected LLM proxy and
does not require an OpenAI API key.

Each service keeps one stable CPU proxy replica because reservations and
latency history are process-local. Its bounded worker inventory may contain
running or paused endpoints. With autoscaling enabled, the proxy discovers URLs
through the endpoint control API, keeps the configured warm floor, records
short load peaks between control polls, and starts one additional worker when
per-worker work or fresh fleet-wide latency crosses its target. The targets are
soft: requests continue using available workers while capacity warms.

Scale-down requires sustained surplus capacity, atomically marks one worker as
draining, sends it no new requests, waits for every active request/stream to
release, and only then pauses it. A burst can cancel a drain before the pause
begins. Pending starts count as capacity, control failures block scale-down,
and ambiguous pause failures remain quarantined. Unhealthy workers are restarted
with bounded attempts and backoff. Parked workers are never HTTP health-probed,
because an inference request can wake a scaled-to-zero service.

Create the complete endpoint inventory first, leaving the standbys paused:

```bash
export HF_TOKEN=...
uv run --default-index https://pypi.org/simple --with-requirements requirements.txt \
  python scripts/create_worker_standbys.py \
  --source reachy-s2s-tts-01 --names reachy-s2s-tts-02 \
  --apply --wait-ready
```

Then configure the service proxy with every worker name and a dedicated HF
control credential. Initial scale-up utilization is 85%: 81.6 five-second STT
equivalents, 6.8 TTS generations, or 54.4 LLM generations per ready worker.
Latency only votes after every ready worker is above its service target for 30
seconds; an isolated slow worker cannot wake capacity.

```bash
export HF_TOKEN=...
export HF_CONTROL_TOKEN=...
uv run --default-index https://pypi.org/simple --with-requirements requirements.txt \
  python scripts/create_speech_proxy_endpoints.py \
  --services stt tts llm --autoscale --min-warm-workers 1 --max-workers 2 \
  --stt-backends reachy-s2s-stt-01 reachy-s2s-stt-02 \
  --tts-backends reachy-s2s-tts-01 reachy-s2s-tts-02 \
  --llm-backends gemma4-26b-a4b-nvfp4-rtx6000-test reachy-s2s-llm-02 \
  --image-url ghcr.io/andimarafioti/s2s-speech-proxy:sha-YOUR_FULL_COMMIT_SHA
```

Worker phases, actions, active work, target work, and lifecycle errors appear in
both proxy health/metrics and the load-balancer dashboard. Set
`SPEECH_WORKER_MIN_WARM=2` when immediate single-worker failover is worth the
second always-on GPU; a warm floor of one optimizes cost but accepts model-load
time after failure or scale-up.

The `Publish speech service images` workflow can selectively publish immutable
`ghcr.io/andimarafioti/s2s-speech-proxy:sha-<full-commit-sha>` images. A manual
run can also promote a version alias.

All STT, TTS, and LLM requests use the same telemetry implementation. Proxy responses
include `X-Speech-Request-Id`, `Server-Timing`, and component latency headers,
and each proxy exposes `/metrics?window_s=...` with p50, p90, p95, and p99
latencies. The metrics separate proxy application work from the backend round
trip. The GPU speech-service images add the same timing middleware to both vLLM
servers, allowing their proxies to split that backend round trip into model
service time and endpoint transport/gateway time. For TTS, proxy latency ends
at the first non-empty audio chunk; for STT, it ends at the transcription body;
for streaming LLM calls, it ends at the first response chunk.
These metrics are process-local, keep the most recent 50,000 requests, and reset
when the proxy restarts.

Configure the load-balancer dashboard with the proxy root URLs. Protected
proxy requests reuse `HF_CONTROL_TOKEN`/`HF_TOKEN` unless
`SPEECH_PROXY_API_KEY` is set explicitly:

```bash
uv run --with-requirements requirements.txt python scripts/update_load_balancer_endpoint_env.py \
  --namespace HuggingFaceM4 \
  --name reachy-s2s-lb \
  --env SPEECH_STT_PROXY_URL=https://STT-PROXY-HOST \
  --env SPEECH_TTS_PROXY_URL=https://TTS-PROXY-HOST \
  --env SPEECH_LLM_PROXY_URL=https://LLM-PROXY-HOST
```

The dashboard displays STT, TTS, and LLM together for the selected dashboard window.
If the GPU images have not yet been redeployed with the timing middleware, the
proxy and backend-round-trip metrics still work, while GPU-service timing is
shown as unavailable with zero reporting coverage.

Build and deploy the CPU-only pipeline after both speech services are running:

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.pipeline \
  -t your-registry/s2s-pipeline:sha-YOUR_FULL_COMMIT_SHA --push .

export HF_TOKEN=...
export OPENAI_API_KEY=...
uv run --with-requirements requirements.txt python scripts/create_pipeline_endpoint.py \
  --namespace HuggingFaceM4 \
  --image-url your-registry/s2s-pipeline:sha-YOUR_FULL_COMMIT_SHA \
  --stt-base-url https://YOUR-STT-PROXY.us-east-1.aws.endpoints.huggingface.cloud/v1 \
  --tts-base-url https://YOUR-TTS-PROXY.us-east-1.aws.endpoints.huggingface.cloud/v1 \
  --dry-run
uv run --with-requirements requirements.txt python scripts/create_pipeline_endpoint.py \
  --namespace HuggingFaceM4 \
  --image-url your-registry/s2s-pipeline:sha-YOUR_FULL_COMMIT_SHA \
  --stt-base-url https://YOUR-STT-PROXY.us-east-1.aws.endpoints.huggingface.cloud/v1 \
  --tts-base-url https://YOUR-TTS-PROXY.us-east-1.aws.endpoints.huggingface.cloud/v1 \
  --wait
```

The `Publish pipeline image` workflow uses the same immutable commit-tag and
manual version-alias convention as the speech-service workflow. The deployment
helpers require explicit image references so the endpoint configuration always
records the exact build selected by the operator.

The default endpoint is a protected, always-warm AWS `intel-spr-x4` CPU
instance in `us-east-1`. It preserves `stream_batch_sentences=3`, disables live
transcription for the baseline, and keeps conversation transcripts out of
retained endpoint logs. Secrets are written only to an owner-readable ephemeral
configuration file inside the container; they are not included in the process
arguments.

Connect the packaged microphone/speaker client directly to the resulting URL:

```bash
speech-to-speech talk \
  --url wss://YOUR-ENDPOINT.us-east-1.aws.endpoints.huggingface.cloud/v1/realtime \
  --api-key "$HF_TOKEN" \
  --playback-buffer-ms 196
```

Current upstream uses the command name `talk`; older checkouts may call the
same client `listen`.

### Managed CPU pipeline workers

For pipeline autoscaling, deploy with `create_pipeline_endpoint.py --managed`
and supply `SESSION_SHARED_SECRET` in the environment, matching the LB's session
signing secret. This selects `PIPELINE_MANAGED=true` and `/health` readiness.
The same CPU image then runs the existing compute wrapper on port 7860 and the
remote-only S2S process on loopback port 9000. Health reports real session slots;
session-token validation, connected/disconnected callbacks, and drain recovery
use the existing LB protocol. The original direct-testing mode stays unchanged
when `PIPELINE_MANAGED` is false.

Add the managed pipeline names to the LB's `COMPUTE_ENDPOINT_NAMES` and use its
existing `COMPUTE_ENDPOINT_MIN_WARM`, `COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS`,
idle parking, and recovery settings. Do not add raw/direct S2S endpoints to this
inventory: their health and session-allocation contracts are different. The LB
now supplies its HF ingress credential when reading protected worker health.
End-user clients must also have HF ingress access to protected pipeline endpoints;
public pipeline endpoints instead rely on signed session tokens.

### Worker scaling controls and rollout boundary

Autoscaling is opt-in (`SPEECH_AUTOSCALE_ENABLED=true`) and requires separate
`HF_CONTROL_TOKEN` and `HF_ENDPOINT_NAMESPACE` configuration. `SPEECH_BACKENDS`
accepts `name=url` entries or, in managed mode, endpoint names alone. All listed
endpoints belong exclusively to that proxy; do not run a second controller over
the same inventory, enable HF replica autoscaling, or send inference directly
to those GPUs. Bypassing the proxy makes active-work/drain accounting incomplete.

Every `WorkerLifecycleSettings` field is configurable as
`SPEECH_WORKER_<UPPERCASE_FIELD>`. Important defaults:

| Setting suffix | Default | Meaning |
| --- | ---: | --- |
| `MIN_WARM` / `MAX_WORKERS` | 1 / inventory size | Warm floor and bounded activation budget |
| `SCALE_UP_UTILIZATION` / `SCALE_DOWN_UTILIZATION` | 0.85 / 0.5 | Growth and consolidation thresholds |
| `RECONCILE_INTERVAL_S` | 5 | Control-plane poll interval |
| `LATENCY_BREACH_S` / `LATENCY_MAX_AGE_S` | 30 / 60 | Sustained breach and freshness windows |
| `SCALE_UP_COOLDOWN_S` / `SCALE_DOWN_COOLDOWN_S` | 30 / 180 | Change-rate limits |
| `IDLE_TIMEOUT_S` / `MIN_UPTIME_S` | 600 / 300 | Sustained surplus window and minimum worker uptime |
| `STARTUP_TIMEOUT_S` / `UNHEALTHY_RESTART_S` | 900 / 120 | Pending-capacity expiry and unhealthy recovery delay |
| `MAX_RESTART_ATTEMPTS` / `RETRY_BACKOFF_S` | 3 / 30 | Bounded restarts with capped exponential backoff |

STT latency is normalized by uploaded duration, not compared as raw seconds.
The fleet can consolidate under light continuous traffic, not only complete
silence. A timed-out remote operation stays out of routing; initialization that
exceeds its timeout stops counting as pending capacity but is not blindly
restarted while HF still reports an ongoing transition.

Provisioning never creates capacity in response to unbounded demand: the helper
pre-creates a fixed inventory, and runtime control only starts/stops those names.
It preserves source runtime arguments, model revision, hardware, and region;
all source secret names must be supplied from environment variables (use
`--secret-from-env ENDPOINT_SECRET=ENV_VAR_NAME` for aliases). Its default is a
read-only plan; `--apply` creates and pauses, and `--wait-ready` boots once first.
Prefer immutable image digests for managed engines with mutable upstream tags.

Roll out a new proxy image with autoscaling disabled first. Verify its inventory,
then enable one service at a time. To disable lifecycle control, set
`SPEECH_AUTOSCALE_ENABLED=false` and use an explicit list of already-warm backend
URLs. This does not pause or delete workers. Metrics and controller state remain
process-local, and durable telemetry/multi-proxy coordination are separate work.

## Direct Session Flow

The LB is no longer in the media path for websocket traffic.

The flow is:

1. Client calls `POST /session` on the LB.
2. The LB reserves a compute endpoint slot and returns:
   - a direct compute websocket URL
   - a signed session token
   - a convenience `connect_url` with the session token embedded as a query parameter
3. Client connects directly to the compute endpoint websocket route returned by the LB, `/v1/realtime`.
4. Compute validates the session token and notifies the LB when the session starts and ends.

This removes the LB from the websocket data path. The LB only handles control-plane allocation and release.

### When every slot is busy: the waiting queue

The queue is **off by default**. Set `SESSION_QUEUE_ENABLED=true` on a load-balancer
instance to turn it on; leave it unset and `POST /session` keeps the pre-queue
contract — the request blocks until a slot frees (up to
`COMPUTE_ENDPOINT_WAIT_TIMEOUT_S`, then `503`), `/queue/*` returns `404`, and the
only response-shape difference from the old behavior is the added
`"state": "granted"` field. Enable it only once the instance's clients understand
`"state": "queued"` responses and poll `GET /queue/{id}` — a pre-queue client on a
queueing instance would get a `200` without a `connect_url` and fail.

With the queue enabled, `POST /session` never blocks. If a slot is free (and no one is already waiting) it
returns a grant as above, marked `"state": "granted"`. Otherwise the caller joins a
FIFO waiting queue and gets a **ticket** instead:

```json
{ "state": "queued", "queue_id": "…", "position": 3, "poll_interval_s": 2, "ticket_ttl_s": 8 }
```

The client then polls `GET /queue/{queue_id}` every `poll_interval_s`:

- still waiting → `{ "state": "queued", "position": N, … }` (position only ever decreases)
- reached the front and a slot freed → the same `"state": "granted"` body as `POST /session`
- unknown/expired ticket → `404`

Only the head of the line claims a freed slot, so admission stays FIFO. A ticket that
isn't polled within `ticket_ttl_s` is dropped (how an abandoned waiter is detected), and
everyone behind shifts up. `DELETE /queue/{queue_id}` leaves the line explicitly (used by
the client's teardown beacon). Waiting reserves no compute and, on the demo Space, no
usage time — only a live connected session counts. If the queue itself is full the
`POST /session` returns `503` with `{ "state": "at_capacity" }`.

Tunable via env: `SESSION_QUEUE_ENABLED` (default false), `QUEUE_MAX_DEPTH` (default 100), `QUEUE_TICKET_TTL_S` (8),
`QUEUE_POLL_INTERVAL_S` (2), `QUEUE_REAP_INTERVAL_S` (2). Setting `QUEUE_MAX_DEPTH=0`
disables the waiting room: any caller who can't be granted a slot immediately gets
`at_capacity` instead of a ticket. The queue is never unbounded.

Sizing `QUEUE_MAX_DEPTH`: it's a ceiling, not a target. Two things bound how high
it's worth setting. Poll load scales as `depth / QUEUE_POLL_INTERVAL_S` requests
per second (on this app, and again on the Space that proxies it), so a very deep
queue at a 2s cadence puts real request pressure on both small containers. And
position is only meaningful if the wait is bearable: the last person's wait is
roughly `depth × avg_session ÷ live_slots`, so past ~10-15 minutes people abandon
regardless. 100 is comfortable headroom on load and keeps the "at capacity" modal
rare; if you push it past ~200, raise `QUEUE_POLL_INTERVAL_S` to 3-4s to hold the
request rate down (the per-poll position lookup is currently O(queue depth)).

In load-balancer mode, the app does not guess endpoint hostnames. It asks the
Hugging Face API for each compute endpoint's canonical HTTPS URL and turns that
into the direct websocket URL by replacing `https://` with `wss://` and appending
that websocket route.

## Swarm Dashboard

The load balancer now exposes a built-in dashboard:

- `GET /dashboard`: HTML dashboard for the current swarm
- `GET /dashboard/data`: JSON feed used by the dashboard UI

The dashboard keeps an in-memory rolling history on the LB itself and shows:

- running, warming, transitioning, and parked endpoint counts
- connected and pending user sessions
- free slots and effective free capacity
- `POST /session` request counts, authentication rejections, allocation
  successes/failures, and connect/disconnect events
- conversation starts/completions plus average and max completed conversation duration
- distinct verified Hugging Face users, token fingerprints, anonymous network
  fingerprints, and client-reported robot fingerprints
- a per-requester leaderboard with allocation and connection outcomes, traffic
  share, burst rate, network count, reported robot count, client type, and
  unusual-usage signals

Clients can optionally send a Hugging Face user access token as
`X-Reachy-Mini-Authorization: Bearer <token>` on `POST /session`. Standard
`Authorization` is also accepted when the hosting gateway passes it through. The
Reachy-specific header takes precedence when both are present. By default, the
request is allowed to continue when the token is missing or cannot be verified,
preserving compatibility with anonymous deployments. Valid tokens are resolved
to the Hugging Face account asynchronously through `whoami`.

Set `SESSION_REQUIRE_VERIFIED_HF_TOKEN=true` to require authentication for session
admission. In this mode, the load balancer waits up to
`SESSION_HF_TOKEN_VERIFY_TIMEOUT_S` for a first-seen token's `whoami` result before
it allocates or queues. A cached account identity is accepted for authentication
only while its `verified_at` proof is no older than
`SESSION_HF_TOKEN_MAX_VERIFIED_AGE_S`; stale proofs are revalidated before a new
admission, and queued grants fail closed if their proof ages out. Missing,
malformed, unrecognized, and invalid credentials
return `401` with `WWW-Authenticate: Bearer`. A timed-out or unavailable Hugging
Face validation returns retryable `503` without allocating, queueing, or waking
compute. When an upstream verification failure is cached, `Retry-After` reports
the cache entry's remaining lifetime (up to 60 seconds) so clients do not retry
before a new `whoami` attempt can occur. The verified privacy-safe requester
identity is retained with a queue ticket, and queued grants are denied if that
authorization context is lost. Raw bearer tokens are never logged or retained.

New remote token validations are admitted through a separate pre-authentication
guard before they enter the resolver queue. It limits both total pending checks
and pending checks from one privacy-safe network fingerprint, so fabricated token
values cannot consume all resolver validation slots from a single source. This
guard is independent of the requester rate limiter, which still runs only after
authentication succeeds.

Reachy clients can also include the daemon's optional 16-character hexadecimal
`hardware_id` in the JSON body. The content type must be `application/json`:

```json
{"hardware_id": "0123456789abcdef"}
```

Missing, malformed, or unsupported request bodies do not deny allocation. A
valid value is normalized to lowercase and immediately converted to a keyed,
one-way `robot:` fingerprint. The raw hardware ID is not retained. The dashboard
counts distinct reported robots for the selected window, reported-robot requests,
and distinct reported robots per requester. This value is supplied by the client;
it is an attribution hint, not proof that a request came from genuine hardware.

The per-requester `Connected` count is stricter than `Allocated`. Allocation
means the load balancer returned session credentials to the client; connection
is recorded only after the compute endpoint sends the first successful websocket
`connected` callback for that session. The dashboard therefore shows which HF
users, token fingerprints, or anonymous network fingerprints actually joined an
allocated session. Allocations and connections are independent event counts in
the selected window, not a cohort conversion rate; a session can be allocated
before one window and connect during the next.

The load balancer never stores raw client tokens, raw IP addresses, or raw robot
hardware IDs. It stores keyed, one-way fingerprints instead. Tokenless traffic
is grouped by the first `X-Forwarded-For` address (or the direct client address
when proxy headers are disabled/unavailable), and user-agent classes such as the
Reachy Mini conversation app, Reachy Mini mobile app, browser, generic mobile
app, `httpx`, `curl`, or bot are counted to help distinguish likely automation.
The dashboard flags high request volume, large per-minute bursts, dominant
traffic share, many networks using one token, mostly automation-like clients,
and invalid tokens. These are operational signals only; this feature does not
rate-limit or block traffic.

The timeline automatically switches between minute-level and hourly rollups depending on the selected window. By default the history is in memory and resets when the LB endpoint restarts.

If you want the dashboard history to survive LB restarts, you can configure it to persist completed minute buckets to a Hugging Face Storage Bucket. The live routing/session state still stays in memory; the bucket is only for historical dashboard data.

Persisted history is restored in the background during load-balancer startup, so
the endpoint can become ready before older dashboard buckets finish loading. The
`/dashboard/data` response includes a `history_restore` object with the restore
status, elapsed time, and restored bucket count. After the initial restore, the
load balancer performs two delayed reconciliation passes by default so it also
sees files that the previous replica writes during a slow shutdown. Each pass is
limited to the previous UTC day through the current minute, and bucket
comparisons release the dashboard lock between bounded chunks. The
`startup_merge` persistence status reports the scheduled, attempted, completed,
and failed pass counts and whether the full schedule completed.

The dashboard store keeps minute files under `minutes/YYYY-MM-DD/` and also
uses `days/YYYY-MM-DD.json` files as a compact cache for UTC days. On restore it
checks `days/` first, falls back to minute files for days without an
authoritative cache, and backfills a complete `days/` file once it has all 1,440
minute buckets for a completed day. While the load balancer stays running, it
also rolls over each completed UTC day from in-memory history into
`days/YYYY-MM-DD.json` shortly after midnight UTC. If the day is missing minute
buckets, the rollover still writes a finalized partial day file with
`complete: false`, `finalized: true`, and a missing-minute count. Restores treat
only complete day files as authoritative: partial day files are merged with any
minute files that appeared later, and the partial cache is refreshed only when
the lookup finds a new bucket. This lets a new load balancer recover minute
files written late by the previous replica during a rolling replacement without
rewriting an unchanged partial cache on every restore.

You can precompute day files without running the load balancer:

```bash
uv run --with-requirements requirements.txt python scripts/backfill_dashboard_day_history.py \
  --bucket-id HuggingFaceM4/reachy-s2s-dashboard \
  --prefix reachy-s2s-lb \
  --days 30
```

Use `--dry-run` to inspect which days would be created without writing files.
By default the script also migrates legacy flat `minutes/<epoch>.json` files
for the requested days into `minutes/YYYY-MM-DD/<epoch>.json`; pass
`--migrate-minutes-only` when day files already exist and you only want to move
minute files, or `--skip-minute-migration` to leave legacy minute paths
untouched. Minute migration uses server-side bucket copies when supported by
`huggingface_hub`, then deletes the old flat paths. The script processes and
uploads one day at a time, so interrupted runs are resumable: the next run skips
any day files that were already created. It also keeps a local minute download
cache under the user cache directory so interrupted day-file backfills can reuse
already downloaded minute files; pass `--no-download-cache` to disable this.
Historical partial days are cached too, which is useful for the first UTC day a
load balancer existed. These partials are finalized after checking available
minute files, so subsequent runs do not keep downloading the same incomplete
day. Pass `--require-complete-days` to only create day files when all 1,440
minute buckets are present.

## Load Balancer Env Vars

Each compute endpoint owns its capacity through `NUM_PIPELINES`. Its `/health`
response publishes that value as `router.max_sessions`, and the load balancer
learns it independently for every running endpoint alongside the active session
count. Updating and restarting a compute endpoint is therefore enough to change
its capacity; the load balancer picks up the new value on its next reconciliation
without a configuration change or restart. The former `COMPUTE_ENDPOINT_SLOTS`
load-balancer variable is ignored and can be removed from existing deployments.

- `HF_ENDPOINT_NAMESPACE`: namespace that owns the compute endpoints
- `COMPUTE_ENDPOINT_NAMES`: comma-separated endpoint names
- `COMPUTE_ENDPOINT_MIN_WARM`: number of compute endpoints that should stay warm
- `COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS`: when total free slots drop to this level,
  the LB starts waking another parked endpoint
- `COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S`: how long an idle compute endpoint stays warm
  before being parked
- `COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S`: background refresh interval
- `COMPUTE_ENDPOINT_RECONCILE_STALE_AFTER_S`: maximum age of the last completed
  reconciliation before `/health` reports the router unhealthy
- `COMPUTE_ENDPOINT_HTTP_TIMEOUT_S`: timeout applied to Hugging Face
  endpoint-control HTTP requests (defaults to 10 seconds)
- `COMPUTE_ENDPOINT_CONTROL_FETCH_TIMEOUT_S`: total deadline for one endpoint
  status refresh (defaults to 30 seconds)
- `COMPUTE_ENDPOINT_CONTROL_OPERATION_TIMEOUT_S`: positive deadline for waiting
  for a wake to become ready (defaults to `COMPUTE_ENDPOINT_WAIT_TIMEOUT_S`)
- `COMPUTE_ENDPOINT_PARK_STRATEGY`: `pause` or `scale_to_zero`
- `HF_CONTROL_TOKEN`: token used to call the Inference Endpoints API
- `LB_ADMIN_AUTH_TOKEN`: dedicated bearer token required by the internal endpoint
  status and drain routes. Send it in
  `X-Reachy-Mini-Admin-Authorization` because HF Inference Endpoints consumes
  the standard `Authorization` header; the LB retains the standard header as a
  fallback for non-HF deployments. Do not reuse `HF_CONTROL_TOKEN`.
- `LB_CALLBACK_AUTH_TOKEN`: dedicated bearer credential required from compute
  endpoints at `/internal/llm-proxy-usage`. Use the same value on the LB and
  every compute endpoint; do not reuse a requester's HF token. Compute sends
  it in `X-Reachy-Mini-Callback-Authorization` because HF Inference Endpoints
  consumes the standard `Authorization` header; the LB retains the standard
  header as a fallback for non-HF deployments.
- `COMPUTE_ENDPOINT_DRAIN_LEASE_TTL_S`: default allocator-drain lease lifetime
  for admin clients that do not request one explicitly (defaults to 3,600 seconds)
- `COMPUTE_ENDPOINT_DRAIN_WARNING_AFTER_S`: age at which the LB starts warning
  about a continuously drained endpoint (defaults to 600 seconds)
- `COMPUTE_ENDPOINT_DRAIN_WARNING_INTERVAL_S`: interval between repeated
  long-running drain warnings (defaults to 300 seconds)
- `SESSION_SHARED_SECRET`: shared secret used to mint and validate direct session tokens
- `SESSION_REQUIRE_VERIFIED_HF_TOKEN`: require a Hugging Face-verified bearer token
  before allocating or queueing a session (defaults to `false`)
- `SESSION_HF_TOKEN_VERIFY_TIMEOUT_S`: maximum time to wait for first-seen token
  verification when session authentication is required (defaults to 5 seconds)
- `SESSION_HF_TOKEN_MAX_VERIFIED_AGE_S`: maximum age of a successful `whoami`
  result when it is used as a session authentication proof (defaults to 1,800
  seconds / 30 minutes). Older cached identities are revalidated before
  admission, so a revoked token can remain accepted until this proof expires.
- `SESSION_HF_TOKEN_VERIFY_MAX_PENDING`: maximum remote token validations admitted
  concurrently for enforced session authentication (defaults to 64)
- `SESSION_HF_TOKEN_VERIFY_MAX_PENDING_PER_NETWORK`: maximum pending enforced
  session validations attributed to one privacy-safe network fingerprint
  (defaults to 4)
- `LLM_PROXY_CLAIM_VERIFY_TIMEOUT_S`: how long session creation waits for a
  first-seen HF token's `whoami` validation before minting the session's LLM
  proxy claim (defaults to 5 seconds). On timeout the session is created
  normally but without LLM proxy access.
- `SESSION_PENDING_TIMEOUT_S`: how long an unused reservation stays alive
- `SESSION_TOKEN_TTL_S`: lifetime of the signed session token
- `SESSION_REAP_INTERVAL_S`: how often the LB reaps unused reservations
- `REQUEST_USAGE_HASH_SECRET`: secret key for stable token, IP, and reported robot
  fingerprints. Defaults to `SESSION_SHARED_SECRET`; set it explicitly if that
  secret is not stable across LB replacements. If neither is set, fingerprints
  change on every process restart.
- `REQUEST_USAGE_TRUST_PROXY_HEADERS`: whether requester attribution trusts the
  first `X-Forwarded-For`/`X-Real-IP` address (defaults to `true`). Disable this
  outside a trusted reverse-proxy deployment.
- `REQUEST_USAGE_MAX_ACTORS_PER_MINUTE`: maximum distinct requester records kept
  in one minute bucket before additional actors roll into an overflow row
  (defaults to 1,000)
- `REQUEST_USAGE_MAX_RETAINED_RECORDS`: maximum detailed requester records kept
  across dashboard retention. Oldest requester details are compacted while the
  minute-level request totals remain available (defaults to 50,000)
- `REQUEST_USAGE_MAX_PENDING_VALIDATIONS`: maximum queued unique HF token
  identity lookups (defaults to 128)
- `REQUEST_USAGE_VALIDATION_CONCURRENCY`: maximum concurrent HF `whoami` lookups
  (defaults to 4)
- `REQUEST_USAGE_HIGH_REQUESTS`: request count that raises a high-volume signal
  in the selected dashboard window (defaults to 100)
- `REQUEST_USAGE_BURST_PER_MINUTE`: per-requester one-minute peak that raises a
  burst signal (defaults to 20)
- `REQUEST_USAGE_MANY_NETWORKS`: distinct network fingerprints for one requester
  that raise a many-networks signal (defaults to 5)
- `REQUEST_RATE_LIMIT_ENABLED`: enforce requester rate limits (defaults to `true`).
  When disabled, the limiter continues tracking outcomes without rejecting requests.
  Limiter state is local to the load-balancer process and resets when it restarts.
- `REQUEST_RATE_LIMIT_WINDOW_S`: rolling request-rate window (defaults to 60 seconds)
- `REQUEST_RATE_LIMIT_REQUESTS_PER_WINDOW`: maximum `POST /session` attempts from
  one requester in the rolling window (defaults to 20)
- `REQUEST_RATE_LIMIT_MAX_PARALLEL`: maximum simultaneous allocation calls, pending
  joins, and connected sessions from one requester (defaults to 10)
- `REQUEST_RATE_LIMIT_NO_CONNECTS`: consecutive allocated sessions that can expire
  or disconnect without joining before a cooldown starts (defaults to 5)
- `REQUEST_RATE_LIMIT_SHORT_SESSION_S`: connected duration at or below which a
  session counts toward reconnect-loop detection (defaults to 10 seconds)
- `REQUEST_RATE_LIMIT_SHORT_SESSIONS`: consecutive short connected sessions before
  a cooldown starts (defaults to 8). A longer session resets this streak.
- `REQUEST_RATE_LIMIT_COOLDOWN_S`: behavioral cooldown duration after repeated
  no-connect allocations or short sessions (defaults to 900 seconds)
- `REQUEST_RATE_LIMIT_ACTOR_RETENTION_S`: idle requester limiter state retention
  (defaults to 3,600 seconds)
- `REQUEST_RATE_LIMIT_MAX_ACTORS`: maximum in-memory requester limiter states
  (defaults to 10,000). Retention-expired inactive states are evicted first; if
  the table remains full of fresh state, previously unseen requesters fail closed.
- `DASHBOARD_SAMPLE_INTERVAL_S`: how often the LB samples swarm state for history
- `DASHBOARD_RETENTION_MINUTES`: in-memory history retention for dashboard data
  (defaults to 28 days so the 14d/28d dashboard windows can load persisted history)
- `DASHBOARD_FLUSH_BATCH_SIZE`: maximum minute files written in one storage call
  (defaults to 100)
- `DASHBOARD_FLUSH_TIMEOUT_S`: age at which an in-flight dashboard storage write
  is reported as stalled (defaults to 60 seconds). The single writer remains in
  flight until it resolves, preventing overlapping writes and stale overwrites.
  This also configures the Hugging Face Hub HTTP client request timeout. During
  shutdown, final dashboard persistence gets a total budget of twice this value;
  the load balancer logs any remaining dirty buckets and continues shutdown if
  the budget expires. Prompt write failures retry with capped exponential
  backoff at 15, 30, 60, 120, 240, and then 300 seconds; a successful write
  resets the sequence. Stalled single-flight writes start this backoff only if
  they eventually resolve with an error. Dashboard persistence status exposes
  the consecutive failure count, current delay, next retry time, and remaining
  delay.
- `DASHBOARD_DIRTY_BUCKET_WARNING_AGE_S`: age at which the load balancer warns
  that dashboard minute persistence is falling behind (defaults to 300 seconds)
- `DASHBOARD_STARTUP_MERGE_DELAY_S`: interval before each of two startup history
  reconciliation reads that merge files written late by the previous LB replica
  (defaults to 60 seconds, so passes run at roughly 60 and 120 seconds; set to 0
  to disable). Each read covers only the previous UTC day through the current
  minute, independently of the full dashboard retention setting. Choose an
  interval whose two-pass window covers the old replica's worst-case shutdown
  drain; a longer interval improves late-write coverage but delays final
  reconciliation status.
- `DASHBOARD_PREVIEW_MODE`: set to `true` to serve the dashboard with synthetic
  endpoint/session data instead of connecting to real compute endpoints. You can
  also set `COMPUTE_ENDPOINT_NAMES=TEST` for the same local preview behavior.
  Synthetic endpoints retain their own last-known `max_sessions` capacity
  through initializing, paused, and updating states.
  If `DASHBOARD_BUCKET_ID` is set, preview mode loads existing dashboard history
  from the bucket read-only and never writes preview data back to the bucket.
- `DASHBOARD_BUCKET_ID`: optional HF storage bucket id used to persist dashboard history
- `DASHBOARD_BUCKET_PREFIX`: path prefix inside the bucket for dashboard files
- `DASHBOARD_BUCKET_TOKEN`: optional token override for bucket reads/writes

## Compute Env Vars

- `NUM_PIPELINES`: concurrent realtime sessions the `speech-to-speech` process handles internally (default `1`)
- `SESSION_SHARED_SECRET`: shared secret used to validate LB-issued session tokens
- `LB_CALLBACK_AUTH_TOKEN`: dedicated bearer credential used to authenticate
  compute accounting callbacks to the LB. This is a service credential, not the
  requester's HF token.
- `LLM_PROXY_ACCOUNTING_CALLBACK_URL`: full load-balancer
  `/internal/llm-proxy-usage` URL. Every LLM proxy attempt sends one best-effort,
  at-most-once callback containing its canonical gate reason and, when
  available, the presented HF token and trusted client IP.
- `LLM_PROXY_TRUST_PROXY_HEADERS`: whether proxy accounting trusts the first
  `X-Forwarded-For`/`X-Real-IP` address (defaults to `true`). Disable this
  outside a trusted reverse-proxy deployment.
- `ENABLE_SMART_TURN`: enables Smart Turn endpointing (default `1`); set it to
  `0` to disable it.
- `SMART_TURN_MODEL_PATH`: optional Smart Turn ONNX checkpoint path. The
  compute image defaults to `/opt/models/smart-turn-v3.2-cpu.onnx`.
- `EXTRA_S2S_ARGS`: optional advanced CLI arguments passed to
  `speech-to-speech`.
- `ENABLE_LLM_PROXY`: master switch for the LLM proxy feature — passes
  `--enable_llm_proxy` to the internal `speech-to-speech` server and opens the
  replica's `/v1/chat/completions` and `/v1/responses` proxy paths. Defaults
  off, in which case those paths answer 404, indistinguishable from a build
  without the feature.
- `LLM_PROXY_REQUESTS_PER_MINUTE`: per-token sliding-window rate limit on the
  replica's LLM proxy paths (defaults to 20; zero or negative closes the paths)

The accounting callback must use HTTPS outside local development. It carries
the presented HF token and trusted client IP in the request body only so the LB
can resolve token identity first and fall back to its privacy-safe network
fingerprint. Neither raw value is retained in dashboard history or exposed by
dashboard APIs/UI, and callback payloads must not be logged by the compute, LB,
or an upstream proxy.

The compute endpoint serves `/v1/realtime`. The LB now serves `POST /session` for allocation.

## Create Compute Endpoints

The repo includes a helper script to create GPU compute endpoints for this app:

```bash
uv run --with-requirements requirements.txt python scripts/create_compute_endpoints.py \
  --namespace your-org \
  --prefix reachy-s2s \
  --count 3 \
  --image-url your-registry/s2s-endpoint-compute:latest \
  --image-port 7860 \
  --session-shared-secret your-shared-secret \
  --lb-callback-auth-token your-callback-secret \
  --llm-proxy-accounting-callback-url https://your-lb.example/internal/llm-proxy-usage \
  --secret HF_TOKEN=$HF_TOKEN \
  --instance-size x1 \
  --instance-type nvidia-a10g \
  --vendor aws \
  --region us-east-1 \
  --wait
```

To create compute endpoints backed by the upstream realtime server on `main`, use the realtime image:

```bash
uv run --with-requirements requirements.txt python scripts/create_compute_endpoints.py \
  --namespace your-org \
  --prefix reachy-s2s \
  --count 3 \
  --image-url your-registry/s2s-endpoint-compute:realtime \
  --session-shared-secret your-shared-secret \
  --lb-callback-auth-token your-callback-secret \
  --llm-proxy-accounting-callback-url https://your-lb.example/internal/llm-proxy-usage \
  --secret HF_TOKEN=$HF_TOKEN \
  --instance-size x1 \
  --instance-type nvidia-a10g \
  --vendor aws \
  --region us-east-1 \
  --wait
```

To add endpoints without touching existing lower-numbered endpoints, use
`--target-total`. For example, to grow a `reachy-s2s-01` through
`reachy-s2s-08` pool to 64 endpoints, the script checks the existing
sequential pool and creates only `reachy-s2s-09` through `reachy-s2s-64`:

```bash
uv run --with-requirements requirements.txt python scripts/create_compute_endpoints.py \
  --namespace your-org \
  --prefix reachy-s2s \
  --target-total 64 \
  --copy-env-from reachy-s2s-01 \
  --image-url your-registry/s2s-endpoint-compute:latest \
  --image-port 7860 \
  --secret-file production-compute-secrets.json \
  --instance-size x1 \
  --instance-type nvidia-a10g \
  --vendor aws \
  --region us-east-1 \
  --wait
```

`--copy-env-from` copies readable env vars from an existing endpoint. Secret
values are not readable from existing endpoints, so pass the same secrets again
with `--secret-file` or `--secret`.

The script prints the created endpoint names and HTTPS URLs as JSON. The LB can
receive that pool either as explicit `COMPUTE_ENDPOINT_NAMES` or through the
helper scripts' prefix/count arguments.

For the direct-session architecture, compute endpoints are usually created as
`public` endpoints so clients can connect directly after the LB assigns them a
session token.

When targeting a third-party service protected by an API key, compute endpoints must also be configured with an `HF_TOKEN` or `RESPONSES_API_API_KEY` secret at runtime.

The speech-to-speech wrapper defaults to `LLM=chat-completions`. To use the Hugging Face router, for example, set the following environment variables:
`MODEL_NAME=google/gemma-4-31B-it:cerebras`,
`RESPONSES_API_BASE_URL=https://router.huggingface.co/v1`, and
`RESPONSES_API_REASONING_EFFORT=none`.

## Update Compute Endpoint Env

To update env vars across an existing compute pool, use the dedicated updater:

```bash
uv run --with-requirements requirements.txt python scripts/update_compute_endpoints_env.py \
  --namespace your-org \
  --prefix reachy-s2s \
  --count 8 \
  --env MODEL_NAME=Qwen/Qwen3.5-72B:together \
  --env LLM=chat-completions \
  --env RESPONSES_API_BASE_URL=https://router.huggingface.co/v1 \
  --env RESPONSES_API_REASONING_EFFORT=none \
  --wait
```

For the Qwen3 CustomVoice setup we used in production, the update command was:

```bash
uv run --with-requirements requirements.txt python scripts/update_compute_endpoints_env.py \
  --namespace HuggingFaceM4 \
  --prefix reachy-s2s \
  --count 8 \
  --env 'EXTRA_S2S_ARGS=--qwen3_tts_model_name Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --qwen3_tts_speaker Aiden --qwen3_tts_language English --qwen3_tts_ref_audio=' \
  --no-wait
```

The trailing `--qwen3_tts_ref_audio=` is intentional. Without it, the upstream
Qwen3 TTS handler keeps its default reference audio path and incorrectly takes
the voice-cloning path even when you are using a `CustomVoice` model.

The script fetches each endpoint's current env, merges the requested changes,
and sends the full updated env back to Hugging Face. By default, it updates the
selected compute endpoints in parallel and waits for them in parallel too. That
matters because the endpoint update API replaces the env payload instead of
patching it.

To repair a newly added tail so it matches an existing endpoint's readable env
without touching lower-numbered endpoints, select the tail explicitly and copy
from a known-good endpoint. Re-supply the production secrets because existing
secret values cannot be copied back from the API:

```bash
uv run --with-requirements requirements.txt python scripts/update_compute_endpoints_env.py \
  --namespace HuggingFaceM4 \
  --names $(printf 'reachy-s2s-%02d ' {9..64}) \
  --copy-env-from reachy-s2s-01 \
  --secret-file production-compute-secrets.json \
  --wait
```

Useful options:

- `--unset-env KEY`: remove an env var from every selected compute endpoint
- `--env-file path.json`: load several env updates from JSON
- `--copy-env-from NAME`: replace selected endpoint envs with readable env vars
  from an existing endpoint before applying overrides
- `--dry-run`: print the planned changes without applying them
- `--no-wait`: submit updates without waiting for each endpoint to return to
  its target state
- `--parallelism 1`: force sequential updates instead of the default parallel
  rollout

## Create Load Balancer Endpoint

The repo also includes a helper script to create the CPU load-balancer endpoint:

```bash
uv run --with-requirements requirements.txt python scripts/create_load_balancer_endpoint.py \
  --name reachy-s2s-lb \
  --namespace your-org \
  --image-url your-registry/s2s-endpoint-lb:latest \
  --image-port 7860 \
  --session-shared-secret your-shared-secret \
  --secret HF_CONTROL_TOKEN=$HF_TOKEN \
  --secret LB_ADMIN_AUTH_TOKEN=$LB_ADMIN_AUTH_TOKEN \
  --secret LB_CALLBACK_AUTH_TOKEN=$LB_CALLBACK_AUTH_TOKEN \
  --instance-size x2 \
  --instance-type intel-icl \
  --vendor aws \
  --region us-east-1 \
  --compute-endpoint-prefix reachy-s2s \
  --compute-endpoint-count 3 \
  --compute-endpoint-min-warm 1 \
  --compute-endpoint-wake-threshold-slots 1 \
  --compute-endpoint-idle-park-timeout-s 300 \
  --compute-endpoint-park-strategy pause \
  --session-pending-timeout-s 60 \
  --wait
```

The load balancer exposes two different health-style routes:

- `/ready`: lightweight process readiness for the Hugging Face platform health check
- `/health`: swarm health, which can return `503` when the compute pool is cold or still warming

For the load balancer image, the endpoint health route should therefore be `/ready`, not `/health`.

Both scripts are specific to this repo and expect the role-specific images:

- compute endpoints: image built from `Dockerfile.compute`
- load balancer endpoint: image built from `Dockerfile.load_balancer`

## Update Load Balancer Endpoint Env

To update env vars on the existing load-balancer endpoint, use the dedicated updater:

```bash
uv run --with-requirements requirements.txt python scripts/update_load_balancer_endpoint_env.py \
  --namespace HuggingFaceM4 \
  --name reachy-s2s-lb \
  --compute-endpoint-prefix reachy-s2s \
  --compute-endpoint-count 64 \
  --compute-endpoint-min-warm 3 \
  --compute-endpoint-wake-threshold-slots 3 \
  --wait
```

To require verified Hugging Face users before granting session capacity:

```bash
uv run --with-requirements requirements.txt python scripts/update_load_balancer_endpoint_env.py \
  --namespace HuggingFaceM4 \
  --name reachy-s2s-lb \
  --env SESSION_REQUIRE_VERIFIED_HF_TOKEN=true \
  --env SESSION_HF_TOKEN_VERIFY_TIMEOUT_S=5 \
  --env SESSION_HF_TOKEN_MAX_VERIFIED_AGE_S=1800 \
  --env SESSION_HF_TOKEN_VERIFY_MAX_PENDING=64 \
  --env SESSION_HF_TOKEN_VERIFY_MAX_PENDING_PER_NETWORK=4 \
  --wait
```

The prefix/count form expands to the existing `COMPUTE_ENDPOINT_NAMES` env var,
so it works with the current load-balancer image while avoiding a long manual
comma-separated list.

To enable persisted dashboard history using a Hugging Face Storage Bucket, the command we used was:

```bash
uv run --with-requirements requirements.txt python scripts/update_load_balancer_endpoint_env.py \
  --namespace HuggingFaceM4 \
  --name reachy-s2s-lb \
  --env DASHBOARD_BUCKET_ID=HuggingFaceM4/reachy-s2s-dashboard \
  --env DASHBOARD_BUCKET_PREFIX=reachy-s2s-lb \
  --wait
```

Like the compute env updater, this script fetches the current env first, merges the requested changes, and sends the full updated env back to Hugging Face.

## Download Endpoint Logs

To download the current Hugging Face Inference Endpoint log tails into local files:

```bash
uv run --with-requirements requirements.txt python scripts/download_endpoint_logs.py \
  --namespace HuggingFaceM4 \
  --output-dir logs/endpoints \
  --tail 10000
```

By default the script downloads the load-balancer logs and discovers the compute
pool from the load balancer's `COMPUTE_ENDPOINT_NAMES` env var. You can override
the compute selection with `--compute-names` or `--compute-prefix` /
`--compute-count`, and use `--skip-load-balancer` or `--no-compute` to narrow the
download. `--tail` means the maximum number of most-recent log lines to request
per endpoint. If some endpoint log requests do not return, reduce
`--timeout-s` to fail those endpoints quickly and report the error in the JSON
summary.

For large replica downloads, the final JSON is concise by default. Add
`--verbose` to print every endpoint/replica as it starts and finishes, or
`--include-results` to include one JSON result object per downloaded file.

To retrieve logs per historical replica for a specific time window, use the
paginated v3 logs API mode:

```bash
uv run --with-requirements requirements.txt python scripts/download_endpoint_logs.py \
  --namespace HuggingFaceM4 \
  --output-dir logs/endpoints-replicas \
  --skip-load-balancer \
  --all-replicas \
  --since 2026-05-05T00:00:00Z \
  --until 2026-05-06T12:00:00Z \
  --parallelism 4 \
  --timeout-s 30
```

This first asks the metrics API which replica ids existed in the window, then
writes one log file per endpoint/replica.

## Update Endpoint Images

To roll out a new compute image, a new load-balancer image, or both, use:

```bash
uv run --with-requirements requirements.txt python scripts/update_endpoints_images.py \
  --namespace HuggingFaceM4 \
  --compute andito/s2s-compute:v0.4
```

That compute-only command updates the current pool configured on the load
balancer. To update both compute and load-balancer images in one run, pass both
image arguments:

```bash
uv run --with-requirements requirements.txt python scripts/update_endpoints_images.py \
  --namespace HuggingFaceM4 \
  --compute andito/s2s-compute:v0.3 \
  --load_balancer andito/s2s-load_balancer:v0.11
```

To avoid interrupting active direct sessions, drain each compute endpoint before
updating it:

```bash
uv run --with-requirements requirements.txt python scripts/update_endpoints_images.py \
  --namespace HuggingFaceM4 \
  --compute andito/s2s-compute:v0.4 \
  --compute-drain \
  --load-balancer-admin-token "$LB_ADMIN_AUTH_TOKEN"
```

Drain mode asks the load balancer to stop assigning new sessions to one compute
endpoint, then waits until it is either parked with zero active sessions or
running with zero active sessions from a successful compute usage request that
started after the drain was acquired. Drain acquisition returns a conflict if
the endpoint is already waking, parking, restarting, or undergoing stuck-pipeline
recovery. It rechecks the drain, transition flags, and safe-idle state immediately
before submitting the image update, then makes the endpoint available again
after a confirmed update. The updater renews a server-side drain lease while it
waits for idle and again immediately before submission. An explicit HF 4xx
rejection proves that the update did not start, so the drain is cleared. Network
errors, timeouts, and HF 5xx responses remain fail-closed, but their drains
expire automatically instead of removing capacity indefinitely. The requested
lease is at least 900 seconds and at least five minutes longer than the endpoint
update wait timeout. The LB logs recurring warnings for drains older than its
configured warning threshold.

To clear a drain sooner after verifying that no update is active and the
endpoint is safe to reopen:

```bash
curl --fail-with-body -X POST \
  -H "X-Reachy-Mini-Admin-Authorization: Bearer $LB_ADMIN_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"draining": false, "force": true}' \
  "$LOAD_BALANCER_URL/internal/endpoints/reachy-s2s-01/drain"
```

The dedicated admin token must match `LB_ADMIN_AUTH_TOKEN` on the deployed load
balancer. For the first deployment of drain support, configure that secret and
deploy the new load-balancer image before running a compute drain rollout. Verify
the authenticated endpoint-status route returns a snapshot containing
`drain_lease_remaining_s`; this ensures the status route, dedicated auth, and
lease-capable router are live. Do not use the combined compute-plus-LB command
for that first rollout because it intentionally updates compute endpoints first.

Behavior:

- if you pass `--compute`, the script updates the compute pool first
- if you pass `--load_balancer`, it updates the load-balancer endpoint
- if you omit either one, that side is skipped
- if you do not provide compute names explicitly, the script reads the current
  compute pool from the load balancer's `COMPUTE_ENDPOINT_NAMES` env var
- if the load balancer does not have `COMPUTE_ENDPOINT_NAMES`, the script falls
  back to deriving the compute prefix from the load-balancer name and scanning
  `-01`, `-02`, ... until the first missing endpoint
- compute endpoint updates now run in parallel by default; use `--compute-parallelism 1` if you want the old sequential rollout behavior
- with `--compute-drain`, compute endpoint updates run one at a time by default
  and only after the load balancer reports a still-active drain and either a
  parked endpoint with zero sessions or a transition-free running endpoint with
  a post-drain usage observation and zero active sessions
- matching compute images are detected before drain acquisition, so no-op
  rollouts do not temporarily remove capacity
- explicit `--compute-parallelism N` permits up to `N` simultaneous drains;
  on systemic ambiguous HF failures those leases remain until they expire, so
  keep the default sequential behavior unless parallel draining is intentional
- `--compute-drain` requires waiting for the endpoint update to finish and cannot
  be combined with `--no-wait`
- malformed safety snapshots fail closed; running endpoints must explicitly
  report required and completed usage synchronization after drain acquisition,
  all control-plane transition flags must be false, and an active drain lease
  must be present
- with `--wait` (the default), the command waits for all selected endpoint updates to finish before returning; use `--no-wait` if you want to submit the updates and return immediately
- completion lines are printed as each endpoint finishes, so parked endpoints are reported immediately even if a few running endpoints are still becoming healthy
- paused or scale-to-zero compute endpoints keep their parked state after the image update, and the script now waits for them to return to that original parked state instead of incorrectly waiting for `running`
- load-balancer updates automatically force the custom-image health route to `/ready`, even if the image URL itself is unchanged

Useful options:

- `--load-balancer-name`: defaults to `reachy-s2s-lb`
- `--compute-names reachy-s2s-01 reachy-s2s-02`: override the LB env and update
  only these compute endpoints
- `--compute-prefix reachy-s2s --compute-count 8`: override the LB env and
  update a generated prefix/count set
- `--compute-parallelism 1`
- `--compute-drain`
- `--compute-drain-timeout-s 7200`
- `--load-balancer-admin-token "$LB_ADMIN_AUTH_TOKEN"`
- `--no-wait`
- `--dry-run`

## Files
- `app/`: application code
- `scripts/`: helper scripts
- `Dockerfile.compute`: compute container definition
- `Dockerfile.load_balancer`: load-balancer container definition
- `requirements.txt`: Python dependencies
- `test_ws_file.py`: websocket test client

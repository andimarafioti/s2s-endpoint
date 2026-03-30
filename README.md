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

This repo now builds two different images with two different app entrypoints:

- compute image: `Dockerfile.compute`
  Starts `app.compute_main:app` on a GPU instance, runs local `speech-to-speech` subprocesses in upstream `--mode realtime`, and serves `/v1/realtime` directly.
- load-balancer image: `Dockerfile.load_balancer`
  Starts `app.load_balancer_main:app` on a CPU instance, tracks a configured set of pre-created compute endpoints, keeps a warm pool, wakes parked endpoints when free session capacity gets tight, and allocates direct compute sessions for clients.

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

Today `Dockerfile.compute` defaults `S2S_REPO_URL=https://github.com/huggingface/speech-to-speech.git` and `S2S_REF=openai_realtime_server_api`, because this repo now assumes the realtime server path. If you need to override that repo/ref explicitly, use:

```bash
docker build --platform linux/amd64 -f Dockerfile.compute \
  --build-arg S2S_REPO_URL=https://github.com/huggingface/speech-to-speech.git \
  --build-arg S2S_REF=openai_realtime_server_api \
  -t your-registry/s2s-endpoint-compute:realtime .
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
- `POST /session` request counts, allocation successes/failures, and connect/disconnect events
- conversation starts/completions plus average and max completed conversation duration

The timeline automatically switches between minute-level and hourly rollups depending on the selected window. By default the history is in memory and resets when the LB endpoint restarts.

If you want the dashboard history to survive LB restarts, you can configure it to persist completed minute buckets to a Hugging Face Storage Bucket. The live routing/session state still stays in memory; the bucket is only for historical dashboard data.

## Load Balancer Env Vars

- `HF_ENDPOINT_NAMESPACE`: namespace that owns the compute endpoints
- `COMPUTE_ENDPOINT_NAMES`: comma-separated endpoint names
- `COMPUTE_ENDPOINT_SLOTS`: concurrent sessions each compute endpoint can handle
- `COMPUTE_ENDPOINT_MIN_WARM`: number of compute endpoints that should stay warm
- `COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS`: when total free slots drop to this level,
  the LB starts waking another parked endpoint
- `COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S`: how long an idle compute endpoint stays warm
  before being parked
- `COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S`: background refresh interval
- `COMPUTE_ENDPOINT_PARK_STRATEGY`: `pause` or `scale_to_zero`
- `HF_CONTROL_TOKEN`: token used to call the Inference Endpoints API
- `SESSION_SHARED_SECRET`: shared secret used to mint and validate direct session tokens
- `SESSION_PENDING_TIMEOUT_S`: how long an unused reservation stays alive
- `SESSION_TOKEN_TTL_S`: lifetime of the signed session token
- `SESSION_REAP_INTERVAL_S`: how often the LB reaps unused reservations
- `DASHBOARD_SAMPLE_INTERVAL_S`: how often the LB samples swarm state for history
- `DASHBOARD_RETENTION_MINUTES`: in-memory history retention for dashboard data
- `DASHBOARD_BUCKET_ID`: optional HF storage bucket id used to persist dashboard history
- `DASHBOARD_BUCKET_PREFIX`: path prefix inside the bucket for dashboard files
- `DASHBOARD_BUCKET_TOKEN`: optional token override for bucket reads/writes

## Compute Env Vars

- `PIPELINE_MAX_INSTANCES`: local `speech-to-speech` pipelines per compute endpoint
- `PIPELINE_MIN_IDLE_INSTANCES`: warm local pipeline slots to keep ready
- `SESSION_SHARED_SECRET`: shared secret used to validate LB-issued session tokens
- `LB_CALLBACK_AUTH_TOKEN`: optional bearer token used when compute endpoints call the LB session-event API

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
  --secret HF_TOKEN=$HF_TOKEN \
  --instance-size x1 \
  --instance-type nvidia-a10g \
  --vendor aws \
  --region us-east-1 \
  --pipeline-max-instances 1 \
  --pipeline-min-idle-instances 1 \
  --wait
```

To create compute endpoints backed by the upstream OpenAI Realtime API branch, use the realtime image:

```bash
uv run --with-requirements requirements.txt python scripts/create_compute_endpoints.py \
  --namespace your-org \
  --prefix reachy-s2s \
  --count 3 \
  --image-url your-registry/s2s-endpoint-compute:realtime \
  --session-shared-secret your-shared-secret \
  --secret HF_TOKEN=$HF_TOKEN \
  --instance-size x1 \
  --instance-type nvidia-a10g \
  --vendor aws \
  --region us-east-1 \
  --pipeline-max-instances 1 \
  --pipeline-min-idle-instances 1 \
  --wait
```

The script prints the created endpoint names and HTTPS URLs as JSON. Those names
should then be passed to the LB with `COMPUTE_ENDPOINT_NAMES`.

For the direct-session architecture, compute endpoints are usually created as
`public` endpoints so clients can connect directly after the LB assigns them a
session token.

With the current defaults, compute endpoints also need an `HF_TOKEN` or
`OPEN_API_API_KEY` secret at runtime because the speech-to-speech wrapper
defaults to `LLM=open_api`.

## Update Compute Endpoint Env

To update env vars across an existing compute pool, use the dedicated updater:

```bash
uv run --with-requirements requirements.txt python scripts/update_compute_endpoints_env.py \
  --namespace your-org \
  --prefix reachy-s2s \
  --count 8 \
  --env OPEN_API_MODEL_NAME=Qwen/Qwen3.5-72B:together \
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

Useful options:

- `--unset-env KEY`: remove an env var from every selected compute endpoint
- `--env-file path.json`: load several env updates from JSON
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
  --instance-size x2 \
  --instance-type intel-icl \
  --vendor aws \
  --region us-east-1 \
  --compute-endpoint-names reachy-s2s-01,reachy-s2s-02,reachy-s2s-03 \
  --compute-endpoint-slots 1 \
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
  --env COMPUTE_ENDPOINT_MIN_WARM=2 \
  --env COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS=2 \
  --wait
```

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

## Update Endpoint Images

To roll out a new compute image, a new load-balancer image, or both, use:

```bash
uv run --with-requirements requirements.txt python scripts/update_endpoints_images.py \
  --namespace HuggingFaceM4 \
  --compute andito/s2s-compute:v0.3 \
  --load_balancer andito/s2s-load_balancer:v0.11
```

Behavior:

- if you pass `--compute`, the script updates the compute pool first
- if you pass `--load_balancer`, it updates the load-balancer endpoint
- if you omit either one, that side is skipped
- if you do not provide compute names explicitly, the script derives the compute prefix from the load-balancer name and scans `-01`, `-02`, ... until the first missing endpoint
- with the default LB name `reachy-s2s-lb`, that means it discovers `reachy-s2s-01`, `reachy-s2s-02`, and so on, then prints a summary like `updated endpoints 1 through 8`
- compute endpoint updates now run in parallel by default; use `--compute-parallelism 1` if you want the old sequential rollout behavior
- with `--wait` (the default), the command waits for all selected endpoint updates to finish before returning; use `--no-wait` if you want to submit the updates and return immediately
- completion lines are printed as each endpoint finishes, so parked endpoints are reported immediately even if a few running endpoints are still becoming healthy
- paused or scale-to-zero compute endpoints keep their parked state after the image update, and the script now waits for them to return to that original parked state instead of incorrectly waiting for `running`
- load-balancer updates automatically force the custom-image health route to `/ready`, even if the image URL itself is unchanged

Useful options:

- `--load-balancer-name`: defaults to `reachy-s2s-lb`
- `--compute-names reachy-s2s-01 reachy-s2s-02`
- `--compute-prefix reachy-s2s --compute-count 8`
- `--compute-parallelism 1`
- `--no-wait`
- `--dry-run`

## Files
- `app/`: application code
- `scripts/`: helper scripts
- `Dockerfile.compute`: compute container definition
- `Dockerfile.load_balancer`: load-balancer container definition
- `requirements.txt`: Python dependencies
- `test_ws_file.py`: websocket test client

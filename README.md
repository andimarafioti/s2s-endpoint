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

Today `Dockerfile.compute` defaults `S2S_REPO_URL=https://github.com/huggingface/speech-to-speech.git` and `S2S_REF=main`, because the realtime server path now lives on upstream `main`. If you need to override that repo/ref explicitly, use:

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
  --build-arg QWENTTS_WHEEL_URL=https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/resolve/main/whl/cu130/qwentts_cpp_python-0.3.0%2Bcu130-py3-none-manylinux_2_39_x86_64.whl \
  --build-arg QWENTTS_WHEEL_FILENAME=qwentts_cpp_python-0.3.0+cu130-py3-none-manylinux_2_39_x86_64.whl \
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

Persisted history is restored in the background during load-balancer startup, so
the endpoint can become ready before older dashboard buckets finish loading. The
`/dashboard/data` response includes a `history_restore` object with the restore
status, elapsed time, and restored bucket count.

The dashboard store keeps minute files under `minutes/YYYY-MM-DD/` and also
uses `days/YYYY-MM-DD.json` files as a compact cache for UTC days. On restore it
checks `days/` first, falls back to minute files for days without an
authoritative cache, and backfills a complete `days/` file once it has all 1,440
minute buckets for a completed day. While the load balancer stays running, it
also rolls over each completed UTC day from in-memory history into
`days/YYYY-MM-DD.json` shortly after midnight UTC. If the day is missing minute
buckets, the rollover still writes a finalized partial day file with
`complete: false`, `finalized: true`, and a missing-minute count so later
restores do not redownload the same minutes forever. Older/open partial day
files without `finalized: true` are still allowed to merge newly appeared minute
files and then become finalized.

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
- `LB_ADMIN_AUTH_TOKEN`: dedicated bearer token required by the internal endpoint
  status and drain routes; do not reuse `HF_CONTROL_TOKEN`
- `COMPUTE_ENDPOINT_DRAIN_LEASE_TTL_S`: default allocator-drain lease lifetime
  for admin clients that do not request one explicitly (defaults to 3,600 seconds)
- `COMPUTE_ENDPOINT_DRAIN_WARNING_AFTER_S`: age at which the LB starts warning
  about a continuously drained endpoint (defaults to 600 seconds)
- `COMPUTE_ENDPOINT_DRAIN_WARNING_INTERVAL_S`: interval between repeated
  long-running drain warnings (defaults to 300 seconds)
- `SESSION_SHARED_SECRET`: shared secret used to mint and validate direct session tokens
- `SESSION_PENDING_TIMEOUT_S`: how long an unused reservation stays alive
- `SESSION_TOKEN_TTL_S`: lifetime of the signed session token
- `SESSION_REAP_INTERVAL_S`: how often the LB reaps unused reservations
- `DASHBOARD_SAMPLE_INTERVAL_S`: how often the LB samples swarm state for history
- `DASHBOARD_RETENTION_MINUTES`: in-memory history retention for dashboard data
  (defaults to 28 days so the 14d/28d dashboard windows can load persisted history)
- `DASHBOARD_PREVIEW_MODE`: set to `true` to serve the dashboard with synthetic
  endpoint/session data instead of connecting to real compute endpoints. You can
  also set `COMPUTE_ENDPOINT_NAMES=TEST` for the same local preview behavior.
  If `DASHBOARD_BUCKET_ID` is set, preview mode loads existing dashboard history
  from the bucket read-only and never writes preview data back to the bucket.
- `DASHBOARD_BUCKET_ID`: optional HF storage bucket id used to persist dashboard history
- `DASHBOARD_BUCKET_PREFIX`: path prefix inside the bucket for dashboard files
- `DASHBOARD_BUCKET_TOKEN`: optional token override for bucket reads/writes

## Compute Env Vars

- `NUM_PIPELINES`: concurrent realtime sessions the `speech-to-speech` process handles internally (default `1`)
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

With the current defaults, compute endpoints also need an `HF_TOKEN` or
`RESPONSES_API_API_KEY` secret at runtime because the speech-to-speech wrapper
defaults to `LLM=responses-api`.

## Update Compute Endpoint Env

To update env vars across an existing compute pool, use the dedicated updater:

```bash
uv run --with-requirements requirements.txt python scripts/update_compute_endpoints_env.py \
  --namespace your-org \
  --prefix reachy-s2s \
  --count 8 \
  --env RESPONSES_API_MODEL_NAME=Qwen/Qwen3.5-72B:together \
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
  --instance-size x2 \
  --instance-type intel-icl \
  --vendor aws \
  --region us-east-1 \
  --compute-endpoint-prefix reachy-s2s \
  --compute-endpoint-count 3 \
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
  --compute-endpoint-prefix reachy-s2s \
  --compute-endpoint-count 64 \
  --compute-endpoint-min-warm 3 \
  --compute-endpoint-wake-threshold-slots 3 \
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
  -H "Authorization: Bearer $LB_ADMIN_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"draining": false}' \
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

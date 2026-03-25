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
  Starts `app.compute_main:app` on a GPU instance, runs local `speech-to-speech` subprocesses, and serves `/ws` directly.
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
3. Client connects directly to the compute endpoint `/ws`.
4. Compute validates the session token and notifies the LB when the session starts and ends.

This removes the LB from the websocket data path. The LB only handles control-plane allocation and release.

In load-balancer mode, the app does not guess endpoint hostnames. It asks the
Hugging Face API for each compute endpoint's canonical HTTPS URL and turns that
into the direct websocket URL by replacing `https://` with `wss://` and appending
`/ws`.

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

## Compute Env Vars

- `PIPELINE_MAX_INSTANCES`: local `speech-to-speech` pipelines per compute endpoint
- `PIPELINE_MIN_IDLE_INSTANCES`: warm local pipeline slots to keep ready
- `SESSION_SHARED_SECRET`: shared secret used to validate LB-issued session tokens
- `LB_CALLBACK_AUTH_TOKEN`: optional bearer token used when compute endpoints call the LB session-event API

The compute endpoint still serves `/ws`. The LB now serves `POST /session` for allocation.

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

## Create Load Balancer Endpoint

The repo also includes a helper script to create the CPU load-balancer endpoint:

```bash
uv run --with-requirements requirements.txt python scripts/create_load_balancer_endpoint.py \
  --name reachy-s2s-lb \
  --namespace your-org \
  --image-url your-registry/s2s-endpoint-lb:latest \
  --image-port 7860 \
  --session-shared-secret your-shared-secret \
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

Both scripts are specific to this repo and expect the role-specific images:

- compute endpoints: image built from `Dockerfile.compute`
- load balancer endpoint: image built from `Dockerfile.load_balancer`

## Files
- `app/`: application code
- `scripts/`: helper scripts
- `Dockerfile.compute`: compute container definition
- `Dockerfile.load_balancer`: load-balancer container definition
- `requirements.txt`: Python dependencies
- `test_ws_file.py`: websocket test client

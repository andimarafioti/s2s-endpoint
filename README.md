---
library_name: none
tags:
- speech
- audio
- inference-endpoint
---

# s2s-endpoint

Speech-to-speech endpoint project.

## Roles

The same container can run in two different roles:

- `APP_ROLE=compute`
  Runs local `speech-to-speech` subprocesses and serves `/ws` directly.
- `APP_ROLE=load_balancer`
  Does not start local `speech-to-speech`. Instead, it tracks a configured set of
  pre-created Hugging Face Inference Endpoints, keeps a warm pool, wakes parked
  endpoints when free session capacity gets tight, and proxies `/ws` to the selected
  compute endpoint.

This is intended for a deployment with:

- one load-balancer endpoint
- multiple compute endpoints
- one compute endpoint per unit of isolated capacity

The load balancer keeps session counts in memory, so it should run as a single
replica unless you add shared state outside this repo.

## URL Selection

In load-balancer mode, the app does not guess endpoint hostnames. It asks the
Hugging Face API for each compute endpoint's canonical HTTPS URL and then turns
that into a websocket URL by replacing `https://` with `wss://` and appending
`/ws`.

For example:

- endpoint URL from HF API: `https://my-endpoint.endpoints.huggingface.cloud`
- websocket URL used by the LB: `wss://my-endpoint.endpoints.huggingface.cloud/ws`

## Load Balancer Env Vars

- `APP_ROLE=load_balancer`
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
- `DOWNSTREAM_ENDPOINT_TOKEN`: optional token used by the LB when opening websocket
  connections to protected compute endpoints

## Compute Env Vars

- `APP_ROLE=compute`
- `PIPELINE_MAX_INSTANCES`: local `speech-to-speech` pipelines per compute endpoint
- `PIPELINE_MIN_IDLE_INSTANCES`: warm local pipeline slots to keep ready

The external websocket API remains `/ws` in both roles.

## Create Compute Endpoints

The repo includes a helper script to create a batch of compute endpoints:

```bash
uv run --with-requirements requirements.txt python scripts/create_compute_endpoints.py \
  --namespace your-org \
  --prefix reachy-s2s \
  --count 3 \
  --repository andimarafioti/s2s-endpoint \
  --framework custom \
  --accelerator gpu \
  --instance-size x1 \
  --instance-type nvidia-a10g \
  --vendor aws \
  --region us-east-1 \
  --custom-image-file custom_image.json \
  --env APP_ROLE=compute \
  --env PIPELINE_MAX_INSTANCES=1 \
  --env PIPELINE_MIN_IDLE_INSTANCES=1 \
  --wait
```

The script prints the created endpoint names and HTTPS URLs as JSON. Those names
should then be passed to the LB with `COMPUTE_ENDPOINT_NAMES`.

## Files
- `app/`: application code
- `scripts/`: helper scripts
- `Dockerfile`: container definition
- `requirements.txt`: Python dependencies
- `test_ws_file.py`: websocket test client

---
title: Reachy Mini Realtime URL
sdk: docker
app_port: 7860
pinned: false
---

# Reachy Mini Realtime URL

Small FastAPI service that gives the Reachy Mini conversation app a stable
session endpoint.

## Configuration

Set this Space variable/secret to the current session allocator URL:

- `UPSTREAM_SESSION_URL`

The service validates that the configured value is an absolute `http` or
`https` URL.

Optional:

- `REQUEST_TIMEOUT_SECONDS`: upstream request timeout, default `10`
- `LOG_LEVEL`: Python logging level, default `INFO`

## Routes

- `GET /health`: process health; does not fail if the upstream URL is missing
- `GET /ready`: readiness; fails if the upstream URL is missing or invalid
- `GET /session-url`: returns the currently configured upstream allocator URL
- `GET /config`: alias for `/session-url`
- `POST /session`: proxies the session allocation request to the upstream URL

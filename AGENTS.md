# User Preferences

- Never include `codex` in branch names or pull request titles.
- Always make changes as consecutive commits, with each commit preserving exactly what changed at that step. Do not fold follow-up work into earlier commits unless explicitly asked.
- For existing/open pull requests, do not amend, squash, rebase-rewrite, or force-push follow-up changes. Make new commits and push normally unless explicitly asked to rewrite history.

# Operational Log Analysis Notes

These notes capture the current working understanding of the Reachy S2S endpoint logs and dashboard semantics.

## Environment

- Production namespace used in recent investigations: `HuggingFaceM4`.
- Load balancer endpoint name: `reachy-s2s-lb`.
- Compute endpoints are normally `reachy-s2s-01` through `reachy-s2s-32`.
- Most live checks require `HF_TOKEN` in the environment.
- Prefer discovering endpoint URLs from the Hugging Face endpoint API or from load-balancer `/health`; do not rely on a hard-coded endpoint URL unless the user just provided one.

## Useful Commands

Download current logs from the load balancer:

```bash
uv run --with-requirements requirements.txt python scripts/download_endpoint_logs.py \
  --namespace HuggingFaceM4 \
  --names reachy-s2s-lb \
  --output-dir logs/lb-current-$(date -u +%Y%m%dT%H%M%SZ) \
  --tail 10000000 \
  --timeout-s 60 \
  --parallelism 1 \
  --quiet \
  --include-results
```

Download a time-bounded load-balancer window with v3 logs:

```bash
uv run --with-requirements requirements.txt python scripts/download_endpoint_logs.py \
  --namespace HuggingFaceM4 \
  --names reachy-s2s-lb \
  --output-dir logs/lb-v3-YYYYMMDDTHHMMZ-YYYYMMDDTHHMMZ \
  --since 2026-06-10T00:00:00Z \
  --until 2026-06-11T00:00:00Z \
  --log-api-version v3 \
  --v3-limit 5000 \
  --max-pages 100 \
  --timeout-s 60 \
  --parallelism 1 \
  --quiet \
  --include-results
```

Download all compute replicas for historical analysis:

```bash
uv run --with-requirements requirements.txt python scripts/download_endpoint_logs.py \
  --namespace HuggingFaceM4 \
  --output-dir logs/endpoints-replicas-YYYYMMDD \
  --since 2026-06-10T00:00:00Z \
  --until 2026-06-11T00:00:00Z \
  --log-api-version v3 \
  --all-replicas \
  --v3-limit 5000 \
  --max-pages 100 \
  --timeout-s 60 \
  --parallelism 8 \
  --quiet \
  --include-results
```

Analyze downloaded compute logs:

```bash
python3 scripts/analyze_endpoint_logs.py \
  --logs-dir logs/endpoints-replicas-YYYYMMDD \
  --output-dir logs/analysis-replicas-YYYYMMDD
```

## Live State Checks

- Start with the load balancer `/health`. It reports `connected_sessions`, `pending_sessions`, router state, endpoint URLs, session ids, and current session durations.
- Cross-check occupied compute endpoints by calling each compute endpoint `/health` URL from the LB health payload. If the LB says active and compute `/health` also says `router.active_sessions=1`, the compute process still believes the slot is occupied.
- If `observed_active_sessions=0` while compute health says active sessions are present, remember that this was a known deployed-state issue until the health-sync fix is actually released.
- Use `/dashboard/data?window=...&resolution=...` for recent counters and time series. It is useful for trend shape, but its conversation counters have specific semantics below.

## Dashboard Semantics

- `conversations_started_window` is `session_connected_events`. This means successful websocket connects, not unique users and not DAU.
- `conversations_completed_window` is completed disconnect events where the session had connected.
- `session_requests_window` counts `POST /session` allocation attempts.
- `session_successes_window` counts successful allocations, not necessarily users who talked.
- A successful `POST /session` only means a compute slot was allocated. The user has actually joined only after the compute endpoint calls `/internal/sessions/{session_id}/event` with `connected`.
- Pair the first and second successful `/internal/sessions/{session_id}/event` callbacks to estimate session duration from LB logs. One callback only means the session is currently open or the disconnect event is outside the log window.
- Dashboard "conversation" counts can be inflated by reconnects, page refreshes, personality/voice restarts, websocket retries, or clients that connect but never produce STT.
- Dashboard duration metrics can be distorted by stale long-running sessions. When an old stale session finally disconnects, it can create a very large completed duration in the current window.

## Common Patterns

- Many very short sessions are normal in the current architecture. In recent checks, a large share of sessions lasted under 30-60 seconds, consistent with reconnect/retry/UI churn rather than unique human conversations.
- The conversation app can allocate new LB sessions during reconnects and when personality or voice changes force a realtime restart. This can create repeated `POST /session` and short websocket sessions.
- `SESSION_TOKEN_TTL_S` controls token validity, not whether an already-open websocket should stay connected. Raising it prevents token expiry but does not clean up stale sessions.
- A stale-looking session is strongest when:
  - LB reports it connected for a long time.
  - Direct compute `/health` reports `active_sessions=1`.
  - Current compute logs show only `/v1/pool` or health polling, with no recent `USER:`, `ASSISTANT:`, `Transcription completed`, VAD, LLM, or TTS activity.
- A long session is less suspicious when logs still show VAD audio frames, STT, LLM, TTS, or recent user/assistant messages.

## Log Sources And Caveats

- v2 logs (`/logs` via default script mode) are useful for the current active replica and often show the latest live activity. Use an intentionally huge `--tail` when trying to avoid truncation.
- v3 logs support `--since`, `--until`, pagination, and replica selection. They are better shaped for historical analysis, especially with `--all-replicas`, but prior investigations found cases where v3 did not exactly match v2 current logs.
- For missing-log investigations, compare:
  - LB `POST /session` access logs.
  - LB `/internal/sessions/{id}/event` callbacks.
  - Compute logs for matching websocket accepts, disconnects, `USER:`, `ASSISTANT:`, STT, LLM, TTS, and VAD lines.
- If logs show successful `POST /session` but no connect callback, look for expired pending sessions, abandoned allocations, client disconnects during allocation, or users leaving before the endpoint was warm.
- If looking for gateway timeouts, app/container logs may not show errors generated by an upstream Hugging Face/router layer before the request reaches the app.

## Conversation Content Analysis

- Treat conversation content as sensitive. Do local ad hoc analysis only when requested, and do not commit rich content-analysis code unless explicitly asked.
- Exclude partial transcripts by default.
- Primary signals: `USER:`, `ASSISTANT:`, `Language:`, tool calls, `Transcription completed`, and response/token/latency lines.

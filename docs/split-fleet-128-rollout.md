# Split fleet: 128-session rollout

Deployment date: 2026-09-04. Namespace: `HuggingFaceM4`.

**The isolated split-pipeline test fleet now uses 16 session slots per worker.**
The density was raised from four after a current-image single-worker validation;
see [16-slot canary validation](#16-slot-canary-validation--2026-09-25). Two warm
workers provide 32 immediately available slots, and eight 4-vCPU workers are
enough to provide the 128-session target within the observed CPU quota.

GPU autoscaling is enabled for the 128-user target, and all 32 CPU workers are
provisioned. At the original four-slot density, AWS Intel SPR
quota is only 60 vCPUs. Existing non-fleet CPU endpoints consume 15 vCPUs, leaving
45; at four vCPUs per worker this permits 11 workers / 44 sessions. The split LB
was therefore initially restricted to `reachy-s2s-pipeline-02` through `-12`.
The other 21 prepared workers remain paused and unregistered. Nothing was deleted.

Keeping four slots per worker would require an Intel SPR quota of at least
160 vCPUs before registering all 32 names:
128 fleet vCPUs + 15 other vCPUs = 143 required, plus headroom. The provider-quota
API reports units as `maxAccelerators` / `usedAccelerators` even for CPUs; these
are vCPUs here, not endpoint counts. Observed A10G quota is 64 and RTX PRO 6000
quota is 4; the target GPU inventories fit those limits alongside the currently
used resources. Quota is still not a guarantee of regional hardware availability.
The density-test follow-up demonstrated this distinction: TTS scale-out attempts
at 15:43 and 15:46 UTC could not obtain A10G hardware. One worker remained waiting
for hardware and another failed with an out-of-availability error. Neither served
the 32-user tests; both runs used the same existing TTS and LLM GPUs. See the
[follow-up evidence](cpu-pipeline-density-20260904.md#follow-up-gpu-scale-out-was-attempted-but-did-not-supply-extra-capacity).
The 128-user GPU scale-out target is therefore not validated by the quota check
or by the successful single-worker CPU density tests.
At the user's request, the two unsuccessful TTS starts were subsequently paused;
the dedicated CPU density test LB and worker were also confirmed paused. Shared
serving workers and the test autoscalers remain running; endpoint configurations
and benchmark results were retained.

This is a separate test fleet behind `reachy-s2s-split-lb`. The existing
`reachy-s2s-lb`, its monolithic GPU workers, and the directly accessible
`reachy-s2s-pipeline-01` are not migrated. The three existing speech proxies
are updated, so the direct-testing pipeline also benefits from GPU autoscaling.

## 16-slot canary validation — 2026-09-25

All 11 workers registered with `reachy-s2s-split-lb` were changed from
`NUM_PIPELINES=4` to `NUM_PIPELINES=16`. They use the upstream-main canary image
`ghcr.io/andimarafioti/s2s-pipeline:sha-4d134f458b8137e615e89a6967a2a8a0036dd2a0`,
which pins `huggingface/speech-to-speech` main at
`60d2cb23533579fc7c6990b3545d23bce4a139b9`.

The density was validated on the isolated `reachy-s2s-pipeline-33` worker, not
distributed across the fleet. That worker used the same image, 4-vCPU Intel SPR
hardware, Gemma 4 31B NVFP4 backend, and deployed STT/TTS proxies as the test
fleet. The dedicated density load balancer and worker were paused after the
test. Production `reachy-s2s-01` through `-32` were not changed.

| Workload on one worker | Completed turns | Speech stop → first audio p50 / p95 | Peak CPU | Peak RAM |
| --- | ---: | ---: | ---: | ---: |
| Short, staggered; 16 users × 3 turns | 48 / 48 | 0.947 / 1.005 s | 26% | 1.40 GB |
| Short, synchronized; 16 users × 1 turn | 16 / 16 | 1.139 / 1.314 s | 37% | 1.45 GB |
| 19.42-second input, staggered; 16 users × 3 turns | 48 / 48 | 1.435 / 1.596 s | 29% | 1.54 GB |

The short staggered stage p50/p95 intervals were 176/196 ms for STT, 623/631 ms
from completed STT to the first LLM output batch, and 150/205 ms from that batch
to first TTS audio. For the long-input test they were 602/809 ms, 534/620 ms,
and 308/366 ms respectively. Sender scheduling lag remained below 18 ms. These
tests validate 16 connected users on one CPU worker for the exercised workloads;
they do not certify a 128-user fleet burst or prove that 16 is the hardware
maximum.

After the capacity update, `reachy-s2s-split-lb` reported two warm workers, 32
free slots, zero connected or pending sessions, and no router errors. The other
registered workers remained available for autoscaling.

## Browser ingress cutover — 2026-09-23

The `smolagents/hf-realtime-voice` Space successfully allocated sessions from the
split LB but browser WebSockets could not carry HF `Authorization` to the
protected CPU endpoints. All 11 CPU workers registered with `reachy-s2s-split-lb`
were changed in place to `type=public`; none were deleted or recreated. The warm
floor stayed on `-02` and `-03`, while `-04` through `-12` stayed paused. No GPU
proxy or backend and no original production endpoint changed visibility.

Both running CPU workers rejected unsigned WebSocket handshakes with HTTP 403.
One short browser-style conversation, using an LB-issued signed `connect_url`
without an HF WebSocket header, completed STT, LLM, and TTS with 936 ms from
speech stop to first audio. This verifies one manual path, not concurrency or
public-load capacity. The Space still needs a user-side retry to confirm its
full UI flow.

## Temporary dense LLM evaluation — 2026-09-24

The split fleet is temporarily configured for a subjective comparison with
`nvidia/Gemma-4-31B-IT-NVFP4` on the existing single RTX PRO 6000 endpoint
`gemma4-31b-nvfp4-rtx6000-test`. The LLM proxy has only that endpoint in its
inventory, a warm floor and maximum of one worker, and the dense model as its
warmup target. All 11 LB-managed CPU pipelines have the matching `MODEL_NAME`.
The previous `gemma4-26b-a4b-nvfp4-rtx6000-test` endpoint and the two A4B
standbys are paused, not deleted.

The cutover was performed with the split LB paused and no connected or pending
sessions. After the dense backend and proxy were healthy, the LB was resumed;
pipelines `-02` and `-03` returned as the two-worker warm floor with eight free
slots, while `-04` through `-12` remained paused. A synthetic conversation
through the public LB completed STT, dense LLM inference, and TTS. It measured
1.278 seconds from speech stop to first audio, including 165 ms from speech stop
to completed STT, 647 ms from completed STT to the first LLM output batch, and
466 ms from that output to first audio. This is a single smoke turn, not a
latency distribution or concurrency result.

This temporary one-backend proxy configuration deliberately suspends LLM fleet
scale-out during the comparison. To restore the A4B fleet, pause the split LB,
restore `MODEL_NAME=nvidia/Gemma-4-26B-A4B-NVFP4` on all managed pipelines, and
restore the proxy inventory to
`gemma4-26b-a4b-nvfp4-rtx6000-test,reachy-s2s-llm-02,reachy-s2s-llm-03`, with
`LLM_WARMUP_MODEL=nvidia/Gemma-4-26B-A4B-NVFP4` and
`SPEECH_WORKER_MAX_WORKERS=3`. Resume the proxy and A4B warm worker, verify one
ready backend, pause the dense endpoint, and only then resume the split LB.

## Capacity and placement

| Stage | Hardware / region | Per-worker operating target | Warm floor | Maximum workers | Inventory |
| --- | --- | ---: | ---: | ---: | --- |
| Pipeline | Intel SPR x4 / us-east-1 | 16 connected sessions | 2 | 11 registered / 32 prepared | `reachy-s2s-pipeline-02` through `-12` registered; through `-33` prepared |
| STT | A10G / us-east-1 | 96 five-second audio equivalents | 1 | 2 | `reachy-s2s-stt-01`, `-02` |
| TTS | A10G / us-east-1 | 8 active generations | 1 | 19 | `reachy-s2s-tts-01` through `-19` |
| LLM | RTX PRO 6000 / us-east-2 | 64 active generations | 1 | 3 | `gemma4-26b-a4b-nvfp4-rtx6000-test`, `reachy-s2s-llm-02`, `-03` |

All endpoints have exactly one HF replica. The CPU LB and speech proxies run
in us-east-1. Gemma retains its tested us-east-2 placement; this is **not** a
same-region LLM proxy/backend pair.

Eight 16-slot pipeline workers provide the 128 connected-session target. The 11
workers currently registered with the isolated test LB expose 176 configured
slots if all are warm, while the normal two-worker warm floor exposes 32. GPU maxima are
`ceil(128 / (target_work * 0.85))`: 2 STT, 19 TTS, and 3 LLM workers. This
allows the configured 85% growth threshold to retain headroom at 128 simultaneous
stage calls, assuming five-second STT audio. A connected user does not constantly
occupy every GPU stage. Long audio, long context, long output, and request bursts
change the actual capacity/latency relationship. These are provisioning targets,
not an end-to-end 128-user latency certification or a cloud capacity reservation.

Four CPU slots was a conservative choice from the earlier 2/4/8-pipeline tests;
eight completed that test but had a worse full-turn tail. The density retest
superseded that sizing recommendation, and the current-image canary subsequently
validated 16 users per worker. Three-sentence TTS batching remains unchanged.

## Images and configuration

- Proxies: `ghcr.io/andimarafioti/s2s-speech-proxy:sha-f303b920f8d6431c1f5fdf85338942074dfa923a`.
- Managed CPU pipelines: `ghcr.io/andimarafioti/s2s-pipeline:sha-4d134f458b8137e615e89a6967a2a8a0036dd2a0`.
- Split LB: `ghcr.io/andimarafioti/s2s-load-balancer:sha-246fdc4673d9d7697a71cac889b8a8f3167ae971`.
- STT/TTS retain their validated `sha-3c6f1d904b95f1a700696b57397d8dc5a82ef244` service images.
- New Gemma replicas pin `vllm/vllm-openai@sha256:383e409fc7695d6e40cd40d452f3ec277a3d1c462d7b1510034768d26f2cd397`, preserving model revision, 128k context, 256 sequences, NVFP4, and MTP.

Each proxy has `SPEECH_AUTOSCALE_ENABLED=true`, the exact inventory above in
`SPEECH_BACKENDS`, `SPEECH_WORKER_MIN_WARM=1`, and its own maximum worker count.
Other lifecycle settings retain the documented defaults: 5-second reconciliation,
30-second growth cooldown, one new worker per growth decision, 600 seconds of
sustained surplus before consolidation, 300-second minimum uptime, and
180-second scale-down cooldown. Busy workers drain before parking.

The split LB currently uses 11 of the 32 prepared managed pipeline names, `COMPUTE_ENDPOINT_MIN_WARM=2`,
`COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS=4`, 5-second reconciliation, 600-second
idle parking, and 180-second parking cooldown. `NUM_PIPELINES=16` is configured
on each managed worker and learned through authenticated health polling.

The LB is public so compute session callbacks can reach it, but session admission
requires a verified HF token (`SESSION_REQUIRE_VERIFIED_HF_TOKEN=true`). Clients
send that token in `X-Reachy-Mini-Authorization`. The 11 LB-managed CPU workers
(`reachy-s2s-pipeline-02` through `-12`) now have public HF ingress so a browser
can open the returned `connect_url` without setting an HF `Authorization` header.
Each worker still requires the LB-issued, HMAC-signed `session_token` in that URL
before it admits a realtime conversation. Existing per-requester rate limits are
retained: 128 total users does not imply 128 parallel sessions permitted for one
identity. The GPU proxies and backends, unregistered CPU workers, and direct
pipeline `-01` retain protected HF ingress.

Public ingress also exposes the CPU workers' health and pool routes. A signed
connection URL is a bearer credential: the current `SESSION_TOKEN_TTL_S=86400`
means someone who obtains that URL could reuse it for up to 24 hours, even after
the original session ends. Token lifetime and replay protection are separate
hardening follow-ups; a session ID alone is not an access credential.

New LB/worker application secrets are shared only within the split fleet. HF
credentials are endpoint secrets, not plain environment configuration. The
rollout uses the existing HF credential for backend ingress and control; a
separately issued least-privilege control credential remains an operational
hardening step. Dashboard persistence uses the existing bucket with the separate
`reachy-s2s-split-lb` prefix.

## Operational boundaries

GPU replicas take roughly 2–4.5 minutes to start based on earlier measurements.
The warm floor covers 32 immediately available CPU slots and one GPU per
stage, not an immediate 128-user burst. TTS grows one GPU every 30 seconds at
most; a sudden jump to peak demand can therefore take several minutes to absorb.
Soft GPU targets continue accepting requests while capacity warms, subject to
the existing upstream/request timeouts. For a scheduled burst, prewarm the fleet
instead of treating paused inventory as ready capacity.

Use each proxy's `/health` and `/metrics` and the split LB's `/dashboard` to
inspect readiness, work, latency, and lifecycle decisions. Metrics on the proxies
reset on restart; pre-rollout snapshots were retained locally. This still has
one LB and one proxy/lifecycle owner per stage, not a highly available control
plane. Do not run a second controller against the same workers.

For rollback, first disable the affected controller and restore an explicit list
of already-running backend URLs. Do not register paused URLs in unmanaged mode,
because health probes can wake scale-to-zero backends. Do not roll a second live
controller over the same inventory while the first is still draining requests.
The original production load balancer remains available and unchanged.

## Manual conversation through the split LB

The packaged `speech-to-speech talk` client rejects URLs with query parameters,
while a managed CPU worker requires the LB-issued `session_token` in its WebSocket
URL. The local bridge supplies that signed URL and also supports protected-worker
HF ingress. Run it in one terminal, using the split LB URL currently reported by
the HF endpoint API:

```bash
uv run --default-index https://pypi.org/simple --with-requirements requirements.txt \
  python scripts/split_talk_bridge.py --lb-url https://YOUR-SPLIT-LB-URL
```

Then use the packaged microphone/speaker client in another terminal:

```bash
speech-to-speech talk \
  --url ws://127.0.0.1:8765/v1/realtime \
  --api-key local \
  --playback-buffer-ms 196
```

The bridge sends `HF_TOKEN` to the LB in `X-Reachy-Mini-Authorization` for
verified session admission. It connects using the signed `connect_url` and also
sends standard `Authorization: Bearer $HF_TOKEN` for compatibility with protected
worker deployments; the current public CPU workers do not need that ingress
header. `local` is used only for the loopback client connection. The bridge
handles one conversation at a time and does not save media or tokens. Stop it
with Ctrl-C after testing.

## Rollout checks

- The new proxy image first served requests with lifecycle disabled, then each
  controller was enabled with its complete inventory. All three health endpoints
  report the expected managed settings and no reconciliation errors.
- An eight-call TTS burst automatically resumed `reachy-s2s-tts-02`; all eight
  streamed requests succeeded, and the new worker passed inference warmup.
- A 56-call LLM burst automatically resumed `reachy-s2s-llm-02`; all 56 streamed
  requests succeeded. This was a short-context lifecycle check, not a context
  capacity or steady-state latency benchmark.
- One conversation, followed by nine synchronized conversations through the new
  LB, completed successfully. The CPU fleet grew from two to four workers (nine
  sessions plus the configured four-slot headroom), then session counts returned
  to zero on both LB and worker health. No manual GPU/CPU resume was used for
  these scale-up checks.
- Nine-turn speech-end to first-audio latency: p50 1.022 s, p95 1.181 s. Stage
  event intervals were STT 203/226 ms, LLM first output batch 600/616 ms, and TTS
  first audio after that batch 218/373 ms (p50/p95). LLM first output batch is
  **not** the same measurement as model first token.
- Ten 50-second synthetic repeated-speech STT uploads completed successfully,
  but the local upload bottleneck spread out backend arrivals. This did not
  cross the production work threshold, so it does not verify deployed STT
  load-triggered scale-up. Its control logic has unit coverage and the standby
  had already passed a remote wake/readiness/transcription/park smoke test.
- A separate five-second all-silence transcription hit the existing 120-second
  proxy timeout while the ASR model kept generating. Ordinary speech succeeded
  afterward and GPU metrics returned to zero running/waiting requests. Do not
  interpret that silence failure as a scaling benchmark; nonspeech generation
  bounds remain a follow-up ASR robustness issue.
- CI is green. A local full-suite run encountered one one-second idle-parking
  test timeout; that test passed on isolated rerun. No runtime source changes
  were made in this rollout, beyond the previously tested lifecycle PR.

The complete 128-conversation workload has not been run. The current 16-slot
configuration can reach it with eight CPU workers within the observed quota, but
full-fleet GPU burst capacity and scale-up latency remain unvalidated. The
prepared inventory, single-worker density tests, warm-floor behavior, and the
scale-up checks above are the validated scope.

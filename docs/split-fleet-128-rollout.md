# Split fleet: 128-session rollout

Deployment date: 2026-09-04. Namespace: `HuggingFaceM4`.

This is a separate live fleet behind `reachy-s2s-split-lb`. The existing
`reachy-s2s-lb`, its monolithic GPU workers, and the directly accessible
`reachy-s2s-pipeline-01` are not migrated. The three existing speech proxies
are updated, so the direct-testing pipeline also benefits from GPU autoscaling.

## Capacity and placement

| Stage | Hardware / region | Per-worker operating target | Warm floor | Maximum workers | Inventory |
| --- | --- | ---: | ---: | ---: | --- |
| Pipeline | Intel SPR x4 / us-east-1 | 4 connected sessions | 2 | 32 | `reachy-s2s-pipeline-02` through `-33` |
| STT | A10G / us-east-1 | 96 five-second audio equivalents | 1 | 2 | `reachy-s2s-stt-01`, `-02` |
| TTS | A10G / us-east-1 | 8 active generations | 1 | 19 | `reachy-s2s-tts-01` through `-19` |
| LLM | RTX PRO 6000 / us-east-2 | 64 active generations | 1 | 3 | `gemma4-26b-a4b-nvfp4-rtx6000-test`, `reachy-s2s-llm-02`, `-03` |

All endpoints have exactly one HF replica. The CPU LB and speech proxies run
in us-east-1. Gemma retains its tested us-east-2 placement; this is **not** a
same-region LLM proxy/backend pair.

The pipeline limit is 32 × 4 = 128 connected sessions. GPU maxima are
`ceil(128 / (target_work * 0.85))`: 2 STT, 19 TTS, and 3 LLM workers. This
allows the configured 85% growth threshold to retain headroom at 128 simultaneous
stage calls, assuming five-second STT audio. A connected user does not constantly
occupy every GPU stage. Long audio, long context, long output, and request bursts
change the actual capacity/latency relationship. These are provisioning targets,
not an end-to-end 128-user latency certification or a cloud capacity reservation.

Four CPU slots is a conservative choice from the earlier 2/4/8-pipeline tests;
eight completed the test but had a worse full-turn tail. Three-sentence TTS
batching remains unchanged.

## Images and configuration

- Proxies: `ghcr.io/andimarafioti/s2s-speech-proxy:sha-f303b920f8d6431c1f5fdf85338942074dfa923a`.
- Managed CPU pipelines: `ghcr.io/andimarafioti/s2s-pipeline:sha-0015309562157e8a5bc031465908f82b9f98d2ad`.
- Split LB: `ghcr.io/andimarafioti/s2s-load-balancer:sha-246fdc4673d9d7697a71cac889b8a8f3167ae971`.
- STT/TTS retain their validated `sha-3c6f1d904b95f1a700696b57397d8dc5a82ef244` service images.
- New Gemma replicas pin `vllm/vllm-openai@sha256:383e409fc7695d6e40cd40d452f3ec277a3d1c462d7b1510034768d26f2cd397`, preserving model revision, 128k context, 256 sequences, NVFP4, and MTP.

Each proxy has `SPEECH_AUTOSCALE_ENABLED=true`, the exact inventory above in
`SPEECH_BACKENDS`, `SPEECH_WORKER_MIN_WARM=1`, and its own maximum worker count.
Other lifecycle settings retain the documented defaults: 5-second reconciliation,
30-second growth cooldown, one new worker per growth decision, 600 seconds of
sustained surplus before consolidation, 300-second minimum uptime, and
180-second scale-down cooldown. Busy workers drain before parking.

The split LB uses the 32 managed pipeline names, `COMPUTE_ENDPOINT_MIN_WARM=2`,
`COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS=4`, 5-second reconciliation, 600-second
idle parking, and 180-second parking cooldown. `NUM_PIPELINES=4` is configured
on each managed worker and learned through authenticated health polling.

The LB is public so compute session callbacks can reach it, but session admission
requires a verified HF token (`SESSION_REQUIRE_VERIFIED_HF_TOKEN=true`). Clients
send that token in `X-Reachy-Mini-Authorization`; protected pipeline ingress uses
standard `Authorization`, while the signed session token is in the returned
connection URL. Existing per-requester rate limits are retained: 128 total users
does not imply 128 parallel sessions permitted for one identity.

New LB/worker application secrets are shared only within the split fleet. HF
credentials are endpoint secrets, not plain environment configuration. The
rollout uses the existing HF credential for backend ingress and control; a
separately issued least-privilege control credential remains an operational
hardening step. Dashboard persistence uses the existing bucket with the separate
`reachy-s2s-split-lb` prefix.

## Operational boundaries

GPU replicas take roughly 2–4.5 minutes to start based on earlier measurements.
The warm floor covers eight immediately available CPU slots and one GPU per
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

The complete 128-conversation workload has not been run. The inventory ceiling,
warm-floor behavior, and the scale-up checks above are the validated scope.

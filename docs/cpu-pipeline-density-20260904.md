# CPU pipeline density retest — 2026-09-04

Four users per CPU worker was a conservative deployment setting, not a measured
limit. Each pipeline still serves one user; `NUM_PIPELINES` determines how many
pipelines share the worker. The retest supports starting with **16 users per
4-vCPU worker**, subject to a longer soak and full-fleet validation. One clean
32-user repeat also passed, but an earlier 32-user run had failures and must not
be discarded or treated as a proven CPU bottleneck.

## Setup

- One Intel SPR x4 worker: 4 vCPUs, 8 GB RAM, AWS us-east-1.
- Actual deployed managed pipeline image, `sha-0015309562157e8a5bc031465908f82b9f98d2ad`.
- Isolated test load balancer with only `reachy-s2s-pipeline-33` registered.
  That worker is not registered with the live split LB. Admission limits were
  raised only on the test LB to permit the synthetic clients from one identity.
- Real deployed STT, Gemma NVFP4 LLM, and TTS through their existing proxies.
  Proxy observations showed one ready backend per stage during sampled periods;
  this was not a controlled GPU capacity benchmark.
- All clients continuously streamed 16-kHz PCM, including microphone silence,
  using real-time pacing and server VAD with interruption enabled. Connections
  were opened serially before streaming to avoid conflating admission storms
  with steady-state CPU density.
- Short fixture: 4.80 seconds including trailing silence; last audible sample
  at approximately 3.573 seconds. Three turns per user, starts staggered by
  0.5 seconds. Turn spacing increased with client count to avoid overlap between
  rounds. Every user stayed connected and streamed audio throughout each round.
- One additional synchronized short-turn burst at 4, 8, and 16 users. TTS
  `STREAM_BATCH_SENTENCES=3` was preserved. Responses requested one short sentence.
- External load generator ran from the local machine, not on the worker.
  This includes WAN variability and is not a controlled same-region network test.

## Short turns: staggered speech

Latency is in seconds, p50 / p95. "Speech-stop event" is the server VAD event as
received by the client. "Source end" uses the last audible PCM sample at the
sender; it additionally includes input transport and turn detection. Neither
includes the client's playback buffer or time to play the whole answer.

| Users on one worker | Completed turns | Speech-stop event → first audio | Source end → first audio | Sampled peak CPU | Peak RAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 12 / 12 | 0.967 / 0.999 | 1.262 / 1.310 | 6.0% | 0.65 GB |
| 8 | 24 / 24 | 0.946 / 1.115 | 1.375 / 2.039 | 30.0% | 0.90 GB |
| 16 | 48 / 48 | 0.953 / 1.014 | 1.264 / 1.925 | 16.2% | 1.40 GB |
| 32, first attempt | 90 / 96 | 0.953 / 4.089 | 1.320 / 13.572 | 72.2% | 2.41 GB |
| 32, repeat | 96 / 96 | 0.995 / 1.055 | 1.298 / 1.366 | 60.4% | 2.40 GB |

Failed/canceled turns are excluded from latency percentiles but included in the
completion denominator. CPU/RAM are sampled HF hardware metrics, not continuous
profiling. Peak CPU includes the cell's admission/streaming interval. In the
clean 32-user repeat, CPU averaged 23.9% over samples after streaming began.
Sample counts are small, particularly at four users; these are not SLA estimates.

Stage event intervals, in milliseconds, p50 / p95:

| Users | Speech stop → STT result | STT result → LLM first output batch | LLM batch → TTS first audio |
| ---: | ---: | ---: | ---: |
| 4 | 174 / 182 | 628 / 653 | 152 / 212 |
| 8 | 176 / 330 | 623 / 724 | 151 / 296 |
| 16 | 176 / 200 | 623 / 720 | 156 / 208 |
| 32, repeat | 175 / 192 | 626 / 636 | 194 / 253 |

The LLM interval measures the first output batch visible to the pipeline/client,
**not model time to first token**. No evidence here supports attributing the
earlier eight-user slowdown specifically to the external LLM or CPU saturation.

## Synchronized short-turn bursts

All 4, 8, and 16 responses completed. Speech-stop event to first audio was
1.027 / 1.173 s, 2.565 / 2.851 s, and 1.419 / 2.459 s, respectively.
The corresponding TTS intervals were 0.195 / 0.343 s, 1.688 / 1.919 s, and
0.693 / 1.623 s. Thus GPU/TTS burst behavior can dominate a full-turn latency
curve even when CPU density is modest. These single bursts are too small and
variable to estimate a monotonic capacity curve. No synchronized 32-user burst
was run after the initial staggered run failed.

## First 32-user run: unresolved anomaly

The first run had two canceled responses and four client timeouts. Successful
turns also experienced substantial delays: maximum source-end to first audio
was 21.0 seconds. VAD/event arrival itself was delayed, with source-end to
speech-stop p95 10.4 seconds; some later response events arrived bunched together.

Worker health RTT p95 rose to 3.62 seconds (maximum 5.16), and a separate proxy
health sampler encountered a connection error. Some proxy-side TTS requests
around the anomaly still showed roughly 100–180 ms first audio. CPU logs show
speculative-turn reopenings and cancellations, rather than an explicit OOM or
worker crash. This is consistent with delayed/bursty input or event delivery,
but does not identify whether the bottleneck was the client network, ingress,
or pipeline scheduling. A clean repeat does not resolve that root cause.

The repeat completed all 96 turns. Its worst source-end to first audio was
1.403 seconds; worker health RTT p95 was 164 ms. Local sender scheduling lag
was at most 65 ms in the failed run and 30 ms in the repeat. That rules out a
large local scheduling pause, but not buffering after a WebSocket send.

## Longer utterances at 16 users

All 48 turns completed across three staggered rounds with a 19.42-second input
fixture (last audible sample at 18.194 seconds). All 16 clients continuously
streamed audio, including silence between utterances. No canceled responses or
client errors were recorded. Peak sampled CPU was 22.2%, with 15.4% average after
streaming began, and peak RAM was 1.52 GB.

- Speech-stop event → first audio: p50 **1.463 s**, p95 **2.448 s**.
- Source audible end → first audio: p50 **1.954 s**, p95 **3.350 s**.
- STT result after speech stop: p50 / p95 **606 / 1,029 ms**.
- LLM first output batch after STT: **225 / 581 ms**.
- TTS first audio after that batch: **533 / 1,302 ms**.
- Returned audio duration: median **9.6 s**, maximum **12.1 s**.

These inputs and generated answers differ from the short fixture, so this is a
robustness check rather than an isolated density comparison. The CPU retained
substantial headroom, but longer-input/output latency and its tail remain
relevant. This was a few-minute test, not a long-conversation memory-leak soak.

## Sizing and retained evidence

Use 16 users per worker as the next conservative deployment target, not a claim
that 16 is the hardware maximum. Eight such workers provide 128 session slots
using 32 fleet vCPUs; adding the existing 15 non-fleet vCPUs gives 47, within the
observed 60-vCPU quota. A quota increase is therefore not intrinsically needed
for 128 users. GPU-stage burst capacity, warmup time, longer conversations, tools,
and full-fleet latency still need their own validation.

The live split LB and its four-slot worker configurations were not changed by
this retest. The test LB and worker are parked between runs; worker configuration
is restored after testing. Benchmark JSON, hardware samples, and downloaded logs
are retained locally under `logs/cpu-density-20260904/`, including the failed run.
The initial matrix has prefix `20260904T153957Z`; the 32-user repeat has prefix
`20260904T155110Z`. Disposable test scripts are not added to the PR.
The 16-user longer-utterance check has prefix `20260904T155422Z`.

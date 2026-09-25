# Direct STT and TTS concurrency benchmark

Date: 2026-08-28

## Setup

- Client: Zurich, Switzerland
- Endpoints: AWS `us-east-1`
- Hardware: one A10G replica per service, with autoscaling fixed at one replica
- STT: `Qwen/Qwen3-ASR-1.7B` through vLLM
- TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` through vLLM-Omni
- TTS output: streaming 24 kHz PCM, voice `aiden`, language `English`
- Main matrix: concurrency 1, 2, 4, and 8 with two waves per cell
- Saturation check: concurrency 16 with one wave per cell
- Load-cell requests: 184 STT and 138 TTS requests

The benchmark generated a speech fixture through the TTS endpoint, repeated or truncated it to exact STT input
durations, and sent the resulting mono WAV audio directly to STT. Each cell captured client-visible latency and
endpoint Prometheus counters before and after the load.

## TTS results

The table reports client-visible p95 time to first audio. Network latency from Zurich is included consistently in
every cell.

| Concurrency | One sentence | Three sentences | Six sentences | Three-sentence audio throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.253 s | 0.208 s | 0.349 s | 2.99 audio-s/s |
| 2 | 0.560 s | 0.590 s | 0.609 s | 5.12 audio-s/s |
| 4 | 0.732 s | 0.734 s | 0.768 s | 8.78 audio-s/s |
| 8 | 2.457 s | 1.591 s | 0.870 s | 13.40 audio-s/s |
| 16 | 4.543 s | 9.323 s | 24.164 s | 13.99 audio-s/s |

All TTS requests succeeded. Throughput increased strongly through concurrency 8, but interactive responsiveness
degraded after concurrency 4. At concurrency 16, throughput was effectively saturated and long-request throughput
fell to 10.24 audio-s/s from 13.72 at concurrency 8.

The three-sentence case most closely represents the pipeline's `stream_batch_sentences=3` configuration. Its p95
time to first audio remained below 0.75 seconds through concurrency 4, reached 1.59 seconds at concurrency 8, and
reached 9.32 seconds at concurrency 16.

### US-hosted TTS confirmation

A follow-up run from GitHub Actions used the same three-sentence, 231-character input. It sent eight waves at
concurrency 1 through 8 and one wave at each higher concurrency. The concurrency-512 cell was stopped before it
completed; the table contains only completed measurements.

| Concurrency | Requests | First-audio p50 | First-audio p95 | Completion p50 | Completion p95 | Audio throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 0.140 s | 0.168 s | 4.552 s | 4.975 s | 3.09 audio-s/s |
| 2 | 16 | 0.179 s | 0.262 s | 5.182 s | 5.724 s | 5.60 audio-s/s |
| 4 | 32 | 0.270 s | 0.444 s | 6.047 s | 6.773 s | 9.37 audio-s/s |
| 8 | 64 | 0.512 s | 0.666 s | 8.044 s | 8.768 s | 14.14 audio-s/s |
| 16 | 16 | 0.803 s | 9.183 s | 9.486 s | 15.883 s | 14.44 audio-s/s |
| 32 | 32 | 9.473 s | 21.558 s | 16.920 s | 27.300 s | 15.24 audio-s/s |
| 64 | 64 | 24.882 s | 46.815 s | 32.115 s | 52.910 s | 16.36 audio-s/s |
| 128 | 128 | 50.467 s | 98.135 s | 57.984 s | 105.416 s | 16.99 audio-s/s |
| 256 | 256 | 101.403 s | 198.368 s | 108.928 s | 205.829 s | 17.08 audio-s/s |

The larger sample agrees with the original run at the important concurrency-16 boundary: first-audio p95 was
9.183 seconds versus 9.323 seconds previously, and completion p95 was 15.883 seconds versus 16.014 seconds.
Throughput was already nearly saturated at concurrency 16. Raising concurrency from 16 to 256 improved audio
throughput by only 18%, while first-audio p95 increased from 9.18 seconds to 198.37 seconds.

## STT results

Client latency for large concurrent uploads was limited by the Zurich-to-`us-east-1` upload path. The table
therefore reports vLLM's server-side mean end-to-end request latency, which excludes that client upload time.

| Concurrency | 2 s audio | 5 s audio | 15 s audio | 30 s audio | Mean model queue time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.099 s | 0.170 s | 0.383 s | 0.510 s | 0.000 s |
| 2 | 0.099 s | 0.160 s | 0.372 s | 0.481 s | 0.000 s |
| 4 | 0.102 s | 0.169 s | 0.378 s | 0.498 s | 0.000 s |
| 8 | 0.105 s | 0.170 s | 0.385 s | 0.503 s | 0.000 s |
| 16 | 0.118 s | 0.175 s | 0.419 s | 0.786 s | 0.000 s |

All STT requests succeeded, and vLLM reported no queued model time at any tested concurrency. Dynamic batching kept
latency essentially flat through concurrency 8. Concurrency 16 remained healthy; only the longest input showed a
material increase, from 0.51 seconds at concurrency 1 to 0.79 seconds.

### Distributed saturation test

The Zurich results established only a lower bound because a single uploader could not keep the endpoint busy. A
second test ran from synchronized US-hosted clients against the same single-A10G endpoint. Each client sent four
waves of five-second WAV inputs. Multiple clients were used above aggregate concurrency 64 so that a single
client's network path would not determine the result.

An additional equal-sample run sent 256 requests at every concurrency from 1 through 128. The complete US-side
latency curve below uses that run through concurrency 64, then the synchronized multi-client results from 128
through 512. Ranges show the lowest and highest p50 or p95 reported by the synchronized clients.

| Aggregate concurrency | Load clients | Client p50 | Client p95 |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 0.186 s | 0.191 s |
| 2 | 1 | 0.202 s | 0.205 s |
| 4 | 1 | 0.205 s | 0.207 s |
| 8 | 1 | 0.211 s | 0.219 s |
| 16 | 1 | 0.236 s | 0.398 s |
| 32 | 1 | 0.275 s | 0.492 s |
| 64 | 1 | 0.385 s | 0.705 s |
| 128 | 2 | 0.513-0.542 s | 0.949-1.435 s |
| 256 | 4 | 0.555-0.685 s | 3.166-4.019 s |
| 512 | 4 | 0.771-0.857 s | 6.521-8.259 s |

| Aggregate concurrency | Clients | Requests | Aggregate throughput | Worst client p95 | Peak vLLM running | Peak vLLM waiting |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 256 | 93.3 req/s | 1.37 s | 41 | 0 |
| 128 | 2 | 512 | 172.6 req/s | 1.44 s | 63 | 0 |
| 256 | 4 | 1,024 | 194.4 req/s | 4.02 s | 64 | 0 |
| 512 | 4 | 2,048 | 187.4 req/s | 8.26 s | 109 | 0 |

All 3,840 load requests succeeded. Throughput nearly doubled from concurrency 64 to 128, increased only 13% from
128 to 256, and then declined slightly at 512. At the same time, worst-client p95 increased from 1.44 seconds at
128 to 4.02 seconds at 256 and 8.26 seconds at 512. The practical latency knee is therefore between aggregate
concurrency 128 and 256, and peak throughput for this five-second fixture is about 190 requests per second.

vLLM's mean end-to-end latency rose only from 0.24 seconds at concurrency 64 to 0.37 seconds at concurrency 512,
and its reported waiting queue remained zero. The latency above the knee is consequently outside the model
scheduler, in request upload, parsing, preprocessing, or endpoint-front-door admission. Routing and autoscaling
should use total outstanding STT calls and observed request latency rather than `vllm:num_requests_waiting` alone.

## Initial capacity recommendation

- Treat four concurrent TTS generations as the interactive capacity of one A10G.
- Start another warm TTS replica when the current replica reaches three concurrent generations.
- Allow up to eight TTS generations only as short overload headroom; do not plan around concurrency 16.
- Use 128 concurrent five-second STT calls as the measured interactive ceiling of one A10G, not as its steady-state
  target.
- Start another warm STT replica around 80-96 outstanding calls, retaining 25-38% headroom before the measured
  latency knee. Cap new admissions or spill to another replica at 128.
- Do not plan around concurrency 256: it adds only 13% throughput while almost tripling worst-client p95 latency.
- Re-run the distributed test with representative production utterances and a longer steady-state soak before
  making these values hard limits.

Both endpoints returned health 200 after the test. STT recorded zero aborts and zero errors, and TTS recorded only
2xx speech responses.

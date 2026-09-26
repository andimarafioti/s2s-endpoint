# LLM model comparison on one RTX PRO 6000

This document preserves the LLM evaluation results collected during the August
31–September 2, 2026 deployment exploration. The decision from these tests was
to use `nvidia/Gemma-4-26B-A4B-NVFP4` as the default self-hosted voice model.
Dense `nvidia/Gemma-4-31B-IT-NVFP4` remains useful as a subjective comparison,
but did not beat A4B on the targeted quality checks and served roughly half as
many requests per second at the highest tested concurrency.

These are exploratory deployment measurements, not a checked-in reproducible
benchmark suite. The one-off harnesses were intentionally not retained. The
tables below preserve the reported results and the important differences in
test shape so later decisions do not rely on memory alone.

## Test shape and caveats

- All self-hosted models ran on one AWS RTX PRO 6000 Blackwell in `us-east-2`.
  The price observed during testing was $2.75/hour, or $66/day continuously.
- Most latency curves were client-visible measurements from Zurich to the Ohio
  endpoint. They therefore include the public request path, but not STT or TTS.
- The realistic voice prompt was approximately 357–368 input tokens and asked
  for a short spoken reply, generally around 17–27 output tokens.
- PhoneLLM's first BF16/NVFP4 comparison used a shorter 98-token prompt. Those
  points are retained separately and must not be compared directly with the
  realistic-prompt curves.
- Reported concurrency is simultaneous LLM requests, not connected voice
  sessions. A connected user spends substantial time listening, speaking, in
  STT, or in TTS without occupying the LLM.
- The final Gemma NVFP4 and Qwen deployments used a 128K context target and a
  256-active-sequence scheduler. Earlier PhoneLLM and Gemma BF16 runs used
  different context/scheduler settings where noted.
- Warmup waves were excluded from the measured concurrency samples. All rows
  below completed without request failures unless explicitly stated.
- OpenAI numbers came from natural production turns, not the controlled prompt,
  and are included only as an operational baseline.

## Final shortlist

| Model | Architecture | Runtime memory | Vision | Tested context / scheduler | Result |
| --- | --- | ---: | --- | --- | --- |
| Gemma 4 26B-A4B NVFP4 | 25.2B total / 3.8B active MoE | 18.94 GiB with MTP | Yes | 128K / 256 sequences | Selected default |
| Gemma 4 31B NVFP4 | Dense 31B | 32.09 GiB with MTP | Yes | 128K / 256 sequences | Similar quality, about half A4B throughput |
| Qwen3.6-35B-A3B BF16 | 35B total / about 3B active MoE | 67.13 GiB with MTP | Yes | 128K / 256 sequences | Best cold long-context alternative; weaker scaling and default tool selection |
| Qwen3.8-27B-FP8 | Dense 27B | 29.0 GiB with MTP | Yes | 128K / 256 sequences | Strong tools/vision; saturated early |
| PhoneLLM NVFP4 | About 30B total / 3.5B active MoE | 19.61 GiB | No | Exercised through 8K; 64 active sequences in high-concurrency run | Voice-tuned and compact; weaker scaling than Gemma |
| PhoneLLM BF16 | About 30B total / 3.5B active MoE | 59.81 GiB | No | Earlier short-prompt run | Superseded by NVFP4 |
| Gemma 4 26B-A4B BF16 | 25.2B total / 3.8B active MoE | 49.3 GiB with MTP | Yes | 32K in initial evaluation | Established the tool/vision result; superseded by NVFP4 |
| Gemma 4 31B BF16 | Dense 31B | About 59.9 GiB with MTP | Yes | 32K in initial evaluation | Viable but displaced by A4B |

## Voice-prompt time to first token

Each cell is TTFT p50 / p95. Values are milliseconds unless marked as seconds.
A dash means that exact checkpoint/configuration was not measured at that cell.

| Concurrency | Gemma A4B NVFP4 | Gemma 31B NVFP4 | Qwen3.6 A3B | Qwen3.8 FP8 | PhoneLLM NVFP4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **204 / 225** | 215 / 225 | 242 / 248 | 245 / 260 | 348 / 495 |
| 8 | **230 / 249** | 267 / 291 | 387 / 687 | 408 / 503 | 398 / 417 |
| 16 | — | — | **445 / 524** | 589 / 770 | 582 / 725 |
| 32 | **318 / 353** | 396 / 436 | 535 / 639 | 1,130 / 1,398 | 698 / 891 |
| 64 | **376 / 424** | 496 / 717 | 764 / 1,033 | 1,685 / 2,939 | 779 / 1,083 |
| 128 | **510 / 835** | 824 / 1,714 | 1,255 / 2,073 | 3,038 / 5,654 | 645 / 2,291 |
| 256 | **1,172 / 1,571** | 2,061 / 2,939 | 3,237 / 3,928 | 8,534 / 10,538 | 2,603 / 5,736 |

The PhoneLLM 128/256 points came from a later run with only 64 sequences
actively scheduled; excess requests queued inside vLLM. The lower C128 median
than C64 is run-to-run variance, not an expected monotonic improvement.

## Completion latency

### Final Gemma NVFP4 comparison

Each cell is p50 / p95 in milliseconds. Both models used MTP.

| Concurrency | A4B TTFT | A4B total | Dense 31B TTFT | Dense 31B total |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 204 / 225 | **281 / 314** | 215 / 225 | 482 / 544 |
| 8 | 230 / 249 | **329 / 377** | 267 / 291 | 532 / 652 |
| 32 | 318 / 353 | **416 / 495** | 396 / 436 | 700 / 757 |
| 64 | 376 / 424 | **500 / 547** | 496 / 717 | 904 / 1,091 |
| 128 | 510 / 835 | **679 / 1,080** | 824 / 1,714 | 1,307 / 2,440 |
| 256 | 1,172 / 1,571 | **1,344 / 1,871** | 2,061 / 2,939 | 2,507 / 3,616 |

At concurrency 256, A4B delivered 161 requests/s and 2,886 output tokens/s.
Dense 31B delivered 79 requests/s and 1,477 output tokens/s. Neither returned a
request failure.

### Qwen comparison

Each cell is completion p50 / p95 in milliseconds.

| Concurrency | Qwen3.6-35B-A3B | Qwen3.8-27B-FP8 |
| ---: | ---: | ---: |
| 1 | **312 / 362** | 482 / 550 |
| 8 | **539 / 804** | 764 / 968 |
| 16 | **624 / 746** | 1,029 / 1,307 |
| 32 | **810 / 923** | 1,599 / 2,404 |
| 64 | **1,148 / 1,453** | 3,095 / 5,310 |
| 128 | **1,864 / 2,907** | 5,134 / 8,510 |
| 256 | **3,796 / 4,702** | 10,565 / 13,371 |

Qwen3.6 reached 60.0 requests/s at C256. Qwen3.8 saturated around 22
requests/s and became unattractive for realtime voice above roughly 8–16
simultaneous generations.

### PhoneLLM realistic-prompt tail

The approximately 357-token NVFP4 run retained full-response p95 rather than a
complete p50/p95 curve.

| Concurrency | TTFT p50 | TTFT p95 | Full response p95 |
| ---: | ---: | ---: | ---: |
| 1 | 348 ms | 495 ms | 511 ms |
| 2 | 415 ms | 601 ms | 606 ms |
| 4 | 407 ms | 425 ms | 596 ms |
| 8 | 398 ms | 417 ms | 573 ms |
| 16 | 582 ms | 725 ms | 1.13 s |
| 32 | 698 ms | 891 ms | 1.83 s |
| 64 | 779 ms | 1.08 s | 2.53 s |
| 128 | 645 ms | 2.29 s | 2.74 s |
| 256 | 2.60 s | 5.74 s | 6.42 s |

The C128 and C256 rows used 256 measured requests and 512 measured requests,
respectively. Every request completed.

## Tool, vision, compaction, and recall checks

| Model | Tool selection | Multi-step tools | Streamed calls | Vision tools | Compaction | Long-context recall |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gemma A4B NVFP4 | 35/35 | 25/25 | 20/20 | 20/20 | 5/5 | Passed every tested length |
| Gemma 31B NVFP4 | 35/35 | 25/25 | 20/20 | 20/20 | 5/5 | Passed every tested length |
| Qwen3.6 A3B | 32/35 | 25/25 | 20/20 | 20/20 | 5/5 | Passed every tested length |
| Qwen3.8 FP8 | Effectively 20/20 | Included parallel and multi-step cases | Passed | 20/20 | 5/5 | Passed every tested length |
| PhoneLLM | Basic pipeline integration passed | Not exhaustively tested | Streaming passed | Not supported | Not tested | Exercised through 8K |

Qwen3.6's three tool-selection failures were meaningful: with the normal
prompt it sometimes skipped the weather tool and invented a current
temperature. An explicit instruction to always use tools for current/external
information produced 20/20, but Gemma passed the original weaker prompt.

Gemma's earlier BF16 A4B evaluation also covered enums, integers, arrays, dates,
nested arguments, missing-information clarification, tool errors, a basic tool
result prompt-injection attempt, and multi-service weather/contact/email and
availability/calendar chains. Its native `auto` behavior was strong. The tested
vLLM/Gemma combination had caveats with generic `tool_choice="none"` and
`tool_choice="required"`; omitting tools for `none` and using named tool choice
were reliable workarounds.

Compaction latency was approximately:

| Model | Compaction p50 / p95 |
| --- | ---: |
| Gemma A4B | 1.02 s p50; p95 not retained |
| Qwen3.6 A3B | 1.09 / 1.20 s |
| Qwen3.8 FP8 | 1.76 / 1.85 s |

## Long-context cold TTFT

These requests used unique or cold prefixes and verified correct recall. Values
are seconds. Prefix-cached repeats can be much faster and are discussed below.

| Context | Gemma A4B NVFP4 | Gemma 31B NVFP4 | Qwen3.6 A3B | Qwen3.8 FP8 |
| ---: | ---: | ---: | ---: | ---: |
| 32K | 1.88 | 9.04 | **1.60** | 4.43 |
| 64K | 3.60 | 21.12 | **3.50** | 10.41 |
| 96K | 6.79 | 37.68 | **6.00** | 18.40 |
| 120K–131K | 11.42 near 131K | 60.43 near 131K | **8.30 at 120K** | 25.73 at 120K |

With prefix caching, both Gemma NVFP4 models returned in roughly 0.34–0.63
seconds even near the configured context limit. PhoneLLM's earlier C1 p95 was
349 ms at 389 tokens, 360 ms at 1,040, 465 ms at 4,040, and 835 ms at 8,040.

## KV-cache headroom

| Metric | Gemma A4B NVFP4 | Gemma 31B NVFP4 | Qwen3.6 A3B | PhoneLLM NVFP4 |
| --- | ---: | ---: | ---: | ---: |
| Model/runtime memory with MTP where applicable | 18.94 GiB | 32.09 GiB | 67.13 GiB | 19.61 GiB |
| KV capacity | 2.85M tokens | 544K tokens | 624K tokens | 64.19 GiB KV memory |
| Equivalent full 128K histories | 21.7 | 4.15 | About 4.8 | Not measured at 128K |

PhoneLLM NVFP4's 64.19 GiB KV allocation had a theoretical capacity of 223
simultaneous 32K sequences. That was a memory calculation, not a latency-safe
operating target. Its BF16 checkpoint left only 23.43 GiB for KV, or a
theoretical 81 simultaneous 32K sequences.

## MTP speculative decoding

| Model | Draft depth | Accepted draft tokens | Operational result |
| --- | ---: | ---: | --- |
| Gemma A4B NVFP4 | Model assistant | 76.6% | Modest gain for short A4B answers |
| Gemma 31B NVFP4 | Model assistant | 80.3% | Materially improved dense decoding |
| Qwen3.6 A3B | 2 tokens | 83.2% | Best default for short, compacted voice conversations |
| Qwen3.8 FP8 | 3 tokens | 60.3% | Helped through C32; became slower at C64 |

Qwen3.8 completion p50 with and without MTP:

| Concurrency | MTP | No MTP | Difference |
| ---: | ---: | ---: | ---: |
| 1 | **482 ms** | 680 ms | 29% faster |
| 8 | **764 ms** | 886 ms | 14% faster |
| 16 | **1,029 ms** | 1,134 ms | 9% faster |
| 32 | **1,599 ms** | 1,653 ms | 3% faster |
| 64 | 3,095 ms | **2,783 ms** | MTP slower |

For Qwen's hybrid architecture, vLLM MTP interfered with reusable cross-request
prefix caching. A repeated 32K Qwen3.8 prompt took about 4.35 seconds with MTP
and 489 ms without it. For Qwen3.6 the corresponding comparison was about 1.58
seconds with MTP and 386 ms without it. In a warm 24-turn short conversation,
however, Qwen3.6 MTP produced 242 ms TTFT / 294 ms completion p50 versus 236 / 309
ms without MTP. With 30-turn conversation compaction and short spoken answers,
MTP remained the chosen voice default.

These tests used vLLM's model-native MTP, not llama.cpp's draft-MTP
implementation, and did not sweep speculative depth beyond the model recipes.

## Earlier checkpoint results

### PhoneLLM BF16 versus NVFP4, short 98-token prompt

Each latency cell is TTFT p50 / p95. Throughput is aggregate output tokens/s.

| Concurrency | BF16 TTFT | NVFP4 TTFT | BF16 / NVFP4 throughput |
| ---: | ---: | ---: | ---: |
| 1 | 246 / 300 ms | 244 / 280 ms | 73 / 88 |
| 2 | 243 / 359 ms | 264 / 303 ms | 132 / 187 |
| 4 | 308 / 363 ms | 292 / 380 ms | 219 / 312 |
| 8 | 353 / 406 ms | 395 / 481 ms | 370 / 448 |
| 16 | 505 / 516 ms | 379 / 497 ms | 577 / 567 |
| 32 | 502 / 556 ms | 479 / 546 ms | 629 / 723 |
| 64 | 700 / 1,003 ms | 702 / 955 ms | 815 / 864 |

All 960 measured requests succeeded. NVFP4 did not materially improve low-load
TTFT, but improved high-concurrency throughput by roughly 6–15% while reducing
weights/runtime from 59.81 GiB to 19.61 GiB.

### Initial Gemma 4 31B BF16 voice curve

The initial dense model used a roughly 360-token prompt and 32K context.

| Concurrency | TTFT p50 / p95 | Completion p50 / p95 |
| ---: | ---: | ---: |
| 1 | 329 / 360 ms | 904 ms / 1.49 s |
| 2 | 358 / 496 ms | 927 ms / 1.47 s |
| 4 | 427 / 552 ms | 993 ms / 1.49 s |
| 8 | 563 / 615 ms | 1.12 / 1.60 s |
| 16 | 566 / 704 ms | 1.20 / 1.78 s |
| 32 | 777 ms / 1.12 s | 1.93 / 2.32 s |

The official 0.5B MTP assistant improved a fixed 46-token response from 2.26 to
1.44 seconds, but required a recent vLLM nightly. The MTP endpoint needed about
five minutes to start versus roughly 2.5 minutes without MTP.

### Initial Gemma 4 A4B BF16 result

The BF16 A4B checkpoint established the model's strong tool behavior before the
NVFP4 checkpoint was tested:

- Tool selection and schema cases: 35/35.
- Multi-step tool loops: 25/25.
- Streamed/parallel tool calls: 20/20.
- Tool calls under concurrency 1–64: 256/256 correct.
- Vision-conditioned calls: 20/20.
- Actual Realtime tool decision: 550 ms; tool-result follow-up: 444 ms.

Regular voice-prompt medians were:

| Concurrency | TTFT p50 | Completion p50 |
| ---: | ---: | ---: |
| 1 | 219 ms | 376 ms |
| 8 | 271 ms | 567 ms |
| 16 | 283 ms | 594 ms |
| 32 | 314 ms | 687 ms |
| 64 | 437 ms | 1.07 s |

## End-to-end pipeline samples

These samples include STT and TTS and were affected by where the CPU pipeline
ran. They validate integration, but should not be used as pure model rankings.

| Model/configuration | Turns | STT median | LLM median | TTS first-audio median | Speech stop to first audio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 31B BF16 | 5 | 1.21 s | 859 ms | 771 ms | 2.93 s |
| Qwen3.8 FP8 | 5 | 837 ms | 521 ms complete short reply | 793 ms | 2.16 s |
| Gemma A4B NVFP4 | 5 | Not separately retained | 442 ms | Not separately retained | 2.11 s |
| Gemma 31B NVFP4 | 5 | Not separately retained | 507 ms | Not separately retained | 2.17 s |

The Zurich-hosted integration runs inflated STT/TTS network time. For example,
Qwen3.8's TTS proxy/backend median was about 432 ms with roughly 6 ms typical
proxy overhead; approximately 360 ms came from the Zurich-to-HF public path.

## OpenAI operational baseline

The available `gpt-5.6-terra` production sample contained 29 natural turns:

- First speakable batch p50: 1.374 seconds.
- First speakable batch p95: 2.062 seconds.
- Maximum: 3.161 seconds.

This was not an identical-prompt benchmark and does not establish quality or
cost equivalence. It only showed that the self-hosted candidates could improve
the latency observed in that production sample.

## Decision

Gemma 4 26B-A4B NVFP4 was selected because it combined the best measured
latency and concurrency with vision, perfect results in the targeted tool suite,
successful compaction, and much more KV headroom than dense Gemma. The intended
operating target is about 64 active generations per GPU, with another worker
warmed as the fleet approaches that level or the fresh TTFT signal exceeds the
approximately 500 ms target.

Qwen3.6 remains the most interesting alternative for cold long-context and
agentic workloads. Dense Gemma 31B remains a useful subjective quality
comparison. PhoneLLM remains interesting for specialized voice behavior, but is
text-only and was not subjected to the same exhaustive tool suite.

As of September 24, 2026 the split test deployment is temporarily serving dense
Gemma 4 31B NVFP4 so it can be compared by conversation. That temporary switch
does not change the benchmark recommendation above.

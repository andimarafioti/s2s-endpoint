# BF16 GGML TTS canary listening test — 2026-09-26

The standalone `reachy-s2s-tts-ggml-canary` endpoint is outside the TTS proxy's
managed backend inventory. It remains private and paused after this test. The
experimental split stack still uses `reachy-s2s-tts-01` through
`reachy-s2s-tts-proxy`; neither was changed.

## Exact configurations

- The live vLLM-Omni 0.28.0 TTS endpoint uses official
  `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` BF16 safetensors at revision
  `0c0e3051f131929182e2c023b9537f8b1c68adfe` on one A10G.
- The canary image `ghcr.io/andimarafioti/s2s-tts-ggml:sha-7ada89b6b153ca4984bc7bee03a8eb3f3d04a0df`
  uses qwentts-cpp-python 0.4.1+cu124 and BF16 GGUF talker plus BF16 codec
  from `Serveurperso/Qwen3-TTS-GGUF` revision
  `b7ee2e8c7459c3bea99da23e3d178125a7d1713c`. It ran on one A10G only
  while collecting samples, then was paused. The image build is
  [GitHub Actions run 36272252734](https://github.com/andimarafioti/s2s-endpoint/actions/runs/36272252734).
- Local GGML comparison uses the same pinned BF16 GGUFs on Apple M3 Pro Metal.
  Local Q8_0 talker/codec and Q4_K_M talker + Q8_0 codec are listening baselines.
  Q4_K_M cannot run on this CUDA wheel because a q6_K embedding reaches an
  unsupported `get_rows` CUDA path; keep it local until that is fixed.

Four synthetic prompts (two German, two French) were synthesized with Aiden,
24 kHz mono, the whole text supplied per request, streaming PCM output, and
requested seeds 42–45. Explicit language and `Auto` variants were collected.
The local listening files and per-clip manifests are at
`logs/tts-bf16-listening-20260926/` in this workspace (ignored by Git).

## Checks and limits

- Four hosted BF16 GGML explicit-language requests and four `Auto` requests
  completed. This confirms the CUDA image works; pronunciation needs a human
  listening judgement.
- Repeating one hosted GGML BF16 request with the same seed produced identical
  PCM bytes. Repeating all four local GGML BF16 requests did the same.
- The live vLLM endpoint produced different audio on three identical German
  requests with seed 42: 5.68, 5.36, and 4.32 seconds. The
  [vLLM-Omni 0.28.0 talker source](https://github.com/vllm-project/vllm-omni/blob/v0.28.0/vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py)
  notes that per-request MTP seeds are not reproducible under full CUDA graphs.
  Compare several vLLM clips before judging a systematic accent difference.
- Changing local GGML BF16 output chunk size from 12 to 8 codec frames yielded
  byte-identical final audio for all four fixed-seed prompts. Audio chunk size
  did not explain these pronunciation differences.
- The experimental STT backend transcribed the first German BF16 GGML clip and
  three vLLM runs correctly. Transcription checks word loss, not accent.
- The split CPU pipeline defaults to `TTS_LANGUAGE=Auto` when Qwen3-ASR does not
  supply language metadata, and it batches up to three completed sentences per
  TTS request. Its default Aiden preset is English-native. The
  [Qwen model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
  recommends each preset's native language for best quality and lists no
  German- or French-native CustomVoice preset.

This aligns BF16 precision and A10G hardware for a closer comparison, while
runtime, model conversion, codec implementation, and stochastic generation
still differ. The live TTS backend should stay unchanged until the clips have
been judged by ear.

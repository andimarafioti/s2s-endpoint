#!/usr/bin/env python3
"""Build an ephemeral speech-to-speech config and exec the Realtime server."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

DEFAULT_STT_BASE_URL = "https://go3quisjv5ta7203.us-east-1.aws.endpoints.huggingface.cloud/v1"
DEFAULT_TTS_BASE_URL = "https://db6lx9j3kdymwu9w.us-east-1.aws.endpoints.huggingface.cloud/v1"
DEFAULT_SMART_TURN_MODEL_PATH = "/opt/models/smart-turn-v3.2-cpu.onnx"


def _required(environ: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = environ.get(name, "").strip()
        if value:
            return value
    joined = " or ".join(names)
    raise ValueError(f"Missing required secret: {joined}")


def _bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name, str(default)).strip()
    value = int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def build_config(environ: Mapping[str, str]) -> dict[str, object]:
    hf_token = _required(environ, "HF_TOKEN")
    openai_key = _required(environ, "RESPONSES_API_API_KEY", "OPENAI_API_KEY")
    config: dict[str, object] = {
        "host": "0.0.0.0",
        "port": int(environ.get("PORT", "7860")),
        "device": "cpu",
        "num_pipelines": _positive_int(environ, "NUM_PIPELINES", 1),
        "log_transcripts": _bool(environ, "LOG_TRANSCRIPTS", False),
        "enable_live_transcription": _bool(environ, "ENABLE_LIVE_TRANSCRIPTION", False),
        "smart_turn_model_path": environ.get("SMART_TURN_MODEL_PATH", DEFAULT_SMART_TURN_MODEL_PATH),
        "stt": "openai",
        "openai_stt_base_url": environ.get("STT_BASE_URL", DEFAULT_STT_BASE_URL),
        "openai_stt_api_key": hf_token,
        "openai_stt_model": environ.get("STT_MODEL", "Qwen/Qwen3-ASR-1.7B"),
        "openai_stt_response_format": "json",
        "llm_backend": "responses-api",
        "model_name": environ.get("MODEL_NAME", "gpt-5.6-terra"),
        "responses_api_api_key": openai_key,
        "responses_api_stream": True,
        "responses_api_reasoning_effort": environ.get("RESPONSES_API_REASONING_EFFORT", "none"),
        "stream_batch_sentences": _positive_int(environ, "STREAM_BATCH_SENTENCES", 3),
        "tts": "openai",
        "openai_tts_base_url": environ.get("TTS_BASE_URL", DEFAULT_TTS_BASE_URL),
        "openai_tts_api_key": hf_token,
        "openai_tts_model": environ.get(
            "TTS_MODEL",
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ),
        "openai_tts_voice": environ.get("TTS_VOICE", "aiden"),
        "openai_tts_language": environ.get("TTS_LANGUAGE", "English"),
        "openai_tts_response_format": "pcm",
        "openai_tts_sample_rate": 24000,
        "openai_tts_stream": True,
    }
    init_chat_prompt = environ.get("INIT_CHAT_PROMPT", "").strip()
    if init_chat_prompt:
        config["init_chat_prompt"] = init_chat_prompt
    return config


def write_private_config(config: dict[str, object]) -> Path:
    descriptor, path = tempfile.mkstemp(prefix="speech-to-speech-", suffix=".json")
    config_path = Path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(config, file)
    except BaseException:
        config_path.unlink(missing_ok=True)
        raise
    config_path.chmod(0o600)
    return config_path


def main() -> None:
    config_path = write_private_config(build_config(os.environ))
    os.execvp(
        "speech-to-speech",
        ["speech-to-speech", "serve", str(config_path)],
    )


if __name__ == "__main__":
    main()

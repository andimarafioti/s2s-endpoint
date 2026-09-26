"""Small OpenAI-compatible Qwen3-TTS server for a protected GGML canary."""

from __future__ import annotations

import io
import os
import wave
from contextlib import asynccontextmanager
from typing import Callable, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SAMPLE_RATE = 24000
LANGUAGES = {
    "auto", "chinese", "english", "japanese", "korean", "german",
    "french", "russian", "portuguese", "spanish", "italian",
}


class SpeechRequest(BaseModel):
    model: str = MODEL
    input: str = Field(min_length=1, max_length=2000)
    voice: str = "aiden"
    language: str = "Auto"
    response_format: str = "wav"
    stream: bool = False
    stream_format: str | None = None
    seed: int = -1


def _pcm(audio: np.ndarray) -> bytes:
    return np.clip(np.asarray(audio, dtype=np.float32) * 32768, -32768, 32767).astype("<i2").tobytes()


def _wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm)
    return output.getvalue()


def _load_model():
    from qwentts_cpp import QwenTTS

    return QwenTTS(
        talker_path=os.environ.get("GGML_TALKER_PATH", "/opt/models/qwen-talker-1.7b-customvoice-Q4_K_M.gguf"),
        codec_path=os.environ.get("GGML_CODEC_PATH", "/opt/models/qwen-tokenizer-12hz-Q8_0.gguf"),
    )


def create_app(model_factory: Callable = _load_model) -> FastAPI:
    model = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal model
        model = model_factory()
        try:
            yield
        finally:
            model.close()
            model = None

    app = FastAPI(title="Qwen3-TTS GGML canary", lifespan=lifespan)

    @app.get("/health")
    def health():
        if model is None:
            raise HTTPException(status_code=503, detail="model loading")
        return {"status": "ok", "model_loaded": True, "backend": "ggml", "model": MODEL}

    @app.post("/v1/audio/speech")
    def speech(request: SpeechRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="model loading")
        if request.model != MODEL:
            raise HTTPException(status_code=400, detail="unsupported model")
        if not request.input.strip():
            raise HTTPException(status_code=400, detail="input is empty")
        if request.language.casefold() not in LANGUAGES:
            raise HTTPException(status_code=400, detail="unsupported language")
        if request.response_format not in {"pcm", "wav"}:
            raise HTTPException(status_code=400, detail="unsupported response format")
        if request.stream_format not in {None, "audio"}:
            raise HTTPException(status_code=400, detail="unsupported stream format")
        speakers = {name.casefold(): name for name in model.speaker_names()}
        voice = speakers.get(request.voice.casefold())
        if voice is None:
            raise HTTPException(status_code=400, detail="unsupported voice")
        kwargs = {
            "text": request.input,
            "lang": request.language,
            "speaker": voice,
            "seed": request.seed,
        }
        if request.stream:
            def pcm_chunks() -> Iterator[bytes]:
                source = model.stream(**kwargs)
                try:
                    for audio, sample_rate in source:
                        if sample_rate != SAMPLE_RATE:
                            raise RuntimeError("unexpected GGML sample rate")
                        yield _pcm(audio)
                finally:
                    source.close()

            if request.response_format == "pcm":
                return StreamingResponse(pcm_chunks(), media_type="audio/pcm")
            raise HTTPException(status_code=400, detail="streaming WAV is not supported")

        audio, sample_rate = model.synthesize(**kwargs)
        if sample_rate != SAMPLE_RATE:
            raise RuntimeError("unexpected GGML sample rate")
        pcm = _pcm(audio)
        if request.response_format == "pcm":
            return Response(content=pcm, media_type="audio/pcm")
        return Response(content=_wav(pcm), media_type="audio/wav")

    return app


app = create_app()

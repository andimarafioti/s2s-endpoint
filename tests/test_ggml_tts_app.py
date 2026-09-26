from __future__ import annotations

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.ggml_tts_app import MODEL, create_app


class FakeGGML:
    def __init__(self):
        self.calls = []
        self.closed = False

    def speaker_names(self):
        return ["Aiden"]

    def stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        yield np.array([0.0, 0.5, -0.5], dtype=np.float32), 24000

    def synthesize(self, **kwargs):
        self.calls.append(("synthesize", kwargs))
        return np.array([0.0, 0.5, -0.5], dtype=np.float32), 24000

    def close(self):
        self.closed = True


def test_proxy_stream_and_wav_contract():
    fake = FakeGGML()
    with TestClient(create_app(lambda: fake)) as client:
        assert client.get("/health").json()["backend"] == "ggml"
        request = {
            "model": MODEL,
            "input": "Guten Morgen",
            "voice": "aiden",
            "language": "German",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
        }
        response = client.post("/v1/audio/speech", json=request)
        assert response.status_code == 200
        assert len(response.content) == 6
        assert fake.calls[0][1]["lang"] == "German"
        assert fake.calls[0][1]["speaker"] == "Aiden"

        request.update(stream=False, response_format="wav")
        response = client.post("/v1/audio/speech", json=request)
        assert response.status_code == 200
        with wave.open(io.BytesIO(response.content)) as output:
            assert output.getframerate() == 24000
            assert output.getnframes() == 3

        request["language"] = "Klingon"
        assert client.post("/v1/audio/speech", json=request).status_code == 400
    assert fake.closed

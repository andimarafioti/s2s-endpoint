import asyncio
import sys
import wave
import websockets


CHUNK_SAMPLES = 512  # matches the old endpoint handler chunking pattern nicely
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1


def read_wav_pcm16_mono(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        sw = wf.getsampwidth()
        ch = wf.getnchannels()

        if sr != SAMPLE_RATE or sw != SAMPLE_WIDTH or ch != CHANNELS:
            raise ValueError(
                f"Expected WAV mono/16kHz/16-bit PCM, got sr={sr}, sw={sw}, ch={ch}"
            )

        return wf.readframes(wf.getnframes())


def write_wav_pcm16_mono(path: str, pcm_bytes: bytes) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


async def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python test_ws_file.py <ws_url> <input.wav> [hf_token]")
        print("Example:")
        print("  python test_ws_file.py ws://localhost:7860/ws input.wav")
        sys.exit(1)

    ws_url = sys.argv[1]
    input_wav = sys.argv[2]
    hf_token = sys.argv[3] if len(sys.argv) > 3 else None

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    audio = read_wav_pcm16_mono(input_wav)
    bytes_per_chunk = CHUNK_SAMPLES * SAMPLE_WIDTH

    received = bytearray()

    async with websockets.connect(
        ws_url,
        additional_headers=headers if headers else None,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        # sender
        for i in range(0, len(audio), bytes_per_chunk):
            await ws.send(audio[i : i + bytes_per_chunk])
            await asyncio.sleep(CHUNK_SAMPLES / SAMPLE_RATE)

        # Give the server some time to answer
        # For a real app you'd use a smarter turn-ending signal or UI behavior.
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=8.0)
                if isinstance(msg, bytes):
                    received.extend(msg)
                else:
                    print("TEXT EVENT:", msg)
        except asyncio.TimeoutError:
            pass

    write_wav_pcm16_mono("response.wav", bytes(received))
    print("Wrote response.wav")


if __name__ == "__main__":
    asyncio.run(main())

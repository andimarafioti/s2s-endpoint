from __future__ import annotations

import argparse
import asyncio
import base64
import json
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import websockets
except ImportError:  # pragma: no cover - handled at runtime for local test environments
    websockets = None

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled at runtime for local test environments
    sd = None


SAMPLE_WIDTH = 2


@dataclass
class ListenAndPlayWSArguments:
    ws_url: str = "ws://127.0.0.1:7860/v1/realtime"
    session_url: Optional[str] = None
    send_rate: int = 16000
    recv_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    authorization: Optional[str] = None
    save_output: Optional[str] = None
    allow_barge_in: bool = False
    instructions: Optional[str] = None
    print_json: bool = False
    list_devices: bool = False


@dataclass
class AllocatedSession:
    session_id: str
    websocket_url: str
    connect_url: str


class PlaybackBuffer:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        with self._lock:
            self._buffer.extend(data)

    def has_data(self) -> bool:
        with self._lock:
            return bool(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def read(self, size: int) -> bytes:
        with self._lock:
            if not self._buffer:
                return b"\x00" * size

            data = bytes(self._buffer[:size])
            del self._buffer[:size]

        if len(data) < size:
            data += b"\x00" * (size - len(data))
        return data


def build_input_audio_append_event(chunk: bytes) -> str:
    return json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }
    )


def decode_output_audio_delta(event: dict[str, object]) -> bytes:
    delta = str(event.get("delta") or "").strip()
    if not delta:
        return b""
    try:
        return base64.b64decode(delta)
    except Exception as exc:
        raise ValueError("Invalid base64 audio delta") from exc


def parse_realtime_event(message: str) -> dict[str, object]:
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object event, got: {type(payload).__name__}")
    return payload


def _maybe_pcm_format(rate: int) -> Optional[dict[str, object]]:
    if rate == 16000:
        return None
    if rate == 24000:
        return {"type": "audio/pcm", "rate": 24000}
    raise ValueError(
        f"Unsupported rate {rate}. Use 16000 for the local pipeline default or 24000 "
        "to match the OpenAI realtime audio format schema."
    )


def build_session_update_event(args: ListenAndPlayWSArguments) -> str:
    input_cfg: dict[str, object] = {
        "turn_detection": {"type": "server_vad", "interrupt_response": True},
    }
    output_cfg: dict[str, object] = {}

    input_format = _maybe_pcm_format(args.send_rate)
    output_format = _maybe_pcm_format(args.recv_rate)
    if input_format is not None:
        input_cfg["format"] = input_format
    if output_format is not None:
        output_cfg["format"] = output_format

    session: dict[str, object] = {
        "type": "realtime",
        "audio": {
            "input": input_cfg,
            "output": output_cfg,
        },
    }
    if args.instructions:
        session["instructions"] = args.instructions

    return json.dumps({"type": "session.update", "session": session})


def parse_args() -> ListenAndPlayWSArguments:
    parser = argparse.ArgumentParser(
        description="Microphone/speaker websocket client for the s2s endpoint.",
    )
    parser.add_argument("--ws-url", default="ws://127.0.0.1:7860/v1/realtime")
    parser.add_argument(
        "--session-url",
        help="Optional load-balancer session allocation URL, for example https://.../session. "
        "If set, the client first requests a direct compute websocket URL from the LB.",
    )
    parser.add_argument("--send-rate", type=int, default=16000)
    parser.add_argument("--recv-rate", type=int, default=16000)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Audio block size in samples.",
    )
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument(
        "--authorization",
        help="Optional Authorization header value, for example 'Bearer <token>'.",
    )
    parser.add_argument(
        "--save-output",
        help="Optional path to store received audio as a WAV file.",
    )
    parser.add_argument(
        "--allow-barge-in",
        action="store_true",
        help="Keep streaming microphone audio while playback audio is buffered.",
    )
    parser.add_argument(
        "--instructions",
        help="Optional realtime session instructions to apply on connect.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print raw realtime event payloads in addition to friendly logs.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print audio devices and exit.",
    )
    namespace = parser.parse_args()
    return ListenAndPlayWSArguments(**vars(namespace))


def require_runtime_dependencies() -> None:
    if websockets is None:
        raise SystemExit(
            "websockets is required for listen_and_play_ws.py. "
            "Install it locally with `pip install websockets`."
        )
    if sd is None:
        raise SystemExit(
            "sounddevice is required for listen_and_play_ws.py. "
            "Install it locally with `pip install sounddevice`."
        )


def write_wav_pcm16(path: str, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
    output_path = Path(path)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


async def listen_and_play_ws(args: ListenAndPlayWSArguments) -> None:
    require_runtime_dependencies()

    if args.list_devices:
        print(sd.query_devices())
        return

    playback = PlaybackBuffer()
    received_audio = bytearray()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=8)
    speaker_active_until = [0.0]
    partial_user_text = {"value": "", "width": 0, "saw_user_speech": False}

    def queue_microphone_frame(frame: bytes) -> None:
        if stop_event.is_set():
            return
        if mic_queue.full():
            with suppress(asyncio.QueueEmpty):
                mic_queue.get_nowait()
        with suppress(asyncio.QueueFull):
            mic_queue.put_nowait(frame)

    def request_stop() -> None:
        if stop_event.is_set():
            return
        stop_event.set()
        with suppress(asyncio.QueueFull):
            mic_queue.put_nowait(None)

    def input_callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"Input stream status: {status}", file=sys.stderr)
        if stop_event.is_set():
            return
        if not args.allow_barge_in and (
            playback.has_data() or time.monotonic() < speaker_active_until[0]
        ):
            return
        loop.call_soon_threadsafe(queue_microphone_frame, bytes(indata))

    def output_callback(outdata, frames, time_info, status) -> None:
        if status:
            print(f"Output stream status: {status}", file=sys.stderr)
        outdata[:] = playback.read(len(outdata))

    def install_signal_handlers() -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, request_stop)

    async def send_audio(ws: websockets.ClientConnection) -> None:
        try:
            while not stop_event.is_set():
                chunk = await mic_queue.get()
                if chunk is None:
                    break
                await ws.send(build_input_audio_append_event(chunk))
        except websockets.ConnectionClosed:
            request_stop()

    async def receive_audio(ws: websockets.ClientConnection) -> None:
        try:
            while not stop_event.is_set():
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    playback.append(msg)
                    received_audio.extend(msg)
                else:
                    handle_realtime_event(
                        msg,
                        playback,
                        received_audio,
                        recv_rate=args.recv_rate,
                        speaker_active_until=speaker_active_until,
                        partial_user_text=partial_user_text,
                        print_json=args.print_json,
                    )
        except websockets.ConnectionClosed:
            request_stop()

    async def wait_for_user_stop() -> None:
        try:
            await asyncio.to_thread(input, "Press Enter to stop...\n")
        except EOFError:
            pass
        request_stop()

    ws_url = args.ws_url
    headers = {}
    if args.authorization:
        headers["Authorization"] = args.authorization

    allocated_session: Optional[AllocatedSession] = None
    if args.session_url:
        allocation = await asyncio.to_thread(allocate_session, args.session_url, args.authorization)
        allocated_session = allocation
        ws_url = allocation.connect_url
        print(f"Allocated session {allocation.session_id}")
        print(f"Direct compute websocket: {allocation.websocket_url}")
        headers = {}

    install_signal_handlers()

    try:
        websocket_cm = websockets.connect(
            ws_url,
            additional_headers=headers or None,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )
        async with websocket_cm as ws:
            print(f"Connected to {ws_url}")
            await ws.send(build_session_update_event(args))
            print("Streaming microphone audio. Press Enter to stop.")

            with sd.RawInputStream(
                samplerate=args.send_rate,
                channels=args.channels,
                dtype="int16",
                blocksize=args.chunk_size,
                device=args.input_device,
                callback=input_callback,
            ), sd.RawOutputStream(
                samplerate=args.recv_rate,
                channels=args.channels,
                dtype="int16",
                blocksize=args.chunk_size,
                device=args.output_device,
                callback=output_callback,
            ):
                tasks = [
                    asyncio.create_task(send_audio(ws)),
                    asyncio.create_task(receive_audio(ws)),
                    asyncio.create_task(wait_for_user_stop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                request_stop()

                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    task.result()
    except websockets.InvalidStatus as exc:
        raise SystemExit(_format_websocket_error(exc, allocated_session, ws_url)) from exc
    except websockets.ConnectionClosedError as exc:
        raise SystemExit(f"Websocket connection closed unexpectedly: {exc}") from exc

    if args.save_output:
        write_wav_pcm16(args.save_output, bytes(received_audio), args.recv_rate, args.channels)
        print(f"Wrote {args.save_output}")


def main() -> None:
    args = parse_args()
    asyncio.run(listen_and_play_ws(args))


def _render_live_user_text(partial_user_text: dict[str, object], text: str, *, final: bool = False) -> None:
    line = f"USER: {text}"
    width = int(partial_user_text["width"])
    padded = line + (" " * max(0, width - len(line)))
    if final:
        print(f"\r{padded}", flush=True)
        partial_user_text["width"] = 0
        return
    print(f"\r{padded}", end="", flush=True)
    partial_user_text["width"] = len(line)


def _clear_live_user_text(partial_user_text: dict[str, object]) -> None:
    width = int(partial_user_text["width"])
    if width == 0:
        return
    print("\r" + (" " * width) + "\r", end="", flush=True)
    partial_user_text["width"] = 0


def handle_realtime_event(
    message: str,
    playback: PlaybackBuffer,
    received_audio: bytearray,
    *,
    recv_rate: int,
    speaker_active_until: list[float],
    partial_user_text: dict[str, object],
    print_json: bool,
) -> None:
    try:
        event = parse_realtime_event(message)
    except Exception as exc:
        print(f"Invalid realtime event: {exc}: {message}", file=sys.stderr)
        return

    if print_json:
        print(f"EVENT: {json.dumps(event, ensure_ascii=True)}")

    event_type = str(event.get("type") or "").strip()

    if event_type == "session.created":
        print("Realtime session created")
        return

    if event_type == "input_audio_buffer.speech_started":
        playback.clear()
        speaker_active_until[0] = 0.0
        partial_user_text["value"] = ""
        if partial_user_text.get("saw_user_speech"):
            print("", flush=True)
        partial_user_text["saw_user_speech"] = True
        return

    if event_type == "input_audio_buffer.speech_stopped":
        return

    if event_type == "conversation.item.input_audio_transcription.delta":
        transcript = str(event.get("delta") or "").strip()
        partial_user_text["value"] = transcript
        if transcript:
            _render_live_user_text(partial_user_text, transcript)
        return

    if event_type == "conversation.item.input_audio_transcription.completed":
        transcript = str(event.get("transcript") or "").strip()
        partial_user_text["value"] = ""
        if transcript:
            _render_live_user_text(partial_user_text, transcript, final=True)
        return

    if event_type == "response.created":
        _clear_live_user_text(partial_user_text)
        print("ASSISTANT:", flush=True)
        return

    if event_type == "response.output_audio.delta":
        try:
            audio_chunk = decode_output_audio_delta(event)
        except ValueError as exc:
            print(f"Failed to decode audio delta: {exc}", file=sys.stderr)
            return
        if audio_chunk:
            playback.append(audio_chunk)
            received_audio.extend(audio_chunk)
            speaker_active_until[0] = time.monotonic() + max(0.15, len(audio_chunk) / (2 * recv_rate))
        return

    if event_type == "response.output_audio.done":
        return

    if event_type == "response.output_audio_transcript.done":
        transcript = str(event.get("transcript") or "").strip()
        if transcript:
            print(f"ASSISTANT: {transcript}", flush=True)
        return

    if event_type == "response.function_call_arguments.done":
        print(
            "TOOL: "
            f"{str(event.get('name') or '').strip()} "
            f"call_id={str(event.get('call_id') or '').strip()} "
            f"arguments={event.get('arguments')}",
            flush=True,
        )
        return

    if event_type == "response.done":
        response = event.get("response")
        status = ""
        if isinstance(response, dict):
            status = str(response.get("status") or "").strip()
        if status == "cancelled":
            playback.clear()
            speaker_active_until[0] = 0.0
            print("ASSISTANT:", flush=True)
        return

    if event_type == "error":
        _clear_live_user_text(partial_user_text)
        error = event.get("error")
        if isinstance(error, dict):
            print(
                f"ERROR: {str(error.get('type') or '').strip()}: {str(error.get('message') or '').strip()}",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: {json.dumps(error, ensure_ascii=True)}", file=sys.stderr)
        return

    _clear_live_user_text(partial_user_text)
    print(f"EVENT {event_type or '<unknown>'}: {json.dumps(event, ensure_ascii=True)}")


def allocate_session(session_url: str, authorization: Optional[str]) -> AllocatedSession:
    headers = {"Content-Type": "application/json"}
    if authorization:
        headers["Authorization"] = authorization

    request = urllib.request.Request(
        session_url,
        data=b"{}",
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Failed to allocate session: HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to allocate session: {exc.reason}") from exc

    connect_url = str(payload.get("connect_url", "")).strip()
    session_id = str(payload.get("session_id", "")).strip()
    websocket_url = str(payload.get("websocket_url", "")).strip()
    if not connect_url or not session_id or not websocket_url:
        raise SystemExit(f"Invalid session allocation response: {payload}")

    return AllocatedSession(
        session_id=session_id,
        websocket_url=websocket_url,
        connect_url=connect_url,
    )


def _format_websocket_error(
    exc: Exception,
    allocated_session: Optional[AllocatedSession],
    ws_url: str,
) -> str:
    lines = [f"Failed to connect to websocket: {exc}"]
    lines.append(f"Websocket URL: {ws_url}")

    if allocated_session is not None:
        lines.append(f"Allocated session id: {allocated_session.session_id}")
        lines.append(f"Allocated compute websocket: {allocated_session.websocket_url}")
        lines.append(
            "A 403 here usually means the compute endpoint rejected the session token or is running an older image."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()

import base64
import unittest

from listen_and_play_ws import (
    PlaybackBuffer,
    build_input_audio_append_event,
    build_session_update_event,
    decode_output_audio_delta,
    handle_realtime_event,
    ListenAndPlayWSArguments,
    parse_realtime_event,
)


class ListenAndPlayWSTests(unittest.TestCase):
    def test_build_input_audio_append_event_encodes_base64_audio(self):
        payload = parse_realtime_event(build_input_audio_append_event(b"\x01\x02\x03"))
        self.assertEqual(payload["type"], "input_audio_buffer.append")
        self.assertEqual(payload["audio"], base64.b64encode(b"\x01\x02\x03").decode("ascii"))

    def test_decode_output_audio_delta_decodes_base64_audio(self):
        event = {
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(b"\x10\x20").decode("ascii"),
        }
        self.assertEqual(decode_output_audio_delta(event), b"\x10\x20")

    def test_build_session_update_event_uses_server_vad_and_optional_pcm_format(self):
        args = ListenAndPlayWSArguments(send_rate=24000, recv_rate=24000, instructions="Be concise")
        payload = parse_realtime_event(build_session_update_event(args))

        self.assertEqual(payload["type"], "session.update")
        session = payload["session"]
        self.assertEqual(session["type"], "realtime")
        self.assertEqual(session["instructions"], "Be concise")
        self.assertEqual(session["audio"]["input"]["turn_detection"]["type"], "server_vad")
        self.assertEqual(session["audio"]["output"]["format"]["rate"], 24000)

    def test_handle_realtime_event_appends_audio_delta_to_playback_and_output(self):
        playback = PlaybackBuffer()
        received_audio = bytearray()
        handle_realtime_event(
            '{"type":"response.output_audio.delta","delta":"AQI="}',
            playback,
            received_audio,
            recv_rate=16000,
            speaker_active_until=[0.0],
            partial_user_text={"value": "", "width": 0, "saw_user_speech": False},
            print_json=False,
        )

        self.assertEqual(playback.read(2), b"\x01\x02")
        self.assertEqual(bytes(received_audio), b"\x01\x02")


if __name__ == "__main__":
    unittest.main()

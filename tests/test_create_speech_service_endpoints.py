import argparse
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from create_speech_service_endpoints import (  # noqa: E402
    DEFAULT_STT_REPOSITORY,
    DEFAULT_TTS_REPOSITORY,
    SpeechServiceSpec,
    build_specs,
    ensure_names_available,
    main,
    parse_args,
)


class FakeResponse:
    status_code = 404


class FakeNotFound(Exception):
    response = FakeResponse()


def make_args(**overrides):
    values = {
        "services": ["stt", "tts"],
        "stt_name": "reachy-s2s-stt-01",
        "tts_name": "reachy-s2s-tts-01",
        "stt_repository": DEFAULT_STT_REPOSITORY,
        "tts_repository": DEFAULT_TTS_REPOSITORY,
        "stt_revision": None,
        "tts_revision": None,
        "stt_image_url": "ghcr.io/andimarafioti/s2s-stt:v0.2",
        "tts_image_url": "ghcr.io/andimarafioti/s2s-tts:v0.2",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SpeechServiceSpecTests(unittest.TestCase):
    def test_build_specs_resolves_each_model_revision(self):
        api = MagicMock()
        api.model_info.side_effect = [
            MagicMock(sha="stt-sha"),
            MagicMock(sha="tts-sha"),
        ]

        specs = build_specs(make_args(), api)

        self.assertEqual([spec.service for spec in specs], ["stt", "tts"])
        self.assertEqual([spec.revision for spec in specs], ["stt-sha", "tts-sha"])
        self.assertEqual([spec.port for spec in specs], [8000, 8091])

    def test_build_specs_uses_explicit_revision_without_lookup(self):
        api = MagicMock()

        specs = build_specs(
            make_args(services=["stt"], stt_revision="pinned-stt-sha"),
            api,
        )

        self.assertEqual(specs[0].revision, "pinned-stt-sha")
        api.model_info.assert_not_called()

    def test_ensure_names_available_rejects_collisions(self):
        api = MagicMock()
        api.get_inference_endpoint.return_value = object()
        specs = [SpeechServiceSpec("stt", "taken", "repo", "sha", "image", 8000)]

        with self.assertRaisesRegex(ValueError, "already exists: taken"):
            ensure_names_available(api, "org", specs)

    def test_ensure_names_available_accepts_404(self):
        api = MagicMock()
        api.get_inference_endpoint.side_effect = FakeNotFound()
        specs = [SpeechServiceSpec("stt", "free", "repo", "sha", "image", 8000)]

        ensure_names_available(api, "org", specs)


class CreateSpeechServiceEndpointsMainTests(unittest.TestCase):
    def test_parse_args_requires_image_for_each_selected_service(self):
        with (
            patch("sys.argv", ["create_speech_service_endpoints", "--services", "stt"]),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_parse_args_accepts_only_the_selected_service_image(self):
        with patch(
            "sys.argv",
            [
                "create_speech_service_endpoints",
                "--services",
                "stt",
                "--stt-image-url",
                "ghcr.io/example/s2s-stt:sha-1234",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.stt_image_url, "ghcr.io/example/s2s-stt:sha-1234")
        self.assertIsNone(args.tts_image_url)

    def test_main_creates_protected_warm_gpu_endpoints(self):
        api = MagicMock()
        api.model_info.side_effect = [MagicMock(sha="stt-sha"), MagicMock(sha="tts-sha")]
        api.get_inference_endpoint.side_effect = FakeNotFound()
        stt_endpoint = MagicMock(name="stt-endpoint")
        stt_endpoint.name = "reachy-s2s-stt-01"
        stt_endpoint.status = "pending"
        stt_endpoint.url = None
        tts_endpoint = MagicMock(name="tts-endpoint")
        tts_endpoint.name = "reachy-s2s-tts-01"
        tts_endpoint.status = "pending"
        tts_endpoint.url = None
        api.create_inference_endpoint.side_effect = [stt_endpoint, tts_endpoint]

        with (
            patch(
                "sys.argv",
                [
                    "create_speech_service_endpoints",
                    "--stt-image-url",
                    "ghcr.io/example/s2s-stt:sha-1234",
                    "--tts-image-url",
                    "ghcr.io/example/s2s-tts:sha-1234",
                ],
            ),
            patch("sys.stdout", io.StringIO()),
            patch("create_speech_service_endpoints.HfApi", return_value=api),
        ):
            main()

        self.assertEqual(api.create_inference_endpoint.call_count, 2)
        stt_call, tts_call = api.create_inference_endpoint.call_args_list
        self.assertEqual(stt_call.kwargs["repository"], DEFAULT_STT_REPOSITORY)
        self.assertEqual(stt_call.kwargs["custom_image"]["port"], 8000)
        self.assertEqual(tts_call.kwargs["repository"], DEFAULT_TTS_REPOSITORY)
        self.assertEqual(tts_call.kwargs["custom_image"]["port"], 8091)
        for call in (stt_call, tts_call):
            self.assertEqual(call.kwargs["type"], "protected")
            self.assertEqual(call.kwargs["min_replica"], 1)
            self.assertEqual(call.kwargs["max_replica"], 1)
            self.assertEqual(call.kwargs["instance_type"], "nvidia-a10g")
            self.assertEqual(call.kwargs["region"], "us-east-1")


if __name__ == "__main__":
    unittest.main()

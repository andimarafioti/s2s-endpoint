import argparse
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from create_speech_proxy_endpoints import (  # noqa: E402
    SpeechBackendTarget,
    SpeechProxySpec,
    build_specs,
    deployment_env,
    ensure_names_available,
    main,
    parse_args,
    resolve_secrets,
)


class FakeResponse:
    status_code = 404


class FakeNotFound(Exception):
    response = FakeResponse()


def make_args(**overrides):
    values = {
        "namespace": "HuggingFaceM4",
        "services": ["stt", "tts"],
        "stt_proxy_name": "reachy-s2s-stt-proxy",
        "tts_proxy_name": "reachy-s2s-tts-proxy",
        "stt_backends": ["reachy-s2s-stt-01"],
        "tts_backends": ["reachy-s2s-tts-01"],
        "stt_target_work": 96.0,
        "stt_max_work": 128.0,
        "stt_latency_target": 0.1,
        "stt_audio_equivalent": 5.0,
        "tts_target_work": 8.0,
        "tts_max_work": 16.0,
        "tts_latency_target": 0.5,
        "latency_weight": 0.25,
        "max_attempts": 2,
        "max_connections": 1024,
        "max_keepalive_connections": 256,
        "health_interval": 10.0,
        "health_timeout": 5.0,
        "request_timeout": 120.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class SpeechProxyDeploymentTests(unittest.TestCase):
    def test_build_specs_resolves_worker_names_to_urls(self):
        api = MagicMock()
        api.get_inference_endpoint.side_effect = [
            MagicMock(url="https://stt.example/"),
            MagicMock(url="https://tts.example/"),
        ]

        specs = build_specs(make_args(), api)

        self.assertEqual([spec.service for spec in specs], ["stt", "tts"])
        self.assertEqual(specs[0].backends[0], SpeechBackendTarget("reachy-s2s-stt-01", "https://stt.example"))
        self.assertEqual(specs[1].backends[0], SpeechBackendTarget("reachy-s2s-tts-01", "https://tts.example"))

    def test_build_specs_rejects_a_worker_without_a_url(self):
        api = MagicMock()
        api.get_inference_endpoint.return_value = MagicMock(url=None)

        with self.assertRaisesRegex(ValueError, "does not have a URL"):
            build_specs(make_args(services=["stt"]), api)

    def test_deployment_env_keeps_stt_and_tts_capacity_independent(self):
        args = make_args()
        stt = SpeechProxySpec(
            "stt",
            "stt-proxy",
            (SpeechBackendTarget("stt-01", "https://stt.example"),),
            96,
            128,
            0.1,
        )
        tts = SpeechProxySpec(
            "tts",
            "tts-proxy",
            (SpeechBackendTarget("tts-01", "https://tts.example"),),
            8,
            16,
            0.5,
        )

        stt_env = deployment_env(args, stt)
        tts_env = deployment_env(args, tts)

        self.assertEqual(stt_env["SPEECH_BACKENDS"], "stt-01=https://stt.example")
        self.assertEqual(stt_env["SPEECH_TARGET_WORK"], "96")
        self.assertEqual(stt_env["STT_AUDIO_EQUIVALENT_S"], "5.0")
        self.assertNotIn("TTS_WARMUP_ENABLED", stt_env)
        self.assertEqual(tts_env["SPEECH_TARGET_WORK"], "8")
        self.assertEqual(tts_env["SPEECH_MAX_WORK"], "16")
        self.assertEqual(tts_env["TTS_WARMUP_ENABLED"], "true")
        self.assertNotIn("STT_AUDIO_EQUIVALENT_S", tts_env)

    def test_resolve_secrets_maps_hf_token_to_backend_key(self):
        self.assertEqual(
            resolve_secrets({"HF_TOKEN": "hf-secret"}),
            {"SPEECH_BACKEND_API_KEY": "hf-secret"},
        )

    def test_ensure_names_available_rejects_collisions(self):
        api = MagicMock()
        spec = SpeechProxySpec("stt", "taken", (), 96, 128, 0.1)

        with self.assertRaisesRegex(ValueError, "already exists: taken"):
            ensure_names_available(api, "org", [spec])

    def test_parse_args_rejects_multiple_proxy_replicas_by_not_exposing_replica_flags(self):
        with (
            patch("sys.argv", ["create_speech_proxy_endpoints", "--image-url", "image", "--max-replica", "2"]),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_main_creates_two_single_replica_cpu_proxies(self):
        api = MagicMock()

        def get_endpoint(name, *, namespace):
            self.assertEqual(namespace, "HuggingFaceM4")
            if name == "reachy-s2s-stt-01":
                return MagicMock(url="https://stt.example")
            if name == "reachy-s2s-tts-01":
                return MagicMock(url="https://tts.example")
            raise FakeNotFound()

        api.get_inference_endpoint.side_effect = get_endpoint
        stt_proxy = MagicMock()
        stt_proxy.name = "reachy-s2s-stt-proxy"
        stt_proxy.status = "pending"
        stt_proxy.url = None
        tts_proxy = MagicMock()
        tts_proxy.name = "reachy-s2s-tts-proxy"
        tts_proxy.status = "pending"
        tts_proxy.url = None
        api.create_inference_endpoint.side_effect = [stt_proxy, tts_proxy]

        with (
            patch(
                "sys.argv",
                [
                    "create_speech_proxy_endpoints",
                    "--image-url",
                    "ghcr.io/example/s2s-speech-proxy:sha-1234",
                ],
            ),
            patch("sys.stdout", io.StringIO()),
            patch.dict("os.environ", {"HF_TOKEN": "hf-secret"}, clear=True),
            patch("create_speech_proxy_endpoints.HfApi", return_value=api),
        ):
            main()

        self.assertEqual(api.create_inference_endpoint.call_count, 2)
        stt_call, tts_call = api.create_inference_endpoint.call_args_list
        for call in (stt_call, tts_call):
            self.assertEqual(call.kwargs["accelerator"], "cpu")
            self.assertEqual(call.kwargs["instance_type"], "intel-spr")
            self.assertEqual(call.kwargs["instance_size"], "x1")
            self.assertEqual(call.kwargs["min_replica"], 1)
            self.assertEqual(call.kwargs["max_replica"], 1)
            self.assertEqual(call.kwargs["custom_image"]["health_route"], "/health")
            self.assertEqual(call.kwargs["secrets"], {"SPEECH_BACKEND_API_KEY": "hf-secret"})
        self.assertEqual(stt_call.kwargs["env"]["SPEECH_PROXY_SERVICE"], "stt")
        self.assertEqual(tts_call.kwargs["env"]["SPEECH_PROXY_SERVICE"], "tts")


if __name__ == "__main__":
    unittest.main()

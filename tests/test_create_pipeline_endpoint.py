import argparse
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from create_pipeline_endpoint import deployment_env, main, parse_args, resolve_secrets  # noqa: E402


class FakeResponse:
    status_code = 404


class FakeNotFound(Exception):
    response = FakeResponse()


class CreatePipelineEndpointTests(unittest.TestCase):
    def test_parse_args_requires_explicit_image(self):
        with (
            patch(
                "sys.argv",
                [
                    "create_pipeline_endpoint",
                    "--stt-base-url",
                    "https://stt.example/v1",
                    "--tts-base-url",
                    "https://tts.example/v1",
                ],
            ),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_deployment_env_preserves_non_streaming_stt_and_sentence_batching(self):
        args = argparse.Namespace(
            stt_base_url="https://stt.example/v1",
            tts_base_url="https://tts.example/v1",
            model_name="gpt-test",
            llm_base_url=None,
            llm_backend="responses-api",
            num_pipelines=1,
            enable_live_transcription=False,
            stream_batch_sentences=3,
        )

        env = deployment_env(args)

        self.assertEqual(env["ENABLE_LIVE_TRANSCRIPTION"], "false")
        self.assertEqual(env["STREAM_BATCH_SENTENCES"], "3")
        self.assertEqual(env["LOG_TRANSCRIPTS"], "false")

    def test_deployment_env_targets_llm_proxy(self):
        args = argparse.Namespace(
            stt_base_url="https://stt.example/v1",
            tts_base_url="https://tts.example/v1",
            llm_base_url="https://llm.example/v1/",
            llm_backend="chat-completions",
            model_name="nvidia/Gemma-4-26B-A4B-NVFP4",
            num_pipelines=4,
            enable_live_transcription=False,
            stream_batch_sentences=3,
        )

        env = deployment_env(args)

        self.assertEqual(env["LLM_BASE_URL"], "https://llm.example/v1")
        self.assertEqual(env["LLM_BACKEND"], "chat-completions")

    def test_resolve_secrets_reuses_hf_token_for_protected_llm_proxy(self):
        self.assertEqual(
            resolve_secrets({"HF_TOKEN": "hf-secret"}, use_hf_llm=True),
            {"HF_TOKEN": "hf-secret", "RESPONSES_API_API_KEY": "hf-secret"},
        )

    def test_resolve_secrets_accepts_openai_key_alias(self):
        secrets = resolve_secrets({"HF_TOKEN": "hf-secret", "OPENAI_API_KEY": "openai-secret"})

        self.assertEqual(
            secrets,
            {"HF_TOKEN": "hf-secret", "RESPONSES_API_API_KEY": "openai-secret"},
        )

    def test_main_creates_protected_warm_cpu_endpoint(self):
        api = MagicMock()
        api.get_inference_endpoint.side_effect = FakeNotFound()
        endpoint = MagicMock()
        endpoint.name = "reachy-s2s-pipeline-01"
        endpoint.status = "pending"
        endpoint.url = None
        api.create_inference_endpoint.return_value = endpoint

        with (
            patch(
                "sys.argv",
                [
                    "create_pipeline_endpoint",
                    "--image-url",
                    "ghcr.io/example/s2s-pipeline:sha-1234",
                    "--stt-base-url",
                    "https://stt.example/v1",
                    "--tts-base-url",
                    "https://tts.example/v1",
                ],
            ),
            patch("sys.stdout", io.StringIO()),
            patch.dict(
                "os.environ",
                {"HF_TOKEN": "hf-secret", "OPENAI_API_KEY": "openai-secret"},
                clear=True,
            ),
            patch("create_pipeline_endpoint.HfApi", return_value=api),
        ):
            main()

        call = api.create_inference_endpoint.call_args
        self.assertEqual(call.kwargs["accelerator"], "cpu")
        self.assertEqual(call.kwargs["instance_type"], "intel-spr")
        self.assertEqual(call.kwargs["instance_size"], "x4")
        self.assertEqual(call.kwargs["region"], "us-east-1")
        self.assertEqual(call.kwargs["type"], "protected")
        self.assertEqual(call.kwargs["min_replica"], 1)
        self.assertEqual(call.kwargs["max_replica"], 1)
        self.assertEqual(call.kwargs["custom_image"]["health_route"], "/v1/usage")
        self.assertEqual(call.kwargs["secrets"]["RESPONSES_API_API_KEY"], "openai-secret")
        self.assertEqual(call.kwargs["env"]["STT_BASE_URL"], "https://stt.example/v1")
        self.assertEqual(call.kwargs["env"]["TTS_BASE_URL"], "https://tts.example/v1")


if __name__ == "__main__":
    unittest.main()

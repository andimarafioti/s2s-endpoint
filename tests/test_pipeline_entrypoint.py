import json
import os
import stat
import unittest
from unittest.mock import patch

from pipeline_entrypoint import build_config, write_private_config


class PipelineEntrypointTests(unittest.TestCase):
    def test_build_config_routes_all_inference_to_remote_services(self):
        config = build_config(
            {
                "HF_TOKEN": "hf-secret",
                "OPENAI_API_KEY": "openai-secret",
                "STT_BASE_URL": "https://stt.example/v1",
                "TTS_BASE_URL": "https://tts.example/v1",
            }
        )

        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["stt"], "openai")
        self.assertEqual(config["openai_stt_base_url"], "https://stt.example/v1")
        self.assertEqual(config["openai_stt_api_key"], "hf-secret")
        self.assertEqual(config["llm_backend"], "responses-api")
        self.assertEqual(config["responses_api_api_key"], "openai-secret")
        self.assertEqual(config["tts"], "openai")
        self.assertEqual(config["openai_tts_base_url"], "https://tts.example/v1")
        self.assertEqual(config["openai_tts_api_key"], "hf-secret")
        self.assertFalse(config["enable_live_transcription"])
        self.assertEqual(config["stream_batch_sentences"], 3)

    def test_build_config_routes_llm_to_protected_chat_completions_proxy(self):
        config = build_config(
            {
                "HF_TOKEN": "hf-secret",
                "STT_BASE_URL": "https://stt.example/v1",
                "TTS_BASE_URL": "https://tts.example/v1",
                "LLM_BASE_URL": "https://llm.example/v1",
                "LLM_BACKEND": "chat-completions",
                "MODEL_NAME": "nvidia/Gemma-4-26B-A4B-NVFP4",
            }
        )

        self.assertEqual(config["llm_backend"], "chat-completions")
        self.assertEqual(config["responses_api_base_url"], "https://llm.example/v1")
        self.assertEqual(config["responses_api_api_key"], "hf-secret")
        self.assertEqual(config["model_name"], "nvidia/Gemma-4-26B-A4B-NVFP4")
        self.assertNotIn("responses_api_reasoning_effort", config)

    def test_build_config_requires_both_credential_domains(self):
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            build_config(
                {
                    "OPENAI_API_KEY": "openai-secret",
                    "STT_BASE_URL": "https://stt.example/v1",
                    "TTS_BASE_URL": "https://tts.example/v1",
                }
            )
        with self.assertRaisesRegex(ValueError, "RESPONSES_API_API_KEY or OPENAI_API_KEY"):
            build_config(
                {
                    "HF_TOKEN": "hf-secret",
                    "STT_BASE_URL": "https://stt.example/v1",
                    "TTS_BASE_URL": "https://tts.example/v1",
                }
            )

    def test_build_config_requires_explicit_speech_service_urls(self):
        base_environ = {
            "HF_TOKEN": "hf-secret",
            "OPENAI_API_KEY": "openai-secret",
        }

        with self.assertRaisesRegex(ValueError, "STT_BASE_URL"):
            build_config({**base_environ, "TTS_BASE_URL": "https://tts.example/v1"})
        with self.assertRaisesRegex(ValueError, "TTS_BASE_URL"):
            build_config({**base_environ, "STT_BASE_URL": "https://stt.example/v1"})

    def test_build_config_rejects_unknown_llm_backend(self):
        with self.assertRaisesRegex(ValueError, "LLM_BACKEND"):
            build_config(
                {
                    "HF_TOKEN": "hf-secret",
                    "STT_BASE_URL": "https://stt.example/v1",
                    "TTS_BASE_URL": "https://tts.example/v1",
                    "LLM_BASE_URL": "https://llm.example/v1",
                    "LLM_BACKEND": "other",
                }
            )

    def test_write_private_config_uses_owner_only_permissions(self):
        config_path = write_private_config({"secret": "value"})
        self.addCleanup(config_path.unlink, missing_ok=True)

        mode = stat.S_IMODE(config_path.stat().st_mode)
        self.assertEqual(mode, 0o600)
        self.assertEqual(json.loads(config_path.read_text(encoding="utf-8")), {"secret": "value"})

    def test_main_execs_serve_without_putting_secrets_in_argv(self):
        with (
            patch.dict(
                os.environ,
                {
                    "HF_TOKEN": "hf-secret",
                    "OPENAI_API_KEY": "openai-secret",
                    "STT_BASE_URL": "https://stt.example/v1",
                    "TTS_BASE_URL": "https://tts.example/v1",
                },
                clear=True,
            ),
            patch("pipeline_entrypoint.write_private_config", return_value="/tmp/private.json"),
            patch("pipeline_entrypoint.os.execvp") as execvp,
        ):
            from pipeline_entrypoint import main

            main()

        execvp.assert_called_once_with(
            "speech-to-speech",
            ["speech-to-speech", "serve", "/tmp/private.json"],
        )


if __name__ == "__main__":
    unittest.main()

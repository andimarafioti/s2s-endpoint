import os
import unittest
from unittest.mock import AsyncMock, patch

from app import compute_app as compute_main


class BuildS2SCommandTests(unittest.TestCase):
    def build_command_with_env(self, env: dict[str, str]) -> list[str]:
        with patch.dict(os.environ, env, clear=True):
            settings = compute_main.ComputeSettings.from_env()
        return compute_main.build_s2s_command("127.0.0.1", 9000, settings)

    def test_uses_serve_subcommand_and_server_address_flags(self):
        command = self.build_command_with_env({})

        self.assertEqual(
            command[:12],
            [
                "uv",
                "run",
                "--no-dev",
                "--no-sync",
                "--directory",
                "/opt/speech-to-speech",
                "speech-to-speech",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
            ],
        )
        self.assertNotIn("--mode", command)
        self.assertNotIn("--ws_host", command)
        self.assertNotIn("--ws_port", command)

    def test_chat_completions_forwards_explicit_openai_compatible_connection_flags(self):
        command = self.build_command_with_env(
            {
                "LLM": "chat-completions",
                "MODEL_NAME": "google/gemma-4-31B-it:cerebras",
                "RESPONSES_API_BASE_URL": "https://router.huggingface.co/v1",
                "RESPONSES_API_REASONING_EFFORT": "none",
                "HF_TOKEN": "hf_token",
            }
        )

        self.assertEqual(command[command.index("--llm_backend") + 1], "chat-completions")
        self.assertEqual(command[command.index("--model_name") + 1], "google/gemma-4-31B-it:cerebras")
        self.assertEqual(command[command.index("--responses_api_base_url") + 1], "https://router.huggingface.co/v1")
        self.assertEqual(command[command.index("--responses_api_api_key") + 1], "hf_token")
        self.assertEqual(command[command.index("--responses_api_reasoning_effort") + 1], "none")

    def test_chat_completions_defaults_leave_provider_configuration_unset(self):
        command = self.build_command_with_env({})

        self.assertEqual(command[command.index("--llm_backend") + 1], "chat-completions")
        self.assertNotIn("--model_name", command)
        self.assertNotIn("--responses_api_base_url", command)
        self.assertNotIn("--responses_api_reasoning_effort", command)

    def test_llm_proxy_is_off_by_default(self):
        command = self.build_command_with_env({})

        self.assertNotIn("--enable_llm_proxy", command)

    def test_enable_llm_proxy_env_passes_the_flag(self):
        command = self.build_command_with_env({"ENABLE_LLM_PROXY": "1"})

        self.assertIn("--enable_llm_proxy", command)

    def test_smart_turn_uses_upstream_defaults(self):
        command = self.build_command_with_env({})

        self.assertIn("--no-sync", command)
        self.assertNotIn("--smart_turn", command)
        self.assertNotIn("--no_smart_turn", command)
        self.assertNotIn("--smart_turn_model_path", command)
        self.assertNotIn("--smart_turn_threshold", command)
        self.assertNotIn("--smart_turn_max_wait_ms", command)
        self.assertNotIn("--smart_turn_incomplete_delay_ms", command)
        self.assertNotIn("--smart_turn_cpu_count", command)

    def test_baked_smart_turn_model_path_is_forwarded(self):
        command = self.build_command_with_env(
            {
                "SMART_TURN_MODEL_PATH": "/opt/models/smart-turn-v3.2-cpu.onnx",
            }
        )

        self.assertNotIn("--smart_turn", command)
        self.assertEqual(
            command[command.index("--smart_turn_model_path") + 1],
            "/opt/models/smart-turn-v3.2-cpu.onnx",
        )

    def test_disabled_smart_turn_ignores_baked_model_path(self):
        command = self.build_command_with_env(
            {
                "ENABLE_SMART_TURN": "0",
                "SMART_TURN_MODEL_PATH": "/opt/models/smart-turn-v3.2-cpu.onnx",
            }
        )

        self.assertNotIn("--smart_turn", command)
        self.assertIn("--no_smart_turn", command)
        self.assertNotIn("--smart_turn_model_path", command)


class WaitForInternalRealtimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_internal_server_uses_usage_endpoint(self):
        observed = {}

        def fake_get_json(url: str):
            observed["url"] = url
            return {"requests": 0}

        connect = AsyncMock(side_effect=AssertionError("websocket handshake should not be used"))

        with (
            patch.object(compute_main, "_http_get_json", fake_get_json),
            patch.object(compute_main.asyncio, "open_connection", connect),
        ):
            await compute_main.wait_for_internal_server(
                "127.0.0.1",
                9000,
                None,
                timeout_s=0.01,
            )

        connect.assert_not_called()
        self.assertEqual(
            observed["url"],
            "http://127.0.0.1:9000/v1/usage",
        )


if __name__ == "__main__":
    unittest.main()

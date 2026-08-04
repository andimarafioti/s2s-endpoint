import importlib
import os
import unittest
from unittest.mock import AsyncMock, patch

from app import compute_main


class BuildS2SCommandTests(unittest.TestCase):
    def build_command_with_env(self, env: dict[str, str]) -> list[str]:
        with patch.dict(os.environ, env, clear=True):
            module = importlib.reload(compute_main)
            command = module.build_s2s_command("127.0.0.1", 9000)
        importlib.reload(compute_main)
        return command

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

    def test_smart_turn_is_disabled_by_default(self):
        command = self.build_command_with_env({})

        self.assertNotIn("--smart_turn", command)
        self.assertNotIn("--smart_turn_device", command)

    def test_smart_turn_cuda_configuration_is_forwarded(self):
        command = self.build_command_with_env(
            {
                "ENABLE_SMART_TURN": "1",
                "SMART_TURN_DEVICE": "cuda",
                "SMART_TURN_MODEL_PATH": "/opt/models/smart-turn-v3.2-gpu.onnx",
                "SMART_TURN_THRESHOLD": "0.65",
                "SMART_TURN_MAX_WAIT_MS": "2500",
                "SMART_TURN_CPU_COUNT": "2",
            }
        )

        self.assertIn("--no-sync", command)
        self.assertIn("--smart_turn", command)
        self.assertEqual(command[command.index("--smart_turn_device") + 1], "cuda")
        self.assertEqual(
            command[command.index("--smart_turn_model_path") + 1],
            "/opt/models/smart-turn-v3.2-gpu.onnx",
        )
        self.assertEqual(command[command.index("--smart_turn_threshold") + 1], "0.65")
        self.assertEqual(command[command.index("--smart_turn_max_wait_ms") + 1], "2500")
        self.assertEqual(command[command.index("--smart_turn_cpu_count") + 1], "2")

    def test_disabled_smart_turn_does_not_forward_tuning(self):
        command = self.build_command_with_env(
            {
                "ENABLE_SMART_TURN": "0",
                "SMART_TURN_DEVICE": "cpu",
                "SMART_TURN_THRESHOLD": "0.75",
            }
        )

        self.assertNotIn("--smart_turn", command)
        self.assertNotIn("--smart_turn_device", command)
        self.assertNotIn("--smart_turn_threshold", command)

    def test_smart_turn_config_reports_effective_settings(self):
        with patch.dict(
            os.environ,
            {
                "ENABLE_SMART_TURN": "1",
                "SMART_TURN_DEVICE": "cuda",
                "SMART_TURN_MODEL_PATH": "/opt/models/smart-turn-v3.2-gpu.onnx",
                "SMART_TURN_THRESHOLD": "0.5",
                "SMART_TURN_MAX_WAIT_MS": "2000",
            },
            clear=True,
        ):
            module = importlib.reload(compute_main)
            config = module._smart_turn_config()
        importlib.reload(compute_main)

        self.assertEqual(
            config,
            {
                "enabled": True,
                "device": "cuda",
                "model_path": "/opt/models/smart-turn-v3.2-gpu.onnx",
                "threshold": "0.5",
                "max_wait_ms": "2000",
                "cpu_count": None,
            },
        )


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
                compute_main.INTERNAL_WS_HOST,
                compute_main.INTERNAL_WS_BASE_PORT,
                None,
                timeout_s=0.01,
            )

        connect.assert_not_called()
        self.assertEqual(
            observed["url"],
            f"http://{compute_main.INTERNAL_WS_HOST}:{compute_main.INTERNAL_WS_BASE_PORT}/v1/usage",
        )


if __name__ == "__main__":
    unittest.main()

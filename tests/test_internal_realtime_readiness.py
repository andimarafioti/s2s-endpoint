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
                "HF_TOKEN": "hf_token",
            }
        )

        self.assertEqual(command[command.index("--llm_backend") + 1], "chat-completions")
        self.assertEqual(command[command.index("--model_name") + 1], "google/gemma-4-31B-it:cerebras")
        self.assertEqual(command[command.index("--responses_api_base_url") + 1], "https://router.huggingface.co/v1")
        self.assertEqual(command[command.index("--responses_api_api_key") + 1], "hf_token")


class WaitForInternalRealtimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_internal_server_uses_usage_endpoint(self):
        observed = {}

        def fake_get_json(url: str):
            observed["url"] = url
            return {"requests": 0}

        connect = AsyncMock(side_effect=AssertionError("websocket handshake should not be used"))

        with patch.object(compute_main, "_http_get_json", fake_get_json), patch.object(
            compute_main.asyncio, "open_connection", connect
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

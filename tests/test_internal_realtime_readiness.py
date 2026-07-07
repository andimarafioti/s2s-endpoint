import unittest
from unittest.mock import AsyncMock, patch

from app import compute_main


class BuildS2SCommandTests(unittest.TestCase):
    def test_chat_completions_defaults_use_hf_router_and_no_reasoning(self):
        with (
            patch.object(compute_main, "LLM", "chat-completions"),
            patch.object(compute_main, "MODEL_NAME", "google/gemma-4-31B-it:cerebras"),
            patch.object(compute_main, "RESPONSES_API_BASE_URL", "https://router.huggingface.co/v1"),
            patch.object(compute_main, "RESPONSES_API_API_KEY", ""),
            patch.object(compute_main, "RESPONSES_API_REASONING_EFFORT", "none"),
        ):
            cmd = compute_main.build_s2s_command("127.0.0.1", 9000)

        self.assertIn("--llm_backend", cmd)
        self.assertEqual(cmd[cmd.index("--llm_backend") + 1], "chat-completions")
        self.assertEqual(cmd[cmd.index("--model_name") + 1], "google/gemma-4-31B-it:cerebras")
        self.assertEqual(cmd[cmd.index("--responses_api_base_url") + 1], "https://router.huggingface.co/v1")
        self.assertEqual(cmd[cmd.index("--responses_api_reasoning_effort") + 1], "none")


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

import unittest
from unittest.mock import AsyncMock, patch

from app import compute_main


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

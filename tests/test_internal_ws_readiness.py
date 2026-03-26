import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from app import compute_main


class WaitForInternalWSTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_internal_ws_uses_websocket_handshake(self):
        observed = {}

        @asynccontextmanager
        async def fake_connect(*args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            yield object()

        open_connection = AsyncMock(side_effect=AssertionError("raw TCP probe should not be used"))

        with patch.object(compute_main.websockets, "connect", fake_connect), patch.object(
            compute_main.asyncio, "open_connection", open_connection
        ):
            await compute_main.wait_for_internal_ws(
                compute_main.INTERNAL_WS_HOST,
                compute_main.INTERNAL_WS_BASE_PORT,
                None,
                timeout_s=0.01,
            )

        open_connection.assert_not_called()
        self.assertEqual(observed["args"], (compute_main.INTERNAL_WS_URL,))
        self.assertEqual(
            observed["kwargs"],
            {
                "open_timeout": 5,
                "ping_interval": None,
                "max_size": None,
            },
        )


if __name__ == "__main__":
    unittest.main()

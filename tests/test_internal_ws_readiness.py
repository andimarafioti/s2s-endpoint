import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from app import main


class WaitForInternalWSTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_for_internal_ws_uses_websocket_handshake(self):
        observed = {}

        @asynccontextmanager
        async def fake_connect(*args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            yield object()

        open_connection = AsyncMock(side_effect=AssertionError("raw TCP probe should not be used"))

        with patch.object(main.websockets, "connect", fake_connect), patch.object(
            main.asyncio, "open_connection", open_connection
        ):
            await main.wait_for_internal_ws(timeout_s=0.01)

        open_connection.assert_not_called()
        self.assertEqual(observed["args"], (main.INTERNAL_WS_URL,))
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

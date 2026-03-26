import asyncio
import logging
from typing import Awaitable, Callable, Optional, Protocol

from fastapi import WebSocket, WebSocketDisconnect
import websockets
from websockets.exceptions import ConnectionClosed


logger = logging.getLogger("s2s-endpoint")


class LeaseLike(Protocol):
    slot_id: object
    ws_url: str


AcquireLease = Callable[[float], Awaitable[LeaseLike]]
ReleaseLease = Callable[[object], Awaitable[None]]
DescribeLease = Callable[[LeaseLike], str]


async def proxy_websocket(
    client_ws: WebSocket,
    *,
    acquire_lease: AcquireLease,
    release_lease: ReleaseLease,
    describe_lease: DescribeLease,
    no_capacity_reason: str,
    no_capacity_log: str,
    additional_headers: Optional[list[tuple[str, str]]] = None,
) -> None:
    lease = None

    try:
        lease = await acquire_lease(900.0)
    except Exception as exc:
        await client_ws.close(code=1013, reason=no_capacity_reason)
        logger.warning("%s: %s", no_capacity_log, exc)
        return

    await client_ws.accept()
    logger.info("Client websocket connected to %s", describe_lease(lease))

    try:
        async with websockets.connect(
            lease.ws_url,
            additional_headers=additional_headers,
            open_timeout=30,
            ping_interval=20,
            ping_timeout=20,
            max_size=None,
        ) as upstream_ws:
            await asyncio.gather(
                _client_to_upstream(client_ws, upstream_ws),
                _upstream_to_client(client_ws, upstream_ws),
            )
    except WebSocketDisconnect:
        logger.info("Client websocket disconnected")
    except ConnectionClosed:
        logger.info("Upstream websocket disconnected")
        try:
            await client_ws.close()
        except Exception:
            pass
    except Exception:
        logger.exception("Websocket proxy failed")
        try:
            await client_ws.close(code=1011, reason="Proxy failure")
        except Exception:
            pass
    finally:
        if lease is not None:
            await release_lease(lease.slot_id)


async def _client_to_upstream(client_ws: WebSocket, upstream_ws) -> None:
    while True:
        message = await client_ws.receive()

        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect()

        if "bytes" in message and message["bytes"] is not None:
            await upstream_ws.send(message["bytes"])
        elif "text" in message and message["text"] is not None:
            await upstream_ws.send(message["text"])


async def _upstream_to_client(client_ws: WebSocket, upstream_ws) -> None:
    while True:
        msg = await upstream_ws.recv()
        if isinstance(msg, bytes):
            await client_ws.send_bytes(msg)
        else:
            await client_ws.send_text(msg)

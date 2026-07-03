import asyncio
import json
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
    on_lease_acquired: Optional[Callable[[], Awaitable[None]]] = None,
) -> bool:
    """Proxy a client websocket to an upstream pipeline slot.

    Returns True if a lease was acquired (whether or not the session then
    succeeded) and False if the connection was rejected for lack of capacity.
    ``on_lease_acquired`` runs after capacity is secured but before the client
    websocket is accepted; if it raises, the lease is released and the client
    is closed without proxying.
    """
    lease = None

    try:
        lease = await acquire_lease(900.0)
    except Exception as exc:
        try:
            await client_ws.accept()
            await client_ws.send_text(json.dumps({
                "type": "error",
                "error": {
                    "type": "session_limit_reached",
                    "message": no_capacity_reason,
                },
            }))
        except Exception:
            pass
        try:
            await client_ws.close(code=1013, reason=no_capacity_reason[:123])
        except Exception:
            pass
        logger.warning("%s: %s", no_capacity_log, exc)
        return False

    if on_lease_acquired is not None:
        try:
            await on_lease_acquired()
        except Exception as exc:
            logger.error("Lease-acquired callback failed, closing session: %s", exc)
            try:
                await client_ws.close(code=1011, reason="Failed to establish reserved session")
            except Exception:
                pass
            await release_lease(lease.slot_id)
            return True

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

    return True


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

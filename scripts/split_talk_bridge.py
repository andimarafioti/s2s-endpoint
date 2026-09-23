#!/usr/bin/env python3
"""Bridge the packaged talk client to one protected split-fleet session.

The packaged client rejects query parameters in --url. Split compute workers
require a signed session_token query parameter in addition to HF ingress auth.
This loopback bridge obtains the signed URL from the split LB and relays one
WebSocket connection without storing credentials or media.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


def allocate_session(lb_url: str, hf_token: str) -> str:
    request = Request(
        f"{lb_url.rstrip('/')}/session",
        data=b"{}",
        headers={
            "Content-Type": "application/json",
            "X-Reachy-Mini-Authorization": f"Bearer {hf_token}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"Split LB session allocation returned HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Split LB session allocation failed: {exc.reason}") from exc
    if payload.get("state") != "granted" or not payload.get("connect_url"):
        raise RuntimeError(f"Split LB did not grant a session (state={payload.get('state')!r})")
    return str(payload["connect_url"])


async def relay(source, destination) -> None:
    try:
        async for message in source:
            await destination.send(message)
    except ConnectionClosed:
        pass


async def serve_client(client, lb_url: str, hf_token: str, active: asyncio.Lock) -> None:
    if client.request.path.split("?", 1)[0] != "/v1/realtime":
        await client.close(code=1008, reason="Expected /v1/realtime")
        return
    if active.locked():
        await client.close(code=1013, reason="A talk session is already active")
        return

    async with active:
        try:
            signed_url = await asyncio.to_thread(allocate_session, lb_url, hf_token)
            async with connect(
                signed_url,
                additional_headers={"Authorization": f"Bearer {hf_token}"},
                max_size=None,
                open_timeout=30,
            ) as worker:
                print("Split talk session connected", flush=True)
                tasks = [
                    asyncio.create_task(relay(client, worker)),
                    asyncio.create_task(relay(worker, client)),
                ]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for task in done:
                    task.result()
        except Exception as exc:
            print(f"Split talk bridge error: {type(exc).__name__}", file=sys.stderr, flush=True)
            await client.close(code=1011, reason="Split session could not be connected")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lb-url", required=True, help="Current split LB URL from the HF endpoint API")
    parser.add_argument("--port", type=int, default=8765, help="Local listening port (default: 8765)")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN must be set")
    active = asyncio.Lock()
    async with serve(
        lambda client: serve_client(client, args.lb_url, hf_token, active),
        "127.0.0.1",
        args.port,
        max_size=None,
    ):
        print(f"Split talk bridge ready at ws://127.0.0.1:{args.port}/v1/realtime", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

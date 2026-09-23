"""Allocator HTTP callbacks + compute proxy against a fault-injecting WS peer.

Inference is deliberately replaced by the peer; these tests cover settlement
and cleanup, not model quality or live provider availability.
"""

import asyncio
import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx
import websockets

from app import compute_app
from app.load_balancer_app import LoadBalancerSettings, build_load_balancer_dependencies, create_app
from app.session_router import SessionRouter
from app.session_routing_proxy import SessionRoutingProxy
from tests import test_pipeline_capacity as capacity_tests
from tests.test_ws_proxy import FakeClientWS, _dependencies


class RoutingLifecycleTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = capacity_tests.PipelineAdmissionTests.asyncSetUp
    manager_with_capacity = capacity_tests.PipelineAdmissionTests.manager_with_capacity

    async def exercise(self, failure=None):
        self.capacity.config = self.capacity.config.model_copy(update={"session_updates_enabled": True})
        manager = self.manager_with_capacity()
        base = LoadBalancerSettings(dashboard_preview_mode=True, session_shared_secret="secret")
        lb_dependencies = replace(build_load_balancer_dependencies(base), session_manager=manager)
        self.addAsyncCleanup(lb_dependencies.requester_identity_resolver.stop)
        settings = replace(
            base,
            pipeline_capacity=self.capacity.config,
            session_queue_enabled=True,
            speech_stt_proxy_url="https://stt",
            speech_llm_proxy_url="https://llm",
            speech_tts_proxy_url="https://tts",
            speech_capacity_api_key="capacity",
            lb_callback_auth_token="callback",
        )
        transport = httpx.ASGITransport(app=create_app(settings, lb_dependencies))
        calls = []
        forwarded = []
        unsettled = asyncio.Event()

        class Client(FakeClientWS):
            def __init__(self):
                super().__init__(headers={"host": "cpu.example"})
                self.incoming = asyncio.Queue()

            async def receive(self):
                return await self.incoming.get()

            async def send_text(self, text):
                await super().send_text(text)
                self.incoming.put_nowait({"type": "websocket.disconnect"})

        ws = Client()

        async def callback(request):
            payload = json.loads(request.content)
            action = payload.get("action", payload.get("event", "allocate"))
            calls.append(action)
            if action == "commit":
                if failure == "commit_rejected":
                    return httpx.Response(503)
                if failure == "disconnect_during_commit":
                    ws.incoming.put_nowait({"type": "websocket.disconnect"})
                    # The client pump is waiting for settlement. Its bounded
                    # acknowledgement wait must still cancel this callback.
                    await unsettled.wait()
            response = await transport.handle_async_request(request)
            await response.aread()
            if (action, failure) in {("prepare", "prepare_response_lost"), ("commit", "commit_response_lost")}:
                self.assertEqual(response.status_code, 200, response.text)
                raise httpx.ReadError("response lost after allocator applied update", request=request)
            return response

        http = httpx.AsyncClient(transport=httpx.MockTransport(callback), base_url="https://lb")
        self.addAsyncCleanup(http.aclose)
        allocation = await http.post("/session", json={"pipeline": "qwen"})
        self.assertEqual(allocation.status_code, 200, allocation.text)
        grant = allocation.json()
        ws.query_params = {"session_token": grant["session_token"]}
        ws.incoming.put_nowait(
            {
                "type": "websocket.receive",
                "text": json.dumps(
                    {"type": "session.update", "event_id": "change-stt", "session": {"models": {"stt": "stt-openai"}}}
                ),
            }
        )

        async def notify(url, token, event, **kwargs):
            response = await http.post(url, json={"session_token": token, "event": event})
            response.raise_for_status()

        async def upstream(connection):
            initial = json.loads(connection.request.headers["X-Speech-Session-Routing"])
            self.assertEqual(initial["id"], grant["session_id"])
            self.assertEqual(initial["routes"]["stt"]["model"], "stt-qwen")
            async for message in connection:
                proposal = json.loads(message)["_session_routing"]
                forwarded.append(proposal)
                self.assertEqual(proposal["routing"]["routes"]["stt"]["model"], "stt-openai")
                self.assertEqual(
                    manager.endpoint_router._pool_counts_unlocked(),
                    {("stt", "qwen"): 1, ("stt", "openai"): 1, ("llm", "shared"): 1, ("tts", "shared"): 1},
                )
                self.assertEqual(sum(compute_dependencies.route_sessions.values()), 1)
                if failure != "ack_lost":
                    await connection.send(
                        json.dumps(
                            {
                                "type": "session.updated",
                                "event_id": "updated",
                                "session": {"models": {"stt": "stt-openai"}},
                                "_session_routing": proposal["update_id"],
                            }
                        )
                    )

        async with websockets.serve(upstream, "127.0.0.1", 0) as server:
            router = SessionRouter(
                host="127.0.0.1",
                base_port=server.sockets[0].getsockname()[1],
                repo_dir="unused",
                build_command=lambda *args: [],
                wait_for_ready=lambda *args: None,
            )
            router._ready = True  # The peer replaces the child inference process.
            compute_dependencies = _dependencies(router, notify)
            real_wait_for = asyncio.wait_for

            async def bounded_wait(awaitable, timeout):
                return await real_wait_for(awaitable, 1 if timeout == 20 else timeout)

            def routing_proxy(*args):
                return SessionRoutingProxy(*args, client=http)

            with (
                patch.object(compute_app, "SessionRoutingProxy", side_effect=routing_proxy),
                patch("app.session_routing_proxy.asyncio.wait_for", side_effect=bounded_wait),
            ):
                await real_wait_for(
                    compute_app.websocket_proxy(
                        ws,
                        compute_app.ComputeSettings(session_shared_secret="secret", lb_callback_auth_token="callback"),
                        compute_dependencies,
                    ),
                    timeout=5,
                )

        self.assertEqual(calls[:3], ["allocate", "connected", "prepare"])
        self.assertEqual(calls[-1], "disconnected")
        self.assertEqual(len(forwarded), 0 if failure == "prepare_response_lost" else 1)
        self.assertEqual(calls.count("commit"), 0 if failure in {"prepare_response_lost", "ack_lost"} else 1)
        self.assertEqual(router._active_sessions, 0)
        self.assertEqual(compute_dependencies.route_sessions, {})
        self.assertEqual(manager._sessions, {})
        self.assertEqual(sum(manager.endpoint_router._pool_counts_unlocked().values()), 0)
        if failure:
            self.assertEqual(ws.sent, [])  # Never acknowledge an uncertain update.
            self.assertEqual(ws.close_calls[-1][0], 1011)
        else:
            self.assertEqual(len(ws.sent), 1)
            self.assertEqual(json.loads(ws.sent[0])["type"], "session.updated")
            self.assertNotIn("_session_routing", json.loads(ws.sent[0]))

    async def test_successful_update_is_acknowledged_after_settlement(self):
        await self.exercise()

    async def test_lost_prepare_response_releases_allocator_hold_on_disconnect(self):
        await self.exercise("prepare_response_lost")

    async def test_lost_upstream_acknowledgement_closes_and_releases_both_selections(self):
        await self.exercise("ack_lost")

    async def test_failed_commit_closes_and_releases_both_selections(self):
        await self.exercise("commit_rejected")

    async def test_lost_commit_response_releases_committed_selection(self):
        await self.exercise("commit_response_lost")

    async def test_client_disconnect_during_commit_cannot_leave_a_capacity_hold(self):
        await self.exercise("disconnect_during_commit")

import unittest

from tests import test_pipeline_capacity as capacity_tests


class SessionModelSelectionTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = capacity_tests.PipelineAdmissionTests.asyncSetUp
    manager_with_capacity = capacity_tests.PipelineAdmissionTests.manager_with_capacity

    def enable_updates(self):
        self.capacity.config = self.capacity.config.model_copy(update={"session_updates_enabled": True})

    async def test_empty_admission_only_spends_cpu_capacity(self):
        self.enable_updates()
        self.capacity._views.clear()
        self.capacity._seen.clear()
        manager = self.manager_with_capacity()
        grant = await manager.allocate("https://allocator.example")
        self.assertEqual(grant["state"], "granted")
        self.assertEqual(grant["routing"]["routes"], {"stt": None, "llm": None, "tts": None})
        self.assertTrue(grant["routing"]["updates_enabled"])
        self.assertEqual(
            sum(self.capacity.pool_counts(manager.endpoint_router._pipeline_counts_unlocked()).values()), 0
        )

    async def test_select_and_remove_stages_preserves_unspecified_choices(self):
        self.enable_updates()
        selected = self.capacity.select_models(None, {"stt": "stt-qwen", "llm": "llm-shared"})
        self.assertEqual(self.capacity.routing(selected)["routes"]["stt"]["model"], "stt-qwen")
        changed = self.capacity.select_models(
            selected, {"stt": None, "tts": {"model": "tts-shared", "provider": "shared"}}
        )
        routes = self.capacity.routing(changed)["routes"]
        self.assertIsNone(routes["stt"])
        self.assertEqual(routes["llm"]["model"], "llm-shared")
        self.assertEqual(routes["tts"]["model"], "tts-shared")
        for invalid in ({"vad": None}, {"stt": "unknown"}, {"stt": {"model": "stt-qwen", "provider": "other"}}):
            with self.assertRaises(ValueError):
                self.capacity.select_models(selected, invalid)

    async def test_switch_reserves_only_added_pools_and_union_survives_health_reconstruction(self):
        self.enable_updates()
        counts = {"qwen": 7}
        self.assertTrue(self.capacity.can_switch("qwen", "openai", counts))
        union = self.capacity.hold_selection("qwen", "openai")
        demand = self.capacity.pool_counts({union: 1})
        self.assertEqual(demand, {("stt", "qwen"): 1, ("stt", "openai"): 1, ("llm", "shared"): 1, ("tts", "shared"): 1})
        self.assertFalse(self.capacity.can_switch("qwen", "openai", {"openai": 7, "qwen": 1}))
        self.assertTrue(self.capacity.can_switch("qwen", self.capacity.select_models("qwen", {"stt": None}), counts))

    async def test_legacy_admission_still_uses_the_default_complete_route(self):
        self.assertEqual(self.capacity.resolve(None), "qwen")
        with self.assertRaises(ValueError):
            self.capacity.select_models(None, {"stt": None})

    async def test_route_update_holds_old_and_new_until_ack_then_releases_on_disconnect(self):
        self.enable_updates()
        manager = self.manager_with_capacity()
        grant = await manager.allocate("https://allocator.example", pipeline="qwen")
        sid, token = grant["session_id"], grant["session_token"]
        await manager.handle_event(sid, token, "connected")
        prepared = await manager.prepare_routing(sid, token, "change-1", {"stt": "stt-openai"})
        counts = self.capacity.pool_counts(manager.endpoint_router._pipeline_counts_unlocked())
        self.assertEqual(counts[("stt", "qwen")], 1)
        self.assertEqual(counts[("stt", "openai")], 1)
        self.assertEqual(counts[("llm", "shared")], 1)
        self.assertEqual(prepared, await manager.prepare_routing(sid, token, "change-1", {"stt": "stt-openai"}))
        with self.assertRaises(ValueError):
            await manager.prepare_routing(sid, token, "change-2", {"tts": None})
        await manager.finish_routing(sid, token, "change-1", accepted=True)
        await manager.finish_routing(sid, token, "change-1", accepted=True)
        counts = self.capacity.pool_counts(manager.endpoint_router._pipeline_counts_unlocked())
        self.assertEqual(counts[("stt", "qwen")], 0)
        self.assertEqual(counts[("stt", "openai")], 1)
        await manager.handle_event(sid, token, "disconnected")
        self.assertEqual(
            sum(self.capacity.pool_counts(manager.endpoint_router._pipeline_counts_unlocked()).values()), 0
        )

    async def test_rejected_update_restores_old_selection_and_disconnect_releases_pending_hold(self):
        self.enable_updates()
        manager = self.manager_with_capacity()
        grant = await manager.allocate("https://allocator.example", pipeline="qwen")
        sid, token = grant["session_id"], grant["session_token"]
        await manager.handle_event(sid, token, "connected")
        with self.assertRaises(ValueError):
            await manager.prepare_routing(sid, "wrong-token", "change-1", {"stt": None})
        await manager.prepare_routing(sid, token, "change-1", {"stt": "stt-openai"})
        await manager.finish_routing(sid, token, "change-1", accepted=False)
        self.assertEqual(manager._sessions[sid].lease.pipeline, "qwen")
        await manager.prepare_routing(sid, token, "change-2", {"stt": "stt-openai"})
        await manager.handle_event(sid, token, "disconnected")
        self.assertEqual(
            sum(self.capacity.pool_counts(manager.endpoint_router._pipeline_counts_unlocked()).values()), 0
        )

    async def test_connected_session_can_update_and_release_after_admission_token_expiry(self):
        from unittest.mock import patch

        self.enable_updates()
        manager = self.manager_with_capacity()
        with patch("app.session_tokens.time.time", return_value=1000):
            grant = await manager.allocate("https://allocator.example", pipeline="qwen")
            sid, token = grant["session_id"], grant["session_token"]
            await manager.handle_event(sid, token, "connected")
        with patch("app.session_tokens.time.time", return_value=1000000):
            await manager.prepare_routing(sid, token, "after-expiry", {"stt": "stt-openai"})
            await manager.handle_event(sid, token, "disconnected")
        self.assertEqual(
            sum(self.capacity.pool_counts(manager.endpoint_router._pipeline_counts_unlocked()).values()), 0
        )

    async def test_http_session_selection_and_private_callback_authentication(self):
        from dataclasses import replace

        import httpx

        from app.load_balancer_app import LoadBalancerSettings, build_load_balancer_dependencies, create_app
        from app.session_tokens import verify_session_token

        self.enable_updates()
        manager = self.manager_with_capacity()
        base = LoadBalancerSettings(dashboard_preview_mode=True, session_shared_secret="secret")
        dependencies = replace(build_load_balancer_dependencies(base), session_manager=manager)
        self.addAsyncCleanup(dependencies.requester_identity_resolver.stop)
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
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings, dependencies)), base_url="https://lb"
        )
        self.addAsyncCleanup(client.aclose)
        empty = await client.post("/session", json={})
        self.assertEqual(empty.status_code, 200, empty.text)
        self.assertEqual(
            verify_session_token(empty.json()["session_token"], "secret")["routing"]["routes"],
            {"stt": None, "llm": None, "tts": None},
        )
        selected = await client.post("/session", json={"models": {"stt": "stt-qwen"}})
        self.assertEqual(selected.status_code, 200, selected.text)
        grant = selected.json()
        self.assertEqual(
            verify_session_token(grant["session_token"], "secret")["routing"]["routes"]["stt"]["model"], "stt-qwen"
        )
        sid = grant["session_id"]
        token = manager._sessions[sid].session_token
        await manager.handle_event(sid, token, "connected")
        path = f"/internal/sessions/{sid}/routing"
        payload = {"session_token": token, "action": "prepare", "update_id": "private", "models": {"stt": None}}
        self.assertIn((await client.post(path, json=payload)).status_code, (401, 403))
        wrong = await client.post(
            path,
            json=payload,
            headers={"Authorization": "Bearer callback", "X-Reachy-Mini-Callback-Authorization": "Bearer wrong"},
        )
        self.assertIn(wrong.status_code, (401, 403))
        accepted = await client.post(
            path,
            json=payload,
            headers={"Authorization": "Bearer ingress", "X-Reachy-Mini-Callback-Authorization": "Bearer callback"},
        )
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertIsNone(accepted.json()["routing"]["routes"]["stt"])

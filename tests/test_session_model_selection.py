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

import json
import unittest

from app.pipeline_turn_latency import (
    TURN_LATENCY_METADATA_KEY,
    PipelineTurnLatency,
    PipelineTurnLatencyMetrics,
    extract_pipeline_turn_latency,
)


def _payload(**overrides):
    payload = {
        "version": 1,
        "turn_id": "turn_3",
        "turn_revision": 2,
        "response_key": "response_4",
        "status": "completed",
        "stt_s": 0.181284,
        "llm_ttft_s": 0.214567,
        "llm_s": 1.241907,
        "tts_ttfa_s": 0.121775,
        "e2e_s": 1.613482,
        "mlx_lock_wait_s": 0.003456,
    }
    payload.update(overrides)
    return payload


class PipelineTurnLatencyParsingTests(unittest.TestCase):
    def test_extracts_raw_terminal_response_metadata(self):
        event = {
            "type": "response.done",
            "response": {"metadata": {TURN_LATENCY_METADATA_KEY: json.dumps(_payload())}},
        }

        latency = extract_pipeline_turn_latency(json.dumps(event))

        self.assertIsNotNone(latency)
        self.assertEqual(latency.response_key, "response_4")
        self.assertEqual(latency.e2e_s, 1.613482)
        self.assertEqual(latency.to_payload(), _payload())

    def test_ignores_non_terminal_and_unmeasured_events(self):
        self.assertIsNone(extract_pipeline_turn_latency('{"type":"response.created"}'))
        self.assertIsNone(extract_pipeline_turn_latency('{"type":"response.done","response":{"metadata":null}}'))
        self.assertIsNone(extract_pipeline_turn_latency("not json"))

    def test_accepts_v1_metadata_without_additive_llm_ttft(self):
        payload = _payload()
        payload.pop("llm_ttft_s")

        latency = PipelineTurnLatency.from_payload(payload)

        self.assertIsNone(latency.llm_ttft_s)

    def test_rejects_invalid_server_measurement(self):
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            PipelineTurnLatency.from_payload(_payload(e2e_s=float("inf")))
        with self.assertRaisesRegex(ValueError, "status"):
            PipelineTurnLatency.from_payload(_payload(status="running"))
        with self.assertRaisesRegex(ValueError, "version"):
            PipelineTurnLatency.from_payload(_payload(version=2))


class PipelineTurnLatencyMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_summarizes_selected_window_and_deduplicates_callbacks(self):
        clock = [100.0]
        metrics = PipelineTurnLatencyMetrics(max_samples=3, time_fn=lambda: clock[0])
        first = PipelineTurnLatency.from_payload(_payload())
        second = PipelineTurnLatency.from_payload(
            _payload(
                response_key="response_5",
                status="failed",
                stt_s=0.3,
                llm_s=2.0,
                tts_ttfa_s=None,
                e2e_s=None,
            )
        )

        self.assertTrue(await metrics.record("session_1", first))
        self.assertFalse(await metrics.record("session_1", first))
        clock[0] = 110.0
        self.assertTrue(await metrics.record("session_1", second))

        snapshot = await metrics.snapshot(20.0)

        self.assertEqual(snapshot["responses"]["window"], 2)
        self.assertEqual(snapshot["responses"]["completed"], 1)
        self.assertEqual(snapshot["responses"]["failed"], 1)
        self.assertEqual(snapshot["responses"]["duplicates_ignored_lifetime"], 1)
        self.assertEqual(snapshot["latency_ms"]["stt"]["p50"], 240.642)
        self.assertEqual(snapshot["latency_ms"]["llm_ttft"]["p50"], 214.567)
        self.assertEqual(snapshot["latency_ms"]["e2e"]["n"], 1)
        self.assertEqual(snapshot["latency_ms"]["e2e"]["p95"], 1613.482)

    async def test_evicts_deduplication_key_with_bounded_sample(self):
        metrics = PipelineTurnLatencyMetrics(max_samples=1)
        first = PipelineTurnLatency.from_payload(_payload())
        second = PipelineTurnLatency.from_payload(_payload(response_key="response_5"))

        await metrics.record("session_1", first)
        await metrics.record("session_1", second)

        self.assertTrue(await metrics.record("session_1", first))


if __name__ == "__main__":
    unittest.main()

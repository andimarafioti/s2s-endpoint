import unittest

from app.speech_proxy_metrics import SpeechProxyMetrics, SpeechRequestTrace, sample_headers


class MutableClock:
    def __init__(self, value: float = 0.0):
        self.value = value

    def now(self):
        return self.value


class SpeechProxyMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_trace_decomposes_proxy_transport_and_service_latency(self):
        monotonic = MutableClock()
        wall = MutableClock(100.0)
        metrics = SpeechProxyMetrics("stt", time_fn=wall.now)
        trace = SpeechRequestTrace(
            metrics,
            request_id="request-1",
            monotonic_fn=monotonic.now,
            wall_time_fn=wall.now,
        )

        monotonic.value = 0.020
        trace.start_upstream("stt-01")
        monotonic.value = 0.120
        trace.finish_upstream("80")
        monotonic.value = 0.130
        sample = await trace.record("success")

        self.assertAlmostEqual(sample.total_ms, 130)
        self.assertAlmostEqual(sample.proxy_application_ms, 30)
        self.assertAlmostEqual(sample.backend_round_trip_ms, 100)
        self.assertAlmostEqual(sample.backend_service_ms or 0, 80)
        self.assertAlmostEqual(sample.backend_transport_ms or 0, 20)
        self.assertAlmostEqual(sample.proxy_path_overhead_ms or 0, 50)

        snapshot = await metrics.snapshot(300)
        self.assertEqual(snapshot["requests"]["successes"], 1)
        self.assertEqual(snapshot["latency_ms"]["proxy_application"]["p50"], 30)
        self.assertEqual(snapshot["latency_ms"]["backend_service"]["p95"], 80)
        self.assertEqual(snapshot["latency_ms"]["proxy_path_overhead"]["p50"], 50)
        self.assertEqual(snapshot["service_timing_coverage"]["ratio"], 1)
        self.assertIn("speech-service;dur=80.000", sample_headers(sample)["Server-Timing"])

    async def test_missing_backend_header_keeps_service_and_transport_unknown(self):
        monotonic = MutableClock()
        wall = MutableClock(100.0)
        metrics = SpeechProxyMetrics("tts", time_fn=wall.now)
        trace = SpeechRequestTrace(metrics, monotonic_fn=monotonic.now, wall_time_fn=wall.now)

        trace.start_upstream("tts-01")
        monotonic.value = 0.250
        trace.finish_upstream()
        sample = await trace.record("success")
        snapshot = await metrics.snapshot(300)

        self.assertIsNone(sample.backend_service_ms)
        self.assertIsNone(sample.backend_transport_ms)
        self.assertIsNone(sample.proxy_path_overhead_ms)
        self.assertEqual(snapshot["latency_ms"]["backend_service"], {"n": 0})
        self.assertEqual(snapshot["service_timing_coverage"]["ratio"], 0)


if __name__ == "__main__":
    unittest.main()

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_speech_service_endpoints import (  # noqa: E402
    metric_delta,
    pcm_duration_s,
    pcm_to_wav,
    percentile,
    prometheus_value,
    repeat_pcm,
    sample_stt_live_metrics,
    server_metric_summary,
    wait_until_epoch,
)


class AudioFixtureTests(unittest.TestCase):
    def test_repeat_pcm_creates_exact_requested_duration(self):
        pcm = b"\x01\x02" * 100

        repeated = repeat_pcm(pcm, 2.0, sample_rate=100)

        self.assertEqual(pcm_duration_s(repeated, sample_rate=100), 2.0)
        self.assertEqual(len(repeated), 400)

    def test_pcm_to_wav_writes_mono_24khz_header(self):
        wav = pcm_to_wav(b"\x00\x00" * 24_000)

        self.assertEqual(wav[:4], b"RIFF")
        self.assertEqual(wav[8:12], b"WAVE")
        self.assertGreater(len(wav), 48_000)


class StatisticsTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 0.95), 1.95)


class PrometheusTests(unittest.TestCase):
    METRICS = """
vllm:e2e_request_latency_seconds_count{engine="0",model_name="model"} 4.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="model"} 1.2
http_request_duration_seconds_count{handler="/v1/audio/speech",method="POST"} 3.0
http_request_duration_seconds_count{handler="/health",method="GET"} 10.0
"""

    def test_prometheus_value_filters_labels(self):
        value = prometheus_value(
            self.METRICS,
            "http_request_duration_seconds_count",
            {"handler": "/v1/audio/speech", "method": "POST"},
        )

        self.assertEqual(value, 3.0)

    def test_metric_delta_handles_missing_values(self):
        self.assertEqual(metric_delta({"x": 2.0}, {"x": 5.0}, "x"), 3.0)
        self.assertIsNone(metric_delta({"x": None}, {"x": 5.0}, "x"))

    def test_server_metric_summary_calculates_stt_averages(self):
        before = {
            "vllm:e2e_request_latency_seconds_count": 10.0,
            "vllm:e2e_request_latency_seconds_sum": 2.0,
            "vllm:request_queue_time_seconds_count": 10.0,
            "vllm:request_queue_time_seconds_sum": 0.1,
            "vllm:time_to_first_token_seconds_count": 10.0,
            "vllm:time_to_first_token_seconds_sum": 0.4,
        }
        after = {
            "vllm:e2e_request_latency_seconds_count": 14.0,
            "vllm:e2e_request_latency_seconds_sum": 3.2,
            "vllm:request_queue_time_seconds_count": 14.0,
            "vllm:request_queue_time_seconds_sum": 0.3,
            "vllm:time_to_first_token_seconds_count": 14.0,
            "vllm:time_to_first_token_seconds_sum": 0.6,
        }

        summary = server_metric_summary(before, after, service="stt")

        self.assertEqual(summary["observed_requests"], 4.0)
        self.assertEqual(summary["mean_e2e_s"], 0.3)
        self.assertEqual(summary["mean_queue_s"], 0.05)
        self.assertEqual(summary["mean_ttft_s"], 0.05)


class LiveMetricSamplerTests(unittest.IsolatedAsyncioTestCase):
    @patch("benchmark_speech_service_endpoints.fetch_metrics", new_callable=AsyncMock)
    async def test_sampler_records_peak_running_and_waiting(self, fetch_metrics):
        responses = iter(
            [
                'vllm:num_requests_running{engine="0"} 4\nvllm:num_requests_waiting{engine="0"} 2',
                'vllm:num_requests_running{engine="0"} 8\nvllm:num_requests_waiting{engine="0"} 1',
            ]
        )
        stop = asyncio.Event()

        async def next_metrics(*_args):
            response = next(responses)
            if fetch_metrics.await_count == 2:
                stop.set()
            return response

        fetch_metrics.side_effect = next_metrics
        summary = await sample_stt_live_metrics(AsyncMock(), "https://example.test", stop, interval_s=0)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["peak_running"], 8.0)
        self.assertEqual(summary["peak_waiting"], 2.0)
        self.assertEqual(summary["errors"], [])

    async def test_wait_until_epoch_sleeps_only_for_future_start(self):
        with (
            patch("benchmark_speech_service_endpoints.time.time", return_value=100.0),
            patch("benchmark_speech_service_endpoints.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            await wait_until_epoch(103.5)
            sleep.assert_awaited_once_with(3.5)

        with patch("benchmark_speech_service_endpoints.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await wait_until_epoch(None)
            sleep.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

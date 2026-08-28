#!/usr/bin/env python3
"""Benchmark the deployed OpenAI-compatible STT and TTS services directly."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import re
import statistics
import time
import wave
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_STT_BASE_URL = "https://go3quisjv5ta7203.us-east-1.aws.endpoints.huggingface.cloud"
DEFAULT_TTS_BASE_URL = "https://db6lx9j3kdymwu9w.us-east-1.aws.endpoints.huggingface.cloud"
DEFAULT_STT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
PCM_SAMPLE_RATE = 24_000
PCM_SAMPLE_WIDTH = 2

TTS_CASES = {
    "one_sentence": "The quick brown fox crosses the quiet garden while the evening light fades behind the trees.",
    "three_sentences": (
        "The quick brown fox crosses the quiet garden while the evening light fades behind the trees. "
        "A small robot watches from the doorway and records the changing colors. "
        "When the path is clear, they continue together toward the station."
    ),
    "six_sentences": (
        "The quick brown fox crosses the quiet garden while the evening light fades behind the trees. "
        "A small robot watches from the doorway and records the changing colors. "
        "When the path is clear, they continue together toward the station. "
        "The platform is almost empty, but a distant train can already be heard. "
        "They compare their maps and choose a route through the mountains. "
        "By morning, both travelers expect to reach the coast."
    ),
}

FIXTURE_TEXT = (
    "This controlled recording is used to measure speech recognition throughput. "
    "It contains ordinary English words, short pauses, and several complete sentences. "
    "The benchmark repeats the recording when a longer audio sample is required. "
    "Consistent input makes concurrency results easier to compare across test runs."
)


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    status_code: int | None
    total_s: float
    first_byte_s: float | None = None
    response_bytes: int = 0
    output_audio_s: float | None = None
    transcript_chars: int | None = None
    error: str | None = None


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def pcm_duration_s(pcm: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> float:
    return len(pcm) / (sample_rate * PCM_SAMPLE_WIDTH)


def repeat_pcm(pcm: bytes, duration_s: float, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    if not pcm:
        raise ValueError("PCM fixture cannot be empty")
    target_bytes = int(duration_s * sample_rate) * PCM_SAMPLE_WIDTH
    repetitions = math.ceil(target_bytes / len(pcm))
    return (pcm * repetitions)[:target_bytes]


def pcm_to_wav(pcm: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(PCM_SAMPLE_WIDTH)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


_METRIC_PATTERN = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+(?P<value>\S+)$")


def prometheus_value(text: str, name: str, required_labels: dict[str, str] | None = None) -> float | None:
    required_labels = required_labels or {}
    total = 0.0
    matched = False
    for line in text.splitlines():
        match = _METRIC_PATTERN.match(line)
        if not match or match.group("name") != name:
            continue
        labels = match.group("labels") or ""
        if any(f'{key}="{value}"' not in labels for key, value in required_labels.items()):
            continue
        total += float(match.group("value"))
        matched = True
    return total if matched else None


def metric_delta(before: dict[str, float | None], after: dict[str, float | None], name: str) -> float | None:
    before_value = before.get(name)
    after_value = after.get(name)
    if before_value is None or after_value is None:
        return None
    return after_value - before_value


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def summarize_results(
    results: Sequence[RequestResult],
    *,
    wall_s: float,
    input_audio_s: float | None = None,
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    totals = [result.total_s for result in successful]
    first_bytes = [result.first_byte_s for result in successful if result.first_byte_s is not None]
    output_audio = [result.output_audio_s for result in successful if result.output_audio_s is not None]
    summary: dict[str, Any] = {
        "requests": len(results),
        "successes": len(successful),
        "errors": len(results) - len(successful),
        "wall_s": rounded(wall_s),
        "requests_per_s": rounded(len(successful) / wall_s if wall_s > 0 else None),
        "latency_p50_s": rounded(percentile(totals, 0.5)),
        "latency_p95_s": rounded(percentile(totals, 0.95)),
        "latency_max_s": rounded(max(totals) if totals else None),
    }
    if first_bytes:
        summary.update(
            {
                "first_byte_p50_s": rounded(percentile(first_bytes, 0.5)),
                "first_byte_p95_s": rounded(percentile(first_bytes, 0.95)),
            }
        )
    if output_audio:
        total_output_audio_s = sum(output_audio)
        summary.update(
            {
                "mean_output_audio_s": rounded(statistics.mean(output_audio)),
                "output_audio_s_per_wall_s": rounded(total_output_audio_s / wall_s if wall_s > 0 else None),
            }
        )
    if input_audio_s is not None and totals:
        summary["latency_to_input_audio_ratio_p50"] = rounded(percentile(totals, 0.5) / input_audio_s)
    errors = [result.error for result in results if result.error]
    if errors:
        summary["error_samples"] = errors[:3]
    return summary


async def run_load_cell(
    request: Callable[[], Awaitable[RequestResult]],
    *,
    concurrency: int,
    waves: int,
) -> tuple[list[RequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request() -> RequestResult:
        async with semaphore:
            return await request()

    started = time.perf_counter()
    results = await asyncio.gather(*(bounded_request() for _ in range(concurrency * waves)))
    return list(results), time.perf_counter() - started


async def fetch_metrics(client: httpx.AsyncClient, base_url: str) -> str:
    response = await client.get(f"{base_url}/metrics")
    response.raise_for_status()
    return response.text


async def sample_stt_live_metrics(
    client: httpx.AsyncClient,
    base_url: str,
    stop: asyncio.Event,
    *,
    interval_s: float = 0.05,
) -> dict[str, Any]:
    peak_running = 0.0
    peak_waiting = 0.0
    samples = 0
    errors: list[str] = []
    while True:
        try:
            text = await fetch_metrics(client, base_url)
            running = prometheus_value(text, "vllm:num_requests_running") or 0.0
            waiting = prometheus_value(text, "vllm:num_requests_waiting") or 0.0
            peak_running = max(peak_running, running)
            peak_waiting = max(peak_waiting, waiting)
            samples += 1
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if stop.is_set():
            break
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            pass
    return {
        "samples": samples,
        "peak_running": rounded(peak_running),
        "peak_waiting": rounded(peak_waiting),
        "errors": errors[:3],
    }


async def wait_until_epoch(start_at_epoch: float | None) -> None:
    if start_at_epoch is None:
        return
    delay_s = start_at_epoch - time.time()
    if delay_s > 0:
        print(f"Waiting {delay_s:.1f}s for the synchronized STT start", flush=True)
        await asyncio.sleep(delay_s)


def stt_metric_snapshot(text: str) -> dict[str, float | None]:
    names = (
        "vllm:e2e_request_latency_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:request_queue_time_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_to_first_token_seconds_sum",
    )
    return {name: prometheus_value(text, name) for name in names}


def tts_metric_snapshot(text: str) -> dict[str, float | None]:
    labels = {"handler": "/v1/audio/speech", "method": "POST"}
    names = ("http_request_duration_seconds_count", "http_request_duration_seconds_sum")
    return {name: prometheus_value(text, name, labels) for name in names}


def server_metric_summary(
    before: dict[str, float | None],
    after: dict[str, float | None],
    *,
    service: str,
) -> dict[str, float | None]:
    if service == "stt":
        count = metric_delta(before, after, "vllm:e2e_request_latency_seconds_count")
        e2e_sum = metric_delta(before, after, "vllm:e2e_request_latency_seconds_sum")
        queue_count = metric_delta(before, after, "vllm:request_queue_time_seconds_count")
        queue_sum = metric_delta(before, after, "vllm:request_queue_time_seconds_sum")
        ttft_count = metric_delta(before, after, "vllm:time_to_first_token_seconds_count")
        ttft_sum = metric_delta(before, after, "vllm:time_to_first_token_seconds_sum")
        return {
            "observed_requests": rounded(count),
            "mean_e2e_s": rounded(ratio(e2e_sum, count)),
            "mean_queue_s": rounded(ratio(queue_sum, queue_count)),
            "mean_ttft_s": rounded(ratio(ttft_sum, ttft_count)),
        }

    count = metric_delta(before, after, "http_request_duration_seconds_count")
    duration_sum = metric_delta(before, after, "http_request_duration_seconds_sum")
    return {
        "observed_requests": rounded(count),
        "mean_request_s": rounded(ratio(duration_sum, count)),
    }


async def request_tts(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    text: str,
) -> tuple[RequestResult, bytes]:
    payload = {
        "model": args.tts_model,
        "input": text,
        "voice": args.voice,
        "language": args.language,
        "response_format": "pcm",
        "stream": True,
    }
    started = time.perf_counter()
    pcm_parts: list[bytes] = []
    first_byte_s = None
    try:
        async with client.stream("POST", f"{args.tts_base_url}/v1/audio/speech", json=payload) as response:
            status_code = response.status_code
            if response.is_error:
                body = (await response.aread()).decode("utf-8", errors="replace")[:300]
                return (
                    RequestResult(
                        ok=False,
                        status_code=status_code,
                        total_s=time.perf_counter() - started,
                        error=f"HTTP {status_code}: {body}",
                    ),
                    b"",
                )
            async for chunk in response.aiter_bytes():
                if chunk and first_byte_s is None:
                    first_byte_s = time.perf_counter() - started
                pcm_parts.append(chunk)
        pcm = b"".join(pcm_parts)
        total_s = time.perf_counter() - started
        return (
            RequestResult(
                ok=True,
                status_code=status_code,
                total_s=total_s,
                first_byte_s=first_byte_s,
                response_bytes=len(pcm),
                output_audio_s=pcm_duration_s(pcm),
            ),
            pcm,
        )
    except Exception as exc:
        return (
            RequestResult(
                ok=False,
                status_code=None,
                total_s=time.perf_counter() - started,
                error=f"{type(exc).__name__}: {exc}",
            ),
            b"",
        )


async def request_stt(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    wav_bytes: bytes,
    duration_s: float,
) -> RequestResult:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{args.stt_base_url}/v1/audio/transcriptions",
            data={"model": args.stt_model, "response_format": "json"},
            files={"file": (f"benchmark-{duration_s:g}s.wav", wav_bytes, "audio/wav")},
        )
        total_s = time.perf_counter() - started
        if response.is_error:
            return RequestResult(
                ok=False,
                status_code=response.status_code,
                total_s=total_s,
                error=f"HTTP {response.status_code}: {response.text[:300]}",
            )
        payload = response.json()
        transcript = payload.get("text", "") if isinstance(payload, dict) else ""
        return RequestResult(
            ok=True,
            status_code=response.status_code,
            total_s=total_s,
            response_bytes=len(response.content),
            transcript_chars=len(transcript),
        )
    except Exception as exc:
        return RequestResult(
            ok=False,
            status_code=None,
            total_s=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


async def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("HF_TOKEN is required")
    headers = {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=max(args.concurrencies) + 8, max_keepalive_connections=max(args.concurrencies) + 8
    )
    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "stt_base_url": args.stt_base_url,
            "tts_base_url": args.tts_base_url,
            "stt_model": args.stt_model,
            "tts_model": args.tts_model,
            "concurrencies": args.concurrencies,
            "waves": args.waves,
            "stt_durations_s": args.stt_durations,
            "tts_cases": args.tts_cases,
            "client_region": args.client_region,
        },
        "tts": [],
        "stt": [],
    }

    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        fixture_result, fixture_pcm = await request_tts(client, args, FIXTURE_TEXT)
        if not fixture_result.ok:
            raise RuntimeError(f"Could not synthesize the STT fixture: {fixture_result.error}")
        report["fixture"] = {
            "source_audio_s": rounded(pcm_duration_s(fixture_pcm)),
            "synthesis_s": rounded(fixture_result.total_s),
        }

        if "tts" in args.services:
            for case_name in args.tts_cases:
                text = TTS_CASES[case_name]
                warmup, _ = await request_tts(client, args, text)
                if not warmup.ok:
                    raise RuntimeError(f"TTS warmup failed for {case_name}: {warmup.error}")
                for concurrency in args.concurrencies:
                    before = tts_metric_snapshot(await fetch_metrics(client, args.tts_base_url))

                    async def tts_request(text: str = text) -> RequestResult:
                        result, _ = await request_tts(client, args, text)
                        return result

                    results, wall_s = await run_load_cell(
                        tts_request,
                        concurrency=concurrency,
                        waves=args.waves,
                    )
                    after = tts_metric_snapshot(await fetch_metrics(client, args.tts_base_url))
                    cell = {
                        "case": case_name,
                        "text_chars": len(text),
                        "concurrency": concurrency,
                        **summarize_results(results, wall_s=wall_s),
                        "server_metrics": server_metric_summary(before, after, service="tts"),
                    }
                    report["tts"].append(cell)
                    print(json.dumps({"service": "tts", **cell}), flush=True)

        if "stt" in args.services:
            fixtures = {
                duration_s: pcm_to_wav(repeat_pcm(fixture_pcm, duration_s)) for duration_s in args.stt_durations
            }
            warmup_duration = min(args.stt_durations)
            warmup = await request_stt(client, args, fixtures[warmup_duration], warmup_duration)
            if not warmup.ok:
                raise RuntimeError(f"STT warmup failed: {warmup.error}")
            await wait_until_epoch(args.stt_start_at_epoch)
            for duration_s, wav_bytes in fixtures.items():
                for concurrency in args.concurrencies:
                    before = stt_metric_snapshot(await fetch_metrics(client, args.stt_base_url))
                    sampler_stop = asyncio.Event()
                    sampler = asyncio.create_task(sample_stt_live_metrics(client, args.stt_base_url, sampler_stop))

                    async def stt_request(
                        wav_bytes: bytes = wav_bytes, duration_s: float = duration_s
                    ) -> RequestResult:
                        return await request_stt(client, args, wav_bytes, duration_s)

                    try:
                        results, wall_s = await run_load_cell(
                            stt_request,
                            concurrency=concurrency,
                            waves=args.waves,
                        )
                    finally:
                        sampler_stop.set()
                    live_metrics = await sampler
                    after = stt_metric_snapshot(await fetch_metrics(client, args.stt_base_url))
                    cell = {
                        "audio_s": duration_s,
                        "concurrency": concurrency,
                        **summarize_results(results, wall_s=wall_s, input_audio_s=duration_s),
                        "server_metrics": server_metric_summary(before, after, service="stt"),
                        "live_metrics": live_metrics,
                    }
                    report["stt"].append(cell)
                    print(json.dumps({"service": "stt", **cell}), flush=True)

    report["completed_at"] = datetime.now(UTC).isoformat()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--services", nargs="+", choices=("stt", "tts"), default=["tts", "stt"])
    parser.add_argument("--stt-base-url", default=DEFAULT_STT_BASE_URL)
    parser.add_argument("--tts-base-url", default=DEFAULT_TTS_BASE_URL)
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL)
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    parser.add_argument("--voice", default="aiden")
    parser.add_argument("--language", default="English")
    parser.add_argument(
        "--tts-cases",
        nargs="+",
        choices=tuple(TTS_CASES),
        default=list(TTS_CASES),
        help="TTS text cases to benchmark",
    )
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--stt-durations", nargs="+", type=float, default=[2, 5, 15, 30])
    parser.add_argument(
        "--stt-start-at-epoch",
        type=float,
        help="Wait until this Unix timestamp after STT warmup, for synchronized distributed load",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--client-region", default="local", help="Label recorded in the benchmark report")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(concurrency < 1 for concurrency in args.concurrencies):
        parser.error("--concurrencies values must be >= 1")
    if args.waves < 1:
        parser.error("--waves must be >= 1")
    if any(duration <= 0 for duration in args.stt_durations):
        parser.error("--stt-durations values must be > 0")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    return args


def main() -> None:
    args = parse_args()
    report = asyncio.run(benchmark(args))
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
        print(f"Wrote benchmark report to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

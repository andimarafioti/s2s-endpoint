#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
from dataclasses import asdict, dataclass
from typing import Any

from _endpoint_helpers import DEFAULT_FRAMEWORK, build_custom_image
from huggingface_hub import HfApi

DEFAULT_NAMESPACE = "HuggingFaceM4"
DEFAULT_VENDOR = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_INSTANCE_SIZE = "x1"
DEFAULT_INSTANCE_TYPE = "nvidia-a10g"
DEFAULT_ENDPOINT_TYPE = "protected"
DEFAULT_STT_NAME = "reachy-s2s-stt-01"
DEFAULT_TTS_NAME = "reachy-s2s-tts-01"
DEFAULT_STT_REPOSITORY = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_TTS_REPOSITORY = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_STT_IMAGE = "andito/s2s-stt:v0.1"
DEFAULT_TTS_IMAGE = "andito/s2s-tts:v0.1"


@dataclass(frozen=True)
class SpeechServiceSpec:
    service: str
    name: str
    repository: str
    revision: str
    image_url: str
    port: int
    health_route: str = "/health"


def resolve_revision(api: HfApi, repository: str, revision: str | None) -> str:
    if revision:
        return revision
    resolved = api.model_info(repository).sha
    if not resolved:
        raise ValueError(f"Could not resolve a revision for {repository}")
    return resolved


def build_specs(args: argparse.Namespace, api: HfApi) -> list[SpeechServiceSpec]:
    requested_services = set(args.services)
    specs: list[SpeechServiceSpec] = []

    if "stt" in requested_services:
        specs.append(
            SpeechServiceSpec(
                service="stt",
                name=args.stt_name,
                repository=args.stt_repository,
                revision=resolve_revision(api, args.stt_repository, args.stt_revision),
                image_url=args.stt_image_url,
                port=8000,
            )
        )
    if "tts" in requested_services:
        specs.append(
            SpeechServiceSpec(
                service="tts",
                name=args.tts_name,
                repository=args.tts_repository,
                revision=resolve_revision(api, args.tts_repository, args.tts_revision),
                image_url=args.tts_image_url,
                port=8091,
            )
        )
    return specs


def ensure_names_available(api: HfApi, namespace: str, specs: list[SpeechServiceSpec]) -> None:
    collisions: list[str] = []
    for spec in specs:
        try:
            api.get_inference_endpoint(spec.name, namespace=namespace)
        except Exception as exc:
            response = getattr(exc, "response", None)
            if getattr(response, "status_code", None) == 404:
                continue
            raise
        collisions.append(spec.name)

    if collisions:
        joined = ", ".join(collisions)
        raise ValueError(f"Inference Endpoint name already exists: {joined}")


def create_endpoint(api: HfApi, args: argparse.Namespace, spec: SpeechServiceSpec):
    return api.create_inference_endpoint(
        spec.name,
        namespace=args.namespace,
        repository=spec.repository,
        revision=spec.revision,
        framework=DEFAULT_FRAMEWORK,
        task="custom",
        accelerator="gpu",
        instance_size=args.instance_size,
        instance_type=args.instance_type,
        vendor=args.vendor,
        region=args.region,
        min_replica=args.min_replica,
        max_replica=args.max_replica,
        custom_image=build_custom_image(spec.image_url, spec.health_route, spec.port),
        env={"VLLM_ENABLE_CUDA_COMPATIBILITY": "1"},
        type=args.type,
        tags=["reachy-s2s", spec.service, "experiment"],
    )


def endpoint_summary(endpoint, service: str) -> dict[str, Any]:
    return {
        "service": service,
        "name": endpoint.name,
        "status": str(endpoint.status),
        "url": getattr(endpoint, "url", None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dedicated GPU STT and TTS endpoints for s2s experiments."
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--services", nargs="+", choices=("stt", "tts"), default=["stt", "tts"])
    parser.add_argument("--stt-name", default=DEFAULT_STT_NAME)
    parser.add_argument("--tts-name", default=DEFAULT_TTS_NAME)
    parser.add_argument("--stt-repository", default=DEFAULT_STT_REPOSITORY)
    parser.add_argument("--tts-repository", default=DEFAULT_TTS_REPOSITORY)
    parser.add_argument("--stt-revision")
    parser.add_argument("--tts-revision")
    parser.add_argument("--stt-image-url", default=DEFAULT_STT_IMAGE)
    parser.add_argument("--tts-image-url", default=DEFAULT_TTS_IMAGE)
    parser.add_argument("--vendor", default=DEFAULT_VENDOR)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--instance-size", default=DEFAULT_INSTANCE_SIZE)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--type", choices=("protected", "public", "private"), default=DEFAULT_ENDPOINT_TYPE)
    parser.add_argument("--min-replica", type=int, default=1)
    parser.add_argument("--max-replica", type=int, default=1)
    parser.add_argument("--wait", action="store_true", help="Wait for both endpoints to become ready")
    parser.add_argument("--wait-timeout", type=float, default=1800, help="Maximum provisioning wait in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Resolve revisions and print the deployment plan")
    args = parser.parse_args()

    if args.min_replica < 0:
        parser.error("--min-replica must be >= 0")
    if args.max_replica < 1:
        parser.error("--max-replica must be >= 1")
    if args.min_replica > args.max_replica:
        parser.error("--min-replica cannot exceed --max-replica")
    if args.wait_timeout <= 0:
        parser.error("--wait-timeout must be > 0")
    return args


def main() -> None:
    args = parse_args()
    api = HfApi()
    specs = build_specs(args, api)
    ensure_names_available(api, args.namespace, specs)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "namespace": args.namespace,
                    "vendor": args.vendor,
                    "region": args.region,
                    "instance": f"{args.instance_type}-{args.instance_size}",
                    "type": args.type,
                    "min_replica": args.min_replica,
                    "max_replica": args.max_replica,
                    "endpoints": [asdict(spec) for spec in specs],
                },
                indent=2,
            )
        )
        return

    endpoints = [(spec, create_endpoint(api, args, spec)) for spec in specs]
    if args.wait:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints) or 1) as executor:
            futures = [executor.submit(endpoint.wait, timeout=args.wait_timeout) for _, endpoint in endpoints]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        for _, endpoint in endpoints:
            endpoint.fetch()

    print(
        json.dumps(
            {"endpoints": [endpoint_summary(endpoint, spec.service) for spec, endpoint in endpoints]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

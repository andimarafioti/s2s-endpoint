#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any

from _endpoint_helpers import DEFAULT_FRAMEWORK, DEFAULT_REPOSITORY, build_custom_image
from huggingface_hub import HfApi

DEFAULT_NAMESPACE = "HuggingFaceM4"
DEFAULT_NAME = "reachy-s2s-pipeline-01"
DEFAULT_IMAGE = "ghcr.io/andimarafioti/s2s-pipeline:v0.1"
DEFAULT_VENDOR = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_INSTANCE_TYPE = "intel-spr"
DEFAULT_INSTANCE_SIZE = "x4"
DEFAULT_ENDPOINT_TYPE = "protected"
DEFAULT_PORT = 7860
DEFAULT_HEALTH_ROUTE = "/v1/usage"


def ensure_name_available(api: HfApi, namespace: str, name: str) -> None:
    try:
        api.get_inference_endpoint(name, namespace=namespace)
    except Exception as exc:
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 404:
            return
        raise
    raise ValueError(f"Inference Endpoint name already exists: {name}")


def resolve_secrets(environ: dict[str, str]) -> dict[str, str]:
    hf_token = environ.get("HF_TOKEN", "").strip()
    openai_key = environ.get("RESPONSES_API_API_KEY", "").strip() or environ.get("OPENAI_API_KEY", "").strip()
    missing = []
    if not hf_token:
        missing.append("HF_TOKEN")
    if not openai_key:
        missing.append("RESPONSES_API_API_KEY or OPENAI_API_KEY")
    if missing:
        raise ValueError(f"Missing required deployment secret(s): {', '.join(missing)}")
    return {
        "HF_TOKEN": hf_token,
        "RESPONSES_API_API_KEY": openai_key,
    }


def deployment_env(args: argparse.Namespace) -> dict[str, str]:
    return {
        "STT_BASE_URL": args.stt_base_url,
        "TTS_BASE_URL": args.tts_base_url,
        "MODEL_NAME": args.model_name,
        "NUM_PIPELINES": str(args.num_pipelines),
        "ENABLE_LIVE_TRANSCRIPTION": str(args.enable_live_transcription).lower(),
        "STREAM_BATCH_SENTENCES": str(args.stream_batch_sentences),
        "LOG_TRANSCRIPTS": "false",
    }


def summary(endpoint) -> dict[str, Any]:
    return {
        "name": endpoint.name,
        "status": str(endpoint.status),
        "url": getattr(endpoint, "url", None),
        "realtime_url": f"{endpoint.url}/v1/realtime" if getattr(endpoint, "url", None) else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CPU speech-to-speech pipeline endpoint backed by remote STT, LLM, and TTS services."
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--image-url", default=DEFAULT_IMAGE)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision")
    parser.add_argument("--vendor", default=DEFAULT_VENDOR)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--instance-size", default=DEFAULT_INSTANCE_SIZE)
    parser.add_argument("--type", choices=("protected", "public", "private"), default=DEFAULT_ENDPOINT_TYPE)
    parser.add_argument("--min-replica", type=int, default=1)
    parser.add_argument("--max-replica", type=int, default=1)
    parser.add_argument(
        "--stt-base-url",
        default="https://go3quisjv5ta7203.us-east-1.aws.endpoints.huggingface.cloud/v1",
    )
    parser.add_argument(
        "--tts-base-url",
        default="https://db6lx9j3kdymwu9w.us-east-1.aws.endpoints.huggingface.cloud/v1",
    )
    parser.add_argument("--model-name", default="gpt-5.6-terra")
    parser.add_argument("--num-pipelines", type=int, default=1)
    parser.add_argument("--stream-batch-sentences", type=int, default=3)
    parser.add_argument("--enable-live-transcription", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.min_replica < 0:
        parser.error("--min-replica must be >= 0")
    if args.max_replica < 1:
        parser.error("--max-replica must be >= 1")
    if args.min_replica > args.max_replica:
        parser.error("--min-replica cannot exceed --max-replica")
    if args.num_pipelines < 1:
        parser.error("--num-pipelines must be >= 1")
    if args.stream_batch_sentences < 1:
        parser.error("--stream-batch-sentences must be >= 1")
    if args.wait_timeout <= 0:
        parser.error("--wait-timeout must be > 0")
    return args


def main() -> None:
    args = parse_args()
    api = HfApi()
    ensure_name_available(api, args.namespace, args.name)
    endpoint_env = deployment_env(args)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "namespace": args.namespace,
                    "name": args.name,
                    "image": args.image_url,
                    "vendor": args.vendor,
                    "region": args.region,
                    "accelerator": "cpu",
                    "instance": f"{args.instance_type}-{args.instance_size}",
                    "type": args.type,
                    "min_replica": args.min_replica,
                    "max_replica": args.max_replica,
                    "env": endpoint_env,
                    "required_secret_names": ["HF_TOKEN", "RESPONSES_API_API_KEY"],
                },
                indent=2,
            )
        )
        return

    secrets = resolve_secrets(dict(os.environ))
    endpoint = api.create_inference_endpoint(
        args.name,
        namespace=args.namespace,
        repository=args.repository,
        revision=args.revision,
        framework=DEFAULT_FRAMEWORK,
        task="custom",
        accelerator="cpu",
        instance_size=args.instance_size,
        instance_type=args.instance_type,
        vendor=args.vendor,
        region=args.region,
        min_replica=args.min_replica,
        max_replica=args.max_replica,
        custom_image=build_custom_image(args.image_url, DEFAULT_HEALTH_ROUTE, DEFAULT_PORT),
        env=endpoint_env,
        secrets=secrets,
        type=args.type,
        tags=["reachy-s2s", "pipeline", "experiment"],
    )
    if args.wait:
        endpoint.wait(timeout=args.wait_timeout)
        endpoint.fetch()
    print(json.dumps({"endpoint": summary(endpoint)}, indent=2))


if __name__ == "__main__":
    main()

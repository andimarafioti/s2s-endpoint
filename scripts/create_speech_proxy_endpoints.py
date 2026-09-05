#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from _endpoint_helpers import DEFAULT_FRAMEWORK, DEFAULT_REPOSITORY, build_custom_image
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.speech_route_catalog import SpeechRouteCatalog  # noqa: E402

DEFAULT_NAMESPACE = "HuggingFaceM4"
DEFAULT_VENDOR = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_INSTANCE_TYPE = "intel-spr"
DEFAULT_INSTANCE_SIZE = "x1"
DEFAULT_ENDPOINT_TYPE = "protected"
DEFAULT_STT_PROXY_NAME = "reachy-s2s-stt-proxy"
DEFAULT_TTS_PROXY_NAME = "reachy-s2s-tts-proxy"
DEFAULT_LLM_PROXY_NAME = "reachy-s2s-llm-proxy"
DEFAULT_STT_BACKENDS = ["reachy-s2s-stt-01"]
DEFAULT_TTS_BACKENDS = ["reachy-s2s-tts-01"]
DEFAULT_LLM_BACKENDS = ["reachy-s2s-llm-01"]
DEFAULT_LLM_MODEL = "nvidia/Gemma-4-26B-A4B-NVFP4"
DEFAULT_PORT = 7860
DEFAULT_HEALTH_ROUTE = "/health"


@dataclass(frozen=True)
class SpeechBackendTarget:
    name: str
    url: str


@dataclass(frozen=True)
class SpeechProxySpec:
    service: str
    name: str
    backends: tuple[SpeechBackendTarget, ...]
    target_work: float
    latency_target: float


def resolve_backend_targets(
    api: HfApi,
    namespace: str,
    names: list[str],
    *,
    managed: bool = False,
) -> tuple[SpeechBackendTarget, ...]:
    if len(names) != len(set(names)):
        raise ValueError("Speech backend endpoint names must be unique")
    targets: list[SpeechBackendTarget] = []
    for name in names:
        endpoint = api.get_inference_endpoint(name, namespace=namespace)
        url = getattr(endpoint, "url", None)
        if not isinstance(url, str) or not url.strip():
            if not managed:
                raise ValueError(f"Speech backend endpoint does not have a URL: {name}")
            url = ""
        targets.append(SpeechBackendTarget(name=name, url=url.rstrip("/")))
    return tuple(targets)


def build_specs(args: argparse.Namespace, api: HfApi) -> list[SpeechProxySpec]:
    requested_services = set(args.services)
    if getattr(args, "catalog", None):
        service = args.services[0]
        return [
            SpeechProxySpec(
                service,
                getattr(args, f"{service}_proxy_name"),
                (),
                getattr(args, f"{service}_target_work"),
                getattr(args, f"{service}_latency_target"),
            )
        ]
    specs: list[SpeechProxySpec] = []
    if "stt" in requested_services:
        specs.append(
            SpeechProxySpec(
                service="stt",
                name=args.stt_proxy_name,
                backends=resolve_backend_targets(
                    api, args.namespace, args.stt_backends, managed=getattr(args, "autoscale", False)
                ),
                target_work=args.stt_target_work,
                latency_target=args.stt_latency_target,
            )
        )
    if "tts" in requested_services:
        specs.append(
            SpeechProxySpec(
                service="tts",
                name=args.tts_proxy_name,
                backends=resolve_backend_targets(
                    api, args.namespace, args.tts_backends, managed=getattr(args, "autoscale", False)
                ),
                target_work=args.tts_target_work,
                latency_target=args.tts_latency_target,
            )
        )
    if "llm" in requested_services:
        specs.append(
            SpeechProxySpec(
                service="llm",
                name=args.llm_proxy_name,
                backends=resolve_backend_targets(
                    api, args.namespace, args.llm_backends, managed=getattr(args, "autoscale", False)
                ),
                target_work=args.llm_target_work,
                latency_target=args.llm_latency_target,
            )
        )
    return specs


def ensure_names_available(api: HfApi, namespace: str, specs: list[SpeechProxySpec]) -> None:
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
        raise ValueError(f"Inference Endpoint name already exists: {', '.join(collisions)}")


def resolve_secrets(
    environ: dict[str, str], *, autoscale: bool = False, catalog: SpeechRouteCatalog | None = None
) -> dict[str, str]:
    if catalog:
        missing = [name for name in catalog.secret_names() if not environ.get(name, "").strip()]
        if missing:
            raise ValueError(f"Missing required route secrets: {', '.join(missing)}")
        return {name: environ[name].strip() for name in catalog.secret_names()}
    token = environ.get("HF_TOKEN", "").strip()
    if not token:
        raise ValueError("Missing required deployment secret: HF_TOKEN")
    secrets = {"SPEECH_BACKEND_API_KEY": token}
    if autoscale:
        control_token = environ.get("HF_CONTROL_TOKEN", "").strip()
        if not control_token:
            raise ValueError("autoscaling requires an explicit HF_CONTROL_TOKEN secret")
        secrets["HF_CONTROL_TOKEN"] = control_token
    return secrets


def deployment_env(args: argparse.Namespace, spec: SpeechProxySpec) -> dict[str, str]:
    backends = ",".join(f"{backend.name}={backend.url}" for backend in spec.backends)
    env = {
        "SPEECH_PROXY_SERVICE": spec.service,
        "SPEECH_BACKENDS": backends,
        "SPEECH_TARGET_WORK": str(spec.target_work),
        "SPEECH_LATENCY_TARGET": str(spec.latency_target),
        "SPEECH_LATENCY_WEIGHT": str(args.latency_weight),
        "SPEECH_MAX_ATTEMPTS": str(args.max_attempts),
        "SPEECH_MAX_CONNECTIONS": str(args.max_connections),
        "SPEECH_MAX_KEEPALIVE_CONNECTIONS": str(args.max_keepalive_connections),
        "SPEECH_HEALTH_INTERVAL_S": str(args.health_interval),
        "SPEECH_HEALTH_TIMEOUT_S": str(args.health_timeout),
        "SPEECH_REQUEST_TIMEOUT_S": str(args.request_timeout),
    }
    if getattr(args, "catalog", None):
        env.pop("SPEECH_BACKENDS")
        env["SPEECH_ROUTE_CATALOG"] = args.catalog.model_dump_json(exclude_unset=True)
        return env
    if spec.service == "stt":
        env["STT_AUDIO_EQUIVALENT_S"] = str(args.stt_audio_equivalent)
    elif spec.service == "tts":
        env["TTS_WARMUP_ENABLED"] = "true"
    else:
        env["LLM_WARMUP_ENABLED"] = "true"
        env["LLM_WARMUP_MODEL"] = args.llm_model
    if getattr(args, "autoscale", False):
        maximum = args.max_workers if args.max_workers is not None else len(spec.backends)
        if not 1 <= args.min_warm_workers <= maximum <= len(spec.backends):
            raise ValueError("autoscaling requires 1 <= min-warm-workers <= max-workers <= inventory size")
        env.update(
            SPEECH_AUTOSCALE_ENABLED="true",
            HF_ENDPOINT_NAMESPACE=args.namespace,
            SPEECH_WORKER_MIN_WARM=str(args.min_warm_workers),
            SPEECH_WORKER_MAX_WORKERS=str(maximum),
        )
    return env


def create_endpoint(
    api: HfApi,
    args: argparse.Namespace,
    spec: SpeechProxySpec,
    secrets: dict[str, str],
):
    return api.create_inference_endpoint(
        spec.name,
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
        min_replica=1,
        max_replica=1,
        custom_image=build_custom_image(args.image_url, DEFAULT_HEALTH_ROUTE, DEFAULT_PORT),
        env=deployment_env(args, spec),
        secrets=secrets,
        type=args.type,
        tags=["reachy-s2s", "speech-proxy", spec.service, "experiment"],
    )


def endpoint_summary(endpoint, service: str) -> dict[str, Any]:
    url = getattr(endpoint, "url", None)
    return {
        "service": service,
        "name": endpoint.name,
        "status": str(endpoint.status),
        "url": url,
        "openai_base_url": f"{url}/v1" if url else None,
    }


def _positive(parser: argparse.ArgumentParser, name: str, value: float) -> None:
    if value <= 0:
        parser.error(f"{name} must be > 0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stable CPU STT, TTS, and LLM proxies in front of dedicated inference workers."
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--services", nargs="+", choices=("stt", "tts", "llm"), default=["stt", "tts"])
    parser.add_argument("--route-catalog", type=Path, help="JSON model/provider catalog for one --services value")
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision")
    parser.add_argument("--stt-proxy-name", default=DEFAULT_STT_PROXY_NAME)
    parser.add_argument("--tts-proxy-name", default=DEFAULT_TTS_PROXY_NAME)
    parser.add_argument("--llm-proxy-name", default=DEFAULT_LLM_PROXY_NAME)
    parser.add_argument("--stt-backends", nargs="+", default=DEFAULT_STT_BACKENDS)
    parser.add_argument("--tts-backends", nargs="+", default=DEFAULT_TTS_BACKENDS)
    parser.add_argument("--llm-backends", nargs="+", default=DEFAULT_LLM_BACKENDS)
    parser.add_argument("--stt-target-work", type=float, default=96)
    parser.add_argument("--stt-latency-target", type=float, default=0.1)
    parser.add_argument("--stt-audio-equivalent", type=float, default=5)
    parser.add_argument("--tts-target-work", type=float, default=8)
    parser.add_argument("--tts-latency-target", type=float, default=0.5)
    parser.add_argument("--llm-target-work", type=float, default=64)
    parser.add_argument("--llm-latency-target", type=float, default=0.5)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--latency-weight", type=float, default=0.25)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-connections", type=int, default=1024)
    parser.add_argument("--max-keepalive-connections", type=int, default=256)
    parser.add_argument("--health-interval", type=float, default=10)
    parser.add_argument("--health-timeout", type=float, default=5)
    parser.add_argument("--request-timeout", type=float, default=120)
    parser.add_argument("--autoscale", action="store_true", help="Manage the supplied endpoint inventory")
    parser.add_argument("--min-warm-workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, help="Maximum active workers; defaults to supplied inventory size")
    parser.add_argument("--vendor", default=DEFAULT_VENDOR)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--instance-size", default=DEFAULT_INSTANCE_SIZE)
    parser.add_argument("--type", choices=("protected", "public", "private"), default=DEFAULT_ENDPOINT_TYPE)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.catalog = None
    if args.route_catalog:
        if len(args.services) != 1 or args.autoscale:
            parser.error("--route-catalog requires exactly one service and per-pool lifecycle settings")
        try:
            args.catalog = SpeechRouteCatalog.model_validate_json(args.route_catalog.read_text())
            args.catalog.validate_service(args.services[0])
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        if args.type == "public" and any(route.access_key_env is None for route in args.catalog.pools):
            parser.error("public catalog gateways require access_key_env on every route")

    for name in (
        "stt_target_work",
        "stt_latency_target",
        "stt_audio_equivalent",
        "tts_target_work",
        "tts_latency_target",
        "llm_target_work",
        "llm_latency_target",
        "health_interval",
        "health_timeout",
        "request_timeout",
        "wait_timeout",
    ):
        _positive(parser, f"--{name.replace('_', '-')}", getattr(args, name))
    if args.latency_weight < 0:
        parser.error("--latency-weight must be >= 0")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be >= 1")
    if args.max_connections < 1:
        parser.error("--max-connections must be >= 1")
    if not 0 <= args.max_keepalive_connections <= args.max_connections:
        parser.error("--max-keepalive-connections must be between 0 and --max-connections")
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
                    "image": args.image_url,
                    "vendor": args.vendor,
                    "region": args.region,
                    "accelerator": "cpu",
                    "instance": f"{args.instance_type}-{args.instance_size}",
                    "type": args.type,
                    "min_replica": 1,
                    "max_replica": 1,
                    "required_secret_names": args.catalog.secret_names()
                    if args.catalog
                    else ["SPEECH_BACKEND_API_KEY"] + (["HF_CONTROL_TOKEN"] if args.autoscale else []),
                    "endpoints": [{**asdict(spec), "env": deployment_env(args, spec)} for spec in specs],
                },
                indent=2,
            )
        )
        return

    secrets = resolve_secrets(dict(os.environ), autoscale=args.autoscale, catalog=args.catalog)
    endpoints = [(spec, create_endpoint(api, args, spec, secrets)) for spec in specs]
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

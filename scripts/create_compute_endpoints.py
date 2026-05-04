#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import sys

from huggingface_hub import HfApi

from _endpoint_helpers import (
    DEFAULT_FRAMEWORK,
    DEFAULT_HEALTH_ROUTE,
    DEFAULT_IMAGE_PORT,
    DEFAULT_REPOSITORY,
    build_custom_image,
    build_names,
    current_model_env,
    load_json_file,
    parse_key_value_pairs,
)

DEFAULT_COMPUTE_INDEX_WIDTH = 2


def resolve_names_to_create(
    *,
    api: HfApi,
    namespace: str | None,
    prefix: str | None,
    count: int | None,
    target_total: int | None,
    names: list[str],
) -> list[str]:
    if target_total is None:
        return build_names(prefix, count, names)

    if names or count is not None:
        raise ValueError("Use only one of --names, --prefix/--count, or --prefix/--target-total")
    if not prefix:
        raise ValueError("--target-total requires --prefix")
    if target_total < 1:
        raise ValueError("--target-total must be >= 1")

    existing_count = discover_existing_sequential_count(
        api=api,
        namespace=namespace,
        prefix=prefix,
        target_total=target_total,
    )
    if existing_count >= target_total:
        return []

    return build_indexed_names(
        prefix=prefix,
        first_index=existing_count + 1,
        end_index=target_total,
    )


def discover_existing_sequential_count(
    *,
    api: HfApi,
    namespace: str | None,
    prefix: str,
    target_total: int,
) -> int:
    width = max(DEFAULT_COMPUTE_INDEX_WIDTH, len(str(target_total)))
    for index in range(1, target_total + 1):
        name = f"{prefix}-{index:0{width}d}"
        try:
            api.get_inference_endpoint(name, namespace=namespace)
        except Exception as exc:
            if _is_not_found_error(exc):
                return index - 1
            raise
    return target_total


def build_indexed_names(*, prefix: str, first_index: int, end_index: int) -> list[str]:
    width = max(DEFAULT_COMPUTE_INDEX_WIDTH, len(str(end_index)))
    return [f"{prefix}-{index:0{width}d}" for index in range(first_index, end_index + 1)]


def _is_not_found_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 404


def load_template_env(
    *,
    api: HfApi,
    namespace: str | None,
    template_name: str | None,
) -> dict[str, str]:
    if not template_name:
        return {}
    endpoint = api.get_inference_endpoint(template_name, namespace=namespace)
    return current_model_env(endpoint.raw)


def build_endpoint_env(
    *,
    base_env: dict[str, str],
    session_shared_secret: str | None,
    pipeline_max_instances: int | None,
    pipeline_min_idle_instances: int | None,
    lb_callback_auth_token: str | None,
) -> dict[str, str]:
    endpoint_env = dict(base_env)

    if session_shared_secret is not None:
        session_shared_secret = session_shared_secret.strip()
        if not session_shared_secret:
            raise ValueError("--session-shared-secret must be a non-empty string")
        endpoint_env["SESSION_SHARED_SECRET"] = session_shared_secret
    elif not endpoint_env.get("SESSION_SHARED_SECRET", "").strip():
        raise ValueError("--session-shared-secret is required unless copied from --copy-env-from")

    if pipeline_max_instances is not None:
        endpoint_env["PIPELINE_MAX_INSTANCES"] = str(pipeline_max_instances)
    elif "PIPELINE_MAX_INSTANCES" not in endpoint_env:
        endpoint_env["PIPELINE_MAX_INSTANCES"] = "1"

    if pipeline_min_idle_instances is not None:
        endpoint_env["PIPELINE_MIN_IDLE_INSTANCES"] = str(pipeline_min_idle_instances)
    elif "PIPELINE_MIN_IDLE_INSTANCES" not in endpoint_env:
        endpoint_env["PIPELINE_MIN_IDLE_INSTANCES"] = "1"

    if lb_callback_auth_token:
        endpoint_env["LB_CALLBACK_AUTH_TOKEN"] = lb_callback_auth_token

    return endpoint_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create GPU compute endpoints for the s2s-endpoint app."
    )
    parser.add_argument("--namespace", help="Endpoint namespace / org")
    parser.add_argument("--prefix", help="Endpoint name prefix, used with --count or --target-total")
    parser.add_argument("--count", type=int, help="Number of endpoints to create, used with --prefix")
    parser.add_argument("--target-total", type=int, help="Create only the missing tail needed for this total sequential pool size")
    parser.add_argument("--names", nargs="*", default=[], help="Explicit endpoint names")
    parser.add_argument("--vendor", required=True, help="Cloud vendor, for example aws")
    parser.add_argument("--region", required=True, help="Cloud region")
    parser.add_argument("--instance-size", required=True, help="Instance size, for example x1")
    parser.add_argument("--instance-type", required=True, help="GPU instance type, for example nvidia-a10g")
    parser.add_argument("--image-url", required=True, help="Custom compute image URL built from Dockerfile.compute")
    parser.add_argument("--image-health-route", default=DEFAULT_HEALTH_ROUTE, help="Health route exposed by the compute image")
    parser.add_argument("--image-port", type=int, default=DEFAULT_IMAGE_PORT, help="Container port exposed by the compute image")
    parser.add_argument("--copy-env-from", help="Copy readable env vars from this existing compute endpoint before applying --env overrides")
    parser.add_argument("--session-shared-secret", help="Shared secret used to validate LB-issued session tokens")
    parser.add_argument("--lb-callback-auth-token", help="Optional bearer token used by compute endpoints when notifying the LB")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help=argparse.SUPPRESS)
    parser.add_argument("--account-id", help="Optional account id")
    parser.add_argument("--revision", help="Optional repo revision")
    parser.add_argument("--type", default="public", help="Endpoint type; direct client connections usually require public compute endpoints")
    parser.add_argument("--min-replica", type=int, default=0, help="Initial min replica count")
    parser.add_argument("--max-replica", type=int, default=1, help="Initial max replica count")
    parser.add_argument("--scale-to-zero-timeout", type=int, help="Optional scale-to-zero timeout")
    parser.add_argument("--pipeline-max-instances", type=int, help="Local pipeline slots per compute endpoint")
    parser.add_argument("--pipeline-min-idle-instances", type=int, help="Warm local pipeline slots per compute endpoint")
    parser.add_argument("--env-file", help="JSON file with extra env vars")
    parser.add_argument("--secret-file", help="JSON file with secrets")
    parser.add_argument("--env", action="append", default=[], help="Extra env var in KEY=VALUE form")
    parser.add_argument("--secret", action="append", default=[], help="Extra secret in KEY=VALUE form")
    parser.add_argument("--wait", action="store_true", help="Wait for each endpoint to finish provisioning")
    args = parser.parse_args()

    api = HfApi()
    names = resolve_names_to_create(
        api=api,
        namespace=args.namespace,
        prefix=args.prefix,
        count=args.count,
        target_total=args.target_total,
        names=args.names,
    )
    if not names:
        print(json.dumps({"endpoints": []}, indent=2))
        return

    env = load_template_env(
        api=api,
        namespace=args.namespace,
        template_name=args.copy_env_from,
    )
    env.update(load_json_file(args.env_file) or {})
    secrets = load_json_file(args.secret_file) or {}
    env.update(parse_key_value_pairs(args.env))
    secrets.update(parse_key_value_pairs(args.secret))

    endpoint_env = build_endpoint_env(
        base_env=env,
        session_shared_secret=args.session_shared_secret,
        pipeline_max_instances=args.pipeline_max_instances,
        pipeline_min_idle_instances=args.pipeline_min_idle_instances,
        lb_callback_auth_token=args.lb_callback_auth_token,
    )

    llm_backend = str(endpoint_env.get("LLM", "open_api")).strip() or "open_api"
    if llm_backend == "open_api" and not (
        secrets.get("OPEN_API_API_KEY")
        or secrets.get("HF_TOKEN")
        or endpoint_env.get("OPEN_API_API_KEY")
        or endpoint_env.get("HF_TOKEN")
    ):
        print(
            "warning: compute endpoints default to LLM=open_api, but neither OPEN_API_API_KEY nor HF_TOKEN was provided. "
            "The container will start, but runtime requests will fail when the speech-to-speech pipeline calls the OpenAI-compatible API.",
            file=sys.stderr,
        )

    custom_image = build_custom_image(args.image_url, args.image_health_route, args.image_port)

    endpoints = []

    for name in names:
        endpoint = api.create_inference_endpoint(
            name,
            namespace=args.namespace,
            repository=args.repository,
            framework=DEFAULT_FRAMEWORK,
            task="custom",
            accelerator="gpu",
            instance_size=args.instance_size,
            instance_type=args.instance_type,
            vendor=args.vendor,
            region=args.region,
            account_id=args.account_id,
            min_replica=args.min_replica,
            max_replica=args.max_replica,
            scale_to_zero_timeout=args.scale_to_zero_timeout,
            revision=args.revision,
            custom_image=custom_image,
            env=dict(endpoint_env) or None,
            secrets=secrets or None,
            type=args.type,
        )
        endpoints.append(endpoint)

    if args.wait:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints) or 1) as executor:
            futures = [executor.submit(_wait_and_fetch_endpoint, endpoint) for endpoint in endpoints]
            for future in concurrent.futures.as_completed(futures):
                future.result()

    print(
        json.dumps(
            {
                "endpoints": [
                    {
                        "name": endpoint.name,
                        "status": str(endpoint.status),
                        "url": getattr(endpoint, "url", None),
                    }
                    for endpoint in endpoints
                ]
            },
            indent=2,
        )
    )


def _wait_and_fetch_endpoint(endpoint) -> None:
    endpoint.wait()
    endpoint.fetch()


if __name__ == "__main__":
    main()

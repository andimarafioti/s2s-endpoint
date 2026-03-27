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
    load_json_file,
    parse_key_value_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create GPU compute endpoints for the s2s-endpoint app."
    )
    parser.add_argument("--namespace", help="Endpoint namespace / org")
    parser.add_argument("--prefix", help="Endpoint name prefix, used with --count")
    parser.add_argument("--count", type=int, help="Number of endpoints to create, used with --prefix")
    parser.add_argument("--names", nargs="*", default=[], help="Explicit endpoint names")
    parser.add_argument("--vendor", required=True, help="Cloud vendor, for example aws")
    parser.add_argument("--region", required=True, help="Cloud region")
    parser.add_argument("--instance-size", required=True, help="Instance size, for example x1")
    parser.add_argument("--instance-type", required=True, help="GPU instance type, for example nvidia-a10g")
    parser.add_argument("--image-url", required=True, help="Custom compute image URL built from Dockerfile.compute")
    parser.add_argument("--image-health-route", default=DEFAULT_HEALTH_ROUTE, help="Health route exposed by the compute image")
    parser.add_argument("--image-port", type=int, default=DEFAULT_IMAGE_PORT, help="Container port exposed by the compute image")
    parser.add_argument("--session-shared-secret", required=True, help="Shared secret used to validate LB-issued session tokens")
    parser.add_argument("--lb-callback-auth-token", help="Optional bearer token used by compute endpoints when notifying the LB")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help=argparse.SUPPRESS)
    parser.add_argument("--account-id", help="Optional account id")
    parser.add_argument("--revision", help="Optional repo revision")
    parser.add_argument("--type", default="public", help="Endpoint type; direct client connections usually require public compute endpoints")
    parser.add_argument("--min-replica", type=int, default=0, help="Initial min replica count")
    parser.add_argument("--max-replica", type=int, default=1, help="Initial max replica count")
    parser.add_argument("--scale-to-zero-timeout", type=int, help="Optional scale-to-zero timeout")
    parser.add_argument("--pipeline-max-instances", type=int, default=1, help="Local pipeline slots per compute endpoint")
    parser.add_argument("--pipeline-min-idle-instances", type=int, default=1, help="Warm local pipeline slots per compute endpoint")
    parser.add_argument("--s2s-server-mode", choices=["websocket", "realtime"], default="websocket", help="speech-to-speech server mode to run inside each compute endpoint")
    parser.add_argument("--env-file", help="JSON file with extra env vars")
    parser.add_argument("--secret-file", help="JSON file with secrets")
    parser.add_argument("--env", action="append", default=[], help="Extra env var in KEY=VALUE form")
    parser.add_argument("--secret", action="append", default=[], help="Extra secret in KEY=VALUE form")
    parser.add_argument("--wait", action="store_true", help="Wait for each endpoint to finish provisioning")
    args = parser.parse_args()
    args.session_shared_secret = args.session_shared_secret.strip()
    if not args.session_shared_secret:
        raise ValueError("--session-shared-secret must be a non-empty string")

    names = build_names(args.prefix, args.count, args.names)

    env = load_json_file(args.env_file) or {}
    secrets = load_json_file(args.secret_file) or {}
    env.update(parse_key_value_pairs(args.env))
    secrets.update(parse_key_value_pairs(args.secret))

    llm_backend = str(env.get("LLM", "open_api")).strip() or "open_api"
    if llm_backend == "open_api" and not (
        secrets.get("OPEN_API_API_KEY") or secrets.get("HF_TOKEN") or env.get("OPEN_API_API_KEY") or env.get("HF_TOKEN")
    ):
        print(
            "warning: compute endpoints default to LLM=open_api, but neither OPEN_API_API_KEY nor HF_TOKEN was provided. "
            "The container will start, but runtime requests will fail when the speech-to-speech pipeline calls the OpenAI-compatible API.",
            file=sys.stderr,
        )

    custom_image = build_custom_image(args.image_url, args.image_health_route, args.image_port)

    api = HfApi()
    endpoints = []

    for name in names:
        endpoint_env = dict(env)
        endpoint_env.update(
            {
                "SESSION_SHARED_SECRET": args.session_shared_secret,
                "PIPELINE_MAX_INSTANCES": str(args.pipeline_max_instances),
                "PIPELINE_MIN_IDLE_INSTANCES": str(args.pipeline_min_idle_instances),
                "S2S_SERVER_MODE": args.s2s_server_mode,
            }
        )
        if args.lb_callback_auth_token:
            endpoint_env["LB_CALLBACK_AUTH_TOKEN"] = args.lb_callback_auth_token

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
            env=endpoint_env or None,
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

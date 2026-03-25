#!/usr/bin/env python3
import argparse
import json

from huggingface_hub import HfApi

from _endpoint_helpers import (
    DEFAULT_FRAMEWORK,
    DEFAULT_HEALTH_ROUTE,
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
    parser.add_argument("--image-url", required=True, help="Custom compute image URL built from Dockerfile")
    parser.add_argument("--image-health-route", default=DEFAULT_HEALTH_ROUTE, help="Health route exposed by the compute image")
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
    parser.add_argument("--env-file", help="JSON file with extra env vars")
    parser.add_argument("--secret-file", help="JSON file with secrets")
    parser.add_argument("--env", action="append", default=[], help="Extra env var in KEY=VALUE form")
    parser.add_argument("--secret", action="append", default=[], help="Extra secret in KEY=VALUE form")
    parser.add_argument("--wait", action="store_true", help="Wait for each endpoint to finish provisioning")
    args = parser.parse_args()

    names = build_names(args.prefix, args.count, args.names)

    env = load_json_file(args.env_file) or {}
    secrets = load_json_file(args.secret_file) or {}
    env.update(parse_key_value_pairs(args.env))
    secrets.update(parse_key_value_pairs(args.secret))

    custom_image = build_custom_image(args.image_url, args.image_health_route)

    api = HfApi()
    created = []

    for name in names:
        endpoint_env = dict(env)
        endpoint_env.update(
            {
                "SESSION_SHARED_SECRET": args.session_shared_secret,
                "PIPELINE_MAX_INSTANCES": str(args.pipeline_max_instances),
                "PIPELINE_MIN_IDLE_INSTANCES": str(args.pipeline_min_idle_instances),
            }
        )
        if args.lb_callback_auth_token:
            endpoint_env["LB_CALLBACK_AUTH_TOKEN"] = args.lb_callback_auth_token

        endpoint = api.create_inference_endpoint(
            name,
            namespace=args.namespace,
            repository=args.repository,
            framework=DEFAULT_FRAMEWORK,
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
        if args.wait:
            endpoint.wait()
            endpoint.fetch()
        created.append(
            {
                "name": name,
                "status": str(endpoint.status),
                "url": getattr(endpoint, "url", None),
            }
        )

    print(json.dumps({"endpoints": created}, indent=2))


if __name__ == "__main__":
    main()

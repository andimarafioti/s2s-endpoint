#!/usr/bin/env python3
import argparse
import json

from huggingface_hub import HfApi

from _endpoint_helpers import (
    DEFAULT_ENDPOINT_TYPE,
    DEFAULT_FRAMEWORK,
    DEFAULT_HEALTH_ROUTE,
    DEFAULT_REPOSITORY,
    build_custom_image,
    load_json_file,
    parse_key_value_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the CPU load-balancer endpoint for the s2s-endpoint app."
    )
    parser.add_argument("--name", required=True, help="Load-balancer endpoint name")
    parser.add_argument("--namespace", help="Endpoint namespace / org")
    parser.add_argument("--vendor", required=True, help="Cloud vendor, for example aws")
    parser.add_argument("--region", required=True, help="Cloud region")
    parser.add_argument("--instance-size", required=True, help="CPU instance size, for example x2")
    parser.add_argument("--instance-type", required=True, help="CPU instance type, for example intel-icl")
    parser.add_argument("--image-url", required=True, help="Custom load-balancer image URL built from Dockerfile.load_balancer")
    parser.add_argument("--image-health-route", default=DEFAULT_HEALTH_ROUTE, help="Health route exposed by the load-balancer image")
    parser.add_argument("--compute-endpoint-names", required=True, help="Comma-separated compute endpoint names")
    parser.add_argument("--compute-endpoint-slots", type=int, default=1, help="Concurrent sessions each compute endpoint can handle")
    parser.add_argument("--compute-endpoint-min-warm", type=int, default=1, help="Warm compute endpoints to keep ready")
    parser.add_argument("--compute-endpoint-wake-threshold-slots", type=int, default=1, help="Wake another compute endpoint when total free slots drop to this value")
    parser.add_argument("--compute-endpoint-idle-park-timeout-s", type=float, default=300, help="Idle timeout before parking a compute endpoint")
    parser.add_argument("--compute-endpoint-reconcile-interval-s", type=float, default=10, help="Refresh interval for compute endpoint state")
    parser.add_argument("--compute-endpoint-wait-timeout-s", type=int, default=900, help="Timeout while waiting for resumed compute endpoints")
    parser.add_argument("--compute-endpoint-park-strategy", choices=["pause", "scale_to_zero"], default="pause")
    parser.add_argument("--hf-endpoint-namespace", help="Namespace that owns the compute endpoints; defaults to --namespace")
    parser.add_argument("--downstream-endpoint-token", help="Optional bearer token used by the LB when connecting to protected compute endpoints")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help=argparse.SUPPRESS)
    parser.add_argument("--account-id", help="Optional account id")
    parser.add_argument("--revision", help="Optional repo revision")
    parser.add_argument("--type", default=DEFAULT_ENDPOINT_TYPE, help="Endpoint type")
    parser.add_argument("--min-replica", type=int, default=1, help="Initial min replica count")
    parser.add_argument("--max-replica", type=int, default=1, help="Initial max replica count")
    parser.add_argument("--env-file", help="JSON file with extra env vars")
    parser.add_argument("--secret-file", help="JSON file with secrets")
    parser.add_argument("--env", action="append", default=[], help="Extra env var in KEY=VALUE form")
    parser.add_argument("--secret", action="append", default=[], help="Extra secret in KEY=VALUE form")
    parser.add_argument("--wait", action="store_true", help="Wait for endpoint to finish provisioning")
    args = parser.parse_args()

    env = load_json_file(args.env_file) or {}
    secrets = load_json_file(args.secret_file) or {}
    env.update(parse_key_value_pairs(args.env))
    secrets.update(parse_key_value_pairs(args.secret))

    compute_namespace = args.hf_endpoint_namespace or args.namespace or ""
    env.update(
        {
            "HF_ENDPOINT_NAMESPACE": compute_namespace,
            "COMPUTE_ENDPOINT_NAMES": args.compute_endpoint_names,
            "COMPUTE_ENDPOINT_SLOTS": str(args.compute_endpoint_slots),
            "COMPUTE_ENDPOINT_MIN_WARM": str(args.compute_endpoint_min_warm),
            "COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS": str(args.compute_endpoint_wake_threshold_slots),
            "COMPUTE_ENDPOINT_IDLE_PARK_TIMEOUT_S": str(args.compute_endpoint_idle_park_timeout_s),
            "COMPUTE_ENDPOINT_RECONCILE_INTERVAL_S": str(args.compute_endpoint_reconcile_interval_s),
            "COMPUTE_ENDPOINT_WAIT_TIMEOUT_S": str(args.compute_endpoint_wait_timeout_s),
            "COMPUTE_ENDPOINT_PARK_STRATEGY": args.compute_endpoint_park_strategy,
        }
    )
    if args.downstream_endpoint_token:
        env["DOWNSTREAM_ENDPOINT_TOKEN"] = args.downstream_endpoint_token

    custom_image = build_custom_image(args.image_url, args.image_health_route)

    api = HfApi()
    endpoint = api.create_inference_endpoint(
        args.name,
        namespace=args.namespace,
        repository=args.repository,
        framework=DEFAULT_FRAMEWORK,
        accelerator="cpu",
        instance_size=args.instance_size,
        instance_type=args.instance_type,
        vendor=args.vendor,
        region=args.region,
        account_id=args.account_id,
        min_replica=args.min_replica,
        max_replica=args.max_replica,
        revision=args.revision,
        custom_image=custom_image,
        env=env or None,
        secrets=secrets or None,
        type=args.type,
    )
    if args.wait:
        endpoint.wait()
        endpoint.fetch()

    print(
        json.dumps(
            {
                "name": args.name,
                "status": str(endpoint.status),
                "url": getattr(endpoint, "url", None),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

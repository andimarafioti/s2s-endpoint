#!/usr/bin/env python3
import argparse
import json

from huggingface_hub import HfApi

from _endpoint_helpers import build_names, current_custom_image, current_model_env


DEFAULT_LOAD_BALANCER_NAME = "reachy-s2s-lb"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update compute and/or load-balancer endpoint images for the s2s-endpoint app."
    )
    parser.add_argument("--namespace", help="Endpoint namespace / org")
    parser.add_argument("--compute", help="New compute image URL")
    parser.add_argument("--load_balancer", help="New load-balancer image URL")
    parser.add_argument(
        "--load-balancer-name",
        default=DEFAULT_LOAD_BALANCER_NAME,
        help=f"Load-balancer endpoint name (default: {DEFAULT_LOAD_BALANCER_NAME})",
    )
    parser.add_argument("--compute-prefix", help="Compute endpoint name prefix, used with --compute-count")
    parser.add_argument("--compute-count", type=int, help="Number of compute endpoints, used with --compute-prefix")
    parser.add_argument("--compute-names", nargs="*", default=[], help="Explicit compute endpoint names")
    parser.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=True,
        help="Wait for each endpoint update to finish before moving to the next one (default).",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit updates without waiting for each endpoint to return to running.",
    )
    parser.add_argument("--wait-timeout-s", type=int, default=1800, help="Timeout when waiting for an endpoint update")
    parser.add_argument("--wait-refresh-every-s", type=int, default=5, help="Polling interval while waiting")
    parser.add_argument("--dry-run", action="store_true", help="Print planned updates without applying them")
    args = parser.parse_args()

    if not args.compute and not args.load_balancer:
        raise ValueError("Provide at least one of --compute or --load_balancer")

    api = HfApi()
    results: dict[str, object] = {}

    if args.compute:
        compute_names = resolve_compute_names(
            api=api,
            namespace=args.namespace,
            load_balancer_name=args.load_balancer_name,
            explicit_names=args.compute_names,
            prefix=args.compute_prefix,
            count=args.compute_count,
        )
        results["compute"] = update_many(
            api=api,
            namespace=args.namespace,
            names=compute_names,
            image_url=args.compute,
            wait=args.wait,
            wait_timeout_s=args.wait_timeout_s,
            wait_refresh_every_s=args.wait_refresh_every_s,
            dry_run=args.dry_run,
        )

    if args.load_balancer:
        results["load_balancer"] = update_one(
            api=api,
            namespace=args.namespace,
            name=args.load_balancer_name,
            image_url=args.load_balancer,
            wait=args.wait,
            wait_timeout_s=args.wait_timeout_s,
            wait_refresh_every_s=args.wait_refresh_every_s,
            dry_run=args.dry_run,
        )

    print(json.dumps(results, indent=2, sort_keys=True))


def resolve_compute_names(
    *,
    api: HfApi,
    namespace: str | None,
    load_balancer_name: str,
    explicit_names: list[str],
    prefix: str | None,
    count: int | None,
) -> list[str]:
    if explicit_names or prefix or count is not None:
        return build_names(prefix, count, explicit_names)

    endpoint = api.get_inference_endpoint(load_balancer_name, namespace=namespace)
    env = current_model_env(endpoint.raw)
    names = [name.strip() for name in env.get("COMPUTE_ENDPOINT_NAMES", "").split(",") if name.strip()]
    if not names:
        raise ValueError(
            "Could not infer compute endpoint names from the load balancer. "
            "Pass --compute-names or --compute-prefix/--compute-count."
        )
    return names


def update_many(
    *,
    api: HfApi,
    namespace: str | None,
    names: list[str],
    image_url: str,
    wait: bool,
    wait_timeout_s: int,
    wait_refresh_every_s: int,
    dry_run: bool,
) -> list[dict[str, object]]:
    return [
        update_one(
            api=api,
            namespace=namespace,
            name=name,
            image_url=image_url,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            dry_run=dry_run,
        )
        for name in names
    ]


def update_one(
    *,
    api: HfApi,
    namespace: str | None,
    name: str,
    image_url: str,
    wait: bool,
    wait_timeout_s: int,
    wait_refresh_every_s: int,
    dry_run: bool,
) -> dict[str, object]:
    endpoint = api.get_inference_endpoint(name, namespace=namespace)
    current_image = current_custom_image(endpoint.raw)
    updated_image = dict(current_image)
    updated_image["url"] = image_url

    result = {
        "name": name,
        "status_before": str(endpoint.status),
        "url": getattr(endpoint, "url", None),
        "image_before": current_image["url"],
        "image_after": image_url,
        "health_route": current_image["health_route"],
        "port": current_image["port"],
        "skipped": False,
    }

    if current_image["url"] == image_url:
        result["skipped"] = True
        result["status_after"] = str(endpoint.status)
        return result

    if dry_run:
        result["status_after"] = "dry_run"
        return result

    endpoint = api.update_inference_endpoint(
        name,
        namespace=namespace,
        custom_image=updated_image,
    )
    if wait:
        endpoint.wait(timeout=wait_timeout_s, refresh_every=wait_refresh_every_s)
        endpoint.fetch()
    result["status_after"] = str(endpoint.status)
    return result


if __name__ == "__main__":
    main()

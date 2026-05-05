#!/usr/bin/env python3
import argparse
import json

from huggingface_hub import HfApi

from _endpoint_helpers import build_names, current_model_env, load_json_file, merge_env_updates, parse_key_value_pairs


def add_compute_pool_updates(
    env_updates: dict[str, str],
    *,
    prefix: str | None,
    count: int | None,
    min_warm: int | None,
    wake_threshold_slots: int | None,
) -> None:
    if prefix or count is not None:
        names = build_names(prefix, count, [])
        env_updates["COMPUTE_ENDPOINT_NAMES"] = ",".join(names)
    if min_warm is not None:
        if min_warm < 0:
            raise ValueError("--compute-endpoint-min-warm must be >= 0")
        env_updates["COMPUTE_ENDPOINT_MIN_WARM"] = str(min_warm)
    if wake_threshold_slots is not None:
        if wake_threshold_slots < 0:
            raise ValueError("--compute-endpoint-wake-threshold-slots must be >= 0")
        env_updates["COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS"] = str(wake_threshold_slots)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update env vars on the load-balancer endpoint for the s2s-endpoint app."
    )
    parser.add_argument("--name", required=True, help="Load-balancer endpoint name")
    parser.add_argument("--namespace", help="Endpoint namespace / org")
    parser.add_argument("--env-file", help="JSON file with env vars to set or overwrite")
    parser.add_argument("--env", action="append", default=[], help="Env var to set in KEY=VALUE form")
    parser.add_argument("--unset-env", action="append", default=[], help="Env var key to remove")
    parser.add_argument("--compute-endpoint-prefix", help="Compute endpoint name prefix, used with --compute-endpoint-count")
    parser.add_argument("--compute-endpoint-count", type=int, help="Number of compute endpoints, used with --compute-endpoint-prefix")
    parser.add_argument("--compute-endpoint-min-warm", type=int, help="Set COMPUTE_ENDPOINT_MIN_WARM")
    parser.add_argument("--compute-endpoint-wake-threshold-slots", type=int, help="Set COMPUTE_ENDPOINT_WAKE_THRESHOLD_SLOTS")
    parser.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=True,
        help="Wait for the endpoint update to finish before exiting (default).",
    )
    parser.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help="Submit the update without waiting for the endpoint to return to running.",
    )
    parser.add_argument("--wait-timeout-s", type=int, default=1800, help="Timeout when waiting for the endpoint update")
    parser.add_argument("--wait-refresh-every-s", type=int, default=5, help="Polling interval while waiting")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned env changes without applying them")
    args = parser.parse_args()

    env_updates = load_json_file(args.env_file) or {}
    env_updates.update(parse_key_value_pairs(args.env))
    add_compute_pool_updates(
        env_updates,
        prefix=args.compute_endpoint_prefix,
        count=args.compute_endpoint_count,
        min_warm=args.compute_endpoint_min_warm,
        wake_threshold_slots=args.compute_endpoint_wake_threshold_slots,
    )
    unset_env = [key.strip() for key in args.unset_env if key.strip()]

    if not env_updates and not unset_env:
        raise ValueError("Provide at least one --env/--env-file entry or one --unset-env key")

    api = HfApi()
    endpoint = api.get_inference_endpoint(args.name, namespace=args.namespace)
    current_env = current_model_env(endpoint.raw)
    updated_env = merge_env_updates(current_env, env_updates, unset_env)

    changed = {
        key: updated_env[key]
        for key in sorted(updated_env)
        if current_env.get(key) != updated_env[key]
    }
    removed = sorted(key for key in current_env if key not in updated_env)

    result = {
        "name": args.name,
        "status_before": str(endpoint.status),
        "url": getattr(endpoint, "url", None),
        "changed": changed,
        "removed": removed,
        "skipped": False,
    }

    if current_env == updated_env:
        result["skipped"] = True
        result["status_after"] = str(endpoint.status)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if not args.dry_run:
        endpoint = api.update_inference_endpoint(
            args.name,
            namespace=args.namespace,
            env=updated_env,
        )
        if args.wait:
            endpoint.wait(timeout=args.wait_timeout_s, refresh_every=args.wait_refresh_every_s)
            endpoint.fetch()
        result["status_after"] = str(endpoint.status)
    else:
        result["status_after"] = "dry_run"

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

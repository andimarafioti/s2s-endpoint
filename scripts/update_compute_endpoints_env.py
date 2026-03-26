#!/usr/bin/env python3
import argparse
import json

from huggingface_hub import HfApi

from _endpoint_helpers import (
    build_names,
    current_model_env,
    load_json_file,
    merge_env_updates,
    parse_key_value_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update env vars on existing compute endpoints for the s2s-endpoint app."
    )
    parser.add_argument("--namespace", help="Endpoint namespace / org")
    parser.add_argument("--prefix", help="Endpoint name prefix, used with --count")
    parser.add_argument("--count", type=int, help="Number of endpoints to update, used with --prefix")
    parser.add_argument("--names", nargs="*", default=[], help="Explicit endpoint names")
    parser.add_argument("--env-file", help="JSON file with env vars to set or overwrite")
    parser.add_argument("--env", action="append", default=[], help="Env var to set in KEY=VALUE form")
    parser.add_argument("--unset-env", action="append", default=[], help="Env var key to remove")
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
    parser.add_argument("--dry-run", action="store_true", help="Print the planned env changes without applying them")
    args = parser.parse_args()

    names = build_names(args.prefix, args.count, args.names)
    env_updates = load_json_file(args.env_file) or {}
    env_updates.update(parse_key_value_pairs(args.env))
    unset_env = [key.strip() for key in args.unset_env if key.strip()]

    if not env_updates and not unset_env:
        raise ValueError("Provide at least one --env/--env-file entry or one --unset-env key")

    api = HfApi()
    results = []

    for name in names:
        endpoint = api.get_inference_endpoint(name, namespace=args.namespace)
        current_env = current_model_env(endpoint.raw)
        updated_env = merge_env_updates(current_env, env_updates, unset_env)

        changed = {
            key: updated_env[key]
            for key in sorted(updated_env)
            if current_env.get(key) != updated_env[key]
        }
        removed = sorted(key for key in current_env if key not in updated_env)

        result = {
            "name": name,
            "status_before": str(endpoint.status),
            "url": getattr(endpoint, "url", None),
            "changed": changed,
            "removed": removed,
            "skipped": False,
        }

        if current_env == updated_env:
            result["skipped"] = True
            result["status_after"] = str(endpoint.status)
            results.append(result)
            continue

        if not args.dry_run:
            endpoint = api.update_inference_endpoint(
                name,
                namespace=args.namespace,
                env=updated_env,
            )
            if args.wait:
                endpoint.wait(timeout=args.wait_timeout_s, refresh_every=args.wait_refresh_every_s)
                endpoint.fetch()
            result["status_after"] = str(endpoint.status)
        else:
            result["status_after"] = "dry_run"

        results.append(result)

    print(json.dumps({"endpoints": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

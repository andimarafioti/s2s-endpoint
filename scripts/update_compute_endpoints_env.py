#!/usr/bin/env python3
import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import sys
import time

from huggingface_hub import HfApi
from huggingface_hub.errors import InferenceEndpointError, InferenceEndpointTimeoutError

from _endpoint_helpers import (
    build_names,
    current_model_env,
    load_json_file,
    merge_env_updates,
    parse_key_value_pairs,
)

FAILED_UPDATE_STATUSES = {"failed", "updateFailed"}
PARKED_STATUSES = {"paused", "scaledToZero"}


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def expected_target_status(status_before: str) -> str:
    return "parked" if status_before in PARKED_STATUSES else "running"


def wait_for_endpoint_update(
    endpoint,
    *,
    target_status: str,
    timeout: int,
    refresh_every: int,
):
    if target_status == "running":
        endpoint.wait(timeout=timeout, refresh_every=refresh_every)
        endpoint.fetch()
        return endpoint

    start = time.time()
    while True:
        current_status = str(endpoint.status)
        if current_status in FAILED_UPDATE_STATUSES:
            raise InferenceEndpointError(
                f"Inference Endpoint {endpoint.name} failed to update. Please check the logs for more information."
            )
        if target_status == "parked" and current_status in PARKED_STATUSES:
            endpoint.fetch()
            return endpoint
        if current_status == target_status:
            endpoint.fetch()
            return endpoint
        if timeout is not None and time.time() - start > timeout:
            raise InferenceEndpointTimeoutError(
                f"Timeout while waiting for Inference Endpoint {endpoint.name} to return to {target_status}."
            )
        time.sleep(refresh_every)
        endpoint.fetch()


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
    parser.add_argument("--secret-file", help="JSON file with secrets to set or overwrite")
    parser.add_argument("--secret", action="append", default=[], help="Secret to set in KEY=VALUE form")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help="Number of compute endpoint env updates to run in parallel. Default: all selected endpoints. Use 1 for sequential updates.",
    )
    parser.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=True,
        help="Wait for each endpoint update to finish before returning (default).",
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
    secret_updates = load_json_file(args.secret_file) or {}
    secret_updates.update(parse_key_value_pairs(args.secret))

    if not env_updates and not unset_env and not secret_updates:
        raise ValueError("Provide at least one --env/--env-file entry, one --unset-env key, or one --secret/--secret-file entry")

    api = HfApi()
    results = update_many(
        api=api,
        namespace=args.namespace,
        names=names,
        env_updates=env_updates,
        unset_env=unset_env,
        secret_updates=secret_updates,
        wait=args.wait,
        wait_timeout_s=args.wait_timeout_s,
        wait_refresh_every_s=args.wait_refresh_every_s,
        dry_run=args.dry_run,
        parallelism=args.parallelism,
    )

    print(json.dumps({"endpoints": results}, indent=2, sort_keys=True))


def update_many(
    *,
    api: HfApi,
    namespace: str | None,
    names: list[str],
    env_updates: dict[str, str],
    unset_env: list[str],
    secret_updates: dict[str, str],
    wait: bool,
    wait_timeout_s: int,
    wait_refresh_every_s: int,
    dry_run: bool,
    parallelism: int,
) -> list[dict[str, object]]:
    total = len(names)
    if total == 0:
        return []

    max_workers = total if parallelism <= 0 else min(total, parallelism)
    if max_workers == 1:
        return update_many_sequential(
            api=api,
            namespace=namespace,
            names=names,
            env_updates=env_updates,
            unset_env=unset_env,
            secret_updates=secret_updates,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            dry_run=dry_run,
        )

    log_progress(f"Updating {total} compute endpoint envs in parallel with {max_workers} workers")
    results: list[dict[str, object] | None] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[dict[str, object]], tuple[int, str]] = {}
        for index, name in enumerate(names, start=1):
            log_progress(f"[{index}/{total}] Submitting compute env update for {name}")
            future = executor.submit(
                update_one,
                api=api,
                namespace=namespace,
                name=name,
                env_updates=env_updates,
                unset_env=unset_env,
                secret_updates=secret_updates,
                wait=wait,
                wait_timeout_s=wait_timeout_s,
                wait_refresh_every_s=wait_refresh_every_s,
                dry_run=dry_run,
            )
            futures[future] = (index, name)

        try:
            for future in as_completed(futures):
                index, name = futures[future]
                result = future.result()
                log_progress(f"[{index}/{total}] {name}: {result['status_before']} -> {result['status_after']}")
                results[index - 1] = result
        except BaseException:
            for pending_future in futures:
                if pending_future is not future:
                    pending_future.cancel()
            raise

    return [result for result in results if result is not None]


def update_many_sequential(
    *,
    api: HfApi,
    namespace: str | None,
    names: list[str],
    env_updates: dict[str, str],
    unset_env: list[str],
    secret_updates: dict[str, str],
    wait: bool,
    wait_timeout_s: int,
    wait_refresh_every_s: int,
    dry_run: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    total = len(names)
    for index, name in enumerate(names, start=1):
        log_progress(f"[{index}/{total}] Updating env on compute endpoint {name}")
        result = update_one(
            api=api,
            namespace=namespace,
            name=name,
            env_updates=env_updates,
            unset_env=unset_env,
            secret_updates=secret_updates,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            dry_run=dry_run,
        )
        log_progress(f"[{index}/{total}] {name}: {result['status_before']} -> {result['status_after']}")
        results.append(result)
    return results


def update_one(
    *,
    api: HfApi,
    namespace: str | None,
    name: str,
    env_updates: dict[str, str],
    unset_env: list[str],
    secret_updates: dict[str, str],
    wait: bool,
    wait_timeout_s: int,
    wait_refresh_every_s: int,
    dry_run: bool,
) -> dict[str, object]:
    endpoint = api.get_inference_endpoint(name, namespace=namespace)
    status_before = str(endpoint.status)
    target_status = expected_target_status(status_before)
    current_env = current_model_env(endpoint.raw)
    updated_env = merge_env_updates(current_env, env_updates, unset_env)

    changed = {
        key: updated_env[key]
        for key in sorted(updated_env)
        if current_env.get(key) != updated_env[key]
    }
    removed = sorted(key for key in current_env if key not in updated_env)
    secrets_set = sorted(secret_updates.keys())

    result = {
        "name": name,
        "status_before": status_before,
        "expected_status_after": target_status,
        "url": getattr(endpoint, "url", None),
        "changed": changed,
        "removed": removed,
        "secrets_set": secrets_set,
        "skipped": False,
    }

    env_unchanged = current_env == updated_env
    if env_unchanged and not secret_updates:
        result["skipped"] = True
        result["status_after"] = str(endpoint.status)
        return result

    if dry_run:
        result["status_after"] = "dry_run"
        return result

    if wait:
        log_progress(
            f"Waiting for {name} to finish updating and return to "
            f"{target_status} (timeout {wait_timeout_s}s, poll every {wait_refresh_every_s}s)"
        )
    endpoint = api.update_inference_endpoint(
        name,
        namespace=namespace,
        env=updated_env,
        secrets=secret_updates or None,
    )
    if wait:
        wait_for_endpoint_update(
            endpoint,
            target_status=target_status,
            timeout=wait_timeout_s,
            refresh_every=wait_refresh_every_s,
        )
    result["status_after"] = str(endpoint.status)
    return result


if __name__ == "__main__":
    main()

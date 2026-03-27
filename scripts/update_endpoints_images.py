#!/usr/bin/env python3
import argparse
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait as wait_futures
import json
import sys
import time
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, InferenceEndpointError, InferenceEndpointTimeoutError

from _endpoint_helpers import build_names, current_custom_image


DEFAULT_LOAD_BALANCER_NAME = "reachy-s2s-lb"
DEFAULT_COMPUTE_INDEX_START = 1
DEFAULT_COMPUTE_INDEX_WIDTH = 2
FAILED_UPDATE_STATUSES = {"failed", "updateFailed"}
PARKED_STATUSES = {"paused", "scaledToZero"}


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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
        "--compute-parallelism",
        type=int,
        default=0,
        help="Number of compute endpoint updates to run in parallel. Default: all selected compute endpoints. Use 1 for sequential updates.",
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
    parser.add_argument("--dry-run", action="store_true", help="Print planned updates without applying them")
    args = parser.parse_args()

    if not args.compute and not args.load_balancer:
        raise ValueError("Provide at least one of --compute or --load_balancer")

    api = HfApi()
    results: dict[str, object] = {}

    if args.compute:
        compute_names, selection = resolve_compute_names(
            api=api,
            namespace=args.namespace,
            load_balancer_name=args.load_balancer_name,
            explicit_names=args.compute_names,
            prefix=args.compute_prefix,
            count=args.compute_count,
        )
        log_progress(
            f"Discovered {len(compute_names)} compute endpoints via {selection['discovery']}: "
            f"{build_compute_summary(compute_names, selection)}"
        )
        results["compute"] = {
            **selection,
            "summary": build_compute_summary(compute_names, selection),
            "updates": update_many(
                api=api,
                namespace=args.namespace,
                names=compute_names,
                image_url=args.compute,
                wait=args.wait,
                wait_timeout_s=args.wait_timeout_s,
                wait_refresh_every_s=args.wait_refresh_every_s,
                dry_run=args.dry_run,
                parallelism=args.compute_parallelism,
            ),
        }

    if args.load_balancer:
        log_progress(f"Updating load balancer endpoint {args.load_balancer_name}")
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
) -> tuple[list[str], dict[str, Any]]:
    if explicit_names:
        return explicit_names, {"discovery": "explicit_names"}

    if prefix or count is not None:
        names = build_names(prefix, count, explicit_names)
        return names, {
            "discovery": "prefix_count",
            "prefix": prefix,
            "start_index": DEFAULT_COMPUTE_INDEX_START,
            "end_index": len(names),
        }

    inferred_prefix = default_compute_prefix(load_balancer_name)
    return discover_sequential_compute_names(
        api=api,
        namespace=namespace,
        prefix=inferred_prefix,
    )


def default_compute_prefix(load_balancer_name: str) -> str:
    if load_balancer_name.endswith("-lb") and len(load_balancer_name) > len("-lb"):
        return load_balancer_name[: -len("-lb")]
    return load_balancer_name


def discover_sequential_compute_names(
    *,
    api: HfApi,
    namespace: str | None,
    prefix: str,
) -> tuple[list[str], dict[str, Any]]:
    names: list[str] = []
    index = DEFAULT_COMPUTE_INDEX_START

    while True:
        name = f"{prefix}-{index:0{DEFAULT_COMPUTE_INDEX_WIDTH}d}"
        try:
            api.get_inference_endpoint(name, namespace=namespace)
        except HfHubHTTPError as exc:
            if not is_not_found_error(exc):
                raise
            if index == DEFAULT_COMPUTE_INDEX_START:
                raise ValueError(
                    f"Could not discover compute endpoints with prefix {prefix!r}. "
                    f"Expected to find {name!r} first."
                ) from exc
            break
        names.append(name)
        index += 1

    return names, {
        "discovery": "sequential_scan",
        "prefix": prefix,
        "start_index": DEFAULT_COMPUTE_INDEX_START,
        "end_index": names_to_end_index(names),
    }


def names_to_end_index(names: list[str]) -> int:
    return DEFAULT_COMPUTE_INDEX_START + len(names) - 1


def build_compute_summary(names: list[str], selection: dict[str, Any]) -> str:
    if not names:
        return "updated 0 compute endpoints"

    start_index = selection.get("start_index")
    end_index = selection.get("end_index")
    if isinstance(start_index, int) and isinstance(end_index, int):
        if start_index == end_index:
            return f"updated endpoint {start_index}"
        return f"updated endpoints {start_index} through {end_index}"

    count = len(names)
    noun = "endpoint" if count == 1 else "endpoints"
    return f"updated {count} compute {noun}"


def is_not_found_error(exc: HfHubHTTPError) -> bool:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 404:
        return True
    message = str(exc).lower()
    return "404" in message or "not found" in message


def expected_target_status(status_before: str) -> str:
    return status_before if status_before in PARKED_STATUSES else "running"


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
            image_url=image_url,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            dry_run=dry_run,
        )

    log_progress(f"Updating {total} compute endpoints in parallel with {max_workers} workers")
    results: list[dict[str, object] | None] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[dict[str, object]], tuple[int, str]] = {}
        for index, name in enumerate(names, start=1):
            log_progress(f"[{index}/{total}] Submitting compute endpoint update for {name}")
            future = executor.submit(
                update_one,
                api=api,
                namespace=namespace,
                name=name,
                image_url=image_url,
                wait=wait,
                wait_timeout_s=wait_timeout_s,
                wait_refresh_every_s=wait_refresh_every_s,
                dry_run=dry_run,
            )
            futures[future] = (index, name)

        done, not_done = wait_futures(futures, return_when=FIRST_EXCEPTION)
        first_error: BaseException | None = None
        for future in done:
            exc = future.exception()
            if exc is not None:
                first_error = exc
                break

        if first_error is not None:
            for future in not_done:
                future.cancel()
            raise first_error

        for future, (index, name) in futures.items():
            result = future.result()
            log_progress(
                f"[{index}/{total}] {name}: {result['status_before']} -> {result['status_after']}"
            )
            results[index - 1] = result

    return [result for result in results if result is not None]


def update_many_sequential(
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
    results: list[dict[str, object]] = []
    total = len(names)
    for index, name in enumerate(names, start=1):
        log_progress(f"[{index}/{total}] Updating compute endpoint {name}")
        result = update_one(
            api=api,
            namespace=namespace,
            name=name,
            image_url=image_url,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            dry_run=dry_run,
        )
        log_progress(
            f"[{index}/{total}] {name}: {result['status_before']} -> {result['status_after']}"
        )
        results.append(result)
    return results


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
    status_before = str(endpoint.status)
    current_image = current_custom_image(endpoint.raw)
    updated_image = dict(current_image)
    updated_image["url"] = image_url
    target_status = expected_target_status(status_before)

    result = {
        "name": name,
        "status_before": status_before,
        "url": getattr(endpoint, "url", None),
        "image_before": current_image["url"],
        "image_after": image_url,
        "health_route": current_image["health_route"],
        "port": current_image["port"],
        "expected_status_after": target_status,
        "skipped": False,
    }

    if current_image["url"] == image_url:
        result["skipped"] = True
        result["status_after"] = str(endpoint.status)
        return result

    if dry_run:
        result["status_after"] = "dry_run"
        return result

    if wait:
        log_progress(
            f"Waiting for {name} to finish updating "
            f"and return to {target_status} (timeout {wait_timeout_s}s, poll every {wait_refresh_every_s}s)"
        )
    endpoint = api.update_inference_endpoint(
        name,
        namespace=namespace,
        custom_image=updated_image,
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
        if current_status == target_status:
            endpoint.fetch()
            return endpoint
        if timeout is not None and time.time() - start > timeout:
            raise InferenceEndpointTimeoutError(
                f"Timeout while waiting for Inference Endpoint {endpoint.name} to return to {target_status}."
            )
        time.sleep(refresh_every)
        endpoint.fetch()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import os
import sys
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import HfHubHTTPError, InferenceEndpointError, InferenceEndpointTimeoutError

from _endpoint_helpers import (
    DEFAULT_LOAD_BALANCER_HEALTH_ROUTE,
    build_names,
    current_custom_image,
    current_model_env,
)


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
        help="Number of compute endpoint updates to run in parallel. Default: all selected compute endpoints, or 1 with --compute-drain.",
    )
    parser.add_argument(
        "--compute-drain",
        action="store_true",
        help="Ask the load balancer to stop assigning each compute endpoint before waiting for it to become idle and updating it.",
    )
    parser.add_argument("--load-balancer-url", help="Load-balancer endpoint URL used by --compute-drain")
    parser.add_argument(
        "--compute-drain-timeout-s",
        type=int,
        default=7200,
        help="Timeout while waiting for a drained compute endpoint to reach zero active sessions.",
    )
    parser.add_argument(
        "--compute-drain-refresh-every-s",
        type=int,
        default=10,
        help="Polling interval while waiting for a drained compute endpoint to become idle.",
    )
    parser.add_argument(
        "--load-balancer-request-timeout-s",
        type=float,
        default=30,
        help="HTTP timeout for load-balancer drain and status requests.",
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
    parser.add_argument(
        "--token",
        default=default_token(),
        help="Hugging Face token. Defaults to HF_TOKEN, HF_CONTROL_TOKEN, or the locally saved token.",
    )
    parser.add_argument(
        "--load-balancer-admin-token",
        default=os.getenv("LB_ADMIN_AUTH_TOKEN"),
        help="Dedicated bearer token for load-balancer admin routes. Defaults only to LB_ADMIN_AUTH_TOKEN.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned updates without applying them")
    args = parser.parse_args()

    if not args.compute and not args.load_balancer:
        raise ValueError("Provide at least one of --compute or --load_balancer")
    if args.compute_drain and not args.wait:
        raise ValueError("--compute-drain cannot be combined with --no-wait")
    if args.compute_drain and not args.dry_run and not args.load_balancer_admin_token:
        raise ValueError(
            "--compute-drain requires --load-balancer-admin-token or LB_ADMIN_AUTH_TOKEN"
        )

    api = HfApi(token=args.token or None)
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
        compute_namespace = selection.get("namespace", args.namespace)
        compute_parallelism = args.compute_parallelism
        if args.compute_drain and compute_parallelism <= 0:
            compute_parallelism = 1
        if args.compute_drain:
            load_balancer_url = resolve_load_balancer_url(
                api=api,
                namespace=args.namespace,
                load_balancer_name=args.load_balancer_name,
                explicit_url=args.load_balancer_url,
                selection=selection,
            )
            update_results = update_many_draining(
                api=api,
                namespace=compute_namespace if isinstance(compute_namespace, str) else args.namespace,
                names=compute_names,
                image_url=args.compute,
                load_balancer_url=load_balancer_url,
                token=args.load_balancer_admin_token,
                wait=args.wait,
                wait_timeout_s=args.wait_timeout_s,
                wait_refresh_every_s=args.wait_refresh_every_s,
                drain_timeout_s=args.compute_drain_timeout_s,
                drain_refresh_every_s=args.compute_drain_refresh_every_s,
                request_timeout_s=args.load_balancer_request_timeout_s,
                dry_run=args.dry_run,
                parallelism=compute_parallelism,
            )
        else:
            update_results = update_many(
                api=api,
                namespace=compute_namespace if isinstance(compute_namespace, str) else args.namespace,
                names=compute_names,
                image_url=args.compute,
                wait=args.wait,
                wait_timeout_s=args.wait_timeout_s,
                wait_refresh_every_s=args.wait_refresh_every_s,
                dry_run=args.dry_run,
                parallelism=compute_parallelism,
            )
        results["compute"] = {
            **selection,
            "summary": build_compute_summary(compute_names, selection),
            "drain": args.compute_drain,
            "updates": update_results,
        }

    if args.load_balancer:
        log_progress(f"Updating load balancer endpoint {args.load_balancer_name}")
        results["load_balancer"] = update_one(
            api=api,
            namespace=args.namespace,
            name=args.load_balancer_name,
            image_url=args.load_balancer,
            health_route_override=DEFAULT_LOAD_BALANCER_HEALTH_ROUTE,
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
            "namespace": namespace,
            "start_index": DEFAULT_COMPUTE_INDEX_START,
            "end_index": len(names),
        }

    load_balancer_names = discover_load_balancer_compute_names(
        api=api,
        namespace=namespace,
        load_balancer_name=load_balancer_name,
    )
    if load_balancer_names is not None:
        return load_balancer_names

    inferred_prefix = default_compute_prefix(load_balancer_name)
    return discover_sequential_compute_names(
        api=api,
        namespace=namespace,
        prefix=inferred_prefix,
    )


def discover_load_balancer_compute_names(
    *,
    api: HfApi,
    namespace: str | None,
    load_balancer_name: str,
) -> tuple[list[str], dict[str, Any]] | None:
    try:
        endpoint = api.get_inference_endpoint(load_balancer_name, namespace=namespace)
    except HfHubHTTPError as exc:
        if is_not_found_error(exc):
            return None
        raise
    env = current_model_env(endpoint.raw)
    names = parse_compute_endpoint_names(env.get("COMPUTE_ENDPOINT_NAMES", ""))
    if not names:
        return None

    compute_namespace = env.get("HF_ENDPOINT_NAMESPACE", "").strip() or namespace
    selection: dict[str, Any] = {
        "discovery": "load_balancer_env",
        "load_balancer_name": load_balancer_name,
        "namespace": compute_namespace,
    }
    load_balancer_url = getattr(endpoint, "url", None)
    if load_balancer_url:
        selection["load_balancer_url"] = str(load_balancer_url)
    return names, selection


def parse_compute_endpoint_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def default_compute_prefix(load_balancer_name: str) -> str:
    if load_balancer_name.endswith("-lb") and len(load_balancer_name) > len("-lb"):
        return load_balancer_name[: -len("-lb")]
    return load_balancer_name


def default_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HF_CONTROL_TOKEN") or get_token()


def resolve_load_balancer_url(
    *,
    api: HfApi,
    namespace: str | None,
    load_balancer_name: str,
    explicit_url: str | None,
    selection: dict[str, Any],
) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")

    selected_url = selection.get("load_balancer_url")
    if isinstance(selected_url, str) and selected_url.strip():
        return selected_url.strip().rstrip("/")

    endpoint = api.get_inference_endpoint(load_balancer_name, namespace=namespace)
    endpoint_url = getattr(endpoint, "url", None)
    if not endpoint_url:
        raise ValueError("Could not resolve load-balancer URL. Pass --load-balancer-url.")
    return str(endpoint_url).rstrip("/")


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
        "namespace": namespace,
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
    return "parked" if status_before in PARKED_STATUSES else "running"


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

        try:
            for future in as_completed(futures):
                index, name = futures[future]
                result = future.result()
                log_progress(
                    f"[{index}/{total}] {name}: {result['status_before']} -> {result['status_after']}"
                )
                results[index - 1] = result
        except BaseException:
            for pending_future in futures:
                if pending_future is not future:
                    pending_future.cancel()
            raise

    return [result for result in results if result is not None]


def update_many_draining(
    *,
    api: HfApi,
    namespace: str | None,
    names: list[str],
    image_url: str,
    load_balancer_url: str,
    token: str | None,
    wait: bool,
    wait_timeout_s: int,
    wait_refresh_every_s: int,
    drain_timeout_s: int,
    drain_refresh_every_s: int,
    request_timeout_s: float,
    dry_run: bool,
    parallelism: int,
) -> list[dict[str, object]]:
    if dry_run:
        return update_many(
            api=api,
            namespace=namespace,
            names=names,
            image_url=image_url,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            dry_run=dry_run,
            parallelism=parallelism,
        )

    total = len(names)
    if total == 0:
        return []

    max_workers = total if parallelism <= 0 else min(total, parallelism)
    if max_workers == 1:
        return update_many_draining_sequential(
            api=api,
            namespace=namespace,
            names=names,
            image_url=image_url,
            load_balancer_url=load_balancer_url,
            token=token,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            drain_timeout_s=drain_timeout_s,
            drain_refresh_every_s=drain_refresh_every_s,
            request_timeout_s=request_timeout_s,
        )

    log_progress(f"Draining and updating {total} compute endpoints in parallel with {max_workers} workers")
    results: list[dict[str, object] | None] = [None] * total
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[dict[str, object]], tuple[int, str]] = {}
        for index, name in enumerate(names, start=1):
            log_progress(f"[{index}/{total}] Submitting drained compute endpoint update for {name}")
            future = executor.submit(
                update_one_draining,
                api=api,
                namespace=namespace,
                name=name,
                image_url=image_url,
                load_balancer_url=load_balancer_url,
                token=token,
                wait=wait,
                wait_timeout_s=wait_timeout_s,
                wait_refresh_every_s=wait_refresh_every_s,
                drain_timeout_s=drain_timeout_s,
                drain_refresh_every_s=drain_refresh_every_s,
                request_timeout_s=request_timeout_s,
            )
            futures[future] = (index, name)

        try:
            for future in as_completed(futures):
                index, name = futures[future]
                result = future.result()
                log_progress(
                    f"[{index}/{total}] {name}: {result['status_before']} -> {result['status_after']}"
                )
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


def update_many_draining_sequential(
    *,
    api: HfApi,
    namespace: str | None,
    names: list[str],
    image_url: str,
    load_balancer_url: str,
    token: str | None,
    wait: bool,
    wait_timeout_s: int,
    wait_refresh_every_s: int,
    drain_timeout_s: int,
    drain_refresh_every_s: int,
    request_timeout_s: float,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    total = len(names)
    for index, name in enumerate(names, start=1):
        log_progress(f"[{index}/{total}] Draining compute endpoint {name}")
        result = update_one_draining(
            api=api,
            namespace=namespace,
            name=name,
            image_url=image_url,
            load_balancer_url=load_balancer_url,
            token=token,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            drain_timeout_s=drain_timeout_s,
            drain_refresh_every_s=drain_refresh_every_s,
            request_timeout_s=request_timeout_s,
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
    health_route_override: str | None = None,
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
    if health_route_override:
        updated_image["health_route"] = health_route_override
    target_status = expected_target_status(status_before)

    result = {
        "name": name,
        "status_before": status_before,
        "url": getattr(endpoint, "url", None),
        "image_before": current_image["url"],
        "image_after": image_url,
        "health_route": updated_image["health_route"],
        "port": current_image["port"],
        "expected_status_after": target_status,
        "skipped": False,
    }

    if (
        current_image["url"] == image_url
        and current_image["health_route"] == updated_image["health_route"]
    ):
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


def update_one_draining(
    *,
    api: HfApi,
    namespace: str | None,
    name: str,
    image_url: str,
    load_balancer_url: str,
    token: str | None,
    wait: bool,
    wait_timeout_s: int,
    wait_refresh_every_s: int,
    drain_timeout_s: int,
    drain_refresh_every_s: int,
    request_timeout_s: float,
) -> dict[str, object]:
    result: dict[str, object] | None = None
    drain_attempted = False
    try:
        drain_attempted = True
        set_compute_endpoint_draining(
            load_balancer_url=load_balancer_url,
            token=token,
            name=name,
            draining=True,
            timeout_s=request_timeout_s,
        )

        drain_snapshot = wait_for_compute_endpoint_free(
            load_balancer_url=load_balancer_url,
            token=token,
            name=name,
            timeout_s=drain_timeout_s,
            refresh_every_s=drain_refresh_every_s,
            request_timeout_s=request_timeout_s,
        )
        result = update_one(
            api=api,
            namespace=namespace,
            name=name,
            image_url=image_url,
            wait=wait,
            wait_timeout_s=wait_timeout_s,
            wait_refresh_every_s=wait_refresh_every_s,
            dry_run=False,
        )
        result["drain"] = {
            "load_balancer_url": load_balancer_url,
            "active_sessions_before_update": int(drain_snapshot.get("active_sessions", 0) or 0),
            "draining_before_update": bool(drain_snapshot.get("draining", False)),
        }
        return result
    finally:
        if drain_attempted:
            # With wait=False this reopens the endpoint immediately after submitting
            # the update; use wait=True when it must stay drained until healthy.
            try:
                set_compute_endpoint_draining(
                    load_balancer_url=load_balancer_url,
                    token=token,
                    name=name,
                    draining=False,
                    timeout_s=request_timeout_s,
                )
            except Exception as exc:
                log_progress(f"Failed to clear drain on {name}: {exc}")
                if result is not None:
                    raise


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


def wait_for_compute_endpoint_free(
    *,
    load_balancer_url: str,
    token: str | None,
    name: str,
    timeout_s: int,
    refresh_every_s: int,
    request_timeout_s: float,
) -> dict[str, object]:
    start = time.time()
    while True:
        endpoint = fetch_compute_endpoint_snapshot(
            load_balancer_url=load_balancer_url,
            token=token,
            name=name,
            timeout_s=request_timeout_s,
        )
        ready, detail = compute_endpoint_ready_for_update(endpoint, name=name)
        if ready:
            return endpoint

        elapsed = time.time() - start
        if timeout_s is not None and elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for compute endpoint {name} to become free "
                f"({detail})."
            )

        log_progress(
            f"{name} is not yet safe to update ({detail}); "
            f"waiting {refresh_every_s}s before checking again"
        )
        time.sleep(refresh_every_s)


def compute_endpoint_ready_for_update(
    endpoint: dict[str, object],
    *,
    name: str,
) -> tuple[bool, str]:
    if endpoint.get("draining") is not True:
        raise RuntimeError(
            f"Compute endpoint {name} is no longer draining; "
            "the load balancer may have restarted. Refusing to update."
        )

    status = str(endpoint.get("status", ""))
    active_sessions = int(endpoint.get("active_sessions", 0) or 0)
    if status in PARKED_STATUSES:
        if active_sessions != 0:
            raise RuntimeError(
                f"Parked compute endpoint {name} unexpectedly reports "
                f"{active_sessions} active session(s). Refusing to update."
            )
        return True, "endpoint is parked with zero active sessions"

    if endpoint.get("running") is not True:
        return False, f"status {status!r} is neither stably running nor parked"

    require_usage_sync = bool(endpoint.get("require_usage_sync", False))
    usage_synced = bool(endpoint.get("usage_synced", False))
    if require_usage_sync and not usage_synced:
        return False, "usage sync is not trustworthy"
    if active_sessions == 0:
        return True, "running endpoint has trustworthy zero active sessions"
    return False, f"{active_sessions} active session(s) remain"


def fetch_compute_endpoint_snapshot(
    *,
    load_balancer_url: str,
    token: str | None,
    name: str,
    timeout_s: float,
) -> dict[str, object]:
    escaped_name = quote(name, safe="")
    payload = request_json(
        f"{load_balancer_url.rstrip('/')}/internal/endpoints/{escaped_name}",
        token=token,
        timeout_s=timeout_s,
    )
    endpoint = payload.get("endpoint")
    if not isinstance(endpoint, dict):
        raise ValueError("Load-balancer endpoint status response does not include an endpoint snapshot")
    return endpoint


def set_compute_endpoint_draining(
    *,
    load_balancer_url: str,
    token: str | None,
    name: str,
    draining: bool,
    timeout_s: float,
) -> dict[str, Any]:
    escaped_name = quote(name, safe="")
    return request_json(
        f"{load_balancer_url.rstrip('/')}/internal/endpoints/{escaped_name}/drain",
        method="POST",
        token=token,
        payload={"draining": draining},
        timeout_s=timeout_s,
    )


def request_json(
    url: str,
    *,
    method: str = "GET",
    token: str | None,
    payload: dict[str, object] | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    headers: dict[str, str] = {}
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=timeout_s) as response:
        body = response.read()

    return json.loads(body.decode("utf-8"))


if __name__ == "__main__":
    main()

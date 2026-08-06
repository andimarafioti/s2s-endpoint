import json
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

from huggingface_hub.errors import InferenceEndpointError, InferenceEndpointTimeoutError

DEFAULT_REPOSITORY = "andito/s2s"
DEFAULT_FRAMEWORK = "custom"
DEFAULT_ENDPOINT_TYPE = "protected"
DEFAULT_HEALTH_ROUTE = "/health"
DEFAULT_LOAD_BALANCER_HEALTH_ROUTE = "/ready"
DEFAULT_IMAGE_PORT = 7860
FAILED_UPDATE_STATUSES = {"failed", "updateFailed"}
PARKED_STATUSES = {"paused", "scaledToZero"}
BatchResult = TypeVar("BatchResult")


def load_json_file(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_key_value_pairs(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def build_names(prefix: str | None, count: int | None, names: list[str]) -> list[str]:
    if names:
        if prefix or count:
            raise ValueError("Use either --names or --prefix/--count, not both")
        return names

    if not prefix or count is None:
        raise ValueError("Provide either --names or both --prefix and --count")

    if count < 1:
        raise ValueError("--count must be >= 1")

    width = max(2, len(str(count)))
    return [f"{prefix}-{idx:0{width}d}" for idx in range(1, count + 1)]


def build_custom_image(url: str, health_route: str, port: int) -> dict[str, str | int]:
    return {
        "url": url,
        "health_route": health_route,
        "port": port,
    }


def current_model_env(raw: dict[str, Any]) -> dict[str, str]:
    model = raw.get("model") or {}
    env = model.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("endpoint model env must be a dictionary")
    return {str(key): str(value) for key, value in env.items()}


def current_custom_image(raw: dict[str, Any]) -> dict[str, str | int]:
    model = raw.get("model") or {}
    image = model.get("image") or {}
    custom = image.get("custom") or {}
    if not isinstance(custom, dict):
        raise ValueError("endpoint custom image must be a dictionary")

    url = str(custom.get("url") or "").strip()
    if not url:
        raise ValueError("endpoint does not have a custom image url")

    health_route = (
        str(custom.get("health_route") or custom.get("healthRoute") or DEFAULT_HEALTH_ROUTE).strip()
        or DEFAULT_HEALTH_ROUTE
    )

    port_value = custom.get("port", DEFAULT_IMAGE_PORT)
    try:
        port = int(port_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid endpoint custom image port: {port_value!r}") from exc

    return {
        "url": url,
        "health_route": health_route,
        "port": port,
    }


def merge_env_updates(
    current_env: dict[str, str] | None,
    updates: dict[str, str],
    unset_keys: list[str],
) -> dict[str, str]:
    merged = dict(current_env or {})
    merged.update(updates)
    for key in unset_keys:
        merged.pop(key, None)
    return merged


def expected_target_status(status_before: str) -> str:
    return "parked" if status_before in PARKED_STATUSES else "running"


def wait_for_endpoint_update(
    endpoint,
    *,
    target_status: str,
    timeout: float | None,
    refresh_every: float,
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


def run_ordered_batch(
    *,
    names: list[str],
    worker: Callable[[str], BatchResult],
    parallelism: int,
    progress: Callable[[str], None],
    parallel_start_message: str,
    sequential_start_message: str,
    parallel_submit_message: str,
    completed_message: Callable[[BatchResult], str],
) -> list[BatchResult]:
    total = len(names)
    if total == 0:
        return []

    max_workers = total if parallelism <= 0 else min(total, parallelism)
    if max_workers == 1:
        results: list[BatchResult] = []
        for index, name in enumerate(names, start=1):
            progress(sequential_start_message.format(index=index, total=total, name=name))
            result = worker(name)
            progress(f"[{index}/{total}] {name}: {completed_message(result)}")
            results.append(result)
        return results

    progress(parallel_start_message.format(total=total, max_workers=max_workers))
    results: dict[int, BatchResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future[BatchResult], tuple[int, str]] = {}
        for index, name in enumerate(names, start=1):
            progress(parallel_submit_message.format(index=index, total=total, name=name))
            future = executor.submit(worker, name)
            futures[future] = (index, name)

        try:
            for future in as_completed(futures):
                index, name = futures[future]
                result = future.result()
                progress(f"[{index}/{total}] {name}: {completed_message(result)}")
                results[index - 1] = result
        except BaseException:
            for pending_future in futures:
                if pending_future is not future:
                    pending_future.cancel()
            raise

    return [results[index] for index in range(total)]

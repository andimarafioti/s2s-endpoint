#!/usr/bin/env python3
import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen

from _endpoint_helpers import build_names, current_model_env


DEFAULT_ENDPOINTS_API_BASE = "https://api.endpoints.huggingface.cloud"
DEFAULT_LOAD_BALANCER_NAME = "reachy-s2s-lb"
DEFAULT_OUTPUT_DIR = "logs/endpoints"
DEFAULT_TAIL_LINES = 10000
DEFAULT_LINE_MAX_LENGTH = 20000
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_PARALLELISM = 8
PROGRESS_MODE = "normal"


@dataclass(frozen=True)
class EndpointLogTarget:
    namespace: str
    name: str
    replica: str | None = None


def log_progress(message: str, *, verbose: bool = False) -> None:
    if PROGRESS_MODE == "quiet":
        return
    if verbose and PROGRESS_MODE != "verbose":
        return
    print(message, file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face Inference Endpoint logs into a local folder."
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("HF_ENDPOINT_NAMESPACE") or os.getenv("HF_NAMESPACE"),
        help="Load-balancer endpoint namespace / org. Defaults to HF_ENDPOINT_NAMESPACE or HF_NAMESPACE.",
    )
    parser.add_argument(
        "--compute-namespace",
        default=os.getenv("HF_COMPUTE_ENDPOINT_NAMESPACE"),
        help="Compute endpoint namespace / org. Defaults to the load-balancer namespace or HF_ENDPOINT_NAMESPACE from the LB env.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HF_CONTROL_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN or HF_CONTROL_TOKEN.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Folder to write logs into (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument(
        "--names",
        nargs="*",
        default=[],
        help="Explicit endpoint names to download from --namespace. When set, skips load-balancer/compute discovery.",
    )
    parser.add_argument(
        "--load-balancer-name",
        default=DEFAULT_LOAD_BALANCER_NAME,
        help=f"Load-balancer endpoint name (default: {DEFAULT_LOAD_BALANCER_NAME})",
    )
    parser.add_argument("--skip-load-balancer", action="store_true", help="Do not download load-balancer logs")
    parser.add_argument("--no-compute", action="store_true", help="Do not download compute endpoint logs")
    parser.add_argument("--compute-prefix", help="Compute endpoint name prefix, used with --compute-count")
    parser.add_argument("--compute-count", type=int, help="Number of compute endpoints, used with --compute-prefix")
    parser.add_argument("--compute-names", nargs="*", default=[], help="Explicit compute endpoint names")
    parser.add_argument(
        "--tail",
        type=int,
        default=DEFAULT_TAIL_LINES,
        help=f"Max log lines to request from each endpoint (default: {DEFAULT_TAIL_LINES})",
    )
    parser.add_argument(
        "--line-max-length",
        type=int,
        default=DEFAULT_LINE_MAX_LENGTH,
        help=f"Max length per log line requested from the API (default: {DEFAULT_LINE_MAX_LENGTH})",
    )
    parser.add_argument("--replica", help="Optional replica id to download from each endpoint")
    parser.add_argument(
        "--all-replicas",
        action="store_true",
        help="Discover replica ids from endpoint metrics for --since/--until and download logs for every replica.",
    )
    parser.add_argument("--since", help="RFC3339 lower time bound for v3 logs and replica discovery")
    parser.add_argument("--until", help="RFC3339 upper time bound for v3 logs and replica discovery")
    parser.add_argument(
        "--log-api-version",
        choices=["v2", "v3"],
        help="Log API to use. Defaults to v3 for --all-replicas or --since/--until, otherwise v2.",
    )
    parser.add_argument("--v3-limit", type=int, default=5000, help="Max rows per v3 logs page (default: 5000)")
    parser.add_argument("--max-pages", type=int, default=100, help="Max v3 log pages per endpoint/replica (default: 100)")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=DEFAULT_PARALLELISM,
        help=f"Number of endpoints to download in parallel (default: {DEFAULT_PARALLELISM})",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-endpoint log download timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress lines; only print final JSON")
    parser.add_argument("--verbose", action="store_true", help="Print per-endpoint/replica progress lines")
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include one JSON result object per endpoint/replica in the final output",
    )
    parser.add_argument("--api-base", default=DEFAULT_ENDPOINTS_API_BASE, help=argparse.SUPPRESS)
    args = parser.parse_args()

    global PROGRESS_MODE
    PROGRESS_MODE = "quiet" if args.quiet else "verbose" if args.verbose else "normal"

    if not args.namespace:
        raise ValueError("Provide --namespace or set HF_ENDPOINT_NAMESPACE/HF_NAMESPACE")
    if not args.token:
        raise ValueError("Provide --token or set HF_TOKEN/HF_CONTROL_TOKEN")
    if args.tail < 0:
        raise ValueError("--tail must be >= 0")
    if args.line_max_length < 0:
        raise ValueError("--line-max-length must be >= 0")
    if args.parallelism < 1:
        raise ValueError("--parallelism must be >= 1")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if args.all_replicas and args.replica:
        raise ValueError("Use either --replica or --all-replicas, not both")
    if args.all_replicas and (not args.since or not args.until):
        raise ValueError("--all-replicas requires --since and --until")
    if args.v3_limit < 1 or args.v3_limit > 5000:
        raise ValueError("--v3-limit must be between 1 and 5000")
    if args.max_pages < 1:
        raise ValueError("--max-pages must be >= 1")

    log_api_version = args.log_api_version or ("v3" if args.all_replicas or args.since or args.until else "v2")

    targets, discovery = resolve_targets(
        api_base=args.api_base,
        namespace=args.namespace,
        compute_namespace=args.compute_namespace,
        token=args.token,
        names=args.names,
        load_balancer_name=args.load_balancer_name,
        skip_load_balancer=args.skip_load_balancer,
        no_compute=args.no_compute,
        compute_prefix=args.compute_prefix,
        compute_count=args.compute_count,
        compute_names=args.compute_names,
        timeout_s=args.timeout_s,
    )
    if not targets:
        raise ValueError("No endpoints selected")

    if args.replica:
        targets = [EndpointLogTarget(target.namespace, target.name, args.replica) for target in targets]
    if args.all_replicas:
        targets = expand_targets_with_replicas(
            api_base=args.api_base,
            token=args.token,
            targets=targets,
            since=args.since,
            until=args.until,
            timeout_s=args.timeout_s,
        )
        if not targets:
            raise ValueError("No endpoint replicas found for the requested time window")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"Downloading logs for {len(targets)} endpoint(s) into {output_dir}")

    results = download_many(
        api_base=args.api_base,
        token=args.token,
        targets=targets,
        output_dir=output_dir,
        tail=args.tail,
        line_max_length=args.line_max_length,
        timeout_s=args.timeout_s,
        parallelism=args.parallelism,
        api_version=log_api_version,
        since=args.since,
        until=args.until,
        v3_limit=args.v3_limit,
        max_pages=args.max_pages,
    )

    payload = {
        "discovery": discovery,
        "output_dir": str(output_dir),
        **summarize_results(results),
    }
    if args.include_results:
        payload["results"] = results
    print(json.dumps(payload, indent=2, sort_keys=True))


def resolve_targets(
    *,
    api_base: str,
    namespace: str,
    compute_namespace: str | None,
    token: str,
    names: list[str],
    load_balancer_name: str,
    skip_load_balancer: bool,
    no_compute: bool,
    compute_prefix: str | None,
    compute_count: int | None,
    compute_names: list[str],
    timeout_s: float,
) -> tuple[list[EndpointLogTarget], dict[str, object]]:
    if names:
        if any([compute_names, compute_prefix, compute_count is not None]):
            raise ValueError("Use either --names or the load-balancer/compute selection options, not both")
        return unique_targets(EndpointLogTarget(namespace, name) for name in names), {
            "mode": "explicit_names",
            "namespace": namespace,
        }

    targets: list[EndpointLogTarget] = []
    discovery: dict[str, object] = {
        "mode": "app_selection",
        "load_balancer_namespace": namespace,
        "compute_namespace": compute_namespace or namespace,
    }

    if not skip_load_balancer:
        targets.append(EndpointLogTarget(namespace, load_balancer_name))

    if no_compute:
        return unique_targets(targets), discovery

    if compute_names:
        resolved_compute_namespace = compute_namespace or namespace
        targets.extend(EndpointLogTarget(resolved_compute_namespace, name) for name in compute_names)
        discovery["compute"] = "explicit_compute_names"
        discovery["compute_namespace"] = resolved_compute_namespace
        return unique_targets(targets), discovery

    if compute_prefix or compute_count is not None:
        resolved_compute_namespace = compute_namespace or namespace
        resolved_names = build_names(compute_prefix, compute_count, [])
        targets.extend(EndpointLogTarget(resolved_compute_namespace, name) for name in resolved_names)
        discovery["compute"] = "prefix_count"
        discovery["compute_namespace"] = resolved_compute_namespace
        discovery["compute_prefix"] = compute_prefix
        discovery["compute_count"] = compute_count
        return unique_targets(targets), discovery

    lb_raw = get_endpoint(
        api_base=api_base,
        namespace=namespace,
        name=load_balancer_name,
        token=token,
        timeout_s=timeout_s,
    )
    env = current_model_env(lb_raw)
    discovered_names = parse_endpoint_names(env.get("COMPUTE_ENDPOINT_NAMES", ""))
    if not discovered_names:
        raise ValueError(
            "Could not discover compute endpoints from the load balancer. "
            "Pass --compute-names, --compute-prefix/--compute-count, or --no-compute."
        )

    resolved_compute_namespace = compute_namespace or env.get("HF_ENDPOINT_NAMESPACE", "").strip() or namespace
    targets.extend(EndpointLogTarget(resolved_compute_namespace, name) for name in discovered_names)
    discovery["compute"] = "load_balancer_env"
    discovery["compute_namespace"] = resolved_compute_namespace
    discovery["compute_count"] = len(discovered_names)
    return unique_targets(targets), discovery


def parse_endpoint_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def expand_targets_with_replicas(
    *,
    api_base: str,
    token: str,
    targets: list[EndpointLogTarget],
    since: str,
    until: str,
    timeout_s: float,
) -> list[EndpointLogTarget]:
    expanded: list[EndpointLogTarget] = []
    for target in targets:
        replica_ids = get_replica_ids(
            api_base=api_base,
            namespace=target.namespace,
            name=target.name,
            token=token,
            since=since,
            until=until,
            timeout_s=timeout_s,
        )
        log_progress(f"Discovered {len(replica_ids)} replica(s) for {target.name}")
        expanded.extend(EndpointLogTarget(target.namespace, target.name, replica_id) for replica_id in replica_ids)
    return unique_targets(expanded)


def get_replica_ids(
    *,
    api_base: str,
    namespace: str,
    name: str,
    token: str,
    since: str,
    until: str,
    timeout_s: float,
) -> list[str]:
    payload = post_json(
        url=build_replica_ids_url(api_base=api_base, namespace=namespace, name=name),
        token=token,
        payload={"start": since, "stop": until},
        timeout_s=timeout_s,
    )
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected replica id response for {name}: {payload!r}")
    return [str(replica_id) for replica_id in payload if str(replica_id).strip()]


def unique_targets(targets: Any) -> list[EndpointLogTarget]:
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[EndpointLogTarget] = []
    for target in targets:
        key = (target.namespace, target.name, target.replica)
        if key in seen:
            continue
        seen.add(key)
        unique.append(target)
    return unique


def get_endpoint(*, api_base: str, namespace: str, name: str, token: str, timeout_s: float) -> dict[str, Any]:
    url = build_endpoint_url(api_base=api_base, namespace=namespace, name=name)
    data = read_url(url=url, token=token, timeout_s=timeout_s)
    return json.loads(data.decode("utf-8"))


def download_many(
    *,
    api_base: str,
    token: str,
    targets: list[EndpointLogTarget],
    output_dir: Path,
    tail: int,
    line_max_length: int,
    timeout_s: float,
    parallelism: int,
    api_version: str,
    since: str | None,
    until: str | None,
    v3_limit: int,
    max_pages: int,
) -> list[dict[str, object]]:
    max_workers = min(parallelism, len(targets))
    results: list[dict[str, object] | None] = [None] * len(targets)
    if max_workers == 0:
        return []

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures: dict[Future[dict[str, object]], tuple[int, EndpointLogTarget]] = {}
        for index, target in enumerate(targets):
            log_progress(f"[{index + 1}/{len(targets)}] Starting {target_label(target)}", verbose=True)
            future = executor.submit(
                download_one,
                api_base=api_base,
                token=token,
                target=target,
                output_dir=output_dir,
                tail=tail,
                line_max_length=line_max_length,
                timeout_s=timeout_s,
                api_version=api_version,
                since=since,
                until=until,
                v3_limit=v3_limit,
                max_pages=max_pages,
            )
            futures[future] = (index, target)

        for future in as_completed(futures):
            index, target = futures[future]
            try:
                result = future.result()
                log_progress(
                    f"[{index + 1}/{len(targets)}] Wrote {target_label(target)} logs to {result['path']}",
                    verbose=True,
                )
            except Exception as exc:
                result = {
                    "namespace": target.namespace,
                    "name": target.name,
                    "replica": target.replica,
                    "skipped": False,
                    "error": str(exc),
                }
                log_progress(f"[{index + 1}/{len(targets)}] Failed to download {target_label(target)}: {exc}")
            results[index] = result
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return [result for result in results if result is not None]


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    downloaded = [result for result in results if not result.get("error")]
    failed = [result for result in results if result.get("error")]
    payload: dict[str, object] = {
        "log_files": len(results),
        "downloaded": len(downloaded),
        "failed": len(failed),
        "bytes": sum(int(result.get("bytes", 0)) for result in downloaded),
        "lines": sum(int(result.get("lines", 0)) for result in downloaded),
    }
    if failed:
        payload["failures"] = [
            {
                "namespace": result.get("namespace"),
                "name": result.get("name"),
                "replica": result.get("replica"),
                "error": result.get("error"),
            }
            for result in failed
        ]
    return payload


def target_label(target: EndpointLogTarget) -> str:
    if target.replica:
        return f"{target.name}/{target.replica}"
    return target.name


def download_one(
    *,
    api_base: str,
    token: str,
    target: EndpointLogTarget,
    output_dir: Path,
    tail: int,
    line_max_length: int,
    timeout_s: float,
    api_version: str,
    since: str | None,
    until: str | None,
    v3_limit: int,
    max_pages: int,
) -> dict[str, object]:
    path = output_dir / log_filename(target)
    if api_version == "v3":
        byte_count, line_count = download_v3_logs_to_file(
            api_base=api_base,
            token=token,
            target=target,
            path=path,
            since=since,
            until=until,
            line_max_length=line_max_length,
            limit=v3_limit,
            max_pages=max_pages,
            timeout_s=timeout_s,
        )
    else:
        url = build_logs_url(
            api_base=api_base,
            namespace=target.namespace,
            name=target.name,
            tail=tail,
            line_max_length=line_max_length,
            replica=target.replica,
        )
        byte_count, line_count = download_url_to_file(
            url=url,
            token=token,
            path=path,
            timeout_s=timeout_s,
        )
    return {
        "namespace": target.namespace,
        "name": target.name,
        "replica": target.replica,
        "path": str(path),
        "bytes": byte_count,
        "lines": line_count,
    }


def download_v3_logs_to_file(
    *,
    api_base: str,
    token: str,
    target: EndpointLogTarget,
    path: Path,
    since: str | None,
    until: str | None,
    line_max_length: int,
    limit: int,
    max_pages: int,
    timeout_s: float,
) -> tuple[int, int]:
    url = build_v3_logs_url(
        api_base=api_base,
        namespace=target.namespace,
        name=target.name,
        replica=target.replica,
        since=since,
        until=until,
        line_max_length=line_max_length,
        limit=limit,
    )
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    page_count = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as output:
            while url:
                page_count += 1
                if page_count > max_pages:
                    raise RuntimeError(f"Exceeded --max-pages={max_pages} while downloading {target.name}")
                body, headers = read_url_with_headers(url=url, token=token, timeout_s=timeout_s)
                for line in structured_log_lines(body):
                    output.write(line)
                    output.write("\n")
                url = next_link(headers.get("Link"), api_base=api_base)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return file_size_and_lines(path)


def download_url_to_file(*, url: str, token: str, path: Path, timeout_s: float) -> tuple[int, int]:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        try:
            run_curl_to_file(url=url, token=token, path=tmp_path, timeout_s=timeout_s)
        except FileNotFoundError:
            tmp_path.write_bytes(read_url(url=url, token=token, timeout_s=timeout_s))
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return file_size_and_lines(path)


def run_curl_to_file(*, url: str, token: str, path: Path, timeout_s: float) -> None:
    config = f'header = "Authorization: Bearer {escape_curl_config_value(token)}"\n'
    completed = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            str(timeout_s),
            "--connect-timeout",
            str(min(timeout_s, 10.0)),
            "--output",
            str(path),
            "--config",
            "-",
            url,
        ],
        input=config,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s + 5.0,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        message = f"curl exited with status {completed.returncode}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message)


def escape_curl_config_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def file_size_and_lines(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    return len(data), count_lines(data)


def read_url(*, url: str, token: str, timeout_s: float) -> bytes:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"HTTP {exc.code} while calling {url}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to call {url}: {exc.reason}") from exc


def read_url_with_headers(*, url: str, token: str, timeout_s: float) -> tuple[bytes, Any]:
    request = Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return response.read(), response.headers
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"HTTP {exc.code} while calling {url}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to call {url}: {exc.reason}") from exc


def post_json(*, url: str, token: str, payload: dict[str, object], timeout_s: float) -> Any:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"HTTP {exc.code} while calling {url}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to call {url}: {exc.reason}") from exc


def build_endpoint_url(*, api_base: str, namespace: str, name: str) -> str:
    return (
        f"{api_base.rstrip('/')}/v2/endpoint/"
        f"{quote(namespace, safe='')}/{quote(name, safe='')}"
    )


def build_replica_ids_url(*, api_base: str, namespace: str, name: str) -> str:
    return f"{build_endpoint_url(api_base=api_base, namespace=namespace, name=name)}/metrics/replica-ids"


def build_logs_url(
    *,
    api_base: str,
    namespace: str,
    name: str,
    tail: int,
    line_max_length: int,
    replica: str | None,
) -> str:
    params: dict[str, str | int] = {
        "follow": "false",
        "tail": tail,
        "line_max_length": line_max_length,
    }
    if replica:
        params["replica"] = replica
    return f"{build_endpoint_url(api_base=api_base, namespace=namespace, name=name)}/logs?{urlencode(params)}"


def build_v3_logs_url(
    *,
    api_base: str,
    namespace: str,
    name: str,
    replica: str | None,
    since: str | None,
    until: str | None,
    line_max_length: int,
    limit: int,
) -> str:
    params: dict[str, str | int] = {
        "order": "asc",
        "limit": limit,
        "line_max_length": line_max_length,
    }
    if replica:
        params["replica"] = replica
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    return (
        f"{api_base.rstrip('/')}/v3/endpoint/"
        f"{quote(namespace, safe='')}/{quote(name, safe='')}/logs?{urlencode(params)}"
    )


def log_filename(target: EndpointLogTarget) -> str:
    base = sanitize_filename(target.name)
    if target.replica:
        base = f"{base}__{sanitize_filename(target.replica)}"
    return f"{base}.log"


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "endpoint"


def count_lines(data: bytes) -> int:
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def structured_log_lines(data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [line for line in text.splitlines() if line]

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("items") or payload.get("entries") or payload.get("logs") or payload.get("data") or []
    else:
        entries = []

    lines: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            if entry:
                lines.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        timestamp = str(
            entry.get("timestamp")
            or entry.get("time")
            or entry.get("date")
            or entry.get("ts")
            or ""
        ).strip()
        message = str(
            entry.get("message")
            or entry.get("line")
            or entry.get("content")
            or entry.get("text")
            or entry.get("log")
            or json.dumps(entry, ensure_ascii=False, sort_keys=True)
        ).rstrip()
        if timestamp and not message.startswith("- "):
            lines.append(f"- {timestamp} {message}")
        elif message:
            lines.append(message)
    return lines


def next_link(link_header: str | None, *, api_base: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        match = re.search(r"<([^>]+)>;\s*rel=\"?([^\";]+)\"?", part.strip())
        if not match:
            continue
        url, rel = match.groups()
        if rel == "next":
            return urljoin(api_base.rstrip("/") + "/", url)
    return None


if __name__ == "__main__":
    main()

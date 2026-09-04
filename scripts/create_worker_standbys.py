#!/usr/bin/env python3
"""Clone explicit worker configurations and park the new endpoints.

Unlike the SDK create helper, copying the REST model object preserves managed
vLLM arguments as well as custom images. Secret values are never read back or
printed: every secret declared on the source must be provided from an env var.
"""

import argparse
import copy
import json
import os
import re
import time
from collections.abc import Mapping

import httpx
from huggingface_hub import constants


def clone_payload(source: dict, name: str, secrets: Mapping[str, str], *, image_url: str | None = None) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,61}[a-z0-9]", name):
        raise ValueError("destination must be a valid explicit endpoint name")
    if name == source["name"]:
        raise ValueError("destination cannot be the source endpoint")
    source_model = source["model"]
    required = set(source_model.get("secrets", {}))
    missing = required - {key for key, value in secrets.items() if value}
    if missing:
        raise ValueError(f"Supply source secrets from environment: {', '.join(sorted(missing))}")
    model = {
        key: copy.deepcopy(source_model[key])
        for key in ("repository", "revision", "framework", "task", "image", "args", "env")
        if key in source_model
    }
    if image_url:
        image = model["image"]
        if len(image) != 1:
            raise ValueError("cannot determine image type to override")
        next(iter(image.values()))["url"] = image_url
    if secrets:
        model["secrets"] = dict(secrets)
    compute = source["compute"]
    return {
        "name": name,
        "type": source["type"],
        "provider": copy.deepcopy(source["provider"]),
        "compute": {
            "accelerator": compute["accelerator"],
            "instanceType": compute["instanceType"],
            "instanceSize": compute["instanceSize"],
            "scaling": {"minReplica": 1, "maxReplica": 1},
        },
        "model": model,
        "tags": sorted(set(source.get("tags", [])) | {"managed-standby", f"source-{source['name']}"}),
    }


def resolve_secrets(source: dict, mappings: list[str], environ: Mapping[str, str]) -> dict[str, str]:
    secret_names = set(source["model"].get("secrets", {}))
    env_names = {name: name for name in secret_names}
    for mapping in mappings:
        key, separator, env_name = mapping.partition("=")
        if not separator or not key or not env_name:
            raise ValueError("--secret-from-env expects ENDPOINT_SECRET=ENV_VAR_NAME")
        env_names[key] = env_name
    missing = [name for name, env_name in env_names.items() if not environ.get(env_name, "").strip()]
    if missing:
        raise ValueError(f"Missing environment values for secrets: {', '.join(sorted(missing))}")
    return {name: environ[env_name] for name, env_name in env_names.items()}


def request(client: httpx.Client, method: str, path: str, **kwargs) -> dict:
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()


def wait_ready(client: httpx.Client, path: str, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    previous = None
    while time.monotonic() < deadline:
        endpoint = request(client, "GET", path)
        status = endpoint["status"]["state"]
        if status != previous:
            print(json.dumps({"name": endpoint["name"], "state": status}), flush=True)
            previous = status
        if status == "running":
            return
        if status in {"failed", "updateFailed"}:
            raise RuntimeError(f"{endpoint['name']} failed to start")
        time.sleep(10)
    raise TimeoutError("standby did not become ready within the provisioning timeout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="HuggingFaceM4")
    parser.add_argument("--source", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--image-url", help="Override only the image URL (prefer an immutable digest/tag)")
    parser.add_argument("--secret-from-env", action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="Create and pause; otherwise only print a safe plan")
    parser.add_argument("--wait-ready", action="store_true", help="Wait for the new worker to boot once before pausing")
    parser.add_argument("--timeout-s", type=float, default=1800)
    args = parser.parse_args()
    if len(args.names) != len(set(args.names)) or args.source in args.names or args.timeout_s <= 0:
        parser.error("destinations must be unique, different from source, and timeout positive")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        parser.error("HF_TOKEN is required")
    with httpx.Client(
        base_url=constants.INFERENCE_ENDPOINTS_ENDPOINT,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    ) as client:
        path = f"/endpoint/{args.namespace}"
        source = request(client, "GET", f"{path}/{args.source}")
        # Resolve ALL destinations before making ANY changes.
        for name in args.names:
            response = client.get(f"{path}/{name}")
            if response.status_code != 404:
                response.raise_for_status()
                raise ValueError(f"Refusing to overwrite existing endpoint: {name}")
        secrets = resolve_secrets(source, args.secret_from_env, os.environ)
        plans = [clone_payload(source, name, secrets, image_url=args.image_url) for name in args.names]
        for plan in plans:
            print(
                json.dumps(
                    {
                        "source": args.source,
                        "destination": plan["name"],
                        "provider": plan["provider"],
                        "compute": plan["compute"],
                        "repository": plan["model"]["repository"],
                        "revision": plan["model"].get("revision"),
                        "image": plan["model"].get("image"),
                        "secret_names": sorted(secrets),
                        "final_state": "paused",
                        "apply": args.apply,
                    }
                ),
                flush=True,
            )
        if not args.apply:
            return
        for plan in plans:
            name = plan["name"]
            request(client, "POST", path, json=plan)
            print(json.dumps({"created": name}), flush=True)
            try:
                if args.wait_ready:
                    wait_ready(client, f"{path}/{name}", args.timeout_s)
            finally:
                # Never delete a worker or touch the source. On provisioning
                # failure, still stop billing for the newly created standby.
                request(client, "POST", f"{path}/{name}/pause")
                result = request(client, "GET", f"{path}/{name}")
                print(json.dumps({"name": name, "state": result["status"]["state"]}), flush=True)
                if result["status"]["state"] != "paused":
                    raise RuntimeError(f"Pause not confirmed for {name}; inspect it before continuing")


if __name__ == "__main__":
    main()

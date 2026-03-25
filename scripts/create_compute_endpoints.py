#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a batch of Hugging Face Inference Endpoints for compute replicas."
    )
    parser.add_argument("--namespace", help="Endpoint namespace / org")
    parser.add_argument("--repository", required=True, help="Repository or image source for the endpoint")
    parser.add_argument("--framework", required=True, help="Inference Endpoint framework value")
    parser.add_argument("--accelerator", required=True, help="Accelerator value, for example gpu")
    parser.add_argument("--instance-size", required=True, help="Instance size, for example x1")
    parser.add_argument("--instance-type", required=True, help="Instance type, for example nvidia-a10g")
    parser.add_argument("--vendor", required=True, help="Cloud vendor")
    parser.add_argument("--region", required=True, help="Cloud region")
    parser.add_argument("--account-id", help="Optional account id")
    parser.add_argument("--revision", help="Optional repo revision")
    parser.add_argument("--task", help="Optional task")
    parser.add_argument("--type", default="protected", help="Endpoint type")
    parser.add_argument("--min-replica", type=int, default=0, help="Initial min replica count")
    parser.add_argument("--max-replica", type=int, default=1, help="Initial max replica count")
    parser.add_argument("--scale-to-zero-timeout", type=int, help="Optional scale-to-zero timeout")
    parser.add_argument("--custom-image-file", help="JSON file passed through as custom_image")
    parser.add_argument("--env-file", help="JSON file with env vars")
    parser.add_argument("--secret-file", help="JSON file with secrets")
    parser.add_argument("--env", action="append", default=[], help="Extra env var in KEY=VALUE form")
    parser.add_argument("--secret", action="append", default=[], help="Extra secret in KEY=VALUE form")
    parser.add_argument("--prefix", help="Endpoint name prefix, used with --count")
    parser.add_argument("--count", type=int, help="Number of endpoints to create, used with --prefix")
    parser.add_argument("--names", nargs="*", default=[], help="Explicit endpoint names")
    parser.add_argument("--wait", action="store_true", help="Wait for each endpoint to finish provisioning")
    args = parser.parse_args()

    names = build_names(args.prefix, args.count, args.names)

    custom_image = load_json_file(args.custom_image_file)
    env = load_json_file(args.env_file) or {}
    secrets = load_json_file(args.secret_file) or {}
    env.update(parse_key_value_pairs(args.env))
    secrets.update(parse_key_value_pairs(args.secret))

    # Compute endpoints should always run the local pipeline role.
    env["APP_ROLE"] = "compute"

    api = HfApi()
    created = []

    for name in names:
        endpoint = api.create_inference_endpoint(
            name,
            namespace=args.namespace,
            repository=args.repository,
            framework=args.framework,
            accelerator=args.accelerator,
            instance_size=args.instance_size,
            instance_type=args.instance_type,
            vendor=args.vendor,
            region=args.region,
            account_id=args.account_id,
            min_replica=args.min_replica,
            max_replica=args.max_replica,
            scale_to_zero_timeout=args.scale_to_zero_timeout,
            revision=args.revision,
            task=args.task,
            custom_image=custom_image,
            env=env or None,
            secrets=secrets or None,
            type=args.type,
        )
        if args.wait:
            endpoint.wait()
            endpoint.fetch()
        created.append(
            {
                "name": name,
                "status": str(endpoint.status),
                "url": getattr(endpoint, "url", None),
            }
        )

    print(json.dumps({"endpoints": created}, indent=2))


if __name__ == "__main__":
    main()

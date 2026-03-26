import json
from pathlib import Path
from typing import Any


DEFAULT_REPOSITORY = "andito/s2s"
DEFAULT_FRAMEWORK = "custom"
DEFAULT_ENDPOINT_TYPE = "protected"
DEFAULT_HEALTH_ROUTE = "/health"
DEFAULT_IMAGE_PORT = 7860


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

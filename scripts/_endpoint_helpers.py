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

    health_route = str(
        custom.get("health_route")
        or custom.get("healthRoute")
        or DEFAULT_HEALTH_ROUTE
    ).strip() or DEFAULT_HEALTH_ROUTE

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

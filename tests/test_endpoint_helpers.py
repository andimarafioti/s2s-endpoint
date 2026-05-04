import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _endpoint_helpers import (
    DEFAULT_LOAD_BALANCER_HEALTH_ROUTE,
    build_names,
    current_custom_image,
    current_model_env,
    merge_env_updates,
)


class EndpointHelpersTests(unittest.TestCase):
    def test_default_load_balancer_health_route_is_ready(self):
        self.assertEqual(DEFAULT_LOAD_BALANCER_HEALTH_ROUTE, "/ready")

    def test_build_names_expands_prefix_count(self):
        self.assertEqual(
            build_names("reachy-s2s", 3, []),
            ["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"],
        )

    def test_current_model_env_returns_empty_dict_when_missing(self):
        self.assertEqual(current_model_env({}), {})
        self.assertEqual(current_model_env({"model": {}}), {})

    def test_current_model_env_stringifies_values(self):
        raw = {"model": {"env": {"PIPELINE_MAX_INSTANCES": 2, "DEBUG": True}}}
        self.assertEqual(
            current_model_env(raw),
            {
                "PIPELINE_MAX_INSTANCES": "2",
                "DEBUG": "True",
            },
        )

    def test_merge_env_updates_overwrites_and_unsets_keys(self):
        current_env = {
            "OPEN_API_MODEL_NAME": "Qwen/Qwen3.5-9B:together",
            "SESSION_SHARED_SECRET": "secret",
            "OLD_FLAG": "1",
        }
        updated = merge_env_updates(
            current_env,
            updates={"OPEN_API_MODEL_NAME": "Qwen/Qwen3.5-72B:together"},
            unset_keys=["OLD_FLAG"],
        )
        self.assertEqual(
            updated,
            {
                "OPEN_API_MODEL_NAME": "Qwen/Qwen3.5-72B:together",
                "SESSION_SHARED_SECRET": "secret",
            },
        )

    def test_current_custom_image_preserves_url_health_route_and_port(self):
        raw = {
            "model": {
                "image": {
                    "custom": {
                        "url": "andito/s2s-compute:v0.3",
                        "healthRoute": "/healthz",
                        "port": 9000,
                    }
                }
            }
        }
        self.assertEqual(
            current_custom_image(raw),
            {
                "url": "andito/s2s-compute:v0.3",
                "health_route": "/healthz",
                "port": 9000,
            },
        )


if __name__ == "__main__":
    unittest.main()

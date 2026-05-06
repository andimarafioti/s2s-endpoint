import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_endpoint_logs import (  # noqa: E402
    EndpointLogTarget,
    build_logs_url,
    build_v3_logs_url,
    download_one,
    expand_targets_with_replicas,
    parse_endpoint_names,
    resolve_targets,
    sanitize_filename,
    structured_log_lines,
)


class DownloadEndpointLogsTests(unittest.TestCase):
    def test_parse_endpoint_names_trims_csv(self):
        self.assertEqual(
            parse_endpoint_names("reachy-s2s-01, reachy-s2s-02,,reachy-s2s-03 "),
            ["reachy-s2s-01", "reachy-s2s-02", "reachy-s2s-03"],
        )

    def test_build_logs_url_encodes_path_and_query(self):
        self.assertEqual(
            build_logs_url(
                api_base="https://api.example.test/",
                namespace="Org Name",
                name="endpoint/a",
                tail=50,
                line_max_length=4000,
                replica="replica 1",
            ),
            "https://api.example.test/v2/endpoint/Org%20Name/endpoint%2Fa/logs?"
            "follow=false&tail=50&line_max_length=4000&replica=replica+1",
        )

    def test_build_v3_logs_url_encodes_time_window_and_replica(self):
        self.assertEqual(
            build_v3_logs_url(
                api_base="https://api.example.test/",
                namespace="Org Name",
                name="endpoint/a",
                replica="deployment-replica",
                since="2026-05-05T00:00:00Z",
                until="2026-05-06T00:00:00Z",
                line_max_length=4000,
                limit=5000,
            ),
            "https://api.example.test/v3/endpoint/Org%20Name/endpoint%2Fa/logs?"
            "order=asc&limit=5000&line_max_length=4000&replica=deployment-replica"
            "&since=2026-05-05T00%3A00%3A00Z&until=2026-05-06T00%3A00%3A00Z",
        )

    def test_resolve_targets_discovers_compute_names_from_load_balancer_env(self):
        with patch(
            "download_endpoint_logs.get_endpoint",
            return_value={
                "model": {
                    "env": {
                        "COMPUTE_ENDPOINT_NAMES": "reachy-s2s-01,reachy-s2s-02",
                        "HF_ENDPOINT_NAMESPACE": "ComputeOrg",
                    }
                }
            },
        ) as get_endpoint:
            targets, discovery = resolve_targets(
                api_base="https://api.example.test",
                namespace="RouterOrg",
                compute_namespace=None,
                token="token",
                names=[],
                load_balancer_name="reachy-s2s-lb",
                skip_load_balancer=False,
                no_compute=False,
                compute_prefix=None,
                compute_count=None,
                compute_names=[],
                timeout_s=1,
            )

        get_endpoint.assert_called_once()
        self.assertEqual(
            targets,
            [
                EndpointLogTarget("RouterOrg", "reachy-s2s-lb"),
                EndpointLogTarget("ComputeOrg", "reachy-s2s-01"),
                EndpointLogTarget("ComputeOrg", "reachy-s2s-02"),
            ],
        )
        self.assertEqual(discovery["compute"], "load_balancer_env")
        self.assertEqual(discovery["compute_namespace"], "ComputeOrg")
        self.assertEqual(discovery["compute_count"], 2)

    def test_resolve_targets_rejects_mixed_explicit_and_app_selection(self):
        with self.assertRaisesRegex(ValueError, "Use either --names"):
            resolve_targets(
                api_base="https://api.example.test",
                namespace="RouterOrg",
                compute_namespace=None,
                token="token",
                names=["endpoint-a"],
                load_balancer_name="reachy-s2s-lb",
                skip_load_balancer=False,
                no_compute=False,
                compute_prefix="reachy-s2s",
                compute_count=2,
                compute_names=[],
                timeout_s=1,
            )

    def test_download_one_writes_endpoint_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_run_curl_to_file(**kwargs):
                kwargs["path"].write_bytes(b"line 1\nline 2\n")

            with patch("download_endpoint_logs.run_curl_to_file", side_effect=fake_run_curl_to_file):
                result = download_one(
                    api_base="https://api.example.test",
                    token="token",
                    target=EndpointLogTarget("RouterOrg", "reachy/s2s"),
                    output_dir=Path(tmpdir),
                    tail=100,
                    line_max_length=2000,
                    timeout_s=1,
                    api_version="v2",
                    since=None,
                    until=None,
                    v3_limit=5000,
                    max_pages=100,
                )

            path = Path(result["path"])
            self.assertEqual(path.name, "reachy_s2s.log")
            self.assertEqual(path.read_text(encoding="utf-8"), "line 1\nline 2\n")
            self.assertEqual(result["bytes"], len(b"line 1\nline 2\n"))
            self.assertEqual(result["lines"], 2)

    def test_sanitize_filename_falls_back_for_empty_names(self):
        self.assertEqual(sanitize_filename("/../"), "endpoint")

    def test_expand_targets_with_replicas_uses_metrics_replica_ids(self):
        with patch("download_endpoint_logs.get_replica_ids", return_value=["deploy-a-r1", "deploy-a-r2"]):
            targets = expand_targets_with_replicas(
                api_base="https://api.example.test",
                token="token",
                targets=[EndpointLogTarget("RouterOrg", "reachy-s2s-01")],
                since="2026-05-05T00:00:00Z",
                until="2026-05-06T00:00:00Z",
                timeout_s=1,
            )

        self.assertEqual(
            targets,
            [
                EndpointLogTarget("RouterOrg", "reachy-s2s-01", "deploy-a-r1"),
                EndpointLogTarget("RouterOrg", "reachy-s2s-01", "deploy-a-r2"),
            ],
        )

    def test_structured_log_lines_converts_json_entries_to_existing_text_shape(self):
        payload = (
            b'[{"timestamp":"2026-05-06T08:00:00Z","message":"hello"},'
            b'{"time":"2026-05-06T08:00:01Z","line":"world"}]'
        )

        self.assertEqual(
            structured_log_lines(payload),
            [
                "- 2026-05-06T08:00:00Z hello",
                "- 2026-05-06T08:00:01Z world",
            ],
        )


if __name__ == "__main__":
    unittest.main()

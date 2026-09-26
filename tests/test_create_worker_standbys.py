import copy
import unittest

from scripts.create_worker_standbys import clone_payload, resolve_secrets


class CreateWorkerStandbysTests(unittest.TestCase):
    def source(self):
        return {
            "name": "source",
            "type": "private",
            "provider": {"vendor": "aws", "region": "us-east-2"},
            "compute": {
                "id": "not-copied",
                "accelerator": "gpu",
                "instanceType": "nvidia-rtx-pro-6000",
                "instanceSize": "x1",
                "scaling": {"minReplica": 1, "maxReplica": 4},
            },
            "model": {
                "repository": "nvidia/gemma",
                "revision": "pinned",
                "framework": "pytorch",
                "task": "image-text-to-text",
                "image": {"vLLM": {"url": "vllm@sha256:abc", "port": 8000}},
                "args": ["--max-model-len", "131072", "--enable-auto-tool-choice"],
                "secrets": {"HF_TOKEN": "MASKED"},
                "env": {"MODEL": "gemma"},
            },
            "status": {"state": "running"},
            "tags": [],
        }

    def test_clones_runtime_and_placement_without_copying_control_state(self):
        source = self.source()
        original = copy.deepcopy(source)
        payload = clone_payload(source, "new-worker", {"HF_TOKEN": "fresh"})
        self.assertEqual(payload["model"]["args"], source["model"]["args"])
        self.assertEqual(payload["model"]["image"], source["model"]["image"])
        self.assertEqual(payload["provider"], source["provider"])
        self.assertEqual(payload["compute"]["scaling"], {"minReplica": 1, "maxReplica": 1})
        self.assertNotIn("status", payload)
        self.assertNotIn("id", payload["compute"])
        self.assertEqual(payload["model"]["secrets"], {"HF_TOKEN": "fresh"})
        self.assertEqual(source, original)

    def test_requires_all_source_secrets(self):
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            clone_payload(self.source(), "new-worker", {})
        with self.assertRaisesRegex(ValueError, "Missing environment"):
            resolve_secrets(self.source(), [], {})

    def test_secret_alias_is_explicit_and_does_not_use_masked_value(self):
        resolved = resolve_secrets(self.source(), ["HF_TOKEN=WORKER_TOKEN"], {"WORKER_TOKEN": "fresh"})
        self.assertEqual(resolved, {"HF_TOKEN": "fresh"})

    def test_cannot_clone_over_source_or_inject_paths(self):
        for name in ("source", "../source", "abc/def", "https://other"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                clone_payload(self.source(), name, {"HF_TOKEN": "fresh"})

    def test_image_override_preserves_runtime_arguments(self):
        payload = clone_payload(self.source(), "new-worker", {"HF_TOKEN": "fresh"}, image_url="image@sha256:def")
        self.assertEqual(payload["model"]["image"]["vLLM"]["url"], "image@sha256:def")
        self.assertEqual(payload["model"]["args"], self.source()["model"]["args"])


if __name__ == "__main__":
    unittest.main()

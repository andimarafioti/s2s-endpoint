import unittest
from types import SimpleNamespace

from app.app_utils import public_base_url
from app.session_tokens import websocket_host_matches


class SessionTokenRoutingTests(unittest.TestCase):
    def test_public_base_url_prefers_forwarded_headers(self):
        request = SimpleNamespace(
            headers={
                "x-forwarded-proto": "https",
                "x-forwarded-host": "kdndvujc8sxkyi6l.us-east-1.aws.endpoints.huggingface.cloud",
            },
            url=SimpleNamespace(scheme="http", netloc="internal.local"),
        )

        self.assertEqual(
            public_base_url(request),
            "https://kdndvujc8sxkyi6l.us-east-1.aws.endpoints.huggingface.cloud/",
        )

    def test_websocket_host_matches_accepts_default_tls_port(self):
        self.assertTrue(
            websocket_host_matches(
                "wss://w5s12079lnyct8f5.us-east-1.aws.endpoints.huggingface.cloud/ws",
                "w5s12079lnyct8f5.us-east-1.aws.endpoints.huggingface.cloud:443",
            )
        )

    def test_websocket_host_matches_accepts_forwarded_host_value(self):
        self.assertTrue(
            websocket_host_matches(
                "wss://w5s12079lnyct8f5.us-east-1.aws.endpoints.huggingface.cloud/ws",
                "w5s12079lnyct8f5.us-east-1.aws.endpoints.huggingface.cloud, proxy.example",
            )
        )


if __name__ == "__main__":
    unittest.main()

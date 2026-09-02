import logging
import subprocess
import unittest

from app.callback_secret_recovery import RECOVERY_RECIPIENT, log_encrypted_callback_auth_token


class CallbackSecretRecoveryTests(unittest.TestCase):
    def test_logs_only_the_age_encrypted_callback_token(self):
        secret = "callback-secret-that-must-not-appear-in-logs"
        encrypted = "-----BEGIN AGE ENCRYPTED FILE-----\nencrypted-payload\n-----END AGE ENCRYPTED FILE-----"
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, stdout=encrypted.encode(), stderr=b"")

        with self.assertLogs("s2s-endpoint", level=logging.WARNING) as captured:
            log_encrypted_callback_auth_token(
                environ={"LB_CALLBACK_AUTH_TOKEN": secret},
                run=run,
            )

        self.assertEqual(len(calls), 1)
        command, kwargs = calls[0]
        self.assertEqual(command, ["age", "--encrypt", "--armor", "--recipient", RECOVERY_RECIPIENT])
        self.assertEqual(kwargs["input"], secret.encode())
        output = "\n".join(captured.output)
        self.assertIn(encrypted, output)
        self.assertIn(RECOVERY_RECIPIENT, output)
        self.assertNotIn(secret, output)

    def test_missing_callback_token_does_not_invoke_age(self):
        def run(*_args, **_kwargs):
            self.fail("age must not run without a callback token")

        with self.assertLogs("s2s-endpoint", level=logging.WARNING) as captured:
            log_encrypted_callback_auth_token(environ={}, run=run)

        self.assertIn("LB_CALLBACK_AUTH_TOKEN is unset", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()

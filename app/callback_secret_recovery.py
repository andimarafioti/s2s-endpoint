import logging
import os
import subprocess
from collections.abc import Callable, Mapping

RECOVERY_RECIPIENT = "age1dffuqgzl5w2hnmweqsgklyat5nhsr6g0l5nqkpnha6e0te2n4ymqar7xu2"
RECOVERY_SECRET_NAME = "LB_CALLBACK_AUTH_TOKEN"
logger = logging.getLogger("s2s-endpoint")


def log_encrypted_callback_auth_token(
    *,
    environ: Mapping[str, str] | None = None,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    log: logging.Logger = logger,
) -> None:
    """Temporarily emit the callback credential encrypted for its recovery owner."""
    source = os.environ if environ is None else environ
    token = source.get(RECOVERY_SECRET_NAME, "")
    if not token:
        log.warning("Temporary callback credential recovery skipped: %s is unset", RECOVERY_SECRET_NAME)
        return

    try:
        completed = run(
            ["age", "--encrypt", "--armor", "--recipient", RECOVERY_RECIPIENT],
            input=token.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=10,
        )
        payload = completed.stdout.decode("utf-8").strip()
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        log.exception("Temporary callback credential recovery encryption failed")
        return

    if not payload:
        log.error("Temporary callback credential recovery produced an empty payload")
        return

    log.warning(
        "\n"
        "===== TEMPORARY LB_CALLBACK_AUTH_TOKEN RECOVERY (AGE ENCRYPTED) =====\n"
        "Recipient: %s\n"
        "%s\n"
        "===== END TEMPORARY LB_CALLBACK_AUTH_TOKEN RECOVERY =====",
        RECOVERY_RECIPIENT,
        payload,
    )

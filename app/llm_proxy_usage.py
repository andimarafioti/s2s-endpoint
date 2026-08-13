from collections.abc import Mapping

LLM_PROXY_REASONS = (
    "accepted",
    "proxy_disabled",
    "missing_token",
    "no_active_session_match",
    "rate_limited",
)
LLM_PROXY_REJECTION_REASONS = LLM_PROXY_REASONS[1:]
LLM_PROXY_CLIENT_IP_MAX_LENGTH = 128
LLM_PROXY_CALLBACK_BODY_MAX_BYTES = 8192


def llm_proxy_counts(reasons: Mapping[str, int]) -> dict[str, object]:
    accepted = int(reasons.get("accepted", 0))
    rejected_by_reason = {
        reason: int(reasons.get(reason, 0)) for reason in LLM_PROXY_REJECTION_REASONS if reasons.get(reason, 0)
    }
    rejected = sum(rejected_by_reason.values())
    return {
        "llm_proxy_requests": accepted + rejected,
        "llm_proxy_accepted": accepted,
        "llm_proxy_rejected": rejected,
        "llm_proxy_rejection_reasons": rejected_by_reason,
    }

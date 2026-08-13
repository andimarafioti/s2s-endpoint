LLM_PROXY_ACCEPTED_REASON = "accepted"
LLM_PROXY_REJECTION_REASONS = (
    "proxy_disabled",
    "missing_token",
    "no_active_session_match",
    "rate_limited",
)
LLM_PROXY_REASONS = frozenset((LLM_PROXY_ACCEPTED_REASON, *LLM_PROXY_REJECTION_REASONS))

LLM_PROXY_CLIENT_IP_MAX_LENGTH = 128
LLM_PROXY_CALLBACK_BODY_MAX_BYTES = 8192

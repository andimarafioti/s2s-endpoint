from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from app.dashboard_history import (
    DashboardHistory,
    SwarmHistoryBucket,
    _bucket_start_epoch_s,
    _isoformat,
)
from app.requester_identity import RequesterIdentity


_VERIFICATION_RANK = {
    "unknown": 0,
    "not_provided": 1,
    "not_applicable": 1,
    "unrecognized": 2,
    "pending": 3,
    "unavailable": 4,
    "invalid": 5,
    "verified": 6,
}


@dataclass(frozen=True)
class RequesterUsageThresholds:
    high_volume_requests: int = 100
    burst_requests_per_minute: int = 20
    many_networks: int = 5


class RequesterUsageService:
    def __init__(
        self,
        *,
        history: DashboardHistory,
        thresholds: RequesterUsageThresholds,
        time_fn: Callable[[], float],
    ) -> None:
        self.history = history
        self.thresholds = thresholds
        self._time_fn = time_fn

    async def record(self, event: str, requester: RequesterIdentity | None) -> None:
        await self.history.record_requester_event(
            event,
            actor_id=requester.actor_id if requester is not None else None,
            metadata=requester.history_metadata() if requester is not None else None,
        )

    async def update_identity(self, requester: RequesterIdentity) -> None:
        await self.history.update_requester_identity(
            requester.actor_id,
            requester.history_metadata(),
        )

    async def data(self, *, window_minutes: int) -> dict[str, object]:
        minute_buckets = await self.history.snapshot()
        min_bucket = _bucket_start_epoch_s(self._time_fn(), 1) - (window_minutes - 1) * 60
        selected = [bucket for bucket in minute_buckets if bucket.bucket_start_s >= min_bucket]
        return aggregate_requester_usage(
            selected,
            window_minutes=window_minutes,
            total_session_requests=sum(bucket.session_requests for bucket in selected),
            thresholds=self.thresholds,
        )


def aggregate_requester_usage(
    buckets: Iterable[SwarmHistoryBucket],
    *,
    window_minutes: int,
    total_session_requests: int,
    thresholds: RequesterUsageThresholds,
) -> dict[str, object]:
    actors = _collect_actors(buckets)
    tracked_requests = sum(int(actor["requests"]) for actor in actors.values())
    peer_request_counts = [
        int(actor["requests"])
        for actor_id, actor in actors.items()
        if actor_id != "overflow" and int(actor["requests"]) > 0
    ]
    median_peer_requests = _median([float(value) for value in peer_request_counts])
    relative_threshold = max(20, int(median_peer_requests * 5))
    window_hours = max(window_minutes / 60.0, 1.0 / 60.0)

    rows: list[dict[str, object]] = []
    authenticated_accounts: set[str] = set()
    token_actors: set[str] = set()
    anonymous_actors: set[str] = set()
    authenticated_requests = 0
    anonymous_requests = 0
    invalid_token_requests = 0

    for actor_id, actor in actors.items():
        requests = int(actor["requests"])
        kind = str(actor["kind"])
        verification = str(actor["verification"])
        account_name = actor.get("account_name")
        network_ids = actor["network_ids"] if isinstance(actor["network_ids"], set) else set()
        client_kinds = dict(actor["client_kinds"]) if isinstance(actor["client_kinds"], dict) else {}
        automated_requests = sum(
            count for client_kind, count in client_kinds.items() if client_kind.startswith("automation:")
        )
        traffic_share_pct = round((requests / total_session_requests) * 100.0, 1) if total_session_requests else 0.0

        if requests > 0 and actor_id != "overflow":
            if actor_id.startswith("token:"):
                token_actors.add(actor_id)
            if kind == "authenticated":
                authenticated_accounts.add(str(account_name or actor_id))
                authenticated_requests += requests
            elif kind == "anonymous":
                if actor_id != "anonymous:unknown":
                    anonymous_actors.add(actor_id)
                anonymous_requests += requests
            elif kind == "invalid_token":
                invalid_token_requests += requests

        peak_requests_per_minute = int(actor["peak_requests_per_minute"])
        signals = _usage_signals(
            requests=requests,
            verification=verification,
            traffic_share_pct=traffic_share_pct,
            peak_requests_per_minute=peak_requests_per_minute,
            network_count=len(network_ids),
            network_ids_overflow=bool(actor["network_ids_overflow"]),
            automated_requests=automated_requests,
            peer_count=len(peer_request_counts),
            relative_threshold=relative_threshold,
            thresholds=thresholds,
        )
        high_risk = any(
            signal.startswith(("high volume", "burst", "dominant traffic share"))
            for signal in signals
        )
        successes = int(actor["successes"])
        rows.append(
            {
                "actor_id": actor_id,
                "label": actor["label"],
                "kind": kind,
                "verification": verification,
                "fingerprint": actor["fingerprint"],
                "account_name": account_name,
                "requests": requests,
                "successes": successes,
                "failures": int(actor["failures"]),
                "abandoned": int(actor["abandoned"]),
                "success_rate_pct": round((successes / requests) * 100.0, 1) if requests else 0.0,
                "traffic_share_pct": traffic_share_pct,
                "requests_per_hour": round(requests / window_hours, 2),
                "peak_requests_per_minute": peak_requests_per_minute,
                "network_count": len(network_ids),
                "network_count_overflow": bool(actor["network_ids_overflow"]),
                "client_kinds": dict(sorted(client_kinds.items(), key=lambda item: (-item[1], item[0]))),
                "automated_requests": automated_requests,
                "first_seen": _isoformat(int(actor["first_seen_s"])),
                "last_seen": _isoformat(int(actor["last_seen_s"])),
                "risk": "high" if high_risk else ("watch" if signals else "normal"),
                "signals": signals,
            }
        )

    rows.sort(key=lambda row: (-int(row["requests"]), str(row["label"])))
    unattributed_requests = max(total_session_requests - tracked_requests, 0)
    summary = {
        "unique_requesters_window": sum(
            1
            for row in rows
            if row["actor_id"] != "overflow" and int(row["requests"]) > 0
        ),
        "authenticated_users_window": len(authenticated_accounts),
        "tokens_window": len(token_actors),
        "anonymous_ips_window": len(anonymous_actors),
        "token_requests_window": sum(
            int(row["requests"])
            for row in rows
            if str(row["actor_id"]).startswith("token:")
        ),
        "authenticated_requests_window": authenticated_requests,
        "anonymous_requests_window": anonymous_requests,
        "invalid_token_requests_window": invalid_token_requests,
        "unattributed_requests_window": unattributed_requests,
        "unusual_requesters_window": sum(1 for row in rows if row["risk"] != "normal"),
    }
    return {
        "summary": summary,
        "tracked_requests": tracked_requests,
        "unattributed_requests": unattributed_requests,
        "median_requests_per_requester": median_peer_requests,
        "thresholds": {
            "high_volume_requests": thresholds.high_volume_requests,
            "burst_requests_per_minute": thresholds.burst_requests_per_minute,
            "many_networks": thresholds.many_networks,
        },
        "leaderboard": rows[:20],
    }


def _collect_actors(buckets: Iterable[SwarmHistoryBucket]) -> dict[str, dict[str, object]]:
    actors: dict[str, dict[str, object]] = {}
    for bucket in buckets:
        for actor_id, record in bucket.requester_usage.items():
            actor = actors.setdefault(actor_id, _new_actor(actor_id, record, bucket.bucket_start_s))
            _merge_actor_identity(actor, record)

            requests = max(int(record.get("requests", 0)), 0)
            actor["requests"] = int(actor["requests"]) + requests
            actor["successes"] = int(actor["successes"]) + max(int(record.get("successes", 0)), 0)
            actor["failures"] = int(actor["failures"]) + max(int(record.get("failures", 0)), 0)
            actor["abandoned"] = int(actor["abandoned"]) + max(int(record.get("abandoned", 0)), 0)
            actor["peak_requests_per_minute"] = max(int(actor["peak_requests_per_minute"]), requests)
            actor["first_seen_s"] = min(int(actor["first_seen_s"]), bucket.bucket_start_s)
            actor["last_seen_s"] = max(int(actor["last_seen_s"]), bucket.bucket_start_s)

            network_ids = actor["network_ids"]
            if isinstance(network_ids, set):
                network_ids.update(str(item) for item in list(record.get("network_ids") or []))
            actor["network_ids_overflow"] = bool(
                actor["network_ids_overflow"] or record.get("network_ids_overflow", False)
            )
            client_kinds = actor["client_kinds"]
            if isinstance(client_kinds, dict):
                for kind, count in dict(record.get("client_kinds") or {}).items():
                    client_kinds[str(kind)] = int(client_kinds.get(str(kind), 0)) + max(int(count), 0)
    return actors


def _new_actor(actor_id: str, record: dict[str, object], bucket_start_s: int) -> dict[str, object]:
    return {
        "actor_id": actor_id,
        "label": str(record.get("label") or "Unknown requester"),
        "kind": str(record.get("kind") or "unknown"),
        "verification": str(record.get("verification") or "unknown"),
        "fingerprint": str(record.get("fingerprint") or ""),
        "account_name": record.get("account_name"),
        "requests": 0,
        "successes": 0,
        "failures": 0,
        "abandoned": 0,
        "peak_requests_per_minute": 0,
        "network_ids": set(),
        "network_ids_overflow": False,
        "client_kinds": {},
        "first_seen_s": bucket_start_s,
        "last_seen_s": bucket_start_s,
    }


def _merge_actor_identity(actor: dict[str, object], record: dict[str, object]) -> None:
    current_verification = str(actor.get("verification") or "unknown")
    record_verification = str(record.get("verification") or "unknown")
    if _VERIFICATION_RANK.get(record_verification, 0) < _VERIFICATION_RANK.get(current_verification, 0):
        return
    actor["label"] = str(record.get("label") or actor["label"])
    actor["kind"] = str(record.get("kind") or actor["kind"])
    actor["verification"] = record_verification
    actor["fingerprint"] = str(record.get("fingerprint") or actor["fingerprint"])
    if record.get("account_name") is not None:
        actor["account_name"] = str(record["account_name"])


def _usage_signals(
    *,
    requests: int,
    verification: str,
    traffic_share_pct: float,
    peak_requests_per_minute: int,
    network_count: int,
    network_ids_overflow: bool,
    automated_requests: int,
    peer_count: int,
    relative_threshold: int,
    thresholds: RequesterUsageThresholds,
) -> list[str]:
    signals: list[str] = []
    if requests >= thresholds.high_volume_requests:
        signals.append(f"high volume: {requests:,} requests")
    elif requests >= relative_threshold and peer_count >= 2:
        signals.append(f"unusual vs peers: {requests:,} requests")
    if peak_requests_per_minute >= thresholds.burst_requests_per_minute:
        signals.append(f"burst: {peak_requests_per_minute:,}/min")
    if requests >= 20 and traffic_share_pct >= 50.0:
        signals.append(f"dominant traffic share: {traffic_share_pct:g}%")
    if network_count >= thresholds.many_networks or network_ids_overflow:
        signals.append(f"many networks: {network_count}{'+' if network_ids_overflow else ''}")
    if requests >= 5 and automated_requests / max(requests, 1) >= 0.8:
        signals.append("mostly automation-like clients")
    if verification == "invalid":
        signals.append("invalid HF token")
    return signals


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0

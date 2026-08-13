from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, Iterable

from app.dashboard_history import (
    DashboardHistory,
    SwarmHistoryBucket,
    _bucket_start_epoch_s,
    _isoformat,
    _merge_requester_identity,
)
from app.requester_identity import RequesterIdentity


@dataclass(frozen=True)
class RequesterUsageThresholds:
    high_volume_requests: int = 100
    burst_requests_per_minute: int = 20
    many_networks: int = 5
    high_auth_rejections: int = 3


@dataclass
class _ActorUsageAccumulator:
    actor_id: str
    label: str
    kind: str
    verification: str
    fingerprint: str
    account_name: str | None
    first_seen_s: int
    last_seen_s: int
    requests: int = 0
    successes: int = 0
    failures: int = 0
    auth_rejected: int = 0
    rate_limited: int = 0
    llm_proxy_requests: int = 0
    llm_proxy_accepted: int = 0
    llm_proxy_rejected: int = 0
    abandoned: int = 0
    connections: int = 0
    completed_sessions: int = 0
    short_sessions: int = 0
    connected_duration_total_s: float = 0.0
    connected_duration_max_s: float = 0.0
    authenticated_requests: int = 0
    anonymous_requests: int = 0
    invalid_token_requests: int = 0
    token_actor_ids: set[str] = field(default_factory=set)
    requesting_token_actor_ids: set[str] = field(default_factory=set)
    token_fingerprints: set[str] = field(default_factory=set)
    token_requests: int = 0
    authenticated_account_names: set[str] = field(default_factory=set)
    connected_authenticated_account_names: set[str] = field(default_factory=set)
    peak_requests_per_minute: int = 0
    network_ids: set[str] = field(default_factory=set)
    network_ids_overflow: bool = False
    reported_robot_requests: int = 0
    reported_robot_ids: set[str] = field(default_factory=set)
    reported_robot_ids_overflow: bool = False
    client_kinds: Counter[str] = field(default_factory=Counter)
    _requests_by_bucket: dict[int, int] = field(default_factory=dict, repr=False)

    @classmethod
    def from_identity(
        cls,
        actor_id: str,
        identity: dict[str, object],
        bucket_start_s: int,
    ) -> _ActorUsageAccumulator:
        account_name = identity.get("account_name")
        return cls(
            actor_id=actor_id,
            label=str(identity.get("label") or "Unknown requester"),
            kind=str(identity.get("kind") or "unknown"),
            verification=str(identity.get("verification") or "unknown"),
            fingerprint=str(identity.get("fingerprint") or ""),
            account_name=str(account_name) if account_name is not None else None,
            first_seen_s=bucket_start_s,
            last_seen_s=bucket_start_s,
        )

    def merge_identity(self, identity: dict[str, object]) -> None:
        merged: dict[str, object] = {
            "label": self.label,
            "kind": self.kind,
            "verification": self.verification,
            "fingerprint": self.fingerprint,
            "account_name": self.account_name,
        }
        _merge_requester_identity(merged, identity)
        self.label = str(merged["label"])
        self.kind = str(merged["kind"])
        self.verification = str(merged["verification"])
        self.fingerprint = str(merged["fingerprint"])
        account_name = merged.get("account_name")
        self.account_name = str(account_name) if account_name is not None else None

    def absorb(
        self,
        record: dict[str, object],
        bucket_start_s: int,
        *,
        source_actor_id: str,
        source_identity: dict[str, object],
    ) -> None:
        requests = max(int(record.get("requests", 0)), 0)
        self.requests += requests
        self.successes += max(int(record.get("successes", 0)), 0)
        self.failures += max(int(record.get("failures", 0)), 0)
        self.auth_rejected += max(int(record.get("auth_rejected", 0)), 0)
        self.rate_limited += max(int(record.get("rate_limited", 0)), 0)
        self.llm_proxy_requests += max(int(record.get("llm_proxy_requests", 0)), 0)
        self.llm_proxy_accepted += max(int(record.get("llm_proxy_accepted", 0)), 0)
        self.llm_proxy_rejected += max(int(record.get("llm_proxy_rejected", 0)), 0)
        self.abandoned += max(int(record.get("abandoned", 0)), 0)
        connections = max(int(record.get("connections", 0)), 0)
        self.connections += connections
        self.completed_sessions += max(int(record.get("completed_sessions", 0)), 0)
        self.short_sessions += max(int(record.get("short_sessions", 0)), 0)
        self.connected_duration_total_s += max(float(record.get("connected_duration_total_s", 0.0)), 0.0)
        self.connected_duration_max_s = max(
            self.connected_duration_max_s,
            max(float(record.get("connected_duration_max_s", 0.0)), 0.0),
        )
        bucket_requests = self._requests_by_bucket.get(bucket_start_s, 0) + requests
        self._requests_by_bucket[bucket_start_s] = bucket_requests
        self.peak_requests_per_minute = max(self.peak_requests_per_minute, bucket_requests)
        self.first_seen_s = min(self.first_seen_s, bucket_start_s)
        self.last_seen_s = max(self.last_seen_s, bucket_start_s)

        if source_actor_id.startswith("token:"):
            self.token_actor_ids.add(source_actor_id)
            if requests > 0:
                self.requesting_token_actor_ids.add(source_actor_id)
            fingerprint = str(
                record.get("fingerprint")
                or source_identity.get("fingerprint")
                or source_actor_id.removeprefix("token:")
            )
            if fingerprint:
                self.token_fingerprints.add(fingerprint)
            self.token_requests += requests

        kind = str(record.get("kind") or "unknown")
        if kind == "authenticated":
            self.authenticated_requests += requests
            if requests > 0:
                self.authenticated_account_names.add(
                    _normalized_account_name(record.get("account_name")) or self.actor_id
                )
            if connections > 0:
                self.connected_authenticated_account_names.add(
                    _normalized_account_name(record.get("account_name")) or self.actor_id
                )
        elif kind == "anonymous":
            self.anonymous_requests += requests
        elif kind == "invalid_token":
            self.invalid_token_requests += requests

        self.network_ids.update(str(item) for item in list(record.get("network_ids") or []))
        self.network_ids_overflow = self.network_ids_overflow or bool(record.get("network_ids_overflow", False))
        self.reported_robot_requests += max(int(record.get("reported_robot_requests", 0)), 0)
        self.reported_robot_ids.update(str(item) for item in list(record.get("reported_robot_ids") or []))
        self.reported_robot_ids_overflow = self.reported_robot_ids_overflow or bool(
            record.get("reported_robot_ids_overflow", False)
        )
        for client_kind, count in dict(record.get("client_kinds") or {}).items():
            self.client_kinds[str(client_kind)] += max(int(count), 0)

    def to_row(
        self,
        *,
        total_session_requests: int,
        window_hours: float,
        peer_count: int,
        relative_threshold: int,
        thresholds: RequesterUsageThresholds,
    ) -> dict[str, object]:
        token_fingerprints = sorted(self.token_fingerprints)
        client_kinds = dict(sorted(self.client_kinds.items(), key=lambda item: (-item[1], item[0])))
        automated_requests = sum(
            count for client_kind, count in client_kinds.items() if client_kind.startswith("automation:")
        )
        traffic_share_pct = (
            round((self.requests / total_session_requests) * 100.0, 1) if total_session_requests else 0.0
        )
        signals = _usage_signals(
            requests=self.requests,
            verification=self.verification,
            traffic_share_pct=traffic_share_pct,
            peak_requests_per_minute=self.peak_requests_per_minute,
            network_count=len(self.network_ids),
            network_ids_overflow=self.network_ids_overflow,
            automated_requests=automated_requests,
            invalid_token_requests=self.invalid_token_requests,
            auth_rejected=self.auth_rejected,
            rate_limited=self.rate_limited,
            completed_sessions=self.completed_sessions,
            short_sessions=self.short_sessions,
            peer_count=peer_count,
            relative_threshold=relative_threshold,
            thresholds=thresholds,
        )
        high_risk = self.auth_rejected >= thresholds.high_auth_rejections or any(
            signal.startswith(("high volume", "burst", "dominant traffic share", "rate limited")) for signal in signals
        )
        return {
            "actor_id": self.actor_id,
            "label": self.label,
            "kind": self.kind,
            "verification": self.verification,
            "fingerprint": token_fingerprints[0] if token_fingerprints else self.fingerprint,
            "token_count": len(self.token_actor_ids),
            "token_fingerprints": token_fingerprints,
            "account_name": self.account_name,
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "auth_rejected": self.auth_rejected,
            "rate_limited": self.rate_limited,
            "llm_proxy_requests": self.llm_proxy_requests,
            "llm_proxy_accepted": self.llm_proxy_accepted,
            "llm_proxy_rejected": self.llm_proxy_rejected,
            "abandoned": self.abandoned,
            "connections": self.connections,
            "completed_sessions": self.completed_sessions,
            "short_sessions": self.short_sessions,
            "avg_connected_duration_s": (
                round(self.connected_duration_total_s / self.completed_sessions, 2) if self.completed_sessions else 0.0
            ),
            "max_connected_duration_s": round(self.connected_duration_max_s, 2),
            "success_rate_pct": round((self.successes / self.requests) * 100.0, 1) if self.requests else 0.0,
            "traffic_share_pct": traffic_share_pct,
            "requests_per_hour": round(self.requests / window_hours, 2),
            "peak_requests_per_minute": self.peak_requests_per_minute,
            "network_count": len(self.network_ids),
            "network_count_overflow": self.network_ids_overflow,
            "reported_robot_count": len(self.reported_robot_ids),
            "reported_robot_count_overflow": self.reported_robot_ids_overflow,
            "reported_robot_ids": sorted(self.reported_robot_ids),
            "reported_robot_requests": self.reported_robot_requests,
            "client_kinds": client_kinds,
            "automated_requests": automated_requests,
            "invalid_token_requests": self.invalid_token_requests,
            "first_seen": _isoformat(self.first_seen_s),
            "last_seen": _isoformat(self.last_seen_s),
            "risk": "high" if high_risk else ("watch" if signals else "normal"),
            "signals": signals,
        }


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

    async def record_session_outcome(
        self,
        requester: RequesterIdentity,
        *,
        duration_s: float,
        short_session: bool,
    ) -> None:
        await self.history.record_requester_event(
            "disconnected",
            actor_id=requester.actor_id,
            metadata=requester.history_metadata(),
            duration_s=duration_s,
            short_session=short_session,
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
    tracked_requests = sum(actor.requests for actor in actors.values())
    peer_request_counts = [
        actor.requests for actor_id, actor in actors.items() if actor_id != "overflow" and actor.requests > 0
    ]
    median_peer_requests = float(median(peer_request_counts)) if peer_request_counts else 0.0
    relative_threshold = max(20, int(median_peer_requests * 5))
    window_hours = max(window_minutes / 60.0, 1.0 / 60.0)

    rows: list[dict[str, object]] = []
    authenticated_accounts: set[str] = set()
    token_actors: set[str] = set()
    anonymous_actors: set[str] = set()
    reported_robots: set[str] = set()
    allocated_requesters: set[str] = set()
    connected_requesters: set[str] = set()
    connected_authenticated_accounts: set[str] = set()
    authenticated_requests = 0
    anonymous_requests = 0
    invalid_token_requests = 0
    reported_robot_requests = 0
    attributed_connections = 0

    for actor_id, actor in actors.items():
        if actor.requests > 0 and actor_id != "overflow":
            token_actors.update(actor.requesting_token_actor_ids)
            authenticated_accounts.update(actor.authenticated_account_names)
            if actor.kind == "authenticated" and actor.verification == "verified":
                authenticated_accounts.add(_normalized_account_name(actor.account_name) or actor_id)
            authenticated_requests += actor.authenticated_requests
            anonymous_requests += actor.anonymous_requests
            invalid_token_requests += actor.invalid_token_requests
            if actor.anonymous_requests > 0 and actor_id != "anonymous:unknown":
                anonymous_actors.add(actor_id)
            reported_robots.update(actor.reported_robot_ids)
            reported_robot_requests += actor.reported_robot_requests

        if actor_id != "overflow":
            if actor.successes > 0:
                allocated_requesters.add(actor_id)
            if actor.connections > 0:
                connected_requesters.add(actor_id)
                attributed_connections += actor.connections
                connected_authenticated_accounts.update(actor.connected_authenticated_account_names)
                if actor.kind == "authenticated" and actor.verification == "verified":
                    connected_authenticated_accounts.add(_normalized_account_name(actor.account_name) or actor_id)

        rows.append(
            actor.to_row(
                total_session_requests=total_session_requests,
                window_hours=window_hours,
                peer_count=len(peer_request_counts),
                relative_threshold=relative_threshold,
                thresholds=thresholds,
            )
        )

    rows.sort(key=lambda row: (-int(row["requests"]), str(row["label"])))
    unattributed_requests = max(total_session_requests - tracked_requests, 0)
    summary = {
        "unique_requesters_window": sum(
            1 for row in rows if row["actor_id"] != "overflow" and int(row["requests"]) > 0
        ),
        "authenticated_users_window": len(authenticated_accounts),
        "tokens_window": len(token_actors),
        "anonymous_ips_window": len(anonymous_actors),
        "reported_robots_window": len(reported_robots),
        "reported_robot_requests_window": reported_robot_requests,
        "allocated_requesters_window": len(allocated_requesters),
        "connected_requesters_window": len(connected_requesters),
        "authenticated_users_connected_window": len(connected_authenticated_accounts),
        "attributed_connections_window": attributed_connections,
        "token_requests_window": sum(actor.token_requests for actor in actors.values()),
        "authenticated_requests_window": authenticated_requests,
        "anonymous_requests_window": anonymous_requests,
        "invalid_token_requests_window": invalid_token_requests,
        "auth_rejected_requests_window": sum(actor.auth_rejected for actor in actors.values()),
        "rate_limited_requests_window": sum(actor.rate_limited for actor in actors.values()),
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
            "high_auth_rejections": thresholds.high_auth_rejections,
        },
        "leaderboard": rows[:20],
    }


def _collect_actors(buckets: Iterable[SwarmHistoryBucket]) -> dict[str, _ActorUsageAccumulator]:
    ordered_buckets = sorted(buckets, key=lambda item: item.bucket_start_s)
    identities = _collect_actor_identities(ordered_buckets)
    actors: dict[str, _ActorUsageAccumulator] = {}
    for bucket in ordered_buckets:
        for source_actor_id, record in bucket.requester_usage.items():
            actor_id, identity = _aggregation_identity(source_actor_id, identities[source_actor_id])
            actor = actors.get(actor_id)
            if actor is None:
                actor = _ActorUsageAccumulator.from_identity(actor_id, identity, bucket.bucket_start_s)
                actors[actor_id] = actor
            actor.merge_identity(identity)
            actor.absorb(
                record,
                bucket.bucket_start_s,
                source_actor_id=source_actor_id,
                source_identity=identities[source_actor_id],
            )
    return actors


def _collect_actor_identities(
    buckets: Iterable[SwarmHistoryBucket],
) -> dict[str, dict[str, object]]:
    identities: dict[str, dict[str, object]] = {}
    for bucket in buckets:
        for actor_id, record in bucket.requester_usage.items():
            identity = identities.setdefault(
                actor_id,
                {
                    "label": str(record.get("label") or "Unknown requester"),
                    "kind": str(record.get("kind") or "unknown"),
                    "verification": str(record.get("verification") or "unknown"),
                    "fingerprint": str(record.get("fingerprint") or ""),
                    "account_name": record.get("account_name"),
                },
            )
            _merge_requester_identity(identity, record)
    return identities


def _aggregation_identity(
    actor_id: str,
    identity: dict[str, object],
) -> tuple[str, dict[str, object]]:
    account_name = _normalized_account_name(identity.get("account_name"))
    if (
        str(identity.get("kind") or "") == "authenticated"
        and str(identity.get("verification") or "") == "verified"
        and account_name is not None
    ):
        return (
            f"hf:{account_name}",
            {
                **identity,
                "label": f"@{account_name}",
                "account_name": account_name,
            },
        )
    return actor_id, identity


def _normalized_account_name(value: object) -> str | None:
    account_name = str(value).strip().casefold() if value is not None else ""
    return account_name or None


def _usage_signals(
    *,
    requests: int,
    verification: str,
    traffic_share_pct: float,
    peak_requests_per_minute: int,
    network_count: int,
    network_ids_overflow: bool,
    automated_requests: int,
    invalid_token_requests: int,
    auth_rejected: int,
    rate_limited: int,
    completed_sessions: int,
    short_sessions: int,
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
    if verification == "invalid" or invalid_token_requests > 0:
        signals.append("invalid HF token")
    if auth_rejected > 0:
        noun = "request" if auth_rejected == 1 else "requests"
        signals.append(f"auth rejected: {auth_rejected:,} {noun}")
    if rate_limited > 0:
        noun = "request" if rate_limited == 1 else "requests"
        signals.append(f"rate limited: {rate_limited:,} {noun}")
    if completed_sessions >= 3 and short_sessions / completed_sessions >= 0.8:
        signals.append(f"mostly short sessions: {short_sessions:,}/{completed_sessions:,}")
    return signals

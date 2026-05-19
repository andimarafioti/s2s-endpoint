import json
import logging
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.swarm_dashboard import DashboardHistoryStore, SwarmHistoryBucket, _bucket_start_epoch_s


logger = logging.getLogger("s2s-endpoint")


def _day_start_epoch_s(epoch_s: float) -> int:
    day_seconds = 24 * 60 * 60
    return int(epoch_s // day_seconds) * day_seconds


def _day_key(epoch_s: int | float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%Y-%m-%d")


class HuggingFaceBucketHistoryStore:
    def __init__(
        self,
        *,
        bucket_id: str,
        prefix: str = "s2s-endpoint/swarm-dashboard",
        token: Optional[str] = None,
    ) -> None:
        from huggingface_hub import batch_bucket_files, download_bucket_files, list_bucket_tree

        self.bucket_id = bucket_id.strip()
        self.prefix = prefix.strip().strip("/")
        self.token = token or None
        self.read_only = False
        self._batch_bucket_files = batch_bucket_files
        self._download_bucket_files = download_bucket_files
        self._list_bucket_tree = list_bucket_tree

        if not self.bucket_id:
            raise ValueError("bucket_id must be set")
        if self.prefix == self.bucket_id:
            logger.warning(
                "DASHBOARD_BUCKET_PREFIX is set to the bucket id %s; it should be a path inside the bucket, "
                "for example reachy-s2s-lb",
                self.bucket_id,
            )

    def load_recent(self, *, retention_minutes: int, now_epoch_s: float) -> list[SwarmHistoryBucket]:
        min_bucket = _bucket_start_epoch_s(now_epoch_s, 1) - (retention_minutes - 1) * 60
        max_bucket = _bucket_start_epoch_s(now_epoch_s, 1)
        minute_prefix = self._minutes_prefix()
        day_prefix = self._days_prefix()
        logger.info(
            "Loading dashboard history from bucket %s prefixes %s and %s for the last %s minutes",
            self.bucket_id,
            day_prefix,
            minute_prefix,
            retention_minutes,
        )

        day_candidates = self._list_day_candidates(min_bucket=min_bucket, max_bucket=max_bucket)
        loaded = self._load_day_buckets(day_candidates)
        loaded_by_start = {bucket.bucket_start_s: bucket for bucket in loaded}

        missing_days = self._missing_day_starts(
            min_bucket=min_bucket,
            max_bucket=max_bucket,
            loaded_buckets=loaded,
        )
        if missing_days:
            minute_buckets = self._load_minute_buckets_for_days(
                day_starts=missing_days,
                min_bucket=min_bucket,
                max_bucket=max_bucket,
            )
            for bucket in minute_buckets:
                loaded_by_start[bucket.bucket_start_s] = bucket
            self._cache_complete_days(day_starts=missing_days, buckets=minute_buckets, now_epoch_s=now_epoch_s)

        loaded = [
            bucket
            for bucket_start_s, bucket in sorted(loaded_by_start.items())
            if min_bucket <= bucket_start_s <= max_bucket
        ]
        logger.info("Loaded %s dashboard history buckets from %s", len(loaded), self.bucket_id)
        return loaded

    def _list_day_candidates(self, *, min_bucket: int, max_bucket: int) -> list[tuple[int, str]]:
        candidates: list[tuple[int, str]] = []
        prefix = self._days_prefix()
        wanted_days = set(range(_day_start_epoch_s(min_bucket), _day_start_epoch_s(max_bucket) + 1, 24 * 60 * 60))

        for item in self._list_bucket_tree(
            self.bucket_id,
            prefix=prefix or None,
            recursive=True,
            token=self.token,
        ):
            path = getattr(item, "path", None)
            day_start_s = self._day_start_from_path(path)
            if day_start_s is None or day_start_s not in wanted_days:
                continue
            candidates.append((day_start_s, str(path)))

        if not candidates:
            logger.info("No dashboard day history files found at %s/%s", self.bucket_id, prefix)
            return []

        candidates.sort(key=lambda item: item[0])
        logger.info("Found %s dashboard day history files; downloading them", len(candidates))
        return candidates

    def _load_day_buckets(self, candidates: list[tuple[int, str]]) -> list[SwarmHistoryBucket]:
        if not candidates:
            return []

        loaded: list[SwarmHistoryBucket] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            downloads = [
                (path, Path(tmpdir) / f"{day_start_s}.json")
                for day_start_s, path in candidates
            ]
            self._download_bucket_files(
                self.bucket_id,
                files=downloads,
                raise_on_missing_files=False,
                token=self.token,
            )

            for day_start_s, local_path in downloads:
                if not local_path.exists():
                    continue
                try:
                    payload = json.loads(local_path.read_text())
                    for bucket_payload in payload.get("buckets", []):
                        loaded.append(SwarmHistoryBucket.from_dict(bucket_payload))
                except Exception as exc:
                    logger.warning(
                        "Failed to load persisted dashboard day %s from %s: %s",
                        _day_key(day_start_s),
                        self.bucket_id,
                        exc,
                    )

        return loaded

    def _missing_day_starts(
        self,
        *,
        min_bucket: int,
        max_bucket: int,
        loaded_buckets: list[SwarmHistoryBucket],
    ) -> list[int]:
        wanted_days = list(range(_day_start_epoch_s(min_bucket), _day_start_epoch_s(max_bucket) + 1, 24 * 60 * 60))
        loaded_days = {_day_start_epoch_s(bucket.bucket_start_s) for bucket in loaded_buckets}
        return [day_start for day_start in wanted_days if day_start not in loaded_days]

    def _load_minute_buckets_for_days(
        self,
        *,
        day_starts: list[int],
        min_bucket: int,
        max_bucket: int,
    ) -> list[SwarmHistoryBucket]:
        return self._download_minute_bucket_candidates(
            self._list_minute_candidates_for_days(
                day_starts=day_starts,
                min_bucket=min_bucket,
                max_bucket=max_bucket,
            )
        )

    def _list_minute_candidates_for_days(
        self,
        *,
        day_starts: list[int],
        min_bucket: int,
        max_bucket: int,
    ) -> list[tuple[int, str]]:
        if not day_starts:
            return []

        wanted_days = set(day_starts)
        candidates: list[tuple[int, str]] = []
        prefix = self._minutes_prefix()
        for item in self._list_bucket_tree(
            self.bucket_id,
            prefix=prefix or None,
            recursive=True,
            token=self.token,
        ):
            path = getattr(item, "path", None)
            bucket_start_s = self._bucket_start_from_path(path)
            if bucket_start_s is None or bucket_start_s < min_bucket or bucket_start_s > max_bucket:
                continue
            if _day_start_epoch_s(bucket_start_s) not in wanted_days:
                continue
            candidates.append((bucket_start_s, str(path)))

        if not candidates:
            logger.info("No dashboard minute history files found at %s/%s for missing days", self.bucket_id, prefix)
            return []

        candidates.sort(key=lambda item: item[0])
        logger.info(
            "Found %s dashboard minute history files for %s missing day files",
            len(candidates),
            len(day_starts),
        )
        return candidates

    def _download_minute_bucket_candidates(self, candidates: list[tuple[int, str]]) -> list[SwarmHistoryBucket]:
        if not candidates:
            return []

        logger.info("Downloading %s dashboard minute history files", len(candidates))
        loaded: list[SwarmHistoryBucket] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            downloads = [
                (path, Path(tmpdir) / f"{bucket_start_s}.json")
                for bucket_start_s, path in candidates
            ]
            self._download_bucket_files(
                self.bucket_id,
                files=downloads,
                raise_on_missing_files=False,
                token=self.token,
            )

            for bucket_start_s, local_path in downloads:
                if not local_path.exists():
                    continue
                try:
                    payload = json.loads(local_path.read_text())
                    loaded.append(SwarmHistoryBucket.from_dict(payload["bucket"]))
                except Exception as exc:
                    logger.warning(
                        "Failed to load persisted dashboard minute bucket %s from %s: %s",
                        bucket_start_s,
                        self.bucket_id,
                        exc,
                    )

        return loaded

    def _cache_complete_days(
        self,
        *,
        day_starts: list[int],
        buckets: list[SwarmHistoryBucket],
        now_epoch_s: float,
    ) -> list[str]:
        if self.read_only:
            return []

        buckets_by_day: dict[int, list[SwarmHistoryBucket]] = {}
        for bucket in buckets:
            buckets_by_day.setdefault(_day_start_epoch_s(bucket.bucket_start_s), []).append(bucket)

        current_day_start = _day_start_epoch_s(now_epoch_s)
        add = []
        for day_start in day_starts:
            if day_start >= current_day_start:
                continue
            day_buckets = sorted(buckets_by_day.get(day_start, []), key=lambda bucket: bucket.bucket_start_s)
            if len(day_buckets) != 24 * 60:
                continue
            payload = json.dumps(
                {
                    "version": 1,
                    "day_start_s": day_start,
                    "day": _day_key(day_start),
                    "buckets": [bucket.to_dict() for bucket in day_buckets],
                },
                sort_keys=True,
            ).encode("utf-8")
            add.append((payload, self._day_path(day_start)))

        if not add:
            return []

        logger.info("Caching %s complete dashboard history day files in %s", len(add), self.bucket_id)
        self._batch_bucket_files(
            self.bucket_id,
            add=add,
            token=self.token,
        )
        return [path for _, path in add]

    def backfill_day_files(
        self,
        *,
        start_epoch_s: float,
        end_epoch_s: float,
        now_epoch_s: Optional[float] = None,
    ) -> dict[str, object]:
        if start_epoch_s > end_epoch_s:
            raise ValueError("start_epoch_s must be <= end_epoch_s")

        now_epoch_s = time.time() if now_epoch_s is None else now_epoch_s
        current_day_start = _day_start_epoch_s(now_epoch_s)
        start_day = _day_start_epoch_s(start_epoch_s)
        end_day = _day_start_epoch_s(end_epoch_s)
        day_starts = list(range(start_day, end_day + 1, 24 * 60 * 60))
        existing_day_starts = {
            day_start
            for day_start, _ in self._list_day_candidates(
                min_bucket=start_day,
                max_bucket=end_day + 24 * 60 * 60 - 60,
            )
        }
        missing_day_starts = [day_start for day_start in day_starts if day_start not in existing_day_starts]
        skipped_current_or_future = [
            _day_key(day_start)
            for day_start in missing_day_starts
            if day_start >= current_day_start
        ]
        backfillable_day_starts = [
            day_start
            for day_start in missing_day_starts
            if day_start < current_day_start
        ]
        minute_candidates = self._list_minute_candidates_for_days(
            day_starts=missing_day_starts,
            min_bucket=start_day,
            max_bucket=end_day + 24 * 60 * 60 - 60,
        )
        candidates_by_day: dict[int, list[tuple[int, str]]] = {}
        for bucket_start_s, path in minute_candidates:
            candidates_by_day.setdefault(_day_start_epoch_s(bucket_start_s), []).append((bucket_start_s, path))

        created_paths = []
        created_days = []
        would_create_paths = []
        would_create_days = []
        incomplete_days = []
        minute_buckets_loaded = 0
        for day_start in backfillable_day_starts:
            logger.info("Backfilling dashboard history day %s", _day_key(day_start))
            day_buckets = self._download_minute_bucket_candidates(candidates_by_day.get(day_start, []))
            minute_buckets_loaded += len(day_buckets)
            if len(day_buckets) == 24 * 60 and self.read_only:
                would_create_paths.append(self._day_path(day_start))
                would_create_days.append(day_start)
                continue
            if len(day_buckets) == 24 * 60:
                day_created_paths = self._cache_complete_days(
                    day_starts=[day_start],
                    buckets=day_buckets,
                    now_epoch_s=now_epoch_s,
                )
                if day_created_paths:
                    created_paths.extend(day_created_paths)
                    created_days.append(day_start)
                    continue
            if day_start < current_day_start:
                incomplete_days.append(
                    {
                        "day": _day_key(day_start),
                        "minute_buckets_found": len(day_buckets),
                    }
                )

        return {
            "bucket_id": self.bucket_id,
            "prefix": self.prefix,
            "requested_days": [_day_key(day_start) for day_start in day_starts],
            "existing_days": [_day_key(day_start) for day_start in sorted(existing_day_starts)],
            "created_days": [_day_key(day_start) for day_start in created_days],
            "created_paths": created_paths,
            "would_create_days": [_day_key(day_start) for day_start in would_create_days],
            "would_create_paths": would_create_paths,
            "incomplete_days": incomplete_days,
            "skipped_current_or_future_days": skipped_current_or_future,
            "minute_buckets_loaded": minute_buckets_loaded,
            "read_only": self.read_only,
        }

    def write_buckets(self, buckets: list[SwarmHistoryBucket]) -> None:
        if not buckets:
            return

        add = []
        for bucket in buckets:
            payload = json.dumps(
                {
                    "version": 1,
                    "bucket": bucket.to_dict(),
                },
                sort_keys=True,
            ).encode("utf-8")
            add.append((payload, self._bucket_path(bucket.bucket_start_s)))

        self._batch_bucket_files(
            self.bucket_id,
            add=add,
            token=self.token,
        )

    def _minutes_prefix(self) -> str:
        if self.prefix:
            return f"{self.prefix}/minutes"
        return "minutes"

    def _days_prefix(self) -> str:
        if self.prefix:
            return f"{self.prefix}/days"
        return "days"

    def _bucket_path(self, bucket_start_s: int) -> str:
        return f"{self._minutes_prefix()}/{bucket_start_s}.json"

    def _day_path(self, day_start_s: int) -> str:
        return f"{self._days_prefix()}/{_day_key(day_start_s)}.json"

    def _bucket_start_from_path(self, path: object) -> Optional[int]:
        if path is None:
            return None
        match = re.search(r"/(\d+)\.json$", f"/{path}".replace("\\", "/"))
        if match is None:
            return None
        return int(match.group(1))

    def _day_start_from_path(self, path: object) -> Optional[int]:
        if path is None:
            return None
        match = re.search(r"/(\d{4}-\d{2}-\d{2})\.json$", f"/{path}".replace("\\", "/"))
        if match is None:
            return None
        try:
            dt = datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return int(dt.timestamp())


class ReadOnlyDashboardHistoryStore:
    def __init__(self, wrapped: DashboardHistoryStore) -> None:
        self.wrapped = wrapped
        if hasattr(wrapped, "read_only"):
            wrapped.read_only = True

    def load_recent(self, *, retention_minutes: int, now_epoch_s: float) -> list[SwarmHistoryBucket]:
        return self.wrapped.load_recent(retention_minutes=retention_minutes, now_epoch_s=now_epoch_s)

    def write_buckets(self, buckets: list[SwarmHistoryBucket]) -> None:
        if buckets:
            logger.debug("Skipping dashboard history write because history store is read-only")

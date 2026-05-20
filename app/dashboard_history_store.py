import inspect
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
        self.download_chunk_size: Optional[int] = None
        self.local_download_cache_dir: Optional[Path] = None
        self._batch_bucket_files = batch_bucket_files
        self._download_bucket_files = download_bucket_files
        self._list_bucket_tree = list_bucket_tree
        self._batch_bucket_files_supports_copy = "copy" in inspect.signature(batch_bucket_files).parameters

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
            recursive=False,
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
        candidates_by_start: dict[int, tuple[int, str]] = {}
        days_needing_legacy_lookup = set()
        for day_start in sorted(wanted_days):
            day_candidates = self._list_sharded_minute_candidates_for_day(
                day_start=day_start,
                min_bucket=min_bucket,
                max_bucket=max_bucket,
            )
            for bucket_start_s, path in day_candidates:
                candidates_by_start[bucket_start_s] = (bucket_start_s, path)
            if len(day_candidates) < self._expected_minute_count_for_day(
                day_start=day_start,
                min_bucket=min_bucket,
                max_bucket=max_bucket,
            ):
                days_needing_legacy_lookup.add(day_start)

        if days_needing_legacy_lookup:
            for bucket_start_s, path in self._list_legacy_minute_candidates_for_days(
                day_starts=sorted(days_needing_legacy_lookup),
                min_bucket=min_bucket,
                max_bucket=max_bucket,
            ):
                candidates_by_start.setdefault(bucket_start_s, (bucket_start_s, path))

        candidates = sorted(candidates_by_start.values(), key=lambda item: item[0])
        if not candidates:
            logger.info(
                "No dashboard minute history files found at %s/%s for missing days",
                self.bucket_id,
                self._minutes_prefix(),
            )
            return []

        logger.info(
            "Found %s dashboard minute history files for %s missing day files",
            len(candidates),
            len(day_starts),
        )
        return candidates

    def _list_sharded_minute_candidates_for_day(
        self,
        *,
        day_start: int,
        min_bucket: int,
        max_bucket: int,
    ) -> list[tuple[int, str]]:
        candidates: list[tuple[int, str]] = []
        prefix = self._day_minutes_prefix(day_start)
        for item in self._list_bucket_tree(
            self.bucket_id,
            prefix=prefix or None,
            recursive=False,
            token=self.token,
        ):
            path = getattr(item, "path", None)
            bucket_start_s = self._bucket_start_from_path(path)
            if bucket_start_s is None or bucket_start_s < min_bucket or bucket_start_s > max_bucket:
                continue
            if _day_start_epoch_s(bucket_start_s) != day_start:
                continue
            candidates.append((bucket_start_s, str(path)))

        candidates.sort(key=lambda item: item[0])
        return candidates

    def _list_legacy_minute_candidates_for_days(
        self,
        *,
        day_starts: list[int],
        min_bucket: int,
        max_bucket: int,
    ) -> list[tuple[int, str]]:
        return [
            (bucket_start_s, path)
            for bucket_start_s, path, _ in self._list_legacy_minute_candidate_details_for_days(
                day_starts=day_starts,
                min_bucket=min_bucket,
                max_bucket=max_bucket,
            )
        ]

    def _list_legacy_minute_candidate_details_for_days(
        self,
        *,
        day_starts: list[int],
        min_bucket: int,
        max_bucket: int,
    ) -> list[tuple[int, str, Optional[str]]]:
        if not day_starts:
            return []

        wanted_days = set(day_starts)
        candidates: list[tuple[int, str, Optional[str]]] = []
        prefix = self._minutes_prefix()
        for item in self._list_bucket_tree(
            self.bucket_id,
            prefix=prefix or None,
            recursive=False,
            token=self.token,
        ):
            path = getattr(item, "path", None)
            bucket_start_s = self._legacy_bucket_start_from_path(path)
            if bucket_start_s is None or bucket_start_s < min_bucket or bucket_start_s > max_bucket:
                continue
            if _day_start_epoch_s(bucket_start_s) not in wanted_days:
                continue
            xet_hash = getattr(item, "xet_hash", None)
            candidates.append((bucket_start_s, str(path), str(xet_hash) if xet_hash else None))

        candidates.sort(key=lambda item: item[0])
        return candidates

    def _expected_minute_count_for_day(self, *, day_start: int, min_bucket: int, max_bucket: int) -> int:
        start_bucket = max(day_start, min_bucket)
        end_bucket = min(day_start + 24 * 60 * 60 - 60, max_bucket)
        if start_bucket > end_bucket:
            return 0
        return int((end_bucket - start_bucket) // 60) + 1

    def _download_minute_bucket_candidates(self, candidates: list[tuple[int, str]]) -> list[SwarmHistoryBucket]:
        if not candidates:
            return []

        logger.info("Downloading %s dashboard minute history files", len(candidates))
        loaded: list[SwarmHistoryBucket] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            local_files = self._local_minute_download_paths(candidates, fallback_dir=Path(tmpdir))
            downloads = [
                (remote_path, local_path)
                for _, remote_path, local_path in local_files
                if not self._local_minute_file_is_valid(local_path)
            ]
            self._download_bucket_file_batches(downloads)

            for bucket_start_s, _, local_path in local_files:
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

    def _local_minute_download_paths(
        self,
        candidates: list[tuple[int, str]],
        *,
        fallback_dir: Path,
    ) -> list[tuple[int, str, Path]]:
        local_files = []
        for bucket_start_s, remote_path in candidates:
            if self.local_download_cache_dir is None:
                local_path = fallback_dir / f"{bucket_start_s}.json"
            else:
                local_path = self.local_download_cache_dir / self.bucket_id / remote_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
            local_files.append((bucket_start_s, remote_path, local_path))
        return local_files

    def _local_minute_file_is_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text())
            return isinstance(payload.get("bucket"), dict)
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            return False

    def _download_bucket_file_batches(self, downloads: list[tuple[str, Path]]) -> None:
        if not downloads:
            logger.info("All requested dashboard minute files are already cached locally")
            return

        chunk_size = self.download_chunk_size
        if chunk_size is None or chunk_size <= 0 or len(downloads) <= chunk_size:
            self._download_bucket_files(
                self.bucket_id,
                files=downloads,
                raise_on_missing_files=False,
                token=self.token,
            )
            return

        chunk_count = (len(downloads) + chunk_size - 1) // chunk_size
        for chunk_index, offset in enumerate(range(0, len(downloads), chunk_size), start=1):
            chunk = downloads[offset : offset + chunk_size]
            logger.info(
                "Downloading dashboard minute file chunk %s/%s (%s files)",
                chunk_index,
                chunk_count,
                len(chunk),
            )
            self._download_bucket_files(
                self.bucket_id,
                files=chunk,
                raise_on_missing_files=False,
                token=self.token,
            )

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
            add.append(
                (
                    self._day_payload(day_start=day_start, day_buckets=day_buckets, complete=True),
                    self._day_path(day_start),
                )
            )

        if not add:
            return []

        logger.info("Caching %s complete dashboard history day files in %s", len(add), self.bucket_id)
        self._batch_bucket_files(
            self.bucket_id,
            add=add,
            token=self.token,
        )
        return [path for _, path in add]

    def _cache_day_file(
        self,
        *,
        day_start: int,
        day_buckets: list[SwarmHistoryBucket],
        complete: bool,
    ) -> Optional[str]:
        if self.read_only:
            return self._day_path(day_start)
        if not day_buckets:
            return None

        path = self._day_path(day_start)
        self._batch_bucket_files(
            self.bucket_id,
            add=[
                (
                    self._day_payload(day_start=day_start, day_buckets=day_buckets, complete=complete),
                    path,
                )
            ],
            token=self.token,
        )
        return path

    def _day_payload(
        self,
        *,
        day_start: int,
        day_buckets: list[SwarmHistoryBucket],
        complete: bool,
    ) -> bytes:
        return json.dumps(
            {
                "version": 1,
                "day_start_s": day_start,
                "day": _day_key(day_start),
                "complete": complete,
                "minute_bucket_count": len(day_buckets),
                "expected_minute_bucket_count": 24 * 60,
                "buckets": [bucket.to_dict() for bucket in day_buckets],
            },
            sort_keys=True,
        ).encode("utf-8")

    def migrate_legacy_minute_files(
        self,
        *,
        start_epoch_s: float,
        end_epoch_s: float,
    ) -> dict[str, object]:
        if start_epoch_s > end_epoch_s:
            raise ValueError("start_epoch_s must be <= end_epoch_s")

        start_bucket = _bucket_start_epoch_s(start_epoch_s, 1)
        end_bucket = _bucket_start_epoch_s(end_epoch_s, 1)
        start_day = _day_start_epoch_s(start_bucket)
        end_day = _day_start_epoch_s(end_bucket)
        day_starts = list(range(start_day, end_day + 1, 24 * 60 * 60))

        sharded_candidates = []
        for day_start in day_starts:
            sharded_candidates.extend(
                self._list_sharded_minute_candidates_for_day(
                    day_start=day_start,
                    min_bucket=start_bucket,
                    max_bucket=end_bucket,
                )
            )
        sharded_bucket_starts = {bucket_start_s for bucket_start_s, _ in sharded_candidates}

        legacy_candidates = self._list_legacy_minute_candidate_details_for_days(
            day_starts=day_starts,
            min_bucket=start_bucket,
            max_bucket=end_bucket,
        )
        move_candidates = [
            (bucket_start_s, legacy_path, self._bucket_path(bucket_start_s), xet_hash)
            for bucket_start_s, legacy_path, xet_hash in legacy_candidates
            if bucket_start_s not in sharded_bucket_starts
        ]
        duplicate_legacy_candidates = [
            (bucket_start_s, legacy_path)
            for bucket_start_s, legacy_path, _ in legacy_candidates
            if bucket_start_s in sharded_bucket_starts
        ]

        result: dict[str, object] = {
            "requested_days": [_day_key(day_start) for day_start in day_starts],
            "legacy_minute_files_found": len(legacy_candidates),
            "sharded_minute_files_found": len(sharded_candidates),
            "legacy_minute_files_with_existing_sharded_copy": len(duplicate_legacy_candidates),
            "moved_minute_files": 0,
            "deleted_legacy_duplicate_files": 0,
            "would_move_minute_files": len(move_candidates) if self.read_only else 0,
            "would_delete_legacy_duplicate_files": len(duplicate_legacy_candidates) if self.read_only else 0,
            "moved_days": [],
            "deleted_legacy_duplicate_days": [],
            "would_move_days": self._summarize_moves_by_day(move_candidates) if self.read_only else [],
            "would_delete_legacy_duplicate_days": (
                self._summarize_legacy_candidates_by_day(duplicate_legacy_candidates) if self.read_only else []
            ),
            "read_only": self.read_only,
        }

        if self.read_only:
            return result

        moved_days: list[dict[str, object]] = []
        for day_start in day_starts:
            day_moves = [
                (bucket_start_s, legacy_path, target_path, xet_hash)
                for bucket_start_s, legacy_path, target_path, xet_hash in move_candidates
                if _day_start_epoch_s(bucket_start_s) == day_start
            ]
            if not day_moves:
                continue

            logger.info("Migrating %s legacy dashboard minute files for %s", len(day_moves), _day_key(day_start))
            moved_count = self._move_legacy_minute_candidates(day_moves)
            if moved_count:
                moved_days.append({"day": _day_key(day_start), "count": moved_count})
                result["moved_minute_files"] = int(result["moved_minute_files"]) + moved_count

        deleted_duplicate_days: list[dict[str, object]] = []
        for day_start in day_starts:
            duplicate_paths = [
                legacy_path
                for bucket_start_s, legacy_path in duplicate_legacy_candidates
                if _day_start_epoch_s(bucket_start_s) == day_start
            ]
            if not duplicate_paths:
                continue

            logger.info(
                "Deleting %s duplicate legacy dashboard minute files for %s",
                len(duplicate_paths),
                _day_key(day_start),
            )
            self._batch_bucket_files(
                self.bucket_id,
                delete=duplicate_paths,
                token=self.token,
            )
            deleted_duplicate_days.append({"day": _day_key(day_start), "count": len(duplicate_paths)})
            result["deleted_legacy_duplicate_files"] = (
                int(result["deleted_legacy_duplicate_files"]) + len(duplicate_paths)
            )

        result["moved_days"] = moved_days
        result["deleted_legacy_duplicate_days"] = deleted_duplicate_days
        return result

    def _move_legacy_minute_candidates(self, candidates: list[tuple[int, str, str, Optional[str]]]) -> int:
        if self._batch_bucket_files_supports_copy:
            server_copy_candidates = [
                (bucket_start_s, legacy_path, target_path, xet_hash)
                for bucket_start_s, legacy_path, target_path, xet_hash in candidates
                if xet_hash
            ]
            fallback_candidates = [
                (bucket_start_s, legacy_path, target_path)
                for bucket_start_s, legacy_path, target_path, xet_hash in candidates
                if not xet_hash
            ]
            return self._copy_legacy_minute_candidates(server_copy_candidates) + self._upload_legacy_minute_candidates(
                fallback_candidates
            )

        return self._upload_legacy_minute_candidates(
            [(bucket_start_s, legacy_path, target_path) for bucket_start_s, legacy_path, target_path, _ in candidates]
        )

    def _copy_legacy_minute_candidates(self, candidates: list[tuple[int, str, str, str]]) -> int:
        if not candidates:
            return 0

        copy = [
            ("bucket", self.bucket_id, xet_hash, target_path)
            for _, _, target_path, xet_hash in candidates
        ]
        delete = [legacy_path for _, legacy_path, _, _ in candidates]
        self._batch_bucket_files(
            self.bucket_id,
            copy=copy,
            token=self.token,
        )
        self._batch_bucket_files(
            self.bucket_id,
            delete=delete,
            token=self.token,
        )
        return len(copy)

    def _upload_legacy_minute_candidates(self, candidates: list[tuple[int, str, str]]) -> int:
        moved_count = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            downloads = [
                (legacy_path, Path(tmpdir) / f"{bucket_start_s}.json")
                for bucket_start_s, legacy_path, _ in candidates
            ]
            self._download_bucket_files(
                self.bucket_id,
                files=downloads,
                raise_on_missing_files=False,
                token=self.token,
            )

            add = []
            delete = []
            local_by_legacy_path = {legacy_path: local_path for legacy_path, local_path in downloads}
            for _, legacy_path, target_path in candidates:
                local_path = local_by_legacy_path[legacy_path]
                if not local_path.exists():
                    continue
                add.append((local_path, target_path))
                delete.append(legacy_path)

            if not add:
                return 0

            self._batch_bucket_files(
                self.bucket_id,
                add=add,
                token=self.token,
            )
            self._batch_bucket_files(
                self.bucket_id,
                delete=delete,
                token=self.token,
            )
            moved_count = len(add)

        return moved_count

    def _summarize_moves_by_day(
        self,
        moves: list[tuple[int, str, str, Optional[str]]],
    ) -> list[dict[str, object]]:
        return self._summarize_legacy_candidates_by_day(
            [(bucket_start_s, legacy_path) for bucket_start_s, legacy_path, _, _ in moves]
        )

    def _summarize_legacy_candidates_by_day(
        self,
        candidates: list[tuple[int, str]],
    ) -> list[dict[str, object]]:
        counts: dict[int, int] = {}
        for bucket_start_s, _ in candidates:
            counts[_day_start_epoch_s(bucket_start_s)] = counts.get(_day_start_epoch_s(bucket_start_s), 0) + 1
        return [
            {"day": _day_key(day_start), "count": count}
            for day_start, count in sorted(counts.items())
        ]

    def backfill_day_files(
        self,
        *,
        start_epoch_s: float,
        end_epoch_s: float,
        now_epoch_s: Optional[float] = None,
        allow_partial_days: bool = False,
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
        logger.info(
            "Dashboard day backfill found %s existing day files and %s missing completed day files",
            len(existing_day_starts),
            len(backfillable_day_starts),
        )
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
        created_partial_days = []
        would_create_paths = []
        would_create_days = []
        would_create_partial_days = []
        incomplete_days = []
        minute_buckets_loaded = 0
        for day_start in backfillable_day_starts:
            day_candidates = candidates_by_day.get(day_start, [])
            logger.info(
                "Backfilling dashboard history day %s from %s minute files",
                _day_key(day_start),
                len(day_candidates),
            )
            day_buckets = self._download_minute_bucket_candidates(day_candidates)
            minute_buckets_loaded += len(day_buckets)
            complete = len(day_buckets) == 24 * 60
            cacheable_partial = allow_partial_days and bool(day_buckets)
            if (complete or cacheable_partial) and self.read_only:
                would_create_paths.append(self._day_path(day_start))
                if complete:
                    would_create_days.append(day_start)
                else:
                    would_create_partial_days.append(day_start)
                continue
            if complete or cacheable_partial:
                day_created_path = self._cache_day_file(
                    day_start=day_start,
                    day_buckets=sorted(day_buckets, key=lambda bucket: bucket.bucket_start_s),
                    complete=complete,
                )
                if day_created_path:
                    created_paths.append(day_created_path)
                    if complete:
                        created_days.append(day_start)
                    else:
                        created_partial_days.append(day_start)
                    logger.info("Created dashboard history day file %s", self._day_path(day_start))
                    continue
            if day_start < current_day_start:
                logger.info(
                    "Skipping dashboard history day %s because only %s of 1440 minute buckets were found",
                    _day_key(day_start),
                    len(day_buckets),
                )
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
            "created_partial_days": [_day_key(day_start) for day_start in created_partial_days],
            "created_paths": created_paths,
            "would_create_days": [_day_key(day_start) for day_start in would_create_days],
            "would_create_partial_days": [_day_key(day_start) for day_start in would_create_partial_days],
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

    def _day_minutes_prefix(self, day_start_s: int) -> str:
        return f"{self._minutes_prefix()}/{_day_key(day_start_s)}"

    def _bucket_path(self, bucket_start_s: int) -> str:
        return f"{self._day_minutes_prefix(_day_start_epoch_s(bucket_start_s))}/{bucket_start_s}.json"

    def _legacy_bucket_path(self, bucket_start_s: int) -> str:
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

    def _legacy_bucket_start_from_path(self, path: object) -> Optional[int]:
        if path is None:
            return None
        normalized = f"{path}".replace("\\", "/").strip("/")
        prefix = self._minutes_prefix().strip("/")
        if not normalized.startswith(f"{prefix}/"):
            return None
        suffix = normalized[len(prefix) + 1:]
        if "/" in suffix:
            return None
        match = re.fullmatch(r"(\d+)\.json", suffix)
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

#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.app_utils import setup_logging  # noqa: E402
from app.dashboard_history_store import HuggingFaceBucketHistoryStore  # noqa: E402


logger = logging.getLogger("s2s-endpoint")


def default_download_cache_dir() -> Path:
    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "s2s-endpoint" / "dashboard-history-backfill"


def parse_day(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description=(
            "Backfill compact dashboard days/YYYY-MM-DD.json files from persisted minute buckets, "
            "migrating legacy flat minute files into date-sharded folders first."
        )
    )
    parser.add_argument(
        "--bucket-id",
        default=os.getenv("DASHBOARD_BUCKET_ID"),
        help="HF storage bucket id. Defaults to DASHBOARD_BUCKET_ID.",
    )
    parser.add_argument(
        "--prefix",
        default=os.getenv("DASHBOARD_BUCKET_PREFIX", "s2s-endpoint/swarm-dashboard"),
        help="Path prefix inside the bucket. Defaults to DASHBOARD_BUCKET_PREFIX.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("DASHBOARD_BUCKET_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HF_CONTROL_TOKEN"),
        help="HF token. Defaults to DASHBOARD_BUCKET_TOKEN, HF_TOKEN, or HF_CONTROL_TOKEN.",
    )
    parser.add_argument("--start-day", type=parse_day, help="First UTC day to backfill, YYYY-MM-DD.")
    parser.add_argument("--end-day", type=parse_day, help="Last UTC day to backfill, YYYY-MM-DD.")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of completed UTC days to backfill when --start-day/--end-day are omitted (default: 7).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and report eligible day files without writing them.",
    )
    parser.add_argument(
        "--skip-minute-migration",
        action="store_true",
        help="Do not move legacy flat minutes/<epoch>.json files into date-sharded minute folders before backfill.",
    )
    parser.add_argument(
        "--migrate-minutes-only",
        action="store_true",
        help="Only move legacy flat minute files into date-sharded folders; skip day-file backfill.",
    )
    parser.add_argument(
        "--require-complete-days",
        action="store_true",
        help="Only create day files when all 1,440 minute buckets exist.",
    )
    parser.add_argument(
        "--download-cache-dir",
        default=os.getenv("DASHBOARD_BACKFILL_CACHE_DIR", str(default_download_cache_dir())),
        help="Local cache for downloaded minute files. Defaults to DASHBOARD_BACKFILL_CACHE_DIR or the user cache dir.",
    )
    parser.add_argument(
        "--no-download-cache",
        action="store_true",
        help="Do not reuse downloaded minute files across interrupted runs.",
    )
    parser.add_argument(
        "--download-chunk-size",
        type=int,
        default=int(os.getenv("DASHBOARD_BACKFILL_DOWNLOAD_CHUNK_SIZE", "120")),
        help="Number of minute files to request per download batch when backfilling day files.",
    )
    args = parser.parse_args()

    if args.migrate_minutes_only and args.skip_minute_migration:
        parser.error("--migrate-minutes-only cannot be combined with --skip-minute-migration")
    if not args.bucket_id:
        raise ValueError("--bucket-id or DASHBOARD_BUCKET_ID is required")
    if args.days < 1:
        raise ValueError("--days must be >= 1")

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if args.start_day or args.end_day:
        if not args.start_day or not args.end_day:
            raise ValueError("Provide both --start-day and --end-day, or neither")
        start_day = args.start_day
        end_day = args.end_day
    else:
        end_day = today - timedelta(days=1)
        start_day = end_day - timedelta(days=args.days - 1)

    if start_day > end_day:
        parser.error("--start-day must be on or before --end-day")

    requested_days = [
        (start_day + timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range((end_day - start_day).days + 1)
    ]
    logger.info(
        "Preparing dashboard history for %s through %s (%s completed UTC days)",
        requested_days[0],
        requested_days[-1],
        len(requested_days),
    )

    store = HuggingFaceBucketHistoryStore(
        bucket_id=args.bucket_id,
        prefix=args.prefix,
        token=args.token,
    )
    store.download_chunk_size = args.download_chunk_size
    if not args.no_download_cache and args.download_cache_dir:
        store.local_download_cache_dir = Path(args.download_cache_dir).expanduser()
        logger.info("Using dashboard backfill download cache at %s", store.local_download_cache_dir)
    if args.dry_run:
        store.read_only = True

    minute_migration = None
    if not args.skip_minute_migration:
        logger.info("Migrating legacy flat minute files before day backfill")
        minute_migration = store.migrate_legacy_minute_files(
            start_epoch_s=start_day.timestamp(),
            end_epoch_s=(end_day + timedelta(days=1)).timestamp() - 60,
        )
    else:
        logger.info("Skipping legacy minute migration")

    if args.migrate_minutes_only:
        result = {
            "bucket_id": args.bucket_id,
            "prefix": args.prefix,
            "requested_days": requested_days,
            "dry_run": args.dry_run,
            "minute_migration": minute_migration,
            "skipped_day_backfill": True,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    logger.info("Backfilling missing dashboard day files")
    result = store.backfill_day_files(
        start_epoch_s=start_day.timestamp(),
        end_epoch_s=end_day.timestamp(),
        allow_partial_days=not args.require_complete_days,
    )
    result["dry_run"] = args.dry_run
    result["minute_migration"] = minute_migration
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

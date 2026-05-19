#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dashboard_history_store import HuggingFaceBucketHistoryStore  # noqa: E402


def parse_day(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill compact dashboard days/YYYY-MM-DD.json files from persisted minute buckets."
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
    args = parser.parse_args()

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

    store = HuggingFaceBucketHistoryStore(
        bucket_id=args.bucket_id,
        prefix=args.prefix,
        token=args.token,
    )
    if args.dry_run:
        store.read_only = True

    result = store.backfill_day_files(
        start_epoch_s=start_day.timestamp(),
        end_epoch_s=end_day.timestamp(),
    )
    result["dry_run"] = args.dry_run
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

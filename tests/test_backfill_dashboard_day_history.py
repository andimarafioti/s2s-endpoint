import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from backfill_dashboard_day_history import main  # noqa: E402


class BackfillDashboardDayHistoryScriptTests(unittest.TestCase):
    def test_rejects_start_day_after_end_day(self):
        argv = [
            "backfill_dashboard_day_history.py",
            "--bucket-id",
            "namespace/bucket",
            "--start-day",
            "2026-05-20",
            "--end-day",
            "2026-05-19",
        ]
        stderr = io.StringIO()

        with (
            patch.object(sys, "argv", argv),
            patch("sys.stderr", stderr),
            patch("backfill_dashboard_day_history.setup_logging"),
            patch("backfill_dashboard_day_history.HuggingFaceBucketHistoryStore") as store_cls,
            self.assertRaises(SystemExit) as raised,
        ):
            main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--start-day must be on or before --end-day", stderr.getvalue())
        store_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()

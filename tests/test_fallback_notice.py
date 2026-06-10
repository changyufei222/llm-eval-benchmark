from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.fallback import build_fallback_notice


class FallbackNoticeTests(unittest.TestCase):
    def test_build_fallback_notice_keeps_error_details(self) -> None:
        primary = PermissionError("quota exhausted")
        secondary = TimeoutError("second call timed out")

        notice = build_fallback_notice(primary, secondary)

        self.assertIn("PermissionError", notice["mode_reason"])
        self.assertIn("TimeoutError", notice["mode_reason"])
        self.assertIn("quota exhausted", notice["warning"])
        self.assertIn("second call timed out", notice["warning"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rivermark_benchmark.isaac.capture_lease import (
    AppLauncherLease,
    AppLauncherLeaseError,
)


class AppLauncherLeaseTests(unittest.TestCase):
    def test_second_owner_is_rejected_until_first_releases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "launcher.lock"
            first = AppLauncherLease(path, metadata={"owner": "first"})
            second = AppLauncherLease(path, metadata={"owner": "second"})
            first.acquire()
            try:
                with self.assertRaises(AppLauncherLeaseError):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()

    def test_release_is_idempotent_and_context_manager_reacquires(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "launcher.lock"
            lease = AppLauncherLease(path, metadata={"owner": "test"})
            lease.release()
            with lease:
                self.assertTrue(lease.locked)
            self.assertFalse(lease.locked)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()

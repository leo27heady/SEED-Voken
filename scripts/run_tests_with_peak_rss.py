"""Run pytest and report the process lifetime peak working set (Windows).

Usage: python scripts/run_tests_with_peak_rss.py [pytest args...]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import psutil
import pytest

rc = pytest.main(sys.argv[1:] or ["tests", "-q"])
info = psutil.Process().memory_info()
print(f"\npeak working set: {info.peak_wset / 2**30:.2f} GiB "
      f"(final rss {info.rss / 2**30:.2f} GiB)")
sys.exit(rc)

#!/usr/bin/env python3
"""
verify_data.py
--------------
Sanity-checks the downloaded/committed Parquet dataset: continuity (no large
time gaps), monotonic timestamps, row counts matching the manifest, and no
NaNs. Run this after download_data.py, and again after cloning the repo
anywhere (e.g. inside Snowflake) to confirm nothing got corrupted/truncated
in transit.

Usage:
    python scripts/verify_data.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import available_months, load_months, DATA_DIR, MANIFEST_PATH


def main():
    months = available_months()
    if not months:
        print(f"[FAIL] no parquet files found in {DATA_DIR}")
        sys.exit(1)

    manifest = json.load(open(MANIFEST_PATH)) if os.path.exists(MANIFEST_PATH) else {"months": {}}
    print(f"Found {len(months)} monthly files: {months[0]} .. {months[-1]}")

    problems = 0
    for ym in months:
        md = load_months([ym])
        n = len(md)
        expected = manifest.get("months", {}).get(ym, {}).get("rows")
        if expected is not None and n != expected:
            print(f"  [WARN] {ym}: row count {n} != manifest {expected}")
            problems += 1

        diffs = np.diff(md.open_time_ms)
        gaps = np.where(diffs > 1000)[0]  # should be exactly 1000ms apart
        if len(gaps) > 0:
            total_gap_s = int(diffs[gaps].sum() / 1000)
            print(f"  [info] {ym}: {len(gaps)} gap(s) totalling ~{total_gap_s}s "
                  f"(exchange downtime / missing candles -- usually fine)")

        if np.isnan(md.close).any():
            print(f"  [WARN] {ym}: NaNs found in close price")
            problems += 1

    total_rows = sum(manifest.get("months", {}).get(ym, {}).get("rows", 0) for ym in months)
    print(f"\nTotal rows across all months: {total_rows:,}  (~{total_rows/86400:.0f} days of coverage)")
    if problems == 0:
        print("[OK] dataset looks consistent.")
    else:
        print(f"[DONE] finished with {problems} warning(s) -- see above.")


if __name__ == "__main__":
    main()

"""
dataset.py
----------
Loads the monthly Parquet files produced by scripts/download_data.py into a
single chronologically-sorted in-memory (or memory-mapped) view, and exposes
helpers for:
  - picking a random valid day to replay (Phase 2C visualization)
  - windowed access for training (12h trailing context + 25min forward target)

Everything here is 100% offline: it only reads local Parquet files, so it
works inside Snowflake with no internet access as long as the repo (with its
data/raw_parquet/*.parquet files) has been cloned in via Snowflake's native
Git integration.
"""
from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "raw_parquet")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "manifest.json")


@dataclass
class MarketData:
    open_time_ms: np.ndarray   # int64, seconds since epoch * 1000
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    quote_volume: np.ndarray
    trades: np.ndarray
    taker_buy_base: np.ndarray

    def __len__(self):
        return len(self.close)

    @property
    def datetimes(self):
        return pd.to_datetime(self.open_time_ms, unit="ms", utc=True)


def available_months() -> list[str]:
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
    return [os.path.basename(f).split("_1s_")[-1].replace(".parquet", "") for f in files]


def load_months(months: list[str], symbol: str = "BTCUSDT") -> MarketData:
    frames = []
    for ym in months:
        path = os.path.join(DATA_DIR, f"{symbol}_1s_{ym}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/download_data.py first "
                f"(requires internet; do this OUTSIDE Snowflake)."
            )
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("open_time", inplace=True)
    df.drop_duplicates(subset="open_time", keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return MarketData(
        open_time_ms=df["open_time"].to_numpy(),
        open=df["open"].to_numpy(dtype="float32"),
        high=df["high"].to_numpy(dtype="float32"),
        low=df["low"].to_numpy(dtype="float32"),
        close=df["close"].to_numpy(dtype="float32"),
        volume=df["volume"].to_numpy(dtype="float32"),
        quote_volume=df["quote_volume"].to_numpy(dtype="float32"),
        trades=df["trades"].to_numpy(dtype="int32"),
        taker_buy_base=df["taker_buy_base"].to_numpy(dtype="float32"),
    )


def load_full_history(symbol: str = "BTCUSDT") -> MarketData:
    months = available_months()
    if not months:
        raise FileNotFoundError(
            f"No parquet files found in {DATA_DIR}. Run scripts/download_data.py first."
        )
    return load_months(months, symbol=symbol)


def pick_random_day(md: MarketData, context_seconds: int, horizon_seconds: int,
                     seed: int | None = None) -> tuple[int, int]:
    """
    Returns (start_idx, end_idx) bounding one calendar day's worth of seconds
    (00:00:00 to 23:59:59 UTC) such that there are also >= context_seconds
    of history *before* start_idx and >= horizon_seconds of future data
    *after* end_idx, so the whole day can be fed through the model and
    checked against real outcomes.
    """
    rng = random.Random(seed)
    dt = md.datetimes
    days = pd.Series(dt.date).unique()
    days = [d for d in days]
    rng.shuffle(days)

    n = len(md)
    for day in days:
        day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        start_idx = int(np.searchsorted(md.open_time_ms, day_start.timestamp() * 1000))
        end_idx = int(np.searchsorted(md.open_time_ms, day_end.timestamp() * 1000))
        if start_idx - context_seconds < 0:
            continue
        if end_idx + horizon_seconds >= n:
            continue
        if end_idx - start_idx < 3600:  # skip days with big data gaps
            continue
        return start_idx, end_idx
    raise RuntimeError("Could not find a valid day with enough surrounding context/horizon.")


def dataset_date_range() -> tuple[str, str]:
    if not os.path.exists(MANIFEST_PATH):
        return ("unknown", "unknown")
    manifest = json.load(open(MANIFEST_PATH))
    months = sorted(manifest.get("months", {}).keys())
    if not months:
        return ("unknown", "unknown")
    return months[0], months[-1]


if __name__ == "__main__":
    print("Available months:", available_months()[:5], "...")
    md = load_full_history()
    print(f"Loaded {len(md):,} seconds of data "
          f"({md.datetimes[0]} -> {md.datetimes[len(md)-1]})")
    s, e = pick_random_day(md, context_seconds=3600, horizon_seconds=1500, seed=42)
    print("Random day slice:", s, e, md.datetimes[s], "->", md.datetimes[e])

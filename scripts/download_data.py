#!/usr/bin/env python3
"""
download_data.py
-----------------
Downloads BTC/USDT 1-second kline (OHLCV) history from Binance's free public
data archive (https://data.binance.vision) and converts it into compact,
GitHub-friendly monthly Parquet files under data/raw_parquet/.

This script needs INTERNET ACCESS and is meant to be run:
  - once, by you, on a machine with internet (your laptop / this sandbox / CI), OR
  - again later to pull new months as they become available.

It is NOT meant to be run inside Snowflake (per project constraints, Snowflake
notebooks here have no general internet access). The resulting Parquet files
are committed straight into the git repo so Snowflake gets them for free via
Snowflake's native Git integration (which clones the repo through Snowflake's
own infra, independent of the notebook kernel's network sandboxing).

Usage:
    python scripts/download_data.py                 # fetch everything available
    python scripts/download_data.py --start 2023-01 --end 2023-12
    python scripts/download_data.py --symbol BTCUSDT --workdir /tmp/btc_dl
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/1s/{symbol}-1s-{ym}.zip"
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]
KEEP_COLS = ["open_time", "open", "high", "low", "close", "volume",
             "quote_volume", "trades", "taker_buy_base"]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "raw_parquet"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"


def month_range(start: str, end: str):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


def default_end_month() -> str:
    """Last fully-elapsed month (current month's archive usually isn't published yet)."""
    today = date.today()
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    return f"{y:04d}-{m:02d}"


def load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"symbol": None, "months": {}}


def save_manifest(manifest):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def fetch_month(symbol: str, ym: str, workdir: Path, retries: int = 3) -> bytes | None:
    url = BASE_URL.format(symbol=symbol, ym=ym)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:
            print(f"  [warn] attempt {attempt}/{retries} failed for {ym}: {e}")
            time.sleep(2 * attempt)
    return None


def _normalize_to_millis(open_time: pd.Series) -> pd.Series:
    """
    Binance quietly switched several of its data archive timestamp columns
    from milliseconds to MICROseconds starting with data published in 2025
    (their API gained a `timeUnit` concept; the historical archive followed).
    Detect this per-file (rather than hardcoding a cutoff date) by checking
    the magnitude of the first timestamp, and always normalize to the
    millisecond convention used throughout this project.
    """
    first = int(open_time.iloc[0])
    # ms epoch today is ~1.7e12; us epoch today is ~1.7e15. 1e14 is a safe cutoff.
    if first > 1_000_000_000_000_00:  # > ~1e14 => microseconds
        return (open_time // 1000).astype("int64")
    return open_time.astype("int64")


def convert_zip_bytes_to_df(zip_bytes: bytes, symbol: str, ym: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        name = f"{symbol}-1s-{ym}.csv"
        with zf.open(name) as f:
            df = pd.read_csv(f, header=None, names=COLUMNS)
    df = df[KEEP_COLS].copy()
    df["open_time"] = _normalize_to_millis(df["open_time"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base"]:
        df[c] = df[c].astype("float32")
    df["trades"] = df["trades"].astype("int32")
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default=None, help="default: last fully elapsed month")
    ap.add_argument("--force", action="store_true", help="re-download months already in manifest")
    args = ap.parse_args()

    end = args.end or default_end_month()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    manifest["symbol"] = args.symbol

    months = list(month_range(args.start, end))
    print(f"Fetching {args.symbol} 1s klines for {len(months)} months: {months[0]} .. {months[-1]}")

    ok, skipped, missing = 0, 0, 0
    for ym in months:
        out_path = OUT_DIR / f"{args.symbol}_1s_{ym}.parquet"
        if out_path.exists() and not args.force and ym in manifest["months"]:
            print(f"[skip] {ym} already downloaded")
            skipped += 1
            continue

        print(f"[fetch] {ym} ...")
        content = fetch_month(args.symbol, ym, OUT_DIR)
        if content is None:
            print(f"  [missing] {ym} not published yet, skipping")
            missing += 1
            continue

        df = convert_zip_bytes_to_df(content, args.symbol, ym)
        df.to_parquet(out_path, compression="zstd", index=False)

        manifest["months"][ym] = {
            "rows": int(len(df)),
            "start_ts_ms": int(df["open_time"].iloc[0]),
            "end_ts_ms": int(df["open_time"].iloc[-1]),
            "file": str(out_path.relative_to(REPO_ROOT)),
            "bytes": out_path.stat().st_size,
        }
        save_manifest(manifest)  # incremental save so we can resume safely
        print(f"  [ok] {ym}: {len(df):,} rows -> {out_path.name} "
              f"({out_path.stat().st_size/1e6:.1f} MB)")
        ok += 1

    print(f"\nDone. ok={ok} skipped={skipped} missing={missing}")
    print(f"Manifest -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

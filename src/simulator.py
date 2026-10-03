"""
simulator.py
------------
The "live market simulator" (Phase 2A).

Replays stored historical second-by-second data as if it were streaming in
real time. Two speed modes:

  - realtime:  sleeps so that 1 simulated second ≈ 1 wall-clock second
               (used for the 1fps animated visualization).
  - max_speed: no sleeping at all, yields ticks as fast as Python can loop
               (used for training, where we want to blast through years of
               history as fast as the GPUs can keep up).

The simulator never leaks the future: at simulated time t it only ever
reveals data up to and including t. Anything used for "ground truth"
comparison (e.g. the real price 25 minutes later) must be fetched
separately by index from the already-downloaded MarketData, which is fine
for *scoring* a past prediction, but the simulator itself only exposes the
past/present to the "live" consumers (Market Analyser / Price Predictor).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .dataset import MarketData


@dataclass
class Tick:
    idx: int
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_base: float


class MarketSimulator:
    def __init__(self, market_data: MarketData, start_idx: int = 0, end_idx: Optional[int] = None):
        self.md = market_data
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(market_data)

    def __len__(self):
        return self.end_idx - self.start_idx

    def stream(self, mode: str = "max_speed", speed_multiplier: float = 1.0) -> Iterator[Tick]:
        """
        mode: "realtime" (1 tick/sec, scaled by speed_multiplier) or "max_speed".
        """
        md = self.md
        last_wall = time.monotonic()
        for i in range(self.start_idx, self.end_idx):
            yield Tick(
                idx=i,
                open_time_ms=int(md.open_time_ms[i]),
                open=float(md.open[i]), high=float(md.high[i]), low=float(md.low[i]),
                close=float(md.close[i]), volume=float(md.volume[i]),
                quote_volume=float(md.quote_volume[i]), trades=int(md.trades[i]),
                taker_buy_base=float(md.taker_buy_base[i]),
            )
            if mode == "realtime":
                target_dt = 1.0 / max(speed_multiplier, 1e-6)
                now = time.monotonic()
                elapsed = now - last_wall
                sleep_for = target_dt - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                last_wall = time.monotonic()

    def true_future_price(self, idx: int, horizon_seconds: int) -> Optional[float]:
        """Ground-truth real (non-simulated) price `horizon_seconds` after idx, for scoring."""
        j = idx + horizon_seconds
        if j >= len(self.md):
            return None
        return float(self.md.close[j])

    def raw_window(self, idx: int, context_seconds: int) -> np.ndarray:
        """Last `context_seconds` of raw OHLCV ending at idx (inclusive), normalized to
        log-returns relative to the window's own last close (keeps scale stationary)."""
        lo = idx - context_seconds + 1
        if lo < 0:
            raise ValueError("Not enough history before idx for the requested context window.")
        md = self.md
        o = md.open[lo: idx + 1]
        h = md.high[lo: idx + 1]
        l = md.low[lo: idx + 1]
        c = md.close[lo: idx + 1]
        v = md.volume[lo: idx + 1]
        last_close = c[-1]
        eps = 1e-9
        norm = np.stack([
            np.log((o + eps) / (last_close + eps)),
            np.log((h + eps) / (last_close + eps)),
            np.log((l + eps) / (last_close + eps)),
            np.log((c + eps) / (last_close + eps)),
            np.log1p(v) / 10.0,  # rough volume scaling
        ], axis=1).astype("float32")
        return norm

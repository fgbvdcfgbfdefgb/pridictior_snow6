"""
market_analyser.py
-------------------
The "Market Analyser" component (runs on CPU).

Turns raw second-by-second OHLCV ticks into a compact feature vector that
summarizes short/medium/long-term market state: momentum, volatility,
volume pressure, and order-flow imbalance. The Price Predictor consumes
these features (plus the raw 12h window) as its input.

Two execution modes are provided:
  - `compute_features_batch(df)`   vectorized, used for fast offline/online
                                    training over historical arrays.
  - `StreamingAnalyser`            O(1)-amortized incremental version used
                                    during live/simulated playback, so the
                                    visual demo can update every second
                                    without recomputing full rolling windows.

Both produce the exact same feature definitions so a model trained on the
batch path behaves identically when deployed on the streaming path.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Rolling windows, expressed in seconds, used for multi-horizon features.
WINDOWS = {
    "fast": 5,
    "short": 30,
    "mid": 60,
    "long": 300,      # 5 min
    "xlong": 1800,    # 30 min
    "xxlong": 3600,   # 1 hour
}

FEATURE_NAMES = (
    [f"ret_{k}" for k in WINDOWS] +
    [f"vol_{k}" for k in WINDOWS] +
    ["ema_fast_dev", "ema_slow_dev", "macd", "rsi_14",
     "bb_pos", "vol_zscore", "taker_buy_ratio", "trade_intensity"]
)
N_FEATURES = len(FEATURE_NAMES)


def _rolling_return(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(window).fillna(0.0)


def _rolling_vol(logret: pd.Series, window: int) -> pd.Series:
    return logret.rolling(window, min_periods=1).std().fillna(0.0)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def compute_features_batch(df: pd.DataFrame) -> np.ndarray:
    """
    df: DataFrame with columns open_time, open, high, low, close, volume,
        quote_volume, trades, taker_buy_base (as produced by download_data.py)
    Returns: float32 array [N, N_FEATURES]
    """
    close = df["close"].astype("float64")
    logret = np.log(close / close.shift(1)).fillna(0.0)

    feats = {}
    for k, w in WINDOWS.items():
        feats[f"ret_{k}"] = _rolling_return(close, w)
        feats[f"vol_{k}"] = _rolling_vol(logret, w)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=60, adjust=False).mean()
    feats["ema_fast_dev"] = (close - ema_fast) / (ema_fast + 1e-9)
    feats["ema_slow_dev"] = (close - ema_slow) / (ema_slow + 1e-9)
    feats["macd"] = (ema_fast - ema_slow) / (ema_slow + 1e-9)
    feats["rsi_14"] = _rsi(close, 14) / 100.0

    mid = close.rolling(60, min_periods=1).mean()
    std = close.rolling(60, min_periods=1).std().fillna(0.0)
    feats["bb_pos"] = ((close - mid) / (2 * std + 1e-9)).clip(-3, 3)

    vol = df["volume"].astype("float64")
    vol_mean = vol.rolling(300, min_periods=1).mean()
    vol_std = vol.rolling(300, min_periods=1).std().fillna(0.0)
    feats["vol_zscore"] = ((vol - vol_mean) / (vol_std + 1e-9)).clip(-5, 5)

    taker_buy = df["taker_buy_base"].astype("float64")
    feats["taker_buy_ratio"] = (taker_buy / (vol + 1e-9)).clip(0, 1)

    trades = df["trades"].astype("float64")
    tr_mean = trades.rolling(300, min_periods=1).mean()
    feats["trade_intensity"] = (trades / (tr_mean + 1e-9)).clip(0, 10)

    mat = np.stack([feats[name].to_numpy(dtype="float32") for name in FEATURE_NAMES], axis=1)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat


@dataclass
class StreamingAnalyser:
    """Incremental version for live/simulated second-by-second playback."""
    max_window: int = max(WINDOWS.values())
    _closes: deque = field(default_factory=lambda: deque(maxlen=3601))
    _vols: deque = field(default_factory=lambda: deque(maxlen=301))
    _trades: deque = field(default_factory=lambda: deque(maxlen=301))
    _taker: deque = field(default_factory=lambda: deque(maxlen=2))
    _ema_fast: float | None = None
    _ema_slow: float | None = None

    def reset(self):
        self._closes.clear()
        self._vols.clear()
        self._trades.clear()
        self._ema_fast = None
        self._ema_slow = None

    def update(self, close: float, volume: float, trades: int, taker_buy_base: float) -> np.ndarray:
        self._closes.append(close)
        self._vols.append(volume)
        self._trades.append(trades)
        arr = np.array(self._closes, dtype="float64")

        self._ema_fast = close if self._ema_fast is None else (
            close * (2 / 13) + self._ema_fast * (11 / 13))
        self._ema_slow = close if self._ema_slow is None else (
            close * (2 / 61) + self._ema_slow * (59 / 61))

        out = np.zeros(N_FEATURES, dtype="float32")
        idx = 0
        n = len(arr)
        logret = np.diff(np.log(arr + 1e-12)) if n > 1 else np.array([0.0])
        for k, w in WINDOWS.items():
            w = min(w, n - 1) if n > 1 else 0
            ret = (arr[-1] / arr[-1 - w] - 1.0) if w > 0 else 0.0
            out[idx] = ret
            idx += 1
            seg = logret[-w:] if w > 0 else logret[-1:]
            out[idx] = float(np.std(seg)) if len(seg) > 0 else 0.0
            idx += 1

        out[idx] = (close - self._ema_fast) / (self._ema_fast + 1e-9); idx += 1
        out[idx] = (close - self._ema_slow) / (self._ema_slow + 1e-9); idx += 1
        out[idx] = (self._ema_fast - self._ema_slow) / (self._ema_slow + 1e-9); idx += 1

        period = min(14, n - 1) if n > 1 else 0
        if period > 0:
            deltas = np.diff(arr[-period - 1:])
            gain = deltas[deltas > 0].sum() / period
            loss = -deltas[deltas < 0].sum() / period
            rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
        else:
            rsi = 50.0
        out[idx] = rsi / 100.0; idx += 1

        seg60 = arr[-60:]
        mid = seg60.mean()
        std = seg60.std()
        out[idx] = float(np.clip((close - mid) / (2 * std + 1e-9), -3, 3)); idx += 1

        vol_arr = np.array(self._vols, dtype="float64")
        vmean, vstd = vol_arr.mean(), vol_arr.std()
        out[idx] = float(np.clip((volume - vmean) / (vstd + 1e-9), -5, 5)); idx += 1

        out[idx] = float(np.clip(taker_buy_base / (volume + 1e-9), 0, 1)); idx += 1

        tr_arr = np.array(self._trades, dtype="float64")
        tmean = tr_arr.mean()
        out[idx] = float(np.clip(trades / (tmean + 1e-9), 0, 10)); idx += 1

        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

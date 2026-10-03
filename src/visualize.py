"""
visualize.py
-------------
Phase 2C: real-time (1 fps) animated visualization.

Shows, for a chosen historical day replayed through the live market
simulator:
  - Actual market price (solid line)
  - Model's predicted price curve, 25 minutes ahead (dotted line)
  - A rolling accuracy bar (1 - normalized error vs. the REAL, non-simulated,
    stored market data for that same moment)

Implementation note: the exact same `LiveChart` class is used by:
  - run_demo.py             -> a local matplotlib window / saved .mp4 / .gif
  - notebooks/snowflake_live_demo.ipynb -> inline Jupyter display, using the
    `IPython.display` clear_output+display loop, which is the one approach
    that renders identically as a plain script AND inside any Jupyter-like
    notebook (including Snowflake Notebooks), without needing ipywidgets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # safe default; run_demo.py switches backend if a display exists
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    plt = None


@dataclass
class LiveChartState:
    times: list = field(default_factory=list)
    actual: list = field(default_factory=list)
    pred_times: list = field(default_factory=list)   # x for the dotted prediction curve
    pred_values: list = field(default_factory=list)
    accuracy_history: list = field(default_factory=list)


class LiveChart:
    """Stateful 1fps chart: call `.push_frame(...)` once per simulated second."""

    def __init__(self, title: str = "BTC Price Predictor - Live Replay", accuracy_window: int = 300):
        self.state = LiveChartState()
        self.title = title
        self.accuracy_window = accuracy_window
        self.fig = None
        self.ax_price = None
        self.ax_acc = None

    def _ensure_fig(self):
        if self.fig is not None:
            return
        self.fig, (self.ax_price, self.ax_acc) = plt.subplots(
            2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]}
        )
        self.fig.suptitle(self.title)

    def push_frame(self, t_index: int, actual_price: float,
                   predicted_path: np.ndarray, accuracy_now: float | None):
        """
        t_index:        current simulated second (monotonic integer / timestamp)
        actual_price:   real price at t_index
        predicted_path: array of length `horizon`, model's predicted prices for
                         t_index+1 .. t_index+horizon
        accuracy_now:   e.g. 1 - |pred_25min_ago_for_now - actual_now| / actual_now,
                         or None if not enough history yet to score
        """
        s = self.state
        s.times.append(t_index)
        s.actual.append(actual_price)

        horizon = len(predicted_path)
        s.pred_times = list(range(t_index, t_index + horizon))
        s.pred_values = list(predicted_path)

        if accuracy_now is not None:
            s.accuracy_history.append(accuracy_now)
            if len(s.accuracy_history) > self.accuracy_window:
                s.accuracy_history.pop(0)

    def render(self):
        """Draw the current state onto self.fig. Call once per second (1 fps)."""
        self._ensure_fig()
        s = self.state
        self.ax_price.clear()
        self.ax_acc.clear()

        self.ax_price.plot(s.times, s.actual, color="#1f77b4", linewidth=1.6, label="Actual price")
        if s.pred_times:
            self.ax_price.plot(s.pred_times, s.pred_values, color="#d62728",
                                linewidth=1.6, linestyle=":", label="Predicted price (+25min)")
        self.ax_price.axvline(s.times[-1], color="gray", linewidth=0.5, alpha=0.5)
        self.ax_price.set_ylabel("USDT")
        self.ax_price.legend(loc="upper left")
        self.ax_price.set_title("Actual vs. Predicted BTC/USDT price")

        acc = np.mean(s.accuracy_history) if s.accuracy_history else 0.0
        self.ax_acc.barh(["Accuracy"], [max(0.0, min(1.0, acc))], color="#2ca02c")
        self.ax_acc.set_xlim(0, 1)
        self.ax_acc.set_title(f"Rolling model accuracy (last {len(s.accuracy_history)}s): {acc*100:.1f}%")

        self.fig.tight_layout()
        return self.fig


def accuracy_from_error(pred_price: float, true_price: float) -> float:
    """1.0 = perfect, decays towards 0 as relative error grows."""
    rel_err = abs(pred_price - true_price) / max(true_price, 1e-9)
    return float(np.clip(1.0 - rel_err / 0.02, 0.0, 1.0))  # 2% error -> 0 accuracy

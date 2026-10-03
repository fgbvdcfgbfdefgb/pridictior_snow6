"""
stability.py
-------------
Utilities that keep the Price Predictor's output smooth/stable, since its
predictions are meant to drive real trading decisions and erratic jumps
second-to-second are unacceptable.

Three complementary mechanisms:

1. Architectural: the model already bounds each second's incremental
   log-return via tanh() before cumsum (see price_predictor.py).
2. Loss-based: `smoothness_penalty` punishes large second-derivative
   (jerk) in the predicted path, and `temporal_consistency_loss` punishes
   the prediction made at t from disagreeing too much with the (shifted)
   prediction made at t-1 for the overlapping part of the horizon.
3. Weight-based: `EMAModel` keeps a Polyak-averaged shadow copy of the
   model's weights; the shadow model (not the raw fast-moving training
   weights) is what actually gets used for live inference/visualization,
   which removes most of the remaining high-frequency jitter.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def smoothness_penalty(path: torch.Tensor) -> torch.Tensor:
    """path: [B, H] predicted (price or log-return) path. Penalize 2nd derivative."""
    if path.size(1) < 3:
        return path.new_zeros(())
    d1 = path[:, 1:] - path[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return (d2 ** 2).mean()


def temporal_consistency_loss(pred_t: torch.Tensor, pred_tm1_shifted: torch.Tensor) -> torch.Tensor:
    """
    pred_t:            [B, H]   prediction made now, for t+1..t+H
    pred_tm1_shifted:  [B, H-1] prediction made 1 second ago for the SAME
                        absolute timestamps (i.e. its [1:] slice), already
                        aligned by the caller.
    Encourages consecutive 1-second-apart forecasts of the same future
    moment to agree, which is the direct definition of "no erratic jumps".
    """
    h = pred_tm1_shifted.size(1)
    return F.smooth_l1_loss(pred_t[:, :h], pred_tm1_shifted)


def prediction_loss(pred_log_return_path: torch.Tensor, true_log_return_path: torch.Tensor,
                     smoothness_weight: float = 0.1,
                     consistency_weight: float = 0.1,
                     prev_pred_shifted: torch.Tensor | None = None) -> dict:
    main = F.smooth_l1_loss(pred_log_return_path, true_log_return_path)
    smooth = smoothness_penalty(pred_log_return_path)
    total = main + smoothness_weight * smooth
    out = {"main": main, "smoothness": smooth}
    if prev_pred_shifted is not None:
        cons = temporal_consistency_loss(pred_log_return_path, prev_pred_shifted)
        total = total + consistency_weight * cons
        out["consistency"] = cons
    out["total"] = total
    return out


class EMAModel:
    """Polyak-averaged shadow weights for stable inference/visualization."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

    def eval_model(self) -> nn.Module:
        self.shadow.eval()
        return self.shadow

#!/usr/bin/env python3
"""
run_demo.py
------------
Phase 2C end-to-end demo: picks a random day from the downloaded history,
replays it second-by-second through the live MarketSimulator, runs it
through the (optionally quickly pre-trained) Price Predictor, and renders
a 1-fps animated chart:
    - actual price (solid)
    - predicted price, 25 min ahead (dotted)
    - rolling accuracy bar vs. the REAL stored market data

This script is CPU-friendly and intended for local/sandbox use and for
quickly sanity-checking a trained checkpoint. For the full-size model
trained with real distributed GPU compute, use scripts on Snowflake
(see train_distributed.sh) and simply point --checkpoint at the resulting
.pt file.

Examples:
    # quick, from-scratch tiny model, 10 minutes of a random day, saved gif
    python run_demo.py --debug_months 3 --train_steps 300 --demo_seconds 600

    # load a real trained checkpoint (produced by src/online_trainer.py)
    python run_demo.py --checkpoint checkpoints/ema_final.pt --demo_seconds 3600
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import available_months, load_months, load_full_history, pick_random_day, MarketData
from src.online_trainer import (
    precompute_shard_tensors, build_training_windows, ReplayBuffer, setup_distributed,
)
from src.price_predictor import PricePredictor, PredictorConfig
from src.stability import EMAModel, prediction_loss
from src.visualize import LiveChart, accuracy_from_error
from src.checkpoint_utils import resolve_checkpoint


def quick_train(model, shard, cfg, steps, batch_size=4, lr=1e-3, train_frac=0.7, seed=0):
    n = len(shard["close"])
    train_end = int(n * train_frac)
    start = cfg.context
    end = min(train_end, n - cfg.horizon)
    if end <= start:
        print("[quick_train] not enough data to train; using randomly initialized weights.")
        return
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(5000)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    idxs = list(range(start, end))
    rng.shuffle(idxs)  # fine here: this is a quick illustrative pretrain, not the real online run
    pos = 0
    for step in range(steps):
        while len(buffer) < batch_size and pos < len(idxs):
            raw, feat, target = build_training_windows(shard, idxs[pos], cfg.context, cfg.horizon)
            buffer.push((raw, feat, target))
            pos += 1
        if len(buffer) < batch_size:
            break
        batch = buffer.sample(batch_size, rng)
        raw_b = torch.from_numpy(np.stack([b[0] for b in batch]))
        feat_b = torch.from_numpy(np.stack([b[1] for b in batch]))
        tgt_b = torch.from_numpy(np.stack([b[2] for b in batch]))
        pred = model(raw_b, feat_b)
        losses = prediction_loss(pred, tgt_b, smoothness_weight=0.1)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"[quick_train] step {step+1}/{steps} loss={losses['total'].item():.6f}")
        while len(buffer) < batch_size and pos < len(idxs):
            raw, feat, target = build_training_windows(shard, idxs[pos], cfg.context, cfg.horizon)
            buffer.push((raw, feat, target))
            pos += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_months", type=int, default=3,
                     help="how many downloaded months to load for this demo (keep small on CPU)")
    ap.add_argument("--context", type=int, default=3600,
                     help="trailing context seconds fed to the model (full prod run uses 43200 = 12h)")
    ap.add_argument("--horizon", type=int, default=1500, help="25 minutes, per spec")
    ap.add_argument("--train_steps", type=int, default=300,
                     help="quick illustrative pretraining steps if no --checkpoint given")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="path to a .pt file saved by src/online_trainer.py (EMA weights)")
    ap.add_argument("--demo_seconds", type=int, default=600,
                     help="how many simulated seconds (frames) to animate, 1 per second of video")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="assets/demo.gif")
    ap.add_argument("--limit_seconds", type=int, default=0,
                     help="if >0, additionally truncate loaded data (low-RAM smoke testing)")
    args = ap.parse_args()

    months = available_months()
    if not months:
        print("No data downloaded yet. Run scripts/download_data.py first.")
        sys.exit(1)
    months = months[: args.debug_months] if args.debug_months > 0 else months
    print(f"[data] loading months: {months}")
    md = load_months(months)
    if args.limit_seconds > 0:
        n = args.limit_seconds
        md = MarketData(
            open_time_ms=md.open_time_ms[:n], open=md.open[:n], high=md.high[:n],
            low=md.low[:n], close=md.close[:n], volume=md.volume[:n],
            quote_volume=md.quote_volume[:n], trades=md.trades[:n],
            taker_buy_base=md.taker_buy_base[:n],
        )
    print(f"[data] {len(md):,} seconds loaded "
          f"({md.datetimes[0]} .. {md.datetimes[len(md)-1]})")

    cfg = PredictorConfig.small_cpu_debug()
    cfg.context, cfg.horizon = args.context, args.horizon
    model = PricePredictor(cfg)

    shard = precompute_shard_tensors(md, 0, len(md))

    if args.checkpoint and (os.path.exists(args.checkpoint) or os.path.exists(args.checkpoint + ".manifest.json")):
        ckpt = torch.load(resolve_checkpoint(args.checkpoint), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"[model] loaded checkpoint {args.checkpoint} (step {ckpt.get('step')})")
    else:
        print("[model] no checkpoint given/found -> running a short illustrative pretrain "
              "(for a real high-accuracy model, train at scale via src/online_trainer.py "
              "on Snowflake's 4xA10 GPUs, then pass --checkpoint).")
        quick_train(model, shard, cfg, steps=args.train_steps, seed=args.seed)

    model.eval()

    n = len(shard["close"])
    lo = args.context
    hi = n - args.horizon
    rng = np.random.default_rng(args.seed)
    day_start = rng.integers(lo, max(lo + 1, hi - args.demo_seconds))
    demo_len = min(args.demo_seconds, hi - day_start)
    print(f"[demo] replaying {demo_len} simulated seconds starting at local idx {day_start} "
          f"({md.datetimes[day_start]})")

    chart = LiveChart(title=f"BTC/USDT Live Replay starting {md.datetimes[day_start]}")
    pending_predictions = {}  # future_abs_idx -> predicted_price (made horizon seconds ago)

    frames_data = []
    with torch.no_grad():
        for step, idx in enumerate(range(day_start, day_start + demo_len)):
            raw, feat, _ = build_training_windows(shard, idx, cfg.context, cfg.horizon)
            raw_t = torch.from_numpy(raw).unsqueeze(0)
            feat_t = torch.from_numpy(feat).unsqueeze(0)
            cum_log_ret = model(raw_t, feat_t).squeeze(0).numpy()
            last_price = shard["close"][idx]
            pred_path = last_price * np.exp(cum_log_ret)

            for h, p in enumerate(pred_path, start=1):
                pending_predictions[idx + h] = float(p)

            accuracy_now = None
            if idx in pending_predictions:
                made_pred = pending_predictions.pop(idx)
                accuracy_now = accuracy_from_error(made_pred, float(shard["close"][idx]))

            frames_data.append({
                "idx": idx, "actual": float(shard["close"][idx]),
                "pred_path": pred_path, "accuracy": accuracy_now,
            })
            if (step + 1) % 100 == 0:
                print(f"[demo] frame {step+1}/{demo_len}")

    fig_holder = {}

    def update(i):
        fd = frames_data[i]
        chart.push_frame(fd["idx"], fd["actual"], fd["pred_path"], fd["accuracy"])
        return [chart.render()]

    chart._ensure_fig()
    ani = animation.FuncAnimation(chart.fig, update, frames=len(frames_data), interval=1000, blit=False)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[render] saving {len(frames_data)} frames @ 1fps -> {args.out}")
    writer = animation.PillowWriter(fps=1)
    ani.save(args.out, writer=writer)
    print(f"[done] saved animation -> {args.out}")


if __name__ == "__main__":
    main()

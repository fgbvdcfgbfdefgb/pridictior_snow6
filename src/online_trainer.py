"""
online_trainer.py
------------------
Continuous / "no epochs" distributed training loop for the Price Predictor.

Design, mapped directly to the project requirements:

  * "no epoches as it will have realtime reward function (sec-sec as per
    market time)"
      -> We never shuffle-and-repeat a fixed dataset for N epochs. Instead we
         walk forward through history exactly once, in chronological order
         (optionally --loop forever for a persistent "live" deployment-style
         process), and the "reward"/loss at simulated second t is only
         computed once the true market outcome at t+25min has actually been
         observed (i.e. is already in our downloaded historical record) --
         never using information from the future relative to t.

  * "Train on multiple gpus using distributed training"
      -> Standard PyTorch DistributedDataParallel (DDP), launched with
         torchrun. Each rank is assigned a disjoint contiguous time shard of
         the full history (so GPU 0 trains on ~the oldest quarter of history,
         GPU 3 on ~the most recent quarter, etc.) and gradients are
         all-reduced across ranks every step, same as any DDP job -- the
         "streaming" nature only affects how each rank *constructs* its
         batches, not the distributed mechanics.

  * "Price Predictor receives the Market Analyser's output ... and also the
    last 12h sec-sec data"
      -> see build_training_windows() below.

  * "Predictions must be stable"
      -> see src/stability.py (smoothness penalty, temporal consistency
         penalty, EMA shadow weights used for the actually-deployed model).

Run (single GPU / CPU smoke test):
    python -m src.online_trainer --config small_cpu_debug --max_steps 50

Run (4x A10 on Snowflake, full size):
    torchrun --standalone --nproc_per_node=4 -m src.online_trainer \
        --config large --checkpoint_dir checkpoints/run1
"""
from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .dataset import load_full_history, load_months, available_months, MarketData
from .market_analyser import compute_features_batch, N_FEATURES
from .price_predictor import PricePredictor, PredictorConfig, count_parameters
from .stability import EMAModel, prediction_loss


def setup_distributed():
    """Initializes torch.distributed if launched via torchrun; else returns a
    single-process (rank 0 / world 1) fallback so this script also works for
    local CPU smoke-testing without torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        return rank, world_size, device, True
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, False


def shard_range(n_total: int, rank: int, world_size: int) -> tuple[int, int]:
    chunk = n_total // world_size
    lo = rank * chunk
    hi = n_total if rank == world_size - 1 else lo + chunk
    return lo, hi


def precompute_shard_tensors(md: MarketData, lo: int, hi: int):
    """Precompute full-resolution log-price arrays + analyser features for a shard.
    Kept in plain numpy (CPU RAM) -- with 100GB RAM this comfortably holds many
    years of 1-second data; GPUs only ever see small windowed batches."""
    import pandas as pd
    df = pd.DataFrame({
        "open_time": md.open_time_ms[lo:hi], "open": md.open[lo:hi], "high": md.high[lo:hi],
        "low": md.low[lo:hi], "close": md.close[lo:hi], "volume": md.volume[lo:hi],
        "quote_volume": md.quote_volume[lo:hi], "trades": md.trades[lo:hi],
        "taker_buy_base": md.taker_buy_base[lo:hi],
    })
    feats = compute_features_batch(df)  # [n, N_FEATURES]
    eps = 1e-9
    log_o = np.log(df["open"].to_numpy(dtype="float64") + eps)
    log_h = np.log(df["high"].to_numpy(dtype="float64") + eps)
    log_l = np.log(df["low"].to_numpy(dtype="float64") + eps)
    log_c = np.log(df["close"].to_numpy(dtype="float64") + eps)
    log_v = np.log1p(df["volume"].to_numpy(dtype="float64")) / 10.0
    return {
        "log_o": log_o.astype("float32"), "log_h": log_h.astype("float32"),
        "log_l": log_l.astype("float32"), "log_c": log_c.astype("float32"),
        "log_v": log_v.astype("float32"), "feats": feats.astype("float32"),
        "close": df["close"].to_numpy(dtype="float32"),
    }


def build_training_windows(shard: dict, idx: int, context: int, horizon: int):
    """Builds (raw_window, feat_window, target_log_return_path) for one anchor idx
    local to the shard. idx must satisfy idx-context+1 >= 0 and idx+horizon < len."""
    lo = idx - context + 1
    anchor = shard["log_c"][idx]
    raw = np.stack([
        shard["log_o"][lo: idx + 1] - anchor,
        shard["log_h"][lo: idx + 1] - anchor,
        shard["log_l"][lo: idx + 1] - anchor,
        shard["log_c"][lo: idx + 1] - anchor,
        shard["log_v"][lo: idx + 1],
    ], axis=1)
    feat = shard["feats"][lo: idx + 1]
    future_log_c = shard["log_c"][idx + 1: idx + 1 + horizon]
    target = future_log_c - anchor
    return raw.astype("float32"), feat.astype("float32"), target.astype("float32")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, item):
        self.buf.append(item)

    def sample(self, batch_size: int, rng: np.random.Generator):
        n = len(self.buf)
        idxs = rng.integers(0, n, size=min(batch_size, n))
        return [self.buf[i] for i in idxs]

    def __len__(self):
        return len(self.buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["small_cpu_debug", "large"], default="large")
    ap.add_argument("--context", type=int, default=12 * 60 * 60)
    ap.add_argument("--horizon", type=int, default=25 * 60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--replay_capacity", type=int, default=20000)
    ap.add_argument("--warmup_seconds", type=int, default=2000,
                     help="min ticks observed before first optimizer step")
    ap.add_argument("--step_every", type=int, default=1,
                     help="take one optimizer step every N simulated seconds "
                          "(1 = truest to 'updates every second')")
    ap.add_argument("--max_steps", type=int, default=0, help="0 = run until shard exhausted")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--checkpoint_every", type=int, default=2000)
    ap.add_argument("--loop", action="store_true", help="loop forever over the shard (live-style)")
    ap.add_argument("--debug_months", type=int, default=0,
                     help="if >0, only load this many months of data (low-RAM smoke testing); "
                          "0 = load the FULL downloaded history (use on Snowflake's 100GB RAM box)")
    ap.add_argument("--debug_limit_seconds", type=int, default=0,
                     help="if >0, additionally truncate loaded data to this many seconds "
                          "(for very low-RAM smoke testing, e.g. this dev sandbox)")
    args = ap.parse_args()

    rank, world_size, device, is_distributed = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"[init] world_size={world_size} device={device} distributed={is_distributed}", flush=True)

    if args.debug_months > 0:
        months = available_months()[: args.debug_months]
        md = load_months(months)
    else:
        md = load_full_history()
    if args.debug_limit_seconds > 0:
        n = args.debug_limit_seconds
        md = MarketData(
            open_time_ms=md.open_time_ms[:n], open=md.open[:n], high=md.high[:n],
            low=md.low[:n], close=md.close[:n], volume=md.volume[:n],
            quote_volume=md.quote_volume[:n], trades=md.trades[:n],
            taker_buy_base=md.taker_buy_base[:n],
        )
    lo, hi = shard_range(len(md), rank, world_size)
    if is_main:
        print(f"[rank {rank}] shard = [{lo}, {hi}) -> {hi-lo:,} seconds "
              f"({md.datetimes[lo]} .. {md.datetimes[hi-1]})")
    shard = precompute_shard_tensors(md, lo, hi)

    cfg = PredictorConfig.small_cpu_debug() if args.config == "small_cpu_debug" else PredictorConfig()
    cfg.context, cfg.horizon = args.context, args.horizon
    assert cfg.n_analyser_features == N_FEATURES

    model = PricePredictor(cfg).to(device)
    if is_main:
        print(f"[model] params = {count_parameters(model):,}")
    if is_distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    ema = EMAModel(model.module if is_distributed else model, decay=0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    buffer = ReplayBuffer(args.replay_capacity)
    rng = np.random.default_rng(1234 + rank)
    prev_preds = {}  # idx -> last prediction, for temporal consistency loss

    n_local = len(shard["close"])
    start = args.context  # need full context before first usable anchor
    end = n_local - args.horizon  # need realized future before last usable anchor
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    t0 = time.time()
    passes = 0
    while True:
        for idx in range(start, end):
            raw, feat, target = build_training_windows(shard, idx, args.context, args.horizon)
            buffer.push((raw, feat, target))

            ready = len(buffer) >= max(args.warmup_seconds, args.batch_size)
            if ready and (idx % args.step_every == 0):
                batch = buffer.sample(args.batch_size, rng)
                raw_b = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
                feat_b = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
                tgt_b = torch.from_numpy(np.stack([b[2] for b in batch])).to(device)

                pred = model(raw_b, feat_b)
                losses = prediction_loss(pred, tgt_b, smoothness_weight=0.1, consistency_weight=0.0)

                opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model.module if is_distributed else model)

                step += 1
                if is_main and step % 50 == 0:
                    dt = time.time() - t0
                    print(f"[step {step}] loss={losses['total'].item():.6f} "
                          f"main={losses['main'].item():.6f} smooth={losses['smoothness'].item():.6f} "
                          f"({step/dt:.1f} steps/s)")
                if is_main and step % args.checkpoint_every == 0:
                    ckpt_path = os.path.join(args.checkpoint_dir, f"ema_step{step}.pt")
                    torch.save({"model": ema.shadow.state_dict(), "cfg": cfg, "step": step}, ckpt_path)
                    print(f"[checkpoint] saved {ckpt_path}")
                if args.max_steps and step >= args.max_steps:
                    if is_main:
                        print("[done] reached max_steps")
                    return
        passes += 1
        if is_main:
            print(f"[info] shard exhausted (pass {passes}); "
                  f"{'looping' if args.loop else 'stopping'} ...")
        if not args.loop:
            break

    if is_main:
        final_path = os.path.join(args.checkpoint_dir, "ema_final.pt")
        torch.save({"model": ema.shadow.state_dict(), "cfg": cfg, "step": step}, final_path)
        print(f"[done] saved {final_path}")


if __name__ == "__main__":
    main()

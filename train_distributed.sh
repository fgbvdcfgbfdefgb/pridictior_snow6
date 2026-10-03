#!/usr/bin/env bash
# train_distributed.sh
# ----------------------
# Launches the full-size Price Predictor training job across all GPUs on the
# box using PyTorch's torchrun (DistributedDataParallel). Designed for
# Snowflake's GPU_NV_M-class compute pool: 4x A10 (23GB VRAM each), 48 vCPU,
# 100GB RAM.
#
# 100% OFFLINE: this only reads the Parquet files already committed under
# data/raw_parquet/ (cloned in via Snowflake's Git integration). No network
# access is required or attempted at runtime.
#
# Usage (from the repo root, inside the Snowflake Notebook terminal / a cell
# with `!bash train_distributed.sh`):
#   bash train_distributed.sh
#
# Tune NUM_GPUS / batch size / etc. via env vars, e.g.:
#   NUM_GPUS=2 BATCH_SIZE=16 bash train_distributed.sh
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-48}"
LR="${LR:-2e-4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/run_$(date +%Y%m%d_%H%M%S)}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-50000}"
WARMUP_SECONDS="${WARMUP_SECONDS:-5000}"
STEP_EVERY="${STEP_EVERY:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$CHECKPOINT_DIR"
echo "== Bitcoin Price Predictor :: distributed training =="
echo "GPUs: $NUM_GPUS | batch_size(per-rank): $BATCH_SIZE | lr: $LR"
echo "checkpoint_dir: $CHECKPOINT_DIR"

torchrun \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  -m src.online_trainer \
  --config large \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --replay_capacity "${REPLAY_CAPACITY}" \
  --warmup_seconds "${WARMUP_SECONDS}" \
  --step_every "${STEP_EVERY}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --checkpoint_every "${CHECKPOINT_EVERY}" \
  ${EXTRA_ARGS}

echo "Training finished. Checkpoints in: ${CHECKPOINT_DIR}"

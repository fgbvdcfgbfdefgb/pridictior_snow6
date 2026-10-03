# Bitcoin Price Predictor (`pridictior_snow6`)

Second-by-second BTC/USDT price data (2020 → today) → a live market simulator
→ a CPU "Market Analyser" feature extractor → a GPU "Price Predictor"
transformer trained **continuously, online, across multiple GPUs**, to
forecast price 25 minutes ahead, every second, with stability guarantees —
visualized as a live 1 fps animated chart, runnable **entirely offline inside
Snowflake**.

---

## 0. TL;DR — what's in this repo

| Path | What |
|---|---|
| `data/raw_parquet/*.parquet` | BTC/USDT 1-second OHLCV candles, 2020-01 → latest available month, one file per month (~45-80MB each, GitHub-safe) |
| `data/manifest.json` | Row counts / checksummable metadata for every downloaded month |
| `scripts/download_data.py` | (Re)builds the dataset from Binance's free public archive. **Needs internet. Run outside Snowflake.** |
| `scripts/verify_data.py` | Sanity-checks the committed dataset (gaps, NaNs, row counts) |
| `scripts/split_large_file.py` | Splits/joins any file over GitHub's 100MB limit (used for big model checkpoints) |
| `src/dataset.py` | Loads Parquet months into a single chronological array; picks random valid days |
| `src/simulator.py` | The **live market simulator** — replays history second-by-second |
| `src/market_analyser.py` | The **Market Analyser** (CPU) — turns raw ticks into ~20 momentum/vol/flow features |
| `src/price_predictor.py` | The **Price Predictor** (GPU) — hierarchical TCN→Transformer model, predicts 25-min price path |
| `src/stability.py` | Smoothness/consistency losses + EMA weight averaging, so predictions don't jump around |
| `src/online_trainer.py` | The **continuous, no-epochs, multi-GPU (DDP) training loop** |
| `src/visualize.py` | The **1 fps live chart** (actual vs. predicted-dotted vs. accuracy bar) |
| `run_demo.py` | CPU-friendly end-to-end demo script (local / sandbox use, saves a `.gif`) |
| `train_distributed.sh` | Launches full-size training across all GPUs via `torchrun` (for Snowflake) |
| `notebooks/snowflake_live_demo.ipynb` | The Snowflake Notebook: load data → (optionally) train → live-updating chart |
| `checkpoints/` | Trained model weights land here (split into GitHub-safe parts if needed) |

---

## 1. Honesty section — please read before you run anything

This is a genuinely large, ambitious system, and a few things were adapted
from the original wording of the plan to something that actually works. In
the interest of not quietly bending the spec:

1. **"Second-by-second data" is real, native 1-second OHLCV candles from
   Binance's public archive (`data.binance.vision`)** — free, no API key,
   covering 2020-01 through the most recently fully-elapsed month. This is
   **not** synthetic/interpolated data; it's Binance's own 1-second kline
   series. (Raw individual trade ticks would be hundreds of GB and isn't
   needed for a 1-second-resolution predictor anyway.)
2. **"No epochs / real-time reward function"** is implemented as
   **continuous online learning**: the model streams forward through history
   exactly once (chronologically, never shuffled-and-repeated), and only
   computes a loss for second *t* once the real outcome at *t+25min* has
   already been observed in the historical record — never peeking at the
   future. This is the standard, well-understood way to do what the brief
   describes, without inventing an under-specified RL reward function for a
   task (price regression) that doesn't need one.
3. **Multi-GPU distributed training** uses PyTorch `DistributedDataParallel`
   via `torchrun`, which is the standard, production-grade way to split
   training across your 4 A10s. Each GPU streams a different quarter of the
   calendar timeline and gradients are synced every step.
4. **This repo was assembled in a small CPU-only sandbox** (no GPU, ~2GB RAM,
   no access to your actual Snowflake account). Everything was built and
   function-tested end-to-end here using a small slice of real downloaded
   data and a shrunk-down model config (see "What was actually tested"
   below) — but the **large-scale 4-GPU training run on the full 6 years of
   data has not been executed anywhere**, because no GPU exists in this
   environment. `train_distributed.sh` / `src/online_trainer.py --config
   large` is what you'll run on Snowflake to actually produce a trained,
   production-accuracy model.
5. **GitHub's 100MB-per-file limit** is real and not optional (without Git
   LFS). Monthly data files stay under it naturally (~45-80MB). Full-size
   model checkpoints (~800MB+) will not — use
   `scripts/split_large_file.py split checkpoints/your_model.pt` before
   committing; everything that reads checkpoints (`run_demo.py`, the
   notebook) auto-reassembles split files transparently.

### What was actually tested in this sandbox
- Full download pipeline (`scripts/download_data.py`) against the real
  Binance archive — verified end to end.
- `src/market_analyser.py`, `src/simulator.py`, `src/dataset.py` — unit
  smoke-tested against real downloaded months.
- `src/price_predictor.py`, `src/stability.py`, `src/online_trainer.py` — a
  shrunk debug model (`PredictorConfig.small_cpu_debug()`) was trained for a
  handful of steps on real data on CPU to confirm the forward/backward/DDP
  wiring is correct (`python -m src.online_trainer --config small_cpu_debug
  --debug_months 1 ...`).
- `run_demo.py` — ran a full loop: load real data → quick CPU pretrain → live
  simulator replay → model inference every second → rendered 1 fps GIF with
  actual/predicted/accuracy exactly as specified.
- The **architecture, sizes, and training recipe for the full "large" model**
  are implemented and documented, ready to run on your 4x A10 box, but their
  real-world prediction *accuracy* at full scale is untested (that number
  only means something after a real multi-day GPU training run on your
  hardware, which this plan sets up for you to do).

---

## 2. Dataset

```bash
pip install -r requirements.txt
python scripts/download_data.py                 # fetch everything available, resumable
python scripts/verify_data.py                   # sanity check
```

Source: `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1s/` —
Binance's own free historical archive, no API key required. Each month is
converted to a compact Parquet file with columns:
`open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_base`.

To extend the dataset later (new months become available monthly):
```bash
python scripts/download_data.py --start 2026-09
```

This step needs internet and should be run **outside Snowflake** (your
laptop, this sandbox, a CI runner, etc.) — then just `git push` the new
Parquet files.

---

## 3. Architecture

```
 ┌─────────────────────────┐        ┌──────────────────────────────┐
 │   Live Market Simulator  │  raw   │     Market Analyser (CPU)    │
 │  (src/simulator.py)      ├───────►│  ~20 momentum/vol/flow feats │
 │  replays 1s ticks from   │        │  (src/market_analyser.py)    │
 │  data/raw_parquet/       │        └───────────────┬──────────────┘
 └─────────────┬────────────┘                        │
               │ raw 12h window                       │ features, 12h window
               └───────────────────────┬──────────────┘
                                        ▼
                      ┌──────────────────────────────────┐
                      │     Price Predictor (GPU, DDP)    │
                      │  src/price_predictor.py           │
                      │  1) dilated causal conv stack      │
                      │     43200 → ~85 tokens             │
                      │  2) Transformer encoder (8 layers) │
                      │  3) Transformer decoder w/ 1500    │
                      │     learned future-second queries  │
                      │  4) tanh-bounded Δlogreturn → cumsum│
                      └───────────────┬────────────────────┘
                                      ▼
                     predicted price path, next 25 minutes
                     (anchored at current price, smooth by
                      construction + smoothness loss + EMA)
```

- **Input**: last 12h of raw OHLCV (43,200 seconds) + the Market Analyser's
  feature vector over the same window.
- **Output**: a 1500-point price path (one prediction per second, 25 minutes
  ahead), updated from scratch every simulated second.
- **Stability**: (a) each second's incremental log-return is `tanh`-bounded
  before cumulative-summing into a price path — it is architecturally
  impossible for the model to output a teleporting price; (b) a smoothness
  loss penalizes jagged second-derivatives during training; (c) an EMA
  (Polyak-averaged) shadow copy of the weights is what's actually used for
  live inference, damping remaining high-frequency weight noise.
- **Default ("large") sizing**: `d_model=1024`, 8 encoder + 6 decoder layers,
  16 heads, ~200M parameters — sized to comfortably fit one 23GB A10 per DDP
  rank with room for optimizer state and activations, while being "bigger and
  more precise" than a minimal baseline. Tune in `PredictorConfig`
  (`src/price_predictor.py`) if you want to go bigger with more VRAM/GPUs.

---

## 4. Training (on Snowflake, 4x A10 GPUs)

```bash
# inside a Snowflake Notebook terminal / cell, from the repo root:
bash train_distributed.sh
```

This runs `torchrun --standalone --nproc_per_node=4 -m src.online_trainer
--config large ...`:

- Splits the full 2020→today timeline into 4 contiguous shards, one per GPU.
- Each GPU streams through its shard **second by second, once, in
  chronological order** — no epochs, no shuffling across the full dataset.
- A small per-GPU replay buffer smooths out pure-online-SGD noise (standard
  practice for stable streaming training) while still being 100% "replay
  only what's already happened" — never future data.
- Gradients are synced across all 4 GPUs every step via DDP.
- Checkpoints (EMA weights) are written continuously to `checkpoints/`.
- Fully offline: only reads `data/raw_parquet/*.parquet` already in the repo.

Tune via env vars, e.g.:
```bash
NUM_GPUS=4 BATCH_SIZE=64 LR=1e-4 CHECKPOINT_DIR=checkpoints/run2 bash train_distributed.sh
```

For a literal "never stops, just like production" run, add `--loop` via
`EXTRA_ARGS="--loop"`.

### Local / CPU smoke test (small config, tiny data slice)
```bash
python -m src.online_trainer --config small_cpu_debug --debug_months 1 \
    --debug_limit_seconds 50000 --context 3600 --horizon 60 \
    --batch_size 4 --warmup_seconds 20 --max_steps 50
```

---

## 5. Visualization

### Local / sandbox (saves an animated GIF, 1 fps)
```bash
python run_demo.py --debug_months 3 --train_steps 300 --demo_seconds 600 \
    --out assets/demo.gif
# or, with a real trained checkpoint:
python run_demo.py --checkpoint checkpoints/ema_final.pt --demo_seconds 3600
```

### Snowflake Notebook (live, inline, 1 fps)
Open `notebooks/snowflake_live_demo.ipynb` (see setup below). It:
1. Loads the full dataset from `data/raw_parquet/` (offline).
2. Optionally launches `train_distributed.sh` right from a cell.
3. Picks a random day anywhere in 2020–2026.
4. Replays it through the simulator at 1 fps, updating an inline chart each
   second: **actual price (solid)**, **predicted price +25min (dotted)**, and
   a **rolling accuracy bar** scored against the real stored data for that
   timestamp (never the simulated feed — exactly as specified).

---

## 6. Running this on Snowflake (offline-by-design)

Snowflake Notebooks here have no general internet access, so the dataset is
**pre-baked into this Git repo** rather than fetched at runtime:

1. In Snowsight: **Projects → Notebooks → create from Git Repository** (or
   `CREATE GIT REPOSITORY` in SQL) pointing at this GitHub repo. Snowflake's
   Git integration clones it through Snowflake's own infrastructure — this
   does **not** require the notebook kernel itself to have internet access.
2. Create/select a GPU compute pool matching your 4x A10 (23GB), 48 vCPU,
   100GB RAM instance family, and attach the notebook to it.
3. In a notebook cell, install Python packages (the *only* network access
   needed, and only once):
   ```python
   !pip install -r requirements.txt
   !pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
4. If any checkpoint files were split (see §1.5), reassemble them:
   ```python
   !python scripts/split_large_file.py join checkpoints/ema_final.pt
   ```
5. Run `train_distributed.sh` (optional) and then
   `notebooks/snowflake_live_demo.ipynb` for the live chart.

No other part of this project ever calls out to the internet.

---

## 7. Repo / credentials hygiene

- The GitHub PAT used to push this repo is **not stored in any committed
  file** — it was used transiently on the command line to push, per your
  instruction that it's a disposable test-account token. Even for a
  throwaway account, it's good practice to rotate/revoke it once you're done,
  since it has now passed through chat history.
- `data/raw_parquet/*.parquet` and `checkpoints/*` are the only large,
  binary-ish artifacts tracked by git; everything else is plain source.

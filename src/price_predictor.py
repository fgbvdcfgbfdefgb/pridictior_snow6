"""
price_predictor.py
-------------------
The "Price Predictor" component (runs on GPU, trained with DistributedDataParallel
across multiple GPUs).

Input (per second t):
  - raw_window:     last 12h of raw OHLCV ticks  -> shape [B, 43200, 5]
  - feature_window:  Market Analyser features over the same 12h -> [B, 43200, N_FEATURES]
Output:
  - predicted price path for the next 25 minutes (1500 seconds), expressed as
    cumulative log-returns from the current price, so the prediction is
    anchored at today's price and physically cannot "teleport".

Architecture ("bigger model" sizing, tuned to comfortably fit a 23GB A10 with
room for optimizer state + activations, while being trainable across 4 GPUs
via DDP):

  Stage 1 - Multi-resolution causal dilated-conv encoder (WaveNet/TCN style).
            Raw 12h @ 1Hz is too long for full attention, so we aggressively
            downsample with strided causal convolutions:
                43200 -> 5400 -> 675 -> ~85 tokens
            This preserves recent high-resolution detail while still seeing
            the full 12h context.
  Stage 2 - Transformer encoder (causal) over the ~85 pooled tokens, large
            hidden size, learns cross-horizon interactions.
  Stage 3 - Autoregressive-free prediction head: a small transformer decoder
            with 1500 learned positional queries (one per future second)
            cross-attends to the encoder memory and regresses log-returns.
            A smoothness-inducing parameterization (cumulative sum of a
            tanh-bounded, temperature-scaled delta) keeps the output curve
            stable rather than jagged.

Model size is configurable via PredictorConfig; the default ("large") is
sized for 4x A10 23GB GPUs with distributed data parallel training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from .market_analyser import N_FEATURES as _N_ANALYSER_FEATURES
except ImportError:  # allow `python src/price_predictor.py` direct execution
    from market_analyser import N_FEATURES as _N_ANALYSER_FEATURES

HORIZON_SECONDS = 25 * 60       # predict 25 minutes ahead
CONTEXT_SECONDS = 12 * 60 * 60  # consume last 12 hours


@dataclass
class PredictorConfig:
    n_raw_features: int = 5          # open, high, low, close, volume (normalized)
    n_analyser_features: int = _N_ANALYSER_FEATURES  # kept in sync with market_analyser.N_FEATURES
    horizon: int = HORIZON_SECONDS
    context: int = CONTEXT_SECONDS

    conv_channels: tuple = (64, 256, 512)   # progressively widened
    conv_strides: tuple = (8, 8, 8)         # 43200 -> 5400 -> 675 -> ~85
    d_model: int = 1024
    n_encoder_layers: int = 8
    n_decoder_layers: int = 6
    n_heads: int = 16
    ffn_dim: int = 4096
    dropout: float = 0.1
    max_delta_per_sec: float = 0.0008   # cap instantaneous log-return change (stability)

    @classmethod
    def small_cpu_debug(cls) -> "PredictorConfig":
        """Tiny config for fast CPU smoke-tests / the sandbox demo."""
        return cls(
            conv_channels=(16, 32, 64),
            conv_strides=(8, 8, 8),
            d_model=64,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_heads=4,
            ffn_dim=128,
            dropout=0.0,
        )


class CausalConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride, kernel_size=9):
        super().__init__()
        pad = kernel_size - 1
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, stride=stride, padding=pad)
        self.norm = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.GELU()
        self.trim = pad // stride  # rough causal trim to drop look-ahead leakage from padding

    def forward(self, x):  # x: [B, C, T]
        y = self.conv(x)
        if self.trim > 0 and y.shape[-1] > self.trim:
            y = y[..., : -self.trim] if self.trim < y.shape[-1] else y
        return self.act(self.norm(y))


class MultiResEncoder(nn.Module):
    """Stage 1: downsamples the raw 12h window into a short token sequence."""

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        in_ch = cfg.n_raw_features + cfg.n_analyser_features
        chans = [in_ch] + list(cfg.conv_channels)
        blocks = []
        for i, stride in enumerate(cfg.conv_strides):
            blocks.append(CausalConvBlock(chans[i], chans[i + 1], stride))
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Linear(cfg.conv_channels[-1], cfg.d_model)

    def forward(self, x):  # x: [B, T, in_ch]
        h = x.transpose(1, 2)  # [B, in_ch, T]
        for b in self.blocks:
            h = b(h)
        h = h.transpose(1, 2)  # [B, T', C]
        return self.proj(h)    # [B, T', d_model]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PricePredictor(nn.Module):
    def __init__(self, cfg: PredictorConfig | None = None):
        super().__init__()
        self.cfg = cfg or PredictorConfig()
        c = self.cfg

        self.encoder_stem = MultiResEncoder(c)
        self.pos_enc = PositionalEncoding(c.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model, nhead=c.n_heads, dim_feedforward=c.ffn_dim,
            dropout=c.dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=c.n_encoder_layers)

        self.future_queries = nn.Parameter(torch.randn(1, c.horizon, c.d_model) * 0.02)
        self.query_pos = PositionalEncoding(c.d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=c.d_model, nhead=c.n_heads, dim_feedforward=c.ffn_dim,
            dropout=c.dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=c.n_decoder_layers)

        self.delta_head = nn.Sequential(
            nn.Linear(c.d_model, c.d_model // 2),
            nn.GELU(),
            nn.Linear(c.d_model // 2, 1),
        )
        # learnable temperature controlling max per-second move -> stability knob
        self.register_buffer("max_delta", torch.tensor(c.max_delta_per_sec))

    def forward(self, raw_window: torch.Tensor, feat_window: torch.Tensor) -> torch.Tensor:
        """
        raw_window:  [B, T_ctx, n_raw_features]  (normalized OHLCV)
        feat_window: [B, T_ctx, n_analyser_features]
        returns: predicted_log_return_path [B, horizon]
                 (cumulative log-return relative to the *last* price in raw_window)
        """
        x = torch.cat([raw_window, feat_window], dim=-1)
        tokens = self.encoder_stem(x)              # [B, T', d_model]
        tokens = self.pos_enc(tokens)
        memory = self.encoder(tokens)               # [B, T', d_model]

        B = raw_window.size(0)
        queries = self.query_pos(self.future_queries.expand(B, -1, -1))
        causal_mask = torch.triu(
            torch.ones(queries.size(1), queries.size(1), device=queries.device) * float("-inf"),
            diagonal=1,
        )
        decoded = self.decoder(queries, memory, tgt_mask=causal_mask)  # [B, horizon, d_model]

        raw_delta = self.delta_head(decoded).squeeze(-1)  # [B, horizon]
        # stability: bound each second's incremental log-return, then cumsum
        bounded_delta = torch.tanh(raw_delta) * self.max_delta
        cum_log_return = torch.cumsum(bounded_delta, dim=1)  # anchored at 0 -> current price
        return cum_log_return

    def predict_price_path(self, raw_window, feat_window, last_price: torch.Tensor) -> torch.Tensor:
        cum_log_return = self.forward(raw_window, feat_window)
        return last_price.unsqueeze(-1) * torch.exp(cum_log_return)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick CPU smoke test with the tiny debug config
    cfg = PredictorConfig.small_cpu_debug()
    cfg.horizon = 60   # shrink for the smoke test
    model = PricePredictor(cfg)
    B, T = 2, 3600
    raw = torch.randn(B, T, cfg.n_raw_features)
    feat = torch.randn(B, T, cfg.n_analyser_features)
    out = model(raw, feat)
    print("output shape:", out.shape, "params:", count_parameters(model))

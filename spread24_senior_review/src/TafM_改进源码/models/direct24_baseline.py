"""Simple signed-regression direct-24 baselines; no attention or Transformer."""
from __future__ import annotations

import torch
from torch import nn


class FutureOnlyDirect24(nn.Module):
    """Legacy future-only input family, explicitly named as such."""
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.n_features = int(n_features)
        self.future_projection = nn.Sequential(nn.Linear(24 * n_features, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 24))

    def forward(self, x_future: torch.Tensor) -> torch.Tensor:
        if x_future.ndim != 3 or x_future.shape[1:] != (24, self.n_features):
            raise ValueError("FutureOnlyDirect24 expects X_future [B,24,F]")
        return self.head(self.future_projection(x_future.reshape(x_future.shape[0], -1)))


class SameInputDirect24(nn.Module):
    """Same legal history/future sequence inputs as V2.1, with simple projections."""
    def __init__(self, n_features: int, hidden: int = 64, history_dim: int = 32):
        super().__init__()
        self.n_features = int(n_features)
        self.history_projection = nn.Sequential(nn.Linear(168 * 7, history_dim), nn.GELU())
        self.future_projection = nn.Sequential(nn.Linear(24 * n_features, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(history_dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, 24))

    def forward(self, x_hist: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        if x_hist.ndim != 3 or x_hist.shape[1:] != (168, 7):
            raise ValueError("SameInputDirect24 expects X_hist [B,168,7]")
        if x_future.ndim != 3 or x_future.shape[1:] != (24, self.n_features):
            raise ValueError("SameInputDirect24 expects X_future_selected [B,24,F]")
        h = self.history_projection(x_hist.reshape(x_hist.shape[0], -1))
        f = self.future_projection(x_future.reshape(x_future.shape[0], -1))
        return self.head(torch.cat([h, f], dim=-1))


def run_direct24_baselines(*args, **kwargs):
    from ..evaluate import run_direct24_baselines as implementation
    return implementation(*args, **kwargs)


__all__ = ["FutureOnlyDirect24", "SameInputDirect24", "run_direct24_baselines"]

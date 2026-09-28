"""Doc17 legal 168-hour temporal encoder with group-shared TimeMLP and FFT residuals."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from ..contracts import TEMPORAL_FEATURES, TEMPORAL_GROUPS


class _GroupTimeMLP(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(168, hidden), nn.GELU(), nn.Linear(hidden, 24))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shared over channels in the physical group: [B,n_features,168] -> [B,n_features,24]
        return self.net(x)


class TemporalEncoder(nn.Module):
    def __init__(self, *, selected_feature_names: Sequence[str], d_time: int = 32,
                 hidden: int = 64, fft_bins: Sequence[int] = (1, 7, 14, 21, 28),
                 fft_enabled: bool = True, future_conditioning: bool = True,
                 temporal_clip_abs: float = 10.0):
        super().__init__()
        self.selected_feature_names = tuple(selected_feature_names)
        self.hist_feature_names = tuple(TEMPORAL_FEATURES)
        if len(self.hist_feature_names) != 7:
            raise RuntimeError("temporal whitelist must remain the canonical seven channels")
        if any("actual" in name.lower() for name in self.selected_feature_names):
            raise ValueError("target-day actual feature is forbidden from temporal future conditioning")
        if "target_spread" in self.selected_feature_names:
            raise ValueError("target-day spread must never enter future conditioning")
        self.future_index = {name: i for i, name in enumerate(self.selected_feature_names)}
        self.future_condition_index: dict[int, int] = {}
        self.future_conditioning_enabled = bool(future_conditioning)
        if self.future_conditioning_enabled:
            for hist_i, name in enumerate(self.hist_feature_names):
                if hist_i == 0:
                    continue
                if name not in self.future_index:
                    raise ValueError(f"legal future forecast counterpart not selected: {name}")
                self.future_condition_index[hist_i] = self.future_index[name]
        self.group_names = tuple(TEMPORAL_GROUPS)
        self.group_indices = {g: tuple(self.hist_feature_names.index(f) for f in features)
                              for g, features in TEMPORAL_GROUPS.items()}
        self.group_mlps = nn.ModuleDict({g: _GroupTimeMLP(hidden) for g in self.group_names})
        self.fft_bins = tuple(int(b) for b in fft_bins)
        if not self.fft_bins or any(b <= 0 or b >= 84 for b in self.fft_bins):
            raise ValueError("FFT residual bins must be legal positive rFFT bins below Nyquist")
        self.rho = nn.Parameter(torch.zeros(len(self.group_names)))
        self.future_eta = nn.Parameter(torch.full((7,), 0.01))
        self.fft_enabled = bool(fft_enabled)
        if temporal_clip_abs <= 0:
            raise ValueError("temporal_clip_abs must be positive")
        self.temporal_clip_abs = float(temporal_clip_abs)
        self.project = nn.Sequential(nn.Linear(7, d_time), nn.GELU())
        self.d_time = int(d_time)

    def _fft_forecast(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier coefficients are computed only from the legal history tensor.
        coeff = torch.fft.rfft(x, dim=-1)
        times = torch.arange(168, 192, device=x.device, dtype=x.dtype)
        residual = torch.zeros((*x.shape[:-1], 24), device=x.device, dtype=x.dtype)
        for freq in self.fft_bins:
            c = coeff[..., freq]
            angle = (2.0 * math.pi * float(freq) / 168.0) * times
            residual = residual + (2.0 / 168.0) * (
                c.real.unsqueeze(-1) * torch.cos(angle) - c.imag.unsqueeze(-1) * torch.sin(angle)
            )
        return residual / math.sqrt(len(self.fft_bins))

    def forward(self, x_hist: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        if x_hist.ndim != 3 or x_hist.shape[1:] != (168, 7):
            raise ValueError("x_hist must be [B,168,7]")
        if x_future.ndim != 3 or x_future.shape[0] != x_hist.shape[0] or x_future.shape[1] != 24:
            raise ValueError("x_future must be [B,24,F_selected]")
        if x_future.shape[2] != len(self.selected_feature_names):
            raise ValueError("x_future selected feature order mismatch")
        if not torch.isfinite(x_hist).all() or not torch.isfinite(x_future).all():
            raise FloatingPointError("temporal encoder input contains NaN/inf")
        if x_hist.abs().max() > self.temporal_clip_abs + 1e-6 or x_future.abs().max() > self.temporal_clip_abs + 1e-6:
            raise ValueError("temporal encoder inputs must be clipped after robust scaling")
        fft = self._fft_forecast(x_hist.transpose(1, 2)) if self.fft_enabled else torch.zeros(
            (x_hist.shape[0], 7, 24), device=x_hist.device, dtype=x_hist.dtype)
        evidence = x_hist.new_zeros((x_hist.shape[0], 7, 24))
        for group_i, group_name in enumerate(self.group_names):
            ids = self.group_indices[group_name]
            h = x_hist[:, :, list(ids)].transpose(1, 2)
            learned = self.group_mlps[group_name](h)
            residual = fft[:, list(ids), :]
            evidence[:, list(ids), :] = learned + self.rho[group_i] * residual
        for hist_i, future_i in self.future_condition_index.items():
            legal_forecast = x_future[:, :, future_i]
            evidence[:, hist_i, :] = evidence[:, hist_i, :] + self.future_eta[hist_i] * legal_forecast
        h_time = self.project(evidence.transpose(1, 2))
        if h_time.shape != (x_hist.shape[0], 24, self.d_time) or not torch.isfinite(h_time).all():
            raise FloatingPointError("TemporalEncoder output shape/non-finite failure")
        return h_time

    def conditioning_audit(self) -> dict[str, object]:
        return {
            "history_channels": list(self.hist_feature_names),
            "future_conditioning": {self.hist_feature_names[i]: self.selected_feature_names[j]
                                    for i, j in self.future_condition_index.items()},
            "target_spread_future_conditioning": False,
            "target_day_actual_conditioning": False,
            "fft_enabled": self.fft_enabled,
            "future_conditioning_enabled": self.future_conditioning_enabled,
            "fft_bins": list(self.fft_bins),
            "group_indices": {k: list(v) for k, v in self.group_indices.items()},
        }

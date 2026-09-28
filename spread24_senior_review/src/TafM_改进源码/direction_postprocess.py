"""Experiment-only Direction probability post-processing for E4-A.

Low-capacity logistic stackers fitted only on the E4-A CALIBRATOR (the newest 90 days of the canonical
FULL_MONITOR, disjoint from the days that selected the deep checkpoint).
No target-day labels, no threshold tuning, no selector/preprocessing changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

POSTPROCESS_MODES = {"none", "segment_logit", "regime_logit"}
REGIME_FEATURES = (
    "residual_load_renew",
    "renewable_share",
    "bidding_space_ratio",
    "net_ramp_pressure",
    "err_net_load_28d_std",
    "uncert_风电总加_width",
    "uncert_光伏总加_width",
    "ctx_spread_positive_rate14",
    "spread_same_slot_28d_positive_rate",
)


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def _segment_terms(n_days: int) -> tuple[np.ndarray, np.ndarray]:
    hours = np.tile(np.arange(24, dtype=np.int64), int(n_days))
    h2 = ((hours >= 8) & (hours < 16)).astype(np.float64)
    h3 = (hours >= 16).astype(np.float64)
    return h2, h3


def build_meta_features(prob: np.ndarray, x_future: np.ndarray, feature_names: list[str] | tuple[str, ...],
                        mode: str) -> tuple[np.ndarray, list[str]]:
    if mode not in POSTPROCESS_MODES - {"none"}:
        raise ValueError("postprocess mode must be segment_logit or regime_logit")
    prob = np.asarray(prob, dtype=np.float64)
    x_future = np.asarray(x_future, dtype=np.float64)
    if prob.ndim != 2 or prob.shape[1] != 24:
        raise ValueError("prob must be [N,24]")
    if x_future.ndim != 3 or x_future.shape[:2] != prob.shape:
        raise ValueError("x_future must be [N,24,F] aligned to prob")
    n_days = prob.shape[0]
    logit = _safe_logit(prob).reshape(-1)
    h2, h3 = _segment_terms(n_days)
    cols = [logit, h2, h3, logit * h2, logit * h3]
    names = ["base_logit", "H2", "H3", "base_logit_x_H2", "base_logit_x_H3"]
    if mode == "regime_logit":
        index = {str(n): i for i, n in enumerate(feature_names)}
        missing = [f for f in REGIME_FEATURES if f not in index]
        if missing:
            raise ValueError(f"regime postprocessor missing required features: {missing}")
        for f in REGIME_FEATURES:
            cols.append(x_future[:, :, index[f]].reshape(-1))
            names.append(f)
    X = np.column_stack(cols).astype(np.float64)
    if not np.isfinite(X).all():
        raise FloatingPointError("postprocessor meta features contain NaN/inf")
    return X, names


@dataclass
class LogisticStacker:
    mode: str
    feature_names: list[str]
    weight: np.ndarray
    bias: float
    l2: float
    iterations: int
    monitor_positive_rate: float
    monitor_base_raw: float
    monitor_post_raw: float

    def predict_proba(self, prob: np.ndarray, x_future: np.ndarray,
                      base_feature_names: list[str] | tuple[str, ...]) -> np.ndarray:
        X, names = build_meta_features(prob, x_future, base_feature_names, self.mode)
        if names != self.feature_names:
            raise ValueError("postprocessor feature order mismatch")
        z = X @ self.weight + float(self.bias)
        z = np.clip(z, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-z))).reshape(prob.shape)

    def audit(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "feature_names": list(self.feature_names),
            "weight": [float(x) for x in self.weight],
            "bias": float(self.bias),
            "l2": float(self.l2),
            "iterations": int(self.iterations),
            "monitor_positive_rate": float(self.monitor_positive_rate),
            "monitor_base_raw": float(self.monitor_base_raw),
            "monitor_post_raw": float(self.monitor_post_raw),
        }


def fit_logistic_stacker(prob: np.ndarray, x_future: np.ndarray, y_model: np.ndarray,
                         feature_names: list[str] | tuple[str, ...], mode: str,
                         *, l2: float = 1e-3, max_iter: int = 100) -> LogisticStacker:
    if mode not in POSTPROCESS_MODES - {"none"}:
        raise ValueError("fit_logistic_stacker requires segment_logit or regime_logit")
    prob = np.asarray(prob, dtype=np.float64)
    y = (np.asarray(y_model, dtype=np.float64) > 0).astype(np.float64)
    if y.shape != prob.shape:
        raise ValueError("y_model and prob must share [N,24] shape")
    if prob.shape[0] < 20:
        raise ValueError("postprocessor requires at least 20 monitor days")
    X, names = build_meta_features(prob, x_future, feature_names, mode)
    yt = torch.as_tensor(y.reshape(-1), dtype=torch.float64)
    Xt = torch.as_tensor(X, dtype=torch.float64)
    w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    # Identity-like initialization: preserve the base logit before fitting.
    with torch.no_grad():
        w[0] = 1.0
    prevalence = float(np.clip(yt.mean().item(), 1e-5, 1 - 1e-5))
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=int(max_iter), tolerance_grad=1e-10,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    calls = {"n": 0}

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = Xt.mv(w) + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + float(l2) * torch.sum(w[1:] ** 2)
        loss.backward()
        calls["n"] += 1
        return loss

    opt.step(closure)
    weight = w.detach().cpu().numpy().astype(np.float64)
    bias = float(b.detach().cpu())
    z = X @ weight + bias
    post = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    base_raw = float(np.mean((prob.reshape(-1) >= .5) == (y.reshape(-1) > 0)))
    post_raw = float(np.mean((post >= .5) == (y.reshape(-1) > 0)))
    return LogisticStacker(
        mode=mode,
        feature_names=names,
        weight=weight,
        bias=bias,
        l2=float(l2),
        iterations=int(calls["n"]),
        monitor_positive_rate=prevalence,
        monitor_base_raw=base_raw,
        monitor_post_raw=post_raw,
    )

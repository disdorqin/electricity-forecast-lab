"""Canonical direction/magnitude metrics with strict zero policy and deterministic AUC."""
from __future__ import annotations

import numpy as np


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    """Mann-Whitney AUC using average ranks for ties; returns None if a class is absent."""
    y = np.asarray(y_true, dtype=bool)
    if not y.any() or y.all():
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and sorted_score[j] == sorted_score[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    rank_sum = ranks[y].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _direction_summary(truth: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    pos, neg = truth, ~truth
    pos_recall = float(np.mean(pred[pos] == truth[pos])) if pos.any() else None
    neg_recall = float(np.mean(pred[neg] == truth[neg])) if neg.any() else None
    balanced = float(np.mean([pos_recall, neg_recall])) if pos_recall is not None and neg_recall is not None else None
    return {
        "raw_direction_accuracy": float(np.mean(pred == truth)),
        "balanced_accuracy": balanced,
        "positive_recall": pos_recall,
        "nonpositive_recall": neg_recall,
        "+Recall": pos_recall,
        "-Recall": neg_recall,
    }


def _group_metrics(values, truth, pred, abs_error):
    vals = np.asarray(values).reshape(-1)
    return {str(k): {**_direction_summary(truth[vals == k], pred[vals == k]),
                     "magnitude_mae": float(abs_error[vals == k].mean())}
            for k in np.unique(vals)}


def canonical_metrics(y_true, direction_hat, magnitude_hat, *, direction_probability=None,
                      month=None, hour=None, tail_quantile=0.9, allow_missing_probability=False):
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred_dir = np.asarray(direction_hat, dtype=bool).reshape(-1)
    mag = np.asarray(magnitude_hat, dtype=float).reshape(-1)
    if not (len(y) == len(pred_dir) == len(mag)) or len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(mag).all():
        raise ValueError("metric inputs must be non-empty, finite and shape-aligned")
    if not (0.0 < tail_quantile < 1.0):
        raise ValueError("tail_quantile must be in (0,1)")
    truth = y > 0  # Exact-zero => non-positive; never use >= 0.
    out = _direction_summary(truth, pred_dir)
    abs_error = np.abs(mag - np.abs(y))
    tail = np.abs(y) >= np.quantile(np.abs(y), tail_quantile)
    out.update({
        "magnitude_mae": float(abs_error.mean()),
        "magnitude_rmse": float(np.sqrt(np.mean((mag - np.abs(y)) ** 2))),
        "magnitude_tail_mae": float(abs_error[tail].mean()),
        "zero_policy": "positive iff Y>0; zero is non-positive",
    })
    if direction_probability is None:
        if truth.any() and (~truth).any() and not allow_missing_probability:
            raise ValueError("direction_probability is required for AUC/Brier when both classes are present")
        out["auc"] = None
        out["auc_status"] = "N/A_NO_PROBABILITY" if allow_missing_probability else "UNDEFINED_SINGLE_CLASS"
        out["brier"] = None
    else:
        probability = np.asarray(direction_probability, dtype=float).reshape(-1)
        if len(probability) != len(y) or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("direction_probability must be finite, aligned and within [0,1]")
        auc = _binary_auc(truth, probability)
        out["auc"] = auc
        out["auc_status"] = "PASS" if auc is not None else "UNDEFINED_SINGLE_CLASS"
        out["brier"] = float(np.mean((probability - truth.astype(float)) ** 2))
    out["magnitude_by_true_sign"] = {
        "positive": {"count": int(truth.sum()), "mae": float(abs_error[truth].mean()) if truth.any() else None},
        "nonpositive": {"count": int((~truth).sum()), "mae": float(abs_error[~truth].mean()) if (~truth).any() else None},
    }
    for key, values in (("month", month), ("hour", hour)):
        if values is not None:
            vals = np.asarray(values).reshape(-1)
            if len(vals) != len(y):
                raise ValueError(f"{key} must align with observations")
            out[f"direction_by_{key}"] = _group_metrics(vals, truth, pred_dir, abs_error)
            out[f"magnitude_by_{key}"] = {str(k): float(abs_error[vals == k].mean()) for k in np.unique(vals)}
    return out

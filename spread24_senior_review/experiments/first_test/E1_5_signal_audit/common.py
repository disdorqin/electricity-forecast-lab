"""Shared constants, paths and metric helpers for the E1.5-A Direction Signal Audit.

This package is strictly read-only over every frozen artifact. It reads the frozen
sequence asset, the frozen selector manifest and the existing E1_mini run outputs,
and writes only inside experiments/first_test/E1_5_signal_audit/.

Metric semantics are copied verbatim from src/TafM_改进源码/metrics.py
(`_direction_summary`, `_binary_auc`) so E1.5-A numbers are directly comparable to
the E1 numbers already reported.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.TafM_改进源码.metrics import _binary_auc  # noqa: E402  (canonical AUC)

AUDIT_DIR = PROJECT_ROOT / "experiments" / "first_test" / "E1_5_signal_audit"
E1_DIR = PROJECT_ROOT / "experiments" / "first_test" / "E1_mini"
E1_RUNS = E1_DIR / "runs"
PRED_DIR = AUDIT_DIR / "predictions"
FIG_DIR = AUDIT_DIR / "figures"

# --- Frozen E1 evaluation scope (byte-identical to run_e1_mini.py WINDOWS) ---
WINDOWS: dict[str, tuple[str, str]] = {
    "W1": ("2026-02-12", "2026-02-18"),
    "W2": ("2026-04-12", "2026-04-18"),
    "W3": ("2026-06-12", "2026-06-18"),
    "W4": ("2026-08-07", "2026-08-13"),
}


def _window_days(start: str, end: str) -> list[date]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    return [s + timedelta(days=i) for i in range((e - s).days + 1)]


TARGET_DAYS: list[date] = [d for s, e in WINDOWS.values() for d in _window_days(s, e)]
DAY_TO_WINDOW: dict[date, str] = {d: w for w, (s, e) in WINDOWS.items() for d in _window_days(s, e)}

HOUR_SEGMENTS: dict[str, tuple[int, int]] = {"H1": (1, 8), "H2": (9, 16), "H3": (17, 24)}
HOUR_TO_SEGMENT: dict[int, str] = {
    h: name for name, (lo, hi) in HOUR_SEGMENTS.items() for h in range(lo, hi + 1)
}

# --- Model registry ---------------------------------------------------------
TAFM_MODELS: dict[str, str] = {"T0": "M0", "T1": "M1", "T2": "M2"}
TAFM_DESCRIPTION: dict[str, str] = {
    "T0": "M0 joint_v21 + v21_guardrail",
    "T1": "M1 joint_v21 + direction_first",
    "T2": "M2 dir_only + direction_first",
}
BASELINE_MODELS: list[str] = [
    "B0_AlwaysNonPositive",
    "B1_HourMajority180",
    "B2_XGB_DIR_180",
    "B3_XGB_DIR_EXPANDING",
    "B4_StrictLightGBM_DIR_180",
]
BASELINE_DESCRIPTION: dict[str, str] = {
    "B0_AlwaysNonPositive": "always predict non-positive (class-prior sanity baseline)",
    "B1_HourMajority180": "per-hour majority direction over legal [D-181, D-2] labels",
    "B2_XGB_DIR_180": "frozen direction-benchmark XGBoost, recent legal 180d",
    "B3_XGB_DIR_EXPANDING": "identical to B2 but all legal history S<=D-2",
    "B4_StrictLightGBM_DIR_180": "run_strict_lightgbm_baseline(window_days=180), sign only",
}
ALL_MODELS: list[str] = ["T0", "T1", "T2"] + BASELINE_MODELS

SEED = 20260924
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_RESAMPLES = 10000
TIE_EPS = 1e-12

# B1 window, exactly as written in the plan: legal labels in [D-181, D-2].
B1_WINDOW_DAYS = 181
# B2 window, exactly as implemented in pipeline.run_direction_benchmark:
#   store.days[i] >= day - timedelta(days=180)
B2_WINDOW_DAYS = 180


# --- Metrics (verbatim semantics from src/TafM_改进源码/metrics.py) ----------
def direction_metrics(
    y_true: np.ndarray,
    direction_hat: np.ndarray,
    p_positive: np.ndarray | None = None,
    rank_score: np.ndarray | None = None,
) -> dict[str, object]:
    """Direction metrics with the project's strict zero policy (positive iff Y > 0)."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(direction_hat).reshape(-1).astype(bool)
    if len(y) == 0:
        raise ValueError("direction_metrics requires at least one observation")
    if len(y) != len(pred):
        raise ValueError("direction_metrics inputs must be shape-aligned")
    if not np.isfinite(y).all():
        raise ValueError("direction_metrics requires finite truth")

    truth = y > 0  # Exact-zero => non-positive. Never >= 0.
    pos, neg = truth, ~truth
    pos_recall = float(np.mean(pred[pos] == truth[pos])) if pos.any() else None
    neg_recall = float(np.mean(pred[neg] == truth[neg])) if neg.any() else None
    balanced = (
        float(np.mean([pos_recall, neg_recall]))
        if pos_recall is not None and neg_recall is not None
        else None
    )

    out: dict[str, object] = {
        "raw_direction_accuracy": float(np.mean(pred == truth)),
        "balanced_accuracy": balanced,
        "positive_recall": pos_recall,
        "nonpositive_recall": neg_recall,
        "predicted_positive_fraction": float(pred.mean()),
        "true_positive_prevalence": float(truth.mean()),
        "n_slots": int(len(y)),
        "n_positive": int(pos.sum()),
        "n_nonpositive": int(neg.sum()),
    }

    if p_positive is not None:
        p = np.asarray(p_positive, dtype=float).reshape(-1)
        if len(p) != len(y) or not np.isfinite(p).all():
            raise ValueError("p_positive must be finite and shape-aligned")
        auc = _binary_auc(truth, p)
        out["auc"] = auc
        out["auc_status"] = "PASS" if auc is not None else "UNDEFINED_SINGLE_CLASS"
        out["brier"] = float(np.mean((p - truth.astype(float)) ** 2))
    elif rank_score is not None:
        # Regression-derived sign baselines expose a ranking score, not a probability.
        # Report rank-AUC and refuse to fabricate a Brier score.
        r = np.asarray(rank_score, dtype=float).reshape(-1)
        if len(r) != len(y) or not np.isfinite(r).all():
            raise ValueError("rank_score must be finite and shape-aligned")
        auc = _binary_auc(truth, r)
        out["auc"] = auc
        out["auc_status"] = "RANK_ONLY_NOT_A_PROBABILITY_METRIC"
        out["brier"] = None
    else:
        out["auc"] = None
        out["brier"] = None
        out["auc_status"] = "N/A_NO_PROBABILITY"
    return out


def majority_baseline_raw(y_true: np.ndarray) -> float:
    """Raw accuracy of always predicting the window's majority class."""
    truth = np.asarray(y_true, dtype=float).reshape(-1) > 0
    if len(truth) == 0:
        return float("nan")
    prevalence = float(truth.mean())
    return max(prevalence, 1.0 - prevalence)


def day_cluster_bootstrap(
    deltas: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Resample target days (not slots) with replacement; 24 hourly slots are not independent."""
    d = np.asarray(deltas, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "ci_low": lo,
        "ci_high": hi,
        "ci_includes_zero": bool(lo <= 0.0 <= hi),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "bootstrap_unit": "target_day",
    }


# --- TafM predictions (read-only over E1 artifacts) -------------------------
def load_tafm_predictions() -> pd.DataFrame:
    """Read per-slot TafM predictions straight out of the completed E1 runs.

    Never retrains, never re-selects a checkpoint, never writes into E1_mini.
    """
    frames = []
    for label, variant in TAFM_MODELS.items():
        for day in TARGET_DAYS:
            run_dir = E1_RUNS / f"E1-{variant}-{day.isoformat()}"
            pointer = run_dir / "RAW_RUN_PATH.txt"
            if not pointer.is_file():
                raise FileNotFoundError(f"missing E1 raw run pointer: {pointer}")
            raw_dir = Path(pointer.read_text(encoding="utf-8").strip())
            parquet = raw_dir / "predictions.parquet"
            if not parquet.is_file():
                raise FileNotFoundError(f"missing E1 predictions: {parquet}")
            frame = pd.read_parquet(parquet)
            if len(frame) != 24:
                raise ValueError(f"{parquet} has {len(frame)} slots, expected 24")
            frames.append(
                pd.DataFrame(
                    {
                        "model": label,
                        "target_day": day,
                        "hour_business": frame["hour_business"].to_numpy(dtype=int),
                        "y_true_model": frame["y_true_model"].to_numpy(dtype=float),
                        "direction_hat": frame["direction_hat"].to_numpy(dtype=float) > 0.5,
                        "p_positive": frame["p_positive"].to_numpy(dtype=float),
                        "source": "E1_mini/predictions.parquet",
                    }
                )
            )
    panel = pd.concat(frames, ignore_index=True)
    panel["window"] = panel["target_day"].map(DAY_TO_WINDOW)
    return panel


# --- Frozen inputs (sequence asset + selector) ------------------------------
def load_store_and_selector():
    """Load the frozen sequence asset and the FROZEN selector manifest."""
    from src.TafM_改进源码.dataset import SequenceStore
    from src.TafM_改进源码.selector import load_selector_manifest

    store = SequenceStore.load()
    selector, selector_hash = load_selector_manifest()
    if selector.get("status") != "FROZEN":
        raise ValueError("E1.5-A requires the frozen selector; refusing to run otherwise")
    return store, selector, selector_hash


def legal_indices(store, selector, target_day: date, *, window_days: int | None) -> list[int]:
    """Legal training indices for `target_day`: S <= D-2, quarantine-clean, optional window."""
    eligible = store.eligibility(selector, current_target_day=target_day)["eligible_indices"]
    if window_days is None:
        return list(eligible)
    earliest = target_day - timedelta(days=window_days)
    days = store.days
    return [i for i in eligible if days[i] >= earliest]

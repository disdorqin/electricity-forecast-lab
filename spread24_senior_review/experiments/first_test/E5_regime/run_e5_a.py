"""Chronological, no-model-change E5-A regime diagnosis."""
from __future__ import annotations

import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                             recall_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.config import default_selector_path
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.direction_postprocess import REGIME_FEATURES
from src.TafM_改进源码.source_resolver import sha256_file

SEED = 20260924
SEEDS = (20260924, 20260925, 20260926)
WINDOWS = {
    "W1": ("2026-02-12", "2026-02-18"),
    "W2": ("2026-04-12", "2026-04-18"),
    "W3": ("2026-06-12", "2026-06-18"),
    "W4": ("2026-08-07", "2026-08-13"),
}
METHODS = (("KMeans", 3), ("KMeans", 5), ("KMeans", 7),
           ("GMM", 3), ("GMM", 5))


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    return {
        "raw": float(np.mean(y == pred)),
        "balanced": float(balanced_accuracy_score(y, pred)),
        "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "nonpositive_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
    }


def _fit_cluster(kind: str, k: int, seed: int, x_train: np.ndarray,
                 x_eval: np.ndarray):
    scaler = StandardScaler().fit(x_train)
    tr = scaler.transform(x_train)
    ev = scaler.transform(x_eval)
    if kind == "KMeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=20, algorithm="lloyd")
        labels_train = model.fit_predict(tr)
        labels_eval = model.predict(ev)
    else:
        model = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-6,
                                random_state=seed, n_init=1, max_iter=200, init_params="kmeans")
        model.fit(tr)
        labels_train = model.predict(tr)
        labels_eval = model.predict(ev)
    return labels_train.astype(int), labels_eval.astype(int), model


def _conditioned_predictions(train_clusters: np.ndarray, eval_clusters: np.ndarray,
                             y_train: np.ndarray):
    """Use train-only cluster x hour majority and median; global hour fallback."""
    train_y = (y_train > 0).astype(np.int8)
    train_mag = np.abs(y_train)
    global_sign = np.zeros(24, dtype=np.int8)
    global_mag = np.zeros(24, dtype=np.float64)
    for h in range(24):
        global_sign[h] = int(np.mean(train_y[:, h]) > 0.5)
        global_mag[h] = float(np.median(train_mag[:, h]))
    sign_table, mag_table, support = {}, {}, {}
    for c in np.unique(train_clusters):
        mask = train_clusters == c
        for h in range(24):
            sign_table[(int(c), h)] = int(np.mean(train_y[mask, h]) > 0.5)
            mag_table[(int(c), h)] = float(np.median(train_mag[mask, h]))
            support[(int(c), h)] = int(mask.sum())

    n_days = len(eval_clusters)
    pred_sign = np.empty((n_days, 24), dtype=np.int8)
    pred_mag = np.empty((n_days, 24), dtype=np.float64)
    for d, c in enumerate(eval_clusters):
        for h in range(24):
            if (int(c), h) in sign_table:
                pred_sign[d, h] = sign_table[(int(c), h)]
                pred_mag[d, h] = mag_table[(int(c), h)]
            else:
                pred_sign[d, h] = global_sign[h]
                pred_mag[d, h] = global_mag[h]
    base_sign = np.broadcast_to(global_sign, (n_days, 24)).copy()
    base_mag = np.broadcast_to(global_mag, (n_days, 24)).copy()
    return pred_sign, pred_mag, base_sign, base_mag, support


def _paired_day_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                          *, seed: int = SEED, draws: int = 10_000):
    day_delta = np.mean(y == pred_a, axis=1) - np.mean(y == pred_b, axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(day_delta), size=(draws, len(day_delta)))
    samples = day_delta[idx].mean(axis=1)
    return {"delta_raw": float(day_delta.mean()),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "better_days": int(np.sum(day_delta > 0)),
            "tie_days": int(np.sum(day_delta == 0)),
            "worse_days": int(np.sum(day_delta < 0)),
            "day_deltas": day_delta}


def main():
    store = SequenceStore.load()
    selector, selector_hash = load_frozen_selector(default_selector_path())
    all_days = pd.to_datetime(store.day_index.target_day).dt.date.to_numpy()
    day_to_idx = {str(day): i for i, day in enumerate(all_days)}
    feature_idx = {name: store.feature_names.index(name) for name in REGIME_FEATURES}
    cols = np.asarray([feature_idx[n] for n in REGIME_FEATURES], dtype=np.int64)
    raw = np.asarray(store.x_future[:, :, cols], dtype=np.float64)
    if np.asarray(store.quarantine_mask[:, :, cols], dtype=bool).any():
        bad = np.asarray(store.quarantine_mask[:, :, cols], dtype=bool).any(axis=(1, 2))
    else:
        bad = np.zeros(len(all_days), dtype=bool)
    day_x = raw.mean(axis=1)
    day_x[bad] = np.nan
    y_model = -np.asarray(store.y_source, dtype=np.float64)
    day_strings = np.asarray([d.isoformat() for d in all_days])

    assignments, cluster_metrics, fold_metrics, stability_rows = [], [], [], []
    split_rows, fitted_meta = [], []
    fold_cache = {}
    for window, (start, end) in WINDOWS.items():
        cutoff = (pd.Timestamp(start) - pd.Timedelta(days=2)).date()
        train_mask = np.asarray([d <= cutoff for d in all_days]) & ~bad
        eval_days = [str(d.date()) for d in pd.date_range(start, end, freq="D")]
        if any(d not in day_to_idx for d in eval_days):
            raise RuntimeError(f"missing expected evaluation dates in {window}")
        eval_idx = np.asarray([day_to_idx[d] for d in eval_days], dtype=np.int64)
        # Reuse the canonical selected-feature eligibility contract for supervised history.
        eligible = set(store.eligibility(selector)["eligible_indices"])
        train_idx = np.asarray([i for i in np.flatnonzero(train_mask) if int(i) in eligible], dtype=np.int64)
        if len(train_idx) < 100 or not np.isfinite(day_x[train_idx]).all() or not np.isfinite(day_x[eval_idx]).all():
            raise RuntimeError(f"insufficient/invalid features in {window}: train={len(train_idx)}")
        if np.any(np.asarray([all_days[i] for i in train_idx]) > cutoff):
            raise RuntimeError(f"chronological training cutoff violated in {window}")
        eval_eligible = set(store.eligibility(selector, requested_days=eval_days)["eligible_indices"])
        if not set(map(int, eval_idx)).issubset(eval_eligible):
            raise RuntimeError(f"frozen panel contains ineligible days in {window}")

        x_train, x_eval = day_x[train_idx], day_x[eval_idx]
        yt_train, yt_eval = y_model[train_idx], y_model[eval_idx]
        threshold = float(np.quantile(np.abs(yt_train), 0.90))
        split_rows.append({"window": window, "eval_start": start, "eval_end": end,
                           "fit_cutoff_D_minus_2": cutoff.isoformat(),
                           "train_days": len(train_idx), "train_start": day_strings[train_idx[0]],
                           "train_end": day_strings[train_idx[-1]], "eval_days": len(eval_idx),
                           "max_train_date_le_cutoff": bool(all_days[train_idx].max() <= cutoff),
                           "eval_in_training_indices": bool(set(map(int, eval_idx)) & set(map(int, train_idx)))})
        fold_cache[window] = {"y": (yt_eval > 0).astype(np.int8), "y_model": yt_eval,
                              "preds": {}, "clusters": {}}

        for kind, k in METHODS:
            seed_clusters = {}
            for seed in SEEDS:
                ct, ce, model = _fit_cluster(kind, k, seed, x_train, x_eval)
                seed_clusters[seed] = (ct, ce)
                if seed == SEED:
                    primary_train, primary_eval = ct, ce
                    fit_meta = {"window": window, "algorithm": kind, "k": k, "seed": seed,
                                "n_iter": int(getattr(model, "n_iter_", 0)),
                                "converged": bool(getattr(model, "converged_", True)),
                                "train_inertia": float(model.inertia_) if kind == "KMeans" else None}
                    fitted_meta.append(fit_meta)
                    for i, c in zip(eval_idx, ce):
                        assignments.append({"window": window, "target_day": day_strings[i],
                                            "algorithm": kind, "k": k, "seed": seed,
                                            "cluster": int(c)})
            eval_labelings = [seed_clusters[s][1] for s in SEEDS]
            aris = [adjusted_rand_score(eval_labelings[a], eval_labelings[b])
                    for a in range(len(SEEDS)) for b in range(a + 1, len(SEEDS))]
            stability_rows.append({"window": window, "algorithm": kind, "k": k,
                                   "heldout_occupied_clusters": int(len(np.unique(primary_eval))),
                                   "mean_pairwise_seed_ari": float(np.mean(aris)),
                                   "min_pairwise_seed_ari": float(np.min(aris))})
            pred, pred_mag, base_pred, base_mag, support = _conditioned_predictions(
                primary_train, primary_eval, yt_train)
            ybin = (yt_eval > 0).astype(np.int8)
            pred_flat, y_flat = pred.reshape(-1), ybin.reshape(-1)
            base_flat = base_pred.reshape(-1)
            ext = np.abs(yt_eval) > threshold
            overall = _metrics(y_flat, pred_flat)
            baseline = _metrics(y_flat, base_flat)
            delta = _paired_day_bootstrap(ybin, pred, base_pred)
            model_key = f"{kind}_K{k}"
            if model_key == "KMeans_K5":
                fold_cache[window]["day_deltas"] = delta["day_deltas"]
            fold_cache[window]["preds"][model_key] = pred
            fold_cache[window]["clusters"][model_key] = primary_eval
            fold_metrics.append({"window": window, "algorithm": kind, "k": k,
                                 "train_days": len(train_idx), "test_days": len(eval_idx),
                                 "occupied_clusters": int(len(np.unique(primary_eval))),
                                 "positive_rate_range_pp": 100.0 * float(max(
                                     np.mean(ybin[primary_eval == c]) for c in np.unique(primary_eval)) -
                                     min(np.mean(ybin[primary_eval == c]) for c in np.unique(primary_eval))),
                                 "cluster_hour_baseline_raw": overall["raw"],
                                 "global_hour_baseline_raw": baseline["raw"],
                                 "delta_raw": delta["delta_raw"], "delta_ci95_low": delta["ci95_low"],
                                 "delta_ci95_high": delta["ci95_high"], "delta_W_T_L":
                                 f"{delta['better_days']}/{delta['tie_days']}/{delta['worse_days']}",
                                 "cluster_hour_balanced": overall["balanced"],
                                 "global_hour_balanced": baseline["balanced"],
                                 "cluster_hour_positive_recall": overall["positive_recall"],
                                 "cluster_hour_nonpositive_recall": overall["nonpositive_recall"],
                                 "cluster_hour_magnitude_mae": float(np.mean(np.abs(yt_eval - pred_mag))),
                                 "global_hour_magnitude_mae": float(np.mean(np.abs(yt_eval - base_mag))),
                                 "extreme_threshold_abs_y_train_q90": threshold,
                                 "extreme_spread_ratio": float(ext.mean())})
            # Held-out cluster summaries; labels are used only here, after assignment.
            for c in sorted(np.unique(primary_eval)):
                m = primary_eval == c
                cm = _metrics(ybin[m].reshape(-1), pred[m].reshape(-1))
                cluster_metrics.append({"window": window, "algorithm": kind, "k": k,
                                        "cluster": int(c), "days": int(m.sum()),
                                        "slots": int(m.sum() * 24),
                                        "positive_rate": float(ybin[m].mean()),
                                        "cluster_hour_direction_accuracy": cm["raw"],
                                        "cluster_hour_balanced_accuracy": cm["balanced"],
                                        "cluster_hour_positive_recall": cm["positive_recall"],
                                        "cluster_hour_nonpositive_recall": cm["nonpositive_recall"],
                                        "magnitude_mae": float(np.mean(np.abs(yt_eval[m] - pred_mag[m]))),
                                        "extreme_ratio": float(ext[m].mean()),
                                        "train_days_in_cluster": int(np.sum(primary_train == c)),
                                        "min_train_cluster_support_per_hour": int(min(
                                            support.get((int(c), h), 0) for h in range(24)))})

    fold_df = pd.DataFrame(fold_metrics)
    primary = fold_df[(fold_df.algorithm == "KMeans") & (fold_df.k == 5)].copy()
    # Paired pooled inference reuses the per-fold comparison rows with a single bootstrap over all held-out days.
    all_days_delta = []
    for w in WINDOWS:
        all_days_delta.extend(fold_cache[w]["day_deltas"])
    rng = np.random.default_rng(SEED)
    dd = np.asarray(all_days_delta, dtype=np.float64)
    boot = dd[rng.integers(0, len(dd), size=(10_000, len(dd)))].mean(axis=1)
    pooled_ci = {"delta_raw": float(dd.mean()), "ci95_low": float(np.quantile(boot, .025)),
                 "ci95_high": float(np.quantile(boot, .975)), "n_days": int(len(dd)),
                 "better_days": int(np.sum(dd > 0)), "tie_days": int(np.sum(dd == 0)),
                 "worse_days": int(np.sum(dd < 0))}

    primary_stability = pd.DataFrame(stability_rows)
    primary_stability = primary_stability[(primary_stability.algorithm == "KMeans") &
                                          (primary_stability.k == 5)]
    sign_heterogeneity = int(np.sum(primary.positive_rate_range_pp >= 10.0))
    baseline_better = int(np.sum(primary.delta_raw > 0))
    stable_structure = bool((primary_stability.heldout_occupied_clusters >= 3).all() and
                            (primary_stability.mean_pairwise_seed_ari >= .60).all())
    signal = ("REGIME_SIGNAL_PASS" if stable_structure and sign_heterogeneity >= 3 and
              baseline_better >= 3 and pooled_ci["ci95_low"] > 0 else "NO_CLEAR_REGIME_SIGNAL")

    pd.DataFrame(assignments).to_csv(HERE / "cluster_assignments.csv", index=False)
    pd.DataFrame(cluster_metrics).to_csv(HERE / "cluster_metrics.csv", index=False)
    fold_df.to_csv(HERE / "fold_metrics.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(HERE / "cluster_stability.csv", index=False)
    pd.DataFrame(split_rows).to_csv(HERE / "split_audit.csv", index=False)
    pd.DataFrame(fitted_meta).to_csv(HERE / "fit_diagnostics.csv", index=False)
    commit_text = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
    commit = commit_text if re.fullmatch(r"[0-9a-f]{40}", commit_text) else None
    report = {"gate": signal, "primary_method": "KMeans_K5", "windows": WINDOWS,
              "hdbscan_available": importlib.util.find_spec("hdbscan") is not None,
              "hdbscan_status": "OMITTED_NOT_INSTALLED", "methods": METHODS,
              "seeds": SEEDS, "feature_names": REGIME_FEATURES,
              "feature_matrix": "per-target-day mean of 24 origin-available X_future values",
              "selector_sha256": selector_hash, "source_sha256": store.source_sha256,
              "sequence_manifest_sha256": store.sequence_sha256,
              "python": platform.python_version(), "numpy": np.__version__,
              "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
              "git_commit": commit,
              "pooled_primary_paired_day_bootstrap": pooled_ci,
              "primary_windows_with_positive_rate_range_ge_10pp": sign_heterogeneity,
              "primary_windows_cluster_baseline_better": baseline_better,
              "primary_stable_structure_all_folds": stable_structure,
              "regime_pass": signal == "REGIME_SIGNAL_PASS",
              "limitations": ["screening panel only; not full Jan-Aug DEV",
                              "not a V2.2 model test", "no E5-B/E5-C authorization"]}
    (HERE / "E5_A_gate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"gate": signal, "pooled_primary": pooled_ci,
                      "windows_with_sign_heterogeneity_ge_10pp": sign_heterogeneity,
                      "windows_baseline_better": baseline_better,
                      "stable_structure_all_folds": stable_structure}, indent=2))


if __name__ == "__main__":
    main()

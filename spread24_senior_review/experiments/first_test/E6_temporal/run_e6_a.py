"""E6-A: frozen, no-training temporal baselines plus saved current TemporalEncoder output."""
from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.config import default_selector_path
from src.TafM_改进源码.contracts import TEMPORAL_FEATURES
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.source_resolver import sha256_file

E2_OBJECTIVE_ROOT = ROOT / "src" / "TafM_改进源码" / "outputs" / "tabm_v21" / "experiments" / "direction_first" / "objective_modes"
WINDOWS = {
    "W1": ("2026-02-12", "2026-02-18"),
    "W2": ("2026-04-12", "2026-04-18"),
    "W3": ("2026-06-12", "2026-06-18"),
    "W4": ("2026-08-07", "2026-08-13"),
}
SEED = 20260924


def _dates() -> list[tuple[str, str]]:
    return [(w, str(d.date())) for w, (a, b) in WINDOWS.items()
            for d in pd.date_range(a, b, freq="D")]


def _temporal_runs(expected_days: set[str], source_sha: str, sequence_sha: str,
                   selector_sha: str) -> tuple[dict[str, dict], list[dict]]:
    candidates: dict[str, list[tuple[Path, dict]]] = {d: [] for d in expected_days}
    for manifest_path in E2_OBJECTIVE_ROOT.rglob("manifest.json"):
        if "timeonly" not in manifest_path.parent.name.lower():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        day = str(manifest.get("target_day", ""))
        if day not in candidates:
            continue
        if manifest.get("status") != "COMPLETE":
            raise RuntimeError(f"E2-A TemporalEncoder record incomplete: {manifest_path}")
        if manifest.get("architecture_mode") != "temporal_only" or manifest.get("objective_mode") != "dir_only":
            raise RuntimeError(f"wrong E2-A diagnostic architecture/objective: {manifest_path}")
        for key, expected in (("source_sha256", source_sha),
                              ("sequence_manifest_sha256", sequence_sha),
                              ("selector_sha256", selector_sha)):
            if manifest.get(key) != expected:
                raise RuntimeError(f"E2-A {key} mismatch for {day}: {manifest.get(key)} != {expected}")
        if int(manifest.get("seed", -1)) != SEED:
            raise RuntimeError(f"unexpected E2-A seed for {day}: {manifest.get('seed')}")
        fit_end = manifest.get("preprocessing_fit_day_end")
        cutoff = (pd.Timestamp(day) - pd.Timedelta(days=2)).date().isoformat()
        if not fit_end or pd.Timestamp(fit_end).date().isoformat() > cutoff:
            raise RuntimeError(f"E2-A preprocessing fit violates D-2 boundary for {day}: {fit_end} > {cutoff}")
        candidates[day].append((manifest_path, manifest))

    chosen: dict[str, dict] = {}
    audit: list[dict] = []
    for day in sorted(expected_days):
        matches = sorted(candidates[day], key=lambda x: x[0].parent.name)
        if not matches:
            raise RuntimeError(f"missing saved current TemporalEncoder output for {day}")
        frames = []
        for path, manifest in matches:
            pred_path = path.parent / manifest.get("predictions_file", "predictions.parquet")
            frame = pd.read_parquet(pred_path).sort_values("hour_business").reset_index(drop=True)
            if frame.hour_business.tolist() != list(range(1, 25)):
                raise RuntimeError(f"E2-A prediction horizons malformed: {pred_path}")
            if not {"direction_hat", "y_true_model"}.issubset(frame.columns):
                raise RuntimeError(f"E2-A predictions missing required fields: {pred_path}")
            frames.append((path, manifest, frame))
        reference = frames[0][2]
        for _, _, frame in frames[1:]:
            if not np.array_equal(reference.direction_hat.to_numpy(), frame.direction_hat.to_numpy()):
                raise RuntimeError(f"duplicate E2-A prediction mismatch for {day}")
            if not np.array_equal(reference.y_true_model.to_numpy(float), frame.y_true_model.to_numpy(float)):
                raise RuntimeError(f"duplicate E2-A labels mismatch for {day}")
        # Directory names contain UTC timestamps; the latest identical rerun is selected deterministically.
        path, manifest, frame = frames[-1]
        chosen[day] = {"frame": frame, "path": path, "manifest": manifest}
        chosen_pred_path = path.parent / manifest.get("predictions_file", "predictions.parquet")
        audit.append({"target_day": day, "candidate_run_count": len(frames),
                      "chosen_run_dir": str(path.parent.relative_to(ROOT)),
                      "chosen_manifest_sha256": sha256_file(path),
                      "chosen_predictions_sha256": sha256_file(chosen_pred_path),
                      "chosen_run_id": manifest.get("run_id"),
                      "duplicate_predictions_identical": True,
                      "architecture_mode": manifest.get("architecture_mode"),
                      "objective_mode": manifest.get("objective_mode"),
                      "seed": manifest.get("seed"),
                      "config_sha256": manifest.get("config_sha256"),
                      "checkpoint_policy": manifest.get("checkpoint_policy"),
                      "device": manifest.get("device"), "amp": manifest.get("amp"),
                      "stage_a_base_train_days": manifest.get("stage_a_base_train_days"),
                      "stage_a_monitor_days": manifest.get("stage_a_monitor_days"),
                      "fit_day_end": manifest.get("preprocessing_fit_day_end"),
                      "checkpoint_raw": manifest.get("checkpoint_selected_raw")})
    return chosen, audit


def _binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    pred = np.asarray(pred, dtype=np.int8).reshape(-1)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    p_count, n_count = int(np.sum(y == 1)), int(np.sum(y == 0))
    pos_recall = tp / p_count if p_count else np.nan
    neg_recall = tn / n_count if n_count else np.nan
    balanced = (pos_recall + neg_recall) / 2 if p_count and n_count else np.nan
    return {"n_slots": int(y.size), "positive_slots": p_count, "nonpositive_slots": n_count,
            "raw_accuracy": float(np.mean(y == pred)), "balanced_accuracy": float(balanced),
            "positive_recall": float(pos_recall), "negative_recall": float(neg_recall),
            "predicted_positive_fraction": float(np.mean(pred == 1))}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    store = SequenceStore.load()
    selector, selector_sha = load_frozen_selector(default_selector_path())
    source_sha = store.source_sha256
    sequence_sha = store.sequence_sha256
    if not TEMPORAL_FEATURES or TEMPORAL_FEATURES[0] != "target_spread":
        raise RuntimeError("frozen temporal channel order no longer places target_spread at channel 0")
    spread_hist = np.asarray(store.x_hist[:, :, 0], dtype=np.float64)
    periodic_name = "spread_same_slot_28d_positive_rate"
    if periodic_name not in selector.get("selected_features", []):
        raise RuntimeError(f"periodic diagnostic input is not in the frozen selector: {periodic_name}")
    periodic_idx = store.feature_names.index(periodic_name)
    periodic_record = next(r for r in store.registry["candidate_features"]
                           if r.get("feature_name") == periodic_name)
    period_contract = periodic_record.get("dependency_contract", {})
    if (not periodic_record.get("allow_in_future") or
            int(period_contract.get("shift_days", 0)) < 2 or
            int(period_contract.get("rolling_window_days", 0)) != 28):
        raise RuntimeError("frozen 28-day periodic feature dependency contract is not the expected D-2-safe contract")
    target_labels = -np.asarray(store.y_source, dtype=np.float64)
    days_arr = pd.to_datetime(store.day_index.target_day).dt.date
    day_to_idx = {str(d): i for i, d in enumerate(days_arr)}
    expected = _dates()
    expected_days = {d for _, d in expected}
    missing_days = expected_days - set(day_to_idx)
    if missing_days:
        raise RuntimeError(f"sequence is missing E6-A frozen dates: {sorted(missing_days)}")
    eligibility = set(store.eligibility(selector, requested_days=sorted(expected_days))["eligible_indices"])
    target_indices = {d: day_to_idx[d] for d in expected_days}
    if not set(target_indices.values()).issubset(eligibility):
        raise RuntimeError("frozen evaluation panel violates selected-feature quarantine eligibility")

    # Assert the approved D-1 14:00 / 168-hour index mapping before evaluating any heuristic.
    if [(177 + h - 24 * 1) for h in (1, 14, 15, 24)] != [154, 167, 168, 177]:
        raise RuntimeError("same-hour history index contract changed")

    temporal, temporal_audit = _temporal_runs(expected_days, source_sha, sequence_sha, selector_sha)
    prediction_rows = []
    for window, day in expected:
        idx = target_indices[day]
        hist = spread_hist[idx]
        if hist.shape != (168,) or not np.isfinite(hist).all():
            raise RuntimeError(f"invalid historical spread input for {day}")
        y = target_labels[idx]
        target_frame = temporal[day]["frame"]
        # Existing prediction artifacts serialize labels as float32; tolerate that round-trip only.
        if not np.allclose(target_frame.y_true_model.to_numpy(float), y, atol=1e-4, rtol=0.0):
            raise RuntimeError(f"saved E2-A target labels do not match frozen sequence for {day}")

        for n in (3, 6, 14):
            pred_value = float(np.mean(hist[-n:]))
            pred = int(pred_value > 0.0)
            for h in range(1, 25):
                prediction_rows.append({"window": window, "target_day": day, "hour_business": h,
                                        "baseline": f"recent_{n}h", "prediction": pred,
                                        "prediction_value": pred_value, "y_model": float(y[h - 1])})

        same_hour_pred, weekly_pred = [], []
        for h in range(1, 25):
            valid = [(d, 177 + h - 24 * d) for d in range(1, 8)
                     if 0 <= 177 + h - 24 * d < 168]
            if not valid:
                raise RuntimeError(f"no legal same-hour historical value for {day} hour {h}")
            # Same-hour lag uses the latest available valid day (smallest lag-day d).
            d_latest, ix_latest = min(valid, key=lambda pair: pair[0])
            same_hour_value = float(hist[ix_latest])
            same_hour_pred.append(int(same_hour_value > 0.0))
            # Weekly periodic state uses a fixed majority over all legal same-hour lags.
            periodic_values = np.asarray([hist[ix] for _, ix in valid], dtype=float)
            periodic_signs = periodic_values > 0.0
            weekly_pred.append(int(periodic_signs.sum() > (len(periodic_signs) / 2)))
            prediction_rows.append({"window": window, "target_day": day, "hour_business": h,
                                    "baseline": "same_hour_latest_legal_lag", "prediction": same_hour_pred[-1],
                                    "prediction_value": same_hour_value, "lag_day_used": d_latest,
                                    "history_index_used": ix_latest, "y_model": float(y[h - 1])})
            prediction_rows.append({"window": window, "target_day": day, "hour_business": h,
                                    "baseline": "weekly_same_hour_majority_1to7d",
                                    "prediction": weekly_pred[-1],
                                    "prediction_value": float(periodic_signs.mean()),
                                    "available_lag_days": ",".join(str(d) for d, _ in valid),
                                    "y_model": float(y[h - 1])})

            rate = float(store.x_future[idx, h - 1, periodic_idx])
            if not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise RuntimeError(f"invalid frozen 28-day same-hour positive rate: {day} hour={h} value={rate}")
            prediction_rows.append({"window": window, "target_day": day, "hour_business": h,
                                    "baseline": "periodic_28d_same_slot_positive_rate",
                                    "prediction": int(rate > 0.5), "prediction_value": rate,
                                    "feature_name": periodic_name, "feature_shift_days": 2,
                                    "y_model": float(y[h - 1])})

        model_pred = target_frame.direction_hat.astype(int).to_numpy()
        for h in range(1, 25):
            prediction_rows.append({"window": window, "target_day": day, "hour_business": h,
                                    "baseline": "current_temporal_encoder_temporal_only",
                                    "prediction": int(model_pred[h - 1]),
                                    "prediction_value": float(target_frame.iloc[h - 1].get("p_positive", np.nan)),
                                    "source_run": temporal[day]["path"].parent.name,
                                    "y_model": float(y[h - 1])})
            prediction_rows.append({"window": window, "target_day": day, "hour_business": h,
                                    "baseline": "all_negative_reference", "prediction": 0,
                                    "prediction_value": 0.0, "y_model": float(y[h - 1])})

    predictions = pd.DataFrame(prediction_rows)
    metric_rows = []
    for baseline, frame in predictions.groupby("baseline", sort=True):
        for scope, part in [("ALL", frame), *[(w, frame[frame.window == w]) for w in WINDOWS]]:
            m = _binary_metrics(part.y_model.to_numpy() > 0, part.prediction.to_numpy())
            metric_rows.append({"baseline": baseline, "scope": scope, **m})

    metrics = pd.DataFrame(metric_rows)
    # Required panel checks: each tested baseline has 672 slots overall and 168 per window.
    if not (metrics[metrics.scope == "ALL"].n_slots == 672).all():
        raise RuntimeError("overall E6-A slot count must be 672 for every baseline")
    if not (metrics[metrics.scope != "ALL"].n_slots == 168).all():
        raise RuntimeError("each E6-A window must contain exactly 168 slots")

    predictions.to_csv(HERE / "baseline_predictions.csv", index=False)
    metrics.to_csv(HERE / "baseline_metrics.csv", index=False)
    audit_frame = pd.DataFrame(temporal_audit)
    audit_frame.to_csv(HERE / "temporal_run_audit.csv", index=False)
    git_text = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    manifest = {
        "experiment": "E6-A Temporal Diagnostic",
        "status": "STRICT_SCREENING_DIAGNOSTIC",
        "seed": SEED,
        "command": "python experiments/first_test/E6_temporal/run_e6_a.py",
        "forecast_origin": "D-1 14:00",
        "label_cutoff_for_saved_model_runs": "D-2",
        "resolution": "hourly, 24 target hours per day",
        "evaluation_days": expected,
        "source_sha256": source_sha,
        "sequence_manifest_sha256": sequence_sha,
        "selector_sha256": selector_sha,
        "temporal_run_records": temporal_audit,
        "duplicate_rule": "sort identical duplicate E2-A run folders lexicographically and choose latest; all duplicate predictions and labels must match exactly",
        "historical_spread_channel": "X_hist target_spread, indices 0..167; no target-day actuals",
        "same_hour_index_formula": "177 + hour_business - 24*lag_day; only indices 0..167 are legal",
        "periodic_definition": "majority sign across legal same-hour lags 1..7; ties non-positive",
        "28d_periodic_feature": periodic_name,
        "28d_periodic_feature_dependency_contract": period_contract,
        "zero_policy": "positive iff value > 0; zero is non-positive",
        "no_threshold_tuning": True,
        "no_class_weight": True,
        "no_feature_expansion": True,
        "no_ensemble": True,
        "python": platform.python_version(),
        "dependencies": {name: _package_version(name) for name in ("numpy", "pandas", "pyarrow")},
        "git_commit": git_text if re.fullmatch(r"[0-9a-f]{40}", git_text) else None,
        "git_commit_status": "recorded" if re.fullmatch(r"[0-9a-f]{40}", git_text) else "no HEAD commit available",
    }
    (HERE / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

"""E1.5-A step 1 — build the per-slot prediction panel for every audited model.

TafM T0/T1/T2 are read straight out of the completed E1_mini runs: no retraining,
no checkpoint re-selection, no threshold change, no writes into E1_mini.

The four pre-registered baselines are fitted here on legally available history only
(S <= D-2, quarantine-clean, target day never enters its own training window):

  B0  always non-positive                       (class-prior sanity baseline)
  B1  per-hour majority over legal [D-181, D-2] (hour + recent-direction prior only)
  B2  frozen direction-benchmark XGBoost, recent legal 180d
  B3  identical to B2 except all legal history S <= D-2
  B4  run_strict_lightgbm_baseline(window_days=180), sign of the regression output

B2/B3 re-implement the baseline logic of pipeline.run_direction_benchmark without
invoking it, so the TabM portion of that benchmark is never duplicated.

Outputs (all under this experiment directory):
  predictions/truth_panel.parquet
  predictions/baseline_predictions.parquet
  predictions/tafm_predictions.parquet
  predictions/panel.parquet
  predictions/baseline_training_log.csv
  predictions/runtime.csv
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

import common as C

PROB_NAN = np.nan


def _model_frame(model: str, day: date, direction_hat, *, p_positive=None, rank_score=None) -> pd.DataFrame:
    n = len(direction_hat)
    hours = list(range(1, n + 1))
    return pd.DataFrame(
        {
            "model": model,
            "target_day": day,
            "hour_business": hours,
            "direction_hat": np.asarray(direction_hat).astype(bool),
            "p_positive": np.full(n, PROB_NAN, dtype=float) if p_positive is None else np.asarray(p_positive, dtype=float),
            "rank_score": np.full(n, PROB_NAN, dtype=float) if rank_score is None else np.asarray(rank_score, dtype=float),
        }
    )


# --- Truth panel ------------------------------------------------------------
def build_truth(store) -> pd.DataFrame:
    """Target-day truth (RT-DA) for evaluation only. Never used to fit anything."""
    days = store.days
    rows = []
    for day in C.TARGET_DAYS:
        matches = np.flatnonzero(days == day)
        if len(matches) != 1:
            raise ValueError(f"target day {day} is not a unique sample in the frozen sequence asset")
        y = -np.asarray(store.y_source[int(matches[0])], dtype=float)  # DA-RT -> RT-DA
        for h in range(24):
            rows.append({"target_day": day, "hour_business": h + 1, "y_true_model": float(y[h])})
    return pd.DataFrame(rows)


# --- B0 ---------------------------------------------------------------------
def build_b0() -> pd.DataFrame:
    frames = [
        _model_frame("B0_AlwaysNonPositive", day, np.zeros(24, dtype=bool))
        for day in C.TARGET_DAYS
    ]
    return pd.concat(frames, ignore_index=True)


# --- B1 ---------------------------------------------------------------------
def build_b1(store, selector) -> tuple[pd.DataFrame, list[dict]]:
    """Per-hour majority direction over legal labels in [D-181, D-2]. Fail closed."""
    days = store.days
    frames, log = [], []
    for day in C.TARGET_DAYS:
        legal = C.legal_indices(store, selector, day, window_days=None)  # already S <= D-2
        earliest = day - timedelta(days=C.B1_WINDOW_DAYS)
        window = [i for i in legal if days[i] >= earliest]
        if not window:
            raise ValueError(f"B1: no legal history at all for {day}; refusing to look forward")
        labels = -np.asarray(store.y_source[window], dtype=float)  # (n_days, 24) RT-DA
        if labels.shape[1] != 24:
            raise ValueError("B1: legal history is not 24 slots per day")

        global_positive_fraction = float((labels > 0).mean())
        global_majority = int(global_positive_fraction > 0.5)
        hours, hats, scores = [], [], []
        n_fallback = 0
        n_tie = 0
        for h in range(24):
            column = labels[:, h]
            if len(column) == 0:  # fail-closed fallback, recorded
                pred, score = global_majority, global_positive_fraction
                n_fallback += 1
            else:
                fraction = float((column > 0).mean())
                if abs(fraction - 0.5) < 1e-12:
                    n_tie += 1
                pred, score = int(fraction > 0.5), fraction
            hours.append(h + 1)
            hats.append(pred)
            scores.append(score)
        frames.append(_model_frame("B1_HourMajority180", day, hats, rank_score=scores))
        log.append(
            {
                "model": "B1_HourMajority180",
                "target_day": day.isoformat(),
                "n_train_days": len(window),
                "train_day_min": min(days[i] for i in window).isoformat(),
                "train_day_max": max(days[i] for i in window).isoformat(),
                "n_train_rows": int(labels.size),
                "n_fallback_hours": n_fallback,
                "n_tie_hours": n_tie,
                "global_positive_fraction": global_positive_fraction,
            }
        )
    return pd.concat(frames, ignore_index=True), log


# --- B2 / B3 ----------------------------------------------------------------
def build_xgb(store, selector, *, window_days: int | None, model_name: str) -> tuple[pd.DataFrame, list[dict]]:
    """Frozen direction-benchmark XGBoost. Only the training window varies B2 -> B3."""
    import xgboost as xgb
    from src.TafM_改进源码.target_adapter import source_to_model_target

    _, cols, _ = store.selector_indices(selector)
    days = store.days
    n_features = len(cols)
    frames, log = [], []
    for day in C.TARGET_DAYS:
        # Keep this as an array: pipeline.run_direction_benchmark indexes x_future with
        # the flatnonzero result, which keeps the array 3-D for the `[:, :, cols]` slice.
        sample = np.flatnonzero(days == day)
        if len(sample) != 1:
            raise ValueError(f"{model_name}: target day {day} is not a unique sample")

        eligible = C.legal_indices(store, selector, day, window_days=window_days)
        if len(eligible) < 30:
            raise ValueError(f"{model_name}: insufficient legal history for {day}: {len(eligible)} days")

        xt = np.asarray(store.x_future[eligible][:, :, cols], dtype=np.float32).reshape(-1, n_features)
        yt = source_to_model_target(np.asarray(store.y_source[eligible], dtype=np.float32)).reshape(-1)
        model = xgb.XGBClassifier(
            n_estimators=90, max_depth=4, learning_rate=0.05, subsample=1.0, colsample_bytree=0.8,
            reg_lambda=1, objective="binary:logistic", tree_method="hist", n_jobs=1,
            random_state=C.SEED, eval_metric="logloss",
        )
        model.fit(xt, (yt > 0).astype(np.int8))

        xq = np.asarray(store.x_future[sample][:, :, cols], dtype=np.float32).reshape(-1, n_features)
        probability = model.predict_proba(xq)[:, 1].astype(float)
        frames.append(
            _model_frame(model_name, day, probability >= 0.5, p_positive=probability, rank_score=probability)
        )
        log.append(
            {
                "model": model_name,
                "target_day": day.isoformat(),
                "n_train_days": len(eligible),
                "train_day_min": min(days[i] for i in eligible).isoformat(),
                "train_day_max": max(days[i] for i in eligible).isoformat(),
                "n_train_rows": int(xt.shape[0]),
                "n_features": n_features,
                "n_fallback_hours": 0,
                "n_tie_hours": 0,
                "global_positive_fraction": float((yt > 0).mean()),
            }
        )
    return pd.concat(frames, ignore_index=True), log


# --- B4 ---------------------------------------------------------------------
def build_b4() -> tuple[pd.DataFrame, list[dict]]:
    """Reuse the unchanged historical LightGBM via run_strict_lightgbm_baseline."""
    from src.TafM_改进源码.evaluate import run_strict_lightgbm_baseline

    frames, log = [], []
    for window, (start, end) in C.WINDOWS.items():
        started = time.perf_counter()
        frame = run_strict_lightgbm_baseline(start, end, window_days=180)
        elapsed = time.perf_counter() - started
        frame = frame.copy()
        frame["target_day"] = pd.to_datetime(frame["target_day"]).dt.date
        frame = frame[frame["target_day"].isin(C.TARGET_DAYS)]
        for day, group in frame.groupby("target_day", sort=True):
            group = group.sort_values("hour_business")
            frames.append(
                _model_frame(
                    "B4_StrictLightGBM_DIR_180",
                    day,
                    group["prediction_model"].to_numpy(dtype=float) > 0,
                    rank_score=group["prediction_model"].to_numpy(dtype=float),
                )
            )
        log.append(
            {
                "model": "B4_StrictLightGBM_DIR_180",
                "target_day": f"{start}..{end}",
                "n_train_days": 180,
                "train_day_min": "",
                "train_day_max": "",
                "n_train_rows": int(len(frame)),
                "n_features": -1,
                "n_fallback_hours": 0,
                "n_tie_hours": 0,
                "global_positive_fraction": float((frame["y_true_model"] > 0).mean()),
                "fit_seconds": elapsed,
                "window": window,
            }
        )
    return pd.concat(frames, ignore_index=True), log


def main() -> int:
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    runtime: list[dict] = []
    started_all = time.perf_counter()

    print("[E1.5-A] loading frozen sequence asset + selector ...")
    store, selector, selector_hash = C.load_store_and_selector()
    print(f"[E1.5-A] selector_sha256={selector_hash}")
    print(f"[E1.5-A] sequence days={len(store.days)} range={store.days.min()}..{store.days.max()}")

    truth = build_truth(store)
    truth.to_parquet(C.PRED_DIR / "truth_panel.parquet", index=False)

    print("[E1.5-A] reading TafM T0/T1/T2 from E1_mini ...")
    tafm = C.load_tafm_predictions()
    tafm.to_parquet(C.PRED_DIR / "tafm_predictions.parquet", index=False)

    # Truth cross-check: E1 predictions and the frozen asset must agree exactly.
    merged = tafm.merge(truth, on=["target_day", "hour_business"], how="inner", validate="many_to_one")
    if len(merged) != len(tafm):
        raise ValueError("TafM predictions did not fully join onto the truth panel")
    worst = float(np.max(np.abs(merged["y_true_model_x"] - merged["y_true_model_y"])))
    print(f"[E1.5-A] truth cross-check: max|E1 predictions - frozen asset| = {worst:.3e}")
    if worst > 1e-3:
        raise ValueError("E1 predictions disagree with the frozen sequence asset on target truth")

    baseline_frames, logs = [], []
    for name, builder in (
        ("B0_AlwaysNonPositive", lambda: (build_b0(), [])),
        ("B1_HourMajority180", lambda: build_b1(store, selector)),
        ("B2_XGB_DIR_180", lambda: build_xgb(store, selector, window_days=C.B2_WINDOW_DAYS, model_name="B2_XGB_DIR_180")),
        ("B3_XGB_DIR_EXPANDING", lambda: build_xgb(store, selector, window_days=None, model_name="B3_XGB_DIR_EXPANDING")),
        ("B4_StrictLightGBM_DIR_180", build_b4),
    ):
        started = time.perf_counter()
        frame, log = builder()
        elapsed = time.perf_counter() - started
        baseline_frames.append(frame)
        logs.extend(log)
        runtime.append({"model": name, "fit_seconds": elapsed, "n_slots": int(len(frame))})
        print(f"[E1.5-A] {name}: {len(frame)} slots in {elapsed:.2f}s")

    baselines = pd.concat(baseline_frames, ignore_index=True)
    baselines.to_parquet(C.PRED_DIR / "baseline_predictions.parquet", index=False)
    pd.DataFrame(logs).to_csv(C.PRED_DIR / "baseline_training_log.csv", index=False)

    # One merged long panel: every model on the same 672 slots and the same truth.
    panel = pd.concat(
        [
            baselines,
            tafm[["model", "target_day", "hour_business", "direction_hat", "p_positive"]].assign(rank_score=np.nan),
        ],
        ignore_index=True,
    )
    panel = panel.merge(truth, on=["target_day", "hour_business"], how="left", validate="many_to_one")
    panel["window"] = panel["target_day"].map(C.DAY_TO_WINDOW)
    panel["hour_segment"] = panel["hour_business"].map(C.HOUR_TO_SEGMENT)
    if panel["y_true_model"].isna().any():
        raise ValueError("merged panel has slots without truth")
    counts = panel.groupby("model").size()
    if not (counts == 672).all():
        raise ValueError(f"every model must cover exactly 672 slots; got\n{counts}")
    panel.to_parquet(C.PRED_DIR / "panel.parquet", index=False)

    runtime.append({"model": "__total__", "fit_seconds": time.perf_counter() - started_all, "n_slots": int(len(panel))})
    pd.DataFrame(runtime).to_csv(C.PRED_DIR / "runtime.csv", index=False)

    summary = {
        "target_days": len(C.TARGET_DAYS),
        "slots": int(len(panel)),
        "models": sorted(panel["model"].unique().tolist()),
        "selector_sha256": selector_hash,
        "truth_cross_check_max_abs_delta": worst,
        "b1_window_days": C.B1_WINDOW_DAYS,
        "b2_window_days": C.B2_WINDOW_DAYS,
        "seed": C.SEED,
    }
    (C.PRED_DIR / "panel_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[E1.5-A] panel built:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

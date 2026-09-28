"""E1-mini aggregation and gate analysis.

Reads only the preserved per-run records under E1_mini/runs and the raw
predictions.parquet of each run. Micro metrics reuse the canonical
`canonical_metrics` implementation so E1 numbers are definitionally identical
to the frozen V2.1 KPI definitions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from TafM_改进源码.metrics import canonical_metrics  # noqa: E402

E1_ROOT = PROJECT_ROOT / "experiments" / "first_test" / "E1_mini"
RUNS_DIR = E1_ROOT / "runs"
FIG_DIR = E1_ROOT / "E1_curves_or_figures"

BOOTSTRAP_SEED = 20260924
BOOTSTRAP_RESAMPLES = 10000
TIE_EPS = 1e-12

VARIANTS = ("M0", "M1", "M2")
WINDOW_LABEL = {"W1": "2026-02-12..18", "W2": "2026-04-12..18", "W3": "2026-06-12..18", "W4": "2026-08-07..13"}
PAIRINGS = (("Delta10", "M1", "M0"), ("Delta21", "M2", "M1"))


def load_records() -> list[dict]:
    records = []
    for record_path in sorted(RUNS_DIR.glob("*/RUN_RECORD.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            continue
        records.append(record)
    return records


def load_predictions(record: dict) -> pd.DataFrame | None:
    raw_dir = record.get("raw_run_dir")
    if not raw_dir:
        return None
    path = Path(raw_dir) / "predictions.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def daily_rows(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    rows, pooled = [], {}
    for record in records:
        variant, day = record["variant"], record["target_day"]
        preds = load_predictions(record)
        if preds is None or len(preds) != 24:
            raise RuntimeError(f"missing/incomplete predictions for {record['run_id']}")
        truth = preds["y_true_model"].to_numpy(dtype=float)
        direction = preds["direction_hat"].to_numpy(dtype=bool)
        magnitude = preds["magnitude_hat"].to_numpy(dtype=float)
        probability = preds["p_positive"].to_numpy(dtype=float)
        metrics = canonical_metrics(truth, direction, magnitude, direction_probability=probability)
        m = record["metrics"]
        t = record["telemetry"]
        c = record.get("checkpoint", {})
        predicted_positive_fraction = float(direction.mean())
        n_pos_true = int((truth > 0).sum())
        n_pred_pos = int(direction.sum())
        rows.append({
            "run_id": record["run_id"], "variant": variant, "window": record["window"], "target_day": day,
            "n_slots": len(preds),
            "micro_raw": metrics["raw_direction_accuracy"],
            "micro_balanced": metrics["balanced_accuracy"],
            "micro_positive_recall": metrics["positive_recall"],
            "micro_nonpositive_recall": metrics["nonpositive_recall"],
            "micro_auc": metrics["auc"], "micro_brier": metrics["brier"],
            "micro_magnitude_mae": metrics["magnitude_mae"],
            "micro_magnitude_rmse": metrics["magnitude_rmse"],
            "micro_magnitude_tail_mae": metrics["magnitude_tail_mae"],
            "magnitude_mae_naive": m.get("magnitude_mae_naive"),
            "magnitude_skill": m.get("magnitude_skill"),
            "predicted_positive_fraction": predicted_positive_fraction,
            "n_true_positive": n_pos_true, "n_predicted_positive": n_pred_pos,
            "n_true_negative": len(preds) - n_pos_true,
            "one_class_prediction_day": predicted_positive_fraction in (0.0, 1.0),
            "positive_recall_zero": metrics["positive_recall"] == 0.0,
            "nonpositive_recall_zero": metrics["nonpositive_recall"] == 0.0,
            "best_epoch": t.get("best_epoch"), "stop_epoch": t.get("stop_epoch"), "epochs_run": t.get("epochs_run"),
            "wall_time_total_seconds": t.get("wall_time_total_seconds"),
            "training_epoch_seconds_mean": t.get("training_epoch_seconds_mean"),
            "prediction_latency_ms": t.get("prediction_latency_ms"),
            "cuda_peak_memory_bytes": t.get("cuda_peak_memory_bytes"),
            "checkpoint_policy": record.get("manifest_fields", {}).get("checkpoint_policy"),
            "objective_mode": record.get("manifest_fields", {}).get("objective_mode"),
            "checkpoint_selected_raw": c.get("selected_raw"),
            "checkpoint_raw_anchor": c.get("raw_anchor"),
            "monitor_raw": c.get("monitor_raw"), "monitor_balanced": c.get("monitor_balanced"),
            "monitor_positive_recall": c.get("monitor_positive_recall"),
            "monitor_nonpositive_recall": c.get("monitor_nonpositive_recall"),
            "monitor_L_dir": c.get("monitor_L_dir"),
            "monitor_magnitude_mae": c.get("monitor_magnitude_mae"),
            "monitor_predicted_positive_fraction": c.get("monitor_predicted_positive_fraction"),
            "warnings": ";".join(record.get("warnings", [])),
        })
        pooled.setdefault(variant, {"truth": [], "direction": [], "magnitude": [], "probability": [],
                                    "naive_mae": [], "days": []})
        pooled[variant]["truth"].append(truth)
        pooled[variant]["direction"].append(direction)
        pooled[variant]["magnitude"].append(magnitude)
        pooled[variant]["probability"].append(probability)
        pooled[variant]["naive_mae"].append(m.get("magnitude_mae_naive"))
        pooled[variant]["days"].append(day)
    return pd.DataFrame(rows).sort_values(["variant", "target_day"]).reset_index(drop=True), pooled


def micro_metrics(pool: dict) -> dict:
    truth = np.concatenate(pool["truth"])
    direction = np.concatenate(pool["direction"])
    magnitude = np.concatenate(pool["magnitude"])
    probability = np.concatenate(pool["probability"])
    metrics = canonical_metrics(truth, direction, magnitude, direction_probability=probability)
    naive_mae = np.concatenate([np.full(24, v, dtype=float) if v is not None else np.full(24, np.nan)
                                for v in pool["naive_mae"]])
    micro_naive = float(np.nanmean(naive_mae))
    return {
        "micro_raw": metrics["raw_direction_accuracy"],
        "micro_balanced": metrics["balanced_accuracy"],
        "micro_positive_recall": metrics["positive_recall"],
        "micro_nonpositive_recall": metrics["nonpositive_recall"],
        "micro_auc": metrics["auc"], "micro_brier": metrics["brier"],
        "micro_magnitude_mae": metrics["magnitude_mae"],
        "micro_magnitude_rmse": metrics["magnitude_rmse"],
        "micro_magnitude_tail_mae": metrics["magnitude_tail_mae"],
        "micro_magnitude_mae_naive": micro_naive,
        "micro_magnitude_skill": 1.0 - metrics["magnitude_mae"] / micro_naive if micro_naive else None,
        "micro_predicted_positive_fraction": float(direction.mean()),
        "n_slots": int(len(truth)), "n_true_positive": int((truth > 0).sum()),
        "n_predicted_positive": int(direction.sum()),
    }


def summarize(daily: pd.DataFrame, pooled: dict) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        sub = daily[daily["variant"] == variant]
        rows.append({
            "variant": variant, "n_days": len(sub), "window_count": sub["window"].nunique(),
            "objective_mode": sub["objective_mode"].iloc[0], "checkpoint_policy": sub["checkpoint_policy"].iloc[0],
            **micro_metrics(pooled[variant]),
            "macro_day_raw": sub["micro_raw"].mean(),
            "macro_day_raw_std": sub["micro_raw"].std(ddof=1),
            "macro_day_raw_median": sub["micro_raw"].median(),
            "macro_day_balanced": sub["micro_balanced"].mean(),
            "macro_day_positive_recall": sub["micro_positive_recall"].mean(),
            "macro_day_nonpositive_recall": sub["micro_nonpositive_recall"].mean(),
            "macro_day_magnitude_mae": sub["micro_magnitude_mae"].mean(),
            "macro_day_magnitude_skill": sub["magnitude_skill"].mean(),
            "macro_day_predicted_positive_fraction": sub["predicted_positive_fraction"].mean(),
            "one_class_prediction_days": int(sub["one_class_prediction_day"].sum()),
            "positive_recall_zero_days": int(sub["positive_recall_zero"].sum()),
            "nonpositive_recall_zero_days": int(sub["nonpositive_recall_zero"].sum()),
            "all_negative_baseline_raw": 1.0 - sub["n_true_positive"].sum() / sub["n_slots"].sum(),
            "mean_best_epoch": sub["best_epoch"].mean(), "mean_stop_epoch": sub["stop_epoch"].mean(),
            "total_wall_time_seconds": sub["wall_time_total_seconds"].sum(),
            "mean_wall_time_seconds": sub["wall_time_total_seconds"].mean(),
        })
    return pd.DataFrame(rows)


def window_table(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        for window in ("W1", "W2", "W3", "W4"):
            sub = daily[(daily["variant"] == variant) & (daily["window"] == window)]
            rows.append({
                "variant": variant, "window": window, "window_days": WINDOW_LABEL[window], "n_days": len(sub),
                "macro_day_raw": sub["micro_raw"].mean(),
                "macro_day_raw_std": sub["micro_raw"].std(ddof=1),
                "raw_min": sub["micro_raw"].min(), "raw_max": sub["micro_raw"].max(),
                "macro_day_balanced": sub["micro_balanced"].mean(),
                "macro_day_positive_recall": sub["micro_positive_recall"].mean(),
                "macro_day_nonpositive_recall": sub["micro_nonpositive_recall"].mean(),
                "macro_day_magnitude_mae": sub["micro_magnitude_mae"].mean(),
                "macro_day_magnitude_skill": sub["magnitude_skill"].mean(),
                "macro_day_predicted_positive_fraction": sub["predicted_positive_fraction"].mean(),
                "one_class_prediction_days": int(sub["one_class_prediction_day"].sum()),
                "positive_recall_zero_days": int(sub["positive_recall_zero"].sum()),
                "nonpositive_recall_zero_days": int(sub["nonpositive_recall_zero"].sum()),
                "mean_best_epoch": sub["best_epoch"].mean(),
            })
    return pd.DataFrame(rows)


def _slots_micro(sub: pd.DataFrame) -> dict:
    """Slot-weighted micro metrics for a (variant, window) subset from its daily rows."""
    n_slots = int(sub["n_slots"].sum())
    if n_slots == 0:
        return {key: None for key in ("micro_raw", "micro_balanced", "micro_positive_recall",
                                      "micro_nonpositive_recall", "micro_magnitude_mae",
                                      "micro_predicted_positive_fraction")}
    n_pos = int(sub["n_true_positive"].sum())
    n_pred_pos = int(sub["n_predicted_positive"].sum())
    tp = float((sub["micro_positive_recall"] * sub["n_true_positive"]).sum())
    tn = float((sub["micro_nonpositive_recall"] * (sub["n_slots"] - sub["n_true_positive"])).sum())
    correct = float((sub["micro_raw"] * sub["n_slots"]).sum())
    pos_recall = tp / n_pos if n_pos else None
    neg_recall = tn / (n_slots - n_pos) if n_slots > n_pos else None
    balanced = (float(np.mean([pos_recall, neg_recall]))
                if pos_recall is not None and neg_recall is not None else None)
    return {
        "micro_raw": correct / n_slots,
        "micro_balanced": balanced,
        "micro_positive_recall": pos_recall, "micro_nonpositive_recall": neg_recall,
        "micro_magnitude_mae": (sub["micro_magnitude_mae"] * sub["n_slots"]).sum() / n_slots,
        "micro_predicted_positive_fraction": n_pred_pos / n_slots,
    }


def window_table_full(daily: pd.DataFrame) -> pd.DataFrame:
    base = window_table(daily)
    slots = []
    for _, row in base.iterrows():
        sub = daily[(daily["variant"] == row["variant"]) & (daily["window"] == row["window"])]
        slots.append(_slots_micro(sub))
    return pd.concat([base, pd.DataFrame(slots)], axis=1)


def day_cluster_bootstrap(deltas: np.ndarray, *, seed: int = BOOTSTRAP_SEED,
                          resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    d = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "ci_low": float(np.percentile(means, 2.5)), "ci_high": float(np.percentile(means, 97.5)),
            "ci_includes_zero": bool(np.percentile(means, 2.5) <= 0.0 <= np.percentile(means, 97.5)),
            "bootstrap_seed": seed, "bootstrap_resamples": resamples, "bootstrap_unit": "target_day"}


DIRECTION_METRICS = [("raw", "micro_raw", True), ("balanced", "micro_balanced", True),
                     ("positive_recall", "micro_positive_recall", True),
                     ("nonpositive_recall", "micro_nonpositive_recall", True),
                     ("predicted_positive_fraction", "predicted_positive_fraction", None),
                     ("magnitude_mae", "micro_magnitude_mae", False),
                     ("best_epoch", "best_epoch", None),
                     ("runtime_seconds", "wall_time_total_seconds", None)]


def paired_analysis(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = daily.set_index(["variant", "target_day"])
    rows, summaries = [], []
    for label, a, b in PAIRINGS:
        for _, day_row in daily[daily["variant"] == a].sort_values("target_day").iterrows():
            day = day_row["target_day"]
            ra, rb = indexed.loc[(a, day)], indexed.loc[(b, day)]
            row = {"comparison": label, "variant_a": a, "variant_b": b, "window": day_row["window"], "target_day": day}
            for name, column, _ in DIRECTION_METRICS:
                row[f"delta_{name}"] = float(ra[column]) - float(rb[column])
            row["improved_raw"] = int(row["delta_raw"] > TIE_EPS)
            row["tie_raw"] = int(abs(row["delta_raw"]) <= TIE_EPS)
            row["worsened_raw"] = int(row["delta_raw"] < -TIE_EPS)
            rows.append(row)
        sub = pd.DataFrame([r for r in rows if r["comparison"] == label])
        summary = {"comparison": label, "variant_a": a, "variant_b": b, "n_days": len(sub)}
        for name, column, higher_is_better in DIRECTION_METRICS:
            deltas = sub[f"delta_{name}"].to_numpy(dtype=float)
            summary[f"mean_delta_{name}"] = float(np.mean(deltas))
            summary[f"median_delta_{name}"] = float(np.median(deltas))
            if higher_is_better is not None:
                wins = int((deltas > TIE_EPS).sum()) if higher_is_better else int((deltas < -TIE_EPS).sum())
                losses = int((deltas < -TIE_EPS).sum()) if higher_is_better else int((deltas > TIE_EPS).sum())
                summary[f"win_{name}"] = wins
                summary[f"tie_{name}"] = int((np.abs(deltas) <= TIE_EPS).sum())
                summary[f"loss_{name}"] = losses
            if name in {"raw", "balanced", "positive_recall", "nonpositive_recall", "magnitude_mae"}:
                boot = day_cluster_bootstrap(deltas)
                summary[f"boot_ci_low_{name}"] = boot["ci_low"]
                summary[f"boot_ci_high_{name}"] = boot["ci_high"]
                summary[f"boot_ci_includes_zero_{name}"] = boot["ci_includes_zero"]
        for window in ("W1", "W2", "W3", "W4"):
            wsub = sub[sub["window"] == window]
            summary[f"window_{window}_mean_delta_raw"] = float(wsub["delta_raw"].mean())
            summary[f"window_{window}_mean_delta_magnitude_mae"] = float(wsub["delta_magnitude_mae"].mean())
        summaries.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def checkpoint_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        c = record.get("checkpoint", {})
        selected = c.get("selected_raw")
        anchor = c.get("raw_anchor")
        collapse = bool(record.get("metrics", {}).get("predicted_positive_fraction") in (0.0, 1.0))
        warning_union = set(record.get("warnings", []))
        target_flags = sorted({w for w in warning_union if w.startswith(("one_class", "positive_recall_zero",
                                                                        "nonpositive_recall_zero"))})
        rows.append({
            "run_id": record["run_id"], "variant": record["variant"], "window": record["window"],
            "target_day": record["target_day"], "objective_mode": record["manifest_fields"].get("objective_mode"),
            "checkpoint_policy": record["manifest_fields"].get("checkpoint_policy"),
            "best_epoch": c.get("best_epoch"), "stop_epoch": record["telemetry"].get("stop_epoch"),
            "selected_monitor_raw": c.get("monitor_raw"),
            "selected_monitor_balanced": c.get("monitor_balanced"),
            "selected_monitor_positive_recall": c.get("monitor_positive_recall"),
            "selected_monitor_nonpositive_recall": c.get("monitor_nonpositive_recall"),
            "selected_monitor_L_dir": c.get("monitor_L_dir"),
            "selected_monitor_magnitude_mae": c.get("monitor_magnitude_mae"),
            "selected_monitor_predicted_positive_fraction": c.get("monitor_predicted_positive_fraction"),
            "raw_anchor": anchor, "checkpoint_selected_raw": selected,
            "selected_minus_anchor": (float(selected) - float(anchor)) if selected is not None and anchor is not None else None,
            "monitor_raw_minus_anchor": (float(c.get("monitor_raw")) - float(anchor))
            if c.get("monitor_raw") is not None and anchor is not None else None,
            "guardrail_status": c.get("guardrail_status"),
            "target_day_one_class_prediction_collapse": collapse,
            "target_day_collapse_warning": bool(target_flags),
            "target_day_warning_flags": ";".join(target_flags),
            "selected_monitor_warning_flags": c.get("monitor_diagnostic_warnings", ""),
        })
    return pd.DataFrame(rows)


def runtime_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        t = record["telemetry"]
        rows.append({
            "run_id": record["run_id"], "variant": record["variant"], "window": record["window"],
            "target_day": record["target_day"], "device": record["device"], "amp": record["amp"],
            "wall_time_total_seconds": t.get("wall_time_total_seconds"),
            "training_epoch_seconds_total": t.get("training_epoch_seconds_total"),
            "training_epoch_seconds_mean": t.get("training_epoch_seconds_mean"),
            "training_epoch_seconds_p50": t.get("training_epoch_seconds_p50"),
            "training_epoch_seconds_p95": t.get("training_epoch_seconds_p95"),
            "epochs_run": t.get("epochs_run"), "best_epoch": t.get("best_epoch"), "stop_epoch": t.get("stop_epoch"),
            "prediction_latency_ms": t.get("prediction_latency_ms"),
            "parameter_count_total": t.get("parameter_count_total"),
            "parameter_count_trainable": t.get("parameter_count_trainable"),
            "checkpoint_size_bytes": t.get("checkpoint_size_bytes"),
            "cuda_peak_memory_bytes": t.get("cuda_peak_memory_bytes"),
        })
    return pd.DataFrame(rows)


def write_figures(daily: pd.DataFrame, paired: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    colors = {"M0": "#4c72b0", "M1": "#dd8452", "M2": "#55a868"}

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for ax, window in zip(axes, ("W1", "W2", "W3", "W4")):
        sub = daily[daily["window"] == window]
        days = sorted(sub["target_day"].unique())
        positions = np.arange(len(days))
        width = 0.26
        for offset, variant in enumerate(VARIANTS):
            values = [sub[(sub["variant"] == variant) & (sub["target_day"] == d)]["micro_raw"].iloc[0] for d in days]
            ax.bar(positions + (offset - 1) * width, values, width, label=variant, color=colors[variant])
        raw = sub[sub["variant"] == "M0"].set_index("target_day")["micro_raw"]
        ax.axhline(float(np.mean(list(raw))), color="grey", linestyle=":", linewidth=1)
        ax.set_ylabel("Daily Raw"); ax.set_title(f"{window} {WINDOW_LABEL[window]}")
        ax.set_xticks(positions); ax.set_xticklabels([d[5:] for d in days], rotation=45, ha="right")
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower right", ncol=3)
    fig.suptitle("E1-mini daily Raw Direction Accuracy by window")
    fig.tight_layout()
    path = FIG_DIR / "e1_daily_raw_by_window.png"
    fig.savefig(path, dpi=140); plt.close(fig); written.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (label, _, _) in zip(axes, PAIRINGS):
        sub = paired[paired["comparison"] == label]
        for column, name, color in (("delta_raw", "Raw", "#4c72b0"), ("delta_magnitude_mae", "Magnitude MAE", "#c44e52")):
            ax.scatter(sub["target_day"], sub[column], s=26, label=name, color=color)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(label); ax.set_ylabel("delta"); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=60)
        ax.legend()
    fig.suptitle("E1-mini paired daily deltas")
    fig.tight_layout()
    path = FIG_DIR / "e1_paired_deltas.png"
    fig.savefig(path, dpi=140); plt.close(fig); written.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["one-class days", "+Recall=0 days", "-Recall=0 days"]
    columns = ["one_class_prediction_day", "positive_recall_zero", "nonpositive_recall_zero"]
    positions = np.arange(len(labels)); width = 0.26
    for offset, variant in enumerate(VARIANTS):
        counts = [int(daily[daily["variant"] == variant][c].sum()) for c in columns]
        ax.bar(positions + (offset - 1) * width, counts, width, label=variant, color=colors[variant])
    ax.set_xticks(positions); ax.set_xticklabels(labels); ax.set_ylabel("target days (of 28)")
    ax.set_title("E1-mini target-day safety/collapse counts"); ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "e1_safety_counts.png"
    fig.savefig(path, dpi=140); plt.close(fig); written.append(path.name)
    return written


def main() -> int:
    records = load_records()
    completeness = {variant: sorted(r["target_day"] for r in records if r["variant"] == variant) for variant in VARIANTS}
    daily, pooled = daily_rows(records)
    summary = summarize(daily, pooled)
    windows = window_table_full(daily)
    paired_rows, paired_summary = paired_analysis(daily)
    checkpoints = checkpoint_table(records)
    runtime = runtime_table(records)

    daily.to_csv(E1_ROOT / "daily_metrics.csv", index=False)
    summary.to_csv(E1_ROOT / "variant_summary.csv", index=False)
    windows.to_csv(E1_ROOT / "window_metrics.csv", index=False)
    paired_rows.to_csv(E1_ROOT / "paired_deltas.csv", index=False)
    paired_summary.to_csv(E1_ROOT / "paired_summary.csv", index=False)
    checkpoints.to_csv(E1_ROOT / "checkpoint_diagnostics.csv", index=False)
    runtime.to_csv(E1_ROOT / "runtime.csv", index=False)
    figures = write_figures(daily, paired_rows)

    print(json.dumps({"runs": len(records), "days_per_variant": {k: len(v) for k, v in completeness.items()},
                      "figures": figures}, indent=2))
    print(summary[["variant", "macro_day_raw", "micro_raw", "micro_balanced", "micro_positive_recall",
                   "micro_nonpositive_recall", "micro_magnitude_mae", "micro_magnitude_skill",
                   "one_class_prediction_days", "positive_recall_zero_days",
                   "nonpositive_recall_zero_days"]].to_string(index=False))
    print(paired_summary[["comparison", "mean_delta_raw", "boot_ci_low_raw", "boot_ci_high_raw",
                          "win_raw", "tie_raw", "loss_raw", "mean_delta_magnitude_mae"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

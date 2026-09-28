"""E1.5-A step 2 — unified metrics, signal diagnostics and figures.

Reads predictions/panel.parquet (built by build_e1_5_predictions.py) and writes every
required CSV plus the figure set. Pure analysis: no model is fitted here and nothing
outside this experiment directory is written.

Unified metrics follow 08_E1_5_DIRECTION_SIGNAL_AUDIT_PLAN.md §6; signal diagnostics
follow §7. The oracle threshold table is explicitly marked LEAKY_DIAGNOSTIC_ONLY --
it uses target-day truth to pick a threshold and must never be promoted to a model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402

METRIC_COLUMNS = [
    "raw_direction_accuracy", "balanced_accuracy", "positive_recall", "nonpositive_recall",
    "predicted_positive_fraction", "true_positive_prevalence", "auc", "brier", "auc_status",
    "n_slots", "n_positive", "n_nonpositive",
]

ORACLE_MARKERS = {
    "LEAKY_DIAGNOSTIC_ONLY": True,
    "NOT_A_MODEL_RESULT": True,
    "NOT_FOR_PROMOTION": True,
}


# --- helpers ----------------------------------------------------------------
def score_args(frame: pd.DataFrame) -> dict:
    """Prefer a real probability; fall back to a ranking score; else nothing."""
    probability = frame["p_positive"].to_numpy(dtype=float)
    rank = frame["rank_score"].to_numpy(dtype=float)
    if not np.isnan(probability).all():
        return {"p_positive": probability}
    if not np.isnan(rank).all():
        return {"rank_score": rank}
    return {}


def metrics_row(frame: pd.DataFrame, **extra) -> dict:
    row = C.direction_metrics(
        frame["y_true_model"].to_numpy(dtype=float),
        frame["direction_hat"].to_numpy(dtype=bool),
        **score_args(frame),
    )
    row.update(extra)
    return row


def _rounded(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype.kind == "f":
            out[column] = out[column].astype(float).round(6)
    return out


# --- tables -----------------------------------------------------------------
def daily_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, day), group in panel.groupby(["model", "target_day"], sort=True):
        row = metrics_row(group)
        row.update(
            {
                "model": model,
                "target_day": day.isoformat(),
                "window": C.DAY_TO_WINDOW[day],
                "truth_positive_fraction": float((group["y_true_model"] > 0).mean()),
                "one_class_prediction_day": bool(
                    row["predicted_positive_fraction"] in (0.0, 1.0)
                ),
                "positive_recall_zero_day": bool(
                    row["n_positive"] > 0 and row["positive_recall"] == 0.0
                ),
                "nonpositive_recall_zero_day": bool(
                    row["n_nonpositive"] > 0 and row["nonpositive_recall"] == 0.0
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)[
        ["model", "window", "target_day", "raw_direction_accuracy", "balanced_accuracy",
         "positive_recall", "nonpositive_recall", "predicted_positive_fraction",
         "truth_positive_fraction", "one_class_prediction_day",
         "positive_recall_zero_day", "nonpositive_recall_zero_day",
         "n_positive", "n_nonpositive", "auc", "brier", "auc_status"]
    ]


def overall_table(panel: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in panel.groupby("model", sort=False):
        days = daily[daily["model"] == model]
        row = metrics_row(group)
        macro = float(days["raw_direction_accuracy"].mean())
        row.update(
            {
                "model": model,
                "macro_day_raw": macro,
                "micro_minus_macro_raw": float(row["raw_direction_accuracy"]) - macro,
                "one_class_prediction_days": int(days["one_class_prediction_day"].sum()),
                "positive_recall_zero_days": int(days["positive_recall_zero_day"].sum()),
                "nonpositive_recall_zero_days": int(days["nonpositive_recall_zero_day"].sum()),
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows)
    order = ["model", "raw_direction_accuracy", "macro_day_raw", "micro_minus_macro_raw",
             "balanced_accuracy", "positive_recall", "nonpositive_recall",
             "predicted_positive_fraction", "true_positive_prevalence", "auc", "brier",
             "auc_status", "one_class_prediction_days", "positive_recall_zero_days",
             "nonpositive_recall_zero_days", "n_slots", "n_positive", "n_nonpositive"]
    return table[order]


def window_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, window), group in panel.groupby(["model", "window"], sort=True):
        row = metrics_row(group)
        row["majority_baseline_raw"] = C.majority_baseline_raw(group["y_true_model"].to_numpy(float))
        row.update({"model": model, "window": window})
        rows.append(row)
    table = pd.DataFrame(rows)
    return table[["model", "window", "raw_direction_accuracy", "balanced_accuracy",
                  "positive_recall", "nonpositive_recall", "predicted_positive_fraction",
                  "true_positive_prevalence", "majority_baseline_raw", "auc", "brier", "n_slots"]]


def segment_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, segment), group in panel.groupby(["model", "hour_segment"], sort=True):
        row = metrics_row(group)
        row.update({"model": model, "hour_segment": segment})
        rows.append(row)
    table = pd.DataFrame(rows)
    return table[["model", "hour_segment", "raw_direction_accuracy", "balanced_accuracy",
                  "positive_recall", "nonpositive_recall", "predicted_positive_fraction",
                  "true_positive_prevalence", "auc", "brier", "n_slots"]]


def prior_gap_table(model_summary: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """model Raw - same-window majority baseline Raw. Positive => real increment over the prior."""
    rows = []
    for _, row in model_summary.iterrows():
        rows.append(
            {
                "model": row["model"], "window": "OVERALL",
                "model_raw": row["raw_direction_accuracy"],
                "majority_baseline_raw": row["true_positive_prevalence"]
                if row["true_positive_prevalence"] >= 0.5 else 1.0 - row["true_positive_prevalence"],
                "prior_gap": row["raw_direction_accuracy"] - max(
                    row["true_positive_prevalence"], 1.0 - row["true_positive_prevalence"]),
            }
        )
    for _, row in window.iterrows():
        rows.append(
            {
                "model": row["model"], "window": row["window"],
                "model_raw": row["raw_direction_accuracy"],
                "majority_baseline_raw": row["majority_baseline_raw"],
                "prior_gap": row["raw_direction_accuracy"] - row["majority_baseline_raw"],
            }
        )
    return pd.DataFrame(rows)


def separation_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Probability separation for T0/T1/T2: p_positive | y>0 versus p_positive | y<=0."""
    rows = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("OVERALL", "ALL", panel)]
    for window in C.WINDOWS:
        scopes.append(("WINDOW", window, panel[panel["window"] == window]))
    for segment in C.HOUR_SEGMENTS:
        scopes.append(("HOUR_SEGMENT", segment, panel[panel["hour_segment"] == segment]))

    for model in ["T0", "T1", "T2"]:
        for scope, name, frame in scopes:
            group = frame[frame["model"] == model]
            if group.empty:
                continue
            p = group["p_positive"].to_numpy(dtype=float)
            truth = group["y_true_model"].to_numpy(dtype=float) > 0
            pos, neg = p[truth], p[~truth]
            row = {
                "model": model, "scope": scope, "scope_name": name,
                "n_positive": int(truth.sum()), "n_nonpositive": int((~truth).sum()),
                "mean_p_given_positive": float(pos.mean()) if pos.size else None,
                "mean_p_given_nonpositive": float(neg.mean()) if neg.size else None,
                "median_p_given_positive": float(np.median(pos)) if pos.size else None,
                "median_p_given_nonpositive": float(np.median(neg)) if neg.size else None,
            }
            row["mean_separation_gap"] = (
                row["mean_p_given_positive"] - row["mean_p_given_nonpositive"]
                if pos.size and neg.size else None
            )
            row["median_separation_gap"] = (
                row["median_p_given_positive"] - row["median_p_given_nonpositive"]
                if pos.size and neg.size else None
            )
            rows.append(row)
    return pd.DataFrame(rows)


def oracle_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc threshold sweep on already-completed E1 predictions. DIAGNOSTIC ONLY."""
    grid = np.round(np.arange(0.0, 1.0001, 0.001), 4)
    rows = []
    for model in ["T0", "T1", "T2"]:
        group = panel[panel["model"] == model]
        p = group["p_positive"].to_numpy(dtype=float)
        truth = group["y_true_model"].to_numpy(dtype=float)

        raws, balanced = [], []
        for threshold in grid:
            metrics = C.direction_metrics(truth, p >= threshold)
            raws.append(metrics["raw_direction_accuracy"])
            balanced.append(metrics["balanced_accuracy"] if metrics["balanced_accuracy"] is not None else -1.0)
        raws = np.asarray(raws)
        balanced = np.asarray(balanced)

        def pick(values: np.ndarray) -> tuple[float, int]:
            best = values.max()
            tied = np.flatnonzero(np.isclose(values, best, atol=1e-12))
            # Among tied optima prefer the threshold closest to the deployed 0.5.
            chosen = tied[np.argmin(np.abs(grid[tied] - 0.5))]
            return float(grid[chosen]), int(len(tied))

        entries = [("threshold_0.5", 0.5, 1)]
        raw_threshold, raw_ties = pick(raws)
        entries.append(("oracle_raw", raw_threshold, raw_ties))
        bal_threshold, bal_ties = pick(balanced)
        entries.append(("oracle_balanced", bal_threshold, bal_ties))

        for label, threshold, ties in entries:
            metrics = C.direction_metrics(truth, p >= threshold)
            rows.append(
                {
                    "model": model, "row": label, "threshold": threshold,
                    "raw_direction_accuracy": metrics["raw_direction_accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "positive_recall": metrics["positive_recall"],
                    "nonpositive_recall": metrics["nonpositive_recall"],
                    "predicted_positive_fraction": metrics["predicted_positive_fraction"],
                    "n_tied_optimal_thresholds": ties,
                    **ORACLE_MARKERS,
                }
            )
    return pd.DataFrame(rows)


def paired_table(daily: pd.DataFrame, reference: str = "T2") -> pd.DataFrame:
    """Day-cluster bootstrap of (model - reference) daily deltas; unit = target day."""
    metrics = ["raw_direction_accuracy", "balanced_accuracy", "positive_recall",
               "nonpositive_recall", "predicted_positive_fraction"]
    base = daily[daily["model"] == reference].set_index("target_day")
    rows = []
    for model in daily["model"].unique():
        if model == reference:
            continue
        other = daily[daily["model"] == model].set_index("target_day")
        row: dict[str, object] = {"model": model, "reference": reference}
        for metric in metrics:
            delta = (other[metric] - base[metric]).dropna()
            if delta.empty:
                continue
            stats = C.day_cluster_bootstrap(delta.to_numpy(dtype=float))
            wins = int((delta > C.TIE_EPS).sum())
            losses = int((delta < -C.TIE_EPS).sum())
            row.update(
                {
                    f"mean_delta_{metric}": stats["mean"],
                    f"median_delta_{metric}": stats["median"],
                    f"ci_low_{metric}": stats["ci_low"],
                    f"ci_high_{metric}": stats["ci_high"],
                    f"ci_includes_zero_{metric}": stats["ci_includes_zero"],
                    f"win_{metric}": wins, f"tie_{metric}": int(len(delta) - wins - losses),
                    f"loss_{metric}": losses,
                }
            )
        for window in C.WINDOWS:
            mask = base.index.map(C.DAY_TO_WINDOW) == window
            delta = (other["raw_direction_accuracy"] - base["raw_direction_accuracy"])[mask].dropna()
            row[f"window_{window}_mean_delta_raw"] = float(delta.mean()) if len(delta) else None
        rows.append(row)
    return pd.DataFrame(rows)


# --- figures ----------------------------------------------------------------
def _bar_labels(axis, bars, fmt="{:.3f}") -> None:
    for bar in bars:
        axis.annotate(fmt.format(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", fontsize=6, rotation=90)


def make_figures(panel, model_summary, window, segment, separation, daily) -> list[str]:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    models = list(model_summary["model"])
    colors = ["#4C72B0" if m.startswith("T") else "#DD8452" for m in models]

    # 1. overall Raw / Balanced
    fig, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(models))
    bars1 = axis.bar(x - 0.2, model_summary["raw_direction_accuracy"], 0.4, label="Raw Direction Accuracy", color="#4C72B0")
    bars2 = axis.bar(x + 0.2, model_summary["balanced_accuracy"], 0.4, label="Balanced Accuracy", color="#DD8452")
    axis.axhline(0.65, color="green", linestyle="--", linewidth=1, label="65% milestone")
    axis.set_xticks(x); axis.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    axis.set_ylim(0, 1); axis.set_ylabel("accuracy"); axis.legend(fontsize=8)
    axis.set_title("E1.5-A overall Raw vs Balanced Direction Accuracy (28 days / 672 slots)")
    _bar_labels(axis, bars1); _bar_labels(axis, bars2)
    fig.tight_layout(); path = C.FIG_DIR / "fig_overall_raw_balanced.png"; fig.savefig(path, dpi=150); plt.close(fig)
    written.append(path.name)

    # 2. W1-W4 Raw with majority baseline
    fig, axis = plt.subplots(figsize=(12, 5))
    windows = list(C.WINDOWS)
    x = np.arange(len(windows)); width = 0.8 / len(models)
    for i, model in enumerate(models):
        values = [window[(window["model"] == model) & (window["window"] == w)]["raw_direction_accuracy"].iloc[0] for w in windows]
        axis.bar(x + i * width - 0.4 + width / 2, values, width, label=model)
    majority = [window[window["window"] == w]["majority_baseline_raw"].iloc[0] for w in windows]
    axis.plot(x, majority, "k--o", markersize=5, label="majority baseline Raw")
    axis.axhline(0.65, color="green", linestyle=":", linewidth=1)
    axis.set_xticks(x); axis.set_xticklabels(windows)
    axis.set_ylim(0, 1); axis.set_ylabel("Raw Direction Accuracy"); axis.legend(fontsize=7, ncol=3)
    axis.set_title("E1.5-A per-window Raw Direction Accuracy vs majority baseline")
    fig.tight_layout(); path = C.FIG_DIR / "fig_window_raw_vs_majority.png"; fig.savefig(path, dpi=150); plt.close(fig)
    written.append(path.name)

    # 3. H1/H2/H3 Raw + Balanced
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    segments = list(C.HOUR_SEGMENTS)
    x = np.arange(len(segments)); width = 0.8 / len(models)
    for axis, metric, title in ((axes[0], "raw_direction_accuracy", "Raw"),
                                (axes[1], "balanced_accuracy", "Balanced")):
        for i, model in enumerate(models):
            values = [segment[(segment["model"] == model) & (segment["hour_segment"] == s)][metric].iloc[0] for s in segments]
            axis.bar(x + i * width - 0.4 + width / 2, values, width, label=model)
        axis.set_xticks(x); axis.set_xticklabels([f"{s} ({C.HOUR_SEGMENTS[s][0]}-{C.HOUR_SEGMENTS[s][1]})" for s in segments])
        axis.set_ylim(0, 1); axis.set_title(f"{title} Direction Accuracy by hour segment")
    axes[0].set_ylabel("accuracy"); axes[1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("E1.5-A hour-segment comparison")
    fig.tight_layout(); path = C.FIG_DIR / "fig_hour_segment_raw_balanced.png"; fig.savefig(path, dpi=150); plt.close(fig)
    written.append(path.name)

    # 4. TafM probability separation
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, scope, names, title in (
        (axes[0], "WINDOW", list(C.WINDOWS), "by window"),
        (axes[1], "HOUR_SEGMENT", list(C.HOUR_SEGMENTS), "by hour segment"),
    ):
        subset = separation[separation["scope"] == scope]
        x = np.arange(len(names)); width = 0.8 / 3
        for i, model in enumerate(["T0", "T1", "T2"]):
            rows = subset[subset["model"] == model].set_index("scope_name")
            pos = [rows.loc[n, "mean_p_given_positive"] for n in names]
            neg = [rows.loc[n, "mean_p_given_nonpositive"] for n in names]
            axis.bar(x + i * width - 0.4 + width / 2, pos, width, label=f"{model} p|y>0", alpha=0.95)
            axis.bar(x + i * width - 0.4 + width / 2, neg, width, label=f"{model} p|y<=0",
                     alpha=0.45, hatch="//", edgecolor="black", linewidth=0.3)
        axis.axhline(0.5, color="red", linestyle="--", linewidth=1)
        axis.set_xticks(x); axis.set_xticklabels(names)
        axis.set_ylim(0, 0.8); axis.set_title(f"TafM probability separation {title}")
    axes[0].set_ylabel("mean p_positive"); axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("E1.5-A TafM probability separation (solid = y>0, hatched = y<=0)")
    fig.tight_layout(); path = C.FIG_DIR / "fig_tafm_probability_separation.png"; fig.savefig(path, dpi=150); plt.close(fig)
    written.append(path.name)

    # 5. day-level Raw curve
    fig, axis = plt.subplots(figsize=(14, 5))
    for model in models:
        rows = daily[daily["model"] == model].sort_values("target_day")
        axis.plot(range(len(rows)), rows["raw_direction_accuracy"], marker="o", markersize=3, label=model)
    boundary = np.cumsum([7, 7, 7])
    for b in boundary:
        axis.axvline(b - 0.5, color="grey", linestyle=":", linewidth=1)
    axis.axhline(0.65, color="green", linestyle="--", linewidth=1, label="65% milestone")
    axis.set_xticks(range(0, 28, 2))
    axis.set_xticklabels([d[5:] for d in sorted(daily["target_day"].unique())][::2], rotation=90, fontsize=7)
    axis.set_ylim(0, 1.05); axis.set_ylabel("daily Raw"); axis.legend(fontsize=7, ncol=4)
    axis.set_title("E1.5-A day-level Raw Direction Accuracy (dotted lines = W1|W2|W3|W4 boundaries)")
    fig.tight_layout(); path = C.FIG_DIR / "fig_daily_raw_curve.png"; fig.savefig(path, dpi=150); plt.close(fig)
    written.append(path.name)

    # 6. day x model heatmap
    pivot = daily.pivot(index="model", columns="target_day", values="raw_direction_accuracy").loc[models]
    fig, axis = plt.subplots(figsize=(15, 4.5))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axis.set_yticks(range(len(pivot.index))); axis.set_yticklabels(pivot.index, fontsize=8)
    axis.set_xticks(range(len(pivot.columns))); axis.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    fig.colorbar(image, ax=axis, label="daily Raw")
    axis.set_title("E1.5-A daily Raw heatmap (model x target day)")
    fig.tight_layout(); path = C.FIG_DIR / "fig_daily_raw_heatmap.png"; fig.savefig(path, dpi=150); plt.close(fig)
    written.append(path.name)
    return written


def main() -> int:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(C.PRED_DIR / "panel.parquet")
    panel["target_day"] = pd.to_datetime(panel["target_day"]).dt.date

    daily = daily_table(panel)
    overall = overall_table(panel, daily)
    window = window_table(panel)
    segment = segment_table(panel)
    prior = prior_gap_table(overall, window)
    separation = separation_table(panel)
    oracle = oracle_table(panel)
    paired = paired_table(daily)

    _rounded(daily).to_csv(C.AUDIT_DIR / "baseline_daily_metrics.csv", index=False)
    _rounded(overall).to_csv(C.AUDIT_DIR / "model_summary.csv", index=False)
    _rounded(window).to_csv(C.AUDIT_DIR / "window_metrics.csv", index=False)
    _rounded(segment).to_csv(C.AUDIT_DIR / "hour_segment_metrics.csv", index=False)
    _rounded(prior).to_csv(C.AUDIT_DIR / "prior_gap.csv", index=False)
    _rounded(separation).to_csv(C.AUDIT_DIR / "probability_separation.csv", index=False)
    _rounded(oracle).to_csv(C.AUDIT_DIR / "oracle_threshold_DIAGNOSTIC_ONLY.csv", index=False)
    _rounded(paired).to_csv(C.AUDIT_DIR / "paired_vs_tafm.csv", index=False)
    figures = make_figures(panel, overall, window, segment, separation, daily)

    # Verification: with exactly 24 slots per day, micro Raw is algebraically identical
    # to macro-day Raw. Any drift means the panel is not slot-balanced.
    worst_identity = float(np.abs(overall["micro_minus_macro_raw"]).max())
    if worst_identity > 1e-9:
        raise ValueError(f"micro/macro Raw identity violated by {worst_identity}")

    # Verification: the oracle row at threshold 0.5 must reproduce the deployed model.
    deployed = overall.set_index("model")["raw_direction_accuracy"]
    for model in ["T0", "T1", "T2"]:
        row = oracle[(oracle["model"] == model) & (oracle["row"] == "threshold_0.5")].iloc[0]
        if abs(row["raw_direction_accuracy"] - deployed[model]) > 1e-9:
            raise ValueError(f"oracle grid does not reproduce deployed Raw for {model}")

    tables = {
        "model_summary": overall.to_dict(orient="records"),
        "overall": overall.to_dict(orient="records"),
        "window": window.to_dict(orient="records"),
        "segment": segment.to_dict(orient="records"),
        "prior_gap": prior.to_dict(orient="records"),
        "separated": separation.to_dict(orient="records"),
        "oracle": oracle.to_dict(orient="records"),
        "paired": paired.to_dict(orient="records"),
        "figures": figures,
        "checks": {
            "micro_macro_identity_max_abs": worst_identity,
            "oracle0.5_reproduces_deployed": True,
            "n_models": int(overall.shape[0]),
            "n_days": int(daily["target_day"].nunique()),
            "slots_per_model": int(panel.groupby("model").size().iloc[0]),
        },
    }
    (C.AUDIT_DIR / "E1_5_tables.json").write_text(json.dumps(tables, indent=2, default=str), encoding="utf-8")

    print(overall[["model", "raw_direction_accuracy", "macro_day_raw", "balanced_accuracy",
                   "positive_recall", "nonpositive_recall", "predicted_positive_fraction",
                   "auc", "brier", "one_class_prediction_days"]].to_string(index=False))
    print("\nfigures:", ", ".join(figures))
    print(f"\nmicro/macro identity max|delta| = {worst_identity:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

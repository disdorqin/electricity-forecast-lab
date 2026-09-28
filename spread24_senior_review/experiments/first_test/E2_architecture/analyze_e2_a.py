"""E2-A big-block aggregation: A0 (reused E1 T2) vs A1 (tabular_only) vs A2 (temporal_only).

A0 is NOT re-run. Its 28-day daily results come from the preserved E1-M2 run records, so the
anchor is literally the E1 T2 artifact rather than a reproduction of it.

All micro metrics go through the canonical `canonical_metrics`, and the paired day-cluster
bootstrap imports E1-mini's own helper, so E2-A numbers are definitionally comparable to E1/E1.5.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC_ROOT = REPO / "src"
for path in (str(SRC_ROOT), str(REPO / "experiments/first_test/E1_mini")):
    if path not in sys.path:
        sys.path.insert(0, path)

from TafM_改进源码.metrics import canonical_metrics  # noqa: E402
from analyze_e1_mini import day_cluster_bootstrap  # noqa: E402

E1_RUNS = REPO / "experiments/first_test/E1_mini/runs"
E2_RUNS = HERE / "runs"
BIG = HERE / "big_block"
FIG = HERE / "figures"
TIE_EPS = 1e-12
ARCHS = ("A0", "A1", "A2")
ARCH_LABEL = {"A0": "A0 full_current", "A1": "A1 tabular_only", "A2": "A2 temporal_only"}
SEGMENTS = {"H1": range(1, 9), "H2": range(9, 17), "H3": range(17, 25)}


def normalize_e1_record(record: dict) -> dict:
    """E1 written records carry telemetry/manifest_fields rather than one manifest block."""
    telemetry, fields = record["telemetry"], record["manifest_fields"]
    return {
        "status": "COMPLETE" if record["status"] == "PASS" else record["status"],
        "architecture_mode": fields.get("architecture_mode", "full_current"),
        "objective_mode": fields.get("objective_mode"),
        "checkpoint_policy": fields.get("checkpoint_policy"),
        "gradient_policy": fields.get("gradient_policy"),
        "train_mode": "stage_a",
        "parameter_count_total": telemetry.get("parameter_count_total"),
        "parameter_count_trainable": telemetry.get("parameter_count_trainable"),
        "best_epoch": telemetry.get("best_epoch"), "stop_epoch": telemetry.get("stop_epoch"),
        "epochs_run": telemetry.get("epochs_run"),
        "stage_a_best_epoch": fields.get("stage_a_best_epoch"), "stage_b_epochs": None,
        "wall_time_total_seconds": telemetry.get("wall_time_total_seconds"),
        "training_epoch_seconds_mean": telemetry.get("training_epoch_seconds_mean"),
        "cuda_peak_memory_bytes": telemetry.get("cuda_peak_memory_bytes"),
        "config_sha256": record.get("config_sha256"),
        "alpha_trajectories": [{"stage": h["stage"], "epoch": h["epoch"],
                                "alpha_dir": h.get("alpha_dir"), "alpha_mag": h.get("alpha_mag")}
                               for h in record.get("training_history", [])]}


def load_records() -> pd.DataFrame:
    rows = []
    for path in sorted(E1_RUNS.glob("E1-M2-*/RUN_RECORD.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "PASS":
            rows.append({"arch": "A0", "window": record["window"], "target_day": record["target_day"],
                         "run_dir": record["raw_run_dir"], "manifest": normalize_e1_record(record),
                         "source": "E1-M2 (reused)"})
    for path in sorted(E2_RUNS.glob("E2-*/RUN_RECORD.json")):
        try:  # a run still being written must not abort the aggregation
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("status") == "PASS":
            rows.append({"arch": record["arch"], "window": record["window"], "target_day": record["target_day"],
                         "run_dir": record["run_dir"], "manifest": record["manifest"],
                         "source": "E2-A Gate C"})
    frame = pd.DataFrame(rows)
    print(f"loaded {len(frame)} runs: " + ", ".join(f"{a}={n}" for a, n in frame.arch.value_counts().items()))
    return frame


def slots_for(record) -> pd.DataFrame:
    frame = pd.read_parquet(Path(record["run_dir"]) / "predictions.parquet")
    return pd.DataFrame({
        "arch": record["arch"], "window": record["window"], "target_day": frame["target_day"].astype(str),
        "hour": frame["hour_business"].astype(int), "y_true": frame["y_true_model"].astype(float),
        "direction_hat": frame["direction_hat"].astype(int), "p_positive": frame["p_positive"].astype(float),
        "magnitude_hat": frame["magnitude_hat"].astype(float)})


def micro(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {}
    result = canonical_metrics(sub["y_true"].to_numpy(), sub["direction_hat"].to_numpy(),
                              sub["magnitude_hat"].to_numpy(),
                              direction_probability=sub["p_positive"].to_numpy(),
                              hour=sub["hour"].to_numpy())
    positives = int(sub["direction_hat"].sum())
    return {"raw": result["raw_direction_accuracy"], "balanced": result["balanced_accuracy"],
            "positive_recall": result["positive_recall"], "nonpositive_recall": result["nonpositive_recall"],
            "auc": result["auc"], "auc_status": result["auc_status"], "brier": result["brier"],
            # canonical_metrics does not emit this one; train.py derives it the same way.
            "predicted_positive_fraction": positives / len(sub), "n_slots": len(sub),
            "true_positive_prevalence": float((sub["y_true"] > 0).mean()),
            "n_true_positive": int((sub["y_true"] > 0).sum()),
            "n_predicted_positive": positives}


def per_day(slots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, window, day), sub in slots.groupby(["arch", "window", "target_day"], sort=True):
        row = {"arch": arch, "architecture": ARCH_LABEL[arch], "window": window, "target_day": day}
        row.update(micro(sub))
        row["one_class_prediction_day"] = int(sub["direction_hat"].nunique() == 1)
        row["positive_recall_zero"] = int((sub["y_true"] > 0).sum() > 0 and (sub[(sub.y_true > 0)].direction_hat == 1).sum() == 0)
        row["nonpositive_recall_zero"] = int((sub["y_true"] <= 0).sum() > 0 and (sub[(sub.y_true <= 0)].direction_hat == 0).sum() == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def window_table(daily: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHS:
        for window in ("W1", "W2", "W3", "W4", "OVERALL"):
            sub = slots[slots.arch == arch] if window == "OVERALL" else slots[(slots.arch == arch) & (slots.window == window)]
            day_sub = daily[daily.arch == arch] if window == "OVERALL" else daily[(daily.arch == arch) & (daily.window == window)]
            row = {"arch": arch, "architecture": ARCH_LABEL[arch], "window": window}
            row.update(micro(sub))
            row["macro_day_raw"] = float(day_sub["raw"].mean())
            row["macro_day_balanced"] = float(day_sub["balanced"].mean())
            row["one_class_prediction_days"] = int(day_sub["one_class_prediction_day"].sum())
            row["positive_recall_zero_days"] = int(day_sub["positive_recall_zero"].sum())
            row["nonpositive_recall_zero_days"] = int(day_sub["nonpositive_recall_zero"].sum())
            row["n_days"] = len(day_sub)
            rows.append(row)
    frame = pd.DataFrame(rows)
    for arch in ARCHS:
        windows = frame[(frame.arch == arch) & (frame.window != "OVERALL")]["raw"]
        frame.loc[(frame.arch == arch) & (frame.window == "OVERALL"), "min_window_raw"] = float(windows.min())
        frame.loc[(frame.arch == arch) & (frame.window == "OVERALL"), "window_raw_std"] = float(windows.std(ddof=0))
    return frame


def hour_segments(slots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arch in ARCHS:
        for segment, hours in SEGMENTS.items():
            sub = slots[(slots.arch == arch) & (slots.hour.isin(list(hours)))]
            row = {"arch": arch, "architecture": ARCH_LABEL[arch], "segment": segment,
                   "hours": f"{min(hours)}-{max(hours)}"}
            row.update(micro(sub))
            rows.append(row)
    return pd.DataFrame(rows)


def paired(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = daily.set_index(["arch", "target_day"])
    metrics = [("raw", True), ("balanced", True), ("positive_recall", True),
               ("nonpositive_recall", True), ("auc", True), ("brier", False),
               ("predicted_positive_fraction", None)]
    rows, summaries = [], []
    for label, a, b in (("A1_minus_A0", "A1", "A0"), ("A2_minus_A0", "A2", "A0")):
        sub_rows = []
        for _, day_row in daily[daily.arch == a].sort_values("target_day").iterrows():
            day = day_row["target_day"]
            ra, rb = indexed.loc[(a, day)], indexed.loc[(b, day)]
            row = {"comparison": label, "arch_a": a, "arch_b": b, "window": day_row["window"], "target_day": day}
            for name, _ in metrics:
                va, vb = ra[name], rb[name]
                row[f"delta_{name}"] = (float(va) - float(vb)) if pd.notna(va) and pd.notna(vb) else np.nan
            for name in ("raw", "balanced"):
                row[f"outcome_{name}"] = ("improved" if row[f"delta_{name}"] > TIE_EPS else
                                          "worsened" if row[f"delta_{name}"] < -TIE_EPS else "tie")
            sub_rows.append(row)
        sub = pd.DataFrame(sub_rows)
        rows.extend(sub_rows)
        summary = {"comparison": label, "arch_a": a, "arch_b": b, "n_days": len(sub)}
        if sub.empty:  # that arch has no completed runs yet
            summaries.append(summary)
            continue
        for name, higher_is_better in metrics:
            deltas = sub[f"delta_{name}"].dropna().to_numpy(dtype=float)
            summary[f"mean_delta_{name}"] = float(np.mean(deltas)) if len(deltas) else np.nan
            if higher_is_better is not None and len(deltas):
                wins = int((deltas > TIE_EPS).sum()) if higher_is_better else int((deltas < -TIE_EPS).sum())
                losses = int((deltas < -TIE_EPS).sum()) if higher_is_better else int((deltas > TIE_EPS).sum())
                summary[f"win_{name}"] = wins
                summary[f"tie_{name}"] = int((np.abs(deltas) <= TIE_EPS).sum())
                summary[f"loss_{name}"] = losses
            if name in {"raw", "balanced", "positive_recall", "nonpositive_recall", "auc"} and len(deltas):
                boot = day_cluster_bootstrap(deltas)
                for key in ("ci_low", "ci_high", "ci_includes_zero"):
                    summary[f"boot_{key}_{name}"] = boot[key]
                summary[f"boot_mean_{name}"] = boot["mean"]
            if name == "raw" and len(deltas):
                summary["bootstrap_seed"] = day_cluster_bootstrap(deltas)["bootstrap_seed"]
                summary["bootstrap_resamples"] = day_cluster_bootstrap(deltas)["bootstrap_resamples"]
                summary["bootstrap_unit"] = "target_day"
        summaries.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def runtime_table(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in records.iterrows():
        manifest = record["manifest"]
        peak = manifest.get("cuda_peak_memory_bytes")
        rows.append({
            "arch": record["arch"], "architecture": ARCH_LABEL[record["arch"]],
            "window": record["window"], "target_day": record["target_day"],
            "architecture_mode": manifest.get("architecture_mode"),
            "objective_mode": manifest.get("objective_mode"),
            "checkpoint_policy": manifest.get("checkpoint_policy"),
            "gradient_policy": manifest.get("gradient_policy"),
            "status": manifest.get("status"),
            "parameter_count_total": manifest.get("parameter_count_total"),
            "parameter_count_trainable": manifest.get("parameter_count_trainable"),
            "best_epoch": manifest.get("best_epoch"), "stop_epoch": manifest.get("stop_epoch"),
            "epochs_run": manifest.get("epochs_run"),
            "wall_time_total_seconds": manifest.get("wall_time_total_seconds"),
            "training_epoch_seconds_mean": manifest.get("training_epoch_seconds_mean"),
            "cuda_peak_memory_mb": (peak / 1024 / 1024) if peak else None,
            "config_sha256": manifest.get("config_sha256")})
    return pd.DataFrame(rows)


def architecture_audit(records: pd.DataFrame) -> pd.DataFrame:
    ownership = json.loads((HERE / "benchmark/gradient_ownership_checks.json").read_text(encoding="utf-8"))
    runtime = runtime_table(records)
    rows = []
    for arch in ARCHS:
        sub = runtime[runtime.arch == arch]
        if sub.empty:  # that arch has no completed runs yet
            rows.append({"arch": arch, "architecture": ARCH_LABEL[arch], "runs": 0,
                         "all_complete": False, "mode_invariant_parameter_count": None,
                         "gradient_ownership_checks": json.dumps(
                             {k: v for k, v in ownership.items() if k.startswith(arch)})})
            continue
        alpha_dir_final, alpha_dir_first, alpha_mag_final = [], [], []
        for _, record in records[records.arch == arch].iterrows():
            trajectory = record["manifest"].get("alpha_trajectories") or []
            if trajectory:
                alpha_dir_first.append(trajectory[0]["alpha_dir"])
                alpha_dir_final.append(trajectory[-1]["alpha_dir"])
                alpha_mag_final.append(trajectory[-1]["alpha_mag"])
        rows.append({
            "arch": arch, "architecture": ARCH_LABEL[arch],
            "architecture_mode_runtime": sorted(set(sub.architecture_mode.dropna())),
            "runs": len(sub), "all_complete": bool((sub.status == "COMPLETE").all()),
            "parameter_count_total": sorted(set(sub.parameter_count_total.dropna())),
            "parameter_count_trainable": sorted(set(sub.parameter_count_trainable.dropna())),
            "mode_invariant_parameter_count": sub.parameter_count_total.nunique() == 1,
            "best_epoch_mean": float(sub.best_epoch.mean()), "best_epoch_min": int(sub.best_epoch.min()),
            "best_epoch_max": int(sub.best_epoch.max()),
            "stop_epoch_mean": float(sub.stop_epoch.mean()), "epochs_run_mean": float(sub.epochs_run.mean()),
            "wall_seconds_mean": float(sub.wall_time_total_seconds.mean()),
            "wall_seconds_total": float(sub.wall_time_total_seconds.sum()),
            "epoch_seconds_mean": float(sub.training_epoch_seconds_mean.mean()),
            "cuda_peak_mb_mean": float(sub.cuda_peak_memory_mb.mean()),
            "cuda_peak_mb_max": float(sub.cuda_peak_memory_mb.max()),
            "alpha_dir_init": float(np.mean(alpha_dir_first)) if alpha_dir_first else None,
            "alpha_dir_final_mean": float(np.mean(alpha_dir_final)) if alpha_dir_final else None,
            "alpha_dir_final_min": float(np.min(alpha_dir_final)) if alpha_dir_final else None,
            "alpha_dir_final_max": float(np.max(alpha_dir_final)) if alpha_dir_final else None,
            "alpha_mag_final_mean": float(np.mean(alpha_mag_final)) if alpha_mag_final else None,
            "gradient_ownership_checks": json.dumps({k: v for k, v in ownership.items() if k.startswith(arch)})})
    return pd.DataFrame(rows)


def write_figures(daily, windows, segments, paired_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"A0": "#1f77b4", "A1": "#d62728", "A2": "#2ca02c"}
    produced = []

    fig, ax = plt.subplots(figsize=(13, 4.5))
    days = sorted(daily.target_day.unique())
    for arch in ARCHS:
        sub = daily[daily.arch == arch].set_index("target_day").reindex(days)
        if sub["raw"].isna().all():  # that arch has no completed runs yet
            continue
        ax.plot(range(len(days)), sub["raw"].to_numpy(dtype=float), marker="o", ms=3.5,
                label=ARCH_LABEL[arch], color=colors[arch], lw=1.4)
    boundaries = [i for i in range(1, len(days)) if days[i][:7] != days[i - 1][:7]]
    for b in boundaries:
        ax.axvline(b - 0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels([d[5:] for d in days], rotation=90, fontsize=7)
    ax.set_ylabel("Raw direction accuracy"); ax.set_ylim(0, 1)
    ax.axhline(0.5, color="black", ls="--", lw=1, label="chance")
    ax.set_title("E2-A daily Raw direction accuracy (28-day DEV panel; single days are NOT a promotion signal)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    path = FIG / "fig_daily_raw.png"; fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(path.name)

    overall = windows[windows.window == "OVERALL"].set_index("arch")
    present = [a for a in ARCHS if a in overall.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(4); width = 0.26
    for i, arch in enumerate(present):
        sub = windows[(windows.arch == arch) & (windows.window != "OVERALL")].set_index("window").reindex(
            ["W1", "W2", "W3", "W4"])
        axes[0].bar(x + i * width, sub["raw"].to_numpy(dtype=float), width, label=ARCH_LABEL[arch], color=colors[arch])
    axes[0].set_xticks(x + width); axes[0].set_xticklabels(["W1 Feb", "W2 Apr", "W3 Jun", "W4 Aug"])
    axes[0].axhline(0.5, color="black", ls="--", lw=1); axes[0].set_ylabel("Raw (pooled)")
    axes[0].set_title("Per-window Raw"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y")
    positions = np.arange(len(present))
    axes[1].bar(positions - 0.2, [overall.loc[a, "raw"] for a in present], 0.2, label="Raw (pooled)", color="#4c72b0")
    axes[1].bar(positions, [overall.loc[a, "balanced"] for a in present], 0.2, label="Balanced", color="#dd8452")
    axes[1].bar(positions + 0.2, [overall.loc[a, "auc"] for a in present], 0.2, label="AUC", color="#55a868")
    axes[1].axhline(0.5, color="black", ls="--", lw=1)
    axes[1].set_xticks(positions); axes[1].set_xticklabels([ARCH_LABEL[a] for a in present], fontsize=8)
    axes[1].set_title("Overall (28 days)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y")
    path = FIG / "fig_window_overall.png"; fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    segs = ["H1", "H2", "H3"]
    x = np.arange(3)
    for i, arch in enumerate(present):
        sub = segments[segments.arch == arch].set_index("segment").reindex(segs)
        axes[0].bar(x + i * 0.26, sub["raw"].to_numpy(dtype=float), 0.26, label=ARCH_LABEL[arch], color=colors[arch])
        axes[1].bar(x + i * 0.26, sub["balanced"].to_numpy(dtype=float), 0.26, label=ARCH_LABEL[arch], color=colors[arch])
    for ax, title in zip(axes, ["Raw by hour segment", "Balanced by hour segment"]):
        ax.set_xticks(x + 0.26); ax.set_xticklabels(["H1 1-8", "H2 9-16", "H3 17-24"])
        ax.axhline(0.5, color="black", ls="--", lw=1); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    path = FIG / "fig_hour_segments.png"; fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(20260924)
    for i, (label, colour) in enumerate((("A1_minus_A0", "#d62728"), ("A2_minus_A0", "#2ca02c"))):
        deltas = paired_rows[paired_rows.comparison == label]["delta_raw"].dropna().to_numpy()
        if len(deltas) == 0:  # that arch has no completed runs yet
            continue
        idx = rng.integers(0, len(deltas), size=(10000, len(deltas)))
        means = deltas[idx].mean(axis=1)
        ax.hist(means, bins=60, alpha=0.55, color=colour,
                label=f"{label} (95% CI [{np.percentile(means, 2.5):+.3f}, {np.percentile(means, 97.5):+.3f}])")
    ax.axvline(0, color="black", ls="--", lw=1.2)
    ax.set_xlabel("bootstrap mean delta Raw (day-cluster, 10000 resamples)")
    ax.set_title("E2-A paired deltas vs A0 anchor"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    path = FIG / "fig_paired_bootstrap.png"; fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    produced.append(path.name)
    return produced


def main():
    BIG.mkdir(parents=True, exist_ok=True)
    records = load_records()
    slots = pd.concat([slots_for(r) for _, r in records.iterrows()], ignore_index=True)
    daily = per_day(slots)
    windows = window_table(daily, slots)
    segments = hour_segments(slots)
    paired_rows, paired_summary = paired(daily)
    runtime = runtime_table(records)
    audit = architecture_audit(records)

    daily.to_csv(BIG / "daily_metrics.csv", index=False)
    windows.to_csv(BIG / "window_metrics.csv", index=False)
    segments.to_csv(BIG / "hour_segment_metrics.csv", index=False)
    paired_rows.to_csv(BIG / "paired_deltas.csv", index=False)
    paired_summary.to_csv(BIG / "paired_summary.csv", index=False)
    runtime.to_csv(BIG / "runtime.csv", index=False)
    audit.to_csv(BIG / "architecture_audit.csv", index=False)

    overall = windows[windows.window == "OVERALL"]
    summary_rows = []
    for arch in ARCHS:
        row = overall[overall.arch == arch].iloc[0]
        summary_rows.append({"arch": arch, "architecture": ARCH_LABEL[arch], "raw": row["raw"],
                             "balanced": row["balanced"], "positive_recall": row["positive_recall"],
                             "nonpositive_recall": row["nonpositive_recall"], "auc": row["auc"],
                             "brier": row["brier"], "predicted_positive_fraction": row["predicted_positive_fraction"],
                             "true_positive_prevalence": row["true_positive_prevalence"],
                             "min_window_raw": row["min_window_raw"], "window_raw_std": row["window_raw_std"],
                             "one_class_prediction_days": row["one_class_prediction_days"],
                             "positive_recall_zero_days": row["positive_recall_zero_days"],
                             "nonpositive_recall_zero_days": row["nonpositive_recall_zero_days"]})
    model_summary = pd.DataFrame(summary_rows)
    model_summary.to_csv(BIG / "model_summary.csv", index=False)

    figures = write_figures(daily, windows, segments, paired_rows)

    print("\n=== OVERALL (pooled 672 slots) ===")
    pd.set_option("display.width", 220)
    print(model_summary[["arch", "raw", "balanced", "positive_recall", "nonpositive_recall",
                         "auc", "brier", "min_window_raw", "window_raw_std"]].to_string(index=False))
    print("\n=== per window Raw ===")
    print(windows.pivot(index="arch", columns="window", values="raw").to_string())
    print("\n=== hour segments Raw / Balanced ===")
    print(segments.pivot(index="arch", columns="segment", values=["raw", "balanced"]).to_string())
    print("\n=== paired vs A0 (day-cluster bootstrap) ===")
    columns = [c for c in paired_summary.columns if c.startswith(("comparison", "mean_delta_raw", "win_raw",
                                                                 "tie_raw", "loss_raw", "boot_ci"))]
    print(paired_summary[columns].to_string(index=False))
    print("\nfigures:", ", ".join(figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

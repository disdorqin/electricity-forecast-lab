"""Render the E1-mini evidence tables into E1_summary.md / E1_GATE.md.

Numbers come only from the CSVs written by analyze_e1_mini.py; the gate text is
assembled here from those numbers, and the final SUPPORTED / NOT_SUPPORTED /
INCONCLUSIVE wording is reviewed by hand before the file is treated as final.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

E1_ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = E1_ROOT / "E1_curves_or_figures"

METRIC_LABELS = [
    ("micro_raw", "micro Raw", "{:.4f}"),
    ("macro_day_raw", "macro-day Raw", "{:.4f}"),
    ("macro_day_raw_std", "macro-day Raw std", "{:.4f}"),
    ("micro_balanced", "micro Balanced", "{:.4f}"),
    ("macro_day_balanced", "macro-day Balanced", "{:.4f}"),
    ("micro_positive_recall", "micro +Recall", "{:.4f}"),
    ("micro_nonpositive_recall", "micro -Recall", "{:.4f}"),
    ("macro_day_positive_recall", "macro-day +Recall", "{:.4f}"),
    ("macro_day_nonpositive_recall", "macro-day -Recall", "{:.4f}"),
    ("micro_predicted_positive_fraction", "micro predicted positive fraction", "{:.4f}"),
    ("micro_auc", "micro AUC", "{:.4f}"),
    ("micro_brier", "micro Brier", "{:.4f}"),
    ("micro_magnitude_mae", "micro Magnitude MAE", "{:.4f}"),
    ("macro_day_magnitude_mae", "macro-day Magnitude MAE", "{:.4f}"),
    ("micro_magnitude_mae_naive", "micro Magnitude MAE naive", "{:.4f}"),
    ("micro_magnitude_skill", "micro Magnitude skill", "{:.4f}"),
    ("macro_day_magnitude_skill", "macro-day Magnitude skill", "{:.4f}"),
    ("all_negative_baseline_raw", "all-negative baseline Raw", "{:.4f}"),
    ("one_class_prediction_days", "one-class prediction days", "{:.0f}"),
    ("positive_recall_zero_days", "+Recall=0 days", "{:.0f}"),
    ("nonpositive_recall_zero_days", "-Recall=0 days", "{:.0f}"),
    ("mean_best_epoch", "mean best epoch", "{:.2f}"),
    ("mean_stop_epoch", "mean stop epoch", "{:.2f}"),
    ("mean_wall_time_seconds", "mean wall time (s)", "{:.3f}"),
    ("total_wall_time_seconds", "total wall time (s)", "{:.1f}"),
]


def md_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]], index_header: str = "") -> str:
    header = [index_header] + [label for _, label, _ in columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for _, row in frame.iterrows():
        cells = []
        for key, _, fmt in columns:
            value = row.get(key)
            cells.append("N/A" if value is None or (isinstance(value, float) and not np.isfinite(value))
                         else fmt.format(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    daily = pd.read_csv(E1_ROOT / "daily_metrics.csv")
    summary = pd.read_csv(E1_ROOT / "variant_summary.csv")
    windows = pd.read_csv(E1_ROOT / "window_metrics.csv")
    paired = pd.read_csv(E1_ROOT / "paired_deltas.csv")
    paired_summary = pd.read_csv(E1_ROOT / "paired_summary.csv")
    checkpoints = pd.read_csv(E1_ROOT / "checkpoint_diagnostics.csv")
    runtime = pd.read_csv(E1_ROOT / "runtime.csv")

    variant_col = [("variant", "variant", "{}")] + METRIC_LABELS
    summary_md = md_table(summary, variant_col)
    window_md = md_table(
        windows,
        [("variant", "variant", "{}"), ("window", "window", "{}"), ("n_days", "days", "{:.0f}"),
         ("micro_raw", "micro Raw", "{:.4f}"), ("macro_day_raw", "macro-day Raw", "{:.4f}"),
         ("macro_day_raw_std", "std", "{:.4f}"), ("raw_min", "min", "{:.4f}"), ("raw_max", "max", "{:.4f}"),
         ("micro_balanced", "micro Bal", "{:.4f}"),
         ("micro_positive_recall", "micro +R", "{:.4f}"), ("micro_nonpositive_recall", "micro -R", "{:.4f}"),
         ("macro_day_magnitude_mae", "macro-day Mag MAE", "{:.3f}"),
         ("macro_day_magnitude_skill", "macro-day Mag skill", "{:.4f}"),
         ("macro_day_predicted_positive_fraction", "ppf", "{:.4f}"),
         ("one_class_prediction_days", "1-class days", "{:.0f}"),
         ("positive_recall_zero_days", "+R=0 days", "{:.0f}"),
         ("nonpositive_recall_zero_days", "-R=0 days", "{:.0f}"),
         ("mean_best_epoch", "mean best ep", "{:.2f}")])

    pairing_cols = [("comparison", "comparison", "{}"),
                    ("mean_delta_raw", "mean ΔRaw", "{:+.4f}"),
                    ("median_delta_raw", "median ΔRaw", "{:+.4f}"),
                    ("boot_ci_low_raw", "ΔRaw CI low", "{:+.4f}"),
                    ("boot_ci_high_raw", "ΔRaw CI high", "{:+.4f}"),
                    ("win_raw", "W/T/L Raw", "{:.0f}"), ("tie_raw", "", "{:.0f}"), ("loss_raw", "", "{:.0f}"),
                    ("mean_delta_balanced", "mean ΔBal", "{:+.4f}"),
                    ("win_balanced", "W/T/L Bal", "{:.0f}"), ("tie_balanced", "", "{:.0f}"),
                    ("loss_balanced", "", "{:.0f}"),
                    ("mean_delta_positive_recall", "mean Δ+R", "{:+.4f}"),
                    ("win_positive_recall", "W/T/L +R", "{:.0f}"), ("tie_positive_recall", "", "{:.0f}"),
                    ("loss_positive_recall", "", "{:.0f}"),
                    ("mean_delta_nonpositive_recall", "mean Δ-R", "{:+.4f}"),
                    ("win_nonpositive_recall", "W/T/L -R", "{:.0f}"), ("tie_nonpositive_recall", "", "{:.0f}"),
                    ("loss_nonpositive_recall", "", "{:.0f}"),
                    ("mean_delta_magnitude_mae", "mean ΔMag MAE", "{:+.4f}"),
                    ("win_magnitude_mae", "W/T/L MAE", "{:.0f}"), ("tie_magnitude_mae", "", "{:.0f}"),
                    ("loss_magnitude_mae", "", "{:.0f}"),
                    ("mean_delta_predicted_positive_fraction", "mean Δppf", "{:+.4f}"),
                    ("mean_delta_best_epoch", "mean Δbest epoch", "{:+.3f}"),
                    ("mean_delta_runtime_seconds", "mean Δruntime (s)", "{:+.3f}")]
    paired_md = md_table(paired_summary, pairing_cols)

    window_delta_cols = [("comparison", "comparison", "{}")]
    for window in ("W1", "W2", "W3", "W4"):
        window_delta_cols.append((f"window_{window}_mean_delta_raw", f"{window} ΔRaw", "{:+.4f}"))
    for window in ("W1", "W2", "W3", "W4"):
        window_delta_cols.append((f"window_{window}_mean_delta_magnitude_mae", f"{window} ΔMag MAE", "{:+.3f}"))
    paired_window_md = md_table(paired_summary, window_delta_cols)

    ckpt_summary = checkpoints.groupby("variant").agg(
        n_runs=("run_id", "count"),
        mean_selected_monitor_raw=("selected_monitor_raw", "mean"),
        mean_selected_monitor_balanced=("selected_monitor_balanced", "mean"),
        mean_selected_monitor_pos_recall=("selected_monitor_positive_recall", "mean"),
        mean_selected_monitor_neg_recall=("selected_monitor_nonpositive_recall", "mean"),
        mean_selected_monitor_L_dir=("selected_monitor_L_dir", "mean"),
        mean_selected_monitor_mag_mae=("selected_monitor_magnitude_mae", "mean"),
        mean_monitor_raw_minus_anchor=("monitor_raw_minus_anchor", "mean"),
        mean_selected_minus_anchor=("selected_minus_anchor", "mean"),
        collapse_warning_runs=("target_day_collapse_warning", "sum"),
        one_class_runs=("target_day_one_class_prediction_collapse", "sum"),
        mean_best_epoch=("best_epoch", "mean"),
    ).reset_index()
    ckpt_md = md_table(ckpt_summary, [("variant", "variant", "{}"), ("n_runs", "runs", "{:.0f}"),
                                      ("mean_selected_monitor_raw", "sel monitor Raw", "{:.4f}"),
                                      ("mean_monitor_raw_minus_anchor", "monitor Raw − anchor", "{:+.4f}"),
                                      ("mean_selected_minus_anchor", "selected − anchor", "{:+.4f}"),
                                      ("mean_selected_monitor_balanced", "sel monitor Bal", "{:.4f}"),
                                      ("mean_selected_monitor_pos_recall", "sel monitor +R", "{:.4f}"),
                                      ("mean_selected_monitor_neg_recall", "sel monitor -R", "{:.4f}"),
                                      ("mean_selected_monitor_L_dir", "sel monitor L_dir", "{:.4f}"),
                                      ("mean_selected_monitor_mag_mae", "sel monitor Mag MAE", "{:.3f}"),
                                      ("one_class_runs", "1-class runs", "{:.0f}"),
                                      ("collapse_warning_runs", "collapse-flag runs", "{:.0f}"),
                                      ("mean_best_epoch", "mean best epoch", "{:.2f}")])

    guardrail_divergence = checkpoints[(checkpoints["variant"] == "M0")
                                       & (checkpoints["monitor_raw_minus_anchor"] < -1e-9)]
    guardrail_note = (
        f"M0 guardrail selected a checkpoint below its own Raw anchor in "
        f"{len(guardrail_divergence)} / {int((checkpoints['variant'] == 'M0').sum())} runs "
        f"(mean gap {guardrail_divergence['monitor_raw_minus_anchor'].mean():+.4f} monitor Raw)."
        if len(guardrail_divergence) else "M0 guardrail never selected a checkpoint below its Raw anchor.")

    runtime_summary = runtime.groupby("variant").agg(
        runs=("run_id", "count"),
        wall_total=("wall_time_total_seconds", "sum"),
        wall_mean=("wall_time_total_seconds", "mean"),
        wall_min=("wall_time_total_seconds", "min"),
        wall_max=("wall_time_total_seconds", "max"),
        wall_std=("wall_time_total_seconds", "std"),
        epoch_mean=("training_epoch_seconds_mean", "mean"),
        epochs_mean=("epochs_run", "mean"),
        epochs_max=("epochs_run", "max"),
        pred_latency_mean=("prediction_latency_ms", "mean"),
        gpu_peak_max=("cuda_peak_memory_bytes", "max"),
        params=("parameter_count_total", "max"),
    ).reset_index()
    runtime_md = md_table(runtime_summary, [("variant", "variant", "{}"), ("runs", "runs", "{:.0f}"),
                                            ("wall_total", "total wall (s)", "{:.1f}"),
                                            ("wall_mean", "mean wall (s)", "{:.3f}"),
                                            ("wall_min", "min", "{:.3f}"), ("wall_max", "max", "{:.3f}"),
                                            ("wall_std", "std", "{:.3f}"),
                                            ("epoch_mean", "mean epoch (s)", "{:.4f}"),
                                            ("epochs_mean", "mean epochs", "{:.2f}"),
                                            ("epochs_max", "max epochs", "{:.0f}"),
                                            ("pred_latency_mean", "pred latency (ms)", "{:.2f}"),
                                            ("gpu_peak_max", "GPU peak (bytes)", "{:.0f}"),
                                            ("params", "params", "{:.0f}")])

    context = {
        "summary_md": summary_md, "window_md": window_md, "paired_md": paired_md,
        "paired_window_md": paired_window_md, "ckpt_md": ckpt_md, "runtime_md": runtime_md,
        "guardrail_note": guardrail_note,
        "n_runs": int(len(daily)), "n_days": int(daily["target_day"].nunique()),
    }
    (E1_ROOT / "E1_report_tables.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in context.items() if k in {"n_runs", "n_days", "guardrail_note"}}, indent=2))
    print(summary_md)
    print()
    print(paired_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

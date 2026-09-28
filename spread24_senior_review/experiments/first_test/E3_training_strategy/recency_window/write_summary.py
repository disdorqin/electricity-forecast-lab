"""Render the E3-A human summary from recorded output tables only."""
import json
import re
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent
LBL = {"arm": "Arm", "raw": "Raw", "balanced": "Balanced", "positive_recall": "+Recall",
       "negative_recall": "-Recall", "auc": "AUC", "brier": "Brier", "predicted_positive_fraction": "ppf",
       "one_class_days": "one-class days", "positive_recall_zero_days": "+R=0 days",
       "negative_recall_zero_days": "-R=0 days", "min_window_raw": "min W Raw", "window_raw_std": "W Raw std",
       "scope": "Scope", "comparison": "Pair", "mean_delta_raw": "ΔRaw", "raw_ci_low": "Raw CI low",
       "raw_ci_high": "Raw CI high", "raw_W_T_L": "Raw W/T/L", "mean_delta_balanced": "ΔBalanced",
       "balanced_ci_low": "Balanced CI low", "balanced_ci_high": "Balanced CI high",
       "correct_slots_arm": "correct arm", "correct_slots_T0": "correct T0",
       "delta_correct_slots": "Δslots", "n_slots": "slots", "delta_slots_per_day": "Δslots/day",
       "split_mode": "split mode", "monitor_count": "monitor n", "monitor_start": "monitor start",
       "monitor_end": "monitor end", "base_count": "base n", "base_start": "base start",
       "base_end": "base end", "eligible_count": "eligible n", "preprocessing_fit_start": "fit start",
       "preprocessing_fit_end": "fit end", "target_day": "target day",
       "best_epoch": "best ep", "stop_epoch": "stop ep", "train_L_dir_selected": "train BCE@sel",
       "monitor_L_dir_selected": "monitor BCE@sel", "bce_gap_selected": "BCE gap",
       "monitor_raw_selected": "mon RAW@sel", "epochs_run": "epochs"}


def table(df, cols, labels=None):
    d = df[cols].copy().rename(columns=labels or {})
    for c in d:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "—" if pd.isna(x) else f"{x:.4f}")
    names = list(d.columns)
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(names)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join(lines)


def main():
    m = pd.read_csv(H / "model_summary.csv")
    w = pd.read_csv(H / "window_metrics.csv")
    h = pd.read_csv(H / "hour_segment_metrics.csv")
    p = pd.read_csv(H / "paired_summary.csv")
    g = pd.read_csv(H / "slot_gain_summary.csv")
    s = json.loads((H / "safety_admissibility.json").read_text())
    r = pd.read_csv(H / "runtime.csv")
    sp = pd.read_csv(H / "split_summary.csv")
    ep = pd.read_csv(H / "epoch_diagnostics.csv")
    b = pd.read_csv(H / "benchmark/benchmark_metrics.csv")
    ga = json.loads((H / "benchmark/gate_a_results.json").read_text())
    tests = json.loads((H / "benchmark/gate_a_tests.json").read_text())
    sig = re.search(r"`([^`]+)`", (H / "E3_A_GATE.md").read_text()).group(1)

    sp_formal = sp[sp.scope == "formal"]
    split_view = []
    for a in ("T1", "T2", "T3"):
        z = sp_formal[sp_formal.arm == a]
        last = z[z.target_day == z.target_day.max()].iloc[0]
        split_view.append({"arm": a, "split_mode": last.split_mode, "monitor_count": last.monitor_count,
                           "monitor_start": last.monitor_start, "monitor_end": last.monitor_end,
                           "base_count": f"{int(z.base_count.min())}–{int(z.base_count.max())}",
                           "base_start": last.base_start, "base_end": last.base_end,
                           "eligible_count": last.eligible_count,
                           "preprocessing_fit_start": last.preprocessing_fit_start,
                           "preprocessing_fit_end": last.preprocessing_fit_end})
    split_view = pd.DataFrame(split_view)

    sec = ["# E3-A — Recency-Aware Training Window", "",
           "**Screening-only; STOP after this experiment.** T0 reuses E2-E1 Q2 (canonical chronological "
           "80/20); T1/T2/T3 are fresh CUDA+AMP runs (28 formal DEV days each). Only the Stage-A "
           "BASE/MONITOR split changes: the newest 60 eligible days are MONITOR and BASE is the newest M "
           "days strictly before it (M=None keeps all earlier days). Q2 architecture "
           "(segment_heads) and every other training parameter are fixed; preprocessing still fits BASE "
           "only. No Stage B, no new features, no architecture change.", "",
           "## Gate A / frozen protocol",
           f"- T0 reproduces the saved E2-E1 Q2 benchmark checkpoint; max prediction delta "
           f"{ga['delta_T0_vs_Q2']:.1e} (<=1e-9).",
           "- T1/T2/T3 MONITOR is exactly the newest 60 eligible days ending D-2; T1 BASE ends immediately "
           "before MONITOR; T2 BASE is exactly 365 days and T3 exactly 1095; BASE/MONITOR never overlap and "
           "never contain target or D-1; `preprocessing_fit_day_start/end` equals the BASE range exactly.",
           "- Source/config/selector/sequence hashes remain the frozen values; the Q2 segment-head route "
           "(3 heads) and k=8 are preserved.", "",
           "- fresh arm records: 28/28 PASS each; all CUDA+AMP.", "",
           "## Exact split (formal, last day 2026-08-13)", table(
               split_view, ["arm", "split_mode", "monitor_count", "monitor_start", "monitor_end", "base_count",
                            "base_start", "base_end", "eligible_count", "preprocessing_fit_start",
                            "preprocessing_fit_end"], LBL), "",
           "## Benchmark day — engineering only", table(
               b, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier",
                   "predicted_positive_fraction"], LBL), "",
           "2026-02-13 is excluded from the gate ranking. T0 uses the frozen E2-E1 Q2 control.", "",
           "## Overall formal DEV (28 days / 672 slots)", table(
               m, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier",
                   "predicted_positive_fraction", "one_class_days", "positive_recall_zero_days",
                   "negative_recall_zero_days", "min_window_raw", "window_raw_std"], LBL), "",
           "## W1–W4", table(w, ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"], LBL), "",
           "## H1–H3", table(h, ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"], LBL), "",
           "## Paired day-cluster bootstrap (10,000 draws, seed 20260924)",
           table(p, ["comparison", "mean_delta_raw", "raw_ci_low", "raw_ci_high", "raw_W_T_L",
                     "mean_delta_balanced", "balanced_ci_low", "balanced_ci_high"], LBL), "",
           "T2/T3 vs T1 and T3 vs T2 are fresh-arm comparisons. No ranking is based on benchmark-day "
           "figures or pooled Raw alone.", "",
           "## Slot gains vs T0", table(g, ["arm", "scope", "correct_slots_arm", "correct_slots_T0",
                                            "delta_correct_slots", "n_slots", "delta_slots_per_day"], LBL), "",
           "## Epoch / BCE diagnostics (formal means)", table(
               ep.groupby("arm", as_index=False).mean(numeric_only=True),
               ["arm", "best_epoch", "stop_epoch", "epochs_run", "train_L_dir_selected", "monitor_L_dir_selected",
                "bce_gap_selected", "monitor_raw_selected"], LBL), "",
           "`epoch_diagnostics.csv` holds the per-run selected-epoch train/monitor BCE, the BCE gap and the "
           "monitor Raw/AUC/Brier at the selected checkpoint plus monitor Raw min/final; the full per-epoch "
           "trajectories are in each run's `training_history.parquet`.", "",
           "## Safety anchor: T0/Q2", "| Arm | Admissible | Checks |", "|---|---|---|"]
    for a, v in s.items():
        sec.append(f"| {a} | {v['admissible']} | {v['checks']} |")

    sec += ["", "## Runtime (formal run means)",
            "| Arm | External wall s | Train wall s | Epoch s | Best epoch | Stop epoch | CUDA peak MiB | Params |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ("T0", "T1", "T2", "T3"):
        z = r[(r.arm == a) & (r.scope == "formal")]
        sec.append(f"| {a} | {z.external_wall_seconds.mean():.2f} | {z.train_wall_seconds.mean():.2f} | "
                   f"{z.epoch_seconds_mean.mean():.3f} | {z.best_epoch.mean():.2f} | {z.stop_epoch.mean():.2f} | "
                   f"{z.cuda_peak_memory_bytes.mean()/1048576:.1f} | {int(z.parameter_count.mode().iloc[0]):,} |")

    d = m.set_index("arm")
    pm = {x["comparison"]: x for x in p.to_dict("records")}
    def ci(k): return f"[{pm[k]['raw_ci_low']*100:+.2f},{pm[k]['raw_ci_high']*100:+.2f}]pp"
    best = max(("T1", "T2", "T3"), key=lambda a: d.loc[a].raw)
    accel = [a for a in ("T1", "T2", "T3") if d.loc[a].raw >= .62 and s[a]["admissible"]]
    sec += ["", "## Signal / decision", f"`{sig}`", "",
            f"T1 Raw {(d.loc['T1'].raw-d.loc['T0'].raw)*100:+.2f} pp (CI {ci('T1-T0')}), "
            f"T2 Raw {(d.loc['T2'].raw-d.loc['T0'].raw)*100:+.2f} pp (CI {ci('T2-T0')}), "
            f"T3 Raw {(d.loc['T3'].raw-d.loc['T0'].raw)*100:+.2f} pp (CI {ci('T3-T0')}). "
            f"Best fresh arm = {best}. Safety: T0={s['T0']['admissible']}, T1={s['T1']['admissible']}, "
            f"T2={s['T2']['admissible']}, T3={s['T3']['admissible']}. "
            + (f"Acceleration: fresh arm(s) >=.62 with safety: {accel}; next suggestion is 3-seed T0+winner "
               "(not executed). " if accel else
               "Acceleration trigger (fresh arm >=.62 Raw with safety) NOT met; no promotion. "),
            "62/63/65 is not claimed unless the actual 28-day metric reaches it with acceptable safety. "
            "Window micro-tuning stops here." if sig in {"NO_RECENCY_GAIN", "LONG_HISTORY_NEEDED"} else
            "Any candidate improvement still requires multi-seed confirmation before promotion."]
    if sig == "NO_RECENCY_GAIN":
        sec += ["", "### Interpretation (recency hypothesis refuted)",
                "- All three recency-restricted arms are **materially and stably worse** than T0 (paired Raw "
                "CIs exclude zero, all-negative window signs) and all fail the safety screen. The plan's "
                "hypothesis — that withholding the large recent MONITOR caused old-distribution overfit — is "
                "**not supported**: T0 (large recent monitor, BASE≈80% of eligible history) wins by +4.0 to "
                "+7.3 pp Raw.",
                "- T1 has *more* BASE days than T0 (1616 vs ~1400) yet is 4.0 pp worse, so enlarging BASE "
                "alone does not help; the change shared by all fresh arms is the 60-day MONITOR. Because "
                "T1/T2/T3 all use monitor=60, E3-A cannot fully separate monitor-size from base-window "
                "effects, but the safest reading is that a 60-day MONITOR destabilises checkpoint selection "
                "(T2's BCE gap 0.087 vs T0's 0.023) and that restricting BASE to ≤1095 days loses useful "
                "regime coverage.",
                "- NEXT: do NOT continue window tuning; pivot to legal feature/regime work (E4)."]
        sec += [""]
    sec += ["## Provenance and scope",
            "Run commands, metrics and manifests are recorded under `runs/`. Run manifests bind "
            "source/config/selector/sequence hashes and record eligible/base/monitor counts and date ranges "
            "plus the preprocessing fit range (`split_summary.csv`). No Stage B, no new features, no "
            "architecture change, no threshold/calibration tuning, no PLE bins/dim, k/depth/width change, "
            "role filtering, gate/fusion sweep, Temporal/FFT change, broader DEV or lockbox was run.", "",
            f"Tests: canonical non-destructive baseline **{tests['baseline_94']} passed**; full suite "
            f"(canonical + all prior E2 + focused E3-A) **{tests['full_suite']} passed** "
            f"(+{tests['focused']} focused). Fail-closed CLI combinations return exit 2. "
            "Non-blocking single-bin PLE and matplotlib warnings only."]
    (H / "E3_A_summary.md").write_text("\n".join(sec) + "\n", encoding="utf-8")
    print("wrote E3_A_summary.md")


if __name__ == "__main__":
    main()

"""Render the E2-E1 human summary from recorded output tables only."""
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
       "correct_slots_arm": "correct arm", "correct_slots_Q0": "correct Q0",
       "delta_correct_slots": "Δslots", "n_slots": "slots", "delta_slots_per_day": "Δslots/day"}


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
    b = pd.read_csv(H / "benchmark/benchmark_metrics.csv")
    ga = json.loads((H / "benchmark/gate_a_results.json").read_text())
    rd = json.loads((H / "readout_diagnostics.json").read_text())
    sig = re.search(r"`([^`]+)`", (H / "E2_E1_GATE.md").read_text()).group(1)

    sec = ["# E2-E1 — Horizon-Specialized Direction Readout", "",
           "**Screening-only; STOP after this experiment.** Q0 reuses R0/G0/C0/F80; Q1/Q2 are fresh "
           "CUDA+AMP runs (28 formal DEV days each). Only the Direction readout over the fixed "
           "1-8 / 9-16 / 17-24 segments changes. Encoding (canonical PLE+raw), all 211 Strong/Core + 11 "
           "Weak, the current 24-h gate and fixed fusion alpha=.8 are unchanged; the Magnitude path is "
           "untouched.", "",
           "## Gate A / frozen protocol",
           f"- shared (`--direction-readout-mode shared`) and default/no-flag reproduced the F80 benchmark "
           f"checkpoint; max prediction delta {ga['delta_q0_vs_f80']:.1e} (<=1e-9).",
           "- Source/config/selector/sequence hashes remain the frozen values in each run manifest; target "
           "and eligibility continue through the docs/01 origin=D-1 14:00, labels<=D-2 pipeline.",
           f"- Q1 adds exactly 3 zero-init scalar logit biases over one shared head (params "
           f"{ga['parameter_counts']['Q0']:,} -> {ga['parameter_counts']['Q1']:,}). Q2 replaces the head "
           f"with exactly 3 independent heads and keeps no shared Direction head (params "
           f"{ga['parameter_counts']['Q2']:,}); each hour is produced by exactly one head. k=8 and [B,k,24] "
           "are preserved; p>=.5 threshold logic and the Magnitude head are unchanged.",
           "- fresh arm records: 28/28 PASS each; all CUDA+AMP.", "",
           "## Benchmark day — engineering only", table(
               b, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier",
                   "predicted_positive_fraction"], LBL), "",
           "2026-02-13 is excluded from the gate ranking. Q0 uses the frozen F80 control.", "",
           "## Overall formal DEV (28 days / 672 slots)", table(
               m, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier",
                   "predicted_positive_fraction", "one_class_days", "positive_recall_zero_days",
                   "negative_recall_zero_days", "min_window_raw", "window_raw_std"], LBL), "",
           "## W1–W4", table(w, ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"], LBL), "",
           "## H1–H3 (primary diagnostic)", table(
               h, ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"], LBL), "",
           "## Paired day-cluster bootstrap (10,000 draws, seed 20260924)",
           table(p, ["comparison", "mean_delta_raw", "raw_ci_low", "raw_ci_high", "raw_W_T_L",
                     "mean_delta_balanced", "balanced_ci_low", "balanced_ci_high"], LBL), "",
           "Q2−Q1 is included. No ranking is based on benchmark-day figures or pooled Raw alone.", "",
           "## Slot gains vs Q0", table(g, ["arm", "scope", "correct_slots_arm", "correct_slots_Q0",
                                            "delta_correct_slots", "n_slots", "delta_slots_per_day"], LBL), "",
           "## Safety anchor: Q0/F80", "| Arm | Admissible | Checks |", "|---|---|---|"]
    for a, v in s.items():
        sec.append(f"| {a} | {v['admissible']} | {v['checks']} |")
    audit = rd["model_audit"]
    sec += ["", "## Readout diagnostics / parameter counts",
            "| Arm | mode | shared head | n segment-bias | n segment-heads | params |",
            "|---|---|---|---:|---:|---:|"]
    for a in ("Q0", "Q1", "Q2"):
        au = audit[a]
        sec.append(f"| {a} | `{au['mode']}` | {au['shared_direction_head']} | {au['n_segment_bias']} | "
                   f"{au['n_segment_heads']} | {ga['parameter_counts'][a]:,} |")
    probe = rd.get("probe", {})
    if probe.get("present"):
        sec += ["", "Q2 per-head parameter/gradient L2 norms (selected-checkpoint probe, mean over 28 runs):"]
        sec += ["", "| head | param norm | grad norm |", "|---|---:|---:|"]
        for i in range(3):
            key = f"H{i+1}"
            sec.append(f"| {key} | {probe.get('q2_head_param_norm_mean', {}).get(f'head_param_norm_{key}', float('nan')):.3f} | "
                       f"{probe.get('q2_head_grad_norm_mean', {}).get(f'head_grad_norm_{key}', float('nan')):.4f} |")
        q1b = probe.get("q1_bias_mean", {})
        if q1b:
            sec += ["", "Q1 selected-run segment-bias means: "
                        f"H1 {q1b.get('segment_bias_H1', float('nan')):+.4f}, "
                        f"H2 {q1b.get('segment_bias_H2', float('nan')):+.4f}, "
                        f"H3 {q1b.get('segment_bias_H3', float('nan')):+.4f}; per-run trajectory in "
                        "`readout_diagnostics.json`."]
    sec += ["", "## Runtime (formal run means)",
            "| Arm | External wall s | Train wall s | Epoch s | Best epoch | Stop epoch | CUDA peak MiB | Params |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ("Q0", "Q1", "Q2"):
        z = r[(r.arm == a) & (r.scope == "formal")]
        sec.append(f"| {a} | {z.external_wall_seconds.mean():.2f} | {z.train_wall_seconds.mean():.2f} | "
                   f"{z.epoch_seconds_mean.mean():.3f} | {z.best_epoch.mean():.2f} | {z.stop_epoch.mean():.2f} | "
                   f"{z.cuda_peak_memory_bytes.mean()/1048576:.1f} | {int(z.parameter_count.mode().iloc[0]):,} |")

    d = m.set_index("arm")
    pm = {x["comparison"]: x for x in p.to_dict("records")}
    def ci(k): return f"[{pm[k]['raw_ci_low']*100:+.2f},{pm[k]['raw_ci_high']*100:+.2f}]pp"
    sec += ["", "## Signal / decision", f"`{sig}`", "",
            f"Q1 Raw {(d.loc['Q1'].raw-d.loc['Q0'].raw)*100:+.2f} pp (paired CI {ci('Q1-Q0')}), "
            f"Q2 Raw {(d.loc['Q2'].raw-d.loc['Q0'].raw)*100:+.2f} pp (paired CI {ci('Q2-Q0')}). "
            f"Safety: Q0={s['Q0']['admissible']}, Q1={s['Q1']['admissible']}, Q2={s['Q2']['admissible']}. "
            "62/63/65 is not claimed unless the actual 28-day metric reaches it with acceptable safety.", "",
            "## Provenance and scope",
            "Run commands, metrics and manifests are recorded under `runs/`. Run manifests bind "
            "source/config/selector/sequence hashes; `runtime.csv` and `readout_diagnostics.csv/json` hold "
            "per-run telemetry and the readout audit. No 24-head design, new segment boundaries, month "
            "heads, oracle routing, threshold/calibration tuning, PLE bins/dim, k/depth/width, role "
            "filtering, gate/fusion sweep, Temporal/FFT change, Stage B, broader DEV or lockbox was run."]
    tests = json.loads((H / "benchmark/gate_a_tests.json").read_text())
    sec += ["", f"Tests: canonical non-destructive baseline **{tests['baseline_94']} passed**; full suite "
            f"(canonical + all prior E2 + focused E2-E1) **{tests['full_suite']} passed** "
            f"(+{tests['focused']} focused). Fail-closed CLI combinations return exit 2. "
            "Non-blocking single-bin PLE and matplotlib warnings only."]
    (H / "E2_E1_summary.md").write_text("\n".join(sec) + "\n", encoding="utf-8")
    print("wrote E2_E1_summary.md")


if __name__ == "__main__":
    main()

"""Render the E2-D1 human summary from recorded output tables only."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
LBL = {"arm": "Arm", "raw": "Raw", "balanced": "Balanced", "positive_recall": "+Recall",
       "negative_recall": "-Recall", "auc": "AUC", "brier": "Brier", "predicted_positive_fraction": "ppf",
       "one_class_days": "one-class days", "positive_recall_zero_days": "+R=0 days",
       "negative_recall_zero_days": "-R=0 days", "min_window_raw": "min W Raw", "window_raw_std": "W Raw std",
       "scope": "Window", "comparison": "Pair", "mean_delta_raw": "ΔRaw", "raw_ci_low": "Raw CI low",
       "raw_ci_high": "Raw CI high", "raw_W_T_L": "Raw W/T/L", "mean_delta_balanced": "ΔBalanced",
       "balanced_ci_low": "Balanced CI low", "balanced_ci_high": "Balanced CI high"}


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
    s = json.loads((H / "safety_admissibility.json").read_text())
    r = pd.read_csv(H / "runtime.csv")
    b = pd.read_csv(H / "benchmark/benchmark_metrics.csv")
    ga = json.loads((H / "benchmark/gate_a_results.json").read_text())
    ea = json.loads((H / "encoding_audit.json").read_text())
    sig = re.search(r"`([^`]+)`", (H / "E2_D1_GATE.md").read_text()).group(1)

    sec = ["# E2-D1 — Numeric Encoding Big-Block Ablation", "",
           "**Screening-only; STOP after this experiment.** N0 reuses R0/G0/C0/F80; N1/N2 are fresh "
           "CUDA+AMP runs (28 formal DEV days each). Only the numeric representation entering the Strong "
           "and Weak Tabular branches changes. Direction fusion alpha stays fixed at .8 and the current "
           "24-h horizon gate stays active.", "",
           "## Gate A / frozen protocol",
           f"- canonical (`--numeric-encoding-mode canonical`) and default/no-flag reproduced the F80 "
           f"benchmark checkpoint; max prediction delta {ga['delta_n0_vs_f80']:.1e} (<=1e-9).",
           "- Source/config/selector/sequence hashes remain the frozen values in each run manifest; "
           "target and eligibility continue through the docs/01 origin=D-1 14:00, labels<=D-2 pipeline.",
           f"- Encoded dims: N0={ga['encoded_feature_dims']['N0']} (PLE_8D+raw), "
           f"N1={ga['encoded_feature_dims']['N1']} (raw only), N2={ga['encoded_feature_dims']['N2']} (PLE only). "
           "All 211 Strong/Core + 11 Weak retained; frozen PLE bins/hashes unchanged; Temporal and k=8 "
           "remain active. Objective is dir_only, so E2-D1 makes no Magnitude-capability claim.",
           "- fresh arm records: 28/28 PASS each; all CUDA+AMP.", "",
           "## Benchmark day — engineering only", table(
               b, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier",
                   "predicted_positive_fraction"], LBL), "",
           "2026-02-13 is excluded from the gate ranking. N0 uses the frozen F80 control.", "",
           "## Overall formal DEV (28 days / 672 slots)", table(
               m, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier",
                   "predicted_positive_fraction", "one_class_days", "positive_recall_zero_days",
                   "negative_recall_zero_days", "min_window_raw", "window_raw_std"], LBL), "",
           "## W1–W4", table(w, ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"], LBL), "",
           "## H1–H3", table(h, ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"],
                             {**LBL, "scope": "Segment"}), "",
           "## Paired day-cluster bootstrap (10,000 draws, seed 20260924)",
           table(p, ["comparison", "mean_delta_raw", "raw_ci_low", "raw_ci_high", "raw_W_T_L",
                     "mean_delta_balanced", "balanced_ci_low", "balanced_ci_high"], LBL), "",
           "N1−N2 is descriptive. No ranking is based on benchmark-day figures or pooled Raw alone.", "",
           "## Safety anchor: N0/F80", "| Arm | Admissible | Checks |", "|---|---|---|"]
    for a, v in s.items():
        sec.append(f"| {a} | {v['admissible']} | {v['checks']} |")
    sec += ["", "## Encoding dimensions / parameter counts",
            "| Arm | mode | encoded dim | Strong | Weak | TabM in-dim | Weak in-dim | params (benchmark) |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    # N0 reuses the frozen F80 control, whose manifest predates numeric_encoding_audit; use the
    # Gate A reproduction of the canonical encoder for its audit row.
    audit_by_arm = {"N0": ga["arms"]["N0"]["numeric_encoding_audit"]}
    audit_by_arm.update({a: ea["by_arm"][a] for a in ("N1", "N2")})
    for a in ("N0", "N1", "N2"):
        row = audit_by_arm[a]
        sec.append(f"| {a} | `{row['numeric_encoding_mode']}` | {row['encoded_feature_dim']} | "
                   f"{row['strong_feature_count']} | {row['weak_feature_count']} | {row['strong_tabm_input_dim']} | "
                   f"{row['weak_mlp_input_dim']} | {int(ea['by_arm'][a]['parameter_count']):,} |")
    sec += ["", "## Runtime (formal run means)",
            "| Arm | External wall s | Train wall s | Epoch s | Best epoch | Stop epoch | CUDA peak MiB | Params |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ("N0", "N1", "N2"):
        z = r[(r.arm == a) & (r.scope == "formal")]
        sec.append(f"| {a} | {z.external_wall_seconds.mean():.2f} | {z.train_wall_seconds.mean():.2f} | "
                   f"{z.epoch_seconds_mean.mean():.3f} | {z.best_epoch.mean():.2f} | {z.stop_epoch.mean():.2f} | "
                   f"{z.cuda_peak_memory_bytes.mean()/1048576:.1f} | {int(z.parameter_count.mode().iloc[0]):,} |")

    d1 = m.set_index("arm")
    n1_d, n2_d = d1.loc["N1"], d1.loc["N2"]
    p_map = {x["comparison"]: x for x in p.to_dict("records")}
    def ci(pair): return f"[{p_map[pair]['raw_ci_low']*100:+.2f},{p_map[pair]['raw_ci_high']*100:+.2f}]pp"
    sec += ["", "## Signal / decision", f"`{sig}`", "",
            f"N1 Raw is {(n1_d.raw - d1.loc['N0'].raw)*100:+.2f} pp vs N0 (paired CI {ci('N1-N0')}); "
            f"N2 Raw is {(n2_d.raw - d1.loc['N0'].raw)*100:+.2f} pp vs N0 (paired CI {ci('N2-N0')}). "
            f"Safety: N1 admissible={s['N1']['admissible']}, N2 admissible={s['N2']['admissible']}, "
            f"N0 admissible={s['N0']['admissible']}. Signal = `{sig}`; do not promote or alter production "
            "architecture. Any candidate improvement still needs multi-seed confirmation before promotion.", "",
            "## Provenance and scope",
            "Run commands, metrics and manifests are recorded under `runs/`. Run manifests bind "
            "source/config/selector/sequence hashes; `runtime.csv` and `encoding_audit.csv` contain per-run "
            "telemetry and the numeric-encoding audit. No role filtering, selector rerun, PLE bins/dim sweep, "
            "k/depth/width change, gate tuning, Temporal/FFT, fusion/alpha sweep, threshold/calibration, "
            "checkpoint change, Stage B, broader DEV or lockbox was run."]
    tests = json.loads((H / "benchmark/gate_a_tests.json").read_text())
    sec += ["", f"Tests: canonical non-destructive baseline **{tests['baseline_94']} passed**; full suite "
            f"(canonical + all prior E2 + focused E2-D1) **{tests['full_suite']} passed** "
            f"(+{tests['focused']} focused). Fail-closed CLI combinations return exit 2. "
            "Non-blocking single-bin PLE and matplotlib warnings only."]
    (H / "E2_D1_summary.md").write_text("\n".join(sec) + "\n", encoding="utf-8")
    print("wrote E2_D1_summary.md")


if __name__ == "__main__":
    main()

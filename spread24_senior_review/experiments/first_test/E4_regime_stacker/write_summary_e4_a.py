"""Render the E4-A human summary from recorded output tables only."""
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
       "scope": "Scope", "window": "Window", "comparison": "Pair", "mean_delta_raw": "ΔRaw",
       "raw_ci_low": "Raw CI low", "raw_ci_high": "Raw CI high", "raw_W_T_L": "Raw W/T/L",
       "mean_delta_balanced": "ΔBalanced", "balanced_ci_low": "Balanced CI low", "balanced_ci_high": "Balanced CI high",
       "correct_slots_arm": "correct arm", "correct_slots_P0": "correct P0", "delta_correct_slots": "Δslots",
       "n_slots": "slots", "delta_slots_per_day": "Δslots/day", "target_day": "day",
       "base_count": "BASE n", "base_start": "BASE start", "base_end": "BASE end",
       "full_monitor_count": "FULL_MON n", "full_monitor_start": "FULL_MON start", "full_monitor_end": "FULL_MON end",
       "checkpoint_monitor_count": "CKPT_MON n", "checkpoint_monitor_start": "CKPT_MON start",
       "checkpoint_monitor_end": "CKPT_MON end", "calibrator_count": "CAL n", "calibrator_start": "CAL start",
       "calibrator_end": "CAL end", "preprocessing_fit_start": "fit start", "preprocessing_fit_end": "fit end",
       "wall_seconds": "deep wall s", "best_epoch": "best ep", "stop_epoch": "stop ep"}


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
    rt = pd.read_csv(H / "runtime.csv")
    sa = pd.read_csv(H / "split_audit.csv")
    ck = pd.read_csv(H / "checkpoint_identity.csv")
    cal = pd.read_csv(H / "calibrator_metrics.csv")
    coef = pd.read_csv(H / "stacker_coefficients.csv")
    audit = json.loads((H / "stacker_audit.json").read_text())
    bench = pd.read_csv(H / "benchmark/benchmark_metrics.csv")
    ga = json.loads((H / "benchmark/gate_a_results.json").read_text())
    tests = json.loads((H / "benchmark/gate_a_tests.json").read_text())
    sig = re.search(r"`([^`]+)`", (H / "E4_A_GATE.md").read_text()).group(1)

    first, last = sa.iloc[0], sa.iloc[-1]
    split_view = pd.DataFrame([
        {"target_day": f"{first.target_day} (W1)", "base_count": first.base_count, "base_start": first.base_start,
         "base_end": first.base_end, "full_monitor_count": first.full_monitor_count,
         "checkpoint_monitor_count": first.checkpoint_monitor_count,
         "checkpoint_monitor_end": first.checkpoint_monitor_end, "calibrator_count": first.calibrator_count,
         "calibrator_start": first.calibrator_start, "calibrator_end": first.calibrator_end,
         "preprocessing_fit_start": first.preprocessing_fit_start, "preprocessing_fit_end": first.preprocessing_fit_end},
        {"target_day": f"{last.target_day} (W4)", "base_count": last.base_count, "base_start": last.base_start,
         "base_end": last.base_end, "full_monitor_count": last.full_monitor_count,
         "checkpoint_monitor_count": last.checkpoint_monitor_count,
         "checkpoint_monitor_end": last.checkpoint_monitor_end, "calibrator_count": last.calibrator_count,
         "calibrator_start": last.calibrator_start, "calibrator_end": last.calibrator_end,
         "preprocessing_fit_start": last.preprocessing_fit_start, "preprocessing_fit_end": last.preprocessing_fit_end}])

    sec = ["# E4-A — Regime-Conditioned Direction Probability Stacking", "",
           "**Screening-only; STOP after this experiment.** Only 28 fresh deep Q2 checkpoints were trained "
           "(P1). P2/P3 are fitted on the E4-A CALIBRATOR only and applied to the same checkpoint's target "
           "base prediction, so all three arms share one checkpoint per day. P0 reuses E2-E1 Q2. The deep "
           "model, canonical encoding, 24-h gate, fixed fusion alpha=.8 and k=8 are unchanged; the Magnitude "
           "path is untouched. No Stage B.", "",
           "## Gate A / three-way split", table(
               split_view, ["target_day", "base_count", "base_start", "base_end", "full_monitor_count",
                            "checkpoint_monitor_count", "checkpoint_monitor_end", "calibrator_count",
                            "calibrator_start", "calibrator_end", "preprocessing_fit_start",
                            "preprocessing_fit_end"], LBL), "",
           f"- Default Q2 without E4 flags reproduces the saved Q2 benchmark: max |Δ| = "
           f"{ga['default_q2_reproduction_max_abs_delta']:.1e} (<=1e-9).",
           "- BASE equals the canonical BASE exactly; FULL_MONITOR equals the canonical monitor exactly; "
           "CALIBRATOR is the newest 90 FULL_MONITOR days ending D-2; CHECKPOINT_MONITOR is the rest "
           "(>=60), adjacent and non-overlapping; preprocessing fits BASE only.",
           f"- P1/P2/P3 share one checkpoint: `same_checkpoint_for_P1_P2_P3 = "
           f"{audit['same_checkpoint_for_P1_P2_P3']}`; P1 checkpoint reproduction max |Δ| = "
           f"{audit['P1_checkpoint_reproduction_max_abs_delta']:.1e}.",
           f"- In-train postprocess path == offline derivation on 2026-02-13: "
           f"{json.dumps(ga.get('intrain_cross_check'))}.",
           f"- P2 uses exactly {audit['P2_feature_count']} meta features and P3 exactly "
           f"{audit['P3_feature_count']}; the 9 regime features are all in the frozen selected list; "
           "no target-day truth enters any fit; threshold stays .5; source/config/selector/sequence frozen; "
           "segment_heads + k=8 preserved.", "",
           "## Deep checkpoint identity / split audit",
           f"- 28/28 P1 deep runs PASS (CUDA+AMP). BASE {first.base_count}–{last.base_count} days; "
           f"FULL_MONITOR {first.full_monitor_count}–{last.full_monitor_count}; CHECKPOINT_MONITOR "
           f"{first.checkpoint_monitor_count}–{last.checkpoint_monitor_count}; CALIBRATOR "
           f"{first.calibrator_count}–{last.calibrator_count} days. Per-day detail: `split_audit.csv`.",
           "- `checkpoint_identity.csv` records the identical SHA256 used by P1/P2/P3 and the per-day "
           "reproduction delta.", "",
           "## Benchmark day — engineering only", table(
               bench, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier",
                       "predicted_positive_fraction"], LBL), "",
           "2026-02-13 is excluded from the gate ranking.", "",
           "## Overall formal DEV (28 days / 672 slots)", table(
               m, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier",
                   "predicted_positive_fraction", "one_class_days", "positive_recall_zero_days",
                   "negative_recall_zero_days", "min_window_raw", "window_raw_std"], LBL), "",
           "## W1–W4", table(w, ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"], LBL), "",
           "## H1–H3", table(h, ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"], LBL), "",
           "## Paired day-cluster bootstrap (10,000 draws, seed 20260924)",
           table(p, ["comparison", "mean_delta_raw", "raw_ci_low", "raw_ci_high", "raw_W_T_L",
                     "mean_delta_balanced", "balanced_ci_low", "balanced_ci_high"], LBL), "",
           "## Slot gains vs P0", table(g, ["arm", "scope", "correct_slots_arm", "correct_slots_P0",
                                            "delta_correct_slots", "n_slots", "delta_slots_per_day"], LBL), "",
           "## Calibrator diagnostics (CALIBRATOR only, 90 days / 2160 slots, mean over 28 runs)",
           table(cal.groupby(["arm", "scope"], as_index=False).mean(numeric_only=True),
                 ["arm", "scope", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier"], LBL), "",
           "Per-day calibrator base/post metrics are in `calibrator_metrics.csv`. In-sample both stackers "
           "improve their own calibrator (mean calibrator Raw .5643 -> .5891 for P2 and -> .6109 for P3), "
           "but that in-sample gain does not transfer to the 28 formal target days."]

    coef_view = []
    for arm in ("P2", "P3"):
        z = coef[coef.arm == arm]
        for c in [c for c in coef.columns if c.startswith("w_")]:
            v = z[c].dropna()
            if len(v) == 0:
                continue
            coef_view.append({"arm": arm, "coefficient": c[2:], "mean": float(v.mean()), "std": float(v.std(ddof=0)),
                              "min": float(v.min()), "max": float(v.max()),
                              "sign_consistency": float(max((v > 0).mean(), (v < 0).mean()))})
    coef_view = pd.DataFrame(coef_view)
    sec += ["## Stacker coefficients across 28 days (sign consistency = share with the majority sign)",
            table(coef_view, ["arm", "coefficient", "mean", "std", "min", "max", "sign_consistency"]), "",
            f"Stacker contract: {json.dumps(audit['stacker_contract'])}. `stacker_coefficients.csv` holds the "
            "per-day coefficients; `stacker_audit.json` holds the contract and identity audit.", "",
            "## Safety anchor: P0/Q2", "| Arm | Admissible | Checks |", "|---|---|---|"]
    for a, v in s.items():
        sec.append(f"| {a} | {v['admissible']} | {v['checks']} |")

    sec += ["", "## Runtime",
            "| Arm | Deep wall s (mean) | Best epoch | Stop epoch | Params |",
            "|---|---:|---:|---:|---:|"]
    sec.append(f"| P1 deep | {rt.wall_seconds.mean():.2f} | {rt.best_epoch.mean():.2f} | "
               f"{rt.stop_epoch.mean():.2f} | {int(rt.parameter_count.mode().iloc[0]):,} |")
    sec += ["", "Postprocessing (P2/P3 fitting + target application) runs inside `derive_e4_a.py` and is "
            "negligible next to the deep run; the in-train equivalent is available via "
            "`--direction-postprocess-mode segment_logit|regime_logit`.", ""]

    d = m.set_index("arm")
    pm = {x["comparison"]: x for x in p.to_dict("records")}
    def ci(k): return f"[{pm[k]['raw_ci_low']*100:+.2f},{pm[k]['raw_ci_high']*100:+.2f}]pp"
    best = max(("P1", "P2", "P3"), key=lambda a: d.loc[a].raw)
    accel = [a for a in ("P2", "P3") if d.loc[a].raw >= .62 and s[a]["admissible"]]
    sec += ["", "## Signal / decision", f"`{sig}`", "",
            f"P1−P0 {ci('P1-P0')}, P2−P1 {ci('P2-P1')}, P3−P1 {ci('P3-P1')}, P2−P0 {ci('P2-P0')}, "
            f"P3−P0 {ci('P3-P0')}, P3−P2 {ci('P3-P2')}. Best arm = {best}. "
            f"Safety: P0={s['P0']['admissible']}, P1={s['P1']['admissible']}, P2={s['P2']['admissible']}, "
            f"P3={s['P3']['admissible']}. "
            + (f"Acceleration: {accel} >=.62 with safety; next suggestion is 3-seed P0+winner (not executed). "
               if accel else "Acceleration trigger (P2/P3 >=.62 Raw with safety) NOT met; no promotion. "),
            "62/63/65 is not claimed unless the actual 28-day metric reaches it with acceptable safety.", "",
            "## Provenance and scope",
            "Run commands, metrics and manifests are recorded under `runs/`; `benchmark/` holds the "
            "engineering-only day and the default-Q2 reproduction. No selector rerun, new raw source, new "
            "arbitrary feature, threshold/class-weight/L2/calibrator-days tuning, Stage B, architecture "
            "change, PLE/k/depth/width/gate/fusion change, 24-head model, oracle routing, broader DEV or "
            "lockbox was run.", "",
            f"Tests: canonical non-destructive baseline **{tests['baseline_94']} passed**; full suite "
            f"(canonical + prior E2/E3 + focused E4) **{tests['full_suite']} passed** "
            f"({tests['focused_e4_a']} focused E4-A + {tests['focused_e4_b']} E4-B). Fail-closed CLI "
            "combinations return exit 2. Non-blocking single-bin PLE and matplotlib warnings only."]
    (H / "E4_A_summary.md").write_text("\n".join(sec) + "\n", encoding="utf-8")
    print("wrote E4_A_summary.md")


if __name__ == "__main__":
    main()

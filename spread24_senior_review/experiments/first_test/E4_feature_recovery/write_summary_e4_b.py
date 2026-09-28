"""E4-B final report generator: E4_B_summary.md (16-section pre-registered report).

Consumes the plan29 §18 artifact set produced by analyze_e4_b.py, family_diag_e4_b.py, gate_a_e4_b.py
and benchmark_e4_b.py. All metrics are taken from the frozen W1-W4 panel; the 2026-02-13 benchmark is
engineering-only and is excluded from any ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent

CHANGED = [
    "src/TafM_改进源码/train.py  (+feature_recovery_profile param; +build_experiment_feature_manifest; in-memory experiment manifest; fail-closed)",
    "src/run_tabm_v21.py  (+--feature-recovery-profile CLI flag routed to train_target_day)",
    "src/TafM_改进源码/tests/test_e4_b_feature_recovery.py  (new focused contract tests)",
    "experiments/first_test/E4_feature_recovery/run_e4_b.py / analyze_e4_b.py / gate_a_e4_b.py / write_summary_e4_b.py / benchmark_e4_b.py / probe_e4_b.py / family_diag_e4_b.py",
]

# Mechanistic groups -> component literature features (plan29 §3 / family_diag_e4_b.py GROUPS).
GROUPS = {
    "A_renewable_adjusted_error": [
        "fcast_renewable_adjusted_direct_load",
        "err_renewable_adjusted_direct_28d_mean", "err_renewable_adjusted_direct_28d_std",
        "err_renewable_adjusted_direct_28d_q10", "err_renewable_adjusted_direct_28d_q90",
        "delta_renewable_adjusted_direct_28d_mean", "delta_renewable_adjusted_direct_28d_std",
        "delta_renewable_adjusted_direct_28d_q90"],
    "B_ramp2": ["ramp2_load", "ramp2_solar", "ramp2_renewable", "ramp2_residual_load"],
    "C_regime_flags": [
        "regime_high_residual_load_renew", "regime_low_residual_load_renew",
        "regime_high_renewable_share", "regime_low_renewable_share",
        "regime_high_bidding_space_ratio", "regime_low_bidding_space_ratio"],
}


def md_table(df, cols=None, fmt=None):
    cols = cols or list(df.columns)
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if c in fmt and v is not None and str(v) != "nan":
                vals.append(fmt[c].format(v))
            else:
                vals.append("" if str(v) in ("nan", "None") else str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([head, sep, *rows])


def _next_rule(signal: str) -> str:
    return {
        "DOMAIN_GUIDED_RECOVERY": "F1 safety-admissible and materially/stably improves F0. STOP after E4-B; next = immediate 3-seed F0+winner; no feature micro-tuning first. If >=.62 freeze Q2 + feature profile and broaden DEV.",
        "BROAD_SELECTOR_BOTTLENECK": "F2 safety-admissible and materially/stably improves F0 and clearly exceeds F1. Next = family attribution on remaining 19 Noise features; consider 3-seed confirmation.",
        "RECOVERY_PROMISING_UNPROVEN": "F1 or F2 gains >=~2pp or reach >=.62 with safety, but paired/cross-window evidence not established. Next = 3 seeds + matched-size random recovery controls from remaining Noise pool.",
        "SELECTOR_ROBUST": "Neither F1 nor F2 improves materially; both remain close/admissible. Conclusion: frozen XGB selector is robust for Q2; stop selector reopening.",
        "RECOVERY_HARMFUL": "Recovered features materially worsen Q2 or fail safety. Next = E5 recurrent-regime / similar-day sample selection or weighting.",
        "NO_CLEAR_RECOVERY_EFFECT": "Mixed/noisy result. No promotion; next = E5 sample selection/weighting, not recency windows.",
    }.get(signal, "See gate flags for interpretation.")


def main():
    over = pd.read_csv(H / "model_summary.csv")
    win = pd.read_csv(H / "window_metrics.csv")
    hour = pd.read_csv(H / "hour_segment_metrics.csv")
    paired = pd.read_csv(H / "paired_summary.csv")
    slot = pd.read_csv(H / "slot_gain_summary.csv")
    safety = json.loads((H / "safety_admissibility.json").read_text(encoding="utf-8"))
    gate = (H / "E4_B_GATE.md").read_text(encoding="utf-8")
    signal = gate.split("`")[1] if "`" in gate else "UNKNOWN"
    prof = pd.read_csv(H / "feature_profile_audit.csv")
    elig = pd.read_csv(H / "eligibility_identity.csv")

    comp_raw = pd.read_csv(H / "component_diagnostics.csv")
    comp = comp_raw.groupby("arm").agg(
        parameter_count=("parameter_count", "mean"),
        strong_count=("strong_count", "mean"),
        weak_count=("weak_count", "mean"),
        weak_mlp_input_dim=("weak_input_dim", "mean"),
        best_epoch=("best_epoch", "mean")).reset_index()

    L = []
    L.append("# E4-B — Frozen Selector Reopening via Weak Feature Recovery\n")
    L.append("STATUS=AUTO_FINAL_REPORT  PARENT=E4-A  ANCHOR=E2-E1_Q2_SEGMENT_HEADS  DATE=2026-09-26\n")
    L.append("## 1. FILES_CHANGED\n")
    for c in CHANGED:
        L.append(f"- {c}")
    L.append("\n## 2. TESTS\n")
    L.append("- Focused E4-B contract tests: `src/TafM_改进源码/tests/test_e4_b_feature_recovery.py` (Gate A 1-13, 18-20).")
    L.append("- Gate A (20 items): `benchmark/gate_a_results.json` + `benchmark/gate_a_tests.json` + `benchmark/gate_a_failclosed.txt`.")
    L.append("- Code-level checks include identical eligible-index SETS for all 28 formal days across F0/F1/F2.\n")
    L.append("## 3. FEATURE_PROFILE_AUDIT\n")
    L.append(md_table(prof, ["arm", "feature_recovery_profile", "selected_feature_count", "strong_count", "weak_count", "recovered_count", "recovered_role"]))
    L.append("\n")
    L.append("## 4. ELIGIBILITY_IDENTITY\n")
    n_ident = int(elig["identical_across_profiles"].sum()) if "identical_across_profiles" in elig.columns else "?"
    L.append(f"- `store.eligibility` returns **identical eligible-index SETS** for selected222 / literature240 / all259 on **{n_ident}/{len(elig)}** formal target days (Gate A item 13). Recovered features carry no quarantine on any in-window day, so widening the input set does not change the training/eval day set.\n")
    L.append("## 5. BENCHMARK (2026-02-13 engineering only)\n")
    bm = H / "benchmark" / "benchmark_2026-02-13.csv"
    if bm.exists():
        b = pd.read_csv(bm)
        cols = [c for c in ["arm", "feature_recovery_profile", "raw", "balanced", "auc", "brier",
                            "parameter_count", "best_epoch", "stop_epoch", "selected_feature_count",
                            "strong_count", "weak_count", "recovered_count", "wall_seconds", "device", "amp",
                            "experiment_feature_profile_sha256", "selector_sha256"] if c in b.columns]
        L.append(md_table(b, cols, fmt={c: "{:.4f}" for c in ["raw", "balanced", "auc", "brier"]}))
        L.append("\n- F0 reuses the E2-E1 Q2 run (selected222 == frozen selector). F1/F2 are fresh deep Q2 trainings under the canonical frozen-split route with only the experiment feature-recovery profile changed. No ranking is derived from the benchmark.\n")
    else:
        L.append("- F0 reuses the E2-E1 Q2 run; F1/F2 are fresh deep Q2 trainings under the canonical frozen-split route with only the experiment feature-recovery profile changed. No ranking derived from benchmark.\n")
    L.append("## 6. F0 / F1 / F2 OVERALL\n")
    L.append(md_table(over, ["arm", "raw", "balanced", "positive_recall", "negative_recall", "auc", "brier", "min_window_raw", "window_raw_std"],
                      fmt={c: "{:.4f}" for c in ["raw", "balanced", "positive_recall", "negative_recall", "auc", "brier", "min_window_raw", "window_raw_std"]}))
    L.append("\n")
    L.append("## 7. W1-W4 WINDOW METRICS (Raw)\n")
    w = win.pivot(index="scope", columns="arm", values="raw").reindex(["W1", "W2", "W3", "W4"])
    L.append(md_table(w.reset_index(), ["scope", "F0", "F1", "F2"], fmt={c: "{:.4f}" for c in ["F0", "F1", "F2"]}))
    L.append("\n")
    L.append("## 8. H1-H3 HOUR-SEGMENT METRICS (Raw)\n")
    h = hour.pivot(index="scope", columns="arm", values="raw").reindex(["H1", "H2", "H3"])
    L.append(md_table(h.reset_index(), ["scope", "F0", "F1", "F2"], fmt={c: "{:.4f}" for c in ["F0", "F1", "F2"]}))
    L.append("\n")
    L.append("## 9. PAIRED COMPARISONS (day-cluster bootstrap 10k, seed 20260924)\n")
    L.append(md_table(paired, ["comparison", "mean_delta_raw", "raw_ci_low", "raw_ci_high", "raw_ci_excludes_zero", "raw_W_T_L"],
                      fmt={"mean_delta_raw": "{:.4f}", "raw_ci_low": "{:.4f}", "raw_ci_high": "{:.4f}"}))
    L.append("\n")
    L.append("## 10. SLOT GAINS vs F0 (correct slots)\n")
    sg = slot[slot.scope == "overall"][["arm", "delta_correct_slots", "delta_slots_per_day"]]
    L.append(md_table(sg, ["arm", "delta_correct_slots", "delta_slots_per_day"], fmt={"delta_correct_slots": "{:.0f}", "delta_slots_per_day": "{:.2f}"}))
    L.append("\n")
    L.append("## 11. SAFETY ADMISSIBILITY (anchor F0)\n")
    for a in ("F0", "F1", "F2"):
        if a in safety:
            s = safety[a]
            L.append(f"- {a}: admissible={s['admissible']} checks={s['checks']}")
    L.append("\n")
    L.append("## 12. FAMILY DIAGNOSTICS (F1 recovered literature features)\n")
    fam = H / "family_diagnostics.csv"
    if fam.exists():
        f = pd.read_csv(fam)
        agg = f.groupby("group").agg(
            n_features=("n_features", "first"),
            perm_mean_abs_dp=("perm_mean_abs_dp", "mean"),
            perm_flip_fraction=("perm_flip_fraction", "mean"),
            occ_mean_abs_dp=("occ_mean_abs_dp", "mean"),
            occ_flip_fraction=("occ_flip_fraction", "mean")).reset_index()
        agg["literature_features"] = agg["group"].map(lambda g: ", ".join(GROUPS.get(g, [])))
        L.append(md_table(agg, ["group", "n_features", "literature_features", "perm_mean_abs_dp", "perm_flip_fraction", "occ_mean_abs_dp", "occ_flip_fraction"],
                         fmt={c: "{:.4f}" for c in ["perm_mean_abs_dp", "perm_flip_fraction", "occ_mean_abs_dp", "occ_flip_fraction"]}))
        L.append("\nPermutation = shuffle the group's 24 slot values (per column); occlusion = zero the group columns. "
                 "Higher |Δp_positive| / flip fraction ⇒ stronger leverage of the recovered feature group on F1's own target-day inputs.\n")
    else:
        L.append("- family_diagnostics.csv not produced (run family_diag_e4_b.py).\n")
    L.append("## 13. COMPONENT / PARAMETER / RUNTIME\n")
    L.append(md_table(comp, ["arm", "parameter_count", "strong_count", "weak_count", "weak_mlp_input_dim", "best_epoch"],
                      fmt={"parameter_count": "{:.0f}", "best_epoch": "{:.1f}"}))
    L.append("\n")
    L.append("## 14. E4_B_GATE FEATURE_RECOVERY_SIGNAL\n")
    L.append(f"\n**E4_B_GATE FEATURE_RECOVERY_SIGNAL = `{signal}`**\n")
    L.append("## 15. NEXT ACCELERATION RULE\n")
    L.append(_next_rule(signal) + "\n")
    L.append("## 16. UNEXECUTED LATER WORK\n")
    L.append("- E5 (if E4-B fails): recurrent-regime / similar-day sample selection or weighting using legal target-day fundamental state; NOT recency windows or model micro-tuning.\n")
    L.append("- If RECOVERY_PROMISING_UNPROVEN: confirmation must include 3 seeds + matched-size random recovery controls from the remaining Noise pool before promotion.\n")
    L.append("- If BROAD_SELECTOR_BOTTLENECK: next family attribution on the remaining 19 Noise features.\n")
    L.append("- No Stage B, no threshold tuning, no alpha/gate/PLE/k/depth/width changes, no new raw source, no per-day oracle feature profile.\n")

    (H / "E4_B_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote E4_B_summary.md with signal", signal)


if __name__ == "__main__":
    main()

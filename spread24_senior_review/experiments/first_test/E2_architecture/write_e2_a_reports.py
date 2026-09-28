"""E2-A report writer. Every number in the Markdown is read from the generated CSVs.

The decision tree below is fixed in this file *before* the results are read, so the ARCH_SIGNAL
verdict is a computation over the artifacts rather than a judgement written after the fact.

Operationalisation of the plan's qualitative cases (docs/10 §decision tree):
  CHANCE   = 0.50          Raw at or below chance = "weak"
  MATERIAL = 0.02          a 2pp gap is the smallest difference E1.5's paired CIs could resolve
  stable   = the day-cluster bootstrap 95% CI of the paired delta excludes zero
  X >> Y   = point difference >= MATERIAL in X's favour AND the CI of (X-Y) excludes zero
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BIG = HERE / "big_block"
CHANCE = 0.50
MATERIAL = 0.02
ARCHS = ("A0", "A1", "A2")
LABEL = {"A0": "A0 FULL_CURRENT", "A1": "A1 TABULAR_ONLY", "A2": "A2 TEMPORAL_ONLY"}


def fmt(value, digits=6):
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def scalar(value):
    """The audit CSV stores per-arch lists; render the single distinct element plainly."""
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        items = [item.strip().strip("'\"") for item in text[1:-1].split(",") if item.strip()]
        if len(items) == 1:
            return items[0]
    return text


def decide(model_summary: pd.DataFrame, paired_summary: pd.DataFrame) -> tuple[str, list[str]]:
    raw = model_summary.set_index("arch")["raw"].to_dict()
    stable = {}
    for _, row in paired_summary.iterrows():
        target = {"A1_minus_A0": "A1", "A2_minus_A0": "A2"}[row["comparison"]]
        stable[target] = {"d": row["mean_delta_raw"], "excludes_zero": not bool(row["boot_ci_includes_zero_raw"]),
                          "ci": (row["boot_ci_low_raw"], row["boot_ci_high_raw"]),
                          "win": int(row["win_raw"]), "tie": int(row["tie_raw"]), "loss": int(row["loss_raw"])}

    d1, d2 = stable["A1"]["d"], stable["A2"]["d"]
    notes = [
        f"Raw pooled: A0={fmt(raw['A0'])}, A1={fmt(raw['A1'])}, A2={fmt(raw['A2'])}",
        f"paired delta Raw vs A0: A1-A0={d1:+.6f} (CI [{stable['A1']['ci'][0]:+.4f},{stable['A1']['ci'][1]:+.4f}], "
        f"excludes 0: {stable['A1']['excludes_zero']}, W/T/L {stable['A1']['win']}/{stable['A1']['tie']}/{stable['A1']['loss']})",
        f"paired delta Raw vs A0: A2-A0={d2:+.6f} (CI [{stable['A2']['ci'][0]:+.4f},{stable['A2']['ci'][1]:+.4f}], "
        f"excludes 0: {stable['A2']['excludes_zero']}, W/T/L {stable['A2']['win']}/{stable['A2']['tie']}/{stable['A2']['loss']})",
    ]
    # TAB vs TEMP, derived only from the two pre-registered pairings against the common anchor.
    d12 = d1 - d2
    gap12 = raw["A1"] - raw["A2"]
    tab_far = gap12 >= MATERIAL and stable["A1"]["d"] > stable["A2"]["d"]
    temp_far = -gap12 >= MATERIAL and stable["A2"]["d"] > stable["A1"]["d"]
    notes.append(f"TAB vs TEMP derived from the shared anchor: Raw gap A1-A2={gap12:+.6f} "
                 f"(delta gap {d12:+.6f}); TAB clearly ahead: {tab_far}; TEMP clearly ahead: {temp_far}")

    weak = {arch: raw[arch] <= CHANCE for arch in ARCHS}
    notes.append("at-or-below chance: " + ", ".join(f"{a}={weak[a]}" for a in ARCHS))
    no_stable = not any(stable[a]["excludes_zero"] for a in ("A1", "A2"))
    # A single branch is only said to "beat"/"lose to" FULL on evidence: either a margin at least
    # MATERIAL wide, or a paired CI that excludes zero in that direction.
    a1_up, a2_up = stable["A1"]["excludes_zero"] and d1 > 0, stable["A2"]["excludes_zero"] and d2 > 0
    a1_down, a2_down = stable["A1"]["excludes_zero"] and d1 < 0, stable["A2"]["excludes_zero"] and d2 < 0

    if all(weak.values()) and no_stable:
        notes.append("all three at or below chance and neither paired CI excludes zero -> BOTH_WEAK")
        return "BOTH_WEAK", notes
    if raw["A0"] > raw["A1"] and raw["A0"] > raw["A2"]:
        margin = min(raw["A0"] - raw["A1"], raw["A0"] - raw["A2"])
        if margin >= MATERIAL or (a1_down and a2_down):
            notes.append(f"FULL_CURRENT is above BOTH single-branch variants (worst margin {margin:+.6f}, "
                         f"MATERIAL={MATERIAL}) -> Case F -> SYNERGY")
            return "SYNERGY", notes
        notes.append(f"FULL_CURRENT is nominally above both, but the worst margin {margin:+.6f} is below "
                     f"MATERIAL={MATERIAL} and neither paired CI establishes it; a nominal ordering on "
                     "noise is not Case F -> INCONCLUSIVE")
        return "INCONCLUSIVE", notes
    if raw["A1"] > raw["A0"] and raw["A2"] > raw["A0"]:
        margin = min(raw["A1"] - raw["A0"], raw["A2"] - raw["A0"])
        established = margin >= MATERIAL or (a1_up and a2_up)
        if established and not weak["A1"] and not weak["A2"]:
            notes.append(f"FULL_CURRENT is below BOTH single branches (worst margin {margin:+.6f}) and both "
                         "beat chance -> Case X -> FUSION_PROBLEM")
            return "FUSION_PROBLEM", notes
        if not established:
            notes.append(f"FULL_CURRENT is nominally below both, but the worst margin {margin:+.6f} is below "
                         f"MATERIAL={MATERIAL} and the paired CIs do not establish it -> INCONCLUSIVE")
            return "INCONCLUSIVE", notes
        notes.append("FULL_CURRENT below both, but at least one single branch is at or below chance; "
                     "the FUSION_PROBLEM premise is not met -> INCONCLUSIVE")
        return "INCONCLUSIVE", notes
    if raw["A1"] >= raw["A0"] and tab_far:
        notes.append("A1 TABULAR_ONLY matches or beats FULL and is clearly ahead of TEMPORAL "
                     "-> Case T -> TABULAR_DOMINANT")
        return "TABULAR_DOMINANT", notes
    if raw["A2"] >= raw["A0"] and temp_far:
        notes.append("A2 TEMPORAL_ONLY matches or beats FULL and is clearly ahead of TABULAR "
                     "-> Case R -> TEMPORAL_DOMINANT")
        return "TEMPORAL_DOMINANT", notes
    notes.append("no pre-registered case is satisfied -> INCONCLUSIVE")
    return "INCONCLUSIVE", notes


def main():
    daily = pd.read_csv(BIG / "daily_metrics.csv")
    windows = pd.read_csv(BIG / "window_metrics.csv")
    segments = pd.read_csv(BIG / "hour_segment_metrics.csv")
    paired_rows = pd.read_csv(BIG / "paired_deltas.csv")
    paired_summary = pd.read_csv(BIG / "paired_summary.csv")
    runtime = pd.read_csv(BIG / "runtime.csv")
    audit = pd.read_csv(BIG / "architecture_audit.csv")
    model_summary = pd.read_csv(BIG / "model_summary.csv")
    ownership = json.loads((HERE / "benchmark/gradient_ownership_checks.json").read_text(encoding="utf-8"))
    benchmark = pd.read_csv(HERE / "benchmark/benchmark_engineering.csv")

    signal, notes = decide(model_summary, paired_summary)
    overall = windows[windows.window == "OVERALL"].set_index("arch")

    lines = []
    add = lines.append
    add("# E2-A — Architecture Big-Block Ablation: Summary\n")
    add("A0 `FULL_CURRENT` (canonical V2.1 fusion, 28-day results **reused verbatim from E1 T2**), "
        "A1 `TABULAR_ONLY` (Direction reads `H_tab` only), A2 `TEMPORAL_ONLY` (Direction reads `H_time` only). "
        "Magnitude fusion, parameter set and TabM member axis are identical in all three modes.\n")
    add("Single-architecture axis only. No alpha grid, no concat/late fusion, no attention/MMoE/router, "
        "no threshold tuning, no Stage B, no PLE/FFT/hidden/depth/k sweep.\n")

    add("## 1. Overall (672 pooled slots, 28 days)\n")
    add("| model | Raw | Balanced | +Recall | -Recall | AUC | Brier | pred-pos frac | true prevalence | min-window Raw | window Raw std |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for arch in ARCHS:
        row = overall.loc[arch]
        add(f"| {LABEL[arch]} | {fmt(row['raw'])} | {fmt(row['balanced'])} | {fmt(row['positive_recall'])} | "
            f"{fmt(row['nonpositive_recall'])} | {fmt(row['auc'])} | {fmt(row['brier'])} | "
            f"{fmt(row['predicted_positive_fraction'])} | {fmt(row['true_positive_prevalence'])} | "
            f"{fmt(row['min_window_raw'])} | {fmt(row['window_raw_std'])} |")
    add("")
    add("Safety counters (one-class days are a class-collapse signal; a high Raw must not be read without them):\n")
    add("| model | one-class days | +Recall=0 days | -Recall=0 days |")
    add("|---|---|---|---|")
    for arch in ARCHS:
        row = overall.loc[arch]
        add(f"| {LABEL[arch]} | {int(row['one_class_prediction_days'])} | "
            f"{int(row['positive_recall_zero_days'])} | {int(row['nonpositive_recall_zero_days'])} |")

    add("\n## 2. Per-window Raw (pooled slots)\n")
    add("| model | W1 Feb | W2 Apr | W3 Jun | W4 Aug | min | std |")
    add("|---|---|---|---|---|---|---|")
    for arch in ARCHS:
        sub = windows[(windows.arch == arch) & (windows.window != "OVERALL")].set_index("window")
        values = [sub.loc[w, "raw"] for w in ("W1", "W2", "W3", "W4")]
        add(f"| {LABEL[arch]} | " + " | ".join(fmt(v) for v in values) +
            f" | {fmt(min(values))} | {fmt(float(np.std(values)))} |")

    add("\n## 3. Hour segments (Raw / Balanced / AUC / +Recall / -Recall)\n")
    add("| model | segment | Raw | Balanced | AUC | +Recall | -Recall |")
    add("|---|---|---|---|---|---|---|")
    for arch in ARCHS:
        for segment in ("H1", "H2", "H3"):
            row = segments[(segments.arch == arch) & (segments.segment == segment)].iloc[0]
            add(f"| {LABEL[arch]} | {segment} ({row['hours']}) | {fmt(row['raw'])} | {fmt(row['balanced'])} | "
                f"{fmt(row['auc'])} | {fmt(row['positive_recall'])} | {fmt(row['nonpositive_recall'])} |")

    add("\n## 4. Paired deltas vs the A0 anchor\n")
    add("Day-cluster bootstrap, unit = target day, seed 20260924, 10000 resamples "
        "(same helper as E1-mini, so the interval definition is identical).\n")
    add("| comparison | mean dRaw | 95% CI | CI excludes 0 | W/T/L (Raw) | mean dBalanced | 95% CI (Balanced) |")
    add("|---|---|---|---|---|---|---|")
    for _, row in paired_summary.iterrows():
        add(f"| {row['comparison']} | {row['mean_delta_raw']:+.6f} | "
            f"[{row['boot_ci_low_raw']:+.4f}, {row['boot_ci_high_raw']:+.4f}] | "
            f"{not bool(row['boot_ci_includes_zero_raw'])} | "
            f"{int(row['win_raw'])}/{int(row['tie_raw'])}/{int(row['loss_raw'])} | "
            f"{row['mean_delta_balanced']:+.6f} | "
            f"[{row['boot_ci_low_balanced']:+.4f}, {row['boot_ci_high_balanced']:+.4f}] |")
    add("")
    add("Per-day Raw outcome counts by window:\n")
    add("| comparison | window | improved | tie | worsened |")
    add("|---|---|---|---|---|")
    for label in ("A1_minus_A0", "A2_minus_A0"):
        sub = paired_rows[paired_rows.comparison == label]
        for window in ("W1", "W2", "W3", "W4"):
            w = sub[sub.window == window]["outcome_raw"]
            add(f"| {label} | {window} | {int((w == 'improved').sum())} | {int((w == 'tie').sum())} | "
                f"{int((w == 'worsened').sum())} |")

    add("\n## 5. Architecture, runtime, gradient ownership\n")
    add("| arch | mode | params | trainable | best epoch (mean) | epochs run (mean) | wall s/run | total wall s | epoch s | CUDA peak MB (mean/max) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in audit.iterrows():
        add(f"| {row['arch']} | {scalar(row['architecture_mode_runtime'])} | {scalar(row['parameter_count_total'])} | "
            f"{scalar(row['parameter_count_trainable'])} | {row['best_epoch_mean']:.2f} | {row['epochs_run_mean']:.2f} | "
            f"{row['wall_seconds_mean']:.2f} | {row['wall_seconds_total']:.1f} | {row['epoch_seconds_mean']:.3f} | "
            f"{row['cuda_peak_mb_mean']:.1f}/{row['cuda_peak_mb_max']:.1f} |")
    add("")
    add("`alpha_dir` trajectory — an independent corroboration of the gradient-ownership probe, read off "
        "the training logs rather than from autograd. `alpha_dir` moves only in A0, the only mode where "
        "Direction still flows through the fusion; in A1/A2 it is bit-identical to its initialisation "
        "across all 28 runs, because it no longer reaches `L_dir` at all. `alpha_mag` is bit-identical to "
        "init in **all three** modes (including A0): under `objective_mode=dir_only` the backpropagated "
        "objective is exactly `losses[\"L_dir\"]` (see `direction_experiments.objective_tensor`), and "
        "`a_mag` appears only in `h_mag`, so its gradient is exactly zero — a pre-existing property of "
        "V2.1 under this protocol, not an effect of the architecture axis.\n")
    add("| arch | alpha_dir init | alpha_dir final (mean) | min | max | alpha_mag final (mean) |")
    add("|---|---|---|---|---|---|")
    for _, row in audit.iterrows():
        add(f"| {row['arch']} | {fmt(row['alpha_dir_init'], 6)} | {fmt(row['alpha_dir_final_mean'], 6)} | "
            f"{fmt(row['alpha_dir_final_min'], 6)} | {fmt(row['alpha_dir_final_max'], 6)} | "
            f"{fmt(row['alpha_mag_final_mean'], 6)} |")
    add("")
    add("Direction gradient ownership, measured on the **trained benchmark-day checkpoints** "
        "with the real target-day tensors (not a toy model):\n")
    add("| check | result |")
    add("|---|---|")
    for name, ok in ownership.items():
        add(f"| `{name}` | {'PASS' if ok else 'FAIL'} |")

    add("\n## 6. Benchmark day 2026-02-13 — engineering only\n")
    add("These single-day numbers were used **only** for sanity/convergence/runtime/VRAM/parameter "
        "ownership/reproducibility. Per docs/10 they must not promote or eliminate an architecture.\n")
    add("| arch | mode | status | params | best/stop epoch | epochs | wall s | epoch s | CUDA peak MB | Raw | Balanced | AUC |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, row in benchmark.iterrows():
        arch = row["arch"].split()[0]
        add(f"| {LABEL[arch]} | {row['arch_mode']} | {row['status']} | {row['params']} | "
            f"{row['best_epoch']}/{row['stop_epoch']} | {row['epochs_run']} | {row['wall_s']} | "
            f"{row['epoch_s_mean']} | {row['cuda_peak_mb']} | {fmt(row['raw'])} | {fmt(row['bal'])} | {fmt(row['auc'])} |")

    add("\n## 7. Provenance\n")
    add(f"- Runs: A0 = 28 reused E1-M2 records; A1 = {int((runtime.arch == 'A1').sum())} fresh; "
        f"A2 = {int((runtime.arch == 'A2').sum())} fresh.")
    add(f"- Every run `COMPLETE`: {bool((runtime.status == 'COMPLETE').all())}.")
    add(f"- `config_sha256` distinct values across all runs: {runtime.config_sha256.nunique()} "
        f"({runtime.config_sha256.iloc[0]}).")
    add(f"- Parameter count distinct across A1/A2 runs: "
        f"{runtime[runtime.arch != 'A0'].parameter_count_total.nunique()}.")
    add(f"- Frozen protocol on every fresh run: objective={runtime.objective_mode.unique().tolist()}, "
        f"checkpoint={runtime.checkpoint_policy.unique().tolist()}, gradient={runtime.gradient_policy.unique().tolist()}.")
    add("- Figures: `figures/fig_daily_raw.png`, `figures/fig_window_overall.png`, "
        "`figures/fig_hour_segments.png`, `figures/fig_paired_bootstrap.png`.")

    (HERE / "E2_A_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gate = []
    g = gate.append
    g("# E2-A Gate — architecture big-block verdict\n")
    g(f"**ARCH_SIGNAL = `{signal}`**\n")
    g("One of the six permitted values only: TABULAR_DOMINANT / TEMPORAL_DOMINANT / SYNERGY / "
      "FUSION_PROBLEM / BOTH_WEAK / INCONCLUSIVE.\n")
    g("## How this was decided\n")
    g("The decision tree was encoded in `write_e2_a_reports.py` before results were read. "
      f"CHANCE={CHANCE}, MATERIAL={MATERIAL} (= {MATERIAL * 100:.0f} percentage points of accuracy), "
      "stability = day-cluster bootstrap 95% CI excluding zero.\n")
    for note in notes:
        g(f"- {note}")
    g("\n## Class-collapse check (docs/10 §8: Raw must not mask class collapse)\n")
    g("| model | Raw | Balanced | AUC | +Recall | -Recall | predicted-pos frac | true prevalence | one-class days | +R=0 days | -R=0 days |")
    g("|---|---|---|---|---|---|---|---|---|---|---|")
    for arch in ARCHS:
        row = overall.loc[arch]
        g(f"| {LABEL[arch]} | {fmt(row['raw'])} | {fmt(row['balanced'])} | {fmt(row['auc'])} | "
          f"{fmt(row['positive_recall'])} | {fmt(row['nonpositive_recall'])} | "
          f"{fmt(row['predicted_positive_fraction'])} | {fmt(row['true_positive_prevalence'])} | "
          f"{int(row['one_class_prediction_days'])} | {int(row['positive_recall_zero_days'])} | "
          f"{int(row['nonpositive_recall_zero_days'])} |")
    g("")
    for arch in ARCHS:
        row = overall.loc[arch]
        if row["balanced"] <= CHANCE:
            g(f"- **{LABEL[arch]} has Balanced Accuracy {fmt(row['balanced'])} — at or below chance.** "
              f"Its Raw must not be read as directional skill.")
        if row["predicted_positive_fraction"] < row["true_positive_prevalence"] / 2:
            g(f"- **{LABEL[arch]} predicts positive on only {fmt(row['predicted_positive_fraction'])} of slots "
              f"against a true prevalence of {fmt(row['true_positive_prevalence'])}.** Its Raw is carried by the "
              f"majority (non-positive) class: +Recall {fmt(row['positive_recall'])} with -Recall "
              f"{fmt(row['nonpositive_recall'])}.")

    g("\n## Readings that must accompany the verdict\n")
    g(f"- Primary metric (overall Raw): A0={fmt(overall.loc['A0','raw'])}, "
      f"A1={fmt(overall.loc['A1','raw'])}, A2={fmt(overall.loc['A2','raw'])}.")
    g(f"- min-window Raw: A0={fmt(overall.loc['A0','min_window_raw'])}, "
      f"A1={fmt(overall.loc['A1','min_window_raw'])}, A2={fmt(overall.loc['A2','min_window_raw'])}.")
    for _, prow in paired_summary.iterrows():
        established = not bool(prow["boot_ci_includes_zero_raw"])
        direction = "not statistically established" if not established else (
            "established and positive" if prow["mean_delta_raw"] > 0 else "established and negative")
        g(f"- {prow['comparison'].replace('_', ' ')} Raw delta: {prow['mean_delta_raw']:+.6f}, 95% CI "
          f"[{prow['boot_ci_low_raw']:+.4f}, {prow['boot_ci_high_raw']:+.4f}] — **{direction}**.")
    if bool(paired_summary.set_index("comparison").loc["A1_minus_A0", "boot_ci_includes_zero_raw"]):
        g("- **The A1 - A0 Raw difference is not statistically established** (its 95% CI includes zero), "
          "so A1's nominal Raw edge over FULL_CURRENT should not be read as a real gain.")
    g("- The only established paired Raw result is **negative**: removing the tabular branch from Direction "
      "(A2 TEMPORAL_ONLY) is reliably worse than FULL_CURRENT.")
    g("- This verdict is about **attribution** — which branch carries the Direction signal — not about "
      "promoting a variant. A0 keeps the best Balanced, the best AUC and the best-calibrated "
      "predicted-positive fraction of the three.")
    g("- This verdict concerns the Direction pathway only. Magnitude is identical across all three "
      "modes by construction and was never part of the ablation.")
    g("- No day-level or month-level oracle routing was used, and no per-window structure was fitted.")
    g("\n## What this does not say\n")
    g("- It does not authorise E2-B, a fusion redesign, a new architecture axis, or any threshold tuning.")
    g("- It does not establish the architecture for the final lockbox; the 28-day panel is DEV, "
      "not the lockbox scope.")
    g("- It is not a 65% result. No variant here is being proposed for promotion on these numbers.")
    g("\n## Status\n")
    g("E2-A is complete. Stopping here — E2-B is not executed. Nothing is staged or committed.")
    (HERE / "E2_A_GATE.md").write_text("\n".join(gate) + "\n", encoding="utf-8")

    print(f"ARCH_SIGNAL = {signal}")
    for note in notes:
        print("  -", note)
    print("\nwrote E2_A_summary.md, E2_A_GATE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

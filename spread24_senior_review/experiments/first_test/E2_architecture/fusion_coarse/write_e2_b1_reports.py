"""Render E2_B1_summary.md and E2_B1_GATE.md from the CSVs the aggregation already wrote.

The verdict is not re-derived here: it is read from gate_decision.json, which analyze_e2_b1.py
produced with the rule that was fixed in 00_PLAN_SNAPSHOT.md before any formal run was read.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ["F00", "F20", "F40", "F60", "F80", "F100", "FL08"]
ANCHOR = "FL08"
CHANCE = 0.50


def scalar(value):
    """CSV round-trips turn some cells into single-element lists; unwrap for display."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def num(value, digits=4):
    value = scalar(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    # Counts arrive from CSV as floats; render them without a spurious fractional part.
    if number.is_integer() and abs(number) >= 1:
        return str(int(number))
    return f"{number:.{digits}f}"


def table(frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 4) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in frame.iterrows():
        cells = []
        for key, _ in columns:
            value = row.get(key)
            cells.append(num(value, digits) if isinstance(value, (int, float)) or
                         (isinstance(value, str) and value.replace(".", "").replace("-", "").isdigit())
                         else str(scalar(value)))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(HERE / name)


def main():
    overall = load("model_summary.csv").set_index("arch")
    windows = load("window_metrics.csv")
    segments = load("hour_segment_metrics.csv")
    paired = load("paired_summary.csv")
    safety = load("safety_admissibility.csv").set_index("arm")
    runtime = load("runtime.csv")
    decision = json.loads((HERE / "gate_decision.json").read_text(encoding="utf-8"))
    gate_a = json.loads((HERE / "benchmark/gate_a_fusion_checks.json").read_text(encoding="utf-8"))
    evidence = pd.read_csv(HERE / "benchmark/gate_a_fusion_evidence.csv")

    diag_path = HERE / "fusion_diagnostics_summary.csv"
    diag = pd.read_csv(diag_path).set_index("arm") if diag_path.exists() else None

    ordered = overall.loc[[a for a in ARMS if a in overall.index]]

    # ---------------------------------------------------------------- summary
    lines = ["# E2-B1 — Direction Fusion Coarse Map: summary", ""]
    lines += [
        "Seven arms over the frozen 28-day DEV panel. `alpha_time` pins the Direction fusion",
        "coefficient `H_dir = alpha*time_dir(H_time) + (1-alpha)*tab_dir(H_tab)`; nothing else moves.",
        "FL08 (canonical learnable alpha, init 0.8) is the safety anchor. Chance = 0.50.", "",
        f"**FUSION_REGION = `{decision['fusion_region']}`**", "",
        "## 5. Overall (28 days, micro over slots)", "",
        table(ordered.reset_index(),
              [("arch", "arm"), ("raw", "Raw"), ("balanced", "Balanced"),
               ("positive_recall", "+Recall"), ("nonpositive_recall", "−Recall"),
               ("auc", "AUC"), ("brier", "Brier"), ("predicted_positive_fraction", "pred. pos. frac"),
               ("true_positive_prevalence", "true prevalence"), ("n_slots", "slots")]),
        "",
    ]

    # class-collapse block, reported for every arm rather than only the good ones
    lines += ["### Class-collapse check (Raw must not mask collapse)", "",
              "| arm | Raw | Balanced | one-class days | +Recall=0 days | −Recall=0 days |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm in ordered.index:
        row = ordered.loc[arm]
        lines.append(f"| {arm} | {num(row['raw'])} | {num(row['balanced'])} | "
                     f"{int(scalar(row['one_class_prediction_days']))} | "
                     f"{int(scalar(row['positive_recall_zero_days']))} | "
                     f"{int(scalar(row['nonpositive_recall_zero_days']))} |")
    lines.append("")

    # windows
    lines += ["## 6. Windows W1–W4 (Raw / Balanced)", "",
              "| arm | " + " | ".join(f"{w} Raw | {w} Bal" for w in ("W1", "W2", "W3", "W4"))
              + " | min-window Raw | window Raw std |", "| --- | " + " | ".join(["---"] * 10) + " |"]
    for arm in ordered.index:
        cells = []
        for window in ("W1", "W2", "W3", "W4"):
            sub = windows[(windows.arch == arm) & (windows.window == window)]
            cells += [num(sub["raw"].iloc[0]) if len(sub) else "n/a",
                      num(sub["balanced"].iloc[0]) if len(sub) else "n/a"]
        cells += [num(ordered.loc[arm].get("min_window_raw")), num(ordered.loc[arm].get("window_raw_std"))]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("")

    # hour segments
    lines += ["## 7. Hour segments H1–H3", "",
              "| arm | " + " | ".join(f"{s} Raw | {s} Bal | {s} +R" for s in ("H1", "H2", "H3")) + " |",
              "| --- | " + " | ".join(["---"] * 9) + " |"]
    for arm in ordered.index:
        cells = []
        for segment in ("H1", "H2", "H3"):
            sub = segments[(segments.arch == arm) & (segments.segment == segment)]
            cells += [num(sub["raw"].iloc[0]) if len(sub) else "n/a",
                      num(sub["balanced"].iloc[0]) if len(sub) else "n/a",
                      num(sub["positive_recall"].iloc[0]) if len(sub) else "n/a"]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines.append("")

    # paired vs anchor
    lines += [f"## 8. Paired vs the {ANCHOR} anchor (day-cluster bootstrap, unit = target day)", "",
              "| comparison | mean ΔRaw | CI low | CI high | CI⊂zero | W/T/L Raw | "
              "mean ΔBalanced | CI low | CI high | CI⊂zero |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['comparison']} | {num(row['mean_delta_raw'], 6)} | {num(row['boot_ci_low_raw'], 6)} | "
            f"{num(row['boot_ci_high_raw'], 6)} | {bool(scalar(row['boot_ci_includes_zero_raw']))} | "
            f"{int(scalar(row['win_raw']))}/{int(scalar(row['tie_raw']))}/{int(scalar(row['loss_raw']))} | "
            f"{num(row['mean_delta_balanced'], 6)} | {num(row['boot_ci_low_balanced'], 6)} | "
            f"{num(row['boot_ci_high_balanced'], 6)} | {bool(scalar(row['boot_ci_includes_zero_balanced']))} |")
    lines.append("")

    # safety
    lines += [f"## 9. Safety admissibility vs {ANCHOR}", "",
              "SAFETY_ADMISSIBLE iff Balanced ≥ anchor−0.01 **and** +Recall ≥ anchor−0.05 "
              "**and** one-class days ≤ anchor+2 **and** +Recall=0 days ≤ anchor+2.", "",
              "| arm | alpha | Balanced | ΔBalanced | +Recall | Δ+Recall | Δone-class | "
              "Δ+R=0 days | SAFETY_ADMISSIBLE |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in [a for a in ARMS if a in safety.index]:
        row = safety.loc[arm]
        lines.append(
            f"| {arm} | {scalar(row['alpha'])} | {num(row['balanced'])} | {num(row['balanced_vs_anchor'], 6)} | "
            f"{num(row['positive_recall'])} | {num(row['positive_recall_vs_anchor'], 6)} | "
            f"{int(scalar(row['one_class_days_vs_anchor'])):+d} | "
            f"{int(scalar(row['positive_recall_zero_days_vs_anchor'])):+d} | "
            f"**{bool(scalar(row['safety_admissible']))}** |")
    lines.append("")
    rejected = [a for a in safety.index if not bool(scalar(safety.loc[a, "safety_admissible"]))]
    lines += ([f"Non-admissible arms (reported, not hidden): **{', '.join(rejected)}**."]
              if rejected else ["Every arm passes the safety filter."])
    lines.append("")

    # diagnostics
    lines += ["## 10. Effective fusion diagnostics (alpha is not a contribution percentage)", ""]
    if diag is None:
        lines.append("_fusion_diagnostics_summary.csv not present._")
    else:
        lines += ["Deterministic diagnostic batch = the target day's own inputs. Observation only.", "",
                  "| arm | alpha used (mean) | range | ‖tab‖ | ‖time‖ | "
                  "effective time-norm fraction (mean ± sd over days) | cosine(tab, time) |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for arm in [a for a in ARMS if a in diag.index]:
            row = diag.loc[arm]
            std = scalar(row.get("effective_time_norm_fraction_std"))
            spread = f" ± {num(std, 4)}" if pd.notna(std) else ""
            lines.append(
                f"| {arm} | {float(scalar(row['alpha_used'])):.6f} | "
                f"{float(scalar(row['alpha_used_min'])):.4f}–{float(scalar(row['alpha_used_max'])):.4f} | "
                f"{num(row['norm_tab_component'], 4)} | {num(row['norm_time_component'], 4)} | "
                f"{num(row['effective_time_norm_fraction'], 4)}{spread} | "
                f"{num(row['cosine_tab_time'], 4)} |")
        lines.append("")
        lines += [
            "The two norm columns and the fraction column are each means over the 28 runs, and every",
            "run is trained separately, so the fraction column is the mean of per-run fractions and",
            "is **not** the ratio of the two norm means shown beside it. The per-run values are in",
            "`fusion_diagnostics.csv`.", "",
            "Read this against the alpha column, not instead of it. Two facts matter: the",
            "effective time-norm fraction at a given alpha is far from alpha (the branches do not",
            "have comparable norms), and it varies **across target days** for a fixed coefficient,",
            "because the encoders are retrained per day. A response curve indexed by alpha is",
            "therefore a curve over *training runs*, not over a smoothly varying mixture weight.", ""]

    # runtime
    lines += ["## 11. Runtime", "",
              "| arm | runs | wall mean (s) | epoch mean (s) | best epoch (mean) | stop epoch (mean) | "
              "params | config_sha256 (distinct) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in [a for a in ARMS if a in set(runtime.arch)]:
        sub = runtime[runtime.arch == arm]
        lines.append(
            f"| {arm} | {len(sub)} | {num(sub['wall_time_total_seconds'].mean(), 2)} | "
            f"{num(sub['training_epoch_seconds_mean'].mean(), 4)} | "
            f"{num(sub['best_epoch'].mean(), 2)} | {num(sub['stop_epoch'].mean(), 2)} | "
            f"{int(scalar(sub['parameter_count_total'].iloc[0])) if pd.notna(sub['parameter_count_total'].iloc[0]) else 'n/a'} | "
            f"{sub['config_sha256'].nunique()} |")
    lines.append("")

    # gate a
    passed = sum(1 for v in gate_a.values() if v)
    lines += ["## 3. Fixed-alpha routing audit (Gate A, benchmark day 2026-02-13)", "",
              f"**{passed}/{len(gate_a)} PASS.** Endpoint reproduction is compared as artifacts.", "",
              "| check | result |", "| --- | --- |"]
    for name, ok in gate_a.items():
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines += ["| arm | alpha | reproduces | parquet SHA match | max abs Δpred | max abs Δmetrics | "
              "distinct alpha values |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, row in evidence.iterrows():
        lines.append(
            f"| {row['arm']} | {scalar(row['direction_fusion_alpha'])} | {row['reproduces']} | "
            f"{scalar(row.get('predictions_parquet_sha256_match', 'n/a'))} | "
            f"{num(row.get('max_abs_delta_predictions', float('nan')), 3)} | "
            f"{num(row.get('max_abs_delta_metrics', float('nan')), 3)} | "
            f"{scalar(row.get('distinct_alpha_dir_values', 'n/a'))} |")
    lines.append("")
    lines += [
        "The immobility check reads `alpha_trajectories[].alpha_dir`, which logs the model's",
        "**learnable** `a_dir` (a sigmoid applied to a logit). Under a fixed coefficient that",
        "parameter is inert, so the proof is that it is bit-constant across every epoch. Its",
        "constant value is the *initialisation* (0.8), not the arm's coefficient; the coefficient",
        "is recorded separately in the run manifest and shown above as `alpha`.", ""]

    # benchmark
    lines += ["## 4. Benchmark day 2026-02-13 (engineering only)", "",
              "| arm | alpha | Raw |", "| --- | --- | --- |"]
    bench_path = HERE / "benchmark" / "GATE_A_BENCHMARK_RUNS.json"
    if bench_path.exists():
        for record in json.loads(bench_path.read_text(encoding="utf-8")):
            raw = (record.get("metrics") or {}).get("raw_direction_accuracy")
            lines.append(f"| {record['arm']} | {record['direction_fusion_alpha']} | {num(raw, 4)} |")
    lines += ["",
              "This is one day. It is recorded for engineering provenance only and was **not** used",
              "to rank, filter or eliminate any arm.", ""]

    (HERE / "E2_B1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------- gate
    gate = ["# E2-B1 Gate", "",
            f"**FUSION_REGION = `{decision['fusion_region']}`**", "",
            "Decision rule (fixed in `00_PLAN_SNAPSHOT.md` before any formal run was read):", ""]
    gate += [f"{i}. {n}" for i, n in enumerate([
        "no SAFETY_ADMISSIBLE arm -> `NO_CLEAR_REGION`",
        "best admissible arm is FL08 -> `CURRENT_FUSION_OK`",
        "best admissible arm is F00 -> `ENDPOINT_TABULAR`",
        "otherwise label by `alpha(best)`: ≤0.4 `TABULAR_HEAVY`, (0.4,0.7) `BALANCED_MIX`, "
        "≥0.7 `TEMPORAL_HEAVY`; but fall back to `NO_CLEAR_REGION` when the leader is neither "
        "separated by ≥0.02 Raw from every admissible arm in another region nor supported by a "
        "paired CI vs FL08 that excludes zero.",
    ], 1)]
    gate += ["", "## How the rule resolved", ""]
    gate += [f"- {note}" for note in decision["notes"]]

    # The literal rule would read alpha(F80)=0.8 >= 0.7 and return TEMPORAL_HEAVY. The uncertainty
    # clause was invoked instead, so the evidence for that choice has to be on the page.
    pairwise_path = HERE / "paired_pairwise.csv"
    gate += ["", "## Why not `TEMPORAL_HEAVY`", "",
             "Read literally, rule 4 returns `TEMPORAL_HEAVY`: the best admissible arm is F80 at",
             "alpha=0.8. The uncertainty clause was invoked instead. The evidence:", ""]
    if pairwise_path.exists():
        pairwise = pd.read_csv(pairwise_path)
        gate += ["| comparison | mean ΔRaw | in slots of 672 | CI low | CI high | CI⊂zero | W/T/L |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        for a, b in (("F80", "FL08"), ("F20", "F80"), ("F20", "FL08")):
            row = pairwise[(pairwise.arm_a == a) & (pairwise.arm_b == b)]
            if row.empty:
                continue
            row = row.iloc[0]
            mean = float(scalar(row["mean_delta_raw"]))
            gate.append(
                f"| {a} − {b} | {mean:+.4f} | **{mean * 672:+.1f}** | {num(row['ci_low_raw'], 4)} | "
                f"{num(row['ci_high_raw'], 4)} | {bool(scalar(row['ci_includes_zero_raw']))} | "
                f"{int(scalar(row['win_raw']))}/{int(scalar(row['tie_raw']))}/{int(scalar(row['loss_raw']))} |")
        gate.append("")
    if diag is not None and "F80" in diag.index and ANCHOR in diag.index:
        f80_frac = scalar(diag.loc["F80", "effective_time_norm_fraction"])
        anchor_frac = scalar(diag.loc[ANCHOR, "effective_time_norm_fraction"])
        gate += [
            f"- **F80 and FL08 finish at nearly the same coefficient/effective mixture, but they are not the same training path.** Their effective time-norm fractions",
            f"  are {num(f80_frac)} and {num(anchor_frac)} respectively. F80 keeps alpha fixed at 0.8, while FL08 backpropagates through a learnable alpha and settles near 0.798.",
            f"  The +7-slot difference is concentrated in only three target days, so it is evidence of a small training-path effect, not established evidence that the 0.8 coefficient region is intrinsically superior.", ""]
    gate += [
        "- **The whole admissible band spans 3–7 slots out of 672.** The day-level statistic has a",
        "  resolution of 1/24 = 0.0417 per day, an order of magnitude larger than the mean",
        "  differences being ranked here. F80 never lost to FL08 on any day, but it tied on 25 of",
        "  28 — a bootstrap CI whose lower bound lands exactly on 0.0000.",
        "- **The `TEMPORAL_HEAVY` claim would rest on the endpoint structure, which is where the",
        "  evidence is strongest but least favourable:** F100 (the temporal endpoint) is",
        "  non-admissible (Balanced −0.0526) and sits at Raw = 0.5000 exactly, so the temporal",
        "  branch cannot carry Direction alone. That is an argument against the endpoints, not for",
        "  the temporal region.",
        "",
        "So the map resolves part of the safety shape: the pure endpoints F00/F100 are non-admissible; F20 and F80/FL08 are admissible; F40/F60 miss the pre-registered Balanced threshold by tiny point-estimate margins.",
        "It does not establish a stable ordering among the admissible operating points, which is exactly what `NO_CLEAR_REGION` records.", "",
        "### Non-admissibility margins (the filter is pre-registered; these are the margins it acted on)", "",
        "| arm | fails on | margin | paired CI vs FL08 |",
        "| --- | --- | --- | --- |",
        f"| F40 | Balanced | −0.0104 vs threshold −0.0100 (shortfall 0.0004) | "
        f"{num(paired.set_index('comparison').loc['F40_minus_FL08', 'boot_ci_low_balanced'], 4)} to "
        f"{num(paired.set_index('comparison').loc['F40_minus_FL08', 'boot_ci_high_balanced'], 4)}, includes zero |",
        f"| F60 | Balanced | −0.0111 vs threshold −0.0100 (shortfall 0.0011) | "
        f"{num(paired.set_index('comparison').loc['F60_minus_FL08', 'boot_ci_low_balanced'], 4)} to "
        f"{num(paired.set_index('comparison').loc['F60_minus_FL08', 'boot_ci_high_balanced'], 4)}, includes zero |",
        f"| F00 | Balanced −0.0288, +Recall −0.1901, one-class +6, +R=0 +9 | fails all four criteria | "
        f"{num(paired.set_index('comparison').loc['F00_minus_FL08', 'boot_ci_low_balanced'], 4)} to "
        f"{num(paired.set_index('comparison').loc['F00_minus_FL08', 'boot_ci_high_balanced'], 4)} |",
        f"| F100 | Balanced −0.0526 | fails one criterion | "
        f"{num(paired.set_index('comparison').loc['F100_minus_FL08', 'boot_ci_low_balanced'], 4)} to "
        f"{num(paired.set_index('comparison').loc['F100_minus_FL08', 'boot_ci_high_balanced'], 4)} |",
        "",
        "F40 and F60 are recorded as non-admissible because the filter was fixed before the runs and",
        "is applied as written. The reader should still see that they miss by 0.0004 and 0.0011 on a",
        "panel whose per-day resolution is 0.0417, and that neither paired CI excludes zero.",
        "Non-admissibility here is a point-estimate verdict, not an established one.", "",
    ]

    gate += ["", "## Readings that must accompany the verdict", ""]
    adm = [a for a in safety.index if bool(scalar(safety.loc[a, "safety_admissible"]))]
    gate += [
        f"- SAFETY_ADMISSIBLE arms: {', '.join(adm) if adm else 'none'}.",
        f"- Non-admissible arms: {', '.join(rejected) if rejected else 'none'}.",
        "- Raw is the primary metric, but no Raw advantage is reported here without the safety "
        "filter above it. An arm with a high Raw and a collapsed class distribution is recorded as "
        "non-admissible rather than as a winner.",
        "- `alpha` is a coefficient, not a contribution share; §10 of the summary reports the "
        "effective mixture actually in force.",
        "- The panel is 28 DEV days over four windows. Nothing here is a lockbox result.",
        "",
        "## What this does not say", "",
        "- It does not select a production coefficient. It maps a region.",
        "- It does not test any fusion form other than the existing convex combination.",
        "- It does not touch thresholds, calibration, checkpoint policy or Stage B.",
        "",
        "Status: E2-B1 is complete. Stopping here — no later stage is executed. "
        "Nothing is staged or committed.",
    ]
    (HERE / "E2_B1_GATE.md").write_text("\n".join(gate) + "\n", encoding="utf-8")
    print(f"wrote E2_B1_summary.md ({len(lines)} lines) and E2_B1_GATE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

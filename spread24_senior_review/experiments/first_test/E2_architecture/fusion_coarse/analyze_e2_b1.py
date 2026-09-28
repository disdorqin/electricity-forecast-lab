"""E2-B1 aggregation — seven arms over the frozen 28-day DEV panel.

Arms:
    F00   fixed alpha_time = 0.0   (reused E2-A A1, architecture_mode=tabular_only)
    F20   fixed alpha_time = 0.2   (fresh)
    F40   fixed alpha_time = 0.4   (fresh)
    F60   fixed alpha_time = 0.6   (fresh)
    F80   fixed alpha_time = 0.8   (fresh)
    F100  fixed alpha_time = 1.0   (reused E2-A A2, architecture_mode=temporal_only)
    FL08  canonical learnable alpha, init 0.8 (reused E2-A A0) -> SAFETY ANCHOR

The E2-A aggregation helpers are reused by overriding their module-level arm list, so no E2-A
source or artifact is edited. Only the paired comparison is rewritten: E2-B1 pairs every arm
against the single FL08 anchor rather than against A0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E2A = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(E2A))

import analyze_e2_a as A  # noqa: E402
from analyze_e1_mini import day_cluster_bootstrap  # noqa: E402

TIE_EPS = A.TIE_EPS
CHANCE = 0.50
MATERIAL = 0.02
ANCHOR = "FL08"

ARMS = ("F00", "F20", "F40", "F60", "F80", "F100", "FL08")
ALPHA = {"F00": 0.0, "F20": 0.2, "F40": 0.4, "F60": 0.6, "F80": 0.8, "F100": 1.0}
LABEL = {
    "F00": "F00 fixed alpha_time=0.0 (tabular endpoint)",
    "F20": "F20 fixed 0.2", "F40": "F40 fixed 0.4",
    "F60": "F60 fixed 0.6", "F80": "F80 fixed 0.8",
    "F100": "F100 fixed alpha_time=1.0 (temporal endpoint)",
    "FL08": "FL08 learnable alpha (init 0.8) - ANCHOR",
}
E1_RUNS = A.E1_RUNS   # authoritative: the frozen E2-A module owns these paths
E2A_RUNS = A.E2_RUNS
B1_RUNS = HERE / "runs"

# Reuse the E2-A helpers unchanged by pointing their arm globals at this round's arms.
A.ARCHS = ARMS
A.ARCH_LABEL = LABEL


def load_records() -> pd.DataFrame:
    rows = []
    for path in sorted(E1_RUNS.glob("E1-M2-*/RUN_RECORD.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "PASS":
            rows.append({"arch": "FL08", "window": record["window"], "target_day": record["target_day"],
                         "run_dir": record["raw_run_dir"], "manifest": A.normalize_e1_record(record),
                         "source": "E1-M2 (reused)"})
    for pattern, arch in (("E2-A1-*", "F00"), ("E2-A2-*", "F100")):
        for path in sorted(E2A_RUNS.glob(f"{pattern}/RUN_RECORD.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "PASS":
                rows.append({"arch": arch, "window": record["window"], "target_day": record["target_day"],
                             "run_dir": record["run_dir"], "manifest": record["manifest"],
                             "source": "E2-A (reused)"})
    for arm in ("F20", "F40", "F60", "F80"):
        for path in sorted(B1_RUNS.glob(f"E2B1-{arm}-*/RUN_RECORD.json")):
            try:  # a run still being written must not abort the aggregation
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if record.get("status") == "PASS":
                rows.append({"arch": arm, "window": record["window"], "target_day": record["target_day"],
                             "run_dir": record["run_dir"], "manifest": record["manifest"],
                             "source": "E2-B1 Gate C"})
    frame = pd.DataFrame(rows)
    print(f"loaded {len(frame)} runs: " + ", ".join(f"{a}={n}" for a, n in frame.arch.value_counts().items()))
    return frame


def paired_vs_anchor(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = daily.set_index(["arch", "target_day"])
    metrics = [("raw", True), ("balanced", True), ("positive_recall", True),
               ("nonpositive_recall", True), ("auc", True), ("brier", False),
               ("predicted_positive_fraction", None)]
    rows, summaries = [], []
    for arm in ARMS:
        if arm == ANCHOR:
            continue
        label = f"{arm}_minus_{ANCHOR}"
        sub_rows = []
        for _, day_row in daily[daily.arch == arm].sort_values("target_day").iterrows():
            day = day_row["target_day"]
            if (ANCHOR, day) not in indexed.index:
                continue
            ra, rb = indexed.loc[(arm, day)], indexed.loc[(ANCHOR, day)]
            row = {"comparison": label, "arch_a": arm, "arch_b": ANCHOR,
                   "window": day_row["window"], "target_day": day}
            for name, _ in metrics:
                va, vb = ra[name], rb[name]
                row[f"delta_{name}"] = (float(va) - float(vb)) if pd.notna(va) and pd.notna(vb) else np.nan
            for name in ("raw", "balanced"):
                row[f"outcome_{name}"] = ("improved" if row[f"delta_{name}"] > TIE_EPS else
                                          "worsened" if row[f"delta_{name}"] < -TIE_EPS else "tie")
            sub_rows.append(row)
        sub = pd.DataFrame(sub_rows)
        rows.extend(sub_rows)
        summary = {"comparison": label, "arch_a": arm, "arch_b": ANCHOR, "n_days": len(sub)}
        if sub.empty:
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
                boot = day_cluster_bootstrap(deltas)
                summary["bootstrap_seed"] = boot["bootstrap_seed"]
                summary["bootstrap_resamples"] = boot["bootstrap_resamples"]
                summary["bootstrap_unit"] = "target_day"
        summaries.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def paired_pairwise(daily: pd.DataFrame) -> pd.DataFrame:
    """Every arm against every other, for the same reason `paired_vs_anchor` exists: the region
    verdict turns on whether the leading admissible arms are separable, and that is a paired
    question about the same 28 days, not a comparison of two overall point estimates."""
    indexed = daily.set_index(["arch", "target_day"])
    days = sorted(daily.target_day.unique())
    rows = []
    for i, arm_a in enumerate(ARMS):
        for arm_b in ARMS[i + 1:]:
            deltas, day_rows = {k: [] for k in ("raw", "balanced", "positive_recall")}, []
            for day in days:
                if (arm_a, day) not in indexed.index or (arm_b, day) not in indexed.index:
                    continue
                ra, rb = indexed.loc[(arm_a, day)], indexed.loc[(arm_b, day)]
                for key in deltas:
                    if pd.notna(ra[key]) and pd.notna(rb[key]):
                        deltas[key].append(float(ra[key]) - float(rb[key]))
                day_rows.append(day)
            if not day_rows:
                continue
            row = {"arm_a": arm_a, "arm_b": arm_b, "n_days": len(day_rows)}
            for key, values in deltas.items():
                values = np.asarray(values, dtype=float)
                boot = day_cluster_bootstrap(values)
                row[f"mean_delta_{key}"] = float(values.mean())
                row[f"ci_low_{key}"] = boot["ci_low"]
                row[f"ci_high_{key}"] = boot["ci_high"]
                row[f"ci_includes_zero_{key}"] = boot["ci_includes_zero"]
                row[f"win_{key}"] = int((values > TIE_EPS).sum())
                row[f"tie_{key}"] = int((np.abs(values) <= TIE_EPS).sum())
                row[f"loss_{key}"] = int((values < -TIE_EPS).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def safety_table(overall: pd.DataFrame) -> pd.DataFrame:
    """SAFETY_ADMISSIBLE per E2-B1 plan §9, applied against the FL08 anchor."""
    anchor = overall.loc[ANCHOR]
    rows = []
    for arm in ARMS:
        row = overall.loc[arm]
        admissible = (row["balanced"] >= anchor["balanced"] - 0.01
                      and row["positive_recall"] >= anchor["positive_recall"] - 0.05
                      and row["one_class_prediction_days"] <= anchor["one_class_prediction_days"] + 2
                      and row["positive_recall_zero_days"] <= anchor["positive_recall_zero_days"] + 2)
        rows.append({
            "arm": arm, "alpha": ALPHA.get(arm, "learnable"), "raw": row["raw"],
            "balanced": row["balanced"], "positive_recall": row["positive_recall"],
            "one_class_prediction_days": int(row["one_class_prediction_days"]),
            "positive_recall_zero_days": int(row["positive_recall_zero_days"]),
            "balanced_vs_anchor": row["balanced"] - anchor["balanced"],
            "positive_recall_vs_anchor": row["positive_recall"] - anchor["positive_recall"],
            "one_class_days_vs_anchor": int(row["one_class_prediction_days"] - anchor["one_class_prediction_days"]),
            "positive_recall_zero_days_vs_anchor": int(row["positive_recall_zero_days"] - anchor["positive_recall_zero_days"]),
            "safety_admissible": bool(admissible),
        })
    return pd.DataFrame(rows)


def decide(overall: pd.DataFrame, safety: pd.DataFrame, paired_summary: pd.DataFrame) -> tuple[str, list[str]]:
    """Pre-registered E2-B1 region rule (encoded before results are read).

    Region labels follow plan §10. `best` = the highest-Raw arm among SAFETY_ADMISSIBLE arms,
    ties broken by Balanced then AUC, because Raw is the primary metric but §9 forbids reading it
    without the safety filter.
    """
    admissible = safety[safety.safety_admissible].set_index("arm")
    notes = [f"admissible arms ({len(admissible)}): {', '.join(admissible.index)}"]
    if admissible.empty:
        notes.append("no arm passes the safety filter -> NO_CLEAR_REGION")
        return "NO_CLEAR_REGION", notes

    best = max(admissible.index, key=lambda a: (admissible.loc[a, "raw"],
                                                admissible.loc[a, "balanced"],
                                                overall.loc[a, "auc"]))
    note_best = (f"best admissible by Raw = {best} (Raw={admissible.loc[best, 'raw']:.6f}, "
                 f"Balanced={admissible.loc[best, 'balanced']:.6f})")
    notes.append(note_best)

    interior = [a for a in admissible.index if a in ALPHA and 0.0 < ALPHA[a] < 1.0]
    if best == "FL08":
        notes.append("the canonical learnable fusion remains the best admissible tradeoff "
                     "-> CURRENT_FUSION_OK")
        return "CURRENT_FUSION_OK", notes
    if best == "F00":
        notes.append("the tabular endpoint is the best admissible arm and no interior admissible "
                     "point improves on it -> ENDPOINT_TABULAR")
        return "ENDPOINT_TABULAR", notes

    alpha = ALPHA[best]
    region = ("TABULAR_HEAVY" if alpha <= 0.4 else
              "BALANCED_MIX" if alpha < 0.7 else "TEMPORAL_HEAVY")
    # Separation: is `best` clearly ahead of the admissible arms that sit in a different region?
    def region_of(arm):
        if arm == "FL08":
            return "ANCHOR"
        a = ALPHA.get(arm)
        if a is None:
            return "ENDPOINT"
        return "TABULAR_HEAVY" if a <= 0.4 else "BALANCED_MIX" if a < 0.7 else "TEMPORAL_HEAVY"

    rivals = [a for a in admissible.index if a != best and region_of(a) != region_of(best)]
    separated = all(admissible.loc[best, "raw"] - admissible.loc[a, "raw"] >= MATERIAL for a in rivals) if rivals else True
    row = paired_summary.set_index("comparison").get(f"{best}_minus_{ANCHOR}")
    established = bool(row is not None and not bool(row["boot_ci_includes_zero_raw"]) and row["mean_delta_raw"] > 0)
    notes.append(f"best={best} region={region}; separated from other-region admissible arms: {separated}; "
                 f"paired Raw vs {ANCHOR} established positive: {established}")
    if not separated and not established:
        notes.append("no admissible arm is separated from its rivals and no paired evidence "
                     "establishes the leader -> NO_CLEAR_REGION")
        return "NO_CLEAR_REGION", notes
    notes.append(f"best admissible region is alpha={alpha} -> {region}")
    return region, notes


def write_figures(overall: pd.DataFrame, windows: pd.DataFrame, segments: pd.DataFrame,
                  paired_summary: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    ordered = [a for a in ("F00", "F20", "F40", "F60", "F80", "F100") if a in overall.index]
    xs = [ALPHA[a] for a in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, column, title in ((axes[0], "raw", "Raw"), (axes[1], "balanced", "Balanced")):
        ax.plot(xs, [overall.loc[a, column] for a in ordered], "o-", label="fixed alpha")
        anchor = overall.loc[ANCHOR, column]
        ax.axhline(anchor, ls="--", c="crimson", label=f"{ANCHOR} learnable")
        ax.axhline(CHANCE, ls=":", c="gray", label="chance")
        ax.set_xlabel("fixed Direction fusion alpha_time")
        ax.set_ylabel(title)
        ax.set_title(f"{title} response vs alpha")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_response_raw_balanced.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for arm in ordered + [ANCHOR]:
        row = overall.loc[arm]
        marker = "*" if arm == ANCHOR else "o"
        ax.scatter(row["balanced"], row["raw"], marker=marker, s=110 if arm == ANCHOR else 55)
        ax.annotate(arm, (row["balanced"], row["raw"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Balanced accuracy"); ax.set_ylabel("Raw accuracy")
    ax.set_title("Pareto view: Raw vs Balanced")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig_pareto_raw_balanced.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, column, title in ((axes[0], "positive_recall", "+Recall"), (axes[1], "nonpositive_recall", "-Recall")):
        ax.plot(xs, [overall.loc[a, column] for a in ordered], "o-", label="fixed alpha")
        ax.axhline(overall.loc[ANCHOR, column], ls="--", c="crimson", label=f"{ANCHOR} learnable")
        ax.set_xlabel("fixed Direction fusion alpha_time"); ax.set_ylabel(title)
        ax.set_title(f"{title} response vs alpha"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_response_recalls.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for arm in ordered:
        sub = windows[(windows.arch == arm) & (windows.window != "OVERALL")]
        if sub.empty:
            continue
        sub = sub.set_index("window").reindex(["W1", "W2", "W3", "W4"])
        ax.plot(["W1", "W2", "W3", "W4"], sub["raw"], "o-", label=arm)
    anchor = windows[(windows.arch == ANCHOR) & (windows.window != "OVERALL")].set_index("window").reindex(["W1", "W2", "W3", "W4"])
    if not anchor["raw"].isna().all():
        ax.plot(["W1", "W2", "W3", "W4"], anchor["raw"], "k--", label=ANCHOR)
    ax.set_ylabel("Raw"); ax.set_title("Per-window Raw vs alpha"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_window_response.png", dpi=130)
    plt.close(fig)


def main():
    records = load_records()
    slots = pd.concat([A.slots_for(r) for _, r in records.iterrows()], ignore_index=True)
    daily = A.per_day(slots)
    windows = A.window_table(daily, slots)
    segments = A.hour_segments(slots)
    paired_rows, paired_summary = paired_vs_anchor(daily)
    overall = windows[windows.window == "OVERALL"].set_index("arch")
    safety = safety_table(overall)
    runtime = A.runtime_table(records)
    # The arm name is the authoritative record of the coefficient: the E2-A reuse records predate
    # the `direction_fusion_alpha` manifest field, so it cannot be read back from them.
    runtime["direction_fusion_alpha"] = runtime["arch"].map(lambda a: ALPHA.get(a, "learnable"))
    region, notes = decide(overall, safety, paired_summary)

    out = HERE
    daily.to_csv(out / "daily_metrics.csv", index=False)
    windows.to_csv(out / "window_metrics.csv", index=False)
    segments.to_csv(out / "hour_segment_metrics.csv", index=False)
    paired_rows.to_csv(out / "paired_deltas.csv", index=False)
    paired_summary.to_csv(out / "paired_summary.csv", index=False)
    paired_pairwise(daily).to_csv(out / "paired_pairwise.csv", index=False)
    runtime.to_csv(out / "runtime.csv", index=False)
    model_summary = overall.reset_index()
    model_summary.to_csv(out / "model_summary.csv", index=False)
    safety.to_csv(out / "safety_admissibility.csv", index=False)
    write_figures(overall, windows, segments, paired_summary)

    print(f"\nFUSION_REGION = {region}")
    for note in notes:
        print("  -", note)
    (out / "gate_decision.json").write_text(
        json.dumps({"fusion_region": region, "notes": notes, "chance": CHANCE,
                    "material": MATERIAL, "anchor": ANCHOR}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""E4-B analysis: Weak feature recovery panel (F0 SELECTED222 / F1 LITERATURE240 / F2 ALL259_WEAK_RECOVERY).

Aggregates the per-run predictions produced by run_e4_b.py (F1/F2 fresh) and the reused E2-E1 Q2
runs (F0) into the full plan29 §18 artifact set:

    feature_profile_audit.csv/json   eligibility_identity.csv
    daily_metrics.csv               model_summary.csv
    window_metrics.csv              hour_segment_metrics.csv
    paired_summary.csv              slot_gain_summary.csv
    family_diagnostics.csv          component_diagnostics.csv
    runtime.csv                     figures/
    E4_B_summary.md                 E4_B_GATE.md

F0 reuses the frozen Q2 runs (selected222 == frozen selector). F1/F2 are fresh deep Q2 trainings
under the canonical frozen-split Q2 dir_only route; only the experiment feature-recovery profile
(and hence Weak input routing) changes. No new raw source, no postprocess, no class weighting.
"""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.TafM_改进源码.config import default_selector_path  # noqa: E402
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector  # noqa: E402
from src.TafM_改进源码.train import build_experiment_feature_manifest  # noqa: E402

H = Path(__file__).resolve().parent
ARMS = ("F0", "F1", "F2")
W = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
     "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
SEGMENTS = (("H1", 1, 8), ("H2", 9, 16), ("H3", 17, 24))
BENCH_DAY = "2026-02-13"
Q2_ROOT = REPO / "experiments/first_test/E2_architecture/horizon_specialized_head/runs"
E4B_ROOT = H / "runs"

WINDOW_OF = {}
for w, (a, b) in W.items():
    for i in range(7):
        WINDOW_OF[(pd.Timestamp(a) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")] = w


def collect_runs():
    """Return {arm: [(target_day, run_dir, status)]}."""
    out = {a: [] for a in ARMS}
    for rec in sorted(Q2_ROOT.glob("E2E1-Q2-*/RUN_RECORD.json")):
        d = json.loads(rec.read_text(encoding="utf-8"))
        if d.get("status") == "PASS":
            out["F0"].append((d["target_day"], d["run_dir"], "PASS"))
    for arm, pat in (("F1", "E4B-F1-*"), ("F2", "E4B-F2-*")):
        for rec in sorted(E4B_ROOT.glob(f"{pat}/RUN_RECORD.json")):
            d = json.loads(rec.read_text(encoding="utf-8"))
            if d.get("status") == "PASS":
                out[arm].append((d["target_day"], d["run_dir"], "PASS"))
    return out


def load_slots(runs, arms_present):
    frames = []
    for arm in arms_present:
        for day, run_dir, _ in runs[arm]:
            p = Path(run_dir) / "predictions.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p).copy()
            df["arm"] = arm
            df["target_day"] = day
            df["scope"] = WINDOW_OF.get(day, "NA")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def metric(q, scope, arm=None):
    if len(q) == 0:
        return {"arm": arm or "NA", "scope": scope, "n_slots": 0, "n_days": 0,
                "raw": np.nan, "balanced": np.nan, "positive_recall": np.nan, "negative_recall": np.nan,
                "auc": np.nan, "brier": np.nan, "predicted_positive_fraction": np.nan,
                "true_positive_fraction": np.nan, "one_class_days": 0,
                "positive_recall_zero_days": 0, "negative_recall_zero_days": 0}
    y = q.y_true_model.to_numpy() > 0
    p = q.direction_hat.to_numpy().astype(bool)
    prob = q.p_positive.to_numpy()
    from sklearn.metrics import roc_auc_score, brier_score_loss
    tp = int((p & y).sum()); tn = int((~p & ~y).sum())
    pos = int(y.sum()); neg = int((~y).sum())
    pr = tp / pos if pos else np.nan
    nr = tn / neg if neg else np.nan
    g = list(q.groupby("target_day"))
    zpos = sum(((z.y_true_model.to_numpy() > 0).sum() > 0) and
               (z.loc[z.y_true_model > 0, "direction_hat"].sum() == 0) for _, z in g)
    zneg = sum(((z.y_true_model.to_numpy() <= 0).sum() > 0) and
               (z.loc[z.y_true_model <= 0, "direction_hat"].sum() == len(z.loc[z.y_true_model <= 0])) for _, z in g)
    return {"arm": q.arm.iloc[0], "scope": scope, "n_slots": len(q), "n_days": q.target_day.nunique(),
            "raw": float((p == y).mean()) if len(q) else np.nan,
            "balanced": float(np.nanmean([pr, nr])) if len(q) else np.nan,
            "positive_recall": float(pr), "negative_recall": float(nr),
            "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
            "brier": float(brier_score_loss(y, prob)) if len(y) else np.nan,
            "predicted_positive_fraction": float(p.mean()),
            "true_positive_fraction": float(y.mean()),
            "one_class_days": int(q.groupby("target_day").direction_hat.nunique().eq(1).sum()),
            "positive_recall_zero_days": int(zpos), "negative_recall_zero_days": int(zneg)}


def bootstrap(day, rng, pairs):
    out = []
    for a, b in pairs:
        if a not in set(day.arm) or b not in set(day.arm):
            continue
        x = day[day.arm == a].set_index("target_day").sort_index()
        y = day[day.arm == b].set_index("target_day").sort_index()
        common = x.index.intersection(y.index)
        if len(common) < 2:
            continue
        x = x.loc[common]; y = y.loc[common]
        dr = x.raw.to_numpy() - y.raw.to_numpy()
        db = x.balanced.to_numpy() - y.balanced.to_numpy()
        ix = rng.integers(0, len(dr), (10000, len(dr)))
        rc = np.quantile(dr[ix].mean(1), [.025, .975]); bc = np.quantile(db[ix].mean(1), [.025, .975])
        signs = []
        for w in W:
            vals = dr[np.array([z == w for z in x.scope])]
            signs.append(float(np.sign(vals.mean())) if len(vals) else np.nan)
        out.append({"comparison": f"{a}-{b}", "mean_delta_raw": float(dr.mean()),
                    "raw_ci_low": float(rc[0]), "raw_ci_high": float(rc[1]),
                    "raw_ci_excludes_zero": bool(rc[0] > 0 or rc[1] < 0),
                    "raw_W_T_L": f"{(dr > 0).sum()}/{(dr == 0).sum()}/{(dr < 0).sum()}",
                    "mean_delta_balanced": float(db.mean()), "balanced_ci_low": float(bc[0]),
                    "balanced_ci_high": float(bc[1]),
                    "window_delta_raw_signs": json.dumps(signs)})
    return out


def safety_block(summary, day, arms_present):
    c = summary.set_index("arm")
    rows = {}
    f0 = "F0" if "F0" in c.index else arms_present[0]
    for a in arms_present:
        r = c.loc[a]; z = day[day.arm == a]
        checks = {"balanced": bool(r.balanced >= c.loc[f0].balanced - .01),
                  "positive_recall": bool(r.positive_recall >= c.loc[f0].positive_recall - .05),
                  "one_class_days": int(z.one_class_days.sum()) <= int(day[day.arm == f0].one_class_days.sum()) + 2,
                  "positive_recall_zero_days": int(z.positive_recall_zero_days.sum()) <=
                  int(day[day.arm == f0].positive_recall_zero_days.sum()) + 2}
        rows[a] = {"admissible": all(checks.values()), "checks": checks}
    return rows


def decide(paired, safety, summary, slot_overall, arms_present):
    pp = {x["comparison"]: x for x in paired}
    c = summary.set_index("arm")
    f0 = "F0" if "F0" in c.index else arms_present[0]
    gain = {a: float(c.loc[a].raw - c.loc[f0].raw) for a in arms_present}
    safe = {a: bool(safety[a]["admissible"]) for a in arms_present}

    def stable(a, b, sign):
        if f"{a}-{b}" not in pp:
            return False
        x = pp[f"{a}-{b}"]
        s = np.sign(json.loads(x["window_delta_raw_signs"]))
        return bool(x["raw_ci_excludes_zero"] and abs(x["mean_delta_raw"]) >= .02 and int((s == sign).sum()) >= 3)

    def beats_in_windows(a, b):
        if f"{a}-{b}" not in pp:
            return 0
        x = pp[f"{a}-{b}"]
        s = np.sign(json.loads(x["window_delta_raw_signs"]))
        return int((s == 1).sum())

    flags = {"gain_vs_F0": gain, "slots_vs_F0": slot_overall, "safe": safe}
    if "F1" in arms_present:
        flags.update({"F1_stable_better_than_F0": stable("F1", "F0", 1),
                      "F1_windows_beat_F0": beats_in_windows("F1", "F0")})
    if "F2" in arms_present:
        flags.update({"F2_stable_better_than_F0": stable("F2", "F0", 1),
                      "F2_windows_beat_F0": beats_in_windows("F2", "F0"),
                      "F2_beats_F1": stable("F2", "F1", 1),
                      "F2_windows_beat_F1": beats_in_windows("F2", "F1")})

    f1_improves = ("F1" in arms_present and gain.get("F1", -9) >= .02 and safe.get("F1") and stable("F1", "F0", 1))
    f2_improves = ("F2" in arms_present and gain.get("F2", -9) >= .02 and safe.get("F2") and stable("F2", "F0", 1))
    f1_prom = ("F1" in arms_present and safe.get("F1") and (gain.get("F1", 0) >= .02 or c.loc["F1"].raw >= .62))
    f2_prom = ("F2" in arms_present and safe.get("F2") and (gain.get("F2", 0) >= .02 or c.loc["F2"].raw >= .62))
    f1_harm = ("F1" in arms_present and ((not safe.get("F1")) or gain.get("F1", 0) <= -0.02))
    f2_harm = ("F2" in arms_present and ((not safe.get("F2")) or gain.get("F2", 0) <= -0.02))

    if f1_improves and (not f2_improves or not flags.get("F2_beats_F1") or
                         flags.get("F2_windows_beat_F1", 0) < flags.get("F1_windows_beat_F0", 0)):
        return "DOMAIN_GUIDED_RECOVERY", flags
    if f2_improves and flags.get("F2_beats_F1"):
        return "BROAD_SELECTOR_BOTTLENECK", flags
    if f1_prom or f2_prom:
        return "RECOVERY_PROMISING_UNPROVEN", flags
    if f1_harm or f2_harm:
        return "RECOVERY_HARMFUL", flags
    if (not f1_improves and not f2_improves and safe.get("F1", True) and safe.get("F2", True)
            and abs(gain.get("F1", 0)) < .02 and abs(gain.get("F2", 0)) < .02):
        return "SELECTOR_ROBUST", flags
    return "NO_CLEAR_RECOVERY_EFFECT", flags


def main():
    runs = collect_runs()
    present = {a: len(runs[a]) for a in ARMS}
    print("run counts:", present)
    arms_present = [a for a in ARMS if present[a] >= 1]
    pred = load_slots(runs, arms_present)
    pred.to_parquet(H / "derived_predictions.parquet", index=False)

    day_rows = []
    for (a, d, w), z in pred.groupby(["arm", "target_day", "scope"]):
        day_rows.append({**metric(z.reset_index(drop=True), w, a), "target_day": d})
    day = pd.DataFrame(day_rows)
    day.to_csv(H / "daily_metrics.csv", index=False)

    over = pd.DataFrame([metric(pred[pred.arm == a], "overall", a) for a in arms_present])
    win = pd.DataFrame([metric(pred[(pred.arm == a) & (pred.scope == w)], w, a) for a in arms_present for w in W])
    win.to_csv(H / "window_metrics.csv", index=False)
    hour = pd.DataFrame([metric(pred[(pred.arm == a) & pred.hour_business.between(lo, hi)], h, a)
                        for a in arms_present for h, lo, hi in SEGMENTS])
    hour.to_csv(H / "hour_segment_metrics.csv", index=False)
    summary = over.copy()
    summary["min_window_raw"] = [win[win.arm == a].raw.min() for a in summary.arm]
    summary["window_raw_std"] = [win[win.arm == a].raw.std(ddof=0) for a in summary.arm]
    summary.to_csv(H / "model_summary.csv", index=False)

    rng = np.random.default_rng(20260924)
    pairs = [("F1", "F0"), ("F2", "F0"), ("F1", "F2")]
    paired = bootstrap(day, rng, pairs)
    pd.DataFrame(paired).to_csv(H / "paired_summary.csv", index=False)

    safety = safety_block(summary, day, arms_present)
    (H / "safety_admissibility.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")

    gain_rows = []
    for a in arms_present:
        for name, lo, hi in [("overall", 1, 24), *SEGMENTS]:
            qa = pred[(pred.arm == a) & pred.hour_business.between(lo, hi)]
            q0 = pred[(pred.arm == "F0") & pred.hour_business.between(lo, hi)]
            ca = int((qa.direction_hat.to_numpy().astype(bool) == (qa.y_true_model.to_numpy() > 0)).sum()) if len(qa) else 0
            c0 = int((q0.direction_hat.to_numpy().astype(bool) == (q0.y_true_model.to_numpy() > 0)).sum()) if len(q0) else 0
            gain_rows.append({"arm": a, "scope": name, "correct_slots_arm": ca, "correct_slots_F0": c0,
                              "delta_correct_slots": ca - c0, "n_slots": len(qa),
                              "delta_slots_per_day": (ca - c0) / max(present.get("F0", 1), 1)})
    pd.DataFrame(gain_rows).to_csv(H / "slot_gain_summary.csv", index=False)
    slot_overall = {r["arm"]: r["delta_correct_slots"] for r in gain_rows if r["scope"] == "overall"}

    # ---- feature profile audit (in-memory experiment manifest) ----
    store = SequenceStore.load()
    frozen, fhash = load_frozen_selector(default_selector_path())
    fa_rows = []
    f0m = build_experiment_feature_manifest(frozen, "selected222", store)
    fa_rows.append({"arm": "F0", "feature_recovery_profile": "selected222",
                    "selected_feature_count": len(f0m["selected_features"]),
                    "strong_count": 211, "weak_count": len([r for r in frozen["feature_roles"] if r["role"] == "Weak"]),
                    "recovered_count": 0, "recovered_role": None,
                    "base_selector_sha256": f0m.get("selector_sha256"),
                    "experiment_feature_profile_sha256": None})
    for arm, prof in (("F1", "literature240"), ("F2", "all259")):
        m = build_experiment_feature_manifest(frozen, prof, store)
        fa_rows.append({"arm": arm, "feature_recovery_profile": prof,
                        "selected_feature_count": len(m["selected_features"]),
                        "strong_count": m["experiment_strong_count"], "weak_count": m["experiment_weak_count"],
                        "recovered_count": len(m["experiment_recovered_features"]), "recovered_role": "Weak",
                        "base_selector_sha256": m["selector_sha256"],
                        "experiment_feature_profile_sha256": m["experiment_feature_profile_sha256"]})
    pd.DataFrame(fa_rows).to_csv(H / "feature_profile_audit.csv", index=False)
    (H / "feature_profile_audit.json").write_text(json.dumps(fa_rows, indent=2), encoding="utf-8")

    # ---- eligibility identity across 28 formal days ----
    elig_rows = []
    manifests = {p: build_experiment_feature_manifest(frozen, p, store) for p in ("selected222", "literature240", "all259")}
    for d in sorted(WINDOW_OF):
        sets = {p: set(store.eligibility(manifests[p], current_target_day=d)["eligible_indices"]) for p in manifests}
        base = sets["selected222"]
        elig_rows.append({"target_day": d, "window": WINDOW_OF[d],
                          "selected222_eligible": len(sets["selected222"]),
                          "literature240_eligible": len(sets["literature240"]),
                          "all259_eligible": len(sets["all259"]),
                          "identical_across_profiles": bool(sets["literature240"] == base and sets["all259"] == base)})
    pd.DataFrame(elig_rows).to_csv(H / "eligibility_identity.csv", index=False)

    # ---- runtime + component diagnostics ----
    rt_rows = []; comp_rows = []
    for arm in arms_present:
        for day_, run_dir, _ in runs[arm]:
            m = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8")) if (Path(run_dir) / "manifest.json").exists() else {}
            rt_rows.append({"arm": arm, "target_day": day_,
                           "wall_seconds": m.get("wall_time_total_seconds"),
                           "best_epoch": m.get("best_epoch"), "stop_epoch": m.get("stop_epoch"),
                           "parameter_count": m.get("parameter_count"),
                           "cuda_peak_memory_mb": (m.get("cuda_peak_memory_bytes") or 0) / 1e6,
                           "epoch_seconds_mean": m.get("training_epoch_seconds_mean"),
                           "device": m.get("device"), "amp": m.get("amp")})
            comp_rows.append({"arm": arm, "target_day": day_,
                             "parameter_count": m.get("parameter_count"),
                             "selected_feature_count": m.get("selected_feature_count") or len(m.get("selected_features", []) or []),
                             "strong_count": m.get("experiment_strong_count"),
                             "weak_count": m.get("experiment_weak_count"),
                             "weak_input_dim": m.get("experiment_weak_count"),
                             "experiment_feature_profile": m.get("feature_recovery_profile"),
                             "best_epoch": m.get("best_epoch"), "stop_epoch": m.get("stop_epoch"),
                             "cuda_peak_memory_mb": (m.get("cuda_peak_memory_bytes") or 0) / 1e6})
    pd.DataFrame(rt_rows).to_csv(H / "runtime.csv", index=False)
    comp = pd.DataFrame(comp_rows)
    comp_agg = comp.groupby("arm").agg(
        parameter_count=("parameter_count", "mean"),
        weak_input_dim=("weak_input_dim", "mean"),
        strong_count=("strong_count", "mean"),
        weak_count=("weak_count", "mean"),
        best_epoch=("best_epoch", "mean"),
        stop_epoch=("stop_epoch", "mean"),
        cuda_peak_memory_mb=("cuda_peak_memory_mb", "mean")).reset_index()
    comp_agg.to_csv(H / "component_diagnostics.csv", index=False)

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (H / "figures").mkdir(exist_ok=True)
        ax = summary.set_index("arm")[["raw", "balanced", "auc"]].plot.bar()
        ax.figure.tight_layout(); ax.figure.savefig(H / "figures/overall_metrics.png", dpi=150); plt.close(ax.figure)
        ax = win.pivot(index="scope", columns="arm", values="raw").reindex(list(W)).plot.bar(ylim=(.3, .85))
        ax.figure.tight_layout(); ax.figure.savefig(H / "figures/window_raw.png", dpi=150); plt.close(ax.figure)
        ax = hour.pivot(index="scope", columns="arm", values="raw").reindex(["H1", "H2", "H3"]).plot.bar(ylim=(.3, .85))
        ax.figure.tight_layout(); ax.figure.savefig(H / "figures/segment_raw.png", dpi=150); plt.close(ax.figure)
        pp = pd.DataFrame(paired)
        fig, ax = plt.subplots(figsize=(6, 3)); yy = np.arange(len(pp))
        ax.errorbar(pp.mean_delta_raw, yy, xerr=[pp.mean_delta_raw - pp.raw_ci_low, pp.raw_ci_high - pp.mean_delta_raw],
                    fmt="o", capsize=4); ax.axvline(0, color="black", lw=1)
        ax.axvline(.02, color="gray", ls="--"); ax.axvline(-.02, color="gray", ls="--")
        ax.set_yticks(yy, pp.comparison); ax.set_xlabel("Day-paired Delta Raw with 95% bootstrap CI")
        fig.tight_layout(); fig.savefig(H / "figures/paired_raw_ci.png", dpi=150); plt.close(fig)
    except Exception as exc:
        print("figure generation skipped:", exc)

    label, flags = decide(paired, safety, summary, slot_overall, arms_present)
    (H / "E4_B_GATE.md").write_text(
        f"# E4-B Gate\n\n**E4_B_GATE FEATURE_RECOVERY_SIGNAL = `{label}`**\n\n"
        "Uses only the frozen W1-W4 panel, paired day-cluster bootstrap (10,000 draws, seed 20260924), "
        "cross-window evidence and F0 safety. Benchmark excluded from ranking. Stop for human review.\n"
        f"\nFlags: {json.dumps(flags, ensure_ascii=False)}\n", encoding="utf-8")

    result = {"signal": label, "flags": flags, "present_runs": present,
              "overall": summary.to_dict("records"), "paired": paired,
              "safety": {a: safety[a]["admissible"] for a in safety},
              "slot_gain": gain_rows, "window": win.to_dict("records"),
              "hour": hour.to_dict("records"),
              "feature_profile_audit": fa_rows,
              "eligibility_identical": bool(all(r["identical_across_profiles"] for r in elig_rows))}
    (H / "E4_B_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"signal": label, "flags": flags, "overall": summary.to_dict("records")}, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

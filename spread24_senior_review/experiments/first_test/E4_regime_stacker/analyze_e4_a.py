"""E4-A analysis: metrics, paired bootstrap, slot gains, safety, calibration audit, gate + summary."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
ARMS = ("P0", "P1", "P2", "P3")
W = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
     "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
SEGMENTS = (("H1", 1, 8), ("H2", 9, 16), ("H3", 17, 24))
P0_BASE = "experiments/first_test/E2_architecture/horizon_specialized_head/runs"


def metric(q, scope):
    y = q.y_true_model.to_numpy() > 0
    p = q.direction_hat.to_numpy().astype(bool)
    prob = q.p_positive.to_numpy()
    from sklearn.metrics import roc_auc_score, brier_score_loss
    tp = (p & y).sum(); tn = (~p & ~y).sum(); pos = y.sum(); neg = (~y).sum()
    pr = tp / pos if pos else np.nan
    nr = tn / neg if neg else np.nan
    g = list(q.groupby("target_day"))
    zpos = sum(((z.y_true_model.to_numpy() > 0).sum() > 0) and (z.loc[z.y_true_model > 0, "direction_hat"].sum() == 0) for _, z in g)
    zneg = sum(((z.y_true_model.to_numpy() <= 0).sum() > 0) and
               (z.loc[z.y_true_model <= 0, "direction_hat"].sum() == len(z.loc[z.y_true_model <= 0])) for _, z in g)
    return {"arm": q.arm.iloc[0], "scope": scope, "n_slots": len(q), "n_days": q.target_day.nunique(),
            "raw": float((p == y).mean()), "balanced": float(np.nanmean([pr, nr])),
            "positive_recall": float(pr), "negative_recall": float(nr),
            "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
            "brier": float(brier_score_loss(y, prob)), "predicted_positive_fraction": float(p.mean()),
            "true_positive_fraction": float(y.mean()),
            "one_class_days": int(q.groupby("target_day").direction_hat.nunique().eq(1).sum()),
            "positive_recall_zero_days": int(zpos), "negative_recall_zero_days": int(zneg)}


def bootstrap(day, rng, pairs):
    out = []
    for a, b in pairs:
        x = day[day.arm == a].set_index("target_day").sort_index()
        y = day[day.arm == b].set_index("target_day").sort_index()
        dr = x.raw.to_numpy() - y.raw.to_numpy()
        db = x.balanced.to_numpy() - y.balanced.to_numpy()
        ix = rng.integers(0, len(dr), (10000, len(dr)))
        rc = np.quantile(dr[ix].mean(1), [.025, .975]); bc = np.quantile(db[ix].mean(1), [.025, .975])
        signs = [float(np.sign(dr[np.array([z == w for z in x.scope])].mean())) for w in W]
        out.append({"comparison": f"{a}-{b}", "mean_delta_raw": dr.mean(), "raw_ci_low": rc[0], "raw_ci_high": rc[1],
                    "raw_ci_excludes_zero": bool(rc[0] > 0 or rc[1] < 0),
                    "raw_W_T_L": f"{(dr>0).sum()}/{(dr==0).sum()}/{(dr<0).sum()}",
                    "mean_delta_balanced": db.mean(), "balanced_ci_low": bc[0], "balanced_ci_high": bc[1],
                    "window_delta_raw_signs": json.dumps(signs)})
    return out


def decide(paired, safety, summary, slot_overall):
    pp = {x["comparison"]: x for x in paired}

    def stable(a, b, sign):
        x = pp[f"{a}-{b}"]
        s = np.sign(json.loads(x["window_delta_raw_signs"]))
        return bool(x["raw_ci_excludes_zero"] and abs(x["mean_delta_raw"]) >= .02 and int((s == sign).sum()) >= 3)
    idx = summary.set_index("arm")
    gain = {a: float(idx.loc[a].raw - idx.loc["P0"].raw) for a in ARMS}
    safe = {a: bool(safety[a]["admissible"]) for a in ARMS}
    flags = {"gain_vs_P0": gain, "slots_vs_P0": slot_overall, "safe": safe,
             "P2_better_than_P1": stable("P2", "P1", 1), "P3_better_than_P1": stable("P3", "P1", 1),
             "P3_better_than_P2": stable("P3", "P2", 1), "P3_better_than_P0": stable("P3", "P0", 1),
             "P1_worse_than_P0": stable("P1", "P0", -1),
             "P2_worse_than_P1": stable("P2", "P1", -1), "P3_worse_than_P1": stable("P3", "P1", -1)}
    if safe["P3"] and flags["P3_better_than_P1"] and (gain["P3"] >= -0.005 or flags["P3_better_than_P0"]):
        return "REGIME_STACKING_CONFIRMED", flags
    if safe["P2"] and flags["P2_better_than_P1"] and not flags["P3_better_than_P2"]:
        return "SEGMENT_CALIBRATION_HELPFUL", flags
    prom2 = safe["P2"] and (gain["P2"] >= .02 or idx.loc["P2"].raw >= .62)
    prom3 = safe["P3"] and (gain["P3"] >= .02 or idx.loc["P3"].raw >= .62)
    if prom2 or prom3:
        return "REGIME_PROMISING_UNPROVEN", flags
    if flags["P1_worse_than_P0"] and not flags["P2_better_than_P1"] and not flags["P3_better_than_P1"]:
        return "SPLIT_COST_DOMINATES", flags
    if (not safe["P2"]) or (not safe["P3"]) or flags["P2_worse_than_P1"] or flags["P3_worse_than_P1"]:
        return "POSTPROCESS_HARMFUL", flags
    return "NO_POSTPROCESS_GAIN", flags


def main():
    pred = pd.read_parquet(H / "derived_predictions.parquet")
    day_rows = []
    for (a, d, w), z in pred.groupby(["arm", "target_day", "scope"]):
        day_rows.append({**metric(z.reset_index(drop=True), w), "target_day": d})
    day = pd.DataFrame(day_rows)
    day.to_csv(H / "daily_metrics.csv", index=False)
    over = pd.DataFrame([metric(pred[pred.arm == a], "overall") for a in ARMS])
    win = pd.DataFrame([metric(pred[(pred.arm == a) & (pred.scope == w)], w) for a in ARMS for w in W])
    win.to_csv(H / "window_metrics.csv", index=False)
    hour = pd.DataFrame([metric(pred[(pred.arm == a) & pred.hour_business.between(lo, hi)], h)
                         for a in ARMS for h, lo, hi in SEGMENTS])
    hour.to_csv(H / "hour_segment_metrics.csv", index=False)
    summary = over.copy()
    summary["min_window_raw"] = [win[win.arm == a].raw.min() for a in summary.arm]
    summary["window_raw_std"] = [win[win.arm == a].raw.std(ddof=0) for a in summary.arm]
    summary.to_csv(H / "model_summary.csv", index=False)

    rng = np.random.default_rng(20260924)
    pairs = [("P1", "P0"), ("P2", "P1"), ("P3", "P1"), ("P2", "P0"), ("P3", "P0"), ("P3", "P2")]
    paired = bootstrap(day, rng, pairs)
    pd.DataFrame(paired).to_csv(H / "paired_summary.csv", index=False)

    c = summary.set_index("arm")
    safety = {}
    for a in ARMS:
        r = c.loc[a]; z = day[day.arm == a]
        checks = {"balanced": bool(r.balanced >= c.loc["P0"].balanced - .01),
                  "positive_recall": bool(r.positive_recall >= c.loc["P0"].positive_recall - .05),
                  "one_class_days": int(z.one_class_days.sum()) <= int(day[day.arm == "P0"].one_class_days.sum()) + 2,
                  "positive_recall_zero_days": int(z.positive_recall_zero_days.sum()) <= int(day[day.arm == "P0"].positive_recall_zero_days.sum()) + 2}
        safety[a] = {"admissible": all(checks.values()), "checks": checks}
    (H / "safety_admissibility.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")

    gain_rows = []
    for a in ARMS:
        for name, lo, hi in [("overall", 1, 24), *SEGMENTS]:
            qa = pred[(pred.arm == a) & pred.hour_business.between(lo, hi)]
            q0 = pred[(pred.arm == "P0") & pred.hour_business.between(lo, hi)]
            ca = int((qa.direction_hat.to_numpy().astype(bool) == (qa.y_true_model.to_numpy() > 0)).sum())
            c0 = int((q0.direction_hat.to_numpy().astype(bool) == (q0.y_true_model.to_numpy() > 0)).sum())
            gain_rows.append({"arm": a, "scope": name, "correct_slots_arm": ca, "correct_slots_P0": c0,
                              "delta_correct_slots": ca - c0, "n_slots": len(qa), "delta_slots_per_day": (ca - c0) / 28})
    pd.DataFrame(gain_rows).to_csv(H / "slot_gain_summary.csv", index=False)
    slot_overall = {r["arm"]: r["delta_correct_slots"] for r in gain_rows if r["scope"] == "overall"}

    label, flags = decide(paired, safety, summary, slot_overall)
    (H / "E4_A_GATE.md").write_text(
        f"# E4-A Gate\n\n**E4_A_GATE POSTPROCESS_SIGNAL = `{label}`**\n\n"
        "Uses only the frozen W1-W4 panel, paired day-cluster bootstrap (10,000 draws, seed 20260924), "
        "cross-window evidence and P0 safety. Benchmark excluded. Stop for human review.\n"
        f"\nFlags: {json.dumps(flags)}\n", encoding="utf-8")

    # runtime split into deep-run and postprocess parts
    ck = pd.read_csv(H / "checkpoint_identity.csv")
    coef = pd.read_csv(H / "stacker_coefficients.csv")
    rt = ck[["target_day", "wall_seconds", "train_wall_seconds", "best_epoch", "stop_epoch", "parameter_count"]].copy()
    rt["arm"] = "P1_deep"
    rt.to_csv(H / "runtime.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = summary.set_index("arm")[["raw", "balanced", "auc"]].plot.bar()
    ax.figure.tight_layout(); ax.figure.savefig(H / "figures/overall_metrics.png", dpi=150); plt.close(ax.figure)
    ax = win.pivot(index="scope", columns="arm", values="raw").reindex(list(W)).plot.bar(ylim=(.3, .85))
    ax.figure.tight_layout(); ax.figure.savefig(H / "figures/window_raw.png", dpi=150); plt.close(ax.figure)
    ax = hour.pivot(index="scope", columns="arm", values="raw").reindex(["H1", "H2", "H3"]).plot.bar(ylim=(.3, .85))
    ax.figure.tight_layout(); ax.figure.savefig(H / "figures/segment_raw.png", dpi=150); plt.close(ax.figure)
    pp = pd.DataFrame(paired).iloc[[1, 2]]
    fig, ax = plt.subplots(figsize=(6, 3)); yy = np.arange(len(pp))
    ax.errorbar(pp.mean_delta_raw, yy, xerr=[pp.mean_delta_raw - pp.raw_ci_low, pp.raw_ci_high - pp.mean_delta_raw],
                fmt="o", capsize=4); ax.axvline(0, color="black", lw=1)
    ax.axvline(.02, color="gray", ls="--"); ax.axvline(-.02, color="gray", ls="--")
    ax.set_yticks(yy, pp.comparison); ax.set_xlabel("Day-paired Delta Raw with 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(H / "figures/paired_raw_ci.png", dpi=150); plt.close(fig)

    json.dump({"signal": label, "flags": flags, "overall": summary.to_dict("records"),
               "paired": paired, "safety": safety, "slot_gain": gain_rows},
              open(H / "benchmark" / "formal_summary.json", "w", encoding="utf-8"), indent=2, default=float)
    print(json.dumps({"signal": label, "flags": flags, "overall": summary.to_dict("records")}, indent=2, default=float))


if __name__ == "__main__":
    main()

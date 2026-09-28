"""Analyze the frozen E2-D1 panel; target-day bootstrap is the paired unit.

N0 reuses the frozen F80 control; N1/N2 are fresh E2-D1 runs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
F80 = HERE.parent / "fusion_coarse" / "runs"
W = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
     "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
ARMS = ("N0", "N1", "N2")
EXPECTED_ENC = {"N0": {"canonical", None}, "N1": {"raw_only"}, "N2": {"ple_only"}}
FROZEN = {"config": "8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c",
          "selector": "ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f",
          "source": "a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea",
          "sequence": "9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82"}


def rec(a, d, benchmark=False):
    if a == "N0":
        return json.loads((F80 / f"E2B1-F80-{d}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
    base = HERE / "runs" / ("benchmark" if benchmark else "")
    return json.loads((base / f"E2D1-{a}-{d}" / "RUN_RECORD.json").read_text(encoding="utf-8"))


def frame(a, d, benchmark=False):
    r = rec(a, d, benchmark)
    folder = Path(r["run_dir"])
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    q = pd.read_parquet(folder / "predictions.parquet")
    checks = {"status": r.get("status") == "PASS", "mode": m.get("mode") == "A2",
              "profile": m.get("profile") == "default",
              "device": str(m.get("device", "")).startswith("cuda"), "amp": m.get("amp") is True,
              "seed": m.get("seed") == 20260924, "objective": m.get("objective_mode") == "dir_only",
              "checkpoint": m.get("checkpoint_policy") == "direction_first",
              "gradient": m.get("gradient_policy") == "vanilla",
              "architecture": m.get("architecture_mode") == "full_current",
              "tabular_route": m.get("direction_tabular_mode", "current") == "current",
              "gate_route": m.get("direction_horizon_gate_mode", "current") == "current",
              "role_profile": m.get("strong_role_profile", "all") == "all",
              "alpha": abs(float(m.get("direction_fusion_alpha", -1)) - .8) < 1e-12,
              "encoding": m.get("numeric_encoding_mode") in EXPECTED_ENC[a],
              "config": m.get("config_sha256") == FROZEN["config"],
              "selector": m.get("selector_sha256") == FROZEN["selector"],
              "source": m.get("source_sha256") == FROZEN["source"],
              "sequence": m.get("sequence_manifest_sha256") == FROZEN["sequence"]}
    if not all(checks.values()) or len(q) != 24 or q.hour_business.tolist() != list(range(1, 25)):
        raise RuntimeError(f"invalid/missing run {a} {d}: {checks}")
    q["arm"] = a
    q["target_day"] = d
    return q


def metric(q, scope):
    y = q.y_true_model.to_numpy() > 0
    p = q.direction_hat.to_numpy().astype(bool)
    prob = q.p_positive.to_numpy()
    from sklearn.metrics import roc_auc_score, brier_score_loss
    tp = (p & y).sum(); tn = (~p & ~y).sum(); pos = y.sum(); neg = (~y).sum()
    pr = tp / pos if pos else np.nan
    nr = tn / neg if neg else np.nan
    groups = list(q.groupby("target_day"))
    zero_pos = sum(((z.y_true_model.to_numpy() > 0).sum() > 0) and
                   (z.loc[z.y_true_model > 0, "direction_hat"].astype(bool).sum() == 0) for _, z in groups)
    zero_neg = sum(((z.y_true_model.to_numpy() <= 0).sum() > 0) and
                   (z.loc[z.y_true_model <= 0, "direction_hat"].astype(bool).sum() == len(z.loc[z.y_true_model <= 0]))
                   for _, z in groups)
    return {"arm": q.arm.iloc[0], "scope": scope, "n_slots": len(q), "n_days": q.target_day.nunique(),
            "raw": float((p == y).mean()), "balanced": float(np.nanmean([pr, nr])),
            "positive_recall": float(pr), "negative_recall": float(nr),
            "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
            "brier": float(brier_score_loss(y, prob)), "predicted_positive_fraction": float(p.mean()),
            "true_positive_fraction": float(y.mean()),
            "one_class_days": int(q.groupby("target_day").direction_hat.nunique().eq(1).sum()),
            "positive_recall_zero_days": int(zero_pos), "negative_recall_zero_days": int(zero_neg)}


def bootstrap(day, rng):
    paired = []
    for a, b in (("N1", "N0"), ("N2", "N0"), ("N1", "N2")):
        x = day[day.arm == a].set_index("target_day").sort_index()
        y = day[day.arm == b].set_index("target_day").sort_index()
        dr = x.raw.to_numpy() - y.raw.to_numpy()
        db = x.balanced.to_numpy() - y.balanced.to_numpy()
        da = x.auc.to_numpy() - y.auc.to_numpy()
        ix = rng.integers(0, 28, (10000, 28))
        rc = np.quantile(dr[ix].mean(1), [.025, .975])
        bc = np.quantile(db[ix].mean(1), [.025, .975])
        ac = np.quantile(da[ix].mean(1), [.025, .975])
        signs = [float(np.sign(dr[np.array([z == w for z in x.scope])].mean())) for w in W]
        paired.append({"comparison": f"{a}-{b}", "mean_delta_raw": dr.mean(),
                       "raw_ci_low": rc[0], "raw_ci_high": rc[1],
                       "raw_ci_excludes_zero": bool(rc[0] > 0 or rc[1] < 0),
                       "raw_W_T_L": f"{(dr>0).sum()}/{(dr==0).sum()}/{(dr<0).sum()}",
                       "mean_delta_balanced": db.mean(), "balanced_ci_low": bc[0], "balanced_ci_high": bc[1],
                       "balanced_ci_excludes_zero": bool(bc[0] > 0 or bc[1] < 0),
                       "mean_delta_auc": da.mean(), "auc_ci_low": ac[0], "auc_ci_high": ac[1],
                       "window_delta_raw_signs": json.dumps(signs)})
    return paired


def decide(paired, safety):
    p = {x["comparison"]: x for x in paired}

    def stably(a, b, sign):
        x = p[f"{a}-{b}"]
        s = np.sign(json.loads(x["window_delta_raw_signs"]))
        return (x["raw_ci_excludes_zero"] and abs(x["mean_delta_raw"]) >= .02 and
                int((s == sign).sum()) >= 3)
    n1_worse = stably("N1", "N0", -1)
    n2_worse = stably("N2", "N0", -1)
    n1_better = stably("N1", "N0", 1)
    n2_better = stably("N2", "N0", 1)
    n1_safe = bool(safety["N1"]["admissible"])
    n2_safe = bool(safety["N2"]["admissible"])
    if n1_better and n1_safe:
        return "RAW_ONLY_BETTER", (n1_worse, n2_worse, n1_better, n2_better)
    if n2_better and n2_safe:
        return "PLE_ONLY_BETTER", (n1_worse, n2_worse, n1_better, n2_better)
    if n1_worse and n2_worse:
        return "PLE_RAW_SYNERGY", (n1_worse, n2_worse, n1_better, n2_better)
    if n1_worse and not n2_worse:
        return "PLE_REQUIRED", (n1_worse, n2_worse, n1_better, n2_better)
    if n2_worse and not n1_worse:
        return "RAW_SKIP_REQUIRED", (n1_worse, n2_worse, n1_better, n2_better)
    if n1_safe and not n1_worse and not n2_better:
        return "RAW_SUFFICIENT", (n1_worse, n2_worse, n1_better, n2_better)
    return "NO_CLEAR_ENCODING_EFFECT", (n1_worse, n2_worse, n1_better, n2_better)


def main():
    days = [(w, str((pd.Timestamp(a) + pd.Timedelta(days=i)).date())) for w, (a, b) in W.items() for i in range(7)]
    allq = pd.concat([frame(a, d) for a in ARMS for _, d in days], ignore_index=True)
    daily = []
    for a in ARMS:
        for w, d in days:
            daily.append({**metric(allq[(allq.arm == a) & (allq.target_day == d)], w), "target_day": d})
    day = pd.DataFrame(daily)
    day.to_csv(HERE / "daily_metrics.csv", index=False)
    over = pd.DataFrame([metric(allq[allq.arm == a], "overall") for a in ARMS])
    win = pd.DataFrame([metric(allq[(allq.arm == a) & allq.target_day.isin(day.loc[day.scope == w, "target_day"])], w)
                        for a in ARMS for w in W])
    win.to_csv(HERE / "window_metrics.csv", index=False)
    hour = pd.DataFrame([metric(allq[(allq.arm == a) & allq.hour_business.between(lo, hi)], h)
                         for a in ARMS for h, lo, hi in (("H1", 1, 8), ("H2", 9, 16), ("H3", 17, 24))])
    hour.to_csv(HERE / "hour_segment_metrics.csv", index=False)
    summary = over.copy()
    summary["min_window_raw"] = [win[win.arm == a].raw.min() for a in summary.arm]
    summary["window_raw_std"] = [win[win.arm == a].raw.std(ddof=0) for a in summary.arm]
    summary.to_csv(HERE / "model_summary.csv", index=False)

    rng = np.random.default_rng(20260924)
    paired = bootstrap(day, rng)
    pd.DataFrame(paired).to_csv(HERE / "paired_summary.csv", index=False)

    c = summary.set_index("arm")
    safety = {}
    for a in ARMS:
        r = c.loc[a]
        z = day[day.arm == a]
        checks = {"balanced": bool(r.balanced >= c.loc["N0"].balanced - .01),
                  "positive_recall": bool(r.positive_recall >= c.loc["N0"].positive_recall - .05),
                  "one_class_days": int(z.one_class_days.sum()) <= int(day[day.arm == "N0"].one_class_days.sum()) + 2,
                  "positive_recall_zero_days": int(z.positive_recall_zero_days.sum()) <= int(day[day.arm == "N0"].positive_recall_zero_days.sum()) + 2}
        safety[a] = {"admissible": all(checks.values()), "checks": checks}
    (HERE / "safety_admissibility.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")

    label, flags = decide(paired, safety)
    (HERE / "E2_D1_GATE.md").write_text(
        f"# E2-D1 Gate\n\n**E2_D1_GATE ENCODING_SIGNAL = `{label}`**\n\n"
        "Uses only frozen W1-W4 panel, paired day-cluster bootstrap (10,000 draws, seed 20260924), "
        "cross-window evidence and N0 safety. Benchmark excluded. Stop for human review.\n"
        f"\nFlags (n1_stably_worse, n2_stably_worse, n1_stably_better, n2_stably_better): {flags}\n",
        encoding="utf-8")

    # encoding audit + runtime from the real run manifests
    ga_file = HERE / "benchmark" / "gate_a_results.json"
    n0_backfill = {}
    if ga_file.exists():
        n0_backfill = json.loads(ga_file.read_text(encoding="utf-8")).get("arms", {}).get("N0", {}).get("numeric_encoding_audit") or {}
    audit_rows, runtime_rows = [], []
    for a in ARMS:
        for benchmark in (False, True):
            dayset = ["2026-02-13"] if benchmark else [d for _, d in days]
            for d in dayset:
                r = rec(a, d, benchmark)
                m = json.loads((Path(r["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
                au = m.get("numeric_encoding_audit") or (n0_backfill if a == "N0" else {})
                audit_rows.append({"arm": a, "scope": "benchmark" if benchmark else "formal", "target_day": d,
                                   "numeric_encoding_mode": m.get("numeric_encoding_mode", "canonical"),
                                   "encoded_feature_dim": au.get("encoded_feature_dim"),
                                   "strong_feature_count": au.get("strong_feature_count"),
                                   "weak_feature_count": au.get("weak_feature_count"),
                                   "strong_tabm_input_dim": au.get("strong_tabm_input_dim"),
                                   "weak_mlp_input_dim": au.get("weak_mlp_input_dim"),
                                   "parameter_count": m.get("parameter_count")})
                runtime_rows.append({"arm": a, "scope": "benchmark" if benchmark else "formal", "target_day": d,
                                     "external_wall_seconds": r.get("wall_seconds"),
                                     "train_wall_seconds": m.get("wall_time_total_seconds"),
                                     "epoch_seconds_mean": m.get("training_epoch_seconds_mean"),
                                     "best_epoch": m.get("best_epoch"), "stop_epoch": m.get("stop_epoch"),
                                     "epochs_run": m.get("epochs_run"),
                                     "cuda_peak_memory_bytes": m.get("cuda_peak_memory_bytes"),
                                     "parameter_count": m.get("parameter_count"),
                                     "device": m.get("device"), "amp": m.get("amp")})
    pd.DataFrame(audit_rows).to_csv(HERE / "encoding_audit.csv", index=False)
    (HERE / "encoding_audit.json").write_text(json.dumps(
        {"by_arm": {a: next(row for row in audit_rows if row["arm"] == a and row["scope"] == "formal")
                    for a in ARMS},
         "benchmark": [row for row in audit_rows if row["scope"] == "benchmark"],
         "rows": len(audit_rows)}, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(runtime_rows).to_csv(HERE / "runtime.csv", index=False)

    # benchmark engineering summary (excluded from gate ranking)
    bench = pd.concat([frame(a, "2026-02-13", benchmark=True) for a in ARMS], ignore_index=True)
    pd.DataFrame([metric(bench[bench.arm == a], "benchmark_engineering_only") for a in ARMS]).to_csv(
        HERE / "benchmark" / "benchmark_metrics.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = summary.set_index("arm")[["raw", "balanced", "auc"]].plot.bar()
    ax.figure.tight_layout(); ax.figure.savefig(HERE / "figures/overall_metrics.png", dpi=150); plt.close(ax.figure)
    ax = win.pivot(index="scope", columns="arm", values="raw").reindex(["W1", "W2", "W3", "W4"]).plot.bar(ylim=(.3, .8))
    ax.figure.tight_layout(); ax.figure.savefig(HERE / "figures/window_raw.png", dpi=150); plt.close(ax.figure)
    pp = pd.DataFrame(paired).iloc[:2]
    fig, ax = plt.subplots(figsize=(6, 3)); yy = np.arange(len(pp))
    ax.errorbar(pp.mean_delta_raw, yy, xerr=[pp.mean_delta_raw - pp.raw_ci_low, pp.raw_ci_high - pp.mean_delta_raw],
                fmt="o", capsize=4); ax.axvline(0, color="black", lw=1)
    ax.axvline(.02, color="gray", ls="--"); ax.axvline(-.02, color="gray", ls="--")
    ax.set_yticks(yy, pp.comparison); ax.set_xlabel("Day-paired Delta Raw with 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(HERE / "figures/paired_raw_ci.png", dpi=150); plt.close(fig)
    print(json.dumps({"signal": label, "flags": flags, "overall": summary.to_dict("records"),
                      "paired": paired, "safety": safety}, indent=2, default=float))


if __name__ == "__main__":
    main()

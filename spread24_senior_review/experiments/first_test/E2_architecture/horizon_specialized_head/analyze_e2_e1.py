"""Analyze the frozen E2-E1 panel; target-day bootstrap is the paired unit.

Q0 reuses the frozen F80 control; Q1/Q2 are fresh E2-E1 runs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
F80 = HERE.parent / "fusion_coarse" / "runs"
W = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
     "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
ARMS = ("Q0", "Q1", "Q2")
EXPECTED_READOUT = {"Q0": {"shared", None}, "Q1": {"segment_bias"}, "Q2": {"segment_heads"}}
SEGMENTS = (("H1", 1, 8), ("H2", 9, 16), ("H3", 17, 24))
FROZEN = {"config": "8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c",
          "selector": "ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f",
          "source": "a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea",
          "sequence": "9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82"}


def rec(a, d, benchmark=False):
    if a == "Q0":
        return json.loads((F80 / f"E2B1-F80-{d}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
    base = HERE / "runs" / ("benchmark" if benchmark else "")
    return json.loads((base / f"E2E1-{a}-{d}" / "RUN_RECORD.json").read_text(encoding="utf-8"))


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
              "encoding": m.get("numeric_encoding_mode", "canonical") == "canonical",
              "readout": m.get("direction_readout_mode") in EXPECTED_READOUT[a],
              "alpha": abs(float(m.get("direction_fusion_alpha", -1)) - .8) < 1e-12,
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
    for a, b in (("Q1", "Q0"), ("Q2", "Q0"), ("Q2", "Q1")):
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


def decide(paired, safety, summary, slot_overall):
    p = {x["comparison"]: x for x in paired}

    def stable(a, b, sign):
        x = p[f"{a}-{b}"]
        s = np.sign(json.loads(x["window_delta_raw_signs"]))
        return (x["raw_ci_excludes_zero"] and abs(x["mean_delta_raw"]) >= .02 and int((s == sign).sum()) >= 3)
    idx = summary.set_index("arm")
    gain = {a: float(idx.loc[a].raw - idx.loc["Q0"].raw) for a in ("Q1", "Q2")}
    q1_safe, q2_safe = bool(safety["Q1"]["admissible"]), bool(safety["Q2"]["admissible"])
    q1_better, q2_better = stable("Q1", "Q0", 1), stable("Q2", "Q0", 1)
    q1_worse, q2_worse = stable("Q1", "Q0", -1), stable("Q2", "Q0", -1)
    q2_over_q1 = stable("Q2", "Q1", 1)
    # "gains >= ~2pp pooled Raw / meaningful slot count and safety, but paired/cross-window evidence
    # is not yet established" (plan 22 section 11). Meaningful slot count = >= 10 extra correct slots.
    q1_mean = gain["Q1"] >= .02 or slot_overall["Q1"] >= 10
    q2_mean = gain["Q2"] >= .02 or slot_overall["Q2"] >= 10
    flags = {"gain_Q1": gain["Q1"], "gain_Q2": gain["Q2"], "slots_Q1": slot_overall["Q1"], "slots_Q2": slot_overall["Q2"],
             "q1_safe": q1_safe, "q2_safe": q2_safe, "q1_better": q1_better, "q2_better": q2_better,
             "q1_worse": q1_worse, "q2_worse": q2_worse, "q2_over_q1": q2_over_q1,
             "meaningful_Q1": q1_mean, "meaningful_Q2": q2_mean}
    if q1_safe and q1_better and not q2_over_q1:
        return "SEGMENT_CALIBRATION_HELPFUL", flags
    if q2_safe and q2_better and q2_over_q1:
        return "SEGMENT_BOUNDARY_HELPFUL", flags
    if (q1_safe and q1_mean) or (q2_safe and q2_mean):
        return "HETEROGENEITY_PROMISING", flags
    if (not q1_safe) or (not q2_safe) or q1_worse or q2_worse:
        return "SPECIALIZATION_HARMFUL", flags
    if not q1_better and not q2_better:
        return "SHARED_HEAD_SUFFICIENT", flags
    return "NO_CLEAR_READOUT_EFFECT", flags


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
                         for a in ARMS for h, lo, hi in SEGMENTS])
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
        checks = {"balanced": bool(r.balanced >= c.loc["Q0"].balanced - .01),
                  "positive_recall": bool(r.positive_recall >= c.loc["Q0"].positive_recall - .05),
                  "one_class_days": int(z.one_class_days.sum()) <= int(day[day.arm == "Q0"].one_class_days.sum()) + 2,
                  "positive_recall_zero_days": int(z.positive_recall_zero_days.sum()) <= int(day[day.arm == "Q0"].positive_recall_zero_days.sum()) + 2}
        safety[a] = {"admissible": all(checks.values()), "checks": checks}
    (HERE / "safety_admissibility.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")

    # slot gains vs Q0 (overall and by segment)
    gain_rows = []
    for a in ("Q1", "Q2"):
        for name, lo, hi in [("overall", 1, 24), *SEGMENTS]:
            qa = allq[(allq.arm == a) & allq.hour_business.between(lo, hi)]
            q0 = allq[(allq.arm == "Q0") & allq.hour_business.between(lo, hi)]
            ca = int((qa.direction_hat.to_numpy().astype(bool) == (qa.y_true_model.to_numpy() > 0)).sum())
            c0 = int((q0.direction_hat.to_numpy().astype(bool) == (q0.y_true_model.to_numpy() > 0)).sum())
            gain_rows.append({"arm": a, "scope": name, "correct_slots_arm": ca, "correct_slots_Q0": c0,
                              "delta_correct_slots": ca - c0, "n_slots": len(qa),
                              "delta_slots_per_day": (ca - c0) / 28})
    pd.DataFrame(gain_rows).to_csv(HERE / "slot_gain_summary.csv", index=False)

    slot_overall = {row["arm"]: row["delta_correct_slots"] for row in gain_rows if row["scope"] == "overall"}
    label, flags = decide(paired, safety, summary, slot_overall)
    (HERE / "E2_E1_GATE.md").write_text(
        f"# E2-E1 Gate\n\n**E2_E1_GATE READOUT_SIGNAL = `{label}`**\n\n"
        "Uses only frozen W1-W4 panel, paired day-cluster bootstrap (10,000 draws, seed 20260924), "
        "cross-window + segment evidence and Q0 safety. Benchmark excluded. Stop for human review.\n"
        f"\nFlags: {json.dumps(flags)}\n", encoding="utf-8")

    # readout diagnostics table (probe output) + runtime/params
    runtime_rows = []
    for a in ARMS:
        for benchmark in (False, True):
            dayset = ["2026-02-13"] if benchmark else [d for _, d in days]
            for d in dayset:
                r = rec(a, d, benchmark)
                m = json.loads((Path(r["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
                runtime_rows.append({"arm": a, "scope": "benchmark" if benchmark else "formal", "target_day": d,
                                     "external_wall_seconds": r.get("wall_seconds"),
                                     "train_wall_seconds": m.get("wall_time_total_seconds"),
                                     "epoch_seconds_mean": m.get("training_epoch_seconds_mean"),
                                     "best_epoch": m.get("best_epoch"), "stop_epoch": m.get("stop_epoch"),
                                     "epochs_run": m.get("epochs_run"),
                                     "cuda_peak_memory_bytes": m.get("cuda_peak_memory_bytes"),
                                     "parameter_count": m.get("parameter_count"),
                                     "parameter_count_trainable": m.get("parameter_count_trainable"),
                                     "device": m.get("device"), "amp": m.get("amp")})
    pd.DataFrame(runtime_rows).to_csv(HERE / "runtime.csv", index=False)

    bench = pd.concat([frame(a, "2026-02-13", benchmark=True) for a in ARMS], ignore_index=True)
    pd.DataFrame([metric(bench[bench.arm == a], "benchmark_engineering_only") for a in ARMS]).to_csv(
        HERE / "benchmark" / "benchmark_metrics.csv", index=False)

    # Q1 bias trajectory (first/last epoch per run) from training_history.parquet
    bias_traj = []
    for d in [x for _, x in days]:
        folder = Path(rec("Q1", d)["run_dir"])
        hist = pd.read_parquet(folder / "training_history.parquet")
        cols = [c for c in ("direction_segment_bias_H1", "direction_segment_bias_H2", "direction_segment_bias_H3") if c in hist.columns]
        if not cols:
            continue
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        sel = int(man["stage_a_best_epoch"])
        bias_traj.append({"target_day": d, "epochs": int(len(hist)),
                          "init": [float(hist.iloc[0][c]) for c in cols],
                          "final": [float(hist.iloc[-1][c]) for c in cols],
                          "selected_epoch": sel,
                          "at_selected_epoch": [float(hist.iloc[max(0, sel - 1)][c]) for c in cols]})
    # Q2 per-head param/grad norm aggregates from the read-only probe
    probe = {"source": "readout_diagnostics.csv", "present": False}
    probe_file = HERE / "readout_diagnostics.csv"
    if probe_file.exists():
        pr = pd.read_csv(probe_file)
        probe = {"source": "readout_diagnostics.csv", "present": True, "rows": len(pr)}
        for tag, cols in (("q2_head_param_norm_mean", ["head_param_norm_H1", "head_param_norm_H2", "head_param_norm_H3"]),
                          ("q2_head_grad_norm_mean", ["head_grad_norm_H1", "head_grad_norm_H2", "head_grad_norm_H3"]),
                          ("q1_bias_grad_norm_mean", ["bias_grad_norm_H1", "bias_grad_norm_H2", "bias_grad_norm_H3"]),
                          ("q1_bias_mean", ["segment_bias_H1", "segment_bias_H2", "segment_bias_H3"])):
            probe[tag] = {c: float(pr[c].dropna().mean()) for c in cols if c in pr.columns and pr[c].notna().any()}
    # Q0 formal runs reuse F80 (no readout audit); use the fresh Q0 benchmark manifest for its audit.
    def audit_for(a):
        if a == "Q0":
            r0 = json.loads((HERE / "runs" / "benchmark" / "E2E1-Q0-2026-02-13" / "RUN_RECORD.json").read_text(encoding="utf-8"))
            path = Path(r0["run_dir"]) / "manifest.json"
        else:
            path = Path(rec(a, days[0][1])["run_dir"]) / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8")).get("direction_readout_audit")
    (HERE / "readout_diagnostics.json").write_text(json.dumps(
        {"q1_bias_trajectory": bias_traj,
         "probe": probe,
         "model_audit": {a: audit_for(a) for a in ("Q0", "Q1", "Q2")}},
        indent=2, default=float), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = summary.set_index("arm")[["raw", "balanced", "auc"]].plot.bar()
    ax.figure.tight_layout(); ax.figure.savefig(HERE / "figures/overall_metrics.png", dpi=150); plt.close(ax.figure)
    ax = win.pivot(index="scope", columns="arm", values="raw").reindex(["W1", "W2", "W3", "W4"]).plot.bar(ylim=(.3, .85))
    ax.figure.tight_layout(); ax.figure.savefig(HERE / "figures/window_raw.png", dpi=150); plt.close(ax.figure)
    ax = hour.pivot(index="scope", columns="arm", values="raw").reindex(["H1", "H2", "H3"]).plot.bar(ylim=(.3, .8))
    ax.figure.tight_layout(); ax.figure.savefig(HERE / "figures/segment_raw.png", dpi=150); plt.close(ax.figure)
    pp = pd.DataFrame(paired).iloc[:2]
    fig, ax = plt.subplots(figsize=(6, 3)); yy = np.arange(len(pp))
    ax.errorbar(pp.mean_delta_raw, yy, xerr=[pp.mean_delta_raw - pp.raw_ci_low, pp.raw_ci_high - pp.mean_delta_raw],
                fmt="o", capsize=4); ax.axvline(0, color="black", lw=1)
    ax.axvline(.02, color="gray", ls="--"); ax.axvline(-.02, color="gray", ls="--")
    ax.set_yticks(yy, pp.comparison); ax.set_xlabel("Day-paired Delta Raw with 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(HERE / "figures/paired_raw_ci.png", dpi=150); plt.close(fig)
    print(json.dumps({"signal": label, "flags": flags, "overall": summary.to_dict("records"),
                      "paired": paired, "safety": safety, "slot_gain": gain_rows}, indent=2, default=float))


if __name__ == "__main__":
    main()

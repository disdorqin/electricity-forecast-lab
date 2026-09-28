"""Analyze the frozen E3-A panel; target-day bootstrap is the paired unit.

T0 reuses E2-E1 Q2 (canonical 80/20); T1/T2/T3 are fresh E3-A runs.
"""
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
E2E1 = HERE.parents[1] / "E2_architecture" / "horizon_specialized_head"
W = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
     "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
ARMS = ("T0", "T1", "T2", "T3")
FRESH = ("T1", "T2", "T3")
EXPECT_SPLIT = {"T0": (None, None), "T1": (60, None), "T2": (60, 365), "T3": (60, 1095)}
SEGMENTS = (("H1", 1, 8), ("H2", 9, 16), ("H3", 17, 24))
FROZEN = {"config": "8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c",
          "selector": "ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f",
          "source": "a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea",
          "sequence": "9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82"}


def rec(a, d, benchmark=False):
    if a == "T0":
        base = E2E1 / "runs" / ("benchmark" if benchmark else "")
        return json.loads((base / f"E2E1-Q2-{d}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
    base = HERE / "runs" / ("benchmark" if benchmark else "")
    return json.loads((base / f"E3A-{a}-{d}" / "RUN_RECORD.json").read_text(encoding="utf-8"))


def manifest_of(a, d, benchmark=False):
    return json.loads((Path(rec(a, d, benchmark)["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))


def frame(a, d, benchmark=False):
    r = rec(a, d, benchmark)
    folder = Path(r["run_dir"])
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    q = pd.read_parquet(folder / "predictions.parquet")
    mon, win = EXPECT_SPLIT[a]
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
              "readout": m.get("direction_readout_mode") == "segment_heads",
              "alpha": abs(float(m.get("direction_fusion_alpha", -1)) - .8) < 1e-12,
              "config": m.get("config_sha256") == FROZEN["config"],
              "selector": m.get("selector_sha256") == FROZEN["selector"],
              "source": m.get("source_sha256") == FROZEN["source"],
              "sequence": m.get("sequence_manifest_sha256") == FROZEN["sequence"]}
    if a == "T0":
        sp = m.get("stage_a_split")
        if sp:
            checks["split"] = (sp.get("split_mode") == "canonical_percent80")
        else:
            # T0 reuses E2-E1 Q2 runs produced before the stage_a_split field existed; verify the
            # canonical percentage split from the counts those manifests do record.
            import math
            b = int(m.get("stage_a_base_train_days", -1)); mo = int(m.get("stage_a_monitor_days", -1))
            checks["split"] = (b >= 2 and mo >= 1 and
                               b == max(2, min(b + mo - 1, int(math.floor(0.8 * (b + mo))))))
    else:
        sp = m.get("stage_a_split", {})
        checks["split"] = (sp.get("stage_a_monitor", {}).get("count") == mon and
                           sp.get("stage_a_monitor_days_requested") == mon and
                           sp.get("stage_a_history_window_days_requested") == win)
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


def bootstrap(day, rng, pairs):
    out = []
    order = list(day[day.arm == "T0"].sort_values("target_day").target_day)
    for a, b in pairs:
        x = day[day.arm == a].set_index("target_day").sort_index()
        y = day[day.arm == b].set_index("target_day").sort_index()
        dr = x.raw.to_numpy() - y.raw.to_numpy()
        db = x.balanced.to_numpy() - y.balanced.to_numpy()
        da = x.auc.to_numpy() - y.auc.to_numpy()
        ix = rng.integers(0, len(dr), (10000, len(dr)))
        rc = np.quantile(dr[ix].mean(1), [.025, .975]); bc = np.quantile(db[ix].mean(1), [.025, .975])
        ac = np.quantile(da[ix].mean(1), [.025, .975])
        signs = [float(np.sign(dr[np.array([z == w for z in x.scope])].mean())) for w in W]
        out.append({"comparison": f"{a}-{b}", "mean_delta_raw": dr.mean(), "raw_ci_low": rc[0], "raw_ci_high": rc[1],
                    "raw_ci_excludes_zero": bool(rc[0] > 0 or rc[1] < 0),
                    "raw_W_T_L": f"{(dr>0).sum()}/{(dr==0).sum()}/{(dr<0).sum()}",
                    "mean_delta_balanced": db.mean(), "balanced_ci_low": bc[0], "balanced_ci_high": bc[1],
                    "mean_delta_auc": da.mean(), "auc_ci_low": ac[0], "auc_ci_high": ac[1],
                    "window_delta_raw_signs": json.dumps(signs)})
    return out


def decide(paired, safety, summary, slot_overall):
    p = {x["comparison"]: x for x in paired}

    def stable(a, b):
        x = p[f"{a}-{b}"]
        s = np.sign(json.loads(x["window_delta_raw_signs"]))
        return x["raw_ci_excludes_zero"] and x["mean_delta_raw"] >= .02 and int((s == 1).sum()) >= 3
    idx = summary.set_index("arm")
    gain = {a: float(idx.loc[a].raw - idx.loc["T0"].raw) for a in FRESH}
    safe = {a: bool(safety[a]["admissible"]) for a in FRESH}
    stb = {a: bool(stable(a, "T0")) for a in FRESH}
    apparent = {a: bool((gain[a] >= .02 or abs(slot_overall[a]) >= 10) and safe[a]) for a in FRESH}
    flags = {"gain": gain, "slots": slot_overall, "safe": safe, "stably_better_than_T0": stb, "apparent": apparent}
    if stb["T1"] and stb["T2"] and stb["T3"] and all(safe.values()):
        return "STALE_SPLIT_CONFIRMED", flags
    if stb["T3"] and safe["T3"]:
        return "THREE_YEAR_BEST", flags
    if stb["T2"] and safe["T2"]:
        return "RECENT_YEAR_BEST", flags
    if stb["T1"] and safe["T1"]:
        return "EXPANDING_RECENT_BEST", flags
    if any(apparent.values()):
        return "PROMISING_RECENCY_UNPROVEN", flags
    # no stable/apparent win: does longer history help relative to the 1-year window?
    if (gain["T3"] - gain["T2"] >= .02) or (gain["T1"] - gain["T2"] >= .02):
        return "LONG_HISTORY_NEEDED", flags
    return "NO_RECENCY_GAIN", flags


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
    pairs = [("T1", "T0"), ("T2", "T0"), ("T3", "T0"), ("T2", "T1"), ("T3", "T1"), ("T3", "T2")]
    paired = bootstrap(day, rng, pairs)
    pd.DataFrame(paired).to_csv(HERE / "paired_summary.csv", index=False)

    c = summary.set_index("arm")
    safety = {}
    for a in ARMS:
        r = c.loc[a]; z = day[day.arm == a]
        checks = {"balanced": bool(r.balanced >= c.loc["T0"].balanced - .01),
                  "positive_recall": bool(r.positive_recall >= c.loc["T0"].positive_recall - .05),
                  "one_class_days": int(z.one_class_days.sum()) <= int(day[day.arm == "T0"].one_class_days.sum()) + 2,
                  "positive_recall_zero_days": int(z.positive_recall_zero_days.sum()) <= int(day[day.arm == "T0"].positive_recall_zero_days.sum()) + 2}
        safety[a] = {"admissible": all(checks.values()), "checks": checks}
    (HERE / "safety_admissibility.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")

    gain_rows = []
    for a in FRESH:
        for name, lo, hi in [("overall", 1, 24), *SEGMENTS]:
            qa = allq[(allq.arm == a) & allq.hour_business.between(lo, hi)]
            q0 = allq[(allq.arm == "T0") & allq.hour_business.between(lo, hi)]
            ca = int((qa.direction_hat.to_numpy().astype(bool) == (qa.y_true_model.to_numpy() > 0)).sum())
            c0 = int((q0.direction_hat.to_numpy().astype(bool) == (q0.y_true_model.to_numpy() > 0)).sum())
            gain_rows.append({"arm": a, "scope": name, "correct_slots_arm": ca, "correct_slots_T0": c0,
                              "delta_correct_slots": ca - c0, "n_slots": len(qa), "delta_slots_per_day": (ca - c0) / 28})
    pd.DataFrame(gain_rows).to_csv(HERE / "slot_gain_summary.csv", index=False)
    slot_overall = {r["arm"]: r["delta_correct_slots"] for r in gain_rows if r["scope"] == "overall"}

    # exact split date ranges per run
    split_rows, epoch_rows, runtime_rows = [], [], []
    for a in ARMS:
        for benchmark in (False, True):
            dayset = ["2026-02-13"] if benchmark else [d for _, d in days]
            for d in dayset:
                rec_ = rec(a, d, benchmark)
                man = json.loads((Path(rec_["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
                sp = man.get("stage_a_split")
                if not sp and a == "T0":
                    sp = {"split_mode": "canonical_percent80",
                          "stage_a_eligible": {"count": (man.get("stage_a_base_train_days", 0) or 0) + (man.get("stage_a_monitor_days", 0) or 0),
                                               "start": None, "end": None},
                          "stage_a_base_train": {"count": man.get("stage_a_base_train_days"),
                                                 "start": man.get("preprocessing_fit_day_start"),
                                                 "end": man.get("preprocessing_fit_day_end")},
                          "stage_a_monitor": {"count": man.get("stage_a_monitor_days"), "start": None, "end": None}}
                sp = sp or {}
                split_rows.append({"arm": a, "scope": "benchmark" if benchmark else "formal", "target_day": d,
                                   "split_mode": sp.get("split_mode"),
                                   "eligible_count": sp.get("stage_a_eligible", {}).get("count"),
                                   "eligible_start": sp.get("stage_a_eligible", {}).get("start"),
                                   "eligible_end": sp.get("stage_a_eligible", {}).get("end"),
                                   "base_count": sp.get("stage_a_base_train", {}).get("count"),
                                   "base_start": sp.get("stage_a_base_train", {}).get("start"),
                                   "base_end": sp.get("stage_a_base_train", {}).get("end"),
                                   "monitor_count": sp.get("stage_a_monitor", {}).get("count"),
                                   "monitor_start": sp.get("stage_a_monitor", {}).get("start"),
                                   "monitor_end": sp.get("stage_a_monitor", {}).get("end"),
                                   "preprocessing_fit_start": man.get("preprocessing_fit_day_start"),
                                   "preprocessing_fit_end": man.get("preprocessing_fit_day_end")})
                runtime_rows.append({"arm": a, "scope": "benchmark" if benchmark else "formal", "target_day": d,
                                     "external_wall_seconds": rec_.get("wall_seconds"),
                                     "train_wall_seconds": man.get("wall_time_total_seconds"),
                                     "epoch_seconds_mean": man.get("training_epoch_seconds_mean"),
                                     "best_epoch": man.get("best_epoch"), "stop_epoch": man.get("stop_epoch"),
                                     "epochs_run": man.get("epochs_run"),
                                     "cuda_peak_memory_bytes": man.get("cuda_peak_memory_bytes"),
                                     "parameter_count": man.get("parameter_count"),
                                     "device": man.get("device"), "amp": man.get("amp")})
                if not benchmark:
                    hist = pd.read_parquet(Path(rec_["run_dir"]) / "training_history.parquet")
                    sel = int(man["stage_a_best_epoch"])
                    row = hist.iloc[max(0, sel - 1)] if len(hist) else None
                    def at(k):
                        return (float(row[k]) if row is not None and k in hist.columns and pd.notna(row[k]) else np.nan)
                    epoch_rows.append({"arm": a, "target_day": d, "best_epoch": sel,
                                       "stop_epoch": man.get("stop_epoch"), "epochs_run": man.get("epochs_run"),
                                       "train_L_dir_selected": at("train_L_dir"), "monitor_L_dir_selected": at("monitor_L_dir"),
                                       "bce_gap_selected": (at("monitor_L_dir") - at("train_L_dir")),
                                       "monitor_raw_selected": at("monitor_raw_direction_accuracy"),
                                       "monitor_auc_selected": at("monitor_auc"),
                                       "monitor_brier_selected": at("monitor_brier"),
                                       "monitor_raw_min": float(hist["monitor_raw_direction_accuracy"].min()) if "monitor_raw_direction_accuracy" in hist.columns else np.nan,
                                       "monitor_raw_final": float(hist["monitor_raw_direction_accuracy"].iloc[-1]) if "monitor_raw_direction_accuracy" in hist.columns else np.nan})
    pd.DataFrame(split_rows).to_csv(HERE / "split_summary.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(HERE / "epoch_diagnostics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(HERE / "runtime.csv", index=False)

    label, flags = decide(paired, safety, summary, slot_overall)
    (HERE / "E3_A_GATE.md").write_text(
        f"# E3-A Gate\n\n**E3_A_GATE RECENCY_SIGNAL = `{label}`**\n\n"
        "Uses only frozen W1-W4 panel, paired day-cluster bootstrap (10,000 draws, seed 20260924), "
        "cross-window + segment evidence and T0 safety. Benchmark excluded. Stop for human review.\n"
        f"\nFlags: {json.dumps(flags)}\n", encoding="utf-8")

    bench = pd.concat([frame(a, "2026-02-13", benchmark=True) for a in ARMS], ignore_index=True)
    pd.DataFrame([metric(bench[bench.arm == a], "benchmark_engineering_only") for a in ARMS]).to_csv(
        HERE / "benchmark" / "benchmark_metrics.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = summary.set_index("arm")[["raw", "balanced", "auc"]].plot.bar()
    ax.figure.tight_layout(); ax.figure.savefig(HERE / "figures/overall_metrics.png", dpi=150); plt.close(ax.figure)
    ax = win.pivot(index="scope", columns="arm", values="raw").reindex(["W1", "W2", "W3", "W4"]).plot.bar(ylim=(.3, .85))
    ax.figure.tight_layout(); ax.figure.savefig(HERE / "figures/window_raw.png", dpi=150); plt.close(ax.figure)
    ax = hour.pivot(index="scope", columns="arm", values="raw").reindex(["H1", "H2", "H3"]).plot.bar(ylim=(.3, .8))
    ax.figure.tight_layout(); ax.figure.savefig(HERE / "figures/segment_raw.png", dpi=150); plt.close(ax.figure)
    pp = pd.DataFrame(paired).iloc[:3]
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

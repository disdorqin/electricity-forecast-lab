"""E4-A Gate A: Q2 exactness, three-way split routing, checkpoint identity, legality, frozen assets."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_e4_a  # noqa: E402  (local helper: experiment_python / FROZEN / scope)
from src.TafM_改进源码.config import default_selector_path, load_v21_config  # noqa: E402
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector  # noqa: E402
from src.TafM_改进源码.direction_postprocess import REGIME_FEATURES  # noqa: E402
from src.TafM_改进源码.train import E4_CALIBRATOR_DAYS, e4a_three_way_split, stage_a_split_indices  # noqa: E402

E2E1 = REPO / "experiments/first_test/E2_architecture/horizon_specialized_head/runs"
DAY = "2026-02-13"
FROZEN_HASHES = {
    "config": "8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c",
    "selector": "ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f",
    "source": "a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea",
    "sequence": "9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82",
}


def run_cli(args, cwd=None):
    py = run_e4_a.experiment_python()
    proc = subprocess.run([py, str(REPO / "src" / "run_tabm_v21.py"), *args], cwd=cwd or (REPO / "src"),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc


def preds(run_dir):
    return pd.read_parquet(Path(run_dir) / "predictions.parquet").sort_values("hour_business").reset_index(drop=True)


def main():
    result = {"benchmark_day": DAY, "checks": {}}
    store = SequenceStore.load()
    selector, _ = load_frozen_selector(default_selector_path())
    cfg = load_v21_config()
    day_idx = {str(d): i for i, d in enumerate(store.days)}

    # 1. default Q2 (no E4 flags) reproduces the saved Q2 benchmark
    repro = (HERE / "benchmark" / "repro_default_q2")
    repro.mkdir(parents=True, exist_ok=True)
    proc = run_cli(["direction-experiment", "--target-day", DAY, "--profile", "default", "--mode", "A2",
                    "--objective-mode", "dir_only", "--checkpoint-policy", "direction_first",
                    "--gradient-policy", "vanilla", "--architecture-mode", "full_current",
                    "--direction-fusion-alpha", "0.8", "--strong-role-profile", "all",
                    "--numeric-encoding-mode", "canonical", "--direction-readout-mode", "segment_heads"])
    (repro / "COMMAND_STDOUT.txt").write_text(proc.stdout or "", encoding="utf-8")
    (repro / "COMMAND_STDERR.txt").write_text(proc.stderr or "", encoding="utf-8")
    repro_delta = None
    if proc.returncode == 0:
        rd = Path(json.loads(proc.stdout)["run_dir"])
        q2 = json.loads((E2E1 / "benchmark" / f"E2E1-Q2-{DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
        a, b = preds(rd), preds(q2["run_dir"])
        cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
        repro_delta = float(np.max([np.max(np.abs(a[c].to_numpy() - b[c].to_numpy())) for c in cols]))
    result["default_q2_reproduction_max_abs_delta"] = repro_delta
    result["checks"]["1_default_q2_reproduces_saved_q2"] = (repro_delta is not None and repro_delta <= 1e-9)

    # 9/10. P1/P2/P3 same checkpoint; P1 base == P2/P3 stored base
    ck = pd.read_csv(HERE / "checkpoint_identity.csv")
    result["checks"]["9_same_checkpoint_P1_P2_P3"] = bool(
        (ck.P1_checkpoint_sha256 == ck.P2_checkpoint_sha256).all() and
        (ck.P1_checkpoint_sha256 == ck.P3_checkpoint_sha256).all())
    pred = pd.read_parquet(HERE / "derived_predictions.parquet")
    # 10. the in-train postprocess path must reproduce the offline P2 derivation exactly on 2026-02-13
    itrain = HERE / "benchmark" / "intrain_segment_logit"
    itrain.mkdir(parents=True, exist_ok=True)
    proc = run_cli(["direction-experiment", "--target-day", DAY, "--profile", "default", "--mode", "A2",
                    "--objective-mode", "dir_only", "--checkpoint-policy", "direction_first",
                    "--gradient-policy", "vanilla", "--architecture-mode", "full_current",
                    "--direction-fusion-alpha", "0.8", "--strong-role-profile", "all",
                    "--numeric-encoding-mode", "canonical", "--direction-readout-mode", "segment_heads",
                    "--e4-three-way-split", "--direction-postprocess-mode", "segment_logit"])
    (itrain / "COMMAND_STDOUT.txt").write_text(proc.stdout or "", encoding="utf-8")
    (itrain / "COMMAND_STDERR.txt").write_text(proc.stderr or "", encoding="utf-8")
    item10 = False
    if proc.returncode == 0:
        rd = Path(json.loads(proc.stdout)["run_dir"])
        audit = json.loads((rd / "direction_postprocessor.json").read_text(encoding="utf-8"))
        coef_all = pd.read_csv(HERE / "stacker_coefficients.csv")
        row = coef_all[(coef_all.arm == "P2") & (coef_all.target_day == DAY)]
        w_ok = bool(len(row) == 1 and all(
            abs(float(audit["weight"][i]) - float(row.iloc[0][f"w_{n}"])) < 1e-12
            for i, n in enumerate(audit["feature_names"])))
        in_pred = preds(rd)
        der = pred[(pred.arm == "P2") & (pred.target_day == DAY)].sort_values("hour_business").reset_index(drop=True)
        der_base = pred[(pred.arm == "P1") & (pred.target_day == DAY)].sort_values("hour_business").reset_index(drop=True)
        p_ok = bool(np.max(np.abs(in_pred.p_positive.to_numpy() - der.p_positive.to_numpy())) <= 1e-9)
        b_ok = bool(np.max(np.abs(in_pred.p_positive_base.to_numpy() - der_base.p_positive.to_numpy())) <= 1e-9)
        item10 = bool(w_ok and p_ok and b_ok)
        result["intrain_cross_check"] = {"weights_match": w_ok, "post_predictions_match": p_ok,
                                         "base_predictions_match": b_ok}
    result["checks"]["10_intrain_postprocess_matches_offline_derivation"] = item10

    # benchmark (2026-02-13) engineering-only table for P0/P1/P2/P3 (never used for ranking)
    reg_run = HERE / "benchmark" / "intrain_regime_logit"
    reg_run.mkdir(parents=True, exist_ok=True)
    proc3 = run_cli(["direction-experiment", "--target-day", DAY, "--profile", "default", "--mode", "A2",
                     "--objective-mode", "dir_only", "--checkpoint-policy", "direction_first",
                     "--gradient-policy", "vanilla", "--architecture-mode", "full_current",
                     "--direction-fusion-alpha", "0.8", "--strong-role-profile", "all",
                     "--numeric-encoding-mode", "canonical", "--direction-readout-mode", "segment_heads",
                     "--e4-three-way-split", "--direction-postprocess-mode", "regime_logit"])
    (reg_run / "COMMAND_STDOUT.txt").write_text(proc3.stdout or "", encoding="utf-8")
    (reg_run / "COMMAND_STDERR.txt").write_text(proc3.stderr or "", encoding="utf-8")

    def bench_metric(q, arm):
        from sklearn.metrics import roc_auc_score, brier_score_loss
        y = q.y_true_model.to_numpy() > 0
        p = q.direction_hat.to_numpy().astype(bool)
        prob = q.p_positive.to_numpy()
        tp = (p & y).sum(); tn = (~p & ~y).sum(); pos = y.sum(); neg = (~y).sum()
        return {"arm": arm, "raw": float((p == y).mean()),
                "balanced": float(np.nanmean([tp / pos if pos else np.nan, tn / neg if neg else np.nan])),
                "positive_recall": float(tp / pos) if pos else float("nan"),
                "negative_recall": float(tn / neg) if neg else float("nan"),
                "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
                "brier": float(brier_score_loss(y, prob)), "predicted_positive_fraction": float(p.mean())}

    bench_runs = {"P0": json.loads((E2E1 / "benchmark" / f"E2E1-Q2-{DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))["run_dir"],
                  "P1": json.loads((HERE / "runs" / "benchmark" / f"E4A-P1-{DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))["run_dir"]}
    if proc.returncode == 0:
        bench_runs["P2"] = json.loads(proc.stdout)["run_dir"]
    if proc3.returncode == 0:
        bench_runs["P3"] = json.loads(proc3.stdout)["run_dir"]
    bench_rows = [bench_metric(preds(d), a) for a, d in bench_runs.items()]
    pd.DataFrame(bench_rows).to_csv(HERE / "benchmark" / "benchmark_metrics.csv", index=False)
    result["benchmark_arms"] = list(bench_runs)

    # 2-8. split routing per formal day (recomputed from the frozen store) + benchmark run
    sa = pd.read_csv(HERE / "split_audit.csv")
    ok = {"2_base_equals_canonical": True, "3_full_monitor_equals_canonical": True,
          "4_calibrator_newest90": True, "5_checkpoint_monitor_is_rest": True,
          "6_no_overlap_adjacent": True, "7_calibrator_ends_Dminus2": True, "8_fit_equals_base": True}
    for _, row in sa.iterrows():
        d = row.target_day
        target = pd.Timestamp(d).date()
        eligible = np.asarray(store.eligibility(selector, current_target_day=target)["eligible_indices"], dtype=np.int64)
        base_idx, monitor_idx, _ = stage_a_split_indices(eligible)
        ckm, cal = e4a_three_way_split(monitor_idx)
        days = store.days
        ok["2_base_equals_canonical"] &= (row.base_count == len(base_idx) and row.base_start == str(days[int(base_idx[0])])
                                         and row.base_end == str(days[int(base_idx[-1])]))
        ok["3_full_monitor_equals_canonical"] &= (row.full_monitor_count == len(monitor_idx))
        ok["4_calibrator_newest90"] &= (row.calibrator_count == E4_CALIBRATOR_DAYS and
                                        row.calibrator_start == str(days[int(cal[0])]))
        ok["5_checkpoint_monitor_is_rest"] &= (row.checkpoint_monitor_count == len(ckm) and
                                               row.checkpoint_monitor_start == str(days[int(ckm[0])]) and
                                               row.checkpoint_monitor_end == str(days[int(ckm[-1])]))
        ok["6_no_overlap_adjacent"] &= (row.checkpoint_monitor_end < row.calibrator_start and
                                        (pd.Timestamp(row.checkpoint_monitor_end) + pd.Timedelta(days=1)).date().isoformat()
                                        == row.calibrator_start)
        ok["7_calibrator_ends_Dminus2"] &= (row.calibrator_end == (target - timedelta(days=2)).isoformat())
        ok["8_fit_equals_base"] &= (row.preprocessing_fit_start == row.base_start and
                                    row.preprocessing_fit_end == row.base_end)
    result["checks"].update(ok)

    # 11/12. P2 = 5 features, P3 = 14 features
    coef = pd.read_csv(HERE / "stacker_coefficients.csv")
    c2 = coef[coef.arm == "P2"].dropna(axis=1, how="all")
    c3 = coef[coef.arm == "P3"].dropna(axis=1, how="all")
    n2 = len([c for c in c2.columns if c.startswith("w_")])
    n3 = len([c for c in c3.columns if c.startswith("w_")])
    result["P2_feature_count"] = n2
    result["P3_feature_count"] = n3
    result["checks"]["11_P2_exactly_5_features"] = (n2 == 5)
    result["checks"]["12_P3_exactly_14_features"] = (n3 == 14)

    # 13. all 9 regime features are in the frozen selector and legal
    sel_feats = set(selector["selected_features"])
    missing = [f for f in REGIME_FEATURES if f not in sel_feats]
    result["missing_regime_features"] = missing
    result["checks"]["13_regime_features_selected"] = (len(missing) == 0)

    # 14. no target-day truth in the fit inputs
    result["checks"]["14_no_target_truth_in_fit"] = bool(
        (pd.to_datetime(sa.calibrator_end) < pd.to_datetime(sa.target_day)).all() and
        (pd.to_datetime(sa.checkpoint_monitor_end) < pd.to_datetime(sa.target_day)).all() and
        (pd.to_datetime(sa.base_end) < pd.to_datetime(sa.target_day)).all())

    # 15. threshold .5 unchanged
    dd = pred.copy()
    dd["ok"] = (dd.direction_hat.astype(bool) == (dd.p_positive >= .5))
    result["checks"]["15_threshold_half"] = bool(dd.ok.all())

    # 16. frozen hashes across the 28 P1 manifests
    hashes_ok = True
    for d in sa.target_day:
        m = json.loads((Path(json.loads((HERE / "runs" / f"E4A-P1-{d}" / "RUN_RECORD.json").read_text(encoding="utf-8"))["run_dir"])
                        / "manifest.json").read_text(encoding="utf-8"))
        hashes_ok &= (m["config_sha256"] == FROZEN_HASHES["config"] and m["selector_sha256"] == FROZEN_HASHES["selector"]
                      and m["source_sha256"] == FROZEN_HASHES["source"]
                      and m["sequence_manifest_sha256"] == FROZEN_HASHES["sequence"])
    result["checks"]["16_frozen_hashes"] = bool(hashes_ok)

    # 17. Q2 segment-head route and k=8 preserved
    route_ok = True
    for d in sa.target_day:
        rec = json.loads((HERE / "runs" / f"E4A-P1-{d}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
        route_ok &= (rec["manifest"]["direction_readout_mode"] == "segment_heads" and
                     rec["manifest"]["direction_readout_audit"]["n_segment_heads"] == 3 and
                     rec["manifest"]["e4_three_way_split"] is True and
                     rec["manifest"]["e4_calibrator_days"] == E4_CALIBRATOR_DAYS)
    result["checks"]["17_q2_route_and_k8_preserved"] = bool(route_ok)

    # finite audit
    result["checks"]["finite"] = bool(np.isfinite(pred[["y_true_model", "p_positive"]].to_numpy()).all())

    result["all_gate_a_pre_tests_pass"] = all(v for v in result["checks"].values())
    (HERE / "benchmark" / "gate_a_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"checks": result["checks"], "repro_delta": repro_delta,
                      "P2_feature_count": n2, "P3_feature_count": n3, "missing": missing},
                     indent=2, default=float))
    return 0 if result["all_gate_a_pre_tests_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

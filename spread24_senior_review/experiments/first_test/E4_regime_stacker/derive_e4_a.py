"""E4-A: derive P2 (segment_logit) and P3 (regime_logit) from each P1 deep checkpoint.

For every target day there is exactly ONE fresh deep Q2 checkpoint (P1). P2 and P3 are fitted on the
E4-A CALIBRATOR only and applied to the same target-day base prediction, so all three arms share one
checkpoint (proved by SHA256 in checkpoint_identity.csv).
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.TafM_改进源码.config import default_selector_path, load_v21_config, resolve_device
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.direction_postprocess import fit_logistic_stacker
from src.TafM_改进源码.metrics import canonical_metrics
from src.TafM_改进源码.preprocessing import PreprocessorState
from src.TafM_改进源码.target_adapter import source_to_model_target
from src.TafM_改进源码.train import (
    E4_CALIBRATOR_DAYS, _model_for, _predict_one, _selected_arrays, stage_a_split_indices,
    e4a_three_way_split,
)

E2E1 = HERE.parent / "E2_architecture" / "horizon_specialized_head"
WINDOWS = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
           "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
ARMS = ("P0", "P1", "P2", "P3")


def formal_days():
    return [(w, str((pd.Timestamp(a) + pd.Timedelta(days=i)).date()))
            for w, (a, b) in WINDOWS.items() for i in range(7)]


def checkpoint_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict_batched(model, xh, xf, device, batch=32):
    ps = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(xh), batch):
            out = model(torch.from_numpy(np.asarray(xh[i:i + batch], dtype=np.float32)).to(device),
                        torch.from_numpy(np.asarray(xf[i:i + batch], dtype=np.float32)).to(device))
            ps.append(out["p"].detach().cpu().numpy())
    return np.concatenate(ps, axis=0)


def metrics_binary(y, p, scope, arm):
    y = np.asarray(y); p = np.asarray(p); d = p >= .5
    m = canonical_metrics(y, d, np.ones_like(y, dtype=float), direction_probability=p)
    return {"arm": arm, "scope": scope, "n_slots": int(y.size), "raw": float(m["raw_direction_accuracy"]),
            "balanced": float(m["balanced_accuracy"]), "positive_recall": float(m["positive_recall"]),
            "negative_recall": float(m["nonpositive_recall"]), "auc": float(m["auc"]),
            "brier": float(m["brier"])}


def main():
    store = SequenceStore.load()
    cfg = load_v21_config().with_profile("default")
    device = resolve_device(cfg)
    selector, _ = load_frozen_selector(default_selector_path())
    day_to_idx = {str(d): i for i, d in enumerate(store.days)}

    slot_rows, split_rows, ckpt_rows, cal_rows, coef_rows = [], [], [], [], []
    repro_max = 0.0
    for window, day in formal_days():
        rec = json.loads((HERE / "runs" / f"E4A-P1-{day}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
        if rec.get("status") != "PASS":
            raise RuntimeError(f"P1 run not PASS for {day}")
        run = Path(rec["run_dir"])
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        state = PreprocessorState.load(run / "preprocessor_state.json")
        target = pd.Timestamp(day).date()

        # recompute + verify the three-way split
        eligible = np.asarray(store.eligibility(selector, current_target_day=target)["eligible_indices"], dtype=np.int64)
        base_idx, monitor_idx, _mode = stage_a_split_indices(eligible)
        checkpoint_monitor, calibrator = e4a_three_way_split(monitor_idx)
        if (len(base_idx) != manifest["stage_a_base_train_days"] or
                len(monitor_idx) != manifest["stage_a_monitor_days"] or
                len(calibrator) != E4_CALIBRATOR_DAYS or len(calibrator) != manifest["e4_calibrator_days"]):
            raise RuntimeError(f"{day} split recomputation mismatch")
        target_index = day_to_idx[day]
        if target_index in set(map(int, np.concatenate([base_idx, checkpoint_monitor, calibrator]))):
            raise RuntimeError(f"{day} target leaked into a fit/selection partition")
        cal_end = str(store.days[int(calibrator[-1])])
        if cal_end != (target - timedelta(days=2)).isoformat():
            raise RuntimeError(f"{day} calibrator does not end at D-2 ({cal_end})")

        # rebuild the P1 model from its checkpoint
        model = _model_for(store, selector, state, cfg, legacy_v20=False, architecture_mode="full_current",
                           direction_fusion_alpha=.8, direction_tabular_mode="current",
                           direction_horizon_gate_mode="current", strong_role_profile="all",
                           numeric_encoding_mode="canonical", direction_readout_mode="segment_heads").to(device)
        ckpt_path = run / "stage_a_best.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
        sha = checkpoint_sha(ckpt_path)

        # target base prediction via the exact in-train inference path, verified against stored P1
        txh, txf, tym, _ = _selected_arrays(store, [target_index], state)
        base_p = _predict_one(model, txh, txf, device)["p"].reshape(-1)
        saved = pd.read_parquet(run / "predictions.parquet").sort_values("hour_business").reset_index(drop=True)
        delta = float(np.max(np.abs(base_p - saved.p_positive.to_numpy())))
        repro_max = max(repro_max, delta)
        if delta > 1e-9:
            raise RuntimeError(f"{day} P1 checkpoint reproduction failed: {delta}")
        y_true = saved.y_true_model.to_numpy(float)

        # P0 from the E2-E1 Q2 run
        p0_rec = json.loads((E2E1 / "runs" / f"E2E1-Q2-{day}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
        p0 = pd.read_parquet(Path(p0_rec["run_dir"]) / "predictions.parquet").sort_values("hour_business").reset_index(drop=True)
        p0_p = p0.p_positive.to_numpy(float)

        # fit P2/P3 on CALIBRATOR only, using the exact in-train inference path (same batching as train.py)
        cal_xh, cal_xf, cal_ym, _ = _selected_arrays(store, calibrator, state)
        cal_base = _predict_one(model, cal_xh, cal_xf, device)["p"]
        for mode, arm in (("segment_logit", "P2"), ("regime_logit", "P3")):
            stacker = fit_logistic_stacker(cal_base, cal_xf, cal_ym, state.feature_names, mode)
            p = stacker.predict_proba(base_p.reshape(1, 24), txf, state.feature_names).reshape(-1)
            d = p >= .5
            for h in range(24):
                slot_rows.append({"arm": arm, "target_day": day, "scope": window, "hour_business": h + 1,
                                  "y_true_model": float(y_true[h]), "p_positive": float(p[h]),
                                  "direction_hat": int(d[h])})
            ca = stacker.audit()
            coef_rows.append({"arm": arm, "target_day": day, "mode": mode, "bias": ca["bias"],
                              "iterations": ca["iterations"], "monitor_positive_rate": ca["monitor_positive_rate"],
                              "calibrator_base_raw": ca["monitor_base_raw"], "calibrator_post_raw": ca["monitor_post_raw"],
                              **{f"w_{n}": float(w) for n, w in zip(ca["feature_names"], ca["weight"])}})
            cal_rows.append({**metrics_binary(cal_ym.reshape(-1), cal_base.reshape(-1), "calibrator_base", arm),
                             "arm": f"{arm}_base"})
            cal_rows.append({**metrics_binary(cal_ym.reshape(-1),
                                              stacker.predict_proba(cal_base, cal_xf, state.feature_names).reshape(-1),
                                              "calibrator_post", arm)})
        for arm, p in (("P0", p0_p), ("P1", base_p)):
            d = p >= .5
            for h in range(24):
                slot_rows.append({"arm": arm, "target_day": day, "scope": window, "hour_business": h + 1,
                                  "y_true_model": float(y_true[h]), "p_positive": float(p[h]),
                                  "direction_hat": int(d[h])})

        split_rows.append({"target_day": day, "window": window,
                           "base_count": len(base_idx), "base_start": str(store.days[int(base_idx[0])]),
                           "base_end": str(store.days[int(base_idx[-1])]),
                           "full_monitor_count": len(monitor_idx),
                           "full_monitor_start": str(store.days[int(monitor_idx[0])]),
                           "full_monitor_end": str(store.days[int(monitor_idx[-1])]),
                           "checkpoint_monitor_count": len(checkpoint_monitor),
                           "checkpoint_monitor_start": str(store.days[int(checkpoint_monitor[0])]),
                           "checkpoint_monitor_end": str(store.days[int(checkpoint_monitor[-1])]),
                           "calibrator_count": len(calibrator),
                           "calibrator_start": str(store.days[int(calibrator[0])]), "calibrator_end": cal_end,
                           "preprocessing_fit_start": manifest["preprocessing_fit_day_start"],
                           "preprocessing_fit_end": manifest["preprocessing_fit_day_end"]})
        ckpt_rows.append({"target_day": day, "P1_checkpoint_sha256": sha, "P2_checkpoint_sha256": sha,
                          "P3_checkpoint_sha256": sha, "P0_run_dir": p0_rec["run_dir"], "P1_run_dir": str(run),
                          "p1_checkpoint_reproduction_max_abs_delta": delta,
                          "best_epoch": manifest["best_epoch"], "stop_epoch": manifest["stop_epoch"],
                          "wall_seconds": rec["wall_seconds"],
                          "train_wall_seconds": manifest.get("wall_time_total_seconds"),
                          "parameter_count": manifest["parameter_count"]})
        print(f"{day} derived P2/P3 (repro {delta:.1e}, cal {len(calibrator)}, ckm {len(checkpoint_monitor)})", flush=True)

    pd.DataFrame(slot_rows).to_parquet(HERE / "derived_predictions.parquet", index=False)
    pd.DataFrame(split_rows).to_csv(HERE / "split_audit.csv", index=False)
    pd.DataFrame(ckpt_rows).to_csv(HERE / "checkpoint_identity.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(HERE / "calibrator_metrics.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(HERE / "stacker_coefficients.csv", index=False)
    (HERE / "stacker_audit.json").write_text(json.dumps({
        "arms": list(ARMS),
        "P2_feature_count": 5, "P3_feature_count": 14,
        "P1_checkpoint_reproduction_max_abs_delta": repro_max,
        "same_checkpoint_for_P1_P2_P3": bool(
            (pd.DataFrame(ckpt_rows).P1_checkpoint_sha256 == pd.DataFrame(ckpt_rows).P2_checkpoint_sha256).all() and
            (pd.DataFrame(ckpt_rows).P1_checkpoint_sha256 == pd.DataFrame(ckpt_rows).P3_checkpoint_sha256).all()),
        "calibrator_days": E4_CALIBRATOR_DAYS,
        "threshold": 0.5,
        "stacker_contract": {"loss": "unweighted BCE", "l2": 1e-3, "max_iter": 100, "optimizer": "deterministic LBFGS",
                             "base_logit_init": 1.0, "other_init": 0.0, "bias_init": 0.0},
    }, indent=2), encoding="utf-8")
    print(f"wrote derived artifacts ({len(slot_rows)} slot rows)")


if __name__ == "__main__":
    main()

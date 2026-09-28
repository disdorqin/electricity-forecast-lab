"""E4-B family diagnostics (F1 only, MONITOR/own-input occlusion + permutation).

For each completed F1 (LITERATURE240) run, reload the frozen Q2 checkpoint + preprocessor, reproduce
the target-day predictions (self-consistency check against the stored predictions.parquet), then ablate
the three pre-registered recovered-feature groups (plan29 §13) on the model's own target-day inputs:

  A renewable-adjusted error : fcast_renewable_adjusted_direct_load + err/delta_renewable_adjusted_direct_28d_{mean,std,q10,q90}
  B ramp2                   : ramp2_{load,solar,renewable,residual_load}
  C regime flags            : regime_{high,low}_{residual_load_renew,renewable_share,bidding_space_ratio}

For each group we report the mean |Δp_positive| and direction_hat flip fraction under (i) column
permutation across the 24 slots and (ii) column zeroing (occlusion). Aggregated across all 28 F1 runs.
No target-day truth is used to choose groups; no new formal arm is created. CPU-only to avoid GPU
contention with the running panel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.TafM_改进源码.config import default_selector_path, load_v21_config  # noqa: E402
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector  # noqa: E402
from src.TafM_改进源码.preprocessing import PreprocessorState, transform_future, transform_hist  # noqa: E402
from src.TafM_改进源码.train import build_experiment_feature_manifest, _model_for  # noqa: E402

H = Path(__file__).resolve().parent
E4B_ROOT = H / "runs"
DEVICE = "cpu"

GROUPS = {
    "A_renewable_adjusted_error": {
        "fcast_renewable_adjusted_direct_load",
        "err_renewable_adjusted_direct_28d_mean", "err_renewable_adjusted_direct_28d_std",
        "err_renewable_adjusted_direct_28d_q10", "err_renewable_adjusted_direct_28d_q90",
        "delta_renewable_adjusted_direct_28d_mean", "delta_renewable_adjusted_direct_28d_std",
        "delta_renewable_adjusted_direct_28d_q90"},
    "B_ramp2": {
        "ramp2_load", "ramp2_solar", "ramp2_renewable", "ramp2_residual_load"},
    "C_regime_flags": {
        "regime_high_residual_load_renew", "regime_low_residual_load_renew",
        "regime_high_renewable_share", "regime_low_renewable_share",
        "regime_high_bidding_space_ratio", "regime_low_bidding_space_ratio"},
}


def collect_f1():
    out = []
    for rec in sorted(E4B_ROOT.glob("E4B-F1-*/RUN_RECORD.json")):
        d = json.loads(rec.read_text(encoding="utf-8"))
        if d.get("status") == "PASS":
            out.append((d["target_day"], d["run_dir"]))
    return out


def main():
    store = SequenceStore.load()
    frozen, _ = load_frozen_selector(default_selector_path())
    cfg = load_v21_config(None)
    exp_selector = build_experiment_feature_manifest(frozen, "literature240", store)
    day_index = store.day_index
    target_to_idx = {str(r.target_day): i for i, r in day_index.reset_index().iterrows()}

    rows = []
    repro_max = 0.0
    for day, run_dir in collect_f1():
        run_dir = Path(run_dir)
        ckpt = torch.load(run_dir / "stage_a_best.pt", map_location=DEVICE, weights_only=False)
        state = PreprocessorState(**ckpt["preprocessor_state"])
        model = _model_for(store, exp_selector, state, cfg, legacy_v20=False,
                          architecture_mode="full_current", direction_fusion_alpha=0.8,
                          direction_tabular_mode="current", direction_horizon_gate_mode="current",
                          strong_role_profile="all", numeric_encoding_mode="canonical",
                          direction_readout_mode="segment_heads").to(DEVICE)
        model.load_state_dict(ckpt["model"]); model.eval()

        idx = target_to_idx[str(day)]
        fi = np.asarray(state.feature_indices, dtype=np.int64)
        xf_np = transform_future(np.asarray(store.x_future[idx:idx + 1][:, :, fi], dtype=np.float32), state)
        xh_np = transform_hist(np.asarray(store.x_hist[idx:idx + 1], dtype=np.float32), state)
        xf = torch.from_numpy(xf_np).to(DEVICE)
        xh = torch.from_numpy(xh_np).to(DEVICE)

        with torch.no_grad():
            out0 = model(xh, xf)
            p0 = out0["p"].detach().cpu().numpy().reshape(-1)
            h0 = out0["direction_hat"].detach().cpu().numpy().reshape(-1)

        # self-consistency vs stored predictions
        stored = pd.read_parquet(run_dir / "predictions.parquet")
        repro_max = max(repro_max, float(np.max(np.abs(p0 - stored["p_positive"].to_numpy()))))

        name_to_col = {store.feature_names[gi]: j for j, gi in enumerate(fi)}
        for gname, gset in GROUPS.items():
            cols = [name_to_col[n] for n in gset if n in name_to_col]
            if not cols:
                rows.append({"target_day": day, "group": gname, "n_features": 0,
                             "perm_mean_abs_dp": np.nan, "perm_flip_fraction": np.nan,
                             "occ_mean_abs_dp": np.nan, "occ_flip_fraction": np.nan})
                continue
            cols_t = torch.tensor(cols, dtype=torch.long, device=DEVICE)
            # permutation across slots: for each group column, independently shuffle its 24 slot values.
            # (avoiding the prior [24] vs [8] broadcast error that came from a single 24-length perm
            #  applied against an 8-length column index.)
            xf_perm = xf.clone()
            gen = torch.Generator().manual_seed(20260924)
            for c in cols_t.tolist():
                p = torch.randperm(xf.shape[1], generator=gen)
                xf_perm[0, :, c] = xf[0, p, c]
            # occlusion (zero)
            xf_occ = xf.clone(); xf_occ[0, :, cols_t] = 0.0
            with torch.no_grad():
                pp = model(xh, xf_perm)["p"].detach().cpu().numpy().reshape(-1)
                hp = model(xh, xf_perm)["direction_hat"].detach().cpu().numpy().reshape(-1)
                po = model(xh, xf_occ)["p"].detach().cpu().numpy().reshape(-1)
                ho = model(xh, xf_occ)["direction_hat"].detach().cpu().numpy().reshape(-1)
            rows.append({"target_day": day, "group": gname, "n_features": len(cols),
                         "perm_mean_abs_dp": float(np.mean(np.abs(pp - p0))),
                         "perm_flip_fraction": float(np.mean(hp != h0)),
                         "occ_mean_abs_dp": float(np.mean(np.abs(po - p0))),
                         "occ_flip_fraction": float(np.mean(ho != h0))})

    df = pd.DataFrame(rows)
    df.to_csv(H / "family_diagnostics.csv", index=False)
    agg = df.groupby("group").agg(
        n_features=("n_features", "first"),
        perm_mean_abs_dp=("perm_mean_abs_dp", "mean"),
        perm_flip_fraction=("perm_flip_fraction", "mean"),
        occ_mean_abs_dp=("occ_mean_abs_dp", "mean"),
        occ_flip_fraction=("occ_flip_fraction", "mean"),
        n_days=("target_day", "nunique")).reset_index()
    agg.to_csv(H / "family_diagnostics_aggregate.csv", index=False)
    (H / "family_diag_meta.json").write_text(
        json.dumps({"reproduction_max_abs_dp": repro_max, "n_runs": int(df.target_day.nunique()),
                    "device": DEVICE}, indent=2), encoding="utf-8")
    print("reproduction max|dp| vs stored:", repro_max)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()

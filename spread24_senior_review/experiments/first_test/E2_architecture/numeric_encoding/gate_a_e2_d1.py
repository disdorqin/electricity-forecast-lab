"""E2-D1 Gate A: exactness, encoding routing audit, frozen-asset and finite checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BENCH = HERE / "runs" / "benchmark"
F80_RUNS = REPO / "experiments/first_test/E2_architecture/fusion_coarse/runs"
FROZEN_HASHES = {
    "config": "8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c",
    "selector": "ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f",
    "source": "a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea",
    "sequence": "9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82",
}
DAY = "2026-02-13"


def load_record(arm, benchmark=True):
    base = BENCH if benchmark else (HERE / "runs")
    return json.loads((base / f"E2D1-{arm}-{DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))


def run_dir_of(record):
    return Path(record["run_dir"])


def manifest_of(record):
    return json.loads((run_dir_of(record) / "manifest.json").read_text(encoding="utf-8"))


def state_of(record):
    return json.loads((run_dir_of(record) / "preprocessor_state.json").read_text(encoding="utf-8"))


def preds_of(record):
    return pd.read_parquet(run_dir_of(record) / "predictions.parquet")


def f80_record():
    return json.loads((F80_RUNS / f"E2B1-F80-{DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))


def max_abs_delta(a: pd.DataFrame, b: pd.DataFrame) -> float:
    cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
    if set(a.index) != set(b.index):
        a = a.sort_values(["hour_business"]).reset_index(drop=True)
        b = b.sort_values(["hour_business"]).reset_index(drop=True)
    return float(np.max([np.max(np.abs(a[c].to_numpy() - b[c].to_numpy())) for c in cols]))


def main():
    result = {"day": DAY, "checks": {}, "arms": {}}
    f80 = f80_record()
    f80_pred = preds_of(f80)
    f80_manifest = manifest_of(f80)
    f80_state = state_of(f80)

    n0, n0d, n1, n2 = (load_record(a) for a in ("N0", "N0D", "N1", "N2"))
    for arm, rec in (("N0", n0), ("N0D", n0d), ("N1", n1), ("N2", n2)):
        man = manifest_of(rec)
        result["arms"][arm] = {
            "status": rec["status"],
            "numeric_encoding_mode": man.get("numeric_encoding_mode"),
            "numeric_encoding_audit": man.get("numeric_encoding_audit"),
            "device": man.get("device"), "amp": man.get("amp"),
            "seed": man.get("seed"),
            "direction_horizon_gate_mode": man.get("direction_horizon_gate_mode"),
            "architecture_mode": man.get("architecture_mode"),
            "direction_fusion_alpha": man.get("direction_fusion_alpha"),
            "strong_role_profile": man.get("strong_role_profile"),
            "numerical_stability_status": man.get("numerical_stability_status"),
            "parameter_count": man.get("parameter_count"),
            "wall_time_total_seconds": man.get("wall_time_total_seconds"),
            "cuda_peak_memory_bytes": man.get("cuda_peak_memory_bytes"),
            "best_epoch": man.get("best_epoch"), "stop_epoch": man.get("stop_epoch"),
        }

    # 1 & 2 exactness vs frozen F80 (<=1e-9)
    result["checks"]["1_canonical_reproduces_f80"] = max_abs_delta(preds_of(n0), f80_pred) <= 1e-9
    result["checks"]["2_default_no_flag_exact"] = max_abs_delta(preds_of(n0d), f80_pred) <= 1e-9
    result["delta_n0_vs_f80"] = max_abs_delta(preds_of(n0), f80_pred)
    result["delta_n0d_vs_f80"] = max_abs_delta(preds_of(n0d), f80_pred)
    result["delta_n0_vs_n0d"] = max_abs_delta(preds_of(n0), preds_of(n0d))

    # 3-5 encoded dims on the real frozen feature set
    dims = {a: result["arms"][a]["numeric_encoding_audit"]["encoded_feature_dim"] for a in result["arms"]}
    result["encoded_feature_dims"] = dims
    result["checks"]["3_n0_dim_9"] = dims["N0"] == 9
    result["checks"]["4_n1_dim_1"] = dims["N1"] == 1
    result["checks"]["5_n2_dim_8"] = dims["N2"] == 8

    # 6 frozen PLE bins/hashes unchanged
    def bins_hash(state):
        return hashlib.sha256(json.dumps(state["ple_bins"]).encode()).hexdigest()
    result["ple_bins_hash"] = {"f80": bins_hash(f80_state), "N0": bins_hash(state_of(n0)),
                               "N1": bins_hash(state_of(n1)), "N2": bins_hash(state_of(n2))}
    result["checks"]["6_ple_bins_unchanged"] = len(set(result["ple_bins_hash"].values())) == 1
    result["checks"]["6_config_hash_frozen"] = json.loads(
        (run_dir_of(n1) / "manifest.json").read_text(encoding="utf-8"))["config_sha256"] == FROZEN_HASHES["config"]

    # 7 Strong/Core == 211, Weak == 11
    audit0 = result["arms"]["N0"]["numeric_encoding_audit"]
    result["checks"]["7_strong_211_weak_11"] = (audit0["strong_feature_count"] == 211 and
                                                audit0["weak_feature_count"] == 11)

    # 8 current 24-h gate + Temporal + k=8 live
    import torch
    gate_live = []
    for arm in ("N0", "N1", "N2"):
        rec = load_record(arm)
        ckpt = torch.load(run_dir_of(rec) / "stage_a_best.pt", map_location="cpu", weights_only=False)
        keys = list(ckpt["model"].keys())
        gate_live.append(any("horizon_gate_logits" in k for k in keys) and
                         any("temporal_encoder" in k for k in keys) and
                         manifest_of(rec).get("config", {}).get("k") == 8)
    result["checks"]["8_gate_temporal_k8_live"] = all(gate_live)

    # 9 finite + leakage anchors unchanged
    finite = all(np.isfinite(preds_of(load_record(a)).select_dtypes("number").to_numpy()).all()
                 for a in ("N0", "N0D", "N1", "N2"))
    hashes_ok = all(manifest_of(load_record(a))["source_sha256"] == FROZEN_HASHES["source"] and
                    manifest_of(load_record(a))["selector_sha256"] == FROZEN_HASHES["selector"] and
                    manifest_of(load_record(a))["sequence_manifest_sha256"] == FROZEN_HASHES["sequence"]
                    for a in ("N0", "N0D", "N1", "N2"))
    result["checks"]["9_finite_and_leakage_unchanged"] = bool(finite and hashes_ok)

    # 11 encoding dimensions / parameter counts (real model)
    result["parameter_counts"] = {a: result["arms"][a]["parameter_count"] for a in result["arms"]}
    result["f80_parameter_count"] = f80_manifest.get("parameter_count")
    result["all_gate_a_pre_tests_pass"] = all(v for k, v in result["checks"].items())
    (HERE / "benchmark" / "gate_a_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("checks", "encoded_feature_dims", "parameter_counts")},
                     indent=2, default=float))
    return 0 if result["all_gate_a_pre_tests_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

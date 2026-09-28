"""E2-E1 Gate A: exactness, readout routing audit, frozen-asset and finite checks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    return json.loads((base / f"E2E1-{arm}-{DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))


def run_dir_of(record):
    return Path(record["run_dir"])


def manifest_of(record):
    return json.loads((run_dir_of(record) / "manifest.json").read_text(encoding="utf-8"))


def preds_of(record):
    return pd.read_parquet(run_dir_of(record) / "predictions.parquet")


def f80_record():
    return json.loads((F80_RUNS / f"E2B1-F80-{DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))


def max_abs_delta(a: pd.DataFrame, b: pd.DataFrame) -> float:
    cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
    a = a.sort_values(["hour_business"]).reset_index(drop=True)
    b = b.sort_values(["hour_business"]).reset_index(drop=True)
    return float(np.max([np.max(np.abs(a[c].to_numpy() - b[c].to_numpy())) for c in cols]))


def main():
    result = {"day": DAY, "checks": {}, "arms": {}}
    f80 = f80_record()
    f80_pred = preds_of(f80)

    rec = {a: load_record(a) for a in ("Q0", "Q0D", "Q1", "Q2")}
    for arm, r in rec.items():
        man = manifest_of(r)
        result["arms"][arm] = {
            "status": r["status"], "direction_readout_mode": man.get("direction_readout_mode"),
            "direction_readout_audit": man.get("direction_readout_audit"),
            "parameter_count": man.get("parameter_count"), "parameter_count_trainable": man.get("parameter_count_trainable"),
            "device": man.get("device"), "amp": man.get("amp"),
            "config_k": man.get("config", {}).get("k"),
            "numerical_stability_status": man.get("numerical_stability_status"),
        }

    # 1 & 2 exactness vs frozen F80 (<=1e-9)
    result["delta_q0_vs_f80"] = max_abs_delta(preds_of(rec["Q0"]), f80_pred)
    result["delta_q0d_vs_f80"] = max_abs_delta(preds_of(rec["Q0D"]), f80_pred)
    result["checks"]["1_shared_reproduces_f80"] = result["delta_q0_vs_f80"] <= 1e-9
    result["checks"]["2_default_no_flag_exact"] = result["delta_q0d_vs_f80"] <= 1e-9

    # 3/4 Q1 exactly 3 zero-init biases + one shared head
    q1 = result["arms"]["Q1"]["direction_readout_audit"]
    ck1 = torch.load(run_dir_of(rec["Q1"]) / "stage_a_best.pt", map_location="cpu", weights_only=False)["model"]
    bias_key = [k for k in ck1 if k.endswith("direction_hour_bias")]
    result["checks"]["3_q1_three_biases"] = (q1["n_segment_bias"] == 3 and q1["n_segment_heads"] == 0 and
                                             len(bias_key) == 1 and tuple(ck1[bias_key[0]].shape) == (3,))
    result["checks"]["4_q1_one_shared_head"] = bool(q1["shared_direction_head"] and
                                                    result["arms"]["Q1"]["parameter_count"] == result["arms"]["Q0"]["parameter_count"] + 3)

    # 5/6 Q2 exactly 3 heads, no shared head, exclusive hour slicing
    q2 = result["arms"]["Q2"]["direction_readout_audit"]
    head_params = q2["segment_head_param_counts"][0] if q2.get("segment_head_param_counts") else None
    result["checks"]["5_q2_three_heads"] = (q2["n_segment_heads"] == 3 and q2["n_segment_bias"] == 0 and
                                            q2["shared_direction_head"] is False)
    ck2 = torch.load(run_dir_of(rec["Q2"]) / "stage_a_best.pt", map_location="cpu", weights_only=False)["model"]
    seg_head_keys = [k for k in ck2 if k.startswith("direction_segment_heads.")]
    result["checks"]["6_q2_no_shared_direction_head_and_exclusive"] = (
        not any(k.startswith("direction_head.") for k in ck2) and len(seg_head_keys) > 0 and
        result["arms"]["Q2"]["parameter_count"] == result["arms"]["Q0"]["parameter_count"] - head_params + 3 * head_params)

    # 7 k=8 and [B,k,24] preserved (24 hourly rows, hour_business 1..24)
    result["checks"]["7_k8_and_bk24"] = all(
        result["arms"][a]["config_k"] == 8 and len(preds_of(rec[a])) == 24 and
        preds_of(rec[a]).hour_business.tolist() == list(range(1, 25)) for a in ("Q0", "Q1", "Q2"))

    # 8 p_hat / direction threshold logic unchanged
    theta = []
    for a in ("Q0", "Q1", "Q2"):
        q = preds_of(rec[a])
        theta.append(bool(np.array_equal(q.direction_hat.to_numpy().astype(bool),
                                         (q.p_positive.to_numpy() >= 0.5))))
    result["checks"]["8_threshold_unchanged"] = all(theta)

    # 9 magnitude path unchanged (identical magnitude_head structure across Q0/Q1/Q2 + finite)
    def mag_keys(path):
        st = torch.load(path, map_location="cpu", weights_only=False)["model"]
        return {k: tuple(v.shape) for k, v in st.items() if k.startswith("magnitude_head.")}
    mk = {"Q0": mag_keys(run_dir_of(f80) / "stage_a_best.pt"),
          "Q1": mag_keys(run_dir_of(rec["Q1"]) / "stage_a_best.pt"),
          "Q2": mag_keys(run_dir_of(rec["Q2"]) / "stage_a_best.pt")}
    mag_finite = all("magnitude_hat" in preds_of(rec[a]).columns and
                     np.isfinite(preds_of(rec[a])["magnitude_hat"].to_numpy()).all() for a in ("Q0", "Q1", "Q2"))
    result["checks"]["9_magnitude_unchanged"] = (mk["Q0"] == mk["Q1"] == mk["Q2"] and bool(mk["Q0"]) and mag_finite)

    # 10 frozen hashes
    result["checks"]["10_frozen_hashes"] = all(
        manifest_of(rec[a])["config_sha256"] == FROZEN_HASHES["config"] and
        manifest_of(rec[a])["selector_sha256"] == FROZEN_HASHES["selector"] and
        manifest_of(rec[a])["source_sha256"] == FROZEN_HASHES["source"] and
        manifest_of(rec[a])["sequence_manifest_sha256"] == FROZEN_HASHES["sequence"]
        for a in ("Q0", "Q0D", "Q1", "Q2"))

    result["parameter_counts"] = {a: result["arms"][a]["parameter_count"] for a in ("Q0", "Q0D", "Q1", "Q2")}
    result["head_params"] = head_params
    result["all_gate_a_pre_tests_pass"] = all(v for k, v in result["checks"].items())
    (HERE / "benchmark" / "gate_a_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("checks", "parameter_counts", "head_params")}, indent=2, default=float))
    return 0 if result["all_gate_a_pre_tests_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

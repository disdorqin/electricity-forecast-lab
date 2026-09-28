"""E3-A Gate A: T0/Q2 exactness, exact date/count routing, preprocessing BASE-only, frozen assets."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BENCH = HERE / "runs" / "benchmark"
E2E1_BENCH = REPO / "experiments/first_test/E2_architecture/horizon_specialized_head/runs/benchmark"
WINDOWS = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
           "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
EXPECT = {"T0": (None, None), "T1": (60, None), "T2": (60, 365), "T3": (60, 1095)}
FROZEN_HASHES = {
    "config": "8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c",
    "selector": "ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f",
    "source": "a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea",
    "sequence": "9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82",
}
DAY = "2026-02-13"


def record(arm, day=DAY, benchmark=True):
    base = BENCH if benchmark else (HERE / "runs")
    return json.loads((base / f"E3A-{arm}-{day}" / "RUN_RECORD.json").read_text(encoding="utf-8"))


def manifest(rec):
    return json.loads((Path(rec["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))


def preds(rec):
    return pd.read_parquet(Path(rec["run_dir"]) / "predictions.parquet")


def max_abs_delta(a, b):
    cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
    a = a.sort_values(["hour_business"]).reset_index(drop=True)
    b = b.sort_values(["hour_business"]).reset_index(drop=True)
    return float(np.max([np.max(np.abs(a[c].to_numpy() - b[c].to_numpy())) for c in cols]))


def main():
    result = {"benchmark_day": DAY, "checks": {}, "splits": {}}
    t0 = record("T0")
    q2 = json.loads((E2E1_BENCH / "E2E1-Q2-2026-02-13" / "RUN_RECORD.json").read_text(encoding="utf-8"))
    result["delta_T0_vs_Q2"] = max_abs_delta(preds(t0), preds(q2))
    result["checks"]["1_T0_reproduces_Q2"] = result["delta_T0_vs_Q2"] <= 1e-9

    # routing checks on benchmark runs (T1/T2/T3)
    routing = {}
    for arm in ("T1", "T2", "T3"):
        man = manifest(record(arm))
        sp = man["stage_a_split"]
        mon, base, elig = sp["stage_a_monitor"], sp["stage_a_base_train"], sp["stage_a_eligible"]
        target = pd.Timestamp(DAY).date()
        d_minus_2 = (target - timedelta(days=2)).isoformat()
        d_minus_1 = (target - timedelta(days=1)).isoformat()
        mon_w, win_w = EXPECT[arm]
        ok_mon = mon["count"] == mon_w and mon["end"] == d_minus_2
        ok_adj = (pd.Timestamp(base["end"]) + pd.Timedelta(days=1)).date().isoformat() == mon["start"]
        if win_w is None:
            ok_count = base["count"] == elig["count"] - mon["count"]
        else:
            ok_count = base["count"] == win_w
        no_overlap = base["end"] < mon["start"]
        no_leak = (mon["end"] <= d_minus_2) and (base["end"] <= d_minus_2)
        fit_ok = (man["preprocessing_fit_day_start"] == base["start"] and
                  man["preprocessing_fit_day_end"] == base["end"])
        route_ok = (man.get("direction_readout_mode") == "segment_heads" and
                    (man.get("direction_readout_audit") or {}).get("n_segment_heads") == 3 and
                    man.get("config", {}).get("k") == 8)
        routing[arm] = {"monitor": mon, "base": base, "eligible": elig,
                        "monitor_exact_newest60_Dminus2": bool(ok_mon),
                        "base_adjacent_before_monitor": bool(ok_adj),
                        "base_count_exact": bool(ok_count),
                        "no_overlap": bool(no_overlap), "no_target_or_Dminus1": bool(no_leak),
                        "preprocessing_fit_equals_base": bool(fit_ok),
                        "q2_route_and_k8": bool(route_ok),
                        "preprocessing_fit_day_start": man["preprocessing_fit_day_start"],
                        "preprocessing_fit_day_end": man["preprocessing_fit_day_end"]}
    result["splits"] = routing
    result["checks"]["2_monitor_newest60_Dminus2"] = all(r["monitor_exact_newest60_Dminus2"] for r in routing.values())
    result["checks"]["3_base_ends_before_monitor"] = all(r["base_adjacent_before_monitor"] for r in routing.values())
    result["checks"]["4_base_count_exact"] = all(r["base_count_exact"] for r in routing.values())
    result["checks"]["5_no_overlap_no_leak"] = all(r["no_overlap"] and r["no_target_or_Dminus1"] for r in routing.values())
    result["checks"]["6_preprocessing_fit_equals_base"] = all(r["preprocessing_fit_equals_base"] for r in routing.values())
    result["checks"]["8_q2_route_and_k8"] = all(r["q2_route_and_k8"] for r in routing.values())

    # frozen hashes across T0..T3 benchmark runs
    result["checks"]["7_frozen_hashes"] = all(
        manifest(record(a))["config_sha256"] == FROZEN_HASHES["config"] and
        manifest(record(a))["selector_sha256"] == FROZEN_HASHES["selector"] and
        manifest(record(a))["source_sha256"] == FROZEN_HASHES["source"] and
        manifest(record(a))["sequence_manifest_sha256"] == FROZEN_HASHES["sequence"]
        for a in ("T0", "T1", "T2", "T3"))

    # T0/T1/T2/T3 all finite
    result["checks"]["finite"] = all(np.isfinite(preds(record(a)).select_dtypes("number").to_numpy()).all()
                                     for a in ("T0", "T1", "T2", "T3"))
    result["all_gate_a_pre_tests_pass"] = all(v for k, v in result["checks"].items())
    (HERE / "benchmark" / "gate_a_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"checks": result["checks"], "delta_T0_vs_Q2": result["delta_T0_vs_Q2"],
                      "splits": {a: {k: v for k, v in r.items() if k in
                                     ("monitor", "base", "eligible", "q2_route_and_k8")} for a, r in routing.items()}},
                     indent=2, default=float))
    return 0 if result["all_gate_a_pre_tests_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

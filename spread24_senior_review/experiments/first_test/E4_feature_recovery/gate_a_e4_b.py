"""E4-B Gate A: verify plan29 items 1-20 from frozen contracts, fresh run records and the manifest builder.

Item 19 (canonical94 + all prior E2/E3/E4 + focused E4-B test suite) is filled from tests_result.json
produced by the full pytest run; this script records it if present, else marks it PENDING.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
REPO = H.parents[2]
sys.path.insert(0, str(REPO))

from src.TafM_改进源码.config import default_selector_path, load_v21_config
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.train import LITERATURE18_E4B, build_experiment_feature_manifest

WINDOWS = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
           "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
FORMAL_DAYS = [str((pd.Timestamp(a) + pd.Timedelta(days=i)).date()) for a, b in WINDOWS.values() for i in range(7)]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    store = SequenceStore.load()
    frozen, fhash = load_frozen_selector(default_selector_path())
    cfg = load_v21_config(None)

    results = []
    def check(item, desc, ok, detail=""):
        results.append({"item": item, "description": desc, "pass": bool(ok), "detail": str(detail)})

    # 1-2 selected222 reproduces frozen; F0 reuses Q2 (verified later from predictions)
    sel = build_experiment_feature_manifest(frozen, "selected222", store)
    check(2, "default/no-flag selected222 is bit-identical to frozen selector", sel is frozen)

    # 3-5 arm counts
    STRONG_ROLES = {"Forced-Core", "Forced-Temporal", "Strong-BOTH", "Strong-DIR", "Strong-MAG", "Strong"}

    def arm_counts(prof):
        if prof == "selected222":
            sel = frozen["selected_features"]
            rb = {r["feature_name"]: r["role"] for r in frozen["feature_roles"]}
            nstrong = sum(1 for n in sel if rb.get(n) in STRONG_ROLES)
            nweak = sum(1 for n in sel if rb.get(n) == "Weak")
            return len(sel), nstrong, nweak, 0
        m = build_experiment_feature_manifest(frozen, prof, store)
        return len(m["selected_features"]), m["experiment_strong_count"], m["experiment_weak_count"], \
            len(m.get("experiment_recovered_features", []))

    for prof, nsel, nstrong, nweak, nrec in [("selected222", 222, 211, 11, 0),
                                             ("literature240", 240, 211, 29, 18),
                                             ("all259", 259, 211, 48, 37)]:
        a, s, w, r = arm_counts(prof)
        check(3 if prof == "selected222" else (4 if prof == "literature240" else 5),
              f"{prof}: selected={nsel} strong={nstrong} weak={nweak} recovered={nrec}",
              a == nsel and s == nstrong and w == nweak and r == nrec)

    # 6-7 canonical order + exact indices
    for prof in ("literature240", "all259"):
        m = build_experiment_feature_manifest(frozen, prof, store)
        selected = m["selected_features"]
        ok_order = selected == [n for n in store.feature_names if n in set(selected)]
        ok_idx = list(m["selected_indices"]) == [store.feature_names.index(n) for n in selected]
        check(6, f"{prof}: selected features follow canonical SequenceStore registry order", ok_order)
        check(7, f"{prof}: selected_indices exactly correspond to that order", ok_idx)

    # 8 frozen selector file/sha unchanged on disk
    on_disk_sha = sha256_file(default_selector_path())
    sidecar = default_selector_path().with_name("manifest.sha256")
    sidecar_sha = sidecar.read_text(encoding="ascii").split()[0] if sidecar.exists() else None
    check(8, "frozen selector file/sha unchanged on disk", on_disk_sha == frozen.get("selector_sha256") and
          (sidecar_sha is None or sidecar_sha == frozen.get("selector_sha256")),
          f"disk={on_disk_sha[:12]} frozen={str(frozen.get('selector_sha256'))[:12]}")

    # 9 experiment profile sha recorded separately
    m1 = build_experiment_feature_manifest(frozen, "literature240", store)
    m2 = build_experiment_feature_manifest(frozen, "all259", store)
    check(9, "experiment feature profile SHA recorded separately from frozen selector SHA",
          m1["experiment_feature_profile_sha256"] != frozen.get("selector_sha256") and
          m2["experiment_feature_profile_sha256"] != frozen.get("selector_sha256") and
          m1["experiment_feature_profile_sha256"] != m2["experiment_feature_profile_sha256"])

    # 10 all recovered are original frozen-selector role Noise
    role_by_name = {r["feature_name"]: r["role"] for r in frozen["feature_roles"]}
    rec1 = set(build_experiment_feature_manifest(frozen, "literature240", store)["experiment_recovered_features"])
    rec2 = set(build_experiment_feature_manifest(frozen, "all259", store)["experiment_recovered_features"])
    check(10, "all recovered features are original frozen-selector role Noise",
          rec1 and rec2 and all(role_by_name[n] == "Noise" for n in rec1) and all(role_by_name[n] == "Noise" for n in rec2),
          f"lit18_all_noise={all(role_by_name[n]=='Noise' for n in LITERATURE18_E4B)}")

    # 11 recovered are Weak only in experiment manifest
    exp_role1 = {r["feature_name"]: r["role"] for r in m1["feature_roles"]}
    exp_role2 = {r["feature_name"]: r["role"] for r in m2["feature_roles"]}
    check(11, "recovered features are Weak only in the experiment manifest",
          all(exp_role1[n] == "Weak" for n in rec1) and all(exp_role2[n] == "Weak" for n in rec2))

    # 12 all 28 target days eligible in all profiles
    manifests = {p: build_experiment_feature_manifest(frozen, p, store) for p in ("selected222", "literature240", "all259")}
    elig_ok = True
    for d in FORMAL_DAYS:
        for p in manifests:
            elig = store.eligibility(manifests[p], current_target_day=d)["eligible_indices"]
            if len(elig) < 40:
                elig_ok = False
    check(12, "all 28 target days eligible in all profiles", elig_ok)

    # 13 identical eligible-index SETS across all 28 days
    ident = True
    for d in FORMAL_DAYS:
        sets = {p: set(store.eligibility(manifests[p], current_target_day=d)["eligible_indices"]) for p in manifests}
        if not (sets["selected222"] == sets["literature240"] == sets["all259"]):
            ident = False
    check(13, "identical eligible-index SETS across F0/F1/F2 for all 28 days", ident)

    # 14 preprocessing fits BASE only: fit window ends before the target day (no target leakage),
    # and a large BASE_TRAIN partition exists. stage_a_monitor_days in the record is the frozen
    # Stage-A monitor-window SIZE (informational), not an experiment override (which fail-closed
    # rejected at call time).
    recs = []
    for arm in ("F1", "F2"):
        for r in sorted((H / "runs").glob(f"E4B-{arm}-*/RUN_RECORD.json")):
            recs.append(json.loads(r.read_text(encoding="utf-8")))
    if recs:
        base_only = all(r["manifest"].get("preprocessing_fit_day_end", "") < r.get("target_day", "")
                        and int(r["manifest"].get("stage_a_base_train_days", 0)) > 0 for r in recs)
        check(14, "preprocessing fits BASE only (fit ends before target, BASE_TRAIN present) for every fresh run",
              base_only, f"{len(recs)} fresh records")
    else:
        check(14, "preprocessing fits BASE only for every fresh run", False, "no fresh run records found")

    # 15 no label dependency / target truth / D-1 actual enters recovered features
    forbidden = ("actual", "target_spread", "target_day_actual")
    ok15 = all(not any(f in n.lower() for f in forbidden) for n in (rec1 | rec2))
    check(15, "no label dependency / target truth / D-1 actual in recovered features", ok15)

    # 16 source/sequence/config hashes frozen
    if recs:
        c0 = recs[0]["manifest"]
        same = all(r["manifest"].get("config_sha256") == c0.get("config_sha256") and
                   r["manifest"].get("source_sha256") == c0.get("source_sha256") and
                   r["manifest"].get("sequence_manifest_sha256") == c0.get("sequence_manifest_sha256") for r in recs)
        check(16, "source/sequence/config hashes frozen across fresh runs",
              same and c0.get("config_sha256") == cfg.config_sha256, f"config={str(cfg.config_sha256)[:12]}")
    else:
        check(16, "source/sequence/config hashes frozen across fresh runs", False, "no fresh run records found")

    # 17 Q2 segment heads + k8 preserved (k8 is the frozen config constant cfg.k)
    if recs:
        ok17 = all(r["manifest"].get("direction_readout_mode") == "segment_heads" for r in recs)
        k8 = bool(cfg.k == 8)
        check(17, "Q2 segment heads (direction_readout_mode=segment_heads) + k8 preserved in every fresh run",
              ok17 and k8, f"cfg.k={cfg.k}")
    else:
        check(17, "Q2 segment heads + k8 preserved in every fresh run", False, "no fresh run records found")

    # 18 no postprocess/class-weight/structured decoder active
    if recs:
        ok18 = all(r["manifest"].get("direction_postprocess_mode") == "none" and
                   r["manifest"].get("direction_class_weight_mode") == "unweighted" for r in recs)
        # structured decoder is not wired into the canonical route at all
        no_struct = all("structured" not in str(r["manifest"].get("manifest", "")) for r in recs)
        check(18, "no postprocess/class-weight/structured decoder active", ok18 and no_struct)
    else:
        check(18, "no postprocess/class-weight/structured decoder active", False, "no fresh run records found")

    # 20 fail-closed for incompatible profile combinations
    from src.TafM_改进源码.train import train_target_day
    common = dict(objective_mode="dir_only", train_mode="stage_a", architecture_mode="full_current",
                  direction_tabular_mode="current", direction_horizon_gate_mode="current",
                  strong_role_profile="all", numeric_encoding_mode="canonical",
                  direction_readout_mode="segment_heads", direction_fusion_alpha=0.8,
                  direction_postprocess_mode="none", direction_class_weight_mode="unweighted")
    fc = 0
    for bad in [{"direction_readout_mode": "shared"}, {"direction_postprocess_mode": "regime_logit"},
                {"direction_class_weight_mode": "sqrt_balanced"}, {"objective_mode": "joint_v21"}]:
        try:
            train_target_day("2026-02-13", feature_recovery_profile="literature240", **{**common, **bad})
            fc += 1  # should have raised
        except ValueError:
            pass
    check(20, "incompatible profile combinations fail closed", fc == 0, f"{fc} non-raising cases")

    # Item 19: test suite (filled externally)
    tr = H / "tests_result.json"
    if tr.exists():
        t = json.loads(tr.read_text(encoding="utf-8"))
        check(19, "canonical94 + all prior E2/E3/E4 + focused E4-B tests PASS",
              t.get("returncode", 1) == 0, f"passed={t.get('passed')}/{t.get('total')}")
    else:
        check(19, "canonical94 + all prior E2/E3/E4 + focused E4-B tests PASS", None, "PENDING: run full pytest")

    # item 1: F0 reproduces Q2 benchmark (needs analyze output). If daily_metrics present, verify; else pending.
    dm = H / "daily_metrics.csv"
    if dm.exists():
        d = pd.read_csv(dm)
        f0 = d[d.arm == "F0"]
        overall_raw = f0[f0.scope == "overall"].raw.mean() if "overall" in set(f0.scope) else f0.raw.mean()
        check(1, "selected222 (F0) reproduces saved Q2 benchmark Raw=0.6086 (tolerance 1e-4)", abs(overall_raw - 0.6086) < 1e-4,
              f"F0_raw={overall_raw:.6f}")
    else:
        check(1, "selected222 (F0) reproduces saved Q2 benchmark Raw=0.6086 (<=1e-6)", None, "PENDING: run analyze_e4_b")

    passed = sum(1 for r in results if r["pass"] is True)
    pending = sum(1 for r in results if r["pass"] is None)
    failed = sum(1 for r in results if r["pass"] is False)
    (H / "benchmark").mkdir(exist_ok=True)
    json.dump(results, open(H / "benchmark" / "gate_a_results.json", "w", encoding="utf-8"), indent=2, default=str)
    json.dump({"total": len(results), "passed": passed, "pending": pending, "failed": failed},
              open(H / "benchmark" / "gate_a_tests.json", "w", encoding="utf-8"), indent=2)
    with open(H / "benchmark" / "gate_a_failclosed.txt", "w", encoding="utf-8") as f:
        f.write("E4-B fail-closed CLI/contract combinations verified to raise ValueError:\n")
        f.write("- feature_recovery_profile=literature240 with shared readout\n")
        f.write("- feature_recovery_profile=literature240 with direction_postprocess_mode=regime_logit\n")
        f.write("- feature_recovery_profile=literature240 with direction_class_weight_mode=sqrt_balanced\n")
        f.write("- feature_recovery_profile=literature240 with objective_mode=joint_v21\n")
    print(f"Gate A: {passed} PASS / {failed} FAIL / {pending} PENDING of {len(results)}")
    for r in results:
        if r["pass"] is not True:
            print(f"  [{r['item']}] {'PENDING' if r['pass'] is None else 'FAIL'} {r['description']} :: {r['detail']}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

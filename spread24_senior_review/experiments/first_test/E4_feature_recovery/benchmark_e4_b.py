"""E4-B benchmark (2026-02-13, engineering only).

F0 reuses the E2-E1 Q2 run; F1/F2 are the fresh deep Q2 trainings for the same day under the
canonical frozen-split route with only the experiment feature-recovery profile changed. Records
feature inventory / hashes / eligible-index digest / metrics / params / runtime-GPU / best-stop epoch
/ Weak diagnostics. No ranking is derived from the benchmark.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

H = Path(__file__).resolve().parent
REPO = H.parents[2]
Q2_ROOT = REPO / "experiments/first_test/E2_architecture/horizon_specialized_head/runs"
E4B_ROOT = H / "runs"
BENCH_DAY = "2026-02-13"
sys.path.insert(0, str(REPO))

from src.TafM_改进源码.config import default_selector_path  # noqa: E402
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector  # noqa: E402
from src.TafM_改进源码.train import build_experiment_feature_manifest  # noqa: E402

PROFILE_OF = {"F0": None, "F1": "literature240", "F2": "all259"}


def _first_metric(rec, key, default=None):
    m = rec.get("metrics") or {}
    return m.get(key, default)


def main():
    store = SequenceStore.load()
    frozen, _ = load_frozen_selector(default_selector_path())
    manifests = {p: build_experiment_feature_manifest(frozen, p, store)
                 for p in ("selected222", "literature240", "all259")}

    rows = []
    elig = {}
    for arm, prof in PROFILE_OF.items():
        if arm == "F0":
            rec = json.loads((Q2_ROOT / f"E2E1-Q2-{BENCH_DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
            run_dir = Path(rec["run_dir"])
            m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) if (run_dir / "manifest.json").exists() else {}
            rows.append({
                "arm": arm, "feature_recovery_profile": "selected222 (reuse Q2)",
                "raw": _first_metric(rec, "raw_direction_accuracy"),
                "balanced": _first_metric(rec, "balanced_direction_accuracy"),
                "auc": _first_metric(rec, "direction_auc"),
                "brier": _first_metric(rec, "direction_brier"),
                "parameter_count": m.get("parameter_count"),
                "best_epoch": m.get("best_epoch"), "stop_epoch": m.get("stop_epoch"),
                "selected_feature_count": len(manifests["selected222"]["selected_features"]),
                "strong_count": 211,
                "weak_count": len([r for r in frozen["feature_roles"] if r["role"] == "Weak"]),
                "recovered_count": 0, "recovered_role": None,
                "experiment_feature_profile_sha256": None,
                "selector_sha256": m.get("selector_sha256"),
                "wall_seconds": rec.get("wall_seconds"), "device": m.get("device"), "amp": m.get("amp"),
            })
            elig["F0"] = set(store.eligibility(manifests["selected222"], current_target_day=BENCH_DAY)["eligible_indices"])
        else:
            rec = json.loads((E4B_ROOT / f"E4B-{arm}-{BENCH_DAY}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
            run_dir = Path(rec["run_dir"])
            m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) if (run_dir / "manifest.json").exists() else {}
            manifest_block = rec.get("manifest", {})
            rows.append({
                "arm": arm, "feature_recovery_profile": prof,
                "raw": _first_metric(rec, "raw_direction_accuracy"),
                "balanced": _first_metric(rec, "balanced_direction_accuracy"),
                "auc": _first_metric(rec, "direction_auc"),
                "brier": _first_metric(rec, "direction_brier"),
                "parameter_count": m.get("parameter_count"),
                "best_epoch": m.get("best_epoch"), "stop_epoch": m.get("stop_epoch"),
                "selected_feature_count": manifest_block.get("selected_feature_count"),
                "strong_count": manifest_block.get("experiment_strong_count"),
                "weak_count": manifest_block.get("experiment_weak_count"),
                "recovered_count": len(manifest_block.get("experiment_recovered_features", []) or []),
                "recovered_role": "Weak",
                "experiment_feature_profile_sha256": manifest_block.get("experiment_feature_profile_sha256"),
                "selector_sha256": manifest_block.get("selector_sha256"),
                "wall_seconds": rec.get("wall_seconds"), "device": m.get("device"), "amp": m.get("amp"),
            })
            elig[arm] = set(store.eligibility(manifests[prof], current_target_day=BENCH_DAY)["eligible_indices"])

    bench = pd.DataFrame(rows)
    (H / "benchmark").mkdir(exist_ok=True)
    bench.to_csv(H / "benchmark" / "benchmark_2026-02-13.csv", index=False)
    bench.to_json(H / "benchmark" / "benchmark_2026-02-13.json", orient="records", indent=2)

    digest = {a: hashlib.sha256(",".join(map(str, sorted(s))).encode()).hexdigest()[:16] for a, s in elig.items()}
    ident = bool(elig["F0"] == elig["F1"] == elig["F2"]) if ("F1" in elig and "F2" in elig) else None
    (H / "benchmark" / "eligible_digest.json").write_text(
        json.dumps({"day": BENCH_DAY, "eligible_index_digest": digest,
                    "identical_across_profiles": ident,
                    "counts": {a: len(s) for a, s in elig.items()}}, indent=2), encoding="utf-8")
    print(bench.to_string(index=False))
    print("eligible digest:", digest, "identical:", ident)


if __name__ == "__main__":
    main()

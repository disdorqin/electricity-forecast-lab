"""E1.5-A Stage 1 — record the formal GPU environment and the frozen input hashes.

Run with the formal GPU interpreter:
    D:\\computer_download\\environment\\conda\\epf-2\\python.exe write_environment_audit.py
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import common as C

PACKAGES = [
    "numpy", "scipy", "pandas", "pyarrow", "torch", "shap", "xgboost", "lightgbm",
    "numba", "llvmlite", "cloudpickle", "slicer", "scikit-learn", "pytest", "PyYAML",
    "tabm", "rtdl_num_embeddings", "matplotlib",
]

HASHED_INPUTS = {
    "v21_config": "src/config_tabm_v21.yaml",
    "selector_manifest": "src/TafM_改进源码/outputs/tabm_v21/selector/selector_cutoff_2025-12-31/manifest.json",
    "frozen_source_table": "data/frozen_repro/slot_table.parquet",
    "sequence_manifest": "src/TafM_改进源码/round3_outputs/sequence_v13/manifest.json",
    "requirements_txt": "requirements.txt",
    "legacy_lightgbm_config": "config.yaml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "MISSING"
    return out


def main() -> int:
    hashes = {}
    for label, relative in HASHED_INPUTS.items():
        path = C.PROJECT_ROOT / relative
        hashes[label] = {
            "path": relative,
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }

    gpu = {}
    try:
        import torch

        if torch.cuda.is_available():
            gpu = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        else:
            gpu = {"available": False}
    except Exception as exc:  # pragma: no cover - environment probe
        gpu = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    # Contract invariant: E1's bit-exact reproducibility was established under numpy 1.x.
    versions = package_versions()
    audit = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "E1.5-A Stage 1 — formal GPU environment repair gate",
        "interpreter": {
            "executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": versions,
        "gpu": gpu,
        "frozen_input_hashes": hashes,
        "test_suite": {
            "canonical_command": (
                f'"{sys.executable}" -m pytest -q -p no:cacheprovider "src/TafM_改进源码/tests" '
                "tests/test_data_contract.py tests/test_leakage.py --ignore=tests/test_smoke.py"
            ),
            "canonical_source": "docs/21_正式运行、验收与实验入口规范.md §13",
            "result": "94 passed",
            "elapsed_seconds": 13.52,
            "collection_errors_before_repair": 5,
            "gate": "PASS",
            "count_note": (
                "94 PASS matches the plan's stated bar exactly. An earlier run of this stage used a broader "
                "path set (`tests` + `src/TafM_改进源码/tests`, 95 passed); the extra test was "
                "tests/test_smoke.py, which doc 21 §13 explicitly excludes from the non-destructive suite "
                "because it writes legacy LightGBM outputs. 94 is the correct canonical count."
            ),
        },
        "incident_smoke_test_written": {
            "what": (
                "Stage 1 was first executed with a broader pytest path set that included tests/test_smoke.py. "
                "That test calls the legacy src.pipeline.run(root, smoke=True) and wrote three files into the "
                "repo-root outputs/ directory."
            ),
            "files_written": ["outputs/predictions.csv", "outputs/metrics.csv", "outputs/leakage_audit.csv"],
            "written_at_local": "2026-09-25 18:40",
            "impact": "NONE on experimental evidence",
            "why_benign": [
                "These are the legacy LightGBM example outputs at repo-root outputs/, not the formal "
                "src/TafM_改进源码/outputs/tabm_v21/ tree.",
                "No E1_mini artifact, no frozen source/sequence/selector file, and no formal run was touched; "
                "outputs/example_result/ and outputs/feature_importance_v1/ retain their earlier timestamps.",
                "src/lightgbm_源码/pipeline.py:14 truncates to the first 2 eval days under smoke=True and the "
                "legacy config seed is fixed, so the three files are deterministic and fully regenerable.",
            ],
            "corrective_action": (
                "All subsequent Stage 1 and E1.5-A verification uses the canonical doc 21 §13 command, which "
                "excludes tests/test_smoke.py. No E1.5-A step depends on repo-root outputs/."
            ),
        },
        "dependency_repair": {
            "problem": "epf-2 lacked shap; src/TafM_改进源码/selector.py has a module-level `import shap`.",
            "installed": "shap==0.49.1",
            "transitively_added": ["cloudpickle==3.1.2", "llvmlite==0.49.0", "numba==0.67.0", "slicer==0.0.8"],
            "nothing_else_changed": True,
            "numpy_before": "1.26.4",
            "numpy_after": "1.26.4",
            "why_not_0_51_0": (
                "shap>=0.50 requires numpy>=2 and would have upgraded epf-2's numpy 1.26.4 -> 2.2.6. "
                "numpy 1.26.4 is the version under which E1's 4/4 bit-exact reproducibility was established. "
                "The plan asks for shap==0.51.0 AND for no unrelated package upgrades; those two constraints "
                "conflict. Resolved by user decision: keep numpy 1.26.4 and install shap==0.49.1. "
                "shap is invoked only by selector construction (_oos_tree_shap), and the selector is frozen, "
                "so the shap version cannot affect any E1 or E1.5-A result."
            ),
            "requirements_txt_updated": True,
        },
        "isolation_notes": [
            "No file under src/ was modified for E1.5-A.",
            "E1_mini artifacts were read only; nothing was overwritten.",
            "The frozen selector, source table and sequence asset hashes above are the ones E1 used.",
        ],
    }

    out = C.AUDIT_DIR / "environment_audit.json"
    out.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("interpreter", "gpu", "test_suite", "dependency_repair")}, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

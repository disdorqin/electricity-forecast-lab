"""Verifier/accessor for the statically captured Cycle88 canonical builder evidence."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .contracts import INHERITED_CONTRACT_ID

EVIDENCE_PATH = Path(__file__).with_name("upstream_builder_evidence.json")
UPSTREAM_LEVEL = "VERIFIED_CANONICAL_UPSTREAM"


def load_upstream_feature_evidence(groups: dict[str, list[str]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return feature evidence only if the complete bundle binds the frozen lineage.

    The bundle records exact upstream registry rows and builder-file SHA256s; no
    dependency is promoted from legacy names/source-string heuristics.
    """
    try:
        bundle = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        expected_assets = {
            "slot_table.parquet": INHERITED_CONTRACT_ID["source_sha256"],
            "feature_registry.json": INHERITED_CONTRACT_ID["feature_registry_hash"],
            "feature_groups.json": INHERITED_CONTRACT_ID["feature_groups_hash"],
            "manifest.json": INHERITED_CONTRACT_ID["source_manifest_sha256"],
        }
        if bundle.get("schema") != "cycle88_upstream_builder_evidence_v1":
            raise ValueError("unknown evidence bundle schema")
        if bundle.get("lineage_assets", {}).get("numeric_v2") != expected_assets:
            raise ValueError("numeric_v2 asset lineage does not exactly match frozen_repro")
        if bundle.get("parent_numeric_v1_slot_sha256") != "c306576b56a11d01697eeab2bc8326bb23b75077237a5559cbd41c963fe604aa":
            raise ValueError("numeric_v1 parent slot lineage mismatch")
        if set(bundle.get("lineage", {})) != {
            "frozen_source_snapshot/build_feature_cube.py", "build_numeric_spread_cube.py", "augment_numeric_cube_v2.py"
        } or any(not re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in bundle["lineage"].values()):
            raise ValueError("missing/malformed upstream builder file SHA256")
        rows = bundle.get("features", [])
        if len(rows) != 244 or bundle.get("feature_count") != 244:
            raise ValueError("upstream bundle must bind all 244 F0-F9 features")
        result: dict[str, dict[str, Any]] = {}
        for item in rows:
            name, group = item["feature_name"], item["feature_group"]
            raw = item["upstream_registry_record"]
            contract = item["dependency_contract"]
            if name in result or name not in groups.get(group, []):
                raise ValueError(f"upstream feature/group mismatch: {name}")
            if raw.get("feature") != name or raw.get("group") != group:
                raise ValueError(f"upstream registry row mismatch: {name}")
            if raw.get("source") in (None, "") or raw.get("transform") in (None, ""):
                raise ValueError(f"upstream row lacks source/transform: {name}")
            if contract.get("evidence_basis") != "upstream builder implementation + exact numeric_v2 registry row + feature_groups membership":
                raise ValueError(f"unbound dependency evidence: {name}")
            builder_files = (["augment_numeric_cube_v2.py"] if group == "F9" else
                             ["frozen_source_snapshot/build_feature_cube.py", "build_numeric_spread_cube.py"])
            result[name] = {**contract, "builder_name": " + ".join(builder_files),
                "builder_version": "cycle88_numeric_v2_sha256_bound",
                "dependency_evidence_status": UPSTREAM_LEVEL,
                "dependency_evidence_source": "upstream_builder_evidence.json; exact Cycle88 builder hashes, numeric_v1/v2 lineage assets and upstream feature_registry/feature_groups row",
                "dependency_closure_status": UPSTREAM_LEVEL,
                "builder_file_sha256s": {path: bundle["lineage"][path] for path in builder_files},
                "upstream_lineage_asset_sha256": expected_assets,
                "upstream_registry_record": raw,
            }
        wanted = {name for group, names in groups.items() if group in {f"F{i}" for i in range(10)} for name in names}
        if set(result) != wanted:
            raise ValueError(f"feature evidence inventory differs: missing={sorted(wanted-set(result))[:5]} extra={sorted(set(result)-wanted)[:5]}")
        return result, {"status": "VERIFIED", "feature_count": len(result), "lineage": bundle["lineage"], "lineage_assets": bundle["lineage_assets"]}
    except Exception as exc:
        return {}, {"status": "UNVERIFIED", "reason": f"{type(exc).__name__}: {exc}", "feature_count": 0}

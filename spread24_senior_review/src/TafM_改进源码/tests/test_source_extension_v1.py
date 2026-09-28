import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.TafM_改进源码.contracts import ContractError
from src.TafM_改进源码.source_resolver import resolve_sources, canonical_key_hash
from src.TafM_改进源码.contracts import INHERITED_CONTRACT_ID


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(spreads):
    return pd.DataFrame(
        {
            "target_day": ["2025-01-01"] * len(spreads),
            "时刻": pd.date_range("2025-01-01 01:00", periods=len(spreads), freq="h"),
            "hour_business": list(range(1, len(spreads) + 1)),
            "period": ["1_8"] * len(spreads),
            "target_spread": spreads,
            "target_direction": [int(x > 0) for x in spreads],
        }
    )


def _write_extension(root: Path, extension_id: str, frame: pd.DataFrame, manifest: dict):
    extension_dir = root / extension_id
    extension_dir.mkdir(parents=True)
    data_path = extension_dir / "slot_table_extension.parquet"
    frame.to_parquet(data_path, index=False)
    manifest["sha256"] = _sha(data_path)
    manifest.setdefault("semantic_identity", {
        "canonical_key": ["target_day", "hour_business"],
        "source_schema_hash": INHERITED_CONTRACT_ID["source_schema_hash"],
        "feature_registry_hash": INHERITED_CONTRACT_ID["feature_registry_hash"],
        "feature_groups_hash": INHERITED_CONTRACT_ID["feature_groups_hash"],
        "source_sign_convention": "DA-RT",
    })
    specials = {
        "target_day": {"role": "canonical target date key", "format": "YYYY-MM-DD"},
        "时刻": {"role": "canonical timestamp", "semantics": "target_day plus hour_business"},
        "hour_business": {"role": "canonical hour key", "domain": "1..24"},
        "period": {"role": "inherited market period", "semantics": "frozen source period code"},
        "target_spread": {"role": "native source target", "sign": "DA-RT"},
        "target_direction": {"role": "native target direction", "formula": "1[target_spread>0]"},
    }
    manifest.setdefault("column_semantics", {c: specials[c] for c in frame.columns})
    if manifest.get("type") == "correction":
        keys = [(item["target_day"], item["hour_business"]) for item in manifest["exact_corrected_keys"]]
        def seq_hash(values): return hashlib.sha256(json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        ev = {
            "evidence_type": "UPSTREAM_CANONICAL_REBUILD", "authority": "UPSTREAM_CANONICAL_BUILDER",
            "source_builder_version": manifest["source_builder_version"],
            "closure_start": manifest["dependency_closure_start"], "closure_end": manifest["dependency_closure_end"],
            "recomputed_keys_count": len(keys), "recomputed_keys_sha256": canonical_key_hash(keys),
            "exact_corrected_keys_count": len(keys), "exact_corrected_keys_sha256": canonical_key_hash(keys),
            "affected_features_sha256": seq_hash(manifest["affected_features"]),
            "affected_source_dependency_sha256": seq_hash(manifest["affected_source_dependency"]),
            "recomputed_feature_groups_sha256": seq_hash(manifest["recomputed_feature_groups"]),
            "affected_feature_groups_sha256": hashlib.sha256(json.dumps(manifest["affected_feature_groups"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "no_additional_keys_required": True, "no_additional_keys_reason": "Synthetic fixture asserts one full canonical row replacement.",
            "upstream_rebuild_manifest_sha256": "a" * 64,
        }
        ev["evidence_sha256"] = hashlib.sha256(json.dumps(ev, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest["dependency_closure_evidence"] = ev
    (extension_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return extension_dir


def test_append_and_exact_correction_are_immutable(tmp_path):
    base_path = tmp_path / "base.parquet"
    base = _rows([1.0, -2.0])
    base.to_parquet(base_path, index=False)
    base_hash = _sha(base_path)

    ext_root = tmp_path / "extensions"
    ext_root.mkdir()
    append = pd.DataFrame(
        {
            "target_day": ["2025-01-02"],
            "时刻": [pd.Timestamp("2025-01-02 01:00")],
            "hour_business": [1],
            "period": ["1_8"],
            "target_spread": [3.0],
            "target_direction": [1],
        }
    )
    _write_extension(ext_root, "extension_append", append, {"type": "append", "extension_id": "append-1"})
    correction = _rows([10.0])
    _write_extension(
        ext_root,
        "extension_fix",
        correction,
        {
            "type": "correction",
            "extension_id": "fix-1",
            "supersedes_source_sha256": base_hash,
            "exact_corrected_keys": [{"target_day": "2025-01-01", "hour_business": 1}],
            "reason": "test exact-key correction",
            "correction_type": "full_canonical_row_replacement",
            "root_bad_keys": [{"target_day": "2025-01-01", "hour_business": 1}],
            "affected_features": ["target_spread"],
            "affected_feature_groups": {"target_spread": "F0"},
            "affected_source_dependency": ["target_spread"],
            "dependency_closure_start": "2025-01-01",
            "dependency_closure_end": "2025-01-02",
            "recomputed_keys": [{"target_day": "2025-01-01", "hour_business": 1}],
            "recomputed_feature_groups": ["F0"],
            "source_builder_version": "test_builder_v1",
        },
    )

    result = resolve_sources(base_path, ext_root, expected_base_sha256=None)
    value = result.frame.loc[
        (result.frame.target_day == pd.Timestamp("2025-01-01").date()) & (result.frame.hour_business == 1),
        "target_spread",
    ].item()
    assert value == 10.0
    assert len(result.frame) == 3
    assert len(result.extension_sha256) == 2
    assert _sha(base_path) == base_hash


def test_append_duplicate_key_fails(tmp_path):
    base_path = tmp_path / "base.parquet"
    base = _rows([1.0])
    base.to_parquet(base_path, index=False)
    ext_root = tmp_path / "extensions"
    ext_root.mkdir()
    _write_extension(ext_root, "extension_duplicate", _rows([9.0]), {"type": "append"})
    with pytest.raises(ContractError, match="append duplicates"):
        resolve_sources(base_path, ext_root, expected_base_sha256=None)

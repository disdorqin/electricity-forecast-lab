import pandas as pd
import pytest
from src.TafM_改进源码.contracts import ContractError, INHERITED_CONTRACT_ID
from src.TafM_改进源码.source_resolver import _validate_semantic_compatibility


def test_extension_semantic_identity_requires_schema_and_column_meaning():
    frame = pd.DataFrame({"target_day": [], "hour_business": [], "target_spread": [], "target_direction": []})
    semantic = {"canonical_key": ["target_day", "hour_business"], "source_schema_hash": INHERITED_CONTRACT_ID["source_schema_hash"],
                "feature_registry_hash": INHERITED_CONTRACT_ID["feature_registry_hash"], "feature_groups_hash": INHERITED_CONTRACT_ID["feature_groups_hash"],
                "source_sign_convention": "DA-RT"}
    manifest = {"semantic_identity": semantic, "column_semantics": {
        "target_day": {"role": "canonical target date key", "format": "YYYY-MM-DD"},
        "hour_business": {"role": "canonical hour key", "domain": "1..24"},
        "target_spread": {"role": "native source target", "sign": "DA-RT"},
        "target_direction": {"role": "native target direction", "formula": "1[target_spread>0]"},
    }}
    _validate_semantic_compatibility(frame, manifest, "ok")
    manifest["semantic_identity"] = {**semantic, "source_sign_convention": "RT-DA"}
    with pytest.raises(ContractError, match="semantic_identity"):
        _validate_semantic_compatibility(frame, manifest, "bad")
    manifest["semantic_identity"] = semantic
    manifest["column_semantics"]["target_spread"] = {"role": "native source target", "sign": "RT-DA"}
    with pytest.raises(ContractError, match="same-name column"):
        _validate_semantic_compatibility(frame, manifest, "same-name-wrong-meaning")

from pathlib import Path
from src.TafM_改进源码.contracts import INHERITED_CONTRACT_ID
from src.TafM_改进源码.source_audit import inspect_source
from src.TafM_改进源码.source_resolver import resolve_sources, schema_identity


def test_machine_readable_inherited_source_identity_matches_assets():
    root = Path(__file__).resolve().parents[3]
    resolved = resolve_sources(root / "data/frozen_repro/slot_table.parquet", root / "data/sequence_sources")
    audit, _, registry = inspect_source(resolved, project=root)
    assert audit["status"] == "PASS", audit["errors"]
    identity = registry["source_identity"]
    assert identity == INHERITED_CONTRACT_ID
    assert identity["source_builder_version"] == "UNKNOWN_INHERITED"
    assert audit["feature_contract"]["machine_readable_source_identity"] is True
    schema = audit["feature_contract"]["source_schema_hash_runtime_verification"]
    assert schema["declared_in_doc17"].startswith("b5091d09")
    assert schema["runtime_hash"] == INHERITED_CONTRACT_ID["source_schema_hash"]
    assert schema["status"] == "VERIFIED"


def test_doc17_schema_hash_on_real_frozen_asset():
    root = Path(__file__).resolve().parents[3]
    asset = root / "data" / "frozen_repro" / "slot_table.parquet"
    import pandas as pd
    frame = pd.read_parquet(asset)
    assert schema_identity(frame) == "b5091d096aba5e99e0f42bbb3fdeb39c57376ead572125b287663017707810c6"

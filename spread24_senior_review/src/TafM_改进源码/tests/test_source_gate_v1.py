from pathlib import Path
import hashlib

from src.TafM_改进源码.source_audit import run_source_gate


def test_frozen_source_gate_passes_without_modifying_base(tmp_path):
    root = Path(__file__).resolve().parents[3]
    base = root / "data/frozen_repro/slot_table.parquet"
    before = hashlib.sha256(base.read_bytes()).hexdigest()
    result = run_source_gate(root, tmp_path / "source_gate")
    after = hashlib.sha256(base.read_bytes()).hexdigest()
    assert result["status"] == "PASS", result["errors"]
    assert result["blocking_gaps"] == []
    assert result["base"]["sha256"] == result["base"]["expected_sha256"]
    assert result["base"]["rows"] == 40670
    assert result["base"]["complete_target_through"] == "2026-08-21"
    assert result["base"]["partial_tail_day"] == "2026-08-22"
    assert result["base"]["partial_tail_hours"] == list(range(1, 15))
    assert result["feature_contract"]["candidate_feature_count"] == 259
    assert result["feature_contract"]["dependency_evidence_status"] == "VERIFIED_CANONICAL"
    assert result["feature_contract"]["dependency_evidence_counts"] == {"VERIFIED_CANONICAL_UPSTREAM": 244, "VERIFIED_CANONICAL": 15}
    assert result["feature_contract"]["source_schema_hash_runtime_verification"]["status"] == "VERIFIED"
    assert result["feature_contract"]["future_vintage_evidence_counts"] == {"CONTRACT_INHERITED": 259}
    assert before == after == result["base"]["expected_sha256"]
    context = result["feature_contract"]["d1_context_max_timestamp_audit"]
    assert context["cutoff_mismatch_days"] == []
    assert context["nonmissing_context_max_timestamp"] == "2026-08-21T14:00:00"
    assert (tmp_path / "source_gate/source_audit.json").is_file()
    assert (tmp_path / "source_gate/source_audit.md").is_file()

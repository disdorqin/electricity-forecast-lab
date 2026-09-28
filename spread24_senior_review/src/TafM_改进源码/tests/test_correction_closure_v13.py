import hashlib
import json
import pytest
from src.TafM_改进源码.contracts import ContractError
from src.TafM_改进源码.source_resolver import _validate_correction_closure, canonical_key_hash


def _manifest(*, evidence_type="SYNTHETIC_TEST_VALIDATION", explicit_no_more=False):
    keys = [{"target_day": "2025-01-01", "hour_business": 1}]
    m = {"correction_type": "full_canonical_row_replacement", "root_bad_keys": keys,
         "exact_corrected_keys": keys, "affected_features": ["err_x"], "affected_source_dependency": ["actual_x"],
         "affected_feature_groups": {"err_x": "F5"},
         "dependency_closure_start": "2025-01-01", "dependency_closure_end": "2025-01-03",
         "recomputed_keys": keys, "recomputed_feature_groups": ["F5"], "source_builder_version": "builder-v1",
         "supersedes_source_sha256": "abc", "reason": "synthetic test"}
    def list_hash(xs): return hashlib.sha256(json.dumps(sorted(set(xs)), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    keys_hash = canonical_key_hash([("2025-01-01", 1)])
    ev = {"evidence_type": evidence_type,
          "authority": "UPSTREAM_CANONICAL_BUILDER" if evidence_type == "UPSTREAM_CANONICAL_REBUILD" else "TEST_FIXTURE",
          "source_builder_version": "builder-v1", "closure_start": "2025-01-01", "closure_end": "2025-01-03",
          "recomputed_keys_count": 1, "recomputed_keys_sha256": keys_hash,
          "exact_corrected_keys_count": 1, "exact_corrected_keys_sha256": keys_hash,
          "affected_features_sha256": list_hash(m["affected_features"]),
          "affected_source_dependency_sha256": list_hash(m["affected_source_dependency"]),
          "recomputed_feature_groups_sha256": list_hash(m["recomputed_feature_groups"]),
          "affected_feature_groups_sha256": hashlib.sha256(json.dumps(m["affected_feature_groups"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    if explicit_no_more:
        ev.update(no_additional_keys_required=True, no_additional_keys_reason="Upstream builder proves no other canonical row keys are affected.", upstream_rebuild_manifest_sha256="b" * 64)
    ev["evidence_sha256"] = hashlib.sha256(json.dumps(ev, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    m["dependency_closure_evidence"] = ev
    return m


def test_multi_day_root_only_closure_fails_without_upstream_evidence():
    m = _manifest()
    with pytest.raises(ContractError, match="multi-day closure"):
        _validate_correction_closure(m, "counterexample", actual_keys={("2025-01-01", 1)}, require_upstream=True)


def test_multi_day_root_only_requires_bound_upstream_evidence():
    m = _manifest(evidence_type="UPSTREAM_CANONICAL_REBUILD", explicit_no_more=True)
    assert _validate_correction_closure(m, "synthetic-upstream-attestation", actual_keys={("2025-01-01", 1)}, require_upstream=True) == "UPSTREAM_EVIDENCE_HASH_VALIDATED"
    with pytest.raises(ContractError, match="actual keys"):
        _validate_correction_closure(m, "wrong-extension-keys", actual_keys={("2025-01-02", 1)}, require_upstream=True)


def test_exact_corrected_recomputed_and_actual_keys_must_match():
    m = _manifest()
    m["dependency_closure_end"] = "2025-01-01"
    m["dependency_closure_evidence"]["closure_end"] = "2025-01-01"
    m["dependency_closure_evidence"].pop("evidence_sha256")
    ev = m["dependency_closure_evidence"]
    ev["evidence_sha256"] = hashlib.sha256(json.dumps(ev, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    m["recomputed_keys"] = [{"target_day": "2025-01-01", "hour_business": 1}, {"target_day": "2025-01-01", "hour_business": 2}]
    with pytest.raises(ContractError, match="must equal recomputed_keys"):
        _validate_correction_closure(m, "inconsistent-keys", actual_keys={("2025-01-01", 1)})

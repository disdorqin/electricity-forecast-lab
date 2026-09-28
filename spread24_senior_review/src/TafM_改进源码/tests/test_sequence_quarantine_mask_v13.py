import numpy as np
import pandas as pd
from src.TafM_改进源码.sequence_builder import evaluate_selected_manifest_quarantine


def test_quarantined_and_unaffected_feature_cells_are_selected_manifest_aware():
    days = pd.DataFrame({"sample_index": [0], "target_day": [pd.Timestamp("2026-08-20").date()]})
    ledger = pd.DataFrame({"sample_index": [0] * 24, "feature_name": ["err_28d"] * 24,
                           "dependency_evidence_status": ["INFERRED_LEGACY"] * 24})
    registry = {"candidate_features": [
        {"feature_name": "err_28d", "dependency_evidence_status": "INFERRED_LEGACY"},
        {"feature_name": "forecast_only", "dependency_evidence_status": "INFERRED_LEGACY"},
    ]}
    # Same target day, two feature columns: only the error-dependent cell is masked.
    x_future = np.column_stack([np.arange(24), np.arange(24) + 100.0])
    mask = np.zeros_like(x_future, dtype=bool); mask[:, 0] = True
    assert np.array_equal(x_future[:, 1][~mask[:, 1]], np.arange(24) + 100.0)

    safe = evaluate_selected_manifest_quarantine(days, ledger, {"status": "FROZEN", "selected_features": ["forecast_only"]}, registry)
    assert safe["sample_availability"] == {0: True}
    assert safe["excluded_sample_indices"] == []
    # The sequence sample can be selected by feature mask, but registry evidence still blocks formal training.
    assert safe["formal_training_eligible"] is False
    contaminated = evaluate_selected_manifest_quarantine(days, ledger, {"status": "FROZEN", "selected_features": ["err_28d"]}, registry)
    assert contaminated["sample_availability"] == {0: False}
    assert contaminated["excluded_sample_indices"] == [0]
    deferred = evaluate_selected_manifest_quarantine(days, ledger, None, registry)
    assert deferred["status"] == "DEFERRED_SELECTOR_NOT_FROZEN"
    assert deferred["sample_availability"] is None

import numpy as np
import pytest
from src.TafM_改进源码.metrics import canonical_metrics


def test_canonical_zero_policy_and_magnitude_mae():
    y = np.array([0.0, 2.0, -3.0, 4.0])
    pred = np.array([False, True, False, False])
    metric = canonical_metrics(y, pred, np.array([1.0, 2.0, 2.0, 4.0]), direction_probability=np.array([.2, .8, .3, .4]),
                               month=["2026-01", "2026-01", "2026-02", "2026-02"], hour=[1, 1, 2, 2])
    assert metric["raw_direction_accuracy"] == .75
    assert metric["magnitude_mae"] == .5
    assert metric["brier"] >= 0 and "auc" in metric
    assert set(metric["direction_by_month"]["2026-01"]) >= {"raw_direction_accuracy", "balanced_accuracy", "positive_recall", "nonpositive_recall"}
    assert set(metric["direction_by_hour"]["1"]) >= {"raw_direction_accuracy", "balanced_accuracy", "positive_recall", "nonpositive_recall"}
    assert metric["magnitude_by_true_sign"]["positive"]["count"] == 2
    assert metric["magnitude_by_true_sign"]["nonpositive"]["count"] == 2
    assert metric["auc"] == 1.0


def test_two_class_auc_may_not_silently_be_nan():
    with pytest.raises(ValueError, match="direction_probability is required"):
        canonical_metrics([1.0, -1.0], [True, False], [1.0, 1.0])

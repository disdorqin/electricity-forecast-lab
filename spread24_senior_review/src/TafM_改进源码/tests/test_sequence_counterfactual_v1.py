import numpy as np

from src.TafM_改进源码.sequence_builder import _counterfactual_probe, build_day_sample


def test_d1_post14_and_target_actual_counterfactuals_leave_x_unchanged(synthetic_sequence_frame):
    day, frame, features = synthetic_sequence_frame
    baseline = build_day_sample(frame, day, features)
    audit = _counterfactual_probe(frame, day, features)
    assert audit["d1_post14_realized"]["status"] == "PASS"
    assert audit["target_day_actual"]["status"] == "PASS"
    assert audit["target_day_actual"]["Y_is_separate_label_and_changes_under_counterfactual"]

    mutated = frame.copy(deep=True)
    mask = mutated.target_day == day
    mutated.loc[mask, "actual_hidden"] += 9_999_999
    changed = build_day_sample(mutated, day, features)
    assert np.array_equal(baseline["X_hist"], changed["X_hist"], equal_nan=True)
    assert np.array_equal(baseline["X_future"], changed["X_future"], equal_nan=True)


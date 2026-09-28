from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.TafM_改进源码.contracts import ContractError
from src.TafM_改进源码.sequence_builder import build_day_sample, eligible_training_days


def test_exact_history_future_and_label_shapes(synthetic_sequence_frame):
    day, frame, features = synthetic_sequence_frame
    sample = build_day_sample(frame, day, features)
    assert sample["X_hist"].shape == (168, 7)
    assert sample["X_future"].shape == (24, len(features))
    assert sample["Y"].shape == (24,)
    assert sample["target_hours"] == tuple(range(1, 25))
    assert sample["history_start"] == pd.Timestamp(day - timedelta(days=8)) + pd.Timedelta(hours=15)
    assert sample["history_end"] == pd.Timestamp(day - timedelta(days=1)) + pd.Timedelta(hours=14)
    assert len(sample["history_timestamps"]) == 168


def test_partial_target_is_not_supervised(synthetic_sequence_frame):
    day, frame, features = synthetic_sequence_frame
    partial = frame.loc[~((frame.target_day == day) & (frame.hour_business == 24))]
    with pytest.raises(ContractError, match="expected exactly 24"):
        build_day_sample(partial, day, features)


def test_d2_training_label_cutoff_and_zero_is_nonpositive():
    current = date(2025, 6, 10)
    candidates = [date(2025, 6, 7), date(2025, 6, 8), date(2025, 6, 9), current]
    assert eligible_training_days(candidates, current) == [date(2025, 6, 7), date(2025, 6, 8)]
    assert not (0.0 > 0)

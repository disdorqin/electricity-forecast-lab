import numpy as np
import pandas as pd
import pytest

from src.TafM_改进源码.contracts import F10_FEATURES, ContractError
from src.TafM_改进源码.f10 import build_f10_for_day


def _day():
    h = np.arange(1, 25, dtype=float)
    return pd.DataFrame(
        {
            "hour_business": h.astype(int),
            "target_spread": np.sin(h),
            "target_direction": (np.sin(h) > 0).astype(int),
            "actual_secret": np.arange(24),
            "fcast_直调负荷": 100 + h * 3,
            "fcast_新能源总加": 40 + h * 2,
            "fcast_风电总加": 20 + np.cos(h),
            "fcast_光伏总加": np.maximum(0, np.sin((h - 6) / 24 * 2 * np.pi) * 20),
            "fcast_竞价空间": 30 + np.sin(h / 2) * 4,
            "residual_load_renew": 70 + h,
            "ramp_load": np.cos(h),
            "ramp_renewable": np.sin(h / 2),
            "ramp_residual_load": np.cos(h / 3),
            "ramp_bidding_space": np.sin(h / 4),
            "ramp_solar": np.cos(h / 5),
            "bidding_space_ratio": np.sin(h / 6),
            "ramp_wind": np.cos(h / 7),
            "renewable_share": np.sin(h / 8),
        }
    )


def test_exact_15_columns_deterministic_and_label_actual_independent():
    frame = _day()
    first = build_f10_for_day(frame)
    second = build_f10_for_day(frame.copy())
    assert tuple(first.columns) == F10_FEATURES
    assert first.shape == (24, 15)
    assert np.array_equal(first.to_numpy(), second.to_numpy())
    assert np.isfinite(first.to_numpy()).all()
    assert abs(float(first.load_day_position.median())) < 1e-12
    assert abs(float(first.load_day_position.median())) < 1e-12  # day position is centered within this day

    counterfactual = frame.copy()
    counterfactual["target_spread"] += 1_000_000
    counterfactual["target_direction"] = 1
    counterfactual["actual_secret"] += 1_000_000
    assert np.array_equal(first.to_numpy(), build_f10_for_day(counterfactual).to_numpy())


def test_requires_complete_24_hour_profile_and_finite_inputs():
    with pytest.raises(ContractError, match="exactly one row"):
        build_f10_for_day(_day().iloc[:-1])
    bad = _day()
    bad.loc[0, "fcast_竞价空间"] = np.nan
    with pytest.raises(ContractError, match="finite"):
        build_f10_for_day(bad)


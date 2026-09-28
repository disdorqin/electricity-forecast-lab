from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from src.TafM_改进源码.contracts import TEMPORAL_FEATURES, F10_FEATURES


@pytest.fixture
def synthetic_sequence_frame():
    """One complete target plus exact origin history and forbidden future realized rows."""

    target_day = pd.Timestamp("2025-06-10").date()
    origin = pd.Timestamp(target_day - timedelta(days=1)) + pd.Timedelta(hours=14)
    history_start = origin - pd.Timedelta(hours=167)
    end = pd.Timestamp(target_day) + pd.Timedelta(days=1)
    rows = []
    for i, timestamp in enumerate(pd.date_range(history_start, end, freq="h")):
        hour = timestamp.hour if timestamp.hour else 24
        business_day = timestamp.date() if timestamp.hour else (timestamp - timedelta(days=1)).date()
        spread = float((i % 13) - 6)
        row = {
            "target_day": business_day,
            "时刻": timestamp,
            "hour_business": hour,
            "period": "1_8" if hour <= 8 else "9_16" if hour <= 16 else "17_24",
            "target_spread": spread,
            "target_direction": int(spread > 0),
            "actual_hidden": float(i),
        }
        for name in TEMPORAL_FEATURES:
            if name != "target_spread":
                row[name] = float(i % 17 + 1)
        row.update(
            {
                "fcast_光伏总加": float(i % 9 + 2),
                "fcast_新能源总加": float(i % 7 + 4),
                "ramp_load": float(i % 5 - 2),
                "ramp_renewable": float(i % 3 - 1),
                "ramp_residual_load": float(i % 7 - 3),
                "ramp_bidding_space": float(i % 4 - 2),
                "ramp_solar": float(i % 6 - 3),
                "bidding_space_ratio": float(i % 8 - 2),
                "ramp_wind": float(i % 10 - 5),
                "renewable_share": float(i % 3) / 4,
                "future_dummy": float(i),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    future_features = ["fcast_直调负荷", "fcast_竞价空间", "future_dummy", *F10_FEATURES]
    return target_day, frame, future_features


"""Exact deterministic F10 builder frozen by docs/17 §14.1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import F10_FEATURES, ContractError
from .registry import F10_INPUTS

F10_BUILDER_VERSION = "f10_v1_doc17_14_1"
POSITION_SOURCES = {
    "load_day_position": "fcast_直调负荷",
    "renewable_day_position": "fcast_新能源总加",
    "wind_day_position": "fcast_风电总加",
    "solar_day_position": "fcast_光伏总加",
    "bidding_space_day_position": "fcast_竞价空间",
    "residual_load_day_position": "residual_load_renew",
}


def _profile_values(day_frame: pd.DataFrame, name: str) -> np.ndarray:
    if "hour_business" not in day_frame:
        raise ContractError("F10 input requires hour_business")
    hours = pd.to_numeric(day_frame["hour_business"], errors="raise").astype(int)
    if len(day_frame) != 24 or set(hours) != set(range(1, 25)) or hours.duplicated().any():
        raise ContractError("F10 requires exactly one row for each business hour 1..24")
    if name not in day_frame:
        raise ContractError(f"F10 required input column missing: {name}")
    ordered = day_frame.assign(_hour=hours).sort_values("_hour")
    values = pd.to_numeric(ordered[name], errors="coerce").to_numpy(dtype=np.float64)
    if values.shape != (24,) or not np.isfinite(values).all():
        raise ContractError(f"F10 input {name!r} must be finite for all 24 target hours")
    return values


def den24(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    floor = 1e-6 * max(1.0, float(np.median(np.abs(values))))
    return max(iqr, floor)


def rpos24(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - float(np.median(values))) / den24(values)


def build_f10_for_day(day_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the 15 F10 values in canonical row order (business hour 1..24).

    Missing/non-finite inputs fail closed. No imputation, label, actual, or
    cross-day statistics are used in this deterministic feature builder.
    """

    hours = pd.to_numeric(day_frame.get("hour_business"), errors="raise").astype(int)
    if len(day_frame) != 24 or set(hours) != set(range(1, 25)) or hours.duplicated().any():
        raise ContractError("F10 supervised build requires exactly one row for each hour 1..24")
    ordered = day_frame.assign(_hour=hours).sort_values("_hour")

    positions: dict[str, np.ndarray] = {}
    for output_name, input_name in POSITION_SOURCES.items():
        positions[output_name] = rpos24(_profile_values(ordered, input_name))

    data = {
        **positions,
        "load_space_pressure": positions["load_day_position"] - positions["bidding_space_day_position"],
        "residual_space_pressure": positions["residual_load_day_position"] - positions["bidding_space_day_position"],
        "renewable_space_balance": positions["renewable_day_position"] - positions["bidding_space_day_position"],
    }

    ramp_inputs = {}
    for name in ("ramp_load", "ramp_renewable", "ramp_residual_load", "ramp_bidding_space", "ramp_solar"):
        ramp_inputs[name] = _profile_values(ordered, name)
    data.update(
        {
            "net_ramp_pressure": ramp_inputs["ramp_load"] - ramp_inputs["ramp_renewable"],
            "ramp_tightness": ramp_inputs["ramp_residual_load"] - ramp_inputs["ramp_bidding_space"],
            "solar_drop_load_rise": np.maximum(0.0, -ramp_inputs["ramp_solar"])
            * np.maximum(0.0, ramp_inputs["ramp_load"]),
            "load_x_space": positions["load_day_position"] * positions["bidding_space_day_position"],
        }
    )

    ratio = rpos24(_profile_values(ordered, "bidding_space_ratio"))
    ramp_wind = rpos24(_profile_values(ordered, "ramp_wind"))
    renewable_share = rpos24(_profile_values(ordered, "renewable_share"))
    data["space_ratio_x_ramp_wind"] = ratio * ramp_wind
    data["residual_x_renewable_share"] = positions["residual_load_day_position"] * renewable_share

    result = pd.DataFrame({name: data[name] for name in F10_FEATURES}, index=ordered.index)
    if tuple(result.columns) != F10_FEATURES:
        raise AssertionError("internal F10 column order differs from canonical contract")
    if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
        raise ContractError("F10 builder produced non-finite values")
    return result


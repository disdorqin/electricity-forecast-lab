"""Canonical data and information-boundary contracts from docs/17."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

BASE_SHA256 = "a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea"
BASE_ROWS = 40_670
COMPLETE_TARGET_THROUGH = date(2026, 8, 21)
PARTIAL_TAIL_DAY = date(2026, 8, 22)
PARTIAL_TAIL_HOURS = tuple(range(1, 15))
# Quarantine is represented at feature/cell granularity in registry.py.  There
# is deliberately no global target-day boundary.

FORECAST_ORIGIN = "D-1 14:00"
HISTORY_HOURS = 168
HORIZON = 24
LABEL_CUTOFF_RULE = "target_day <= current_target_day - 2 calendar days"
ZERO_POLICY = "nonpositive: positive iff spread > 0; zero is non-positive"
FUTURE_VINTAGE_EVIDENCE = "CONTRACT_INHERITED"
INHERITED_CONTRACT_ID = {
    "contract_name": "DOC17_V1.3_FROZEN_FORECAST_CONTRACT",
    "source_asset": "data/frozen_repro/slot_table.parquet",
    "source_sha256": BASE_SHA256,
    "source_manifest_path": "data/frozen_repro/manifest.json",
    "source_manifest_sha256": "8a68e6b8fec74fa45b1b6b467cfccae63a9a484e3c0811a0cf7beb825a2abcd1",
    "source_schema_hash": "b5091d096aba5e99e0f42bbb3fdeb39c57376ead572125b287663017707810c6",
    "feature_registry_hash": "f06cfc1cb83bea08cec86fe4ed29431d18853fb5648e6b51f5eac49c0720b825",
    "feature_groups_hash": "efc92c22ddbc84d1248f639e6364152ec9a89c55aae6483ff1abd3109e42d9e2",
    "source_builder_version": "UNKNOWN_INHERITED",
}

FEATURE_GROUP_COUNTS = {
    "F0": 11,
    "F1": 12,
    "F2": 10,
    "F3": 11,
    "F4": 12,
    "F5": 82,
    "F6": 40,
    "F7": 24,
    "F8": 14,
    "F9": 28,
}
F10_FEATURES = (
    "load_day_position",
    "renewable_day_position",
    "wind_day_position",
    "solar_day_position",
    "bidding_space_day_position",
    "residual_load_day_position",
    "load_space_pressure",
    "residual_space_pressure",
    "renewable_space_balance",
    "net_ramp_pressure",
    "ramp_tightness",
    "solar_drop_load_rise",
    "load_x_space",
    "space_ratio_x_ramp_wind",
    "residual_x_renewable_share",
)

TEMPORAL_FEATURES = (
    "target_spread",
    "fcast_直调负荷",
    "fcast_竞价空间",
    "residual_load_renew",
    "fcast_风电总加",
    "fcast_光伏总加",
    "fcast_新能源总加",
)
TEMPORAL_GROUPS = {
    "price_context": ("target_spread",),
    "demand_tightness": (
        "fcast_直调负荷",
        "fcast_竞价空间",
        "residual_load_renew",
    ),
    "renewable": (
        "fcast_风电总加",
        "fcast_光伏总加",
        "fcast_新能源总加",
    ),
}

REQUIRED_SOURCE_COLUMNS = (
    "target_day",
    "时刻",
    "hour_business",
    "period",
    "target_spread",
    "target_direction",
)
SOURCE_KEY = ("target_day", "hour_business")


class ContractError(ValueError):
    """Raised when source data cannot satisfy the frozen canonical contract."""


def project_root() -> Path:
    """Return the repository root without relying on an absolute machine path."""

    return Path(__file__).resolve().parents[2]


def training_last_day(target_day: date) -> date:
    """Latest complete label day at forecast origin D-1 14:00."""

    return target_day - timedelta(days=2)


def is_quarantined(target_day: date) -> bool:
    """Deprecated compatibility helper: date-only quarantine is forbidden."""
    return False

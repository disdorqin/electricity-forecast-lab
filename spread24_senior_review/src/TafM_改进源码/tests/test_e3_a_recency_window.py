"""E3-A recency-aware Stage-A split: canonical 80/20 vs explicit monitor/history windows."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.contracts import ContractError
from src.TafM_改进源码.train import stage_a_split_indices, train_target_day


def idx(n):
    return np.arange(n, dtype=np.int64)


def test_default_none_none_is_the_canonical_80_20_split():
    base, monitor, mode = stage_a_split_indices(idx(1000))
    assert mode == "canonical_percent80"
    assert len(base) == 800 and len(monitor) == 200
    assert np.array_equal(base, idx(1000)[:800]) and np.array_equal(monitor, idx(1000)[800:])


def test_expanding_split_monitor_is_newest_and_base_ends_immediately_before():
    base, monitor, mode = stage_a_split_indices(idx(1000), monitor_days=60)
    assert mode == "explicit_monitor_expanding"
    assert np.array_equal(monitor, idx(1000)[-60:])
    assert np.array_equal(base, idx(1000)[:940])
    assert len(base) == 940
    assert base[-1] + 1 == monitor[0]  # adjacent, no overlap


def test_rolling_split_exact_counts():
    base, monitor, mode = stage_a_split_indices(idx(1000), monitor_days=60, history_window_days=365)
    assert mode == "explicit_monitor_rolling"
    assert len(base) == 365 and len(monitor) == 60
    assert np.array_equal(base, idx(1000)[940 - 365:940])
    base2, monitor2, _ = stage_a_split_indices(idx(2000), monitor_days=60, history_window_days=1095)
    assert len(base2) == 1095 and len(monitor2) == 60
    assert np.array_equal(base2, idx(2000)[1940 - 1095:1940])


def test_no_overlap_target_excluded_and_chronology_preserved():
    base, monitor, _ = stage_a_split_indices(idx(1500), monitor_days=60, history_window_days=365)
    assert set(base).isdisjoint(set(monitor))
    # union is exactly the newest (base+monitor) eligible days; nothing older than the window is used
    assert np.array_equal(np.sort(np.concatenate([base, monitor])), idx(1500)[1075:1500])
    assert np.all(base[:-1] < base[1:]) and np.all(monitor[:-1] < monitor[1:])
    assert base.max() < monitor.min()
    assert monitor[-1] == 1499  # newest eligible day is the monitor end (D-2)


def test_history_window_requires_monitor_days():
    with pytest.raises(ContractError):
        stage_a_split_indices(idx(1000), history_window_days=365)


def test_monitor_days_must_be_smaller_than_history():
    with pytest.raises(ContractError):
        stage_a_split_indices(idx(60), monitor_days=60)
    with pytest.raises(ContractError):
        stage_a_split_indices(idx(1000), monitor_days=0)


def test_history_window_must_fit_candidate_base():
    with pytest.raises(ContractError):
        stage_a_split_indices(idx(1000), monitor_days=60, history_window_days=1000)


def test_train_target_day_defaults_and_fail_closed():
    sig = inspect.signature(train_target_day).parameters
    assert sig["stage_a_monitor_days"].default is None
    assert sig["stage_a_history_window_days"].default is None
    common = dict(objective_mode="dir_only", train_mode="stage_a", architecture_mode="full_current",
                  direction_tabular_mode="current", direction_horizon_gate_mode="current",
                  strong_role_profile="all", numeric_encoding_mode="canonical", direction_fusion_alpha=0.8)
    # explicit split is only valid on the E3-A route (segment_heads)
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", direction_readout_mode="shared", stage_a_monitor_days=60, **common)
    # and only with the fixed alpha=.8
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", direction_readout_mode="segment_heads", stage_a_monitor_days=60,
                         **{**common, "direction_fusion_alpha": 0.5})

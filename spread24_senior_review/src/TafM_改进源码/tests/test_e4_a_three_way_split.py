"""E4-A: three-way chronological split (BASE / CHECKPOINT_MONITOR / CALIBRATOR) wiring."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.contracts import ContractError
from src.TafM_改进源码.direction_postprocess import (
    LogisticStacker, REGIME_FEATURES, _safe_logit, build_meta_features,
)
from src.TafM_改进源码.train import (
    E4_CALIBRATOR_DAYS, E4_CHECKPOINT_MONITOR_MIN_DAYS, e4a_three_way_split, train_target_day,
)


def test_three_way_split_exact_counts_and_adjacency():
    monitor = np.arange(300, dtype=np.int64)
    checkpoint, calibrator = e4a_three_way_split(monitor)
    assert len(calibrator) == E4_CALIBRATOR_DAYS == 90
    assert len(checkpoint) == 300 - 90 == E4_CHECKPOINT_MONITOR_MIN_DAYS + 150
    assert np.array_equal(calibrator, monitor[-90:])
    assert np.array_equal(checkpoint, monitor[:-90])
    assert checkpoint[-1] + 1 == calibrator[0]          # chronological adjacency
    assert set(checkpoint).isdisjoint(set(calibrator))  # no overlap
    assert calibrator[-1] == 299                        # newest eligible day ends the calibrator


def test_three_way_split_minimum_and_guards():
    # FULL_MONITOR must hold calibrator(90) + checkpoint_monitor(>=60)
    e4a_three_way_split(np.arange(150, dtype=np.int64))
    with pytest.raises(ContractError):
        e4a_three_way_split(np.arange(149, dtype=np.int64))
    with pytest.raises(ContractError):
        e4a_three_way_split(np.arange(300, dtype=np.int64), calibrator_days=0)


def test_three_way_split_never_touches_base():
    # BASE is the older 80%; the carve only partitions the canonical monitor
    monitor = np.arange(1200, 1500, dtype=np.int64)
    checkpoint, calibrator = e4a_three_way_split(monitor)
    assert checkpoint[0] == 1200 and calibrator[-1] == 1499


def test_meta_feature_anchor_is_the_base_logit_and_counts_are_5_and_14():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, size=(6, 24))
    names = list(REGIME_FEATURES)
    x = rng.normal(size=(6, 24, len(names)))
    X5, c5 = build_meta_features(p, x, names, "segment_logit")
    assert c5 == ["base_logit", "H2", "H3", "base_logit_x_H2", "base_logit_x_H3"] and len(c5) == 5
    assert np.allclose(X5[:, 0], _safe_logit(p).reshape(-1))
    X14, c14 = build_meta_features(p, x, names, "regime_logit")
    assert len(c14) == 5 + len(REGIME_FEATURES) == 14
    assert c14[:5] == c5 and c14[5:] == names


def test_identity_anchor_weights_reproduce_the_base_probability():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, size=(4, 24))
    names = list(REGIME_FEATURES)
    x = rng.normal(size=(4, 24, len(names)))
    stacker = LogisticStacker(mode="segment_logit", feature_names=["base_logit", "H2", "H3",
                            "base_logit_x_H2", "base_logit_x_H3"],
                            weight=np.array([1.0, 0, 0, 0, 0]), bias=0.0, l2=1e-3, iterations=0,
                            monitor_positive_rate=0.5, monitor_base_raw=0.5, monitor_post_raw=0.5)
    assert np.allclose(stacker.predict_proba(p, x, names), p, atol=1e-6)


def test_train_target_day_defaults_and_fail_closed():
    sig = inspect.signature(train_target_day).parameters
    assert sig["direction_postprocess_mode"].default == "none"
    assert sig["e4_three_way_split"].default is False
    common = dict(objective_mode="dir_only", train_mode="stage_a", architecture_mode="full_current",
                  direction_tabular_mode="current", direction_horizon_gate_mode="current",
                  strong_role_profile="all", numeric_encoding_mode="canonical",
                  direction_readout_mode="segment_heads", direction_fusion_alpha=0.8)
    # postprocessing without the E4-A split is refused (prevents double-dipping the checkpoint monitor)
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", direction_postprocess_mode="segment_logit", **common)
    # the E4-A split itself is only valid on the canonical Q2 route
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", e4_three_way_split=True,
                         **{**common, "direction_readout_mode": "shared"})
    # and not together with an explicit recency split
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", e4_three_way_split=True, stage_a_monitor_days=60, **common)

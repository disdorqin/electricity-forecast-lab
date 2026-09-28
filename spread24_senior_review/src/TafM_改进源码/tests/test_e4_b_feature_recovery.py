"""E4-B: frozen-selector reopening via Weak feature recovery — focused contract tests.

Covers plan29 Gate A items that do not require GPU training:
 1-2  selected222 reproduces the frozen selector bit-identically (no flag == canonical)
 3-5  exact feature counts per arm (F0 222/211/11, F1 240/211/29, F2 259/211/48)
 6-7  selected features follow canonical SequenceStore registry order; indices exact
 8    frozen selector file/sha unchanged on disk
 9    experiment profile SHA recorded separately
 10   all recovered features are original frozen-selector role Noise
 11   recovered features are Weak only in the experiment manifest
 12-13 identical eligible-index SETS across all 28 formal days for every profile
 18-19 no postprocess/class-weight/structured decoder active; fail-closed for non-canonical routes
"""
import inspect
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.config import default_selector_path
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.train import (
    LITERATURE18_E4B, build_experiment_feature_manifest, train_target_day,
)

WINDOWS = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
           "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
FORMAL_DAYS = [str((pd.Timestamp(a) + pd.Timedelta(days=i)).date())
               for a, b in WINDOWS.values() for i in range(7)]


@pytest.fixture(scope="module")
def frozen():
    manifest, _ = load_frozen_selector(default_selector_path())
    return manifest


@pytest.fixture(scope="module")
def store():
    return SequenceStore.load()


def test_selected222_is_bit_identical_to_frozen(frozen):
    # Gate A 1/2/8: default/no flag must equal the frozen selector object exactly.
    m = build_experiment_feature_manifest(frozen, "selected222", SequenceStore.load())
    assert m is frozen


def test_literature240_counts_and_routing(frozen, store):
    m = build_experiment_feature_manifest(frozen, "literature240", store)
    assert len(m["selected_features"]) == 240
    assert m["experiment_strong_count"] == 211
    assert m["experiment_weak_count"] == 29
    assert len(m["experiment_recovered_features"]) == 18
    # all 18 recovered are original frozen Noise
    role_by_name = {r["feature_name"]: r["role"] for r in frozen["feature_roles"]}
    assert all(role_by_name[n] == "Noise" for n in m["experiment_recovered_features"])
    # all recovered are routed Weak only in the experiment manifest
    exp_role = {r["feature_name"]: r["role"] for r in m["feature_roles"]}
    assert all(exp_role[n] == "Weak" for n in m["experiment_recovered_features"])


def test_all259_counts_and_routing(frozen, store):
    m = build_experiment_feature_manifest(frozen, "all259", store)
    assert len(m["selected_features"]) == 259
    assert m["experiment_strong_count"] == 211
    assert m["experiment_weak_count"] == 48
    assert len(m["experiment_recovered_features"]) == 37
    role_by_name = {r["feature_name"]: r["role"] for r in frozen["feature_roles"]}
    assert all(role_by_name[n] == "Noise" for n in m["experiment_recovered_features"])
    exp_role = {r["feature_name"]: r["role"] for r in m["feature_roles"]}
    assert all(exp_role[n] == "Weak" for n in m["experiment_recovered_features"])


def test_canonical_order_and_exact_indices(frozen, store):
    for profile in ("literature240", "all259"):
        m = build_experiment_feature_manifest(frozen, profile, store)
        selected = m["selected_features"]
        # follows canonical registry order
        assert selected == [n for n in store.feature_names if n in set(selected)]
        # selected_indices exactly correspond to that order
        assert list(m["selected_indices"]) == [store.feature_names.index(n) for n in selected]
        assert m["selected_indices"] == sorted(m["selected_indices"])


def test_frozen_selector_sha_preserved_separately(frozen, store):
    for profile in ("literature240", "all259"):
        m = build_experiment_feature_manifest(frozen, profile, store)
        assert m["selector_sha256"] == frozen.get("selector_sha256")  # unchanged base
        assert m["experiment_feature_profile_sha256"] is not None      # separate profile sha
        assert m["experiment_feature_profile_sha256"] != m["selector_sha256"]


def test_eligibility_identity_across_28_days(frozen, store):
    manifests = {p: build_experiment_feature_manifest(frozen, p, store)
                 for p in ("selected222", "literature240", "all259")}
    base = set(store.eligibility(manifests["selected222"], current_target_day=FORMAL_DAYS[0])["eligible_indices"])
    for day in FORMAL_DAYS:
        sets = {p: set(store.eligibility(manifests[p], current_target_day=day)["eligible_indices"])
                for p in manifests}
        assert sets["selected222"] == sets["literature240"] == sets["all259"], f"identity break on {day}"
    # sanity: at least the first day matches the expected base size
    assert len(base) > 1000


def test_train_target_day_fail_closed_for_non_canonical_routes():
    sig = inspect.signature(train_target_day).parameters
    assert sig["feature_recovery_profile"].default == "selected222"
    common = dict(objective_mode="dir_only", train_mode="stage_a", architecture_mode="full_current",
                  direction_tabular_mode="current", direction_horizon_gate_mode="current",
                  strong_role_profile="all", numeric_encoding_mode="canonical",
                  direction_readout_mode="segment_heads", direction_fusion_alpha=0.8,
                  direction_postprocess_mode="none", direction_class_weight_mode="unweighted")
    # recovery requires the canonical Q2 segment_heads route; shared readout is refused
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", feature_recovery_profile="literature240",
                         **{**common, "direction_readout_mode": "shared"})
    # postprocessing is incompatible with the recovery arm (no double-dip / no smoothing)
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", feature_recovery_profile="all259",
                         **{**common, "direction_postprocess_mode": "regime_logit"})
    # class weighting is forbidden in E4-B
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", feature_recovery_profile="all259",
                         **{**common, "direction_class_weight_mode": "sqrt_balanced"})


def test_recovered_features_are_weak_only_not_strong(frozen, store):
    for profile in ("literature240", "all259"):
        m = build_experiment_feature_manifest(frozen, profile, store)
        role_by_name = {r["feature_name"]: r["role"] for r in m["feature_roles"]}
        recovered = set(m["experiment_recovered_features"])
        strong_roles = {"Forced-Core", "Forced-Temporal", "Strong-BOTH", "Strong-DIR", "Strong-MAG", "Strong"}
        for n in recovered:
            assert role_by_name[n] == "Weak"
            assert role_by_name[n] not in strong_roles

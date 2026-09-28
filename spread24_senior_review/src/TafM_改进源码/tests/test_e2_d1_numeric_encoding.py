"""E2-D1 numeric-encoding family: canonical / raw_only / ple_only."""
import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.config import V2Config
from src.TafM_改进源码.models.tabular_encoder import TabularEncoder
from src.TafM_改进源码.train import train_target_day


NAMES = ["strong_a", "strong_b", "weak_a"]
ROLES = [{"feature_name": "strong_a", "role": "Strong-DIR"},
         {"feature_name": "strong_b", "role": "Forced-Core"},
         {"feature_name": "weak_a", "role": "Weak"}]
BINS = [[-1.0, 0.0, 1.0] for _ in NAMES]


def enc(**kwargs):
    base = dict(feature_names=NAMES, feature_roles=ROLES, ple_bins=BINS,
                d_tab=8, k=2, n_blocks=1, dropout=0.0, ple_embedding_dim=2)
    base.update(kwargs)
    return TabularEncoder(**base)


def test_canonical_default_matches_frozen_semantics_and_dims():
    on = enc(ple_enabled=True)
    assert on.numeric_encoding_mode == "canonical"
    assert on.ple_enabled is True and on.use_raw is True
    assert on.encoded_feature_dim == on._ple_embedding_dim + 1 == 3
    off = enc(ple_enabled=False)
    assert off.ple is None and off.encoded_feature_dim == 1


def test_canonical_encoded_is_exactly_ple_concat_raw():
    torch.manual_seed(7)
    model = enc(ple_enabled=True)
    x = torch.randn(2, 24, 3)
    flat = x.reshape(48, 3)
    ref = torch.cat([model.ple(flat), flat.unsqueeze(-1)], dim=-1)
    got = model._encoded(x)
    assert torch.equal(got, ref)
    assert got.shape == (48, 3, 3)


def test_raw_only_dim_one_and_no_ple_reaches_either_branch():
    torch.manual_seed(11)
    model = enc(ple_enabled=True, numeric_encoding_mode="raw_only")
    assert model.numeric_encoding_mode == "raw_only"
    assert model.ple is None and model.ple_enabled is False and model.use_raw is True
    assert model.encoded_feature_dim == 1
    x = torch.randn(2, 24, 3)
    got = model._encoded(x)
    assert got.shape == (48, 3, 1)
    assert torch.equal(got.squeeze(-1), x.reshape(48, 3))
    assert model.encoding_audit()["strong_tabm_input_dim"] == 2


def test_ple_only_dim_eight_drops_raw_and_requires_ple_enabled():
    torch.manual_seed(13)
    model = enc(ple_enabled=True, numeric_encoding_mode="ple_only")
    assert model.ple is not None and model.use_raw is False
    assert model.encoded_feature_dim == model._ple_embedding_dim == 2
    x = torch.randn(2, 24, 3)
    got = model._encoded(x)
    assert torch.equal(got, model.ple(x.reshape(48, 3)))
    assert got.shape == (48, 3, 2)
    with pytest.raises(ValueError):
        enc(ple_enabled=False, numeric_encoding_mode="ple_only")


def test_invalid_numeric_encoding_mode_fails_closed():
    with pytest.raises(ValueError):
        enc(numeric_encoding_mode="raw_plus_ple")


def test_encoding_audit_reports_the_three_families():
    assert enc(numeric_encoding_mode="canonical").encoding_audit()["encoded_feature_dim"] == 3
    assert enc(numeric_encoding_mode="raw_only").encoding_audit()["encoded_feature_dim"] == 1
    assert enc(numeric_encoding_mode="ple_only").encoding_audit()["encoded_feature_dim"] == 2


def test_train_target_day_default_is_canonical():
    assert inspect.signature(train_target_day).parameters["numeric_encoding_mode"].default == "canonical"


def test_non_canonical_modes_are_fail_closed_before_any_data_load():
    common = dict(architecture_mode="full_current", direction_tabular_mode="current",
                  direction_horizon_gate_mode="current", strong_role_profile="all",
                  direction_fusion_alpha=0.8)
    # wrong objective
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", objective_mode="joint_v21", train_mode="stage_a",
                         numeric_encoding_mode="raw_only", **common)
    # missing fixed alpha
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", objective_mode="dir_only", train_mode="stage_a",
                         numeric_encoding_mode="raw_only",
                         architecture_mode="full_current", direction_tabular_mode="current",
                         direction_horizon_gate_mode="current", strong_role_profile="all",
                         direction_fusion_alpha=None)
    # non-stage_a
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", objective_mode="dir_only", train_mode="stage_ab",
                         numeric_encoding_mode="raw_only", **common)
    # ple_only with ple disabled
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", objective_mode="dir_only", train_mode="stage_a",
                         numeric_encoding_mode="ple_only", config=V2Config(ple_enabled=False), **common)


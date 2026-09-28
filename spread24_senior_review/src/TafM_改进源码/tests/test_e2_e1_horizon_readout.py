"""E2-E1 Direction readout family: shared / segment_bias / segment_heads."""
import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.models.tabular_encoder import TabularEncoder
from src.TafM_改进源码.models.temporal_encoder import TemporalEncoder
from src.TafM_改进源码.models.dual_branch_v21 import DualBranchV21, SEGMENT_SLICES
from src.TafM_改进源码.train import train_target_day

NAMES = ["s1", "s2", "w1"]
ROLES = [{"feature_name": "s1", "role": "Strong-DIR"},
         {"feature_name": "s2", "role": "Forced-Core"},
         {"feature_name": "w1", "role": "Weak"}]
BINS = [[-1.0, 0.0, 1.0] for _ in NAMES]


def make_model(mode="shared", k=2, d_task=4):
    tab = TabularEncoder(feature_names=NAMES, feature_roles=ROLES, ple_bins=BINS, d_tab=8, k=k,
                         n_blocks=1, dropout=0.0, ple_embedding_dim=2, ple_enabled=True)
    temporal = TemporalEncoder(selected_feature_names=NAMES, d_time=4, hidden=4, fft_bins=[1],
                               fft_enabled=False, future_conditioning=False, temporal_clip_abs=10.0)
    return DualBranchV21(8, 4, d_task, mode="A2", tabular_encoder=tab, temporal_encoder=temporal,
                         k=k, magnitude_scale=1.0, direction_fusion_alpha=0.8,
                         direction_readout_mode=mode)


def _forward(model, seed=7):
    torch.manual_seed(seed)
    xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, 3)
    return model(xh, xf)


def test_shared_readout_is_the_canonical_single_head():
    torch.manual_seed(3)
    model = make_model("shared")
    assert model.direction_readout_mode == "shared"
    assert model.direction_head is not None and model.direction_hour_bias is None
    assert model.direction_segment_heads is None
    out = _forward(model)
    assert out["z_members"].shape == (2, 2, 24)
    tab = model.tabular_encoder
    xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, 3)
    _, _, _, h_tab = tab.forward_components(xf)
    h_time = model.temporal_encoder(xh, xf)
    h_dir, _ = model.adapters(h_tab, h_time)
    b, k, h, d = h_dir.shape
    ref = model.direction_head(h_dir.permute(0, 2, 1, 3).reshape(b * h, k, d)).squeeze(-1).reshape(b, h, k).permute(0, 2, 1)
    assert torch.equal(model(xh, xf)["z_members"], ref)


def test_segment_bias_has_exactly_three_zero_init_scalars_and_one_shared_head():
    model = make_model("segment_bias")
    assert model.direction_head is not None
    assert model.direction_hour_bias is not None and model.direction_hour_bias.shape == (3,)
    assert torch.equal(model.direction_hour_bias, torch.zeros(3))
    assert model.direction_segment_heads is None
    audit = model.direction_readout_audit()
    assert audit["n_segment_bias"] == 3 and audit["n_segment_heads"] == 0
    assert audit["shared_direction_head"] is True and audit["segment_bias"] == [0.0, 0.0, 0.0]


def test_segment_bias_adds_only_per_segment_logit_offsets_before_sigmoid():
    torch.manual_seed(5)
    torch.manual_seed(5)
    base = make_model("shared")
    torch.manual_seed(5)
    biased = make_model("segment_bias")
    biased.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        biased.direction_hour_bias.copy_(torch.tensor([1.0, -2.0, 0.5]))
    xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, 3)
    z0 = base(xh, xf)["z_members"]
    out = biased(xh, xf)
    z = out["z_members"]
    for (lo, hi), b in zip(SEGMENT_SLICES, (1.0, -2.0, 0.5)):
        assert torch.allclose(z[:, :, lo:hi], z0[:, :, lo:hi] + b, atol=1e-6)
    assert torch.allclose(out["p_members"], torch.sigmoid(z), atol=1e-6)
    assert torch.allclose(out["p"], out["p_members"].mean(1), atol=1e-6)


def test_segment_heads_has_exactly_three_independent_heads_and_no_shared_head():
    model = make_model("segment_heads")
    assert model.direction_head is None
    assert model.direction_segment_heads is not None and len(model.direction_segment_heads) == 3
    assert isinstance(model.direction_segment_heads, torch.nn.ModuleList)
    audit = model.direction_readout_audit()
    assert audit["n_segment_heads"] == 3 and audit["n_segment_bias"] == 0
    assert audit["shared_direction_head"] is False
    assert len(audit["segment_head_param_norms"]) == 3 and len(audit["segment_head_param_counts"]) == 3


def test_segment_heads_each_hour_is_produced_by_exactly_one_head():
    torch.manual_seed(9)
    model = make_model("segment_heads")
    xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, 3)
    z_base = model(xh, xf)["z_members"].detach()
    for i, (lo, hi) in enumerate(SEGMENT_SLICES):
        clone = make_model("segment_heads")
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            for p in clone.direction_segment_heads[i].parameters():
                p.add_(0.5)
        z = clone(xh, xf)["z_members"].detach()
        assert not torch.allclose(z[:, :, lo:hi], z_base[:, :, lo:hi]), f"segment {i} not affected"
        outside = [slice(0, lo), slice(hi, 24)]
        for sl in outside:
            if sl.stop > sl.start:
                assert torch.equal(z[:, :, sl], z_base[:, :, sl]), f"segment {i} leaked into hours {sl}"


def test_readout_preserves_member_axis_threshold_and_magnitude():
    for mode in ("shared", "segment_bias", "segment_heads"):
        torch.manual_seed(11)
        model = make_model(mode, k=8)
        out = _forward(model)
        assert out["z_members"].shape == (2, 8, 24)
        assert out["p_members"].shape == (2, 8, 24)
        assert torch.equal(out["direction_hat"], (out["p"] >= 0.5))
        assert torch.allclose(out["p"], out["p_members"].mean(1), atol=1e-6)
        assert out["magnitude_members"].shape == (2, 8, 24)
        assert isinstance(model.magnitude_head, torch.nn.Module)


def test_magnitude_head_structure_is_mode_invariant():
    counts = {m: sum(p.numel() for p in make_model(m).magnitude_head.parameters())
              for m in ("shared", "segment_bias", "segment_heads")}
    assert len(set(counts.values())) == 1


def test_train_target_day_default_is_shared():
    assert inspect.signature(train_target_day).parameters["direction_readout_mode"].default == "shared"


def test_non_shared_readout_is_fail_closed_before_any_data_load():
    common = dict(architecture_mode="full_current", direction_tabular_mode="current",
                  direction_horizon_gate_mode="current", strong_role_profile="all",
                  numeric_encoding_mode="canonical", direction_fusion_alpha=0.8)
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", objective_mode="joint_v21", train_mode="stage_a",
                         direction_readout_mode="segment_heads", **common)
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", objective_mode="dir_only", train_mode="stage_a",
                         direction_readout_mode="segment_heads",
                         architecture_mode="full_current", direction_tabular_mode="current",
                         direction_horizon_gate_mode="current", strong_role_profile="all",
                         numeric_encoding_mode="canonical", direction_fusion_alpha=None)
    with pytest.raises(ValueError):
        train_target_day("2026-02-13", objective_mode="dir_only", train_mode="stage_a",
                         direction_readout_mode="segment_bias",
                         architecture_mode="full_current", direction_tabular_mode="current",
                         direction_horizon_gate_mode="current", strong_role_profile="all",
                         numeric_encoding_mode="raw_only", direction_fusion_alpha=0.8)
    with pytest.raises(ValueError):
        make_model("not_a_mode")

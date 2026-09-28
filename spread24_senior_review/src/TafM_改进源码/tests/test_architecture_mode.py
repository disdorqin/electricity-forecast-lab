"""E2-A Gate A — experiment-only `architecture_mode` big-block ablation contract.

The three modes must differ *only* in how the Direction representation is formed:

    full_current   canonical alpha fusion (default; must be bit-identical to today's model)
    tabular_only   Direction reads H_tab only  -> temporal branch cannot reach Direction
    temporal_only  Direction reads H_time only -> tabular branch cannot reach Direction

The Magnitude fusion, the parameter set and the TabM member axis are mode-invariant, so
A0/A1/A2 stay directly comparable and A0 reproduces the frozen canonical path exactly.
"""
import inspect

import pytest
import torch

from src.TafM_改进源码.contracts import TEMPORAL_FEATURES
from src.TafM_改进源码.losses import v21_loss
from src.TafM_改进源码.models.dual_branch_v21 import DualBranchV21
from src.TafM_改进源码.models.tabular_encoder import TabularEncoder
from src.TafM_改进源码.models.task_adapters import ARCHITECTURE_MODES, MemberWiseTaskAdaptersV21
from src.TafM_改进源码.models.temporal_encoder import TemporalEncoder
from src.TafM_改进源码.train import train_target_day

MODES = ("full_current", "tabular_only", "temporal_only")


# --- canonical default preservation ----------------------------------------
def test_canonical_defaults_are_preserved_on_every_public_entry_point():
    assert ARCHITECTURE_MODES == {"full_current", "tabular_only", "temporal_only"}
    assert inspect.signature(train_target_day).parameters["architecture_mode"].default == "full_current"
    assert inspect.signature(DualBranchV21).parameters["architecture_mode"].default == "full_current"
    assert inspect.signature(MemberWiseTaskAdaptersV21).parameters["architecture_mode"].default == "full_current"


def test_default_construction_equals_explicit_full_current():
    torch.manual_seed(0)
    default = MemberWiseTaskAdaptersV21(4, 4, 4, "A2")
    torch.manual_seed(0)
    explicit = MemberWiseTaskAdaptersV21(4, 4, 4, "A2", architecture_mode="full_current")
    assert default.architecture_mode == "full_current" == explicit.architecture_mode
    assert torch.equal(default.a_dir.detach(), explicit.a_dir.detach())
    assert torch.equal(default.tab_dir.weight.detach(), explicit.tab_dir.weight.detach())


# --- toy adapter harness ----------------------------------------------------
def _adapter(architecture_mode, *, seed=0, mode="A2"):
    torch.manual_seed(seed)
    return MemberWiseTaskAdaptersV21(4, 4, 4, mode, architecture_mode=architecture_mode)


def _inputs(*, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 2, 24, 4, generator=generator),
            torch.randn(1, 24, 4, generator=generator))


def test_full_current_direction_is_the_canonical_alpha_fusion():
    h_tab, h_time = _inputs()
    adapter = _adapter("full_current")
    h_dir, _ = adapter(h_tab, h_time)
    alpha, _ = adapter.alphas()
    time_dir = adapter.time_dir(h_time).unsqueeze(1).expand(-1, h_tab.shape[1], -1, -1)
    tab_dir = adapter.tab_dir(h_tab)
    assert torch.equal(h_dir, alpha * time_dir + (1.0 - alpha) * tab_dir)


def test_full_current_is_bit_identical_whether_default_or_explicit():
    h_tab, h_time = _inputs()
    default, explicit = _adapter("full_current"), _adapter("full_current")
    d_dir, d_mag = default(h_tab, h_time)
    e_dir, e_mag = explicit(h_tab, h_time)
    assert torch.equal(d_dir, e_dir) and torch.equal(d_mag, e_mag)


def test_tabular_only_direction_reads_the_tabular_branch_only():
    h_tab, h_time = _inputs()
    adapter = _adapter("tabular_only")
    h_dir, _ = adapter(h_tab, h_time)
    assert torch.equal(h_dir, adapter.tab_dir(h_tab))


def test_temporal_only_direction_reads_the_temporal_branch_only():
    h_tab, h_time = _inputs()
    adapter = _adapter("temporal_only")
    h_dir, _ = adapter(h_tab, h_time)
    expected = adapter.time_dir(h_time).unsqueeze(1).expand(-1, h_tab.shape[1], -1, -1)
    assert torch.equal(h_dir, expected)


def test_magnitude_fusion_is_identical_in_all_three_modes():
    h_tab, h_time = _inputs()
    magnitudes = [_adapter(mode)(h_tab, h_time)[1] for mode in MODES]
    assert torch.equal(magnitudes[0], magnitudes[1])
    assert torch.equal(magnitudes[0], magnitudes[2])


# --- gradient ownership (Gate A requirements 2 and 3) -----------------------
def test_tabular_only_direction_cannot_reach_temporal_parameters():
    h_tab, h_time = _inputs()
    adapter = _adapter("tabular_only")
    h_dir, _ = adapter(h_tab, h_time)
    grads = torch.autograd.grad(h_dir.sum(), tuple(adapter.time_dir.parameters()), allow_unused=True)
    assert all(g is None for g in grads)


def test_temporal_only_direction_cannot_reach_tabular_parameters():
    h_tab, h_time = _inputs()
    adapter = _adapter("temporal_only")
    h_dir, _ = adapter(h_tab, h_time)
    grads = torch.autograd.grad(h_dir.sum(), tuple(adapter.tab_dir.parameters()), allow_unused=True)
    assert all(g is None for g in grads)


# --- official-model end-to-end ---------------------------------------------
def _official_model(architecture_mode, *, k=2, seed=0):
    """Same construct as the existing V2.1 end-to-end contract, plus the arch axis."""
    torch.manual_seed(seed)
    names = list(TEMPORAL_FEATURES[1:]) + ["extra_forecast"]
    roles = [{"feature_name": n, "role": "Strong" if i == 0 else "Weak"}
             for i, n in enumerate(names)]
    bins = [[-3.0, 0.0, 3.0] for _ in names]
    tabular = TabularEncoder(feature_names=names, feature_roles=roles, ple_bins=bins,
                             d_tab=8, k=k, n_blocks=2, ple_embedding_dim=2)
    temporal = TemporalEncoder(selected_feature_names=names, d_time=8, hidden=8, fft_bins=(1, 7))
    model = DualBranchV21(d_tab=8, d_time=8, d_task=4, mode="A2", tabular_encoder=tabular,
                          temporal_encoder=temporal, k=k, magnitude_scale=2.5,
                          architecture_mode=architecture_mode)
    return model, names


@pytest.mark.parametrize("architecture_mode", MODES)
def test_all_modes_keep_finite_shapes_and_the_member_axis(architecture_mode):
    model, names = _official_model(architecture_mode)
    xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, len(names))
    out = model(xh, xf)
    assert model.architecture_mode == architecture_mode
    assert out["H_tab"].shape == (2, 2, 24, 8)
    assert out["H_time"].shape == (2, 24, 8)
    assert out["z_members"].shape == (2, 2, 24)
    assert out["a_scaled_members"].shape == (2, 2, 24)
    assert torch.equal(out["direction_hat"], out["p"] >= 0.5)
    assert all(torch.isfinite(v).all() for v in out.values() if isinstance(v, torch.Tensor))
    assert torch.isfinite(v21_loss(out, torch.randn(2, 24))["L_total"])


def test_tabular_only_blocks_direction_gradient_but_keeps_magnitude_gradient():
    model, names = _official_model("tabular_only")
    xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, len(names))
    out = model(xh, xf)
    losses = v21_loss(out, torch.randn(2, 24))
    temporal = tuple(model.temporal_encoder.parameters())
    direction_grads = torch.autograd.grad(losses["L_dir"], temporal, allow_unused=True, retain_graph=True)
    assert all(g is None for g in direction_grads)
    magnitude_grads = torch.autograd.grad(losses["L_mag"], temporal, allow_unused=True)
    assert any(g is not None for g in magnitude_grads)


def test_temporal_only_blocks_direction_gradient_but_keeps_magnitude_gradient():
    model, names = _official_model("temporal_only")
    xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, len(names))
    out = model(xh, xf)
    losses = v21_loss(out, torch.randn(2, 24))
    tabular = tuple(model.tabular_encoder.parameters())
    direction_grads = torch.autograd.grad(losses["L_dir"], tabular, allow_unused=True, retain_graph=True)
    assert all(g is None for g in direction_grads)
    magnitude_grads = torch.autograd.grad(losses["L_mag"], tabular, allow_unused=True)
    assert any(g is not None for g in magnitude_grads)


def test_parameter_count_and_state_dict_shape_are_mode_invariant():
    shapes = []
    for architecture_mode in MODES:
        model, _ = _official_model(architecture_mode)
        shapes.append((sum(p.numel() for p in model.parameters()),
                       {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}))
    assert shapes[1] == shapes[0]
    assert shapes[2] == shapes[0]


# --- rejection --------------------------------------------------------------
def test_unknown_architecture_mode_is_rejected():
    with pytest.raises(ValueError, match="architecture_mode"):
        MemberWiseTaskAdaptersV21(4, 4, 4, "A2", architecture_mode="late_fusion")
    with pytest.raises(ValueError, match="architecture_mode"):
        DualBranchV21(d_tab=8, d_time=8, d_task=4, mode="A2", tabular_encoder=None,
                      temporal_encoder=None, k=2, magnitude_scale=2.5, architecture_mode="mmoe")


def test_legacy_v20_cannot_use_an_architecture_mode():
    with pytest.raises(ValueError, match="legacy_v20"):
        train_target_day("2026-02-13", legacy_v20=True, architecture_mode="tabular_only")

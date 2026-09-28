"""E2-B1 Gate A — experiment-only fixed Direction fusion alpha contract.

`direction_fusion_alpha=None` (default) is the canonical learnable alpha_time and must stay
bit-identical to today's model. A float pins alpha_time for the Direction fusion only:

    alpha=0.0  -> h_dir = tab_dir(H_tab)    (structurally E2-A `tabular_only`)
    alpha=1.0  -> h_dir = time_dir(H_time)  (structurally E2-A `temporal_only`)
    0<a<1      -> h_dir = a*td + (1-a)*sd

The Magnitude fusion, the parameter set and the TabM member axis are untouched, so the fixed
alpha is a pure Direction-mixture axis and the endpoints stay directly comparable to E2-A A1/A2.
"""
import inspect

import pytest
import torch

from src.TafM_改进源码.contracts import TEMPORAL_FEATURES
from src.TafM_改进源码.losses import v21_loss
from src.TafM_改进源码.models.dual_branch_v21 import DualBranchV21
from src.TafM_改进源码.models.tabular_encoder import TabularEncoder
from src.TafM_改进源码.models.task_adapters import MemberWiseTaskAdaptersV21
from src.TafM_改进源码.models.temporal_encoder import TemporalEncoder
from src.TafM_改进源码.train import train_target_day

ENDPOINTS = (0.0, 1.0)
INTERIOR = (0.2, 0.4, 0.6, 0.8)


# --- canonical default preservation ----------------------------------------
def test_canonical_default_is_none_on_every_public_entry_point():
    assert inspect.signature(train_target_day).parameters["direction_fusion_alpha"].default is None
    assert inspect.signature(DualBranchV21).parameters["direction_fusion_alpha"].default is None
    assert inspect.signature(MemberWiseTaskAdaptersV21).parameters[
        "direction_fusion_alpha"].default is None


def test_default_construction_equals_explicit_none():
    torch.manual_seed(0)
    default = MemberWiseTaskAdaptersV21(4, 4, 4, "A2")
    torch.manual_seed(0)
    explicit = MemberWiseTaskAdaptersV21(4, 4, 4, "A2", direction_fusion_alpha=None)
    assert default.direction_fusion_alpha is None
    for a, b in zip(default.state_dict().values(), explicit.state_dict().values()):
        assert torch.equal(a, b)


# --- endpoint structural equivalence ---------------------------------------
def _adapter(direction_fusion_alpha=None, *, architecture_mode="full_current", seed=0):
    torch.manual_seed(seed)
    return MemberWiseTaskAdaptersV21(4, 4, 4, "A2", architecture_mode=architecture_mode,
                                     direction_fusion_alpha=direction_fusion_alpha)


def _inputs(*, seed=1):
    torch.manual_seed(seed)
    return torch.randn(1, 2, 24, 4), torch.randn(1, 24, 4)


@pytest.mark.parametrize("alpha,mode", [(0.0, "tabular_only"), (1.0, "temporal_only")])
def test_fixed_endpoint_is_bit_identical_to_the_matching_architecture_mode(alpha, mode):
    tab, time = _inputs()
    torch.manual_seed(0)
    fixed = MemberWiseTaskAdaptersV21(4, 4, 4, "A2", direction_fusion_alpha=alpha)
    torch.manual_seed(0)
    branch = MemberWiseTaskAdaptersV21(4, 4, 4, "A2", architecture_mode=mode)
    f_dir, f_mag = fixed(tab, time)
    b_dir, b_mag = branch(tab, time)
    assert torch.equal(f_dir, b_dir)
    assert torch.equal(f_mag, b_mag)


def test_fixed_alpha_08_and_learnable_08_are_different_arms_despite_equal_initial_value():
    """The plan is explicit that these are distinct arms. At init they coincide numerically, so the
    real distinction is behavioural: the learnable alpha can move, the fixed one cannot."""
    tab, time = _inputs()
    torch.manual_seed(0)
    fixed = MemberWiseTaskAdaptersV21(4, 4, 4, "A2", direction_fusion_alpha=0.8)
    torch.manual_seed(0)
    learnable = MemberWiseTaskAdaptersV21(4, 4, 4, "A2")
    assert torch.equal(fixed(tab, time)[0], learnable(tab, time)[0])
    assert torch.sigmoid(learnable.a_dir).item() == pytest.approx(0.8, abs=1e-7)
    # Both keep an `a_dir` Parameter (the parameter set is unchanged); the difference is that the
    # fixed arm's copy never receives a gradient, so it cannot move.

    # `a_dir` holds the logit, and alphas() applies the sigmoid.
    before = fixed.a_dir.detach().clone()
    optimizer = torch.optim.AdamW(list(fixed.parameters()) + list(learnable.parameters()), lr=0.5)
    for _ in range(3):
        optimizer.zero_grad()
        (fixed(tab, time)[0].sum() + learnable(tab, time)[0].sum()).backward()
        optimizer.step()
    assert torch.equal(fixed.a_dir.detach(), before)
    assert not torch.equal(learnable.a_dir.detach(), before)
    assert not torch.equal(fixed(tab, time)[0], learnable(tab, time)[0])


@pytest.mark.parametrize("alpha", INTERIOR)
def test_interior_alpha_is_the_exact_convex_combination(alpha):
    tab, time = _inputs()
    torch.manual_seed(0)
    model = MemberWiseTaskAdaptersV21(4, 4, 4, "A2", direction_fusion_alpha=alpha)
    h_dir, _ = model(tab, time)
    td = model.time_dir(time).unsqueeze(1).expand(-1, tab.shape[1], -1, -1)
    sd = model.tab_dir(tab)
    # Mirror the module's float32 arithmetic order: (1-weight) is evaluated in float32, which is
    # not the same rounding as computing it in float64 and casting.
    weight = torch.as_tensor(alpha, dtype=td.dtype)
    assert torch.equal(h_dir, weight * td + (1 - weight) * sd)


# --- alpha is fixed, and the unused branch leaves the graph ----------------
@pytest.mark.parametrize("alpha", ENDPOINTS + INTERIOR)
def test_fixed_alpha_receives_no_update(alpha):
    model = _adapter(alpha)
    tab, time = _inputs()
    before = model.a_dir.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
    for _ in range(3):
        h_dir, h_mag = model(tab, time)
        optimizer.zero_grad()
        (h_dir.sum() + h_mag.sum()).backward()
        optimizer.step()
    assert model.a_dir.grad is None
    assert torch.equal(model.a_dir.detach(), before)


def _grads(loss, params):
    return torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)


@pytest.mark.parametrize("alpha", INTERIOR)
def test_interior_alpha_lets_direction_reach_both_branches(alpha):
    model = _adapter(alpha)
    tab, time = _inputs()
    h_dir, _ = model(tab, time)
    for params in (tuple(model.tab_dir.parameters()), tuple(model.time_dir.parameters())):
        assert all(g is not None for g in _grads(h_dir.sum(), params))


def test_endpoint_alpha_zero_blocks_direction_from_the_temporal_branch():
    model = _adapter(0.0)
    tab, time = _inputs()
    h_dir, _ = model(tab, time)
    assert all(g is None for g in _grads(h_dir.sum(), tuple(model.time_dir.parameters())))
    assert all(g is not None for g in _grads(h_dir.sum(), tuple(model.tab_dir.parameters())))


def test_endpoint_alpha_one_blocks_direction_from_the_tabular_branch():
    model = _adapter(1.0)
    tab, time = _inputs()
    h_dir, _ = model(tab, time)
    assert all(g is None for g in _grads(h_dir.sum(), tuple(model.tab_dir.parameters())))
    assert all(g is not None for g in _grads(h_dir.sum(), tuple(model.time_dir.parameters())))


# --- magnitude and shape invariance ----------------------------------------
@pytest.mark.parametrize("alpha", (None,) + ENDPOINTS + INTERIOR)
def test_magnitude_fusion_is_identical_for_every_alpha(alpha):
    tab, time = _inputs()
    torch.manual_seed(0)
    model = MemberWiseTaskAdaptersV21(4, 4, 4, "A2", direction_fusion_alpha=alpha)
    torch.manual_seed(0)
    canonical = MemberWiseTaskAdaptersV21(4, 4, 4, "A2")
    assert torch.equal(model(tab, time)[1], canonical(tab, time)[1])


def test_all_alphas_keep_finite_shapes_and_the_member_axis():
    for alpha in (None,) + ENDPOINTS + INTERIOR:
        model, names = _official_model(alpha)
        xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, len(names))
        out = model(xh, xf)
        assert model.direction_fusion_alpha == alpha
        assert out["H_tab"].shape == (2, 2, 24, 8)
        assert out["H_time"].shape == (2, 24, 8)
        assert out["z_members"].shape == (2, 2, 24)
        assert out["a_scaled_members"].shape == (2, 2, 24)
        assert torch.equal(out["direction_hat"], out["p"] >= 0.5)
        assert all(torch.isfinite(v).all() for v in out.values() if isinstance(v, torch.Tensor))


def _official_model(direction_fusion_alpha=None, *, k=2, seed=0):
    """Same construct as the existing V2.1 end-to-end contract, plus the fusion-alpha axis."""
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
                          direction_fusion_alpha=direction_fusion_alpha)
    return model, names


def test_parameter_count_and_state_dict_shape_are_alpha_invariant():
    reference = None
    for alpha in (None,) + ENDPOINTS + INTERIOR:
        model, _ = _official_model(alpha)
        shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        count = sum(p.numel() for p in model.parameters())
        if reference is None:
            reference = (shapes, count)
        assert (shapes, count) == reference


# --- rejection ---------------------------------------------------------------
@pytest.mark.parametrize("alpha", [-0.01, 1.01, 2.0, -1.0])
def test_alpha_outside_unit_interval_is_rejected(alpha):
    with pytest.raises(ValueError, match="within \\[0,1\\]"):
        MemberWiseTaskAdaptersV21(4, 4, 4, "A2", direction_fusion_alpha=alpha)
    with pytest.raises(ValueError, match="within \\[0,1\\]"):
        DualBranchV21(d_tab=8, d_time=8, d_task=4, mode="A2",
                      tabular_encoder=torch.nn.Identity(), temporal_encoder=torch.nn.Identity(),
                      k=2, magnitude_scale=2.5, direction_fusion_alpha=alpha)


@pytest.mark.parametrize("mode", ["tabular_only", "temporal_only"])
def test_fixed_alpha_cannot_be_combined_with_a_non_default_architecture_mode(mode):
    with pytest.raises(ValueError, match="requires architecture_mode=full_current"):
        MemberWiseTaskAdaptersV21(4, 4, 4, "A2", architecture_mode=mode, direction_fusion_alpha=0.5)


@pytest.mark.parametrize("mode", ["tabular_only", "temporal_only"])
def test_train_target_day_rejects_the_same_combination(mode):
    with pytest.raises(ValueError, match="requires architecture_mode=full_current"):
        train_target_day("2026-02-13", profile="smoke", train_mode="stage_a",
                         architecture_mode=mode, direction_fusion_alpha=0.5)


def test_train_target_day_rejects_an_out_of_range_alpha():
    with pytest.raises(ValueError, match="within \\[0,1\\]"):
        train_target_day("2026-02-13", profile="smoke", train_mode="stage_a",
                         direction_fusion_alpha=1.5)


def test_legacy_v20_cannot_use_a_fixed_alpha():
    with pytest.raises(ValueError, match="canonical V2.1 model"):
        train_target_day("2026-02-13", profile="smoke", train_mode="stage_a",
                         legacy_v20=True, direction_fusion_alpha=0.5)

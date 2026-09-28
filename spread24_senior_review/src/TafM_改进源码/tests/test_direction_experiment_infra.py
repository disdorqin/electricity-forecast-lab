import copy
import inspect

import pytest
import torch
from torch import nn

from src.TafM_改进源码.direction_experiments import (
    audit_protected_parameter_groups, clip_protected_gradient_groups, direction_first_checkpoint_is_better,
    experiment_output_category, gradient_diagnostic_rows, magnitude_first_checkpoint_is_better,
    objective_tensor, project_magnitude_gradient, protected_gradients, summarize_gradient_rows,
    summarize_projection_rows, update_direction_first_checkpoint, update_direction_first_progress,
    update_magnitude_first_progress,
)
from src.TafM_改进源码.config import V2Config
from src.TafM_改进源码.losses import v21_loss
from src.TafM_改进源码.models.task_adapters import MemberWiseTaskAdaptersV21
from src.TafM_改进源码.train import _train_epoch_experiment, update_checkpoint_state, train_target_day


def test_experiment_outputs_are_partitioned_from_formal_runs():
    assert experiment_output_category("joint_v21", "v21_guardrail", "vanilla", False) == "smoke"
    assert experiment_output_category("joint_v21", "direction_first", "vanilla", True) == "gradient_diagnostics"
    assert experiment_output_category("dir_only", "direction_first", "vanilla", False) == "objective_modes"
    assert experiment_output_category("joint_v21", "direction_first", "direction_protected", False) == "protected_gradient"


def _loss_inputs():
    z = torch.linspace(-1, 1, 2 * 24).reshape(1, 2, 24).requires_grad_()
    a = torch.rand(1, 2, 24, requires_grad=True)
    y = torch.linspace(-2, 2, 24).reshape(1, 24)
    return z, a, y


def test_joint_v21_matches_current_loss():
    z, a, y = _loss_inputs()
    losses = v21_loss({"z_members": z, "a_scaled_members": a}, y)
    selected = objective_tensor(losses, "joint_v21")
    assert selected is losses["L_total"]
    direct = losses["L_dir"] + losses["L_mag"]
    assert torch.equal(selected, direct)


def test_dir_only_has_no_magnitude_training_gradient():
    z, a, y = _loss_inputs()
    losses = v21_loss({"z_members": z, "a_scaled_members": a}, y)
    objective_tensor(losses, "dir_only").backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert a.grad is None


def test_mag_only_has_no_direction_training_gradient():
    z, a, y = _loss_inputs()
    losses = v21_loss({"z_members": z, "a_scaled_members": a}, y)
    objective_tensor(losses, "mag_only").backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert z.grad is None


def test_direction_first_higher_raw_always_wins():
    high = {"raw_direction_accuracy": .7, "L_dir": 2., "magnitude_mae": 100., "epoch": 2}
    low = {"raw_direction_accuracy": .69, "L_dir": .1, "magnitude_mae": .1, "epoch": 1}
    assert direction_first_checkpoint_is_better(high, low)
    assert not direction_first_checkpoint_is_better(low, high)


def test_direction_first_raw_tie_uses_lower_bce():
    best = {"raw_direction_accuracy": .7, "L_dir": .4, "magnitude_mae": 2., "epoch": 1}
    candidate = {"raw_direction_accuracy": .7, "L_dir": .3, "magnitude_mae": 9., "epoch": 2}
    assert direction_first_checkpoint_is_better(candidate, best)


def test_direction_first_bce_tie_uses_magnitude():
    best = {"raw_direction_accuracy": .7, "L_dir": .4, "magnitude_mae": 2., "epoch": 1}
    candidate = {"raw_direction_accuracy": .7, "L_dir": .4, "magnitude_mae": 1., "epoch": 2}
    assert direction_first_checkpoint_is_better(candidate, best)


def test_direction_first_final_tie_uses_earlier_epoch():
    early = {"raw_direction_accuracy": .7, "L_dir": .4, "magnitude_mae": 1., "epoch": 1}
    later = {**early, "epoch": 2}
    assert not direction_first_checkpoint_is_better(later, early)
    assert update_direction_first_checkpoint(later, early)["selected"] == early


def test_v21_guardrail_unchanged():
    first = {"raw_direction_accuracy": .55, "magnitude_mae": 30., "L_total": 2., "epoch": 1}
    second = {"raw_direction_accuracy": .54, "magnitude_mae": 20., "L_total": 1., "epoch": 2}
    updated = update_checkpoint_state(second, .55, first, tolerance=.02)
    assert updated["selected"]["epoch"] == 2
    assert updated["raw_anchor"] == .55
    assert inspect.signature(train_target_day).parameters["objective_mode"].default == "joint_v21"
    assert inspect.signature(train_target_day).parameters["checkpoint_policy"].default == "v21_guardrail"


def test_magnitude_first_ignores_direction_and_prefers_lower_mae():
    best = {"raw_direction_accuracy": .90, "magnitude_mae": 20., "L_mag": .5, "epoch": 1}
    candidate = {"raw_direction_accuracy": .10, "magnitude_mae": 19., "L_mag": .9, "epoch": 2}
    assert magnitude_first_checkpoint_is_better(candidate, best)


def test_magnitude_first_mae_tie_uses_lmag_then_earlier_epoch():
    best = {"raw_direction_accuracy": .90, "magnitude_mae": 20., "L_mag": .5, "epoch": 2}
    lower_l1 = {"raw_direction_accuracy": .10, "magnitude_mae": 20., "L_mag": .4, "epoch": 3}
    assert magnitude_first_checkpoint_is_better(lower_l1, best)
    exact_late = {**lower_l1, "epoch": 4}
    assert not magnitude_first_checkpoint_is_better(exact_late, lower_l1)


def test_magnitude_first_progress_uses_mae_then_lmag_only():
    first = update_magnitude_first_progress({"magnitude_mae": 20., "L_mag": .5}, None, None)
    assert first["progress"] and first["best_mae"] == 20.
    tied = update_magnitude_first_progress({"magnitude_mae": 20., "L_mag": .4998}, 20., .5, min_delta=1e-4)
    assert tied["progress"] and tied["l1_improved_at_best_mae"]
    worse = update_magnitude_first_progress(
        {"magnitude_mae": 21., "L_mag": .1, "raw_direction_accuracy": 1.0}, 20., .5, min_delta=1e-4)
    assert not worse["progress"]


def test_direction_first_new_raw_best_resets_patience():
    state = update_direction_first_progress({"raw_direction_accuracy": .6, "L_dir": .5}, .59, .4)
    assert state["progress"] and state["raw_anchor"] == .6


def test_direction_first_same_raw_bce_delta_resets_patience():
    state = update_direction_first_progress({"raw_direction_accuracy": .6, "L_dir": .4999}, .6, .5,
                                            min_delta=1e-4)
    assert state["progress"] and state["bce_improved_at_best_raw"]


def test_direction_first_magnitude_only_improvement_does_not_reset_patience():
    # Early-stop progress consumes only Raw and BCE; Magnitude is intentionally absent.
    state = update_direction_first_progress({"raw_direction_accuracy": .6, "L_dir": .5,
        "magnitude_mae": 1.}, .6, .5, min_delta=1e-4)
    assert not state["progress"]


def test_gradient_diagnostics_do_not_mutate_existing_grad_or_parameters():
    p = nn.Parameter(torch.tensor([1., -2.])); q = nn.Parameter(torch.tensor([3.]))
    p.grad = torch.tensor([9., 8.]); q.grad = torch.tensor([7.])
    params_before = (p.detach().clone(), q.detach().clone())
    grads_before = (p.grad.clone(), q.grad.clone())
    ld = (p * torch.tensor([1., 0.])).sum()
    lm = (p * torch.tensor([-1., 0.])).sum()
    rows = gradient_diagnostic_rows(ld, lm, {"tabular_encoder": [p], "temporal_encoder": [q]},
                                    epoch=1, batch_index=0)
    assert rows[0]["cosine_similarity"] < 0
    assert torch.equal(p, params_before[0]) and torch.equal(q, params_before[1])
    assert torch.equal(p.grad, grads_before[0]) and torch.equal(q.grad, grads_before[1])


def test_conflicting_toy_gradient_has_negative_cosine():
    p = nn.Parameter(torch.tensor([1., 2.]))
    rows = gradient_diagnostic_rows((p * torch.tensor([1., 0.])).sum(),
        (p * torch.tensor([-1., 0.])).sum(), {"tabular_encoder": [p]}, epoch=1, batch_index=0)
    assert rows[0]["cosine_similarity"] < 0 and rows[0]["conflict"] is True


def test_aligned_toy_gradient_has_nonnegative_cosine():
    p = nn.Parameter(torch.tensor([1., 2.]))
    rows = gradient_diagnostic_rows((p * torch.tensor([1., 0.])).sum(),
        (p * torch.tensor([2., 0.])).sum(), {"temporal_encoder": [p]}, epoch=1, batch_index=0)
    assert rows[0]["cosine_similarity"] >= 0 and rows[0]["conflict"] is False


def test_unused_gradient_group_is_explicit_na():
    p = nn.Parameter(torch.tensor([1.])); unused = nn.Parameter(torch.tensor([2.]))
    rows = gradient_diagnostic_rows(p.sum(), (p * 2).sum(), {"unused": [unused]}, epoch=1, batch_index=0)
    assert rows[0]["status"] == "N/A_NO_GRADIENT"
    assert rows[0]["cosine_similarity"] == "N/A"


def test_gradient_summary_uses_statistical_median_for_even_batch_count():
    rows = [{"parameter_group": "tabular_encoder", "cosine_similarity": value,
             "dir_grad_norm": value, "mag_grad_norm": value, "norm_ratio": value,
             "conflict": value < 0}
            for value in (-0.5, 0.5)]
    summary = summarize_gradient_rows(rows)["tabular_encoder"]
    assert summary["median_cosine"] == 0.0
    assert summary["median_norm_ratio"] == 0.0


def test_diagnostic_rows_are_grouped_under_the_actual_training_epoch():
    from torch.utils.data import DataLoader, TensorDataset
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tabular_encoder = nn.Linear(1, 1)
            self.temporal_encoder = nn.Linear(1, 1)

        def forward(self, xh, xf, **kwargs):
            logits = (self.tabular_encoder(xf) + self.temporal_encoder(xh)).reshape(-1, 1, 1)
            z = logits.expand(-1, 2, 24)
            a = torch.nn.functional.softplus(logits).expand(-1, 2, 24)
            return {"z_members": z, "a_scaled_members": a}

    model = TinyModel()
    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.ones(2, 1),
        torch.zeros(2, 1), torch.ones(2, 24)), batch_size=1, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    rows = []
    _train_epoch_experiment(model, loader, optimizer, "cpu", V2Config(),
        torch.amp.GradScaler("cuda", enabled=False), objective_mode="joint_v21",
        gradient_policy="vanilla", amp=False, epoch=7, diagnostic=True, batches_per_epoch=2,
        diagnostic_rows=rows)
    assert len(rows) == 4  # two parameter groups for each of two diagnostic batches
    assert {row["epoch"] for row in rows} == {7}
    assert sorted(row["batch_index"] for row in rows) == [0, 0, 1, 1]


def test_gradient_diagnostics_preserve_the_gradient_used_by_backward():
    p0 = nn.Parameter(torch.tensor([1., -2.])); p1 = nn.Parameter(p0.detach().clone())
    ld0=(p0.square()).sum(); lm0=((p0-2).square()).sum()
    ld1=(p1.square()).sum(); lm1=((p1-2).square()).sum()
    objective0=ld0+lm0; objective1=ld1+lm1
    objective0.backward()
    gradient_diagnostic_rows(ld1,lm1,{"tabular_encoder":[p1]},epoch=1,batch_index=0)
    objective1.backward()
    assert torch.equal(p0.grad,p1.grad)


def test_conflicting_gradient_projection_is_asymmetric_and_direction_unchanged():
    gd=torch.tensor([1.,0.]); gm=torch.tensor([-2.,3.]); before=gd.clone()
    safe, audit=project_magnitude_gradient(gd,gm)
    assert audit["status"]=="PROJECTED_CONFLICT"
    assert torch.dot(gd,safe)>=-1e-6
    assert torch.equal(gd,before)


def test_aligned_gradient_projection_leaves_magnitude_exactly_unchanged():
    gd=torch.tensor([1.,0.]); gm=torch.tensor([2.,3.]); before=gm.clone()
    safe,audit=project_magnitude_gradient(gd,gm)
    assert audit["status"]=="UNCHANGED_ALIGNED_OR_ZERO"
    assert torch.equal(safe,before)


class _ProtectedToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.tabular_encoder=nn.Linear(3,4)
        self.temporal_encoder=nn.Linear(5,4)
        self.adapters=MemberWiseTaskAdaptersV21(4,4,4,"A2")
        self.direction_head=nn.Linear(4,1)
        self.magnitude_head=nn.Linear(4,1)

    def forward(self, tab, temporal, protected=True):
        h_tab=self.tabular_encoder(tab).reshape(1,1,1,4).expand(-1,2,24,-1)
        h_time=self.temporal_encoder(temporal).reshape(1,1,4).expand(-1,24,-1)
        h_dir,h_mag=self.adapters(h_tab,h_time,detach_magnitude_time=protected)
        return self.direction_head(h_dir).mean(),torch.nn.functional.softplus(self.magnitude_head(h_mag)).mean()


def test_parameter_ownership_groups_have_no_overlap_or_unknown_trainables():
    model=_ProtectedToy()
    audit,groups=audit_protected_parameter_groups(model)
    assert audit["status"]=="PASS"
    assert audit["overlap"]==[] and audit["unknown_trainable_parameter_tensors"]==0
    assert set(groups)=={"direction_owned","magnitude_owned","shared"}
    assert audit["all_trainable_parameters_owned"]


def test_magnitude_loss_cannot_modify_direction_owned_temporal_grads():
    model=_ProtectedToy()
    audit,groups=audit_protected_parameter_groups(model)
    x=torch.randn(1,3); t=torch.randn(1,5)
    ld,lm=model(x,t,protected=True)
    temporal=tuple(model.temporal_encoder.parameters())
    mag_grads=torch.autograd.grad(lm,temporal,allow_unused=True,retain_graph=True)
    assert all(g is None for g in mag_grads)
    direction_only_params=groups["direction_owned"]
    mag_head_ids={id(p) for p in model.magnitude_head.parameters()}
    direction_grads=torch.autograd.grad(ld,tuple(p for p in model.parameters() if id(p) in mag_head_ids),
                                        allow_unused=True,retain_graph=True)
    assert all(g is None for g in direction_grads)


def test_protected_combined_gradient_assigns_owned_objectives_and_projects_shared():
    model=_ProtectedToy(); _,groups=audit_protected_parameter_groups(model)
    x=torch.randn(1,3); t=torch.randn(1,5)
    ld,lm=model(x,t,protected=True)
    status,_=protected_gradients(ld,lm,groups)
    assert status["status"]=="PASS"
    assert status["projection_name"]=="direction_protected_asymmetric"
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is not None for p in groups["direction_owned"])
    assert all(p.grad is not None for p in groups["magnitude_owned"])
    assert len(status["projection_rows"]) == 1
    row=status["projection_rows"][0]
    assert {"dot_product","post_projection_dot_product","projected","dir_grad_norm",
            "mag_grad_norm","safe_mag_grad_norm","removed_fraction"} <= set(row)


def test_groupwise_clipping_prevents_magnitude_owned_norm_from_rescaling_direction_owned():
    def make_groups(magnitude_value):
        direction=nn.Parameter(torch.zeros(2)); magnitude=nn.Parameter(torch.zeros(1)); shared=nn.Parameter(torch.zeros(1))
        direction.grad=torch.tensor([3.,4.])
        magnitude.grad=torch.tensor([magnitude_value])
        shared.grad=torch.tensor([0.5])
        return {"direction_owned":(direction,), "magnitude_owned":(magnitude,), "shared":(shared,)}, direction
    groups_small,d_small=make_groups(1.0)
    groups_huge,d_huge=make_groups(1000.0)
    norms_small=clip_protected_gradient_groups(groups_small,max_norm=1.0)
    norms_huge=clip_protected_gradient_groups(groups_huge,max_norm=1.0)
    assert torch.allclose(d_small.grad,d_huge.grad,atol=0,rtol=0)
    assert torch.linalg.vector_norm(d_small.grad) <= 1.000001
    assert norms_small["direction_owned"] == pytest.approx(norms_huge["direction_owned"])
    assert norms_huge["magnitude_owned"] > norms_small["magnitude_owned"]


def test_projection_summary_reports_projection_rate_and_removed_fraction():
    rows=[
        {"projected":True,"dot_product":-2.0,"post_projection_dot_product":0.0,"removed_fraction":0.5},
        {"projected":False,"dot_product":1.0,"post_projection_dot_product":1.0,"removed_fraction":0.0},
    ]
    summary=summarize_projection_rows(rows)
    assert summary["batches"]==2
    assert summary["projection_rate"]==pytest.approx(0.5)
    assert summary["mean_pre_dot"]==pytest.approx(-0.5)
    assert summary["mean_post_dot"]==pytest.approx(0.5)
    assert summary["mean_removed_fraction"]==pytest.approx(0.25)

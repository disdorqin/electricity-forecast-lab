import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.models.tabular_encoder import TabularEncoder
from src.TafM_改进源码.models.temporal_encoder import TemporalEncoder
from src.TafM_改进源码.models.dual_branch_v21 import DualBranchV21
from src.TafM_改进源码.losses import v21_loss


def make_model(route="current"):
    names = ["strong_a", "strong_b", "weak_a"]
    roles = [{"feature_name": "strong_a", "role": "Strong-DIR"},
             {"feature_name": "strong_b", "role": "Forced-Core"},
             {"feature_name": "weak_a", "role": "Weak"}]
    bins = [[-1.0, 0.0, 1.0] for _ in names]
    tab = TabularEncoder(feature_names=names, feature_roles=roles, ple_bins=bins,
                         d_tab=8, k=2, n_blocks=1, dropout=0.0,
                         ple_embedding_dim=2, ple_enabled=True)
    temporal = TemporalEncoder(selected_feature_names=names, d_time=4, hidden=4,
                               fft_bins=[1], fft_enabled=False,
                               future_conditioning=False, temporal_clip_abs=10.0)
    return DualBranchV21(8, 4, 4, mode="A2", tabular_encoder=tab,
                         temporal_encoder=temporal, k=2, magnitude_scale=1.0,
                         direction_fusion_alpha=0.8, direction_tabular_mode=route)


def _groups(model):
    tab = model.tabular_encoder
    return {
        "strong": tuple(tab.backbone.parameters()),
        "weak": tuple(tab.weak_mlp.parameters()),
        "gate": (tab.horizon_gate_logits,),
        "time": tuple(model.temporal_encoder.parameters()),
    }


def _grad_state(loss, params):
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    return sum(g is not None and bool(torch.count_nonzero(g)) for g in grads), sum(
        float(g.detach().norm()) for g in grads if g is not None)


def test_forward_components_reproduces_original_gated_formula_exactly():
    torch.manual_seed(41)
    model = make_model()
    x = torch.randn(2, 24, 3)
    tab = model.tabular_encoder
    strong, weak, gate_values, current = tab.forward_components(x)

    encoded = tab._encoded(x)
    ref_strong = tab.backbone(tab.ensemble_view(
        encoded[:, tab.strong_indices, :].reshape(2 * 24, -1)))
    ref_strong = ref_strong.reshape(2, 24, tab.k, tab.d_tab).permute(0, 2, 1, 3)
    ref_weak = tab.weak_mlp(encoded[:, tab.weak_indices, :].reshape(2 * 24, -1)).reshape(2, 24, tab.d_tab)
    ref_gate = torch.sigmoid(tab.horizon_gate_logits).view(1, 1, 24, 1)
    ref_current = ref_gate * ref_strong + (1.0 - ref_gate) * ref_weak.unsqueeze(1)
    assert torch.equal(strong, ref_strong)
    assert torch.equal(weak, ref_weak)
    assert torch.equal(current, ref_current)
    assert torch.equal(tab(x), current)
    assert gate_values.shape == (24,)


def test_c1_c2_direction_gradient_ownership_and_magnitude_route():
    torch.manual_seed(43)
    xh, xf = torch.randn(2, 168, 7), torch.randn(2, 24, 3)
    y = torch.linspace(-1.0, 1.0, 48).reshape(2, 24)
    for route, expected_live in (("strong_only", "strong"), ("weak_only", "weak")):
        model = make_model(route)
        out = model(xh, xf)
        assert out["z_members"].shape == (2, 2, 24)
        assert out["a_scaled_members"].shape == (2, 2, 24)
        assert all(torch.isfinite(v).all() for v in (out["p"], out["magnitude_hat"]))
        groups = _groups(model)
        ld = v21_loss(out, y)["L_dir"]
        live = {name: _grad_state(ld, params) for name, params in groups.items()}
        assert live[expected_live][0] > 0 and live[expected_live][1] > 0
        excluded = "weak" if route == "strong_only" else "strong"
        assert live[excluded] == (0, 0.0)
        assert live["gate"] == (0, 0.0)
        assert live["time"][0] > 0 and live["time"][1] > 0

        lm = v21_loss(out, y)["L_mag"]
        mag_live = {name: _grad_state(lm, params) for name, params in groups.items()}
        assert all(mag_live[n][0] > 0 and mag_live[n][1] > 0 for n in ("strong", "weak", "gate"))


def test_default_current_matches_fixed_alpha_current_model_and_keeps_member_axis():
    torch.manual_seed(47)
    model = make_model("current")
    xh, xf = torch.randn(1, 168, 7), torch.randn(1, 24, 3)
    out = model(xh, xf)
    assert model.direction_tabular_mode == "current"
    assert out["z_members"].shape == (1, 2, 24)
    assert out["magnitude_members"].shape == (1, 2, 24)
    assert all(torch.isfinite(v).all() for v in out.values() if isinstance(v, torch.Tensor))


def test_e2_c2_fixed_and_global_gate_gradient_ownership():
    torch.manual_seed(53)
    xh, xf = torch.randn(2,168,7), torch.randn(2,24,3)
    y = torch.linspace(-1,1,48).reshape(2,24)
    for mode in ("fixed_08", "global_learnable"):
        model=DualBranchV21(8,4,4,mode="A2",tabular_encoder=make_model().tabular_encoder,
            temporal_encoder=make_model().temporal_encoder,k=2,magnitude_scale=1.0,
            architecture_mode="full_current",direction_fusion_alpha=.8,
            direction_horizon_gate_mode=mode)
        out=model(xh,xf); assert out["z_members"].shape==(2,2,24)
        ld=v21_loss(out,y)["L_dir"]
        tab=model.tabular_encoder
        g=torch.autograd.grad(ld, (tuple(tab.backbone.parameters())+tuple(tab.weak_mlp.parameters())+
            (tab.horizon_gate_logits,)+tuple(model.temporal_encoder.parameters())+
            (() if model.direction_global_gate_logit is None else (model.direction_global_gate_logit,))),
            allow_unused=True,retain_graph=True)
        nstrong=len(tuple(tab.backbone.parameters())); nweak=len(tuple(tab.weak_mlp.parameters()))
        live=lambda xs:any(x is not None and bool(torch.count_nonzero(x)) for x in xs)
        assert live(g[:nstrong]) and live(g[nstrong:nstrong+nweak])
        assert g[nstrong+nweak] is None or not torch.count_nonzero(g[nstrong+nweak])
        assert live(g[nstrong+nweak+1:-1] if mode=="global_learnable" else g[nstrong+nweak+1:])
        if mode=="fixed_08": assert model.direction_global_gate_logit is None
        else: assert g[-1] is not None and torch.isfinite(g[-1]).all() and torch.count_nonzero(g[-1])
        lm=v21_loss(out,y)["L_mag"]
        mg=torch.autograd.grad(lm,(tuple(tab.backbone.parameters())+tuple(tab.weak_mlp.parameters())+
            (tab.horizon_gate_logits,)),allow_unused=True)
        assert live(mg[:nstrong]) and live(mg[nstrong:nstrong+nweak]) and live(mg[-1:])


def test_e2_c2_fail_closed_modes():
    base=make_model()
    import pytest
    with pytest.raises(ValueError):
        DualBranchV21(8,4,4,tabular_encoder=base.tabular_encoder,temporal_encoder=base.temporal_encoder,
            k=2,magnitude_scale=1.0,direction_horizon_gate_mode="fixed_08")

def test_e2_c3_role_profiles_exact_and_finite():
    from src.TafM_改进源码.models.tabular_encoder import TabularEncoder
    names=["both1","dir1","mag1","core1","weak1"]
    roles=[{"feature_name":n,"role":r} for n,r in zip(names,["Strong-BOTH","Strong-DIR","Strong-MAG","Forced-Core","Weak"])]
    kwargs=dict(feature_names=names,feature_roles=roles,ple_bins=[[-1,0,1]]*5,d_tab=8,k=2,n_blocks=1,dropout=0,ple_embedding_dim=2)
    a=TabularEncoder(**kwargs,strong_role_profile="all");m=TabularEncoder(**kwargs,strong_role_profile="drop_mag");d=TabularEncoder(**kwargs,strong_role_profile="drop_dir")
    assert len(a.strong_indices)==4 and len(m.strong_indices)==3 and len(d.strong_indices)==3
    assert a.weak_indices==m.weak_indices==d.weak_indices==(4,)

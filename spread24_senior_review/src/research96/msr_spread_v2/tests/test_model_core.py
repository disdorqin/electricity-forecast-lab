import copy
import numpy as np
import pytest
import torch
from ..config import Config,FAMILIES
from ..dataset import build_sample,true_label
from ..model import MSRSpreadV2
from ..spectral_router import SpectralRouter,dct_matrix
from ..quantile_head import QuantileHead,quantile
from ..dependence import DynamicDependence
from ..reconstruct import scenario_crps,evaluation_outputs,soft_median
from ..losses import marginal_score,objectives
from ..trainer import seed_everything,gradient_audit,set_stage,Preprocessor

@pytest.fixture
def graph(store):
    seed_everything(20261009)
    samples = [build_sample(store,d) for d in ("2023-02-10","2023-02-11","2023-02-12")]
    y = np.stack([true_label(store,s.target_day) for s in samples])
    raw = np.stack([s.x for s in samples])
    pre = Preprocessor().fit(raw,y,[s.target_day for s in samples])
    x = torch.tensor(pre.transform(raw))
    model = MSRSpreadV2(samples[0].names,Config(),pre.stats)
    return model,x,torch.tensor(y,dtype=torch.float32)

def test_T08_family_router():
    torch.manual_seed(7)
    router = SpectralRouter(8).double()
    e = {g:torch.randn(2,96,8,dtype=torch.float64)*10 for g in FAMILIES}
    low,high,info = router(e)
    assert len({id(v.c0_raw) for v in router.gates.values()})==len(FAMILIES)
    for g in FAMILIES:
        np.testing.assert_allclose(router.inverse(router.transform(e[g])).detach(),e[g],atol=2e-5)
        np.testing.assert_allclose((low[g]+high[g]).detach(),e[g],atol=2e-5)
        i = info[g]
        assert ((i["gate"]>=0)&(i["gate"]<=1)).all()
        assert (i["gate"].diff(dim=1)<=0).all()
        assert ((i["cutoff"]>=1)&(i["cutoff"]<=32)).all()
        assert .5<=i["tau"]<=8
        assert (i["cutoff"]-i["c0"]).abs().max()<=i["delta"]<=3
        assert torch.isfinite(i["gate"]).all()

def test_T09_monotone_quantiles():
    torch.manual_seed(3)
    head = QuantileHead(8)
    q = head(torch.randn(2,96,8))
    assert (q.diff(dim=-1)>0).all()
    u = torch.linspace(0,1,33).view(1,33,1).expand(2,-1,96).requires_grad_()
    s,clamp = quantile(q,u)
    assert clamp>0 and torch.isfinite(s).all()
    s.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in head.parameters())
    assert u.grad.abs().sum()>0

def test_T11_dynamic_dependence_normal_simulation():
    seed_everything(20261009)
    dep = DynamicDependence(7).double()
    context = torch.randn(2,7,dtype=torch.float64)
    curves = dep.curves(context)
    for key,lo,hi in (("rho",-.95,.95),("phiL",0,.98),("phiH",-.95,.95)):
        v = curves[key]
        assert (v>lo).all() and (v<hi).all() and torch.isfinite(v).all()
        # No p32/p33/p64/p65 artificial jump; smooth six-basis parametrization.
        assert v.diff(dim=-1).abs().max()<.03
    u,v,zl,zh = dep.ranks(curves,4096)
    for ranks,z in ((u,zl),(v,zh)):
        assert abs(float(z.detach().mean()))<.02
        assert abs(float(z.detach().var(unbiased=False))-1)<.03
        assert abs(float(ranks.detach().mean())-.5)<.01
        assert abs(float(ranks.detach().var(unbiased=False))-1/12)<.005
    again = dep.ranks(curves,4096)
    assert torch.equal(u,again[0]) and torch.equal(v,again[1])

def test_T12_scenario_identity(graph):
    model,x,y = graph
    out = model(x)
    assert torch.equal(out["spread_scenarios"],out["low_scenarios"]+out["high_scenarios"])
    assert out["regime_context"].shape==(3,96,1)
    assert torch.equal(out["regime_context"],torch.ones(3,96,1))
    assert sum(p.numel() for p in model.parameters())<3_000_000

def test_T13_final_empirical_outputs():
    s = torch.tensor([-1.,0.,1.,2.]).view(1,4,1).expand(1,4,96)
    da = torch.full((1,96),300.)
    out = evaluation_outputs(s,da)
    assert torch.equal(out["p_positive"],torch.full((1,96),.5))
    assert torch.equal(out["rt_median"],da+out["spread_median"])
    assert torch.equal(out["rt_quantiles"],da[...,None]+out["spread_quantiles"])
    assert torch.equal(out["spread_median"],torch.full((1,96),.5))
    assert torch.isfinite(soft_median(s)).all()

@pytest.mark.parametrize("term",["crps","brier","point"])
def test_T14_each_final_loss_reaches_all_groups(graph,term):
    model,x,y = graph
    model.eval()
    out = model(x,scenarios=32)
    terms = objectives(out,y,model)
    params = list(model.parameters())
    grads = torch.autograd.grad(terms[term],params,allow_unused=True)
    gmap = {id(p):g for p,g in zip(params,grads)}
    for name,ps in model.gradient_groups().items():
        gs = [gmap[id(p)] for p in ps if gmap[id(p)] is not None]
        assert gs and all(torch.isfinite(g).all() for g in gs),name
        norm = sum(float(g.double().square().sum()) for g in gs)**.5
        assert norm>1e-12,(term,name,norm)

def test_T14_gradient_norm_and_conflict_audit(graph):
    model,x,y = graph
    model.eval()
    result = gradient_audit(model,objectives(model(x),y,model))
    assert result["reachability"]=="PASS"
    assert result["final_vs_component_cosine"]["low_temporal/low"] is not None
    assert result["final_vs_component_cosine"]["high_temporal/high"] is not None

def test_T15_hand_scores():
    s = torch.tensor([0.,2.]).view(1,2,1)
    y = torch.tensor([[1.]])
    assert scenario_crps(s,y)==.5
    q = torch.full((1,96,9),2.)
    y = torch.zeros(1,96)
    assert torch.allclose(marginal_score(y,q),torch.tensor(2.))
    expected = (s-torch.ones(1,1,1)).abs().mean()-.5*(s[:, :,None,:]-s[:,None,:,:]).abs().mean()
    assert torch.allclose(scenario_crps(s,torch.ones(1,1)),expected)

@pytest.mark.parametrize("stage",["A","B","C"])
def test_T16_real_optimizer_freeze(graph,stage):
    model,x,y = graph
    set_stage(model,stage)
    before = {n:p.detach().clone() for n,p in model.named_parameters()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    out = model(x,components_only=stage=="A")
    losses = objectives(out,y,model)
    losses[{"A":"stage_a","B":"stage_b","C":"total"}[stage]].backward()
    opt.step()
    changed = [n for n,p in model.named_parameters() if not torch.equal(p,before[n])]
    assert changed
    if stage=="A":
        assert all(not n.startswith("dependence.") for n in changed)
    if stage=="B":
        assert all(n.startswith("dependence.") for n in changed)
        assert all(p.requires_grad==n.startswith("dependence.") for n,p in model.named_parameters())
    if stage=="C":
        assert all(p.requires_grad for p in model.parameters())
        assert any(n.startswith("low.") for n in changed) and any(n.startswith("high.") for n in changed)
        assert any(n.startswith("router.") for n in changed) and any(n.startswith("dependence.") for n in changed)

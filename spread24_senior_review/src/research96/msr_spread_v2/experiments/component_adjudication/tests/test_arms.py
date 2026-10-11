import numpy as np
import torch
from src.research96.msr_spread_v2.tests.conftest import store as fixture
from src.research96.msr_spread_v2.dataset import build_sample,true_label
from src.research96.msr_spread_v2.trainer import Preprocessor,seed_everything
from src.research96.msr_spread_v2.model import MSRSpreadV2
from src.research96.msr_spread_v2.losses import objectives
from src.research96.msr_spread_v2.experiments.component_adjudication.arms import make_model,stage_mode,arm_objectives
from src.research96.msr_spread_v2.experiments.component_adjudication.profile import scientific_config

def data():
    store=fixture.__wrapped__();sample=build_sample(store,'2023-02-10');label=true_label(store,sample.target_day)
    pre=Preprocessor().fit(sample.x[None],label[None],[sample.target_day])
    return sample.names,pre.stats,torch.tensor(pre.transform(sample.x[None])),torch.tensor(label[None],dtype=torch.float32)

def test_full_is_original_graph_and_objective_exact():
    names,stats,x,y=data();cfg=scientific_config(64)
    seed_everything(cfg.seed);a=MSRSpreadV2(names,cfg,stats).eval()
    seed_everything(cfg.seed);b=make_model(names,cfg,stats,'FULL_V2').eval()
    assert type(b) is MSRSpreadV2
    with torch.no_grad():
        u=a(x,scenarios=32);v=b(x,scenarios=32)
        assert torch.equal(u['spread_scenarios'],v['spread_scenarios'])
        assert torch.equal(objectives(u,y,a)['total'],arm_objectives(v,y,b)['total'])

def test_no_dynamic_dep_and_stage_b_noop():
    names,stats,x,y=data();m=make_model(names,scientific_config(64),stats,'NO_DYNAMIC_DEP').eval()
    with torch.no_grad(): out=m(x,scenarios=32)
    assert all(torch.count_nonzero(out['dependence'][k])==0 for k in ('rho','phiL','phiH'))
    stage_mode(m,'B');assert not any(p.requires_grad for p in m.parameters())

def test_fixed_gates_complementary_invariant_across_days():
    names,stats,x,y=data();m=make_model(names,scientific_config(64),stats,'FIXED_BUSINESS_GATES').eval()
    e=m.embeddings(x);l,h,info=m.router(e)
    for g in info:
        assert torch.allclose(l[g]+h[g],e[g],atol=2e-6)
        assert torch.allclose(info[g]['cutoff'],torch.tensor([8.]))
        assert info[g]['delta']==0
    stage_mode(m,'C');assert not any(p.requires_grad for p in m.router.parameters())

def test_no_low_to_high_context_has_no_context_gradient():
    names,stats,x,y=data();m=make_model(names,scientific_config(64),stats,'NO_LOW_TO_HIGH_CONTEXT').eval()
    width=m.high.temporal.project.in_features
    inp=torch.randn(1,96,width,requires_grad=True);cw=torch.ones(1,96,1,requires_grad=True)
    q,_,_=m.high(inp,cw,torch.ones_like(cw),torch.ones_like(cw))
    q.sum().backward()
    assert torch.count_nonzero(inp.grad[...,-7:])==0
    assert cw.grad is None

def test_no_brier_only_coefficient_changes():
    names,stats,x,y=data();m=make_model(names,scientific_config(64),stats,'NO_BRIER0').eval()
    with torch.no_grad():
        out=m(x,scenarios=32);a=objectives(out,y,m);b=arm_objectives(out,y,m)
    assert torch.allclose(a['total']-b['total'],.30*a['brier'],atol=1e-6)
    assert torch.equal(a['stage_a'],b['stage_a'])

def test_direct_native_monotone_capacity_and_no_t24():
    names,stats,x,y=data();cfg=scientific_config(64);m=make_model(names,cfg,stats,'DIRECT96').eval()
    assert all(not names[i].startswith(('hist_low_','hist_high_')) for i in m.ids)
    assert not hasattr(m,'dependence') and not hasattr(m,'router')
    with torch.no_grad(): out=m(x,scenarios=32)
    assert torch.all(out['direct_q'].diff(dim=-1)>0)
    assert out['spread_scenarios'].shape==(1,32,96)
    baseline=sum(p.numel() for p in MSRSpreadV2(names,cfg,stats).parameters())
    assert .8<=sum(p.numel() for p in m.parameters())/baseline<=1.2
    stage_mode(m,'B');assert not any(p.requires_grad for p in m.parameters())

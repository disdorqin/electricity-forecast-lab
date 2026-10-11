"""docs86 isolated controls. FULL_V2 is the original class, not a patched forward."""
import math
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from ...config import FAMILIES
from ...model import MSRSpreadV2
from ...low_branch import LowBranch
from ...quantile_head import quantile
from ...dependence import _normal_base
from ...losses import objectives, marginal_score
from ...reconstruct import scenario_crps, soft_probability, soft_median, evaluation_outputs
from ...trainer import set_stage

class FixedGate(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original=original
    def forward(self,e):
        c0=4+24*torch.sigmoid(self.original.c0_raw.detach())
        tau=.5+7.5*torch.sigmoid(self.original.tau_raw.detach())
        cutoff=c0.expand(e.shape[0])
        gate=torch.sigmoid((cutoff[:,None]-torch.arange(96,device=e.device,dtype=e.dtype))/tau)
        return gate,dict(cutoff=cutoff,c0=c0,tau=tau,delta=e.new_tensor(0.),gate=gate)

class IndependentDependence(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.original=original
    @property
    def coefficients(self):
        return self.original.coefficients
    def forward(self,context,m):
        curves={k:context.new_zeros((context.shape[0],96)) for k in ("rho","phiL","phiH")}
        curves['coefficients']=context.new_zeros((context.shape[0],3,6))
        u,v,_,_=self.original.ranks(curves,m)
        return curves,u,v

class NoLowContext(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.original=original
    @property
    def temporal(self): return self.original.temporal
    @property
    def tabm(self): return self.original.tabm
    @property
    def head(self): return self.original.head
    def forward(self,x,low_width,ramp_norm,proxy_mask,u_context=None):
        # Seven distribution summary columns and the fusion width are the ONLY
        # current High-to-Low conditioning paths. Keep dimensions/capacity fixed.
        masked=torch.cat((x[...,:-7],torch.zeros_like(x[...,-7:])),dim=-1)
        return self.original(masked,torch.zeros_like(low_width),ramp_norm,proxy_mask,u_context)

class Direct96(nn.Module):
    """One 80-wide long TCN + official width64/k4 TabM; no T24 input/output split."""
    def __init__(self,names,cfg,stats):
        super().__init__()
        self.cfg,self.names=cfg,tuple(names)
        # Same primitive Native96 data. Only deterministic T24-derived anchors
        # are excluded; original mature Spread/DA/RT history remains available.
        self.ids=[i for i,n in enumerate(names) if not n.startswith(('hist_low_','hist_high_'))]
        self.history_ids=[i for i,n in enumerate(names) if n.startswith('hist_spread_')]
        self.branch=LowBranch(len(self.ids),replace(cfg,low_width=80))
        self.register_buffer('spread_scale',torch.tensor(stats['spread_scale'],dtype=torch.float32))
    def forward(self,x,da=None,scenarios=None,components_only=False,**unused):
        q,h,g=self.branch(x[...,self.ids],x[...,self.history_ids].std(-1,unbiased=False,keepdim=True))
        out=dict(direct_q=q,direct_q_raw=q*self.spread_scale,direct_repr=h,fusion_direct=g)
        if components_only: return out
        m=scenarios or (self.cfg.m_train if self.training else self.cfg.m_eval)
        z=_normal_base(m,self.cfg.seed).to(device=x.device,dtype=x.dtype)[:,:,0]
        u=.5*(1+torch.erf(z/math.sqrt(2)))
        s,clamp=quantile(q,u.unsqueeze(0).expand(x.shape[0],-1,-1))
        out.update(spread_scenarios=s*self.spread_scale,clamp_fraction=clamp)
        if da is not None: out.update(evaluation_outputs(out['spread_scenarios'],da))
        return out
    def gradient_groups(self):
        return {'direct_'+n:list(getattr(self.branch,n).parameters()) for n in ('temporal','tabm','head')}

def make_model(names,cfg,stats,arm):
    if arm=='DIRECT96':
        model=Direct96(names,cfg,stats)
    else:
        model=MSRSpreadV2(names,cfg,stats)
        if arm=='FIXED_BUSINESS_GATES':
            for g in FAMILIES: model.router.gates[g]=FixedGate(model.router.gates[g])
        elif arm=='NO_DYNAMIC_DEP': model.dependence=IndependentDependence(model.dependence)
        elif arm=='NO_LOW_TO_HIGH_CONTEXT': model.high=NoLowContext(model.high)
        elif arm not in ('FULL_V2','NO_BRIER0'): raise ValueError(arm)
    model.arm=arm
    return model

def stage_mode(model,stage):
    arm=getattr(model,'arm','FULL_V2')
    if arm=='DIRECT96':
        for p in model.parameters(): p.requires_grad_(stage!='B')
        model.train(stage!='B')
    else:
        set_stage(model,stage)
        if arm=='NO_DYNAMIC_DEP':
            for p in model.dependence.parameters(): p.requires_grad_(False)
        if arm=='FIXED_BUSINESS_GATES':
            for p in model.router.parameters(): p.requires_grad_(False)

def arm_objectives(out,labels,model):
    arm=getattr(model,'arm','FULL_V2')
    if arm!='DIRECT96':
        result=objectives(out,labels,model)
        if arm=='NO_BRIER0' and 'final' in result:
            result['final']=result['crps']+.25*result['point']
            result['total']=result['final']+.20*result['low']+.20*result['high']+.01*result['dep_smooth']
            result['stage_b']=result['final']+.01*result['dep_smooth']
        return result
    target=labels[...,0]/model.spread_scale
    result=dict(stage_a=marginal_score(target,out['direct_q']))
    if 'spread_scenarios' not in out: return result
    s=out['spread_scenarios']/model.spread_scale
    result.update(crps=scenario_crps(s,target),
        brier=(soft_probability(s,model.cfg.tau0)-(labels[...,0]>0).float()).square().mean(),
        point=F.smooth_l1_loss(soft_median(s,model.cfg.rank_bandwidth),target))
    result['final']=result['crps']+.30*result['brier']+.25*result['point']
    result['total']=result['stage_b']=result['final']
    return result

def arm_gradient_audit(model,losses):
    from ...trainer import gradient_audit
    if getattr(model,'arm','FULL_V2') in ('FULL_V2','NO_BRIER0','NO_LOW_TO_HIGH_CONTEXT'):
        return gradient_audit(model,losses)
    groups=model.gradient_groups()
    params=[p for p in model.parameters() if p.requires_grad]
    pos={id(p):i for i,p in enumerate(params)}
    vectors,norms={},{}
    for term,loss in losses.items():
        if term not in ('crps','brier','point','final','low','high','dep_smooth','total'): continue
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True) if loss.requires_grad else [None]*len(params)
        vectors[term],norms[term]={},{}
        for name,ps in groups.items():
            parts=[]
            for p in ps:
                i=pos.get(id(p)); grad=grads[i] if i is not None else None
                parts.append((torch.zeros_like(p) if grad is None else grad.detach()).flatten())
            v=torch.cat(parts); vectors[term][name]=v; norms[term][name]=float(v.double().norm())
    cosines={}
    for group in groups:
        for term in ('low','high'):
            if term not in vectors: continue
            a,b=vectors['final'][group],vectors[term][group]; d=a.norm()*b.norm()
            cosines[group+'/'+term]=float(torch.dot(a,b)/d) if d>0 else None
    intentional=[g for g,ps in groups.items() if not any(p.requires_grad for p in ps)]
    failed=[g for g,v in norms['final'].items() if g not in intentional and (not math.isfinite(v) or v<=1e-12)]
    if failed: raise RuntimeError('non-ablated final gradient paths absent: '+str(failed))
    return dict(norms=norms,final_vs_component_cosine=cosines,reachability='PASS',
                intentionally_ablated_groups=intentional,threshold=1e-12)

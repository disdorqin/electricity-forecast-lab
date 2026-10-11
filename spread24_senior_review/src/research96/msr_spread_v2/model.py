"""One Native-96 V2 graph, without training logic or mutable reference imports."""
import torch
from torch import nn
from .config import Config,FAMILIES,BYPASS
from .contracts import assert_feature_names
from .feature_registry import registry
from .embeddings import FamilyEmbeddings
from .spectral_router import SpectralRouter
from .low_branch import LowBranch
from .high_branch import HighBranch
from .quantile_head import low_summary,quantile
from .dependence import DynamicDependence
from .reconstruct import evaluation_outputs

class RegimeContext(nn.Module):
    """K=1 identity soft simplex hook; no classifier/state/transition predictor."""
    k = 1
    def forward(self,x):
        return x.new_ones((*x.shape[:2],1))

class MSRSpreadV2(nn.Module):
    def __init__(self,names,cfg=None,stats=None):
        super().__init__()
        self.cfg = cfg or Config()
        assert_feature_names(names)
        self.names = tuple(names)
        reg = registry()
        self.low_ids = [i for i,n in enumerate(names) if reg[n]["role"]=="low_anchor"]
        self.high_ids = [i for i,n in enumerate(names) if reg[n]["role"]=="high_anchor"]
        # Legal raw context augments routed representations; primitive semantics retained.
        self.low_raw_ids = [i for i,n in enumerate(names) if n in
            ("target_da","target_da_mean","target_da_std","target_forecast_load",
             "target_residual_load","target_renewable_share","target_bidding_space","target_load_gap")]
        self.high_raw_ids = [i for i,n in enumerate(names) if
            n.startswith(("d1_actual_","target_forecast_","d1_error_","d1_","target_bidding_space_"))
            and reg[n]["role"]=="spectral"]
        self.proxy_id = names.index("price_proxy_mask")
        self.ramp_id = names.index("target_da_ramp15")
        c = self.cfg
        self.embeddings = FamilyEmbeddings(names,c.family_width)
        self.router = SpectralRouter(c.family_width)
        base_width = (len(FAMILIES)+1)*c.family_width
        self.low = LowBranch(base_width+7+len(self.low_raw_ids),c)
        self.high = HighBranch(base_width+7+len(self.high_raw_ids)+7,c)
        self.regime_context = RegimeContext()
        # Pooled fused Low/High plus known DA/grid/price-mask context.
        self.dep_raw_ids = sorted(set(self.low_raw_ids+self.high_raw_ids+[self.proxy_id]))
        self.dependence = DynamicDependence(c.low_width+c.high_width+len(self.dep_raw_ids),c.k_dep,c.seed)
        stats = stats or dict(component_center=[0.,0.],component_scale=[1.,1.],spread_scale=1.)
        self.register_buffer("component_center",torch.tensor(stats["component_center"],dtype=torch.float32))
        self.register_buffer("component_scale",torch.tensor(stats["component_scale"],dtype=torch.float32))
        self.register_buffer("spread_scale",torch.tensor(stats["spread_scale"],dtype=torch.float32))
    def forward(self,x,da=None,scenarios=None,components_only=False,u_context=None):
        if x.ndim!=3 or x.shape[1:]!=(96,len(self.names)) or not torch.isfinite(x).all():
            raise ValueError("preprocessed finite [B,96,C] required")
        e = self.embeddings(x)
        low_route,high_route,gates = self.router(e)
        low_input = torch.cat([low_route[g] for g in FAMILIES]+
                              [e[BYPASS],x[...,self.low_ids],x[...,self.low_raw_ids]],dim=-1)
        uncertainty = x[...,self.low_ids].std(-1,unbiased=False,keepdim=True)
        ql,hl,gl = self.low(low_input,uncertainty)
        context = low_summary(ql)
        high_input = torch.cat([high_route[g] for g in FAMILIES]+
                               [e[BYPASS],x[...,self.high_ids],x[...,self.high_raw_ids],context],dim=-1)
        qh,hh,gh = self.high(high_input,context[...,-2:-1],x[...,self.ramp_id:self.ramp_id+1].abs(),
                            x[...,self.proxy_id:self.proxy_id+1],u_context=u_context)
        out = dict(low_q=ql,high_q=qh,low_repr=hl,high_repr=hh,
                   fusion_low=gl,fusion_high=gh,spectral=gates,regime_context=self.regime_context(x))
        if components_only:
            return out
        m = scenarios or (self.cfg.m_train if self.training else self.cfg.m_eval)
        pooled = torch.cat((hl.mean(1),hh.mean(1),x[...,self.dep_raw_ids].mean(1)),-1)
        curves,u,v = self.dependence(pooled,m)
        l,cl = quantile(ql,u)
        h,ch = quantile(qh,v)
        # Each component has its own training-only transform; invert before summation.
        l = l*self.component_scale[0]+self.component_center[0]
        h = h*self.component_scale[1]+self.component_center[1]
        s = l+h
        out.update(low_scenarios=l,high_scenarios=h,spread_scenarios=s,dependence=curves,
                   clamp_fraction=(cl+ch)/2,
                   low_q_raw=ql*self.component_scale[0]+self.component_center[0],
                   high_q_raw=qh*self.component_scale[1]+self.component_center[1])
        if da is not None:
            out.update(evaluation_outputs(s,da))
        return out

    def gradient_groups(self):
        groups = {g+"_gate":list(self.router.gates[g].parameters()) for g in FAMILIES}
        for side in ("low","high"):
            b = getattr(self,side)
            for name in ("temporal","tabm","head"):
                groups[side+"_"+name] = list(getattr(b,name).parameters())
        groups["dependence"] = list(self.dependence.coefficients.parameters())
        return groups

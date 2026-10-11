"""Exact frozen docs/84 objective; no sign-class weighting/threshold selection."""
import torch
from torch.nn import functional as F
from .config import Q_KNOTS
from .reconstruct import scenario_crps,soft_median,soft_probability
from .dependence import smoothness

def integration_weights(reference):
    lv = reference.new_tensor(Q_KNOTS)
    edges = torch.cat((lv.new_zeros(1),(lv[:-1]+lv[1:])/2,lv.new_ones(1)))
    return edges.diff() # midpoint cells normalized to one, including outer tail cells

def marginal_score(y,q):
    lv = q.new_tensor(Q_KNOTS)
    err = y[...,None]-q
    pin = torch.maximum(lv*err,(lv-1)*err)
    return 2*(pin*integration_weights(q)).sum(-1).mean()

def component_losses(out,labels,model):
    l = (labels[...,1]-model.component_center[0])/model.component_scale[0]
    h = (labels[...,2]-model.component_center[1])/model.component_scale[1]
    return dict(low=marginal_score(l,out["low_q"]),high=marginal_score(h,out["high_q"]))

def objectives(out,labels,model):
    result = component_losses(out,labels,model)
    if "spread_scenarios" not in out:
        result["stage_a"] = result["low"]+result["high"]
        return result
    s = out["spread_scenarios"]/model.spread_scale
    y = labels[...,0]/model.spread_scale
    result.update(crps=scenario_crps(s,y),
                  brier=(soft_probability(s,model.cfg.tau0)-(labels[...,0]>0).float()).square().mean(),
                  point=F.smooth_l1_loss(soft_median(s,model.cfg.rank_bandwidth),y),
                  dep_smooth=smoothness(out["dependence"]))
    result["final"] = result["crps"]+.30*result["brier"]+.25*result["point"]
    result["total"] = result["final"]+.20*result["low"]+.20*result["high"]+.01*result["dep_smooth"]
    result["stage_a"] = result["low"]+result["high"]
    result["stage_b"] = result["final"]+.01*result["dep_smooth"]
    return result

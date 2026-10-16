"""Scenario identity and differentiable vs empirical outputs are separate."""
import torch
from .config import Q_KNOTS

def scenario_crps(s,y):
    # Exact sorted-sample formula for 0.5*mean_{m,n}|s_m-s_n|, O(M log M).
    m = s.shape[1]
    sorted_s = s.sort(dim=1).values
    weights = (2*torch.arange(1,m+1,device=s.device,dtype=s.dtype)-m-1).view(1,m,1)
    pair_half = (weights*sorted_s).sum(1)/(m*m)
    return ((s-y[:,None,:]).abs().mean(1)-pair_half).mean()

def soft_median(s,bandwidth=.08):
    m = s.shape[1]
    ranks = (torch.arange(m,device=s.device,dtype=s.dtype)+.5)/m
    weights = torch.softmax(-.5*((ranks-.5)/bandwidth)**2,dim=0).view(1,m,1)
    return (s.sort(dim=1).values*weights).sum(1)

def soft_probability(s,tau=.10):
    return torch.sigmoid(s/tau).mean(1)

def evaluation_outputs(s,da):
    q = torch.quantile(s,s.new_tensor(Q_KNOTS),dim=1).permute(1,2,0)
    median = q[...,4]
    return dict(spread_median=median,spread_quantiles=q,p_positive=(s>0).float().mean(1),
                rt_median=da+median,rt_quantiles=da[...,None]+q)

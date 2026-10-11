"""Structural monotone median +/- softplus increments; continuous inverse CDF."""
import torch
from torch import nn
from torch.nn import functional as F
from .config import Q_KNOTS

class QuantileHead(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.linear = nn.Linear(width,9)
        with torch.no_grad():
            self.linear.bias[1:].fill_(-1.)
    def forward(self,h):
        raw = self.linear(h)
        m = raw[...,:1]
        inc = F.softplus(raw[...,1:])+1e-5
        lower = m-torch.cumsum(inc[...,:4],-1)
        upper = m+torch.cumsum(inc[...,4:],-1)
        return torch.cat((lower.flip(-1),m,upper),dim=-1)

def quantile(knots,u):
    """knots[B,96,9], ranks[B,M,96] -> values[B,M,96]; u clamped/audited."""
    if not torch.isfinite(knots).all() or not torch.isfinite(u).all():
        raise FloatingPointError("nonfinite inverse CDF input")
    lv = knots.new_tensor(Q_KNOTS)
    clamp = ((u<lv[0])|(u>lv[-1])).float().mean()
    v = u.clamp(float(Q_KNOTS[0]),float(Q_KNOTS[-1])).transpose(1,2).contiguous()
    i = torch.searchsorted(lv,v).sub(1).clamp(0,len(Q_KNOTS)-2)
    lo,hi = knots.gather(-1,i),knots.gather(-1,i+1)
    w = (v-lv[i])/(lv[i+1]-lv[i])
    return (lo+w*(hi-lo)).transpose(1,2),clamp

def low_summary(q):
    nodes = q[...,[2,3,4,5,6]]
    return torch.cat((nodes,(q[...,6]-q[...,2]).unsqueeze(-1),
                      (q[...,5]-q[...,3]).unsqueeze(-1)),dim=-1)

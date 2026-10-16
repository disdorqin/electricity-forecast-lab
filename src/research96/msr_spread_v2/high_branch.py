"""Residual history + legal fast context + differentiable Low summary -> Q_H."""
import torch
from torch import nn
from .temporal import HighTCN
from .tabm_encoder import TabMEncoder
from .quantile_head import QuantileHead

class HighBranch(nn.Module):
    def __init__(self,n_features,cfg):
        super().__init__()
        w = cfg.high_width
        self.temporal = HighTCN(n_features,w,cfg.dropout)
        self.tabm = TabMEncoder(n_features,cfg.tab_width,w,cfg.tab_k,cfg.dropout)
        self.fusion = nn.Sequential(nn.Linear(2*w+3,24),nn.GELU(),nn.Linear(24,1))
        self.norm = nn.LayerNorm(w)
        self.head = QuantileHead(w)
    def forward(self,x,low_width,ramp_norm,proxy_mask,u_context=None):
        if u_context is not None:
            raise ValueError("U is interface-only, not authorized in Core V2")
        a,b = self.temporal(x),self.tabm(x)
        gate = torch.sigmoid(self.fusion(torch.cat((a,b,low_width,ramp_norm,proxy_mask),-1)))
        h = self.norm(gate*a+(1-gate)*b)
        return self.head(h),h,gate

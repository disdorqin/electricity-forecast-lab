"""Long TCN + local TabM bounded convex fusion -> Q_L."""
import torch
from torch import nn
from .temporal import LowTCN
from .tabm_encoder import TabMEncoder
from .quantile_head import QuantileHead

class LowBranch(nn.Module):
    def __init__(self,n_features,cfg):
        super().__init__()
        w = cfg.low_width
        self.temporal = LowTCN(n_features,w,cfg.dropout)
        self.tabm = TabMEncoder(n_features,cfg.tab_width,w,cfg.tab_k,cfg.dropout)
        self.fusion = nn.Sequential(nn.Linear(2*w+1,24),nn.GELU(),nn.Linear(24,1))
        self.norm = nn.LayerNorm(w)
        self.head = QuantileHead(w)
    def forward(self,x,uncertainty):
        a,b = self.temporal(x),self.tabm(x)
        gate = torch.sigmoid(self.fusion(torch.cat((a,b,uncertainty),-1)))
        h = self.norm(gate*a+(1-gate)*b)
        return self.head(h),h,gate

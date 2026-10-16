"""Native-96 shallow phase-aligned residual TCNs; no attention/upsampling."""
import torch
from torch import nn

class ResidualTCNBlock(nn.Module):
    def __init__(self,width,dilation,dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        y = self.norm(x).transpose(1,2)
        y = self.dropout(self.activation(self.conv(y))).transpose(1,2)
        return x+.1*y

class TCN(nn.Module):
    def __init__(self,n_features,width,dilations,dropout):
        super().__init__()
        self.project = nn.Linear(n_features,width)
        self.blocks = nn.ModuleList([ResidualTCNBlock(width,d,dropout) for d in dilations])
    def forward(self,x):
        if x.shape[1]!=96:
            raise ValueError("TCN requires native 96")
        x = self.project(x)
        for block in self.blocks:
            x = block(x)
        return x

class LowTCN(TCN):
    def __init__(self,n_features,width=48,dropout=.05):
        super().__init__(n_features,width,(1,2,4,8,16,32),dropout)

class HighTCN(TCN):
    def __init__(self,n_features,width=64,dropout=.05):
        super().__init__(n_features,width,(1,2,4,8),dropout)

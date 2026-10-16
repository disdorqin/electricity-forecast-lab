"""V2-local official TabM mini; reference TafM/models/tabular_encoder.py.
Own native-96 wrapper, no frozen runtime import or substitute backend.
"""
import torch
from torch import nn
import tabm

class TabMEncoder(nn.Module):
    def __init__(self,n_features,width,out_width,k=4,dropout=.05):
        super().__init__()
        self.k,self.width = k,width
        self.backbone = tabm.TabM.make(n_num_features=n_features,d_out=None,
            n_blocks=2,d_block=width,dropout=dropout,k=k,arch_type="tabm-mini",activation="GELU")
        self.project = nn.Linear(width,out_width)
    def forward(self,x):
        b,t,c = x.shape
        members = self.backbone(x.reshape(b*t,c))
        if members.shape != (b*t,self.k,self.width):
            raise RuntimeError("installed official TabM API shape mismatch")
        return self.project(members.mean(1)).reshape(b,t,-1)

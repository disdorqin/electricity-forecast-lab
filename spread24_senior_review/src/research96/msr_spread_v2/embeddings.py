"""Independent business-family scalar projections at native 96."""
import torch
from torch import nn
from .config import FAMILIES, BYPASS
from .feature_registry import indices

class FamilyEmbeddings(nn.Module):
    def __init__(self,names,width):
        super().__init__()
        self.ids = {g:indices(g,names) for g in (*FAMILIES,BYPASS)}
        if any(not v for v in self.ids.values()):
            raise ValueError("all five spectral families and bypass required")
        self.projections = nn.ModuleDict({g:nn.Linear(len(ids),width) for g,ids in self.ids.items()})
    def forward(self,x):
        return {g:self.projections[g](x[...,ids]) for g,ids in self.ids.items()}

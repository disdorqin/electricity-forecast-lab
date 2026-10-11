"""Family-specific bounded differentiable DCT-II low-pass + exact complement."""
import math
import torch
from torch import nn
from .config import FAMILIES

def dct_matrix(n=96):
    t = torch.arange(n,dtype=torch.float64)+.5
    k = torch.arange(n,dtype=torch.float64)
    a = torch.cos(math.pi*k[:,None]*t[None,:]/n)*math.sqrt(2/n)
    a[0] /= math.sqrt(2)
    return a

class FamilyGate(nn.Module):
    def __init__(self,width):
        super().__init__()
        # T24 half-gain characteristic bin 2*96/24=8.
        self.c0_raw = nn.Parameter(torch.tensor(math.log((8-4)/(28-8))))
        self.tau_raw = nn.Parameter(torch.tensor(math.log((2-.5)/(8-2))))
        self.delta_raw = nn.Parameter(torch.tensor(0.0))
        self.context = nn.Linear(2*width,1)
        nn.init.normal_(self.context.weight,std=.02)
        nn.init.zeros_(self.context.bias)
    def forward(self,e):
        c0 = 4+24*torch.sigmoid(self.c0_raw)
        tau = .5+7.5*torch.sigmoid(self.tau_raw)
        delta = 3*torch.sigmoid(self.delta_raw)
        context = torch.cat((e.mean(1),e.std(1,unbiased=False)),dim=-1)
        cutoff = c0+delta*torch.tanh(self.context(context)).squeeze(-1)
        bins = torch.arange(96,device=e.device,dtype=e.dtype)
        gate = torch.sigmoid((cutoff[:,None]-bins)/tau)
        return gate,dict(cutoff=cutoff,c0=c0,tau=tau,delta=delta,gate=gate)

class SpectralRouter(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.register_buffer("dct",dct_matrix().float())
        self.gates = nn.ModuleDict({g:FamilyGate(width) for g in FAMILIES})
    def transform(self,x):
        return torch.einsum("kt,btd->bkd",self.dct,x)
    def inverse(self,a):
        return torch.einsum("kt,bkd->btd",self.dct,a)
    def forward(self,embeddings):
        low,high,info = {},{},{}
        for g in FAMILIES:
            e = embeddings[g]
            a = self.transform(e)
            gate,record = self.gates[g](e)
            low[g] = self.inverse(a*gate[...,None])
            high[g] = self.inverse(a*(1-gate[...,None]))
            info[g] = record
        return low,high,info

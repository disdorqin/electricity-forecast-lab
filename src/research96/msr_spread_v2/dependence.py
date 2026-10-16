"""Six smooth DCT basis curves and seeded Sobol Gaussian-copula-like process."""
import math
from functools import lru_cache
import torch
from torch import nn
from .spectral_router import dct_matrix

@lru_cache(maxsize=32)
def _normal_base(m,seed):
    u = torch.quasirandom.SobolEngine(192,scramble=True,seed=seed).draw(m,dtype=torch.float64)
    u = u.clamp(1e-7,1-1e-7)
    return (math.sqrt(2)*torch.erfinv(2*u-1)).reshape(m,96,2)

class DynamicDependence(nn.Module):
    def __init__(self,context_dim,k=6,seed=20261009):
        super().__init__()
        if k != 6:
            raise ValueError("docs/84 K_DEP frozen at 6")
        self.seed = seed
        self.register_buffer("basis",dct_matrix()[:k].T.float())
        self.coefficients = nn.Sequential(nn.Linear(context_dim,32),nn.GELU(),nn.Linear(32,3*k))
    def curves(self,context):
        a = self.coefficients(context).reshape(-1,3,6)
        smooth = torch.einsum("tk,bck->bct",self.basis,a)
        return dict(rho=.95*torch.tanh(smooth[:,0]),
                    phiL=.98*torch.sigmoid(smooth[:,1]),
                    phiH=.95*torch.tanh(smooth[:,2]),coefficients=a)
    def ranks(self,curves,m):
        rho,pl,ph = (curves[k] for k in ("rho","phiL","phiH"))
        eps = _normal_base(m,self.seed).to(device=rho.device,dtype=rho.dtype)
        el,eh = eps[:,:,0].unsqueeze(0),eps[:,:,1].unsqueeze(0)
        innovation = rho[:,None,:]*el+torch.sqrt(1-rho[:,None,:]**2)*eh
        l,h = [el[:,:,0].expand(rho.shape[0],-1)],[innovation[:,:,0]]
        for t in range(1,96):
            fl,fh = pl[:,t,None],ph[:,t,None]
            l.append(fl*l[-1]+torch.sqrt(1-fl**2)*el[:,:,t])
            h.append(fh*h[-1]+torch.sqrt(1-fh**2)*innovation[:,:,t])
        zl,zh = torch.stack(l,-1),torch.stack(h,-1)
        u,v = .5*(1+torch.erf(zl/math.sqrt(2))),.5*(1+torch.erf(zh/math.sqrt(2)))
        return u,v,zl,zh
    def forward(self,context,m):
        curves = self.curves(context)
        u,v,zl,zh = self.ranks(curves,m)
        return curves,u,v

def smoothness(curves):
    return (curves["rho"].diff(dim=-1).square().mean()
            +.5*curves["phiL"].diff(dim=-1).square().mean()
            +.5*curves["phiH"].diff(dim=-1).square().mean())

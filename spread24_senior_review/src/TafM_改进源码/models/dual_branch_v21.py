"""Canonical V2.1 KPI-aligned dual branch; member axis survives to both heads."""
from __future__ import annotations
import torch
from torch import nn
from .task_adapters import ARCHITECTURE_MODES, MemberWiseTaskAdaptersV21

READOUT_MODES = {"shared", "segment_bias", "segment_heads"}
# Fixed pre-existing business segments (hours 1-8 / 9-16 / 17-24) -> 0-based index slices.
SEGMENT_SLICES = ((0, 8), (8, 16), (16, 24))
SEGMENT_HOURS = {"H1": tuple(range(1, 9)), "H2": tuple(range(9, 17)), "H3": tuple(range(17, 25))}


class DualBranchV21(nn.Module):
    legacy_v20=False
    def __init__(self,d_tab:int,d_time:int,d_task:int=32,mode="A2",*,tabular_encoder:nn.Module,
                 temporal_encoder:nn.Module,k:int,magnitude_scale:float,magnitude_eps:float=1e-8,
                 architecture_mode:str="full_current",direction_fusion_alpha:float|None=None,
                 direction_tabular_mode:str="current",direction_horizon_gate_mode:str="current",
                 direction_readout_mode:str="shared"):
        super().__init__()
        if k<1 or magnitude_scale<=0 or magnitude_eps<=0: raise ValueError("invalid V2.1 model dimensions/scales")
        if architecture_mode not in ARCHITECTURE_MODES:
            raise ValueError("architecture_mode must be full_current/tabular_only/temporal_only")
        import tabm
        self.tabular_encoder=tabular_encoder; self.temporal_encoder=temporal_encoder
        self.architecture_mode=architecture_mode
        if direction_tabular_mode not in {"current", "strong_only", "weak_only"}:
            raise ValueError("direction_tabular_mode must be current/strong_only/weak_only")
        self.direction_tabular_mode = direction_tabular_mode
        if direction_horizon_gate_mode not in {"current", "fixed_08", "global_learnable"}:
            raise ValueError("direction_horizon_gate_mode must be current/fixed_08/global_learnable")
        if direction_horizon_gate_mode != "current" and (
            direction_tabular_mode != "current" or architecture_mode != "full_current" or
            direction_fusion_alpha is None or abs(float(direction_fusion_alpha)-.8)>1e-12
        ):
            raise ValueError("non-current horizon gate requires current tabular mode, full_current architecture, and fixed alpha=.8")
        self.direction_horizon_gate_mode = direction_horizon_gate_mode
        self.direction_global_gate_logit = (nn.Parameter(torch.tensor(1.3862943611198906))
            if direction_horizon_gate_mode == "global_learnable" else None)
        self.direction_fusion_alpha=(None if direction_fusion_alpha is None
                                     else float(direction_fusion_alpha))
        self.adapters=MemberWiseTaskAdaptersV21(d_tab,d_time,d_task,mode,architecture_mode=architecture_mode,
            direction_fusion_alpha=direction_fusion_alpha)
        if direction_readout_mode not in READOUT_MODES:
            raise ValueError("direction_readout_mode must be shared/segment_bias/segment_heads")
        self.direction_readout_mode = direction_readout_mode
        # E2-E1 experiment-only Direction readout. shared reproduces the canonical single member-wise
        # head exactly; segment_bias keeps that head and adds 3 zero-init logit offsets; segment_heads
        # replaces it with exactly three independent member-wise heads. Magnitude is never touched.
        self.direction_head = (tabm.LinearEnsemble(d_task,1,k=k)
                               if direction_readout_mode != "segment_heads" else None)
        self.direction_hour_bias = (nn.Parameter(torch.zeros(3))
                                    if direction_readout_mode == "segment_bias" else None)
        self.direction_segment_heads = ([tabm.LinearEnsemble(d_task,1,k=k) for _ in range(3)]
                                        if direction_readout_mode == "segment_heads" else None)
        if self.direction_segment_heads is not None:
            self.direction_segment_heads = nn.ModuleList(self.direction_segment_heads)
        self.magnitude_head=tabm.LinearEnsemble(d_task,1,k=k)
        self.k=int(k);self.magnitude_scale=float(magnitude_scale);self.magnitude_eps=float(magnitude_eps);self.mode=mode

    def _direction_logits(self,h_dir:torch.Tensor) -> torch.Tensor:
        """Return Direction member logits [B,k,24]; segment slicing preserves the canonical order."""
        b,k,h,d=h_dir.shape
        if self.direction_readout_mode == "segment_heads":
            hs = h_dir.permute(0,2,1,3).reshape(b,h,k,d)
            pieces=[]
            for (lo,hi),head in zip(SEGMENT_SLICES, self.direction_segment_heads):
                seg=hs[:,lo:hi].reshape(b*(hi-lo),k,d)
                pieces.append(head(seg).squeeze(-1).reshape(b,hi-lo,k))
            return torch.cat(pieces,dim=1).permute(0,2,1)
        # shared and segment_bias both use the one shared member-wise head.
        z=self.direction_head(h_dir.permute(0,2,1,3).reshape(b*h,k,d)).squeeze(-1).reshape(b,h,k).permute(0,2,1)
        if self.direction_readout_mode == "segment_bias":
            seg_index=torch.tensor([0]*8+[1]*8+[2]*8,device=z.device)
            z = z + self.direction_hour_bias[seg_index]
        return z

    def direction_readout_audit(self) -> dict:
        audit={"mode": self.direction_readout_mode,
               "segment_hour_map": {name: list(hours) for name,hours in SEGMENT_HOURS.items()},
               "shared_direction_head": self.direction_head is not None,
               "n_segment_bias": 0, "n_segment_heads": 0,
               "segment_bias": None, "segment_head_param_norms": None, "segment_head_param_counts": None}
        if self.direction_hour_bias is not None:
            audit["n_segment_bias"]=3
            audit["segment_bias"]=[float(x) for x in self.direction_hour_bias.detach().cpu()]
        if self.direction_segment_heads is not None:
            audit["n_segment_heads"]=len(self.direction_segment_heads)
            audit["segment_head_param_norms"]=[float(torch.linalg.vector_norm(
                torch.cat([p.detach().reshape(-1) for p in head.parameters()]))) for head in self.direction_segment_heads]
            audit["segment_head_param_counts"]=[sum(p.numel() for p in head.parameters()) for head in self.direction_segment_heads]
        return audit

    def forward(self,x_hist,x_future,*,direction_protected: bool=False):
        if hasattr(self.tabular_encoder, "forward_components"):
            h_strong,h_weak,gate,h_tab=self.tabular_encoder.forward_components(x_future)
            if self.direction_tabular_mode == "strong_only":
                h_tab_direction = h_strong
            elif self.direction_tabular_mode == "weak_only":
                h_tab_direction = h_weak.unsqueeze(1).expand(-1,self.k,-1,-1)
            else:
                h_tab_direction = None
            if self.direction_horizon_gate_mode == "fixed_08":
                h_tab_direction = .8*h_strong + .2*h_weak.unsqueeze(1).expand(-1,self.k,-1,-1)
            elif self.direction_horizon_gate_mode == "global_learnable":
                g = torch.sigmoid(self.direction_global_gate_logit)
                h_tab_direction = g*h_strong + (1-g)*h_weak.unsqueeze(1).expand(-1,self.k,-1,-1)
        else:
            if self.direction_tabular_mode != "current":
                raise TypeError("Strong/Weak Direction routing requires TabularEncoder.forward_components")
            h_tab=self.tabular_encoder(x_future);h_tab_direction=None
        h_time=self.temporal_encoder(x_hist,x_future)
        h_dir,h_mag=self.adapters(h_tab,h_time,h_tab_direction=h_tab_direction,
                                  detach_magnitude_time=direction_protected)
        b,k,h,d=h_dir.shape
        if k!=self.k: raise RuntimeError("TabM member axis mismatch")
        z=self._direction_logits(h_dir)
        p_members=torch.sigmoid(z);p_hat=p_members.mean(1)
        b,k,h,d=h_mag.shape
        a_scaled=torch.nn.functional.softplus(self.magnitude_head(h_mag.permute(0,2,1,3).reshape(b*h,k,d)).squeeze(-1).reshape(b,h,k).permute(0,2,1))
        magnitude_members=self.magnitude_scale*a_scaled;magnitude_hat=magnitude_members.mean(1)
        direction_hat=p_hat>=.5
        signed=torch.where(direction_hat,magnitude_hat.clamp_min(self.magnitude_eps),-magnitude_hat.clamp_min(self.magnitude_eps))
        votes=p_members>=.5;fraction=votes.float().mean(1)
        entropy=-(fraction.clamp(1e-8,1-1e-8)*fraction.clamp(1e-8,1-1e-8).log()+(1-fraction).clamp(1e-8,1-1e-8)*(1-fraction).clamp(1e-8,1-1e-8).log())
        result={"z_members":z,"p_members":p_members,"p":p_hat,"direction_hat":direction_hat,
            "a_scaled_members":a_scaled,"magnitude_members":magnitude_members,"magnitude_hat":magnitude_hat,
            "signed_kpi_hat":signed,"direction_member_std":p_members.std(1,unbiased=False),
            "direction_vote_fraction":fraction,"direction_vote_entropy":entropy,
            "magnitude_member_std":magnitude_members.std(1,unbiased=False),
            "alpha_dir":self.adapters.alphas()[0],"alpha_mag":self.adapters.alphas()[1],"H_tab":h_tab,"H_time":h_time}
        if self.direction_global_gate_logit is not None:
            result["direction_global_gate"] = torch.sigmoid(self.direction_global_gate_logit)
        if not all(torch.isfinite(v).all() for v in result.values() if isinstance(v,torch.Tensor)):
            raise FloatingPointError("non-finite V2.1 forward output")
        return result

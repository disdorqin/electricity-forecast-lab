from __future__ import annotations

import torch
from torch import nn
from .task_adapters import TaskAdapters, MemberWiseTaskAdaptersV21


def dual_readouts(z, m_pos, m_nonpos, *, magnitude_eps=1e-8):
    p = torch.sigmoid(z)
    y_soft_members = p.unsqueeze(1) * m_pos - (1.0 - p.unsqueeze(1)) * m_nonpos
    y_soft = y_soft_members.mean(dim=1)
    magnitude_members = p.unsqueeze(1) * m_pos + (1.0 - p.unsqueeze(1)) * m_nonpos
    magnitude_hat = magnitude_members.mean(dim=1)
    direction_hat = p >= 0.5
    magnitude_for_signed = magnitude_hat.clamp_min(magnitude_eps)
    signed = torch.where(direction_hat, magnitude_for_signed, -magnitude_for_signed)
    return p, y_soft_members, y_soft, direction_hat, magnitude_members, magnitude_hat, signed


class DualReadoutV2(nn.Module):
    """V2 adapter/readout interface; H_tab is supplied by official TabM upstream."""
    def __init__(self, d_tab: int, d_time: int, d_task: int = 32, mode: str = "A2", magnitude_eps: float = 1e-8,
                 *, tabular_encoder: nn.Module | None = None, temporal_encoder: nn.Module | None = None,
                 ensemble_size: int | None = None, magnitude_scale: float = 1.0):
        super().__init__()
        if magnitude_eps <= 0: raise ValueError("magnitude_eps must be positive")
        if (tabular_encoder is None) != (temporal_encoder is None):
            raise ValueError("provide both real encoders or neither")
        if ensemble_size is not None:
            raise ValueError("DualReadoutV2 is legacy-only; use DualBranchV21 for formal training")
        self.legacy_v20 = ensemble_size is None
        self.adapters = TaskAdapters(d_tab, d_time, d_task, mode) if self.legacy_v20 else MemberWiseTaskAdaptersV21(d_tab,d_time,d_task,mode)
        if not self.legacy_v20 and (magnitude_scale <= 0): raise ValueError("magnitude_scale must be positive")
        self.magnitude_scale=float(magnitude_scale)
        self.direction_head = nn.Linear(d_task, 1) if self.legacy_v20 else __import__("tabm").LinearEnsemble(d_task,1,k=ensemble_size)
        self.ensemble_size = ensemble_size
        if self.legacy_v20:
            self.m_pos_head = nn.Linear(d_task, 1)
            self.m_nonpos_head = nn.Linear(d_task, 1)
        else:
            # Official member-wise readouts preserve k through both task heads.
            import tabm
            self.magnitude_head = tabm.LinearEnsemble(d_task, 1, k=ensemble_size)
        self.tabular_encoder = tabular_encoder
        self.temporal_encoder = temporal_encoder
        self.magnitude_eps = float(magnitude_eps)
        self.mode = mode

    def forward(self, first, second):
        if self.tabular_encoder is not None:
            x_hist, x_future = first, second
            h_tab = self.tabular_encoder(x_future)
            h_time = self.temporal_encoder(x_hist, x_future)
        else:
            h_tab, h_time = first, second
        h_dir, h_mag = self.adapters(h_tab, h_time)
        if self.legacy_v20:
            return self._legacy_forward(h_dir,h_mag,h_tab,h_time)
        b,k,horizons,d=h_dir.shape
        flat_dir=h_dir.permute(0,2,1,3).reshape(b*horizons,k,d)
        z=self.direction_head(flat_dir).squeeze(-1).reshape(b,horizons,k).permute(0,2,1)
        p_members=torch.sigmoid(z)
        p=p_members.mean(dim=1)
        b,k,horizons,d=h_mag.shape
        flat_mag=h_mag.permute(0,2,1,3).reshape(b*horizons,k,d)
        a_scaled_members=torch.nn.functional.softplus(self.magnitude_head(flat_mag).squeeze(-1)).reshape(b,horizons,k).permute(0,2,1)
        magnitude_members=self.magnitude_scale*a_scaled_members
        magnitude_hat=magnitude_members.mean(dim=1)
        direction_hat=p>=0.5
        signed=torch.where(direction_hat,magnitude_hat.clamp_min(self.magnitude_eps),-magnitude_hat.clamp_min(self.magnitude_eps))
        member_votes=p_members>=0.5
        vote_fraction=member_votes.float().mean(dim=1)
        vote_entropy=-(vote_fraction.clamp(1e-8,1-1e-8)*vote_fraction.clamp(1e-8,1-1e-8).log()+(1-vote_fraction).clamp(1e-8,1-1e-8)*(1-vote_fraction).clamp(1e-8,1-1e-8).log())
        alpha_dir,alpha_mag=self.adapters.alphas()
        if not all(torch.isfinite(v).all() for v in (z,p,a_scaled_members,magnitude_hat,signed)):
            raise FloatingPointError("non-finite V2.1 model output")
        return {"z_members":z,"p_members":p_members,"p":p,"direction_hat":direction_hat,
            "a_scaled_members":a_scaled_members,"magnitude_members":magnitude_members,
            "magnitude_hat":magnitude_hat,"signed_kpi_hat":signed,
            "direction_member_std":p_members.std(dim=1,unbiased=False),"direction_vote_fraction":vote_fraction,
            "direction_vote_entropy":vote_entropy,"magnitude_member_std":magnitude_members.std(dim=1,unbiased=False),
            "alpha_dir":alpha_dir,"alpha_mag":alpha_mag,
            **({"H_tab":h_tab,"H_time":h_time} if self.tabular_encoder is not None else {})}

    def _legacy_forward(self,h_dir,h_mag,h_tab,h_time):
        # Preserved only for existing V2.0 synthetic regression tests. The real
        # trainer always passes ensemble_size and therefore cannot enter here.
        z = self.direction_head(h_dir).squeeze(-1)
        p = torch.sigmoid(z)
        if self.ensemble_size is None:
            m_pos = torch.nn.functional.softplus(self.m_pos_head(h_mag).squeeze(-1))
            m_nonpos = torch.nn.functional.softplus(self.m_nonpos_head(h_mag).squeeze(-1))
        else:
            b, k, horizons, d = h_mag.shape
            flat = h_mag.permute(0, 2, 1, 3).reshape(b * horizons, k, d)
            m_pos = torch.nn.functional.softplus(self.m_pos_head(flat).squeeze(-1)).reshape(b, horizons, k).permute(0, 2, 1)
            m_nonpos = torch.nn.functional.softplus(self.m_nonpos_head(flat).squeeze(-1)).reshape(b, horizons, k).permute(0, 2, 1)
        p, y_soft_members, y_soft, direction_hat, magnitude_members, magnitude_hat, signed = dual_readouts(
            z, m_pos, m_nonpos, magnitude_eps=self.magnitude_eps)
        alpha_dir, alpha_mag = self.adapters.alphas()
        if not (torch.isfinite(y_soft).all() and torch.isfinite(signed).all()):
            raise FloatingPointError("non-finite V2 readout")
        return {
            "z": z, "p": p, "m_pos_members": m_pos, "m_nonpos_members": m_nonpos,
            "y_soft_train_members": y_soft_members, "y_soft_train": y_soft,
            "direction_hat": direction_hat, "magnitude_members": magnitude_members,
            "magnitude_hat": magnitude_hat, "signed_kpi_hat": signed,
            "alpha_dir": alpha_dir, "alpha_mag": alpha_mag,
            **({"H_tab": h_tab, "H_time": h_time} if self.tabular_encoder is not None else {}),
        }


def official_tabm_import_status():
    try:
        import tabm  # noqa: F401
        import rtdl_num_embeddings  # noqa: F401
        import shap  # noqa: F401
        return "PASS"
    except ImportError as exc:
        return f"BLOCKED: {type(exc).__name__}: {exc}"

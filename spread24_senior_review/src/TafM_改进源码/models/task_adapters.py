from __future__ import annotations

import math
import torch
from torch import nn


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class TaskAdapters(nn.Module):
    """Dual global-scalar task adapters over injected H_tab/H_time representations."""
    def __init__(self, d_tab: int, d_time: int, d_task: int = 32, mode: str = "A2"):
        super().__init__()
        if mode not in {"A0", "A1", "A2"}:
            raise ValueError("mode must be A0/A1/A2")
        self.mode = mode
        self.tab_dir = nn.Linear(d_tab, d_task)
        self.time_dir = nn.Linear(d_time, d_task)
        self.tab_mag = nn.Linear(d_tab, d_task)
        self.time_mag = nn.Linear(d_time, d_task)
        init = 1.0 if mode == "A0" else 0.8
        self.alpha_dir_init = float(init)
        self.alpha_mag_init = float(init)
        for name in ("a_dir", "a_mag"):
            value = torch.tensor(_logit(0.8), dtype=torch.float32) if mode == "A2" else torch.tensor(init, dtype=torch.float32)
            if mode == "A2":
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)

    def alphas(self):
        if self.mode == "A2":
            return torch.sigmoid(self.a_dir), torch.sigmoid(self.a_mag)
        return self.a_dir, self.a_mag

    def mode_state(self):
        alpha_dir, alpha_mag = self.alphas()
        return {"mode": self.mode, "alpha_dir_init": self.alpha_dir_init, "alpha_mag_init": self.alpha_mag_init,
                "alpha_dir_current": float(alpha_dir.detach().cpu()), "alpha_mag_current": float(alpha_mag.detach().cpu()),
                "learnable": self.mode == "A2"}

    def forward(self, h_tab, h_time):
        # H_tab [B,k,24,d_tab], H_time [B,24,d_time].
        if h_tab.ndim != 4 or h_time.ndim != 3 or h_tab.shape[0] != h_time.shape[0] or h_tab.shape[2] != h_time.shape[1]:
            raise ValueError("expected H_tab [B,k,24,d] and H_time [B,24,d]")
        tab_dir = self.tab_dir(h_tab.mean(dim=1))
        time_dir = self.time_dir(h_time)
        tab_mag = self.tab_mag(h_tab)
        time_mag = self.time_mag(h_time).unsqueeze(1).expand(-1, h_tab.shape[1], -1, -1)
        alpha_dir, alpha_mag = self.alphas()
        h_dir = alpha_dir * time_dir + (1.0 - alpha_dir) * tab_dir
        h_mag = alpha_mag * tab_mag + (1.0 - alpha_mag) * time_mag
        return h_dir, h_mag


ARCHITECTURE_MODES = {"full_current", "tabular_only", "temporal_only"}


class MemberWiseTaskAdaptersV21(nn.Module):
    """Canonical V2.1 task adapters; the TabM member axis is never collapsed.

    `architecture_mode` is experiment-only (E2-A big-block ablation). It selects how the
    Direction representation is formed and nothing else; the Magnitude fusion is identical
    in all three modes, and the parameter set is identical in all three modes (the unused
    alpha stays registered so checkpoints and parameter counts remain comparable).

      full_current   h_dir = alpha*time_dir(H_time) + (1-alpha)*tab_dir(H_tab)   [canonical]
      tabular_only   h_dir = tab_dir(H_tab)     -> temporal branch cannot reach Direction
      temporal_only  h_dir = time_dir(H_time)   -> tabular branch cannot reach Direction
    """
    def __init__(self, d_tab: int, d_time: int, d_task: int = 32, mode: str = "A2",
                 architecture_mode: str = "full_current",
                 direction_fusion_alpha: float | None = None):
        super().__init__()
        if mode not in {"A0", "A1", "A2"}:
            raise ValueError("mode must be A0/A1/A2")
        if architecture_mode not in ARCHITECTURE_MODES:
            raise ValueError("architecture_mode must be full_current/tabular_only/temporal_only")
        if direction_fusion_alpha is not None:
            if architecture_mode != "full_current":
                raise ValueError("direction_fusion_alpha requires architecture_mode=full_current")
            if not 0.0 <= float(direction_fusion_alpha) <= 1.0:
                raise ValueError("direction_fusion_alpha must be within [0,1]")
        self.mode=mode
        self.architecture_mode=architecture_mode
        self.direction_fusion_alpha=(None if direction_fusion_alpha is None
                                     else float(direction_fusion_alpha))
        self.tab_dir=nn.Linear(d_tab,d_task); self.time_dir=nn.Linear(d_time,d_task)
        self.tab_mag=nn.Linear(d_tab,d_task); self.time_mag=nn.Linear(d_time,d_task)
        initial=1.0 if mode=="A0" else 0.8
        self.alpha_dir_init=initial; self.alpha_mag_init=initial
        for name in ("a_dir","a_mag"):
            value=torch.tensor(_logit(0.8) if mode=="A2" else initial,dtype=torch.float32)
            if mode=="A2": self.register_parameter(name,nn.Parameter(value))
            else: self.register_buffer(name,value)

    def alphas(self):
        if self.mode=="A2": return torch.sigmoid(self.a_dir),torch.sigmoid(self.a_mag)
        return self.a_dir,self.a_mag

    def forward(self,h_tab,h_time,*,h_tab_direction=None,detach_magnitude_time: bool=False):
        if h_tab.ndim!=4 or h_time.ndim!=3 or h_tab.shape[0]!=h_time.shape[0] or h_tab.shape[2]!=h_time.shape[1]:
            raise ValueError("expected H_tab [B,k,24,d] and H_time [B,24,d]")
        h_tab_direction = h_tab if h_tab_direction is None else h_tab_direction
        if h_tab_direction.shape != h_tab.shape:
            raise ValueError("H_tab_direction must preserve canonical [B,k,24,d] shape")
        td=self.time_dir(h_time).unsqueeze(1).expand(-1,h_tab.shape[1],-1,-1)
        sd=self.tab_dir(h_tab_direction)
        sm=self.tab_mag(h_tab)
        time_for_magnitude = h_time.detach() if detach_magnitude_time else h_time
        tm=self.time_mag(time_for_magnitude).unsqueeze(1).expand_as(sm)
        ad,am=self.alphas()
        fixed=self.direction_fusion_alpha
        # Endpoints short-circuit rather than multiplying by zero, so a fixed-alpha endpoint is
        # structurally the same graph as the matching architecture_mode branch: the unused branch
        # is absent from the Direction graph instead of present with a zero gradient.
        if self.architecture_mode=="tabular_only" or fixed==0.0:
            h_dir=sd
        elif self.architecture_mode=="temporal_only" or fixed==1.0:
            h_dir=td
        else:
            weight=ad if fixed is None else torch.as_tensor(fixed,dtype=td.dtype,device=td.device)
            h_dir=weight*td+(1-weight)*sd
        return h_dir, am*sm+(1-am)*tm

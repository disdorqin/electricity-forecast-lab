"""Doc17 V1.3 loss-family compatibility for V2 readout tensors (not a trainer)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def v2_loss_compat(outputs, y_scaled, *, w_pos=1.0, w_nonpos=1.0, lambda_dir=1.0, lambda_mag=1.0, beta=1.0):
    y = y_scaled
    if y.ndim != 2 or outputs["p"].shape != y.shape:
        raise ValueError("y_scaled and p must have shape [B,24]")
    positive = y > 0
    nonpositive = ~positive
    weights = torch.where(positive, torch.as_tensor(w_pos, device=y.device, dtype=y.dtype),
                         torch.as_tensor(w_nonpos, device=y.device, dtype=y.dtype))
    l_dir = (F.binary_cross_entropy_with_logits(outputs["z"], positive.to(y.dtype), reduction="none") * weights).mean()
    abs_y = y.abs().unsqueeze(1)
    pos_mask, neg_mask = positive.unsqueeze(1).expand_as(outputs["m_pos_members"]), nonpositive.unsqueeze(1).expand_as(outputs["m_nonpos_members"])
    active = []
    if pos_mask.any(): active.append(F.smooth_l1_loss(outputs["m_pos_members"][pos_mask], abs_y.expand_as(outputs["m_pos_members"])[pos_mask], beta=beta))
    if neg_mask.any(): active.append(F.smooth_l1_loss(outputs["m_nonpos_members"][neg_mask], abs_y.expand_as(outputs["m_nonpos_members"])[neg_mask], beta=beta))
    if not active: raise RuntimeError("empty target batch")
    l_mag = torch.stack(active).mean()
    l_point = F.smooth_l1_loss(outputs["y_soft_train_members"], y.unsqueeze(1).expand_as(outputs["y_soft_train_members"]), beta=beta)
    total = l_point + lambda_dir * l_dir + lambda_mag * l_mag
    return {"L_dir": l_dir, "L_mag": l_mag, "L_point": l_point, "L_total": total,
            "batch_positive_count": int(positive.sum().item()), "batch_nonpositive_count": int(nonpositive.sum().item())}


def v21_loss(outputs, y_scaled, *, lambda_dir=1.0, lambda_mag=1.0,
             direction_weight_pos: float = 1.0, direction_weight_nonpos: float = 1.0):
    """Doc17 KPI-aligned loss; canonical defaults remain unweighted BCE + direct L1.

    Non-unit Direction weights are experiment-only and are applied per target label before
    averaging.  Defaults preserve the canonical arithmetic exactly.
    """
    y=y_scaled
    z=outputs["z_members"]
    a=outputs["a_scaled_members"]
    if y.ndim!=2 or z.ndim!=3 or a.shape!=z.shape or z.shape[0]!=y.shape[0] or z.shape[2]!=y.shape[1]:
        raise ValueError("expected y [B,24], z/a_scaled [B,k,24]")
    if direction_weight_pos <= 0 or direction_weight_nonpos <= 0:
        raise ValueError("Direction class weights must be positive")
    positive=(y>0).to(y.dtype).unsqueeze(1).expand_as(z)
    if direction_weight_pos == 1.0 and direction_weight_nonpos == 1.0:
        l_dir=F.binary_cross_entropy_with_logits(z,positive,reduction="mean")
    else:
        per=F.binary_cross_entropy_with_logits(z,positive,reduction="none")
        weights=torch.where(positive>0.5,
            torch.as_tensor(direction_weight_pos,device=z.device,dtype=z.dtype),
            torch.as_tensor(direction_weight_nonpos,device=z.device,dtype=z.dtype))
        l_dir=(per*weights).mean()
    target=y.abs().unsqueeze(1).expand_as(a)
    l_mag=F.l1_loss(a,target,reduction="mean")
    total=lambda_dir*l_dir+lambda_mag*l_mag
    return {"L_dir":l_dir,"L_mag":l_mag,"L_total":total,
            "batch_positive_count":int((y>0).sum().item()),"batch_nonpositive_count":int((y<=0).sum().item())}

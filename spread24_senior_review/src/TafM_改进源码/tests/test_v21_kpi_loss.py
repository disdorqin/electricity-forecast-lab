import torch
import torch.nn.functional as F

from src.TafM_改进源码.losses import v21_loss


def test_v21_loss_is_unweighted_memberwise_bce_plus_l1_only():
    z=torch.tensor([[[0.0,1.0],[-1.0,2.0]]],requires_grad=True)
    a=torch.tensor([[[1.0,2.0],[3.0,4.0]]],requires_grad=True)
    y=torch.tensor([[1.0,-2.0]])
    out=v21_loss({"z_members":z,"a_scaled_members":a},y)
    target=(y>0).float().unsqueeze(1).expand_as(z)
    assert torch.allclose(out["L_dir"],F.binary_cross_entropy_with_logits(z,target))
    assert torch.allclose(out["L_mag"],F.l1_loss(a,y.abs().unsqueeze(1).expand_as(a)))
    assert torch.allclose(out["L_total"],out["L_dir"]+out["L_mag"])
    assert "L_point" not in out
    out["L_total"].backward()
    assert torch.isfinite(z.grad).all() and torch.isfinite(a.grad).all()

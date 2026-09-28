import torch
from src.TafM_改进源码.losses import v2_loss_compat
from src.TafM_改进源码.models.dual_branch_v2 import DualReadoutV2


def test_v1_loss_family_compatibility_and_backward():
    model = DualReadoutV2(4, 3, 8, mode="A2")
    out = model(torch.randn(4, 2, 24, 4), torch.randn(4, 24, 3))
    y = torch.randn(4, 24); y[0, 0] = 0
    losses = v2_loss_compat(out, y)
    losses["L_total"].backward()
    assert torch.isfinite(losses["L_total"]) and losses["batch_nonpositive_count"] > 0

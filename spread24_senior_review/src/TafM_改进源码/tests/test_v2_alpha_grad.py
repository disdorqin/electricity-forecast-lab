import torch
from src.TafM_改进源码.models.dual_branch_v2 import DualReadoutV2


def test_a2_both_global_scalar_gates_have_finite_gradients():
    model = DualReadoutV2(4, 3, 8, mode="A2")
    out = model(torch.randn(3, 2, 24, 4), torch.randn(3, 24, 3))
    loss = out["y_soft_train"].square().mean() + out["p"].mean()
    loss.backward()
    assert model.adapters.a_dir.grad is not None and torch.isfinite(model.adapters.a_dir.grad)
    assert model.adapters.a_mag.grad is not None and torch.isfinite(model.adapters.a_mag.grad)

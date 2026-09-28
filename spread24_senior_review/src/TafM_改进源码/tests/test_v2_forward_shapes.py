import torch
from src.TafM_改进源码.models.dual_branch_v2 import DualReadoutV2


def test_v2_forward_contract_shapes_and_finite_values():
    model = DualReadoutV2(5, 3, 8, mode="A2")
    out = model(torch.randn(2, 4, 24, 5), torch.randn(2, 24, 3))
    assert out["z"].shape == out["p"].shape == (2, 24)
    assert out["m_pos_members"].shape == out["y_soft_train_members"].shape == (2, 4, 24)
    assert out["signed_kpi_hat"].shape == out["magnitude_hat"].shape == (2, 24)
    assert all(torch.isfinite(v).all() for v in out.values() if isinstance(v, torch.Tensor))

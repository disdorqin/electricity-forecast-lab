import torch
from src.TafM_改进源码.models.dual_branch_v2 import DualReadoutV2


def test_a0_a1_a2_scalar_mode_contracts():
    for mode, expected, learnable in (("A0", 1.0, False), ("A1", .8, False), ("A2", .8, True)):
        model = DualReadoutV2(3, 2, 4, mode=mode)
        a, b = model.adapters.alphas()
        assert torch.allclose(a, torch.tensor(expected)) and torch.allclose(b, torch.tensor(expected))
        assert model.adapters.a_dir.requires_grad is learnable and model.adapters.a_mag.requires_grad is learnable

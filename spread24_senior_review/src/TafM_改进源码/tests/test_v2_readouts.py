import torch
from src.TafM_改进源码.models.dual_branch_v2 import dual_readouts


def test_dual_readout_magnitude_and_direction_own_sign():
    z = torch.zeros(1, 24); mpos = torch.full((1, 3, 24), 100.); mneg = torch.full_like(mpos, 100.)
    p, train_m, train, direction, magnitude_m, magnitude, signed = dual_readouts(z, mpos, mneg)
    assert torch.allclose(train, torch.zeros_like(train), atol=1e-6)
    assert torch.allclose(magnitude, torch.full_like(magnitude, 100.))
    z.fill_(1.0); mpos.fill_(1.0); mneg.fill_(10000.)
    *_, direction, magnitude_m, magnitude, signed = dual_readouts(z, mpos, mneg)
    assert direction.all() and (signed > 0).all()

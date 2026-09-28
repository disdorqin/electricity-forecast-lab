import math
import torch

from src.TafM_改进源码.losses import v21_loss
from src.TafM_改进源码.train import direction_class_weights


class S:
    positive_count = 36
    nonpositive_count = 64


def test_unweighted_v21_exact_default():
    z=torch.tensor([[[0.2,-0.3]]],dtype=torch.float32)
    a=torch.ones_like(z)
    y=torch.tensor([[1.0,-2.0]],dtype=torch.float32)
    out={"z_members":z,"a_scaled_members":a}
    got=v21_loss(out,y)
    exp=torch.nn.functional.binary_cross_entropy_with_logits(
        z,(y>0).float().unsqueeze(1).expand_as(z),reduction="mean")
    assert torch.equal(got["L_dir"],exp)


def test_direction_weight_modes():
    assert direction_class_weights(S(),"unweighted")== (1.0,1.0)
    wp,wn=direction_class_weights(S(),"full_balanced")
    assert abs(wp-100/(2*36))<1e-12
    assert abs(wn-100/(2*64))<1e-12
    sp,sn=direction_class_weights(S(),"sqrt_balanced")
    assert sp>1 and sn<1
    assert abs((.36*sp+.64*sn)-1)<1e-12


def test_weighted_bce_penalizes_positive_more():
    z=torch.zeros((1,1,2),dtype=torch.float32)
    a=torch.ones_like(z)
    y=torch.tensor([[1.0,-1.0]],dtype=torch.float32)
    out={"z_members":z,"a_scaled_members":a}
    u=v21_loss(out,y)["L_dir"]
    w=v21_loss(out,y,direction_weight_pos=2.0,direction_weight_nonpos=1.0)["L_dir"]
    assert w>u

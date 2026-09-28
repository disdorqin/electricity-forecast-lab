import pytest
import torch

from src.TafM_改进源码.contracts import TEMPORAL_FEATURES
from src.TafM_改进源码.losses import v21_loss
from src.TafM_改进源码.models.dual_branch_v21 import DualBranchV21
from src.TafM_改进源码.models.tabular_encoder import TabularEncoder
from src.TafM_改进源码.models.temporal_encoder import TemporalEncoder
from src.TafM_改进源码.train import _stage_b_freeze_audit


def _small_official_model(k=2):
    # Use the installed official TabM and official PLE APIs, not a local stand-in.
    names = list(TEMPORAL_FEATURES[1:]) + ["extra_forecast"]
    roles = [{"feature_name": n, "role": "Strong" if i == 0 else "Weak"}
             for i, n in enumerate(names)]
    bins = [[-3.0, 0.0, 3.0] for _ in names]
    tab = TabularEncoder(feature_names=names, feature_roles=roles, ple_bins=bins,
                         d_tab=8, k=k, n_blocks=2, ple_embedding_dim=2)
    temporal = TemporalEncoder(selected_feature_names=names, d_time=8, hidden=8, fft_bins=(1, 7))
    model = DualBranchV21(d_tab=8, d_time=8, d_task=4, mode="A2", tabular_encoder=tab,
                          temporal_encoder=temporal, k=k, magnitude_scale=2.5)
    return model


def test_official_tabm_member_axis_survives_to_v21_heads_and_backward():
    model = _small_official_model(k=2)
    xh = torch.randn(2, 168, 7)
    xf = torch.randn(2, 24, len(TEMPORAL_FEATURES[1:]) + 1)
    out = model(xh, xf)
    assert out["H_tab"].shape == (2, 2, 24, 8)
    assert out["H_time"].shape == (2, 24, 8)
    assert out["z_members"].shape == (2, 2, 24)
    assert out["a_scaled_members"].shape == (2, 2, 24)
    assert torch.equal(out["direction_hat"], out["p"] >= 0.5)
    loss = v21_loss(out, torch.randn(2, 24))["L_total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.tabular_encoder.parameter_topology()["topology_audit"] == "PASS"


def test_temporal_encoder_rejects_target_actual_and_target_spread_conditioning():
    legal = list(TEMPORAL_FEATURES[1:])
    with pytest.raises(ValueError, match="actual"):
        TemporalEncoder(selected_feature_names=legal + ["target_day_actual"], d_time=8)
    with pytest.raises(ValueError, match="target-day spread"):
        TemporalEncoder(selected_feature_names=legal + ["target_spread"], d_time=8)


def test_stage_b_freeze_audit_uses_module_ownership_and_closes_unknowns():
    report = _stage_b_freeze_audit(_small_official_model(k=2))
    assert report["status"] == "PASS"
    assert report["unknown_parameter_tensors"] == 0
    assert report["shared_tabm_backbone_frozen"] is True
    assert report["ple_frozen"] is True
    assert report["overlapping_owner_groups"] is False

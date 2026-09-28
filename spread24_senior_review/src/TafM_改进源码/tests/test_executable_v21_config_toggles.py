import hashlib

import numpy as np
import pytest
import torch
import yaml

from src.TafM_改进源码.config import default_v21_config_path, load_v21_config
from src.TafM_改进源码.contracts import TEMPORAL_FEATURES
from src.TafM_改进源码.models.tabular_encoder import TabularEncoder
from src.TafM_改进源码.models.temporal_encoder import TemporalEncoder


def test_yaml_is_runtime_source_of_truth_and_sha_is_content_addressed():
    path = default_v21_config_path()
    cfg = load_v21_config()
    assert cfg.config_path == str(path.resolve())
    assert cfg.config_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert cfg.future_clip_abs == 10.0 and cfg.temporal_clip_abs == 10.0
    assert cfg.ple_enabled and cfg.fft_enabled and cfg.future_conditioning
    assert cfg.stage_b_production_default is False


def test_unknown_and_invalid_config_fields_fail_closed(tmp_path):
    base = yaml.safe_load(default_v21_config_path().read_text(encoding="utf-8"))
    unknown = dict(base, unrecognized_switch=True)
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(unknown), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown V2.1 config keys"):
        load_v21_config(path)
    invalid = dict(base, k="4")
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid type"):
        load_v21_config(path)
    invalid = dict(base, ple_bins=1)
    path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="bins >= 2"):
        load_v21_config(path)


def test_ple_toggle_changes_module_graph_but_preserves_output_shape():
    names = list(TEMPORAL_FEATURES[1:])
    roles = [{"feature_name": n, "role": "Strong"} for n in names]
    bins = [[-10.0, 0.0, 10.0] for _ in names]
    on = TabularEncoder(feature_names=names, feature_roles=roles, ple_bins=bins,
                        d_tab=8, k=2, n_blocks=2, ple_embedding_dim=2, ple_enabled=True)
    off = TabularEncoder(feature_names=names, feature_roles=roles, ple_bins=[],
                         d_tab=8, k=2, n_blocks=2, ple_embedding_dim=2, ple_enabled=False)
    x = torch.zeros(1, 24, len(names))
    assert on.ple is not None and off.ple is None
    assert on(x).shape == off(x).shape == (1, 2, 24, 8)
    assert torch.isfinite(off(x)).all()


def test_fft_toggle_bypasses_fft_path(monkeypatch):
    names = list(TEMPORAL_FEATURES[1:])
    off = TemporalEncoder(selected_feature_names=names, d_time=8, hidden=8, fft_enabled=False)
    on = TemporalEncoder(selected_feature_names=names, d_time=8, hidden=8, fft_enabled=True)
    calls = {"count": 0}
    original = on._fft_forecast
    def count_fft(x):
        calls["count"] += 1
        return original(x)
    monkeypatch.setattr(on, "_fft_forecast", count_fft)
    xh = torch.randn(1, 168, 7).clamp(-10, 10)
    xf = torch.randn(1, 24, len(names)).clamp(-10, 10)
    assert off(xh, xf).shape == on(xh, xf).shape == (1, 24, 8)
    assert calls["count"] == 1


def test_future_conditioning_off_is_counterfactual_invariant():
    names = list(TEMPORAL_FEATURES[1:])
    on = TemporalEncoder(selected_feature_names=names, d_time=8, hidden=8, future_conditioning=True)
    off = TemporalEncoder(selected_feature_names=names, d_time=8, hidden=8, future_conditioning=False)
    off.load_state_dict(on.state_dict())
    with torch.no_grad():
        on.future_eta.fill_(1.0)
        off.future_eta.fill_(1.0)
    xh = torch.randn(1, 168, 7).clamp(-10, 10)
    x1 = torch.zeros(1, 24, len(names))
    x2 = x1.clone()
    x2[..., 0] = 2.0
    assert torch.equal(off(xh, x1), off(xh, x2))
    assert not torch.equal(on(xh, x1), on(xh, x2))

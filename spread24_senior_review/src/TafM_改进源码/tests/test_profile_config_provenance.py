from src.TafM_改进源码.config import V2Config
from src.TafM_改进源码.evaluate import _effective_profile_config


def test_smoke_profile_config_is_executable_and_manifest_ready():
    base = V2Config(seed=77)
    effective = _effective_profile_config(base, "smoke", mode="A1")
    assert (effective.max_epochs, effective.patience, effective.k, effective.batch_size) == (2, 2, 4, 32)
    assert effective.seed == 77
    assert effective.mode == "A1"
    assert effective.resolved_config()["max_epochs"] == 2


def test_default_profile_preserves_yaml_budget():
    base = V2Config(max_epochs=5, patience=3, k=6)
    effective = _effective_profile_config(base, "default", mode="A0")
    assert (effective.max_epochs, effective.patience, effective.k) == (5, 3, 6)
    assert effective.mode == "A0"

from pathlib import Path
from src.TafM_改进源码.registry import build_feature_registry, load_feature_groups
from collections import Counter


def _root(): return Path(__file__).resolve().parents[3]


def test_all_259_candidate_features_have_dependency_contract():
    root = _root(); groups = load_feature_groups(root / "data/frozen_repro/feature_groups.json")
    reg = build_feature_registry(groups, present_columns=set(sum(groups.values(), [])) | set(__import__('src.TafM_改进源码.contracts', fromlist=['F10_FEATURES']).F10_FEATURES),
                                 legacy_registry_path=root / "data/frozen_repro/feature_registry.json")
    required = {"actual_dependency", "forecast_dependency", "label_dependency", "derived_from", "max_dependency_window", "builder_name", "builder_version"}
    assert reg["candidate_feature_count"] == len(reg["candidate_features"]) == 259
    assert all(required <= set(r) and isinstance(r["actual_dependency"], list) for r in reg["candidate_features"])
    counts = Counter(r["dependency_evidence_status"] for r in reg["candidate_features"])
    assert counts == {"VERIFIED_CANONICAL_UPSTREAM": 244, "VERIFIED_CANONICAL": 15}
    by_name = {r["feature_name"]: r for r in reg["candidate_features"]}
    f5 = by_name["err_地方电厂总加_7d_mean"]["dependency_contract"]
    assert f5["actual_dependency"] == ["地方电厂总加实际值"]
    assert f5["forecast_dependency"] == ["地方电厂总加预测值"]
    assert f5["shift_days"] == 2 and f5["rolling_window_days"] == 7 and f5["statistic"] == "mean"
    qfeature = by_name["err_地方电厂总加_28d_q10"]["dependency_contract"]
    assert qfeature["shift_days"] == 2 and qfeature["rolling_window_days"] == 28 and qfeature["statistic"] == "q10"
    f6 = by_name["uncert_地方电厂总加_low"]["dependency_contract"]
    assert "err_地方电厂总加_28d_q10" in f6["depends_on_features"]
    assert f6["forecast_dependency"] == ["地方电厂总加预测值"]

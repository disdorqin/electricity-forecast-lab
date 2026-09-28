from datetime import date
from pathlib import Path
from src.TafM_改进源码.registry import build_feature_registry, load_feature_groups, quarantined_features_for_day
from src.TafM_改进源码.contracts import F10_FEATURES


def test_quarantine_is_feature_level_not_date_wide():
    root = Path(__file__).resolve().parents[3]
    groups = load_feature_groups(root / "data/frozen_repro/feature_groups.json")
    reg = build_feature_registry(groups, present_columns=set(sum(groups.values(), [])) | set(F10_FEATURES),
                                 legacy_registry_path=root / "data/frozen_repro/feature_registry.json")
    q = quarantined_features_for_day(reg, date(2026, 8, 20))
    assert q and all(reg["candidate_features"][[r["feature_name"] for r in reg["candidate_features"]].index(n)]["feature_group"] in {"F5", "F6"} for n in q)
    assert "forecast_day_fcast_直调负荷_mean" not in q
    assert not quarantined_features_for_day(reg, date(2026, 8, 16))

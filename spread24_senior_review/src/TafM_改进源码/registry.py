"""Field-level feature registry derived from the frozen F0-F9 inventory."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .contracts import (
    FEATURE_GROUP_COUNTS,
    F10_FEATURES,
    FUTURE_VINTAGE_EVIDENCE,
    INHERITED_CONTRACT_ID,
    TEMPORAL_FEATURES,
)
from .upstream_evidence import load_upstream_feature_evidence

BUSINESS_CORE = {
    "fcast_直调负荷",
    "fcast_竞价空间",
    "fcast_新能源总加",
    "fcast_风电总加",
    "fcast_光伏总加",
    "bidding_space_ratio",
    "renewable_minus_space",
    "residual_load_renew",
    "renewable_share",
    "ramp_wind",
}

F10_INPUTS: dict[str, tuple[str, ...]] = {
    "load_day_position": ("fcast_直调负荷",),
    "renewable_day_position": ("fcast_新能源总加",),
    "wind_day_position": ("fcast_风电总加",),
    "solar_day_position": ("fcast_光伏总加",),
    "bidding_space_day_position": ("fcast_竞价空间",),
    "residual_load_day_position": ("residual_load_renew",),
    "load_space_pressure": ("fcast_直调负荷", "fcast_竞价空间"),
    "residual_space_pressure": ("residual_load_renew", "fcast_竞价空间"),
    "renewable_space_balance": ("fcast_新能源总加", "fcast_竞价空间"),
    "net_ramp_pressure": ("ramp_load", "ramp_renewable"),
    "ramp_tightness": ("ramp_residual_load", "ramp_bidding_space"),
    "solar_drop_load_rise": ("ramp_solar", "ramp_load"),
    "load_x_space": ("fcast_直调负荷", "fcast_竞价空间"),
    "space_ratio_x_ramp_wind": ("bidding_space_ratio", "ramp_wind"),
    "residual_x_renewable_share": ("residual_load_renew", "renewable_share"),
}


def load_feature_groups(path: Path) -> dict[str, list[str]]:
    groups = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(groups, dict):
        raise ValueError("feature_groups.json must contain an object")
    ordered = {name: list(groups.get(name, [])) for name in FEATURE_GROUP_COUNTS}
    observed = {name: len(columns) for name, columns in ordered.items()}
    if observed != FEATURE_GROUP_COUNTS:
        raise ValueError(f"F0-F9 feature counts differ from the frozen contract: {observed}")
    flattened = [name for columns in ordered.values() for name in columns]
    if len(flattened) != len(set(flattened)):
        raise ValueError("duplicate feature name across F0-F9 groups")
    if set(groups) != set(FEATURE_GROUP_COUNTS):
        raise ValueError(f"unexpected feature groups: {sorted(set(groups) ^ set(FEATURE_GROUP_COUNTS))}")
    return ordered


def build_feature_registry(
    groups: dict[str, list[str]],
    *,
    present_columns: set[str] | None = None,
    legacy_registry_path: Path | None = None,
) -> dict[str, Any]:
    """Create registry records without inventing forecast-vintage evidence."""

    legacy: dict[str, dict[str, Any]] = {}
    if legacy_registry_path is not None and Path(legacy_registry_path).is_file():
        legacy_obj = json.loads(Path(legacy_registry_path).read_text(encoding="utf-8"))
        legacy = {item["feature"]: item for item in legacy_obj.get("features", []) if "feature" in item}

    candidate: list[dict[str, Any]] = []
    upstream_evidence, upstream_summary = load_upstream_feature_evidence(groups)
    for group, names in groups.items():
        for name in names:
            old = legacy.get(name, {})
            available = old.get("availability", "inherited from frozen feature builder")
            source_text = str(old.get("source", ""))
            transform = str(old.get("transform", ""))
            deps = _source_dependencies(source_text, transform, group, name)
            window = _dependency_window(transform, name, group)
            evidence = upstream_evidence.get(name)
            if evidence:
                # Canonical dependencies come from builder-hash-bound evidence,
                # never from the legacy source/transform inference above.
                deps = {key: list(evidence[key]) for key in
                        ("actual_dependency", "forecast_dependency", "label_dependency")}
                window = evidence["rolling_window_days"] or evidence["shift_days"] or "pointwise"
            candidate.append(
                {
                    "feature_name": name,
                    "feature_group": group,
                    "business_core": name in BUSINESS_CORE,
                    "future_known": True,
                    "temporal_capable": name in TEMPORAL_FEATURES,
                    "branch_hint": "both",
                    "history_source": f"historical source row / {old.get('source', 'frozen engineered feature')}",
                    "future_source": f"frozen_repro slot_table.parquet[{name}]",
                    "history_availability": available,
                    "future_availability": "at forecast origin by frozen builder contract",
                    "history_max_timestamp_rule": "history is selected by timestamp <= D-1 14:00; feature-specific whitelist only",
                    "future_vintage_rule": "target-day forecast version available by D-1 14:00 where forecast-derived",
                    "future_vintage_evidence_level": FUTURE_VINTAGE_EVIDENCE,
                    "inherited_contract_id": INHERITED_CONTRACT_ID,
                    **deps,
                    "derived_from": (list(evidence["derived_from"]) + list(evidence["depends_on_features"])
                                     if evidence else deps["actual_dependency"] + deps["forecast_dependency"] + deps["label_dependency"]),
                    "max_dependency_window": window,
                    "builder_name": evidence["builder_name"] if evidence else "UNKNOWN_INHERITED",
                    "builder_version": evidence["builder_version"] if evidence else "UNKNOWN_INHERITED",
                    "dependency_authority": "UPSTREAM_CANONICAL_BUILDER",
                    "dependency_evidence_status": evidence["dependency_evidence_status"] if evidence else "INFERRED_LEGACY",
                    "dependency_evidence_source": evidence["dependency_evidence_source"] if evidence else "legacy feature_registry source/transform metadata; upstream evidence bundle missing or invalid",
                    "dependency_closure_status": evidence["dependency_closure_status"] if evidence else "UNVERIFIED",
                    "dependency_contract": ({key: evidence[key] for key in ("actual_dependency", "forecast_dependency", "label_dependency", "derived_from", "depends_on_features", "shift_days", "rolling_window_days", "statistic")} if evidence else None),
                    "upstream_builder_evidence": ({key: evidence[key] for key in ("builder_file_sha256s", "upstream_lineage_asset_sha256", "upstream_registry_record")} if evidence else None),
                    "source_sign_convention": "DA-RT",
                    "model_target_sign_convention": "RT-DA",
                    "availability_status": "AVAILABLE",
                    "allow_in_hist": name in TEMPORAL_FEATURES,
                    "allow_in_future": True,
                    "legacy_source": old.get("source"),
                    "legacy_transform": old.get("transform"),
                    "legacy_availability": available,
                }
            )

    f10_records = []
    for name in F10_FEATURES:
        f10_records.append(
            {
                "feature_name": name,
                "feature_group": "F10",
                "business_core": False,
                "future_known": True,
                "temporal_capable": False,
                "branch_hint": "both",
                "history_source": "forbidden in historical temporal input",
                "future_source": "deterministic target-day forecast / derived forecast",
                "input_columns": list(F10_INPUTS[name]),
                "history_availability": "not allowed in history",
                "future_availability": "target-day complete forecast profile available at origin by frozen contract",
                "history_max_timestamp_rule": "not applicable; allow_in_hist=false",
                "future_vintage_rule": "CONTRACT_INHERITED from canonical forecast source contract",
                "future_vintage_evidence_level": FUTURE_VINTAGE_EVIDENCE,
                "inherited_contract_id": INHERITED_CONTRACT_ID,
                "allow_in_hist": False,
                "allow_in_future": True,
                "actual_dependency": [],
                "forecast_dependency": list(F10_INPUTS[name]),
                "label_dependency": [],
                "derived_from": list(F10_INPUTS[name]),
                "max_dependency_window": "intraday_24h",
                "builder_name": "build_f10_for_day",
                "builder_version": "f10_v1_doc17_14_1",
                "dependency_authority": "LOCAL_DETERMINISTIC_BUILDER",
                "dependency_evidence_status": "VERIFIED_CANONICAL",
                "dependency_evidence_source": "docs/17 §14.1 deterministic F10 formulas",
                "dependency_closure_status": "VERIFIED_CANONICAL",
                "availability_status": "AVAILABLE",
                "formula_version": "f10_v1_doc17_14_1",
            }
        )

    temporal = []
    for name in TEMPORAL_FEATURES:
        temporal.append(
            {
                "feature_name": name,
                "feature_group": "history_source",
                "source": "historical realized spread" if name == "target_spread" else f"historical canonical forecast series: {name}",
                "allow_in_hist": True,
                "allow_in_future": False,
                "temporal_capable": True,
                "max_timestamp_rule": "sample origin = D-1 14:00 inclusive",
                "future_vintage_evidence_level": (
                    "CONTRACT_INHERITED" if name != "target_spread" else "EMPIRICALLY_VERIFIED"
                ),
                "inherited_contract_id": INHERITED_CONTRACT_ID if name != "target_spread" else None,
                "actual_dependency": ["target_spread"] if name == "target_spread" else [],
                "forecast_dependency": [] if name == "target_spread" else [name],
                "label_dependency": [],
                "derived_from": [name],
                "max_dependency_window": "168h",
                "builder_name": "canonicalize_source",
                "builder_version": "UNKNOWN_INHERITED",
                "dependency_authority": "UPSTREAM_CANONICAL_BUILDER",
                "dependency_evidence_status": "INFERRED_LEGACY",
                "dependency_evidence_source": "docs/17 temporal whitelist plus inherited source identity",
                "dependency_closure_status": "UNVERIFIED",
            }
        )

    all_records = candidate + f10_records
    if len(all_records) != 259 or len({r["feature_name"] for r in all_records}) != 259:
        raise ValueError("candidate registry must contain exactly 259 unique F0-F10 features")
    if present_columns is not None:
        missing = sorted({r["feature_name"] for r in all_records} - present_columns)
        if missing:
            raise ValueError(f"candidate registry columns missing from source: {missing}")
    return {
        "schema": "spread24_sequence_feature_registry_v1.3",
        "canonical_spec": "docs/17_最终模型设计与编码规范.md#14",
        "candidate_feature_count": len(all_records),
        "future_feature_count": len(all_records),
        "history_temporal_feature_count": len(temporal),
        "candidate_features": all_records,
        "history_temporal_features": temporal,
        "dependency_authority": "UPSTREAM_CANONICAL_BUILDER",
        "upstream_builder_evidence_status": upstream_summary,
        "quarantine_rules": quarantine_rules(all_records),
        "source_identity": dict(INHERITED_CONTRACT_ID),
    }


def _source_dependencies(source: str, transform: str, group: str, name: str) -> dict[str, list[str]]:
    """Expand inherited source semantics into explicit dependency lists."""
    if group == "F5" or "实际值" in source:
        actual = [source.split("-", 1)[0] if "实际值" in source else "historical realized forecast error"]
        forecast = [source.split("-", 1)[1] if "实际值-" in source else "forecast vintage paired with historical actual"]
        label: list[str] = []
    elif group == "F6" or "historical error" in source:
        actual = ["historical realized forecast errors"]
        forecast = [source]
        label: list[str] = []
    elif group in {"F0", "F1", "F8"} or "spread" in source.lower() or name.startswith(("spread_", "ctx_spread", "ctxraw_spread")):
        actual, forecast, label = [], [], [source or "DA-RT target/spread history"]
    else:
        actual, forecast, label = [], [source or name], []
    return {"actual_dependency": actual, "forecast_dependency": forecast, "label_dependency": label}


def _dependency_window(transform: str, name: str, group: str) -> str | int:
    m = re.search(r"rolling\s*(\d+)", transform, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"lag\s*(\d+)", transform, re.IGNORECASE)
    if m:
        return int(m.group(1))
    if group == "F4":
        return "intraday_2h"
    if group == "F7":
        return "UNKNOWN_INHERITED"
    if group == "F9":
        return 28
    return {"F0": 28, "F1": 14, "F5": 28, "F6": 28, "F8": 1}.get(group, "pointwise")


def quarantine_rules(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    affected = [r for r in records if r.get("actual_dependency") and r["feature_group"] in {"F5", "F6"}]
    if not affected:
        return []
    root_day = date(2026, 8, 15)
    by_window: dict[int, list[dict[str, Any]]] = {}
    for row in affected:
        window = row.get("max_dependency_window")
        if isinstance(window, int): by_window.setdefault(window, []).append(row)
    # A supervised feature at D can first observe actuals from D-2 and retains
    # the root anomaly through its declared rolling window (inclusive).
    return [{
        "affected_keys": [{"target_day": root_day.isoformat(), "hour_business": h} for h in range(1, 25)],
        "affected_features": [r["feature_name"] for r in rows],
        "affected_source_dependency": ["2026-08-15 actual fundamentals / realized forecast errors"],
        "dependency_closure_start": (root_day + timedelta(days=2)).isoformat(),
        "dependency_closure_end": (root_day + timedelta(days=window + 1)).isoformat(),
        "reason": f"2026-08-15 actual anomaly propagated through {window}-day declared upstream dependency window",
        "resolution_source_sha256": None,
        "status": "UNVERIFIED",
        "dependency_evidence_status": "INFERRED_LEGACY",
        "dependency_authority": "UPSTREAM_CANONICAL_BUILDER",
    } for window, rows in sorted(by_window.items())]


def quarantined_features_for_day(registry: dict[str, Any], target_day: date) -> set[str]:
    result: set[str] = set()
    for rule in registry.get("quarantine_rules", []):
        if rule.get("status") not in {"UNRESOLVED", "UNVERIFIED"}:
            continue
        start, end = date.fromisoformat(rule["dependency_closure_start"]), date.fromisoformat(rule["dependency_closure_end"])
        if start <= target_day <= end:
            result.update(rule["affected_features"])
    return result

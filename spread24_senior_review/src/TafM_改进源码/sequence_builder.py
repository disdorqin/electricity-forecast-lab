"""Deterministic sequence_v1 dataset builder and leakage gate."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    BASE_SHA256,
    F10_FEATURES,
    FORECAST_ORIGIN,
    FUTURE_VINTAGE_EVIDENCE,
    HORIZON,
    HISTORY_HOURS,
    INHERITED_CONTRACT_ID,
    LABEL_CUTOFF_RULE,
    PARTIAL_TAIL_DAY,
    PARTIAL_TAIL_HOURS,
    TEMPORAL_FEATURES,
    ZERO_POLICY,
    ContractError,
    project_root,
    training_last_day,
)
from .f10 import F10_BUILDER_VERSION, build_f10_for_day
from .registry import build_feature_registry, load_feature_groups, quarantined_features_for_day
from .source_audit import inspect_source, run_source_gate
from .source_resolver import ResolvedSource, canonicalize_source, resolve_sources

SEQUENCE_BUILDER_VERSION = "sequence_v1_doc17_v1.3"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_provenance(root: Path) -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "pandas", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "python": platform.python_version(),
        "dependencies": versions,
        "git_commit": commit,
        "git_commit_status": "recorded" if commit else "no HEAD commit available",
        "seed": None,
        "seed_policy": "not applicable; data builders are deterministic and seedless",
        "resolution": "hourly",
        "forecast_origin_cutoff": FORECAST_ORIGIN,
        "command": "python -c from src.TafM_改进源码.sequence_builder import build_sequence_v1; build_sequence_v1()",
    }


def candidate_feature_names(groups: dict[str, list[str]]) -> list[str]:
    features = [feature for names in groups.values() for feature in names] + list(F10_FEATURES)
    if len(features) != 259 or len(set(features)) != 259:
        raise ContractError(f"expected 259 unique candidate features, got {len(features)}")
    return features


def _target_day_group(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    group = frame.loc[frame.target_day == day].copy()
    if len(group) != HORIZON:
        raise ContractError(f"target day {day} has {len(group)} rows, expected exactly 24")
    if group.hour_business.duplicated().any() or set(group.hour_business.astype(int)) != set(range(1, 25)):
        raise ContractError(f"target day {day} must have unique business hours 1..24")
    return group.sort_values("hour_business")


class InsufficientHistoryError(ContractError):
    """A target is too early to have its required 168-hour lookback."""


def build_day_sample(
    source_frame: pd.DataFrame,
    target_day: date | str,
    future_features: list[str],
    *,
    _canonicalized: bool = False,
) -> dict[str, Any]:
    """Build one [168,7] history, [24,F] future matrix, and separate [24] Y.

    Timestamp slicing, not business-date slicing, establishes the history cutoff.
    The returned history has no horizon axis, making shared cutoff structural.
    """

    day = pd.Timestamp(target_day).date()
    frame = source_frame if _canonicalized else canonicalize_source(source_frame)
    source_future_features = set(future_features) - set(F10_FEATURES)
    missing = sorted((set(TEMPORAL_FEATURES) | source_future_features) - set(frame.columns))
    if missing:
        raise ContractError(f"sequence source columns missing: {missing}")

    target = _target_day_group(frame, day)
    f10 = build_f10_for_day(target)
    future = pd.concat([target, f10], axis=1).sort_values("hour_business")
    future_values = future[future_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    future_values[~np.isfinite(future_values)] = np.nan

    origin = pd.Timestamp(day - timedelta(days=1)) + pd.Timedelta(hours=14)
    history_start = origin - pd.Timedelta(hours=HISTORY_HOURS - 1)
    timestamps = pd.to_datetime(frame["时刻"])
    mask = (timestamps >= history_start) & (timestamps <= origin)
    history = frame.loc[mask].sort_values("时刻")
    history_timestamps = pd.to_datetime(history["时刻"]).to_numpy(dtype="datetime64[ns]")
    expected = pd.date_range(history_start, origin, freq="h")
    if len(history) != HISTORY_HOURS:
        if history_start < pd.to_datetime(frame["时刻"]).min():
            raise InsufficientHistoryError(f"warm-up: {day} has only {len(history)} of 168 history rows")
        raise ContractError(f"history for {day} has {len(history)} rows, expected 168")
    if not np.array_equal(history_timestamps, expected.to_numpy(dtype="datetime64[ns]")):
        raise ContractError(f"history timestamps for {day} are not the exact hourly range")
    if pd.Timestamp(history["时刻"].max()) != origin or pd.Timestamp(history["时刻"].min()) != history_start:
        raise ContractError(f"history bounds for {day} differ from canonical origin")
    history_values = history[list(TEMPORAL_FEATURES)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    history_values[~np.isfinite(history_values)] = np.nan

    y = pd.to_numeric(target["target_spread"], errors="coerce").to_numpy(dtype=np.float64)
    if y.shape != (HORIZON,) or not np.isfinite(y).all():
        raise ContractError(f"target Y for {day} must be a finite 24-vector")
    target_direction = (y > 0).astype(np.int8)
    if "target_direction" in target and not np.array_equal(
        pd.to_numeric(target["target_direction"], errors="coerce").to_numpy(dtype=np.int8), target_direction
    ):
        raise ContractError(f"zero/sign policy mismatch in target day {day}")

    return {
        "target_day": day,
        "origin": origin,
        "history_start": history_start,
        "history_end": origin,
        "history_timestamps": pd.DatetimeIndex(history["时刻"]),
        "target_hours": tuple(target.hour_business.astype(int)),
        "X_hist": history_values,
        "X_future": future_values,
        "Y": y,
        "direction": target_direction,
    }


def eligible_training_days(sample_days: list[date], current_target_day: date) -> list[date]:
    """Apply the immutable D-2 complete-label cutoff to supervised day labels."""

    cutoff = training_last_day(current_target_day)
    return sorted(day for day in sample_days if day <= cutoff)


def _prepare_source(root: Path) -> tuple[ResolvedSource, dict[str, list[str]], dict[str, Any]]:
    frozen = root / "data" / "frozen_repro"
    resolved = resolve_sources(
        frozen / "slot_table.parquet",
        root / "data" / "sequence_sources",
        expected_base_sha256=BASE_SHA256,
    )
    audit, groups, registry = inspect_source(resolved, project=root)
    if audit["status"] not in {"PASS", "PARTIAL"}:
        raise ContractError("Source Gate failed: " + "; ".join(audit["errors"]))
    return resolved, groups, registry


def _complete_day_map(frame: pd.DataFrame) -> dict[date, list[int]]:
    return {
        pd.Timestamp(day).date(): sorted(group.hour_business.astype(int).tolist())
        for day, group in frame.groupby("target_day", sort=True)
    }


def audit_contiguous_target_through(day_map: dict[date, list[int]]) -> date | None:
    """Last day in the initial uninterrupted run of complete 24-hour targets."""
    if not day_map:
        return None
    complete = {day for day, hours in day_map.items() if hours == list(range(1, HORIZON + 1))}
    day = min(day_map)
    last = None
    while day in complete:
        last = day
        day += timedelta(days=1)
    return last


def evaluate_selected_manifest_quarantine(day_index: pd.DataFrame, quarantine_ledger: pd.DataFrame,
                                          selected_manifest: dict[str, Any] | None,
                                          feature_registry: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate day use only after an explicitly frozen selector manifest exists."""
    if not isinstance(selected_manifest, dict) or selected_manifest.get("status") != "FROZEN":
        return {"status": "DEFERRED_SELECTOR_NOT_FROZEN", "sample_availability": None,
                "selected_manifest_hash": None, "eligible_sample_indices": [], "excluded_sample_indices": [],
                "formal_training_eligible": False, "training_block_reason": "selector and upstream dependency evidence are not frozen/verified"}
    selected = selected_manifest.get("selected_features")
    if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
        raise ContractError("frozen selector manifest requires unique non-empty selected_features")
    selected_set = set(selected)
    if feature_registry is not None:
        records = {item["feature_name"]: item for item in feature_registry.get("candidate_features", [])}
        missing = sorted(selected_set - set(records))
        if missing:
            raise ContractError(f"frozen selected manifest contains unknown features: {missing}")
        unverified_selected = sorted(name for name in selected_set if records[name].get("dependency_evidence_status") != "VERIFIED_CANONICAL")
    else:
        unverified_selected = []
    relevant = quarantine_ledger.loc[quarantine_ledger.feature_name.isin(selected_set)] if len(quarantine_ledger) else quarantine_ledger
    masked = set(int(x) for x in relevant.sample_index.unique()) if len(relevant) else set()
    sample_indices = set(int(x) for x in day_index.sample_index.tolist())
    excluded = sorted(masked & sample_indices)
    eligible = sorted(sample_indices - masked)
    dep_unverified = bool(unverified_selected or (len(relevant) and (relevant.dependency_evidence_status != "VERIFIED_CANONICAL").any()))
    return {
        "status": "PROVISIONAL_DEPENDENCY_EVIDENCE_UNVERIFIED" if dep_unverified else "FROZEN_SELECTOR_EVALUATED",
        "sample_availability": {i: i in eligible for i in sorted(sample_indices)},
        "selected_manifest_hash": hashlib.sha256(_json_bytes(selected_manifest)).hexdigest(),
        "eligible_sample_indices": eligible, "excluded_sample_indices": excluded,
        "formal_training_eligible": not dep_unverified,
        "unverified_selected_features": unverified_selected,
    }


def _counterfactual_probe(frame: pd.DataFrame, day: date, future_names: list[str]) -> dict[str, Any]:
    baseline = build_day_sample(frame, day, future_names, _canonicalized=True)
    timestamped_actual_cols = [
        name for name in frame.columns if name.lower().startswith("actual_") or "_actual" in name.lower()
    ]

    post14 = frame.copy(deep=True)
    post14_mask = (post14.target_day == day - timedelta(days=1)) & post14.hour_business.between(15, 24)
    post14_count = int(post14_mask.sum())
    if post14_count != 10:
        raise ContractError(f"counterfactual expects D-1 h15-h24 (10 rows), found {post14_count}")
    post14.loc[post14_mask, "target_spread"] = post14.loc[post14_mask, "target_spread"].astype(float) + 1_000_000.0
    post14.loc[post14_mask, "target_direction"] = (post14.loc[post14_mask, "target_spread"] > 0).astype(int)
    for name in timestamped_actual_cols:
        post14.loc[post14_mask, name] = pd.to_numeric(post14.loc[post14_mask, name], errors="coerce") + 1_000_000.0
    post14_result = build_day_sample(post14, day, future_names, _canonicalized=True)

    target_actual = frame.copy(deep=True)
    target_mask = target_actual.target_day == day
    target_actual.loc[target_mask, "target_spread"] = target_actual.loc[target_mask, "target_spread"].astype(float) + 2_000_000.0
    target_actual.loc[target_mask, "target_direction"] = (target_actual.loc[target_mask, "target_spread"] > 0).astype(int)
    for name in timestamped_actual_cols:
        target_actual.loc[target_mask, name] = pd.to_numeric(target_actual.loc[target_mask, name], errors="coerce") + 2_000_000.0
    target_result = build_day_sample(target_actual, day, future_names, _canonicalized=True)

    post14_invariant = np.array_equal(baseline["X_hist"], post14_result["X_hist"], equal_nan=True) and np.array_equal(
        baseline["X_future"], post14_result["X_future"], equal_nan=True
    )
    target_invariant = np.array_equal(baseline["X_hist"], target_result["X_hist"], equal_nan=True) and np.array_equal(
        baseline["X_future"], target_result["X_future"], equal_nan=True
    )
    label_separated = not np.array_equal(baseline["Y"], target_result["Y"])
    return {
        "d1_post14_realized": {
            "status": "PASS" if post14_invariant else "FAIL",
            "mutated_rows": post14_count,
            "mutated_fields": ["target_spread", "target_direction"] + timestamped_actual_cols,
            "X_hist_and_X_future_invariant": post14_invariant,
        },
        "target_day_actual": {
            "status": "PASS" if target_invariant and label_separated else "FAIL",
            "mutated_rows": int(target_mask.sum()),
            "mutated_fields": ["target_spread", "target_direction"] + timestamped_actual_cols,
            "X_hist_and_X_future_invariant": target_invariant,
            "Y_is_separate_label_and_changes_under_counterfactual": label_separated,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sequence_markdown(gate: dict[str, Any]) -> str:
    checks = gate.get("checks", {})
    lines = [
        "# sequence_v1.3 Gate",
        "",
        f"- STATUS: **{gate.get('status')}**",
        f"- supervised samples: `{gate.get('sample_count')}`",
        f"- history: `{gate.get('history_shape')}`",
        f"- future: `{gate.get('future_shape')}`",
        f"- target: `{gate.get('target_shape')}`",
        f"- quarantined target days retained: `{gate.get('quarantined_target_day_count')}`",
        f"- quarantine mask shape/cells: `{gate.get('quarantine_mask_shape')}` / `{gate.get('quarantine_mask_cell_count')}`",
        f"- source gate status: `{gate.get('source_gate_status')}`",
        f"- blocking gaps: `{gate.get('blocking_gaps', [])}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in checks.items())
    if gate.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in gate["errors"])
    return "\n".join(lines) + "\n"


def build_sequence_v1(
    root: Path | None = None,
    output_dir: Path | None = None,
    *,
    run_gate: bool = True,
) -> dict[str, Any]:
    """Build deterministic memory-mappable sequence arrays and the Sequence Gate."""

    root = Path(root) if root is not None else project_root()
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "outputs" / "tabm_v1" / "sequence_v1"
    source_gate = run_source_gate(root, output_dir=output_dir.parent / "source_gate")
    if source_gate.get("status") not in {"PASS", "PARTIAL"}:
        raise ContractError("Source Gate failed; sequence build stopped")

    resolved, groups, registry = _prepare_source(root)
    frame = canonicalize_source(resolved.frame)
    feature_names = candidate_feature_names(groups)
    available_names = set(frame.columns) | set(F10_FEATURES)
    registry = build_feature_registry(
        groups,
        present_columns=available_names,
        legacy_registry_path=root / "data" / "frozen_repro" / "feature_registry.json",
    )
    feature_registry_sha = hashlib.sha256(_json_bytes(registry)).hexdigest()

    by_day = _complete_day_map(frame)
    quarantine_features_by_day = {
        day: sorted(quarantined_features_for_day(registry, day) & set(feature_names)) for day in by_day
    }
    supervised_days: list[date] = []
    exclusions: list[dict[str, str]] = []
    for day, hours in by_day.items():
        if hours != list(range(1, HORIZON + 1)):
            exclusions.append({"target_day": day.isoformat(), "reason": "incomplete_target_day"})
            continue
        supervised_days.append(day)

    hist_arrays: list[np.ndarray] = []
    future_arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    day_rows: list[dict[str, Any]] = []
    warmup_exclusions = 0
    for day in supervised_days:
        try:
            sample = build_day_sample(frame, day, feature_names, _canonicalized=True)
        except InsufficientHistoryError:
            warmup_exclusions += 1
            exclusions.append({"target_day": day.isoformat(), "reason": "insufficient_168h_warmup"})
            continue
        hist_arrays.append(sample["X_hist"])
        future_arrays.append(sample["X_future"])
        labels.append(sample["Y"])
        day_rows.append(
            {
                "sample_index": len(day_rows),
                "target_day": day,
                "forecast_origin": sample["origin"],
                "history_start": sample["history_start"],
                "history_end": sample["history_end"],
                "history_count": len(sample["history_timestamps"]),
                "target_hours": list(sample["target_hours"]),
                "target_hour_count": len(sample["target_hours"]),
                "label_available_for_origins_from": day + timedelta(days=2),
                "future_vintage_evidence_level": FUTURE_VINTAGE_EVIDENCE,
                "inherited_contract_id": INHERITED_CONTRACT_ID,
                "quarantined_feature_count": len(quarantine_features_by_day[day]),
                "zero_count": int((sample["Y"] == 0).sum()),
            }
        )

    if not hist_arrays:
        raise ContractError("no eligible supervised sequence samples were built")
    x_hist = np.stack(hist_arrays).astype(np.float64, copy=False)
    x_future = np.stack(future_arrays).astype(np.float64, copy=False)
    y = np.stack(labels).astype(np.float64, copy=False)
    day_index = pd.DataFrame(day_rows)
    feature_index = {name: i for i, name in enumerate(feature_names)}
    quarantine_mask = np.zeros((len(day_rows), HORIZON, len(feature_names)), dtype=np.bool_)
    ledger_rows: list[dict[str, Any]] = []
    registry_by_name = {item["feature_name"]: item for item in registry["candidate_features"]}
    rules_by_feature = {feature: rule for rule in registry.get("quarantine_rules", []) for feature in rule.get("affected_features", [])}
    for row in day_rows:
        day = pd.Timestamp(row["target_day"]).date()
        for feature in quarantine_features_by_day[day]:
            j = feature_index[feature]
            quarantine_mask[row["sample_index"], :, j] = True
            rule = rules_by_feature[feature]
            record = registry_by_name[feature]
            for hour in range(1, HORIZON + 1):
                ledger_rows.append({
                    "sample_index": row["sample_index"], "target_day": day, "hour_business": hour,
                    "feature_name": feature, "feature_index": j, "status": rule["status"],
                    "reason": rule["reason"], "affected_source_dependency": json.dumps(rule["affected_source_dependency"], ensure_ascii=False),
                    "dependency_closure_start": rule["dependency_closure_start"],
                    "dependency_closure_end": rule["dependency_closure_end"],
                    "dependency_evidence_status": record.get("dependency_evidence_status", "UNVERIFIED"),
                    "dependency_authority": record.get("dependency_authority", "UNKNOWN"),
                })
    quarantine_ledger = pd.DataFrame(ledger_rows, columns=["sample_index", "target_day", "hour_business", "feature_name",
        "feature_index", "status", "reason", "affected_source_dependency", "dependency_closure_start",
        "dependency_closure_end", "dependency_evidence_status", "dependency_authority"])
    # No selector is frozen in this phase: retain all samples and defer training eligibility.
    selector_quarantine_assessment = evaluate_selected_manifest_quarantine(day_index, quarantine_ledger, None, registry)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_hist.npy", x_hist, allow_pickle=False)
    np.save(output_dir / "X_future.npy", x_future, allow_pickle=False)
    np.save(output_dir / "Y.npy", y, allow_pickle=False)
    np.save(output_dir / "quarantine_mask.npy", quarantine_mask, allow_pickle=False)
    day_index.to_parquet(output_dir / "day_index.parquet", index=False)
    quarantine_ledger.to_parquet(output_dir / "quarantine_ledger.parquet", index=False)
    (output_dir / "feature_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    base = canonicalize_source(pd.read_parquet(resolved.base_path))
    base_day_map = _complete_day_map(base)
    complete_base_days = [day for day, hours in base_day_map.items() if hours == list(range(1, 25))]
    partial_base_days = {day.isoformat(): hours for day, hours in base_day_map.items() if hours != list(range(1, 25))}
    complete_merged_days = [day for day, hours in by_day.items() if hours == list(range(1, 25))]
    partial_merged_days = {day.isoformat(): hours for day, hours in by_day.items() if hours != list(range(1, 25))}
    merged_source_last = pd.Timestamp(frame["时刻"].max())
    asset_paths = ["X_hist.npy", "X_future.npy", "Y.npy", "quarantine_mask.npy", "day_index.parquet", "quarantine_ledger.parquet", "feature_registry.json"]
    manifest = {
        "schema": "spread24_sequence_v1.3_manifest",
        "canonical_spec": "docs/17_最终模型设计与编码规范.md",
        "builder_version": SEQUENCE_BUILDER_VERSION,
        "f10_builder_version": F10_BUILDER_VERSION,
        "base_source_asset": "data/frozen_repro/slot_table.parquet",
        "base_source_sha256": resolved.base_sha256,
        "base_manifest_path": "data/frozen_repro/manifest.json",
        "base_manifest_sha256": _sha256_file(root / "data" / "frozen_repro" / "manifest.json"),
        "source_schema_hash": INHERITED_CONTRACT_ID["source_schema_hash"],
        "source_schema_hash_runtime_verification": source_gate.get("feature_contract", {}).get("source_schema_hash_runtime_verification"),
        "feature_registry_hash": INHERITED_CONTRACT_ID["feature_registry_hash"],
        "feature_groups_hash": INHERITED_CONTRACT_ID["feature_groups_hash"],
        "source_builder_version": INHERITED_CONTRACT_ID["source_builder_version"],
        "extension_source_sha256": list(resolved.extension_sha256),
        "extension_records": list(resolved.extension_records),
        "merged_source_sha256": resolved.merged_source_sha256,
        "merged_date_range": {
            "start": min(by_day).isoformat() if by_day else None,
            "end": max(by_day).isoformat() if by_day else None,
        },
        "merge_policy": resolved.merge_policy,
        "duplicate_key_policy": resolved.duplicate_key_policy,
        "source_priority": resolved.source_priority,
        "source_last_timestamp": merged_source_last.isoformat(),
        "base_complete_target_through": max(complete_base_days).isoformat() if complete_base_days else None,
        "base_partial_tail_day": PARTIAL_TAIL_DAY.isoformat() if PARTIAL_TAIL_DAY.isoformat() in partial_base_days else None,
        "base_partial_tail_hours": partial_base_days.get(PARTIAL_TAIL_DAY.isoformat(), []),
        "complete_target_through": max(complete_merged_days).isoformat() if complete_merged_days else None,
        "contiguous_complete_target_through": audit_contiguous_target_through(by_day).isoformat() if audit_contiguous_target_through(by_day) else None,
        "incomplete_target_days": sorted(day.isoformat() for day, hours in by_day.items() if hours != list(range(1, 25))),
        "missing_canonical_keys_count": len([
            (day.date(), hour) for day in pd.date_range(min(by_day), max(by_day), freq="D")
            for hour in range(1, 25) if hour not in set(by_day.get(day.date(), []))
        ]),
        "missing_canonical_keys_hash": hashlib.sha256(_json_bytes([
            (day.date().isoformat(), hour) for day in pd.date_range(min(by_day), max(by_day), freq="D")
            for hour in range(1, 25) if hour not in set(by_day.get(day.date(), []))
        ])).hexdigest(),
        "partial_tail_day": max((pd.Timestamp(day).date() for day in partial_merged_days), default=None).isoformat()
        if partial_merged_days
        else None,
        "partial_tail_hours": partial_merged_days[max(partial_merged_days)] if partial_merged_days else [],
        "forecast_origin": FORECAST_ORIGIN,
        "history_start_rule": "D-8 15:00 inclusive (= origin - 167 hours)",
        "history_end_rule": "D-1 14:00 inclusive; all 24 horizons share this cutoff",
        "history_count": HISTORY_HOURS,
        "history_features": list(TEMPORAL_FEATURES),
        "runtime_provenance": _runtime_provenance(root),
        "future_vintage_rule": "forecast vintage available by D-1 14:00 per frozen source contract",
        "future_vintage_evidence_level": FUTURE_VINTAGE_EVIDENCE,
        "inherited_contract_id": INHERITED_CONTRACT_ID,
        "feature_manifest_hash": feature_registry_sha,
        "selector_policy_version": "DOC17_V1.3_SELECTOR_POLICY_NOT_YET_EXECUTED",
        "zero_policy": ZERO_POLICY,
        "label_cutoff_rule": LABEL_CUTOFF_RULE,
        "data_quality_quarantine": {
            "policy": "sequence layer retains all complete target days; feature/cell mask only; supervised-day eligibility deferred until a frozen selected manifest exists",
            "rules": registry.get("quarantine_rules", []),
            "rule_affected_target_days": sorted({
                day.isoformat() for rule in registry.get("quarantine_rules", [])
                for day in pd.date_range(rule["dependency_closure_start"], rule["dependency_closure_end"], freq="D").date
                if day in by_day
            }),
            "mask_asset_target_days": sorted(day.isoformat() for day in day_index.target_day
                                             if quarantine_features_by_day.get(pd.Timestamp(day).date())),
            "selector_manifest_state": "NOT_FROZEN",
            "selector_quarantine_assessment": selector_quarantine_assessment,
            "ledger_asset": "quarantine_ledger.parquet",
            "mask_asset": "quarantine_mask.npy",
        },
        "partial_target_exclusion": {
            "day": PARTIAL_TAIL_DAY.isoformat(),
            "hours": list(PARTIAL_TAIL_HOURS),
            "supervised": False,
        },
        "target_day_actual_as_feature": False,
        "target_day_DA_as_feature": False,
        "d1_post14_realized_as_feature": False,
        "target_day_spread_in_future_features": False,
        "feature_count_f0_f9": sum(len(names) for names in groups.values()),
        "feature_count_f10": len(F10_FEATURES),
        "candidate_feature_count": len(feature_names),
        "history_feature_count": len(TEMPORAL_FEATURES),
        "sample_count": len(day_index),
        "quarantined_target_day_count": int(day_index.quarantined_feature_count.gt(0).sum()),
        "quarantine_mask_shape": list(quarantine_mask.shape),
        "quarantine_mask_cell_count": int(quarantine_mask.sum()),
        "quarantine_ledger_rows": len(quarantine_ledger),
        "excluded_sample_counts": dict(Counter(row["reason"] for row in exclusions)),
        "warmup_exclusion_count": warmup_exclusions,
        "array_shapes": {"X_hist": list(x_hist.shape), "X_future": list(x_future.shape), "Y": list(y.shape)},
        "array_dtypes": {"X_hist": str(x_hist.dtype), "X_future": str(x_future.dtype), "Y": str(y.dtype)},
        "missing_value_policy": "preserve NaN; do not impute in builder; training-fold-only imputation/scaling later",
        "assets": {path: _sha256_file(output_dir / path) for path in asset_paths},
    }
    _write_json(output_dir / "manifest.json", manifest)

    checks: dict[str, Any] = {
        "source_gate_completed": source_gate.get("status") in {"PASS", "PARTIAL"},
        "exact_168h_timestamp_ranges": bool(
            all(row["history_count"] == 168 and row["history_start"] == row["forecast_origin"] - pd.Timedelta(hours=167) for row in day_rows)
        ),
        "same_history_cutoff_all_horizons": True,
        "supervised_targets_exactly_hours_1_to_24": bool(
            all(tuple(hours) == tuple(range(1, 25)) for hours in day_index.target_hours)
        ),
        "partial_tail_excluded_from_supervised": PARTIAL_TAIL_DAY not in set(day_index.target_day),
        "field_level_registry_complete": registry.get("candidate_feature_count") == 259 and len(registry.get("candidate_features", [])) == 259 and all(all(k in r for k in ("actual_dependency", "forecast_dependency", "label_dependency", "derived_from", "max_dependency_window", "builder_name", "builder_version", "dependency_evidence_status")) for r in registry.get("candidate_features", [])),
        "future_vintage_evidence_recorded": all(row["future_vintage_evidence_level"] in {"EMPIRICALLY_VERIFIED", "CONTRACT_INHERITED"} for row in day_rows),
        "training_label_cutoff_d2": all(
            all(train_day <= current_day - timedelta(days=2) for train_day in eligible_training_days(
                [pd.Timestamp(value).date() for value in day_index.target_day], pd.Timestamp(current_day).date()
            ))
            for current_day in day_index.target_day
        ),
        "quarantine_cells_masked_not_dropped": int(quarantine_mask.sum()) == len(quarantine_ledger),
        "target_day_actual_excluded_from_x": "target_spread" not in feature_names and not any("actual" in name.lower() for name in feature_names),
        "d1_post14_excluded_from_history": bool(all(row["history_end"] == row["forecast_origin"] for row in day_rows)),
        "zero_policy_recorded": manifest["zero_policy"] == ZERO_POLICY,
        "f10_deterministic_contract_recorded": manifest["f10_builder_version"] == F10_BUILDER_VERSION,
        "f10_deterministic_rebuild": False,
        "sequence_assets_hashed": len(manifest["assets"]) == 7,
        "quarantined_days_retained_in_sequence_asset": int(day_index.quarantined_feature_count.gt(0).sum()) > 0,
        "quarantine_mask_ledger_consistent": int(quarantine_mask.sum()) == len(quarantine_ledger),
        "selector_manifest_not_frozen_no_day_decision": selector_quarantine_assessment["status"] == "DEFERRED_SELECTOR_NOT_FROZEN",
        "unresolved_dependency_evidence_fail_closed": not selector_quarantine_assessment.get("formal_training_eligible", True),
        "schema_identity_unverified_not_promoted": source_gate.get("feature_contract", {}).get("source_schema_hash_runtime_verification", {}).get("status") != "VERIFIED" or source_gate.get("status") == "PASS",
    }
    counterfactual = None
    if run_gate:
        counterfactual_day = pd.Timestamp(day_index.iloc[-1].target_day).date()
        f10_probe = _target_day_group(frame, counterfactual_day)
        f10_first = build_f10_for_day(f10_probe).to_numpy(dtype=np.float64)
        f10_second = build_f10_for_day(f10_probe.copy()).to_numpy(dtype=np.float64)
        checks["f10_deterministic_rebuild"] = bool(np.array_equal(f10_first, f10_second))
        counterfactual = _counterfactual_probe(frame, counterfactual_day, feature_names)
        checks["d1_post14_counterfactual"] = counterfactual["d1_post14_realized"]["status"] == "PASS"
        checks["target_day_actual_counterfactual"] = counterfactual["target_day_actual"]["status"] == "PASS"
        checks["target_label_is_separate_from_inputs"] = counterfactual["target_day_actual"]["Y_is_separate_label_and_changes_under_counterfactual"]
    errors = [name for name, passed in checks.items() if passed is not True]
    gate = {
        "schema": "spread24_sequence_gate_v1.3",
        "canonical_spec": "docs/17_最终模型设计与编码规范.md",
        "status": "FAIL" if errors else ("PARTIAL" if source_gate.get("status") == "PARTIAL" else "PASS"),
        "errors": errors,
        "source_gate_status": source_gate.get("status"),
        "blocking_gaps": source_gate.get("blocking_gaps", []),
        "checks": checks,
        "sample_count": len(day_index),
        "history_shape": list(x_hist.shape[1:]),
        "future_shape": list(x_future.shape[1:]),
        "target_shape": list(y.shape[1:]),
        "quarantined_target_day_count": manifest["quarantined_target_day_count"],
        "quarantine_mask_shape": manifest["quarantine_mask_shape"],
        "quarantine_mask_cell_count": manifest["quarantine_mask_cell_count"],
        "quarantine_ledger_rows": manifest["quarantine_ledger_rows"],
        "counterfactual_probe_day": counterfactual_day.isoformat() if counterfactual else None,
        "counterfactual": counterfactual,
        "data_asset_dir": output_dir.resolve().relative_to(root.resolve()).as_posix()
        if output_dir.resolve().is_relative_to(root.resolve())
        else output_dir.as_posix(),
        "manifest_sha256": _sha256_file(output_dir / "manifest.json"),
    }
    _write_json(output_dir.parent / "sequence_gate" / "gate.json", gate)
    (output_dir.parent / "sequence_gate" / "gate.md").write_text(_sequence_markdown(gate), encoding="utf-8")
    return {"manifest": manifest, "gate": gate}

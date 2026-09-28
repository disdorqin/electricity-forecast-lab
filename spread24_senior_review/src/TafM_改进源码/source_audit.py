"""Machine-readable Source Gate for the canonical frozen source contract."""

from __future__ import annotations

from collections import Counter
from datetime import date
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    BASE_ROWS,
    BASE_SHA256,
    COMPLETE_TARGET_THROUGH,
    FEATURE_GROUP_COUNTS,
    FUTURE_VINTAGE_EVIDENCE,
    F10_FEATURES,
    HORIZON,
    INHERITED_CONTRACT_ID,
    PARTIAL_TAIL_DAY,
    PARTIAL_TAIL_HOURS,
    REQUIRED_SOURCE_COLUMNS,
    SOURCE_KEY,
    ContractError,
    project_root,
)
from .registry import build_feature_registry, load_feature_groups
from .source_resolver import (ResolvedSource, canonicalize_source, resolve_sources, sha256_file,
                              schema_identity, SCHEMA_IDENTITY_ALGORITHM)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normal_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _day_counts(frame: pd.DataFrame) -> dict[date, list[int]]:
    return {
        pd.Timestamp(day).date(): sorted(group.hour_business.astype(int).tolist())
        for day, group in frame.groupby("target_day", sort=True)
    }


def inspect_source(
    resolved: ResolvedSource,
    *,
    project: Path,
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    """Return audit summary, frozen groups, and the candidate field registry."""

    project = Path(project)
    frame = canonicalize_source(resolved.frame)
    errors: list[str] = []
    base_raw = pd.read_parquet(resolved.base_path)
    base = canonicalize_source(base_raw)
    base_sha = sha256_file(resolved.base_path)
    if base_sha != BASE_SHA256:
        errors.append(f"frozen base sha256 mismatch: {base_sha}")
    if len(base) != BASE_ROWS:
        errors.append(f"frozen base row count expected {BASE_ROWS}, observed {len(base)}")
    missing_required = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(frame.columns))
    if missing_required:
        errors.append(f"merged source missing required columns: {missing_required}")

    if frame.duplicated(list(SOURCE_KEY)).any():
        errors.append("merged source contains duplicate (target_day,hour_business) keys")
    invalid_hours = sorted(set(frame.hour_business.astype(int)) - set(range(1, 25)))
    if invalid_hours:
        errors.append(f"business hours outside 1..24: {invalid_hours}")

    base_counts = _day_counts(base)
    complete_days = [day for day, hours in base_counts.items() if hours == list(range(1, 25))]
    partial_days = {day.isoformat(): hours for day, hours in base_counts.items() if hours != list(range(1, 25))}
    if len(complete_days) != 1694:
        errors.append(f"frozen base complete target days expected 1694, observed {len(complete_days)}")
    if max(complete_days, default=None) != COMPLETE_TARGET_THROUGH:
        errors.append("frozen base complete_target_through differs from docs/17")
    if partial_days != {PARTIAL_TAIL_DAY.isoformat(): list(PARTIAL_TAIL_HOURS)}:
        errors.append(f"frozen base partial tail differs from docs/17: {partial_days}")
    base_last_timestamp = pd.to_datetime(base["时刻"]).max()
    if base_last_timestamp != pd.Timestamp("2026-08-22 14:00:00"):
        errors.append(f"frozen base source_last_timestamp differs from docs/17: {base_last_timestamp}")

    expected_timestamps = pd.to_datetime(frame.target_day) + pd.to_timedelta(frame.hour_business, unit="h")
    actual_timestamps = pd.to_datetime(frame["时刻"])
    timestamp_mapping_mismatches = int((expected_timestamps.to_numpy() != actual_timestamps.to_numpy()).sum())
    if timestamp_mapping_mismatches:
        errors.append(f"business-hour/timestamp mapping mismatch rows: {timestamp_mapping_mismatches}")
    if frame["时刻"].duplicated().any():
        errors.append("source has duplicate canonical timestamps")

    # Keep the historical zero policy explicit: direction is positive iff spread > 0.
    spread = pd.to_numeric(frame.target_spread, errors="coerce").to_numpy(dtype=np.float64)
    direction = pd.to_numeric(frame.target_direction, errors="coerce").to_numpy(dtype=np.float64)
    nonfinite_labels = int((~np.isfinite(spread) | ~np.isfinite(direction)).sum())
    direction_mismatches = int((direction != (spread > 0).astype(np.float64)).sum())
    if nonfinite_labels:
        errors.append(f"non-finite target rows: {nonfinite_labels}")
    if direction_mismatches:
        errors.append(f"target_direction mismatches target_spread>0 on {direction_mismatches} rows")

    frozen_dir = project / "data" / "frozen_repro"
    groups_path = frozen_dir / "feature_groups.json"
    legacy_registry_path = frozen_dir / "feature_registry.json"
    try:
        groups = load_feature_groups(groups_path)
        registry = build_feature_registry(
            groups,
            present_columns=set(frame.columns) | set(F10_FEATURES),
            legacy_registry_path=legacy_registry_path,
        )
    except Exception as exc:
        groups = {}
        registry = {}
        errors.append(f"frozen feature inventory invalid: {type(exc).__name__}: {exc}")

    f0_f9_names = [name for names in groups.values() for name in names]
    missing_features = sorted(set(f0_f9_names) - set(frame.columns))
    if missing_features:
        errors.append(f"F0-F9 source columns missing: {missing_features}")
    feature_count_by_group = {name: len(names) for name, names in groups.items()}
    if feature_count_by_group != FEATURE_GROUP_COUNTS:
        errors.append(f"F0-F9 group counts mismatch: {feature_count_by_group}")

    candidate_names = {record["feature_name"] for record in registry.get("candidate_features", [])}
    forbidden_feature_names = sorted(
        name for name in candidate_names if "actual" in name.lower() or name in {"target_spread", "target_direction"}
    )
    if forbidden_feature_names:
        errors.append(f"target actual/label fields entered candidate features: {forbidden_feature_names}")
    feature_present_actual = sorted(name for name in candidate_names if name.lower().startswith("actual_"))
    if feature_present_actual:
        errors.append(f"actual_* fields entered candidate features: {feature_present_actual}")

    registry_by_name = {
        record.get("feature_name"): record for record in registry.get("candidate_features", [])
    }
    f1_f8_names = groups.get("F1", []) + groups.get("F8", [])
    d1_context_registry_ok = bool(f1_f8_names) and all(
        registry_by_name.get(name, {}).get("legacy_availability") == "D-1 <=14:00"
        and registry_by_name.get(name, {}).get("legacy_source") == "D-1 spread p1-p14"
        for name in f1_f8_names
    )
    if not d1_context_registry_ok:
        errors.append("F1/F8 frozen registry must source only D-1 spread p1-p14 available by 14:00")
    target_forecast_names = groups.get("F2", [])
    target_forecast_registry_ok = bool(target_forecast_names) and all(
        registry_by_name.get(name, {}).get("legacy_availability") == "target D forecast known at origin"
        for name in target_forecast_names
    )
    if not target_forecast_registry_ok:
        errors.append("F2 frozen registry does not consistently assert target-day forecast available at origin")

    context_cutoff_audit = {
        "column_present": "context_source_max_ds" in frame.columns,
        "target_days_checked": 0,
        "missing_context_days": [],
        "cutoff_mismatch_days": [],
    }
    if "context_source_max_ds" in frame.columns:
        context = frame[["target_day", "context_source_max_ds"]].copy()
        context["context_source_max_ds"] = pd.to_datetime(context["context_source_max_ds"], errors="coerce")
        per_day = context.groupby("target_day")["context_source_max_ds"].max().sort_index()
        missing_context_days = [pd.Timestamp(day).date() for day, value in per_day.items() if pd.isna(value)]
        present_context = {pd.Timestamp(day).date(): value for day, value in per_day.items() if pd.notna(value)}
        cutoff_mismatches = [
            day.isoformat()
            for day, value in present_context.items()
            if value != pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        ]
        if missing_context_days and present_context:
            first_present = min(present_context)
            if any(day >= first_present for day in missing_context_days):
                errors.append("context_source_max_ds missing after context becomes available")
        if cutoff_mismatches:
            errors.append(f"context_source_max_ds exceeds/differs from D-1 14:00 for target days: {cutoff_mismatches[:5]}")
        context_cutoff_audit = {
            "column_present": True,
            "target_days_checked": len(per_day),
            "missing_context_days": [day.isoformat() for day in missing_context_days],
            "cutoff_mismatch_days": cutoff_mismatches,
            "nonmissing_context_max_timestamp": max(present_context.values()).isoformat() if present_context else None,
            "rule": "context_source_max_ds == target_day-1 14:00 when context exists",
        }

    source_manifest = _normal_manifest(frozen_dir / "manifest.json")
    if source_manifest.get("target_day_actual_as_feature") is not False:
        errors.append("frozen source manifest does not explicitly forbid target-day actual")
    if source_manifest.get("d1_post14_realized_as_feature") is not False:
        errors.append("frozen source manifest does not explicitly forbid D-1 post14 realized features")
    if source_manifest.get("target_day_DA_as_feature") is not False:
        errors.append("frozen source manifest does not explicitly forbid target-day DA features")
    if source_manifest.get("forecast_origin") != "D-1 14:00":
        errors.append("frozen source manifest forecast origin differs from canonical contract")
    if source_manifest.get("training_label_cutoff") != "D-2 completed days only":
        errors.append("frozen source manifest training label cutoff differs from canonical contract")

    vintage_columns = [
        name for name in frame.columns
        if any(token in name.lower() for token in ("issue_time", "available_at", "generated_at", "version_timestamp", "vintage"))
    ]
    future_vintage_count = Counter(
        record.get("future_vintage_evidence_level") for record in registry.get("candidate_features", [])
    )
    if "UNVERIFIED" in future_vintage_count:
        errors.append("UNVERIFIED future feature source is forbidden")
    required_dependency_fields = ("actual_dependency", "forecast_dependency", "label_dependency", "derived_from",
                                  "max_dependency_window", "builder_name", "builder_version", "dependency_evidence_status")
    dependency_records = registry.get("candidate_features", [])
    dependency_complete = len(dependency_records) == 259 and all(
        all(key in row and row[key] is not None for key in required_dependency_fields) for row in dependency_records
    )
    if not dependency_complete:
        errors.append("feature dependency registry is not semantically complete for all 259 features")
    if not all(isinstance(row.get("inherited_contract_id"), dict) for row in dependency_records):
        errors.append("candidate feature lacks machine-readable inherited source identity")
    dependency_evidence_counts = Counter(row.get("dependency_evidence_status", "MISSING") for row in dependency_records)
    evidence_bundle_status = registry.get("upstream_builder_evidence_status", {})
    accepted_dependency_levels = {"VERIFIED_CANONICAL_UPSTREAM", "VERIFIED_CANONICAL"}
    dependency_evidence_status = (
        "VERIFIED_CANONICAL" if len(dependency_records) == 259
        and all(row.get("dependency_evidence_status") in accepted_dependency_levels for row in dependency_records)
        and dependency_evidence_counts.get("VERIFIED_CANONICAL_UPSTREAM", 0) == 244
        and dependency_evidence_counts.get("VERIFIED_CANONICAL", 0) == 15
        and evidence_bundle_status.get("status") == "VERIFIED"
        else "PARTIAL_UNVERIFIED"
    )

    identity_dir = project / "data" / "frozen_repro"
    actual_source_identity = {
        **INHERITED_CONTRACT_ID,
        "source_manifest_sha256": sha256_file(identity_dir / "manifest.json"),
        "feature_registry_hash": sha256_file(identity_dir / "feature_registry.json"),
        "feature_groups_hash": sha256_file(identity_dir / "feature_groups.json"),
    }
    for key in ("source_sha256", "source_manifest_sha256", "feature_registry_hash", "feature_groups_hash"):
        if actual_source_identity[key] != INHERITED_CONTRACT_ID[key]:
            errors.append(f"inherited source identity hash mismatch: {key}")
    runtime_schema_hash = schema_identity(base_raw)
    schema_hash_runtime = {
        "declared_in_doc17": INHERITED_CONTRACT_ID["source_schema_hash"],
        "runtime_hash": runtime_schema_hash,
        "runtime_algorithm": SCHEMA_IDENTITY_ALGORITHM,
        "runtime_equals_declared": runtime_schema_hash == INHERITED_CONTRACT_ID["source_schema_hash"],
        "status": "VERIFIED" if runtime_schema_hash == INHERITED_CONTRACT_ID["source_schema_hash"] else "INHERITED_UNVERIFIED_ALGORITHM_NOT_RECOVERED",
    }
    identity_gaps = []
    if dependency_evidence_status != "VERIFIED_CANONICAL":
        identity_gaps.append("feature_dependency_graph_not_upstream_verified")
    if schema_hash_runtime["status"] != "VERIFIED":
        identity_gaps.append("source_schema_hash_algorithm_not_recovered")
    source_gate_status = "FAIL" if errors else ("PARTIAL" if identity_gaps else "PASS")

    merged_counts = _day_counts(frame)
    complete_days_set = {day for day, hours in merged_counts.items() if hours == list(range(1, 25))}
    contiguous_through = None
    if merged_counts:
        cur = min(merged_counts)
        while cur in complete_days_set:
            contiguous_through = cur
            from datetime import timedelta
            cur += timedelta(days=1)
    expected_keys = {(day.date(), hour) for day in pd.date_range(min(merged_counts), max(merged_counts), freq="D") for hour in range(1, 25)} if merged_counts else set()
    missing_keys = sorted(expected_keys - set(zip(frame.target_day, frame.hour_business.astype(int))))
    missing_key_hash = hashlib.sha256(json.dumps([(d.isoformat(), h) for d, h in missing_keys], separators=(",", ":")).encode()).hexdigest()
    merged_complete_through = max(
        (day for day, hours in merged_counts.items() if hours == list(range(1, 25))), default=None
    )
    merged_partial = {day.isoformat(): hours for day, hours in merged_counts.items() if hours != list(range(1, 25))}

    audit = {
        "schema": "spread24_source_gate_v1.3",
        "canonical_spec": "docs/17_最终模型设计与编码规范.md",
        "gate": "SOURCE_GATE",
        "status": source_gate_status,
        "errors": errors,
        "blocking_gaps": identity_gaps,
        "base": {
            "path": "data/frozen_repro/slot_table.parquet",
            "sha256": base_sha,
            "expected_sha256": BASE_SHA256,
            "rows": int(len(base)),
            "target_day_count": int(base.target_day.nunique()),
            "complete_target_day_count": len(complete_days),
            "complete_target_through": max(complete_days).isoformat() if complete_days else None,
            "partial_tail_day": PARTIAL_TAIL_DAY.isoformat(),
            "partial_tail_hours": list(PARTIAL_TAIL_HOURS),
            "source_last_timestamp": pd.Timestamp(base_last_timestamp).isoformat(),
            "immutable": True,
        },
        "merged_source": {
            "sha256": resolved.merged_source_sha256,
            "rows": int(len(frame)),
            "date_min": min(merged_counts).isoformat() if merged_counts else None,
            "date_max": max(merged_counts).isoformat() if merged_counts else None,
            "complete_target_through": merged_complete_through.isoformat() if merged_complete_through else None,
            "contiguous_complete_target_through": contiguous_through.isoformat() if contiguous_through else None,
            "incomplete_target_days": sorted(day.isoformat() for day, hours in merged_counts.items() if hours != list(range(1, 25))),
            "missing_canonical_keys_count": len(missing_keys),
            "missing_canonical_keys_hash": missing_key_hash,
            "partial_days": merged_partial,
            "duplicate_key_count": int(frame.duplicated(list(SOURCE_KEY)).sum()),
            "duplicate_timestamp_count": int(frame["时刻"].duplicated().sum()),
            "timestamp_mapping_mismatches": timestamp_mapping_mismatches,
            "extension_records": list(resolved.extension_records),
            "extension_sha256": list(resolved.extension_sha256),
            "merge_policy": resolved.merge_policy,
            "duplicate_key_policy": resolved.duplicate_key_policy,
            "source_priority": resolved.source_priority,
            "append_correction_mechanism": "IMPLEMENTED; append duplicates FAIL; corrections require exact key declaration",
        },
        "feature_contract": {
            "f0_f9_group_counts": feature_count_by_group,
            "f0_f9_feature_count": len(f0_f9_names),
            "f10_feature_count": 15,
            "candidate_feature_count": registry.get("candidate_feature_count"),
            "dependency_registry_count": len(dependency_records),
            "dependency_registry_complete": dependency_complete,
            "dependency_evidence_status": dependency_evidence_status,
            "dependency_evidence_counts": dict(dependency_evidence_counts),
            "upstream_builder_evidence": evidence_bundle_status,
            "dependency_evidence_levels_allowed": sorted(accepted_dependency_levels),
            "source_schema_hash_runtime_verification": schema_hash_runtime,
            "machine_readable_source_identity": all(actual_source_identity.get(k) == INHERITED_CONTRACT_ID[k] for k in INHERITED_CONTRACT_ID),
            "identity_contract_recorded": True,
            "source_asset_hashes_runtime_verified": all(actual_source_identity[k] == INHERITED_CONTRACT_ID[k] for k in ("source_sha256", "source_manifest_sha256", "feature_registry_hash", "feature_groups_hash")),
            "source_identity": actual_source_identity,
            "extension_semantic_compatibility": "SYNTHETIC_VALIDATION_ONLY_NO_EXTENSIONS" if not resolved.extension_records else ("PASS" if all(r.get("semantic_compatibility") == "PASS" for r in resolved.extension_records) else "FAIL"),
            "correction_closure_validation": "SYNTHETIC_VALIDATION_ONLY_NO_REAL_CORRECTIONS" if not any(r.get("type") == "correction" for r in resolved.extension_records) else ("PASS" if all(r.get("type") != "correction" or r.get("dependency_closure") == "UPSTREAM_EVIDENCE_HASH_VALIDATED" for r in resolved.extension_records) else "FAIL"),
            "global_quarantine_removed": True,
            "quarantined_feature_names": sorted({name for rule in registry.get("quarantine_rules", []) for name in rule.get("affected_features", [])}),
            "unaffected_f7_f9_not_quarantined": not any(record.get("feature_group") in {"F7", "F9"} and record.get("feature_name") in {name for rule in registry.get("quarantine_rules", []) for name in rule.get("affected_features", [])} for record in dependency_records),
            "history_temporal_feature_count": registry.get("history_temporal_feature_count"),
            "target_day_actual_as_feature": False,
            "d1_post14_realized_as_feature": False,
            "target_day_spread_as_feature": False,
            "d1_context_registry_contract_pass": d1_context_registry_ok,
            "target_forecast_registry_contract_pass": target_forecast_registry_ok,
            "d1_context_max_timestamp_audit": context_cutoff_audit,
            "future_vintage_evidence_counts": dict(future_vintage_count),
            "future_vintage_columns_available_for_empirical_check": vintage_columns,
            "future_vintage_inherited_contract_id": INHERITED_CONTRACT_ID,
        },
        "data_quality_quarantine": {"rule": registry.get("quarantine_rules", []), "policy": "feature/cell-level; no date-wide exclusion"},
        "label_contract": {
            "target": "DA-RT",
            "forecast_origin": "D-1 14:00",
            "complete_label_cutoff": "D-2 or earlier",
            "target_day_actual_as_feature": False,
            "d1_post14_realized_as_feature": False,
            "target_day_DA_as_feature": False,
            "zero_policy": "Y>0 positive; Y<=0 non-positive",
            "nonfinite_target_rows": nonfinite_labels,
            "direction_mismatch_rows": direction_mismatches,
        },
        "training_contract_readiness": {
            "target_scale": "BASE_TRAIN-only IQR(Y)/1.349, lower bounded by eps; Stage B freezes Stage-A scale",
            "direction_class_weights": "BASE_TRAIN-only; Stage B freezes Stage-A weights",
            "empty_severity_mask": "only average classes present in batch; never mean an empty mask",
        },
    }
    return audit, groups, registry


def _render_markdown(audit: dict[str, Any]) -> str:
    base = audit.get("base", {})
    merged = audit.get("merged_source", {})
    lines = [
        "# Source Gate Audit",
        "",
        f"- STATUS: **{audit.get('status')}**",
        f"- Frozen base SHA256: `{base.get('sha256', 'n/a')}`",
        f"- Base rows: `{base.get('rows', 'n/a')}`",
        f"- Complete target through: `{base.get('complete_target_through', 'n/a')}`",
        f"- Partial tail: `{base.get('partial_tail_day', 'n/a')}` hours `{base.get('partial_tail_hours', [])}`",
        f"- Merged source rows: `{merged.get('rows', 'n/a')}`",
        f"- Merged source SHA256: `{merged.get('sha256', 'n/a')}`",
        f"- Extensions: `{len(merged.get('extension_records', []))}`",
        f"- Dependency evidence: `{audit.get('feature_contract', {}).get('dependency_evidence_status', 'n/a')}`",
        f"- Schema hash verification: `{audit.get('feature_contract', {}).get('source_schema_hash_runtime_verification', {}).get('status', 'n/a')}`",
        f"- Blocking gaps: `{audit.get('blocking_gaps', [])}`",
        f"- Vintage evidence: `{audit.get('feature_contract', {}).get('future_vintage_evidence_counts', {})}`",
        f"- Quarantine starts: `{audit.get('data_quality_quarantine', {}).get('start', 'n/a')}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {error}" for error in audit.get("errors", [])] or ["- None"])
    lines.extend(["", "## Boundary assertions", "", "- target = DA - RT", "- forecast origin = D-1 14:00", "- complete labels <= D-2", "- target-day actual / DA and D-1 h15-h24 realized are not candidate features", "- frozen base file is read-only; extensions merge through immutable append/correction sources"])
    return "\n".join(lines) + "\n"


def run_source_gate(
    root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    root = Path(root) if root is not None else project_root()
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "outputs" / "tabm_v1" / "source_gate"
    try:
        frozen = root / "data" / "frozen_repro"
        resolved = resolve_sources(
            frozen / "slot_table.parquet",
            root / "data" / "sequence_sources",
            expected_base_sha256=BASE_SHA256,
        )
        audit, _, _ = inspect_source(resolved, project=root)
    except Exception as exc:
        audit = {
            "schema": "spread24_source_gate_v1",
            "canonical_spec": "docs/17_最终模型设计与编码规范.md",
            "gate": "SOURCE_GATE",
            "status": "FAIL",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    _write_json(output_dir / "source_audit.json", audit)
    (output_dir / "source_audit.md").write_text(_render_markdown(audit), encoding="utf-8")
    return audit

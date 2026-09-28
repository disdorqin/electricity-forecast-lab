"""Read-only base plus manifest-governed immutable source extensions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import BASE_SHA256, INHERITED_CONTRACT_ID, ContractError, REQUIRED_SOURCE_COLUMNS, SOURCE_KEY, project_root


SCHEMA_IDENTITY_ALGORITHM = "sha256(utf8(json.dumps([(str(column),str(dtype)),...],ensure_ascii=False,separators=(',',':'))))"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_source(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_SOURCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ContractError(f"source missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["target_day"] = pd.to_datetime(result["target_day"], errors="raise").dt.date
    result["时刻"] = pd.to_datetime(result["时刻"], errors="raise")
    hours = pd.to_numeric(result["hour_business"], errors="raise")
    if (hours % 1 != 0).any():
        raise ContractError("hour_business must contain integer business hours")
    result["hour_business"] = hours.astype("int64")
    return result


@dataclass(frozen=True)
class ResolvedSource:
    frame: pd.DataFrame
    base_path: Path
    base_sha256: str
    extension_sha256: tuple[str, ...]
    extension_records: tuple[dict[str, Any], ...]
    merged_source_sha256: str
    merge_policy: str = "append_then_exact_key_correction; base file remains immutable"
    duplicate_key_policy: str = "append duplicates fail; correction requires exact declared keys"
    source_priority: str = "correction > append > base"


def _load_extension(extension_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    data_path = extension_dir / "slot_table_extension.parquet"
    manifest_path = extension_dir / "manifest.json"
    if not data_path.is_file() or not manifest_path.is_file():
        raise ContractError(f"incomplete extension directory: {extension_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ContractError(f"extension manifest must be a JSON object: {manifest_path}")
    extension_hash = sha256_file(data_path)
    declared_hash = manifest.get("slot_table_sha256", manifest.get("sha256"))
    if not declared_hash or declared_hash != extension_hash:
        raise ContractError(f"extension SHA256 missing/mismatch: {extension_dir.name}")
    ext_type = manifest.get("type")
    if ext_type not in {"append", "correction"}:
        raise ContractError(f"extension type must be append/correction: {extension_dir.name}")
    manifest = dict(manifest)
    manifest["extension_id"] = manifest.get("extension_id", extension_dir.name)
    manifest["type"] = ext_type
    frame = canonicalize_source(pd.read_parquet(data_path))
    _validate_semantic_compatibility(frame, manifest, extension_dir.name)
    return frame, manifest, extension_hash


def schema_identity(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    payload = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_key_hash(keys: set[tuple[Any, int]] | list[tuple[Any, int]]) -> str:
    normalized = sorted((pd.Timestamp(day).date().isoformat(), int(hour)) for day, hour in keys)
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _list_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_semantic_compatibility(frame: pd.DataFrame, manifest: dict[str, Any], extension_id: str) -> None:
    """Fail closed unless an extension declares the canonical schema and meanings."""
    identity = manifest.get("semantic_identity", {})
    expected = {
        "canonical_key": list(SOURCE_KEY),
        "source_schema_hash": INHERITED_CONTRACT_ID["source_schema_hash"],
        "feature_registry_hash": INHERITED_CONTRACT_ID["feature_registry_hash"],
        "feature_groups_hash": INHERITED_CONTRACT_ID["feature_groups_hash"],
        "source_sign_convention": "DA-RT",
    }
    if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in expected.items()):
        raise ContractError(f"extension semantic_identity incompatible/incomplete: {extension_id}")
    semantics = manifest.get("column_semantics")
    if not isinstance(semantics, dict) or not set(frame.columns).issubset(semantics):
        raise ContractError(f"extension column_semantics must declare every column: {extension_id}")
    root = project_root()
    registry_path = root / "data" / "frozen_repro" / "feature_registry.json"
    frozen = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {"features": []}
    expected_columns = {item["feature"]: {k: item.get(k) for k in ("group", "source", "availability", "transform")}
                        for item in frozen.get("features", []) if "feature" in item}
    expected_columns.update({
        "target_day": {"role": "canonical target date key", "format": "YYYY-MM-DD"},
        "时刻": {"role": "canonical timestamp", "semantics": "target_day plus hour_business"},
        "hour_business": {"role": "canonical hour key", "domain": "1..24"},
        "period": {"role": "inherited market period", "semantics": "frozen source period code"},
        "target_spread": {"role": "native source target", "sign": "DA-RT"},
        "target_direction": {"role": "native target direction", "formula": "1[target_spread>0]"},
        "context_source_max_ds": {"role": "D-1 context max timestamp", "cutoff": "D-1 14:00"},
    })
    for column in frame.columns:
        expected_semantics = expected_columns.get(column)
        if expected_semantics is not None and semantics[column] != expected_semantics:
            raise ContractError(f"extension same-name column source/availability semantics mismatch for {column}: {extension_id}")


def _key_set(frame: pd.DataFrame) -> set[tuple[Any, int]]:
    return set(zip(frame["target_day"], frame["hour_business"].astype(int)))


def _declared_exact_keys(manifest: dict[str, Any]) -> set[tuple[Any, int]]:
    raw = manifest.get("exact_corrected_keys", manifest.get("exact_keys"))
    if not isinstance(raw, list) or not raw:
        raise ContractError("correction manifest must declare non-empty exact_corrected_keys")
    keys: set[tuple[Any, int]] = set()
    for item in raw:
        if not isinstance(item, dict) or "target_day" not in item or "hour_business" not in item:
            raise ContractError("exact_corrected_keys items require target_day/hour_business")
        day = pd.Timestamp(item["target_day"]).date()
        hour = int(item["hour_business"])
        keys.add((day, hour))
    if len(keys) != len(raw):
        raise ContractError("correction exact_corrected_keys contain duplicates")
    return keys


def resolve_sources(
    base_path: Path,
    extensions_root: Path | None = None,
    *,
    expected_base_sha256: str | None = BASE_SHA256,
) -> ResolvedSource:
    """Load and merge source files; the frozen base is only ever read.

    Extension manifest contract implemented here:
    - append: ``type``, ``sha256`` (or ``slot_table_sha256``), optional ``extension_id``;
    - correction: append fields plus ``supersedes_source_sha256``, ``reason`` and
      exact ``exact_corrected_keys`` matching every/only key overridden.
    """

    base_path = Path(base_path)
    if not base_path.is_file():
        raise ContractError(f"base source not found: {base_path}")
    base_hash = sha256_file(base_path)
    if expected_base_sha256 is not None and base_hash != expected_base_sha256:
        raise ContractError(
            f"frozen base SHA256 changed: expected {expected_base_sha256}, got {base_hash}"
        )
    base = canonicalize_source(pd.read_parquet(base_path))
    if base.duplicated(list(SOURCE_KEY)).any():
        raise ContractError("duplicate key in immutable base source")

    extension_dirs = (
        sorted(p for p in Path(extensions_root).glob("extension_*") if p.is_dir())
        if extensions_root is not None and Path(extensions_root).exists()
        else []
    )
    if extension_dirs and expected_base_sha256 == BASE_SHA256 and schema_identity(pd.read_parquet(base_path)) != INHERITED_CONTRACT_ID["source_schema_hash"]:
        raise ContractError("extension merge blocked: canonical source_schema_hash algorithm is not runtime-verified")
    extensions = []
    for extension_dir in extension_dirs:
        data, manifest, ext_hash = _load_extension(extension_dir)
        if set(data.columns) != set(base.columns):
            raise ContractError(f"extension schema differs from frozen base: {extension_dir.name}")
        if data.duplicated(list(SOURCE_KEY)).any():
            raise ContractError(f"duplicate key within extension: {extension_dir.name}")
        extensions.append((data, manifest, ext_hash, extension_dir))

    merged = base
    extension_records: list[dict[str, Any]] = []
    append_hashes: set[str] = set()

    # Appends are applied first. An append is never allowed to shadow any source key.
    for ext, manifest, ext_hash, ext_dir in extensions:
        if manifest["type"] != "append":
            continue
        keys = _key_set(ext)
        overlap = keys & _key_set(merged)
        if overlap:
            raise ContractError(f"append duplicates existing keys ({len(overlap)}): {ext_dir.name}")
        merged = pd.concat([merged, ext], ignore_index=True, sort=False)
        append_hashes.add(ext_hash)
        extension_records.append({"extension_id": manifest["extension_id"], "type": "append", "sha256": ext_hash, "semantic_compatibility": "PASS"})

    # Corrections have higher priority than base/append, but can only replace declared keys.
    corrected_keys: set[tuple[Any, int]] = set()
    for ext, manifest, ext_hash, ext_dir in extensions:
        if manifest["type"] != "correction":
            continue
        supersedes = manifest.get("supersedes_source_sha256")
        if supersedes not in ({base_hash} | append_hashes):
            raise ContractError(f"correction has unknown supersedes_source_sha256: {ext_dir.name}")
        if not str(manifest.get("reason", "")).strip():
            raise ContractError(f"correction reason is required: {ext_dir.name}")
        declared = _declared_exact_keys(manifest)
        actual = _key_set(ext)
        closure_level = _validate_correction_closure(manifest, ext_dir.name, actual_keys=actual, require_upstream=True)
        overlap = actual & _key_set(merged)
        if actual != declared or overlap != actual:
            raise ContractError(
                f"correction keys must equal declared exact keys and existing keys: {ext_dir.name}"
            )
        if corrected_keys & actual:
            raise ContractError(f"overlapping correction extensions are ambiguous: {ext_dir.name}")
        corrected_keys |= actual
        merged_index = pd.MultiIndex.from_frame(merged[list(SOURCE_KEY)])
        ext_index = pd.MultiIndex.from_frame(ext[list(SOURCE_KEY)])
        replacement = ext.set_index(list(SOURCE_KEY))
        merged = merged.set_index(list(SOURCE_KEY))
        merged.loc[ext_index, :] = replacement.loc[ext_index, merged.columns].to_numpy()
        merged = merged.reset_index()
        extension_records.append(
            {"extension_id": manifest["extension_id"], "type": "correction", "sha256": ext_hash,
             "supersedes_source_sha256": supersedes, "corrected_key_count": len(actual),
             "reason": manifest["reason"], "semantic_compatibility": "PASS", "dependency_closure": closure_level}
        )

    merged = canonicalize_source(merged).sort_values(list(SOURCE_KEY)).reset_index(drop=True)
    if merged.duplicated(list(SOURCE_KEY)).any():
        raise ContractError("merged source contains duplicate keys")
    provenance = {
        "base_sha256": base_hash,
        "extensions": sorted((r["extension_id"], r["type"], r["sha256"]) for r in extension_records),
        "merge_policy": "append_then_exact_key_correction; base file remains immutable",
        "duplicate_key_policy": "append duplicates fail; correction requires exact declared keys",
        "source_priority": "correction > append > base",
    }
    merged_hash = hashlib.sha256(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ResolvedSource(
        frame=merged,
        base_path=base_path,
        base_sha256=base_hash,
        extension_sha256=tuple(r["sha256"] for r in extension_records),
        extension_records=tuple(extension_records),
        merged_source_sha256=merged_hash,
    )


def _validate_correction_closure(manifest: dict[str, Any], extension_id: str, *,
                                  actual_keys: set[tuple[Any, int]] | None = None,
                                  require_upstream: bool = False) -> str:
    required = ("correction_type", "root_bad_keys", "affected_features", "affected_source_dependency",
                "dependency_closure_start", "dependency_closure_end", "recomputed_keys",
                "recomputed_feature_groups", "affected_feature_groups", "source_builder_version", "supersedes_source_sha256", "reason",
                "exact_corrected_keys", "dependency_closure_evidence")
    missing = [key for key in required if manifest.get(key) in (None, "", [])]
    if missing:
        raise ContractError(f"correction dependency-closure evidence missing {missing}: {extension_id}")
    if not isinstance(manifest["root_bad_keys"], list) or not isinstance(manifest["recomputed_keys"], list):
        raise ContractError(f"correction root_bad_keys/recomputed_keys must be lists: {extension_id}")
    start, end = pd.Timestamp(manifest["dependency_closure_start"]), pd.Timestamp(manifest["dependency_closure_end"])
    if start > end:
        raise ContractError(f"correction dependency closure has reversed bounds: {extension_id}")
    def keypair(item):
        if isinstance(item, dict):
            return (str(pd.Timestamp(item.get("target_day")).date()), int(item.get("hour_business")))
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return (str(pd.Timestamp(item[0]).date()), int(item[1]))
        raise ContractError(f"invalid key in correction closure evidence: {extension_id}")
    roots = {keypair(item) for item in manifest["root_bad_keys"]}
    recomputed = {keypair(item) for item in manifest["recomputed_keys"]}
    exact = {keypair(item) for item in manifest["exact_corrected_keys"]}
    if len(roots) != len(manifest["root_bad_keys"]) or len(recomputed) != len(manifest["recomputed_keys"]) or len(exact) != len(manifest["exact_corrected_keys"]):
        raise ContractError(f"duplicate keys in correction closure declarations: {extension_id}")
    if not roots.issubset(recomputed):
        raise ContractError(f"correction recomputed_keys omit declared root keys: {extension_id}")
    if recomputed != exact:
        raise ContractError(f"exact_corrected_keys must equal recomputed_keys for full-row closure replacement: {extension_id}")
    if actual_keys is not None:
        normalized_actual = {(str(pd.Timestamp(d).date()), int(h)) for d, h in actual_keys}
        if normalized_actual != exact:
            raise ContractError(f"extension actual keys must equal exact_corrected_keys and recomputed_keys: {extension_id}")
    groups, features, dependencies = manifest["recomputed_feature_groups"], manifest["affected_features"], manifest["affected_source_dependency"]
    if not isinstance(groups, list) or not groups or not isinstance(features, list) or not features or not isinstance(dependencies, list) or not dependencies:
        raise ContractError(f"correction features/groups/dependencies must be non-empty lists: {extension_id}")
    feature_groups = manifest["affected_feature_groups"]
    if not isinstance(feature_groups, dict) or set(feature_groups) != set(features) or not set(feature_groups.values()).issubset(set(groups)):
        raise ContractError(f"affected_feature_groups must bind every affected feature to a recomputed group: {extension_id}")
    evidence = manifest["dependency_closure_evidence"]
    if not isinstance(evidence, dict):
        raise ContractError(f"dependency_closure_evidence must be an object: {extension_id}")
    e_required = ("evidence_type", "authority", "source_builder_version", "closure_start", "closure_end",
                  "recomputed_keys_count", "recomputed_keys_sha256", "exact_corrected_keys_count",
                  "exact_corrected_keys_sha256", "affected_features_sha256", "affected_source_dependency_sha256",
                  "recomputed_feature_groups_sha256", "affected_feature_groups_sha256", "evidence_sha256")
    if any(evidence.get(k) in (None, "") for k in e_required):
        raise ContractError(f"dependency closure evidence lacks hash/count bindings: {extension_id}")
    if evidence["source_builder_version"] != manifest["source_builder_version"] or manifest["source_builder_version"] in {"UNKNOWN", "UNKNOWN_INHERITED"}:
        raise ContractError(f"closure evidence must bind a known source_builder_version: {extension_id}")
    if str(evidence["closure_start"])[:10] != str(manifest["dependency_closure_start"])[:10] or str(evidence["closure_end"])[:10] != str(manifest["dependency_closure_end"])[:10]:
        raise ContractError(f"closure evidence bounds differ from correction manifest: {extension_id}")
    expected_hashes = {
        "recomputed_keys_count": len(recomputed), "recomputed_keys_sha256": canonical_key_hash(recomputed),
        "exact_corrected_keys_count": len(exact), "exact_corrected_keys_sha256": canonical_key_hash(exact),
        "affected_features_sha256": _list_hash(features),
        "affected_source_dependency_sha256": _list_hash(dependencies),
        "recomputed_feature_groups_sha256": _list_hash(groups),
        "affected_feature_groups_sha256": hashlib.sha256(json.dumps(manifest["affected_feature_groups"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }
    if any(evidence.get(k) != v for k, v in expected_hashes.items()):
        raise ContractError(f"correction evidence count/hash binding mismatch: {extension_id}")
    evidence_payload = {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    payload_hash = hashlib.sha256(json.dumps(evidence_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if evidence["evidence_sha256"] != payload_hash:
        raise ContractError(f"correction evidence checksum mismatch: {extension_id}")
    spans_days = (end.date() - start.date()).days > 0
    if spans_days and recomputed == roots and not (evidence.get("no_additional_keys_required") is True and str(evidence.get("no_additional_keys_reason", "")).strip()):
        raise ContractError(f"multi-day closure with root-only recomputation requires verifiable no-additional-keys evidence: {extension_id}")
    if require_upstream:
        if evidence.get("evidence_type") != "UPSTREAM_CANONICAL_REBUILD" or evidence.get("authority") != "UPSTREAM_CANONICAL_BUILDER":
            raise ContractError(f"real correction requires upstream canonical rebuild evidence: {extension_id}")
        upstream_manifest_hash = evidence.get("upstream_rebuild_manifest_sha256")
        if not isinstance(upstream_manifest_hash, str) or len(upstream_manifest_hash) != 64 or any(c not in "0123456789abcdef" for c in upstream_manifest_hash.lower()):
            raise ContractError(f"real correction requires upstream rebuild manifest SHA256: {extension_id}")
        return "UPSTREAM_EVIDENCE_HASH_VALIDATED"
    return "SYNTHETIC_VALIDATION_ONLY"

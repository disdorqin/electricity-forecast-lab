"""Runtime verification for the clean, self-contained delivery bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class BundleIdentityError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_bundle(root: Path) -> dict:
    root = Path(root).resolve()
    path = root / "BUNDLE_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleIdentityError("bundle manifest file list missing")
    rows = []
    for rel, expected in sorted(files.items()):
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            raise BundleIdentityError("manifest path escapes bundle")
        item = (root / p).resolve()
        if not item.is_relative_to(root) or not item.is_file():
            raise BundleIdentityError("manifest file missing/outside bundle: " + rel)
        actual = sha256_file(item)
        if actual != expected:
            raise BundleIdentityError("bundle file SHA mismatch: " + rel)
        rows.append(rel.replace("\\", "/") + "\0" + actual + "\n")
    aggregate = hashlib.sha256("".join(rows).encode("utf-8")).hexdigest()
    if aggregate != manifest.get("aggregate_sha256"):
        raise BundleIdentityError("bundle aggregate SHA mismatch")
    actual_files = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*") if p.is_file() and p.name != "BUNDLE_MANIFEST.json"
        and not any(part in {"__pycache__", ".pytest_cache"} or part.startswith((".tmp", ".pytest")) for part in p.parts)
    }
    if actual_files != set(files):
        raise BundleIdentityError("bundle file inventory differs from manifest")
    freeze_rel = manifest.get("scientific_source_freeze")
    if freeze_rel != "SCIENTIFIC_FREEZE.json":
        raise BundleIdentityError("bundle must embed SCIENTIFIC_FREEZE.json")
    freeze_path = (root / freeze_rel).resolve()
    if not freeze_path.is_relative_to(root) or not freeze_path.is_file():
        raise BundleIdentityError("embedded scientific freeze missing")
    freeze_sha = sha256_file(freeze_path)
    if freeze_sha != manifest.get("scientific_source_freeze_sha256"):
        raise BundleIdentityError("embedded scientific freeze SHA mismatch")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_PRE_RUNTIME_GLUE_BASELINE":
        raise BundleIdentityError("scientific freeze status mismatch")
    protected = freeze.get("protected_scientific_hashes")
    if not isinstance(protected, dict) or not protected:
        raise BundleIdentityError("scientific freeze protected hashes missing")
    package = root / "src/research96/msr_spread_v2"
    for name, expected in sorted(protected.items()):
        item = package / name
        if not item.is_file() or sha256_file(item) != expected:
            raise BundleIdentityError("protected scientific hash mismatch: " + name)
    return {"aggregate_sha256": aggregate, "manifest_sha256": sha256_file(path),
            "scientific_freeze_sha256": freeze_sha, "file_count": len(files)}


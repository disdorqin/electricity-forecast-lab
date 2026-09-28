"""Verified mmap access and selected-manifest quarantine eligibility for real sequence_v13."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import sequence_asset_dir
from .contracts import ContractError, project_root
from .source_resolver import sha256_file
from .target_adapter import source_to_model_target


@dataclass
class SequenceStore:
    asset_dir: Path
    x_hist: np.ndarray
    x_future: np.ndarray
    y_source: np.ndarray
    quarantine_mask: np.ndarray
    day_index: pd.DataFrame
    feature_names: tuple[str, ...]
    registry: dict[str, Any]
    sequence_manifest: dict[str, Any]
    source_gate: dict[str, Any]
    sequence_gate: dict[str, Any]

    @classmethod
    def load(cls, asset_dir: Path | None = None, *, root: Path | None = None) -> "SequenceStore":
        root = Path(root) if root is not None else project_root()
        asset_dir = Path(asset_dir) if asset_dir is not None else sequence_asset_dir(root)
        gate_root = asset_dir.parent
        source_gate_path = gate_root / "source_gate" / "source_audit.json"
        sequence_gate_path = gate_root / "sequence_gate" / "gate.json"
        source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
        sequence_gate = json.loads(sequence_gate_path.read_text(encoding="utf-8"))
        manifest_path = asset_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if source_gate.get("status") != "PASS" or sequence_gate.get("status") != "PASS":
            raise ContractError("dataset requires both SOURCE_GATE_V13 and SEQUENCE_GATE_V13 PASS")
        if sequence_gate.get("source_gate_status") != "PASS":
            raise ContractError("Sequence Gate does not inherit Source Gate PASS")
        if sequence_gate.get("manifest_sha256") != sha256_file(manifest_path):
            raise ContractError("sequence manifest SHA256 differs from Sequence Gate")
        for rel, expected in manifest.get("assets", {}).items():
            path = asset_dir / rel
            if not path.is_file() or sha256_file(path) != expected:
                raise ContractError(f"sequence asset missing/hash mismatch: {rel}")
        x_hist = np.load(asset_dir / "X_hist.npy", mmap_mode="r", allow_pickle=False)
        x_future = np.load(asset_dir / "X_future.npy", mmap_mode="r", allow_pickle=False)
        y_source = np.load(asset_dir / "Y.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(asset_dir / "quarantine_mask.npy", mmap_mode="r", allow_pickle=False)
        days = pd.read_parquet(asset_dir / "day_index.parquet")
        registry = json.loads((asset_dir / "feature_registry.json").read_text(encoding="utf-8"))
        names = tuple(row["feature_name"] for row in registry.get("candidate_features", []))
        expected = (len(days), 168, 7)
        if x_hist.shape != expected or x_future.shape != (len(days), 24, 259) or y_source.shape != (len(days), 24):
            raise ContractError(f"sequence asset shapes violate current contract: {x_hist.shape}, {x_future.shape}, {y_source.shape}")
        if mask.shape != x_future.shape or len(names) != x_future.shape[-1]:
            raise ContractError("quarantine mask/feature registry dimensions do not align with X_future")
        if "sample_index" not in days or "target_day" not in days:
            raise ContractError("day_index requires sample_index and target_day")
        days = days.sort_values("sample_index").reset_index(drop=True)
        if not np.array_equal(days.sample_index.to_numpy(dtype=np.int64), np.arange(len(days))):
            raise ContractError("day_index sample_index must be contiguous and match mmap row order")
        if len(set(names)) != len(names) or manifest.get("sample_count") != len(days):
            raise ContractError("feature inventory/sample count is inconsistent")
        return cls(asset_dir, x_hist, x_future, y_source, mask, days, names, registry, manifest, source_gate, sequence_gate)

    @property
    def days(self) -> np.ndarray:
        return pd.to_datetime(self.day_index.target_day).dt.date.to_numpy()

    @property
    def sequence_sha256(self) -> str:
        return sha256_file(self.asset_dir / "manifest.json")

    @property
    def source_sha256(self) -> str:
        return str(self.sequence_manifest["base_source_sha256"])

    def selector_indices(self, selector_manifest: dict[str, Any]) -> tuple[list[str], np.ndarray, str]:
        if selector_manifest.get("status") != "FROZEN":
            raise ContractError("formal dataset requires a frozen selector manifest")
        selected = selector_manifest.get("selected_features")
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
            raise ContractError("selector manifest needs unique non-empty selected_features")
        index = {name: i for i, name in enumerate(self.feature_names)}
        missing = sorted(set(selected) - set(index))
        if missing:
            raise ContractError(f"selector features absent from current sequence: {missing[:10]}")
        expected_indices = [index[name] for name in selected]
        declared_indices = selector_manifest.get("selected_indices")
        if declared_indices is not None and list(declared_indices) != expected_indices:
            raise ContractError("selector selected_indices do not follow its feature order")
        manifest_hash = selector_manifest.get("selector_sha256")
        return list(selected), np.asarray(expected_indices, dtype=np.int64), str(manifest_hash or "")

    def eligibility(self, selector_manifest: dict[str, Any], *, current_target_day: date | str | None = None,
                    requested_days: Sequence[date | str] | None = None) -> dict[str, Any]:
        names, cols, selector_hash = self.selector_indices(selector_manifest)
        all_days = self.days
        keep = np.ones(len(all_days), dtype=bool)
        if current_target_day is not None:
            cutoff = pd.Timestamp(current_target_day).date() - pd.Timedelta(days=2)
            keep &= np.asarray([d <= cutoff for d in all_days])
        if requested_days is not None:
            wanted = {pd.Timestamp(d).date() for d in requested_days}
            keep &= np.asarray([d in wanted for d in all_days])
        selected_cells = np.asarray(self.quarantine_mask[:, :, cols], dtype=bool)
        contaminated = selected_cells.any(axis=(1, 2))
        eligible = keep & ~contaminated
        return {
            "eligible_indices": np.flatnonzero(eligible).astype(int).tolist(),
            "temporal_or_cutoff_excluded_indices": np.flatnonzero(~keep).astype(int).tolist(),
            "quarantine_excluded_indices": np.flatnonzero(keep & contaminated).astype(int).tolist(),
            "selected_features": names,
            "selected_indices": cols.tolist(),
            "selector_sha256": selector_hash,
            "eligibility_rule": "complete sample; target_day <= current_target_day-2 when training; selected quarantine cells must all be false",
        }

    def candidate_selector_matrix(self, *, cutoff: date | str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten legal candidate values and RT-DA labels for selector construction."""
        cutoff_day = pd.Timestamp(cutoff).date()
        days = self.days
        rows = np.flatnonzero(np.asarray([d <= cutoff_day for d in days]))
        masked = np.asarray(self.quarantine_mask[rows], dtype=bool)
        if masked.any():
            raise ContractError("selector cutoff intersects quarantine cells; fail closed before fitting")
        x = np.asarray(self.x_future[rows], dtype=np.float32).reshape(-1, self.x_future.shape[-1])
        y = source_to_model_target(np.asarray(self.y_source[rows], dtype=np.float32)).reshape(-1)
        day_labels = np.repeat(days[rows], 24)
        hours = np.tile(np.arange(1, 25, dtype=np.int8), len(rows))
        return x, np.asarray(y, dtype=np.float32), day_labels, hours


class SequenceDayDataset(Dataset):
    """Torch Dataset returning real histories, selected futures and model-target labels."""
    def __init__(self, store: SequenceStore, indices: Sequence[int], feature_indices: Sequence[int],
                 *, target_scale: float = 1.0):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.feature_indices = np.asarray(feature_indices, dtype=np.int64)
        self.target_scale = float(target_scale)
        if self.target_scale <= 0:
            raise ValueError("target_scale must be positive")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> dict[str, Any]:
        i = int(self.indices[item])
        x_hist = np.array(self.store.x_hist[i], dtype=np.float32, copy=True)
        x_future = np.array(self.store.x_future[i, :, self.feature_indices], dtype=np.float32, copy=True)
        y_model = -np.array(self.store.y_source[i], dtype=np.float32, copy=True)
        return {
            "x_hist": torch.from_numpy(x_hist),
            "x_future": torch.from_numpy(x_future),
            "y_model": torch.from_numpy(y_model),
            "y_scaled": torch.from_numpy(y_model / self.target_scale),
            "target_day": str(pd.Timestamp(self.store.day_index.iloc[i].target_day).date()),
            "sample_index": i,
        }


def load_frozen_selector(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if manifest.get("status") != "FROZEN":
        raise ContractError("selector manifest is not frozen")
    if manifest.get("selector_sha256") not in (None, digest):
        raise ContractError("selector manifest self-recorded SHA mismatch")
    manifest["selector_sha256"] = digest
    return manifest, digest

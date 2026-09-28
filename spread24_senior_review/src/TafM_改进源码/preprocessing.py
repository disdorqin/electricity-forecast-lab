"""BASE_TRAIN-only imputation, robust scaling, target statistics and official PLE bins."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .contracts import ContractError, TEMPORAL_FEATURES
from .dataset import SequenceStore
from .source_resolver import sha256_file
from .target_adapter import source_to_model_target


@dataclass
class PreprocessorState:
    feature_names: list[str]
    feature_indices: list[int]
    future_median: list[float]
    future_center: list[float]
    future_scale: list[float]
    temporal_median: list[float]
    temporal_center: list[float]
    temporal_scale: list[float]
    ple_bins: list[list[float]]
    ple_n_bins: int
    ple_embedding_dim: int
    target_scale_c: float
    w_pos: float
    w_nonpos: float
    positive_count: int
    nonpositive_count: int
    fit_day_start: str
    fit_day_end: str
    fit_sample_count: int
    fit_sample_indices_sha256: str
    selector_sha256: str
    source_sha256: str
    sequence_manifest_sha256: str
    fit_scope: str = "chronological Stage-A BASE_TRAIN only"
    future_clip_abs: float = 10.0
    temporal_clip_abs: float = 10.0
    clip_after_robust_scale: bool = True
    ple_enabled: bool = True
    future_clip_fraction_fit: float = 0.0
    temporal_clip_fraction_fit: float = 0.0
    future_preclip_abs_p99: float = 0.0
    future_preclip_abs_p999: float = 0.0
    future_preclip_abs_max: float = 0.0
    future_postclip_abs_max: float = 0.0
    temporal_preclip_abs_p99: float = 0.0
    temporal_preclip_abs_p999: float = 0.0
    temporal_preclip_abs_max: float = 0.0
    temporal_postclip_abs_max: float = 0.0
    future_feature_audit: list[dict[str, Any]] | None = None
    temporal_channel_audit: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return sha256_file(path)

    @classmethod
    def load(cls, path: Path) -> "PreprocessorState":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _median_and_robust_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        median = np.nanmedian(x, axis=0)
        q25, q75 = np.nanquantile(x, [0.25, 0.75], axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    center = median.copy()
    scale = (q75 - q25) / 1.349
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return median.astype(np.float32), center.astype(np.float32), scale.astype(np.float32)


def _official_quantile_bins(x: np.ndarray, n_bins: int) -> list[list[float]]:
    import rtdl_num_embeddings as rtdl
    values = torch.as_tensor(np.asarray(x, dtype=np.float32), device="cpu")
    if values.ndim != 2 or values.shape[0] < 2:
        raise ContractError("PLE bin fitting needs BASE_TRAIN rows by feature")
    edges: list[list[float] | None] = [None] * values.shape[1]
    active = []
    for j in range(values.shape[1]):
        unique = torch.unique(values[:, j])
        if len(unique) < 2:
            # Official PLE's contract requires >=2 edges per feature even for
            # constant BASE_TRAIN columns. Keep a tiny finite interval; the
            # robust-scaled input itself remains exactly constant at zero.
            value = float(unique[0])
            edges[j] = [value - 5e-7, value + 5e-7]
        else:
            active.append(j)
    if active:
        fitted = rtdl.compute_bins(values[:, active], n_bins=min(n_bins, len(values) - 1))
        for j, bins in zip(active, fitted):
            edges[j] = [float(v) for v in bins.tolist()]
    if any(x is None or not x for x in edges):
        raise ContractError("official compute_bins failed to produce every feature edge")
    return edges  # type: ignore[return-value]


def _numerical_summary(values: np.ndarray, names: Sequence[str], clip_abs: float) -> tuple[dict[str, float], list[dict[str, Any]]]:
    x = np.asarray(values, dtype=np.float64)
    absolute = np.abs(x)
    finite = np.isfinite(absolute)
    if not finite.all():
        raise ContractError("scaled preprocessing fit values must be finite before clipping")
    clipped = absolute > clip_abs
    summary = {
        "preclip_abs_p99": float(np.quantile(absolute, 0.99)),
        "preclip_abs_p999": float(np.quantile(absolute, 0.999)),
        "preclip_abs_max": float(absolute.max(initial=0.0)),
        "postclip_abs_max": float(np.clip(absolute, 0.0, clip_abs).max(initial=0.0)),
        "clip_count": int(clipped.sum()),
        "clip_fraction": float(clipped.mean()) if clipped.size else 0.0,
    }
    per_feature = []
    for j, name in enumerate(names):
        col = absolute[:, j]
        count = int(np.count_nonzero(col > clip_abs))
        per_feature.append({"feature_name": str(name), "preclip_max_abs": float(col.max(initial=0.0)),
            "postclip_max_abs": float(np.clip(col, 0.0, clip_abs).max(initial=0.0)), "clip_count": count,
            "clip_fraction": float(count / len(col)) if len(col) else 0.0})
    per_feature.sort(key=lambda item: item["preclip_max_abs"], reverse=True)
    return summary, per_feature[:20]


def fit_preprocessor(store: SequenceStore, base_indices: Sequence[int], selector_manifest: dict[str, Any],
                     *, selector_sha256: str, n_bins: int = 16, ple_embedding_dim: int = 8,
                     future_clip_abs: float = 10.0, temporal_clip_abs: float = 10.0,
                     clip_after_robust_scale: bool = True, ple_enabled: bool = True) -> PreprocessorState:
    if future_clip_abs <= 0 or temporal_clip_abs <= 0:
        raise ContractError("preprocessing clipping bounds must be positive")
    if selector_manifest.get("status") != "FROZEN":
        raise ContractError("fit_preprocessor requires a frozen selector manifest")
    indices = np.asarray(base_indices, dtype=np.int64)
    if len(indices) < 2 or len(set(indices.tolist())) != len(indices):
        raise ContractError("BASE_TRAIN indices must be unique with at least two samples")
    names, feature_indices, _ = store.selector_indices(selector_manifest)
    eligibility = store.eligibility(selector_manifest, requested_days=store.day_index.iloc[indices].target_day.tolist())
    eligible_set = set(eligibility["eligible_indices"])
    if any(int(i) not in eligible_set for i in indices):
        raise ContractError("BASE_TRAIN contains an incomplete/quarantined selected-feature day")
    future = np.asarray(store.x_future[indices][:, :, feature_indices], dtype=np.float32).reshape(-1, len(names))
    future_median, future_center, future_scale = _median_and_robust_scale(future)
    future_imputed = np.where(np.isfinite(future), future, future_median[None, :])
    future_scaled = (future_imputed.astype(np.float64) - future_center[None, :]) / future_scale[None, :]
    future_summary, future_audit = _numerical_summary(future_scaled, names, future_clip_abs)
    future_index_by_name = {name: i for i, name in enumerate(names)}
    for item in future_audit:
        j = future_index_by_name[item["feature_name"]]
        item["robust_scale"] = float(future_scale[j])
        item["median"] = float(future_median[j])
    future_model = np.clip(future_scaled, -future_clip_abs, future_clip_abs) if clip_after_robust_scale else future_scaled
    ple_bins = _official_quantile_bins(future_model, n_bins) if ple_enabled else []

    hist = np.asarray(store.x_hist[indices], dtype=np.float32).reshape(-1, len(TEMPORAL_FEATURES))
    hist_median, hist_center, hist_scale = _median_and_robust_scale(hist)
    hist_imputed = np.where(np.isfinite(hist), hist, hist_median[None, :])
    hist_scaled = (hist_imputed.astype(np.float64) - hist_center[None, :]) / hist_scale[None, :]
    temporal_summary, temporal_audit = _numerical_summary(hist_scaled, TEMPORAL_FEATURES, temporal_clip_abs)
    temporal_index_by_name = {name: i for i, name in enumerate(TEMPORAL_FEATURES)}
    for item in temporal_audit:
        j = temporal_index_by_name[item["feature_name"]]
        item["robust_scale"] = float(hist_scale[j])
        item["median"] = float(hist_median[j])
    y_source = np.asarray(store.y_source[indices], dtype=np.float64)
    y_model = source_to_model_target(y_source).reshape(-1)
    if not np.isfinite(y_model).all():
        raise ContractError("BASE_TRAIN target must be finite")
    q25, q75 = np.quantile(y_model, [0.25, 0.75])
    target_c = max(float((q75 - q25) / 1.349), 1e-8)
    y_positive = y_model > 0
    n_pos, n_nonpos = int(y_positive.sum()), int((~y_positive).sum())
    # Direction class weights are not part of canonical V2.1 preprocessing.
    days = store.days[indices]
    index_digest = __import__("hashlib").sha256(np.sort(indices).astype("<i8").tobytes()).hexdigest()
    return PreprocessorState(
        names, feature_indices.tolist(), future_median.tolist(), future_center.tolist(), future_scale.tolist(),
        hist_median.tolist(), hist_center.tolist(), hist_scale.tolist(), ple_bins, int(n_bins),
        int(ple_embedding_dim), target_c, 1.0, 1.0,
        n_pos, n_nonpos, min(days).isoformat(), max(days).isoformat(), len(indices), index_digest,
        selector_sha256, store.source_sha256, store.sequence_sha256,
        "chronological Stage-A BASE_TRAIN only", float(future_clip_abs), float(temporal_clip_abs),
        bool(clip_after_robust_scale), bool(ple_enabled), future_summary["clip_fraction"],
        temporal_summary["clip_fraction"], future_summary["preclip_abs_p99"], future_summary["preclip_abs_p999"],
        future_summary["preclip_abs_max"], future_summary["postclip_abs_max"], temporal_summary["preclip_abs_p99"],
        temporal_summary["preclip_abs_p999"], temporal_summary["preclip_abs_max"], temporal_summary["postclip_abs_max"],
        future_audit, temporal_audit,
    )


def transform_future(x: np.ndarray, state: PreprocessorState) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    median, center, scale = map(lambda z: np.asarray(z, dtype=np.float64),
                                (state.future_median, state.future_center, state.future_scale))
    if x.shape[-1] != len(state.feature_names):
        raise ValueError("future input feature dimension differs from frozen preprocessor order")
    normalized = (np.where(np.isfinite(x), x, median) - center) / scale
    if state.clip_after_robust_scale:
        normalized = np.clip(normalized, -state.future_clip_abs, state.future_clip_abs)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("future preprocessing produced NaN/inf")
    return normalized.astype(np.float32)


def transform_hist(x: np.ndarray, state: PreprocessorState) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    median, center, scale = map(lambda z: np.asarray(z, dtype=np.float64),
                                (state.temporal_median, state.temporal_center, state.temporal_scale))
    if x.shape[-1] != len(TEMPORAL_FEATURES):
        raise ValueError("history input must contain canonical 7-channel whitelist")
    normalized = (np.where(np.isfinite(x), x, median) - center) / scale
    if state.clip_after_robust_scale:
        normalized = np.clip(normalized, -state.temporal_clip_abs, state.temporal_clip_abs)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("temporal preprocessing produced NaN/inf")
    return normalized.astype(np.float32)

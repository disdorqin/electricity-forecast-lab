"""V2 end-to-end configuration; defaults are pinned to the active canonical docs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
from pathlib import Path
from typing import Any

import yaml

from .contracts import project_root


@dataclass(frozen=True)
class V2Config:
    seed: int = 20260924
    selector_cutoff: str = "2025-12-31"
    selector_q: float = 0.90
    strong_hit_rate: float = 0.75
    weak_hit_rate: float = 0.50
    shadow_policy: str = "within-hour-across-day"
    future_clip_abs: float = 10.0
    temporal_clip_abs: float = 10.0
    clip_after_robust_scale: bool = True
    ple_enabled: bool = True
    ple_bins: int = 16
    ple_embedding_dim: int = 8
    arch_type: str = "tabm-mini"
    k: int = 8
    d_tab: int = 128
    n_blocks: int = 2
    dropout: float = 0.05
    d_time: int = 32
    time_hidden: int = 64
    fft_enabled: bool = True
    fft_bins: tuple[int, ...] = (1, 7, 14, 21, 28)
    future_conditioning: bool = True
    d_task: int = 32
    mode: str = "A2"
    alpha_init: float = 0.8
    batch_size: int = 64
    checkpoint_raw_tolerance: float = 0.02
    stage_b_production_default: bool = False
    smooth_l1_beta: float = 1.0
    max_epochs: int = 120
    patience: int = 15
    stage_b_epochs: int = 8
    lr: float = 0.002
    weight_decay: float = 0.0003
    lambda_dir: float = 1.0
    lambda_mag: float = 1.0
    stage_b_recent_quota: float = 0.80
    stage_b_lr_ratio: float = 0.10
    magnitude_eps: float = 1e-8
    device: str = "auto"
    amp: bool = True
    config_path: str | None = None
    config_sha256: str | None = None

    @property
    def n_bins(self) -> int:
        """Compatibility accessor; YAML and provenance use the canonical ple_bins name."""
        return self.ple_bins

    def with_profile(self, profile: str, *, k_override: int | None = None) -> "V2Config":
        if profile == "default":
            return self
        if profile != "smoke":
            raise ValueError(f"unknown profile: {profile}")
        from dataclasses import replace
        return replace(self, max_epochs=2, patience=2, stage_b_epochs=1,
                       k=4 if k_override is None else k_override, batch_size=32)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fft_bins"] = list(self.fft_bins)
        return result

    def resolved_config(self) -> dict[str, Any]:
        return {k: v for k, v in self.to_dict().items() if k not in {"config_path", "config_sha256"}}


def _validate_config(config: V2Config) -> None:
    if config.shadow_policy != "within-hour-across-day":
        raise ValueError("shadow_policy is frozen to within-hour-across-day")
    if config.selector_cutoff != "2025-12-31":
        raise ValueError("selector_cutoff is frozen to 2025-12-31")
    for name in ("selector_q", "strong_hit_rate", "weak_hit_rate", "alpha_init", "stage_b_recent_quota", "stage_b_lr_ratio"):
        value = getattr(config, name)
        if not 0.0 <= value <= 1.0 or (name in {"selector_q", "alpha_init"} and value in {0.0, 1.0}):
            raise ValueError(f"{name} out of range")
    if config.future_clip_abs <= 0 or config.temporal_clip_abs <= 0:
        raise ValueError("clip bounds must be positive")
    if config.future_clip_abs != 10.0 or config.temporal_clip_abs != 10.0 or not config.clip_after_robust_scale:
        raise ValueError("canonical preprocessing requires post-scale clipping at abs=10.0")
    if config.checkpoint_raw_tolerance < 0 or config.checkpoint_raw_tolerance > 1:
        raise ValueError("checkpoint_raw_tolerance must be in [0,1]")
    if config.checkpoint_raw_tolerance != 0.02:
        raise ValueError("checkpoint_raw_tolerance is frozen to 0.02")
    if config.stage_b_production_default:
        raise ValueError("Stage B remains explicit-only; production default must be false")
    if config.ple_bins < 2 or config.ple_embedding_dim < 1 or config.k < 1 or config.d_tab < 1 or config.n_blocks < 1:
        raise ValueError("PLE/TabM dimensions must be positive and bins >= 2")
    if config.d_time < 1 or config.time_hidden < 1 or config.d_task < 1 or config.batch_size < 1:
        raise ValueError("encoder dimensions and batch_size must be positive")
    if config.max_epochs < 1 or config.patience < 1 or config.stage_b_epochs < 1:
        raise ValueError("epoch counts and patience must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be in [0,1)")
    if any(x <= 0 or x >= 84 for x in config.fft_bins) or not config.fft_bins:
        raise ValueError("fft_bins must be nonempty rFFT bins in (0,84)")
    if config.mode not in {"A0", "A1", "A2"}:
        raise ValueError("mode must be A0/A1/A2")
    if config.alpha_init != 0.8:
        raise ValueError("canonical A1/A2 alpha_init is frozen to 0.8")
    if config.arch_type != "tabm-mini":
        raise ValueError("first implementation only accepts official tabm-mini")
    if config.device != "auto" and config.device != "cpu" and not config.device.startswith("cuda"):
        raise ValueError("device must be auto, cpu, or a CUDA device")
    for name in ("lr", "lambda_dir", "lambda_mag", "magnitude_eps"):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")


def default_v21_config_path() -> Path:
    return project_root() / "src" / "config_tabm_v21.yaml"


def load_v21_config(path: str | Path | None = None) -> V2Config:
    """Load the strict executable V2.1 YAML source of truth and attach its hash."""
    config_path = Path(path) if path is not None else default_v21_config_path()
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V2.1 YAML must contain a mapping")
    valid = {f.name for f in fields(V2Config)} - {"config_path", "config_sha256"}
    unknown = set(raw) - valid
    if unknown:
        raise ValueError(f"unknown V2.1 config keys: {sorted(unknown)}")
    hints = {
        "seed": int, "selector_cutoff": str, "selector_q": float, "strong_hit_rate": float,
        "weak_hit_rate": float, "shadow_policy": str, "future_clip_abs": float,
        "temporal_clip_abs": float, "clip_after_robust_scale": bool, "ple_enabled": bool,
        "ple_bins": int, "ple_embedding_dim": int, "arch_type": str, "k": int,
        "d_tab": int, "n_blocks": int, "dropout": float, "d_time": int, "time_hidden": int,
        "fft_enabled": bool, "fft_bins": tuple, "future_conditioning": bool, "d_task": int,
        "mode": str, "alpha_init": float, "batch_size": int, "max_epochs": int, "patience": int,
        "checkpoint_raw_tolerance": float, "stage_b_epochs": int, "stage_b_recent_quota": float,
        "stage_b_lr_ratio": float, "stage_b_production_default": bool, "lr": float,
        "weight_decay": float, "lambda_dir": float, "lambda_mag": float, "smooth_l1_beta": float,
        "magnitude_eps": float, "device": str, "amp": bool,
    }
    for key, value in raw.items():
        expected = hints[key]
        if expected is float:
            valid_type = type(value) in {int, float}
        elif expected is tuple:
            valid_type = isinstance(value, list) and all(type(v) is int for v in value)
        else:
            valid_type = type(value) is expected
        if not valid_type:
            raise ValueError(f"invalid type for config field {key}: expected {expected.__name__}")
    if "fft_bins" in raw:
        raw["fft_bins"] = tuple(raw["fft_bins"])
    raw["config_path"] = str(config_path)
    raw["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config = V2Config(**raw)
    _validate_config(config)
    return config


def config_provenance(config: V2Config) -> dict[str, Any]:
    if not config.config_path or not config.config_sha256:
        return {"config_path": None, "config_sha256": None, "resolved_config": config.resolved_config()}
    return {"config_path": config.config_path, "config_sha256": config.config_sha256,
            "resolved_config": config.resolved_config()}


def sequence_asset_dir(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else project_root()
    # Frozen PASS artifacts from the completed V1.3 Gate; never rebuild or mutate here.
    return root / "src" / "TafM_改进源码" / "round3_outputs" / "sequence_v13"


def formal_output_dir(root: Path | None = None) -> Path:
    root = Path(root) if root is not None else project_root()
    return root / "src" / "TafM_改进源码" / "outputs" / "tabm_v21"


def default_selector_path(root: Path | None = None) -> Path:
    return formal_output_dir(root) / "selector" / "selector_cutoff_2025-12-31" / "manifest.json"


def resolve_device(config: V2Config) -> str:
    import torch
    if config.device != "auto":
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {config.device}, CUDA unavailable")
        return config.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def package_versions() -> dict[str, str | None]:
    import importlib.metadata as metadata
    import platform
    names = ("torch", "tabm", "rtdl_num_embeddings", "shap", "xgboost", "numpy", "pandas", "pyarrow")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import torch
        versions["cuda"] = torch.version.cuda
        versions["cuda_available"] = str(torch.cuda.is_available())
    except ImportError:
        versions["cuda"] = None
        versions["cuda_available"] = "False"
    return versions

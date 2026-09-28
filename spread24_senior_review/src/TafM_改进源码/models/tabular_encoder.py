"""Official tabm mini ensemble encoder with PLE, explicit member axis and Weak branch."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class TabularEncoder(nn.Module):
    def __init__(self, *, feature_names: list[str], feature_roles: list[dict[str, Any]],
                 ple_bins: list[list[float]], d_tab: int = 128, k: int = 8, n_blocks: int = 2,
                 dropout: float = 0.05, ple_embedding_dim: int = 8, arch_type: str = "tabm-mini",
                 ple_enabled: bool = True, strong_role_profile: str = "all",
                 numeric_encoding_mode: str = "canonical"):
        super().__init__()
        if arch_type != "tabm-mini":
            raise ValueError("the validated official encoder profile is tabm-mini")
        if numeric_encoding_mode not in {"canonical", "raw_only", "ple_only"}:
            raise ValueError("numeric_encoding_mode must be canonical/raw_only/ple_only")
        # E2-D1 experiment-only numeric-encoding family. "canonical" reproduces the frozen
        # PLE+raw behaviour exactly; raw_only/ple_only never alter preprocessing or bins.
        use_ple = bool(ple_enabled)
        use_raw = True
        if numeric_encoding_mode == "raw_only":
            use_ple, use_raw = False, True
        elif numeric_encoding_mode == "ple_only":
            if not ple_enabled:
                raise ValueError("numeric_encoding_mode=ple_only requires ple_enabled=True")
            use_ple, use_raw = True, False
        if not feature_names or (use_ple and len(feature_names) != len(ple_bins)):
            raise ValueError("PLE bins must match the non-empty frozen selected feature order")
        import tabm
        import rtdl_num_embeddings as rtdl

        self.feature_names = tuple(feature_names)
        self.d_tab = int(d_tab)
        self.k = int(k)
        self.arch_type = arch_type
        role_by_name = {r["feature_name"]: r["role"] for r in feature_roles}
        if set(feature_names) - set(role_by_name):
            raise ValueError("selector roles are incomplete")
        if strong_role_profile not in {"all", "drop_mag", "drop_dir"}:
            raise ValueError("strong_role_profile must be all/drop_mag/drop_dir")
        self.strong_role_profile = strong_role_profile
        allowed = {r for r in role_by_name.values() if r != "Weak"}
        if strong_role_profile != "all":
            allowed = {"Strong-BOTH", "Strong-DIR", "Strong-MAG", "Forced-Core", "Strong"}
        if strong_role_profile == "drop_mag": allowed.discard("Strong-MAG")
        if strong_role_profile == "drop_dir": allowed.discard("Strong-DIR")
        self.strong_indices = tuple(i for i, name in enumerate(feature_names) if role_by_name[name] in allowed)
        self.weak_indices = tuple(i for i, name in enumerate(feature_names) if role_by_name[name] == "Weak")
        if not self.strong_indices:
            raise ValueError("Strong/Core TabM branch cannot be empty")
        self.ple_enabled = bool(use_ple)
        self.numeric_encoding_mode = numeric_encoding_mode
        self.use_raw = bool(use_raw)
        if self.ple_enabled:
            bins = [torch.as_tensor(b, dtype=torch.float32) for b in ple_bins]
            self.ple = rtdl.PiecewiseLinearEmbeddings(bins, d_embedding=ple_embedding_dim, activation=True, version="A")
            self.encoded_feature_dim = ple_embedding_dim + (1 if self.use_raw else 0)
            self._ple_embedding_dim = int(ple_embedding_dim)
        else:
            self.ple = None
            self.encoded_feature_dim = 1
            self._ple_embedding_dim = 0
        tabm_input_dim = len(self.strong_indices) * self.encoded_feature_dim
        self.ensemble_view = tabm.EnsembleView(k=self.k)
        # Official package API: tabm-mini uses member-wise input affine scaling
        # followed by shared dense blocks. No local TabM implementation exists.
        self.backbone = tabm.make_tabm_backbone(
            d_in=tabm_input_dim, n_blocks=n_blocks, d_block=d_tab, dropout=dropout,
            activation="GELU", k=self.k, arch_type="tabm-mini",
            start_scaling_init="random-signs", start_scaling_init_chunks=[tabm_input_dim],
        )
        if not isinstance(getattr(self.backbone, "affine", None), nn.Module):
            raise RuntimeError("installed official tabm-mini exposes no auditable affine member module")
        if not hasattr(tabm, "LinearEnsemble"):
            raise RuntimeError("installed official tabm lacks LinearEnsemble API")
        if self.weak_indices:
            weak_dim = len(self.weak_indices) * self.encoded_feature_dim
            self.weak_mlp = nn.Sequential(nn.Linear(weak_dim, d_tab), nn.GELU(), nn.Linear(d_tab, d_tab))
        else:
            self.weak_mlp = None
        self.horizon_gate_logits = nn.Parameter(torch.full((24,), _logit(0.8)))
        self.api_identity = {
            "package": "tabm",
            "version": __import__("importlib.metadata", fromlist=["version"]).version("tabm"),
            "view": "tabm.EnsembleView",
            "backbone_factory": "tabm.make_tabm_backbone",
            "arch_type": arch_type,
            "member_head": "tabm.LinearEnsemble",
            "ple": "rtdl_num_embeddings.PiecewiseLinearEmbeddings(version=A)",
            "member_specific_owner": "official backbone.affine module object",
        }

    def _encoded(self, x_future: torch.Tensor) -> torch.Tensor:
        if x_future.ndim != 3 or x_future.shape[1] != 24 or x_future.shape[2] != len(self.feature_names):
            raise ValueError(f"x_future must be [B,24,{len(self.feature_names)}]")
        if not torch.isfinite(x_future).all():
            raise FloatingPointError("preprocessed future tensor contains NaN/inf")
        b = x_future.shape[0]
        flat = x_future.reshape(b * 24, x_future.shape[-1])
        if not self.ple_enabled:
            return flat.unsqueeze(-1)
        assert self.ple is not None
        ple = self.ple(flat)
        if ple.ndim != 3 or ple.shape[:2] != flat.shape:
            raise RuntimeError(f"official PLE returned unsupported shape {tuple(ple.shape)}")
        if not self.use_raw:
            return ple
        return torch.cat([ple, flat.unsqueeze(-1)], dim=-1)

    def encoding_audit(self) -> dict[str, Any]:
        """Read-only E2-D1 audit of the numeric representation actually used."""
        return {
            "numeric_encoding_mode": self.numeric_encoding_mode,
            "ple_enabled": bool(self.ple_enabled),
            "raw_skip_enabled": bool(self.use_raw),
            "encoded_feature_dim": int(self.encoded_feature_dim),
            "ple_embedding_dim": int(self._ple_embedding_dim),
            "strong_feature_count": len(self.strong_indices),
            "weak_feature_count": len(self.weak_indices),
            "strong_tabm_input_dim": len(self.strong_indices) * int(self.encoded_feature_dim),
            "weak_mlp_input_dim": len(self.weak_indices) * int(self.encoded_feature_dim),
        }

    def forward_components(self, x_future: torch.Tensor):
        """Return the untouched Strong, Weak and gated representations for experiments.

        The component computation is the canonical forward graph: no branch is recomputed,
        detached, or suppressed here. ``forward`` remains the canonical H_current interface.
        """
        encoded = self._encoded(x_future)
        b = x_future.shape[0]
        strong = encoded[:, self.strong_indices, :].reshape(b * 24, -1)
        member_input = self.ensemble_view(strong)
        h_strong_flat = self.backbone(member_input)
        if h_strong_flat.shape != (b * 24, self.k, self.d_tab):
            raise RuntimeError(f"official TabM member output shape mismatch: {tuple(h_strong_flat.shape)}")
        h_strong = h_strong_flat.reshape(b, 24, self.k, self.d_tab).permute(0, 2, 1, 3)
        if self.weak_indices:
            weak = encoded[:, self.weak_indices, :].reshape(b * 24, -1)
            h_weak = self.weak_mlp(weak).reshape(b, 24, self.d_tab)
        else:
            h_weak = h_strong.new_zeros((b, 24, self.d_tab))
        gate_values = torch.sigmoid(self.horizon_gate_logits)
        gate = gate_values.view(1, 1, 24, 1)
        h_tab = gate * h_strong + (1.0 - gate) * h_weak.unsqueeze(1)
        if not torch.isfinite(h_tab).all():
            raise FloatingPointError("non-finite official TabM tabular representation")
        return h_strong, h_weak, gate_values, h_tab

    def forward(self, x_future: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x_future)[-1]

    def member_specific_parameter_ids(self) -> set[int]:
        """Use official module ownership, not parameter-name substring guesses."""
        return {id(p) for p in self.backbone.affine.parameters()}

    def shared_backbone_parameter_ids(self) -> set[int]:
        member = self.member_specific_parameter_ids()
        return {id(p) for p in self.backbone.parameters() if id(p) not in member}

    def parameter_topology(self) -> dict[str, Any]:
        member_ids = self.member_specific_parameter_ids()
        shared_ids = self.shared_backbone_parameter_ids()
        return {
            "official_api": self.api_identity,
            "official_member_module_type": f"{type(self.backbone.affine).__module__}.{type(self.backbone.affine).__qualname__}",
            "member_specific_param_count": sum(p.numel() for p in self.backbone.parameters() if id(p) in member_ids),
            "shared_backbone_param_count": sum(p.numel() for p in self.backbone.parameters() if id(p) in shared_ids),
            "topology_audit": "PASS" if member_ids and shared_ids and member_ids.isdisjoint(shared_ids) else "UNAVAILABLE",
        }

    def horizon_gate_values(self) -> list[float]:
        return torch.sigmoid(self.horizon_gate_logits.detach()).cpu().tolist()

"""V2.1 engineering gate: official-package API and member-wise forward/backward smoke."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .config import formal_output_dir, package_versions
from .contracts import ContractError, TEMPORAL_FEATURES
from .dataset import SequenceStore
from .losses import v21_loss
from .models.dual_branch_v21 import DualBranchV21
from .models.tabular_encoder import TabularEncoder
from .models.temporal_encoder import TemporalEncoder


def run_v21_gate(output_path: Path | None = None) -> dict:
    """Run the V2.1 non-training engineering gate.

    This gate deliberately uses a tiny legal feature subset and multi-edge synthetic
    PLE bins. It validates the installed official APIs and member axis without
    creating hundreds of irrelevant single-bin warnings.
    """
    store = SequenceStore.load()

    import tabm
    import rtdl_num_embeddings as rtdl

    versions = package_versions()
    api = {
        "EnsembleView": str(inspect.signature(tabm.EnsembleView)),
        "make_tabm_backbone": str(inspect.signature(tabm.make_tabm_backbone)),
        "LinearEnsemble": str(inspect.signature(tabm.LinearEnsemble)),
        "PiecewiseLinearEmbeddings": str(inspect.signature(rtdl.PiecewiseLinearEmbeddings)),
        "compute_bins": str(inspect.signature(rtdl.compute_bins)),
        "tabm_version": versions["tabm"],
        "rtdl_num_embeddings_version": versions["rtdl_num_embeddings"],
    }

    # Use exactly the six legal target-day forecast counterparts required by the
    # temporal encoder. These are source features, not labels/actuals.
    feature_names = list(TEMPORAL_FEATURES[1:])
    missing = [name for name in feature_names if name not in set(store.feature_names)]
    if missing:
        raise ContractError(f"V2.1 gate missing legal future-conditioning features: {missing}")

    roles = [{"feature_name": name, "role": "Strong-GATE"} for name in feature_names]
    ple_bins = [[-1.0, 0.0, 1.0] for _ in feature_names]
    enc = TabularEncoder(
        feature_names=feature_names,
        feature_roles=roles,
        ple_bins=ple_bins,
        d_tab=16,
        k=2,
        n_blocks=2,
        dropout=0.0,
        ple_embedding_dim=2,
    )

    try:
        with torch.no_grad():
            h = enc(torch.zeros((2, 24, len(feature_names)), dtype=torch.float32))
    except Exception as exc:
        raise ContractError(f"official TabM API smoke failed: {exc}") from exc

    if tuple(h.shape) != (2, 2, 24, 16):
        raise ContractError(f"official TabM member axis mismatch: {tuple(h.shape)}")

    model = DualBranchV21(
        16,
        8,
        8,
        mode="A2",
        tabular_encoder=enc,
        temporal_encoder=TemporalEncoder(
            selected_feature_names=feature_names,
            d_time=8,
            hidden=16,
        ),
        k=2,
        magnitude_scale=1.0,
    )
    xh = torch.zeros((2, 168, 7), dtype=torch.float32)
    xf = torch.zeros((2, 24, len(feature_names)), dtype=torch.float32)
    out = model(xh, xf)
    loss = v21_loss(out, torch.zeros((2, 24), dtype=torch.float32))
    loss["L_total"].backward()

    v21_shapes = {
        "z_members": list(out["z_members"].shape),
        "a_scaled_members": list(out["a_scaled_members"].shape),
        "p": list(out["p"].shape),
        "finite": all(
            torch.isfinite(value).all().item()
            for value in out.values()
            if isinstance(value, torch.Tensor)
        ),
        "backward": all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ),
    }
    if (
        v21_shapes["z_members"] != [2, 2, 24]
        or v21_shapes["a_scaled_members"] != [2, 2, 24]
        or not v21_shapes["finite"]
        or not v21_shapes["backward"]
    ):
        raise ContractError(f"V2.1 member-wise forward/backward failed: {v21_shapes}")

    report = {
        "status": "PASS",
        "source_gate": store.source_gate["status"],
        "sequence_gate": store.sequence_gate["status"],
        "samples": len(store.days),
        "x_hist_shape": list(store.x_hist.shape),
        "x_future_shape": list(store.x_future.shape),
        "quarantine_mask_shape": list(store.quarantine_mask.shape),
        "official_tabm_api": api,
        "gate_feature_names": feature_names,
        "gate_ple_bins": ple_bins,
        "official_tabm_member_shape": list(h.shape),
        "v2_1_memberwise_forward_backward": v21_shapes,
        "package_versions": versions,
        "frozen_sequence_manifest_sha256": store.sequence_sha256,
        "source_sha256": store.source_sha256,
    }

    path = Path(output_path) if output_path is not None else formal_output_dir() / "gates" / "phase_a_v21_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report

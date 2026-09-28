"""Real-sequence V2 Stage A/B trainer with checkpoint reload and provenance."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .config import V2Config, config_provenance, default_selector_path, formal_output_dir, package_versions, resolve_device
from .contracts import ContractError, TEMPORAL_FEATURES
from .dataset import SequenceStore, load_frozen_selector
from .losses import v21_loss, v2_loss_compat
from .direction_postprocess import POSTPROCESS_MODES, fit_logistic_stacker
from .direction_experiments import (OBJECTIVE_MODES, CHECKPOINT_POLICIES, GRADIENT_POLICIES,
    audit_protected_parameter_groups, checkpoint_diagnostic_warnings, clip_protected_gradient_groups,
    direction_first_checkpoint_is_better, experiment_output_category, gradient_diagnostic_rows,
    magnitude_first_checkpoint_is_better, objective_tensor, protected_gradients, summarize_gradient_rows,
    summarize_projection_rows, update_direction_first_progress, update_magnitude_first_progress)
from .metrics import canonical_metrics
from .preprocessing import PreprocessorState, fit_preprocessor, transform_future, transform_hist
from .source_resolver import sha256_file
from .target_adapter import source_to_model_target
from .models.dual_branch_v2 import DualReadoutV2
from .models.dual_branch_v21 import DualBranchV21
from .models.task_adapters import ARCHITECTURE_MODES
from .models.tabular_encoder import TabularEncoder
from .models.temporal_encoder import TemporalEncoder


def _checkpoint_tiebreak_key(metrics: dict[str, float | int]) -> tuple[float, float, int]:
    return (-float(metrics["magnitude_mae"]), -float(metrics["L_total"]), -int(metrics["epoch"]))


def checkpoint_is_better(candidate: dict[str, float | int], best: dict[str, float | int] | None,
                          *, raw_anchor: float | None = None, tolerance: float = 0.02) -> bool:
    """Compare eligible checkpoints inside a frozen Raw guardrail; no composite score."""
    anchor = max(float(candidate["raw_direction_accuracy"]),
                 float(best["raw_direction_accuracy"]) if best is not None else float("-inf"),
                 float(raw_anchor) if raw_anchor is not None else float("-inf"))
    if float(candidate["raw_direction_accuracy"]) < anchor - tolerance:
        return False
    if best is None or float(best["raw_direction_accuracy"]) < anchor - tolerance:
        return True
    return _checkpoint_tiebreak_key(candidate) > _checkpoint_tiebreak_key(best)


def update_checkpoint_state(candidate: dict[str, float | int], raw_anchor: float | None,
                            selected: dict[str, float | int] | None, *, tolerance: float = 0.02) -> dict[str, Any]:
    """Advance raw anchor then select only guardrail-eligible monitor checkpoints."""
    raw = float(candidate["raw_direction_accuracy"])
    anchor_improved = raw_anchor is None or raw > raw_anchor
    new_anchor = raw if raw_anchor is None else max(float(raw_anchor), raw)
    invalidated = selected is not None and float(selected["raw_direction_accuracy"]) < new_anchor - tolerance
    eligible_selected = None if invalidated else selected
    candidate_eligible = raw >= new_anchor - tolerance
    selected_improved = candidate_eligible and checkpoint_is_better(
        candidate, eligible_selected, raw_anchor=new_anchor, tolerance=tolerance)
    new_selected = candidate if selected_improved else eligible_selected
    return {"raw_anchor": new_anchor, "selected": new_selected,
            "raw_anchor_improved": anchor_improved, "selected_improved": selected_improved,
            "selected_invalidated": invalidated,
            "progress": anchor_improved or selected_improved}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def direction_class_weights(state: PreprocessorState, mode: str) -> tuple[float, float]:
    """Experiment-only Direction BCE weights derived from BASE_TRAIN labels only."""
    if mode not in {"unweighted", "sqrt_balanced", "full_balanced"}:
        raise ValueError("direction_class_weight_mode must be unweighted/sqrt_balanced/full_balanced")
    if mode == "unweighted":
        return 1.0, 1.0
    n_pos = float(state.positive_count)
    n_neg = float(state.nonpositive_count)
    total = n_pos + n_neg
    if n_pos <= 0 or n_neg <= 0 or total <= 0:
        raise ContractError("Direction class weighting requires both classes in BASE_TRAIN")
    p = n_pos / total
    if mode == "full_balanced":
        return total / (2.0 * n_pos), total / (2.0 * n_neg)
    ratio = math.sqrt(n_neg / n_pos)
    w_neg = 1.0 / (p * ratio + (1.0 - p))
    return ratio * w_neg, w_neg


# E4-B pre-registered literature/regime recovery set (EXACTLY 18), from plan 29 §3.
# All 18 are frozen-selector role "Noise" recovered as Weak only in the experiment manifest.
LITERATURE18_E4B = (
    "fcast_renewable_adjusted_direct_load",
    "ramp2_load",
    "ramp2_solar",
    "ramp2_renewable",
    "ramp2_residual_load",
    "err_renewable_adjusted_direct_28d_mean",
    "delta_renewable_adjusted_direct_28d_mean",
    "err_renewable_adjusted_direct_28d_std",
    "delta_renewable_adjusted_direct_28d_std",
    "err_renewable_adjusted_direct_28d_q10",
    "err_renewable_adjusted_direct_28d_q90",
    "delta_renewable_adjusted_direct_28d_q90",
    "regime_high_residual_load_renew",
    "regime_low_residual_load_renew",
    "regime_high_renewable_share",
    "regime_low_renewable_share",
    "regime_high_bidding_space_ratio",
    "regime_low_bidding_space_ratio",
)


def build_experiment_feature_manifest(frozen_selector: dict[str, Any], profile: str,
                                      store: "SequenceStore") -> dict[str, Any]:
    """E4-B: build an IN-MEMORY experiment feature manifest from the frozen selector.

    Does NOT modify the frozen selector JSON on disk. The base ``selector_sha256`` is preserved; a
    separate ``experiment_feature_profile_sha256`` records the experiment profile. Recovered features
    are re-routed to the Weak role only. The manifest retains FROZEN status so the eligibility,
    preprocessing and model contracts continue to accept it.

    Arms:
      selected222  -> identical to frozen (222 selected, Strong211 + Weak11).
      literature240 -> frozen 222 + EXACT 18 literature features (all originally Noise) -> Weak. Strong211 Weak29.
      all259       -> all 259 candidates; the 37 originally-Noise features -> Weak. Strong211 Weak48.
    """
    if profile == "selected222":
        return frozen_selector
    if profile not in {"literature240", "all259"}:
        raise ValueError("feature_recovery_profile must be selected222/literature240/all259")
    registry_order = list(store.feature_names)  # canonical 259 registry order
    role_by_name = {r["feature_name"]: dict(r) for r in frozen_selector["feature_roles"]}
    missing = [n for n in registry_order if n not in role_by_name]
    if missing:
        raise ContractError(f"frozen selector roles missing for {missing[:5]}")
    strong_roles = {"Forced-Core", "Forced-Temporal", "Strong-BOTH", "Strong-DIR",
                    "Strong-MAG", "Strong"}
    if profile == "literature240":
        absent = [n for n in LITERATURE18_E4B if n not in set(registry_order)]
        if absent:
            raise ContractError(f"E4-B literature18 features absent from candidate cube: {absent[:5]}")
        recovered = [n for n in LITERATURE18_E4B if role_by_name[n]["role"] == "Noise"]
        if len(recovered) != len(LITERATURE18_E4B):
            raise ContractError(
                f"E4-B literature240 requires all 18 features be frozen-selector Noise; "
                f"only {len(recovered)} are Noise")
        selected_set = set(frozen_selector["selected_features"]) | set(LITERATURE18_E4B)
    else:  # all259
        recovered = [n for n in registry_order if role_by_name[n]["role"] == "Noise"]
        if len(recovered) != 37:
            raise ContractError(f"E4-B all259 requires 37 frozen-selector Noise features; got {len(recovered)}")
        selected_set = set(registry_order)
    selected = [n for n in registry_order if n in selected_set]
    idx = [registry_order.index(n) for n in selected]
    if idx != sorted(idx):
        raise ContractError("E4-B selected_features are not in canonical registry order")
    rec_set = set(recovered)
    roles: list[dict[str, Any]] = []
    for name in registry_order:
        r = role_by_name[name]
        if name in rec_set and r["role"] == "Noise":
            r = dict(r)
            r["role"] = "Weak"
            r["experiment_recovered"] = True
        roles.append(r)
    weak_count = sum(1 for n in selected if role_by_name[n]["role"] == "Weak" or n in rec_set)
    strong_count = len(selected) - weak_count
    expected_strong = {"literature240": 211, "all259": 211}[profile]
    expected_weak = {"literature240": 29, "all259": 48}[profile]
    if strong_count != expected_strong or weak_count != expected_weak:
        raise ContractError(
            f"E4-B {profile} role counts wrong: strong={strong_count}(exp {expected_strong}), "
            f"weak={weak_count}(exp {expected_weak})")
    digest = hashlib.sha256()
    digest.update(json.dumps({"selected_features": selected, "feature_roles": roles},
                             ensure_ascii=False, sort_keys=True).encode("utf-8"))
    manifest = dict(frozen_selector)
    manifest["selected_features"] = selected
    manifest["selected_indices"] = idx
    manifest["feature_roles"] = roles
    manifest["status"] = "FROZEN"
    manifest["selector_sha256"] = frozen_selector.get("selector_sha256")
    manifest["experiment_feature_profile"] = profile
    manifest["experiment_feature_profile_sha256"] = digest.hexdigest()
    manifest["experiment_recovered_features"] = sorted(recovered)
    manifest["experiment_recovered_role"] = "Weak"
    manifest["experiment_strong_count"] = int(strong_count)
    manifest["experiment_weak_count"] = int(weak_count)
    return manifest


def _model_for(store: SequenceStore, selector: dict[str, Any], state: PreprocessorState,
               cfg: V2Config, *, legacy_v20: bool = False,
               architecture_mode: str = "full_current",
               direction_fusion_alpha: float | None = None,
               direction_tabular_mode: str = "current",
               direction_horizon_gate_mode: str = "current",
               strong_role_profile: str = "all",
               numeric_encoding_mode: str = "canonical",
               direction_readout_mode: str = "shared") -> nn.Module:
    roles = selector["feature_roles"]
    tabular = TabularEncoder(feature_names=state.feature_names, feature_roles=roles, ple_bins=state.ple_bins,
        d_tab=cfg.d_tab, k=cfg.k, n_blocks=cfg.n_blocks, dropout=cfg.dropout,
        ple_embedding_dim=cfg.ple_embedding_dim, arch_type=cfg.arch_type, ple_enabled=cfg.ple_enabled,
        strong_role_profile=strong_role_profile, numeric_encoding_mode=numeric_encoding_mode)
    temporal = TemporalEncoder(selected_feature_names=state.feature_names, d_time=cfg.d_time,
        hidden=cfg.time_hidden, fft_bins=cfg.fft_bins, fft_enabled=cfg.fft_enabled,
        future_conditioning=cfg.future_conditioning, temporal_clip_abs=cfg.temporal_clip_abs)
    if legacy_v20:
        return DualReadoutV2(cfg.d_tab,cfg.d_time,cfg.d_task,mode=cfg.mode,magnitude_eps=cfg.magnitude_eps,
            tabular_encoder=tabular,temporal_encoder=temporal,ensemble_size=None,magnitude_scale=state.target_scale_c)
    return DualBranchV21(cfg.d_tab,cfg.d_time,cfg.d_task,mode=cfg.mode,tabular_encoder=tabular,
        temporal_encoder=temporal,k=cfg.k,magnitude_scale=state.target_scale_c,magnitude_eps=cfg.magnitude_eps,
        architecture_mode=architecture_mode,direction_fusion_alpha=direction_fusion_alpha,
        direction_tabular_mode=direction_tabular_mode,
        direction_horizon_gate_mode=direction_horizon_gate_mode,
        direction_readout_mode=direction_readout_mode)


def _selected_arrays(store: SequenceStore, indices: Sequence[int], state: PreprocessorState):
    idx = np.asarray(indices, dtype=np.int64)
    x_hist = np.asarray(store.x_hist[idx], dtype=np.float32)
    x_future = np.asarray(store.x_future[idx][:, :, state.feature_indices], dtype=np.float32)
    y_source = np.asarray(store.y_source[idx], dtype=np.float32)
    y_model = source_to_model_target(y_source).astype(np.float32)
    return transform_hist(x_hist, state), transform_future(x_future, state), y_model, y_model / state.target_scale_c


def _loader(arrays, *, batch_size: int, shuffle: bool, seed: int, device: str) -> DataLoader:
    xh, xf, ym, ys = arrays
    tensors = TensorDataset(torch.from_numpy(xh), torch.from_numpy(xf), torch.from_numpy(ym), torch.from_numpy(ys))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, pin_memory=device.startswith("cuda"))


def _loss_on_loader(model: nn.Module, loader: DataLoader, device: str, state: PreprocessorState,
                    cfg: V2Config, *, legacy_v20: bool = False,
                    naive_magnitude: float | None = None) -> dict[str, Any]:
    model.eval()
    acc: dict[str, float] = {}
    n = 0
    truth_all=[]; direction_all=[]; probability_all=[]; magnitude_all=[]
    with torch.no_grad():
        for xh, xf, ym, ys in loader:
            xh, xf, ym, ys = xh.to(device), xf.to(device), ym.to(device), ys.to(device)
            out = model(xh, xf)
            loss = (v2_loss_compat(out,ys,w_pos=state.w_pos,w_nonpos=state.w_nonpos,
                    lambda_dir=cfg.lambda_dir,lambda_mag=cfg.lambda_mag,beta=cfg.smooth_l1_beta)
                    if legacy_v20 else v21_loss(out,ys,lambda_dir=cfg.lambda_dir,lambda_mag=cfg.lambda_mag))
            count = len(ys)
            n += count
            truth_all.append(ym.detach().cpu().numpy().reshape(-1))
            direction_all.append(out["direction_hat"].detach().cpu().numpy().reshape(-1))
            probability_all.append(out["p"].detach().cpu().numpy().reshape(-1))
            scale=state.target_scale_c if legacy_v20 else 1.0
            magnitude_all.append((out["magnitude_hat"]*scale).detach().cpu().numpy().reshape(-1))
            for key in (("L_total","L_dir","L_mag","L_point") if legacy_v20 else ("L_total","L_dir","L_mag")):
                acc[key] = acc.get(key, 0.0) + float(loss[key].detach().cpu()) * count
    if not n:
        raise ContractError("empty monitor/evaluation dataset")
    y=np.concatenate(truth_all); pred=np.concatenate(direction_all); prob=np.concatenate(probability_all); mag=np.concatenate(magnitude_all)
    kpi=canonical_metrics(y,pred,mag,direction_probability=prob)
    result = {**{key:value/n for key,value in acc.items()},"raw_direction_accuracy":kpi["raw_direction_accuracy"],
            "balanced_accuracy":kpi["balanced_accuracy"],"positive_recall":kpi["positive_recall"],
            "nonpositive_recall":kpi["nonpositive_recall"],"auc":kpi["auc"],"brier":kpi["brier"],
            "magnitude_mae":kpi["magnitude_mae"]}
    if naive_magnitude is not None:
        naive_mae = float(np.mean(np.abs(np.abs(y) - float(naive_magnitude))))
        result.update({"magnitude_naive_value": float(naive_magnitude), "magnitude_mae_naive": naive_mae,
                       "magnitude_skill": 1.0 - float(kpi["magnitude_mae"]) / naive_mae if naive_mae > 0 else "N/A_ZERO_NAIVE_MAE",
                       "predicted_positive_fraction": float(np.mean(pred))})
    return result


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str,
                 state: PreprocessorState, cfg: V2Config, scaler, *, amp: bool,legacy_v20: bool=False) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    count = 0
    for xh, xf, _, ys in loader:
        xh, xf, ys = xh.to(device), xf.to(device), ys.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            enabled=amp, dtype=torch.float16 if device.startswith("cuda") else torch.bfloat16):
            out = model(xh, xf)
            if not torch.isfinite(out["p"]).all() or not torch.isfinite(out["magnitude_hat"]).all():
                raise FloatingPointError("non-finite model outputs during training")
            losses = (v2_loss_compat(out,ys,w_pos=state.w_pos,w_nonpos=state.w_nonpos,
                lambda_dir=cfg.lambda_dir,lambda_mag=cfg.lambda_mag,beta=cfg.smooth_l1_beta)
                if legacy_v20 else v21_loss(out,ys,lambda_dir=cfg.lambda_dir,lambda_mag=cfg.lambda_mag))
        loss = losses["L_total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        n = len(ys); count += n
        for key in (("L_total","L_dir","L_mag","L_point") if legacy_v20 else ("L_total","L_dir","L_mag")):
            sums[key] = sums.get(key, 0.0) + float(losses[key].detach().cpu()) * n
    if not count:
        raise ContractError("empty training dataloader")
    return {key: value / count for key, value in sums.items()}


def _train_epoch_experiment(model: DualBranchV21, loader: DataLoader, optimizer: torch.optim.Optimizer,
                            device: str, cfg: V2Config, scaler, *, objective_mode: str,
                            gradient_policy: str, amp: bool, epoch: int, diagnostic: bool,
                            batches_per_epoch: int, diagnostic_rows: list[dict[str, Any]],
                            direction_weight_pos: float = 1.0, direction_weight_nonpos: float = 1.0,
                            protected_group_params: dict[str, tuple[nn.Parameter, ...]] | None = None,
                            projection_rows: list[dict[str, Any]] | None = None) -> dict[str, float]:
    """Experiment-only update path. Canonical default continues through _train_epoch unchanged."""
    if objective_mode not in OBJECTIVE_MODES or gradient_policy not in GRADIENT_POLICIES:
        raise ValueError("invalid objective/gradient policy")
    if gradient_policy == "direction_protected" and (objective_mode != "joint_v21" or amp):
        raise ValueError("direction_protected requires joint_v21 and AMP disabled")
    parameter_groups = {"tabular_encoder": tuple(p for p in model.tabular_encoder.parameters() if p.requires_grad),
                        "temporal_encoder": tuple(p for p in model.temporal_encoder.parameters() if p.requires_grad)}
    model.train(); sums: dict[str, float] = {}; count = 0
    for batch_index, (xh, xf, _, ys) in enumerate(loader):
        xh, xf, ys = xh.to(device), xf.to(device), ys.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            enabled=amp, dtype=torch.float16 if device.startswith("cuda") else torch.bfloat16):
            out = model(xh, xf, direction_protected=gradient_policy == "direction_protected")
            losses = v21_loss(out, ys, lambda_dir=cfg.lambda_dir, lambda_mag=cfg.lambda_mag,
                              direction_weight_pos=direction_weight_pos,
                              direction_weight_nonpos=direction_weight_nonpos)
        if not all(torch.isfinite(losses[key]) for key in ("L_dir", "L_mag", "L_total")):
            raise FloatingPointError("non-finite V2.1 experiment loss")
        if diagnostic and batch_index < batches_per_epoch:
            diagnostic_rows.extend(gradient_diagnostic_rows(losses["L_dir"], losses["L_mag"],
                parameter_groups, epoch=epoch,
                batch_index=batch_index))
        objective = objective_tensor(losses, objective_mode, lambda_dir=cfg.lambda_dir, lambda_mag=cfg.lambda_mag)
        if gradient_policy == "direction_protected":
            if protected_group_params is None:
                raise RuntimeError("protected parameter groups were not audited")
            protected_status, _ = protected_gradients(losses["L_dir"], losses["L_mag"], protected_group_params,
                                lambda_dir=cfg.lambda_dir, lambda_mag=cfg.lambda_mag)
            group_grad_norms = clip_protected_gradient_groups(protected_group_params, max_norm=1.0)
            if projection_rows is not None:
                for row in protected_status["projection_rows"]:
                    projection_rows.append({
                        "epoch": epoch,
                        "batch_index": batch_index,
                        **row,
                        "direction_owned_preclip_norm": group_grad_norms["direction_owned"],
                        "magnitude_owned_preclip_norm": group_grad_norms["magnitude_owned"],
                        "shared_preclip_norm": group_grad_norms["shared"],
                    })
            grad_norm = max(group_grad_norms.values()) if group_grad_norms else 0.0
            optimizer.step()
        else:
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer); scaler.update()
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite V2.1 experiment gradient norm")
        n = len(ys); count += n
        for key in ("L_total", "L_dir", "L_mag"):
            sums[key] = sums.get(key, 0.0) + float(losses[key].detach().cpu()) * n
        sums["objective"] = sums.get("objective", 0.0) + float(objective.detach().cpu()) * n
    if not count:
        raise ContractError("empty training dataloader")
    return {key: value / count for key, value in sums.items()}


def _stage_b_freeze_audit(model: nn.Module) -> dict[str, Any]:
    """Parameter groups are built from module ownership/identity, never name substrings."""
    groups: dict[str, set[int]] = {}
    tab = model.tabular_encoder
    backbone = tab.backbone
    topology = tab.parameter_topology()
    member_ids = tab.member_specific_parameter_ids() if topology["topology_audit"] == "PASS" else set()
    shared_ids = tab.shared_backbone_parameter_ids()
    groups["shared_backbone_frozen"] = shared_ids
    groups["member_specific_trainable"] = member_ids
    groups["ple_preprocessing_frozen"] = {id(p) for p in tab.ple.parameters()} if tab.ple is not None else set()
    groups["weak_trainable"] = {id(p) for p in tab.weak_mlp.parameters()} if tab.weak_mlp is not None else set()
    groups["strong_weak_gates_trainable"] = {id(tab.horizon_gate_logits)}
    groups["temporal_trainable"] = {id(p) for p in model.temporal_encoder.parameters()}
    groups["task_adapter_trainable"] = {id(p) for p in model.adapters.parameters() if p.requires_grad or p.numel() > 0}
    magnitude_heads=("magnitude_head",) if not model.legacy_v20 else ("m_pos_head","m_nonpos_head")
    groups["heads_trainable"] = {id(p) for p in model.direction_head.parameters()}
    for head_name in magnitude_heads:
        groups["heads_trainable"] |= {id(p) for p in getattr(model,head_name).parameters()}
    owner_sets = [ids for ids in groups.values() if ids]
    overlap = any(owner_sets[i] & owner_sets[j] for i in range(len(owner_sets)) for j in range(i + 1, len(owner_sets)))
    all_ids = {id(p) for p in model.parameters()}
    owned = set().union(*owner_sets) if owner_sets else set()
    unknown = all_ids - owned
    for p in model.parameters():
        p.requires_grad_(False)
    if topology["topology_audit"] == "PASS":
        for p in backbone.affine.parameters():
            p.requires_grad_(True)
    head_modules=[model.direction_head]+[getattr(model,name) for name in magnitude_heads]
    for module in (tab.weak_mlp, model.temporal_encoder, model.adapters, *head_modules):
        if module is not None:
            for p in module.parameters():
                p.requires_grad_(True)
    tab.horizon_gate_logits.requires_grad_(True)
    actual_trainable = {id(p) for p in model.parameters() if p.requires_grad}
    expected_trainable = groups["member_specific_trainable"] | groups["weak_trainable"] | groups["strong_weak_gates_trainable"] | groups["temporal_trainable"] | groups["task_adapter_trainable"] | groups["heads_trainable"]
    audit_pass = not overlap and not unknown and actual_trainable == expected_trainable
    return {
        "status": "PASS" if audit_pass else "FAIL",
        "ownership_method": "Python module object ownership + parameter object identity",
        "groups": {key: {"parameter_count": sum(p.numel() for p in model.parameters() if id(p) in ids),
                         "parameter_tensors": len(ids)} for key, ids in groups.items()},
        "unknown_parameter_tensors": len(unknown),
        "overlapping_owner_groups": overlap,
        "shared_tabm_backbone_frozen": not any(p.requires_grad for p in backbone.parameters() if id(p) in shared_ids),
        "ple_frozen": tab.ple is None or not any(p.requires_grad for p in tab.ple.parameters()),
        "official_tabm_topology": topology,
        "stage_b_tabm_member_adaptation": "PASS" if topology["topology_audit"] == "PASS" else "UNAVAILABLE_WITH_OFFICIAL_API",
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "trainable_parameter_tensors": len(actual_trainable),
    }


def _predict_one(model: DualReadoutV2, xh: np.ndarray, xf: np.ndarray, device: str) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(xh).to(device), torch.from_numpy(xf).to(device))
    result = {key: value.detach().cpu().numpy() for key, value in out.items()
              if isinstance(value, torch.Tensor) and key not in {"H_tab", "H_time"}}
    return result


def stage_a_split_indices(train_indices, *, monitor_days: int | None = None,
                          history_window_days: int | None = None):
    """Chronological Stage-A BASE/MONITOR split.

    Default (None/None) is the canonical 80/20 percentage split, bit-identical to prior behaviour.
    E3-A explicit split: the newest ``monitor_days`` eligible days are MONITOR; BASE is the newest
    ``history_window_days`` days strictly before MONITOR (None keeps all earlier days). BASE and
    MONITOR never overlap and preserve chronological order.
    """
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if monitor_days is None:
        if history_window_days is not None:
            raise ContractError("stage_a_history_window_days requires stage_a_monitor_days")
        split = max(2, min(len(train_indices) - 1, int(math.floor(0.8 * len(train_indices)))))
        return train_indices[:split], train_indices[split:], "canonical_percent80"
    monitor_days = int(monitor_days)
    if monitor_days < 1 or (history_window_days is not None and int(history_window_days) < 1):
        raise ContractError("stage_a_monitor_days/stage_a_history_window_days must be >= 1")
    if len(train_indices) <= monitor_days:
        raise ContractError("stage_a_monitor_days must be smaller than the eligible history")
    monitor_indices = train_indices[-monitor_days:]
    candidate_base = train_indices[:-monitor_days]
    if history_window_days is None:
        base_indices, mode = candidate_base, "explicit_monitor_expanding"
    else:
        if len(candidate_base) < int(history_window_days):
            raise ContractError("stage_a_history_window_days exceeds the candidate base history")
        base_indices, mode = candidate_base[-int(history_window_days):], "explicit_monitor_rolling"
    if len(base_indices) < 2:
        raise ContractError("explicit split left fewer than 2 BASE days")
    return base_indices, monitor_indices, mode


E4_CALIBRATOR_DAYS = 90
E4_CHECKPOINT_MONITOR_MIN_DAYS = 60


def e4a_three_way_split(monitor_indices, *, calibrator_days: int = E4_CALIBRATOR_DAYS):
    """E4-A experiment-only carve of the canonical FULL_MONITOR (never changes BASE).

    CHECKPOINT_MONITOR = older FULL_MONITOR days (used for Stage-A checkpoint selection)
    CALIBRATOR        = newest ``calibrator_days`` FULL_MONITOR days (used only to fit the stacker)
    Preserves chronological order; the two parts are adjacent and never overlap.
    """
    monitor = np.asarray(monitor_indices, dtype=np.int64)
    calibrator_days = int(calibrator_days)
    if calibrator_days < 1:
        raise ContractError("calibrator_days must be >= 1")
    if len(monitor) < calibrator_days + E4_CHECKPOINT_MONITOR_MIN_DAYS:
        raise ContractError(
            f"E4-A requires FULL_MONITOR >= {calibrator_days + E4_CHECKPOINT_MONITOR_MIN_DAYS} days "
            f"(CALIBRATOR={calibrator_days} + CHECKPOINT_MONITOR>={E4_CHECKPOINT_MONITOR_MIN_DAYS})")
    calibrator = monitor[-calibrator_days:]
    checkpoint_monitor = monitor[:-calibrator_days]
    return checkpoint_monitor, calibrator



def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def train_target_day(target_day: date | str, *, mode: str = "A2", profile: str = "default",
                     train_mode: str = "stage_a", legacy_v20: bool = False,
                     config: V2Config | None = None, store: SequenceStore | None = None,
                     selector_path: Path | None = None, output_dir: Path | None = None,
                     objective_mode: str = "joint_v21", checkpoint_policy: str = "v21_guardrail",
                     direction_bce_min_delta: float = 1e-4, magnitude_l1_min_delta: float = 1e-4,
                     gradient_policy: str = "vanilla", gradient_diagnostics: bool = False,
                     gradient_batches_per_epoch: int = 2, experiment_only: bool = False,
                     architecture_mode: str = "full_current",
                     direction_fusion_alpha: float | None = None,
                     direction_tabular_mode: str = "current",
                     direction_horizon_gate_mode: str = "current",
                     strong_role_profile: str = "all",
                     numeric_encoding_mode: str = "canonical",
                     direction_readout_mode: str = "shared",
                     direction_postprocess_mode: str = "none",
                     e4_three_way_split: bool = False,
                     direction_class_weight_mode: str = "unweighted",
                     stage_a_monitor_days: int | None = None,
                     stage_a_history_window_days: int | None = None,
                     feature_recovery_profile: str = "selected222") -> dict[str, Any]:
                     
    wall_start = time.perf_counter()
    if train_mode not in {"stage_a","stage_ab","full_retrain"}:
        raise ValueError("train_mode must be stage_a/stage_ab/full_retrain")
    cfg = config or V2Config(mode=mode)
    from dataclasses import replace
    cfg = replace(cfg, mode=mode)
    if profile == "smoke":
        cfg = cfg.with_profile("smoke")
    if objective_mode not in OBJECTIVE_MODES or checkpoint_policy not in CHECKPOINT_POLICIES or gradient_policy not in GRADIENT_POLICIES:
        raise ValueError("invalid experiment policy")
    if architecture_mode not in ARCHITECTURE_MODES:
        raise ValueError("architecture_mode must be full_current/tabular_only/temporal_only")
    if direction_tabular_mode not in {"current", "strong_only", "weak_only"}:
        raise ValueError("direction_tabular_mode must be current/strong_only/weak_only")
    if direction_horizon_gate_mode not in {"current", "fixed_08", "global_learnable"}:
        raise ValueError("direction_horizon_gate_mode must be current/fixed_08/global_learnable")
    if direction_horizon_gate_mode != "current" and (
        direction_tabular_mode != "current" or architecture_mode != "full_current" or
        direction_fusion_alpha is None or abs(float(direction_fusion_alpha)-.8)>1e-12):
        raise ValueError("non-current horizon gate requires direction_tabular_mode=current, architecture_mode=full_current, direction_fusion_alpha=.8")
    if strong_role_profile not in {"all", "drop_mag", "drop_dir"}:
        raise ValueError("strong_role_profile must be all/drop_mag/drop_dir")
    if strong_role_profile != "all" and (legacy_v20 or objective_mode != "dir_only" or architecture_mode != "full_current" or direction_tabular_mode != "current" or direction_horizon_gate_mode != "current" or direction_fusion_alpha is None or abs(float(direction_fusion_alpha)-.8)>1e-12):
        raise ValueError("non-default strong_role_profile requires dir_only/full_current/current routes and fixed alpha=.8")
    if numeric_encoding_mode not in {"canonical", "raw_only", "ple_only"}:
        raise ValueError("numeric_encoding_mode must be canonical/raw_only/ple_only")
    if numeric_encoding_mode != "canonical" and (
        legacy_v20 or objective_mode != "dir_only" or architecture_mode != "full_current" or
        direction_tabular_mode != "current" or direction_horizon_gate_mode != "current" or
        strong_role_profile != "all" or direction_fusion_alpha is None or
        abs(float(direction_fusion_alpha)-.8) > 1e-12 or train_mode != "stage_a"):
        raise ValueError("non-canonical numeric_encoding_mode requires dir_only/full_current/current routes, strong_role_profile=all, fixed alpha=.8 and train_mode=stage_a")
    if numeric_encoding_mode == "ple_only" and not cfg.ple_enabled:
        raise ValueError("numeric_encoding_mode=ple_only requires cfg.ple_enabled=True")
    if direction_readout_mode not in {"shared", "segment_bias", "segment_heads"}:
        raise ValueError("direction_readout_mode must be shared/segment_bias/segment_heads")
    if direction_postprocess_mode not in POSTPROCESS_MODES:
        raise ValueError("direction_postprocess_mode must be none/segment_logit/regime_logit")
    if direction_class_weight_mode not in {"unweighted", "sqrt_balanced", "full_balanced"}:
        raise ValueError("direction_class_weight_mode must be unweighted/sqrt_balanced/full_balanced")
    if direction_readout_mode != "shared" and (
        legacy_v20 or objective_mode != "dir_only" or architecture_mode != "full_current" or
        direction_tabular_mode != "current" or direction_horizon_gate_mode != "current" or
        strong_role_profile != "all" or numeric_encoding_mode != "canonical" or
        direction_fusion_alpha is None or abs(float(direction_fusion_alpha)-.8) > 1e-12 or
        train_mode != "stage_a"):
        raise ValueError("non-shared direction_readout_mode requires dir_only/full_current/current routes, all roles, canonical encoding, fixed alpha=.8 and train_mode=stage_a")
    if e4_three_way_split and (
        legacy_v20 or objective_mode != "dir_only" or train_mode != "stage_a" or
        architecture_mode != "full_current" or direction_tabular_mode != "current" or
        direction_horizon_gate_mode != "current" or strong_role_profile != "all" or
        numeric_encoding_mode != "canonical" or direction_readout_mode != "segment_heads" or
        direction_fusion_alpha is None or abs(float(direction_fusion_alpha)-.8) > 1e-12 or
        stage_a_monitor_days is not None or stage_a_history_window_days is not None or
        direction_class_weight_mode != "unweighted"):
        raise ValueError("E4-A three-way split requires the canonical-split Q2 dir_only route: dir_only/stage_a/segment_heads/all roles/canonical encoding/fixed alpha=.8")
    if direction_postprocess_mode != "none" and not e4_three_way_split:
        raise ValueError("Direction postprocessing requires the E4-A three-way split so the stacker never refits the checkpoint-selection monitor")
    if direction_class_weight_mode != "unweighted" and (
        legacy_v20 or objective_mode != "dir_only" or train_mode != "stage_a" or
        architecture_mode != "full_current" or direction_tabular_mode != "current" or
        direction_horizon_gate_mode != "current" or strong_role_profile != "all" or
        numeric_encoding_mode != "canonical" or direction_readout_mode != "segment_heads" or
        direction_postprocess_mode != "none" or direction_fusion_alpha is None or
        abs(float(direction_fusion_alpha)-.8) > 1e-12 or stage_a_monitor_days is not None or
        stage_a_history_window_days is not None):
        raise ValueError("Direction class weighting requires canonical-split Q2 dir_only route with no postprocessing")
    if feature_recovery_profile not in {"selected222", "literature240", "all259"}:
        raise ValueError("feature_recovery_profile must be selected222/literature240/all259")
    if feature_recovery_profile != "selected222" and (
        legacy_v20 or objective_mode != "dir_only" or train_mode != "stage_a" or
        architecture_mode != "full_current" or direction_tabular_mode != "current" or
        direction_horizon_gate_mode != "current" or strong_role_profile != "all" or
        numeric_encoding_mode != "canonical" or direction_readout_mode != "segment_heads" or
        direction_postprocess_mode != "none" or direction_class_weight_mode != "unweighted" or
        direction_fusion_alpha is None or abs(float(direction_fusion_alpha) - .8) > 1e-12 or
        e4_three_way_split or stage_a_monitor_days is not None or
        stage_a_history_window_days is not None):
        raise ValueError("feature_recovery_profile requires the canonical Q2 dir_only route: "
                         "dir_only/stage_a/full_current/current routes/all roles/canonical encoding/"
                         "segment_heads/unweighted/no postprocess/no split")
    if stage_a_monitor_days is not None:
        if stage_a_monitor_days < 1 or (stage_a_history_window_days is not None and stage_a_history_window_days < 1):
            raise ValueError("stage_a_monitor_days/stage_a_history_window_days must be >= 1")
        if (legacy_v20 or objective_mode != "dir_only" or train_mode != "stage_a" or
            architecture_mode != "full_current" or direction_tabular_mode != "current" or
            direction_horizon_gate_mode != "current" or strong_role_profile != "all" or
            numeric_encoding_mode != "canonical" or direction_readout_mode != "segment_heads" or
            direction_fusion_alpha is None or abs(float(direction_fusion_alpha)-.8) > 1e-12):
            raise ValueError("experiment-only stage_a_monitor_days requires the E3-A route: dir_only/stage_a/full_current/current routes/all roles/canonical encoding/segment_heads/fixed alpha=.8")
    elif stage_a_history_window_days is not None:
        raise ValueError("stage_a_history_window_days requires stage_a_monitor_days")
    if direction_tabular_mode != "current" and architecture_mode != "full_current":
        raise ValueError("Strong/Weak Direction routing requires architecture_mode=full_current")
    if direction_fusion_alpha is not None:
        if architecture_mode != "full_current":
            raise ValueError("direction_fusion_alpha requires architecture_mode=full_current")
        if not 0.0 <= float(direction_fusion_alpha) <= 1.0:
            raise ValueError("direction_fusion_alpha must be within [0,1]")
    if direction_bce_min_delta < 0 or magnitude_l1_min_delta < 0 or gradient_batches_per_epoch < 1:
        raise ValueError("invalid experiment early-stop delta or diagnostics batch count")
    if objective_mode == "mag_only" and checkpoint_policy != "magnitude_first":
        raise ValueError("mag_only requires checkpoint_policy=magnitude_first")
    if checkpoint_policy == "magnitude_first" and objective_mode != "mag_only":
        raise ValueError("magnitude_first checkpoint is reserved for mag_only experiments")
    if legacy_v20 and (objective_mode != "joint_v21" or checkpoint_policy != "v21_guardrail" or
                       gradient_policy != "vanilla" or gradient_diagnostics or
                       architecture_mode != "full_current" or direction_fusion_alpha is not None or
                       direction_tabular_mode != "current" or direction_horizon_gate_mode != "current" or strong_role_profile != "all" or
                       numeric_encoding_mode != "canonical" or direction_readout_mode != "shared" or
                       direction_postprocess_mode != "none" or e4_three_way_split or direction_class_weight_mode != "unweighted" or
                       stage_a_monitor_days is not None or stage_a_history_window_days is not None):
        raise ValueError("experiment policies require the canonical V2.1 model, not legacy_v20")
    if gradient_policy == "direction_protected" and objective_mode != "joint_v21":
        raise ValueError("direction_protected only supports joint_v21 objective")
    target = pd.Timestamp(target_day).date()
    seed_everything(cfg.seed)
    store = store or SequenceStore.load()
    selector, selector_hash = load_frozen_selector(selector_path or default_selector_path())
    if selector.get("selector_cutoff") != cfg.selector_cutoff:
        raise ContractError("selector cutoff differs from V2 configured fixed cutoff")
    exp_selector = build_experiment_feature_manifest(selector, feature_recovery_profile, store)
    eligibility = store.eligibility(exp_selector, current_target_day=target)
    train_indices = np.asarray(eligibility["eligible_indices"], dtype=np.int64)
    if len(train_indices) < 40:
        raise ContractError("fewer than 40 eligible D-2 training samples")
    target_matches = np.flatnonzero(np.asarray([d == target for d in store.days]))
    if len(target_matches) != 1:
        raise ContractError(f"target day {target} is not a unique sequence sample")
    target_index = int(target_matches[0])
    target_ok = store.eligibility(exp_selector, requested_days=[target])["eligible_indices"]
    if target_index not in target_ok:
        raise ContractError(f"selected-feature quarantine excludes target/evaluation day {target}")

    def _day_span(idx):
        if len(idx) == 0:
            return {"count": 0, "start": None, "end": None}
        ds = [store.days[int(i)] for i in idx]
        return {"count": int(len(idx)), "start": str(min(ds)), "end": str(max(ds))}

    if train_mode=="full_retrain":
        base_indices,monitor_indices=train_indices,np.asarray([],dtype=np.int64)
        split_mode = "full_retrain"
    else:
        base_indices, monitor_indices, split_mode = stage_a_split_indices(
            train_indices, monitor_days=stage_a_monitor_days,
            history_window_days=stage_a_history_window_days)
    split_audit = {"split_mode": split_mode,
                   "stage_a_eligible": _day_span(train_indices),
                   "stage_a_base_train": _day_span(base_indices),
                   "stage_a_monitor": _day_span(monitor_indices),
                   "stage_a_monitor_days_requested": stage_a_monitor_days,
                   "stage_a_history_window_days_requested": stage_a_history_window_days}
    if e4_three_way_split:
        # E4-A: BASE and FULL_MONITOR are unchanged; carve CALIBRATOR out of the newest monitor days so
        # the stacker is never fitted on the days that selected the deep checkpoint.
        checkpoint_monitor_indices, calibrator_indices = e4a_three_way_split(monitor_indices)
        split_audit.update({"e4_three_way_split": True,
                            "e4_calibrator_days": int(E4_CALIBRATOR_DAYS),
                            "e4_checkpoint_monitor": _day_span(checkpoint_monitor_indices),
                            "e4_calibrator": _day_span(calibrator_indices)})
    else:
        checkpoint_monitor_indices, calibrator_indices = monitor_indices, np.asarray([], dtype=np.int64)
        split_audit["e4_three_way_split"] = False
    state = fit_preprocessor(store, base_indices, exp_selector, selector_sha256=selector_hash,
        n_bins=cfg.ple_bins, ple_embedding_dim=cfg.ple_embedding_dim,
        future_clip_abs=cfg.future_clip_abs, temporal_clip_abs=cfg.temporal_clip_abs,
        clip_after_robust_scale=cfg.clip_after_robust_scale, ple_enabled=cfg.ple_enabled)
    if legacy_v20:
        from dataclasses import replace as dc_replace
        train_y=source_to_model_target(np.asarray(store.y_source[base_indices],dtype=np.float64)).reshape(-1)
        npos=int((train_y>0).sum()); nneg=int((train_y<=0).sum()); nt=npos+nneg
        state=dc_replace(state,w_pos=nt/(2*max(npos,1)),w_nonpos=nt/(2*max(nneg,1)))
    dir_w_pos, dir_w_nonpos = direction_class_weights(state, direction_class_weight_mode)
    device = resolve_device(cfg)
    model = _model_for(store, exp_selector, state, cfg,legacy_v20=legacy_v20,
                       architecture_mode=architecture_mode,
                       direction_fusion_alpha=direction_fusion_alpha,
                       direction_tabular_mode=direction_tabular_mode,
                       direction_horizon_gate_mode=direction_horizon_gate_mode,
                       strong_role_profile=strong_role_profile,
                       numeric_encoding_mode=numeric_encoding_mode,
                       direction_readout_mode=direction_readout_mode).to(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    amp = bool(cfg.amp and device.startswith("cuda"))
    protected_amp_reason = None
    if gradient_policy == "direction_protected":
        amp = False
        protected_amp_reason = "manual_gradient_projection_safety"
    stage_a_train = _loader(_selected_arrays(store, base_indices, state), batch_size=cfg.batch_size,
                            shuffle=True, seed=cfg.seed, device=device)
    stage_a_monitor = (_loader(_selected_arrays(store, checkpoint_monitor_indices, state), batch_size=cfg.batch_size,
                              shuffle=False, seed=cfg.seed, device=device) if len(checkpoint_monitor_indices) else None)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    is_experiment = (experiment_only or objective_mode != "joint_v21" or checkpoint_policy != "v21_guardrail" or
                     gradient_policy != "vanilla" or gradient_diagnostics or
                     architecture_mode != "full_current" or direction_fusion_alpha is not None or
                     direction_tabular_mode != "current" or direction_horizon_gate_mode != "current" or strong_role_profile != "all" or
                     numeric_encoding_mode != "canonical" or direction_readout_mode != "shared" or
                     direction_postprocess_mode != "none" or e4_three_way_split or direction_class_weight_mode != "unweighted" or
                     stage_a_monitor_days is not None)
    use_experiment_epoch = (objective_mode != "joint_v21" or gradient_policy != "vanilla" or gradient_diagnostics)
    protected_audit = None
    protected_group_params = None
    if gradient_policy == "direction_protected":
        protected_audit, protected_group_params = audit_protected_parameter_groups(model)
    base_y_raw = source_to_model_target(np.asarray(store.y_source[base_indices], dtype=np.float64)).reshape(-1)
    magnitude_naive = float(np.median(np.abs(base_y_raw))) if is_experiment else None
    variant="legacy_v20_" if legacy_v20 else ""
    run_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f"{target.isoformat()}_{variant}{mode}_{train_mode}_{profile}_seed{cfg.seed}_{run_stamp}"
    if is_experiment:
        category = experiment_output_category(objective_mode, checkpoint_policy, gradient_policy, gradient_diagnostics)
        experiment_root = formal_output_dir() / "experiments" / "direction_first" / category
        # Keep the leaf short: the canonical Windows workspace path is long enough
        # that descriptive policy names can push predictions.parquet beyond MAX_PATH.
        short_objective = {"joint_v21": "joint", "dir_only": "dir", "mag_only": "mag"}[objective_mode]
        short_checkpoint = {"v21_guardrail": "v21g", "direction_first": "dirfirst",
                            "magnitude_first": "magfirst"}[checkpoint_policy]
        short_gradient = {"vanilla": "van", "direction_protected": "protected"}[gradient_policy]
        # Empty for full_current so the canonical experiment leaf keeps its exact historical shape.
        short_architecture = {"full_current": "", "tabular_only": "_tabonly",
                              "temporal_only": "_timeonly"}[architecture_mode]
        short_fusion_alpha = ("" if direction_fusion_alpha is None
                              else f"_fa{int(round(float(direction_fusion_alpha) * 100)):03d}")
        short_tabular_route = {"current":"", "strong_only":"_tstrong", "weak_only":"_tweak"}[direction_tabular_mode]
        short_gate_route = {"current":"", "fixed_08":"_gfix08", "global_learnable":"_gglobal"}[direction_horizon_gate_mode]
        short_role_route = {"all":"", "drop_mag":"_rdmag", "drop_dir":"_rddir"}[strong_role_profile]
        short_encoding_route = {"canonical":"", "raw_only":"_ncraw", "ple_only":"_ncple"}[numeric_encoding_mode]
        short_readout_route = {"shared":"", "segment_bias":"_rb", "segment_heads":"_rh"}[direction_readout_mode]
        short_post_route = {"none":"", "segment_logit":"_ps", "regime_logit":"_pr"}[direction_postprocess_mode]
        short_e4_route = "_e4" if e4_three_way_split else ""
        short_weight_route = {"unweighted":"", "sqrt_balanced":"_wsqrt", "full_balanced":"_wfull"}[direction_class_weight_mode]
        short_recovery_route = {"selected222":"", "literature240":"_fr240", "all259":"_fr259"}[feature_recovery_profile]
        # Keep these suffixes short: the frozen Windows workspace path is already long enough that
        # training_history.parquet can approach MAX_PATH (260) for deep experiment leaves.
        short_split_route = ("" if stage_a_monitor_days is None else
                             f"_m{stage_a_monitor_days}" + ("" if stage_a_history_window_days is None else f"w{stage_a_history_window_days}"))
        generated_dir = experiment_root / f"{target.isoformat()}_{short_objective}_{short_checkpoint}_{short_gradient}{short_architecture}{short_fusion_alpha}{short_tabular_route}{short_gate_route}{short_role_route}{short_encoding_route}{short_readout_route}{short_post_route}{short_e4_route}{short_weight_route}{short_recovery_route}{short_split_route}_{run_stamp}"
        if output_dir is not None and not Path(output_dir).resolve().is_relative_to(experiment_root.resolve()):
            raise ValueError("experiment output_dir must remain under outputs/tabm_v21/experiments/direction_first")
        run_dir = Path(output_dir) if output_dir is not None else generated_dir
    else:
        run_dir = Path(output_dir) if output_dir is not None else formal_output_dir() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    history: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    best_checkpoint_metrics, best_epoch, stale = None, 0, 0
    raw_anchor: float | None = None
    direction_raw_anchor: float | None = None
    direction_best_raw_bce: float | None = None
    magnitude_best_mae: float | None = None
    magnitude_best_l1: float | None = None
    diagnostic_warning_union: set[str] = set()
    amp_enabled = amp
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        if use_experiment_epoch:
            train_metrics = _train_epoch_experiment(model, stage_a_train, optimizer, device, cfg, scaler,
                objective_mode=objective_mode, gradient_policy=gradient_policy, amp=amp,
                epoch=epoch,
                diagnostic=gradient_diagnostics, batches_per_epoch=gradient_batches_per_epoch,
                diagnostic_rows=gradient_rows, direction_weight_pos=dir_w_pos,
                direction_weight_nonpos=dir_w_nonpos, protected_group_params=protected_group_params,
                projection_rows=projection_rows)
        else:
            train_metrics = _train_epoch(model, stage_a_train, optimizer, device, state, cfg, scaler, amp=amp,legacy_v20=legacy_v20)
        monitor_metrics = (_loss_on_loader(model, stage_a_monitor, device, state, cfg,legacy_v20=legacy_v20,
                                           naive_magnitude=magnitude_naive)
                           if stage_a_monitor is not None else None)
        elapsed = time.perf_counter() - t0
        alpha_dir, alpha_mag = model.adapters.alphas()
        gate_values = model.tabular_encoder.horizon_gate_values()
        seg_bias = (model.direction_hour_bias.detach().cpu().tolist()
                    if getattr(model, "direction_hour_bias", None) is not None else [None, None, None])
        history.append({"stage": "A", "epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                        **({f"monitor_{k}": v for k, v in monitor_metrics.items()} if monitor_metrics else {}),
                        "alpha_dir": float(alpha_dir.detach().cpu()), "alpha_mag": float(alpha_mag.detach().cpu()),
                        "strong_gate_mean": float(np.mean(gate_values)),
                        "direction_segment_bias_H1": seg_bias[0], "direction_segment_bias_H2": seg_bias[1],
                        "direction_segment_bias_H3": seg_bias[2],
                        "direction_global_gate": (float(torch.sigmoid(model.direction_global_gate_logit).detach().cpu())
                            if model.direction_global_gate_logit is not None else None), "epoch_seconds": elapsed})
        candidate=({"raw_direction_accuracy":monitor_metrics["raw_direction_accuracy"],
                    "magnitude_mae":monitor_metrics["magnitude_mae"],"L_total":monitor_metrics["L_total"],
                    "L_dir":monitor_metrics["L_dir"],"L_mag":monitor_metrics["L_mag"],"epoch":epoch} if monitor_metrics else None)
        if monitor_metrics and is_experiment:
            warnings = checkpoint_diagnostic_warnings(monitor_metrics)
            diagnostic_warning_union.update(warnings)
            history[-1]["monitor_diagnostic_warnings"] = ";".join(warnings)
        if candidate is None:  # Full Retrain: fixed-epoch final checkpoint, no monitor selection.
            best_checkpoint_metrics = {"epoch": epoch}
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "selector_sha256": selector_hash,
                        "preprocessor_state": state.to_dict(), "monitor_kpi": None}, run_dir / "full_retrain_final.pt")
        else:
            if checkpoint_policy == "v21_guardrail":
                checkpoint_state = update_checkpoint_state(candidate, raw_anchor, best_checkpoint_metrics,
                    tolerance=cfg.checkpoint_raw_tolerance)
                raw_anchor = float(checkpoint_state["raw_anchor"])
                if checkpoint_state["selected_improved"]:
                    best_checkpoint_metrics = checkpoint_state["selected"]
                    best_epoch = epoch
                    torch.save({"model": model.state_dict(), "epoch": epoch, "selector_sha256": selector_hash,
                        "preprocessor_state": state.to_dict(), "monitor_kpi": monitor_metrics,
                        "checkpoint_policy": {"name": "v21_guardrail", "raw_anchor": raw_anchor,
                            "raw_tolerance": cfg.checkpoint_raw_tolerance,
                            "selected_checkpoint": best_checkpoint_metrics}}, run_dir / "stage_a_best.pt")
                progress = checkpoint_state["progress"]
            elif checkpoint_policy == "direction_first":
                selection = {"selected": best_checkpoint_metrics,
                             "selected_improved": direction_first_checkpoint_is_better(candidate, best_checkpoint_metrics)}
                if selection["selected_improved"]:
                    best_checkpoint_metrics = candidate.copy()
                    best_epoch = epoch
                    torch.save({"model": model.state_dict(), "epoch": epoch, "selector_sha256": selector_hash,
                        "preprocessor_state": state.to_dict(), "monitor_kpi": monitor_metrics,
                        "checkpoint_policy": {"name": "direction_first", "raw_anchor": max(
                            float(candidate["raw_direction_accuracy"]), float(direction_raw_anchor or float("-inf"))),
                            "selected_checkpoint": best_checkpoint_metrics}}, run_dir / "stage_a_best.pt")
                progress_state = update_direction_first_progress(candidate, direction_raw_anchor,
                    direction_best_raw_bce, min_delta=direction_bce_min_delta)
                direction_raw_anchor = progress_state["raw_anchor"]
                direction_best_raw_bce = progress_state["best_raw_bce"]
                progress = progress_state["progress"]
                history[-1]["direction_progress"] = progress
                history[-1]["direction_raw_anchor"] = direction_raw_anchor
                history[-1]["direction_best_raw_bce"] = direction_best_raw_bce
            else:
                improved = magnitude_first_checkpoint_is_better(candidate, best_checkpoint_metrics)
                if improved:
                    best_checkpoint_metrics = candidate.copy()
                    best_epoch = epoch
                    torch.save({"model": model.state_dict(), "epoch": epoch, "selector_sha256": selector_hash,
                        "preprocessor_state": state.to_dict(), "monitor_kpi": monitor_metrics,
                        "checkpoint_policy": {"name": "magnitude_first",
                            "selected_checkpoint": best_checkpoint_metrics}}, run_dir / "stage_a_best.pt")
                progress_state = update_magnitude_first_progress(candidate, magnitude_best_mae,
                    magnitude_best_l1, min_delta=magnitude_l1_min_delta)
                magnitude_best_mae = progress_state["best_mae"]
                magnitude_best_l1 = progress_state["best_l1_at_mae"]
                progress = progress_state["progress"]
                history[-1]["magnitude_progress"] = progress
                history[-1]["magnitude_best_mae"] = magnitude_best_mae
                history[-1]["magnitude_best_l1"] = magnitude_best_l1
            history[-1]["early_stop_progress"] = bool(progress)
            stale = 0 if progress else stale + 1
        if monitor_metrics is not None and stale >= cfg.patience:
            break
    if best_epoch == 0 or (train_mode != "full_retrain" and best_checkpoint_metrics is None):
        raise RuntimeError("training did not produce a checkpoint")
    stage_a_file=run_dir/("full_retrain_final.pt" if train_mode=="full_retrain" else "stage_a_best.pt")
    stage_a_checkpoint = torch.load(stage_a_file, map_location=device, weights_only=False)
    model.load_state_dict(stage_a_checkpoint["model"])

    freeze_audit = _stage_b_freeze_audit(model) if train_mode=="stage_ab" else {"status":"NOT_RUN","reason":"Stage B disabled for selected train_mode"}
    if train_mode=="stage_ab" and freeze_audit["status"] != "PASS":
        raise RuntimeError(f"Stage B ownership audit failed: {freeze_audit}")
    if train_mode=="stage_ab":
        recent_arrays = _selected_arrays(store, monitor_indices, state)
        base_arrays = _selected_arrays(store, base_indices, state)
        recent_ds = TensorDataset(*(torch.from_numpy(a) for a in recent_arrays))
        base_ds = TensorDataset(*(torch.from_numpy(a) for a in base_arrays))
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([recent_ds, base_ds])
        weights = torch.cat([torch.full((len(recent_ds),), cfg.stage_b_recent_quota / len(recent_ds)),
                             torch.full((len(base_ds),), (1.0 - cfg.stage_b_recent_quota) / len(base_ds))])
        stage_b_optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr * cfg.stage_b_lr_ratio, weight_decay=cfg.weight_decay)
        for epoch in range(1, cfg.stage_b_epochs + 1):
            sampler = WeightedRandomSampler(weights, num_samples=max(len(recent_ds), cfg.batch_size),
                replacement=True, generator=torch.Generator().manual_seed(cfg.seed + 1000 + epoch))
            loader = DataLoader(combined, batch_size=cfg.batch_size, sampler=sampler, num_workers=0,
                                pin_memory=device.startswith("cuda"))
            t0 = time.perf_counter()
            train_metrics = _train_epoch(model, loader, stage_b_optimizer, device, state, cfg, scaler, amp=amp_enabled,legacy_v20=legacy_v20)
            alpha_dir, alpha_mag = model.adapters.alphas()
            gate_values = model.tabular_encoder.horizon_gate_values()
            history.append({"stage": "B", "epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                            "alpha_dir": float(alpha_dir.detach().cpu()), "alpha_mag": float(alpha_mag.detach().cpu()),
                            "strong_gate_mean": float(np.mean(gate_values)), "epoch_seconds": time.perf_counter()-t0,
                            "recent_quota": cfg.stage_b_recent_quota, "base_replay_quota": 1-cfg.stage_b_recent_quota})
        final_checkpoint_path=run_dir / "stage_b_final.pt"
        torch.save({"model": model.state_dict(), "epoch": cfg.stage_b_epochs,
                    "selector_sha256": selector_hash, "preprocessor_state": state.to_dict()}, final_checkpoint_path)
    else:
        final_checkpoint_path=stage_a_file

    # Required checkpoint reload is checked before issuing a 24-hour prediction.
    final_checkpoint = torch.load(final_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(final_checkpoint["model"])
    target_xh = transform_hist(np.asarray(store.x_hist[target_index:target_index+1], dtype=np.float32), state)
    target_xf = transform_future(np.asarray(store.x_future[target_index:target_index+1][:, :, state.feature_indices], dtype=np.float32), state)
    target_future_max = float(np.max(np.abs(target_xf)))
    target_temporal_max = float(np.max(np.abs(target_xh)))
    numerical_stability_status = ("PASS" if np.isfinite(target_xf).all() and np.isfinite(target_xh).all()
        and target_future_max <= cfg.future_clip_abs + 1e-6
        and target_temporal_max <= cfg.temporal_clip_abs + 1e-6 else "FAIL")
    if numerical_stability_status != "PASS":
        raise FloatingPointError("target-day preprocessing failed finite/clip numerical audit")
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    prediction_start = time.perf_counter()
    prediction = _predict_one(model, target_xh, target_xf, device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    prediction_latency_ms = (time.perf_counter() - prediction_start) * 1000.0
    if prediction["p"].shape != (1, 24) or prediction["magnitude_hat"].shape != (1, 24):
        raise RuntimeError("prediction must contain all 24 canonical hours")
    y_true = source_to_model_target(np.asarray(store.y_source[target_index], dtype=np.float32)).reshape(-1)
    base_p = prediction["p"].reshape(-1)
    base_direction = prediction["direction_hat"].reshape(-1).astype(bool)
    p = base_p.copy()
    direction = base_direction.copy()
    postprocess_audit = None
    postprocess_audit_file = None
    if direction_postprocess_mode != "none":
        if len(calibrator_indices) != int(E4_CALIBRATOR_DAYS):
            raise ContractError(f"Direction postprocessor requires the E4-A CALIBRATOR ({E4_CALIBRATOR_DAYS} days)")
        cal_xh, cal_xf, cal_ym, _ = _selected_arrays(store, calibrator_indices, state)
        cal_prediction = _predict_one(model, cal_xh, cal_xf, device)
        stacker = fit_logistic_stacker(cal_prediction["p"], cal_xf, cal_ym, state.feature_names,
                                       direction_postprocess_mode)
        p = stacker.predict_proba(prediction["p"], target_xf, state.feature_names).reshape(-1)
        direction = p >= .5
        postprocess_audit = stacker.audit()
        postprocess_audit.update({
            "fit_scope": "E4-A_CALIBRATOR_only",
            "calibrator_day_count": int(len(calibrator_indices)),
            "calibrator_day_start": split_audit["e4_calibrator"]["start"],
            "calibrator_day_end": split_audit["e4_calibrator"]["end"],
            "checkpoint_monitor_day_count": int(len(checkpoint_monitor_indices)),
            "checkpoint_monitor_day_start": split_audit["e4_checkpoint_monitor"]["start"],
            "checkpoint_monitor_day_end": split_audit["e4_checkpoint_monitor"]["end"],
            "full_monitor_day_count": int(len(monitor_indices)),
            "target_day": target.isoformat(),
            "target_base_positive_fraction": float(np.mean(base_direction)),
            "target_post_positive_fraction": float(np.mean(direction)),
        })
        postprocess_audit_file = "direction_postprocessor.json"
        _write_json(run_dir / postprocess_audit_file, postprocess_audit)
    legacy_scale=state.target_scale_c if legacy_v20 else 1.0
    mag = prediction["magnitude_hat"].reshape(-1)*legacy_scale
    metrics = canonical_metrics(y_true, direction, mag, direction_probability=p,
                                month=np.repeat(target.month, 24), hour=np.arange(1, 25))
    if is_experiment:
        naive_mae = float(np.mean(np.abs(np.abs(y_true) - magnitude_naive)))
        metrics.update({"magnitude_naive_value": magnitude_naive, "magnitude_mae_naive": naive_mae,
                        "magnitude_skill": 1.0 - float(metrics["magnitude_mae"]) / naive_mae if naive_mae > 0 else "N/A_ZERO_NAIVE_MAE",
                        "predicted_positive_fraction": float(np.mean(direction))})
        diagnostic_warning_union.update(checkpoint_diagnostic_warnings({**metrics,
            "positive_recall": metrics.get("positive_recall"), "nonpositive_recall": metrics.get("nonpositive_recall")}))
        metrics["diagnostic_warnings"] = sorted(diagnostic_warning_union)
    pred_data={
        "target_day": target.isoformat(), "hour_business": np.arange(1, 25), "y_true_model": y_true,
        "p_positive": p, "direction_hat": direction.astype(np.int8), "magnitude_hat": mag,
        "signed_kpi_hat": (np.where(direction, np.maximum(mag, cfg.magnitude_eps), -np.maximum(mag, cfg.magnitude_eps))
            if direction_postprocess_mode != "none" else prediction["signed_kpi_hat"].reshape(-1)*legacy_scale),
        "alpha_dir": float(prediction["alpha_dir"]), "alpha_mag": float(prediction["alpha_mag"]),
    }
    if direction_postprocess_mode != "none":
        pred_data.update({"p_positive_base": base_p, "direction_hat_base": base_direction.astype(np.int8)})
    if not legacy_v20:
        pred_data.update({"direction_member_std":prediction["direction_member_std"].reshape(-1),
            "direction_vote_fraction":prediction["direction_vote_fraction"].reshape(-1),
            "direction_vote_entropy":prediction["direction_vote_entropy"].reshape(-1),
            "magnitude_member_std":prediction["magnitude_member_std"].reshape(-1),
            "magnitude_member_mean_scaled":prediction["a_scaled_members"].mean(axis=1).reshape(-1)})
    else:
        pred_data["legacy_y_soft"] = prediction["y_soft_train"].reshape(-1)*state.target_scale_c
    pred_frame = pd.DataFrame(pred_data)
    pred_frame.to_parquet(run_dir / "predictions.parquet", index=False)
    _write_json(run_dir / "metrics.json", metrics)
    pd.DataFrame(history).to_parquet(run_dir / "training_history.parquet", index=False)
    state_hash = state.save(run_dir / "preprocessor_state.json")
    numerical_audit = {
        "fit_scope": state.fit_scope,
        "clip_after_robust_scale": state.clip_after_robust_scale,
        "future_clip_abs": state.future_clip_abs,
        "temporal_clip_abs": state.temporal_clip_abs,
        "future": {"preclip_abs_p99": state.future_preclip_abs_p99,
            "preclip_abs_p999": state.future_preclip_abs_p999, "preclip_abs_max": state.future_preclip_abs_max,
            "postclip_abs_max": state.future_postclip_abs_max, "clip_fraction": state.future_clip_fraction_fit,
            "features": state.future_feature_audit, "robust_scale": dict(zip(state.feature_names, state.future_scale)),
            "median": dict(zip(state.feature_names, state.future_median))},
        "temporal": {"preclip_abs_p99": state.temporal_preclip_abs_p99,
            "preclip_abs_p999": state.temporal_preclip_abs_p999, "preclip_abs_max": state.temporal_preclip_abs_max,
            "postclip_abs_max": state.temporal_postclip_abs_max, "clip_fraction": state.temporal_clip_fraction_fit,
            "features": state.temporal_channel_audit,
            "robust_scale": dict(zip(TEMPORAL_FEATURES, state.temporal_scale)),
            "median": dict(zip(TEMPORAL_FEATURES, state.temporal_median))},
        "target_sample": {"target_day": target.isoformat(),
            "future_postclip_abs_max": target_future_max,
            "temporal_postclip_abs_max": target_temporal_max,
            "finite": True, "status": numerical_stability_status},
    }
    _write_json(run_dir / "numerical_audit.json", numerical_audit)
    gradient_summary_by_epoch: dict[str, Any] = {}
    gradient_conflict_file = None
    if gradient_diagnostics:
        pd.DataFrame(gradient_rows).to_parquet(run_dir / "gradient_conflict.parquet", index=False)
        gradient_conflict_file = "gradient_conflict.parquet"
        for epoch_value in sorted({int(row["epoch"]) for row in gradient_rows}):
            gradient_summary_by_epoch[str(epoch_value)] = summarize_gradient_rows(
                [row for row in gradient_rows if int(row["epoch"]) == epoch_value])
    protected_projection_file = None
    protected_projection_summary_by_epoch: dict[str, Any] = {}
    if gradient_policy == "direction_protected":
        pd.DataFrame(projection_rows).to_parquet(run_dir / "protected_projection.parquet", index=False)
        protected_projection_file = "protected_projection.parquet"
        for epoch_value in sorted({int(row["epoch"]) for row in projection_rows}):
            protected_projection_summary_by_epoch[str(epoch_value)] = summarize_projection_rows(
                [row for row in projection_rows if int(row["epoch"]) == epoch_value])
    parameter_group_audit_file = None
    if protected_audit is not None:
        _write_json(run_dir / "parameter_group_audit.json", protected_audit)
        parameter_group_audit_file = "parameter_group_audit.json"
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stage_a_history = [row for row in history if row.get("stage") == "A"]
    stage_a_epoch_seconds = [float(row["epoch_seconds"]) for row in stage_a_history]
    checkpoint_size_bytes = int(final_checkpoint_path.stat().st_size)
    cuda_peak_memory_bytes = (int(torch.cuda.max_memory_allocated(device))
        if device.startswith("cuda") and torch.cuda.is_available() else None)
    telemetry = ({
        "wall_time_total_seconds": float(time.perf_counter() - wall_start),
        "training_epoch_seconds_total": float(sum(stage_a_epoch_seconds)),
        "training_epoch_seconds_mean": float(np.mean(stage_a_epoch_seconds)) if stage_a_epoch_seconds else None,
        "training_epoch_seconds_p50": float(np.percentile(stage_a_epoch_seconds, 50)) if stage_a_epoch_seconds else None,
        "training_epoch_seconds_p95": float(np.percentile(stage_a_epoch_seconds, 95)) if stage_a_epoch_seconds else None,
        "best_epoch": int(best_epoch),
        "stop_epoch": int(stage_a_history[-1]["epoch"]) if stage_a_history else None,
        "epochs_run": len(stage_a_history),
        "prediction_latency_ms": float(prediction_latency_ms),
        "parameter_count_total": int(parameter_count),
        "parameter_count_trainable": int(trainable_parameter_count),
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "device": device,
        "cuda_peak_memory_bytes": cuda_peak_memory_bytes,
    } if is_experiment else {})
    if checkpoint_policy == "direction_first":
        raw_anchor = direction_raw_anchor
    guardrail_status = ("PASS" if train_mode == "full_retrain" or checkpoint_policy in {"direction_first", "magnitude_first"} or
        (best_checkpoint_metrics is not None and raw_anchor is not None and
         best_checkpoint_metrics["raw_direction_accuracy"] >= raw_anchor-cfg.checkpoint_raw_tolerance) else "FAIL")
    manifest = {
        "schema": "spread24_tabm_v21_run_v1", "run_id": run_id, "status": "COMPLETE",
        "target_day": target.isoformat(), "mode": mode, "profile": profile, "seed": cfg.seed,
        "train_mode":train_mode,"legacy_v20_ablation":legacy_v20,
        "config": cfg.to_dict(), "source_gate": store.source_gate["status"], "sequence_gate": store.sequence_gate["status"],
        **config_provenance(cfg),
        "source_sha256": store.source_sha256, "sequence_manifest_sha256": store.sequence_sha256,
        "selector_manifest": str(selector_path or default_selector_path()),
        "selector_sha256": selector_hash, "selected_feature_count": len(state.feature_names),
        "selected_features": state.feature_names, "preprocessor_state_sha256": state_hash,
        "preprocessing_fit_scope": state.fit_scope, "preprocessing_fit_day_start": state.fit_day_start,
        "numerical_audit_file": "numerical_audit.json",
        "numerical_stability_status": numerical_stability_status,
        "target_future_postclip_abs_max": target_future_max,
        "target_temporal_postclip_abs_max": target_temporal_max,
        "future_postclip_abs_max_fit": state.future_postclip_abs_max,
        "temporal_postclip_abs_max_fit": state.temporal_postclip_abs_max,
        "preprocessing_fit_day_end": state.fit_day_end, "stage_a_base_train_days": len(base_indices),
        "stage_a_monitor_days": len(monitor_indices), "stage_a_best_epoch": best_epoch,
        "stage_b_epochs": cfg.stage_b_epochs if train_mode=="stage_ab" else 0,
        "stage_b_implemented":True,"stage_b_production_default":False,
        "full_retrain_protocol":"all eligible D-2 days fit preprocessing; fixed epochs; no monitor/early-stop" if train_mode=="full_retrain" else None,
        "stage_b_recent_quota": cfg.stage_b_recent_quota,
        "stage_b_base_replay_quota": 1-cfg.stage_b_recent_quota,
        "canonical_spec":"legacy_v20_ablation" if legacy_v20 else "docs/17_最终模型设计与编码规范.md V2.1_KPI_ALIGNED_SPEC_FROZEN",
        "loss_contract":"unweighted BCEWithLogits(z_members,D) + L1(a_scaled_members,|Y|/c_mag); no signed point loss",
        "objective_mode": objective_mode,
        "architecture_mode": architecture_mode,
        "architecture_mode_canonical_spec": ("docs/17 V2.1 fusion: alpha*time_dir+(1-alpha)*tab_dir"
            if architecture_mode == "full_current" else
            "E2-A experiment-only big-block ablation; magnitude fusion unchanged"),
        "direction_fusion_alpha": direction_fusion_alpha,
        "direction_tabular_mode": direction_tabular_mode,
        "direction_horizon_gate_mode": direction_horizon_gate_mode,
        "strong_role_profile": strong_role_profile,
        "numeric_encoding_mode": numeric_encoding_mode,
        "numeric_encoding_mode_spec": (
            "E2-D1 experiment-only numeric representation family for Strong and Weak branches; "
            "preprocessing/robust scaling/clipping/PLE bins untouched"
            if numeric_encoding_mode != "canonical" else
            "canonical PLE(16,8)+raw skip when ple_enabled else raw-only"),
        "numeric_encoding_audit": model.tabular_encoder.encoding_audit() if hasattr(model.tabular_encoder, "encoding_audit") else None,
        "direction_readout_mode": direction_readout_mode,
        "direction_readout_mode_spec": (
            "E2-E1 experiment-only Direction readout over the fixed 1-8/9-16/17-24 segments; "
            "Magnitude head unchanged" if direction_readout_mode != "shared" else
            "canonical single shared member-wise Direction head across all 24 hours"),
        "direction_readout_audit": (model.direction_readout_audit()
            if hasattr(model, "direction_readout_audit") else None),
        "direction_postprocess_mode": direction_postprocess_mode,
        "direction_postprocess_mode_spec": (
            "E4-A experiment-only low-capacity logistic stacker fitted on the E4-A CALIBRATOR only; "
            "threshold remains 0.5" if direction_postprocess_mode != "none" else "none"),
        "direction_postprocess_audit_file": postprocess_audit_file,
        "direction_postprocess_audit": postprocess_audit,
        "e4_three_way_split": bool(e4_three_way_split),
        "e4_three_way_split_spec": (
            "E4-A experiment-only three-way chronological split: canonical BASE/FULL_MONITOR unchanged; "
            "CHECKPOINT_MONITOR selects the Stage-A checkpoint; CALIBRATOR (newest "
            f"{int(E4_CALIBRATOR_DAYS)}) is used only to fit the stacker" if e4_three_way_split else "none"),
        "e4_checkpoint_monitor_days": int(len(checkpoint_monitor_indices)),
        "e4_calibrator_days": int(len(calibrator_indices)),
        "feature_recovery_profile": feature_recovery_profile,
        "experiment_feature_profile_sha256": (exp_selector.get("experiment_feature_profile_sha256")
            if feature_recovery_profile != "selected222" else None),
        "experiment_recovered_features": (exp_selector.get("experiment_recovered_features")
            if feature_recovery_profile != "selected222" else []),
        "experiment_recovered_role": ("Weak" if feature_recovery_profile != "selected222" else None),
        "experiment_strong_count": (exp_selector.get("experiment_strong_count")
            if feature_recovery_profile != "selected222" else None),
        "experiment_weak_count": (exp_selector.get("experiment_weak_count")
            if feature_recovery_profile != "selected222" else None),
        "direction_class_weight_mode": direction_class_weight_mode,
        "direction_class_weight_pos": float(dir_w_pos),
        "direction_class_weight_nonpositive": float(dir_w_nonpos),
        "direction_class_weight_spec": ("E4-B experiment-only BASE_TRAIN-derived BCE weighting; monitor/checkpoint metrics remain unweighted"
            if direction_class_weight_mode != "unweighted" else "canonical unweighted BCE"),
        "stage_a_split": split_audit,
        "stage_a_split_spec": (
            "E3-A experiment-only recency split: newest monitor_days eligible days are MONITOR; "
            "BASE is the newest history_window_days days strictly before MONITOR (None=all earlier days)"
            if stage_a_monitor_days is not None else
            "canonical chronological 80/20 percentage split; monitor = newest 20% eligible days"),
        "direction_horizon_gate_mode_spec": ("E2-C2 Direction-only horizon gate; Magnitude retains canonical H_current"
            if direction_horizon_gate_mode != "current" else "canonical 24-h learnable Strong/Weak gate"),
        "direction_tabular_mode_spec": ("E2-C1 experiment-only Strong/Weak tabular source for Direction; Magnitude remains canonical H_current"
            if direction_tabular_mode != "current" else "canonical gated Strong+Weak H_current"),
        "direction_fusion_alpha_canonical_spec": ("E2-B1 experiment-only fixed Direction fusion alpha_time; "
            "magnitude fusion and a_mag untouched" if direction_fusion_alpha is not None else
            "canonical learnable alpha_time (default)"),
        "checkpoint_policy": checkpoint_policy,
        "direction_bce_min_delta": direction_bce_min_delta,
        "magnitude_l1_min_delta": magnitude_l1_min_delta,
        "gradient_policy": gradient_policy,
        "gradient_diagnostics": {"enabled": gradient_diagnostics,
            "batches_per_epoch": gradient_batches_per_epoch if gradient_diagnostics else 0,
            "parameter_groups": ["tabular_encoder", "temporal_encoder"],
            "summary_by_epoch": gradient_summary_by_epoch,
            "output_file": gradient_conflict_file},
        "checkpoint_rule": (
            "direction_first: maximize Raw, exact Raw tie -> minimize monitor L_dir -> Magnitude MAE -> earlier epoch; no Raw tolerance"
            if checkpoint_policy == "direction_first" else
            "magnitude_first: minimize Magnitude MAE, exact tie -> minimize monitor L_mag -> earlier epoch"
            if checkpoint_policy == "magnitude_first" else
            "maintain raw_anchor=max Raw; require selected Raw >= raw_anchor-tolerance; within guardrail "
            "minimize Magnitude MAE, then L_total, then epoch"),
        "checkpoint_raw_tolerance":cfg.checkpoint_raw_tolerance,
        "checkpoint_raw_anchor":raw_anchor,
        "checkpoint_selected_raw":(best_checkpoint_metrics.get("raw_direction_accuracy") if best_checkpoint_metrics and "raw_direction_accuracy" in best_checkpoint_metrics else None),
        "checkpoint_guardrail_status": guardrail_status,
        "checkpoint_warnings": sorted(diagnostic_warning_union),
        "magnitude_naive_baseline": ({"A_naive": magnitude_naive,
            "monitor_MAE_naive": (monitor_metrics.get("magnitude_mae_naive") if monitor_metrics else None),
            "target_MAE_naive": metrics.get("magnitude_mae_naive"),
            "target_magnitude_skill": metrics.get("magnitude_skill"), "hard_constraint": False} if is_experiment else None),
        "parameter_group_audit": protected_audit,
        "parameter_group_audit_file": parameter_group_audit_file,
        "protected_gradient_amp": False if gradient_policy == "direction_protected" else None,
        "protected_gradient_amp_reason": protected_amp_reason,
        "protected_gradient": ({"name": "direction_protected_asymmetric",
            "inspired_by": ["PCGrad", "AMTL"], "amp": amp,
            "reason": protected_amp_reason,
            "direction_owned_temporal_magnitude_detach": True,
            "direction_gradient_projection": "NEVER",
            "magnitude_projection": "shared-conflict-component-only",
            "gradient_clipping": "groupwise_independent_direction_owned_magnitude_owned_shared",
            "projection_output_file": protected_projection_file,
            "projection_summary_by_epoch": protected_projection_summary_by_epoch} if gradient_policy == "direction_protected" else None),
        "legacy_v20_dual_severity_readout_used":legacy_v20,
        "stage_b_freeze_audit": freeze_audit, "parameter_count": parameter_count,
        "official_tabm_member_shape": [1, cfg.k, 24, cfg.d_tab],
        "tabm_topology": model.tabular_encoder.parameter_topology(),
        "temporal_conditioning": model.temporal_encoder.conditioning_audit(),
        "alpha_trajectories": [{"stage": r["stage"], "epoch": r["epoch"], "alpha_dir": r["alpha_dir"], "alpha_mag": r["alpha_mag"]} for r in history],
            "strong_weak_gate_trajectories": [{"stage": r["stage"], "epoch": r["epoch"], "mean_gate": r["strong_gate_mean"]} for r in history],
            "direction_global_gate_trajectories": [{"stage": r["stage"], "epoch": r["epoch"], "gate": r["direction_global_gate"]}
                for r in history if r.get("direction_global_gate") is not None],
        "package_versions": package_versions(), "device": device, "amp": amp,
        **telemetry,
        "checkpoint_reload": "PASS", "prediction_hours": len(pred_frame),
        "metrics_file": "metrics.json", "predictions_file": "predictions.parquet",
    }
    _write_json(run_dir / "manifest.json", manifest)
    manifest["manifest_sha256"] = sha256_file(run_dir / "manifest.json")
    return {"run_dir": str(run_dir), "run_id": run_id, "manifest": manifest,
            "metrics": metrics, "predictions": pred_frame, "training_history": pd.DataFrame(history),
            "model": model, "preprocessor_state": state}

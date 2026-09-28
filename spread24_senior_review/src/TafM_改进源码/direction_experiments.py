"""Experiment-only Direction-first policies; never imported by the V2.1 default path."""
from __future__ import annotations

from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn


OBJECTIVE_MODES = {"joint_v21", "dir_only", "mag_only"}
CHECKPOINT_POLICIES = {"v21_guardrail", "direction_first", "magnitude_first"}
GRADIENT_POLICIES = {"vanilla", "direction_protected"}


def experiment_output_category(objective_mode: str, checkpoint_policy: str, gradient_policy: str,
                               gradient_diagnostics: bool) -> str:
    if objective_mode not in OBJECTIVE_MODES or checkpoint_policy not in CHECKPOINT_POLICIES or gradient_policy not in GRADIENT_POLICIES:
        raise ValueError("invalid experiment output policy")
    if gradient_policy == "direction_protected":
        return "protected_gradient"
    if gradient_diagnostics:
        return "gradient_diagnostics"
    if objective_mode != "joint_v21":
        return "objective_modes"
    if checkpoint_policy != "v21_guardrail":
        return "checkpoint_policy"
    return "smoke"


def objective_tensor(losses: Mapping[str, torch.Tensor], mode: str, *, lambda_dir: float = 1.0,
                     lambda_mag: float = 1.0) -> torch.Tensor:
    if mode not in OBJECTIVE_MODES:
        raise ValueError(f"unknown objective_mode: {mode}")
    if mode == "dir_only":
        return losses["L_dir"]
    if mode == "mag_only":
        return losses["L_mag"]
    # Keep the canonical V2.1 operation and its arithmetic order unchanged.
    if lambda_dir == 1.0 and lambda_mag == 1.0 and "L_total" in losses:
        return losses["L_total"]
    return lambda_dir * losses["L_dir"] + lambda_mag * losses["L_mag"]


def direction_first_checkpoint_is_better(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    """Strict lexicographic Raw -> monitor BCE -> Magnitude MAE -> earlier epoch."""
    if best is None:
        return True
    c = (float(candidate["raw_direction_accuracy"]), -float(candidate["L_dir"]),
         -float(candidate["magnitude_mae"]), -int(candidate["epoch"]))
    b = (float(best["raw_direction_accuracy"]), -float(best["L_dir"]),
         -float(best["magnitude_mae"]), -int(best["epoch"]))
    return c > b


def update_direction_first_checkpoint(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> dict[str, Any]:
    improved = direction_first_checkpoint_is_better(candidate, best)
    return {"selected": dict(candidate) if improved else best, "selected_improved": improved}


def update_direction_first_progress(candidate: Mapping[str, Any], raw_anchor: float | None,
                                     best_raw_bce: float | None, *, min_delta: float = 1e-4) -> dict[str, Any]:
    if min_delta < 0:
        raise ValueError("direction_bce_min_delta must be non-negative")
    raw = float(candidate["raw_direction_accuracy"])
    bce = float(candidate["L_dir"])
    raw_improved = raw_anchor is None or raw > float(raw_anchor)
    if raw_improved:
        return {"raw_anchor": raw, "best_raw_bce": bce, "progress": True,
                "raw_improved": True, "bce_improved_at_best_raw": False}
    if raw == raw_anchor and (best_raw_bce is None or bce <= float(best_raw_bce) - min_delta):
        return {"raw_anchor": raw_anchor, "best_raw_bce": bce, "progress": True,
                "raw_improved": False, "bce_improved_at_best_raw": True}
    return {"raw_anchor": raw_anchor, "best_raw_bce": best_raw_bce, "progress": False,
            "raw_improved": False, "bce_improved_at_best_raw": False}


def magnitude_first_checkpoint_is_better(candidate: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    """Strict lexicographic Magnitude MAE -> monitor L_mag -> earlier epoch."""
    if best is None:
        return True
    c = (-float(candidate["magnitude_mae"]), -float(candidate["L_mag"]), -int(candidate["epoch"]))
    b = (-float(best["magnitude_mae"]), -float(best["L_mag"]), -int(best["epoch"]))
    return c > b


def update_magnitude_first_progress(candidate: Mapping[str, Any], best_mae: float | None,
                                    best_l1_at_mae: float | None, *, min_delta: float = 1e-4) -> dict[str, Any]:
    if min_delta < 0:
        raise ValueError("magnitude_l1_min_delta must be non-negative")
    mae = float(candidate["magnitude_mae"])
    l1 = float(candidate["L_mag"])
    mae_improved = best_mae is None or mae < float(best_mae)
    if mae_improved:
        return {"best_mae": mae, "best_l1_at_mae": l1, "progress": True,
                "mae_improved": True, "l1_improved_at_best_mae": False}
    if mae == best_mae and (best_l1_at_mae is None or l1 <= float(best_l1_at_mae) - min_delta):
        return {"best_mae": best_mae, "best_l1_at_mae": l1, "progress": True,
                "mae_improved": False, "l1_improved_at_best_mae": True}
    return {"best_mae": best_mae, "best_l1_at_mae": best_l1_at_mae, "progress": False,
            "mae_improved": False, "l1_improved_at_best_mae": False}


def _flatten_grads(grads: Sequence[torch.Tensor | None], params: Sequence[nn.Parameter]) -> torch.Tensor | None:
    if not any(g is not None for g in grads):
        return None
    pieces = [torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1)
              for p, g in zip(params, grads)]
    return torch.cat(pieces)


def _grad_for(loss: torch.Tensor, params: Sequence[nn.Parameter], *, retain_graph: bool) -> tuple[torch.Tensor | None, ...]:
    if not params or not loss.requires_grad:
        return tuple(None for _ in params)
    return torch.autograd.grad(loss, tuple(params), retain_graph=retain_graph,
                               create_graph=False, allow_unused=True)


def gradient_diagnostic_rows(loss_dir: torch.Tensor, loss_mag: torch.Tensor,
                             parameter_groups: Mapping[str, Iterable[nn.Parameter]], *,
                             epoch: int, batch_index: int) -> list[dict[str, Any]]:
    """Read-only gradient diagnostics; autograd.grad never touches Parameter.grad."""
    rows = []
    groups = {name: tuple(p for p in params if p.requires_grad) for name, params in parameter_groups.items()}
    for group_name, params in groups.items():
        gd = _flatten_grads(_grad_for(loss_dir, params, retain_graph=True), params)
        gm = _flatten_grads(_grad_for(loss_mag, params, retain_graph=True), params)
        if gd is None or gm is None:
            rows.append({"epoch": epoch, "batch_index": batch_index, "parameter_group": group_name,
                         "dot_product": "N/A", "cosine_similarity": "N/A", "dir_grad_norm": "N/A",
                         "mag_grad_norm": "N/A", "norm_ratio": "N/A", "conflict": "N/A",
                         "status": "N/A_NO_GRADIENT"})
            continue
        dot = torch.dot(gd, gm)
        nd, nm = torch.linalg.vector_norm(gd), torch.linalg.vector_norm(gm)
        if float(nd) == 0.0 or float(nm) == 0.0:
            cosine: float | str = "N/A"
            conflict: bool | str = "N/A"
            status = "N/A_ZERO_NORM"
        else:
            cosine = float((dot / (nd * nm)).cpu())
            conflict = cosine < 0.0
            status = "PASS"
        ratio: float | str = float((nm / nd).cpu()) if float(nd) > 0.0 else "N/A"
        rows.append({"epoch": epoch, "batch_index": batch_index, "parameter_group": group_name,
                     "dot_product": float(dot.cpu()), "cosine_similarity": cosine,
                     "dir_grad_norm": float(nd.cpu()), "mag_grad_norm": float(nm.cpu()),
                     "norm_ratio": ratio, "conflict": conflict, "status": status})
    return rows


def summarize_gradient_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name in sorted({str(r["parameter_group"]) for r in rows}):
        group = [r for r in rows if r["parameter_group"] == name]
        def valid(column: str) -> list[float]:
            return [float(r[column]) for r in group if isinstance(r.get(column), (int, float))]
        cosine, dir_norm, mag_norm, ratios = (valid(c) for c in
            ("cosine_similarity", "dir_grad_norm", "mag_grad_norm", "norm_ratio"))
        conflicts = [bool(r["conflict"]) for r in group if isinstance(r.get("conflict"), bool)]
        summary[name] = {
            "mean_cosine": sum(cosine) / len(cosine) if cosine else "N/A",
            "median_cosine": float(median(cosine)) if cosine else "N/A",
            "conflict_rate": sum(conflicts) / len(conflicts) if conflicts else "N/A",
            "mean_dir_grad_norm": sum(dir_norm) / len(dir_norm) if dir_norm else "N/A",
            "mean_mag_grad_norm": sum(mag_norm) / len(mag_norm) if mag_norm else "N/A",
            "median_norm_ratio": float(median(ratios)) if ratios else "N/A",
            "valid_batches": len(cosine), "batches": len(group),
        }
    return summary


def audit_protected_parameter_groups(model: nn.Module) -> tuple[dict[str, Any], dict[str, tuple[nn.Parameter, ...]]]:
    """Explicit module/object-identity ownership; fail closed on overlap/unknown params."""
    required = ("tabular_encoder", "temporal_encoder", "adapters", "direction_head", "magnitude_head")
    if any(not hasattr(model, attr) for attr in required):
        raise RuntimeError("protected ownership audit requires canonical V2.1 module topology")
    adapters = model.adapters
    module_lists = {
        "direction_owned": (model.temporal_encoder, adapters.time_dir, adapters.tab_dir, model.direction_head),
        "magnitude_owned": (adapters.time_mag, adapters.tab_mag, model.magnitude_head),
        "shared": (model.tabular_encoder,),
    }
    groups: dict[str, tuple[nn.Parameter, ...]] = {}
    for key, modules in module_lists.items():
        params: dict[int, nn.Parameter] = {}
        for module in modules:
            for p in module.parameters():
                if p.requires_grad:
                    params[id(p)] = p
        scalar_name = "a_dir" if key == "direction_owned" else "a_mag" if key == "magnitude_owned" else None
        scalar = getattr(adapters, scalar_name, None) if scalar_name else None
        if isinstance(scalar, nn.Parameter) and scalar.requires_grad:
            params[id(scalar)] = scalar
        groups[key] = tuple(params.values())
    identities = {key: {id(p) for p in params} for key, params in groups.items()}
    overlap = sorted((a, b) for i, a in enumerate(identities) for b in list(identities)[i+1:]
                     if identities[a] & identities[b])
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    owned = set().union(*identities.values())
    unknown = trainable - owned
    audit = {
        "status": "PASS" if not overlap and not unknown and owned == trainable else "FAIL",
        "ownership_method": "explicit module ownership + Python parameter object id; no name matching",
        "groups": {name: {"parameter_tensors": len(params), "parameter_count": sum(p.numel() for p in groups[name])}
                   for name, params in groups.items()},
        "overlap": overlap, "unknown_trainable_parameter_tensors": len(unknown),
        "all_trainable_parameters_owned": owned == trainable,
    }
    if audit["status"] != "PASS":
        raise RuntimeError(f"protected parameter ownership failed closed: {audit}")
    return audit, groups


def project_magnitude_gradient(g_dir: torch.Tensor, g_mag: torch.Tensor,
                               *, eps: float = 1e-12) -> tuple[torch.Tensor, dict[str, Any]]:
    if g_dir.shape != g_mag.shape:
        raise ValueError("Direction/Magnitude gradient vectors must align")
    flat_dir = g_dir.reshape(-1)
    flat_mag = g_mag.reshape(-1)
    dot = torch.dot(flat_dir, flat_mag)
    dir_sq = torch.dot(flat_dir, flat_dir)
    dir_norm = torch.linalg.vector_norm(flat_dir)
    mag_norm = torch.linalg.vector_norm(flat_mag)
    if float(dot) < 0.0 and float(dir_sq) > 0.0:
        projected = g_mag - (dot / (dir_sq + eps)) * g_dir
        status = "PROJECTED_CONFLICT"
    else:
        projected = g_mag.clone()
        status = "UNCHANGED_ALIGNED_OR_ZERO"
    flat_safe = projected.reshape(-1)
    safe_norm = torch.linalg.vector_norm(flat_safe)
    post_dot = torch.dot(flat_dir, flat_safe)
    removed_norm = torch.linalg.vector_norm(flat_mag - flat_safe)
    return projected, {
        "dot_product": float(dot.detach().cpu()),
        "post_projection_dot_product": float(post_dot.detach().cpu()),
        "status": status,
        "projected": status == "PROJECTED_CONFLICT",
        "direction_gradient_unchanged": True,
        "dir_grad_norm": float(dir_norm.detach().cpu()),
        "mag_grad_norm": float(mag_norm.detach().cpu()),
        "safe_mag_grad_norm": float(safe_norm.detach().cpu()),
        "removed_grad_norm": float(removed_norm.detach().cpu()),
        "removed_fraction": (float((removed_norm / mag_norm).detach().cpu()) if float(mag_norm) > 0.0 else 0.0),
    }


def protected_gradients(loss_dir: torch.Tensor, loss_mag: torch.Tensor,
                        groups: Mapping[str, Sequence[nn.Parameter]], *,
                        lambda_dir: float = 1.0, lambda_mag: float = 1.0,
                        eps: float = 1e-12) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build final grads before optimizer.step; only conflict components of shared g_mag are projected."""
    group_names = ("direction_owned", "magnitude_owned", "shared")
    params_by_group = {name: tuple(groups[name]) for name in group_names}
    all_params = tuple(p for name in group_names for p in params_by_group[name])
    if len({id(p) for p in all_params}) != len(all_params):
        raise RuntimeError("parameter groups overlap; refusing protected update")
    d_all = _grad_for(loss_dir, all_params, retain_graph=True)
    m_all = _grad_for(loss_mag, all_params, retain_graph=False)
    dmap = {id(p): g for p, g in zip(all_params, d_all)}
    mmap = {id(p): g for p, g in zip(all_params, m_all)}
    proj_rows = []
    for group_name in group_names:
        params = params_by_group[group_name]
        if group_name == "shared":
            active = [(p, dmap[id(p)], mmap[id(p)]) for p in params
                      if dmap[id(p)] is not None or mmap[id(p)] is not None]
            if active:
                d_vec = torch.cat([(torch.zeros_like(p) if gd is None else gd).reshape(-1) for p, gd, _ in active])
                m_vec = torch.cat([(torch.zeros_like(p) if gm is None else gm).reshape(-1) for p, _, gm in active])
                safe_vec, projection = project_magnitude_gradient(d_vec, m_vec, eps=eps)
                proj_rows.append(projection)
                offset = 0
                for p, gd, gm in active:
                    n = p.numel(); safe_gm = safe_vec[offset:offset+n].view_as(p); offset += n
                    gd_safe = torch.zeros_like(p) if gd is None else gd
                    p.grad = lambda_dir * gd_safe + lambda_mag * safe_gm
            else:
                for p in params: p.grad = None
        else:
            choose = dmap if group_name == "direction_owned" else mmap
            weight = lambda_dir if group_name == "direction_owned" else lambda_mag
            for p in params:
                g = choose[id(p)]
                p.grad = None if g is None else weight * g
    return {"status": "PASS", "projection_rows": proj_rows,
            "projection_name": "direction_protected_asymmetric", "amp": False}, {
                "direction_gradient_projection": "NEVER", "magnitude_gradient_projection": "SHARED_CONFLICT_COMPONENT_ONLY"}


def clip_protected_gradient_groups(groups: Mapping[str, Sequence[nn.Parameter]], *,
                                   max_norm: float = 1.0) -> dict[str, float]:
    """Clip protected ownership groups independently so Magnitude cannot rescale Direction-owned updates."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norms: dict[str, float] = {}
    for name in ("direction_owned", "magnitude_owned", "shared"):
        params = tuple(p for p in groups[name] if p.requires_grad and p.grad is not None)
        if not params:
            norms[name] = 0.0
            continue
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if not torch.isfinite(torch.as_tensor(norm)):
            raise FloatingPointError(f"non-finite protected gradient norm for {name}")
        norms[name] = float(torch.as_tensor(norm).detach().cpu())
    return norms


def summarize_projection_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"batches": 0, "projection_rate": "N/A", "mean_pre_dot": "N/A",
                "mean_post_dot": "N/A", "mean_removed_fraction": "N/A"}
    projected = [bool(row["projected"]) for row in rows]
    return {
        "batches": len(rows),
        "projection_rate": sum(projected) / len(projected),
        "mean_pre_dot": sum(float(row["dot_product"]) for row in rows) / len(rows),
        "mean_post_dot": sum(float(row["post_projection_dot_product"]) for row in rows) / len(rows),
        "mean_removed_fraction": sum(float(row["removed_fraction"]) for row in rows) / len(rows),
    }


def checkpoint_diagnostic_warnings(metrics: Mapping[str, Any]) -> list[str]:
    """Surface degenerate/collapsed class behavior without inventing a recall threshold."""
    warnings = []
    if metrics.get("positive_recall") is None or metrics.get("nonpositive_recall") is None:
        warnings.append("one_class_truth_or_one_class_collapse")
    if metrics.get("positive_recall") == 0.0:
        warnings.append("positive_recall_zero")
    if metrics.get("nonpositive_recall") == 0.0:
        warnings.append("nonpositive_recall_zero")
    if metrics.get("predicted_positive_fraction") in {0.0, 1.0}:
        warnings.append("one_class_prediction_collapse")
    return sorted(set(warnings))

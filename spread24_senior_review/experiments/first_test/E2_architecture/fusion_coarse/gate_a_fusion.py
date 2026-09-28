"""E2-B1 Gate A — endpoint reproduction, fixed-alpha immobility, gradient ownership.

Three independent proofs, all on the benchmark day 2026-02-13:

1. fixed alpha=0.0 must reproduce the E2-A `tabular_only` run, and fixed alpha=1.0 the
   `temporal_only` run. Compared as artifacts: `predictions.parquet` SHA256 plus a numeric
   element-wise max |delta| over every shared column, and every scalar in `metrics.json`.
2. a fixed alpha must not move: its `alpha_trajectories` must be constant across every epoch.
3. the branch gradients must follow the coefficient endpoints exactly, measured on the *trained*
   checkpoints with the real target-day tensors rather than on a toy adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.TafM_改进源码.config import default_selector_path, load_v21_config  # noqa: E402
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector  # noqa: E402
from src.TafM_改进源码.losses import v21_loss  # noqa: E402
from src.TafM_改进源码.preprocessing import fit_preprocessor  # noqa: E402
from src.TafM_改进源码.train import _model_for, _selected_arrays  # noqa: E402

TARGET_DAY = "2026-02-13"
# Gate A endpoint runs (fresh, this round) paired with the E2-A run they must reproduce.
ENDPOINTS = {
    "F00": ("tabular_only", 0.0),
    "F100": ("temporal_only", 1.0),
}
E2A_ROOT = REPO / "src/TafM_改进源码/outputs/tabm_v21/experiments/direction_first/objective_modes"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_dir(record_path: Path) -> Path:
    return Path(record_path.read_text(encoding="utf-8").strip())


def compare_runs(fixed_dir: Path, mode_dir: Path) -> dict:
    fixed_pred = pd.read_parquet(fixed_dir / "predictions.parquet")
    mode_pred = pd.read_parquet(mode_dir / "predictions.parquet")
    shared = [c for c in fixed_pred.columns if c in mode_pred.columns]
    numeric = [c for c in shared if pd.api.types.is_numeric_dtype(fixed_pred[c])]
    max_delta = float(max(
        (fixed_pred[c].to_numpy(dtype=np.float64) - mode_pred[c].to_numpy(dtype=np.float64)).__abs__().max()
        for c in numeric))
    fixed_metrics = json.loads((fixed_dir / "metrics.json").read_text(encoding="utf-8"))
    mode_metrics = json.loads((mode_dir / "metrics.json").read_text(encoding="utf-8"))
    scalars = [k for k, v in fixed_metrics.items()
               if isinstance(v, (int, float)) and isinstance(mode_metrics.get(k), (int, float))]
    metric_delta = max(abs(float(fixed_metrics[k]) - float(mode_metrics[k])) for k in scalars)
    return {
        "predictions_parquet_sha256_match":
            sha256(fixed_dir / "predictions.parquet") == sha256(mode_dir / "predictions.parquet"),
        "n_rows": int(len(fixed_pred)),
        "n_numeric_columns_compared": len(numeric),
        "max_abs_delta_predictions": max_delta,
        "n_metric_scalars_compared": len(scalars),
        "max_abs_delta_metrics": metric_delta,
        "within_1e-9": max_delta <= 1e-9 and metric_delta <= 1e-9,
    }


def alpha_is_immobile(run: Path, alpha: float) -> dict:
    """A fixed coefficient must not update. `alpha_trajectories[].alpha_dir` logs the model's
    learnable `a_dir` (sigmoid-applied); under a fixed alpha that parameter is inert, so the proof
    is that it is bit-constant across every epoch. Its constant value is the initialisation, NOT
    the fixed coefficient — the authoritative record of the fixed value is the manifest's
    `direction_fusion_alpha` field, which is reported alongside rather than conflated."""
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    trajectory = manifest.get("alpha_trajectories") or []
    values = [float(h["alpha_dir"]) for h in trajectory if h.get("alpha_dir") is not None]
    recorded = manifest.get("direction_fusion_alpha")
    return {
        "manifest_direction_fusion_alpha": recorded,
        "manifest_alpha_matches_arm": recorded is not None and abs(float(recorded) - alpha) <= 1e-12,
        "n_epoch_points": len(values),
        "distinct_alpha_dir_values": len(set(values)),
        "inert_alpha_dir_constant": values[0] if values else None,
        "immobile": bool(values) and len(set(values)) == 1,
    }


def _grads(loss, params):
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    return sum(1 for g in grads if g is not None), len(grads)


def gradient_ownership(rows: list[dict]) -> None:
    cfg = replace(load_v21_config(), mode="A2")
    store = SequenceStore.load()
    selector, selector_hash = load_frozen_selector(default_selector_path())
    target = pd.Timestamp(TARGET_DAY).date()
    eligibility = store.eligibility(selector, current_target_day=target)
    train_indices = np.asarray(eligibility["eligible_indices"], dtype=np.int64)
    split = max(2, min(len(train_indices) - 1, int(math.floor(0.8 * len(train_indices)))))
    state = fit_preprocessor(store, train_indices[:split], selector, selector_sha256=selector_hash,
        n_bins=cfg.ple_bins, ple_embedding_dim=cfg.ple_embedding_dim, future_clip_abs=cfg.future_clip_abs,
        temporal_clip_abs=cfg.temporal_clip_abs, clip_after_robust_scale=cfg.clip_after_robust_scale,
        ple_enabled=cfg.ple_enabled)
    target_index = int(np.flatnonzero(np.asarray([d == target for d in store.days]))[0])
    xh, xf, _, y_scaled = _selected_arrays(store, [target_index], state)
    xh_t, xf_t, y_t = torch.from_numpy(xh), torch.from_numpy(xf), torch.from_numpy(y_scaled)

    for row in rows:
        model = _model_for(store, selector, state, cfg,
                           direction_fusion_alpha=row["direction_fusion_alpha"])
        payload = torch.load(row["run_dir"] / "stage_a_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        losses = v21_loss(model(xh_t, xf_t), y_t)
        tab = tuple(model.adapters.tab_dir.parameters())
        tim = tuple(model.adapters.time_dir.parameters())
        tab_live, tab_n = _grads(losses["L_dir"], tab)
        tim_live, tim_n = _grads(losses["L_dir"], tim)
        a_dir_live, a_dir_n = _grads(losses["L_dir"], (model.adapters.a_dir,))
        row["gradient_ownership"] = {
            "L_dir_tab_dir_live": f"{tab_live}/{tab_n}",
            "L_dir_time_dir_live": f"{tim_live}/{tim_n}",
            "L_dir_a_dir_live": f"{a_dir_live}/{a_dir_n}",
        }
        row["alpha_receives_no_direction_gradient"] = a_dir_live == 0


def main():
    benchmark = json.loads((HERE / "runs/GATE_A_BENCHMARK_RUNS.json").read_text(encoding="utf-8"))
    rows = []
    checks = {}

    for arm, (mode, alpha) in ENDPOINTS.items():
        record = next(r for r in benchmark if r["arm"] == arm)
        fixed_dir = Path(record["run_dir"])
        mode_record = next(
            r for r in json.loads(
                (REPO / "experiments/first_test/E2_architecture/runs/GATE_C_RUNS.json")
                .read_text(encoding="utf-8"))
            if r["architecture_mode"] == mode and r["target_day"] == TARGET_DAY)
        mode_dir = Path(mode_record["run_dir"])
        row = {
            "arm": arm, "direction_fusion_alpha": alpha,
            "reproduces": f"E2-A {mode}", "run_dir": fixed_dir,
            **compare_runs(fixed_dir, mode_dir),
            **alpha_is_immobile(fixed_dir, alpha),
        }
        rows.append(row)
        checks[f"{arm}_fixed_{alpha}_reproduces_{mode}_within_1e-9"] = bool(row["within_1e-9"])
        checks[f"{arm}_fixed_{alpha}_alpha_receives_no_update"] = bool(row["immobile"])

    # The four interior arms must also show an immobile alpha, and must be labelled as their own arms.
    for arm in ("F20", "F40", "F60", "F80"):
        record = next(r for r in benchmark if r["arm"] == arm)
        immobile = alpha_is_immobile(Path(record["run_dir"]), float(record["direction_fusion_alpha"]))
        rows.append({"arm": arm, "direction_fusion_alpha": record["direction_fusion_alpha"],
                     "reproduces": "fresh fixed-alpha arm", **immobile})
        checks[f"{arm}_alpha_receives_no_update"] = bool(immobile["immobile"])

    gradient_ownership([r for r in rows if "run_dir" in r])
    for row in rows:
        own = row.get("gradient_ownership")
        if not own:
            continue
        alpha = row["direction_fusion_alpha"]
        tab_live = int(own["L_dir_tab_dir_live"].split("/")[0])
        tim_live = int(own["L_dir_time_dir_live"].split("/")[0])
        if alpha == 0.0:
            checks[f"{row['arm']}_alpha0_direction_blocked_from_temporal"] = tim_live == 0
            checks[f"{row['arm']}_alpha0_direction_reaches_tabular"] = tab_live > 0
        elif alpha == 1.0:
            checks[f"{row['arm']}_alpha1_direction_blocked_from_tabular"] = tab_live == 0
            checks[f"{row['arm']}_alpha1_direction_reaches_temporal"] = tim_live > 0
        else:
            checks[f"{row['arm']}_interior_direction_reaches_both"] = tab_live > 0 and tim_live > 0
        checks[f"{row['arm']}_alpha_receives_no_direction_gradient"] = bool(
            row["alpha_receives_no_direction_gradient"])

    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "benchmark/gate_a_fusion_evidence.csv", index=False)
    (HERE / "benchmark/gate_a_fusion_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")

    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nGATE A: {sum(checks.values())}/{len(checks)} PASS")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""E2-A Gate A runtime proof of Direction gradient ownership.

Gate A requires more than a unit test on a toy adapter: the *trained artifacts* must show that
the excluded branch cannot reach the Direction loss. This rebuilds each benchmark-day model from
its frozen config + preprocessor state, loads the trained checkpoint, runs the real target-day
tensors through it, and asks autograd directly.

Expected, per mode:
    full_current   L_dir reaches BOTH encoders
    tabular_only   L_dir -> temporal_encoder is None (and L_mag still reaches it, so the
                   disconnect is specific to Direction rather than a dead graph)
    temporal_only  L_dir -> tabular_encoder is None (same magnitude sanity check)
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.TafM_改进源码.config import default_selector_path, load_v21_config  # noqa: E402
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector  # noqa: E402
from src.TafM_改进源码.losses import v21_loss  # noqa: E402
from src.TafM_改进源码.preprocessing import fit_preprocessor  # noqa: E402
from src.TafM_改进源码.train import _model_for, _selected_arrays  # noqa: E402

HERE = Path(__file__).resolve().parent
TARGET_DAY = "2026-02-13"
RUNS = {
    "A0": ("full_current", "2026-02-13_dir_dirfirst_van_20260925T111526536685Z"),
    "A1": ("tabular_only", "2026-02-13_dir_dirfirst_van_tabonly_20260925T111557839758Z"),
    "A2": ("temporal_only", "2026-02-13_dir_dirfirst_van_timeonly_20260925T111616442706Z"),
}
RAW_ROOT = REPO / "src/TafM_改进源码/outputs/tabm_v21/experiments/direction_first/objective_modes"


def _grads(loss, params):
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
    live = [i for i, g in enumerate(grads) if g is not None]
    return {"n_params": len(grads), "n_with_grad": len(live),
            "abs_sum": float(sum(float(g.abs().sum()) for g in grads if g is not None))}


def main():
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
    xh, xf, y_model, y_scaled = _selected_arrays(store, [target_index], state)
    xh_t = torch.from_numpy(xh)
    xf_t = torch.from_numpy(xf)
    y_t = torch.from_numpy(y_scaled)

    rows = []
    for arch, (mode, leaf) in RUNS.items():
        run_dir = RAW_ROOT / leaf
        model = _model_for(store, selector, state, cfg, architecture_mode=mode)
        payload = torch.load(run_dir / "stage_a_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        out = model(xh_t, xf_t)
        losses = v21_loss(out, y_t)
        tab_params = tuple(model.tabular_encoder.parameters())
        time_params = tuple(model.temporal_encoder.parameters())
        row = {"arch": arch, "architecture_mode": mode, "checkpoint_epoch": payload["epoch"],
               "load_state_dict": "strict-OK", "model_arch_mode": model.architecture_mode,
               "L_dir": float(losses["L_dir"]), "L_mag": float(losses["L_mag"])}
        for label, params in (("tabular_encoder", tab_params), ("temporal_encoder", time_params)):
            row[f"L_dir->{label}"] = json.dumps(_grads(losses["L_dir"], params))
            row[f"L_mag->{label}"] = json.dumps(_grads(losses["L_mag"], params))
        rows.append(row)
        print(f"{arch} {mode:14s} L_dir={row['L_dir']:.6f} "
              f"L_dir->tab={json.loads(row['L_dir->tabular_encoder'])['n_with_grad']}/"
              f"{json.loads(row['L_dir->tabular_encoder'])['n_params']} "
              f"L_dir->time={json.loads(row['L_dir->temporal_encoder'])['n_with_grad']}/"
              f"{json.loads(row['L_dir->temporal_encoder'])['n_params']}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "benchmark/gradient_ownership_runtime.csv", index=False)

    def n_with(row, key):
        return json.loads(row[key])["n_with_grad"]

    checks = {
        "A1_tabular_only__L_dir_blocked_from_temporal":
            n_with(frame.loc[frame.arch == "A1"].iloc[0], "L_dir->temporal_encoder") == 0,
        "A1_tabular_only__L_mag_still_reaches_temporal":
            n_with(frame.loc[frame.arch == "A1"].iloc[0], "L_mag->temporal_encoder") > 0,
        "A1_tabular_only__L_dir_reaches_tabular":
            n_with(frame.loc[frame.arch == "A1"].iloc[0], "L_dir->tabular_encoder") > 0,
        "A2_temporal_only__L_dir_blocked_from_tabular":
            n_with(frame.loc[frame.arch == "A2"].iloc[0], "L_dir->tabular_encoder") == 0,
        "A2_temporal_only__L_mag_still_reaches_tabular":
            n_with(frame.loc[frame.arch == "A2"].iloc[0], "L_mag->tabular_encoder") > 0,
        "A2_temporal_only__L_dir_reaches_temporal":
            n_with(frame.loc[frame.arch == "A2"].iloc[0], "L_dir->temporal_encoder") > 0,
        "A0_full_current__L_dir_reaches_both":
            n_with(frame.loc[frame.arch == "A0"].iloc[0], "L_dir->tabular_encoder") > 0 and
            n_with(frame.loc[frame.arch == "A0"].iloc[0], "L_dir->temporal_encoder") > 0,
    }
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    (HERE / "benchmark/gradient_ownership_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

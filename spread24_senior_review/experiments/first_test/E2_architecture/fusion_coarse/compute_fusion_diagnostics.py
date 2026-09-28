"""E2-B1 effective fusion diagnostics — observation only, no gradients, no ranking.

`alpha` alone is not interpretable: a coefficient of 0.5 says nothing about how much of the
fused Direction vector actually comes from each branch, because the two branches can differ by
orders of magnitude in norm. Per the E2-B1 plan §8 this script records, on a deterministic
diagnostic batch (the target day's own inputs, exactly the sample the metrics score):

    norm_tab  = || (1 - alpha) * tab_dir(H_tab) ||
    norm_time = || alpha * time_dir(H_time) ||
    effective_time_norm_fraction = norm_time / (norm_tab + norm_time + eps)
    cosine(tab_component, time_component)

Each formal run is rebuilt from its own checkpoint and its own preprocessing state (refit per
target day with the same eligibility split training used), so the numbers describe the trained
model rather than a fresh initialisation.

The alpha actually in force is read from the model, not assumed from the arm name:
`direction_fusion_alpha` for a fixed arm, and `sigmoid(a_dir)` for the canonical learnable arm.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
E2A = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from src.TafM_改进源码.config import default_selector_path, load_v21_config  # noqa: E402
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector  # noqa: E402
from src.TafM_改进源码.preprocessing import fit_preprocessor  # noqa: E402
from src.TafM_改进源码.train import _model_for, _selected_arrays  # noqa: E402

EPS = 1e-12
E1_RUNS = REPO / "experiments/first_test/E1_mini/runs"   # E1-M2 carries the canonical learnable arm
E2A_RUNS = E2A / "runs"
B1_RUNS = HERE / "runs"


def collect_runs() -> list[dict]:
    jobs = []
    for path in sorted(E1_RUNS.glob("E1-M2-*/RUN_RECORD.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "PASS":
            jobs.append({"arm": "FL08", "target_day": record["target_day"], "window": record["window"],
                         "run_dir": Path(record["raw_run_dir"]), "fixed_alpha": None})
    for pattern, arm, mode in (("E2-A1-*", "F00", "tabular_only"), ("E2-A2-*", "F100", "temporal_only")):
        for path in sorted(E2A_RUNS.glob(f"{pattern}/RUN_RECORD.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "PASS":
                jobs.append({"arm": arm, "target_day": record["target_day"], "window": record["window"],
                             "run_dir": Path(record["run_dir"]), "fixed_alpha": 0.0 if mode == "tabular_only" else 1.0})
    for arm in ("F20", "F40", "F60", "F80"):
        for path in sorted(B1_RUNS.glob(f"E2B1-{arm}-*/RUN_RECORD.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "PASS":
                jobs.append({"arm": arm, "target_day": record["target_day"], "window": record["window"],
                             "run_dir": Path(record["run_dir"]),
                             "fixed_alpha": float(record["direction_fusion_alpha"])})
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fusion_diagnostics.csv")
    parser.add_argument("--arms", default=None, help="comma list, for smoke-testing only")
    parser.add_argument("--limit", type=int, default=None, help="max runs, for smoke-testing only")
    args = parser.parse_args()

    cfg = replace(load_v21_config(), mode="A2")
    store = SequenceStore.load()
    selector, selector_hash = load_frozen_selector(default_selector_path())
    day_to_index = {d: i for i, d in enumerate(store.days)}

    state_cache: dict[object, object] = {}

    def state_for(target):
        if target not in state_cache:
            eligibility = store.eligibility(selector, current_target_day=target)
            train_indices = np.asarray(eligibility["eligible_indices"], dtype=np.int64)
            split = max(2, min(len(train_indices) - 1, int(math.floor(0.8 * len(train_indices)))))
            state_cache[target] = fit_preprocessor(
                store, train_indices[:split], selector, selector_sha256=selector_hash,
                n_bins=cfg.ple_bins, ple_embedding_dim=cfg.ple_embedding_dim,
                future_clip_abs=cfg.future_clip_abs, temporal_clip_abs=cfg.temporal_clip_abs,
                clip_after_robust_scale=cfg.clip_after_robust_scale, ple_enabled=cfg.ple_enabled)
        return state_cache[target]

    jobs = collect_runs()
    if args.arms:
        wanted = {a.strip() for a in args.arms.split(",")}
        jobs = [j for j in jobs if j["arm"] in wanted]
    if args.limit:
        jobs = jobs[: args.limit]

    rows = []
    for job in jobs:
        target = pd.Timestamp(job["target_day"]).date()
        state = state_for(target)
        index = day_to_index[target]
        model = _model_for(store, selector, state, cfg, direction_fusion_alpha=job["fixed_alpha"])
        payload = torch.load(job["run_dir"] / "stage_a_best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()

        xh, xf, _, _ = _selected_arrays(store, [index], state)
        xh_t, xf_t = torch.from_numpy(xh), torch.from_numpy(xf)
        with torch.no_grad():
            out = model(xh_t, xf_t)
            h_tab, h_time = out["H_tab"], out["H_time"]
            sd = model.adapters.tab_dir(h_tab)                                    # (B, M, 24, d_task)
            td = model.adapters.time_dir(h_time).unsqueeze(1).expand_as(sd)
            if job["fixed_alpha"] is not None:
                alpha = float(job["fixed_alpha"])
                source = "fixed"
            else:
                alpha = float(torch.sigmoid(model.adapters.a_dir).detach())
                source = "learned (sigmoid(a_dir) at the reused checkpoint)"
            weight = alpha
            comp_tab = (1.0 - weight) * sd
            comp_time = weight * td
            norm_tab = float(torch.linalg.vector_norm(comp_tab, dim=(1, 2, 3)).mean())
            norm_time = float(torch.linalg.vector_norm(comp_time, dim=(1, 2, 3)).mean())
            flat_tab = comp_tab.reshape(comp_tab.shape[0], -1)
            flat_time = comp_time.reshape(comp_time.shape[0], -1)
            # At an endpoint one component is exactly the zero vector, so the angle between the
            # branches is undefined rather than zero. Record it as missing instead of as 0.
            if norm_tab <= EPS or norm_time <= EPS:
                cosine = float("nan")
            else:
                cosine = float(torch.nn.functional.cosine_similarity(
                    flat_tab, flat_time, dim=1, eps=1e-8).mean())
            h_dir = weight * td + (1.0 - weight) * sd
            norm_dir = float(torch.linalg.vector_norm(h_dir, dim=(1, 2, 3)).mean())

        rows.append({
            "arm": job["arm"], "window": job["window"], "target_day": job["target_day"],
            "alpha_used": alpha, "alpha_source": source,
            "n_diagnostic_samples": int(h_tab.shape[0]),
            "norm_tab_component": norm_tab, "norm_time_component": norm_time,
            "effective_time_norm_fraction": norm_time / (norm_tab + norm_time + EPS),
            "cosine_tab_time": cosine, "norm_h_dir": norm_dir,
        })
        print(f"{job['arm']:5} {job['target_day']} alpha={alpha:.6f} "
              f"frac_time={rows[-1]['effective_time_norm_fraction']:.6f} cos={cosine:+.6f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / args.out, index=False)

    summary = (frame.groupby("arm")
               .agg(n_runs=("target_day", "size"), alpha_used=("alpha_used", "mean"),
                    norm_tab_component=("norm_tab_component", "mean"),
                    norm_time_component=("norm_time_component", "mean"),
                    effective_time_norm_fraction=("effective_time_norm_fraction", "mean"),
                    effective_time_norm_fraction_std=("effective_time_norm_fraction", "std"),
                    cosine_tab_time=("cosine_tab_time", "mean"),
                    alpha_used_min=("alpha_used", "min"), alpha_used_max=("alpha_used", "max"))
               .reset_index())
    summary.to_csv(HERE / args.out.replace(".csv", "_summary.csv"), index=False)
    print(f"\nwrote {args.out} ({len(frame)} rows) and its summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

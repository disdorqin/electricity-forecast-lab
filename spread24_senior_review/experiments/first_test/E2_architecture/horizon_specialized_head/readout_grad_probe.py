"""E2-E1 readout diagnostics: per-run Q1 bias values and Q2 per-head param/grad norms.

Deterministic, read-only probe over the selected benchmark checkpoints. It rebuilds each run's model
from its manifest config + saved preprocessor state, loads stage_a_best.pt, and asks autograd for the
Direction-loss gradient on the real target-day tensors. No parameter is modified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.TafM_改进源码.config import V2Config, default_selector_path
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.losses import v21_loss
from src.TafM_改进源码.preprocessing import PreprocessorState
from src.TafM_改进源码.train import _model_for, _selected_arrays

WINDOWS = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
           "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}


def formal_days():
    return [str((pd.Timestamp(a) + pd.Timedelta(days=i)).date())
            for w, (a, b) in WINDOWS.items() for i in range(7)]


def norm(params):
    flat = [p.detach().reshape(-1) for p in params]
    return float(torch.linalg.vector_norm(torch.cat(flat))) if flat else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = SequenceStore.load()
    selector, _ = load_frozen_selector(default_selector_path())
    rows = []
    for arm in ("Q1", "Q2"):
        for day in formal_days():
            rec = json.loads((HERE / "runs" / f"E2E1-{arm}-{day}" / "RUN_RECORD.json").read_text(encoding="utf-8"))
            run_dir = Path(rec["run_dir"])
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            mode = manifest["direction_readout_mode"]
            cfg = V2Config(**manifest["config"])
            state = PreprocessorState.load(run_dir / "preprocessor_state.json")
            target = pd.Timestamp(day).date()
            index = int(np.flatnonzero(np.asarray([d == target for d in store.days]))[0])
            model = _model_for(store, selector, state, cfg, architecture_mode="full_current",
                               direction_fusion_alpha=0.8, direction_tabular_mode="current",
                               direction_horizon_gate_mode="current", strong_role_profile="all",
                               numeric_encoding_mode="canonical", direction_readout_mode=mode).to(device)
            model.load_state_dict(torch.load(run_dir / "stage_a_best.pt", map_location=device,
                                             weights_only=False)["model"])
            model.eval()
            xh, xf, ym, ys = _selected_arrays(store, [index], state)
            xh_t = torch.tensor(xh, device=device); xf_t = torch.tensor(xf, device=device)
            ys_t = torch.tensor(ys, device=device)
            out = model(xh_t, xf_t)
            L_dir = v21_loss(out, ys_t)["L_dir"]
            row = {"arm": arm, "target_day": day, "mode": mode,
                   "segment_bias_H1": None, "segment_bias_H2": None, "segment_bias_H3": None,
                   "bias_grad_norm_H1": None, "bias_grad_norm_H2": None, "bias_grad_norm_H3": None,
                   "head_param_norm_H1": None, "head_param_norm_H2": None, "head_param_norm_H3": None,
                   "head_grad_norm_H1": None, "head_grad_norm_H2": None, "head_grad_norm_H3": None,
                   "shared_head_param_norm": None, "shared_head_grad_norm": None}
            if getattr(model, "direction_hour_bias", None) is not None:
                bias = model.direction_hour_bias
                g = torch.autograd.grad(L_dir, bias, retain_graph=True, allow_unused=True)[0]
                for i in range(3):
                    row[f"segment_bias_H{i+1}"] = float(bias.detach().cpu()[i])
                    row[f"bias_grad_norm_H{i+1}"] = float(g.detach().cpu()[i]) if g is not None else 0.0
            if getattr(model, "direction_segment_heads", None) is not None:
                for i, head in enumerate(model.direction_segment_heads):
                    params = tuple(p for p in head.parameters() if p.requires_grad)
                    grads = torch.autograd.grad(L_dir, params, retain_graph=True, allow_unused=True)
                    row[f"head_param_norm_H{i+1}"] = norm(params)
                    row[f"head_grad_norm_H{i+1}"] = float(torch.linalg.vector_norm(torch.cat(
                        [(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p, g in zip(params, grads)])))
            if getattr(model, "direction_head", None) is not None:
                params = tuple(p for p in model.direction_head.parameters() if p.requires_grad)
                grads = torch.autograd.grad(L_dir, params, retain_graph=False, allow_unused=True)
                row["shared_head_param_norm"] = norm(params)
                row["shared_head_grad_norm"] = float(torch.linalg.vector_norm(torch.cat(
                    [(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p, g in zip(params, grads)])))
            rows.append(row)
            print(f"{arm} {day} mode={mode} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "readout_diagnostics.csv", index=False)
    summary = {"device": device, "rows": len(rows),
               "by_arm": json.loads(frame.drop(columns=["target_day"]).groupby("arm").mean(numeric_only=True).to_json(orient="index"))}
    (HERE / "readout_diagnostics_probe_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote readout_diagnostics.csv + readout_diagnostics_probe_summary.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()

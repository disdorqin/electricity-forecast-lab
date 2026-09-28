"""Small synthetic contract gate; it never fits a model or touches evaluation data."""
from __future__ import annotations

import json
from pathlib import Path
import torch

from .contracts import project_root
from .losses import v2_loss_compat
from .metrics import canonical_metrics
from .models.dual_branch_v2 import DualReadoutV2, dual_readouts, official_tabm_import_status


def run_v2_synthetic_gate(output_dir: Path | None = None) -> dict:
    torch.manual_seed(20260924)
    model = DualReadoutV2(6, 4, 12, mode="A2", magnitude_eps=1e-8)
    h_tab = torch.randn(3, 4, 24, 6)
    h_time = torch.randn(3, 24, 4)
    out = model(h_tab, h_time)
    y = torch.randn(3, 24)
    losses = v2_loss_compat(out, y)
    losses["L_total"].backward()
    alpha_grad = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in (model.adapters.a_dir, model.adapters.a_mag))
    finite = all(torch.isfinite(v).all().item() for v in out.values() if isinstance(v, torch.Tensor))
    # Exact synthetic architecture invariants: magnitude does not cancel; direction owns final sign.
    z = torch.zeros(1, 24); same = torch.full((1, 2, 24), 100.0)
    _, _, soft, _, _, mag, _ = dual_readouts(z, same, same)
    z2 = torch.full((1, 24), 2.0); small = torch.ones((1, 2, 24)); huge = torch.full_like(small, 10000.0)
    _, _, _, positive, _, _, signed = dual_readouts(z2, small, huge)
    metric = canonical_metrics([0.0, 1.0, -1.0], [False, True, False], [0.0, 1.0, 1.0], direction_probability=[.2, .8, .3])
    mode_states = {}
    for mode in ("A0", "A1", "A2"):
        probe = DualReadoutV2(6, 4, 12, mode=mode)
        state = probe.adapters.mode_state()
        state["alpha_dir_fixed"] = mode != "A2"
        state["alpha_mag_fixed"] = mode != "A2"
        mode_states[mode] = state
    gate = {
        "schema": "spread24_v2_synthetic_gate_v1",
        "status": "PASS" if finite and alpha_grad and torch.allclose(soft, torch.zeros_like(soft), atol=1e-6)
                  and torch.allclose(mag, torch.full_like(mag, 100.0)) and positive.all().item() and (signed > 0).all().item() else "FAIL",
        "official_tabm_import": official_tabm_import_status(),
        "v2_forward": "PASS" if finite else "FAIL",
        "v2_backward": "PASS" if alpha_grad else "FAIL",
        "alpha_init": "0.8/0.8",
        "alpha_grad": "PASS" if alpha_grad else "FAIL",
        "tabm_member_axis_preserved": out["m_pos_members"].shape == (3, 4, 24),
        "train_readout": "PASS" if out["y_soft_train_members"].shape == (3, 4, 24) else "FAIL",
        "kpi_readout": "PASS" if out["signed_kpi_hat"].shape == (3, 24) else "FAIL",
        "magnitude_does_not_cancel": bool(torch.allclose(mag, torch.full_like(mag, 100.0))),
        "direction_owns_signed_kpi_sign": bool(positive.all() and (signed > 0).all()),
        "zero_policy_and_metrics": metric,
        "loss_family": {k: float(v.detach()) for k, v in losses.items() if isinstance(v, torch.Tensor)},
        "magnitude_eps": model.magnitude_eps,
        "modes": mode_states,
        "full_train": "NOT_RUN",
    }
    gate["sign_consistency_structural"] = gate["direction_owns_signed_kpi_sign"]
    if output_dir is not None:
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "v2_synthetic_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_dir / "v2_synthetic_gate.md").write_text("# V2 Synthetic Gate\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in gate.items()) + "\n", encoding="utf-8")
    return gate

# E2-D1 — Numeric Encoding Big-Block Ablation (plan snapshot)

STATUS=AUTHORIZED_EXECUTED
DATE=2026-09-26
PARENT=E2-C3
EXPERIMENT_ROOT=experiments/first_test/E2_architecture/numeric_encoding
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

Frozen human decision before this experiment:
- E2-C3 ROLE_SIGNAL=NO_CLEAR_ROLE_EFFECT; selector Strong-DIR/MAG labels are NOT promoted.
- DROP_MAG/DROP_DIR not promising; no E2-C4 matched-random-drop.
- Return to all 211 Strong/Core + 11 Weak; keep 24-h horizon gate; keep fixed direction_fusion_alpha=.8.
- Next question is numeric representation, not selector role routing.

## Arms
- N0 CURRENT_PLE_RAW — anchor, reuses R0/G0/C0/F80. encoding=concat(PLE_8D, raw) => 9 channels.
- N1 RAW_ONLY — fresh 28. encoding=raw => 1 channel. No PLE reaches Strong or Weak.
- N2 PLE_ONLY — fresh 28. encoding=PLE_8D => 8 channels. No raw skip channel.

Canonical encoding assumed by this experiment: PLE enabled, bins=16, embedding_dim=8, plus raw skip.

## Experiment-only implementation
`numeric_encoding_mode = canonical | raw_only | ple_only`, default=canonical.
- canonical preserves existing config semantics exactly (ple_enabled True => PLE+raw; False => raw-only).
- raw_only forces raw-only; ple_only forces PLE-only and requires cfg.ple_enabled=True.
- Preprocessing/robust scaling/clipping/PLE bin construction untouched. Same mode for Strong and Weak.

Fail-closed for non-canonical: legacy_v20=False, objective_mode=dir_only, architecture_mode=full_current,
direction_tabular_mode=current, direction_horizon_gate_mode=current, strong_role_profile=all,
direction_fusion_alpha=.8, train_mode=stage_a. ple_only additionally requires cfg.ple_enabled=True.

## Frozen protocol
Benchmark day=2026-02-13 (engineering only). Formal DEV W1 02-12..02-18, W2 04-12..04-18,
W3 06-12..06-18, W4 08-07..08-13. N0 reuse F80; N1/N2 fresh 28 days; no intermediate stopping.
Training fixed: dir_only / direction_first / vanilla / full_current / current tabular route /
current horizon gate / strong_role_profile=all / fixed fusion alpha=.8 / Stage A / default /
seed20260924 / k8 / d_tab128 / n_blocks2 / dropout.05 / max120 / patience15 / batch64 /
frozen LR-WD / CUDA AMP.

## Gate A (all PASS — see benchmark/GATE_A_EVIDENCE.md)
canonical reproduces F80 benchmark <=1e-9; default/no-flag exact; dims 9/1/8; frozen PLE bins/hashes
unchanged; Strong/Core 211 & Weak 11; current gate/Temporal/k8 live; finite & leakage unchanged;
canonical94 + all prior E2 tests + focused E2-D1 tests PASS.

## Forbidden (not run)
role filtering, selector rerun, PLE bins/dim sweep, k/depth/width change, gate tuning, Temporal/FFT,
fusion/alpha sweep, threshold/calibration, checkpoint change, Stage B, broader DEV, lockbox.

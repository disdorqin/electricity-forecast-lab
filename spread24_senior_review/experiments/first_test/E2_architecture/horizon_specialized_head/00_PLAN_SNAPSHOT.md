# E2-E1 — Horizon-Specialized Direction Readout (plan snapshot)

STATUS=AUTHORIZED_EXECUTED
DATE=2026-09-26
PARENT=E2-D1
EXPERIMENT_ROOT=experiments/first_test/E2_architecture/horizon_specialized_head
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS
ACCELERATION_GOAL=BREAK_60_AND_TEST_PATH_TO_62_65

## Why pivot here
E2-D1 Gate = NO_CLEAR_ENCODING_EFFECT; canonical PLE+raw stays the anchor
(Raw .5893 / Balanced .5372 / AUC .5671). PLE bins/dim, k, depth, width are not touched next.
The strongest unresolved asymmetry is the readout: one shared member-wise Direction head over all 24
horizons, yet Q0 is heterogeneous by segment:
H1 1-8 Raw .5000 / Bal .4714 / AUC .4919; H2 9-16 Raw .6607 / Bal .5545 / AUC .5751;
H3 17-24 Raw .6071 / Bal .5719 / AUC .6136. Segments 1-8 / 9-16 / 17-24 are pre-existing and frozen.

## Arms
- Q0 SHARED_HEAD — anchor, reuses N0/R0/G0/F80. One shared member-wise Direction head for all 24 hours.
- Q1 SEGMENT_BIAS — fresh 28. Same shared head weights + exactly 3 zero-init learnable scalar logit
  offsets (H1 1-8, H2 9-16, H3 17-24) applied to member logits before sigmoid; threshold stays .5.
- Q2 SEGMENT_HEADS — fresh 28. Three independent tabm.LinearEnsemble(d_task,1,k=k) Direction heads
  (H1/H2/H3), concatenated back to [B,k,24]; no 24-head design; no shared Direction head retained.

## Experiment-only implementation
`direction_readout_mode = shared | segment_bias | segment_heads`, default=shared. Isolated inside
DualBranchV21 readout; encoders/adapters/selector/preprocessing untouched; Magnitude head unchanged.
Fail-closed for non-shared: legacy_v20=False, objective_mode=dir_only, architecture_mode=full_current,
direction_tabular_mode=current, direction_horizon_gate_mode=current, strong_role_profile=all,
numeric_encoding_mode=canonical, direction_fusion_alpha=.8, train_mode=stage_a.

## Frozen protocol
Benchmark day=2026-02-13 (engineering only). Formal DEV W1 02-12..02-18, W2 04-12..04-18,
W3 06-12..06-18, W4 08-07..08-13. Q0 reuse F80; Q1/Q2 fresh 28 days; no intermediate stopping.
Training fixed: dir_only / direction_first / vanilla / full_current / current tabular / current gate /
all roles / canonical encoding / fixed alpha=.8 / Stage A / default / seed20260924 / k8 / d_task32 /
max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP.

## Gate A (all PASS — see benchmark/GATE_A_EVIDENCE.md)
shared + no-flag reproduce F80 <=1e-9; Q1 exactly 3 zero-init biases over one shared head;
Q2 exactly 3 independent heads with exclusive hour slicing and no shared head; k=8 & [B,k,24] preserved;
threshold/Magnitude unchanged; frozen hashes; baseline 94 + full 174 tests PASS.

## Forbidden (not run)
24 heads, new segment boundaries, month heads, oracle routing, threshold/calibration tuning,
PLE bins/dim, k/depth/width, role filtering, gate/fusion sweep, Temporal/FFT, Stage B, broader
DEV, lockbox.

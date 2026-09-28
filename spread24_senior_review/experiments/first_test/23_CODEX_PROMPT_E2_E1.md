# Codex Prompt — E2-E1 Horizon-Specialized Direction Readout

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

Only authorized task:
Execute E2-E1 Horizon-Specialized Direction Readout.

Read first:
1. experiments/first_test/22_E2_E1_HORIZON_SPECIALIZED_HEAD_PLAN.md
2. experiments/first_test/E2_architecture/numeric_encoding/E2_D1_summary.md
3. experiments/first_test/E2_architecture/numeric_encoding/E2_D1_GATE.md
4. experiments/first_test/E2_architecture/strong_role_routing/E2_C3_summary.md
5. docs/01_业务数据与防泄漏合同.md
6. docs/17_最终模型设计与编码规范.md
7. docs/21_正式运行、验收与实验入口规范.md

Follow plan 22 exactly.

Frozen decision:
- E2-D1 ENCODING_SIGNAL=NO_CLEAR_ENCODING_EFFECT.
- do not tune PLE bins/dim, k, depth or width next.
- use canonical PLE+raw, all 211 Strong/Core + 11 Weak, current 24-h gate, fixed fusion alpha=.8.
- current Direction head is one shared LinearEnsemble across all 24 hours.
- N0 segment metrics are highly heterogeneous:
  H1 Raw .5000 / Bal .4714 / AUC .4919
  H2 Raw .6607 / Bal .5545 / AUC .5751
  H3 Raw .6071 / Bal .5719 / AUC .6136
- fixed business segments are 1-8, 9-16, 17-24. Do not invent new segment boundaries.

Arms:

Q0 SHARED_HEAD:
reuse N0/F80.
Current single shared Direction head.

Q1 SEGMENT_BIAS:
same shared head weights plus exactly 3 learnable scalar logit biases:
H1 1-8, H2 9-16, H3 17-24.
Initialize all 0.
Apply to member logits before sigmoid.
Threshold remains .5.

Q2 SEGMENT_HEADS:
three independent tabm.LinearEnsemble(d_task,1,k=k) Direction heads:
head_H1 for 1-8
head_H2 for 9-16
head_H3 for 17-24.
Concatenate back to [B,k,24].
No 24-head design.

Implement experiment-only:
direction_readout_mode=shared/segment_bias/segment_heads
default=shared.

Only Direction readout changes.
Magnitude path/head unchanged.

Fail closed for non-shared:
legacy_v20=False
objective_mode=dir_only
architecture_mode=full_current
direction_tabular_mode=current
direction_horizon_gate_mode=current
strong_role_profile=all
numeric_encoding_mode=canonical
direction_fusion_alpha=.8
train_mode=stage_a.

Gate A:
- shared reproduces N0/F80 benchmark <=1e-9.
- default/no flag exact.
- Q1 exactly 3 biases, zero init, one shared head.
- Q2 exactly 3 independent heads, correct hour slicing.
- k8 and [B,k,24] preserved.
- threshold/readout semantics unchanged.
- Magnitude unchanged.
- hashes/leakage frozen.
- canonical94 + prior E2 + focused tests PASS.

Benchmark=2026-02-13 engineering only.

Formal panel same 28 days:
W1 Feb12-18
W2 Apr12-18
W3 Jun12-18
W4 Aug07-13.

Q0 reuse.
Q1 fresh28.
Q2 fresh28.
No intermediate stop.

Fresh training fixed:
dir_only/direction_first/vanilla/
full_current/current tabular/current horizon gate/
all roles/canonical encoding/fixed alpha=.8/
Stage A/default/seed20260924/k8/d_task32/
max120/patience15/batch64/frozen LR-WD/CUDA AMP.

Report:
overall Raw/Balanced/+R/-R/AUC/Brier/ppf/collapse.
W1-W4 + min/std.
H1-H3 with Raw/Balanced/AUC/+R/-R/Brier.
paired Q1-Q0, Q2-Q0, Q2-Q1 with day-bootstrap 95%CI + W/T/L.
slot gains vs Q0 overall and by H1/H2/H3.
params/runtime/GPU/best-stop.
Q1 bias trajectory.
Q2 per-head norm/gradient diagnostics.

Safety anchor Q0:
Balanced>=Q0-.01
+Recall>=Q0-.05
one-class<=Q0+2
+R=0<=Q0+2.

E2_E1_GATE READOUT_SIGNAL exactly one:
SEGMENT_CALIBRATION_HELPFUL
SEGMENT_BOUNDARY_HELPFUL
HETEROGENEITY_PROMISING
SHARED_HEAD_SUFFICIENT
SPECIALIZATION_HARMFUL
NO_CLEAR_READOUT_EFFECT

Acceleration rule:
- if Q1/Q2 >=.61 Raw with safety: next do 3-seed confirmation immediately.
- if confirmed >=.61 but <.62: move to E3 Stage A/B; no more head micro-tuning.
- if confirmed >=.62: freeze architecture candidate and move directly to E3/broader DEV.
- if no material gain: stop architecture micro-ablation and move to E3 training strategy + feature/regime work.

Forbidden:
24 heads, new segment boundaries, month heads, oracle routing,
threshold/calibration tuning, PLE bins/dim, k/depth/width, role filtering,
gate/fusion sweep, Temporal/FFT, Stage B, broader DEV, lockbox.

Output:
experiments/first_test/E2_architecture/horizon_specialized_head/
with 00_PLAN_SNAPSHOT.md, benchmark/, runs/, daily_metrics.csv,
model_summary.csv, window_metrics.csv, hour_segment_metrics.csv,
paired_summary.csv, slot_gain_summary.csv, readout_diagnostics.csv/json,
runtime.csv, E2_E1_summary.md, E2_E1_GATE.md, figures/.

After E2-E1 STOP.

Final report only:
1 FILES_CHANGED
2 TESTS
3 READOUT_ROUTING_AUDIT
4 BENCHMARK engineering
5 Q0/Q1/Q2 overall
6 W1-W4
7 H1-H3
8 paired Q1-Q0
9 paired Q2-Q0
10 slot gains
11 safety
12 readout diagnostics/params/runtime
13 E2_E1_GATE READOUT_SIGNAL
14 NEXT under acceleration rule
15 unexecuted later work

Do not stage/commit.

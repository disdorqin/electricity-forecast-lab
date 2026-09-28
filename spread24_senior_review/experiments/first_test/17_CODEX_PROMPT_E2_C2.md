# Codex Prompt — E2-C2 Horizon Gate Necessity & Simplification

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

Only authorized task:
Execute E2-C2 Horizon Gate Necessity & Simplification.

Read first:
1. experiments/first_test/16_E2_C2_HORIZON_GATE_PLAN.md
2. experiments/first_test/E2_architecture/tabular_big_block/E2_C1_summary.md
3. experiments/first_test/E2_architecture/tabular_big_block/E2_C1_GATE.md
4. experiments/first_test/E2_architecture/fusion_coarse/E2_B1_summary.md
5. docs/01_业务数据与防泄漏合同.md
6. docs/17_最终模型设计与编码规范.md
7. docs/21_正式运行、验收与实验入口规范.md

Do not redesign the experiment. Follow 16 exactly.

Frozen reading:
- formal E2-C1 gate=INCONCLUSIVE.
- C2 WEAK_ONLY is clearly worse than C0.
- C1 STRONG_ONLY trends below C0: ΔRaw -1.49pp, CI [-4.02,+0.15], W/T/L 2/21/5; 3/4 windows lower and safety Balanced fails.
- therefore Weak is a plausible residual, not a standalone predictor.
- current 24-h horizon gate is almost pinned near init .8: mean .80114, global range .78797-.80938.
- effective Weak norm fraction ~6.54%.
- next question is whether 24 separate learnable horizon parameters are necessary.

Use G0=C0/F80 as screening anchor:
fixed temporal fusion alpha=.8.
Magnitude always remains canonical H_current.

Implement experiment-only:
direction_horizon_gate_mode =
current / fixed_08 / global_learnable
default=current.

G0 CURRENT_24H_LEARNABLE:
reuse C0/F80.

G1 FIXED_08:
Direction h_tab = .8*H_strong + .2*H_weak_broadcast.
canonical horizon_gate_logits must not receive L_dir gradient.
No new Direction gate parameter.

G2 GLOBAL_LEARNABLE:
one scalar g=sigmoid(a_global), init .8, shared across 24 hours.
Direction h_tab = g*H_strong + (1-g)*H_weak_broadcast.
canonical horizon_gate_logits must not receive L_dir gradient.
a_global must receive L_dir gradient.

Context endpoint:
reuse C1 STRONG_ONLY descriptively; do not retrain it.

Targeted code is allowed and required if needed.
Preferred: build h_tab_direction from TabularEncoder.forward_components() inside DualBranchV21.
Do not change canonical H_current used by Magnitude.

Fail-closed validation:
non-current direction_horizon_gate_mode requires direction_tabular_mode=current,
architecture_mode=full_current,
direction_fusion_alpha=.8 for this experiment.
Reject incompatible combinations.

Gate A:
- current mode benchmark reproduces C0/F80 <=1e-9.
- default/no flag canonical exact.
- G1 L_dir reaches Strong+Weak, not canonical horizon gate.
- G2 L_dir reaches Strong+Weak + global scalar, not canonical horizon gate.
- Temporal L_dir remains live.
- L_mag still reaches canonical Strong+Weak+canonical gate.
- finite shapes, k preserved.
- hashes/leakage unchanged.
- canonical 94 + all previous E2 tests + new focused tests PASS.

Benchmark day=2026-02-13 engineering only.

Formal DEV:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

G0 reuse.
G1 fresh 28.
G2 fresh 28.
No intermediate stopping.

Fresh training fixed:
dir_only / direction_first / vanilla /
architecture_mode=full_current /
direction_tabular_mode=current /
direction_fusion_alpha=.8 /
Stage A / default / seed20260924 / k8 /
max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP / Stage B OFF.

Report:
overall Raw/Balanced/+R/-R/AUC/Brier/ppf/collapse counters.
W1-W4 + min/std.
H1-H3.
paired G1-G0 and G2-G0 with day-cluster bootstrap CI + W/T/L.
G1-G2 descriptive comparison.
runtime.

Gate diagnostics:
G0 24-h gate values.
G1 fixed .8 proof.
G2 scalar trajectory init/final/min/max/delta across runs.
component norms/effective Weak fraction/cosine.
L_dir gradient norms Strong/Weak/gate/Temporal.

Safety anchor=G0:
Balanced >= G0-.01
+Recall >= G0-.05
one-class <= G0+2
+Recall=0 <= G0+2.

E2_C2_GATE HORIZON_GATE_SIGNAL exactly one:
FIXED_SUFFICIENT
GLOBAL_SUFFICIENT
GLOBAL_BETTER
HORIZON_SPECIFIC_USEFUL
GATE_LEARNING_HARMFUL
NO_CLEAR_GATE_EFFECT

Use paired/bootstrap + cross-window evidence. Do not promote from pooled Raw alone.

Forbidden:
no horizon alpha grid;
no .7/.75/.85/.9;
no Strong role split yet;
no PLE/depth/width/k;
no Temporal/FFT;
no residual adapter/new fusion form;
no threshold/calibration;
no checkpoint change;
no Stage B;
no broader DEV/lockbox.

Output:
experiments/first_test/E2_architecture/horizon_gate/
with 00_PLAN_SNAPSHOT.md, benchmark/, runs/, daily_metrics.csv, model_summary.csv,
window_metrics.csv, hour_segment_metrics.csv, paired_summary.csv,
gate_diagnostics.csv, gradient_ownership.json, runtime.csv,
E2_C2_summary.md, E2_C2_GATE.md, figures/.

After E2-C2 STOP.

Final report only:
1 FILES_CHANGED
2 TESTS
3 GATE_ROUTING_AUDIT
4 BENCHMARK_DAY engineering
5 G0/G1/G2 overall
6 W1-W4
7 H1-H3
8 paired G1-G0
9 paired G2-G0
10 safety
11 gate/component diagnostics
12 runtime
13 E2_C2_GATE HORIZON_GATE_SIGNAL
14 NEXT suggestion only
15 unexecuted later work

Do not stage/commit.

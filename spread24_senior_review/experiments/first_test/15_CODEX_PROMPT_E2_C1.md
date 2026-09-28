# Codex Prompt — E2-C1 Tabular Strong/Weak Big-Block Attribution

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

Only authorized task:
Execute E2-C1 Tabular Strong/Weak Big-Block Attribution.

Read first:
1. experiments/first_test/14_E2_C1_TABULAR_BIG_BLOCK_PLAN.md
2. experiments/first_test/E2_architecture/fusion_coarse/E2_B1_summary.md
3. experiments/first_test/E2_architecture/fusion_coarse/E2_B1_GATE.md
4. experiments/first_test/E2_architecture/E2_A_summary.md
5. experiments/first_test/E2_architecture/E2_A_GATE.md
6. docs/01_业务数据与防泄漏合同.md
7. docs/17_最终模型设计与编码规范.md
8. docs/21_正式运行、验收与实验入口规范.md

Do not redesign the experiment. Follow 14 exactly.

Research principle:
If evidence points to a structural/code problem, targeted source changes are allowed and required.
Do not preserve a bad structure merely because it is canonical.
But canonical defaults and frozen data/leakage contracts must remain unchanged until human-reviewed promotion.

Frozen facts:
- Tabular is the main Direction information carrier.
- Temporal-only is weak, but Temporal assists class balance in fusion.
- scalar alpha coarse map produced NO_CLEAR_REGION.
- no local alpha refinement is authorized.
- F80 fixed alpha=.8 is NOT promoted; it is used only as a clean screening control to remove learnable-alpha confounding.

Selector composition:
222 selected:
Strong-BOTH 147
Strong-DIR 32
Strong-MAG 22
Forced-Core 10
Weak 11
=> Strong/Core 211, Weak 11.

Current Tabular:
Strong/Core -> official TabM -> H_strong
Weak -> MLP -> H_weak
H_tab = horizon_gate * H_strong + (1-horizon_gate) * H_weak

Implement a clean experiment-only Direction tabular mode:
current / strong_only / weak_only.

Preferred targeted refactor:
- TabularEncoder.forward_components() exposes H_strong, H_weak, gate, H_current.
- existing TabularEncoder.forward() still returns H_current bit-identically.
- Direction can receive a separate h_tab_direction.
- Magnitude always receives canonical H_current.
- MemberWiseTaskAdaptersV21 may accept h_tab_direction=None, with default preserving old behavior exactly.
- strong_only uses H_strong.
- weak_only broadcasts H_weak across k.
- do not collapse TabM member/head semantics.
- do not implement fake ablation by multiplying a branch by zero while leaving unintended gradients.

All E2-C1 arms use fixed direction_fusion_alpha=.8 so Strong/Weak is the only experimental axis.

Arms:
C0 CURRENT_TABULAR:
reuse F80 28-day artifacts; Strong+Weak+gate.

C1 STRONG_ONLY:
Direction receives H_strong only.
Weak MLP and horizon gate must not receive L_dir gradient.

C2 WEAK_ONLY:
Direction receives broadcast H_weak only.
Strong/TabM and horizon gate must not receive L_dir gradient.

Temporal Direction branch remains live in C1/C2 because alpha_time=.8.
Magnitude remains canonical current H_tab in every arm.

Gate A must prove:
1. current mode + fixed .8 reproduces F80 <=1e-9.
2. default/no flag preserves canonical behavior.
3. C1 L_dir reaches Strong/TabM, not Weak or gate.
4. C2 L_dir reaches Weak, not Strong/TabM or gate.
5. Temporal still reaches L_dir in C1/C2.
6. Magnitude still reaches canonical Strong+Weak+gate.
7. finite shapes and k axis preserved.
8. config/source/selector/leakage unchanged.
9. canonical 94 tests plus all E2-A/B1 tests remain green, with focused new tests.

Benchmark day=2026-02-13, engineering only. Do not rank from it.

Formal DEV panel:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

C0 reuse F80.
C1 run all 28.
C2 run all 28.
No intermediate stopping.

Training fixed:
dir_only / direction_first / vanilla / architecture_mode=full_current /
direction_fusion_alpha=.8 / Stage A / default / seed 20260924 / k8 /
max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP / Stage B OFF.

Report:
overall Raw/Balanced/+R/-R/AUC/Brier/ppf/collapse counters.
W1-W4 + min-window/std.
H1-H3.
paired C1-C0 and C2-C0 with day-cluster bootstrap 95% CI and W/T/L.
runtime/best-stop/GPU.

Required component diagnostics:
- feature count/names by selector role
- selected-checkpoint 24 horizon gate values for C0
- gate movement from init .8 overall and H1/H2/H3
- ||g*H_strong|| and ||(1-g)*H_weak||
- effective weak norm fraction
- cosine Strong vs broadcast Weak
- L_dir gradient norms for Strong/TabM, Weak MLP, horizon gate

Safety anchor=C0 F80.
Admissible iff:
Balanced >= C0-.01
+Recall >= C0-.05
one-class days <= C0+2
+Recall=0 days <= C0+2

E2_C1_GATE TABULAR_SIGNAL must be exactly one:
STRONG_DOMINANT
WEAK_RESIDUAL
WEAK_UNEXPECTEDLY_STRONG
STRONG_WEAK_SYNERGY
TABULAR_MIXING_PROBLEM
BOTH_INTERNAL_WEAK
INCONCLUSIVE

Interpretation is pre-registered in 14. Use paired day bootstrap + cross-window evidence; do not call a pooled Raw edge stable by itself.

Forbidden:
PLE ablation, Strong-DIR/MAG/Core role split, TabM depth/width/k, selector changes, Temporal group/FFT changes, alpha sweep, concat/attention/MMoE/router, threshold/calibration, checkpoint tolerance, Stage B, full DEV, lockbox.

Output:
experiments/first_test/E2_architecture/tabular_big_block/
with 00_PLAN_SNAPSHOT.md, benchmark/, runs/, daily_metrics.csv, model_summary.csv,
window_metrics.csv, hour_segment_metrics.csv, paired_summary.csv,
tabular_component_diagnostics.csv, gate_values_by_run.csv,
gradient_ownership.json, runtime.csv, E2_C1_summary.md, E2_C1_GATE.md, figures/.

After E2-C1 STOP.

Final report only:
1 FILES_CHANGED
2 TESTS
3 TABULAR_ROUTING_AUDIT
4 BENCHMARK_DAY engineering
5 C0/C1/C2 overall
6 W1-W4
7 H1-H3
8 paired C1-C0
9 paired C2-C0
10 safety
11 horizon-gate/component diagnostics
12 runtime
13 E2_C1_GATE TABULAR_SIGNAL
14 NEXT suggestion only
15 unexecuted later work

Do not stage/commit.

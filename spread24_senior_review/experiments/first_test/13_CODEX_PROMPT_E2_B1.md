# Codex Prompt — E2-B1 Direction Fusion Coarse Map

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

Only authorized task: E2-B1 Direction Fusion Coarse Map.

Read first:
- experiments/first_test/12_E2_B1_FUSION_COARSE_PLAN.md
- experiments/first_test/E2_architecture/E2_A_summary.md
- experiments/first_test/E2_architecture/E2_A_GATE.md
- experiments/first_test/10_E2_ARCHITECTURE_BIG_BLOCK_PLAN.md
- docs/01_业务数据与防泄漏合同.md
- docs/17_最终模型设计与编码规范.md
- docs/21_正式运行、验收与实验入口规范.md

Frozen E2-A facts:
A0 FULL_CURRENT Raw .578869 / Bal .527302 / AUC .566327.
A1 TABULAR_ONLY Raw .595238 / Bal .498539 / +R .152893: majority-biased, NOT promotable.
A2 TEMPORAL_ONLY Raw .500000 / Bal .474707.
A2-A0 Raw delta -7.89pp has day-bootstrap CI excluding zero.
Gate attribution = TABULAR_DOMINANT, but human reading is “Tabular main signal + Temporal complementary correction”.

Current formula:
H_dir = alpha_time*time_dir(H_time) + (1-alpha_time)*tab_dir(H_tab)

Do not interpret alpha as literal contribution percentage.

Implement experiment-only fixed Direction fusion alpha.
Prefer CLI:
--direction-fusion-alpha FLOAT
default=None.

None preserves canonical learnable fusion exactly.
FLOAT fixes alpha_time in [0,1] for Direction only.
Canonical defaults must not change.

Arms:
F00 alpha=0.0: reuse E2-A A1.
F20 alpha=0.2: fresh.
F40 alpha=0.4: fresh.
F60 alpha=0.6: fresh.
F80 alpha=0.8 fixed: fresh.
F100 alpha=1.0: reuse E2-A A2.
FL08: reuse E2-A A0 current learnable init .8.

Gate A:
- fixed 0 benchmark reproduces tabular_only <=1e-9 if semantics equivalent
- fixed 1 benchmark reproduces temporal_only <=1e-9
- fixed alpha receives no update
- endpoint gradient ownership correct
- finite/shape/leakage PASS
- canonical non-destructive suite >=94 PASS + new tests
- formal/default behavior unchanged

Benchmark day=2026-02-13 engineering only. Never rank/promote from it.

DEV panel:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

Fresh F20/F40/F60/F80 each run all 28 days; total 112 fresh formal runs.
Do not stop on intermediate results.

All fresh runs fixed:
dir_only / direction_first / vanilla / Stage A / default profile / seed 20260924 / k8 / max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP / Stage B OFF.

Report for all 7 arms:
overall Raw/Balanced/+R/-R/AUC/Brier/ppf
collapse and zero-recall counts
W1-W4 and min-window/std
H1/H2/H3
daily paired deltas vs FL08
day-cluster bootstrap 95% CI + W/T/L
runtime/best-stop/GPU.

Add effective fusion diagnostics; do not use alpha alone:
norm_tab=||(1-alpha)*tab_dir(H_tab)||
norm_time=||alpha*time_dir(H_time)||
effective_time_norm_fraction=norm_time/(norm_tab+norm_time+eps)
cosine(tab_component,time_component)
Use deterministic diagnostic batch, observation only.

Safety anchor=FL08.
SAFETY_ADMISSIBLE iff:
Balanced >= FL08-0.01
+Recall >= FL08-0.05
one-class days <= FL08+2
+Recall=0 days <= FL08+2

Always report non-admissible arms; never hide them.

Generate response curves and Pareto frontier Raw vs Balanced.

E2-B1 Gate:
FUSION_REGION =
TABULAR_HEAVY / BALANCED_MIX / TEMPORAL_HEAVY /
ENDPOINT_TABULAR / CURRENT_FUSION_OK / NO_CLEAR_REGION

Interpretation:
TABULAR_HEAVY if best admissible region alpha<=.4.
BALANCED_MIX if .4<alpha<.7.
TEMPORAL_HEAVY if alpha>=.7.
ENDPOINT_TABULAR if no interior point improves endpoint structure.
CURRENT_FUSION_OK if FL08 remains best admissible tradeoff.
NO_CLEAR_REGION if uncertainty prevents conclusion.

Do not automatically execute the next step.

Forbidden:
PLE/FFT/hidden/depth/k, Strong/Weak ablation, concat/late fusion, attention/MMoE/router, threshold/calibration, checkpoint tolerance, protected-gradient, Stage B, full DEV, lockbox, oracle routing.

Output under:
experiments/first_test/E2_architecture/fusion_coarse/

At least:
00_PLAN_SNAPSHOT.md
benchmark/
runs/
daily_metrics.csv
model_summary.csv
window_metrics.csv
hour_segment_metrics.csv
paired_summary.csv
fusion_diagnostics.csv
runtime.csv
E2_B1_summary.md
E2_B1_GATE.md
figures/

After completion STOP.

Final report only:
1 FILES_CHANGED
2 TESTS
3 FIXED_ALPHA_ROUTING_AUDIT
4 BENCHMARK_DAY engineering
5 seven-arm overall table
6 W1-W4
7 H1-H3
8 paired vs FL08
9 safety admissibility
10 effective fusion diagnostics
11 runtime
12 E2_B1_GATE FUSION_REGION
13 NEXT suggestion only
14 explicit list of unexecuted later work.

Do not stage/commit.

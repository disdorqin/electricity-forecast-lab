# Codex Prompt — E2-C3 Task-Aware Strong Role Routing

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

Only authorized task:
Execute E2-C3 Task-Aware Strong Role Routing.

Read first:
1. experiments/first_test/18_E2_C3_STRONG_ROLE_ROUTING_PLAN.md
2. experiments/first_test/E2_architecture/horizon_gate/E2_C2_summary.md
3. experiments/first_test/E2_architecture/horizon_gate/E2_C2_GATE.md
4. experiments/first_test/E2_architecture/tabular_big_block/E2_C1_summary.md
5. experiments/first_test/E2_architecture/fusion_coarse/E2_B1_summary.md
6. docs/01_业务数据与防泄漏合同.md
7. docs/17_最终模型设计与编码规范.md
8. docs/21_正式运行、验收与实验入口规范.md

Follow plan 18 exactly. Do not redesign the study.

Frozen human decision:
- E2-C2 Gate=NO_CLEAR_GATE_EFFECT.
- fixed .8 horizon gate is weaker.
- global scalar is close/safe but never beats current 24-h gate on a day and loses on two days.
- retain current 24-h gate as the controlled screening architecture; no more gate tuning.
- now test whether selector Strong task roles should affect Direction architecture.

Frozen selector counts:
Strong-BOTH 147
Strong-DIR 32
Strong-MAG 22
Forced-Core 10
Weak 11

Arms:

R0 ALL_STRONG:
reuse G0/C0/F80.
211 Strong/Core + 11 Weak.
current 24-h gate.
fixed direction_fusion_alpha=.8.

R1 DROP_MAG:
fresh.
Strong-BOTH + Strong-DIR + Forced-Core = 189 Strong/Core.
Exclude exactly 22 Strong-MAG from Tabular Strong input.
Weak 11 retained.

R2 DROP_DIR:
fresh.
Strong-BOTH + Strong-MAG + Forced-Core = 179 Strong/Core.
Exclude exactly 32 Strong-DIR.
Weak 11 retained.

Preferred targeted implementation:
TabularEncoder experiment-only strong_role_profile =
all / drop_mag / drop_dir
default=all.

The frozen 222-feature preprocessed tensor remains unchanged.
Only TabularEncoder.strong_indices changes at construction.
weak_indices remain exactly the 11 Weak features.

This intentionally changes TabM input dimension / parameter count in R1/R2.
Because objective_mode=dir_only, Direction metrics are valid but Magnitude capability is NOT part of E2-C3 and must not be interpreted.
If a filtered role profile later survives confirmation, joint training must receive a proper task-specific Direction/Magnitude routing design so Magnitude can retain Strong-MAG.

Fail closed:
non-default strong_role_profile requires:
legacy_v20=False,
objective_mode=dir_only,
architecture_mode=full_current,
direction_tabular_mode=current,
direction_horizon_gate_mode=current,
direction_fusion_alpha=.8.

Gate A:
- all profile reproduces G0/F80 benchmark <=1e-9.
- default/no flag canonical exact.
- R1 role inventory exactly 147 BOTH + 32 DIR + 10 Core =189; zero MAG.
- R2 exactly 147 BOTH +22 MAG +10 Core =179; zero DIR.
- Weak exactly 11 all arms.
- Temporal and current horizon gate live.
- finite, k8.
- hashes/leakage frozen.
- canonical94 + all previous E2 tests + focused tests PASS.

Benchmark day=2026-02-13 engineering only.

Formal DEV:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

R0 reuse.
R1 fresh 28.
R2 fresh 28.
No intermediate stopping.

Training:
dir_only / direction_first / vanilla /
architecture_mode full_current /
direction_tabular_mode current /
direction_horizon_gate_mode current /
direction_fusion_alpha=.8 /
Stage A / default / seed20260924 / k8 /
max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP / Stage B OFF.

Report:
overall Raw/Balanced/+R/-R/AUC/Brier/ppf/collapse.
W1-W4 + min/std.
H1-H3.
paired R1-R0, R2-R0, descriptive R1-R2.
day-cluster bootstrap 95% CI + W/T/L.
parameter/runtime/GPU/best-stop.
role inventory and exact included/excluded feature names.

Safety anchor=R0:
Balanced>=R0-.01
+Recall>=R0-.05
one-class<=R0+2
+R=0 days<=R0+2.

E2_C3_GATE ROLE_SIGNAL exactly one:
DIR_ROLE_ALIGNED
MAG_FEATURES_HELP_DIRECTION
DIR_FEATURES_REDUNDANT
BOTH_TASK_SPECIFIC_GROUPS_HELP
ROLE_FILTER_PROMISING_BUT_UNPROVEN
NO_CLEAR_ROLE_EFFECT

Critical anti-overfit rule:
If R1 DROP_MAG improves, DO NOT promote it.
Next must be E2-C4 matched-random-drop + >=3-seed confirmation to distinguish true role alignment from generic feature-count regularization/optimization-path effects.

Forbidden:
selector rerun, frozen feature deletion, BOTH/Core split, random-drop controls now, PLE, depth/width/k, gate tuning, Temporal/FFT, new fusion/attention, threshold/calibration, checkpoint change, Stage B, broader DEV, lockbox.

Output under:
experiments/first_test/E2_architecture/strong_role_routing/

At least:
00_PLAN_SNAPSHOT.md
benchmark/
runs/
daily_metrics.csv
model_summary.csv
window_metrics.csv
hour_segment_metrics.csv
paired_summary.csv
role_inventory.csv/json
runtime.csv
gradient_ownership.json
E2_C3_summary.md
E2_C3_GATE.md
figures/

After E2-C3 STOP.

Final report only:
1 FILES_CHANGED
2 TESTS
3 ROLE_ROUTING_AUDIT
4 BENCHMARK engineering
5 R0/R1/R2 overall
6 W1-W4
7 H1-H3
8 paired R1-R0
9 paired R2-R0
10 safety
11 role inventory / parameter counts
12 runtime
13 E2_C3_GATE ROLE_SIGNAL
14 NEXT suggestion only
15 unexecuted later work

Do not stage/commit.

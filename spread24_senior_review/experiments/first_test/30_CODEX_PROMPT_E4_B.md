# Codex Prompt — E4-B Frozen Selector Reopening / Weak Feature Recovery

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

ONLY authorized task:
Execute experiments/first_test/29_E4_B_FEATURE_RECOVERY_PLAN.md exactly.

Read first:
1 experiments/first_test/29_E4_B_FEATURE_RECOVERY_PLAN.md
2 experiments/first_test/26_E4_ROOT_CAUSE_AND_LITERATURE.md
3 experiments/first_test/E4_regime_stacker/E4_A_summary.md
4 experiments/first_test/E2_architecture/horizon_specialized_head/E2_E1_summary.md
5 docs/01_业务数据与防泄漏合同.md
6 docs/17_最终模型设计与编码规范.md
7 docs/21_正式运行、验收与实验入口规范.md

Frozen decision:
- valid best remains Q2 SEGMENT_HEADS Raw=.6086 / Balanced=.5424 / AUC=.5911 / safety PASS.
- E4-A postprocess failed; stop postprocessing.
- structured smoothing >.63 Raw is invalid due severe positive-class collapse.
- sqrt class weighting full28 ~.6161 is not promotion evidence and fails safety.
- E4-B MUST use unweighted Direction loss and no structured decoder.

Formal arms:
F0 SELECTED222: reuse Q2; Strong211 Weak11.
F1 LITERATURE240: selected222 + EXACT18 from plan29, recovered as Weak; Strong211 Weak29.
F2 ALL259_WEAK_RECOVERY: all259, all37 dropped Noise recovered as Weak; Strong211 Weak48.
No other arm.

Implement experiment-only feature_recovery_profile=selected222/literature240/all259; default selected222.
Do NOT edit frozen selector JSON. Do NOT rerun selector.
Build in-memory experiment manifest from frozen selector + frozen registry candidate order.
selected_features must follow registry order; selected_indices exact; existing roles unchanged; recovered Noise -> Weak only in experiment manifest.
Record base selector SHA unchanged and separate experiment feature-profile SHA.
All normal eligibility/quarantine/preprocessing APIs must consume the experiment manifest. Never bypass contracts.

Q2 fixed:
dir_only / direction_first / vanilla / segment_heads / full_current / current tabular / current horizon gate / strong_role_profile all / canonical encoding / alpha .8 / postprocess none / class weight unweighted / canonical Stage-A split / default / seed20260924 / k8 / max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP / Stage B OFF.

Gate A must prove all 20 items in plan29, especially identical eligible-index SETS across F0/F1/F2 for all 28 target days and exact Q2 reproduction for selected222.
If eligibility sets differ, STOP before formal runs.

Benchmark 2026-02-13 engineering only.
Formal panel same 28 days; F0 reuse; F1/F2 fresh28; no intermediate stopping.

Report overall/window/hour metrics, paired 10k day bootstrap, slot gains, safety, component/Weak diagnostics, params/runtime.
For F1 MONITOR-only diagnostics, fixed groups A renewable-adjusted errors / B ramp2 / C regime flags. Do not create oracle arms.

Safety anchor F0:
Balanced>=F0-.01; +Recall>=F0-.05; one-class<=F0+2; +R=0<=F0+2.

E4_B_GATE FEATURE_RECOVERY_SIGNAL exactly one:
DOMAIN_GUIDED_RECOVERY / BROAD_SELECTOR_BOTTLENECK / RECOVERY_PROMISING_UNPROVEN / SELECTOR_ROBUST / RECOVERY_HARMFUL / NO_CLEAR_RECOVERY_EFFECT.

If F1/F2 >=.62 with safety: STOP and recommend immediate 3-seed F0+winner; do not micro-tune feature subsets.
If E4-B fails: next recommendation E5 recurrent-regime/similar-day sample selection/weighting, not recency windows or model micro-tuning.

Existing E4-A postprocess and class-weight code/assets may remain. Do not delete. Keep them inactive and explicitly record inactive in E4-B manifests.
Do not stage/commit.

Output under experiments/first_test/E4_feature_recovery/ with all artifacts specified in plan29.
After E4-B STOP.

Final report only:
1 FILES_CHANGED
2 TESTS
3 FEATURE_PROFILE_AUDIT
4 ELIGIBILITY_IDENTITY
5 BENCHMARK engineering
6 F0/F1/F2 overall
7 W1-W4
8 H1-H3
9 paired comparisons
10 slot gains
11 safety
12 family diagnostics
13 component/parameter/runtime
14 E4_B_GATE FEATURE_RECOVERY_SIGNAL
15 NEXT acceleration rule
16 unexecuted later work
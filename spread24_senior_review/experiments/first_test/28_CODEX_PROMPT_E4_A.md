# Codex Prompt — E4-A Regime-Conditioned Direction Probability Stacking

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

Only authorized task:
Execute E4-A exactly as specified in:
experiments/first_test/27_E4_A_REGIME_STACKER_PLAN.md

Read first:
1. experiments/first_test/26_E4_ROOT_CAUSE_AND_LITERATURE.md
2. experiments/first_test/27_E4_A_REGIME_STACKER_PLAN.md
3. experiments/first_test/E2_architecture/horizon_specialized_head/E2_E1_summary.md
4. docs/17_最终模型设计与编码规范.md
5. docs/01_业务数据与防泄漏合同.md
6. docs/21_正式运行、验收与实验入口规范.md

Also inspect the current source before editing:
- src/TafM_改进源码/train.py
- src/TafM_改进源码/direction_postprocess.py
- src/TafM_改进源码/models/dual_branch_v21.py
- src/run_tabm_v21.py

IMPORTANT WIP NOTE:
The advisor has already created direction_postprocess.py and added a preliminary import / unused direction_postprocess_mode argument in train.py.
Treat this as WIP only.
Do not assume it is correct or complete.
You must either complete it coherently or revise/remove the WIP parts.
Never leave a CLI/function argument that is accepted but silently ignored.

## Frozen facts

Current best deep architecture is Q2 SEGMENT_HEADS:
Raw=.6086
Balanced=.5424
AUC=.5911
Safety PASS
+13/672 correct slots vs shared head.

E3-A recency-window experiment failed:
T0=.6086
T1=.5685
T2=.5506
T3=.5357
all fresh recency arms materially worse.
Do not continue recency window tuning.

Literature/diagnosis says next test is regime-aware correction, not a bigger neural backbone.

## Experiment design

Use ORIGINAL_Q2 as P0 anchor.

For each target day create ONE fresh E4 deep run with a clean three-way chronological split.

First create canonical Q2 split exactly:
eligible <= D-2
BASE = older80%
FULL_MONITOR = newest20%

Then carve:
CALIBRATOR = newest 90 days of FULL_MONITOR
CHECKPOINT_MONITOR = all older FULL_MONITOR days

BASE must remain exactly the canonical Q2 BASE.
Preprocessor fits BASE only.

CHECKPOINT_MONITOR selects the Q2 checkpoint.
CALIBRATOR labels are not used in checkpoint selection.
Target-day truth is never used in fitting.

Require:
CALIBRATOR exactly90 days.
CHECKPOINT_MONITOR >=60 days.
chronological adjacency.
CALIBRATOR ends D-2.
no overlap.

Deep model is exactly Q2:
dir_only / direction_first / vanilla /
direction_readout_mode=segment_heads /
full_current /
direction_tabular_mode=current /
direction_horizon_gate_mode=current /
strong_role_profile=all /
numeric_encoding_mode=canonical /
direction_fusion_alpha=.8 /
Stage A / default / seed20260924 / k8 /
max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP.

No Stage B.

## Arms from same deep checkpoint

P0 ORIGINAL_Q2:
reuse E2-E1 Q2, no fresh training.

P1 SPLIT_CONTROL:
fresh E4 Q2 checkpoint; no postprocessing.

P2 SEGMENT_LOGIT:
same exact P1 checkpoint.
Fit on CALIBRATOR only with:
base_logit
H2
H3
base_logit*H2
base_logit*H3

P3 REGIME_LOGIT:
same exact P1 checkpoint.
P2 features plus exactly:
residual_load_renew
renewable_share
bidding_space_ratio
net_ramp_pressure
err_net_load_28d_std
uncert_风电总加_width
uncert_光伏总加_width
ctx_spread_positive_rate14
spread_same_slot_28d_positive_rate

Use the BASE-fitted normalized x_future tensor.
All nine must already be in frozen selected_features.
Fail if any is missing.

## Stacker contract

Implement internally with torch/numpy.
Do not add sklearn to formal requirements.

sigmoid(Xw+b)

Frozen:
unweighted BCE
L2=1e-3
max_iter=100
deterministic LBFGS
base_logit weight init=1
all other weights init=0
bias init=0
threshold remains .5

No L2 tuning.
No class weighting.
No threshold tuning.
No feature search.

## Efficiency requirement

ONLY 28 fresh deep Q2 trainings.

Do not train P2/P3 separately.

For each day:
1 train deep Q2 once
2 save checkpoint SHA256
3 produce CHECKPOINT_MONITOR / CALIBRATOR audit
4 generate CALIBRATOR base predictions and normalized meta features
5 fit P2 and P3
6 generate one target base prediction
7 derive P1/P2/P3 from same target/base/checkpoint

Manifest/artifact must prove identical deep checkpoint for P1/P2/P3.

## Gate A

Must prove before formal runs:

- default Q2 without E4 flags reproduces saved Q2 benchmark <=1e-9
- E4 BASE equals canonical Q2 BASE exactly
- FULL_MONITOR equals canonical Q2 monitor exactly
- CHECKPOINT_MONITOR + CALIBRATOR exactly partition FULL_MONITOR
- CALIBRATOR newest90 and ends D-2
- preprocessing BASE-only
- P1/P2/P3 same deep checkpoint hash
- P1 target base probabilities exactly equal P2/P3 stored base probabilities
- P2 feature count exactly5
- P3 feature count exactly14
- 9 regime features in frozen selector and legal
- target truth absent from fit
- threshold .5 unchanged
- source/config/selector/sequence hashes frozen
- segment_heads and k8 preserved
- canonical94 + all existing E2/E3 tests + focused E4 tests PASS
- all incompatible combinations fail closed

## Benchmark

2026-02-13 engineering only.
No ranking.

## Formal panel

Same 28 days:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

P0 reuse.
Only 28 deep E4 runs.
P1/P2/P3 derived.

No intermediate stopping.

## Report

Overall:
Raw/Balanced/+Recall/-Recall/AUC/Brier/ppf/collapse.

W1-W4:
Raw/Balanced/AUC/+R/-R
min Raw / std.

H1-H3:
Raw/Balanced/AUC/+R/-R/Brier.

Paired day-cluster bootstrap 10,000:
P1-P0
P2-P1
P3-P1
P2-P0
P3-P0
P3-P2
95% CI + W/T/L.

Slot gain:
overall/H1/H2/H3.

Calibration:
calibrator base/post Raw/Balanced/AUC/Brier;
calibrator prevalence / mean p;
coefficient distributions/sign consistency;
finite/condition diagnostics.

Runtime:
deep run and postprocess separately.

## Safety

Anchor P0 original Q2.

Candidate admissible iff:
Balanced >= P0-.01
+Recall >= P0-.05
one-class days <= P0+2
+R=0 days <= P0+2.

## Gate

E4_A_GATE POSTPROCESS_SIGNAL exactly one:

REGIME_STACKING_CONFIRMED
SEGMENT_CALIBRATION_HELPFUL
REGIME_PROMISING_UNPROVEN
SPLIT_COST_DOMINATES
POSTPROCESS_HARMFUL
NO_POSTPROCESS_GAIN

Use paired CI + cross-window evidence.
Do not claim 62/63/65 unless actual 28-day Raw reaches it with safety.

## Acceleration

If P2 or P3 >=.62 with safety:
STOP after E4-A and recommend immediate 3-seed P0 + winner.
Do not tune L2/calibrator days/threshold/features first.

If confirmed >=.62:
freeze candidate then broader DEV.

If no gain:
STOP postprocessing.
Next recommendation = E4-B selector reopening:
selected222 vs all259 vs pre-registered literature/regime reintroduced subset.
Do not execute E4-B now.

## Forbidden

No selector rerun.
No new raw source.
No arbitrary new feature.
No threshold tuning.
No class weighting.
No L2 search.
No calibrator-days sweep.
No Stage B.
No architecture change.
No PLE/k/depth/width/gate/fusion changes.
No broader DEV/lockbox.

## Output

experiments/first_test/E4_regime_stacker/

At least:
00_PLAN_SNAPSHOT.md
benchmark/
runs/
split_audit.csv
checkpoint_identity.csv
calibrator_metrics.csv
daily_metrics.csv
model_summary.csv
window_metrics.csv
hour_segment_metrics.csv
paired_summary.csv
slot_gain_summary.csv
stacker_coefficients.csv
stacker_audit.json
runtime.csv
E4_A_summary.md
E4_A_GATE.md
figures/

Final report only:
1 FILES_CHANGED
2 TESTS
3 THREE_WAY_SPLIT_AUDIT
4 DEEP_CHECKPOINT_IDENTITY
5 BENCHMARK engineering
6 P0/P1/P2/P3 overall
7 W1-W4
8 H1-H3
9 paired comparisons
10 slot gains
11 calibrator metrics
12 coefficient/finite diagnostics
13 safety
14 runtime
15 E4_A_GATE POSTPROCESS_SIGNAL
16 NEXT under acceleration rule
17 unexecuted later work

After E4-A STOP.
Do not stage/commit.

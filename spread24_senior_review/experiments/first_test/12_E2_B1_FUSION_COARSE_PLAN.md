# E2-B1 — Direction Fusion Coarse Map

STATUS=AUTHORIZED_NEXT
DATE=2026-09-25
PARENT=E2-A
EXPERIMENT_ROOT=experiments/first_test/E2_architecture/fusion_coarse
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## 1. E2-A frozen evidence

E2-A 56/56 fresh runs complete plus A0 reused 28-day anchor.

A0 FULL_CURRENT:
Raw=0.578869, Balanced=0.527302, AUC=0.566327, +R=0.342975, -R=0.711628.
A1 TABULAR_ONLY:
Raw=0.595238, Balanced=0.498539, AUC=0.563223, +R=0.152893, -R=0.844186.
A2 TEMPORAL_ONLY:
Raw=0.500000, Balanced=0.474707, AUC=0.497751.

Paired:
A1-A0 Raw=+0.01637, 95% CI [-0.0432,+0.0744], not established.
A2-A0 Raw=-0.07887, 95% CI [-0.1339,-0.0253], established negative.

Formal attribution gate:
ARCH_SIGNAL=TABULAR_DOMINANT.

Human reading:
- Tabular carries the main ranking/Raw signal.
- Tabular-only is NOT promotable: it is majority-biased and collapses +Recall.
- Temporal-only is weak globally.
- Temporal nevertheless provides complementary correction inside FULL_CURRENT: A0 restores +Recall/Balanced relative to A1.
- Therefore the next question is not “remove Temporal”, but “what Tabular/Temporal mixture uses Tabular as the primary signal while retaining Temporal correction?”

Current formula:
H_dir = alpha_time * time_dir(H_time) + (1-alpha_time) * tab_dir(H_tab)

Current learnable alpha_time starts near 0.8 and finishes ~0.793. This scalar alone is NOT an effective contribution percentage because branch representation norms can rescale. E2-B1 must not describe it as “79% Temporal contribution”.

## 2. Fixed benchmark

ARCH_BENCHMARK_DAY=2026-02-13.
Engineering only. Never used for promotion.

ARCH_DEV_PANEL unchanged:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

## 3. Single experimental axis

Only vary fixed Direction fusion coefficient alpha_time.

Reuse endpoints:
F00 = alpha_time=0.0 = E2-A A1 TABULAR_ONLY
F100 = alpha_time=1.0 = E2-A A2 TEMPORAL_ONLY

Reuse learnable canonical anchor:
FL08 = E2-A A0 FULL_CURRENT, learnable alpha init 0.8

Fresh fixed-alpha variants:
F20 = 0.20
F40 = 0.40
F60 = 0.60
F80 = 0.80

No other architecture/training setting may change.

Important:
fixed-alpha mode is experiment-only.
Canonical default remains current learnable A2 fusion.
Do not repurpose canonical A0/A1/A2 mode names.

## 4. Gate A implementation

Prefer an explicit experiment-only parameter/CLI such as:
--direction-fusion-alpha <0..1>
with default None.

Semantics:
None => canonical current learnable alpha.
float => fixed alpha_time for Direction fusion only.

Requirements:
- formal/default behavior unchanged.
- fixed 0.0 benchmark must reproduce TABULAR_ONLY <=1e-9 if implementation semantics are equivalent.
- fixed 1.0 benchmark must reproduce TEMPORAL_ONLY <=1e-9.
- fixed 0.8 and learnable-0.8 are different experiment arms; do not claim equivalence.
- alpha fixed means no gradient/update to direction alpha.
- branch gradients must follow coefficient endpoints exactly.
- source/selector/target adapter/data boundary unchanged.
- non-destructive test suite >=94 canonical PASS plus new tests.

## 5. Training protocol

All fresh variants:
objective_mode=dir_only
checkpoint_policy=direction_first
gradient_policy=vanilla
train_mode=stage_a
profile=default
seed=20260924
k=8
max_epochs=120
patience=15
batch_size=64
frozen LR/WD
CUDA+AMP
Stage B OFF

This is architecture screening protocol only; it does not promote direction_first as production checkpoint.

## 6. Execution

Gate A:
implementation + endpoint reproduction + gradient ownership + tests.

Gate B:
2026-02-13 for F20/F40/F60/F80; engineering only, no ranking.

Gate C:
F20/F40/F60/F80 each run all 28 days.
Do not stop based on intermediate windows.

Total fresh formal runs=112.
F00/F100/FL08 are reused.

## 7. Metrics

For every arm:
overall Raw/Balanced/+R/-R/AUC/Brier/ppf
one-class days/+R=0/-R=0
W1-W4 Raw/Balanced/AUC/+R/-R
min-window Raw/window std
H1/H2/H3 Raw/Balanced/AUC/+R/-R
daily paired deltas vs FL08
day-cluster bootstrap 95% CI and W/T/L

Also record:
best/stop epoch, runtime, GPU memory.

## 8. Effective fusion diagnostics

Do NOT infer effective branch contribution from alpha_time alone.

For benchmark and a small deterministic diagnostic batch per formal run, record pre-fusion adapted representation norms:
norm_tab = ||(1-alpha)*tab_dir(H_tab)||
norm_time = ||alpha*time_dir(H_time)||
effective_time_norm_fraction = norm_time/(norm_tab+norm_time+eps)

Also record cosine(tab_component,time_component).

These are diagnostics only, not optimization targets.

## 9. Safety/Pareto reading

A high Raw is not sufficient.

Use FL08=A0 as safety anchor.
A fixed-alpha candidate is SAFETY_ADMISSIBLE only if:
- Balanced >= FL08 Balanced - 0.01
- +Recall >= FL08 +Recall - 0.05
- one-class days <= FL08 + 2
- +Recall=0 days <= FL08 + 2

Report all variants regardless of admissibility.

Primary architecture map:
- Raw response vs alpha
- Balanced response vs alpha
- +R/-R response vs alpha
- W1-W4 response vs alpha
- Pareto frontier Raw vs Balanced

No automatic final architecture promotion in E2-B1.

## 10. E2-B1 interpretation

FUSION_REGION may be:
- TABULAR_HEAVY: best admissible region alpha_time <=0.4
- BALANCED_MIX: best admissible region 0.4<alpha_time<0.7
- TEMPORAL_HEAVY: best admissible region >=0.7
- ENDPOINT_TABULAR: no interior admissible point improves on tabular endpoint structure
- CURRENT_FUSION_OK: FL08 remains the best admissible tradeoff
- NO_CLEAR_REGION: noise/CI prevents conclusion

Next step after human review:
- TABULAR_HEAVY/BALANCED_MIX/TEMPORAL_HEAVY => E2-B2 local fusion refinement + learnable-alpha initialization study.
- ENDPOINT_TABULAR => E2-C-TAB branch decomposition.
- CURRENT_FUSION_OK => E2-C-TAB branch decomposition while retaining current fusion.
- NO_CLEAR_REGION => REVIEW.

## 11. Forbidden

No PLE/FFT/hidden/depth/k changes.
No Strong/Weak ablation yet.
No concat/late fusion yet.
No attention/MMoE/router.
No threshold/calibration.
No checkpoint tolerance.
No protected-gradient.
No Stage B.
No full Jan-Aug DEV.
No lockbox.
No day/month oracle routing.

## 12. Stop

After E2-B1 map and Gate, STOP for human review.

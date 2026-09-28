# E2-C3 — Task-Aware Strong Role Routing

STATUS=COMPLETE_STOP_FOR_HUMAN_REVIEW
DATE=2026-09-26
PARENT=E2-C2
EXPERIMENT_ROOT=experiments/first_test/E2_architecture/strong_role_routing
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## 0. Frozen interpretation before this experiment

E2-C2 formal Gate:
HORIZON_GATE_SIGNAL=NO_CLEAR_GATE_EFFECT.

Human review:
- G1 fixed .8 gate is weaker than current 24-h gate: ΔRaw=-2.08pp, 0/25/3 days, W2-W4 negative, safety fails.
- G2 global scalar is safety-admissible and close to G0, but never beats G0 on a day (0/26/2) and loses on 2026-04-13 and 2026-08-11.
- The 2026-08-11 five changed slots are the same five slots that previously distinguished F80 from FL08, providing cross-experiment evidence that small horizon-specific training differences can matter in a regime-specific way.
- Therefore the 24-h gate is NOT promoted as proven superior, but it is retained as the controlled architecture-screening gate. No further gate tuning is authorized.

E2-C1:
- Strong/Core carries most Direction signal.
- Weak-only is stably worse.
- Current Strong+Weak tends to beat Strong-only, so Weak remains as a small residual path.
- Weak effective norm fraction ~6.5%.

Next large question:
Does the selector's task-specific Strong role assignment belong in the Direction architecture?

## 1. Selector role inventory

Frozen selected 222:
- Strong-BOTH = 147
- Strong-DIR = 32
- Strong-MAG = 22
- Forced-Core = 10
- Weak = 11

Current Strong/Core path uses all non-Weak:
211 features.

E2-C3 only changes which Strong role families enter the Tabular encoder for the Direction-only screening model.
Weak remains the same 11 features.
Temporal remains unchanged.
Selector itself is NOT rerun or changed.

## 2. Experimental arms

### R0 ALL_STRONG — anchor
Reuse G0/C0/F80:
Strong-BOTH + Strong-DIR + Strong-MAG + Forced-Core = 211 Strong/Core
plus 11 Weak
current 24-h learnable horizon gate
fixed direction_fusion_alpha=.8.

### R1 DROP_MAG
Fresh 28 days.

Strong path:
Strong-BOTH + Strong-DIR + Forced-Core
= 147 + 32 + 10 = 189 Strong/Core.

Remove the 22 Strong-MAG features from the Tabular encoder input for this Direction-only screening run.
Weak 11 retained.
Current 24-h gate retained.

Question:
Are features selected as magnitude-specific harmful/redundant for Direction?

### R2 DROP_DIR
Fresh 28 days.

Strong path:
Strong-BOTH + Strong-MAG + Forced-Core
= 147 + 22 + 10 = 179 Strong/Core.

Remove the 32 Strong-DIR features.
Weak 11 retained.
Current 24-h gate retained.

This is the task-role negative control:
If selector task roles are meaningful for neural Direction use, removing Strong-DIR should be more harmful than removing Strong-MAG.

No SHARED_ONLY arm yet.
No random-drop control yet.
Those are reserved for follow-up only if R1 produces promising evidence.

## 3. Clean targeted code design

This experiment is objective_mode=dir_only.

Preferred minimal implementation:
add experiment-only TabularEncoder construction option, e.g.

strong_role_profile =
- all
- drop_mag
- drop_dir

Default=all.

The encoder still receives the frozen 222-feature preprocessed tensor and frozen feature_roles.
Only its Strong/Core index set changes at construction:

all:
role != Weak

drop_mag:
role in {Strong-BOTH, Strong-DIR, Forced-Core}

drop_dir:
role in {Strong-BOTH, Strong-MAG, Forced-Core}

Weak indices remain exactly role==Weak.

This intentionally changes the TabM input dimension and parameter count in R1/R2.
That is part of the architecture question.

Because this stage is dir_only:
- Direction metrics are valid.
- Magnitude capability must NOT be interpreted or compared in E2-C3.
- If a role-filtered candidate is later promoted toward joint training, a task-specific Direction/Magnitude routing design must be implemented and revalidated; do not silently remove Strong-MAG from the future joint magnitude pathway.

Do not duplicate the full encoder or run two stochastic TabM passes merely to preserve unused magnitude outputs in this screening stage.
Keep the experiment scientifically simple.

Canonical/default strong_role_profile=all must remain bit-identical.

## 4. Fail-closed rules

Non-default strong_role_profile is experiment-only and requires:
- legacy_v20=False
- objective_mode=dir_only
- architecture_mode=full_current
- direction_tabular_mode=current
- direction_horizon_gate_mode=current
- direction_fusion_alpha=.8

Reject incompatible combinations.

No selector regeneration.
No preprocessing changes.
No source/label boundary changes.

## 5. Gate A — exactness and role inventory

Before formal runs:

1. strong_role_profile=all reproduces G0/F80 benchmark <=1e-9.
2. default/no flag reproduces canonical behavior.
3. R1 Strong indices contain exactly:
   147 Strong-BOTH + 32 Strong-DIR + 10 Forced-Core = 189.
   No Strong-MAG index enters Strong path.
4. R2 Strong indices contain exactly:
   147 Strong-BOTH + 22 Strong-MAG + 10 Forced-Core = 179.
   No Strong-DIR index enters Strong path.
5. Weak indices remain exactly the same 11 in all arms.
6. Temporal pathway unchanged/live.
7. current 24-h horizon gate remains live.
8. shapes finite, k=8 preserved.
9. source/config/selector/sequence hashes frozen.
10. canonical 94 + all previous E2 tests + focused new tests PASS.

## 6. Benchmark

ARCH_BENCHMARK_DAY=2026-02-13.

Engineering only:
R0 reuse where possible.
R1/R2 fresh benchmark.

Record:
metrics, convergence, parameter count, runtime, GPU memory,
feature-role inventory, gate trajectory, gradient norms.

Never rank from the benchmark day.

## 7. Formal DEV panel

Same 28 frozen days:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

R0 reuse G0/C0/F80.
R1 fresh 28.
R2 fresh 28.

No stopping on intermediate results.

## 8. Training protocol

R1/R2:
- objective_mode=dir_only
- checkpoint_policy=direction_first
- gradient_policy=vanilla
- architecture_mode=full_current
- direction_tabular_mode=current
- direction_horizon_gate_mode=current
- direction_fusion_alpha=.8 fixed
- train_mode=stage_a
- profile=default
- seed=20260924
- k=8
- max_epochs=120
- patience=15
- batch_size=64
- frozen LR/WD
- CUDA+AMP
- Stage B OFF

Only strong role composition changes.

## 9. Metrics

Overall:
Raw / Balanced / +Recall / -Recall / AUC / Brier / ppf
one-class / +Recall=0 / -Recall=0.

Stability:
W1-W4 Raw/Balanced/AUC/+R/-R
min-window Raw
window Raw std.

Hour:
H1/H2/H3 Raw/Balanced/AUC/+R/-R.

Paired:
R1-R0
R2-R0
R1-R2 descriptive
day-cluster bootstrap 95% CI + W/T/L.

Architecture:
parameter counts, trainable counts, runtime, GPU memory, best/stop epoch.

## 10. Safety

Anchor R0.

SAFETY_ADMISSIBLE iff:
Balanced >= R0-.01
+Recall >= R0-.05
one-class days <= R0+2
+Recall=0 days <= R0+2

All arms reported even if non-admissible.

## 11. Pre-registered interpretation

ROLE_SIGNAL exactly one:

DIR_ROLE_ALIGNED
- R1 DROP_MAG is admissible and matches/beats R0;
- R2 DROP_DIR is materially/stably worse than R0 or clearly worse than R1.
Interpretation: selector task roles are directionally meaningful; Strong-MAG is unnecessary/harmful for Direction while Strong-DIR matters.

MAG_FEATURES_HELP_DIRECTION
- R1 is materially/stably worse than R0.
Interpretation: Strong-MAG-labelled features still carry useful nonlinear Direction information; do not task-filter by selector label.

DIR_FEATURES_REDUNDANT
- R2 matches/beats R0 without safety loss.
Interpretation: Strong-DIR role does not uniquely contribute under current nonlinear encoder; selector task labels do not map directly to neural utility.

BOTH_TASK_SPECIFIC_GROUPS_HELP
- both R1 and R2 are materially/stably worse than R0.
Interpretation: Direction benefits from interactions across both task-specific role groups.

ROLE_FILTER_PROMISING_BUT_UNPROVEN
- R1 point-estimate improvement/admissibility is promising but paired/cross-window evidence is not established.
Next: matched random-drop controls + multi-seed confirmation before any code promotion.

NO_CLEAR_ROLE_EFFECT
- differences are too small/noisy to support the above.

## 12. Crucial anti-overfitting rule

If R1 improves, DO NOT immediately promote DROP_MAG.

A positive R1 can be caused by:
- true task-role alignment;
- generic regularization from removing 22 inputs / reducing parameters;
- seed-specific optimization path.

Therefore any promising DROP_MAG result must next pass E2-C4 confirmation:
- matched random removal of 22 non-Weak Strong features (multiple deterministic sets),
- at least 3 training seeds for R0 and R1,
- same 28-day DEV first,
before architecture promotion or broader DEV.

This rule is frozen before seeing R1/R2 results.

## 13. What code may change later

If E2-C3/E2-C4 establishes true Direction-specific role value:
implement targeted task-specific tabular input routing so Direction can exclude Strong-MAG while Magnitude retains it.

That will be a real architecture change, not a data deletion.
Selector remains frozen.

If evidence instead shows all roles help, keep the current shared Strong input and proceed to PLE/TabM structural study.

## 14. Forbidden

No rerunning selector.
No deleting features from frozen data.
No Strong-BOTH/Forced-Core split yet.
No random-drop controls in E2-C3.
No PLE on/off.
No TabM depth/width/k.
No horizon-gate tuning.
No Temporal/FFT changes.
No new fusion/attention/MMoE.
No threshold/calibration.
No checkpoint policy change.
No Stage B.
No broader DEV/lockbox.

## 15. Outputs and stop

experiments/first_test/E2_architecture/strong_role_routing/

- 00_PLAN_SNAPSHOT.md
- benchmark/
- runs/
- daily_metrics.csv
- model_summary.csv
- window_metrics.csv
- hour_segment_metrics.csv
- paired_summary.csv
- role_inventory.csv/json
- runtime.csv
- gradient_ownership.json
- E2_C3_summary.md
- E2_C3_GATE.md
- figures/

After E2-C3 STOP for human review.

## 16. Completion record — 2026-09-26

- Formal DEV runs completed: R0 reused anchor; R1/R2 each have 28/28 PASS runs.
- Gate result: `NO_CLEAR_ROLE_EFFECT`; no architecture promotion.
- R1 fails the pre-registered safety gate (Balanced and positive recall / zero-positive-recall-day checks).
- R2 is safety-admissible, but paired intervals include zero and cross-window evidence does not establish the pre-registered superiority/equivalence pattern.
- No E2-C4 was authorized or run: its frozen trigger (promising R1) was not met.
- Screening-only result; canonical V2.1 and frozen selector/features remain unchanged. Stop for human review.

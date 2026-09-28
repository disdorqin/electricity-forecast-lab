# E4-A — Regime-Conditioned Direction Probability Stacking

STATUS=AUTHORIZED_NEXT
DATE=2026-09-26
PARENT=E3-A
EXPERIMENT_ROOT=experiments/first_test/E4_regime_stacker
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS
ARCHITECTURE_CARRIER=E2-E1 Q2 SEGMENT_HEADS

## 0. Frozen evidence

Current strongest architecture:
Q2 SEGMENT_HEADS
- Raw=.6086
- Balanced=.5424
- AUC=.5911
- safety PASS
- +13/672 correct slots vs shared-head Q0

E3-A:
- T0/Q2=.6086
- T1=.5685
- T2=.5506
- T3=.5357
- all fresh recency arms materially worse; RECENCY_SIGNAL=NO_RECENCY_GAIN.

Therefore:
- do NOT continue Stage-A window sweeps;
- do NOT return to architecture micro-ablation;
- preserve Q2 as base predictor;
- next test is low-capacity regime-aware correction.

Read:
experiments/first_test/26_E4_ROOT_CAUSE_AND_LITERATURE.md

## 1. Scientific hypothesis

Q2 has useful ranking information but its probability/decision mapping is not stable across market regimes.

Diagnostic-only 672-slot slicing shows large Raw variation across quartiles of:
- residual_load_renew
- renewable_share
- bidding_space_ratio
- net_ramp_pressure
- err_net_load_28d_std
- wind/solar uncertainty width
- historical spread positive-rate state

This is consistent with literature that DA→RT / balancing corrections depend on:
- lagged price/spread state
- wind forecast errors
- solar forecast errors
- load forecast errors
- residual demand / reserve/scarcity state
- recurrent market regimes

E4-A asks:

> Can a low-capacity, independently fitted, regime-conditioned logistic correction convert Q2's existing signal into more accurate Direction decisions without changing the deep model?

## 2. Critical anti-leakage / anti-double-dip design

Do NOT fit a postprocessor on the same MONITOR labels used to select the Q2 checkpoint.

For each target day D:

1. Build the canonical Stage-A split exactly as before:
   eligible <= D-2
   canonical BASE = older80%
   canonical FULL_MONITOR = newest20%

2. Keep BASE unchanged.

3. Split FULL_MONITOR chronologically into:
   CHECKPOINT_MONITOR = all FULL_MONITOR except newest 90 days
   CALIBRATOR = newest 90 days

4. Requirements:
   - CHECKPOINT_MONITOR >= 60 days
   - CALIBRATOR exactly 90 days
   - CHECKPOINT_MONITOR ends one day before CALIBRATOR starts
   - CALIBRATOR ends at D-2
   - no overlap
   - preprocessor still fits BASE only
   - Q2 checkpoint selection uses CHECKPOINT_MONITOR only
   - calibrator labels never influence deep checkpoint selection
   - target-day truth never influences anything

This three-way split is experiment-only.

## 3. Deep model fixed

All fresh deep runs use exactly Q2:

- objective_mode=dir_only
- checkpoint_policy=direction_first
- gradient_policy=vanilla
- direction_readout_mode=segment_heads
- architecture_mode=full_current
- direction_tabular_mode=current
- direction_horizon_gate_mode=current
- strong_role_profile=all
- numeric_encoding_mode=canonical
- direction_fusion_alpha=.8 fixed
- Stage A only
- profile=default
- seed=20260924
- k=8
- max_epochs=120
- patience=15
- batch_size=64
- LR/WD frozen
- CUDA+AMP

No Stage B.

## 4. Arms

### P0 ORIGINAL_Q2
Reuse E2-E1 Q2 exactly.

This is the external architecture anchor:
Raw=.6086.

### P1 SPLIT_CONTROL
Fresh 28 deep runs.

Uses the E4-A three-way split:
BASE unchanged
CHECKPOINT_MONITOR = canonical monitor minus newest90
CALIBRATOR = newest90

No probability postprocessing.

Purpose:
quantify the effect of reserving an independent calibrator set.

### P2 SEGMENT_LOGIT
Must use the EXACT SAME trained deep checkpoint/predictions as P1 for each day.

Fit a logistic stacker on CALIBRATOR only.

Meta features per slot:
- logit(Q2 p)
- H2 indicator
- H3 indicator
- logit(Q2 p) * H2
- logit(Q2 p) * H3

H1 is the reference segment.
No target feature.
No threshold tuning.

Apply to target day.
Final threshold stays p>=.5.

### P3 REGIME_LOGIT
Must use the EXACT SAME P1 deep checkpoint for each day.

Fit on CALIBRATOR only.

Features = all P2 meta features plus these 9 pre-registered, already-selected legal features from the BASE-fitted normalized target/future tensor:

1. residual_load_renew
2. renewable_share
3. bidding_space_ratio
4. net_ramp_pressure
5. err_net_load_28d_std
6. uncert_风电总加_width
7. uncert_光伏总加_width
8. ctx_spread_positive_rate14
9. spread_same_slot_28d_positive_rate

No feature selection inside E4-A.
No interaction search beyond the fixed P2 logit×segment terms.
No threshold tuning.

## 5. Logistic stacker contract

Use an internal torch/numpy implementation; do NOT add sklearn as a formal dependency.

Model:
sigmoid(X @ w + b)

Loss:
unweighted binary cross-entropy + fixed L2.

Frozen:
- L2 = 1e-3
- max_iter = 100
- deterministic LBFGS
- initial base_logit coefficient = 1
- all other weights = 0
- bias = 0

Do not tune L2.
Do not class-weight.
Do not optimize Raw directly.
Do not tune threshold.

Reason:
E4-A tests whether regime information adds out-of-sample calibration value, not whether a flexible meta-model can overfit 90 days.

## 6. Efficient implementation

Only 28 fresh deep Q2 trainings are authorized.

P1/P2/P3 must share each day's exact deep checkpoint.

Preferred flow:
- train Q2 once under E4-A three-way split;
- save CALIBRATOR predictions + normalized meta features + truth;
- save target base prediction;
- derive P1/P2/P3 from that single checkpoint.

Do NOT retrain Q2 separately for P2/P3.

Manifest/artifact must prove same checkpoint SHA256 for P1/P2/P3.

## 7. Existing WIP source note

Advisor already created:
src/TafM_改进源码/direction_postprocess.py

Advisor also added an import and an unused direction_postprocess_mode argument to train.py as WIP.

This is NOT accepted implementation yet.

Codex must:
- inspect it;
- keep/rewrite/remove WIP code as needed;
- finish a coherent three-way split implementation;
- never leave a flag that is accepted but silently ignored;
- preserve all canonical defaults bit-identically.

## 8. Gate A — routing / leakage / exactness

Before formal runs:

1. Default canonical Q2 route without E4 flags reproduces saved Q2 benchmark <=1e-9.
2. E4 P1 BASE indices exactly equal canonical Q2 BASE indices.
3. FULL_MONITOR equals canonical Q2 MONITOR.
4. CALIBRATOR = newest90 FULL_MONITOR days.
5. CHECKPOINT_MONITOR = FULL_MONITOR excluding CALIBRATOR.
6. CHECKPOINT_MONITOR/CALIBRATOR no overlap, chronological adjacency.
7. CALIBRATOR end = D-2.
8. preprocessor fit range remains canonical BASE only.
9. P1/P2/P3 same deep checkpoint hash.
10. P1 target base prediction equals P2/P3 stored base prediction exactly.
11. P2 uses exactly 5 meta features listed above.
12. P3 uses exactly 14 meta features listed above.
13. all 9 regime features exist in the frozen selected list and are available at forecast origin.
14. no target-day truth/actual enters meta features or fit.
15. final decision threshold remains .5.
16. source/config/selector/sequence hashes frozen.
17. Q2 segment-head route and k=8 preserved.
18. canonical 94 + all prior E2/E3 focused tests + new E4-A tests PASS.

Fail closed on any mismatch.

## 9. Benchmark day

2026-02-13 engineering only.

Run one E4 deep checkpoint and derive P1/P2/P3.

Record:
- split dates/counts
- deep checkpoint hash
- calibrator n slots / positive rate
- P1/P2/P3 metrics
- fitted coefficient audit
- runtime

No ranking from benchmark.

## 10. Formal DEV

Same frozen 28 days:

W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

P0 reuse Q2.
P1 fresh 28.
P2/P3 derived from same 28 P1 deep checkpoints.

No intermediate stopping.

## 11. Metrics

Overall:
- Raw
- Balanced
- +Recall
- -Recall
- AUC
- Brier
- ppf
- one-class days
- +Recall=0 days
- -Recall=0 days

Windows:
W1-W4 Raw/Balanced/AUC/+R/-R
min-window Raw
window std

Hours:
H1/H2/H3 Raw/Balanced/AUC/+R/-R/Brier

Paired day-cluster bootstrap:
- P1-P0
- P2-P1
- P3-P1
- P2-P0
- P3-P0
- P3-P2
95% CI + W/T/L

Slot gains:
overall + H1/H2/H3.

Calibration diagnostics:
- Brier
- mean p
- true positive prevalence
- calibrator base Raw/Balanced/AUC/Brier
- calibrator post Raw/Balanced/AUC/Brier
- coefficient distributions across 28 days
- coefficient sign consistency
- condition number / finite audit

## 12. Safety

Production-relevant anchor = P0 original Q2.

A candidate is SAFETY_ADMISSIBLE iff:
- Balanced >= P0-.01
- +Recall >= P0-.05
- one-class days <= P0+2
- +Recall=0 days <= P0+2

Also report P2/P3 relative to P1 because P1 isolates the calibrator-holdout split effect.

## 13. Pre-registered interpretation

POSTPROCESS_SIGNAL exactly one:

REGIME_STACKING_CONFIRMED
- P3 safety-admissible;
- P3 materially/stably improves P1;
- P3 also beats or matches P0 on pooled Raw and cross-window evidence.

SEGMENT_CALIBRATION_HELPFUL
- P2 improves P1 materially/stably;
- P3 does not clearly exceed P2.

REGIME_PROMISING_UNPROVEN
- P3 gains >= ~2pp or reaches >=.62 with safety, but CI/cross-window evidence not established.

SPLIT_COST_DOMINATES
- P1 loses materially vs P0 and P2/P3 fail to recover the loss.

POSTPROCESS_HARMFUL
- P2/P3 materially worsen P1 or fail safety.

NO_POSTPROCESS_GAIN
- P2/P3 do not materially improve P1/P0.

## 14. Acceleration rules

If P3 or P2 >= .62 Raw with safety:
- immediately run 3-seed confirmation of P0 + best candidate;
- do not execute any further postprocess tuning first.

If confirmed >= .62:
- freeze Q2 + postprocessor candidate;
- broaden DEV before any lockbox;
- then decide whether corrected Stage B is still needed.

If confirmed >= .63:
- broader DEV is priority; no more architecture micro-ablation.

If no postprocess gain:
- STOP postprocessing.
- next = E4-B selector reopening, not more calibrator/L2/threshold sweeps.

## 15. E4-B trigger if E4-A fails

Do not execute in E4-A.

E4-B will test whether frozen XGB selector removed neural/regime-useful candidates.

37 selector-dropped Noise features include:
- second-order ramp_load/solar/renewable/residual_load
- renewable-adjusted direct-load error/delta statistics
- high/low residual-load / renewable-share / bidding-space regime flags
- several forecast-error quantiles

E4-B should compare:
- selected222 anchor
- all259
- literature/regime reintroduced subset
with matched controls and no leakage.

## 16. Forbidden

No new raw data source.
No selector rerun.
No adding arbitrary new features.
No threshold tuning.
No class weighting.
No L2/C search.
No calibrator-days sweep.
No Stage B.
No architecture changes.
No PLE/k/depth/width/fusion/gate changes.
No 24-head model.
No target-day oracle routing.
No broader DEV/lockbox in E4-A.

## 17. Outputs

experiments/first_test/E4_regime_stacker/

- 00_PLAN_SNAPSHOT.md
- benchmark/
- runs/
- split_audit.csv
- checkpoint_identity.csv
- calibrator_metrics.csv
- daily_metrics.csv
- model_summary.csv
- window_metrics.csv
- hour_segment_metrics.csv
- paired_summary.csv
- slot_gain_summary.csv
- stacker_coefficients.csv
- stacker_audit.json
- runtime.csv
- E4_A_summary.md
- E4_A_GATE.md
- figures/

After E4-A STOP for human review.

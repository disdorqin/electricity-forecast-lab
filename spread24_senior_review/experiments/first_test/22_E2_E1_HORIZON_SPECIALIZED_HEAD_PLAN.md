# E2-E1 — Horizon-Specialized Direction Readout

STATUS=AUTHORIZED_NEXT
DATE=2026-09-26
PARENT=E2-D1
EXPERIMENT_ROOT=experiments/first_test/E2_architecture/horizon_specialized_head
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS
ACCELERATION_GOAL=BREAK_60_AND_TEST_PATH_TO_62_65

## 0. Why pivot here

E2-D1 Gate = NO_CLEAR_ENCODING_EFFECT.

N0 canonical PLE+raw remains the screening anchor:
Raw=.5893 / Balanced=.5372 / AUC=.5671.

RAW_ONLY and PLE_ONLY do not establish an improvement and both fail the frozen safety screen.
Therefore do not spend the next round on PLE bins/dim, k, depth or width.

The strongest unresolved architecture asymmetry is now at the readout:

Current V2.1 uses ONE shared member-wise Direction head across all 24 horizons:
h_dir [B,k,24,d_task]
-> reshape B*24
-> one tabm.LinearEnsemble(d_task,1,k)
-> logits for every hour.

But N0 segment performance is strongly heterogeneous:
H1 1-8: Raw .5000 / Balanced .4714 / AUC .4919
H2 9-16: Raw .6607 / Balanced .5545 / AUC .5751
H3 17-24: Raw .6071 / Balanced .5719 / AUC .6136

This creates a high-value hypothesis:
one shared decision boundary may be forcing incompatible hour regimes to share calibration and/or weights.

The 1-8 / 9-16 / 17-24 segmentation is PRE-EXISTING and frozen before this experiment.
Do not discover new segments from E2-E1 results.

## 1. Fixed architecture context

All arms use:
- canonical numeric encoding PLE+raw
- all 211 Strong/Core + 11 Weak
- current 24-h horizon gate
- full_current Tabular+Temporal
- fixed direction_fusion_alpha=.8
- k=8
- Stage A
- dir_only / direction_first / vanilla
- same frozen source/selector/preprocessing/data contract

Only Direction readout changes.

Magnitude readout remains canonical and is not interpreted as capability in dir_only.

## 2. Arms

### Q0 SHARED_HEAD — anchor
Reuse N0/R0/G0/F80.

Current:
one shared LinearEnsemble Direction head for all 24 horizons.

### Q1 SEGMENT_BIAS
Fresh 28 days.

Keep the same shared Direction head weights.
Add exactly 3 learnable scalar logit offsets:
b_H1, b_H2, b_H3
initialized at 0.

For each member logit z:
hours 1-8: z += b_H1
hours 9-16: z += b_H2
hours 17-24: z += b_H3

Biases are applied BEFORE sigmoid/member averaging.

Purpose:
test whether the main issue is segment-specific class prior / calibration while preserving one common decision boundary.

No threshold tuning.
Readout threshold remains p>=.5.

### Q2 SEGMENT_HEADS
Fresh 28 days.

Replace the single Direction head with exactly three independent member-wise LinearEnsemble heads:
head_H1 for hours 1-8
head_H2 for hours 9-16
head_H3 for hours 17-24

Each receives the same h_dir representation for its segment.
Each preserves k=8 member semantics.
Concatenate logits back into canonical [B,k,24] order.

Purpose:
test whether the three time blocks require genuinely different learned decision boundaries.

Do NOT create 24 separate heads in E2-E1.

## 3. Clean code design

Add experiment-only:
direction_readout_mode =
- shared
- segment_bias
- segment_heads

Default=shared.

Preferred implementation is isolated inside DualBranchV21 Direction readout.
Do not alter encoders/adapters/selector/preprocessing.

For segment_bias:
- canonical direction_head remains the only weight matrix.
- register 3 scalar parameters only.
- map fixed hours [1..8],[9..16],[17..24].
- result must expose the three selected biases for diagnostics.

For segment_heads:
- register 3 LinearEnsemble(d_task,1,k=k) modules.
- do not additionally keep/train the shared direction_head for Direction.
- build z with exact segment slicing.
- output p_members/p_hat/direction_hat in existing shapes.

Magnitude head unchanged.

## 4. Fail-closed rules

Non-shared direction_readout_mode requires:
- legacy_v20=False
- objective_mode=dir_only
- architecture_mode=full_current
- direction_tabular_mode=current
- direction_horizon_gate_mode=current
- strong_role_profile=all
- numeric_encoding_mode=canonical
- direction_fusion_alpha=.8
- train_mode=stage_a

Reject incompatible combinations.

Canonical shared mode must remain bit-identical.

## 5. Gate A

Before formal runs:

1. direction_readout_mode=shared reproduces N0/F80 benchmark <=1e-9.
2. default/no flag exact.
3. Q1 contains exactly 3 learnable segment biases, initialized zero.
4. Q1 shared head weights remain one shared module.
5. Q2 contains exactly 3 independent Direction heads.
6. Q2 has no accidental cross-segment indexing; each hour is produced by exactly one head.
7. k=8 and [B,k,24] logits preserved.
8. p_hat/direction threshold logic unchanged.
9. Magnitude path unchanged.
10. source/config/selector/sequence/leakage frozen.
11. canonical 94 + all previous E2 + focused tests PASS.

## 6. Benchmark

ARCH_BENCHMARK_DAY=2026-02-13.
Engineering only; never rank/promote from it.

Q0 reuse.
Q1/Q2 fresh.

Record:
- metrics
- params/trainable params
- runtime/GPU
- best/stop epoch
- Q1 bias trajectory
- Q2 per-head parameter norms/gradient norms.

## 7. Formal DEV panel

Same frozen 28 days:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

Q0 reuse.
Q1 fresh 28.
Q2 fresh 28.
No intermediate stopping.

## 8. Training protocol

Fresh Q1/Q2:
dir_only
direction_first
vanilla
full_current
current tabular route
current horizon gate
strong_role_profile=all
numeric_encoding_mode=canonical
direction_fusion_alpha=.8
Stage A
default profile
seed=20260924
k=8
d_task=32
max_epochs=120
patience=15
batch_size=64
frozen LR/WD
CUDA+AMP
Stage B OFF

Only Direction readout changes.

## 9. Metrics

Overall:
Raw / Balanced / +Recall / -Recall / AUC / Brier / ppf
one-class / +R=0 / -R=0.

Windows:
W1-W4 Raw/Balanced/AUC/+R/-R
min-window Raw
window std.

Segments:
H1/H2/H3 Raw/Balanced/AUC/+R/-R/Brier.
This is PRIMARY diagnostic for this experiment.

Paired:
Q1-Q0
Q2-Q0
Q2-Q1
day-cluster bootstrap 95% CI + W/T/L.

Also report slot gains:
number of additional correct slots vs Q0 overall and by H1/H2/H3.

## 10. Safety

Anchor Q0.

SAFETY_ADMISSIBLE iff:
Balanced >= Q0-.01
+Recall >= Q0-.05
one-class days <= Q0+2
+R=0 days <= Q0+2.

Any Raw gain with collapse is rejected.

## 11. Pre-registered interpretation

READOUT_SIGNAL exactly one:

SEGMENT_CALIBRATION_HELPFUL
- Q1 is safety-admissible and materially/stably improves Q0;
- Q2 does not clearly exceed Q1.
Interpretation: shared representation/boundary is adequate; segment-specific bias/calibration is the missing piece.

SEGMENT_BOUNDARY_HELPFUL
- Q2 is safety-admissible and materially/stably improves Q0 and clearly exceeds Q1.
Interpretation: H1/H2/H3 require different decision boundaries.

HETEROGENEITY_PROMISING
- Q1 or Q2 gains >= ~2pp pooled Raw / meaningful slot count and safety, but paired/cross-window evidence is not yet established.
Next = 3-seed confirmation before any promotion.

SHARED_HEAD_SUFFICIENT
- neither specialized arm materially/stably improves Q0 and both are admissible/close.

SPECIALIZATION_HARMFUL
- specialized arms are materially/stably worse or fail safety.

NO_CLEAR_READOUT_EFFECT
- differences remain too noisy.

Do not call 62/63/65 achieved unless the actual 28-day metric reaches it and safety is acceptable.

## 12. Acceleration rule

This experiment is deliberately a high-impact architecture test.

If Q2 or Q1 reaches >=.61 Raw with safety:
- immediately run 3-seed confirmation on Q0 + winner before more architecture micro-ablation.

If confirmed >=.61 but <.62:
- next move is E3 training strategy (Stage A/B), not more readout micro-tuning.

If a confirmed candidate reaches >=.62:
- freeze it as the architecture candidate and move directly to E3 Stage A/B / broader DEV.

If E2-E1 fails to improve materially:
- stop architecture micro-ablation.
- move to E3 training strategy and parallel E4 feature/regime-signal work.
- do NOT continue PLE bins/k/depth sweeps by default.

## 13. Why this can plausibly move more than previous axes

Current Q0 has:
H1 ~50%
H2 ~66%
H3 ~61%.

A single shared head is trained across these different distributions.
Segment specialization can change both bias and boundary without changing the legal input set.
It is a low-parameter, high-structural-impact modification.

This is not a promise of 62/65.
It is the next architecture test with the strongest current evidence-to-impact ratio.

## 14. Forbidden

No 24 independent heads.
No new segment boundaries.
No per-month heads.
No oracle routing.
No threshold/calibration post-hoc tuning.
No PLE bins/dim.
No k/depth/width.
No role filtering.
No gate/fusion sweep.
No Temporal/FFT change.
No Stage B in E2-E1.
No broader DEV/lockbox.

## 15. Outputs and STOP

experiments/first_test/E2_architecture/horizon_specialized_head/

- 00_PLAN_SNAPSHOT.md
- benchmark/
- runs/
- daily_metrics.csv
- model_summary.csv
- window_metrics.csv
- hour_segment_metrics.csv
- paired_summary.csv
- slot_gain_summary.csv
- readout_diagnostics.csv/json
- runtime.csv
- E2_E1_summary.md
- E2_E1_GATE.md
- figures/

After E2-E1 STOP for human review.

# E2-D1 — Numeric Encoding Big-Block Ablation

STATUS=AUTHORIZED_NEXT
DATE=2026-09-26
PARENT=E2-C3
EXPERIMENT_ROOT=experiments/first_test/E2_architecture/numeric_encoding
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## 0. Frozen interpretation before this experiment

E2-C3 Gate:
ROLE_SIGNAL=NO_CLEAR_ROLE_EFFECT.

Human review:
- DROP_MAG (remove 22 Strong-MAG) is not promising: Raw -1.34pp vs R0, fails Balanced/+Recall safety, 3/4 windows non-positive.
- DROP_DIR is safety-admissible but has only +0.30pp pooled Raw, paired CI [-4.17,+4.61]pp and unstable window signs.
- Therefore selector Strong-DIR/Strong-MAG labels must NOT be promoted into task-specific neural routing.
- The pre-registered matched-random-drop/3-seed confirmation was only required for a promising DROP_MAG result; that trigger was not met.
- Retain all 211 Strong/Core + 11 Weak for the next structural study.
- Retain current 24-h horizon gate as controlled screening architecture.
- Retain fixed Direction fusion alpha=.8 to avoid learnable-alpha confounding.

Also note:
E2-C3 summary had stale copied E2-C2 prose; report generator was corrected. Formal metrics/Gate/routing artifacts were unaffected.

## 1. Why numeric encoding is next

Canonical tabular encoding currently uses:
- PLE enabled
- ple_bins=16
- ple_embedding_dim=8
- raw value concatenated to the 8-D PLE representation

Thus every selected scalar feature enters the downstream tabular branches as 9 channels.

For the 211 Strong/Core features this creates an input width of:
211 * 9 = 1899
before the TabM backbone.

The current training sample budget is small and selected checkpoints are typically very early.
Therefore the next architecture question is fundamental:

> Is the high-dimensional piecewise-linear representation actually adding stable Direction signal, or is raw-only / PLE-only simpler and better?

This is a representation-architecture test, not a bin-count or embedding-dimension sweep.

## 2. Arms

All arms:
- all 211 Strong/Core + 11 Weak
- current 24-h learnable horizon gate
- fixed direction_fusion_alpha=.8
- same Temporal branch
- k=8, d_tab=128, n_blocks=2

### N0 CURRENT_PLE_RAW — anchor
Reuse R0/G0/C0/F80.

Encoding per feature:
[PLE_8D, raw] => 9 channels.

### N1 RAW_ONLY
Fresh 28 days.

Encoding per feature:
[raw] => 1 channel.

No PLE representation reaches either Strong or Weak branch.

Question:
Does PLE add useful nonlinear representation, or only capacity/variance?

### N2 PLE_ONLY
Fresh 28 days.

Encoding per feature:
[PLE_8D] => 8 channels.

The raw skip channel is removed.

Question:
Is raw identity information required in addition to PLE, or is PLE itself sufficient?

No other encoding arm.
Do not change bins or PLE embedding dimension in E2-D1.

## 3. Clean experiment-only implementation

Preferred experiment-only axis:

numeric_encoding_mode =
- canonical
- raw_only
- ple_only

Default=canonical.

Canonical must preserve existing behavior:
if cfg.ple_enabled=True:
  encoded = concat(PLE, raw)
if cfg.ple_enabled=False:
  encoded = raw
This preserves existing tests/config semantics.

Experiment modes:
raw_only:
  encoded = raw only, irrespective of canonical PLE path.
ple_only:
  encoded = PLE only; requires canonical ple_enabled=True and valid frozen PLE bins.

Do not change preprocessing state, robust scaling, clipping or PLE bin construction.
PLE bins remain built only from the same legal Stage-A BASE_TRAIN scope.

The same encoding mode applies to Strong and Weak inputs. This is intentional: E2-D1 asks about the numeric representation as a whole.

## 4. Fail-closed rules

Non-canonical numeric_encoding_mode requires:
- legacy_v20=False
- objective_mode=dir_only
- architecture_mode=full_current
- direction_tabular_mode=current
- direction_horizon_gate_mode=current
- strong_role_profile=all
- direction_fusion_alpha=.8
- train_mode=stage_a

For ple_only:
- cfg.ple_enabled must be true.

Reject incompatible combinations.

Canonical production/default path must remain unchanged.

## 5. Gate A

Before formal runs:

1. numeric_encoding_mode=canonical reproduces R0/F80 benchmark <=1e-9.
2. default/no flag reproduces canonical behavior.
3. N1 encoded_feature_dim=1 for Strong and Weak.
4. N2 encoded_feature_dim=8; raw channel absent.
5. N0 encoded_feature_dim=9 under canonical config.
6. Frozen PLE bins/hashes unchanged.
7. Strong/Core counts remain 211; Weak remains 11.
8. current 24-h gate, Temporal and k=8 remain live.
9. finite shapes/output; no target leakage change.
10. canonical 94 + all prior E2 tests + focused E2-D1 tests PASS.

## 6. Benchmark

ARCH_BENCHMARK_DAY=2026-02-13.

Engineering only:
N0 reuse; N1/N2 fresh.

Record:
- metrics
- parameter count
- encoded dimensions
- convergence
- runtime/GPU
- gradient ownership

Do not rank from benchmark day.

## 7. Formal DEV

Same frozen 28-day panel:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

N0 reuse R0/F80.
N1 fresh 28.
N2 fresh 28.

No intermediate stopping.

## 8. Training protocol

Fresh N1/N2:
- objective_mode=dir_only
- checkpoint_policy=direction_first
- gradient_policy=vanilla
- architecture_mode=full_current
- direction_tabular_mode=current
- direction_horizon_gate_mode=current
- strong_role_profile=all
- direction_fusion_alpha=.8 fixed
- Stage A
- profile=default
- seed=20260924
- k=8
- d_tab=128
- n_blocks=2
- dropout=.05
- max_epochs=120
- patience=15
- batch_size=64
- frozen LR/WD
- CUDA+AMP
- Stage B OFF

Only numeric encoding changes.

## 9. Metrics

Overall:
Raw / Balanced / +Recall / -Recall / AUC / Brier / ppf
one-class / +Recall=0 / -Recall=0.

Stability:
W1-W4 Raw/Balanced/AUC/+R/-R
min-window Raw
window Raw std.

Hours:
H1/H2/H3 Raw/Balanced/AUC/+R/-R.

Paired:
N1-N0
N2-N0
N1-N2 descriptive
day-cluster bootstrap 95% CI + W/T/L.

Architecture:
- encoded feature dim
- TabM input dim
- total/trainable params
- runtime/GPU
- best/stop epoch

## 10. Safety

Anchor N0.

SAFETY_ADMISSIBLE iff:
Balanced >= N0-.01
+Recall >= N0-.05
one-class days <= N0+2
+Recall=0 days <= N0+2

All arms reported.

## 11. Pre-registered interpretation

ENCODING_SIGNAL exactly one:

RAW_SUFFICIENT
- N1 safety-admissible and matches N0 without material/stable loss;
- N2 does not establish a clear advantage.
Interpretation: PLE adds no established Direction value; raw-only becomes simplification candidate.

PLE_REQUIRED
- N1 materially/stably worse than N0 while N2 is closer/admissible.
Interpretation: nonlinear PLE representation is important.

RAW_SKIP_REQUIRED
- N2 materially/stably worse than N0 while N1 is closer/admissible.
Interpretation: raw identity channel contributes information not captured by PLE.

PLE_RAW_SYNERGY
- both N1 and N2 materially/stably worse than N0.
Interpretation: PLE and raw skip are complementary.

RAW_ONLY_BETTER
- N1 materially/stably improves over N0 and is safety-admissible.
Interpretation: current PLE likely over-complex/harmful; raw-only candidate requires 3-seed confirmation before promotion.

PLE_ONLY_BETTER
- N2 materially/stably improves over N0 and is safety-admissible.
Interpretation: raw skip may interfere; PLE-only candidate requires confirmation.

NO_CLEAR_ENCODING_EFFECT
- differences remain small/noisy.

Material/stable language requires paired CI/cross-window support; pooled score alone is insufficient.

## 12. Follow-up rules

If RAW_ONLY_BETTER or RAW_SUFFICIENT:
next E2-D2 = raw-only capacity study:
- k/member axis
- n_blocks/depth
- d_tab width
one axis at a time.

If PLE_REQUIRED / PLE_RAW_SYNERGY:
retain PLE and next E2-D2 tests capacity before any PLE bins/dim micro-tuning.

If PLE_ONLY_BETTER:
confirm PLE-only with >=3 seeds before capacity study.

If NO_CLEAR_ENCODING_EFFECT:
retain canonical PLE+raw for production, but move to TabM capacity study; do not tune bins.

## 13. Important anti-overfit rule

Do NOT tune:
- ple_bins
- ple_embedding_dim
- feature-specific bins
- per-window encoding
after seeing E2-D1.

First determine whether the representation family itself matters.

Any candidate improvement must later pass multi-seed confirmation before architecture promotion.

## 14. Forbidden

No role filtering.
No selector rerun.
No PLE bin/dim sweep.
No k/depth/width change in this stage.
No horizon gate tuning.
No Temporal/FFT.
No fusion redesign/alpha sweep.
No threshold/calibration.
No checkpoint change.
No Stage B.
No broader DEV/lockbox.

## 15. Outputs and STOP

experiments/first_test/E2_architecture/numeric_encoding/

- 00_PLAN_SNAPSHOT.md
- benchmark/
- runs/
- daily_metrics.csv
- model_summary.csv
- window_metrics.csv
- hour_segment_metrics.csv
- paired_summary.csv
- encoding_audit.json/csv
- runtime.csv
- E2_D1_summary.md
- E2_D1_GATE.md
- figures/

After E2-D1 STOP for human review.

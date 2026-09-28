# E2-C2 — Horizon Gate Necessity & Simplification

STATUS=AUTHORIZED_NEXT
DATE=2026-09-25
PARENT=E2-C1
EXPERIMENT_ROOT=experiments/first_test/E2_architecture/horizon_gate
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## 0. Why this experiment now

E2-C1 formal gate is INCONCLUSIVE, but the evidence is asymmetric:

C0 CURRENT_TABULAR:
Raw .589286 / Balanced .537248 / AUC .567086.

C1 STRONG_ONLY:
Raw .574405 / Balanced .521147 / AUC .5479.
C1-C0 ΔRaw=-1.49pp, CI [-4.02,+0.15]pp, W/T/L 2/21/5.
3/4 windows move in the same negative direction and C1 fails the safety bound on Balanced.

C2 WEAK_ONLY:
Raw .529762 / Balanced .5043.
C2-C0 ΔRaw=-5.95pp, CI excludes zero; 4/4 windows lower.

Therefore:
- Weak cannot carry Direction alone.
- Strong carries most Direction signal.
- C0 consistently trends above Strong-only, so Weak is a plausible residual/correction source, but the frozen gate is not strong enough to formally label WEAK_RESIDUAL.
- Current horizon gate starts at .8 and barely moves:
  mean .80114, range across all selected checkpoints .78797–.80938.
- Effective Weak norm fraction is only ~6.54%.
- L_dir gradient norm to horizon gate is tiny (~.0062) compared with Strong (~.316) and Temporal (~.797).

The next clean structural question is:

> Does the 24-dimensional learnable horizon gate contribute anything, or is a much simpler fixed/global gate sufficient?

This is a structure test, not a coefficient sweep.

## 1. Screening anchor and fixed context

Use E2-C1 C0 / E2-B1 F80 as G0 anchor:
- direction_fusion_alpha = .8 fixed
- current Strong+Weak representation
- Stage A / dir_only / direction_first
- same 28-day DEV panel

Reason:
fixed temporal fusion removes alpha-learning confounding.
E2-C2 changes only the Strong/Weak horizon-gate mechanism for Direction.

Magnitude remains canonical H_current in every arm.

## 2. Arms

### G0 CURRENT_24H_LEARNABLE
Reuse C0/F80.
Direction tabular representation:
g_h is the existing 24-vector sigmoid(horizon_gate_logits), initialized .8 and learnable.

No fresh 28-day run.

### G1 FIXED_08
Fresh 28 days.

Direction tabular representation:
H_tab_dir = .8 * H_strong + .2 * broadcast(H_weak)

Requirements:
- canonical horizon_gate_logits must NOT receive L_dir gradient.
- no new learnable Direction gate parameter.
- Magnitude still uses canonical learnable H_current.

This tests whether the learned 24-h gate is needed at all.

### G2 GLOBAL_LEARNABLE
Fresh 28 days.

Introduce one experiment-only scalar Direction gate:
g_global = sigmoid(a_global), initialized at .8.

Direction:
H_tab_dir = g_global * H_strong + (1-g_global) * broadcast(H_weak)

Requirements:
- one scalar shared across all 24 horizons.
- canonical horizon_gate_logits must NOT receive L_dir gradient.
- a_global must receive L_dir gradient.
- Magnitude still uses canonical H_current.

This tests whether the 24 separate horizon parameters can be regularized to one global learnable scalar.

### Context endpoint
Reuse C1 STRONG_ONLY as G100_STRONG endpoint for descriptive context only.
It is not a fresh E2-C2 arm and does not enter a new promotion rule as an independently tuned model.

No WEAK_ONLY rerun.

## 3. Targeted code design

Existing E2-C1 refactor already exposes:
H_strong, H_weak, canonical gate, H_current.

Add one experiment-only axis, e.g.:

direction_horizon_gate_mode =
- current
- fixed_08
- global_learnable

Default=current.

Rules:
- non-current gate mode requires direction_tabular_mode=current.
- for this experiment, non-current mode requires architecture_mode=full_current and direction_fusion_alpha=.8.
- reject incompatible combinations fail-closed.
- canonical/default behavior must remain bit-identical.

Preferred implementation:
construct h_tab_direction inside DualBranchV21 from forward_components().
Do not alter canonical H_current used by Magnitude.

For global_learnable:
register exactly one new Direction-only parameter, initialized logit(.8).
Expose it in run manifest/history for audit.

Do not modify selector/preprocessing/data source.

## 4. Gate A exactness and ownership

Must PASS before formal runs:

1. direction_horizon_gate_mode=current reproduces C0/F80 benchmark exactly <=1e-9.
2. default/no flag reproduces canonical behavior.
3. G1 fixed_08:
   - L_dir reaches Strong and Weak.
   - canonical horizon_gate_logits receive zero/no L_dir gradient.
   - no global gate parameter exists/is active.
4. G2 global_learnable:
   - L_dir reaches Strong and Weak.
   - canonical horizon_gate_logits receive zero/no L_dir gradient.
   - global scalar receives finite nonzero L_dir gradient.
5. Temporal Direction path remains live in all arms because alpha_time=.8.
6. L_mag reaches canonical Strong, Weak and canonical horizon gate in all arms.
7. shapes finite, k=8 preserved.
8. canonical non-destructive 94 PASS and all previous E2 tests remain green.
9. config/source/selector/leakage hashes unchanged.

## 5. Benchmark

ARCH_BENCHMARK_DAY=2026-02-13.

Run G0/G1/G2 for engineering only:
- exact routing
- convergence
- scalar/gate trajectory
- runtime
- memory
- gradient ownership

Never rank from one day.

## 6. Formal DEV panel

Same frozen 28 days:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

G0 reuse C0/F80.
G1 fresh 28.
G2 fresh 28.
G100 context reuse C1 Strong-only.

No intermediate stopping.

## 7. Training protocol

Fresh G1/G2:
- objective_mode=dir_only
- checkpoint_policy=direction_first
- gradient_policy=vanilla
- architecture_mode=full_current
- direction_tabular_mode=current
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

Only horizon-gate mechanism changes.

## 8. Metrics

Overall:
Raw / Balanced / +Recall / -Recall / AUC / Brier / ppf
one-class days / +Recall=0 / -Recall=0

Stability:
W1-W4 Raw/Balanced/AUC/+R/-R
min-window Raw
window Raw std

Hours:
H1/H2/H3 Raw/Balanced/AUC/+R/-R

Paired:
G1-G0
G2-G0
day-cluster bootstrap 95% CI + W/T/L.

Also report G1-G2 descriptive paired comparison.

## 9. Gate diagnostics

G0:
reuse full 24-value gate diagnostics.

G1:
record fixed .8 and prove no Direction gate gradient.

G2:
record scalar trajectory per epoch/run:
init/final/min/max/delta
selected checkpoint global gate distribution across 28 days.

For all:
- ||g*H_strong||
- ||(1-g)*H_weak||
- effective Weak norm fraction
- Strong/Weak cosine
- L_dir gradient norms Strong/Weak/gate/Temporal.

Do not optimize from target-day diagnostics.

## 10. Safety anchor

G0=C0/F80.

SAFETY_ADMISSIBLE iff:
Balanced >= G0-.01
+Recall >= G0-.05
one-class days <= G0+2
+Recall=0 days <= G0+2

All arms still reported.

## 11. Pre-registered gate interpretation

HORIZON_GATE_SIGNAL = one of:

FIXED_SUFFICIENT
- G1 is safety-admissible;
- G1 does not show a material/stable loss vs G0.
Interpretation: 24-h learnable gate adds no established value; simplify Direction screening gate to fixed .8.

GLOBAL_SUFFICIENT
- G2 admissible and matches/beats G0;
- G1 is materially worse or G2 clearly dominates G1.
Interpretation: one global learnable scalar is enough; 24-h gate unnecessary.

GLOBAL_BETTER
- G2 shows material/stable improvement vs G0 while admissible.
Interpretation: per-horizon gate is over-parameterized/poorly optimized; promote global-gate candidate for later seed confirmation.

HORIZON_SPECIFIC_USEFUL
- G0 materially/stably outperforms both G1 and G2 with safety preserved.
Interpretation: small per-horizon gate movements are genuinely useful; retain 24-h gate.

GATE_LEARNING_HARMFUL
- G1 fixed .8 materially/stably beats G0 and is admissible.
Interpretation: current gate learning is harmful; fixed gate candidate proceeds.

NO_CLEAR_GATE_EFFECT
- differences remain too small/noisy to establish.

Important:
If both G1 and G2 are statistically indistinguishable from G0 and admissible, prefer the simpler structure as an engineering candidate, but do not claim performance improvement.

## 12. What happens next — planning only

If FIXED_SUFFICIENT / GLOBAL_SUFFICIENT:
simplify gate for architecture research, then E2-C3 task-aware Strong/Core role routing.

If GLOBAL_BETTER / GATE_LEARNING_HARMFUL:
candidate gets a later 3-seed confirmation before architecture freeze.

If HORIZON_SPECIFIC_USEFUL:
keep current gate, then E2-C3 Strong/Core role routing.

If NO_CLEAR_GATE_EFFECT:
retain canonical gate conservatively for production, but use fixed .8 as the controlled architecture-screening gate; then E2-C3 role routing.

E2-C3 likely tests Direction-specific Strong role routing:
Strong-BOTH + Strong-DIR (+ Forced-Core) versus inclusion of Strong-MAG.
Do NOT execute it in E2-C2.

## 13. Forbidden

No horizon coefficient grid.
No .7/.75/.85/.9 sweep.
No Strong role split yet.
No PLE/depth/width/k.
No Temporal/FFT changes.
No new fusion form/residual adapter yet.
No threshold/calibration.
No checkpoint-policy change.
No Stage B.
No broader DEV/lockbox.

## 14. Outputs and STOP

experiments/first_test/E2_architecture/horizon_gate/

- 00_PLAN_SNAPSHOT.md
- benchmark/
- runs/
- daily_metrics.csv
- model_summary.csv
- window_metrics.csv
- hour_segment_metrics.csv
- paired_summary.csv
- gate_diagnostics.csv
- gradient_ownership.json
- runtime.csv
- E2_C2_summary.md
- E2_C2_GATE.md
- figures/

After E2-C2 STOP for human review.

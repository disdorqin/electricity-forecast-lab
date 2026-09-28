# E2-C1 — Tabular Strong/Weak Big-Block Attribution

STATUS=AUTHORIZED_NEXT
DATE=2026-09-25
PARENT=E2-B1
EXPERIMENT_ROOT=experiments/first_test/E2_architecture/tabular_big_block
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## 0. Research principle

This stage follows the agreed order:

architecture big blocks
-> fusion coarse map
-> tabular internal big blocks
-> tabular/fusion small blocks
-> architecture freeze
-> only then Stage A/B training-strategy research.

Do not preserve an existing code path merely because it is canonical.
If the experiment shows that a structural assumption is wrong, targeted experiment-only code changes are explicitly allowed and required.
However:
- frozen data/source/selector/target adapter/leakage contract must not change;
- canonical/default behavior must remain bit-identical unless a later human-reviewed promotion explicitly changes it;
- only one architecture axis may change per experiment.

## 1. Frozen evidence from E2-A / E2-B1

E2-A:
- FULL_CURRENT: Raw .578869 / Balanced .527302 / AUC .566327.
- TABULAR_ONLY: Raw .595238 but Balanced .498539 / +Recall .152893 / 16 positive-recall-zero days.
- TEMPORAL_ONLY: Raw .500000 / Balanced .474707 / AUC .497751.
- attribution gate: TABULAR_DOMINANT.
Human interpretation: Tabular is the main Direction information carrier; Temporal is weak alone but provides complementary correction.

E2-B1:
- fixed-alpha map did not establish a unique optimal scalar region.
- F80 fixed alpha_time=.8: Raw .589286 / Balanced .537248 / AUC .567086.
- FL08 learnable alpha init .8: Raw .578869 / Balanced .527302.
- F80 vs FL08 +7/672 slots is concentrated in only 3 target days, including +5 slots on 2026-08-11.
- therefore F80 is NOT promoted; no 0.7/0.75/0.85/0.9 local alpha sweep is authorized.
- fixed alpha=.8 is used in E2-C1 only as a CONTROL to remove alpha-learning as a confound.

## 2. Why fixed alpha=.8 is the E2-C1 screening anchor

E2-C1 asks only:
Where does H_tab Direction signal come from?

Using learnable alpha would allow alpha-learning to react differently to each tabular ablation, mixing two architecture axes.
Therefore all E2-C1 fresh arms use:
direction_fusion_alpha=0.8 fixed.

C0 CURRENT_TABULAR is exactly F80 and is reused, not retrained.

Any E2-C1 winner is only a screening candidate.
Before architecture promotion it must later be re-tested under canonical learnable fusion.

## 3. Current Tabular structure

Frozen selector has 222 selected features:

- Strong-BOTH: 147
- Strong-DIR: 32
- Strong-MAG: 22
- Forced-Core: 10
- Weak: 11

Current TabularEncoder:

211 Strong/Core features
 -> PLE/raw encoding
 -> official TabM strong path
 -> H_strong [B,k,24,d_tab]

11 Weak features
 -> PLE/raw encoding
 -> weak MLP
 -> H_weak [B,24,d_tab]

current horizon gate g_h:
H_tab = g_h * H_strong + (1-g_h) * H_weak

The first question is NOT whether PLE/depth/k is optimal.
The first question is whether Strong/Core, Weak, or their current gate combination carries the useful Direction signal.

## 4. Clean targeted code design

Do NOT duplicate encoders or create parallel data preprocessing.

Refactor TabularEncoder minimally so canonical behavior is preserved exactly.

Preferred design:

1. Add a component-returning method, e.g.
   forward_components(x_future)

returning:
- h_strong [B,k,24,d_tab]
- h_weak [B,24,d_tab]
- gate [24]
- h_current [B,k,24,d_tab]

2. Existing forward(x_future) must still return h_current and must be bit-identical to pre-E2-C1 canonical behavior.

3. Add experiment-only Direction tabular mode:
   direction_tabular_mode =
   current / strong_only / weak_only

4. Direction uses:
- current: h_tab_dir = h_current
- strong_only: h_tab_dir = h_strong
- weak_only: h_tab_dir = h_weak.unsqueeze(1).expand(-1,k,-1,-1)

5. Magnitude must continue to receive canonical h_current in every mode.

Therefore adapt the model/adapters cleanly, e.g. allow:
MemberWiseTaskAdaptersV21.forward(h_tab, h_time, h_tab_direction=None, ...)
where default None means canonical h_tab.

This is preferred over globally changing TabularEncoder output because E2-C1 is a Direction-path experiment and should not accidentally alter Magnitude representation.

6. direction_tabular_mode default=None/current must preserve canonical behavior exactly.

7. Do not implement hacks by zeroing tensors after the fact if they leave unintended gradients. Use structural bypass and verify gradient ownership.

Targeted source changes are allowed because the experiment requires a clean structural ablation.

## 5. Arms

All use fixed direction_fusion_alpha=.8.

C0 CURRENT_TABULAR
- reuse E2-B1 F80 28 days
- Strong/Core + Weak + current horizon gate
- no fresh training

C1 STRONG_ONLY
- Direction receives H_strong only
- Weak path cannot reach L_dir
- horizon gate cannot reach L_dir
- Temporal remains in fusion at fixed alpha=.8

C2 WEAK_ONLY
- Direction receives broadcast H_weak only
- Strong/TabM path cannot reach L_dir
- horizon gate cannot reach L_dir
- Temporal remains in fusion at fixed alpha=.8
- preserve k member axis via broadcast before member-wise adapter/head; do not collapse the head.

Magnitude remains canonical current H_tab in C0/C1/C2.

No other arm in E2-C1.

## 6. Gate A — implementation and exactness

Required:

1. direction_tabular_mode=current with fixed alpha=.8 reproduces F80 on benchmark day exactly / <=1e-9.
2. canonical default without the new flag reproduces previous canonical behavior exactly.
3. C1 L_dir reaches Strong/Core path and does NOT reach Weak MLP or horizon gate.
4. C2 L_dir reaches Weak MLP and does NOT reach Strong/TabM path or horizon gate.
5. Temporal Direction path remains live in C1/C2 because alpha_time=.8.
6. Magnitude path still reaches canonical Strong + Weak + gate in all modes.
7. output shapes finite, k preserved.
8. source/config/selector hashes and leakage contract unchanged.
9. canonical non-destructive suite 94 PASS; E2-A/B1 tests also stay green; add focused E2-C1 tests.

## 7. Benchmark

ARCH_BENCHMARK_DAY=2026-02-13.

Run C0/C1/C2 for engineering only:
- convergence
- best/stop epoch
- runtime
- GPU memory
- active Direction gradient ownership
- parameter counts
- gate/component diagnostics

Never rank/eliminate from benchmark day.

## 8. Formal DEV panel

Same fixed 28 days:

W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

C0 reuse F80.
C1 fresh 28.
C2 fresh 28.

Do not stop from intermediate windows.

## 9. Training protocol

For C1/C2:
- objective_mode=dir_only
- checkpoint_policy=direction_first
- gradient_policy=vanilla
- architecture_mode=full_current
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

No other training axis changes.

## 10. Metrics

Overall:
- Raw
- Balanced
- +Recall
- -Recall
- AUC
- Brier
- predicted positive fraction
- one-class days
- +Recall=0 days
- -Recall=0 days

Stability:
- W1/W2/W3/W4 Raw/Balanced/AUC/+R/-R
- min-window Raw
- window std

Hours:
- H1 1-8
- H2 9-16
- H3 17-24
Raw/Balanced/AUC/+R/-R

Paired:
- C1-C0
- C2-C0
day-cluster bootstrap 95% CI, W/T/L.

## 11. Tabular component diagnostics

For C0 and on deterministic diagnostic batches:

A. Selected feature counts / names by role.
B. final 24 horizon gate values at selected checkpoint, not only mean.
C. gate change from init=.8:
   mean, min, max, per H1/H2/H3.
D. component norms before current gate:
   ||g * H_strong||
   ||(1-g) * H_weak||
   effective_weak_norm_fraction
E. cosine(H_strong, broadcast H_weak), after dimensional alignment.
F. gradient norms from L_dir:
   Strong/TabM
   Weak MLP
   horizon gate

Diagnostics are explanatory, never used as an oracle to choose per-day models.

## 12. Safety anchor

C0/F80 is the screening safety anchor.

Candidate is SAFETY_ADMISSIBLE if:
- Balanced >= C0 - .01
- +Recall >= C0 - .05
- one-class days <= C0 +2
- +Recall=0 days <= C0 +2

All arms must still be reported.

## 13. Pre-registered interpretation

TABULAR_SIGNAL may be:

STRONG_DOMINANT
- C1 matches/beats C0 on Raw without material safety loss;
- C2 is weak / clearly below C1.
Next: E2-C2 Strong/Core role decomposition + PLE/horizon decision.

WEAK_RESIDUAL
- C0 is materially/stably better than C1;
- C2 is weak alone.
Interpretation: Weak branch works as residual/correction, not standalone.
Next: E2-C2 gate/residual fusion study.

WEAK_UNEXPECTEDLY_STRONG
- C2 shows non-trivial Direction skill and is competitive with C1/C0.
Next: audit selector Strong/Weak semantics and Weak feature family before changing capacity.

STRONG_WEAK_SYNERGY
- C1 and C2 both materially below C0 with C0 safety/Raw clearly better.
Next: gate/residual mechanism study.

TABULAR_MIXING_PROBLEM
- either C1 or C2 clearly improves over C0 while safety remains admissible, implying current gate combination suppresses useful branch signal.
Next: replace/refine Strong/Weak mixing.

BOTH_INTERNAL_WEAK
- C1/C2 both near chance and C0 gains are not stable enough to attribute.
Next: feature work/review.

INCONCLUSIVE
- none established.

Use paired day-bootstrap and cross-window direction to support words like materially/stably.
Do not promote from pooled Raw alone.

## 14. Important coding rule

If E2-C1 points to a real structural flaw, the next plan is allowed to require targeted source redesign.

Examples:
- if horizon gate is effectively frozen/no-op and C1 >= C0, later remove/bypass it;
- if Weak is residual-useful but current gate is poor, redesign only the Strong/Weak mixing mechanism;
- if Weak unexpectedly dominates, inspect selector role semantics and route before adding capacity.

Do not protect the current implementation from evidence.

## 15. Forbidden

No PLE on/off yet.
No split Strong-DIR/Strong-MAG/Forced-Core yet.
No TabM depth/width/k.
No new feature selection.
No Temporal group/FFT changes.
No alpha sweep.
No concat/attention/MMoE/router.
No threshold/calibration.
No checkpoint tolerance.
No Stage B.
No full DEV/lockbox.

## 16. Outputs and stop

experiments/first_test/E2_architecture/tabular_big_block/

- 00_PLAN_SNAPSHOT.md
- benchmark/
- runs/
- daily_metrics.csv
- model_summary.csv
- window_metrics.csv
- hour_segment_metrics.csv
- paired_summary.csv
- tabular_component_diagnostics.csv
- gate_values_by_run.csv
- gradient_ownership.json
- runtime.csv
- E2_C1_summary.md
- E2_C1_GATE.md
- figures/

E2_C1_GATE:
TABULAR_SIGNAL =
STRONG_DOMINANT /
WEAK_RESIDUAL /
WEAK_UNEXPECTEDLY_STRONG /
STRONG_WEAK_SYNERGY /
TABULAR_MIXING_PROBLEM /
BOTH_INTERNAL_WEAK /
INCONCLUSIVE

After E2-C1 STOP for human review.

# E4-B — Frozen Selector Reopening via Weak Feature Recovery — Plan Snapshot

STATUS=COMPLETE  DATE=2026-09-26  PARENT=E4-A  EXPERIMENT_ROOT=experiments/first_test/E4_feature_recovery
ANCHOR=E2-E1_Q2_SEGMENT_HEADS (Raw=.6086 / Balanced=.5424 / AUC=.5911 / safety PASS)
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## Why now
E4-A postprocessing failed (SPLIT_COST_DOMINATES). The next high-value question: does the frozen XGB
selector drop globally-weak features that carry conditional regime signal useful to Q2?

## Arms (formal)
- F0 SELECTED222  : reuse E2-E1 Q2; Strong211 + Weak11.  (control)
- F1 LITERATURE240: frozen 222 + EXACT 18 pre-registered literature/regime features recovered as Weak; Strong211 Weak29.
- F2 ALL259_WEAK_RECOVERY: all 259 candidates; the 37 frozen-selector Noise features recovered as Weak; Strong211 Weak48.

## Implemented mechanism (experiment-only, in-memory)
`feature_recovery_profile = selected222|literature240|all259` (default selected222).
- `build_experiment_feature_manifest` (train.py) builds an IN-MEMORY manifest from the frozen selector +
  canonical 259 registry order. Does NOT modify the frozen selector JSON on disk.
- `selected_features` follows canonical SequenceStore registry order; `selected_indices` exact.
- Recovered features are re-routed to role "Weak" only; Strong/Core path (211) untouched.
- Base `selector_sha256` preserved; a separate `experiment_feature_profile_sha256` is recorded.
- Manifest retains FROZEN status so eligibility / preprocessing / model contracts keep accepting it.
- Run-dir leaf gains `_fr240` / `_fr259` suffix to avoid collision between F1/F2 and the canonical Q2 runs.

## Fail-closed
Non-default profile requires the canonical Q2 dir_only route: dir_only / stage_a / full_current / current
tabular / current horizon gate / strong_role_profile=all / canonical encoding / segment_heads /
unweighted class weight / no postprocess / no split / fixed alpha=.8.

## Gate A (20 items, plan29 §9)
1 selected222 reproduces Q2 benchmark <=1e-9   2 default no-flag bit-identical
3 F0=222/S211/W11  4 F1=240/S211/W29/rec18  5 F2=259/S211/W48/rec37
6 canonical order  7 exact indices  8 frozen sha unchanged  9 experiment profile sha separate
10 recovered all original Noise  11 recovered Weak-only  12 all 28 days eligible  13 identical eligible-index SETS
14 preprocessing BASE-only  15 no target truth in recovered  16 source/seq/config hashes frozen
17 segment_heads + k8 preserved  18 no postprocess/class-weight/structured decoder  19 full test suite PASS  20 fail-closed.

## Metrics / Gate / Outputs
Overall, W1-W4, H1-H3, paired 10k day bootstrap, slot gains, safety (anchor F0), family diagnostics (F1
groups A/B/C), component/parameter/runtime. FEATURE_RECOVERY_SIGNAL ∈
{DOMAIN_GUIDED_RECOVERY, BROAD_SELECTOR_BOTTLENECK, RECOVERY_PROMISING_UNPROVEN, SELECTOR_ROBUST,
RECOVERY_HARMFUL, NO_CLEAR_RECOVERY_EFFECT}. After E4-B STOP for human review.

## Forbidden
No class weighting, no structured decoder/smoothing, no postprocess/stacker, no selector rerun, no new raw
source, no arbitrary engineering beyond frozen259, no threshold tuning, no alpha/gate/PLE/k/depth/width
changes, no Stage B, no recency sweep, no per-day oracle profile, no broader DEV/lockbox.

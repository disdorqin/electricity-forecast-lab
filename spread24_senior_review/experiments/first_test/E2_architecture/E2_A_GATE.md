# E2-A Gate — architecture big-block verdict

**ARCH_SIGNAL = `TABULAR_DOMINANT`**

One of the six permitted values only: TABULAR_DOMINANT / TEMPORAL_DOMINANT / SYNERGY / FUSION_PROBLEM / BOTH_WEAK / INCONCLUSIVE.

## How this was decided

The decision tree was encoded in `write_e2_a_reports.py` before results were read. CHANCE=0.5, MATERIAL=0.02 (= 2 percentage points of accuracy), stability = day-cluster bootstrap 95% CI excluding zero.

- Raw pooled: A0=0.578869, A1=0.595238, A2=0.500000
- paired delta Raw vs A0: A1-A0=+0.016369 (CI [-0.0432,+0.0744], excludes 0: False, W/T/L 13/5/10)
- paired delta Raw vs A0: A2-A0=-0.078869 (CI [-0.1339,-0.0253], excludes 0: True, W/T/L 7/4/17)
- TAB vs TEMP derived from the shared anchor: Raw gap A1-A2=+0.095238 (delta gap +0.095238); TAB clearly ahead: True; TEMP clearly ahead: False
- at-or-below chance: A0=False, A1=False, A2=True
- A1 TABULAR_ONLY matches or beats FULL and is clearly ahead of TEMPORAL -> Case T -> TABULAR_DOMINANT

## Class-collapse check (docs/10 §8: Raw must not mask class collapse)

| model | Raw | Balanced | AUC | +Recall | -Recall | predicted-pos frac | true prevalence | one-class days | +R=0 days | -R=0 days |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 FULL_CURRENT | 0.578869 | 0.527302 | 0.566327 | 0.342975 | 0.711628 | 0.308036 | 0.360119 | 3 | 7 | 1 |
| A1 TABULAR_ONLY | 0.595238 | 0.498539 | 0.563223 | 0.152893 | 0.844186 | 0.154762 | 0.360119 | 9 | 16 | 0 |
| A2 TEMPORAL_ONLY | 0.500000 | 0.474707 | 0.497751 | 0.384298 | 0.565116 | 0.416667 | 0.360119 | 0 | 4 | 1 |

- **A1 TABULAR_ONLY has Balanced Accuracy 0.498539 — at or below chance.** Its Raw must not be read as directional skill.
- **A1 TABULAR_ONLY predicts positive on only 0.154762 of slots against a true prevalence of 0.360119.** Its Raw is carried by the majority (non-positive) class: +Recall 0.152893 with -Recall 0.844186.
- **A2 TEMPORAL_ONLY has Balanced Accuracy 0.474707 — at or below chance.** Its Raw must not be read as directional skill.

## Readings that must accompany the verdict

- Primary metric (overall Raw): A0=0.578869, A1=0.595238, A2=0.500000.
- min-window Raw: A0=0.517857, A1=0.505952, A2=0.446429.
- A1 minus A0 Raw delta: +0.016369, 95% CI [-0.0432, +0.0744] — **not statistically established**.
- A2 minus A0 Raw delta: -0.078869, 95% CI [-0.1339, -0.0253] — **established and negative**.
- **The A1 - A0 Raw difference is not statistically established** (its 95% CI includes zero), so A1's nominal Raw edge over FULL_CURRENT should not be read as a real gain.
- The only established paired Raw result is **negative**: removing the tabular branch from Direction (A2 TEMPORAL_ONLY) is reliably worse than FULL_CURRENT.
- This verdict is about **attribution** — which branch carries the Direction signal — not about promoting a variant. A0 keeps the best Balanced, the best AUC and the best-calibrated predicted-positive fraction of the three.
- This verdict concerns the Direction pathway only. Magnitude is identical across all three modes by construction and was never part of the ablation.
- No day-level or month-level oracle routing was used, and no per-window structure was fitted.

## What this does not say

- It does not authorise E2-B, a fusion redesign, a new architecture axis, or any threshold tuning.
- It does not establish the architecture for the final lockbox; the 28-day panel is DEV, not the lockbox scope.
- It is not a 65% result. No variant here is being proposed for promotion on these numbers.

## Status

E2-A is complete. Stopping here — E2-B is not executed. Nothing is staged or committed.

# E2-B1 Gate

**FUSION_REGION = `NO_CLEAR_REGION`**

Decision rule (fixed in `00_PLAN_SNAPSHOT.md` before any formal run was read):

1. no SAFETY_ADMISSIBLE arm -> `NO_CLEAR_REGION`
2. best admissible arm is FL08 -> `CURRENT_FUSION_OK`
3. best admissible arm is F00 -> `ENDPOINT_TABULAR`
4. otherwise label by `alpha(best)`: ≤0.4 `TABULAR_HEAVY`, (0.4,0.7) `BALANCED_MIX`, ≥0.7 `TEMPORAL_HEAVY`; but fall back to `NO_CLEAR_REGION` when the leader is neither separated by ≥0.02 Raw from every admissible arm in another region nor supported by a paired CI vs FL08 that excludes zero.

## How the rule resolved

- admissible arms (3): F20, F80, FL08
- best admissible by Raw = F80 (Raw=0.589286, Balanced=0.537248)
- best=F80 region=TEMPORAL_HEAVY; separated from other-region admissible arms: False; paired Raw vs FL08 established positive: False
- no admissible arm is separated from its rivals and no paired evidence establishes the leader -> NO_CLEAR_REGION

## Why not `TEMPORAL_HEAVY`

Read literally, rule 4 returns `TEMPORAL_HEAVY`: the best admissible arm is F80 at
alpha=0.8. The uncertainty clause was invoked instead. The evidence:

| comparison | mean ΔRaw | in slots of 672 | CI low | CI high | CI⊂zero | W/T/L |
| --- | --- | --- | --- | --- | --- | --- |
| F80 − FL08 | +0.0104 | **+7.0** | 0.0000 | 0.0268 | True | 3/25/0 |
| F20 − F80 | -0.0045 | **-3.0** | -0.0446 | 0.0312 | True | 12/7/9 |
| F20 − FL08 | +0.0060 | **+4.0** | -0.0342 | 0.0402 | True | 14/7/7 |

- **F80 and FL08 finish at nearly the same coefficient/effective mixture, but they are not the same training path.** Their effective time-norm fractions
  are 0.5688 and 0.5735 respectively. F80 keeps alpha fixed at 0.8, while FL08 backpropagates through a learnable alpha and settles near 0.798.
  The +7-slot difference is concentrated in only three target days, so it is evidence of a small training-path effect, not established evidence that the 0.8 coefficient region is intrinsically superior.

- **The whole admissible band spans 3–7 slots out of 672.** The day-level statistic has a
  resolution of 1/24 = 0.0417 per day, an order of magnitude larger than the mean
  differences being ranked here. F80 never lost to FL08 on any day, but it tied on 25 of
  28 — a bootstrap CI whose lower bound lands exactly on 0.0000.
- **The `TEMPORAL_HEAVY` claim would rest on the endpoint structure, which is where the
  evidence is strongest but least favourable:** F100 (the temporal endpoint) is
  non-admissible (Balanced −0.0526) and sits at Raw = 0.5000 exactly, so the temporal
  branch cannot carry Direction alone. That is an argument against the endpoints, not for
  the temporal region.

So the map resolves part of the safety shape: the pure endpoints F00/F100 are non-admissible; F20 and F80/FL08 are admissible; F40/F60 miss the pre-registered Balanced threshold by tiny point-estimate margins.
It does not establish a stable ordering among the admissible operating points, which is exactly what `NO_CLEAR_REGION` records.

### Non-admissibility margins (the filter is pre-registered; these are the margins it acted on)

| arm | fails on | margin | paired CI vs FL08 |
| --- | --- | --- | --- |
| F40 | Balanced | −0.0104 vs threshold −0.0100 (shortfall 0.0004) | -0.0380 to 0.0242, includes zero |
| F60 | Balanced | −0.0111 vs threshold −0.0100 (shortfall 0.0011) | -0.0334 to 0.0290, includes zero |
| F00 | Balanced −0.0288, +Recall −0.1901, one-class +6, +R=0 +9 | fails all four criteria | -0.0474 to 0.0562 |
| F100 | Balanced −0.0526 | fails one criterion | -0.0903 to 0.0073 |

F40 and F60 are recorded as non-admissible because the filter was fixed before the runs and
is applied as written. The reader should still see that they miss by 0.0004 and 0.0011 on a
panel whose per-day resolution is 0.0417, and that neither paired CI excludes zero.
Non-admissibility here is a point-estimate verdict, not an established one.


## Readings that must accompany the verdict

- SAFETY_ADMISSIBLE arms: F20, F80, FL08.
- Non-admissible arms: F00, F40, F60, F100.
- Raw is the primary metric, but no Raw advantage is reported here without the safety filter above it. An arm with a high Raw and a collapsed class distribution is recorded as non-admissible rather than as a winner.
- `alpha` is a coefficient, not a contribution share; §10 of the summary reports the effective mixture actually in force.
- The panel is 28 DEV days over four windows. Nothing here is a lockbox result.

## What this does not say

- It does not select a production coefficient. It maps a region.
- It does not test any fusion form other than the existing convex combination.
- It does not touch thresholds, calibration, checkpoint policy or Stage B.

Status: E2-B1 is complete. Stopping here — no later stage is executed. Nothing is staged or committed.

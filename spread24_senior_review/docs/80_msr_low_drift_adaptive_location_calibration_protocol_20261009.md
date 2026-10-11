# 80 — MSR Low Drift-Adaptive Location Calibration Protocol

STATUS=AUTHORIZED_NEXT_SCIENTIFIC_AXIS
DATE=2026-10-09
PROJECT=spread24_senior_review
TARGET=Spread = RT - DA
RESOLUTION=96 points/day
FORECAST_ORIGIN=D-1 14:00 Asia/Shanghai
DECOMPOSITION=FROZEN NATURAL_TREND_T24
STATE_KAPPA=0.5 FROZEN
STATE_PERSISTENCE=0.0 FOR THIS ROUND
Z=EXCLUDED
OCTOBER_LOCKBOX=FORBIDDEN
LATE_SEPTEMBER_HOLDOUT=SEALED (2026-09-16..2026-09-30)
PRODUCTION_2_5=STRICTLY_READ_ONLY

## 0. Authority and purpose

This document is the next scientific execution authority after docs/79.

The previous two experiments established:

1. full-strength/tempered persistence cannot rescue the current Low-derived unary evidence;
2. static training-era PIT maps (GLOBAL_PIT / BLOCK_PIT_3) do not transfer to Jan-Feb 2026;
3. the Low distribution is materially shifted in Jan-Feb 2026 relative to chronological OOF training history;
4. the finite 5-quantile ladder also has a real endpoint limitation, especially when +delta lies above q90.

This round changes exactly one scientific object: **the location of the predicted Low quantile ladder through a legal rolling residual correction**. It does not retrain Low, change quantile widths, add quantiles, change T24, change kappa, activate persistence, train Z, or change High architecture.

The scientific question is:

> Is the severe POS-state collapse primarily caused by temporal location drift of the Low distribution, and can a simple D-2-safe rolling robust location correction restore state separability without retraining the Low model?

If the answer is no, stop. The next axis would then be finite-ladder/tail representation or Low model redesign, not another persistence search.

---

## 1. Independent audit finding from docs/79

The docs/79 run and implementation are accepted as a valid negative result for the pre-registered static PIT mappings. The targeted `test_cdf_unary78.py` suite was independently rerun in the correct `epf-2` environment and passed 35/35.

However, the earlier descriptive token `D1A_CDF_DIAGNOSIS=CALIBRATION_DOMINANT` must not be interpreted as proof that a stationary calibration map is the main solution. The evidence is more specific:

- Jan-Feb pooled q50 coverage = 0.3286, far below 0.5;
- Jan-Feb q50 residual mean = +23.66, indicating Low q50 is systematically too low;
- January residual mean = +30.26, February = +16.35;
- chronological OOF fold q50 coverage is much closer to nominal (roughly 0.49–0.58), with residual means around -21 to +4;
- static PIT maps fitted on 2023–2025 therefore have no reason to correct a new 2026 location shift;
- +delta lies above q90 for 29.4% of Jan-Feb rows, so finite-ladder endpoint censoring remains a secondary but material limitation.

The working diagnosis for this protocol is therefore:

`TEMPORAL_LOW_LOCATION_DRIFT + FINITE_LADDER_ENDPOINT_LIMITATION`

The two mechanisms must be separated. This round tests location drift first.

---

## 2. Frozen objects — must not change

Freeze all of the following:

- T24 operator and hash `7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7`;
- existing frozen Low model and raw Low prediction artifact;
- Low feature set and LightGBM hyperparameters;
- quantile levels `{0.10,0.25,0.50,0.75,0.90}`;
- one 672-point history contract;
- D-1 14:00 feature boundary;
- D-2 label maturity rule;
- state truth definition and `kappa=0.5`;
- finite-ladder CDF arithmetic from the uncalibrated path;
- High feature/model/reconstruction definitions;
- Z model and features;
- late-September and October protections.

Do not use GLOBAL_PIT or BLOCK_PIT_3 as active calibrators in this round. They remain historical negative controls only.

Do not search a new persistence lambda. State persistence is fixed to:

`lambda_state = 0.0`

for the full round. This deliberately separates location calibration from temporal persistence.

---

## 3. New single scientific axis: rolling location correction

Let the frozen raw Low quantiles for target day D and slot t be:

`Q_raw(q, D, t)`.

Define historical median residual:

`e(d,t) = L_true(d,t) - Q_raw(0.5,d,t)`.

For a prediction day D, only residuals whose labels are legally mature are allowed:

`d <= D-2`.

The corrected Low ladder is:

`Q_adj(q,D,t) = Q_raw(q,D,t) + b(D,t)`

for all q in the frozen five-quantile set.

Applying the same additive shift to all quantiles preserves:

- quantile ordering;
- interval width;
- ladder shape;
- the original Low model ranking structure.

Only the distribution location changes.

### 3.1 Fixed rolling window

Use exactly the most recent 28 complete mature days.

`ROLLING_DAYS = 28`

No 7/14/56-day search is authorized in this round.

If fewer than 28 complete mature days are available for a historical OOF prediction, use all available complete mature days only if at least 14 complete days exist; otherwise fail closed for that OOF fold/date. Do not backfill from future dates.

### 3.2 Pre-registered candidate families

Only three candidates are allowed:

1. `RAW_LOCATION` — no shift, b=0.
2. `ROLL28_GLOBAL` — one robust daily shift:
   `b_D = median(e(d,t))` over all 96 slots and the last 28 mature days.
3. `ROLL28_BLOCK3` — three fixed 8-hour shifts:
   - B1: p1..p32
   - B2: p33..p64
   - B3: p65..p96
   For target slot t in block b:
   `b(D,t)=median(e(d,s))` over the same block across the last 28 mature days.

No slot-wise 96-calibrator, no hour-wise calibrator, no exponential decay parameter, no learned bias model, and no tail model may be added after results are visible.

---

## 4. Why Jan-Feb are no longer a selection window

Jan-Feb 2026 were used in docs/79 to diagnose the failed static PIT approach and therefore are consumed for hypothesis generation.

They may be used only for descriptive drift plots and as legally mature historical residuals for later dates.

They must not select the new location-correction family.

The new split is:

- `D2_SELECTION = 2026-03-01 .. 2026-03-31`
- `D2_TRANSFER = 2026-04-01 .. 2026-04-28`

March chooses the correction family using Low/State metrics only.

April is a frozen-family transfer test. No family may be changed after March is opened.

May/June/July remain downstream retrospective evaluation only and are not used to choose the location-correction family.

Late September remains sealed.

---

## 5. Stage D2-A — implementation and chronology audit

Implement a new isolated module, suggested:

`rolling_low_location.py`

and an execution orchestrator, suggested:

`low_location_recovery.py`.

Required chronology behavior:

For every target day D:

1. identify mature cutoff = D-2;
2. collect only complete historical dates <= cutoff;
3. keep the most recent 28 complete days;
4. compute residuals from historical Low predictions with the correct prediction identity:
   - historical High-training folds must use chronological OOF Low predictions;
   - Jan-Apr holdout history may use the frozen out-of-sample fitted Low predictions that were generated without 2026 labels;
5. compute the candidate shift;
6. shift all five quantiles equally;
7. never use target-day truth or D-1 truth that is not mature.

Future-label perturbation after a prediction date must not alter the correction for that date.

Required audit artifacts:

- `ROLLING_HISTORY_AUDIT.json`
- `CORRECTION_SEMANTICS.json`
- per-day history start/end dates and count;
- correction value(s) per target day;
- correction semantic fingerprint;
- source/Low/T24 hashes.

No scientific selection is allowed until chronology/unit tests pass.

---

## 6. Stage D2-B — March location-correction selection

Run exactly the three pre-registered candidates on 2026-03-01..2026-03-31 with state persistence fixed at zero.

For every candidate produce both Low calibration metrics and RAW-state metrics.

### 6.1 Low metrics

Report:

- q10/q25/q50/q75/q90 empirical coverage;
- `abs(q50_coverage - 0.5)`;
- Low q50 MAE/RMSE;
- residual mean and median;
- residual by B1/B2/B3;
- quantile crossing rate after correction (must remain zero if raw ladder was monotone);
- +delta above q90 rate;
- -delta below q10 rate;
- either-threshold-outside rate.

### 6.2 State metrics, with lambda_state=0

Report:

- multiclass Brier;
- Balanced Accuracy;
- NEG/NEU/POS recall;
- predicted NEG/NEU/POS prevalence;
- true NEG/NEU/POS prevalence;
- POS prevalence absolute error;
- plain accuracy;
- mean max probability;
- entropy;
- transitions/day;
- ABA/day;
- dwell median/mean/P90.

Dwell is descriptive only in this round because persistence is frozen off.

### 6.3 March admissibility gate

RAW_LOCATION is the reference and cannot be declared a calibration success.

A non-RAW candidate is admissible only if all are true:

1. q50 coverage error improves by at least 0.05 absolute OR final q50 coverage error <= 0.05;
2. State Brier improves by at least 0.01 relative to RAW;
3. POS recall improves by at least 0.03 absolute OR POS prevalence absolute error decreases by at least 25%;
4. Balanced Accuracy delta relative to RAW >= -0.005;
5. no state class with non-zero RAW recall collapses to zero;
6. both March halves (1-15 and 16-31) have State Brier delta <= +0.02 versus RAW;
7. output probabilities are finite/normalized and quantile ladder remains monotone.

Among admissible candidates choose lowest State Brier. If Brier differs by <=0.005, prefer `ROLL28_GLOBAL` as the simpler candidate.

Seal the winner before April is read:

`LOCATION_SELECTION.json` + SHA256.

If no non-RAW candidate is admissible, stop with:

`LOCATION_CALIBRATION_SELECTION=NO_SUPPORTED_LOCATION_CORRECTION`

and do not train High.

---

## 7. Stage D2-C — April frozen-family transfer

Apply the March-selected family unchanged to 2026-04-01..2026-04-28.

The rolling correction remains day-adaptive because it is defined from each day's D-2-safe history, but the family, 28-day window and block routing are frozen.

Transfer passes only if all are true:

1. q50 coverage error is no worse than RAW and is <= 0.10 OR improves by at least 0.03;
2. State Brier improves by at least 0.005 relative to RAW;
3. POS recall improves by at least 0.02 OR POS prevalence absolute error decreases by at least 20%;
4. Balanced Accuracy delta >= -0.005;
5. no class collapse;
6. first and second half of April each have Brier delta <= +0.02;
7. finite/normalized/monotone checks pass.

If transfer fails, stop. Do not switch from GLOBAL to BLOCK or vice versa using April truth.

Required token:

`LOCATION_CALIBRATION_TRANSFER=PASS|FAIL`

---

## 8. Endpoint limitation is diagnostic only in this round

The docs/79 finite-ladder issue is real: +delta exceeded q90 for about 29% of Jan-Feb rows, and the right-continuous empirical PIT map collapsed these rows to zero POS probability.

This round does not add a tail distribution or new quantile levels.

Continue to report:

- +delta above q90 rate;
- -delta below q10 rate;
- either-threshold-outside rate;
- outer-tail probability mass under the raw finite ladder.

The active CDF after location correction is the original uncalibrated finite-ladder CDF. Therefore an x above q90 retains the original outer-tail probability floor implied by the q90 level; it is not mapped to exactly zero by an ECDF endpoint.

If location correction passes but threshold-outside rates remain material and State POS recognition remains capped, the next isolated axis may be finite-tail representation / extra quantiles. Do not implement that in this round.

---

## 9. No persistence search in this round

This protocol deliberately fixes:

`lambda_state = 0.0`.

Do not run `{0,.25,.5,.75,1}` again.

Reason: docs/77 showed persistence can match dwell while amplifying wrong unary probabilities. The current question is whether the unary evidence itself can be repaired by a legal adaptive Low location correction.

Persistence may be reconsidered only after calibrated RAW state is scientifically supported and transferred downstream.

---

## 10. Stage D2-D — chronological OOF corrected State for High training

Only after March selection and April transfer pass may downstream training start.

For each High-training OOF prediction row/date:

1. use that row's existing chronological OOF Low quantiles;
2. compute the rolling location correction using only earlier mature residuals from chronological OOF Low predictions;
3. apply the frozen selected family;
4. convert corrected Low ladder to RAW state unary with kappa=0.5;
5. use no persistence (`lambda_state=0`);
6. attach correction/state semantic fingerprint.

The correction may not use a global map fitted from later OOF periods. It is sequential and fold/date local.

Perturbing labels after a date's legal cutoff must not alter that date's corrected state.

Required outputs:

- `OOF_CORRECTION_AUDIT.json`
- `OOF_STATE_PREDICTIONS.parquet`
- per-fold correction history summaries;
- semantic fingerprints and hashes.

---

## 11. Stage D2-E — High V1a only

Train only V1a/no-Z.

At minimum produce two active arms:

1. `UNCAL_RAW_V1a` — original raw Low ladder + raw state, no persistence;
2. `ROLLCAL_RAW_V1a` — selected rolling-corrected Low ladder + corrected raw state, no persistence.

The Direct baseline remains frozen external reference.

Historical old T24 V1a may be shown only as `HISTORICAL_REFERENCE`; it has different state semantics and must not be treated as the matched active control.

Freeze all High model features/hyperparameters/quantiles/loss/seed. The only scientific difference between the two active High arms is the rolling Low location correction and the state probabilities it induces.

Do not train V1b/Z in this round.

---

## 12. Stage D2-F — reconstruction and May/June/July retrospective evaluation

Only after D2-E succeeds may final Spread metrics be opened for:

- 2026-05-01..05-31;
- 2026-06-01..06-30;
- 2026-07-01..07-31.

Report monthly and pooled metrics for:

- DIRECT_LGBM;
- UNCAL_RAW_V1a;
- ROLLCAL_RAW_V1a;
- optional historical T24 V1a reference clearly marked mismatched semantics.

Required final metrics:

- MAE, RMSE, Bias, MedianAE, P90AE;
- Direction Accuracy;
- Balanced Accuracy;
- +Recall, -Recall;
- AUC;
- Brier;
- predicted positive rate vs observed positive rate;
- High q50 MAE/pinball;
- positive-crossing recovery;
- Low q50 coverage/residual;
- State Brier/recalls/prevalence;
- slot profile and monthly stability.

### 12.1 Rolling-calibration transfer gate to final Spread

`ROLLCAL_RAW_V1a` must demonstrate that the upstream correction survives downstream High/reconstruction.

Require pooled versus `UNCAL_RAW_V1a`:

1. +Recall gain >= +0.02 OR POS prevalence error reduced by >=20%;
2. Balanced delta >= 0.00;
3. AUC delta >= -0.005;
4. -Recall delta >= -0.02;
5. MAE ratio <= 1.02;
6. Brier delta <= +0.005;
7. at least two of three months have non-negative Balanced delta;
8. worst monthly Balanced deficit >= -0.03.

If this transfer gate fails, stop and classify:

`LOW_LOCATION_CORRECTION_SPREAD_TRANSFER=NOT_SUPPORTED`.

Do not switch model/candidate based on May-July.

### 12.2 Readiness versus Direct

Only if the transfer gate passes, apply the existing docs/72 readiness principles versus DIRECT_LGBM:

- pooled Balanced gain >= +0.01;
- pooled AUC gain >= +0.01;
- +Recall delta >= -0.02;
- -Recall delta >= -0.02;
- at least one recall improves >= +0.02 OR both non-decreasing;
- MAE <= 1.05 * Direct;
- Brier <= Direct + 0.01;
- monthly stability requirements preserved.

Only then may `READY_FOR_FRESH_CONFIRM=YES`.

---

## 13. Fresh confirmation remains unchanged

Fresh holdout:

`2026-09-16 .. 2026-09-30` (15 days / 1440 slots).

If and only if `READY_FOR_FRESH_CONFIRM=YES`:

1. freeze correction family and all model identities;
2. construct label-masked confirm feature view;
3. generate Direct / UNCAL_RAW / ROLLCAL_RAW predictions;
4. verify exactly 1440 rows per arm and unique keys;
5. SHA256 seal predictions before truth is read;
6. open metrics once through a separate command;
7. never retune after opening.

October remains forbidden.

---

## 14. Mandatory tests before scientific execution

Add tests for at least:

T01 — 28-day mature-history window uses only dates <= D-2.

T02 — future-label perturbation cannot alter an earlier correction.

T03 — global correction equals explicit median residual reference.

T04 — block3 routing p1..32 / p33..64 / p65..96 is exact.

T05 — one common shift is applied to all five quantiles and preserves widths/order.

T06 — RAW_LOCATION is exact identity.

T07 — less than 14 mature complete days fails closed; 14..27 uses all available only where allowed.

T08 — historical OOF correction uses OOF Low residuals, not in-sample fitted Low.

T09 — March selection cannot read April/May+ target metrics.

T10 — April transfer cannot reselect the family.

T11 — semantic fingerprint mismatch fails before High training.

T12 — T24/Low raw artifact hashes unchanged.

T13 — late-September and October guards remain closed.

T14 — production and frozen source trees remain read-only.

Run full MSR suite after implementation.

---

## 15. Output namespace

Use a new non-overwriting namespace:

`outputs/experiments/msr_spread_v1/low_location_recovery_20261009/<run_id>/`

Recommended structure:

```text
preflight/
d2a_implementation_audit/
d2b_march_selection/
  RAW_LOCATION/
  ROLL28_GLOBAL/
  ROLL28_BLOCK3/
d2c_april_transfer/
d2d_oof_state/
d2e_high/
  UNCAL_RAW_V1a/
  ROLLCAL_RAW_V1a/
d2f_reconstruction/
retrospective/
  2026-05/
  2026-06/
  2026-07/
  pooled/
report/
fresh_confirm/   # only if authorized
```

Required artifacts include:

- `LOCATION_CANDIDATES.csv/json`
- `LOCATION_SELECTION.json` + seal
- `LOCATION_TRANSFER.json`
- `ROLLING_HISTORY_AUDIT.json`
- `CORRECTION_SERIES.parquet`
- `OOF_CORRECTION_AUDIT.json`
- `STATE_METRICS.csv`
- `MONTHLY_METRICS.csv`
- `POOLED_METRICS.csv`
- `CROSSING_DIAGNOSTICS.csv`
- `READINESS.json`
- `FINAL_REPORT.md`

---

## 16. Required acceptance tokens

Final output must include enough evidence for:

```text
INDEPENDENT_REVIEW=PASS|FAIL
LOW_REUSE_MIGRATION_GATE=PASS|FAIL
ROLLING_LOCATION_IMPLEMENTATION=PASS|FAIL
ROLLING_DAYS=28
LOCATION_CANDIDATES=RAW_LOCATION,ROLL28_GLOBAL,ROLL28_BLOCK3
LOCATION_SELECTION=<family|NO_SUPPORTED_LOCATION_CORRECTION>
LOCATION_SELECTION_USED_JAN_FEB=NO
LOCATION_SELECTION_USED_APRIL=NO
LOCATION_SELECTION_USED_FINAL_SPREAD=NO
LOCATION_CALIBRATION_TRANSFER=PASS|FAIL|NOT_RUN
STATE_KAPPA=0.5
STATE_PERSISTENCE=0.0
OOF_STATE_REGEN=PASS|NOT_RUN
HIGH_V1A_TRAIN=PASS|NOT_RUN
RECONSTRUCTION=PASS|NOT_RUN
RETRO_MAY=<metrics|NOT_RUN>
RETRO_JUNE=<metrics|NOT_RUN>
RETRO_JULY=<metrics|NOT_RUN>
LOW_LOCATION_CORRECTION_SPREAD_TRANSFER=<SUPPORTED|NOT_SUPPORTED|NOT_EVALUATED>
READY_FOR_FRESH_CONFIRM=YES|NO
CONFIRM_SEAL=PASS|FAIL|NOT_RUN
LATE_SEPTEMBER_METRICS_OPENED=YES|NO
OCTOBER_LOCKBOX_ACCESSED=NO
PRODUCTION_WRITE=NO
FULL_TESTS=<passed>/<total>
```

Every downstream NOT_RUN must state the exact upstream stop reason.

---

## 17. Stop / branch rules

### Branch A — no March candidate

If no rolling correction passes March Gate:

- stop;
- do not read April for selection;
- do not train High;
- classify the static Low predictor / finite-ladder representation as the next research target.

Recommended next axis: finite-tail representation or Low model redesign, decided only after reviewing March diagnostics.

### Branch B — March passes, April transfer fails

Stop. This indicates the adaptive correction itself is unstable over one month. Do not widen the rolling-window search automatically.

### Branch C — State transfer passes, final Spread transfer fails

Stop. This means correcting Low/State probabilities does not materially help the High/reconstruction task. Do not force persistence or Z back in automatically.

### Branch D — final Spread transfer passes but Direct readiness fails

Retain the mechanism as partial signal, do not open late September, and review whether the next bottleneck is High positive-crossing mapping or finite-tail probability representation.

### Branch E — all readiness gates pass

Seal and open the late-September one-shot holdout according to §13.

---

## 18. Interpretation discipline

A successful rolling correction supports the claim that Low-state failure was partly a **temporal calibration/location-drift problem**, not that the raw Low model is universally correct.

A failed rolling correction supports escalation toward either:

- finite-ladder/tail representation, if threshold-outside diagnostics remain dominant; or
- Low model signal redesign, if even legally adaptive location correction cannot recover state separability.

Do not call any predictive driver causal. Do not interpret retrospective May-July results as fresh confirmation.

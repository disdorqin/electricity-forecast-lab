# 76 — MSR T24 State-Only Scientific Experiment Protocol

STATUS=AUTHORIZED_STATE_ONLY_SCIENTIFIC_EXPERIMENT
DATE=2026-10-09
PROJECT=spread24_senior_review
TARGET=Spread = RT - DA
RESOLUTION=96 points/day
FORECAST_ORIGIN=D-1 14:00 Asia/Shanghai
DECOMPOSITION=FROZEN NATURAL_TREND_T24
VARIANT=V1a / NO-Z ONLY
STATE_KAPPA=FROZEN 0.5
STATE_SEMANTICS=TEMPERED_CHAIN_UNARY
OCTOBER_LOCKBOX=FORBIDDEN
LATE_SEPTEMBER_HOLDOUT=SEALED UNTIL RETROSPECTIVE READINESS PASSES
PRODUCTION_2_5=STRICTLY READ ONLY

## 0. Purpose

Docs/74–75 repaired the state layer and passed independent review. This experiment is the first scientific test of that repair.

The experiment asks one narrow question:

> After fixing train/inference state semantics and replacing full-strength HMM-style smoothing with an explicit tempered persistence potential, can Low-state persistence improve positive-event recognition without destroying the value/ranking gains already obtained by the T24 Low/High decomposition?

This round is NOT a model-family search. It does not modify T24, Low, High, LightGBM, quantiles, reconstruction, feature families, Z, point threshold, class weights or loss functions.

The only scientific free variable is:

`state_persistence_strength = lambda_s in {0, 0.25, 0.5, 0.75, 1.0}`.

Persistence strength is selected only from state-level development diagnostics. Final Spread metrics may not select lambda.

---

## 1. Frozen scientific object

The following are frozen before this experiment begins:

- `Spread = RT - DA`;
- `NATURAL_TREND_T24` Low/High label split;
- T24 operator SHA256:
  `7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7`;
- 672-quarter historical input;
- D-1 14:00 information boundary;
- D-2 label maturity rule;
- Low target and Low quantile model family/features/parameters;
- High target and High quantile model family/features/parameters;
- quantiles `(0.10, 0.25, 0.50, 0.75, 0.90)`;
- state `kappa = 0.5` everywhere;
- state semantic version `TEMPERED_CHAIN_UNARY_V1`;
- reconstruction/scenario coupling implementation;
- final point = reconstructed median;
- final direction probability = `P(Spread > 0)`;
- no Z mediator in the scientific arm;
- no separate Direction head or point loss.

Low calibration is still a known WARNING (`Q50 coverage ~= 0.3551`) and must only be reported, not repaired in this experiment.

---

## 2. Data windows and why these dates are used

### 2.1 Model-fit history

`TRAIN = 2023-01-01 .. 2025-12-31`

Low and High keep the existing chronological OOF protocol. No May–September target labels are used for model fitting.

### 2.2 State persistence selection window

`STATE_SELECTION = 2026-01-01 .. 2026-04-28`

`STATE_DEV_REPORT = 2026-01-01 .. 2026-04-30`

Only state-level metrics are allowed to choose `lambda_s`.

This window is already the development window and is the only place where the persistence strength may be selected.

### 2.3 Retrospective test months

After `lambda_s` is frozen, evaluate the reconstructed Spread on exactly three complete out-of-sample months:

- `RETRO_MAY = 2026-05-01 .. 2026-05-31`;
- `RETRO_JUNE = 2026-06-01 .. 2026-06-30`;
- `RETRO_JULY = 2026-07-01 .. 2026-07-31`.

These months are selected because they expose different known regimes while remaining fully retrospective and already opened:

- May is the regime where the direct baseline was comparatively strong and therefore tests whether MSR can avoid damaging an easy regime;
- June is a middle regime where T24 previously improved Balanced Accuracy materially;
- July is the hardest positive-recognition regime and previously exposed POS-state collapse and weak positive-crossing recovery.

These months are `RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC`, not fresh proof. They may decide whether the repaired state mechanism is promising enough to preserve the sealed holdout, but they may not be used to go back and re-select `lambda_s`.

### 2.4 Excluded windows

Do not use for selection in this round:

- `2026-08-01 .. 2026-08-14` — already heavily consumed diagnostics and unnecessary here;
- `2026-08-15 .. 2026-09-15` — consumed prospective panel;
- `2026-09-16 .. 2026-09-30` — sealed fresh holdout;
- any `2026-10-*` labels — forbidden lockbox.

### 2.5 Sealed fresh confirmation

`FRESH_CONFIRM = 2026-09-16 .. 2026-09-30`

Exactly 15 complete days / 1440 quarter-hour slots are reserved.

Late-September truth may be opened only if the state-only retrospective readiness gates in this document pass. Until then:

`LATE_SEPTEMBER_METRICS_OPENED=NO`.

---

## 3. Preflight and Low migration gate

The experiment should not create a new Low scientific result. The frozen T24 Low artifact may be reused read-only if and only if a migration audit passes.

Reference Low run:

`20261009T025641665850Z`

Reference Low prediction SHA256:

`f7b43c1b428d9488888560e7e4f8e08f9adfe92daf84361a2c2705e4b3259050`

Required `LOW_REUSE_MIGRATION_GATE=PASS` checks:

1. prediction file SHA matches exactly;
2. run manifest is COMPLETE;
3. canonical source SHA matches `49fade43b0f3505a437e8704377bf5ef07f780429f9052caaca74aa3525e95a2`;
4. T24 operator hash matches the frozen hash;
5. `low_quantile.py`, Low feature definitions, dataset scientific columns and T24 code are unchanged relative to docs/75 protected evidence;
6. row keys are unique and complete for the required train/dev/retrospective rows;
7. no late-September or October target metric is read during migration;
8. old state/High artifacts are NOT reused as active upstreams.

If this gate fails, rerun C1_LOW deterministically under the frozen Low scientific contract and require numerical equivalence before continuing. Do not modify Low to force equivalence.

---

## 4. Stage A — state candidate construction, no High training yet

For every candidate:

`lambda_s in {0.00, 0.25, 0.50, 0.75, 1.00}`

use exactly the same Low predictions, `kappa=0.5`, fold-local state thresholds and fold-local transition matrices.

For development inference, use the same frozen holdout state parameter history as the repaired pipeline; do not extend the state transition/IQR fit into May–July during this experiment.

For each target day, calculate:

- raw `state_neg_raw/state_neu_raw/state_pos_raw`;
- tempered `state_neg/state_neu/state_pos`;
- hard predicted state for diagnostics only;
- true Low state from `L_true` and the same `kappa=0.5` threshold definition.

No High model is trained in Stage A.

### Stage A required metrics

For every lambda on `2026-01-01 .. 2026-04-28`, output:

- multiclass state Brier:
  `mean(sum_k (p_k - 1[y=k])^2)`;
- state balanced accuracy;
- NEG recall;
- NEU recall;
- POS recall;
- plain state accuracy;
- predicted class proportions;
- truth class proportions;
- transitions/day;
- ABA/day if available;
- median/mean/P90 predicted dwell;
- median/mean/P90 truth dwell;
- `dwell_ratio = predicted_median_dwell / truth_median_dwell`;
- mean maximum state probability;
- state entropy;
- semantic fingerprint;
- delta/IQR and transition-matrix provenance hashes.

Also report Jan/Feb/Mar/Apr monthly state metrics as diagnostics. Monthly results cannot alter the selection rule below.

---

## 5. Stage B — persistence admissibility and frozen winner

Use the gates already pre-registered in docs/74. Do not change them after seeing results.

Relative to RAW (`lambda=0`), a candidate is admissible only if all are true:

1. multiclass Brier <= RAW Brier + 0.005;
2. POS-state recall >= RAW POS recall - 0.01;
3. median dwell ratio is in `[0.75, 1.50]`;
4. no class with nonzero RAW recall collapses to zero recall.

Among admissible candidates:

- choose the lowest multiclass Brier;
- if Brier difference is <= 0.002, choose the smaller `lambda_s`.

Freeze one value only:

`SELECTED_STATE_PERSISTENCE = lambda*`.

The file `STATE_SELECTION.json` must be written before any May/June/July final Spread metric is computed and must contain:

- the full five-candidate state table;
- admissibility result for every gate;
- selected lambda;
- selection timestamp;
- code/config/source/T24/Low hashes;
- proof that final Spread metrics were not used for selection.

### Stop condition

If no candidate is admissible, output:

`STATE_SELECTION=NO_ADMISSIBLE_PERSISTENCE`

and STOP before High training. Do not lower the gates.

---

## 6. Stage C — regenerate chronological OOF state features

After `lambda*` is frozen, regenerate the state features consumed by High training.

For every OOF fold:

1. use the existing chronological OOF Low quantiles;
2. use only Low labels <= the fold's legal D-2 cutoff to compute IQR/delta;
3. use frozen `kappa=0.5`;
4. estimate the transition matrix from the same legal historical state labels;
5. derive raw state evidence;
6. apply the frozen `lambda*`;
7. save semantic fingerprint and cutoff hashes.

Generate three state arms for downstream scientific diagnostics:

- `STATE_RAW`: lambda=0;
- `STATE_SELECTED`: lambda=lambda*;
- `STATE_FULL`: lambda=1.

If lambda* equals 0 or 1, de-duplicate the corresponding arm but retain the label in the report.

These controls are predeclared. May–July results may not replace `lambda*` with another intermediate lambda.

---

## 7. Stage D — High training

Use **V1a / no-Z only**.

Do not train or evaluate a new V1b in this round.

Train one High quantile set for each distinct state arm from Stage C:

- `HIGH_RAW`;
- `HIGH_SELECTED`;
- `HIGH_FULL`.

Every High arm must use exactly the same:

- training rows;
- frozen High feature family;
- Low quantiles;
- LightGBM parameters;
- random seed;
- quantile levels;
- mechanism features;
- reconstruction downstream contract.

The only difference is the state columns and their semantic fingerprint.

High training uses chronological OOF Low/state features exactly as in the accepted pipeline. It must never consume true Low state.

For each arm save five quantile model files and OOF/fitted predictions.

Required High component metrics, by train-OOF/dev/May/June/July where legally reportable:

- q50 MAE/RMSE;
- pinball by quantile and averaged;
- sign accuracy of H;
- correlation;
- q10/q25/q50/q75/q90 empirical coverage;
- interval width;
- tail MAE/P90 AE;
- positive-crossing recovery diagnostics.

---

## 8. Stage E — Spread reconstruction

For every repaired arm, use the frozen reconstruction implementation.

Do not add a point head and do not directly add same-index quantiles as if exact.

Output per slot:

- `spread_q10/q25/q50/q75/q90` where available from the frozen scenario interface;
- final median point prediction;
- `P(Spread>0)`;
- Low q50;
- raw/tempered state probabilities;
- High q50;
- selected state lambda and semantic fingerprint.

The reconstruction code and scenario grid must be identical between arms.

---

## 9. Stage F — retrospective evaluation dates and comparison arms

Only after `STATE_SELECTION.json` is sealed may May/June/July Spread metrics be computed.

Primary repaired arm:

`T24_STATE_SELECTED_V1a`.

Mandatory controls:

- `DIRECT_LGBM` — frozen historical direct baseline from docs/73;
- `T24_STATE_RAW_V1a` — repaired semantics, lambda=0;
- `T24_STATE_SELECTED_V1a` — repaired semantics, lambda=lambda*;
- `T24_STATE_FULL_V1a` — repaired semantics, lambda=1;
- `T24_LEGACY_V1a` — docs/73 historical T24 V1a, marked `HISTORICAL_MISMATCHED_STATE_SEMANTICS_REFERENCE` because its High OOF state task used the old mismatch.

Do not present the legacy arm as an apples-to-apples active candidate. Its purpose is to show whether the repair changed behavior in the intended direction.

### Per-month and pooled metrics

For May, June, July separately and pooled 92 days / 8832 slots report:

- MAE;
- RMSE;
- bias;
- median AE;
- P90 AE;
- raw Direction Accuracy;
- Balanced Accuracy;
- positive recall;
- negative recall;
- AUC of `P(Spread>0)`;
- Brier;
- predicted positive rate;
- observed positive rate;
- probability calibration bins;
- day/slot counts.

Also report:

- 96-slot metric profiles;
- three dayparts `p1-32 / p33-64 / p65-96` as descriptive diagnostics;
- positive-crossing count and recovery rate;
- negative-crossing count and recovery rate;
- state POS recall and dwell by month;
- High error conditioned on true/predicted Low regime;
- paired day-clustered bootstrap for `Balanced`, `AUC` and `MAE` deltas versus Direct, using the frozen bootstrap policy where applicable.

Do not use any of these retrospective Spread results to change `lambda*`.

---

## 10. Retrospective scientific gates

The experiment has two distinct questions and therefore two gate families.

### 10.1 Mechanism gate — repaired selected arm vs Direct

`T24_STATE_SELECTED_V1a` must satisfy the existing docs/72 readiness criteria on pooled May+June+July:

1. Balanced gain >= +0.01;
2. AUC gain >= +0.01;
3. positive-recall delta >= -0.02;
4. negative-recall delta >= -0.02;
5. at least one recall improves >= +0.02 OR both recalls are non-decreasing;
6. MAE <= 1.05 * Direct MAE;
7. Brier <= Direct Brier + 0.01;
8. at most 2 of 3 months have negative Balanced delta;
9. worst monthly Balanced deficit >= -0.05;
10. no pathological predicted-positive collapse/explosion.

### 10.2 Repair-target gate — did the State repair actually fix the previous failure?

Compare the repaired selected arm with the docs/73 historical `T24_MSR_V1a` reference.

For the repair to be called `STATE_REPAIR_TRANSFER_SUPPORTED`, require:

- pooled positive recall improves by at least +0.02 absolute;
- pooled negative recall loss is no worse than -0.02 absolute;
- pooled Balanced Accuracy is not lower by more than 0.005;
- pooled AUC is not lower by more than 0.005;
- pooled MAE is <= 1.02 * historical T24 V1a MAE;
- state smoothing Brier is lower than the old full-strength smoother on the same retrospective diagnostics;
- state median dwell is materially closer to truth than the old 27–30-slot behavior.

This gate does NOT make the legacy run a fair active competitor; it only checks whether the bug/over-persistence repair solved the failure mode it was designed to solve.

### 10.3 Persistence transfer diagnostic — selected lambda vs repaired RAW

Compare `STATE_SELECTED` against `STATE_RAW` after final reconstruction.

This comparison is diagnostic only. Do not change lambda based on it.

If selected persistence materially underperforms RAW, output:

`PERSISTENCE_TRANSFER=NOT_SUPPORTED`

and do not open late September. Return for review rather than silently switching the primary arm to RAW.

A material underperformance trigger is any of:

- Balanced selected - RAW < -0.005;
- AUC selected - RAW < -0.005;
- positive recall selected - RAW < -0.02;
- MAE selected / RAW > 1.02.

---

## 11. Decision before fresh confirmation

Set:

`READY_FOR_FRESH_CONFIRM=YES`

only if all are true:

- state selection produced an admissible frozen lambda;
- mechanism gate vs Direct passes;
- repair-target gate passes;
- persistence transfer is not `NOT_SUPPORTED`;
- no leakage/provenance/runtime gate failed;
- no class collapse exists;
- Low calibration remains a documented warning but no new unplanned calibration issue invalidates the state interpretation.

Otherwise:

`READY_FOR_FRESH_CONFIRM=NO`

and STOP.

When stopped, late September remains unopened and the report must isolate the next likely axis:

- State still wrong -> persistence/state definition;
- State repaired but final result weak -> Low calibration is next priority;
- High positive crossings remain weak despite repaired state -> High shock/sensitivity mapping is next priority.

Do not implement the next patch automatically.

---

## 12. Fresh late-September confirmation, only after readiness YES

If and only if retrospective readiness passes, freeze the complete selected arm and open a separate confirm workflow for:

`2026-09-16 .. 2026-09-30`.

Before reading target labels:

1. freeze model/config/code/source/T24/Low/state/High hashes;
2. generate predictions for Direct, RAW, SELECTED and FULL controls using target-label-masked feature paths;
3. save exactly 1440 unique `(market_date, period)` predictions per arm;
4. SHA256-seal every prediction file;
5. write `CONFIRM_SEAL.json`;
6. verify no missing or duplicate slots;
7. only then execute a separate metrics-opening command.

Primary fresh comparison is `SELECTED` vs Direct. RAW/FULL are pre-sealed diagnostics and may be reported after truth opening, but may not replace the frozen primary arm.

Use the docs/72 fresh-confirm success gates unchanged:

- Balanced gain >= +0.01;
- AUC gain >= +0.01;
- +Recall delta >= -0.02;
- -Recall delta >= -0.02;
- one recall improves >= +0.02 OR both non-decreasing;
- Brier delta <= +0.01;
- MAE ratio <= 1.05;
- split 15 days into three fixed 5-day chronological blocks; at least 2/3 blocks have non-negative Balanced delta and worst block deficit >= -0.05.

Classification:

- `MSR_STATE_REPAIR_CONFIRMED`;
- `MSR_STATE_REPAIR_PARTIAL_SIGNAL`;
- `MSR_STATE_REPAIR_NOT_SUPPORTED`.

No model change is permitted after confirm metrics are opened under the same confirmation claim.

---

## 13. Required artifact structure

Use a new namespace and never overwrite docs/73 or docs/75 artifacts:

```text
outputs/experiments/msr_spread_v1/state_only_experiment_20261009/
  preflight/
    LOW_REUSE_MIGRATION.json
    SCIENTIFIC_AUTHORIZATION.json
  state_selection/
    STATE_CANDIDATES.csv
    STATE_CANDIDATES.json
    STATE_SELECTION.json
    dev_state_predictions/
  state_oof/
    raw/
    selected/
    full/
  high/
    raw/
    selected/
    full/
  reconstruction/
    raw/
    selected/
    full/
  retrospective/
    2026-05/
    2026-06/
    2026-07/
    pooled/
  report/
    MONTHLY_METRICS.csv
    POOLED_METRICS.csv
    STATE_TRANSFER.csv
    CROSSING_DIAGNOSTICS.csv
    SLOT_PROFILE.csv
    READINESS.json
    FINAL_REPORT.md
  fresh_confirm/              # create only if readiness YES
    sealed_predictions/
    CONFIRM_SEAL.json
    metrics_opened/
```

Every stage must include a manifest with upstream hashes and state semantic fingerprint.

---

## 14. Required final report

Before late September is opened, the report must contain all of the following scientific conclusions explicitly:

1. Which `lambda_s` was selected and why, with the complete five-candidate state table.
2. Whether state dwell/transition/Brier/POS recall were repaired on dev.
3. Whether the selected state transfers to May, June and July separately.
4. Whether final positive Spread recall improved versus the previous T24 V1a failure.
5. Whether Balanced/AUC/MAE gains versus Direct were preserved.
6. Whether persistence itself adds value relative to repaired RAW.
7. Whether High positive-crossing recovery improved.
8. Whether the Low calibration warning is now the dominant remaining bottleneck.
9. Whether fresh confirmation is authorized.

The report must not summarize success using Direction Accuracy alone. Always include majority-class context and predicted-positive prevalence.

---

## 15. Required acceptance tokens

Before fresh confirm:

```text
INDEPENDENT_REVIEW=PASS
LOW_REUSE_MIGRATION_GATE=PASS|FAIL
STATE_SELECTION=PASS|NO_ADMISSIBLE_PERSISTENCE
STATE_KAPPA=0.5
STATE_CANDIDATES=0,0.25,0.5,0.75,1.0
SELECTED_STATE_PERSISTENCE=<value|NONE>
STATE_SELECTION_USED_FINAL_SPREAD_METRICS=NO
OOF_STATE_REGEN=PASS|FAIL
HIGH_V1A_TRAIN=PASS|FAIL
RECONSTRUCTION=PASS|FAIL
RETRO_MAY=<metrics summary>
RETRO_JUNE=<metrics summary>
RETRO_JULY=<metrics summary>
MECHANISM_GATE=<PASS|FAIL>
STATE_REPAIR_TRANSFER=<SUPPORTED|PARTIAL|NOT_SUPPORTED>
PERSISTENCE_TRANSFER=<SUPPORTED|NEUTRAL|NOT_SUPPORTED>
LOW_CALIBRATION_STATUS=WARNING|ACCEPTABLE
READY_FOR_FRESH_CONFIRM=YES|NO
LATE_SEPTEMBER_METRICS_OPENED=NO
OCTOBER_LOCKBOX_ACCESSED=NO
PRODUCTION_WRITE=NO
```

If fresh confirm is authorized and completed:

```text
CONFIRM_SEAL=PASS|FAIL
CONFIRM_ROWS_PER_ARM=1440
LATE_SEPTEMBER_METRICS_OPENED=YES
MSR_STATE_REPAIR_CONFIRMATION=<CONFIRMED|PARTIAL_SIGNAL|NOT_SUPPORTED>
OCTOBER_LOCKBOX_ACCESSED=NO
```

---

## 16. Hard prohibitions

- Do not change `kappa=0.5`.
- Do not add persistence candidates outside the pre-registered grid.
- Do not choose lambda from final Spread metrics.
- Do not calibrate Low in this experiment.
- Do not add Z back into the primary arm.
- Do not tune the final probability threshold.
- Do not add class weighting, a Direction head, TCN, Transformer, attention or new experts.
- Do not change T24/decomposition/reconstruction.
- Do not use true Low state in High training.
- Do not modify production 2.5 or frozen historical source trees.
- Do not inspect 2026-09-16..09-30 truth before retrospective readiness is YES.
- Do not access October.
- Do not stage/commit/push unless the user separately asks.

---

## 17. Interpretation policy

This experiment can produce three important outcomes.

If an intermediate lambda is selected and survives May–July, the evidence supports the claim that Low-state persistence is useful but must be tempered.

If lambda=0 wins at state level, the correct interpretation is that the Low state may still be useful, but historical transition persistence adds no incremental value under the current information boundary.

If state metrics improve but final Spread positive recall remains weak, do not keep tuning persistence. The next investigation should move to the already documented Low calibration warning or to the High positive-crossing/shock-sensitivity mapping, depending on the component evidence.

Late September is the only remaining clean one-shot confirmation in the current canonical source. Preserve it until the retrospective mechanism is genuinely ready.

# 72 — MSR-T24 修复、阶段训练与多月份验证执行协议

STATUS=AUTHORIZED_REPAIR_THEN_CONTROLLED_RERUN
DATE=2026-10-09
PROJECT=spread24_senior_review
MODEL_FAMILY=MSR-Spread V1
SELECTED_DECOMPOSITION=NATURAL_TREND_T24
TARGET=Spread = RT - DA
RESOLUTION=96 points/day
FORECAST_ORIGIN=D-1 14:00 Asia/Shanghai
HISTORY_LENGTH=672
OCTOBER_LOCKBOX=FORBIDDEN
PRODUCTION_2_5=STRICTLY_READ_ONLY

## 0. Authority and scope

This document is the execution authority for the next MSR round.

Scientific model contract remains `docs/63_msr_spread_v1_final_code_handoff_20261007.md`.
The decomposition winner and boundary evidence remain frozen by `docs/69_boundary_safe_low_high_decomposition_adjudication_20261008.md`.
Independent audit findings are in `docs/71_d0_decomposition_audit_20261009.md`.

This round has two purposes only:

1. repair the three audit issues from docs/71 without changing the scientific winner;
2. rerun the previously defined stage-wise MSR pipeline with `NATURAL_TREND_T24`, first on already-open retrospective months, and only if readiness gates pass, perform one sealed fresh late-September confirmation.

Do not redesign the architecture, add attention/TCN/Transformer, reopen wavelet/lambda search, change quantile sets, change the Z definition, or access October labels.

---

## 1. Frozen scientific object

The Low/High label definition is fixed to:

\[
L^*=(I+\lambda D_2^TD_2)^{-1}S,
\qquad H=S-L^*,
\]

with:

- `T24` parameter id;
- `lambda = 215.32246474604284`;
- 96-point ordinary second-difference matrix;
- no wrap row;
- matrix SHA256 `7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7`.

The numerical operator MUST NOT change in this round.

### Terminology correction

The existing derivation sets:

\[
G(2\pi/T_c)=0.5.
\]

Therefore `T_c` is a **half-gain / half-amplitude period**, not a half-power period.

For T24:

- the frozen parameter id remains `T24`;
- lambda remains exactly unchanged;
- old historical artifacts remain immutable;
- all new source/doc/report wording must use `half_gain_period_quarters` or an equivalent precise term;
- if backward compatibility requires reading old `half_power_period_quarters`, treat it as a legacy alias and never rewrite frozen artifacts.

Do not rerun D0 selection because of this naming correction.

---

## 2. Gate-9 wording correction

The natural trend smoother is dense, so distant samples can have tiny nonzero influence through the chain of local penalties. The scientific requirement is NOT that every numerical `p96 -> p1` influence be exactly zero.

The correct structural requirement is:

> there is no adjacent-like circular coupling and no p96/p1 wrap adjacency in the penalty/operator definition.

For B2 this is established by:

- `D2` has exactly 94 ordinary line rows and no wrap row;
- precision-matrix corner coupling is effectively zero;
- `p96 -> p1` influence is many orders smaller than real nearest-neighbour influence;
- the operator is not translation-invariant across a same-day wrap.

Change only explanatory strings/assertion wording where required. Do not change the frozen winner rule, numerical T24 operator, or historical D0 artifacts.

---

## 3. Production isolation amendment

The production project is being modified by independent crawler/platform work. Current production Git status contains multiple changes unrelated to MSR. Therefore a whole-tree `zero drift during experiment` requirement cannot distinguish legitimate concurrent production work from MSR behavior.

This is an engineering isolation amendment, NOT a scientific metric change.

### 3.1 Required evidence before the MSR rerun

The local lab experiment is allowed only if ALL are true:

1. MSR code has zero production write call sites;
2. all experiment outputs resolve inside `spread24_senior_review`;
3. canonical training source is the lab-local parquet with frozen SHA256 `49fade43b0f3505a437e8704377bf5ef07f780429f9052caaca74aa3525e95a2`;
4. no production file is required as a runtime training/evaluation dependency;
5. no production DB write is performed;
6. October market/lockbox data are not opened;
7. the earlier four AUX/crawler drifts are classified as an external committed/concurrent workflow using Git commit evidence, not by requiring the files to remain uncommitted.

The provenance classifier must support at least:

- `EXTERNAL_UNCOMMITTED_WORKFLOW`;
- `EXTERNAL_COMMITTED_WORKFLOW`;
- `UNRESOLVED`.

A clean tracked file with a last-touching crawler commit after the docs/68 snapshot is positive evidence for `EXTERNAL_COMMITTED_WORKFLOW`, not a reason to force `UNRESOLVED`.

### 3.2 Scientific run gate

Replace the impossible whole-production-tree freeze with:

`PRODUCTION_INDEPENDENCE_GATE=PASS`

which means the experiment is self-contained in the lab and has no production write/dependency path.

Continue to report concurrent production drift if observed, but do not let unrelated crawler edits alter the lab model or artifacts.

Do not modify, revert, stage, commit, or clean the production repository.

---

## 4. Repair verification before training

Before any model is trained, complete these engineering checks:

- terminology tests for half-gain naming and legacy alias compatibility;
- Gate-9 structural tests for no wrap row / no adjacent-like circular coupling;
- provenance classifier tests for committed and uncommitted external workflow cases;
- full `src/research96/msr_spread_v1/tests` test suite;
- CLI smoke that does not train/open the fresh confirmation panel;
- source SHA and output-root assertions;
- protected old MSR artifacts remain unchanged.

Required repair token:

`REPAIR_GATE=PASS`

If any repair test fails, stop before training.

---

## 5. Controlled rerun principle

The first scientific rerun changes exactly ONE scientific item relative to the previous MSR C1 chain:

`PERIODIC_MODWT_J3 -> NATURAL_TREND_T24`.

Freeze all other first-rerun choices:

- 672-point history interface;
- Low/High feature families;
- Mechanism State features;
- A03/Z mediator definition;
- quantiles `0.10,0.25,0.50,0.75,0.90` for Low/High;
- Z quantiles `0.10,0.50,0.90`;
- LightGBM family/hyperparameters;
- OOF fold count and chronological policy;
- Markov state mechanism;
- state-kappa semantics and existing frozen selection behavior;
- reconstruction/coupling approximation;
- no independent point head/loss;
- existing explanation logic.

No parameter may be tuned from May/June/July results during this first controlled rerun.

---

## 6. Stage-wise training order

Execute strictly in this order. Each stage must save a versioned run directory, config, predictions, metrics, upstream hashes and manifests.

### S0 — T24 label construction audit

Use the frozen T24 operator only.

Verify again on training/dev rows:

- exact `S=L+H` reconstruction;
- no wrap row;
- no D+1 dependency;
- label statistics match the frozen T24 definition.

Do not perform a new decomposition search.

### S1 — C1_LOW

Train Low quantile models:

\[
X_{slow}\rightarrow Q_L(q,t).
\]

Training period remains:

`2023-01-01 .. 2025-12-31`.

Produce chronological OOF Low predictions for downstream use.

### S2 — C1_LOW_STATE

Use Low quantiles to derive CDF/state probabilities and Markov smoothing.

Preserve the existing state task semantics from the latest accepted C1_LOW_STATE implementation. Do not start a new kappa search merely because the Low labels changed.

Output raw and smoothed NEG/NEU/POS probabilities and transition diagnostics.

### S3 — C1_Z

In parallel, train the probabilistic physical mediator Z using only legal information.

Output at least:

- `z_q10`;
- `z_q50`;
- `z_q90`;
- `P(Z>0)`;
- Z interval width.

All downstream training features must use chronological OOF Z predictions, never true Z.

### S4 — C1_HIGH_NO_Z / V1a

Train signed High quantiles from fast features + predicted Low/state context, without Z.

### S5 — C1_HIGH_WITH_Z / V1b

Train the identical High branch with the additional OOF probabilistic Z summaries.

This is the formal incremental test of the A03 mediator.

### S6 — C1_RECONSTRUCTION

Reconstruct final Spread distributions from Low + High using the same previously frozen coupling approximation/interface.

Output:

- final quantiles/distribution representation;
- median point `Q_S(0.5)`;
- `P(S>0)`.

The point output has no independent point head and no independent MAE/Huber training loss.

### S7 — C1_EXPLANATION / REPORT

Produce component and driver explanations:

- Low base;
- smoothed Low regime probabilities;
- Z mediator distribution;
- High correction;
- final Spread median/probability;
- top predictive drivers.

Never label SHAP/model contributions as causal effects.

---

## 7. Baselines and controlled comparison arms

Every evaluation must contain at least these arms:

1. `DIRECT_LGBM` — frozen direct signed-Spread LightGBM baseline;
2. `PERIODIC_MSR_V1a` — old decomposition control, same stage logic, no Z;
3. `PERIODIC_MSR_V1b` — old decomposition control, with Z;
4. `T24_MSR_V1a` — new decomposition, no Z;
5. `T24_MSR_V1b` — new decomposition, with Z.

No arm receives test-specific hyperparameter tuning.

The purpose of the five-arm table is to answer separately:

- does MSR beat direct Spread modeling?
- does T24 improve on the old periodic Low/High definition?
- does predicted Z add value beyond Low/state/fast features?

---

## 8. Evaluation windows

### 8.1 Development / selection window

The historical model-development window remains:

- `DEV_SELECTION = 2026-01-01 .. 2026-04-28`;
- `DEV_REPORT = 2026-01-01 .. 2026-04-30`.

Only previously frozen selection logic may operate here. Do not create a new open-ended search.

### 8.2 Multi-month retrospective diagnostic panel

Use three complete months that are already opened historically and are out-of-sample relative to the 2023–2025 fit:

- `RETRO_M1 = 2026-05-01 .. 2026-05-31`;
- `RETRO_M2 = 2026-06-01 .. 2026-06-30`;
- `RETRO_M3 = 2026-07-01 .. 2026-07-31`.

These months are explicitly:

`RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC`.

They can show temporal stability and locate a failing component, but they are NOT fresh confirmation evidence.

Do not use 2026-08-15..2026-09-15 for selection; that prospective panel is already consumed.

### 8.3 Fresh one-shot confirmation panel

Reserve:

`FRESH_CONFIRM = 2026-09-16 .. 2026-09-30`.

Availability-only audit on 2026-10-09 confirmed 15 complete 96-point days (1440 slots) through 2026-09-30. No target metric, Spread distribution, model score or candidate comparison was opened in that audit.

Repository scan found no current MSR artifact referencing `2026-09-16` before this protocol.

Treat this panel as:

`METRIC_UNOPENED_ONE_SHOT_HOLDOUT`.

It is historical rather than live prospective data, so final wording must be "sealed fresh holdout confirmation", not "live prospective".

October remains forbidden.

---

## 9. Rule for opening the fresh confirmation panel

The AI MUST first run the complete T24 controlled chain and report May/June/July diagnostics.

Only if `READY_FOR_FRESH_CONFIRM=YES` may it generate and seal late-September predictions.

If retrospective diagnostics reveal a substantive problem, output `READY_FOR_FRESH_CONFIRM=NO` and STOP. Do not inspect late-September target metrics. Return component diagnosis to the user so the next modification can be designed jointly.

This preserves late September as a useful untouched metric holdout for a later corrected model.

---

## 10. Retrospective readiness gates

Evaluate the dev-frozen primary T24 variant against `DIRECT_LGBM` on pooled May+June+July and separately by month.

Report all metrics, but `READY_FOR_FRESH_CONFIRM=YES` requires:

1. pooled Balanced Accuracy gain >= +0.01;
2. pooled AUC gain >= +0.01;
3. pooled positive recall delta >= -0.02;
4. pooled negative recall delta >= -0.02;
5. at least one of positive/negative recall improves by >= +0.02 OR both recalls are non-decreasing;
6. final median Spread MAE <= 1.05 * direct-baseline MAE;
7. Brier <= direct-baseline Brier + 0.01;
8. monthly stability: at most 2 of the 3 months have negative Balanced delta;
9. worst monthly Balanced deficit >= -0.05;
10. no pathological predicted-positive collapse/explosion relative to observed prevalence; always report predicted positive rate and observed positive rate.

The +0.01 Balanced/AUC requirements preserve the already-frozen project convention in `REPORT_GATES`.

Raw Direction Accuracy must be reported but is NOT sufficient by itself because class imbalance can make it misleading.

The teacher's long-term Direction >=70% target remains a secondary project target, not the sole gate for this mechanism experiment.

---

## 11. Component diagnosis before any later modification

If readiness fails, do not automatically search new models. Produce a component diagnosis table:

### Low failure signature

Inspect:

- Low q50 MAE/pinball by month and slot;
- state calibration/accuracy;
- transition and dwell diagnostics;
- whether failure concentrates near zero or in a regime/month.

### Z failure signature

Inspect:

- Z q50 MAE;
- Z sign/positive-state ranking;
- `P(Z>0)` AUC/Brier;
- V1b - V1a deltas.

If V1a works and V1b consistently hurts, the mediator is not promoted; do not force it into the final model.

### High failure signature

Inspect:

- High q50 MAE/pinball;
- error conditional on ramp/tightness/Z/Low regime;
- whether positive crossings are systematically under-corrected;
- whether errors are concentrated in tails or ordinary local variation.

### Reconstruction/calibration failure signature

If Low/High component predictions rank well but final `P(S>0)` calibration is poor, identify this separately from representation failure.

The first failed rerun must end with a `NEXT_PATCH_RECOMMENDATION`; do not implement a free search automatically.

---

## 12. Fresh confirmation sealing protocol

If readiness passes:

1. freeze the exact primary model identity and all upstream run IDs/hashes;
2. construct a confirm feature view for 2026-09-16..2026-09-30 with target DA/RT/Spread labels masked/unavailable to the prediction path;
3. generate predictions for all required arms;
4. save prediction files and SHA256 manifest;
5. create `CONFIRM_SEAL.json` containing model/config/source/code fingerprints and 15 target dates;
6. verify exactly 1440 predictions per arm and no duplicate/missing `(date, period)` keys;
7. only after sealing may a separate metric-opening step read canonical target labels;
8. compute metrics exactly once;
9. no threshold, calibration, feature, kappa, decomposition or model change may be made after metrics are opened and still be called the same confirmation.

If a code/runtime bug prevents predictions before labels are opened, repair the bug and reseal. If metrics have already been opened, any subsequent model change requires a future holdout and late September becomes consumed.

---

## 13. Fresh confirmation success classification

Primary comparison is the dev-frozen T24 primary variant versus `DIRECT_LGBM` on all 1440 late-September slots.

### `MSR_T24_CONFIRMED`

Require all:

- Balanced gain >= +0.01;
- AUC gain >= +0.01;
- positive recall delta >= -0.02;
- negative recall delta >= -0.02;
- at least one recall improves >= +0.02 OR both are non-decreasing;
- Brier delta <= +0.01;
- MAE ratio <= 1.05;
- split the 15 days into three fixed 5-day chronological blocks: at least 2/3 blocks have non-negative Balanced delta and worst block deficit >= -0.05.

### `MSR_T24_PARTIAL_SIGNAL`

Use when separation improves but one important dimension fails, e.g. AUC rises while recall/calibration/value error trades off, or improvement is unstable across blocks.

### `MSR_T24_NOT_SUPPORTED`

Use when Balanced/AUC do not improve materially, class collapse occurs, or value/probability performance deteriorates enough to invalidate the mechanism claim.

Do not change these gates after opening late-September metrics.

---

## 14. Z-mediator conclusion

Report Z separately from the overall MSR result.

At minimum provide:

- V1b - V1a Balanced delta;
- V1b - V1a AUC delta;
- V1b - V1a +Recall/-Recall deltas;
- V1b - V1a Brier delta;
- V1b - V1a MAE delta;
- direct Z prediction quality.

Final mediator token:

- `Z_MEDIATOR=SUPPORTED`;
- `Z_MEDIATOR=PARTIAL`;
- `Z_MEDIATOR=NOT_SUPPORTED`.

Do not call Z causal; it is a probabilistic physical mediator / predictive state unless a future causal-identification design is added.

---

## 15. Required experiment outputs

Use a new namespace, never overwrite historical C0/C1 runs:

`outputs/experiments/msr_spread_v1/t24_controlled_rerun_20261009/`

Recommended structure:

```text
repair/
controlled_rerun/
  S0_T24_LABEL_AUDIT/
  C1_LOW/
  C1_LOW_STATE/
  C1_Z/
  C1_HIGH_NO_Z/
  C1_HIGH_WITH_Z/
  C1_RECONSTRUCTION/
  C1_EXPLANATION/
retrospective_months/
  2026-05/
  2026-06/
  2026-07/
fresh_confirm/
  feature_view/
  sealed_predictions/
  metrics_opened/   # only if READY_FOR_FRESH_CONFIRM=YES
report/
```

Every stage must contain a run manifest and hashes sufficient to reproduce lineage.

---

## 16. Final report requirements

Before late-September is opened, report:

```text
REPAIR_GATE=<PASS/FAIL>
PRODUCTION_INDEPENDENCE_GATE=<PASS/FAIL>
T24_OPERATOR_HASH=<sha256>
TESTS=<passed/total>
STAGE_TRAINING=<PASS/FAIL>
RETRO_MAY=<metrics>
RETRO_JUNE=<metrics>
RETRO_JULY=<metrics>
PRIMARY_VARIANT=<V1a/V1b>
Z_MEDIATOR_RETRO=<SUPPORTED/PARTIAL/NOT_SUPPORTED>
READY_FOR_FRESH_CONFIRM=<YES/NO>
```

If `READY_FOR_FRESH_CONFIRM=NO`, STOP and provide:

- failing gate(s);
- component diagnosis;
- `NEXT_PATCH_RECOMMENDATION`;
- explicit `LATE_SEPTEMBER_METRICS_OPENED=NO`.

If readiness is YES and the sealed confirmation completes, additionally report:

```text
CONFIRM_SEAL=<PASS/FAIL>
CONFIRM_ROWS_PER_ARM=1440
LATE_SEPTEMBER_METRICS_OPENED=YES
MSR_T24_CONFIRMATION=<CONFIRMED/PARTIAL_SIGNAL/NOT_SUPPORTED>
Z_MEDIATOR=<SUPPORTED/PARTIAL/NOT_SUPPORTED>
OCTOBER_LOCKBOX_ACCESSED=NO
```

And include a compact five-arm comparison table for Direct / Periodic V1a / Periodic V1b / T24 V1a / T24 V1b, pooled and by month/block.

---

## 17. Hard prohibitions

- Do not access October target labels.
- Do not use 2026-08-15..2026-09-15 consumed prospective metrics to tune this run.
- Do not use late-September metrics unless retrospective readiness first passes.
- Do not auto-modify the model after a retrospective failure; return diagnosis for review.
- Do not retune T24/lambda/wavelet family.
- Do not add neural backbones, attention, TCN, extra experts or new loss functions in this controlled rerun.
- Do not add an independent point head/loss.
- Do not feed true Low state or true Z downstream; OOF predictions only.
- Do not modify frozen TafM/LightGBM historical source directories.
- Do not modify production 2.5.
- Do not stage/commit/push unless separately requested.

---

## 18. Scientific interpretation

This protocol is deliberately conservative.

May/June/July answer whether the redesigned T24 decomposition produces a stable, useful MSR mechanism on several already-open out-of-sample months. They are development diagnostics, not independent proof.

Late September is preserved as the actual one-shot metric-unopened holdout. If the model fails before that point, the holdout remains available for a corrected design. If it passes readiness and then succeeds on sealed late-September predictions, that is materially stronger evidence that the model is not merely fitting one month or one historical regime.

A positive result supports the MSR mechanism; it does not imply production promotion or universal market validity.

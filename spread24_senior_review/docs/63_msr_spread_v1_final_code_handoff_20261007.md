# 63 — MSR-Spread V1 Final Code Handoff (2026-10-07)

STATUS=FINAL_IMPLEMENTATION_HANDOFF
PROJECT=spread24_senior_review
TARGET=Spread = RT - DA
RESOLUTION=96 points/day, 15-minute
FORECAST_ORIGIN=D-1 14:00
HISTORY_LENGTH=672 quarter-hours (7 days x 96)
PRODUCTION_2_5=STRICTLY_READ_ONLY
OCTOBER_LOCKBOX=FORBIDDEN

## 0. Purpose and precedence

This document is the final implementation handoff for MSR-Spread V1. It closes the design discussion and is intended to be directly executable by a local coding agent.

For MSR-Spread V1 implementation, this document takes precedence over earlier provisional architecture notes in docs/56, docs/60 and docs/61. Earlier documents remain useful for background, literature motivation and historical decisions, but must not override the contracts below.

The frozen old implementation under `src/TafM_改进源码` is reference-only and must not be modified.

All new code belongs under:

`src/research96/msr_spread_v1/`

## 1. Scientific model in one sentence

MSR-Spread V1 models the final 96-point spread distribution by decomposing historical spread into a persistent low-frequency market baseline and a signed high-frequency correction, while explicitly inserting a probabilistic physical intermediate state describing future balancing surprise:

`X_672 -> p(L) + p(Z) -> Markov-smoothed low regime -> p(H | X_fast, p(L), regime, p(Z)) -> p(S=L+H)`

where:

- `S = RT - DA` is the final target;
- `L` is the slow/low-frequency spread baseline;
- `H = S - L` is the signed high-frequency correction;
- `Z` is the probabilistic physical mediator / balancing-surprise state.

The model does NOT use Direction->Magnitude decomposition as its main architecture.

## 2. Data boundary and input contract

### 2.1 Unified input length

The external historical sequence interface is unified to 672 quarter-hour points:

`7 days x 96 = 672`.

Do not create a second independent 168-step dataset.

The low branch may internally form a coarse representation by pooling every four consecutive quarters:

`672 -> 168`.

This is a representation transform of the same 96-point history, not a second source.

### 2.2 Legal information

Prediction origin is fixed at D-1 14:00.

Allowed inputs may include:

- target-day forecast fundamentals available by the origin;
- D-1 visible prefix information under the frozen research boundary;
- D-2 and older complete historical truth;
- derived statistics fitted only from legal history.

Forbidden as model input:

- target-day DA truth;
- target-day RT truth;
- target-day spread truth;
- target-day actual fundamentals;
- October final lockbox data.

Historical proxy/vintage limitations must remain explicitly labelled as such.

## 3. Historical target decomposition

For every historical target day D, define the 96-point signed spread:

`S_D,t = RT_D,t - DA_D,t`.

Apply a fixed shift-invariant wavelet decomposition to the historical label only.

First controlled choice:

- wavelet: `sym4`;
- levels compared only between J=2 and J=3 on development/validation data;
- no broad transform sweep.

Define:

`L_true = A_J`

and

`H_true = S_true - L_true`.

Therefore reconstruction is exact by construction:

`S_true = L_true + H_true`.

Wavelet decomposition is a label-construction mechanism, not a way to access future target-day information at inference.

## 4. C0 decomposition gate

Before training any predictor, verify that the decomposition supports the scientific hypothesis.

Required diagnostics:

- exact reconstruction error;
- raw Spread sign transitions/day;
- Low sign transitions/day;
- High sign transitions/day;
- ABA chatter (`+ - +`, `- + -`) counts;
- Low-state dwell-time distribution;
- proportion of final sign flips caused by `H` overcoming the sign/magnitude of `L`;
- J=2 versus J=3 comparison.

C0 passes only if Low is materially more persistent/smooth than raw Spread while High retains meaningful local corrections and true sign-crossing events.

If this separation is not observed, stop before C1 and report `MSR_V1_NOT_SUPPORTED_AT_DECOMPOSITION_GATE`.

## 5. Low-frequency branch

### 5.1 Task

Predict the low-frequency distribution:

`X_slow -> Q_L(q,t)`

for each of the 96 target periods.

Initial quantiles:

`q = {0.10, 0.25, 0.50, 0.75, 0.90}`.

C1 implementation uses LightGBM Quantile or a simple quantile-regression baseline. Do not introduce a deep backbone in C1.

There is no separate point head. The median is `Q_L(0.50)`.

### 5.2 Low feature families

Low features focus on slow market state and broad system pressure:

- internal 672->168 coarse history summaries;
- D-2 / D-3 / D-7 same-quarter Spread;
- trailing 7d/28d same-quarter mean, median, IQR, std, positive rate and quantiles;
- recent daily spread median/IQR/volatility;
- target-day forecast levels for stable fundamentals;
- residual load and whole-day forecast-profile summaries;
- renewable/wind/solar share;
- broad bidding-space / residual-space pressure;
- long-run forecast reliability from D-2 and older history;
- source-disagreement summaries when legal;
- period/hour/quarter-in-hour/weekday/month cyclic context.

Month identity must remain weak context, not a month-specific expert.

## 6. Low distribution -> three-state regime -> anti-chatter smoothing

Do not train an unrelated three-class head in C1.

Convert the predicted low quantiles into a monotonic approximate CDF `F_L(x)`.

Define three low-frequency regimes using training-only robust thresholds:

- NEG: `L < -delta_t`;
- NEU: `-delta_t <= L <= delta_t`;
- POS: `L > delta_t`.

A preferred initial threshold is:

`delta_t = kappa * IQR(L_train,t)`

with `kappa` selected on validation only.

Compute raw regime probabilities:

`p_neg = F_L(-delta_t)`

`p_neu = F_L(delta_t) - F_L(-delta_t)`

`p_pos = 1 - F_L(delta_t)`.

Estimate a three-state transition matrix from training-only Low labels:

`A_ij = P(Z_t=j | Z_{t-1}=i)`.

Use Forward-Backward to obtain smoothed posteriors:

`p_tilde_neg, p_tilde_neu, p_tilde_pos`.

Viterbi may be used for interpretation and dwell-time diagnostics, but hard Viterbi paths must not be required as the only downstream state representation.

The persistence mechanism acts on Low state only. It must not directly force the final Spread sign to remain constant.

## 7. Probabilistic physical intermediate state Z (A03 integration)

### 7.1 Role

A03's intermediate state is a formal V1 component, but it is a soft probabilistic mediator, not a hard bottleneck.

The main High model still receives its own fast features and Low-state context even if Z is imperfect.

Therefore:

`H = f(X_fast, p(L), LowRegime, p(Z))`

not

`H = f(Z)`.

### 7.2 Physical interpretation

Z represents future balancing / net-surprise pressure between the day-ahead expected state and realized real-time physical state.

The initial historical target candidate is:

`Z_true = NetLoad_actual - NetLoad_forecast`.

The exact NetLoad composition and signs MUST NOT be guessed from field names. Before freezing Z labels, generate a semantic audit that lists each candidate component, physical meaning, sign convention, missingness and deployment availability.

Required artifact:

`INTERMEDIATE_STATE_SEMANTIC_AUDIT.md` and/or JSON equivalent.

### 7.3 Keep shock separate from sensitivity

Do not fold reserves directly into the Z label by default.

Interpret:

- Z = size/direction of the physical forecast surprise (`Shock`);
- reserve / bidding space / residual demand / available supply = how sensitive price is to that shock (`Sensitivity`).

This motivates the approximate mechanism:

`HighCorrection ~= Shock x Sensitivity + Residual`.

The implementation does not need to enforce literal multiplication in C1, but feature routing and explanation should respect this distinction.

### 7.4 Z predictive distribution

Predict Z probabilistically using legal information:

`X_state -> Q_Z(q,t)`

with at least:

`q = {0.10, 0.50, 0.90}`

and derive:

- `z_q10`;
- `z_q50`;
- `z_q90`;
- `z_p_positive = P(Z>0)`;
- `z_width = q90-q10`.

All training-side Z features supplied to downstream models must be chronological OOF predictions. True Z may be used only as a label or explicit oracle diagnostic.

### 7.5 Required V1 ablation

Run both:

- `MSR-V1a`: no predicted-Z features in the High branch;
- `MSR-V1b`: same model plus OOF probabilistic-Z features.

This isolates the actual incremental value of the mediator.

Predicted Z is promoted to V1 core only if V1b provides stable gain beyond V1a on regression, ranking/calibration and class-balance metrics.

## 8. Mechanism-state feature engineering

Previous literature experiments are integrated as structured features, not independent backbones.

### 8.1 Forecast-surprise family

From legally visible D-1 prefix actual-vs-forecast information, derive items such as:

- last error;
- recent 4-quarter mean;
- recent 12-quarter mean;
- std;
- slope;
- net-surprise summary.

Apply where semantics/history are trustworthy: load, wind, solar, import/interchange, local generation, etc.

### 8.2 Tightness / sensitivity family

Retain interpretable state descriptors such as:

- residual load;
- forecast net load;
- upward/downward pressure proxies;
- renewable surplus;
- bidding-space margin/ratio;
- reserve ratio/asymmetry when legal;
- available/committed supply proxy only when semantics are proven;
- boundary transition from visible D-1 state toward D forecast state;
- forecast uncertainty/disagreement.

A02-style features that previously improved MAE/AUC/ranking should remain available for controlled ablation even if they did not solve Direction by themselves.

## 9. High-frequency branch

### 9.1 Task

Predict the signed high-frequency correction distribution only after Low and Z have been predicted OOF:

`X_fast + Q_L + smoothed LowRegime + Q_Z -> Q_H(q,t)`.

Initial quantiles:

`q = {0.10, 0.25, 0.50, 0.75, 0.90}`.

C1 uses one LightGBM Quantile high-frequency model family. Do not use a Direction/Magnitude head, dual expert, full Transformer, or full attention stack.

### 9.2 High feature families

High features focus on local changes and physical shocks:

- full 672-quarter fine history representation/statistics;
- 15/30/60-minute ramps;
- local curvature;
- load/wind/solar/renewable/import/local-generation ramps;
- residual-load ramp;
- net-ramp pressure;
- solar-drop-load-rise interaction;
- local volatility;
- D-1 p1..p56 forecast-surprise summaries;
- forecast uncertainty and source disagreement;
- availability masks;
- Low quantiles / low median;
- smoothed low-regime probabilities;
- optional probabilistic Z summaries in V1b;
- sensitivity features: reserve/tightness/bidding-space/residual-demand context.

The High branch is intended to learn how a short-run physical shock changes spread conditional on the prevailing slow market state and system sensitivity.

## 10. Probabilistic reconstruction and outputs

The internal model is distributional. It does not maintain a separate point-prediction objective/head.

Construct Spread scenarios from Low and High distributions:

`S^(m)_t = L^(m)_t + H^(m)_t`.

C1 may begin with an explicitly labelled simple coupling approximation such as same-quantile pairing, but code must expose a reconstruction interface so empirical OOF dependence/copula-style coupling can be tested later without changing the Low/High predictors.

Final outputs include:

- Spread quantiles / scenarios;
- `spread_point = median(Spread distribution)`;
- `p_spread_positive = P(S>0)`.

Point output is derived from the predictive distribution and has no independent point head.

## 11. Loss contract

C1 is staged.

Low training objective:

`L_low = multi-quantile pinball(L_true, Q_L)`.

Z training objective:

`L_Z = multi-quantile pinball(Z_true, Q_Z)`.

High training objective:

`L_high = multi-quantile pinball(H_true, Q_H)`.

Do not add by default:

- BCE Direction loss;
- Direction/Magnitude decoupled loss;
- class weights;
- hard sign penalty;
- separate MAE/MSE/Huber point head loss.

The q=0.5 component is supervised through pinball loss as part of the distribution.

If later joint fine-tuning is justified, prefer final-Spread probabilistic objectives such as quantile pinball/CRPS and possibly Brier for `P(S>0)`, introduced one axis at a time after C1 diagnostics.

## 12. Chronological OOF training order

Required execution sequence:

1. build legal 672-point samples and labels;
2. C0 decomposition audit;
3. train Low quantile model on chronological training data;
4. generate chronological OOF Low quantiles;
5. derive OOF Low regime probabilities and run Markov smoothing;
6. audit and freeze the Z target semantics;
7. train Z quantile model;
8. generate chronological OOF Z distributions;
9. train High model using only OOF Low/State and, for V1b, OOF Z;
10. reconstruct final Spread distribution;
11. evaluate V1a and V1b side by side.

Never train High on true Low regime or true Z and then evaluate on predicted versions without explicitly labeling that run as oracle-only.

## 13. Interpretability contract

The model must support the teacher-facing question: "which variables/state caused this spread prediction?"

For each target period, save an explanation record containing at least:

- predicted final spread median;
- `P(Spread>0)`;
- Low median / relevant Low quantiles;
- Low NEG/NEU/POS smoothed probabilities;
- High median / relevant quantiles;
- Z median / q10 / q90 / `P(Z>0)` / width;
- top predictive contributors for Z;
- top predictive contributors for High;
- important tightness/sensitivity features.

For tree models use SHAP/tree contribution where practical. Label explanations as `predictive driver/model contribution`, not causal effect.

Preferred narrative decomposition for examples:

`Final Spread = Slow Base + Fast Correction`

with the auxiliary physical explanation:

`Fast Correction <- predicted balancing Shock (Z) under current market Sensitivity`.

## 14. Code location and package structure

All new source files belong only in:

`src/research96/msr_spread_v1/`

Recommended modules:

- `config.py`
- `data_contract.py`
- `dataset.py`
- `decomposition.py`
- `features_common.py`
- `features_low.py`
- `features_high.py`
- `mechanism_state.py`
- `intermediate_state.py`
- `low_quantile.py`
- `markov_state.py`
- `high_quantile.py`
- `reconstruct.py`
- `diagnostics.py`
- `explain.py`
- `evaluate.py`
- `pipeline.py`
- `run_experiment.py`
- `tests/`

The package README must explain current status, entry points, output directories and the distinction between V1a and V1b.

Do not move/rewrite the existing `research96` data infrastructure unless a minimal reusable import is clearly safer than duplication.

## 15. Checkpoints and saved artifacts

Required checkpoints:

- `C0_DECOMPOSITION`
- `C1_LOW`
- `C1_LOW_STATE`
- `C1_Z`
- `C1_HIGH_NO_Z` (V1a)
- `C1_HIGH_WITH_Z` (V1b)
- `C1_RECONSTRUCTION`
- `C1_EXPLANATION`

Each checkpoint must save its config, data boundary, metrics and prediction artifacts without overwriting prior runs.

## 16. Metrics

Final metrics:

- MAE;
- RMSE;
- median AE;
- P90 AE;
- raw Direction accuracy;
- Balanced Accuracy;
- positive recall;
- negative recall;
- AUC using `P(Spread>0)`;
- Brier score;
- predicted positive rate;
- monthly and p1..p96 breakdown.

Temporal-structure metrics:

- sign transitions/day;
- ABA chatter/day;
- dwell-time distribution;
- predicted vs true transition counts;
- legitimate high-frequency override rate.

Low metrics:

- pinball;
- median MAE;
- quantile coverage;
- state calibration/persistence diagnostics.

Z metrics:

- median MAE;
- RMSE if useful;
- sign accuracy;
- correlation;
- AUC of `P(Z>0)`;
- q10-q90 interval coverage;
- uncertainty width.

High metrics:

- pinball;
- median MAE;
- tail-conditioned performance.

## 17. Baselines and ablations

At minimum compare on the same resolution and same held-out period:

- stable direct Spread LightGBM baseline;
- MSR-V1a (no Z);
- MSR-V1b (predicted probabilistic Z).

Where directly comparable, also report existing A04/A05 evidence as historical context, but do not mix incompatible resolutions/time windows in one ranking table without explicit caveat.

No-Z vs With-Z is mandatory and is the primary decision for whether the A03 mediator is promoted.

## 18. Implementation discipline

- No production 2.5 writes.
- No modification of frozen `src/TafM_改进源码`.
- No October final lockbox access.
- No target-day actual/DA/RT/spread features.
- No random chronological split.
- No test-set selection of wavelet level, kappa, threshold, feature family or hyperparameters.
- No automatic architecture growth after a weak result.
- No Git stage/commit/push unless explicitly requested.

C1 is a mechanism-validation stage, not a neural-architecture competition.

## 19. Final reporting token

After C0/C1 completion, final scientific status must be one of:

- `MSR_V1_MECHANISM_SUPPORTED`
- `MSR_V1_PARTIAL_SIGNAL`
- `MSR_V1_NOT_SUPPORTED`

The report must identify separately whether evidence supports:

1. Low/High decomposition;
2. Low-state persistence;
3. probabilistic Z predictability;
4. Z incremental value in High prediction;
5. final Spread distribution/direction quality;
6. interpretability usefulness.

Do not collapse all six into one headline number.

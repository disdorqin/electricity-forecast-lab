# 74 — MSR T24 State Consistency + Tempered Persistence Repair

STATUS=AUTHORIZED_CODE_REPAIR_ONLY
DATE=2026-10-09
PROJECT=spread24_senior_review
TARGET=Spread = RT - DA
RESOLUTION=96 points/day
FORECAST_ORIGIN=D-1 14:00 Asia/Shanghai
DECOMPOSITION=FROZEN NATURAL_TREND_T24
OCTOBER_LOCKBOX=FORBIDDEN
LATE_SEPTEMBER_HOLDOUT=SEALED (2026-09-16..2026-09-30)
PRODUCTION_2_5=STRICTLY_READ_ONLY

## 0. Purpose

Docs/72–73 established that the T24 Low/High decomposition is an improvement over the old periodic split, but the retrospective readiness gate still failed because positive-spread recall remained too weak. The component diagnostics isolate the largest current defect in the **Low-state persistence layer**, not in the T24 decomposition itself:

- May/June/July T24 V1a improves pooled MAE, Balanced Accuracy and AUC over Direct, so the multiscale decomposition is not rejected.
- State smoothing materially worsens multiclass Brier in all three months.
- Median state dwell roughly doubles after smoothing (truth about 13–15 slots, smoothed about 27–30.5 slots).
- Transition count is compressed from roughly 3.8 -> 1.0/day on dev and 9.0 -> 1.9/day on the retrospective final panel.
- POS-state recall is weak and can collapse to zero.
- The High model is trained with OOF state semantics based on `kappa=1.0`, while retrospective inference uses the dev-selected `kappa=0.5`; this is a train/inference state-definition mismatch.
- Low quantiles themselves are not well calibrated (`Q0.5` empirical coverage about 0.355 on dev), therefore state probabilities derived from the Low CDF must not be treated as perfectly calibrated probabilities.
- Z has predictive signal by itself, but `V1b - V1a` does not improve the final task; Z is **not** part of this repair round.

This document authorizes **code repair and tests only**. It does not authorize a new scientific result, May/June/July re-adjudication, or opening the late-September holdout. After Codex completes this repair, the code must be independently reviewed before the state-only experiment is executed.

---

## 1. Scientific scope: what is frozen and what may change

### 1.1 Frozen; do not change

The following remain exactly as in docs/63/69/72:

- `NATURAL_TREND_T24` operator and its hash:
  `7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7`;
- 672-point history contract;
- D-1 14:00 information boundary and D-2 label maturity rule;
- Low and High targets;
- Low/High feature families;
- LightGBM architecture and parameters;
- quantile sets;
- final reconstruction logic;
- Z predictor/model/features and V1b logic;
- T24 decomposition choice and lambda;
- May/June/July and late-September window definitions.

Do **not** change T24, LightGBM depth/leaves/estimators, add a network, add class weighting, change quantile levels, add a Direction head, or tune the final decision threshold in this repair.

### 1.2 Authorized changes

Only these areas may change:

1. State smoothing semantics and implementation.
2. State `kappa` consistency between OOF training features and deployment/holdout features.
3. State-layer diagnostics and provenance.
4. Low-quantile calibration **diagnostics only**; no Low calibration transform is activated in this round.
5. Tests and artifact metadata required to prove 1–4.

---

## 2. Root diagnosis A — current Markov layer is over-persistent

The current transition matrices have diagonal probabilities around 0.95–0.98. Combined with a full-strength chain smoother, this creates much longer predicted dwell than truth and pushes state confidence toward ~0.9–0.96 even when the raw Low-derived state evidence is uncertain.

The intention of the state layer is valid:

> Low regime should have persistence and should not chatter quarter-by-quarter.

The implementation must not imply:

> once a regime is entered, it should remain there unless overwhelming evidence arrives.

The repaired state layer therefore treats historical transitions as a **tempered persistence potential**, not as an absolute HMM emission model.

---

## 3. Root diagnosis B — probability semantics must be explicit

`raw_regime_probs()` is constructed from the predicted Low quantile CDF. It is best interpreted as an approximate discriminative posterior / unary state evidence:

\[
r_t(z) \approx P(Z_t=z\mid X_t),
\]

not as a generative emission likelihood:

\[
P(X_t\mid Z_t=z).
\]

The old `forward_backward()` function currently accepts this `r_t` under an HMM-style `emission` name. The arithmetic can be understood as a valid chain-potential heuristic, but the probabilistic interpretation is misleading and the persistence strength is fixed implicitly at one.

The repair must make the actual model explicit.

---

## 4. New state smoothing definition

Introduce a new active function, e.g.:

```python
smooth_state_evidence(
    raw_probs: np.ndarray,
    transition: np.ndarray,
    persistence_strength: float,
    prior: np.ndarray | None = None,
) -> np.ndarray
```

The chain distribution is defined by unary state evidence and a tempered transition potential:

\[
q(z_{1:T}\mid X)
\propto
\exp\left(
\sum_{t=1}^{T}\log r_t(z_t)
+
\lambda_s\sum_{t=2}^{T}\log A_{z_{t-1},z_t}
\right),
\]

where:

- `r_t` is the Low-derived raw state probability/evidence;
- `A` is the training-only transition matrix;
- `lambda_s >= 0` is a persistence-strength parameter.

Equivalently the pairwise potential is:

\[
\Psi_{ij}=A_{ij}^{\lambda_s}.
\]

This is a chain CRF / pairwise-potential interpretation. Do not call `r_t` an HMM likelihood in new artifacts.

### 4.1 Required limiting behavior

The implementation must satisfy:

**lambda = 0**

\[
q_t(z)=r_t(z)
\]

up to floating-point tolerance. This is the RAW state control.

**lambda = 1**

It must reproduce the legacy full-strength chain smoother, subject only to floating-point differences, when the same uniform prior and transition matrix are used. This is required for historical reproducibility; it does not mean lambda=1 remains the active choice.

### 4.2 Numerical implementation

Prefer log-space forward/backward with `logsumexp`:

- clip raw probabilities only to a tiny epsilon needed for log safety;
- row-normalize raw probabilities before use;
- compute `lambda_s * log(A)` for pairwise potentials;
- normalize posterior rows to sum to one;
- reject NaN/Inf inputs;
- do not smooth across day boundaries: each 96-slot target day is an independent chain.

The previous function may remain as `legacy_forward_backward()` or a thin wrapper for reproduction tests, but new active code must call the explicit tempered implementation.

---

## 5. Pre-registered persistence candidates for the *later* state-only experiment

Implement support for, but **do not scientifically select in this repair task**:

```text
STATE_PERSISTENCE_GRID = {0.00, 0.25, 0.50, 0.75, 1.00}
```

No additional values may be added after the later state experiment begins.

Interpretation:

- 0.00 = no temporal persistence; raw Low-derived state evidence;
- 1.00 = legacy full-strength persistence;
- intermediate values = tempered persistence.

This grid is frozen now so that later results do not determine the search space.

---

## 6. Root diagnosis C — kappa semantics are inconsistent across train and inference

Current artifacts show:

- OOF state features used for High training: `kappa=1.0`;
- dev-selected and retrospective inference state semantics: `kappa=0.5`.

Because

\[
\delta_t=\kappa\,IQR(L_{train,t}),
\]

changing kappa changes the meaning and class proportions of NEG/NEU/POS. Therefore High currently sees different state semantics during training and inference.

This must be repaired before any new model result is trusted.

### 6.1 Frozen kappa for the repaired pipeline

The previous dev-only selection already chose:

```text
KAPPA_FROZEN = 0.5
```

Do **not** re-search kappa in this repair or in the upcoming state-only experiment.

This value may be used because it was selected before May/June/July retrospective adjudication and before the sealed late-September confirmation window was opened.

### 6.2 Two-pass OOF rule

After `kappa=0.5` is frozen, regenerate state features used by the High training set as follows:

For every OOF fold:

1. fit/read Low predictions exactly as before;
2. compute `delta_t` using only labels legally available before that fold's prediction origin;
3. multiply the fold-local IQR by the same **frozen kappa=0.5**;
4. estimate the fold-local transition matrix only from legally available historical state labels under the same kappa=0.5 definition;
5. convert that fold's OOF Low quantiles to raw state evidence with kappa=0.5;
6. apply the requested persistence strength to obtain smoothed state evidence;
7. store these OOF states for High training.

For dev/retrospective/future inference, use the same kappa=0.5 state definition.

The only global information flowing backward into historical OOF construction is the already frozen hyperparameter value `0.5`. No future labels may enter any fold-specific IQR or transition estimate.

### 6.3 Deprecate active `STATE_KAPPA_OOF=1.0`

It may remain in historical manifests or compatibility code, but the repaired active pipeline must not use different `kappa` values for High-train OOF and inference.

Artifacts must report one field:

```text
state_kappa_frozen = 0.5
```

and every state feature consumed by the repaired High pipeline must carry the same semantic hash/version.

---

## 7. Low-quantile calibration problem: diagnose now, do not silently repair

The current Low dev artifact reports empirical `Q0.5` coverage about 0.355 rather than ~0.5. This means the Low quantile distribution is not well calibrated and therefore raw state probabilities derived from that CDF can also be biased.

This is a real scientific issue, but modifying Low calibration in the same patch would confound the state-persistence experiment. Therefore this round adds diagnostics only.

Add a reusable Low calibration report containing at least:

- empirical coverage for q10/q25/q50/q75/q90;
- quantile crossing rate before/after the existing monotonicity enforcement, if any;
- mean/median Low residual by month and by slot;
- Q50 coverage by month;
- an explicit warning if `abs(coverage_q50 - 0.5) > 0.05`;
- no fitted recalibration transform and no change to Low predictions.

Required artifact token:

```text
LOW_CALIBRATION_STATUS=WARNING | ACCEPTABLE
```

Given the existing artifact, a warning is expected; the code must not fake a PASS by changing Low values.

This will be the next scientific axis only if the repaired state layer remains insufficient.

---

## 8. Z mediator: unchanged and excluded from this repair

Z is not deleted. Existing results show Z has standalone ranking signal, but V1b does not improve pooled final metrics over V1a.

For this repair:

- do not alter `intermediate_state.py` scientific target;
- do not tune Z;
- do not change Z features;
- do not use Z to select persistence strength;
- do not claim Z is disproved.

The later state-only experiment must use **V1a/no-Z** first to isolate state repair. Only after state repair is adjudicated may V1b be revisited.

---

## 9. High model interface changes

Do not change High LightGBM architecture or its feature list except for the provenance/semantic identity of state features.

The repaired High feature contract remains:

```text
low_q10, low_q25, low_q50, low_q75, low_q90
state_neg_raw, state_neu_raw, state_pos_raw
state_neg, state_neu, state_pos
+ existing frozen High features
```

But every active state feature used for High training and inference must be generated with:

```text
state_kappa = 0.5
state_persistence_strength = the run's explicit value
state_smoothing_semantics = TEMPERED_CHAIN_UNARY
```

The state feature manifest must include these fields and a semantic fingerprint so artifacts with old kappa/state semantics cannot be accidentally mixed with repaired runs.

---

## 10. Compatibility and provenance

Historical outputs must remain readable and unchanged.

Do not overwrite:

- docs/73 successful chain;
- previous `C1_LOW_STATE` / High / reconstruction artifacts;
- old periodic runs;
- T24 run `20261009T025637339105Z`.

New repair artifacts belong in a distinct namespace, suggested:

```text
outputs/experiments/msr_spread_v1/state_repair_20261009/
```

Every new artifact must record:

- code fingerprint;
- config fingerprint;
- T24 operator hash;
- Low upstream run/hash if reused;
- frozen kappa;
- persistence strength;
- state semantic version;
- label cutoff / max label day used;
- whether any May/June/July scientific metrics were computed;
- whether late-September target metrics were opened (must be NO in this repair);
- whether October was accessed (must be NO).

---

## 11. Code areas expected to change

Expected primary files:

```text
src/research96/msr_spread_v1/config.py
src/research96/msr_spread_v1/markov_state.py
src/research96/msr_spread_v1/pipeline.py              (or the actual stage orchestrator)
src/research96/msr_spread_v1/tests/...
```

Likely state-stage/High-stage files may require interface edits if they currently hard-code kappa or legacy smoothing.

Do not perform unrelated refactors.

Before changing anything, grep for all uses of:

```text
STATE_KAPPA_OOF
forward_backward
raw_regime_probs
state_neg
state_neu
state_pos
kappa_source
```

and produce a short impact map.

---

## 12. Mandatory tests

Codex must add regression tests proving all of the following.

### T01 — lambda-zero identity

Random valid raw state probabilities:

```text
smooth_state_evidence(raw, A, 0.0) == raw
```

within tight floating tolerance.

### T02 — lambda-one legacy reproduction

For the same day, transition matrix and uniform prior:

```text
new(lambda=1.0) ~= legacy_forward_backward(...)
```

within declared numerical tolerance.

### T03 — normalization and finite output

All posterior rows sum to one; no NaN/Inf; probabilities are in [0,1].

### T04 — day-boundary isolation

Changing day D+1 state evidence cannot change day D smoothing. No chain crosses p96 -> next day's p1.

### T05 — persistence monotonic sanity

On a synthetic stable sequence, increasing lambda from 0 to 1 should not increase chatter. This is a behavioral test, not a selection rule.

### T06 — shock escape

On a synthetic sequence with strong one/two-slot POS evidence inside a NEG background, lambda<1 must still permit a posterior POS increase and must not numerically hard-lock the state.

### T07 — one kappa semantic contract

Repaired active High-train OOF and inference state artifacts both report `kappa=0.5`. Any attempt to combine mismatched state semantic fingerprints must fail fast.

### T08 — chronological OOF legality

Each OOF fold's delta/IQR and transition matrix may use only labels <= its legal cutoff. Recompute from an input where future labels are perturbed; earlier OOF states must not change.

### T09 — Low calibration report is read-only

Generating calibration diagnostics must not alter Low prediction bytes/hashes.

### T10 — sealed-window guard

Repair/test commands must not calculate or print target metrics for 2026-09-16..2026-09-30. October remains forbidden.

### T11 — T24 immutability

T24 operator hash remains exactly:

`7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7`.

### T12 — protected repository isolation

No write call to production 2.5, frozen `TafM_改进源码`, or `lightgbm_源码`. Do not revert or clean unrelated production changes.

---

## 13. Non-scientific smoke only

After unit/integration tests pass, run a smoke that exercises the repaired interfaces using synthetic data or a tiny already-consumed development slice.

The smoke may verify:

- Low artifact can feed state repair;
- kappa=0.5 OOF state generation works;
- lambda 0/0.5/1.0 all produce valid state tensors;
- High feature assembly accepts matching semantic fingerprints and rejects mismatches.

The smoke must **not** produce May/June/July model comparison results and must not open late-September labels.

---

## 14. Pre-registered later state-only experiment (do not execute before review)

Once this repair is independently approved, the next experiment is already frozen conceptually:

- decomposition: T24 unchanged;
- variant: V1a/no-Z only;
- kappa: 0.5 everywhere;
- state persistence candidates: 0, 0.25, 0.5, 0.75, 1.0;
- Low/High models/features/quantiles unchanged;
- select persistence only from state-level dev diagnostics, not final Spread metrics.

Later persistence admissibility gates:

1. multiclass state Brier must not be worse than RAW (`lambda=0`) by > 0.005;
2. POS state recall must not be worse than RAW by > 0.01 absolute;
3. median predicted dwell / true dwell must lie in [0.75, 1.50];
4. no state class may have zero recall if RAW has non-zero recall for that class;
5. among admissible candidates, choose lowest multiclass Brier; tie within 0.002 -> choose smaller lambda.

These gates are pre-registered now. They are not to be changed after viewing the later results.

Only after a state persistence choice is frozen will High and final Spread be rerun on already-consumed May/June/July retrospective windows. The sealed 2026-09-16..09-30 holdout remains unopened until the later readiness contract is satisfied.

---

## 15. Acceptance tokens for this repair task

Codex must finish with exactly enough evidence to support these fields:

```text
STATE_REPAIR_IMPLEMENTATION=PASS|FAIL
STATE_SEMANTICS=TEMPERED_CHAIN_UNARY
LAMBDA0_IDENTITY=PASS|FAIL
LAMBDA1_LEGACY_REPRO=PASS|FAIL
KAPPA_FROZEN=0.5
KAPPA_TRAIN_INFERENCE_CONSISTENCY=PASS|FAIL
OOF_CHRONOLOGY=PASS|FAIL
LOW_CALIBRATION_DIAGNOSTIC=PASS|FAIL
LOW_CALIBRATION_STATUS=WARNING|ACCEPTABLE
T24_OPERATOR_HASH=7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7
FULL_TESTS=<passed>/<total>
CLI_SMOKE=PASS|FAIL
SCIENTIFIC_STATE_EXPERIMENT_RUN=NO
MAY_JULY_NEW_METRICS_OPENED=NO
LATE_SEPTEMBER_METRICS_OPENED=NO
OCTOBER_LOCKBOX_ACCESSED=NO
PRODUCTION_WRITE=NO
READY_FOR_STATE_ONLY_EXPERIMENT=YES|NO
```

`READY_FOR_STATE_ONLY_EXPERIMENT=YES` is allowed only if all correctness/isolation tests pass. It does **not** mean the scientific model is validated; it only means the repaired code is ready for the next independently reviewed state-only experiment.

---

## 16. Stop conditions

Stop and report rather than improvising if any of the following occurs:

- lambda=0 does not reproduce raw probabilities;
- lambda=1 cannot reproduce legacy smoothing under the same inputs;
- regenerated OOF state features require future labels;
- kappa=0.5 cannot be made semantically consistent without modifying Low/High model architecture;
- T24 hash changes;
- a repair requires touching production 2.5 or frozen source trees;
- late-September/October target labels would be needed;
- an unrelated model redesign appears necessary.

Do not solve a failed repair by changing the scientific architecture. Return the blocker for review.

---

## 17. Expected interpretation after this repair

This repair does **not** assume that Markov/persistence is useful. It restores a fair experiment in which:

- RAW (`lambda=0`) is a first-class candidate;
- persistence is explicitly tempered rather than implicitly fixed at full strength;
- High sees exactly the same NEG/NEU/POS semantics in OOF training and inference;
- Low calibration weakness is measured rather than hidden;
- Z is kept outside the state repair so causal attribution of gains/losses remains possible.

If the later experiment selects lambda=0, the correct conclusion will be that the T24 Low state is useful but historical Markov persistence adds no incremental value under the current information boundary. That is a valid scientific result, not a failure of the entire MSR architecture.

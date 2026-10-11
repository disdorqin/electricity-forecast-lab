# MSR-Spread V1

STATUS=NEW_ISOLATED_RESEARCH_MODEL
TARGET=Spread = RT - DA
RESOLUTION=96 points/day
FORECAST_ORIGIN=D-1 14:00
HISTORY_LENGTH=672 quarter-hours

## 1. Purpose

This directory is the isolated implementation home for the new multi-scale probabilistic Spread model.

Do not modify `src/TafM_改进源码` or `src/lightgbm_源码` while implementing this model. Reuse ideas/components only through explicit imports or clean local adaptations when scientifically and contractually justified.

## 2. Core model contract

Historical labels are decomposed as:

`S = L + H`

where:
- `S`: 96-point RT-DA spread;
- `L`: low-frequency/slow market baseline;
- `H`: high-frequency signed correction.

The model uses one external historical length only:

`7 days x 96 = 672 points`.

Low-frequency processing may internally pool/aggregate the 672 sequence, but there is no second external 168-point dataset contract.

## 3. V1 computation order

1. Build legal 672-point history and target-day legal forecast/context features.
2. Construct historical `L_true/H_true` labels using training-only wavelet decomposition rules.
3. Predict low-frequency quantiles `Q_L`.
4. Convert `Q_L` to Negative/Neutral/Positive state probabilities.
5. Apply Markov forward-backward smoothing to the low-state probabilities.
6. Predict the A03 physical mediator distribution `p(Z|X)` using chronological OOF predictions for training downstream modules.
7. Predict high-frequency quantiles `Q_H` conditioned on fast features + predicted Low distribution/state + predicted Z distribution.
8. Reconstruct final Spread distribution from Low + High.
9. Publish both:
   - point forecast = final distribution median;
   - direction probability = `P(Spread > 0)`.

No standalone point head is used in V1.

## 4. A03 physical mediator Z

V1 promotes A03 from a shadow-only diagnostic to a **soft probabilistic mediator**.

Concept:

`Z = future balancing / net-surprise state`

Candidate historical target starts from:

`actual net-load state - forecast net-load state`

but exact field composition and signs must be frozen only after a semantic audit.

Required outputs:
- `Z_q10`
- `Z_q25`
- `Z_q50`
- `Z_q75`
- `Z_q90`
- `P(Z>0)`
- `Z_IQR` / uncertainty width

Important:
- downstream training receives OOF predicted Z only;
- true Z is diagnostic/oracle only;
- High is `f(X_fast, Low, State, predicted-Z)`, not `f(Z)`; therefore Z is informative but not a hard bottleneck;
- formal ablation must compare `MSR without Z` vs `MSR + predicted-Z`.

## 5. Mechanism interpretation

Use the conceptual decomposition:

`High correction ≈ Shock x Sensitivity + residual`

where:
- Shock is represented primarily by predicted `Z` / forecast surprise;
- Sensitivity is represented by reserve/tightness, residual load, bidding-space pressure, renewable surplus, local ramps, and low-frequency regime.

This is an interpretation and feature-routing principle, not a fixed multiplicative equation unless later experiments explicitly validate such a parametric form.

## 6. V1 implementation phases

### C0 — decomposition audit
Verify whether Low is truly persistent and High carries fast excursions.

### C1-A — Low quantile model
Start with an interpretable LightGBM Quantile implementation.

### C1-B — Low-state + Markov
Convert Low distribution into state probabilities and smooth across the 96-point day.

### C1-C — chronological OOF Low context
Generate deployable-style Low/state inputs for downstream training.

### C1-D — A03 mediator
Freeze semantic audit, train probabilistic Z, generate OOF Z features.

### C1-E — High quantile model
Start with one unified LightGBM Quantile high-frequency branch. No dual experts in C1.

### C1-F — reconstruction
Produce final Spread distribution, median forecast and `P(Spread>0)`.

### C1-G — ablation
Compare at minimum:
- direct LightGBM Spread baseline;
- MSR without Z;
- MSR + predicted-Z;
- oracle-Z diagnostic only.

### C2 — neural enhancement (only after C1 supports the mechanism)
Potentially replace component predictors with lightweight TafM/TCN-style encoders while preserving the same labels, states, Z contract and final probabilistic reconstruction.

## 7. Planned files

Actual module names after implementation (deviations noted inline):

- `config.py`: frozen experiment configuration.
- `data_contract.py`: feature/label/information-boundary contracts (planned name `contracts.py`).
- `dataset.py`: 672-history sample construction.
- `decomposition.py`: wavelet Low/High label construction and audit.
- `features_common.py`: shared legal features; the planned `features_low.py` /
  `features_high.py` were folded into it as `low_features()` / `high_features()`,
  plus the shared quantile-fitting, chronological OOF and CDF helpers.
- `mechanism_state.py`: Shock / Sensitivity / Slow mechanism columns (matrix-first).
- `intermediate_state.py`: Z semantic audit, labels, prediction and OOF export.
- `low_quantile.py`: Low probabilistic model.
- `markov_state.py`: state thresholds, transition matrix, forward-backward/Viterbi diagnostics.
- `high_quantile.py`: High probabilistic model.
- `reconstruct.py`: Low/High distribution reconstruction.
- `explain.py`: Low/High/Z/feature-contribution explanation outputs.
- `evaluate.py`: component/final metrics and day-clustered paired bootstrap.
- `diagnostics.py`: transition/ABA/dwell/override and calibration diagnostics.
- `checkpoints.py`: non-overwriting timestamped artifact persistence.
- `pipeline.py`: checkpoint orchestration.
- `run_experiment.py`: `--stage` CLI entry.
- `tests/`: leakage, reconstruction, OOF, state and probability tests.

## 8. First-version exclusions

Do not add by default:
- Transformer/full self-attention;
- dual high-frequency experts;
- Direction/Magnitude heads;
- hard sign penalties;
- class weighting;
- target-day actual fundamentals;
- oracle Z as a deployable feature;
- separate point-output loss;
- complex copula/KDE unless later diagnostics justify them.

## 9. Required explanation output

Each predicted quarter should be auditable through:
- final Spread median;
- `P(Spread>0)`;
- Low median / interval;
- smoothed low-state probabilities;
- High median / interval;
- predicted Z distribution / `P(Z>0)`;
- top positive and negative model-contribution drivers.

Driver language must be `predictive driver/model contribution`, not causal effect.

## 10. Authority

1. repository AGENTS.md and docs/README.md — isolation / information boundary / navigation.
2. docs/63_msr_spread_v1_final_code_handoff_20261007.md — highest-priority model implementation contract.
3. docs/67_msr_v1_pre_experiment_audit_and_repo_cleanup_plan_20261008.md — this remediation task, no model redesign.
4. docs/68_msr_v1_pre_experiment_closure_20261008.md — this round's acceptance evidence.
5. docs/66_msr_spread_v1_remaining_closure_20261008.md — preceding engineering closure evidence, not a design override.

Design origins docs/56, docs/60 and docs/61 are ACTIVE_REFERENCE, not overriding docs/63.
Historical absent reference: docs/62 is **NOT PRESENT in `docs/`**; its content is not guessed.
`docs/63` is the operative frozen implementation contract. This is provenance, not a startup dependency.
Old docs/64 and archived docs/archive/msr_v1_remediation_history/65_msr_spread_v1_remediation_evidence_20261008.md are historical only; their fingerprints are not current identity.

## 11. Run status — pre-remediation C0/C1 run of 2026-10-07 (historical record, superseded)

> **This section describes the run audited as defective in `docs/64_msr_spread_v1_audit_remediation_plan_20261008.md`.**
> Two rows of the table below are **invalid as delivered**: `Interpretability = SUPPORTED` (docs/64 R03 — the
> explanation stage refit a *different* High model, so the drivers did not describe the numbers being explained)
> and `Z incremental value: 0/5 metrics` (docs/64 R09 — the MAE/Brier bootstrap verdicts had the sign flipped).
> Neither defect was known when this section was written. The remediated chain is in §12; the artifacts referenced
> here (`.../20261007T17*` run dirs) were **not** modified, and are retained as provenance only.


Environment `conda epf-2` (python 3.11, numpy 2.4.6, pandas 3.0.6, PyWavelets 1.8.0,
lightgbm 4.7.0, scikit-learn 1.9.1), seed `20261007`, source snapshot sha256
`49fade43b0f3505a437e8704377bf5ef07f780429f9052caaca74aa3525e95a2`.
Artifacts: `outputs/experiments/msr_spread_v1/<CHECKPOINT>/<UTC stamp>/`.
Windows: train 2023-01-01..2025-12-31, dev 2026-01-01..2026-04-30, final 2026-05-01..2026-08-14.
Source-loading scope (docs/64 §6.4): the snapshot is read **in full** (2022-03-18..2026-09-30,
no October rows) when the curve matrices and the field-coverage/semantic audits are built, so
"never read" would be a false claim. The accurate statement is that the consumed
2026-08-15..2026-09-15 panel was **未用于本次模型选择和评估 / not used for this run's model
selection or evaluation** — it enters no fit row, no dev row and no final row — and the
October 2026 FINAL LOCKBOX was not accessed.

**FINAL TOKEN: `MSR_V1_NOT_SUPPORTED`** (all ten stages ran in contract order; `tests/` 8 passed).

| branch | verdict | decisive number |
|---|---|---|
| Low/High decomposition (C0) | SUPPORTED | J3, recon error 5.7e-14, Low churn 5.33 vs Spread 9.48/day, Low dwell median 11 vs 4, High var share 0.208 |
| State persistence | NOT SUPPORTED | smoothing cuts churn 8.56→1.80/day but smoothed balanced state accuracy 0.397 < raw 0.408 and POS recall 0.007 < majority-class baseline 0.567 |
| Probabilistic Z | NOT SUPPORTED (narrow) | final AUC 0.681, MAE 0.769× / 0.897× same-window climatology, but dev AUC 0.592 < pre-registered 0.60 |
| Z incremental value | NOT SUPPORTED | 0/5 metrics improved with bootstrap CI above zero (required 3) |
| Final Spread probability | NOT SUPPORTED | V1b direction 0.647 / AUC 0.582 beat direct 0.611 / 0.568, but +recall 0.224 vs 0.312 — the two classes still trade |
| Interpretability | SUPPORTED | 5 final-window extreme cases with Low/High/Z distributions and SHAP-style drivers |

Reading: the *label decomposition* is real and the *physical shock magnitude* is forecastable, but inside the
2026 legal information set neither the Slow base nor the Z sign carries enough within-window signal to move
positive and negative Spread recognition together. Dev and final disagree (MSR below the direct baseline on dev,
above it on final), so the final-window gain is reported as instability, not as a win. No Transformer/Attention/
deeper network was added after the weak result, per the contract.

Known caveats recorded with the artifacts:
- 2026 Spread volatility is ~1.7–1.9× the 2023–2025 training regime, so all levels are under-dispersed
  (MSR q25/q75 coverage 0.22/0.83 on final) and all methods share a +29…+36 bias.
- `Z = NetLoad_actual − NetLoad_forecast` has a +1575 MW mean and 63.4% positive share, i.e. a definitional
  level offset, not a centred surprise; the audit states this explicitly.
- Reconstruction uses a same-quantile (comonotonic) coupling; the countermonotonic sensitivity run drops
  AUC to 0.505, so the coupling assumption is material and is reported, not hidden.
- `C0_GATES.override_share_all_cells` replaces docs/63's literal `flip & |H|>|L|` share, which is tautologically
  1.0 given `S = L + H`; the substitution is documented in `decomposition.py`.

## 12. Previous remediated chain — historical evidence (2026-10-08, superseded by §13)

The previous remediation reported R01–R10 as fixed and `137 passed`. A subsequent review found
remaining gaps in audit classification, lineage enforcement, CDF ties, dev chronology and reporting.
The numbers below describe that historical implementation only; see §13 and docs/66 for closure evidence.

Delivered chain (one `--all` run, nine stages, one code fingerprint
`b7bdb5c5780f2e27ab288cccae6355b3e1e14ed5d9f33becd3fc4497984f3e01`;
config `a7658157e46302b5822c55c4ed57e9aab36774ee98745a263a675e48ad92d5a3`).
Every consuming stage freezes the upstream it read — run id, both fingerprints and the digests of the
artifacts it consumed — inside its own `RUN_MANIFEST.json`, so a mixed-chain read is detectable from the
run directory alone and `resolve_upstream` refuses an upstream whose digest disagrees with its manifest:

| checkpoint | run id |
|---|---|
| C0_DECOMPOSITION | 20261008T094117330828Z |
| C1_LOW | 20261008T094121815175Z |
| C1_LOW_STATE | 20261008T094223398542Z |
| C1_Z | 20261008T094224477493Z |
| C1_HIGH_NO_Z | 20261008T094312292085Z |
| C1_HIGH_WITH_Z | 20261008T094342815992Z |
| C1_RECONSTRUCTION | 20261008T094414506817Z |
| C1_EXPLANATION | 20261008T094459182972Z |
| REPORT | 20261008T094545812855Z |

Determinism check: the previous chain (`b9eff0fa…`) and this one differ in `FINAL_REPORT.json` only by the
18 newly disclosed `zero_label_share` fields and 3 same-chain run-id strings — 0 numeric metric differences,
same token, same six verdicts. The pre-fix 60-file run-space baseline re-hashes to 0 drift.

**FINAL TOKEN: `MSR_V1_PARTIAL_SIGNAL`** under
`EVALUATION_LABEL = RETROSPECTIVE_BUGFIX_REEVALUATION` — the final window 2026-05-01..2026-08-14 was
already opened on 2026-10-07, so this rerun validates the *code*, not a new prospective confirmation.

| branch | verdict | decisive number (same chain) |
|---|---|---|
| 1 Low/High decomposition | SUPPORTED | J3 the only level passing the dev gate; recon error 5.68e-14; Low churn 5.288 vs Spread 9.415/day; Low dwell median 11 vs 4 |
| 2 State persistence | SUPPORTED (gate-mechanical) | churn 8.50→1.9917/day (ratio 0.2343 ≤ 0.90) and smoothed accuracy 0.42558 ≥ 0.40 floor — but still below the majority-class 0.48698 and below raw 0.43404, POS recall 0.0674. Not usable as a state recognizer |
| 3 Probabilistic Z | SUPPORTED | AUC 0.6209 dev / 0.6960 final (≥ 0.60); MAE 0.8451 / 0.7532 × same-window climatology (≤ 0.95) |
| 4 Z incremental value | NOT SUPPORTED | family gate 0/3 in dev and 0/3 in final; all six per-metric verdicts `neutral` (CI spans zero); V1a 142 vs V1b 147 features |
| 5 Final Spread probability | NOT SUPPORTED | primary variant frozen on dev = `msr_v1a`; final gain vs direct balanced −0.0074, AUC −0.0082; direction 0.6343 is *below* the explicit all-negative baseline 0.6843 |
| 6 Interpretability | SUPPORTED (now with valid evidence) | 10,176 records = one per final predicted quarter, zero refits, same-chain models loaded, contribution-sum-vs-own-model-output error 1.46e-12, 198 crossing rows attributed to their source alpha |

Metrics now include an explicit `all_negative_baseline` next to `majority_class_baseline` (AGENTS.md §3). This
matters in dev, where the positive class is the majority (0.5175): "always negative" is 0.4825, not 0.5175.

Caveats carried by this chain: all levels are under-dispersed because 2026 volatility exceeds the 2023–2025
training regime (Low q25/q75 coverage 0.320/0.758, High 0.275/0.724); `Z` keeps a definitional level offset
(component-complete panel: mean +1207.7 MW, positive share 0.608 — the pre-remediation +1575 MW / 0.634 in §11
was computed with fabricated zero-filled cells and is superseded);
the coupling assumption stays material (comonotonic AUC 0.5940 vs countermonotonic 0.5727); and
`d1_prefix_vintage` remains LEGACY-UNVERIFIED / PROXY-EXPERIMENTAL, so passing tests proves implementation
correctness, never historical live-snapshot information authenticity.
No 2026-10-07 artifact was rewritten (60/60 baseline hashes re-verified, drift 0); failed intermediate
chains are preserved as `INTERRUPTED`.

## 13. Remaining-review closure (2026-10-08)

Authority remains docs/63. Acceptance evidence and the new run chain are recorded in
`docs/66_msr_spread_v1_remaining_closure_20261008.md`.

- A stage session freezes code/config/source identity, all resolved run IDs and consumed files.
  Missing digests, incompatible fingerprints, conflicting ancestors and incomplete runs are rejected.
  C0 and booster consumption are included in the manifest; `--all` pins each stage it produced.
  A single-stage `--upstream` pin to a descendant also freezes its ancestors.
- Audit results come from actual cutoff records, row masks, Z label consistency, feature roles,
  explanation identities and the manifest dependency graph. Failure cannot leave a STRICT/PASS headline.
  Check `protocol_audit.eligible_for_scientific_claim`, not just the scientific token.
- Dev state inputs use the predeclared kappa. Dev-selected kappa applies only to final rows, after
  its labels mature. Dev metrics with a dev-selected wavelet/primary are `SELECTION_DIAGNOSTIC`,
  not an independent chronological confirmation. Final remains `RETROSPECTIVE_BUGFIX_REEVALUATION`.
- Repeated quantile nodes contribute a right-continuous CDF jump; outer tails still use the explicit
  flat-CDF approximation. Zero is not positive. Atoms on Low's +/-delta boundaries belong to NEU.
- Prediction eligibility does not inspect D truth or the unseen D-1 price suffix. Labels and
  their completeness are filtered independently by fitting/evaluation.
- C0 and reported temporal diagnostics share zero-as-negative signs and pooled dwell runs reset
  at each day/missing cell. Bootstrap evidence follows the dev-selected primary variant.
- Explanations explicitly describe V1b, including all three loaded branches; selected source-alpha
  output must equal the saved median and the tree-contribution sum.

Historical prefix/forecast vintage is still `LEGACY-UNVERIFIED / PROXY-EXPERIMENTAL`.
Engineering closure neither proves scientific gain nor promotes the model to production.

Closure acceptance: **161 tests passed**; real `--all` completed nine stages with fingerprint
`d069c97245dd555bab673dfaeaac0899b718792443975bc23dc8cf732d877795`.
Full-chain report: `REPORT/20261008T104219346662Z/FINAL_REPORT.json`.
Single-stage descendant-pinned resume: `REPORT/20261008T104408544989Z`, identical report JSON.
Actual mechanics checks pass; the six scientific verdicts are **yes / no / yes / no / no / yes**,
giving **MSR_V1_NOT_SUPPORTED**. Dev uses predeclared kappa=1.0; selected kappa=0.5 applies only
to final. The previous §12 PARTIAL_SIGNAL is historical, not the current result.

## 14. docs/67 pre-experiment remediation (acceptance in docs/68)

N01: required physical sums/ratios preserve NaN; explicit forecast availability masks include nuclear, both reserves and full-system load.
N02: unknown reserve yields NaN pressure; only finite known tiny reserves use the 100MW numerical floor.
N03: system/direct load gap and D-1 p56 netload transition have unambiguous names. Genuine BoundaryForecast fields have long history, but dev-selection capture timestamps are after origin. New boundary-source features remain DEFERRED_NOT_IN_V1_SOURCE, not fabricated.
N04: periodic/reflection label audit on dev-selection only gives NEEDS_REDESIGN under the prior screening gate. Main sym4/J contract is unchanged. No new experiment is authorized by engineering test success.
N05: P(S>0) is a **finite-quantile interpolation approximation**, not a complete probability density/model. Five quantiles (.1,.25,.5,.75,.9), right-continuous atoms and flat CDF tails restrict ladder P(S>0) to [.1,.9]. Brier/calibration evaluate this approximation and depend on tail/coupling assumptions. No quantile/copula search here.
N06: build_panel constructs one flat dictionary; same repaired arrays match sequential assembly elementwise (dtype, NaN, column/index order). Missing-data corrections are not claimed identical to old fabricated zeros.

Latest completed closure fingerprint/run is referenced in docs/68 and its Gate evidence; §13 is a historical snapshot, not eternal current identity. Reruns of 2026-05-01..08-14 are RETROSPECTIVE_BUGFIX_REEVALUATION, never a fresh confirmation.

### Complete Python module inventory

| module | status | responsibility |
|---|---|---|
| __init__.py | ACTIVE_MODEL | MSR-Spread V1: multiscale regime probabilistic Spread model (docs/63). |
| boundary_audit.py | ACTIVE_MODEL / AUDIT_ONLY | Label-only daily edge sensitivity; frozen screening criteria in docs/68. |
| checkpoints.py | ACTIVE_MODEL | Checkpoint persistence with frozen lineage (docs/63 §15, docs/64 R10). |
| config.py | ACTIVE_MODEL | Frozen experiment configuration for MSR-Spread V1 (docs/63 contract). |
| data_contract.py | ACTIVE_MODEL | Field roles, information-boundary contract and label definitions for MSR-Spread V1. |
| dataset.py | ACTIVE_MODEL | 672-point legal history tensors and per-cell panel construction (docs/63 §2, §5, §9). |
| decomposition.py | ACTIVE_MODEL | Shift-invariant wavelet Low/High label decomposition (docs/63 §3-§4). |
| decomposition_adjudication.py | D0_ADJUDICATION_ONLY | D0 boundary-safe Low/High decomposition adjudication (docs/69); fits no model and cannot reach a C1 stage. |
| drift_provenance.py | READ_ONLY_PRODUCTION_AUDIT | docs/69 production drift attribution; no production writes/restores, fail-closed classification and baseline verification. |
| diagnostics.py | ACTIVE_MODEL | Temporal-structure and state diagnostics (docs/63 §16). |
| evaluate.py | ACTIVE_MODEL | Final and component metrics (docs/63 §16, AGENTS.md direction-report rules). |
| explain.py | ACTIVE_MODEL | Per-quarter predictive-driver explanations (docs/63 §13, docs/64 R03). |
| features_common.py | ACTIVE_MODEL | Shared feature routing, chronological splitting and quantile-model helpers. |
| high_quantile.py | ACTIVE_MODEL | High-frequency probabilistic branch (docs/63 §9). |
| intermediate_state.py | ACTIVE_MODEL | A03 probabilistic physical intermediate state Z (docs/63 §7). |
| low_quantile.py | ACTIVE_MODEL | Low-frequency probabilistic branch (docs/63 §5). |
| markov_state.py | ACTIVE_MODEL | Low-distribution -> three-state regime -> Markov anti-chatter smoothing (docs/63 §6). |
| mechanism_state.py | ACTIVE_MODEL | Mechanism-state features (docs/63 §8): Shock family and Sensitivity family. |
| pipeline.py | ACTIVE_MODEL | Stage orchestration for MSR-Spread V1 (docs/63 §12, §15; docs/64 R01-R10). |
| reconstruct.py | ACTIVE_MODEL | Probabilistic reconstruction of the Spread distribution (docs/63 §10, docs/64 R08). |
| run_experiment.py | ACTIVE_MODEL | CLI entry for MSR-Spread V1 stages. |
| state_only_experiment.py | DOCS76_STATE_ONLY | Read-only frozen Low migration; five development State candidates; immutable admissibility selection seal; stop before High when no candidate passes. Generic scientific stages stay locked. |

## 15. D0 completion (2026-10-09)

D0 selection is already frozen in `decomposition_adjudication_20261008/20261008T133948429497Z/FROZEN_DECOMPOSITION.json`: **NATURAL_TREND_T24**, lambda from `1/(16*sin(pi/24)^4)`. The original periodic reference and C0/C1 artifacts remain unchanged. No Low/State/Z/High predictor was trained or modified in this completion.

The later `20261008T134348928008Z` result is **RETROSPECTIVE_SANITY_CHECK**, not new confirmation; no catastrophic failure and no winner change. Completion evidence resides in `decomposition_adjudication_20261008/completion_20261008T151318Z/`. Independent time-domain reflection and natural normal-equation checks supplement the D0 tests.

Production classification remains **UNRESOLVED**: foreign CPython 3.10 caches and external crawler changes are observed, but one cache does not bind the current edited source. The classification criteria were not relaxed and unknown edits were not reverted. **READY_FOR_MSR_RERUN=NO**; no replacement production baseline is represented as approved.

## docs/72 controlled T24 round

`t24_controlled_rerun.py` is the repair-gated staged runner, capped to 2026-07-31 before curves are built. `--smoke` fits nothing and opens no confirm labels. `--run` consumes a PASS repair receipt and stops on retrospective readiness failure. S0 uses C0_DECOMPOSITION as the canonical lineage compatibility key, but the stage is named S0_T24_LABEL_AUDIT and never searches J/T24. Half-gain terminology and committed/uncommitted external workflow attribution supersede the earlier completion limitations; numerical T24 operator and old artifacts remain immutable.

## docs/74 state repair (code/test only)

`markov_state.py` active semantics are **TEMPERED_CHAIN_UNARY**, explicit run persistence strength,
with frozen kappa **0.5** for both OOF and inference. The OOF delta/transition fits still use
only each fold's D-2 history. `legacy_forward_backward` preserves historical arithmetic.
Lambda=0 returns normalized RAW exactly. Lambda=1 retains the legacy 1e-6 unary floor to
reproduce zero-evidence historical inputs; intermediate strengths use 1e-12 log safety.
This numerical compatibility policy, kappa, strength and day isolation are semantic-hashed.
No lambda is scientifically selected here. Grid stays 0/.25/.5/.75/1.

`low_calibration.py` reports coverage/crossing/residuals only, with no transformed predictions.
Pre-enforcement crossing is UNKNOWN when upstream saved only the enforced ladder.
`state_repair.py --smoke --dev-calibration` uses synthetic states plus frozen consumed dev Low
columns, reader-filtered through April 30. It fits no predictor, opens no May–July comparison,
and reads no September holdout/October truth. New outputs: state_repair_20261009/.

All added state metadata are excluded from High model features. High checks semantic row fields
and STATE_SEMANTICS.json before fitting; config lineage also binds strength/version/kappa.
Old Low/state/High chains remain immutable; cross-version artifacts cannot be silently promoted
into repaired training. Explicit independent compatibility review is needed for any later reuse.
Generic stage and old T24 scientific CLI execution are gated off under CODE_REPAIR_ONLY;
independent approval is required before the later V1a/no-Z state-only experiment.

## docs/76 state-only execution / docs/77 closure

`python -B -m src.research96.msr_spread_v1.state_only_experiment --select --out <new versioned run>`
authorizes only the frozen Low migration and five-candidate dev State adjudication.
It does not unlock generic `--all`. The first authorized run has no admissible
lambda and stops at Stage B, before OOF-State regeneration/High/reconstruction or
new May–July metrics. Selection JSON/SHA seal, candidate predictions, monthly state
diagnostics and FINAL_REPORT are under state_only_experiment_20261009/.
No new Low/Z/High model, no fresh_confirm directory, no September/October truth.
The stopped run does not pre-implement unauthorized fallback training.

## docs/78 gated CDF/unary recovery

- `cdf_unary.py` — Active: unchanged five-quantile CDF diagnosis, training-only deterministic right-continuous PIT ECDF, fixed three-block routing, calibrated unary semantic identity.
- `low_cdf_state_recovery.py` — Active: separate D1A/D1B/D1C/D1D CLI stages; test/migration/seal gates; stop before any unearned downstream work. Generic model CLI remains locked. Low ladders, T24, kappa=0.5 and frozen predictors are unchanged.

## docs/80 rolling Low location recovery

- `rolling_low_location.py` — Active: raw-q50 residual median, latest 28 complete D-2-mature days; RAW_LOCATION / ROLL28_GLOBAL / ROLL28_BLOCK3 only; five quantiles share one per-row shift; raw CDF, kappa=.5, persistence=0.
- `low_location_recovery.py` — Active: isolated D2A implementation audit, D2B March-only selection and SHA seal, D2C fixed-family April transfer. Jan-Feb is mature residual history, not selection. Failed Gate stops before OOF/High/new May-July/confirm. Frozen raw Low and historical runs never overwritten.

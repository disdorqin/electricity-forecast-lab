# 69 — MSR V1 Boundary-Safe Low/High Decomposition Adjudication

STATUS=AUTHORIZED_PRE_EXPERIMENT_REDESIGN_ONLY
DATE=2026-10-08
PROJECT=spread24_senior_review
TARGET=Spread = RT - DA
RESOLUTION=96 points/day
FORECAST_ORIGIN=D-1 14:00 Asia/Shanghai
HISTORY_LENGTH=672
OCTOBER_LOCKBOX=FORBIDDEN
PRODUCTION_2_5=STRICTLY_READ_ONLY

## 0. Purpose

Docs/68 closed the engineering audit but blocked new model experiments because the current per-day periodic zero-phase wavelet decomposition is materially edge-sensitive. The observed J3 dev-selection audit showed:

- periodic-vs-reflect mean |Low difference|: edge 18.7810 vs interior 0.8430;
- edge Low sign disagreement: 10.858%;
- Low transition rate changed 5.448% -> 2.845%;
- high-override rate changed 17.585% -> 11.441%.

This is not a numerical reconstruction failure and is not evidence that reflection is automatically correct. It is evidence that the current periodic boundary condition is too influential for the scientific Low/High label definition.

This round therefore does **not** train a new Spread predictor. It adjudicates the deterministic decomposition used to define `L_true` and `H_true` before any Low/State/Z/High model is allowed to run again.

The active question is:

> Can we define an exact additive `S = L + H` split in which Low represents multi-hour market base state, High carries short excursions, and the first/last quarters of the day are not dominated by an arbitrary circular boundary assumption?

## 1. Hard restrictions

1. Do not change the MSR model architecture, feature families, quantiles, Markov logic, Z mediator, LightGBM parameters or evaluation windows in this round.
2. Do not train Low/High/Z models while selecting the decomposition.
3. Do not access 2026-10 market labels or the October final lockbox.
4. Do not use 2026-05-01..2026-08-14 final metrics to select a decomposition.
5. Use only the already-authorized training/development decomposition scope. Selection/adjudication is confined to `DEV_START .. DEV_SELECTION_END` with label maturity respected.
6. Do not modify frozen `src/TafM_改进源码`, `src/lightgbm_源码` or production `electricity_forecast_model2.5`.
7. Do not revert the four production-path drifts reported by docs/68. Investigate provenance only; unknown external edits remain untouched.

## 2. Why the old periodic definition is suspect

The current daily transform treats a 96-point target day as a circular signal. Under periodic extension, p96 is adjacent to p1 of the **same day**. That is convenient mathematically, but the electricity market clock is not physically periodic at that boundary: the temporal successor of D p96 is D+1 p1, not D p1.

The result is a deterministic label artifact: a large mismatch between D p96 and D p1 can be interpreted as high-frequency energy even though those points are almost 24 hours apart in real chronology.

The redesign must therefore remove circular same-day coupling while keeping:

- exact additive reconstruction;
- no target-day feature leakage;
- clear frequency/roughness interpretation;
- deterministic labels;
- no dependence on D+1 truth for the deployed label definition.

## 3. Candidate decomposition family

Three candidates are adjudicated. The current periodic transform remains a reference only.

### B0 — PERIODIC_MODWT_J3_REFERENCE

Existing definition, unchanged:

`L = A_3(periodic sym4)`

`H = S - L`

B0 is not eligible to win unless it passes the new boundary gates; it is retained for reproducibility and delta reporting.

### B1 — REFLECT_MODWT_J3

For each complete 96-point day, create a symmetric reflected extension on both sides, apply the same zero-phase sym4 J3 transfer, and crop the central original 96 points.

Requirements:

- reflection length must be at least the effective J3 filter support; use a fixed predeclared pad (64 quarters is the baseline from docs/68 audit);
- no D-1 or D+1 prices are injected into the label;
- exact output definition remains `H = S - L`, so reconstruction is exact even if the filter approximation itself has numerical error;
- the code must record pad mode, pad length and transfer definition in the artifact.

Reflection is a candidate, not an assumed truth.

### B2 — NATURAL_SECOND_DIFFERENCE_TREND_FILTER

Define Low as the unique solution of the convex quadratic problem

\[
L^* = \arg\min_L \frac12\|S-L\|_2^2 + \frac{\lambda}{2}\|D_2 L\|_2^2,
\]

where `D_2` is the ordinary second-difference operator on the 96-point line:

\[
(D_2L)_t=L_{t+2}-2L_{t+1}+L_t.
\]

There is **no wrap-around row** in `D_2`. Therefore p96 is never connected to p1.

Closed form:

\[
L^*=(I+\lambda D_2^T D_2)^{-1}S,
\]

and

\[
H=S-L^*.
\]

This candidate is deterministic, exact-additive and has natural line-boundary behavior. It is not a neural model and adds no learned parameters from final data.

Instead of an arbitrary lambda grid, parameterize candidates by an interpretable approximate half-power period `T_c` in quarters. For the infinite-grid second-difference smoother,

\[
G(\omega)=\frac{1}{1+16\lambda\sin^4(\omega/2)}.
\]

Setting `G(2*pi/T_c)=0.5` gives

\[
\lambda(T_c)=\frac{1}{16\sin^4(\pi/T_c)}.
\]

Predeclared candidate periods:

- `T_c = 8` quarters (2 h), lambda about 2.91;
- `T_c = 12` quarters (3 h), lambda about 13.93;
- `T_c = 16` quarters (4 h), lambda about 43.15;
- `T_c = 24` quarters (6 h), lambda about 215.3.

No additional periods may be added after viewing adjudication metrics.

## 4. Why B2 is scientifically coherent with MSR

The MSR hypothesis is not ordinary trend/season decomposition. It is:

`Spread = slow market base + fast correction`.

The second-difference penalty says that the Low base should change smoothly in **slope**, not necessarily remain flat. A genuine multi-hour transition can therefore move Low from negative toward neutral/positive, while a one- or two-quarter spike is expensive to place into Low and is preferentially left in `H`.

This is aligned with the observed market statement:

- large magnitude motion is allowed;
- sign/regime should not chatter quarter by quarter without strong evidence;
- High must retain the ability to override a negative Low and produce a short positive Spread event.

The method also removes the arbitrary same-day circular join that failed docs/68.

## 5. Adjudication data scope

No predictive model is fit.

Use only complete Spread label days in:

- diagnostic train: 2023-01-01..2025-12-31 for descriptive stability only;
- selection/adjudication: 2026-01-01..2026-04-28;
- do **not** inspect 2026-05-01..2026-08-14 decomposition scores until a winner is frozen;
- consumed 2026-08-15..2026-09-15 and October remain excluded.

The winner must be frozen from the selection window before any later retrospective report.

## 6. Metrics

For each candidate and each selection day compute all of the following.

### 6.1 Exactness

- max reconstruction error `max |S-(L+H)|`;
- mean reconstruction error.

Hard requirement: max <= 1e-8.

### 6.2 Slow-state persistence

Using the package-wide zero semantics (`S>0` positive, zero non-positive):

- Low sign transitions/day;
- Spread sign transitions/day;
- transition ratio `LowTransitions / SpreadTransitions`;
- Low ABA chatter/day;
- Low dwell median and p90.

### 6.3 Frequency / roughness separation

- `var(L)/var(S)`;
- `var(H)/var(S)`;
- `corr(L,H)`;
- median and p90 of `|Delta L|`;
- median and p90 of `|Delta^2 L|`;
- corresponding H ramp/curvature summaries.

### 6.4 High override behavior

- share of all cells where `sign(S) != sign(L)`;
- among such cells, share with `|H|>|L|`;
- positive-Spread crossing share where Low <= 0 but S > 0;
- negative-Spread crossing share where Low > 0 but S <= 0.

No candidate is allowed to eliminate genuine crossings simply to look smoother.

### 6.5 Boundary artifact diagnostics

Define edge = p1..p8 and p89..p96; interior = p9..p88.

For each candidate compute:

- mean/p90 `|H|` on edge and interior;
- mean/p90 `|Delta L|` on edge and interior, counting only real within-day adjacent pairs;
- mean/p90 `|Delta^2 L|` edge and interior;
- normalized edge High energy:
  `(mean |H|_edge / std(S_edge)) / (mean |H|_interior / std(S_interior))`;
- normalized edge Low curvature:
  `(mean |Delta^2 L|_edge / std(S_edge)) / (mean |Delta^2 L|_interior / std(S_interior))`.

No cross-day or p96->p1 synthetic adjacency may enter these statistics.

### 6.6 Method-sensitivity diagnostics

For B1 and each B2 candidate compare Low against the other boundary-safe candidates, separately edge/interior:

- mean/p90 absolute Low difference;
- sign disagreement;
- transition-count difference.

This is diagnostic only; agreement is not itself proof of physical correctness.

## 7. Hard gates

A candidate is admissible only if all hard gates pass on DEV_SELECTION:

1. reconstruction max error <= 1e-8;
2. Low transition ratio <= 0.70;
3. Low ABA <= Spread ABA and <= 0.75/day;
4. `var(L)/var(S)` in [0.25, 0.95];
5. `var(H)/var(S)` in [0.03, 0.75];
6. positive crossing retention >= 0.50 relative to raw Spread positive crossings represented by `Low<=0, S>0` / analogous decomposition availability; the implementation must document the exact denominator and reject tautological definitions;
7. normalized edge High energy in [0.50, 2.00];
8. normalized edge Low curvature <= 2.00;
9. no same-day circular coupling in implementation for B1/B2;
10. no D+1 truth dependency.

If none pass, output `DECOMPOSITION_REDESIGN_NOT_RESOLVED` and stop. Do not weaken gates after seeing results.

## 8. Winner rule

Among admissible candidates, select using a predeclared lexicographic rule:

1. smallest normalized edge Low curvature;
2. then smallest Low ABA chatter;
3. then smallest transition ratio;
4. then, if still tied within 5% on all three, choose the simpler deterministic operator in this order: B2 natural quadratic smoother > B1 reflected wavelet > B0 periodic reference.

Do not use downstream Spread MAE, Direction, AUC, Z metrics or final-window model performance to choose the decomposition.

After selection, freeze:

- method id;
- wavelet/pad or lambda/T_c;
- exact matrix/filter definition;
- code hash;
- selection data cutoff;
- metrics and winner rule.

## 9. Post-freeze retrospective sanity check

Only after the winner is frozen may it be evaluated on the already-open 2026-05-01..2026-08-14 window, strictly as `RETROSPECTIVE_SANITY_CHECK`, never as new confirmation and never to change the winner.

Report whether edge/interior behavior is directionally stable. If it catastrophically fails, mark the decomposition unstable and stop; do not return to selection and pick another winner.

## 10. Protected-directory drift side task

The four production-path drifts from docs/68 are a separate isolation issue. Before any full new MSR model run:

1. inspect git/status/mtime/process evidence for the two source files and two CPython-3.10 cache files;
2. do not open market/lockbox data;
3. do not revert or overwrite any unknown edit;
4. record whether the changes belong to an external/concurrent workflow;
5. once provenance is understood, freeze a **new read-only production baseline** for the next lab run and verify zero drift relative to that new baseline during the lab experiment.

A provenance-classified external edit does not become MSR code and does not authorize production modification.

## 11. Implementation scope

Add decomposition-redesign code only under:

`src/research96/msr_spread_v1/`

Recommended files:

- extend `decomposition.py` with explicit candidate functions, without deleting the historical periodic implementation;
- add `decomposition_adjudication.py` for D0-only metrics/selection;
- add focused tests for no-wrap matrix structure, exact reconstruction, no D+1 dependency, edge metric semantics and deterministic winner rule;
- add a CLI stage `D0_DECOMPOSITION_ADJUDICATION` or a separate non-model adjudication CLI that cannot accidentally run C1.

Do not overwrite old C0/C1 artifacts. Use a new output namespace:

`outputs/experiments/msr_spread_v1/decomposition_adjudication_20261008/`

## 12. Required output

Return exactly these top-level decisions plus concise evidence:

- `PERIODIC_REFERENCE=<PASS/FAIL>`
- `REFLECT_MODWT_J3=<PASS/FAIL>`
- `NATURAL_TREND_T8=<PASS/FAIL>`
- `NATURAL_TREND_T12=<PASS/FAIL>`
- `NATURAL_TREND_T16=<PASS/FAIL>`
- `NATURAL_TREND_T24=<PASS/FAIL>`
- `SELECTED_DECOMPOSITION=<method id or NONE>`
- `SELECTION_WINDOW=2026-01-01..2026-04-28`
- `FINAL_WINDOW_USED_FOR_SELECTION=NO`
- `NO_DPLUS1_DEPENDENCY=PASS/FAIL`
- `BOUNDARY_REDESIGN=<PASS/FAIL>`
- `PRODUCTION_DRIFT_PROVENANCE=<CLASSIFIED/UNRESOLVED>`
- `READY_FOR_MSR_RERUN=<YES/NO>`

`READY_FOR_MSR_RERUN=YES` requires a selected decomposition, all hard isolation checks, and a newly frozen production read-only baseline with zero drift during this adjudication run.

## 13. Scientific interpretation boundary

Passing D0 only means the Low/High **label definition** is sufficiently stable to justify rerunning the existing MSR mechanism. It does not mean MSR improves Spread prediction.

After D0 passes, the next experiment is a controlled rerun of the same V1a/V1b pipeline with exactly one scientific change: the frozen decomposition operator. Low/State/Z/High features and model hyperparameters remain unchanged for the first rerun, so any delta can be attributed to the decomposition redesign rather than architecture drift.

# Spread24 Senior Review — Agent Research Contract

> Scope: this repository only.
> Purpose: make every Codex/agent/new chat inherit the current scientific state before changing experiments.

## 0. Mandatory startup reads

Before research design, model/feature change, training or evaluation, read this current set:

1. `docs/README.md`
2. `docs/01_业务数据与防泄漏合同.md`
3. `experiments/first_test/96_information_challenge/NATIVE96_FIELD_POLICY_20261005.json`
4. `docs/57_spread_literature_research_protocol_20261006.md`
5. `docs/58_spread_literature_review_ledger.md`
6. `docs/63_msr_spread_v1_final_code_handoff_20261007.md`
7. `docs/69_boundary_safe_low_high_decomposition_adjudication_20261008.md`
8. `docs/71_d0_decomposition_audit_20261009.md`
9. `docs/72_msr_t24_repair_staged_rerun_and_multimonth_validation_20261009.md`
10. `docs/73_msr_t24_retrospective_closure_20261009.md`
11. `docs/74_msr_state_consistency_and_tempered_persistence_repair_20261009.md`
12. `docs/75_msr_state_repair_closure_20261009.md`
13. `docs/76_msr_state_only_experiment_protocol_20261009.md`
14. `docs/79_msr_low_cdf_unary_calibration_closure_20261009.md`
15. `docs/80_msr_low_drift_adaptive_location_calibration_protocol_20261009.md`
16. `docs/81_msr_low_location_recovery_closure_20261009.md`
17. `docs/82_msr_daily_asof_training_and_fast24_protocol_20261009.md`
18. `docs/84_msr_v2_native96_unified_architecture_implementation_spec_20261009.md`
19. `src/README.md`
20. `src/research96/msr_spread_v1/README.md`

Historical protocols, ledgers and handoffs remain evidence, retrieved through docs/README.md as needed. All scientific/isolation rules below remain binding. docs/82 is the current training-paradigm authority (DAILY_ASOF_V1 / FAST-24). docs/84 is the current MSR-V2 architecture and implementation authority and supersedes docs/83's draft architecture only. For V2 implementation, full target-day DA[1:96] is explicitly allowed as an isolated research assumption, while target-day RT/Spread and forbidden actuals remain illegal. New V2 code must live under `src/research96/msr_spread_v2/**`; `src/research96/msr_spread_v1/**` and `src/TafM_改进源码/**` are read-only references and must not be modified. docs/85 is the one-shot Codex implementation prompt, not a scientific authority. docs/81 is the accepted closure of docs/80; docs/79 remains the accepted static-PIT closure; docs/76 is earlier state-only history; docs/72–73 remain retrospective evidence.

The earlier `docs/35_r96_d_information_challenge_execution_protocol.md`, `RUN_MATRIX_IC0_IC4.json`, `run_matrix.py`, `seal_matrix.py`, and single-run `open_metrics.py` belong to `LEGACY_R96_D_V0_DO_NOT_EXECUTE`; retain them only for provenance audit.

For all current R96 work, the main production repository is STRICTLY READ-ONLY:

`D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.5`

The user explicitly forbids modifying that project. Agents may read its docs/code/schema, read its DB connection configuration, and run SELECT-only database audits. Agents must never create, modify, delete, rename, format, stage, commit, or otherwise write any file in the 2.5 project, and must never write to the production database.

All new R96 code, exports, snapshots, manifests, features, models, predictions, and experiment artifacts must stay inside this `spread24_senior_review` repository.

The production project’s `AGENTS.md`, `.agents/skills/efm3-lessons/SKILL.md`, `electricity_forecast_model2.5/docs/DATA_CONTRACT_96.md`, `electricity_forecast_model2.5/docs/LEAKAGE_AUDIT_96.md`, and `utils/asof_view_96.py` remain read-only authoritative references for current formal96 semantics.

## 1. Scientific state — do not regress

The historical DEV28 panel is exhausted. It may be used only for reproduction, debugging, mechanism diagnosis, and retrospective slicing. It must never again select a model, feature set, threshold, loss, retrieval k, gate, or hyperparameter.

The first true prospective C1-C4 CONFIRM panel has been consumed by the one-shot Q2 vs V3 test:

- 23 days / 552 hourly slots
- C1: 2026-08-15..2026-08-21
- C2: 2026-08-31..2026-09-05
- C3: 2026-09-06..2026-09-10
- C4: 2026-09-11..2026-09-15
- three paired seeds: 20260924 / 20260925 / 20260926
- 138/138 predictions sealed before metrics were opened
- FINAL LOCKBOX 2026-10-01..2026-10-31 remains unread and must not be accessed without separate explicit authorization

Prospective result:

- Q2 mean Raw = 0.5181
- V3 mean Raw = 0.5399
- paired Raw gain = +0.0217
- bootstrap 95% CI = [-0.0030, +0.0465]
- Q2 mean Balanced = 0.5143; V3 = 0.5353
- Q2 mean +Recall = 0.3656; V3 = 0.3569
- Q2 mean -Recall = 0.6631; V3 = 0.7138
- Q2 AUC = 0.5091; V3 AUC = 0.5195
- official classification = `STOP_CURRENT_FEATURE_ARCHITECTURE_SEARCH`

Interpretation: V3 contains a real but modest residual signal; it did not solve positive-event recognition and did not approach the 65% goal. Current evidence does not justify more backbone/hidden/depth/k/gate/threshold/class-weight micro-tuning inside the same feature universe.

## 2. Current research question

The active research question is:

> Can positive-spread and negative-spread accuracy/recall improve together, rather than trading one class for the other?

Treat this primarily as a **separability / information** problem, not a threshold problem.

A method that raises Raw only by shifting predictions toward one class is not a breakthrough. Every formal comparison must report at least:

- Raw direction accuracy
- Balanced accuracy
- positive recall
- negative recall
- AUC/ranking quality
- predicted positive-hour count / calibration
- temporal block/month stability
- majority-class baseline

Priority is to find variables/mechanisms that increase class separability and improve both sides, especially positive-event and large-positive recognition.

## 3. Research direction after prospective failure

The current phase is:

`MISSING_MARKET_INFORMATION_AUDIT + 96_POINT_INFORMATION_RESEARCH`

The next high-value hypotheses are about market information, including:

- positive/negative reserve tightness
- load / wind / solar / renewable forecast errors
- provisional/temporary actual vs forecast discrepancies available at the decision boundary
- partial RT-DA spread state before the boundary
- full-system load vs direct-load gap
- interchange / boundary forecasts and congestion proxies
- ramp and reserve-margin pressure
- forecast uncertainty / tail quantiles, not only point forecasts
- outages / unit availability / bid-stack / unit-constraint information if a trustworthy source becomes available

Do not reopen generic model-family search until a new-information experiment demonstrates materially better OOS separability.

## 4. 96-point research boundary

The main project formal96 production path uses Dynamic-v1:

`DB sync -> immutable D/T Snapshot -> FeatureView -> model legs`

Formal FeatureView routing for decision-day fundamentals is:

`final actual -> RealityTmp -> same-field forecast -> latest closed same-period actual -> recent same-period median`

Decision-day RT routing is:

`snapshot RT -> same-day DA -> latest closed same-period RT -> recent same-period median`

Target-day actual/DA/RT truth is masked.

Historical `HISTORICAL_PROXY_V1` uses p56 / 14:00 only as operational proxy evidence. It is explicitly `UNVERIFIED_LEGACY_VINTAGE` and must not be presented as a historically authentic live snapshot.

For research, if historical real snapshots are unavailable, use a separately named protocol such as:

`EXPERIMENTAL_FIXED_BOUNDARY_96_V1`

Rules:

- fixed boundary must be declared before results;
- p56 / 14:00 is the natural first boundary because it aligns with the legacy strict business origin and the existing historical proxy;
- only pre-boundary realized cells may be treated as truly visible;
- post-boundary effective features must use a frozen fallback rule;
- synthetic/effective FeatureView values are features only, never supervised truth;
- target-day RT/spread/actual remains label-only;
- any historical temporary-actual field without capture-time vintage evidence is forbidden from being retrospectively fabricated;
- experiment metadata must say `PROXY/EXPERIMENTAL`, never `LIVE_SNAPSHOT`.

96-to-24 aggregation, when needed for comparison, follows the established project rule: p1-p4 -> h1, ..., p93-p96 -> h24 by arithmetic mean. Preserve 96-point primary metrics separately.

## 5. Historical inventory facts (2026-09-30 audit)

These facts describe the earlier mirror audit, not every later canonical snapshot. The current MSR source BoundaryForecast coverage/vintage audit in docs/68 finds long 2023-04 onward coverage but no dev origin-time capture proof. Historical conclusions below are retained, not substituted for the new source audit.

Read-only audit of the authoritative 96 table found long history for:

- positive reserve forecast
- negative reserve forecast
- the core forecast/actual fundamentals
- DA and RT prices

Positive/negative reserve forecasts have 1,681 complete days through 2026-08-16 and are therefore the strongest immediate new-information family for historical experiments.

Recent-only fields currently have very short history in the local authoritative mirror (about 12 days around 2026-09-18 onward):

- full-system load forecast / actual
- temporary actual fundamentals
- boundary forecasts (full-system load, direct load, interchange, wind, solar)
- boundary nuclear forecast currently has no usable observations

However, temporary actual fundamentals are now a **potentially backfillable core family**: the main project crawler has dedicated RealityTmpData collection and the user has started historical backfill. Their research status is therefore `BACKFILL_IN_PROGRESS / PROVENANCE_PENDING`, not permanently forward-only.

Before using backfilled temporary actuals in historical experiments, verify for sampled dates that the endpoint returns the historical temporary publication rather than a final/latest-state reconstruction; retain raw response/source endpoint/date/hash evidence. Never substitute final actuals for missing temporary actuals.

Boundary forecasts and other short-history fields remain forward-collection candidates unless a trustworthy historical source is independently established.

## 6. Literature discipline

For DA / RT / spread paper search and review, first read:

- `docs/57_spread_literature_research_protocol_20261006.md`
- `docs/58_spread_literature_review_ledger.md`

The canonical A/B paper-review ledger is now:

`docs/58_spread_literature_review_ledger.md`

Use `experiments/first_test/RESEARCH_LITERATURE_LEDGER.md` only as the broader historical experiment-literature ledger. `docs/46_*`, `docs/47_*` and the old ledger are provenance / experiment-history sources only: **never rebuild the active A/B paper pool from them, and never let their obsolete paper classifications override `docs/57`/`docs/58`.** If a reviewed paper changes a formal experiment hypothesis, add only the project implication / testable decision there; do not duplicate the full paper review.

Current Route A paper-selection gate (2026-10-07): default to 2022–2026, strong/top venue, strict DA–RT–Spread structural coupling, and problem/mathematics-first methodology. Older papers may remain only as explicitly accepted foundations; single-leg DA/RT, ordinary probabilistic postprocessing, and architecture-stacking papers do not enter the active A pool merely because they are recent.

For each paper preserve:

- citation / DOI / publication status
- Route A or B and point/probabilistic classification
- task, market, resolution, forecast origin and horizon
- original spread sign convention and mapping to project `RT-DA`
- relevant inputs / mathematical mechanism
- key empirical finding and exact compared baselines
- direct implication for this project
- caveat / mismatch with our information boundary
- one testable hypothesis or decision when applicable

Distinguish peer-reviewed papers from preprints/watchlist items.

Do not claim that a traditional/statistical method beats deep learning unless the paper's test setting, metric and named deep baselines support that statement. Do not cite a paper as support for a model family when the actual useful lesson is its information set or market mechanism.

## 7. Experiment discipline from now on

Before any new formal experiment:

1. write the forecast origin and feature availability contract;
2. identify whether each feature has true historical vintage or only latest-state history;
3. define selection/dev/confirm/final panels before viewing target metrics;
4. use a simple signal audit first (univariate/GLM/LightGBM or equivalent) before a new deep architecture;
5. report whether the new information improves both positive and negative class behavior;
6. keep consumed C1-C4 out of future selection/confirmation;
7. do not access October FINAL LOCKBOX unless separately authorized.

If a new 96 information family does not materially improve AUC/Balanced/+Recall/-Recall on a fresh validation design, stop that information family before adding model complexity.

## 8. Production isolation

This lab is a research environment. Do not modify the main project's formal96 production Snapshot/FeatureView/runner/ledger behavior as a side effect of research.

Any experimental Snapshot builder, feature view, proxy, or 96 spread dataset belongs in this lab / experiment namespace until separately promoted by the user.

No production promotion is implied by a research result.

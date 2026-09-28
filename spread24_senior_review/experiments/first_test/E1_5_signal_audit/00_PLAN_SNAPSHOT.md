# E1.5-A Direction Signal Audit — Plan Snapshot (pre-registration record)

This file records the design **exactly as authorized before any result was produced**.
It is the frozen reference for everything in this directory. No item below was changed
after seeing results.

Source of authority: `experiments/first_test/08_E1_5_DIRECTION_SIGNAL_AUDIT_PLAN.md`,
executed under the standing instruction *"不要重新设计实验，严格执行 …"*.

---

## 1. Stage 1 — blocking environment gate

| item | requirement |
|---|---|
| interpreter | formal GPU env `epf-2` |
| defect | `shap` missing; 5 test modules failed collection; `requirements.txt` did not declare `shap` |
| repair | install `shap` into `epf-2`; declare `shap` in the project dependency contract; **upgrade nothing unrelated** |
| gate | canonical suite must reach **94 PASS** before any experiment runs |
| on failure | STOP, report the environment problem only, run no experiment |

Recorded in `environment_audit.json`. The `shap==0.51.0` / "no unrelated upgrades" pair was
found to be mutually unsatisfiable (`shap>=0.50` forces `numpy>=2`, which would have moved
`numpy 1.26.4` — the version under which E1's 4/4 bit-exact reproducibility was established).
Resolved by explicit user decision: `shap==0.49.1`, `numpy` untouched. `shap` is imported only
by frozen selector construction, so it cannot reach any E1 or E1.5-A number.

## 2. Evaluation scope (reused verbatim from E1 — not re-chosen)

28 target days / 672 slots:

- W1 `2026-02-12 .. 2026-02-18`
- W2 `2026-04-12 .. 2026-04-18`
- W3 `2026-06-12 .. 2026-06-18`
- W4 `2026-08-07 .. 2026-08-13`

## 3. Anti-leakage contract (hard boundaries)

- target `TARGET = RT − DA`; `Y_source = DA − RT`; sign flip via the existing adapter only
- forecast origin `D-1 14:00`; auxiliary labels only `S <= D-2`
- target-day truth used **for evaluation only** — never to fit, select, or threshold anything
- selector frozen (`status == "FROZEN"`, asserted at load; refuses to run otherwise)
- E1 TafM checkpoints frozen — read only, no retraining, no re-selection, no threshold change
- no lockbox; no full Jan–Aug DEV

## 4. Models under audit

**TafM (reused from completed E1 runs, read-only):**

| id | E1 variant | meaning |
|---|---|---|
| T0 | M0 | `joint_v21` + `v21_guardrail` |
| T1 | M1 | `joint_v21` + `direction_first` |
| T2 | M2 | `dir_only` + `direction_first` (current screening candidate) |

T0 and T1 are retained alongside T2 because the audit must support *attribution*, not just a
winner. T2 being the candidate does not license dropping the other two.

**Pre-registered baselines (no tuning permitted):**

| id | definition |
|---|---|
| B0 | always non-positive (`direction_hat = 0`) — class-prior sanity baseline |
| B1 | per-hour majority over legal `[D-181, D-2]` labels, grouped by `hour_business` 1..24; fail-closed if an hour has no legal history (only permitted fallback is that legal window's global majority, and it must be recorded) |
| B2 | XGBoost Direction, frozen hyperparameters/features/label/seed, recent legal 180d, cutoff `D-2` |
| B3 | identical to B2; only the training history changes `180d → all legal <= D-2` |
| B4 | `run_strict_lightgbm_baseline(..., window_days=180)`, direction = sign of the regression output |

Frozen XGBoost hyperparameters (verbatim, `n_estimators=90, max_depth=4, learning_rate=0.05,
subsample=1.0, colsample_bytree=0.8, reg_lambda=1, objective="binary:logistic",
tree_method="hist", n_jobs=1, random_state=seed, eval_metric="logloss"`), `seed = 20260924`.

B2 vs B3 answers exactly one question: *is recent-regime information materially better than
expanding history?* No other difference is permitted between them.

B4 keeps the existing adapter's DA−RT → RT−DA sign contract, reports `rank_auc`, and marks
Brier `N/A` — fabricating a calibrated Brier from a signed regression output is forbidden.

## 5. Metric contract (every model, no exceptions)

- **Overall**: Raw Direction Accuracy, Balanced Accuracy, +Recall, −Recall, AUC/rank-AUC where
  applicable, Brier where applicable, predicted-positive fraction, one-class prediction days,
  `+Recall = 0` days, `−Recall = 0` days
- **W1–W4 separately**: Raw, Balanced, +Recall, −Recall, ppf, true positive prevalence,
  majority-baseline Raw — reporting overall alone is forbidden
- **Hour segments**: H1 = 1–8, H2 = 9–16, H3 = 17–24 — Raw, Balanced, +Recall, −Recall, prevalence
- **Day level**, per target day: Raw, Balanced, +Recall, −Recall, ppf, truth positive fraction

Semantics copied from `src/TafM_改进源码/metrics.py`: `truth = y > 0` (exact zero is
non-positive; `>= 0` is forbidden), `balanced = mean([+Recall, −Recall])`, AUC = Mann–Whitney
with average ranks for ties.

## 6. TafM signal diagnostics (no retraining)

1. **Probability separation** — mean/median `p | y>0` vs `p | y<=0` and the gap; overall, W1–W4,
   H1/H2/H3.
2. **Oracle threshold** — one post-hoc threshold curve over already-completed E1 predictions.
   Every output row must carry `LEAKY_DIAGNOSTIC_ONLY`, `NOT_A_MODEL_RESULT`,
   `NOT_FOR_PROMOTION`. Report `threshold=0.5`, oracle-Raw, oracle-Balanced. Writing any of it
   back into a formal model is forbidden. It may never be described as a 65% result.
3. **Prior gap** — per model per window, `model Raw − same-window majority baseline Raw`.

## 7. Case judgment matrix (plan §10) — fixed in advance

| case | condition | verdict |
|---|---|---|
| A | tree baseline ≥65 and clearly above TafM | `MODEL_UTILIZATION_GAP` |
| B | `XGB-180 >> XGB-expanding` | `RECENCY_REGIME_SIGNAL` |
| C | `XGB-expanding >= XGB-180` **and** tree baseline > TafM | `ARCHITECTURE_OR_OPTIMIZATION_GAP` |
| D | TafM AUC/Brier clearly better than baselines but Raw conversion poor | `DECISION_LAYER_GAP` / `TAFM_RANKING_SIGNAL` |
| E | all non-trivial models ~55–60% with AUC near 0.5 | `FEATURE_SIGNAL_LIMITED` |

## 8. 65% reading rules

Report overall Raw, W1–W4 Raw, min-window Raw, window std, Balanced, +Recall, −Recall.

- overall ≥65% but a window collapses → may **not** claim `跨月稳定65+`
- Raw ≥65% obtained mainly by one majority class (Balanced ≈ 0.5 or a class Recall ≈ 0) →
  must be marked `MAJORITY_DRIVEN`, may **not** be claimed as genuine model success

Forcing 65% is not a goal of this round. The goal is to locate the source of usable Direction signal.

## 9. Isolation and stop rules

- audit runner lives entirely under `experiments/first_test/E1_5_signal_audit/`
- `E1_mini` and every other existing result tree are read-only, never overwritten
- `src/` was not modified
- the existing `direction-benchmark` is **not** invoked (it would duplicate TafM training);
  only its frozen XGBoost baseline logic is re-implemented here, experiment-locally
- on completion: **STOP**. No TabM-only, no Temporal-only, no TafM recent180 training, no
  checkpoint equivalence band, no calibration, no threshold tuning, no protected-gradient,
  no Stage B, no full DEV, no lockbox
- no stage, no commit

## 10. Required artifacts

`00_PLAN_SNAPSHOT.md`, `environment_audit.json`, `baseline_daily_metrics.csv`,
`model_summary.csv`, `window_metrics.csv`, `hour_segment_metrics.csv`, `prior_gap.csv`,
`probability_separation.csv`, `oracle_threshold_DIAGNOSTIC_ONLY.csv`, `paired_vs_tafm.csv`,
`E1_5_summary.md`, `E1_5_GATE.md`, `figures/` (≥5), `predictions/`, audit scripts.

# E5-A Regime Diagnosis — Frozen Plan

STATUS=AUTHORIZED_EXECUTED
DATE=2026-09-26
PARENT=docs/24_E5_A_regime_diagnosis_experiment_spec.md
DESIGN=docs/25_E5_regime_aware_architecture_design_v2.md
SCOPE=SCREENING_ONLY
DATA_CONTRACT=docs/01_业务数据与防泄漏合同.md

## Question

Do origin-available market-state clusters have meaningfully different held-out RT-DA direction distributions / error profiles, and does a cluster-conditioned historical baseline outperform a non-regime historical baseline? This stage does not modify a model.

## Frozen inputs and feature construction

- Use only the nine already-selected, E4-A registered regime features from `src/TafM_改进源码/direction_postprocess.py::REGIME_FEATURES`.
- One observation per target day: mean of each feature's 24 forecast-horizon values. These are forecast/context inputs already in the frozen sequence at the D-1 14:00 origin; no actual or label-derived columns are included in the clustering matrix.
- Fail closed if any of these feature cells are quarantined or non-finite for a fitted/evaluated sample.
- Target is RT-DA; positive iff `y_model > 0`; zero is non-positive. Labels are used only for training-only baseline summaries and post-assignment diagnostics.

## Chronological folds

Four pre-existing frozen DEV windows, seven days each:

| Fold | Evaluation days | Unsupervised/scoring history cutoff |
|---|---|---|
| W1 | 2026-02-12..2026-02-18 | 2026-02-10 (D-2) |
| W2 | 2026-04-12..2026-04-18 | 2026-04-10 (D-2) |
| W3 | 2026-06-12..2026-06-18 | 2026-06-10 (D-2) |
| W4 | 2026-08-07..2026-08-13 | 2026-08-05 (D-2) |

Only eligible dates on/before each cutoff enter scaler, cluster fitting, and historical baselines. Evaluation covariates and labels do not enter fitting. Earlier evaluation windows may enter a later fold only after their target dates meet the later fold's D-2 cutoff.

## Pre-registered methods

- StandardScaler fit on each fold's historical daily matrix only.
- KMeans K=3,5,7 and GaussianMixture K=3,5; fixed seed 20260924 for reported assignments. Three fixed seeds (20260924, 20260925, 20260926) quantify initialization stability via held-out ARI. No K/model selection or cluster relabeling by target labels.
- KMeans K=5 is the sole primary diagnostic; all other settings are sensitivity analyses. Primary cluster structure is called non-trivial only when at least 3 clusters occur in each fold and mean held-out pairwise-seed ARI is at least 0.60.
- HDBSCAN is omitted if unavailable in the pinned runtime; no package installation.
- Report held-out cluster size, RT-DA positive rate, cluster-hour historical-majority direction accuracy, cluster-hour historical-median magnitude MAE, train-only 90th-percentile extreme-spread rate, and paired comparison vs global-hour historical baselines.
- Bootstrap differences by target day (10,000 draws, seed 20260924); no threshold tuning.

## Gate interpretation

E5-A is a diagnostic gate, not a model KPI. For primary KMeans K=5, PASS requires (i) non-trivial stable structure and held-out positive-rate range >=10 percentage points in at least 3/4 windows, and (ii) cluster-hour baseline Raw is higher than global-hour baseline in at least 3/4 windows with pooled paired 95% CI above zero, without an entire-window collapse. Otherwise `NO_CLEAR_REGIME_SIGNAL` / stop; mixed diagnostics are reported without promoting E5-B.

## Limits

No model/code/config/selector/data-contract changes; no new features; no full Jan-Aug rerun; no lockbox; no E5-B/E5-C. All outputs stay under this experiment directory. The model design's `63%` target is not tested by E5-A and cannot be inferred from this diagnosis.

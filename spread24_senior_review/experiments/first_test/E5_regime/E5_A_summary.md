# E5-A Regime Diagnosis — Results

**Status: SCREENING COMPLETE; stop for review. Gate: `NO_CLEAR_REGIME_SIGNAL`.** This is not a model change and does not establish or refute a market regime in general; it says the frozen E5-A criteria were not met on this pre-existing 28-day DEV panel.

## Protocol and audit

- Frozen evaluation days: 2026-02-12..18, 2026-04-12..18, 2026-06-12..18, 2026-08-07..13 (28 days / 672 slots).
- Fit each fold's scaler, unsupervised clusterer, direction/magnitude historical baselines on eligible target days no later than that fold's D-2 cutoff. Test covariates/labels are excluded from fitting. Split audit confirms max train dates at the exact cutoffs and no target-day overlap.
- Clustering uses only the nine already-selected E4-A regime features, aggregated as per-day means across the 24 forecast horizons. No new features or model training. All nine fields are checked for quarantine and finite values.
- KMeans K=3/5/7 and GMM K=3/5 ran on each chronological fold. Three fixed seeds assessed assignment stability; KMeans K=5 was pre-registered primary. HDBSCAN was not run because it is absent from the runtime; no installation was attempted.
- Reproduction command: `python experiments/first_test/E5_regime/run_e5_a.py` (Python 3.11.9, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.5.2). No resolvable git commit ID was available in the current workspace; `E5_A_gate.json` records `git_commit: null`.
- Frozen source SHA256: `a1b86f95…349ea`; sequence manifest SHA256: `9144bbed…ce7a82`; selector SHA256 and package versions are in `E5_A_gate.json`. No canonical model, feature contract, selector, or data files changed.

## Primary KMeans K=5 results

The diagnostic baseline predicts each hour using a training-only cluster×hour majority sign (or cluster×hour median magnitude), compared with an hour-only historical baseline. Thresholds, scaling and all cluster fits use past data only.

| Window | Train days (end) | Held-out clusters | Positive-rate range | Cluster-hour Raw | Hour-only Raw | Δ Raw | Magnitude MAE: cluster / hour-only |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 1,494 (Feb 10) | 1 | 0.00pp | 63.10% | 58.33% | +4.76pp | 82.62 / 87.78 |
| W2 | 1,553 (Apr 10) | 2 | 9.58pp | 47.62% | 44.05% | +3.57pp | 100.60 / 101.19 |
| W3 | 1,614 (Jun 10) | 2 | 21.53pp | 60.12% | 57.14% | +2.98pp | 100.07 / 95.23 |
| W4 | 1,670 (Aug 5) | 2 | 9.03pp | 60.71% | 63.10% | -2.38pp | 57.71 / 56.96 |

Pooled paired day-bootstrap (10,000 draws, seed 20260924): **ΔRaw +2.23pp**, 95% CI **[-2.53pp, +6.85pp]**, 13 better / 7 tied / 8 worse days. The interval crosses zero. Although point Raw is higher in 3/4 windows, held-out KMeans K=5 assignments occupy only 1–2 of the five clusters in every window, and positive-rate heterogeneity reaches the pre-registered 10pp threshold in only 1/4 windows. Seed ARI is 0.75–1.00, but stable assignments do not overcome the lack of held-out cluster coverage/separation.

Balanced accuracy, recalls, train-only extreme threshold and cluster-level positive-rate/accuracy/magnitude/extreme diagnostics are in `fold_metrics.csv` and `cluster_metrics.csv`. Sensitivity results for all five algorithm/K configurations are in the same tables. They are descriptive only; no configuration was selected after viewing labels.

## Gate and decision

Pre-registered PASS required stable/non-trivial clusters, ≥10pp held-out positive-rate range in at least 3/4 windows, and a cluster-hour baseline beating the hour-only baseline in at least 3/4 windows with pooled paired CI above zero. Only the last point's point-estimate count passed (3/4); the heterogeneity (1/4), held-out cluster coverage, and CI conditions failed.

**Decision: do not proceed to E5-B or E5-C under this plan.** No state encoder/MoE implementation is justified by these results; no features are deleted or promoted, no model is changed, and no full Jan-Aug rerun or lockbox was used. Any new E5 route requires human review and a new, pre-registered design.

## Artifacts

- `00_PLAN_SNAPSHOT.md` — frozen protocol / Gate.
- `run_e5_a.py` — reproducible analysis script.
- `split_audit.csv`, `fit_diagnostics.csv`, `cluster_stability.csv`.
- `cluster_assignments.csv`, `cluster_metrics.csv`, `fold_metrics.csv`.
- `E5_A_gate.json` — machine-readable final gate and provenance.

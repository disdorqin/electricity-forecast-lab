# E5-A Gate

**E5-A = `NO_CLEAR_REGIME_SIGNAL`**  
**Decision: STOP; no E5-B/E5-C implementation authorized by this experiment.**

## Frozen Gate audit (primary: KMeans K=5)

| Criterion | Required | Observed | Result |
|---|---|---|---|
| Stable/non-trivial held-out structure | ≥3 occupied clusters in each fold; mean seed ARI ≥0.60 | Occupied clusters W1-W4 = 1/2/2/2; ARI = 1.00/0.75/1.00/1.00 | FAIL |
| Held-out sign heterogeneity | positive-rate range ≥10pp in ≥3/4 folds | 0.00/9.58/21.53/9.03pp; only 1/4 | FAIL |
| Regime-conditioned baseline | higher Raw in ≥3/4 folds and pooled paired 95% CI above zero | higher in 3/4; pooled Δ=+2.23pp, CI [-2.53,+6.85]pp | FAIL |

Chronological audit: all fitting/scaling dates are at or before the fold's D-2 cutoff; no evaluation day entered fitting. Labels were used only for past-only historical baselines and post-assignment scoring. HDBSCAN was omitted (not installed). Full values and hashes: `E5_A_gate.json`, `split_audit.csv`, and `E5_A_summary.md`.

## Scope

This is a DEV-panel diagnostic on 28 dates, not a 63% model-success test or a claim that regimes never exist. Canonical V2.1, formal features, selector, source assets and production defaults remain unchanged. No full DEV rerun, lockbox use, E5-B, or E5-C.

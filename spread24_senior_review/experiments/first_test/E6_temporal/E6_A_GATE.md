# E6-A Gate

**E6-A status: `NO_CLEAR_TEMPORAL_STRUCTURE_SIGNAL` — STOP before TemporalEncoder modification.**

## Evidence assessment

| Source | Overall evidence | Cross-window assessment | Gate interpretation |
|---|---|---|---|
| Recent spread (3h) | Raw 42.86%, Balanced 46.59%; +Recall 59.92%, -Recall 33.26% | W1 predicts all-positive (Raw 25%); inconsistent elsewhere | Too short / unstable |
| Recent spread (6h) | Raw 49.40%, Balanced 52.61% | W1 collapse (Raw 25%); better W3/W4, not W1/W2 | Weak recent-state hint only |
| Recent spread (14h) | Raw 50.89%, Balanced 52.05% | W3 Raw 61.90%, but W4 falls to 43.45% | Not stable |
| Same-hour latest legal lag | Raw 43.01%, Balanced 46.34% | Below 50% Raw in all windows | No useful persistence evidence |
| Weekly same-hour majority | Raw 45.83%, Balanced 48.55% | W3 is 57.14% Raw; remaining windows are weak | Periodic pattern not stable |
| Existing 28d same-hour positive-rate | Raw 62.50%, Balanced 53.26%, +Recall 20.25% | Predicts positive for only 16.07% of slots; +Recall is 4.76% (W1) and 1.67% (W3) | Raw mainly reflects negative-class prediction, not robust periodic direction |
| Current TemporalEncoder (temporal-only) | Raw 50.00%, Balanced 47.47% | Raw ranges 44.64–56.55% across windows | Standalone current representation is weak/unstable |

The all-negative reference is Raw 63.99%, Balanced 50%, +Recall 0%, -Recall 100%; its Raw is driven by the 63.99% non-positive label share and is not a useful two-class direction model. Recent 6h/14h state has a small Balanced-Accuracy lift over .50 overall, but the lift and class recalls are not stable by window. These deterministic diagnostics do not establish a time structure worth encoding.

## Decision

- Do not implement E6-B / modify the TemporalEncoder from this evidence.
- Preserve V2.1 canonical behavior, selector, features, information boundary and production defaults.
- Any follow-up requires a separately reviewed, pre-registered design with a specific hypothesis and a stronger held-out confirmation criterion. No hidden tuning/threshold selection is authorized.

## Audit

- 28/28 target days (672 slots) passed sequence availability and selected-feature quarantine eligibility.
- Existing TemporalEncoder outputs were reused from saved E2-A temporal-only runs; all source/sequence/selector hashes matched. Duplicate 2026-02-13 runs had identical predictions and labels. Saved preprocessing fit dates were no later than D-2.
- No training, feature addition, selector/model/data-contract changes, threshold adjustment, class weighting, ensemble or lockbox access occurred.
- Full metrics: `baseline_metrics.csv`; exact predictions: `baseline_predictions.csv`; run provenance: `run_manifest.json` and `temporal_run_audit.csv`.

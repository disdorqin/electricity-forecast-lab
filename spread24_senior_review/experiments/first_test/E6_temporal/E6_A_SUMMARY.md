# E6-A Temporal Diagnostic Summary

**Result: diagnostic complete; `NO_CLEAR_TEMPORAL_STRUCTURE_SIGNAL`.** This is a frozen 28-day DEV-panel screen, not a production result or a full Jan–Aug evaluation. No model, selector, data contract, or production default was modified.

## Protocol

- W1 2026-02-12..18; W2 2026-04-12..18; W3 2026-06-12..18; W4 2026-08-07..13: 28 days / 672 hourly targets.
- History is the existing 168-hour `X_hist`, with origin D-1 14:00. Recent state uses only the final 3/6/14 observed spread values. Same-hour lags follow the frozen 168-hour index mapping; when D-1 h15..h24 lags are not yet observed, the latest available same-hour observation is used. Weekly baseline votes across legal same-hour lags 1..7.
- To cover the frozen asset's longer periodic summary without adding features, the periodic diagnostic also uses selected `spread_same_slot_28d_positive_rate` (its upstream contract is shifted by 2 days). No 14-day raw same-hour history exists in the 168-hour temporal input, so none was synthesized.
- Current TemporalEncoder row reuses the pre-existing E2-A `architecture_mode=temporal_only`, `objective_mode=dir_only` outputs on the exact 28 days; no training was run. Two duplicate 2026-02-13 artifacts were verified identical and the latest run folder was selected. Frozen source, sequence and selector hashes match; saved training preprocessing ends no later than D-2. Details are in `run_manifest.json` and `temporal_run_audit.csv`.
- Fixed sign rule: positive iff value > 0; ties/zero are non-positive. No threshold tuning, class weights, feature expansion, postprocessing, or ensemble. All-negative is included as an orientation/reference baseline.

## Overall results

| Temporal source | Raw | Balanced | Positive recall | Negative recall |
|---|---:|---:|---:|---:|
| All-negative reference | 63.99% | 50.00% | 0.00% | 100.00% |
| Recent spread, mean last 3h | 42.86% | 46.59% | 59.92% | 33.26% |
| Recent spread, mean last 6h | 49.40% | 52.61% | 64.05% | 41.16% |
| Recent spread, mean last 14h | 50.89% | 52.05% | 56.20% | 47.91% |
| Same-hour, latest legal lag | 43.01% | 46.34% | 58.26% | 34.42% |
| Weekly periodic, same-hour 1–7d majority | 45.83% | 48.55% | 58.26% | 38.84% |
| Periodic, existing same-slot 28d positive rate | 62.50% | 53.26% | 20.25% | 86.28% |
| Current TemporalEncoder, temporal-only | 50.00% | 47.47% | 38.43% | 56.51% |

The all-negative Raw reference is high because only 242/672 labels (36.01%) are positive; it is not a useful positive-class predictor. Raw must therefore be read alongside balanced accuracy and both recalls.

## W1–W4 results

Each window cell reports **Raw / Balanced / +Recall / -Recall**.

| Source | W1 | W2 | W3 | W4 |
|---|---|---|---|---|
| All-negative reference | 75.00 / 50.00 / 0.00 / 100.00 | 49.40 / 50.00 / 0.00 / 100.00 | 64.29 / 50.00 / 0.00 / 100.00 | 67.26 / 50.00 / 0.00 / 100.00 |
| Recent 3h | 25.00 / 50.00 / 100.00 / 0.00 | 50.60 / 50.68 / 43.53 / 57.83 | 45.24 / 47.04 / 53.33 / 40.74 | 50.60 / 53.48 / 61.82 / 45.13 |
| Recent 6h | 25.00 / 50.00 / 100.00 / 0.00 | 49.40 / 49.15 / 70.59 / 27.71 | 60.71 / 55.00 / 35.00 / 75.00 | 62.50 / 61.39 / 58.18 / 64.60 |
| Recent 14h | 51.19 / 46.83 / 38.10 / 55.56 | 47.02 / 46.94 / 54.12 / 39.76 | 61.90 / 60.74 / 56.67 / 64.81 | 43.45 / 50.97 / 72.73 / 29.20 |
| Same-hour latest legal lag | 36.90 / 48.41 / 71.43 / 25.40 | 39.29 / 39.23 / 43.53 / 34.94 | 48.81 / 50.93 / 58.33 / 43.52 | 47.02 / 53.15 / 70.91 / 35.40 |
| Weekly periodic, same-hour 1–7d majority | 31.55 / 43.25 / 66.67 / 19.84 | 45.24 / 45.03 / 62.35 / 27.71 | 57.14 / 55.56 / 50.00 / 61.11 | 49.40 / 50.72 / 54.55 / 46.90 |
| Periodic, existing same-slot 28d positive rate | 72.02 / 49.60 / 4.76 / 94.44 | 52.38 / 52.62 / 32.94 / 72.29 | 61.90 / 48.52 / 1.67 / 95.37 | 63.69 / 55.74 / 32.73 / 78.76 |
| Current TemporalEncoder, temporal-only | 53.57 / 46.03 / 30.95 / 61.11 | 44.64 / 44.74 / 36.47 / 53.01 | 45.24 / 44.81 / 43.33 / 46.30 | 56.55 / 52.77 / 41.82 / 63.72 |

All metric values are percentages. Machine-readable full precision is in `baseline_metrics.csv`; hourly predictions and lag provenance are in `baseline_predictions.csv`.

## Interpretation

- The clearest descriptive hint is **short recent state (6–14h)**: its overall Balanced Accuracy is 52.61% / 52.05%, but neither is stable across windows and Raw stays below the all-negative reference. The 3h rule collapses to all-positive in W1.
- **Same-hour persistence and weekly same-hour majority** are weak on this panel (overall Raw 43.01% and 45.83%). W3 is relatively better for weekly voting, but that advantage does not carry to W1/W2/W4. The historical 28-day positive-rate summary has 62.50% Raw, but predicts positive only 16.07% of slots (positive recall 20.25%); it is largely a negative-class predictor, not robust periodic sign recovery.
- The **current TemporalEncoder alone** is near chance on balanced accuracy (47.47%) and varies from 44.64% to 56.55% Raw by window. It is not evidence that the full Q2 model's temporal branch is useless; this is the existing temporal-only ablation, not a causal branch attribution inside Q2.

The pattern points to a tentative recent-state signal, not a robust, isolated temporal structure that justifies an encoder rewrite. See `E6_A_GATE.md` for the stop decision.

## Reproduction artifacts

- `00_PLAN_SNAPSHOT.md` — fixed definitions and information boundary.
- `run_e6_a.py` — deterministic diagnostic/reuse script; command: `python experiments/first_test/E6_temporal/run_e6_a.py`.
- `run_manifest.json`, `temporal_run_audit.csv` — provenance and existing E2-A run identity checks.
- `baseline_metrics.csv`, `baseline_predictions.csv` — full-precision outputs.

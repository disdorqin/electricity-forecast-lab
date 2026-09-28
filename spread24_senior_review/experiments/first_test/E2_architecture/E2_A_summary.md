# E2-A — Architecture Big-Block Ablation: Summary

A0 `FULL_CURRENT` (canonical V2.1 fusion, 28-day results **reused verbatim from E1 T2**), A1 `TABULAR_ONLY` (Direction reads `H_tab` only), A2 `TEMPORAL_ONLY` (Direction reads `H_time` only). Magnitude fusion, parameter set and TabM member axis are identical in all three modes.

Single-architecture axis only. No alpha grid, no concat/late fusion, no attention/MMoE/router, no threshold tuning, no Stage B, no PLE/FFT/hidden/depth/k sweep.

## 1. Overall (672 pooled slots, 28 days)

| model | Raw | Balanced | +Recall | -Recall | AUC | Brier | pred-pos frac | true prevalence | min-window Raw | window Raw std |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 FULL_CURRENT | 0.578869 | 0.527302 | 0.342975 | 0.711628 | 0.566327 | 0.235005 | 0.308036 | 0.360119 | 0.517857 | 0.053963 |
| A1 TABULAR_ONLY | 0.595238 | 0.498539 | 0.152893 | 0.844186 | 0.563223 | 0.228257 | 0.154762 | 0.360119 | 0.505952 | 0.090174 |
| A2 TEMPORAL_ONLY | 0.500000 | 0.474707 | 0.384298 | 0.565116 | 0.497751 | 0.259474 | 0.416667 | 0.360119 | 0.446429 | 0.051721 |

Safety counters (one-class days are a class-collapse signal; a high Raw must not be read without them):

| model | one-class days | +Recall=0 days | -Recall=0 days |
|---|---|---|---|
| A0 FULL_CURRENT | 3 | 7 | 1 |
| A1 TABULAR_ONLY | 9 | 16 | 0 |
| A2 TEMPORAL_ONLY | 0 | 4 | 1 |

## 2. Per-window Raw (pooled slots)

| model | W1 Feb | W2 Apr | W3 Jun | W4 Aug | min | std |
|---|---|---|---|---|---|---|
| A0 FULL_CURRENT | 0.648810 | 0.517857 | 0.535714 | 0.613095 | 0.517857 | 0.053963 |
| A1 TABULAR_ONLY | 0.702381 | 0.505952 | 0.505952 | 0.666667 | 0.505952 | 0.090174 |
| A2 TEMPORAL_ONLY | 0.535714 | 0.446429 | 0.452381 | 0.565476 | 0.446429 | 0.051721 |

## 3. Hour segments (Raw / Balanced / AUC / +Recall / -Recall)

| model | segment | Raw | Balanced | AUC | +Recall | -Recall |
|---|---|---|---|---|---|---|
| A0 FULL_CURRENT | H1 (1-8) | 0.491071 | 0.464240 | 0.493102 | 0.352941 | 0.575540 |
| A0 FULL_CURRENT | H2 (9-16) | 0.642857 | 0.537662 | 0.571243 | 0.257143 | 0.818182 |
| A0 FULL_CURRENT | H3 (17-24) | 0.602679 | 0.566113 | 0.615068 | 0.402299 | 0.729927 |
| A1 TABULAR_ONLY | H1 (1-8) | 0.558036 | 0.486204 | 0.500973 | 0.188235 | 0.784173 |
| A1 TABULAR_ONLY | H2 (9-16) | 0.687500 | 0.519481 | 0.667718 | 0.071429 | 0.967532 |
| A1 TABULAR_ONLY | H3 (17-24) | 0.540179 | 0.475166 | 0.505495 | 0.183908 | 0.766423 |
| A2 TEMPORAL_ONLY | H1 (1-8) | 0.406250 | 0.402751 | 0.417520 | 0.388235 | 0.417266 |
| A2 TEMPORAL_ONLY | H2 (9-16) | 0.549107 | 0.504545 | 0.492486 | 0.385714 | 0.623377 |
| A2 TEMPORAL_ONLY | H3 (17-24) | 0.544643 | 0.514473 | 0.574293 | 0.379310 | 0.649635 |

## 4. Paired deltas vs the A0 anchor

Day-cluster bootstrap, unit = target day, seed 20260924, 10000 resamples (same helper as E1-mini, so the interval definition is identical).

| comparison | mean dRaw | 95% CI | CI excludes 0 | W/T/L (Raw) | mean dBalanced | 95% CI (Balanced) |
|---|---|---|---|---|---|---|
| A1_minus_A0 | +0.016369 | [-0.0432, +0.0744] | False | 13/5/10 | +0.005190 | [-0.0474, +0.0562] |
| A2_minus_A0 | -0.078869 | [-0.1339, -0.0253] | True | 7/4/17 | -0.040715 | [-0.0903, +0.0073] |

Per-day Raw outcome counts by window:

| comparison | window | improved | tie | worsened |
|---|---|---|---|---|
| A1_minus_A0 | W1 | 4 | 2 | 1 |
| A1_minus_A0 | W2 | 3 | 0 | 4 |
| A1_minus_A0 | W3 | 2 | 2 | 3 |
| A1_minus_A0 | W4 | 4 | 1 | 2 |
| A2_minus_A0 | W1 | 1 | 0 | 6 |
| A2_minus_A0 | W2 | 2 | 1 | 4 |
| A2_minus_A0 | W3 | 3 | 0 | 4 |
| A2_minus_A0 | W4 | 1 | 3 | 3 |

## 5. Architecture, runtime, gradient ownership

| arch | mode | params | trainable | best epoch (mean) | epochs run (mean) | wall s/run | total wall s | epoch s | CUDA peak MB (mean/max) |
|---|---|---|---|---|---|---|---|---|---|
| A0 | full_current | 382724 | 382724 | 2.00 | 17.00 | 21.00 | 588.0 | 0.845 | 371.8/371.8 |
| A1 | tabular_only | 382724 | 382724 | 2.43 | 17.43 | 14.35 | 401.8 | 0.556 | 371.9/371.9 |
| A2 | temporal_only | 382724 | 382724 | 2.36 | 17.36 | 14.74 | 412.8 | 0.571 | 317.2/319.2 |

`alpha_dir` trajectory — an independent corroboration of the gradient-ownership probe, read off the training logs rather than from autograd. `alpha_dir` moves only in A0, the only mode where Direction still flows through the fusion; in A1/A2 it is bit-identical to its initialisation across all 28 runs, because it no longer reaches `L_dir` at all. `alpha_mag` is bit-identical to init in **all three** modes (including A0): under `objective_mode=dir_only` the backpropagated objective is exactly `losses["L_dir"]` (see `direction_experiments.objective_tensor`), and `a_mag` appears only in `h_mag`, so its gradient is exactly zero — a pre-existing property of V2.1 under this protocol, not an effect of the architecture axis.

| arch | alpha_dir init | alpha_dir final (mean) | min | max | alpha_mag final (mean) |
|---|---|---|---|---|---|
| A0 | 0.798153 | 0.792667 | 0.789419 | 0.796065 | 0.800000 |
| A1 | 0.800000 | 0.800000 | 0.800000 | 0.800000 | 0.800000 |
| A2 | 0.800000 | 0.800000 | 0.800000 | 0.800000 | 0.800000 |

Direction gradient ownership, measured on the **trained benchmark-day checkpoints** with the real target-day tensors (not a toy model):

| check | result |
|---|---|
| `A1_tabular_only__L_dir_blocked_from_temporal` | PASS |
| `A1_tabular_only__L_mag_still_reaches_temporal` | PASS |
| `A1_tabular_only__L_dir_reaches_tabular` | PASS |
| `A2_temporal_only__L_dir_blocked_from_tabular` | PASS |
| `A2_temporal_only__L_mag_still_reaches_tabular` | PASS |
| `A2_temporal_only__L_dir_reaches_temporal` | PASS |
| `A0_full_current__L_dir_reaches_both` | PASS |

## 6. Benchmark day 2026-02-13 — engineering only

These single-day numbers were used **only** for sanity/convergence/runtime/VRAM/parameter ownership/reproducibility. Per docs/10 they must not promote or eliminate an architecture.

| arch | mode | status | params | best/stop epoch | epochs | wall s | epoch s | CUDA peak MB | Raw | Balanced | AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 FULL_CURRENT | full_current | COMPLETE | 382724 | 3/18 | 18 | 16.0 | 0.644 | 371.8 | 0.583333 | 0.537815 | 0.563025 |
| A1 TABULAR_ONLY | tabular_only | COMPLETE | 382724 | 2/17 | 17 | 13.99 | 0.564 | 371.9 | 0.541667 | 0.466387 | 0.579832 |
| A2 TEMPORAL_ONLY | temporal_only | COMPLETE | 382724 | 3/18 | 18 | 15.67 | 0.562 | 317.0 | 0.458333 | 0.449580 | 0.478992 |

## 7. Provenance

- Runs: A0 = 28 reused E1-M2 records; A1 = 28 fresh; A2 = 28 fresh.
- Every run `COMPLETE`: True.
- `config_sha256` distinct values across all runs: 1 (8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c).
- Parameter count distinct across A1/A2 runs: 1.
- Frozen protocol on every fresh run: objective=['dir_only'], checkpoint=['direction_first'], gradient=['vanilla'].
- Figures: `figures/fig_daily_raw.png`, `figures/fig_window_overall.png`, `figures/fig_hour_segments.png`, `figures/fig_paired_bootstrap.png`.

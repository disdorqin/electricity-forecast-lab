# E2-B1 — Direction Fusion Coarse Map: summary

Seven arms over the frozen 28-day DEV panel. `alpha_time` pins the Direction fusion
coefficient `H_dir = alpha*time_dir(H_time) + (1-alpha)*tab_dir(H_tab)`; nothing else moves.
FL08 (canonical learnable alpha, init 0.8) is the safety anchor. Chance = 0.50.

**FUSION_REGION = `NO_CLEAR_REGION`**

## 5. Overall (28 days, micro over slots)

| arm | Raw | Balanced | +Recall | −Recall | AUC | Brier | pred. pos. frac | true prevalence | slots |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 0.5952 | 0.4985 | 0.1529 | 0.8442 | 0.5632 | 0.2283 | 0.1548 | 0.3601 | 672 |
| F20 | 0.5848 | 0.5211 | 0.2934 | 0.7488 | 0.5662 | 0.2322 | 0.2664 | 0.3601 | 672 |
| F40 | 0.5863 | 0.5169 | 0.2686 | 0.7651 | 0.5558 | 0.2334 | 0.2470 | 0.3601 | 672 |
| F60 | 0.5774 | 0.5162 | 0.2975 | 0.7349 | 0.5606 | 0.2335 | 0.2768 | 0.3601 | 672 |
| F80 | 0.5893 | 0.5372 | 0.3512 | 0.7233 | 0.5671 | 0.2351 | 0.3036 | 0.3601 | 672 |
| F100 | 0.5000 | 0.4747 | 0.3843 | 0.5651 | 0.4978 | 0.2595 | 0.4167 | 0.3601 | 672 |
| FL08 | 0.5789 | 0.5273 | 0.3430 | 0.7116 | 0.5663 | 0.2350 | 0.3080 | 0.3601 | 672 |

### Class-collapse check (Raw must not mask collapse)

| arm | Raw | Balanced | one-class days | +Recall=0 days | −Recall=0 days |
| --- | --- | --- | --- | --- | --- |
| F00 | 0.5952 | 0.4985 | 9 | 16 | 0 |
| F20 | 0.5848 | 0.5211 | 5 | 9 | 0 |
| F40 | 0.5863 | 0.5169 | 3 | 9 | 1 |
| F60 | 0.5774 | 0.5162 | 2 | 7 | 1 |
| F80 | 0.5893 | 0.5372 | 3 | 7 | 1 |
| F100 | 0.5000 | 0.4747 | 0 | 4 | 1 |
| FL08 | 0.5789 | 0.5273 | 3 | 7 | 1 |

## 6. Windows W1–W4 (Raw / Balanced)

| arm | W1 Raw | W1 Bal | W2 Raw | W2 Bal | W3 Raw | W3 Bal | W4 Raw | W4 Bal | min-window Raw | window Raw std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 0.7024 | 0.5238 | 0.5060 | 0.5094 | 0.5060 | 0.4343 | 0.6667 | 0.4956 | 0.5060 | 0.0902 |
| F20 | 0.6845 | 0.4960 | 0.5060 | 0.5082 | 0.5298 | 0.5157 | 0.6190 | 0.5115 | 0.5060 | 0.0714 |
| F40 | 0.6845 | 0.5119 | 0.5000 | 0.5022 | 0.5595 | 0.5056 | 0.6012 | 0.5029 | 0.5000 | 0.0671 |
| F60 | 0.6786 | 0.5317 | 0.5119 | 0.5137 | 0.5000 | 0.4481 | 0.6190 | 0.5302 | 0.5000 | 0.0746 |
| F80 | 0.6488 | 0.5040 | 0.5298 | 0.5303 | 0.5357 | 0.4870 | 0.6429 | 0.5525 | 0.5298 | 0.0566 |
| F100 | 0.5357 | 0.4603 | 0.4464 | 0.4474 | 0.4524 | 0.4481 | 0.5655 | 0.5277 | 0.4464 | 0.0517 |
| FL08 | 0.6488 | 0.5040 | 0.5179 | 0.5184 | 0.5357 | 0.4870 | 0.6131 | 0.5257 | 0.5179 | 0.0540 |

## 7. Hour segments H1–H3

| arm | H1 Raw | H1 Bal | H1 +R | H2 Raw | H2 Bal | H2 +R | H3 Raw | H3 Bal | H3 +R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 0.5580 | 0.4862 | 0.1882 | 0.6875 | 0.5195 | 0.0714 | 0.5402 | 0.4752 | 0.1839 |
| F20 | 0.5536 | 0.5260 | 0.4118 | 0.6607 | 0.5351 | 0.2000 | 0.5402 | 0.4878 | 0.2529 |
| F40 | 0.5312 | 0.5012 | 0.3765 | 0.6607 | 0.5312 | 0.1857 | 0.5670 | 0.5055 | 0.2299 |
| F60 | 0.5089 | 0.4832 | 0.3765 | 0.6429 | 0.5182 | 0.1857 | 0.5804 | 0.5311 | 0.3103 |
| F80 | 0.5000 | 0.4714 | 0.3529 | 0.6607 | 0.5545 | 0.2714 | 0.6071 | 0.5719 | 0.4138 |
| F100 | 0.4062 | 0.4028 | 0.3882 | 0.5491 | 0.5045 | 0.3857 | 0.5446 | 0.5145 | 0.3793 |
| FL08 | 0.4911 | 0.4642 | 0.3529 | 0.6429 | 0.5377 | 0.2571 | 0.6027 | 0.5661 | 0.4023 |

## 8. Paired vs the FL08 anchor (day-cluster bootstrap, unit = target day)

| comparison | mean ΔRaw | CI low | CI high | CI⊂zero | W/T/L Raw | mean ΔBalanced | CI low | CI high | CI⊂zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00_minus_FL08 | 0.016369 | -0.043155 | 0.074405 | True | 13/5/10 | 0.005190 | -0.047357 | 0.056233 | True |
| F20_minus_FL08 | 0.005952 | -0.034226 | 0.040179 | True | 14/7/7 | 0.004694 | -0.025611 | 0.034385 | True |
| F40_minus_FL08 | 0.007440 | -0.029762 | 0.040179 | True | 11/9/8 | -0.006523 | -0.037953 | 0.024172 | True |
| F60_minus_FL08 | -0.001488 | -0.034226 | 0.029762 | True | 7/13/8 | -0.002783 | -0.033446 | 0.029016 | True |
| F80_minus_FL08 | 0.010417 | 0.000000 | 0.026786 | True | 3/25/0 | 0.009497 | 0.000000 | 0.025275 | True |
| F100_minus_FL08 | -0.078869 | -0.133929 | -0.025298 | False | 7/4/17 | -0.040715 | -0.090266 | 0.007338 | True |

## 9. Safety admissibility vs FL08

SAFETY_ADMISSIBLE iff Balanced ≥ anchor−0.01 **and** +Recall ≥ anchor−0.05 **and** one-class days ≤ anchor+2 **and** +Recall=0 days ≤ anchor+2.

| arm | alpha | Balanced | ΔBalanced | +Recall | Δ+Recall | Δone-class | Δ+R=0 days | SAFETY_ADMISSIBLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 0.0 | 0.4985 | -0.028762 | 0.1529 | -0.190083 | +6 | +9 | **False** |
| F20 | 0.2 | 0.5211 | -0.006189 | 0.2934 | -0.049587 | +2 | +2 | **True** |
| F40 | 0.4 | 0.5169 | -0.010446 | 0.2686 | -0.074380 | +0 | +2 | **False** |
| F60 | 0.6 | 0.5162 | -0.011099 | 0.2975 | -0.045455 | -1 | +0 | **False** |
| F80 | 0.8 | 0.5372 | 0.009946 | 0.3512 | 0.008264 | +0 | +0 | **True** |
| F100 | 1.0 | 0.4747 | -0.052595 | 0.3843 | 0.041322 | -3 | -3 | **False** |
| FL08 | learnable | 0.5273 | 0.000000 | 0.3430 | 0.000000 | +0 | +0 | **True** |

Non-admissible arms (reported, not hidden): **F00, F40, F60, F100**.

## 10. Effective fusion diagnostics (alpha is not a contribution percentage)

Deterministic diagnostic batch = the target day's own inputs. Observation only.

| arm | alpha used (mean) | range | ‖tab‖ | ‖time‖ | effective time-norm fraction (mean ± sd over days) | cosine(tab, time) |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | 0.000000 | 0.0000–0.0000 | 37.5166 | 0.0000 | 0.0000 ± 0.0000 | nan |
| F20 | 0.200000 | 0.2000–0.2000 | 30.6223 | 17.2482 | 0.3744 ± 0.1406 | 0.0626 |
| F40 | 0.400000 | 0.4000–0.4000 | 27.4466 | 20.4501 | 0.4405 ± 0.1413 | 0.0605 |
| F60 | 0.600000 | 0.6000–0.6000 | 24.3101 | 23.2438 | 0.4988 ± 0.1341 | 0.0570 |
| F80 | 0.800000 | 0.8000–0.8000 | 20.4098 | 25.7749 | 0.5688 ± 0.1385 | 0.0429 |
| F100 | 1.000000 | 1.0000–1.0000 | 0.0000 | 29.3496 | 1.0000 ± 0.0000 | nan |
| FL08 | 0.798383 | 0.7975–0.7996 | 20.2219 | 25.9155 | 0.5735 ± 0.1464 | 0.0431 |

The two norm columns and the fraction column are each means over the 28 runs, and every
run is trained separately, so the fraction column is the mean of per-run fractions and
is **not** the ratio of the two norm means shown beside it. The per-run values are in
`fusion_diagnostics.csv`.

Read this against the alpha column, not instead of it. Two facts matter: the
effective time-norm fraction at a given alpha is far from alpha (the branches do not
have comparable norms), and it varies **across target days** for a fixed coefficient,
because the encoders are retrained per day. A response curve indexed by alpha is
therefore a curve over *training runs*, not over a smoothly varying mixture weight.

## 11. Runtime

| arm | runs | wall mean (s) | epoch mean (s) | best epoch (mean) | stop epoch (mean) | params | config_sha256 (distinct) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F00 | 28 | 14.35 | 0.5564 | 2.43 | 17.43 | 382724 | 1 |
| F20 | 28 | 14.46 | 0.5728 | 2.43 | 17.43 | 382724 | 1 |
| F40 | 28 | 12.23 | 0.4947 | 2.21 | 17.21 | 382724 | 1 |
| F60 | 28 | 11.98 | 0.4857 | 2.11 | 17.11 | 382724 | 1 |
| F80 | 28 | 12.06 | 0.4913 | 2.11 | 17.11 | 382724 | 1 |
| F100 | 28 | 14.74 | 0.5705 | 2.36 | 17.36 | 382724 | 1 |
| FL08 | 28 | 21.00 | 0.8446 | 2 | 17 | 382724 | 1 |

## 3. Fixed-alpha routing audit (Gate A, benchmark day 2026-02-13)

**14/14 PASS.** Endpoint reproduction is compared as artifacts.

| check | result |
| --- | --- |
| `F00_fixed_0.0_reproduces_tabular_only_within_1e-9` | PASS |
| `F00_fixed_0.0_alpha_receives_no_update` | PASS |
| `F100_fixed_1.0_reproduces_temporal_only_within_1e-9` | PASS |
| `F100_fixed_1.0_alpha_receives_no_update` | PASS |
| `F20_alpha_receives_no_update` | PASS |
| `F40_alpha_receives_no_update` | PASS |
| `F60_alpha_receives_no_update` | PASS |
| `F80_alpha_receives_no_update` | PASS |
| `F00_alpha0_direction_blocked_from_temporal` | PASS |
| `F00_alpha0_direction_reaches_tabular` | PASS |
| `F00_alpha_receives_no_direction_gradient` | PASS |
| `F100_alpha1_direction_blocked_from_tabular` | PASS |
| `F100_alpha1_direction_reaches_temporal` | PASS |
| `F100_alpha_receives_no_direction_gradient` | PASS |

| arm | alpha | reproduces | parquet SHA match | max abs Δpred | max abs Δmetrics | distinct alpha values |
| --- | --- | --- | --- | --- | --- | --- |
| F00 | 0.0 | E2-A tabular_only | True | 0.000 | 0.000 | 1 |
| F100 | 1.0 | E2-A temporal_only | True | 0.000 | 0.000 | 1 |
| F20 | 0.2 | fresh fixed-alpha arm | nan | nan | nan | 1 |
| F40 | 0.4 | fresh fixed-alpha arm | nan | nan | nan | 1 |
| F60 | 0.6 | fresh fixed-alpha arm | nan | nan | nan | 1 |
| F80 | 0.8 | fresh fixed-alpha arm | nan | nan | nan | 1 |

The immobility check reads `alpha_trajectories[].alpha_dir`, which logs the model's
**learnable** `a_dir` (a sigmoid applied to a logit). Under a fixed coefficient that
parameter is inert, so the proof is that it is bit-constant across every epoch. Its
constant value is the *initialisation* (0.8), not the arm's coefficient; the coefficient
is recorded separately in the run manifest and shown above as `alpha`.

## 4. Benchmark day 2026-02-13 (engineering only)

| arm | alpha | Raw |
| --- | --- | --- |

This is one day. It is recorded for engineering provenance only and was **not** used
to rank, filter or eliminate any arm.


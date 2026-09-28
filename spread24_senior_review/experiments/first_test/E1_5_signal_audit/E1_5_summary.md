# E1.5-A Direction Signal Audit — Results

Scope: E1's exact 28 target days / 672 slots (W1 `2026-02-12..02-18`, W2 `2026-04-12..04-18`,
W3 `2026-06-12..06-18`, W4 `2026-08-07..08-13`). 8 models × 672 slots = 5376 scored slots,
all against one shared truth panel. TafM T0/T1/T2 read read-only from the completed E1 runs;
no retraining, no re-selection, no threshold change. `src/` unmodified.

Truth cross-check: `max |E1 predictions − frozen sequence asset| = 2.93e-05` (float32 parquet
round-trip, inside the 1e-3 tolerance). Micro-Raw vs macro-day-Raw identity verified at
`max|delta| = 1.110e-16` (algebraic identity: every day has exactly 24 slots).

Overall true positive prevalence = **0.360119** (242 positive / 430 non-positive slots).

---

## 1. Overall metrics — all 8 models

| model | Raw | Balanced | +Recall | −Recall | AUC | Brier | ppf | one-class days | +R=0 days | −R=0 days |
|---|---|---|---|---|---|---|---|---|---|---|
| **B0_AlwaysNonPositive** | **0.639881** | 0.500000 | 0.000000 | 1.000000 | N/A | N/A | 0.000000 | 28 | 28 | 0 |
| B1_HourMajority180 | 0.586310 | 0.497886 | 0.181818 | 0.813953 | 0.479046 † | N/A | 0.184524 | 7 | 10 | 0 |
| **T2** (M2 dir_only+direction_first) | 0.578869 | **0.527302** | 0.342975 | 0.711628 | **0.566327** | **0.235005** | 0.308036 | 3 | 7 | 1 |
| B3_XGB_DIR_EXPANDING | 0.578869 | 0.497492 | 0.206612 | 0.788372 | 0.541659 | 0.235368 | 0.209821 | 9 | 15 | 0 |
| T1 (M1 joint_v21+direction_first) | 0.566964 | 0.517999 | 0.342975 | 0.693023 | 0.553094 | 0.239293 | 0.319940 | 1 | 8 | 1 |
| B4_StrictLightGBM_DIR_180 | 0.544643 | 0.495137 | 0.318182 | 0.672093 | 0.523381 † | N/A | 0.324405 | 4 | 8 | 0 |
| T0 (M0 joint_v21+v21_guardrail) | 0.534226 | 0.505968 | 0.404959 | 0.606977 | 0.517624 | 0.265968 | 0.397321 | 2 | 4 | 2 |
| B2_XGB_DIR_180 | 0.529762 | 0.468153 | 0.247934 | 0.688372 | 0.509009 | 0.252410 | 0.288690 | 11 | 16 | 1 |

† `RANK_ONLY_NOT_A_PROBABILITY_METRIC` — a signed regression output ranked, not a probability.
Brier is `N/A` by contract; no calibrated Brier was fabricated.

### The three highest Raw scores are all MAJORITY_DRIVEN

| model | Raw | Balanced | verdict |
|---|---|---|---|
| B0_AlwaysNonPositive | 0.639881 | **0.500000** | `MAJORITY_DRIVEN` — +Recall is exactly 0.000 on all 28 days |
| B1_HourMajority180 | 0.586310 | **0.497886** | `MAJORITY_DRIVEN` — Balanced *below* chance |
| B3_XGB_DIR_EXPANDING | 0.578869 | **0.497492** | `MAJORITY_DRIVEN` — Balanced *below* chance |

The trivial constant baseline `direction_hat ≡ 0` — which is not a model in any sense —
outscores **every** real model in this audit, including all three TafM variants. Any Raw
figure at or below 0.6399 must therefore be read against that floor, not against 0.5.

Only T1 (0.5180) and T2 (0.5273) sit above 0.55 Raw *and* above chance on Balanced, and both
only marginally. **No model in this audit shows genuine two-sided direction skill.**

---

## 2. Per-window metrics (W1–W4)

Majority-baseline Raw per window: **W1 0.750000, W2 0.505952, W3 0.642857, W4 0.672619**
(truth positive prevalence W1 0.250000, W2 0.505952, W3 0.357143, W4 0.327381).

### Raw Direction Accuracy

| model | W1 | W2 | W3 | W4 | min | std | spread |
|---|---|---|---|---|---|---|---|
| B0_AlwaysNonPositive | 0.7500 | 0.4940 | 0.6429 | 0.6726 | 0.4940 | 0.0928 | 0.2560 |
| B1_HourMajority180 | 0.5714 | 0.4702 | 0.6310 | 0.6726 | 0.4702 | 0.0761 | 0.2024 |
| B2_XGB_DIR_180 | 0.5774 | 0.4167 | 0.5060 | 0.6190 | 0.4167 | 0.0768 | 0.2024 |
| B3_XGB_DIR_EXPANDING | 0.7024 | 0.5179 | 0.4762 | 0.6190 | 0.4762 | 0.0882 | 0.2262 |
| B4_StrictLightGBM_DIR_180 | 0.6012 | 0.3929 | 0.5476 | 0.6369 | 0.3929 | 0.0932 | 0.2440 |
| T0 | 0.6548 | 0.4881 | 0.4821 | 0.5119 | 0.4821 | 0.0705 | 0.1726 |
| T1 | 0.6369 | 0.5179 | 0.5417 | 0.5714 | 0.5179 | 0.0446 | 0.1190 |
| **T2** | 0.6488 | 0.5179 | 0.5357 | 0.6131 | **0.5179** | **0.0540** | **0.1310** |

T1 is the most *stable* model (spread 0.1190) and T2 second (0.1310), but T2's floor is 0.5179 —
barely above a coin flip — and no window reaches 0.65 for any model except the trivial ones
(B0 W1 0.7500, B3 W1 0.7024).

### Per-window detail for the two candidate models

| model | window | Raw | Balanced | +Recall | −Recall | ppf | prevalence | majority Raw | **prior gap** |
|---|---|---|---|---|---|---|---|---|---|
| T2 | W1 | 0.648810 | 0.503968 | 0.214286 | 0.793651 | 0.208333 | 0.250000 | 0.750000 | **−0.101190** |
| T2 | W2 | 0.517857 | 0.518427 | 0.470588 | 0.566265 | 0.452381 | 0.505952 | 0.505952 | **+0.011905** |
| T2 | W3 | 0.535714 | 0.487037 | 0.316667 | 0.657407 | 0.333333 | 0.357143 | 0.642857 | **−0.107143** |
| T2 | W4 | 0.613095 | 0.525744 | 0.272727 | 0.778761 | 0.238095 | 0.327381 | 0.672619 | **−0.059524** |
| T1 | W1 | 0.636905 | 0.472222 | 0.142857 | 0.801587 | 0.184524 | 0.250000 | 0.750000 | **−0.113095** |
| T1 | W2 | 0.517857 | 0.519419 | 0.388235 | 0.650602 | 0.369048 | 0.505952 | 0.505952 | **+0.011905** |
| T1 | W3 | 0.541667 | 0.513889 | 0.416667 | 0.611111 | 0.398810 | 0.357143 | 0.642857 | **−0.101190** |
| T1 | W4 | 0.571429 | 0.513435 | 0.345455 | 0.681416 | 0.327381 | 0.327381 | 0.672619 | **−0.101190** |

Note T2's W3 Balanced Accuracy is **0.487037 — below chance**, while its W1 and W4 Balanced sit
at 0.5040 and 0.5257. The overall Balanced of 0.5273 is not carried uniformly across windows.

---

## 3. Hour-segment metrics (H1 = 1–8, H2 = 9–16, H3 = 17–24)

| model | segment | Raw | Balanced | +Recall | −Recall | ppf | prevalence | AUC |
|---|---|---|---|---|---|---|---|---|
| B0_AlwaysNonPositive | H1/H2/H3 | .6205/.6875/.6116 | 0.500 | 0.000 | 1.000 | 0.000 | .3795/.3125/.3884 | N/A |
| B1_HourMajority180 | H1 | 0.5848 | 0.5284 | 0.2941 | 0.7626 | 0.2589 | 0.3795 | 0.5380 |
| B1_HourMajority180 | H2 | 0.5402 | 0.3968 | 0.0143 | 0.7792 | 0.1563 | 0.3125 | 0.2833 |
| B1_HourMajority180 | H3 | 0.6339 | 0.5560 | 0.2069 | 0.9051 | 0.1384 | 0.3884 | 0.6116 |
| B2_XGB_DIR_180 | H1/H2/H3 | .5045/.6205/.4643 | .4636/.4786/.4383 | .2941/.1000/.3218 | .6331/.8571/.5547 | — | — | .4823/.5696/.4612 |
| B3_XGB_DIR_EXPANDING | H1 | 0.5938 | 0.5401 | 0.3176 | 0.7626 | 0.2679 | 0.3795 | 0.5002 |
| B3_XGB_DIR_EXPANDING | H2 | 0.6384 | 0.4877 | 0.0857 | 0.8896 | 0.1027 | 0.3125 | 0.6276 |
| B3_XGB_DIR_EXPANDING | H3 | 0.5045 | 0.4481 | 0.1954 | 0.7007 | 0.2589 | 0.3884 | 0.4703 |
| B4_StrictLightGBM_DIR_180 | H1/H2/H3 | .6205/.5089/.5045 | .5457/.5026/.4607 | .2353/.4857/.2644 | .8561/.5195/.6569 | — | — | .6141/.5419/.4359 |
| T0 | H1 | 0.4375 | 0.4211 | 0.3529 | 0.4892 | 0.4509 | 0.3795 | 0.4185 |
| **T0** | **H2** | **0.6518** | **0.6182** | 0.5286 | 0.7078 | 0.3661 | 0.3125 | **0.6558** |
| T0 | H3 | 0.5134 | 0.4847 | 0.3563 | 0.6131 | 0.3750 | 0.3884 | 0.4926 |
| T1 | H1 | 0.4911 | 0.4734 | 0.4000 | 0.5468 | 0.4330 | 0.3795 | 0.4844 |
| T1 | H2 | 0.6562 | 0.5630 | 0.3143 | 0.8117 | 0.2277 | 0.3125 | 0.5968 |
| T1 | H3 | 0.5536 | 0.5092 | 0.3103 | 0.7080 | 0.2991 | 0.3884 | 0.5571 |
| T2 | H1 | 0.4911 | 0.4642 | 0.3529 | 0.5755 | 0.3973 | 0.3795 | 0.4931 |
| T2 | H2 | 0.6429 | 0.5377 | 0.2571 | 0.8182 | 0.2054 | 0.3125 | 0.5712 |
| **T2** | **H3** | 0.6027 | 0.5661 | 0.4023 | 0.7299 | 0.3214 | 0.3884 | 0.6151 |

**The single best cell in the entire audit is `T0 / H2`: Raw 0.6518, Balanced 0.6182,
AUC 0.6558.** It is also the least stable kind of result — it belongs to the *worst* overall
TafM variant, appears in exactly one segment, and T0's H1 is 0.4375 (worse than chance). It is
recorded as an observation, not as a promotable finding.

Note the recurring shape: H2 (09–16) is where several models look best — but H2 is also the
segment with the *lowest positive prevalence* (0.3125), so high H2 Raw is partly the majority
class again. B3's H2 Raw 0.6384 comes with Balanced 0.4877, i.e. entirely majority-driven.
Only T0's H2 survives that check (Balanced 0.6182), and it does not replicate in H1 or H3.

---

## 4. B2 (recent 180d) vs B3 (expanding history) — the recency question

| | B2_XGB_DIR_180 | B3_XGB_DIR_EXPANDING | delta |
|---|---|---|---|
| Overall Raw | 0.529762 | 0.578869 | **−0.049107** |
| Overall Balanced | 0.468153 | 0.497492 | −0.029339 |
| Overall AUC | 0.509009 | 0.541659 | −0.032650 |
| W1 Raw | 0.577381 | 0.702381 | −0.125000 |
| W2 Raw | 0.416667 | 0.517857 | −0.101190 |
| W3 Raw | 0.505952 | 0.476190 | +0.029762 |
| W4 Raw | 0.619048 | 0.619048 | 0.000000 |
| W1 majority baseline | 0.750000 | 0.750000 | — |

**Restricting training to the recent legal 180 days makes the XGBoost direction baseline
*worse*, by 4.91 pp overall, and it loses in 3 of 4 windows (winning W3 by 2.98 pp, tying W4).**
B2 is in fact the single worst non-trivial model in the audit, and it is worse than B1's plain
per-hour historical majority in every window.

The plan's Case B condition was `XGB-180 >> XGB-expanding`. The measured relationship is the
**opposite sign**. Recent-regime information is not merely absent here — it is actively
harmful, which is itself a finding: whatever weak signal the expanding history carries is
carried by the longer record, not by a recent regime.

---

## 5. Paired comparison vs T2 (day-cluster bootstrap, unit = target day, 10 000 resamples, seed 20260924)

Delta = model − T2. 28 independent units (days), not 672.

| model | mean ΔRaw | median ΔRaw | 95% CI | CI∋0 | W/T/L days | mean ΔBalanced |
|---|---|---|---|---|---|---|
| B0_AlwaysNonPositive | +0.061012 | +0.041667 | [−0.020833, +0.138393] | yes | 15/6/7 | −0.006703 |
| B1_HourMajority180 | +0.007440 | 0.000000 | [−0.075893, +0.087798] | yes | 13/2/13 | +0.006193 |
| B2_XGB_DIR_180 | −0.049107 | −0.041667 | [−0.139881, +0.041667] | yes | 10/2/16 | −0.015794 |
| B3_XGB_DIR_EXPANDING | 0.000000 | +0.020833 | [−0.069940, +0.069940] | yes | 14/3/11 | −0.024023 |
| B4_StrictLightGBM_DIR_180 | −0.034226 | −0.020833 | [−0.105655, +0.035714] | yes | 10/4/14 | −0.016025 |
| T0 | −0.044643 | 0.000000 | [−0.108631, +0.013393] | yes | 11/4/13 | +0.001598 |
| T1 | −0.011905 | 0.000000 | [−0.052083, +0.026786] | yes | 13/2/13 | +0.005055 |

**Every 95% CI includes zero.** At the day-cluster level, no model in this audit — including the
trivial constant baseline — is statistically distinguishable from T2 on Raw Direction Accuracy.
B3's overall Raw ties T2 exactly (both 389/672 = 0.578869); that is an exact tie, not a win.

This is the honest uncertainty statement, not an auto-promotion threshold. It says the audit
cannot rank the field; it does not say the field is good.

---

## 6. TafM probability separation

Mean / median `p_positive` conditioned on the true class (TafM models emit a genuine probability):

| model | mean p \| y>0 | mean p \| y<=0 | **mean gap** | median p \| y>0 | median p \| y<=0 | **median gap** |
|---|---|---|---|---|---|---|
| T0 | 0.461310 | 0.453608 | **+0.007703** | 0.450916 | 0.441639 | +0.009277 |
| T1 | 0.448330 | 0.418299 | **+0.030031** | 0.455308 | 0.435290 | +0.020018 |
| T2 | 0.441151 | 0.412299 | **+0.028852** | 0.450944 | 0.413495 | +0.037448 |

Two things stand out:

1. **The separation gap is 0.8–3.0 pp.** A model with usable direction signal separates the two
   conditional distributions by a wide margin; T2 separates them by under 3 percentage points.
   T0 — the variant with the guardrail loss — separates them by 0.77 pp, essentially nothing.
2. **`mean p | y>0` is below 0.5 for every TafM variant** (T2: 0.4412; T1: 0.4483; T0: 0.4613).
   On average the models assign *less* than even odds to slots that are truly positive. This is
   why +Recall is only 0.343 at the deployed 0.5 threshold — it is a systematic centring bias,
   and it is visible in the oracle diagnostic below.

Per-window and per-hour-segment separation figures are in `probability_separation.csv`.

---

## 7. Oracle threshold diagnostic

> **`LEAKY_DIAGNOSTIC_ONLY` · `NOT_A_MODEL_RESULT` · `NOT_FOR_PROMOTION`**
>
> Every row in `oracle_threshold_DIAGNOSTIC_ONLY.csv` carries these three markers. The oracle
> threshold is chosen *on the target days themselves* using their own labels. It is therefore
> leaky by construction and is **not a model result**. It must never be written back into any
> formal model, and **none of these numbers may be described as a 65% result** — the highest
> figure below is 0.6458 and it is degenerate (see ppf).

Grid: `threshold ∈ [0.000, 1.000]` step 0.001; among tied optima the threshold closest to 0.5
is reported. Verification check: the `threshold_0.5` row reproduces the deployed Raw exactly.

| model | row | threshold | Raw | Balanced | +Recall | −Recall | ppf | tied optima |
|---|---|---|---|---|---|---|---|---|
| T0 | threshold_0.5 | 0.500 | 0.534226 | 0.505968 | 0.404959 | 0.606977 | 0.397321 | 1 |
| T0 | oracle_raw | 0.896 | 0.641369 | 0.502066 | 0.004132 | 1.000000 | **0.001488** | 50 |
| T0 | oracle_balanced | 0.366 | 0.489583 | 0.539737 | 0.719008 | 0.360465 | 0.668155 | 1 |
| T1 | threshold_0.5 | 0.500 | 0.566964 | 0.517999 | 0.342975 | 0.693023 | 0.319940 | 1 |
| T1 | oracle_raw | 0.773 | 0.644345 | 0.506198 | 0.012397 | 1.000000 | **0.004464** | 33 |
| T1 | oracle_balanced | 0.364 | 0.502976 | **0.569172** | 0.805785 | 0.332558 | 0.717262 | 1 |
| T2 | threshold_0.5 | 0.500 | 0.578869 | 0.527302 | 0.342975 | 0.711628 | 0.308036 | 1 |
| T2 | oracle_raw | 0.627 | **0.645833** | 0.511878 | 0.033058 | 0.990698 | **0.017857** | 1 |
| T2 | oracle_balanced | 0.383 | 0.523810 | **0.564674** | 0.710744 | 0.418605 | 0.627976 | 1 |

### What the leaky ceiling actually says

- **The oracle-Raw "wins" are degenerate.** T2's best possible Raw is 0.645833, achieved at
  threshold 0.627 with predicted-positive fraction **0.0179** — it predicts positive on under
  2% of slots. Its Balanced Accuracy there is 0.5119 and +Recall is 0.0331. This is the trivial
  all-non-positive solution wearing a threshold, entering the same 0.64 region as B0. Its
  apparent advantage over B0 (0.6458 vs 0.6399) is 0.6 pp and comes entirely from the handful
  of slots it does not swallow.
- **There is a real but small threshold mis-centring.** T2 at threshold 0.383 (below 0.5, which
  matches the `mean p | y>0 = 0.4412` finding above) lifts Balanced from 0.5273 → **0.5647** and
  +Recall from 0.3430 → 0.7107. So ~3.7 pp of Balanced Accuracy is being left on the table by
  the deployed 0.5 cut.
- **That is the ceiling.** Even with a threshold chosen using the target labels themselves,
  **no TafM variant exceeds Balanced Accuracy 0.5692 (T1)**. The decision layer costs roughly
  3–4 pp of Balanced Accuracy — real, worth knowing, and far too small to explain the gap to a
  useful classifier.

The diagnostic therefore **rules out a decision-layer explanation**. If a strong ranker were
hidden behind a bad threshold, the oracle curve would expose it. It does not: the leaky ceiling
sits where the model already is.

---

## 8. Prior gap — model Raw minus its own window's majority-baseline Raw

| model | OVERALL | W1 | W2 | W3 | W4 |
|---|---|---|---|---|---|
| B0_AlwaysNonPositive | 0.000000 | 0.000000 | −0.011905 | 0.000000 | 0.000000 |
| B1_HourMajority180 | −0.053571 | −0.178571 | −0.035714 | −0.011905 | 0.000000 |
| B2_XGB_DIR_180 | **−0.110119** | −0.172619 | −0.089286 | −0.136905 | −0.053571 |
| B3_XGB_DIR_EXPANDING | −0.061012 | −0.097619 | +0.011905 | −0.166667 | −0.053571 |
| B4_StrictLightGBM_DIR_180 | −0.095238 | −0.148810 | −0.113095 | −0.095238 | −0.035714 |
| T0 | **−0.105655** | −0.095238 | −0.017857 | −0.160714 | −0.160714 |
| T1 | −0.072917 | −0.113095 | +0.011905 | −0.101190 | −0.101190 |
| **T2** | −0.061012 | −0.101190 | +0.011905 | −0.107143 | −0.059524 |

(B0 shows 0.000 where the window's majority class *is* non-positive — there B0 **is** the
majority baseline. W2 is the one window whose majority class is positive, hence B0's −0.0119.)

**Every non-trivial model has a negative prior gap overall, and in every window except W2.**
The single positive cell in the entire matrix is **+0.011905 (1.19 pp)** in W2, for B3, T1 and
T2 alike — and W2 is precisely the window where the majority baseline is 0.5060, i.e. where the
"prior" being beaten is a coin flip. Beating a coin flip by 1.19 pp is not evidence of skill.

Translated: **every model in this audit is worse than simply predicting the majority direction
of its own evaluation window.** The models are not extracting direction information that the
class prior does not already contain.

---

## 9. Case judgment matrix

| case | condition | measured | verdict |
|---|---|---|---|
| A | tree baseline ≥65 and clearly above TafM | best tree baseline B3 = 0.5789, **not ≥65**; **exactly ties** T2, does not exceed it | **FAILS** |
| B | `XGB-180 >> XGB-expanding` | B2 0.5298 **<** B3 0.5789 by 4.91 pp — **opposite sign** | **FAILS (inverted)** |
| C | `XGB-expanding ≥ XGB-180` **and** tree baseline > TafM | first conjunct holds (0.5789 > 0.5298); second **fails** (B3 ties T2; B2 0.5298 and B4 0.5446 are both below T2 0.5789) | **FAILS** |
| D | TafM AUC/Brier clearly better than baselines but Raw conversion poor | T2 has the best AUC (0.5663) and Brier (0.2350) of all 8 models — but only 2.5 pp of AUC over B3 (0.5417), and the **leaky oracle ceiling is Balanced 0.5647**, so no strong ranker is hidden behind the threshold | **FAILS (premise not supported)** |
| E | all non-trivial models ~55–60% with AUC near 0.5 | non-trivial Raw spans **0.5298–0.5863**; AUC spans **0.4790–0.5663**, all near 0.5; the trivial constant baseline at 0.6399 beats every real model | **FITS** |

---

## 10. The 65% reading

| model | overall Raw | W1 | W2 | W3 | W4 | min-window | window std | Balanced | +Recall | −Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| T2 | 0.578869 | 0.6488 | 0.5179 | 0.5357 | 0.6131 | 0.5179 | 0.0540 | 0.5273 | 0.3430 | 0.7116 |
| T1 | 0.566964 | 0.6369 | 0.5179 | 0.5417 | 0.5714 | 0.5179 | 0.0446 | 0.5180 | 0.3430 | 0.6930 |
| B1 (best non-trivial) | 0.586310 | 0.5714 | 0.4702 | 0.6310 | 0.6726 | 0.4702 | 0.0761 | 0.4979 | 0.1818 | 0.8140 |

- **No legal 65%+ candidate appeared.** The highest non-trivial overall Raw is B1 at 0.586310.
- Cross-window stability is therefore moot for the 65% claim, but the spread is informative:
  T2's best window (0.6488) and worst (0.5179) differ by 13.10 pp; T1's by 11.90 pp. Neither is
  cross-window stable. Even the *trivial* baseline swings 25.60 pp across the four windows.
- The models whose Raw does approach 0.65 — B0 (0.6399) and B3 in W1 (0.7024) — are
  `MAJORITY_DRIVEN` (Balanced 0.500 and 0.4975 respectively). Per §8, **they may not be claimed
  as genuine model success, and the 0.6399 / 0.7024 figures may not be reported as 65% progress.**

The task of this round was not to reach 65%; it was to locate the source of usable Direction
signal. No such source was found in this evaluation scope.

---

## 11. Figures

`figures/` (6):

1. `fig_overall_raw_balanced.png` — overall Raw vs Balanced, all 8 models
2. `fig_window_raw_vs_majority.png` — W1–W4 Raw with the per-window majority baseline overlaid
3. `fig_hour_segment_raw_balanced.png` — H1/H2/H3 Raw and Balanced, all models
4. `fig_tafm_probability_separation.png` — TafM conditional `p_positive` distributions
5. `fig_daily_raw_curve.png` — day-level Raw trajectory across the 28 target days
6. `fig_daily_raw_heatmap.png` — model × target-day Raw heatmap

## 12. Provenance

| item | value |
|---|---|
| frozen source `data/frozen_repro/slot_table.parquet` | `a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea` |
| selector manifest | `ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f` (status `FROZEN`) |
| `src/config_tabm_v21.yaml` | `8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c` |
| sequence manifest | `9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82` |
| seed | 20260924 (B2/B3/XGB); bootstrap seed 20260924, 10 000 day-cluster resamples |
| truth cross-check | max abs delta 2.93e-05 (tolerance 1e-3) |
| micro/macro Raw identity | max abs delta 1.110e-16 |

Fit times: B0 0.01 s, B1 0.85 s, B2 34.16 s, B3 124.55 s, B4 121.50 s. TafM: 0 s (read from E1).

`src/` was **not modified**. `requirements.txt` was the only file changed outside this
experiment directory (declaring `shap==0.49.1`). `E1_mini` and every other existing result tree
were read only.

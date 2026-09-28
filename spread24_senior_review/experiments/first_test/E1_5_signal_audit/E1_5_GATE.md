# E1.5-A Direction Signal Audit — GATE

```
SIGNAL_SOURCE = FEATURE_SIGNAL_LIMITED
```

Scope of this verdict: E1's frozen 28 target days / 672 slots, the frozen selector, the frozen
259-feature sequence representation, and the four pre-registered baselines. It is a statement
about **the direction signal recoverable from the current feature/representation family at this
evaluation scope**. It is not a claim about electricity price spreads in general, and not a
claim about any architecture in the abstract.

---

## 1. Why `FEATURE_SIGNAL_LIMITED`

Case E of the pre-registered matrix (§10 of the plan) is the only case whose conditions hold:

1. **All non-trivial models cluster in 0.5298–0.5863 Raw, with AUCs 0.4790–0.5663.** Eight models
   spanning three architecture families (TafM ×3, XGBoost ×2, LightGBM ×1, historical majority ×1)
   and two training regimes land inside a 5.7 pp band, and every AUC is within 7 pp of 0.5.
2. **The trivial constant baseline wins.** `B0_AlwaysNonPositive` — `direction_hat ≡ 0`, no
   features, no training — scores **Raw 0.639881**, higher than every real model in the audit,
   with Balanced exactly 0.500 and +Recall exactly 0.000 on all 28 days. This is the cleanest
   possible statement that the feature family adds nothing the class prior does not already carry.
3. **The prior gap is negative everywhere it matters.** Every non-trivial model, in every window
   except W2, is *worse* than predicting its own window's majority direction. The only positive
   cells are +1.19 pp in W2 for B3/T1/T2 — in the one window whose majority baseline is 0.5060,
   i.e. beating a coin flip by a hair.
4. **The leaky oracle ceiling confirms it.** Even choosing the threshold *on the target labels*,
   no TafM variant exceeds **Balanced Accuracy 0.5692**. The decision layer is worth ~3.7 pp of
   Balanced Accuracy, not a rescue.
5. **Probability separation is 0.8–3.0 pp**, and `mean p | y>0` is below 0.5 for all three TafM
   variants — the conditional distributions barely move.

### Cases that were checked and rejected

| case | why rejected |
|---|---|
| A `MODEL_UTILIZATION_GAP` | No tree baseline reaches 65%. The best (B3, 0.5789) **exactly ties** T2 rather than exceeding it, and B3's Balanced Accuracy is 0.4975 — below chance, i.e. majority-driven. A tie between a gradient-boosted tree and a deep sequence model at near-chance AUC is not a utilization gap; it is a floor. |
| B `RECENCY_SIGNAL` | **Inverted.** `XGB-180` (0.5298) is *worse* than `XGB-expanding` (0.5789) by 4.91 pp, losing in 3 of 4 windows. Recent-regime information is not merely unhelpful here; it is harmful. B2 is the worst non-trivial model in the audit. |
| C `ARCHITECTURE_OR_OPTIMIZATION_GAP` | First conjunct holds (0.5789 > 0.5298) but the second fails: no tree baseline exceeds T2; B3 ties, B2 (0.5298) and B4 (0.5446) are below. An architecture/optimization gap requires a simpler model to beat the complex one. None does. |
| D `DECISION_LAYER_GAP` / `TAFM_RANKING_SIGNAL` | T2 does hold the best AUC (0.5663) and Brier (0.2350) of all eight models — but by only 2.5 pp of AUC over B3, which is not "clearly better", and there is no hidden ranker behind the threshold: the leaky oracle tops out at Balanced 0.5647. A decision-layer gap would have been exposed by the oracle curve. It was not. |

---

## 2. What this verdict does **not** say

- It does **not** say the TafM architecture is inadequate. The architecture was never given a
  signal to exploit; a null result on a near-noise feature set is uninformative about the model.
- It does **not** close the 65% line of work. It closes *this* route to it: more capacity,
  longer training, or a better threshold over the current representation.
- It is **not** an oracle result and **not** a promotion. The 0.6458 / 0.5647 oracle figures are
  `LEAKY_DIAGNOSTIC_ONLY`, `NOT_A_MODEL_RESULT`, `NOT_FOR_PROMOTION`, and are not 65% results.
- It does **not** re-open any E1 conclusion. `CHECKPOINT_DIRECTION_FIRST=NOT_SUPPORTED` and
  `NEGATIVE_TRANSFER=NOT_SUPPORTED` stand unchanged.

---

## 3. Next-step suggestion — NOT AUTHORIZED, NOT EXECUTED

Advisory only. Nothing below has been run, scheduled, or prepared, and none of it may be started
without a new explicit instruction.

The evidence points at the input representation rather than the model or the decision layer.
The natural next probe is therefore a **feature-signal audit** on the frozen representation,
before any further training budget is spent on architecture or window variants:

1. **Univariate signal screen.** For each of the 259 future-window features, compute its
   rank-AUC against the `RT − DA > 0` label over legal history only (`S <= D-2`). If no single
   feature clears ~0.55, and no small group does either, the representation is the constraint.
2. **Label-shuffle noise floor.** Run the identical B2/B3 pipeline on permuted labels to
   establish the empirical chance band for this exact 28-day / 672-slot scope. Several results
   here (B1 0.5863, B3 0.5789, T2 0.5789) sit close enough to that band that it must be measured
   rather than assumed.
3. **Feature-family comparison, if (1) shows anything.** Only then is it worth asking whether a
   different feature family — e.g. reserve/load-forecast derived features, or the day-ahead
   curve shape rather than level — carries the missing signal.

Deliberately **excluded** from this suggestion, because the audit gives them no supporting
evidence: TabM-only runs, Temporal-only runs, TafM `recent180` retraining (Case B measured the
opposite sign), checkpoint equivalence banding, calibration, threshold tuning (the oracle shows
a 3.7 pp ceiling), protected-gradient, Stage B, full Jan–Aug DEV, and the lockbox.

---

## 4. Status

**E1.5-A is complete. Stopping here.**

The plan's stop rule applies in full: no next stage, no new training, no promotion, no stage,
no commit. Any continuation requires a new explicit authorization.

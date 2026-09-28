# 31 — ROOT CAUSE RESET AUDIT: Why the model is stuck near 60%

STATUS=FROZEN
DATE=2026-09-26
PURPOSE=Stop local-search experimentation and identify the dominant modeling failure before E5.

## Executive conclusion

The dominant current problem is NOT:
- PLE choice,
- scalar fusion alpha,
- Strong/Weak gate,
- XGB selector recovery,
- simple recent-window training,
- post-hoc calibration,
- class weighting,
- or output smoothing.

The strongest evidence points to:

1. **Temporal representation failure**:
   the legal 168-hour history contains predictive information, but the current TemporalEncoder fails to extract it.

2. **Regime-dependent relationship inversion**:
   the mapping from legal market fundamentals to spread sign changes across windows, especially W3 June.

3. **Potential upstream information loss**:
   frozen_repro keeps target_spread and engineered states but no independent DA/RT price-level sequence; target_day_DA_as_feature=false.
   This is not yet proven to be wrong because availability at D-1 14:00 must be re-audited before any price-level source is added.

No further local architecture or hyperparameter sweep is authorized before E5-A.

## 1. E4-B closes selector recovery

E4-B:
F0 Q2 selected222:
Raw .6086 / Balanced .5424 / AUC .5911

F1 literature240:
Raw .5923 / Balanced .5269

F2 all259:
Raw .6012 / Balanced .5366

F1 loses 11 correct slots; F2 loses 5.
Gate=RECOVERY_HARMFUL.

Therefore the main bottleneck is not simply that the frozen selector removed 18/37 useful candidate features.

## 2. Current TemporalEncoder is structurally weak

Current TemporalEncoder:
- input [B,168,7]
- each physical group uses shared TimeMLP:
  Linear(168,64) -> GELU -> Linear(64,24)
- optional fixed FFT residual bins
- 6 legal target-day forecast counterparts added with scalar eta
- final Linear(7,d_time)

It does NOT explicitly model:
- target-hour-aligned daily lags,
- missing D-1 h15-24 availability masks,
- same-hour persistence across days,
- D-1 p1-p14 sequence as a recent context token set,
- recurrent/similar regimes,
- learned temporal attention or recurrent state transitions.

## 3. Direct evidence that temporal information exists but encoder misses it

E2-A A2 TEMPORAL_ONLY:
Raw=.5000
Balanced=.4747
AUC=.4978

Using the exact same legal X_hist and no learned deep model:
previous-available same-hour spread sign:
Raw=.5699
Balanced=.5366
+Recall=.4174
-Recall=.6558

Therefore:
the input temporal history contains useful signal that the current neural TemporalEncoder fails to retain.

## 4. Horizon-aligned diagnostic prototype

A fixed, non-tuned LightGBM diagnostic was built from:
- horizon-aligned spread lag1d/2d/3d/7d,
- lag1 availability mask,
- same-hour history mean/std/median/positive-rate,
- previous-available same-hour values for all 7 temporal channels,
- D-1 latest14 spread summary,
- legal target-day core fundamentals,
- hour sin/cos.

Same strict training eligibility. No target-day oracle.

Overall:
Raw=.5818
Balanced=.5007

Not a replacement model.

But by horizon segment:
H1 1-8 = .6027
H2 9-16 = .6250
H3 17-24 = .5179

The key result is H1:
Q2 H1=.5179
aligned diagnostic H1=.6027

Thus horizon alignment recovers about +8.5pp on the weakest segment.

## 5. Fixed non-oracle hybrid proves >63 is possible

A fixed architecture diagnostic:
- H1 1-8 uses aligned-temporal LightGBM
- H2/H3 use Q2
- segmentation was pre-existing and fixed; no per-day/month oracle choice.

Result:
Raw=.6369 = 428/672
Balanced=.5609
+Recall=.2893
-Recall=.8326
pred-positive fraction=.2113

Windows:
W1=.7440
W2=.5893
W3=.5536
W4=.6607

Segments:
H1=.6027
H2=.6920
H3=.6161

This proves:
- legal information/modeling can exceed 63 overall;
- the remaining obstacle is cross-regime stability, not an absolute 60% ceiling.

This hybrid is diagnostic only and is NOT promoted because W2/W3 are below the desired cross-month level.

## 6. W3 June is relationship inversion, not ordinary OOD

Fundamental-profile nearest-10 distance to legal historical days:
W1=.5678
W2=.1514
W3=.1272
W4=.1434

W3 is actually closest to historical fundamental profiles.

But label relationships flip.

Examples: high-quartile minus low-quartile positive-rate effect:

bidding_space_ratio:
W2 +.283
W3 -.346
W4 +.328

residual_load_renew:
W2 +.232
W3 -.418
W4 +.331

renewable_share:
W1 -.319
W2 -.300
W3 +.514
W4 -.297

Therefore W3 is not solved by nearest-history recency/similarity alone.
The model needs a regime variable/representation capable of changing the conditional mapping.

## 7. Similar-day diagnostic

Using only legal target-day fundamentals to retrieve historical similar days:

best fixed diagnostic among k={3,5,10,20,40,80} was k=10:
Raw=.6042
Balanced=.5697

Windows:
W1=.6964
W2=.6012
W3=.4524
W4=.6667

Again:
regime/sample similarity carries real signal and high Balanced Accuracy, but W3 relationship inversion remains unsolved.

## 8. Existing experts are complementary

Q2:
overall .6086; W3=.5595

R2 DROP_DIR historical artifact:
overall .5923; W3=.6071

Equal Q2/R2 probability ensemble:
overall=.6012
W3=.6012

Q2 + aligned equal:
overall=.6161

Fixed H1-aligned + Q2:
overall=.6369

Three-expert oracle upper bound, DIAGNOSTIC ONLY:
.7872

This is not a result to claim.
It only shows strong complementarity exists and motivates a learned regime gate rather than a single global mapping.

## 9. Literature alignment

### LBRM 2026
Ma et al., Journal of Renewable and Sustainable Energy 18, 045901.
DOI 10.1063/5.0336800.

Publicly verifiable abstract:
- predicts DA-vs-RT price-spread sign;
- reports average directional accuracy 74.99% across four seasons;
- explicitly targets complex nonlinear relationships and high-dimensional temporal dependencies;
- segments temporal data into patches and reprograms temporal representations.

We cannot verify from the accessible preview that its forecast-origin/data-availability contract is identical to D-1 14:00 -> full next-day 24h.
Therefore 74.99% is relevant evidence that the task can be learnable, but NOT a directly comparable benchmark until full methodology is verified.

### Recurrent regime literature
Marcos et al. 2020:
the most relevant calibration history need not be the most recent; fundamental regimes recur.

### Spread mechanism literature
Maciejowska et al. 2019:
direct spread-sign modeling is valid.

Hou & Bunn 2024:
DA-to-RT corrections depend on lagged price difference plus wind/solar/load forecast errors and residual-demand/state variables.

These support redesigning temporal/regime representation, not another scalar hyperparameter sweep.

## 10. Upstream data-interface concern

data/frozen_repro/manifest.json explicitly states:
target_day_actual_as_feature=false
target_day_DA_as_feature=false
d1_post14_realized_as_feature=false

Current future candidate registry includes spread history and many fundamentals but no independent DA/RT price-level feature family.

The current X_hist first channel is target_spread, so D-1 p1-p14 spread is present.
The issue is NOT loss of spread context.

But the model cannot distinguish price-level regimes from spread alone if DA/RT level information is genuinely available upstream.

Do NOT add target-day DA/RT prices until an upstream availability audit proves they are legal at D-1 14:00.

If E5-A does not produce material improvement, E5-B must audit the parent/raw source for:
- historical DA price sequence,
- historical RT price sequence,
- D-1 p1-p14 RT price level,
- legally available target-day DA cleared/forecast price,
- congestion/outage/tie-line/scarcity variables,
and their exact publication timestamps.

## 11. Frozen decision

Next = E5-A Horizon-Aligned Regime Temporal Rebuild.

No more:
selector recovery,
class weighting,
threshold tuning,
postprocessing,
smoothing,
PLE/k/depth/alpha/gate sweeps,
recent-window sweeps.

E5-A changes the temporal representation itself.

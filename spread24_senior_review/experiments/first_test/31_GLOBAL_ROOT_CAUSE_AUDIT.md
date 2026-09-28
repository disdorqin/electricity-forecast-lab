# 31 — Global Root-Cause Audit Before V3

STATUS=ROOT_CAUSE_FROZEN
DATE=2026-09-26
CURRENT_VALID_BEST=Q2_SEGMENT_HEADS
CURRENT_VALID_RAW=0.6086309524
PRIMARY_TARGET=stable cross-month Raw Direction >= 0.63, no class collapse

## 1. E4-B closes the selector-recovery branch

Formal E4-B:
- F0 selected222: Raw .6086 / Balanced .5424 / AUC .5911.
- F1 literature240: Raw .5923 / Balanced .5269.
- F2 all259: Raw .6012 / Balanced .5366.
- paired F1-F0 = -1.64pp, CI [-4.46,+0.89]pp.
- paired F2-F0 = -0.74pp, CI [-3.27,+2.08]pp.
- F1 safety FAIL; F2 safety PASS but worse.
- Gate = RECOVERY_HARMFUL.

Conclusion:
The 222-feature selector is NOT the primary reason the model is stuck near 60%.
Do not continue selector reopening or add more static features by default.

## 2. Branches already falsified

The following have been directly tested and are no longer priority:
- shared vs strict Direction checkpoint: no robust large gain.
- joint vs dir_only: no negative-transfer proof.
- Tabular/Temporal fusion alpha sweep: no clear scalar region.
- Strong-only / Weak-only: current combination best; Weak-only poor.
- horizon gate fixed/global/24h: no large effect.
- selector Strong-DIR/Strong-MAG routing: no stable role effect.
- PLE+raw vs raw-only vs PLE-only: no large encoding effect.
- recent60 / rolling365 / rolling1095 training windows: all worse than canonical.
- independent calibrator + segment/regime logistic stacking: failed; split cost dominates.
- recovered 18 / all37 Noise features: no gain.
- class weighting: diagnostic sqrt-balanced full28 Raw .6161 but safety fails.
- structured smoothing: Raw can exceed .63 only with severe positive-class collapse.

This pattern means the bottleneck is NOT one small knob.

## 3. What the source code actually does

### 3.1 Tabular path is slot-independent

TabularEncoder reshapes:
[B,24,F] -> [B*24,F]

The same TabM backbone is applied independently to every target hour.
Therefore the nonlinear feature extractor for hour h cannot directly see the other 23 target-day forecast hours.

It sees engineered cross-hour summaries/ramps, but it does NOT learn the whole target-day load/renewable/space profile jointly.

This is a major mismatch with day-ahead market physics, where the relative 24h profile carries regime information.

### 3.2 Q2 only specialized the final head

Q2 segment_heads fixed one problem:
H1/H2/H3 no longer share the final 32->1 Direction boundary.

But:
- TabM backbone remains shared across all 24 hours.
- tab_dir/time_dir adapters remain shared.
- the future feature representation remains hour-independent.

Q2 therefore has segment-specific DECISIONS but not segment-specific REPRESENTATIONS.

### 3.3 Temporal path is not a real multivariate sequence model

X_hist is legal and DOES contain target_spread as the first of 7 channels through D-1 14:00.
So D-1 h1-h14 spread was NOT omitted.

However TemporalEncoder:
- processes each history channel with a channel/group MLP mapping 168 -> 24;
- adds selected Fourier residuals;
- only mixes the 7 channel forecasts after that with Linear(7,d_time).

It has no GRU/TCN/attention over the multivariate 168h sequence and no explicit same-slot/day retrieval mechanism.

Temporal-only E2-A was near chance.
Thus the issue is representation, not missing history input.

## 4. New legal diagnostics performed after E4-B

### 4.1 Simple historical-spread rules

Using only strict X_hist target_spread:
- previous available same-hour sign: Raw .5699 / Balanced .5366.
- H2 alone: Raw .6652.
- recent3 mean sign: Raw .5714.

History has local signal but cannot explain overall 63+ alone.

### 4.2 Similar-day fundamental retrieval

Pre-registered daily profile used:
- fcast_直调负荷
- fcast_竞价空间
- fcast_新能源总加
- fcast_风电总加
- fcast_光伏总加
- residual_load_renew
- bidding_space_ratio
- renewable_share
- net_ramp_pressure
- ramp_tightness

For each target D:
- only legal historical days <= D-2;
- robust distance scale fit on those historical days only;
- compare complete 24x10 target-day known forecast/fundamental profiles;
- inverse-distance sign vote.

Diagnostic k values were inspected; k=10 gave:
Raw=.6042
Balanced=.5697
+Recall=.4463
-Recall=.6930

By segment:
H1=.5982
H2=.6384
H3=.5759

Q2 by segment:
H1=.5179
H2=.6920
H3=.6161

Key observation:
similar-day is much better in H1; Q2 is much better in H2/H3.

### 4.3 Fixed segment hybrid — FIRST non-collapse 63+ discovery candidate

Fixed business rule:
H1 (1-8) -> similar-day k=10
H2/H3 (9-24) -> Q2

No per-day oracle.
No target-truth routing.
No threshold tuning.
The H1/H2/H3 segmentation existed before this diagnostic.

On the CURRENT 28-day discovery panel:

Raw = .6354
Balanced = .5715
+Recall = .3430
-Recall = .8000
pred+ fraction = .2515
one-class days = 1
+Recall=0 days = 5
-R=0 days = 0

Q2:
Raw=.6086 / Balanced=.5424 / +R=.3058 / -R=.7791
one-class=2 / +R0=8.

Thus the 63.54% hybrid is NOT a majority-collapse artifact.

Paired hybrid-Q2:
delta Raw = +2.6786pp
95% day-bootstrap CI = [-1.64,+6.99]pp
W/T/L = 15/4/9.

Windows:
W1=.7143 / Balanced=.6032
W2=.6190 / Balanced=.6210
W3=.5595 / Balanced=.5093
W4=.6488 / Balanced=.5196

Conclusion:
We have crossed 63 overall in discovery, safely.
We have NOT achieved stable cross-month 63 because W3 remains the failure regime.

This result is post-hoc discovery because k=10 and the hybrid idea were inspected on the same 28-day panel.
It MUST NOT be called promoted until independent confirmation.

## 5. Why W3 is the real remaining problem

W3 positive prevalence=.3571.

W3 segment Raw:

Q2:
H1=.4821
H2=.5000
H3=.6964

similar-day k10:
H1=.4821
H2=.4464
H3=.4286

So W3 is NOT an H3 problem.
It is specifically H1/H2 where BOTH the neural model and simple retrieval are near chance.

W3 fundamental state differs:
- residual_load_renew mean ~52095
- renewable_share ~.233
- bidding_space_ratio ~.210
- ctx_spread_positive_rate14 ~.469 (highest among W1-W4)
- target positive prevalence only .357

The recent D-1 spread state is unusually positive while target-day signs are less positive.
This is a regime-reversal / conditional-mapping problem, not a simple class-prior problem.

## 6. Historical evidence we should not forget

Old strict-history reference:
N-BEATSx on a short legal window reached roughly:
Raw 65.48%
+Recall 72.68%
-Recall 55.63%

It did not prove cross-month stability, but it demonstrates that strict leakage-free data can support >65 locally with a different temporal inductive bias.

Historical P6 Similar-Day ~69.44% is INVALID because it used D-1 complete labels.
Do not reuse that number as evidence.

## 7. Literature audit — what is actually comparable

### Ma et al., JRSE 2026
A large language model-based reprogramming method for electricity price spread prediction.
DOI 10.1063/5.0336800.

Reported average directional accuracy 74.99% and MCC .5706 across four seasons.
The paper explicitly argues existing models struggle with complex nonlinear relationships and high-dimensional temporal dependencies, and uses patch/token reprogramming into a frozen LLM.
It compares against ConvTrans, BiLSTM, CNN-self-attention, LSTM-Attention, PatchTST, iTransformer and Time-LLM.

Important:
This proves 70+ spread-sign accuracy is possible in at least one study/dataset.
It does NOT prove 80-90% is the normal leakage-free level for our D-1 14:00 Shandong contract.

### Sun et al., ICPE 2024 / IEEE 2025
Price Spread Direction Prediction Based on an Improved LSTM Model in China's Electricity Spot Market.
DOI 10.1109/ICPE64565.2024.10929104.

Uses real wind-farm data from Shandong and emphasizes feature engineering + Multi-Layer LSTM.
The accessible abstract states improvements in F1/precision/backtest return, not a directly comparable 80-90% accuracy claim.

### Recent Joint forecasting/bidding work
A 2025 multi-agent spread/bidding paper reports Precision 53.25%, Recall 40.45%, AA 56.05%, SWA 57.36%.
Therefore public spread-direction work is NOT uniformly 80-90%.

### Li 2026 Shandong working paper
The Missing Arbitrageur: Information, Learning, and Storage in China's Electricity Spot Market.
Uses 4.7 years Shandong hourly prices and reports that rich public information sets and multiple forecasting models have little meaningful OOS spread predictability as the market matures.
This is a working paper, not treated as final peer-reviewed truth, but it is a strong warning against assuming 90% is a universal attainable benchmark.

## 8. Root-cause verdict

The current bottleneck is primarily REPRESENTATION / EXPERT SPECIALIZATION:

1. target-day 24h future fundamentals are encoded hour-by-hour, not as a joint daily profile;
2. historical 168h multivariate sequence is flattened/channel-wise rather than modeled with a true sequence encoder;
3. Q2 specialized only the last readout, not the feature extractor;
4. a legal nonparametric model using the whole 24h profile recovers ~8pp in H1 versus Q2 and creates a safe 63.54% fixed hybrid.

Therefore further selector/loss/threshold/gate micro-tuning is low priority.

## 9. Next architecture: V3 Retrieval/Profile Expert

V3 must add two pieces WITHOUT replacing Q2 prematurely:

A. GLOBAL DAY-PROFILE CONTEXT
Encode the complete legal 24x10 target-day fundamental profile jointly, so each hour can condition on the full-day shape.

B. CAUSAL SIMILAR-DAY RETRIEVAL PRIOR
For every training/evaluation day t, retrieve only days <=t-2.
Produce per-hour historical sign probability + retrieval confidence.
Never compute training priors using the sample's own/future labels.

The new branches are Direction-only.
Magnitude/production remain untouched.

## 10. Three-conversation acceleration rule

This is the final architecture pivot before reassessing the project target.

Conversation/run 1:
E5-A discovery V3 on current 28-day panel.

If any V3 arm >=.63 Raw + safety:
freeze exact architecture/hyperparameters.

Immediately in SAME Codex task, run independent confirmation:
C1 Jan12-18
C2 Mar12-18
C3 May12-18
C4 Jul12-18 2026

No tuning on confirmation.

Report:
- confirmation overall Raw/Balanced
- each C1-C4 Raw/Balanced
- STRICT_STABLE_63 = min(C1..C4 Raw) >= .63
- CROSS_MONTH_63_CANDIDATE = overall >=.63, safety PASS, no window below .60.

If confirmation fails:
do NOT tune E5-A on confirmation.
Next and last major path is a true multivariate sequence encoder (GRU/TCN/Patch) replacing the current TemporalEncoder, guided by the LBRM/N-BEATSx evidence.

No more PLE/alpha/selector/class-weight/postprocess micro-sweeps.

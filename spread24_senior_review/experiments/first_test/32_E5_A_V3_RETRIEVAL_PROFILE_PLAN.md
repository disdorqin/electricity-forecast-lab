# 32 — E5-A Retrieval-Augmented Profile Expert (V3)

STATUS=AUTHORIZED_MAJOR_PIVOT
DATE=2026-09-26
PARENT=E4-B
DISCOVERY_PANEL=W1_Feb/W2_Apr/W3_Jun/W4_Aug
INDEPENDENT_CONFIRMATION=Jan/Mar/May/Jul_if_triggered
PRIMARY_TARGET=stable cross-month Raw Direction >=63%, safety PASS

## 1. Fixed anchor

V0 = Q2 SEGMENT_HEADS
- selected222
- canonical PLE+raw
- Strong211 + Weak11
- current 24h gate
- full_current Tabular+Temporal
- fusion alpha=.8
- segment_heads
- unweighted BCE
- canonical Stage-A split
- seed20260924
- k8

V0 discovery Raw=.6086 / Balanced=.5424.

All E4-A/E4-B/class-weight/structured-decoder switches must be OFF.

## 2. Legal profile feature set — EXACT10

PROFILE_FEATURES:
1 fcast_直调负荷
2 fcast_竞价空间
3 fcast_新能源总加
4 fcast_风电总加
5 fcast_光伏总加
6 residual_load_renew
7 bidding_space_ratio
8 renewable_share
9 net_ramp_pressure
10 ramp_tightness

All are already legal target-day future features available by D-1 14:00.

No new raw source.

## 3. Causal retrieval prior

For EACH sample day t, including BASE_TRAIN and MONITOR samples:

candidate historical neighbor days must satisfy:
neighbor_day <= t-2
and pass the same selected222 quarantine contract.

Distance input:
complete 24x10 PROFILE_FEATURES.

Scaling:
fit robust median/IQR using that sample's legal neighbor pool only.
No global future fit.

Distance:
mean squared robust-scaled profile distance over 24x10.

k = 10 FIXED for E5-A.
No k sweep.

Weight:
w_i = 1/(distance_i + 1e-4), normalized.

Outputs per sample:
- retrieval_p[24] = weighted historical P(Y>0) by hour
- retrieval_mean_distance scalar
- retrieval_min_distance scalar
- retrieval_effective_neighbors = 1/sum(w_i^2)

Fail closed if <10 legal historical neighbors.

IMPORTANT:
For a training sample t, retrieval labels come only from days <=t-2.
No self label, t-1 label, or future label.
This is the central leakage gate.

Cache retrieval features deterministically with provenance:
source/sequence/selector/profile feature hashes + k + distance contract.

## 4. Whole-day profile encoder

Input:
BASE-preprocessor-scaled target-day PROFILE_FEATURES [B,24,10].

Because all 24 target-day forecasts are legally known at D-1 14:00, bidirectional processing across the 24 future hours is legal.

Use a SMALL profile encoder:
BiGRU(input=10, hidden=32 per direction, num_layers=1)
-> output [B,24,64]
-> Linear(64,d_task=32) + GELU
-> profile_context [B,24,32]

No Transformer.
No architecture sweep.
No hidden-size sweep.

Purpose:
give every target hour access to the full-day load/renewable/space shape.

## 5. Retrieval injection

Convert retrieval_p to clipped logit:
rlogit = logit(clamp(p,.05,.95)).

Create retrieval feature vector per hour:
[rlogit, normalized mean_distance, normalized effective_neighbors]

Small MLP:
Linear(3,16)->GELU->Linear(16,d_task)

This produces retrieval_context [B,24,32].

Distance/confidence normalization must be fit on BASE only.

## 6. Integration with Q2

Keep Q2 h_dir [B,k,24,d_task].

Add Direction-only residual context before segment heads:

h_dir_v3 =
h_dir_q2
+ profile_gate_seg[s] * profile_proj(profile_context)
+ retrieval_gate_seg[s] * retrieval_proj(retrieval_context)

Use exactly 3 profile gates + 3 retrieval gates, one per fixed H1/H2/H3 segment.

Gate parameterization:
sigmoid(logit)
initial effective gate = .10.

No target-day oracle gating.
No per-month gate.
Same gates learned from BASE.

Magnitude path unchanged.

## 7. Arms

V0 Q2_ANCHOR
reuse Q2.

V1 PROFILE_ONLY
Q2 + whole-day BiGRU profile context.
retrieval gate structurally absent.

V2 RETRIEVAL_ONLY
Q2 + causal similar-day retrieval context.
profile gate structurally absent.

V3 PROFILE_PLUS_RETRIEVAL
both branches active.

No other arms.

## 8. Why V1/V2/V3 are big-block, not micro-tuning

V1 tests the identified slot-independence flaw.
V2 tests the recurrent-regime / similar-day signal.
V3 tests whether parametric daily context and nonparametric historical analogs are complementary.

No feature/hidden/k/alpha tuning.

## 9. Gate A

Before formal runs:

1 V0/default reproduces saved Q2 benchmark <=1e-9.
2 PROFILE_FEATURES exact10 and all selected/legal.
3 profile encoder consumes [B,24,10], output [B,24,32].
4 bidirectional future-profile processing uses only target-day forecasts known at origin.
5 retrieval for every sample t uses only neighbor days <=t-2.
6 retrieval training sample never sees own/t-1/future labels.
7 retrieval candidate days obey selected222 quarantine.
8 deterministic retrieval cache hash reproducible.
9 target-day retrieval uses <=D-2.
10 k exactly10; no tuning.
11 retrieval_p finite/in [.0,1].
12 confidence normalization BASE-only.
13 V1 profile branch receives L_dir gradient; retrieval absent.
14 V2 retrieval branch receives L_dir gradient; profile absent.
15 V3 both receive L_dir gradient.
16 Q2 TabM/Temporal/segment_heads remain active.
17 Magnitude path bit-identical when Direction-only branches toggle.
18 postprocess/class-weight/feature-recovery/structured decoder OFF.
19 source/config/selector/sequence hashes frozen.
20 canonical94 + all prior tests + new E5 tests PASS.

## 10. Benchmark day

2026-02-13 engineering only.
No ranking.

Record:
metrics, gates, gradients, params, runtime, retrieval neighbors/distances.

## 11. Discovery formal panel

Same 28:
W1 2026-02-12..18
W2 2026-04-12..18
W3 2026-06-12..18
W4 2026-08-07..13.

V0 reuse.
V1/V2/V3 fresh28.

No intermediate stopping.

Training:
same Q2 protocol.
Only V3 branches differ.

## 12. Reference hybrid (diagnostic comparator)

Also report HFIX:
H1 retrieval hard decision (p>=.5)
H2/H3 Q2

On discovery it should reproduce approximately:
Raw=.6354 / Balanced=.5715.

HFIX is a reference, not automatically promotable because the rule/k were discovered on this same panel.

## 13. Metrics

Overall Raw/Balanced/+R/-R/AUC/Brier/ppf/collapse.
W1-W4 + min-window.
H1-H3.
Paired V1/V2/V3 vs V0.
Slot gains by segment/window.
Gate values and retrieval confidence.

Safety anchor V0:
Balanced >= V0-.01
+Recall >= V0-.05
one-class <= V0+2
+R0 <= V0+2.

## 14. Discovery Gate

V3_SIGNAL one of:
PROFILE_CONTEXT_CONFIRMED
RETRIEVAL_CONFIRMED
COMPLEMENTARY_V3
PROMISING_63_UNCONFIRMED
NO_V3_GAIN
V3_HARMFUL

If any V1/V2/V3 >=.63 AND safety PASS:
freeze exact winning configuration.
NO tuning.
Proceed automatically to independent confirmation below.

If none >=.63:
STOP before confirmation.
Next major route = true multivariate temporal encoder replacement.

## 15. Independent confirmation panel — only if discovery trigger fires

Exact dates frozen now:
C1 2026-01-12..01-18
C2 2026-03-12..03-18
C3 2026-05-12..05-18
C4 2026-07-12..07-18

These dates are not used to choose E5 architecture/k/gates.

Run:
- V0 Q2 anchor fresh
- frozen E5 discovery winner fresh
- HFIX frozen reference

No tuning, no arm changes, no threshold changes.

Report each C1-C4 and overall.

Definitions:
STRICT_STABLE_63:
min(C1,C2,C3,C4 Raw) >= .63 AND safety PASS.

CROSS_MONTH_63_CANDIDATE:
confirmation overall Raw >= .63,
Balanced safety PASS,
no confirmation window Raw < .60.

If either succeeds:
candidate is frozen for broader DEV.

If neither:
do not tune on confirmation.
Proceed to E6 true multivariate sequence encoder.

## 16. Forbidden

No selector changes.
No new feature source.
No class weighting.
No postprocess.
No threshold tuning.
No structured smoothing.
No k sweep.
No distance metric sweep.
No GRU hidden sweep.
No alpha/PLE/depth/k TabM sweep.
No Stage B.
No per-window/per-month routing.
No confirmation-panel tuning.

## 17. Outputs

experiments/first_test/E5_v3_retrieval_profile/

root_cause_snapshot.md
retrieval_cache_audit.json
retrieval_neighbor_audit.csv
benchmark/
discovery/
confirmation/ if triggered
daily_metrics.csv
window_metrics.csv
hour_metrics.csv
paired_summary.csv
slot_gain_summary.csv
gate_trajectories.csv
component_diagnostics.csv
runtime.csv
E5_A_summary.md
E5_A_GATE.md

STOP after discovery if no >=.63 arm.
If triggered, STOP after confirmation.

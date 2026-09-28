# E4-B — Frozen Selector Reopening via Weak Feature Recovery

STATUS=AUTHORIZED_NEXT
DATE=2026-09-26
PARENT=E4-A
EXPERIMENT_ROOT=experiments/first_test/E4_feature_recovery
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS
ANCHOR=E2-E1_Q2_SEGMENT_HEADS

## 0. Why this experiment now

Current valid best: Q2 SEGMENT_HEADS Raw=.6086 / Balanced=.5424 / AUC=.5911 / safety PASS.
E4-A postprocessing failed: P0=.6086, P1=.5833, P2=.5908, P3=.5848; no postprocess arm is safety-admissible.
Stop postprocess/calibrator/L2/threshold work.

Two diagnostic side branches are NOT promotion evidence:
- structured decoder can reach Raw=.6324, but +Recall=.1653 and one-class days=19 -> majority-collapse, invalid.
- sqrt-balanced class weighting completed 28 days at Raw=.6161, but overall Balanced=.5287 and W4 Balanced=.4593; paired Raw CI includes zero and safety fails.
Therefore do not chase Raw via class bias / smoothing.

The next high-value question is whether the frozen XGB selector removed features that are weak globally but useful to Q2 under specific market regimes.

## 1. Literature rationale

A. Maciejowska, Nitka, Weron (Energies 2019, DOI 10.3390/en12040631): direct spread/sign modelling is valid and economically relevant.
B. Hou & Bunn (The Energy Journal 2024, DOI 10.1177/01956574241281143): DA-to-real-time correction explicitly depends on lagged price difference plus wind, solar and load forecast errors; day-ahead side also uses residual demand/fundamental state.
C. Marcos et al. (Energies 2020, DOI 10.3390/en13205452): electricity dynamics are recurrent/episodic; the most relevant information is not necessarily the most recent. Fundamental regime indicators can identify relevant states.
D. Huang et al. (Applied Energy 2024, DOI 10.1016/j.apenergy.2024.123863): Shandong forecasting combines similar-day information, XGBoost feature selection and DNN. Selector use is valuable but a selector trained for another learner/task is not automatically optimal for final neural Direction modelling.
E. Sun et al. (ICPE 2024 / IEEE Xplore 2025, DOI 10.1109/ICPE64565.2024.10929104): Shandong actual-data spread-direction work performs feature engineering before Multi-Layer LSTM.

## 2. Frozen inventory

Candidate future features=259. Frozen selector selected=222. Dropped=37, all labelled Noise.
Current selected222 already contain most major families:
- forecast-error 63/78
- spread-state 33/33
- scarcity/bidding-space 33/36
- ramp/pressure 13/17
- uncertainty 34/40
- residual-load/renewable 32/47

The hypothesis is NOT that the project lacks forecast-error features.
The hypothesis is that some globally weak/dropped features may carry conditional regime signal useful to Q2.

## 3. Pre-registered literature/regime recovery set — EXACTLY 18

1 fcast_renewable_adjusted_direct_load
2 ramp2_load
3 ramp2_solar
4 ramp2_renewable
5 ramp2_residual_load
6 err_renewable_adjusted_direct_28d_mean
7 delta_renewable_adjusted_direct_28d_mean
8 err_renewable_adjusted_direct_28d_std
9 delta_renewable_adjusted_direct_28d_std
10 err_renewable_adjusted_direct_28d_q10
11 err_renewable_adjusted_direct_28d_q90
12 delta_renewable_adjusted_direct_28d_q90
13 regime_high_residual_load_renew
14 regime_low_residual_load_renew
15 regime_high_renewable_share
16 regime_low_renewable_share
17 regime_high_bidding_space_ratio
18 regime_low_bidding_space_ratio

Mechanistic groups:
A renewable-adjusted direct-load forecast/error state = 1,6-12
B second-order ramp dynamics = 2-5
C explicit fundamental regime flags = 13-18

All 18 already exist in the frozen SequenceStore candidate cube. No new raw source is introduced.
Registry audit before this plan: availability AVAILABLE, allow_in_future=true, no label dependency; target-day forecast-derived values obey D-1 14:00; historical actual-error stats use causal D-2-or-earlier shifts where required.

## 4. Eligibility audit already performed

An experiment manifest rebuilt in canonical 259-feature registry order gives the SAME eligible-day counts for selected222, literature240 and all259 on representative W1-W4 targets:
2026-02-12: 1494 / 1494 / 1494
2026-04-12: 1553 / 1553 / 1553
2026-06-12: 1614 / 1614 / 1614
2026-08-07: 1670 / 1670 / 1670

Gate A must repeat for all 28 formal days and require identical eligible-index SETS, not just counts.

## 5. Why recovered features enter Weak, not Strong

Do NOT relabel recovered Noise as Strong in E4-B.
Current Strong/Core path=211 and is the main Q2 signal carrier.
Changing Strong input would confound selector recovery with TabM high-capacity input/capacity changes.
Recovered features must be routed as low-capacity Weak/residual inputs.

Existing selected Weak=11.
F0 selected222: Weak11
F1 literature240: Weak29
F2 all259: Weak48
Strong/Core must remain exactly211 in all arms.

## 6. Arms

F0 SELECTED222 — reuse Q2, Strong211 + Weak11, Raw=.6086.
F1 LITERATURE240 — fresh28; selected222 + exact18; recovered18 routed Weak; Strong211 Weak29.
F2 ALL259_WEAK_RECOVERY — fresh28; all259; all37 dropped Noise routed Weak; Strong211 Weak48.
No other formal arm.

## 7. Clean implementation contract

Add experiment-only feature_recovery_profile = selected222 / literature240 / all259. Default=selected222.
DO NOT modify frozen selector JSON on disk. DO NOT rerun selector.

Build an in-memory experiment feature manifest from the frozen selector + frozen feature-registry candidate order:
1 selected_features follows canonical 259 registry order.
2 selected_indices exactly match names in SequenceStore.feature_names.
3 existing selected roles unchanged.
4 recovered features assigned model role Weak only in experiment manifest.
5 preserve base frozen selector SHA separately.
6 compute experiment_feature_profile_sha256 separately.

Do not claim experiment manifest has original selector SHA.
fit_preprocessor, SequenceStore.eligibility, selector_indices and quarantine logic must operate normally on the experiment manifest.
Never bypass order/quarantine contracts.

## 8. Q2 architecture fixed

Fresh runs:
- objective_mode=dir_only
- checkpoint_policy=direction_first
- gradient_policy=vanilla
- direction_readout_mode=segment_heads
- architecture_mode=full_current
- direction_tabular_mode=current
- direction_horizon_gate_mode=current
- strong_role_profile=all
- numeric_encoding_mode=canonical
- direction_fusion_alpha=.8
- direction_postprocess_mode=none
- direction_class_weight_mode=unweighted
- canonical Stage-A split
- profile=default
- seed=20260924
- k=8
- max_epochs=120
- patience=15
- batch_size=64
- frozen LR/WD
- CUDA+AMP
- Stage B OFF

Only feature profile / Weak input changes.

## 9. Gate A

Before formal runs:
1 selected222 profile reproduces saved Q2 benchmark <=1e-9.
2 default/no flag remains bit-identical canonical selected222.
3 F0 exactly222 / Strong211 / Weak11.
4 F1 exactly240 / recovered exact18 / Strong211 / Weak29.
5 F2 exactly259 / recovered exact37 / Strong211 / Weak48.
6 selected feature order equals canonical SequenceStore registry order.
7 selected_indices exactly correspond to that order.
8 frozen selector file/sha unchanged on disk.
9 experiment profile SHA recorded separately.
10 all recovered features are original frozen-selector role Noise.
11 recovered features are Weak only in experiment manifest.
12 all 28 target days eligible in all profiles.
13 all 28 profiles have identical eligible-index SETS.
14 preprocessing fits BASE only for that profile.
15 no label dependency / target truth / D-1 unavailable actual enters recovered features.
16 source/sequence/config hashes frozen.
17 Q2 segment heads + k8 preserved.
18 no postprocess/class-weight/structured decoder active.
19 canonical94 + all prior E2/E3/E4 tests + focused E4-B tests PASS.
20 incompatible profile combinations fail closed.

## 10. Benchmark

2026-02-13 engineering only. F0 reuse/exact reproduction; F1/F2 fresh.
Record feature inventory/hash, eligible-index digest, metrics, params, runtime/GPU, best/stop epoch, Weak contribution diagnostics.
No ranking from benchmark.

## 11. Formal DEV

Same frozen28 days:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

F0 reuse Q2; F1 fresh28; F2 fresh28. No intermediate stopping.

## 12. Metrics

Overall Raw/Balanced/+Recall/-Recall/AUC/Brier/ppf/collapse.
W1-W4 Raw/Balanced/AUC/+R/-R + min-window Raw + window std.
H1-H3 Raw/Balanced/AUC/+R/-R/Brier.
Paired F1-F0, F2-F0, F1-F2 with day-cluster bootstrap 10,000 seed20260924; 95% CI + W/T/L.
Slot gains overall/H1/H2/H3.
Architecture diagnostics: params, Weak input dim, Weak gated norm/effective Weak fraction, Strong/Weak cosine, gradient norms Strong/Weak/Temporal, best/stop epoch, runtime/GPU.

## 13. Feature-family diagnostics

For F1 diagnostic only, on MONITOR only, report aggregate occlusion/permutation diagnostics for fixed groups:
A renewable-adjusted error group 1,6-12
B ramp2 group 2-5
C regime flag group 13-18

Do NOT use target-day truth to choose groups. Do NOT create new formal arms inside E4-B.
For F2 report the 19 additional non-literature recovered Noise names separately.

## 14. Safety

Anchor F0/Q2.
SAFETY_ADMISSIBLE iff:
Balanced >= F0-.01
+Recall >= F0-.05
one-class days <= F0+2
+R=0 days <= F0+2
Raw-only majority gains are rejected.

## 15. Pre-registered Gate

FEATURE_RECOVERY_SIGNAL exactly one:
DOMAIN_GUIDED_RECOVERY
BROAD_SELECTOR_BOTTLENECK
RECOVERY_PROMISING_UNPROVEN
SELECTOR_ROBUST
RECOVERY_HARMFUL
NO_CLEAR_RECOVERY_EFFECT

DOMAIN_GUIDED_RECOVERY: F1 safety-admissible and materially/stably improves F0; F1 matches/beats F2 or F2 adds no clear extra value.
BROAD_SELECTOR_BOTTLENECK: F2 safety-admissible and materially/stably improves F0; F2 clearly exceeds F1.
RECOVERY_PROMISING_UNPROVEN: F1 or F2 gains >=~2pp or reaches >=.62 with safety, but paired/cross-window evidence not established.
SELECTOR_ROBUST: neither improves materially and both remain close/admissible.
RECOVERY_HARMFUL: recovered features materially worsen Q2 or fail safety.
NO_CLEAR_RECOVERY_EFFECT: mixed/noisy result.

## 16. Acceleration

If F1 or F2 >=.62 Raw AND safety PASS: STOP after E4-B; next immediate 3-seed F0+winner; no feature micro-tuning first.
If confirmed >=.62: freeze Q2 + feature profile and broaden DEV.
If confirmed >=.63: broader DEV before further architecture work.
If F1 promising: confirmation must include 3 seeds + matched-size random recovery controls from remaining Noise pool before promotion.
If F2 improves but F1 does not: next family attribution on remaining19 Noise features.
If E4-B fails: next = E5 recurrent-regime/similar-day sample selection or weighting using legal target-day fundamental state. Do NOT return to simple recency windows.

## 17. Forbidden

No class weighting in E4-B.
No structured decoder/smoothing.
No postprocess/logistic stacker.
No selector rerun.
No new raw source.
No arbitrary feature engineering beyond frozen259.
No threshold tuning.
No alpha/gate/PLE/k/depth/width changes.
No Stage B.
No recency-window sweep.
No per-day oracle feature profile.
No broader DEV/lockbox.

## 18. Outputs

experiments/first_test/E4_feature_recovery/
00_PLAN_SNAPSHOT.md
benchmark/
runs/
feature_profile_audit.csv/json
eligibility_identity.csv
daily_metrics.csv
model_summary.csv
window_metrics.csv
hour_segment_metrics.csv
paired_summary.csv
slot_gain_summary.csv
family_diagnostics.csv
component_diagnostics.csv
runtime.csv
E4_B_summary.md
E4_B_GATE.md
figures/

After E4-B STOP for human review.
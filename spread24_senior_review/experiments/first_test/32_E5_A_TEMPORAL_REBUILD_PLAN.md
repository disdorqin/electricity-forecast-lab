# E5-A — Temporal Representation Rebuild: GRU Query + Explicit Lag State

STATUS=AUTHORIZED_NEXT
DATE=2026-09-26
PARENT=E4-B
EXPERIMENT_ROOT=experiments/first_test/E5_temporal_rebuild
PRIMARY_GOAL=TRUE_DIRECTION_SIGNAL_NOT_RAW_COLLAPSE
DEVELOPMENT_TARGET=OVERALL_63_PLUS_WITH_SAFETY
STABLE_TARGET=REPORT_ALL_FOUR_WINDOWS_SEPARATELY; DO_NOT CALL STABLE_63 UNLESS EACH WINDOW >= .63

## 0. Why this is a major redesign, not another micro-ablation

Read first: experiments/first_test/31_E5_ROOT_CAUSE_AUDIT.md

Valid best remains Q2 SEGMENT_HEADS:
Raw=.6086 / Balanced=.5424 / AUC=.5911 / safety PASS.

E4-B feature recovery failed:
F1 literature240=.5923, F2 all259=.6012, Gate=RECOVERY_HARMFUL.

Simple legal temporal/regime diagnostics:
- current E2-A TEMPORAL_ONLY: Raw=.5000 / Balanced=.4747 / AUC=.4978 (old shared head)
- previous-available-same-hour spread-sign rule: Raw=.5699 / Balanced=.5366
- day-profile similar-day diagnostic k=10: Raw=.6042 / Balanced=.5697
- Q2 vs similar-day probability correlation ~.171
- hard disagreement ~.394
- either-expert-correct oracle union ~.8036
- simple fixed blend max DEV diagnostic only ~.6131
- cross-window shallow router ~.6176 but collapses +Recall ~.095

Interpretation:
Temporal/regime information exists and is complementary, but current representation and output-level patching do not exploit it robustly.

Current TemporalEncoder is shallow:
7 legal history channels -> group TimeMLP Linear(168,64)->GELU->Linear(64,24) + sparse FFT residual -> 7-to-d_time projection.
It has no recurrent sequence state, no horizon-query attention, no explicit same-hour lag structure, and only tiny scalar future conditioning.

E5-A changes the temporal inductive bias itself.

## 1. Fixed non-temporal architecture

All arms keep exactly:
- Q2 H1/H2/H3 segment Direction heads
- Tabular current selected222: Strong211 + Weak11
- canonical PLE+raw
- current 24-h tabular horizon gate
- full_current dual branch
- fixed direction_fusion_alpha=.8
- objective_mode=dir_only
- checkpoint_policy=direction_first
- gradient_policy=vanilla
- canonical Stage-A 80/20 split
- no Stage B
- direction_postprocess_mode=none
- direction_class_weight_mode=unweighted
- feature_recovery_profile=selected222
- no structured decoder

Only TemporalEncoder implementation changes.

## 2. Arms

### V0 CURRENT_TIMEMLP
Reuse Q2.
Existing TemporalEncoder.

### V1 GRU_QUERY
Fresh 28 days.

Replace current TemporalEncoder with a true sequence encoder:
- input x_hist [B,168,7]
- one bidirectional GRU, input_size=7, hidden_size=32 per direction
- output history states H [B,168,64]
- 24 learned horizon embeddings [24,64]
- legal target-day future fundamental query projection
- horizon-specific dot-product attention from 24 queries over all 168 history states
- context [B,24,64]
- Linear(64,d_time=32)+GELU -> h_time

No FFT in V1.
No explicit lag branch in V1.

### V2 GRU_QUERY_LAG
Fresh 28 days.

Same GRU-query encoder as V1 plus an explicit legal lag-state residual.

For each target horizon h=1..24, construct from the normalized target_spread history channel:
- same-business-hour lag values for lag-day 1..7 when available
- lag1 unavailable mask for h15..24
- recent D-1 p1-p14 state summaries:
  last p14 value
  mean last3
  mean last6
  mean all14
  std all14

Map lag-state vector through a small MLP to 32 dims.
Concatenate [attention_context64, lag_embed32] -> Linear(96,d_time=32)+GELU.

Do NOT add KNN/similar-day prior in E5-A.
That is reserved for E5-B only if needed, because the diagnostic k=10 was observed on DEV and must not silently become a formal hyperparameter.

## 3. Future query features — exact frozen list

Use exactly these already-selected legal target-day features:
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
11 ctx_spread_positive_rate14
12 spread_same_slot_28d_positive_rate

All 12 must already be in selected222.
No actual target-day feature.
No target_spread future feature.

future_query = learned_horizon_embedding + Linear(12,64)(x_future_query_features).

## 4. Exact lag-index contract

History timestamps are frozen:
D-8 15:00 ... D-1 14:00, indices 0..167.

hour_business mapping is timestamp.hour, with midnight represented as business hour24 of the previous business day.

For target business hour h in 1..24 and lag-day d:
same-hour history index = 177 + h - 24*d.

Required unit examples:
- h1 lag1 = 154
- h14 lag1 = 167
- h15 lag1 = 168 => unavailable, MUST mask
- h24 lag1 = 177 => unavailable, MUST mask
- h15 lag2 = 144
- h24 lag2 = 153
- h1 lag7 = 10
- h24 lag7 = 33

For an unavailable lag1:
- value input = 0 after normalization
- separate missing mask = 1
Available lag values have mask=0.

Do not use D-1 h15-24 realized spread.

## 5. Clean implementation

Add experiment-only temporal_encoder_mode:
- current
- gru_query
- gru_query_lag
default=current.

Preferred new module:
src/TafM_改进源码/models/temporal_encoder_v2.py

Do not rewrite canonical temporal_encoder.py except minimal shared utilities if necessary.
DualBranchV21 chooses encoder by mode.

Canonical current mode must remain bit-identical.

Do not touch source dataset or rebuild SequenceStore.
All lag-state features are gathered from existing transformed x_hist only.
All query features come from existing transformed selected x_future only.

## 6. Fail-closed contract

Non-current temporal_encoder_mode requires:
- legacy_v20=False
- objective_mode=dir_only
- architecture_mode=full_current
- direction_readout_mode=segment_heads
- direction_tabular_mode=current
- direction_horizon_gate_mode=current
- strong_role_profile=all
- numeric_encoding_mode=canonical
- feature_recovery_profile=selected222
- direction_postprocess_mode=none
- direction_class_weight_mode=unweighted
- direction_fusion_alpha=.8
- train_mode=stage_a

Reject incompatible combinations.

## 7. Gate A

Must PASS before formal runs:
1 current mode reproduces saved Q2 benchmark <=1e-9.
2 default/no flag remains canonical bit-identical.
3 V1/V2 x_hist input exactly [B,168,7].
4 future-query list exactly12 and all exist in frozen selected222.
5 query list contains no target-day actual and no target_spread.
6 attention keys/values use only 168 legal history timestamps.
7 V2 exact lag-index tests above all PASS.
8 V2 h15-24 lag1 is always masked; no D-1 h15-24 actual read.
9 V1 has no lag branch; V2 has exactly one lag branch.
10 GRU/query/attention receive finite nonzero L_dir gradient.
11 V2 lag branch receives finite nonzero L_dir gradient.
12 Tabular branch, segment heads and fixed fusion remain live.
13 k=8 preserved.
14 output h_time shape [B,24,32] finite.
15 source/config/selector/sequence hashes unchanged.
16 canonical 94 + current full non-destructive suite + focused E5-A tests PASS.
17 E4-A/E4-B/class-weight/structured-decoder experiment switches explicitly inactive.

## 8. Benchmark day

2026-02-13 engineering only.
V0 reuse; V1/V2 fresh.
Do not rank/eliminate.

Record:
- params/trainable params
- runtime/GPU
- best/stop epoch
- gradient ownership
- attention entropy
- attention mass on recent24h / D-1 p1-p14 / same-hour lag positions
- V2 lag-embed norm vs attention-context norm.

## 9. Formal DEV panel

Same frozen 28 days:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13


V0 reuse Q2.
V1 fresh28.
V2 fresh28.
No intermediate stopping.

## 10. Training fixed

- seed=20260924
- k=8
- max_epochs=120
- patience=15
- batch_size=64
- frozen LR/WD
- CUDA+AMP
- Stage B OFF

Only temporal_encoder_mode changes.

## 11. Metrics

Primary:
- Raw
- Balanced
- +Recall
- -Recall
- AUC
- Brier
- MCC
- ppf
- one-class days
- +Recall=0 days
- -Recall=0 days

Cross-window:
W1-W4 Raw/Balanced/MCC/AUC/+R/-R
min-window Raw
window Raw std

Hours:
H1/H2/H3 Raw/Balanced/MCC/AUC/+R/-R/Brier

Paired day-cluster bootstrap 10k seed20260924:
V1-V0
V2-V0
V2-V1
95% CI + W/T/L.

Slot gains overall and H1/H2/H3.

Explicit W3 stress report:
- W3 Raw/Balanced/MCC
- delta vs V0
- D-1 partial-state sensitivity diagnostics
- attention distribution vs W1/W2/W4.

## 12. Safety

Anchor V0/Q2.

SAFETY_ADMISSIBLE iff:
Balanced >= V0-.01
+Recall >= V0-.05
one-class days <= V0+2
+Recall=0 days <= V0+2

Additionally report MCC; Raw gain with MCC collapse is rejected.

## 13. Pre-registered Gate

TEMPORAL_REBUILD_SIGNAL exactly one:

LAG_AWARE_BREAKTHROUGH
- V2 safety-admissible
- V2 Raw gain >=2pp vs V0 OR reaches >=.63
- at least 3/4 windows nonnegative vs V0
- W3 Raw improves by >=2pp
- no MCC/Balanced collapse.

GRU_SEQUENCE_HELPFUL
- V1 materially/stably improves V0
- V2 adds no clear further benefit.

LAG_SKIP_HELPFUL
- V2 clearly exceeds V1 and is safety-admissible, even if overall gain is <2pp.

TEMPORAL_PROMISING_UNPROVEN
- V1/V2 reaches >=.62 or gains >=~1.5pp with safety, but CI/cross-window evidence not established.

CURRENT_TEMPORAL_SUFFICIENT
- neither V1 nor V2 materially improves V0 and both remain close/admissible.

TEMPORAL_REBUILD_HARMFUL
- V1/V2 materially worsen or fail safety.

NO_CLEAR_TEMPORAL_EFFECT
- mixed/noisy result.

## 14. Acceleration rules

If V1 or V2 >=.63 Raw AND safety:
- STOP after E5-A
- immediate 3-seed confirmation V0 + winner
- report whether EVERY window >=.63; only then call stable cross-window63.

If winner >=.62 but <.63 with safety:
- immediate 3-seed confirmation
- if confirmed, E5-B may add causal similar-day memory prior; no more GRU micro-tuning first.

If winner gains >=2pp but stays <.62:
- confirm 3 seeds before adding complexity.

If E5-A fails:
- do NOT tune GRU hidden size/layers/dropout.
- next E5-B = causal recurrent-regime memory prior / similar-day retrieval integrated at representation level.
- no output router, no recency-window sweep.

## 15. Deferred E5-B only

Diagnostic similar-day k=10 is highly complementary to Q2 but k=10 was observed on DEV.
E5-B must therefore avoid pretending k=10 is a frozen production hyperparameter.
Preferred E5-B design:
- causal memory priors at multiple fixed neighborhood scales (e.g. 5/10/20) computed for every training sample using only earlier legal days
- learned representation-level fusion
- final validation on unused dates.

Do NOT execute E5-B in E5-A.

## 16. Forbidden

No LBRM/LLM implementation yet.
No Transformer/PatchTST sweep.
No GRU hidden/layer sweep.
No KNN prior in E5-A.
No feature recovery.
No class weighting.
No structured smoothing.
No postprocess.
No threshold tuning.
No Stage B.
No recency window.
No selector rerun.
No new raw source.
No broader DEV/lockbox.

## 17. Outputs

experiments/first_test/E5_temporal_rebuild/
00_PLAN_SNAPSHOT.md
benchmark/
runs/
daily_metrics.csv
model_summary.csv
window_metrics.csv
hour_segment_metrics.csv
paired_summary.csv
slot_gain_summary.csv
attention_diagnostics.csv
lag_diagnostics.csv
gradient_ownership.json
runtime.csv
E5_A_summary.md
E5_A_GATE.md
figures/

After E5-A STOP for human review.
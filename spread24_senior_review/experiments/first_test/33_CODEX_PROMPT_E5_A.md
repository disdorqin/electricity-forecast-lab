# Codex Prompt — E5-A Temporal Representation Rebuild

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

ONLY authorized task:
Execute experiments/first_test/32_E5_A_TEMPORAL_REBUILD_PLAN.md exactly.

Read first:
1 experiments/first_test/31_E5_ROOT_CAUSE_AUDIT.md
2 experiments/first_test/32_E5_A_TEMPORAL_REBUILD_PLAN.md
3 experiments/first_test/E2_architecture/horizon_specialized_head/E2_E1_summary.md
4 experiments/first_test/E4_feature_recovery/E4_B_summary.md
5 docs/01_业务数据与防泄漏合同.md
6 docs/17_最终模型设计与编码规范.md
7 docs/21_正式运行、验收与实验入口规范.md

Frozen diagnosis:
- valid best Q2 segment_heads Raw=.6086 safety PASS.
- selector recovery, recency windows, postprocess, class-bias tricks failed.
- current TemporalEncoder is shallow flatten-TimeMLP+FFT.
- same legal history supports a trivial previous-same-hour rule Raw=.5699/Bal=.5366 while old TEMPORAL_ONLY is .5000/.4747.
- fundamental similar-day diagnostic reaches Raw=.6042/Bal=.5697 and is weakly correlated with Q2, proving complementary temporal/regime signal exists.
- output-level router/stacking collapses; temporal/regime information must enter representation.

Implement experiment-only temporal_encoder_mode=current/gru_query/gru_query_lag, default current.

Preferred new file:
src/TafM_改进源码/models/temporal_encoder_v2.py

V0 CURRENT_TIMEMLP: reuse Q2.

V1 GRU_QUERY:
- x_hist [B,168,7]
- bidirectional GRU input7, hidden32 per direction -> [B,168,64]
- 24 learned horizon embeddings64
- future query projection from EXACT12 selected legal features listed in plan32
- q = horizon embedding + projected future fundamentals
- dot-product attention over all168 GRU states
- context64 -> Linear64->d_time32 + GELU
- no FFT, no lag branch.

V2 GRU_QUERY_LAG:
same as V1 plus explicit target_spread lag-state branch.

Exact same-hour lag index:
index = 177 + h - 24*d, where h=1..24, d=1..7.
Unit assertions:
h1 d1=154
h14 d1=167
h15 d1=168 unavailable/masked
h24 d1=177 unavailable/masked
h15 d2=144
h24 d2=153
h1 d7=10
h24 d7=33.

Recent D-1 state from x_hist target_spread positions154:168:
last p14, mean last3, mean last6, mean14, std14.
Use normalized values only; no raw target-day truth.
Unavailable lag1 value=0 with explicit missing mask=1.

V2 lag vector -> small MLP ->32.
Concatenate attention context64 + lag_embed32 -> Linear96->32 + GELU.

Exact future query feature list:
fcast_直调负荷
fcast_竞价空间
fcast_新能源总加
fcast_风电总加
fcast_光伏总加
residual_load_renew
bidding_space_ratio
renewable_share
net_ramp_pressure
ramp_tightness
ctx_spread_positive_rate14
spread_same_slot_28d_positive_rate

All must exist in selected222. Fail otherwise.
No target_spread future conditioning and no actual target-day feature.

All other architecture/training fixed exactly to Q2:
dir_only / direction_first / vanilla / segment_heads / full_current / current tabular / current 24h tabular gate / strong_role all / canonical PLE+raw / selected222 / fusion alpha .8 / postprocess none / class weight unweighted / canonical Stage-A / seed20260924 / k8 / max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP / Stage B OFF.

Gate A must prove all 17 items in plan32.
Especially canonical current exact reproduction, exact lag indices, h15-24 lag1 masking, no future actual, gradients, hashes, side experiments inactive.

Benchmark 2026-02-13 engineering only.

Formal panel:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13
Ignore the obvious W2 typo in one explanatory line of plan32; the correct W2 is April12-18.

V0 reuse; V1/V2 fresh28; no intermediate stopping.

Report Raw/Balanced/+R/-R/AUC/Brier/MCC/ppf/collapse; W1-W4; H1-H3; paired bootstrap; slot gains; W3 stress; attention and lag diagnostics; params/runtime.

Safety anchor V0:
Balanced>=V0-.01; +Recall>=V0-.05; one-class<=V0+2; +R=0<=V0+2; reject MCC collapse.

E5_A_GATE TEMPORAL_REBUILD_SIGNAL exactly one:
LAG_AWARE_BREAKTHROUGH / GRU_SEQUENCE_HELPFUL / LAG_SKIP_HELPFUL / TEMPORAL_PROMISING_UNPROVEN / CURRENT_TEMPORAL_SUFFICIENT / TEMPORAL_REBUILD_HARMFUL / NO_CLEAR_TEMPORAL_EFFECT.

Acceleration:
- >=.63 + safety: STOP, next immediate 3-seed V0+winner. Only call stable63 if every W1-W4 >=.63.
- >=.62 <.63 + safety: immediate 3-seed; if confirmed next E5-B causal similar-day memory prior.
- gain>=2pp but <.62: 3-seed before more complexity.
- failure: no GRU hyperparameter sweep; next E5-B representation-level recurrent-regime memory.

Do not implement LBRM/LLM, Transformer, PatchTST, KNN prior, class weight, structured decoder, postprocess, threshold tuning, Stage B, recency, selector rerun or broader DEV in E5-A.

Output under experiments/first_test/E5_temporal_rebuild/ with all artifacts in plan32.
After E5-A STOP.
Do not stage/commit.

Final report only:
1 FILES_CHANGED
2 TESTS
3 TEMPORAL_ROUTING_AUDIT
4 LAG_INDEX_AUDIT
5 BENCHMARK engineering
6 V0/V1/V2 overall
7 W1-W4
8 H1-H3
9 paired comparisons
10 W3 stress
11 attention/lag diagnostics
12 slot gains
13 safety + MCC
14 params/runtime
15 E5_A_GATE TEMPORAL_REBUILD_SIGNAL
16 NEXT acceleration rule
17 unexecuted later work
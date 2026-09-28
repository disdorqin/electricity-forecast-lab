# Codex Prompt — E2-A Architecture Big-Block Ablation

项目根：D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

当前唯一授权任务：执行 E2-A Architecture Big-Block Ablation。

先读：
- experiments/first_test/10_E2_ARCHITECTURE_BIG_BLOCK_PLAN.md
- experiments/first_test/E1_5_signal_audit/E1_5_summary.md
- experiments/first_test/E1_5_signal_audit/E1_5_GATE.md
- experiments/first_test/E1_mini/E1_summary.md
- docs/01_业务数据与防泄漏合同.md
- docs/17_最终模型设计与编码规范.md
- docs/21_正式运行、验收与实验入口规范.md
- docs/22_V2.1方向优先实验路线与V2.2候选设计.md

核心原则：先架构、后训练方式；本轮只拆大块，不调小超参，不运行 Stage B。

固定快速 benchmark day=2026-02-13。它只用于 sanity/convergence/runtime/gradient ownership/reproducibility，严禁用单日效果排名或淘汰架构。

正式 architecture DEV panel：W1 2026-02-12..02-18；W2 2026-04-12..04-18；W3 2026-06-12..06-18；W4 2026-08-07..08-13。

A0 FULL_CURRENT：直接复用 E1 T2 的28天结果；dir_only + direction_first + vanilla + Stage A + current Tabular/Temporal fusion。
A1 TABULAR_ONLY：Direction only H_dir=tab_dir(H_tab)。Temporal 不得影响 Direction output，Direction gradient=0/断开，保留 member-wise TabM。
A2 TEMPORAL_ONLY：Direction only H_dir=time_dir(H_time)。Tabular 不得影响 Direction output，Tabular Direction gradient=0/断开；Temporal 保留合法 future conditioning。

只新增 experiment-only architecture_mode=full_current/tabular_only/temporal_only。canonical formal train默认不变；direction-experiment默认不变；frozen source/selector/target adapter/data contract不动；不要顺带重构。

Gate A 必须证明：
1. architecture_mode=full_current 在 2026-02-13 与旧 E1 T2 exact 或 <=1e-9；
2. tabular_only 的 Temporal Direction gradient=0/断开；
3. temporal_only 的 Tabular Direction gradient=0/断开；
4. finite/shape PASS；
5. 防泄漏边界不变；
6. epf-2 non-destructive suite 94 PASS。

Gate B：在 2026-02-13 跑 A0/A1/A2，只记录，不排名，不淘汰。

Gate C：A1/A2 各完整跑28天；A0复用 E1 T2。不得因中间窗口结果停止。

所有新训练固定：objective=dir_only；checkpoint=direction_first（仅筛查协议）；gradient=vanilla；Stage A；profile=default；seed=20260924；k=8；max_epochs=120；patience=15；batch=64；frozen LR/WD；CUDA+AMP；Stage B OFF。

统一汇总 Overall/W1-W4/H1-H3/paired。至少包含 Raw、Balanced、+R、-R、AUC、Brier、ppf、collapse/zero-recall、min-window、window std。paired A1-A0 和 A2-A0 做 day-cluster bootstrap 95% CI + W/T/L。

记录 architecture audit：active/trainable params、best/stop epoch、wall、GPU memory、gradient ownership、A0 alpha_dir trajectory。

预注册 Gate：
- FULL>TAB 且 FULL>TEMP => SYNERGY / NEXT E2-B-FUSION
- TAB>>TEMP 且 FULL<=TAB => TABULAR_DOMINANT / NEXT E2-B-TAB
- TEMP>>TAB 且 FULL<=TEMP => TEMPORAL_DOMINANT / NEXT E2-B-TEMP
- TAB/TEMP 都有用但 FULL 更差 => FUSION_PROBLEM / NEXT E2-B-FUSION
- 全部弱 => BOTH_WEAK / NEXT FEATURE_WORK
- 其它 => INCONCLUSIVE / REVIEW

本轮禁止：PLE on/off、FFT on/off/bins、hidden/depth/k sweep、alpha grid、concat/late fusion、attention/MMoE/router、threshold tuning、checkpoint tolerance、protected-gradient、Stage B、full Jan-Aug DEV、lockbox。

输出到 experiments/first_test/E2_architecture/，至少包含 00_ARCHITECTURE_PLAN.md、benchmark/、big_block/daily_metrics.csv、model_summary.csv、window_metrics.csv、hour_segment_metrics.csv、paired_deltas.csv、runtime.csv、architecture_audit.csv、E2_A_summary.md、E2_A_GATE.md、figures/、runs/。

完成 E2-A 后必须停止，不执行 E2-B。

最终只汇报：1 FILES_CHANGED；2 TESTS；3 ROUTING_AUDIT；4 BENCHMARK_DAY仅工程；5 A0/A1/A2 Overall；6 W1-W4；7 H1/H2/H3；8 paired A1-A0；9 paired A2-A0；10 architecture/runtime/gradient ownership；11 E2_A_GATE ARCH_SIGNAL；12 NEXT；13 未执行后续项。

禁止 stage/commit。
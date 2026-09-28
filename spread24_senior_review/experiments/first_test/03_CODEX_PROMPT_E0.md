# Codex Prompt — FIRST_TEST E0

你现在接手 spread24_senior_review 的 FIRST_TEST 第一轮性能实验。

项目根目录：
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

实验账本：
experiments\first_test

首先完整阅读并严格服从：
1. docs\01_业务数据与防泄漏合同.md
2. docs\17_最终模型设计与编码规范.md
3. docs\21_正式运行、验收与实验入口规范.md
4. docs\22_V2.1方向优先实验路线与V2.2候选设计.md
5. experiments\first_test\00_EXPERIMENT_PLAN.md
6. experiments\first_test\01_EXPERIMENT_TRACKER.md
7. experiments\first_test\02_RUN_RECORD_TEMPLATE.md

本轮目标不是改模型，而是完成 P0 + E0 性能摸底。

硬边界：
- TARGET=RT-DA。
- forecast origin=D-1 14:00。
- labels only S<=D-2。
- V2.1 canonical 保持冻结。
- 不修改 frozen source / target adapter / selector。
- 不运行 Jan-Aug full DEV。
- 不触碰 lockbox。
- Stage B 不改、不跑。
- 不调参。
- 不把 smoke 当性能结论。
- 不删除失败运行；失败/无效运行保留记录并新建 run。
- 不 stage / commit，除非我之后明确要求。
- 不覆盖已有 outputs。
- 所有新实验必须可追溯到原始 raw_run_dir。

第一步先做安全检查：
- git status --short
- 阅读项目 AGENTS/相关 skill（若存在）
- 运行文档规定的 non-destructive tests
- 确认当前正式 train 默认仍是 joint_v21 + v21_guardrail + vanilla

P0 只做最小必要基础设施：
1. direction-experiment 支持 profile=default，但不得改变正式 train 默认。
2. 增加 experiment-only telemetry：
   wall_time_total_seconds
   training_epoch_seconds_total/mean
   best_epoch
   stop_epoch
   prediction_latency_ms
   parameter_count_total/trainable
   checkpoint_size_bytes
   device
   cuda_peak_memory_bytes（无 GPU 则 null）
3. 为 experiments/first_test 增加最小记录逻辑：
   每个 run 保存 RUN_RECORD.json / RUN_SUMMARY.md / RAW_RUN_PATH.txt。
   原始模型输出仍遵循 outputs/tabm_v21/experiments/direction_first 现有合同。
4. 任何代码改动必须最小化，并补测试；不得重构无关代码。

P0 验收后，严格按顺序执行 E0：
- E0-D1: 2026-02-15
- E0-D2: 2026-04-15
- E0-D3: 2026-06-15
- E0-D4: 2026-08-10

四个 run 固定：
- mode=A2
- train_mode=stage_a
- objective_mode=joint_v21
- checkpoint_policy=direction_first
- gradient_policy=vanilla
- gradient_diagnostics=false
- profile=default
- seed=20260924
- Stage B OFF

然后执行 E0-G：
- target=2026-06-15
- 与 E0-D3 相同
- gradient_diagnostics=true
- batches_per_epoch=2
- 此 run 不用于 runtime baseline

每次 run：
1. 运行前更新 01_EXPERIMENT_TRACKER.md 状态；
2. 运行；
3. 保存 first_test 对应 RUN_RECORD / RUN_SUMMARY / RAW_RUN_PATH；
4. 回填 tracker；
5. 保留失败/INVALID 证据；
6. 不因结果不好临时改超参。

E0 全部完成后生成：
- experiments/first_test/E0_pilot/E0_summary.csv
- experiments/first_test/E0_pilot/E0_runtime.csv
- experiments/first_test/E0_pilot/E0_summary.md
- experiments/first_test/E0_pilot/E0_GATE.md
- experiments/first_test/E0_pilot/E0_curves/ 下的收敛图

汇总至少包含：
- Raw / Balanced / +Recall / -Recall
- Magnitude MAE / skill
- best_epoch / stop_epoch / epochs_run
- wall time / epoch mean,p50,p95 / prediction latency
- total/trainable params / checkpoint size / GPU peak memory
- class-collapse warnings
- E0-G 的 tabular/temporal cosine、conflict rate、norm ratio

Gate 判断必须严格按 00_EXPERIMENT_PLAN.md。
不要自动进入 E1。

完成后只向我汇报：
1. FILES_CHANGED
2. TESTS
3. P0_STATUS
4. E0_RUN_STATUS（5 个 run）
5. E0 核心结果表
6. 参数量/耗时
7. 收敛观察
8. gradient-conflict 观察
9. E0_GATE=GO/HOLD/REVIEW
10. NEXT（只给建议，不执行 E1）

如果环境或时间不允许完成全部 E0：
- 完成已能可靠完成的部分；
- 明确列出未完成项和原因；
- 不伪造结果；
- 不降低实验标准；
- 不进入 E1。

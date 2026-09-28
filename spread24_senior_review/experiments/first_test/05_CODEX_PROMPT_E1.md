# Codex Prompt — FIRST_TEST E1-mini

继续 spread24_senior_review 的 FIRST_TEST。

先完整阅读：
- docs/01_业务数据与防泄漏合同.md
- docs/17_最终模型设计与编码规范.md
- docs/21_正式运行、验收与实验入口规范.md
- docs/22_V2.1方向优先实验路线与V2.2候选设计.md
- experiments/first_test/00_EXPERIMENT_PLAN.md
- experiments/first_test/E0_pilot/E0_summary.md
- experiments/first_test/E0_pilot/E0_GATE.md
- experiments/first_test/04_E1_MINI_PLAN.md
- experiments/first_test/01_EXPERIMENT_TRACKER.md

人工审阅已经授权：
E0_GATE=GO
E1-mini=AUTHORIZED

注意：授权仅限 E1-mini；不授权 protected-gradient、Stage B、full DEV、lockbox。

先做一致性收口：
1. non-destructive tests；
2. 确认 epf-2 / CUDA / AMP；
3. E0_summary.csv 是后续引用的机器事实源；
4. 修正 tracker 中与事实不一致的 checklist（E1尚未运行、full DEV未运行、lockbox未触碰应按实际状态记录），不要改历史结果；
5. 不因 E0 单日结果修改任何模型超参或 checkpoint 规则。

严格执行 experiments/first_test/04_E1_MINI_PLAN.md。

Target days 共 28 天：
- 2026-02-12..2026-02-18
- 2026-04-12..2026-04-18
- 2026-06-12..2026-06-18
- 2026-08-07..2026-08-13

Variant：

M0:
- joint_v21
- v21_guardrail
- vanilla
- A2
- stage_a
- profile=default
- seed=20260924

M1:
- joint_v21
- direction_first
- vanilla
- 其余与 M0 相同

M2:
- dir_only
- direction_first
- vanilla
- 其余与 M1 相同

执行顺序：
A. M0 + M1 全部 28 天
B. M2 全部 28 天

中途禁止根据结果改变后续配置。

所有正式 run 使用 epf-2 CUDA；device=auto 可解析 cuda，AMP 按现有 canonical 路径。
不要为了 GPU 修改 canonical config。

必须记录并聚合：

Direction:
- micro Raw (672 slots)
- macro daily Raw
- Balanced
- +Recall
- -Recall
- predicted positive fraction
- one-class prediction day count
- +Recall=0 day count
- -Recall=0 day count

Secondary:
- Magnitude MAE
- magnitude skill
- AUC
- Brier

Stability:
- W1/W2/W3/W4 分窗口指标

Paired:
- M1-M0 daily deltas
- M2-M1 daily deltas
- win/tie/loss
- day-cluster bootstrap 95% CI for paired daily Raw delta（固定seed；day为重采样单位）

Checkpoint diagnostics:
- selected monitor Raw
- selected monitor Balanced
- selected monitor +Recall/-Recall
- monitor L_dir
- monitor Magnitude MAE
- target collapse warnings

Runtime:
- per-run wall
- epoch count
- best/stop epoch
- prediction latency
- GPU memory

输出：
experiments/first_test/E1_mini/
  daily_metrics.csv
  variant_summary.csv
  window_metrics.csv
  paired_deltas.csv
  checkpoint_diagnostics.csv
  runtime.csv
  E1_summary.md
  E1_GATE.md
  E1_curves_or_figures/

每个 run 继续保存 first_test record，并引用 raw_run_dir，不复制/覆盖历史 checkpoint。

研究纪律：
- same source/sequence/selector/preprocessing
- target=RT-DA
- origin=D-1 14:00
- labels<=D-2
- 不调 threshold
- 不改 loss 权重
- 不改 k/LR/patience/max_epochs
- 不改 selector
- Stage B OFF
- gradient_protection OFF
- 失败/INVALID run 保留
- 不覆盖
- 不 stage/commit

E1 完成后必须停止。

不要运行：
- protected-gradient 正式比较
- Top-2 3 seeds
- Stage B
- Jan-Aug full DEV
- lockbox
- k sweep
- selector tuning

最终只汇报：
1. FILES_CHANGED
2. TESTS
3. E1_RUN_COMPLETENESS
4. M0/M1/M2 主表（micro + macro-day）
5. W1-W4 稳定性表
6. M1-M0 paired result + bootstrap CI
7. M2-M1 paired result + bootstrap CI
8. collapse/zero-recall safety table
9. checkpoint diagnostic
10. runtime
11. E1_GATE：
   - CHECKPOINT_DIRECTION_FIRST = SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
   - NEGATIVE_TRANSFER = SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
12. NEXT，只建议，不执行。

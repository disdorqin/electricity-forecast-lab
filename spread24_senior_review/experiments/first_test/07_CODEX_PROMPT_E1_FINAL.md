继续 spread24_senior_review FIRST_TEST。

当前已完成：
- P0 PASS
- E0 PASS
- E0_GATE=GO

当前唯一授权任务：
执行 E1-mini。

必须先读：
- experiments/first_test/04_E1_MINI_PLAN.md
- experiments/first_test/06_POST_E1_DECISION_TREE.md
- experiments/first_test/E0_pilot/E0_summary.md
- experiments/first_test/E0_pilot/E0_GATE.md
- docs/01_业务数据与防泄漏合同.md
- docs/17_最终模型设计与编码规范.md
- docs/21_正式运行、验收与实验入口规范.md
- docs/22_V2.1方向优先实验路线与V2.2候选设计.md

严格执行 E1-mini：

Target windows：
- 2026-02-12..2026-02-18
- 2026-04-12..2026-04-18
- 2026-06-12..2026-06-18
- 2026-08-07..2026-08-13

共 28 target days。

M0：
- objective=joint_v21
- checkpoint=v21_guardrail
- gradient=vanilla

M1：
- objective=joint_v21
- checkpoint=direction_first
- gradient=vanilla

M2：
- objective=dir_only
- checkpoint=direction_first
- gradient=vanilla

全部固定：
- A2
- Stage A only
- profile=default
- k=8
- seed=20260924
- epf-2
- CUDA
- AMP
- same source/sequence/selector/preprocessing
- target=RT-DA
- origin=D-1 14:00
- labels<=D-2

禁止：
- 改 threshold
- 改 loss 权重
- 改 LR
- 改 patience
- 改 max_epochs
- 改 k
- 改 selector
- Stage B
- protected-gradient
- full DEV
- lockbox
- 根据中间结果改变后续配置

必须输出：
- daily_metrics.csv
- variant_summary.csv
- window_metrics.csv
- paired_deltas.csv
- checkpoint_diagnostics.csv
- runtime.csv
- E1_summary.md
- E1_GATE.md

必须同时报告：
- micro Raw
- macro-day Raw
- Balanced
- +Recall
- -Recall
- collapse days
- +Recall=0 days
- -Recall=0 days
- predicted positive fraction
- Magnitude MAE
- Magnitude skill
- AUC
- Brier

必须做：
- M1-M0 paired daily delta
- M2-M1 paired daily delta
- win/tie/loss
- day-cluster bootstrap 95% CI，day 为重采样单位，固定 seed

E1_GATE 必须分别判断：
- CHECKPOINT_DIRECTION_FIRST = SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE
- NEGATIVE_TRANSFER = SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE

E1 完成后必须停止。

不要自动执行 06_POST_E1_DECISION_TREE.md 中的任何后续分支。
只根据结果指出“下一步应该走 Branch A/B/C 中哪一条”，等待人工批准。

最终汇报：
1. FILES_CHANGED
2. TESTS
3. E1_RUN_COMPLETENESS
4. M0/M1/M2 主表
5. W1-W4 稳定性
6. M1-M0 paired result + bootstrap CI
7. M2-M1 paired result + bootstrap CI
8. safety/collapse table
9. checkpoint diagnostics
10. runtime
11. E1_GATE
12. RECOMMENDED_BRANCH=A/B/C
13. NEXT，仅建议，不执行。

# Codex Prompt — FIRST_TEST E1.5-A Direction Signal Audit

项目：
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

当前状态：
- E0 complete
- E1-mini complete, 84/84 PASS
- CHECKPOINT_DIRECTION_FIRST=NOT_SUPPORTED
- NEGATIVE_TRANSFER=NOT_SUPPORTED
- protected-gradient HOLD
- Stage B HOLD
- full DEV / lockbox HOLD

当前唯一授权任务：
执行 E1.5-A Direction Signal Audit。

首先完整阅读：
1. experiments\first_test\08_E1_5_DIRECTION_SIGNAL_AUDIT_PLAN.md
2. experiments\first_test\E1_mini\E1_summary.md
3. experiments\first_test\E1_mini\E1_GATE.md
4. experiments\first_test\04_E1_MINI_PLAN.md
5. docs\01_业务数据与防泄漏合同.md
6. docs\17_最终模型设计与编码规范.md
7. docs\21_正式运行、验收与实验入口规范.md
8. docs\22_V2.1方向优先实验路线与V2.2候选设计.md

不要重新设计实验，严格执行 08 文档。

第一步：修复 epf-2 环境完整性。
已知：
- 默认 Python non-destructive suite = 94 PASS
- epf-2 缺 shap，导致5个测试模块 collection error
- 默认环境 shap==0.51.0
- requirements.txt 当前未声明 shap

要求：
- epf-2 安装/补齐 shap==0.51.0
- 最小化更新依赖合同，明确 shap
- 不升级无关包
- epf-2 完整 non-destructive suite 必须 94 PASS
- 若达不到，STOP，不跑 E1.5

随后执行同一 E1 28-day signal audit：

W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

复用 E1 TafM artifacts，不重跑：
T0=M0 JOINT+v21_guardrail
T1=M1 JOINT+direction_first
T2=M2 DIR-only+direction_first

新 baseline：

B0 Always Non-positive

B1 Hour-of-day Majority-180
- 每个 target day 只用 [D-181,D-2] labels
- 每小时独立多数类
- 无历史时只允许回退该合法训练窗全局多数，并记录

B2 XGBoost-DIR-180
- 复用 direction-benchmark 当前 XGB 超参/selected features
- recent180d
- D-2
- seed=20260924
- 不调参

B3 XGBoost-DIR-EXPANDING
- 与 B2 完全相同
- 唯一变量=全部合法历史 <=D-2

B4 Strict LightGBM-DIR-180
- 复用 run_strict_lightgbm_baseline(window_days=180)
- direction=prediction_model>0
- rank_auc only; Brier=N/A
- 不改历史LightGBM实现

不要调用现有 direction-benchmark 去无谓重跑 TafM；可复用其中 XGB 逻辑，在 experiments/first_test/E1_5_signal_audit 下建立独立 audit runner。
尽量不修改 src；若必须复用函数，只做最小、可测试改动。

统一输出：
Overall:
Raw, Balanced, +Recall, -Recall, AUC/rank-AUC, Brier, ppf, one-class days, zero-recall days

W1-W4:
Raw, Balanced, +R, -R, ppf, truth prevalence, majority baseline

H1=1-8 / H2=9-16 / H3=17-24:
Raw, Balanced, +R, -R, truth prevalence

Day-level:
Raw, Balanced, +R, -R, ppf, truth positive fraction

额外 signal diagnostics：

1. TafM T0/T1/T2 probability separation：
   mean/median p_positive conditional on true positive/nonpositive
   overall + W1-W4 + H1/H2/H3

2. Oracle threshold diagnostic：
   允许基于已经完成的 E1 predictions 做一次事后诊断。
   输出必须写：
   LEAKY_DIAGNOSTIC_ONLY
   NOT_A_MODEL_RESULT
   NOT_FOR_PROMOTION
   绝不能写回正式模型或作为65%成绩。

3. Prior gap：
   model Raw - same-window majority baseline Raw

比较重点：
- B2 XGB-180 vs B3 XGB-expanding：只解释 recency window
- best baseline vs TafM T2
- 每个窗口/时段是否存在真正增量

阶段目标：
未来模型需要跨月稳定65%+，但本轮不是强行调到65。
如果某模型 overall>=65，必须同时检查每个窗口、Balanced、±Recall、majority baseline。
由多数类坍塌得到的65%不得称为成功。

生成：
experiments/first_test/E1_5_signal_audit/
  00_PLAN_SNAPSHOT.md
  environment_audit.json
  baseline_daily_metrics.csv
  model_summary.csv
  window_metrics.csv
  hour_segment_metrics.csv
  prior_gap.csv
  probability_separation.csv
  oracle_threshold_DIAGNOSTIC_ONLY.csv
  paired_vs_tafm.csv
  E1_5_summary.md
  E1_5_GATE.md
  figures/
  predictions/（如需要）
  audit scripts

防泄漏硬边界：
- target RT-DA
- origin D-1 14:00
- labels <= D-2
- target truth only evaluation
- selector不改
- E1 TafM checkpoint不重选
- oracle threshold不得进入正式预测
- no lockbox
- no full DEV

E1.5 完成后必须停止。
不要自动执行：
- TabM-only / Temporal-only
- TafM recent180训练
- checkpoint equivalence band
- calibration
- threshold tuning
- protected-gradient
- Stage B
- full DEV
- lockbox

最终只汇报：
1. FILES_CHANGED
2. ENVIRONMENT_TESTS
3. E1_5_RUN_COMPLETENESS
4. Overall model table
5. W1-W4 table
6. H1/H2/H3 table
7. B2 vs B3 recency result
8. best baseline vs TafM T2
9. probability separation
10. oracle diagnostic（明确LEAKY）
11. majority/prior gap
12. 是否出现任何合法的65%+候选，以及是否跨窗口稳定
13. E1_5_GATE SIGNAL_SOURCE=
   TREE_STRONG / RECENCY_SIGNAL / TAFM_RANKING_SIGNAL / FEATURE_SIGNAL_LIMITED / MIXED / INCONCLUSIVE
14. NEXT，只建议，不执行

禁止 stage/commit，除非之后明确要求。

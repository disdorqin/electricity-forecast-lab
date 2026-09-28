# E1.5-A — Direction Signal Audit & Strong Baseline Attribution

STATUS=AUTHORIZED_NEXT
DATE=2026-09-25
PARENT=E1_mini
EXPERIMENT_ROOT=experiments/first_test/E1_5_signal_audit
TARGET=RT-DA
FORECAST_ORIGIN=D-1 14:00
LABEL_CUTOFF=S<=D-2

> 本轮目标不是调 TafM，而是定位 Direction signal 到底在哪里。
> E1.5-A 完成后必须停止人工审阅；不自动进入结构改造、checkpoint 新规则、Stage B 或 full DEV。

---

# 0. 上一轮事实冻结

E1-mini 已完成 84/84 runs，4 个 E0↔E1 重叠日逐位复现。

冻结结论：

1. M0 JOINT + V2.1 guardrail:
   Raw=0.5342, Balanced=0.5060, AUC=0.5176。
2. M1 JOINT + strict Direction-first:
   Raw=0.5670, Balanced=0.5180, AUC=0.5531。
3. M2 DIR-only + Direction-first:
   Raw=0.5789, Balanced=0.5273, AUC=0.5663。
4. all-negative baseline Raw=0.6399。
5. M1-M0 Raw +3.27pp，但 paired day-bootstrap CI 跨0；+Recall下降，-Recall显著上升，Mag MAE显著恶化。
6. M2-M1 +1.19pp，所有主要 paired CI 跨0；NEGATIVE_TRANSFER 未得到支持。
7. gradient conflict 存在，但没有转化成 DIR-only 明确优势，因此 protected-gradient 暂停。
8. 事后 oracle-threshold 诊断表明：即使作弊式使用 E1 真值选择阈值，M2 Raw 也仅约64.6%，主要通过近似 all-negative 达成。因此 threshold tuning 不是当前主解。

本轮禁止推翻以上结论或重复无必要的84个 TafM runs。

---

# 1. 研究目标

用户阶段目标：

> Direction Accuracy 如果能够跨月稳定达到 65%+，即视为重要成功里程碑。

但任何 65% 结果必须同时报告：
- Balanced Accuracy
- +Recall
- -Recall
- class prevalence / majority baseline
- one-class / zero-recall warnings

不能用类别不平衡制造“65%成功”。

E1.5-A 要回答：

## Q1
这 28 天在严格 D-1 14:00 / D-2 label boundary 下，传统强模型能否明显超过 TafM 当前 57.9%？

## Q2
如果 XGBoost 强于 TafM，收益来自：
- tree inductive bias；
还是
- recent-180d training window？

## Q3
Direction signal 在哪些月份 / 时段存在或消失？

## Q4
TafM 是：
- ranking signal 不足；
- class prior / calibration 偏移；
- 某些时段/regime失效；
还是
- 与简单模型相比根本没有额外表示收益？

---

# 2. 首先修复正式 GPU 环境完整性

当前事实：

- 默认 Python non-destructive suite = 94 PASS。
- epf-2 因缺少 shap，5个模块 collection error。
- 默认 Python 已安装 shap==0.51.0。
- requirements.txt 当前未声明 shap。

E1.5-A 开始前：

1. 在 epf-2 补齐与项目一致的 shap==0.51.0。
2. 将 shap 明确加入项目依赖合同（最小改动；不得顺带升级其它包）。
3. 在 epf-2 重新执行正式 non-destructive suite。
4. 必须达到 94 PASS 后才允许运行 E1.5-A。
5. 若无法达到，STOP，报告环境问题，不运行实验。

记录：
- Python
- torch
- CUDA
- GPU
- shap
- xgboost
- lightgbm（若由历史代码依赖）
- config/selector/source hashes

---

# 3. 数据与评估窗口完全复用 E1

W1:
2026-02-12 .. 2026-02-18

W2:
2026-04-12 .. 2026-04-18

W3:
2026-06-12 .. 2026-06-18

W4:
2026-08-07 .. 2026-08-13

总计 28 days / 672 slots。

TafM M0/M1/M2：
- 直接读取 E1_mini 已有 artifacts。
- 不重跑。
- 不重新选 checkpoint。
- 不改变 threshold。

---

# 4. E1.5-A 对照模型

## B0 — Always Non-positive

direction_hat = 0 for all slots。

目的：
- 显式 class-prior sanity baseline。
- 必须按 overall / W1-W4 分别报告。

## B1 — Hour-of-day Majority-180

对每个 target day D：
- 只使用 [D-181, D-2] 合法历史标签；
- 按 hour_business=1..24 分组；
- 每个小时预测该小时历史多数方向；
- 若该小时恰好无合法历史，fail closed，不允许偷看未来；可回退为该训练窗全局多数类并记录 fallback。

目的：
- 测试“仅靠小时+近期方向先验”能达到什么水平。
- 不使用任何 target-day truth 进行规则选择。

## B2 — XGBoost-DIR-180

复用当前 direction-benchmark 的严格逻辑：
- target = 1[RT-DA > 0]
- selected future/tabular features
- recent legal 180d
- D-2 cutoff
- random_state=20260924
- 不调参

目的：
- 强 tree classification baseline。
- 与历史 benchmark 口径一致。

## B3 — XGBoost-DIR-EXPANDING

与 B2 完全相同，唯一变量：
- 使用所有合法历史 S<=D-2，而不是 recent180d。

目的：
- 隔离 training-window recency 的影响。
- B2 vs B3 只允许差 training window。

## B4 — Strict LightGBM-DIR-180

复用现有：
run_strict_lightgbm_baseline(..., window_days=180)

方向：
prediction_model > 0

注意：
- 这是 regression-derived sign baseline，不是概率分类器。
- AUC 只能用 rank_auc；Brier=N/A。
- 不修改历史 LightGBM 代码。

---

# 5. TafM 对照

复用：

T0 = E1 M0
JOINT + v21_guardrail

T1 = E1 M1
JOINT + direction_first

T2 = E1 M2
DIR-only + direction_first

当前主要 TafM screening candidate 是 T2，但不得删除 T0/T1，因为它们解释训练策略。

---

# 6. 统一指标

每个模型必须报告：

## Overall
- Raw Direction Accuracy
- Balanced Accuracy
- +Recall
- -Recall
- AUC / rank-AUC（适用时）
- Brier（适用时）
- predicted positive fraction
- one-class days
- +Recall=0 days
- -Recall=0 days

## W1-W4
同上至少：
- Raw
- Balanced
- +Recall
- -Recall
- ppf
- true positive prevalence
- majority baseline Raw

## Hour segment
三个业务段：
- H1 = 1–8
- H2 = 9–16
- H3 = 17–24

至少报告：
- Raw
- Balanced
- +Recall
- -Recall
- true positive prevalence

## Day-level
每个 target day：
- Raw
- Balanced
- +Recall
- -Recall
- ppf
- truth positive fraction

---

# 7. 必须增加的 Signal Diagnostics

基于已有 TafM predictions（不重新训练）：

## 7.1 Probability separation
对 T0/T1/T2：
- mean p_positive | y>0
- mean p_positive | y<=0
- median p_positive | y>0
- median p_positive | y<=0
- separation gap
- 按 W1-W4
- 按 H1/H2/H3

## 7.2 Oracle threshold — DIAGNOSTIC ONLY
允许复现一次 E1 已做的事后 oracle threshold 曲线，仅用于定位。

必须醒目标注：
LEAKY_DIAGNOSTIC_ONLY
NOT_A_MODEL_RESULT
NOT_FOR_PROMOTION

报告：
- threshold=0.5
- oracle Raw threshold
- oracle Balanced threshold
- 对应 Raw/Balanced/+R/-R/ppf

不得把 oracle threshold 写回正式模型或用于后续 target day。

## 7.3 Prior gap
每个 window 比较：
model Raw - window majority baseline Raw

用于判断模型是获得真实增量，还是仅跟随 class prior。

---

# 8. 公平性与防泄漏

严格：

- target=RT-DA
- D-1 14:00 origin
- label cutoff D-2
- target-day truth 只能用于最终 evaluation
- B1/B2/B3/B4 的训练均不得包含 target day / D-1 / future labels
- fixed E1 28 days
- fixed seed
- 不根据 W1-W4 结果修改模型
- 不根据 oracle threshold 修改正式 threshold
- 不修改 selector
- 不重新训练 TafM M0/M1/M2
- 不碰 lockbox

XGB-180 vs XGB-expanding：
唯一允许变化的是 training-window definition。

---

# 9. 65% 里程碑的正式读法

本项目阶段性业务目标：

CROSS_MONTH_65_TARGET

不能只看 overall 65%。

报告必须给：
- overall Raw
- W1/W2/W3/W4 Raw
- min-window Raw
- std across windows
- Balanced / recalls

若出现 overall >=65 但某个月/窗口明显坍塌：
不得称“跨月稳定65+”。

若 Raw>=65 主要来自单一多数类且 Balanced≈0.5 / 某类 recall≈0：
标记：
MAJORITY_DRIVEN
不得称“模型达到有效65+”。

本轮不预设必须达到65才PASS；本轮成功标准是把 signal 来源定位清楚。

---

# 10. E1.5-A 判读矩阵

## Case A — Tree baseline >=65 且明显高于 TafM

结论候选：
MODEL_UTILIZATION_GAP

下一轮优先：
- TabM-only / Temporal-only / fused Direction attribution
- training-window matched TafM（180d vs expanding）
- feature pathway / architecture检查

不先调 threshold。

## Case B — XGB-180 明显优于 XGB-expanding

结论候选：
RECENCY_REGIME_SIGNAL

下一轮优先：
- TafM recent-window / sample-weighting 实验
- 不先改大架构。

## Case C — XGB-expanding >= XGB-180 且 tree baseline > TafM

结论候选：
ARCHITECTURE_OR_OPTIMIZATION_GAP

下一轮优先：
- TabM-only / Temporal-only
- feature/representation attribution
- optimizer/loss trajectory

## Case D — TafM AUC/Brier 明显优于 baselines，但 Raw 转化差

结论候选：
DECISION_LAYER_GAP

下一轮才允许：
- leakage-safe calibration
- checkpoint equivalence band
- safety guardrail

仍禁止用 E1 真值直接调 threshold。

## Case E — 所有非trivial模型都约55–60，AUC接近0.5

结论候选：
FEATURE_SIGNAL_LIMITED

下一轮优先：
- 新的合法 direction features / regime features
- 业务结构研究
- 不再在 checkpoint 上反复微调。

---

# 11. 输出目录

experiments/first_test/E1_5_signal_audit/

必须生成：

- 00_PLAN_SNAPSHOT.md
- environment_audit.json
- baseline_daily_metrics.csv
- model_summary.csv
- window_metrics.csv
- hour_segment_metrics.csv
- prior_gap.csv
- probability_separation.csv
- oracle_threshold_DIAGNOSTIC_ONLY.csv
- paired_vs_tafm.csv
- E1_5_summary.md
- E1_5_GATE.md
- figures/

figures 至少：
- overall Raw/Balanced baseline comparison
- W1-W4 Raw comparison with majority baseline
- H1/H2/H3 Raw/Balanced comparison
- TafM probability separation
- day-level Raw heat/line plot

所有脚本也放在该 experiment 目录。
原始 baseline predictions 可放 predictions/。
不得覆盖 E1_mini。

---

# 12. 本轮结束条件

完成 E1.5-A 后必须停止。

不得自动执行：
- TabM-only / Temporal-only 新结构实验
- training-window TafM 改造
- checkpoint tolerance/equivalence band
- calibration
- threshold tuning
- protected-gradient
- Stage B
- full Jan-Aug DEV
- lockbox

E1_5_GATE 只允许给：
SIGNAL_SOURCE =
TREE_STRONG /
RECENCY_SIGNAL /
TAFM_RANKING_SIGNAL /
FEATURE_SIGNAL_LIMITED /
MIXED /
INCONCLUSIVE

并给下一步建议，不执行。

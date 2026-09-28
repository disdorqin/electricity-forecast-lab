# E5-A Regime Diagnosis 实验设计规范

STATUS=DESIGN_ONLY
PROMOTION_STATUS=NOT_CANONICAL
PARENT=docs/23_E5_regime_aware_framework_reconstruction_plan.md
TARGET=RT-DA direction
ORIGIN=D-1 14:00

## 1. 目的

本阶段不修改模型，不追求提升指标。

唯一问题：

> 当前价差方向预测是否存在稳定的市场状态(regime)，而 V2.1 失败主要来自未建模状态变化？

如果不存在 regime，则停止 E5 路线，回到其他架构假设。
如果存在，则进入 E5-B state embedding。

---

## 2. 防泄漏约束

允许：

- D-1 14:00 已知 future forecast features
- D-1 历史 spread
- 历史训练期统计量
- 历史市场状态标签

禁止：

- 使用 D 日 RT
- 使用目标 spread
- 使用未来真实负荷
- 使用测试月标签参与聚类调整

所有 scaler、cluster fitting 必须遵循 chronological split。

---

## 3. Regime 输入候选

### A. 供需状态

- residual load
- renewable share
- bidding space
- load forecast
- renewable forecast

### B. 波动状态

- ramp pressure
- renewable ramp
- load ramp
- forecast uncertainty proxy

### C. 价差历史状态

- 最近14小时 spread sign
- recent spread magnitude
- same-hour historical spread statistics
- positive-rate statistics

价差历史只使用 D-1 14:00 前信息。

---

## 4. 第一轮诊断方法

不直接选择最优聚类。

并行测试：

1. KMeans K=3,5,7
2. GMM K=3,5
3. HDBSCAN(若环境允许)

记录：

- cluster size
- cluster stability
- positive spread ratio
- direction accuracy
- balanced accuracy
- magnitude MAE
- extreme event ratio

---

## 5. 成功标准

不是要求提升整体准确率。

认为存在可利用 regime 的证据：

- 不同 cluster 的 direction accuracy 差异明显
- 不同 cluster 的正负比例明显不同
- cluster 与 W1-W4 月份变化存在解释关系
- simple regime-conditioned baseline 优于 unconditional baseline

---

## 6. 后续路线

如果 PASS：

进入 E5-B:

Market State Encoder + state embedding 注入 Direction Head。

如果 FAIL：

停止 MoE/regime 路线，重新检查标签定义和信息边界。

---

# Codex执行原则

禁止：

- 修改 src canonical model
- 修改 docs/17
- 修改正式配置
- 重跑完整 DEV

只允许新增：

experiments/first_test/E5_regime/

输出：

- plan snapshot
- cluster assignments
- metrics tables
- diagnostics
- gate report

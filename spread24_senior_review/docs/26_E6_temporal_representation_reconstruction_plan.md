# E6 Temporal Representation Reconstruction Plan

STATUS=DESIGN_PHASE
DATE=2026-09-26
BASELINE=V2.1 Q2 Direction Head
CURRENT_BEST_RAW=60.86%
TARGET=stable_cross_month_63_plus

---

# 1. 本阶段目的

E5-A Regime Diagnosis 未通过，未发现稳定的 market regime signal，因此暂停 State-Aware / MoE 路线。

当前重新定位瓶颈：

> 模型已经拥有历史 spread、未来基本面等信息，但 Temporal 表示学习没有充分提取对方向判断有价值的时间结构。

本阶段不是继续调参，而是重新审查 temporal representation 是否符合 RT-DA spread direction 任务。

---

# 2. 已有证据

## 2.1 Temporal branch 信号利用不足

已有实验：

- Temporal-only 表现接近随机；
- 简单历史规则（same-hour lag、similar-day）能够达到接近当前模型水平；
- 说明历史价格状态存在预测信息，但当前编码方式没有充分利用。

## 2.2 当前模型假设

当前 Temporal Encoder 默认：

168小时历史序列经过统一压缩后，可以自动学习所有周期关系。

但是电价方向任务可能需要显式区分：

- 最近市场记忆；
- 同小时周期；
- 周期性模式；
- 波动状态。

---

# 3. E6核心假设

H1:
RT-DA spread direction 的关键时间信息不是单一长窗口，而是多个尺度时间结构组合。

形式：

```
Temporal Representation

=

Recent Market Memory
+
Hourly Periodicity
+
Volatility State
```

---

# 4. E6-A Temporal Diagnostic（不改模型）

目的：确定不同时间信息源贡献。

建立诊断矩阵：

## Baseline 1

Recent spread state

- D-1 已知14小时spread
- 最近3/6/14小时统计

## Baseline 2

Same-hour historical pattern

- 前1-7日同小时spread

## Baseline 3

Weekly periodic pattern

- 7/14/28天周期信息

## Baseline 4

Current Temporal Encoder output

比较：

- Raw accuracy
- Balanced accuracy
- Positive recall
- Window stability

目标：确认模型缺失哪类时间信息。

---

# 5. E6-B Temporal Encoder 重构候选

仅在E6-A确认后实施。

不直接增加Transformer规模。

候选结构：

```
                 Temporal Representation

        -----------------------------------
        |                |                |
 Recent Encoder   Periodic Encoder   Volatility Encoder
        |                |                |
        -----------------------------------
                         |
                  Direction Head
```

---

# 6. 模块设计原则

保持不变：

- 数据合同；
- D-1 14:00信息边界；
- selector；
- Tabular Encoder；
- Magnitude任务。

允许修改：

- Temporal Encoder；
- Direction representation；
- temporal fusion接口。

---

# 7. 禁止事项

本阶段停止：

- alpha搜索；
- feature无限增加；
- threshold优化；
- class weighting继续调优；
- 直接引入MoE。

原因：这些方法不能解决时间结构表示问题。

---

# 8. 成功标准

第一阶段：

E6-A 找到明确时间信号来源。

第二阶段：

新Temporal Encoder：

- Raw >=63%
- Balanced 不明显下降；
- 正类recall保持；
- 跨窗口稳定。

---

# 9. 实施顺序

Step 1:

完成 E6-A diagnostic。

Step 2:

冻结最佳时间表示设计。

Step 3:

实现 V2.2 Temporal Candidate。

Step 4:

28 DEV days验证。

Step 5:

通过后再考虑更复杂架构。

---

# 10. 研究定位

本阶段不是追求更大的模型。

核心问题：

> 如何让模型理解电力市场价差中的时间依赖结构。


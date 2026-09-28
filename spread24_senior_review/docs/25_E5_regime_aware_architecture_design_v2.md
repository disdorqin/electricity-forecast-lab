# E5 Regime-Aware Architecture Design V2

STATUS=ARCHITECTURE_FREEZE_DRAFT
DATE=2026-09-26
BASELINE=V2.1 Q2 Direction Head
CURRENT_BEST_RAW=60.86%
TARGET=stable_cross_month_63_plus

---

# 1. 本轮改动目的

本轮不是继续优化现有模型，而是解决 V2.1 最大结构假设问题：

> 当前模型假设所有交易日属于同一价格形成机制。

但 RT-DA spread 本质受到供需状态、可再生能源渗透、负荷爬坡、竞价空间等因素影响，存在明显 market regime shift。

已有实验说明：

- fusion alpha 调整无稳定收益；
- feature recovery 未解决跨月瓶颈；
- post calibration 在训练内有效，但跨月失效；
- similar-day 简单方法达到约60%，说明历史状态信息存在；
- Q2 与 similar-day 相关性较低，说明二者学习的是不同信息。

因此下一阶段核心问题不是：

"如何让单模型更强"

而是：

"如何让模型知道当前处于什么市场状态，并选择合适的预测机制。"

---

# 2. 文献依据与设计启发

## 2.1 Regime switching

电力价格预测研究中，市场状态切换通常通过：

- clustering
- hidden regime model
- mixture-of-experts
- adaptive weighting

处理非平稳市场。

核心思想：不同市场状态存在不同价格生成机制。

## 2.2 Similar-day forecasting

电力市场中常利用历史相似日：

- 相似负荷
- 相似新能源水平
- 相似供需状态

寻找具有相同机制的历史样本。

我们的 similar-day 结果支持该方向：历史状态具有独立预测价值。

## 2.3 Mixture of Experts

MoE 的关键不是增加模型规模，而是：

输入状态 -> gate -> 专家选择。

因此适合解决：

同一个预测任务包含多个隐藏规律的问题。

---

# 3. 新总体架构

## V2.2 State-Aware Forecasting

原结构：

```
Tabular Encoder
        \
         Fusion -> Direction Head
        /
Temporal Encoder
```

升级：

```
                 Market State Encoder
                         |
                         v
Tabular Encoder ---- State Fusion ---- Direction Head
                         ^
                         |
Temporal Encoder
```

核心新增：Market State Representation。

不是替代已有信息，而是增加条件变量。

---

# 4. 模块设计

## 4.1 Market State Encoder

输入必须满足 D-1 14:00 可用约束。

禁止：

- 使用目标日真实RT；
- 使用未来spread；
- 使用测试标签统计。

输入候选：

### Supply-demand state

- residual load
- renewable share
- bidding space
- load ramp
- renewable ramp

### Forecast uncertainty state

- historical forecast error
- recent volatility

### Market memory

- D-1 known spread pattern
- recent positive-rate
- recent same-hour behavior

输出：

```
state_embedding
```

维度建议：16~32。

---

# 5. 分阶段实现路线

## E5-A Regime Diagnosis

目的：验证 regime 是否真实存在。

不修改模型。

方法：

KMeans / GMM / HDBSCAN

输出：

- cluster distribution
- cluster spread sign ratio
- cluster accuracy
- cluster MAE
- cluster stability

Gate:

PASS 条件：

不同cluster具有明显性能差异。

---

## E5-B State Embedding

最小架构修改。

新增：

```
market_state_features
          |
   State Encoder
          |
 state embedding
          |
Direction Head
```

保持：

- Tabular Encoder 不变；
- Temporal Encoder 不变；
- Magnitude 不变；
- 数据合同不变。

目的：验证状态信息是否能改善方向判断。

---

## E5-C Lightweight MoE

仅当 E5-B 有效时进行。

结构：

```
                 State Gate
                     |
       -----------------------------
       |             |             |
   Normal Expert  Extreme Expert Transition Expert
```

限制：

- 小专家；
- 不扩大参数规模；
- 不做简单ensemble。

---

# 6. 不允许的方向

本阶段冻结：

- alpha搜索
- threshold tuning
- class weight
- feature无限增加
- 单纯增加hidden size
- 更换大模型

原因：这些已经证明不能解决跨月问题。

---

# 7. 代码改造边界（供Codex）

允许修改：

```
models/
    state_encoder.py      (new)
    dual_branch_v22.py    (new candidate)
    direction_head.py

train.py
run_tabm_v21.py
```

禁止修改：

```
selector
frozen feature contract
data loading
leakage boundary
production default
```

所有实验进入：

```
experiments/first_test/E5_regime/
```

---

# 8. 成功标准

第一阶段：

Raw >=63%

同时要求：

- Balanced不显著下降；
- +Recall保持；
- 四窗口不存在单月崩溃。

最终：

稳定65%+。

---

# 9. 下一步

顺序固定：

1. E5-A regime diagnosis
2. 审核结果
3. E5-B state embedding
4. DEV完整验证
5. 再考虑MoE

本轮目标不是制造一个更复杂模型，而是让模型第一次具备市场状态感知能力。

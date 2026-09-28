# E5 Regime-Aware Framework Reconstruction Plan

STATUS=DESIGN_PHASE
DATE=2026-09-26
BASELINE=Q2_SEGMENT_HEADS
CURRENT_BEST=Raw Direction 60.86%
TARGET=stable_cross_month_63_plus

> 本文不是参数微调计划，而是针对 V2.1 瓶颈进行架构级重构验证。

# 1. 为什么停止微调

已有实验形成一致结论：

- fusion alpha 无稳定收益
- feature recovery 未突破瓶颈
- postprocess/calibrator 跨月失效
- class weighting 可以提高 Raw 但 safety 失败
- decoder smoothing 存在 majority collapse

当前主要问题不是参数量，而是模型假设：

当前模型默认所有历史样本共享同一市场规律。

但 RT-DA spread 存在 regime shift。

# 2. 核心假设

H1:
市场状态漂移限制跨月泛化。

已有证据：

- Q2 在不同月份表现差异明显
- similar-day 基线接近Q2
- 两者预测相关性低，说明存在互补信号

问题不是没有信息，而是不知道何时相信哪类信息。

# 3. 新架构方向

由：

Tabular Expert + Temporal Expert -> Fusion -> Direction

升级为：

Market State Encoder
        |
        v
Tabular Expert -> State Gate <- Temporal Expert
        |
        v
Direction Head

Magnitude保持独立。

# 4. E5-A Regime Diagnosis

先不改模型，验证市场状态是否存在。

输入：

- residual load
- renewable share
- bidding space
- ramp pressure
- forecast error state
- recent spread history

方法：KMeans/GMM/HDBSCAN。

分析：

- 每个cluster正价差比例
- direction accuracy
- MAE
- extreme spread比例

若状态差异明显，再进入架构修改。

# 5. E5-B State Embedding

新增轻量state encoder：

market features -> state embedding -> direction head

目标：

让模型知道今天属于哪个市场环境。

不改变：

- 数据合同
- selector
- 原始feature
- temporal/tabular主体

# 6. E5-C Lightweight MoE

若E5-B有效，再增加轻量专家：

- Normal regime expert
- Extreme spread expert
- Transition expert

由state gate动态选择。

限制：

- 不大幅增加参数
- 不改变预测流程
- 不做暴力ensemble

# 7. 实验纪律

停止：

- alpha微调
- threshold搜索
- postprocess
- class weighting
- feature无限扩展

允许：

- state representation
- routing
- expert specialization

# 8. 成功标准

第一目标：

Raw >=63%

同时：

- Balanced不下降
- 正类recall稳定
- W窗口不崩

第二目标：65%+

# 9. 执行顺序

Step 1: E5-A regime存在性验证

Step 2: state embedding最小改动

Step 3: 通过Gate后28天DEV

Step 4: 再考虑MoE

最终目标：

不是继续优化60.86，而是解决模型无法识别市场状态这一根本限制。

# E4 根因重定位与文献对照（Q2 / E3-A 后）

STATUS=FROZEN_FOR_E4_A
DATE=2026-09-26
PROJECT=spread24_senior_review
PRIMARY_KPI=28-day cross-month Raw Direction Accuracy
TARGET=65% with non-collapse safety

## 1. 当前事实，不再争论

### Q2 是当前最强架构候选
E2-E1 SEGMENT_HEADS:
- Raw = 0.6086
- Balanced = 0.5424
- AUC = 0.5911
- +13 / 672 correct slots vs shared-head Q0
- Safety PASS

结论：1-8 / 9-16 / 17-24 共用一个 Direction decision boundary 确实是一个瓶颈，但不是全部瓶颈。

### E3-A 明确否定“简单缩短到 recent window”
28-day:
- T0/Q2 = 0.6086
- T1 expanding + recent60 = 0.5685
- T2 rolling365 + recent60 = 0.5506
- T3 rolling1095 + recent60 = 0.5357

paired Raw CI:
- T1-T0 = -4.02pp, CI [-7.44,-0.74]
- T2-T0 = -5.80pp, CI [-11.01,-0.74]
- T3-T0 = -7.29pp, CI [-11.90,-2.53]

RECENCY_SIGNAL=NO_RECENCY_GAIN.

因此：
- 不再扫 monitor=30/60/90/120；
- 不再扫 rolling365/730/1095；
- “越近越好”不是当前数据的答案。

## 2. 文献为什么支持我们转向 regime，而不是继续 recency

### A. Lago et al., Applied Energy 2021
Forecasting day-ahead electricity prices: a review/best-practice benchmark.
DOI: 10.1016/j.apenergy.2021.116983

可借鉴：
- 必须用长时间、多市场/多期间和统计检验评价；
- 新模型不能只看单个窗口点估计。
与本项目一致：继续坚持28-day跨4窗口 + paired/day-cluster CI。

### B. Marcos et al., Energies 2020
Short-Term Electricity Price Forecasting with Recurrent Regimes and Structural Breaks.
DOI: 10.3390/en13205452

关键结论：
- electricity dynamics 是 episodic/recurrent；
- 最相关的训练样本不一定是最近样本；
- 可依据 fundamental market regime indicators + structural breaks 选择 calibration data。

与 E3-A 对照：
E3-A 的 recent60 / rolling365 全部显著变差，正说明“简单 recency”不是我们下一步。
正确问题应变成“当前 target 属于什么 regime、模型在这个 regime 下如何修正”。

### C. Maciejowska, Nitka, Weron, Energies 2019
Day-Ahead vs. Intraday—Forecasting the Price Spread to Maximize Economic Benefits.
DOI: 10.3390/en12040631

关键：
- spread 本身可以直接建模；
- spread sign 可以用 ARX / probit 直接预测；
- 不必先分别预测两个市场价格再相减。

与本项目一致：
Y=RT-DA、Direction-first 是合理任务定义，不需要推翻回双价格路线。

### D. Hou & Bunn, The Energy Journal 2024
Daily Episodic and Continuous Arbitrage Trading With Electric Batteries.
DOI: 10.1177/01956574241281143

其 MIP-DAP correction 显式使用：
- lagged price difference
- wind forecast error
- solar forecast error
- load forecast error

同时 DAP 侧使用：
- residual demand
- reserve margin
- wind / solar / load forecasts

含义：
DA→RT 的修正由“基本面预测偏差 + 系统紧张程度”驱动，而不是纯价格历史。

### E. Sun et al., ICPE 2024 / IEEE Xplore 2025
Price Spread Direction Prediction Based on an Improved LSTM Model in China’s Electricity Spot Market.
DOI: 10.1109/ICPE64565.2024.10929104

Shandong wind-farm actual data:
先 feature engineering，再 Multi-Layer LSTM；方向任务本身有实际价值。
不能从摘要推断其准确率可直接与本项目比较，但支持“特征/状态设计优先于继续堆主干”。

### F. Huang et al., Applied Energy 2024
A hybrid framework for day-ahead electricity spot-price forecasting: A case study in China.
DOI: 10.1016/j.apenergy.2024.123863

Shandong:
similar-day construction + XGBoost feature selection + DNN。
支持两个观点：
- 相似状态/相似日是合理结构；
- selector 很重要，但其作用必须在最终神经模型上验证，不能把 selector label 当神经模型真理。

### G. 2026 Shandong regime paper（支持性证据，不作唯一依据）
Risk-Aware Trading Signals for Smart Aggregators...
DOI / journal: Energies 2026, 19(15), 3592.

报告 Shandong RT-DA spread 存在 negative / near-zero / positive 三类 persistent regime，并有多小时持续性。
这与我们“同一Q2在不同物理状态下误差差异巨大”方向一致。
由于论文很新，只作支持性证据，不单独驱动设计。

## 3. 仓库 feature inventory 结论

candidate future features = 259
frozen selector selected = 222
selector-dropped Noise = 37

已 selected 的关键 family：
- forecast-error family: 63 selected / 78 candidates
- spread-state: 33 / 33
- scarcity / bidding-space: 33 / 36
- ramp/pressure: 13 / 17
- uncertainty: 34 / 40
- residual-load / renewable family: 32 / 47

所以错误说法是：
“我们缺风光负荷 forecast-error 特征。”

正确说法：
“这些信息大多已经存在，但 Q2 对不同 market regime 的概率映射不稳定；另外 selector 丢掉的37个Noise中包含若干 regime binary、second-ramp 和 renewable-adjusted error family，后续值得单独验证。”

E4-A 先不重开 selector，也不造新 source feature。
先验证：现有合法 regime information 能否校正 Q2。

## 4. Q2 error slicing（672 slots，diagnostic only）

用 Q2 target-day合法 future features 做四分位切片，未用于训练/调参。

Raw 差异明显：
- residual_load_renew: 0.548 ~ 0.685
- renewable_share: 0.542 ~ 0.738
- bidding_space_ratio: 0.542 ~ 0.738
- net_ramp_pressure: 0.565 ~ 0.673
- err_net_load_28d_std: 0.524 ~ 0.673
- wind uncertainty width: 0.542 ~ 0.697
- solar uncertainty width: 0.560 ~ 0.690

这不是因果证明，但说明：
Q2 的错误明显与 market-state / uncertainty regime 相关，而不是均匀随机。

## 5. 下一步设计判断

停止：
- architecture micro-ablation
- recent-window sweep
- alpha/gate/PLE/k/depth tuning

E4-A 优先验证：
Q2 deep representation + recent independent calibration set + low-capacity regime-conditioned logistic stacker。

目的不是“用更复杂模型”。
目的：
让 Q2 的 base probability 在不同 H1/H2/H3 和 market regime 下拥有不同、可解释、低容量的 decision correction。

如果 E4-A 失败：
E4-B 才重开 selector，重点验证37个被标 Noise 的候选，而不是盲目新增特征。

如果 E4-A >=62% 且 safety：
立即3-seed确认，不再继续微调。

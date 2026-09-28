# E5 根因总审计：为什么模型长期卡在 60% 附近

STATUS=FROZEN_ROOT_CAUSE_AUDIT
DATE=2026-09-26
PRIMARY_VALID_BEST=Q2_SEGMENT_HEADS
PRIMARY_VALID_RAW=0.6086

## 1. 已排除的主因

### 1.1 不是数据泄漏/路由错误
E0-E4 多轮 Gate 已验证：
- forecast origin = D-1 14:00
- labels <= D-2
- frozen source/selector/sequence hashes
- formal defaults fail-closed
- segment-head / feature-profile / split routes均有精确复现和测试。

### 1.2 不是简单增加 TabM/PLE/selector feature 就能解决
已完成：
- Tabular vs Temporal big-block
- fusion alpha coarse map
- Strong/Weak
- horizon gate
- Strong-DIR/MAG role
- PLE+raw/raw-only/PLE-only
- feature recovery selected222 vs literature240 vs all259

均没有稳定 >2pp 的安全提升。
E4-B:
F0 Q2=.6086
F1 literature240=.5923
F2 all259=.6012
Gate=RECOVERY_HARMFUL.

### 1.3 不是 simple recency window
E3-A:
T0=.6086
T1 expanding recent60=.5685
T2 rolling365=.5506
T3 rolling1095=.5357
Gate=NO_RECENCY_GAIN.

### 1.4 不是输出端校正/平滑
E4-A postprocess:
P0=.6086
P1=.5833
P2=.5908
P3=.5848
Gate=SPLIT_COST_DOMINATES.

Diagnostic structured decoder can reach ~.6324 Raw only with severe positive-class collapse (+Recall ~.165, one-class days=19).
sqrt class weighting full28 reaches ~.6161 but safety fails and W4 Balanced ~.459.
因此不能通过 class bias / smoothing 把 Raw 假抬高。

## 2. 当前有效架构发现

### 2.1 Shared 24h head 是真实问题之一
Q2 segment_heads:
Raw .6086 vs shared Q0 .5893
+13/672 correct slots
Safety PASS.
因此 H1/H2/H3 确实需要不同 decision boundary。

### 2.2 但 Q2 不是终点
Q2:
H1 ~.518
H2 ~.692
H3 ~.616
跨时段能力高度不均。
W3 June仍是共同失效区。

## 3. 最大新发现：Temporal representation 欠建模

Current TemporalEncoder:
- input [B,168,7]
- target_spread + 6 system/fundamental channels
- each physical group uses a shallow TimeMLP:
  Linear(168,64)->GELU->Linear(64,24)
- optional fixed FFT-bin residual
- future conditioning only adds tiny scalar eta * matched future feature
- then 7 channels -> Linear(7,d_time)

它没有：
- recurrent state modelling
- temporal convolution
- horizon query/attention over 168 timestamps
- explicit same-hour lag bank
- explicit partial D-1 spread-state skip
- regime-conditioned temporal retrieval

### 3.1 输入里最有价值的信息没有丢
target_spread 是 TEMPORAL_FEATURES 第1通道。
History严格是 D-8 15:00 到 D-1 14:00，共168小时。
因此 D-1 p1-p14真实 spread 已经合法进入 X_hist。
问题不是没喂数据，而是怎么编码。

### 3.2 简单历史规则已经超过 current Temporal-only
同一 frozen 28-day panel，严格合法：
previous-available-same-hour spread sign
Raw=.5699
Balanced=.5366

而 E2-A current TEMPORAL_ONLY:
Raw=.5000
Balanced=.4747
AUC=.4978

因此：
current Temporal branch没有充分提取其输入中已经存在的可用历史结构。

注意：
早期 TEMPORAL_ONLY 使用 shared Direction head，因此它不是对 TemporalEncoder 的最终孤立定罪；但结合 Q2/head结果，它足以触发重新设计 Temporal representation。

## 4. Similar-day / recurrent-regime 诊断

用 target-day 合法未来基本面：
- load
- bidding space
- renewable/wind/solar forecasts
- residual load
- bidding-space ratio
- renewable share
- ramp pressure/tightness

在所有合法历史日里做 day-profile nearest-neighbor。

Diagnostic k=10:
Raw=.6042
Balanced=.5697
W1=.6964
W2=.6012
W3=.4524
W4=.6667

这不是 formal promoted model，因为 k grid 是在 DEV 上诊断出来的。
但它证明：
只用很低容量的 market-state similarity，也能接近 Q2 .6086，并且 Balanced 更好。

## 5. Q2 与 similar-day 的互补性

28-day diagnostic:
probability correlation ≈ .171
hard prediction disagreement ≈ .394
either-expert-correct oracle union ≈ .8036

因此两者不是重复模型。

但是：
fixed convex blend best DEV diagnostic ≈ .6131 Raw；
4-fold window router OOF ≈ .6176 Raw，但 +Recall≈.095，明显 majority collapse。

结论：
互补信息存在，但不能通过末端浅层 router/平均解决。
Regime/temporal information需要更早进入 representation。

## 6. W3 June 共同失效

W3:
target positive prevalence ≈ .357
D-1 p1-p14 positive rate ≈ .469
same-slot 28d positive rate ≈ .368
mean |spread| ≈83.8

Compared with other windows, W3同时表现出：
- recent partial-day sign state与target-day prevalence明显错配
- higher load/residual load regime
- forecast-error volatility elevated
- Q2 / similar-day / recovered-feature arms共同偏弱

这说明单一 persistence、单一 recent state、单一 static regime都不足。
模型需要学习 temporal state transition conditioned on target-day fundamentals。

## 7. 文献对照

### LBRM 2026, Journal of Renewable and Sustainable Energy
DOI: 10.1063/5.0336800
报告 price-spread sign 四季平均 directional accuracy=74.99%，MCC=0.5706。
公开页面显示其核心是：
- segment temporal data into patches
- reprogram temporal patches
- explicitly target high-dimensional temporal dependencies
Benchmark包含 ConvTrans, BiLSTM, CNN-self-attention, LSTM-Attention, PatchTST, iTransformer, Time-LLM.

公开可访问内容没有给出足够细节证明其 forecast origin / label availability 与本项目 D-1 14:00 -> D 24h 完全一致，因此74.99%不能直接当作 apples-to-apples production target。
但它强烈支持：spread-sign任务的主要提升点可以来自更强 temporal representation，而不是仅靠 tabular tuning。

### Sun et al., ICPE 2024
DOI: 10.1109/ICPE64565.2024.10929104
Shandong actual wind-farm data，feature engineering + Multi-Layer LSTM。
公开摘要未提供足够 protocol 细节用于直接数值对比。
支持“temporal sequence modelling + feature engineering”方向。

### Shi & Wang 2024, IJEPES
DOI: 10.1016/j.ijepes.2024.110177
EPF使用 preceding 28-day historical price/load/generation forecast + future generation/load + time variables，并采用 temporal convolution / feature selection。
支持长 receptive field temporal modelling。

### Recurrent-regime literature
Marcos et al., Energies 2020, DOI 10.3390/en13205452
说明最相关历史不必是最近历史；fundamental regime / recurrent episodes值得显式建模。
与 E3-A simple recency失败一致。

## 8. 关于“别人80-90%”

目前检索到的最贴近 price-spread sign 的2026 LBRM公开结果是74.99%，不是“普通模型普遍80-90%”。
同一LBRM论文比较的现代baseline包括BiLSTM/PatchTST/iTransformer/Time-LLM，论文只明确说LBRM比七个baseline平均更高15.23%，公开摘要不足以恢复每个baseline完整数值和同一forecast-origin合同。

因此：
不能把历史96点链路中曾经出现、后来被泄漏审计推翻的89-91%，或其他不同市场/时点/指标的80-90%，当作本严格24点任务的合理基线。

目标65仍值得追，但必须靠真实泛化信号，不允许多数类坍塌或不同任务数字替代。

## 9. 根因排序

当前证据优先级：

P1 — TEMPORAL REPRESENTATION DESIGN
当前168h历史被浅层 flatten-TimeMLP处理，显式时序/lag/regime transition能力不足。

P2 — REGIME-CONDITIONED TEMPORAL TRANSITION
Q2和similar-day互补，但末端融合失败；需要在representation层融合。

P3 — W3/JUNE STATE SHIFT
必须作为架构是否真正跨月的关键压力测试，而不是靠其他月份平均覆盖。

P4 — OUTPUT CLASS BIAS
存在，但 class weighting/smoothing只会抬Raw并损害Balanced，不是根治。

已降级：
selector / PLE / fixed gate / simple recency / postprocess / feature recovery。

## 10. 下一步

E5-A = Temporal Representation Rebuild.

保留：
- Q2 Tabular branch
- H1/H2/H3 segment heads
- canonical selected222
- current 24h gate
- fixed fusion alpha=.8
- Stage-A data/leakage contract

只替换 TemporalEncoder，并显式加入：
1. sequential GRU representation
2. 24 horizon-specific queries
3. target-day legal future fundamental conditioning
4. explicit same-hour lag / recent D-1 spread-state residual

目标：
不是再找0.5pp。
要求测试一个真正能改变 temporal inductive bias 的大结构。

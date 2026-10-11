# 82 — MSR DAILY_ASOF_V1 训练范式与 FAST-24 快速回测协议（2026-10-09）

STATUS=TRAINING_PARADIGM_AUTHORITY
PROJECT=spread24_senior_review
TARGET=Spread = RT - DA
RESOLUTION=96 points/day, 15-minute
FORECAST_ORIGIN=D-1 14:00
LABEL_MATURITY=D-2
TRAINING_MODE=DAILY_EXPANDING_FRESH_FIT
FAST_BACKTEST=FAST24
PRODUCTION_2_5=STRICTLY_READ_ONLY
LATE_SEPTEMBER_FRESH_CONFIRM=SEALED
OCTOBER_LOCKBOX=FORBIDDEN

## 0. Purpose

本文冻结后续 MSR 研究统一使用的训练/回测范式。它不是新的模型结构合同，也不声明 Low、State、High、Z 任一部件已经有效。

当前已有结果已经证明：T24 label decomposition 可保留，但静态 State persistence、静态 PIT calibration 与 28-day rolling location correction 都没有形成稳定的跨期支持。与此同时，现有正式 C1/V1 回顾实验对 DEV/FINAL 使用的是一次训练、跨数月冻结预测，并未模拟真实生产中的逐日更新。

因此，在继续判断各部件价值之前，先统一训练语义：

> 对每个目标日 D，严格模拟 D-1 14:00 的真实签发时点，所有可学习标签最多到 D-2；Low/High 等需要训练的模型按日 fresh fit；所有训练侧上游预测必须来自当时合法生成的 OOF/as-of 预测，不得由未来已知标签回填。

后续 Low/State/High/Z 的价值实验都必须建立在本协议之上，除非新的更高编号协议显式修改。

---

## 1. 当前运行环境与成本基准

当前 MSR-V1 Low/High 均为 LightGBM quantile，不依赖 PyTorch，也不要求 CUDA torch。

当前 `config.LGBM_COMMON` 未启用 LightGBM GPU 参数，因此本轮训练按 CPU 语义执行。若未来晋升到 docs/61 的 TCN/TabM 深度架构，再单独建立 GPU/PyTorch 环境；不得因为当前 `epf-2` 没有 CUDA torch 而改变本轮结论。

在当前机器、`epf-2 / Python 3.11` 环境下，已实测一个目标日：

- 一次性 panel/features/labels 构建：约 `74.1 s`；
- 5 个 Low quantile LightGBM fresh fit + 当天 96 点预测：约 `41.2 s`；
- State 参数计算：约 `0.23 s`。

注意：74.1 s 的 panel/features 构建必须视为可缓存的一次性成本，不得在 FAST-24 中每天重复构建。Low 单日 fresh-fit 的主要成本约为 40–45 s。

历史完整 `C1_HIGH_NO_Z` stage 的量级约 42 s，但它不是严格的“单日 High fresh-fit benchmark”；后续实现 daily High runner 后必须单独记录真实 wall-clock。

---

## 2. DAILY_ASOF_V1：每日训练数学合同

对目标日 D：

- 预测签发时点固定为 `D-1 14:00`；
- 最新完整监督标签固定为 `D-2`；
- 训练历史第一版固定为 expanding history，不搜索窗口长度；
- 每天 fresh fit，固定随机种子，不做 warm-start。

定义：

`T_D = { d | 2023-01-01 <= d <= D-2 }`。

每天训练：

`M_D = Fit(T_D)`，

再仅对 D 的合法特征预测：

`Yhat_D = M_D(X_D^asof)`。

第二天 D+1 重新训练：

`T_{D+1} = { d | 2023-01-01 <= d <= D-1 }`。

因此新增成熟标签会逐日进入模型，但第一轮不同时引入 recency weighting / 180d / 365d rolling window / exponential decay，以免把“更新频率”和“历史记忆长度”混成一个变量。

### 2.1 Fresh fit contract

第一版必须：

- 每个目标日重新构建模型对象并 fit；
- 固定 seed；
- 不从前一天 booster warm-start；
- 不用未来日期选择 early-stopping epoch、模型参数或阈值；
- 模型训练截止必须在 artifact 中显式记录。

Warm-start 以后只能作为工程加速实验，不能在 DAILY_ASOF_V1 科学基线里默认开启。

---

## 3. 一次缓存、逐日切片

为了把 FAST-24 控制在约 20 min 级别，必须把“不会因目标日变化而重新拟合”的 deterministic 数据准备与“必须每天重新 fit”的统计学习过程分离。

允许一次性缓存：

- canonical source 的只读投影；
- T24 historical `L_true/H_true` label construction；
- deterministic、严格 causal 的基础 feature panel；
- calendar / lag / forecast-derived feature values；
- source/provenance hashes。

禁止全局预拟合并跨日复用：

- 任何使用标签估计的 scaler/calibrator/threshold；
- Low/High/Z model parameters；
- State `delta_t`；
- State transition matrix（若后续 persistence 被重新启用）；
- validation-selected threshold；
- 任何需要利用 D 之后数据才能得到的统计量。

如果未来某个新模型引入标准化、编码器、target transform 或 feature selection，该步骤必须在每个 D 的 `T_D` 内拟合，不能因为 panel 被缓存就全局拟合。

---

## 4. Low daily contract

对每个 D：

1. 读取已缓存的 deterministic panel；
2. 训练行严格切到 `target_day <= D-2`；
3. 使用当前冻结的 Low features 与五个 quantile `{.10,.25,.50,.75,.90}`；
4. fresh fit 五个 LightGBM quantile models；
5. 只预测 D 的 96 个 slot；
6. 保存当天 quantile ladder 与模型/训练截止 provenance。

第一轮训练范式验证中不使用 docs/78/80 的 PIT 或 rolling location correction；先直接测 DAILY_EXPANDING 是否已经吸收跨季节/跨月漂移。

---

## 5. State daily contract

在训练范式验证阶段：

- `kappa=0.5` 冻结；
- persistence 固定 `lambda_state=0`；
- 不重新搜索 Markov strength；
- `delta_D,t = 0.5 * IQR(L_true_{d,t}, d in T_D)`，按 D 的合法成熟历史逐日重估；
- State probability 只由当天 fresh Low quantiles 产生。

目的：先区分“State 本身/Low 表达问题”与“静态半年冻结参数问题”。

若后续部件价值实验重新评估 persistence，transition matrix 也必须只由 `T_D` 内成熟标签估计，并作为独立科学变量重新授权。

---

## 6. High / Z 的严格 as-of 上游合同

这是后续全链实验最重要的防泄漏规则。

High 在训练历史某一天 d 上使用的 Low/State/Z 特征，必须是对 d 当时合法产生的内部预测：

- chronological OOF prediction；或
- 已封存的 historical as-of prediction ledger。

绝对禁止：

- 用 `L_true_d` 生成 perfect State 后作为 High feature；
- 在已经知道 d 标签以后重新 fit Low，再把 in-sample `Q_L(d)` 回填成 High 的历史 feature；
- 将 true Z、true State、target-day actual、target-day RT/DA/Spread 作为 deployable High feature。

对于目标日 D，High 只能使用：

- `X_fast_D^asof`；
- 当天 fresh Low 的预测量；
- 当天合法 State prediction；
- 若某实验授权 Z，则只能使用当天 predicted Z；
- 历史训练行中的上游变量必须来自 OOF/as-of ledger。

### 6.1 As-of ledger

后续 runner 必须建立 append-only research ledger，至少包含：

- `target_day`, `slot`；
- `forecast_origin`；
- `max_label_day_used`；
- Low quantiles；
- State probabilities / semantics；
- optional predicted Z；
- source/config/code/model hash；
- upstream prediction hash；
- generated timestamp；
- whether the row is OOF / as-of / fitted（fitted rows不得进入 High 训练 feature）。

对历史训练期可继续使用已经通过 chronology audit 的 OOF artifact；新逐日回测产生的 as-of predictions 在标签成熟后才可以进入后续 High 训练。

---

## 7. FAST-24 固定研发压力测试集

FAST-24 用于每次训练方式/部件代码修改后的快速方向判定。它是 consumed development/retrospective diagnostic set，不是 fresh confirm，也不产生独立泛化 claim。

固定日期如下，今后不得根据结果替换：

- 2026-01-05, 2026-01-15, 2026-01-25
- 2026-02-05, 2026-02-15, 2026-02-25
- 2026-03-05, 2026-03-15, 2026-03-31
- 2026-04-01, 2026-04-15, 2026-04-25
- 2026-05-05, 2026-05-15, 2026-05-25
- 2026-06-05, 2026-06-15, 2026-06-25
- 2026-07-05, 2026-07-15, 2026-07-25
- 2026-08-03, 2026-08-08, 2026-08-13

设计原则：

- Jan–Jul 每月 3 天，覆盖月初/中/末；
- 特意保留 2026-03-31 / 2026-04-01 的 regime reversal 边界；
- August 仅使用 consumed/final 合法范围 2026-08-14 之前的日期；
- 不触碰 2026-09-16..09-30 fresh confirmation；
- October 全程禁止。

FAST-24 主要用于“杀死错误方向/筛掉无增量部件”，不能因为 24 天平均表现好就直接宣布正式跨月泛化。

---

## 8. 三层实验预算

### Level A — FAST24_LOW_STATE

每次 Low/State/训练协议修改必须先跑。

目标 wall-clock：约 20 min，软上限约 25 min。

若超过，优先优化：

- panel cache；
- date slicing；
- artifact I/O；
- parallelism（不改变模型语义）。

不得首先减少固定日期数来掩盖低效实现。

### Level B — FAST24_FULL_V1A

只有 Level A 显示明确 headroom 才跑：

`Low -> raw State(lambda=0) -> High V1a(no-Z) -> reconstruction`。

允许超过 20 min；必须记录真实 Low/High 分项 wall-clock。V1b/Z 不默认运行。

### Level C — Extended consumed backtest

只有 FAST-24 支持候选后才允许运行更多日期/连续区间甚至完整 daily backtest。用于月度稳定性和 transition/regime shift 验证。

### Level D — Fresh confirm

架构、训练范式、参数、阈值、部件全部冻结后，且 retrospective readiness 通过，才允许对 2026-09-16..09-30 做 label-masked prediction seal，再一次性开 truth。

October 不在本协议任何执行范围。

---

## 9. 第一项训练范式实验：FROZEN vs DAILY_EXPANDING

在开展部件价值实验之前，先建立统一基线。

对 FAST-24 比较：

- `FROZEN_20251230`：历史 C1/V1 语义，一次 fit 后跨期应用；
- `DAILY_EXPANDING_D2`：本文定义的逐日 fresh fit。

第一阶段只比较 Low + raw State (`lambda=0`)；不跑 High、不跑 Z、不做 rolling correction/PIT。

必须输出：

Low：
- median MAE；
- multi-quantile pinball；
- q10/q25/q50/q75/q90 empirical coverage；
- q50 coverage error；
- Low residual mean/median；
- 每月/每个 FAST 日期明细。

State：
- multiclass Brier；
- Balanced Accuracy；
- NEG/NEU/POS recall；
- truth/predicted prevalence；
- State entropy / mean max probability；
- `delta_D` 的逐日变化。

Runtime：
- feature-cache build；
- Low fit；
- State calculation；
- per-day total；
- FAST-24 total。

该实验不是为了强行证明 daily 一定更好，而是裁决“半年 frozen training 是否是当前主要误差来源”。

---

## 10. 训练范式实验的解释规则

不设置为了追求 PASS 而任意微调的硬阈值，但必须按以下逻辑解释：

### DAILY clearly useful

若 Daily expanding 同时表现出多项稳定改善，例如：

- Low MAE/pinball 明显降低；
- q50/quantile coverage 跨月摆动明显收窄；
- State Brier / Balanced / POS recall 有一致改善；
- March/April 已知 drift/reversal 不再依赖后处理补偿；

则后续所有部件实验以 DAILY_ASOF_V1 为统一训练底座。

### DAILY weak / negligible

若主要指标仅有很小变化、方向不一致，或者改善仅来自 class prevalence 平移而非可分性提升，则不得继续把“未及时更新”当作主要解释。

下一步应进入：

1. Low/State separability probe；
2. 训练记忆设计（recency weighting / rolling history）与架构问题二选一；
3. 根据 probe 决定是否重构 Low/State，而不是继续增加后处理补丁。

### DAILY harmful

若 Daily expanding 在多个季节/月份系统恶化，则必须检查：

- expanding history 新标签是否只是加噪声；
- LightGBM fixed hyperparameters 是否随样本规模失配；
- feature/label concept drift；
- 训练范式是否需要 recent weighting。

不得自动搜索大量 rolling-window 长度。

---

## 11. 后续部件价值实验原则

本协议只冻结训练范式。后续部件价值实验必须遵守“部件淘汰制”：

- Low：先判断现有合法 X 是否有足够 Low/State 可分信息；
- State：比较 `Q_L -> State` 与诊断性的 direct State probe，并使用 oracle State 评估 theoretical headroom；
- High：比较无 State / predicted State / oracle State 条件下的 High；
- Z：比较 no-Z / predicted-Z / oracle-Z，但 oracle 只能做机制诊断；
- V2：只有某部件存在明确 oracle/headroom 且 V1 表达不足时，才允许晋升 docs/61 中更复杂结构。

任何部件若 oracle 也无增量，应优先删除，而不是继续调参。

任何局部改动前都必须问：

> 即使这个部件被完美修复，理论上还能给最终 Spread / Direction 带来多少收益？

如果 headroom 很小，应停止投入。

---

## 12. 当前硬边界

- Production `electricity_forecast_model2.5` 严格只读；
- 不写生产数据库；
- 当前 canonical source SHA 不变；
- T24 decomposition/operator 不重选；
- 2026-09-16..09-30 保持 sealed；
- 2026-10 全部禁止；
- 不执行 Git stage/commit/push；
- FAST-24 是研发压力测试，不得冒充 independent holdout。

---

## 13. 下一执行动作

本文件落盘后，下一步先实现 `DAILY_ASOF_V1 + FAST24` runner 和对应 leakage/runtime tests，再执行 `FROZEN_20251230 vs DAILY_EXPANDING_D2` 的 Low+State 训练范式裁决。

只有训练范式裁决完成后，才进入 Low / State / High / Z 的逐部件价值实验。

预期下一份科学文档应记录：

- daily runner 工程审计；
- FAST-24 训练范式结果；
- 是否正式把 DAILY_ASOF_V1 升格为后续统一科研/生产训练语义；
- 下一部件实验的唯一授权范围。

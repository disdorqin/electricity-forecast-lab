# 58 — Spread Literature Review Ledger

STATUS=ACTIVE_CANONICAL_REVIEW_LEDGER
STARTED=2026-10-06
SIGN_CONVENTION=project uses S = RT - DA
PURPOSE=Route A / Route B 论文候选池、精读、综述素材

> 本文件只保留对 DA / RT / Spread 研究路线真正有用的内容。完整规则见 `docs/57_spread_literature_research_protocol_20261006.md`。

## 0. 当前结论：先建立“小而强”的论文池

现有旧 ledger 中真正应该迁移到本 A/B 文献库的论文并不多。

当前最有价值的起点是：

- Route A（2026-10-07 corrected）：只保留 **DA / RT / Spread 结构约束联合预测**。单独 DA leg、单独 RT leg、普通概率后处理不再算 A 主线。A 的目标是利用 `S = RT - DA`、DA→RT 传导、共享基本面、forecast-error shock、协整/联合分布等关系约束 Spread 预测。当前确认：这种严格 A 文献明显少于 direct-spread 文献，不为凑数量纳入边缘论文。
- Route B：已经有几篇非常贴合用户偏好的统计/传统方法论文，尤其 Energy Economics 2025 的 covariate-dependent mixture model，应作为 B 路线第一优先级精读对象。
- 同市场山东论文非常有价值，但必须把“15 min 分辨率相同”与“forecast horizon 相同”分开。近目标时点使用 lagged spread 的高指标不能直接对标 D-1 14:00 全次日预测。

---

## 0.5 山东 Spread 真实数据诊断先验（2026-10-07）

本节用于约束后续文献筛选：不再泛泛寻找“能预测价差”的方法，而优先针对本地数据里已经确认的困难。

### 当前固定实践窗（2026-08-01..2026-08-14，96 点）
- 正价差比例仅约 `17.78%`（239 / 1344）；负价差约 `82.22%`。
- 正价差多数幅值很小：median `+5.16`，mean `+17.57`，90% 分位约 `+47.87`；239 个正样本中 116 个不超过 `+5`，186 个不超过 `+20`。
- 负价差明显更重尾：median `-32.68`，mean `-169.59`；负向绝对值 90% 分位约 `735.13`。
- 因而普通回归 MAE/RMSE 会被大幅负价差主导，容易学成“整体偏负”的数值模型；这会改善幅值却伤害 `+Recall`。

### Baseline / A02 对正价差的具体误判
- LightGBM baseline 在 239 个真实正样本中只命中 28 个，`+Recall=11.72%`；211 个被错判为负。
- baseline 在真实正样本上的预测均值约 `-109.74`、中位数约 `-122.33`，而真值均值仅 `+17.57`；FALSE-NEGATIVE 的平均预测误差约 `-146.61`。
- A02（DA Buffer + Balancing Pressure）虽然把真实正样本上的预测均值拉近到 `-94.77`，MAE 从约 `131.88` 降到 `112.40`，但其正样本预测 90% 分位仍为 `-18.54`，几乎不跨零，最终只命中 3 个正样本，`+Recall=1.26%`。
- 说明 A02 的结构特征对“幅值收缩/排序”有信号，却没有学到“什么时候应该跨过 0 进入正价差 regime”。

### 正价差具有明显的事件簇，而非均匀随机出现
固定测试窗中，正样本高度集中在少数日期：
- 2026-08-02：31 / 96
- 2026-08-03：45 / 96
- 2026-08-10：46 / 96
- 2026-08-11：49 / 96
- 2026-08-12：25 / 96
- 2026-08-14：24 / 96
其余多天只有 0–5 个正样本。baseline 对 8/2、8/3、8/11、8/12 几乎完全未识别，表明问题更像 day/regime detection，而不只是一个全局 threshold。

### 月份 regime（2026 年）
正价差率出现明显阶段切换：Jan `52.6%`、Feb `42.9%`、Mar `60.7%`、Apr `49.9%`，随后 May `29.0%`、Jun `37.8%`、Jul `34.3%`、Aug `32.0%`、Sep `28.3%`。平均 Spread 也从 Mar `+11.5` 转为 Jul `-75.5`、Aug `-79.5`、Sep `-89.9`。
这意味着存在强 concept/regime drift；固定的“全年同一正负先验”不够。

### 日内规律
跨多年汇总，`9-16` 时段同时具有最高的正向大尖峰率和负向大尖峰率，是最难、最重尾的区域；`17-24` 整体更偏负。
2026 年小时规律更清楚：
- 12–17 点相对容易出现正价差，17 点正价差率最高约 `56.3%`；
- 5–7 点偏负，正价差率约 `32.8% / 29.1% / 26.5%`；
- 19–22 点明显偏负，均值约 `-70.9 / -84.9 / -83.7 / -86.1`，正价差率仅约 `31.7%–36.0%`。
- 月份×时段交互很强：例如 2026-01 的 9–16 点正价差率 `61.8%`，到 2026-05 降到 `16.9%`；2026-08 的 9–16 点仍有 `45.7%`，但 1–8 和 17–24 均只有约 `25%`。

### 对后续 A 路线文献的新增筛选重点
优先寻找能显式处理以下问题的强论文：
1. `regime / state transition`：先判断 DA→RT correction 属于 upward / downward / neutral 哪个状态，再预测幅值；
2. `asymmetric / hurdle / two-stage`：符号与幅值分开建模，避免 MAE 被负向厚尾支配；
3. `time-varying prior / seasonal interaction`：允许 month × hour × market state 的先验随时间变化；
4. `tail-aware structural loss`：不能只优化平均点误差，同时保留 positive-event separability；
5. `DA/RT joint structure`：A 路线仍必须显式利用 DA、RT、Spread 关系，而不是退化成普通 direct-spread 分类器。

---

## 0.6 多尺度 + 状态持续研究转向（2026-10-07 晚）

用户明确提出：方向/幅值解耦此前已经尝试过且效果一般，因此降级为候选，不再默认作为主架构。新的主问题是：

1. `long/short fluctuation decomposition`：将 Spread 拆为较慢的基态/低频成分与较快的局部修正/高频成分，再分别建模；
2. `state persistence`：允许幅值在相邻点间剧烈变化，但不希望模型在缺乏强证据时频繁出现 `+ -> - -> +` 的不真实符号抖动；
3. `B route reopened`：允许直接建模 Spread，但仍要求机制清晰、统计结构明确，并优先服务于上述两类问题。

当前优先文献组：
- Kapoor et al., Journal of Forecasting 2023：Markov regime switching / time-varying transition，负责“状态持续性”；
- 山东 15min Spread 三态 Markov 论文（Energies 2026）：作为本地市场存在 multi-hour persistence 的机制证据，不作为强刊锚；
- `A Dynamic Decomposition-Based Modeling Approach for Electricity Spot Price Forecasting`（2025，山西现货）：rolling wavelet + LGBM，负责“因果多尺度分解”；
- Lu, Li & Wang, `Electricity price forecasting: a transparently structured approach via adaptive heterogeneous model selection and real-time dynamic integration`, Electric Power Systems Research 261 (2026), 113506, DOI 10.1016/j.epsr.2026.113506：STL 将价格显式拆成 trend / seasonal / residual，并通过 Kalman-filter dynamic integration 在线调整各成分贡献，作为“长短成分分开预测再动态重构”的近期主参考。

待融合的候选结构不是简单 `sign -> magnitude`，而是优先考虑：`slow baseline + fast residual + persistent latent regime + tail/noise`。其中 decomposition 必须 rolling/causal，禁止全样本分解导致未来泄露。

---

## 1. Route A 候选池

> **CORRECTION 2026-10-07**：前一轮将“分别预测 DA/RT”“IDR/conformal 概率后处理”等过宽地归入 A，这是错误分类。以下重新按用户定义筛选。
>
> **A 主线必要条件**：方法必须显式利用 DA、RT、Spread 之间的结构关系；理想形式为 joint / constrained / conditional / error-correction / cointegration / structural regression，而不是只训练两个互不相关的预测器。
>
> 当前核心层级：
> 1. `A-CORE-PREDICTION`：真正联合预测/联合概率并导出 Spread；
> 2. `A-STRUCTURAL-FOUNDATION`：不是最终预测器，但明确推导 DA→RT divergence 的结构方程，可用于我们自己构造约束；
> 3. `A-AUXILIARY`：单腿或普通概率方法，仅作 baseline，不再代表 A。


### [A01] Li et al. — IEEE Transactions on Smart Grid, 2026

**Title**: Probability Density Forecasting of Electricity Price Difference in the Spot Market  
**Authors**: Zhenghui Li, Kangping Li, Chunyi Huang, Mahmud Fotuhi-Firuzabad  
**Venue**: IEEE Transactions on Smart Grid, 17(3), 2287–2299  
**Published**: 2026-01-12  
**DOI**: 10.1109/TSG.2026.3651988  
**Status**: peer-reviewed  
**Classification**: `A2 + 概率预测 / joint density -> Spread density`  
**Priority**: A-P0 for route definition; A-P1 for user method preference  
**Review status**: `READ_AND_ACCEPTED_AS_CORE_ROUTE_A_REFERENCE`

**Why A**:
论文先分别预测 DA 与 RT 的概率密度，再用联合概率密度 / Sklar theorem 描述依赖关系，最后通过积分关系把联合密度投影到 price-difference density。因此 spread 是由 DA、RT 分布派生得到，而不是直接把 spread 当监督目标。

**Verified core facts**:
- IEEE 官方元数据确认发表信息与 DOI；
- 方法包含 skewed DeepAR 分别预测 DA / RT density；
- 再构造 joint density；
- 数值积分 / conditional density projection 得到 price-difference density；
- PJM case study 报告 overall probabilistic forecasting performance 至少提升 10.15%。

**Why important**:
这是路线 A 最直接的概率版理论模板：它明确处理“DA 与 RT 高相关，分别预测后直接相减可能放大误差”的问题，并把二者依赖结构放进联合密度。

**Caveat**:
方法核心包含 DeepAR，不符合我们“优先传统/统计、显式约束”的首选风格。因此它更适合作为 Route A 的强基准 / 概率理论参考，而不是当前首选复现方法。

**Review takeaway**:
已完成精读。用户认可其“从 DA/RT 强关联与联合分布问题出发，再数学推导 Spread 分布”的研究范式；保留为严格 Route A 的核心参考，但不把 DeepAR 架构本身当作首选创新方向。

---

### [A02] Ghiassi-Farrokhfal et al. — Manufacturing & Service Operations Management, 2023

**Title**: Optimal Electricity Imbalance Pricing for the Emerging Penetration of Renewable and Low-Cost Generators  
**Authors**: Yashar Ghiassi-Farrokhfal, Rodrigo Belo, Mohammad Reza Hesamzadeh, Derek Bunn  
**Venue**: Manufacturing & Service Operations Management, 25(5), 1966–1983  
**Published online**: 2023-07-13  
**DOI**: 10.1287/msom.2021.0555  
**Status**: peer-reviewed  
**Classification**: `A-STRUCTURAL / two-stage imbalance pricing -> DA-buffer + balancing-pressure interpretation`  
**Review status**: `READ_AND_PRACTICED_ON_SHANDONG_R96`

**Methodology retained**:
1. 将市场写成两阶段：Stage 1 的日前承诺/低成本供给决定第一阶段价格锚，Stage 2 的高成本实时平衡调整决定实时价格修正；价格不是外生标签，而是清算结构的结果。
2. 解析比较 single pricing / typical dual pricing / renewable-based dual pricing，在低成本/新能源市场份额、预测不确定性、竞争者数量等参数空间中推导不同机制适用的 regime，而不是只做经验模型比较。
3. 进一步反推 optimal pricing：论文 Definition 4 的核心形式为 `p2,op^i = p1 + alpha2*H2 - alpha1*L1^i`，并证明该定价在其两阶段模型下可使 LCG 选择 `d_i=0`；确定性情形达到 full economic efficiency。对本项目最重要的启发是把 `RT-DA` 分解成“实时 balancing pressure - 日前低成本供给/承诺 buffer”的结构。

**Shandong R96 local adaptation**:
不是逐式复现论文市场设计，而是在固定 D-1 14:00 信息边界下，把可部署字段组织成两个 latent block：
- `DA Buffer`：目标日直调负荷、外电、风电、光伏、核电、地方电厂、自备/试验机组等 forecast，低成本/总承诺供给 share、residual demand、正负备用比例；
- `Balancing Pressure`：D-1 p1-p56 `actual-forecast` 的 net surprise、14:00 boundary actual state、目标日 net-pressure transition、up/down reserve-normalized pressure。
固定模型为 LightGBM；Baseline 62 个合法/实验性可见特征，A02 在其上增加 20 个结构特征。训练至 2026-07-31，测试 2026-08-01..2026-08-14，共 153,312 / 1,344 个 15-min 样本。历史 vintage 口径明确为 `PROXY_EXPERIMENTAL_NOT_LIVE_SNAPSHOT`。

**Local result — primary LightGBM**:
- R96 Spread MAE `162.362 -> 141.763`，改善 **12.69%**；
- R96 RMSE `237.009 -> 217.011`，改善 **8.44%**；
- R96 AUC `0.5441 -> 0.5532`；
- 但 Balanced accuracy `54.00% -> 49.22%`，+Recall `11.72% -> 1.26%`，Direction `81.25% -> 80.13%`；
- 24点聚合 MAE 改善 **13.18%**、RMSE 改善 **10.42%**、AUC `0.5905 -> 0.6198`，但 +Recall 降到 `0%`。
- 论文结构特征约占 LightGBM split importance 的 `22.0%`，并非完全被模型忽略。

**Robustness / diagnosis**:
- 按论文解析式方向对关键结构变量施加 LightGBM monotonic constraints 后，R96 AUC 比 baseline 提高约 `+0.0357`，MAE 改善 `6.75%`，但 Balanced / +Recall 仍下降；
- XGBoost 同样得到 RMSE 改善 `5.02%`、AUC `+0.0276`，但 Balanced accuracy 下降约 `0.98pp`。因此“结构有 magnitude/ranking signal”具有一定跨树模型一致性，但尚不是 direction solution。

**Verdict**: `PARTIAL_SIGNAL_NOT_ACCEPTED_AS_DIRECTION_METHOD`

**Project implication**:
保留 `DA Buffer / Balancing Pressure / reserve-normalized up-down pressure` 作为解释性状态；不能宣称该适配解决价差方向。它的重要实验教训是：**一个结构可能真实改善幅值误差和排序能力，却同时恶化正负方向平衡**。以后每篇论文必须分别报告 magnitude、AUC/separability、Balanced、+Recall、-Recall，不能只看 headline accuracy。

**Local evidence**:
- source: `experiments/literature_method_lab/r96_data.py`, `r96_run.py`
- result: `experiments/literature_method_lab/results/A02_MSOM_2023_R96_20261007.json`
- cumulative scorecard: `experiments/literature_method_lab/METHOD_SCORECARD.md`

---

### [A03] Guo et al. — Applied Energy, 2023

**Title**: Incorporate Day-Ahead Robustness and Real-Time Incentives for Electricity Market Design  
**Venue**: Applied Energy, Vol. 332, 120484  
**DOI**: 10.1016/j.apenergy.2022.120484  
**Status**: peer-reviewed  
**Classification**: `A-STRUCTURAL / DA robust decision -> RT reference -> RT balancing`  
**Review status**: `READ_AND_LOCALLY_ADAPTED`

**Technical structure learned**:
1. DA stage is a bilevel market problem: an upper decision-maker chooses DA trading/recourse decisions, while the lower market-clearing problem determines DA dispatch and price; the lower-level optimality conditions can be replaced by KKT conditions and reformulated into a single-level mixed-integer program.
2. Forecast uncertainty is represented by a distributional ambiguity set rather than one fixed distribution. The paper constrains moments / absolute moments / support of forecast error, then optimizes against the worst distribution in that set. A linear decision rule parameterizes balancing recourse as a function of forecast error so the DRO remains tractable.
3. DA dispatch is converted into an RT reference. RT optimization penalizes deviation between actual aggregate RT dispatch and this DA-derived reference, and online primal-dual updates continuously change incentive signals and DER setpoints as system measurements evolve.

**Local forecasting translation**:
The paper itself is market design, not a Spread predictor. The local experiment therefore tests the structural idea rather than claiming literal reproduction:

`D-1 causal state + forecast uncertainty -> Z_hat (future balancing state)`  
`D-1 causal state + uncertainty -> DA_hat`  
`(DA_hat, Z_hat, causal state, uncertainty) -> RT_hat`  
`Spread_hat = RT_hat - DA_hat`.

`Z_true` is defined only as a training/evaluation label: target-day actual net load minus forecast net load. A reserve-normalized version divides positive surprise by positive reserve and negative surprise by negative reserve. Historical uncertainty features are same-slot rolling 28-day mean/std/MAD/q10/q90/width of `Z`, shifted so only D-2 or earlier values are used. Stage-2 RT training uses chronological OOF `DA_hat` and `Z_hat`, never oracle target-day `Z`.

**Local R96 result**:
- `Z_hat` MAE ≈ `2660 MW`, RMSE ≈ `3540 MW`, sign accuracy `51.71%`, correlation `0.259`;
- ordinary two-leg LightGBM: R96 Spread MAE `158.795`, Balanced `50.76%`, +Recall `4.60%`, AUC `0.5710`;
- A03 two-stage: R96 Spread MAE `143.774`, Balanced `53.76%`, +Recall `37.66%`, -Recall `69.86%`, AUC `0.5557`;
- 24-point aggregate: MAE `134.948`, Balanced `58.51%`, +Recall `45.83%`, AUC `0.6179`;
- however RT MAE worsens from ordinary two-leg `81.0` to two-stage `102.4`, with RT bias about `+73.8`; predicted positive rate rises to about `31%` vs true `17.8%`.

**Oracle-state diagnostic**:
Feeding the true future `Z` to the RT model is impossible at deployment but isolates whether the state itself contains useful information. With oracle `Z`, direction AUC rises to `0.6879` at R96 and `0.7368` after 96->24 aggregation. Thus the balancing state is materially informative when known; the main bottleneck is forecasting that state from D-1 14:00 information.

**Verdict**: `PROMISING_INTERMEDIATE_STATE_BUT_STAGE1_NOT_YET_PREDICTABLE`

**Project implication**:
This experiment supports the emerging Route-A architecture `DA anchor + predicted intermediate balancing state -> RT -> Spread`, but does not yet support claiming improved RT prediction. The next literature search should prioritize methods that improve **state predictability / regime identification / uncertainty-conditioned state forecasting**, not merely larger RT backbones.

**Local evidence**:
- code: `experiments/literature_method_lab/a03_applied_energy_two_stage.py`
- result: `experiments/literature_method_lab/results/A03_APPLIED_ENERGY_2023_TWO_STAGE_20261007.json`
- cumulative scorecard: `experiments/literature_method_lab/METHOD_SCORECARD.md`

---

### [A04] Matsumoto, Bunn & Yamada — IEEE Transactions on Power Systems, 2022

**Title**: Mitigation of the Inefficiency in Imbalance Settlement Designs Using Day-Ahead Prices  
**Venue**: IEEE Transactions on Power Systems, 37(5), 3333–3345  
**DOI**: 10.1109/TPWRS.2021.3135334  
**Status**: peer-reviewed, open-access  
**Classification**: `A-CORE / two-step quantile state->RT density`  
**Review status**: `READ_AND_PAPER_FAITHFUL_LOCAL_REPRODUCTION_COMPLETED`

**Paper model retained**:
1. Stage-1 QR predicts the distribution of system imbalance using lagged imbalance plus recent solar and demand forecast errors. Models are hour-specific and the paper studies disclosure lags `r=1..8`.
2. Stage-2 QR models RT imbalance price conditional on DA price, latest observed RT and current imbalance. In Japan the paper uses log prices because prices are positive; it explicitly notes that markets allowing negative prices generally should not apply that log transform.
3. The paper uses `N=10` midpoint quantiles `q_n=(2n-1)/(2N)={0.05,0.15,...,0.95}`. Each of 10 predicted imbalance quantiles is fed through 10 price quantiles, creating a `10x10=100` price mesh. The expected RT price is approximated by the equal-weight average over that mesh.
4. Out-of-sample quantiles are sorted to recover monotonic quantile curves; the paper also reports lasso-penalized QR for robustness with small samples.

**Shandong reproduction contract**:
This is not claimed as a numerical replication of Japan because the information horizon differs materially. The paper is near-real-time with 1–8h delayed imbalance information; our task is D-1 14:00 -> full next day. The local implementation preserves the model structure but makes explicit causal substitutions:
- `Imb[t-r]` -> D-1 14:00 boundary imbalance proxy;
- `Imb[t-24]` -> latest same-hour state actually visible at origin (D-1 for h<=14, otherwise D-2);
- latest solar/demand errors -> D-1 14:00 boundary errors;
- current-day `RT[t-r]` -> D-2 same-hour RT because current-day RT is not operationally available in this project;
- system imbalance is proxied by `actual net load - forecast net load`, label-only;
- Shandong prices can be negative, so Stage 2 uses raw-price QR with an intercept instead of the Japanese log-price equation;
- hourly aggregation is used because the paper estimates hour-specific models; native 96-point data remain the source.

The exact lasso penalty is not available from the accessible paper text. Local QR therefore fixes `alpha=0.001` after standardization before viewing the test metrics; no test-window tuning is performed.

**Local core vs extension**:
The strict paper-core Stage 1 uses boundary imbalance, causal same-hour imbalance history, solar error and load error. A separately labeled Shandong extension adds wind/import/local-generation boundary errors, positive/negative reserves and target forecast net load.

On `2026-08-01..14` (336 hourly targets):
- same-hour LightGBM direct baseline: Spread MAE `129.85`, Balanced `50.17%`, +Recall `6.25%`, AUC `0.6370`;
- paper-core deployable: MAE `131.94`, Balanced `47.05%`, +Recall `0%`, point AUC `0.5524`;
- Shandong-ext deployable: MAE `134.69`, Balanced `46.88%`, +Recall `0%`, point AUC `0.5607`, probability AUC `0.5417`.
Therefore the deployable strict reproduction does **not** beat the direct baseline on our long-horizon task.

However, the Shandong extension clearly improves the intermediate-state density itself relative to strict paper-core:
- mean pinball loss `1053.5 -> 851.7`;
- median-state MAE `3002.6 -> 2323.1 MW`;
- 90% interval coverage `86.0% -> 90.5%`;
- `P(Z>0)` AUC `0.449 -> 0.591`;
- state sign accuracy `43.15% -> 56.85%`.

**Oracle mechanism diagnostics**:
These are label-only diagnostics and must never be reported as deployable performance.
- true DA + predicted Z: Shandong extension Spread MAE `86.73`, Balanced `52.95%`, +Recall `14.58%`;
- true Z + predicted DA: probability Direction Balanced `59.03%`, +Recall `27.08%`, -Recall `90.97%`, AUC `0.7461`;
- true Z + true DA: probability Direction Balanced `68.06%`, +Recall `47.92%`, -Recall `88.19%`, AUC `0.8277`, predicted-positive rate `16.96%` vs true `14.29%`.

**Verdict**: `MECHANISM_VALIDATED_BUT_DEPLOYABLE_REPRODUCTION_BLOCKED_BY_LONG_HORIZON_STATE_AND_DA_PREDICTION`

**Project implication**:
This paper strongly supports keeping a **distribution-valued intermediate state** instead of collapsing future balancing pressure to one point. It also gives a natural final direction score `P(RT>DA)` from the two-dimensional quantile mesh. But the local result demonstrates that the paper's near-real-time information advantage is crucial: at D-1 14:00, the central problem remains forecasting the future balancing-state distribution and DA anchor. Future A papers should therefore be judged by whether they improve those two inputs before RT mapping, not by adding a larger RT backbone.

**Retained design decisions for later architecture**:
- Stage 1 should preferably output a distribution / quantile set for the future balancing state `Z`, not only a point estimate.
- Stage 2 should preserve an RT conditional distribution where practical; for Direction, evaluate `P(RT>DA)` directly instead of only `sign(E[RT]-DA)`.
- Separate diagnostics must distinguish `state usefulness` from `state predictability`, and `DA-anchor error` from `RT-mapping error`; oracle inputs are allowed only as diagnostic upper bounds.
- The deployable score remains the causal D-1 14:00 result. Oracle gains are architectural evidence only and can never be promoted as forecast performance.

**Local evidence**:
- code: `experiments/literature_method_lab/a04_tpwrs_2022_two_step_qr.py`
- result: `experiments/literature_method_lab/results/A04_TPWRS_2022_TWO_STEP_QR_20261007.json`
- cumulative scorecard: `experiments/literature_method_lab/METHOD_SCORECARD.md`

---

### [A05] Browell & Gilbert — Energies, 2022

**Title**: Predicting Electricity Imbalance Prices and Volumes: Capabilities and Opportunities  
**Venue**: Energies, 15(10), 3645  
**DOI**: 10.3390/en15103645  
**Status**: peer-reviewed short communication  
**Classification**: `A-STRUCTURAL-AUXILIARY / regime probability -> regime-conditional price density`  
**Review status**: `READ_AND_FUSED_WITH_A04_LOCALLY`

**Why retained despite venue**:
The venue is below TPWRS/TSG/Applied Energy and this is not promoted as a flagship reference. It is retained because its mechanism directly addresses the current failure mode: predict the future balancing regime first (Long/Short; locally Up/Down), then model the conditional price distribution inside each regime and mix the distributions by regime probability.

**Local fusion architecture**:
The local A05 experiment combines Browell's regime idea with A04's distribution-valued intermediate state rather than replacing the whole pipeline:

`X -> p_up=P(Z>0)`  
`X, regime -> Q_q(|Z| | Up/Down)`  
`p_up * positive-Z distribution + (1-p_up) * negative-Z distribution -> f(Z)`  
`DA + f(Z) -> f(RT)`  
`Direction score = P(RT>DA)`.

Checkpoints are deliberately separated:
- `C1`: Logistic direction Gate + separate positive/negative magnitude quantiles + shared A04 RT-QR;
- `C2`: add only causal D-2-or-earlier 28-day same-hour regime priors and month cycle;
- `C2-R`: split RT quantile models into Up and Down regimes;
- `C3`: replace RT-QR by Browell-style regime-conditional KDE;
- `Oracle Direction`: reveal only the true future sign of Z to measure the value of correct regime identification.

**Deployable results, 2026-08-01..14, 336 hourly targets**:
- direct hourly LightGBM baseline: Spread MAE `129.85`, Balanced `50.17%`, +Recall `6.25%`, -Recall `94.10%`, continuous-score AUC `0.6370`;
- A04 deployable reference: MAE `134.69`, Balanced `46.88%`, +Recall `0%`, probability AUC `0.5417`;
- C1: MAE `134.12`, Balanced `47.57%`, +Recall `0%`, -Recall `95.14%`, probability AUC `0.5413`;
- C2: MAE `133.49`, Balanced `47.40%`, +Recall `0%`, -Recall `94.79%`, probability AUC `0.5444`;
- C2-R: MAE `138.37`, Balanced `46.88%`, +Recall `0%`, probability AUC `0.5367`;
- C3 KDE: MAE `140.23`, Balanced `47.40%`, +Recall `2.08%`, -Recall `92.71%`, probability AUC `0.5383`.

No deployable checkpoint beats the direct baseline. Simple regime-conditional KDE is therefore not retained as an operational component.

**Intermediate Gate findings**:
The regime Gate itself is materially better than the downstream final score:
- C1 Gate Balanced `56.21%`, Up-recall `65.25%`, Down-recall `47.18%`, AUC `0.5945`, Brier `0.2544`;
- C2 Gate Balanced `57.95%`, Up-recall `66.67%`, Down-recall `49.23%`, AUC `0.5809`, Brier `0.2614`.

Thus the direction/magnitude decomposition does create a real state signal. However, the Gate overpredicts Up-state (~57-58% vs actual ~42%) and its calibration is weak. The extra regime priors improve the fixed 0.5-threshold balance but worsen AUC/Brier and magnitude errors, so they are not promoted.

**Oracle Direction diagnosis**:
When only the true future sign of Z is revealed while magnitude and RT remain predicted:
- shared RT-QR reaches Balanced `55.90%`, +Recall `18.75%`, -Recall `93.06%`, AUC `0.7150`;
- regime-specific RT-QR reaches Balanced `55.03%`, +Recall `22.92%`, -Recall `87.15%`, AUC `0.7281`.

These are diagnostic upper bounds, not deployable scores. They show that correct future regime information is valuable, but current causal Gate calibration and the downstream `(regime, magnitude, DA) -> RT` transmission remain limiting.

**Verdict**: `DIRECTION_MAGNITUDE_DECOMPOSITION_HAS_REAL_STATE_SIGNAL_BUT_DEPLOYABLE_GATE_TO_RT_TRANSFER_REMAINS_BLOCKED`

**Project implication**:
Keep the **direction-first, magnitude-second** intermediate-state decomposition as a serious Route-A hypothesis. It is more interpretable and, at the intermediate-state level, more learnable than one-shot continuous Z prediction. But do not yet harden Browell KDE or regime-specific RT models into the main architecture. The next high-value literature should target either (i) well-calibrated long-horizon regime probability, or (ii) a cleaner mechanism that maps predicted regime/state into RT repricing without washing out positive-spread signal.

**Local evidence**:
- code: `experiments/literature_method_lab/a05_regime_mixture_fusion.py`
- result: `experiments/literature_method_lab/results/A05_BROWELL_A04_FUSION_20261007.json`
- cumulative scorecard: `experiments/literature_method_lab/METHOD_SCORECARD.md`

---

## 1.1 已精读并保留的 A 路线方法学基础

> 这两篇不是后续“近五年顶刊优先”的候选池，而是已经完成学习、且确实改变我们研究思路的 foundation。以后保留它们的机制，不再把年份较老的类似论文继续扩张成主候选池。

### [A-FND-01] Woo et al. — Energy Policy, 2016

**Title**: Merit-order effects of renewable energy and price divergence in California’s day-ahead and real-time electricity markets  
**Venue**: Energy Policy, Vol. 92, 299–312  
**DOI**: 10.1016/j.enpol.2016.02.023  
**Status**: peer-reviewed  
**Classification**: `A-STRUCTURAL-FOUNDATION / structural regression / DA→RT divergence mechanism`  
**Review status**: `READ_AND_ACCEPTED_AS_METHODOLOGY_FOUNDATION`

**Methodology lesson retained**:
1. 不把 DA 与 RT 当作两条彼此无关的价格序列，而是在同一价格形成框架中比较两阶段的共同驱动和 RT 特有的新信息。
2. 核心变量分解是 `Actual = Forecast + Surprise`。DA 主要反映日前 forecast state；RT 额外吸收 `actual - forecast` 的 realized surprise，因此 Spread 可解释为“日前预期在实时阶段被修正的价格结果”。
3. 用可检验参数约束而不是事后 feature importance 来验证“forecast error 是否真的解释 DA/RT divergence”。ITSUR/SUR 是估计工具，不是论文最值得迁移的创新。

**Shandong translation retained**:
对本项目，目标日真实 `Actual_D - Forecast_D` 在 D-1 14:00 不可用，不能照搬。可部署替代结构为：
- `F_D`：目标日已知预测状态；
- `E_{D-1}^{visible} = Actual_{D-1}^{1:14} - Forecast_{D-1}^{1:14}`：当前可见的预测偏差状态；
- `G_D = Forecast_D - Actual_{D-1}^{visible}`：从当前真实状态到目标日预测状态的 transition / ramp。
这些量应被解释成 DA→RT correction pressure，而不是匿名 feature。

**Real-data rehearsal note**:
在本地山东数据的示例回放中，D-1 可见 prefix 的负荷、风光、竞价空间等 `actual-forecast` 与 D 日 forecast transition 可以真实构造；该练习只用于验证数据流和机制，不是正式性能实验。

**Project implication**:
后续无论最终采用 A 还是 B，先问“RT 相比 DA 新获得了什么信息、这些信息如何转成价格修正”，再决定模型，而不是先换 backbone。

---

### [A-FND-02] Eicke, Ruhnau & Hirth — Energy Economics, 2021

**Title**: Electricity balancing as a market equilibrium: An instrument-based estimation of supply and demand for imbalance energy  
**Venue**: Energy Economics, Vol. 102, Article 105455  
**DOI**: 10.1016/j.eneco.2021.105455  
**Status**: peer-reviewed  
**Classification**: `A-STRUCTURAL-FOUNDATION / market equilibrium + IV/2SLS + imbalance intermediate state`  
**Review status**: `READ_AND_ACCEPTED_AS_METHODOLOGY_FOUNDATION`

**Methodology lesson retained**:
1. 将 electricity balancing 解释为供给与需求共同决定的 market equilibrium，而不是“forecast error 直接跳到 price”。
2. 显式引入 system imbalance / tightness 作为中间状态，形成 `forecast/system shock -> imbalance state -> imbalance/RT price` 的传导链。
3. 价格和 imbalance quantity 相互决定，存在 endogeneity；论文用 instrumental variables / 2SLS 识别价格响应。IV/2SLS 本身不必照搬到我们的预测任务，但“先识别中间态与相互作用，再预测”是关键方法论。

**Shandong translation retained**:
本地 96 点数据没有一个可直接当作论文真实 `system imbalance MW` 的干净标签，因此不能假装完整复现。我们做了一个只用于机制演练的可解释 proxy：
- `NetLoad = Load - Wind - Solar - Import`；
- 当前可见 net-load surprise 由 D-1 prefix 的 `actual - forecast` 构造；
- 目标日 forecast net-load 与 D-1 当前 actual net-load 形成 transition；
- 正/负备用用于把 upward/downward surprise 转成 reserve-normalized tightness。

**Real-data rehearsal (not a formal benchmark)**:
以 `D=2026-08-14`、D-1 14:00 为预测边界、目标小时 12:00 的一次本地演练：
- D-1 1–14h 平均 net-load surprise 约 `-3995.77 MW`；
- D-1 14:00 边界 net-load error 约 `-7878.44 MW`；
- D 日 12:00 forecast net-load 约 `31603.09 MW`；
- 从边界 actual 到 D 日 12:00 forecast 的 transition 约 `+1382.67 MW`；
- D 日 12:00 正/负备用预测约 `4000 / 700 MW`；
- 构造的 downward reserve-normalized pressure 约 `-5.71`；
- 真实标签 `DA=309.58, RT=-77.23, S=-386.81`。
简单 Ridge 结构演练预测 `S≈-55.81`，方向正确但极端幅值远远不足；即便用真实 DA 做诊断，Spread 仍只有约 `-72.19`。这说明当前可见供需/备用状态能表达一部分方向压力，但没有解释极端 RT correction 的全部机制。

**Project implication**:
下一步真正值得追的不是更深网络，而是哪些可部署信息能解释从“普通负 Spread”跨入极端负价 regime，例如 bid stack、机组下调能力、最小出力/必开约束、断面/拥塞、负备用不足及非线性阈值。

---

## 1.2 已清出主动候选池的旧条目

以下条目不再放在 `docs/58` 主动 A 候选池中，避免后续 Agent 误把它们当成当前路线：
- 2018 Saez-Gallego et al.：主任务是 bidding，且超过五年；
- 2025 O'Connor et al. conformal：是 DA/RT 概率区间方法，但缺少严格 DA–RT–Spread 结构耦合；
- 2018 Ziel & Weron：DA 单腿；
- 2024 O'Connor et al. Irish balancing：RT 单腿。
如以后需要 baseline，可通过旧历史 ledger / DOI 重新定位；它们不再主导当前科研。

---

## 2. Route B 候选池

### [B01] Forgetta, Godin & Augustyniak — Energy Economics, 2025

**Title**: Distributional forecasting of electricity DART spreads with a covariate-dependent mixture model  
**Venue**: Energy Economics, Vol. 144, Article 108332  
**DOI**: 10.1016/j.eneco.2025.108332  
**Status**: peer-reviewed  
**Classification**: `B + 概率预测`  
**Priority**: `B-P0 / 当前最优先精读`

**Verified mechanism**:
- 直接对 DART spread 分布建模；
- 三个 regime：regular、positive spike、negative spike；
- regular component 用 Gaussian；tail regimes 用 Generalized Pareto Distribution；
- covariates 同时进入 spike frequency 与 severity；
- 包括 load、weather、natural-gas-price forecast 等协变量；
- Long Island / NYISO 实证；
- 论文明确报告：severity component 中加入 covariates 很关键；
- neural-network-based quantile regression benchmarks 没有优于该 mixture model。

**Why this matches our preference**:
这篇几乎完全符合当前筛选偏好：Energy Economics；数学结构明确；统计 mixture + EVT；直接处理正负尖峰与厚尾；而且实验结论明确涉及“传统概率模型 vs neural quantile regression”。

**Must audit in full review**:
- 原始 DART 符号定义；
- train/test 时间切分；
- 每个 neural baseline 的具体结构；
- tail metric / scoring rule；
- spike threshold；
- forecast origin 与可用 covariates；
- 是否能迁移到山东 `RT-DA`。

---

### [B02] Galarneau-Vincent, Gauthier & Godin — Energy Economics, 2023

**Title**: Foreseeing the worst: Forecasting electricity DART spikes  
**Venue**: Energy Economics, Vol. 119, Article 106521  
**DOI**: 10.1016/j.eneco.2023.106521  
**Status**: peer-reviewed  
**Classification**: `B + 概率预测 / spike classification`  
**Priority**: B-P0

**Verified mechanism**:
- 直接预测 DART spread spike 的发生概率；
- Long Island / NYISO；
- 使用 tailored engineered feature set；
- 结果对 predictive algorithm 的选择较稳健；
- 进一步把 spike probability 用作 trading signal。

**Why important**:
它强调“信息与特征设计比换一个模型家族更重要”，非常适合与我们当前正/负 spread direction 的 separability 问题对照。

**Caveat**:
官方摘要把 DART 写作 day-ahead minus real-time；本项目固定 `RT-DA`。进入公式与数值实例前必须统一符号，正 spike / 负 spike 语义也要一起翻转核对。

---

### [B03] Hubert, Lolas & Sircar — Energy Economics, 2026

**Title**: Trading electrons: Predicting DART spread spikes in ISO electricity markets  
**Venue**: Energy Economics, Vol. 162, Article 109590  
**DOI**: 10.1016/j.eneco.2026.109590  
**Status**: peer-reviewed; 2026-10 publication  
**Classification**: `B + 概率预测 / spike classification + structural trading optimization`  
**Priority**: B-P0/P1

**Verified mechanism**:
- 从单区扩展到 multi-zone DART spike prediction；
- 在统一 statistical model 中同时处理 positive / negative spikes；
- 预测部分与 logistic regression / spike probability 密切相关；
- 下游显式建立 day-ahead bid-stack price impact；
- 推导 zonal INC/DEC 最优交易量的 closed-form expressions；
- 模型显式考虑 asymmetric buy/sell impact 与 cross-zone congestion。

**Why important**:
它非常符合“传统统计 + 数学推导 + 显式市场约束”的偏好，而且能够帮助我们回答审稿问题：正负 spread 是否应该共享一个框架、但保留不同机制。

**Caveat**:
论文的重要结果之一是 risk-return / trading performance，不可以把交易收益直接等价成 direction accuracy。预测部分与交易优化部分必须拆开阅读。

---

### [B04] Nizharadze, Farokhi Soofi & Manshadi — Algorithms, 2023

**Title**: Predicting the Gap in the Day-Ahead and Real-Time Market Prices Leveraging Exogenous Weather Data  
**Venue**: Algorithms, 16(11), 508  
**DOI**: 10.3390/a16110508  
**Status**: peer-reviewed  
**Classification**: `B + 点预测`  
**Priority**: B-P1

**Verified mechanism / result**:
- 直接预测 DAM / RTM price gap；
- 对比 statistical machine learning 与 deep neural networks；
- RF 用于重要特征分析 / prediction，LSTM 用于时序依赖；
- CAISO 两年数据；
- 论文专门比较 direct gap prediction 与分别预测 DAM/RTM 后做差；
- 加入 exogenous weather features 后，论文报告 learning-method error 降低约 46%。

**Why important**:
它直接触碰 A vs B 的核心争议：到底应该“分别预测再相减”还是“直接预测 gap”。即使 venue 不是首选顶刊，也很适合做方法路线论证与实验基线设计。

**Caveat**:
需要在全文精读时确认 46% 的具体 metric、baseline 和 split；不能把摘要里的 46% 直接写成“B 比 A 提升 46%”。

---

### [B05] Wang et al. — Energies, 2026 — 山东同市场参考

**Title**: Risk-Aware Trading Signals for Smart Aggregators in Multi-Time-Scale Electricity Markets Using Regime-Switching and Tail-Risk Analysis  
**Venue**: Energies, 19(15), 3592  
**DOI**: 10.3390/en19153592  
**Status**: peer-reviewed  
**Classification**: `B + 概率/分位数/Regime`  
**Priority**: B-P1 for same-market mechanism; not direct performance benchmark

**Verified data / method**:
- Shandong 2024；
- 35,136 observations；
- 15 min / 96 intervals；
- spread 明确定义为 `RT - DA`，与本项目符号一致；
- three-state Markov regime switching；
- quantile regression；
- negative / near-zero / positive regime；
- December OOS tail signal recall reported around 0.872 / 0.841；
- removing lagged spreads raises MAE from 24.015 to 54.278 CNY/MWh。

**Why important**:
这是当前最贴近山东 96 点 spread 分布形态、负价、正负尾和 regime persistence 的公开论文之一。

**Critical caveat**:
它的大量预测能力来自近目标 lagged spread。我们的正式业务是 D-1 14:00 一次性预测 D 日全 96 点，无法看到大部分 target-day lagged spread。因此这篇不能拿 0.87 tail recall 来证明我们的 D-1 任务可达到相同水平。

它应主要用于：
- 证明山东 spread 存在三态、厚尾、正负不对称；
- 设计 regime / tail / quantile 评价；
- 说明 forecast horizon 对可预测性影响巨大。

---

## 3. 当前路线判断

### Route A

本轮检索后，Route A 已经从“只有 A01 一篇”变成一条可讨论、可做实验的路线。

建议固定拆成：

`A1: strong DA leg + strong RT leg -> S_hat = RT_hat - DA_hat`

`A2: p(DA) + p(RT|DA) / copula / joint density -> p(S)`

`A3: same probabilistic framework on DA & RT -> calibrated intervals/quantiles -> spread uncertainty propagation`

当前最值得精读的顺序：

1. **A01 Li et al., IEEE TSG 2026**：最直接的 A2，专门解决 DA/RT correlation 与 spread density projection；
2. **A02 Saez-Gallego et al., IEEE TSG 2018**：传统统计/条件场景骨架，最贴近我们偏好的“数学结构清楚、低容量”；
3. **A05 O'Connor et al., ESR 2024**：RT leg，LEAR 统计模型明确优于更复杂方法；
4. **A04 Ziel & Weron, Energy Economics 2018**：DA leg，高维 LASSO 与 univariate/multivariate 系统比较；
5. **A03 O'Connor et al., Energy and AI 2025**：A3 概率区间与 conformal calibration；
6. **A06 Woo et al., Energy Policy 2016**：结构机制，帮助决定 DA 与 RT 两条腿到底应该如何耦合。

当前实验启发不再是“找两个最强模型然后相减”这么简单，而是正式比较三种 A：

- `A1-independent`：DA/RT 各自最优模型；
- `A2-conditional/joint`：显式利用 DA-RT dependence；
- `A3-calibrated probabilistic`：分别保证两端不确定性可靠，再传播到 Spread。

与 B 的核心比较也因此更公平：B 直接学习 `S`，A 则保留两个价格过程的结构信息，代价是可能发生两端误差叠加。

### Route B

当前已经有清晰主线：

`直接 spread -> spike/regime/tail -> covariate-dependent distribution -> 正负事件统一但不强制对称`。

现阶段 B 路线最符合用户偏好的第一篇是 **B01 Forgetta et al. (Energy Economics 2025)**。

若用户输入 `B`，默认先从 B01 开始完整 8 段精读；除非新检索发现更高质量、更贴近 D-1 full-next-day horizon 的顶刊论文。

### 2026-10-07 research pivot after A05

A05 之后不再默认把 `Direction -> Magnitude` 作为主架构。该分解保留为候选诊断，但其历史实验已说明仅做 sign/magnitude decoupling 并不足以稳定提升最终 Spread Direction。

新的优先问题来自山东真实序列形态：
1. `multi-scale`: Spread 更适合看作 `low-frequency state + high-frequency correction + noise/tail`，而不是单一平稳序列；
2. `state persistence`: 幅值可在相邻点大幅跳动，但符号/市场状态通常具有持续性；模型产生频繁 `+/-/+/-` 翻转应被视为结构失真，而不只是逐点分类错误；
3. Route B 正式开启，并与 Route A 并行。优先寻找能够显式处理多尺度、regime persistence、positive/negative tails 和 transition dynamics 的统计/结构模型。

新的 B 路线候选优先级：
- `B01 Forgetta et al., Energy Economics 2025`: covariate-dependent 3-regime DART mixture，regular/positive-spike/negative-spike，Gaussian center + GPD tails；强刊、Spread-native，优先精读。
- `B-PERSIST Kapoor et al., Journal of Forecasting 2023`: Markov regime switching + time-varying transition probabilities，用于研究状态持续性和抑制不真实的逐点符号抖动；不是 DART-specific，但机制高度相关。
- `B-MS Wang et al., 2025`: rolling-window wavelet + LightGBM on Shanxi spot price，用于研究因果动态多尺度分解；venue 较弱，但数据形态和 `low/high-frequency` 问题高度贴近，只作方法学候选，不因本地市场贴近而自动晋升核心文献。
- `B-SD 2026 Shandong MRS`: 三态 Markov（negative/near-zero/positive）在山东 15-min Spread 上显示 multi-hour persistence；venue 较弱，保留为本地机制证据而非核心质量锚。

---

## 4. 逐篇精读区

后续每篇正式精读都追加在这里，并使用固定模板：

1. 论文标题、出处、分类标记  
2. 研究动机  
3. 核心原理与数学思路  
4. 完整数据流  
5. 数值实例  
6. 核心创新点  
7. 实验设置 & 关键结论  
8. 审稿人高频问题

每篇精读完成后，还要在本文件顶部候选条目增加：`REVIEWED=YES`、`TRANSFER_SCORE`、`REPRODUCTION_VALUE`、`RELATED_WORK_VALUE`。

# 57 — DA / RT / Spread 文献研究协议与资料收口

STATUS=ACTIVE_CANONICAL_LITERATURE_PROTOCOL
DATE=2026-10-06
SCOPE=spread24_senior_review 文献阅读与科研辅助资料
MODEL_RUNS=NOT_IN_SCOPE

## 0. 目的

本文件夹接下来的文献工作只服务一个问题：

> 围绕日前（DA）、实时（RT）电价以及价差 `S = RT - DA` 的预测，系统筛选、阅读、比较能够真正帮助我们形成研究路线的学术论文。

文献库不是“论文越多越好”的 bibliography dump，而是为后续方法设计、实验基线、Related Work、审稿答辩提供可核验依据。

今后用户输入：

- `A`：只进入路线 A 搜索与精读；
- `B`：只进入路线 B 搜索与精读。

没有收到 A/B 时，不自行扩展另一条路线。

---

## 1. 两条固定研究路线

### 路线 A — DA / RT / Spread 结构约束联合预测

路线 A 的核心不是“分别预测两个价格再机械相减”，而是：**同时利用日前价 DA、实时价 RT 与价差 `S = RT - DA` 三者之间的确定性恒等关系和强关联波动，对预测过程施加结构约束。**

典型形式包括：

- 联合预测 `DA_hat`、`RT_hat`、`S_hat`，并显式约束 `S_hat = RT_hat - DA_hat` 或其软约束/残差形式；
- 将 RT 建模为 `RT = DA + correction/shock`，其中 correction 由负荷、新能源、备用、拥塞、预测误差等状态解释；
- 使用多变量回归、VAR/VECM、协整、状态空间、结构方程、copula/joint density、多任务或其他显式联合模型刻画 DA→RT 的传导关系；
- 通过 DA/RT 共享驱动项与 RT 特有冲击项，解释 Spread 的形成，而不是只把 Spread 当作一条孤立时序。

允许点预测和概率预测，但**概率预测不是 A 的定义**；A 的定义是结构耦合与联合约束。

判定规则：
- 论文必须把 DA、RT 与二者差值之间的关系作为方法核心，而不是仅在实验中同时报告 DA/RT；
- 只预测 DA 或只预测 RT 的论文不属于 A 主线；
- 只把 `RT-DA` / `DA-RT` 作为单一监督目标、再做 decomposition / frequency / trend-seasonal 建模，归 B；
- 单纯概率后处理（如 IDR、conformal）若没有 DA/RT/Spread 联合结构，只能算辅助方法，不再归 A 主线。

### 路线 B — Spread 本体直接建模 / 分解预测

直接令 `S_t = RT_t - DA_t` 为目标序列，不再通过 DA/RT 联合结构约束。

重点关注：
- trend / seasonality / residual；
- high-frequency / low-frequency decomposition；
- VMD / EMD / wavelet / SSA / Fourier / frequency-domain methods；
- 多尺度时序网络、Transformer、CNN/RNN、mixture-of-experts 等；
- 对 Spread 自身的 regime / spike / tail / direction / magnitude 建模。

路线 B 可以有统计方法，但现实文献中更常见的是分解 + 大参数时序网络；路线 A 则优先寻找传统机器学习、统计/计量、显式数学约束的方法。

## 2. 筛选优先级

### P0 — 最优先

从 2026-10-07 起，下一篇论文默认执行更严格的筛选闸门：**优先近五年（2022–2026）正式发表的 Route A 论文，且优先顶刊 / 强领域期刊 / 公认高水平会议。** 电力系统方向优先关注 IEEE Transactions on Smart Grid、IEEE Transactions on Power Systems、IEEE Transactions on Sustainable Energy 等经典强刊；能源与预测方向优先 Applied Energy、Energy Economics、International Journal of Forecasting 等强刊。直接命中严格 Route A 的顶会论文若稀缺，不为“顶会”标签强行纳入纯 Transformer/大模型架构论文。

满足越多越优先：

1. 2022–2026 正式发表；超过五年的论文只有在“机制奠基、难以被新文献替代”时才作为历史 foundation，不作为下一篇首选；
2. 顶刊 / 强期刊 / 顶会正式发表；
3. 严格属于 Route A：把 DA、RT、Spread 的关系写进方法核心，而不是单腿预测或普通后处理；
4. 方法从市场问题或数学约束出发，以统计、传统机器学习、显式概率模型、优化、结构方程或可解释约束为核心；
5. 有可解释数学结构，而不是纯模块堆叠；
6. 最好专门处理尖峰、厚尾、负电价、异方差、regime、概率密度或不确定性；
7. 实验最好包含现代深度学习 / SOTA baseline，并能说明结构方法的优势来自哪里；
8. 时间切分、forecast origin、输入可用性说明清楚；
9. 与山东 24/96 点、D-1 14:00 预测 D 日的业务边界容易比较。

### P1 — 很有价值但需谨慎迁移

- 同一 spread 对象，但预测 horizon 明显更短；
- 同一 horizon，但市场机制不同；
- 主要用于 spike / tail / trading signal，而不是完整点预测；
- 期刊层级一般，但方法非常贴近本项目。

### P2 — 只做机制或 baseline 参考

- 纯 DA price forecasting；
- 纯 RT price forecasting；
- 只研究 imbalance sign / balancing price；
- 大模型 / 超大网络，且数学新意弱；
- 没有严格 OOS 或存在明显信息泄漏风险。

P2 不作为当前方法路线的核心依据，除非其信息集、数学约束或实验设计特别有价值。

---

## 2.1 固定本地实践底座（2026-10-07）

后续论文实践统一优先复用 `experiments/literature_method_lab/`，避免每篇论文重新搭模型导致无法判断增益来自方法思想还是 backbone。

固定模型池：`Ridge / RandomForest / LightGBM / XGBoost`；固定对照模式至少包含 `direct_spread` 与 `two_leg(DA/RT -> RT-DA)`。默认只改论文要求的特征、中间态、loss / sample weight、约束、target 或必要参数。项目价差符号始终重新统一为 `RT-DA`。

该 lab 仅用于 retrospective methodology practice，不自动升级为正式 confirm；正式实验仍遵守本仓库 lockbox / prospective / leakage contract。

---

## 3. 明确不优先的论文

除非用于对照，不优先：

- 单纯增加 Transformer 层、attention 分支、CNN/RNN 模块的论文；
- 依靠巨大参数量但没有显式市场机制或统计假设的方案；
- 只报告随机切分指标、不交代 forecast origin 的时序论文；
- 把目标日未来真实量、最终 actual、未来 RT/DA 当输入的不可部署方案；
- 只用交易利润证明预测有效、却不报告预测误差 / ranking / calibration 的工作；
- 只在平均样本好看、完全不讨论 spike / negative / tail 的工作。

---

## 4. 每篇论文固定输出结构

后续每篇精读必须严格按以下 8 段写，不随意改结构。

### 1. 论文标题、出处、分类标记

必须包含：
- 完整标题；
- 作者；
- 年份；
- venue；
- DOI / 官方链接；
- peer-reviewed / preprint；
- `A/B + 点预测/概率预测`；
- 目标价差符号原文定义，并转换到本项目 `RT-DA`。

### 2. 研究动机

回答：为什么单独预测 DA/RT 或直接预测 spread 有必要？现有方法失败在哪里？尖峰、厚尾、相关性、regime、天气/负荷误差等哪个问题是主矛盾？

### 3. 核心原理与数学思路

要求：
- 写出关键公式；
- 逐个解释随机变量、参数和约束；
- 区分“论文原式”和“我们的通俗解释”；
- 若有 mixture / copula / EVT / quantile / decomposition，要写清概率意义；
- 不用“用了某网络所以更强”代替原理。

### 4. 完整数据流

分成：

**训练阶段**：原始数据 → 特征 → 目标 → 拟合对象 → loss / likelihood / optimization → 参数。

**推理阶段**：D-1 可用信息 → 模型输入 → 中间量 → DA_hat / RT_hat / S_hat / distribution → 最终结果。

特别检查是否存在 target-day future leakage。

### 5. 数值实例

必须用真实电价量级举例，并固定本项目符号：

例如：
- `DA = 320 CNY/MWh`
- `RT = 470 CNY/MWh`
- `S = RT - DA = +150 CNY/MWh`

然后按论文公式一步一步代入，说明模型如何产生预测。

### 6. 核心创新点

写成可以直接放进 Related Work / 文献综述的 2–4 条正式句子。

同时区分：
- 方法创新；
- 信息集创新；
- 评估 / 业务创新。

### 7. 实验设置 & 关键结论

至少核对：
- 市场、年份、分辨率；
- forecast horizon；
- train/val/test；
- baseline；
- 指标；
- 是否包含最新深度学习模型；
- 传统/统计方法是否真的优于深度模型；
- 是否对 spike / tail / negative price 单独评价；
- 最关键数值结果。

不能只摘摘要里的百分比。

### 8. 审稿人高频问题

每篇至少回答：
- 为什么不直接用深度学习？
- 为什么 A 比 B 合理，或为什么 B 比 A 合理？
- forecast origin 是否公平？
- spread 两端误差会不会叠加？
- 极端样本是否只是 class imbalance？
- 换市场是否还能成立？
- 数学假设是否过强？
- 与简单 AR/GBDT/LightGBM/quantile baseline 相比增益来自哪里？

---

## 5. 文献证据纪律

1. 元数据优先使用出版社 / DOI / IEEE / Elsevier / Springer / ACM / Wiley / INFORMS / 官方期刊页面核验。
2. 关键实验结论必须尽量查看全文表格、实验章节或作者正式版本，不能仅凭搜索摘要扩写。
3. peer-reviewed 与 preprint 分开。
4. headline accuracy 不能跨 horizon、跨目标、跨信息集直接比较。
5. 每篇记录 forecast origin；不知道就明确写 `NOT VERIFIED`。
6. 每篇记录原始 spread 符号；所有数值例子再统一换算到 `RT-DA`。
7. “优于深度学习”必须指出具体深度 baseline、指标与 test setting；否则不得写成结论。
8. 若论文价值主要来自信息集而非模型，不得包装成“模型 SOTA”。

---

## 6. 与现有项目文档的关系

### 保留但降级为历史/实验依据

- `experiments/first_test/RESEARCH_LITERATURE_LEDGER.md`
  - 内容丰富，但混合了 96 点信息审计、实验决策和文献。
  - 以后不再作为 A/B 论文精读的主入口。
  - 只有当某篇论文影响正式实验假设时，再把一句“项目影响”同步回去。

- `docs/46_r96_literature_reframe_information_ceiling_and_research_agenda.md`
  - 保留。
  - 它解决的是 96 点 horizon / information ceiling，不是 A/B 文献库。

- `docs/47_r96_focused_literature_to_code_rounds_20261002.md`
  - 保留。
  - 它解决的是 literature-to-code 语义闭合，不是论文逐篇精读。

### 明确归档，不再作为有效入口

- `experiments/first_test/RESEARCH_LITERATURE_LEDGER_CORRUPTED_20261002_50KB_TRUNCATION.md`
  - 损坏快照，只允许 provenance audit。

- `experiments/first_test/RESEARCH_LITERATURE_LEDGER_RECOVERY_APPENDIX_20261002.md`
  - 恢复过程说明，只允许 provenance / legacy anchor 查询。

### 新的主入口

- 本文件：规则与筛选协议；
- `docs/58_spread_literature_review_ledger.md`：A/B 论文候选池 + 后续逐篇精读结果。

---

## 7. 后续工作约定

收到 `A` 后：

1. 先查 Route A 顶刊/强期刊；
2. 优先传统/统计/显式联合分布方法；
3. 再补少量深度 SOTA 作为实验对照；
4. 排出 3–6 篇最值得读的论文；
5. 先精读最高优先级 1 篇，按固定 8 段输出；
6. 同步写入 `docs/58_spread_literature_review_ledger.md`。

收到 `B` 后同理，但检索对象改为 direct spread / DART / price gap / spread distribution / spike / regime / decomposition。

默认不一次性堆十几篇摘要；重在真正读懂数学、数据流、实验公平性与可迁移性。

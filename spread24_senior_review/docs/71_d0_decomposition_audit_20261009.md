# 71 — D0 Boundary-Safe Decomposition 独立复审

STATUS=PASS_WITH_PRE_RERUN_REMEDIATION
DATE=2026-10-09
AUTHORITY_BASE=docs/69_boundary_safe_low_high_decomposition_adjudication_20261008.md
SCOPE=D0 decomposition selection + production-drift provenance only

## 0. 审核结论

D0 的核心科学裁决可以接受：`NATURAL_TREND_T24` 是在冻结的 2026-01-01..2026-04-28 selection window 上、未使用 final 模型指标、未训练预测器的情况下，唯一通过 docs/69 既定 10 项 Gate 的候选；冻结 operator hash、selection run、retrospective run 和保护目录哈希链完整。

但当前仍不应直接开始完整 MSR rerun。阻断原因不在分解 winner，而在两类审计问题：

1. production drift classifier 把“漂移文件已经被生产仓库正式 commit”当成不满足 external-workflow 条件，逻辑过窄；现有 Git 证据实际上比“仍是未提交改动”更强，需要修正 provenance 判定后再冻结新的只读生产 baseline。
2. D0 文档/代码存在两个不改变数值结果、但必须在实验前修正的数学/审计口径问题：`half-power` 实际实现为 `G=0.5` 的 half-amplitude / half-gain；Gate 9 的 artifact rule 文案比实际判定逻辑更严格，造成 T24 “数值高于 tolerance 但 Gate=PASS”的表面矛盾。

因此本轮结论：

`DECOMPOSITION_SELECTION=ACCEPT`

`SELECTED_DECOMPOSITION=NATURAL_TREND_T24`

`PRODUCTION_ISOLATION=REMEDIATE_BEFORE_RERUN`

`READY_FOR_MSR_RERUN=NO_YET`

## 1. D0 结果复核

Selection run: `20261008T133948429497Z`。

Retrospective run: `20261008T134348928008Z`。

冻结算子：

`L = (I + lambda D2^T D2)^-1 S`, `H = S-L`, no wrap row in `D2`。

T24 lambda = `215.32246474604284`。

matrix sha256 = `7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7`。

Selection window 上 T24：

- reconstruction max error = `1.1369e-13`；
- Low transition ratio = `0.28855`；
- Low ABA = `0.01695/day`；
- var(L)/var(S) = `0.68105`；
- var(H)/var(S) = `0.22893`；
- positive crossing retention = `0.51223`；
- normalized edge High energy = `0.84664`；
- normalized edge Low curvature = `0.45193`；
- D2 无 wrap row；
- D+1 perturbation 对目标日输出影响为 0。

以上与冻结 JSON 一致。

当前 periodic reference 仅因 circular coupling fail；reflect/T8/T12/T16 在 selection window 均因 positive-crossing retention <0.50 fail。T24 是 selection window 唯一 admissible candidate，因此 winner rule 本身没有隐藏的 downstream-model 选择。

## 2. Retrospective 使用边界复核

冻结后才打开 2026-05-01..2026-08-14 retrospective sanity window，winner 未重开。T24 在 retrospective 仍通过原 Gate，positive crossing retention 上升至约 `0.54915`，normalized edge Low curvature 约 `0.38449`，未出现 catastrophic boundary failure。

因此不存在“用 final 重新选 T24”的证据。

该窗口已经消费，只能继续标记 `RETROSPECTIVE_SANITY_CHECK`，不能作为新的 confirmation。

## 3. Finding A — `half-power` 命名错误，数值算子本身不错误

当前 docs/69 与 `decomposition.lambda_for_period()` 写：

`G(2*pi/Tc)=0.5`

并把 `Tc` 称为 `half-power period`。

对线性幅频响应来说，`G=0.5` 是 half-amplitude / half-gain（约 -6.02 dB amplitude），不是通常意义的 half-power；half-power 应为 `|G|=1/sqrt(2)`。

因此：

- lambda 数值与实际 operator 完全匹配冻结公式，不改；
- T8/T12/T16/T24 候选不重选；
- 只把当前和后续文档/代码字段更名为 `gain_0p5_period_quarters` / `half_gain_period`；
- 历史 frozen JSON 不覆盖，增加 amendment 解释。

按真正 half-power 定义，当前 T24 operator 的 half-power 周期约为 29.95 quarters（约 7.49 h），而不是 24 quarters；这只影响频率解释，不改变已冻结矩阵。

## 4. Finding B — Gate 9 文案与实际逻辑有表面冲突

T24 的 smoothing matrix 是 dense 的，`p96 -> p1` 仍存在沿真实 95 步链传播后的极小影响：

`wrap_influence_share_of_max = 4.55e-08`

artifact 文案写成“p96 influence on p1 below tolerance=1e-8”，按字面会 fail。

实际代码正确检查的是：

- p96 是否像 p1 的直接邻点一样起作用；
- 是否具有 circular translation invariance；
- 对 B2，precision / penalty matrix 是否存在 p1-p96 wrap adjacency row。

T24 的 `wrap_to_adjacent_influence_ratio ≈ 5.57e-08`，而 `D2`/precision corner 约 `9.59e-22` relative，不存在 direct circular adjacency。因此 Gate 9 的 PASS 在数学上成立，错误在 rule 文案，不在 operator。

修复：只改 Gate 9 描述为“no adjacent-like circular coupling / no wrap row in penalty”，不得把 frozen selection 重新解释成 `absolute dense influence must be <1e-8`。

## 5. Finding C — positive crossing Gate 通过较窄，需标记敏感性，不得重选

Selection window：

- T24 = `335/654 = 0.51223`；
- Reflect = `325/654 = 0.49694`；
- T16 = `310/654 = 0.47401`。

T24 仅比 0.50 Gate 高 8 个事件，Reflect 仅低 2 个事件。因此“只有 T24 PASS”是合同上正确的 point-estimate Gate 结果，但不能包装成强统计证据证明 T24 显著优于 Reflect。

更重要的是 descriptive train window 2023-2025 的 T24 crossing retention 约 `0.442`，selection 约 `0.512`，retrospective 约 `0.549`。这显示标签分解本身随市场 regime 的 crossing allocation 有明显漂移。

处置：

- 不改变 winner；
- 在正式 rerun 报告中增加按 year/month 的 decomposition-regime diagnostics；
- 明确 train-vs-2026 shift 是预测实验需要面对的数据分布变化，不把它当作重新调 lambda 的理由；
- 如果后续 MSR 失败，必须区分“predictor 不行”与“train/2026 decomposition allocation shift”。

## 6. Finding D — production provenance classifier 逻辑需要修复

最新 read-only evidence 显示两个 source drift 文件当前已经不在 git status 中，而是被 production repo 自己的提交记录吸收：

- `scripts/tests/test_disclosure_aux.py` last touching commit `6964396...`, 2026-10-08 23:03:54+08:00，crawler AUX-V1-r14；
- `scripts/crawler/collect/disclosure_aux.py` last touching commit `f4d21b8...`, 2026-10-08 23:05:24+08:00，crawler AUX-V1-r14 BUILD_VERSION 修正。

两者：

- 都位于 crawler/tests external area；
- 当前 HEAD diff=0；
- 无 MSR/decomposition tokens；
- 对应 pyc 为 CPython 3.10，而实验室是 3.11 且 `-B`；
- MSR package 静态扫描没有 production write call site；
- 漂移在 docs/68 之后继续发生。

当前 `drift_provenance.py` 却硬要求：

`c1_tracked_files_are_uncommitted_in_production_itself == True`

这意味着一旦外部 workflow 把自己的改动 commit，classifier 反而永久 UNRESOLVED。这不是 docs/69 的科学 Gate，而是审计工具过窄。

此外，一个 pyc header 不匹配“当前” source，是因为 source 在 cache 之后又继续被修改/commit；这不应成为否定 external workflow 的必要条件。

修复方向：classifier 必须支持两种互斥且可审计的 external workflow branch：

A. `EXTERNAL_UNCOMMITTED_WORKFLOW`：drift source 仍为 production working-tree modification；

B. `EXTERNAL_COMMITTED_WORKFLOW`：drift source 当前 clean，HEAD last-touching commit 时间晚于 docs/68 drift snapshot，commit/path/domain 与 crawler workflow 一致，lab write closure 为 0，cache/interpreter evidence 与 lab 不一致。

不得通过降低 production zero-drift Gate 来解决。只有 provenance 分类后，才允许 freeze 新 baseline，然后在后续 MSR rerun 前后做 zero-drift verify。

## 7. Test audit

本次独立复跑 focused D0/provenance tests：`34 passed`。

D0 completion 记录：独立 operator/provenance tests `45 PASS`，navigation `4 PASS`。

完整 predictive suite 在 D0 completion 中明确未重跑，因为 docs/69 禁止模型训练；这在 D0 阶段本身可接受。但在下一次完整 MSR rerun 前，完成 Finding A/B/D 的纯工程修正后，应先执行整个 `src/research96/msr_spread_v1/tests`，不得只跑 D0 tests。

## 8. 下一步 Gate

在开始 V1a/V1b controlled rerun 前只允许完成以下四件事：

1. 修 `half-power` -> `half-gain` 术语，不改 lambda/operator/hash；
2. 修 Gate 9 rule 文案，不改实际 pass/fail 算法与 frozen winner；
3. 修 production provenance classifier 的 committed-external branch，并形成新的 CLASSIFIED provenance；
4. provenance classified 后冻结 production baseline v2，并在测试前后验证 zero drift。

全部通过后：

`READY_FOR_MSR_RERUN=YES`

下一轮 MSR rerun 只允许一个科学变化：

`periodic J3 label decomposition -> frozen NATURAL_TREND_T24 operator`

Low/State/Z/High feature routing、quantiles、LightGBM 参数、Markov、reconstruction coupling、窗口和 reporting gates 全部保持不变。

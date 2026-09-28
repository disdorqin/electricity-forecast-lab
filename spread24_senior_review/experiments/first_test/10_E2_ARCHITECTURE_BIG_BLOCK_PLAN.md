# E2-A Architecture Big-Block Ablation

STATUS=AUTHORIZED_NEXT
DATE=2026-09-25
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS
TARGET=RT-DA
FORECAST_ORIGIN=D-1 14:00
LABEL_CUTOFF=S<=D-2

## 1. 固定基准

ARCH_BENCHMARK_DAY=2026-02-13
用途仅限工程 sanity、收敛、runtime、显存、参数量、gradient ownership、重复性。
禁止用单日分数做架构晋升或淘汰。

选择理由（E1 T2）：Raw=0.583333, Balanced=0.537815, AUC=0.563025, +R=0.428571, -R=0.647059, prevalence=0.291667，无 collapse/zero-recall。
历史常用 2026-06-15 只有 3/24 正类，不再作为架构能力基准。

ARCH_DEV_PANEL 固定复用 E1 的 28 天：
W1=2026-02-12..02-18
W2=2026-04-12..04-18
W3=2026-06-12..06-18
W4=2026-08-07..08-13
这 28 天是 architecture DEV，不是最终 lockbox。最终架构冻结后必须用新的更广验证范围确认。

## 2. 当前事实

- E1 T2（DIR-only + Direction-first + current fusion）：Raw=0.578869, Balanced=0.527302, AUC=0.566327。
- strict Direction-first 不具备正式 promotion 证据；E2-A 仅固定沿用它作为 architecture-screening protocol，避免同时改变 checkpoint。
- Negative transfer 未获支持；protected-gradient HOLD。
- threshold/calibration 不是主瓶颈。
- recent180 对 XGB 更差；Stage B HOLD。
- 当前 Direction signal 弱且 regime-dependent。

## 3. 大块架构只比较三种

A0 FULL_CURRENT：现有 T2 原样，Tabular + Temporal + current A2 learnable alpha_dir。28天正式结果直接复用 E1 T2。
新增 architecture_mode=full_current 后，benchmark day 必须与旧 T2 exact 或 <=1e-9 对齐。

A1 TABULAR_ONLY：Direction 只来自 H_tab，H_dir=tab_dir(H_tab)。Temporal 不得影响 Direction output，Direction gradient 必须断开/为0；保留 TabM member axis。

A2 TEMPORAL_ONLY：Direction 只来自 H_time，H_dir=time_dir(H_time)。Tabular 不得影响 Direction output；Temporal 保留 canonical legal future conditioning。

本轮故意不做 fixed alpha grid、concat、late fusion、attention、MMoE、router。先识别 branch signal，再改 fusion。

## 4. 预注册决策树

Case F: FULL > TAB 且 FULL > TEMP => ARCH_SIGNAL=SYNERGY；下一步 E2-B-FUSION。
Case T: TAB >> TEMP 且 FULL<=TAB => ARCH_SIGNAL=TABULAR_DOMINANT；下一步 E2-B-TAB。
Case R: TEMP >> TAB 且 FULL<=TEMP => ARCH_SIGNAL=TEMPORAL_DOMINANT；下一步 E2-B-TEMP。
Case X: TAB/TEMP 都有用但 FULL 更差 => ARCH_SIGNAL=FUSION_PROBLEM；下一步 E2-B-FUSION。
若三者都弱且无稳定差异 => ARCH_SIGNAL=BOTH_WEAK；下一步 FEATURE_WORK。
否则 INCONCLUSIVE/REVIEW。

## 5. 固定训练协议

- objective_mode=dir_only
- checkpoint_policy=direction_first（仅架构筛查固定协议，不推翻 E1 Gate）
- gradient_policy=vanilla
- train_mode=stage_a
- profile=default
- seed=20260924
- k=8（Tabular 适用）
- max_epochs=120, patience=15, batch_size=64
- LR/WD 使用 frozen config
- CUDA + AMP
- Stage B OFF

## 6. 执行顺序

Gate A Routing：只新增 experiment-only architecture_mode=full_current/tabular_only/temporal_only。canonical formal train 默认不变；frozen source/selector/target adapter/data contract 不动。
必须测试：full_current exact reproduction；TAB only 的 Temporal Direction gradient=0；TEMP only 的 Tabular Direction gradient=0；finite/shape PASS；94-test non-destructive suite PASS。

Gate B Benchmark：2026-02-13 跑 A0/A1/A2，只记录工程与收敛；不得排名淘汰。只要工程正常，A1/A2 都进入 Gate C。

Gate C 28-day：A0 复用 E1 T2；A1 跑28天；A2 跑28天。不得根据中间窗口结果提前停。

## 7. 指标

Primary：overall Raw、W1-W4 Raw、min-window Raw、window std。
Safety：Balanced、+R、-R、AUC、Brier、ppf、one-class days、+R=0 days、-R=0 days。
Hour segments：H1=1-8、H2=9-16、H3=17-24，各报 Raw/Balanced/AUC/+R/-R。
Paired：A1-A0、A2-A0，day-cluster bootstrap 95% CI、W/T/L。
Architecture：active/trainable params、best/stop epoch、wall、GPU memory、gradient ownership、A0 alpha_dir trajectory。
禁止每天 oracle 选择 A1/A2 拼接虚假结果。

## 8. 避免局部最优

1. 单日 benchmark 永不参与晋升。
2. 不因某窗口特别好设计窗口专属结构。
3. 大块归因完成前不调 hidden/depth/k/FFT bins/PLE bins。
4. 每轮只改变一个 architecture axis。
5. 始终保留 A0 anchor。
6. 架构 small-block 完成并冻结后，才进入 Stage A/B 训练策略研究。
7. 最终结论必须在未用于选择的新验证范围确认。
8. Raw 不得掩盖 class collapse。
9. 禁止 day/month oracle routing。

## 9. 后续全局路线（仅规划）

E2-A big block -> E2-B branch/fusion small-block -> E2-C top architecture 3-seed -> ARCHITECTURE_FROZEN -> E3 training strategy(Stage A/B) -> broader DEV -> lockbox。
原则：先确定“什么模型”，再确定“怎么训练这个模型”。

## 10. 输出与停止

输出 experiments/first_test/E2_architecture/：00_ARCHITECTURE_PLAN.md、benchmark/、big_block/daily_metrics.csv、model_summary.csv、window_metrics.csv、hour_segment_metrics.csv、paired_deltas.csv、runtime.csv、architecture_audit.csv、E2_A_summary.md、E2_A_GATE.md、figures/、runs/。

E2_A_GATE 只允许 ARCH_SIGNAL=TABULAR_DOMINANT/TEMPORAL_DOMINANT/SYNERGY/FUSION_PROBLEM/BOTH_WEAK/INCONCLUSIVE。
完成 E2-A 后必须停止人工审阅，不执行 E2-B。
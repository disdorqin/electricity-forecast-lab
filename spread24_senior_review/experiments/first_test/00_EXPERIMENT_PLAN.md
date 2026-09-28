# FIRST_TEST 实验计划：V2.1 Direction-First 性能摸底与 E0/E1 Gate

STATUS=READY_TO_EXECUTE
DATE=2026-09-25
PROJECT=spread24_senior_review
TARGET=RT-DA
FORECAST_ORIGIN=D-1 14:00
LABEL_CUTOFF=S<=D-2
CANONICAL_MODEL=V2.1
EXPERIMENT_ROOT=experiments/first_test

> 本目录是实验账本，不替代 docs/17 canonical 规范，也不替代 outputs/tabm_v21 的原始运行产出。
> 每个实验必须先写假设、再运行、再记录结果、再做 Gate 判断。
> 未通过当前 Gate，不得自动进入下一 Gate。
> full Jan-Aug DEV 与 lockbox 在本轮均禁止触碰。

---

# 0. 本轮实验要回答什么

## Primary Claim C1

> Direction-first training policy 是否比当前 V2.1 2pp guardrail 更符合业务目标，并能在不造成类别坍塌的前提下提升或保持 Raw Direction Accuracy？

## Supporting Claim C2

> JOINT Direction+Magnitude 训练是否对 Direction 产生可重复的 negative transfer；若有，是否值得进入 Direction-protected asymmetric gradient 路线？

## Engineering Claim C3

> 当前 V2.1 模型在正式 k=8 配置下的参数量、训练时间、推理时间和收敛预算是否足够轻量，适合后续 rolling-origin 实验与生产候选？

## Anti-claims to rule out

必须排除以下伪结论：

1. 单日多数类比例导致 Raw 虚高；
2. 2-epoch smoke 被误当成模型真实能力；
3. Direction-first 的收益只是 checkpoint 偶然选择；
4. DIR-only 与 JOINT 的差异来自不同数据、不同 seed、不同预算；
5. protected-gradient 的收益只是额外计算量/不同训练流程；
6. 更好结果来自数据泄漏或 target-day truth 参与模型选择；
7. 一次性看完大 DEV 后再回头调规则。

---

# 1. 实验纪律

所有正式比较必须满足：

~~~text
same source
same sequence asset
same selector
same preprocessing contract
same target sign = RT-DA
same D-1 14:00 information boundary
same D-2 label cutoff
same target days
same seed policy
same training budget
same model mode unless该项本身是变量
same metrics
~~~

永久禁止：

~~~text
target-day truth 用于训练/early-stop/checkpoint
修改 frozen source
修改 target adapter
修改 selector 以迎合本轮结果
运行 lockbox
未经 Gate 批准自动扩大到 Jan-Aug 全 DEV
删除或覆盖已有正式 outputs
把 smoke 数字写成性能结论
~~~

---

# 2. first_test 目录规范

建议结构：

~~~text
experiments/first_test/
  00_EXPERIMENT_PLAN.md
  01_EXPERIMENT_TRACKER.md
  02_RUN_RECORD_TEMPLATE.md
  03_CODEX_PROMPT_E0.md

  E0_pilot/
    runs/
      E0_D1_2026-02-15/
      E0_D2_2026-04-15/
      E0_D3_2026-06-15/
      E0_D4_2026-08-10/
      E0_G_2026-06-15_gradient/
    E0_summary.csv
    E0_summary.md
    E0_runtime.csv
    E0_curves/
    E0_GATE.md

  E1_mini/
    runs/
    daily_metrics.csv
    monthly_metrics.csv
    paired_deltas.csv
    E1_summary.md
    E1_GATE.md

  logs/
    code_changes.md
    failures.md
    decisions.md
~~~

原始模型产出继续放：

~~~text
src/TafM_改进源码/outputs/tabm_v21/experiments/direction_first/**
~~~

first_test 每个 run 记录至少：

~~~text
run_id
raw_run_dir
command
git/status snapshot if available
resolved config
seed
target day/window
parameter count
wall time
epoch timing
best epoch
stop epoch
metrics
warnings
decision
~~~

不要求复制全部 checkpoint；必须保存 raw_run_dir 与 manifest hash/路径以便追溯。

---

# 3. Phase P0：实验入口与遥测补齐

在跑任何正式效果前，只允许做必要的 experiment-only 基础设施补齐。

## P0.1 direction-experiment 开放正式训练预算

当前 CLI 仅允许：

~~~text
--profile smoke
~~~

新增 experiment-only：

~~~text
--profile default
~~~

要求：

- 不修改正式 `train` 默认行为；
- 不修改 V2.1 YAML canonical；
- default profile 使用现有正式配置：k=8, max_epochs=120, patience=15, batch=64；
- 所有 direction-experiment 仍写入 experiment output tree；
- smoke 行为保持不变。

## P0.2 增加运行时间遥测

每个 experiment run 至少记录：

~~~text
wall_time_total_seconds
training_epoch_seconds_total
training_epoch_seconds_mean
best_epoch
stop_epoch
prediction_latency_ms
parameter_count_trainable
parameter_count_total
checkpoint_size_bytes
device
amp
~~~

如 GPU 可用，建议记录：

~~~text
cuda_peak_memory_bytes
~~~

若 CPU，则写：

~~~text
cuda_peak_memory_bytes = null
~~~

不要伪造。

## P0.3 first_test 记录器

新增最小 orchestration/recording 逻辑，把每个 run 的摘要写入：

~~~text
experiments/first_test/<stage>/runs/<run_id>/
  RUN_RECORD.json
  RUN_SUMMARY.md
  RAW_RUN_PATH.txt
~~~

允许引用原始 output，不必复制 checkpoint。

## P0.4 P0 验收

必须：

~~~text
canonical non-destructive suite = 91 PASS
direction-experiment smoke remains PASS
default-profile single-run dry/sanity can start
formal train defaults unchanged
experiment output routing unchanged
first_test record created
no lockbox
no full DEV
~~~

P0 只修实验基础设施，不改模型数学。

---

# 4. Gate E0：模型本体性能摸底

## 4.1 目的

回答：

1. 正式 k=8 下通常多少 epoch 收敛？
2. Direction Raw / BCE / Magnitude MAE 的学习轨迹是什么？
3. 是否存在明显 under-training 或 over-fitting？
4. 单日正式训练时间、每 epoch 时间、推理时间是多少？
5. 参数量和存储成本是多少？
6. Direction-first checkpoint 在真实训练长度下如何选择？

## 4.2 固定配置

四个主运行全部固定：

~~~text
model mode=A2
train_mode=stage_a
objective_mode=joint_v21
checkpoint_policy=direction_first
gradient_policy=vanilla
gradient_diagnostics=false

profile=default
k=8
max_epochs=120
patience=15
batch_size=64
seed=20260924
Stage B=OFF
~~~

不得调参。

## 4.3 E0 target days

~~~text
E0-D1 = 2026-02-15
E0-D2 = 2026-04-15
E0-D3 = 2026-06-15
E0-D4 = 2026-08-10
~~~

理由：

- 时间上分散；
- 覆盖不同季节/市场状态；
- 避免只看 6 月单点；
- 均位于已知 DEV 范围内；
- 不触碰 selector 2025-12-31 对 2026-01-01 的边界问题。

## 4.4 E0-G 梯度诊断

单独增加一个诊断 run：

~~~text
target_day=2026-06-15
objective=joint_v21
checkpoint=direction_first
gradient_policy=vanilla
gradient_diagnostics=true
batches_per_epoch=2
profile=default
~~~

它不用于 runtime baseline。

目的：

~~~text
观察正式训练长度下：
tabular_encoder gradient cosine
temporal_encoder gradient cosine
conflict_rate
norm ratio
随 epoch 的变化
~~~

## 4.5 E0 必须记录的指标

每个 target day：

### 收敛

~~~text
best_epoch
stop_epoch
epochs_run
monitor Raw trajectory
monitor Balanced trajectory
monitor +Recall trajectory
monitor -Recall trajectory
monitor L_dir trajectory
monitor L_mag trajectory
monitor Magnitude MAE trajectory
~~~

### Target-day

~~~text
Raw Direction
Balanced Accuracy
+Recall
-Recall
AUC
Brier
Magnitude MAE
Magnitude RMSE
Magnitude tail MAE
Magnitude skill vs naive
~~~

单日 target 指标仅作诊断，不用于模型晋升。

### 工程

~~~text
total params
trainable params
parameter breakdown
wall time
mean/p50/p95 epoch seconds
prediction latency
checkpoint size
device
peak GPU memory if available
~~~

## 4.6 E0 图

至少生成四类图：

~~~text
epoch -> monitor Raw
epoch -> monitor L_dir
epoch -> monitor Magnitude MAE
epoch -> train vs monitor L_dir
~~~

E0-G 再生成：

~~~text
epoch -> gradient cosine by parameter group
epoch -> conflict rate by parameter group
~~~

图只作为分析，不修改训练。

## 4.7 E0 Gate

### GO

同时满足：

1. 4 个 target day 均完整成功；
2. 无 leakage/contract/NaN；
3. 至少能观察到合理收敛轨迹；
4. runtime 可以支持下一阶段约 28×3 规模实验；
5. 没有明显 one-class collapse 成为所有日的默认行为；
6. 不需要修改数据合同即可继续。

### REVIEW / HOLD

出现任一：

~~~text
多数 target day 全程 one-class collapse
best_epoch 常等于 max_epochs，疑似未收敛
大多数 run 前几 epoch 就异常停止
monitor Raw/BCE 行为与实现规则明显矛盾
runtime 显著超出下一阶段可接受预算
数值不稳定/NaN
~~~

若 HOLD，只允许修暴露出的工程/训练预算问题，不进入 E1。

---

# 5. Gate E1-mini：第一次真正性能比较

仅在 E0 Gate=GO 后执行。

## 5.1 比较模型

### M0：canonical checkpoint baseline

~~~text
objective=joint_v21
checkpoint=v21_guardrail
gradient=vanilla
~~~

### M1：Direction-first checkpoint

~~~text
objective=joint_v21
checkpoint=direction_first
gradient=vanilla
~~~

### M2：Direction-only

~~~text
objective=dir_only
checkpoint=direction_first
gradient=vanilla
~~~

第一轮不运行 protected-gradient。

MAG-only 只可作为独立 Magnitude 诊断，不加入 Direction 主排名。

## 5.2 target windows

采用四个 7-day chronological windows，总计 28 个 target days。

建议：

~~~text
2026-02-12 .. 2026-02-18
2026-04-12 .. 2026-04-18
2026-06-12 .. 2026-06-18
2026-08-07 .. 2026-08-13
~~~

每个 M0/M1/M2 使用完全相同的 28 天。

第一轮 seed：

~~~text
20260924
~~~

只有 Top-2 候选进入后续 3-seed。

## 5.3 E1 主指标

Primary：

~~~text
Raw Direction Accuracy across all 28*24 slots
~~~

Safety：

~~~text
Balanced Accuracy
+Recall
-Recall
one-class prediction count
per-month / per-window Raw
~~~

Secondary：

~~~text
Magnitude MAE
Magnitude skill
AUC
Brier
~~~

不得只汇总 672 slots 后隐藏某个月崩塌。

## 5.4 Paired analysis

按完全相同的 target day 做 paired delta：

~~~text
M1 - M0
M2 - M1
~~~

至少保存：

~~~text
daily Raw delta
daily Balanced delta
daily +/-Recall delta
daily Magnitude MAE delta
win/tie/loss counts
~~~

统计不把同一天 24 个小时当成完全独立样本。

若做 bootstrap，优先 day-cluster bootstrap。

## 5.5 E1 Gate

### 若 M1 > M0 且跨窗口稳定

说明 Direction-first checkpoint 值得继续。

### 若 M2 > M1 且跨窗口稳定

说明 JOINT 可能损害 Direction，进入 E1-G / protected candidate。

### 若 M2 ≈ M1

说明没有足够 evidence 支持 multi-task negative transfer；暂不引入复杂 gradient protection。

### 若 M2 < M1

说明 Magnitude task 可能提供正迁移；protected-gradient 必须谨慎，甚至可能不需要。

任何判断必须结合：

~~~text
Raw
Balanced
+Recall
-Recall
gradient diagnostics
~~~

不能只看一个总平均。

---

# 6. Gate E1-G：仅在 E1 支持 negative transfer 时进入

比较：

~~~text
M1 = JOINT + Direction-first + vanilla
M2 = DIR-only + Direction-first
M3 = JOINT + Direction-first + direction_protected
~~~

期望的理想模式：

~~~text
Direction:
M3 ~= M2 > M1

Magnitude:
M3 > M2
~~~

只有这种模式或等价稳定证据，才说明 asymmetric protection 有真正研究价值。

若 M3 只是更慢而 Direction 不提升，则不晋升。

---

# 7. Stage B 暂时 HOLD

在 E0/E1/E1-G 结束前，不实现/不运行：

~~~text
fixed optimizer steps
B-LITE/B-MID/B-FULL
L2-SP anchor
recent90/recent180
two-step Stage B
distillation
weight interpolation
drift-triggered update
~~~

原因：

> 先把 Stage A 主模型与 Direction/Magnitude 关系弄清楚，否则 Stage B 会把变量混在一起。

---

# 8. 计算预算记录方式

实际耗时必须由运行测得，不在计划里伪造。

E0 结束后，用实测：

~~~text
mean seconds/run
p95 seconds/run
mean epochs/run
~~~

估算 E1：

~~~text
estimated_E1_walltime =
mean_E0_run_time * 28 * 3
~~~

注明：

~~~text
串行估算
可并行资源
CPU/GPU 环境
~~~

不要把 smoke 的 1.8s/epoch 直接外推成正式结论。

已知参考：

~~~text
smoke k=4 CPU vanilla:
~1.8-1.9 s/epoch

smoke protected CPU:
~6.7-7.2 s/epoch

正式 k=8 params:
382,724
~~~

这些只作为 planning reference。

---

# 9. 参数量分析

正式 k=8 当前约：

~~~text
TOTAL = 382,724

Tabular Encoder ~334,432
Temporal Encoder = 37,394
Adapters = 10,370
Direction + Magnitude heads = 528
~~~

E0 必须由当前代码重新计算并写入结果，不能只复制本文件。

参数量实验暂不作为主变量。

后续只有在性能稳定后才考虑：

~~~text
k=4 / 8 / 16
TabM-mini vs TabM
~~~

---

# 10. 每个 run 的结论格式

每个 RUN_SUMMARY.md 必须回答：

~~~text
RUN_ID:
PURPOSE:
CONFIG:
RAW_RUN_DIR:
STATUS:

RESULT:
- Direction Raw:
- Balanced:
- +Recall:
- -Recall:
- Magnitude MAE:
- Best epoch:
- Stop epoch:
- Wall time:
- Params:

OBSERVATION:
- 只写本 run 能支持的事实

WARNING:
- class collapse / leakage / instability / abnormal runtime

DECISION:
- KEEP / REJECT / DIAGNOSTIC_ONLY / NEED_REVIEW

NEXT:
- 仅指向当前 Gate 内下一步
~~~

禁止：

~~~text
单日就宣布模型有效
smoke 就宣布 SOTA
没有跨窗口证据就宣布 negative transfer
失败后偷偷改超参再覆盖旧结果
~~~

---

# 11. Tracker 更新纪律

每次运行前：

~~~text
Status=RUNNING
写 command/config
~~~

运行后：

~~~text
Status=PASS / FAIL / INVALID
写 raw_run_dir
写关键 metrics
写 wall time
写异常
~~~

若 run INVALID：

- 保留记录；
- 不删除；
- 写明 INVALID 原因；
- 修复后创建新 run ID。

---

# 12. 当前执行顺序

严格：

~~~text
P0 infrastructure audit/fix
  ->
P0 tests
  ->
E0-D1
  ->
E0-D2
  ->
E0-D3
  ->
E0-D4
  ->
E0-G
  ->
E0 aggregate
  ->
E0_GATE.md
  ->
人工 review
~~~

**到此停止。**

只有人工确认 E0 Gate=GO 后，才允许运行 E1-mini。

---

# 13. 本轮 Codex 成功标准

Codex 第一轮任务完成时，应交付：

~~~text
1. experiment-only default-profile 能运行
2. runtime/params telemetry 完整
3. first_test 记录器可用
4. canonical non-destructive tests 仍 PASS
5. 四个 E0 target day 完成
6. 一个 E0-G gradient diagnostic 完成
7. E0_summary.csv
8. E0_summary.md
9. E0_GATE.md
10. 不运行 E1
11. 不运行 full Jan-Aug DEV
12. 不触碰 lockbox
~~~

如计算环境/时间不允许全部完成：

> 完成已经可以可靠完成的 E0 runs，清晰标记未运行项，不伪造结果，不自动降低实验标准。

# E1-mini 实验计划：Checkpoint 与 Multi-task Negative Transfer

STATUS=AUTHORIZED_AFTER_E0_REVIEW
DATE=2026-09-25
PARENT=experiments/first_test/00_EXPERIMENT_PLAN.md
E0_GATE=GO
TARGET=RT-DA
FORECAST_ORIGIN=D-1 14:00
LABEL_CUTOFF=S<=D-2

> E1-mini 是第一次允许做跨日效果比较的实验。
> E0 的四个 target day 只用于工程/收敛诊断，不允许据此调阈值、loss、selector 或架构。
> E1-mini 结束后再次停止人工 review；不得自动进入 protected-gradient、Stage B 或 full DEV。

# 1. E0 人工审阅结论

E0 工程与收敛验收通过：

- 94-test non-destructive suite PASS。
- 正式 k=8 / CUDA / AMP 路径稳定。
- 参数量 382,724；checkpoint ~1.63 MB；GPU peak ~368 MiB。
- 正式 run wall 11.40–13.99 s。
- best epoch 1–4，stop epoch 16–19，未接近 max_epochs=120。
- monitor L_dir 在早期达到较好状态后快速恶化，说明当前预算不是 under-training，early stop 有必要。
- E0-G 显示 shared gradient conflict 约 34.2%，但仅为诊断，尚不足以证明 negative transfer。
- target-day 出现 class-collapse/zero-recall 风险，E1 必须显式统计 safety metrics。

E0 不能支持：
- 模型有效性结论；
- strict Direction-first 优于 V2.1；
- JOINT 存在可重复 negative transfer；
- protected-gradient 有效。

# 2. E1 要回答的两个问题

## Q1 Checkpoint policy

在完全相同 JOINT 模型、数据、seed 和训练预算下：

M0 = V2.1 2pp guardrail
M1 = strict Direction-first

判断 strict Direction-first 是否真正改善跨日 Direction，而不是单日 Raw 噪声。

## Q2 Multi-task effect

在完全相同 Direction-first checkpoint 下：

M1 = JOINT
M2 = DIR-only

判断 Magnitude task 对 Direction 是：
- negative transfer；
- neutral；
- positive transfer。

# 3. 固定 target windows

总计 28 个 target days：

W1:
2026-02-12 .. 2026-02-18

W2:
2026-04-12 .. 2026-04-18

W3:
2026-06-12 .. 2026-06-18

W4:
2026-08-07 .. 2026-08-13

这些日期全部使用同一 source/sequence/selector/config provenance。

# 4. Variant matrix

## M0 — Canonical checkpoint baseline

objective_mode=joint_v21
checkpoint_policy=v21_guardrail
gradient_policy=vanilla
mode=A2
train_mode=stage_a
profile=default
seed=20260924
device=cuda
AMP=true

## M1 — Direction-first checkpoint

仅改：
checkpoint_policy=direction_first

其它与 M0 完全一致。

## M2 — DIR-only

objective_mode=dir_only
checkpoint_policy=direction_first
gradient_policy=vanilla

其它与 M1 完全一致。

不得在 E1 中改变：
- selector
- preprocessing
- k
- batch size
- LR
- patience
- max_epochs
- loss weights
- threshold 0.5
- Stage B
- gradient protection

# 5. 执行顺序

严格分成两块，但允许在同一 E1 Codex 任务内完成：

E1-A:
先跑 M0 + M1 的 28×2 runs。
完成 checkpoint-policy 聚合。

E1-B:
再跑 M2 的 28 runs。
完成 multi-task 聚合。

禁止依据 E1-A 中间结果改变 E1-B 配置。

# 6. 核心指标

## 6.1 Direction primary

必须同时报告两种聚合：

micro Raw:
所有 28*24 slots 合并后的 Raw Direction Accuracy。

macro-day Raw:
28 个 daily Raw 的算术平均。

Safety:
Balanced Accuracy
+Recall
-Recall
one-class prediction days
positive_recall_zero days
nonpositive_recall_zero days
predicted_positive_fraction by day

## 6.2 Secondary

Magnitude MAE
Magnitude skill vs naive
AUC
Brier

注意：
DIR-only 的 Magnitude 指标只作接口诊断，不用于评价 M2 优劣。

## 6.3 Stability

分别报告 W1/W2/W3/W4：

Raw
Balanced
+Recall
-Recall
Magnitude MAE

不得只报告 672 slots 的总平均。

# 7. Paired analysis

比较：

Delta10 = M1 - M0
Delta21 = M2 - M1

每个 target day 保存：

daily Raw delta
daily Balanced delta
daily +Recall delta
daily -Recall delta
daily Magnitude MAE delta
predicted_positive_fraction delta
best_epoch delta
runtime delta

汇总：
win / tie / loss counts

推荐对 daily paired Raw delta 做 day-cluster bootstrap 95% CI：
- resampling unit = target day
- 不把 24 个小时当成独立 bootstrap unit
- 固定随机 seed
- bootstrap 只作 uncertainty 描述，不作为自动晋升阈值

# 8. Checkpoint safety diagnostics

E0 显示 strict Raw checkpoint 可能在很小 Raw 差异下选择类别更偏的 epoch。
E1 不修改 checkpoint 规则，只增加观察：

每个 run 保存 selected checkpoint 的 monitor：
- Raw
- Balanced
- +Recall
- -Recall
- L_dir
- Magnitude MAE
- predicted_positive_fraction（若当前 monitor evaluator可提供）

统计 M0/M1：
- selected monitor Raw
- selected monitor Balanced
- target-day collapse warning rate

目的：
区分“strict Raw 真正泛化更好”与“monitor Raw 微增但 target safety 变差”。

不要在 E1 临时发明 recall threshold 或 Raw tolerance。

# 9. Gate interpretation

## Checkpoint policy

若 M1 相对 M0：
- micro/macro Raw 有一致改善；
- 多数窗口不反向崩溃；
- safety metrics 不显著恶化；

则 Direction-first checkpoint 保留为 V2.2 candidate。

若 Raw 无稳定改善，或 collapse/zero recall 明显增加：
- 不晋升 strict Direction-first；
- E1 后再讨论 small equivalence band / safety guardrail；
- 本轮不得边跑边改。

## Multi-task

若 M2 > M1 且跨窗口/paired days 具有重复性：
negative-transfer hypothesis = SUPPORTED
=> 下一阶段允许测试 protected-gradient。

若 M2 ~= M1：
negative-transfer hypothesis = INCONCLUSIVE/NEUTRAL
=> 暂不增加 protected complexity。

若 M2 < M1：
Magnitude task 可能存在 positive transfer
=> protected-gradient 不应自动继续。

# 10. Runtime

E0 四个正式 run 平均 wall 约 12.6 s。
84 个 E1 runs 的纯训练 wall 粗略量级约 18 min（串行计算参考，不含 orchestration/IO；实际以 E1 telemetry 为准）。

因此本轮不需要为了省时间减少 target days。

# 11. 输出

experiments/first_test/E1_mini/
  runs/
  daily_metrics.csv
  variant_summary.csv
  window_metrics.csv
  paired_deltas.csv
  checkpoint_diagnostics.csv
  runtime.csv
  E1_summary.md
  E1_GATE.md
  E1_curves_or_figures/

每个 run 继续：
RUN_RECORD.json
RUN_SUMMARY.md
RAW_RUN_PATH.txt

原始模型输出继续留在 outputs/tabm_v21/experiments/direction_first/**。

# 12. E1 结束后必须停止

禁止自动执行：
- direction_protected 正式效果实验
- 3-seed Top-2
- Stage B
- full Jan-Aug DEV
- lockbox
- k sweep
- selector tuning
- threshold tuning

E1_GATE 写完后人工 review。

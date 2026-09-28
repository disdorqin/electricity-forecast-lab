# FIRST_TEST 后续路线：E1-mini 之后的决策树

STATUS=PLANNED_NOT_AUTHORIZED
DATE=2026-09-25
PARENT=experiments/first_test/04_E1_MINI_PLAN.md

> 本文件只定义 E1-mini 完成后的决策逻辑。
> 当前授权范围仍然只有 E1-mini。
> 在 E1_GATE 人工审阅前，不得自动执行本文件中的后续实验。

# 1. 当前阶段

当前已完成：

- P0 实验基础设施
- E0 正式性能摸底
- E0_GATE=GO

当前允许：

- E1-mini
- M0 / M1 / M2 三组 28-day rolling 对比

当前禁止：

- protected-gradient 正式效果实验
- Top-2 多 seed
- Stage B
- full Jan-Aug DEV
- lockbox
- k sweep
- selector tuning
- threshold tuning

# 2. E1 结束后先回答两个问题

## Q1：Direction-first checkpoint 是否值得保留？

比较：

M0 = JOINT + V2.1 2pp guardrail
M1 = JOINT + strict Direction-first

若同时满足：

- micro Raw 不劣；
- macro-day Raw 有稳定改善；
- W1-W4 至少多数窗口同向；
- collapse / zero-recall 不明显恶化；

则：

CHECKPOINT_DIRECTION_FIRST=SUPPORTED

否则：

CHECKPOINT_DIRECTION_FIRST=NOT_SUPPORTED 或 INCONCLUSIVE

若 strict Raw 不稳定，不得立刻拍脑袋改 tolerance。
下一步应专门设计 checkpoint equivalence-band / safety-guardrail 实验。

## Q2：JOINT 是否存在可重复 Direction negative transfer？

比较：

M1 = JOINT + Direction-first
M2 = DIR-only + Direction-first

若：

- M2 micro/macro Raw 明显高于 M1；
- paired daily delta 多数为正；
- 至少 3/4 窗口同向；
- safety metrics 不靠 collapse 换来的；
- gradient diagnostics 与方向一致；

则：

NEGATIVE_TRANSFER=SUPPORTED

否则：

NEGATIVE_TRANSFER=NOT_SUPPORTED 或 INCONCLUSIVE

# 3. E1 后的三条分支

## Branch A：Direction-first SUPPORTED + Negative Transfer SUPPORTED

这是最值得继续的路线。

下一阶段：

E2 = Direction-Protected Candidate

比较：

P0 = M1
JOINT + Direction-first + vanilla

P1 = M2
DIR-only + Direction-first

P2 = M3
JOINT + Direction-first + direction_protected

目标：

Direction:
M3 ~= M2 > M1

Magnitude:
M3 > M2

如果成立：
- protected-gradient 成为 V2.2 主候选；
- 再做 Top-2 3-seed；
- 再讨论 Stage B。

如果不成立：
- 不保留 protected complexity；
- 退回 DIR-only 或 JOINT 简单路线。

## Branch B：Direction-first SUPPORTED + Negative Transfer NOT SUPPORTED

说明 checkpoint 改进有价值，但 Magnitude task 未必伤害 Direction。

下一阶段不做 protected。

优先：

- 固化 M1；
- 3-seed 稳定性；
- 再进入 Stage B B0/B1；
- protected-gradient 暂停。

## Branch C：Direction-first NOT SUPPORTED

先不要继续 protected 或 Stage B。

下一阶段：

E1.5 = Checkpoint Policy Study

候选只允许研究：

- small Raw equivalence band
- Raw + safety guardrail
- Raw primary + BCE tie-break

要求：
- 预先冻结规则；
- 不看 lockbox；
- 同一 28-day 或新的 DEV 子窗口验证；
- 一次只改 checkpoint policy。

只有 checkpoint 稳定后再谈后续模型。

# 4. 多 seed 规则

E1 第一轮仍是单 seed screening。

只有真正候选 Top-2 进入：

seed 1 = 20260924
seed 2 = 20260925
seed 3 = 20260926

多 seed 汇总至少：

- mean/std Raw
- macro-day Raw
- Balanced
- +Recall
- -Recall
- collapse days
- Magnitude MAE
- runtime

不要求所有失败方案都三 seed。

# 5. Stage B 启动条件

只有 Stage A 主方案已经明确后，才能进入 Stage B。

Stage B 顺序：

B0:
Stage A vs current Stage A+B vs Full Retrain

B1:
fixed epochs vs fixed optimizer steps

B2:
B-LITE vs B-MID vs B-FULL
固定 replay，无 anchor

B3:
固定 scope 后测 Stage-A parameter anchor

B4:
recent20 vs recent90d vs recent180d

B5 only if needed:
two-step Direction-first
distillation
weight interpolation
drift-triggered Stage B

禁止同时改多个主变量。

# 6. Full DEV 与 lockbox

在以下条件同时满足前：

- Stage A candidate frozen
- 3-seed stability acceptable
- checkpoint policy frozen
- Stage B 是否启用已冻结
- selector / model / threshold 不再调整

不得运行 full Jan-Aug DEV。

lockbox 必须最后使用，且只能用于最终确认，不得反向调参。

# 7. 论文证据链目标

最终希望形成：

1. Baseline：
   canonical V2.1

2. Direction-first：
   checkpoint/optimization 优先级与业务目标一致

3. Negative-transfer diagnosis：
   JOINT vs DIR-only + gradient conflict

4. Proposed method：
   若必要，Direction-protected asymmetric MTL

5. Continual adaptation：
   Stage B 只作为后续独立贡献/增强，不和主贡献混在一起

核心原则：

> 先证明问题存在，再证明干预有效，再证明复杂度值得。

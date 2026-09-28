# E1-mini GATE

STATUS=COMPLETE
DATE=2026-09-25
EVIDENCE=experiments/first_test/E1_mini/E1_summary.md
RUNS=84 PASS=84 FAIL=0
SEED=20260924（单 seed screening）
REPRODUCIBILITY=PASS（4/4 EXACT，见 E1_summary.md §1.4）

> 本文件只做判定，不授权任何后续实验。
> 下面两条判定都严格按 `04_E1_MINI_PLAN.md` / `06_POST_E1_DECISION_TREE.md`
> 中**预先写定**的判据逐条核对，不使用事后新增的规则。
> 本轮没有加入 recall threshold、Raw tolerance 或任何新的分类规则。

---

# 判定 1：CHECKPOINT_DIRECTION_FIRST

## 结论

```
CHECKPOINT_DIRECTION_FIRST = NOT_SUPPORTED
```

## 预注册判据逐条核对

`06_POST_E1_DECISION_TREE.md` §2 Q1 要求**同时满足**以下四条才判 SUPPORTED。
M1 = JOINT + strict Direction-first，M0 = JOINT + V2.1 2pp guardrail。

| # | 判据 | 实测 | 判定 |
|---|---|---|---|
| 1 | micro Raw 不劣 | 0.5670 vs 0.5342（+0.0327） | **满足** |
| 2 | macro-day Raw 有**稳定**改善 | 与 #1 恒等；CI [−0.0074, +0.0789] 含 0；median Δ = 0.0000 | **未成立** |
| 3 | W1–W4 至少多数窗口同向 | 3/4 为正（W1 −0.0179、W2 +0.0298、W3 +0.0595、W4 +0.0595） | **满足** |
| 4 | collapse / zero-recall 不明显恶化 | `+Recall=0` 天数 4 → 8；collapse-flag run 6 → 9；−Recall 显著上升；Mag MAE 显著恶化 | **不满足** |

四条中 1、3 满足，2 未成立，4 明确不满足。合取不成立 → 非 SUPPORTED。

## 判定理由

**决定性是第 4 条，不是第 1 条。** M1 − M0 的配对分析中，唯一与 0 可分离的两项是：

- micro −Recall **+0.0817**，95% CI **[+0.0169, +0.1543]**（不含 0）
- Magnitude MAE **+3.9147**，95% CI **[+1.9367, +6.1629]**（不含 0）

而 Raw（[−0.0074, +0.0789]）与 +Recall（[−0.1868, +0.0467]）的 CI 都跨 0。
配合 ppf 从 0.3973 降到 0.3199、`+Recall=0` 天数翻倍，
可复现的效应是**运行点向负类偏移**，而不是 Raw 提升。

**关于第 2 条**：`micro Raw ≡ macro-day Raw` 在本设计下是代数恒等
（每天恰好 24 slot，已核验），所以 plan 中"micro 不劣 **且** macro-day 有稳定改善"
实际是同一个统计量，不构成两条独立证据。该统计量的 CI 跨 0、median Δ = 0.0000，
W/T/L = 10/8/10，因此"稳定改善"不成立。

## 必须同时记录的有利证据（避免过度否定）

- Direction-first 让**阈值无关**的概率质量单调改善：
  AUC 0.5176 → 0.5531，Brier 0.2660 → 0.2393，monitor L_dir 0.7288 → 0.7072。
- Raw 点估计为正（+3.27pp），且 3/4 窗口同向。
- 在 28 天里 M1 的 `1-class prediction days` 反而是三组最低（1 vs M0 的 2）。

也就是说：**"Direction-first 提升 direction 能力"这一命题本身是 INCONCLUSIVE
（CI 跨 0），判 NOT_SUPPORTED 的决定性依据是安全条款被违反**——
strict Raw 最大化的代价是可复现的少数类召回退化与幅值精度退化。

若审阅者认为安全条款应从宽解释，本条可下调为 INCONCLUSIVE；
但即便如此，第 2 条未成立仍使其无法判为 SUPPORTED。
按 plan 的字面判据，NOT_SUPPORTED 是更忠实的读法。

## 机制层面（来自 checkpoint diagnostics，用于解释而非判定）

- **M0 的 V2.1 guardrail 在 23/28 个 run 中选中了低于自身 Raw anchor 的 checkpoint**
  （平均低 0.0078 monitor Raw）。这是 guardrail 的既定设计：在 2pp 容差内让位给
  Magnitude MAE。M0 的 Mag MAE 因此最好（52.975）、Raw 最低。
- 所以 M0 vs M1 是**两种显式优先级的对比**，不是"对 vs 错"。
- 值得记录的放大效应：guardrail 只在 monitor 上让出 0.0078，但 target-day micro Raw
  差了 0.0327（约 4 倍）。monitor 集与 target day 的差距会放大策略差异，
  这是 checkpoint 选择的结构性风险。
- `direction_first` 选中的 epoch 明显更早（M1 2.68 / M2 2.00 vs M0 3.79）。

---

# 判定 2：NEGATIVE_TRANSFER

## 结论

```
NEGATIVE_TRANSFER = NOT_SUPPORTED
```

## 预注册判据逐条核对

`06_POST_E1_DECISION_TREE.md` §2 Q2 要求**同时满足**以下五条才判 SUPPORTED。
M2 = DIR-only + Direction-first，M1 = JOINT + Direction-first。

| # | 判据 | 实测 | 判定 |
|---|---|---|---|
| 1 | M2 micro/macro Raw **明显**高于 M1 | +0.0119；CI [−0.0268, +0.0521] 含 0 | **不满足** |
| 2 | paired daily delta 多数为正 | W/T/L = **13/2/13**（中位数 0.0000） | **不满足** |
| 3 | 至少 3/4 窗口同向 | W1 +0.0119、W2 0.0000、W3 −0.0060、W4 +0.0417（2 正 / 1 平 / 1 负） | **不满足** |
| 4 | safety metrics 不靠 collapse 换来 | M2 的 `1-class prediction days` 是三组最高（3 vs M1 的 1） | **不满足** |
| 5 | 与 gradient diagnostics 方向一致 | E0-G 显示 ~34.2% 冲突，但移除 Magnitude 目标后 Direction 无优势 | **不满足** |

五条全部不满足 → NOT_SUPPORTED。

## 判定理由

Delta21 的**每一项** delta（Raw、Balanced、+Recall、−Recall）95% CI 都跨 0，
W/T/L 基本对半。M2 相对 M1 没有可检出的 Direction 优势。

**这个设计的检验力是充分的，不是"没测出来"。** M2 是 `dir_only`：
目标函数直接返回 `L_dir`（`direction_experiments.py:35-36`），Magnitude head
完全不接收梯度。因此 M2 是"Direction 不受任何 Magnitude 目标干扰"的**上限臂**。
如果 JOINT 真的通过梯度冲突损害 Direction，那么完全移除 Magnitude 目标的 M2
应当**明显**高于 M1。实际没有。这是负迁移假设的最强形式反驳，
而不是弱检验。

## 必须同时记录的限制

**M2 的 Magnitude 数字不能用于支持任何结论。**
`dir_only` 下 Magnitude head 未被训练，M2 的 Mag MAE 56.9005 ≈ naive 56.9795
（skill +0.0014），是"未训练 head ≈ 常数预测"的表现。
因此 Delta21 的 `Magnitude MAE −3.0134` 行（CI [−6.3024, +0.3660] 跨 0）
读作"M1 的已训练 Magnitude head 反而差于常数预测"，
**不能**读作"M2 的 Magnitude 更好"。

---

# 汇总

| Gate | Verdict |
|---|---|
| CHECKPOINT_DIRECTION_FIRST | **NOT_SUPPORTED** |
| NEGATIVE_TRANSFER | **NOT_SUPPORTED** |

## 对决策树的映射

按 `06_POST_E1_DECISION_TREE.md` §3：

- Branch A 需要两条都 SUPPORTED → **排除**
- Branch B 需要 CHECKPOINT_DIRECTION_FIRST = SUPPORTED → **排除**
- Branch C 对应 CHECKPOINT_DIRECTION_FIRST = NOT_SUPPORTED → **命中**

```
RECOMMENDED_BRANCH = C
```

Branch C 的下一步是 **E1.5 Checkpoint Policy Study**，plan 只允许研究：
small Raw equivalence band、Raw + safety guardrail、Raw primary + BCE tie-break。
要求预先冻结规则、不看 lockbox、同一 28-day 或新的 DEV 子窗口验证、一次只改 checkpoint policy。

plan 同时明确：**"若 strict Raw 不稳定，不得立刻拍脑袋改 tolerance。"**
本轮判定的正是"strict Raw 不稳定"（CI 跨 0 且安全条款被违反），
所以下一步必须先做等价带/安全护栏的**预先注册**实验，而不是直接调 tolerance。

## 与论文证据链的关系

`06_POST_E1_DECISION_TREE.md` §7 希望先证明"问题存在"再证明"干预有效"。
本轮结论对证据链的影响：

- **问题存在**：部分成立但需要改写。原始表述是"JOINT 通过梯度冲突损害 Direction"。
  E1 不支持这个表述（NEGATIVE_TRANSFER = NOT_SUPPORTED）。
  本轮支持的是另一个问题：**strict Raw-max checkpoint 会以少数类召回与幅值精度
  为代价换取统计上不可分离的 Raw 点估计**（CHECKPOINT_DIRECTION_FIRST 的安全条款违反）。
- **干预是否有效**：protected-gradient 的前提（负迁移存在）本轮未获支持，
  因此 §7 第 4 条（Direction-protected asymmetric MTL）目前**缺少动机证据**。

---

# 未授权事项（本轮一律未执行）

protected-gradient 正式实验、Top-2 多 seed、Stage B、full Jan-Aug DEV、lockbox、
k sweep、selector tuning、threshold tuning（含 recall threshold 与 Raw tolerance）。

`06_POST_E1_DECISION_TREE.md` 只用于判断路线，不构成执行授权。等待人工批准。

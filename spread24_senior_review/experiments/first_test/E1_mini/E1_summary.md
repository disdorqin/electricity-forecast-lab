# E1-mini 实验总结

STATUS=COMPLETE
DATE=2026-09-25
PARENT=experiments/first_test/04_E1_MINI_PLAN.md
GATE=experiments/first_test/E1_mini/E1_GATE.md

> E1-mini 是单 seed screening 实验，用于回答两个问题：Direction-first checkpoint
> 是否值得保留、JOINT 是否存在可重复的 Direction negative transfer。
> 本文件只报告证据，不做后续实验授权。

---

# 1. 实验范围与完整性

| 项 | 值 |
|---|---|
| Target windows | W1 2026-02-12..02-18、W2 2026-04-12..04-18、W3 2026-06-12..06-18、W4 2026-08-07..08-13 |
| Target days | 28 |
| Variants | M0 / M1 / M2 |
| 计划 runs | 84 |
| 完成 runs | 84 |
| PASS | 84 |
| FAIL / INVALID | 0 |
| Retry | 0（无 run 需要重试） |
| 未参与聚合的失败 run | 无 |

每个 run 的记录位于 `experiments/first_test/E1_mini/runs/<RUN_ID>/`，包含
`RUN_RECORD.json`、`RUN_SUMMARY.md`、`RAW_RUN_PATH.txt`、`COMMAND_STDOUT.txt`、
`COMMAND_STDERR.txt`。RUN_ID 格式 `E1-{variant}-{target_day}`。

原始模型产出写入既有 output tree，未覆盖任何历史 run：

| Variant | Category | E1 新增 | 历史保留 |
|---|---|---|---|
| M0 | `outputs/tabm_v21/experiments/direction_first/smoke/` | 28 | 是 |
| M1 | `.../direction_first/checkpoint_policy/` | 28 | 是 |
| M2 | `.../direction_first/objective_modes/` | 28 | 是 |

目录名带 UTC 时间戳，`run_dir` 使用 `exist_ok=False`，结构上不可能覆盖历史 run。

## 1.1 变体定义（冻结）

| Variant | objective | checkpoint | gradient |
|---|---|---|---|
| M0 | `joint_v21` | `v21_guardrail` | `vanilla` |
| M1 | `joint_v21` | `direction_first` | `vanilla` |
| M2 | `dir_only` | `direction_first` | `vanilla` |

三组共同固定：A2、Stage A only、`profile=default`、k=8、`seed=20260924`、
epf-2、CUDA、AMP、同一 source/sequence/selector/preprocessing、
target=RT-DA、origin=D-1 14:00、labels <= D-2。

本轮未改动 threshold、loss 权重、LR、patience、max_epochs、k、selector，
未运行 Stage B / protected-gradient / full DEV / lockbox。

## 1.2 溯源

| 项 | 值 |
|---|---|
| config_sha256 | `8c981156cecf6e11…` |
| selector_sha256 | `ed348cfd9fd911bc…` |
| source_sha256 | `a1b86f956d9fb18a…` |
| seed | 20260924 |
| device / amp | cuda / True |
| parameter_count_total | 382,724 |

完整 64 位 sha256 见各 run 的 `RUN_RECORD.json`。

## 1.3 Preflight

28 个 target day 全部存在、可训练（train samples 1494–1676）、均未被 self-quarantine。
`PREFLIGHT=PASS`，明细见 `preflight_target_days.csv`。

## 1.4 可复现性交叉验证

E0-D1..D4 与 E1-M1 的配置三元组完全相同（`joint_v21` / `direction_first` /
`vanilla`、A2、`profile=default`、seed 20260924），target day 分别为
2026-02-15 / 04-15 / 06-15 / 08-10，均落在 E1 窗口内。4 个重合日的结果：

| target_day | E0 run | E1 run | 状态 | max abs metric delta | best_epoch |
|---|---|---|---|---|---|
| 2026-02-15 | E0-D1 | E1-M1-2026-02-15 | EXACT | 0.0 | 2 = 2 |
| 2026-04-15 | E0-D2 | E1-M1-2026-04-15 | EXACT | 0.0 | 1 = 1 |
| 2026-06-15 | E0-D3 | E1-M1-2026-06-15 | EXACT | 0.0 | 4 = 4 |
| 2026-08-10 | E0-D4 | E1-M1-2026-08-10 | EXACT | 0.0 | 3 = 3 |

9 项指标（Raw / Balanced / ±Recall / AUC / Brier / Mag MAE / Mag skill / ppf）
在独立重跑之间**逐位相同**，best epoch 一致。`REPRODUCIBILITY=PASS`。

副产品：E0 记录 wall time 11.4–14.0s，E1 同配置 15.7–21.5s，而指标与 epoch 完全一致，
说明 E1 的 wall time 差异来自共享机器负载，不是配置差异。

---

# 2. 主结果（28 target days）

micro = 672 个 slot 池化；macro-day = 28 个 target day 先算再平均。

| Variant | micro Raw | macro-day Raw | Raw std | micro Bal | macro-day Bal | micro +R | micro −R | macro-day +R | macro-day −R |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 0.5342 | 0.5342 | 0.1786 | 0.5060 | 0.5083 | 0.4050 | 0.6070 | 0.4175 | 0.5991 |
| M1 | 0.5670 | 0.5670 | 0.1386 | 0.5180 | 0.5118 | 0.3430 | 0.6930 | 0.3427 | 0.6808 |
| M2 | 0.5789 | 0.5789 | 0.1641 | 0.5273 | 0.5067 | 0.3430 | 0.7116 | 0.3195 | 0.6939 |

| Variant | ppf | micro AUC | micro Brier | micro Mag MAE | macro-day Mag MAE | naive Mag MAE | micro Mag skill | macro-day Mag skill | all-neg baseline Raw |
|---|---|---|---|---|---|---|---|---|---|
| M0 | 0.3973 | 0.5176 | 0.2660 | 55.9992 | 55.9992 | 56.9795 | +0.0172 | −0.0417 | 0.6399 |
| M1 | 0.3199 | 0.5531 | 0.2393 | 59.9139 | 59.9139 | 56.9795 | −0.0515 | −0.1323 | 0.6399 |
| M2 | 0.3080 | 0.5663 | 0.2350 | 56.9005 | 56.9005 | 56.9795 | +0.0014 | +0.0003 | 0.6399 |

## 2.1 两条必须先说明的读数约束

**(a) micro Raw 与 macro-day Raw 在本设计下是恒等的。**
每个 target day 恰好 24 个 slot（已核验：84 个 run-day 的 `n_slots` 全为 24），
所以 `Σcorrect/672` 与 `mean_d( correct_d/24 )` 代数上相等。表中两列相同不是笔误，
而是设计结果。因此 plan 里"micro Raw 不劣 **且** macro-day Raw 有稳定改善"这两条
判据实际是同一个统计量，不能当作两个独立证据。Balanced 因为是非线性函数，
micro 与 macro-day 才会分叉（M0 0.5060 vs 0.5083）；Magnitude MAE 是均值，同样恒等。

**(b) M2 的 Magnitude 数字不是 Magnitude 能力。**
`dir_only` 的目标函数直接返回 `losses["L_dir"]`（`direction_experiments.py:35-36`），
Magnitude head **不接收任何梯度**。M2 的 Mag MAE 56.9005 ≈ naive 56.9795、
skill +0.0014，正是"未训练 head ≈ 常数预测"的表现，与 baseline 的 44.035（E0）
相差甚远。因此 M2 与 M1 的 Magnitude 对比不构成 Magnitude 能力的比较，
只能读作"移除 Magnitude 目标后 Direction 上限如何"。

## 2.2 方向判读（本轮主要发现）

M0 → M1 → M2 上，阈值无关的概率质量指标**单调改善**，而阈值化后的准确率没有同步改善：

| 指标 | M0 | M1 | M2 | 方向 |
|---|---|---|---|---|
| micro AUC | 0.5176 | 0.5531 | 0.5663 | 单调改善 |
| micro Brier | 0.2660 | 0.2393 | 0.2350 | 单调改善 |
| monitor L_dir | 0.7288 | 0.7072 | 0.6887 | 单调改善 |
| micro Raw | 0.5342 | 0.5670 | 0.5789 | 变化但不显著（见 §4） |
| micro +R | 0.4050 | 0.3430 | 0.3430 | M0 最好 |
| micro −R | 0.6070 | 0.6930 | 0.7116 | 单调改善 |
| ppf | 0.3973 | 0.3199 | 0.3080 | 单调下降 |

即：Direction-first 确实让**排序质量**（AUC/Brier/L_dir）变好，但代价是运行点
系统性偏向负类——预测为正的比例从 0.3973 降到 0.3080，+Recall 下降、−Recall 上升。
阈值化准确率因此没有同步收益。

三个变体都远低于 all-negative 多数类基线 Raw 0.6399，**排除了多数类虚高**解释。

---

# 3. W1–W4 稳定性（禁止只给 672 slots 总平均）

| Variant | Window | days | micro Raw | std | min | max | micro Bal | micro +R | micro −R | macro-day Mag MAE | ppf | 1-class days | +R=0 days | −R=0 days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | W1 | 7 | 0.6548 | 0.1662 | 0.4167 | 0.8750 | 0.4762 | 0.1190 | 0.8333 | 46.554 | 0.1548 | 1 | 3 | 0 |
| M0 | W2 | 7 | 0.4881 | 0.1662 | 0.2500 | 0.7500 | 0.4900 | 0.3294 | 0.6506 | 81.787 | 0.3393 | 0 | 0 | 0 |
| M0 | W3 | 7 | 0.4821 | 0.1716 | 0.1667 | 0.7083 | 0.5083 | 0.6000 | 0.4167 | 61.776 | 0.5893 | 0 | 1 | 0 |
| M0 | W4 | 7 | 0.5119 | 0.1890 | 0.1667 | 0.7500 | 0.5158 | 0.5273 | 0.5044 | 33.879 | 0.5060 | 1 | 0 | 2 |
| M1 | W1 | 7 | 0.6369 | 0.1730 | 0.3333 | 0.8750 | 0.4722 | 0.1429 | 0.8016 | 51.778 | 0.1845 | 1 | 4 | 0 |
| M1 | W2 | 7 | 0.5179 | 0.1199 | 0.3750 | 0.7500 | 0.5194 | 0.3882 | 0.6506 | 83.064 | 0.3690 | 0 | 0 | 0 |
| M1 | W3 | 7 | 0.5417 | 0.0962 | 0.4167 | 0.6667 | 0.5139 | 0.4167 | 0.6111 | 65.506 | 0.3988 | 0 | 2 | 0 |
| M1 | W4 | 7 | 0.5714 | 0.1535 | 0.3750 | 0.8333 | 0.5134 | 0.3455 | 0.6814 | 39.307 | 0.3274 | 0 | 2 | 1 |
| M2 | W1 | 7 | 0.6488 | 0.1611 | 0.3750 | 0.9167 | 0.5040 | 0.2143 | 0.7937 | 58.045 | 0.2083 | 2 | 3 | 0 |
| M2 | W2 | 7 | 0.5179 | 0.1629 | 0.2917 | 0.7500 | 0.5184 | 0.4706 | 0.5663 | 79.890 | 0.4524 | 0 | 0 | 0 |
| M2 | W3 | 7 | 0.5357 | 0.1305 | 0.3750 | 0.7083 | 0.4870 | 0.3167 | 0.6574 | 59.717 | 0.3333 | 1 | 2 | 0 |
| M2 | W4 | 7 | 0.6131 | 0.1950 | 0.3333 | 0.8750 | 0.5257 | 0.2727 | 0.7788 | 29.950 | 0.2381 | 0 | 2 | 1 |

共同结构：W2 的 Magnitude MAE 最差（79.9–83.1，skill 为负），W4 最好（29.9–39.3）；
W1 的 +Recall 极低（0.119–0.214，+R=0 days 3–4）。这些窗口差异大于变体差异，
是 7 天窗口 + 单 seed 下判读时必须保留的噪声尺度。

---

# 4. Paired analysis（重采样单位 = target day，固定 seed）

Bootstrap：`seed=20260924`、10000 次、重采样单位 = target day（24 个 slot 不独立）。
Bootstrap 只描述不确定性，**不用于决定 promotion**。

## 4.1 M1 − M0（Delta10）

| 指标 | mean Δ | 95% CI | 含 0 | W/T/L |
|---|---|---|---|---|
| micro Raw | +0.0327 | [−0.0074, +0.0789] | 是 | 10/8/10 |
| micro Balanced | +0.0035 | [−0.0411, +0.0526] | 是 | 10/5/13 |
| micro +Recall | −0.0748 | [−0.1868, +0.0467] | 是 | 5/9/14 |
| micro −Recall | **+0.0817** | **[+0.0169, +0.1543]** | **否** | 14/8/6 |
| Magnitude MAE | **+3.9147** | **[+1.9367, +6.1629]** | **否** | 4/4/20 |
| ppf | −0.0774 | — | — | — |
| best epoch | −1.107 | — | — | — |
| runtime (s) | +2.392 | — | — | — |

median ΔRaw = 0.0000。

分窗口 ΔRaw：W1 −0.0179、W2 +0.0298、W3 +0.0595、W4 +0.0595（3/4 为正）。
分窗口 ΔMag MAE：W1 +5.224、W2 +1.276、W3 +3.730、W4 +5.428（4/4 为正，M1 全面更差）。

**读法**：M1 − M0 中唯一与 0 可分离的两项是 −Recall 上升和 Magnitude MAE 恶化。
Raw 与 +Recall 的 CI 都跨 0。也就是说，Direction-first 相对 V2.1 guardrail 的
**可靠效应是运行点向负类偏移**，而不是 Raw 提升。

## 4.2 M2 − M1（Delta21）

| 指标 | mean Δ | 95% CI | 含 0 | W/T/L |
|---|---|---|---|---|
| micro Raw | +0.0119 | [−0.0268, +0.0521] | 是 | 13/2/13 |
| micro Balanced | −0.0051 | [−0.0489, +0.0328] | 是 | 12/4/12 |
| micro +Recall | −0.0232 | [−0.1256, +0.0625] | 是 | 10/7/11 |
| micro −Recall | +0.0131 | [−0.0360, +0.0633] | 是 | 10/9/9 |
| Magnitude MAE | −3.0134 | [−6.3024, +0.3660] | 是 | 17/0/11 |
| ppf | −0.0119 | — | — | — |
| best epoch | −0.679 | — | — | — |
| runtime (s) | +3.263 | — | — | — |

median ΔRaw = 0.0000。

分窗口 ΔRaw：W1 +0.0119、W2 0.0000、W3 −0.0060、W4 +0.0417（2 正 / 1 平 / 1 负）。

**读法**：Delta21 的**每一项** CI 都跨 0，W/T/L 基本对半。Magnitude 一项不可用
（§2.1b）。因此 M2 相对 M1 没有可检出的 Direction 优势。

---

# 5. Safety / collapse

| Variant | 1-class prediction days | +Recall=0 days | −Recall=0 days | collapse-flag runs | micro +R | micro −R |
|---|---|---|---|---|---|---|
| M0 | 2 | 4 | 2 | 6 | 0.4050 | 0.6070 |
| M1 | 1 | **8** | 1 | **9** | 0.3430 | 0.6930 |
| M2 | **3** | 7 | 1 | 8 | 0.3430 | 0.7116 |

- M0 → M1：`+Recall=0` 的天数翻倍（4 → 8），collapse-flag run 从 6 升到 9。
  结合 §4.1 中 −Recall 显著上升、ppf 下降 0.077，方向一致。
- M2 的 `1-class prediction days` 是三组最高（3），说明其 Raw 点估计伴随更强的退化倾向。
- 需要保留的反例：M1 的 `1-class prediction days` 是三组最低（1），所以 M1 并非
  在"整日单类"意义上更差，而是在"少数类召回被压到 0 的天数"上更差。

---

# 6. Checkpoint diagnostics

| Variant | runs | sel monitor Raw | monitor Raw − anchor | sel monitor Bal | sel monitor +R | sel monitor −R | sel monitor L_dir | sel monitor Mag MAE | 1-class runs | collapse-flag runs | mean best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 28 | 0.5427 | **−0.0078** | 0.5330 | 0.4220 | 0.6439 | 0.7288 | 52.975 | 2 | 6 | 3.79 |
| M1 | 28 | 0.5505 | 0.0000 | 0.5337 | 0.3438 | 0.7237 | 0.7072 | 54.303 | 1 | 9 | 2.68 |
| M2 | 28 | 0.5566 | 0.0000 | 0.5421 | 0.3909 | 0.6934 | 0.6887 | 56.093 | 3 | 8 | 2.00 |

- M1/M2 的 `monitor Raw − anchor` 恒为 0，是 `direction_first` 严格字典序的构造结果
  （先最大化 monitor Raw），属于策略定义，不是证据。
- **M0 的 guardrail 在 23 / 28 个 run 里选中了低于自身 Raw anchor 的 checkpoint**
  （平均低 0.0078 monitor Raw）。这是 V2.1 guardrail 的设计行为：在 2pp 容差内
  让位给 Magnitude MAE。它解释了 M0 的 Mag MAE 为何最好（52.975）、Raw 为何最低。
  也就是说，**M0 与 M1 的差异本质上是两种显式优先级之争，而不是"有没有做对"**。
- 值得注意的是 guardrail 只让出 0.0078 monitor Raw，但 target-day micro Raw 的
  差距是 0.0327。monitor 集与 target day 的差距放大了策略差异——这是
  checkpoint 选择的一个结构性风险，值得在后续 checkpoint 研究中记录。
- `direction_first` 选中的 epoch 明显更早（M1 2.68、M2 2.00 vs M0 3.79）。
- **target-day collapse warning**：三组都存在，见 §5。

---

# 7. Runtime

| Variant | runs | total wall (s) | mean wall (s) | min | max | std | mean epoch (s) | mean epochs | pred latency (ms) | GPU peak (bytes) |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 28 | 429.7 | 15.345 | 13.293 | 17.102 | 1.134 | 0.5969 | 19.14 | 10.35 | 385,927,168 |
| M1 | 28 | 496.6 | 17.737 | 11.188 | 24.913 | 3.236 | 0.7249 | 17.68 | 12.07 | 385,927,168 |
| M2 | 28 | 588.0 | 21.000 | 13.985 | 26.305 | 2.772 | 0.8446 | 17.00 | 15.96 | 389,894,656 |

参数总量三组一致（382,724）。`dir_only` 的 GPU peak 略高，属实现细节，无实际影响。

**关于 wall time 的诚实说明**：均值顺序 M0 < M1 < M2 与 epoch 数顺序一致
（19.14 > 17.68 > 17.00 反向），但 §1.4 证明同一配置在 E0（11.4–14.0s）与
E1（15.7–21.5s）下指标逐位相同却 wall time 差 40%。因此本轮 wall time 是
**在同一台机器上先后运行、受共享负载影响的测量**，不足以支持"某个变体更快"的结论。
Paired Δruntime（+2.392s、+3.263s）同样受此污染，只作记录。

---

# 8. 本轮限制

1. **单 seed**（20260924），28 天。所有 CI 都只描述这 28 天的不确定性，
   不能外推为"模型能力"的置信区间。
2. **micro Raw ≡ macro-day Raw**（§2.1a），判据的独立性弱于 plan 的字面表述。
3. **M2 的 Magnitude 不可用**（§2.1b），Delta21 的 Magnitude 行不构成证据。
4. **窗口噪声 > 变体差异**：W1–W4 的 Raw 极差达 0.17，而变体间均值差 0.03–0.04。
5. **Runtime 受机器负载污染**（§7）。
6. 本轮不涉及 Stage B、protected-gradient、lockbox、k sweep、selector tuning，
   结论只在 Stage A + 单 seed + 28 天范围内成立。

---

# 9. 文件清单

`experiments/first_test/E1_mini/` 产出：

- `daily_metrics.csv`（84 行）、`variant_summary.csv`、`window_metrics.csv`
- `paired_deltas.csv`（56 行 = 2 comparison × 28 day）、`paired_summary.csv`
- `checkpoint_diagnostics.csv`、`runtime.csv`
- `preflight_target_days.csv`、`reproducibility_check.csv`、`E1_report_tables.json`
- `E1_curves_or_figures/e1_daily_raw_by_window.png`、`e1_paired_deltas.png`、`e1_safety_counts.png`
- `runs/<RUN_ID>/`（84 个 run 记录）
- `E1_summary.md`、`E1_GATE.md`
- 脚本：`run_e1_mini.py`、`analyze_e1_mini.py`、`write_e1_reports.py`、`check_e1_reproducibility.py`

未修改 `src/TafM_改进源码/**` 与 `docs/**` 下任何文件（`git status` 无 tracked 变更）。

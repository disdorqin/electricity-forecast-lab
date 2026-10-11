# 73 — MSR T24 修复与三个月阶段验证闭环

STATUS=RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC

执行合同：docs/72；科学模型合同：docs/63。仅更换冻结 T24 分解，其余模型/特征/quantile/Markov/Z 参数不变。

## 工程与阶段闭环

修复 Gate、production independence、222/222 测试、非训练 CLI smoke、八阶段训练及上游哈希校验均 PASS。生产漂移归属 EXTERNAL_COMMITTED_WORKFLOW；不修改、还原或清理生产项目。
旧证据及冻结文件基线 669 项复核，漂移 0；范围以 PROTECTED_BEFORE.json 为准，不声称整个生产树零漂移。

第一次链因 Windows 超长 JSON 路径中断，原链保留。修复纯 I/O 路径后重新通过完整测试，另起新链；没有更改科学参数。

成功 chain：`20261009T025637339105Z`；完整阶段 run IDs：CHAIN_IDS.json。S0→Low→State→Z→High no-Z→High with-Z→重构→解释顺序完成。

## 五组对照（方向/召回/率为百分比）

| 时段 | 模型 | MAE | Direction | Balanced | +Recall | -Recall | AUC | Brier | 预测正率 | 真正率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | DIRECT_LGBM | 116.220 | 56.91 | 52.33 | 38.30 | 66.35 | 0.5454 | 0.2457 | 35.21 | 33.67 |
| pooled | PERIODIC_MSR_V1a | 115.070 | 60.79 | 52.40 | 26.70 | 78.10 | 0.5574 | 0.2341 | 23.52 | 33.67 |
| pooled | PERIODIC_MSR_V1b | 115.245 | 60.99 | 52.53 | 26.60 | 78.46 | 0.5618 | 0.2326 | 23.25 | 33.67 |
| pooled | T24_MSR_V1a | 113.954 | 60.46 | 53.63 | 32.72 | 74.55 | 0.5682 | 0.2359 | 27.90 | 33.67 |
| pooled | T24_MSR_V1b | 114.117 | 59.93 | 53.40 | 33.39 | 73.40 | 0.5664 | 0.2367 | 28.88 | 33.67 |
| 2026-05 | DIRECT_LGBM | 99.544 | 60.79 | 56.09 | 44.91 | 67.28 | 0.5459 | 0.2493 | 36.26 | 29.03 |
| 2026-05 | PERIODIC_MSR_V1a | 99.536 | 61.93 | 53.21 | 32.41 | 74.01 | 0.5450 | 0.2348 | 27.86 | 29.03 |
| 2026-05 | PERIODIC_MSR_V1b | 99.602 | 62.33 | 53.90 | 33.80 | 74.01 | 0.5542 | 0.2320 | 28.26 | 29.03 |
| 2026-05 | T24_MSR_V1a | 99.681 | 60.15 | 52.46 | 34.14 | 70.79 | 0.5469 | 0.2370 | 30.65 | 29.03 |
| 2026-05 | T24_MSR_V1b | 99.825 | 59.58 | 52.23 | 34.72 | 69.74 | 0.5442 | 0.2381 | 31.55 | 29.03 |
| 2026-06 | DIRECT_LGBM | 116.706 | 49.13 | 50.04 | 53.76 | 46.31 | 0.5246 | 0.2599 | 53.72 | 37.85 |
| 2026-06 | PERIODIC_MSR_V1a | 115.989 | 56.67 | 51.92 | 32.39 | 71.45 | 0.5514 | 0.2429 | 30.00 | 37.85 |
| 2026-06 | PERIODIC_MSR_V1b | 116.237 | 56.74 | 51.76 | 31.28 | 72.23 | 0.5584 | 0.2407 | 29.10 | 37.85 |
| 2026-06 | T24_MSR_V1a | 114.681 | 56.28 | 53.50 | 42.02 | 64.97 | 0.5690 | 0.2462 | 37.67 | 37.85 |
| 2026-06 | T24_MSR_V1b | 114.772 | 55.87 | 53.54 | 43.94 | 63.13 | 0.5627 | 0.2475 | 39.55 | 37.85 |
| 2026-07 | DIRECT_LGBM | 132.426 | 60.55 | 49.93 | 16.18 | 83.69 | 0.5617 | 0.2283 | 16.26 | 34.27 |
| 2026-07 | PERIODIC_MSR_V1a | 129.716 | 63.64 | 52.19 | 15.78 | 88.60 | 0.5840 | 0.2249 | 12.90 | 34.27 |
| 2026-07 | PERIODIC_MSR_V1b | 129.927 | 63.78 | 52.22 | 15.49 | 88.96 | 0.5788 | 0.2254 | 12.57 | 34.27 |
| 2026-07 | T24_MSR_V1a | 127.525 | 64.82 | 54.47 | 21.57 | 87.37 | 0.5859 | 0.2249 | 15.69 | 34.27 |
| 2026-07 | T24_MSR_V1b | 127.775 | 64.21 | 53.87 | 20.98 | 86.76 | 0.5895 | 0.2247 | 15.89 | 34.27 |

## Gate 判定

开发窗冻结主版本：T24_MSR_V1b，不因回顾结果改选 V1a。

Gate 明细：`{"balanced_gain": true, "auc_gain": true, "positive_recall": false, "negative_recall": true, "recall_improvement": true, "mae": true, "brier": true, "losing_months": true, "worst_month": true, "no_class_collapse": true}`。唯一失败项是正类召回损失。
正类召回 33.39% vs 38.30%，下降 -4.91 个百分点；Gate 要求至少 -2 个百分点。Balanced 增益 1.07 个百分点、AUC 增益 0.0210，不足以豁免正类损失。
全负方向基线 66.33%，不得把主版本 Direction 当作超过多数类基线的突破。

## 组件定位

| 月份 | Low MAE | Low 相关 | State balanced | State POS recall | Z MAE(MW) | Z AUC | High MAE | 正 crossing 恢复率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-05 | 90.4285 | 0.0643 | 0.4373 | 0.1448 | 2083.5126 | 0.7116 | 49.8924 | 0.3593 |
| 2026-06 | 104.2684 | 0.0816 | 0.3954 | 0.0831 | 2471.7971 | 0.6438 | 52.8951 | 0.2654 |
| 2026-07 | 118.8336 | 0.2020 | 0.3989 | 0.0000 | 3014.4502 | 0.6066 | 52.5154 | 0.1685 |

Low/State：平滑后置信度高但正类识别弱，需区分 Low 中位数偏差与 Markov 过度持续/校准问题。High：尾部误差显著大于主体，正 crossing 恢复率随月走弱。Z：物理正负排名具有信号，但加入重构未转化为收益。补充诊断保存 ramp/tightness/Z/预测 Low regime 条件误差、状态 dwell/Brier、最终概率校准和 T24 年/月稳定性；这些分箱仅为描述，不参与选择或拟合。

重构继续采用冻结有限 quantile CDF/耦合近似；有限分位数、尾部截平及依赖近似的概率限制未被新的预测模型解决，不把其数值当作精确分布或因果效应。

## Z 增量（V1b−V1a）

`{"pooled": {"balanced_accuracy": -0.002356200534368247, "auc_prob_positive": -0.0018201446794031684, "positive_recall": 0.00672494956287828, "negative_recall": -0.011437350631614884, "brier_prob_positive": 0.0007195259919183516, "mae": 0.16259633958608788}, "2026-05": {"balanced_accuracy": -0.0023148148148148806, "auc_prob_positive": -0.0027428144290123635, "positive_recall": 0.005787037037037035, "negative_recall": -0.01041666666666674, "brier_prob_positive": 0.0010710247944538398, "mae": 0.14438091097554206}, "2026-06": {"balanced_accuracy": 0.0004151504279636953, "auc_prob_positive": -0.0062354569217364375, "positive_recall": 0.019266055045871533, "negative_recall": -0.018435754189944142, "brier_prob_positive": 0.0013352423263322255, "mae": 0.0910222464842434}, "2026-07": {"balanced_accuracy": -0.006008661133164894, "auc_prob_positive": 0.0035456513893901143, "positive_recall": -0.0058823529411764774, "negative_recall": -0.006134969325153339, "brier_prob_positive": -0.00022782732779177972, "mae": 0.250077019585504}}`

Z_MEDIATOR_RETRO=NOT_SUPPORTED：pooled Balanced/AUC 下降，Brier/MAE 上升，不因 Z 自身 AUC>0.5 而宣称最终增量成立。

## 停止与下一步

READY_FOR_FRESH_CONFIRM=NO
LATE_SEPTEMBER_METRICS_OPENED=NO
OCTOBER_LOCKBOX_ACCESSED=NO

最终判定：MSR_T24_NOT_SUPPORTED_ON_RETROSPECTIVE_READINESS；这是回顾 readiness 失败，不是未打开的 fresh-confirm 失败。保留 late September holdout。

NEXT_PATCH_RECOMMENDATION：先联合审定一个单轴 Low/State 正类偏差与平滑校准排查补丁，并以正 crossing 恢复为机制诊断；分别定位 Low 中位数、Markov 置信度及 High 修正的贡献，不自动调 kappa/阈值/损失/分解，不增加架构，不自动推广 Z。

## 证据

- 成功链：`outputs/experiments/msr_spread_v1/t24_controlled_rerun_20261009/controlled_rerun/20261009T025637339105Z/`
- 原始完整报告与五组预测：`outputs/experiments/msr_spread_v1/t24_controlled_rerun_20261009/report/20261009T025637339105Z/`
- 追加闭环：`outputs/experiments/msr_spread_v1/t24_controlled_rerun_20261009/report/20261009T025637339105Z_closure/`
- 测试日志：repair/FULL_TESTS_IO_FIXED.log；修复封存：repair/20261009T025635Z/REPAIR_GATE.json。

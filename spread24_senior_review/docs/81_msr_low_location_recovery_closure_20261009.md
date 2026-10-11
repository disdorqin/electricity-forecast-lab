# 81 — FINAL_REPORT: MSR Rolling Low Location Recovery 闭环

STATUS=STOP_AT_D2C_APRIL_TRANSFER
DATE=2026-10-09
EXECUTION_AUTHORITY=docs/80
RUN_ID=20261009T063119153421Z

## 1. 最终判定与验收 tokens

**March selection PASS，冻结 ROLL28_GLOBAL；April transfer FAIL，按合同停止。** 工程测试/chronology 与数值实现通过，不等于科学迁移成立。未生成 OOF active State、未训练 High、未评估 May–July Spread，也未打开 late September。

```ini
INDEPENDENT_REVIEW=PASS
LOW_REUSE_MIGRATION_GATE=PASS
ROLLING_LOCATION_IMPLEMENTATION=PASS
ROLLING_DAYS=28
LOCATION_CANDIDATES=RAW_LOCATION,ROLL28_GLOBAL,ROLL28_BLOCK3
LOCATION_SELECTION=ROLL28_GLOBAL
LOCATION_SELECTION_USED_JAN_FEB=NO
LOCATION_SELECTION_USED_APRIL=NO
LOCATION_SELECTION_USED_FINAL_SPREAD=NO
LOCATION_CALIBRATION_TRANSFER=FAIL
STATE_KAPPA=0.5
STATE_PERSISTENCE=0.0
OOF_STATE_REGEN=NOT_RUN
HIGH_V1A_TRAIN=NOT_RUN
RECONSTRUCTION=NOT_RUN
RETRO_MAY=NOT_RUN
RETRO_JUNE=NOT_RUN
RETRO_JULY=NOT_RUN
LOW_LOCATION_CORRECTION_SPREAD_TRANSFER=NOT_EVALUATED
READY_FOR_FRESH_CONFIRM=NO
CONFIRM_SEAL=NOT_RUN
LATE_SEPTEMBER_METRICS_OPENED=NO
OCTOBER_LOCKBOX_ACCESSED=NO
PRODUCTION_WRITE=NO
FULL_TESTS=341/341
```

## 2. 单变量与输入时间线

- Origin=D−1 14:00；所有 rolling residual labels <=D−2；history=672；T24/kappa=0.5 不变。
- 唯一科学变量为 RAW_LOCATION / ROLL28_GLOBAL / ROLL28_BLOCK3；rolling_days=28，不搜索窗口。
- `e(d,t)=L_true(d,t)−Q_RAW(q50,d,t)`；同一目标 slot 的五个 quantile 共同加 b(D,t)，不能反馈此前 adjusted q50。
- 取最新 28 个完整成熟日；完整日要求唯一 p1–96、五个 raw quantile 与 L_true finite。历史 OOF 可 14–27 日暖启动，<14 日拒绝；推理必须满 28 日。
- 历史 training correction 只能用 train/oof/fold>0；Jan–Apr residual history 使用冻结 out-of-sample `pred_kind=fitted` Low。它并非对 2026 标签的 in-sample fitting。
- Jan–Feb 仅作为成熟残差来源；选择指标仅 March 1–31；April 1–28 只有已封存 family 的 transfer，未查看 BLOCK3 April 指标。
- State persistence 全程 0.0；original uncalibrated finite-ladder CDF；无 GLOBAL_PIT/BLOCK_PIT_3；不训练 Low 或 Z。
- April half split 在查看科学结果前固定为 1–14 /15–28，March 为 1–15 /16–31；完整配置先写 preflight/PREREGISTERED_CONFIGURATION.json。
- 完整回归使用 synthetic/已消费 development fixture，不能参与 family 选择；新的 March 科学 reader 投影/filter 到 March31，April 科学 reader 必须先验证 supported selection seal。
- 保留原数据 historical proxy/vintage 限制；chronology PASS 不等于已证明真实 historical origin-time capture，结果不作 production promotion。

## 3. Low 只读复用与冻结对象

LOW_REUSE_MIGRATION_GATE=PASS。只读验证 COMPLETE Low manifest、prediction hash、canonical source、T24 算子、科学源码、Low feature/config、LightGBM/seed、完整行键。
- Raw Low 125,568 keys /1,308完整96-slot日，迁移检查仅读键，未加载 May+ 标签。
- Raw Low SHA: `f7b43c1b428d9488888560e7e4f8e08f9adfe92daf84361a2c2705e4b3259050`。
- Source SHA: `49fade43b0f3505a437e8704377bf5ef07f780429f9052caaca74aa3525e95a2`。
- T24 SHA: `7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7`，lambda=215.32246474604284，未重搜/重分解。
- 五 quantile levels=.10/.25/.50/.75/.90；RAW artifact 不写回；所有宽度/排序核查通过（float64 roundoff tolerance，仅 correctness，不是科学 Gate 放宽）。
- State delta/transition 参数仅使用同一合法 train history <=2025-12-30；kappa=0.5；本轮不平滑。

## 4. March selection：31 日 /2,976 slots

| Candidate | q50 coverage | coverage error | Low q50 MAE | Low q50 RMSE | residual mean | residual median | State Brier | Balanced | NEG recall | NEU recall | POS recall | admissible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RAW_LOCATION | 0.285282 | 0.214718 | 93.014724 | 152.257613 | 34.765724 | 25.508661 | 0.699241 | 0.411655 | 0.642055 | 0.519084 | 0.073826 | False |
| ROLL28_GLOBAL | 0.432796 | 0.067204 | 88.510072 | 149.553369 | 17.420157 | 7.143127 | 0.660895 | 0.447237 | 0.481541 | 0.568702 | 0.291467 | True |
| ROLL28_BLOCK3 | 0.419691 | 0.080309 | 88.677764 | 148.744733 | 17.394417 | 9.441802 | 0.664973 | 0.428674 | 0.460674 | 0.592366 | 0.232982 | True |

两非 RAW family 均通过全部七项 March Gate；GLOBAL 的 State Brier 更低（与 BLOCK 相差 0.004078，亦处于简单 family tie 优先规则内）。RAW 只是对照，不是校准成功。
GLOBAL q50 coverage error 0.214718→0.067204，改善 0.147513，满足改善 >=.05 的 OR 条件，虽最终 error 仍略大于 .05；没有改 Gate。
GLOBAL POS recall 上升 21.7641pp，NEG recall 下降 16.0514pp，Balanced 增加 3.5582pp；不能把 POS 召回改善当成“两类同时改善”。

| Candidate | q10 | q25 | q50 | q75 | q90 | POS threshold above q90 | NEG threshold below q10 | either outside | crossing rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW_LOCATION | 0.121304 | 0.163306 | 0.285282 | 0.558804 | 0.753696 | 0.195228 | 0.099126 | 0.278226 | 0.000000 |
| ROLL28_GLOBAL | 0.157930 | 0.243952 | 0.432796 | 0.674059 | 0.802755 | 0.085685 | 0.257056 | 0.313508 | 0.000000 |
| ROLL28_BLOCK3 | 0.159274 | 0.244960 | 0.419691 | 0.666667 | 0.800403 | 0.065860 | 0.257056 | 0.300739 | 0.000000 |

### March 逐 Gate 与半月结果

| Family | q50 coverage | State Brier | POS recall/prevalence | Balanced | no collapse | halves Brier | finite/normalized/monotone |
|---|---|---|---|---|---|---|
| ROLL28_GLOBAL | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| ROLL28_BLOCK3 | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

| Candidate | March1–15 Brier | March16–31 Brier |
|---|---:|---:|
| RAW_LOCATION | 0.677844 | 0.719301 |
| ROLL28_GLOBAL | 0.654857 | 0.666556 |
| ROLL28_BLOCK3 | 0.649843 | 0.679157 |

### Selection SHA seal（先于 April 科学 reader）

Family=`ROLL28_GLOBAL`；selection SHA256=`44a506d7d7326301be3ca902723bccea22059f649035480883564796d0ad4738`；sealed_utc=`2026-10-09T06:43:02.423335+00:00`。
后续只允许同 family、同 28 日、同 maturity、同 block routing；rolling shift 按每个 D 的合法历史更新，但不是重选 family。

## 5. April fixed-family transfer：28 日 /2,688 slots

| Arm | q50 coverage | coverage error | Low MAE | Low RMSE | residual mean | residual median | State Brier | Balanced | NEG recall | NEU recall | POS recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW_LOCATION | 0.477307 | 0.022693 | 88.686809 | 172.925832 | -14.778934 | 2.217785 | 0.637406 | 0.361765 | 0.381013 | 0.565089 | 0.139194 |
| ROLL28_GLOBAL | 0.590030 | 0.090030 | 91.093757 | 176.240188 | -28.913654 | -12.034461 | 0.678207 | 0.344485 | 0.253165 | 0.507396 | 0.272894 |

### 四项失败，不能用 POS recall 单独越过 Gate

| Gate | Requirement | Observed | Result |
|---|---|---|---|
| Low coverage | error 不比 RAW 差且 <=.10，或改善 >=.03 | .022693→.090030，恶化 .067336 | FAIL |
| State Brier | 相对 RAW 改善 >=.005 | .637406→.678207，恶化 .040801 | FAIL |
| POS recall/prevalence | recall >=+.02 或 error下降20% | recall .139194→.272894，+13.3700pp | PASS |
| Balanced | delta >=−.005 | .361765→.344485，−.017280 | FAIL |
| no collapse | RAW非零recall不得变0 | 各类recall仍非零 | PASS |
| each half Brier | delta <=+.02 | 前半月 +.080160；后半月 +.001442 | FAIL |
| finite/normalized/monotone | 全部有效 | 无非有限/归一化/排序失败 | PASS |

尤其不能因 adjusted error=.09003 小于 .10 就宣布 coverage Gate PASS：合同同时要求不比 RAW 差，或绝对改善 >=.03。这里两分支均不满足。

| Arm | April1–14 Brier | April15–28 Brier | POS above q90 | NEG below q10 | either outside |
|---|---:|---:|---:|---:|---:|
| RAW_LOCATION | 0.643479 | 0.631333 | 0.184524 | 0.247768 | 0.337426 |
| ROLL28_GLOBAL | 0.723640 | 0.632775 | 0.088170 | 0.339658 | 0.364583 |

April corrected POS recall 提升，但 NEG recall 38.1013%→25.3165%（−12.7848pp），NEU 亦下降；Balanced/Brier 不支持迁移成功。
April raw residual median仅 +2.217785；使用过去28日残差形成的每日正向 shift 均值为 +14.134720（前半 19.723023、后半 8.546416）。corrected residual median转为 −12.034461，q50 coverage超过 nominal。与位置漂移变化/校正滞后兼容，不作因果证明，不据此搜索新窗口。

## 6. State 三类、概率与 temporal diagnostics（非 Spread 指标）

| Window | Family | pred NEG | pred NEU | pred POS | true NEG | true NEU | true POS | accuracy | mean max prob | entropy | transitions/day | ABA/day | dwell median | dwell mean | dwell P90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| March | RAW_LOCATION | 0.488575 | 0.457997 | 0.053427 | 0.209341 | 0.440188 | 0.350470 | 0.388777 | 0.544580 | 0.957944 | 8.354839 | 2.419355 | 3.000000 | 10.262069 | 31.000000 |
| March | ROLL28_GLOBAL | 0.279570 | 0.457997 | 0.262433 | 0.209341 | 0.440188 | 0.350470 | 0.453293 | 0.538144 | 0.957678 | 9.612903 | 2.903226 | 3.000000 | 9.045593 | 29.000000 |
| March | ROLL28_BLOCK3 | 0.301747 | 0.490255 | 0.207997 | 0.209341 | 0.440188 | 0.350470 | 0.438844 | 0.534592 | 0.962127 | 10.193548 | 3.290323 | 3.000000 | 8.576369 | 26.400000 |
| April | RAW_LOCATION | 0.347470 | 0.526786 | 0.125744 | 0.293899 | 0.502976 | 0.203125 | 0.424479 | 0.557798 | 0.936893 | 9.357143 | 2.964286 | 3.000000 | 9.268966 | 29.000000 |
| April | ROLL28_GLOBAL | 0.236607 | 0.477679 | 0.285714 | 0.293899 | 0.502976 | 0.203125 | 0.385045 | 0.550737 | 0.943754 | 8.428571 | 2.178571 | 4.000000 | 10.181818 | 30.000000 |

三态 majority accuracy：March NEU=44.0188%；April NEU=50.2976%；常量单类 Balanced=1/3（全部三类存在）。不把 State plain accuracy 当最终 Spread Direction。dwell只是描述，lambda_state=0且未搜索。
Original finite-ladder 在外侧保留约 .1 概率 floor；本轮没有 ECDF把 POS tail 压成0。逐 residual B1/B2/B3、q10..q90、原始半月/temporal全量值见每候选 METRICS.json 与原始 report/FINAL_REPORT.md。

## 7. 工程正确性、独立重算、保护核查

- 新模块 `rolling_low_location.py`：固定28日 residual median、RAW identity、block routing、共同 shift、chronology audit 与 correction/state fingerprints。
- 新模块 `low_location_recovery.py`：D2A/D2B/D2C 分命令、tests/migration/seal Gate、按首个失败科学 Gate 输出 NOT_RUN。通用 pipeline 科学 guard 未解锁。
- 新测试 `test_rolling_location80.py`：38 项；完整 MSR **341/341**。唯一既有 Python 修改是 startup reads 测试断言15→17，匹配用户已更新 AGENTS，模型源码未改。
- D2A CLI 为 synthetic 未来扰动 chronology audit + 真实 Low migration；D2B/D2C 为真实契约科学入口，全部 exit_code=0；停止是科学结果，不是运行错误。
- 独立重算：另用保存 raw Low residual 显式 median，逐31/28个D校验日期/96 slots/shift/全部五 quantile/width；Brier、recall、coverage、half Brier、Gate重算均PASS。独立指独立数学/产物验证，不宣称外部人员审核。
- 该轮所有 correction/state 逐日期 fingerprint 包含 family、28d、history cutoff/keys/raw predictions/residual SHA、Low/T24 identity、kappa=.5、lambda=0；High接口拒绝混用身份。

| Protected group | File SHA checks | Result |
|---|---:|---|
| src/TafM_改进源码 | 9266 | PASS |
| src/lightgbm_源码 | 15 | PASS |
| outputs/experiments/msr_spread_v1/state_only_experiment_20261009 | 95 | PASS |
| historical | 800 | PASS |
| closed_docs | 3 | PASS |
| docs78_prior_run | 110 | PASS |
| current_authority_and_closures | 4 | PASS |

production 2.5 metadata-only stat 对比 94423 文件：changed/added/missing全0。未读取 production 目标标签、无 runtime dependency、无 production 写入；该 stat 口径不冒充内容 SHA 验证。未知工作区修改保留原状，无 Git 写操作，无冻结目录缓存清理。

## 8. NOT_RUN 精确原因

上游停止原因：**docs/80 §7 / §17 Branch B：March-frozen ROLL28_GLOBAL 未通过 April coverage、State Brier、Balanced、half-Brier 四 Gate。**

| Downstream | Status | Exact reason |
|---|---|---|
| D2-D OOF correction/state artifacts | NOT_RUN | April transfer 未 PASS |
| D2-E UNCAL_RAW/ROLLCAL_RAW High训练 | NOT_RUN | 未获准 D2-D，April transfer 未 PASS |
| D2-F reconstruction | NOT_RUN | High训练未运行，April transfer 未 PASS |
| May/June/July Spread/High/crossing/slot metrics | NOT_RUN | 上游 transfer失败，未形成合法 active High arms |
| upstream correction Spread transfer / Direct readiness | NOT_EVALUATED | 没有最终 Spread 科学结果 |
| late-Sep feature view / predictions / seal / truth open | NOT_RUN | READY_FOR_FRESH_CONFIRM=NO（April Gate失败） |
| April BLOCK3 | NOT_RUN | March封存family，不得用April切换 |
| Jan-Feb selection / PIT/lambda/kappa/window/structure search | NOT_RUN | 非本轮授权轴 |

未创建 d2d_oof_state/d2e_high/d2f_reconstruction/retrospective/fresh_confirm 目录，未编造 downstream FAIL/PASS 或引用旧 Spread 数字替代本轮。October全程禁止且未读。

## 9. NEXT_PATCH_RECOMMENDATION

March 的合法 28-day location correction 有局部 Low/State 增益，但未通过 April 固定 family transfer；不能宣布机制稳定。优先复核 Low 位置漂移回归与 rolling residual 的滞后，以及 finite-ladder 外侧概率限制，交由下一份预注册协议决定唯一轴（有限-tail 表示或 Low 信号重设计）。本轮不切换 GLOBAL/BLOCK、不搜索 7/14/56-day window、不增加 quantiles/tail model、不启用 persistence/Z、不改 High。

**最终科学判定：合法滚动 Low 位置修复在 March 有局部支持，但本轮固定28日 GLOBAL correction 未通过 April transfer，未证明跨月稳定的 State recovery。不得对 High/final Spread/确认窗做未经执行的结论。**

## 10. 复现与交付资产

```powershell
python -B -m pytest src/research96/msr_spread_v1/tests -q -p no:cacheprovider --basetemp=.roll80_full_0631
python -B -m src.research96.msr_spread_v1.low_location_recovery --stage D2A --out <versioned_run>
python -B -m src.research96.msr_spread_v1.low_location_recovery --stage D2B --out <versioned_run>
python -B -m src.research96.msr_spread_v1.low_location_recovery --stage D2C --out <versioned_run>
```

实际 absolute epf-2 Python 路径、out路径、退出状态和 stdout SHA 在 CLI_D2A/B/C_RECEIPT.json；完整源代码保存在 source_snapshot/。
Code fingerprint=`9b029e7f21f985471e50c17055ed78a4ad0e94decea3c534e66140618edb39e9`；config fingerprint=`e58bafa1dc790482c8b5fe92b1fb374d51fcd7b83fddf7a3a500d55d0a62b9f2`。
原始大JSON stop report 为 run/report/FINAL_REPORT.md，保持其 manifest 不变；本最终报告追加易读统计、故障定位和保护核查。完整 run index / manifests / final integrity receipt 在 run 根。

RUN_ROOT=D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\outputs\experiments\msr_spread_v1\low_location_recovery_20261009\20261009T063119153421Z

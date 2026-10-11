# FINAL_REPORT — docs/78 Low-CDF / Unary Calibration 闭环

STATUS=SCIENTIFIC_STOP_D1B_NO_SUPPORTED_CALIBRATOR
DATE=2026-10-09
RUN_ID=20261009T054445579001Z

## 1. 最终判定

工程 Gate 全部通过；科学 Gate 在 **D1-B** 失败，按合同立即停止。两个非 RAW 校准器都使 pooled Brier 更差，POS recall 未恢复，一月 Brier 恶化超过 +0.02。RAW 只是对照，不能被宣布为校准成功。
**没有执行 D1-C / D1-D / OOF-State 再生 / High / reconstruction / May–July / fresh confirm。** 这些项目是 NOT_RUN，不是假造 FAIL。

```ini
INDEPENDENT_REVIEW=PASS
LOW_REUSE_MIGRATION_GATE=PASS
D1A_CDF_DIAGNOSIS=CALIBRATION_DOMINANT
CDF_CALIBRATOR_CANDIDATES=RAW,GLOBAL_PIT,BLOCK_PIT_3
CDF_CALIBRATION_SELECTION=NO_SUPPORTED_CALIBRATOR
CDF_SELECTION_USED_MAR_APR=NO
CDF_SELECTION_USED_FINAL_SPREAD=NO
CDF_CALIBRATION_TRANSFER=NOT_RUN
STATE_KAPPA=0.5
STATE_CANDIDATES=0,0.25,0.5,0.75,1.0
SELECTED_STATE_PERSISTENCE=NOT_RUN
PERSISTENCE_SCIENTIFIC_CONCLUSION=NOT_RUN
OOF_STATE_REGEN=NOT_RUN
HIGH_V1A_TRAIN=NOT_RUN
RECONSTRUCTION=NOT_RUN
RETRO_MAY=NOT_RUN
RETRO_JUNE=NOT_RUN
RETRO_JULY=NOT_RUN
CDF_CALIBRATION_SPREAD_TRANSFER=NOT_EVALUATED
PERSISTENCE_TRANSFER=NOT_EVALUATED
PRIMARY_VARIANT=NONE
READY_FOR_FRESH_CONFIRM=NO
CONFIRM_SEAL=NOT_RUN
LATE_SEPTEMBER_METRICS_OPENED=NO
OCTOBER_LOCKBOX_ACCESSED=NO
PRODUCTION_WRITE=NO
FULL_TESTS=303/303
```

## 2. 合同、信息时间线与 Low 只读迁移

最高执行合同为 docs/78；已完整阅读 AGENTS.md、docs/63、74、76、77、78。docs/63 的未替代模型契约与实验室隔离边界继续生效；旧通用科学 CLI 不解锁。
- Target Spread=RT−DA；96 slots/day；history=672；origin=D−1 14:00；监督标签最多 D−2。
- 新科学读取只涉及 train 2023-01-01..2025-12-30 与 Jan–Feb 2026。2026 dev 标签绝不拟合校准参数。
- 校准 fit 使用有合法 chronological OOF 来源的 68,352 行，首个应用日 2026-01-01，最大成熟标签 2025-12-30。BLOCK 三段各 22,784 行。
- Low key-only 迁移核查：125,568 rows / 1,308 完整 96-slot 日；仅读取 target_day/slot，无标签加载。
- Low、High、Z、quantile levels、features、loss、reconstruction、seed、LightGBM 参数未修改；所有既有 package Python 源码逐文件 SHA 不变，仅新增三个文件。
- Low SHA256: `f7b43c1b428d9488888560e7e4f8e08f9adfe92daf84361a2c2705e4b3259050`。
- Source SHA256: `49fade43b0f3505a437e8704377bf5ef07f780429f9052caaca74aa3525e95a2`。
- T24 operator SHA256: `7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7`；lambda=215.32246474604284；kappa=0.5。
- 冻结 quantiles=0.10/0.25/0.50/0.75/0.90；不改变 Low ladder 数值。
- 完整测试使用 synthetic 或已消费 development fixture；没有生成 Mar–Apr 新科学比较或 May–July 新指标。Opaque SHA 和 production metadata 扫描不打开目标指标。

## 3. 工程实现与验证

- 新增 `cdf_unary.py`：CDF/PIT、threshold position、GLOBAL/BLOCK ECDF、校准 unary、combined semantic fingerprint、High 混用拒绝。
- 新增 `low_cdf_state_recovery.py`：D1A/B/C/D 独立命令与前序 seal Gate；NO_SUPPORTED_CALIBRATOR 禁止 Mar–Apr reader。未预先实现或运行未获准的下游训练。
- 新增 `tests/test_cdf_unary78.py`：35 项新测试，涵盖 docs/78 §17 的概率、chronology、hash、routing、保护及 guard 接口。完整 suite **303/303**。
- 独立复核为独立数学参考/保存预测的结果重算，不宣称外部人员审核：np.interp 参考、显式 PIT count 参考、未来标签扰动、逐概率 Brier/recall/confusion 重算、冻结 Gate 重算均 PASS。
- 真实 CLI D1A 与 D1B 均 exit_code=0，运行命令、完整 stdout 和 SHA receipt 原样保留。
- RAW 严格委托旧 raw_regime_probs；probability normalization、NEG 严格边界与 NEU 含阈值、三块固定 routing 均测试通过。
- 下游 nested map 必须 strictly before fold D−2 cutoff；空合法历史拒绝，不用未来 fit fallback；combined fingerprint 绑定 map/family/cutoff/kappa/lambda。

## 4. D1-A：Jan–Feb Low / PIT / threshold diagnosis

CDF_DIAGNOSIS=CALIBRATION_DOMINANT
该描述 token 的 operationalization 在结果读取前写入代码与 SCIENTIFIC_AUTHORIZATION：两条 threshold 合计的外侧率 >=20% 为 material，q50 coverage error>5pp 或 POS prevalence error>10pp 为 bias；不是新增科学 Gate。
**CALIBRATION_DOMINANT 不代表有限尾部问题可忽略**：合计外侧 threshold 比例 19.2090% 接近描述边界，任一 threshold 外侧的行占 33.4393%，POS threshold above q90 为 29.4138%。本轮不据此反向改变描述阈值或候选。

| Group | Rows | q10 coverage | q25 | q50 | q75 | q90 | residual mean | residual median | PIT mean | PIT var | KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled | 5664 | 0.128531 | 0.180438 | 0.328566 | 0.634004 | 0.794845 | 23.658466 | 19.347656 | 0.588212 | 0.076427 | 0.174145 |
| B1 | 1888 | 0.136653 | 0.189089 | 0.358581 | 0.731462 | 0.886653 | 4.968444 | 12.469443 | 0.552348 | 0.068700 | 0.142132 |
| B2 | 1888 | 0.108051 | 0.142479 | 0.235699 | 0.495763 | 0.674788 | 53.970254 | 41.030546 | 0.655956 | 0.074317 | 0.272858 |
| B3 | 1888 | 0.140890 | 0.209746 | 0.391419 | 0.674788 | 0.823093 | 12.036701 | 14.498401 | 0.556331 | 0.079371 | 0.116183 |
| 2026-01 | 2976 | 0.144489 | 0.180108 | 0.263441 | 0.577285 | 0.739247 | 30.260464 | 30.251963 | 0.620802 | 0.079622 | 0.242837 |
| 2026-02 | 2688 | 0.110863 | 0.180804 | 0.400670 | 0.696801 | 0.856399 | 16.349112 | 9.264569 | 0.552129 | 0.070410 | 0.102412 |

PIT 10 bins 固定 [0,.1,...,1]，右端最后一格含 1；pooled counts：`728,154,281,314,384,609,804,685,543,1162`。逐 block/月 bins 完整保存 PIT_HISTOGRAM.csv。KS 仅距离诊断，无 p-value，不作为候选选择。

| Threshold metric | Rate |
|---|---:|
| POS_THRESHOLD_ABOVE_Q90_RATE | 0.29413842 |
| NEG_THRESHOLD_BELOW_Q10_RATE | 0.09004237 |
| flat_outer_tail_threshold_fraction | 0.19209040 |
| either_threshold_outside_rate | 0.33439266 |
| NEG_strict_boundary_CDF_flat_tail_rate | 0.09004237 |

位置端点：[q10,q25]、(q25,q50]、(q50,q75]、(q75,q90]；below/above 为严格不等号；NEG CDF 使用 nextafter(-delta,−inf) 独立报告 strict boundary 外侧率。

| Group | Threshold | below_q10 | q10_q25 | q25_q50 | q50_q75 | q75_q90 | above_q90 |
|---|---|---:|---:|---:|---:|---:|---:|
| pooled | NEG | 0.090042 | 0.132768 | 0.324682 | 0.441031 | 0.011476 | 0.000000 |
| pooled | POS | 0.000000 | 0.000000 | 0.000883 | 0.227048 | 0.477931 | 0.294138 |
| B1 | NEG | 0.028072 | 0.151483 | 0.301907 | 0.517479 | 0.001059 | 0.000000 |
| B1 | POS | 0.000000 | 0.000000 | 0.002648 | 0.356992 | 0.467691 | 0.172669 |
| B2 | NEG | 0.180085 | 0.074682 | 0.174788 | 0.537076 | 0.033369 | 0.000000 |
| B2 | POS | 0.000000 | 0.000000 | 0.000000 | 0.084216 | 0.461335 | 0.454449 |
| B3 | NEG | 0.061970 | 0.172140 | 0.497352 | 0.268538 | 0.000000 | 0.000000 |
| B3 | POS | 0.000000 | 0.000000 | 0.000000 | 0.239936 | 0.504767 | 0.255297 |

### Training OOF reference（各自合法 fold-local delta/transition）

| Fold | Legal cutoff | q50 coverage | residual mean | PIT mean | POS above q90 |
|---|---|---:|---:|---:|---:|
| 1 | 2024-01-17 | 0.530203 | -5.620344 | 0.475916 | 0.269029 |
| 2 | 2024-07-14 | 0.581519 | -13.276657 | 0.449004 | 0.200433 |
| 3 | 2025-01-08 | 0.487886 | 3.702341 | 0.488823 | 0.214068 |
| 4 | 2025-07-05 | 0.524423 | -21.338455 | 0.464114 | 0.406015 |

训练 OOF q50 coverage 48.79%–58.15%，而 Jan–Feb 为 32.86%；该反差提示分布/可迁移性问题，但不能仅靠这一次诊断证明 Low 的信息上限。

## 5. D1-B：唯一三候选、Jan–Feb State 选择

确定性 finite sample convention：`G(u)=count(training PIT<=u)/n`，exact right-continuous ECDF，包含所有 ties；无 interpolation/tail fit。RAW identity 不拟合。GLOBAL 一个 map；BLOCK 恰好 p1–32/p33–64/p65–96 三个 map。

| Candidate | Brier | Balanced | NEG recall | NEU recall | POS recall | predicted POS | POS prevalence abs error | mean max prob | entropy | dwell median | transitions/day | admissible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RAW | 0.636436 | 0.435794 | 0.778075 | 0.529308 | 0.000000 | 0.013771 | 0.222105 | 0.572772 | 0.929274 | 4.000000 | 6.559322 | False |
| GLOBAL_PIT | 0.657008 | 0.436791 | 0.781417 | 0.528955 | 0.000000 | 0.008121 | 0.227754 | 0.598595 | 0.848923 | 5.000000 | 6.440678 | False |
| BLOCK_PIT_3 | 0.651223 | 0.442642 | 0.789439 | 0.538489 | 0.000000 | 0.007592 | 0.228284 | 0.596025 | 0.851654 | 5.000000 | 6.355932 | False |

Truth NEG/NEU/POS=26.4124%/50.0000%/23.5876%；truth dwell median=15 slots、transitions/day=4.203390。三状态 NEU-majority Accuracy=50%，Balanced=1/3；这里是 Low-State 指标，不是最终 Spread Direction/AUC。RAW predicted NEG/NEU/POS=60.7521%/37.8708%/1.3771%。

| Candidate | Month | Brier | Balanced | NEG recall | NEU recall | POS recall | pred NEG | pred NEU | pred POS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW | 2026-01 | 0.696527 | 0.414752 | 0.801598 | 0.442659 | 0.000000 | 0.691196 | 0.307124 | 0.001680 |
| RAW | 2026-02 | 0.569908 | 0.454917 | 0.754362 | 0.610390 | 0.000000 | 0.514881 | 0.457961 | 0.027158 |
| GLOBAL_PIT | 2026-01 | 0.727125 | 0.412419 | 0.790945 | 0.446311 | 0.000000 | 0.688508 | 0.310820 | 0.000672 |
| GLOBAL_PIT | 2026-02 | 0.579378 | 0.459367 | 0.771812 | 0.606288 | 0.000000 | 0.531994 | 0.451637 | 0.016369 |
| BLOCK_PIT_3 | 2026-01 | 0.722755 | 0.417316 | 0.797603 | 0.454346 | 0.000000 | 0.684476 | 0.315188 | 0.000336 |
| BLOCK_PIT_3 | 2026-02 | 0.572028 | 0.466144 | 0.781208 | 0.617225 | 0.000000 | 0.524554 | 0.459821 | 0.015625 |

### 科学 Gate 逐项判定

| Candidate | Brier improvement >=.01 | POS recall gain / prevalence reduction | Balanced >=−.005 | no class collapse | monthly Brier <=+.02 | finite normalized |
|---|---|---|---|---|---|---|
| GLOBAL_PIT | FAIL | FAIL | PASS | PASS | FAIL | PASS |
| BLOCK_PIT_3 | FAIL | FAIL | PASS | PASS | FAIL | PASS |

- GLOBAL_PIT: pooled Brier improvement=-0.02057161（为负）；January Brier delta=+0.03059811；February delta=+0.00947084；POS recall gain=0，prevalence error 反而增大。

- BLOCK_PIT_3: pooled Brier improvement=-0.01478682（为负）；January Brier delta=+0.02622782；February delta=+0.00212000；POS recall gain=0，prevalence error 反而增大。

两候选均不 admissible；无需执行 min-Brier tie rule。selection 已 SHA 封存，selected_family=null；没有将 RAW 当成成功 winner。

### 有限 ladder + empirical PIT 的约束

原 CDF 外侧平坦，training PIT 最大值为 0.9。按预先冻结的右连续 ECDF，`G(0.9)=1`；当 +delta>q90 时，calibrated POS=1−G(0.9)=0。本轮必须如实保留这种端点退化，不能见结果后改 ties、smooth rank、尾部拟合或新增候选。这是对本次具体映射与 censored PIT 组合的否定，不是证明所有概率校准无效。

## 6. Hard stop 与未执行项

```text
D1-C calibration transfer: NOT_RUN
D1-D persistence grid: NOT_RUN
OOF State regeneration: NOT_RUN
High UNCAL_RAW / CAL_RAW / CAL_SELECTED: NOT_RUN
Reconstruction: NOT_RUN
May/June/July State/High/crossing/MAE/RMSE/Direction/Balanced/recalls/AUC/Brier: NOT_RUN
CDF calibration Spread transfer / persistence transfer: NOT_EVALUATED
Fresh-confirm predictions, SHA seal, metric opening: NOT_RUN
```
没有 d1c_transfer/d1d_persistence/d1e_oof_state/d1f_high/d1g_reconstruction/retrospective/fresh_confirm 目录。下游 mandatory artifacts 的 NOT_RUN 原因显式列入 DOWNSTREAM_NOT_RUN.json，不伪造空候选或 fake state selection seal。

## 7. 保护、运行证据与复现

| Protected group | Checked file hashes | Result |
|---|---:|---|
| src/TafM_改进源码 | 9266 | PASS |
| src/lightgbm_源码 | 15 | PASS |
| outputs/experiments/msr_spread_v1/state_only_experiment_20261009 | 95 | PASS |
| historical | 800 | PASS |
| closed_docs | 3 | PASS |

Production 2.5：只读 os.stat(size,mtime_ns) baseline 对比，94,423 文件，changed/added/missing 全 0。仅证明本轮 metadata 无漂移，不伪装为 production 内容 SHA 审计；未导入/写入 production。
早期 PRODUCTION_READONLY_AUDIT.json 因 baseline dict 与 after list 比较产生全量假差异，标为 INVALID_SCHEMA_COMPARISON；同 schema 重算的 **PRODUCTION_READONLY_AUDIT_V2.json** 为唯一有效结论，两份都保留以便追溯。该修正只针对只读 audit 格式，不涉及科学代码、指标或 Gate。
逐文件 SOURCE_CHANGE_AUDIT.json：既有 *.py changed=[]；新增 cdf_unary.py、low_cdf_state_recovery.py、test_cdf_unary78.py。保留执行源码 source_snapshot/，不覆盖 docs/73/75/77 或原 C0/C1。
候选科学 stage manifests、selection seal、测试 stdout seal 与保存预测重算独立验证均 PASS。执行无 Git 写操作；无 production 写入；Late September 与 October 没有目标指标读取。

运行命令（均在本 repo 根，以 epf-2 Python -B 执行，out 为本次新 run）：

```powershell
python -B -m pytest src/research96/msr_spread_v1/tests -q -p no:cacheprovider --basetemp=.cdf78_full_tmp_0544
python -B -m src.research96.msr_spread_v1.low_cdf_state_recovery --stage D1A --out <run_path>
python -B -m src.research96.msr_spread_v1.low_cdf_state_recovery --stage D1B --out <run_path>
```

原始命令 absolute Python/run paths 与 exit code 见根 CLI_D1A_RECEIPT.json、CLI_D1B_RECEIPT.json；FULL_TEST_GATE.json 绑定完整测试 code fingerprint。README/catalog 导航更改不修改科学源代码。

Code fingerprint: `1d5453de0b0ea53d42a600fbd7a1d2207e06b47f848d7ad213fb882660a4e445`。Config fingerprint: `e58bafa1dc790482c8b5fe92b1fb374d51fcd7b83fddf7a3a500d55d0a62b9f2`。

## 8. NEXT_PATCH_RECOMMENDATION

下一轮先预注册并独立审核唯一上游修复轴：优先审计有限 quantile ladder 的 censored PIT/端点解释，同时区分 training OOF 与 Jan–Feb 的 Low 分布漂移。右连续经验 G(0.9)=1 会把 +delta 超过 q90 的 POS 概率压成 0；不能把本轮否定扩大为所有 CDF 校准均无效。先由团队决定有限 ladder/端点表示修复还是 Low 信号重设计，再授权新协议。本轮不改映射约定、不增加候选、不重选 lambda、不校准 Low 数值、不训练 High/Z。

**科学结论**：在固定 T24/Low/kappa=0.5 下，预注册 training-only GLOBAL/BLOCK PIT 校准未在 Jan–Feb 产生受支持的 State recovery；当前证据不能回答 persistence 或 High 的下游增量，因为这些实验没有执行。Fresh confirmation 继续保留。

## 9. 完整原始报告

最初 stop report（包含所有 JSON 明细）原位保留：`../report/FINAL_REPORT.md`。本最终报告增加了月度/pooled人类可读表、逐 Gate 原因、独立重算和保护核查。完整 artifact index 与 final run manifest 在 run 根。

RUN_ROOT=D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\outputs\experiments\msr_spread_v1\low_cdf_state_recovery_20261009\20261009T054445579001Z

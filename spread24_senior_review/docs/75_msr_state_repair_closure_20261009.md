# 75 — docs/74 State Repair 工程闭环

STATUS=CODE_REPAIR_COMPLETE_PENDING_INDEPENDENT_REVIEW
EXECUTION_AUTHORITY=docs/74
SCIENTIFIC_EXECUTION=NOT_AUTHORIZED_BY_THIS_CLOSURE

## 验收

```ini
STATE_REPAIR_IMPLEMENTATION=PASS
STATE_SEMANTICS=TEMPERED_CHAIN_UNARY
LAMBDA0_IDENTITY=PASS
LAMBDA1_LEGACY_REPRO=PASS
KAPPA_FROZEN=0.5
KAPPA_TRAIN_INFERENCE_CONSISTENCY=PASS
OOF_CHRONOLOGY=PASS
LOW_CALIBRATION_DIAGNOSTIC=PASS
LOW_CALIBRATION_STATUS=WARNING
T24_OPERATOR_HASH=7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7
FULL_TESTS=253/253
CLI_SMOKE=PASS
SCIENTIFIC_STATE_EXPERIMENT_RUN=NO
MAY_JULY_NEW_METRICS_OPENED=NO
LATE_SEPTEMBER_METRICS_OPENED=NO
OCTOBER_LOCKBOX_ACCESSED=NO
PRODUCTION_WRITE=NO
READY_FOR_STATE_ONLY_EXPERIMENT=YES
```

YES 仅证明修复 correctness/isolation Gate；不代表科学增益，仍须独立审核批准后才能执行 state-only 实验。本轮没有选择 lambda、没有训练新科学模型。

## Impact map 与实现

编码前 grep 所有指定 token，完整路径/行号：STATE_USAGE_GREP.json；职责影响图：IMPACT_MAP.md。

- config.py：active OOF/inference kappa=0.5，STATE_KAPPA_OOF 是 deprecated 0.5 alias，不是独立状态任务。固定 grid 0/.25/.5/.75/1，active strength=None；必须显式声明。
- markov_state.py：新增 log-space TEMPERED_CHAIN_UNARY，raw CDF 是近似 discriminative unary，不是 HMM emission likelihood。保留 legacy_forward_backward 原算术。lambda0 normalized RAW 精确；lambda1 uniform-prior legacy 复现容差 2e-12。
- 数值兼容：lambda1 保留 legacy 1e-6 evidence floor（含 raw=0），中间强度 1e-12 log safety；lambda0 不 clip。此兼容政策与 optional prior tempering 明确纳入 semantic fingerprint，不隐藏为科学变化。
- pipeline.py：统一 frozen .5，不调用 active kappa search。各 fold 自己的合法 D-2 标签拟合 delta/IQR/transition，验证唯一完整 96 slots，每日独立链。未来标签扰动不能改变早期 OOF state features。
- 每行 state、STATE_SEMANTICS.json、config/run manifest 绑定 kappa/strength/version/数值/先验/day-boundary 语义与 SHA。High 拟合前验证 row metadata 和 manifest，缺失/legacy/混合语义 fail fast；所有新增 audit metadata 从模型 feature list 排除。
- run_experiment.py / 旧 t24_controlled_rerun.py / pipeline stage decorator 启用 CODE_REPAIR_ONLY execution guard，独立评审前不能科学重跑。
- low_calibration.py 只诊断；state_repair.py 仅非科学 smoke。Low/High/Z 参数、feature families、quantiles、T24、reconstruction 不变；Z 无科学改动。

## Low 校准

只读冻结 Low run 20261009T025641665850Z，预测 SHA f7b43c1b428d9488888560e7e4f8e08f9adfe92daf84361a2c2705e4b3259050。
reader 在读取列值前投影 KEY/L_true/Low quantiles，并过滤 fitted 2026-01-01..04-30；没有从该文件取 May–July truth。

Coverage：`{"q10": 0.14618055555555556, "q25": 0.20503472222222222, "q50": 0.3551215277777778, "q75": 0.64140625, "q90": 0.8038194444444444}`；Q50=0.355122，偏离 .5 超过 .05，真实标记 WARNING。这是已有 Low 校准问题，不是本轮 correctness FAIL。

输出每月 Q50 coverage、mean/median residual 和每 slot residual。现有 enforced quantile crossing rate 如实输出；上游没保存 pre-enforcement ladder，因此 before rate 必须 UNKNOWN_NOT_SAVED_UPSTREAM，绝不伪造 0；helper 接受显式 before ladder 时可计算真实 before/after。
无 fitted recalibration transform，无 Low prediction bytes/values 改动。

## 测试与隔离

T01–T12 全覆盖：identity、legacy、finite/normalization/input rejection、日边界、synthetic chatter、1/2-slot shock、kappa/fingerprint/High mixing rejection、逐 fold future-label perturbation/非法 cutoff、Low read-only、sealed-date guard/reader projection、T24 hash、protected isolation。

完整 suite **253 passed，无 skipped**。前两次失败日志保留：历史 fingerprint/startup 数量/字面源码断言及限缩 truth 读取后的 fixture 预期需要更新；没有降低数学或隔离 Gate。fixture 的真实数据只有已消费 dev slice，train 是 synthetic，包括 89 个注入缺失-Z 日以保留原回归覆盖；没有额外取真实 train/final/holdout。

真实 CLI：`python -B -m src.research96.msr_spread_v1.state_repair --smoke --dev-calibration`，PASS。lambda 0/.5/1 只测试 synthetic 接口，不做科学比较。canonical parquet 仅字节 hash 校验，不额外加载源 truth。

保护基线 **800 项、漂移 0**，包括旧 C0/C1/D0、docs/73 成功及中断 T24 链和冻结源文件。既有 TafM dirty 修改保留，不修改/还原/清理 production 和冻结目录。不是声称整个生产树零漂移。

独立分解/Low/High/Z/机制/重构/dataset 源文件 SHA 与编码前相同；LightGBM 和 quantile 与冻结 FROZEN_RUN_CHOICES 逐项一致。

## 证据与下一步

- 修复证据：`outputs/experiments/msr_spread_v1/state_repair_20261009/20261009T041637857647Z/`。
- CLI evidence：`outputs/experiments/msr_spread_v1/state_repair_20261009/smoke/20261009T042920880656Z/`。
- code fingerprint：0d6534debc16a76e69ca8b4ce53776b87bdab325f49a457513c6ab4dfd34ee0e。
- 原始/失败/成功日志、SOURCE_BEFORE/AFTER、保护哈希、TEST_COVERAGE_MAP、STATE_REPAIR_CLOSURE 和 manifest 均保留。

旧 artifact 可只读；不能绕过 code/config/state semantic hash 混入新 High/state 链。跨版本复用旧 Low 须独立审核迁移兼容性。本轮不执行该科学迁移。

下一步只允许独立审核；批准后才执行 docs/74 预注册 V1a/no-Z、kappa=.5 的 state-only 实验。不得在本轮自动重跑、改 loss/阈值/kappa/T24、增加结构或推广 Z。Late September 与 October 均保持未打开。

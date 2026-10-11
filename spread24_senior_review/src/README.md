# 源码总目录 — 唯一当前导航

模型最高实现合同：[docs/63](../docs/63_msr_spread_v1_final_code_handoff_20261007.md)。
工程证据：[docs/66](../docs/66_msr_spread_v1_remaining_closure_20261008.md)；
本轮任务：[docs/67](../docs/67_msr_v1_pre_experiment_audit_and_repo_cleanup_plan_20261008.md)；
本轮闭环：[docs/68](../docs/68_msr_v1_pre_experiment_closure_20261008.md)。
编号更新本身不会自动替代模型合同。文档总导航为 [docs/README](../docs/README.md)。

| 目录 | 状态 | 责任/资产 | 修改边界 | 入口 |
|---|---|---|---|---|
| lightgbm_源码/ | HISTORICAL_BASELINE / READ_ONLY_FOR_MSR | 历史 24 点基线；data/features/model/evaluate/leakage/pipeline/feature_importance | 本轮只读 | 仅参考 |
| TafM_改进源码/ | FROZEN_HISTORICAL_ENGINEERING | TafM/TabM V2/V2.1，Direction/Magnitude；outputs/round3_outputs 为历史证据 | 禁止修改、还原或清缓存；相对 HEAD 的已有 dirty 不等于本轮漂移 | 仅参考 |
| research96/ | ACTIVE_SHARED_INFRA | Native-96 source、vintage、canonical/materialization、审计；历史 runner 单独标识 | 仅必要的最小共用改动，本轮未改共用 Python | [模块索引](research96/README.md) |
| research96/msr_spread_v1/ | ACTIVE_MODEL | Low/High、Markov、Z、重构、解释及测试 | 本轮唯一模型修改区 | [run_experiment.py](research96/msr_spread_v1/run_experiment.py)、[模块索引](research96/msr_spread_v1/README.md) |

## 顶层模块（不得误认为 MSR 主线）

| 文件 | 状态 | 职责 |
|---|---|---|
| __init__.py | SHARED_NAMESPACE | src 包标记 |
| data.py / features.py / model.py / evaluate.py / leakage.py / pipeline.py | HISTORICAL_COMPATIBILITY_SHIM | 向 unchanged lightgbm_源码 的 backward-compatible 导入桥；不是独立新实现 |
| run_tabm_v2.py | LEGACY_DO_NOT_EXECUTE | 已退役、fail-closed 的 V2 CLI |
| run_tabm_v21.py | HISTORICAL_RUNNER | V2.1 研究 CLI；输出 outputs/tabm_v21；不是 MSR 当前入口 |
| config_tabm_v2.yaml / config_tabm_v21.yaml | HISTORICAL_CONFIG | 上述历史体系配置；本轮不改 |

src/src/ 当前不存在，不是保留入口。__pycache__/.pytest-* 是生成目录；混合内容或
归属不明的测试目录不直接删除，清理决定见 docs/68。冻结目录历史输出不可清理。
生产 electricity_forecast_model2.5 严格只读；不访问 October lockbox；所有研究产物
只在本包功能目录。真实 CLI 使用既定配置，不把已经打开的 final 当新确认。

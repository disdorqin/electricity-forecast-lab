# 文档总索引 — 当前唯一导航

模型实现以 **docs/63** 为最高合同；docs/67 是修复/整理任务书，docs/68 是本轮验收；docs/66 是上一轮工程证据。
历史更高/更低编号、旧统计或旧实验指令都不会自动覆盖当前合同。CURRENT_AUTHORITY 也不表示各文件都是模型设计合同。
Native-96 当前可见性合同：[NATIVE96_FIELD_POLICY](../experiments/first_test/96_information_challenge/NATIVE96_FIELD_POLICY_20261005.json)，配合 AGENTS.md D-1 14:00 / D-2。
文献 current authority 为 57/58；60/61/56 仅是设计来源。旧 final（2026-05-01..08-14）已打开，任何重跑仅 retrospective。

## 日常只看这些

- **当前模型与工程闭环**：63（实现合同）、66（上一轮）、67（修复任务）、68（验收）；01（数据/防泄漏）。
- **文献与设计来源**：57/58（当前文献 authority）；56/60/61（设计参考，不覆盖 63）。
- **仍有用的数据工具协议**：38/40/41/42/43/45/49/51/53/55；模块入口见 [research96 索引](../src/research96/README.md)。
- **本次整理结果**：[70_repository_cleanup_20261008.md](70_repository_cleanup_20261008.md)。历史表格不代表当前执行队列。

12 篇过时设计/交接/开指标任务书的原始正文已移至 `archive/`；原路径仅保留短跳转页，避免破坏历史快照引用。不是保留两套正文。

## 全部文档分类

| filename | status | superseded_by | purpose |
|---|---|---|---|
| [00_项目总览与文档索引.md](00_项目总览与文档索引.md) | HISTORICAL_PROVENANCE | README.md | 历史 V2.1 项目索引，正文保留但不再 canonical |
| [01_业务数据与防泄漏合同.md](01_业务数据与防泄漏合同.md) | CURRENT_AUTHORITY | — | 01 业务数据与防泄漏合同 |
| [02_历史基线与实验结论.md](02_历史基线与实验结论.md) | ACTIVE_REFERENCE | — | 02 历史基线与实验结论 |
| [17_最终模型设计与编码规范.md](17_最终模型设计与编码规范.md) | LEGACY_DO_NOT_EXECUTE | 63 / AGENTS 当前隔离协议；原历史约束仍供审计 | 17 最终模型设计与编码规范 |
| [21_正式运行、验收与实验入口规范.md](21_正式运行、验收与实验入口规范.md) | LEGACY_DO_NOT_EXECUTE | 63 / AGENTS 当前隔离协议；原历史约束仍供审计 | 21 正式运行、验收与实验入口规范 |
| [22_V2.1方向优先实验路线与V2.2候选设计.md](22_V2.1方向优先实验路线与V2.2候选设计.md) | LEGACY_DO_NOT_EXECUTE | archive/pre_r96_design_history/22_V2.1方向优先实验路线与V2.2候选设计.md | 兼容旧引用的短跳转页；正文已归档 |
| [23_E5_regime_aware_framework_reconstruction_plan.md](23_E5_regime_aware_framework_reconstruction_plan.md) | LEGACY_DO_NOT_EXECUTE | archive/pre_r96_design_history/23_E5_regime_aware_framework_reconstruction_plan.md | 兼容旧引用的短跳转页；正文已归档 |
| [24_E5_A_regime_diagnosis_experiment_spec.md](24_E5_A_regime_diagnosis_experiment_spec.md) | LEGACY_DO_NOT_EXECUTE | archive/pre_r96_design_history/24_E5_A_regime_diagnosis_experiment_spec.md | 兼容旧引用的短跳转页；正文已归档 |
| [25_E5_regime_aware_architecture_design_v2.md](25_E5_regime_aware_architecture_design_v2.md) | LEGACY_DO_NOT_EXECUTE | archive/pre_r96_design_history/25_E5_regime_aware_architecture_design_v2.md | 兼容旧引用的短跳转页；正文已归档 |
| [26_E6_temporal_representation_reconstruction_plan.md](26_E6_temporal_representation_reconstruction_plan.md) | LEGACY_DO_NOT_EXECUTE | archive/pre_r96_design_history/26_E6_temporal_representation_reconstruction_plan.md | 兼容旧引用的短跳转页；正文已归档 |
| [27_research_reset_v3_cross_horizon_profile_distributional_plan.md](27_research_reset_v3_cross_horizon_profile_distributional_plan.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 27 — Research Reset：V3 Cross-Horizon Profile Distributional Model |
| [28_prospective_confirm_source_extension_plan.md](28_prospective_confirm_source_extension_plan.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 28 — Prospective CONFIRM Source Extension Plan |
| [29_s2_numerical_equivalence_amendment.md](29_s2_numerical_equivalence_amendment.md) | ACTIVE_REFERENCE | — | 29 — S2 Numerical Equivalence Amendment |
| [30_history_gap_recovery_and_research_synthesis.md](30_history_gap_recovery_and_research_synthesis.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 30 — History-Gap Recovery, Eligibility Separation, and V3 Research Synthesis |
| [31_r6_1_correction_materialization_and_quarantine_closure.md](31_r6_1_correction_materialization_and_quarantine_closure.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 31 — R6.1 Correction Materialization and Quarantine-Closure Amendment |
| [32_one_shot_prospective_confirm_execution_protocol.md](32_one_shot_prospective_confirm_execution_protocol.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 32 — One-Shot Prospective CONFIRM Execution Protocol |
| [33_96_point_information_challenge_design.md](33_96_point_information_challenge_design.md) | ACTIVE_REFERENCE | — | 33 — 96-Point Information Challenge Design |
| [34_r96_abc_engineering_implementation_plan.md](34_r96_abc_engineering_implementation_plan.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 34 — R96-A/B/C Engineering Implementation Plan |
| [35_r96_d_information_challenge_execution_protocol.md](35_r96_d_information_challenge_execution_protocol.md) | LEGACY_DO_NOT_EXECUTE | 63 / AGENTS 当前隔离协议；原历史约束仍供审计 | 35 — R96-D Information Challenge Execution Protocol |
| [35_r96_d_one_shot_information_matrix_protocol.md](35_r96_d_one_shot_information_matrix_protocol.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 35 — R96-D One-Shot Information Matrix Protocol |
| [36_r96_d_execution_canonicalization.md](36_r96_d_execution_canonicalization.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 36 — R96-D Execution Canonicalization and Legacy Freeze |
| [37_r96_ic5_plus_strict_information_expansion_plan.md](37_r96_ic5_plus_strict_information_expansion_plan.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 37 — IC5+ Strict Information Expansion Plan |
| [38_r96_f0_forecast_vintage_recovery_plan.md](38_r96_f0_forecast_vintage_recovery_plan.md) | ACTIVE_REFERENCE | — | 38 — R96-F0 Strict Forecast Vintage Recovery Plan |
| [38_strict_forecast_vintage_recovery_protocol.md](38_strict_forecast_vintage_recovery_protocol.md) | ACTIVE_REFERENCE | — | 38 — Strict Forecast Vintage Recovery Protocol |
| [39_r96_post_information_three_state_distribution_architecture.md](39_r96_post_information_three_state_distribution_architecture.md) | LEGACY_DO_NOT_EXECUTE | archive/r96_superseded_handoffs/39_r96_post_information_three_state_distribution_architecture.md | 兼容旧引用的短跳转页；正文已归档 |
| [40_r96_f0_versioned_optional_endpoint_recovery.md](40_r96_f0_versioned_optional_endpoint_recovery.md) | ACTIVE_REFERENCE | — | 40 — R96-F0 Versioned Optional Endpoint Recovery Plan |
| [41_r96_ic5c_forecast_revision_path.md](41_r96_ic5c_forecast_revision_path.md) | ACTIVE_REFERENCE | — | 41 — IC5C Strict Forecast Revision Path |
| [42_r96_short_history_new_information_adaptation_protocol.md](42_r96_short_history_new_information_adaptation_protocol.md) | ACTIVE_REFERENCE | — | 42 — Short-History New-Information Adaptation Protocol |
| [43_r96_ic6b_ic7_strict_market_state_contract.md](43_r96_ic6b_ic7_strict_market_state_contract.md) | ACTIVE_REFERENCE | — | 43 — IC6B / IC7 Strict Market-State Experiment Contract |
| [44_r96_current_state_handoff_20261001.md](44_r96_current_state_handoff_20261001.md) | LEGACY_DO_NOT_EXECUTE | archive/r96_superseded_handoffs/44_r96_current_state_handoff_20261001.md | 兼容旧引用的短跳转页；正文已归档 |
| [45_r96_data_arrival_to_training_readiness_plan.md](45_r96_data_arrival_to_training_readiness_plan.md) | ACTIVE_REFERENCE | — | 45 — R96 Data-Arrival-to-Training Readiness Plan |
| [46_r96_literature_reframe_information_ceiling_and_research_agenda.md](46_r96_literature_reframe_information_ceiling_and_research_agenda.md) | LEGACY_DO_NOT_EXECUTE | archive/r96_superseded_handoffs/46_r96_literature_reframe_information_ceiling_and_research_agenda.md | 兼容旧引用的短跳转页；正文已归档 |
| [47_r96_focused_literature_to_code_rounds_20261002.md](47_r96_focused_literature_to_code_rounds_20261002.md) | LEGACY_DO_NOT_EXECUTE | archive/r96_superseded_handoffs/47_r96_focused_literature_to_code_rounds_20261002.md | 兼容旧引用的短跳转页；正文已归档 |
| [48_r96_privileged_future_information_research_recovery.md](48_r96_privileged_future_information_research_recovery.md) | HISTORICAL_PROVENANCE | 对应新协议见本索引；不是新实验授权 | 48 — R96 Privileged Future Information Research Recovery |
| [49_r96_f0_collector_code_handoff_20261002.md](49_r96_f0_collector_code_handoff_20261002.md) | ACTIVE_REFERENCE | — | 49 — R96 F0 Collector Code Handoff |
| [50_r96_real_data_execution_design_v2_20261002.md](50_r96_real_data_execution_design_v2_20261002.md) | LEGACY_DO_NOT_EXECUTE | archive/r96_superseded_handoffs/50_r96_real_data_execution_design_v2_20261002.md | 兼容旧引用的短跳转页；正文已归档 |
| [51_r96_f0b_self_dated_pre_p56_protocol_20261003.md](51_r96_f0b_self_dated_pre_p56_protocol_20261003.md) | ACTIVE_REFERENCE | — | 51 — R96 F0-B Self-Dated Pre-p56 ForecastData Protocol |
| [52_r96_f0b_pre_metric_adjudication_plan_20261003.md](52_r96_f0b_pre_metric_adjudication_plan_20261003.md) | LEGACY_DO_NOT_EXECUTE | archive/r96_superseded_handoffs/52_r96_f0b_pre_metric_adjudication_plan_20261003.md | 兼容旧引用的短跳转页；正文已归档 |
| [53_r96_canonical_store_and_ic7a_execution_plan_20261003.md](53_r96_canonical_store_and_ic7a_execution_plan_20261003.md) | ACTIVE_REFERENCE | — | 53 — R96 Canonical Local Research Store + IC7A Execution Plan |
| [54_r96_ic7a_literature_and_pre_metric_adjudication_20261004.md](54_r96_ic7a_literature_and_pre_metric_adjudication_20261004.md) | LEGACY_DO_NOT_EXECUTE | archive/r96_superseded_handoffs/54_r96_ic7a_literature_and_pre_metric_adjudication_20261004.md | 兼容旧引用的短跳转页；正文已归档 |
| [55_r96_wave2_driver_state_prior_fixed_cutoff_design_20261004.md](55_r96_wave2_driver_state_prior_fixed_cutoff_design_20261004.md) | ACTIVE_REFERENCE | — | 55 — R96 Wave-2 Driver–State–Prior with Fixed Research Cutoff |
| [56_native96_tafm_engineering_design_20261005.md](56_native96_tafm_engineering_design_20261005.md) | ACTIVE_REFERENCE | 63 (MSR implementation only) | 56 — Native-96 TafM Engineering Design V1 |
| [57_spread_literature_research_protocol_20261006.md](57_spread_literature_research_protocol_20261006.md) | CURRENT_AUTHORITY | — | 57 — DA / RT / Spread 文献研究协议与资料收口 |
| [58_spread_literature_review_ledger.md](58_spread_literature_review_ledger.md) | CURRENT_AUTHORITY | — | 58 — Spread Literature Review Ledger |
| [59_spread_literature_cleanup_map_20261006.md](59_spread_literature_cleanup_map_20261006.md) | ACTIVE_REFERENCE | — | 59 — Spread Literature Cleanup Map |
| [60_multiscale_regime_spread_model_design_20261007.md](60_multiscale_regime_spread_model_design_20261007.md) | ACTIVE_REFERENCE | 63 (MSR implementation only) | 60 — Multiscale Regime Spread Model Design (2026-10-07) |
| [61_msr_tafm_implementation_spec_20261007.md](61_msr_tafm_implementation_spec_20261007.md) | ACTIVE_REFERENCE | 63 (MSR implementation only) | 61 — MSR-TafM Unified Probabilistic Implementation Specification (2026-10-07) |
| [63_msr_spread_v1_final_code_handoff_20261007.md](63_msr_spread_v1_final_code_handoff_20261007.md) | CURRENT_AUTHORITY | — | 63 — MSR-Spread V1 Final Code Handoff (2026-10-07) |
| [64_msr_spread_v1_audit_remediation_plan_20261008.md](64_msr_spread_v1_audit_remediation_plan_20261008.md) | HISTORICAL_PROVENANCE | 66 / 68 (闭环证据；缺陷记录仍保留) | 64 — MSR-Spread V1 问题审计与改进计划（2026-10-08） |
| [66_msr_spread_v1_remaining_closure_20261008.md](66_msr_spread_v1_remaining_closure_20261008.md) | CURRENT_AUTHORITY | — | 66 — MSR-Spread V1 复审剩余项闭环 |
| [67_msr_v1_pre_experiment_audit_and_repo_cleanup_plan_20261008.md](67_msr_v1_pre_experiment_audit_and_repo_cleanup_plan_20261008.md) | CURRENT_AUTHORITY | — | 67 — MSR-Spread V1 预实验代码复审与仓库整理计划（2026-10-08） |
| [68_msr_v1_pre_experiment_closure_20261008.md](68_msr_v1_pre_experiment_closure_20261008.md) | CURRENT_AUTHORITY | — | 68 — MSR V1 预实验修复与整理闭环 |
| [archive/msr_v1_remediation_history/65_msr_spread_v1_remediation_evidence_20261008.md](archive/msr_v1_remediation_history/65_msr_spread_v1_remediation_evidence_20261008.md) | LEGACY_DO_NOT_EXECUTE | 66 (上轮)，68 (本轮) | 旧 137 项修复链证据，后续剩余缺陷由 66 闭环；归档而非删除 |
| README.md | CURRENT_AUTHORITY (navigation only) | — | 当前唯一文档导航；不定义模型 |

## 同期新增任务（保留工作区修改）

| filename | status | superseded_by | purpose |
|---|---|---|---|
| [69_boundary_safe_low_high_decomposition_adjudication_20261008.md](69_boundary_safe_low_high_decomposition_adjudication_20261008.md) | CURRENT_AUTHORITY | — | 边界分解预实验任务书；仅其授权范围有效，不在本清理中执行 |

## 本次新增归档正文

| filename | status | superseded_by | purpose |
|---|---|---|---|
| [archive/pre_r96_design_history/22_V2.1方向优先实验路线与V2.2候选设计.md](archive/pre_r96_design_history/22_V2.1方向优先实验路线与V2.2候选设计.md) | LEGACY_DO_NOT_EXECUTE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/pre_r96_design_history/23_E5_regime_aware_framework_reconstruction_plan.md](archive/pre_r96_design_history/23_E5_regime_aware_framework_reconstruction_plan.md) | LEGACY_DO_NOT_EXECUTE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/pre_r96_design_history/24_E5_A_regime_diagnosis_experiment_spec.md](archive/pre_r96_design_history/24_E5_A_regime_diagnosis_experiment_spec.md) | LEGACY_DO_NOT_EXECUTE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/pre_r96_design_history/25_E5_regime_aware_architecture_design_v2.md](archive/pre_r96_design_history/25_E5_regime_aware_architecture_design_v2.md) | LEGACY_DO_NOT_EXECUTE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/pre_r96_design_history/26_E6_temporal_representation_reconstruction_plan.md](archive/pre_r96_design_history/26_E6_temporal_representation_reconstruction_plan.md) | LEGACY_DO_NOT_EXECUTE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/r96_superseded_handoffs/39_r96_post_information_three_state_distribution_architecture.md](archive/r96_superseded_handoffs/39_r96_post_information_three_state_distribution_architecture.md) | HISTORICAL_PROVENANCE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/r96_superseded_handoffs/44_r96_current_state_handoff_20261001.md](archive/r96_superseded_handoffs/44_r96_current_state_handoff_20261001.md) | HISTORICAL_PROVENANCE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/r96_superseded_handoffs/46_r96_literature_reframe_information_ceiling_and_research_agenda.md](archive/r96_superseded_handoffs/46_r96_literature_reframe_information_ceiling_and_research_agenda.md) | HISTORICAL_PROVENANCE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/r96_superseded_handoffs/47_r96_focused_literature_to_code_rounds_20261002.md](archive/r96_superseded_handoffs/47_r96_focused_literature_to_code_rounds_20261002.md) | HISTORICAL_PROVENANCE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/r96_superseded_handoffs/50_r96_real_data_execution_design_v2_20261002.md](archive/r96_superseded_handoffs/50_r96_real_data_execution_design_v2_20261002.md) | HISTORICAL_PROVENANCE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/r96_superseded_handoffs/52_r96_f0b_pre_metric_adjudication_plan_20261003.md](archive/r96_superseded_handoffs/52_r96_f0b_pre_metric_adjudication_plan_20261003.md) | HISTORICAL_PROVENANCE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [archive/r96_superseded_handoffs/54_r96_ic7a_literature_and_pre_metric_adjudication_20261004.md](archive/r96_superseded_handoffs/54_r96_ic7a_literature_and_pre_metric_adjudication_20261004.md) | HISTORICAL_PROVENANCE | 63 / 57–58 / 68（各自职责） | 原始历史正文，保留溯源，不直接执行 |
| [70_repository_cleanup_20261008.md](70_repository_cleanup_20261008.md) | ACTIVE_REFERENCE | — | 本次源码/文档整理审计；不修改模型合同 |

## 引用完整性与归档策略

全仓引用扫描（含代码、测试、manifest 和 Markdown）先于移动；完整文件名与 docs/64、docs/65 短引用分别扫描。
证据：outputs/experiments/msr_spread_v1/pre_experiment_20261008/DOC_REFERENCE_AUDIT.json/md、DOC_REFERENCE_SHORT_ALIASES.json、ARCHIVE_DECISIONS.json。
预实验闭环一轮仅物理归档明确 superseded 且无硬依赖的 docs/65；docs/64 仍被测试/源码注释引用，旧 docs/35 仍被冻结 manifest 引用，原位逻辑降级，禁止删改原证据。
原 docs/62 从未存在，不是任何 active authority。历史 provenance 中 production docs/DATA_CONTRACT_96.md、LEAKAGE_AUDIT_96.md 指向生产项目，不是本包缺失文件；不改旧 manifest。
若旧文档正文的 historical 指令与 current 冲突，只用于理解当时实验，禁止直接执行。分类不授权访问已消费确认窗口或 October lockbox。

2026-10-08 后续整理：先保存全仓 `PRE_MOVE_AUDIT.json` 与短编号扫描，再原样归档 22–26、39、44、46、47、50、52、54；原路径保留跳转。32 因历史执行代码注释引用而不移动。审计及决策位于 `outputs/repository_cleanup/20261008T195658/`。

| [71_d0_decomposition_audit_20261009.md](71_d0_decomposition_audit_20261009.md) | CURRENT_AUTHORITY | — | 当前 D0 审计 / T24 修复与阶段验证协议（各自职责） |

| [72_msr_t24_repair_staged_rerun_and_multimonth_validation_20261009.md](72_msr_t24_repair_staged_rerun_and_multimonth_validation_20261009.md) | CURRENT_AUTHORITY | — | 当前 D0 审计 / T24 修复与阶段验证协议（各自职责） |

| [73_msr_t24_retrospective_closure_20261009.md](73_msr_t24_retrospective_closure_20261009.md) | ACTIVE_REFERENCE | — | docs/72 修复与三个月五组验证闭环；readiness NO，未打开 late September |

| [74_msr_state_consistency_and_tempered_persistence_repair_20261009.md](74_msr_state_consistency_and_tempered_persistence_repair_20261009.md) | ACTIVE_REFERENCE | 76（科学执行） | state 语义修复、tempered persistence 定义与预注册 gate |

| [75_msr_state_repair_closure_20261009.md](75_msr_state_repair_closure_20261009.md) | ACTIVE_REFERENCE | — | docs/74 state 工程闭环；253/253，通过独立审核后进入 76 |

| [76_msr_state_only_experiment_protocol_20261009.md](76_msr_state_only_experiment_protocol_20261009.md) | ACTIVE_REFERENCE | — | State-only persistence 科学实验协议；docs/77 已证明无 admissible persistence |
| [77_msr_state_only_selection_closure_20261009.md](77_msr_state_only_selection_closure_20261009.md) | ACTIVE_REFERENCE | — | docs/76 五个 State 候选均不 admissible，Stage B 强制停止；不训练 High、不打开 late September |

| [78_msr_low_cdf_unary_calibration_and_state_recovery_protocol_20261009.md](78_msr_low_cdf_unary_calibration_and_state_recovery_protocol_20261009.md) | ACTIVE_REFERENCE | 76（历史范围） | 静态 Low-CDF/PIT calibration 协议；docs/79 已于 D1-B 强制停止 |

| [79_msr_low_cdf_unary_calibration_closure_20261009.md](79_msr_low_cdf_unary_calibration_closure_20261009.md) | ACTIVE_REFERENCE | — | docs/78 D1-A/B 闭环；静态 GLOBAL/BLOCK PIT 未恢复 State，暴露 2026 Low location drift 与有限 ladder endpoint 限制；303/303 |
| [80_msr_low_drift_adaptive_location_calibration_protocol_20261009.md](80_msr_low_drift_adaptive_location_calibration_protocol_20261009.md) | ACTIVE_REFERENCE | — | 历史 Low drift-adaptive location correction 协议；docs/81 已闭环，ROLL28_GLOBAL 未通过 April transfer |

| [81_msr_low_location_recovery_closure_20261009.md](81_msr_low_location_recovery_closure_20261009.md) | ACTIVE_REFERENCE | — | docs/80 March选择ROLL28_GLOBAL，April transfer四Gate失败即停；341/341；不训练High、不打开May–July新指标或late-Sep |

| [82_msr_daily_asof_training_and_fast24_protocol_20261009.md](82_msr_daily_asof_training_and_fast24_protocol_20261009.md) | CURRENT_AUTHORITY | — | 当前训练范式权威：DAILY_ASOF_V1，D-2-safe daily expanding fresh fit + 固定 FAST-24；被 docs/84 作为 V2 训练底座继承 |
| [83_msr_core_v1_architecture_redesign_draft_20261009.md](83_msr_core_v1_architecture_redesign_draft_20261009.md) | SUPERSEDED_DRAFT | 84 | 架构讨论历史；已由 docs/84 的 Native-96 V2 统一架构取代，不得作为实现权威 |
| [84_msr_v2_native96_unified_architecture_implementation_spec_20261009.md](84_msr_v2_native96_unified_architecture_implementation_spec_20261009.md) | CURRENT_AUTHORITY | — | MSR-V2 当前最高架构/实现合同：原生96、多业务语义频率gate、Low/High可微分位函数、smooth dynamic dependence、统一最终分布/损失；State/U仅保留扩展接口 |
| [85_msr_v2_codex_one_shot_implementation_prompt_20261009.md](85_msr_v2_codex_one_shot_implementation_prompt_20261009.md) | ACTIVE_IMPLEMENTATION_PROMPT | 84 | 面向 Codex 的一次性实现导演提示词；不得覆盖 docs/84 科学设计，只负责按合同落地、测试、smoke 与零污染证明 |
| [86_msr_v2_gpu_core_and_component_adjudication_protocol_20261009.md](86_msr_v2_gpu_core_and_component_adjudication_protocol_20261009.md) | CURRENT_EXPERIMENT_AUTHORITY | — | MSR-V2 首轮 GPU 科学实验权威：先 FULL_V2 vs DIRECT96，再做依赖/gate/Low→High/Brier 部件归因；FAST-24/可选FULL-30；禁止调参式扫网格 |
| [87_msr_v2_gpu_component_adjudication_codex_prompt_20261009.md](87_msr_v2_gpu_component_adjudication_codex_prompt_20261009.md) | ACTIVE_EXPERIMENT_PROMPT | 86 | 面向 Codex 的 GPU 环境资格、正式训练、核心对照与部件消融一次性执行提示词；结果差只诊断不擅自重构 |
| [88_msr_v2_gpu_adjudication_independent_audit_20261010.md](88_msr_v2_gpu_adjudication_independent_audit_20261010.md) | ACTIVE_REFERENCE | — | 对 docs/86 完成结果的独立复核；接受 MIXED 终态，但将首要根因修正为 Low 跨月稳定性/适应性瓶颈，并记录 oracle substitution 证据 |

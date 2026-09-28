# EFM3 实验室 Agent 约束

> 本文件是 electricity_forecast_lab 的工作约束。实验室用于研究新功能，不直接改变成熟主项目。

## 1. 隔离边界

- 主项目 electricity_forecast_model2.5 是生产事实源；实验室默认只读主项目。
- 实验室每个一级子目录代表一个独立功能/实验，功能之间默认不互相导入。
- 实验室的代码、数据、模型、输出和文档不得静默写回主项目；接入主项目必须经过单独评审和用户明确批准。
- 本实验室不承载生产预测，不替代主项目的 python main.py --96 DATE 正式入口。

## 2. 设计原则

- **大道至简、最小熵增**：为满足目标只引入必要的状态、接口和文件；不为理论完整性堆叠框架、重复数据表示或并行契约。
- **最小改动、影响面优先**：先明确输入、输出、调用关系和边界，再在最窄责任层修改；不借实验机会重构主项目。
- **可恢复、可审计**：实验应保留来源、版本、配置、seed、时间边界、运行命令和结果摘要；重复运行应幂等或明确版本化。
- **先契约后实现**：开始编码前写清目标、输入/输出、信息可见性、指标、验收门和失败条件。
- **生产与研究分离**：实验结果只能说明实验本身，不得未经审计宣传为正式生产结果。

## 3. 数据与防泄漏

- 任何时间序列实验先画 forecast-origin 信息时间线，再写特征和训练切分。
- 24 点价差实验的严格协议：预测日 D 的 origin 为 D-1 14:00；完整监督标签最晚到 D-2。
- D-1 仅允许使用当时已经发生的部分信息；不得使用 D-1 14:00 之后的实际值、目标日实际值或完整 D-1 标签。
- threshold、weight、router、feature selection、similar-day、calibration 和 early stopping 同样受信息边界约束。
- 结果必须标注 STRICT/PASS、LEGACY-UNVERIFIED、ORACLE/PRIVILEGED 或 INVALID-LEAKAGE；未通过严格审计的数字不得作为当前最优。
- 报告方向指标时，至少同时看 direction、positive/negative recall、balanced accuracy、all-negative baseline，并进行跨月份检查。

## 4. 可复现与资产管理

- 使用相对路径或配置，不依赖主项目绝对路径。
- 记录 Python/依赖版本、git commit、数据快照或哈希、seed、resolution、cutoff 和运行命令。
- 输出只放在当前功能目录内；临时缓存、__pycache__、.pytest_cache 和日志不应成为交付资产。
- 不在实验室保存密码、Cookie、token 或其他密钥。
- 运行前后优先使用 smoke、契约检查和 leakage audit；轻量测试不能替代真实入口验证。

## 5. 接入主项目前检查清单

- [ ] 目标和业务价值明确，且确有必要新增功能。
- [ ] 输入、输出、forecast-origin 和信息边界已写入文档。
- [ ] 数据来源、版本、哈希和可追溯路径已记录。
- [ ] 通过最小回归、smoke 和泄漏审计。
- [ ] 评估包含跨时间窗口和基线，不只看单日/单月漂亮数字。
- [ ] 已评估对 parser、runner、scheduler、adapter、ledger、report、tests 和 active docs 的影响面。
- [ ] 用户明确批准后，才制定主项目接入补丁。

## 6. 首个功能的特别约束

spread24_senior_review 是从主项目复制的独立 24 点价差研究包。它遵循上面的 D-1 14:00 / D-2 标签边界；其结果是研究结果，不自动进入 24 点或 96 点正式生产链路。

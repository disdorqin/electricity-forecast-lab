# MSR-V2 Native-96 独立研究 Core

最高架构合同：docs/84；执行提示：docs/85；逐日训练合同：docs/82。
V1 / TafM / electricity_forecast_model2.5 只读；无 runtime V1/TafM/生产导入，无 DB 依赖。

## 输入时间线与科研边界

- origin = D-1 14:00 Asia/Shanghai；target = 96点 Spread=RT-DA；完整监督 <=D-2。
- D-8..D-2 七天真实 DA/RT/Spread/T24 Low/High，按目标 quarter 对齐到 channels。
- D-1 DA全96；RT仅p1–56；p57–96复制D-2同slot Spread，再用D-1 DA重建RT proxy，保留mask。
- docs/84特许研究假设：D全96 DA、D-1全96 grid actual、D全96 grid forecast。
- target RT/Spread/actual不进入特征；proxy不是L/H真值；未知字段fail closed。
- 历史 source 为 LEGACY-UNVERIFIED / PROXY-EXPERIMENTAL，不声称真实 origin vintage。
- Native96输入不降成24点，不创建672-token Transformer；无hard State、Markov、U predictor。
- 仅K=1 RegimeContext identity与High u_context=None扩展接口。

## 图与冻结常数

五族独立embedding/DCT gate：PRICE、DEMAND、RENEWABLE、SUPPLY_SYSTEM、MARKET_TIGHTNESS；
CALENDAR_MASK bypass。每族gate是有界单调低通、High精确互补，初始化cutoff=8（T24半增益特征尺度）。
Low：48通道TCN（1,2,4,8,16,32）+官方TabM-mini（k4/width64）；
High：64通道TCN（1,2,4,8）+官方TabM-mini；bounded convex fusion；High读取当前Low分布summary。
分位节点.01,.05,.10,.25,.50,.75,.90,.95,.99，以median±softplus累计增量保证单调，不排序独立输出。
dependence：6个固定DCT平滑基、learned daily rho/phiL/phiH、固定Sobol noise；训练32、评估256。
原单位逐scenario精确S=L+H；Spread/RT和P(S>0)由同一scenario分布派生。
训练事件阈值为零，Spread仅按训练IQR缩放，不平移零；Low/High各自训练median/IQR变换，重构前逆变换。
objective：CRPS_S+.30 Brier0+.25 SmoothL1软median+.20 Low+.20 High+.01 dependence smoothness。
训练A组件、B仅依赖网络（冻结所有分支及router）、C全部解冻；28日chronological inner validation；
scientific最后fresh exact refit，no-refit-fast-dev仅debug。

## T24字节身份与来源

t24.py逻辑来源V1/decomposition.py自然second-difference算子，lambda=215.32246474604284。
t24_operator.npy由原epf-2 / NumPy2.4.6重新计算，数组SHA=
7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7。
加载后校验SHA与96x96 float64并设只读，避免平台BLAS inversion字节漂移；不重选任何参数。

## 运行（从spread24_senior_review根）

```powershell
python -B -m pytest src/research96/msr_spread_v2/tests --import-mode=importlib -q -p no:cacheprovider --basetemp=src/research96/msr_spread_v2/tests/.tmp
python -B -m src.research96.msr_spread_v2.run_experiment --mode smoke --date 2026-01-05 --date 2026-01-15 --device cpu --profile scientific
```

CPU支持，不要求CUDA。smoke仅1 epoch A/B/C，显式取最新46天mature debug历史；
保留28日inner tail并执行选定epoch数的fresh refit，不构成scientific FAST24结果。
真实source固定lab-local canonical SHA49fade43...95a2；读取投影过滤不触碰late-Sep/October。
测试使用synthetic/consumed样本，T18另以真实CLI两日验证。

single-day、fast24-lowhigh、fast24-full需要绑定当前源码SHA的独立架构审查PASS receipt；
工程smoke完成不自动开科学指标。receipt含status=PASS、independent_review=PASS、code_hashes逐*.py映射。
FAST24两个mode均运行统一Core Low/High重构（无State或U差异）；日期不可按结果替换。
本次不运行FAST24科学比较，不创建伪造独立review receipt。

## 产物

每次独立timestamp+UUID目录，文件exclusive创建不覆盖：
RUN_MANIFEST.json：source/code/config/依赖/Python/git/seed/resolution/origin/cutoff/命令/科研flags。
每target：DISTRIBUTIONS.npz（组件knots/scenarios/最终Spread及RT/依赖/fusion/clamp）、
MODEL.pt、PREPROCESSOR.json、FEATURE_REGISTRY.json、SPECTRAL_GATES.json、TRAINING.json、
GRADIENT_AUDIT.json、DAY_MANIFEST.json、METRICS.json、SLOT_METRICS.csv、TIMINGS.json。
sealed/target下prediction arrays与SHA在evaluation truth reader之前写入。
汇总DAILY_METRICS.csv、MONTHLY_DIAGNOSTICS.csv。每loss到全部gate/TCN/TabM/head/dependence的norm；
final与Low/High辅助loss的共享group cosine；stage freeze检查。
所有输出路径经过contracts.assert_write_path，越出V2 package/output根立即拒绝。

## 限制

工程ready不是科学有效、fresh confirmation或生产promotion。研究假设未认证historical vintage。
u限定[.01,.99]导致tail clamp，明确统计；Sobol Gaussian-copula-like不是完整市场joint law。
核心forecast/price/历史标签缺失拒绝该日；optional primitive以training-only median填补并保留mask；
某feature整个train无finite evidence则fail closed，绝不静默zero-fill。
收敛/FAST24科学价值尚未评估。训练源码与冻结hash/两日smoke由V2-local验收记录审计。

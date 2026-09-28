# spread24_senior_review

24 点山东电力市场价差预测研究包。

## 当前任务

~~~text
forecast origin = D-1 14:00
predict target day D, 24 hours

Y_source = DA - RT
Y_model  = RT - DA
~~~

frozen source 不修改，由 target_adapter.py 显式翻号。

正式 KPI：

~~~text
1. Raw Direction Accuracy
2. Magnitude MAE = MAE(pred |Spread|, true |Spread|)
~~~

## 当前状态

~~~text
SOURCE_GATE_V13=PASS
SEQUENCE_GATE_V13=PASS

samples=1686
X_hist=[N,168,7]
X_future=[N,24,259]

dependency evidence=259/259 verified
selector DIR+MAG-ALL=222/259
official TabM 0.0.3=PASS

V2.1 engineering=FROZEN
canonical pre-experiment tests=62 PASS
current full non-destructive tests=91 PASS
real one-day smoke=PASS
A0/A1/A2 engineering rerun=PASS

FULL_JAN_AUG_DEV=NOT_RUN
LOCKBOX=NOT_TOUCHED
~~~

当前阶段：

~~~text
READY_FOR_E0_E1_DIRECTION_FIRST_EXPERIMENTS
~~~

不再继续修改底层工程，除非实验暴露真实 bug。

## 文档

只阅读：

1. docs/00_项目总览与文档索引.md
2. docs/01_业务数据与防泄漏合同.md
3. docs/02_历史基线与实验结论.md
4. docs/17_最终模型设计与编码规范.md
5. docs/21_正式运行、验收与实验入口规范.md

## 正式入口

~~~text
python run_tabm_v21.py ...
~~~

默认配置：

~~~text
src/config_tabm_v21.yaml
~~~

历史 src/run_tabm_v2.py 已 retired/fail-closed。

## 安全纪律

- 不修改 frozen source；
- 不放宽 D-1 14:00 / D-2；
- 不使用 target-day actual；
- 不覆盖历史 LightGBM outputs；
- 不 stage/commit，除非用户明确要求；
- 未经实验计划，不运行完整 Jan-Aug DEV 或 lockbox。

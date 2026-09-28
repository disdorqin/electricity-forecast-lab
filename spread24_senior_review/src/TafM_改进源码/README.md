# TafM_改进源码

spread24 V2.1 正式实现目录。

Canonical：

~~~text
docs/01_业务数据与防泄漏合同.md
docs/17_最终模型设计与编码规范.md
docs/21_正式运行、验收与实验入口规范.md
~~~

当前状态：

~~~text
SOURCE_GATE_V13=PASS
SEQUENCE_GATE_V13=PASS
samples=1686

official TabM=PASS
selector DIR+MAG-ALL=PASS
preprocessing/clipping=PASS
member-wise Direction=PASS
direct Magnitude=PASS
Stage A=PASS
Stage B implementation=PASS
Full Retrain implementation=PASS

config provenance=PASS
root CLI=PASS
legacy V2 CLI=RETIRED_FAIL_CLOSED

canonical pre-experiment tests=62 PASS
current full non-destructive tests=91 PASS
real one-day smoke=PASS

ENGINEERING_FROZEN
READY_FOR_E0_E1_DIRECTION_FIRST_EXPERIMENTS
~~~

正式入口：

~~~text
python run_tabm_v21.py ...
~~~

配置：

~~~text
src/config_tabm_v21.yaml
~~~

默认训练模式：

~~~text
stage_a
~~~

Stage B 必须显式请求。

禁止：

- 修改 frozen base；
- 放宽 D-1 14:00 / D-2；
- target-day actual；
- 自写 substitute TabM；
- 绕过 YAML/provenance；
- 恢复旧 V2 CLI；
- 覆盖历史 LightGBM outputs；
- stage/commit。

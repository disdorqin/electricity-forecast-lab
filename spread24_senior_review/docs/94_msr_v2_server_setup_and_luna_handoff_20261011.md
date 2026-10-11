# 94 — MSR-V2 服务器配置与 Luna 交接
日期：2026-10-11。任务仅为同步科研代码、迁移必要数据、建立并验收服务器环境；不启动新科学实验。
配套文件：docs/94_msr_v2_server_code_snapshot_20261011.json（49个源代码/README/T24/包入口文件的本机SHA清单）。

## 0. 可直接交给 Luna 的任务
先完整阅读本文件与仓库AGENTS.md。执行：
1. 审计本地Git工作区，用显式白名单同步当前V2代码、必要合同和本交接文档到GitHub研究分支。
2. 服务器克隆并checkout同一提交，核对本机源代码快照。
3. 私下传输canonical数据，不把市场数据、模型、旧预测/truth上传GitHub。
4. 建立独立Linux Python3.11 CUDA环境，验收imports、TabM真实API、CPU回归、GPU A/B/C synthetic smoke。
5. 准备可从本机执行的SSH非交互连接，交回SERVER_READY.json和SERVER_HANDOFF.md。
到此停下，把科学实验执行权交回Implementation Director。不要自行跑FAST24/消融/R4/R5、改架构、扫参数或开新truth。
本轮用户允许未来Luna同步Git；本交接文档作者本轮未stage/commit/push，也未连接/配置服务器。

## 1. GitHub、目录与当前真实状态
GitHub remote：https://github.com/disdorqin/electricity-forecast-lab.git
本机Git仓库根：
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab
科研包目录是仓库下 spread24_senior_review，不是Git仓库根。
读取时本地分支master，HEAD=f5ccb6b01950dfc3ff3b1720fa0e38dd57d19e8f。
这只是同步前本地HEAD，不代表GitHub已包含最新V2；remote可见性、权限及远端HEAD尚需Luna验证。

**关键：当前整个 src/research96/msr_spread_v2/ 与 docs/82、84、86 均为 UNTRACKED。直接clone现有HEAD不足以获得最新V2。**
工作区还有已有README/.gitignore/TafM等修改，不是本轮改动；不得restore、删除或顺手提交。
不要用git add -A或直接把整个V2目录递归stage：测试目录有隐藏scratch/model/truth资产。
用配套snapshot的code_files逐文件白名单；这些文件仅.py/.md与唯一t24_operator.npy，不含生成产物。
src/__init__.py和src/research96/__init__.py已包含在清单中。

同步流程：
- 先保存git status、git diff、git diff --cached证据，确认没有他人已staged文件进入本次commit。
- 分支使用codex/msr-v2-server-handoff-20261011；不强推master，不改变生产仓库。
- 在仓库根将snapshot中的相对路径统一加spread24_senior_review/前缀；只stage白名单文件和明确列出的合同。
- 必要合同：科研包AGENTS.md、docs/README.md、其Mandatory startup reads所列的20份文件、docs/86、87、本94及配套JSON；这些只同步代码/文档，不运行其历史实验。
- snapshot不包含V1源码；V2运行不依赖V1/TafM导入。服务器旧仓库内已有V1/TafM只读。
- 审查git diff --cached --name-status和完整diff，确认无data、outputs、.env、密钥、.pt/.npz/.parquet、__pycache__或测试scratch。
- 源码生成清单原始SHA必须与当前文件一致，否则报告并停下，不刷新snapshot掩盖变化。
- commit/push指定分支；记录新commitSHA，服务器checkout此SHA，不只依赖浮动分支名。
- 把GitHub提交URL及提交前后状态交回。若缺权限/凭据，用现有安全认证或向用户请求，不把token写进文档、脚本或仓库。

示例服务器拉取（WORKROOT由实例实际目录决定，不假定任何平台）：
~~~bash
WORKROOT=/your/persistent/workdir
git clone --branch codex/msr-v2-server-handoff-20261011 \
  https://github.com/disdorqin/electricity-forecast-lab.git "$WORKROOT/electricity-forecast-lab"
cd "$WORKROOT/electricity-forecast-lab"
git checkout <LUNA实际同步完成的COMMIT_SHA>
cd spread24_senior_review
~~~
Gitcheckout可能转换CRLF/LF：snapshot同时记录raw_sha256与utf8_lf_sha256。
先比较raw；若仅EOL变化，逐文件证实LF-normalized SHA一致并记录差异，不能把原Windows review原样宣称为Linux逐字节review。
T24二进制与canonical数据SHA必须精确相同，不适用EOL豁免；任何非EOL代码变化都阻断验收。

## 2. 本机已验证环境（不是猜测）
Windows；conda环境epf-2：
D:\computer_download\environment\conda\epf-2\python.exe
Python3.11.16，RTX4060 Laptop GPU，约8GB VRAM，torch.cuda.is_available()=True，torch CUDA runtime12.6。

| 包 | 实测版本 |
|---|---|
| torch | 2.13.0+cu126 |
| tabm | 0.0.3 |
| rtdl_num_embeddings | 0.0.12 |
| typing_extensions | 4.16.0 |
| numpy | 2.4.6 |
| pandas | 3.0.6 |
| pyarrow | 25.0.1 |
| scikit-learn | 1.9.1 |
| PyWavelets | 1.9.0 |
| lightgbm | 4.7.0 |
| scipy | 1.17.1 |
| pytest | 9.1.1 |

根requirements.txt是早期LightGBM/SHAP配置，不是当前V2完整lock，不要直接依赖它搭V2环境；当前epf-2连PyYAML都未安装但V2正常工作。
Windowsconda环境不能直接搬成Linux；创建新环境，不改服务器base/共享科学环境，不复制Windows DLL或site-packages。

## 3. 服务器配置与安装
目标：Linux x86_64；RTX4090/24GB是首选试租，强CPU8–16核、RAM32GB起步，64GB方便；持久SSD。
模型约149810参数，实测峰值allocated约0.4GB/reserved约0.5GB；不是大显存任务。
配置阶段只验收单进程，暂不做并行科学测速、DDP、AMP或torch.compile优化。

~~~bash
conda create -n msr-v2 python=3.11 pip -y
conda activate msr-v2
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export GIT_OPTIONAL_LOCKS=0

python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install \
  numpy==2.4.6 pandas==3.0.6 pyarrow==25.0.1 scipy==1.17.1 \
  scikit-learn==1.9.1 PyWavelets==1.9.0 lightgbm==4.7.0 pytest==9.1.1 \
  tabm==0.0.3 rtdl_num_embeddings==0.0.12 typing_extensions==4.16.0
python -m pip check
~~~
2026-10-11已核对官方cu126索引存在torch2.13.0的cp311 manylinux_2_28_x86_64 wheel。
参考：[官方安装页](https://pytorch.org/get-started/locally/)、[官方cu126 wheel索引](https://download.pytorch.org/whl/cu126/torch/)。
先检查实例nvidia-smi与glibc/架构；不要为安装PyTorch随意重装主机驱动。
若精确版本不可安装，保留报错与来源，交回ENV_COMPATIBILITY_BLOCKED，不擅自大版本升级/降级后宣布复现成功。
不用安装torchvision、torchaudio、Jupyter、SHAP、生产DB客户端等无关依赖。
最终用conda list、pip freeze、Python路径、torch/CUDA/GPU查询记录实际环境；不是只记录安装命令。

## 4. 代码情况、权威与入口
所有相对路径以下都相对服务器科研包根 spread24_senior_review：
- src/research96/msr_spread_v2/：独立V2核心，冻结只读。
- experiments/component_adjudication/（在上述包内）：正式scientific profile、trainer、受控arms。
- config.py：source相对科研包定位；t24.py读取固定本地.npy，不在Linux重算/替换算子。
- dataset.py：只投影合法信息；training_panel截止D-2，正式从2023-01-01 expanding。
- trainer.py：通用trainer；正式实验以component_adjudication/trainer.py的fit_day为当前profile。
- artifacts.py/contracts.py：输出守卫；新结果只能在V2输出目录，exclusive创建不覆盖。

最高架构合同docs/84；训练范式docs/82；GPU scientific profile docs/86。
正式profile：A15–40/B5–12/C10–30，patience7、最新28成熟日inner validation，然后全成熟历史fresh exact refit。
seed20261009、batch64已在本机资格通过、M_TRAIN32/M_EVAL256；lr/weight/loss不变。
Config()默认epoch与docs86不同，不要拿默认30/8/20或smoke1/1/1冒充冻结GPU科学profile。

V1、TafM、生产2.5全部只读。V2不需要生产目录/DB/生产模型，也不要把生产项目搬到服务器。
新服务器尚无完整独立运行入口资格：旧pipeline.py绑定本机review、基线及旧run目录，不是“一条命令自动重新跑所有科学实验”的授权。
原封存REVIEW.json如需迁移，通过私有文件传输保存原件；不要生成伪review或把平台变化静默重新签成PASS。
本轮只建设可用的Python/SSH/GPU/数据基础设施；Director后续负责选择唯一实验和必要的便携output-only orchestration。

## 5. 数据迁移（私有通道，保持布局）
唯一必需canonical文件：
data/r96_wave2_stage_a_20261004/canonical_v2/r96store_v2_3d083991b4fcd963e2ab/EPF_PMOS_96_FULL.parquet
大小：22821821 bytes（约21.8MiB）。
文件SHA256：
49fade43b0f3505a437e8704377bf5ef07f780429f9052caaca74aa3525e95a2

保留上述相对路径，用用户认可的scp/SFTP传输，不进GitHub；服务器hashlib.sha256核对后才验收。
文件含late-September历史行；可做文件hash，不得为检查覆盖率全表to_pandas或打印RT/Spread/Low。
合法读取由原CurveStore.canonical执行origin过滤；配置阶段不materialize target evaluation truth。
October lockbox不能下载、读取或上传。canonical本身不是生产live-vintage证明。

T24数组SHA（对np.load得到数组的tobytes，不是.npy文件字节）：
7bbe44ce66510c8ec24be4bb32446ea7e5de171f532e2727e4bc5fc9e7c704f7
.npy文件SHA另外见code snapshot；两种hash不要混用。

非必需：R1/R3/R4/R5的模型、预测/truth NPZ、旧数据缓存和全部outputs。
现在不需要把整个outputs上传服务器。未来如实验确需封存资产，由Director给精确文件白名单/哈希，用私有传输；不能重新训练一个“差不多”的teacher替代原封存teacher。

## 6. 验收：先CPU与源代码，再CUDA synthetic
验收输出放：
outputs/experiments/msr_spread_v2/server_qualification_20261011/<独立timestamp>/
至少记录Git commit、code snapshot对比、实际env、source/T24 SHA、nvidia-smi、测试stdout与exitcode。

跨平台测试的已知陷阱：
- 核心26项测试中的T01会读取默认outputs/experiments/msr_spread_v2/implementation_20261009/BASELINE.json，比较V1/TafM树。
- Windows旧baseline含.pyc/旧本机状态，直接拿它在Linux比较会失败；缺文件也会失败，不代表模型数学错误。
- 可以在全新服务器、测试前为当地只读V1/TafM实际树生成SERVER_QUALIFICATION_ONLY baseline，写入这一新服务器上尚不存在的默认BASELINE.json；使用audit.tree_hash实际计算，不伪造空树。记录其kind、时点、原snapshot对比结果和新文件SHA。
- 只用于本地树不被测试污染的资格检查，不覆盖/改写任何已有历史baseline。若该路径已存在，停下报告，不覆盖。
- 服务器没有production项目，production审计标记NOT_APPLICABLE_NO_PRODUCTION_PROJECT，不伪称本机生产已验证。
- 不修改核心tests/模型来使测试变绿；不忽略失败、不把deselected算PASS。生成适用baseline后的完整26项必须实跑。

从科研包根：
~~~bash
mkdir -p outputs/experiments/msr_spread_v2/server_qualification_20261011
python -B -m pytest src/research96/msr_spread_v2/tests \
  src/research96/msr_spread_v2/experiments/component_adjudication/tests/test_profile.py \
  src/research96/msr_spread_v2/experiments/component_adjudication/tests/test_metrics.py \
  --import-mode=importlib -q -p no:cacheprovider \
  --basetemp=outputs/experiments/msr_spread_v2/server_test_tmp
~~~
最新同组合在本机35/35通过，其中核心26项覆盖T01–T18含两个synthetic CPU日复现。
测试中1/1/1是明确engineering/synthetic，不构成正式scientific epoch设置。

随后新建输出目录内的qualification脚本（不能改核心），使用tests/conftest.py的synthetic CurveStore：
- build_sample/training_panel只针对合成2023-02-10，不读真实canonical evaluation_label。
- 使用原Preprocessor、scientific_config(64)、make_model(..., FULL_V2)；模型/tensors放CUDA。
- 使用原stage_mode、arm_objectives，分别跑A/B/C各一次forward/backward/optimizer step。
- 检查finite loss/grad、无OOM、阶段冻结、scenario S=L+H、TabM output shape，记录wall time和peak VRAM。
- batch64固定；若不能通过，报告资格阻断，别用变小batch偷偷改科学配置。
- torch.cuda.is_available()必须True。仅nvidia-smi正常或创建一个CUDA tensor不算完整资格通过。
- 此脚本是engineering-only；不选择epoch、不训练真实目标、不调用evaluation_label，不开启FAST24。

## 7. 当前科学状态（不要复活停止路线）
R5正式：STOP_RECENT_MEMORY_ROUTE，禁止把MED7/MED28替换回Low数值预测、扫历史窗口救实验。
最新Direction-only targeted8：
DIRECTION_PRIOR_SIMULTANEOUS_RECALL_NOT_SUPPORTED。
Direction60.2865%→59.2448%；Balanced54.5037%→54.8188%；
正Recall40.6250%→44.1964%，负Recall68.3824%→65.4412%。
数值/概率全不变，174次覆盖fixed83/broken91；35回归与1139独立检查通过。
没有下一神经训练方案已获冻结授权。服务器配好不等于可以自动重跑/继续调参。

本机最新证据位置：
outputs/experiments/msr_spread_v2/direction_decision_prior_20261011/20261011T025723Z/technical_continuation_02/
首看DIRECTION_PRIOR_ADJUDICATION_ZH.md、RESULTS.json、DELIVERY_MANIFEST.json。
只需把上述小报告作为私有交接材料，不上传NPZ truth/model。
本机单日median135.42s，range122.52–243.17s，正式8日含回归/审计约22分钟。
服务器配置阶段不跑真实单日来测速；通过后Director另冻结单日benchmark，避免环境工程消耗科学truth。

## 8. SSH与“配置后我能直接跑”
只在网页终端能操作不等于Director可直接执行。
Luna必须确认本机Codex可通过SSH到实例，记录实际host/port/user与SSH alias；已有密钥优先，不在仓库保存私钥/密码/token。
端口/认证由用户或平台提供；未知时请求这些缺失信息，不猜测公网可达。

SSH命令的工作目录、conda激活路径、Python绝对路径必须明确。验证非交互模式：
~~~bash
ssh <alias> '<ABS_ENV_PYTHON> -c "import torch; print(torch.cuda.is_available())"'
~~~
回传同样的可执行启动模板，后续日志在服务器独立run目录、后台运行用tmux等；不使用前台交互conda作为唯一入口。
如果服务器只能平台网页访问，返回SSH_ACCESS_BLOCKED；不要声称“Director已经可以直接跑”。

## 9. Luna必须交回的SERVER_HANDOFF.md / SERVER_READY.json
- status=MSR_V2_SERVER_ENV_READY仅当代码、数据、imports/pip check、35测试、GPU A/B/C synthetic资格与本机SSH都通过。
- 若代码/数据/环境已完成但SSH未可用，状态READY_LOCAL_ONLY_SSH_BLOCKED，不混淆。
- GitHub提交链接、branch、commit、服务器git status；与本机snapshot的raw/LF hash差异明细。
- SSH alias/host/port/user（非凭据），服务器绝对repo_root/package_root/conda_prefix/python路径；可直接复制执行的非交互命令。
- OS/CPU/RAM/disk/GPU/driver、Python与全部实际依赖版本、torch CUDA、pip check结果。
- canonical文件绝对路径/bytes/SHA、T24文件SHA与array SHA。
- 测试命令、实际passed count、退出码与log路径；新server-only baseline provenance；GPU资格loss/finite/VRAM/耗时。
- V2源代码前后SHA、V1/TafM实际树前后比较；production=NOT_APPLICABLE，无模型/架构改动。
- no_real_scientific_training=true、no_target_evaluation_truth_opened=true、no_late_september_or_october_truth=true。
- 所有问题与替代版本不能隐藏；不得有credentials字段、密钥或token。
最后只报告配置/验收结果，等待Director接手，不自动扩科学实验。


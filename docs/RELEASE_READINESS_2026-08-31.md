# Release 准备度评估（2026-08-31）

本轮评估范围：`runabfe.py`、`abfe_pipeline.py`、`abfe_preoptimizer.py`、
`ibs_engine.py`、`abfe_core.py`，以及运行依赖、配置、测试、CI、资源与项目文档。
依据当前本地工作区；没有核验远端分支或 CI 状态，没有修改生产代码、历史结果或启动模拟。
这是发布评估，不是五万多行代码的逐行正确性证明。评估期间核心文件仍有更新，
文中行号用于定位检查时的实现；正式验收需要在冻结版本上重新执行。

## 结论与建议发布范围

核心工作流已经形成，不建议为了 release 继续增加新的势函数、采样算法或输入后端。
当前缺口主要是：**最新版本的运行时验证、可安装的交付物、可解释的结果资格、
可复现的基准，以及长期任务的运行保护**。

建议先准备范围明确的研究预览版（例如 `0.1.0a1`，仅为建议版本号）：

- 首个支持目标限定为：经验证的中性配体、可溶体系、softcore、dual-lambda、
  PME charging 与物理目标态的 IBS/TMBAR vanishing。
- Linux CPU 与一种实际验证过的 CUDA 组合先进入支持矩阵；其他平台不先承诺。
- traditional、single-lambda、2D、膜体系、带电路线：各自验收后再扩大支持范围。
- residual sampling 保持显式启用，DEXP 不作为原始 LJ ABFE 路线发布。
- 研究版可以公开尚未完成的方法工作，但不能把代码可运行、软件版本发布和
  科学结果达到生产资格合并成一个“成功”。

即使缩小范围，以下安装、核心回归和复现契约也需要通过后才能称为可使用的预览版。

## 已有基础：不要重复建设

- 有 complex/solvent 两腿、预平衡、路径预优化、采样、分析和热力学循环汇总。
- 有 Boresch attachment/release、LRC、输入与物理协议校验。
- 有协议指纹、随机种子派生、checkpoint、部分多文件提交协议和 resume 验证。
- 有最终腿结果的数值校验，不应再把缺字段、NaN、负误差静默当有效结果。
- 有 `.github/workflows/cpu-ci.yml`、`environment-ci.yml` 和分层测试入口。
- 本次静态清点为 **130 个 `test_*.py` 文件、1309 个测试函数定义**。
  定义数不是 pytest 实际 collected/passed 数，参数化与跳过还会改变运行数量。
- 有安装、迁移、续跑、维护文档及历史证据归档。

因此“加测试”“加 CI”“加 README”都过于笼统；需要补的是这些能力的发布验收闭环。

## 首先修复的两个具体问题

### R1：配置中的种子与 IBS 主入口实际传参不一致

位置：`runabfe.py:2343`、`runabfe.py:5462`、`runabfe.py:6083`、
`runabfe.py:6369`。

`RunConfig` 会保留配置文件中的 `repeat_seed`；但 `main()` 把局部 `_repeat_seed`
初始化为 `None`，只从 `ABFE_RANDOM_SEED` 赋值，再将局部值传给 IBS 两腿。
traditional 路径则读取 `config.get("repeat_seed")`，两种模式的来源并不统一。

本次抽取并执行实际配置类及 `main()` 中的种子解析语句，未构建 OpenMM Context：

```text
config.json: {"repeat_seed": 12345}
RunConfig.repeat_seed = 12345
ABFE_RANDOM_SEED 未设置时，main._repeat_seed = None
```

影响：用户在 JSON 中指定 seed，配置快照可能记录该值，而 IBS 管线没有收到它。
不能仅凭配置快照宣称独立重复使用了所指定的种子。

建议：建立唯一 resolved seed，明确 CLI/config/environment 优先级，统一传入两腿、
预平衡、attachment 与采样，并记录实际派生映射。可添加 `--seed`，但重点是接线一致。
保留未显式设置 seed 的历史行为；不要为旧 resume 偷补新 seed 或更改随机流。

验收：JSON 单独指定、环境变量单独指定、冲突优先级、traditional/IBS、首次运行/resume
都有行为测试；结果快照与 backend seed ledger 一致。

### R2：状态锁把 PermissionError 判成进程已退出

位置：`abfe_pipeline.py:1295`。

`_PipelineStateLock._pid_is_alive()` 先捕获 `OSError` 返回 False，再捕获
`PermissionError` 返回 True。后者是前者的子类，因此该 True 分支不可达。
本次直接执行原方法，并让进程检查抛出 PermissionError，实际返回 False。

影响：无法确认进程是否存活时，后续 stale-lock 清理可能删除仍在使用的锁。
此复现证明异常处理错误，不代表本次观测到真实生产锁冲突。

建议：先处理 PermissionError；仅在明确确认进程不存在时判 stale。
共享文件系统还需区分 hostname/进程身份，不能把另一节点的 PID 当本机 PID 检查。
补充锁创建后尚未写 PID 的竞争场景测试。

验收：活进程、已退出进程、权限不足、锁元数据写入中、不同主机、竞争清理均有测试。

## 发布前应补齐的交付项

| 优先级 | 补什么 | 当前证据 | 可检查的完成标准 |
|---|---|---|---|
| 预览版前 | 可安装的 Python 包与命令 | `pyproject.toml` 只有工具设置，没有 `[build-system]`、`[project]`、依赖或 console script | 构建 wheel/sdist；在干净目录安装 wheel，离开源码目录仍能运行 help、诊断和示例；安装资源齐全 |
| 预览版前 | 环境与支持矩阵 | `environment.yml` 含 `/home/canna/...` prefix 与 CUDA 12.9 开发工具链；文档写 Python 3.10+，CI 只测 3.12 | 基础环境与 GPU/ML 可选环境分离；声明已测版本组合；干净机器按文档安装成功；不依赖个人路径 |
| 预览版前 | 最新核心回归证据 | 8 月 31 日交接记录明确只有静态通过；本轮也缺必需运行依赖 | 锁定待发布源码，在完整环境跑 CPU 全套，记录 passed/failed/skipped；必需功能不能因 importorskip 被跳过后算通过；GPU smoke 另跑 |
| 预览版前 | 小型端到端 fixture | 现有大量单元、源码契约与实验测试不能单独证明产品入口贯通 | 一个可分发的小体系走实际 CLI，覆盖两腿、attachment、结果落盘、resume、analyze-only；失败/未收敛时准确输出状态，不降低门槛换成功 |
| 预览版前 | 用户可读的配置与运行诊断 | 配置解析集中但不是完整 schema；help 顶层先 import OpenMM；没有独立 doctor/dry-run 命令 | 新增只读 `doctor`、`validate-config`、`dry-run`；给出最终参数及来源、设备、输入依赖、阶段计划、输出位置；诊断不启动 MD |
| 预览版前 | 输出目录与长任务保护 | 有 checkpoint 和短时 pipeline state lock；未见覆盖整个作业生命周期的输出目录独占锁、SIGTERM 处理或磁盘预检 | 重复启动同一输出目录被明确拒绝；调度器中断后可从一致边界续跑；磁盘不足失败可解释；测试多文件写入中断 |
| 预览版前 | 版本、发布清单与维护材料 | 当前 `.git` 不是有效 checkout；无项目级 LICENSE/CITATION；尚无软件版本及发布构建流水线 | 在真实上游 checkout 冻结版本；确定代码和数据授权；加入变更说明、引用信息、支持范围；构建产物不混入原始轨迹、旧副本和开发缓存 |
| 稳定生产版前 | 跨体系 benchmark 与误差验收 | TODO 中仍有基准与统计口径课题；历史最终文件名不代表结果合格 | 对声明支持的范围提供公开输入、冻结配置、独立 seed、环境、原始分析证据和复现脚本；记录不确定度方法与失败项；在运行前固定验收标准 |

配置兼容特别说明：`runabfe.py:2360` 已记录未知键硬拒绝曾因影响 resume 被撤回。
本次不建议恢复这种启动硬门。可先让 `validate-config` 检查未知键和拼写，
运行时告警并标注参数是否被消费；strict 行为只在显式选择或新版本配置中生效。
本次也确认 `temprature=310` 会被保留，而有效 `temperature` 仍为 300；这是配置诊断缺口，
不应借机改变历史 resume 兼容策略。

原子写现状也应准确描述：JSON 已有 fsync/replace，部分 NPY/NPZ/checkpoint 用
固定 `.tmp` 路径再 replace；已有机制不能等同于整个作业独占与所有产物成组一致。
应复用现有提交协议并补故障注入测试，不必重新设计全部存储。

## 科学结果资格应单独完成

### 带电 charge-transfer 仍是条件性阻塞

`abfe_core.py:663` 起显式设置实验状态，C4/C5 均为 False。
`docs/status/BUGFIX_HANDOFF_2026-08-29.md:601` 说明仍缺真实带电体系扫描 artifact。
双腿 reservoir-release correction 的校验与汇总接口已经接入；**接口存在不等于
修正已经测量，更不等于生产资格成立**。

若首发不支持该路线，它可以留在 experimental 边界内，不必阻塞中性研究版。
若 release 宣称带电/膜生产能力，则必须完成真实两腿修正、C4/C5、盒尺寸/锚点/
约束参数敏感性与闭合证据；不能只改 qualification 常量。

### residual sampling 还不是通用、已验证的默认加速功能

`resources/outer_lambda_local_residual/manifest.json` 明确绑定 Atenolol 的
41 原子与键图。`local_residual/openmm_plugin.py:403` 依赖模型资源及原生插件，
默认插件位置仍是源码树中的 build 目录。

若随包发布：应包含资源、许可与插件构建/安装说明，测试 CPU/CUDA 数值一致性、
序列化与关闭时基线不变。若不随包发布：基础入口仍必须可导入且说明如何获取扩展。
不能只分发用户提到的五个文件：`runabfe.py` 还直接导入 `local_residual`。

EXP-030 当前材料已包含三个 repeat，但完整六窗验收与 window_5 覆盖问题仍须保留。
五窗敏感性分析可作为局部证据，不升级成通用生产结论。换体系验证要独立冻结协议。

### 统一“算完”和“可引用”的机器可读状态

已有 `validate_final_leg_result`、阶段收敛门和 charge-transfer qualification；
仍建议在最终 binding result 上形成所有路线一致的状态契约，例如：

```text
execution_status: completed / failed / interrupted
result_status: diagnostic / candidate / validated
qualification_reasons: [...]
uncertainty_method: ...
required_gates: {gate_name: passed / failed / not_evaluated / not_applicable}
software_version + protocol_versions + input_identity + actual_seed_ledger
```

例：`abfe_core.py` 的跨腿构象检查可记录 not_evaluated；
`runabfe.py` 最终结果固定说明 independent_repeats.performed=False。
这些可以保留为诚实的诊断状态，但不能让下游仅凭 `final_binding_results.json` 存在、
退出码 0 或日志“计算完成”就把结果视为已验证。

不确定度工作应遵守现有独立课题边界：先核验跨重复散布、帧选择与现有误差口径，
再决定经过验证的报告方法；本轮不改估计器、不更换 target、不调整历史验收阈值。

## 五个文件分别应补什么

| 文件 | 优先职责 | 暂不建议为首发做的事 |
|---|---|---|
| `runabfe.py` | 单一种子解析、配置来源诊断、统一结果状态、安装后的 CLI smoke | 再扩展一批实验开关 |
| `abfe_pipeline.py` | 修复锁异常分支，作业级资源保护，首次/恢复/分析的端到端一致性 | 没有回归基线就全面重写编排 |
| `ibs_engine.py` | 新版真实 OpenMM 回归、能量/偏置账本与 checkpoint 一致性、完整路径统计验收 | 为获得好看的数字替换物理 target 或放松门槛 |
| `abfe_core.py` | 集中最终结果资格与协议登记，维护物理端点/约束/修正契约 | 再添加未验证的物理路线 |
| `abfe_preoptimizer.py` | 输出完整路径决策记录；默认与非默认分窗的真实调用测试 | 对所有新体系承诺自动收敛 |

五文件首次清点共 **54,599 行**，结束时为 **54,610 行**。检查时 AST 统计最长函数包括：
`IBSWindowManagerDualLambda.run_all_windows` 3674 行、
`ABFEPipeline.run_full_pipeline` 2299 行、`main` 1332 行。
拆分有长期价值，但不是第一件发布工作；先冻结行为并补端到端证据，再按配置、
协议、持久化、采样、分析拆边界。统一协议注册表与迁移说明比一次性搬文件更有用。

## 文档需要同步，而不是再增加平行“当前状态”

- 本轮实际运行 `check_doc_staleness.py`，5 个入口均报 STALE。
  工具识别到的活动前沿是 8 月 24 日、入口为 8 月 12 日；这不代表项目只更新到 24 日。
- `check_doc_crossrefs.py memtodolist` 通过；`exp-docs` 检查失败，3 份材料缺状态字段。
- `docs/TODO.md` 仍将 checkpoint 验证和 GROMACS 自动发现等列为未修，
  新交接文档与代码已有相应修复；不能直接将旧 TODO 复制成 release blocker。
- README 的旧数值保持历史证据身份，另用一个当前支持/资格矩阵链接最新材料。
- 核对文档推荐环境、CLI 帮助、当前 PyMBAR 依赖约束和参考 provenance 的日期，
  不把旧运行环境当成新安装要求。

## 本轮验证与没有验证的内容

已完成：

1. 五个核心文件和 130 个测试文件 AST 解析通过。
2. 直接调用 `tests/test_ci_quality_contract.py` 的三个无参数纯静态检查，均通过；
   这不是 pytest 全套通过。
3. 直接执行源代码中抽取的配置/seed 块、状态锁方法，复现 R1/R2。
4. 运行文档时效与交叉引用检查，结果如上。
5. 检查打包、环境、工作流与资源路径；当前目录没有可用 Git 元数据。

限制：当前解释器找不到 OpenMM、SciPy、PyMBAR、pytest 和 ruff。
没有安装/升级依赖，没有运行完整 pytest、ruff、wheel 安装验证或 CPU/GPU MD。
不能根据本次评估声称最新修复已通过数值与真实体系回归。

## 建议执行顺序

1. 确定首发支持范围，修 R1/R2；保持历史生产数据不变。
2. 补 package/CLI/资源与可移植环境，在真实 checkout 中冻结待测版本。
3. 跑完整 CPU 回归、安装测试与小体系真实入口测试，补作业中断/重复启动验证。
4. 同步 README、支持矩阵、变更说明和结果状态规范，发布有明确限制的研究预览版。
5. 独立完成 benchmark、统计口径与条件性科学验证，再决定稳定生产版的支持范围。

不要求先做完所有实验路线、图形界面、容器、多平台和大规模架构重构。
这些可以按用户需求后续扩展，不能替代上述验收证据。

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

> **状态（2026-09-01）：R1、R2 均已修复**，见下方各自的「处置」小节与
> 「第九轮审查 backlog」B 组。行为测试在 `openmm_dev` 环境下通过；
> 仍未做的是 R1 验收清单里的 "首次运行/resume" 真机场景与 seed ledger 落盘对账。

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

**处置（2026-09-01，已修）**：新增 `_resolve_repeat_seed(args, config)` 作为唯一入口，
优先级 **`--seed` > `ABFE_RANDOM_SEED` > config `repeat_seed`**（与 `RunConfig` 自身
"命令行 > 配置文件 > 预设"同向）；解析结果由 `main()` 回写 `config.data["repeat_seed"]`，
所以 traditional 路径（读 `config.get("repeat_seed")`）与 IBS 路径（读局部 `_repeat_seed`）
从此是同一个值，配置快照不可能再声明一个管线没收到的 seed。同时落
`config.data["repeat_seed_source"]` 并在启动日志打印 resolved 值与来源。
非正整数在启动期 `SystemExit`。三者都不给时返回 `(None, None)` ——
**刻意保留未显式设置 seed 时的历史随机流，不偷补新 seed**。
新增 `--seed` CLI 参数。已实测四种优先级组合与三类非法输入。

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

**处置（2026-09-01，已修）**：三处一起收紧，方向统一为"只在**能证明**锁已无主人时才删"。

1. `_pid_is_alive` 按三类分开处理：`ProcessLookupError` → 确认不存在（唯一算 stale 的信号）；
   `PermissionError` → 进程存在、只是不属本用户 → 算活着；其它 `OSError` → 判不了 → 算活着。
2. 建锁与写 PID 之间的竞态：读到空/不可解析 payload 时不再立即判 stale，
   要求该文件 mtime 已老于 `_EMPTY_PAYLOAD_GRACE_S`（30 s）——正常竞态窗口是微秒级，
   宽限期足以覆盖，同时仍能清理"建锁后被 kill -9"留下的空壳。
3. 共享文件系统：payload 改为 JSON `{"pid", "hostname"}`（仍兼容旧的裸整数写法），
   hostname 与本机不一致时一律视为"还活着"，**绝不删别的节点的锁**；
   写入后加一次 `fsync`，保证别人判断得了归属。

正常路径（无竞争时的获取/释放）行为不变，只改变"判不了"时的默认。

## 发布前应补齐的交付项

| 优先级 | 补什么 | 当前证据 | 可检查的完成标准 |
|---|---|---|---|
| 预览版前 | 可安装的 Python 包与命令 | `pyproject.toml` 只有工具设置，没有 `[build-system]`、`[project]`、依赖或 console script | 构建 wheel/sdist；在干净目录安装 wheel，离开源码目录仍能运行 help、诊断和示例；安装资源齐全 |
| 预览版前 | 环境与支持矩阵 | `environment.yml` 含 `/home/canna/...` prefix 与 CUDA 12.9 开发工具链；文档写 Python 3.10+，CI 只测 3.12 | 基础环境与 GPU/ML 可选环境分离；声明已测版本组合；干净机器按文档安装成功；不依赖个人路径 |
| 预览版前 | 最新核心回归证据 | 8 月 31 日交接记录明确只有静态通过；本轮也缺必需运行依赖 | 锁定待发布源码，在完整环境跑 CPU 全套，记录 passed/failed/skipped；必需功能不能因 importorskip 被跳过后算通过；GPU smoke 另跑 |
| ~~预览版前~~ | ~~小型端到端 fixture~~ | **2026-09-02 用户决定：不作为本仓的缺口，已关闭。** 本仓是**工程区分支**（见 [PROJECT_LAYOUT.md](../PROJECT_LAYOUT.md)），端到端贯通的证据是**真实生产运行本身**——4W53 那次热力学循环闭合就是走真实 `runabfe.py` 入口跑完的，原始轨迹、checkpoint 与产物在 `Atenolol-rank11`，不在本仓 | —（不要再在 `tests/` 里加"走真实 CLI 的小体系 fixture"当验收项；要复核贯通性去看生产运行的原始产物） |
| 预览版前 | 用户可读的配置与运行诊断 | **2026-09-02 完成**：`doctor` / `validate-config` / `config-template` 三个只读命令已落地（`abfe_diagnostics.py`），覆盖依赖版本·平台·GPU·磁盘·GROMACS、配置未知键·取值·路径·参数来源，以及带说明和默认值的配置模板；均不启动 MD。`--help` 从 4.3 s 降到 ~2.2 s（torch/pymbar 改惰性 import）。**`dry-run` 按 2026-09-02 用户决定不做**——它要预测的东西里，静态部分（有哪些键、默认值、输出位置）由 `config-template` + `validate-config` 覆盖，而 Stage 2 的实际窗口划分是运行期 Fisher 探针的结果，任何"预演"都只能是猜 | `config-template` 生成的模板必须零错误通过 `validate-config`（已由 `test_template_round_trips_through_validate_config` 钉住） |
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
`docs/status/BUGFIX_HANDOFF_2026-08-29.md:601`（在 `Atenolol-rank11`，**不在本仓**）说明仍缺真实带电体系扫描 artifact。
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

> **2026-09-01 补充：** 上面这条限制针对的是 08-31 那次评估的解释器环境。
> `openmm_dev` 环境（`/home/ruigengji/miniforge3/envs/openmm_dev`）可用，
> 该环境下 `tests/` 全套为 **1568 passed / 0 failed / 5 skipped / 3 xfailed / 1 xpassed**
> （2026-09-01 最终一次干净全量运行，162 s；改动前基线 1515 passed）。
> R1、R2 与并入的 P2 backlog 已在该环境下逐批验证，见上方「第九轮审查 backlog」。
> **仍然没有任何 GPU / 真实体系回归**，wheel 安装验证与端到端 CLI 测试也仍未做——
> 这两条限制不因 CPU 全套通过而解除。

## 第九轮审查 backlog 已并入本文（2026-09-01）

原 `Atenolol-rank11/0831issue.md` 是第九轮 7 路分片审查的清单，位于**旧工地目录**、
不在主线库里。为免发布验收再去翻另一个仓库，把它的**剩余项**整体并到本文管理。

### 并入时的账目核对（先说清数目，不照抄汇总）

原文档「总汇总」写 **0 P0 / 13 P1 / 43 P2**。逐条点数后，真实情况是：

| 项 | 汇总声称 | 实际正文 | 差异原因 |
|---|---|---|---|
| P1 | 13 | 13 | 一致；**已全部收口**（见下） |
| P2 | 43 | **37** | `abfe_core` 分片的 5 条只有汇总行、**正文完全缺失**；`abfe_pipeline` 分片汇总写 7 条而正文只有 6 条 |

所以 P2 的真实可执行清单是 **37 条**，另有 **5 条 abfe_core P2 无正文**、**1 条 pipeline P2 数目对不上**。
`abfe_core.py` 分片在原文档的状态栏至今仍是「审查中」，却已被计入汇总——这一条本身
就是发布验收的缺口：**不能把一份自己都没写完的清单当成"已审查过"的证据。**

### P1：13 条全部处置完毕

| 批次 | 结果 |
|---|---|
| 前 8 条（2026-08-31） | 5 已修（含 `IBS_BIAS_PROTOCOL_VERSION` 31→32）、2 误报、1 转交 4w53-21 |
| 后 5 条（2026-09-01） | 3 已修（含 `ESS_GATE_PROTOCOL_VERSION` 3→4）、1 误报、1 加固（当前不可达） |

合计：**8 已修、3 误报、1 加固、1 转交**。逐条依据与数值复核留在 `0831issue.md`（在 `Atenolol-rank11`，**不在本仓**）的两个
回填小节里，不在本文重复。三条与发布直接相关的结论：

* `_compute_geom_gradients` 的键角解析梯度错了 **130%~200%**（原报告写 +29%），
  经 `--boresch-source auto` / `orb_simple` 到达，直接决定 `kthetaA/kthetaB`。
* residual 臂的混合覆盖度门此前用错口径（物理 `u_kn` 配 sampling-gauge `f_k`）。
  ΔG 不受影响，但 **EXP-030 candidate 臂的收敛门读数在修复前不可引用**。
* 传统 REMD "resume 清空 DCD" 是误报：`_steps_completed` 没有任何从盘恢复的入口，
  截断是调用方主动作废旧采样后的正确行为。已在测试里钉住这个前提。

### P2：本轮处置结果（37 条）

**已修 30 条 / 加标注 4 条 / 明确暂缓 3 条。** 全量 `tests/`：**1568 passed / 0 failed / 5 skipped**
（含本轮新增 53 条回归；改动前基线是 1515 passed）。

以下按"改了什么行为"分组，只列发布相关的判断，逐条位置见 `0831issue.md`（在 `Atenolol-rank11`，**不在本仓**）正文。

> 📄 **两件本节没有、只在归档技术报告里的东西**：
> [archive/TECH_REPORT_0831issue_P2_2026-09-01.md](archive/TECH_REPORT_0831issue_P2_2026-09-01.md)
> 的 **§7** 给了 `ESS_GATE_PROTOCOL_VERSION` 3→4 为什么不作废缓存的完整论证；
> **§8** 逐条记了改动期间调整的 5 处既有测试锚点（原锚点、为什么改、断言是更强
> 还是更弱）——那是判断「有没有偷偷放松测试」的唯一记录。

**A. 会算错数的（已修，7 条）**

* `abfe_preoptimizer` CDF 构造：`xp[-1] = 1.0` 事后覆盖把最后一个区间宽度从 `w[N-2]`
  变成 `w[N-2]+w[N-1]`，λ[N-2] 权重双重计入。改为用前 N-1 个权重按自身和归一化，
  末端天然为 1.0（实测最后一段 0.65 → 0.53，两处副本同改）。
* `_reduced_energies_for_record`：LJ 尾项系数用"记录列布局位置"索引，而系数按物理 λ
  态编址。当前生产者写恒等布局故不可达，但任何非恒等布局都会静默把尾项配错 λ。
* `pme_offset_charge_square_sum`：`Σq(λ)²` 只算 `λ²Σscale²`，隐含"base 电荷为 0"。
  前提成立但无守护，已就地断言（否则改成分段去电荷时两腿同错、循环里不抵消）。
* `_normalize_softcore_params`：`n_lj <= 0` 会让 λ=0 态的 LJ 尾项系数被静默置零，改为拒绝。
* `_collect_shadow_cross_exclusions`：跨组 1-4 静电在背景力与 shadow 力**两边都不算**。
  Atenolol 非共价 → 当前为 0 影响；改为 fail closed，共价体系不会静默少一项。
* `_safe_boresch_ramp`：灾难判据 `abs(总势能) > 1e5` 对 7 万原子盒恒真（−5e5~−1e6 量级），
  一旦重新接线会把全部正常体系判失败。改判 ΔE。
* `analyze_gradient_and_optimize_path`：仍用 `Var(U_group1)` 而非 `beta²Var[dU/dλ]`，
  且 NaN 样本被替换成前值/0.0 后继续计入方差。已按 **PHY-08 同等处置** fail-closed
  （唯一调用者 `run_preoptimization` 自身零调用方，生产不可达），NaN 改为丢弃。

**B. 口径 / 契约不一致（已修，10 条）**

* **R1（本文原有条目）**：seed 解析三处不统一。新增 `--seed`，优先级
  `--seed > ABFE_RANDOM_SEED > config.repeat_seed`，集中解析后回写 `config.data`，
  IBS 与 traditional 从同一个值出发；三者都不给时**保留历史随机流**，不注入新 seed。
  同时落 `repeat_seed_source`，配置快照不可能再声明一个管线没收到的 seed。
* **R2（本文原有条目）**：`_pid_is_alive` 的 `PermissionError` 分支不可达（`OSError` 子类
  写在后面），"判不了"被当成"已退出"。改为只有 `ProcessLookupError` 算 stale；
  同时修掉建锁与写 PID 之间的竞态（空 payload 需老于宽限期才判残留），
  并把 payload 改成含 hostname 的 JSON（兼容旧裸 PID），**绝不删别的节点的锁**。
* `run_full_abfe_loop` 溶剂腿拿不到 repeat-seed contract（`seed_ledger` 恒 None，
  两条腿不是同一 repeat 的独立腿）。已透传 `repeat_seed`/`leg_name`。
* traditional 两条路径对同一批工件报两个不同的 ±：`err_boresch` 只加在一条路径上。
  按 `combine_binding_free_energy` docstring 的唯一约定（解析量不并入）统一为不并入，
  `boresch_correction_error_kJ_mol` 仍作独立字段落盘；analyze-only 侧补齐同名字段
  （显式 `None` + 原因，不用 0.0 假装误差为零）。
* `--only-complex-charging` / `--only-boresch-attachment` 不校验冻结腿采样温度，
  可拼出跨温度非法求和（kBT 差 ~3%）。已在 `_load_frozen_stage_result` 一处集中比对：
  不一致拒绝，字段缺失则大声告警（不硬拒，免得堵掉合法旧工件）。
* `_strip_unit_suffix` 接受 `_deg` 却按弧度消费（释放项错约 57 倍）→ 直接拒绝
  （与 P1 #5 在 `format_boresch_json` 的处置同源）。
* `pilot_shadow_checkpoint_interval` 在 `abfe_config.json` 文档化为可用开关，
  `main()` 从不透传 → 已透传（两条腿）。
* `--analyze-only` 恢复了 APBS 修正值却把 `apbs_correction_note` 重置为空 → 与值同源恢复。
* `CUDA:N` 写法跳过预平衡/再平衡的显式 Context 释放（裸 `== "CUDA"` 匹配不上）→ 用 base 名比较。
* REMD 种子域 phase 硬编码 `"charging"`（该类同时服务 mixed/vanishing）→ 改为可注入，
  **默认值不变**（改字符串就是改随机流，属协议变更）。

**C. 落盘 / 日志 / 可审计性（已修，13 条）**

* 同进程第二条腿的日志：`logging` FileHandler 每次**追加**从不摘除（第 N 条腿的行写 N 遍、
  第一条腿的文件继续收后面的行），stdout tee 又用 `isinstance` 短路导致第二条腿根本不装 tee
  （它的裸 print 全进第一条腿的文件、自己的 `pipeline.log` 几乎是空的）。改为进程级单例
  「摘旧挂新 + tee retarget」，已实测两条腿各自干净分离、无重复。
* 独立端点 walker 记录与湿种子缓存改**原子写**（`_atomic_save_npz`），读取侧加损坏容错
  （截断 npz 以前会让每次 resume 都崩且指不出该删哪个文件）。
* 2D 测地线寻径失败会静默回退对角线线性路径，返回值与成功路径**无法区分**，
  次优路径被当成功结果写进 `geodesic_path.json` 并被后续 run 复用。现在寻径 provenance
  （`fallback` / `fallback_reason` / `magnitude_gate_dropped_edges`）一起落盘 + 告警。
* 测地线 `|g| > 1e7` 量级闸门弃边数已计入诊断（**阈值未动**——动它会改已验证路径的数值）。
* `window_overlap_records` 与 `local_results` 两个 append 放进同一原子块，
  消除"进了落盘统计却没进协方差链"的孤儿窗口。
* split-half σ 膨胀：缺证据窗口以前静默 `floor=0.0`（看不出哪些窗口其实无实测），
  现在逐窗口 `sigma_floor_unavailable` + 汇总名单/计数；并显式标注 `df_k` 与
  `endpoint_error_after_offset` **仍是 MBAR-only 口径**
  （逐窗口 σ 下界无法无歧义映射到逐态 df_k，随手缩放等于编误差棒）。
* `top1pct_raw_weight` 在 N<100 时退化成"最大单帧"（阈值 0.35 按 N≈330–430 校准）：
  落盘 `n_top_frames` 与 `degenerate_max_single_frame` 标注。**门的行为未改**——
  置 not-evaluable 同样 fail-closed，并不能让小样本窗口通过，真正的修法是门的重新设计。
* 在线 early-stop 的 `step_at_check` 加回 resume 前已完成步数（纯诊断标签）。
* `diagnose_force_breakdown` 标注为"近似重建"：丢 switching / λ 电荷 offset /
  配体内部清零，Group-12 `max|F|` 可能被幻影项主导、误导爆炸源定位。
* 三个 decharging builder 的 `frozen_ll_pairs` 收集后从不读取——**P0-01 赖以成立的
  「既有 L–L exception 已冻结」这个前提此前零守护**。已加共享断言：逐对复核 chargeProd
  与改写前一致、且没有任何 λ offset 指向它们。
* attachment 腿 `n_samples` 的 `max(2, ...)` 下限会让实跑步数**超过设定**且 split-half
  每半只剩 1 帧 → 改为直接拒绝该参数组合（生产 250000/1000 不受影响）。
* 软核告警块每个 λ 态重复打印（K=23 时刷 23 行，淹掉真告警）→ 只打一次。
* `generate_overlapping_windows` docstring 示例与实测不符（银行家舍入，
  `(8,13)` 实为 `(7,13)`）→ 已改正并写明原因。
* 模块常量被捕获为函数默认值（`VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS` 历史 2→6→4）
  → 默认 `None`、函数体内读当前值，与校验方同源。

**D. 明确暂缓（3 条，都需要先做决定，不是漏掉）**

| 条目 | 为什么不在本轮做 |
|---|---|
| 主/生产窗口 checkpoint manifest 与窗口级缓存门不含 repeat-seed 身份 | 往这三处指纹加字段会**作废现有 GPU checkpoint**。生产确实启用 seed_ledger，所以条件插入也躲不开冷启动代价。属于需要用户明确批准的协议变更（先例：`IBS_BIAS_PROTOCOL_VERSION` 32 那次） |
| `build_ibs_dual_system` 的 WCA 力与软核 CV 排除表不一致（WCA 用 `softcore_excl`，CV 用 `full_softcore_excl`；`build_shadow_bridge_system` 口径相反） | 改排除表会**改变能量**。原条目自评置信度低、"后果未验证"，且生产正在跑。需要先在最小体系上量化两种口径的能量差再决定，不能盲改 |
| `compute_boresch_attachment_u_kn` / `attachment_convergence_diagnostics` 零调用方 | 这是 #79 的 attachment 收敛诊断（round-trip 硬门、⟨U_B⟩ 单调性），**当前不在线上执行**。接线＝新增硬门（协议变更、要定阈值）；删除有 att27 先例但 #79 未被撤销。已在源码就地写明这个决定点，不擅自二选一 |

### 仍缺的两件事（发布验收口径）

1. **`abfe_core.py` 分片没审完。** 原文档状态栏是「审查中」，5 条 P2 只有汇总行、无正文。
   本文上方「五个文件分别应补什么」把 `abfe_core.py` 的职责定为"集中最终结果资格与协议登记"，
   而这个文件恰恰是唯一没有分片正文的。补审它属于预览版前的工作。
2. **本轮全部是 CPU/静态验证。** 1515 passed 是完整 CPU 全套（不是静态清点），但
   **没有任何 GPU 运行验证**。受影响最需要真机复验的三处：
   * residual 臂混合覆盖度门换口径后，EXP-030 candidate 臂的门读数（`ess_gate_mixture_gauge`
     应为 `sampling_states`）；
   * 三个 decharging builder 新增的 `frozen_ll_pairs` 断言（若它在真实体系上触发，
     说明 P0-01 的前提本来就不成立，那是一个需要立刻处理的发现，不是这次改动的回归）；
   * 两条腿同进程时的 `pipeline.log` 分离（已用最小复现验证，未在真实两腿运行上确认）。

### 2026-09-02 追加：CLI-01（用户诊断与启动开销）

已落地，全部在 CPU 上验证过（`tests/test_cli_diagnostics_and_lazy_imports.py`，17 passed）：

1. **`runabfe.py doctor`**——只读环境体检：解释器、必需/可选依赖版本（pymbar 偏离
   契约版本 4.2.0 会告警）、OpenMM 真实可用平台、`nvidia-smi` 的 GPU/显存、
   GROMACS 四条入口的解析结果、输出目录所在盘余量、原生插件已构建的 `.so`。
2. **`runabfe.py validate-config`**——只读配置检查：未知键与拼写建议
   （`temprature` → `temperature`）、类型与取值（约束从真 parser 上读，不维护第二份
   参数表）、输入路径存在性、GROMACS include 解析、关键参数的**来源**
   （配置文件／预设／argparse 默认值）。
   两个命令都支持 `--json`，有错误时退出码 1。
3. **不是启动硬门**——未知键仍然照常合并进运行配置；拼写检查只在这个命令里生效。
   2026-08-24 那次启动期硬拒绝未知键炸掉 resume 的结论没有被推翻，
   由 `test_run_config_still_accepts_unknown_keys` 钉住。
4. **启动开销**——`torch`/`openmmml`/`pymbar` 在 `abfe_core.py`、`ibs_engine.py` 里
   改成惰性 memoized 探测（`has_orb()` / `has_pymbar()` / `_require_*()`），
   `HAS_ORB`/`HAS_PYMBAR` 作为模块属性仍可读。`--help` 4.3 s → ~2.2 s。
   剩下的 ~2.2 s 主要是 openmm 0.56 s + mdtraj ~1.0 s；mdtraj 同样可以惰性化
   （19 处 `mdtraj.` 调用点），本次没做。
5. **CACHE-01 一并关闭**——`find_gmx_include_dir` 挪进非 cache-only 分支。
6. **`runabfe.py config-template`**——打印可直接改的配置模板：键顺序与人工
   `_comment_*` 说明取自仓库 `abfe_config.json`，取值重新解析
   （预设 > 该文件的生产值 > argparse 默认），机器本地路径与必填输入一律留空，
   运行期回填的 provenance 字段（`repeat_seed_source`、`openmm_cache_only_*` 等）
   不输出。`--all` 追加不常用/实验开关并附 argparse 帮助文本。
   模板零错误通过 `validate-config`，构成"生成 → 改 → 自查"的闭环。
   这条替代了原计划的 `dry-run`（见上方表格该行）。

本条只改 CLI 表层与 import 结构，**没有**改任何 Hamiltonian、估计器、协议指纹或
验收阈值；协议版本号未动。

## 建议执行顺序

1. 确定首发支持范围，修 R1/R2；保持历史生产数据不变。
2. 补 package/CLI/资源与可移植环境，在真实 checkout 中冻结待测版本。
3. 跑完整 CPU 回归、安装测试与小体系真实入口测试，补作业中断/重复启动验证。
4. 同步 README、支持矩阵、变更说明和结果状态规范，发布有明确限制的研究预览版。
5. 独立完成 benchmark、统计口径与条件性科学验证，再决定稳定生产版的支持范围。

不要求先做完所有实验路线、图形界面、容器、多平台和大规模架构重构。
这些可以按用户需求后续扩展，不能替代上述验收证据。

# ABFE-IBS 文档

[项目入口](../README.md) · [中文完整说明](../README_cn.md) · [English](../README_en.md)

本目录是本仓库**唯一**的文档集。2026-08-31 的发布整理把原来并存的两套文档
（`docs/` 与 `curated_project/` 整理版知识库，两边有 46 份逐字重复件）合并成这一份，
并把开发期的过程材料（实验记录、交接单、审计快照、阶段性结论）压缩成
[历史材料 log](HISTORY_LOG.md)。

**原文全部保存在 `Atenolol-rank11` 工作区。** 本仓库是 ABFE-IBS 的**工程区分支**：
只保留发布所需的生产代码、生产回归测试和使用文档。

## 按任务阅读

| 任务 | 文档 |
|---|---|
| 查当前科学状态、结果能不能引用、协议版本 | [STATUS.md](STATUS.md) |
| 查某天改了什么、协议版本为什么跳、破不破缓存 | [CHANGELOG.md](CHANGELOG.md)（全程缩略时间线） |
| 查某一步实现的是哪篇文献的方法、怎么引用 | [METHODS.md](METHODS.md) |
| 安装依赖、准备输入、首次运行 | [GETTING_STARTED.md](GETTING_STARTED.md) |
| CUDA 插件：**随仓库分发**，按 `environment.yml` 建环境开箱可用；只有换了 OpenMM 版本或改了插件源码才要重编（秒级） | [GETTING_STARTED.md](GETTING_STARTED.md)《CUDA 插件》 |
| 理解输出、符号、缓存和续跑 | [OUTPUTS_AND_RESUME.md](OUTPUTS_AND_RESUME.md) |
| 定位常见错误 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| 迁移到新蛋白–配体体系 | [MIGRATING_TO_A_NEW_SYSTEM.md](MIGRATING_TO_A_NEW_SYSTEM.md) |
| 修改源码、运行最低验证 | [MAINTAINING.md](MAINTAINING.md) |
| **查看未完成工作（唯一待办清单）** | [TODO.md](TODO.md) —— 已关闭条目整段进 `archive/`，本文不留 `[x]` 尸体 |
| 判断能不能发布、还缺什么 | [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md) |
| 不确定度口径 | [PYMBAR_UNCERTAINTY_PROTOCOL.md](PYMBAR_UNCERTAINTY_PROTOCOL.md) |
| 查某份历史材料写过什么 | [HISTORY_LOG.md](HISTORY_LOG.md) |
| 看当前流程全貌（一张图） | [current-pipeline.svg](current-pipeline.svg) |
| stage2 溶剂腿那 5.5σ 去哪了（**已关闭 2026-09-10**：参照臂的盒错了） | [STAGE2_SOLVENT_LEG_ERROR_BUDGET.md](STAGE2_SOLVENT_LEG_ERROR_BUDGET.md) |
| 查 stage2 的失效机制（仍是活的参考） | [STAGE2_ROOT_CAUSE_2026-08-28.md](STAGE2_ROOT_CAUSE_2026-08-28.md) |
| 重新设计 stage2 控制流前先读（分窗口/分 λ/分采样量 三轴当前怎么耦合的） | [STAGE2_THREE_AXES_COUPLING_2026-09-11.md](STAGE2_THREE_AXES_COUPLING_2026-09-11.md) |
| 路径最小修补：Type I/II/III + 修补顺序（**部分已实现**，代码里 10 余处引它作设计依据；缺口见 TODO §1） | [PLAN_PATH_REPAIR_2026-09-11.md](PLAN_PATH_REPAIR_2026-09-11.md) |
| 🔑 **stage2 自治控制器：设计 + 实证 + 陷阱清单**（**接手先读这份**；含首次跑通的实证与 10 条真机咬过的坑） | [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](STAGE2_CONTROLLER_DESIGN_2026-09-12.md) |
| **stage2 自治控制器：设计 + 实证 + 十条真机陷阱**（碰 stage2 先读这份，**别重推**） | [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](STAGE2_CONTROLLER_DESIGN_2026-09-12.md) |
| stage2 自治循环逐个 bug 的修复流水账（历史，不是待办） | [STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md](STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md) |
| stage2 自治闭环的**验收指标**（老板定案，唯一指标） | [AUTONOMOUS_STAGE2_LOOP_SPEC_2026-09-11.md](AUTONOMOUS_STAGE2_LOOP_SPEC_2026-09-11.md) |
| stage2 分窗与多采样段：09-10/11 那八个 bug 改了什么（**已全部落地，09-12 归档**；仍是这几处当前行为的唯一出处） | [archive/STAGE2_WINDOW_AND_SEGMENT_REDESIGN_2026-09-11.md](archive/STAGE2_WINDOW_AND_SEGMENT_REDESIGN_2026-09-11.md) |
| 2026-09-09 全仓审计原始记录（**不是待办**，09-12 归档；7 条不修已进 TODO §4，52 处待 GPU 复验进 TODO §3） | [archive/AUDIT_2026-09-09_full_repo.md](archive/AUDIT_2026-09-09_full_repo.md) |
| 给新配体重训 local-residual R1 权重（操作手册） | [RETRAIN_LOCAL_RESIDUAL.md](RETRAIN_LOCAL_RESIDUAL.md) |
| local-residual 换体系**别再踩的坑**（接线部分已被 EXP-033 P1 取代，09-12 归档；§6 仍全部有效） | [archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md](archive/HANDOFF_LOCAL_RESIDUAL_2026-09-11.md) |
| 查 λ-WCA 壳为什么退役（已结案，历史） | [archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md](archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md) |
| 拿独立参照真值对生产结果 | [reference_data/README.md](reference_data/README.md) |
| 查 GPU 性能优化做过什么、结论是什么 | [EXP-031_GPU_OPTIMIZATION_2026-09-09.md](EXP-031_GPU_OPTIMIZATION_2026-09-09.md)（**取代 [09-04 那份](archive/EXP-031_GPU_OPTIMIZATION_2026-09-04.md)，那份三条结论有两条已被推翻、09-12 归档**；正文与脚本在沙箱 `ABFE_IBS_CUDA`） |
| 换配体要重训 local-residual 模型：方案与证据（**P1 闭式重训已接主线 09-12，真机未跑**） | [EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md](EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md) |
| **闭式重训到底落成什么样**（代码已落地、离线全绿、真机一次没跑） | [EXP-033_P1_LANDED_2026-09-12.md](EXP-033_P1_LANDED_2026-09-12.md) |

## 结果与有效性

**全部登记在 [STATUS.md](STATUS.md)。** 那是本仓库唯一声明"当前科学结论"的文档：
主线体系、结果登记表、协议版本、开放问题、状态词和证据保全目录都在那一份。

2026-09-05 之前，同样三张表在 `README.md` / `README_cn.md` / `README_en.md` /
本文件各存了一份，热力学路径版本已经在其中三份里烂成了旧值。**不要再复制过来。**

## 设计与协议

- [design/README.md](design/README.md)：**设计文档状态索引**——哪份是仍在跑的合同、
  哪份已实施、哪份一步没动，逐份对过源码（复核日期 2026-09-02）。读 `design/` 先读这份；
- `design/`：当前合同和候选设计——**提案不等于已实现**；
- [current-pipeline.svg](current-pipeline.svg)：当前 softcore ABFE dual-lambda
  生产流程图（从配置与建系到汇总）；
- `archive/`：只读存档，两类内容——
  - `removed_*.md`：2026-07-27 移除的那几块不可达代码的逐字存档。它们不是历史资料，
    是**防回归凭证**：`tests/test_att27_dead_code_removed.py` 断言这些文件存在且
    不可执行，防止有人把已判定不可达的路线重新引进代码。删掉会让那条防线失效。
  - `LAMBDA_SCHEDULE_CONTRACT.md`：2026-08-31 归档的旧 λ 调度合同——描述的 23 态路径
    当前体系不走。**当前 λ 布点没有现行合同文档，只有代码。**
  - `PROPOSAL_*.md`：2026-08-31 从 `design/` 移入的设计提案——**已实施且已被后续决策覆盖**，
    保留原文只为追溯决策依据，不是待办。每份页首有归档标记，状态见
    [design/README.md](design/README.md)。
  - `TODO_2026-08-06_unreconciled.md`：`TODO.md` 截至 2026-08-06 的主表（1350 行），
    2026-08-31 整段归档，内容一字未改。里面的 `- [ ]` 只表示"当时未完成"，
    需要人逐条对账后才能重新变成待办。
  - `TECH_REPORT_0831issue_P2_2026-09-01.md` / `RUNTIME_ISSUES_2026-09-02.md`：
    2026-09-02 从已撤销的 `docs/status/` 移入。主题都已关闭、结论都已归位到正式文档，
    两份页首的告示写明「哪一节是别处没有的」。**都不是待办。**
  - **2026-09-12 归档的五份**——主题都已结案，每份页首有归档告示写明「结论去了哪、
    还剩什么已抄进 [TODO.md](TODO.md)」。**都不是待办：**

    | 文件 | 为什么归档 | 仍然有效的部分 |
    |---|---|---|
    | `TODO_closed_2026-09-09.md` | `MIGRATE-01`/`PBC-01`/`XFAIL-01`/`XFAIL-02`/`CACHE-01`/`CFG-01` 六条已关闭 | 每条留下的那句规矩 |
    | `AUDIT_2026-09-09_full_repo.md` | 62 条候选已收口（52 修 + 7 不修） | §4 的 GPU 复跑命令（→ TODO `REL-04`） |
    | `STAGE2_WINDOW_AND_SEGMENT_REDESIGN_2026-09-11.md` | 八个 bug 全部落地 | 分窗判据用 `∫g dλ`、f_k 加帧前必须重标定的**唯一出处** |
    | `HANDOFF_LOCAL_RESIDUAL_2026-09-11.md` | 接线部分被 EXP-033 P1 取代 | §6《别再踩的坑》全部有效；§4 的 3 条未修已进 TODO `LR-02`~`LR-04` |
    | `EXP-031_GPU_OPTIMIZATION_2026-09-04.md` | 三条结论有两条被 09-09 那份推翻 | 仅追溯用 |

## 运行期发现往哪写

**没有 `docs/status/` 这个目录了。** 它 2026-09-02 被撤掉——它本身就是维护规则
第 1 条不许有的「平行当前状态文档」，两份内容已按下面的分工归位、原文归档：

| 发现的性质 | 写进哪 |
|---|---|
| 用户会遇到的症状 + 完整因果链 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| 已定位但没修的缺陷 | [TODO.md](TODO.md) —— 归到对应的编号段（`S2-` / `LR-` / `REL-` / `AUDIT-`），**必须带位置和判据** |
| 发布阻塞判断 | [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md) |
| 一次性运行的原始记录（不分析） | 直接进 `archive/`，页首写清结论去了哪 |

> ## 🛑 归档前必查：这两份被**代码注释**引用
>
> 「结案了就进 `archive/`」有例外——下面这些是源码注释的靶子，
> 移动它们**必须在同一次改动里把引用一起改**。
>
> | 文档 | 状态 | 为什么不能随手移 |
> |---|---|---|
> | [reference_data/](reference_data/) | **live** | 真值数据本身；`attribute_stage2_solvent_leg_gap.py` 直接读它 |
> | [STAGE2_SOLVENT_LEG_ERROR_BUDGET.md](STAGE2_SOLVENT_LEG_ERROR_BUDGET.md) | **live**（主题 09-10 已关闭，文档仍是 live 靶子） | 溶剂腿当前口径的唯一出处 |
> | [STAGE2_ROOT_CAUSE_2026-08-28.md](STAGE2_ROOT_CAUSE_2026-08-28.md) | **live 参考** | §3.2/§3.3/§4/§8.2 被多处注释当作现行行为的依据 |
> | [archive/BUG_LOCATION_…](archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md) | 已归档 | §2.9/§2.10 是「`WCA_SHIELD_RETIRED` 为什么是 True」「那段为什么是死代码」「三处断言方向为什么是反的」的**唯一**出处 |
>
> 复核命令：
>
> ```bash
> grep -rn "BUG_LOCATION_stage2\|STAGE2_ROOT_CAUSE\|STAGE2_SOLVENT_LEG\|reference_data" \
>   *.py local_residual/*.py tools/*/*.py
> ```
>
> **2026-09-09 已按这条约定做过一次**：`BUG_LOCATION_…` 移进 `archive/`，
> 同一次改完全部 32 处引用（其中 6 处改指新的 live 文档），
> `docs/` 内 143 条链接校验 0 坏链。
>
> ⚠️ 反面教材：`.py` 里现存 **28 处**指向 `docs/status/` / `docs/experiments/` /
> `docs/handoffs/` 的引用（2026-09-09 实测，此前本节写的 14 是漏数），
> **那些路径在本工程区分支已不存在**，原文在 `Atenolol-rank11`
> （见 [HISTORY_LOG.md](HISTORY_LOG.md)）。那批当时刻意没改，成了永久的债 ——
> 这就是为什么现在移动文档要当场改引用。
>
> `docs/` **内部**链接由 `tests/test_doc_staleness_contract.py::test_docs_internal_links_resolve`
> 钉住（当前 147 条、0 坏链）；上面那 28 处是**代码注释里的路径**，不在该测试范围内。

## 还有一个子目录

- `reference_data/`：**带 provenance 的外部参照真值**——独立于本管线算出来的靶子。
  「生产算对了没有」只能拿这里的数比，不能拿生产自己的数互相比。
  见 [reference_data/README.md](reference_data/README.md)。

## 引用约定：`Atenolol-rank11` 里的材料

本目录的正文里会出现一些**本仓库找不到的路径**，例如 `0831issue.md`、
`docs/status/memtodolist.md`、`BUGFIX_HANDOFF_2026-08-29.md`、
`RESULT_REGISTRY.csv`、`4W53/toluene_hydration_reference.py`。
它们不是断链——**原文在 `Atenolol-rank11` 工作区**，2026-08-31 发布整理时
刻意没有搬进工程区分支，逐份登记在 [HISTORY_LOG.md](HISTORY_LOG.md)。

这类引用尽量都带 `（在 Atenolol-rank11，不在本仓）` 后缀标记。**最容易踩的一个**：
正文里的 `docs/status/xxx.md`（`memtodolist*.md`、`AUDIT_STATUS.md`、
`BUGFIX_HANDOFF_*.md`）全部指 rank11 里的路径。本仓库的 `docs/status/` 已于
2026-09-02 撤销，**不要在本仓里找**。

另外还有两类看着像断链、其实不是的：

- **运行期产物**（`output/final_binding_results.json`、`run_provenance.json`、
  `checkpoints/*.json`、`*.npy`）—— 跑起来才生成，不在版本控制里；
- **计划中还没写的文件**（`design/` 里提到的 `remd_backends.py`、
  `edge_manifest.json` 等）—— 提案不等于已实现，见
  [design/README.md](design/README.md)。

## 文档维护规则

0. 改了代码或协议，往 [CHANGELOG.md](CHANGELOG.md) 加**一行**（规则见该文件末尾）；
1. 稳定用法写入教程；当前科学结论**只**写 [STATUS.md](STATUS.md)（唯一一份，
   别在 README 或别处复制它的表）；体系、日期和实验相关的过程材料写入
   [HISTORY_LOG.md](HISTORY_LOG.md) 或留在 `Atenolol-rank11`。
2. 计划、代码实现、测试通过和科学验证是四种不同状态，不要混为"成功"。
3. 旧状态文档不原地重写为当前版；用替代关系保留历史。
4. 新数字必须附来源、单位、符号、协议身份、有效性和是否可引用。
5. 文档整理不移动或改写 `output*`、轨迹、checkpoint、日志和诊断 artifact。
6. [STATUS.md](STATUS.md) 的日期戳由 `tools/diagnostics/check_doc_staleness.py`
   盯着，协议版本表由同一份契约测试 `tests/test_doc_staleness_contract.py`
   对着源码常量钉住。三份 README 不再声明科学状态，因此不再进追踪表。

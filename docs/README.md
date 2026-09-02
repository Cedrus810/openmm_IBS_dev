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
| 安装依赖、准备输入、首次运行 | [GETTING_STARTED.md](GETTING_STARTED.md) |
| 理解输出、符号、缓存和续跑 | [OUTPUTS_AND_RESUME.md](OUTPUTS_AND_RESUME.md) |
| 定位常见错误 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| 迁移到新蛋白–配体体系 | [MIGRATING_TO_A_NEW_SYSTEM.md](MIGRATING_TO_A_NEW_SYSTEM.md) |
| 修改源码、运行最低验证 | [MAINTAINING.md](MAINTAINING.md) |
| 查看未完成工作 | [TODO.md](TODO.md)（295 行；08-06 旧主表已归档） |
| 判断能不能发布、还缺什么 | [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md) |
| 不确定度口径 | [PYMBAR_UNCERTAINTY_PROTOCOL.md](PYMBAR_UNCERTAINTY_PROTOCOL.md) |
| 查某份历史材料写过什么 | [HISTORY_LOG.md](HISTORY_LOG.md) |
| 看当前流程全貌（一张图） | [current-pipeline.svg](current-pipeline.svg) |
| 查 stage2 为什么算错过、怎么定位的 | [STAGE2_ROOT_CAUSE_2026-08-28.md](STAGE2_ROOT_CAUSE_2026-08-28.md) → [BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md](BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md) |
| 拿独立参照真值对生产结果 | [reference_data/README.md](reference_data/README.md) |

## 结果与有效性

当前符号约定：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

**当前主线体系是 4W53（T4 lysozyme L99A + toluene）。** 完整登记表在
[../README_cn.md](../README_cn.md)《当前科学状态》一节；本页只给判断口径：

- **4W53 `−21.36 ± 0.93 kJ/mol`**（`output_v3_seed20260908`，2026-09-02）——
  实验 −23.10，1.83σ 内。**单 seed，无独立重复；注册状态标签待维护者指定**，
  现阶段不可作最终结论引用。证据见
  [BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md](BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md)。
- **Atenolol `−23.1622 ± 2.5139 kJ/mol`（`output_lrc_fix`）已于 2026-08-24 判定作废**，
  不得再引用。旧的 `+40.8362` 和 `+16.00 kJ/mol` 早已标为无效。

> ⚠ **Atenolol** 那几个数字的原始 artifact、结果登记表（`RESULT_REGISTRY.csv`）和
> 有效性复核记录都在 `Atenolol-rank11`，不在本工程区分支——引用前先回那边核对，
> 不要仅凭本文件转述。**4W53** 的证据在本目录内，可直接查。

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

## 运行期发现往哪写

**没有 `docs/status/` 这个目录了。** 它 2026-09-02 被撤掉——它本身就是维护规则
第 1 条不许有的「平行当前状态文档」，两份内容已按下面的分工归位、原文归档：

| 发现的性质 | 写进哪 |
|---|---|
| 用户会遇到的症状 + 完整因果链 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| 已定位但没修的缺陷 | [TODO.md](TODO.md)《未关闭的代码缺陷》，带编号和位置 |
| 发布阻塞判断 | [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md) |
| 一次性运行的原始记录（不分析） | 直接进 `archive/`，页首写清结论去了哪 |

> ## 🛑 归档前必查：这两份被**代码注释**引用
>
> 上面那条「结案了就进 `archive/`」有两个例外。它们头部都写着"已修复/结案"，
> 因此**最容易被判成一次性记录而归档**，但它们同时是源码注释的靶子：
>
> | 文档 | 被引用处数 | 引用它的文件 |
> |---|---|---|
> | [BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md](BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md) | 合计 **28**（与下一行共享计数） | `ibs_engine.py`(14)、`abfe_pipeline.py`(8)、`runabfe.py`(2)、`tools/diagnostics/`(4) |
> | [reference_data/](reference_data/) | 同上 | 同上 |
>
> 复核命令：
>
> ```bash
> grep -rn "BUG_LOCATION_stage2\|reference_data" *.py local_residual/*.py tools/*/*.py | wc -l
> ```
>
> 那 28 处不是随手写的出处，是**退役决定的依据链**——`WCA_SHIELD_RETIRED` 常量处
> 的长注释、`build_ibs_dual_system` 里死代码分支、三处方向被反转的断言、
> `bias_to_signal_ratio` 的注释，全都指向 `BUG_LOCATION…§2.10` 与 `reference_data/`。
> 那是"这段代码为什么是死的""这个断言方向为什么是反的"的**唯一**解释。
>
> **要移动它们，必须在同一次改动里把那 28 处引用一起改。**
>
> ⚠️ 这不是假想风险，同样的病已经在本仓库发生过：顶层 `.py` 里现存 **14 处**
> 指向 `docs/status/` / `docs/experiments/` / `docs/handoffs/` 的引用，
> **那些路径在本工程区分支已不存在**（见 [HISTORY_LOG.md](HISTORY_LOG.md)）。
> 那批当时刻意没改（移动已经发生，补救是零收益高风险 churn）。
> 现在是**移动之前**，成本几乎为零——别再欠一笔。

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

1. 稳定用法写入教程；体系、日期和实验相关结论写入 [HISTORY_LOG.md](HISTORY_LOG.md)
   或留在 `Atenolol-rank11`，不在本分支新开平行的"当前状态"文档。
2. 计划、代码实现、测试通过和科学验证是四种不同状态，不要混为"成功"。
3. 旧状态文档不原地重写为当前版；用替代关系保留历史。
4. 新数字必须附来源、单位、符号、协议身份、有效性和是否可引用。
5. 文档整理不移动或改写 `output*`、轨迹、checkpoint、日志和诊断 artifact。
6. 文档日期戳由 `tools/diagnostics/check_doc_staleness.py` 盯着，
   契约测试是 `tests/test_doc_staleness_contract.py`。

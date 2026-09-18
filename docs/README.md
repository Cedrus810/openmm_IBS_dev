# ABFE-IBS 文档

[项目入口 (English)](../README.md) · [中文完整说明](../README_cn.md)

本目录是本仓库**唯一**的文档集。本仓库是 ABFE-IBS 的**工程区分支**：只保留发布所需的
生产代码、生产回归测试和使用文档；开发期的过程材料压缩成
[历史材料 log](HISTORY_LOG.md)，原文在 `Atenolol-rank11` 工作区。

> **每份文档自己的页首才是权威。** 本页只是路牌 —— 状态、口径、「还剩什么有效」
> 都写在各文档页首，不在这张表里复制一遍。

## 手册（稳定用法）

| 任务 | 文档 |
|---|---|
| 安装依赖、准备输入、首次运行、CUDA 插件 | [GETTING_STARTED.md](GETTING_STARTED.md) |
| 理解输出、符号、缓存和续跑 | [OUTPUTS_AND_RESUME.md](OUTPUTS_AND_RESUME.md) |
| 定位常见错误 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| 迁移到新蛋白–配体体系 | [MIGRATING_TO_A_NEW_SYSTEM.md](MIGRATING_TO_A_NEW_SYSTEM.md) |
| 改源码前先跑什么、改完该更新哪份文档 | [MAINTAINING.md](MAINTAINING.md) |
| 每一步实现的是哪篇文献的方法、怎么引用 | [METHODS.md](METHODS.md) |
| 不确定度口径 | [METHODS.md](METHODS.md)《PyMBAR uncertainty protocol》 |
| 给新配体重训 local-residual R1 权重 | [RETRAIN_LOCAL_RESIDUAL.md](RETRAIN_LOCAL_RESIDUAL.md) |
| 当前流程全貌（一张图） | [current-pipeline.svg](current-pipeline.svg) |

## 状态与待办

| 要查什么 | 文档 |
|---|---|
| 当前科学状态、结果能不能引用、协议版本 —— **唯一出处** | [STATUS.md](STATUS.md) |
| 某天改了什么、协议版本为什么跳、破不破缓存 | [CHANGELOG.md](CHANGELOG.md) |
| **今天有什么挡路的**（P1） | [TODO.md](TODO.md) |
| 该做但不挡路（P2） | [TODO_P2.md](TODO_P2.md) |
| **现在明确不做**（P3）—— 想重新论证某条前先查这里 | [TODO_P3.md](TODO_P3.md) |
| 能不能发布、还缺什么 | [RELEASE_READINESS_2026-08-31.md](archive/RELEASE_READINESS_2026-08-31.md) |
| 某份历史材料写过什么（原文在 `Atenolol-rank11`） | [HISTORY_LOG.md](HISTORY_LOG.md) |
| 拿独立参照真值对生产结果 | [reference_data/README.md](reference_data/README.md) |

待办按优先级分三份，`TODO.md` 是唯一入口。

> ### 📌 每一条待办都必须带重要等级
>
> **等级写在编号后面的方括号里**，跟着条目走，不靠它在哪个文件里推断：
>
> ```
> - [ ] **S2-N [P1] 块账写侧记 4 个动作、读侧只拦 1 个。** …
> - [ ] **AUDIT-S2-02 [P2] 判断函数已经 1795 行 / 63 个 return。** …
> ```
>
> | 等级 | 含义 | 落在哪 |
> |---|---|---|
> | `[P1]` | **挡住在推的工作** | [TODO.md](TODO.md) |
> | `[P2]` | 该做，但不挡路 | [TODO_P2.md](TODO_P2.md) |
> | `[P3]` | **现在明确不做**（判定不改 / 不是 bug / 暂停 / 押后），每条都要写「理由别重新论证」 | [TODO_P3.md](TODO_P3.md) |
>
> 等级和文件是**两份记录同一件事**，所以：**改优先级 = 同时移动条目和改方括号**，
> 只改一处就会出现「躺在 `TODO_P2.md` 里标着 `[P1]`」这种查不出来的条目。
> 清点用等级、不用文件：`grep -rn '\[P1\]' docs/`。
>
> 新条目除了等级，还**必须带位置（`文件:行号`）和判据**（怎样算做完）。

已关闭的条目整段移进 `archive/TODO_closed_<日期>.md`，取证在
[archive/TODO_evidence_2026-09-16.md](archive/TODO_evidence_2026-09-16.md)。

## stage-2 控制器（当前主战场）

**按这个顺序读**，前三份是活的，第四份是今天的盘面：

1. [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](archive/STAGE2_CONTROLLER_DESIGN_2026-09-12.md)
   —— 🔑 **接手先读这份**。核心验收指标、设计、实证、十条真机陷阱、别重新论证的事。
2. [PLAN_PATH_REPAIR_2026-09-11.md](archive/PLAN_PATH_REPAIR_2026-09-11.md)
   —— 设计要求的来源（Type I/II/III + 修补顺序）。**部分已实现**，缺口见 `TODO.md` §1。
3. [STAGE2_ROOT_CAUSE_2026-08-28.md](archive/STAGE2_ROOT_CAUSE_2026-08-28.md)
   —— stage-2 的失效机制。**live 参考**，多处源码注释引它。
4. [STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md](archive/STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md)
   —— 🔴 09-18 benchmark 7 次崩溃分诊（只分诊未动手）+
   [STAGE2_OFFLINE_FORENSICS_2026-09-18.md](archive/STAGE2_OFFLINE_FORENSICS_2026-09-18.md)（24 run 误差棒取证）。

历史批次都在 `archive/`，页首各自写清「还剩什么有效」：09-17 整波（`STAGE2_CONTROLLER_WAVE`）、
09-14 六路审计（`CONTROLLER_BUDGET_AUDIT`）、09-11 三轴快照与修复流水账、
09-11 验收规格原件。**别当待办读。**

## 其它活文档

| 主题 | 文档 |
|---|---|
| 溶剂腿那 5.5σ 去哪了（主题 09-10 已关闭，文档是 **live 靶子**） | [STAGE2_SOLVENT_LEG_ERROR_BUDGET.md](archive/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md) |
| 换配体重训 local-residual：方案与证据 | [EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md](archive/EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md) |
| 闭式重训落成什么样（P1 已落地，`EXP-033-P2` 仍开着） | [EXP-033_P1_LANDED_2026-09-12.md](archive/EXP-033_P1_LANDED_2026-09-12.md) |
| stage-2 后分析：单跑演化报告 / 两跑 A/B 对比 | 脚本 [`stage2_ab_report.py`](../stage2_ab_report.py)（`<run_dir>`｜`--ab <baseline> <cand>`｜`--json`），清单落在 `<run>/checkpoints/controller_comparison_manifest.json` |

## 三个子目录

- **`design/`** —— 当前合同和候选设计。**提案不等于已实现**：先读
  [design/README.md](design/README.md) 的状态索引（哪份在跑、哪份已实施、哪份一步没动，
  逐份对过源码）。
- **`reference_data/`** —— 带 provenance 的**外部**参照真值，独立于本管线算出来的靶子。
  「生产算对了没有」只能拿这里的数比，不能拿生产自己的数互相比。
- **`archive/`** —— 只读存档。**一份都不是待办**，每份页首都写了「为什么归档 / 结论去了哪 /
  还剩什么有效」；`ls docs/archive/` 就是目录。其中四类不要按"过期资料"处理：

  | 文件 | 性质 |
  |---|---|
  | `removed_*.md` | **防回归凭证，不能删** —— `tests/test_att27_dead_code_removed.py` 断言它们存在且不可执行，防止已判定不可达的路线被重新引进代码 |
  | `LAMBDA_SCHEDULE_CONTRACT.md` | 旧 λ 调度合同，描述的 23 态路径当前体系不走。**当前 λ 布点没有现行合同文档，只有代码** |
  | `TODO_2026-08-06_unreconciled.md` | `TODO.md` 截至 2026-08-06 的主表（1350 行），一字未改。里面的 `- [ ]` 只表示"当时未完成"，需逐条对账后才能重新变成待办 |
  | 被源码注释引用的几份 | 见下面「归档前必查」 |

> ## 🛑 归档前必查：被**源码注释**引用的文档
>
> 「结案了就进 `archive/`」有例外 —— 有些文档是源码注释和测试的靶子，
> 移动它们**必须在同一次改动里把引用一起改**。**权威是这条命令，不是名单：**
>
> ```bash
> grep -rn "docs/[^\"' ]*\.md" *.py tests/*.py tools/*/*.py local_residual/*.py
> ```
>
> 当前仍在 `docs/` 顶层、且被源码引用的：[STAGE2_ROOT_CAUSE_2026-08-28.md](archive/STAGE2_ROOT_CAUSE_2026-08-28.md)、
> [STAGE2_SOLVENT_LEG_ERROR_BUDGET.md](archive/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md)、
> [PLAN_PATH_REPAIR_2026-09-11.md](archive/PLAN_PATH_REPAIR_2026-09-11.md)、
> [reference_data/](reference_data/)（`attribute_stage2_solvent_leg_gap.py` 直接读它）。
> 已在 `archive/` 而仍被引用的：`BUG_LOCATION_…`（`WCA_SHIELD_RETIRED` 为什么是 True 的唯一出处）、
> `AUDIT_GATES_AND_CRITERIA_2026-09-17.md`、`EXP-031_GPU_OPTIMIZATION_2026-09-09.md`、
> `STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md`、`CONTROLLER_BUDGET_AUDIT_2026-09-14.md`。
>
> **这条约定已经按规矩执行过三次**（09-09 `BUG_LOCATION_…` 32 处引用、09-17 四份、
> 09-18 六份），每次都在同一改动里改完引用并跑链接校验。
>
> ⚠️ 反面教材：`.py` 里现存 **28 处**指向 `docs/status/` / `docs/experiments/` /
> `docs/handoffs/` 的引用（2026-09-09 实测），**那些路径在本分支已不存在**，
> 原文在 `Atenolol-rank11`。那批当时刻意没改，成了永久的债 —— 这就是为什么
> 现在移动文档要当场改引用。
>
> `docs/` **内部**链接由
> `tests/test_doc_staleness_contract.py::test_docs_internal_links_resolve` 钉住；
> 上面那 28 处是**代码注释里的路径**，不在该测试范围内。

## 运行期发现往哪写

**没有 `docs/status/` 这个目录了。** 它 2026-09-02 被撤掉 —— 它本身就是维护规则
第 1 条不许有的「平行当前状态文档」。

| 发现的性质 | 写进哪 |
|---|---|
| 用户会遇到的症状 + 完整因果链 | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| 已定位但没修的缺陷 | **先定重要等级**：`[P1]` → [TODO.md](TODO.md)、`[P2]` → [TODO_P2.md](TODO_P2.md)、`[P3]` → [TODO_P3.md](TODO_P3.md)，归到对应编号段（`S2-` / `LR-` / `REL-` / `AUDIT-`）。**等级写进方括号、位置写 `文件:行号`、判据写怎样算做完，三样缺一不可** |
| 新的科学结论、数值 | [STATUS.md](STATUS.md)（唯一一份），必须附来源、单位、符号、协议身份、有效性、是否可引用 |
| 发布阻塞判断 | [RELEASE_READINESS_2026-08-31.md](archive/RELEASE_READINESS_2026-08-31.md) |
| 一次性运行的原始记录（不分析） | 直接进 `archive/`，页首写清结论去了哪 |

## 看着像断链、其实不是

三类：

- **`Atenolol-rank11` 里的材料** —— `0831issue.md`、`docs/status/memtodolist.md`、
  `BUGFIX_HANDOFF_2026-08-29.md`、`RESULT_REGISTRY.csv` 等，2026-08-31 发布整理时
  刻意没搬进本分支，逐份登记在 [HISTORY_LOG.md](HISTORY_LOG.md)。
  **最容易踩的一个**：正文里的 `docs/status/xxx.md` 全部指 rank11 的路径。
- **运行期产物** —— `output/final_binding_results.json`、`run_provenance.json`、
  `checkpoints/*.json`、`*.npy`，跑起来才生成，不在版本控制里。
- **计划中还没写的文件** —— `design/` 里提到的 `remd_backends.py`、`edge_manifest.json` 等，
  提案不等于已实现，见 [design/README.md](design/README.md)。

## 文档维护规则

0. 改了代码或协议，往 [CHANGELOG.md](CHANGELOG.md) 加**一行**（规则见该文件末尾）；
1. 稳定用法写入教程；当前科学结论**只**写 [STATUS.md](STATUS.md)（别在 README 或别处
   复制它的表）；体系、日期和实验相关的过程材料写入 [HISTORY_LOG.md](HISTORY_LOG.md)
   或留在 `Atenolol-rank11`。**不要新开平行的「当前状态」文档。**
2. **每条待办带重要等级** `[P1]`/`[P2]`/`[P3]`（见上面《每一条待办都必须带重要等级》），
   加条目和改优先级都要同时动方括号和文件，别只改一处。
3. 计划、代码实现、测试通过和科学验证是四种不同状态，不要混为"成功"。
4. 旧状态文档不原地重写为当前版；用替代关系保留历史。
5. 新数字必须附来源、单位、符号、协议身份、有效性和是否可引用。
6. 文档整理不移动或改写 `output*`、轨迹、checkpoint、日志和诊断 artifact。
7. 结案了就进 `archive/`，页首写清「为什么归档 / 结论去了哪 / 还剩什么有效」 ——
   **本页不再逐批复制那张表**，各文档页首才是权威。移动前先过「归档前必查」。
8. [STATUS.md](STATUS.md) 的日期戳由 `tools/diagnostics/check_doc_staleness.py` 盯着，
   协议版本表由 `tests/test_doc_staleness_contract.py` 对着源码常量钉住。
   **改了任何 `*_PROTOCOL_VERSION` 常量，同一次改动里要把 STATUS.md 的表改掉。**

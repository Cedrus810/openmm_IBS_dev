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

## 结果与有效性

当前符号约定：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

`−23.1622 ± 2.5139 kJ/mol` 在 2026-08-12 的整理中登记为 **CANDIDATE**，不是最终结果，
不可作为结论引用。旧的 `+40.8362` 和 `+16.00 kJ/mol` 已标为无效。

> ⚠ 这些数字的原始 artifact、结果登记表（`RESULT_REGISTRY.csv`）和有效性复核记录
> 都在 `Atenolol-rank11`，不在本工程区分支。引用任何数字前先回那边核对状态，
> 不要仅凭本文件转述。

## 设计与协议

- [LAMBDA_SCHEDULE_CONTRACT.md](design/LAMBDA_SCHEDULE_CONTRACT.md)：当前 lambda path 合同；
- `design/`：当前合同和候选设计——**提案不等于已实现**；
- [current-pipeline.svg](current-pipeline.svg)：当前 softcore ABFE dual-lambda
  生产流程图（从配置与建系到汇总）；
- `archive/`：只读存档，两类内容——
  - `removed_*.md`：2026-07-27 移除的那几块不可达代码的逐字存档。它们不是历史资料，
    是**防回归凭证**：`tests/test_att27_dead_code_removed.py` 断言这些文件存在且
    不可执行，防止有人把已判定不可达的路线重新引进代码。删掉会让那条防线失效。
  - `TODO_2026-08-06_unreconciled.md`：`TODO.md` 截至 2026-08-06 的主表（1350 行），
    2026-08-31 整段归档，内容一字未改。里面的 `- [ ]` 只表示"当时未完成"，
    需要人逐条对账后才能重新变成待办。

## 文档维护规则

1. 稳定用法写入教程；体系、日期和实验相关结论写入 [HISTORY_LOG.md](HISTORY_LOG.md)
   或留在 `Atenolol-rank11`，不在本分支新开平行的"当前状态"文档。
2. 计划、代码实现、测试通过和科学验证是四种不同状态，不要混为"成功"。
3. 旧状态文档不原地重写为当前版；用替代关系保留历史。
4. 新数字必须附来源、单位、符号、协议身份、有效性和是否可引用。
5. 文档整理不移动或改写 `output*`、轨迹、checkpoint、日志和诊断 artifact。
6. 文档日期戳由 `tools/diagnostics/check_doc_staleness.py` 盯着，
   契约测试是 `tests/test_doc_staleness_contract.py`。

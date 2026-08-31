# ABFE-IBS

[中文完整说明](README_cn.md) · [Full English README](README_en.md) · [文档导航](docs/README.md)

ABFE-IBS 是一个基于 OpenMM、面向 GROMACS 输入的绝对结合自由能工作流。

本仓库是**工程区分支**：只保留发布所需的生产代码、生产回归测试和使用文档。
成功、失败和已失效路线的完整开发记录、原始轨迹与结果 artifact 都在
`Atenolol-rank11` 工作区，本仓库只保留一份
[历史材料 log](docs/HISTORY_LOG.md) 做索引。

> 当前科学边界：截至 2026-08-12，软件与方法开发已经形成系统证据，但尚无可正式发表的
> Atenolol 最终结合自由能。`output_lrc_fix` 的 `−23.1622 ± 2.5139 kJ/mol`
> 是 `CANDIDATE`，不是最终结论。

## 从这里开始

- 快速了解项目：[中文 README](README_cn.md) / [English README](README_en.md)
- 历史材料索引（实验记录、交接单、审计快照）：[docs/HISTORY_LOG.md](docs/HISTORY_LOG.md)
- 安装、输入与运行：[GETTING_STARTED.md](docs/GETTING_STARTED.md)
- 输出、符号与续跑：[OUTPUTS_AND_RESUME.md](docs/OUTPUTS_AND_RESUME.md)
- 迁移到新体系：[MIGRATING_TO_A_NEW_SYSTEM.md](docs/MIGRATING_TO_A_NEW_SYSTEM.md)
- 完整文档地图：[docs/README.md](docs/README.md)

修改代码后的最低 CPU 检查：

```bash
./tests/run_offline_tests.sh
```

原始 `output*`、验证轨迹、checkpoint、日志和历史结论均为证据，保存在 `Atenolol-rank11`；
文档整理不删除或原地改写它们。


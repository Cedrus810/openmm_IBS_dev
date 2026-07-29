# ABFE-IBS 文档导航

[返回项目首页](../README.md)

## 使用教程

1. [安装、输入与运行](GETTING_STARTED.md)
2. [输出、结果解读与续跑](OUTPUTS_AND_RESUME.md)
3. [常见问题与排障](TROUBLESHOOTING.md)
4. [迁移到新的蛋白–配体体系](MIGRATING_TO_A_NEW_SYSTEM.md)
5. [维护与修改代码](MAINTAINING.md)

## 当前权威入口

- [TODO.md](TODO.md)：唯一的当前行动清单。
- [status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md](status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md)：
  IBS 预热/生产边界、固定 `f_k`、定向补采和 immutable rescue ensemble 协议。
- [status/AUDIT_STATUS.md](status/AUDIT_STATUS.md)：历史审计、修复依据和当前结论。
- [status/VALIDATION_MATRIX.md](status/VALIDATION_MATRIX.md)：代码已完成但仍缺真实
  CPU/GPU/依赖环境证据的项目。
- [current-pipeline.svg](current-pipeline.svg)：当前默认生产调用链。

## 历史结果与状态

- [status/RESULT_2026-07-27_atenolol_rank11.md](status/RESULT_2026-07-27_atenolol_rank11.md)：
  Atenolol 旧轮次结果与排查记录；文档自身已标明不可作为科学结论引用。
- [status/README_STATUS_SNAPSHOT_2026-07-29.md](status/README_STATUS_SNAPSHOT_2026-07-29.md)：
  重写根 README 时迁出的状态快照，仅供审计。
- `status/evidence_2026-07-27/`：对应诊断证据。

## 专题资料

- `design/`：lambda 路径与候选设计。
- `experiments/`：DEXP 等实验分支说明。
- `handoffs/`：专题排障/实验交接快照。
- `archive/`：已被当前文档取代的旧计划与旧待办，只读保留。

## 文档维护规则

1. 可执行工作只写入 `TODO.md`。
2. 代码完成但缺真实环境证据时，记录到 `status/VALIDATION_MATRIX.md`。
3. 验证完成后，在 `status/AUDIT_STATUS.md` 留简短结论并关闭验证项。
4. 教程描述稳定用法；某个体系或某一轮数值只进入 `status/`，不要塞回根 README。
5. 计算输入和结果保持原路径；文档整理不移动 `output*` 或诊断结果。

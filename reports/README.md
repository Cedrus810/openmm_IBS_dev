# 报告、结论与项目记录

[项目入口](../README.md) · [文档导航](../docs/README.md) · [整理版报告入口](../curated_project/02_当前综合报告/README.md)

本目录保存带证据截止日期的综合报告、实验里程碑和项目维护记录。它不替代原始轨迹、
checkpoint、诊断 JSON、日志或结果 artifact；这些仍保留在各自输出目录。

## 当前综合报告（更新至 2026-08-18）

- [导师详细数据中英对照版（2026-08-18 SUM1）](ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA_2026-08-18_SUM1.md)：面向导师的完整中英对照报告，整合方法、数据、验证、失败路线、EXP-020～030、两条 MACE 路线及 EDS/λ-EDS 比较；
- [2026-08-18 决策更新版](ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA_2026-08-18.md)：更新 seeded ABFE 结果、EXP-026～030 状态，并判定 fixed-target sampling 路线、endpoint-constrained path 路线及其与 EDS/λ-EDS 的关系；
- [导师详细介绍数据版](ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA_2026-08-13.md)：30–50 分钟详细汇报总稿，可直接讲述或自行拆分为 PPT；
- [软件进度与技术底稿](SOFTWARE_PROGRESS_AND_TECHNICAL_DRAFT_2026-08-12.md)：总进度、科学边界和论文母稿；
- [当前代码与新设计工作原理](CURRENT_CODE_AND_NEW_DESIGNS_WORKING_PRINCIPLES_2026-08-12.md)：当前实现和设计原则；
- [流程与方法全景](PIPELINE_AND_METHODS_LANDSCAPE_2026-08-12.md)：方法族、生产主线和状态地图；
- [失败路线与证据](DEVELOPMENT_FAILURES_AND_EVIDENCE_2026-08-12.md)：失败、负结果和适用边界。

这些报告职责不同，不应选一个改名成“唯一最终报告”。导师介绍优先使用数据版总稿；技术写作从
技术底稿开始，具体实现、方法比较和负结果再进入其他三份。

## 里程碑与历史记录

- `EXP-012_*.md`：EXP-012 的初始实现和 MM ledger 预检，属于阶段性证据；
- `project/`：工作区恢复、清理和维护 provenance，以及旧行动清单；
- 旧 TODO 和 cleanup manifest 保留用于审计，不是当前行动入口。

当前全局行动以 [docs/TODO.md](../docs/TODO.md) 为准；数字状态以
[RESULT_REGISTRY.csv](../docs/curation/RESULT_REGISTRY.csv) 为准。

## 写作规则

1. 文件名包含证据截止日期或实验 ID。
2. 报告区分事实、推断、限制和下一步，不把计划写成结果。
3. 每个关键数字链接到 artifact，并记录符号、单位、协议和有效性。
4. 失败和无效路线保留；通过新报告或登记表说明替代关系，不改写历史。
5. 报告可以总结结果，但不得复制或覆盖原始计算目录。


# 项目文档导航

根目录只保留运行入口、三份 README、科学参考材料和兼容入口 `todo2.txt`。项目维护文档按用途放在这里。

## 当前入口

- [TODO.md](TODO.md)：唯一的当前行动清单；新问题、状态和优先级只在这里维护。
- [status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md](status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md)：当前 IBS 预热/生产边界、`f_k` 硬锁、定向补采与 immutable rescue ensemble 的权威协议。
- [status/RESULT_2026-07-27_atenolol_rank11.md](status/RESULT_2026-07-27_atenolol_rank11.md)：2026-07-27 那一轮的结合自由能结果与完整排查记录。**结论不可用**（根因 P0-10：Boresch 平衡值陈旧），保留为审计基线；原 `output_lrc_fix/` 已于同日 18:16 清空。
- [status/VALIDATION_MATRIX.md](status/VALIDATION_MATRIX.md)：代码已完成、仍需 CPU/GPU/依赖环境证据的验证项。
- [status/AUDIT_STATUS.md](status/AUDIT_STATUS.md)：历史审计、修复依据和结论。

## 分类

- `status/`：审计、验证矩阵和已结案状态。
- `handoffs/`：某次实验/排障会话的交接快照，不作为全局待办。
- `design/`：尚未批准或仍需论证的方案。
- `experiments/`：DEXP 等实验分支的说明与结果。
- `archive/`：被当前清单取代的旧待办；只读保留，禁止继续追加。

## 维护规则

1. 可执行工作只进 `TODO.md`，不要同时复制到 handoff、audit 和验证矩阵。
2. 代码修完但缺真实环境证据时，从 `TODO.md` 移到 `status/VALIDATION_MATRIX.md`。
3. 完成验证后，在 `status/AUDIT_STATUS.md` 留一条简短结论并从验证矩阵关闭。
4. handoff 只记录复现实验所需上下文，并从顶部链接回对应 TODO/验证编号。

科学论文抽取文件仍保留在根目录及其同名数据目录，因为它们是输入参考资料，不是项目维护文档。

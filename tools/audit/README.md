# 2026-09-17 判据/门审查脚本

产出文档：`docs/archive/AUDIT_GATES_AND_CRITERIA_2026-09-17.md`、
`docs/archive/STAGE2_CONTROLLER_FLOW_2026-09-17.md` §2.5（静态遍历结果）。

全部**只读** benchmark 产物，不改动任何 run 数据。
硬编码数据根：`/home/ruigengji/abfe-benchmark/openmm_IBS/runs`，换机器要改。
只统计蛋白体系（brd4 / cmet / jnk1 / p38）—— 环糊精是本仓库全部分窗参数的标定集，
拿它做验证是循环论证。

| 脚本 | 做什么 |
|---|---|
| `extract_2026-09-17.py` | 逐窗口拉 (K, ∫g, ΔF, maxedge) 与实测 (rawESS, top1%) |
| `validate_2026-09-17.py` | Spearman + 偏相关：哪个预测量真的预测支撑；K 与窗口位置的混淆 |
| `compare_2026-09-17.py` | 逐窗对比 ∫g 与 L=∫√g |
| `premise_2026-09-17.py` | 验证「λ 布点等热力学长度 δ」这个前提（∫g 退役论证的基础） |
| `gates_2026-09-17.py` | 每道质量门 vs 与实验的偏差 |
| `audit_2026-09-17.py` | 全部 ΔG（含归档）+ 自报 σ vs 重复离散 |
| `audit2_2026-09-17.py` | 同上，区分独立重复与「同一份计算的重分析」 |
| `static_controller_2026-09-17.py` | 静态遍历 `_decide_once`：死动作/死出口/死赋值/执行器缺分支/重复谓词/return 后不可达。**纯 AST，不跑任何东西** |
| `indep_2026-09-17.py` | 九道门之间的共线性与有效独立方向数（证明「八道反号」不是八份证据） |
| `audit3_2026-09-17.py` | 窗口级全量（n=170，含未完成 run 的 inprogress 结果） |

运行：`/home/ruigengji/miniforge3/envs/openmm_dev/bin/python tools/audit/<script>.py`

⚠️ `validate` / `extract` 的小样本版本（n=29）产出过一个**已被撤回**的结论，
见审查文档 §4.3。要复现撤回前后的差异就对比 `extract`(n=29) 与 `audit3`(n=170)。

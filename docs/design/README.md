# 设计文档状态索引

复核日期：**2026-09-02**（逐份对当前源码核对过，不是转述）

> 本轮除了复核，还实现了 REMD 的 P0/P2′ 与 RBFE 的 R0，见下方状态表与各计划的第 0 节。

`docs/design/` 里有两类东西，**混在一起读会出事**：
「合同」描述的是**代码现在就在跑的规则**；「提案／计划」描述的是**当时想做的事**，
其中有的做完了、有的一步没动、有的做完之后又被后来的决策覆盖。
每份文件的第 0 节（或页首告示）写了它自己的状态，本表是总览。

| 文件 | 类型 | 状态 | 一句话 |
|---|---|---|---|
| [PLAN_openmm_8_6_remd_backend.md](PLAN_openmm_8_6_remd_backend.md) | 计划 | 🟡 **P0 完成 / P2′ 未接线** | `free_energy_engine.py`：选择器已接进 `--remd-backend`；采样契约已写好但三个 REMDManager 调用点仍走老路；P1 等 OpenMM 升级 |
| [PLAN_rbfe_interface_and_implementation.md](PLAN_rbfe_interface_and_implementation.md) | 计划 | 🟢 **R0 完成** | `rbfe_core.py` + `runrbfe.py`（validate/combine/template）；R1 起未开始；**配体 B 至今未提供，R3 无法开始** |
| [PROPOSAL_rbfe_r1_fragment_mapping.md](PROPOSAL_rbfe_r1_fragment_mapping.md) | 提案 | ⬜ **待决定** | R1 原子映射走「片段级」路线的论证：可复用的四块图论代码、三条候选路线的取舍、以及三个需要用户拍板的问题 |
| [PROPOSAL_periodic_box_geometry_detection.md](PROPOSAL_periodic_box_geometry_detection.md) | 提案 | ⬜ **待决定** | 非长方体输入盒（截角八面体 / 菱形十二面体 / 一般三斜）的早期识别与统一处理：为什么不能靠盒矩阵字面值分类、各输入格式的盒子从哪读、以及一个「所有格式收敛成同一个报告对象」的漏斗设计 |

## 待办前沿（2026-08-31 更新）

**已落地的生产模块**：`free_energy_engine.py`（P0 选择器）、`rbfe_core.py`（R0 契约）。
两者都不改变现有生产行为——选择器恒定解析为 legacy，RBFE 还没有任何入口调用。

本轮新增的生产模块：

| 模块 | 内容 | 是否影响现有生产行为 |
|---|---|---|
| `free_energy_engine.py` | P0 后端选择器（已接线）+ P2′ 采样契约（未接线） | ❌ 无——选择器恒定解析为 legacy |
| `rbfe_core.py` | R0 契约 + 验证 + ΔΔG 汇总 | ❌ 无——ABFE 不 import 它 |
| `runrbfe.py` | R0 CLI：validate / combine / template | ❌ 无——独立入口 |

新增测试 **151** 条（37+36+13+46+19）；全量 `tests/` 由 1342 增至 **1493 passed / 0 failed**，两次连跑一致。

> 🛑 **2026-08-31 用户决定：以下全部暂缓，不要开工。** 原因是整体会有一次更大的
> 更新，届时这些接线点很可能要重做。本轮落地的三个模块都是**行为中立**的
> （选择器恒定解析为 legacy、RBFE 没有任何调用点），可以原样搁置任意长时间，
> 不会腐烂、也不会挡住大更新。
>
> 下面这张表保留下来只是**记录当时的依赖分析**，不是待办清单。大更新之后要重新
> 判断哪些还成立，别直接照做。

按依赖顺序（当时的分析，已暂缓）：
> ⚠ 两个硬前提还没解决：**OpenMM 8.6.0 目前不稳定且没有 GPU 版本**（2026-09-01 实测，
> 已降级回 8.5.2；详见 REMD 计划 §0 的环境前提），P1 因此继续阻塞；
> RBFE 的配体 B 至今没有提供，R3 无法开始。

另有一项与上面的顺序无关、但同样悬着的债：**λ 调度合同没有现行版本**。
旧的 `LAMBDA_SCHEDULE_CONTRACT.md` 已归档（它描述的 23 态路径当前体系不走），
当前 4W53 的 12 态路径只有代码、没有书面合同。要重写就基于
`_greedy_vanishing_window_ranges()` 的实际行为写，**不要**照着归档件改。

## 2026-09-02 新增

[PROPOSAL_periodic_box_geometry_detection.md](PROPOSAL_periodic_box_geometry_detection.md)
起因是「除了膜以外其他盒子可能奇形怪状，我们是不是要有识别和处理的能力」。
核对结论：**三斜盒的核心数学在这个库里基本是对的**（`abfe_core.py:1415` 的
minimum-image 是精确最近格点搜索、所有体积走 `abs(det)`、插件 G3 的 cell list
已按三斜面高算网格），缺的是「识别」——soluble 路径上**没有任何盒型判据**，
而 `_validate_minimum_image`（`abfe_core.py:6344`）这段写得完全正确的面间距校验
**全仓只有一个调用点**（DEXP 路径）。提案本身不改任何数学，只加一个只读的漏斗。

⚠️ 提案 §6 记了一条**本轮明确不碰**的分歧：局部残差那条线上有两套不同的
minimum-image 定义（`local_residual/geometry.py:65` 的 Babai 分量 round
vs. CUDA kernel 的 c→b→a 顺序归约），正交盒下逐位相同、斜盒下可以不同。
EXP-025 G4 的三方等价是在正交盒上验的，结构上抓不到它。要动需要 GPU 且会重开
那套等价验证，必须单独立项。

## 已归档出本目录（2026-08-31）

两份提案「已实施 + 已被后续决策覆盖」，2026-08-31 移入 [`../archive/`](../archive/)。
它们在原知识库里本来就登记在 `04_历史与无效证据/design_proposals/` 下（见
[HISTORY_LOG.md](../HISTORY_LOG.md) 第 42 / 135 行），归档只是把这个分类落实到目录。
**两份都不是待办**，各自页首有归档标记说明为什么。

| 文件 | 去向 | 为什么归档 |
|---|---|---|
| `PROPOSAL_dexp_merge_into_core.md` | `docs/archive/` | 六条改动全部落地；退役 fitter 已整体删除，softcore cutoff 被 MEM-00h 改成 1.0，基线验收对象已作废 |
| `PROPOSAL_frozen_validation_fallback.md` | `docs/archive/` | v25 落地并保留至今；但其描述的「3 次连续候选通过」gate 已被 Candidate-first/Validate-or-Learn v1 整体撤掉 |
| `LAMBDA_SCHEDULE_CONTRACT.md` | `docs/archive/` | 整篇写的是 23 态 + 6 窗固定表；当前 4W53 走 12 态的 ≠23 贪心分组分支，不经过那张表。仅 v21 布点算法本身与历史论证仍可取用 |

⚠ `ibs_engine.py:6306` 的注释按**文件名**引用 `PROPOSAL_frozen_validation_fallback.md`
（不带路径），移动后仍可 grep 到，但下次改那段注释时请顺手补上 `docs/archive/` 前缀。

## 维护规则

沿用 [docs/README.md](../README.md) 的第 2 条：**计划、代码实现、测试通过和科学验证是四种
不同状态**。本目录额外一条：改动本目录任何一份文件时，同步更新本表的状态列和复核日期；
不要在文件正文里原地改写历史结论——用第 0 节的「实施状态」把现实叠上去，保留原文。

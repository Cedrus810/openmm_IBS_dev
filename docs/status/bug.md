# ABFE/IBS 流水线 Bug 审查报告（第 3 版 — 修复复查 · 结案）

> **⚠️ 2026-07-27：本文件是 2026-07-01 那一轮审查的历史结案报告，不是当前状态。**
> 下面"代码层面已经没有已知的阻断性或静默错误结果类 bug"这句只对**那一轮列出的
> P1/P2 项**成立。此后又发现并确认了多个真实缺陷，其中数条至今开着——当前状态
> 一律以 [`../TODO.md`](../TODO.md) 为准，本文件只读、不再更新。
>
> 例如本文件结案之后确认的：P0-8（缺首/末窗口静默产出截断 ΔG，2026-07-27 修）、
> ATT-09（`run_traditional_mode` 与 `run_full_abfe_loop` 完全没加 APBS 修正，
> 2026-07-27 修）、ATT-04（import 期 CUDA 初始化，2026-07-27 修）、
> P2-14（`final_results.json` 对 DEXP 谎报 LRC 已应用，2026-07-27 修），
> 以及仍开着的 P1-13 / P1-14 / ATT-11 与整组 DEXP 缓期项。

前两版分别完成了初次审查（5 文件 ~14560 行）和第一轮修复复查。本版核对了上一轮列出的全部 P1/P2 待办项的当前状态。

**结论：阶段 1（P1，7 项）全部确认修复；阶段 2（P2）除 1 项低优先级代码卫生问题外全部修复。`python -m py_compile` 全量通过，`python runabfe.py self-test` 6/6 通过。** 代码层面已经没有已知的阻断性或静默错误结果类 bug。剩下的距离发布的差距是**运行时验证**，不是代码问题（见文末"仍需做"）。

---

## 阶段 1（P1）逐项复核 — 7/7 已修复

| # | 文件 | 问题 | 状态 | 证据 |
|---|------|------|------|------|
| 1 | `abfe_core.py` | `app` 未导入导致两处 NameError | ✅ 已修复 | 文件头 `from openmm import app, unit`（第11行），`app.Topology()`/`app.GromacsGroFile` 等调用均可正常解析 |
| 2 | `runabfe.py` | `--boresch-source auto` + `--resume` 崩溃 | ✅ 已修复 | `resolve_boresch_restraint` 现在识别 `candidates` 列表结构并按 `boresch_select` 取值（约第1124-1131行），缓存读取与首次生成路径一致 |
| 3 | `ibs_engine.py` | `_collect_softcore_exclusions` 遗漏部分扭转力类型 | ✅ 已修复 | 新增 `CustomTorsionForce`、`RBTorsionForce`（`hasattr` 守卫）分支（约第1077-1084行） |
| 4 | `ibs_engine.py` | NaN/Inf 检测被 `debug_mode` 整体门控 | ✅ 已修复 | `has_bad_values` 判断与 `raise RuntimeError`（约第1913-1923行）已移出 `debug_mode` 判断，无条件执行；仅详细打印仍受 `debug_mode` 控制 |
| 5 | `abfe_preoptimizer.py` | 最小间距去重静默丢点 | ✅ 已修复（优于原建议） | 去重判断改用 `(unique_lambdas[-1]-lam_val) >= (min_spacing-eps)`（第525行），强制端点后加了 `np.isclose` 断言（第537-540行），另外还在插值后新增了显式 NaN 检测与告警（第505-507行），比原来的诊断更清晰 |
| 6 | `runabfe.py` | `_analyze_dual_leg` vdw 阶段缺 checkpoint 短路 | ✅ 已修复（优于原建议） | coul/vdw 现在共用同一循环与短路逻辑（约第1349-1384行），且当能量文件和 checkpoint 都缺失时改为显式 `raise FileNotFoundError` 而不是静默产生 0 |
| 7 | `abfe_pipeline.py` | 溶剂腿 `decoupling_scheme` 静默丢失 | ✅ 已修复 | `solvent_kwargs.setdefault("decoupling_scheme", decoupling_scheme)`（第2195行），并确认 `run_solvent_decoupling` 透传 `**pipeline_kwargs` 无二次硬编码 |

（上一版另修复的"致命" `_run_2d_lambda_stage` 重复 `resume=resume` 关键字导致 `SyntaxError` 的问题，也已确认保持修复状态，`py_compile` 全量通过。）

---

## 阶段 2（P2）逐项复核 — 4/5 已修复，1 项仍待处理

| # | 文件 | 问题 | 状态 |
|---|------|------|------|
| 1 | `abfe_core.py` | `auto_select_boresch_anchors_rmsf` 函数体缺失 | ✅ 已修复 — 现在是完整实现（RMSF 筛选、锚点组合枚举、几何/稳定性过滤、打分选优），`combinations` 已在模块顶部导入，无遗留未定义变量 |
| 2 | `runabfe.py` | `NumpyEncoder` 本地重复定义 | ✅ 已修复 — 本地 `class NumpyEncoder` 已删除，全文件统一使用从 `abfe_core` 导入的版本 |
| 2b | `runabfe.py` | `_sanitize_boresch_params_strict` 未同步 null 防护 | ✅ 已修复 — 改为 `(params.get("boresch_anchors") or params)`，与主函数一致 |
| 3 | `abfe_core.py` | `sync_all_exclusions` 跨 interaction group 无差别并集同步 | ✅ 已修复 — 新增 `_pair_in_scope`，按每个 `CustomNonbondedForce` 的 interaction group 范围只补齐落在其实际作用范围内的 exclusion |
| 4 | `ibs_engine.py` | 死代码 `_create_pure_vdw_softcore_force`；`REMDManager._clear_replica_contexts` 无效 `del` | ✅ 已修复 — 死函数已删除；清理逻辑改为直接重新赋值 `self.contexts = []` 等，不再有无意义的循环变量 `del` |
| 5a | `abfe_preoptimizer.py` | stage1/2 中 `np.interp(...)[::-1]` 自相抵消的多余翻转 | ✅ 已修复 — 两处均已去掉多余 `[::-1]` |
| 5b | `abfe_preoptimizer.py` | Dijkstra 对角跳跃未校验中间格点安全性 | ✅ 已修复 — 新增 `if di > 1 or dj > 1:` 分支，显式遍历途经的所有中间格点校验 `_is_safe_dual_lambda_state`（约第1128-1138行） |
| **5c** | `abfe_preoptimizer.py` | `_normalize_variance_weights` 共享函数未被单 λ 主路径复用 | ⬜ **仍未修复** — `optimize_lambda_path_adaptive`（单 λ 路径）仍是内联的 `log1p` 归一化 + 高 λ×1.5 加密逻辑，`optimize_stage1_decharging`/`optimize_stage2_vanishing`（双 λ 路径）才调用共享函数。两套算法行为依旧不一致。**纯代码卫生问题，不影响正确性，可以延后处理或明确写文档说明二者刻意不同的原因。** |

---

## 仍需做（不是代码 bug，是发布前的验证工作）

代码层面已经没有已知的阻断性问题。距离真正可以发布，剩下的是**上一轮 TODO 里的"阶段 3 验证"**，这部分无法靠静态审查替代：

- [ ] 针对本轮修复的逻辑类 bug（auto+resume 选择、vdw checkpoint 对称短路、去重丢点边界情形）写最小单元测试固化，防止未来回归——这几个都不需要真实 GPU，可以纯 Python 单测覆盖
- [ ] 按 README 建议，用 CPU + 极小步数跑一次端到端 smoke test，确认系统构建、Boresch、复合物腿、溶剂腿全部能跑通：
  ```
  python runabfe.py --config config.json --platform CPU --n-steps-per-window 1000 --n-states-per-stage 4
  ```
- [ ] 用真实体系在目标 GPU 上跑一次生产规模的完整 ABFE，确认收敛判据、overlap、最终 ΔG 数值合理——这一步没法从代码审查里替代，只能靠真实运行验证
- [ ] 确认 `--parallel-stages` + `--resume` 组合在真实多 GPU 环境下表现正常（这两个功能在本轮之前都出过问题，虽然代码层面已修，但没有实跑验证过组合场景）
- [ ] （可选，低优先级）处理 5c：统一或明确区分单/双 λ 路径的密度权重逻辑

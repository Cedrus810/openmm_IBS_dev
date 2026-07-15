# Release 审计遗留问题清单

审计范围：`abfe_pipeline.py` / `ibs_engine.py` / `abfe_core.py` / `runabfe.py` / `abfe_preoptimizer.py`（用户重写版）。

**已修复（1-5，不在本清单）：**
1. `ibs_engine.py` `IBSSampler.collect_energies()` 裸吞异常导致 `e_base` 静默归零污染 MBAR — 已加日志。
2. `abfe_pipeline.py` PBC 分子完整性修复被 `if False` 永久禁用（引用了不存在的 `runabfe.py::center_and_wrap_molecules`）— 已重新启用。
3. `runabfe.py` Boresch 平衡值最后一帧刷新失败时静默回退旧值 — 已加 `is_fallback`/`equilibrium_update_error` 标记，随 `_sanitize_boresch_params_strict` 透传进 provenance。
4. `ibs_engine.py` `run_all_windows` 的 `debug_mode` 默认值为 `True`，且生产采样阶段（原 2198-2267 行）有一段完全不受 `debug_mode` 门控、无条件执行的 CV/力组诊断打印 — 已改默认 `False` 并补上门控，同时删掉未使用的 `state_all`/`E_total_all`。
5. `ibs_engine.py` `run_all_windows` 的 `resume` 只热启动 IBS bias 权重，不检查单个窗口是否已有有效能量文件，中途崩溃后 resume 会重跑所有已完成窗口 — 已加逐窗口的能量文件形状校验与跳过逻辑。

以下是本轮**未处理**、留待后续决定是否修改的问题。

---

## 6. `ibs_engine.py:947-956` — `_normalize_softcore_params` 硬编码覆盖 alpha 参数
**现象**：无论调用方传入什么 `alpha_lj`/`alpha_coul`，都被强制改写为 `0.7`/`0.5`，只打印一行提示。

**影响**：`abfe_core.py::ACESoftcorePotential.optimize_alpha()` 会按配体原子数动态计算 alpha（`abfe_pipeline.py:83-91` 确实调用了它），但计算结果在这里被整体丢弃，`optimize_alpha` 实质上变成死代码。如果这是有意的"生产模式统一锁定softcore软度"，建议：
- 在 `optimize_alpha` 的调用点直接跳过（不算不调），或
- 把这个硬编码值和覆盖逻辑写进 provenance/日志更醒目的位置，避免以后有人以为改了 config 里的 alpha 就会生效。

**建议动作**：确认这是不是有意设计；如果是，删掉 `optimize_alpha` 调用链上的死码路径，保留一处清晰注释说明"生产模式固定 softcore 参数，不做逐配体自适应"。

---

## 7. `abfe_core.py:4249`（`Orbv3DEXPFittingPipeline` 逐帧标注循环）— 异常被吞且不记录内容
**现象**：`except Exception as e: stats["skip_outlier"] += 1; continue`，`e` 从未被打印或记录。

**影响**：一个真正的 bug（数组形状不匹配、context 报错等）会被计入 "outlier" 帧数，只有当成功帧数跌破 30（`fit_parameters` 里的最低样本量检查）才会暴露，且暴露时也看不出具体原因。

**建议动作**：至少在 debug/verbose 模式下把 `e` 打印出来，或者用 `stats.setdefault("skip_outlier_reasons", []).append(str(e))` 之类的方式记录前几条异常，方便事后排查是"正常异常帧"还是"隐藏的代码 bug"。

---

## 8. `abfe_core.py:870-887`（`OrbVacuumContext.__init__`）与 `898` 附近（`OrbBoreschEstimator.__init__`）— 未检查 `HAS_ORB` 就引用 `torch`/`MLPotential`
**现象**：这两个类的构造函数直接使用 `torch.cuda.is_available()` / `MLPotential(...)`，没有先检查模块顶部的 `HAS_ORB` 标志。

**影响**：如果运行环境没装 `torch`/`openmmml`（这两个包体积大、常常是可选依赖），调用这两个类时会抛出裸 `NameError: name 'torch' is not defined`，而不是一条说明"缺少 ORB/ML 依赖，请 pip install openmm-ml torch"的清晰错误。只有真正调用到 ORB 相关 Boresch 估算路径（`--boresch-source auto/simple`）时才会触发，不影响不用 ORB 的运行方式。

**建议动作**：在这两个 `__init__` 开头加一行：
```python
if not HAS_ORB:
    raise ImportError("OrbBoreschEstimator/OrbVacuumContext 需要 torch 和 openmm-ml，请先安装：pip install torch openmm-ml")
```

---

## 9. `runabfe.py` 约 1855-1875 行 vs 1899-1925 行 — 溶剂腿漏传并行参数
**现象**：复合物腿调用 `pipeline.run_full_pipeline(...)` 时传了 `n_workers=config.n_workers, parallel_stages=config.parallel_stages`（约 1872-1873 行），但溶剂腿调用 `pipeline_solv.run_full_pipeline(...)`（约 1907 行起）没有传这两个参数。

**影响**：只是性能不对称——溶剂腿会静默退化为单进程/串行运行，即使用户显式要求了并行，不影响正确性，但会拖慢总运行时间（尤其溶剂腿窗口数通常不比复合物腿少）。

**建议动作**：在溶剂腿的 `run_full_pipeline` 调用里补上同样的 `n_workers=config.n_workers, parallel_stages=config.parallel_stages`。

---

## 可选清理项（Nitpick，不影响正确性，纯代码卫生）

| 位置 | 问题 |
|---|---|
| `abfe_core.py:270` / `1536` / `1976` | `NumpyEncoder` 在两处函数内被局部重复定义，与模块级定义功能等价，属于冗余 |
| `abfe_core.py:1970` | 局部变量 `top`（结果 dict）与同函数更早的 mdtraj `topology` 别名 `top` 撞名，目前无害但是后续编辑的地雷 |
| `abfe_core.py:2426-2431` | 一段不可达的死代码分支（`scan_boresch_1d_pes` 对 `scan_coord != "r"` 已经在更早处抛出 `NotImplementedError`） |
| `abfe_core.py:2577/3143`、`3471/3691` | 章节编号注释 "# 7."、"# 8." 各自重复出现两次，纯格式问题 |
| `abfe_core.py:3664-3671`（`GeometricRestraintEstimator`） | `kr = kBT/var` 没有像 `OrbBoreschEstimator` 那样 clip 到 [100,2000]，理论上可能产出超出 `calculate_boresch_analytical_correction` 接受范围 [50,5000] 的 `kr`，导致运行到后期才报错而不是在估算阶段就发现 |
| `ibs_engine.py:572-574` | 文件中部又重复 `import openmm`/`from openmm import app, unit`/`import numpy as np`，合并模块时的残留 |
| `abfe_pipeline.py:33-36` 与 `43` | `generate_overlapping_windows` 被重复从 `abfe_preoptimizer` 和 `ibs_engine` 两处导入（其中一行注释写着"✅ 保留这个"），指向同一实现所以无害，但属于合并残留 |
| `abfe_pipeline.py:1344-1346` | `_setup_boresch_params` 是硬编码 `return None` 的桩函数；其唯一调用方 `run_preoptimization` 本身也从未被调用，纯死代码 |
| `abfe_pipeline.py:2188` | `run_full_abfe_loop`（完整的 complex+solvent ΔG_bind 编排函数）功能正确但没人调用，`runabfe.py` 自己内联重写了一遍同样的减法逻辑，属于功能重复 |
| `runabfe.py:33-48` | 若干 import（`ACESoftcorePotential`、`UnitFormatter`、`LambdaDependentBoreschForce`、`DualLambdaPreOptimizer`、`build_aces_probe_system_dual_lambda`、`generate_overlapping_windows`）在文件中从未被引用，重写后留下的死 import |
| `runabfe.py:110-116` | GROMACS 安装路径的硬编码兜底列表（`/home/ruigengji/gmx25.1/...` 等），不便携，但已经排在 `--gmx-path`/`GMXDATA` 探测之后，只是最后一道保险 |

---

## 总结

- 🔴 阻塞项（1项）：已修复。
- 🟡 应修项（原 9 项中的 4 项）：已修复 1-5（含子项）。剩余 #6-#9 属于"确认是否有意设计"或"性能/易用性"级别，不阻塞本次 release，但建议在下一轮迭代中处理。
- ⚪ Nitpick：不影响运行正确性，建议找空档批量清理一次即可，不必单独排期。

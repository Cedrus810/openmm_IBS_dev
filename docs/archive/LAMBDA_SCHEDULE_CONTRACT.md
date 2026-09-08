# Stage 2 λ 调度契约（协议 v21）

> **📦 已归档（2026-08-31）——不要当作当前合同读。**
>
> 本文件整篇描述的是 **23 态路径 + 那张 6 窗固定表**。当前生产体系（4W53）跑的是
> **12 态路径**，走 `_greedy_vanishing_window_ranges()` 的 ≠23 分支（window 0 取最小、
> 其余均摊，`abfe_preoptimizer.py:642` 的 2026-08-28 注释），**根本不经过那张表**。
> 协议号 `THERMODYNAMIC_PATH_PROTOCOL_VERSION` 确实仍是 21——但版本号没动不等于
> 合同还成立，2026-08-26/27/28 那批改动（探针网格可变、态数可配、window 0 重新分组）
> 已经把这份文档描述的对象换掉了。
>
> 仍然有效、可以从这里取用的只有两样：**v21 混合度规布点算法本身**（`√g` 弧长 +
> `β(1-λ)` 几何下限，`abfe_preoptimizer.py:515 blended_metric_vanishing_lambdas`），
> 和**为什么不能用纯 Fisher 排点**的历史论证。其余（λ 表、诊断数值、窗口划分、
> 缓存声明）一律不可引用。
>
> 当前 λ 布点要看代码，本仓库暂无对应的现行合同文档——见
> [design/README.md](../design/README.md) 的待办。

> **2026-07-27 原始更新说明。** 本文件此前描述的是 **v20**（固定 17 点 `λ=x²` 锚点 + λ≈1 四点增密
> + 2 个 Fisher bridge，末边 `0.00390625→0`）。代码早已推进到
> `abfe_preoptimizer.py:340 THERMODYNAMIC_PATH_PROTOCOL_VERSION = 21`，布点算法**整体换掉**了，
> v20 的任何特征（平方锚点、Fisher bridge、`0.00390625` 末边）在当前生产路径上都不存在。
> 依据 v20 写的验收条件（尤其 `TODO.md` 的 V-03）无法满足，已一并改写。

## 唯一 λ 的生成（v21）

实现：`abfe_preoptimizer.blended_metric_vanishing_lambdas()`
（`abfe_preoptimizer.py:515`），由 `Stage2Preoptimizer.optimize_stage2_vanishing()`
（`abfe_preoptimizer.py:3142`）经 `redistribute_vanishing_lambda_subdomains()`
（`abfe_preoptimizer.py:834`，调用点 `:914`）调用。

1. **pilot 探针**先在 17 点均匀网格 `λ = linspace(1, 0, 17)` 上测
   `metric_g = β² Var[∂U/∂λ]`（有限差分，`finite_difference_delta = 0.01`），
   **然后由 `_refine_pilot_grid_in_steep_segments()`（`abfe_preoptimizer.py:3051`）
   在陡峭区间里再插点**——每段最多 `refine_extra_points_per_segment`（默认 4，
   `abfe_preoptimizer.py:3151`，CLI `--stage2-refine-extra-points`）个额外探针。
   所以**真机跑法里 `pilot_lambdas` 通常比 17 长**，`VANISHING_PROBE_BASE_STATE_COUNT = 17`
   （`abfe_preoptimizer.py:342`）只是**粗网格基数**，既不是生产态数、也不是最终探针数。
   （window0 ESS 塌缩当初就是靠这个加密修好的，见该函数文档串。）
2. 由 `√g` 的梯形累积得到弧长 `s(λ) = ∫√g dλ`，归一化为 `s_hat ∈ [0,1]`。
3. **混合坐标**：`blended = (1-β)·s_hat + β·(1-λ)`，其中
   `β = VANISHING_GEOMETRIC_FLOOR_WEIGHT = 0.3`（`abfe_preoptimizer.py:349`）。
   在 `blended` 上等分 `final_state_count`（默认 `VANISHING_FINAL_STATE_COUNT = 23`,
   `abfe_preoptimizer.py:343`）份并反解回 λ。
4. 强制 `λ₀ = 1`、`λ₂₂ = 0`，并校验严格单调与
   `vanishing_max_lambda_gap_bound`（`abfe_preoptimizer.py:494`；23 态下 `0.151515`）。

**为什么是混合而不是纯度规**（模块顶部版本史）：v18 纯 Fisher 排点把去耦尾部压成
`0.9225, 0.8382, 0`；v19/v20 反过来完全忽略实测度规、改用写死的平方调度 + 4 个手挑点
+ 2 个 bridge。v21 用几何下限项 `β(1-λ)` 给尾部兜底，同时让度规**真正控制**布点密度。

## 本体系当时的生产路径（23 个唯一 λ）

> ⚠ **2026-08-31：本节是历史快照，不是当前可引用的数字。** 来源目录 `output_lrc_fix/`
> **不在本工程区分支里**，且这条可溶基线线（`output_lrc_fix` + repeat01/02/03）已于
> 2026-08-24 判定作废（协议错误）。下面的 λ 表和诊断数值只用于说明 v21 算法**产出的形状**，
> 不得当作当前体系的生产路径引用。要看当前路径，读实际运行目录下的
> `checkpoints/preopt_dual_vanishing.json`。

来源：`output_lrc_fix/checkpoints/preopt_dual_vanishing.json`，
`path_protocol_version = 21`，
`lambda_placement_method = "fisher_metric_blended_with_geometric_floor_v21"`。

```text
1.000000, 0.923009, 0.853847, 0.793054, 0.737394, 0.684883,
0.638592, 0.595608, 0.554482, 0.515238, 0.477839, 0.441611,
0.410187, 0.379378, 0.349479, 0.319732, 0.289445, 0.258984,
0.227742, 0.196172, 0.155180, 0.100049, 0.000000
```

实测诊断：

| 量 | 值 |
|---|---|
| `total_thermodynamic_length` | 19.3278 |
| `realized_max_edge_thermodynamic_length` | 1.0086 |
| `realized_min_edge_thermodynamic_length` | 0.4263 |
| `max_lambda_gap_bound` | 0.151515 |
| `realized_max_lambda_gap` | 0.100049（末边 `0.100049 → 0`） |

**末边是 `0.100049 → 0`，不是 v20 的 `0.00390625 → 0`。** 任何仍在检查
`0.00390625` 的验收条件都是对着退役协议写的。

## 窗口画线（23 态默认路径）

态数与窗口划分**没有**随 v20→v21 改变（仍是 6 个 ensemble、单边界共享）：

人类使用闭区间：

```text
[0, 4]    5 个状态
[4, 7]    4 个状态，只共享 λ4
[7, 11]   5 个状态，只共享 λ7
[11, 15]  5 个状态，只共享 λ11
[15, 19]  5 个状态，只共享 λ15
[19, 22]  4 个状态，只共享 λ19
```

Python 使用等价半开区间：

```python
[(0, 5), (4, 8), (7, 12), (11, 16), (15, 20), (19, 23)]
```

因此共有 23 个唯一 λ、5 次公共边界复用、28 个窗口采样槽位。每条 λ 边只属于一个窗口；
任何相邻窗口共享两个节点、任何遗漏最终 state 22 的实现都必须拒绝
（`validate_single_shared_boundary_ranges`）。
`provenance.sliding_overlap_states = 0`、`shared_endpoint_states_per_neighbor = 1`
即为该契约的落盘证据。

## 态数已可配置（2026-08-27/28，本文件此前未记）

上面那张 6 窗表是 **23 态默认路径**的契约，**不再是唯一可能的分组**。当前代码：

| 入口 | 作用 | 位置 |
|---|---|---|
| `--stage2-final-n-states` | 最终生产态数，独立于探针网格密度；不传 = 23 | `runabfe.py:3557` |
| `--stage2-refine-extra-points` | 陡峭区间每段额外探针数（默认 4） | `runabfe.py:3558` |
| `--stage2-window-min-states` / `--stage2-window-max-states` | 非 23 态时的贪心分窗上下界（默认 4 / 6） | `runabfe.py:3559` / `:3560` |

`vanishing_subdomain_ranges_from_lambdas()`（`abfe_preoptimizer.py:766`）按态数分岔：

- **恰好 23 态** → 逐字节走 `VANISHING_FIXED_WINDOW_RANGES`（`abfe_preoptimizer.py:350`），
  即上面那张手工调过、含 window0 ESS 塌缩修复的表。行为与本文件描述完全一致。
- **其它态数** → `_greedy_vanishing_window_ranges()`：每窗
  `min_states_per_window..max_states_per_window` 态、贪心从头填满、尾窗不足 min 就并进前一窗。
  **这条分组不是从 23 态那张表反推出来的，两者给出的分组本来就不一样**（新路径没有 window0 的特殊收窄）。

> ⚠ **非 23 态路径的 window0 行为从未在真机 GPU 上验证过**（源码注释原话）。
> 改 `--stage2-final-n-states` 等于走一条只过了静态校验的分组算法，不是走本契约。

## 缓存兼容性

`path_protocol_version` 进预优化缓存指纹，v18/v19/v20 缓存对 v21 一律失效、不得迁移复用。
（历史记录：`output_lrc_fix` 的缓存生成于 2026-07-26 16:54，已是 v21——见上方作废声明。）

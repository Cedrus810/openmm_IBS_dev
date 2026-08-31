# Stage 2 λ 调度契约（协议 v21）

> **2026-07-27 更新。** 本文件此前描述的是 **v20**（固定 17 点 `λ=x²` 锚点 + λ≈1 四点增密
> + 2 个 Fisher bridge，末边 `0.00390625→0`）。代码早已推进到
> `abfe_preoptimizer.py:328 THERMODYNAMIC_PATH_PROTOCOL_VERSION = 21`，布点算法**整体换掉**了，
> v20 的任何特征（平方锚点、Fisher bridge、`0.00390625` 末边）在当前生产路径上都不存在。
> 依据 v20 写的验收条件（尤其 `TODO.md` 的 V-03）无法满足，已一并改写。

## 唯一 λ 的生成（v21）

实现：`abfe_preoptimizer.blended_metric_vanishing_lambdas()`
（`abfe_preoptimizer.py:475`），由 `optimize_vanishing_lambdas_from_metric()`
（`abfe_preoptimizer.py:~676`）调用。

1. **pilot 探针**在固定的 17 点均匀网格 `λ = linspace(1, 0, 17)` 上测
   `metric_g = β² Var[∂U/∂λ]`（有限差分，`finite_difference_delta = 0.01`）。
   `VANISHING_PROBE_BASE_STATE_COUNT = 17` 是**探针**态数，不是生产态数。
2. 由 `√g` 的梯形累积得到弧长 `s(λ) = ∫√g dλ`，归一化为 `s_hat ∈ [0,1]`。
3. **混合坐标**：`blended = (1-β)·s_hat + β·(1-λ)`，其中
   `β = VANISHING_GEOMETRIC_FLOOR_WEIGHT = 0.3`（`abfe_preoptimizer.py:337`）。
   在 `blended` 上等分 `VANISHING_FINAL_STATE_COUNT` 份并反解回 λ。
4. 强制 `λ₀ = 1`、`λ₂₂ = 0`，并校验严格单调与
   `vanishing_max_lambda_gap_bound`（本例 `0.151515`）。

**为什么是混合而不是纯度规**（模块顶部版本史）：v18 纯 Fisher 排点把去耦尾部压成
`0.9225, 0.8382, 0`；v19/v20 反过来完全忽略实测度规、改用写死的平方调度 + 4 个手挑点
+ 2 个 bridge。v21 用几何下限项 `β(1-λ)` 给尾部兜底，同时让度规**真正控制**布点密度。

## 本体系当前生产路径（23 个唯一 λ）

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

## 窗口画线

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

## 缓存兼容性

`path_protocol_version` 进预优化缓存指纹，v18/v19/v20 缓存对 v21 一律失效、不得迁移复用。
当前 `output_lrc_fix` 的缓存生成于 2026-07-26 16:54，已是 v21。

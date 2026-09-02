# `docs/reference_data/` —— 带 provenance 的外部参照真值

> 🛑 **不要移动本目录——源码注释指着它。** 与
> `../BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md` 合计被 **28 处**代码
> 注释引用（`ibs_engine.py` 14 / `abfe_pipeline.py` 8 / `runabfe.py` 2 /
> `tools/diagnostics/` 4）。要移动就必须同一次改完那 28 处。
> 详见 [../README.md](../README.md)《归档前必查》。


这里只放**独立于本管线算出来的**参照值。任何"生产算对了没有"的判断都必须以
这里的数为靶子，而不是拿生产自己的数互相比。

---

## `stage2_vanishing_truth_toluene_2026-09-02.json`

**这是目前唯一带 provenance 的 stage2（λ_vdw 解耦段）真值。** 在此之前，全仓
只有三处**转述**"独立参考真值 ΔG_LJ = −6.26 kJ/mol"（`runabfe.py:3832`、
`abfe_pipeline.py:7144/7171`），不写方向、不写 LRC 口径、不写软核指数 —— 那三处
已经害人两次（见 `docs/STAGE2_ROOT_CAUSE_2026-08-28.md` §9.1）。

### 怎么产生的

`4W53/toluene_hydration_reference.py`：**完整独立实现**——自己建 alchemical
system、**逐态独立采样**（每个 λ 各自平衡 + 生产，绝不重加权）、pymbar MBAR。
CUDA / RTX 2080 Ti / mixed，每态 100k equil + 500k prod @ 1 fs（500 样本），
另有 200k 全耦合预平衡。λ 表用**生产真正跑过的那 23 个 λ_vdw**。

体系：甲苯（15 原子 = 7C+8H）在纯水中，4208 原子 / 4193 约束。
输入取自 `4W53/output/{system_solvent.xml, topology_solvent.cif,
box_vectors_solvent.npy, ligand_indices_solvent.json}`。

### 两份，因为软核指数必须匹配才能逐窗口比

| key | λ 前因子 / 分母 (1−λ) 指数 | ΔG_LJ (λ_vdw 1→0) |
|---|---|---|
| `m1n1` | `λ¹` / `0.5(1−λ)¹ + (r/σ)^6` —— 脚本原生 | **−6.818 ± 0.245** |
| `m2n2` | `λ²` / `0.5(1−λ)² + (r/σ)^6` —— **与生产一致** | **−6.581 ± 0.256** |

生产的指数取自那次运行**自己的**协议指纹：
`stage_protocol_key.payload.aces_softcore_params = {alpha_lj: 0.5,
power_lj: [2,2], alpha_convention: dimensionless_sigma_scaled_v2}`。

**自检**：两者差 0.237、合并 σ 0.354 → **0.67σ 一致**。理论要求如此
（λ=1 恒为精确 LJ、λ=0 恒为 0，端点与指数无关），实测证实 ⟹ `m2n2` 那份的
逐态数可信。

### 用法：三条硬规矩

1. **要跟生产逐窗口比，只能用 `m2n2`。** `m1n1` 的**中间态是不同的哈密顿量**，
   逐窗口 ΔF 不可比 —— 只有总量（λ=1→0）可比。这个坑我踩过一次，差点发出一份
   错的误差归因表。
2. **两份都是 no-LRC。** 生产 stage2 含 LJ 尾项（`build_ibs_dual_system` 逐 λ_vdw
   算系数、`IBSSampler` 每帧加进 `target_energies`）。直接拿这里的数减会
   **重复计 LRC**。历史上 08-29 给过 default 梯子的 LRC 校正 **+2.746**；
   本表未做校正，自己按当次 `lj_tail_lrc_coeff_kj_mol` 折算。
3. **`per_state_Delta_f_kT` 是相对态 0 的累积值，单位 kT（T=300 K）。**
   乘 `kT = 8.314462618e-3 × 300 = 2.49434 kJ/mol` 得 kJ/mol。

### 曲线形状（`m2n2`，kJ/mol）

```
λ=1.0000   +0.000     完全耦合
λ=0.7236  +17.818
λ=0.4978  +22.574     ← 极大值
λ=0.2346   +2.151
λ=0.0000   -6.581     完全解耦（总计）
```

**真值非单调：先升到 +22.6 再降到 −6.58**（先失色散吸引，再塌空腔赚回更多）。
2026-09-01 的生产结果 6 段全正、单调递增到 +38.72 —— **整条下降支一段都没有**。
逐窗口误差归因见 `docs/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md` §2.8。

### 为什么放进 repo 而不是留在 scratchpad

原始产物在会话级 scratchpad 里。实测（另一会话 mlpath-6c 2026-09-02 的经验）：
scratchpad **不会**在会话结束时立刻消失，但它在 `/tmp` 底下、没人知道它什么时候没，
**而且它还在的时候你也不会想起来去找** —— 那比数据真丢了更糟：你会在数据其实还在的
情况下写下一段基于间接推断的结论。这份数据是整条误差归因链的唯一参照物，
代价约 80 分钟 GPU，不能挂在一个清理策略上。

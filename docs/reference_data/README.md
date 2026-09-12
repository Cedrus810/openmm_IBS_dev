# `docs/reference_data/` —— 带 provenance 的外部参照真值

> 🛑 **不要移动本目录——源码注释指着它。** 与
> `../archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md` 合计被 **28 处**代码
> 注释引用（`ibs_engine.py` 14 / `abfe_pipeline.py` 8 / `runabfe.py` 2 /
> `tools/diagnostics/` 4）。要移动就必须同一次改完那 28 处。
> 详见 [../README.md](../README.md)《归档前必查》。


这里只放**独立于本管线算出来的**参照值。任何"生产算对了没有"的判断都必须以
这里的数为靶子，而不是拿生产自己的数互相比。

---

## `stage2_vanishing_truth_toluene_2026-09-02.json`

> 🛑 **这份真值跑在错的密度上，2026-09-10 已判定。别再拿它当靶子。**
>
> 它固定用 `output/box_vectors_solvent.npy` 的**建系盒 43.9496 nm³**，而那个盒比
> 1 bar 平衡密度**大 3.15%**（真值臂自己的输入加个恒压器跑 200 ps NPT 就收缩到
> 42.61 ± 0.27；生产 NPT 预平衡 42.63；生产 stage2 冻结盒 42.75 —— 三个独立测量一致）。
>
> | | ΔG_LJ (kJ/mol) |
> |---|---|
> | 本文件 `m2n2`（建系盒 43.9496） | −6.581 ± 0.256 |
> | **同 λ 表、同 seed，换到生产盒 42.7468** | **−11.490 ± 0.342** |
>
> 差 **−4.896**，把对生产的判读从「差 5.5σ」翻成「差 0.72σ」。
> 全部证据与复现命令见
> [../STAGE2_SOLVENT_LEG_ERROR_BUDGET.md](../STAGE2_SOLVENT_LEG_ERROR_BUDGET.md)。
>
> 新的可用数值在
> `4W53/reference_at_npt_box/hydration_reference/hydration_reference_vdwonly_prodlambda_results.json`
> （13 态、生产盒）。本文件保留只为留痕和 `m1n1`/`m2n2` 软核指数那条自检。

**这是目前唯一带 provenance 的 stage2（λ_vdw 解耦段）真值。** 在此之前，全仓
只有三处**转述**"独立参考真值 ΔG_LJ = −6.26 kJ/mol"（`runabfe.py:3832`、
`abfe_pipeline.py:7144/7171`），不写方向、不写 LRC 口径、不写软核指数 —— 那三处
已经害人两次（见 `docs/STAGE2_ROOT_CAUSE_2026-08-28.md` §9.1）。

### 怎么产生的

`4W53/toluene_hydration_reference.py`：**完整独立实现**——自己建 alchemical
system、**逐态独立采样**（每个 λ 各自平衡 + 生产，绝不重加权）、pymbar MBAR。
CUDA / RTX 2080 Ti / mixed，每态 100k equil + 500k prod @ 1 fs（500 样本），
另有 200k 全耦合预平衡。λ 表用的是 `baseline_v3_pre_wca_retirement_20260901/solvent_leg/checkpoints/preopt_dual_vanishing.json`（λ[1]=0.920887）。
⚠️ **那是 09-01 有壳那次运行的备份，不是它要对照的 09-02 实跑**
（`output_v3_seed20260908/solvent_leg/`，λ[1]=0.919767，逐点差 0.001~0.007）。
端点相同（1.0/0.0）⟹ **总量仍可比**；但**逐窗口比必须把真值插值到实跑 λ**，
否则 win4 那种 dF/dλ≈−157 的段，λ 错 0.0069 就假造出 1.1 kJ/mol 的假残差。
`tools/diagnostics/attribute_stage2_solvent_leg_gap.py` 已代做。
**不要因为这条去重跑真值。**

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
   算系数、`IBSSampler` 每帧加进 `target_energies`）。直接拿这里的数减是
   **拿两个不同的哈密顿量比**。

   ✅ **别手工折算、也别重算——用 `tools/diagnostics/attribute_stage2_solvent_leg_gap.py`**
   （零 GPU，只读）。它**从产物直接读**该次运行真正施加的尾项：
   `energies.npy − sampling_states.npy.T` 是**逐 λ 态的严格常数**（实测 sd~1e-16），
   λ=0 处恰好为 0。重算要猜 V，而生产实际用的 V 比 `box_vectors_solvent.npy` 小 2.7%
   （4W53 那次：重算 +2.746 vs 实际 **+2.823**）。

   4W53 那次：生产 −8.075 − 2.823 = **−10.898**，对上真值 −6.581
   ⟹ 残差 **−4.318（5.5σ）**。

   ⚠️ **真正致命的是"根本不对齐"，不是对齐的方向。** 「加在真值侧」与
   「减在生产侧」**等价**（`−6.581+2.823=−3.758` 对 `−8.075`
   ≡ `−8.075−2.823=−10.898` 对 `−6.581`，都是 −4.317）。
   本目录统一用「减在生产侧」，只为让"真值"一栏永远是原始测量值。
   历史上写的 −4.07 **不是方向错**，只是尾项取了偏小的 2.51。

   ⚠️ **不对齐的那个数会骗人**：+2.823 与真实误差 −4.318 符号相反、相消成
   **−1.494**，把 5.5σ 伪装成 1.9σ。历史文档把 −1.7 和 −4.2 并列成
   「差 1.7~4.2」的区间 —— 那不是区间，是**一个没对齐、一个对齐了**。

3. **⚠️ 盒体积不一致：本目录的真值和生产不在同一密度。**
   本目录的臂（和 `tools/diagnostics/attribute_shell_vs_single_ensemble.py`）固定用
   `4W53/output/box_vectors_solvent.npy` 的**建系盒 43.950 nm³**，且显式拒绝恒压器（NVT）。
   而生产 stage2 用的是**自己 NPT 预平衡末帧冻结的盒**（`output_v3_seed20260908` 是
   42.747 nm³）⟹ **生产密度高 2.81%**，且**建系盒才是偏离 1 bar 平衡的那个**
   （生产 NPT 预平衡在 1 bar 稳定在 ≈42.63±0.30）。

   实测灵敏度（分子质心仿射缩放有限差分，100 帧）：
   `∂ΔA_LJ/∂V = P_coupled − P_decoupled = +0.71 ± 0.11 kJ/mol/nm³`（≈12 bar）
   ⟹ 1.203 nm³ 的差值值 **−0.86 ± 0.13 kJ/mol**，约占 4W53 残差的 **20%**。
   `attribute_stage2_solvent_leg_gap.py` 会自动反解生产盒并告警。
   ⚠️ 用水的压缩率粗估会得到 5~7 kJ/mol，**大 5~8 倍，别用**。

4. **`per_state_Delta_f_kT` 是相对态 0 的累积值，单位 kT（T=300 K）。**
   乘 `kT = 8.314462618e-3 × 300 = 2.49434 kJ/mol` 得 kJ/mol。

### ⚠️ `_attribution_2026_09_02` 那一块里的 `single_mixture` 是错的

该块记 `single_mixture_reweighting_kJ_mol = 1.83`（占 4%）、`total_error = 45.30`。
两个数都带**同一个 LRC 口径错误**（生产含尾项、臂不含，相减前没对齐）：

| 字段 | 记的 | 对齐后 | 说明 |
|---|---|---|---|
| `wca_shield_kJ_mol` | 44.21 | **44.21（干净）** | 两条臂都不含 LRC，相减无问题 |
| `single_mixture_reweighting_kJ_mol` | +1.83 | **−0.99** | `38.724(含LRC) − 36.889(不含)` |
| `total_error_production_minus_truth_kJ_mol` | 45.30 | **42.48** | 同上 |

而且 `single_mixture` 是**带壳**测的。壳退役后同一个量是 **−3.57**（放大 3.6 倍）
⟹ 结论「单系综抓不到空腔重组、只占 4%」**不成立**。逐条见
[../STAGE2_SOLVENT_LEG_ERROR_BUDGET.md](../STAGE2_SOLVENT_LEG_ERROR_BUDGET.md)。

**JSON 本身未改**（它是带 provenance 的实验数据）——更正只记在这里。

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
逐窗口误差归因见 `../archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md` §2.8。

### 为什么放进 repo 而不是留在 scratchpad

原始产物在会话级 scratchpad 里。实测（另一会话 mlpath-6c 2026-09-02 的经验）：
scratchpad **不会**在会话结束时立刻消失，但它在 `/tmp` 底下、没人知道它什么时候没，
**而且它还在的时候你也不会想起来去找** —— 那比数据真丢了更糟：你会在数据其实还在的
情况下写下一段基于间接推断的结论。这份数据是整条误差归因链的唯一参照物，
代价约 80 分钟 GPU，不能挂在一个清理策略上。

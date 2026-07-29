# 原 README 当前状态快照（2026-07-29）

> 这是重写项目首页时从旧 README 原样拆出的状态说明，用于审计信息迁移。
> 它可能被后续运行更新或取代，不能单独作为最终科学结论。当前行动项以
> [`../TODO.md`](../TODO.md)、[`AUDIT_STATUS.md`](AUDIT_STATUS.md) 和
> [`VALIDATION_MATRIX.md`](VALIDATION_MATRIX.md) 为准。

## 当前状态

⚠️ **`output/final_binding_results.json` 里现有的数值是用旧的 `Delta G_bind = Delta
G_complex - Delta G_solvent` 符号约定算出来的，与当前代码（`runabfe.py` 里
`delta_g_bind_uncorrected = dg_solvent - dg_complex`）不一致，需要重新跑一遍才能
得到跟当前公式匹配的号（详见下方“结果解读”和 `status/AUDIT_STATUS.md`）。** 下面这组
数字按当前公式重新算过（ΔG_complex/ΔG_solvent 两个原始值不变，只是改了合并符号），
仅供参考，不代表已经重新采样：

```text
Delta G_complex = 192.8876 kJ/mol
Delta G_solvent = 152.0514 kJ/mol
Boresch correction = -36.5108 kJ/mol
APBS correction = 0.0000 kJ/mol
Delta G_bind = Delta G_solvent - Delta G_complex = -40.8362 kJ/mol = -9.7601 kcal/mol
reported error = 1.3178 kJ/mol
```

（负值表示有利结合：口袋里去耦花的自由能 ΔG_complex 比溶液里去耦花的 ΔG_solvent 更
大，说明配体在口袋里的相互作用更强，这正是它愿意结合的原因；磁盘上残留的正号结果
物理上是反的，不要拿它做任何结论。）

这不是一个可以直接当作最终发表数值的“全修正闭环”结果。当前最重要的物理边界是：

- 默认 ACE/`dual_lambda` 的 VDW/vanishing 腿已启用解析 LJ 长程尾项修正（`traditional_lj_lrc_protocol_version=2`）：对每个 λ_vdw 数值积分 switching-aware、softcore-aware 的真实径向尾项，同时包含吸引 `r^-6` 与排斥 `r^-12`；不启用会把组合表达式中的 Coulomb 尾项一并错误积分、并可能令 CUDA 崩溃的 OpenMM `CustomNonbondedForce` 内建 LRC。`single_lambda`/REMD 的 Beutler 路径使用同一公式作离线修正；协议版本低于 2 的旧输出不得与当前结果混用。
- APBS 修正只作为最终外部项 `Delta G_APBS` 加到 `Delta G_bind`（当前 `apbs_correction_kJ_mol = 0.0`，未启用），不能替代 LJ tail correction。
- 旧输出中的 `thermodynamic_cycle.md` 和 provenance 里仍可能包含历史 PME self-correction 描述；`output/final_binding_results.json` 里 `provenance.thermodynamic_cycle` 目前就是这种未刷新的旧文本快照。请以 `status/AUDIT_STATUS.md` 和当前 diagnostics 为准。当前结论是：手动 `+C*lambda^2` PME 自能“修正”已撤销，不应作为生产修正项使用。
- 当前 Boresch 谐振性校验通过（`diagnostics.boresch.boresch_harmonicity_check.harmonic_assumption_ok = true`），但 6 个力常数里有 3 个（`kr`、`kthetaA`、`kphiA`）被裁剪到保守范围（`force_constant_clipped`），需要在结果解释中保留。
- Stage 2 采用 `Local-TMBAR-Stitched`，误差已传播窗口 offset 方差（复合物腿 `offset_error_contribution ≈ 0.52 kJ/mol`，溶剂腿 `≈ 0.82 kJ/mol`），但尚未包含完整全局 MBAR 协方差、自相关时间和有效样本数修正。当前实现会把低 overlap/ESS 精确定位到失败窗口、相邻状态和 λ；先只续采失败窗口，仍不足时再用已有 λ 状态建立独立、不可变的重叠 rescue ensembles，原 ensemble 与生产 `f_k` 均不原地修改。
- 尚未做独立重复运行：`diagnostics.independent_repeats.performed = false`。

更详细的方法学缺陷、工程审计遗留项和修复状态见 `status/AUDIT_STATUS.md`；文档导航见 `README.md`。

截至 2026-07-22 的实现快照：默认生产主链使用
`IBS_BIAS_PROTOCOL_VERSION=27`（兼容读取 v27/v28 缓存）、`THERMODYNAMIC_PATH_PROTOCOL_VERSION=20`、
`TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=2` 和 `WCA_ACCOUNTING_VERSION=2`。v12 已加入
按状态共享、带协议指纹和 OpenMM checkpoint 的 fixed-H 探针轨迹库。当前 IBS 协议已
修正 pilot-TI/TMBAR 的 `f_k` 符号；预热完成一次完整的固定权重验证后即冻结候选 `f_k`，
验证残差作为效率诊断，不再误当作生产准入的无限重学习条件。生产从独立的第 0 步开始，
不继承预热/验证帧，且 `f_k` 受运行时只读锁保护，生产阶段不得更新。Stage 2 质量门失败时，
默认按 250k→500k→1M 的累计预算只补失败窗口；若仍不足，则创建独立 rescue ensemble。
完整协议、边界与待验证项见 `status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md`。
当前没有剩余的、已确认会阻断默认生产主链的 P0/P1；未修代码行动以 `TODO.md`
为准，代码已修但尚待真实 GPU/完整依赖验证的项目以 `status/VALIDATION_MATRIX.md` 为准。

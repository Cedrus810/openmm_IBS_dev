# 物理/建模缺陷清单

与 `RELEASE_AUDIT_REMAINING.md`（代码工程问题）不同，这份清单只关注**会影响 ΔG_bind 数值本身的物理近似/建模假设**——包括已确认会引入系统偏差的、以及方法本身固有但未做交叉验证的近似。审计范围同前：`abfe_pipeline.py` / `ibs_engine.py` / `abfe_core.py` / `runabfe.py` / `abfe_preoptimizer.py`。

已确认：当前 Atenolol-rank11 体系的配体按**中性**建模（`Atenolol-rank1.itp` 里 [atoms] 电荷求和 ≈ 2×10⁻⁸ e），下面每一条都标注了"是否影响本次实际运行"。

---

## 1. 🔴 VDW（去 VDW/vanishing）腿缺少 LJ 长程色散(tail)修正 —— 未修复，暂缓

**位置**：三处 `CustomNonbondedForce` 都显式关闭了长程修正——`ibs_engine.py:961`（`_create_softcore_force`，配体-环境软核势）、`ibs_engine.py:1502`（`int_f_cv`，IBS CV 探针用的同款软核势）、`abfe_core.py:3525`（`ll_force`，配体-配体内部 LJ+Coulomb）。全仓库搜索 `dispersion`/`tail_correction` 仍然没有任何补偿实现。

**物理影响**：配体-环境的 VDW 相互作用走的是 `CustomNonbondedForce`（ACES 软核势），OpenMM 对 `CustomNonbondedForce` 的长程色散修正默认关闭且这里显式再次关掉。这意味着截断距离（1.2 nm）以外的 LJ 吸引尾部对 ΔG_vdw 的贡献被完全丢弃——这是 FEP/ABFE 里一个已知且通常不可忽略的系统偏差（典型量级 0.1–0.5 kcal/mol，随配体尺寸、极化率和截断距离变化），且方向通常是让"去溶剂化"/"去 VDW"的自由能变得比真实值更不利（缺少长程吸引部分）。

**是否影响本次运行**：**是**，这是主链上必然经过的一步（不区分配体电荷状态），当前代码里没有任何地方计算或事后补偿它。

**当前状态：本轮不修复，明确暂缓。** 之前一版曾把新增的 `apbs_correction.py`（外部 APBS/Poisson-Boltzmann 连续介质静电求解工作流）当作这一条的"当前处理"记录在案——**这是一次概念性张冠李戴，已撤回该说法**：

- APBS 解的是**连续介质静电溶剂化自由能**（Poisson-Boltzmann 方程），本条缺陷是 **LJ 色散尾部修正**（van der Waals，与静电完全无关的物理量）。两者不可互相替代，`apbs_correction.py` 文件头和 `abfe_core.py` 的 `THERMODYNAMIC_CYCLE_DOC` 里其实都已经写明"APBS does not replace a Lennard-Jones dispersion/tail correction"——只是旧版文档把它错误地当成了本条的修复来引用。
- 即使按预期使用 APBS 工作流，运行它对这条缺陷描述的偏差**贡献为零**：代码里三处 `setUseLongRangeCorrection(False)` 一个都没有改动，`ibs_engine.py:1499-1500` 的运行时提示语（"APBS 修正应作为最终外部项记录"）本身也带有同样的概念混淆，暂未一并清理。
- 当前 Atenolol 按中性建模，APBS/PB 连续介质修正在 alchemical FEP 里通常对应的是**带电配体的周期性/有限尺寸静电修正**（这类场景应参照 #2），中性配体场景下这个 workflow 目前也没有明确对应到循环里的哪一项缺口。

`apbs_correction.py` 脚本本身机制上没有 bug（已用合成 PQR/log 跑通 prepare/collect），如果未来确实需要一个静电有限尺寸修正，可以继续用它，但**它不能替代、也不解决本条 LJ tail correction 缺陷**。真正的修复方向是：给软核 VDW 力补一个 LJ 长程色散估计（评估 `CustomNonbondedForce.setUseLongRangeCorrection(True)` 在软核能量表达式下是否可用，或手写一个假设体相均匀密度的解析/数值尾部修正项），并作为循环内部项而非外部程序处理。

**剩余注意**：在此项修复前，正式产出的 ΔG_vdw / ΔG_bind 应视为存在约 0.1–0.5 kcal/mol、方向已知（偏"去溶剂化更不利"）的未修正系统偏差。

---

## 2. ✅ 带电配体的 PME 自能修正被整体禁用，而非按（配体+共炼金反离子）联合电荷平方和正确计算
**位置**：`ibs_engine.py:3927`（`TraditionalMBARAnalyzer`）及其 `compute_u_kn`（`ibs_engine.py:3933` 起，PME 自能修正的具体应用逻辑在 ~3978-4066）。

**现象**：对于中性配体，代码会正确地对总 PME 能量加回 `+C·λ²` 来消除 OpenMM 在做电荷线性 offset 时引入的坐标无关自能伪项（`pme_self_correction_prefactor_kj`，`ibs_engine.py:158-171`）。但一旦检测到配体净电荷 ≠ 0（走 `configure_coalchemical_neutral_decharging` 的共炼金反离子路径），这条修正曾经被**整体关闭**，而不是把反离子的 q² 也计入 `ligand_charge_square_sum` 后按同一公式正确修正。

**物理影响**：共炼金反离子的电荷也是随同一个 `lambda_coul` 线性缩放到 0 的，理论上会产生同一类型、只是系数不同的 `-C_total·λ²` 自能伪项。当前"整体不修"意味着带电配体的去电荷腿会残留一个随 λ 平滑变化、量级可达数十到数百 kJ/mol 的系统偏差（具体大小取决于反离子电荷和 PME α 参数）。

**是否影响本次运行**：**否**（当前 Atenolol 建模为中性，走的是 `apply_pme_self_correction=True` 的正确分支）。但这是代码里一个真实存在、只是**当前恰好没被触发**的缺陷——如果以后换成质子化态（Atenolol 生理 pH 下的仲胺通常带 +1）或换到别的带电配体，会立刻踩中。

**修复状态**：已修。`pme_offset_charge_square_sum()`（`ibs_engine.py:173-196`）从 prepared `NonbondedForce` 的 `lambda_coul` particle offsets 统计 `Σq_offset²`，因此共炼金反离子会和配体原子一起进入 PME 自能修正；诊断写入 MBAR result 的 `diagnostics.pme_self_correction`。

---

## 3. ✅ `GeometricRestraintEstimator` 的力常数估计（k = kBT/var）没有范围裁剪或收敛性校验
**位置**：`abfe_core.py:3527` 起的 `GeometricRestraintEstimator`（`--boresch-source fluctuation`/`simple` 时使用）。

**物理影响**：用涨落方差反推谐振力常数（k=kBT/⟨δx²⟩）是等配分定理的标准做法，但只在两个前提下成立：(a) 该自由度附近的自由能面确实近似谐振、单阱；(b) 采样时间足够长，方差估计已收敛。一旦某个锚点自由度涨落异常小（比如刚好卡在一个局部极小），会给出一个远超合理范围的力常数，直接喂进 `calculate_boresch_analytical_correction` 的硬校验——运行时报错还算好的情况；如果恰好卡在合法范围内但物理上不合理，则会悄悄给出一个错误的标准态释放修正而不报错。

**是否影响本次运行**：本次 `abfe_config.json` 里 `boresch_source = "simple"`（几何涨落估算器）会用到这条修复。

**修复状态**：已修。`GeometricRestraintEstimator` 会保存 raw force constants、clip ranges、clip flags、每个 restraint coordinate 的 skew/kurtosis/percentile 诊断，并在偏离高斯或采样不足时写 warning。`runabfe.py` 的 Boresch 参数清洗不再丢弃这些诊断。

---

## 4. ✅ Boresch 解析释放修正本身的谐振近似 —— 本轮修复：诊断结果已真正接入决策/报警路径

**位置**：解析公式本身在 `abfe_core.py::calculate_boresch_analytical_correction`；新增的模型无关校验在 `abfe_core.py::assess_boresch_harmonicity`（紧跟在 `calc_boresch_from_last_frame` 之后）；接入点在 `runabfe.py::resolve_boresch_restraint`（生成/更新 Boresch 参数之后）以及 `final_binding_results.json` 的写出逻辑。

**物理影响**：该解析公式（标准 Boresch 2003 形式）假设 6 个约束自由度各自独立、且围绕平衡值做小幅高斯涨落。如果预平衡轨迹里锚点距离/角度/二面角的实际涨落明显非谐（比如受体侧链在两个转子异构态之间跳），解析修正会系统性偏离真实的标准态释放自由能。

**原始缺陷**：代码里曾经实现过一个检测非谐性的工具（`OrbScanner.scan_boresch_1d_pes`，会标记 `anharmonic_flag`），但它需要 ML 势（`HAS_ORB`/MACE-OFF，受限许可证）、只实现了 `r` 距离一个坐标的扫描（θ/φ 会直接 `NotImplementedError`），而且**全仓库从未被任何流程调用过**——即使探测到非谐信号，流程也不会自动切换到数值积分或警示用户。上一版文档曾把"最终 JSON 里多记录了一句谐振假设的静态文字说明"当作"已修"，但那句话是恒定文本，不随实际轨迹变化，且只有 `--boresch-source fluctuation` 才带涨落诊断——本次实际用的 `boresch_source = "simple"`（ORB 估算路径）当时完全没有这项校验，等于原始缺陷原封不动。

**本次修复**：新增 `assess_boresch_harmonicity(traj, receptor_indices, ligand_indices)`（`abfe_core.py`），直接在锁定的 6 个锚点上、用 mdtraj 重新计算 r/θA/θB/φA/φB/φC 时间序列，复用 `GeometricRestraintEstimator._fluctuation_diagnostics` 的偏度/峰度/欠采样判据——**不依赖 ML 势，因此对 `auto`/`orb_simple`/`simple`/`fluctuation` 全部四种 Boresch 来源统一生效**，不再局限于 fluctuation 一种。`runabfe.py::resolve_boresch_restraint` 在锚点确定、平衡值用最后一帧更新之后立即调用该函数：
- 结果写入 `boresch["diagnostics"]["boresch_harmonicity_check"]`（含每坐标诊断、`harmonic_assumption_ok` 布尔值、非空 `warning` 文本），并随 `_sanitize_boresch_params_strict` 一并透传进 `boresch_<source>.json` 缓存文件。
- 若判定非谐/欠采样，会调用 `log.warning` 实时报警，并把警告追加进 `boresch["diagnostics"]["warnings"]`。
- `final_binding_results.json` 里新增了真实计算出的 `analytical_release_assumption_checked`（是否执行过校验）和 `analytical_release_reliable`（校验结果，`None`/`True`/`False`）字段，取代原来那句不随数据变化的固定文案。

**已知局限（如实记录，非隐藏）**：
- 该校验只在 Boresch 参数由内部估算器（`auto`/`orb_simple`/`simple`/`fluctuation`）从 `pre_equilibration.dcd` 现场生成时运行；`traditional`/`orb_ml`（直接读取外部锚点 JSON 文件）分支在 `resolve_boresch_restraint` 里提前 `return`，没有绑定的轨迹可用，**不会**自动获得这项诊断。
- 这是一个基于涨落分布的统计判据（偏度/峰度/欠采样），跟 `scan_boresch_1d_pes` 沿真实 PES 做能量扫描的判据不是同一种方法——但它不需要 ML 势、覆盖全部内部估算路径，且被证实真正接入了报警链路，弥补了原始缺陷"诊断结果不影响任何下游决策"的核心问题。`scan_boresch_1d_pes` 仍保留在代码里，可作为需要更强证据时的可选深度校验，但不是自动流程的一部分。

**是否影响本次运行**：**是**——`boresch_source = "simple"` 现在会在每次生成/续跑 Boresch 参数时自动做这项校验，请在 `output/boresch_simple.json` 与 `final_binding_results.json` 的 `diagnostics.boresch.analytical_release_reliable` 里确认结果。

---

## 5. ✅ 软核/防护壳超参数是固定经验值，未针对具体配体重新校准
**位置**：`ibs_engine.py:973`（`_normalize_softcore_params`，此前硬编码 `alpha_lj=0.7, alpha_coul=0.5`，覆盖了 `ACESoftcorePotential.optimize_alpha()` 按配体原子数算出的自适应值）；`ibs_engine.py:1170`（`_estimate_wca_shield_parameters`，此前 λ-WCA 防护壳硬编码 `rc=0.22 nm`, `eps_wca=1.0`，与配体大小/极化率无关）。

**物理影响**：ACES 论文的方法学卖点之一就是根据配体性质动态选取软核衰减参数，这里曾被"生产模式统一锁定"覆盖掉。这不代表这组经验值是错的（0.7/0.5 nm⁶/nm² 是文献常见量级），但意味着"这组参数对当前阿替洛尔+受体体系确实合适"这件事没有被显式验证过，纯粹是沿用了一组通用默认值。

**修复状态**：已修。`ibs_engine.py` 不再静默覆盖为固定 `alpha_lj=0.7, alpha_coul=0.5`；默认按配体扰动原子数使用 `ACESoftcorePotential.optimize_alpha()`，若用户显式提供 softcore 参数则尊重用户值。λ-WCA 防护壳也改为基于配体-环境 LJ sigma 的有界估计，并在日志中记录 `rc/eps/source`。

---

## 6. ✅ 2D 测地线 λ 路径规划的统计噪声（影响效率，不直接影响最终 ΔG）
**位置**：`abfe_preoptimizer.py::optimize_2d_geodesic_path` / `compute_2d_metric_grid`（默认 `n_grid=16`, `n_steps_per_point=3000`，用有限差分 `delta=0.02` 估计度量张量）。

**物理影响**：3000 步（约 6 ps，2 fs 步长下）内对 dU/dλ 做协方差估计，噪声较大；16×16=256 个网格点意味着这一步的总采样量并不小，但每个点仍然很短。这一步只是用来**规划**后续生产采样该在哪些 λ 密集取点，本身不进入最终自由能计算（真正的 ΔG 来自后续 IBS/TMBAR 生产阶段的完整采样），所以噪声不会直接引入系统偏差，但如果度量张量场因为噪声而"看错"了高方差区域，会导致 λ 点分布不合理，间接拖累后续窗口的重叠率、收敛速度和误差棒可信度。

**修复状态**：已修。`compute_2d_metric_grid()` 可返回采样诊断，`optimize_2d_geodesic_path()` 会打印有效网格比例、失败/unsafe 点数、每点导数样本数，并在样本过少时明确提示该路径只应作为效率启发式，需结合 overlap diagnostics 判断。

---

## 总结（按对 ΔG_bind 数值可信度的影响排序）

| # | 问题 | 当前状态 | 备注 |
|---|---|---|---|
| 1 | VDW 腿缺 LJ 长程色散修正 | 🔴 未修复，本轮明确暂缓 | `apbs_correction.py` 是静电连续介质修正，不能替代本条；核心 `setUseLongRangeCorrection(False)` 三处均未改动 |
| 2 | 带电配体 PME 自能修正 | ✅ 已修 | offset `Σq²` 包含共炼金反离子；当前中性配体未触发 |
| 3 | fluctuation 法力常数无 clip/收敛校验 | ✅ 已修 | raw/clip/分布诊断会保留；本次 `boresch_source=simple` 会用到 |
| 4 | Boresch 谐振近似未接入决策/报警 | ✅ 本轮已修 | 新增 `assess_boresch_harmonicity`，四种内部估算来源统一校验并真实报警；外部锚点文件来源（traditional/orb_ml）暂不覆盖 |
| 5 | 软核/WCA 超参数固定 | ✅ 已修 | 默认自适应，显式参数可覆盖 |
| 6 | 2D 测地线路径规划采样噪声 | ✅ 已修 | 打印有效网格和样本数诊断 |

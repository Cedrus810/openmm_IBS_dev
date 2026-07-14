# ABFE 审计状态总表（2026-07-10）

本文件合并并取代以下三份旧审计文档的当前有效内容：

- `RELEASE_AUDIT_REMAINING.md`
- `PHYSICS_DEFECTS.md`
- `RE_AUDIT_2026-07-10.md`

审计范围：`abfe_core.py` / `abfe_pipeline.py` / `ibs_engine.py` / `runabfe.py` / `abfe_preoptimizer.py` / `apbs_correction.py`。`dexp_experiment.py` 属于独立实验模块，不作为当前主链审计对象。

运行上下文：中性配体 Atenolol（总电荷约 0）、`decoupling=dual_lambda`、`potential=softcore`、`boresch_source=simple`、IBS/TMBAR、CUDA、production 预设、T=300 K。当前结果中 APBS correction 默认为 0，未作为主链采样的一部分启用。

---

## 当前结论

在当前主路线下，热力学循环符号和核心物理组装已经核实：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

其中 `Delta G_complex` 已包含复合物腿 decoupling、约束/Jacobian 项和 Boresch 标准态释放项；`Delta G_solvent` 使用溶剂腿 total 结果；`Delta G_APBS` 只有显式传入 `--apbs-correction-kj-mol` 时才加入。

没有发现新的“当前主链会静默把 Delta G_bind 符号算反或重复加减 Boresch”的 bug。剩余最高风险集中在三类：

1. ~~误差棒和收敛判据可信度不足。~~ 已修复（见下）。
2. VDW/vanishing 腿缺少 LJ long-range dispersion/tail correction。
3. 若启用当前不常用路径，存在若干潜在静默错误或旧文件误读风险。

---

## 最高优先级：影响当前结果可信度

### 1. ✅ 已修复 — 缺少自相关子采样，误差棒系统性偏小

位置：`ibs_engine.py` 中 `TraditionalMBARAnalyzer.solve` 和 `GlobalMBARAnalyzer.solve_stage_integrated`；共享 helper `abfe_core.py::subsample_series_by_autocorrelation`。

原问题：之前把每一帧原始 MD 轨迹当作独立样本喂给 MBAR。`n_k` / `N_eff` 会把相关帧当作不相关帧计数，导致报告的 `error` / `ddf` 偏小。已有 statistical inefficiency 计算只作为诊断，不真正用于下采样，而且计算对象也不是每个状态的时间序列。

修复内容：

- 新增 `subsample_series_by_autocorrelation(series)`（`abfe_core.py`）：用 `pymbar.timeseries.statistical_inefficiency` 估计单一状态能量序列的统计非效率 `g`，再用 `subsample_correlated_data` 取出近似独立的帧索引；样本过少（<20）或 pymbar 不可用时原样返回全部帧（`g=1.0`），不强行子采样。
- `TraditionalMBARAnalyzer.solve()`：按 `n_k` 分块，对每个态自己的能量时间序列做去相关子采样后再建 MBAR（新增 `decorrelate: bool = True` 参数，默认开启）；原来对"跨状态 pooled 列做差"的无意义 `statistical_inefficiency` 诊断已删除，替换为每态真实的 `g` 值（`diagnostics.decorrelation`）。
- `GlobalMBARAnalyzer.solve_stage_integrated()`：对每个窗口的采样分布能量序列（`base+bias`）做同样的去相关子采样，再建局部 MBAR。
- 已用合成 AR(1) 相关序列验证：子采样后误差棒变大（更保守、更真实），帧数按 `g` 比例下降；iid 序列 `g≈1`（几乎不子采样）。

状态：已修复。

### 2. ✅ 已修复 — `GlobalMBARAnalyzer` 的收敛/重叠指标不是真正的 MBAR overlap

位置：`ibs_engine.py` 中 `GlobalMBARAnalyzer.solve_stage_integrated`、`TraditionalMBARAnalyzer.solve`；门控点在 `abfe_pipeline.py::ABFEPipeline._assert_stage_result_sane`。

原问题：

- `converged = len(local_results) == len(valid_windows)` 只表示每个窗口都解出了结果，不代表统计收敛。
- `min_overlap_proxy = 1/(1+max|Delta f|)` 是自由能间距的单调函数，不是 MBAR overlap matrix。
- `abfe_pipeline.py` 当前没有真正读取并门控 `converged`。

修复内容：

- `GlobalMBARAnalyzer.solve_stage_integrated`（单一采样分布 + 多个零样本目标态的场景，标准 overlap matrix 会退化）：改用 `mbar.compute_effective_sample_number()` 算出每个目标 λ 态的重加权有效样本比例（ESS ratio = neff / n_sampled），窗口 `min_overlap` = 该窗口最差的 ESS ratio；已用合成数据验证该指标随真实重叠单调变化（近距离目标 λ → ESS ratio≈0.96；极远目标 λ → ESS ratio≈0.006）。`converged` 现在要求所有窗口都解出 **且** 全局最小 ESS ratio ≥ 阈值 0.05。旧的 Δf 间距量保留在新字段 `lambda_spacing_max_step_kJ_mol`，不再冒充"重叠度"。
- `TraditionalMBARAnalyzer.solve`（REMD 场景，所有态都真实有样本，标准 overlap matrix 有效）：改用 `mbar.compute_overlap()["matrix"]` 的相邻态（`|i-j|=1`）最小值作为 `min_overlap`，阈值 0.03（与 `abfe_core.py` 在线监控已有的相邻窗口重叠阈值保持一致）。
- `abfe_pipeline.py::_assert_stage_result_sane` 新增硬性检查：结果带 `converged`/`min_overlap` 字段时，`converged is False` 或 `min_overlap < min_overlap_threshold` 直接 `raise RuntimeError`，拒绝把重叠不足的阶段标记为 completed；不带这些字段的旧路径结果不受影响（向后兼容）。已用单测验证三种情况（正常通过/低重叠正确拒绝/无字段的旧结果正常通过）。

状态：已修复。

### 3. VDW/vanishing 腿缺少 LJ 长程色散 tail 修正

位置：三处 `CustomNonbondedForce.setUseLongRangeCorrection(False)`：

- `ibs_engine.py` 的配体-环境软核势。
- `ibs_engine.py` 的 IBS CV 探针软核势。
- `abfe_core.py` 的配体内部 LJ+Coulomb 自定义力。

问题：softcore VDW 交互组力不自动包含原始 `NonbondedForce` 的 LJ dispersion correction。截断外 LJ 吸引尾部对 `Delta G_vdw` 的贡献被丢弃，复合物盒子和纯水盒子环境不同，不能假设在 `Delta G_solvent - Delta G_complex` 中完全抵消。

影响：当前主链必经，影响 Delta G 数值。典型量级约 0.1-0.5 kcal/mol，随配体尺寸、极化率、cutoff 和环境密度变化。

状态：未修复，明确暂缓。

重要边界：`apbs_correction.py` 是连续介质静电/极化 correction helper，不能替代 LJ tail correction。APBS v2 的改写不改变本条状态。

### 4. VDW 窗口拼接 offset 权重不一致

位置：`ibs_engine.py` 中 VDW stage integrated/global MBAR 拼接逻辑（`ibs_engine.py:3604` 附近）。

问题（历史）：窗口间 offset 曾使用重叠点上的非加权平均，但每个 lambda 的合并使用逆方差加权。当重叠点不确定度差异较大时，offset 会相对最终合并曲线偏移，影响 `f_curve[-1] - f_curve[0]`。

状态：已修复。offset 计算已改为对重叠 lambda 的逆方差加权平均（`inv_var = 1 / max(offset_vars, 1e-12)`），并显式传播 `offset_var` 累加进每个窗口的 `var_loc`，与下方逐 lambda 合并使用同一套逆方差加权逻辑一致。

---

## 物理/建模状态

### PME self correction：`+C*lambda^2` 已撤销，仅保留诊断

位置：`ibs_engine.py` 的 PME decharging offline `u_kn` 分支和 `pme_self_correction_*` helper。

结论：此前“需要手动加回 `+C*lambda^2`”的判断是错误的。OpenMM 的 `NonbondedForce.addParticleParameterOffset` 会在每个 lambda 状态下用已缩放电荷重新计算完整 PME 能量，包含 Ewald self-energy。该 self-energy 是该 lambda 态哈密顿量的一部分，不是缺失伪项。手动加回 `+C*lambda^2` 会反向抵消真实存在的能量项。

当前状态：

- `apply_pme_self_correction` 在生产路径中保持 `False`。
- `pme_offset_charge_square_sum()` 仅用于诊断记录 `charge_square_sum_e2`。
- 旧输出若包含历史 PME self-correction 文字或旧数值，必须以当前代码和本文件为准。

仍需：用当前代码做一次干净重跑，确认 decharging 腿数值和旧缓存完全脱钩。

### Boresch 力常数估计与谐振假设

已修复内容：

- `GeometricRestraintEstimator` 会保留 raw force constants、clip ranges、clip flags、分布诊断和 warnings。
- `assess_boresch_harmonicity` 已接入内部 Boresch 来源（`auto` / `orb_simple` / `simple` / `fluctuation`），对 r / thetaA / thetaB / phiA / phiB / phiC 做分布诊断并写入 provenance。
- `final_binding_results.json` 使用真实计算出的 `analytical_release_assumption_checked` 和 `analytical_release_reliable`，不再只写静态说明。

局限：

- 外部锚点文件来源（`traditional` / `orb_ml`）没有绑定预平衡轨迹，不自动获得这项诊断。
- 当前诊断是涨落分布统计判据，不等价于沿 PES 的数值积分或能量扫描。

状态：主链已修，仍建议检查输出 JSON 中的 Boresch diagnostics。

### Softcore / WCA 参数

已修复内容：

- `_normalize_softcore_params` 不再把 softcore alpha 硬覆盖为固定 `alpha_lj=0.7` / `alpha_coul=0.5`。
- 默认使用 `ACESoftcorePotential.optimize_alpha()` 的自适应值。
- 显式传入 softcore 参数时尊重用户值。
- WCA shield 参数改为基于配体-环境 LJ sigma 的有界估计，并记录来源。

状态：已修复。

### 2D geodesic lambda 路径规划噪声

问题本质：短采样估计度量张量会有噪声，影响 lambda 点分布效率，但不直接进入最终 Delta G。

已修复内容：

- `compute_2d_metric_grid()` 可返回采样诊断。
- `optimize_2d_geodesic_path()` 打印有效网格比例、失败/unsafe 点数、每点导数样本数。

状态：已修复为可审计的效率启发式；最终仍需结合 overlap diagnostics 判断。

---

## APBS correction helper 当前状态

`apbs_correction.py` 已升级为 v2 工作流。

新定义：

```text
G_component = E_environment - E_reference
Delta G_APBS = G_complex - G_receptor - G_ligand
```

关键变化：

- 每个组分生成同网格 `environment` 和 `reference` 两个 APBS `elec` block。
- 默认使用 complex common grid。
- 支持 `--diel-map-x/y/z` 与 `--kappa-map` 描述水-膜-蛋白体系中的膜介电与离子排除。
- 默认只把膜 maps 用于 `complex` 和 `receptor`；孤立 `ligand` 保持 bulk water reference。可用 `--map-targets` 覆盖。
- manifest 记录 map hash、map targets、reference dielectric、warnings、每个组分是否使用 maps。
- `collect` 兼容旧 v1 manifest。

已处理的旧审计问题：

- 旧版“单次 `mg-auto` total energy 不是溶剂化/参考差分”的问题已由 `environment-reference` 处理。
- 旧版 `--common-grid` 默认关闭的问题已改为默认 common grid。
- 旧版多能量行 `matches[-1]` 风险已通过 v2 的 environment/reference 解析路径降低；同时保留旧 manifest 兼容路径。

仍然重要：

- APBS 是静电/连续介质外部项，不是 LJ tail correction。
- 当前主链默认 APBS correction 为 0；只有显式使用 `--apbs-correction-kj-mol` 才进入最终 Delta G。
- 对中性 Atenolol，APBS correction 必须有单独定义清楚的热力学含义，不能默认认为它修补了主链物理缺口。

---

## 当前不触发但启用相关路径前应修

### 1. ✅ 已修复 — decharging 分支边界情形可能掉到截断 Coulomb

位置：`ibs_engine.py::compute_u_kn`（`REMDManager._build_replicas` 的同名判定逻辑暂未改动，风险敞口不变但当前主链不走该类）。

原问题：`is_pme_coulomb_leg` 仅靠 `np.allclose(lambdas_vdw_arr, 1.0)` 静默判定 PME Coulomb leg，未来若误传非 decharging 的 λ 表会静默走错分支、给出错误结果，且没有硬性报错保护。

修复内容：在 `compute_u_kn` 进入 `is_pme_coulomb_leg` 分支前新增显式断言——若 `lambdas_vdw_arr` 不严格满足 `np.allclose(..., 1.0, atol=1e-6)`，直接 `raise RuntimeError` 并说明可能的误配置原因，而不是继续静默执行。

状态：已修复。`REMDManager.__init__` 里同名的 `is_pme_coulomb_leg` 判定（在线 REMD 路径）仍是纯 `np.allclose` 推断，未加同等断言，留作后续可选加固项。

### 2. ✅ 已修复 — `run_full_abfe_loop` 是 unused 但组装不一致

位置：`abfe_pipeline.py::run_full_abfe_loop`。

原问题：复合物侧正确取 `total_delta_G_complex_kJ_mol`（取负号），但溶剂侧优先取旧口径 `decoupling_delta_G_kJ_mol`，与 `total_delta_G_complex_kJ_mol`（已含约束/Boresch 修正）不一致；主链当前不调用它，但一旦接线会给出偏差结果。

修复内容：调整键优先级为先取 `total_delta_G_complex_kJ_mol`，再回退 `decoupling_delta_G_kJ_mol`/`total_delta_G`，与 `runabfe.py` 主流程及复合物侧口径保持一致。

状态：已修复（函数本身仍未被主链调用，属于孤立但已自洽的工具函数）。

### 3. ✅ 已修复 — parallel stages worker 崩溃可能读到旧 JSON

位置：`abfe_pipeline.py` 的 `parallel_results/` 写入/读取逻辑。

原问题：`parallel_results/` 目录用 `exist_ok=True` 创建且不清空；worker 子进程崩溃/被杀而未写出新 `stage1.json`/`stage2.json` 时，父进程会静默读到上一次运行遗留的结果，误判为本轮成功。

修复内容：在派生 worker 前，若 `stage1.json`/`stage2.json` 已存在则先删除，确保 worker 未正常写出时后续 `open()` 直接抛 `FileNotFoundError`，而不是读到陈旧数据。

状态：已修复（未额外引入 run fingerprint / 时间戳校验，删除已足以消除"静默读旧结果"的风险；如需更强审计可再加 run id）。

### 4. ✅ 已修复 — 角度力常数 clip 范围和解析函数接受范围不一致

位置：`GeometricRestraintEstimator`（`abfe_core.py`，clip 上界 1000）与 `calculate_boresch_analytical_correction`（原上界 500）。

原问题：估计器可把角度力常数 clip 到 1000 kJ/mol/rad²，但解析修正函数对 > 500 会硬报错，导致估计器给出的合法值（500~1000 区间）在下游直接调用时崩溃。

修复内容：`calculate_boresch_analytical_correction` 的角度力常数校验上界从 500 统一改为 1000，与 `GeometricRestraintEstimator` 的 clip 范围完全一致。

状态：已修复。

### 5. ✅ 已修复 — ORB 相关类缺少 `HAS_ORB` 前置检查

位置：`OrbBoreschEstimator.__init__`（`abfe_core.py`）；`Orbv3DEXPFittingPipeline.__init__` 已有该检查，未受影响。

原问题：未安装 `torch` / `openmmml` 时，`OrbBoreschEstimator.__init__` 直接引用 `torch.cuda.is_available()`，会抛出裸 `NameError` 而非清楚的依赖缺失说明。

修复内容：构造函数开头新增 `if not HAS_ORB: raise ImportError(...)`，与 `Orbv3DEXPFittingPipeline` 的既有模式一致。

状态：已修复。

### 6. ✅ 已修复 — DEXP 拟合逐帧异常被归入 outlier 但不记录原因

位置：`Orbv3DEXPFittingPipeline` 逐帧标注循环（`abfe_core.py`）。

原问题：`except Exception as e` 只累计 `stats["skip_outlier"]` 并 `continue`，异常内容完全丢失，真实代码 bug（而非物理离群帧）可能被长期伪装成"outlier"。

修复内容：新增 `stats["skip_outlier_reasons"]` 列表记录每次异常的帧号 + 类型 + 消息；前 5 条异常立即打印，超出部分在采样诊断汇总时报告剩余条数，不再静默吞掉。

状态：已修复。

---

## 轻微报告/下游误用风险

### 1. ✅ 已修复 — `--analyze-only` 组装口径可能与正式输出不一致

问题：dual_lambda `--analyze-only` 路径此前恒用 `_analyze_dual_leg` 从原始窗口能量文件重新估算 `decoupling_delta_G_kJ_mol`，不含正式 pipeline 烘焙的 PME 自能/约束修正等项，也从未应用 `--apbs-correction-kj-mol`（即便该 CLI flag 本身存在）。

修复内容：

- 若复合物腿与溶剂腿的 `final_results.json` 均存在，优先复用其中权威的 `total_delta_G_complex_kJ_mol`（与主流程/正式组装口径一致），并相应把 `dg_boresch_term` 清零以避免对已烘焙的 Boresch 修正二次扣减；只有在缺少正式结果文件时才回退到 `_analyze_dual_leg` 重新估算，并在日志中明确提示这是粗略核查值。
- 补上此前完全缺失的 `--apbs-correction-kj-mol` / `--apbs-correction-note` 应用，输出中新增 `delta_G_bind_uncorrected_kJ_mol` / `apbs_correction_kJ_mol` 字段，与主流程组装口径对齐。

状态：已修复。仍非与主流程完全共享同一份组装函数（存在少量重复逻辑），但字段口径和修正项已对齐。

### 2. ✅ 已修复 — 最终 JSON 中 `boresch_correction_kJ_mol` 容易被下游二重减法

问题：`Delta G_complex`/`total_delta_G_complex_kJ_mol` 在部分路径下已经把 Boresch release 烘焙在内，但 JSON 里仍只并列写出 `boresch_correction_kJ_mol`，未标注它是否已被计入，容易让下游脚本误以为是独立可加项而二次扣减。

修复内容：在 `abfe_pipeline.py`（`run_full_pipeline` 的 `final_results.json`、`TraditionalABFEPipeline.run_full` 的 `final_results.json`）和 `runabfe.py`（主流程 `final_binding_results.json`、traditional 模式 `final_binding_results_traditional.json`、`--analyze-only` 的 `final_results_postprocess.json`，共 5 处写出点）新增显式布尔字段 `boresch_correction_already_included_in_complex_delta_G`（或 `..._in_total_delta_G`），并按各路径实际组装逻辑正确标注 `true`/`false`，附加说明性 note。修复过程中还发现并同步修正了一个由此暴露的潜在双重扣减风险：`--analyze-only` 复用权威 `final_results.json` 时必须把独立计算的 `dg_boresch_term` 清零（见上一条），否则会把同一个 Boresch 修正减两次。

状态：已修复。

---

## 已验证正确 / 已修复

核心物理与数值：

- 热力学循环符号：`Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS`。
- Boresch 解析修正公式与单位：Boresch 2003 形式、`V0=1.6605 nm^3`、`sin(theta)` 奇点保护、角/二面角力常数单位自洽。
- PME self correction `+C*lambda^2` 已停用，仅保留 inert diagnostics。
- `assess_boresch_harmonicity` 的 6 个坐标定义与 `LambdaDependentBoreschForce` / estimator 一致，二面角诊断前 unwrap。
- 采样哈密顿量和评估哈密顿量在当前主链中匹配：VDW 腿 IBS/ACE softcore/GlobalMBAR，Coul 腿 REMD/PME/TraditionalMBAR。
- `collect_energies` 中 `base_energies` 失败归 0 不会偏置 MBAR Delta G，因为它作为每列常数相消，只影响数值 conditioning 和日志可解释性。

工程修复已验证：

- `IBSSampler.collect_energies()` 不再裸吞异常导致 `e_base` 静默归零。
- PBC 分子完整性修复已重新启用。
- Boresch 平衡值最后一帧刷新失败时记录 `is_fallback` / `equilibrium_update_error`。
- `debug_mode` 默认改为 `False`，生产采样诊断打印已门控。
- resume 会逐窗口检查能量文件形状并跳过已完成窗口。
- 溶剂腿已传递 `n_workers` / `parallel_stages`。
- softcore alpha 默认自适应，WCA 参数基于 sigma。
- NaN/Inf 检测无条件执行。
- torsion exclusions 覆盖 CustomTorsionForce / RBTorsionForce。
- checkpoint 缺文件时抛错而非静默返回 0。
- 局部重复 `NumpyEncoder` 已删，统一从 `abfe_core` 导入。
- 预优化器端点和插值后 NaN 检测已补。
- `apbs_correction.py` v2 已用合成 PQR/log smoke test 通过 `prepare` / `collect`。
- `compute_u_kn` 的 PME decharging 分支新增 `lambda_vdw == 1.0` 硬性断言。
- `run_full_abfe_loop` 溶剂侧键优先级已改为 `total_delta_G_complex_kJ_mol` 优先。
- `parallel_results/` 的 `stage1.json`/`stage2.json` 在派生 worker 前会先清空，避免读到旧运行结果。
- `calculate_boresch_analytical_correction` 角度力常数上界统一为 1000 kJ/mol/rad²，与 `GeometricRestraintEstimator` clip 范围一致。
- `OrbBoreschEstimator.__init__` 新增 `HAS_ORB` 前置检查。
- DEXP 逐帧异常改为记录原因（帧号+类型+消息）而非静默归入 outlier。
- `--analyze-only`（dual_lambda）优先复用正式 `final_results.json` 的 `total_delta_G_complex_kJ_mol`，并补上此前完全缺失的 APBS 修正应用。
- 5 处最终 JSON 写出点新增 `boresch_correction_already_included_in_*` 显式标记，避免下游对 Boresch 修正二次扣减。

---

## 建议修复优先级

1. ~~P0：实现自相关子采样，修正误差棒。~~ 已修复（见「最高优先级」第 1 条）。
2. ~~P0：使用真实 overlap matrix，替换假的 `converged` / `min_overlap_proxy`，并让 pipeline 门控。~~ 已修复（见「最高优先级」第 2 条）。
3. ~~P1：修正 VDW 窗口拼接 offset 权重或改为真正 global MBAR。~~ 已修复（offset 改为逆方差加权并传播 offset 方差，见「物理/建模状态」第 4 条）。
4. ~~P1：给 decharging 分支加 `lambda_vdw == 1.0` 断言。~~ 已修复（`compute_u_kn`；断言原用 `np.allclose(..., atol=1e-6)` 未设 `rtol=0`，`lambda_vdw=0.99999` 仍会因默认 `rtol=1e-5` 而通过，现已改为 `rtol=0.0, atol=1e-6` 使其真正严格；`REMDManager.__init__` 同名判定仍待加固）。
5. P1：处理 LJ tail/LRC correction，或明确作为外部非 APBS 项处理。
6. ~~P2：修正 unused `run_full_abfe_loop` 或删除。~~ 已修复（键优先级已与主流程对齐）。
7. ~~P2：修复 parallel stages 旧 JSON 误读风险。~~ 已修复（写入前清空 stage1.json/stage2.json）。
8. ~~P2：统一角度力常数上界。~~ 已修复（统一为 1000 kJ/mol/rad²）。
9. ~~P3：澄清 final JSON 的 Boresch 字段，修正 analyze-only 口径。~~ 已修复（新增 `boresch_correction_already_included_in_*` 标记；analyze-only 优先复用正式 `final_results.json` 并补上 APBS 修正）。
10. ~~P3：补 ORB 依赖前置检查和 DEXP 异常记录。~~ 已修复（`OrbBoreschEstimator` 加 `HAS_ORB` 检查；DEXP 逐帧异常记录原因）。

---

## 可选清理项

| 位置 | 问题 |
|---|---|
| `abfe_core.py` | 局部变量 `top` 与 mdtraj topology 别名撞名，当前无害 |
| `abfe_core.py` | `scan_boresch_1d_pes` 中保留不可达分支 |
| `abfe_core.py` | 章节编号注释重复 |
| `ibs_engine.py` | 文件中部重复 import |
| `abfe_pipeline.py` | `generate_overlapping_windows` 重复导入 |
| `abfe_pipeline.py` | `_setup_boresch_params` 是未调用桩函数 |
| `runabfe.py` | 若干 import 未使用 |
| `runabfe.py` | GROMACS 路径硬编码兜底不便携，但已排在显式参数和环境变量之后 |

---

## 仍需动态验证

- CPU 极小步数端到端 smoke test：系统构建、Boresch、复合物腿、溶剂腿、最终 assembly。
- 目标 GPU 上完整 ABFE 实跑：确认修正后的 overlap、误差棒、Delta G 数值。
- `--parallel-stages` + `--resume` 在真实多 GPU 环境下验证。
- 若启用 APBS：用真实膜 dielectric/kappa maps 和实际 APBS binary 跑通 v2 输入，并把 correction 的热力学含义写进 provenance。

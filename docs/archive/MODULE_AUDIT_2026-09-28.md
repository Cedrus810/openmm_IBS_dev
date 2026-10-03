# 模块化审计（2026-09-28）：correctness / 数值稳定性 / 性能 / API / 测试缺口

> **一次性审核的原始记录，不是待办清单。** 与 [CODE_REVIEW_2026-09-26.md](CODE_REVIEW_2026-09-26.md)、
> [PERF_REVIEW_2026-09-28.md](PERF_REVIEW_2026-09-28.md) 互补：那两轮按"发现"组织，本轮按"模块 × 五维"组织，
> 并补上此前从未审过的两个维度（**API/config 接线**、**测试缺口**）。
> 已登记条目（9-26 CR/PERF/QUALITY、9-28 A/B/C 组、TODO §2.6/§3.7）一律**不重复报**，只在矩阵里标注归属。
>
> ⚠️ **范围缩减**：原计划 14 个模块全量并行代理审计，因用户并发额度限制中途改为**主对话直接审核心四模块**
> （用户指令「你就审理核心吧」）。外围模块只做了机械测试覆盖映射，**没有做五维审计**——补审方式见 §5。
> 本轮所有断言都有主对话实读的行号证据；数值抽查函数均为逐行读，非 grep 猜测。HEAD `e576a54`，runabfe.py 为工作树现状。

---

## 0. 模块 × 五维矩阵

| 模块 | correctness | 数值稳定性 | 性能 | API | 测试缺口 |
|---|---|---|---|---|---|
| ibs_engine.py (28k) | 9-26/9-28 已覆盖；本轮抽查未见新增默认路径 bug | **抽查 8 处全部干净**（§2.1） | 9-28 已覆盖（A2/A4/A5/A6/A11–A14/B1、PERF-01–05） | 见 §2.1 干净结论 | **新发现密集**（§3，TG-1/3/4） |
| abfe_pipeline.py (20k) | CR-1/CR-2/CR-4/A1 已登记；本轮未深审该区 | 指纹量化/磁盘门/终止处理抽查干净（§2.2） | A1/A7/A9/B2/B4 已登记 | config 键经 kwargs.get 全部接线（§1） | 偏少（§3 TG-5 附近） |
| abfe_core.py (14k) | A3/A10/B5/QUALITY-02/06 已登记；循环闭合抽查干净 | **循环闭合/周期统计抽查干净**（§2.3） | A10/B5 已登记 | charge_at_lambda 契约与测试脱节（API-1） | **修正项数学无直接测试**（TG-1/2） |
| runabfe.py (9k) | CR-2/CR-3 部分、BORESCH-AUTO-01 已登记 | seed 解析防截断干净（§2.4） | A3/A8/B3 已登记 | **生产 preset 33 键无一死键**（§1，本轮正面结论） | TG-5 |
| abfe_preoptimizer.py (14k) | 未审（外围） | 未审 | QUALITY-04 部分已登记 | 未审 | **17 个公共函数无直接测试**（TG-6） |
| outer_lambda_neural_basis.py (8k) | 未审（外围，实验线） | 未审 | 未审 | 未审 | 大量（实验线，低优先） |
| rbfe_* (5k) | RBFE-A/B/C 已登记 | 未审 | 未审 | 未审 | TG-8 部分 |
| 小模块群（multi_segment_analysis/stage2_*/step_guard/apbs/free_energy_engine 等） | CR-5、QUALITY-07、C2/C3 已登记 | 未审 | C2/C3 已登记 | 未审 | §3 TG-7/9；multi_segment_analysis 与 step_guard **全覆盖**（正面） |
| local_residual/ | C4–C7、LR-* 已登记 | 未审 | C4–C7 已登记 | 未审 | 未单独映射（todo 提及有专项测试） |
| tools/ | C1 已登记 | 未审 | C1 已登记 | 未审 | 未映射 |

---

## 1. API / config 接线全量核对（本轮新维度，正面结论）

方法：把 `abfe_config.json` 生产 preset 的全部 33 个键逐一 grep 消费点
（`config.get("k"` / `kwargs.get("k"` / `args.k` / kwarg 传名）。

**结论：生产 preset 无死键。** 两个疑似零消费键查明是接线形态特殊：

- `required_consecutive_bias_updates`：runabfe 仅 2 处字面量，实际经 `abfe_pipeline.py:6946-6947、18401-18402、18535-18536、19342-19343` 的 `kwargs.get(..., 3)` 进 IBSSampler（`ibs_engine.py:15102` 形参，消费于 16696/16716/18000/18324）。
- `ibs_lse_log_residual_tolerance`：同形态，`abfe_pipeline.py:6982-6983、18539-18540`，默认 0.5。

`pilot_shadow_checkpoint_interval`（配置为 null）经 `runabfe.py:8609/8873 → abfe_pipeline.py:5971 → 6159` 接到 pilot 快照间隔；`stage2_window_partition` 经 `runabfe.py:7188` 进窗口划分。其余键均有 3+ 直接消费点。

附注：`boresch_source` 配置值为 `"simple"` 而 runabfe 复合物腿 auto 链的默认分支是另一套——该分歧即已登记的 `BORESCH-AUTO-01`（TODO §2.6），本轮不重判。

## 2. 数值稳定性抽查（本轮新维度；抽查 15 处，仅 1 条低级别观察）

### 2.1 ibs_engine.py

逐行实读并确认干净的：

- `adjacent_bar_chain`（2176–2235）：零样本态 fail-closed 抛错（明确阻止 stage2 偏置混合帧被当端点帧）；sum(n_k) 与形状双校验；边方差独立相加有文档标注"保守近似"。
- `_attachment_ti` / `reweighted_fd_ti`（2245–2325）：`|dlam|<1e-12` 拒绝、MBAR 权重和非正拒绝、λ 降序符号约定有数值核对注释。
- `attachment_bar_ti_gate` / `attachment_split_half_gate`（2326–2351）：绝对下限防 σ≈0 时噪声判 7σ（实测教训已写入代码）。
- `pme_self_correction_prefactor_kj` / `energy_kj`（1428–1442）：alpha/qsq 非正返回 0；`pme_offset_charge_square_sum`（1443–1489）对"base 电荷必须清零"前提**就地断言**（0831issue P2 修复在位）。
- `circular_std_rad`（abfe_core 7977，被 ibs 侧引用）：clip 1e-12 防 log(0)；空序列返回 inf 有约定。
- `cluster_periodic_coordinate`（abfe_core 5734）：非有限/L≤0 拒绝；单值退化核对无误。
- REMD 交换核（26250–26300）：**四能量非有限守卫**在 Metropolis 之前；非有限 delta 显式处理（`isneginf` → 接受）；坏探针对不计入接受率分母（有注释说明）。
- `subsample_series_by_autocorrelation`（abfe_core 7786–7847）：09-09 起 fail-closed（非有限输入、pymbar 失败均抛），短序列/无涨落按设计放行。

**NUM-1 [P3|观测性] 非有限交换能量只有 WARN 日志行，无累计计数器。** 位置：`ibs_engine.py:26264-26273`。长时间生产里探针持续坏掉时，唯一痕迹是逐条 WARN + `_diagnose_context_failure` 调用；收敛 JSON/manifest 里没有"非有限交换次数"字段可供事后审计。建议在 stage 采样 meta 里加一个计数器字段。默认路径可达（REMD 是默认 Stage1），影响仅观测性、不影响物理。

### 2.2 abfe_pipeline.py（抽查 4 处，干净）

- 指纹量化链 `_quantize_positions_for_identity`(798)/`_q8`(3513)/round(·,12) 口径一致；`_protocol_fingerprint_ignoring_code_hash`(1178) 的存在本身正确处理了"代码 hash 不该进协议身份"。
- `graceful_termination`(1833) 用 `BaseException` 派生 + `install_termination_handlers`(1872)：保证 Ctrl-C 不被 `except Exception` 吞掉——与 `_PipelineStateLock`(1660) 的注释链一致，语义自洽。
- `ensure_free_disk_for_stage`(1928) 的估算算术（`estimate_stage_trajectory_bytes` 1915）量纲一致，无除法。
- `_pre_equilibration_fingerprint`(812)/`_box_vectors_hash`(997) 走 nm 数组 + 量化，浮点稳定。

### 2.3 abfe_core.py（循环闭合，干净）

`combine_binding_free_energy`（~9030–9110）：所有输入经 `_validate_cycle_input_scalar`（非有限拒绝、误差为负拒绝）；Boresch 项按 `boresch_already_included_in_complex` 防双计；跨腿构象门恒写 WARN report（老板裁决注释在位）。热力学循环唯一实现 + 唯一门，结构正确。

### 2.4 runabfe.py（seed 链，干净）

`_coerce_positive_seed`(3464–3495)：显式拒绝 float 截断（1.5 ≠ 1 时 SystemExit），注释写明"seed 决定随机流身份不允许静默改写"；`_attachment_leg_seed`(3496–3530) 修复了 2026-09-20 的固定常量 seed bug，fail-closed 注释在位。

## 3. 测试缺口映射（本轮主产出；方法：顶层公共符号 → tests/ 全词 grep，零引用即列）

> 注意口径：零引用 ≠ 无行为覆盖（可能经被测上层间接覆盖），但下列均为**值得有直接测试**的数学/守卫函数。判定未对照间接路径，登记前应按 TODO 惯例核实。

### TG-1 [P2] ibs_engine：PME 自能修正数学无任何直接测试
`pme_self_correction_prefactor_kj`(1428)、`pme_self_correction_energy_kj`(1437)、`pme_offset_charge_square_sum`(1443，含关键的 base-charge=0 fail-closed 分支)。该项直接进带电配体的离线 u_kn 账本，公式唯一实现、前提唯一断言，回归时无网。配套 `lambda_endpoint_diagnostics`(1489) 同样零引用。**这是本轮最值得补的一组测试**（纯函数、无需 OpenMM context，成本极低）。

### TG-2 [P2] ibs_engine：冻结 f_k 防篡改/封存守卫簇无直接测试
`sealed_candidate_matches`(8660)、`refuted_candidate_identity`(8678)、`assert_candidate_not_sealed`(8718)、`heldout_candidate_verdict`、`cumulative_fk_span_verdict`。这簇是"冻结候选不得被静默复用"语义的唯一实现，出错形态恰是静默接受旧 f_k——比崩溃更危险。同样多为纯函数。

### TG-3 [P2] ibs_engine：probe 统计判据无直接测试
`probe_bidirectional_overlap`(10863)、`probe_bidirectional_overlap_for_bias_calibration`(11043)、`delta_u_distribution_diagnostics`(1561)、`synthetic_mbar_u_kn`(1697)。其中 `synthetic_mbar_u_kn` 本身就是测试夹具性质的工具，无测试略显讽刺。

### TG-4 [P2] abfe_core：修正项与周期统计无直接测试
`charge_at_lambda`(1992)——docstring 自述"测试、诊断与断言都调它"，实际 tests/ 零引用（**API-1 [P3]**：文档契约与事实脱节）；`circular_std_rad`(7977)、`cluster_periodic_coordinate`(5734)、`co_alchemical_ion_anchor_atom_index`(1710)、`assess_boresch_harmonicity`（B5 已登记其 unwrap 性能，但正确性侧也无测试）。`calculate_constraint_jacobian_correction`(11346) 是已退役 shim 返回 0.0，本身无需测试，但其替代物 `constraint_identity_fingerprint`(11363) 的指纹语义（HMR 质量/约束变化 → 缓存失效）无直接测试。

### TG-5 [P3] runabfe：CLI 自检层无测试
`run_self_tests`、`validate_openmm_cache_only`、`boresch_params_ready`、`resolve_ligand_ffxml`、`run_prepare_command`。

### TG-6 [P3] abfe_preoptimizer：17 个公共函数零直接引用
含 λ 路径数学核心：`quadratic_vanishing_base_lambdas`、`finalize_descending_lambda_path`、`refine_stage_lambda_path_by_overlap`、`densify_lambdas_by_free_energy`、`partition_tail_by_arclength`、`insert_thermodynamic_midpoint_from_ibs_lse_failure`、`rebuild_window_dependent_allocation`、`redistribute_lambda_by_delta_f` 等。该模块有 QUALITY-04 押着的改进计划，改进落地前先把现有行为钉住是前置条件。

### TG-7 [P3] apbs_correction：几何解析层无测试
`DXGeometry`、`PQRGeometry`、`read_pqr_geometry`、`run_apbs`。C2（DX 逐 token 解析）改造时这些是唯一安全网。

### TG-8 [P3] RBFE 管线骨架
`rbfe_pipeline.assert_reusable`、`CycleClosure`、`leg_fingerprint`、`NetworkReport`、`RepeatAggregate`；`rbfe_core` 的 Fragment 分解 schema 类。RBFE 线本身有 RBFE-A/B/C 三个物理缺口待裁决，测试缺口随之押后合理，登记备查。

### TG-9 [正面] 覆盖良好区
`multi_segment_analysis.py`、`step_guard.py` 公共符号**全部**有测试引用；`stage2_rounds.py` 仅 `SegmentSpec` 一个 schema 类零引用。核心统计模块的测试纪律明显好于数学工具函数。

## 4. 本轮明确没做的

- 正确性/性能维度的**全量**重审：控制器/LCR/缓存区（abfe_pipeline 8600–19985）与 ibs_engine 采样循环已被 9-26/9-28 两轮覆盖且有 P1 在账（CR-1、CR-2、A1、BORESCH-AUTO-01），本轮未重读；新 P1 应先等那批核实/修复，不宜再叠第三轮静态报告。
- 外围模块五维审计：abfe_preoptimizer、outer_lambda_neural_basis、rbfe、小模块群、local_residual、tools——只做了测试映射。
- 运行时验证：未跑 pytest、未跑 GPU、无 profiler；全部为静态实读。

## 5. 建议的补审方式（外围 6 区，待用户点名后执行）

沿用本轮已备好的代理分工（每区一个只读审计代理，提示词含：五维要求 + 已知条目清单 + "默认路径必须给 runabfe→config 追踪证据"纪律 + 覆盖声明格式）：

1. abfe_preoptimizer.py（QUALITY-04 关联）
2. outer_lambda_neural_basis.py + lambda_path_versions.py
3. rbfe_core + rbfe_pipeline + runrbfe（RBFE-A/B/C 之外的新项）
4. 小模块群（multi_segment_analysis/stage2_*/step_guard/free_energy_engine/abfe_diagnostics/apbs_correction）
5. local_residual/ 包
6. tools/（pocket_water 新线优先，C1 关联）

并发受限时可改为主对话逐区做，或每波 2–3 个代理。核实与登记按惯例：逐条对照源码后，默认路径成立的进 TODO §2，非默认/驳回进 §3，本文件保持原文不改。

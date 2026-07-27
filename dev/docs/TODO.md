# 当前行动清单

更新：2026-07-27。此文件合并并取代根目录旧 `todo2.txt` 审查报告和 `todolist.md`；历史原文保存在 [archive/todolist-2026-07-20.md](archive/todolist-2026-07-20.md)，审计证据见 [status/AUDIT_STATUS.md](status/AUDIT_STATUS.md)。

## 2026-07-27 本轮改动摘要

用户决定：**接受当前误差（total σ = 0.695 kcal/mol），先把数出完、验证整条链路对不对。** 取舍原则随之确定——凡是会改变 Hamiltonian、输入坐标、Boresch 限制或任何缓存指纹的改动本轮一律不做，否则会让 `output_lrc_fix/` 已积累的采样失效。本轮只做分析侧 / 报告侧 / 进程侧。

已完成（均为纯 CPU 侧改动，不动采样）：**P0-8**（缺首/末窗口 fail closed）、**ATT-09**（统一热力学循环，补上两处缺失的 APBS）、**ATT-04**（消除 import 期 CUDA 初始化）、**P2-14**（LRC 报告诚实性）、以及本文件与 `design/`、`status/`、README 的一批失真同步。新增回归：`test_stage_coverage_fail_closed.py`、`test_thermodynamic_cycle.py`、`test_import_time_side_effects.py`、`test_lrc_reporting_honesty.py`。

**出数之后再做**（都是真 bug，但会毁掉已采数据）：P1-13（可能正是 V-05 里"σ 对 N 不响应"的成因，建议优先）、P1-14、ATT-11。

**2026-07-27 复审新增（本轮只登记，整条链跑完后再修）：**P0-9（`--analyze-only` 的 stage checkpoint/coverage/f_k 契约不完整）、P1-15（stage 缓存丢失收敛与覆盖证据）、P1-16（traditional `--resume` 未接通）、P2-15（单阶段 endpoint diagnostics 语义错误），以及 ATT-22/ATT-24 下新增的测试入口和显式输入文件 fail-closed 缺口。这些改动均不影响当前生产 Hamiltonian，本轮不实施。

**核实过程中发现三处附件/清单本身失真**，已就地更正：ATT-04 的根因归错了模块、ATT-09 的"公式均正确"不成立、ATT-26 的行数差 2.5 倍；另有 ATT-12/λ-19/V-03 描述的是已退役的 v20 λ 协议。

## P0/P1

以下问题已逐项核对当前源码。打勾项已实现 CPU/OpenMM 回归测试；DEXP 项因已有全新 DEXP 版本，按本轮约定暂不修改。

### P0

- [x] **P0-1：主 System 缓存身份已改为 fail closed。** manifest 绑定 GRO/TOP/include 依赖、配体名和构建参数，并校验 System XML、ligand indices、mmCIF topology、box vectors 的哈希；预平衡 DCD 只有在调用方提供完整匹配 fingerprint 时才允许自动读取。

- [x] **P0-2：溶剂腿缓存已绑定配体和力场身份。** v3 manifest 绑定 complex-leg/配体拓扑、参数、FFXML/TOP include 身份及产物哈希；旧缓存或身份不匹配时重建。

- [x] **P0-3：Boresch 再平衡已使用 triclinic minimum-image 距离。** 周期盒缺失、非有限或奇异时直接拒绝计算，不再用裸笛卡尔跨盒距离改写 `r0`。

- [x] **P0-4：会令两个锚点重叠的 IBS 坐标移动已删除。** Boresch 安全检查只以 minimum-image 几何诊断，不再修改配体坐标。

- [x] **P0-5：IBS energy/bias/base 已作为不可分割三文件处理。** 保存时记录协议、shape、帧数及逐文件 SHA-256；resume、skip、pipeline 收集和最终分析均要求三者存在、等长、有限且 manifest 完全匹配，不再补零或按最短长度截断。

- [ ] **P0-6：DEXP 拟合模型与生产模型身份不一致。** Orb fitter 拟合全局 `alpha_vdw`、`beta_vdw`、`r0_vdw`、`A_fit`、`B_fit`、`offset_c0` 模型；生产 `DEXPSurrogatePotential` 却使用 pair-specific `sigma_ij`/`epsilon_ij` 形式，`from_dict()` 不读取 `r0_vdw`、`A_fit`、`B_fit`。CLI 加载拟合 JSON 后会静默丢弃核心拟合参数；必须统一拟合与生产 Hamiltonian，或拒绝不兼容参数文件。

- [ ] **P0-7：DEXP 的 `offset_c0` 被保存但未进入能量。** fitter 计算并保存最优常数偏移，`DEXPSurrogatePotential` 也保存该字段，但 `build_expression()` 不使用它。该偏移随 λ 消失时会贡献自由能差，不能随意忽略，也不能按 pair 重复加入；需以每系统一次、λ-dependent constant 的正确形式进入能量或明确从拟合模型移除。

- [x] **P0-8：IBS 阶段缺预期窗口时已 fail closed（2026-07-27）。** 两个 loader（`ibs_engine.get_stage_data_for_analysis`、`abfe_pipeline._load_ibs_window_outputs_from_dir`）不再对缺文件静默 `continue`，改为累积 `missing_windows` 并在出口调用新的 `ibs_engine._assert_expected_windows_all_loaded()` 抛 `IBSIncompleteStageCoverageError`；`solve_stage_integrated` 结果新增 `coverage_diagnostics`（expected/valid/solved 窗口、dropped、covered λ 索引与首末端点），`local_results`/`chain_segments` 新增 `source_window_index`（原 `window_index` 其实是 local_results 位置下标，有窗口被跳过时两者会分叉）。

  **核实中修正了原描述：中间缺窗其实抓得到**——协方差链要求非首窗与已覆盖态共享 λ，否则走 `_fallback("window_overlap_broken_for_covariance_chain")`（`ibs_engine.py:11390`）。真正会漏的只有缺**首**窗（window 1 自然变成 `local_idx==0`，走 `join_lam = local_lams[0]`）和缺**末**窗（链提前正常结束），两者都产出截断 ΔG 且报 `converged=True`。

  **关键约束**：vanishing rescue 合法地会排除窗口（`abfe_pipeline.py:7269` `excluded_local_windows=set(failing_windows)`，由 rescue ensemble 补上），所以 expected 必须是显式参数，不能直接用 `self.ranges`。回归见 `test_stage_coverage_fail_closed.py`（缺首/缺末/缺中/只缺 bias 各一条 fail-closed，外加"original 排除 + rescue 补上"的正例与 `window_index_offset` 不破坏判定）。

- [ ] **P0-9：`--analyze-only` 没有继承当前 stage 完整性与 ESS 契约（2026-07-27 复审新增；出数后修）。** `runabfe.run_post_analysis()` 的 `_analyze_dual_leg()` 若发现 `stage1_decharging.json` / `stage2_vanishing.json`，只验证 `stage`、`total_delta_G`、`total_error` 的类型与有限性就直接采用；不验证 `protocol_key`、`lambda_path_fingerprint`、`converged`、`coverage_diagnostics` 或窗口 manifest。因此 P0-8 上线前生成的旧/截断 stage checkpoint 仍可能被 `--analyze-only` 当成权威结果。

  没有 stage checkpoint 时的原始窗口回退也未接上新契约：窗口编号只要求等于 `range(len(parsed_indices))`，能发现中间缺窗但发现不了缺**末**窗；构造的 `window_data` 没有从 `checkpoints/ibs_state_*.json` 读取冻结 `f_k`，当前 `ESS_GATE_PROTOCOL_VERSION=2` 的 mixture ESS/occupancy 门因此无法计算，回退分析会 fail closed 为 `converged=False`，功能事实上不可用。修复时应复用 `ABFEPipeline._load_ibs_window_outputs_from_dir()` 的三文件 manifest、expected windows、checkpoint/f_k 读取逻辑，并对 stage checkpoint 执行与主 resume 路径一致的协议与覆盖验证。

### P1

- [ ] **P1-8：IBS 中 DEXP 的 cutoff/switch 配置失效。** DEXP 配置为 `cutoff_distance=0.70`、`switch_width=0.20`，但 `_create_softcore_force()` 无条件使用 1.2 nm cutoff/1.0 nm switch，未区分 DEXP。生产势、成本和拟合范围不一致；必须传递并验证 DEXP 专用参数。

- [x] **P1-9：共炼金反离子选择已改为 PBC-aware bulk-water 判据。** 以离子到最近溶质的 minimum-image 距离为主评分、水配位为次评分；配体电荷在取整前必须落在整数容差内，并支持用多个单价反离子中和多价配体。

- [x] **P1-10：全局 TMBAR 最终误差已使用端点差协方差。** 每个独立窗口段直接读取 `dDelta_f[join,end]`，总误差合并独立段方差；结果中记录 chain segments 和误差方法，uncertainty hard gate 使用同一估计。

- [x] **P1-11：预平衡 fingerprint 已绑定起始坐标、盒子和请求步数。** 所有生产调用点都传入完整身份；缺少完整 fingerprint 时 `load_native_system()` 保守加载初始缓存而不自动读取 DCD。

- [x] **P1-12：溶剂盒已改用 OpenMM `padding=1.5 nm` 语义。** `box_size_nm` 仅保留为带警告的兼容参数，不再作为默认构盒公式。

- [ ] **P1-13：生产灾难回退没有同步记录或截断采样历史分支。** 主循环只把坐标回退到最近一次完整力检查的 `production_pos_backup`，却保留该备份之后已经写入的 `energy_history`/`bias_history`/`base_energy_history`，随后从旧坐标和新随机速度重新生成另一条分支。触发灾难的当前帧在 `collect_energies()` 之前已被跳过，并未直接混入数据；真实问题是被放弃分支与重启分支共享祖先，却仍被当作一条连续时间序列做自相关子采样，相关性和有效样本数口径不再可靠。每次刷新备份时应同时保存三份 history 长度并在回退时同步截断，或显式保存独立 trajectory segment 并按多段轨迹估计相关性；三份历史必须始终同长。

- [ ] **P1-14：首次预平衡前没有修复输入坐标中已经跨盒的分子。** `runabfe.center_system_rigidly()` 只把整个体系做一次质心平移，却有调用点随后声称“分子完整性修复完毕”；真正的 `mdtraj.image_molecules()` 位于 `ABFEPipeline.run_full_pipeline()` 的预平衡之后，因此 GRO/缓存里已经跨边界的分子会先进入最小化或 NPT 预平衡。应在第一次创建 Context/最小化/预平衡之前，依据拓扑和 triclinic 盒矢量把每个连通分子仅作整分子周期平移，再做全体系/配体居中；禁止旋转、缩放或改变任何分子内相对坐标，修复失败时 fail closed，不能回退为仅质心平移后继续。

- [~] **P1-15：stage checkpoint 丢失关键收敛与覆盖证据（2026-07-27 复审新增）。主因已修，resume 侧复检仍待做。** `ABFEPipeline._build_stage_cache_payload()` 的注释声称保存 `_run_dual_lambda_stage` 的“完整结果”，实际只落盘 `stage/total_delta_G/total_error/n_states/protocol_key/lambda_path_fingerprint/method/diagnostics/lambda_endpoint_diagnostics`。`solve_stage_integrated()` 顶层的 `converged`、`coverage_diagnostics`、`window_overlap_diagnostics`、ESS/occupancy、去相关样本数、最大端点 σ、covariance segments 与 rescue provenance 均被丢弃。

  **2026-07-27 追查到的真正根因（比原描述更具体）：** 不只是 payload 少存字段——vanishing rescue 合并那条路径在 `run_full_pipeline` 里**直接调 `solve_stage_integrated`、绕过 `_run_ibs_stage`**，于是那段 `stage_result["diagnostics"].update({...})` 从来没执行过，payload 存的 `result.get("diagnostics", {})` 自然是空字典。这就是为什么 2026-07-27 那次 `ΔG_vdw = 145.908 ± 1.384` **完全没有审计痕迹**：`stage_diagnostics.stage2 = {}`、`immutable_bridge_rescue` 全盘搜索零命中、`pipeline.log` 在 11:48:00–12:12:21 之间一行都没有。

  - [x] 抽出 `ABFEPipeline._populate_stage_diagnostics()`（原 `_run_ibs_stage` 内联块，逐字不变），并补齐 `converged`/`coverage_diagnostics`/`covariance_chain_segments`/四个门的实际值与阈值/`immutable_bridge_rescue`/`production_rescue_targets`。
  - [x] rescue 合并分支也调它，并补一行合并摘要日志 + 逐段 ΔG/σ，消除 24 分钟日志空白。
  - [x] `_build_stage_cache_payload` 落盘 `converged` 与 `coverage_diagnostics`。**陷阱**：`_atomic_write_json` 用的是不带 `cls=NumpyEncoder` 的 `json.dump`，而这些字段含 numpy，直接塞进去会 `TypeError` 让整个 checkpoint 写失败——新增 `_json_safe()` 递归转原生类型（NaN/Inf → `null`）。`protocol_key` / `lambda_path_fingerprint` **刻意不过** `_json_safe`，它们参与缓存身份比对，形状变换会让 resume 误判。
  - [x] 回归 `test_stage_diagnostics_persistence.py`：`_json_safe` 覆盖、payload 必须能被不带 `cls=` 的 `json.dumps` 直接写出、每个直调 solver 的函数必须在**同一函数体内**填 diagnostics（早先只断言"文件里 solve 之后某处有 populate"会因行号顺序平凡通过）。`_run_shadow_ibs_decharging_leg` 显式豁免——它返回的是 bridge+shadow 两段的组合结果，顶层本就没有那些量，且已把子结果嵌在 `diagnostics.shadow_ibs_leg`；另有一条测试钉住这个豁免前提。
  - [ ] **仍未做**：resume 命中后重新执行 `_assert_stage_result_sane()`（现在只是把证据存下来了，没有在复用时重新验一遍）。

- [ ] **P1-16：traditional 模式的 `--resume` 没有接通（2026-07-27 复审新增；出数后修）。** `TraditionalABFEPipeline.run_leg()` 已有 `resume` 参数和 u_kn/REMD 轨迹复用逻辑，但 `run_full()` 没有 resume 参数，并在 decharging/vanishing 两次调用中都硬编码 `resume=False`；`runabfe.run_traditional_mode()` 也无法向下传递 `config.resume`。因此 CLI/配置里的 `--resume` 对 traditional 模式完全无效，会重复运行两条腿。应把同一个显式 resume 值贯穿 `run_traditional_mode → run_full → run_leg`，同时保留现有协议指纹和完整轨迹门。

## 附件审查复核

以下项来自用户提供的审查文本；打勾项已经源码核实并关闭或修复，未打勾项仍保留为待办。

GitHub 跟踪（均为未验证审查发现）：

- P0：ATT-01 [#42](https://github.com/Cedrus810/openmm_IBS_dev/issues/42)、ATT-02 [#43](https://github.com/Cedrus810/openmm_IBS_dev/issues/43)、ATT-03 [#44](https://github.com/Cedrus810/openmm_IBS_dev/issues/44)、ATT-04 [#45](https://github.com/Cedrus810/openmm_IBS_dev/issues/45)、ATT-05 [#46](https://github.com/Cedrus810/openmm_IBS_dev/issues/46)、ATT-06/07 [#47](https://github.com/Cedrus810/openmm_IBS_dev/issues/47)。
- P1：ATT-08 [#48](https://github.com/Cedrus810/openmm_IBS_dev/issues/48)、ATT-09 [#49](https://github.com/Cedrus810/openmm_IBS_dev/issues/49)、ATT-10 [#50](https://github.com/Cedrus810/openmm_IBS_dev/issues/50)、ATT-11 [#51](https://github.com/Cedrus810/openmm_IBS_dev/issues/51)、ATT-12 [#52](https://github.com/Cedrus810/openmm_IBS_dev/issues/52)、ATT-13 [#53](https://github.com/Cedrus810/openmm_IBS_dev/issues/53)、ATT-14 [#54](https://github.com/Cedrus810/openmm_IBS_dev/issues/54)、ATT-15 [#55](https://github.com/Cedrus810/openmm_IBS_dev/issues/55)、ATT-16 [#56](https://github.com/Cedrus810/openmm_IBS_dev/issues/56)、ATT-17 [#57](https://github.com/Cedrus810/openmm_IBS_dev/issues/57)、ATT-18 [#58](https://github.com/Cedrus810/openmm_IBS_dev/issues/58)。
- P2：ATT-19 [#59](https://github.com/Cedrus810/openmm_IBS_dev/issues/59)、ATT-20 [#60](https://github.com/Cedrus810/openmm_IBS_dev/issues/60)、ATT-21 [#61](https://github.com/Cedrus810/openmm_IBS_dev/issues/61)、ATT-22 [#62](https://github.com/Cedrus810/openmm_IBS_dev/issues/62)、ATT-23 [#63](https://github.com/Cedrus810/openmm_IBS_dev/issues/63)、ATT-24 [#64](https://github.com/Cedrus810/openmm_IBS_dev/issues/64)、ATT-25 [#65](https://github.com/Cedrus810/openmm_IBS_dev/issues/65)、ATT-26 [#66](https://github.com/Cedrus810/openmm_IBS_dev/issues/66)；ATT-27 复用 [#35](https://github.com/Cedrus810/openmm_IBS_dev/issues/35)，ATT-28 复用 [#41](https://github.com/Cedrus810/openmm_IBS_dev/issues/41)。

### P0 候选

- [x] **ATT-01：附件结论不成立。** `scan_boresch_1d_pes` 是 `OrbScanner` 的实例方法，`self` 并非孤立顶层参数。

- [x] **ATT-02：附件结论不成立。** 当前常量拼写为 `VANISHING_TARGET_INTERVALS_PER_ENSEMBLE`，模块可导入且相关路径测试通过。

- [x] **ATT-03：单 GPU Context 无界常驻风险已消除。** GPU/OpenCL 默认上限为一个常驻 Context；当前交换实现超过上限时会在创建任何 GPU Context 前回退 CPU，避免 OOM 后补救。若构建仍发生 GPU OOM，也会先释放已建 Context 再回退；这是安全优先的实现，尚未实现 GPU 分批加速。

- [x] **ATT-04：import 期 CUDA 初始化已消除（2026-07-27）。附件的归因是错的。**

  它怪 `_run_stage_worker_process()` 里函数作用域的 `from abfe_pipeline import ABFEPipeline`（`abfe_pipeline.py:929`）——**删掉那行没用**，spawn 反序列化 target 时本来就要 import 该模块。真正的根因是一行模块级语句 `abfe_core.py:321`：`GLOBAL_DEVICE, SUPPORTS_TF32 = get_optimal_device_settings()`，它在 import 期调 `torch.cuda.get_device_capability()`（惰性建 CUDA context）和 `torch.set_float32_matmul_precision("high")`。链路 `abfe_pipeline → ibs_engine:31 → abfe_core`，每个 spawn 子进程（`abfe_pipeline.py:6848`、`ibs_engine.py:13219` 均用 `mp.get_context("spawn")`）必经。

  **附件的另一半也不成立**：三个文件里所有 OpenMM `Platform.getPlatformByName` / `Context` 调用**都在函数内**，OpenMM 侧没有 import 期副作用。

  **加重情节**：子进程 GPU 归属只通过 OpenMM `props["DeviceIndex"]` 表达，从不设 `CUDA_VISIBLE_DEVICES`，所以双 GPU 并行时两个子进程都会先在 device 0 上建 torch context。

  已改为惰性 memoized（`_resolve_device_settings` / `get_global_device()` / `supports_tf32()`）；全仓仅两个消费者 `abfe_core.py:2629`、`:2887`，都是 MACE/ML 入口的默认参数值，改成 `device=None` 后在函数体内解析，不在 softcore IBS 生产路径上。回归见 `test_import_time_side_effects.py`（干净子进程 import 后 `torch.cuda.is_initialized()` 必须为 False，外加三个模块顶层调用的 AST 静态兜底）。

- [x] **ATT-05：同 P0-5，已实现三文件 manifest 与 fail-closed 验证。**

- [ ] **ATT-06：DEXP 拟合模型与生产 Hamiltonian 不同。** 附件称 Orb fitter 的全局 `r0_vdw`、`A_fit`、`B_fit` 等参数未被生产 `DEXPSurrogatePotential.from_dict()` 读取，造成拟合参数静默丢失。需统一拟合/生产模型或拒绝不兼容拟合 JSON。

- [ ] **ATT-07：DEXP `offset_c0` 未进入能量。** 附件称 offset 被拟合、保存但未在 `build_expression()` 使用；若其随 λ 消失会贡献 ΔG。需要以每系统一次的 λ-dependent constant 正确纳入，或从拟合契约中显式移除。

### P1 候选

- [x] **ATT-08：附件结论不成立。** `topology` 用于核对 complex topology 中的配体原子数与 `ligand_indices`，是输入一致性 hard gate。

- [x] **ATT-09：已提取单一实现，并修好两处漏项（2026-07-27）。附件"公式均正确、只是分散维护"这句不成立。**

  逐处核对的真实差异：

  | 位置 | Boresch | APBS |
  |---|---|---|
  | `runabfe.main()` | 已内含 | ✅ 加了 |
  | `runabfe.run_traditional_mode()` | 显式减 | ❌ **完全没有** |
  | `runabfe.run_post_analysis()` | 条件置零 | ✅ 从 `run_provenance.json` 重推 |
  | `abfe_pipeline.run_full_abfe_loop()` | 已内含 | ❌ **完全没有** |

  后两条路径对带电配体静默漏掉整项有限尺寸静电修正——数值错误，不是整洁性问题。`THERMODYNAMIC_CYCLE_DOC` 只是散文，不是可复用代码；`grep "delta_g_bind" test_*.py` 此前零命中。

  现已统一到 `abfe_core.combine_binding_free_energy()`（紧邻 `THERMODYNAMIC_CYCLE_DOC`），四处全部改调它并显式传 `boresch_already_included_in_complex` 与 `apbs_correction_kJ_mol`。APBS 仍是离线 `apbs_correction.py`（prepare → run → collect）算出的标量，经 `--apbs-correction-kj-mol` 传入——流程内不调 APBS，只消费同一个值（collect 的输出里直接给了要传的参数）。顺带删掉 `run_post_analysis` 里与 `boresch_included_in_complex_dg` 重复记账、且已无人读取的 `dg_boresch_term`。

  **这不是纯重构**：site 2 / site 4 在 APBS≠0 时输出会变，那是修复。`run_full_abfe_loop` 目前无仓内调用方（对外 API），修复是潜在生效的。回归见 `test_thermodynamic_cycle.py`（解析 toy cycle 钉到 1e-12、两种 Boresch 约定交叉验证、APBS 加性与缺省为 0、真结合物符号为负，外加四个调用点必须调该实现且必须显式传两个开关的 AST 契约）。

- [x] **ATT-10：能量查询失败 hard gate 已实现。** 连续失败上限 5、总失败上限 10、失败率上限 1%；生产结束时即使样本不足 100 帧也检查失败率，并把 attempts/failures/reasons/limits 写入诊断。

- [ ] **ATT-11：`GeometricRestraintEstimator` 的 0.22 nm 键距离阈值。** 该阈值可能漏掉边界 S-S/配位键并误判近距离非键接触；需以元素/拓扑信息或可配置、经验证的判据替代单一几何阈值。

- [x] **ATT-12：vanishing 路径/子域范围契约已明确。** ~~v20 采用确定性的 17 点 `λ=x²` 锚点、λ≈1 四点增密及两个数据驱动 Fisher bridge；pilot 只可插入 bridge，不得移动/删除平方锚点。~~
  **⚠️ 2026-07-27：删除线部分描述的是已退役协议。** 代码是 `THERMODYNAMIC_PATH_PROTOCOL_VERSION = 21`，布点整体换成 `blended_metric_vanishing_lambdas`（度规弧长与几何进度按 β=0.3 混合后等分），没有平方锚点、没有 Fisher bridge。仍然成立的只有"六个单边界共享窗口"这一半。当前契约见 [design/LAMBDA_SCHEDULE_CONTRACT.md](design/LAMBDA_SCHEDULE_CONTRACT.md)。

- [x] **ATT-13：TMBAR history 已设有界契约。** 最多保留 200 个 minibatch，保存累计丢弃计数，恢复时只校验并加载最新有界后缀。

- [ ] **ATT-14：IBS 中 DEXP cutoff/switch 配置可能被硬编码覆盖。** 附件称 `_create_softcore_force()` 对 DEXP 也无条件写 1.2 nm cutoff/1.0 nm switch，而配置为 0.70/0.20。需验证参数实际传递并禁止配置/生产势不一致。

- [x] **ATT-15：同 P1-9，PBC/bulk-water、电荷整数性及多价配体均已修复。**

- [x] **ATT-16：同 P1-10，已使用逐窗口段的直接 `dDelta_f`。**

- [x] **ATT-17：同 P1-11，完整预平衡身份已接入所有生产调用点。**

- [x] **ATT-18：同 P1-12，已改用 `addSolvent(..., padding=1.5 nm)`。**

### P2/发布质量候选

- [ ] **ATT-19：核心物理单元测试覆盖不足。** 附件列出软核势 λ=0/0.5/1、DEXP LJ-matching、IBS log-sum-exp 稳定性、窗口拼接连续性、PBC、离子计数、resume、并行 worker 等缺口；需先盘点现有测试，形成最小覆盖矩阵。

  **2026-07-26 进展（部分完成，仍 open）。** 已盘点：原有 5 个测试文件（3752 行）几乎全是**协议契约**测试——读源码文本断言门存在、协议版本进指纹、旧缓存 fail-closed；能防"改代码忘了递增版本号"，但没有任何一条把数值跟手算期望值比过。本轮新增两个文件补数值侧：
    - `test_core_physics_numerics.py`：Boresch 解析修正（独立 log-sum 参考实现，<1e-6；并用变温反解验证 `(2πRT)³` 的指数确为 3；r0 单位量级错误可显形）、`solve_stage_integrated`（合成同弹簧常数谐振井使 `F_k-F_j=Δ_k-Δ_j` 精确成立 → 2 窗口拼接的 ΔG 有解析真值；退化权重那条同时是 `u_mbar *= beta` 修复的回归测试）、`IBSBiasForce`（Reference platform 求值 vs 手算 log-sum-exp，含 `logit≈+1000` 的 max-pivot 抗溢出回归）、`estimate_f_k_from_pilot_ti`（梯形 TI + mean-centering 数值验证，**显式断言符号**而非仅单调性）。
    - `test_resume_reuse_contracts.py`：`_invalidate_stage_window_files` 的 reuse_map（含 idx 交换不互相覆盖、记账口径任一不符即清理）、`_resume_cached_window_gate_status` 的 8 个门（逐门打偏 + **逐字段删除**两套参数化，验证缺字段 fail-closed）、`ShadowBridgeREMDManager` 的 s 参数（断言只写 `lambda_bridge_s`，绝不写基类占位的 `lambda_coul`/`lambda_vdw`）。
    - 配套 `pytest.ini` / `conftest.py` / `run_offline_tests.sh`（见 ATT-22）。
    - 为了让 resume 门可单测，`ibs_engine.py` 把 `run_all_windows` 里内联的 ~110 行 8 门判断抽成模块级纯函数 `_resume_cached_window_gate_status`，**逐门语义与阈值一字未改**，调用侧那串逐门诊断打印保持原样。等待运行证据，登记为 `VAL-TEST-006`。

  **仍未覆盖（ATT-19 保持 open 的理由）：** 软核势 λ=0/0.5/1 端点行为、DEXP LJ-matching、PBC/离子计数（部分已在 `test_todo_verified_fixes.py`）、并行 stage worker（与 ATT-04 相关）、热力学循环 toy-cycle（属 ATT-09）。

- [ ] **ATT-20：缺少已知 ABFE 基准的端到端集成验证。** 需为公开基准体系建立可复现脚本，覆盖中性/带电配体、两腿循环闭合与实验对比；在此之前不能以发布级精度宣称生产结果。

- [ ] **ATT-21：文档缺口（2026-07-27 已核对，附件前提大半不成立）。**

  **已存在、附件说缺其实不缺**：根目录三份 README（`README.md` 466 行 + `README_en.md` + `README_cn.md`），含 依赖(:83) / 输入要求(:102) / 快速运行(:114) / 配置示例(:160) / 命令入口(:212) / 输出结构(:285) / 结果解读(:323) / 缓存与续跑(:351) / 并行与 GPU(:373) / FAQ(:392，8 个小节)。

  **真正缺的只有三样**：① API 参考（全仓没有模块/类/函数级文档）；② 独立的热力学循环推导文档（目前只散落在 `README.md:443` 与 `AUDIT_STATUS.md:694`，且公式的唯一实现现在是 `abfe_core.combine_binding_free_energy`，文档应指向它）；③ 打包元数据（无 `pyproject.toml`/`setup.py`，安装只有散文 + `environment.yml`）。

  - [x] **README 死链已修（2026-07-27）**：`PHYSICS_DEFECTS.md`（不存在）、`todolist.md`（不存在，已被 `docs/TODO.md` 取代）、以及用裸文件名引 `AUDIT_STATUS.md` / `VALIDATION_MATRIX.md`（实际在 `docs/status/`）。

- [ ] **ATT-22：缺少 CI/CD、静态检查与格式化。** 建立 GitHub Actions（或等价 CI）、pytest、ruff/flake8、mypy、black/isort 等分层门槛，并避免把需要 GPU 的作业误放入普通 CI。

  **2026-07-26 进展（只完成"统一 pytest 入口"这一小步，仍 open）。** 此前根目录没有任何 pytest 配置，测试文件混用 unittest/pytest 风格，无法只挑"不需要 GPU 的"跑，也没有固定的 pre-flight 命令。已补：
    - `pytest.ini`：`testpaths`/`python_files`、`norecursedirs`（排除 `output*`/`tmp`/`docs`/`.pycache_check` 等，避免在 NFS 上递归收集运行产物与旧同名副本）、注册 `cpu_only`/`needs_gpu` 两个 marker、`addopts = -ra`（让 `importorskip` 的静默跳过在末尾显形，不再伪装成 all-passed）。
    - `conftest.py`：只把仓库根目录加进 `sys.path`（此前 `import ibs_engine` 能成功纯靠"恰好从根目录启动"）。**刻意不在其中 import openmm**——首次导入要 60-100 s，且 ATT-04 正在追踪导入期 CUDA 初始化风险。
    - `run_offline_tests.sh`：`mamba activate openmm_dev` + `pytest -m "not needs_gpu"`，一条命令跑完全部 CPU 测试。
  **仍缺：** GitHub Actions（本仓库无 CI）、ruff/flake8、mypy、black/isort，以及"GPU 作业不得进普通 CI"的实际隔离配置。

  **2026-07-27 复审新增：统一入口本身当前不可直接运行。** 脚本先执行 `set -euo pipefail`，随后 `mamba activate openmm_dev` 触发环境的 `activate.d/env_vars.sh`；该脚本读取未定义的 `CPATH`/`LIBRARY_PATH`，在 `nounset` 下立即退出。直接调用 `/home/ruigengji/mambaforge/envs/openmm_dev/bin/python -m pytest -m "not needs_gpu" -q` 可正常跑完 280 tests。修复入口时应在激活前安全初始化这些可选变量，或把 `nounset` 的启用移到环境激活之后；不能把“脚本失败”误报成测试失败。

- [ ] **ATT-23：运行恢复与资源保护能力不足。** 评估 GPU OOM 的降级/Context 回收、长任务中断后的窗口重跑判定、磁盘空间预检和运行时估计；必须保持科学状态不可变与 fail-closed 原则。

- [ ] **ATT-24：输入验证不足。** 补充 ligand 残基名、TOP include、配体原子数/Boresch 可构建性、最小盒尺寸（至少 2×cutoff）等明确的前置报错和诊断。

  **2026-07-27 复审新增三个确定的静默降级入口：**显式 `--config missing.json` 时 `RunConfig` 会跳过加载并继续使用 production preset；显式 `--dexp-params missing.json` 时主流程会把 `dexp_params=None` 并使用默认 DEXP 参数；显式 torsion 参数文件不存在时会静默当作无 torsion 修正。用户已经明确指定文件时，文件缺失/不可读/格式不符必须 fail closed，不能换一套 Hamiltonian 或配置继续运行。

- [ ] **ATT-25：协议版本矩阵缺少统一注册/迁移工具。** 当前多处独立 protocol version 需要统一注册表、缓存指纹组合规则、迁移说明和兼容性测试，避免单个版本更新遗漏缓存失效。

- [ ] **ATT-26：`run_all_windows` 过长且职责混杂。** 将流程逐步拆分为恢复/建系/最小化与 Boresch/预热状态机/生产采样/落盘与 checkpoint 等独立方法，并保持行为回归覆盖。
  **2026-07-27 更正规模**：不是"约 1200 行"，实测 `ibs_engine.py:7127-10181` = **3055 行**（差 2.5 倍）。同理下方 ATT-19 里说抽出的 `_resume_cached_window_gate_status` "~110 行"，实际 `ibs_engine.py:3398-3602` = 205 行。行数会随本轮改动小幅漂移，重新核对时以函数首尾行为准。
  **暂缓理由**：3055 行、行为高度耦合，在没有真实端到端回归之前拆它风险大于收益。

- [ ] **ATT-27：死代码与不可达逻辑清理。** 包括已标记 deprecated 的 overlap autorepair、withdrawn 协议注释、未调用扫描函数和立即 raise 的 `enable_lambda_refine`；与 E-03 协同，先归档再移除可执行死路径。

- [ ] **ATT-28：日志实现分裂。** 多个模块分别覆盖 `print` 或使用标准 `logging`，可能导致导入顺序相关行为；需统一结构化日志入口、级别和文件/控制台策略。

本轮从旧 `todo2.txt` 复核并完成：

- [x] A-01/A-02：传统 REMD LRC 生产者当前采用 v3 逐 λ、switching+softcore-aware `r^-6/r^-12` 系数，并使用 worker 实际读取的 `lj_tail_lrc_coeff_kj_mol` 键。
- [x] A-03：PME context 查询失败后改用 cutoff/tolerance 闭式派生 alpha，不再读取自动 PME 下通常为零的静态参数。
- [x] A-04：`tmbar_history` 上限设为 200 个 minibatch，checkpoint 保存丢弃计数，resume 只恢复最新有界后缀。
- [x] A-09：`ensure_owned_system` 的早退路径先验证底层 OpenMM 对象仍可访问。
- [x] A-11：base 能量第一次失败时立即检查坐标和力；发现非有限值则停止 MD。
- [x] A-18：JSON checkpoint 在原子替换前执行 flush/fsync；POSIX 额外同步父目录。
- [x] λ-19：真实 v18 运行暴露 Fisher 排点把去耦尾部压缩为 `0.9225, 0.8382, 0`。~~v20 生产路径固定保留 17 点 `λ=x²` 和 λ≈1 的 4 个增密点，再由 pilot 只插入 2 个 Fisher bridge~~。
  **⚠️ 2026-07-27：v20 那套已退役。** v21 改为把 pilot 度规弧长 `s_hat` 与几何进度 `1-λ` 按 β=0.3 混合后等分（`blended_metric_vanishing_lambdas`）——既不像 v18 让度规把尾部饿死，也不像 v19/v20 完全忽略实测度规。结果仍是 23 态、6 个单边界共享窗口、总槽位 28，但 λ 值完全不同（末边 `0.100049→0`，不是 `0.00390625→0`）。
- [x] SOLV-ION：溶剂腿从隐式纯水/仅中和改为显式 0.15 M NaCl，并保留必要中和离子；当前 v3 manifest 绑定完整输入身份，旧的 `0 Na / 0 Cl` 或旧身份缓存自动失效重建。

上述修改仍需按 [status/VALIDATION_MATRIX.md](status/VALIDATION_MATRIX.md) 的环境门槛补齐完整依赖/GPU 证据。

### 2026-07-26 本地验证

> **⚠️ 2026-07-27 一致性提醒：本节的勾与验证矩阵互相矛盾。** 下面打勾的
> "119 passed"/"231 passed" 与 [status/VALIDATION_MATRIX.md](status/VALIDATION_MATRIX.md)
> 的 `VAL-TEST-006`（验收命令正是这批、日期同为 2026-07-26、状态 `待运行`）冲突，
> 且该矩阵里**没有任何一项是 `通过`**。需要一次真实运行来决定：是把矩阵条目关掉，
> 还是把这里的勾去掉。在此之前，这两个数字只能当"某次本地跑过"的记录，不能当验收证据。

- [x] 全量 CPU/OpenMM pytest：119 passed。
- [x] 新增 `test_todo_verified_fixes.py`，覆盖三文件 manifest、缺文件/篡改/长度不符拒绝、1%/10 帧/连续 5 帧能量失败阈值、triclinic minimum-image、PBC 反离子与多价/非整数电荷、预平衡 pose/box/steps 身份、主缓存 topology hash，以及 REMD 在创建 GPU Context 前的上限回退。
- [x] DEXP 相关 P0-6/P0-7、P1-8、ATT-06/07/14、P2-14 本轮明确不改，等待新 DEXP 版本接入后按新实现重新审计。
- [ ] 仍需真实 CUDA/GPU 与生产体系运行证据；本轮测试没有宣称完成这些环境验收。

### 2026-07-27 复审验证

- [x] 五个主模块 `py_compile` 通过。
- [x] `python runabfe.py self-test` 通过（Boresch、PME helper、λ 端点、缓存指纹、合成 MBAR 与循环文档检查均 PASS）。
- [x] 直接使用 `openmm_dev` 环境 Python 执行完整 CPU/OpenMM/PyMBAR 套件：**280 passed**。
- [ ] `run_offline_tests.sh` 入口自身因 ATT-22 新增的 `set -u`/mamba 激活环境变量冲突而失败；这是脚本入口 bug，不是 pytest 失败。
- [ ] 仍没有据此宣称完成真实 CUDA/GPU 验收；GPU 项继续按 `status/VALIDATION_MATRIX.md` 跟踪。

### 2026-07-26 ESS 门重构（`ESS_GATE_PROTOCOL_VERSION = 2`）

起因：Stage 2 的 `ess_ratio`/`absolute_ess` 门长期报 1–4%，rescue 循环永远过不了。根因**不是**采样不足、也**不是** f_k 有问题（f_k 是收敛的，占据 0.249–0.251 平坦）。

单参考增广 MBAR（`n_k=[N,0,…,0]`）的权重是 `exp[(V_bias − U'_k)/kT]`，而 `V_bias` 读的是 OpenMM force groups **{1,4}**——Group 4 是 λ-WCA 防护壳，窗口内对所有 k 用同一个 `lambda_shield`；加上 LRC（target 有、bias CV 没有），两者构成一个**逐帧共模因子** `r_n`。实测 σ_r 从 λ_vdw≈1 端的 0.95 kT 涨到 λ_vdw→0 端的 2.40 kT，按 `exp(−σ_r²)` 把 ESS/N 上限压到 0.40→0.003（window 5 实测 0.0029，预测 0.0029）。共模项在 occupancy、相邻 ΔF、以及**真正报出去的物理态↔物理态 ΔF** 里大部分抵消，所以旧门衡量的是"防护壳收了多少重加权税"，与输出精度无关——window 5 raw ESS 最差（0.0029）但 endpoint_σ 最好（0.068 kJ/mol）。

- [x] 收敛门换成三份正交证据：`min_overlap`=扣掉共模因子的混合覆盖度 ESS(p_k)/N_decorr（阈值 0.05）、`min_occupancy_normalized`=min K·⟨p_k⟩（复用 `IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION`=0.5）、`min_decorrelated_samples`(20) + `max_endpoint_uncertainty_kJ_mol`(1.0)。`raw_min_overlap`/`raw_min_absolute_ess`/`top1pct_raw_weight`/`common_mode_log_sigma_kT` 只报告不设门。
- [x] `absolute_ess` 阈值退役：它恒等于 `min_ess_ratio × n_frames_decorrelated`（denom 是标量，`min(neff)/denom == min(neff/denom)`），不是第二份独立证据。`final_min_absolute_ess=50` 在 N_decorr=114 时等价于要求 ratio≥0.44，而日志报的门是 0.05——真实条件是 `ratio ≥ 50/N_decorr`，样本越少门越严，与"延长采样"方向相反。实现上把 `min_absolute_ess_threshold` 置 `None`，`abfe_pipeline.py` 那两处"字段存在才检查"的镜像判据自动失活，未改 pipeline 逻辑。
- [x] 去相关序列从 `base_kj + bias_kj`（被溶剂涨落主导）换成每个目标态的 `Δu_k=(U'_k − V_bias)/kT`、取 g 最大者（`_decorrelate_by_worst_target_state`）。实测旧口径低估 g 3–10 倍（窗口 0/2/5：1.76/2.58/2.44 vs 19.6/26.5/7.7），会让喂进 MBAR 的 n_k 虚高、误差棒系统性偏小。
- [x] 两份增广矩阵实现都对 `sampled_distribution_row != 0` fail closed。旧代码只对越界值回退到 0，放过"合法但非零"的值——那会让 MBAR 以为样本来自某个*物理*态（物理错误），并让 `zip(win_lams, ess_ratio)` 的 ESS→λ 映射整体错位。该不变量原先只写在注释里、没有 enforce。
- [x] f_k 从 `checkpoints/ibs_state_*.json` 读入 `get_stage_data_for_analysis` 与 `_load_ibs_window_outputs_from_dir`（后者新增必填 `checkpoint_dir`，original/rescue 两个 output_dir 必须各配自己的 checkpoint 目录）。不新存文件、旧缓存可用、无需协议号升级或重算。
- [x] 全量 CPU/OpenMM pytest：231 passed。新增回归：受门量必须是 mixture 版、raw 只作诊断、缺 f_k 时 fail closed、去相关序列必须来自权重指数、"ESS 报健康 + 占据报饿死"的组合。

**三个踩到才发现的坑，改这块之前必读：**

1. **gauge-free 捷径不成立。** 试过用逐帧 softmax（`r_n=mean_k` 或 `logsumexp_k`）除掉共模因子以省掉 f_k 传参：对相邻 ΔU ≲2.5 kT 的窄窗口（3/4/5）与真 f_k 一致，但 window 0（相邻 ΔU≈10 kT）给 0.014 vs 真值 0.500，差 36 倍——f_k 加权的 logsumexp 与无权算术平均在谱宽大时是完全不同的逐帧函数。**f_k 必须真的传进来。**
2. **ESS 逐态尺度不变，单用有盲点。** `(Σw)²/Σw²` 在 `w→c·w` 下不变，所以一个"均匀"被饿死的态（f_k 未补偿、U' 高出 80 kT，p_k 逐帧都是 ~e⁻⁸⁰ 但相对起伏很小）ESS/N 仍报 0.34。必须配一阶矩 K·⟨p_k⟩（该例给 3.25e-35）。ESS 查二阶矩（权重是否集中在少数帧），⟨p_k⟩ 查一阶矩（这个态到底有没有拿到权重），两个失效模式各需一项覆盖。
3. **warmup 的 TMBAR trust 门是另一个消费者，必须继续读 `raw_*`。** `tmbar_history` 的 entry 天生没有 f_k（每条在不同 f_k 下采的，这正是 TMBAR 存在的理由），读 mixture 量会让 `tmbar_candidate_trusted` 永久 False、把 warmup 永久钉死在受限占据反馈上。它问的是"从这批原始权重解出的绝对 f_k 敢不敢直接用"——权重退化本身就是不敢用的理由，raw 单参考 ESS 才是对的悲观度量。两个消费者要的确实不是同一件事。

改完后真实运行（Atenolol vdw，rescue round 2/2）：rescue 目标从 6 个窗口收缩到 [2, 3]，`ESS_ratio` 从 0.012–0.039 变成 0.485–0.988，`failed` 只剩 `['endpoint_uncertainty']`。`absolute_ESS=27.70`（<50）已不再触发失败，确认阈值退役生效。

## 当前运行验证

- [x] **V-01：已满足（2026-07-27 复核关闭）。** 验收条件是"窗口 0 进入生产，而不是再次抛 `IBSWarmupConvergenceError`"。2026-07-26 真实运行里窗口 0 已完整跑完预热+生产并给出 ΔG=76.28 kJ/mol、`endpoint_σ`=0.934（见下方 V-05 的表），落盘证据 `output_lrc_fix/vanishing/dual_window_0_vdw_*.npy`。
  注意 [handoffs/VANISHING_WINDOW0_HANDOFF.md](handoffs/VANISHING_WINDOW0_HANDOFF.md) 写于 2026-07-19/20，停在 v23 且自述"尚未经过 GPU 验证"；`IBS_BIAS_PROTOCOL_VERSION` 现为 **29**，其中 v21/v22 的反号已在 **v27 被证实为错并撤销**（changelog `ibs_engine.py:3216-3228`）。该 handoff 只能当历史时间线读，不要当现状。
- [ ] V-02：运行传统 `single_lambda`/REMD 的小型固定盒回归，确认每个 task 收到有限、长度等于态数的 v3 LRC 数组，且 worker 的每帧修正等于 `coeff[k]/V(t)`；GitHub [#32](https://github.com/Cedrus810/openmm_IBS_dev/issues/32)。
- [ ] **V-03（验收条件已按 v21 重写，2026-07-27）。** 原条目写的是 v20 特征——"两个 Fisher bridge、末边 `0.00390625→0`"——**在当前代码上物理上不可能满足**：`abfe_preoptimizer.py:328 THERMODYNAMIC_PATH_PROTOCOL_VERSION = 21`，布点算法整体换成 `blended_metric_vanishing_lambdas`（`(1-β)·s_hat + β·(1-λ)` 等分，β=0.3），v20 的平方锚点/bridge/`0.00390625` 全部不存在。已打勾的 ATT-12 与下方 λ-19 同样描述的是**退役协议**，只作历史记录。

  **v21 新验收条件**（缓存证据已在 `output_lrc_fix/checkpoints/preopt_dual_vanishing.json`，生成于 2026-07-26 16:54）：`path_protocol_version=21`、`lambda_placement_method="fisher_metric_blended_with_geometric_floor_v21"`、23 个唯一 λ、末边 `0.100049→0`、`realized_max_lambda_gap=0.100049 ≤ max_lambda_gap_bound=0.151515`、窗口 `[(0,5),(4,8),(7,12),(11,16),(15,20),(19,23)]`、槽位 28、`sliding_overlap_states=0`。完整 λ 表与诊断见 [design/LAMBDA_SCHEDULE_CONTRACT.md](design/LAMBDA_SCHEDULE_CONTRACT.md)（已同步为 v21）。仍需在 OpenMM/CUDA 环境确认 v18/v19/v20 缓存对 v21 fail closed；验证矩阵 `VAL-TEST-004` 的描述也要一并从 v20 改到 v21。
- [ ] V-04：重建溶剂腿缓存并核对 `solvent_cache_manifest.json` 与 `topology_solvent.cif`：Na/Cl 均非零、目标浓度 0.15 M，随后确认旧溶剂腿 checkpoint 因 System 指纹变化被拒绝；GitHub [#34](https://github.com/Cedrus810/openmm_IBS_dev/issues/34)。

- [x] **V-05（历史原始窗口问题已由 immutable rescue 闭合；2026-07-27 复审更新）。** 下方 500k→1M 的表和判断保留为原始窗口历史证据，但“本轮刻意不拆窗/仍未闭合”已经不是当前状态。真实日志显示 2026-07-27 11:47，原窗口 3 仍未通过后，流程已自动建立两个 immutable rescue ensembles `[(11,14),(13,16)]`，仅排除原窗口 3，并于 12:12 完成 Stage 2；不是下方旧预案所写的同时排除窗口 2/3、建立四个 rescue ensembles。

  以当前源码从磁盘重新加载 original（显式 `excluded_local_windows={3}`）与 rescue 两套三文件数据和各自冻结 `f_k` 后重算，得到：

  - Stage 2 `ΔG = 145.9084716821 kJ/mol`，`total σ = 1.3844433223 kJ/mol`；
  - `converged=True`，`min_overlap=0.4851948395`，`min_occupancy_normalized=0.5272412124`；
  - `min_decorrelated_samples=52`，`max_endpoint_uncertainty_kJ_mol=0.9339715337`；
  - input/valid/solved 窗口一致，覆盖 λ 索引完整为 `0…22`，无 dropped window。

  重算值与 `output_lrc_fix/checkpoints/stage2_vanishing.json` 完全一致。当前复合物腿 `output_lrc_fix/final_results.json` 为 `145.5394904682 ± 1.6370635850 kJ/mol`；但溶剂腿 `output_lrc_fix/solvent_leg/final_results.json` 和最终 `final_binding_results.json` 尚不存在，所以“整条 ABFE 结合自由能链完成”仍未验收。P1-13 的 history 分支问题仍是真 bug，保留为出数后优先修复项。

- [ ] **V-06：endpoint_σ 到底能不能信（2026-07-27 新增）。** 上面的重算确认了数字本身可复现，但**没有回答那个 σ 是否可信**。两条独立的怀疑理由：

  1. **误差棒与实测漂移不自洽。** 总量从 07-26 六窗的 141.65（当时 total σ=2.91）变成 145.91 ± 1.38，**漂移 +4.26 kJ/mol ≈ 3× 新 total σ**。w0/w1/w4/w5 的落盘文件自 07-26 未变，漂移几乎全来自 w2（加采样）和 w3（换成 rescue）。关键是：两个 rescue ensemble 合起来覆盖 λ 索引 11–15，与原 w3 是**同一物理区间**——磁盘上已经躺着一次免费的独立重复测量，且原 w3 那 1M 步数据仍在（只是被 `excluded_local_windows` 排除，没被删）。
     *（注：本条早先草稿里"w2+w3 的 σ 只有 0.46、位移是它的 9 倍"是把 07-26 逐窗 σ 硬扣出来的**近似分解**，合并求解会重算 offset，该数不作数；以 total 口径的 ~3σ 为准。）*

  2. **渐近协方差在当前样本量下本就不成立。** 整条路径只有 pymbar 的渐近 `svd-ew` 协方差（`uncertainty_method` 从不传，见 `abfe_core._compute_free_energy_result_compatible`），而 pymbar 自己的 docstring 写着 "This will break down in cases where the number of samples is not large enough to reach the asymptotic normal limit."；`n_k=[N,0,…,0]` 意味着 K 个目标态的整个协方差都由同一批 N 个样本估出。实测 `output_lrc_fix/vanishing/dual_window_0_vdw_convergence.json` 里 `min_absolute_ess = 1.0000000000001` 却报出 `max_endpoint_uncertainty = 0.0618 kJ/mol`——**约 1 个独立样本撑起 0.06 kJ/mol 的精度声明**，这个数没有统计意义。整个 stage 的 `min_decorrelated_samples` 也只有 52。

  已新增 `diagnose_endpoint_sigma.py`（只读、不重采样、不碰 GPU、不写 `output_lrc_fix`）：A1 先复现 `145.90847 / 1.38444` 作为控制实验（对不上就停）；A2 用老 w3 vs 两个 rescue ensemble 对同一 λ 区间做重复测量对照，三种口径各算一次 z 值；顺带扫三份 history 的跳变，判断 P1-13 是否真在这批数据上发生过。

  验收：**z < 2** → 误差棒暂时可信，漂移主要来自 w2，按正常加采样推进；**z ≫ 2** → 报出去的 σ 不能用，需改用 bootstrap（pymbar 原生支持 `MBAR(..., n_bootstraps=K)` 与 `uncertainty_method="bootstrap"`；本仓库 IBS 路径目前**没有任何** bootstrap/blocking 机制）或 block SEM 作为门量，并重新标定 1.0 kJ/mol 阈值。

  ### 2026-07-27 13:13 实测结果（`/tmp/sigma_diag/endpoint_sigma_diagnosis.json`）

  **A1 复现成功**：`145.90847168207642 / 1.384443322336141`，`converged=True`。逐段明细（此前从未落盘）：

  | 来源窗口 | λ join→end | ΔG (kJ/mol) | σ (kJ/mol) |
  |---|---|---|---|
  | 0 | 0→4 | 76.2842 | 0.9340 |
  | 1 | 4→7 | 25.4415 | 0.8463 |
  | 2 | 7→11 | 17.6881 | 0.3256 |
  | rescue 0 | 11→13 | 5.8613 | 0.2496 |
  | rescue 1 | 13→15 | 5.1953 | 0.2126 |
  | 4 | 15→19 | 9.3916 | 0.3372 |
  | 5 | 19→22 | 6.0464 | 0.0315 |

  **A2 结论：z < 2，误差棒站得住。** 同一 λ 区间 11→15 的两个独立估计：老 w3（1M 步）`8.8779 ± 1.4934`，新 rescue（2×250k）`11.0566 ± 0.3279`，差 `2.1787`，段口径 z=1.42、全链口径 z=0.89。新估计 σ 小 4.5 倍是拆窗改善 overlap 的直接结果（rescue 两窗 `min_ess_ratio` 0.94/0.97，而老 w3 有个占据 0.098 < floor 的饿死态），物理上说得通。

  **漂移分解被纠正**：全链含老 w3 = `143.7298 ± 2.0098`，含 rescue = `145.9085 ± 1.3844`。所以 +4.26 里只有 **2.18 来自 w3 换 rescue**，另外约 2.08 来自 w2 从 500k 加到 1M（141.65→143.73）。两段都在各自误差棒内，不构成矛盾。

  **上面第 2 条（渐近协方差不成立）对最终结果不成立——引错了数。** `min_absolute_ess = 1.0000000000001` 出自 convergence.json 的 `last_tmbar_update`，那是 **warmup 期 TMBAR 的诊断**，不是生产求解。生产求解逐窗 `absolute_ess` 实测为 26.31 / 32.99 / 53.63 / 60.47 / 105.79 / 69.87 / 107.69，**全部 ≥ 10**。渐近极限对这批最终数据是成立的。（该顾虑对 warmup 期的 TMBAR trust 门仍然适用，但那是另一件事。）

  **轨迹不连续存在，但没有污染 g。** history 扫描在 `base` 序列扫到 21 处跳变（`bias` 一处都没有）：w1=4、w3=9、w4=4、rescue_w1=4；有跳变的窗口 `base` 相邻帧最大差 ~88,000 kJ/mol，无跳变的 ~4,000，差 20 倍，不是热涨落。**但真正拿去估 g 的 `(u_kn[k]-bias)/kT` 逐态扫描全部为 0 跳变**（decorr MAD 1.33–1.95 kT，最大相邻差 7.8–14.0 kT，各窗口一致）。说明 `U_k` 与 `base` 同步跳、在 `u_kn = U_k − base` 里抵消掉了 —— **g / N_decorr / 误差棒未受影响**。

  **这些跳变不是 P1-13。** `grep -c "触发回退\|灾难检测触发" output_lrc_fix/pipeline.log` = **0**，生产灾难回退那条路径（`sim.context.setPositions(pos_backup)`）在整条历史里从未触发过；而同一日志有 58 次进程启动。所以这 21 处不连续来自**跨进程续跑 / 窗口重建边界**（resume 时会重新最小化 + dt 测试步进 + Boresch 爬坡），与 P1-13 描述的机制不同，**修 P1-13 不会消除它们**。

  ### 结论（2026-07-27 收）

  **`ΔG_vdw = 145.90847 ± 1.38444 kJ/mol` 的误差棒可以信。** 三条怀疑理由逐一被否：
  重复测量 z=0.89 < 2；逐窗 `absolute_ess` 全部 ≥ 26（渐近极限成立）；轨迹不连续没有传到 g。
  141.65 → 145.91 的 +4.26 拆成 +2.08（w2 加采样）+ 2.18（w3 换 rescue），两段都在各自误差棒内。

  遗留（都不阻塞出数）：

  - [ ] **P1-13 降级**：仍是真 bug（回退路径确实不截断 history），但**在这批数据上从未触发**，不能用它解释任何 σ 行为。修它的理由只剩稳健性，优先级低于原先判断。
  - [ ] **新问题（低优先）：跨进程续跑/窗口重建会在生产序列里留下不连续，而这些片段仍被当作一条连续轨迹估自相关。** 本批因 base 项抵消而无害，但这是运气不是设计。脚本已记录 `base_jump_frame_indices`，重跑一次即可确认是否都落在会话边界。
  - [ ] `environment.yml:18` 的 `pymbar-core` **未锁版本**。本机 `openmm_dev` 解析到 4.0.3，`env3.txt:227`（另一台机器的 env dump）记的是 4.2.0。两版 `uncertainty_method` 语义相同（`None → "svd-ew"`），当前结论不受影响，但误差语义依赖一个不受约束的依赖，建议锁版本。

  以下为历史实测（Atenolol vdw，各原始窗口 500k 步累计时）：

  | 窗口 | λ_vdw 范围 | 段 ΔG (kJ/mol) | endpoint_σ (kJ/mol) |
  |---|---|---|---|
  | 0 | 1.000→0.737 | 76.28 | 0.934 |
  | 1 | 0.737→0.596 | 25.44 | 0.846 |
  | **2** | **0.596→0.442** | **12.77** | **1.905** |
  | **3** | **0.442→0.320** | **11.72** | **1.765** |
  | 4 | 0.320→0.196 | 9.39 | 0.337 |
  | 5 | 0.196→0.000 | 6.05 | 0.032 |

  合计 ΔG = 141.65 kJ/mol = 33.86 kcal/mol，total σ = 2.91 kJ/mol = **0.695 kcal/mol**。窗口 2+3 占总方差 **79.8%**，但即使把它们都压到阈值 1.0，total σ 也只从 0.695 降到 **0.460 kcal/mol**——只买到 0.24 kcal/mol。

  **当时决定：先出数、验证整条链路对不对，精修留到后面。** 下述收益/代价分析是进入 immutable rescue 前的历史判断；当前实际控制流已经执行 rescue 并通过。

  **⚠️ 2026-07-27 实测：方案 1 的 1/√N 前提被数据否掉了。** 同一个 `worst_state=11`，`output_lrc_fix/pipeline.log`：

  | 时间 | 累计步数 | N_decorrelated | endpoint_σ (kJ/mol) |
  |---|---|---|---|
  | 07-26 19:43 | 500k | 49 | 1.883 |
  | 07-27 11:20 | 1M | 142 | **1.897** |

  有效样本数涨了约 2.9 倍，σ **纹丝不动**（1/√N 应给 1.10）。所以"再翻两番到 4M"大概率买不到精度。**在解释清楚 σ 为何对 N 不响应之前，方案 1 和方案 2 都是在赌。** 一个待排除的候选成因是 P1-13：灾难回退不截断三份 history，被丢弃分支与重启分支被当成一条连续轨迹估自相关，`g` 的口径本身可能有问题（`_decorrelate_by_worst_target_state`，`ibs_engine.py:11046`）。建议出数之后先做 P1-13，再谈精修。

  另注：rescue 轮次计数**每个进程从 1 重新开始**（`abfe_pipeline.py:7119-7125`），所以每次 resume 都会再给两轮；轮次用尽后才走 `_build_vanishing_rescue_ranges`。

  当时评估的两条可选路径（历史记录）：

  1. **加 doubling 轮次**（不丢数据，但见上方实测）：传 `stage2_production_rescue_rounds=4`（默认 2），配合默认 `stage2_production_rescue_growth=2.0`，累计步数 250k→500k→1M→2M→4M。原按 1/√N 外推预测 2M 时 w2≈0.94/w3≈0.88、4M 时 w2≈0.67/w3≈0.62——**该外推已被 500k→1M 的实测证伪**。w2 的 g=20.6，自相关很长。
  2. **拆窗桥接**（默认 fallback，**会丢数据**）：rescue 轮次用尽后代码会走 `_build_vanishing_rescue_ranges`，把 w2=[7,12) 拆成 (7,10)+(9,12)、w3=[11,16) 拆成 (11,14)+(13,16)，在 `vanishing_rescue/<plan_id>/` 新建 4 个 3 态 ensemble，各自重新预热并锁自己的 f_k。合并时 `excluded_local_windows=set(failing_windows)` 会把 w2/w3 已投入的 1M 步生产数据整个排除。λ 跨度对半确实同时治"跨度宽"这个成因，但 `rescue_steps` 默认回落到 `n_steps_per_window`(250k)，跨度收益能否盖住步数减少尚未标定。**为 0.24 kcal/mol 丢弃 1M 步已采数据并重新预热，不划算。**

  验收已满足：`max_endpoint_uncertainty_kJ_mol ≤ 1.0` 且 `converged=True`，另外三个门（`min_overlap ≥ 0.05`、`min_occupancy_normalized ≥ 0.5`、`min_decorrelated_samples ≥ 20`）全部通过、`min_absolute_ess_threshold` 仍为 `None`。

## P2 工程工作

- [ ] E-01：production ESS 自动修复的第二调用点迁移到 per-state trajectory bank；对应 GitHub `openmm_IBS_dev#1`。
- [ ] E-02：为 pilot 热力学长度探测设计安全的长探针重测机制；对应 GitHub `openmm_IBS_dev#29`。不得复活已废弃的 fixed-H adjacent-overlap 自动变异环。
- [ ] E-03：把 `_run_stage_with_overlap_autorepair` 中早退之后约 900 行不可达旧变异逻辑移出生产类；保留历史可读性时放入归档文档，不得保留可被误激活的可执行代码；GitHub [#35](https://github.com/Cedrus810/openmm_IBS_dev/issues/35)。
- [ ] E-04：为膜/各向异性盒实现并验证适用的有限尺寸静电修正；当前 `apbs_correction.py` 对纵横比大于 1.10 的盒子应继续 fail closed；GitHub [#36](https://github.com/Cedrus810/openmm_IBS_dev/issues/36)。
- [x] **P2-13：LSE 默认容差已统一为 0.5。** engine、pipeline、complex/solvent CLI 调用路径和 `abfe_config.json` 不再显式覆盖为 0.25。
- [x] **P2-14：`final_results.json` 已如实报告 LJ LRC（2026-07-27）。** 原来 `abfe_pipeline.py:3327` 的 `"applied": True` 是**裸字面量**，整个 dict literal 无任何分支，后面 ~30 行 `note` 还用散文再断言一遍；而 `ibs_engine.py:2224-2226` 对 DEXP 明确不附加，`None` 一路 short-circuit 到零。

  修法**刻意不是**在报告侧另写一个 `potential_type != "dexp"`（两处会分叉）。新增共享谓词 `ibs_engine.ibs_lj_tail_lrc_is_applicable()`，生产者 `build_ibs_dual_system` 与报告者 `compute_final_results` 共用——DEXP 的解析尾项公式一旦被验证/替换，只需改这一处，行为和报告一起动。传统 REMD 路径另有自己的 `lj_lrc_metadata`（`ibs_engine.py:12973`），它才是那条路径的真相，优先级更高。新增字段 `applicable`/`potential_type`/`truth_source`/`not_applied_reason`，`status` 与 `note` 均改条件文案。

  **注：本条虽列在 DEXP 缓期组（下方 2026-07-26 条目）里，但它是报告诚实性，与将来 DEXP 怎么实现无关，故独立先落。** 用户的新 DEXP 已在 `dexp_experiment.py` 改过、尚未并入主项目；届时若它开始附加 LRC，只需改上述谓词。回归见 `test_lrc_reporting_honesty.py`。

- [ ] **P2-15：单阶段 endpoint diagnostics 使用了整条双阶段路径的判据（2026-07-27 复审新增；出数后修）。** 当前 `lambda_endpoint_diagnostics()` 的 `ok` 同时要求“起点 fully coupled”与“终点 fully decoupled”，但它分别被用于 Stage 1（`(1,1)→(0,1)`）和 Stage 2（`(0,1)→(0,0)`）；两个合法半程因此都会在 `final_results.json` 中报告 `ok=false`。这不会改变 ΔG 数值，但会把正常路径标成失败，误导自动审计。应为单阶段报告各自的预期端点/固定 λ 不变量，整条双阶段闭合另设组合诊断。

## P2/P3 科学与稳健性评估

- [ ] R-01：量化 Boresch `k(1-cos(delta))` 角度/二面角势相对纯谐波解析释放公式的非谐性误差；`GeometricRestraintEstimator` 当前又以纯谐波关系 `kBT/var` 估力常数，而现有 harmonicity 诊断只报警、不阻断。需用解析积分或数值积分给出误差界，并基于实际涨落宽度确定可审计的 hard gate；“涨落 >15°”和“偏差 1–3 kJ/mol”目前没有被本仓库证据定量证实，不能先把它们写成硬阈值，也不能在有依据前直接改生产公式；GitHub [#37](https://github.com/Cedrus810/openmm_IBS_dev/issues/37)。
- [ ] R-02：用真实 bridge 数据重新标定 Shadow-Coulomb 的窗口数和 overlap 阈值；GitHub [#38](https://github.com/Cedrus810/openmm_IBS_dev/issues/38)。
- [ ] R-03：评估 `refine_stage_lambda_path_by_overlap` 的点 ESS 插点启发式；它目前不在 `non_mutating_v1` 生产路径上；GitHub [#39](https://github.com/Cedrus810/openmm_IBS_dev/issues/39)。
- [ ] R-04：评估 ACE softcore 分母 `1e-6` floor、pilot metric 最小样本数、PBC 重居中对 Boresch 几何的影响；GitHub [#40](https://github.com/Cedrus810/openmm_IBS_dev/issues/40)。
- [ ] R-05：统一 warning/stdout 日志，并评估 DEXP 多随机种子优化；均不改变当前数值定义；GitHub [#41](https://github.com/Cedrus810/openmm_IBS_dev/issues/41)。

## 复核关闭（不再作为待办）

- [x] A-05：`gauss_coul` 在当前 `dexp_experiment.py` 中按 fit mode 参与 `delta_gauss_replacement`，并非“构建后从未使用”；旧报告已过时。
- [x] A-07：裁剪后的 Boresch 力常数同时用于实际施力和解析修正；原始轨迹只是估计器输入，旧报告所称“施力与修正使用不同 k”不成立。裁剪仍会留下显式诊断。
- [x] A-12：各向异性 APBS 目前是明确不支持并 fail closed，不是静默错误；功能扩展保留为 E-04。
- [x] A-14/A-17/A-19：属于开发期缓存语义、数值敏感性或日志体验，不是已确认的生产数值缺陷；归入 R-04/R-05 或不再单列。
- [x] 2026-07-26 新清单中的其余六项未形成新的生产缺陷：传统 Beutler LRC 对固定盒 NVT 可作为逐态常数离线补入，体积明显变化时现有代码已 fail closed；`addSolvent(padding=1.5 nm)` 的最小立方盒边长为 3.0 nm，不会因“小配体”低于 2×1.2 nm cutoff；主窗口与 production checkpoint manifest 均已绑定 `lambda_shield`；vanishing 度规插值前后均用运行时有限性/严格单调检查，不是仅开发期 `assert`；base-energy 连续失败计数在每次成功查询时重置，清空 `energy_buffer` 不应打断“连续查询”语义；旧窗口 remap 逻辑虽未比较目标步数，但当前 `non_mutating_v1` 生产路径不可达，且 `run_all_windows` 的独立 resume gate 会拒绝 `n_steps_per_window_effective` 低于当前预算的缓存。

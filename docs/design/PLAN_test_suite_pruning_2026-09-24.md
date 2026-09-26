# 测试集压缩计划（2026-09-24）

[设计文档索引](README.md) · [待办 `TEST-02`](../TODO.md)

## 0. 状态

| 部分 | 状态 |
|---|---|
| 提速（§1） | ✅ **已落地** 2026-09-24：全量 263 s → 40 s |
| 删除 EXP-012/019/020 测试 | ✅ **已落地**：11 个文件 / 70 条（用户点名；`exp012_xed/` 模块本身和 `tests/fixtures/output/outer_lambda_exp012/` 没动，后者 `exp012_xed/` 自己在用） |
| 压条数 2810 → ~1300（§2–§4） | ⬜ **押后到重构**，一条没删 |

**为什么押后**：
1. 要重构。重构会大面积改被测函数的签名/位置，现在逐条删并、重构时再改一遍是白做；而且下面的分类是按**当前**代码的覆盖行算的，重构后必须重算（§5 给了命令）。
2. `tests/` 里大量文件**有未提交改动或根本没入库**（09-2x 的日期文件）。删了在 git 里找不回来——09-24 第一次尝试时权限分类器就是按「不可逆本地破坏」拦下的。**动手前先提交或打快照**。

## 1. 已落地的提速

| 改动 | 位置 | 效果 |
|---|---|---|
| 默认 xdist 并行（16 worker，`--dist loadfile`） | `tests/run_offline_tests.sh` | 主因。串行/并行结果逐条一致已核。调用方给了 `-n`/`-s`/`--pdb`/`-p no:xdist` 时自动串行；没装 xdist 退回串行；`PYTEST_WORKERS` 覆盖 worker 数。16 与 32 墙钟差不多（每个 worker 各自 import 一遍 openmm/torch，收集约 10 s/worker） |
| `OMP/MKL/OPENBLAS_NUM_THREADS` 默认 1 | `tests/conftest.py` | 测试张量都很小，80 线程反而慢一倍；并行时还超订。只 `setdefault` |
| 只扫入库 `.py` | `tests/test_no_personal_paths_as_defaults.py` | 26 s → 3 s（原来 `rglob` 走进 `refactor/` 的 683 个未入库 .py 和 `output/`、`tmp/`） |
| `addopts` 加 `-m "not needs_gpu"` | `pytest.ini` | 原来只有脚本排除，裸跑 `pytest` 会多跑 44 条 GPU 测试（加载真实 MACE 那条 23 s）。GPU 节点用 `pytest -m needs_gpu` |

没放进 `pytest.ini` 的：`-n`。CI 环境（`environment-ci.yml`）没有 xdist，写进去 CI 直接报 unknown arg；单条调试时 `-s`/pdb 在 xdist 下也不好用。
裸跑 `pytest` 仍是串行（~4 分钟），要快用脚本或 `pytest -n 16`。

现状（09-24 删 EXP 之后）：**2810 条收集 / 2796 passed / 8 skipped / 6 xfailed**，217 个文件。

## 2. 度量方法

逐条测试的**生产代码行覆盖**（`--cov-context=test`，只量仓库根的生产模块、`local_residual/`、`tools/` 等，排除 `tests/ refactor/ output/ tmp/ archive/ docs/`），然后：

- **全局贪心最小覆盖集**：**965 条**就能保住 100% 的生产行覆盖（27,416 行）。
- **逐文件最小覆盖集**（`keep_cover`）：每个文件内挑出能盖住该文件全部覆盖行的最少测试，合计 **1235 条**。按文件而不是全局做，是为了主题内自洽、合并文件时不跨主题找依赖。
- **零生产行**（`zero_line`）：**503 条**一行生产代码都不跑——绝大多数是 `inspect.getsource`/`read_text` 读源码字符串断言，另有文档检查和**子进程测试**（覆盖率量不到子进程，这类不能按零行处理）。
- **文件内冗余**（`redundant_in_file`）：**1072 条**，覆盖行是本文件 `keep_cover` 的子集。

⚠️ **行覆盖是子集 ≠ 断言冗余。** 同一段代码喂不同输入、断言不同数值（符号约定、边界 raise）行覆盖完全一样。所以冗余清单只是**候选**，要逐条判（§3），不能按清单机械删。

## 3. 判定规则

**`keep_cover`**：必须保留。可改名、可往里并断言、可挪进合并后的文件，但函数体执行的生产代码不能变少。

**`redundant_in_file`**：
- 与某条保留测试是**同一场景的变体/重复** → 删；
- 断言了保留测试没有的**独特且有意义的东西**（符号约定、具体数值、某输入必须 raise/fail-closed）→ 把那几行断言并进一条保留测试（或写成保留测试里的小循环），删原函数；
- parametrize 展开的多条能改成函数体内循环的就改（条数降、断言不丢）。

**`zero_line`**：
- **留**：全仓/全模块 AST 扫描类 lint（`no_undefined_globals`、bare except、个人路径、fresh clone import）、真起子进程跑生产代码的行为测试、文档过期契约（`doc_staleness_contract`、`ci_quality_contract`）；
- **删**：只盯某一个函数里「必须/不许出现某字符串/某个名字」的窄源码断言（重构时只会误报，行为由别的测试守）；守「已删除代码不许回来」的（如 `att27_dead_code_removed`）；
- 拿不准的：多个窄断言并成一条。

**合并文件**：同主题小文件并成一个（`test_boresch.py` 这种主题名）；日期后缀的一次性回归文件优先并进主题文件；模块 docstring 的历史叙述压到 1–3 行（历史在 CHANGELOG）。

**硬约束**：
- 不改生产代码；不为变绿放宽断言；strict-xfail 原样保留（`COMBINE-01` 的 6 条）。
- 下列模块被别的测试 `import`，被 import 的名字必须在同名模块里继续存在（可以把别的文件并**进**它，不能把它并走/改名）：
  `test_stage2_repair_controller`（**28 个文件**在 import）、`test_rewindow_sampling_units`（4）、`test_resume_reuse_contracts`（4）、
  `test_preopt_protocol_scope`（2）、`production_artifact_fixtures`（3）、`test_rbfe_hybrid_r1b`、`test_core_physics_numerics`、
  `test_top_level_sampling_cache_identity`、`test_boresch_attachment_leg`、`test_membrane_observable_extractor`、
  `test_pme_decharge_endpoint_equivalence`、`test_join_lambda_two_sided_support`、`test_stage2_subwindow_terminal_retirement_2026_09_18`（各 1）。
  重构时重新生成这张表（§5）。
- 名字带 `repro` / `coords_sidecar` 的属于当前复现主线，谨慎。

**验收**：压完后全量绿、xfail/skip 数不变；再量一次覆盖率，生产行覆盖数不低于压前（重构后的新基线）。

## 4. 09-24 分类快照

按主题切 8 批（切批是为了并行派代理，每批 21–32 个文件；控制器与核心按候选量均分成 2/3 份）。
**数字是 09-24 代码的**，重构后作废，只用来估量级。

| 批次 | 主题 | 文件 | 现有条数 | 覆盖必留 | 零生产行 | 文件内冗余 |
|---|---|---|---|---|---|---|
| b1 | RBFE / REMD / Boresch / split-half | 21 | 376 | 173 | 33 | 170 |
| b2 | 膜线 / charge transfer / coion / 充放电 | 22 | 438 | 201 | 43 | 194 |
| b3 | local residual / outer-λ / 仓库卫生与源码契约 | 32 | 389 | 157 | 141 | 91 |
| b4 | Stage-2 控制器（前半） | 28 | 286 | 139 | 62 | 85 |
| b5 | Stage-2 控制器（后半） | 30 | 251 | 106 | 56 | 89 |
| b6 | 核心（物理/IBS/缓存/resume/预平衡）其一 | 28 | 339 | 135 | 54 | 150 |
| b7 | 核心 其二 | 28 | 375 | 171 | 58 | 146 |
| b8 | 核心 其三 | 28 | 356 | 153 | 56 | 147 |


#### b1 · RBFE / REMD / Boresch / split-half

| 文件 | 必留 | 零行 | 冗余 |
|---|---|---|---|
| `test_rbfe_hybrid_r1b.py` | 18 | 0 | 30 |
| `test_rbfe_mapping_r1.py` | 29 | 1 | 25 |
| `test_rbfe_pipeline_framework.py` | 24 | 2 | 20 |
| `test_boresch_committed_gate.py` | 6 | 2 | 18 |
| `test_rbfe_core_r0.py` | 27 | 3 | 17 |
| `test_boresch_anchor_and_pbc_fixes.py` | 10 | 8 | 7 |
| `test_remd_backend_cli_wiring.py` | 3 | 3 | 7 |
| `test_runrbfe_cli.py` | 10 | 2 | 7 |
| `test_boresch_openfe_anchor_criteria_2026_09_22.py` | 5 | 3 | 5 |
| `test_boresch_dihedral_convention.py` | 1 | 0 | 7 |
| `test_rbfe_analysis_r2.py` | 14 | 0 | 7 |
| `test_boresch_attachment_leg.py` | 7 | 0 | 6 |
| `test_remd_gpu_context_budget.py` | 1 | 2 | 4 |
| `test_split_half_and_merged_path_2026_09_20.py` | 1 | 4 | 0 |
| `test_boresch_json_serialization.py` | 3 | 1 | 2 |
| `test_boresch_minimum_image_bor01.py` | 2 | 0 | 3 |
| `test_split_half_reports_the_total_2026_09_15.py` | 1 | 1 | 2 |
| `test_remd_dcd_append_mode.py` | 3 | 1 | 1 |
| `test_split_half_uses_the_intersection_2026_09_18.py` | 3 | 0 | 2 |
| `test_repro_budget_and_attachment_gate_2026_09_23.py` | 2 | 0 | 0 |
| `test_split_half_mainpath_and_final_state_atomic_2026_09_24.py` | 3 | 0 | 0 |

#### b2 · 膜线 / charge transfer / coion / 充放电

| 文件 | 必留 | 零行 | 冗余 |
|---|---|---|---|
| `test_membrane_input_and_quality_gate.py` | 24 | 0 | 46 |
| `test_membrane_observable_extractor.py` | 18 | 1 | 31 |
| `test_membrane_barostat_protocol.py` | 23 | 12 | 11 |
| `test_charge_transfer_hamiltonian.py` | 7 | 2 | 19 |
| `test_charge_transfer_real_endpoints.py` | 16 | 0 | 20 |
| `test_charge_treatment_protocol.py` | 22 | 0 | 15 |
| `test_coalchemical_ion_identity.py` | 7 | 7 | 6 |
| `test_c2_lipid_slab_validation.py` | 20 | 1 | 11 |
| `test_decharging_seed_contract.py` | 4 | 8 | 4 |
| `test_complex_leg_reserved_coion_2026_09_16.py` | 8 | 0 | 9 |
| `test_pme_decharge_endpoint_equivalence.py` | 1 | 1 | 6 |
| `test_charging_estimator_protocol.py` | 8 | 1 | 5 |
| `test_charging_only_mode.py` | 4 | 4 | 1 |
| `test_charge_transfer_admission_override_2026_09_18.py` | 8 | 1 | 3 |
| `test_block_gate_covers_all_charging_actions_2026_09_18.py` | 2 | 2 | 1 |
| `test_solvent_leg_coion_builder.py` | 4 | 0 | 3 |
| `test_charge_transfer_production_qualification.py` | 9 | 0 | 2 |
| `test_coion_cache_resume_provenance.py` | 8 | 1 | 1 |
| `test_membrane_gate_first_run_regressions.py` | 2 | 1 | 0 |
| `test_runtime_gromacs_top_after_coion_derivation_2026_09_18.py` | 3 | 1 | 0 |
| `test_membrane_barostat_context_smoke.py` | 2 | 0 | 0 |
| `test_preopt_probe_coion_hamiltonian.py` | 1 | 0 | 0 |

#### b3 · local residual / outer-λ / 仓库卫生与源码契约

| 文件 | 必留 | 零行 | 冗余 |
|---|---|---|---|
| `test_audit_protocol_regressions.py` | 9 | 43 | 6 |
| `test_p1_completion_and_ensemble_regressions.py` | 16 | 0 | 23 |
| `test_0831issue_p2_batch.py` | 8 | 20 | 0 |
| `test_cli_diagnostics_and_lazy_imports.py` | 15 | 7 | 7 |
| `test_release_readiness_r1_r2.py` | 14 | 3 | 9 |
| `test_import_time_side_effects.py` | 1 | 9 | 0 |
| `test_retrain_offline_frame_source.py` | 2 | 1 | 8 |
| `test_att27_dead_code_removed.py` | 0 | 8 | 0 |
| `test_local_residual_refit.py` | 5 | 0 | 8 |
| `test_outer_lambda_refit_wiring.py` | 1 | 7 | 0 |
| `test_issues_67_27_75_83.py` | 22 | 0 | 6 |
| `test_no_phantom_view_keys.py` | 4 | 1 | 5 |
| `test_outer_lambda_cli.py` | 0 | 6 | 0 |
| `test_outer_lambda_controller.py` | 9 | 0 | 6 |
| `test_local_residual_autofit.py` | 15 | 3 | 2 |
| `test_lr06_residual_distance_clamp_2026_09_16.py` | 0 | 5 | 0 |
| `test_only_ligand_in_water_lr04.py` | 0 | 5 | 0 |
| `test_outer_lambda_torchforce_standalone.py` | 3 | 1 | 4 |
| `test_high_medium_fixes_2026_09_20.py` | 3 | 4 | 0 |
| `test_ci_quality_contract.py` | 0 | 3 | 0 |
| `test_doc_staleness_contract.py` | 1 | 2 | 1 |
| `test_open_issue_fail_closed_contracts.py` | 5 | 2 | 1 |
| `test_outer_lambda_local_residual_runtime.py` | 4 | 3 | 0 |
| `test_todo_verified_fixes.py` | 11 | 0 | 3 |
| `test_fresh_clone_imports.py` | 0 | 2 | 0 |
| `test_mace_decomposition_shared_minimum_image.py` | 1 | 0 | 2 |
| `test_open_issue_protocol_contracts.py` | 2 | 2 | 0 |
| `test_core_runabfe_bugfix_batch.py` | 4 | 1 | 0 |
| `test_no_personal_paths_as_defaults.py` | 0 | 1 | 0 |
| `test_no_undefined_globals.py` | 0 | 1 | 0 |
| `test_outer_lambda_existing_api_compat.py` | 2 | 1 | 0 |
| `test_outer_lambda_torchforce_gpu_standalone.py` | 0 | 0 | 0 |

#### b4 · Stage-2 控制器（前半）

| 文件 | 必留 | 零行 | 冗余 |
|---|---|---|---|
| `test_stage2_repair_controller.py` | 26 | 3 | 29 |
| `test_validation_indeterminate_routing.py` | 6 | 9 | 4 |
| `test_stage2_retire_and_result_status_2026_09_16.py` | 11 | 2 | 9 |
| `test_path_repair_model_b.py` | 15 | 5 | 4 |
| `test_stage2_judge_executor_divergence_2026_09_15.py` | 7 | 2 | 7 |
| `test_platform_routing_regressions.py` | 4 | 6 | 2 |
| `test_stage2_controller_vocabulary.py` | 1 | 7 | 1 |
| `test_engine_signal_registry_2026_09_18.py` | 1 | 5 | 0 |
| `test_routing_signal_not_terminal.py` | 5 | 4 | 1 |
| `test_stage2_states_not_contaminated_by_rejected_cache.py` | 0 | 5 | 0 |
| `test_ctl09_decorrelation_cross_check.py` | 4 | 0 | 4 |
| `test_layout_change_does_not_crash_the_loop_2026_09_16.py` | 4 | 1 | 3 |
| `test_worst_window_handles_per_state_lists_2026_09_15.py` | 2 | 1 | 3 |
| `test_bottleneck_g_eta_ledger_2026_09_17.py` | 3 | 0 | 3 |
| `test_ctl10_warmup_budget_authority.py` | 2 | 0 | 3 |
| `test_no_window_signal_is_silently_dropped_2026_09_17.py` | 7 | 0 | 3 |
| `test_stage2_repro_frozen_path_2026_09_22.py` | 14 | 0 | 3 |
| `test_unknown_parent_window_is_not_window_zero_2026_09_17.py` | 0 | 3 | 0 |
| `test_ctl0467_controller_misroutes.py` | 4 | 1 | 1 |
| `test_ctl08_skipped_window_unmeasured.py` | 1 | 2 | 0 |
| `test_executor_never_declines_silently.py` | 0 | 2 | 0 |
| `test_indeterminate_window_does_not_kill_the_run_2026_09_16.py` | 6 | 0 | 2 |
| `test_skew_vs_sample_size_attribution_2026_09_15.py` | 5 | 2 | 0 |
| `test_stage2_subwindow_terminal_retirement_2026_09_18.py` | 4 | 1 | 1 |
| `test_local_validation_cap_question_2026_09_17.py` | 2 | 0 | 1 |
| `test_skipped_window_outranks_earliest_2026_09_15.py` | 2 | 1 | 0 |
| `test_stale_layout_window_can_only_resample.py` | 1 | 0 | 1 |
| `test_ctl03_replaced_parent_does_not_decide.py` | 2 | 0 | 0 |

#### b5 · Stage-2 控制器（后半）

| 文件 | 必留 | 零行 | 冗余 |
|---|---|---|---|
| `test_tail_window_cap_2026_09_18.py` | 1 | 2 | 16 |
| `test_audit_2026_09_14_controller_budget.py` | 22 | 3 | 14 |
| `test_stage2_autonomous_segment_routing.py` | 2 | 8 | 4 |
| `test_eta_lever_bound_2026_09_18.py` | 5 | 0 | 9 |
| `test_stage2_control_chain_audit_2026_09_15.py` | 4 | 4 | 5 |
| `test_exit_scope_contract_2026_09_17.py` | 7 | 5 | 3 |
| `test_provisional_production_2026_09_15.py` | 4 | 8 | 0 |
| `test_failed_window_lambda_insertion.py` | 3 | 2 | 5 |
| `test_path_evolution_loop.py` | 8 | 2 | 5 |
| `test_stage2_controller_module_boundary.py` | 1 | 5 | 0 |
| `test_tail_window_overflow_and_legalization.py` | 5 | 0 | 5 |
| `test_ctl11_split_anchor_and_insert_budget.py` | 4 | 0 | 4 |
| `test_structural_action_budget_2026_09_17.py` | 5 | 1 | 3 |
| `test_autonomous_controller_toggle_is_wired_2026_09_16.py` | 0 | 3 | 0 |
| `test_ctl02_scheduling_vs_attribution.py` | 2 | 1 | 2 |
| `test_lrc_pocket_target_met_2026_09_15.py` | 3 | 0 | 3 |
| `test_rewindow_sampling_units.py` | 4 | 0 | 3 |
| `test_stall_escalation_passes_the_budget_gates_2026_09_17.py` | 0 | 3 | 0 |
| `test_budget_wiring_and_final_analyze_2026_09_15.py` | 2 | 2 | 0 |
| `test_ctl01_stage_result_authority.py` | 2 | 1 | 1 |
| `test_ctl05_ledger_path_version_isolation.py` | 1 | 0 | 2 |
| `test_epoch_unaffordable_does_not_degrade_skew_2026_09_15.py` | 1 | 1 | 1 |
| `test_identity_recovery_and_physical_window_skew.py` | 4 | 1 | 1 |
| `test_single_run_can_finish_2026_09_15.py` | 6 | 1 | 1 |
| `test_stage2_refinement_restart.py` | 0 | 2 | 0 |
| `test_ctl04_rewindow_intent_ledger.py` | 5 | 0 | 1 |
| `test_routing_context_reaches_every_exit_2026_09_17.py` | 1 | 1 | 0 |
| `test_stage2_coords_sidecar_2026_09_24.py` | 1 | 0 | 1 |
| `test_analyze_is_not_gated_by_the_self_check_2026_09_16.py` | 2 | 0 | 0 |
| `test_stage2_single_controller_ownership.py` | 1 | 0 | 0 |

#### b6 · 核心（物理/IBS/缓存/resume/预平衡）其一

| 文件 | 必留 | 零行 | 冗余 |
|---|---|---|---|
| `test_resume_reuse_contracts.py` | 20 | 3 | 50 |
| `test_gromacs_composition.py` | 10 | 1 | 16 |
| `test_fk_relearn_and_reachability.py` | 0 | 14 | 0 |
| `test_metric_integral_partition.py` | 5 | 7 | 6 |
| `test_preopt_cache_two_layer_split.py` | 8 | 1 | 11 |
| `test_free_energy_engine_sampling_contract.py` | 25 | 1 | 10 |
| `test_lrc_reporting_honesty.py` | 3 | 2 | 8 |
| `test_ligand_conformer_gate.py` | 6 | 2 | 6 |
| `test_lambda_path_versions.py` | 10 | 0 | 7 |
| `test_pocket_projection_angle_gradients.py` | 1 | 1 | 6 |
| `test_no_ibs_only_vanishing_mainvalue.py` | 0 | 6 | 0 |
| `test_bake_global_parameter.py` | 9 | 0 | 5 |
| `test_ligand_com_restraint_removed.py` | 1 | 4 | 1 |
| `test_s2a_per_segment_fk_gate.py` | 4 | 1 | 4 |
| `test_ess_gate_sampling_gauge.py` | 5 | 1 | 3 |
| `test_mbar_convergence_gate_regressions.py` | 4 | 2 | 2 |
| `test_vanishing_window_partition_regressions.py` | 5 | 0 | 4 |
| `test_format_results_human_leg_label.py` | 3 | 0 | 3 |
| `test_preopt_protocol_scope.py` | 3 | 0 | 3 |
| `test_warmup_guard_asymmetry_2026_09_16.py` | 1 | 3 | 0 |
| `test_endpoint_switch_wiring_2026_09_22.py` | 1 | 2 | 0 |
| `test_exclusion_order_is_ascending.py` | 2 | 0 | 2 |
| `test_partial_convergence_is_not_a_policy_mismatch.py` | 2 | 2 | 0 |
| `test_amber_topology_loading.py` | 3 | 0 | 1 |
| `test_exp030_ess_diagnostic_distinction.py` | 1 | 0 | 1 |
| `test_pymbar_version_and_uncertainty_contract.py` | 0 | 1 | 0 |
| `test_tmbar_entry_uses_unoffset_energies.py` | 2 | 0 | 1 |
| `test_solvent_leg_torsions_from_top_2026_09_23.py` | 1 | 0 | 0 |

#### b7 · 核心 其二

| 文件 | 必留 | 零行 | 冗余 |
|---|---|---|---|
| `test_core_physics_numerics.py` | 36 | 0 | 47 |
| `test_dexp_new_production.py` | 7 | 0 | 17 |
| `test_gromacs_pairs_funct2_conversion.py` | 6 | 4 | 13 |
| `test_fk_recalibration_from_production.py` | 4 | 11 | 2 |
| `test_thermodynamic_cycle.py` | 1 | 8 | 5 |
| `test_dispersion_and_forcefield_protocol.py` | 20 | 4 | 7 |
| `test_dispersion_protocol_is_honored.py` | 5 | 7 | 3 |
| `test_stage_diagnostics_persistence.py` | 11 | 4 | 5 |
| `test_water_model_identification.py` | 5 | 0 | 8 |
| `test_checkpoint_platform_migration.py` | 3 | 2 | 4 |
| `test_input_file_fail_closed.py` | 7 | 1 | 5 |
| `test_run_directory_guard_att23.py` | 4 | 1 | 5 |
| `test_probe_and_hysteresis_limits.py` | 2 | 5 | 0 |
| `test_step_guard.py` | 11 | 0 | 5 |
| `test_frames_admission_and_stop.py` | 4 | 2 | 2 |
| `test_multi_segment_analysis.py` | 4 | 0 | 4 |
| `test_analyze_only_and_refine_regressions.py` | 5 | 3 | 0 |
| `test_decorr01_same_input_comparison.py` | 3 | 1 | 2 |
| `test_intramolecular_coulomb_is_lambda_independent.py` | 1 | 0 | 3 |
| `test_protocol_mismatch_summary_is_bounded.py` | 3 | 0 | 3 |
| `test_wet_dry_gate_is_conditional.py` | 4 | 2 | 1 |
| `test_generate_c3_summary_reports.py` | 1 | 2 | 0 |
| `test_partial_sum_can_never_become_delta_G.py` | 3 | 1 | 1 |
| `test_equilibrium_is_done_strict_validation.py` | 5 | 0 | 1 |
| `test_non_mutating_policy.py` | 9 | 0 | 1 |
| `test_shadow_coul_no_double_counted_ligand_internals.py` | 2 | 0 | 1 |
| `test_top_level_sampling_cache_identity.py` | 3 | 0 | 1 |
| `test_traditional_pipeline_interfaces.py` | 2 | 0 | 0 |

#### b8 · 核心 其三

| 文件 | 必留 | 零行 | 冗余 |
|---|---|---|---|
| `test_independent_endpoint_sampling.py` | 20 | 7 | 32 |
| `test_free_energy_engine_backend_selection.py` | 12 | 2 | 27 |
| `test_warmup_overlap_protocol.py` | 39 | 5 | 12 |
| `test_stage_coverage_fail_closed.py` | 3 | 0 | 13 |
| `test_wet_cavity_seed_cache.py` | 4 | 1 | 11 |
| `test_join_lambda_two_sided_support.py` | 12 | 8 | 3 |
| `test_sampler_identity_is_stamped.py` | 0 | 9 | 0 |
| `test_untrusted_switch_is_not_identity.py` | 3 | 4 | 4 |
| `test_physical_input_preflight_att24.py` | 5 | 0 | 7 |
| `test_final_result_schema_gate.py` | 4 | 2 | 4 |
| `test_lj_tail_sigma_moments_bit_identical.py` | 1 | 0 | 6 |
| `test_traditional_mode_preflight_and_equil.py` | 4 | 4 | 2 |
| `test_repro_fk_tilt_and_full_frame_2026_09_23.py` | 3 | 4 | 1 |
| `test_ab_report_aligns_by_lambda_2026_09_16.py` | 3 | 0 | 4 |
| `test_em_noresidual_patch_contract.py` | 3 | 0 | 4 |
| `test_leg_constraint_identity_tolerance.py` | 3 | 1 | 3 |
| `test_target_support_gate.py` | 5 | 1 | 3 |
| `test_element_coverage.py` | 3 | 3 | 0 |
| `test_nvcc_discovery_and_compile_failure_2026_09_15.py` | 3 | 0 | 3 |
| `test_shadow_coul_ibs_builder.py` | 2 | 0 | 3 |
| `test_analysis_f_k_is_the_production_entry_one.py` | 4 | 0 | 2 |
| `test_ess_gate_not_measured_on_weight_thinned_frames.py` | 2 | 1 | 1 |
| `test_ligand_xml_extraction.py` | 3 | 2 | 0 |
| `test_pre_equil_final_state_box_2026_09_23.py` | 4 | 1 | 1 |
| `test_pre_equil_pose_drift_is_block_averaged_2026_09_22.py` | 1 | 0 | 1 |
| `test_stage_range_kwargs_signature_2026_09_19.py` | 0 | 1 | 0 |
| `test_exp030_mainline_runtime_contracts.py` | 4 | 0 | 0 |
| `test_pipeline_pre_equilibration_regressions.py` | 3 | 0 | 0 |

## 5. 重算命令

覆盖率工具**不要装进 `openmm_dev`**，装到临时目录挂 `PYTHONPATH`：

```bash
S=/tmp/test_prune   # 任意临时目录
python -m pip install -q --target $S/covlib coverage pytest-cov
cat > $S/covrc <<RC
[run]
source = .
omit = tests/*
    refactor/*
    output/*
    tmp/*
    .codex-test-tmp/*
    archive/*
    docs/*
data_file = $S/cov/.coverage
relative_files = True
RC
mkdir -p $S/cov
PYTHONPATH=$S/covlib COVERAGE_CORE=ctrace PYMBAR_DISABLE_JAX=1 \
  python -m pytest -q -p no:cacheprovider -n 32 --dist loadfile \
  --cov --cov-config=$S/covrc --cov-context=test --cov-report=
```

⚠️ **必须 `COVERAGE_CORE=ctrace`**。Python 3.12 上 coverage 默认的 `sysmon` 内核**不支持动态上下文**，每行只记第一次命中的测试
（09-24 实测：只有 1380 条测试拿到上下文，本应是 2307 条），逐条冗余分析会全错。开 ctrace 后全量 44 s，几乎没有额外开销。

分析（读 `$S/cov/.coverage`，输出每文件的三个列表）：

```python
import coverage, collections, re, subprocess, json
d = coverage.CoverageData(f"{S}/cov/.coverage"); d.read()
tl = collections.defaultdict(set)                      # test -> {(file, line)}
for f in d.measured_files():
    for ln, cs in d.contexts_by_lineno(f).items():
        for c in cs:
            if c: tl[re.sub(r"\|(run|setup|teardown)$", "", c)].add((f, ln))
ids = [i for i in subprocess.run(["python", "-m", "pytest", "--collect-only", "-q"],
       capture_output=True, text=True).stdout.splitlines() if "::" in i]
plan = {}
for f in sorted({i.split("::")[0] for i in ids}):
    ts = [i for i in ids if i.startswith(f + "::")]
    cov = [t for t in ts if tl.get(t)]
    rem, keep = set().union(*(tl[t] for t in cov)) if cov else set(), []
    while rem:                                          # 文件内贪心覆盖
        t = max(cov, key=lambda t: len(tl[t] & rem)); keep.append(t); rem -= tl[t]
    plan[f] = {"keep_cover": keep,
               "zero_line": [t for t in ts if not tl.get(t)],
               "redundant_in_file": [t for t in cov if t not in keep]}
json.dump(plan, open(f"{S}/prune_plan.json", "w"), ensure_ascii=False, indent=1)
```

跨文件 import 表：对 `tests/*.py` 做 AST，收集 `from test_xxx import …` / `import test_xxx` / `from production_artifact_fixtures import …`。

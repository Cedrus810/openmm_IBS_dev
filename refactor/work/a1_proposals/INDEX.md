# A1 测试守卫解耦 —— 提案索引

由 `refactor/tools/a1_tool.py` 生成（纯 AST，不 import 主线模块）。
每个 `*.diff` 是**提案**：把「读主线 .py 文本」换成 `inspect.getsource(对象)`，
**断言字面量一字不改**。落地前逐条人工过（HOW_TO_CONTINUE 3.4）。

## 总计

- 扫描测试文件：216 个
- 断言处（assert 内，C1–C5）：179
  - C1: 35
  - C2: 125
  - C3: 18
  - C4: 1
  - C5: 0
- 另有 9 处非 assert 的 .index/.find/.count 定位辅助使用（已参与收窄安全性判断）
- 状态分布：auto=75 ambiguous=65 module_level=4 c1_no_target=35
- **not_found（守卫钉错地方/已腐烂）: 0**
- **c1_exposed（C1 守卫靠扫全文件才成立/命中注释字符串）: 0**
- ambiguous: 65

## 分类说明

| 状态 | 含义 |
|---|---|
| auto | 字面量唯一命中一个顶层定义，提案直接收窄到 `inspect.getsource(模块.类.方法)` |
| ambiguous | 命中多个定义，收窄会改变守卫强度；diff 保底绑模块对象，候选见明细 |
| module_level | 字面量在模块顶层（import 区/模块级语句），提案 `inspect.getsource(模块)` |
| not_found | 字面量不在目标主线文件里 —— 守卫钉错地方或测试已腐烂，**必须逐条报告** |
| c1_no_target | C1（not in 全文件）且字面量当前不存在 = 守卫生效中；收窄无目标，保底绑模块 |
| c1_exposed | C1 但字面量在文件中出现 —— 守卫靠「扫全文件」的语义与该出现并存，单独标记 |

## 汇总表（仅列出有断言的测试文件）

| 测试文件 | 断言数 | C1 | C2 | C3 | C4 | C5 | auto | ambiguous | module_level | not_found | c1_no_target | c1_exposed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test_0831issue_p2_batch.py | 33 | 9 | 19 | 5 | 0 | 0 | 17 | 7 | 0 | 0 | 9 | 0 |
| test_autonomous_controller_toggle_is_wired_2026_09_16.py | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| test_boresch_anchor_and_pbc_fixes.py | 4 | 0 | 4 | 0 | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 |
| test_charge_transfer_hamiltonian.py | 4 | 2 | 1 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 2 | 0 |
| test_checkpoint_platform_migration.py | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| test_coalchemical_ion_identity.py | 5 | 0 | 2 | 3 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| test_coion_cache_resume_provenance.py | 8 | 0 | 7 | 1 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 0 |
| test_decharging_seed_contract.py | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| test_dispersion_protocol_is_honored.py | 7 | 0 | 7 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| test_final_result_schema_gate.py | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 |
| test_fk_relearn_and_reachability.py | 9 | 1 | 7 | 1 | 0 | 0 | 0 | 7 | 1 | 0 | 1 | 0 |
| test_gromacs_pairs_funct2_conversion.py | 5 | 0 | 5 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| test_import_time_side_effects.py | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| test_independent_endpoint_sampling.py | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 |
| test_input_file_fail_closed.py | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| test_ligand_com_restraint_removed.py | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| test_ligand_conformer_gate.py | 5 | 0 | 3 | 2 | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 0 |
| test_ligand_xml_extraction.py | 6 | 1 | 5 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 1 | 0 |
| test_membrane_barostat_protocol.py | 10 | 0 | 9 | 1 | 0 | 0 | 6 | 1 | 3 | 0 | 0 | 0 |
| test_membrane_observable_extractor.py | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| test_no_ibs_only_vanishing_mainvalue.py | 7 | 0 | 7 | 0 | 0 | 0 | 2 | 5 | 0 | 0 | 0 | 0 |
| test_platform_routing_regressions.py | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| test_probe_and_hysteresis_limits.py | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| test_rbfe_pipeline_framework.py | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| test_remd_backend_cli_wiring.py | 2 | 0 | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| test_routing_signal_not_terminal.py | 3 | 1 | 2 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| test_runrbfe_cli.py | 4 | 3 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 0 |
| test_stage2_controller_module_boundary.py | 3 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 |
| test_stage2_retire_and_result_status_2026_09_16.py | 5 | 1 | 4 | 0 | 0 | 0 | 1 | 3 | 0 | 0 | 1 | 0 |
| test_stage_diagnostics_persistence.py | 2 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| test_untrusted_switch_is_not_identity.py | 8 | 0 | 7 | 0 | 1 | 0 | 7 | 1 | 0 | 0 | 0 | 0 |
| test_validation_indeterminate_routing.py | 23 | 2 | 21 | 0 | 0 | 0 | 18 | 3 | 0 | 0 | 2 | 0 |

## not_found 明细（守卫钉错地方 / 测试已腐烂 —— 重点）

（无）

## C1 收窄暴露明细（字面量在目标文件中出现 —— 守卫语义存疑）

（无）

## ambiguous 明细（字面量命中多个定义）

- `tests/test_0831issue_p2_batch.py:53` — C3（count_eq）字面量 `'interval_weights = np.asarray(density_weight'` 在 `abfe_preoptimizer` 命中 2 处：abfe_preoptimizer.ABFEPreOptimizer.optimize_lambda_path_adaptive, abfe_preoptimizer.DualLambdaPreOptimizer.optimize_stage1_decharging
- `tests/test_0831issue_p2_batch.py:232` — C2（count_ge）字面量 `'"boresch_correction_error_kJ_mol"'` 在 `runabfe` 命中 2 处：runabfe.run_post_analysis, runabfe.run_traditional_mode
- `tests/test_0831issue_p2_batch.py:245` — C2（in）字面量 `'if _flag_present("--apbs-correction-note"):'` 在 `runabfe` 命中 2 处：runabfe.RunConfig.__init__, runabfe.run_post_analysis
- `tests/test_0831issue_p2_batch.py:254` — C3（count_eq）字面量 `'if _split_platform_spec(equil_platform)[0].upper() == "CUDA"…'` 在 `abfe_pipeline` 命中 3 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._rebalance_with_boresch, abfe_pipeline.ABFEPipeline.pre_equilibrate
- `tests/test_0831issue_p2_batch.py:275` — C2（in）字面量 `'solvent_kwargs["repeat_seed"] = int(_repeat_seed)'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.run_full_abfe_loop
- `tests/test_0831issue_p2_batch.py:276` — C2（in）字面量 `'solvent_kwargs["leg_name"] = "solvent"'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.run_full_abfe_loop
- `tests/test_0831issue_p2_batch.py:435` — C3（count_eq）字面量 `'_assert_frozen_ligand_ligand_exceptions(\n'` 在 `ibs_engine` 命中 4 处：ibs_engine._assert_frozen_ligand_ligand_exceptions, ibs_engine.configure_charge_transfer_decharging, ibs_engine.configure_coalchemical_neutral_decharging, ibs_engine.configure_pme_ligand_charge_offsets
- `tests/test_autonomous_controller_toggle_is_wired_2026_09_16.py:64` — C2（in）字面量 `'kwargs.get("'` 在 `abfe_pipeline` 命中 7 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._preopt_cache_matches_ignoring_code_hash, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage, abfe_pipeline.ABFEPipeline._run_independent_endpoint_segment, abfe_pipeline.ABFEPipeline._window_ranges_for_lambdas…
- `tests/test_autonomous_controller_toggle_is_wired_2026_09_16.py:64` — C2（in）字面量 `'"'` 在 `abfe_pipeline` 命中 194 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.__init__, abfe_pipeline.ABFEPipeline._apply_already_good_repairs, abfe_pipeline.ABFEPipeline._apply_fourier_torsions, abfe_pipeline.ABFEPipeline._apply_traditional_torsions…
- `tests/test_boresch_anchor_and_pbc_fixes.py:215` — C2（count_ge）字面量 `'_truncate_production_history('` 在 `ibs_engine` 命中 2 处：ibs_engine.IBSWindowManagerDualLambda._production_disaster_rollback, ibs_engine._truncate_production_history
- `tests/test_boresch_anchor_and_pbc_fixes.py:418` — C2（in）字面量 `'"'` 在 `abfe_core` 命中 218 处：abfe_core.ACESoftcorePotential.<classbody>, abfe_core.ACESoftcorePotential._normalize_alpha_units, abfe_core.ACESoftcorePotential.build_expression, abfe_core.ACESoftcorePotential.from_dict, abfe_core.ACESoftcorePotential.get_parameters_dict…
- `tests/test_boresch_anchor_and_pbc_fixes.py:418` — C2（in）字面量 `'"'` 在 `abfe_core` 命中 218 处：abfe_core.ACESoftcorePotential.<classbody>, abfe_core.ACESoftcorePotential._normalize_alpha_units, abfe_core.ACESoftcorePotential.build_expression, abfe_core.ACESoftcorePotential.from_dict, abfe_core.ACESoftcorePotential.get_parameters_dict…
- `tests/test_charge_transfer_hamiltonian.py:856` — C3（count_eq）字面量 `'configure_charge_transfer_decharging('` 在 `ibs_engine` 命中 2 处：ibs_engine.configure_charge_transfer_decharging, ibs_engine.configure_pme_ligand_charge_offsets
- `tests/test_charge_transfer_hamiltonian.py:874` — C2（in）字面量 `'_insert_reserved_coalchemical_ion_dummies'` 在 `runabfe` 命中 3 处：runabfe._insert_reserved_coalchemical_ion_dummies, runabfe._select_reserved_coion_water_sites, runabfe.build_and_cache_solvent_leg
- `tests/test_checkpoint_platform_migration.py:168` — C3（count_eq）字面量 `'load_checkpoint_with_platform_migration('` 在 `abfe_pipeline` 命中 5 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._rebalance_with_boresch, abfe_pipeline.ABFEPipeline.pre_equilibrate, abfe_pipeline._is_checkpoint_valid, abfe_pipeline.load_checkpoint_with_platform_migration
- `tests/test_coalchemical_ion_identity.py:174` — C3（count_eq）字面量 `'_select_bulk_water_counterion('` 在 `ibs_engine` 命中 2 处：ibs_engine._select_bulk_water_counterion, ibs_engine.select_co_alchemical_ion_once
- `tests/test_coalchemical_ion_identity.py:544` — C3（count_eq）字面量 `'select_co_alchemical_ion_once('` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.resolve_co_alchemical_ion_spec
- `tests/test_coalchemical_ion_identity.py:549` — C2（in）字面量 `'coalchemical_ion_spec.json'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._co_alchemical_ion_spec_path
- `tests/test_coalchemical_ion_identity.py:550` — C2（in）字面量 `'verify_co_alchemical_ion_identity('` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.resolve_co_alchemical_ion_spec
- `tests/test_coalchemical_ion_identity.py:552` — C3（count_eq）字面量 `'co_alchemical_ion_spec=self.resolve_co_alchemical_ion_spec()'` 在 `abfe_pipeline` 命中 4 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_2d_lambda_stage, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage, abfe_pipeline.TraditionalABFEPipeline.run_leg
- `tests/test_coion_cache_resume_provenance.py:308` — C3（count_eq）字面量 `'select_co_alchemical_ion_once('` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.resolve_co_alchemical_ion_spec
- `tests/test_coion_cache_resume_provenance.py:316` — C2（in）字面量 `'co_alchemical_ion_runtime_identity'` 在 `abfe_pipeline` 命中 7 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._preopt_cache_matches_ignoring_code_hash, abfe_pipeline.ABFEPipeline._preopt_protocol_key, abfe_pipeline.ABFEPipeline._stage_protocol_key, abfe_pipeline.ABFEPipeline.co_alchemical_ion_runtime_identity…
- `tests/test_coion_cache_resume_provenance.py:316` — C2（in）字面量 `'_stage_protocol_key'` 在 `abfe_pipeline` 命中 9 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._assert_stage_result_sane, abfe_pipeline.ABFEPipeline._charge_transfer_vanishing_handoff_active, abfe_pipeline.ABFEPipeline._preopt_protocol_key, abfe_pipeline.ABFEPipeline._stage_protocol_key…
- `tests/test_coion_cache_resume_provenance.py:316` — C2（in）字面量 `'_preopt_protocol_key'` 在 `abfe_pipeline` 命中 4 处：abfe_pipeline.ABFEPipeline._preopt_protocol_key, abfe_pipeline.ABFEPipeline._run_stage_with_overlap_autorepair, abfe_pipeline.ABFEPipeline.run_full_pipeline, abfe_pipeline._split_preopt_protocol_key
- `tests/test_coion_cache_resume_provenance.py:316` — C2（in）字面量 `'coion_identity'` 在 `abfe_pipeline` 命中 11 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._probe_vdw_window_fixed_overlap, abfe_pipeline.ABFEPipeline._run_2d_lambda_stage, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage, abfe_pipeline.ABFEPipeline._run_independent_endpoint_segment…
- `tests/test_coion_cache_resume_provenance.py:316` — C2（in）字面量 `'co_alchemical_ion_runtime_identity'` 在 `abfe_pipeline` 命中 7 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._preopt_cache_matches_ignoring_code_hash, abfe_pipeline.ABFEPipeline._preopt_protocol_key, abfe_pipeline.ABFEPipeline._stage_protocol_key, abfe_pipeline.ABFEPipeline.co_alchemical_ion_runtime_identity…
- `tests/test_coion_cache_resume_provenance.py:317` — C2（in）字面量 `'_atomic_write_json_with_encoder'` 在 `runabfe` 命中 6 处：runabfe._atomic_write_json_with_encoder, runabfe._augment_final_result_with_co_alchemical_ions, runabfe._update_run_provenance_co_alchemical_ions, runabfe._update_run_provenance_seed_contract, runabfe._write_run_provenance…
- `tests/test_coion_cache_resume_provenance.py:318` — C2（in）字面量 `'co_alchemical_ions'` 在 `runabfe` 命中 4 处：runabfe._augment_final_result_with_co_alchemical_ions, runabfe._update_run_provenance_co_alchemical_ions, runabfe._write_run_provenance, runabfe.main
- `tests/test_dispersion_protocol_is_honored.py:210` — C2（in）字面量 `'ibs_lj_tail_lrc_is_applicable(\n        potential_type, dispe…'` 在 `ibs_engine` 命中 2 处：ibs_engine.build_ibs_dual_system, ibs_engine.ibs_lj_tail_lrc_inapplicable_reason
- `tests/test_dispersion_protocol_is_honored.py:217` — C2（in）字面量 `'getattr(self, "dispersion_protocol", None),'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.compute_final_results
- `tests/test_dispersion_protocol_is_honored.py:220` — C2（in）字面量 `'getattr(self, "environment_type", None),'` 在 `abfe_pipeline` 命中 3 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._stage_protocol_key, abfe_pipeline.ABFEPipeline.compute_final_results
- `tests/test_dispersion_protocol_is_honored.py:228` — C2（in）字面量 `'"dispersion_protocol": getattr(self, "dispersion_protocol", …'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.compute_final_results
- `tests/test_dispersion_protocol_is_honored.py:232` — C2（in）字面量 `'"environment_type": getattr(self, "environment_type", None),'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.compute_final_results
- `tests/test_dispersion_protocol_is_honored.py:276` — C2（in）字面量 `'self.dispersion_protocol_info = resolve_dispersion_protocol('` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.__init__
- `tests/test_dispersion_protocol_is_honored.py:278` — C2（in）字面量 `'self.dispersion_protocol = self.dispersion_protocol_info["di…'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.__init__
- `tests/test_final_result_schema_gate.py:188` — C2（count_ge）字面量 `'validate_final_leg_result('` 在 `runabfe` 命中 2 处：runabfe.main, runabfe.run_post_analysis
- `tests/test_fk_relearn_and_reachability.py:69` — C3（count_eq）字面量 `'min_frames: int = IBS_LOCAL_MBAR_GATE_MIN_FRAMES,'` 在 `ibs_engine` 命中 2 处：ibs_engine._solve_single_window_local_mbar, ibs_engine.recalibrate_f_k_from_production
- `tests/test_fk_relearn_and_reachability.py:72` — C2（in）字面量 `'or _ie_min_frames()'` 在 `abfe_preoptimizer` 命中 2 处：abfe_preoptimizer.Stage2RepairController.<classbody>, abfe_preoptimizer.Stage2RepairController._read_window
- `tests/test_fk_relearn_and_reachability.py:153` — C2（in）字面量 `'"RELEARN_FK_EPOCH"'` 在 `abfe_preoptimizer` 命中 2 处：abfe_preoptimizer.Stage2RepairController.<classbody>, abfe_preoptimizer.Stage2RepairController._decide_once
- `tests/test_fk_relearn_and_reachability.py:154` — C2（in）字面量 `'"RECALIBRATE_FK"'` 在 `abfe_preoptimizer` 命中 2 处：abfe_preoptimizer.Stage2RepairController.<classbody>, abfe_preoptimizer.Stage2RepairController._decide_once
- `tests/test_fk_relearn_and_reachability.py:275` — C2（in）字面量 `'"decorrelated_frames_required"'` 在 `abfe_preoptimizer` 命中 2 处：abfe_preoptimizer.Stage2RepairController.<classbody>, abfe_preoptimizer.Stage2RepairController._read_window
- `tests/test_fk_relearn_and_reachability.py:276` — C2（in）字面量 `'_ie_min_frames()'` 在 `abfe_preoptimizer` 命中 3 处：abfe_preoptimizer.Stage2RepairController.<classbody>, abfe_preoptimizer.Stage2RepairController._read_window, abfe_preoptimizer._ie_min_frames
- `tests/test_fk_relearn_and_reachability.py:278` — C2（in）字面量 `'validation_completeness_frames_REPORT_ONLY'` 在 `abfe_preoptimizer` 命中 2 处：abfe_preoptimizer.Stage2RepairController.<classbody>, abfe_preoptimizer.Stage2RepairController._read_window
- `tests/test_independent_endpoint_sampling.py:587` — C2（in）字面量 `'_independent_endpoint_enabled'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_independent_endpoint_sampling.py:589` — C2（in）字面量 `'_excluded_for_analysis.add(int(_endpoint_window_index))'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_independent_endpoint_sampling.py:590` — C2（in）字面量 `'excluded_local_windows=_excluded_for_analysis or None,'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_independent_endpoint_sampling.py:591` — C2（in）字面量 `'combine_ibs_and_independent_endpoint('` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_ligand_conformer_gate.py:122` — C3（count_eq）字面量 `'assert_cross_leg_conformer_consistency('` 在 `abfe_core` 命中 2 处：abfe_core.assert_cross_leg_conformer_consistency, abfe_core.combine_binding_free_energy
- `tests/test_ligand_conformer_gate.py:291` — C2（count_ge）字面量 `'positions=positions,'` 在 `runabfe` 命中 5 处：runabfe._ensure_complex_reserved_coions, runabfe._write_run_provenance, runabfe.main, runabfe.run_prepare_command, runabfe.run_traditional_mode
- `tests/test_membrane_barostat_protocol.py:523` — C2（in）字面量 `'self.temperature, MEMBRANE_EQUILIBRATION_VELOCITY_SEED'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline.pre_equilibrate
- `tests/test_no_ibs_only_vanishing_mainvalue.py:41` — C2（in）字面量 `'if "stage2_independent_endpoint_states" in kwargs:'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_no_ibs_only_vanishing_mainvalue.py:60` — C2（in）字面量 `'kwargs.get("stage2_independent_endpoint", False)'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_no_ibs_only_vanishing_mainvalue.py:62` — C2（in）字面量 `'显式请求了独立端点段'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_no_ibs_only_vanishing_mainvalue.py:64` — C2（in）字面量 `'docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §3.3'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_no_ibs_only_vanishing_mainvalue.py:103` — C2（in）字面量 `'n_k_local[sampled_row] = n_frames'` 在 `ibs_engine` 命中 3 处：ibs_engine.GlobalMBARAnalyzer.solve_stage_integrated, ibs_engine._solve_single_window_local_mbar, ibs_engine.adjacent_bar_chain
- `tests/test_platform_routing_regressions.py:477` — C2（count_ge）字面量 `'pre_equilibration_identity_fingerprint('` 在 `runabfe` 命中 3 处：runabfe.main, runabfe.resolve_boresch_restraint, runabfe.run_traditional_mode
- `tests/test_routing_signal_not_terminal.py:114` — C2（in）字面量 `'STAGE_SOLVE_ROUTING_ERRORS'` 在 `abfe_pipeline` 命中 2 处：abfe_pipeline.ABFEPipeline.<classbody>, abfe_pipeline.ABFEPipeline._run_dual_lambda_stage
- `tests/test_runrbfe_cli.py:211` — C2（in）字面量 `'combine_rbfe'` 在 `runrbfe` 命中 2 处：runrbfe.cmd_combine, runrbfe.load_leg_result
- `tests/test_stage2_retire_and_result_status_2026_09_16.py:172` — C2（in）字面量 `'WINDOW_SELF_SUPPORT_N_EFF_OVER_G_ELIGIBLE'` 在 `ibs_engine` 命中 1 处：ibs_engine.window_self_support_check
- `tests/test_stage2_retire_and_result_status_2026_09_16.py:176` — C2（in）字面量 `'selfchk.get(\n                "n_eff_over_g_eligible_threshol…'` 在 `abfe_preoptimizer` 命中 2 处：abfe_preoptimizer.Stage2RepairController.<classbody>, abfe_preoptimizer.Stage2RepairController._read_window
- `tests/test_stage2_retire_and_result_status_2026_09_16.py:177` — C2（in）字面量 `'n_eff_over_g_target=w.get("self_n_eff_over_g_eligible")'` 在 `abfe_preoptimizer` 命中 2 处：abfe_preoptimizer.Stage2RepairController.<classbody>, abfe_preoptimizer.Stage2RepairController._decide_once
- `tests/test_untrusted_switch_is_not_identity.py:133` — C2（in）字面量 `'allow_untrusted_stage_results=bool('` 在 `runabfe` 命中 2 处：runabfe.main, runabfe.run_traditional_mode
- `tests/test_validation_indeterminate_routing.py:317` — C2（in）字面量 `'F_K_EVIDENCE_INDETERMINATE'` 在 `ibs_engine` 命中 1 处：ibs_engine.IBSWindowManagerDualLambda.run_all_windows
- `tests/test_validation_indeterminate_routing.py:318` — C2（in）字面量 `'F_K_EVIDENCE_CALIBRATED'` 在 `ibs_engine` 命中 1 处：ibs_engine.IBSWindowManagerDualLambda.run_all_windows
- `tests/test_validation_indeterminate_routing.py:398` — C2（in）字面量 `'"stage_protocol_key": getattr(self, "stage_protocol_key", No…'` 在 `ibs_engine` 命中 2 处：ibs_engine.IBSSampler.save_ibs_state, ibs_engine.IBSWindowManagerDualLambda.run_all_windows

## C1 无目标守卫（字面量当前全文件不存在 —— 守卫生效中，收窄需人工选目标）

共 35 条。这类守卫的语义是「这个坏写法不许在任何地方回来」，
收窄到单个函数会丢掉全文件覆盖面。提案 diff 里保底绑到**模块对象**
（`inspect.getsource(模块)`，文本与现 read_text 等价），A2 再升包级 AST 扫描。

- `tests/test_0831issue_p2_batch.py:52` — `assert 'xp[-1] = 1.0' not in 'xp[-1] = 1.0' not in src`（目标 `abfe_preoptimizer`）
- `tests/test_0831issue_p2_batch.py:130` — `assert 'e = energies[-1] if energies else 0.0' not in 'e = energies[-1] if energies else 0.0' not in src`（目标 `abfe_preoptimizer`）
- `tests/test_0831issue_p2_batch.py:229` — `assert 'np.sqrt(float(cycle["total_error_kJ_mol"]) ** 2 + err_boresch**2)' not in 'np.sqrt(float(cycle["total_error_kJ_mol"]) ** 2 + err_boresch**2)' not in src`（目标 `runabfe`）
- `tests/test_0831issue_p2_batch.py:247` — `assert '"apbs_correction_note": getattr(args, "apbs_correction_note", None) or "",' not in '"apbs_correction_note": getattr(args, "apbs_correction_note", None) or "",' not in src`（目标 `runabfe`）
- `tests/test_0831issue_p2_batch.py:253` — `assert 'if equil_platform.upper() == "CUDA":' not in 'if equil_platform.upper() == "CUDA":' not in src`（目标 `abfe_pipeline`）
- `tests/test_0831issue_p2_batch.py:269` — `assert 'self._seed_for(\n                    "charging", self.seed_stage' not in 'self._seed_for(\n                    "charging", self.seed_stage' not in src`（目标 `ibs_engine`）
- `tests/test_0831issue_p2_batch.py:342` — `assert 'np.savez(\n        os.path.join(cache_dir' not in 'np.savez(\n        os.path.join(cache_dir' not in src`（目标 `ibs_engine`）
- `tests/test_0831issue_p2_batch.py:425` — `assert 'n_samples = max(2, int(n_steps_per_state) // int(steps_per_sample))' not in 'n_samples = max(2, int(n_steps_per_state) // int(steps_per_sample))' not in src`（目标 `ibs_engine`）
- `tests/test_0831issue_p2_batch.py:462` — `assert '[(0,6), (4,10), (8,13)]' not in '[(0,6), (4,10), (8,13)]' not in src`（目标 `ibs_engine`）
- `tests/test_charge_transfer_hamiltonian.py:587` — `assert 'GhostIonHandler' not in 'GhostIonHandler' not in src`（目标 `ibs_engine`）
- `tests/test_charge_transfer_hamiltonian.py:875` — `assert '溶剂腿 builder 尚未实现' not in '溶剂腿 builder 尚未实现' not in src`（目标 `runabfe`）
- `tests/test_checkpoint_platform_migration.py:162` — `assert 'load_checkpoint_with_platform_migration' not in 'load_checkpoint_with_platform_migration' not in engine`（目标 `ibs_engine`）
- `tests/test_final_result_schema_gate.py:182` — `assert 'complex_results.get("total_delta_G_complex_kJ_mol", complex_results.get(' not in 'complex_results.get("total_delta_G_complex_kJ_mol", complex_results.get(' not in src`（目标 `runabfe`）
- `tests/test_final_result_schema_gate.py:185` — `assert 'solv_results.get("total_delta_G_complex_kJ_mol", solv_results.get(' not in 'solv_results.get("total_delta_G_complex_kJ_mol", solv_results.get(' not in src`（目标 `runabfe`）
- `tests/test_fk_relearn_and_reachability.py:68` — `assert 'min_frames: int = 10,' not in 'min_frames: int = 10,' not in src`（目标 `ibs_engine`）
- `tests/test_import_time_side_effects.py:175` — `assert 'os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] =' not in 'os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] =' not in source`（目标 `abfe_core`）
- `tests/test_import_time_side_effects.py:211` — `assert 'os.environ["PYMBAR_DISABLE_JAX"] =' not in 'os.environ["PYMBAR_DISABLE_JAX"] =' not in source`（目标 `abfe_core`）
- `tests/test_input_file_fail_closed.py:87` — `assert 'config.torsion_params and os.path.exists(config.torsion_params)' not in 'config.torsion_params and os.path.exists(config.torsion_params)' not in source`（目标 `runabfe`）
- `tests/test_ligand_com_restraint_removed.py:38` — `assert 'setForceGroup(5)' not in 'setForceGroup(5)' not in src`（目标 `ibs_engine`）
- `tests/test_ligand_com_restraint_removed.py:44` — `assert 'addGlobalParameter("k_com"' not in 'addGlobalParameter("k_com"' not in src`（目标 `ibs_engine`）
- `tests/test_ligand_com_restraint_removed.py:45` — `assert 'addGlobalParameter("r0_com"' not in 'addGlobalParameter("r0_com"' not in src`（目标 `ibs_engine`）
- `tests/test_ligand_com_restraint_removed.py:46` — `assert 'CustomCentroidBondForce(1, com_expr)' not in 'CustomCentroidBondForce(1, com_expr)' not in src`（目标 `ibs_engine`）
- `tests/test_ligand_xml_extraction.py:171` — `assert 'angle_force = next(' not in 'angle_force = next(' not in src`（目标 `runabfe`）
- `tests/test_membrane_observable_extractor.py:1044` — `assert '"expected_pre_equilibration_frames":' not in '"expected_pre_equilibration_frames":' not in runabfe_src`（目标 `runabfe`）
- `tests/test_probe_and_hysteresis_limits.py:78` — `assert 'gid = 16 + idx' not in 'gid = 16 + idx' not in src`（目标 `ibs_engine`）
- `tests/test_rbfe_pipeline_framework.py:255` — `assert 'complex_result.delta_g -' not in 'complex_result.delta_g -' not in source`（目标 `rbfe_pipeline`）
- `tests/test_routing_signal_not_terminal.py:118` — `assert 'stage_result["analysis_status"] = ANALYSIS_COMPLETE' not in 'stage_result["analysis_status"] = ANALYSIS_COMPLETE' not in src`（目标 `abfe_pipeline`）
- `tests/test_runrbfe_cli.py:213` — `assert 'complex_leg.delta_g -' not in 'complex_leg.delta_g -' not in source`（目标 `runrbfe`）
- `tests/test_runrbfe_cli.py:214` — `assert 'delta_g - ' not in 'delta_g - ' not in source`（目标 `runrbfe`）
- `tests/test_runrbfe_cli.py:221` — `assert 'import openmm' not in 'import openmm' not in source`（目标 `runrbfe`）
- `tests/test_stage2_controller_module_boundary.py:71` — `assert '_need = relearn_epoch_required_steps(' not in '_need = relearn_epoch_required_steps(' not in src`（目标 `abfe_pipeline`）
- `tests/test_stage2_controller_module_boundary.py:72` — `assert '_need = _relearn_epoch_required_steps(' not in '_need = _relearn_epoch_required_steps(' not in src`（目标 `abfe_pipeline`）
- `tests/test_stage2_retire_and_result_status_2026_09_16.py:171` — `assert 'elif _ratio < 10.0:' not in 'elif _ratio < 10.0:' not in engine`（目标 `ibs_engine`）
- `tests/test_validation_indeterminate_routing.py:122` — `assert 'child_windows' not in 'child_windows' not in src`（目标 `abfe_pipeline`）
- `tests/test_validation_indeterminate_routing.py:336` — `assert '_hopeless' not in '_hopeless' not in src`（目标 `ibs_engine`）

## spec_from_file_location 站点

- `tests/test_c2_lipid_slab_validation.py:57` → `（动态，静态解析不了）`
- `tests/test_charge_transfer_real_endpoints.py:53` → `（动态，静态解析不了）`
- `tests/test_exp019_softlift_loro.py:10` → `（动态，静态解析不了）`
- `tests/test_exp020_softlift_d2.py:16` → `（动态，静态解析不了）`
- `tests/test_generate_c3_summary_reports.py:55` → `（动态，静态解析不了）`
- `tests/test_local_residual_autofit.py:203` → `（动态，静态解析不了）`
- `tests/test_local_residual_autofit.py:423` → `（动态，静态解析不了）`

## 动态/未解析的读文本站点（diff 未覆盖，需人工）

- `tests/test_c2_lipid_slab_validation.py:57`（匿名）
- `tests/test_charge_transfer_real_endpoints.py:53`（匿名）
- `tests/test_exp019_softlift_loro.py:10`（匿名）
- `tests/test_exp020_softlift_d2.py:16`（匿名）
- `tests/test_fk_relearn_and_reachability.py:25`（匿名）
- `tests/test_fk_relearn_and_reachability.py:235`（匿名）
- `tests/test_generate_c3_summary_reports.py:55`（匿名）
- `tests/test_gromacs_pairs_funct2_conversion.py:152`（converted）
- `tests/test_lambda_path_versions.py:135`（before）
- `tests/test_lambda_path_versions.py:137`（匿名）
- `tests/test_local_residual_autofit.py:203`（匿名）
- `tests/test_local_residual_autofit.py:423`（匿名）
- `tests/test_no_undefined_globals.py:52`（src）
- `tests/test_stage2_autonomous_segment_routing.py:28`（_SRC）
- `tests/test_stage2_controller_module_boundary.py:35`（匿名）

## 孤儿断言（断言目标解析到了主线模块，但静态追不到它的读文本站点 —— 需人工）

（无）


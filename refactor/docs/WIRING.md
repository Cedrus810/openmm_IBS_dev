# refactor/ 功能域拆分 — 接线指南

**生成方式**：2026-09-19，手工定边界 + sed 行区间逐条抄写。只搬完整顶层定义，
巨函数/巨类整块未切；**import 未重写**——这是重构工作台，不是可直接运行
的包。主线零改动；本目录旧的机器拆分包（abfe_core/、ibs_engine/ 等）原样保留未动。

## 每个文件从哪来（源行号，按原文件对账用）

### boresch/（6 件）
| 文件 | 来源 | 行 |
|---|---|---|
| boresch_forces | abfe_core | 8191-8546 |
| analytic_corrections_fixups（修复2/6/3） | abfe_core | 10859-11229 |
| boresch_geometry_estimators | abfe_core | 12676-13207 |
| attachment_leg（含 run_boresch_attachment_leg 665行） | ibs_engine | 1994-3308 |
| boresch_attachment_remd | ibs_engine | 25825-26145 |
| runabfe_boresch_params | runabfe | 4149-4529 |

### mbar_analysis/（9 件）
| 文件 | 来源 | 行 |
|---|---|---|
| ibs_data_loading_head（种子契约/窗口数据校验/PME自校验） | ibs_engine | 1-1993 |
| stage1_crosschecks | ibs_engine | 26449-26576 |
| tmbar_global_analyzer（GlobalMBARAnalyzer 1350行） | ibs_engine | 22486-23838 |
| traditional_mbar_analyzer（897行） | ibs_engine | 26577-27473 |
| ess_target_support_gates | ibs_engine | 20265-22485 |
| split_half_diagnostics | ibs_engine | 23839-24376 |
| chunked_mbar（修复4） | abfe_core | 11230-11331 |
| solvation_closure（修复5） | abfe_core | 11332-11747 |
| multi_segment_analysis | 整文件 | — |

### windows_sampling/（7 件）
| 文件 | 来源 | 行 |
|---|---|---|
| ibs_bias_sampler（IBSSampler 1880行 + IBSBiasForce） | ibs_engine | 6421-11124 |
| fixed_state_probe_bank | ibs_engine | 11125-12654 |
| independent_endpoints | ibs_engine | 12655-14609 |
| **window_manager_dual_lambda（IBSWindowManagerDualLambda 5407行）** | ibs_engine | 14610-20016 |
| window_manager_shadow_coul | ibs_engine | 20017-20264 |
| online_monitor | abfe_core | 10632-10858 |
| free_energy_engine | 整文件 | — |

### remd/（2 件）：remd_manager（1099行）ibs_engine 24723-25824；shadow_bridge_remd 26146-26448
### membrane/（2 件）：membrane_protocol abfe_core 218-714；membrane_gates 4316-5179 + 6138-7430
### misc/（2 件）：force_diagnostics ibs_engine 24377-24722；ibs_engine_module_tail（`__getattr__` shim）27474-27477

### charging/（5 件）
| 文件 | 来源 | 行 |
|---|---|---|
| charge_protocol_core（电荷协议+co-ion restraint+阈值 大块） | abfe_core | 715-2832 |
| coion_charge_transfer | ibs_engine | 3309-4238 |
| shadow_coulomb_decharge（PME-Bridge） | ibs_engine | 5792-6420 |
| reserved_coions_build | runabfe | 1757-2493 |
| charging_only_rerun | runabfe | 6118-6822 |
| apbs_correction | 整文件 | — |

### lambda_path/（8 件）
| 文件 | 来源 | 行 |
|---|---|---|
| preoptimizer_head（版本史+imports） | abfe_preoptimizer | 1-410 |
| path_math（λ路径数学，零依赖可先重构） | abfe_preoptimizer | 411-1984 |
| abfe_preoptimizer_class（578行） | abfe_preoptimizer | 11580-12157 |
| dual_lambda_preoptimizer（569行类） | abfe_preoptimizer | 12342-12950 |
| probe_systems_2d（探针体系+2D测地线） | abfe_preoptimizer | 12951-13746 |
| forcefield_dispersion | abfe_core | 2833-4315 |
| dual_lambda_system（build_ibs_dual_system 497行） | ibs_engine | 4239-5791 |
| lambda_path_versions | 整文件 | — |

### stage2_controller/（3 件）
| 文件 | 来源 | 行 |
|---|---|---|
| **stage2_repair_controller（Stage2RepairController 6386行类，_decide_once 2791行）** | abfe_preoptimizer | 2111-9385 |
| stage2_repair_actions（S2-D 纯函数） | abfe_preoptimizer | 9386-11579 |
| pilot_diagnostics | abfe_preoptimizer | 12158-12341 |
| epoch_ledger | abfe_preoptimizer | 1985-2110 |

### pipeline_orchestration/（9 件）
| 文件 | 来源 | 行 |
|---|---|---|
| pipeline_head | abfe_pipeline | 1-3225 |
| **abfe_pipeline_class（ABFEPipeline 15446行巨类）** | abfe_pipeline | 3226-18671 |
| traditional_abfe_pipeline（476行类） | abfe_pipeline | 18672-19150 |
| runabfe_head / runabfe_config / runabfe_parser（490行）/ runabfe_analysis_mode / runabfe_traditional_mode / runabfe_main（main 1773行） | runabfe | 1-629 / 3208-4148 / 4530-5026 / 5027-5720 / 5721-6117 / 6823-8948 |

### system_build/（7 件）：gmx_paths_status runabfe 630-1756；solvent_leg_cache 2494-3104；build_from_gromacs 3105-3207；system_prep abfe_core 5180-6137；system_routing_ghost_ions 10289-10631；io_system_utils 11748-12675；units_validators 7852-7888
### result_schema/（3 件）：result_schema_gates abfe_core 8547-9147；stage2_ab_report 整文件；rbfe 契约见 rbfe/
### orb_surrogate/（3 件）：orb_pocket_estimator（1141行类）abfe_core 9148-10288；softcore_fitters 7889-8190；orbv3_dexp_pipeline 13208-13223
### runtime_env/（4 件）：runtime_env abfe_core 1-217；mdtraj_compat 7431-7851；abfe_diagnostics、step_guard 整文件
### outer_lambda_neural/（3 件）：protocol_head 1-660；lambda_controller_forces 661-2562；deployment_assessment_cli 2563-8109
### rbfe/（6 件）：rbfe_contracts 1-583；rbfe_mapping 584-1927；rbfe_hybrid_builder 1928-3411；rbfe_mbar 3412-3722；rbfe_pipeline、runrbfe 整文件

## 巨定义登记（重构下刀处，全部整块未切）

| 名字 | 所在域文件 | 原位置 | 行数 |
|---|---|---|---|
| ABFEPipeline | pipeline_orchestration/abfe_pipeline_class | abfe_pipeline L3226 | 15446 |
| IBSWindowManagerDualLambda | windows_sampling/window_manager_dual_lambda | ibs_engine L14610 | 5407 |
| Stage2RepairController._decide_once | stage2_controller/stage2_repair_controller | abfe_preoptimizer L5467 | 2791 |
| run_full_pipeline | pipeline_orchestration/abfe_pipeline_class | abfe_pipeline L15296 | 3376 |
| IBSSampler | windows_sampling/ibs_bias_sampler | ibs_engine L8745 | 1880 |
| main | pipeline_orchestration/runabfe_main | runabfe L7176 | 1773 |
| GlobalMBARAnalyzer | mbar_analysis/tmbar_global_analyzer | ibs_engine L22489 | 1350 |
| REMDManager | remd/remd_manager | ibs_engine L24726 | 1099 |
| _run_stage2_autonomous | pipeline_orchestration/abfe_pipeline_class | abfe_pipeline L11527 | 1554 |
| TraditionalMBARAnalyzer | mbar_analysis/traditional_mbar_analyzer | ibs_engine L26577 | 897 |

## 巨类内部子区索引（类不切，方法按行号找）

- **ABFEPipeline**（abfe_pipeline L3229-18671）：类头/init L3229；System缓存 L3579；
  断点续传 L3673；co-ion 身份 L3683；预平衡 L3931（pre_equilibrate 510行 L4361）；
  Boresch 再平衡 L4868；预优化 L5210；二面角 L5290；双λ路由 L5460
  （_run_dual_lambda_stage 1050行 L5739）；shadow-ibs L6881；2D λ L7149；
  控制器机制群 L7346-15292；主流程 run_full_pipeline L15296。
- **IBSWindowManagerDualLambda**（ibs_engine L14610-20016）：读盘/可行性/决策/聚合
  分区见 stage2_repair_controller 同构。
- **Stage2RepairController**（abfe_preoptimizer L3020-9405）：读盘 L3292；可行性 L4931；
  决策 L5114（decide L5443 → _decide_once L5467）；聚合 L8641；replay L9040；人读 L9205。

## 接线事实（重构时要知道的跨域关系）

1. 原文件的 import 关系：abfe_pipeline ← abfe_core + ibs_engine；runabfe ← 三个全引；
   rbfe_core 惰性引 abfe_core/ibs_engine；outer_lambda_neural_basis 不 import 主线。
2. 域文件里的函数引用原模块级名字（np/unit/常量等）时，去源文件头部 import 区拿
   （各 *head 文件保留了原 import 块）。
3. `print = _log_print`（abfe_core L7409，在 membrane_gates 域）：跨域引用它的函数
   在原文件里全部读模块全局，接线时统一从 membrane_gates 域引。
4. 函数内 `global` 名字（_PYMBAR_MODULE 等 8 个）全在 runtime_env 域，跨域别拆散。
5. tests/、tools/、local_residual/、plugins/、exp012_xed/、docs/ 未搬，原地未动。

## 验证

- 80 个文件全部 `py_compile` 通过（一次边界修正：outer_lambda protocol_head 661 行
  的装饰器归位到类文件）。
- 行数对账：window_manager_dual_lambda 5407、abfe_pipeline_class 15446、
  tmbar_global_analyzer 1353 与源逐段一致；合计约 10 万行 + 9 个整文件。
- 本地未跑测试（按规矩）；最终验收 = 远程 `./tests/run_offline_tests.sh`。

# 历史材料 log

本仓库是 **ABFE-IBS 工程区分支**：只保留发布所需的生产代码、生产回归测试和使用文档。

2026-08-31 的发布整理把开发期的过程材料（实验记录、交接单、审计快照、
阶段性结论、`curated_project/` 整理版知识库）从本仓库移除，压缩成下面这一份 log。
**原文全部保存在 `Atenolol-rank11` 工作区**，本表只登记它们存在过、写了什么、日期。

本表是历史索引，不是当前状态。当前状态看 [README](README.md)、
[当前行动清单](TODO.md) 和 [发布准备度评估](RELEASE_READINESS_2026-08-31.md)。

下面的主表逐份登记 **148** 份文档（另有 46 份与已登记材料逐字重复，见《一、文档》）。
移出的代码、脚本、实验协议和模型资源分别登记在第二至第四节。

| 日期 | 原路径 | 标题 / 摘要 | 字节 |
|---|---|---|---|
| 2026-07-10 | `docs/handoffs/RESUME_DEXP_SESSION.md` | DEXP / MACE surrogate 工作恢复笔记 | 32113 |
| 2026-07-13 | `docs/experiments/DEXP_KERNEL_PHYSICS_ISSUES.md` | DEXP 核参数选择与物理问题记录 | 119091 |
| 2026-07-17 | `docs/status/ABFE_FULL_AUDIT_2026-07-17.md` | Atenolol-rank11 ABFE/IBS 项目全量物理与代码审计 | 20724 |
| 2026-07-18 | `docs/status/NON_MUTATING_V1_STATUS.md` | IBS `non_mutating_v1` — Status & Handoff (2026-07-18) | 19896 |
| 2026-07-19 | `docs/handoffs/VANISHING_WINDOW0_HANDOFF.md` | Vanishing 阶段 / 窗口 0 交接笔记（写于 2026-07-19，2026-07-20 更新，下一个 agent/会话先看这个） | 27148 |
| 2026-07-20 | `docs/archive/todolist-2026-07-20.md` | 当前仍未完成项（复核于 2026-07-20，pilot 加密修复 v15 后更新） | 21941 |
| 2026-07-21 | `docs/status/AUDIT_STATUS.md` | ABFE 审计状态总表（历史记录；当前覆盖更新至 2026-07-21） | 169285 |
| 2026-07-22 | `docs/status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md` | IBS 预热、生产与 Coverage Rescue 协议（2026-07-22） | 13991 |
| 2026-07-22 | `curated_project/01_当前使用文档/文档导航_原文.md` | ABFE-IBS 文档导航 | 2057 |
| 2026-07-26 | `docs/status/VALIDATION_MATRIX.md` | 运行验证监控表 | 24500 |
| 2026-07-26 | `docs/archive/存档.md` | 存档 | 11172 |
| 2026-07-27 | `docs/status/RESULT_2026-07-27_atenolol_rank11.md` | Atenolol-rank11 结合自由能结果记录（2026-07-27 那一轮） | 11495 |
| 2026-07-27 | `docs/status/bug.md` | ABFE/IBS 流水线 Bug 审查报告（第 3 版 — 修复复查 · 结案） | 7069 |
| 2026-07-27 | `docs/status/evidence_2026-07-27/README.md` | 2026-07-27 诊断证据（V-06 / P0-10） | 4280 |
| 2026-07-27 | `docs/archive/TODO-2026-07-27-full.md` | 当前行动清单 | 66886 |
| 2026-07-27 | `curated_project/00_从这里开始/RESULTS.md` | 数字与科学结论 | 1726 |
| 2026-07-27 | `curated_project/03_当前工作线/04_协议/LAMBDA_SCHEDULE_CONTRACT.md` | Stage 2 λ 调度契约（协议 v21） | 3849 |
| 2026-07-27 | `curated_project/04_历史与无效证据/docs_archive/removed_overlap_autorepair_mutation_loop.md` | 已移除：`_run_stage_with_overlap_autorepair` 的 ensemble 变异自动修复循环 | 63571 |
| 2026-07-27 | `curated_project/04_历史与无效证据/docs_archive/removed_parallel_stages.md` | 已移除：`--parallel-stages` 并行阶段执行 | 16992 |
| 2026-07-27 | `curated_project/04_历史与无效证据/docs_archive/removed_refine_lambda_path_with_medium_probe.md` | 已移除：`_refine_lambda_path_with_medium_probe` | 4987 |
| 2026-07-27 | `curated_project/04_历史与无效证据/docs_archive/removed_retired_overlapping_vdw_schedule_design.md` | 已移除：`_retired_overlapping_vdw_schedule_design` | 10538 |
| 2026-07-28 | `docs/archive/TODO-2026-07-28-completed.md` | 2026-07-28 完成项与复核归档 | 21339 |
| 2026-07-29 | `docs/status/README_STATUS_SNAPSHOT_2026-07-29.md` | 原 README 当前状态快照（2026-07-29） | 5011 |
| 2026-07-29 | `docs/handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md` | Boresch 二面角符号反号事故 —— 诊断、修复、遗留项 | 23188 |
| 2026-07-29 | `docs/archive/BOR-01-boresch-dihedral-sign-fixed-2026-07-29.md` | BOR-01 Boresch dihedral sign bug archive | 2819 |
| 2026-07-29 | `curated_project/04_历史与无效证据/design_proposals/PROPOSAL_dexp_merge_into_core.md` | 提案：把 `dexp_NEW.py` 合并进 `abfe_core.py`，并隔离退役的 Orb 拟合代码 | 9977 |
| 2026-07-29 | `curated_project/04_历史与无效证据/misc/docs_issue.MD` | Open GitHub Issues | 4191 |
| 2026-08-02 | `docs/experiments/EXP-011_PREREGISTRATION.md` | EXP-011 / WP-4B 预注册与验收协议 | 4190 |
| 2026-08-03 | `curated_project/04_历史与无效证据/historical_reports/EXP-012_INITIAL_IMPLEMENTATION_2026-08-03.md` | EXP-012 初始工程切片与结论（2026-08-03） | 1890 |
| 2026-08-03 | `curated_project/04_历史与无效证据/historical_reports/EXP-012_MM_LEDGER_PREFLIGHT_2026-08-03.md` | EXP-012 完整 MM ledger 预检（2026-08-03） | 2225 |
| 2026-08-03 | `curated_project/04_历史与无效证据/root_archive/exp012_radial_support_bug_20260803.md` | EXP-012 radial-support bug archive (2026-08-03) | 1243 |
| 2026-08-04 | `docs/handoffs/CHARGE_TRANSFER_B3_HANDOFF.md` | B3：PME co-alchemical charge-transfer charging Hamiltonian（含 MEM-00d） | 10304 |
| 2026-08-04 | `docs/handoffs/MEMBRANE_SOLVENT_LEG_P013_HANDOFF.md` | 膜体系溶剂腿：配体键角被静默丢掉（P0-13）+ 检测层（P0-12a/b）+ 色散分层（B6-FIX） | 9754 |
| 2026-08-06 | `docs/status/memtodolist_archive.md` | 膜受体–配体 ABFE 专项行动清单（历史归档） | 256133 |
| 2026-08-06 | `curated_project/03_当前工作线/00_全局行动/TODO.md` | 当前行动清单 | 107994 |
| 2026-08-07 | `docs/status/memory.md` | 项目记忆导出（交接用） | 10082 |
| 2026-08-07 | `docs/experiments/IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md` | 外层 λ 神经基势详细实施计划 | 62035 |
| 2026-08-07 | `docs/experiments/IMPLEMENTATION_PLAN_outer_lambda_neural_basis_archive.md` | 外层 λ 神经基势详细实施计划（归档） | 87031 |
| 2026-08-09 | `docs/experiments/TODO_outer_lambda_orb_neural_basis.md` | TODO：ORB-v3 浅层神经基势分支实验 | 49473 |
| 2026-08-10 | `docs/experiments/PLAN_EXP-017_overlap_first.md` | EXP-017 计划：Overlap-first 的非神经 λ 路径与条件性解析 CV | 23267 |
| 2026-08-10 | `docs/experiments/PLAN_EXP-018_stationarity_confirmation.md` | PLAN-EXP-018：window 5 stationarity confirmation | 8146 |
| 2026-08-11 | `docs/status/memtodolist.md` | 膜受体–配体 ABFE 当前行动清单 | 10811 |
| 2026-08-11 | `docs/experiments/PLAN_EXP-019_baseline_reproducibility_uncertainty_calibration.md` | PLAN-EXP-019：baseline reproducibility / uncertainty calibration | 6352 |
| 2026-08-11 | `docs/experiments/STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md` | Stage2 charging→vanishing handoff：设计提案 + 已实现的工具函数 | 15780 |
| 2026-08-11 | `curated_project/03_当前工作线/01_膜与带电配体/memtodolist.md` | 膜受体–配体 ABFE 当前行动清单 | 81127 |
| 2026-08-12 | `docs/experiments/PLAN_EXP-021_grouped_density_cv.md` | EXP-021 GroupedDensityCV：成本优先的局部多体路径基函数设计 | 39063 |
| 2026-08-12 | `docs/curation/CONFLICTS.md` | Scientific and document conflicts | 2391 |
| 2026-08-12 | `docs/curation/CURATION_LOG.md` | Curation log | 1826 |
| 2026-08-12 | `curated_project/00_从这里开始/CURRENT_STATUS.md` | 当前状态 | 2255 |
| 2026-08-12 | `curated_project/00_从这里开始/DOCUMENT_MAP.md` | 文档地图 | 1125 |
| 2026-08-12 | `curated_project/02_当前综合报告/CURRENT_CODE_AND_NEW_DESIGNS_WORKING_PRINCIPLES_2026-08-12.md` | ABFE-IBS 当前代码与新设计的工作原理 | 27625 |
| 2026-08-12 | `curated_project/02_当前综合报告/DEVELOPMENT_FAILURES_AND_EVIDENCE_2026-08-12.md` | ABFE-IBS 开发失败路线、负结果与证据底稿 | 13706 |
| 2026-08-12 | `curated_project/02_当前综合报告/PIPELINE_AND_METHODS_LANDSCAPE_2026-08-12.md` | ABFE-IBS 整体 Pipeline 与方法全景 | 18240 |
| 2026-08-12 | `curated_project/02_当前综合报告/README.md` | 当前综合报告 | 1409 |
| 2026-08-12 | `curated_project/02_当前综合报告/SOFTWARE_PROGRESS_AND_TECHNICAL_DRAFT_2026-08-12.md` | ABFE-IBS 软件开发进度与技术方法底稿 | 16338 |
| 2026-08-12 | `curated_project/README.md` | Atenolol-rank11 整理版知识库 | 2578 |
| 2026-08-13 | `docs/BUG_FIX_TEST_EVIDENCE_2026-08-13.md` | Bug 修复合同测试执行证据（2026-08-13） | 2522 |
| 2026-08-13 | `docs/BUG_FIX_TEST_EVIDENCE_2026-08-13_ISSUE84_CLOSURE.md` | Issue #84 关闭证据（2026-08-13） | 1397 |
| 2026-08-13 | `docs/BUG_FIX_TEST_EVIDENCE_2026-08-13_ROUND2.md` | Bug 修复测试证据：#75 / #84 专项（2026-08-13） | 1559 |
| 2026-08-13 | `docs/BUG_FIX_TEST_PLAN_2026-08-13.md` | Bug 修复 pytest 实施明细（2026-08-13） | 9581 |
| 2026-08-13 | `docs/BUG_FIX_TODO_2026-08-13.md` | GitHub Issues Bug 修复 TODO（2026-08-13） | 12543 |
| 2026-08-13 | `docs/ISSUE_62_76_IMPLEMENTATION_2026-08-13.md` | Issues #62 and #76 implementation evidence (2026-08-13) | 2874 |
| 2026-08-13 | `docs/PYMBAR_COMPATIBILITY_NOTE_2026-08-13.md` | PyMBAR compatibility versus reproducibility pin | 1220 |
| 2026-08-13 | `docs/experiments/MERGE_TO_MAIN_PLAN.md` | 带电膜 charge-transfer 工程基础合并到主线计划 | 8521 |
| 2026-08-13 | `docs/experiments/PLAN_EXP-026_cuda_control_plane_optimization.md` | PLAN EXP-026 — Local Many-Body CUDA 控制面优化 | 86080 |
| 2026-08-13 | `docs/experiments/PLAN_EXP-027_online_utility.md` | PLAN EXP-027 — A1.1 Native SoftLift 在线效用资格化 | 46699 |
| 2026-08-13 | `curated_project/02_当前综合报告/ADVISOR_DETAILED_PROJECT_REPORT_WITH_DATA_2026-08-13.md` | ABFE-IBS 项目详细进展报告：方法、数据、失败路线与下一阶段计划 | 37873 |
| 2026-08-14 | `docs/experiments/exp-30.md` | EXP-030：统一状态条件化 IBS score——数学理论与操作方法 | 26143 |
| 2026-08-14 | `docs/experiments/exp027_result.md` | EXP-027 结果汇总（含 EXP-028 修复） | 22616 |
| 2026-08-24 | `docs/experiments/EXP-029_WIRING_HANDOFF_2026-08-24.md` | EXP-029 接线交接（2026-08-24） | 10183 |
| 2026-08-24 | `docs/experiments/EXP-030_IMPLEMENTATION_HANDOFF_2026-08-24.md` | EXP-030 大规模代码补全与对话交接（2026-08-24） | 19975 |
| 2026-08-24 | `docs/experiments/P1-19_ONLINE_SLIDING_WINDOW_SPLITHALF_MISMATCH.md` | P1-19 附属发现：在线 TMBAR 滑动窗口下，split-half 诊断的"window 0"不是物理窗口 | 7039 |
| 2026-08-24 | `docs/experiments/exp029_result.md` | EXP-029 结果汇总（会话中间记录，未合并主线） | 11306 |
| 2026-08-25 | `docs/experiments/EXP-030_EM_COLLAPSE_AND_NORESIDUAL_PATCH_2026-08-25.md` | EXP-030 window_0 EM 崩溃排查 + s_residual 修复 + EM-no-residual patch（2026-08-25） | 20527 |
| 2026-08-25 | `docs/experiments/SESSION_CHANGELOG_2026-08-25_performance_and_warmup_redesign.md` | 会话总结：性能优化 + 窗口预热状态机重构（2026-08-25） | 33852 |
| 2026-08-26 | `docs/experiments/EXP-030_AB_COMPARISON_SUMMARY_2026-08-26.md` | EXP-030 baseline vs candidate 对比总结（2026-08-26） | 5360 |
| 2026-08-26 | `docs/experiments/EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md` | EXP-030：`frozen_score.json` 快照时机早于生产入口自我修正，导致偶发 f_k 不一致（2026-08-26） | 7910 |
| 2026-08-26 | `docs/experiments/EXP-030_IBS_FK_RESIDUAL_TRAINING_BUG_2026-08-26.md` | EXP-030：candidate 臂 f_k 在线学习训练目标缺失残差项的 bug（2026-08-26） | 8625 |
| 2026-08-26 | `docs/experiments/PLAN_pipeline_speed_2026-08-26.md` | PLAN：Pipeline 速度优化 — 下一批候选（2026-08-26） | 9738 |
| 2026-08-27 | `docs/experiments/EXP-030_FINAL_STATUS_2026-08-27.md` | EXP-030：v31 修复后的完整状态（2026-08-27） | 73238 |
| 2026-08-29 | `docs/status/BUGFIX_HANDOFF_2026-08-29.md` | ABFE-IBS 代码缺陷修复交接单（2026-08-29） | 3041 |
| 2026-08-30 | `docs/experiments/EXP-030_MAINLINE_INTEGRATION_AND_NEW_SYSTEM_PLAN_2026-08-30.md` | EXP-030 合并主线与换体系验证方案（2026-08-30） | 25252 |
| 2026-08-31 | `docs/ISSUES_67_27_75_83_FIXES.md` | #67 / #27 / #75 / #83 修复记录 | 5026 |
| 2026-08-31 | `docs/status/github issue.md` | GitHub Issues 快照 — Cedrus810/openmm_IBS_dev | 13480 |
| — | `docs/status/VALIDATION_MATRIX.md.pre_resume_gate_extraction` | VALIDATION_MATRIX.md | 20537 |
| — | `docs/status/evidence_2026-07-27/charging_linear_response.json` | charging_linear_response | 2049 |
| — | `docs/status/evidence_2026-07-27/endpoint_sigma_diagnosis.json` | endpoint_sigma_diagnosis | 38442 |
| — | `docs/status/evidence_2026-07-27/pose_drift.json` | pose_drift | 1050 |
| — | `docs/handoffs/POSE_SCAN_HANDOFF.md` | pose-scan / pull-scan 交接笔记（临时，下次接着 debug 用） | 7612 |
| — | `docs/experiments/DiffLift.MD` | EXP-020 SoftLift：面向 Outer-λ 守恒路径势的可实现设计 | 75400 |
| — | `docs/experiments/EXPERIMENT_LOG_outer_lambda_neural_basis.md` | 外层 λ 神经基势实验日志 | 207623 |
| — | `docs/experiments/PLAN_EXP-025_local_manybody_cuda.md` | EXP-025：Local Many-Body Residual CUDA Plugin | 31616 |
| — | `docs/experiments/PLAN_outer_lambda_neural_basis.md` | 外层 λ 神经基势研发计划 | 54778 |
| — | `docs/experiments/TODO_outer_lambda_neural_basis_reframed.md` | TODO：Outer-λ Neural Basis 重构后的研究与实现清单（复核修订版） | 41211 |
| — | `docs/experiments/dexp_experiment.md` | dexp_experiment.py | 10240 |
| — | `docs/curation/BATCH_PLAN.md` | Zero-deletion curation batch plan | 2302 |
| — | `docs/curation/DOCUMENT_ROLE_MAP.csv` | DOCUMENT_ROLE_MAP | 2760 |
| — | `docs/curation/FILE_INVENTORY.csv` | FILE_INVENTORY | 1243372 |
| — | `docs/curation/IMMUTABILITY_POLICY.md` | Immutable evidence policy | 830 |
| — | `docs/curation/MOVE_MAP.csv` | MOVE_MAP | 123 |
| — | `docs/curation/README.md` | 项目整理控制中心 | 2822 |
| — | `docs/curation/RESULT_REGISTRY.csv` | RESULT_REGISTRY | 2296 |
| — | `curated_project/00_从这里开始/COPY_POLICY.md` | 副本与原始资料政策 | 701 |
| — | `curated_project/00_从这里开始/README.md` | 从这里开始 | 1548 |
| — | `curated_project/01_当前使用文档/README.md` | 当前使用文档 | 1329 |
| — | `curated_project/01_当前使用文档/教程/GETTING_STARTED.md` | 安装、输入与运行 | 5905 |
| — | `curated_project/01_当前使用文档/教程/MAINTAINING.md` | 维护与修改代码 | 1998 |
| — | `curated_project/01_当前使用文档/教程/MIGRATING_TO_A_NEW_SYSTEM.md` | 迁移到新的蛋白–配体体系 | 2977 |
| — | `curated_project/01_当前使用文档/教程/OUTPUTS_AND_RESUME.md` | 输出、结果解读与续跑 | 6016 |
| — | `curated_project/01_当前使用文档/教程/TROUBLESHOOTING.md` | 常见问题与排障 | 2484 |
| — | `curated_project/01_当前使用文档/教程/current-pipeline.svg` | current-pipeline | 13114 |
| — | `curated_project/01_当前使用文档/语言入口/README_cn.md` | 中文入口 | 561 |
| — | `curated_project/01_当前使用文档/语言入口/README_en.md` | English entry | 595 |
| — | `curated_project/01_当前使用文档/项目原始入口_README.md` | ABFE-IBS workflow | 905 |
| — | `curated_project/03_当前工作线/02_outer_lambda/EXPERIMENT_LOG_outer_lambda_neural_basis.md` | 外层 λ 神经基势实验日志 | 207725 |
| — | `curated_project/03_当前工作线/04_协议/PROJECT_LAYOUT.md` | 项目目录导航 | 2057 |
| — | `curated_project/03_当前工作线/README.md` | 当前工作线 | 1431 |
| — | `curated_project/04_历史与无效证据/README.md` | 历史与无效证据 | 1316 |
| — | `curated_project/04_历史与无效证据/design_proposals/PROPOSAL_frozen_validation_fallback.md` | Proposal: let frozen validation run once even if the candidate streak never completes | 3729 |
| — | `curated_project/04_历史与无效证据/misc/reports_README.md` | 项目记录与结论 | 305 |
| — | `curated_project/04_历史与无效证据/project_maintenance/calculation_results_restore_20260729.json` | calculation_results_restore_20260729 | 2659 |
| — | `curated_project/04_历史与无效证据/project_maintenance/final_cleanup_audit_20260729.json` | final_cleanup_audit_20260729 | 1056 |
| — | `curated_project/04_历史与无效证据/project_maintenance/tmp_cleanup_20260729.json` | tmp_cleanup_20260729 | 427 |
| — | `curated_project/04_历史与无效证据/project_maintenance/todo0728.txt` | todo0728 | 3776 |
| — | `curated_project/04_历史与无效证据/project_maintenance/todo2.txt` | todo2 | 356 |
| — | `curated_project/04_历史与无效证据/project_maintenance/workspace_cleanup_manifest_20260729.json` | workspace_cleanup_manifest_20260729 | 23694 |
| — | `curated_project/04_历史与无效证据/root_archive/README.md` | 历史归档 | 329 |
| — | `curated_project/04_历史与无效证据/root_archive/__init__.py` | __init__ | 76 |
| — | `curated_project/04_历史与无效证据/root_archive/outer_lambda_exp010_exp011_legacy.py` | outer_lambda_exp010_exp011_legacy | 74241 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/1.diff.txt` | 1.diff | 10357 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/abfe_pipeline.py.pre_warmup_overlap_patch` | abfe_pipeline.py | 228319 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/abfe_preoptimizer.py.pre_warmup_overlap_patch` | abfe_preoptimizer.py | 77468 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/dexp_experiment.py.bak` | dexp_experiment.py | 109351 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/dexp_experiment1 - 副本.py` | dexp_experiment1 - 副本 | 167471 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/ibs_engine.py.pre_resume_gate_extraction` | ibs_engine.py | 710909 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/ibs_engine.py.pre_warmup_overlap_patch` | ibs_engine.py | 299285 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/proposed_audit_remaining_fixes_current.diff` | proposed_audit_remaining_fixes_current | 52374 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/proposed_audit_remaining_fixes_v1.diff` | proposed_audit_remaining_fixes_v1 | 49512 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/proposed_non_mutating_resume_fail_closed_v1.diff` | proposed_non_mutating_resume_fail_closed_v1 | 11496 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/proposed_thermodynamic_path_bigfix_v1.diff` | proposed_thermodynamic_path_bigfix_v1 | 51625 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/proposed_warmup_fixed_overlap_v2.diff` | proposed_warmup_fixed_overlap_v2 | 40820 |
| — | `curated_project/04_历史与无效证据/root_archive/patches/test_audit_protocol_regressions.py.pre_resume_gate_extraction` | test_audit_protocol_regressions.py | 47935 |
| — | `curated_project/04_历史与无效证据/root_archive/project_notes/新建文本文档.txt` | 新建文本文档 | 239 |
| — | `curated_project/04_历史与无效证据/root_legacy/#README.md` | ABFE-IBS workflow | 3653 |
| — | `curated_project/90_整理控制/README.md` | 整理控制与来源追溯 | 1434 |
| — | `curated_project/90_整理控制/SOURCE_MAP.csv` | SOURCE_MAP | 26417 |
| — | `curated_project/90_整理控制/tools/generate_curation_inventory.ps1` | generate_curation_inventory | 267 |
| — | `curated_project/90_整理控制/tools/generate_curation_inventory.py` | generate_curation_inventory | 3647 |

---

## 一、文档



开发期的过程材料（实验记录、交接单、审计快照、阶段性结论）与 `curated_project/`

整理版知识库。上面的主表是逐份登记。



### 逐字重复、未单独登记的材料

`curated_project/` 是 `docs/` 的整理版副本，两边有大量同内容文件。

| 重复件 | 与之相同的已登记件 |
|---|---|
| `curated_project/03_当前工作线/01_膜与带电配体/STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md` | `docs/experiments/STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md` |
| `curated_project/03_当前工作线/02_outer_lambda/IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md` | `docs/experiments/IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md` |
| `curated_project/03_当前工作线/02_outer_lambda/PLAN_outer_lambda_neural_basis.md` | `docs/experiments/PLAN_outer_lambda_neural_basis.md` |
| `curated_project/03_当前工作线/02_outer_lambda/TODO_outer_lambda_neural_basis_reframed.md` | `docs/experiments/TODO_outer_lambda_neural_basis_reframed.md` |
| `curated_project/03_当前工作线/02_outer_lambda/TODO_outer_lambda_orb_neural_basis.md` | `docs/experiments/TODO_outer_lambda_orb_neural_basis.md` |
| `curated_project/03_当前工作线/03_实验设计/DiffLift.MD` | `docs/experiments/DiffLift.MD` |
| `curated_project/03_当前工作线/03_实验设计/PLAN_EXP-018_stationarity_confirmation.md` | `docs/experiments/PLAN_EXP-018_stationarity_confirmation.md` |
| `curated_project/03_当前工作线/03_实验设计/PLAN_EXP-019_baseline_reproducibility_uncertainty_calibration.md` | `docs/experiments/PLAN_EXP-019_baseline_reproducibility_uncertainty_calibration.md` |
| `curated_project/03_当前工作线/03_实验设计/PLAN_EXP-025_local_manybody_cuda.md` | `docs/experiments/PLAN_EXP-025_local_manybody_cuda.md` |
| `curated_project/04_历史与无效证据/docs_archive/BOR-01-boresch-dihedral-sign-fixed-2026-07-29.md` | `docs/archive/BOR-01-boresch-dihedral-sign-fixed-2026-07-29.md` |
| `curated_project/04_历史与无效证据/docs_archive/TODO-2026-07-27-full.md` | `docs/archive/TODO-2026-07-27-full.md` |
| `curated_project/04_历史与无效证据/docs_archive/TODO-2026-07-28-completed.md` | `docs/archive/TODO-2026-07-28-completed.md` |
| `curated_project/04_历史与无效证据/docs_archive/todolist-2026-07-20.md` | `docs/archive/todolist-2026-07-20.md` |
| `curated_project/04_历史与无效证据/docs_archive/存档.md` | `docs/archive/存档.md` |
| `curated_project/04_历史与无效证据/docs_status/ABFE_FULL_AUDIT_2026-07-17.md` | `docs/status/ABFE_FULL_AUDIT_2026-07-17.md` |
| `curated_project/04_历史与无效证据/docs_status/AUDIT_STATUS.md` | `docs/status/AUDIT_STATUS.md` |
| `curated_project/04_历史与无效证据/docs_status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md` | `docs/status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md` |
| `curated_project/04_历史与无效证据/docs_status/NON_MUTATING_V1_STATUS.md` | `docs/status/NON_MUTATING_V1_STATUS.md` |
| `curated_project/04_历史与无效证据/docs_status/README_STATUS_SNAPSHOT_2026-07-29.md` | `docs/status/README_STATUS_SNAPSHOT_2026-07-29.md` |
| `curated_project/04_历史与无效证据/docs_status/RESULT_2026-07-27_atenolol_rank11.md` | `docs/status/RESULT_2026-07-27_atenolol_rank11.md` |
| `curated_project/04_历史与无效证据/docs_status/VALIDATION_MATRIX.md` | `docs/status/VALIDATION_MATRIX.md` |
| `curated_project/04_历史与无效证据/docs_status/VALIDATION_MATRIX.md.pre_resume_gate_extraction` | `docs/status/VALIDATION_MATRIX.md.pre_resume_gate_extraction` |
| `curated_project/04_历史与无效证据/docs_status/bug.md` | `docs/status/bug.md` |
| `curated_project/04_历史与无效证据/docs_status/evidence_2026-07-27/README.md` | `docs/status/evidence_2026-07-27/README.md` |
| `curated_project/04_历史与无效证据/docs_status/evidence_2026-07-27/charging_linear_response.json` | `docs/status/evidence_2026-07-27/charging_linear_response.json` |
| `curated_project/04_历史与无效证据/docs_status/evidence_2026-07-27/endpoint_sigma_diagnosis.json` | `docs/status/evidence_2026-07-27/endpoint_sigma_diagnosis.json` |
| `curated_project/04_历史与无效证据/docs_status/evidence_2026-07-27/pose_drift.json` | `docs/status/evidence_2026-07-27/pose_drift.json` |
| `curated_project/04_历史与无效证据/experiment_docs/DEXP_KERNEL_PHYSICS_ISSUES.md` | `docs/experiments/DEXP_KERNEL_PHYSICS_ISSUES.md` |
| `curated_project/04_历史与无效证据/experiment_docs/EXP-011_PREREGISTRATION.md` | `docs/experiments/EXP-011_PREREGISTRATION.md` |
| `curated_project/04_历史与无效证据/experiment_docs/dexp_experiment.md` | `docs/experiments/dexp_experiment.md` |
| `curated_project/04_历史与无效证据/handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md` | `docs/handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md` |
| `curated_project/04_历史与无效证据/handoffs/CHARGE_TRANSFER_B3_HANDOFF.md` | `docs/handoffs/CHARGE_TRANSFER_B3_HANDOFF.md` |
| `curated_project/04_历史与无效证据/handoffs/MEMBRANE_SOLVENT_LEG_P013_HANDOFF.md` | `docs/handoffs/MEMBRANE_SOLVENT_LEG_P013_HANDOFF.md` |
| `curated_project/04_历史与无效证据/handoffs/POSE_SCAN_HANDOFF.md` | `docs/handoffs/POSE_SCAN_HANDOFF.md` |
| `curated_project/04_历史与无效证据/handoffs/RESUME_DEXP_SESSION.md` | `docs/handoffs/RESUME_DEXP_SESSION.md` |
| `curated_project/04_历史与无效证据/handoffs/VANISHING_WINDOW0_HANDOFF.md` | `docs/handoffs/VANISHING_WINDOW0_HANDOFF.md` |
| `curated_project/04_历史与无效证据/root_archive/patches/proposed_audit_remaining_fixes_v2.diff` | `curated_project/04_历史与无效证据/root_archive/patches/proposed_audit_remaining_fixes_current.diff` |
| `curated_project/04_历史与无效证据/root_legacy/memory.md` | `docs/status/memory.md` |
| `curated_project/90_整理控制/BATCH_PLAN.md` | `docs/curation/BATCH_PLAN.md` |
| `curated_project/90_整理控制/CONFLICTS.md` | `docs/curation/CONFLICTS.md` |
| `curated_project/90_整理控制/CURATION_LOG.md` | `docs/curation/CURATION_LOG.md` |
| `curated_project/90_整理控制/DOCUMENT_ROLE_MAP.csv` | `docs/curation/DOCUMENT_ROLE_MAP.csv` |
| `curated_project/90_整理控制/FILE_INVENTORY.csv` | `docs/curation/FILE_INVENTORY.csv` |
| `curated_project/90_整理控制/IMMUTABILITY_POLICY.md` | `docs/curation/IMMUTABILITY_POLICY.md` |
| `curated_project/90_整理控制/MOVE_MAP.csv` | `docs/curation/MOVE_MAP.csv` |
| `curated_project/90_整理控制/RESULT_REGISTRY.csv` | `docs/curation/RESULT_REGISTRY.csv` |

---

## 二、代码与脚本



### 研究模块与一次性实验脚本

| 类别 | 数量 | 说明 |
|---|---|---|
| `local_residual/` 研究模块 | 25 | softlift 训练/部署、student/teacher、MACE/ORB latent、grouped-density CV 等；生产只需要 `openmm_plugin` / `em_no_residual` / `geometry` |
| `exp012_xed/` | 5 | EXP-012 的 DEC-018 早期命名空间，已被 `local_residual` 取代 |
| 根目录实验模块 | 6 | `dexp_experiment.py`、`dexp_退役.py`（DEXP 已并入 `abfe_core`）、`exp029_protocol.py`、`exp030_protocol.py`、`exp030_analysis.py`、`exp030_joint_score.py`（EXP-030 已合并主线） |
| `scripts/` 一次性实验脚本 | 122 | `exp0XX_*`、`audit_*`、`benchmark_*`、`diag_*`、`reseal_*` 等 |
| 对应的研究测试 | 44 | 只测上述实验代码，随之移出 |
| `plugins/` 历史变体 | 4 | `_exp025_reconstructed`、`_exp026_a2_draft`、`_exp026_a2_nosync_probe`、`exp026_a3_energybuffer_probe`；live 版本是 `plugins/LocalManyBodyResidual/` |
| 文档校验工具 | 2 | `check_doc_crossrefs.py` 与 `_doc_conventions.py`——它们校验的 `docs/status/memtodolist` 与 `docs/experiments/EXP-*` 结构已不在本分支 |

### 随 `dexp_experiment.py` 一起移出的诊断工具

`tools/diagnostics/diagnose_window0_vdw_clash.py` 依赖 `dexp_experiment` 的三个
轨迹分析辅助函数（`load_analysis_traj`、`get_ligand_env_heavy_indices`、
`compute_pairwise_distances_nm`）。DEXP 并入 `abfe_core` 时只并了**势能解析形式**
（alpha/beta 参数化），这三个 harness 辅助函数没有跟着进 core，所以该工具在本分支
无法工作，随其依赖一起移出。原文在 `Atenolol-rank11`。

### `scripts/`：PBS 与运行脚本

第一轮清掉 122 个一次性实验脚本后，这里只剩 `README.md` 和
`examples/atenolol-rank11/` 下的 4 个 zsh 启动脚本。复核后 4 个**全部是死的**：

| 脚本 | 为什么不能用 |
|---|---|
| `experiments/run_dexp_longmd.zsh` | 调 `python dexp_experiment.py`，该文件已并入主线并移出 |
| `experiments/run_dexp_relabel_10ns.zsh` | 同上 |
| `pbs/run_abfe_lrc_fix_reset.pbs.zsh` | `--output ./output_lrc_fix`，该目录不在本分支 |
| `pbs/run_abfe_rank9_resume.pbs.zsh` | 见下面两条共性问题 |

四个共有的两个硬伤：

1. 都 `source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh`。那个安装的
   `envs/` 目录已经没了（环境早搬到 miniforge3），激活必然失败——正是
   `tests/run_offline_tests.sh` 头部注释里记过的同一个坑：脚本会在**一条命令都没跑**
   的情况下退出，表现成"入口失败"而不是"任务失败"。
2. 都硬绑本集群（`#PBS -l nodes=groupG/groupE`、`/home/apps/Modules`、
   `/home/ruigengji/modulefiles`），换机器用不上。

原文在 `Atenolol-rank11`。发布物不提供 PBS 模板；运行方式见
`docs/GETTING_STARTED.md`。

---

## 三、实验协议（`protocols/`）



### 为什么整体移出

上一轮曾保留两份"仍被引用"的：`EXP-012_preregistration.json`（只被
`docs/TODO.md` 的一段历史叙述提到）和 `EXP-020_preregistration.json`
（被 `tests/test_exp019_outer_lambda_accounting.py` 读取）。复核后两条都不成立为保留理由：

- 生产代码对 `protocols/` **零依赖**——`runabfe.py` / `abfe_pipeline.py` 里的
  `stage_protocol_keys`、`_analysis_stage_protocols` 都是运行时内部变量，
  跟这个目录无关。
- `tests/test_exp019_outer_lambda_accounting.py` 只做一件事：读 EXP-020 那份
  sealed JSON、断言它自己的四个字段（`freeze.status == "sealed"` 等）。
  它不验证任何生产行为，而它所属的 EXP-019/EXP-020 研究代码
  （softlift / outer_lambda 训练栈）已在上一轮移出。该测试随之移出。

26 份协议的登记表见上文《已移出的实验预注册协议》一节。原文在 `Atenolol-rank11`。

### 登记表

这些是 sealed 预注册 / 冻结协议 JSON，属于实验记录而不是发布物。
本分支只保留仍被现存测试引用的两份：`EXP-012_preregistration.json`、
`EXP-020_preregistration.json`。其余原文在 `Atenolol-rank11`。

**声明状态是协议自己写的字段，不代表该实验的结论已被独立验证。**

| 文件 | 声明状态 | 字节 |
|---|---|---|
| `EXP-011_augmentation_001_p127p5.json` | FROZEN_AFTER_FORMAL_RUN1_OVERLAP_FAILURE_BEFORE_AUGMENTATION | 2039 |
| `EXP-011_preregistration.json` | PREREGISTERED_NOT_STARTED | 2703 |
| `EXP-011_umbrella_sampling_plan.json` | FROZEN_AFTER_PILOT_BEFORE_FORMAL_RESULTS | 2432 |
| `EXP-017_preregistration.json` | sealed | 7477 |
| `EXP-018_preregistration.json` | SEALED | 7354 |
| `EXP-019_analysis_freeze.json` | SEALED | 2856 |
| `EXP-019_analysis_freeze_v2.json` | SEALED | 2915 |
| `EXP-019_analysis_freeze_v3.json` | SEALED | 3539 |
| `EXP-019_baseline_sampling_addendum.json` | SEALED | 7040 |
| `EXP-019_baseline_sampling_addendum_v2.json` | SEALED | 8622 |
| `EXP-019_baseline_sampling_addendum_v3.json` | SEALED | 9361 |
| `EXP-019_preregistration.json` | SEALED | 9855 |
| `EXP-021_d0_frame_manifest.json` | SEALED | 2419 |
| `EXP-021_preregistration.json` | SEALED | 6658 |
| `EXP-025_G4_preregistration.json` | SEALED_BEFORE_COST_RESULTS_KNOWN | 9835 |
| `EXP-027_online_utility_preregistration.json` | EXP027_PREREGISTERED_READY_FOR_U1 | 37913 |
| `EXP-027_U4_stage2_confirmation_addendum_DRAFT.json` | DRAFT_NOT_PREREGISTERED | 9903 |
| `EXP-029_production_ab.json` | FROZEN_PRODUCTION_AUTHORIZED | 3322 |
| `EXP-030_extended_sampling_diagnostic_windows123_1M.json` | FROZEN_PRODUCTION_AUTHORIZED | 6177 |
| `EXP-030_joint_state_score_preregistration_DRAFT.json` | DRAFT_NOT_AUTHORIZED | 4326 |
| `EXP-030_joint_state_score_preregistration_FROZEN_PRODUCTION.json` | FROZEN_PRODUCTION_AUTHORIZED | 5494 |
| `EXP-030_joint_state_score_preregistration_FROZEN_SMOKE.json` | FROZEN_SMOKE_ONLY | 5484 |
| `EXP-030_joint_state_score_preregistration_FROZEN_SMOKE_R03.json` | FROZEN_SMOKE_ONLY | 5484 |
| `EXP-030_joint_state_score_preregistration.schema.json` | — | 7945 |

---

## 四、模型资源（`resources/`）



### 为什么移出：residual sampling 不随首发

`resources/outer_lambda_local_residual/`
（`manifest.json` + `r1_model_payload_v1.json` + `r1_model_weights_f64.bin`，共 44K）
移出本工程区分支。

**原因不是"占地方"，是这个组合在 release 里说不通。** 冻结 R1 模型的 manifest
硬绑 Atenolol 的 41 个原子与具体内部键图（`identity_protocol =
local_atomic_numbers_and_internal_bond_graph_v1`，带 fingerprint 校验），
换体系用不上；而"换体系重训"需要的训练/部署栈——`softlift*`、`student*`、
`teacher_graph`、`loss`、`atom_mapping`、`environment`、`mace_*`、`orb_*`
共 25 个模块，加上 `train_exp012_local_residual_student.py`、
`train_exp019_softlift_loro.py`、`build_exp012_student_training_dataset.py`、
`build_exp012_teacher_latent_cache.py` 等约 20 个脚本——已在第一轮作为研究代码移出。

留着资源就等于发布一个"能打开、只能跑 Atenolol、且永远没法换体系"的开关。
用户决定：**residual sampling 不随首发**。

#### 仍然保留的部分

- `local_residual/openmm_plugin.py`（加载器）和原生 CUDA 插件
  `plugins/LocalManyBodyResidual/` 都还在，仍可导入。
- `outer_lambda_local_residual_ibs` 默认 `false`，默认生产路径完全不受影响
  （实测 `import runabfe` 正常）。
- 打开开关时 **fail closed**：`_load_resource_manifest` 新增缺文件守卫，
  抛 `RESOURCE_MISSING_HINT`，说明这份资源是什么、为什么不随包、去
  `Atenolol-rank11` 取什么、换配体为什么不能只换 manifest、不用时怎么办。
  原来只会抛一个裸 `FileNotFoundError`（只有路径），读起来像"装坏了"。
- `tests/test_outer_lambda_local_residual_runtime.py` **没有删**：三条真正加载
  模型的用例改成缺资源时 skip（`requires_frozen_r1_resource`），把资源拷回来
  就该重新变绿；另加一条 `test_missing_frozen_resource_fails_closed_with_an_actionable_message`
  锁住上面那段错误信息的内容。

---

## 五、遗留说明



### 源码注释里指向已移出文档的出处引用

清理后仓库里仍有 **60 处**源码注释/docstring 引用 `docs/status/…`（41 处）、
`docs/experiments/…`（18 处）和 `reports/…`（1 处）作为出处溯源，例如：

```python
# 膜体系协议：system_type + 膜恒压器（docs/status/memtodolist.md §3.1 / §3.2；MEM-00i）
# 🔑 [2026-08-27，见 docs/experiments/EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md]
```

**这些路径在本工程区分支已不存在**，指向的原文在 `Atenolol-rank11`，本表上面已逐份登记。

刻意没有逐条改写它们：这些注释解释的是"这段代码为什么写成这样"，
把出处换成一个泛指链接会丢掉具体是哪份材料、哪一节；而在 1.1 MB 的
`ibs_engine.py` 和 0.5 MB 的 `abfe_core.py` 里批量改 60 处注释，是零收益的
高风险 churn。要查某处出处，用文件名到本表里定位，再回 `Atenolol-rank11` 取原文。

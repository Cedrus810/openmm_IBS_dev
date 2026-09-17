"""EXP-010 / EXP-011 的 CLI 表面 —— 2026-09-17 从 `outer_lambda_neural_basis.py` 移出。

**这个文件不被任何东西 import，也不该被 import。** 留着只为「删了什么」有出处。

## 为什么删

这些函数全是**转发壳**，转发目标 `archive.outer_lambda_exp010_exp011_legacy`
**从来没有进过本仓库**：`archive/` 目录不存在、`git log --all -- archive/` 是空的。
壳是 2026-08-31 `169514e`（主线迁移 / 发布清理）引入的，真正的 1718 行实现留在了
旧工地 `/home/ruigengji/ABFE_IBS/Atenolol-rank11/archive/outer_lambda_exp010_exp011_legacy.py`。

⟹ 这 9 个子命令从迁移那天起就是 `ImportError`，一次都没能跑起来过。
13 个壳在主线与测试里的外部引用数是 **0**（`_periodic_fourier_wavevectors` 连文件内都没人调）。

## 删掉的 9 个子命令

`exp011-coverage` `exp011-fit-pmf` `exp011-umbrella-sample` `exp011-reweight-umbrella`
`exp010-label` `exp010-fit` `exp010-prepare-selection` `freeze-slow-variable`
`sample-hard-window-scratch`

⚠️ 同期的 `screen-slow-variables` / `compare-slow-variable-screens` / `wp0-select`
**没有**删 —— 它们不碰这些壳，仍然能跑。

## 要恢复的话

把旧工地那份 legacy 搬进来纳管，再把下面的代码贴回去。但先问一句 EXP-010/011
是不是还需要 —— 两个实验都已结案。
"""

# ruff: noqa


# ==========================================================================
# shim ×6 (freeze_slow_variable_manifest … _periodic_fourier_wavevectors)
# 原 outer_lambda_neural_basis.py 行 6310-6369
# ==========================================================================

# def freeze_slow_variable_manifest(comparison_report: Mapping[str, Any], difficult_window: Mapping[str, Any], *, replicate_rank: int=1, experiment_id: str='EXP-010') -> dict[str, Any]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         freeze_slow_variable_manifest as _legacy_implementation,
#     )
#
#     return _legacy_implementation(comparison_report, difficult_window, replicate_rank=replicate_rank, experiment_id=experiment_id)
#
#
# def torsion_coordinate_gradient_radians(positions_nm: Sequence[Sequence[float]], atom_indices: Sequence[int], *, box_vectors_nm: Sequence[Sequence[float]] | None=None, displacement_nm: float=1e-06) -> tuple[tuple[float, float, float], ...]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         torsion_coordinate_gradient_radians as _legacy_implementation,
#     )
#
#     return _legacy_implementation(positions_nm, atom_indices, box_vectors_nm=box_vectors_nm, displacement_nm=displacement_nm)
#
#
# def build_exp010_protein_only_selection(selection_meta: Mapping[str, Any], topology: Any) -> dict[str, Any]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         build_exp010_protein_only_selection as _legacy_implementation,
#     )
#
#     return _legacy_implementation(selection_meta, topology)
#
#
# def project_force_onto_torsion(forces_kj_mol_nm: Sequence[Sequence[float]], torsion_gradient_radian_per_nm: Sequence[Sequence[float]]) -> dict[str, float]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         project_force_onto_torsion as _legacy_implementation,
#     )
#
#     return _legacy_implementation(forces_kj_mol_nm, torsion_gradient_radian_per_nm)
#
#
# def build_exp010_teacher_dataset(adapter: Any, frame_records: Iterable[Mapping[str, Any]], slow_variable_manifest: Mapping[str, Any], *, ligand_indices: Sequence[int], environment_indices: Sequence[int], atomic_numbers: Sequence[int], energy_offset_kj_mol: float | None, include_secondary: bool=True) -> dict[str, Any]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         build_exp010_teacher_dataset as _legacy_implementation,
#     )
#
#     return _legacy_implementation(adapter, frame_records, slow_variable_manifest, ligand_indices=ligand_indices, environment_indices=environment_indices, atomic_numbers=atomic_numbers, energy_offset_kj_mol=energy_offset_kj_mol, include_secondary=include_secondary)
#
#
# def _periodic_fourier_wavevectors(dimensions: int, order: int) -> list[tuple[int, ...]]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         _periodic_fourier_wavevectors as _legacy_implementation,
#     )
#
#     return _legacy_implementation(dimensions, order)
#
#


# ==========================================================================
# shim ×4 (fit_periodic_fourier_distillation … run_hard_window_scratch_trajectory)
# 原 outer_lambda_neural_basis.py 行 6411-6450
# ==========================================================================

# def fit_periodic_fourier_distillation(teacher_dataset: Mapping[str, Any], *, dimensions: int=1, order: int=4, ridge: float=1e-06, conditional_bins: int=24) -> dict[str, Any]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         fit_periodic_fourier_distillation as _legacy_implementation,
#     )
#
#     return _legacy_implementation(teacher_dataset, dimensions=dimensions, order=order, ridge=ridge, conditional_bins=conditional_bins)
#
#
# def build_periodic_fourier_openmm_force(model: Mapping[str, Any], *, force_group: int=0) -> Any:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         build_periodic_fourier_openmm_force as _legacy_implementation,
#     )
#
#     return _legacy_implementation(model, force_group=force_group)
#
#
# def build_exp011_periodic_umbrella_force(atom_indices: Sequence[int], *, center_degrees: float, force_constant_kj_mol_radian2: float, force_group: int=31):
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         build_exp011_periodic_umbrella_force as _legacy_implementation,
#     )
#
#     return _legacy_implementation(atom_indices, center_degrees=center_degrees, force_constant_kj_mol_radian2=force_constant_kj_mol_radian2, force_group=force_group)
#
#
# def run_hard_window_scratch_trajectory(baseline_root: str | Path, output_dir: str | Path, *, window_index: int=0, initial_trajectory_path: str | Path | None=None, burnin_steps: int=10000, sampling_steps: int=100000, report_interval_steps: int=500, platform_name: str='CUDA', random_seed: int=20260731, umbrella_torsion_atom_indices: Sequence[int] | None=None, umbrella_center_degrees: float | None=None, umbrella_force_constant_kj_mol_radian2: float | None=None, umbrella_run_id: str | None=None, minimize_max_iterations: int=200) -> dict[str, Any]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         run_hard_window_scratch_trajectory as _legacy_implementation,
#     )
#
#     return _legacy_implementation(baseline_root, output_dir, window_index=window_index, initial_trajectory_path=initial_trajectory_path, burnin_steps=burnin_steps, sampling_steps=sampling_steps, report_interval_steps=report_interval_steps, platform_name=platform_name, random_seed=random_seed, umbrella_torsion_atom_indices=umbrella_torsion_atom_indices, umbrella_center_degrees=umbrella_center_degrees, umbrella_force_constant_kj_mol_radian2=umbrella_force_constant_kj_mol_radian2, umbrella_run_id=umbrella_run_id, minimize_max_iterations=minimize_max_iterations)
#
#


# ==========================================================================
# 9 个子命令的 argparse 定义 (freeze-slow-variable … sample-hard-window-scratch)
# 原 outer_lambda_neural_basis.py 行 7251-7410
# ==========================================================================

#     slow_freeze_parser = subparsers.add_parser(
#         "freeze-slow-variable",
#         help="把已通过三种子门的周期候选冻结为指定实验输入 manifest",
#     )
#     slow_freeze_parser.add_argument("--comparison", required=True)
#     slow_freeze_parser.add_argument("--final-results", required=True)
#     slow_freeze_parser.add_argument("--replicate-rank", type=int, default=1)
#     slow_freeze_parser.add_argument(
#         "--experiment",
#         choices=("exp010", "exp011"),
#         default="exp010",
#     )
#     slow_freeze_parser.add_argument("-o", "--output")
#
#     exp011_coverage_parser = subparsers.add_parser(
#         "exp011-coverage",
#         help="按冻结协议诊断三条 run 的周期覆盖、重叠和有效样本数",
#     )
#     exp011_coverage_parser.add_argument("--protocol", required=True)
#     exp011_coverage_parser.add_argument(
#         "--trajectory", action="append"
#     )
#     exp011_coverage_parser.add_argument(
#         "--screen-report",
#         action="append",
#         help="可重复提供已有 candidate_screen JSON，避免重新读取 DCD",
#     )
#     exp011_coverage_parser.add_argument("--topology", required=True)
#     exp011_coverage_parser.add_argument("--manifest", required=True)
#     exp011_coverage_parser.add_argument(
#         "--frames", default="all", help="all 或每条轨迹共用的 frame spec"
#     )
#     exp011_coverage_parser.add_argument("-o", "--output")
#
#     exp011_fit_parser = subparsers.add_parser(
#         "exp011-fit-pmf",
#         help="从显式目标权重样本拟合周期 PMF，并执行整条 run 留一硬门",
#     )
#     exp011_fit_parser.add_argument("--protocol", required=True)
#     exp011_fit_parser.add_argument("--dataset", required=True)
#     exp011_fit_parser.add_argument("-o", "--output")
#
#     exp011_umbrella_parser = subparsers.add_parser(
#         "exp011-umbrella-sample",
#         help="在完整困难窗口 MM System 上运行单个周期 torsion umbrella window",
#     )
#     exp011_umbrella_parser.add_argument("--baseline-root", required=True)
#     exp011_umbrella_parser.add_argument("--manifest", required=True)
#     exp011_umbrella_parser.add_argument("--protocol", required=True)
#     exp011_umbrella_parser.add_argument("--output-dir", required=True)
#     exp011_umbrella_parser.add_argument("--run-id", required=True)
#     exp011_umbrella_parser.add_argument("--center-degrees", required=True, type=float)
#     exp011_umbrella_parser.add_argument(
#         "--force-constant-kj-mol-radian2", type=float, default=100.0
#     )
#     exp011_umbrella_parser.add_argument("--initial-trajectory")
#     exp011_umbrella_parser.add_argument("--burnin-steps", type=int, default=1000)
#     exp011_umbrella_parser.add_argument(
#         "--minimize-max-iterations", type=int, default=200
#     )
#     exp011_umbrella_parser.add_argument("--sampling-steps", type=int, default=5000)
#     exp011_umbrella_parser.add_argument("--report-interval-steps", type=int, default=500)
#     exp011_umbrella_parser.add_argument("--platform", default="Reference")
#     exp011_umbrella_parser.add_argument("--seed", type=int, default=20260802)
#     exp011_umbrella_parser.add_argument("-o", "--output")
#
#     exp011_reweight_parser = subparsers.add_parser(
#         "exp011-reweight-umbrella",
#         help="对多个 umbrella window 去相关并用 MBAR 导出目标权重",
#     )
#     exp011_reweight_parser.add_argument("--protocol", required=True)
#     exp011_reweight_parser.add_argument(
#         "--input", required=True, action="append", help="可重复提供 umbrella report JSON"
#     )
#     exp011_reweight_parser.add_argument("--minimum-neighbor-overlap", type=float, default=0.03)
#     exp011_reweight_parser.add_argument("--output-dataset", required=True)
#     exp011_reweight_parser.add_argument("-o", "--output")
#
#     exp010_label_parser = subparsers.add_parser(
#         "exp010-label",
#         help="用冻结 MACE 为慢变量轨迹生成能量和广义力教师数据集",
#     )
#     exp010_label_parser.add_argument("-c", "--config", required=True)
#     exp010_label_parser.add_argument("--manifest", required=True)
#     exp010_label_parser.add_argument(
#         "--trajectory", required=True, action="append"
#     )
#     exp010_label_parser.add_argument("--topology", required=True)
#     exp010_label_parser.add_argument("--selection-meta", required=True)
#     exp010_label_parser.add_argument(
#         "--frames",
#         default="::5",
#         help="每条轨迹独立应用的 frame spec；默认每 5 帧",
#     )
#     exp010_label_parser.add_argument(
#         "--device", choices=("cpu", "cuda"), default="cuda"
#     )
#     exp010_label_parser.add_argument(
#         "--primary-only", action="store_true"
#     )
#     exp010_label_parser.add_argument(
#         "--energy-offset-mode",
#         choices=("dataset_mean", "config"),
#         default="dataset_mean",
#     )
#     exp010_label_parser.add_argument(
#         "--support-violation-policy",
#         choices=("exclude", "reject"),
#         default="exclude",
#     )
#     exp010_label_parser.add_argument(
#         "--max-support-exclusion-fraction", type=float, default=0.05
#     )
#     exp010_label_parser.add_argument("-o", "--output")
#
#     exp010_selection_parser = subparsers.add_parser(
#         "exp010-prepare-selection",
#         help="从旧局部选择移除交换水，冻结 protein-only 教师环境",
#     )
#     exp010_selection_parser.add_argument("--selection-meta", required=True)
#     exp010_selection_parser.add_argument("--topology", required=True)
#     exp010_selection_parser.add_argument(
#         "--output-selection-meta", required=True
#     )
#     exp010_selection_parser.add_argument("-o", "--output")
#
#     exp010_fit_parser = subparsers.add_parser(
#         "exp010-fit",
#         help="拟合周期 1D/2D Fourier cheap-CV 并做整条 run 留一验证",
#     )
#     exp010_fit_parser.add_argument("--dataset", required=True)
#     exp010_fit_parser.add_argument(
#         "--dimensions", type=int, choices=(1, 2), default=1
#     )
#     exp010_fit_parser.add_argument("--order", type=int, default=4)
#     exp010_fit_parser.add_argument("--ridge", type=float, default=1.0e-6)
#     exp010_fit_parser.add_argument(
#         "--conditional-bins", type=int, default=24
#     )
#     exp010_fit_parser.add_argument("-o", "--output")
#
#     hard_window_parser = subparsers.add_parser(
#         "sample-hard-window-scratch",
#         help="只读历史 IBS 协议，在独立目录生成困难窗口 CV-screening 轨迹",
#     )
#     hard_window_parser.add_argument("--baseline-root", required=True)
#     hard_window_parser.add_argument("--output-dir", required=True)
#     hard_window_parser.add_argument("--window-index", type=int, default=0)
#     hard_window_parser.add_argument("--initial-trajectory")
#     hard_window_parser.add_argument("--burnin-steps", type=int, default=10_000)
#     hard_window_parser.add_argument(
#         "--sampling-steps", type=int, default=100_000
#     )
#     hard_window_parser.add_argument(
#         "--report-interval-steps", type=int, default=500
#     )
#     hard_window_parser.add_argument("--platform", default="CUDA")
#     hard_window_parser.add_argument("--seed", type=int, default=20260731)
#     hard_window_parser.add_argument("-o", "--output")
#


# ==========================================================================
# shim ×3 (assess_exp011_periodic_coverage … reweight_exp011_umbrella_reports)
# 原 outer_lambda_neural_basis.py 行 7485-7965
# ==========================================================================

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def assess_exp011_periodic_coverage(run_angles_degrees: Mapping[str, Sequence[float]], *, run_log_target_weights: Mapping[str, Sequence[float]] | None=None, run_statistical_inefficiencies: Mapping[str, float] | None=None, bins: int=24, minimum_runs: int=3, minimum_frames_per_run: int=500, minimum_effective_samples_per_run: float=25.0, minimum_occupied_fraction_per_run: float=0.5, minimum_effective_samples_per_pooled_bin: float=2.0, minimum_runs_per_bin: int=2, minimum_raw_samples_per_run_bin: int=3, minimum_pairwise_bhattacharyya: float=0.5, minimum_effective_samples_per_basin: float=5.0) -> dict[str, Any]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         assess_exp011_periodic_coverage as _legacy_implementation,
#     )
#
#     return _legacy_implementation(run_angles_degrees, run_log_target_weights=run_log_target_weights, run_statistical_inefficiencies=run_statistical_inefficiencies, bins=bins, minimum_runs=minimum_runs, minimum_frames_per_run=minimum_frames_per_run, minimum_effective_samples_per_run=minimum_effective_samples_per_run, minimum_occupied_fraction_per_run=minimum_occupied_fraction_per_run, minimum_effective_samples_per_pooled_bin=minimum_effective_samples_per_pooled_bin, minimum_runs_per_bin=minimum_runs_per_bin, minimum_raw_samples_per_run_bin=minimum_raw_samples_per_run_bin, minimum_pairwise_bhattacharyya=minimum_pairwise_bhattacharyya, minimum_effective_samples_per_basin=minimum_effective_samples_per_basin)
#
#
# def fit_exp011_reweighted_periodic_pmf(dataset: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         fit_exp011_reweighted_periodic_pmf as _legacy_implementation,
#     )
#
#     return _legacy_implementation(dataset, protocol)
#
#
# def reweight_exp011_umbrella_reports(reports: Sequence[Mapping[str, Any]], *, target_hamiltonian_id: str, minimum_neighbor_overlap: float=0.03) -> dict[str, Any]:
#     """已归档历史实现的延迟兼容入口；新实验请勿依赖。"""
#
#     from archive.outer_lambda_exp010_exp011_legacy import (
#         reweight_exp011_umbrella_reports as _legacy_implementation,
#     )
#
#     return _legacy_implementation(reports, target_hamiltonian_id=target_hamiltonian_id, minimum_neighbor_overlap=minimum_neighbor_overlap)


# ==========================================================================
# _run_cli_command 里 exp011-* / sample-hard-window-scratch 的处理块
# 原 outer_lambda_neural_basis.py 行 7969-8229
# ==========================================================================

#     if args.command in {"exp011-umbrella-sample", "exp011-reweight-umbrella"}:
#         protocol = _cli_read_json_mapping(args.protocol, "EXP-011 protocol")
#         stored_hash = protocol.get("protocol_sha256")
#         protocol_core = dict(protocol)
#         protocol_core.pop("protocol_sha256", None)
#         if (
#             protocol.get("protocol_type") != "outer_lambda_exp011_preregistration"
#             or not isinstance(stored_hash, str)
#             or stable_payload_sha256(protocol_core) != stored_hash
#         ):
#             raise NeuralPathIntegrityError("EXP-011 protocol 类型或内部 SHA-256 不匹配")
#         target = protocol.get("target")
#         if not isinstance(target, Mapping):
#             raise NeuralPathConfigError("EXP-011 protocol 缺少 target")
#
#         if args.command == "exp011-umbrella-sample":
#             manifest = _cli_read_json_mapping(args.manifest, "EXP-011 manifest")
#             if (
#                 manifest.get("status") != "frozen_for_exp011_complete_mm_pmf"
#                 or manifest.get("target_experiment") != "EXP-011"
#                 or manifest.get("production_approval") is not False
#             ):
#                 raise NeuralPathConfigError("manifest 不是冻结的 EXP-011 非生产 CV")
#             torsion = manifest.get("primary_slow_variable", {}).get("atom_indices")
#             if torsion != target.get("atom_indices"):
#                 raise NeuralPathIntegrityError("manifest torsion 与 EXP-011 protocol 不一致")
#             report = run_hard_window_scratch_trajectory(
#                 args.baseline_root,
#                 args.output_dir,
#                 window_index=0,
#                 initial_trajectory_path=args.initial_trajectory,
#                 burnin_steps=args.burnin_steps,
#                 sampling_steps=args.sampling_steps,
#                 report_interval_steps=args.report_interval_steps,
#                 platform_name=args.platform,
#                 random_seed=args.seed,
#                 umbrella_torsion_atom_indices=torsion,
#                 umbrella_center_degrees=args.center_degrees,
#                 umbrella_force_constant_kj_mol_radian2=(
#                     args.force_constant_kj_mol_radian2
#                 ),
#                 umbrella_run_id=args.run_id,
#                 minimize_max_iterations=args.minimize_max_iterations,
#             )
#             report.update(
#                 {
#                     "command": "exp011-umbrella-sample",
#                     "protocol_sha256": stored_hash,
#                     "protocol_file_sha256": sha256_file(args.protocol),
#                     "manifest_file_sha256": sha256_file(args.manifest),
#                 }
#             )
#             return report
#
#         reports = [
#             _cli_read_json_mapping(path, f"umbrella input[{index}]")
#             for index, path in enumerate(args.input)
#         ]
#         report = reweight_exp011_umbrella_reports(
#             reports,
#             target_hamiltonian_id=target.get("target_hamiltonian_id"),
#             minimum_neighbor_overlap=args.minimum_neighbor_overlap,
#         )
#         dataset = report.pop("dataset")
#         dataset.update(
#             {
#                 "protocol_sha256": stored_hash,
#                 "source_report_file_sha256": [sha256_file(path) for path in args.input],
#             }
#         )
#         _cli_write_json(dataset, args.output_dataset)
#         report.update(
#             {
#                 "ok": True,
#                 "command": "exp011-reweight-umbrella",
#                 "protocol_sha256": stored_hash,
#                 "output_dataset": str(Path(args.output_dataset).expanduser().resolve()),
#                 "output_dataset_sha256": sha256_file(args.output_dataset),
#             }
#         )
#         return report
#
#     if args.command in {"exp011-coverage", "exp011-fit-pmf"}:
#         protocol = _cli_read_json_mapping(args.protocol, "EXP-011 protocol")
#         stored_hash = protocol.get("protocol_sha256")
#         protocol_core = dict(protocol)
#         protocol_core.pop("protocol_sha256", None)
#         if (
#             protocol.get("protocol_type") != "outer_lambda_exp011_preregistration"
#             or not isinstance(stored_hash, str)
#             or stable_payload_sha256(protocol_core) != stored_hash
#         ):
#             raise NeuralPathIntegrityError(
#                 "EXP-011 protocol 类型或内部 SHA-256 不匹配"
#             )
#         if protocol.get("status") != "PREREGISTERED_NOT_STARTED":
#             raise NeuralPathConfigError(
#                 "EXP-011 protocol status 必须为 PREREGISTERED_NOT_STARTED"
#             )
#
#         if args.command == "exp011-fit-pmf":
#             dataset = _cli_read_json_mapping(args.dataset, "EXP-011 dataset")
#             report = fit_exp011_reweighted_periodic_pmf(dataset, protocol)
#             report.update(
#                 {
#                     "ok": True,
#                     "command": "exp011-fit-pmf",
#                     "protocol_path": str(Path(args.protocol).expanduser().resolve()),
#                     "protocol_file_sha256": sha256_file(args.protocol),
#                     "protocol_sha256": stored_hash,
#                     "dataset_path": str(Path(args.dataset).expanduser().resolve()),
#                     "dataset_file_sha256": sha256_file(args.dataset),
#                 }
#             )
#             return report
#
#         topology_path = Path(args.topology).expanduser()
#         manifest_path = Path(args.manifest).expanduser()
#         if not topology_path.is_file() or not manifest_path.is_file():
#             raise NeuralPathConfigError("EXP-011 topology/manifest 文件不存在")
#         manifest = _cli_read_json_mapping(manifest_path, "slow-variable manifest")
#         primary = manifest.get("primary_slow_variable")
#         atom_indices = primary.get("atom_indices") if isinstance(primary, Mapping) else None
#         if not isinstance(atom_indices, list) or len(atom_indices) != 4:
#             raise NeuralPathConfigError("slow-variable manifest 缺少 primary 四原子 torsion")
#         run_angles = {}
#         trajectory_records = []
#         frozen_g = None
#         if bool(args.trajectory) == bool(args.screen_report):
#             raise NeuralPathConfigError(
#                 "exp011-coverage 必须且只能提供 trajectory 或 screen-report"
#             )
#         if args.screen_report:
#             frozen_g = {}
#             primary_id = primary.get("stable_id")
#             for run_index, raw_report_path in enumerate(args.screen_report, start=1):
#                 report_path = Path(raw_report_path).expanduser()
#                 screen = _cli_read_json_mapping(report_path, "screen-report")
#                 candidates = screen.get("periodic_torsion_candidates")
#                 if not isinstance(candidates, list):
#                     raise NeuralPathConfigError("screen-report 缺少 periodic candidates")
#                 candidate = next(
#                     (item for item in candidates if item.get("stable_id") == primary_id),
#                     None,
#                 )
#                 if candidate is None:
#                     raise NeuralPathConfigError("screen-report 缺少冻结 primary torsion")
#                 histogram = candidate.get("torsion", {}).get("histogram", {})
#                 counts = histogram.get("counts")
#                 edges = histogram.get("bin_edges_degrees")
#                 if (
#                     not isinstance(counts, list)
#                     or not isinstance(edges, list)
#                     or len(edges) != len(counts) + 1
#                     or len(counts) != int(protocol["coverage"]["bins"])
#                 ):
#                     raise NeuralPathConfigError("screen-report histogram 与冻结 bins 不一致")
#                 angles = []
#                 for bin_index, count in enumerate(counts):
#                     center = 0.5 * (float(edges[bin_index]) + float(edges[bin_index + 1]))
#                     angles.extend([center] * int(count))
#                 run_id = f"run{run_index}:{report_path.parent.name}"
#                 run_angles[run_id] = angles
#                 frozen_g[run_id] = _finite_float(
#                     candidate.get("periodic_statistical_inefficiency"),
#                     "screen-report periodic g",
#                 )
#                 trajectory_records.append(
#                     {
#                         "run_id": run_id,
#                         "screen_report_path": str(report_path.resolve()),
#                         "screen_report_sha256": sha256_file(report_path),
#                         "selected_frame_count": len(angles),
#                         "statistics_source": "precomputed_exact_histogram_and_periodic_g",
#                     }
#                 )
#         else:
#             try:
#                 import mdtraj as md
#             except ImportError as exc:
#                 raise NeuralPathConfigError("trajectory 模式需要安装 mdtraj") from exc
#             trajectories = [Path(path).expanduser() for path in args.trajectory]
#             if any(not path.is_file() for path in trajectories):
#                 raise NeuralPathConfigError("EXP-011 trajectory 文件不存在")
#             for run_index, trajectory_path in enumerate(trajectories, start=1):
#                 with md.open(str(trajectory_path)) as handle:
#                     frame_count = len(handle)
#                 frame_indices = (
#                     tuple(range(frame_count))
#                     if args.frames == "all"
#                     else _resolve_trajectory_frame_spec(args.frames, frame_count)
#                 )
#                 selected = set(frame_indices)
#                 angles = []
#                 offset = 0
#                 for chunk in md.iterload(
#                     str(trajectory_path), top=str(topology_path), chunk=250
#                 ):
#                     values = md.compute_dihedrals(
#                         chunk, [list(map(int, atom_indices))], periodic=True
#                     )[:, 0]
#                     for local_index, value in enumerate(values):
#                         if offset + local_index in selected:
#                             angles.append(math.degrees(float(value)))
#                     offset += len(chunk)
#                 run_id = f"run{run_index}:{trajectory_path.parent.parent.name}"
#                 run_angles[run_id] = angles
#                 trajectory_records.append(
#                     {
#                         "run_id": run_id,
#                         "path": str(trajectory_path.resolve()),
#                         "sha256": sha256_file(trajectory_path),
#                         "total_frame_count": frame_count,
#                         "selected_frame_count": len(angles),
#                     }
#                 )
#         report = assess_exp011_periodic_coverage(
#             run_angles,
#             run_statistical_inefficiencies=frozen_g,
#             **dict(protocol["coverage"]),
#         )
#         report.update(
#             {
#                 "ok": True,
#                 "command": "exp011-coverage",
#                 "analysis_scope": "coverage_only_not_target_pmf_samples",
#                 "torsion_atom_indices": list(map(int, atom_indices)),
#                 "protocol_path": str(Path(args.protocol).expanduser().resolve()),
#                 "protocol_file_sha256": sha256_file(args.protocol),
#                 "protocol_sha256": stored_hash,
#                 "manifest_path": str(manifest_path.resolve()),
#                 "manifest_sha256": sha256_file(manifest_path),
#                 "topology_path": str(topology_path.resolve()),
#                 "topology_sha256": sha256_file(topology_path),
#                 "frame_spec": args.frames,
#                 "trajectories": trajectory_records,
#                 "pmf_prohibition": (
#                     "这些 IBS mixture scratch 轨迹没有逐帧目标态权重；只能判定 CV 覆盖，"
#                     "不得直接当作单一目标 Hamiltonian 的 PMF 样本"
#                 ),
#             }
#         )
#         return report
#
#     if args.command == "sample-hard-window-scratch":
#         report = run_hard_window_scratch_trajectory(
#             args.baseline_root,
#             args.output_dir,
#             window_index=args.window_index,
#             initial_trajectory_path=args.initial_trajectory,
#             burnin_steps=args.burnin_steps,
#             sampling_steps=args.sampling_steps,
#             report_interval_steps=args.report_interval_steps,
#             platform_name=args.platform,
#             random_seed=args.seed,
#         )
#         report["command"] = "sample-hard-window-scratch"
#         return report
#
#
#


# ==========================================================================
# _run_cli_command 里 freeze-slow-variable / exp010-* 的处理块
# 原 outer_lambda_neural_basis.py 行 8417-8765
# ==========================================================================

#     if args.command == "freeze-slow-variable":
#         comparison = _cli_read_json_mapping(
#             args.comparison, "comparison"
#         )
#         final_results = _cli_read_json_mapping(
#             args.final_results, "final-results"
#         )
#         selected_window = select_wp0_difficult_window(final_results)[
#             "selected_window"
#         ]
#         report = freeze_slow_variable_manifest(
#             comparison,
#             selected_window,
#             replicate_rank=args.replicate_rank,
#             experiment_id=args.experiment,
#         )
#         report.update(
#             {
#                 "ok": True,
#                 "command": "freeze-slow-variable",
#                 "comparison_path": str(
#                     Path(args.comparison).expanduser().resolve()
#                 ),
#                 "comparison_sha256": sha256_file(args.comparison),
#                 "final_results_path": str(
#                     Path(args.final_results).expanduser().resolve()
#                 ),
#                 "final_results_sha256": sha256_file(args.final_results),
#             }
#         )
#         return report
#
#     if args.command == "exp010-prepare-selection":
#         try:
#             import mdtraj as md
#         except ImportError as exc:
#             raise NeuralPathConfigError(
#                 "exp010-prepare-selection 需要安装 mdtraj"
#             ) from exc
#         topology_path = Path(args.topology).expanduser()
#         if not topology_path.is_file():
#             raise NeuralPathConfigError("exp010 topology 文件不存在")
#         topology = md.load(str(topology_path)).topology
#         source = _cli_read_json_mapping(
#             args.selection_meta, "selection-meta"
#         )
#         selection = build_exp010_protein_only_selection(source, topology)
#         _cli_write_json(selection, args.output_selection_meta)
#         return {
#             "ok": True,
#             "command": "exp010-prepare-selection",
#             "report_type": "outer_lambda_exp010_selection_preparation",
#             "report_version": 1,
#             "source_selection_meta": str(
#                 Path(args.selection_meta).expanduser().resolve()
#             ),
#             "source_selection_meta_sha256": sha256_file(args.selection_meta),
#             "output_selection_meta": str(
#                 Path(args.output_selection_meta).expanduser().resolve()
#             ),
#             "output_selection_meta_sha256": sha256_file(
#                 args.output_selection_meta
#             ),
#             "selection_policy": selection[
#                 "outer_lambda_exp010_selection_policy"
#             ],
#             "selection_protocol_sha256": selection["selection_sha256"],
#         }
#
#     if args.command == "exp010-label":
#         try:
#             import mdtraj as md
#         except ImportError as exc:
#             raise NeuralPathConfigError("exp010-label 需要安装 mdtraj") from exc
#         controller = load_neural_path_config(
#             args.config, verify_basis_files=True
#         )
#         if not controller.enabled or controller.basis_count != 1:
#             raise NeuralPathConfigError("exp010-label 要求启用且严格 M=1")
#         basis = controller.bases[0]
#         if basis.backend != "existing_openmmml" or not basis.model_name:
#             raise NeuralPathConfigError(
#                 "exp010-label 要求 existing_openmmml MACE/ORB basis"
#             )
#         manifest = _cli_read_json_mapping(args.manifest, "manifest")
#         stored_manifest_sha = manifest.get("manifest_sha256")
#         manifest_core = dict(manifest)
#         for key in (
#             "manifest_sha256",
#             "ok",
#             "command",
#             "comparison_path",
#             "comparison_sha256",
#             "final_results_path",
#             "final_results_sha256",
#         ):
#             manifest_core.pop(key, None)
#         if (
#             not isinstance(stored_manifest_sha, str)
#             or stable_payload_sha256(manifest_core) != stored_manifest_sha
#         ):
#             raise NeuralPathIntegrityError(
#                 "slow-variable manifest 内部 SHA-256 不匹配"
#             )
#         selection = _cli_read_json_mapping(
#             args.selection_meta, "selection-meta"
#         )
#         ligand_indices = selection.get("ligand_indices")
#         environment_indices = selection.get("env_indices")
#         if not isinstance(ligand_indices, list) or not isinstance(
#             environment_indices, list
#         ):
#             raise NeuralPathConfigError(
#                 "selection-meta 缺少 ligand_indices/env_indices"
#             )
#         if set(basis.atom_indices()) != set(ligand_indices).union(
#             environment_indices
#         ):
#             raise NeuralPathConfigError(
#                 "配置 atom selection 与 selection-meta 不一致"
#             )
#         topology_path = Path(args.topology).expanduser()
#         if not topology_path.is_file():
#             raise NeuralPathConfigError("exp010 topology 文件不存在")
#         trajectory_paths = [
#             Path(path).expanduser() for path in args.trajectory
#         ]
#         if any(not path.is_file() for path in trajectory_paths):
#             raise NeuralPathConfigError("exp010 trajectory 文件不存在")
#         reference = md.load_frame(
#             str(trajectory_paths[0]), 0, top=str(topology_path)
#         )
#         atomic_numbers = []
#         for atom in reference.topology.atoms:
#             if atom.element is None:
#                 raise NeuralPathConfigError(
#                     f"topology atom {atom.index} 缺少元素"
#                 )
#             atomic_numbers.append(int(atom.element.atomic_number))
#         run_specs = []
#         for run_index, path in enumerate(trajectory_paths):
#             with md.open(str(path)) as handle:
#                 frame_count = len(handle)
#             indices = _resolve_trajectory_frame_spec(args.frames, frame_count)
#             run_specs.append(
#                 {
#                     "run_id": f"run{run_index + 1}:{path.parent.name}",
#                     "path": path,
#                     "frame_count": frame_count,
#                     "frame_indices": indices,
#                 }
#             )
#
#         support_evaluations = []
#         exclusion_limit = _finite_float(
#             args.max_support_exclusion_fraction,
#             "max_support_exclusion_fraction",
#         )
#         if not 0.0 <= exclusion_limit < 1.0:
#             raise NeuralPathConfigError(
#                 "max_support_exclusion_fraction 必须位于 [0,1)"
#             )
#
#         def frame_records():
#             for run_spec in run_specs:
#                 indices = run_spec["frame_indices"]
#                 step = indices[1] - indices[0] if len(indices) > 1 else 0
#                 arithmetic = (
#                     step > 0
#                     and tuple(range(indices[0], indices[-1] + step, step))
#                     == tuple(indices)
#                     and indices[0] % step == 0
#                 )
#                 strided = (
#                     md.load(
#                         str(run_spec["path"]),
#                         top=str(topology_path),
#                         stride=step,
#                     )
#                     if arithmetic
#                     else None
#                 )
#                 for frame_index in indices:
#                     frame = (
#                         strided[frame_index // step]
#                         if strided is not None
#                         else md.load_frame(
#                             str(run_spec["path"]),
#                             frame_index,
#                             top=str(topology_path),
#                         )
#                     )
#                     box_vectors = (
#                         frame.unitcell_vectors[0].tolist()
#                         if frame.unitcell_vectors is not None
#                         else None
#                     )
#                     support = controller.evaluate_support_domains(
#                         frame.xyz[0].tolist(),
#                         box_vectors_nm=box_vectors,
#                     )
#                     supported = all(item.supported for item in support)
#                     support_record = {
#                         "run_id": run_spec["run_id"],
#                         "frame_index": frame_index,
#                         "supported": supported,
#                         "included_in_teacher_dataset": supported,
#                         "details": [
#                             item.payload() for item in support
#                         ],
#                     }
#                     support_evaluations.append(support_record)
#                     if not supported:
#                         if args.support_violation_policy == "reject":
#                             raise NeuralPathConfigError(
#                                 "EXP-010 source frame 超出冻结 MACE 支持域: "
#                                 f"{run_spec['run_id']} frame={frame_index}"
#                             )
#                         continue
#                     yield {
#                         "run_id": run_spec["run_id"],
#                         "frame_index": frame_index,
#                         "positions_nm": frame.xyz[0].tolist(),
#                         "box_vectors_nm": box_vectors,
#                     }
#
#         with ExistingOrbMaceBasisAdapter(
#             model_name=basis.model_name, device=args.device
#         ) as adapter:
#             report = build_exp010_teacher_dataset(
#                 adapter,
#                 frame_records(),
#                 manifest,
#                 ligand_indices=ligand_indices,
#                 environment_indices=environment_indices,
#                 atomic_numbers=atomic_numbers,
#                 energy_offset_kj_mol=(
#                     basis.energy_offset_kj_mol
#                     if args.energy_offset_mode == "config"
#                     else None
#                 ),
#                 include_secondary=not args.primary_only,
#             )
#         source_support_violation_count = sum(
#             not item["supported"] for item in support_evaluations
#         )
#         source_frame_count = len(support_evaluations)
#         exclusion_fraction = (
#             source_support_violation_count / source_frame_count
#             if source_frame_count
#             else 1.0
#         )
#         safety_violation_count = 0
#         if controller.safety is not None:
#             for sample in report["samples"]:
#                 if (
#                     abs(sample["teacher_centered_energy_kj_mol"])
#                     > controller.safety.max_abs_basis_energy_kj_mol
#                     or sample["teacher_max_force_kj_mol_nm"]
#                     > controller.safety.max_force_norm_kj_mol_nm
#                 ):
#                     safety_violation_count += 1
#         report["support_domain_violation_count"] = 0
#         report["source_support_domain_violation_count"] = (
#             source_support_violation_count
#         )
#         report["source_frame_count"] = source_frame_count
#         report["support_exclusion_fraction"] = exclusion_fraction
#         report["max_support_exclusion_fraction"] = exclusion_limit
#         report["support_violation_policy"] = args.support_violation_policy
#         report["safety_violation_count"] = safety_violation_count
#         report["qualified_for_fit"] = (
#             safety_violation_count == 0
#             and exclusion_fraction <= exclusion_limit
#         )
#         report["support_domain"] = support_evaluations
#         report.update(
#             {
#                 "ok": True,
#                 "command": "exp010-label",
#                 "config_path": str(Path(args.config).expanduser().resolve()),
#                 "config_sha256": sha256_file(args.config),
#                 "controller_protocol_sha256": controller.protocol_sha256(),
#                 "teacher_model_sha256": basis.sha256,
#                 "atom_selection_sha256": basis.atom_indices_sha256,
#                 "slow_variable_manifest_path": str(
#                     Path(args.manifest).expanduser().resolve()
#                 ),
#                 "slow_variable_manifest_file_sha256": sha256_file(
#                     args.manifest
#                 ),
#                 "slow_variable_manifest_protocol_sha256": stored_manifest_sha,
#                 "topology_path": str(topology_path.resolve()),
#                 "topology_sha256": sha256_file(topology_path),
#                 "selection_meta_path": str(
#                     Path(args.selection_meta).expanduser().resolve()
#                 ),
#                 "selection_meta_sha256": sha256_file(args.selection_meta),
#                 "frame_spec": args.frames,
#                 "trajectories": [
#                     {
#                         "run_id": spec["run_id"],
#                         "path": str(spec["path"].resolve()),
#                         "sha256": sha256_file(spec["path"]),
#                         "total_frame_count": spec["frame_count"],
#                         "selected_frame_indices": list(spec["frame_indices"]),
#                     }
#                     for spec in run_specs
#                 ],
#             }
#         )
#         return report
#
#     if args.command == "exp010-fit":
#         dataset = _cli_read_json_mapping(args.dataset, "dataset")
#         if dataset.get("qualified_for_fit") is not True:
#             raise NeuralPathConfigError(
#                 "teacher dataset 未通过 support/safety 门，拒绝拟合"
#             )
#         report = fit_periodic_fourier_distillation(
#             dataset,
#             dimensions=args.dimensions,
#             order=args.order,
#             ridge=args.ridge,
#             conditional_bins=args.conditional_bins,
#         )
#         dataset_sha = sha256_file(args.dataset)
#         model = report["model"]
#         model.pop("model_sha256", None)
#         model["training_dataset_sha256"] = dataset_sha
#         model["teacher_model_sha256"] = dataset.get(
#             "teacher_model_sha256"
#         )
#         model["slow_variable_manifest_protocol_sha256"] = dataset.get(
#             "slow_variable_manifest_protocol_sha256"
#         )
#         model["model_sha256"] = stable_payload_sha256(model)
#         report.update(
#             {
#                 "ok": True,
#                 "command": "exp010-fit",
#                 "dataset_path": str(
#                     Path(args.dataset).expanduser().resolve()
#                 ),
#                 "dataset_sha256": dataset_sha,
#             }
#         )
#         return report
#

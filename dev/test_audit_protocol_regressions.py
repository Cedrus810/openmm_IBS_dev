"""Regression tests for audit items #11, #14, #20 and #23."""

import math
import unittest
from pathlib import Path

import numpy as np

try:
    from abfe_preoptimizer import estimate_f_k_from_pilot_ti
    from abfe_pipeline import ABFEPipeline, _protocol_fingerprint
    from ibs_engine import (
        TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
        LJ_TAIL_LRC_R_SWITCH_NM,
        LJ_TAIL_LRC_R_CUTOFF_NM,
        _lj_tail_lrc_coefficients_kj_mol,
        _lj_softcore_tail_radial_integrals,
        _lj_switching_function_value,
        _periodic_box_volume_nm3,
    )
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # CPU-only lint hosts may not have OpenMM.
    estimate_f_k_from_pilot_ti = None
    ABFEPipeline = None
    _protocol_fingerprint = None
    TRADITIONAL_LJ_LRC_PROTOCOL_VERSION = None
    LJ_TAIL_LRC_R_SWITCH_NM = None
    LJ_TAIL_LRC_R_CUTOFF_NM = None
    _lj_tail_lrc_coefficients_kj_mol = None
    _lj_softcore_tail_radial_integrals = None
    _lj_switching_function_value = None
    _periodic_box_volume_nm3 = None
    _IMPORT_ERROR = exc


# Some HPC/network filesystems do not implement Windows final-path resolution
# (Path.resolve raises WinError 1005 even though ordinary reads work).  These
# source-contract tests only need a stable lexical absolute path.
ROOT = Path(__file__).absolute().parent


class SourceContractTests(unittest.TestCase):
    """Cheap contracts that run even on lint hosts without OpenMM/GPU."""

    @classmethod
    def setUpClass(cls):
        cls.engine = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
        cls.pipeline = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
        cls.preoptimizer = (ROOT / "abfe_preoptimizer.py").read_text(encoding="utf-8")
        cls.runabfe = (ROOT / "runabfe.py").read_text(encoding="utf-8")

    def test_wca_is_reweighted_as_sampling_bias(self):
        self.assertIn("groups={0, 2, 3, 5}", self.engine)
        self.assertIn("groups={1, 4}", self.engine)

    def test_lrc_enters_mbar_target_not_bias_training(self):
        self.assertIn(
            "target_energies = softcore_energies + lrc_energies", self.engine
        )
        self.assertIn("self.energy_buffer.append(bias_cv_energies)", self.engine)
        self.assertIn("self.energy_history.append(target_energies.copy())", self.engine)

    def test_nonfinite_frames_are_rejected(self):
        start = self.engine.index("frame_finite = (")
        block = self.engine[start:start + 500]
        self.assertIn("np.all(np.isfinite(bias_cv_energies))", block)
        self.assertIn("np.all(np.isfinite(target_energies))", block)
        self.assertIn("np.isfinite(e_base)", block)
        self.assertIn("np.isfinite(e_bias)", block)

    def test_bias_protocol_and_update_count_gate_are_locked(self):
        self.assertIn("IBS_BIAS_PROTOCOL_VERSION = 27", self.engine)
        self.assertIn(
            "IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS = frozenset((27, 28))",
            self.engine,
        )
        self.assertIn(
            "bias_protocol_match = _ibs_bias_protocol_version_is_cache_compatible(",
            self.engine,
        )
        self.assertIn("bias_update_count += 1", self.engine)
        self.assertIn("if f_updated is None:", self.engine)
        self.assertIn("_meets_minimum_with_roundoff(min_overlap", self.engine)
        self.assertIn("if len(sampler.f_history) < int(min_bias_updates):", self.engine)

    def test_safety_cap_accepts_only_completed_sane_best_effort(self):
        self.assertIn("def _best_effort_validation_is_acceptable(", self.engine)
        self.assertIn(
            'best_effort_acceptance_reason = "global_safety_cap"', self.engine
        )
        self.assertIn(
            "n_frames >= int(minimum_complete_frames)", self.engine
        )
        self.assertIn(
            "safety_cap_best_effort_tmbar_converged", self.engine
        )
        self.assertIn(
            "truncated_validation_frames_ignored", self.engine
        )
        self.assertIn(
            "sim.step(int(frozen_burn_in_steps))", self.engine
        )

    def test_production_warmup_is_bounded_and_residual_is_diagnostic(self):
        self.assertIn(
            "max_frozen_validation_cycles_before_accept_best: int = 1",
            self.engine,
        )
        self.assertIn(
            'best_effort_acceptance_reason = "bounded_warmup_budget"',
            self.engine,
        )
        self.assertIn(
            "and (not warmup_only or best_effort_within_sane_bound)",
            self.engine,
        )
        self.assertIn("占据残差仅记为效率诊断", self.engine)

    def test_dominant_flip_watchdog_ignores_near_flat_argmax_noise(self):
        self.assertIn("dominance_ratio = float(mean_p_batch[dominant_k]) * float(K)", self.engine)
        self.assertIn("dominance_ratio >= 1.5", self.engine)

    def test_physical_free_energy_seeds_are_not_sign_inverted(self):
        pilot_start = self.preoptimizer.index("def estimate_f_k_from_pilot_ti(")
        pilot_end = self.preoptimizer.index(
            "def partition_windows_by_delta_f_budget(", pilot_start
        )
        pilot_body = self.preoptimizer[pilot_start:pilot_end]
        self.assertNotIn("f_at_target = -f_at_target", pilot_body)

        tmbar_start = self.engine.index("    def _solve_tmbar_and_recenter(")
        tmbar_end = self.engine.index(
            "    def _bounded_log_occupancy_update(", tmbar_start
        )
        tmbar_body = self.engine[tmbar_start:tmbar_end]
        self.assertIn(
            "[f_by_lambda[k] for k in range(self.n_states)]", tmbar_body
        )
        self.assertNotIn(
            "[-f_by_lambda[k] for k in range(self.n_states)]", tmbar_body
        )

    def test_fixed_h_probe_cache_protocol_version_is_explicit(self):
        self.assertIn("FIXED_H_PROBE_CACHE_PROTOCOL_VERSION = 3", self.engine)

    def test_production_section_keeps_f_k_read_only(self):
        production = self.engine[
            self.engine.index("# ---------- 生产采样 ----------"):
            self.engine.index(
                "# ---------- 保存能量 ----------",
                self.engine.index("# ---------- 生产采样 ----------"),
            )
        ]
        # Mentions in explanatory comments are allowed; executable mutation or
        # recalibration calls are forbidden after the production boundary.
        executable_lines = [
            line.strip() for line in production.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertFalse(any("sampler.update_weights(" in line for line in executable_lines))
        self.assertFalse(any(".setParameter(" in line for line in executable_lines))
        self.assertFalse(any("_solve_tmbar" in line for line in executable_lines))
        self.assertFalse(any("_bounded_log_occupancy_update" in line for line in executable_lines))
        self.assertIn("production_f_k_lock", production)
        self.assertIn("生产阶段 update={up} 检测到 f_k 被修改", production)

    def test_validation_samples_are_not_carried_into_production(self):
        self.assertNotIn("carried_validation_energy_history", self.engine)
        self.assertNotIn("best_effort_carried_energy_history", self.engine)
        self.assertNotIn("validating_start_index", self.engine)
        self.assertIn("预热/冻结验证与生产严格隔离", self.engine)
        self.assertIn(
            "if resumed_production_checkpoint and prior_energy_history is not None:",
            self.engine,
        )

    def test_stage2_reports_and_repairs_exact_failing_windows(self):
        self.assertIn("def _stage_quality_failure_details(", self.pipeline)
        self.assertIn("worst_global_state", self.pipeline)
        self.assertIn("具体瓶颈：", self.pipeline)
        self.assertIn("stage2_production_rescue_rounds", self.pipeline)
        self.assertIn("production_rescue_targets", self.pipeline)
        self.assertIn("沿用各窗口 production checkpoint 与已锁定 f_k", self.pipeline)

    def test_stage2_bridge_rescue_is_immutable_and_separate(self):
        self.assertIn("def _build_vanishing_rescue_ranges(", self.pipeline)
        self.assertIn("stage2_enable_bridge_rescue", self.pipeline)
        self.assertIn("allow_partial_vanishing_rescue=True", self.pipeline)
        self.assertIn("immutable_bridge_rescue", self.pipeline)
        self.assertIn("excluded_local_windows=set(failing_windows)", self.pipeline)
        self.assertIn('"vanishing_rescue"', self.pipeline)

    def test_optimizer_failure_is_fail_closed(self):
        self.assertIn("热力学路径 pilot 失败；拒绝静默回退线性 λ 路径", self.pipeline)
        self.assertIn("Stage 2 自适应优化失败，拒绝静默回退线性路径", self.pipeline)

    def test_solvent_leg_has_explicit_salt_and_invalidates_pure_water_cache(self):
        self.assertIn("SOLVENT_CACHE_PROTOCOL_VERSION = 2", self.runabfe)
        self.assertIn("DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR = 0.15", self.runabfe)
        self.assertIn('positiveIon="Na+"', self.runabfe)
        self.assertIn('negativeIon="Cl-"', self.runabfe)
        self.assertIn("ionicStrength=ionic_strength_molar * unit.molar", self.runabfe)
        self.assertIn("neutralize=True", self.runabfe)
        self.assertIn('"solvent_cache_manifest.json"', self.runabfe)
        self.assertIn('manifest.get("na_count", 0)', self.runabfe)
        self.assertIn('manifest.get("cl_count", 0)', self.runabfe)

    def test_vanishing_uses_thermodynamic_few_state_subdomains_without_overlap_two(self):
        self.assertIn("THERMODYNAMIC_PATH_PROTOCOL_VERSION = 18", self.preoptimizer)
        self.assertIn(
            "redistribute_vanishing_lambda_subdomains(",
            self.preoptimizer,
        )
        self.assertIn("VANISHING_PROBE_BASE_STATE_COUNT = 17", self.preoptimizer)
        self.assertIn("VANISHING_FINAL_STATE_COUNT = 21", self.preoptimizer)
        self.assertIn("(17, 21)", self.preoptimizer)
        self.assertIn("pilot_fisher_17_plus_human_endpoint_4", self.preoptimizer)
        self.assertIn("probe_controls_base_lambda_placement", self.preoptimizer)
        self.assertIn("validate_human_vanishing_anchors_preserved", self.pipeline)
        self.assertIn("validate_single_shared_boundary_ranges", self.pipeline)
        redistribute_start = self.preoptimizer.index(
            "def redistribute_vanishing_lambda_subdomains("
        )
        redistribute_end = self.preoptimizer.index(
            "def partition_windows_by_thermodynamic_length(", redistribute_start
        )
        redistribute_body = self.preoptimizer[redistribute_start:redistribute_end]
        self.assertIn("redistribute_lambda_by_thermodynamic_length(", redistribute_body)
        self.assertIn("insert_human_vanishing_endpoint_lambdas(base_lambdas)", redistribute_body)
        self.assertIn('"total_window_state_slots": int(', redistribute_body)
        self.assertNotIn("_pilot_ti_cumulative_f", redistribute_body)
        self.assertNotIn("mean_dU_dlambda", redistribute_body)
        self.assertIn(
            '"ibs_ensemble_layout": "few_state_thermodynamic_subdomains"',
            self.preoptimizer,
        )
        self.assertIn('"sliding_overlap_states": 0', self.preoptimizer)
        serial_start = self.pipeline.rindex("if should_run_stage2:")
        serial_end = self.pipeline.index("            sampling = {", serial_start)
        serial_vanishing = self.pipeline[serial_start:serial_end]
        self.assertIn("stage2 = _run_stage2_once(", serial_vanishing)
        self.assertIn("expected_vanishing_ranges", serial_vanishing)
        self.assertNotIn("_run_stage_with_overlap_autorepair(", serial_vanishing)
        self.assertNotIn("_probe_vdw_window_fixed_overlap(", serial_vanishing)
        self.assertNotIn("generate_overlapping_windows(", serial_vanishing)

    def test_ibs_learning_uses_tmbar_and_full_non_mutating_budget(self):
        update_start = self.engine.index("    def update_weights(")
        update_end = self.engine.index("    def apply_learning_rate_penalty(", update_start)
        update_body = self.engine[update_start:update_end]
        self.assertIn("self._append_tmbar_batch_from_buffer()", update_body)
        self.assertIn("self._solve_tmbar_and_recenter(", update_body)
        self.assertIn("self._bounded_log_occupancy_update(", update_body)
        self.assertNotIn("ibs_lse_time_averaged_update(", update_body)
        self.assertNotIn("context.setParameter(f\"{self.prefix}_f_{k}\", float(_tmbar_absolute_candidate", update_body)
        self.assertIn("def _solve_tmbar_and_recenter(", self.engine)
        self.assertIn("self.tmbar_history: List[Dict[str, Any]] = []", self.engine)
        self.assertNotIn("def ibs_lse_time_averaged_update(", self.engine)
        self.assertIn("validation_probability_sum +=", self.engine)
        self.assertIn("validation_steps_this_freeze <", self.engine)
        self.assertIn("failed_frozen_cumulative_validation", self.engine)
        self.assertIn(
            "effective_mbar_calibration_reserved_steps = (",
            self.engine,
        )
        self.assertIn(
            "int(mbar_calibration_reserved_steps) if legacy_repair else 0",
            self.engine,
        )

    def test_cache_versions_and_hash_fingerprint_are_explicit(self):
        self.assertIn("PROTOCOL_FINGERPRINT_SCHEMA_VERSION = 1", self.pipeline)
        self.assertIn("TRADITIONAL_LJ_LRC_PROTOCOL_VERSION = 2", self.engine)
        self.assertIn('"lambda_path_fingerprint"', self.pipeline)

    def test_lrc_consumers_share_one_coefficient_array(self):
        # Production build site: the one place lj_tail_lrc_coeff_kj_mol is computed
        # for the ACE/dual_lambda path.
        self.assertIn(
            "ibs_wrapper.lj_tail_lrc_coeff_kj_mol = _lj_tail_lrc_coefficients_kj_mol(",
            self.engine,
        )
        # Both dual_lambda consumers (IBSSampler production sampling, fixed-H
        # overlap probe) read that exact same attribute -- not two
        # independently-recomputed values.
        self.assertIn(
            'getattr(self.ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None)', self.engine
        )
        self.assertIn(
            'getattr(ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None)', self.engine
        )
        # The offline traditional-path recompute (TraditionalMBARAnalyzer.compute_u_kn)
        # is a third, independent call site of the *same* helper function (not a
        # separately-maintained formula) -- two total call sites in the engine.
        self.assertEqual(
            self.engine.count("_lj_tail_lrc_coefficients_kj_mol("),
            3,  # 1 def + 2 call sites (dual_lambda build, traditional compute_u_kn)
        )
        self.assertIn('"lj_tail_lrc_coeff_kj_mol": (', self.engine)

    def test_traditional_lrc_producer_uses_v2_worker_key(self):
        self.assertNotIn('"lj_tail_prefactor_kj_nm3_mol": lj_tail_prefactor', self.engine)
        self.assertNotIn("_lj_tail_correction_S_kj_nm6(", self.engine)
        self.assertIn(
            "tail_s6, tail_s12 = _lj_tail_correction_moments_kj_nm6_nm12(",
            self.engine,
        )

    def test_pme_context_fallback_uses_closed_form_alpha(self):
        worker_start = self.engine.index("def _compute_u_kn_chunk(")
        worker_end = self.engine.index("def _split_platform_spec(", worker_start)
        worker = self.engine[worker_start:worker_end]
        self.assertIn("alpha_ewald = get_pme_alpha_for_system(eval_sys)", worker)
        self.assertNotIn(
            "alpha_q, _, _, _ = nb_force.getPMEParameters()",
            worker,
        )

    def test_tmbar_history_is_bounded_and_cap_is_persisted(self):
        self.assertIn("TMBAR_HISTORY_MAX_ENTRIES = 200", self.engine)
        self.assertIn("del self.tmbar_history[:overflow]", self.engine)
        self.assertIn('"tmbar_history_dropped_entries"', self.engine)

    def test_checkpoint_json_is_flushed_before_replace(self):
        start = self.engine.index("def _atomic_write_json(")
        end = self.engine.index("def _atomic_save_npz(", start)
        writer = self.engine[start:end]
        self.assertIn("handle.flush()", writer)
        self.assertIn("os.fsync(handle.fileno())", writer)
        self.assertLess(writer.index("os.fsync(handle.fileno())"), writer.index("os.replace("))

    def test_first_base_energy_failure_checks_positions_and_forces(self):
        self.assertIn("getPositions=True", self.engine)
        self.assertIn("getForces=True", self.engine)

    def test_owned_system_early_return_validates_swig_object(self):
        core = (ROOT / "abfe_core.py").read_text(encoding="utf-8")
        start = core.index("def ensure_owned_system(")
        end = core.index("def sync_all_exclusions(", start)
        block = core[start:end]
        self.assertLess(block.index("system.getNumParticles()"), block.index("return system"))

    def test_resume_rejects_mismatched_lrc_protocol_version(self):
        self.assertIn(
            'lrc_version_match = cached_conv.get("lj_tail_lrc_protocol_version") == '
            "TRADITIONAL_LJ_LRC_PROTOCOL_VERSION",
            self.engine,
        )
        self.assertIn("and lrc_version_match", self.engine)
        # Same gate must also apply to the λ-content window-reuse path (splitting/
        # inserting lambdas), not just the plain resume-skip path.
        self.assertIn(
            'conv.get("lj_tail_lrc_protocol_version") == TRADITIONAL_LJ_LRC_PROTOCOL_VERSION',
            self.pipeline,
        )

    def test_stage1_cdf_endpoint_is_clamped_to_one(self):
        # todolist.md P1: optimize_stage1_decharging's CDF interpolation used to
        # omit the xp[-1]=1.0 clamp that optimize_lambda_path_adaptive already
        # has, letting np.interp silently clamp the last 1-2 target points to
        # fp[-1]=0.0 (duplicate lambda_coul=0.0 states).
        idx = self.preoptimizer.index(
            "xp = np.concatenate(([0.0], cumulative_density[:-1] / total_density)).astype(float).ravel()"
        )
        block = self.preoptimizer[idx:idx + 200]
        self.assertIn("xp[-1] = 1.0", block)

    def test_analyze_only_apbs_correction_reads_provenance_not_raw_cli_attr(self):
        # todolist.md P1: --analyze-only used to read the lower-case CLI dest
        # apbs_correction_kj_mol (only set if re-passed on this invocation)
        # instead of the capital-J apbs_correction_kJ_mol resolved during the
        # formal run, silently dropping the correction on re-analysis.
        self.assertIn('"run_provenance.json"', self.runabfe)
        self.assertIn(
            '_provenance.get("config", {}).get("apbs_correction_kJ_mol", 0.0)',
            self.runabfe,
        )

    def test_analyze_only_window_files_are_numerically_sorted_and_contiguous(self):
        # todolist.md P1: the --analyze-only fallback used to sort
        # dual_window_*_{stage}_energies.npy as plain strings (window_10 sorts
        # before window_2) and used the enumerate() position as window_idx.
        self.assertIn(
            r'_window_idx_re = re.compile(rf"dual_window_(\d+)_{stage}_energies\.npy$")',
            self.runabfe,
        )
        self.assertIn("indexed_e_files.sort(key=lambda pair: pair[0])", self.runabfe)
        self.assertIn("parsed_indices != list(range(len(parsed_indices)))", self.runabfe)

    def test_analyze_only_fallback_fails_closed_on_unconverged_solve(self):
        # todolist.md P1: the --analyze-only fallback used to ignore
        # solve_stage_integrated's error/converged fields and default
        # total_delta_G to 0.0 for a failed/partial solve.
        self.assertIn('if res.get("error"):', self.runabfe)
        self.assertIn('if res.get("converged") is not True:', self.runabfe)

    def test_preopt_cache_requires_full_protocol_match_not_just_state_count(self):
        # todolist.md P1: Stage1/Stage2 preopt-cache (lambda path + window
        # layout) acceptance used to only check protocol fingerprint inside
        # is_verified_auto_repair (the changed-state-count branch), never in
        # the common "state count unchanged" branch -- switching potential
        # function/Boresch/decharge_method/system coords with the same state
        # count would silently reuse a stale-protocol lambda path.
        self.assertEqual(
            self.pipeline.count(
                "protocol_match = cached_protocol is not None and cached_protocol == _stage1_preopt_key"
            ),
            1,
        )
        self.assertEqual(
            self.pipeline.count(
                "protocol_match = cached_protocol is not None and cached_protocol == _stage2_preopt_key"
            ),
            1,
        )
        self.assertIn(
            "if protocol_match and (len(cached_lambdas) == stage1_states or is_verified_auto_repair):",
            self.pipeline,
        )
        self.assertIn(
            "if protocol_match and anchor_contract_match and (\n"
            "                        len(cached_lambdas) == stage2_states or is_verified_auto_repair\n"
            "                    ):",
            self.pipeline,
        )

    def test_final_convergence_gate_checks_absolute_ess_and_uncertainty(self):
        # todolist.md P1: GlobalMBARAnalyzer.solve_stage_integrated's converged
        # used to only require min ESS ratio >= 0.05 -- with very few total
        # samples, a single-digit absolute ESS could still pass a >5% ratio.
        conv_idx = self.engine.index("converged = bool(\n            len(local_results) == len(valid_windows)")
        conv_block = self.engine[conv_idx:conv_idx + 700]
        self.assertIn(
            "_meets_minimum_with_roundoff(\n"
            "                min_absolute_ess, float(final_min_absolute_ess)",
            conv_block,
        )
        self.assertIn(
            "min_decorrelated_samples >= int(final_min_decorrelated_samples)",
            conv_block,
        )
        self.assertIn(
            "_meets_maximum_with_roundoff(", conv_block
        )
        # The four threshold values that actually govern acceptance must be
        # folded into the stage protocol fingerprint, or changing them (via
        # kwarg or code-default) would silently fail to invalidate a
        # previously-"completed" stage-result cache.
        self.assertIn('"final_gate_thresholds": final_gate_thresholds,', self.pipeline)
        self.assertIn("final_gate_thresholds: Optional[Dict] = None,", self.pipeline)


@unittest.skipIf(_IMPORT_ERROR is not None, f"project runtime unavailable: {_IMPORT_ERROR}")
class ProtocolFingerprintTests(unittest.TestCase):
    def test_mapping_order_is_canonical(self):
        left = _protocol_fingerprint({"b": [2, 3], "a": {"y": 2, "x": 1}})
        right = _protocol_fingerprint({"a": {"x": 1, "y": 2}, "b": [2, 3]})
        self.assertEqual(left, right)
        self.assertEqual(len(left["sha256"]), 64)

    def test_any_lambda_or_window_change_invalidates_path(self):
        original = ABFEPipeline._lambda_path_fingerprint(
            [1.0, 0.5, 0.0], [(0, 3)]
        )
        changed_lambda = ABFEPipeline._lambda_path_fingerprint(
            [1.0, 0.49, 0.0], [(0, 3)]
        )
        changed_window = ABFEPipeline._lambda_path_fingerprint(
            [1.0, 0.5, 0.0], [(0, 2), (1, 3)]
        )
        self.assertNotEqual(original["sha256"], changed_lambda["sha256"])
        self.assertNotEqual(original["sha256"], changed_window["sha256"])

    def test_nonfinite_value_is_rejected(self):
        with self.assertRaises(ValueError):
            _protocol_fingerprint({"bad": float("nan")})


@unittest.skipIf(_IMPORT_ERROR is not None, f"project runtime unavailable: {_IMPORT_ERROR}")
class OverlapFailureClassificationTests(unittest.TestCase):
    @staticmethod
    def _valid_result():
        return {
            "total_delta_G": 1.0,
            "total_error": 0.1,
            "converged": False,
            "min_overlap": 0.01,
            "min_overlap_threshold": 0.05,
            "window_overlap_diagnostics": [{
                "window_index": 0,
                "lambdas": [0, 1, 2],
                "min_ess_ratio": 0.01,
                "ess_ratio_per_lambda": {0: 0.5, 1: 0.1, 2: 0.01},
            }],
        }

    def test_converged_false_alone_is_not_called_overlap(self):
        result = self._valid_result()
        result.pop("window_overlap_diagnostics")
        self.assertFalse(ABFEPipeline._is_overlap_failure(result))

    def test_complete_low_ess_diagnostics_are_overlap_failure(self):
        self.assertTrue(ABFEPipeline._is_overlap_failure(self._valid_result()))

    def test_nan_or_incomplete_diagnostics_are_not_auto_repaired(self):
        result = self._valid_result()
        result["total_error"] = float("nan")
        self.assertFalse(ABFEPipeline._is_overlap_failure(result))
        result = self._valid_result()
        result["window_overlap_diagnostics"][0]["ess_ratio_per_lambda"] = None
        self.assertFalse(ABFEPipeline._is_overlap_failure(result))


@unittest.skipIf(_IMPORT_ERROR is not None, f"project runtime unavailable: {_IMPORT_ERROR}")
class SwitchingAwareLJTailLRCTests(unittest.TestCase):
    """Correctness tests for the v2 switching+softcore-aware LJ tail LRC
    (_lj_tail_lrc_coefficients_kj_mol / _lj_softcore_tail_radial_integrals).
    """

    S6 = 12.5   # kJ*nm^6/mol, arbitrary but fixed test geometry moment
    S12 = 3.4   # kJ*nm^12/mol
    ALPHA_LJ = 0.5
    M_LJ = 2.0
    N_LJ = 2.0

    def test_lambda_zero_gives_exact_zero_lrc(self):
        coeffs = _lj_tail_lrc_coefficients_kj_mol(
            [0.0, 0.3, 1.0], self.S6, self.S12, self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )
        # Must be an exact 0.0, not merely close to it: lambda_vdw**n_lj is
        # exactly 0.0 in floating point for any n_lj > 0, and the integral is
        # skipped entirely for that state rather than computed and multiplied
        # by zero (see _lj_tail_lrc_coefficients_kj_mol's lambda==0.0 branch).
        self.assertEqual(coeffs[0], 0.0)
        self.assertNotEqual(coeffs[1], 0.0)
        self.assertNotEqual(coeffs[2], 0.0)

    def test_lambda_one_matches_independent_plain_lj_switching_integral(self):
        # At lambda_vdw=1, alpha_lj*(1-1)**m_lj == 0 exactly, so the softcore
        # denominator D(r) reduces to the plain r^6 -- this should equal the
        # standard (non-softcore) 12-6 LJ switching-aware tail correction.
        # Computed here via an independent inline integration (not by calling
        # any of the project's own helper functions) so this is a genuine
        # cross-check of the implementation, not a tautology.
        from scipy.integrate import quad
        r_switch, r_cutoff = LJ_TAIL_LRC_R_SWITCH_NM, LJ_TAIL_LRC_R_CUTOFF_NM

        def switch(r):
            if r <= r_switch:
                return 1.0
            if r >= r_cutoff:
                return 0.0
            x = (r - r_switch) / (r_cutoff - r_switch)
            return 1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5

        i6 = (
            quad(lambda r: (1.0 - switch(r)) * r ** 2 / r ** 6, r_switch, r_cutoff)[0]
            + quad(lambda r: r ** 2 / r ** 6, r_cutoff, np.inf)[0]
        )
        i12 = (
            quad(lambda r: (1.0 - switch(r)) * r ** 2 / r ** 12, r_switch, r_cutoff)[0]
            + quad(lambda r: r ** 2 / r ** 12, r_cutoff, np.inf)[0]
        )
        expected = 16.0 * math.pi * (self.S12 * i12 - self.S6 * i6)

        coeffs = _lj_tail_lrc_coefficients_kj_mol(
            [1.0], self.S6, self.S12, self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )
        self.assertTrue(math.isclose(coeffs[0], expected, rel_tol=1e-6))

    def test_switching_aware_result_differs_from_old_cutoff_only_r6(self):
        # v1 (superseded) formula: -(16*pi/3)*S6/rc^3 -- attractive-only,
        # cutoff-only, no switching region, no softcore/repulsive terms.
        rc = LJ_TAIL_LRC_R_CUTOFF_NM
        old_v1_coeff = -(16.0 * math.pi / 3.0) * self.S6 / rc ** 3
        new_coeff = _lj_tail_lrc_coefficients_kj_mol(
            [1.0], self.S6, self.S12, self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )[0]
        self.assertFalse(math.isclose(new_coeff, old_v1_coeff, rel_tol=1e-3))

    def test_softcore_denominator_changes_integrals_away_from_lambda_one(self):
        # At lambda_vdw < 1 the softcore shift alpha_lj*(1-lambda)^m_lj > 0
        # makes D(r) larger than plain r^6 everywhere, which must shrink the
        # magnitude of both radial integrals relative to the lambda=1 case.
        i6_full, i12_full = _lj_softcore_tail_radial_integrals(
            1.0, self.ALPHA_LJ, self.M_LJ,
        )
        i6_half, i12_half = _lj_softcore_tail_radial_integrals(
            0.5, self.ALPHA_LJ, self.M_LJ,
        )
        self.assertLess(i6_half, i6_full)
        self.assertLess(i12_half, i12_full)

    def test_invalid_box_fails_closed(self):
        with self.assertRaises(ValueError):
            _periodic_box_volume_nm3(np.zeros((3, 3)))

    def test_lrc_cache_version_is_explicit(self):
        self.assertGreaterEqual(TRADITIONAL_LJ_LRC_PROTOCOL_VERSION, 2)


if __name__ == "__main__":
    unittest.main()

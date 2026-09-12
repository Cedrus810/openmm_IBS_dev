"""Regression tests for audit items #11, #14, #20 and #23."""

import ast
import math
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import pytest

pytestmark = pytest.mark.cpu_only

try:
    from abfe_preoptimizer import estimate_f_k_from_pilot_ti
    from abfe_pipeline import ABFEPipeline, _protocol_fingerprint
    from ibs_engine import (
        IBSSampler,
        IBS_TMBAR_UPDATE_DAMPING,
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
    IBSSampler = None
    IBS_TMBAR_UPDATE_DAMPING = None
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
ROOT = Path(__file__).absolute().parents[1]


class SourceContractTests(unittest.TestCase):
    """Cheap contracts that run even on lint hosts without OpenMM/GPU."""

    @classmethod
    def setUpClass(cls):
        cls.engine = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
        cls.pipeline = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
        cls.preoptimizer = (ROOT / "abfe_preoptimizer.py").read_text(encoding="utf-8")
        cls.runabfe = (ROOT / "runabfe.py").read_text(encoding="utf-8")
        cls.core = (ROOT / "abfe_core.py").read_text(encoding="utf-8")

    def _extracted_resume_gate_source(self):
        """Source text of `_resume_cached_window_gate_status`'s body only.

        The resume-reuse gates were extracted out of `run_all_windows` into this
        module-level pure function so they can be unit-tested directly against a
        mock convergence.json (test_resume_reuse_contracts.py).  Source-contract
        assertions about those gates should be scoped to this function rather than
        to the whole 12k-line file: an unscoped `assertIn` would keep passing if a
        gate were deleted here but the same string happened to survive somewhere
        else (a comment, a docstring, a diagnostic message).

        Fails the test if the function is gone -- reverting the extraction without
        updating these contracts must be loud, not silently vacuous.
        """
        marker = "def _resume_cached_window_gate_status("
        self.assertIn(
            marker,
            self.engine,
            "resume-reuse gates must live in the module-level pure helper "
            "_resume_cached_window_gate_status (see test_resume_reuse_contracts.py)",
        )
        start = self.engine.index(marker)
        # Bound the slice at the function's own end: the first column-0 statement
        # that is not a comment.  Stopping at the next `def`/`class` instead would
        # drag in the ~135 lines of module constants that follow and make these
        # contracts looser than they look.
        # A column-0 line only ends the function if it actually *starts* a new
        # statement (identifier / `_` / decorator).  The multi-line signature's own
        # closing `) -> Dict[str, Any]:` also sits at column 0, so a naive
        # "not indented" test would cut the slice off after the signature and make
        # every assertion below vacuously fail.
        lines = self.engine[start:].splitlines(keepends=True)
        body = [lines[0]]
        for line in lines[1:]:
            stripped = line.strip()
            starts_statement = bool(stripped) and (line[0].isalnum() or line[0] in "_@")
            if starts_statement and not stripped.startswith("#"):
                break
            body.append(line)
        return "".join(body)

    def test_softcore_alpha_is_dimensionless_and_sigma_scaled(self):
        # 🔑 alpha_lj/alpha_coul 是无量纲 Beutler 系数，两套 softcore 构造器都必须在
        # 表达式里乘 sigma_ij^6 / sigma_ij^2。曾经把它们当绝对 nm^6/nm^2 直接相加，
        # 使软核偏移放大 ~685 倍（sigma≈0.3 nm），硬核→全软的全过程被压缩进
        # lambda_vdw∈[0.96,1]，vanishing 窗口 0 因此零重叠。
        self.assertIn('ALPHA_CONVENTION = "dimensionless_sigma_scaled_v2"', self.core)
        # ACESoftcorePotential.build_expression（dual_lambda 实际路径）
        self.assertIn(
            'f"max({self.alpha_lj}*{sigma12}^6*(1.0-{lam_vdw})^{self.m_lj} + r^6, 1e-6)"',
            self.core,
        )
        self.assertIn(
            'f"sqrt(max(r^2 + {self.alpha_coul}*{sigma12}^2"', self.core
        )
        # BeutlerSoftcoreBuilder.build（传统 single_lambda REMD 路径）
        self.assertIn(
            'f"(r^6 + {alpha_lj}*sigma12^6*(1-lambda_vdw)^{power_lj}"', self.core
        )
        self.assertIn(
            'f"sqrt(r^2 + {alpha_coul}*sigma12^2*(1-lambda_coul)^{power_coul}"',
            self.core,
        )
        # 绝对数值兜底项也必须按 sigma 缩放，否则 1e-4 nm^6 相对 sigma^6≈7.3e-4
        # 自己就成了一个 ~14% 的额外软化项。
        self.assertIn('f" + 1e-4*sigma12^6*(1-lambda_vdw))"', self.core)
        self.assertIn('f" + 1e-3*sigma12^2)"', self.core)
        # 旧的绝对形式必须已经绝迹。
        self.assertNotIn("{self.alpha_lj}*(1.0-{lam_vdw})", self.core)
        self.assertNotIn("{alpha_lj}*(1-lambda_vdw)", self.core)

    def test_alpha_convention_enters_protocol_fingerprint(self):
        # alpha 的数值没变（仍是 0.5/0.3），变的只是它乘不乘 sigma^6，所以指纹必须
        # 靠这个显式标签区分，否则旧缓存会被静默复用到新哈密顿量上。
        self.assertIn(
            '"alpha_convention": ACESoftcorePotential.ALPHA_CONVENTION', self.core
        )
        self.assertIn('"alpha_convention": self.ALPHA_CONVENTION', self.core)
        self.assertIn('convention != cls.ALPHA_CONVENTION', self.core)

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
        # 31->32（2026-08-31，0831issue.md P1）：修掉逐帧 e_offset 泄漏进
        # tmbar_history 的 u_kn。在线学习/冻结判定的输入变了，按 v30/v31 的既有
        # 处理方式，兼容集合收窄成只有 v32 自己。
        #
        # 32->33（2026-09-05，EXP-031 S1）：`IBS_BIAS_OMIT_ZERO_REST_CVS` 默认翻成
        # True —— Group-1 不再注册那 K 个恒零的 `cv_k_rest` 占位 CV。采样哈密顿量
        # 在数学上不变（`cv_k_rest` 是零粒子的 `CustomExternalForce("0")`，贡献严格
        # 为 0，三条独立路径实测 Group-1 能量逐比特相同），但**表达式字符串与 CV
        # 集合变了** ⟹ system XML 变 ⟹ 旧的 dual_window_*/ibs_state_* 的
        # `_int_cv_indices` 与 CV 计数都对不上，不能续用。同样按 v30/v31/v32 的
        # 处理方式把兼容集合收窄成只有 v33 自己。收益：每步省 0.377 ms
        # （真 Atenolol 膜体系整步 1.2206x）。
        #
        # ⚠️ 这两个常量必须**一起**改：只升版本号而不改兼容集合，会让新版本与它
        # 自己都判不兼容，resume 全线失效（本次实测 27 条测试因此变红，正是这条
        # 断言存在的意义）。
        self.assertIn("IBS_BIAS_PROTOCOL_VERSION = 33", self.engine)
        self.assertIn(
            "IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS = frozenset((33,))",
            self.engine,
        )
        self.assertIn(
            "bias_protocol_match = _ibs_bias_protocol_version_is_cache_compatible(",
            self.engine,
        )
        self.assertIn("bias_update_count += 1", self.engine)

    def test_production_entry_self_correction_stays_before_production_sampling(self):
        """docs/experiments/EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md's fix (in
        scripts/exp030_window_state_machine.py) is safe only as long as the
        one-time damped+pairwise-capped f_k self-correction on production
        entry stays structurally confined to the warmup/validation
        while-loop, strictly before the code that actually enters production
        sampling ("f_k 冻结、不再调用 update_weights()"). If a future refactor
        moved or duplicated the correction so it could fire after production
        sampling has already started, the exp030 fix's "any late mismatch is
        a legitimate one-time correction" assumption (bounded reconciliation
        in scripts/exp030_window_state_machine.py::
        _reconcile_frozen_snapshot_if_legitimate) would silently paper over
        real corruption instead of catching it. This only checks source
        *position*, not runtime behavior -- see
        tests/test_exp030_frozen_snapshot_reconciliation.py for the
        reconciliation function's own deterministic behavior tests.
        """
        correction_marker = "occupancy 尚可但 local-MBAR gap 未过：对冻结 f_k 应用一次"
        production_entry_marker = "---- 进入生产采样：不再改变 bias_scale/最小化/重新爬坡 ----"
        correction_pos = self.engine.find(correction_marker)
        production_entry_pos = self.engine.find(production_entry_marker)
        self.assertGreater(correction_pos, -1, "self-correction marker not found -- did its message text change?")
        self.assertGreater(production_entry_pos, -1, "production-entry marker not found -- did its message text change?")
        self.assertLess(
            correction_pos, production_entry_pos,
            "the f_k self-correction branch must stay textually before the production-sampling "
            "entry point, or the exp030 frozen-snapshot reconciliation fix's bounded-correction "
            "assumption no longer holds",
        )
        self.assertIn("_meets_minimum_with_roundoff(min_overlap", self.engine)
        # [Candidate-first, Validate-or-Learn v1] LEARN no longer has a fixed
        # min_bias_updates batch-count gate; freeze the instant the raw
        # occupancy residual drops to/below IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW.
        self.assertIn(
            "if residual_severity <= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW:", self.engine
        )
        self.assertNotIn("if f_updated is None:", self.engine)
        self.assertNotIn(
            "len(sampler.f_history) >= int(min_bias_updates)", self.engine
        )

    def test_budget_exhaustion_accepts_current_f_k_for_production(self):
        # [Candidate-first, Validate-or-Learn v1] Best-effort acceptance has
        # no place in a production protocol by default: the fallback code
        # path is left in place (a caller may still opt back in explicitly),
        # but allow_best_effort_warmup now defaults False, so a production
        # caller that doesn't pass it hard-fails instead of silently
        # promoting an unvalidated f_k when the budget is exhausted.
        self.assertIn(
            "if not bias_converged and not warmup_only and allow_best_effort_warmup:",
            self.engine,
        )
        self.assertIn("allow_best_effort_warmup: bool = False,", self.engine)
        self.assertIn(
            'best_effort_acceptance_reason = "warmup_budget_exhausted_loose_gate"',
            self.engine,
        )
        self.assertIn(
            'mode = "best_effort_budget_exhausted_accepted"', self.engine
        )

    def test_convergence_speedups_bootstrap_seed_and_bounded_cap_exemption(self):
        # [IBS_BIAS_PROTOCOL_VERSION=29] 加速收敛：(a) 冷启动无 pilot 时用本窗口首批
        # 每态平均 softcore 能量自举播 f_k=⟨u_k⟩，避免从 0 慢爬；(b) 硬 2 kT pairwise
        # cap 只加在可信绝对 TMBAR 路径，bounded occupancy 反馈改用其自适应上限
        # （严重塌陷区最高 ~10 kT），让大谱宽窗口快速建起 f_k。
        self.assertIn("[自举 TI 种子]", self.engine)
        self.assertIn("f_k_warm_started", self.engine)
        update_start = self.engine.index("    def update_weights(")
        update_end = self.engine.index("    def apply_learning_rate_penalty(", update_start)
        update_body = self.engine[update_start:update_end]
        # 硬 cap 只在 trusted 分支；bounded 分支显式不加外部硬 cap（None）。
        self.assertIn("if tmbar_candidate_trusted:", update_body)
        self.assertIn('weight_update_diag["hard_pairwise_cap_kJ_mol"] = None', update_body)

    def test_learning_to_freeze_freezes_only_when_step_settled(self):
        # [Candidate-first, Validate-or-Learn v1] LEARN controls occupancy
        # only: freeze the instant the raw log-occupancy residual from
        # _bounded_log_occupancy_update drops to/below
        # IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW -- no fixed min_bias_updates batch
        # count, no consecutive-pass streak, no "applied pairwise step
        # settled" proxy (that only proved the SGD step had shrunk, not that
        # the candidate had been proven against the real Hamiltonian; VALIDATE
        # -- the unchanged local-MBAR loose gate -- is the sole proof now).
        self.assertIn(
            "residual_severity = float(\n"
            "                        learn_diag.get(\"residual_severity\", float(\"inf\"))",
            self.engine,
        )
        self.assertIn(
            "if residual_severity <= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW:", self.engine
        )
        self.assertIn(
            "sampler._bounded_log_occupancy_update(\n"
            "                        f_old,\n"
            "                        mean_p_batch,",
            self.engine,
        )
        # 旧的"应用步长已稳定 + 连续通过 + 固定 min_bias_updates 批数"三件套必须
        # 已从 LEARN 冻结判据删除（曾是隐藏的、跟真实 Hamiltonian 无关的严格门）。
        self.assertNotIn("learning_ready = bool(", self.engine)
        self.assertNotIn(
            "consecutive_pass_count >= int(IBS_LEARNING_READY_CONSECUTIVE)",
            self.engine,
        )
        self.assertNotIn(
            "len(sampler.f_history) >= int(min_bias_updates)", self.engine
        )
        self.assertNotIn("IBS_LEARNING_READY_MAX_RESIDUAL_SEVERITY", self.engine)
        self.assertNotIn("IBS_LEARNING_READY_MIN_COVERAGE_ESS_FRACTION", self.engine)
        # 旧的"失败后只更新一次就重冻"逻辑必须已删除。
        self.assertNotIn(
            "updates_needed = int(min_bias_updates) if not have_frozen_once else 1",
            self.engine,
        )

    def test_frozen_convergence_uses_local_mbar_loose_gate(self):
        # [IBS_BIAS_PROTOCOL_VERSION=29] 冻结收敛判据换成局部滑窗 MBAR loose gate：
        # 相邻态 |Δf_k − ΔF^MBAR| < 阈值（gauge 无关），无连续通过 / 无 LSE 占据门 /
        # 无 50k→150k→300k 冻结验证阶梯 / 无 warmup ESS 四联门。
        self.assertIn(
            "IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL = 10.0", self.engine
        )
        self.assertIn("IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES = 5", self.engine)
        self.assertIn("gate_mbar = _solve_single_window_local_mbar(", self.engine)
        # 2026-09-11：拆成带符号 + 绝对值两步。带符号那份是端到端累计偏差门的输入 ——
        # 只存绝对值会在这一步丢掉符号，累计就只剩 Σ|r| 的上界，判不出"单向累积"
        # 还是"先偏后抵消"。实测 win3：Σ|r| 与 |Σr| 相等（零抵消）、span=11.47 超门，
        # 而单边最大仅 4.03、现有 max-adjacent 门完全看不见。
        self.assertIn("adjacent_gaps_signed = df_current - dF_mbar", self.engine)
        self.assertIn("adjacent_gaps = np.abs(adjacent_gaps_signed)", self.engine)
        self.assertIn('"phase": "frozen_local_mbar_loose_gate"', self.engine)
        # 现场诊断：饿死态/边 + global 索引 + 原始 softcore Δu，供预算耗尽接受时
        # 判"是可恢复的慢弛豫还是需要插 λ/拆窗的硬瓶颈"。
        self.assertIn("def _diagnose_local_mbar_situation(", self.engine)
        self.assertIn("gate_situation = _diagnose_local_mbar_situation(", self.engine)
        self.assertIn("starved_global_state", self.engine)

    def test_occupancy_early_exit_cannot_preempt_the_first_gate_evaluation(self):
        # 占据早退是纯路由优化，但它 continue 回 learning 的位置在完整 gate 求解
        # （validation_attempts += 1）之前。若它在本窗口首次完整评估之前就能命中，
        # 每个 freeze cycle 都会被它截胡，占据就事实上成了唯一的生产放行门——实测
        # 4/4 失败的 vanishing 窗口 validation_attempts≈0 即由此而来，且失败后本该
        # 接手的 3-way router / damped 直接重验分支从未执行过。故 gate 至少被真正
        # 评估一次之前不得早退。
        start = self.engine.index('# mode == "validating"')
        end = self.engine.index("gate_mbar = _solve_single_window_local_mbar(", start)
        validate_block = self.engine[start:end]
        guard = validate_block.index("ever_completed_a_validate_attempt\n")
        probe = validate_block.index("_early_exit_collapsed = bool(")
        self.assertLess(
            guard,
            probe,
            "占据早退必须被 ever_completed_a_validate_attempt 守住，否则本窗口可能"
            "一次 local-MBAR loose gate 都评不到就烧完预算",
        )
        # 计数只发生在完整 gate 求解之后；早退路径绝不能让它 +1，否则守卫会被自解锁。
        self.assertNotIn("sampler.validation_attempts = (", validate_block)

    def test_production_entry_is_pure_delta_f_gate_ess_diagnostic_only(self):
        # [IBS_BIAS_PROTOCOL_VERSION=29] 生产入口门 = 纯 max|Δf_k−ΔF^MBAR| < 阈值：
        # abs_ess / min_ess_ratio / 占据平坦度 / coverage_ESS / raw_residual 全部只作
        # 诊断，不参与放行（否则会把宽松 10 kJ/mol 门偷偷变成严格收敛门）。零重叠外推
        # 出的 908 kJ/mol 自然因 >阈值被拒，不需 abs_ess 门。
        self.assertIn("gate_ok = bool(", self.engine)
        self.assertIn("IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL", self.engine)
        self.assertIn("diagnostics_only_note", self.engine)
        # abs_ess 仍算，但只进诊断字段，不门控。
        self.assertIn('"min_absolute_ess": float(_gate_abs_ess)', self.engine)
        # 曾把宽松门变严格门的逻辑必须已删除。
        self.assertNotIn("reliable_local_mbar", self.engine)
        self.assertNotIn("occupancy_gate_ok", self.engine)
        self.assertNotIn('gate_error = "insufficient_overlap_or_ess"', self.engine)
        self.assertNotIn("gate_ok = bool(reliable_gate_ok or occupancy_gate_ok)", self.engine)
        # OCC_* 常量仍在，但只用于 _diagnose_local_mbar_situation 的占据平坦/塌陷诊断。
        self.assertIn("occupancy_is_flat", self.engine)
        self.assertIn("occupancy_collapsed", self.engine)
        # 旧的占据 LSE 自洽收敛门/best-effort 残差门已彻底移除。
        self.assertNotIn('best_effort_acceptance_reason = "global_safety_cap"', self.engine)
        self.assertNotIn('best_effort_acceptance_reason = "bounded_warmup_budget"', self.engine)

    def test_untrusted_tmbar_falls_back_to_bounded_with_hard_pairwise_cap(self):
        # [IBS_BIAS_PROTOCOL_VERSION=29] 破"低重叠 TMBAR 错误大步→占据塌缩→TMBAR
        # 更不可靠"循环：TMBAR 候选质量不可信时退回 bounded feedback；任何更新硬
        # cap 2 kT；local-MBAR gate 低 ESS 视为 insufficient_overlap（不当门残差）。
        self.assertIn("IBS_MAX_APPLIED_PAIRWISE_STEP_KT = 2.0", self.engine)
        self.assertIn("IBS_TMBAR_TRUST_MIN_OVERLAP = 0.05", self.engine)
        self.assertIn("IBS_TMBAR_TRUST_MIN_ABSOLUTE_ESS = 10.0", self.engine)
        self.assertIn("IBS_TMBAR_TRUST_MIN_DECORRELATED_SAMPLES = 10", self.engine)
        self.assertIn("IBS_TMBAR_TRUST_MAX_UNCERTAINTY_KJ_MOL = 5.0", self.engine)
        self.assertIn("IBS_TMBAR_TRUST_MIN_COVERAGE_ESS_FRACTION = 0.8", self.engine)
        # 控制器选择由 trust 门控，硬 cap 在 update_weights 两个控制器之后统一施加。
        update_start = self.engine.index("    def update_weights(")
        update_end = self.engine.index("    def apply_learning_rate_penalty(", update_start)
        update_body = self.engine[update_start:update_end]
        self.assertIn("if tmbar_candidate_trusted:", update_body)
        self.assertIn("IBS_MAX_APPLIED_PAIRWISE_STEP_KT", update_body)
        self.assertIn("hard_pairwise_cap_applied", update_body)
        # cap 不在 _damped_tmbar_absolute_update 本体（保持该单元的独立可测性）。
        damped_start = self.engine.index("    def _damped_tmbar_absolute_update(")
        damped_end = self.engine.index("    def _bounded_log_occupancy_update(", damped_start)
        self.assertNotIn(
            "IBS_MAX_APPLIED_PAIRWISE_STEP_KT",
            self.engine[damped_start:damped_end],
        )
        # 生产入口门是纯 Δf−ΔF<阈值（见 test_production_entry_is_pure_delta_f_gate_
        # ess_diagnostic_only）；这里只锁 TMBAR 更新可信门与硬 cap 的 trusted-only 施加。
        # abs_ess 不再门控生产入口放行，故已无 insufficient_overlap_or_ess 分支。
        self.assertNotIn('gate_error = "insufficient_overlap_or_ess"', self.engine)

    def test_dominant_identity_is_diagnostic_only(self):
        update_start = self.engine.index("    def update_weights(")
        update_end = self.engine.index("    def apply_learning_rate_penalty(", update_start)
        update_body = self.engine[update_start:update_end]
        self.assertIn("dominant诊断=state", update_body)
        self.assertNotIn("dominant_switched", update_body)
        self.assertNotIn("_dominant_hold_streak", update_body)

    def test_warmup_update_uses_sample_hold_pairwise_controller(self):
        # [Candidate-first, Validate-or-Learn v1] 9 -> 10: LEARN no longer
        # calls update_weights()/the full-history TMBAR selector at all (see
        # test_learning_to_freeze_freezes_only_when_step_settled); the
        # semantic change is enough to require invalidating old in-flight
        # unconverged caches under the new protocol version.
        self.assertIn("IBS_WARMUP_UPDATE_PROTOCOL_VERSION = 10", self.engine)
        # [IBS_BIAS_PROTOCOL_VERSION=29] damping 0.20->0.10, minibatch 20->40.
        # severe 区两档：residual<70 固定 4 kT，>=70 才放开到 severe_max=10 kT。
        self.assertIn("IBS_TMBAR_UPDATE_DAMPING = 0.10", self.engine)
        self.assertIn("IBS_TMBAR_FALLBACK_SGD_PAIRWISE_STEP_KT = 10.0", self.engine)
        self.assertIn("IBS_UPDATE_ADAPTIVE_RESIDUAL_COLLAPSE = 70.0", self.engine)
        self.assertIn("IBS_WARMUP_FRAME_STRIDE_STEPS = 250", self.engine)
        self.assertIn("IBS_TMBAR_LEARNING_MINIBATCH_FRAMES = 40", self.engine)
        self.assertIn("IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT = 1.0", self.engine)
        update_start = self.engine.index("    def update_weights(")
        update_end = self.engine.index("    def apply_learning_rate_penalty(", update_start)
        update_body = self.engine[update_start:update_end]
        self.assertIn("self._damped_tmbar_absolute_update(", update_body)
        self.assertIn("IBS_TMBAR_FALLBACK_SGD_PAIRWISE_STEP_KT", update_body)
        self.assertNotIn("dominant_switched", update_body)
        self.assertNotIn("_dominant_hold_streak", update_body)
        self.assertIn(
            '"warmup_update_protocol_version": IBS_WARMUP_UPDATE_PROTOCOL_VERSION',
            self.engine,
        )
        self.assertIn("cached_warmup_update_version", self.engine)
        self.assertIn(
            "self._bounded_log_occupancy_update(\n"
            "                f_old,\n"
            "                mean_p_batch,",
            self.engine,
        )
        # update_weights 仍用阻尼 TMBAR 自洽步长做诊断（不再驱动 warmup 收敛门，
        # 收敛门已换成 run_all_windows 里的 local-MBAR loose gate）。
        self.assertIn("tmbar_self_consistent", update_body)

    def test_v9_tmbar_damping_reaches_the_production_fraction_in_ten_updates(self):
        """跑**生产的**方法与**生产的**阻尼常数，不注入测试自己的值。

        原实现把 `_damped_tmbar_absolute_update` 用 AST 抽出来、在
        `namespace = {..., "IBS_TMBAR_UPDATE_DAMPING": 0.20}` 里 exec，于是被测
        函数的默认参数绑到 0.20，断言 `1-0.8**10` 与 `effective_damping == 0.20`。
        生产常数是 0.10（本文件另一条测试自己 grep 着 "= 0.10"），所以那条断言
        钉的是一个生产永远不会产生的收敛速率，且对生产常数的任何改动免疫。
        现在直接调真方法、从真常数推期望值：改了 0.10 这里就会跟着变，
        而"十次更新走完 1-(1-d)^10"这个几何收敛律仍然被钉住。
        """
        if _IMPORT_ERROR is not None:
            self.skipTest(f"project runtime unavailable: {_IMPORT_ERROR}")

        damping = float(IBS_TMBAR_UPDATE_DAMPING)
        self.assertTrue(0.0 < damping <= 1.0, "生产阻尼必须落在 (0, 1]")

        dummy = type("DummySampler", (), {"n_states": 5, "f_history": []})()
        target = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
        current = np.zeros(5, dtype=float)
        diagnostics = None
        for _ in range(10):
            current, diagnostics = IBSSampler._damped_tmbar_absolute_update(
                dummy, current, target
            )
        expected_fraction = 1.0 - (1.0 - damping) ** 10
        np.testing.assert_allclose(
            current, expected_fraction * target, rtol=0.0, atol=1.0e-12
        )
        self.assertEqual(diagnostics["method"], "damped_absolute_tmbar_v9")
        self.assertAlmostEqual(diagnostics["effective_damping"], damping)
        # 若哪天有人把默认阻尼改成 1.0（一步到位），这条几何律测试就失去分辨力。
        self.assertLess(
            expected_fraction, 1.0 - 1.0e-9,
            "阻尼为 1.0 时十次更新与一次等价，本测试不再有分辨力",
        )

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

    # [2026-09-09] test_fixed_h_probe_cache_protocol_version_is_explicit 已删除：
    # tests/test_warmup_overlap_protocol.py:820 有同名测试，断言的是**导入的常量**
    # (`assert FIXED_H_PROBE_CACHE_PROTOCOL_VERSION == 3`)。那一份严格更强 ——
    # 不依赖源码里 " = 3" 的排版，重新格式化不会假失败。这里这份是同一契约的
    # 文本重复，删掉不损失覆盖。

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
        # 4 → 5（2026-08-04，P0-12b）：身份里加入配体**起始构象**指纹。实测同一分子
        # 换一个起始构象，溶剂腿去电荷 62.80 → 191.05 kJ/mol，而旧口径两次都判
        # "缓存有效"（P0-12）。升版本号让所有旧盒缓存重建，这是刻意的。
        # 5 → 6（2026-08-05，B4）：manifest 加入 reserved co-ion 字段，旧缓存不含
        # dummy 粒子，必须重建。
        self.assertIn("SOLVENT_CACHE_PROTOCOL_VERSION = 7", self.runabfe)
        self.assertIn('"ligand_start_conformer"', self.runabfe)
        self.assertIn("DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR = 0.15", self.runabfe)
        self.assertIn('positiveIon="Na+"', self.runabfe)
        self.assertIn('negativeIon="Cl-"', self.runabfe)
        self.assertIn("ionicStrength=ionic_strength_molar * unit.molar", self.runabfe)
        self.assertIn("neutralize=True", self.runabfe)
        self.assertIn('"solvent_cache_manifest.json"', self.runabfe)
        self.assertIn('manifest["ordinary_na_count"]', self.runabfe)
        self.assertIn('manifest["ordinary_cl_count"]', self.runabfe)

    def test_vanishing_uses_thermodynamic_few_state_subdomains_without_overlap_two(self):
        self.assertIn("THERMODYNAMIC_PATH_PROTOCOL_VERSION = 22", self.preoptimizer)
        self.assertIn(
            "redistribute_vanishing_lambda_subdomains(",
            self.preoptimizer,
        )
        self.assertIn("VANISHING_PROBE_BASE_STATE_COUNT = 17", self.preoptimizer)
        self.assertIn("VANISHING_FINAL_STATE_COUNT = 23", self.preoptimizer)
        # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=21] 实测 Fisher 度规现在【控制】
        # 生产布点，而不是算完就丢。v19/v20 用固定二次网格 + 4 个人工点 + 2 个 bridge
        # 点，实测后果是 window 0 用 4 条边扛了全程 47.22 中的 41.32 热力学长度
        # （单边 8.82~11.83），其余 18 条边合计 5.90，最短 0.0002 —— 零重叠。
        # 但 v18 的纯等长布点又把解耦端拉断（0.9225, 0.8382, 0.0）。v21 等分
        # u=(1-beta)*s_hat+beta*(1-lambda)，度规驱动 + 可证明的几何覆盖下限。
        self.assertIn("VANISHING_GEOMETRIC_FLOOR_WEIGHT = 0.3", self.preoptimizer)
        self.assertIn("def blended_metric_vanishing_lambdas(", self.preoptimizer)
        self.assertIn("def vanishing_max_lambda_gap_bound(", self.preoptimizer)
        self.assertIn(
            "fisher_metric_blended_with_geometric_floor_v21", self.preoptimizer
        )
        self.assertIn(
            '"probe_controls_base_lambda_placement": True', self.preoptimizer
        )
        self.assertIn("validate_vanishing_lambda_path_invariants", self.pipeline)
        self.assertIn("validate_single_shared_boundary_ranges", self.pipeline)
        # n_states 必须是 keyword-only：v20 的前身第二位置参数是【请求探针态数】17，
        # 本函数的第二参是【期望产出路径长度】23。留着位置传参，改名时残留的 `, 17`
        # / `, stage2_states` 会静默拿错数字去比对（实测已发生过一次，Stage 2 直接
        # 抛「必须恰好 17 态，实际 23」）。
        self.assertIn(
            "def validate_vanishing_lambda_path_invariants(\n"
            "    lambdas_vdw,\n"
            "    *,\n",
            self.preoptimizer,
        )
        # 🔑 [2026-08-28] 上面这三个调用点在 2026-08-27 final_state_count 变成
        # 可配置之后曾经继续"裸调用"（不传 n_states，静默默认拿
        # VANISHING_FINAL_STATE_COUNT=23 校验）——final_state_count 配置成非 23
        # (比如 12) 时，一个完全正确的产出会被错误拒绝（实测：4W53 生产事故，
        # 抛出「必须恰好 23 态，实际 12」）。现在必须显式传
        # n_states=<产出本身的长度>（不是残留的错误数字），下面三条断言确认
        # 修复没有再回退成裸调用。
        self.assertIn(
            "cached_lambdas, n_states=len(cached_lambdas)", self.pipeline
        )
        self.assertEqual(
            self.pipeline.count(
                "optimized_lambdas_2, n_states=len(optimized_lambdas_2)"
            ),
            2,
        )
        self.assertIn(
            "validate_vanishing_lambda_path_invariants(lambdas_var, n_states=n_states)",
            self.pipeline,
        )
        # 被 v21 取代的固定调度链必须已经从源码里消失，避免再被当成可选路径捡回去。
        self.assertNotIn("insert_human_vanishing_endpoint_lambdas", self.preoptimizer)
        self.assertNotIn("insert_fisher_bridge_lambdas", self.preoptimizer)
        self.assertNotIn("VANISHING_PREBRIDGE_STATE_COUNT", self.preoptimizer)
        redistribute_start = self.preoptimizer.index(
            "def redistribute_vanishing_lambda_subdomains("
        )
        redistribute_end = self.preoptimizer.index(
            "def partition_windows_by_thermodynamic_length(", redistribute_start
        )
        redistribute_body = self.preoptimizer[redistribute_start:redistribute_end]
        self.assertIn("blended_metric_vanishing_lambdas(", redistribute_body)
        # 2026-08-27: redistribute_vanishing_lambda_subdomains 新增
        # final_state_count 参数（默认仍是 VANISHING_FINAL_STATE_COUNT=23，
        # 见 project_stage2_final_state_count_2026-08-27），调用改成显式传
        # keyword n_states=final_state_count——仍然是 keyword-only，不是
        # 本测试原本要防的"残留位置参数拿错数字"回归。
        self.assertIn(
            "validate_vanishing_lambda_path_invariants(optimized_lambdas, n_states=final_state_count)",
            redistribute_body,
        )
        # 度规必须真的进入布点，不能只当诊断留在一边。
        self.assertNotIn("_fisher_lambdas", redistribute_body)
        self.assertIn('"total_window_state_slots": int(', redistribute_body)
        self.assertNotIn("_pilot_ti_cumulative_f", redistribute_body)
        # 🔑 [2026-09-03, 路径协议 v22] 这里原本是 assertNotIn("mean_dU_dlambda")。
        # 它要守的是「TI 自由能不得**取代**度规成为基础布点」（v19/v20 的回归），
        # 不是「模块里不准出现梯度」。v22 加了自由能定向加密：基础布点仍然是
        # blended_metric_vanishing_lambdas，ΔF 只在其之上**追加**点，且默认关闭。
        # 所以把断言收窄到原本的意图，并补上 v22 自己的守卫。
        self.assertIn(
            "blended_metric_vanishing_lambdas(", redistribute_body
        )
        # 基础布点吃的必须是 base_state_count（final_state_count 减去加密点数），
        # 不能是别的东西——否则总态数会对不上。
        self.assertIn("base_state_count = final_state_count - n_densify", redistribute_body)
        # 加密只能发生在 n_densify 为真时；默认参数必须是 0（关闭 = 与 v21 逐字节相同）。
        self.assertIn("free_energy_densify_points: int = 0", redistribute_body)
        self.assertIn("if n_densify:", redistribute_body)
        # 没给实测梯度就要求加密 → 必须 fail closed，不准猜。
        self.assertIn("if pilot_mean_dU_dlambda is None:", redistribute_body)
        # 梯度绝不能出现在基础布点那次调用的参数里。
        base_call_start = redistribute_body.index("optimized_lambdas, cumulative, optimized_edge_lengths = (")
        base_call_end = redistribute_body.index("if n_densify:", base_call_start)
        self.assertNotIn("mean_dU_dlambda", redistribute_body[base_call_start:base_call_end])
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
        # [IBS_BIAS_PROTOCOL_VERSION=29] validating 阶段：累计最近若干批固定-f_k
        # minibatch，攒满滑窗深度就跑一次 local MBAR loose gate；未过则退回 learning。
        self.assertIn(
            "frozen_mbar_batches.append(sampler.tmbar_history[-1])", self.engine
        )
        self.assertIn(
            "len(frozen_mbar_batches) < IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES",
            self.engine,
        )
        self.assertIn("failed_local_mbar_loose_gate", self.engine)
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
        self.assertIn("TRADITIONAL_LJ_LRC_PROTOCOL_VERSION = 3", self.engine)
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

    def test_traditional_lrc_producer_uses_v3_worker_key(self):
        self.assertNotIn('"lj_tail_prefactor_kj_nm3_mol": lj_tail_prefactor', self.engine)
        self.assertNotIn("_lj_tail_correction_S_kj_nm6(", self.engine)
        self.assertIn(
            "tail_sigma, tail_s6_per_sigma, tail_s12_per_sigma = (",
            self.engine,
        )
        self.assertIn("_lj_tail_correction_sigma_resolved_moments(", self.engine)

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
        """**首次** e_base 读取失败时必须立刻连坐标和受力一起查。

        2026-09-09 重写：原实现是两条无作用域的全文件 grep
        （`assertIn("getPositions=True", self.engine)`）。这两个字符串在
        ibs_engine.py 里分别出现 29 / 20 次，所以把这段代码整段删掉它照样通过 ——
        它测不到任何东西。真正的位置是 `IBSSampler.collect_energies` 里
        `self._consecutive_base_failures == 1` 那一支：区分"偶发一帧读不到"与
        "轨迹已经 NaN"，靠的就是立刻补查坐标/受力，而不是再等五个 MD chunk。
        现在用 AST 把断言钉在那一支上。
        """
        tree = ast.parse(self.engine)
        fn = next(
            (n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "collect_energies"),
            None,
        )
        self.assertIsNotNone(fn, "找不到 collect_energies")

        branches = [
            node for node in ast.walk(fn)
            if isinstance(node, ast.If)
            and "_consecutive_base_failures" in ast.unparse(node.test)
            and "1" in ast.unparse(node.test)
        ]
        self.assertTrue(
            branches,
            "collect_energies 里找不到"
            " `self._consecutive_base_failures == 1` 这一支 —— 首次失败的即时"
            "诊断被删掉了，或判据改了名（改了请同步本测试）",
        )

        for node in branches:
            requested = set()
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                if getattr(call.func, "attr", None) != "getState":
                    continue
                requested |= {
                    kw.arg for kw in call.keywords
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True
                }
            self.assertIn("getPositions", requested, (
                "首次 e_base 失败没有立刻查坐标 —— 区分不了"
                "「偶发一帧读不到」和「轨迹已经 NaN」"
            ))
            self.assertIn("getForces", requested, "首次 e_base 失败没有立刻查受力")

    def test_owned_system_early_return_validates_swig_object(self):
        core = (ROOT / "abfe_core.py").read_text(encoding="utf-8")
        start = core.index("def ensure_owned_system(")
        end = core.index("def sync_all_exclusions(", start)
        block = core[start:end]
        self.assertLess(block.index("system.getNumParticles()"), block.index("return system"))

    def test_resume_rejects_mismatched_lrc_protocol_version(self):
        # The eight resume-reuse gates used to be inlined in run_all_windows (~110
        # lines depending on 8 locals, impossible to call in isolation).  They now
        # live in the module-level pure helper _resume_cached_window_gate_status,
        # extracted so each gate can be exercised against a mock convergence.json
        # -- see test_resume_reuse_contracts.py.  Gate semantics and thresholds are
        # unchanged; only the location is.
        #
        # Anchor INSIDE that function's body rather than anywhere in the file, so
        # the contract cannot be satisfied by the same text appearing in an
        # unrelated place, and normalize whitespace so a pure reformat (wrapping
        # the comparison across lines) does not read as a missing gate.
        gate_fn = self._extracted_resume_gate_source()
        self.assertIn(
            'lrc_version_match = ( cached_conv.get("lj_tail_lrc_protocol_version") '
            "== TRADITIONAL_LJ_LRC_PROTOCOL_VERSION )",
            " ".join(gate_fn.split()),
        )
        self.assertIn("and lrc_version_match", gate_fn)
        # Same gate must also apply to the λ-content window-reuse path (splitting/
        # inserting lambdas), not just the plain resume-skip path.
        self.assertIn(
            'conv.get("lj_tail_lrc_protocol_version") == TRADITIONAL_LJ_LRC_PROTOCOL_VERSION',
            self.pipeline,
        )

    def test_stage1_cdf_endpoint_reaches_one_without_double_counting(self):
        # todolist.md P1: optimize_stage1_decharging's CDF interpolation used to
        # omit the endpoint guarantee that optimize_lambda_path_adaptive already
        # has, letting np.interp silently clamp the last 1-2 target points to
        # fp[-1]=0.0 (duplicate lambda_coul=0.0 states).  The invariant to keep
        # is therefore "xp ends at exactly 1.0".
        #
        # [0831issue P2] 那个不变量原来是靠 `xp[-1] = 1.0` **事后覆盖**实现的，而这个
        # 赋值同时把倒数第二个累积坐标 c_{N-2} 也覆盖掉了 —— 最后一个区间的宽度从
        # w[N-2] 变成 w[N-2]+w[N-1]，λ[N-2] 的权重被双重计入。现在改成"用前 N-1 个
        # 权重当区间宽度、按它们自己的和归一化"，末端天然为 1.0。
        # 因此这条测试不再 grep 那行赋值（它已经不该存在），而是直接验证构造本身：
        # 既保住端点不变量，又不能再有双重计入。
        both = [
            "interval_weights = np.asarray(density_weight, dtype=float)[:-1]",
            "interval_weights = np.asarray(density_weight, dtype=float).ravel()[:-1]",
        ]
        for needle in both:
            self.assertIn(needle, self.preoptimizer)
        # 覆盖式写法必须彻底消失（两处都是）。
        self.assertNotIn("xp[-1] = 1.0", self.preoptimizer)

        # 2026-09-09：这里原本紧跟一段"在测试里把修好的公式重写一遍、再断言它
        # 自己"的算术（xp/interval_total/np.diff 四条 assert）。那是同义反复，
        # production 一行都没跑到，且与
        # tests/test_0831issue_p2_batch.py::test_cdf_no_longer_double_counts_...
        # 逐字重复。已删除；为什么无法改成行为断言，见那条测试的 docstring。

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
        # The shared loader enumerates expected integer window indices; it does
        # not discover/sort a possibly truncated subset of filenames.
        self.assertIn("engine.load_ibs_window_outputs_from_dir(", self.runabfe)
        self.assertIn("for local_idx, (start, end) in enumerate(ranges):", self.engine)
        self.assertIn("_assert_expected_windows_all_loaded(", self.engine)

    def test_analyze_only_fallback_fails_closed_on_unconverged_solve(self):
        self.assertIn("validator._assert_stage_result_sane(name, cached)", self.runabfe)
        self.assertIn("_assert_sampling_result_converged(cached, context=name)", self.runabfe)

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
        # [2026-09-10] Stage 2 的态数校验对的是**最终生产态数**
        # (stage2_final_n_states)，不是 pilot 探针密度 stage2_states —— 两者
        # 允许不同（生产 config 就是 17 探针 / 18 生产态），拿探针数校验会在每次
        # resume 上误判「状态数不匹配」并重烧一整轮 pilot。protocol_match /
        # anchor_contract_match 这两个必须条件不变。
        self.assertIn(
            "if protocol_match and anchor_contract_match and (\n"
            "                        len(cached_lambdas) == _expected_final_states\n"
            "                        or is_verified_auto_repair\n"
            "                    ):",
            self.pipeline,
        )
        self.assertIn(
            '_expected_final_states = _resolve(\n'
            '                        "stage2_final_n_states", "final_state_count"\n'
            '                    )',
            self.pipeline,
        )

    def test_final_convergence_gate_uses_orthogonal_evidence_not_duplicated_ess(self):
        # todolist.md P1 原意：converged 曾经只要求 min ESS ratio >= 0.05，样本总数
        # 很少时个位数的绝对 ESS 也能通过 >5% 的比例门。当时补的 absolute ESS 门
        # 修的是真问题，但用错了量——见下。
        #
        # [ESS_GATE_PROTOCOL_VERSION=2] absolute_ess 在构造上恒等于
        # min_ess_ratio × n_frames_decorrelated（denom 是同一个标量，
        # min(neff)/denom == min(neff/denom)），所以它不是第二份独立证据；给同一个量
        # 配两个阈值的实际效果是让 ratio 阈值失去意义（final_min_absolute_ess=50 在
        # N_decorrelated=114 时等价于要求 ratio>=0.44，而日志里报的门是 0.05），而且
        # 样本越少门越严——与"延长采样"的修复方向相反。
        # 现在换成三份真正正交的证据，本测试钉住这个结构，防止 absolute-ESS 门被
        # "顺手加回来"，也防止 min_ess_ratio 被换回 raw 单参考 ESS。occupancy 与
        # warmup 协议一致，只作诊断：不能等全部 GPU 采样完成后再用同一指标反向否决。
        conv_idx = self.engine.index("converged = bool(\n            len(local_results) == len(valid_windows)")
        conv_block = self.engine[conv_idx:conv_idx + 700]
        # (1) 权重质量：扣掉共模因子后的混合覆盖度比例
        self.assertIn(
            "_meets_minimum_with_roundoff(min_overlap, min_overlap_threshold)",
            conv_block,
        )
        # occupancy 保留为一阶矩伴随诊断，但不得进入最终 converged 门。
        self.assertNotIn("min_occupancy_normalized", conv_block)
        self.assertIn('"min_occupancy_normalized_threshold": None,', self.engine)
        self.assertIn('"min_occupancy_is_gate": False,', self.engine)
        self.assertIn('"min_occupancy_gate_retired_reason": (', self.engine)
        # (2) 样本量：与比例正交的那份证据
        self.assertIn(
            "min_decorrelated_samples >= int(final_min_decorrelated_samples)",
            conv_block,
        )
        # (3) 输出精度：MBAR 自带全协方差的端点不确定度
        self.assertIn(
            "_meets_maximum_with_roundoff(", conv_block
        )
        # 退役的 absolute-ESS 门不得出现在 converged 里
        self.assertNotIn("final_min_absolute_ess", conv_block)
        self.assertNotIn("min_absolute_ess", conv_block)
        # 受门量必须是 mixture 版本，raw 版本只能是诊断
        self.assertIn(
            '"min_overlap_method": (\n'
            '                "per_window_mixture_coverage_ess_ratio_common_mode_removed"',
            self.engine,
        )
        self.assertIn('"min_absolute_ess_threshold": None,', self.engine)
        self.assertIn('"raw_min_overlap": raw_min_overlap,', self.engine)
        # warmup 的 TMBAR trust 门是另一个消费者，必须继续读 raw 量（tmbar_history
        # 天生没有 f_k，读 mixture 量会让它永久 False、把 warmup 钉死在受限反馈上）
        self.assertIn('_q_overlap = tmbar_res.get("raw_min_overlap")', self.engine)
        self.assertIn('_q_abs_ess = tmbar_res.get("raw_min_absolute_ess")', self.engine)
        # 两份增广矩阵实现都必须对 sampled_distribution_row != 0 fail closed
        self.assertEqual(2, self.engine.count("if sampled_row != 0:"))
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
    """Correctness tests for the v3 sigma-resolved LJ tail LRC
    (_lj_tail_lrc_coefficients_kj_mol / _lj_softcore_tail_radial_integrals).
    """

    SIGMA = np.asarray([0.31, 0.37])
    S6 = np.asarray([5.0, 7.5])   # kJ*nm^6/mol, fixed per-sigma moments
    S12 = np.asarray([1.4, 2.0])  # kJ*nm^12/mol
    ALPHA_LJ = 0.5
    M_LJ = 2.0
    N_LJ = 2.0

    def test_lambda_zero_gives_exact_zero_lrc(self):
        coeffs = _lj_tail_lrc_coefficients_kj_mol(
            [0.0, 0.3, 1.0], self.SIGMA, self.S6, self.S12,
            self.ALPHA_LJ, self.M_LJ, self.N_LJ,
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
        expected = 16.0 * math.pi * (
            np.sum(self.S12) * i12 - np.sum(self.S6) * i6
        )

        coeffs = _lj_tail_lrc_coefficients_kj_mol(
            [1.0], self.SIGMA, self.S6, self.S12,
            self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )
        self.assertTrue(math.isclose(coeffs[0], expected, rel_tol=1e-6))

    def test_switching_aware_result_differs_from_old_cutoff_only_r6(self):
        # v1 (superseded) formula: -(16*pi/3)*S6/rc^3 -- attractive-only,
        # cutoff-only, no switching region, no softcore/repulsive terms.
        rc = LJ_TAIL_LRC_R_CUTOFF_NM
        old_v1_coeff = -(16.0 * math.pi / 3.0) * np.sum(self.S6) / rc ** 3
        new_coeff = _lj_tail_lrc_coefficients_kj_mol(
            [1.0], self.SIGMA, self.S6, self.S12,
            self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )[0]
        self.assertFalse(math.isclose(new_coeff, old_v1_coeff, rel_tol=1e-3))

    def test_softcore_denominator_changes_integrals_away_from_lambda_one(self):
        # At lambda_vdw < 1 the softcore shift alpha_lj*(1-lambda)^m_lj > 0
        # makes D(r) larger than plain r^6 everywhere, which must shrink the
        # magnitude of both radial integrals relative to the lambda=1 case.
        i6_full, i12_full = _lj_softcore_tail_radial_integrals(
            1.0, self.ALPHA_LJ, self.M_LJ, self.SIGMA[0],
        )
        i6_half, i12_half = _lj_softcore_tail_radial_integrals(
            0.5, self.ALPHA_LJ, self.M_LJ, self.SIGMA[0],
        )
        self.assertLess(i6_half, i6_full)
        self.assertLess(i12_half, i12_full)

    def test_sigma_resolution_is_not_equivalent_to_lumped_scalars(self):
        # 分组求和必须真的按组用各自的 D_ij(r) 积分。若实现退化成「先把 S6/S12 加总、
        # 再用某一个 sigma 积一次」，在 lambda<1 时结果会与逐组积分不同——这条测试
        # 正是要让那种退化实现失败。lambda=1 时 D 与 sigma 无关，两者本就应相等。
        grouped = _lj_tail_lrc_coefficients_kj_mol(
            [0.5], self.SIGMA, self.S6, self.S12,
            self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )[0]
        lumped = _lj_tail_lrc_coefficients_kj_mol(
            [0.5],
            np.asarray([self.SIGMA[0]]),
            np.asarray([float(np.sum(self.S6))]),
            np.asarray([float(np.sum(self.S12))]),
            self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )[0]
        self.assertNotEqual(grouped, lumped)
        # lambda=1 处 alpha*(1-lambda)^m == 0，sigma 依赖消失，两者必须一致。
        at_one_grouped = _lj_tail_lrc_coefficients_kj_mol(
            [1.0], self.SIGMA, self.S6, self.S12,
            self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )[0]
        at_one_lumped = _lj_tail_lrc_coefficients_kj_mol(
            [1.0],
            np.asarray([self.SIGMA[0]]),
            np.asarray([float(np.sum(self.S6))]),
            np.asarray([float(np.sum(self.S12))]),
            self.ALPHA_LJ, self.M_LJ, self.N_LJ,
        )[0]
        self.assertTrue(math.isclose(at_one_grouped, at_one_lumped, rel_tol=1e-9))

    def test_bigger_sigma_softens_more_at_partial_lambda(self):
        # D_ij(r)=alpha*sigma^6*(1-lambda)^m + r^6：sigma 越大，同一 lambda 下软核偏移
        # 越大，两个径向积分都应更小。这是 sigma 缩放方向正确性的直接判据。
        i6_small, i12_small = _lj_softcore_tail_radial_integrals(
            0.5, self.ALPHA_LJ, self.M_LJ, 0.25,
        )
        i6_big, i12_big = _lj_softcore_tail_radial_integrals(
            0.5, self.ALPHA_LJ, self.M_LJ, 0.40,
        )
        self.assertLess(i6_big, i6_small)
        self.assertLess(i12_big, i12_small)

    def test_mismatched_sigma_spectrum_lengths_fail_closed(self):
        with self.assertRaises(ValueError):
            _lj_tail_lrc_coefficients_kj_mol(
                [0.5], np.asarray([0.3, 0.35]), np.asarray([1.0]),
                np.asarray([1.0, 2.0]), self.ALPHA_LJ, self.M_LJ, self.N_LJ,
            )

    def test_invalid_box_fails_closed(self):
        with self.assertRaises(ValueError):
            _periodic_box_volume_nm3(np.zeros((3, 3)))

    def test_lrc_cache_version_is_explicit(self):
        self.assertGreaterEqual(TRADITIONAL_LJ_LRC_PROTOCOL_VERSION, 3)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# [2026-09-09] 结构性守卫：缓存身份里不得出现"自产产物的哈希"——按**形状**判，
# 不按 helper 名字判。
#
# 已有的 test_todo_verified_fixes.py::test_no_self_generated_hash_may_enter_a_
# cache_identity 断的是那五个 helper（_code_hash / _system_xml_hash /
# _topology_hash / _positions_hash / _box_vectors_hash）恒返回 None。那条是对的，
# 但它按**名字**枚举，抓不到第六条用别的函数名走同一件事的路径 —— 本次静态审计
# 就找到了一条：`_build_top_level_protocol_key` 里
#
#     "preopt_cache_sha256": {
#         "decharging": _file_sha256(os.path.join(self.checkpoint_dir, "preopt_dual_decharging...json")),
#         "vanishing":  _file_sha256(os.path.join(self.checkpoint_dir, "preopt_dual_vanishing...json")),
#     }
#
# `_file_sha256` 没有被中性化，哈希的是**本管线自己写出来的** preopt 缓存文件，
# 而它进的是顶层最终结果缓存身份。形状与已复发三次的那条完全一致（code_sha256
# 2026-08-24；system_xml_hash + positions_sha256 2026-09-09）：任何让 preopt 缓存
# 字节变化（键序、多一个诊断字段、浮点 repr）而物理不变的改动，都会作废用户已经
# 算完的最终结果。而 λ 路径本身在同一个 payload 里已经由 `stage1_protocol_key` /
# `stage2_protocol_key` 覆盖，这个文件哈希是多余的第二套机制。
# ---------------------------------------------------------------------------

_CACHE_IDENTITY_BUILDERS = (
    "_remd_sampling_fingerprint",
    "_rebalance_fingerprint",
    "_pre_equilibration_fingerprint",
    "_stage_protocol_key",
    "_preopt_protocol_key",
    "_build_top_level_protocol_key",
    "_build_sampling_protocol_key",
    "_build_geodesic_path_protocol_key",
)


def _byte_hashing_functions(tree):
    """名字无关地找出"读文件字节 → 取哈希"的函数。

    判据是**行为**：函数体里同时出现 `open(..., "rb")` 和一次 hashlib/sha256 调用。
    `abfe_pipeline._file_sha256` 就是这个形状。按行为认而不是按名字枚举，所以
    改名、或者有人新写一个同形状的 helper，都还在网里。

    🔑 不要把"对**语义值**取哈希"一起算进来。`ABFEPipeline._lambda_path_fingerprint`
    内部也算 sha256，但它哈希的是四舍五入后的 λ 数组（内存里的值），不是我们生成的
    文件字节 —— 那是**正确的**身份，删掉会开真洞（见下面测试的 docstring）。
    """
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.dump(node)
        opens_binary = any(
            isinstance(c, ast.Call)
            and (getattr(c.func, "id", None) == "open")
            and any(
                isinstance(a, ast.Constant) and isinstance(a.value, str) and "b" in a.value
                for a in c.args[1:]
            )
            for c in ast.walk(node)
        )
        hashes = "hashlib" in body or "sha256" in body
        if opens_binary and hashes:
            found.add(node.name)
    return found


def _self_product_byte_hashes_in_cache_identities(source=None):
    """返回 [(builder, lineno, 片段)]：缓存身份里对**文件字节**取哈希的调用。

    `source` 可传源码字符串，供自检用**同一份**判据 —— 不另抄一遍逻辑
    （抄一遍就变成"测试自己的副本"，正是本文件今天修掉的那类问题）。

    [2026-09-09 第二版] 第一版的判据是「函数名含 sha256/hash 且实参里出现
    checkpoint_dir/output_dir/…」。那个判据两头都漏：
      * 漏坏的 —— 把哈希调用挪进一个名字不含 hash 的 helper 就看不见了；
      * 漏好的 —— 一个读 JSON **内容**、只对语义值取哈希的 helper（现在
        `_build_top_level_protocol_key` 里的 `_preopt_lambda_identity` 正是如此）
        同样会用到 `self.checkpoint_dir`，按目录名判会误报，逼后人把正确的绑定删掉。
    现在改成两段式：先按行为找出"读字节→哈希"的函数集合，再看缓存身份里有没有调它。
    """
    if source is None:
        source = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="abfe_pipeline.py")
    byte_hashers = _byte_hashing_functions(tree)
    if not byte_hashers:
        return []

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in _CACHE_IDENTITY_BUILDERS:
            continue
        for call in ast.walk(node):   # ast.walk 会进嵌套 def，嵌套 helper 也在网里
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "attr", None) or getattr(call.func, "id", None)
            if name in byte_hashers:
                offenders.append((node.name, call.lineno, ast.unparse(call)[:120]))
    return offenders


def test_no_hash_of_our_own_artifact_bytes_enters_a_cache_identity():
    """缓存身份里不得对**我们自己生成的文件的字节**取哈希。

    已复发四次（code_sha256 2026-08-24；system_xml + positions 2026-09-09；
    `_file_sha256(preopt 缓存)` 同日）。字节哈希会让任何"物理不变、字节变了"的
    改动作废用户已经烧掉的 GPU 时间：键序、多一个诊断字段、浮点 repr、
    甚至排除表灌入顺序换成 sorted() 都够了。

    ⚠️ **对语义值取哈希是对的，别一起清掉。** 两个必须留着的例子：
      * `runabfe._sha256_file` 绑用户给的 gro/top/box —— 用户输入才配做身份；
      * `ABFEPipeline._lambda_path_fingerprint` 对四舍五入后的 λ 数组取 sha256，
        `_build_top_level_protocol_key` 的 `preopt_lambda_path` 正是用它。
        **λ 值不在任何 stage protocol key 里**（那两个键只有 `requested_n_states`），
        λ 路径是预优化器运行期算出来的、只落在 `preopt_dual_*.json` 里。所以这一项
        不是多余的：删掉它，一条改过的同长度 λ 路径会命中旧的最终结果缓存。
        （本条 2026-09-09 由 `review project todo list` 会话核实纠正 —— 我先前
        断言"已被 stage1/stage2 覆盖、是多余的"，那是**错的**。）
    """
    offenders = _self_product_byte_hashes_in_cache_identities()
    assert not offenders, (
        "这些缓存身份里对**我们自己生成的文件的字节**取了哈希：\n  "
        + "\n  ".join(f"abfe_pipeline.py:{line} {fn}: {src}" for fn, line, src in offenders)
        + "\n只有**用户输入**的字节、或**语义值**才配做身份。"
        "\n正解不是加一层更精细的字节哈希，是改成读内容取语义身份"
        "（参考 _build_top_level_protocol_key 里的 _preopt_lambda_identity）。"
    )


def test_the_byte_hash_detector_distinguishes_bytes_from_semantic_values():
    """探测器必须两头都准：报得出字节哈希，且**不**误报语义哈希。

    用**同一个** `_self_product_byte_hashes_in_cache_identities` 跑三段合成源码。
    第三段是关键：它就是生产现在的写法（读 JSON 内容 → 对语义值取指纹），必须
    判干净 —— 否则这条测试会逼着后人把 `preopt_lambda_path` 删掉，而那会让改过的
    λ 路径命中旧的最终结果缓存（真洞）。
    """
    dirty = (
        "import hashlib\n"
        "def _file_sha256(path):\n"
        "    with open(path, 'rb') as h:\n"
        "        return hashlib.sha256(h.read()).hexdigest()\n"
        "def _stage_protocol_key(self):\n"
        "    return {'x': _file_sha256(os.path.join(self.checkpoint_dir, 'a.json'))}\n"
    )
    renamed = dirty.replace("_file_sha256", "_totally_innocent_name")
    semantic = (
        "import hashlib\n"
        "def _stage_protocol_key(self):\n"
        "    def _identity(filename):\n"
        "        with open(os.path.join(self.checkpoint_dir, filename), encoding='utf-8') as h:\n"
        "            payload = json.load(h)\n"
        "        return ABFEPipeline._lambda_path_fingerprint(payload.get('lambdas_var'), None)\n"
        "    return {'x': _identity('preopt_dual_vanishing.json')}\n"
    )

    assert _self_product_byte_hashes_in_cache_identities(dirty), (
        "对一段明确的字节哈希都报不出来——探测器坏了，上面那条测试没有意义"
    )
    assert _self_product_byte_hashes_in_cache_identities(renamed), (
        "换个不含 hash 的函数名就躲过去了——判据又退回按名字枚举了"
    )
    assert _self_product_byte_hashes_in_cache_identities(semantic) == [], (
        "对'读内容→语义指纹'的正确写法误报了。按这个误报去删，改过的 λ 路径就会"
        "命中旧的最终结果缓存 —— 那是真洞，不是整洁问题。"
    )

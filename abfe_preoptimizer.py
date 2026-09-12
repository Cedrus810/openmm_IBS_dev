# =============================================================================
# ABFE 预采样优化器 - ACES 路径优化版 (v5.0 - 真正的 Pathfinding)
# 基于：ACES (JCTC 2023), IBS (JCTC 2026), CBFE (JCIM 2026)
# =============================================================================
"""
修复清单：
✅ 真正的 ACES Pathfinding：分析能量梯度，自动计算最优衰减指数
✅ 不是固定的λ²，而是根据配体性质动态调整 charge_exponent 和 vdw_exponent
✅ 基于能量方差σ²(U) 重分布 Lambda 点
✅ 确保每段ΔG 近似相等（热力学长度最小化）
"""

import openmm
from openmm import app, unit, XmlSerializer
import numpy as np
import os
import glob
import json
import re
import shutil
import time
from typing import Any, Dict, List, Sequence, Tuple, Optional
from step_guard import guarded_step
from abfe_core import (
    ACESoftcorePotential,
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
)
from ibs_engine import (
    generate_overlapping_windows,
    configure_charge_transfer_decharging,
    configure_coalchemical_neutral_decharging,
    _build_platform_properties,
    _timed,
)
try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Increment whenever the pilot metric, lambda placement, or thermodynamic-window
# partitioning semantics change.  Pre-opt caches without this exact version are
# not safe to resume because an equal-sized lambda array can still represent a
# completely different Hamiltonian path.
#   version 2: warmup-failure feedback switched from evolving-IBS-mixture
#              std(beta Delta-u) + arithmetic-midpoint bisection + halved-edge-
#              length bookkeeping to split-first / fixed-H bidirectional overlap
#              probe / measured-only insertion.
#   version 3: production-time ESS auto-repair (the OTHER failure branch in
#              _run_stage_with_overlap_autorepair, triggered after a full stage
#              run reports low ESS rather than a warmup exception) unified with
#              the same split-first / fixed-H probe / measured-only insertion
#              policy for the vdw stage, replacing the old worst-per-lambda-
#              state + arithmetic-midpoint + fixed re-partition in
#              refine_stage_lambda_path_by_overlap (still used for the coul
#              stage only, where the fixed-H probe's required
#              ibs_wrap._common_system_xml is not yet constructed).
#   version 4: partition_windows_by_thermodynamic_length gained a hard
#              max_states_per_window cap (default 6) alongside the existing
#              distance budget. Distance-only partitioning could pack an
#              arbitrary number of states into one window whenever pilot
#              edges were short and numerous (observed: 17 edges at ~0.85
#              each, max_window_length=6.0 -> three 8-state windows), which
#              is exactly the "one IBS bias, too many states" case this cap
#              exists to prevent. A v3 cache's window_ranges may not respect
#              this cap and must not be resumed as-is.
#   version 5: the production-ESS batch-split branch in
#              _run_stage_with_overlap_autorepair now canonicalizes
#              window_ranges (canonicalize_window_ranges) after splitting
#              several overlapping failing parent windows independently.
#              Splitting each parent on its own midpoint can produce a child
#              that lands entirely inside a NEIGHBORING parent's child (IBS
#              windows overlap by design) -- observed: 5 overlapping 6-state
#              parents over 18 states all split in one round produced 10
#              windows, 4 of them strictly contained in a neighbor, instead
#              of the minimal connected 6-window chain. Not a correctness bug
#              (coverage was still complete, just redundantly re-sampled),
#              but a v4-or-earlier cache's window_ranges may contain this
#              kind of redundancy and must be regenerated, not resumed as-is.
#   version 6: split_window_from_warmup_failure now reflows the split window's
#              immediate next neighbor down to single-state overlap when it
#              still shares more than one state with the new right child.
#              The initial pilot layout intentionally allows wide cumulative
#              overlap (pilot_overlap_thermodynamic_length), so a 6-state
#              parent sharing 3 states with its next neighbor is by design --
#              but after that parent splits, the untouched neighbor still
#              shares those same 3 states with the much smaller right child
#              (observed: a 4-state child sharing 3 states, 75%, with its
#              neighbor), which canonicalize_window_ranges deliberately does
#              not touch (partial, non-containing overlap is legitimate
#              elsewhere). A v5-or-earlier cache produced by a warmup-failure
#              split may still carry this disproportionate overlap and should
#              be regenerated, not resumed as-is.
#   version 7: fixed a real boundary-condition bug: splitting a window into
#              two children that each need >=3 states (a 2-state IBS window
#              is statistically fragile) while sharing exactly 1 state
#              requires a parent of at least 3+3-1=5 states, but
#              split_window_from_warmup_failure/plan_vdw_overlap_repair_targets
#              used min_states_before_split=4 and a child-size floor of 2 --
#              so a 4-state window (e.g. [2,6)) would get bisected into a
#              2-state child ([2,4)) plus a 3-state child ([3,6)), the exact
#              kind of fragile window this whole split-vs-probe policy exists
#              to avoid. Both now require >=5 states before splitting; K<=4
#              goes straight to the fixed-H bidirectional overlap probe
#              instead (see IBS_BIAS_PROTOCOL_VERSION's matching K<=4 change).
#              A v6-or-earlier cache may contain a window produced by the old
#              buggy K=4 bisection and must be regenerated, not resumed.
#   version 8: withdrawn.  A briefly implemented design attempted to gate the
#              grid with adjacent fixed-H/MBAR overlap.  That is a replica-path
#              diagnostic, not the IBS Log-Sum-Exp fixed-point criterion, and
#              it caused one MBAR solve per edge.  Any v8 cache is invalid.
#   version 9: withdrawn.  It removed fixed-H/MBAR as the schedule arbiter but
#              still fed the grid into the old overlapping-window architecture
#              and recursively produced sliding K=2 ensembles such as [6:8]
#              and [7:9].  That tests local-window stitching, not one integrated
#              IBS ensemble.
#   version 10: withdrawn.  Putting every vanishing lambda node into one IBS
#               ensemble contradicted the paper's few-state/subinterval design
#               and can exceed OpenMM's 32-CV CustomCVForce limit (two CVs are
#               currently built per lambda state).
#   version 11: withdrawn.  It copied the paper's example split at lambda=0.5
#               literally; on this ABFE metric that produced a 14-state first
#               ensemble and a 5-state tail ensemble.  The example boundary is
#               not a universal vanishing-path rule.
#   version 12: thermodynamic length places all lambda nodes (therefore the
#               lambda->0 region can become physically denser), then the path is
#               partitioned along that thermodynamic coordinate into few-state
#               IBS ensembles representing about 4-5 conventional lambda
#               intervals each.  Every lambda interval belongs to exactly one
#               ensemble; adjacent ensembles deliberately reuse their boundary
#               node as a common free-energy reference.  The legacy layout that
#               reused two nodes (and therefore duplicated one lambda interval),
#               plus a fixed lambda=0.5 boundary, are forbidden.
#   version 13: v12's 4-5-interval grouping put 6 states (5 intervals) into
#               window 0 at the fully-coupled vdW endpoint (lambda 1.0 ->
#               0.963), a real GPU run of that window hit IBSWarmupConvergence
#               Error: occupation stuck at state 0 (mean_p~0.994), TMBAR
#               min_absolute_ess~1.0 -- state0/state1 overlap this wide a
#               window can't bridge within the online-learning budget. Checked
#               against the real cached pilot metric_g (this session): raising
#               total state count to compensate needs ~64 states (the 8-point
#               pilot grid near lambda=1.0 is too coarse to place finer density
#               there anyway); adding a cap to the thermodynamic-length metric
#               (borrowing the old, since-withdrawn log1p/clip density) was
#               checked numerically and makes window 0 WIDER (1.0->0.934), not
#               narrower -- rejected. What's actually verified (against the
#               same cached lambda array) is finer grouping: 3
#               intervals/ensemble instead of 5 puts only 4 states (3
#               intervals) in window 0 -- narrower, still one common-boundary-
#               node design, no change to the lambda placement/density
#               computation itself.
#   version 14: v13's uniform 3-interval regrouping (4 states) STILL failed a
#               real GPU run of window 0: occupation only moved 99.4%->96.9% at
#               state 0, min_absolute_ess still ~1.0. Per the paper's own Sec
#               2.4 iterative subdivide-until-stable procedure, and per the
#               user's explicit rejection of uniformly shrinking the whole path
#               to 2-state windows everywhere (throws away IBS's efficiency
#               benefit for the rest of the path, which isn't failing),
#               grouping is now position-dependent: vanishing_subdomain_ranges_
#               from_lambdas gained an optional first_ensemble_target_intervals
#               override, used only for window 0. User capped this at 2
#               intervals (3 states) -- matching the state count of the
#               2026-07-17 configuration known to have actually converged for
#               this exact endpoint -- explicitly not 1 interval/2 states. New
#               module constant VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS is the
#               single source of truth for this value: every caller that
#               independently recomputes "what window_ranges SHOULD look like"
#               to validate a cache (abfe_pipeline.py has 4 such call sites)
#               must pass this same constant, or a freshly-regenerated v14
#               path would fail its own cache-validation check on the very next
#               resume. Still one shared boundary node between every adjacent
#               pair, no overlap=2.
#   version 15: v14's window0-specific 2-interval override (3 states) STILL
#               failed a real GPU run: occupation 98.2% at state 0,
#               min_absolute_ess~1.0 again. Three independent real runs (6/4/3
#               states) now conclusively prove regrouping alone cannot fix
#               this -- it never changes the actual lambda values, only which
#               states share one IBS bias, and all three runs used
#               essentially the same ~0.006-0.007 state0/state1 spacing
#               because window_ranges is derived from, but does not feed back
#               into, redistribute_lambda_by_thermodynamic_length's lambda
#               placement. Root cause found by reading the cached pilot data
#               directly: the coarse pilot grid's very first segment
#               (lambda=1.0 -> ~0.94) alone contributed ~47% of the entire
#               path's thermodynamic length, but is defined by only 2 raw
#               pilot points, so the arc-length redistribution could only
#               ever place new states *linearly* inside it -- no real
#               measurement of how the true difficulty is actually
#               distributed there. Fix: optimize_stage2_vanishing now calls
#               the new _refine_pilot_grid_in_steep_segments after the coarse
#               pilot pass, which probes additional points strictly inside
#               whichever segment dominates total thermodynamic length
#               (default threshold 20%) and merges them back in before
#               redistribution runs. This is upstream of and independent from
#               the v13/v14 grouping constants/override, which are left
#               unchanged.
#   version 16: v15's pilot-grid refinement did trigger on a real GPU run (26
#               pilot points, 2 refinement rounds) but did NOT fix window 0 --
#               it revealed the true difficulty is a sharp, non-monotonic
#               metric_g peak around lambda~0.96-0.97, roughly 50x the
#               endpoint value, which sits OUTSIDE window 0's own span
#               (window 0 only covers lambda=1.0 -> ~0.9848). Window 0 itself
#               still failed (occupation 98.2%, min_absolute_ess~1.0) for a
#               separate, more mundane reason: the probe (crude by design,
#               per the user's own diagnosis) cannot give a precise estimate
#               of window 0's *own* internal energy landscape either. Instead
#               of tuning the probe further, this version uses REAL measured
#               data: the failed run's own IBSSampler.save_ibs_state left
#               behind window 0's real tmbar_history (~1000 real sampled
#               frames across its 3 states). Re-solving those with
#               GlobalMBARAnalyzer.solve_stage_integrated gives real, measured
#               f_k at window 0's 3 real lambda points: state0/1 real
#               Delta_f=-25.3 kJ/mol (~10.2 kT), state1/2 real
#               Delta_f=-15.7 kJ/mol (~6.3 kT) -- both far above the ~2-3 kT
#               overlap budget IBS/BAR needs, and critically the implied
#               dF/dlambda is nearly identical across both edges (~2694 vs
#               ~2702 kJ/mol per unit lambda) -- this specific span is close
#               to LINEAR, not pathological, so it is a "too few states for a
#               steep but well-behaved slope" problem, straightforwardly
#               fixable with real-Delta_f-placed intermediate states (see
#               repair_stage2_window0_real_delta_f.py). Window 0 goes from 3
#               states to 7 (6 real-Delta_f-equalized steps, ~6.8 kJ/mol/step
#               ~2.7 kT), so VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS goes
#               2->6. This does NOT touch the pilot-based lambda placement for
#               any other window (still v15's refined pilot grid); only
#               window 0's own lambdas_var entries are replaced, by the repair
#               script, with real-data-derived values.
#   version 17: real GPU run of the FULL 5-window vanishing path (v16
#               grouping + v26 IBS bias-learner fixes) converged windows 0-3
#               but window 4 (states [15,16,17], lambda=[0.9051,0.8322,0.0])
#               showed occupation collapsed onto a single state
#               (coverage_ess=1.0) no matter how much online learning ran --
#               a genuinely different failure shape from windows 0's history
#               above (control-loop/bootstrap problems in the *learner*):
#               here the lambda schedule itself is unbridgeable. Root cause,
#               found by hand-integrating the already-cached pilot's own
#               mean_dU_dlambda_kJ_mol (path_diagnostics.pilot_points, no new
#               simulation needed): the 0.83->0.0 tail carries a real,
#               sustained ~-200 kJ/mol (~80 kT) free-energy change, but its
#               local VARIANCE is small (std 78->0.02 kJ/mol) -- v13-v16's
#               metric_g=beta^2*Var[dU/dlambda] and the equal-thermodynamic-
#               length placement built on it have no mean-gradient term at
#               all, so this ~80-kT-wide region was allocated only ~1.3% of
#               the total path length, packing 17 of 18 states into the top
#               17% of the raw lambda range and leaving one un-bridgeable
#               final interval to cover the rest. This is a structural
#               allocation bug (confirmed independently: window grouping,
#               vanishing_subdomain_ranges_from_lambdas, is a blind interval-
#               COUNT partition with no energy awareness of its own, and is
#               hard-enforced by abfe_pipeline.py's Stage-2 cache-validation
#               gate recomputing+requiring an exact window_ranges match --
#               so a hand-picked, energy-aware window_ranges would silently
#               be rejected on the next resume regardless), not something the
#               online IBS bias-learner can ever converge by retrying harder.
#               Fixed at the source: redistribute_vanishing_lambda_subdomains
#               now places lambda at equal cumulative |Delta F| from the
#               pilot's own measured mean gradient (trapezoidal TI via the
#               shared _pilot_ti_cumulative_f helper, also used by
#               estimate_f_k_from_pilot_ti) through the already-validated
#               redistribute_lambda_by_delta_f (same function the window-0
#               real-Delta_f repair already used successfully), instead of
#               the variance-only metric_g. No new pilot simulation is
#               required. vanishing_subdomain_ranges_from_lambdas' interval-
#               count grouping and abfe_pipeline.py's validation gate are
#               UNCHANGED -- their own correctness assumption (equal count of
#               correctly-spaced intervals ~= equal difficulty) is restored,
#               not bypassed, once the upstream spacing is no longer wrong.
#               metric_g/the variance estimator are still computed and
#               stored (now diagnostic-only, and still drive
#               _refine_pilot_grid_in_steep_segments' extra-probing
#               decision) -- this version does not touch window 0's separate
#               first_ensemble_target_intervals=6 override, which addresses
#               a different, already-diagnosed high-*variance* problem near
#               lambda=1; re-verify it after this fix rather than assume it
#               still needs the same value.
#   version 18: withdraws v17's equal-cumulative-|Delta F| lambda placement.
#               A lambda-dependent additive energy constant changes Delta F
#               without changing any Boltzmann distribution or phase-space
#               overlap, so |Delta F| is not a valid state-density coordinate.
#               The beta^2 Var[dU/dlambda] Fisher probe first generates exactly
#               17 conventional lambda nodes. Human endpoint densification then
#               INSERTS four nodes without moving/deleting any probe node: three
#               quarter-points inside base edge 0->1 and one midpoint inside
#               base edge 1->2. Final nodes are lambda_0..lambda_20 (21 unique,
#               lambda_20=0). The five human-drawn CLOSED windows are [0,5],
#               [5,9], [9,13], [13,17], [17,20], represented in Python as the
#               half-open ranges below. They contain 6+5+5+5+4=25 state slots,
#               with exactly four single-node boundary reuses and no duplicated
#               lambda edge.
#   version 19: the 17-node production base path is now the deterministic
#               quadratic schedule lambda=x^2, x=linspace(1,0,17).  v18 let
#               the Fisher metric place the production nodes; in the observed
#               run this collapsed the decoupled tail to 0.9225, 0.8382, 0.0,
#               leaving an unbridgeable final edge despite four extra nodes
#               having been inserted at the opposite (lambda~1) endpoint.
#               Fisher probing remains diagnostic, but it no longer gets to
#               remove geometric coverage near lambda=0.  The existing four
#               lambda~1 insertions are retained, so both endpoints are dense.
#   version 20: v19 real warm-up exposed two Fisher-length gaps at the coupled
#               endpoint despite its geometric lambda~1 insertions. Insert one
#               measured thermodynamic midpoint into each of the two longest
#               production edges, then split the former first ensemble into
#               two. The observed pilot gives ~0.980304 and ~0.962885; these
#               values are computed from the cached sqrt(g) arc length, never
#               hard-coded. Final path: 23 states / 6 immutable ensembles.
#   version 21: the measured Fisher metric now CONTROLS production lambda
#               placement instead of only annotating it.  v19/v20 placed the
#               nodes with a fixed quadratic schedule plus four hand-picked
#               lambda~1 insertions plus two bridge bisections, and discarded
#               the equal-thermodynamic-length solution it had just computed
#               (probe_controls_base_lambda_placement=false).  The observed
#               consequence on the real cached path: window 0 held 41.32 of the
#               path's 47.22 total thermodynamic length in four edges (8.82,
#               8.82, 11.83, 11.83) while the remaining 18 edges shared 5.90,
#               with tail edges as short as 0.0002 -- zero overlap in window 0,
#               IBS occupancy degenerating to a hard argmax (mean_p=1.000000),
#               TMBAR never self-consistent.
#
#               v18 already tried pure Fisher equipartition and produced the
#               opposite failure (the decoupled tail collapsed to 0.9225,
#               0.8382, 0.0 -- an unbridgeable final edge), which is why v19
#               reverted to a geometric schedule.  v21 does not repeat either
#               mistake: nodes equipartition a BLEND of normalized arc length
#               and geometric progress,
#                   u(lam) = (1-beta)*s_hat(lam) + beta*(1-lam),
#               so placement is metric-driven while every edge still satisfies
#               the provable geometric bound |d lam| <= 1/(beta*(n_states-1)).
#               beta=VANISHING_GEOMETRIC_FLOOR_WEIGHT.  With no metric available
#               (fallback paths) the quadratic schedule remains, now generated
#               directly at the final state count.
#
#               Ordering note: this only pays off on a pilot measured under
#               SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2.  The
#               old metric's concentration near lambda=1 was itself an artifact
#               of treating alpha_lj as an absolute nm^6 offset (~685x too
#               large), which compressed the entire hard->soft core transition
#               into lambda_vdw in [0.96, 1].  Re-pilot before trusting any
#               placement computed from a cached metric.
# v22 (2026-09-03): 在 v21 的度规布点之后增加一个可选的**自由能定向加密**后处理
#   （densify_lambdas_by_free_energy）。动机：pilot 实测显示这条路径上平均梯度
#   <dU/dlambda> 与度规 beta^2 Var[dU/dlambda] 是**反相关**的——4W53 复合物腿在
#   lambda=1 处 <dU/dl>=-144.8 kJ/mol 而 g=20.4，在 lambda=0.69 处 <dU/dl>=-46.7
#   而 g=179。纯 sqrt(g) 布点因此系统性地在自由能落差最大的 lambda~1 段少放点：
#   16 态时前 3 条边装了 54% 的自由能却只占 17% 的热力学长度，单条边 13.6 kJ/mol
#   (5.5 kT)。这正是 window 0 历史上 ESS 塌缩的机制（不是重叠不足——那几条边的
#   delta~0.6，交换接受率 66%），而重新分窗救不了它（穷举过所有合法分窗，最大窗
#   ΔF 完全不变），全局改布点权重是零和的（ΔF 砍一半 delta_max 要涨到 1.8）。
#   v22 保持总态数不变，只把节点从平坦中段挪到陡峭段：14+2 使 4W53 两条腿的最大
#   边 ΔF 从 13.6/9.2 降到 7.7/6.4（比 23 态生产路径的 9.9/6.6 还好），delta_max
#   1.08/1.10 仍在 delta~1 目标上。
#   ⚠️ free_energy_densify_points=0（默认）时布点与 v21 逐字节相同。
#   与 v20 那个「lambda~1 四点增密」的区别：v20 的点是**手挑常数**、完全无视实测
#   度规，v22 的点由 pilot 实测的 <dU/dlambda> 推出来，且不动 v21 的基础布点。
THERMODYNAMIC_PATH_PROTOCOL_VERSION = 22

VANISHING_PROBE_BASE_STATE_COUNT = 17
VANISHING_FINAL_STATE_COUNT = 23
# Geometric floor weight in the blended placement measure.  Larger = closer to a
# uniform-lambda path (safer coverage, less metric control); smaller = closer to
# pure equal-thermodynamic-length (better overlap where the metric is real, but
# v18 showed pure equipartition can strand the decoupled endpoint).  0.3 bounds
# any single lambda gap at 1/(0.3*22) = 0.152 for the 23-state vanishing path.
VANISHING_GEOMETRIC_FLOOR_WEIGHT = 0.3
# [v22] 默认 0 = 关闭自由能定向加密，布点与 v21 逐字节相同。设成 k>0 时，基础布点
# 用 (final_state_count - k) 态，再贪心插入 k 个点：每次找 |ΔF| 最大的那条边、在
# 它的等 ΔF 中点插一个。总态数不变，所以采样成本完全不变。
VANISHING_FREE_ENERGY_DENSIFY_POINTS = 0
VANISHING_FIXED_WINDOW_RANGES = (
    (0, 5),
    (4, 8),
    (7, 12),
    (11, 16),
    (15, 20),
    (19, 23),
)

# Single source of truth for the position-dependent override above -- every
# caller of vanishing_subdomain_ranges_from_lambdas/redistribute_vanishing_
# lambda_subdomains that needs to know "what does the CURRENT vanishing design
# actually produce" (both the real generator in optimize_stage2_vanishing and
# every cache-validation call site in abfe_pipeline.py) must use this constant,
# not a hardcoded literal, or validation and generation will silently diverge.
VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS: Optional[int] = 4

VANISHING_TARGET_INTERVALS_PER_ENSEMBLE = 3
VANISHING_MIN_INTERVALS_PER_ENSEMBLE = 2
# OpenMM CustomCVForce supports at most 32 CVs.  build_ibs_dual_system currently
# adds two CVs per lambda state (interaction + zero restraint bookkeeping).
VANISHING_MAX_STATES_PER_IBS_ENSEMBLE = 16


# 探针系统里原生 NonbondedForce（PME）单独占一个 force group。
#
# 🔑 [2026-09-10] 它原来被塞进 group 1，和软核力混在一起，而度规的有限差分
# 只读 group 1。问题是 group 1 里那两项对 λ 的依赖**不一样**：
#   * 软核 ACES 力：同时带 lam_coul 和 lam_vdw；
#   * 原生 NB（带电腿）：只带 lam_coul —— B3 的 PME ParameterOffset 装在它上面，
#     所以 Stage 1 差分 lam_coul 时**必须**算上它。
# Stage 2 固定 lam_coul=0、只差分 lam_vdw，此时原生 NB 是个与 lam_vdw 完全无关的
# 常数，但量级在 ~10^6 kJ/mol：在它上面做差再除以 delta≈0.02，是灾难性相消，
# 而 mixed precision 下两次求值未必逐比特相同。
# 解法不是"把它挪去 group 0"（那会让 Stage 1 的 lam_coul 依赖整个消失），
# 而是给它自己的 group，再由差分按**被差分的参数**选 group 集合。
# 组号 3：探针里 0=默认、1=软核、2=配体内部、6=co-ion flat-bottom 限制，3 空着。
# ⚠️ force group 只影响能量分解读数，不影响积分的哈密顿量。
PREOPT_NATIVE_NONBONDED_FORCE_GROUP = 3


def validate_single_shared_boundary_ranges(
    window_ranges: List[Tuple[int, int]],
    n_states: int,
) -> None:
    """Validate the actual state/edge sets of half-open IBS window ranges.

    This intentionally avoids reasoning from tuple endpoints.  For example,
    ``(0, 7)`` contains states {0..6} and ``(6, 10)`` contains {6..9}; their
    intersection is exactly {6}.  Sharing {5, 6} would duplicate one lambda
    edge and is rejected.
    """
    ranges = [(int(start), int(end)) for start, end in window_ranges]
    if not ranges:
        raise ValueError("vanishing IBS window_ranges 不能为空")
    actual_state_sets = []
    edge_use_count = np.zeros(max(0, int(n_states) - 1), dtype=int)
    for start, end in ranges:
        if not (0 <= start < end <= int(n_states)):
            raise ValueError(
                f"vanishing IBS 窗口越界或为空: {(start, end)}, n_states={n_states}"
            )
        states = set(range(start, end))
        actual_state_sets.append(states)
        for edge in range(start, end - 1):
            edge_use_count[edge] += 1

    covered_states = set().union(*actual_state_sets)
    if covered_states != set(range(int(n_states))):
        raise RuntimeError(
            "vanishing IBS 窗口没有完整覆盖所有 lambda 状态: "
            f"covered={sorted(covered_states)}, n_states={n_states}"
        )
    if edge_use_count.size and not np.all(edge_use_count == 1):
        raise RuntimeError(
            "每条 lambda 边必须恰好属于一个 IBS ensemble；"
            f"实际 edge_use_count={edge_use_count.tolist()}"
        )

    for idx in range(1, len(actual_state_sets)):
        shared = actual_state_sets[idx - 1] & actual_state_sets[idx]
        expected = {ranges[idx][0]}
        if shared != expected:
            raise RuntimeError(
                "相邻 IBS ensemble 必须严格只共享一个边界 lambda："
                f"windows={ranges[idx - 1]}, {ranges[idx]}, "
                f"shared={sorted(shared)}, expected={sorted(expected)}"
            )
    for left in range(len(actual_state_sets)):
        for right in range(left + 2, len(actual_state_sets)):
            shared = actual_state_sets[left] & actual_state_sets[right]
            if shared:
                raise RuntimeError(
                    "非相邻 IBS ensemble 不得共享 lambda："
                    f"windows={ranges[left]}, {ranges[right]}, shared={sorted(shared)}"
                )


def human_vanishing_initial_lambdas(requested_base_n_states: int) -> np.ndarray:
    """Return the conventional *probe* input grid (default: 17 points).

    This is the grid the Fisher pilot measures on (before
    _refine_pilot_grid_in_steep_segments adds probes); the production path is
    placed separately by blended_metric_vanishing_lambdas at
    VANISHING_FINAL_STATE_COUNT nodes.

    🔑 [2026-08-27] Before this, ``requested_base_n_states`` had to be exactly
    ``VANISHING_PROBE_BASE_STATE_COUNT`` (17) or this raised — meaning
    ``--stage2-n-states``/``stage2_n_states`` in runabfe.py's CLI/presets was
    a lie for any other value: it parsed fine and then crashed here. The
    linspace construction below never assumed exactly 17 points; the
    hard-equality check was gatekeeping a value nothing downstream in *this*
    function actually depended on. Widened to any n>=2 probe grid — the probe
    density only affects how finely the Fisher metric g(lambda) is sampled
    before placement, not the production window layout (see
    VANISHING_FINAL_STATE_COUNT / vanishing_subdomain_ranges_from_lambdas,
    which remain their own, separately-gated contract).

    🔑 [2026-08-28] The 2026-08-27 widening above removed the fail-fast: a
    stray/wrong ``n_states`` (e.g. a stale config value) used to crash here
    before any GPU work happened, now it silently runs to completion instead
    — real incident: a 4W53 production run sat at ``stage2_n_states=8`` from
    a leftover config, burned real GPU integration steps, and only got
    noticed when the user manually interrupted it. Not re-adding the hard
    equality check (``--stage2-n-states`` must stay configurable to any
    n>=2); instead, warn loudly whenever the value is non-default so it's
    visible in the log before compute is spent, not just from an
    unexplained slow run.
    """
    n = int(requested_base_n_states)
    if n < 2:
        raise ValueError(f"vanishing pilot 探针网格至少需要 2 个点；收到 base_n_states={n}")
    if n != VANISHING_PROBE_BASE_STATE_COUNT:
        print(
            f"  [WARN] [vanishing pilot 探针网格] 探针密度 base_n_states={n}，"
            f"偏离常规默认值 {VANISHING_PROBE_BASE_STATE_COUNT}——如果这不是故意"
            f"传的，请检查 --stage2-n-states / config 里的 stage2_n_states 是不是"
            f"设错了，再决定要不要现在就烧 GPU 时间跑下去。"
        )
    return np.linspace(1.0, 0.0, n)


def quadratic_vanishing_base_lambdas(
    n_states: int = VANISHING_FINAL_STATE_COUNT,
) -> np.ndarray:
    """Metric-free fallback path ``lambda=x^2`` (dense near lambda=0).

    Only used when no pilot metric is available (see the fallback branch in
    abfe_pipeline).  When a metric exists, blended_metric_vanishing_lambdas
    places the nodes instead -- see THERMODYNAMIC_PATH_PROTOCOL_VERSION 21.
    """
    if int(n_states) < 2:
        raise ValueError("vanishing 路径至少需要 2 个态")
    x = np.linspace(1.0, 0.0, int(n_states))
    base = np.square(x)
    base[0], base[-1] = 1.0, 0.0
    if not np.all(np.diff(base) < 0.0):
        raise RuntimeError("平方 vanishing 基础路径没有严格从 1 递减到 0")
    return base


def vanishing_max_lambda_gap_bound(
    n_states: int = VANISHING_FINAL_STATE_COUNT,
    geometric_floor_weight: float = VANISHING_GEOMETRIC_FLOOR_WEIGHT,
) -> float:
    """Provable per-edge |Delta lambda| ceiling of the blended placement.

    Consecutive nodes are spaced by exactly ``du = 1/(n_states-1)`` in the
    blended coordinate ``u = (1-beta)*s_hat + beta*(1-lambda)``.  Because
    ``s_hat`` is non-decreasing along the path, ``du >= beta*|d lambda|``,
    hence ``|d lambda| <= du/beta``.  This is what keeps a metric that is
    heavily concentrated at one end from stranding the other end the way pure
    equipartition did in v18.
    """
    beta = float(geometric_floor_weight)
    if not (0.0 < beta < 1.0):
        raise ValueError(f"geometric_floor_weight 必须在 (0,1)：{beta}")
    if int(n_states) < 2:
        raise ValueError("vanishing 路径至少需要 2 个态")
    return 1.0 / (beta * float(int(n_states) - 1))


def blended_metric_vanishing_lambdas(
    pilot_lambdas,
    metric_g,
    n_states: int = VANISHING_FINAL_STATE_COUNT,
    geometric_floor_weight: float = VANISHING_GEOMETRIC_FLOOR_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place production lambdas by equipartitioning a metric/geometry blend.

    Returns ``(lambdas, pilot_cumulative, edge_thermodynamic_lengths)``.

    ``s_hat(lambda)`` is the pilot arc length ``integral sqrt(g) d lambda``
    normalized to [0,1]; ``1-lambda`` is geometric progress from the coupled to
    the decoupled endpoint.  Equipartitioning ``(1-beta)*s_hat + beta*(1-lambda)``
    concentrates nodes where the measured metric is large while guaranteeing
    ``vanishing_max_lambda_gap_bound`` -- neither v19/v20's "ignore the metric"
    nor v18's "let the metric strand the tail".
    """
    lam = np.asarray(pilot_lambdas, dtype=float).ravel()
    g = np.asarray(metric_g, dtype=float).ravel()
    if lam.size != g.size or lam.size < 2:
        raise ValueError("pilot_lambdas/metric_g 必须等长且至少包含两个点")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("pilot/metric 必须有限且 metric_g 非负")
    order = np.argsort(-lam)
    lam = lam[order]
    g = g[order]
    if not np.all(np.diff(lam) < 0.0):
        raise ValueError("pilot lambda 必须唯一且严格单调")
    if not (np.isclose(lam[0], 1.0) and np.isclose(lam[-1], 0.0)):
        raise ValueError("pilot lambda 必须覆盖完整的 1 -> 0 区间")
    beta = float(geometric_floor_weight)
    if not (0.0 < beta < 1.0):
        raise ValueError(f"geometric_floor_weight 必须在 (0,1)：{beta}")

    root_g = np.sqrt(np.maximum(g, 1.0e-12))
    cumulative = np.concatenate((
        [0.0],
        np.cumsum(0.5 * (root_g[:-1] + root_g[1:]) * np.abs(np.diff(lam))),
    ))
    total_length = float(cumulative[-1])
    if not np.isfinite(total_length) or total_length <= 0.0:
        raise ValueError(
            f"pilot 热力学总长非正/非有限（{total_length}），无法用度规布点"
        )
    s_hat = cumulative / total_length
    blended = (1.0 - beta) * s_hat + beta * (1.0 - lam)
    if not np.all(np.diff(blended) > 0.0):
        raise RuntimeError("混合布点坐标不是严格递增的，无法反解 lambda")

    targets = np.linspace(0.0, 1.0, int(n_states))
    placed = np.interp(targets, blended, lam)
    placed[0], placed[-1] = 1.0, 0.0
    if not np.all(np.diff(placed) < 0.0):
        raise RuntimeError("混合布点没有产生严格递减的 lambda 路径")

    gap_bound = vanishing_max_lambda_gap_bound(int(n_states), beta)
    realized_gap = float(np.max(np.abs(np.diff(placed))))
    # 允许极小的插值/端点钳制浮点余量，但不允许真正越界。
    if realized_gap > gap_bound * (1.0 + 1.0e-6):
        raise RuntimeError(
            f"混合布点越过几何覆盖上限：max|Δλ|={realized_gap:.6f} > {gap_bound:.6f}"
        )
    placed_cumulative = np.interp(placed[::-1], lam[::-1], cumulative[::-1])[::-1]
    return placed, cumulative, np.abs(np.diff(placed_cumulative))


def _free_energy_arclength(pilot_lambdas, mean_dU_dlambda):
    """Cumulative |<dU/dlambda>| integral as a strictly increasing function of
    ``u = 1 - lambda``.

    Returns ``(u_grid_ascending, cumulative_ascending)``.  Total variation, not
    net displacement -- same reasoning as
    ``partition_windows_by_delta_f_budget``: <dU/dlambda> is not guaranteed
    monotonic along a softcore path, and a net-displacement measure would call a
    segment that goes up and comes back "flat" and refuse to densify it.
    """
    lam = np.asarray(pilot_lambdas, dtype=float).ravel()
    grad = np.asarray(mean_dU_dlambda, dtype=float).ravel()
    if lam.size != grad.size or lam.size < 2:
        raise ValueError("pilot_lambdas/mean_dU_dlambda 必须等长且至少两个点")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(grad)):
        raise ValueError("pilot lambda 与 <dU/dlambda> 必须有限")
    order = np.argsort(lam)          # lambda 升序 -> u 降序，取反得到 u 升序
    lam = lam[order][::-1]
    grad = grad[order][::-1]
    u = 1.0 - lam
    if not np.all(np.diff(u) > 0.0):
        raise ValueError("pilot lambda 必须唯一且严格单调")
    mag = np.abs(grad)
    cumulative = np.concatenate((
        [0.0], np.cumsum(0.5 * (mag[:-1] + mag[1:]) * np.diff(u)),
    ))
    return u, cumulative


def _pilot_mean_gradients_or_none(pilot_points) -> Optional[np.ndarray]:
    """Collect ``mean_dU_dlambda_kJ_mol`` from pilot points, or None.

    Returns None -- never a substitute value -- if ANY point is missing the key
    or carries a non-finite gradient.  Some probe paths (a failed
    ``_sample_scalar_metric``, reduced test doubles) legitimately produce points
    without it.  Downstream, None means the free-energy diagnostics are simply
    absent and ``free_energy_densify_points > 0`` fails closed, which is the
    intended behaviour: densifying by free energy with a guessed gradient would
    silently move production lambda states based on a number nobody measured.
    """
    values = []
    for point in pilot_points:
        try:
            value = float(point["mean_dU_dlambda_kJ_mol"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        values.append(value)
    if len(values) < 2:
        return None
    return np.asarray(values, dtype=float)


def densify_lambdas_by_free_energy(
    lambdas,
    pilot_lambdas,
    mean_dU_dlambda,
    n_extra: int,
    min_lambda_gap: float = 1.0e-4,
) -> np.ndarray:
    """[THERMODYNAMIC_PATH_PROTOCOL_VERSION=22] Insert ``n_extra`` states into
    the edges carrying the most free energy, keeping every existing state.

    ``lambdas`` is a strictly decreasing 1 -> 0 production path (typically the
    output of :func:`blended_metric_vanishing_lambdas`); ``pilot_lambdas`` and
    ``mean_dU_dlambda`` are the pilot grid and its measured
    ``<dU/dlambda>`` (kJ/mol), i.e. ``pilot_points[i]["mean_dU_dlambda_kJ_mol"]``
    -- data the probe ALREADY collects, so this costs no extra sampling.

    Greedy, one point at a time: find the edge with the largest |Delta F|, insert
    the lambda that splits that edge's |Delta F| in half, repeat.  Splitting by
    free energy (not by lambda, not by thermodynamic length) is the whole point
    -- the thermodynamic metric is what is already driving the base placement,
    and on a real path the two disagree exactly where it matters.

    Fail-closed: refuses to insert into an edge whose |Delta F| is numerically
    zero (nothing to split -- densifying there would be arbitrary), and refuses
    to produce two states closer than ``min_lambda_gap``.  Endpoints
    ``lambda = 1`` and ``lambda = 0`` are never moved.
    """
    path = np.asarray(lambdas, dtype=float).ravel().copy()
    n_extra = int(n_extra)
    if n_extra < 0:
        raise ValueError(f"n_extra 不能为负：{n_extra}")
    if n_extra == 0:
        return path
    if path.size < 2 or not np.all(np.diff(path) < 0.0):
        raise ValueError("待加密的 lambda 路径必须严格递减且至少 2 态")

    u_pilot, cum = _free_energy_arclength(pilot_lambdas, mean_dU_dlambda)
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"pilot 的 |<dU/dlambda>| 累积积分非正/非有限（{total}），无法按自由能加密"
        )
    # 严格递增才可反解；相邻相等只可能来自梯度恒零的区段。
    if not np.all(np.diff(cum) > 0.0):
        raise ValueError(
            "pilot 自由能弧长不是严格递增的（存在 <dU/dlambda> 恒为零的区段），"
            "无法按自由能反解插点"
        )
    f_of = lambda lam_q: np.interp(1.0 - np.asarray(lam_q, dtype=float), u_pilot, cum)

    for _ in range(n_extra):
        f_nodes = f_of(path)
        edge_df = np.abs(np.diff(f_nodes))
        worst = int(np.argmax(edge_df))
        if not np.isfinite(edge_df[worst]) or edge_df[worst] <= 0.0:
            raise RuntimeError(
                "所有边的 |Delta F| 都为零，没有可加密的目标；"
                "请检查 pilot 的 mean_dU_dlambda 是否真的被采到"
            )
        target = 0.5 * (f_nodes[worst] + f_nodes[worst + 1])
        # cum 随 u 严格递增 -> 用 u 反解再换回 lambda
        u_new = float(np.interp(target, cum, u_pilot))
        lam_new = 1.0 - u_new
        hi, lo = float(path[worst]), float(path[worst + 1])
        if not (lo + min_lambda_gap <= lam_new <= hi - min_lambda_gap):
            raise RuntimeError(
                f"自由能加密解出的 lambda={lam_new:.6f} 不在待拆边 "
                f"({hi:.6f}, {lo:.6f}) 内、或与端点间距小于 {min_lambda_gap}；"
                "拒绝插入退化状态"
            )
        path = np.insert(path, worst + 1, lam_new)

    if not np.all(np.diff(path) < 0.0):
        raise RuntimeError("自由能加密后 lambda 路径不再严格递减")
    if not (np.isclose(path[0], 1.0) and np.isclose(path[-1], 0.0)):
        raise RuntimeError("自由能加密不得移动 lambda=1 / lambda=0 端点")
    return path


def edge_free_energy_kJ_mol(lambdas, pilot_lambdas, mean_dU_dlambda) -> np.ndarray:
    """Per-edge |Delta F| (kJ/mol) of a lambda path, from the pilot TI gradients."""
    u_pilot, cum = _free_energy_arclength(pilot_lambdas, mean_dU_dlambda)
    f_nodes = np.interp(
        1.0 - np.asarray(lambdas, dtype=float).ravel(), u_pilot, cum
    )
    return np.abs(np.diff(f_nodes))


def validate_vanishing_lambda_path_invariants(
    lambdas_vdw,
    *,
    n_states: int = VANISHING_FINAL_STATE_COUNT,
    geometric_floor_weight: float = VANISHING_GEOMETRIC_FLOOR_WEIGHT,
) -> None:
    """Structural invariants every production vanishing path must satisfy.

    ``n_states`` is keyword-only on purpose.  The v20 predecessor
    (validate_human_vanishing_anchors_preserved) took ``requested_base_n_states``
    -- the *requested* probe count, 17 -- as its second positional argument,
    while this one takes the *expected produced path length*, 23.  Call sites
    that kept passing the old positional value would otherwise silently
    validate against the wrong number instead of failing loudly.

    v20 and earlier validated *identity* against a hard-coded quadratic+manual
    anchor set, which is meaningless once the metric places the nodes.  What
    actually has to hold is: the right number of states, a strictly decreasing
    1 -> 0 path with exact endpoints, and no lambda gap wider than the blended
    placement's geometric floor (the invariant that prevents v18's stranded
    decoupled tail).  The quadratic fallback satisfies this too.
    """
    lambdas = np.asarray(lambdas_vdw, dtype=float).ravel()
    if lambdas.size != int(n_states):
        raise ValueError(
            f"vanishing 路径必须恰好 {int(n_states)} 态，实际 {lambdas.size}"
        )
    if not np.all(np.isfinite(lambdas)):
        raise ValueError("vanishing 路径含非有限 lambda")
    if not np.all(np.diff(lambdas) < 0.0):
        raise ValueError("vanishing 路径必须严格递减")
    if not (np.isclose(lambdas[0], 1.0) and np.isclose(lambdas[-1], 0.0)):
        raise ValueError("vanishing 路径端点必须恰好是 lambda=1 和 lambda=0")
    gap_bound = vanishing_max_lambda_gap_bound(int(n_states), geometric_floor_weight)
    realized_gap = float(np.max(np.abs(np.diff(lambdas))))
    if realized_gap > gap_bound * (1.0 + 1.0e-6):
        raise ValueError(
            f"vanishing 路径存在超过几何覆盖上限的 lambda 断层："
            f"max|Δλ|={realized_gap:.6f} > {gap_bound:.6f}（"
            "v18 曾因纯等热力学长度布点把解耦端拉断，这条门就是防它复发）"
        )


def _greedy_vanishing_window_ranges(
    n_states: int,
    min_states_per_window: int,
    max_states_per_window: int,
) -> List[Tuple[int, int]]:
    """Group ``n_states`` states into windows of
    ``min_states_per_window``..``max_states_per_window`` states each, EVERY
    window within bounds (not just avoiding a too-short trailing one).

    Picks the number of windows ``W`` first (the smallest ``W`` for which an
    even split can keep every window's size within bounds), then distributes
    states across those ``W`` windows. This two-pass approach is deliberate: a
    pure left-to-right greedy fill (take max_states_per_window every time) can
    strand a remainder smaller than min_states_per_window that no single merge
    fixes -- verified this the hard way, see the fix note. One boundary state is
    shared between adjacent windows, same convention as the hand-tuned
    23-state table (e.g. ``(0,5),(4,8)`` share state 4).

    🔑 [2026-08-28] WINDOW 0 IS THE SMALLEST WINDOW, by explicit user request.
    The previous distribution handed the leftover states to the FRONT
    (``[base+1]*extra + [base]*(W-extra)``), so window 0 was tied-largest -- on
    the real 4W53 12-state path that made window 0 carry +33.5 kJ/mol, 54.7% of
    the whole path's total variation, in the same 5 states the flat middle got.
    The cause is that lambda placement follows the Fisher metric
    beta**2 Var[dU/dlambda], which is *smallest* at lambda=1 (24.2 there vs 1590
    at lambda~0.34) exactly where the *mean* gradient is largest (-145.8 kJ/mol).
    Equal-thermodynamic-length spacing is still the right overlap criterion, so
    this does not move a single lambda node -- it only regroups them, giving
    window 0 ``min_states_per_window`` and spreading the rest evenly, sizes
    non-decreasing. Every window still lands inside [min, max].

    This does NOT touch the 23-state path (that returns the hand-tuned table
    before ever reaching this function) and does not change the number of
    windows ``W`` for any input -- only how many states each one gets.
    """
    n_states = int(n_states)
    min_states_per_window = int(min_states_per_window)
    max_states_per_window = int(max_states_per_window)
    if min_states_per_window < 2:
        raise ValueError(f"min_states_per_window 至少为 2：收到 {min_states_per_window}")
    if max_states_per_window < min_states_per_window:
        raise ValueError(
            f"max_states_per_window ({max_states_per_window}) 不能小于 "
            f"min_states_per_window ({min_states_per_window})"
        )
    total_intervals = n_states - 1
    if total_intervals < 1:
        raise ValueError(f"n_states 至少为 2：收到 {n_states}")

    # sum(sizes) = n_states + W - 1 (W-1 shared boundary states double-counted).
    # A legal window count must satisfy
    # W*(min-1) <= total_intervals <= W*(max-1).  Determine feasibility before
    # constructing anything; the previous best-effort decrement could collapse
    # an infeasible two-window request to one oversized window (7 states with
    # min=max=6).
    max_interval_span = max_states_per_window - 1
    min_interval_span = min_states_per_window - 1
    min_windows = -(-total_intervals // max_interval_span)  # ceil division
    max_windows = total_intervals // min_interval_span
    if min_windows > max_windows:
        raise ValueError(
            "不存在满足 vanishing 分窗约束的窗口数："
            f"n_states={n_states}, min_states_per_window={min_states_per_window}, "
            f"max_states_per_window={max_states_per_window}"
        )
    n_windows = min_windows

    # Distribute interval spans evenly, with smaller windows first.  Adding one
    # shared boundary node converts each span to its window size.
    base_span, extra = divmod(total_intervals, n_windows)
    spans = [base_span] * (n_windows - extra) + [base_span + 1] * extra
    sizes = [span + 1 for span in spans]

    ranges: List[Tuple[int, int]] = []
    start = 0
    for size in sizes:
        ranges.append((start, start + size))
        start += size - 1

    # Final construction audit: bounds, complete coverage, and exactly one
    # shared boundary state between adjacent windows are all part of the public
    # contract, not assumptions of the allocator above.
    if any(
        not (min_states_per_window <= end - begin <= max_states_per_window)
        for begin, end in ranges
    ):
        raise RuntimeError(f"内部错误：vanishing 分窗尺寸越界：{ranges}")
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != n_states:
        raise RuntimeError(f"内部错误：vanishing 分窗未覆盖完整端点：{ranges}")
    for left, right in zip(ranges, ranges[1:]):
        if left[1] - 1 != right[0]:
            raise RuntimeError(f"内部错误：相邻 vanishing 分窗必须只共享一个边界：{ranges}")
    for left_index, left in enumerate(ranges):
        for right in ranges[left_index + 2:]:
            if right[0] < left[1]:
                raise RuntimeError(f"内部错误：非相邻 vanishing 分窗发生重叠：{ranges}")
    covered = {state for begin, end in ranges for state in range(begin, end)}
    if covered != set(range(n_states)):
        raise RuntimeError(f"内部错误：vanishing 分窗覆盖不完整：{ranges}")
    return ranges


def vanishing_subdomain_ranges_from_lambdas(
    lambdas_vdw,
    target_intervals_per_ensemble: int = VANISHING_TARGET_INTERVALS_PER_ENSEMBLE,
    min_intervals_per_ensemble: int = VANISHING_MIN_INTERVALS_PER_ENSEMBLE,
    max_states_per_ensemble: int = VANISHING_MAX_STATES_PER_IBS_ENSEMBLE,
    first_ensemble_target_intervals: Optional[int] = None,
    # 🔑 [2026-08-27] Only consumed on the != 23 (greedy) path below. The four
    # params above are the *frozen* 23-state contract (validated to equal
    # their defaults, unused for computation once lambdas.size==23 since that
    # path just returns the hand-tuned table). These two are in STATES, not
    # intervals -- ask was literally "每个窗口最少4最多6[态]".
    min_states_per_window: int = 4,
    max_states_per_window: int = 6,
) -> List[Tuple[int, int]]:
    """Partition an adaptive lambda path into few-state IBS subintervals.

    Lambda locations already encode the thermodynamic metric.  This routine
    only groups consecutive thermodynamic intervals; it never cuts at a fixed
    physical lambda.  Every lambda edge is assigned exactly once.  Adjacent
    ensembles do reuse one boundary *node* as the common free-energy reference;
    this is not zero shared states and must not be logged as "no overlap".  What
    is forbidden is the legacy overlap=2 sliding layout, which reused two nodes
    and duplicated the lambda edge between them.

    ``first_ensemble_target_intervals``: [THERMODYNAMIC_PATH_PROTOCOL_VERSION=14]
    optional override carving the FIRST ensemble (the fully-coupled vdW endpoint,
    where a real GPU run showed occupation stuck at state 0 and
    `min_absolute_ess~1.0` even after uniformly shrinking every ensemble to 3
    intervals) down to a specific interval count, independent of
    ``target_intervals_per_ensemble``. The remaining intervals are grouped with
    the existing uniform-target logic exactly as before -- this only changes
    the FIRST ensemble's size, not the rest of the path (which is not showing
    this failure and doesn't need paying the cost of smaller ensembles
    everywhere, per the user's explicit rejection of a uniform global
    resubdivision). Still one shared boundary node between ensemble 0 and 1, no
    `overlap=2` reintroduced.
    """
    lambdas = np.asarray(lambdas_vdw, dtype=float).ravel()
    if lambdas.size < 2 or not np.all(np.diff(lambdas) < 0.0):
        raise ValueError("vanishing lambda 路径必须至少 2 态且严格递减")
    if lambdas.size == VANISHING_FINAL_STATE_COUNT:
        # 23 态：仍然走原来手工调出来的固定 6 窗表——含 window0 ESS 塌缩修复，
        # 逐字节不变。
        if first_ensemble_target_intervals not in (
            None,
            VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
        ):
            raise ValueError("第一窗口固定为闭区间 [0,4]，即 4 条 lambda 边")
        ranges = [tuple(r) for r in VANISHING_FIXED_WINDOW_RANGES]
    else:
        # 🔑 [2026-08-27] 别的态数：每个窗口 min_states_per_window..
        # max_states_per_window 态，贪心从头填满，尾窗不够 min 就并进前一个窗口
        # （见 _greedy_vanishing_window_ranges）。不是把上面 23 态那张表反推
        # 出来的——两者给出的分组本来就不一样（上表是手工调过的，含 window0
        # ESS 塌缩修复的特殊收窄，这条路径目前没有）。只对 n_states=23 之外的
        # 态数生效，23 态路径完全不受影响。这条新路径的 window0 行为还没在
        # 真机上验证过。
        ranges = _greedy_vanishing_window_ranges(
            int(lambdas.size),
            min_states_per_window=int(min_states_per_window),
            max_states_per_window=int(max_states_per_window),
        )
    validate_single_shared_boundary_ranges(ranges, int(lambdas.size))
    return ranges


def redistribute_lambda_by_thermodynamic_length(
    pilot_lambdas: np.ndarray,
    metric_g: np.ndarray,
    n_states: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place states uniformly in cumulative thermodynamic length.

    ``metric_g`` is the dimensionless scalar Fisher metric
    beta**2 Var[dU/dlambda] evaluated at ``pilot_lambdas``.  The path may run in
    either lambda direction; only monotonicity is required.

    Returns ``(new_lambdas, pilot_cumulative_length, new_edge_lengths)``.
    """
    pilot_lambdas = np.asarray(pilot_lambdas, dtype=float).ravel()
    metric_g = np.asarray(metric_g, dtype=float).ravel()
    if pilot_lambdas.size != metric_g.size or pilot_lambdas.size < 2:
        raise ValueError("pilot_lambdas/metric_g 必须等长且至少包含两个点")
    if n_states < 2:
        raise ValueError("n_states 必须至少为 2")
    if not np.all(np.isfinite(metric_g)) or np.any(metric_g < 0.0):
        raise ValueError("热力学度量 g(lambda) 含 NaN/Inf 或负值")
    delta_lambda = np.diff(pilot_lambdas)
    if not (np.all(delta_lambda > 0.0) or np.all(delta_lambda < 0.0)):
        raise ValueError("pilot lambda 必须严格单调")

    # Trapezoidal quadrature of integral sqrt(g(lambda)) |dlambda|.  The tiny
    # floor only regularizes an exactly flat numerical segment; unlike the old
    # log1p/clipping path it does not compress real high-metric regions.
    sqrt_g = np.sqrt(np.maximum(metric_g, 1.0e-12))
    pilot_edges = 0.5 * (sqrt_g[:-1] + sqrt_g[1:]) * np.abs(delta_lambda)
    cumulative = np.concatenate(([0.0], np.cumsum(pilot_edges)))
    total_length = float(cumulative[-1])
    if not np.isfinite(total_length) or total_length <= 1.0e-8:
        raise RuntimeError("pilot 得到的总热力学长度为零或非有限，拒绝伪装成有效自适应路径")

    targets = np.linspace(0.0, total_length, int(n_states))
    new_lambdas = np.interp(targets, cumulative, pilot_lambdas)
    new_lambdas[0] = pilot_lambdas[0]
    new_lambdas[-1] = pilot_lambdas[-1]
    return new_lambdas, cumulative, np.diff(targets)


def recompute_vanishing_path_from_cached_pilot(
    path_diagnostics: Dict,
    *,
    n_states: int,
    final_state_count: int,
    min_states_per_window: int,
    max_states_per_window: int,
    free_energy_densify_points: int,
) -> Dict:
    """从**已落盘的 pilot 测量**离线重算 λ 路径与窗口布局。**不跑任何 MD。**

    ## 为什么需要它

    preopt 缓存原来是一整块：`stage2_final_n_states` / `densify` /
    `window_min|max_states` 这类**只影响布点与分窗**的参数一改，整份缓存失配，
    连那份要跑几十分钟 GPU 的 pilot 测量一起作废重跑。

    但 pilot→λ 这一步是**纯函数**（`redistribute_vanishing_lambda_subdomains`），
    它需要的 `pilot_lambdas` / `metric_g` / 每点的 `mean_dU_dlambda_kJ_mol`
    全都已经在缓存的 `path_diagnostics` 里。所以这类改动应当从旧 pilot
    **离线重算**，而不是重烧 GPU。

    这就是缓存两层拆分里的第 2 层（派生路径）。第 1 层（原始 pilot 测量：
    Hamiltonian、采样步数、差分步长、遍历顺序、加密探针协议）变了才必须重跑。

    Parameters
    ----------
    path_diagnostics
        缓存里的 `path_diagnostics` 字典，必须含 `pilot_lambdas`、`metric_g`、
        `pilot_points`。缺任何一项直接抛——**不猜**，猜出来的 λ 会静默改变生产态。

    Returns
    -------
    dict
        `{"lambdas_vdw", "window_ranges", "subdomain_allocation",
          "cumulative_length", "optimized_edge_lengths"}`
    """
    for key in ("pilot_lambdas", "metric_g", "pilot_points"):
        if not path_diagnostics.get(key):
            raise ValueError(
                f"缓存的 path_diagnostics 缺少 {key!r}，无法离线重算派生路径。"
                "这份缓存太旧（早于 pilot 测量落盘），只能重跑 pilot。"
            )
    pilot_lambdas = [float(x) for x in path_diagnostics["pilot_lambdas"]]
    metric_g = np.asarray(path_diagnostics["metric_g"], dtype=float)
    if len(pilot_lambdas) != metric_g.size:
        raise ValueError(
            f"缓存的 pilot_lambdas ({len(pilot_lambdas)}) 与 metric_g "
            f"({metric_g.size}) 长度不一致，拒绝据此重算。"
        )
    (
        optimized_lambdas,
        cumulative_length,
        optimized_edge_lengths,
        window_ranges,
        subdomain_allocation,
    ) = redistribute_vanishing_lambda_subdomains(
        pilot_lambdas,
        metric_g,
        int(n_states),
        first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
        final_state_count=int(final_state_count),
        min_states_per_window=int(min_states_per_window),
        max_states_per_window=int(max_states_per_window),
        free_energy_densify_points=int(free_energy_densify_points),
        pilot_mean_dU_dlambda=_pilot_mean_gradients_or_none(
            path_diagnostics["pilot_points"]
        ),
    )
    # 与 optimize_stage2_vanishing 里的后处理逐字一致：端点必须精确是 1 和 0。
    optimized_lambdas = np.asarray(optimized_lambdas, dtype=float).ravel()
    optimized_lambdas = np.clip(optimized_lambdas, 0.0, 1.0)
    optimized_lambdas[0], optimized_lambdas[-1] = 1.0, 0.0
    return {
        "lambdas_vdw": optimized_lambdas,
        "window_ranges": window_ranges,
        "subdomain_allocation": subdomain_allocation,
        "cumulative_length": cumulative_length,
        "optimized_edge_lengths": optimized_edge_lengths,
    }


def redistribute_vanishing_lambda_subdomains(
    pilot_lambdas: np.ndarray,
    metric_g: np.ndarray,
    n_states: int,
    target_intervals_per_ensemble: int = VANISHING_TARGET_INTERVALS_PER_ENSEMBLE,
    min_intervals_per_ensemble: int = VANISHING_MIN_INTERVALS_PER_ENSEMBLE,
    max_states_per_ensemble: int = VANISHING_MAX_STATES_PER_IBS_ENSEMBLE,
    first_ensemble_target_intervals: Optional[int] = None,
    final_state_count: int = VANISHING_FINAL_STATE_COUNT,
    # 只在 final_state_count != VANISHING_FINAL_STATE_COUNT 时生效，见
    # vanishing_subdomain_ranges_from_lambdas。默认不传等于什么都不变——
    # final_state_count 留默认(23) 就还是走老的固定表，这两个参数根本不会
    # 被用到。
    min_states_per_window: int = 4,
    max_states_per_window: int = 6,
    # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=22] 自由能定向加密。0 = 关闭，布点与
    # v21 逐字节相同。k>0 时基础布点用 (final_state_count - k) 态，再按实测
    # <dU/dlambda> 贪心插 k 个点；**总态数不变**，采样成本不变。需要
    # pilot_mean_dU_dlambda（pilot_points[i]["mean_dU_dlambda_kJ_mol"]）。
    free_energy_densify_points: int = 0,
    pilot_mean_dU_dlambda: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]], Dict]:
    """Place the production vanishing lambdas from the measured Fisher metric.

    🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=21] The metric now CONTROLS
    placement (blended with a geometric floor); v19/v20 computed the
    equal-thermodynamic-length solution here and then threw it away in favour
    of a fixed quadratic schedule + 4 hand-picked + 2 bridge nodes.  See the
    version history at the top of this module for why both extremes failed.

    🔑 [2026-08-27] ``n_states`` used to have to equal the module constant
    ``VANISHING_PROBE_BASE_STATE_COUNT`` (17) or this raised — a check against
    a fixed global that had nothing to do with what was actually probed.
    Replaced with a minimal sanity check (``n_states >= 2``) instead of a
    magic-number lock. Note ``n_states`` is the caller's *original* probe
    count and is deliberately NOT compared against ``len(pilot_lambdas)``:
    ``_refine_pilot_grid_in_steep_segments`` adds extra points inside steep
    segments before this is called (that's the real window0 ESS-collapse
    fix, protocol version 15), so ``pilot_lambdas`` legitimately grows past
    ``n_states`` in the normal/expected path. This is what let
    ``human_vanishing_initial_lambdas`` widen to any probe density.

    ``final_state_count`` is new and *not* the same knob as ``n_states``: it is
    the number of production windows the metric gets placed onto, i.e. how
    many actual λ states you end up with. Default (23) still goes through
    ``VANISHING_FIXED_WINDOW_RANGES``, the hand-tuned 6-window partition
    (first window pinned to the closed interval [0,4]) built specifically
    because a real GPU run showed window 0 collapse to min_absolute_ess~1.0 —
    byte-for-byte unchanged. Anything else goes through
    ``_greedy_vanishing_window_ranges`` instead: windows of
    ``min_states_per_window``..``max_states_per_window`` states each, greedily
    filled front-to-back. This is a genuinely different (simpler, no
    window0-specific narrowing) algorithm from the 23-state table, added
    2026-08-27, not yet checked against a real GPU run — the thing to look at
    first is whether window 0 still shows the same occupancy collapse.
    """
    if int(target_intervals_per_ensemble) != VANISHING_TARGET_INTERVALS_PER_ENSEMBLE:
        raise ValueError("人工 vanishing 窗口契约禁止覆盖 target_intervals_per_ensemble")
    if int(min_intervals_per_ensemble) != VANISHING_MIN_INTERVALS_PER_ENSEMBLE:
        raise ValueError("人工 vanishing 窗口契约禁止覆盖 min_intervals_per_ensemble")
    if int(max_states_per_ensemble) != VANISHING_MAX_STATES_PER_IBS_ENSEMBLE:
        raise ValueError("人工 vanishing 窗口契约禁止覆盖 max_states_per_ensemble")
    if first_ensemble_target_intervals not in (
        None,
        VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
    ):
        raise ValueError("人工 vanishing 窗口契约禁止覆盖第一窗口区间数")

    pilot_lambdas = np.asarray(pilot_lambdas, dtype=float)
    # 🔑 [2026-08-27] `n_states` 是调用方原始探针网格点数，不是 `pilot_lambdas`
    # 当前长度——`_refine_pilot_grid_in_steep_segments` 会在陡峭区间插点，实测
    # 真机跑法里 `pilot_lambdas` 之后通常比 `n_states` 更长（见该函数文档串，
    # 就是靠这个才修好 window0 ESS 塌缩）。这里不拿它俩比对，只做基本合法性
    # 检查；下面实际用来布点的是 `pilot_lambdas`/`metric_g` 本身的长度。
    if int(n_states) < 2:
        raise ValueError(f"n_states 必须至少为 2：收到 {int(n_states)}")
    if pilot_lambdas.size < 2:
        raise ValueError(f"pilot_lambdas 必须至少有 2 个点：收到 {pilot_lambdas.size}")
    final_state_count = int(final_state_count)
    if final_state_count < 2:
        raise ValueError(f"final_state_count 必须至少为 2：收到 {final_state_count}")
    # 🔑 [2026-08-27] 之前这里对任何非 23 的 final_state_count 都硬拒绝。现在
    # 真正生效：!= 23 时 vanishing_subdomain_ranges_from_lambdas 走
    # _greedy_vanishing_window_ranges（见该函数），23 时逐字节走原来那张手工
    # 表，两条路径互不影响。新路径的 window0 行为还没在真机上跑过。
    n_densify = int(free_energy_densify_points)
    if n_densify < 0:
        raise ValueError(f"free_energy_densify_points 不能为负：{n_densify}")
    base_state_count = final_state_count - n_densify
    if n_densify and base_state_count < 2:
        raise ValueError(
            f"free_energy_densify_points={n_densify} too large for "
            f"final_state_count={final_state_count}：基础布点只剩 {base_state_count} 态"
        )
    optimized_lambdas, cumulative, optimized_edge_lengths = (
        blended_metric_vanishing_lambdas(
            pilot_lambdas,
            np.asarray(metric_g, dtype=float),
            base_state_count,
            VANISHING_GEOMETRIC_FLOOR_WEIGHT,
        )
    )
    if n_densify:
        if pilot_mean_dU_dlambda is None:
            raise ValueError(
                "free_energy_densify_points > 0 需要 pilot_mean_dU_dlambda "
                "（pilot_points 里的 mean_dU_dlambda_kJ_mol）；拒绝在没有实测梯度的"
                "情况下猜测加密位置"
            )
        optimized_lambdas = densify_lambdas_by_free_energy(
            optimized_lambdas,
            pilot_lambdas,
            np.asarray(pilot_mean_dU_dlambda, dtype=float),
            n_densify,
        )
        if len(optimized_lambdas) != final_state_count:
            raise RuntimeError(
                f"自由能加密后态数 {len(optimized_lambdas)} != "
                f"final_state_count {final_state_count}"
            )
        # 边热力学长度必须按加密后的实际网格重算——基础布点返回的那份是
        # (final_state_count - k) 态的，直接沿用会让所有 delta 诊断全错。
        pilot_desc = np.sort(np.asarray(pilot_lambdas, dtype=float).ravel())[::-1]
        placed_cum = np.interp(
            optimized_lambdas[::-1], pilot_desc[::-1], cumulative[::-1]
        )[::-1]
        optimized_edge_lengths = np.abs(np.diff(placed_cum))
    validate_vanishing_lambda_path_invariants(optimized_lambdas, n_states=final_state_count)
    window_ranges = vanishing_subdomain_ranges_from_lambdas(
        optimized_lambdas,
        target_intervals_per_ensemble=target_intervals_per_ensemble,
        min_intervals_per_ensemble=min_intervals_per_ensemble,
        max_states_per_ensemble=max_states_per_ensemble,
        first_ensemble_target_intervals=first_ensemble_target_intervals,
        min_states_per_window=min_states_per_window,
        max_states_per_window=max_states_per_window,
    )
    validate_single_shared_boundary_ranges(window_ranges, len(optimized_lambdas))
    interval_counts = [end - start - 1 for start, end in window_ranges]
    edge_dF = (
        edge_free_energy_kJ_mol(
            optimized_lambdas, pilot_lambdas, np.asarray(pilot_mean_dU_dlambda, dtype=float)
        )
        if pilot_mean_dU_dlambda is not None
        else None
    )
    allocation = {
        "base_lambda_placement": "fisher_metric_blended_with_geometric_floor_v21",
        "free_energy_densify_points": n_densify,
        "base_state_count_before_densify": int(base_state_count),
        "geometric_floor_weight": float(VANISHING_GEOMETRIC_FLOOR_WEIGHT),
        "max_lambda_gap_bound": float(
            vanishing_max_lambda_gap_bound(final_state_count)
        ),
        "realized_max_lambda_gap": float(
            np.max(np.abs(np.diff(optimized_lambdas)))
        ),
        "realized_max_edge_thermodynamic_length": float(
            np.max(optimized_edge_lengths)
        ) if len(optimized_edge_lengths) else 0.0,
        "realized_min_edge_thermodynamic_length": float(
            np.min(optimized_edge_lengths)
        ) if len(optimized_edge_lengths) else 0.0,
        "actual_state_count": int(len(optimized_lambdas)),
        "total_window_state_slots": int(
            sum(end - start for start, end in window_ranges)
        ),
        "subdomain_interval_counts": interval_counts,
        "subdomain_state_counts": [count + 1 for count in interval_counts],
        "subdomain_lambda_bounds": [
            [float(optimized_lambdas[start]), float(optimized_lambdas[end - 1])]
            for start, end in window_ranges
        ],
        "actual_shared_state_indices": [
            int(window_ranges[i][0]) for i in range(1, len(window_ranges))
        ],
        "actual_state_index_sets": [
            list(range(start, end)) for start, end in window_ranges
        ],
    }
    # [v22] 自由能诊断：探针一直在测 <dU/dlambda>，之前从没按边/按窗积出来过。
    # 这是判断 window 的 IBS 偏置要爬多高的量，与 delta（重叠判据）是两个轴。
    if edge_dF is not None:
        allocation["edge_free_energy_kJ_mol"] = [float(x) for x in edge_dF]
        allocation["max_edge_free_energy_kJ_mol"] = float(np.max(edge_dF)) if edge_dF.size else 0.0
        allocation["total_free_energy_variation_kJ_mol"] = float(np.sum(edge_dF))
        allocation["subdomain_free_energy_kJ_mol"] = [
            float(np.sum(edge_dF[start:end - 1])) for start, end in window_ranges
        ]
        allocation["subdomain_max_edge_free_energy_kJ_mol"] = [
            float(np.max(edge_dF[start:end - 1])) if end - 1 > start else 0.0
            for start, end in window_ranges
        ]
    return (
        optimized_lambdas,
        cumulative,
        optimized_edge_lengths,
        window_ranges,
        allocation,
    )


def partition_windows_by_thermodynamic_length(
    edge_lengths: np.ndarray,
    max_window_length: float,
    overlap_length: float,
    min_states_per_window: int = 3,
    max_states_per_window: Optional[int] = 6,
) -> List[Tuple[int, int]]:
    """Partition a path by cumulative thermodynamic distance, not state count.

    Distance alone decides *where* to cut, but it cannot replace a hard cap on
    IBS window size: many short, evenly-spaced pilot edges (e.g. 17 edges at
    ~0.85 each, max_window_length=6.0) let the distance-only growth loop pack
    7+ edges (8+ states) into one window before it ever exceeds the distance
    budget -- an IBS bias handling that many states at once is exactly the
    "8 states, one bias" case this cap exists to prevent. ``max_states_per_window``
    (default 6, matching the previous fixed pts_per_window convention) is
    therefore enforced as a second, independent stopping condition in the same
    growth loop, not a post-hoc truncation that would silently disagree with
    the distance/overlap bookkeeping below.
    """
    edge_lengths = np.asarray(edge_lengths, dtype=float).ravel()
    if not np.all(np.isfinite(edge_lengths)) or np.any(edge_lengths < 0.0):
        raise ValueError("edge_lengths 含 NaN/Inf 或负值")
    if max_window_length <= 0.0:
        raise ValueError("max_window_length 必须 > 0")
    if not 0.0 <= overlap_length < max_window_length:
        raise ValueError("overlap_length 必须位于 [0, max_window_length) 内")

    n_states = edge_lengths.size + 1
    if n_states <= 2:
        return [(0, n_states)]
    min_states = max(2, int(min_states_per_window))
    if max_states_per_window is not None and int(max_states_per_window) < min_states:
        raise ValueError(
            f"max_states_per_window ({max_states_per_window}) 不能小于 "
            f"min_states_per_window ({min_states})"
        )
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < n_states - 1:
        end = start + 1
        state_cap_end = (
            start + int(max_states_per_window) - 1
            if max_states_per_window is not None
            else n_states - 1
        )
        while (
            end + 1 < n_states
            and end + 1 <= state_cap_end
            and cumulative[end + 1] - cumulative[start] <= max_window_length
        ):
            end += 1
        # A two-state IBS window is statistically fragile.  Keep at least the
        # requested number of states when possible, even if one exceptional
        # pilot edge alone exceeds the distance budget; diagnostics will expose
        # that overspend instead of silently dropping connectivity.  This can
        # never push end past state_cap_end since max_states_per_window is
        # asserted >= min_states_per_window above.
        end = min(n_states - 1, max(end, start + min_states - 1))
        windows.append((start, end + 1))
        if end >= n_states - 1:
            break

        next_start = end
        while (
            next_start > start
            and cumulative[end] - cumulative[next_start] < overlap_length
        ):
            next_start -= 1
        if next_start <= start:
            next_start = start + 1
        start = next_start

    covered = sorted({i for start, end in windows for i in range(start, end)})
    if covered != list(range(n_states)):
        raise RuntimeError(f"热力学窗口未完整覆盖路径: {covered}")
    return windows


def split_window_from_ibs_lse_failure(
    window_ranges: List[Tuple[int, int]],
    warmup_diagnostics: Dict,
    n_states: int,
) -> Tuple[List[Tuple[int, int]], Dict]:
    """Split an LSE-unstable IBS ensemble without changing the lambda grid.

    Design refinement is allowed to use two-state IBS ensembles.  Therefore a
    K=3 parent can still be split into two K=2 children sharing one existing
    state.  K=2 is irreducible and must be handled by thermodynamic-midpoint
    insertion, never by fixed-H overlap.
    """
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    failed = tuple(int(x) for x in warmup_diagnostics["global_state_range"])
    if failed not in ranges:
        raise RuntimeError(f"LSE 失败窗口 {failed} 不在当前窗口列表 {ranges} 中")
    start, end = failed
    if end - start < 3:
        raise RuntimeError("两态 IBS 窗口不可再拆，必须插入热力学长度中点后复验")

    middle = (start + end - 1) // 2
    children = [(start, middle + 1), (middle, end)]
    if min(e - s for s, e in children) < 2:
        raise RuntimeError(f"LSE 拆窗会产生少于两个态的窗口: {children}")

    expanded = []
    for current in ranges:
        if current == failed:
            expanded.extend(children)
        else:
            expanded.append(current)
    new_ranges = canonicalize_window_ranges(expanded, int(n_states))
    return new_ranges, {
        "source": "ibs_lse_design_window_split",
        "failed_global_state_range": [start, end],
        "child_ranges": [list(r) for r in children],
        "shared_global_state": int(middle),
        "inserted_lambda": None,
        "lse_balance": warmup_diagnostics.get("lse_balance"),
    }


def _pilot_arclength_of(lambda_value, pilot_lam_desc, pilot_s_asc):
    """某个 λ 在 pilot 实测累计热力学坐标上的位置。pilot λ 递减、s 递增。"""
    return float(np.interp(lambda_value, pilot_lam_desc[::-1], pilot_s_asc[::-1]))


def metric_integral_cumulative(
    lambdas: Sequence[float],
    pilot_lambdas: Sequence[float],
    metric_g: Sequence[float],
) -> np.ndarray:
    """把 pilot 的 ∫g dλ 累积到给定 λ 表上。

    注意与热力学长度 ``∫√g dλ`` 的区别：等**弧长**布点均衡的是相邻态之间的重叠，
    而 IBS 是一条轨迹重加权到窗口内**全部** K 个态，难度更接近窗口内的总方差
    ``∫g dλ``。⚠️ 这是一个**有实验动机的候选指标，不是已证明的 IBS 难度**：柯西–
    施瓦茨 ``∫g dλ >= L²/Δλ`` 只说明两者不等价，并没有证明 ∫g 预测 IBS 收敛；而且
    热力学长度有 Fisher 度规基础、∫g dλ 则依赖 λ 的参数化方式（变量变换下会变）。
    当前证据是"换上去之后布局更好、跑得通"，据此继续实验，不据此宣称机制。
    经验上：**同样弧长的窗口，落在度规
    尖峰上的那个 Δλ 很小、∫g 却大得多** —— 实测 4W53 cyclod 21 态五窗弧长大致
    相等（2.28~3.29），∫g 却是 19/37/67/97/54，差 5 倍，而失败的正是 ∫g 最大那个。
    """
    pl = np.asarray(pilot_lambdas, dtype=float).ravel()
    g = np.asarray(metric_g, dtype=float).ravel()
    if pl.size != g.size or pl.size < 2:
        raise ValueError("pilot_lambdas 与 metric_g 必须等长且至少两个点")
    if not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("metric_g 含非有限值或负值")
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (g[:-1] + g[1:]) * np.abs(np.diff(pl)))])
    return np.interp(np.asarray(lambdas, dtype=float), pl[::-1], cum[::-1])


def partition_windows_by_metric_integral(
    lambdas: Sequence[float],
    pilot_lambdas: Sequence[float],
    metric_g: Sequence[float],
    *,
    min_states_per_window: int = 4,
    max_states_per_window: int = 8,
    n_windows: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """按 ``∫g dλ`` 均衡划分 IBS 窗口（**尚未接入生产,独立函数**）。

    相邻窗口共享且只共享一个边界态。先用 DP 最小化"最大窗 ∫g"，再在所有取得该
    最小值的布局里最小化 ∫g 的平方和，得到确定性的、尽量均匀的结果。

    为什么要放开 ``max_states_per_window``：卡住均衡的从来不是 ``min``（那是一条
    **工程下限**，动机是窗口两端各有一个与邻窗共享的边界态、内部只剩 K-2 个自由
    态；⚠️ 共享边界态并不意味着这两个态的 f_k 不能调整，所以这**不是**已证明的数学
    必要条件，别当定理引用），而是 ``max``。它逼着**便宜的地方也只能用小窗**，白白多切几刀，却在贵的地方
    切不动。实测（4W53 cyclod，21 态）：

        max=5  强制 [5,5,5,5,5]  峰值 ∫g=97.0  不均衡 5.08
        max=8  得到 [8,5,4,4,4]  峰值 ∫g=74.2  不均衡 2.36
        max=12 且只要 4 个窗 [10,5,4,5] 峰值 75.3 不均衡 1.40

    ``n_windows=None`` 时在所有可行窗口数里自动选：先比峰值 ∫g，再比窗口数（少
    的省 GPU），最后比平方和。

    ⚠️ 峰值有地板：它等于尖峰处一个**最小 4 态窗**的 ∫g（上例 74.2）。想再低只能
    在尖峰那段**加 λ 态**，分窗解决不了。
    """
    lam = np.asarray(lambdas, dtype=float).ravel()
    n = lam.size
    lo, hi = int(min_states_per_window), int(max_states_per_window)
    if lo < 2 or hi < lo:
        raise ValueError(f"min/max_states_per_window 非法：{lo}/{hi}")
    if n < lo:
        raise ValueError(f"λ 表只有 {n} 个态，不足 min_states_per_window={lo}")
    gcum = metric_integral_cumulative(lam, pilot_lambdas, metric_g)
    cost = lambda a, b: abs(float(gcum[b - 1] - gcum[a]))

    def _solve(w_target: int):
        """(峰值, 平方和, ranges)；不可行返回 None。"""
        INF = float("inf")
        best = [[INF] * (w_target + 1) for _ in range(n)]
        back = [[None] * (w_target + 1) for _ in range(n)]
        best[0][0] = 0.0
        for i in range(n):
            for w in range(w_target):
                if best[i][w] == INF:
                    continue
                for size in range(lo, hi + 1):
                    j = i + size - 1
                    if j > n - 1:
                        break
                    cand = max(best[i][w], cost(i, j + 1))
                    if cand < best[j][w + 1]:
                        best[j][w + 1] = cand
                        back[j][w + 1] = (i, w)
        peak = best[n - 1][w_target]
        if peak == INF:
            return None
        # 第二遍：在"每个窗都不超过 peak"的约束下最小化平方和，结果确定且更均匀。
        tol = peak * (1.0 + 1e-12) + 1e-12
        ss = [[INF] * (w_target + 1) for _ in range(n)]
        bk2 = [[None] * (w_target + 1) for _ in range(n)]
        ss[0][0] = 0.0
        for i in range(n):
            for w in range(w_target):
                if ss[i][w] == INF:
                    continue
                for size in range(lo, hi + 1):
                    j = i + size - 1
                    if j > n - 1:
                        break
                    c = cost(i, j + 1)
                    if c > tol:
                        continue
                    cand = ss[i][w] + c * c
                    if cand < ss[j][w + 1]:
                        ss[j][w + 1] = cand
                        bk2[j][w + 1] = (i, w)
        if ss[n - 1][w_target] == INF:
            return None
        ranges, i, w = [], n - 1, w_target
        while w > 0:
            pi, pw = bk2[i][w]
            ranges.append((pi, i + 1))
            i, w = pi, pw
        return peak, ss[n - 1][w_target], list(reversed(ranges))

    if n_windows is not None:
        got = _solve(int(n_windows))
        if got is None:
            raise RuntimeError(
                f"{n} 个态在 [{lo},{hi}] 约束下切不出 {n_windows} 个窗口"
            )
        candidates = [(got[0], int(n_windows), got[1], got[2])]
    else:
        candidates = []
        for w in range(1, n):
            got = _solve(w)
            if got is not None:
                candidates.append((got[0], w, got[1], got[2]))
        if not candidates:
            raise RuntimeError(f"{n} 个态在 [{lo},{hi}] 约束下无可行窗口划分")
    peak, w_used, ssq, ranges = min(candidates)

    validate_single_shared_boundary_ranges(ranges, n)
    per_window = [cost(a, b) for a, b in ranges]
    return ranges, {
        "criterion": "metric_integral_g",
        "n_windows": int(w_used),
        "sizes": [int(b - a) for a, b in ranges],
        "metric_integral_per_window": [float(x) for x in per_window],
        "peak_metric_integral": float(peak),
        "imbalance_max_over_min": (
            float(max(per_window) / min(per_window)) if min(per_window) > 0 else None
        ),
        "min_states_per_window": lo,
        "max_states_per_window": hi,
        "note": (
            "峰值有地板：等于尖峰处一个最小窗的 ∫g；再低只能在尖峰段加 λ 态。"
        ),
    }


def partition_tail_by_arclength(
    arc_tail: Sequence[float],
    min_states_per_window: int = 4,
    max_states_per_window: int = 5,
    *,
    n_windows: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """把**尚未采样的那一段**按热力学长度重新划成 few-state IBS 窗口。

    只作用于给定的这一段（下标相对本段）。用途：失败窗口插 λ 之后，前缀（已经
    采完的窗口）必须逐字冻结，而从失败窗口起到路径末尾的部分还没跑过、可以自由
    重排 —— 但它**必须仍然是按热力学长度均衡的窗口**，不能只是把原来的边界往后
    推一格。否则剩余布局就不再是热力学窗，只是"结构合法"而已。

    窗口数由约束定死：相邻共享一个边界态 ⟹ ``L = Σsizes - (W-1)``，两侧都要落在
    ``[min, max]`` ⟹ ``W = ceil((L-1)/(max-1))``，再校验 ``min*W - (W-1) <= L``。
    切点在可行范围内取最接近等弧长的那个。
    """
    arc = np.asarray(arc_tail, dtype=float).ravel()
    n = arc.size
    lo, hi = int(min_states_per_window), int(max_states_per_window)
    if hi < lo or lo < 2:
        raise ValueError(f"min/max_states_per_window 非法：{lo}/{hi}")
    if n < lo:
        raise RuntimeError(f"尾段只有 {n} 个态，不足 min_states_per_window={lo}")
    # 🔑 [2026-09-11] n_windows 从"自己按 max 算最少窗口数"改成可由调用方**硬指定**。
    # 原来它无条件取 ceil((n-1)/(max-1))，也就是尽可能少切；插点补救时这会把尾段
    # 原有的 3 个窗重新合并成 3 个大窗（max=8 之后尤其明显），于是"插了点、失败窗口
    # 反而更大"。补救路径现在显式传 w_before+1，逼它真的多切一刀。
    if n_windows is None:
        if n <= hi:
            return [(0, n)]
        n_windows = int(np.ceil((n - 1) / (hi - 1)))
    else:
        n_windows = int(n_windows)
        if n_windows < 1:
            raise ValueError(f"n_windows 必须 >= 1，收到 {n_windows}")
    if n_windows == 1:
        if not lo <= n <= hi:
            raise RuntimeError(f"尾段 {n} 个态装不进 1 个 [{lo},{hi}] 态窗口")
        return [(0, n)]
    if n_windows * (hi - 1) + 1 < n:
        raise RuntimeError(
            f"尾段 {n} 个态装不进 {n_windows} 个窗口（max_states_per_window={hi}）"
        )
    if lo * n_windows - (n_windows - 1) > n:
        raise RuntimeError(f"尾段 {n} 个态切不出 {n_windows} 个 [{lo},{hi}] 态窗口")

    span = float(arc[-1] - arc[0])
    cuts = [0]
    for w in range(n_windows - 1):
        start = cuts[-1]
        remaining = n_windows - w - 1
        target = arc[0] + span * (w + 1) / n_windows
        lo_cut, hi_cut = start + lo - 1, min(start + hi - 1, n - 1)
        # 切完之后剩下的态数（含共享边界）必须还够剩余窗口各自满足 [lo,hi]
        while hi_cut > lo_cut and (n - hi_cut) < lo * remaining - (remaining - 1):
            hi_cut -= 1
        while lo_cut < hi_cut and (n - lo_cut) > hi * remaining - (remaining - 1):
            lo_cut += 1
        if lo_cut > hi_cut:
            raise RuntimeError(f"尾段无可行切点：start={start} n={n} W={n_windows}")
        cand = np.arange(lo_cut, hi_cut + 1)
        cuts.append(int(cand[int(np.argmin(np.abs(arc[cand] - target)))]))
    cuts.append(n - 1)
    ranges = [(cuts[i], cuts[i + 1] + 1) for i in range(len(cuts) - 1)]
    if sorted({i for a, b in ranges for i in range(a, b)}) != list(range(n)):
        raise RuntimeError(f"尾段切分未完整覆盖：{ranges}")
    return ranges



STAGE2_CONTROLLER_PROTOCOL_VERSION = 1


def _ie_min_frames():
    """去相关帧数下限（可达性预检的 T）。惰性读，避免顶层拖 ibs_engine。"""
    try:
        from ibs_engine import IBS_LOCAL_MBAR_GATE_MIN_FRAMES
        return int(IBS_LOCAL_MBAR_GATE_MIN_FRAMES)
    except Exception:
        return 10



def _ie_reach(*args, **kwargs):
    """惰性转发到 `ibs_engine.validation_reachability_verdict`（别在顶层拖重依赖）。"""
    from ibs_engine import validation_reachability_verdict
    return validation_reachability_verdict(*args, **kwargs)


SEALED_CANDIDATES_FILENAME = "stage2_fk_sealed_candidates.json"


def _sealed_candidates_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, SEALED_CANDIDATES_FILENAME)


def read_sealed_candidates(checkpoint_dir):
    """读被统计驳回、**永不续验**的 f_k 候选台账。读不到返回空表。"""
    try:
        with open(_sealed_candidates_path(checkpoint_dir), encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return []
    return list((payload or {}).get("sealed") or [])


def seal_refuted_candidate(
    checkpoint_dir,
    *,
    path_version,
    window_idx,
    lambdas_vdw,
    f_k,
    reason,
    fingerprint=None,
):
    """封存一份被驳回的 f_k 候选：**永不续验**。

    身份 = `path_version + window + λ 身份 + f_k 向量本身`。

    ⚠️ **候选身份不用 hash。** 主线在快速反复变动，哈希里放什么一改，所有已封存
    记录就全部失配、被驳回的候选会被当成新的重新试一遍 —— 正是"反复试到偶然
    通过"。而且这是本仓库记录在案、**已经复发四次**的同一个坑：自产产物的
    sha256 进身份。规则是「只有用户输入才配做身份」，而 f_k 是我们自己算出来的。
    正解不是删掉身份，是改成**语义身份**：f_k 向量（mean-center 后比距离，
    见 `ibs_engine.sealed_candidate_matches`）。
    `fingerprint` 只作为可选的溯源线索留着，**不参与任何判定**。

    ⚠️ 这里**只记录**，不决定下一步动作。统计驳回不得被解释成"λ 太稀"，
    所以它永远不触发插 λ / 拆窗。
    """
    entries = read_sealed_candidates(checkpoint_dir)
    record = {
        "path_version": int(path_version),
        "window_idx": int(window_idx),
        "lambda_identity": [round(float(x), 10) for x in (lambdas_vdw or [])],
        # **身份**：f_k 向量本身。判"实质上是不是同一份"在 f_k 空间里比距离
        # （mean-center 后，见 ibs_engine.sealed_candidate_matches）。
        "f_k_kJ_mol": [float(x) for x in (f_k or [])],
        # 仅溯源，不参与判定。主线变动会让它失配，所以它不配做身份。
        "candidate_fingerprint_PROVENANCE_ONLY": fingerprint,
        "reason": reason,
        "never_revalidate": True,
    }
    entries.append(record)
    payload = {
        "note": (
            "被统计驳回的 f_k 候选。**永不续验**；每个 (path_version, window) "
            "只允许一次替代候选（RELEARN_FK_EPOCH），防止反复试到偶然通过。"
        ),
        "sealed": entries,
    }
    tmp = _sealed_candidates_path(checkpoint_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, _sealed_candidates_path(checkpoint_dir))
    return record


def relearn_epoch_used(checkpoint_dir, path_version, window_idx):
    """这个 (path_version, window) 是否已经用掉了那**唯一一次**替代候选。

    ⚠️ `REJECTED → RELEARN` 和 `UNREACHABLE → RELEARN` 共用同一个配额 ——
    否则一个窗口能走两扇门拿两次 fresh Epoch，一次性护栏就形同虚设。
    """
    return any(
        int(e.get("path_version", -1)) == int(path_version)
        and int(e.get("window_idx", -1)) == int(window_idx)
        and e.get("relearn_consumed")
        for e in read_sealed_candidates(checkpoint_dir)
    )


def mark_relearn_epoch_consumed(checkpoint_dir, path_version, window_idx, detail=None):
    """记下那唯一一次替代候选已被使用。找不到对应封存记录时补一条。"""
    entries = read_sealed_candidates(checkpoint_dir)
    hit = [
        e for e in entries
        if int(e.get("path_version", -1)) == int(path_version)
        and int(e.get("window_idx", -1)) == int(window_idx)
    ]
    if not hit:
        entries.append({
            "path_version": int(path_version),
            "window_idx": int(window_idx),
            "reason": "relearn_without_sealed_candidate",
        })
        hit = [entries[-1]]
    for e in hit:
        e["relearn_consumed"] = True
        if detail:
            e["relearn_detail"] = detail
    payload = {"note": "见 seal_refuted_candidate", "sealed": entries}
    tmp = _sealed_candidates_path(checkpoint_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, _sealed_candidates_path(checkpoint_dir))


class Stage2RepairController:
    """Stage-2 的统一控制器：**看状态 → 选一个动作 → 交执行器 → 回来重新判断。**

    设计依据：``docs/PLAN_PATH_REPAIR_2026-09-11.md``（定位、三类修补、判别、可行性、
    停止标准）。放在 `abfe_preoptimizer` 里、而且是**一个类**，是刻意的：

      · **决策不许散开。** 现状是三层各用一种机制 —— `ibs_engine.run_all_windows`
        尾部 4200 行函数里的 if/else（窗口内 f_k）、`abfe_pipeline` 的异常展开
        （插 λ）、`abfe_pipeline:13027` 的 while 循环（补采）；状态写点 25 处、
        分三个区，其中 resume 那一段是**第二套**从盘上推状态的规则。本类把这些
        判断收到一处。
      · 修补的判据（`insert_lambda_in_failed_ibs_window`、`feasible_repair_actions`）
        本来就在本模块，决策跟它们同处一室，不必跨文件对齐。
      · **状态唯一真源是盘。** 本类只读盘 + 纯判断，**不执行任何动作、不改任何文件**。
        于是 resume 就等于"拿同一份盘上状态再 new 一次、再 decide 一次"，结构上
        不可能长出第二套状态重建规则。

    **本类不做**：任何数值计算。逐态 ESS / g 剖面、mixture 覆盖度这些量已经由
    `ibs_engine` 落在 `*_convergence.json` / stage 结果里；这里只读汇总值。需要
    重新算数的诊断另开脚本，别塞进控制器 —— 控制器一旦自己算数，就没法"拿历史
    run 的产物离线重放决策"了。
    """

    # 动作：执行器能做的事。分两类 —— 不改结构的（前三个）与改结构的（后两个）。
    ACTIONS = (
        "CONTINUE_WARMUP",      # 继续学习 / 继续验证（同一个 Epoch，加预算）
        "RUN_PRODUCTION",       # 跑/补生产帧（同一个 f_k，接着原段）
        "RECALIBRATE_FK",       # 用生产帧重解 f_k → 新 Epoch，旧段保留
        "INSERT_LAMBDA",        # 补 λ 缩窗跨度（model B；溢出落末窗）
        "SPLIT_TAIL_WINDOW",    # 拆末窗（仅末窗，K ∈ [2lo−1, 2hi−1]）
        "PROBE_CANDIDATE_FK",   # **非变异**：离线算候选 f_k + 评估，不切换
        "PROBE_REANCHOR_EPOCH", # held-out **判不了**时：候选 f_k + 独立 burn-in + 一块
        # 🔑 **与 RECALIBRATE_FK 科学语义不同，绝不合并。**
        # RECALIBRATE_FK 是拿**已有生产帧**重解 f_k（信息来自旧轨迹）；
        # RELEARN_FK_EPOCH 是**从头 LEARN**一份全新候选（fresh learn → 新 f_k →
        # burn-in → 独立 held-out 验证），因为旧那份已经被统计驳回、永不续验。
        "RELEARN_FK_EPOCH",     # 候选被驳回/不可达 ⟹ 全新 Epoch 从头学一份 f_k
        "INSERT_LAMBDA",        # 候选也救不了 ⟹ 布局动作（缩窗跨度）
        "ANALYZE",              # 只读：跑 stage 分析（MBAR + 生产质量门）
        "DONE",
    )
    # 🔑🔑 [2026-09-11 老板改目标] **控制器是驱动，不是影子。**
    # 「控制器必须自己诊断、自己换动作、自己继续跑，最终自动产出结果。
    #   任何 LOCAL_* 都必须被主循环消费，**不能炸出流水线**。」
    #
    #     while not DONE:
    #         evidence = read()
    #         action   = decide(evidence, feasible_actions, global_budget)
    #         execute(action)
    #
    # ⟹ 出口要分成两类。**只有这三种允许真正终止**，其余一律是**路由信号**
    #    （被主循环消费、换个动作继续跑）：
    TERMINAL_EXITS = (
        "GLOBAL_BUDGET_EXHAUSTED",   # 全局预算真的没了（**局部**耗尽不算）
        "NO_FEASIBLE_ACTION",        # 所有动作都不可行
        "HALT_INVALID_INPUT",        # 输入 / Hamiltonian 无效
        "DONE",
        "DONE_UNTRUSTED",
    )

    # 出口：不是动作，是结局。每一个都必须说明"缺什么"或"下一步谁来做"。
    EXITS = (
        "DONE",
        "DONE_UNTRUSTED",                    # 门未过但调用方显式放行
        "HALT_BUDGET",                       # **全局**预算耗尽，动作本身可行
        # ⚠️ 与 HALT_BUDGET **不是**同一回事：撞的是单周期的验证**批次上限**
        # （IBS_LOCAL_MBAR_GATE_MAX_BATCHES），而全局预算**还有钱**。
        # 实测 win4：批次打满、`global_budget_remaining = 760k`。
        # 复用 HALT_BUDGET 会把"没钱"和"这一轮批次用完"混成一个归因。
        "HALT_LOCAL_VALIDATION_CAP",
        # ⚠️ 与 HALT_LOCAL_VALIDATION_CAP 的区别：那个是"这一轮批次用完了、
        # 再给一轮也许行"；这个是**算术上证明了**在剩余预算内不可能凑够去相关
        # 帧数（g 太大）。**它只改路由，不改 verdict** —— 证据仍是 UNMEASURED，
        # 绝不因此变成 REJECTED，也绝不因此插 λ / 拆窗。
        "HALT_VALIDATION_BUDGET_UNREACHABLE",
        "HALT_NO_ATTRIBUTION",               # 测不动且归因不出来（合法结局）
        "HALT_NO_FEASIBLE_ACTION",           # 归因成功但动作都不可行
        "HALT_FK_REFUTED",                   # f_k 有证据被驳回（终态）
        "HALT_LAMBDA_BUDGET_INSUFFICIENT",   # 溢出槽耗尽 ⟹ 输入 λ 总数不够
        "HALT_TRUNCATED_PATH",               # 缺窗口 ⟹ 总和不是完整 ΔG
        "HALT_INVALID_INPUT",                # 身份/输入不一致
    )

    # `bias_status` 六个值混了"还在流程中"（前三）与"已有裁决"（后三）。
    # 拆开才知道是"在跑"还是"有结论了"。
    _PHASE = {
        "unconverged": "WARMUP_LEARN",
        "calibrated_pending_validation": "WARMUP_VALIDATE",
        "frozen_validation_indeterminate": "WARMUP_VALIDATE",
        "converged": "PRODUCTION",
        "failed": "TERMINAL",
        "calibrated_validation_failed": "TERMINAL",
    }
    # f_k 证据 → verdict 词汇表。**三值不够**：只有 PASS/FAIL 时，"FAIL 但有预算
    # → 再测一次"会退化成重试到碰巧通过。有统计功效的否决必须立刻换 Epoch。
    # ⚠️ 低支撑**永远不是 FAIL**。"尚不可测"（INSUFFICIENT_DATA / HARD_INSUFFICIENT）
    # 加预算；FAIL（STATISTICALLY_REJECTED）换 Epoch。混起来会退化成"再测一次直到
    # 碰巧通过"。`ANALYSIS_ELIGIBLE` 只表示**可以进入分析**，不是通过验收 ——
    # 最终 PASS 还要 endpoint CI、block 稳定性、全路径完整性。
    _VERDICT = {
        "verified": "VALID_PASS",
        "calibrated": "INSUFFICIENT_DATA",
        "indeterminate": "INSUFFICIENT_DATA",
        "refuted": "STATISTICALLY_REJECTED",
        "none": "INSUFFICIENT_DATA",
    }

    def __init__(
        self,
        run_dir: str,
        stage_name: str = "vanishing",
        stage_type: str = "vdw",
        *,
        min_states_per_window: Optional[int] = None,
        max_states_per_window: Optional[int] = None,
        max_path_insertions: Optional[int] = None,
        allow_untrusted_stage_results: bool = False,
    ):
        self.run_dir = os.path.abspath(run_dir)
        self.stage_name = str(stage_name)
        self.stage_type = str(stage_type)
        # 🔑 lo/hi **默认从 run 自己的 run_provenance.json 读**，不要求调用方记得传。
        # 手动传错的后果很实在：可拆区间是 [2lo−1, 2hi−1]，4/5 是 7..9、4/8 是
        # 7..15，判出来的"可不可行"会完全不同。run 自己记了它跑的是什么，就用那个。
        _cfg = (self._json(os.path.join(self.run_dir, "run_provenance.json")) or {}).get("config") or {}
        self.config_source = "run_provenance.json" if _cfg else "caller/default"
        self.lo = int(min_states_per_window if min_states_per_window is not None
                      else _cfg.get("stage2_window_min_states", 4))
        self.hi = int(max_states_per_window if max_states_per_window is not None
                      else _cfg.get("stage2_window_max_states", 5))
        self.max_path_insertions = int(
            max_path_insertions if max_path_insertions is not None
            else _cfg.get("max_path_insertions", 3)
        )
        self.allow_untrusted = bool(allow_untrusted_stage_results)
        self.stage_dir = os.path.join(self.run_dir, self.stage_name)
        # 🔑 **多采样段有自己的 checkpoint 命名空间。** 约定见
        # `abfe_pipeline._recalibrate_fk_and_resample_segment`：
        #   段输出目录 = f"{stage_dir}_{segment_index}"       → `vanishing_2`
        #   段 checkpoint = checkpoints/f"segment_{index}"     → `checkpoints/segment_2`
        # 读错子目录的后果很具体：段 2 会读到段 1 的 ibs_state，把段 1 的
        # frozen_validation_cumulative_steps（实测 205000）当成段 2 的
        # （真值 0）报出来。路径版本链仍在顶层 checkpoints（λ 路径是全局的）。
        _base_ckpt = os.path.join(self.run_dir, "checkpoints")
        _m = re.match(r"^(.+)_(\d+)$", self.stage_name)
        self.segment_index = int(_m.group(2)) if _m else None
        self.checkpoint_dir = (
            os.path.join(_base_ckpt, f"segment_{self.segment_index}")
            if self.segment_index is not None else _base_ckpt
        )
        self.path_checkpoint_dir = _base_ckpt

    # ---------------------------------------------------------------- 读盘

    @staticmethod
    def _json(path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _read_path(self) -> Dict[str, Any]:
        """λ 路径版本 + 演化事件计数。事件必须数**整条链**，那才是跨 resume 的
        累计轮数（`_run_stage2_with_path_evolution` 的 rounds_done 同口径）。"""
        # λ 路径是**全局**的（不分段），版本链永远在顶层 checkpoints。
        _pd = self.path_checkpoint_dir
        cur = self._json(os.path.join(_pd, "path_current.json")) or {}
        version = cur.get("version")
        record = None
        if version is not None:
            record = self._json(
                os.path.join(_pd, "path_versions", f"v{int(version)}.json")
            )
        events: Dict[str, int] = {}
        for vf in sorted(glob.glob(os.path.join(_pd, "path_versions", "v*.json"))):
            kind = (self._json(vf) or {}).get("kind")
            if kind:
                events[kind] = events.get(kind, 0) + 1
        ranges = (record or {}).get("window_ranges")
        return {
            "version": version,
            "events": events,
            "n_states": len((record or {}).get("states", [])) or None,
            "window_ranges": [tuple(int(x) for x in r) for r in ranges] if ranges else None,
        }

    def _read_stage_result(self) -> Optional[Dict[str, Any]]:
        """stage 级结果（含生产质量门、缺窗口清单）。**只在 stage 跑完才存在。**"""
        for f in sorted(glob.glob(os.path.join(self.path_checkpoint_dir, "stage2_*.json"))):
            d = self._json(f)
            if isinstance(d, dict) and ("converged" in d or "total_delta_G" in d):
                return d
        return None

    def _read_window(self, idx: int) -> Dict[str, Any]:
        conv = self._json(os.path.join(
            self.stage_dir, f"dual_window_{idx}_{self.stage_type}_convergence.json"))
        fail = self._json(os.path.join(
            self.stage_dir, f"dual_window_{idx}_{self.stage_type}_warmup_failure.json"))
        state = self._json(os.path.join(
            self.checkpoint_dir, f"ibs_state_{self.stage_type}_window_{idx}.json"))
        lam = (conv or {}).get("lambdas_vdw") or (state or {}).get("lambdas_vdw") or []
        warm = (conv or {}).get("bias_warmup") or (fail or {}).get("bias_warmup") or fail or {}
        ledger = warm.get("warmup_budget_ledger") or {}
        # 🔑 [P2-9a+ / 老板第 3 条] **warmup 剖面是一等证据** ——
        # 「哪个窗弱，就可以提前增加采样了」。上一跑的实测支持这条：两个
        # occupancy 塌的窗口正是 coverage_ess 最低的两个，而 win3 拿到 2× 预算活了、
        # win0 拿默认 1× 死了（去相关后只剩 6 帧）。win3/win4 那 2×/4× 是上一轮
        # rescue **事后**补的，不是按 warmup 信号**提前**给的 —— 所以 win0 从来没
        # 进过那个名单。⟹ 生产预算本可以在**生产开始之前**按该窗口自己的剖面配。
        # ⚠️ 这里**只落盘 + 只排序，不给公式**：五个点不足以定"coverage_ess < X
        # 就给 N× 预算"，拍一个就是引入未验证阈值（同 P3-9c 那条线）。
        gate = (warm.get("local_mbar_loose_gate") or {}).get("last") or {}
        # [P2-9c] 窗口跑完那一刻的自检：它自己的去相关帧数够不够。
        # 与 `solve_stage_integrated` 判跳过用的**同一个量**（energies + 门槛 10），
        # 只是提前到窗口刚跑完就算。**只报告**，不改预算、不碰放行判据。
        selfchk = self._json(os.path.join(
            self.stage_dir, f"dual_window_{idx}_{self.stage_type}_self_support.json"
        )) or {}
        bias_status = (state or {}).get("bias_status")
        evid = (state or {}).get("f_k_evidence_status")
        spent = (
            sum(int(ledger.get(k, 0) or 0) for k in
                ("learning_steps", "freeze_burn_in_steps", "frozen_validation_steps"))
            if ledger else None
        )
        cap = ledger.get("cumulative_cap_steps") if ledger else None
        prod = (conv or {}).get("cumulative_production_steps")
        if prod is None:
            prod = (conv or {}).get("actual_production_steps")
        return {
            "window_idx": idx,
            "has_convergence": conv is not None,
            "has_warmup_failure": fail is not None,
            "n_states": len(lam) or None,
            "lambda_span": (max(lam) - min(lam)) if lam else None,
            "bias_status": bias_status,
            "phase": (
                # 身份对不上时，盘上的 converged 不算数 —— 那是另一个系综的结论。
                "IDENTITY_MISMATCH"
                if (
                    (state or {}).get("stage_protocol_key") is not None
                    and (conv or {}).get("stage_protocol_key") is not None
                    and (state or {}).get("stage_protocol_key")
                    != (conv or {}).get("stage_protocol_key")
                )
                else self._PHASE.get(bias_status or "", "UNKNOWN")
            ),
            "f_k_evidence_status": evid,
            "verdict": self._VERDICT.get(evid or "", "INSUFFICIENT_DATA"),
            "last_failure_reason": (state or {}).get("last_failure_reason"),
            "last_gate_error": warm.get("last_gate_error") or (fail or {}).get("last_gate_error"),
            "best_effort_acceptance": bool(warm.get("best_effort_acceptance")),
            "best_effort_reason": warm.get("best_effort_acceptance_reason"),
            "bias_update_count": warm.get("bias_update_count"),
            # warmup 剖面（只报告；预算映射未定，见 PLAN P3）
            "warmup_g": gate.get("statistical_inefficiency"),
            "warmup_n_frames_used": gate.get("n_frames_used"),
            "warmup_min_absolute_ess": gate.get("min_absolute_ess"),
            "warmup_min_ess_ratio": gate.get("min_ess_ratio"),
            "warmup_max_adjacent_delta_kJ_mol": gate.get("max_adjacent_delta_kJ_mol"),
            "warmup_gate_threshold_kJ_mol": gate.get("gate_threshold_kJ_mol"),
            # 生产后自检（P2-9c）
            "self_n_frames_decorrelated": selfchk.get("n_frames_decorrelated"),
            "self_min_frames": selfchk.get("min_frames_per_window"),
            "self_sufficient": selfchk.get("sufficient"),
            "self_verdict": selfchk.get("verdict"),
            # 为什么是这个 verdict：solver_eligibility / min_n_eff_over_g /
            # top1pct_veto。三者补救方向都是加采样，但归因不同，对账时要分得开。
            "self_verdict_source": selfchk.get("verdict_source"),
            "self_frames_short_by": selfchk.get("frames_short_by"),
            # 验收量：N_eff,k / g_k（未抽稀帧上逐目标态 support ÷ 时间自相关）。
            # ⚠️ occupancy **不是**验收判据 —— 它是 f_k 的训练目标；win1 段1 的
            # occupancy_collapsed=True 但支撑健康，是实测假阳性。
            "min_n_eff_over_g": selfchk.get("min_n_eff_over_g"),
            # 生产侧累计 f_k 偏差（scope=production），由 solve_stage_integrated 落在
            # stage 结果的 `cumulative_fk_residual_production` 里。
            "cum_fk_span": None, "cum_fk_verdict": None,  # 在 read() 里按窗口填
            "worst_state_by_n_eff": selfchk.get("worst_state_by_n_eff"),
            # 早判：边际 N_eff 断崖（比固定节奏更准，实测脱轨点跨 4.5 倍）
            "derail_at_trajectory_fraction": selfchk.get("derail_at_trajectory_fraction"),
            "derailment_status": selfchk.get("derailment_status"),
            "derail_block_index_single_block": selfchk.get("derail_block_index_single_block"),
            # 身份：state 里持久化的采样身份 vs 本窗产物的身份。
            # 不一致 ⟹ 盘上那个 bias_status=converged 是**另一个系综**的旧结论，
            # 控制器不得据此报"已在生产"（同僚今天真机踩到过：能量缓存因身份不符
            # 被拒、整窗重采，而 bias_converged=True 照样粘过去）。
            "state_protocol_key": (state or {}).get("stage_protocol_key"),
            "product_protocol_key": (conv or {}).get("stage_protocol_key"),
            "n_eff_marginal_by_block": selfchk.get("n_eff_marginal_by_block"),
            "warmup_steps_spent": spent,
            "warmup_steps_cap": cap,
            "warmup_steps_left": (
                max(0, int(cap) - int(spent)) if (cap and spent is not None) else None
            ),
            # [2026-09-12] 验证**可达性**预检的原料。同一候选、连续数据上的
            # 多个检查点（`insufficient_attempts` 每次都记一个 g），用来算保守
            # 下界 g_L；再配上 T / Ncap / 剩余预算就能判"在算术上还可不可能"。
            "validation_indeterminate": warm.get("validation_indeterminate"),
            "validation_g_checkpoints": [
                float(x) for x in (
                    warm.get("validation_g_history")
                    or [(warm.get("validation_indeterminate") or {}).get(
                        "statistical_inefficiency")]
                ) if x
            ],
            # 🔑 T = **去相关**帧数下限（触发 insufficient_frames_after_decorrelation
            # 的那个），不是 `minimum_complete_validation_frames`（原始帧完整性要求）。
            # 混掉会把 gcrit 算小 20 倍：win4 只差 26% 帧数却被判成差 7.5 倍不可达。
            # 老产物没落这个字段时回退到常量，绝不回退到 200。
            "validation_required_frames": (
                (warm.get("validation_indeterminate") or {}).get(
                    "decorrelated_frames_required")
                or _ie_min_frames()
            ),
            "validation_completeness_frames_REPORT_ONLY": warm.get(
                "minimum_complete_validation_frames"),
            "validation_sample_count": (
                (warm.get("validation_indeterminate") or {}).get("validation_sample_count")
            ),
            "frozen_validation_steps": (state or {}).get("frozen_validation_cumulative_steps"),
            "frozen_validation_batches": (state or {}).get("frozen_validation_batches_done"),
            "production_steps": prod,
            "production_steps_target": (conv or {}).get("n_steps_per_window_effective"),
            "n_production_segments": len((conv or {}).get("production_segments") or []) or None,
            "n_frames": ((conv or {}).get("window_data") or {}).get("n_frames"),
        }

    def _read_single_stage(self) -> Dict[str, Any]:
        """**单个段**的状态 + 证据 + 剩余预算。只读盘。

        ⚠️ 物理 stage 的决策**不要**直接用它 —— 走 `for_physical_stage()` 的聚合
        视图，否则同一个 stage 会被当成多个独立 stage 各判一次。
        """
        found: List[int] = []
        for pat, rx in (
            (f"dual_window_*_{self.stage_type}_convergence.json", r"dual_window_(\d+)_"),
            (f"dual_window_*_{self.stage_type}_warmup_failure.json", r"dual_window_(\d+)_"),
        ):
            for f in glob.glob(os.path.join(self.stage_dir, pat)):
                m = re.search(rx, os.path.basename(f))
                if m and int(m.group(1)) not in found:
                    found.append(int(m.group(1)))
        # [P2-9a] join λ 两侧支撑：相邻窗口对**共享的那一个 λ** 各自的重要性支撑。
        # 由 `ibs_engine.join_lambda_two_sided_support` 在每个窗口落盘后自动算并
        # 落成 `dual_join_{up}_{down}_{type}_support.json`。**只报告、不参与放行。**
        joins = []
        for f in sorted(glob.glob(os.path.join(
            self.stage_dir, f"dual_join_*_{self.stage_type}_support.json"
        ))):
            d = self._json(f)
            if isinstance(d, dict):
                joins.append(d)
        joins.sort(key=lambda d: int(d.get("upstream_window", -1)))
        # [P2-9h] f_k 重标定探针：在 rescue **之前**判"该重标定还是该加帧"。
        # 位移（`max_adjacent_shift_kJ_mol`）是直接证据，偏斜（top1%）只是症状。
        fk_probe = self._json(os.path.join(
            self.path_checkpoint_dir, "stage2_fk_recalibration_probe.json"
        )) or {}
        path = self._read_path()
        expected = len(path["window_ranges"] or []) or None
        windows = [self._read_window(i) for i in sorted(found)]
        # 生产侧累计 f_k 偏差在 **stage 结果**里（scope=production），逐窗填回。
        # ⚠️ 必须在 `stage` 读出来**之后**做 —— 顺序写反过一次，UnboundLocalError。
        _stage_for_cum = self._read_stage_result()
        _cum_by_win = {
            int(x.get("window_index", -1)): x
            for x in ((_stage_for_cum or {}).get(
                "cumulative_fk_residual_production") or [])
        }
        for _w in windows:
            _c = _cum_by_win.get(int(_w["window_idx"])) or {}
            _w["cum_fk_span"] = _c.get("cumulative_residual_span_kJ_mol")
            _w["cum_fk_verdict"] = _c.get("verdict")
            _h = self._json(os.path.join(
                self.stage_dir,
                f"dual_window_{_w['window_idx']}_{self.stage_type}_heldout.json",
            )) or {}
            _w["heldout_verdict"] = _h.get("verdict")
            _w["heldout_worst_before"] = _h.get("worst_before")
            _w["heldout_worst_after"] = _h.get("worst_after")
        stage = self._read_stage_result()
        # 缺窗口：布局里有、产物里没有。这是"截断的 ΔG"这类失效的直接信号，
        # 现在只在日志里出现一次 WARN。stage 结果里的 skipped_windows 是另一种
        # （产物在、但去相关后帧数不足被踢出协方差链），两者都要算进来。
        missing = [i for i in range(expected) if i not in found] if expected else []
        skipped = [
            int(x.get("window_index", -1))
            for x in ((stage or {}).get("skipped_windows") or [])
        ]
        return {
            "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
            "run_dir": self.run_dir,
            "stage_name": self.stage_name,
            "stage_type": self.stage_type,
            "min_states_per_window": self.lo,
            "max_states_per_window": self.hi,
            "config_source": self.config_source,
            "segment_index": self.segment_index,
            "checkpoint_dir_used": self.checkpoint_dir,
            # 插点是有**终身**预算的（rounds_done 从版本链累计，跨 resume 有效）。
            "path_insertions_done": int(path["events"].get("insert_lambda", 0)),
            # 尾段重分过几次 —— 崩溃恢复靠它判「已经重分过没有」，
            # 没有它第二次启动会重新重分、把刚跑的新尾段作废。
            "tail_repartitions_done": int(path["events"].get("tail_repartition", 0)),
            "path_insertions_budget": self.max_path_insertions,
            # 🔑🔑 [2026-09-11 更正] **不许把各窗余量求和当"全局预算"。**
            #
            # `cumulative_cap_steps` 是**逐窗**的（实测每窗 955k = max_bias_warmup
            # 900k + burn-in 5k + 验证预留 50k），**没有任何全局池** ——
            # 你不能拿 win0 的余额去给 win4 花。求和（实测 4.29M）是个**无意义的量**，
            # 而且后果严重：它会让"全局还有钱"几乎永真 ⟹ 所有 HALT_BUDGET 都被判成
            # **路由** ⟹ 主循环**永不终止**。
            #
            # 正确口径：
            #   · 路由/终止用**相关窗口自己的**余量；
            #   · "全局耗尽"在双层预算（PLAN 第 5 步 F）做出来之前**没有可靠数据源**，
            #     这里实现成保守占位：**所有**窗口余量都为 0 才算全局耗尽。
            "per_window_budget_remaining": {
                int(w["window_idx"]): int(w.get("warmup_steps_left") or 0)
                for w in windows
            },
            "all_windows_budget_exhausted": bool(
                windows and all(
                    int(w.get("warmup_steps_left") or 0) <= 0 for w in windows
                )
            ),
            "global_budget_source": (
                "placeholder: 逐窗 cap，无全局池；双层预算（F）未实现前的保守口径"
            ),
            "path_insertions_left": max(
                0, self.max_path_insertions - int(path["events"].get("insert_lambda", 0))
            ),
            "path": path,
            "n_windows_found": len(windows),
            "n_windows_expected": expected,
            "missing_windows": missing,
            "skipped_windows": skipped,
            "production_rescue_targets": (stage or {}).get("production_rescue_targets") or {},
            "stage_converged": (stage or {}).get("converged"),
            "stage_path_is_complete": (stage or {}).get("path_is_complete"),
            "stage_total_delta_G": (stage or {}).get("total_delta_G"),
            "has_stage_result": stage is not None,
            "joins": joins,
            "fk_probe": fk_probe,
            "windows": windows,
        }

    # ------------------------------------------------------------ 可行性

    def feasible(self, view: Optional[Dict[str, Any]] = None, n_insert: int = 1) -> Dict[str, Optional[str]]:
        """结构性动作在当前布局下可不可行。

        复用模块级的 `feasible_repair_actions` —— 那条规则
        （可拆区间 `[2lo−1, 2hi−1]`、溢出槽上界）必须与
        `insert_lambda_in_failed_ibs_window` 里的守卫**同一份**，不许各写一遍。
        """
        view = view or self.read()
        ranges = view["path"]["window_ranges"]
        n_states = view["path"]["n_states"]
        if not ranges or not n_states:
            return {
                "insert_lambda": "读不到路径版本链，无法判断布局可行性",
                "split_tail_window": "读不到路径版本链，无法判断布局可行性",
            }
        return feasible_repair_actions(
            ranges, int(n_states),
            min_states_per_window=self.lo, max_states_per_window=self.hi,
            n_insert=int(n_insert),
        )

    # -------------------------------------------------------------- 决策

    def decide(self, view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """**纯函数式判断：下一步做什么。** 不执行、不落盘。

        优先级是刻意排的，理由写在每一条里。核心一条：**`UNMEASURED`（没测出来）
        永远不当 `FAIL`（测出来不合格）用** —— 前者加预算，后者换 Epoch；混起来
        就会退化成"再测一次直到碰巧通过"。
        """
        view = view or self.read()
        W = view["windows"]
        feas = self.feasible(view)

        def plan(action, reason, *, exit_=None, windows=None, missing=None,
                 blocked=None, earliest=None):
            """⚠️ `blocked` / `earliest` **必须显式传**，不许从闭包里取。

            闭包引用会在**全新运行**上炸 NameError：分支 0（读不到任何窗口）在
            `earliest` 被构造**之前**就 return，而 `plan()` 里引用了它。
            而全新运行正是自治系统**必须首先支持**的路径 —— 路径写错只是把它
            提前暴露出来。默认 None/[] 让每个调用点都能安全地不传。
            """
            blocked = list(blocked or [])
            return {
                "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
                "action": action,
                "exit": exit_,
                "reason": reason,
                "windows": windows or [],
                "missing_evidence": missing or [],
                "feasible_structural_actions": [k for k, v in feas.items() if v is None],
                "infeasible_structural_actions": {k: v for k, v in feas.items() if v},
                # 停止标准三维：执行状态 / 证据状态 / 可信级别，三件不同的事。
                # `allow_untrusted_stage_results` **只能改 trust_level**，
                # 不得把 evidence_status 改写成 CONVERGED。
                "execution_status": (
                    # 只有**真终止**才是 HALTED；路由信号仍然是 IN_PROGRESS ——
                    # 控制器换个动作继续跑，流水线没有停。
                    "HALTED" if (exit_ in self.TERMINAL_EXITS and action != "DONE")
                    else ("COMPLETE" if action == "DONE" else "IN_PROGRESS")
                ),
                "evidence_status": self._evidence_status(view, action, exit_),
                "trust_level": (
                    "OVERRIDDEN_UNTRUSTED" if self.allow_untrusted else "STATISTICAL_ONLY"
                ),
                # **终止 vs 路由**：路由信号必须被主循环消费，不得炸出流水线。
                # 被上游阻塞的窗口：不是"排队等"，而是**它们的证据前提可能已作废**
                # （上游重锚会换掉共享态起点 / 末帧 lineage）。
                "blocked_by_upstream": blocked if windows else [],
                "blocked_by": (
                    int(earliest) if (blocked and windows and earliest is not None)
                    else None
                ),
                "terminal": bool(exit_ in self.TERMINAL_EXITS),
                "routing": bool(exit_ is not None and exit_ not in self.TERMINAL_EXITS),
                # ⚠️ **只是提示，不是动作。** 影子模式报"按 warmup 剖面谁最弱"，
                # 好让"提前配预算"这件事先被看见；**不给阈值、不给倍数**，
                # 剖面→预算的映射是未定项（PLAN P3）。
                "warmup_weakness_ranking": self._warmup_weakness_ranking(view),
            }

        # 0) 读不到任何窗口 —— 还没开跑，或者路径/目录给错了。别猜。
        if not W:
            return plan(
                "CONTINUE_WARMUP",
                f"{self.stage_dir} 下没有任何窗口产物 —— stage 还没开始，或 "
                "run_dir/stage_name 给错了。",
                missing=["任一窗口的 convergence/warmup_failure 产物"],
            )

        # 🔑🔑 [2026-09-11 老板定案] **按因果依赖顺序处理"最早的未解决窗口"**，
        # **不是**按错误类型设全局优先级、也不是"看见缺窗就补"。
        #
        #     for window_idx in stage_order:
        #         if window is not ANALYSIS_ELIGIBLE:
        #             return route_this_window(window_idx)
        #     return RUN_NEXT_MISSING_WINDOW
        #
        # 实测 mismatch（replay 基线照出来的）：rep1 里 win3 支撑不足、win4 卡在
        # local cap，而"缺窗口"分支优先级最高 ⟹ 目标被判成 **win5**，把 win3/win4
        # 全盖住了。
        #
        # **物理理由，不只是调度洁癖**：win3 重锚会产生**新的末帧 / 共享态起点**，
        # win4 当前的 warmup 属于**旧上游 lineage**。win3 成功之后才谈得上 win4 能
        # 不能复用；若它的初始化依赖 win3，就得**重新开始 win4**，不能直接接旧 warmup。
        # （这跟"所有窗口共用同一份起始坐标"是同一件事的两面。）
        # ⚠️ **三态，不是两态。** 先前写成 `self_verdict != "ANALYSIS_ELIGIBLE"`，
        # 于是**没有自检产物**的老 run（`self_verdict is None`）里每个窗口都被判成
        # "未解决" ⟹ earliest 永远是 win0，动作永远落在第一个窗口上。
        #
        # "缺证据不是通过"是对的，但**缺证据也不等于「这个窗口有问题」** ——
        # 它等于「不知道」。对「不知道」的正确动作是**去产出证据**（分析/自检），
        # 不是重标定。混成一态会让控制器在毫无根据的窗口上换 Epoch。
        _order = sorted(W, key=lambda x: int(x["window_idx"]))

        def _window_state(x: Dict[str, Any]) -> str:
            if x.get("phase") in ("IDENTITY_MISMATCH", "TERMINAL"):
                return "PROBLEM"
            if x.get("verdict") == "STATISTICALLY_REJECTED":
                return "PROBLEM"
            v = x.get("self_verdict")
            if v in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA"):
                return "PROBLEM"
            if v == "ANALYSIS_ELIGIBLE":
                return "ELIGIBLE"
            # 还在预热/验证、或根本没有自检产物 ⟹ 不知道
            if x.get("phase") in ("WARMUP_LEARN", "WARMUP_VALIDATE"):
                return "PROBLEM"     # 在跑但没跑完，属于"这个窗口还没解决"
            return "UNKNOWN"

        _states = {int(x["window_idx"]): _window_state(x) for x in _order}
        earliest = next(
            (i for i in sorted(_states) if _states[i] == "PROBLEM"), None
        )
        unknown = [i for i in sorted(_states) if _states[i] == "UNKNOWN"]
        blocked = (
            [int(x["window_idx"]) for x in _order if int(x["window_idx"]) > earliest]
            if earliest is not None else []
        )
        _window_states = dict(_states)

        def _pick(items, key: Optional[str] = "window_idx"):
            """**只路由 earliest 这一个窗口**：它不在本分支的候选里就不触发。"""
            if earliest is None:
                return []
            idxs = [int(i) if key is None else int(i[key]) for i in items]
            return [earliest] if earliest in idxs else []

        # 🔑🔑 [2026-09-11 老板定的决策顺序] 选动作**之前先消费预算可行性**：
        #
        #     聚合同一物理 stage 的全部 Segment
        #     → 找 earliest unresolved window
        #     → **检查该窗口动作是否有预算**      ← 这一步
        #     → 按证据选择 RECALIBRATE / PROBE / RESCUE
        #     → 只有完整前缀全部 eligible，才运行下一个缺失窗口
        #
        # 实测 mismatch：`run2/vanishing_2` 在 555k/555k（零余量）时仍被判
        # `RECALIBRATE_FK` —— 动作可行性没有先消费预算。零预算窗口开新 Epoch
        # 只会得到一份**永远验不了**的 f_k（win2 连死三次就是这个形状）。
        if earliest is not None:
            _no_budget = self._epoch_validation_unaffordable(view, [earliest])
            if _no_budget:
                _all_dry = bool(view.get("all_windows_budget_exhausted"))
                return plan(
                    "RUN_PRODUCTION",
                    f"最早未解决的是窗口 {earliest}，但它**付不起新 Epoch 的最低验证"
                    f"额度**（{_no_budget}）⟹ **任何换 Epoch 的动作都不许启动**。"
                    "启动之后才发现没预算验，正是 win2 连死三次的形状。"
                    "低支撑/没预算永远是「尚不可测」，不是 FAIL。",
                    exit_=("GLOBAL_BUDGET_EXHAUSTED" if _all_dry else "HALT_BUDGET"),
                    windows=[earliest], blocked=blocked, earliest=earliest,
                )

        # 1b) **救援已经跑过、仍有窗口被踢出 ⟹ 预算耗尽，不是"结果差一点"。**
        #     rescue 循环只在两种情况退出：converged，或轮数用完。所以
        #     "有 rescue 目标 **且** 仍有 skipped_windows" = 轮数用完仍不够。
        #     此时**必须**落成 HALT_BUDGET + INSUFFICIENT_DATA，**禁止静默缺窗**
        #     然后把部分和端出去 —— 那正是老板划 ✗ 的结局。
        _sel = _pick(view["skipped_windows"], key=None)
        if _sel and view.get("production_rescue_targets"):
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {view['skipped_windows']} 在 rescue 跑过之后**仍然**被踢出"
                f"协方差链（rescue 目标 {view['production_rescue_targets']}）⟹ "
                "轮数已用完而支撑仍不足。这是**预算耗尽**，不是终态失败：低支撑永远是"
                "「尚不可测」（INSUFFICIENT_DATA），不是 FAIL。"
                "⚠️ **在此之前产出的总和缺窗口，不得当作 ΔG 使用。**"
                "要继续必须显式增加预算（或按 §4 改走重标定）。",  # noqa: E501
exit_=(
                    "GLOBAL_BUDGET_EXHAUSTED"
                    if view.get("all_windows_budget_exhausted")
                    # **相关窗口自己**还有钱 ⟹ 路由；它自己没钱但别的窗口有 ⟹
                    # 仍是路由（可以先去跑别的窗口），只有全部为 0 才终止。
                    else "HALT_BUDGET"
                ),
                windows=_sel,
                blocked=blocked, earliest=earliest,
            )

        # 2) f_k 被**有统计功效地**驳回 ⟹ 终态。不许"再测一次"。
        refuted = _pick([w for w in W if w["verdict"] == "STATISTICALLY_REJECTED"])
        if refuted:
            _broke = self._epoch_validation_unaffordable(view, refuted)
            if _broke:
                return plan(
                    "RUN_PRODUCTION",
                    f"窗口 {refuted} 的 f_k 被驳回、本该重标定，但**付不起新 Epoch 的"
                    f"最低验证额度**（{_broke}）⟹ **不启动重标定**。"
                    "启动之后才发现没预算验，正是 win2 连死三次的形状："
                    "循环一次都没进、却被标成「f_k 不收敛」。"
                    "低支撑/没预算永远是「尚不可测」，不是 FAIL。",
exit_=(
                    "GLOBAL_BUDGET_EXHAUSTED"
                    if view.get("all_windows_budget_exhausted")
                    # **相关窗口自己**还有钱 ⟹ 路由；它自己没钱但别的窗口有 ⟹
                    # 仍是路由（可以先去跑别的窗口），只有全部为 0 才终止。
                    else "HALT_BUDGET"
                ),
                    windows=refuted,
                    blocked=blocked, earliest=earliest,
                )
            return plan(
                "RECALIBRATE_FK",
                f"窗口 {refuted} 的 f_k 证据是 refuted（有统计功效的否决）⟹ 必须换 "
                "Epoch 重标定，**不能**再加验证预算。"
                "⚠️ 注意现状：这条路径在代码里抛 IBSFrozenCalibrationValidationError，"
                "而**全仓库没有任何 except 捕获它** —— 现在会直接炸穿整个 run。",
                exit_="HALT_FK_REFUTED", windows=refuted,
            )

        # 3) 还在预热、且预算有余 ⟹ 继续（学习或验证，按 phase 分）。
        #    这是 UNMEASURED 的正确回应：加同类预算，**不换轴**。
        warming = [w for w in W if w["phase"] in ("WARMUP_LEARN", "WARMUP_VALIDATE")]
        with_budget = [w for w in warming if (w["warmup_steps_left"] or 0) > 0]

        # 3a) **批次上限打满、但全局预算还有钱** ⟹ 这不是 HALT_BUDGET。
        #     老板定的口径：`LOCAL_VALIDATION_CAP_EXHAUSTED`，
        #     evidence = INSUFFICIENT_DATA，**不是** F_K_REFUTED（没有任何证据
        #     驳回这份 f_k，只是这一轮没测出来）。
        #     处置（老板给的，win4 是第一个真实用例）：
        #       · **不扩大旧 warmup ladder**（15 批上限不动）
        #       · 进 `PROVISIONAL_PRODUCTION`，**不是可信 PASS**
        #       · **只给一个 +250k 诊断块**
        #       · 块后立即判：N_eff/g 明显增长并达到 10 → 继续；
        #         边际停滞/下降 或 far-end support 单调塌陷 → 关闭 Epoch、tail rewindow；
        #         top1% 灾难性集中 → 停止同分布加帧
        #     ⚠️ 实测块大小：win4 现在 N/g=4.19，+250k ≈ 1369 帧 ⟹ N/g ≈ 9.6，
        #     **恰好达不到 10**。这不是坏事 —— 它让这一块成为**决定性诊断**：
        #     g 若真已平台会稳步走到 ~9.6（再给一块即可）；g 若继续涨会明显低于 9.6
        #     （转 tail rewindow）。两种结局数值上分得很开。
        #     **别因为"差一点到 10"就自作主张给 +500k。**
        try:
            import ibs_engine as _ie_caps
            _batch_cap = int(_ie_caps.IBS_LOCAL_MBAR_GATE_MAX_BATCHES)
        except Exception:
            _batch_cap = None
        _cap_hit = [
            w for w in with_budget
            if _batch_cap is not None
            and (w.get("frozen_validation_batches") or 0) >= _batch_cap
        ]
        # ── 验证**可达性**预检：在剩余预算内还有没有可能凑够去相关帧数 ──────
        # ⚠️ 边界很重要。2026-09-11 已经否决过「周期**内**按 n_eff 外推提前判死」
        # （见 ibs_engine 那段：g 只在 N ≫ τ 后才稳，区间内外推会把本来能测出来的
        # 窗口提前判死）。这里**不违反**那条：
        #   · 判定发生在**周期用尽之后**，决定要不要开**下一个**周期，不掐断本周期；
        #   · `g_L` 取多检查点最小值，不是外推；
        #   · 而且 g 还在随 N 涨这件事**加强**不可达的结论 —— 真 g ≥ 实测 g，
        #     需要的帧数只会更多，缺口只会更大。
        # ⚠️⚠️ **只改路由，不改 verdict**：证据仍是 UNMEASURED，绝不变成 REJECTED，
        # 也绝不仅凭高 g 去插 λ / 拆窗。
        _unreach = []
        for w in _cap_hit:
            gs = w.get("validation_g_checkpoints") or []
            T = w.get("validation_required_frames")
            ncap = w.get("validation_sample_count")
            left = w.get("warmup_steps_left")
            if not gs or not T or not ncap or left is None:
                continue
            r = _ie_reach(
                gs,
                required_decorrelated_frames=int(T),
                cycle_frame_cap=int(ncap),
                budget_remaining_steps=int(left),
                frames_already=int(ncap),
            )
            if r.get("verdict") == "UNREACHABLE":
                _unreach.append((w, r))
        _un_sel = _pick([w for w, _ in _unreach])
        if _un_sel:
            _w, _r = next((w, r) for w, r in _unreach
                          if int(w["window_idx"]) in _un_sel)
            _wi = int(_w["window_idx"])
            _relearned = relearn_epoch_used(
                self.checkpoint_dir, int(view.get("path_version") or 0), _wi)
            _need = int(_r["projected_steps_needed"])
            _why = (
                f"窗口 {_un_sel} 的冻结验证在剩余预算内**算术上不可达**："
                f"g 检查点 {[round(x, 1) for x in _r['g_measurements']]} ⟹ "
                f"保守下界 g_L={_r['g_lower_bound']:.1f}，"
                f"而本周期阈值 gcrit_cycle={_r['gcrit_cycle']:.2f}、"
                f"整预算阈值 gcrit_budget={_r['gcrit_budget']:.2f}。"
                f"凑够 {int(_r['required_decorrelated_frames'])} 个去相关帧需要约 "
                f"{int(_r['projected_raw_frames_needed'])} 原始帧 = {_need} 步，"
                f"剩余仅 {_w.get('warmup_steps_left')} 步。"
                "⚠️ 这是**预算**结论不是 f_k 结论：证据仍是 UNMEASURED，"
                "**不得**改写成 REJECTED，**不得**因此插 λ / 拆窗，"
                "**也不得**因为高 g 就接受一份未验证的 f_k。"
            )
            if not _relearned:
                return plan(
                    "RELEARN_FK_EPOCH",
                    _why + "处置：这个窗口还没用过那**唯一一次**替代候选 ⟹ "
                    "开一个全新 Epoch，从头 LEARN 一份 f_k（不是拿旧生产帧重解）。",
                    exit_="HALT_VALIDATION_BUDGET_UNREACHABLE", windows=_un_sel,
                )
            return plan(
                "DONE",
                _why + "处置：替代候选**已经用过一次**（一个窗口只给一次，"
                "否则就是反复试到偶然通过）⟹ NO_FEASIBLE_ACTION，"
                "留完整诊断终止。",
                exit_="NO_FEASIBLE_ACTION", windows=_un_sel,
            )

        _cap_sel = _pick(_cap_hit)
        if _cap_sel:
            _cap_hit = [w for w in _cap_hit if int(w["window_idx"]) in _cap_sel]
            _idx = _cap_sel
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {_idx} 的单周期验证**批次上限**已打满"
                f"（{[w.get('frozen_validation_batches') for w in _cap_hit]}/{_batch_cap} 批），"
                f"而全局预算**还有钱**（剩 {[w.get('warmup_steps_left') for w in _cap_hit]} 步）"
                " ⟹ 这是 `LOCAL_VALIDATION_CAP_EXHAUSTED`，**不是 HALT_BUDGET、"
                "更不是 F_K_REFUTED**（没有任何证据驳回这份 f_k，只是这一轮没测出来）。"
                "处置：**不扩大 15 批上限**；进 PROVISIONAL_PRODUCTION（**不是可信 PASS**）；"
                "**只给一个 +250k 诊断块**，块后立即用 N_eff/g 的**边际增长**判 —— "
                "达到 10 则继续；边际停滞/下降或 far-end support 单调塌陷则关闭 Epoch 走 "
                "tail rewindow；top1% 灾难性集中则停止同分布加帧。"
                "⚠️ 一块 +250k 预计到 N/g≈9.6、**恰好达不到 10**，这是有意的诊断设计，"
                "**别因此改成 +500k**。"
                "⚠️⚠️ **前提：当前布局不会被重分。** 若尾段即将 tail-only 重分，"
                "这一块就是在**即将作废的布局上取证** —— 应**先重分再取证**，"
                "把预算留给新尾段。",
                exit_="HALT_LOCAL_VALIDATION_CAP", windows=_idx,
            )

        _wb_sel = _pick(with_budget)
        if _wb_sel:
            with_budget = [w for w in with_budget if int(w["window_idx"]) in _wb_sel]
            idxs = _wb_sel
            return plan(
                "CONTINUE_WARMUP",
                f"窗口 {idxs} 仍在 "
                f"{sorted({w['phase'] for w in with_budget})}，且 warmup 预算有余"
                f"（剩 {[w['warmup_steps_left'] for w in with_budget]} 步）。"
                "证据是 INSUFFICIENT_DATA（没测出来），不是 FAIL —— 加同类预算，不换轴。",
                windows=idxs,
                blocked=blocked, earliest=earliest,
            )

        # 4) 预热预算耗尽、仍未收敛 ⟹ 需要归因。而归因要的三个量
        #    （逐态 g 剖面 / mixture coverage / 逐边 δ）本控制器**不算数**，
        #    只在 stage 结果里才有。没有就如实说"归因不出来"。
        stuck = [w for w in warming if not (w["warmup_steps_left"] or 0) > 0]
        _st_sel = _pick(stuck)
        if _st_sel:
            idxs = _st_sel
            if not view["has_stage_result"]:
                return plan(
                    "CONTINUE_WARMUP",
                    f"窗口 {idxs} 的 warmup 预算已耗尽且未收敛，但 stage 结果还没落盘 ⟹ "
                    "归因需要的逐态 g 剖面 / mixture 覆盖度 / 逐边 δ 一个都读不到。"
                    "**归因不出来是合法结局**，不许为了凑一个动作而编理由。"
                    "要继续必须显式升档 warmup 预算，或先把 stage 分析跑完。",
                    # [老板] 只有三种允许终止 ⟹ "归因不出来"本身**不是**终止条件。
                    # 全局还有预算、或还有可行动作时，它是**路由信号**：主循环应该
                    # 去跑最小诊断动作（tail-only 重分 / +250k 诊断块）。
                    # 真的无路可走才 NO_FEASIBLE_ACTION。
                    exit_=(
                        "HALT_NO_ATTRIBUTION"
                        if (not view.get("all_windows_budget_exhausted")
                            or any(v is None for v in feas.values()))
                        else "NO_FEASIBLE_ACTION"
                    ),
                    windows=idxs,
                    missing=["stage2_*.json（逐窗 overlap/ESS 诊断）",
                             "逐态 statistical_inefficiency 剖面",
                             "逐边热力学长度 δ"],
                )
            return plan(
                "INSERT_LAMBDA" if feas.get("insert_lambda") is None else "RECALIBRATE_FK",
                f"窗口 {idxs} 的 warmup 预算已耗尽且未收敛。按 PLAN §3ter.4，窗口级"
                "缺陷的对症动作依次是：重标定 f_k → 仍压不住则补 λ 缩跨度（model B）。"
                + ("" if feas.get("insert_lambda") is None
                   else f" ⚠️ 补 λ 当前不可行：{feas['insert_lambda']}"),
                exit_=None if feas.get("insert_lambda") is None else "HALT_LAMBDA_BUDGET_INSUFFICIENT",
                windows=idxs,
                blocked=blocked, earliest=earliest,
            )

        # 5) 生产帧没攒够 ⟹ 接着跑（同一个 f_k、接着原段，不改任何结构）。
        short = [
            w for w in W
            if w["production_steps_target"] and (w["production_steps"] or 0)
            < int(w["production_steps_target"])
        ]
        _sh_sel = _pick(short)
        if _sh_sel:
            short = [w for w in short if int(w["window_idx"]) in _sh_sel]
            idxs = _sh_sel
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {idxs} 的生产步数未达目标"
                f"（{[(w['production_steps'], w['production_steps_target']) for w in short]}）。"
                "同一个冻结 f_k、接着原段继续，不改结构、不丢已有帧。",
                windows=idxs,
                blocked=blocked, earliest=earliest,
            )

        # 5a) **f_k 明确不符 ⟹ 先重标定，不要先加帧。**
        #     PLAN §4：「I 与 II 之间**不排序**：按判别结果直接选类型，不做『先便宜
        #     后贵』的阶梯（证据表明 f_k 明显不符时先加帧是浪费）」。
        #     加帧治不了偏斜是仓库自己的结论，本轮实测复现（win0 250k→1M：
        #     N_decorrelated 40→182 变好，但 ESS_ratio 0.037→0.0264、
        #     top1% 0.545→0.604 **变差**）。
        #     ⚠️ 这一条**故意排在 5b/6（加帧）之前** —— 生产代码现在的顺序是
        #     "rescue 加帧两轮之后才重标定"，所以影子在这里会与生产分歧，
        #     那正是要被记录下来的对账数据。
        # 0b) **身份不一致**：盘上的 converged 是另一个系综的结论，不得当依据。
        _stale = _pick([w for w in W if w.get("phase") == "IDENTITY_MISMATCH"])
        if _stale:
            return plan(
                "CONTINUE_WARMUP",
                f"窗口 {_stale} 的 `stage_protocol_key` 在 ibs_state 与产物之间不一致 ⟹ "
                "盘上那个 `bias_status=converged` 是**另一个系综**的旧结论，"
                "不得据此声称已在生产。λ / Hamiltonian / box / 规范版本任一变化，"
                "旧 PASS 都不能继续粘住。",
                exit_="HALT_INVALID_INPUT", windows=_stale,
            )

        # 5a-0) **SUSPECTED_DERAILMENT（单块）⟹ 只做非变异的候选计算，不切换。**
        #   老板定案：单块只产生"疑似"，触发廉价诊断；**连续两块**才允许关闭 Epoch。
        #   ⚠️ 而且在 **held-out 反事实验收**接好之前，单块**一律不得**升级成真重标定 ——
        #   否则就是"因一个震荡低块贸然切换到更差的 f_k"（win1 已经吃过这个亏）。
        #   held-out 验收目前**未实现**，所以这一档现在只能到"算候选 + 报告"为止。
        _suspected = _pick([
            w for w in W
            if w.get("derailment_status") == "SUSPECTED_DERAILMENT"
        ])
        _confirmed = [
            w["window_idx"] for w in W
            if w.get("derailment_status") == "CONFIRMED_DERAILMENT"
        ]
        if _suspected and not _confirmed:
            return plan(
                "PROBE_CANDIDATE_FK",
                f"窗口 {_suspected} 单块边际 N_eff 低于前半程中位数 1/4 ⟹ "
                "**SUSPECTED_DERAILMENT**：启动廉价的**非变异**诊断（离线算候选 f_k 并评估），"
                "**不关闭 Epoch、不切换 f_k**。只有连续两块才允许进入动作决策。"
                "⚠️ 并且 held-out 反事实验收（候选必须在未参与拟合的 block 上改善"
                "最差态 N_eff/g）**尚未实现**，所以这一档现在到「算候选 + 报告」为止 —— "
                "在它接通之前，单块预警不得升级成真重标定。",
                windows=_suspected,
                missing=["held-out 反事实验收（候选是否真的改善最差态 N_eff/g）"],
            )

        _probe = view.get("fk_probe") or {}
        _recal = _pick(
            [int(x) for x in (_probe.get("recalibration_recommended_windows") or [])],
            key=None,
        )
        if _recal:
            _broke = self._epoch_validation_unaffordable(view, _recal)
            if _broke:
                return plan(
                    "RUN_PRODUCTION",
                    f"探针判定窗口 {_recal} 该重标定，但**付不起新 Epoch 的最低验证"
                    f"额度**（{_broke}）⟹ **不启动**。没钱验证就别开新 Epoch —— "
                    "那会得到一份永远验不了的 f_k（win2 三次都是这么死的）。",
exit_=(
                    "GLOBAL_BUDGET_EXHAUSTED"
                    if view.get("all_windows_budget_exhausted")
                    # **相关窗口自己**还有钱 ⟹ 路由；它自己没钱但别的窗口有 ⟹
                    # 仍是路由（可以先去跑别的窗口），只有全部为 0 才终止。
                    else "HALT_BUDGET"
                ),
                    windows=_recal,
                    blocked=blocked, earliest=earliest,
                )
            return plan(
                "RECALIBRATE_FK",
                f"f_k 探针判定窗口 {_recal} 的相邻位移超过 "
                f"{_probe.get('min_adjacent_shift_kJ_mol')} kJ/mol ⟹ **先重标定 f_k，"
                "不要先加帧**（PLAN §4：f_k 明显不符时先加帧是浪费；加帧只会往同一个"
                "偏斜分布里加更多帧，绝对样本数涨、ESS 比值不动）。"
                "⚠️ 生产代码当前把重标定挂在 rescue 两轮**之后**"
                "（`stage2_recalibrate_f_k_on_rescue`），所以此处与生产分歧 —— "
                "这是影子要记录的对账点，不是控制器在开车。",
                windows=_recal,
                blocked=blocked, earliest=earliest,
            )

        # 5a-1) **支撑不足时，先判累计 f_k 偏差，不要先加帧。**
        #   老板给的路径第一步就是「win3 支撑不足 → **判断累计 f_k 偏差** → 生成候选」。
        #   而"已经脱轨就禁止继续加帧"也是同一条：加帧治不了 f_k 偏斜
        #   （实测 250k→1M 让 top1% 从 0.545 涨到 0.762）。
        #   证据缺失时的正确动作是**把证据做出来**（跑分析落 cumulative 残差），
        #   不是先烧 250k 去加帧再说。
        _need_cum = [
            w["window_idx"] for w in W
            if earliest is not None and int(w["window_idx"]) == earliest
            and w.get("self_verdict") in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA")
            and w.get("cum_fk_verdict") is None
        ]
        if _need_cum:
            return plan(
                "ANALYZE",
                f"窗口 {_need_cum} 支撑不足，但**累计 f_k 偏差证据还不存在**"
                "（`cumulative_fk_residual_production` 缺失）⟹ 先把它算出来，"
                "**不要先加帧**：加帧治不了 f_k 偏斜（实测 250k→1M 让 top1% 从 "
                "0.545 涨到 0.762、ESS 比值反而变差），而累计偏差判据在**现有帧上**"
                "就能算，零额外采样。",
                windows=_need_cum, blocked=blocked, earliest=earliest,
                missing=["cumulative_fk_residual_production（生产侧累计 f_k 残差）"],
            )

        # 5a-2) **累计 f_k 偏差已判 ⟹ 按 held-out 的三种结局分岔。**
        #   老板给的链：判累计偏差 → 生成候选 → **held-out 可测则验收** →
        #   **不可测则自动开 PROBE_REANCHOR_EPOCH** → 支撑改善则继续 →
        #   **不改善则自动 tail-repartition / 插 λ**。
        #   三种结局的处置**互不相同**，混起来就回到"用支撑换占据"那个老错：
        #     ACCEPT     → 换 Epoch（候选确实在未参与拟合的块上改善了最差态）
        #     UNMEASURED → **不是拒绝**，是"判不了" ⟹ 开有界的 PROBE_REANCHOR_EPOCH
        #                  拿独立证据；**绝不回去给旧 f_k 加帧**
        #     REJECT     → 候选救不了 ⟹ 转布局动作（拆末窗 / 插 λ）
        _cum_bad = [
            w for w in W
            if earliest is not None and int(w["window_idx"]) == earliest
            and w.get("cum_fk_verdict") in ("FAIL_CUMULATIVE_FK", "UNMEASURED")
        ]
        if _cum_bad:
            _w0 = _cum_bad[0]
            _hv = _w0.get("heldout_verdict")
            if _hv == "ACCEPT":
                return plan(
                    "RECALIBRATE_FK",
                    f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}"
                    f"（{_w0.get('cum_fk_verdict')}），且候选在 **held-out 连续时间块**上"
                    f"把最差态 N_eff/g 从 {_w0.get('heldout_worst_before')} 提到 "
                    f"{_w0.get('heldout_worst_after')}、没伤到健康态 ⟹ **换 Epoch**。",
                    windows=[earliest], blocked=blocked, earliest=earliest,
                )
            if _hv == "REJECT":
                _feas_split = feas.get("split_tail_window") is None
                return plan(
                    "SPLIT_TAIL_WINDOW" if _feas_split else "INSERT_LAMBDA",
                    f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}，"
                    "但候选在 held-out 上**没有改善最差态**（或伤到了健康态）⟹ "
                    "**f_k 救不了它，转布局动作**。"
                    + ("拆末窗。" if _feas_split
                       else f"拆窗不可行（{feas.get('split_tail_window')}）⟹ 插 λ 缩跨度。"),
                    windows=[earliest], blocked=blocked, earliest=earliest,
                )
            # None（还没算）或 UNMEASURED（算了但判不了）：都走有界探针。
            return plan(
                "PROBE_REANCHOR_EPOCH",
                f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}"
                f"（{_w0.get('cum_fk_verdict')}），而 held-out 验收"
                + ("尚未进行" if _hv is None else "判不了（UNMEASURED）")
                + " ⟹ 开一个**有界的** PROBE_REANCHOR_EPOCH：候选 f_k + 独立 burn-in"
                " + 一个 +250k 块；新 Epoch 支撑改善才晋升，恶化则拒绝候选、走下一动作。"
                "⚠️ **绝不回去给旧 f_k 加帧** —— 加帧治不了偏斜，那只会把同一个偏斜"
                "分布采得更久。",
                windows=[earliest], blocked=blocked, earliest=earliest,
            )

        # 5b) 窗口自检已经判出"帧数不够" ⟹ 补采。**这是 (6) 的提前版**：同一个量、
        #     同一个门槛，只是在该窗口刚跑完那一刻就知道了，不用等全部窗口跑完
        #     （实测浪费：5×250k=125 万步烧完才做第一次预算判断）。
        #     verdict 是 INSUFFICIENT_DATA（还没测够）⟹ 加预算，不是换 Epoch。
        short_self = _pick([w for w in W if w.get("self_sufficient") is False])
        if short_self:
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {short_self} 的生产后自检判定去相关帧数不足"
                f"（{[(w.get('self_n_frames_decorrelated'), w.get('self_min_frames')) for w in W if w.get('self_sufficient') is False]}）。"
                "这不是终态 —— 语义是「这个窗口还需要加采样」（INSUFFICIENT_DATA ≠ FAIL）。"
                "它们的 checkpoint 此刻还热，补采不作废任何已有帧。",
                windows=short_self,
                blocked=blocked, earliest=earliest,
            )

        # 6) 被踢出协方差链的窗口（去相关后帧数不足）⟹ 补采。
        #    这条路径以前不可达：被跳过的窗口在生成 overlap 诊断**之前**就 continue 了，
        #    于是进不了 rescue 候选名单（P0-2b 修的就是这个）。
        _sk_sel = _pick(view["skipped_windows"], key=None)
        if _sk_sel:
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {_sk_sel} 去相关后有效帧数不足、被踢出协方差链。"
                "先补采（同一分布、接着原段）；这是最便宜且不作废任何已有帧的动作。",
                windows=_sk_sel,
                blocked=blocked, earliest=earliest,
            )

        # 6a) 没有任何 PROBLEM，但有窗口**证据缺失**（UNKNOWN）⟹ 先把证据做出来。
        #     绝不能因为"不知道"就去换 Epoch —— 那是在毫无根据的窗口上烧 GPU。
        if earliest is None and unknown:
            return plan(
                "ANALYZE",
                f"窗口 {unknown} 没有自检证据（`self_verdict` 不存在）⟹ 状态是"
                "**UNKNOWN，不是有问题**。对「不知道」的正确动作是**产出证据**"
                "（跑 stage 分析、落自检产物），不是重标定。"
                "⚠️ 缺证据同样**不等于通过** —— 在证据补齐之前不得判 DONE。",
                windows=unknown, blocked=blocked, earliest=earliest,
                missing=["逐窗自检产物 dual_window_*_self_support.json"],
            )

        # 6b) **前缀全部合格**之后，才轮到"缺窗口"。它此前是全局最高优先级，
        #     会把上游未解决的窗口整个盖住（实测把 win3/win4 盖成 win5）。
        #     缺窗**仍然**是硬约束——缺窗口的和不是 ΔG——只是正确位置在因果顺序之后。
        if view["missing_windows"]:
            _next_missing = min(view["missing_windows"])
            return plan(
                "RUN_PRODUCTION",
                f"前缀窗口全部合格；布局里有 {view['n_windows_expected']} 个窗口、"
                f"产物里只有 {view['n_windows_found']} 个，缺 {view['missing_windows']}。"
                "**缺窗口的和不是 ΔG，禁止当结果使用**（不是「精度差一点」，是**另一个量**）。"
                f"按因果顺序先跑 win{_next_missing}。",
                windows=[_next_missing],
                blocked=blocked, earliest=earliest,
            )

        # 7) stage 已判 converged ⟹ 完成。注意 DONE 的含义。
        if view["stage_converged"] is True:
            return plan(
                "DONE",
                "所有**已定义**的判据都通过了。⚠️ 这不等于答案正确 —— "
                "STAGE2_ROOT_CAUSE_2026-08-28.md §2「所有收敛门对该失效模式失明」"
                "仍然有效（该节被其自身的超越声明明确标为仍然有效）。",
                exit_="DONE_UNTRUSTED" if self.allow_untrusted else "DONE",
                blocked=blocked, earliest=earliest,
            )

        # 8) 窗口都跑完了但 stage 没判过 converged ⟹ 只差分析。
        return plan(
            "DONE" if view["has_stage_result"] else "ANALYZE",
            "所有窗口的预热与生产都已完成"
            + ("，但 stage 判据未通过（converged=%r）——"
               "需要看 stage 结果里的生产质量门。" % (view["stage_converged"],)
               if view["has_stage_result"] else
               "，但 stage 分析还没跑（stage2_*.json 不存在）⟹ 先跑分析。"),
            exit_=None if view["has_stage_result"] else None,
            missing=[] if view["has_stage_result"] else ["stage2_*.json"],
        )

    @staticmethod
    def first_untrusted_window(view: Dict[str, Any]) -> Optional[int]:
        """**第一个不可信窗口**的下标 —— tail-only 重分的起点，由控制器自己算。

        [老板] 「当前从 `first_untrusted_window` 开始重分；**不是让用户手选 anchor**。」
        判据就是主验收量：verdict 不是 `ANALYSIS_ELIGIBLE` 的第一个窗口。
        （本例是 win3，也是第一个 `min(N_eff/g) < 10` 的 —— 人选的 anchor 恰好同解，
        但**机制必须是自动的**。）
        """
        for w in sorted(view.get("windows") or [], key=lambda x: int(x["window_idx"])):
            v = w.get("self_verdict")
            if v is not None and v != "ANALYSIS_ELIGIBLE":
                return int(w["window_idx"])
        return None

    def tail_repartition_anchor(self, view: Dict[str, Any]) -> Optional[float]:
        """`first_untrusted_window` 的**首态** λ —— 它就是与前一窗共享的那个节点。

        冻结的是「anchor 之前」的窗口，所以从**不可信窗口自己的首态**切，
        才能把这个已知弱的窗口一起重分掉；若从它的**末态**切，会把它永久保留。
        """
        idx = self.first_untrusted_window(view)
        if idx is None or idx <= 0:
            return None
        ranges = (view.get("path") or {}).get("window_ranges")
        lam = None
        for w in view.get("windows") or []:
            if int(w["window_idx"]) == idx:
                lam = w.get("lambda_vdw_hi")
        if not ranges or idx >= len(ranges):
            return None
        return float(lam) if lam is not None else None

    @staticmethod
    def _epoch_validation_unaffordable(
        view: Dict[str, Any], windows: Sequence[int]
    ) -> Optional[Dict[str, Any]]:
        """**决定重标定之前，先确认新 Epoch 还付得起最低验证额度。**

        [老板定案 F] 预算要拆两本账：
            global_consumed_budget        永远继承，防止新段白送额度
            epoch_validation_reservation  每个新候选**单独预留**

        「**没钱验证就不启动重标定**」—— 不能启动动作之后才发现新 f_k 没预算验。
        win2 **连续三次**死在这：位移触发把它拖进新 Epoch，新 f_k 要过验证门，
        而账本已经 555000/555000、剩 0，于是循环一次都没进就被标成
        `f_k 不收敛`（第三个变体：预算为零）。

        最低额度用既有的冻结验证阶梯**第一档**，不新发明阈值。
        返回 None = 付得起；否则返回一份说明（调用方据此走 HALT_BUDGET）。
        """
        try:
            import ibs_engine as _ie
            floor = int(_ie.FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0])
        except Exception:
            return None  # 拿不到阈值就不假装判断（fail-open：只影响诊断）
        by_idx = {int(w["window_idx"]): w for w in (view.get("windows") or [])}
        broke = []
        for i in windows:
            w = by_idx.get(int(i))
            if w is None:
                continue
            left = w.get("warmup_steps_left")
            # ⚠️ **预算未知 = 不可行**（fail-closed）。以前 `left is None` 会跳过检查，
            # 于是"账本读不到"的窗口能一路走到重标定 —— 而真实零预算窗口
            # （555k/555k）正是这样漏过去的。不知道有没有钱，就不许开新 Epoch。
            if left is None:
                broke.append({
                    "window_idx": int(i), "warmup_steps_left": None,
                    "min_epoch_validation_reservation": floor,
                    "reason": "budget_unknown_fail_closed",
                })
            elif int(left) < floor:
                broke.append({
                    "window_idx": int(i),
                    "warmup_steps_left": int(left),
                    "min_epoch_validation_reservation": floor,
                })
        if not broke:
            return None
        return {
            "min_epoch_validation_reservation": floor,
            "windows": broke,
            "source": "ibs_engine.FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0]",
        }

    @staticmethod
    def _warmup_weakness_ranking(view: Dict[str, Any]) -> Dict[str, Any]:
        """按 warmup 剖面给窗口**排序**（谁最可能在生产里撑不住）。

        **刻意只排序、不打分、不给阈值。** 三个口径各排一次、不做加权合成 ——
        合成就等于凭空发明一个权重。谁在多个口径上都排在最弱端，才是真信号。

        口径与方向：
          · ``warmup_g``（统计低效率）越大越弱
          · ``warmup_n_frames_used``（gate 实际用上的去相关帧数）越小越弱
          · ``warmup_min_absolute_ess`` 越小越弱
        """
        W = [w for w in view.get("windows") or []]
        out: Dict[str, Any] = {
            "note": ("只排序、不打分、不给阈值；剖面→生产预算的映射未定"
                     "（PLAN P3）。三个口径不做加权合成。"),
            "by_metric": {},
            "weakest_overall": None,
        }
        specs = (("warmup_g", True), ("warmup_n_frames_used", False),
                 ("warmup_min_absolute_ess", False))
        score: Dict[int, int] = {}
        for key, bigger_is_worse in specs:
            vals = [(w["window_idx"], w.get(key)) for w in W if w.get(key) is not None]
            if not vals:
                continue
            vals.sort(key=lambda t: float(t[1]), reverse=bigger_is_worse)
            out["by_metric"][key] = [
                {"window_idx": i, "value": float(v)} for i, v in vals
            ]
            for rank, (i, _v) in enumerate(vals):
                score[i] = score.get(i, 0) + rank
        if score:
            out["weakest_overall"] = [
                i for i, _ in sorted(score.items(), key=lambda t: t[1])
            ]
        return out

    def _evidence_status(self, view, action, exit_) -> str:
        """证据状态跟执行状态是两件事。`allow_untrusted` 不得影响这一维。

        [老板定案 I] **缺窗与救援耗尽必须是 `INSUFFICIENT_DATA`，不是含糊的
        `INCONCLUSIVE`** —— 前者明确说"证据不够、还能补"，后者容易被读成
        "结果差一点"。禁止部分和冒充结果。
        """
        if any(w["verdict"] == "STATISTICALLY_REJECTED" for w in view["windows"]):
            # REJECTED 优先：它是关于**某份 f_k** 的更强、更具体的结论，
            # 即便同时还缺预算。
            return "REJECTED"
        # 预算耗尽**永远**是"还没测够" —— 老板把 HALT_BUDGET 与
        # evidence_status=INSUFFICIENT_DATA 绑定。绝不是 FAIL。
        if exit_ in ("HALT_BUDGET", "HALT_LOCAL_VALIDATION_CAP"):
            return "INSUFFICIENT_DATA"
        if view["missing_windows"] or view["skipped_windows"]:
            return "INSUFFICIENT_DATA"
        # 任一窗口的主验收量（N_eff/g）判不可分析 ⟹ 整条路径的证据就不够。
        if any(
            w.get("self_verdict") in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA")
            for w in view["windows"]
        ):
            return "INSUFFICIENT_DATA"
        if view["stage_converged"] is True:
            return "CONVERGED"
        return "INCONCLUSIVE"

    # ----------------------------------------------------------- Segment 聚合

    @classmethod
    def for_physical_stage(
        cls,
        run_dir: str,
        stage_base: str = "vanishing",
        stage_type: str = "vdw",
        **kwargs: Any,
    ) -> "Stage2RepairController":
        """同一**物理 stage** 的控制器（把 `vanishing` / `vanishing_2` / … 合起来）。

        🔑 [2026-09-11 老板定案] **Segment 不是独立的 stage。**
        先前每个段各起一个控制器各判一次 ⟹ 同一个物理 stage 出两个互相矛盾的动作，
        而且 `vanishing_2` 缺 win0 会被判成"缺窗口"，其实 win0 在段 1 里是好的。

        正确顺序的第一步就是它：

            **聚合同一物理 stage 的全部 Segment**
            → 找 earliest unresolved window
            → 检查该窗口动作是否有预算
            → 按证据选择动作
            → 只有完整前缀全部 eligible，才运行下一个缺失窗口

        本类只在 `read()` 上分叉：聚合视图逐窗取**最新有产物的那个段**的状态
        （段号大的更新），其余（路径版本链、stage 结果、join、f_k 探针）本来就是
        全局的，仍从基准 checkpoint 读。
        """
        obj = cls(run_dir, stage_base, stage_type, **kwargs)
        obj._aggregate_segments = True
        obj._stage_base = str(stage_base)
        return obj

    def _segment_stage_names(self) -> List[str]:
        """同一物理 stage 的所有段目录名，按段号升序（基准段在最前）。"""
        base = getattr(self, "_stage_base", self.stage_name)
        names = []
        for d in glob.glob(os.path.join(self.run_dir, base + "*")):
            if not os.path.isdir(d):
                continue
            nm = os.path.basename(d)
            if nm == base:
                names.append((0, nm))
            else:
                suf = nm[len(base):].lstrip("_")
                if suf.isdigit():
                    names.append((int(suf), nm))
        return [nm for _i, nm in sorted(names)]

    def read_aggregated(self) -> Dict[str, Any]:
        """把同一物理 stage 的全部段合成**一个**视图。

        合并规则（刻意保守）：
          · 逐窗取**段号最大且有 convergence 产物**的那份；都没有就取有任何产物的最新一份。
          · `missing_windows` = 布局里有、但**所有段**都没有的窗口。
            （先前按单段判，于是段 2 缺 win0 被当成缺窗口 —— 其实段 1 有。）
          · 路径版本链 / stage 结果 / join / f_k 探针是全局的，从基准 checkpoint 读。
        """
        names = self._segment_stage_names() or [self.stage_name]
        views = []
        for nm in names:
            sub = Stage2RepairController(
                self.run_dir, nm, self.stage_type,
                min_states_per_window=self.lo, max_states_per_window=self.hi,
                max_path_insertions=self.max_path_insertions,
                allow_untrusted_stage_results=self.allow_untrusted,
            )
            views.append((nm, sub.read()))
        base_view = dict(views[0][1])

        merged: Dict[int, Dict[str, Any]] = {}
        provenance: Dict[int, str] = {}
        for nm, v in views:          # 段号升序 ⟹ 后面的覆盖前面的
            for w in v.get("windows") or []:
                i = int(w["window_idx"])
                if i not in merged or w.get("has_convergence"):
                    merged[i] = dict(w, segment=nm)
                    provenance[i] = nm
        windows = [merged[i] for i in sorted(merged)]

        expected = base_view.get("n_windows_expected")
        missing = (
            [i for i in range(int(expected)) if i not in merged] if expected else []
        )
        skipped = sorted({
            int(x) for _nm, v in views for x in (v.get("skipped_windows") or [])
        })
        base_view.update({
            "stage_name": getattr(self, "_stage_base", self.stage_name),
            "aggregated_segments": names,
            "window_provenance": provenance,
            "windows": windows,
            "n_windows_found": len(windows),
            "missing_windows": missing,
            "skipped_windows": skipped,
            "per_window_budget_remaining": {
                int(w["window_idx"]): int(w.get("warmup_steps_left") or 0)
                for w in windows
            },
            "all_windows_budget_exhausted": bool(
                windows and all(
                    int(w.get("warmup_steps_left") or 0) <= 0 for w in windows
                )
            ),
        })
        return base_view

    def read(self) -> Dict[str, Any]:  # noqa: F811 —— 覆盖：聚合模式走合并视图
        if getattr(self, "_aggregate_segments", False):
            return self.read_aggregated()
        return self._read_single_stage()

    # ------------------------------------------------------- 影子对账 replay

    # 每类证据在产物里的身份：字段名 → 它来自哪、缺了意味着什么。
    # 历史 run **没有**这些新字段，replay 必须如实标 ABSENT_IN_ARTIFACT，
    # **绝不把缺失当通过**。
    _EVIDENCE_KEYS = (
        ("self_verdict", "窗口自检 verdict（N_eff/g 主验收量）"),
        ("min_n_eff_over_g", "主验收量数值"),
        ("derailment_status", "脱轨两档（单块=疑似/连续两块=确认）"),
        ("warmup_g", "warmup 剖面 g"),
        ("warmup_min_absolute_ess", "warmup 剖面 minESS"),
        ("state_protocol_key", "采样身份（旧 PASS 能不能粘住）"),
    )

    def decision_trace(self, view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """一次**只读**的决策轨迹：控制器**会**选什么、依据什么、缺什么证据。

        🔑 [2026-09-11] 影子对账用。**不执行动作、不改预算、不写 run** ——
        它复用同一个 `read()` / `decide()`，所以 replay 和生产**不会各判各的**
        （分开实现就等于对账两个不同的东西，那对账本身就没意义了）。

        老板指定的输出口径：run / window / evidence scope+gauge / verdict /
        reason / selected action / target windows / budget view。
        """
        view = view or self.read()
        plan = self.decide(view)
        windows = []
        for w in view.get("windows") or []:
            avail = {}
            for key, what in self._EVIDENCE_KEYS:
                v = w.get(key)
                avail[key] = {
                    "present": v is not None,
                    "value": v,
                    "what": what,
                    # ⚠️ 缺失**不是**通过。历史产物没有新字段是正常的，但那意味着
                    # "这条证据不存在"，不意味着"这条证据判了通过"。
                    "status": "present" if v is not None else "ABSENT_IN_ARTIFACT",
                }
            windows.append({
                "window_idx": w["window_idx"],
                # 别名：生产侧产物用的是 `window_index`，两个名字并存本身就是坑，
                # 这里两个都给，读的人不必记住哪边用哪个。
                "window_index": w["window_idx"],
                "phase": w.get("phase"),
                "verdict": w.get("self_verdict"),
                "verdict_source": w.get("self_verdict_source"),
                "min_n_eff_over_g": w.get("min_n_eff_over_g"),
                "evidence_availability": avail,
                "budget_view": {
                    "warmup_spent": w.get("warmup_steps_spent"),
                    "warmup_cap": w.get("warmup_steps_cap"),
                    "warmup_left": w.get("warmup_steps_left"),
                    "production_steps": w.get("production_steps"),
                },
            })
        # 老板指定的措辞：「产物里有 5 个」不准确 —— 有 warmup 失败状态但没有
        # production convergence 的窗口是"**已知但未完成**"，不是普通完成产物。
        _known = len(view.get("windows") or [])
        _prod_done = sum(
            1 for w in (view.get("windows") or []) if w.get("has_convergence")
        )
        _never = list(view.get("missing_windows") or [])
        n_absent = sum(
            1 for w in windows for e in w["evidence_availability"].values()
            if not e["present"]
        )
        return {
            "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
            "readonly": True,
            "run_dir": view["run_dir"],
            "stage_name": view["stage_name"],
            # evidence scope + gauge：口径必须跟着轨迹走，否则对账时分不清
            # "两次判得不同"是策略变了还是口径变了。
            "evidence_scope": "on_disk_artifacts_only",
            "evidence_gauge": {
                "fk_residual": "sampling_states",
                "support_metrics": "gauge_invariant",
            },
            "selected_action": plan["action"],
            "exit": plan.get("exit"),
            "terminal": plan.get("terminal"),
            "routing": plan.get("routing"),
            "target_windows": plan.get("windows"),
            "reason": plan.get("reason"),
            "missing_evidence": plan.get("missing_evidence"),
            "execution_status": plan.get("execution_status"),
            "evidence_status": plan.get("evidence_status"),
            "trust_level": plan.get("trust_level"),
            "budget_view": {
                "per_window_remaining": view.get("per_window_budget_remaining"),
                "all_windows_exhausted": view.get("all_windows_budget_exhausted"),
                "source": view.get("global_budget_source"),
            },
            "path": {
                "version": (view.get("path") or {}).get("version"),
                "events": (view.get("path") or {}).get("events"),
                "tail_repartitions_done": view.get("tail_repartitions_done"),
            },
            "known_window_states": _known,
            "production_complete": _prod_done,
            "never_started": _never,
            "earliest_unresolved_window": self.first_untrusted_window(view),
            "blocked_by_upstream": plan.get("blocked_by_upstream"),
            "windows": windows,
            "n_absent_evidence_fields": n_absent,
        }

    @classmethod
    def replay(
        cls,
        run_dirs: Sequence[str],
        stage_names: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """在一批历史 run 上离线重放决策。**只读、零 GPU。**

        缺字段的老产物必须**跑得起来**并如实标记，不能崩、也不能把缺失当通过 ——
        所以每个 run 单独 try/except，失败只记错误、不打断整批。
        """
        out: List[Dict[str, Any]] = []
        for rd in run_dirs:
            # **按物理 stage 重放，不是按段** —— 段是同一个 stage 的多次采样，
            # 分开判会得到两个互相矛盾的动作（实测：段 2 缺 win0 被判成"缺窗口"，
            # 其实 win0 在段 1 里是好的）。
            names = list(stage_names) if stage_names else ["vanishing"]
            for st in names:
                try:
                    out.append(
                        cls.for_physical_stage(rd, st, **kwargs).decision_trace()
                    )
                except Exception as err:  # noqa: BLE001 —— 一个 run 崩不该毁掉整批
                    out.append({
                        "readonly": True, "run_dir": os.path.abspath(rd),
                        "stage_name": st, "error": repr(err),
                        "selected_action": None,
                    })
        return out

    @staticmethod
    def render_replay(traces: Sequence[Dict[str, Any]]) -> str:
        """人读的 mismatch 基线表。"""
        out = ["决策轨迹 replay（**只读**：不执行动作、不改预算、不写 run）", ""]
        hdr = (f"{'run':<26} {'stage':<13} {'action':<20} {'exit':<28} "
               f"{'exec':<12} {'evidence':<18} {'缺证据':>6}")
        out.append(hdr)
        out.append("-" * len(hdr))
        for t in traces:
            if t.get("error"):
                out.append(f"{os.path.basename(t['run_dir']):<26} "
                           f"{t.get('stage_name',''):<13} **ERROR** {t['error'][:60]}")
                continue
            out.append(
                f"{os.path.basename(t['run_dir']):<26} {t['stage_name']:<13} "
                f"{str(t['selected_action']):<20} {str(t.get('exit') or '-'):<28} "
                f"{str(t.get('execution_status')):<12} "
                f"{str(t.get('evidence_status')):<18} "
                f"{t.get('n_absent_evidence_fields', 0):>6}"
            )
        out.append("")
        out.append("⚠️ 「缺证据」= 该 run 的产物里不存在的证据字段数。"
                   "**缺失不是通过** —— 历史 run 没有新字段是正常的，但那意味着"
                   "「这条证据不存在」，对账时必须当成一类 mismatch，不能读成「判了通过」。")
        return "\n".join(out)

    # -------------------------------------------------------------- 人读

    def render(self, view: Optional[Dict[str, Any]] = None) -> str:
        view = view or self.read()
        plan = self.decide(view)
        p = view["path"]
        out: List[str] = []
        out.append(f"run    : {view['run_dir']}")
        out.append(f"stage  : {view['stage_name']} ({view['stage_type']})  "
                   f"lo/hi={view['min_states_per_window']}/{view['max_states_per_window']}"
                   f"  (来自 {view['config_source']})  可拆区间="
                   f"[{2 * view['min_states_per_window'] - 1},"
                   f"{2 * view['max_states_per_window'] - 1}]")
        out.append(f"插点   : 已用 {view['path_insertions_done']}/"
                   f"{view['path_insertions_budget']}  剩 {view['path_insertions_left']}"
                   "（终身预算，从版本链累计，跨 resume 有效）")
        out.append(f"path   : v{p['version']}  n_states={p['n_states']}  "
                   f"events={p['events'] or '{}'}")
        if view["segment_index"] is not None:
            out.append(f"段     : segment_{view['segment_index']}  "
                       f"ibs_state 读自 {view['checkpoint_dir_used']}")
        line = f"windows: 找到 {view['n_windows_found']}"
        if view["n_windows_expected"]:
            line += f" / 布局 {view['n_windows_expected']}"
        if view["missing_windows"]:
            line += f"   ⚠️ 缺 {view['missing_windows']} —— 缺窗口的总和不是完整 ΔG"
        if view["skipped_windows"]:
            line += f"   ⚠️ 被踢出协方差链 {view['skipped_windows']}"
        out.append(line)
        out.append(f"stage  : converged={view['stage_converged']}  "
                   f"ΔG={view['stage_total_delta_G']}  "
                   f"(stage 结果{'已' if view['has_stage_result'] else '**未**'}落盘)")
        out.append("")
        hdr = (f"{'win':>3} {'K':>2} {'λ跨度':>8} {'phase':<16} {'verdict':<22} "
               f"{'warmup':>15} {'验证':>8} {'生产':>9} {'段':>3} {'帧':>6}  note")
        out.append(hdr)
        out.append("-" * len(hdr))
        for w in view["windows"]:
            note = []
            if w["best_effort_acceptance"]:
                note.append(f"best-effort({w['best_effort_reason']})")
            if w["last_gate_error"]:
                note.append(f"gate={w['last_gate_error']}")
            if w["last_failure_reason"]:
                note.append(f"fail={w['last_failure_reason']}")
            if w["has_warmup_failure"] and not w["has_convergence"]:
                note.append("只有 warmup_failure，未产出 convergence")
            if w.get("derail_at_trajectory_fraction") is not None:
                note.append(
                    f"f_k 在 {w['derail_at_trajectory_fraction']:.0%} 轨迹处脱轨"
                    "（边际 N_eff 断崖）"
                )
            if w.get("min_n_eff_over_g") is not None:
                note.append(f"N_eff/g={float(w['min_n_eff_over_g']):.2f}")
            if w.get("self_verdict") and w.get("self_verdict") != "ANALYSIS_ELIGIBLE":
                note.append(
                    f"{w.get('self_verdict')}"
                    f"(N_eff/g={w.get('min_n_eff_over_g')}, "
                    f"n_decorr={w.get('self_n_frames_decorrelated')})"
                )
            wm = "-" if w["warmup_steps_spent"] is None else str(w["warmup_steps_spent"])
            if w["warmup_steps_cap"]:
                wm = f"{wm}/{w['warmup_steps_cap']}"
            span = "-" if w["lambda_span"] is None else f"{w['lambda_span']:.4f}"
            out.append(
                f"{w['window_idx']:>3} {str(w['n_states'] or '-'):>2} {span:>8} "
                f"{w['phase']:<16} {w['verdict']:<22} {wm:>15} "
                f"{('-' if w['frozen_validation_steps'] is None else w['frozen_validation_steps']):>8} "
                f"{('-' if w['production_steps'] is None else w['production_steps']):>9} "
                f"{str(w['n_production_segments'] or '-'):>3} "
                f"{('-' if w['n_frames'] is None else w['n_frames']):>6}  " + "; ".join(note)
            )
        _fp = view.get("fk_probe") or {}
        if _fp:
            out.append("")
            out.append(
                f"f_k 探针 : verdict={_fp.get('verdict')} "
                f"建议重标定={_fp.get('recalibration_recommended_windows')} "
                f"（阈值 {_fp.get('min_adjacent_shift_kJ_mol')} kJ/mol 相邻位移）"
            )
            for r in (_fp.get("windows") or []):
                if r.get("max_adjacent_shift_kJ_mol") is None:
                    out.append(f"           w{r.get('window')}: 跳过 {r.get('skipped')}")
                else:
                    out.append(
                        f"           w{r.get('window')}: 相邻位移="
                        f"{float(r['max_adjacent_shift_kJ_mol']):.3f} "
                        f"绝对位移={float(r.get('max_abs_shift_kJ_mol') or 0):.3f} "
                        f"n_used={r.get('n_frames_used')} "
                        f"规范对账sd={r.get('f_k_consistency_sd_kJ_mol')}"
                    )
        _rank = plan.get("warmup_weakness_ranking") or {}
        if _rank.get("by_metric"):
            out.append("")
            out.append("warmup 剖面（**只报告**：哪个窗口在生产前就显出弱相）"
                       " —— 剖面→预算的映射未定，不据此做任何事")
            wh = (f"{'win':>3} {'g':>8} {'n_used':>7} {'minESS':>8} "
                  f"{'max_adj':>8} {'门槛':>6}")
            out.append(wh)
            out.append("-" * len(wh))
            for w in view["windows"]:
                f3 = lambda x, wd, pr=2: ("-".rjust(wd) if x is None
                                          else f"{float(x):.{pr}f}".rjust(wd))
                out.append(
                    f"{w['window_idx']:>3} {f3(w.get('warmup_g'), 8)} "
                    f"{str(w.get('warmup_n_frames_used') or '-'):>7} "
                    f"{f3(w.get('warmup_min_absolute_ess'), 8)} "
                    f"{f3(w.get('warmup_max_adjacent_delta_kJ_mol'), 8, 3)} "
                    f"{f3(w.get('warmup_gate_threshold_kJ_mol'), 6, 1)}"
                )
            out.append(f"弱→强（三口径名次和，不加权）: {_rank.get('weakest_overall')}")
        if view.get("joins"):
            out.append("")
            out.append("join λ 两侧支撑（相邻窗口对共享的那一个 λ 各自的 rawESS）"
                       " —— **只报告，不参与放行**")
            out.append("           ⚠️ 只证明**共享态自身**采得好；缺跨窗交叉能量 ⟹ "
                       "下游强支撑进不了上游 local MBAR，**不构成上游 PASS**")
            jh = (f"{'join λ':>8} {'上游':>5} {'rawESS':>9} {'g':>7} {'top1%':>6}  "
                  f"{'下游':>5} {'rawESS':>9} {'g':>7} {'top1%':>6}  {'下/上':>7}  判读")
            out.append(jh)
            out.append("-" * len(jh))
            for j in view["joins"]:
                u, d = j.get("upstream") or {}, j.get("downstream") or {}
                f2 = lambda x, w, p=1: ("-".rjust(w) if x is None
                                        else f"{float(x):.{p}f}".rjust(w))
                r = j.get("downstream_over_upstream_raw_ess")
                # ⚠️ 判读只写**事实**，绝不写"可以兜底/可以参考下游"。
                # 老板纠正：只保存了共享的那一个 λ、没有跨窗交叉能量 ⟹ 下游的强支撑
                # **进不了上游的 local MBAR**，改善不了上游的相对自由能。所以
                # 「上游端点弱、下游强」**不构成**上游可以 PASS —— 上游保持
                # PENDING / INSUFFICIENT。（这句以前写成"端点值可参考下游"，是误用。）
                verdict = "-"
                if r is not None:
                    verdict = ("上游端点弱、下游强 %.1f×（仅诊断：**不**构成上游放行）" % r
                               if r > 2.0 else
                               "下游端点弱、上游强 %.1f×（仅诊断）" % (1.0 / r) if r < 0.5 else
                               "两侧相当（仅诊断）")
                out.append(
                    f"{f2(j.get('join_lambda_vdw'), 8, 4)} "
                    f"{('w%d' % j.get('upstream_window', -1)):>5} "
                    f"{f2(u.get('raw_ess'), 9)} {f2(u.get('tau_int'), 7, 2)} "
                    f"{f2(u.get('top1pct_weight'), 6, 3)}  "
                    f"{('w%d' % j.get('downstream_window', -1)):>5} "
                    f"{f2(d.get('raw_ess'), 9)} {f2(d.get('tau_int'), 7, 2)} "
                    f"{f2(d.get('top1pct_weight'), 6, 3)}  "
                    f"{f2(r, 7, 2)}  {verdict}"
                )
        out.append("")
        out.append("=" * 78)
        out.append(f"下一步动作 : {plan['action']}" + (f"   出口: {plan['exit']}" if plan["exit"] else ""))
        out.append(f"涉及窗口   : {plan['windows'] or '-'}")
        out.append(f"理由       : {plan['reason']}")
        if plan["missing_evidence"]:
            out.append(f"缺什么证据 : {plan['missing_evidence']}")
        out.append(f"结构动作   : 可行={plan['feasible_structural_actions']}")
        for k, v in plan["infeasible_structural_actions"].items():
            out.append(f"             {k} 不可行: {v}")
        out.append(f"三维状态   : execution={plan['execution_status']}  "
                   f"evidence={plan['evidence_status']}  trust={plan['trust_level']}")
        return "\n".join(out)


TAIL_REPARTITION_PROTOCOL_VERSION = 1


def repartition_tail_from_anchor(
    lambdas_var: Sequence[float],
    window_ranges: Sequence[Sequence[int]],
    anchor: float,
    *,
    min_states_per_window: int,
    max_states_per_window: int,
    anchor_atol: float = 1e-9,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """**只重分尾段**：anchor 之前的窗口逐字冻结，只有 anchor 起的重新分配。

    设计依据：``docs/PLAN_PATH_REPAIR_2026-09-11.md``。老板的原话是
    「若当前实现要求全部从头重跑，那是路径版本/reuse 规则仍过粗，**不是物理上的
    必要代价**」。

    **为什么原来做不到**（查过，归因不在指纹）：
      · ``stage2_window_max_states`` 在 ``_PREOPT_DERIVED_PATH_KEYS`` 里，属于
        **第 2 层（派生路径）**；两层拆分的设计意图本来就是「第 1 层若匹配，
        **仍可离线重算**」—— 指纹层不是阻塞点，改它**不需要重跑 pilot**。
      · 真正作废前缀的是：窗口产物的复用键是**每个窗口自己的 λ 集合**
        （``_window_lambda_key`` / ``_resume_cached_window_gate_status`` 的
        ``np.allclose``），而改 max_states 会走**全局**分窗器
        ``vanishing_subdomain_ranges_from_lambdas`` ⟹ **所有**窗口边界重排 ⟹
        连健康的前缀也对不上。
    ⟹ 缺的就是本函数：一个**只动尾段**的分窗操作。

    老板给的验收条件，逐条在下面用断言钉住：
      1. 不重新跑 pilot、**不改变 λ 表**（本函数只返回 ranges，λ 原样带回并断言相等）
      2. anchor 必须是**现有共享态**
      3. anchor 以前的 ranges **逐字相等**
      4. 仅 anchor 起的尾段重新分配
      5. **完整覆盖、无缺口**，相邻窗口恰好共享一个态
      6. **确定性、幂等**（同输入同输出；对自身输出再跑一次不变）
      7. 输出明确的 ``changed_windows`` 与**复用清单**
      8. **只作废 anchor 以后的产物**
      9. execution 层的 K + 难度检查 → 见 ``feasible_repair_actions(layer="execution")``
     10. **先 shadow/preview，不自动启动 GPU** → 本函数是**纯函数**，不落盘、不采样

    ``anchor`` 是 λ 值（不是下标）—— 调用方看到的是 λ，下标随布局变。
    """
    lam = [float(x) for x in lambdas_var]
    ranges = [(int(a), int(b)) for a, b in window_ranges]
    lo_n, hi_n = int(min_states_per_window), int(max_states_per_window)
    if ranges != sorted(ranges):
        raise ValueError(f"window_ranges 必须按起点升序: {ranges}")
    if not ranges or ranges[-1][1] != len(lam):
        raise ValueError(f"末窗 {ranges[-1] if ranges else None} 必须覆盖到最后一个态")

    # ---- 验收 2：anchor 必须是现有的**共享**态 ----
    shared_pre = sorted({b - 1 for _a, b in ranges[:-1]} & {a for a, _b in ranges[1:]})
    cand = [i for i, v in enumerate(lam) if abs(v - float(anchor)) <= float(anchor_atol)]
    if len(cand) != 1:
        # 调用方最容易踩的是"传了四舍五入过的 λ" ⟹ 匹配 0 个。把**合法 anchor 全列出来**，
        # 免得还要自己去翻 λ 表数小数位。
        raise ValueError(
            f"anchor λ={anchor!r} 在 λ 表里匹配到 {len(cand)} 个态（需恰好 1 个，"
            f"容差 {anchor_atol}）。合法的 anchor（现有共享态）是："
            + ", ".join(f"win{i+1}起点 λ={lam[j]!r}" for i, j in enumerate(shared_pre))
        )
    a_idx = cand[0]
    shared = {b - 1 for _a, b in ranges[:-1]} & {a for a, _b in ranges[1:]}
    if a_idx not in shared:
        raise ValueError(
            f"anchor λ={anchor}（下标 {a_idx}）不是现有共享态；"
            f"现有共享态下标={sorted(shared)}。只允许在共享节点上冻结。"
        )

    prefix = [(a, b) for a, b in ranges if b - 1 <= a_idx]
    if not prefix or prefix[-1][1] - 1 != a_idx:
        raise ValueError(
            f"anchor 下标 {a_idx} 不是任何窗口的末态；前缀={prefix}"
        )

    # ---- 尾段重分：复用既有分窗器，只喂给它 anchor 起的那段 λ ----
    tail_lam = np.asarray(lam[a_idx:], dtype=float)
    if tail_lam.size < lo_n:
        raise RuntimeError(
            f"anchor 之后只有 {tail_lam.size} 个态，不足 min_states_per_window={lo_n}"
        )
    tail_ranges_local = [
        (int(x), int(y)) for x, y in vanishing_subdomain_ranges_from_lambdas(
            tail_lam,
            min_states_per_window=lo_n,
            max_states_per_window=hi_n,
        )
    ]
    tail_ranges = [(x + a_idx, y + a_idx) for x, y in tail_ranges_local]
    new_ranges = list(prefix) + tail_ranges

    # ---- 验收 1/3/4/5：λ 未变、前缀逐字相等、覆盖完整、单一共享边界 ----
    if [float(x) for x in lambdas_var] != lam:
        raise RuntimeError("λ 表被改动了 —— 本函数只重分窗口，不动 λ")
    if new_ranges[: len(prefix)] != list(prefix):
        raise RuntimeError("前缀 ranges 不再逐字相等")
    validate_single_shared_boundary_ranges(new_ranges, len(lam))
    # 🔑 ``validate_single_shared_boundary_ranges`` **只**校验覆盖与单一共享边界，
    # **不**校验 [lo, hi]。少了这一条就可能返回一个"结构合法但每窗超限"的尾段
    # （验证时实测踩到：尾段被并成 K=10 而 hi=8，居然一路通过）。
    # ⚠️ 只校验**新产生的尾窗**：前缀是冻结带回来的历史布局，它可能是在别的
    # min/max 下定的，不该因为本次参数变了就被判非法。
    for a, b in tail_ranges:
        if not (lo_n <= b - a <= hi_n):
            raise RuntimeError(
                f"重分后的尾窗 {(a, b)} 有 {b - a} 个态，越出 [{lo_n},{hi_n}]；"
                "分窗器给了非法布局，拒绝返回。"
            )

    # ---- 验收 7/8：changed / reusable，按**λ 集合**判（复用真正的键就是它）----
    def _key(rng):
        a, b = rng
        return tuple(round(float(x), 12) for x in lam[a:b])

    old_keys = {i: _key(r) for i, r in enumerate(ranges)}
    new_keys = {i: _key(r) for i, r in enumerate(new_ranges)}
    reusable, changed = [], []
    for ni, nk in new_keys.items():
        hit = next((oi for oi, ok in old_keys.items() if ok == nk), None)
        (reusable if hit is not None else changed).append(
            {"new_window": ni, "old_window": hit, "n_states": len(nk)}
        )
    invalidated = sorted(
        oi for oi, ok in old_keys.items() if ok not in set(new_keys.values())
    )
    if any(i <= len(prefix) - 1 for i in invalidated):
        raise RuntimeError(
            f"前缀窗口被作废了（{invalidated}），违反「只作废 anchor 以后产物」"
        )

    return new_ranges, {
        "protocol_version": TAIL_REPARTITION_PROTOCOL_VERSION,
        "source": "tail_only_repartition_from_shared_anchor",
        "anchor_lambda_vdw": float(lam[a_idx]),
        "anchor_state_index": int(a_idx),
        "frozen_prefix_windows": [list(r) for r in prefix],
        "old_ranges": [list(r) for r in ranges],
        "new_ranges": [list(r) for r in new_ranges],
        "n_windows_before": len(ranges),
        "n_windows_after": len(new_ranges),
        "reusable_windows": reusable,
        "changed_windows": changed,
        "invalidated_old_windows": invalidated,
        "min_states_per_window": lo_n,
        "max_states_per_window": hi_n,
        "lambda_table_unchanged": True,
        "note": (
            "纯函数：不落盘、不采样、不启动 GPU。执行前还要过 "
            "feasible_repair_actions(layer='execution')，它同时检查 K 与物理难度，"
            "**缺难度证据时 fail-closed**。"
        ),
    }


def record_tail_repartition_version(
    checkpoint_dir: str,
    lambdas_var: Sequence[float],
    new_ranges: Sequence[Sequence[int]],
    diag: Dict[str, Any],
    *,
    first_untrusted_window: Optional[int] = None,
    reason: str = "tail_only_repartition_from_first_untrusted_window",
) -> Dict[str, Any]:
    """把一次尾段重分登记成**路径版本链事件**。这是 execute 通路的**前置**。

    🔑 [2026-09-11] 为什么它不能等 J 一起做：没有版本链事件，**崩溃恢复分不清
    「已经重分过没有」** —— 第二次启动会再重分一次，把刚跑完的新尾段又作废。
    老板的话：「不能等 win4/win5 再次失败后才补。」

    幂等由 `lambda_path_versions.append_version` 的确定性 event_id 保证：
    同一个 (λ表, ranges, kind, detail) 重复登记不会产生第二个版本。

    ⚠️ 本函数**只登记，不采样**。λ 表原样带回（尾段重分**不改 λ**）。
    """
    import lambda_path_versions as _lpv

    lam = [float(x) for x in lambdas_var]
    detail = {
        "anchor_lambda_vdw": diag.get("anchor_lambda_vdw"),
        "anchor_state_index": diag.get("anchor_state_index"),
        "first_untrusted_window": (
            None if first_untrusted_window is None else int(first_untrusted_window)
        ),
        "frozen_prefix_windows": diag.get("frozen_prefix_windows"),
        "old_ranges": diag.get("old_ranges"),
        "new_ranges": diag.get("new_ranges"),
        "changed_windows": diag.get("changed_windows"),
        "reused_windows": diag.get("reusable_windows"),
        "invalidated_old_windows": diag.get("invalidated_old_windows"),
        "n_windows_before": diag.get("n_windows_before"),
        "n_windows_after": diag.get("n_windows_after"),
        "min_states_per_window": diag.get("min_states_per_window"),
        "max_states_per_window": diag.get("max_states_per_window"),
        # 尾段重分**不动 λ 表** —— 登记这一条，好让读版本链的人不必去 diff λ。
        "lambda_table_unchanged": True,
    }
    return _lpv.append_version(
        checkpoint_dir,
        [0.0] * len(lam), lam, [list(r) for r in new_ranges],
        kind="tail_repartition",
        reason=reason,
        detail=detail,
    )


def render_tail_repartition_preview(diag: Dict[str, Any], lambdas_var: Sequence[float]) -> str:
    """人读的 preview。**先看这个，再决定要不要跑 GPU。**"""
    lam = [float(x) for x in lambdas_var]
    out = [
        "尾段重分预览（**纯计算，未启动任何采样**）",
        f"  anchor : λ={diag['anchor_lambda_vdw']:.4f}（下标 {diag['anchor_state_index']}，现有共享态）",
        f"  λ 表   : {len(lam)} 态，**未改动**",
        f"  窗口数 : {diag['n_windows_before']} → {diag['n_windows_after']}",
        "",
        f"  {'窗口':>6} {'范围':>12} {'K':>3} {'λ 区间':>20}  处置",
        "  " + "-" * 62,
    ]
    old = [tuple(r) for r in diag["old_ranges"]]
    reuse_new = {d["new_window"]: d["old_window"] for d in diag["reusable_windows"]}
    for i, r in enumerate(diag["new_ranges"]):
        a, b = int(r[0]), int(r[1])
        oi = reuse_new.get(i)
        tag = (f"**复用**旧 win{oi}（λ 逐位相同）" if oi is not None
               else "**重采**（新 λ 集合）")
        out.append(
            f"  {('win%d' % i):>6} {str((a, b)):>12} {b - a:>3} "
            f"{f'{lam[a]:.4f}→{lam[b-1]:.4f}':>20}  {tag}"
        )
    out.append("")
    out.append(f"  作废的旧窗口 : {diag['invalidated_old_windows']}（anchor 之前的一个都没有）")
    out.append(f"  冻结的前缀   : {diag['frozen_prefix_windows']}")
    out.append("  ⚠️ 执行前必须过 feasible_repair_actions(layer='execution')："
               "同时检查 K 与物理难度，缺难度证据 fail-closed。")
    return "\n".join(out)


REPAIR_ACTIONS = ("insert_lambda", "split_tail_window")


def feasible_repair_actions(
    window_ranges: Sequence[Sequence[int]],
    n_states: int,
    *,
    min_states_per_window: int,
    max_states_per_window: int,
    tail_exempt_from_max: bool = True,
    n_insert: int = 1,
    layer: str = "path_version",
    tail_difficulty_ok: Optional[bool] = None,
) -> Dict[str, Optional[str]]:
    """哪些**结构性**修补动作在当前布局下可行。纯计算，零 GPU。

    设计依据：``docs/PLAN_PATH_REPAIR_2026-09-11.md`` §2「可行性规则」。

    返回 ``{动作名: None 表示可行 | str 表示不可行的理由}``。
    调用方取可行集：``[k for k, v in d.items() if v is None]``。

    **为什么要单独一个纯函数**：动作"不可行"必须以"不在候选集里"的形式呈现给
    控制器，而不是以"调用下去抛异常"的形式。后者会让控制器看不见自己少了一档，
    于是被迫用另一类动作去回答（这就是历史上"覆盖不足 → 插 λ"那个 type error
    的机制）。

    ``layer``：**豁免分两层，别混**（2026-09-11 老板定案）::

        "path_version"  路径版本层：末窗**可以**暂时充当溢出槽（豁免 max）
        "execution"     采样执行层：任何**实际要跑**的窗口都必须满足与普通窗口
                        相同的物理难度约束 —— 末窗不例外

    ⚠️ 执行层**不能只检查 ``K <= hi``**。实测反例：末窗 ``K=4`` 已经满足 max，
    却仍然是全场最难的那个（预热就停了、g 22.8→143.3）。真正缺的是
    ``max_states`` **加上** 热力学/动力学**难度上限**（Fisher / metric integral /
    预测 support / 已有的 ``N_eff/g`` 剖面）。所以执行层要求调用方通过
    ``tail_difficulty_ok`` 显式给出难度判定；**没给就 fail-closed**，
    而不是拿 ``K <= hi`` 冒充"可执行"。
    末窗超过难度上限时，应在**采样前**把 overflow materialize 成两个尾窗。

    ``tail_exempt_from_max``：末窗是 model B 的溢出槽，**在被拆过之前豁免
    ``max_states_per_window``**（仅路径版本层）。拆过之后两个孩子都受约束，溢出槽消失 ——
    此后插点无处可去，唯一出口是报「输入 λ 数目不够」并退出进程（控制器
    **不得**自行加总态数重跑 preopt：总态数是输入，属于 prescribed path 的定义）。
    调用方应从路径版本链里数 ``split_tail_window`` 事件来决定这个标志。

    只覆盖**结构性**动作。"延长采样 / 重标定 f_k"不改结构、永远可行，不在此列。
    """
    ranges = [(int(a), int(b)) for a, b in window_ranges]
    if not ranges:
        raise ValueError("window_ranges 不能为空")
    if ranges != sorted(ranges):
        raise ValueError(f"window_ranges 必须按起点升序: {ranges}")
    lo_n = int(min_states_per_window)
    hi_n = int(max_states_per_window)
    n_ins = int(n_insert)
    if n_ins < 1:
        raise ValueError(f"n_insert 必须 >= 1，收到 {n_insert}")
    if int(ranges[-1][1]) != int(n_states):
        raise ValueError(
            f"末窗 {ranges[-1]} 必须覆盖到最后一个态（n_states={n_states}）"
        )

    if layer not in ("path_version", "execution"):
        raise ValueError(f"未知 layer {layer!r}；只接受 path_version / execution")
    tail_lo, tail_hi = ranges[-1]
    k_tail = tail_hi - tail_lo
    out: Dict[str, Optional[str]] = {}

    # ---- 执行层：末窗**取消**豁免，且必须另有难度判据 ----
    # [老板] 「execution 层取消末窗豁免；overflow 只允许作为**内部路径记账状态**，
    #        不能直接拿去采样。」
    if layer == "execution":
        # 执行层一律不豁免，调用方传什么都不算数。
        tail_exempt_from_max = False
        if k_tail > hi_n:
            out["run_tail_window"] = (
                f"执行层不豁免上限：末窗 K={k_tail} > max_states_per_window={hi_n}。"
                "采样前必须先把 overflow materialize 成两个尾窗。"
            )
        elif tail_difficulty_ok is None:
            # **不拿 K<=hi 冒充可执行。** 实测 K=4 的末窗满足 max 却仍是最难的那个。
            out["run_tail_window"] = (
                f"末窗 K={k_tail} 满足 max_states={hi_n}，但**没有提供难度判据**"
                "（tail_difficulty_ok=None）⟹ fail-closed。K<=max 不足以说明可执行："
                "实测 K=4 的末窗满足 max、却是全场最难的那个（预热就停了，"
                "g 22.8→143.3）。执行层需要 max_states **加上** 热力学/动力学难度上限"
                "（Fisher / metric integral / 预测 support / N_eff/g 剖面）。"
            )
        elif not tail_difficulty_ok:
            out["run_tail_window"] = (
                f"末窗 K={k_tail} 满足 max_states，但**超过难度上限** ⟹ "
                "采样前把 overflow materialize 成两个尾窗。"
            )
        else:
            out["run_tail_window"] = None

    # ---- 插 λ（model B）：ranges 不动、末窗吸收 n 个溢出态 ----
    tail_k_split_max = 2 * hi_n - 1
    if tail_exempt_from_max and k_tail + n_ins > tail_k_split_max:
        # 豁免上限 ≠ 可以无限长：超过 2·hi−1 之后末窗永远拆不开，而拆它正是
        # 唯一的收尾动作。见 insert_lambda_in_failed_ibs_window 里同名守卫。
        out["insert_lambda"] = (
            f"末窗再吸收 {n_ins} 个态会到 K={k_tail + n_ins}，超过可拆上限 "
            f"2·max−1={tail_k_split_max} ⟹ 此后永远切不出合法子窗，溢出槽变死胡同。"
            f"先拆末窗（K 现在 {k_tail}，可拆区间 {2 * lo_n - 1}..{tail_k_split_max}），"
            "或判 HALT_LAMBDA_BUDGET_INSUFFICIENT。"
        )
    elif tail_exempt_from_max:
        out["insert_lambda"] = None
    elif k_tail + n_ins <= hi_n:
        out["insert_lambda"] = None
    else:
        out["insert_lambda"] = (
            f"末窗已被拆过（不再豁免 max_states_per_window={hi_n}），"
            f"当前 K_tail={k_tail}，再吸收 {n_ins} 个态会到 {k_tail + n_ins} > {hi_n}"
            " —— 溢出槽已耗尽。这不是「再想个办法」的问题：**输入的 λ 总数从一开始"
            "就不够**。应退出进程、由人工改输入文件的 λ 数目后重跑"
            "（HALT_LAMBDA_BUDGET_INSUFFICIENT）。"
        )

    # ---- 拆末窗：两子窗共享一个边界态 ⟹ p + q − 1 = K，两侧都要落在 [lo, hi] ----
    split_ok = any(
        p + q - 1 == k_tail and lo_n <= p <= hi_n and lo_n <= q <= hi_n
        for p in range(lo_n, hi_n + 1)
        for q in range(lo_n, hi_n + 1)
    )
    if split_ok:
        out["split_tail_window"] = None
    else:
        out["split_tail_window"] = (
            f"末窗 K_tail={k_tail} 在 [{lo_n},{hi_n}] 下切不出两个合法子窗"
            f"（需 p+q−1={k_tail} 且两侧都在区间内 ⟹ 最小可拆 K = {2 * lo_n - 1}）。"
            "继续插 λ 让末窗长大即可到达；拆窗只是末窗溢出压不住时的收尾动作。"
        )
    return out


def insert_lambda_in_failed_ibs_window(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    failed_range: Tuple[int, int],
    pilot_lambdas: List[float],
    pilot_cumulative_length: List[float],
    *,
    min_states_per_window: int = 4,
    max_states_per_window: int = 5,
    partition_criterion: str = "arclength",
    pilot_metric_g: Optional[Sequence[float]] = None,
    n_insert: Optional[int] = None,
) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
    """在失败窗口内插 λ，**缩小该窗口的 λ 跨度**；多出来的态由**末窗**吸收。

    ⚠️ 2026-09-11 重写。设计依据：``docs/PLAN_PATH_REPAIR_2026-09-11.md`` §2 更正
    与 §3ter。两条关键改动，改前先读那份文档。

    **记账口径 = model B：窗口的下标区间一律不动，只有末窗上界 += n。**

    插一个 λ 之后，窗口区间"跟不跟着走"是两套记账，物理效果相反::

        原始           win0 = (0,4) 装 {λ0,λ1,λ2,λ3}      跨度 λ0→λ3
        model A        win0 = (0,5) 装 {λ0,λ1,λnew,λ2,λ3} 跨度 λ0→λ3 **不变**
        model B(本函数) win0 = (0,4) 装 {λ0,λ1,λnew,λ2}    跨度 λ0→λ2 **缩了**

    ``ΔF(λ0→λ3)`` 是物理量，与中间放几个点无关 —— 所以 **model A 插点之后 bias
    要压平的总落差一分没少，对"窗口太宽"是无效动作**。只有 model B 真的缩小了
    窗口要跨的自由能落差，这才是"补 λ 治窗口太宽"的机制。

    代价与安全性：

      · **前缀（插入点之前的窗口）λ 逐位不变** —— 已采完的窗口全部复用，后置断言保证。
      · 下游窗口的 λ 内容左移一格。窗口是按 0,1,2… 顺序跑的，win_i 失败时
        i+1..N **还没采**，所以顺序跑时这是零成本。
      · resume 安全：``lambda_path_versions.append_version`` 同时记 λ 表与
        window_ranges，``resolve_path`` 取回当前版本，``changed_windows()`` 给
        可复用清单，插点事件带确定性 ID 防重复插。
      · **末窗吸收溢出，因此豁免 ``max_states_per_window``。** 非末窗的态数永远
        是布局期定的那个值（本函数不改，后置断言保证）。

    **本函数不拆窗。** 拆窗只发生在末窗、只由"末窗 K 累积到 f_k 重标定后仍压不住"
    触发，是另一条路径的事。旧实现在这里无条件就地拆窗，而且为了凑拆窗门槛
    ``n = max(1, (2*lo−1) − K)`` 强行插点 —— 那些 λ 没有任何边级证据支持，纯粹是
    成本（多几个态要采）。插点是**边级**工具，不该为窗口级簿记门槛服务。

    ``n_insert``：

      · 给了就用它（调用方按窗口级判据决定插几个，这是控制器的职责）。
      · 不给则走**边级**判据：``n = Σ_边 max(0, ⌈L_edge / L_target⌉ − 1)``，
        ``L_target`` = 全路径平均边长。窗内没有任何边超过 L_target 时**明着报错**，
        不静默插 1 个 —— "边级证据不支持插点"是一个有意义的结论。

    插点**位置** = 窗内最长边在**配置度量**下的中点（pilot 反插值）。
    ``partition_criterion`` 贯穿"定 n / 选边 / 定位"三处：初始布局按哪个度量
    定的布点，补救就按哪个 —— 旧实现里它只影响拆窗切点，拆窗一去掉就成了
    无效参数，而"缺 metric_g 就 fail-closed"那条守卫正是为了防止这种悄悄换判据。

    ⚠️ 措辞：这是**按 pilot 实测长度插值得到的候选加密位置**，不是本次 VALIDATE
    已证实的故障边。调用方应把它标成"学习失败后的补救尝试"。
    """
    lambdas = [float(x) for x in lambdas_var]
    ranges = [(int(a), int(b)) for a, b in window_ranges]
    failed = (int(failed_range[0]), int(failed_range[1]))
    if failed not in ranges:
        raise RuntimeError(f"失败窗口 {failed} 不在当前窗口列表 {ranges} 中")
    if ranges != sorted(ranges):
        raise RuntimeError(f"window_ranges 必须按起点升序（model B 依赖末窗是最后一项）: {ranges}")
    if ranges[-1][1] != len(lambdas):
        raise RuntimeError(
            f"末窗 {ranges[-1]} 必须覆盖到最后一个态（n_states={len(lambdas)}）"
        )
    lo_n, hi_n = int(min_states_per_window), int(max_states_per_window)
    if failed[1] - failed[0] < 2:
        raise RuntimeError(f"失败窗口 {failed} 不足两个态，无法在其中插点")

    pilot_lam = np.asarray(pilot_lambdas, dtype=float).ravel()
    pilot_s = np.asarray(pilot_cumulative_length, dtype=float).ravel()
    if pilot_lam.size != pilot_s.size or pilot_lam.size < 2:
        raise ValueError("pilot lambda 与累计热力学长度必须等长且至少含两个点")
    if not np.all(np.diff(pilot_s) > 0.0):
        raise ValueError("pilot 累计热力学长度必须严格递增")

    criterion = str(partition_criterion).lower()
    if criterion not in ("arclength", "metric_integral"):
        raise ValueError(f"未知分窗判据 {partition_criterion!r}")
    if criterion == "metric_integral" and pilot_metric_g is None:
        # 初始布局按 ∫g 定的度量，补救却按 ∫√g 定 n，等于第一次自动补救就换了判据。
        # 拿不到 metric_g 就明着拒绝，不静默退回等弧长。
        raise ValueError(
            "partition_criterion='metric_integral' 必须同时提供 pilot_metric_g；"
            "拒绝静默退回等弧长——那会让补救悄悄换掉度量。"
        )

    start, end = failed
    is_last = (failed == ranges[-1])
    original_span = abs(lambdas[end - 1] - lambdas[start])
    n_states_before = len(lambdas)
    w_before = len(ranges)

    # 🔑 **判据必须贯穿到底。** 旧实现里 partition_criterion 只影响拆窗切点，
    # 选边与定位一律按弧长；拆窗一去掉，判据就变成完全无效的参数了 —— 而那条
    # "缺 metric_g 就 fail-closed"的守卫存在的全部意义就是防止补救悄悄换判据。
    # 所以这里把 pilot 的累计度量换成配置的那一个，选边（比较）和定位（反插值）
    # 都用它。两个方向都只是对 (pilot_lam, pilot_measure) 做 np.interp。
    if criterion == "metric_integral":
        pilot_measure = np.asarray(
            metric_integral_cumulative(list(pilot_lam), pilot_lambdas, pilot_metric_g),
            dtype=float,
        )
        if not np.all(np.diff(pilot_measure) > 0.0):
            raise ValueError(
                "pilot 的 ∫g 累计度量不是严格递增，无法用于选边/反插值"
            )
    else:
        pilot_measure = pilot_s

    def _measure(vals) -> np.ndarray:
        """把一串 λ 换算成配置度量下的累计坐标（用于比较边长）。"""
        return np.asarray(
            [_pilot_arclength_of(x, pilot_lam, pilot_measure) for x in vals],
            dtype=float,
        )

    # ---- 插几个 ----
    if n_insert is None:
        cum_all = _measure(lambdas)
        all_edges = np.abs(np.diff(cum_all))
        if all_edges.size == 0 or not np.all(np.isfinite(all_edges)):
            raise RuntimeError("全路径边长不可用，无法定 L_target")
        l_target = float(np.mean(all_edges))
        if not (l_target > 0.0):
            raise RuntimeError(f"L_target={l_target} 非正，无法定插点数")
        win_edges = np.abs(np.diff(cum_all[start:end]))
        # 🔑 [2026-09-11] 必须留相对容差。L_target 是**全路径均值**，而生产布局
        # 本来就是按热力学长度等分的 ⟹ 每条边都恰好≈均值，`e/L_target` 在 1 的
        # 两侧抖 ~1e-15。裸 ceil 会把高出 1e-15 的那半数边判成"要插点"：实测
        # 16 个等距 λ、边长逐位不等仅 1.5e-15，就算出 n_needed=3（应为 0）。
        # 于是"边级证据不支持插点"这道守卫被浮点噪声整个击穿。
        _rel_tol = 1e-9
        n_needed = int(
            sum(
                max(0, int(np.ceil(float(e) / l_target - _rel_tol)) - 1)
                for e in win_edges
            )
        )
        if n_needed == 0:
            raise RuntimeError(
                f"失败窗口 {failed} 内没有任何一条边超过全路径平均热力学长度 "
                f"L_target={l_target:.4f}（窗内各边 "
                f"{[round(float(e), 4) for e in win_edges]}）—— **边级证据不支持插点**。"
                "这个窗口的缺陷是窗口级的（跨度 / 混合 / f_k），插几个点必须由调用方"
                "按窗口级判据给出 n_insert；本函数不替它猜，也不静默插 1 个。"
                "见 docs/PLAN_PATH_REPAIR_2026-09-11.md §3ter.4。"
            )
    else:
        n_needed = int(n_insert)
        if n_needed < 1:
            raise ValueError(f"n_insert 必须 >= 1，收到 {n_insert}")

    # ---- 插在哪：窗内最长边的 pilot 弧长中点（与重写前逐字一致） ----
    inserted: List[float] = []
    inserted_edges: List[List[int]] = []
    for _ in range(n_needed):
        arc = [_pilot_arclength_of(lambdas[i], pilot_lam, pilot_measure)
               for i in range(start, end)]
        edges = [abs(arc[i + 1] - arc[i]) for i in range(len(arc) - 1)]
        if not edges or max(edges) <= 0.0:
            raise RuntimeError(f"窗口 {(start, end)} 内各边度量长度为零，无法选插点")
        worst = int(np.argmax(edges))
        left = start + worst
        lam_mid = float(
            np.interp(0.5 * (arc[worst] + arc[worst + 1]), pilot_measure, pilot_lam)
        )
        a_lo, a_hi = sorted((lambdas[left], lambdas[left + 1]))
        if not a_lo < lam_mid < a_hi:
            raise RuntimeError(
                f"热力学中点未严格落在边内部：{lambdas[left]}, {lam_mid}, {lambdas[left + 1]}"
            )
        lambdas.insert(left + 1, lam_mid)
        inserted.append(lam_mid)
        inserted_edges.append([int(left), int(left + 1)])
        # 🔑 **ranges 不动。** 插入之后 lambdas[start:end] 的内容整体左移一格，
        # 失败窗口的跨度因此缩小；被挤出去的那个态归下一个窗口。

    # ---- 末窗吸收溢出；其余区间逐字不变 ----
    # 🔑 [2026-09-11] **溢出槽有上界，别让它变成死胡同。**
    # 末窗豁免 max_states_per_window，但"能一分为二"这件事本身有区间：
    # 两子窗共享一个边界态 ⟹ p+q−1=K，两侧都要落在 [lo,hi] ⟹ 可拆的 K 只有
    # ``2*lo−1 .. 2*hi−1``（4/5 配置下就是 7..9，只有三个值宽）。末窗一旦被插到
    # K > 2*hi−1 就**永远拆不开**了，溢出槽从此是死胡同：既不能再吸收（迟早
    # 压不住），也不能拆（切不出合法子窗），而唯一的收尾动作恰恰是拆它。
    # 当前默认 max_insertions=3 只是**巧合**地把 K 压在 5+3=8 ≤ 9 以内
    # （`_run_stage2_with_path_evolution` 的 rounds_done 是跨 resume 累计的），
    # 那是巧合不是守卫 —— 谁把 max_insertions 调大就会踩进去。所以这里 fail-closed。
    _tail_k_after = (ranges[-1][1] + n_needed) - ranges[-1][0]
    _tail_k_split_max = 2 * hi_n - 1
    if _tail_k_after > _tail_k_split_max:
        raise RuntimeError(
            f"末窗 {ranges[-1]}（当前 {ranges[-1][1] - ranges[-1][0]} 态）再吸收 "
            f"{n_needed} 个溢出态会到 K={_tail_k_after}，超过**可拆上限** "
            f"2·max−1={_tail_k_split_max}（可拆区间 "
            f"{2 * lo_n - 1}..{_tail_k_split_max}）—— 此后末窗永远切不出两个落在 "
            f"[{lo_n},{hi_n}] 的子窗，溢出槽变成死胡同。"
            + (
                f"⚠️ 这次的 n={n_needed} 是**边级判据**算出来的（窗内有边长到需要"
                f"这么多点），而溢出预算只有 {_tail_k_split_max - (ranges[-1][1] - ranges[-1][0])} "
                "个态 —— 两者之间没有任何约束保证前者 ≤ 后者。一条边长到平均的几倍，"
                "本身说明**初始布点**就不对（度规布点不该产生这种边），那是布局期的"
                "问题，不是运行时修补能补的。"
                if n_insert is None else
                "这次的 n 是调用方给的（窗口级判据）。"
            )
            + "正确处置二选一：(a) 趁 K 还在可拆区间内**先拆末窗**，拆完的右孩子重新"
            "成为溢出槽（注意：孩子受 hi 约束，只能再腾出很少几个）；"
            "(b) 承认这是 HALT_LAMBDA_BUDGET_INSUFFICIENT —— 输入的 λ 总数从一开始"
            "就不够，退出进程、人工改输入文件的 λ 数目。**不要把 n 封顶到预算**，"
            "那会静默产出一个不满足触发它那份证据的布局；理由见 "
            "docs/PLAN_PATH_REPAIR_2026-09-11.md §2。"
        )
    new_ranges = list(ranges[:-1]) + [(ranges[-1][0], ranges[-1][1] + n_needed)]
    validate_single_shared_boundary_ranges(new_ranges, len(lambdas))

    # ---- 后置断言：model B 的三条不变量 ----
    if len(lambdas) != n_states_before + n_needed:
        raise RuntimeError(
            f"插点后态数应为 {n_states_before + n_needed}，实得 {len(lambdas)}"
        )
    # (1) 前缀（插入点之前、可能已经采完的窗口）λ 必须逐位不变。
    for (a0, b0) in ranges:
        if b0 <= start + 1:
            before_vals = [round(float(x), 12) for x in list(lambdas_var)[a0:b0]]
            after_vals = [round(float(x), 12) for x in lambdas[a0:b0]]
            if before_vals != after_vals:
                raise RuntimeError(
                    f"前缀窗口 {(a0, b0)} 装的 λ 被改动了（model B 要求逐位不变）：\n"
                    f"  before={before_vals}\n  after ={after_vals}"
                )
    # (2) 非末窗的态数必须一个都没变（溢出只许落末窗）。
    for (a0, b0), (a1, b1) in zip(ranges[:-1], new_ranges[:-1]):
        if (b1 - a1) != (b0 - a0):
            raise RuntimeError(
                f"非末窗 {(a0, b0)} → {(a1, b1)} 的态数变了；溢出只允许由末窗吸收"
            )
        if (b1 - a1) > hi_n:
            raise RuntimeError(
                f"非末窗 {(a1, b1)} 有 {b1 - a1} 个态，超过 max_states_per_window={hi_n}"
            )
    if (new_ranges[-1][1] - new_ranges[-1][0]) != (ranges[-1][1] - ranges[-1][0]) + n_needed:
        raise RuntimeError("末窗没有吸收全部溢出态")
    # (3) 失败窗口的 λ 跨度必须真的变小 —— 除非它本身就是末窗（末窗是溢出槽，
    #     它的正常行为是**变大**，缩不了；那种情况下该做的是拆末窗，不是插点）。
    new_span = abs(lambdas[new_ranges[-1][1] - 1] - lambdas[start]) if is_last \
        else abs(lambdas[end - 1] - lambdas[start])
    if not is_last and not (new_span < original_span - 1e-12):
        raise RuntimeError(
            f"插点没有缩小失败窗口的 λ 跨度（{original_span:.6f} → {new_span:.6f}），"
            "拒绝返回。"
        )

    return lambdas, new_ranges, {
        "source": "ibs_failed_window_pilot_interpolated_midpoint_insertion",
        "accounting": "model_B_fixed_ranges_tail_absorbs_overflow",
        "note": "pilot 插值候选加密位置，非本次 VALIDATE 证实的故障边",
        "failed_global_state_range": [int(failed[0]), int(failed[1])],
        "failed_window_is_last": bool(is_last),
        "n_inserted": int(n_needed),
        "n_insert_source": "caller" if n_insert is not None else "edge_length",
        "inserted_lambdas": [float(x) for x in inserted],
        "inserted_lambda": float(inserted[-1]) if inserted else None,
        "inserted_global_edges": inserted_edges,
        "failed_global_edge": list(inserted_edges[0]) if inserted_edges else None,
        "partition_criterion": criterion,
        "n_windows_before": int(w_before),
        "n_windows_after": int(len(new_ranges)),
        "failed_window_span_before": float(original_span),
        "failed_window_span_after": float(new_span),
        "tail_window_before": list(ranges[-1]),
        "tail_window_after": list(new_ranges[-1]),
        "frozen_prefix_windows": [list(r) for r in ranges if r[1] <= start + 1],
        "min_states_per_window": lo_n,
        "max_states_per_window": hi_n,
        "max_states_per_window_exempt": "tail_window",
    }


def insert_thermodynamic_midpoint_from_ibs_lse_failure(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    warmup_diagnostics: Dict,
    pilot_lambdas: List[float],
    pilot_cumulative_length: List[float],
) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
    """Bridge an irreducible two-state LSE failure with one measured midpoint.

    The inserted coordinate is halfway in the pilot thermodynamic coordinate,
    not the arithmetic lambda midpoint.  The failed [a,b] ensemble is replaced
    by [a,m] and [m,b], and both must later pass a fresh IBS LSE design probe.
    """
    lambdas = [float(x) for x in lambdas_var]
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    failed = tuple(int(x) for x in warmup_diagnostics["global_state_range"])
    if failed not in ranges:
        raise RuntimeError(f"LSE 失败窗口 {failed} 不在当前窗口列表 {ranges} 中")
    start, end = failed
    if end - start != 2:
        raise RuntimeError("只有不可再拆的两态 IBS 窗口允许插入 lambda")

    pilot_lam = np.asarray(pilot_lambdas, dtype=float).ravel()
    pilot_s = np.asarray(pilot_cumulative_length, dtype=float).ravel()
    if pilot_lam.size != pilot_s.size or pilot_lam.size < 2:
        raise ValueError("pilot lambda 与累计热力学长度必须等长且至少含两个点")
    if not np.all(np.diff(pilot_s) > 0.0):
        raise ValueError("pilot 累计热力学长度必须严格递增")

    lambda_left, lambda_right = lambdas[start], lambdas[start + 1]
    # pilot lambda descends while np.interp requires ascending xp.
    s_left = float(np.interp(lambda_left, pilot_lam[::-1], pilot_s[::-1]))
    s_right = float(np.interp(lambda_right, pilot_lam[::-1], pilot_s[::-1]))
    s_mid = 0.5 * (s_left + s_right)
    lambda_mid = float(np.interp(s_mid, pilot_s, pilot_lam))
    lo, hi = sorted((lambda_left, lambda_right))
    if not lo < lambda_mid < hi:
        raise RuntimeError(
            f"热力学中点未严格位于失败边内部: {lambda_left}, {lambda_mid}, {lambda_right}"
        )

    insert_at = start + 1
    new_lambdas = list(lambdas)
    new_lambdas.insert(insert_at, lambda_mid)
    shifted_end = end + 1
    new_ranges = []
    for current_start, current_end in ranges:
        if (current_start, current_end) == failed:
            new_ranges.extend([(start, insert_at + 1), (insert_at, shifted_end)])
        elif current_end <= insert_at:
            new_ranges.append((current_start, current_end))
        elif current_start >= insert_at:
            new_ranges.append((current_start + 1, current_end + 1))
        else:
            new_ranges.append((current_start, current_end + 1))
    new_ranges = canonicalize_window_ranges(new_ranges, len(new_lambdas))
    return new_lambdas, new_ranges, {
        "source": "ibs_lse_design_thermodynamic_midpoint_insertion",
        "failed_global_state_range": [start, end],
        "failed_global_edge": [start, start + 1],
        "failed_lambdas": [lambda_left, lambda_right],
        "inserted_global_state": int(insert_at),
        "inserted_lambda": lambda_mid,
        "inserted_thermodynamic_coordinate": s_mid,
        "replacement_ranges": [[start, insert_at + 1], [insert_at, shifted_end]],
        "lse_balance": warmup_diagnostics.get("lse_balance"),
    }


def split_window_from_warmup_failure(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    warmup_diagnostics: Dict,
    min_states_before_split: int = 5,
    min_states_per_window_floor: int = 3,
) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
    """Split a failed IBS window without inventing a new thermodynamic state.

    Warmup probabilities are measured in a time-dependent IBS mixture, so
    ``std(beta Delta-u)`` from that mixture is not a valid estimate of the
    fixed-lambda thermodynamic metric.  A warmup coverage failure therefore
    changes only the window partition.  The two children share exactly one
    existing lambda state, which is sufficient to stitch their free energies.

    The initial pilot layout intentionally uses a wide cumulative-overlap
    budget (``pilot_overlap_thermodynamic_length``), so neighboring windows
    can legitimately share several states -- e.g. a 6-state parent sharing 3
    states with its next neighbor. That is by design, not a bug. But once this
    parent SPLITS, only ``start`` is preserved on the left child and only
    ``end`` on the right child -- the right child's ``end`` is identical to
    the parent's, so it still inherits the *same* multi-state overlap with
    whatever untouched neighbor came after the parent. For the right child
    (now much smaller than the original parent) that overlap can become a
    large fraction of its own span (observed: a 4-state child sharing 3 states
    -- 75% -- with an untouched 6-state neighbor it used to share only 3-of-6
    with). ``canonicalize_window_ranges`` deliberately does not touch this
    (partial, non-containing overlap is its own legitimate case), so this must
    be fixed here, right after the split: reduce the immediate next
    neighbor's overlap with the new right child down to exactly one shared
    state, the same convention used everywhere else new windows get stitched
    together. Nothing else is re-laid-out -- only ``start`` moves on that one
    neighbor, which cannot change its own overlap with whatever follows IT
    (that overlap is governed by its unchanged ``end``), so this does not
    cascade any further down the path.
    """
    lambdas = [float(x) for x in lambdas_var]
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    failed = tuple(int(x) for x in warmup_diagnostics["global_state_range"])
    if failed not in ranges:
        raise RuntimeError(f"warmup 失败窗口 {failed} 不在当前窗口列表 {ranges} 中")
    start, end = failed
    if end - start < int(min_states_before_split):
        raise RuntimeError(
            f"warmup 失败窗口只有 {end-start} 个态，不能再盲拆；必须使用 fixed-lambda overlap 探针"
        )

    # m is an existing global state.  [start:m+1) and [m:end) share only m.
    # A 2-state IBS window is statistically fragile (see run_all_windows'
    # own comment to that effect), so each child must have >= 3 states;
    # sharing exactly 1 state means the parent needs >= 3+3-1=5 states for
    # this to even be possible -- enforced by min_states_before_split's
    # default above, not just by this floor check.
    middle = (start + end - 1) // 2
    children = [(start, middle + 1), (middle, end)]
    if min(e - s for s, e in children) < 3:
        raise RuntimeError(f"拆分会产生少于三个态的窗口: {children}")
    right_child_end = children[-1][1]

    # 找到失败窗口在原列表中的位置，只调整紧随其后的那一个邻窗（如果存在且
    # 目前跟新右孩子共享超过一个态）；不触碰失败窗口左侧的邻窗——左孩子的
    # start 跟原失败窗口完全相同，它与左侧邻窗的重叠（由左侧邻窗的 end 决定）
    # 不受这次拆分影响。
    failed_pos = ranges.index(failed)
    next_start_override: Optional[int] = None
    if failed_pos + 1 < len(ranges):
        next_s, next_e = ranges[failed_pos + 1]
        if next_s < right_child_end - 1:
            candidate_start = right_child_end - 1
            if next_e - candidate_start >= int(min_states_per_window_floor):
                next_start_override = candidate_start
            # 否则调整后邻窗会小于最小态数下限，保留原样，不强行压缩——
            # 这种情形应该很少见（邻窗本身已经接近最小尺寸）。

    new_ranges: List[Tuple[int, int]] = []
    neighbor_adjustment = None
    for idx, current in enumerate(ranges):
        if current == failed:
            new_ranges.extend(children)
        elif idx == failed_pos + 1 and next_start_override is not None:
            adjusted = (next_start_override, current[1])
            neighbor_adjustment = {
                "old_range": list(current),
                "new_range": list(adjusted),
            }
            new_ranges.append(adjusted)
        else:
            new_ranges.append(current)
    covered = sorted({i for s, e in new_ranges for i in range(s, e)})
    if covered != list(range(len(lambdas))):
        raise RuntimeError(f"拆窗后未完整覆盖 lambda 路径: {covered}")

    feedback = {
        "source": "warmup_window_split_only",
        "failed_window": int(warmup_diagnostics.get("window_index", -1)),
        "failed_global_state_range": [start, end],
        "child_ranges": [list(r) for r in children],
        "shared_global_state": int(middle),
        "next_neighbor_reflowed_to_single_state_overlap": neighbor_adjustment,
        "inserted_lambda": None,
    }
    return lambdas, new_ranges, feedback


def insert_lambda_from_overlap_failure(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    warmup_diagnostics: Dict,
) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
    """Insert one state only after a real bidirectional fixed-H overlap failure.

    The coordinate midpoint is merely the next point to *measure*.  No child
    thermodynamic lengths are fabricated; the pilot cache is explicitly
    invalidated by the caller and the new edges must be measured next round.
    """
    overlap = warmup_diagnostics.get("bidirectional_overlap_probe", {})
    pairs = overlap.get("pairs", [])
    failed_pairs = [p for p in pairs if not bool(p.get("passed", False))]

    asymmetric = overlap.get("passed_but_asymmetric_bottleneck")
    preserve_expanded_parent = False

    if failed_pairs:
        worst = min(failed_pairs, key=lambda p: float(p.get("min_bidirectional_overlap", np.inf)))
        source = "fixed_hamiltonian_bidirectional_overlap"
    elif asymmetric and asymmetric.get("qualified"):
        worst = dict(asymmetric["pair"])
        preserve_expanded_parent = True
        source = "fixed_hamiltonian_passed_but_asymmetric_bottleneck"
    else:
        raise RuntimeError("没有 fixed-H 失败边或合格的通过但不对称瓶颈边，拒绝插点")

    global_edge = int(worst["global_edge"][0])
    lambdas = [float(x) for x in lambdas_var]
    if not 0 <= global_edge < len(lambdas) - 1:
        raise RuntimeError(f"fixed-lambda overlap 失败边索引越界: {global_edge}")

    midpoint = 0.5 * (lambdas[global_edge] + lambdas[global_edge + 1])
    insert_at = global_edge + 1
    new_lambdas = list(lambdas)
    new_lambdas.insert(insert_at, float(midpoint))

    failed_range = tuple(int(x) for x in warmup_diagnostics["global_state_range"])
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    if failed_range not in ranges:
        raise RuntimeError(f"overlap 失败窗口 {failed_range} 不在当前窗口列表 {ranges} 中")

    # K<=4 是 fixed-H overlap 探针/MBAR 校准通道自身的准入上限
    # (ibs_engine.py: `K <= 4 and stage_type == "vdw"`)。合并成单一父窗口
    # 只在结果仍落在这个上限内时才安全——否则父窗口会静默失去重新进入该
    # 通道的资格，比制造一个两态脆弱子窗口更糟。
    if preserve_expanded_parent and (failed_range[1] + 1 - failed_range[0]) > 4:
        preserve_expanded_parent = False

    new_ranges: List[Tuple[int, int]] = []
    for start, end in ranges:
        if (start, end) == failed_range:
            shifted_end = end + 1
            if preserve_expanded_parent:
                # 三态父窗口插点后保留为四态窗口，避免产生 3态+2态。
                new_ranges.append((start, shifted_end))
            else:
                # Both children contain the new state at insert_at.
                new_ranges.extend([(start, insert_at + 1), (insert_at, shifted_end)])
        elif end <= insert_at:
            new_ranges.append((start, end))
        elif start >= insert_at:
            new_ranges.append((start + 1, end + 1))
        else:
            new_ranges.append((start, end + 1))

    covered = sorted({i for s, e in new_ranges for i in range(s, e)})
    if covered != list(range(len(new_lambdas))):
        raise RuntimeError(f"fixed-overlap 插点后未完整覆盖 lambda 路径: {covered}")
    feedback = {
        "source": source,
        "failed_window": int(warmup_diagnostics.get("window_index", -1)),
        "failed_global_edge": [global_edge, global_edge + 1],
        "selected_global_edge": [global_edge, global_edge + 1],
        "inserted_lambda": float(midpoint),
        "measured_min_bidirectional_overlap": float(worst["min_bidirectional_overlap"]),
        "overlap_threshold": float(worst["threshold"]),
        "preserved_expanded_parent_window": preserve_expanded_parent,
        "asymmetry_diagnostics": asymmetric if preserve_expanded_parent else None,
        "thermodynamic_lengths_invalidated": True,
    }
    return new_lambdas, new_ranges, feedback


def plan_vdw_overlap_repair_targets(
    window_ranges: List[Tuple[int, int]],
    window_overlap_diagnostics: List[Dict],
    min_overlap_threshold: float,
    min_states_before_split: int = 5,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Classify production-time low-ESS windows, without touching any lambda.

    Mirrors the warmup-failure split-first-then-probe policy
    (``split_window_from_warmup_failure`` / ``insert_lambda_from_overlap_failure``)
    instead of the old worst-per-lambda-state + arithmetic-midpoint path in
    ``refine_stage_lambda_path_by_overlap``: a whole window reporting low ESS is
    not evidence that one particular lambda edge is too wide (a saturated IBS
    bias or slow conformational relaxation depresses every state's ESS equally,
    which is exactly why the old code's "worst per-lambda state" pick could be
    pure noise -- see ``window 3`` in the reported case, where every state sat
    at min_ess_ratio~=0.0035). So a failing window is only ever split here
    (sharing one existing state, no new lambda), never bisected directly. The
    caller is expected to run a real fixed-Hamiltonian bidirectional overlap
    probe on any window this returns as un-splittable (already at or below
    ``min_states_before_split`` states) before allowing a lambda insertion.

    Returns ``(windows_to_split, windows_needing_probe)``, both lists of
    ``(start, end)`` tuples drawn verbatim from ``window_ranges`` (not
    reconstructed from the diagnostics' own ``lambdas`` field beyond using it
    to locate the match, so downstream code can keep operating on the caller's
    own range objects).
    """
    ranges = [(int(s), int(e)) for s, e in window_ranges]
    ranges_set = set(ranges)
    to_split: List[Tuple[int, int]] = []
    to_probe: List[Tuple[int, int]] = []
    seen = set()
    for rec in window_overlap_diagnostics or []:
        ratio = rec.get("min_ess_ratio")
        lambdas_idx = rec.get("lambdas")
        if ratio is None or not np.isfinite(ratio) or ratio >= min_overlap_threshold:
            continue
        if not lambdas_idx:
            continue
        start, end = int(min(lambdas_idx)), int(max(lambdas_idx)) + 1
        if (start, end) not in ranges_set or (start, end) in seen:
            continue
        seen.add((start, end))
        if (end - start) >= int(min_states_before_split):
            to_split.append((start, end))
        else:
            to_probe.append((start, end))
    return to_split, to_probe


def canonicalize_window_ranges(
    window_ranges: List[Tuple[int, int]],
    n_states: int,
) -> List[Tuple[int, int]]:
    """Remove exact duplicates and strictly-contained windows after a batch
    split, then verify the survivors still fully cover [0, n_states) with
    every adjacent pair (sorted by start) sharing at least one state.

    Splitting several overlapping *parent* windows independently -- one
    ``split_window_from_warmup_failure`` call per failing parent -- can
    produce a child that lands entirely inside a NEIGHBORING parent's span,
    because IBS windows overlap by design. Concretely: parents (0,6) and
    (3,9) (sharing states 3,4,5) each split independently via
    ``middle=(s+e-1)//2``: (0,6) -> (0,3),(2,6); (3,9) -> (3,6),(5,9). The
    child (3,6) is then a strict subset of the child (2,6) (both cover a
    span within {2,3,4,5}), a real case observed when 5 overlapping 6-state
    parents were all split in one round, producing 10 windows instead of the
    minimal connected 6-window chain. Coverage is never actually at risk from
    this (a contained window adds no lambda index its superset doesn't
    already have), but the redundant windows get sampled anyway -- wasted
    GPU time now, and unbounded window-count growth if a later round splits
    them again. This does NOT merge partially-overlapping-but-not-nested
    windows (neither contains the other) -- those provide genuine additional
    overlap and are kept as-is.
    """
    ranges = sorted({(int(s), int(e)) for s, e in window_ranges})
    kept: List[Tuple[int, int]] = []
    for s, e in ranges:
        # A strict subset of an already-kept window contributes no new
        # coverage/adjacency; skip it entirely.
        if any(ks <= s and e <= ke and (ks, ke) != (s, e) for ks, ke in kept):
            continue
        # A previously-kept window can only be a strict subset of this one
        # when they share the same start (sort order guarantees any kept
        # window with a smaller start cannot be contained in this one).
        kept = [
            (ks, ke) for ks, ke in kept
            if not (s <= ks and ke <= e and (ks, ke) != (s, e))
        ]
        kept.append((s, e))

    kept.sort()
    covered = sorted({i for s, e in kept for i in range(s, e)})
    if covered != list(range(n_states)):
        raise RuntimeError(
            f"窗口归约后未完整覆盖 [0,{n_states})，覆盖到 {covered}"
        )
    for (s0, e0), (s1, e1) in zip(kept, kept[1:]):
        if s1 >= e0:
            raise RuntimeError(
                f"窗口归约后相邻窗口不再共享任何状态: {(s0, e0)} 与 {(s1, e1)}"
            )
    for i, (si, ei) in enumerate(kept):
        for j, (sj, ej) in enumerate(kept):
            if i != j and sj <= si and ei <= ej:
                raise RuntimeError(
                    f"窗口归约后仍残留嵌套窗口: {(si, ei)} 被 {(sj, ej)} 严格包含"
                )
    return kept


# #[P1 FIX] 抽取共享方差归一化逻辑，消除重复代码
# abfe_preoptimizer.py 顶部 (约第 15 行)
def _normalize_variance_weights(std_dev_clipped, max_ratio=0.15):
    """共享方差归一化函数，消除代码重复"""
    density_weight = np.log1p(std_dev_clipped) + 0.1
    max_weight = np.sum(density_weight) * max_ratio
    clipped = np.clip(density_weight, None, max_weight)
    # ✅ 修复：Clip 后必须重新归一化，保证 ∑w = 1（等熵长度分布前提）
    return clipped / (np.sum(clipped) + 1e-10)


def finalize_descending_lambda_path(
    optimized_lambdas: np.ndarray,
    target_n_states: int,
    min_spacing: Optional[float] = None,
) -> Tuple[np.ndarray, float, bool]:
    """Shared post-interpolation invariant enforcement for a descending
    (1.0 -> 0.0) CDF-interpolated lambda path.

    Both ``ABFEPreOptimizer.optimize_lambda_path_adaptive`` (single-lambda
    vdw path) and ``DualLambdaPreOptimizer.optimize_stage1_decharging``
    (dual-lambda decharging path) build their own ``optimized_lambdas`` array
    from CDF interpolation, using different density-weight formulas -- that
    physics-specific weighting is intentionally left to each caller. But the
    *invariants* the result must satisfy afterward (finite, bounded to
    [0,1], strictly descending with a minimum spacing, deduplicated, and a
    fail-closed fallback to a linear path if too few distinct states survive)
    are identical, and used to be duplicated only in the single-lambda path;
    ``optimize_stage1_decharging`` only clipped/sorted/pinned endpoints with
    no min-spacing, no dedup, and no fail-closed fallback -- a valid-looking
    CDF interpolation could silently hand back two states with (numerically)
    the same lambda, which breaks MBAR's distinct-state assumption for that
    edge without ever raising or logging.

    Returns ``(lambdas, min_spacing_used, fell_back_to_linear)`` so a caller
    can log/record what was actually applied.
    """
    optimized_lambdas = np.asarray(optimized_lambdas, dtype=float).ravel()
    target_n_states = int(target_n_states)
    if not np.all(np.isfinite(optimized_lambdas)):
        optimized_lambdas = np.linspace(1.0, 0.0, target_n_states)
    optimized_lambdas = np.clip(optimized_lambdas, 0.0, 1.0)
    optimized_lambdas = np.sort(optimized_lambdas)[::-1]
    optimized_lambdas = np.minimum.accumulate(optimized_lambdas)
    optimized_lambdas[0], optimized_lambdas[-1] = 1.0, 0.0

    if min_spacing is None:
        min_spacing = max(0.02, 0.9 / max(target_n_states - 1, 1))
    min_spacing = float(min_spacing)

    for i in range(1, len(optimized_lambdas)):
        if optimized_lambdas[i] < 0.0:
            optimized_lambdas[i] = 0.0
        if optimized_lambdas[i - 1] - optimized_lambdas[i] < min_spacing:
            optimized_lambdas[i] = max(0.0, optimized_lambdas[i - 1] - min_spacing)

    unique_lambdas = []
    spacing_eps = 1e-9
    for lam in optimized_lambdas:
        lam_val = float(lam)
        if not unique_lambdas or (unique_lambdas[-1] - lam_val) >= (min_spacing - spacing_eps):
            unique_lambdas.append(lam_val)
    if unique_lambdas:
        unique_lambdas[0] = 1.0
        unique_lambdas[-1] = 0.0

    if len(unique_lambdas) < target_n_states:
        return np.linspace(1.0, 0.0, target_n_states), min_spacing, True
    return np.array(unique_lambdas), min_spacing, False


def redistribute_lambda_by_delta_f(
    lambdas_in_order: np.ndarray,
    f_k_in_order: np.ndarray,
    n_states: Optional[int] = None,
) -> np.ndarray:
    """
    按累积 |Δf|（真实自由能曲线的弧长）重新分布 λ 点，而不是等 λ 间距，也不是
    `_normalize_variance_weights` 那种基于短程试探采样、又被 log1p 压缩过的方差代理。

    输入的 f_k_in_order 必须是"已经实测、已修正单位"的自由能曲线（比如
    solve_stage_integrated 的输出），跟 lambdas_in_order 按同一顺序对齐。
    端点 λ 值保持不变，中间点按累积 |Δf| 等分，使每一步的自由能变化量大致相等。
    """
    lambdas_in_order = np.asarray(lambdas_in_order, dtype=float)
    f_k_in_order = np.asarray(f_k_in_order, dtype=float)
    if n_states is None:
        n_states = len(lambdas_in_order)
    if n_states < 2:
        raise ValueError("n_states 必须至少为 2（保留两个端点）")

    abs_steps = np.abs(np.diff(f_k_in_order))
    cum = np.concatenate([[0.0], np.cumsum(abs_steps)])
    total = float(cum[-1])
    if total <= 1e-8:
        # 曲线几乎平坦，退化为等 λ 间距。
        return np.linspace(lambdas_in_order[0], lambdas_in_order[-1], n_states)

    targets = np.linspace(0.0, total, n_states)
    new_lambdas = np.interp(targets, cum, lambdas_in_order)
    new_lambdas[0] = lambdas_in_order[0]
    new_lambdas[-1] = lambdas_in_order[-1]
    return new_lambdas


def _pilot_ti_cumulative_f(lam_sorted: np.ndarray, grad_sorted: np.ndarray) -> np.ndarray:
    """Trapezoidal thermodynamic integration of a pilot's measured mean
    gradient <dU/dlambda> into a raw F(lambda) curve, gauge-referenced to
    F(lam_sorted[0]) = 0. Inputs must already be sorted ascending in lambda.

    Shared by ``estimate_f_k_from_pilot_ti`` (bias-seed use case: mean-centers
    the physical free-energy curve into the IBS bias-parameter convention) and
    ``redistribute_vanishing_lambda_subdomains`` (lambda-spacing use case:
    only needs real |Delta F| magnitudes, no sign/gauge convention) -- kept
    as one function so both stay derived from the same integration, not two
    independently-maintained copies of the same trapezoidal rule.
    """
    seg = 0.5 * (grad_sorted[:-1] + grad_sorted[1:]) * np.diff(lam_sorted)
    return np.concatenate(([0.0], np.cumsum(seg)))


def estimate_f_k_from_pilot_ti(
    pilot_lambdas: Optional[List[float]],
    pilot_mean_dU_dlambda: Optional[List[float]],
    target_lambdas: List[float],
) -> Optional[np.ndarray]:
    """[IBS_BIAS_PROTOCOL_VERSION warm-start] Estimate a mean-centered f_k seed
    for ``target_lambdas`` via thermodynamic integration of the pilot's own
    measured mean gradient, instead of cold-starting online learning at
    f_k=0.0 for every state.  The returned array is already in the IBS
    bias-parameter convention.  For

    ``V_IBS = -kT log sum_k exp[-beta (U_k - f_k)]``,

    the integrated contribution of state ``k`` is proportional to
    ``exp(beta*f_k) Z_k = exp[beta*(f_k - F_k)]``.  Flat state weights therefore
    require ``f_k = F_k + constant``: the physical TI curve is mean-centered,
    not sign-inverted.  Occupancy feedback still has the complementary rule
    that an *observed overrepresented* state must have its ``f_k`` lowered.
    Confusing physical free energy with observed occupancy previously inverted
    this warm-start seed and made the online feedback spend most of its budget
    undoing the initialization.

    Five independent real GPU attempts at fixing vanishing window 0 by
    reshaping the lambda grid (6/4/3-state regrouping, adaptive pilot-grid
    refinement, real-Delta_f-equalized placement) all failed with occupation
    pinned at 96-99% on state 0 -- the online SGD/TMBAR loop must *discover*
    the needed bias purely from ~20-frame batches, and if the underlying
    transition is itself hard to sample, batches see almost no evidence from
    the underrepresented states, starving the learner of what it needs to
    grow the bias further (a genuine bootstrap problem, independent of how
    the lambda grid is spaced). ``_sample_scalar_metric`` already records
    ``mean_dU_dlambda_kJ_mol`` (the mean gradient, not just the variance proxy
    ``metric_g`` the rest of this file uses for spacing) at every pilot
    point -- integrating it via the trapezoidal rule gives a real F(lambda)
    estimate, available *before* any window is ever sampled, that can seed
    f_k with roughly the right scale from the first learning update instead
    of requiring the SGD loop to bootstrap that scale from scratch under
    exactly the sampling conditions that make bootstrapping hard.

    Deliberately conservative: returns ``None`` (meaning "no seed; caller
    falls back to today's implicit f_k=0.0") rather than raising or silently
    fabricating a value, in every case where the estimate would not be
    trustworthy:
      - ``pilot_lambdas``/``pilot_mean_dU_dlambda`` missing or empty (old,
        pre-this-feature preopt cache with no ``pilot_points`` data).
      - Fewer than 2 pilot points (trapezoidal integration needs at least 2).
      - Any non-finite value in either input array (a failed/corrupt pilot
        sample must not silently poison the seed).

    ``target_lambdas`` outside ``pilot_lambdas``'s actual measured range are
    NOT extrapolated (unreliable) -- they are clamped to the nearest boundary
    F(lambda) value via ``np.interp``'s ``left``/``right`` parameters, and a
    warning is printed so this isn't silently mistaken for a real estimate.
    """
    if not pilot_lambdas or not pilot_mean_dU_dlambda:
        return None
    try:
        pilot_lambdas = np.asarray(pilot_lambdas, dtype=float).ravel()
        grad = np.asarray(pilot_mean_dU_dlambda, dtype=float).ravel()
    except (TypeError, ValueError):
        # e.g. a None entry from an older/partial pilot_points record that's
        # missing mean_dU_dlambda_kJ_mol for some point -- can't safely cast,
        # not a real estimate either way.
        print("  [WARN] [pilot TI 热启动] pilot_lambdas/mean_dU_dlambda 无法转换为数值数组，放弃热启动，回退 f_k=0.0")
        return None
    if pilot_lambdas.size < 2 or grad.size != pilot_lambdas.size:
        return None
    if not np.all(np.isfinite(pilot_lambdas)) or not np.all(np.isfinite(grad)):
        print("  [WARN] [pilot TI 热启动] pilot_lambdas/mean_dU_dlambda 含非有限值，放弃热启动，回退 f_k=0.0")
        return None

    order = np.argsort(pilot_lambdas)
    lam_sorted = pilot_lambdas[order]
    grad_sorted = grad[order]
    # F(lambda): trapezoidal TI, referenced to lam_sorted[0] (arbitrary gauge
    # -- mean-centering below removes it anyway).
    f_at_pilot = _pilot_ti_cumulative_f(lam_sorted, grad_sorted)

    target = np.asarray(target_lambdas, dtype=float).ravel()
    if target.size == 0 or not np.all(np.isfinite(target)):
        return None
    lo, hi = float(lam_sorted[0]), float(lam_sorted[-1])
    if np.any(target < lo) or np.any(target > hi):
        print(
            f"  [WARN] [pilot TI 热启动] target_lambdas 超出 pilot 实测范围 "
            f"[{lo:.4f}, {hi:.4f}]，越界部分钳位到边界值，不做外推"
        )
    f_at_target = np.interp(target, lam_sorted, f_at_pilot, left=f_at_pilot[0], right=f_at_pilot[-1])
    # [IBS_BIAS_PROTOCOL_VERSION=27] Keep the physical TI sign.  Since the IBS
    # mixture uses exp[-beta*(U_k-f_k)], equal integrated state weights require
    # f_k=F_k+constant.  The former sign inversion produced the exact opposite
    # seed and forced the bounded occupancy feedback to undo it online.
    f_at_target = f_at_target - float(np.mean(f_at_target))
    return f_at_target


def pilot_ti_seed_trust_diagnostics(
    pilot_lambdas: Optional[List[float]],
    pilot_mean_dU_dlambda: Optional[List[float]],
    pilot_std_dU_dlambda: Optional[List[float]],
    pilot_n_dU_dlambda_samples: Optional[List[int]],
    target_lambdas: List[float],
    max_sem_kJ_mol: float = 2.0,
    max_propagated_uncertainty_kJ_mol: float = 5.0,
) -> Dict[str, Any]:
    """评估 `estimate_f_k_from_pilot_ti()` 给出的 pilot TI 种子，对某个具体
    窗口（`target_lambdas`）是否**精度**足够，可以被上游当作"跳过在线学习、
    直接尝试冻结验证"（pilot-first）的候选。

    ⚠️ 这只是精度判断（pilot 网格自己的 TI 积分测得多准），不是准确性判断
    （pilot 探针系统的物理环境——通常跟真实窗口环境不完全一样——测到的
    dU/dlambda 是否真的能代表这个窗口）。后一半必须由调用方另外用同一个
    窗口的独立自举 TI 估计（真实 Hamiltonian 下采样）做交叉验证；这个函数
    单独返回 `trustworthy=True` **不足以**允许 pilot-first，只是必要条件
    之一。见 memtodolist 里"窗口预热状态机重构"计划的风险复核结论。

    纯 Python，不依赖 OpenMM，可离线单元测试。永不抛异常——精度数据缺失、
    形状不对、含非有限值时一律 `trustworthy=False`，不当作调用方的 bug，
    也不当作"数据没问题只是精度不够"（旧的、本次改动之前生成的 preopt
    cache 就没有 `std_dU_dlambda_kJ_mol`/`n_derivative_samples` 这两个字段，
    必须能安全地退化成"不可信"而不是报错）。

    Returns
    -------
    dict，键固定为：
      - ``trustworthy``: bool，下面全部检查通过才是 True。
      - ``reason``: str，第一个未通过的检查名；`trustworthy=True` 时是 "ok"。
      - ``propagated_uncertainty_kJ_mol``: float，覆盖这个窗口 λ 跨度的
        pilot 点子集上，对 F(target_hi)-F(target_lo) 做的粗略 trapezoidal
        误差传播估计（`sqrt(sum((0.5*dlambda)^2 * (sem_i^2+sem_{i+1}^2)))`）。
        更早的检查失败时是 ``nan``。
      - ``max_sem_kJ_mol``: float，同一个局部子集里最差的标准误
        （`std_dU_dlambda_kJ_mol / sqrt(n_derivative_samples)`）。同样，
        更早失败时是 ``nan``。
    """
    nan = float("nan")

    def _fail(reason: str) -> Dict[str, Any]:
        return {
            "trustworthy": False,
            "reason": reason,
            "propagated_uncertainty_kJ_mol": nan,
            "max_sem_kJ_mol": nan,
        }

    if not pilot_lambdas or not pilot_mean_dU_dlambda:
        return _fail("missing_pilot_data")
    if not pilot_std_dU_dlambda or not pilot_n_dU_dlambda_samples:
        return _fail("missing_pilot_precision_fields")

    try:
        lam = np.asarray(pilot_lambdas, dtype=float).ravel()
        grad = np.asarray(pilot_mean_dU_dlambda, dtype=float).ravel()
        std = np.asarray(pilot_std_dU_dlambda, dtype=float).ravel()
        n_samples = np.asarray(pilot_n_dU_dlambda_samples, dtype=float).ravel()
    except (TypeError, ValueError):
        return _fail("non_numeric_pilot_data")

    if not (lam.size == grad.size == std.size == n_samples.size) or lam.size < 2:
        return _fail("shape_mismatch_or_too_few_points")
    if not (
        np.all(np.isfinite(lam))
        and np.all(np.isfinite(grad))
        and np.all(np.isfinite(std))
        and np.all(np.isfinite(n_samples))
    ):
        return _fail("non_finite_pilot_data")
    if np.any(n_samples < 1):
        return _fail("zero_sample_pilot_point")

    order = np.argsort(lam)
    lam_sorted = lam[order]
    std_sorted = std[order]
    n_sorted = n_samples[order]

    target = np.asarray(target_lambdas, dtype=float).ravel()
    if target.size == 0 or not np.all(np.isfinite(target)):
        return _fail("invalid_target_lambdas")

    lo, hi = float(lam_sorted[0]), float(lam_sorted[-1])
    target_lo, target_hi = float(np.min(target)), float(np.max(target))
    if target_lo < lo or target_hi > hi:
        # estimate_f_k_from_pilot_ti() 在这种情况下会钳位到边界值当近似——
        # 对"热启动初值"这种用途足够了；但对"直接当冻结候选"，钳位意味着
        # 这段窗口跨度里根本没有真实 pilot 测量，不能算可信。
        return _fail("target_lambdas_require_extrapolation")

    sem = std_sorted / np.sqrt(n_sorted)

    # 取覆盖这个窗口 λ 跨度的最小 pilot 点子集（跨度两端之外各留一个相邻
    # 点，保证跨度边界所在的那一段梯形也被计入），只在这个局部子集上做
    # 误差传播——关心的是这一个窗口自己的 F(target_hi)-F(target_lo) 有多
    # 不确定，不是整条 pilot 曲线的全局不确定度。
    lo_idx = max(0, int(np.searchsorted(lam_sorted, target_lo, side="right")) - 1)
    hi_idx = min(lam_sorted.size - 1, int(np.searchsorted(lam_sorted, target_hi, side="left")))
    if hi_idx <= lo_idx:
        hi_idx = min(lam_sorted.size - 1, lo_idx + 1)

    local_sem = sem[lo_idx : hi_idx + 1]
    d_lam = np.diff(lam_sorted[lo_idx : hi_idx + 1])
    if local_sem.size < 2:
        return _fail("insufficient_local_pilot_coverage")

    max_local_sem = float(np.max(local_sem))
    variance_terms = (0.5 * d_lam) ** 2 * (local_sem[:-1] ** 2 + local_sem[1:] ** 2)
    propagated_uncertainty = float(np.sqrt(np.sum(variance_terms)))

    if max_local_sem > float(max_sem_kJ_mol):
        return {
            "trustworthy": False,
            "reason": "pilot_sem_too_large",
            "propagated_uncertainty_kJ_mol": propagated_uncertainty,
            "max_sem_kJ_mol": max_local_sem,
        }
    if propagated_uncertainty > float(max_propagated_uncertainty_kJ_mol):
        return {
            "trustworthy": False,
            "reason": "propagated_uncertainty_too_large",
            "propagated_uncertainty_kJ_mol": propagated_uncertainty,
            "max_sem_kJ_mol": max_local_sem,
        }

    return {
        "trustworthy": True,
        "reason": "ok",
        "propagated_uncertainty_kJ_mol": propagated_uncertainty,
        "max_sem_kJ_mol": max_local_sem,
    }


def partition_windows_by_delta_f_budget(
    f_k_in_order: np.ndarray,
    max_window_span_kJ: float,
    overlap: int = 2,
) -> List[Tuple[int, int]]:
    """
    按累积 |Δf| 预算切分窗口边界，而不是按 state 数等分（`generate_overlapping_windows`
    那种纯按索引切分完全不知道每一段 λ 实际有多"陡"）。

    贪心地从每个窗口起点尽量往后扩，直到"再加一个点"就会让该窗口跨度超过
    max_window_span_kJ 才停止（提前判断下一步会不会超标，而不是超标之后才发现），
    确保每个窗口自身的能量跨度尽量贴着预算、不会系统性超支。如果单独一步的
    |Δf| 本身就已经超过预算（说明这里的 λ 点还不够密），该窗口会退化为只包含这
    一对相邻点，不会被强行拉宽掩盖问题。

    🔑 用的是逐步 |Δf| 的累积和（总变差/弧长），不是"终点减起点"的净位移——
    f_k(λ) 在软核/WCA shield 存在时不保证单调（尤其是精修探针步数较短、噪声
    较大时更容易出现局部反复），如果用净位移判断，一段先涨后跌又绕回起点附近
    的区间会被误判成"几乎没变化"，导致该窗口被贪心地拉得异常宽（曾实测出现单
    个窗口吞掉 7 个态、跟前后 3 态一组的窗口极不协调）。用累积和可以保证任何
    真实的往返波动都会被如实计入预算，不会被净位移抵消掩盖。
    """
    f_k_in_order = np.asarray(f_k_in_order, dtype=float)
    n = len(f_k_in_order)
    if n <= 2:
        return [(0, n)]

    overlap = max(1, int(overlap))
    # cum[i] = 从 f_k[0] 到 f_k[i] 逐步 |Δf| 的累积和（总变差），cum[j] - cum[i]
    # 即区间 [i, j] 内实际"走过"的能量距离，而不是端点净位移。
    cum = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(f_k_in_order)))))
    windows = []
    start = 0
    while start < n - 1:
        end = start + 1
        while end + 1 < n and (cum[end + 1] - cum[start]) <= max_window_span_kJ:
            end += 1
        windows.append((start, end + 1))
        if end >= n - 1:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1  # 保证每轮都严格前进，避免死循环
        start = next_start

    # 清理没有带来新覆盖范围的冗余窗口：如果某个窗口因为一开始就撞到预算上限
    # 而提前收尾、右端点没有超过前一个窗口的右端点，它对拼接毫无帮助（完全被
    # 前一个窗口包含），直接丢弃，避免因为固定的 overlap 步长在预算吃紧的区域
    # 里反复产生"原地踏步"的窗口。
    merged = [windows[0]]
    for s, e in windows[1:]:
        if e <= merged[-1][1]:
            continue
        merged.append((s, e))
    return merged


def refine_stage_lambda_path_by_overlap(
    lambdas_var: List[float],
    window_ranges: List[Tuple[int, int]],
    window_overlap_diagnostics: List[Dict],
    min_overlap_threshold: float,
    pts_per_window: int = 6,
    overlap: int = 2,
) -> Tuple[Optional[List[float]], Optional[List[Tuple[int, int]]]]:
    """
    数据驱动地在重叠不足的地方加密 λ 点 —— 用的是这次采样*已经算出来*的
    per-window ESS (有效样本数) 重叠诊断，不是拍脑袋的固定间距或手写 λ 值。

    为什么不能复用 partition_windows_by_delta_f_budget/redistribute_lambda_by_delta_f：
    那条路径把"Δf 曲线陡不陡"当成重叠的代理指标，但 GlobalMBARAnalyzer.
    solve_stage_integrated 自己的审查报告已经指出这只是代理、不是真重叠——一个
    窗口可以 Δf 很平滑但仍然因为 IBS 偏置没收敛/构象弛豫慢等原因导致真实的
    reweight 有效样本比例（ess_ratio）很差。abfe_pipeline._assert_stage_result_sane
    用的正是后者（min_overlap/min_overlap_threshold），所以这里的加密逻辑也必须
    直接读同一个 ess_ratio 诊断，而不是去看 Δf 曲线。

    做法：对每个 min_ess_ratio < 阈值 的窗口，从它自带的 ess_ratio_per_lambda
    （每个目标 λ 态各自的有效样本比例，见 ibs_engine.py solve_stage_integrated）
    里找出全窗口最差的那个 λ 态，在它两侧（窗口内)较宽的那个物理 λ 间隔上插入
    一个新的中点 —— 更宽的间隔更可能是重叠瓶颈。多个窗口同时不达标时会分别
    处理、去重合并。窗口边界不手工指定，插入新点后统一交给
    generate_overlapping_windows 按现有约定（pts_per_window/overlap，跟这个流水线
    别处用的常量一致）重新切分，避免手工窗口边界产生索引错位。

    返回 (None, None) 表示诊断里找不到任何低于阈值、且带有 ess_ratio_per_lambda
    明细的窗口 —— 调用方应该把这当成"自动修复无法定位问题"，而不是继续盲目重试。
    """
    lambdas_var = list(lambdas_var)
    gaps_to_bisect = set()

    for rec in window_overlap_diagnostics or []:
        ratio = rec.get("min_ess_ratio")
        per_lambda = rec.get("ess_ratio_per_lambda")
        win_lams = rec.get("lambdas") or []
        if ratio is None or ratio >= min_overlap_threshold:
            continue
        if not per_lambda or len(win_lams) < 2:
            continue

        worst_lambda_idx = min(per_lambda, key=lambda k: per_lambda[k])
        worst_lambda_idx = int(worst_lambda_idx)
        if worst_lambda_idx not in win_lams:
            continue
        pos = win_lams.index(worst_lambda_idx)

        candidates = []
        if pos > 0:
            candidates.append((win_lams[pos - 1], win_lams[pos]))
        if pos < len(win_lams) - 1:
            candidates.append((win_lams[pos], win_lams[pos + 1]))
        if not candidates:
            continue

        # 两个相邻间隔里，物理 λ 跨度更宽的那个更可能是重叠瓶颈。
        lo, hi = max(
            candidates,
            key=lambda ab: abs(lambdas_var[ab[0]] - lambdas_var[ab[1]]),
        )
        gaps_to_bisect.add((min(lo, hi), max(lo, hi)))

    if not gaps_to_bisect:
        return None, None

    new_lambdas = list(lambdas_var)
    # 从高索引往低索引插入，这样前面插入不会打乱还没处理的间隔的索引。
    for lo, hi in sorted(gaps_to_bisect, key=lambda ab: -ab[0]):
        if hi != lo + 1:
            # 不是相邻的一对（例如上一轮已经在中间插过点导致索引偏移），跳过，
            # 交给下一轮基于新诊断重新定位，而不是插到错误的位置。
            continue
        midpoint = (lambdas_var[lo] + lambdas_var[hi]) / 2.0
        new_lambdas.insert(hi, midpoint)

    if len(new_lambdas) == len(lambdas_var):
        return None, None

    new_window_ranges = generate_overlapping_windows(
        n_states=len(new_lambdas), pts_per_window=pts_per_window, overlap=overlap
    )
    return new_lambdas, new_window_ranges


def refine_stage_lambda_path_from_data(
    stage_dir: str,
    preopt_path: str,
    temperature_k: float = 300.0,
    n_states: Optional[int] = None,
    max_window_span_kJ: float = 35.0,
    overlap: int = 2,
    stage_type: str = "vdw",
) -> Dict:
    """
    用该 stage 已经真实采集到的窗口能量数据，重新设计 λ 分布与窗口边界：
    - λ 点按累积 |Δf|（实测自由能曲线弧长）等分，不是等 λ 间距，也不是被压缩过的
      方差代理。
    - 窗口边界按累积 |Δf| 预算切分，不是按 state 数等分。
    所有数字都从这次真实采样数据现场算出来，不手写任何"魔法数字"。

    直接读取 preopt_path 里现有的 lambdas_var/window_ranges 去定位、加载已有窗口
    能量文件，用 GlobalMBARAnalyzer.solve_stage_integrated（已修复 β 换算）求出当前
    真实 f(λ) 曲线，再基于这条曲线重新设计。旧文件会先备份为 `<preopt_path>.bak`，
    新方案覆盖写回原路径。

    注意：这一步只能"重新规划下一轮该怎么采样"，不能凭空补全还没跑过的数据——
    重新规划后的窗口大多数会跟旧窗口边界不一致，下次 resume 时会被判定为形状不
    匹配、重新采样，这是预期行为，不是 bug。
    """
    from ibs_engine import solve_stage_integrated

    with open(preopt_path, "r", encoding="utf-8") as f:
        preopt = json.load(f)
    lambdas_var = preopt["lambdas_var"]
    window_ranges = preopt["window_ranges"]

    # 🔑 [P1-15] 从文件名解析**真实**窗口编号，按数值排序——此前
    # `sorted(glob.glob(...))` 是字典序，窗口数达到两位数时 window_10/window_11
    # 会排在 window_2 之前，再用 enumerate 的位置当窗口编号就会把 u_kn/bias/
    # base 与 window_ranges 错配，写出错误的新 λ 路径。与
    # runabfe._analyze_dual_leg / abfe_pipeline 清理窗口产物用的是同一套正则
    # `dual_window_(\d+)_{stage_type}_energies\.npy`；编号必须从 0 连续到 N-1，
    # 重复或缺失一律拒绝（不能悄悄错配）。
    _window_idx_re = re.compile(rf"dual_window_(\d+)_{stage_type}_energies\.npy$")
    indexed_e_files = []
    for e_file in glob.glob(os.path.join(stage_dir, f"dual_window_*_{stage_type}_energies.npy")):
        match = _window_idx_re.search(os.path.basename(e_file))
        if not match:
            raise RuntimeError(
                f"无法从文件名解析窗口编号（期望 dual_window_<int>_{stage_type}_energies.npy）: "
                f"{e_file}"
            )
        indexed_e_files.append((int(match.group(1)), e_file))
    indexed_e_files.sort(key=lambda pair: pair[0])
    parsed_indices = [idx for idx, _ in indexed_e_files]
    if parsed_indices != list(range(len(window_ranges))):
        raise RuntimeError(
            f"窗口能量文件编号（解析得到 {parsed_indices}）与 preopt 缓存里的 "
            f"window_ranges 数 ({len(window_ranges)}) 不一致（要求从 0 连续编号），"
            "无法基于现有数据重新设计路径；"
            "请先确认该 stage 的采样已经完整跑完（每个窗口都有对应的 "
            f"dual_window_<int>_{stage_type}_energies.npy，且没有重复/缺失编号）。"
        )

    window_data = []
    for w_idx, (_parsed_idx, e_file) in enumerate(indexed_e_files):
        u_kn = np.load(e_file)
        bias = np.load(e_file.replace("_energies.npy", "_bias.npy"))
        base = np.load(e_file.replace("_energies.npy", "_base.npy"))
        start, end = window_ranges[w_idx]
        window_data.append({
            "u_kn": u_kn,
            "bias_energies": bias,
            "base_energies": base,
            "lambda_indices": list(range(start, end)),
        })

    kt = 0.008314462618 * float(temperature_k)
    res = solve_stage_integrated(window_data, kt, stage_name=stage_type)
    if res.get("error"):
        raise RuntimeError(f"基于现有数据求解当前 f(λ) 曲线失败: {res['error']}")

    lambdas_sorted = res["lambdas"]
    f_k = np.asarray(res["f_k"], dtype=float)
    lam_in_order = np.asarray([lambdas_var[i] for i in lambdas_sorted], dtype=float)

    n_new = int(n_states or len(lambdas_var))
    new_lambdas = redistribute_lambda_by_delta_f(lam_in_order, f_k, n_new)

    # 用旧曲线插值出新 λ 点对应的 f 值，仅用于指导窗口切分。
    interp_order = np.argsort(lam_in_order)
    f_at_new = np.interp(new_lambdas, lam_in_order[interp_order], f_k[interp_order])
    new_windows = partition_windows_by_delta_f_budget(f_at_new, max_window_span_kJ, overlap=overlap)

    covered = sorted({i for s, e in new_windows for i in range(s, e)})
    if covered != list(range(n_new)):
        raise RuntimeError(
            f"内部错误：新窗口划分未能完整覆盖 [0,{n_new})，拒绝写出（覆盖到 {covered}）。"
        )

    new_preopt = {
        "lambdas_var": [float(x) for x in new_lambdas],
        "window_ranges": [[int(s), int(e)] for s, e in new_windows],
        "n_states": n_new,
        "provenance": {
            "source": "refine_stage_lambda_path_from_data",
            "based_on_measured_f_curve": True,
            "max_window_span_kJ_mol": float(max_window_span_kJ),
            "prior_n_states": len(lambdas_var),
            "prior_window_ranges": [list(w) for w in window_ranges],
            "prior_total_delta_G_kJ_mol": float(res.get("total_delta_G", float("nan"))),
            "prior_min_overlap": res.get("min_overlap"),
        },
    }

    backup_path = preopt_path + ".bak"
    shutil.copy(preopt_path, backup_path)
    with open(preopt_path, "w", encoding="utf-8") as f:
        json.dump(new_preopt, f, indent=2)

    return new_preopt


def _sample_group1_energies(context, total_steps, sample_interval=50):
    """批量推进积分器，保留固定采样间隔，减少 Python/C++ 边界往返。"""
    if total_steps <= 0:
        return []

    integrator = context.getIntegrator()
    energies = []
    full_batches, remainder = divmod(int(total_steps), int(sample_interval))

    for _ in range(full_batches):
        guarded_step(integrator, sample_interval, "Group1 能量采样")
        state = context.getState(getEnergy=True, groups={1})
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    if remainder:
        guarded_step(integrator, remainder, "Group1 能量采样（余数段）")
        state = context.getState(getEnergy=True, groups={1})
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    return energies

class ABFEPreOptimizer:
    """ABFE 预采样优化器 - ACES 路径优化版
    【修复 3】绑定 context 生命周期，不再单独保存 system
    """

    def __init__(
        self,
        system: openmm.System,
        context: openmm.Context,
        lambdas: List[float],
        temperature: float = 300.0,
        target_phase: str = "auto",
    ):
        # ✅ 修复 3: 只通过 context 获取 system，确保生命周期一致
        self.context = context
        self.lambdas = np.array(lambdas)
        self.n_states = len(lambdas)
        self.temperature = temperature

        # ✅ 修复 5: 保存 lambdas 为实例变量
        self.original_lambdas = lambdas.copy()  # 修复变量名错误

        # 结果存储
        self.optimized_params = {}
        self.initial_weights = None
        self.energy_history = []
        self.lambda_density = None

        # 【新增】最优衰减指数
        self.optimal_charge_exponent = 2.0
        self.optimal_vdw_exponent = 1.0
        self.boresch_params = None
        self.optimized_params["lambda_path"] = {}
        
        # #[P1 FIX] 动态探测 lambda 参数名，避免硬编码导致的更新失效问题
        # 🔑 [0831issue P2] `target_phase` 现在可显式指定。默认 "auto" 的优先级表把
        # `lam_coul` 排在 `lam_vdw` 前面（见 _detect_active_parameter），所以对一个
        # **同时注册了两个轴**的探针系统，vdW/vanishing 阶段会沿着 λ_coul 那根轴去
        # 测方差、路径密度权重与目标阶段完全不对应。构造时就知道自己是哪个阶段的
        # 调用方应显式传 "vdw"/"vanishing" 或 "coul"/"decharging"；不传则保持
        # 原来的 auto 行为，逐位不变。
        self.target_phase = str(target_phase or "auto")
        self._active_lambda_param = self._detect_active_parameter(self.target_phase)

    def _detect_active_parameter(self, target_phase: str = "auto") -> str:
        """探测系统中实际使用的 Lambda 参数名（支持双λ动态优先级）"""
        params = []
        try:
            system = self.context.getSystem()
            # 1. 获取 System 级别的全局参数
            for i in range(system.getNumGlobalParameters()):
                params.append(system.getGlobalParameterName(i))
            # 2. 获取所有 Force 级别的全局参数 (CustomForce 通常将 lambda 挂载在此)
            for i in range(system.getNumForces()):
                force = system.getForce(i)
                if hasattr(force, 'getNumGlobalParameters'):
                    for j in range(force.getNumGlobalParameters()):
                        params.append(force.getGlobalParameterName(j))
        except Exception as e:
            print(f"  [WARN] 获取系统参数失败: {e}，探针系统可能未正确注入软核力。")
            return "lam_coul"

        # ✅ 核心修复：根据目标阶段动态调整匹配优先级
        if target_phase.lower() in ("vdw", "vanishing"):
            priority_order = ["lam_vdw", "lambda_vdw", "lam_coul", "lambda_coul"]
        elif target_phase.lower() in ("coul", "decharging"):
            priority_order = ["lam_coul", "lambda_coul", "lam_vdw", "lambda_vdw"]
        else:
            # auto 模式：优先返回已存在的任意 λ 参数
            priority_order = ["lam_coul", "lam_vdw", "lambda_coul", "lambda_vdw"]

        # 1. 精确匹配
        for name in priority_order:
            if name in params:
                print(f"  探测到有效 Lambda 参数: '{name}' (phase={target_phase})")
                return name

        # 2. 模糊匹配（含关键字的候选）
        coul_candidates = [p for p in params if "coul" in p.lower() and "lam" in p.lower()]
        vdw_candidates  = [p for p in params if ("vdw" in p.lower() or "lj" in p.lower()) and "lam" in p.lower()]

        if target_phase.lower() in ("vdw", "vanishing") and vdw_candidates:
            return vdw_candidates[0]
        if coul_candidates:
            return coul_candidates[0]

        # 3. 通用/历史别名回退
        for name in ["lam", "lambda", "lambda1", "LIG_lambda", "lig_lambda"]:
            if name in params:
                print(f"  探测到通用 Lambda 参数: '{name}'")
                return name

        # 4. 终极兜底
        lam_params = [p for p in params if "lam" in p.lower()]
        if lam_params:
            print(f"  模糊匹配到 Lambda 参数: '{lam_params[0]}'")
            return lam_params[0]

        raise RuntimeError("系统中找不到有效的 Lambda 参数名，请检查探针系统构建逻辑")

    def analyze_gradient_and_optimize_path(self, n_steps_per_state: int = 5000) -> Dict:
        """
        [步骤 1.5] 轻量级能量景观分析 (Pathfinding)
        【修复 4】安全参数设置

        🚨 [0831issue P2] **已禁用，与 PHY-08 对 `optimize_stage1_decharging` 的处置
        完全同源同理由**：本方法用 `Var(U_group1)` 当度量（见下方 `np.var(energies)`），
        而生产 PME Hamiltonian 的 Fisher 度量是 `beta² Var[dU/dλ]`。
        `Var(U)` 里混着大量 λ 无关的环境涨落（总势能被溶剂主导），据它排出来的 λ 路径
        不能用于热力学采样。PHY-08 当时只禁掉了同模式的 Stage-1 入口，漏了这个
        同样公开的姐妹入口。

        到达性已核实：唯一调用者是 `abfe_pipeline.ABFEPipeline.run_preoptimization`，
        而 `run_preoptimization` **全仓库没有任何调用者**（连测试都没有）；另一个是
        本类的兼容包装 `run_probing_sampling`。所以这道 fail-closed 不影响任何生产路径，
        作用是让将来复活这条路的人先把度量换成 `beta² Var[dU/dλ]`
        （本类 `_sample_scalar_metric` 已有正确实现，冻结构型有限差分 + force group 隔离）。
        """
        raise RuntimeError(
            "轻量能量景观分析已禁用（0831issue P2 / 同 PHY-08）：旧实现使用 Var(U_group1) "
            "而非生产 PME 的 beta² Var[dU/dlambda]，其路径不能用于热力学采样。"
            "正确度量见 ABFEPreOptimizer._sample_scalar_metric；"
            "生产流程请使用已验证的线性/测地线路径。"
        )

        print(f"\n→ 正在执行能量景观分析 ({n_steps_per_state} 步/状态)... ")

        # === 【修复 4】安全检查参数是否存在 ===
        # abfe_preoptimizer.py -> analyze_gradient_and_optimize_path 方法 (约第 120 行)

        # === 【修复】安全参数设置：先检查后设置 ===
        initial_lam = float(self.lambdas[0])
        active_p = self._active_lambda_param
        param_exists = False
        
        try:
            params_dict = self.context.getParameters()
            if active_p in params_dict:
                self.context.setParameter(active_p, initial_lam)
                param_exists = True
            else:
                # 🔑 修复：兼容别名回退并同步局部变量
                for param_name in ["lambda", "lambda1", "lambda_vdw", "lambda_coul"]:
                    if param_name in params_dict:
                        self.context.setParameter(param_name, initial_lam)
                        self._active_lambda_param = param_name
                        active_p = param_name  # ✅ 关键：同步更新循环使用的变量名
                        param_exists = True
                        print(f"  已回退至 Lambda 别名: '{param_name}'")
                        break
        except (openmm.OpenMMException, AttributeError) as e:
            print(f"  [WARN] 参数 '{active_p}' 设置失败: {e}")
            # ✅ 修复：不 pass，记录失败并尝试强制注入常见名称
            for p_name in list(self.context.getParameters().keys()):
                if "lam" in p_name.lower():
                    try:
                        self.context.setParameter(p_name, initial_lam)
                        active_p = p_name
                        param_exists = True
                        print(f"  [OK] 强制注入成功: {p_name}")
                        break
                    except (openmm.OpenMMException, AttributeError, TypeError, ValueError):
                        continue

        if not param_exists:
            raise RuntimeError(f"[ERR] 无法在 Context 中找到或设置任何 Lambda 参数，优化终止。")
            
        if param_exists:
            print(f"  设置初始 {active_p}={initial_lam:.2f} 进行预平衡...")
            guarded_step(self.context.getIntegrator(), 25000, "单 λ 路径优化：初始预平衡")

        variance_data = []
        mean_energy = []

        # 主采样循环
        for i, lam in enumerate(self.lambdas):
            try:
                self.context.setParameter(active_p, float(lam))  # ✅ 此时 active_p 已是有效名称
            except openmm.OpenMMException as e:
                print(f"  [ERR] 无法设置 Lambda={lam:.3f}: {e}。采样中断。")
                raise

            # 先平衡 500 步再采样
            guarded_step(self.context.getIntegrator(), 500, "单 λ 路径优化：采样前平衡")

            energies = []
            nan_count = 0
            n_sampled = 0

            # 🔑 [0831issue P2 / PHY-08 同类] NaN/Inf 样本必须**丢弃**，不能替换成
            # "前一帧的值"或 0.0 再计入方差。旧写法把坏帧换成前值后照样 append，于是
            # (a) 方差被人为压低（重复值零离差），(b) 首帧就坏时注入一个纯虚构的 0.0，
            # 两者都直接歪曲这个度量，而它是 λ 路径密度的唯一依据。
            # nan_count/n_sampled 的比例判据保持原语义（分母仍是总采样帧数）。
            for e in _sample_group1_energies(self.context, n_steps_per_state, sample_interval=50):
                n_sampled += 1
                if np.isnan(e) or np.isinf(e):
                    nan_count += 1
                    continue
                energies.append(e)

            if nan_count > n_sampled * 0.5:
                print(
                    f"  [WARN] lam={lam:.2f} 能量异常过多 ({nan_count}/{n_sampled})，使用默认值 "
                )
                variance_data.append(1.0)
                mean_energy.append(0.0)
            elif len(energies) > 1:
                variance = np.var(energies)
                if np.isnan(variance):
                    variance = 1.0
                variance_data.append(variance)
                mean_energy.append(np.mean(energies))
            else:
                variance_data.append(1.0)
                mean_energy.append(energies[0] if energies else 0.0)

        variance_data = np.array(variance_data)
        std_dev = np.sqrt(variance_data + 1e-10)

        # === 【修复 4】方差截断 (防止异常值主导) ===
        threshold = np.percentile(std_dev, 90) * 2.0  # ✅ 从 3.0 改为 2.0 更保守
        if np.isnan(threshold):
            threshold = 10.0

        std_dev_clipped = np.clip(std_dev, None, threshold)
        norm_variance = std_dev_clipped / (np.max(std_dev_clipped) + 1e-6)

        print(
            f"  [OK] 能量景观分析完成。最大标准差位置：lam={self.lambdas[np.argmax(std_dev)]:.2f} "
        )
        print(f"  [OK] 方差截断阈值：{threshold:.2f} (原始最大：{np.max(std_dev):.2f}) ")

        return {
            "variance": variance_data,
            "std_dev": std_dev,
            "std_dev_clipped": std_dev_clipped,
            "norm_variance": norm_variance,
            "mean_energy": mean_energy,
        }

    def optimize_softcore_parameters(
        self, ligand_indices: List[int]
    ) -> ACESoftcorePotential:
        """[步骤 1] 优化软核参数"""
        n_ligand_atoms = len(ligand_indices)
        params = ACESoftcorePotential.optimize_alpha(n_ligand_atoms)

        softcore_obj = ACESoftcorePotential(
            alpha_lj=params["alpha_lj"],
            alpha_coul=params["alpha_coul"],
            power_lj=params["power_lj"],
            power_coul=params["power_coul"],
        )

        self.optimized_params["softcore"] = softcore_obj
        print(
            f"→ 软核参数已优化：α_LJ={softcore_obj.alpha_lj}, α_Coul={softcore_obj.alpha_coul}"
        )

        return softcore_obj

    def generate_lambda_path(
        self, phase: str = "vdw", n_windows: int = 4, states_per_window: int = 10
    ) -> List[float]:
        """
        [步骤 2 备选] Lambda 路径预设（如果不用 Pathfinding）
        【修复】添加此方法以兼容 openmm_abfe_pipeline.py
        """
        total_states = n_windows * states_per_window

        if phase == "vdw":
            lambdas = (np.linspace(1.0, 0.0, total_states) ** 2).tolist()
            print(f"→ VdW 阶段：生成 {total_states} 状态非线性 Lambda 路径 (λ²)")
        elif phase == "charge":
            lambdas = np.linspace(1.0, 0.0, total_states).tolist()
            print(f"→ 电荷阶段：生成 {total_states} 状态线性 Lambda 路径")
        else:
            lambdas = np.linspace(1.0, 0.0, total_states).tolist()

        self.optimized_params["lambda_path"] = {
            "phase": phase,
            "n_windows": n_windows,
            "states_per_window": states_per_window,
            "distribution": "nonlinear" if phase == "vdw" else "linear",
        }

        self.lambdas = np.array(lambdas)
        self.n_states = len(lambdas)

        return lambdas

    def optimize_window_ranges(
        self, n_ib_windows: int = 4, overlap: int = 3
    ) -> List[Tuple[int, int]]:
        """[步骤 3] IBS 窗口划分"""
        total = self.n_states

        if total <= 6 or n_ib_windows == 1:
            ranges = [(0, total)]
            print(f"→ 状态数较少 ({total})，使用单窗口：{ranges}")
            return ranges

        min_window_size = overlap + 1
        if total < n_ib_windows * min_window_size:
            n_ib_windows = max(1, total // min_window_size)
            print(f"  [WARN] 状态数不足，调整窗口数为 {n_ib_windows}")

        if n_ib_windows > 1:
            step = (total - overlap) // n_ib_windows
            step = max(1, step)
        else:
            step = total

        ranges = []
        for i in range(n_ib_windows):
            start = i * step
            if start >= total:
                break

            if i < n_ib_windows - 1:
                end = start + step + overlap
            else:
                end = total

            end = min(end, total)
            if end > start:
                ranges.append((start, end))

            if end == total:
                break

        if not ranges or ranges[-1][1] < total:
            ranges = []
            simple_step = max(1, (total - overlap) // n_ib_windows)
            for i in range(n_ib_windows):
                s = i * simple_step
                if i < n_ib_windows - 1:
                    e = s + simple_step + overlap
                else:
                    e = total
                e = min(e, total)
                if s < e:
                    ranges.append((s, e))
            if ranges and ranges[-1][1] < total:
                ranges[-1] = (ranges[-1][0], total)

        print(f"→ ACES 建议窗口划分 ({len(ranges)} 个): {ranges}")
        return ranges

    def optimize_window_ranges_for_ibes(
        self, n_ib_windows: int = 3, overlap: int = 4
    ) -> List[Tuple[int, int]]:
        """[步骤 3 备选] IBS 窗口划分（别名）"""
        return self.optimize_window_ranges(n_ib_windows=n_ib_windows, overlap=overlap)

    def get_optimization_report(self) -> Dict:
        """获取优化报告"""
        softcore_dict = {}
        if "softcore" in self.optimized_params:
            sc = self.optimized_params["softcore"]
            if hasattr(sc, "alpha_lj"):
                softcore_dict = sc.get_parameters_dict()

        return {
            "temperature": float(self.temperature),
            "n_states": int(self.n_states),
            "softcore_params": softcore_dict,
            "lambda_path": self.optimized_params.get("lambda_path", {}),
            "initial_weights": self.initial_weights.tolist()
            if self.initial_weights is not None
            else None,
            "boresch_correction": self.boresch_params.get("analytical_correction")
            if self.boresch_params
            else None,  # ✅ 现在可以安全访问
        }

    # =============================================================================
    # 替换 optimize_lambda_path_adaptive 方法 (完整修复版)
    # =============================================================================
    def optimize_lambda_path_adaptive(
        self,
        landscape_data,
        target_n_states: int = None,
        charge_exponent: float = 2.0,
        vdw_exponent: float = 1.0,
    ) -> List[float]:
        """
        [步骤 1.6] 根据能量方差自适应调整 Lambda 分布

        【关键修复】
        1. 确保插值前 Lambda 序列转为升序 (np.interp 要求 xp 递增)
        2. 使用对数平滑方差，防止单个点主导
        3. 强制边界为 1.0 和 0.0
        4. 添加最小间距检查，防止负数
        """
        if "lambda_path" not in self.optimized_params:
            self.optimized_params["lambda_path"] = {}

        # === 目标状态数处理 ===
        if target_n_states is None:
            target_n_states = self.n_states

        # 【修复】确保 target_n_states 至少为 12
        if target_n_states < 12:
            print(f"  [WARN] 目标状态数 ({target_n_states}) 太少，调整为 12 ")
            target_n_states = 12

        # === 检查 landscape_data 有效性 ===
        if landscape_data is None or landscape_data.get("std_dev_clipped") is None:
            print("  [WARN] landscape_data 无效，使用线性 Lambda 路径 ")
            return np.linspace(1.0, 0.0, target_n_states).tolist()

        # === 【步骤 1】获取方差数据 ===
        std_dev_clipped = landscape_data["std_dev_clipped"].copy()

        # === 【步骤 2】长度检查与对齐 ===
        if len(std_dev_clipped) != len(self.lambdas):
            print(
                f"[WARN] 警告：std_dev_clipped 长度 ({len(std_dev_clipped)}) 与 self.lambdas 长度 ({len(self.lambdas)}) 不匹配 "
            )
            min_len = min(len(std_dev_clipped), len(self.lambdas))
            std_dev_clipped = std_dev_clipped[:min_len]
            self.lambdas = self.lambdas[:min_len]
            self.n_states = min_len

        # === 【步骤 3】方差平滑 (对数化防止极值主导) ===
        # 【关键修复】使用 log1p 平滑，缓解λ=1.0 处 Clash 带来的极值影响
        try:
            from scipy.ndimage import gaussian_filter1d

            std_dev_smooth = gaussian_filter1d(std_dev_clipped, sigma=1)
            std_dev_smooth[0] = std_dev_clipped[0]
            std_dev_smooth[-1] = std_dev_clipped[-1]
            std_dev_clipped = std_dev_smooth
        except ImportError:
            pass

        # === 【步骤 4】计算密度权重 (使用对数缩放 + 软归一化) ===
        # ✅ 修复10：使用对数缩放 + 软归一化，避免硬截断破坏概率密度
        MAX_RATIO = 0.10  # ✅ 显式声明，避免后续引用报错
        log_std = np.log1p(std_dev_clipped + 1e-6)
        # 归一化为概率密度，保证 ∫ρ(x)dx = 1
        density_weight = log_std / (np.sum(log_std) + 1e-10)
        
        # 保留高λ区加密逻辑
        for i, lam in enumerate(self.lambdas):
            if lam > 0.8:
                density_weight[i] *= 1.5
        # 重新归一化
        density_weight /= np.sum(density_weight)

        # === 【步骤 7】累积分布与插值 ===
        # 🔑 [0831issue P2] 构造与 lambda 节点一一对应的单调 CDF：首节点 0、末节点 1。
        # 旧写法是 `xp = [0] + cumsum(w)[:-1]/sum(w)` 然后把末元素**覆盖**成 1.0 —— 那个赋值
        # **覆盖**掉了倒数第二个累积坐标 c_{N-2}/T，于是最后一个区间的宽度从
        # w[N-2] 变成 w[N-2]+w[N-1]，λ[N-2] 的权重被双重计入，λ→0 尾段的加密方向失真。
        # 正解：N 个节点之间只有 N-1 个区间，就用前 N-1 个权重当区间宽度、并按
        # **它们自己的和**归一化——末端于是天然等于 1.0，不需要事后覆盖。
        interval_weights = np.asarray(density_weight, dtype=float)[:-1]
        interval_total = max(1e-10, float(np.sum(interval_weights)))
        xp = np.concatenate(([0.0], np.cumsum(interval_weights) / interval_total))

        # 原始 lambdas 是降序 [1.0, ..., 0.0]，长度必须与 xp 严格一致。
        original_lambdas = np.asarray(self.lambdas.copy(), dtype=float)
        lambda_xp = original_lambdas

        if HAS_SCIPY and len(xp) >= 3:
            # ✅ 使用 PCHIP 保持单调性，无需手动翻转
            # 注意：xp 递增，original_lambdas 递减 → 插值函数自动处理反向映射
            interp_func = PchipInterpolator(xp, lambda_xp, extrapolate=False)
            target_cumulative = np.linspace(0, 1.0, target_n_states)
            optimized_lambdas = interp_func(target_cumulative)
        else:
            # 回退到原逻辑 (带翻转)
            target_cumulative = np.linspace(0, 1.0, target_n_states)
            unique_xp, idx_map = np.unique(xp, return_index=True)
            fp_filtered = lambda_xp[idx_map]
            xp = unique_xp
            if len(unique_xp) < 2:
                return np.linspace(1.0, 0.0, target_n_states).tolist()
            optimized_lambdas = np.interp(target_cumulative, xp, fp_filtered)

        optimized_lambdas = np.asarray(optimized_lambdas, dtype=float).ravel()
        if not np.all(np.isfinite(optimized_lambdas)):
            print("  [WARN] 自适应插值产生非有限 λ，使用线性路径")
            optimized_lambdas = np.linspace(1.0, 0.0, target_n_states)

        # === 【步骤 8】边界强制、最小间距与去重 (共享纯函数，见
        # finalize_descending_lambda_path；DualLambdaPreOptimizer.
        # optimize_stage1_decharging 复用同一份逻辑) ===
        optimized_lambdas, min_spacing, fell_back = finalize_descending_lambda_path(
            optimized_lambdas, target_n_states
        )
        if fell_back:
            print(f"  [WARN] 去重后状态数少于目标 ({target_n_states})，使用线性路径 ")

        if not (np.isclose(optimized_lambdas[0], 1.0) and np.isclose(optimized_lambdas[-1], 0.0)):
            raise RuntimeError(
                f"优化后的 lambda 路径端点异常: first={optimized_lambdas[0]}, last={optimized_lambdas[-1]}"
            )

        # === 【步骤 9】更新状态 ===
        self.optimized_params["lambda_path"].update(
            {
                "method": "adaptive_variance_v6",
                "n_states": len(optimized_lambdas),
                "target_n_states": target_n_states,
                "log_scaling": True,
                "max_ratio": MAX_RATIO,
                "min_spacing": min_spacing,
            }
        )

        self.lambdas = np.array(optimized_lambdas)
        self.n_states = len(optimized_lambdas)

        # === 输出诊断信息 ===
        print(f"→ Lambda 路径已优化。高方差区已加密。 ")
        print(
            f"  Lambda 范围：[{np.min(optimized_lambdas):.3f}, {np.max(optimized_lambdas):.3f}] "
        )
        print(f"  总状态数：{len(optimized_lambdas)} (目标：{target_n_states}) ")
        print(f"  Lambda 间距：{np.diff(optimized_lambdas)} ")

        # 【关键验证】检查是否有负数
        if np.any(optimized_lambdas < 0.0):
            print(f"  [WARN] 警告：检测到负 Lambda 值，已修正 ")
        if np.any(optimized_lambdas > 1.0):
            print(f"  [WARN] 警告：检测到 Lambda>1.0，已修正 ")

        # === 确保返回 list ===
        return optimized_lambdas.tolist()

    # 在 ABFEPreOptimizer 类中添加以下兼容方法

    def run_probing_sampling(self, n_steps: int = 5000) -> Dict:
        """【兼容方法】调用 analyze_gradient_and_optimize_path"""
        return self.analyze_gradient_and_optimize_path(n_steps_per_state=n_steps)

    def optimize_path(self, landscape_data: Dict) -> List[float]:
        """【兼容方法】调用 optimize_lambda_path_adaptive"""
        return self.optimize_lambda_path_adaptive(landscape_data=landscape_data)

    # =============================================================================
    # 在 ABFEPreOptimizer 类中添加窗口划分方法
    # =============================================================================
    # 替换原 partition_ibs_windows_fixed 方法体为：
    def partition_ibs_windows_fixed(self, n_states: int = None, n_ib_windows: int = 4, pts_per_window: int = 6, overlap: int = 2) -> List[Tuple[int, int]]:
        if n_states is None: n_states = self.n_states
        windows = generate_overlapping_windows(
            n_states,
            pts_per_window=pts_per_window,
            overlap=overlap,
            n_windows=n_ib_windows,
        )
        print(f"→ IBS 窗口划分 ({len(windows)} 个): {windows} (覆盖 {n_states} 个状态)")
        return windows


# =============================================================================
# λ 路径 pilot 探针 shadow early-stop 诊断（Phase A，2026-08-26）
# =============================================================================
# 下面这组是纯 Python/numpy 函数，不依赖 OpenMM Context，可离线单测。目的是
# 回答"Stage2 vanishing pilot 的 n_steps_per_state=30000 是不是处处都要跑
# 满"——但本阶段（Phase A）只做诊断/记录，不改变任何真实采样长度：
# `_sample_scalar_metric`/`optimize_stage2_vanishing`/
# `_refine_pilot_grid_in_steep_segments` 在 shadow_checkpoint_steps /
# shadow_checkpoint_interval 为 None（默认值）时逐字节保持原行为不变。
#
# 背景（详见 abfe_pipeline.py 里 "vanishing" 分支调用 _run_dual_lambda_
# optimization 处 2026-07-19 的原地注释）：那次真实 GPU 回归发现 10000 步的
# 短 pilot 会系统性低估 λ≈1 端点由稀有/发作性事件主导的
# beta^2*Var[dU/dlambda]，才把预算拉长到当前生产用的 30000。任何缩短 pilot
# 预算的方案都必须先证明不会重新踩这个坑——这组函数只是用来在真机上收集
# "如果提前停会怎样"的影子数据供之后离线验证，本身不做任何提前停的决定。


def _pilot_segment_lengths(pilot_lambdas, metric_g) -> np.ndarray:
    """相邻 pilot 点之间的热力学长度 ``0.5*(sqrt(g_i)+sqrt(g_{i+1}))*|dλ|``。

    从 `_refine_pilot_grid_in_steep_segments` 里抽出来的共享实现（原来那里
    是内联重复代码），数值行为不变；`classify_pilot_point_risk_zone` 也用它
    判断"当前最长热力学区间"。
    """
    sqrt_g = np.sqrt(np.clip(np.asarray(metric_g, dtype=float), 1.0e-12, None))
    lam = np.asarray(pilot_lambdas, dtype=float)
    if lam.size < 2:
        return np.zeros(0, dtype=float)
    return 0.5 * (sqrt_g[:-1] + sqrt_g[1:]) * np.abs(np.diff(lam))


def pilot_block_running_diagnostics(
    values: np.ndarray, temperature_K: float
) -> Dict[str, Any]:
    """给定某个 pilot 点截至目前采到的 dU/dlambda 样本，算一组"假想现在停
    下"的诊断量。纯数值，不抛异常——样本太少时相应字段退化成 NaN，由调用方
    按 ``n_samples`` 自己决定要不要信。

    Returns
    -------
    dict：``n_samples``、``mean_dU_dlambda_kJ_mol``、``std_dU_dlambda_kJ_mol``、
    ``sem_kJ_mol``、``metric_g``（beta^2*Var）、``excess_kurtosis``（超额峰
    度，>0 说明比正态分布更厚尾，可能是还没等到的稀有事件的早期信号）、
    ``max_abs_robust_zscore``（基于 MAD 的稳健 z 分数最大绝对值，抓单个突发
    异常值，不像普通 z 分数那样会被该值自己拉高的标准差稀释）。
    """
    values = np.asarray(values, dtype=float).ravel()
    n = int(values.size)
    out: Dict[str, Any] = {"n_samples": n}
    if n < 2:
        out.update(
            mean_dU_dlambda_kJ_mol=float("nan"),
            std_dU_dlambda_kJ_mol=float("nan"),
            sem_kJ_mol=float("nan"),
            metric_g=float("nan"),
            excess_kurtosis=float("nan"),
            max_abs_robust_zscore=float("nan"),
        )
        return out

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    beta = 1.0 / (0.008314462618 * float(temperature_K))
    out["mean_dU_dlambda_kJ_mol"] = mean
    out["std_dU_dlambda_kJ_mol"] = std
    out["sem_kJ_mol"] = float(std / np.sqrt(n))
    out["metric_g"] = float(beta * beta * std * std)

    if n >= 4 and std > 0.0:
        out["excess_kurtosis"] = float(np.mean((values - mean) ** 4) / std**4 - 3.0)
    else:
        out["excess_kurtosis"] = float("nan")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0.0:
        robust_z = 0.6745 * (values - median) / mad
        out["max_abs_robust_zscore"] = float(np.max(np.abs(robust_z)))
    else:
        out["max_abs_robust_zscore"] = 0.0
    return out


def classify_pilot_point_risk_zone(
    pilot_lambdas,
    metric_g,
    is_refinement_point,
    lambda_near_one_floor: float = 0.875,
) -> List[str]:
    """对每个 pilot 点标 "risk" / "easy"，纯事后打标签，不影响任何真实采样。

    风险判据（跟用户敲定的设计一一对应）：
      - 加密点（``is_refinement_point[i]`` 为 True，来自
        `_refine_pilot_grid_in_steep_segments`）——插入的理由本来就是父区间
        空间信息不足，继承父区间风险，不因为是"额外点"缩短预算。
      - λ ≥ ``lambda_near_one_floor``（默认 0.875，覆盖
        `human_vanishing_initial_lambdas` 17 点网格里 λ=1.0 起最前两段）——
        07-19 那次真实回归的端点区域。
      - 当前最长热力学区间（`_pilot_segment_lengths` 最大值，允许并列）的两
        个端点。

    其余点标 "easy"。数组长度不一致时整体退化成全 "risk"（宁可保守，不猜）。
    """
    lam = np.asarray(pilot_lambdas, dtype=float).ravel()
    g = np.asarray(metric_g, dtype=float).ravel()
    refine_flags = list(is_refinement_point)
    n = int(lam.size)
    if not (n == g.size == len(refine_flags)) or n < 2:
        return ["risk"] * max(n, 0)

    tags = ["easy"] * n
    for i in range(n):
        if bool(refine_flags[i]):
            tags[i] = "risk"
        elif lam[i] >= float(lambda_near_one_floor):
            tags[i] = "risk"

    seg_lengths = _pilot_segment_lengths(lam, g)
    if seg_lengths.size:
        worst = float(np.max(seg_lengths))
        for i, length in enumerate(seg_lengths):
            if length >= worst - 1.0e-12 * max(worst, 1.0):
                tags[i] = "risk"
                tags[i + 1] = "risk"
    return tags


def pilot_early_stop_pressure_test(
    pilot_lambdas,
    final_metric_g,
    point_index: int,
    checkpoint_metric_g: float,
    worst_case_inflation_ratio: float = 3.0,
    max_allowed_lambda_shift: float = 0.01,
    # 🔑 [0831issue P2] 默认值改成 None、在函数体内再读模块常量。
    # 默认参数在**函数定义时**求值一次，所以写成
    # `= VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS` 会把常量当时的值永久焊进签名；
    # 该常量历史上经过 2→6→4 的演进，而校验方 `redistribute_vanishing_lambda_subdomains`
    # 读的是**当前**全局值——两者会静默失配，压力测试基线与生产布点契约就对不上了。
    first_ensemble_target_intervals: Optional[int] = None,
) -> Dict[str, Any]:
    """压力测试：如果 ``point_index`` 这个点在某个 checkpoint 就已经拿到了
    ``checkpoint_metric_g``（而不是跑满 30000 步后的真实
    ``final_metric_g[point_index]``），production λ 布点会挪动多少；再把这
    个 checkpoint 估计按 ``worst_case_inflation_ratio`` 向上膨胀重算一次，两
    次位移都要低于 ``max_allowed_lambda_shift`` 才算通过压力测试。

    🔑 ``worst_case_inflation_ratio`` 默认值 3.0 是占位符，不是已验证的数
    字——本函数落地时仓库里还没有真实的 shadow 数据；Phase B 拿到真机 30000
    步的 checkpoint 序列、反推出真实的"部分估计 vs 最终估计"比值分布之后，
    必须回填一个有实测依据的值，调用方不应该信任这个默认值本身代表任何安全
    边际。

    永不抛异常：`redistribute_vanishing_lambda_subdomains` 失败（输入不满足
    不变量等）时返回 ``{"valid": False, "reason": ...}``——这是离线诊断函
    数，不能让分析脚本因为一次异常输入就整体崩溃。
    """
    # [0831issue P2] None → 此刻读模块常量的当前值，跟校验方
    # redistribute_vanishing_lambda_subdomains 用同一个来源，不会被定义期快照冻住。
    if first_ensemble_target_intervals is None:
        first_ensemble_target_intervals = VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS
    try:
        lam = np.asarray(pilot_lambdas, dtype=float)
        g_final = np.asarray(final_metric_g, dtype=float)
        if not (0 <= int(point_index) < lam.size) or lam.size != g_final.size:
            return {"valid": False, "reason": "bad_point_index_or_shape_mismatch"}

        baseline_lambdas, *_ = redistribute_vanishing_lambda_subdomains(
            lam, g_final, VANISHING_PROBE_BASE_STATE_COUNT,
            first_ensemble_target_intervals=first_ensemble_target_intervals,
        )

        def _shift_for(substitute_metric_g: float) -> float:
            g_mod = g_final.copy()
            g_mod[int(point_index)] = float(substitute_metric_g)
            candidate_lambdas, *_ = redistribute_vanishing_lambda_subdomains(
                lam, g_mod, VANISHING_PROBE_BASE_STATE_COUNT,
                first_ensemble_target_intervals=first_ensemble_target_intervals,
            )
            return float(np.max(np.abs(candidate_lambdas - baseline_lambdas)))

        raw_shift = _shift_for(checkpoint_metric_g)
        inflated_shift = _shift_for(
            float(checkpoint_metric_g) * float(worst_case_inflation_ratio)
        )
        passes = (
            raw_shift <= max_allowed_lambda_shift
            and inflated_shift <= max_allowed_lambda_shift
        )
        return {
            "valid": True,
            "raw_lambda_shift": raw_shift,
            "inflated_lambda_shift": inflated_shift,
            "max_allowed_lambda_shift": float(max_allowed_lambda_shift),
            "worst_case_inflation_ratio": float(worst_case_inflation_ratio),
            "would_pass_pressure_test": bool(passes),
        }
    except Exception as e:  # noqa: BLE001 -- 离线诊断，fail-closed 不能崩调用方
        return {"valid": False, "reason": f"redistribute_failed: {e}"}


# =============================================================================
# 添加双λ路径优化类
# =============================================================================
# 修复 DualLambdaPreOptimizer 类
# =============================================================================
# =============================================================================
# 修复 DualLambdaPreOptimizer 类 (完整修复版)
# =============================================================================
# ================= abfe_preoptimizer.py =================
# 完整替换 DualLambdaPreOptimizer 类
class DualLambdaPreOptimizer:
    """双λ预采样优化器 (全链路 Debug 版)"""
    def __init__(self, system, context, temperature=300.0):
        print(f"\n[DEBUG-OPT] DualLambdaPreOptimizer 初始化...")
        self.system = system
        self.context = context
        self.temperature = temperature
        self.param_coul = self._normalize_param_name(
            self._detect_param("coul", ["lam_coul", "lambda_coul"])
        )
        self.param_vdw = self._normalize_param_name(
            self._detect_param("vdw", ["lam_vdw", "lambda_vdw"])
        )
        print(f"[DEBUG-OPT] 探测结果 -> Coul: '{self.param_coul}', VdW: '{self.param_vdw}'")

    @staticmethod
    def _normalize_param_name(param) -> Optional[str]:
        if param is None:
            return None
        if isinstance(param, np.ndarray):
            flat = np.asarray(param).ravel()
            if flat.size == 0:
                return None
            param = flat[0]
        return str(param)

    def _detect_param(self, keyword: str, fallbacks: list) -> Optional[str]:
        print(f"  [SCAN] 搜索关键词: '{keyword}', 候选: {fallbacks}")
        # 1. 扫 Force
        if self.system is not None:
            for f in self.system.getForces():
                if isinstance(f, openmm.CustomNonbondedForce):
                    names = [f.getGlobalParameterName(i) for i in range(f.getNumGlobalParameters())]
                    print(f"  [SCAN] Force 包含参数: {names}")
                    for n in names:
                        if keyword in n.lower(): 
                            print(f"  [SCAN] [OK] Force 匹配到: {n}")
                            return n
        # 2. 扫 Context
        try:
            ctx_p = list(self.context.getParameters().keys())
            print(f"  [SCAN] Context 包含参数: {ctx_p}")
            for k in ctx_p:
                if keyword in k.lower(): 
                    print(f"  [SCAN] [OK] Context 匹配到: {k}")
                    return k
        except Exception as e: print(f"  [SCAN] Context 读取失败: {e}")
        return None

    def _metric_force_groups(self, parameter_name: str) -> set:
        """差分 `parameter_name` 时要计入哪些 force group。

        规则只有一条：**把该参数的依赖项全部算进来，且只算这些**。

        * 软核 ACES 力（group 1）同时带 lam_coul 与 lam_vdw ⟹ 永远计入。
        * 原生 NonbondedForce（`PREOPT_NATIVE_NONBONDED_FORCE_GROUP`，只在带电腿
          存在）只带 lam_coul（B3 的 PME ParameterOffset）⟹ **只在差分 lam_coul
          时**计入。差分 lam_vdw 时它是个 ~10^6 kJ/mol 的常数，算进来就是在巨大
          公共项上做差再除以 delta≈0.02，纯灾难性相消。
        * 中性腿本来就没有力在那个 group 里，集合多写一个空 group 无副作用。
        """
        if self.param_coul is not None and parameter_name == self.param_coul:
            return {1, PREOPT_NATIVE_NONBONDED_FORCE_GROUP}
        return {1}

    def _metric_energy_at(self, parameter_name: str, lam: float) -> float:
        self.context.setParameter(parameter_name, float(lam))
        state = self.context.getState(
            getEnergy=True, groups=self._metric_force_groups(parameter_name)
        )
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def _finite_difference_derivative_1d(
        self,
        parameter_name: str,
        lam: float,
        delta: float,
    ) -> float:
        """Evaluate dU/dlambda on one frozen configuration and restore lambda."""
        lam = float(lam)
        lo = max(0.0, lam - float(delta))
        hi = min(1.0, lam + float(delta))
        try:
            if hi > lam and lo < lam:
                e_hi = self._metric_energy_at(parameter_name, hi)
                e_lo = self._metric_energy_at(parameter_name, lo)
                return (e_hi - e_lo) / (hi - lo)
            e_0 = self._metric_energy_at(parameter_name, lam)
            if hi > lam:
                return (self._metric_energy_at(parameter_name, hi) - e_0) / (hi - lam)
            if lo < lam:
                return (e_0 - self._metric_energy_at(parameter_name, lo)) / (lam - lo)
            raise RuntimeError(f"lambda={lam} 没有可用的有限差分邻点")
        finally:
            self.context.setParameter(parameter_name, lam)

    def _sample_scalar_metric(
        self,
        parameter_name: str,
        lam: float,
        n_steps: int,
        delta: float,
        sample_interval: int = 50,
        shadow_checkpoint_steps: Optional[List[int]] = None,
    ) -> Tuple[float, Dict]:
        """Short-pilot estimate of beta**2 Var[dU/dlambda]."""
        # 🔑 [性能计时] 只加计时，不改任何积分/有限差分逻辑或默认参数——
        # sample_interval/n_steps 直接影响 λ 路径优化结果，这次不动，见
        # optimize_stage2_vanishing 调用处的说明。目的是把"这个 λ 点到底
        # 花在积分 vs 有限差分能量读取上多少时间"变成可测量的数字。
        point_timers: Dict[str, float] = {}
        derivative_samples = []
        # 🔑 [shadow early-stop 插桩，Phase A，2026-08-26] shadow_checkpoint_
        # steps 为 None（默认）时下面这段完全不产生任何额外计算/字段——循环
        # 仍然无条件跑满传入的 n_steps，真实采样长度、返回值形状逐字节不变。
        # 传入时也不改变真实采样长度：batches 仍然全部跑完；只是在累积步数
        # 跨过每个请求的 checkpoint 时，用当时已经采到的 derivative_samples
        # 多算一次"假想现在停下会怎样"的诊断，写进 shadow_trace，不参与任何
        # 真实判断分支（是否继续采样、metric_g 怎么算，都跟今天完全一样）。
        pending_checkpoints = (
            sorted({int(s) for s in shadow_checkpoint_steps})
            if shadow_checkpoint_steps
            else []
        )
        shadow_trace: List[Dict[str, Any]] = []
        cumulative_steps = 0
        full_batches, remainder = divmod(int(n_steps), int(sample_interval))
        batches = [int(sample_interval)] * full_batches
        if remainder:
            batches.append(remainder)
        for batch_steps in batches:
            with _timed(point_timers, "integration_s"):
                guarded_step(self.context.getIntegrator(), batch_steps, "pilot 标量度量采样")
            with _timed(point_timers, "finite_difference_s"):
                derivative = self._finite_difference_derivative_1d(
                    parameter_name, lam, delta
                )
            if np.isfinite(derivative):
                derivative_samples.append(float(derivative))
            cumulative_steps += int(batch_steps)
            while pending_checkpoints and cumulative_steps >= pending_checkpoints[0]:
                checkpoint_step = pending_checkpoints.pop(0)
                snapshot = pilot_block_running_diagnostics(
                    np.asarray(derivative_samples, dtype=float),
                    float(self.temperature),
                )
                snapshot["cumulative_steps"] = int(cumulative_steps)
                snapshot["requested_checkpoint_steps"] = int(checkpoint_step)
                shadow_trace.append(snapshot)

        if len(derivative_samples) < 10:
            raise RuntimeError(
                f"lambda={lam:.6f} 只有 {len(derivative_samples)} 个有效 dU/dlambda 样本；"
                "至少需要 10 个，拒绝用欠采样度量生成路径"
            )
        values = np.asarray(derivative_samples, dtype=float)
        beta = 1.0 / (0.008314462618 * float(self.temperature))
        metric_g = float(beta * beta * np.var(values, ddof=1))
        diag = {
            "lambda": float(lam),
            "n_derivative_samples": int(values.size),
            "mean_dU_dlambda_kJ_mol": float(np.mean(values)),
            "std_dU_dlambda_kJ_mol": float(np.std(values, ddof=1)),
            "metric_g": metric_g,
            "timing_s": dict(point_timers),
        }
        if shadow_checkpoint_steps:
            diag["shadow_trace"] = shadow_trace
        return metric_g, diag

    def optimize_stage1_decharging(self, n_states=12, n_steps_per_state=2000):
        # This legacy entry point samples ``Var(U_group1)`` from a cutoff
        # probe.  That is not the Fisher metric of the production PME
        # Hamiltonian (which requires beta² Var[dU/dlambda]), and it can also
        # include lambda-independent environment noise.  The production
        # pipeline already uses a validated linear Stage-1 path; keep this
        # public API fail-closed until a PME derivative sampler is implemented.
        raise RuntimeError(
            "Stage 1 自适应去电荷预优化已禁用：旧实现使用 Var(U) 而非生产 PME 的 "
            "beta² Var[dU/dlambda]，其路径不能用于热力学采样。请使用 pipeline 的线性路径。"
        )

        print(f"\n[STAGE1] 开始去电荷路径优化 (n_states={n_states})...")
        print(f"[STAGE1] 当前 param_coul='{self.param_coul}', param_vdw='{self.param_vdw}'")
        
        if self.param_coul is None:
            print(f"[STAGE1] [WARN] 探针系统未注册 Coulomb λ 参数，直接生成线性回退路径")
            return {"stage": "decharging", "lambdas_coul": np.linspace(1.0, 0.0, n_states).tolist(), "lambdas_vdw": [1.0]*n_states, "n_states": n_states}
            
        # 安全设置
        # ✅ 修复：统一增加 None 保护
        current_params = dict(self.context.getParameters())
        if self.param_vdw is not None:
            if self.param_vdw in current_params:
                self.context.setParameter(self.param_vdw, 1.0)
                print(f"[STAGE1] 固定 λ_vdw = 1.0")
                
        self.context.setParameter(self.param_coul, 1.0)
        print(f"[STAGE1] 设置初始 λ_coul = 1.0")
        guarded_step(self.context.getIntegrator(), 5000, "Stage1 pilot 起始预平衡")

        variance_data = []
        lambdas = np.linspace(1.0, 0.0, n_states)
        print(f"[STAGE1] 线性采样点: {lambdas}")

        for i, lam in enumerate(lambdas):
            self.context.setParameter(self.param_coul, float(lam))
            if self.param_vdw is not None:  # ✅ 修复：增加 None 守卫，确保 Stage1 安全
                self.context.setParameter(self.param_vdw, 1.0)
            guarded_step(self.context.getIntegrator(), 500, f"Stage1 pilot λ_coul={lam:.4f} 预平衡")
            energies = [
                e for e in _sample_group1_energies(self.context, n_steps_per_state, sample_interval=50)
                if not (np.isnan(e) or np.isinf(e))
            ]
            variance_data.append(np.var(energies) if len(energies) >1 else 1.0)

        # 路径重分布 (保持原逻辑)
        std_dev = np.sqrt(np.array(variance_data) + 1e-10)
        density_weight = np.asarray(_normalize_variance_weights(std_dev, max_ratio=0.15), dtype=float).ravel()
        # [0831issue P2] 同 optimize_lambda_path_adaptive：用前 N-1 个权重当区间宽度、
        # 按它们自己的和归一化，末端天然为 1.0，不再事后覆盖 c_{N-2}（那会把
        # λ[N-2] 的权重双重计入）。
        interval_weights = np.asarray(density_weight, dtype=float).ravel()[:-1]
        interval_total = max(1e-10, float(np.sum(interval_weights)))
        xp = np.concatenate(
            ([0.0], np.cumsum(interval_weights) / interval_total)
        ).astype(float).ravel()
        fp = np.asarray(lambdas, dtype=float).ravel()
        target_cumulative = np.linspace(0, 1.0, n_states)
        optimized_lambdas = np.asarray(np.interp(target_cumulative, xp, fp), dtype=float).ravel()
        # 🔑 之前这里只 clip/排序/钉端点，没有最小间距、没有去重、状态数不足时
        # 也没有 fail-closed/回退——跟单 λ optimize_lambda_path_adaptive 的同一步
        # 逻辑不一致，一次看似正常的 CDF 插值完全可能悄悄给出两个数值相同的 λ，
        # 破坏 MBAR 对相邻态"确实是不同态"的假设，且不会有任何报错/日志。改用
        # 共享的 finalize_descending_lambda_path，跟单 λ 路径统一同一套不变量
        # （密度权重公式本身仍保持 Stage1 自己的实现，未改动）。
        optimized_lambdas, _min_spacing, fell_back = finalize_descending_lambda_path(
            optimized_lambdas, n_states
        )
        if fell_back:
            print(f"[STAGE1] [WARN] 去重后状态数少于目标 ({n_states})，使用线性路径")

        print(f"[STAGE1] [OK] 优化完成，返回前5个λ: {optimized_lambdas[:5]}")
        return {"stage": "decharging", "lambdas_coul": optimized_lambdas.tolist(), "lambdas_vdw": [1.0]*len(optimized_lambdas), "n_states": len(optimized_lambdas)}

    def _refine_pilot_grid_in_steep_segments(
        self,
        pilot_lambdas,
        metric_g,
        pilot_points,
        n_steps_per_state,
        finite_difference_delta,
        max_segment_length_fraction: float = 0.2,
        extra_points_per_segment: int = 4,
        max_rounds: int = 2,
        shadow_checkpoint_steps: Optional[List[int]] = None,
        pilot_states=None,
    ):
        """[THERMODYNAMIC_PATH_PROTOCOL_VERSION=15] Probe additional points
        strictly inside whichever single coarse pilot segment dominates the
        total thermodynamic length, instead of trusting a straight-line
        interpolation across it.

        Three independent real GPU runs of vanishing window 0 (6/4/3-state
        groupings) all failed with occupation stuck ~97-99% at state 0 and
        min_absolute_ess~1.0 -- regrouping never changes the actual lambda
        values, only which states share one IBS bias, so it could not (and
        did not) fix a real overlap problem at the state0/state1 edge. The
        real cached pilot data showed why: the coarse grid's very first
        segment (lambda=1.0 -> ~0.94) alone contributed ~47% of the entire
        path's thermodynamic length, but is defined by only 2 raw pilot
        points -- so redistribute_lambda_by_thermodynamic_length's arc-length
        interpolation could only ever place new states *linearly* inside it,
        with no real data on how the true difficulty is actually distributed
        there. This method is the fix: reuse the exact same
        _sample_scalar_metric measurement, just at more points, specifically
        inside whatever segment is currently blind, and merge the results
        back into the pilot arrays before redistribution ever runs. Returns
        (pilot_lambdas, metric_g, pilot_points) with the same shapes/meaning
        as the inputs, just more entries -- every downstream consumer already
        works generically on however many pilot points it receives.
        """
        # 🔑 [2026-09-10] 加密点必须从**相邻的高 λ 端点**续接采样，不能从上一个
        # 主 pilot 点（λ=0，配体已完全解耦）直接跳回 λ≈0.99。
        #
        # 主 pilot 是 λ=1 → 0 顺序走的，跑完才进这个函数。原来这里直接
        # `setParameter(λ_vdw, 0.99)` 然后只给 500 步预平衡就开测 —— 那是在
        # "配体突然重新长回一个已经塌陷的空腔"的**非平衡**构型上测 dU/dλ，
        # mean 和 Var 都被系统性抬高。而这批点恰恰是 λ 布点与 f_k 种子的输入，
        # 同时也是崩溃源（重新耦合方向的 500 步很容易在 guarded_step 里抛 NaN）。
        #
        # 现在要求调用方把每个主 pilot λ 的 (坐标, 速度, 盒子) 快照一并传进来：
        # 细化某段之前先恢复该段**高 λ 端点**的状态，再向低 λ 顺序采；新插入点
        # 自己的状态也存回列表，供下一轮从它续接。这样加密点的采样历史与主网格
        # 同类，500 步预平衡才站得住。
        #
        # ⚠️ 这改变了 pilot 的采样语义 ⟹ 旧 preopt 缓存不能冒充等价，必须由
        # 上游的 layer-1 采样语义字段挡住（不是靠版本号）。
        if pilot_states is None:
            raise ValueError(
                "pilot 网格加密需要每个主 pilot λ 的 endpoint states（坐标/速度/盒子快照）："
                "加密点必须从相邻高 λ 端点续接采样，不能从 λ=0 的完全解耦构型跳回去。"
                "调用方未提供 pilot_states —— 拒绝在没有续接状态的情况下加密（fail closed）。"
            )
        pilot_states = list(pilot_states)
        if len(pilot_states) != len(list(pilot_lambdas)):
            raise ValueError(
                "endpoint states 数量与 pilot λ 数量不一致："
                f"{len(pilot_states)} != {len(list(pilot_lambdas))}。"
                "两者必须逐位对应，否则会从错误的构型续接。"
            )

        pilot_lambdas = [float(x) for x in pilot_lambdas]
        metric_g = list(metric_g)
        pilot_points = list(pilot_points)
        current_params = dict(self.context.getParameters())

        for _round in range(int(max_rounds)):
            # 🔑 [重用] 跟 classify_pilot_point_risk_zone 用同一份共享实现
            # （原来这里是内联重复代码），数值行为不变。
            seg_lengths = _pilot_segment_lengths(pilot_lambdas, metric_g).tolist()
            total_length = float(sum(seg_lengths))
            if total_length <= 0.0 or not seg_lengths:
                break
            worst_idx = max(range(len(seg_lengths)), key=lambda i: seg_lengths[i])
            worst_fraction = seg_lengths[worst_idx] / total_length
            if worst_fraction <= float(max_segment_length_fraction):
                break

            lam_hi = pilot_lambdas[worst_idx]
            lam_lo = pilot_lambdas[worst_idx + 1]
            new_lams = np.linspace(lam_hi, lam_lo, int(extra_points_per_segment) + 2)[1:-1]
            print(
                f"  [pilot 加密] 段 [{lam_lo:.4f}, {lam_hi:.4f}] 占当前总热力学长度 "
                f"{worst_fraction * 100:.1f}%（阈值 {float(max_segment_length_fraction) * 100:.0f}%），"
                f"插入 {len(new_lams)} 个额外探针点重测（第 {_round + 1} 轮）"
            )

            insert_at = worst_idx + 1
            # 恢复该段**高 λ 端点**的状态，然后向低 λ 顺序采（new_lams 已是 hi→lo）。
            # 只在进入这一段时恢复一次；段内各点自然地一个接一个续下去，
            # 与主 pilot 的遍历方向一致。
            self.context.setState(pilot_states[worst_idx])
            for lam in new_lams:
                self.context.setParameter(self.param_vdw, float(lam))
                if self.param_coul is not None and self.param_coul in current_params:
                    self.context.setParameter(self.param_coul, 0.0)
                guarded_step(self.context.getIntegrator(), 500, f"pilot 网格加密 λ_vdw={lam:.4f} 预平衡")
                g_lam, point_diag = self._sample_scalar_metric(
                    self.param_vdw,
                    float(lam),
                    n_steps=int(n_steps_per_state),
                    delta=float(finite_difference_delta),
                    shadow_checkpoint_steps=shadow_checkpoint_steps,
                )
                # 🔑 加密点永远标记为风险点（classify_pilot_point_risk_zone
                # 消费这个字段）——插入的理由本来就是父区间空间信息不足，不
                # 因为是"额外点"缩短预算判断。
                point_diag["is_refinement_point"] = True
                _timing = point_diag.get("timing_s", {})
                print(
                    f"    [preopt 加密 λ={float(lam):.4f}] "
                    + ", ".join(f"{k}={v:.1f}s" for k, v in _timing.items())
                )
                pilot_lambdas.insert(insert_at, float(lam))
                metric_g.insert(insert_at, g_lam)
                pilot_points.insert(insert_at, point_diag)
                # 新点自己的状态存回同一位置：下一轮若选中与它相邻的段，
                # 就从这里续接，而不是回退到更高的 λ。
                pilot_states.insert(
                    insert_at,
                    self.context.getState(
                        getPositions=True, getVelocities=True, getParameters=True
                    ),
                )
                insert_at += 1

        return np.asarray(pilot_lambdas, dtype=float), metric_g, pilot_points

    def optimize_stage2_vanishing(
        self,
        n_states=VANISHING_PROBE_BASE_STATE_COUNT,
        n_steps_per_state=2000,
        finite_difference_delta=0.01,
        shadow_checkpoint_interval: Optional[int] = None,
        final_state_count: int = VANISHING_FINAL_STATE_COUNT,
        # 🔑 [2026-08-27] 之前硬编码在 _refine_pilot_grid_in_steep_segments 的
        # 默认参数里（这里没暴露），现在做成真正能传的参数，默认值不变。
        refine_extra_points_per_segment: int = 4,
        # 🔑 [2026-08-27] 只在 final_state_count != VANISHING_FINAL_STATE_COUNT
        # 时生效，见 redistribute_vanishing_lambda_subdomains。
        min_states_per_window: int = 4,
        max_states_per_window: int = 6,
        # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=22] 自由能定向加密点数，默认 0
        # （布点与 v21 逐字节相同）。总态数仍是 final_state_count，成本不变。
        free_energy_densify_points: int = VANISHING_FREE_ENERGY_DENSIFY_POINTS,
    ):
        print(
            f"\n→ Stage 2: 去 VDW 路径优化 "
            f"({n_states} 点 Fisher 探针网格 → {final_state_count} 态"
            f"度规布点，几何覆盖下限 beta={VANISHING_GEOMETRIC_FLOOR_WEIGHT})..."
        )
        current_params = dict(self.context.getParameters())
        if self.param_vdw is None or self.param_vdw not in current_params:
            raise RuntimeError(f"探针系统未注册 VdW λ 参数，无法执行自适应优化")

        # 🔑 [shadow early-stop 插桩，Phase A，2026-08-26] shadow_checkpoint_
        # interval 默认 None——下面这行给出 None，_sample_scalar_metric 里
        # pending_checkpoints 恒为空列表，真实采样长度/返回值形状逐字节不
        # 变。显式传入正整数时才会在每跑够这么多步就多记一次"假想提前停"的
        # 诊断，不改变任何一次真实采样的步数或判断分支。
        shadow_checkpoint_steps = (
            list(
                range(
                    int(shadow_checkpoint_interval),
                    int(n_steps_per_state) + 1,
                    int(shadow_checkpoint_interval),
                )
            )
            if shadow_checkpoint_interval
            else None
        )

        if self.param_coul is not None and self.param_coul in current_params:
            self.context.setParameter(self.param_coul, 0.0)
        self.context.setParameter(self.param_vdw, 1.0)
        guarded_step(self.context.getIntegrator(), 5000, "Stage2 pilot 起始预平衡")

        # Probe a conventional grid for diagnostics.  Production lambda
        # placement keeps the v19 quadratic base so the lambda~0 tail cannot
        # collapse again; v20 additionally lets this metric insert two bridge
        # states into the longest remaining production edges.
        pilot_lambdas = human_vanishing_initial_lambdas(int(n_states))
        metric_g = []
        pilot_points = []
        # 🔑 [2026-09-10] 逐 λ 存一份 (坐标, 速度, 盒子) 快照。
        #
        # 加密阶段（`_refine_pilot_grid_in_steep_segments`）要从**相邻高 λ 端点**
        # 续接采样，而不是从这个循环结束时的 λ=0（配体完全解耦）构型跳回 λ≈0.99。
        # 没有这些快照它会 fail closed，理由见那个函数的说明。
        pilot_states = []
        for lam in pilot_lambdas:
            self.context.setParameter(self.param_vdw, float(lam))
            if self.param_coul is not None and self.param_coul in current_params:
                self.context.setParameter(self.param_coul, 0.0)
            guarded_step(self.context.getIntegrator(), 500, f"Stage2 pilot λ_vdw={lam:.4f} 预平衡")
            g_lam, point_diag = self._sample_scalar_metric(
                self.param_vdw,
                float(lam),
                n_steps=int(n_steps_per_state),
                delta=float(finite_difference_delta),
                shadow_checkpoint_steps=shadow_checkpoint_steps,
            )
            point_diag["is_refinement_point"] = False
            metric_g.append(g_lam)
            pilot_points.append(point_diag)
            pilot_states.append(
                self.context.getState(
                    getPositions=True, getVelocities=True, getParameters=True
                )
            )
            _timing = point_diag.get("timing_s", {})
            print(
                f"    [preopt λ={float(lam):.4f}] "
                + ", ".join(f"{k}={v:.1f}s" for k, v in _timing.items())
            )

        pilot_lambdas, metric_g, pilot_points = self._refine_pilot_grid_in_steep_segments(
            pilot_lambdas,
            metric_g,
            pilot_points,
            pilot_states=pilot_states,
            n_steps_per_state=n_steps_per_state,
            finite_difference_delta=finite_difference_delta,
            shadow_checkpoint_steps=shadow_checkpoint_steps,
            extra_points_per_segment=int(refine_extra_points_per_segment),
        )

        (
            optimized_lambdas,
            cumulative_length,
            optimized_edge_lengths,
            window_ranges,
            subdomain_allocation,
        ) = redistribute_vanishing_lambda_subdomains(
                pilot_lambdas,
                np.asarray(metric_g, dtype=float),
                int(n_states),
                first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
                final_state_count=int(final_state_count),
                min_states_per_window=int(min_states_per_window),
                max_states_per_window=int(max_states_per_window),
                free_energy_densify_points=int(free_energy_densify_points),
                pilot_mean_dU_dlambda=_pilot_mean_gradients_or_none(pilot_points),
        )
        optimized_lambdas = np.asarray(optimized_lambdas, dtype=float).ravel()
        optimized_lambdas = np.clip(optimized_lambdas, 0.0, 1.0)
        optimized_lambdas[0], optimized_lambdas[-1] = 1.0, 0.0
        # Thermodynamic length determines lambda density; few-state grouping is
        # performed afterwards along that coordinate.  No fixed lambda=0.5 cut
        # and no legacy overlap=2 construction that duplicates an interval are
        # used; one boundary node is still shared as the ensemble reference.
        
        # 🔑 [shadow early-stop 插桩，Phase A] 纯事后打标签，只读 pilot_points
        # 里已经落盘的 is_refinement_point，不影响上面任何一次真实采样/布点
        # 决定。shadow_checkpoint_steps 为 None 时 risk_zone_tags 也是 None，
        # 诊断字典形状对未启用 shadow 模式的调用完全不变。
        risk_zone_tags = None
        if shadow_checkpoint_steps:
            risk_zone_tags = classify_pilot_point_risk_zone(
                pilot_lambdas,
                metric_g,
                [bool(p.get("is_refinement_point", False)) for p in pilot_points],
            )

        diagnostics = {
            "estimator": "beta^2_var_dU_dlambda_finite_difference",
            "lambda_placement_method": (
                "fisher_metric_blended_with_geometric_floor_v21"
                if not int(free_energy_densify_points)
                else "fisher_metric_blended_with_geometric_floor_v21"
                     "+free_energy_densified_v22"
            ),
            "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
            "probe_controls_base_lambda_placement": True,
            "geometric_floor_weight": subdomain_allocation["geometric_floor_weight"],
            "max_lambda_gap_bound": subdomain_allocation["max_lambda_gap_bound"],
            "realized_max_lambda_gap": subdomain_allocation["realized_max_lambda_gap"],
            "realized_max_edge_thermodynamic_length": subdomain_allocation[
                "realized_max_edge_thermodynamic_length"
            ],
            "realized_min_edge_thermodynamic_length": subdomain_allocation[
                "realized_min_edge_thermodynamic_length"
            ],
            "requested_probe_base_state_count": int(n_states),
            "actual_state_count": int(len(optimized_lambdas)),
            "pilot_lambdas": [float(x) for x in pilot_lambdas],
            "metric_g": [float(x) for x in metric_g],
            "pilot_cumulative_thermodynamic_length": [float(x) for x in cumulative_length],
            "total_thermodynamic_length": float(cumulative_length[-1]),
            "optimized_edge_thermodynamic_lengths": [float(x) for x in optimized_edge_lengths],
            "finite_difference_delta": float(finite_difference_delta),
            "ibs_ensemble_layout": "few_state_thermodynamic_subdomains",
            "subdomain_allocation": subdomain_allocation,
            "sliding_overlap_states": 0,
            "common_boundary_state_count": 1,
            "pilot_points": pilot_points,
            "shadow_mode_enabled": bool(shadow_checkpoint_steps is not None),
        }
        if risk_zone_tags is not None:
            diagnostics["risk_zone_tags"] = risk_zone_tags
        print(
            f"  [OK] Stage 2 热力学长度路径完成：L={cumulative_length[-1]:.3f}, "
            f"{len(optimized_lambdas)} 态, {len(window_ranges)} 个 IBS 子区间"
        )
        print(f"    λ_vdw: {optimized_lambdas}")
        print(f"    windows: {window_ranges}")
        return {
            "stage": "vanishing",
            "lambdas_coul": [0.0] * len(optimized_lambdas),
            "lambdas_vdw": optimized_lambdas.tolist(),
            "n_states": len(optimized_lambdas),
            "window_ranges": window_ranges,
            "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
            "path_diagnostics": diagnostics,
        }


# 修复 9: warmup safety check
def apply_safety_checks_on_disable_warmup(simulation, enable_warmup, warmup_steps):
    from openmm import unit
    import numpy as np
    import warnings
    if not enable_warmup:
        if simulation.context is None:
            print("  [WARN] simulation.context 未初始化，跳过安全检查")
            return
        try:
            state = simulation.context.getState(getEnergy=True, getForces=True)
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            force_norms = np.linalg.norm(forces, axis=1)
            rms_force = np.sqrt(np.mean(force_norms**2))
            max_force = np.max(force_norms)
            
            # ✅ RMS 阈值 5000 + 极值兜底 20000
            if np.isnan(rms_force) or np.isinf(rms_force) or rms_force > 5000 or max_force > 20000:
                warnings.warn("[WARN] 检测到不合理 RMS 力或极值，强制能量最小化...", UserWarning)
                simulation.minimizeEnergy(maxIterations=10000)
                state = simulation.context.getState(getEnergy=True, getForces=True)
                forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                force_norms = np.linalg.norm(forces, axis=1)
                rms_force = np.sqrt(np.mean(force_norms**2))
                print(f"  [OK] 最小化后: RMS|F|={rms_force:.2e}, max|F|={np.max(force_norms):.2e}")
            else:
                print(f"  [OK] 安全检查通过: RMS|F|={rms_force:.2e}")
        except Exception as e:
            print(f"  [ERR] 安全检查失败: {e}")
            raise

# ============================================================================
# 探针系统构建函数 (迁移自 ibs_engine)
# ============================================================================
from abfe_core import (
    ensure_owned_system,
    sync_all_exclusions,
    create_ligand_internal_force,
    AlchemicalPotentialFactory,
)



def _create_softcore_force_dual_lambda(
    nb_force,
    perturbed_indices,
    environment_indices,
    lam_coul,
    lam_vdw,
    softcore_params,
    reference_exclusions=None,
    particle_params_override=None,
    num_particles=None,
    use_global_lambda=False,
    cutoff_distance=1.0,
):
    """构建带双全局 lambda 的软核力，用于探针系统。

    [MEM-00h，2026-08-06] 默认值从 1.2 nm 改为 1.0 nm、关闭 switching——探针
    体系是给 λ 路径预优化用的，如果它的非键协议跟生产 `ibs_engine.
    _create_softcore_force` 不一致，预优化出来的 overlap/度量场就是在一个跟
    实际采样不一样的哈密顿量上算的，没有意义。所有调用方都没有显式传
    `cutoff_distance`，全部吃这个默认值，改这里即可。
    """
    if num_particles is None:
        num_particles = nb_force.getNumParticles()
    perturbed_set = set(perturbed_indices)
    env_set = set(environment_indices)

    lam_c_str = "lam_coul" if use_global_lambda else f"{lam_coul:.6f}"
    lam_v_str = "lam_vdw" if use_global_lambda else f"{lam_vdw:.6f}"
    expr, _ = AlchemicalPotentialFactory.build("softcore", softcore_params, lam_c_str, lam_v_str)

    force = openmm.CustomNonbondedForce(expr)
    for p in ["q", "sigma", "epsilon"]:
        force.addPerParticleParameter(p)

    if use_global_lambda:
        force.addGlobalParameter("lam_coul", lam_coul)
        force.addGlobalParameter("lam_vdw", lam_vdw)

    for i in range(num_particles):
        if particle_params_override and i < len(particle_params_override):
            q, sig, eps = particle_params_override[i]
        else:
            q, sig, eps = nb_force.getParticleParameters(i)
        force.addParticle([
            q.value_in_unit(unit.elementary_charge),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole)
        ])

    force.addInteractionGroup(perturbed_set, env_set)
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff_distance * unit.nanometer)

    if reference_exclusions is not None:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 < num_particles and p2 < num_particles:
                force.addExclusion(p1, p2)

    force.setUseSwitchingFunction(False)
    force.setSwitchingDistance(cutoff_distance * unit.nanometer)
    return force


def build_aces_probe_system_dual_lambda(
    system,
    perturbed_indices,
    softcore_params,
    fixed_lam_coul=None,
    fixed_lam_vdw=None,
    cutoff_distance=1.0,  # [MEM-00h，2026-08-06] 1.2→1.0，见 _create_softcore_force_dual_lambda 的说明
    use_reaction_field=False,
    topology=None,
    positions=None,
    box_vectors=None,
    co_alchemical_ion_spec=None,
):
    """双λ探针系统构建 (用于预优化).

    For a charged leg the probe must contain the *same* frozen co-ion
    Hamiltonian as production.  The canonical B3 builders in ``ibs_engine``
    write the ligand/co-ion ``NonbondedForce`` offsets and inject the shared
    flat-bottom restraint; this function only adapts their result to the
    ACES pilot (soft-core Lennard-Jones in the custom force).  In particular,
    it does not re-derive a second charge-transfer formula.
    """
    system = ensure_owned_system(system)
    # 🔑 [2026-09-10] 全仓 serialize→deserialize 深拷贝改 `XmlSerializer.clone()` 时
    # 漏了这一处，而它在 Stage 2 preopt 的活路径上（build_aces_probe_system_dual_lambda
    # 每次都走）。clone() 直接走 SerializationNode，不生成那份 7.7 MB 中间字符串
    # ——实测的 `char const *` typemap 崩溃现场就是这种大字符串往返。
    new_sys = ensure_owned_system(XmlSerializer.clone(system))
    num_atoms = new_sys.getNumParticles()
    perturbed_set = set(perturbed_indices)
    env_idx = [i for i in range(num_atoms) if i not in perturbed_set]

    nb_forces = [f for f in new_sys.getForces() if isinstance(f, openmm.NonbondedForce)]
    nb = nb_forces[0]
    # Keep the physical parameters for the ACES force before B3 mutates the
    # native NonbondedForce to its base+offset representation.
    all_p = [nb.getParticleParameters(i) for i in range(num_atoms)]
    ref_excl = [
        (int(nb.getExceptionParameters(i)[0]), int(nb.getExceptionParameters(i)[1]))
        for i in range(nb.getNumExceptions())
    ]

    charge_offsets_active = False
    if co_alchemical_ion_spec is not None:
        if topology is None:
            raise ValueError(
                "带 co-ion spec 的 ACES probe 必须传入 topology，"
                "以便复用 B3 verify_co_alchemical_ion_identity()。"
            )
        treatment = str(co_alchemical_ion_spec.get("charge_treatment", ""))
        if treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
            configure_charge_transfer_decharging(
                new_sys,
                list(perturbed_indices),
                topology,
                lambda_name="lam_coul",
                co_alchemical_ion_spec=co_alchemical_ion_spec,
            )
            charge_offsets_active = True
        elif treatment == CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL:
            if positions is None:
                raise ValueError(
                    "co-annihilation ACES probe 需要 positions，"
                    "以复用 B3 restraint 注入路径。"
                )
            configure_coalchemical_neutral_decharging(
                new_sys,
                list(perturbed_indices),
                topology,
                positions,
                box_vectors=box_vectors,
                lambda_name="lam_coul",
                co_alchemical_ion_spec=co_alchemical_ion_spec,
            )
            charge_offsets_active = True
        else:
            raise ValueError(
                f"ACES probe 收到未知 co-ion charge_treatment={treatment!r}；"
                "拒绝生成未绑定生产路线的 Hamiltonian。"
            )

        # The custom ACES force remains responsible for the soft-core LJ
        # interaction.  Its per-particle q for ligand atoms must be zero when
        # B3's PME ParameterOffset carries the real λ-dependent Coulomb term,
        # otherwise ligand/environment Coulomb would be counted twice.  The
        # native NonbondedForce still exposes the actual offsets for both the
        # ligand and co-ion, so endpoint charge audits inspect the real force.
        for idx in range(nb.getNumGlobalParameters()):
            if nb.getGlobalParameterName(idx) == "lam_coul":
                default_lam = 1.0 if fixed_lam_coul is None else float(fixed_lam_coul)
                nb.setGlobalParameterDefaultValue(idx, default_lam)
                break
        # Dual-lambda pilot diagnostics difference the native PME/Nonbonded
        # Force only when the Coulomb lambda is the one being varied -- that is
        # where B3's ParameterOffset installs the real λ_coul dependence.
        # 它因此有自己的 force group（见 PREOPT_NATIVE_NONBONDED_FORCE_GROUP），
        # 由 `_metric_force_groups()` 按被差分的参数决定要不要计入。
        nb.setForceGroup(PREOPT_NATIVE_NONBONDED_FORCE_GROUP)

    zero_q = 0.0 * unit.elementary_charge
    zero_sig = 0.1 * unit.nanometer  # 保留极小半径防除零，但能量为0
    zero_eps = 0.0 * unit.kilojoule_per_mole
    for idx in perturbed_indices:
        nb.setParticleParameters(idx, zero_q, zero_sig, zero_eps)

    # Group 2: 配体内部力
    ll_f, ll_14_f = create_ligand_internal_force(
        nb, perturbed_indices, all_p, ref_excl, num_atoms, system=system
    )
    if ll_f:
        ll_f.setForceGroup(2)
        new_sys.addForce(ll_f)
    if ll_14_f:
        ll_14_f.setForceGroup(2)
        new_sys.addForce(ll_14_f)

    # The custom Group 2 force now owns ligand 1-4 interactions. Particle
    # parameters do not disable NonbondedForce exceptions; clear those only
    # after create_ligand_internal_force has copied their original parameters.
    perturbed_set = set(perturbed_indices)
    for exception_index in range(nb.getNumExceptions()):
        p1, p2, charge_product, sigma, epsilon = nb.getExceptionParameters(exception_index)
        if int(p1) in perturbed_set and int(p2) in perturbed_set:
            nb.setExceptionParameters(
                exception_index, p1, p2, 0.0 * charge_product, sigma, 0.0 * epsilon
            )

    # Group 1: 双λ软核力
    aces_particle_params = all_p
    if charge_offsets_active:
        # Keep LJ sigma/epsilon and environment charges in the probe payload,
        # but leave ligand Coulomb to the canonical B3 offsets above.
        aces_particle_params = list(all_p)
        for idx in perturbed_indices:
            _q, sig, eps = aces_particle_params[int(idx)]
            aces_particle_params[int(idx)] = (
                zero_q,
                sig,
                eps,
            )

    ac_f = _create_softcore_force_dual_lambda(
        nb, perturbed_indices, env_idx,
        fixed_lam_coul, fixed_lam_vdw,
        softcore_params,
        reference_exclusions=ref_excl,
        particle_params_override=aces_particle_params,
        num_particles=num_atoms,
        use_global_lambda=True,
        cutoff_distance=cutoff_distance
    )
    if ac_f is not None:
        ac_f.setForceGroup(1)
        new_sys.addForce(ac_f)

    sync_all_exclusions(new_sys)
    new_sys.thisown = 1
    return new_sys


def build_aces_probe_system(
    system,
    perturbed_indices,
    softcore_params,
    prefix="aces_pre",
    fixed_lam_coul=0.5,
    fixed_lam_vdw=1.0,
    **probe_kwargs,
):
    """单λ探针系统构建（内部委托给双λ版本）"""
    return build_aces_probe_system_dual_lambda(
        system, perturbed_indices, softcore_params,
        fixed_lam_coul=fixed_lam_coul,
        fixed_lam_vdw=fixed_lam_vdw,
        **probe_kwargs,
    )


# ============================================================================
# 双λ 2D 度量张量场采集与单调有向图寻径 (DualLambdaPreOptimizer 扩展)
# ============================================================================
def _is_safe_dual_lambda_state(lam_coul: float, lam_vdw: float, tol: float = 1e-8) -> bool:
    """硬性物理边界：禁止在 VDW 斥力消失过快时保留过多电荷。"""
    return float(lam_vdw) + tol >= float(lam_coul)


def _safe_lambda_delta(base: float, trial: float) -> float:
    return max(0.0, min(1.0, float(trial))) - float(base)


def _sample_metric_energy(context, lam_coul: float, lam_vdw: float, axis: str) -> float:
    """2D 度规路径的能量读数。`axis` 决定计入哪些 force group。

    与 `DualLambdaPreOptimizer._metric_force_groups()` 同一条规则：原生
    NonbondedForce（带电腿的 PME ParameterOffset）只带 lam_coul 依赖，
    所以只有沿 coul 轴差分时才计入；沿 vdw 轴差分时它是个 ~10^6 kJ/mol 的常数，
    算进去就是灾难性相消。中性腿那个 group 是空的，多写无副作用。
    """
    context.setParameter("lam_coul", float(lam_coul))
    context.setParameter("lam_vdw", float(lam_vdw))
    groups = {1, PREOPT_NATIVE_NONBONDED_FORCE_GROUP} if axis == "coul" else {1}
    return context.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _finite_difference_on_safe_region(context, lc: float, lv: float, axis: str, delta: float) -> float:
    """
    仅在安全区域内估计 dU/dlambda，优先中心差分，否则退回单边差分。
    """
    if axis == "coul":
        plus = (lc + delta, lv)
        minus = (lc - delta, lv)
    else:
        plus = (lc, lv + delta)
        minus = (lc, lv - delta)

    plus_ok = 0.0 <= plus[0] <= 1.0 and 0.0 <= plus[1] <= 1.0 and _is_safe_dual_lambda_state(*plus)
    minus_ok = 0.0 <= minus[0] <= 1.0 and 0.0 <= minus[1] <= 1.0 and _is_safe_dual_lambda_state(*minus)

    e0 = None
    if plus_ok and minus_ok:
        e_plus = _sample_metric_energy(context, *plus, axis=axis)
        e_minus = _sample_metric_energy(context, *minus, axis=axis)
        return (e_plus - e_minus) / (2.0 * delta)
    if plus_ok:
        e_plus = _sample_metric_energy(context, *plus, axis=axis)
        e0 = _sample_metric_energy(context, lc, lv, axis=axis)
        step = _safe_lambda_delta(lc if axis == "coul" else lv, plus[0] if axis == "coul" else plus[1])
        return (e_plus - e0) / max(step, 1e-8)
    if minus_ok:
        e_minus = _sample_metric_energy(context, *minus, axis=axis)
        e0 = _sample_metric_energy(context, lc, lv, axis=axis)
        step = _safe_lambda_delta(lc if axis == "coul" else lv, minus[0] if axis == "coul" else minus[1])
        return (e0 - e_minus) / max(abs(step), 1e-8)
    raise RuntimeError(
        f"状态 (lambda_coul={lc:.3f}, lambda_vdw={lv:.3f}) 在 axis={axis} 上缺少安全差分邻点"
    )


def compute_2d_metric_grid(
    context,
    lam_c_grid,
    lam_v_grid,
    n_steps=3000,
    delta=0.02,
    temperature=300.0,
    return_diagnostics: bool = False,
):
    """采集 2D 度量张量场 g_cc, g_vv, g_cv 用于黎曼几何路径规划"""
    if int(len(lam_c_grid)) < 2 or int(len(lam_v_grid)) < 2:
        raise ValueError("2D metric grid 每个维度至少需要 2 个 lambda 点")
    if int(n_steps) < 2:
        raise ValueError("2D metric grid 的 n_steps 至少为 2，才能估计协方差")
    if not np.isfinite(float(delta)) or float(delta) <= 0.0:
        raise ValueError("finite-difference delta 必须为正有限数")
    # ``compute_2d_metric_grid`` is also a public low-level entry point (not
    # only called through ``optimize_2d_geodesic_path``), so validate the
    # temperature here as well.  Letting NaN/zero reach beta would silently
    # turn the Fisher metric into NaNs/Infs and make Dijkstra choose a bogus
    # path.
    try:
        temperature_value = float(
            temperature.value_in_unit(unit.kelvin)
            if hasattr(temperature, "value_in_unit")
            else temperature
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("metric temperature 必须是正有限值") from exc
    if not np.isfinite(temperature_value) or temperature_value <= 0.0:
        raise ValueError("metric temperature 必须是正有限值")
    beta = 1.0 / (0.00831446 * temperature_value)
    G = np.zeros((len(lam_c_grid), len(lam_v_grid), 2, 2))
    diagnostics = {
        "n_grid_coul": int(len(lam_c_grid)),
        "n_grid_vdw": int(len(lam_v_grid)),
        "requested_steps_per_point": int(n_steps),
        "finite_difference_delta": float(delta),
        "temperature_K": temperature_value,
        "valid_points": 0,
        "unsafe_points": 0,
        "failed_points": 0,
        "under_sampled_points": 0,
        "samples_per_valid_point": [],
        "warning": "",
    }

    for i, lc in enumerate(lam_c_grid):
        for j, lv in enumerate(lam_v_grid):
            if not _is_safe_dual_lambda_state(lc, lv):
                # 用巨大各向同性度量把不安全区域标成“不可通行”。
                G[i, j] = np.eye(2, dtype=float) * 1e8
                diagnostics["unsafe_points"] += 1
                continue

            context.setParameter("lam_coul", float(lc))
            context.setParameter("lam_vdw", float(lv))
            try:
                context.getIntegrator().step(500)
            except Exception as exc:
                print(f"  [WARN] 2D 度量预采样失败 (λc={lc:.3f}, λv={lv:.3f}): {exc}")
                G[i, j] = np.eye(2, dtype=float) * 1e8
                diagnostics["failed_points"] += 1
                continue

            dc_vals, dv_vals = [], []
            full_batches, remainder = divmod(int(n_steps), 50)
            sample_count = full_batches + (1 if remainder else 0)
            for sample_idx in range(sample_count):
                batch_steps = 50 if sample_idx < full_batches else remainder
                if batch_steps <= 0:
                    continue
                try:
                    context.getIntegrator().step(batch_steps)
                    dc_vals.append(_finite_difference_on_safe_region(context, lc, lv, "coul", delta))
                    dv_vals.append(_finite_difference_on_safe_region(context, lc, lv, "vdw", delta))
                    context.setParameter("lam_coul", float(lc))
                    context.setParameter("lam_vdw", float(lv))
                except Exception as exc:
                    print(f"  [WARN] 2D 度量采样失败 (λc={lc:.3f}, λv={lv:.3f}): {exc}")
                    dc_vals = []
                    dv_vals = []
                    break

            if len(dc_vals) < 2 or len(dv_vals) < 2:
                G[i, j] = np.eye(2, dtype=float) * 1e8
                diagnostics["under_sampled_points"] += 1
                continue

            dc, dv = np.array(dc_vals), np.array(dv_vals)
            cov = np.cov([dc, dv]) * beta ** 2
            if not np.all(np.isfinite(cov)):
                G[i, j] = np.eye(2, dtype=float) * 1e8
                diagnostics["failed_points"] += 1
                continue
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-4)
            G[i, j] = eigvecs @ np.diag(eigvals) @ eigvecs.T
            diagnostics["valid_points"] += 1
            diagnostics["samples_per_valid_point"].append(int(len(dc_vals)))
    total_points = int(len(lam_c_grid) * len(lam_v_grid))
    diagnostics["total_points"] = total_points
    diagnostics["valid_fraction"] = float(diagnostics["valid_points"] / max(total_points, 1))
    if diagnostics["samples_per_valid_point"]:
        diagnostics["min_samples_per_valid_point"] = int(min(diagnostics["samples_per_valid_point"]))
        diagnostics["median_samples_per_valid_point"] = float(np.median(diagnostics["samples_per_valid_point"]))
    else:
        diagnostics["min_samples_per_valid_point"] = 0
        diagnostics["median_samples_per_valid_point"] = 0.0
    if diagnostics["median_samples_per_valid_point"] < 20:
        diagnostics["warning"] = (
            "2D geodesic metric was estimated from few derivative samples per point; "
            "treat the path as an efficiency heuristic and inspect overlap diagnostics."
        )
    if return_diagnostics:
        return G, diagnostics
    return G


def optimize_2d_geodesic_path(
    system,
    topology,
    positions,
    box_vectors,
    ligand_indices,
    n_grid: int = 16,
    n_steps_per_point: int = 3000,
    temperature: float = 300.0,
    platform_name: str = "CUDA",
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> List[Tuple[float, float]]:
    """运行 2D 度量张量场采集 + Dijkstra 测地线寻径

    返回从 (1.0, 1.0) 到 (0.0, 0.0) 的最优 (λ_coul, λ_vdw) 路径

    🔑 [0831issue P2] `diagnostics`：调用方可以传一个 dict 进来，本函数会往里写
    寻径过程的可审计事实。**返回值类型刻意不变**（现有调用方与测试都按
    `List[Tuple[float,float]]` 消费），所以用 out-param 而不是改成元组返回。
    写入的键：

      * `fallback` (bool)：寻径是否失败并回退到对角线线性路径。以前这个回退只
        print 一行、返回值与成功路径**完全无法区分**，于是次优路径会被当成功
        路径写进 `geodesic_path.json` 缓存并被后续 run 复用。
      * `fallback_reason` (str|None)：回退原因。
      * `magnitude_gate_dropped_edges` (int)：被 `|g_mid| > 1e7` 量级闸门判为不可
        通行、因而被 Dijkstra 静默丢弃的边数。`g = β²·Cov(dU/dλ)`，`g > 1e7` 对应
        `std(dU/dλ) ≳ 7.9e3 kJ/mol`——软核去 LJ 的陡峭/冲突区并不是真的不可达，
        这个闸门丢边过多正是上面那个静默回退的常见触发路径。本轮**不改闸门阈值**
        （那会改变已验证路径的数值），只把它丢了多少边如实记下来。
    """
    if diagnostics is not None:
        diagnostics.setdefault("fallback", False)
        diagnostics.setdefault("fallback_reason", None)
        diagnostics.setdefault("magnitude_gate_dropped_edges", 0)
    try:
        n_grid_float = float(n_grid)
        n_grid_int = int(n_grid)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"2D geodesic n_grid 必须是整数，收到 {n_grid!r}") from exc
    if not np.isfinite(n_grid_float) or n_grid_float != n_grid_int:
        raise ValueError(f"2D geodesic n_grid 必须是整数，收到 {n_grid!r}")
    if n_grid_int < 2:
        raise ValueError("2D geodesic n_grid 至少为 2（必须包含两个端点）")
    try:
        n_steps_float = float(n_steps_per_point)
        n_steps_int = int(n_steps_per_point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"2D geodesic n_steps_per_point 必须是整数，收到 {n_steps_per_point!r}"
        ) from exc
    if not np.isfinite(n_steps_float) or n_steps_float != n_steps_int:
        raise ValueError(
            f"2D geodesic n_steps_per_point 必须是整数，收到 {n_steps_per_point!r}"
        )
    if n_steps_int < 2:
        raise ValueError("2D geodesic n_steps_per_point 至少为 2")
    try:
        temperature_float = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"2D geodesic temperature 必须是正有限值，收到 {temperature!r}") from exc
    if not np.isfinite(temperature_float) or temperature_float <= 0.0:
        raise ValueError(f"2D geodesic temperature 必须是正有限值，收到 {temperature!r}")
    n_grid = n_grid_int
    n_steps_per_point = n_steps_int
    temperature = temperature_float
    import gc as _gc
    softcore_params = ACESoftcorePotential.optimize_alpha(len(ligand_indices))
    sc_obj = ACESoftcorePotential.from_dict(softcore_params)

    probe_sys = build_aces_probe_system_dual_lambda(
        system, ligand_indices, sc_obj,
        fixed_lam_coul=0.5, fixed_lam_vdw=1.0,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        co_alchemical_ion_spec=co_alchemical_ion_spec,
    )

    resolved_platform_name, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform_name)
    integ = openmm.LangevinMiddleIntegrator(temperature, 1.0/unit.picosecond, 0.002*unit.picosecond)
    ctx = openmm.Context(probe_sys, integ, platform, props)
    ctx.setPositions(positions)
    if box_vectors is not None:
        ctx.setPeriodicBoxVectors(*box_vectors)

    lam_c_grid = np.linspace(1.0, 0.0, n_grid)
    lam_v_grid = np.linspace(1.0, 0.0, n_grid)

    print(f"\n采集 2D 度量张量场 | {n_grid}×{n_grid} 网格 | {n_steps_per_point} 步/点")
    G, metric_diagnostics = compute_2d_metric_grid(
        ctx, lam_c_grid, lam_v_grid,
        n_steps=n_steps_per_point,
        temperature=temperature,
        return_diagnostics=True,
    )

    print(f"  [OK] 度量张量场完成 | 形状: {G.shape}")
    print(
        "  2D 度量诊断: "
        f"valid={metric_diagnostics['valid_points']}/{metric_diagnostics['total_points']} "
        f"({metric_diagnostics['valid_fraction']:.2%}), "
        f"median_samples={metric_diagnostics['median_samples_per_valid_point']:.1f}, "
        f"failed={metric_diagnostics['failed_points']}, unsafe={metric_diagnostics['unsafe_points']}"
    )
    if metric_diagnostics.get("warning"):
        print(f"  [WARN] {metric_diagnostics['warning']}")
    _search_diag: Dict[str, Any] = {}
    try:
        path = dijkstra_monotonic_geodesic(
            G, lam_c_grid, lam_v_grid, diagnostics=_search_diag
        )
        print(f"  测地线路径: {len(path)} 个状态")
        print(f"     λ_coul: {path[0][0]:.3f} → {path[-1][0]:.3f}")
        print(f"     λ_vdw:  {path[0][1]:.3f} → {path[-1][1]:.3f}")
    except Exception as e:
        # [0831issue P2] 回退必须可审计：见本函数 docstring 的 `diagnostics`。
        print(
            f"  [WARN] 测地线寻径失败 ({e})，回退到对角线线性路径 —— "
            "这条路径是次优的，不要把它当成测地线结果引用。"
        )
        path = list(zip(np.linspace(1.0, 0.0, n_grid), np.linspace(1.0, 0.0, n_grid)))
        if diagnostics is not None:
            diagnostics["fallback"] = True
            diagnostics["fallback_reason"] = f"{type(e).__name__}: {e}"
    if diagnostics is not None:
        dropped = int(_search_diag.get("magnitude_gate_dropped_edges", 0) or 0)
        diagnostics["magnitude_gate_dropped_edges"] = dropped
        if dropped:
            print(
                f"  [WARN] 测地线寻径中有 {dropped} 条边被 |g_mid|>1e7 量级闸门判为"
                "不可通行并丢弃（合法的高方差格点也会被它挡住，见 0831issue P2）。"
            )

    path_arr = np.array(path)
    # 确保 lam_coul 和 lam_vdw 严格单调递减 (从 1.0 -> 0.0)
    path_arr[:, 0] = np.minimum.accumulate(path_arr[:, 0])
    path_arr[:, 1] = np.minimum.accumulate(path_arr[:, 1])
    path_arr[:, 1] = np.maximum(path_arr[:, 1], path_arr[:, 0])
    
    # 强制锚定边界
    path_arr[0, :] = [1.0, 1.0]
    path_arr[-1, :] = [0.0, 0.0]
    
    # 去除因单调化可能产生的重复点
    unique_mask = np.abs(np.diff(path_arr, axis=0)).sum(axis=1) > 1e-6
    unique_mask = np.append([True], unique_mask)
    path_arr = path_arr[unique_mask]
    
    path = [tuple(p) for p in path_arr]
    print(f"  测地线路径 (单调性已校准): {len(path)} 个状态")
    
    del ctx, integ, probe_sys
    _gc.collect()
    return path


def _bilinear_interp_metric(G: np.ndarray, ci: float, cj: float) -> np.ndarray:
    """Bilinearly interpolate the 2x2 metric tensor field ``G`` at a
    continuous (fractional) grid-index coordinate ``(ci, cj)``.

    Used by ``_integrated_geodesic_move_cost`` to evaluate the metric at the
    intermediate grid points a long (knight-style) move skips over, rather
    than only ever looking at the two endpoints it actually lands on.
    """
    nc, nv = G.shape[:2]
    i0 = int(np.floor(ci))
    j0 = int(np.floor(cj))
    i1 = min(i0 + 1, nc - 1)
    j1 = min(j0 + 1, nv - 1)
    i0 = min(max(i0, 0), nc - 1)
    j0 = min(max(j0, 0), nv - 1)
    ti = ci - i0
    tj = cj - j0
    top = G[i0, j0] * (1.0 - tj) + G[i0, j1] * tj
    bot = G[i1, j0] * (1.0 - tj) + G[i1, j1] * tj
    return top * (1.0 - ti) + bot * ti


def _integrated_geodesic_move_cost(
    G: np.ndarray,
    lam_c_grid: np.ndarray,
    lam_v_grid: np.ndarray,
    i: int,
    j: int,
    ni: int,
    nj: int,
) -> Optional[float]:
    """Thermodynamic-length cost of one Dijkstra move, integrated along the
    straight line from ``(i, j)`` to ``(ni, nj)`` in as many equal
    sub-segments as the move spans grid cells (``max(|di|, |dj|)``).

    For an adjacent move (``di, dj`` both <= 1) this has exactly one segment
    and reduces to the previous "average the two endpoint metrics" formula.
    For a long (knight-style) move such as ``(1, 2)``/``(2, 1)`` -- added so
    the path can route around a single bad/unsafe grid cell -- the previous
    code used that same single-segment endpoint-average formula across the
    whole jump, never sampling the metric at the point actually being
    skipped over; a high-variance ridge sitting exactly at that skipped
    point (already confirmed "not unsafe" by the caller's bounding-box check,
    just expensive) could be cut through for free. Each sub-segment here
    uses the trapezoidal average of the metric at its own two ends (with
    interior points bilinearly interpolated from the grid field), so the
    integrated cost can no longer ignore a spike sitting between the two
    move endpoints.

    Returns ``None`` if any sampled metric is non-finite or unreasonably
    large (mirrors the previous single-segment finite/magnitude guard).
    """
    n_segments = max(abs(ni - i), abs(nj - j), 1)
    dlc_seg = (lam_c_grid[ni] - lam_c_grid[i]) / n_segments
    dlv_seg = (lam_v_grid[nj] - lam_v_grid[j]) / n_segments
    dlam_seg = np.array([dlc_seg, dlv_seg])

    total = 0.0
    prev_g = G[i, j]
    for k in range(1, n_segments + 1):
        if k == n_segments:
            g_k = G[ni, nj]
        else:
            t = k / n_segments
            g_k = _bilinear_interp_metric(G, i + (ni - i) * t, j + (nj - j) * t)
        g_mid = 0.5 * (prev_g + g_k)
        if not np.all(np.isfinite(g_mid)) or np.max(np.abs(g_mid)) > 1e7:
            return None
        total += float(np.sqrt(max(0.0, dlam_seg @ g_mid @ dlam_seg)))
        prev_g = g_k
    return total


def dijkstra_monotonic_geodesic(
    G, lam_c_grid, lam_v_grid, diagnostics: Optional[Dict[str, Any]] = None
):
    """单调有向图 Dijkstra 寻径 — 在 (λ_coul, λ_vdw) 2D 平面上找最短热力学路径

    [0831issue P2] `diagnostics`（可选 out-param）会收到
    `magnitude_gate_dropped_edges`：被 `_integrated_geodesic_move_cost` 的
    `|g_mid| > 1e7` 量级闸门判为不可通行、因而被这里静默丢弃的边数。丢边太多是
    "终点不可达 → 上层静默回退对角线"的常见前因，必须可见。闸门本身不动。
    """
    import heapq
    nc, nv = G.shape[:2]
    dist = np.full((nc, nv), np.inf)
    prev = np.full((nc, nv), None, dtype=object)
    dist[0, 0] = 0.0
    pq = [(0.0, 0, 0)]

    moves = [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1)]

    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        if i == nc - 1 and j == nv - 1:
            break

        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < nc and 0 <= nj < nv:
                if not _is_safe_dual_lambda_state(lam_c_grid[ni], lam_v_grid[nj]):
                    continue
                if di > 1 or dj > 1:
                    crosses_unsafe = False
                    for ii in range(min(i, ni), max(i, ni) + 1):
                        for jj in range(min(j, nj), max(j, nj) + 1):
                            if not _is_safe_dual_lambda_state(lam_c_grid[ii], lam_v_grid[jj]):
                                crosses_unsafe = True
                                break
                        if crosses_unsafe:
                            break
                    if crosses_unsafe:
                        continue
                # 🔑 之前这里对角/多格移动（尤其是 (1,2)/(2,1) 这类跳过一个格点
                # 的"骑士步"）只算 0.5*(G[i,j]+G[ni,nj])，从不采样被跳过的中间
                # 格点的度量——上面 crosses_unsafe 只确认了中间格点"没有被标记
                # 为不可通行"，不代表它的度量本身很小；一条真实的高方差脊完全
                # 可能就架在这个被跳过的点上，被当作免费近道抄过去。改为沿这条
                # 移动路径按跨越的格数分段积分（相邻移动天然只有一段，行为不变）。
                w = _integrated_geodesic_move_cost(G, lam_c_grid, lam_v_grid, i, j, ni, nj)
                if w is None:
                    # [0831issue P2] 量级闸门/非有限度量弃边，如实计数。
                    if diagnostics is not None:
                        diagnostics["magnitude_gate_dropped_edges"] = int(
                            diagnostics.get("magnitude_gate_dropped_edges", 0) or 0
                        ) + 1
                    continue
                w += 1e-4
                if dist[i, j] + w < dist[ni, nj]:
                    dist[ni, nj] = dist[i, j] + w
                    prev[ni, nj] = (i, j)
                    heapq.heappush(pq, (dist[ni, nj], ni, nj))

    ci, cj = nc - 1, nv - 1
    if not np.isfinite(dist[ci, cj]) or prev[ci, cj] is None:
        raise RuntimeError(
            "测地线寻径失败：lambda 图不连通或终点不可达。"
            "请检查度量张量是否含 NaN/Inf，或回退到线性/对角路径。"
        )
    path = []
    while (ci, cj) != (0, 0):
        path.append((lam_c_grid[ci], lam_v_grid[cj]))
        parent = prev[ci, cj]
        if parent is None:
            raise RuntimeError(
                f"测地线寻径中断：节点 ({ci}, {cj}) 缺少前驱，图可能不连通。"
            )
        ci, cj = parent
    path.append((lam_c_grid[0], lam_v_grid[0]))
    return path[::-1]


if __name__ == "__main__":
    # 只读入口：看 stage-2 现在什么状态、控制器会选什么动作。**不执行任何动作。**
    #   python abfe_preoptimizer.py <run_dir> [stage_name] [--json]
    # lo/hi 与插点预算默认从 <run_dir>/run_provenance.json 的 config 读，不用手传。
    import argparse as _ap

    _p = _ap.ArgumentParser(
        description="Stage-2 统一控制器的只读视图（状态 + 证据 + 下一步动作）"
    )
    _p.add_argument("run_dir")
    _p.add_argument("stage_name", nargs="?", default=None,
                    help="默认列出所有 vanishing* 段（多采样段只看一个会漏）")
    _p.add_argument("--json", action="store_true")
    _p.add_argument(
        "--replay", nargs="*", metavar="RUN_DIR",
        help="影子对账：在这些 run 上**只读**重放决策（不执行、不改预算、不写 run）。"
             "不给路径时把 run_dir 本身当唯一 corpus。",
    )
    _p.add_argument("--allow-untrusted", action="store_true",
                    help="只改 trust_level，**不**把 evidence_status 改写成 CONVERGED")
    _a = _p.parse_args()

    if _a.replay is not None:
        _corpus = list(_a.replay) or [_a.run_dir]
        _traces = Stage2RepairController.replay(_corpus)
        if _a.json:
            print(json.dumps(_traces, indent=2, ensure_ascii=False))
        else:
            print(Stage2RepairController.render_replay(_traces))
        raise SystemExit(0)

    _stages = [_a.stage_name] if _a.stage_name else sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(_a.run_dir, "vanishing*"))
        if os.path.isdir(d)
    ) or ["vanishing"]

    _out = []
    for _st in _stages:
        _c = Stage2RepairController(
            _a.run_dir, _st, allow_untrusted_stage_results=_a.allow_untrusted
        )
        _v = _c.read()
        if _a.json:
            _out.append({"view": _v, "plan": _c.decide(_v)})
        else:
            print(_c.render(_v))
            print()
    if _a.json:
        print(json.dumps(_out if len(_out) > 1 else _out[0], indent=2, ensure_ascii=False))

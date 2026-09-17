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
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple, Optional
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
    # 🔑🔑 [审计 #50，2026-09-14] 见下方 docstring「子路径」一节。
    allow_fixed_23_state_table: bool = True,
) -> List[Tuple[int, int]]:
    """Partition an adaptive lambda path into few-state IBS subintervals.

    ⚠️⚠️ [审计 #49，2026-09-14] **这个函数分的是「等边数」，不是「等热力学长度」。**
    下面那句 "Lambda locations already encode the thermodynamic metric" 的前提
    **不成立**，别照着它推理：

      · 本函数只把 ``lambdas.size`` 传给贪心分组（或走 23 态固定表），
        **全程不看任何度量**；
      · 而 λ 表来自 ``blended_metric_vanishing_lambdas`` 的**混合坐标**
        ``(1−β)·ŝ + β·(1−λ)``（β 是几何地板权重，刻意**不**等弧长），
        之后还会被 ``densify_lambdas_by_free_energy`` 按 |ΔF| 再插点。
      · 所以「等边数」≠「等 ∫g」。实测后果是难度集中在度规尖峰（λ_vdw→0），
        与仓库里反复出现的 window-0 / 端点失败同址。

    真正按 ∫g 均衡的分窗是 ``partition_windows_by_metric_integral``。
    **[2026-09-14 用户拍板] 它现在是生产默认**：``abfe_config.json`` 里
    ``stage2_window_partition = "metric_integral"``、``stage2_window_max_states = 8``
    （max 是卡住均衡的那一项：max=5 会强制 [5,5,5,5,5]，等于退化回等边数）。

    ⟹ **本函数只在显式设 ``stage2_window_partition="arclength"`` 时才会被生产
    路径调用**，其余场合它是历史布局的复现工具（以及 ``repartition_tail_from_anchor``
    的尾段分组）。别再把它当成「生产分窗器」来推理。

    **子路径（审计 #50）**：``allow_fixed_23_state_table=False`` 时，即使
    ``lambdas.size == 23`` 也**不**走那张手工固定表。那张表是**全路径**的
    （含专为 λ=1 耦合端做的 window-0 收窄），而且会**完全忽略**
    ``min/max_states_per_window``。把它套在一段尾段子路径上是张冠李戴：
    子路径的起点根本不是 λ=1。尾段重分（``repartition_tail_from_anchor``）
    必须传 False。

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
    if lambdas.size == VANISHING_FINAL_STATE_COUNT and allow_fixed_23_state_table:
        # 23 态：仍然走原来手工调出来的固定 6 窗表——含 window0 ESS 塌缩修复，
        # 逐字节不变。
        # ⚠️ [审计 #50] 只对**全路径**成立。子路径（尾段重分）必须传
        # `allow_fixed_23_state_table=False`：这张表的 window-0 收窄是为 λ=1
        # 耦合端做的，而子路径的起点不是 λ=1；它还会完全忽略
        # min/max_states_per_window，而尾段重分恰恰要按调用方给的 lo/hi 切。
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


def rebuild_window_dependent_allocation(
    allocation: Dict[str, Any],
    lambdas: Sequence[float],
    window_ranges: Sequence[Sequence[int]],
    *,
    edge_free_energy: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """按新的 ``window_ranges`` 重建 ``subdomain_allocation`` 里**所有按窗口统计**的字段。

    🔑🔑 [2026-09-15] **λ 布点与分窗本来就是两个阶段，耦在一个返回值里才出的事。**
    ``redistribute_vanishing_lambda_subdomains`` 一次性返回 λ **和** ranges，而它的
    ranges 永远是等边数贪心的。于是 ``recompute_vanishing_path_from_cached_pilot``
    （派生层离线重算）顺手把**旧判据的 ranges** 带了回来 —— 生产明明是
    ``stage2_window_partition=metric_integral``。后果：resume 写回缓存的
    ``window_ranges`` 与 ``subdomain_*`` 描述的是另一套布局（``_run_dual_lambda_stage``
    会自己按 metric_integral 重算一遍才没串到生产，但缓存里那份是错的，
    每次 resume 还会打一条"缓存不是 vanishing v12 的热力学 few-state 子区间布局"的 WARN）。

    本函数只改**窗口维**的字段；λ 维的（``realized_max_lambda_gap`` /
    ``edge_free_energy_kJ_mol`` / 布点方法标签等）一个字都不碰 —— 它们与分窗无关。
    """
    out = dict(allocation)
    rngs = [(int(a), int(b)) for a, b in window_ranges]
    lam = [float(x) for x in lambdas]
    counts = [b - a - 1 for a, b in rngs]
    out["total_window_state_slots"] = int(sum(b - a for a, b in rngs))
    out["subdomain_interval_counts"] = counts
    out["subdomain_state_counts"] = [c + 1 for c in counts]
    out["subdomain_lambda_bounds"] = [
        [lam[a], lam[b - 1]] for a, b in rngs
    ]
    out["actual_shared_state_indices"] = [rngs[i][0] for i in range(1, len(rngs))]
    out["actual_state_index_sets"] = [list(range(a, b)) for a, b in rngs]
    if edge_free_energy is not None:
        dF = np.asarray(edge_free_energy, dtype=float)
        out["subdomain_free_energy_kJ_mol"] = [
            float(np.sum(dF[a:b - 1])) for a, b in rngs
        ]
        out["subdomain_max_edge_free_energy_kJ_mol"] = [
            float(np.max(dF[a:b - 1])) if b - 1 > a else 0.0 for a, b in rngs
        ]
    return out


def recompute_vanishing_path_from_cached_pilot(
    path_diagnostics: Dict,
    *,
    n_states: int,
    final_state_count: int,
    min_states_per_window: int,
    max_states_per_window: int,
    free_energy_densify_points: int,
    # 🔑🔑 [2026-09-15] **分窗判据必须跟着一起进来。**
    # 不传就退回 `redistribute_vanishing_lambda_subdomains` 内部那套等边数 ranges，
    # 而生产是 metric_integral ⟹ 离线重算出来的布局与 fresh preopt 不一致。
    # 这三个键都在 `_PREOPT_DERIVED_PATH_KEYS` 里，本来就该由调用方透传。
    partition_criterion: str = "arclength",
    n_windows: Optional[int] = None,
    first_window_max_states: Optional[int] = None,
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

    # 🔑🔑 [2026-09-15] **λ 布点完成之后，分窗单独再算一遍。**
    # `redistribute_vanishing_lambda_subdomains` 把「布点」和「分窗」耦在一个返回值里，
    # 而它的 ranges 永远是等边数贪心的。生产走 metric_integral 时，这里必须用**同一个**
    # 分窗器重算，否则离线重算出来的布局与 fresh preopt 不一致（实质一致性 bug）。
    # ⚠️ 只重算 ranges 与**按窗口统计**的字段；λ 布点（`blended_metric_vanishing_lambdas`
    # / densify）一个字不动 —— 两者本来就该是两个阶段。
    _crit = str(partition_criterion).lower()
    _partition_diag = None
    if _crit in ("metric_integral", "state_count"):
        window_ranges, _partition_diag = partition_windows_by_metric_integral(
            [float(x) for x in optimized_lambdas],
            pilot_lambdas,
            [float(x) for x in metric_g],
            min_states_per_window=int(min_states_per_window),
            max_states_per_window=int(max_states_per_window),
            n_windows=(int(n_windows) if n_windows is not None else None),
            first_window_max_states=(
                int(first_window_max_states)
                if first_window_max_states is not None else None
            ),
            pilot_mean_dU_dlambda=_pilot_mean_gradients_or_none(
                path_diagnostics["pilot_points"]
            ),
            objective=_crit,
        )
        window_ranges = [(int(a), int(b)) for a, b in window_ranges]
        subdomain_allocation = rebuild_window_dependent_allocation(
            subdomain_allocation, optimized_lambdas, window_ranges,
            edge_free_energy=subdomain_allocation.get("edge_free_energy_kJ_mol"),
        )
    return {
        "lambdas_vdw": optimized_lambdas,
        "window_ranges": window_ranges,
        "subdomain_allocation": subdomain_allocation,
        "cumulative_length": cumulative_length,
        "optimized_edge_lengths": optimized_edge_lengths,
        # 缓存里记下**实际用的**判据与 DP 读数，否则"这份 ranges 是按什么切的"
        # 事后无从对账（正是这个 bug 能藏这么久的原因）。
        "partition_criterion": _crit,
        "partition_diagnostics": _partition_diag,
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
    power: float = 1.0,
) -> np.ndarray:
    """把 pilot 的 ∫g^power dλ 累积到给定 λ 表上（``power=0.5`` 即热力学长度 ∫√g）。

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
    gp = g if float(power) == 1.0 else np.power(g, float(power))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (gp[:-1] + gp[1:]) * np.abs(np.diff(pl)))])
    return np.interp(np.asarray(lambdas, dtype=float), pl[::-1], cum[::-1])


def partition_windows_by_metric_integral(
    lambdas: Sequence[float],
    pilot_lambdas: Sequence[float],
    metric_g: Sequence[float],
    *,
    min_states_per_window: int = 4,
    max_states_per_window: int = 8,
    n_windows: Optional[int] = None,
    # 🔑🔑 [2026-09-15] **耦合端（win0）专用的态数上限。**
    # ∫g 均衡**没有 K 这一维**，而 IBS 是一条轨迹重加权到窗内全部 K 个目标态：
    # 同样的 ∫g，K=8 的窗比 K=4 的窗支撑差得多。真机 4W53 vanishing（22 态、
    # 真实 pilot）实测：DP 给出 [8,5,4,4,5]，逐窗 ∫g = 130/130/126/135/113，
    # **不均衡度仅 1.19** —— win0 拿 8 个态不是"被免费吞并"，而是 ∫g 意义上
    # 完全均衡的解；它失败（min N_eff/g=0.40，门 10）恰恰是在 ∫g 均衡的前提下。
    #
    # 为什么**只**卡第一个窗：布局类修复动作一个都够不着 win0 ——
    # `SPLIT_TAIL_WINDOW` 要求 `window_idx > 0`（win0 的首态不是共享态，
    # `tail_repartition_anchor` 返回 None），`INSERT_LAMBDA` 会重排全局边界。
    # ⟹ **win0 的跨度只能在分窗这一刻决定，事后没有任何修复路径。**
    # 实测 cap=4 得 [4,6,4,4,4,5]：maxK 从 8 降到 6，且那个 6 落在 win1
    # （split-tail 够得着）。代价是 +1 个系综；峰值 ∫g 三档都是 134.6，
    # 即"封 win0 在 ∫g 账本上一分钱不赚"——它买的是 K，不是 ∫g。
    first_window_max_states: Optional[int] = None,
    # 🔑 [2026-09-15] **仅报告，不是判据。** 给了 pilot 的 <dU/dλ> 就顺手算出
    # 每个窗口的预测 |ΔF| 跨度（总变差口径，与 `edge_free_energy_kJ_mol` 同一条
    # 积分）落进诊断。它是 f_k 跨度的保守代理，而 f_k 跨度正是 IBS 单系综重加权
    # 难度里 ∫g **看不见**的那一维（真机 win0：∫g 与其余窗齐平、f_k 却跨 58 kJ/mol，
    # min N_eff/g=0.40）。**不设阈值、不参与 DP** —— 本仓库刚因为"未标定的拟合
    # 阈值"退役过整套 rescue 触发条件，这里先攒实测（哪些 span 对应多少 N_eff/g），
    # 标定过了再谈要不要变成硬约束。
    pilot_mean_dU_dlambda: Optional[Sequence[float]] = None,
    # 🔑🔑 [2026-09-17] **目标函数。** ``"state_count"`` 才是数学上正确的那个，
    # 理由见 `_solve` 的 docstring；``"metric_integral"`` 保留原语义不动，
    # 因为它是既有 run 的协议指纹的一部分，不做同名改义。
    objective: str = "metric_integral",
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

    ``n_windows=None`` 时在所有可行窗口数里自动选：峰值 ∫g → 窗口数（少的省
    GPU）→ **最大窗态数 maxK** → ∫g 平方和。

    🔑 [2026-09-15] maxK 是后加的第三个目标，理由见 ``_solve`` 的 docstring：
    ∫g 判据在构造上**看不见 K**，而 IBS 是一条轨迹重加权到窗内全部 K 个目标态，
    同样的 ∫g，K=8 比 K=4 难得多。真机 cyclod_ligand1 上它把首窗从 8 态降到 7 态，
    峰值 ∫g 与窗口数都不变。

    ⚠️ 峰值有地板：它等于尖峰处一个**最小 4 态窗**的 ∫g（上例 74.2）。想再低只能
    在尖峰那段**加 λ 态**，分窗解决不了。
    """
    lam = np.asarray(lambdas, dtype=float).ravel()
    n = lam.size
    lo, hi = int(min_states_per_window), int(max_states_per_window)
    if lo < 2 or hi < lo:
        raise ValueError(f"min/max_states_per_window 非法：{lo}/{hi}")
    first_hi = hi if first_window_max_states is None else int(first_window_max_states)
    if first_hi < lo:
        raise ValueError(
            f"first_window_max_states={first_hi} 小于 min_states_per_window={lo}"
        )
    first_hi = min(first_hi, hi)
    # 约束**进 DP 的候选枚举**，不是解完再过滤。两者不等价：先解再滤只会把
    # 整条布局丢掉（然后回退到窗口数更多的那条），拿不到"在该约束下的最优"。
    _cap = lambda a: first_hi if a == 0 else hi
    if n < lo:
        raise ValueError(f"λ 表只有 {n} 个态，不足 min_states_per_window={lo}")
    gcum = metric_integral_cumulative(lam, pilot_lambdas, metric_g)
    cost = lambda a, b: abs(float(gcum[b - 1] - gcum[a]))
    lcum = metric_integral_cumulative(lam, pilot_lambdas, metric_g, power=0.5)
    length = lambda a, b: abs(float(lcum[b - 1] - lcum[a]))
    _obj = str(objective).lower()
    if _obj not in ("metric_integral", "state_count"):
        raise ValueError(f"未知分窗目标函数 {objective!r}")
    # 三遍 DP 的目标，按字典序。``size`` 用 (j+1-i) 表达，与 `range(lo, cap+1)` 同源。
    if _obj == "state_count":
        _p1 = lambda a, b: float(b - a)      # minimax K —— 主目标
        _p2 = length                         # 同 K 下取最短窗
        _p3 = length                         # 平方和也用长度
    else:
        _p1 = cost
        _p2 = lambda a, b: float(b - a)
        _p3 = cost

    def _solve(w_target: int):
        """(峰值, 最大窗态数, 平方和, ranges)；不可行返回 None。

        🔑🔑 [2026-09-15] **三遍，不是两遍：峰值 → maxK → 平方和。**

        先前是「峰值 → 平方和」，于是在峰值相同的一族解里，K 这一维**完全没人
        看**，由平方和任意决定。真机 cyclod_ligand1（21 态、真实 pilot）实测：
        `[8,6,4,6]` 与 `[7,7,4,6]` 峰值 ∫g **都是 85.5**、窗数**都是 4**，
        DP 按平方和选了前者（20252.7 < 20770.1）⟹ 交付一个 8 态首窗，预测 ΔF
        跨度 74 kJ/mol（其余三窗 5/7/25），而 win0 事后**没有任何布局修复路径**
        （`SPLIT_TAIL_WINDOW` 要求 window_idx>0，`INSERT_LAMBDA` 重排全局边界）。

        ⚠️ 这**不是**在引入阈值。没有新常数、没有 `max_f_span`、不读
        `predicted_f_span_kJ_mol`。只是把「同样的 ∫g，K 越大越难」这条**已经写在
        代码注释里、但判据看不见**的事实，放进字典序的第三位。

        ⚠️ 也**不是**免费的：代价是 ∫g 平方和（上例 +2.6%）。真正免费的那条
        （同窗数同峰值下按 maxK 打平局、平方和不变）经实测**不存在**。
        窗数仍然排在 maxK **之前**（见下面 `candidates` 的排序）—— 少一个系综
        省 250k 步 GPU，比降一档 K 值钱。所以本改动**不会**增加窗口数。
        """
        INF = float("inf")
        best = [[INF] * (w_target + 1) for _ in range(n)]
        back = [[None] * (w_target + 1) for _ in range(n)]
        best[0][0] = 0.0
        for i in range(n):
            for w in range(w_target):
                if best[i][w] == INF:
                    continue
                for size in range(lo, _cap(i) + 1):
                    j = i + size - 1
                    if j > n - 1:
                        break
                    cand = max(best[i][w], _p1(i, j + 1))
                    if cand < best[j][w + 1]:
                        best[j][w + 1] = cand
                        back[j][w + 1] = (i, w)
        peak = best[n - 1][w_target]
        if peak == INF:
            return None
        tol = peak * (1.0 + 1e-12) + 1e-12
        # 第二遍：在"每个窗都不超过 peak"的约束下最小化**最大窗态数**。
        # 与第一遍同构，只是把代价从 ∫g 换成窗口态数（同样是 minimax，
        # 所以同样不能和下面的求和目标合并成一遍）。
        kbest = [[INF] * (w_target + 1) for _ in range(n)]
        kbest[0][0] = 0.0
        for i in range(n):
            for w in range(w_target):
                if kbest[i][w] == INF:
                    continue
                for size in range(lo, _cap(i) + 1):
                    j = i + size - 1
                    if j > n - 1:
                        break
                    if _p1(i, j + 1) > tol:
                        continue
                    cand = max(kbest[i][w], _p2(i, j + 1))
                    if cand < kbest[j][w + 1]:
                        kbest[j][w + 1] = cand
        p2_best = kbest[n - 1][w_target]
        if p2_best == INF:
            return None
        p2_tol = p2_best * (1.0 + 1e-12) + 1e-12
        # 第三遍：在"峰值不超 tol **且**没有窗口超过 max_k"的约束下最小化平方和，
        # 结果确定且在这两个约束下尽量均匀。
        ss = [[INF] * (w_target + 1) for _ in range(n)]
        bk2 = [[None] * (w_target + 1) for _ in range(n)]
        ss[0][0] = 0.0
        for i in range(n):
            for w in range(w_target):
                if ss[i][w] == INF:
                    continue
                for size in range(lo, _cap(i) + 1):
                    j = i + size - 1
                    if j > n - 1:
                        break
                    if _p1(i, j + 1) > tol or _p2(i, j + 1) > p2_tol:
                        continue
                    c = _p3(i, j + 1)
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
        max_k = float(max(b - a for a, b in ranges))
        return peak, max_k, ss[n - 1][w_target], list(reversed(ranges))

    if n_windows is not None:
        got = _solve(int(n_windows))
        if got is None:
            raise RuntimeError(
                f"{n} 个态在 [{lo},{hi}]（首窗 ≤{first_hi}）约束下"
                f"切不出 {n_windows} 个窗口"
            )
        candidates = [(got[0], int(n_windows), got[1], got[2], got[3])]
    else:
        candidates = []
        for w in range(1, n):
            got = _solve(w)
            if got is not None:
                # 排序键：峰值 ∫g → **窗口数** → maxK → ∫g 平方和。
                # 窗数排在 maxK 之前是刻意的：少一个系综省一整块生产预算，
                # 比降一档 K 值钱。把 maxK 提前会让分窗器用 GPU 去买 K。
                candidates.append((got[0], w, got[1], got[2], got[3]))
        if not candidates:
            raise RuntimeError(
                f"{n} 个态在 [{lo},{hi}]（首窗 ≤{first_hi}）约束下无可行窗口划分"
            )
    peak, w_used, max_k, ssq, ranges = min(candidates)

    validate_single_shared_boundary_ranges(ranges, n)
    per_window = [cost(a, b) for a, b in ranges]
    # 热力学长度 ∫√g dλ（与 ∫g dλ 是两个量：前者控相邻重叠，后者近似单系综
    # 重加权的总方差）。逐窗都落盘，省得事后还要重算一遍才能对读。
    _sqrt_g = np.sqrt(np.maximum(np.asarray(metric_g, dtype=float), 1.0e-12))
    _pl = np.asarray(pilot_lambdas, dtype=float)
    _order = np.argsort(_pl)
    _tl_nodes = np.interp(
        np.asarray(lam, dtype=float),
        _pl[_order],
        np.concatenate(([0.0], np.cumsum(
            0.5 * (_sqrt_g[_order][:-1] + _sqrt_g[_order][1:])
            * np.diff(_pl[_order])
        ))),
    )
    _tl_per_window = [abs(float(_tl_nodes[b - 1] - _tl_nodes[a])) for a, b in ranges]
    _df_span = None
    if pilot_mean_dU_dlambda is not None:
        try:
            _edges = edge_free_energy_kJ_mol(
                lam, pilot_lambdas, pilot_mean_dU_dlambda
            )
            _df_span = [float(np.sum(_edges[a:b - 1])) for a, b in ranges]
        except (ValueError, IndexError):
            _df_span = None
    return ranges, {
        # 🔑 [2026-09-17] `objective="state_count"` 时 DP 的主目标是 **max K**，
        # 下面 `peak_metric_integral` 仍如实报 ∫g（从 per_window 算，不是 DP 的 peak）。
        "criterion": ("state_count_minimax" if _obj == "state_count"
                      else "metric_integral_g"),
        "objective": _obj,
        "n_windows": int(w_used),
        "sizes": [int(b - a) for a, b in ranges],
        # [2026-09-15] DP 的**第三个**目标（峰值 → 窗数 → 这个 → 平方和）。
        "max_window_states": int(max_k),
        "first_window_max_states": int(first_hi),
        # 🔑🔑 [2026-09-15 老板定案] **逐窗布局画像，只报告。**
        # 不设 PASS/FAIL、不进 DP 硬过滤、不触发拆窗或补采。等多体系数据能标定
        # 「f-span 对实际支撑的预测能力」之后再谈升级。当前的紧急保护是明确的
        # **结构协议** `first_window_max_states=4`（针对已知的耦合端失效），
        # 而不是假装已经标定出一个通用的 `max_f_span` 阈值。
        # `observed_*` 两项在这里恒为 None —— 它们要等采样跑完才存在，由窗口
        # 自检产物填。放在同一张表里是为了让"预测 vs 实测"能直接对读。
        "window_profile_REPORT_ONLY": [
            {
                "window": int(i),
                "K": int(b - a),
                "metric_integral": float(per_window[i]),
                "thermodynamic_length": float(_tl_per_window[i]),
                "predicted_f_span_kJ_mol": (
                    None if _df_span is None else float(_df_span[i])
                ),
                "observed_f_k_span_kJ_mol": None,
                "observed_min_N_eff_over_g": None,
            }
            for i, (a, b) in enumerate(ranges)
        ],
        "predicted_delta_f_span_kJ_mol_REPORT_ONLY": _df_span,
        "metric_integral_per_window": [float(x) for x in per_window],
        "peak_metric_integral": float(max(per_window)) if per_window else None,
        "peak_primary_objective": float(peak),
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



# v2 [2026-09-13]：决策词汇与终态语义变更 ——
#   · 新增动作 `NO_ACTION`（配 `exit=NO_FEASIBLE_ACTION`）；
#   · stage 判据未通过时**不再**返回 `DONE`，改为按失败的那道门归因分岔
#     （见 `stage_quality_gate_failures` 与 `decide()` 分支 9）。
# v3 [2026-09-15]：**`converged` 删除 + 精度与分析两维分开**（老板定案）——
#   · stage 结果契约换成 `analysis_status` / `analysis_incomplete_reasons` /
#     `precision_status` / `precision_evidence`；`converged` 不再存在，
#     **不做同名改义、不做兼容别名**（故意 fail-loud）；
#   · 逐段质量门降级为**纯报告**（每条带 `drives_action: False`），
#     v2 的分支 9b/9c（按门归因发 GPU 动作）**已删除**；
#   · 新终态出口 `ANALYSIS_COMPLETE_PRECISION_UNMEASURED`：分析完整但精度未测，
#     该停就停，但**不是 `DONE`**；
#   · `DONE` 的充要条件收紧成 `precision_status == MEETS_CROSS_REPEAT_TARGET`
#     —— 单次 run 里恒不成立，`DONE` 因此实际不可达，这是**有意的**。
# ⚠️ 这个版本号**只盖在 read/decide/replay 的 payload 上，不进任何缓存指纹**
#   （全仓 grep 确认），所以升它不会作废任何窗口缓存。
STAGE2_CONTROLLER_PROTOCOL_VERSION = 3


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


# ---------------------------------------------------------------------------
# stage 级质量门的**报告**（不是动作依据）
# ---------------------------------------------------------------------------
# 🔑🔑 [2026-09-15 老板定案] **这些门一条都不许再驱动自动补帧。**
#
# 它们原来是 `solve_stage_integrated` 的 `converged` 那个合取的分项，而
# `converged` 已经被删除（「算出了数」被误报成「精度已验收」）。合取里的四个
# 判据全是**未经标定**的拟合阈值，其中 `max_endpoint_uncertainty_kJ_mol ≤ 1.0`
# 的来历是抄自一个**默认关闭**的 early-stop 启发式的默认形参、全仓无任何依据；
# 而且它是**逐段**量，科学目标是两腿合成的 `σ_bind`（实测各段 `[0.3×5, 2.0]`
# ⟹ σ_bind = 2.98 已达标，门却判未收敛并命令补帧；把那段从 2.0 压到 1.0 要
# 4× 采样量、精度提升为零）。更根本：MBAR 的 σ 是渐近估计，对「该采的构型一次
# 都没采到」结构性失明 —— 4W53 (+32 kJ/mol) 与 decharging (−88 kJ/mol) 两次大错
# σ 都很小、都没报警。
#
# 所以本函数**只剩报告能力**：逐条给出实测值 / 阈值 / 最差窗口，对人工审查有用。
# 每个条目都带 `drives_action: False`，`decide()` 里**没有任何分支读它**。
# 下面三个词是**科学分类**（这道门在物理上属于哪一类失效），不是「对症动作」：
#   · 支撑/偏斜类（目标态支撑度、mixture 重叠）：同分布加帧治不了
#     （§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 0.762、ESS 比值反而变差）；
#   · 样本量类（去相关样本数、端点 σ）：语义是「尚不可测」而不是「不合格」（§3）；
#   · 读不出来的 ⟹ **不归类**，标 `unattributed`。
# ⚠️ 不得据此在 `decide()` 里发 `RUN_PRODUCTION` / `INSERT_LAMBDA` /
#   `SPLIT_TAIL_WINDOW` 等花 GPU 的动作 —— 那正是被删掉的东西。
STAGE_GATE_MORE_SAMPLING = "sample_size_diagnostic"
STAGE_GATE_NARROW_SPAN = "span_diagnostic"
STAGE_GATE_UNATTRIBUTED = "unattributed"

# 🔑🔑 [审计 #47，2026-09-14] **边际增益刹车的三个常量，刻意保守。**
#
# 先前两处判据（`_frames_admission` 的 `max(fin[1:]) <= fin[0]`、`_no_gain` 的
# `max(h[1:]) <= h[0]`）都把 `min N_eff/g` / `solver_n_decorrelated` 当成**随采样
# 单调增**的量。它们不是：
#   · 分子 Kish ESS `(Σw)²/Σw²` 在小 N 时**乐观偏高**（少数几帧的权重还没分化开）；
#   · 分母 ĝ 在 N ≫ τ 之前**单调上涨**（自相关时间被系统性低估，随样本增加才现形）。
# 两个偏差**同向**（都让早期点偏高），而旧判据的基准还偏偏取 `h[0]` —— 偏高最严重
# 的那一点。于是「没涨」几乎必然成立，刹车对正常窗口误触发。
# 这与本文件 `_unreach` 那段**自己否决**「周期内按 n_eff 外推提前判死」的论证
# 直接矛盾：那里承认 g 只在 N ≫ τ 之后才稳。
#
# 改成单侧、带余量的形式：
#   · 至少 `MIN_POINTS` 个点才判（两点根本分不出趋势与噪声）；
#   · 基准是**除最后一点外全部点的中位数**（中位数对早期那个偏高点稳健）；
#   · 只有最后一点**低于基准的 `STALL_RATIO`** 才算"证伪了加帧"。
# ⚠️ 这是**刹车**不是精确判据：它只回答"要不要停止自动补帧"，
# 不回答"这个窗口收敛了没有"。**不要**在这里发明新的统计检验。
MARGINAL_GAIN_MIN_POINTS = 3
MARGINAL_GAIN_STALL_RATIO = 0.9

# 🔑🔑 [审计 #46，2026-09-14] **「加帧治不了」的归因只能有一份判定。**
#
# `window_self_support_check` 写侧特意把三个来源分开：
#   · `top1pct_veto`      权重塌缩（单帧支配）⟹ **加帧治不了**，要缩跨度
#   · `min_n_eff_over_g`  支撑比值偏低       ⟹ 样本量问题，加帧对症
#   · `solver_eligibility` 去相关帧数不够资格 ⟹ 纯样本量，加帧**正是**对症动作
# 而它同时把 `solver_eligibility` 这一档的 verdict **强制置成 `HARD_INSUFFICIENT`**
# （"更前置的失效优先"）。控制器两处判据（`_read_single_stage._support_failed`、
# `decide._is_skew`）只认 verdict、不看来源 ⟹ 长 τ 的解耦端窗口（帧数确实不够）
# 被判成"偏斜、加帧治不了"，送去插 λ —— 而插 λ **不缩短构象慢模态的 τ_int**。
# 写侧自己的注释就是「帧数不够 ≠ 支撑不够」，读侧必须同口径。
# 🔑🔑 [2026-09-15 真机 brd4_ligand1/rep2] **`min_n_eff_over_g` 从这张表里拿掉。**
#
# 上面第 2099 行原来写「`min_n_eff_over_g` 支撑比值偏低 ⟹ 样本量问题，加帧对症」。
# 反例就在同一个 run 里：
#     win1  n_decorr = 888（地板 10 的 **88 倍**）、min N_eff/g = 8.68
# **帧一点都不缺**，缺的是权重压不到目标态上。同分布再加一块，只会得到权重剖面
# 一模一样的更多帧 —— 这正是本仓 §5.1 实测过的：250k→1M 让 top1% 从 0.545 涨到
# 0.762、ESS 比值**反而更差**；而重标定一次 rawESS 27.5→503。
# 真机后果：控制器连发 4 块 `RUN_PRODUCTION` 给 win1，配额烧光 → NO_FEASIBLE_ACTION，
# 而真正缺窗的 win4 全程排在 `blocked_by_upstream` 里、一次都没被看过。
#
# ⚠️ **`solver_eligibility` 必须留下**，理由见上一段（长 τ 的解耦端窗口是真的帧不够，
# 插 λ 不缩短构象慢模态的 τ_int）。这次只动 `min_n_eff_over_g` 一个。
#
# ⚠️ 写侧有个硬不变量让这条改动是安全的：`window_self_support_check` 里
#       solver_eligible = n_dec >= floor
#       if not solver_eligible: verdict_source = "solver_eligibility"   # 覆盖
#   这一步排在比值分档**之后** ⟹ `verdict_source == "min_n_eff_over_g"` 出现时，
#   `n_dec >= floor` **必然成立**（帧数已经够了）。即便如此，下面的
#   `support_failure_is_skew` 仍然接受可选的帧数并自己再验一次 —— 不把正确性
#   押在另一个文件的不变量上。
_SAMPLE_SIZE_VERDICT_SOURCES = ("solver_eligibility",)

# 🔑 [审计 #58] rewindow 子窗在**求解器命名空间**里的索引基数。
# 执行器建窗时按 `10_000 + 100 * n` 钉死（`_immutable_rewindow_step`）；
# 控制器只用它区分「这个 `window_index` 是物理窗口还是子窗」。
SOLVER_UNIT_INDEX_BASE = 10_000


def frames_growth_headroom(view, window_idx=None, *, unit_id=None):
    """把**剩下的补帧配额全花掉**，帧数还能涨几倍。读不出来返回 `None`。

    `(1 + cap) / (1 + used)`：每块补帧加的是一个初始生产块，所以用掉 `used` 块的
    窗口现在有 `(1+used)` 份帧，配额见底时最多 `(1+cap)` 份。
    ponytail: 「一块 = 一个初始块」是实测值（brd4_ligand2/rep1 win2：4 块把
    250k 走到 1.25M），不是从 config 读的。哪天补帧块长度可变，就改成按
    `production_steps` 与本轮块步数算，别在这里加分支。
    """
    cap = view.get("max_production_blocks_per_window")
    if cap is None:
        return None                     # 上限未知 ⟹ 射程算不出来，不猜
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        return None
    if cap < 0:
        return None
    if unit_id is not None:
        rows = (view.get("production_blocks_total_by_unit")
                or view.get("production_blocks_by_unit") or {}).get(str(unit_id))
    else:
        rows = (view.get("production_blocks_total_by_window")
                or view.get("production_blocks_by_window") or {}).get(int(window_idx))
    used = len(rows or [])
    if used >= cap:
        return 1.0                      # 配额已见底 ⟹ 一帧都加不了了
    return (1.0 + cap) / (1.0 + used)


def n_eff_over_g_reachable_by_frames(ratio, target, headroom):
    """**最乐观**假设下，把剩余补帧配额全花掉能不能把 `N_eff/g` 推过门。

    判据是恒等式，不是新阈值：

        N_eff/g = n_decorr × (N_eff/N)

    右边第二项是权重剖面的**效率**，是强度量 —— 同分布再采只放大 n_decorr，
    不改效率。所以「加帧的射程」就是 `ratio × headroom`。

    返回 `None` = 判不了（缺数），调用方按既有行为处理。
    ⚠️ 这是**上界**：实测中效率会随帧数变差（§5.1，250k→1M 让 top1% 0.545→0.762），
    所以 `True` 只表示"值得一试"，不表示"一定推得过"。真正的刹车是事后的
    `marginal_gain_stalled()` —— 两者一前一后，缺一不可。
    """
    if ratio is None or target is None or headroom is None:
        return None
    try:
        r, t, h = float(ratio), float(target), float(headroom)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(r) and np.isfinite(t) and np.isfinite(h)):
        return None
    if h < 1.0 or t <= 0.0:
        return None
    return bool(r * h >= t)


def support_failure_is_skew(
    verdict, verdict_source, *, n_decorrelated=None, min_frames=None,
    min_n_eff_over_g=None, n_eff_over_g_target=None, frames_headroom=None
) -> bool:
    """这次自检失败是**支撑/偏斜类**（加帧治不了）还是**样本量类**（加帧对症）。

    两处调用点（物理窗口 / rewindow 子窗）共用这一份，见上面的长注释。

    `n_decorrelated` / `min_frames` 可选：给了就**自己再验一次**帧数够不够，
    不把正确性押在写侧的覆盖顺序上（见 `_SAMPLE_SIZE_VERDICT_SOURCES` 那段
    ⚠️）。读不到就退回只看 verdict + 来源，行为与先前一致。

    🔑🔑 [2026-09-16] **`min_n_eff_over_g` 偏低不再无条件判成"加帧治不了"。**

    低比值**不蕴含**加帧无用：稳定分布下 `N_eff ∝ N`、`g` 不变 ⟹ 比值随帧数线性
    涨，5 可以长到 10。先前那条规则是从一个真反例（brd4_ligand1/rep2 win1：
    n_decorr=888、比值 8.68 ⟹ 帧一点不缺）推广出来的，但推广过头了：它对
    「帧确实少、比值因此低」的窗口同样成立，于是把一个补帧能治的窗口送去改布局。

    改法不是换个阈值，而是**把射程算出来**（`n_eff_over_g_reachable_by_frames`）：
    剩余配额全花掉仍够不着门 ⟹ 才叫"加帧治不了"。三个数缺任何一个就判不了射程，
    此时保持既有判法（保守当偏斜）—— 老产物、子窗那类拿不到块账的调用点因此
    逐位不变。
    """
    return support_failure_attribution(
        verdict, verdict_source, n_decorrelated=n_decorrelated,
        min_frames=min_frames, min_n_eff_over_g=min_n_eff_over_g,
        n_eff_over_g_target=n_eff_over_g_target,
        frames_headroom=frames_headroom) == SUPPORT_FAILURE_STRUCTURAL


# 归因三态。⚠️ 第三态**不是**"两者之间"，它是「**还没判出来**」。
SUPPORT_FAILURE_SAMPLE_SIZE = "SAMPLE_SIZE"   # 帧确实少 ⟹ 加帧对症
SUPPORT_FAILURE_STRUCTURAL = "STRUCTURAL"     # 权重塌缩/跨度过大 ⟹ 加帧治不了
SUPPORT_FAILURE_UNKNOWN = "UNKNOWN"           # 判不出来 ⟹ **不得**授权任何覆盖


def support_failure_attribution(
    verdict, verdict_source, *, n_decorrelated=None, min_frames=None,
    min_n_eff_over_g=None, n_eff_over_g_target=None, frames_headroom=None
) -> Optional[str]:
    """支撑失败的归因，**三态**。非失败返回 `None`。

    🔑🔑 [2026-09-17，用户拍板 A] **「乐观上界说也许够得着」曾被升级成「这是样本量
    问题」，那是本条链的真错误。**

    真实关系是

        R_future = R_now × H × (η_future / η_now) × (g_now / g_future)，  η = N_eff/N

    而 `n_eff_over_g_reachable_by_frames` 只留下 `R_now × H`，**同时假定 η 与 g 恒定**，
    两个假设实测都朝不利方向走（g 实测 12.25→17.63 = 1.44×；η 另有恶化，
    §5.1 250k→1M 让 top1% 0.545→0.762）。所以它的 `True` 只够说「**还没被证伪**」。

    ⚠️ **不做 `R_now × H ÷ 1.44` 那种"已观测 g 惩罚"**，两个理由（用户核实）：
      · `R_now` 已经是用**当前** g 算出来的，再乘 `g_prev/g_now` 是**重复扣除**；
      · 拿过去的 1.44 用于未来 = 假设未来 g 还会再涨同样比例，**那仍是外推**，
        只是不叫"斜率"。
      · 而且 1.44 只解释 1.44×，文档里的约 2.5× 还含 η 恶化 ⟹ 说"共同根因只有 g"是错的。
    要做数值修正，先得有数据契约：production 的 g 与 η 目前**没有**按
    `path_version/f_k/segment/window` 落进历史（ledger 只有步数、去相关帧数、ratio）。

    三态的判法（严格化的那一条在 `min_n_eff_over_g`）：
      · `SAMPLE_SIZE` 只由**硬证据**给出 —— `solver_eligibility` 那族来源，
        或明确的 `n_decorrelated < min_frames`；
      · 射程 `ratio × headroom ≥ target` 只给 `UNKNOWN`（"可能够得着"），
        **不再**算作"这是样本量问题"；
      · 其余（top1pct 否决、射程都够不着、老产物 + HARD_INSUFFICIENT）= `STRUCTURAL`。

    ⚠️ `support_failure_is_skew()` 的布尔值**逐位不变**（它问的是"是不是 STRUCTURAL"），
    所以既有调用点行为一字不改。变的只有 O1：它以前拿 `not is_skew` 当"样本量类"，
    那等于把 `UNKNOWN` 也算进去了 —— 见 `plan()` 里 O1 那段。
    """
    src = str(verdict_source or "")
    if verdict not in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA"):
        return None           # 通过的窗口同样带 verdict_source，不能只看来源
    if src in _SAMPLE_SIZE_VERDICT_SOURCES:
        return SUPPORT_FAILURE_SAMPLE_SIZE    # 样本量不够 ⟹ 加帧就是对症动作
    if src == "min_n_eff_over_g":
        # 帧数本身就没到地板 ⟹ **硬证据**的样本量问题（先补帧，比值以后再说）。
        # 写侧的覆盖顺序保证这种情形会写成 `solver_eligibility`，这里只是不依赖它。
        if (n_decorrelated is not None and min_frames is not None
                and int(n_decorrelated) < int(min_frames)):
            return SUPPORT_FAILURE_SAMPLE_SIZE
        # 🔑🔑 [2026-09-17 P0] **只有明确算出 `False` 才是 STRUCTURAL。**
        # 先前写成「`True` ⟹ UNKNOWN，其余 ⟹ STRUCTURAL」，于是 `None`
        # （三个输入缺任何一个、门缺失、读数非有限）落进了 STRUCTURAL ——
        # **数据缺口被当成了结构性结论**，而 STRUCTURAL 会授权缩跨度类动作
        # （插 λ / 拆窗 / 有界重窗 / D3 停机）。那是拿"没测出来"当"测出来是坏的"。
        _reach = n_eff_over_g_reachable_by_frames(
            min_n_eff_over_g, n_eff_over_g_target, frames_headroom)
        if _reach is False:
            return SUPPORT_FAILURE_STRUCTURAL   # 完整数据 + 连乐观上界都够不着
        # `True`（只是还没被证伪，见上面的 R_future 展开）与 `None`（判不了）
        # **都不授权任何结构动作**。
        return SUPPORT_FAILURE_UNKNOWN
    if src == "top1pct_veto":
        return SUPPORT_FAILURE_STRUCTURAL       # 权重塌缩是直接证据
    # 🔑 [2026-09-17 P0] **老产物缺来源 ⟹ UNKNOWN，不再按 verdict 兜成 STRUCTURAL。**
    # 那条兜底的原意是"保守、不拿加帧顶替"，但它选错了保守方向：STRUCTURAL
    # **授权**缩跨度，而缩跨度同样是要花 GPU、要改布局的动作。真正保守的是
    # UNKNOWN —— 它两边都不授权（既不补帧也不改布局）。同类的"数据缺口授权"。
    return SUPPORT_FAILURE_UNKNOWN


def marginal_gain_stalled(series) -> Tuple[bool, Dict[str, Any]]:
    """加帧还有没有用。返回 `(是否已被证伪, 诊断)`。见上面三个常量的长注释。"""
    vals = [float(x) for x in (series or []) if x is not None]
    if len(vals) < MARGINAL_GAIN_MIN_POINTS:
        return False, {
            "verdict": "NOT_ENOUGH_POINTS",
            "n_points": len(vals),
            "min_points": MARGINAL_GAIN_MIN_POINTS,
            "series": vals,
        }
    head = sorted(vals[:-1])
    m = len(head)
    baseline = head[m // 2] if m % 2 else 0.5 * (head[m // 2 - 1] + head[m // 2])
    last = vals[-1]
    ratio = (last / baseline) if baseline > 0 else None
    stalled = bool(baseline > 0 and last < baseline * MARGINAL_GAIN_STALL_RATIO)
    return stalled, {
        "verdict": "STALLED" if stalled else "STILL_GAINING_OR_FLAT",
        "series": vals,
        "baseline_median_of_earlier_points": baseline,
        "last": last,
        "last_over_baseline": ratio,
        "stall_ratio": MARGINAL_GAIN_STALL_RATIO,
        "n_points": len(vals),
    }


def _worst_window_by(records, key, *, largest=False):
    """从逐窗记录里挑最差的那个窗口下标；读不出来就返回 None（不猜）。

    🔑🔑 [2026-09-15 真机] **逐窗记录里有一半的指标是「逐 λ 态」的 list。**

    `float(r[key])` 直接假定标量 ⟹ 遇到 list 当场 `TypeError` **炸穿 `decide()`**。
    真机 11 个 run 里 **5 个**这样崩（`top1pct_raw_weight` 是 `list[7]`；
    该窗口记录里**根本没有**标量版本，受门的标量 `max_top1pct_raw_weight`
    在 stage 顶层、不在逐窗记录里）。

    ⚠️ 这是**预先就存在**的 bug（HEAD 上同样崩），先前一直没暴露是因为
    stage 结果从不落盘（`ANALYZE` 轮不到 ⟹ `window_overlap_diagnostics` 读不到）
    ⟹ 这段代码走不到。「退出前必跑一次全路径 ANALYZE」把它变成**每轮必经**。

    归约口径：逐态 list ⟹ 取该窗口**最差的那个态**，方向与选窗一致
    （`largest=True` 时越大越差 ⟹ 取 `max`；否则取 `min`）。这与写侧
    `max_top1pct_raw_weight = max(逐态)` 的口径一致，不新发明。
    读不成数（dict / 字符串 / 全是 NaN / 空 list）⟹ 该窗口**不进候选**，
    与 docstring 的"读不出来就不猜"一致 —— 绝不在这里抛。
    """
    def _scalar(v):
        vals = v if isinstance(v, (list, tuple)) else [v]
        out = []
        for x in vals:
            try:
                fx = float(x)
            except (TypeError, ValueError):
                continue
            if fx == fx and abs(fx) != float("inf"):   # 排掉 NaN / ±inf
                out.append(fx)
        if not out:
            return None
        return max(out) if largest else min(out)

    cand = [
        (_scalar(r[key]), int(r["window_index"]))
        for r in (records or [])
        if isinstance(r, dict)
        and r.get(key) is not None
        and r.get("window_index") is not None
        and _scalar(r[key]) is not None
    ]
    if not cand:
        return None
    return (max(cand) if largest else min(cand))[1]



def _bottleneck_observation(selfchk) -> Dict[str, Any]:
    """从一份 `window_self_support_check` 产物里抽出瓶颈态的 (g, η, ratio) 观测。

    🔑🔑 [2026-09-17] **这是 `n_eff_over_g` 可达性判据的数据契约。**
    现在的射程判据 `R_now × H` 假定 η 与 g 恒定，两个假设实测都朝不利方向走
    （g 实测 12.25→17.63）。要把实测增长纳进来，前提是先**逐块记下** g 与 η ——
    而历史台账原来只有步数、去相关帧数、ratio，连 g 都没有。本函数补的就是这个缺口。

    ⚠️ **本轮只记录，不参与任何决策**（用户拍板：先记、可回填、再 shadow 对比，
    数据够了才谈启用）。所以这里一个阈值都没有。

    ⚠️ 老产物缺键 ⟹ 全部 `None`。**不填 0**：0 会被下游读成"效率为零"，
    而真相是"没测"。三态里这是 UNKNOWN。
    """
    _empty = {
        "bottleneck_state": None, "bottleneck_g": None, "bottleneck_n_eff": None,
        "bottleneck_eta": None, "bottleneck_ratio": None,
        "bottleneck_n_eff_input_n_frames": None,
        "n_eff_frame_set": None, "n_eff_frame_set_n_frames_raw": None,
    }
    if not isinstance(selfchk, dict):
        return dict(_empty)
    k = selfchk.get("worst_state_by_n_eff_over_g")
    if k is None:
        return dict(_empty)
    k = int(k)

    def _at(key):
        arr = selfchk.get(key)
        if not isinstance(arr, (list, tuple)) or not (0 <= k < len(arr)):
            return None
        return arr[k]

    g = _at("statistical_inefficiency_per_lambda")
    ne = _at("n_eff_per_state")
    nin = _at("n_eff_input_n_frames_per_state")
    ratio = _at("n_eff_over_g_per_state")
    eta = None
    if ne is not None and nin:
        try:
            eta = float(ne) / float(nin)
        except (TypeError, ValueError, ZeroDivisionError):
            eta = None
    return {
        "bottleneck_state": k,
        "bottleneck_g": None if g is None else float(g),
        "bottleneck_n_eff": None if ne is None else float(ne),
        "bottleneck_eta": eta,
        "bottleneck_ratio": None if ratio is None else float(ratio),
        "bottleneck_n_eff_input_n_frames": None if nin is None else int(nin),
        "n_eff_frame_set": selfchk.get("n_eff_frame_set"),
        "n_eff_frame_set_n_frames_raw": selfchk.get("n_eff_frame_set_n_frames_raw"),
    }


def stage_quality_gate_failures(stage):
    """stage 结果里**哪几道门没过**，逐条带实测值/阈值/科学分类/最差窗口。

    只读、纯函数。**fail-closed**：值或阈值读不出来时记成
    `unattributed` 失败，绝不当成「过了」—— 「缺证据 ≠ 通过」。

    ⚠️⚠️ [2026-09-15] **纯报告。** 每个条目带 `drives_action: False`，
    `report_category` 是这道门在物理上属于哪一类失效，**不是**"对症动作"。
    `decide()` 不读本函数的任何一条去发动作（见模块顶上那段长注释）。
    原来的 `remedy` 键已删除，不做同名改义 —— 谁还在读它，就让它 KeyError。
    """
    if not isinstance(stage, dict):
        return []
    out = []

    def add(gate, value, threshold, category, worst=None, note=None):
        out.append({
            "gate": gate, "value": value, "threshold": threshold,
            "report_category": category, "worst_window": worst, "note": note,
            # 🔑 这条诊断**不驱动任何动作**。两侧一致：`decide()` 只把整份清单
            # 原样放进返回值供人读，不按它分岔。
            "drives_action": False,
        })

    # 🔑🔑 [审计 #59，2026-09-14] **docstring 说 fail-closed，代码是 fail-open。**
    # 先前每道门都包在 `if <threshold> is not None:` 里 ⟹ **阈值读不出来就静默
    # 当成「这道门过了」**，与紧邻上面那句「缺证据 ≠ 通过」直接矛盾。
    # 而 NaN 更隐蔽：`float("nan") > x` 恒为 False ⟹ 一个 NaN 的端点 σ 会给出
    # `converged=False` + **零条归因** ⟹ `decide()` 落到分支 9d 的
    # `NO_FEASIBLE_ACTION`（「归因不到任何一道具体的门」），而真相是「这道门的
    # 读数是 NaN」。`ibs_engine` 算 `converged` 时显式 `np.isfinite(...)` 守过，
    # 这里没有 —— 同一道门两份实现，只有一份认得 NaN。
    def _num(x):
        """能比较的有限数 → float；None / 非数 / NaN / inf → None（= 没测出来）。"""
        if x is None or isinstance(x, bool):
            return None
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if bool(np.isfinite(v)) else None

    def add_numeric_gate(gate, raw_value, raw_threshold, category, worst_key=None,
                         *, largest_is_worse=False, fails_when_greater=False):
        """一道数值门的统一判读，**三态**：过 / 没过 / 没测出来。

        「没测出来」含**阈值缺失**与**值不是有限数**两种，一律记成
        `unattributed` 失败 —— 没有哪一类会改路由（本函数整体不驱动动作），
        但它让报告说出真话：「这道门没判成」而不是「没有任何门失败」。
        """
        # ⚠️ **「这道门不适用」与「这道门判不了」是两回事，别混。**
        # 值和阈值**都**不在 ⟹ 这份 stage 结果压根没带这道门（不同 stage / 旧产物
        # 都会这样）⟹ 跳过，不记失败。只报一条「阈值缺了」的 unattributed 会把
        # 每一份不带全部门的结果都污染成有失败，9d 的理由文本反而更难读。
        # 只要**有一侧在**，这道门就算在场 ⟹ 另一侧缺了就是「判不了」，记 unattributed。
        if raw_value is None and raw_threshold is None:
            return
        v, thr = _num(raw_value), _num(raw_threshold)
        if thr is None or v is None:
            add(gate, raw_value, raw_threshold, STAGE_GATE_UNATTRIBUTED,
                note=("阈值读不出来" if thr is None else
                      f"实测值不是有限数（{raw_value!r}）⟹ 这道门没判成，"
                      "**不得当成通过**"))
            return
        failed = (v > thr) if fails_when_greater else (v < thr)
        if failed:
            add(gate, v, thr, category,
                _worst_window_by(recs, worst_key, largest=largest_is_worse)
                if worst_key else None)

    # 0) 求解器自己就报了错（端点段帧数不够、拼接不上、pymbar 缺失……）。
    #    错误字符串里带 `decorrelated_samples` 的是样本量问题，其余不归类。
    err = stage.get("error")
    if err:
        add("solver_error", str(err), None,
            STAGE_GATE_MORE_SAMPLING if "decorrelated_samples" in str(err)
            else STAGE_GATE_UNATTRIBUTED)

    # 1) 求解器把窗口丢了（`len(local_results) == len(valid_windows)` 那一条）。
    #    被跳过的窗口有**上游专门的分支**在处理；走到这里说明那条没抓住，
    #    所以不归类。**这里不发任何动作** —— 本函数只报告。
    _in = stage.get("input_window_indices")
    _solved = stage.get("solved_window_indices")
    if isinstance(_in, list) and isinstance(_solved, list) and len(_solved) < len(_in):
        add("windows_dropped_by_solver",
            sorted(set(map(int, _in)) - set(map(int, _solved))), None,
            STAGE_GATE_UNATTRIBUTED)

    recs = stage.get("window_overlap_diagnostics") or []

    # 2) mixture 重叠：**跨度类**诊断（同分布加帧治不了）。仅报告。
    add_numeric_gate("min_overlap", stage.get("min_overlap"),
                     stage.get("min_overlap_threshold"),
                     STAGE_GATE_NARROW_SPAN, "min_ess_ratio")

    # 3) 物理目标态支撑度（raw ESS / top1% 权重集中度）：同样是**跨度类**。仅报告。
    tsg = stage.get("target_support_gate")
    if isinstance(tsg, dict) and tsg.get("passed") is False:
        # ⚠️ 最差窗口必须按**这道门自己判的那个量**挑。先前挑的是 `absolute_ess`
        # —— 那是去相关后的 mixture 覆盖度（共模已扣掉、且早已降级为纯诊断），
        # 与本门判的 raw 单参考量是两把尺子：实测同一批窗口 mixture 0.4684 /
        # raw 0.0196，排序可以完全不同 ⟹ 会把 λ 插到不是元凶的窗口上。
        # （TODO.md「四个量不许混用」说的就是这件事。）
        _ibs_gate = tsg.get("ibs_segment_gate") or {}
        # stage 级 `failed_checks` 只到「IBS 段/端点段」这一层（`ibs_segment_target_support`），
        # 具体是 raw ESS 还是 top1% 写在嵌套的 IBS 段门里 —— 两层都要看。
        _checks = [str(c) for c in ((tsg.get("failed_checks") or [])
                                    + (_ibs_gate.get("failed_checks") or []))]
        # 🔑🔑 [审计 #60，2026-09-14] **top1% 先前是「无条件」覆盖 raw ESS 的归因。**
        # 先前条件是 `any("top1pct" in c) or _worst is None` —— 只要失败清单里
        # 提到 top1%，就把 `worst_window` 换成 top1% 最大的那个窗口，**即使 raw ESS
        # 那条也失败、并且指了另一个窗口**。这与紧邻上面那段注释（「最差窗口必须按
        # 这道门自己判的那个量挑」「两把尺子排序可以完全不同」）自相矛盾：两条子门
        # 同时失败时，它单方面让 top1% 那把尺子赢。
        # 现在：raw ESS 那条失败且归因得出窗口时，**保留 raw ESS 的归因**（它是这道
        # 门的主判据）；只有 raw 那条没失败、或它归因不出窗口时才改用 top1%。
        # 两把尺子的读数**都**带进 note，归因信息不丢 —— 不合并、不加权、不排序。
        _raw_failed = any(("absolute_ess" in c) or ("raw" in c and "ess" in c.lower())
                          for c in _checks)
        _top1_failed = any("top1pct" in c for c in _checks)
        _worst_raw = _worst_window_by(recs, "raw_min_absolute_ess")
        _t1 = _worst_window_by(recs, "top1pct_raw_weight", largest=True)
        _worst, _worst_src = _worst_raw, "raw_min_absolute_ess"
        if ((_top1_failed and not _raw_failed) or _worst_raw is None) and _t1 is not None:
            _worst, _worst_src = _t1, "top1pct_raw_weight"
        add("target_support_gate", tsg.get("failed_checks"),
            {"raw_min_absolute_ess": tsg.get("raw_min_absolute_ess_threshold"),
             # 合并后的 stage 级门只带 raw ESS 那个阈值，top1% 的在 IBS 段的门里。
             "max_top1pct_raw_weight": (
                 tsg.get("max_top1pct_raw_weight_threshold")
                 or _ibs_gate.get("max_top1pct_raw_weight_threshold"))},
            STAGE_GATE_NARROW_SPAN, _worst,
            note={"worst_window_selected_by": _worst_src,
                  "worst_by_raw_min_absolute_ess": _worst,
                  "worst_by_top1pct_raw_weight": _t1,
                  "raw_ess_check_failed": _raw_failed,
                  "top1pct_check_failed": _top1_failed,
                  "do_not_merge": "两把尺子排序可以完全不同（实测 mixture 0.4684 / "
                                  "raw 0.0196），不得合并、加权或互相替代"})

    # 4) 去相关样本数：**样本量类**（语义是「尚不可测」而不是「不合格」）。仅报告。
    add_numeric_gate("min_decorrelated_samples",
                     stage.get("min_decorrelated_samples"),
                     stage.get("min_decorrelated_samples_threshold"),
                     STAGE_GATE_MORE_SAMPLING, "n_frames_decorrelated")

    # 5) 端点不确定度：**样本量类**，无逐窗归因。仅报告。
    #    ⚠️ 这道门的阈值（1.0 kJ/mol）抄自一个默认关闭的 early-stop 启发式的默认
    #    形参、全仓无依据，而且它是**逐段**量而目标是合成的 σ_bind ——
    #    **绝不得**再据此命令补帧（2026-09-15 定案，见模块顶上那段长注释）。
    add_numeric_gate("max_endpoint_uncertainty_kJ_mol",
                     stage.get("max_endpoint_uncertainty_kJ_mol"),
                     stage.get("max_endpoint_uncertainty_kJ_mol_threshold"),
                     STAGE_GATE_MORE_SAMPLING, None, fails_when_greater=True)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 出口注册表：**唯一**一张表，`EXITS` / `TERMINAL_EXITS` 全部从它派生。
#
# 🔑🔑 [2026-09-17，用户拍板] **`_RETIRABLE_EXITS` 删除，可退役性升成 plan 的一等语义。**
#
# 旧设计把**三个正交概念**混进字符串名单，分散在三张手写表里：
#   ① 为什么停（exit reason）
#   ② 停的是**谁**（窗口 / stage / 路径）
#   ③ 主循环终不终止（terminal）
# 必然继续漂。实证四次：
#   · `HALT_LAMBDA_BUDGET_INSUFFICIENT` 只在 `EXITS`、不在 `TERMINAL_EXITS`
#     ⟹「看起来像终止、实际不终止」（审计 #14）；
#   · `_RETIRABLE_EXITS` 漏了 `SUPPORT_ATTRIBUTION_UNKNOWN`（2026-09-17 补）；
#   · **还漏了 `D3_REWINDOW_DEPTH_EXHAUSTED`** —— 它同样是窗口局部卡死，
#     却会停掉一个**仍有其他工作**的 stage（用户 2026-09-17 抓出，本次热修）；
#   · `HALT_FRAMES_ADMISSION_CAP` 在两张表里的注释**互相矛盾**：`EXITS` 那边写
#     「刻意**不**进 `TERMINAL_EXITS`」，而它就在 `TERMINAL_EXITS` 里。
#
# 新契约：`plan()` 返回 `halt_scope` / `blocking_window` / `blocking_unit`，
# `_retirable_window()` **只看 scope、不认 exit 字符串**。
# 以后新增一个卡住原因，只在这张表里声明一次；漏了立刻报错，不会再静默停整跑。
#
#   TARGET_LOCAL  —— 当前窗口/单元无路，**允许换窗**（退役换窗的唯一入口）
#   STAGE_GLOBAL  —— 整条 stage 无路，换谁都没用
#   PATH_INVALID  —— 身份/布局/证据不成立，**不得绕过**
#                    （绕过去采下游 = 拿一条已知不成立的路径继续烧 GPU）
HALT_SCOPES = ("TARGET_LOCAL", "STAGE_GLOBAL", "PATH_INVALID")


class ExitSpec(NamedTuple):
    terminal: bool
    # `None` = **不给默认值，发出点必须显式传**。只有 `NO_FEASIBLE_ACTION` 是这样：
    # 它同时承载"这个窗口无路"和"整条 stage 无路"两种语义，给任何默认都会猜错一半。
    default_scope: Optional[str]
    why: str


EXIT_SPECS: Dict[str, ExitSpec] = {
    # ── 终态 ────────────────────────────────────────────────────────────────
    "DONE": ExitSpec(True, "STAGE_GLOBAL", "跨重复精度达标；单次 run 里恒不成立"),
    "DONE_UNTRUSTED": ExitSpec(True, "STAGE_GLOBAL", "门未过但调用方显式放行"),
    "ANALYSIS_COMPLETE_PRECISION_UNMEASURED": ExitSpec(
        True, "STAGE_GLOBAL",
        "硬不变量全过但精度从未被测过 ⟹ 这一跑没有别的事可做，但**不是 DONE**，"
        "结果不得作为已验收结果发布"),
    "GLOBAL_BUDGET_EXHAUSTED": ExitSpec(
        True, "STAGE_GLOBAL", "全局生产预算真的没了（**局部**耗尽不算）"),
    "HALT_FRAMES_ADMISSION_CAP": ExitSpec(
        True, "STAGE_GLOBAL",
        "**所有**窗口的补帧块数配额都用尽。它只在全窗满额时才发（部分满额走"
        "「剔掉满额的、只补剩下的」）⟹ 换谁都没用，是 stage 级。"
        "⚠️ 它先前在 `_RETIRABLE_EXITS` 里，那是**冗余且误导**：按定义此时没有"
        "别的窗口可退役到，退役判定的最后一道守卫必然返回 None。"),
    "HALT_LAMBDA_BUDGET_INSUFFICIENT": ExitSpec(
        True, "STAGE_GLOBAL",
        "插点预算耗尽 + 没有别的可行动作 ⟹ λ 总数不够是**输入问题**，交人工改输入"),
    "D3_REWINDOW_DEPTH_EXHAUSTED": ExitSpec(
        True, "TARGET_LOCAL",
        "有界重窗这条路对**这个单元**用尽（数据模型只能表达一次物理父窗替代）。"
        "🔑 [2026-09-17 热修] 它是**窗口局部**卡死，先前不在 `_RETIRABLE_EXITS` 里 ⟹ "
        "会停掉一个仍有其他工作的 stage。"),
    "SUPPORT_ATTRIBUTION_UNKNOWN": ExitSpec(
        True, "TARGET_LOCAL",
        "这个单元的失败分不出是样本量还是结构性 ⟹ 两边都不授权。"
        "别处还有活就绕过它继续调度（用户规格：不能让它停掉整跑）"),
    "HALT_INVALID_INPUT": ExitSpec(
        True, "PATH_INVALID",
        "身份/输入不一致 ⟹ 盘上的结论属于**另一个系综**，绕过去采下游无意义"),
    "HALT_EVIDENCE_CONTRADICTS_DONE": ExitSpec(
        True, "PATH_INVALID",
        "`DONE` 的证据不是 `CONVERGED` ⟹ 这份证据本身不成立"),
    "NO_FEASIBLE_ACTION": ExitSpec(
        True, None,
        "**没有默认 scope**：它同时承载「这个窗口无路」（可换窗）与「整条 stage "
        "无路」（换谁都没用）。每个发出点必须显式传 `halt_scope=`，漏传 `plan()` 抛错 —— "
        "静默按某个默认处理，要么停掉还有活干的 stage、要么绕过一条不成立的路径。"),
    # ── 路由信号（**必须**被主循环消费，不得终止）────────────────────────────
    "HALT_BUDGET": ExitSpec(False, None, "全局预算耗尽但动作本身可行 ⟹ 换动作继续"),
    "HALT_LOCAL_VALIDATION_CAP": ExitSpec(
        False, None, "撞单周期验证**批次**上限，而全局预算还有钱（≠ HALT_BUDGET）"),
    "HALT_VALIDATION_BUDGET_UNREACHABLE": ExitSpec(
        False, None,
        "算术上证明剩余预算内凑不够去相关帧数。**只改路由、不改 verdict** —— "
        "证据仍是 UNMEASURED，绝不变成 REJECTED"),
    "HALT_NO_ATTRIBUTION": ExitSpec(
        False, None, "测不动且归因不出来 —— stage 级「没有任何门失败」，合法结局"),
    "HALT_FK_REFUTED": ExitSpec(
        False, None,
        "f_k 被有统计功效地驳回 ⟹ 换 Epoch。**是路由**：发它的分支给的是一个"
        "**要执行的动作**（`RECALIBRATE_FK`），进终态集会让那次重标定根本不执行"),
}


def classify_layout_evidence(segments, path) -> Dict[str, Any]:
    """按**当前布局**给逐段窗口证据分类。两个视图构造器共用的唯一实现。

    🔑🔑 [2026-09-17] **先前这套判定只活在 `read_aggregated()` 的一个闭包里**
    （`_layout_matches`），于是单段视图（直接构造 `Stage2RepairController(run, stage)`）
    产不出 `out_of_range_windows` / `unverifiable_layout_evidence` /
    `window_provenance` / `aggregated_segments` 四个键，而：
      · `_coverage_incomplete()` 读 `out_of_range_windows` ⟹ 单段上「证据里混进了
        当前布局根本没有的窗口号」这道闸**恒不触发**；
      · 单段的 `stale_layout_evidence` 是按 `stale_layout_evidence_only` 标记算的，
        而那个标记**只有聚合视图的占位记录才会设** ⟹ 单段上它恒为 `{}`，
        「证据被布局变更作废」这道闸同样恒不触发。
    受影响的是直接构造器 / `decision_trace()` / 非 `--replay` 的 CLI 路径 / 离线脚本；
    `replay()` 走 `for_physical_stage()`，不受影响。

    ⚠️ **不能靠给单段补四个空默认值了事** —— 那只让 schema 表面一致，守卫照样静默
    失效。判定逻辑必须是同一份，否则两边规则还会再漂一次。

    `segments`：`[(段名, [窗口记录, ...]), ...]`，段号升序。
    `path`：`_read_path()` 的返回（要 `lambdas_vdw` / `window_ranges`）。

    四态（顺序有意义，照抄原闭包）：
      · `OUT_OF_RANGE`  当前布局里根本没有这个下标（旧布局遗留产物）。
        **越界检查不依赖 λ 值**，只要有 `window_ranges` 就判得了，所以排在最前 ——
        放到 `not cur_lam` 之后会被它整个吞掉（老产物最常见的形状就是没落 λ 值）。
      · `UNCHECKABLE_NO_CURRENT_LAMBDAS`  连当前 λ 表都没有，无从核对。不记账。
      · `UNVERIFIABLE`  证据自己没带 λ 列表 ⟹ 核不了（≠「核对过、不匹配」）。
      · `STALE`         核对过、与当前布局不符。
      · `MATCH`         身份一致。
    ⚠️ 只有 `STALE` 表示"这份证据不能用"；另外三种**照旧参与合并**（判死会重现
    2026-09-15 那次 win4 占位记录死锁），但必须**记下来**让下游有得判。
    """
    cur_lam = list((path or {}).get("lambdas_vdw") or [])
    cur_rng = [tuple(r) for r in ((path or {}).get("window_ranges") or [])]
    verdict: Dict[Tuple[str, int], str] = {}
    stale: Dict[int, List[str]] = {}
    unverifiable: Dict[int, List[str]] = {}
    out_of_range: Dict[int, List[str]] = {}
    for seg, wins in segments:
        for w in (wins or []):
            i = int(w["window_idx"])
            if cur_rng and i >= len(cur_rng):
                out_of_range.setdefault(i, []).append(seg)
                verdict[(seg, i)] = "OUT_OF_RANGE"
                continue
            if not cur_lam:
                verdict[(seg, i)] = "UNCHECKABLE_NO_CURRENT_LAMBDAS"
                continue
            got = w.get("lambdas_vdw")
            if not got:
                unverifiable.setdefault(i, []).append(seg)
                verdict[(seg, i)] = "UNVERIFIABLE"
                continue
            a, b = cur_rng[i]
            want = cur_lam[a:b]
            if len(want) != len(got):
                stale.setdefault(i, []).append(seg)
                verdict[(seg, i)] = "STALE"
                continue
            # 🔑 **必须跟引擎用同一个栅格。** 先前某版写 `abs(x-y) <= 1e-9`，
            # 而 `ibs_engine` 判 λ 身份是量化到 LAMBDA_GRID_DECIMALS 位后精确比较；
            # 路径落盘量化 / 采样未量化之间天然差 5e-9 ⟹ 引擎说"匹配"、这里说"过期"。
            from ibs_engine import LAMBDA_GRID_DECIMALS as _GD
            _q = lambda vs: [round(float(x), _GD) + 0.0 for x in vs]   # noqa: E731
            if _q(want) != _q(got):
                stale.setdefault(i, []).append(seg)
                verdict[(seg, i)] = "STALE"
                continue
            verdict[(seg, i)] = "MATCH"
    return {
        "verdict_by": verdict,
        "stale": stale,
        "unverifiable": unverifiable,
        "out_of_range": out_of_range,
    }


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
      · **状态唯一真源是盘。** 本类只读盘 + 纯判断，**不执行任何动作**；
        唯一的例外是 `write_comparison_manifest()`（把纯聚合结果落一份盘），
        它不改任何**决策输入**。[审计 #63] 原话「不改任何文件」与该方法直接矛盾，
        照字面读会让人以为控制器绝不写盘、从而放心地并发调用它。
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
        # （"候选也救不了 ⟹ 布局动作" 用的就是上面那个 `INSERT_LAMBDA`，
        #   2026-09-13 前这里重复列了一次，同一个词在清单里出现两遍。）
        # [2026-09-14 裁决] 固定 λ 表上的**有界**重窗：一次只修一个被归因的窗口，
        # 建可完整覆盖其原区间的重叠子系综，只采一个预算块，然后立刻
        # 读盘 → 求解 → 读回求解器帧数与跳窗 → 回到 decide()。
        # 它**不等同于** INSERT_LAMBDA / SPLIT_TAIL_WINDOW：那两个修不了固定 λ 表
        # 下的中间窗。也**不是**已退役的 bridge rescue（批量建系综、一次重解、
        # 直接撞门）—— 那条不许搬回来。
        "IMMUTABLE_REWINDOW",
        # 🔑 [2026-09-15] 单周期验证**批次上限**打满、Δf−ΔF 从未被求出（无结论、
        # **未被驳回**）⟹ 用这份冻结 f_k 跑**一块**诊断生产。
        # **与 RUN_PRODUCTION 绝不合并**，两点不同：
        #   · RUN_PRODUCTION 的前提是这个窗口的 f_k **已经通过**冻结验证；
        #     这个动作恰恰相反 —— f_k 从未被验证过，执行器必须显式授权引擎
        #     （`provisional_production_windows`），否则引擎会抛路由信号。
        #   · 它的产物 `bias_status=provisional_production` / 证据仍是
        #     `indeterminate`，**不是可信 PASS**；warmup_failure.json 保留为证据。
        # 没有这个动作时，这样的窗口进不了生产、又拿不到新证据 ⟹ 整跑停在
        # NO_FEASIBLE_ACTION（真机 cyclod_ligand2/rep2 win4）。
        "PROVISIONAL_PRODUCTION",
        "ANALYZE",              # 只读：跑 stage 分析（MBAR + 生产质量门）
        "DONE",
        # 🔑 [2026-09-13] **「没有可做的动作」必须有自己的词**，不能借 `DONE`。
        # `DONE` 的定义是「全路径证据齐备且达标」（设计 §2）；归因成功但动作都不可行、
        # 或干脆归因不出来时，结果**没有达标**，写 `DONE` 既与 §2 冲突，又会让
        # `plan()` 把 `execution_status` 算成 `COMPLETE`（它按 `action == "DONE"` 判）。
        # 配 `exit_="NO_FEASIBLE_ACTION"` 使用 ⟹ terminal=True、execution_status=HALTED。
        # 终态在主循环里**先于**执行器分发被 break，所以执行器不需要认识它。
        "NO_ACTION",
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
    # 🔑🔑 [2026-09-17，用户拍板] **两张表都从 `EXIT_SPECS` 派生，不再手写。**
    # 手写两张表的代价已经实证四次（漏项、互相矛盾的注释、静默停整跑）——
    # 理由与逐个出口的语义全部写在模块级 `EXIT_SPECS` 那张表里，只此一份。
    EXITS = tuple(EXIT_SPECS)
    TERMINAL_EXITS = tuple(k for k, v in EXIT_SPECS.items() if v.terminal)
    # 🗑️ [2026-09-13 清理] 删掉两个**从未被发出过**的死词：
    #   · `HALT_NO_FEASIBLE_ACTION` —— 真正在用的是 `NO_FEASIBLE_ACTION`（无 HALT_
    #     前缀）。它不只是冗余：**它不在 `TERMINAL_EXITS` 里**，谁照着这份清单发它，
    #     `terminal` 就不成立、主循环不 break —— 一个看起来像终止、实际不终止的词。
    #   · `HALT_TRUNCATED_PATH` —— 缺窗口现在走 `decide()` 分支 6b（`RUN_PRODUCTION`，
    #     无出口、是路由），因为「缺窗口」的正确位置在因果顺序**之后**而不是终止。
    # 两者全仓（含 docs/）零引用，删除是纯清理。由
    # `tests/test_stage2_controller_vocabulary.py` 钉住不再漂移。

    # `bias_status` 六个值混了"还在流程中"（前三）与"已有裁决"（后三）。
    # 拆开才知道是"在跑"还是"有结论了"。
    _PHASE = {
        "unconverged": "WARMUP_LEARN",
        "calibrated_pending_validation": "WARMUP_VALIDATE",
        "frozen_validation_indeterminate": "WARMUP_VALIDATE",
        "converged": "PRODUCTION",
        # 临时生产：已经在生产里（Epoch 已花掉、产物在盘上），但 f_k 从未验证过。
        # 强度差异由 `f_k_evidence_status`（= indeterminate）表达，不在这里混。
        "provisional_production": "PRODUCTION",
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
        effective_config: Optional[Dict[str, Any]] = None,
    ):
        self.run_dir = os.path.abspath(run_dir)
        self.stage_name = str(stage_name)
        self.stage_type = str(stage_type)
        # 🔑 lo/hi **默认从 run 自己的 run_provenance.json 读**，不要求调用方记得传。
        # 手动传错的后果很实在：可拆区间是 [2lo−1, 2hi−1]，4/5 是 7..9、4/8 是
        # 7..15，判出来的"可不可行"会完全不同。run 自己记了它跑的是什么，就用那个。
        #
        # 🔑🔑 [2026-09-15 审计 #1] **但"run 自己记了"这件事只对复合物腿成立。**
        # `_write_run_provenance` 只往**总**输出目录写一份 `run_provenance.json`，
        # 而溶剂腿的 run_dir 是 `output_dir/solvent_leg`（`runabfe.py` 约 7515），
        # 那底下没有这个文件 ⟹ `_cfg` 为空 ⟹ 分窗判据、插点上限、块数上限、
        # 生产预算**全部退回默认值**，与复合物腿不是同一套。真机实测两腿读出：
        #     分窗判据 metric_integral vs arclength
        #     每块生产步数 500000 vs 250000
        #     生产总预算 2000000 vs 未知
        #     插点上限 8 vs 3；每窗块数上限 7 vs 4
        # 同一次 run 的两条腿用不同预算和不同修复策略，而这个差异只在日志里
        # 露出一行 `config_source`。
        #
        # 修法：调用方**显式传**解析后的有效配置（`effective_config`），它优先于
        # 盘上那份。不传时行为逐字不变（仍读 provenance）—— 离线 replay 与旧
        # 产物不受影响。绝不改成"去父目录找一找"：那是靠目录结构猜配置，
        # 换个布局就又静默错一次。
        # 🔑 **合并，不是替换。** 三层优先级：调用方显式给的 > run 自己记的 >
        # 读侧默认值。
        # ⚠️ 写成"给了 effective_config 就不读 provenance"会**倒退**：调用方只传
        # 得出它手上有的那几个键，而 provenance 里可能记着更多（复合物腿就是），
        # 于是没被显式传的键从"provenance 里的真值"掉回"默认值" —— 修溶剂腿的洞
        # 反而把复合物腿弄降级。
        # ⚠️ 值为 `None` 的键**不算给过**：读侧一律 `_cfg.get(k, <默认>)`，而
        # `{"k": None}.get("k", 4)` 返回 None 不是 4。「未知」用**缺键**表达。
        # 🔑 [2026-09-16] **两边都要滤 None，不是只滤调用方那半边。**
        # 原来只有 `_explicit` 滤了，而 `run_provenance.json` 的 config 是
        # `argparse` 的完整命名空间落盘的 —— 未给的开关一律记成 `null`。真机
        # 13 个 run 的 provenance 每一份都带 `stage2_production_budget_steps: null`
        # 等 17~18 个 None 键。今天不炸只是因为那几个键恰好没走 `int()`；
        # `stage2_max_production_blocks_per_window` 走了 `int()`，一旦它哪天以
        # `null` 进 provenance 就是同一个 `TypeError: int(None)`（09-15 已踩过一次）。
        # 「未知」只有一种表达：**缺键**。
        _disk = {
            k: v for k, v in (
                (self._json(os.path.join(self.run_dir, "run_provenance.json")) or {})
                .get("config") or {}
            ).items()
            if v is not None
        }
        _explicit = {k: v for k, v in (effective_config or {}).items()
                     if v is not None}
        _cfg = dict(_disk)
        _cfg.update(_explicit)
        # 🔑🔑 [2026-09-15] **留着它，因为 `read_aggregated()` 要造子控制器。**
        # 段级子控制器不带这份配置的话，合并视图的生产账是从**子控制器**的
        # `_production_budget_inputs` 生成的（见 read_aggregated 的
        # `production_budget`）⟹ cap/块大小又退回默认，父控制器收到什么都没用。
        # 实测：父 cap=2,000,000 → 视图 cap_known=False；父块 500k → 视图 250k。
        self._explicit_config = dict(_explicit)
        self.config_source = (
            "caller:effective_config+run_provenance.json"
            if (_explicit and _disk) else
            "caller:effective_config" if _explicit else
            "run_provenance.json" if _disk else "caller/default"
        )
        self.lo = int(min_states_per_window if min_states_per_window is not None
                      else _cfg.get("stage2_window_min_states", 4))
        self.hi = int(max_states_per_window if max_states_per_window is not None
                      else _cfg.get("stage2_window_max_states", 5))
        self.max_path_insertions = int(
            max_path_insertions if max_path_insertions is not None
            else _cfg.get("max_path_insertions", 3)
        )
        # 🔑 [2026-09-15] 尾段重分必须用**与生产同一个**分窗判据；从 run 自己记的
        # config 读，缺省与 `_run_dual_lambda_stage` 那侧的缺省一致。
        self.partition_criterion = str(
            _cfg.get("stage2_window_partition", "arclength")
        ).lower()
        self.allow_untrusted = bool(allow_untrusted_stage_results)
        # 🔑🔑 [裁决 2，2026-09-14] **生产补帧预算是独立的一本账。**
        # 先前控制器只有 `warmup_steps_left`（`per_window_budget_remaining` 也是它），
        # 于是「还能不能补帧」根本判不出来，只好拿预热余额去终止一个只花生产帧的
        # 动作。现在上限从 config 读；**读不到就是 unknown，不是 0，也不许用预热
        # 余额代替** —— 上限未知时"预算耗尽"这个结论本身不成立。
        # 🔑 [审计 #31] **这两个键的缺省语义（读侧口径，写侧由 CLI/config 提供）：**
        #   · `stage2_production_budget_steps` 缺省 = **unknown**（不是 0、不是无限）
        #     ⟹ `cap_known=False` ⟹ `plan()` 的生产预算闸**不拦**。
        #     「预算耗尽」是个断言，上限未知时这个结论本身不成立。
        #   · `stage2_max_production_blocks_per_window` 缺省 = **4**，而且
        #     **必须有界** —— cap 缺省既然是 unknown，块数上限就是唯一的刹车；
        #     默认无限正是 BUD-03 要修的那个行为（真机单窗烧到 150 万步）。
        _cap = _cfg.get("stage2_production_budget_steps")
        self.production_cap_steps = (
            int(_cap) if isinstance(_cap, (int, float)) and int(_cap) > 0 else None
        )
        self.production_cap_source = (
            "config:stage2_production_budget_steps"
            if self.production_cap_steps is not None else "unknown"
        )
        # 🔑 **cap 的范围定死为「生产步数」**（名字就叫 production）。
        # 先前建窗准入拿「预热 + 生产」去比这个 cap、完成后却只计生产，
        # 计量范围两头不一致 ⟹ 准入比实际严、账又对不上。
        # 预热/验证走**另一本账**（`warmup_steps_left`），不进这里。
        self.production_block_steps = int(_cfg.get("n_steps_per_window", 250_000))
        # 🔑 **每个采样单元的补帧块数硬上限。** 没有它，补帧就没有停止条件
        # （真机走到过 150 万步）。config 没给就用一个**有界**的默认值 ——
        # 绝不能默认无限，那正是要修的那个行为。
        self.max_blocks_per_window = max(1, int(
            _cfg.get("stage2_max_production_blocks_per_window", 4)))
        # 新系综（rewindow 子窗）要预留的生产额度：**每个子窗一个首块**。
        self.new_ensemble_reserve_steps = int(self.production_block_steps)
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
        # 🔑 [审计 #57，2026-09-14] **事件要沿祖先链数，不是 glob 数所有版本文件。**
        # 先前 `glob("path_versions/v*.json")` 把**孤儿版本**（写了文件但没推进
        # `path_current.json` 指针 —— 那是设计内的正常残留）也算进插点预算，于是
        # 预算被永久占用、插 λ 提前判不可行；而且绕过了 `lambda_path_versions._validate()`
        # 的 `content_sha256` 校验（读到被改过的版本文件也不会发现）。
        # docstring 自己写的就是「必须数**整条链**」，实现与它相反。
        # 链式回溯已经有权威实现（`lambda_path_versions.history()`，带环检测 +
        # `_validate`），直接用它，不在这里再写一遍。
        try:
            import lambda_path_versions as _lpv_events
            for _rec in _lpv_events.history(_pd):
                kind = (_rec.get("event") or {}).get("kind")
                if kind:
                    events[kind] = events.get(kind, 0) + 1
        except Exception:  # noqa: BLE001
            # 读不到 / 链损坏 ⟹ 退回逐文件扫描（保守：宁可多算也不少算插点数，
            # 少算会让插点预算失效，那正是原来的失败形状）。
            for vf in sorted(glob.glob(os.path.join(_pd, "path_versions", "v*.json"))):
                # 🔑 kind 在 `record["event"]["kind"]`，不在顶层。读错键 ⟹ `events`
                # 恒为 {} ⟹ `path_insertions_left` 恒等于满额预算。
                kind = ((self._json(vf) or {}).get("event") or {}).get("kind")
                if kind:
                    events[kind] = events.get(kind, 0) + 1
        ranges = (record or {}).get("window_ranges")
        return {
            "version": version,
            "events": events,
            "n_states": len((record or {}).get("states", [])) or None,
            "window_ranges": [tuple(int(x) for x in r) for r in ranges] if ranges else None,
            # 当前布局的 λ 表。判"某份证据是不是这个布局产出的"要靠它。
            "lambdas_vdw": [
                float(st.get("lambda_vdw"))
                for st in ((record or {}).get("states") or [])
                if st.get("lambda_vdw") is not None
            ] or None,
        }

    def layout_change_invalidated_segments(self) -> List[str]:
        """布局变更**之前**就已经存在的采样段 —— 它们描述的是另一个窗口几何。

        🔑 [2026-09-17] 与 `classify_layout_evidence` 同一个理由抽出来：先前它只在
        `read_aggregated()` 里算，单段视图没有这个键。它是**全局事实**（读路径版本
        链上当前那条事件），跟合并没有关系，两个视图构造器算出来必须一样。

        真机 15:59 实证过为什么需要它：插 λ 之后旧段的 min N_eff/g 描述的是另一个
        跨度，拿它判「加帧有没有用」会让判据读出一个不存在的趋势 —— 那次连插两次
        把末窗顶到 K=6 越界炸掉。
        """
        try:
            import lambda_path_versions as _lpv_inv
            _ev = (_lpv_inv.load_current(self.path_checkpoint_dir)
                   or {}).get("event") or {}
            if _ev.get("kind") not in ("insert_lambda", "tail_repartition"):
                return []
            # `note` 是 2026-09-14 之后的落点（不进 event_id）；`detail` 是之前的
            # 旧记录，必须继续读得到，否则老 checkpoint 的过期段保护会静默失效。
            return sorted({
                str(x) for x in (
                    (_ev.get("note") or {}).get("segments_before_change")
                    or (_ev.get("detail") or {}).get("segments_before_change")
                    or [])
            })
        except Exception:  # noqa: BLE001 —— 读不到就当没有变更过，与原实现一致
            return []

    def comparison_manifest(self, view=None) -> Dict[str, Any]:
        """把 A/B 对比要用的量**拍平到一份**。纯聚合，不新算任何东西。

        动机：outer-λ 增强采样要跟 baseline 对照，而现在这些数散在
        6 个窗口 × 多个采样段的几十个文件里（`*_self_support.json` /
        `*_convergence.json` / `dual_join_*` / 版本链 / 自治历史 / 探针）。
        对比时一个个去捞既慢又容易捞错段（段号 ≠ 新旧，见 read_aggregated 注释）。

        三块内容，缺一不可：
          · **身份** —— 这是哪条臂、哪套 λ、哪些协议版本。不钉住身份的对比是空的。
          · **验收量** —— 逐窗 min N_eff/g 及其逐态剖面、ΔF±σ、门的结论。
          · **代价** —— 步数与开过几个 Epoch。outer 臂每步更贵，
            只比 ΔG 不比代价等于没比。
        """
        view = view if view is not None else self.read()
        path = self._read_path()
        stage = self._read_stage_result() or {}
        hist = self._json(os.path.join(
            self.path_checkpoint_dir, "stage2_autonomous_history.json")) or {}
        probe = view.get("fk_probe") or {}

        windows = []
        total_prod = 0
        # 🔑 [2026-09-14] **读不到步数的窗口不按 0 步计。**
        # 先前 `int(... or 0)` 把未知压成 0 再累加 ⟹ A/B 的代价栏系统性偏低，
        # 而代价栏是本函数 docstring 里点名的三大必看之一
        # （「只比 ΔG 不比代价等于没比」）。未知不参与求和，并显式报出有几个未知。
        n_unknown_prod = 0
        # 🔑🔑 [2026-09-16] **"无结论"必须和"不存在"分开。**
        # 固定预算模式（自治控制器关闭）下，冻结验证在单周期批数上限内始终求不出
        # Δf−ΔF 的窗口被记成 `indeterminate`（见 abfe_pipeline `_guarded_once`）。
        # 它**没有** self_support 产物（生产就没跑完）⟹ 下面每个验收量都是 None
        # ⟹ 报告里显示成 `·`，和"这个窗口根本不在布局里"长得一模一样。
        # 那正是"静默从配对分析里删掉"，是明令禁止的。所以这里把 stage 结果里的
        # `indeterminate_windows` 读出来，逐窗打标 + 带上原因。
        _indet = {}
        for _x in (stage.get("indeterminate_windows") or []):
            try:
                _indet[int(_x["window_idx"])] = _x
            except (KeyError, TypeError, ValueError):
                continue
        for w in view.get("windows") or []:
            i = int(w["window_idx"])
            prod = w.get("production_steps")
            if prod is None:
                n_unknown_prod += 1
            else:
                total_prod += int(prod)
            windows.append({
                "window_idx": i,
                "segment": w.get("segment"),
                "lambdas_vdw": w.get("lambdas_vdw"),
                "lambda_span": w.get("lambda_span"),
                "n_states": w.get("n_states"),
                # —— 验收量 ——
                "verdict": w.get("self_verdict"),
                "min_n_eff_over_g": w.get("min_n_eff_over_g"),
                "worst_state_by_n_eff": w.get("worst_state_by_n_eff"),
                "n_decorrelated": w.get("self_n_frames_decorrelated"),
                "min_frames_floor": w.get("self_min_frames"),
                "solver_eligible": w.get("self_sufficient"),
                "verdict_source": w.get("self_verdict_source"),
                # None = 有结论（通过或不通过）；非 None = **这个窗口没测出来**。
                # 下游必须据此拒绝为它计算任何 A/B 差值，也不得把它当 0 或失败。
                "indeterminate": (
                    None if i not in _indet
                    else {"reason": (_indet[i] or {}).get(
                              "reason", "frozen_validation_budget_indeterminate"),
                          "detail": str((_indet[i] or {}).get("detail") or "")[:400]}),
                # 判据量是**块内独立 ESS**，不是前缀差（见 `_read_window` 的长注释）。
                "block_local_ess_by_block": w.get("block_local_ess_by_block"),
                "derailment_status": w.get("derailment_status"),
                "derail_at_trajectory_fraction": w.get("derail_at_trajectory_fraction"),
                "cumulative_fk_residual_span_kJ_mol": w.get("cum_fk_span"),
                "cumulative_fk_residual_verdict": w.get("cum_fk_verdict"),
                # —— 代价 ——
                "production_steps": prod,
                "n_frames": w.get("n_frames"),
                "warmup_steps_spent": w.get("warmup_steps_spent"),
                "warmup_steps_cap": w.get("warmup_steps_cap"),
            })

        actions = [
            {"iteration": it.get("iteration"), "action": it.get("action"),
             "windows": it.get("windows"), "exit": it.get("exit"),
             "routing_signal": it.get("routing_signal")}
            for it in (hist.get("iterations") or [])
        ]
        return {
            "manifest_version": 1,
            "generated_for": "baseline_vs_outer_lambda_ab_comparison",
            # ── 身份：不钉住这些，两条臂的数字不可比 ──
            "identity": {
                "run_dir": self.run_dir,
                "stage": getattr(self, "_stage_base", self.stage_name),
                "stage_type": self.stage_type,
                "path_version": path.get("version"),
                "path_events": path.get("events"),
                "n_states": path.get("n_states"),
                "lambdas_vdw": path.get("lambdas_vdw"),
                "window_ranges": [list(r) for r in (path.get("window_ranges") or [])],
                "aggregated_segments": view.get("aggregated_segments"),
                "min_states_per_window": self.lo,
                "max_states_per_window": self.hi,
            },
            # ── 路径级结果 ──
            "path_result": {
                "total_delta_G_kJ_mol": stage.get("total_delta_G"),
                "total_error_kJ_mol": stage.get("total_error"),
                # [2026-09-15] `converged` 已删除（不做兼容别名）。导出新契约两维，
                # 缺键 fail-closed：读不到 `analysis_status` 一律记 INCOMPLETE。
                "analysis_status": (stage.get("analysis_status")
                                    if stage.get("analysis_status") in
                                    ("ANALYSIS_COMPLETE", "ANALYSIS_INCOMPLETE")
                                    else "ANALYSIS_INCOMPLETE"),
                "analysis_incomplete_reasons":
                    stage.get("analysis_incomplete_reasons") or [],
                "precision_status": stage.get("precision_status") or "UNMEASURED",
                "precision_evidence": stage.get("precision_evidence") or {},
                "path_is_complete": stage.get("path_is_complete"),
                "analysis_mode": stage.get("analysis_mode"),
                "skipped_windows": view.get("skipped_windows"),
                # [审计 #58] 子窗与物理窗口是两个命名空间，导出层也必须两个都带 ——
                # 只搬前一个会让所有后分析层（`stage2_ab_report.py` 等）对子窗
                # 结构性失明（写侧有、导出层漏，是「读一个没人写的键」的镜像）。
                "skipped_sampling_units": view.get("skipped_sampling_units"),
                "missing_windows": view.get("missing_windows"),
            },
            "windows": windows,
            # ── 代价：只比 ΔG 不比代价等于没比 ──
            "cost": {
                "total_production_steps": total_prod,
                # 未知不参与求和；有未知时这个总数是**下界**，如实标出来。
                "n_windows_with_unknown_production_steps": int(n_unknown_prod),
                "total_production_steps_is_lower_bound": bool(n_unknown_prod),
                "n_sampling_segments": len(view.get("aggregated_segments") or []),
                "n_autonomous_iterations": len(actions),
                "per_window_budget_remaining": view.get("per_window_budget_remaining"),
            },
            # ── 控制器都做了什么（对比时解释差异从哪来）──
            "controller": {
                "actions": actions,
                "fk_probe_verdict": probe.get("verdict"),
                "fk_probe_recommended_windows": probe.get(
                    "recalibration_recommended_windows"),
                "stale_layout_evidence": view.get("stale_layout_evidence"),
            },
        }

    def write_comparison_manifest(self, view=None, filename=None) -> str:
        """把 `comparison_manifest()` 落到 checkpoints，返回路径。

        ⚠️ **消费者是 `stage2_ab_report.py`**（`manifest_for()`）—— 改名必须同时改
        它，否则 A/B 报告静默读到 None。2026-09-14 改名时它已加了新旧两级回退。

        🔑 [审计 #63] 文件名**不得以 `stage2_` 开头**：`_read_stage_result()` 的
        兜底分支 glob 的正是 `stage2_*.json`，把控制器自己的产物放进那个命名空间，
        等于让「读 stage 结果」有机会读到「控制器对 stage 结果的摘要」。
        （今天它恰好被 `analysis_status`/`total_delta_G` 的内容检查挡住 —— 那是**巧合**，
        不是设计；manifest 以后多一个同名键就会真的命中。）
        """
        payload = self.comparison_manifest(view)
        out = os.path.join(
            self.path_checkpoint_dir,
            filename or "controller_comparison_manifest.json",
        )
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, out)
        return out

    # 候选 stage 结果文件，**按权威性排序**（前面的优先）。
    # ⚠️ 名字顺序 ≠ 新旧：`stage2_vanishing.json`（stage 缓存完成标记，可能是**几轮
    # 之前**写的）字典序排在 `stage2_vanishing_autonomous_inprogress.json 之前。
    _STAGE_RESULT_CANDIDATES = (
        "stage2_{stage}_autonomous_inprogress.json",   # 自治循环每轮 ANALYZE/rewindow 后写
        "stage2_{stage}.json",                          # stage 缓存完成标记
    )

    def _production_blocks_ledger(self, windows, path_version, units=None):
        """逐窗补帧块账：**已经批过几块、每块换来了什么**。

        🔑 [审计 #8/#9，契约 B] 返回 `{"by_window": {...}, "by_unit": {...}}` ——
        **两本账**。子窗（sampling unit）有自己的目录/checkpoint/冻结 f_k，它的块
        必须记在自己名下；先前只有 window 那本，而子窗一个都不在 `view["windows"]`
        里 ⟹ 子窗的块数硬上限与边际刹车**双双失效**（BUD-03 修过的「单窗烧 150 万步」
        在子窗这个新位置原样复发）。

        没有这本账，补帧就没有停止条件：真机实测可以一路 25→50→…→150 万步 ——
        早期分支（求解器跳窗 ⟹ 补帧）走不到后面的边际收益刹车，而停滞保护又把
        "多跑了一块"算成盘面进展、把重复计数清零。
        **「又产生了帧」不等于「获得了有用证据」。**

        判据量取**求解器侧**的去相关帧数（不是自检那份，两者实测差 2–8 倍，
        更不是步数）。

        可比性同 S2-C：同 `path_version`、同段才放进同一条轨迹。
        ⚠️ 这条「同段」是**边际增益曲线**的口径，由
        `test_frames_admission_and_stop.py::test_blocks_from_another_segment_do_not_count`
        钉住，**不要动**：跨 f_k epoch 比数字是没有意义的（设计 §6.10）。
        「这个窗口一共烧过几块 GPU」是**另一个量**，走
        `_production_blocks_total_by_window()`，见那里的 BUD-03 注释。
        """
        return self._production_blocks_scan(windows, path_version,
                                            same_segment_only=True, units=units)

    def _production_blocks_total_by_window(self, windows, path_version, units=None):
        """逐窗**跨全部采样段**的补帧块数。只给「块数硬上限」用。
        同上，返回 `{"by_window": ..., "by_unit": ...}` 两本账。

        🔑🔑 [BUD-03，2026-09-14] **换段不该把已经烧掉的 GPU 退回去。**

        先前「块数硬上限」和「边际增益」共用同一本账，而那本账只留**同段**的行。
        换段正是循环自己的动作（换 Epoch 就换段）⟹ 窗口每换一次段，配额清零、
        重新发满 4 块。真机 cyclod_ligand2/rep1 win5：history 里记的是 `vanishing`，
        当前视图是 `vanishing_7`，于是全部被过滤掉、`production_blocks_by_window`
        直接是空 `{}` —— 硬上限与边际刹车**一次都没触发过**，实测单窗烧到
        1,500,000 步（cyclod_ligand1/rep3 win4、cyclod_ligand2/rep1 win4）。

        两者的可比性要求本来就不同，所以分成两个量、各自过滤：
          · **硬上限** = 资源账，跨段累计（本函数）；
          · **边际增益** = 同一条曲线上的趋势，必须同段（`_production_blocks_ledger`）。
        """
        return self._production_blocks_scan(windows, path_version,
                                            same_segment_only=False, units=units)

    # 🔑 [审计 #40②] **哪些动作确实花掉一个生产块。**
    # 先前只数 `RUN_PRODUCTION`。`PROBE_REANCHOR_EPOCH` 明确「候选 f_k + 独立 burn-in
    # + 一个 +250k 块」、`RECALIBRATE_FK` 派发时 `probe_only=False` 会**开新段跑满**
    # `n_steps_per_window` —— 两个都实打实烧 GPU，却一块都没记进块账 ⟹ 块数硬上限
    # 和边际刹车都看不见它们烧掉的量。
    # [2026-09-15] `PROVISIONAL_PRODUCTION` 采的就是一个实打实的生产块，
    # 不记进块账 = 给它开一条绕过块数硬上限与边际刹车的旁路。
    _BLOCK_CHARGING_ACTIONS = (
        "RUN_PRODUCTION", "PROBE_REANCHOR_EPOCH", "RECALIBRATE_FK",
        "PROVISIONAL_PRODUCTION",
    )

    def _production_blocks_scan(self, windows, path_version, *,
                                same_segment_only, units=None):
        out: Dict[int, List[Any]] = {}
        out_u: Dict[str, List[Any]] = {}
        try:
            hist = self._json(os.path.join(
                self.path_checkpoint_dir, "stage2_autonomous_history.json")) or {}
            # 单段视图的窗口记录没有 `segment` 字段（那是合并视图加的）⟹ 用本 stage
            # 的名字兜底，两侧同口径，否则账永远是空的、准入门形同虚设。
            _base = getattr(self, "_stage_base", self.stage_name)
            seg_of = {int(w["window_idx"]): (w.get("segment") or _base)
                      for w in (windows or [])}
            # ⚠️ 子窗没有 `segment`：可比性维度是 `identity`（每个子系综一份 f_k）。
            id_of_unit = {str(u["unit_id"]): u.get("identity")
                          for u in (units or []) if u.get("unit_id")}
            for it in (hist.get("iterations") or []):
                if path_version is not None and it.get("path_version") != path_version:
                    continue
                if it.get("action") not in self._BLOCK_CHARGING_ACTIONS:
                    continue
                # 🔑🔑 [审计 #40①] **只给这一轮实际点名的窗口记账。**
                # 先前给快照里**每个**窗口都记一行 —— 而快照是全窗的。于是一轮只补
                # win0 的迭代，会把 win1..win5 的配额也各扣一格；换段导致步数变化的
                # 窗口更是白占（去重键含 `production_steps`）。配额被别人的动作吃掉，
                # 真正在烧 GPU 的那个窗口反而提前撞上限。
                _named = it.get("windows")
                _named_set = ({int(x) for x in _named}
                              if isinstance(_named, (list, tuple)) and _named else None)
                for sn in (it.get("snapshot") or []):
                    i = int(sn.get("window_idx", -1))
                    if _named_set is not None and i not in _named_set:
                        continue
                    _seg = sn.get("segment") or _base
                    # 🔑 [审计 #41] **「这个窗口现在在哪个段」读不到时，不做同段过滤。**
                    # 先前 `_seg != seg_of.get(i)`：窗口不在当前视图里（证据被布局变更
                    # 作废、或该窗口这一轮没有产物）时 `seg_of.get` 返回 `None`，与任何
                    # 段名恒不相等 ⟹ **全部行被丢掉** ⟹ 边际增益刹车对「证据残缺」的
                    # 窗口静默失效 —— 而那正是最需要刹车的那一类窗口。
                    # 未知就是未知：保留全部行，并在行里标注段口径不可判。
                    _seg_known = int(i) in seg_of
                    if same_segment_only and _seg_known and _seg != seg_of.get(i):
                        continue          # 换段 = 换 f_k，不是同一条曲线
                    out.setdefault(i, []).append({
                        "iteration": it.get("iteration"),
                        "segment": _seg,
                        "segment_comparability": (
                            "same_segment" if _seg_known else "segment_unknown"),
                        "production_steps": sn.get("production_steps"),
                        "solver_n_decorrelated": sn.get("solver_n_decorrelated"),
                        "min_n_eff_over_g": sn.get("min_n_eff_over_g"),
                    })
                # ── 第二本账：子窗（契约 B）──────────────────────────────
                # 老 history 没有 `sampling_units_snapshot` ⟹ 「没有记录」，
                # 第一块照常批。**但**有些历史把子窗行直接写在 `snapshot` 里
                # （带 `unit_id`），那也是真实烧过的块，不能漏账 —— 按数组优先、
                # 不并用，避免同一块被两种形状各记一次（多记会提前刹车）。
                _uid_named = str(it.get("unit_id") or "") or None
                _u_rows = it.get("sampling_units_snapshot")
                if not _u_rows:
                    _u_rows = [r for r in (it.get("snapshot") or [])
                               if isinstance(r, dict) and r.get("unit_id")]
                for su in _u_rows:
                    uid = str(su.get("unit_id") or "")
                    if not uid:
                        continue
                    # 与窗口侧同一条规矩：只记这一轮实际点名的那个单元。
                    # 迭代没写 `unit_id`（老格式）⟹ 无从点名，全记（保守：宁可
                    # 多记也不少记，少记就是配额白送）。
                    if _uid_named is not None and uid != _uid_named:
                        continue
                    # 子窗的"段"是 `identity`；写在 `snapshot` 里的老形状只有
                    # `segment`（子系综目录名），两者都能承担"换系综就不可比"。
                    _uid_ident = su.get("identity") or su.get("segment")
                    _ident_known = (uid in id_of_unit
                                    and id_of_unit.get(uid) is not None
                                    and su.get("identity") is not None)
                    if (same_segment_only and _ident_known
                            and _uid_ident != id_of_unit.get(uid)):
                        continue
                    out_u.setdefault(uid, []).append({
                        "iteration": it.get("iteration"),
                        # 子窗这本账的"段"就是 `identity`；去重/同段过滤都用它。
                        "segment": _uid_ident,
                        "identity": _uid_ident,
                        "segment_comparability": (
                            "same_identity" if _ident_known else "identity_unknown"),
                        "production_steps": su.get("production_steps"),
                        # 执行器 snapshot 的输出键名沿用块账消费侧现有读法：
                        # 源是 `u["solver_n_frames_decorrelated"]`，落盘键是
                        # `solver_n_decorrelated`（同名不同源，别混）。
                        "solver_n_decorrelated": su.get("solver_n_decorrelated"),
                        "min_n_eff_over_g": su.get("min_n_eff_over_g"),
                    })

            def _dedup(rows):
                # 去重键含段：跨段扫描时，不同段的同一个 production_steps 是**两块**
                # 帧，只按步数去重会把跨段的块吞掉（那正是 BUD-03 要修的漏账）。
                # ⚠️ 步数未知的行**各算一块**（键里保留 `None`，不压成 0）——
                # 压成 0 会把多块"读不到步数"的行去重成一块，块数硬上限跟着少算。
                def _k(r):
                    _s = r.get("production_steps")
                    if _s is None:
                        return (str(r.get("segment") or ""), "unknown",
                                int(r.get("iteration") or 0))
                    return (str(r.get("segment") or ""), int(_s), 0)

                seen, uniq = set(), []
                for r in sorted(rows, key=lambda x: (
                        str(x.get("segment") or ""),
                        int(x.get("production_steps") or 0),
                        int(x.get("iteration") or 0))):
                    k = _k(r)
                    if k in seen:
                        continue
                    seen.add(k)
                    uniq.append(r)
                return uniq

            for i, rows in out.items():
                out[i] = _dedup(rows)
            for uid, rows in out_u.items():
                out_u[uid] = _dedup(rows)
        except Exception:  # noqa: BLE001 —— 读不到就当没有历史（第一块照常批）
            return {"by_window": {}, "by_unit": {}}
        return {"by_window": out, "by_unit": out_u}

    def _read_stage_result(self) -> Optional[Dict[str, Any]]:
        """stage 级结果（含生产质量门、缺窗口清单）。**只在 stage 跑完才存在。**

        🔑🔑 [CTL-01，2026-09-14] **不靠文件名顺序选权威。**
        先前这里 `sorted(glob("stage2_*.json"))` 取第一个带 `converged` 的 ——
        而 `.`(46) < `_`(95) ⟹ `stage2_vanishing.json` **永远排在**
        `stage2_vanishing_autonomous_inprogress.json 之前。于是自治循环每轮
        （ANALYZE / rewindow 之后）刚写出的新结果**根本读不到**，子窗的 solver
        帧数与跳窗状态全从一份过期结果里读 —— 11/20 的新证据进不了 `decide()`。
        （那句"真缓存一旦存在就排在前面、优先被采纳"的注释描述的是**写盘**时的
        意图，读盘这一侧照抄它就反了。）

        现在：**显式候选顺序** + **路径版本必须匹配当前布局**。
        带 `path_version` 且与当前不符的结果是**另一条布局**的结论，直接跳过；
        不带该字段的老产物按"无法判定版本"放行（保持既有行为，不制造新的 fail）。
        """
        _cur_pv = (self._read_path() or {}).get("version")
        stage = getattr(self, "_stage_base", self.stage_name)
        # [审计 #63] 原来这里有个 `seen` 列表：只被 append、从不被读，删掉。
        for pat in self._STAGE_RESULT_CANDIDATES:
            f = os.path.join(self.path_checkpoint_dir, pat.format(stage=stage))
            d = self._json(f)
            # [2026-09-15] `converged` 已从 stage 结果里删除 ⟹ 嗅探键换成
            # `analysis_status`。仍然认 `total_delta_G`，**不是**为了兼容旧语义，
            # 而是为了让老产物**能被读到**、随后在 `read()` 里 fail-closed 判成
            # `ANALYSIS_INCOMPLETE`（读不到 = 不得当成通过）。
            if not (isinstance(d, dict)
                    and ("analysis_status" in d or "total_delta_G" in d)):
                continue
            _pv = d.get("path_version")
            if _pv is not None and _cur_pv is not None and int(_pv) != int(_cur_pv):
                continue          # 另一条布局的结论，不是"旧一点"而是**另一个量**
            return d
        # 兜底：候选名都没命中（改过命名 / 非标准 stage 名）时回到全扫，
        # 但仍然按**修改时间**取最新的那份，而不是按文件名。
        for f in sorted(
            glob.glob(os.path.join(self.path_checkpoint_dir, "stage2_*.json")),
            key=lambda x: os.path.getmtime(x), reverse=True,
        ):
            d = self._json(f)
            if isinstance(d, dict) and ("analysis_status" in d
                                        or "total_delta_G" in d):
                _pv = d.get("path_version")
                if _pv is not None and _cur_pv is not None and int(_pv) != int(_cur_pv):
                    continue
                return d
        return None

    def _read_window(self, idx: int, stage_dir: Optional[str] = None,
                     checkpoint_dir: Optional[str] = None) -> Dict[str, Any]:
        # `stage_dir` / `checkpoint_dir` 可覆盖：immutable rewindow 的子窗住在
        # 自己的目录里（局部下标从 0 起，与父窗索引**同名不同义**），读法一样。
        stage_dir = stage_dir or self.stage_dir
        checkpoint_dir = checkpoint_dir or self.checkpoint_dir
        conv = self._json(os.path.join(
            stage_dir, f"dual_window_{idx}_{self.stage_type}_convergence.json"))
        fail = self._json(os.path.join(
            stage_dir, f"dual_window_{idx}_{self.stage_type}_warmup_failure.json"))
        state = self._json(os.path.join(
            checkpoint_dir, f"ibs_state_{self.stage_type}_window_{idx}.json"))
        lam = (conv or {}).get("lambdas_vdw") or (state or {}).get("lambdas_vdw") or []
        # 🔑🔑 [审计 #56] **`warm` 本身也必须取"消耗更多"的那一份，不是优先取
        # convergence。** 上一次修复（见下面 `spent` 的长注释）只落在 `spent` /
        # `_engine_left` 两个量上，`warm = conv.bias_warmup or fail...` 一字未动 ⟹
        # 同一条窗口记录里新旧两种口径混用：`spent` 来自 warmup_failure（新），
        # 而 gate 读数 / `validation_indeterminate` / `best_effort` 来自 convergence（旧）。
        # 口径只能有一套：`warm` 与 `ledger` / `spent` / `cap` 全部取同一份。
        _conv_warm = (conv or {}).get("bias_warmup") or {}
        _fail_warm = (fail or {}).get("bias_warmup") or fail or {}
        _conv_ledger = _conv_warm.get("warmup_budget_ledger") or {}
        _fail_ledger = _fail_warm.get("warmup_budget_ledger") or {}

        def _spent_of(_led):
            if not _led:
                return None
            return sum(int(_led.get(k, 0) or 0) for k in
                       ("learning_steps", "freeze_burn_in_steps",
                        "frozen_validation_steps"))

        def _cap_of(_led):
            # 🔑 [审计 #36] **`cumulative_cap_steps == 0` 是真实的「上限 0」，不是未知。**
            # 先前 `cap = (...) or (... or None)` 用真值判，而 `new_warmup_budget_ledger`
            # 的默认值恰恰就是 0 ⟹ 真上限 0 被读成未知 ⟹ `warmup_steps_left=None` ⟹
            # `all_windows_budget_exhausted` 恒假（#15 的反方向）。只能用 `is not None`。
            _c = (_led or {}).get("cumulative_cap_steps")
            return int(_c) if _c is not None else None

        _conv_spent, _fail_spent = _spent_of(_conv_ledger), _spent_of(_fail_ledger)
        # 🔑🔑 [审计 #35] **`cap` 与 `spent` 必须取同一份 ledger。**
        # 先前 `spent` 取两份里**较新**（消耗更多）那份、`cap` 却优先取 convergence
        # 那份**较旧**的 ⟹ 用户抬高 `max_bias_warmup_steps` 之后，新 cap 在
        # warmup_failure 里、旧 spent 已经追平旧 cap ⟹ `cap − spent ≡ 0`，
        # 而且这个假零还会把引擎的权威 `warmup_budget_remaining_steps` 挡掉
        # （旧逻辑要求 engine_left ≤ cap−spent 才采信）⟹ **显式升档永不生效**。
        # 现在成对取：消耗更多的那一份的 cap 与 spent 一起用。
        _use_fail_ledger = (_fail_spent if _fail_spent is not None else -1) > \
                           (_conv_spent if _conv_spent is not None else -1)
        warm = (_fail_warm if _use_fail_ledger else _conv_warm) \
            or _conv_warm or _fail_warm or {}
        ledger = _fail_ledger if _use_fail_ledger else _conv_ledger
        spent = _fail_spent if _use_fail_ledger else _conv_spent
        cap = _cap_of(ledger)
        _cap_conv, _cap_fail = _cap_of(_conv_ledger), _cap_of(_fail_ledger)
        # 两份 cap **都在场且不一致** = 有人改过上限（或跨段继承了）⟹ 只有引擎
        # 知道真实剩余，无条件采信它；一致时才谈"取更保守的那个"。
        _caps_disagree = (_cap_conv is not None and _cap_fail is not None
                          and _cap_conv != _cap_fail)
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
            stage_dir, f"dual_window_{idx}_{self.stage_type}_self_support.json"
        )) or {}
        bias_status = (state or {}).get("bias_status")
        evid = (state or {}).get("f_k_evidence_status")
        # 🔑🔑 [2026-09-14 真机根因] **「还剩多少预热预算」有两份记录，必须取权威那份。**
        #
        # `convergence.json` 的 ledger 是**生产跑完那一刻**写的；而 warmup 之后还会
        # 继续消耗（续跑再进一次预热、跨段继承），最新值写在
        # `*_warmup_failure.json` 里，引擎自己还算好了 `warmup_budget_remaining_steps`。
        # 先前 `warm = conv.bias_warmup or fail...` **优先取 convergence** ⟹ 控制器
        # 读到的是过期的那份。
        #
        # 实测 rep2 win4：convergence 说 spent=815000 / left=140000，
        # warmup_failure 说 spent=955000 / **remaining=0**。于是控制器以为"预算有余"
        # 一直发补帧，而引擎每次都以「累计预算已用尽，本次可用 0 步」把它弹回来 ——
        # 连发 40 轮、盘面一字节没变。
        #
        # 口径：**引擎自己算的 `warmup_budget_remaining_steps` 是权威**；拿不到才
        # 退回逐项相加。`warm` / `ledger` / `spent` / `cap` 的成对选取见上面
        # `_use_fail_ledger` 那一段（审计 #35/#36/#56）。
        # 引擎的权威剩余量（它才知道跨段继承了多少）。
        _engine_left = _fail_warm.get("warmup_budget_remaining_steps")
        if _engine_left is None:
            _engine_left = (fail or {}).get("warmup_budget_remaining_steps")
        _ledger_left = (max(0, int(cap) - int(spent))
                        if (cap is not None and spent is not None) else None)
        # 🔑 [审计 #43] **来源标签必须如实反映实际采用的那个数。**
        # 先前只判 `_engine_left is not None` 就写 `"engine:..."`，而引擎值可能被
        # 上面的三元式否决、实际用的是 ledger —— 一个会撒谎的标签比没有标签更坏
        # （`tests/test_ctl10_warmup_budget_authority.py` 钉的正是这个字符串）。
        if _engine_left is not None and (_caps_disagree or _ledger_left is None):
            _left_val, _left_src = int(_engine_left), "engine:warmup_budget_remaining_steps"
        elif _engine_left is not None and int(_engine_left) <= _ledger_left:
            _left_val, _left_src = int(_engine_left), "engine:warmup_budget_remaining_steps"
        elif _ledger_left is not None:
            _left_val, _left_src = int(_ledger_left), "ledger:cap-spent"
        else:
            _left_val, _left_src = None, "unknown"
        prod = (conv or {}).get("cumulative_production_steps")
        if prod is None:
            prod = (conv or {}).get("actual_production_steps")
        return {
            "window_idx": idx,
            "has_convergence": conv is not None,
            "has_warmup_failure": fail is not None,
            "n_states": len(lam) or None,
            # 这份证据**是在哪套 λ 上产出的**。布局一变它就不再描述这个窗口 ——
            # 用它做身份，比拿段号/mtime 猜新旧可靠。
            "lambdas_vdw": [float(x) for x in lam] or None,
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
            # [2026-09-16] 门槛跟着读数一起进视图：判"加帧还能不能推过门"必须用
            # **写侧这一次实际生效**的档位（`window_self_support_check` v2 起落盘）。
            # 老产物没有这个键 ⟹ None ⟹ `support_failure_is_skew` 保持既有判法。
            "self_n_eff_over_g_eligible": selfchk.get(
                "n_eff_over_g_eligible_threshold"),
            # 生产侧累计 f_k 偏差（scope=production），由 solve_stage_integrated 落在
            # stage 结果的 `cumulative_fk_residual_production` 里。
            "cum_fk_span": None, "cum_fk_verdict": None,  # 在 read() 里按窗口填
            "worst_state_by_n_eff": selfchk.get("worst_state_by_n_eff"),
            # 🔑🔑 [2026-09-17，用户拍板 #3] **瓶颈态的 g / η / ratio 观测量。**
            # 目的只有一个：把 `n_eff_over_g` 的**可达性**从"假定 η、g 恒定"的
            # 乐观上界，换成有数据支撑的判断。**本轮只记录，不改任何门**。
            #
            # 三个量必须来自**同一个 k\*、同一个 frame set**，否则记了也没法用：
            #   · `k* = worst_state_by_n_eff_over_g`（不是 `worst_state_by_n_eff`，
            #     那是另一个量 —— 分子最小 ≠ 比值最小）；
            #   · `g[k*]`  取 `statistical_inefficiency_per_lambda[k*]`；
            #   · `η[k*] = n_eff_per_state[k*] / n_eff_input_n_frames_per_state[k*]`，
            #     分母是**逐态、finite mask 之后、未按 sub_idx 去相关稀疏**的帧数。
            #     ⚠️ 绝不能用 `n_frames_decorrelated`（自检侧）或
            #     `solver_n_frames_decorrelated`（求解器侧）—— 那是 solver eligibility
            #     的量，与 η 差好几倍；也不能用 `n_eff_frame_set_n_frames_raw`，
            #     那是 finite mask **之前**的数。
            # 只认 `window_self_support_check` 这一份；solver g / warmup g /
            # validation g 一概不得代替。读不出来一律 `None`（= UNKNOWN），
            # **不填 0、不当作可达** —— 老产物没有这些键，必须如实说"不知道"。
            **_bottleneck_observation(selfchk),
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
            # 🔑 [审计 #1 的读侧] **语义变了，键名跟着变。**
            # 原来落盘的 `n_eff_marginal_by_block` 是前缀差
            # `ESS(prefix_b) − ESS(prefix_{b−1})` —— ESS 是非可加的比值泛函，
            # 做差不是"边际量"（加一帧高权重帧 ESS 反而塌向 1，被误判 CONFIRMED_DERAILMENT）。
            # 权威判据量现在是**块内独立 ESS** `block_local_ess_by_block`；
            # 前缀差改名 `prefix_ess_by_block` 只作诊断。
            # 视图键跟着改名，否则 `n_eff_marginal_by_block` 这个名字会长期骗人。
            "block_local_ess_by_block": (
                selfchk.get("block_local_ess_by_block")
                or selfchk.get("n_eff_marginal_by_block")   # 旧产物回退
            ),
            "prefix_ess_by_block_REPORT_ONLY": selfchk.get("prefix_ess_by_block"),
            "warmup_steps_spent": spent,
            "warmup_steps_cap": cap,
            # 🔑 [2026-09-14 真机] **账本必须原样带出来。**
            # `relearn_epoch_required_steps()` 读 `window_record["warmup_budget_ledger"]`
            # 来自校准"开一个新 Epoch 要多少步"，而这个键此前**根本没被放进窗口记录**
            # ⟹ 它永远走 80000+10000+50000 的兜底常量。真机 cyclod_ligand1/rep2 win3
            # 的真实账本是 learning=230000 / burn_in=10000 ⟹ 真实需求 290000，
            # 被低估了一半还多。方向是**危险**的那一侧：会批准一个跑不完的 Epoch，
            # 把预算烧光再半路死掉 —— 正是那段注释声称要防止的事。
            # 取消耗更多的那一份（与上面 `spent` 同口径，保守）。
            # 与上面 `spent` / `cap` 同一份 ledger（审计 #35：不许两个量取不同来源）。
            "warmup_budget_ledger": ledger,
            "warmup_steps_left": _left_val,
            "warmup_steps_left_source": _left_src,
            # [2026-09-12] 验证**可达性**预检的原料。同一候选、连续数据上的
            # 多个检查点（`insufficient_attempts` 每次都记一个 g），用来算保守
            # 下界 g_L；再配上 T / Ncap / 剩余预算就能判"在算术上还可不可能"。
            "validation_indeterminate": warm.get("validation_indeterminate"),
            # 🔑🔑 [审计 #55] **嵌套层级读错了。** 引擎把 `validation_g_history`
            # 写在 `bias_warmup["validation_indeterminate"]` **内部**
            # （`ibs_engine.py` 的 `validation_indeterminate_diag`），控制器却在
            # `bias_warmup` **顶层**读 ⟹ 恒为空 ⟹ 每次都退回那个单点兜底，
            # 而引擎自己的注释明写「可达性预检要的是多个检查点，不是单点：
            # 单点在小样本下会误杀」。可达性预检因此一直在用它被警告过的那个口径。
            # 正确层级优先，顶层留作兜底（万一以后被提上去）。
            "validation_g_checkpoints": [
                float(x) for x in (
                    (warm.get("validation_indeterminate") or {}).get(
                        "validation_g_history")
                    or warm.get("validation_g_history")
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

    def _make_production_budget(self, *, cap, cap_source, per_window_used,
                                per_unit_used, unknown_usage, note):
        """生产预算账的**唯一**构造器（审计 #39）。

        单段视图和合并视图先前各写一遍，差别不只是写法：单段那份用 `or 0`、没有
        `usage_complete` / `unknown_usage`、还无条件算 `stage_remaining_steps` 和
        `exhausted` ⟹ 同一个 stage 换个视图读，"还剩多少 / 有没有耗尽"会得到两个
        答案，而 `plan()` 的预算闸就靠这两个字段。

        三条规矩（都是"未知不是零"的实例）：
          · **上限未知 ⟹ 余额未知**，「预算耗尽」这个结论不成立，绝不因此终止；
          · **用量不完整 ⟹ 余额未知**，不许拿一个少算过的数当余量去准入；
          · `exhausted` 只有在「上限已知 **且** 用量完整 **且** 确实超了」时才为 True。
        """
        _unknown = list(unknown_usage or [])
        _used = (sum(int(v) for v in (per_window_used or {}).values() if v is not None)
                 + sum(int(v) for v in (per_unit_used or {}).values() if v is not None))
        _cap_known = cap is not None
        return {
            "stage_cap_steps": cap,
            "cap_source": cap_source,
            "cap_known": _cap_known,
            "stage_used_steps": int(_used),
            "usage_complete": not _unknown,
            "unknown_usage": _unknown,
            "stage_remaining_steps": (
                None if (not _cap_known or _unknown) else int(cap) - int(_used)
            ),
            "per_window_used_steps": dict(per_window_used or {}),
            "per_sampling_unit_used_steps": dict(per_unit_used or {}),
            "new_ensemble_reserve_steps": int(self.new_ensemble_reserve_steps),
            "production_block_steps": int(self.production_block_steps),
            "cap_scope": "production_steps_only",
            "exhausted": bool(_cap_known and not _unknown and _used >= int(cap)),
            "note": note,
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
        # [2026-09-14] 有界 rewindow 的台账：哪一个父窗口、什么身份、采了几块。
        # 进停滞签名（新建子系综 = 盘上状态确实变了），并让 `decide()` 知道
        # 这个父窗口已经建过子系综 —— 别重复建第二个。
        _rw_ledger = self._json(os.path.join(
            self.path_checkpoint_dir, "stage2_rewindow_ledger.json")) or {}
        # 「某个动作在当前盘面上是 no-op」的记账（执行器写，控制器读）。
        noop_actions = self._json(os.path.join(
            self.path_checkpoint_dir, "stage2_noop_actions.json")) or {}
        # `_rw_latest` 的赋值挪到 `_rw_entry_is_current()` 定义之后（审计 #42），
        # 因为选"最新"必须先能判"是不是当前布局的"。
        # 🔑🔑 [2026-09-14 裁决 1] **子窗是控制器可调度的采样单元。**
        # `windows` 继续表示**物理窗口**；子窗另立一张表 `sampling_units`：
        # 它有自己的目录、自己的 checkpoint、自己的冻结 f_k，局部下标从 0 起
        # （与父窗索引**同名不同义**，所以绝不能靠放宽 `_segment_stage_names()`
        # 把它并进段发现），`solver_index` 只是求解器的命名空间偏移。
        # 🔑🔑 [CTL-05，2026-09-14] **ledger 必须按当前 `path_version` 隔离。**
        # 路径演化（插 λ / 拆末窗）之后，旧布局下建的子系综描述的是**另一套 λ 区间**；
        # 拿它们参与当前求解会混进不可比的帧，而旧的 `parents_done` 还会把新的修补
        # 挡在门外（"这个父窗已经建过子系综了"——那是上一条布局的事）。
        # 不带 `path_version` 的老条目按"无法判定版本"放行，不制造新的 fail。
        # ⚠️ 不能写 `path` —— 它在本函数里要到后面才赋值（用了就是 UnboundLocalError，
        # 而且包在 try 里会被静默吞掉）。直接读。
        _cur_pv_for_rw = (self._read_path() or {}).get("version")

        def _rw_entry_is_current(_e):
            _pv = (_e or {}).get("path_version")
            if _pv is None or _cur_pv_for_rw is None:
                return True
            return int(_pv) == int(_cur_pv_for_rw)

        # 🔑🔑 [审计 #42] **"最新的那条 rewindow 记录"不能按 identity 字典序取。**
        # 先前 `for _k in sorted(_rw_ledger): _rw_latest = ...` —— identity 是内容
        # 哈希，字典序与时间毫无关系；而且**不过 `_rw_entry_is_current()`** ⟹
        # `immutable_rewindow` 报出来的 identity / child_ranges 可能描述**另一条
        # 布局**下的子系综。后果不止是显示错：`identity` 进停滞签名，会把一次真实
        # 推进记成 no-op（签名没变 ⟹ 判"推不动" ⟹ 提前放弃）。
        # 正解：只在当前 `path_version` 的条目里选，按条目自带的时间戳/序号取最新，
        # 都没有就按插入序取最后一个 current 的（dict 保插入序）。
        _rw_current = [(_k, _v) for _k, _v in _rw_ledger.items()
                       if isinstance(_v, dict) and _rw_entry_is_current(_v)]

        def _rw_recency(_kv):
            # `solver_index_base` 是执行器建窗时按**台账已有条目数**递增钉死的
            # （`10000 + 100 * n`），一旦写入不再变 ⟹ 它就是这份台账里唯一真实的
            # 序号。没有它的老条目退回插入序。
            _b = (_kv[1] or {}).get("solver_index_base")
            try:
                return (1, float(_b)) if _b is not None else (0, 0.0)
            except (TypeError, ValueError):
                return (0, 0.0)

        _rw_latest = None
        if _rw_current:
            _stamped = [kv for kv in _rw_current if _rw_recency(kv)[0]]
            _rw_latest = (max(_stamped, key=_rw_recency)[1] if _stamped
                          else _rw_current[-1][1])

        sampling_units = []
        for _rid in sorted(_rw_ledger):
            _e = _rw_ledger[_rid]
            if not isinstance(_e, dict) or not _rw_entry_is_current(_e):
                continue
            # 🔑🔑 [审计 #7，2026-09-14] **`status` 必须参与过滤。**
            # 先前只按 `path_version` 过滤。而求解侧
            # （`_solve_with_rewindow_children`）**只采信 `SAMPLED`**，
            # `_topup_rewindow_child` 又从不把 status 推回 `SAMPLED` ⟹
            # 一条 `ABANDONED_NO_PRODUCT`（或卡住的 `SAMPLING_INTENT`）条目会每轮
            # 被判成「缺帧 ⟹ 补一块」，烧到 40 轮上限，而**一帧都进不了 ΔG**；
            # 停滞保护还看不见它（签名里它的 `production_steps` 每轮都在变）。
            # 兜底：老条目缺 `status` 按 `SAMPLED`（与
            # `abfe_pipeline._rewindow_entry_status` 同一条既有先例，别删 ——
            # 删了会把所有既有 run 的子系综一次性判废）。
            _e_status = str(_e.get("status") or "SAMPLED")
            _schedulable = (_e_status == "SAMPLED")
            _od, _cd = _e.get("output_dir"), _e.get("checkpoint_dir")
            for _li, _rng in enumerate(_e.get("child_ranges") or []):
                _w = self._read_window(_li, stage_dir=_od, checkpoint_dir=_cd) \
                    if (_od and _cd) else {}
                sampling_units.append({
                    "unit_id": f"rw:{_e.get('identity')}:{_li}",
                    "kind": "rewindow_child",
                    # 台账状态原样带出来供审计；`schedulable=False` 的单元不产生
                    # 任何动作（见下面 `needs_frames` / `complete` 的赋值）。
                    "status": _e_status,
                    "schedulable": bool(_schedulable),
                    # ⚠️ [契约 B] 子窗**没有 `segment`**。「换段 = 换 f_k」这一维在
                    # 子窗这边由 `identity` 承担（每个子系综自己锁一份 f_k），
                    # 块账的可比性与 no-op 指纹都用它，别再造一个 `segment` 出来。
                    "parent_window": _e.get("parent_window"),
                    "local_index": int(_li),
                    "range": [int(x) for x in _rng],
                    # 续跑这个子窗要用整套子布局（执行器传 window_ranges）
                    "all_child_ranges": [[int(a), int(b)]
                                         for a, b in (_e.get("child_ranges") or [])],
                    "identity": _e.get("identity"),
                    "parent_range": [int(x) for x in (_e.get("parent_range") or [])],
                    "output_dir": _od,
                    "checkpoint_dir": _cd,
                    # 求解器命名空间，**不是**调度身份。
                    # 🔑 基数从台账读（建窗时钉死），**不再**所有 identity 共用
                    # `10000/10001` —— 那会让两组子窗串号、solver 的跳窗记录被
                    # 误分配给另一组。
                    "solver_index": int(_e.get("solver_index_base") or 10_000) + int(_li),
                    "production_steps": _w.get("production_steps"),
                    "self_verdict": _w.get("self_verdict"),
                    # 归因来源要跟着单元走：下面那个循环里 `_w` 是**上一轮遗留**的
                    # 变量（永远指向最后一个子窗），从它读等于全体子窗共用一份归因。
                    "self_verdict_source": _w.get("self_verdict_source"),
                    "self_sufficient": _w.get("self_sufficient"),
                    "self_n_frames_decorrelated": _w.get("self_n_frames_decorrelated"),
                    "min_n_eff_over_g": _w.get("min_n_eff_over_g"),
                    # 🔑🔑 [2026-09-17] **下面这 6 个键先前没被抄进来，而有 3 个消费者在读。**
                    # `sampling_units` 是 `_read_window()` 记录的**手抄子集**，抄漏了不报错，
                    # 行为静默退化成「那个条件永远不成立」—— 与
                    # `tests/test_no_phantom_view_keys.py` 开头列的 5 次是同一形状，
                    # 只是发生在数据面的**第二层**（unit 记录），而那个测试只盯 view 顶层。
                    #   · `self_min_frames`            → `support_failure_is_skew(min_frames=None)`
                    #     ⟹ 「n_decorrelated < min_frames」这条**样本量硬证据**对子窗恒不成立
                    #     ⟹ 子窗的支撑失败一律被归成偏斜/UNKNOWN，走 D3 停机而不是补帧；
                    #   · `self_n_eff_over_g_eligible` → `_frames_admission` 第三道闸的门槛恒 None
                    #     ⟹ `n_eff_over_g_reachable_by_frames` 返回 None ⟹ **该闸对子窗恒放行**；
                    #   · `bottleneck_*` 四个         → D3 终态诊断里恒 None，而这四个正是
                    #     2026-09-17 拍板 #3 专门加来把可达性从「假定 η、g 恒定的乐观上界」
                    #     换成有数据支撑的判断的量。
                    "self_min_frames": _w.get("self_min_frames"),
                    "self_n_eff_over_g_eligible": _w.get("self_n_eff_over_g_eligible"),
                    "bottleneck_state": _w.get("bottleneck_state"),
                    "bottleneck_g": _w.get("bottleneck_g"),
                    "bottleneck_eta": _w.get("bottleneck_eta"),
                    "bottleneck_ratio": _w.get("bottleneck_ratio"),
                    "warmup_steps_left": _w.get("warmup_steps_left"),
                    "phase": _w.get("phase"),
                    "has_convergence": bool(_w.get("has_convergence")),
                    # 求解器跳窗按 solver_index 映回来（见下）
                    "solver_skip": None,
                })
        # 把 stage 结果的 solver 读数按 solver_index 映射回子窗。
        _sr = self._read_stage_result() or {}
        _sk_by_solver = {
            int(x.get("window_index", -1)): x
            for x in (_sr.get("skipped_windows") or [])
        }
        _ov_by_solver = {
            int(x.get("window_index", -1)): x
            for x in (_sr.get("window_overlap_diagnostics") or [])
        }
        # 🔑🔑 [2026-09-14] **完成 = 过最终的 solver 门，不是"过了入场下限"。**
        # 先前只看"自检通过 + 没被跳窗"，而跳窗门是**入场**下限（10 帧）；
        # 求解器一旦让它进了协方差链，子窗就从待补列表消失 —— 可最终门要的是
        # `min_decorrelated_samples`（20）。于是 11/20 的子窗被判"已完成"，
        # 而最终门报 `worst_window = solver 索引`，9c 发的是**不带 `unit_id`** 的
        # 普通补帧，执行器把一个越界的窗口号交给原窗口路径。
        _final_floor = _sr.get("min_decorrelated_samples_threshold")
        for _u in sampling_units:
            _si = int(_u["solver_index"])
            _u["solver_skip"] = _sk_by_solver.get(_si)
            _ov = _ov_by_solver.get(_si) or {}
            _u["solver_n_frames_decorrelated"] = _ov.get("n_frames_decorrelated")
            _u["solver_min_decorrelated_samples_threshold"] = _final_floor
            _nd = _ov.get("n_frames_decorrelated")
            _meets_final = (
                None if (_final_floor is None or _nd is None)
                else int(_nd) >= int(_final_floor)
            )
            _u["meets_final_solver_gate"] = _meets_final
            # 🔑🔑 [CTL-02，2026-09-14] **调度状态 ≠ 失败归因。**
            #
            # 先前只有一个 `complete` 布尔，而它要求 `self_sufficient is True` ——
            # 那个量**混合**了帧数、`N_eff/g`、top1% 三项。分支 1c 对所有
            # `complete=False` 一律发 `RUN_PRODUCTION` ⟹ **偏斜类**失败
            # （top1% / raw ESS）的子窗被**反复加帧**，而加帧治不了偏斜
            # （§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 0.762、ESS 比值反而更差）。
            #
            # 现在拆成两个正交的量：
            #   · `needs_frames`  —— **样本量**不够（还没测够）⟹ 动作是加帧；
            #   · `support_failed` —— **支撑/偏斜**不合格 ⟹ 加帧治不了，要缩跨度；
            #     没有有界的缩跨度动作时，如实停在 NO_FEASIBLE_ACTION，
            #     **不许拿加帧顶替**。
            # `complete` 保留，但只表示"两者都没问题"，不再被当成动作依据。
            _self_src = str(_u.get("self_verdict_source") or "")
            # ⚠️ `verdict_source` 只说明"这个结论由哪个判据得出"，**通过**的窗口
            # 同样带 `min_n_eff_over_g` —— 只看来源会把健康窗口也标成支撑失败。
            # 必须先确认 verdict 本身是**失败**。
            # 归因口径（TODO「四个量不许混用」）：
            #   · `top1pct_veto`       = 权重塌缩，**否决警报** ⟹ 加帧治不了
            #   · `HARD_INSUFFICIENT`  = 支撑低到测不出来   ⟹ 同上
            #   · `solver_eligibility` = 纯帧数不够 ⟹ 按样本量处理，加帧**正是**对症
            # [审计 #46] 归因判定收敛到 `support_failure_is_skew()` 这一份实现 ——
            # `solver_eligibility`（纯帧数不够，写侧被强制置成 `HARD_INSUFFICIENT`）
            # 原来在这里被当成"加帧治不了"，与写侧「帧数不够 ≠ 支撑不够」直接冲突。
            # 🔑 [2026-09-15] `min_n_eff_over_g` 从"样本量"改判到**偏斜**（真机
            # win1：n_decorr=888、比值 8.68 —— 帧一点不缺，是权重压不到目标态上）。
            # 见 `_SAMPLE_SIZE_VERDICT_SOURCES` 那段。帧数一并传进去自验。
            # 🔑🔑 [2026-09-17 P0] **子窗直接存三态，不许再压成布尔。**
            # 先前 `_support_failed = support_failure_is_skew(...)` 把三态压回两态，
            # 于是 `UNKNOWN` 被读成"不是 structural" ⟹ 走进下面的 `needs_frames`
            # ⟹ 从「错误授权 D3 停机」换成「错误授权补帧」。同一个语义丢两次。
            # ⚠️ 而且原调用**只传了 n_decorrelated/min_frames**，没传 ratio/target/
            # headroom ⟹ 射程恒判不了 ⟹ 恒 `None`。三个参数必须真正接进来，
            # 否则三态里只会出现两态。
            # ⚠️ 射程要块账，而块账在本循环**之后**才算出来 ⟹ 这里先按"没有射程"
            # 归因（⟹ `min_n_eff_over_g` 那支恒 UNKNOWN），**块账就位后由下面的
            # 第二遍重算**。别在这里省掉第二遍：省了就等于永远判不出射程。
            _attr = support_failure_attribution(
                _u.get("self_verdict"), _self_src,
                n_decorrelated=_u.get("self_n_frames_decorrelated"),
                min_frames=_u.get("self_min_frames"))
            _support_failed = (_attr == SUPPORT_FAILURE_STRUCTURAL)
            # 🔑 [2026-09-17] **三条「独立于归因」的补帧理由**，单独存 ——
            # 它们说的是"证据本身还没产出来"，跟这次失败是样本量还是结构性无关。
            # 第二遍重算归因时**不许**把它们一起 AND 掉（实测那样会让一个还没跑完
            # 的子窗拿不到帧，`test_an_incomplete_child_gets_a_run_production…` 变红）。
            _needs_frames_independent = bool(
                not _u["has_convergence"]
                or _u.get("solver_skip") is not None
                # 读不到最终门的读数时**不算过**（缺证据 ≠ 通过）⟹ 先补帧拿证据。
                or _meets_final is not True
            )
            _u["needs_frames_independent"] = _needs_frames_independent
            _needs_frames = bool(
                _needs_frames_independent
                # 🔑 自检判不够 **且归因明确是样本量类** ⟹ 才补帧。
                # `UNKNOWN` 不授权补帧（先前 `not _support_failed` 把它放行了）。
                or (_u.get("self_sufficient") is False
                    and _attr == SUPPORT_FAILURE_SAMPLE_SIZE)
            )
            # [审计 #7] 不可调度的条目（ABANDONED_NO_PRODUCT / 卡住的
            # SAMPLING_INTENT）**一律不产生动作**：它的帧进不了求解，补给它的每一块
            # 都是纯烧 GPU。留在视图里只为审计。
            if not _u.get("schedulable", True):
                _needs_frames = False
                _support_failed = False
                _attr = None
            _u["needs_frames"] = _needs_frames
            _u["support_failed"] = bool(_support_failed)
            _u["support_failure_source"] = _self_src or None
            # 三态原样带到调度层。`UNKNOWN` 既不是"要补帧"也不是"结构性失败"，
            # 它是**归因判不出来** —— 不得授权 RUN_PRODUCTION / D3 停机 /
            # IMMUTABLE_REWINDOW / SPLIT_TAIL_WINDOW / INSERT_LAMBDA 任何一个。
            _u["support_attribution"] = _attr
            _u["attribution_unknown"] = bool(_attr == SUPPORT_FAILURE_UNKNOWN)
            # `complete` 只在**没有任何失败归因**时成立（`_attr is None` = 自检通过）。
            _u["complete"] = bool(not _needs_frames and _attr is None)

        # solver 索引 → unit_id 的反查表：最终门报的是 solver 索引，
        # 而调度身份是 `unit_id` —— 没有这张表就只能把越界的窗口号交给原窗口路径。
        solver_index_to_unit = {
            int(u["solver_index"]): str(u["unit_id"]) for u in sampling_units
        }

        immutable_rewindow = {
            "identity": (_rw_latest or {}).get("identity"),
            "parent_window": (_rw_latest or {}).get("parent_window"),
            "child_ranges": (_rw_latest or {}).get("child_ranges"),
            "n_blocks": len(((_rw_latest or {}).get("blocks") or [])),
            # 只有**当前布局**下建过的才算"已替代"；旧布局的记录不得挡住新修补。
            # 🔑🔑 [2026-09-17，用户拍板 D3] **`status` 必须参与过滤，否则会静默丢窗。**
            #
            # 这里先前只看 `path_version`。而 `parents_done` 有两个消费者，
            # 第二个的后果比第一个重得多：
            #   · `rewindow_feasible()` 拿它判「这个父窗已经切过一层」⟹ 不再切；
            #   · **CTL-03 拿它把父窗整个踢出 `earliest` 排序**（理由是"它已退出
            #     求解覆盖、子系综接管了那段 λ"）。
            # 于是一条 `ABANDONED_NO_PRODUCT`（子系综建了但一帧都没产出）或卡住的
            # `SAMPLING_INTENT` 记录，会让那个物理窗口**永久隐身**：当不上 earliest、
            # 退役轮不到它、也不能再建第二个 rewindow —— 而它那段 λ **根本没有人接管**。
            # 这是"静默丢窗"那一族（同 `_solve_with_rewindow_children` 只采信
            # `SAMPLED` 那条既有先例，见 `sampling_units` 里 `_schedulable` 的注释）。
            #
            # 判据与那一处**同一份**：只有 `SAMPLED` 才算"这个父窗真的被接管了"。
            # 兜底同样是「老条目缺 `status` 按 `SAMPLED`」—— 删了会把所有既有 run 的
            # 子系综一次性判废。
            "parents_done": sorted({
                int(v["parent_window"]) for v in _rw_ledger.values()
                if isinstance(v, dict) and v.get("parent_window") is not None
                and _rw_entry_is_current(v)
                and str(v.get("status") or "SAMPLED") == "SAMPLED"
                and len(v.get("child_ranges") or []) >= 2
            }),
            # 同一父窗在当前布局下的**总尝试次数**（不分 status）——
            # `rewindow_feasible` 拿它封"建→废→再建"的无限循环。
            "attempts_by_parent": {
                str(_pw): sum(
                    1 for v in _rw_ledger.values()
                    if isinstance(v, dict) and _rw_entry_is_current(v)
                    and v.get("parent_window") is not None
                    and int(v["parent_window"]) == int(_pw))
                for _pw in {
                    int(v["parent_window"]) for v in _rw_ledger.values()
                    if isinstance(v, dict) and v.get("parent_window") is not None
                    and _rw_entry_is_current(v)}
            },
            # 建过但**没被接管**的父窗（status 非 SAMPLED / 子窗不全）。
            # 它们**不**进 `parents_done` ⟹ 照常参与 earliest 排序、照常可以再修。
            # 单独列出来只为让人看得见「这里试过一次、没成」。
            "parents_abandoned": sorted({
                int(v["parent_window"]) for v in _rw_ledger.values()
                if isinstance(v, dict) and v.get("parent_window") is not None
                and _rw_entry_is_current(v)
                and (str(v.get("status") or "SAMPLED") != "SAMPLED"
                     or len(v.get("child_ranges") or []) < 2)
            }),
            "entries_from_other_path_versions": sorted({
                str(v.get("identity")) for v in _rw_ledger.values()
                if isinstance(v, dict) and not _rw_entry_is_current(v)
            }),
            # 子系综各自预热、各自锁一份 f_k ⟹ 与基准段**不是同一份**账，
            # 残差/支撑度证据分开记，不许混进基准段。
            "f_k_scope": (_rw_latest or {}).get("f_k_scope"),
        }
        path = self._read_path()
        expected = len(path["window_ranges"] or []) or None
        windows = [self._read_window(i) for i in sorted(found)]
        # 🔑🔑 [2026-09-17] **单段视图跑与聚合视图同一份布局判定。**
        # 先前这四个量只有 `read_aggregated()` 算，单段视图整套都没有 ⟹
        # `_coverage_incomplete()` 的 `out_of_range_windows` 与分支 0a /
        # `_evidence_status` 的 `stale_layout_evidence` 两道闸在单段上**恒不触发**
        # （后者更隐蔽：它按 `stale_layout_evidence_only` 标记算，而那个标记只有
        # 聚合视图的占位记录才会设 ⟹ 恒 `{}`）。
        # 补空默认值不行 —— 那只让 schema 表面一致、守卫照样失效，所以判定共用
        # `classify_layout_evidence()`。
        _cls_single = classify_layout_evidence([(self.stage_name, windows)], path)
        # ⚠️⚠️ **只出视图键，不动逐窗记录。** 我一度顺手给 STALE 的窗口打上
        # `stale_layout_evidence_only`，理由是"让 `_window_routing_state` 也看见"。
        # 那是**超范围**的改动，而且有真实爆炸半径：那个标记被
        # `first_untrusted_window()` 读，它决定 tail anchor ——
        # 实测四个窗口被判 stale 之后 anchor 从 win3 变成 win0，
        # `SPLIT_TAIL_WINDOW` 整条链的四个测试当场红。
        # 逐窗标记是**聚合视图的占位记录机制**（它把 STALE 记录换成一条清空证据的
        # 占位行，因为别的段可能有有效证据）；单段没有"别的段"、也没有占位这一步，
        # 就不该借用那个标记。
        # 这里要补的那道闸（stale ⟹ 不得判 DONE）本来就走**视图键**：
        # 分支 0a 与 `_evidence_status` 读 `view.get("stale_layout_evidence")`，
        # `stale_layout_windows()` 也同时读视图键和逐窗标记 ⟹ 只出视图键，
        # 1d-0 那条路由照样生效，而路由顺序一个字节不变。
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
            # 🔑🔑 [S2-A，2026-09-14] **算不出来 ≠ 没这回事，要显式记 `UNMEASURED`。**
            # 多段窗口的帧采自**两份不同偏置**，`_load_ibs_window_outputs_merged`
            # 因此显式 `base.pop("f_k")` ⟹ 求解器给的是
            # `{"error": "no_effective_f_k_for_these_frames"}`、没有 `verdict`。
            # 先前这里落成 `None`，而 `None` 在 5a-1 的语义是「证据还没产出 ⟹ 去跑
            # ANALYZE 把它算出来」—— 可这个窗口**结构上**就算不出来，于是又是一个
            # 无限 ANALYZE。`UNMEASURED` 才是真话，5a-2 已经按它分岔。
            # ⚠️ 有界 rewindow 的子系综**各自锁一份自己的 f_k**，所以它**不能**
            # 替父窗口把这道残差门宣称通过 —— 缺段证据仍然是 UNMEASURED。
            # 父窗口被有界 rewindow 的子系综取代之后，它**整个不在求解覆盖里**
            # （子系综的 window_index 带 10000 偏移），于是连一条残差记录都没有。
            # 那同样是 `UNMEASURED` 而不是 `None` —— 子系综各自锁自己的 f_k，
            # **不能**替父窗口把这道门宣称通过，也不该把控制器送去空跑 ANALYZE。
            _rewindowed = int(_w["window_idx"]) in set(
                immutable_rewindow.get("parents_done") or [])
            _w["cum_fk_verdict"] = (
                _c.get("verdict")
                or ("UNMEASURED" if (_c.get("error") or _rewindowed) else None)
            )
            # [S2-A 逐段门] 多段窗口给的是**逐段**结论，没有窗口级 span（那会是
            # 跨段望远镜相消）。逐段身份/帧范围/结论原样带上，供审计。
            _w["cum_fk_per_segment"] = _c.get("segments")
            _w["cum_fk_gate_is_per_segment"] = bool(_c.get("per_segment_gate"))
            _w["cum_fk_unmeasured_reason"] = (
                None if _c.get("verdict") else (
                    _c.get("error")
                    or ("replaced_by_immutable_rewindow_children_own_f_k"
                        if _rewindowed else None)
                )
            )
            _h = self._json(os.path.join(
                self.stage_dir,
                f"dual_window_{_w['window_idx']}_{self.stage_type}_heldout.json",
            )) or {}
            _w["heldout_verdict"] = _h.get("verdict")
            _w["heldout_worst_before"] = _h.get("worst_before")
            _w["heldout_worst_after"] = _h.get("worst_after")
        stage = self._read_stage_result()
        # 🔑🔑 [2026-09-15] **新契约的两维读法，fail-closed。**
        # `analysis_status` 只回答「分析跑完整了没有」（硬不变量），
        # `precision_status` 只回答「精度测过没有」（跨重复目标）。
        # 读不到 `analysis_status`（老产物、或写侧还没升级）⟹ **不得当成通过**，
        # 一律记 `ANALYSIS_INCOMPLETE` 并把理由写明。
        _stage_incomplete_reasons: List[str] = []
        if stage is None:
            _stage_analysis_status = "ANALYSIS_INCOMPLETE"
            _stage_incomplete_reasons = ["stage 结果尚未落盘"]
        else:
            _as = stage.get("analysis_status")
            if _as == "ANALYSIS_COMPLETE":
                _stage_analysis_status = "ANALYSIS_COMPLETE"
            elif _as == "ANALYSIS_INCOMPLETE":
                _stage_analysis_status = "ANALYSIS_INCOMPLETE"
                _stage_incomplete_reasons = [
                    str(x) for x in (stage.get("analysis_incomplete_reasons") or [])
                ] or ["写侧判了 ANALYSIS_INCOMPLETE 但没给理由"]
            else:
                _stage_analysis_status = "ANALYSIS_INCOMPLETE"
                _stage_incomplete_reasons = [
                    f"stage 结果里没有 `analysis_status`（读到 {_as!r}）⟹ "
                    "**缺键 fail-closed**，不得当成通过。"
                    "（`converged` 已于 2026-09-15 删除，不做兼容别名。）"
                ]
        # 精度同样 fail-closed：读不到就是 `UNMEASURED`（没测 ≠ 达标）。
        _stage_precision_status = str(
            ((stage or {}).get("precision_status") or "UNMEASURED")
        )
        if _stage_precision_status not in (
                "UNMEASURED", "MEETS_CROSS_REPEAT_TARGET",
                "EXCEEDS_CROSS_REPEAT_TARGET"):
            _stage_precision_status = "UNMEASURED"
        # 缺窗口：布局里有、产物里没有。这是"截断的 ΔG"这类失效的直接信号，
        # 现在只在日志里出现一次 WARN。stage 结果里的 skipped_windows 是另一种
        # （产物在、但去相关后帧数不足被踢出协方差链），两者都要算进来。
        missing = [i for i in range(expected) if i not in found] if expected else []
        # 🔑🔑 [审计 #58，2026-09-14] **`skipped_windows` 混了两个命名空间。**
        # 求解器报的 `window_index` 对物理窗口是下标、对 rewindow 子窗是
        # **solver 命名空间**的偏移索引（≥ `SOLVER_UNIT_INDEX_BASE`）。先前两者
        # 一起塞进 `skipped_windows`，后果：
        #   · 分支 0a 的 `DONE` 与 `_evidence_status` 的 `CONVERGED` 都查
        #     `view["skipped_windows"]` 非空 ⟹ 一个被跳的**子窗**会以一个**根本
        #     不存在的物理窗口号**（10000+）永久封死 DONE；
        #   · `render` 还把 `10000` 当窗口号打印给人看。
        # 拆成两个键：`skipped_windows` 只放物理窗口，`skipped_sampling_units`
        # 放翻成 `unit_id` 的子窗（翻不出来的保留 solver 索引，如实标出来）。
        _skip_all = {
            int(x.get("window_index", -1)): x
            for x in ((stage or {}).get("skipped_windows") or [])
        }
        _skip_recs = {i: r for i, r in _skip_all.items()
                      if i < SOLVER_UNIT_INDEX_BASE}
        skipped = sorted(_skip_recs)
        skipped_units = sorted(
            str(solver_index_to_unit.get(i) or f"solver_index:{i}")
            for i in _skip_all if i >= SOLVER_UNIT_INDEX_BASE
        )
        # 🔑🔑 [DECORR-01，2026-09-14 裁决] **求解器对「能否进入求解、是否跳窗」
        # 有操作权威。** 它实际用的去相关帧与 `skipped_windows` 决定这个窗口进不进
        # MBAR 与协方差链；自检的 `n_decorr` 只是**早期 target-support 诊断**，而且
        # `self_sufficient` 里还混着 `N_eff/g` 与 top1% 的条件，**不能**被读成
        # "求解器帧数够"。
        #
        # 实测（cyclod_ligand2/rep2）：win3 自检 56 帧、`sufficient=True`，求解器
        # 报 9 / 7 帧并**跳窗**；win0 自检 21 帧、g=24.5、`sufficient=False`，
        # 求解器报 9。两个数**门着不同的东西**，差 2–8 倍。
        #
        # ⚠️ **不让两边数字强行看齐**：去相关抽样依赖输入时间序列与所选 g，
        # 拿另一份输出的帧数反推它是错的（pymbar timeseries 的语义）。这里只做
        # 一件事 —— 把两侧**分别命名、分别展示**，并让 solver skip 优先。
        _ov_by_win = {
            int(x.get("window_index", -1)): x
            for x in ((stage or {}).get("window_overlap_diagnostics") or [])
        }
        _final_floor_w = (stage or {}).get("min_decorrelated_samples_threshold")
        for _w in windows:
            _sk = _skip_recs.get(int(_w["window_idx"]))
            _w["solver_skip"] = _sk            # None = 求解器没跳它
            # 🔑 [2026-09-14] **求解器侧的去相关帧数**（不是自检那份，两者差 2–8 倍）。
            # 这是"再给一块帧到底有没有用"的**判据量** —— 补帧准入看它的增长，
            # 而不是看"又跑了 25 万步"。
            _w["solver_n_frames_decorrelated"] = (
                (_ov_by_win.get(int(_w["window_idx"])) or {}).get("n_frames_decorrelated")
                if _sk is None
                else _sk.get("n_frames_after_decorrelation")
            )
            _w["solver_min_decorrelated_samples_threshold"] = _final_floor_w
            _w["evidence_decorrelation"] = {
                # —— 求解器侧：**操作权威**，决定进不进求解 ——
                "solver": None if _sk is None else {
                    "n_frames_after_decorrelation": _sk.get(
                        "n_frames_after_decorrelation"),
                    "min_frames_per_window": _sk.get("min_frames_per_window"),
                    "statistical_inefficiency": _sk.get("statistical_inefficiency"),
                    "statistical_inefficiency_per_lambda": _sk.get(
                        "statistical_inefficiency_per_lambda"),
                    "lambda_indices": _sk.get("lambda_indices"),
                    "reason": _sk.get("reason"),
                    # 段名在**这一层**才是确定的（合并视图里会被覆盖成胜出段）。
                    "segment": self.stage_name,
                    "observable": "u_kn_local(sampled_row)",
                },
                # —— 自检侧：早期诊断，**不是**求解器资格 ——
                "self_check": {
                    "n_frames_decorrelated": _w.get("self_n_frames_decorrelated"),
                    "min_frames": _w.get("self_min_frames"),
                    "verdict": _w.get("self_verdict"),
                    "verdict_source": _w.get("self_verdict_source"),
                    "sufficient_INCLUDES_N_eff_AND_top1pct": _w.get("self_sufficient"),
                    "min_n_eff_over_g": _w.get("min_n_eff_over_g"),
                    "observable": "window_self_support_check",
                },
                "authority": "solver_decides_eligibility_self_check_is_diagnostic",
                "do_not_reconcile": (
                    "两侧的去相关抽样依赖各自的输入帧集与所选 g；"
                    "不得用一侧的帧数反推另一侧，也不得为了让数字一致而改任一侧。"
                    "要诊断差异请用**完全相同的输入**另跑一次对照。"
                ),
                "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
            }
        # ---- 生产预算账（与预热账**分开**）----
        # 🔑🔑 [审计 #38/#39] **未知不是零，而且单段视图与合并视图必须同一份实现。**
        # 先前这里 `int(... or 0)`：读不到步数与"确实 0 步"压成同一个数，
        # `usage_complete` / `unknown_usage` 两个字段根本不存在 ⟹ 单段视图与合并
        # 视图对**同一个量**给两个答案。现在两处都走 `_make_production_budget()`。
        _pw_used: Dict[int, Any] = {}
        _unknown: List[Any] = []
        for w in windows:
            _ps = w.get("production_steps")
            if _ps is None:
                if w.get("has_convergence"):
                    _unknown.append({"window_idx": int(w["window_idx"]),
                                     "segment": self.stage_name})
                continue
            _pw_used[int(w["window_idx"])] = int(_ps)
        _unit_used: Dict[str, Any] = {}
        for u in sampling_units:
            _ps = u.get("production_steps")
            # 读不到就**登记为未知**（None），不写 0 —— 读侧的未知检测靠它才成立。
            _unit_used[str(u["unit_id"])] = None if _ps is None else int(_ps)
            if _ps is None and u.get("has_convergence"):
                _unknown.append({"unit_id": str(u["unit_id"])})
        _pv_now = (path or {}).get("version")
        _blocks_same_seg = self._production_blocks_ledger(
            windows, _pv_now, units=sampling_units)
        _blocks_all_seg = self._production_blocks_total_by_window(
            windows, _pv_now, units=sampling_units)
        # 🔑🔑 [2026-09-17 P0] **子窗归因第二遍：块账就位后把射程补上。**
        # 上面第一遍跑在块账之前，`frames_headroom` 拿不到 ⟹ `min_n_eff_over_g`
        # 那支恒 `None` ⟹ 恒 UNKNOWN。三态里只会出现两态。
        # 这里用**真实的子窗块账**重算一次；三个输入缺任何一个仍然是 UNKNOWN
        # （那是对的：判不了就是判不了，不许兜成 STRUCTURAL 去授权缩跨度）。
        _hr_view = {
            "max_production_blocks_per_window": int(self.max_blocks_per_window),
            "production_blocks_total_by_unit": (_blocks_all_seg or {}).get("by_unit") or {},
            "production_blocks_by_unit": (_blocks_same_seg or {}).get("by_unit") or {},
        }
        for _u in sampling_units:
            if not _u.get("schedulable", True):
                continue
            _attr2 = support_failure_attribution(
                _u.get("self_verdict"), _u.get("support_failure_source"),
                n_decorrelated=_u.get("self_n_frames_decorrelated"),
                min_frames=_u.get("self_min_frames"),
                min_n_eff_over_g=_u.get("min_n_eff_over_g"),
                n_eff_over_g_target=_u.get("self_n_eff_over_g_eligible"),
                frames_headroom=frames_growth_headroom(
                    _hr_view, None, unit_id=str(_u.get("unit_id"))))
            _u["support_attribution"] = _attr2
            _u["support_failed"] = bool(_attr2 == SUPPORT_FAILURE_STRUCTURAL)
            _u["attribution_unknown"] = bool(_attr2 == SUPPORT_FAILURE_UNKNOWN)
            # ⚠️ **整式重算**，不是在旧值上 AND 掩码：三条独立理由与归因是**或**的
            # 关系，掩码会把它们一起抹掉。
            _u["needs_frames"] = bool(
                _u.get("needs_frames_independent")
                or (_u.get("self_sufficient") is False
                    and _attr2 == SUPPORT_FAILURE_SAMPLE_SIZE))
            _u["complete"] = bool(not _u["needs_frames"] and _attr2 is None)

        production_budget = self._make_production_budget(
            cap=self.production_cap_steps,
            cap_source=self.production_cap_source,
            per_window_used=_pw_used,
            per_unit_used=_unit_used,
            unknown_usage=_unknown,
            note=(
                "单段视图：只算本段的生产步数（跨段累计见合并视图）。"
                "生产补帧花的是这本账；换 Epoch 的最低验证额度花的是 warmup 账"
                "（`warmup_steps_left`）。两者不可互相代替 —— 拿预热余额去终止一个"
                "只花生产帧的动作，正是 2026-09-14 修掉的那个缺陷。"
            ),
        )

        return {
            "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
            # 🔑🔑 [2026-09-17] **两个视图构造器的键集必须一致。**
            # 这两个键先前**只在 `read_aggregated()` 里产出**。生产走
            # `for_physical_stage` ⟹ 聚合视图 ⟹ 生产没事；但单段视图
            # （`Stage2RepairController(run, stage)` 直接构造，测试与离线回放都走它）
            # 缺了它们，而 `_decide_once` 照读不误：
            #   · `min_n_eff_over_g_history` 缺 ⟹ `_hist = {}` ⟹ **加帧刹车恒不触发**；
            #   · `stale_layout_evidence` 缺 ⟹ 分支 0a 与 `_evidence_status` 的
            #     「证据被布局作废」守卫 **恒放行** ⟹ 可以带着过期证据判 DONE。
            # 两条都是**静默**失效：不报错、不留痕，只是判据悄悄不生效。
            # 这里给出单段的**退化值**（数据本来就读了，不重写那份跨段逻辑）：
            #   · 历史 = 本段块账里逐块的 min N_eff/g（跨段合并是聚合视图的事）；
            #   · stale = 逐窗 `stale_layout_evidence_only` 标记（单段没有"别的段"
            #     可比，所以它就是全部）。
            # 由 `tests/test_no_phantom_view_keys.py` 的键集对账钉住不再分叉。
            "min_n_eff_over_g_history": {
                int(_i): [(r.get("segment"), r.get("min_n_eff_over_g"))
                          for r in (_rows or [])
                          if r.get("min_n_eff_over_g") is not None]
                for _i, _rows in ((_blocks_same_seg or {}).get("by_window") or {}).items()
            },
            "stale_layout_evidence": {
                int(k): list(v) for k, v in _cls_single["stale"].items()},
            # 下面四个键先前**只有聚合视图产出**，单段缺 ⟹ 读侧静默失效
            # （见 `classify_layout_evidence` 的长注释）。给的是单段的等价语义，
            # 不是空默认值。
            "unverifiable_layout_evidence": {
                int(k): list(v) for k, v in _cls_single["unverifiable"].items()},
            "out_of_range_windows": sorted(_cls_single["out_of_range"]),
            "layout_change_invalidated_segments":
                self.layout_change_invalidated_segments(),
            "window_provenance": {
                int(w["window_idx"]): self.stage_name for w in windows},
            # 单段就是"只有自己这一段"。执行器的 `_disk_signature` 读它，
            # 缺了停滞签名少一维。
            "aggregated_segments": [self.stage_name],
            "immutable_rewindow": immutable_rewindow,
            "sampling_units": sampling_units,
            "solver_index_to_unit": solver_index_to_unit,
            "noop_actions": noop_actions,
            # 🔑🔑 [2026-09-14 真机，同形状第 4 次] **这个键此前根本不存在。**
            # `action_noop_fingerprint()` 的两个调用方（控制器判读 / 执行器记账）
            # 都写 `view.get("path_version")` —— 而路径版本在 `view["path"]["version"]`。
            # 两侧一致地拿到 None ⟹ 指纹里的版本位恒为 0 ⟹ **布局一变就该失效的
            # no-op 记录，永远不会因布局变化而失效**，尽管它的 docstring 明写
            # 「记的是『在这个盘面上没用』，不是『这个动作永远没用』」。
            # 插 λ / 拆末窗恰恰是「让一个先前没用的动作重新变得有用」的操作。
            "path_version": (path or {}).get("version"),
            "production_blocks_by_window": _blocks_same_seg["by_window"],
            "production_blocks_by_unit": _blocks_same_seg["by_unit"],
            # 🔑 [BUD-03] 硬上限用的是**跨段累计**的那本（换段不重置配额）。
            "production_blocks_total_by_window": _blocks_all_seg["by_window"],
            "production_blocks_total_by_unit": _blocks_all_seg["by_unit"],
            "max_production_blocks_per_window": int(self.max_blocks_per_window),
            "production_budget": production_budget,
            "_production_budget_inputs": {
                "cap": self.production_cap_steps,
                "cap_source": self.production_cap_source,
                "reserve": int(self.new_ensemble_reserve_steps),
                "block": int(self.production_block_steps),
                # [审计 #38] **未知就写 None**，不许 `int(... or 0)` —— 读侧
                # （`read_aggregated` 的 `if _v is None`）就是靠它才不是死代码。
                "unit_used": dict(_unit_used),
            },
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
            # 🔑🔑 [BUD-05，2026-09-14] **未知就写 None，不许写 0。**
            # 先前 `int(... or 0)` 把「账本读不到」写成「余量 0」= 耗尽 —— 而同一个
            # 量在 `_epoch_validation_unaffordable` 里是 fail-closed、在 `decide()`
            # 分支 1e 里是 `left is not None and <=0`（未知 = 有钱）。同一个未知，
            # 三套语义。实测 cyclod_ligand1/rep3 win4 的 ledger 确实读不到，
            # 在这里被显示成 0。规矩（TODO §1）：**未知不是零，两个方向都不是。**
            "per_window_budget_remaining": {
                int(w["window_idx"]): (
                    None if w.get("warmup_steps_left") is None
                    else int(w["warmup_steps_left"])
                )
                for w in windows
            },
            # ⚠️ 这是**预热/验证**账，不是生产账。名字里没写"warmup"是历史遗留；
            # 换 Epoch 类动作查它，补生产帧**绝不**查它（见 production_budget）。
            # 🔑 [BUD-05] 「全部耗尽」是个**断言**，账不完整就不成立：
            # 只要有一个窗口的余量未知，就不得宣称全局耗尽（否则是拿少算过的账
            # 去论证一个终止性结论）。
            "all_windows_budget_exhausted": bool(
                windows
                and all(w.get("warmup_steps_left") is not None for w in windows)
                and all(int(w["warmup_steps_left"]) <= 0 for w in windows)
            ),
            "global_budget_source": "per-window warmup cap (NOT the production budget)",
            "path_insertions_left": max(
                0, self.max_path_insertions - int(path["events"].get("insert_lambda", 0))
            ),
            "path": path,
            "n_windows_found": len(windows),
            "n_windows_expected": expected,
            "missing_windows": missing,
            # [审计 #58] 只放**物理窗口**；被跳的子窗在下面那个键里（unit_id 命名空间）。
            "skipped_windows": skipped,
            "skipped_sampling_units": skipped_units,
            "production_rescue_targets": (stage or {}).get("production_rescue_targets") or {},
            # 🔑🔑 [2026-09-15 老板定案] `converged` 已删除，**不做同名改义、
            # 不做兼容别名**。新契约两维分开：
            #   · `analysis_status`  —— 只含**硬不变量**（求解跑完了没有、
            #     窗口丢没丢）。它 **≠ 可以判 DONE**。
            #   · `precision_status` —— 跨重复精度。单次 run 里恒为 `UNMEASURED`：
            #     一个 run 本来就无法自证精度。
            # **fail-closed**：读不到 `analysis_status` 的老产物一律当
            # `ANALYSIS_INCOMPLETE`，绝不当成通过。
            "stage_analysis_status": _stage_analysis_status,
            "stage_analysis_incomplete_reasons": _stage_incomplete_reasons,
            "stage_precision_status": _stage_precision_status,
            "stage_precision_evidence": (stage or {}).get("precision_evidence") or {},
            "stage_path_is_complete": (stage or {}).get("path_is_complete"),
            "stage_total_delta_G": (stage or {}).get("total_delta_G"),
            "has_stage_result": stage is not None,
            # 🔑🔑 [2026-09-14 自查] **这份 stage 结果的版本到底核实过没有。**
            # `_read_stage_result()` 对**不带 `path_version` 的老产物**是"无法判定
            # 版本 ⟹ 放行"（保持既有行为、不制造新 fail）。而 stage 缓存
            # `stage2_vanishing.json` 恰恰不盖版本号——只有
            # `_persist_inprogress_stage_result` 写的中间产物盖。
            # ⟹ 「布局 v1 收敛过、后来插了 λ 变 v2」时，中间产物因版本不符被跳过，
            #    老缓存被放行，`converged=True` 描述的却是**另一条布局**。
            # 任何**据此终止**的判断（分支 0a 的 DONE）必须先问这一条；
            # 只用来路由的读法不受影响。
            "stage_result_path_version_verified": bool(
                stage is not None
                and stage.get("path_version") is not None
                and path.get("version") is not None
                and int(stage["path_version"]) == int(path["version"])
            ),
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
        # 🔑🔑 [审计 #24，2026-09-14] **必须把重分起点传下去。**
        # `SPLIT_TAIL_WINDOW` 落到执行器是 `repartition_tail_from_anchor`：
        # 从 `first_untrusted_window` 的**首态**起重分整个尾段。不传这个参数时
        # `feasible_repair_actions` 会如实返回「判不了」（**不再拿「末窗能否一分为二」
        # 冒充**），于是这个动作永远不可行 —— 所以这里必须算出来传进去。
        # 取不到（没有不可信窗口 / 读不到 λ 表）就不传，保持 fail-closed。
        _t_idx = self.first_untrusted_window(view)
        _t_start = None
        if _t_idx is not None and 0 <= int(_t_idx) < len(ranges):
            _t_start = int(ranges[int(_t_idx)][0])
        out = feasible_repair_actions(
            ranges, int(n_states),
            min_states_per_window=self.lo, max_states_per_window=self.hi,
            n_insert=int(n_insert),
            tail_repartition_start_state=_t_start,
        )
        # 🔑🔑 [2026-09-14 真机] **「结构上拆得开」不等于「执行器拆得了」。**
        # ⚠️ 这道检查与上面那条是**互补**的，两道都要留：
        #   · `feasible_repair_actions` 问「布局上切不切得开」（能否切成更细的合法分窗）；
        #   · 这里问「**执行器拿不拿得到起点**」（anchor 是不是现有共享态）。
        # 上面那份只问布局：K 在不在可拆区间 [2lo−1, 2hi−1]。但执行器还需要一个
        # **anchor**（tail-only 重分的起点），取不到就只能跳过 —— 于是控制器发一个
        # 注定落空的动作，盘面不变，靠通用停滞探测连发 4 次才停。那是失败形状 ①
        # （动作对该窗口的状态在结构上不可能）被当成"再试一次"。
        # 可行性判据必须包含执行器真正需要的每一个前提，就在这一处。
        _tail_anchor = self.tail_repartition_anchor(view)
        if out.get("split_tail_window") is None and _tail_anchor is None:
            out["split_tail_window"] = (
                "取不到 tail anchor（没有 window_idx > 0 的不可信窗口，或读不到 λ 表）"
                " ⟹ 执行器无从下刀。这不是「再试一次」能变的。"
            )
        # 🔑🔑 [2026-09-16 真机] **插 λ 的可行性必须包含「插完以后还能不能合法化」。**
        #
        # 末窗豁免 `max_states_per_window` 只活在 path-version 层：插 λ 让末窗吸收
        # 溢出，允许它涨到可拆上限 `2*hi−1`。但 stage 的权威 `window_ranges` 校验
        # 对**所有**窗口一律要求 K ∈ [lo, hi]，所以末窗一旦超过 `hi` 就**必须**先拆。
        # 而拆末窗要 tail anchor，anchor 取自 `first_untrusted_window`，
        # `tail_repartition_anchor` 在 `idx <= 0` 时恒为 None ——
        # **卡住的窗口是 window 0 时，拆末窗在构造上永远不可行。**
        #
        # ⚠️⚠️ [2026-09-17] **下面这段别叫它「死区」** —— 仓里已经有一个叫死区的东西，
        # 而且是**另一件事**：`abfe_pipeline._legalize_tail_window` 的 docstring 把死区定义为
        # `K ∈ (hi, 2*lo−1)`——既当不了单窗又**根本拆不开**（落在可拆区间
        # `[2lo−1, 2hi−1]` 之外），那是一个**纯区间性质**，与有没有 anchor 无关。
        # 而下面这一段 `hi < K_tail ≤ 2*hi−1` 恰恰**全部落在可拆区间里**：它机械上
        # 拆得开，只是**取不到下刀的点**。两件事共用一个词已经造成过一次误诊
        # （2026-09-17，K=9/lo=4/hi=8 被读成「落在死区」，而真实死区 (8,7) 是空集）。
        # 下面叫它**anchor 缺口**：吸收得进、拆得开、就是没人告诉它从哪切。真机
        # cyclod_ligand3/rep1 连插三次（窗口 0 每次都是失败窗口），末窗
        # K=6→7→8→9，第三次插完 K=9 > hi=8 ⟹ 执行器 fail-closed
        # `RuntimeError('末窗 K=9 超执行层上限 8 但取不到 tail anchor ⟹ 无法拆')`，
        # 而那个非法布局**已经落盘**（见 `_legalize_tail_window` 的调用顺序）。
        # cmet_ligand1/rep1、jnk1_ligand1/rep3、p38_ligand2/rep1 是同一个死局的
        # 「插 λ 预算先用完所以没崩」版本。
        #
        # 判据放在可行性这一处：插完会越过 `hi` 且拆不了 ⟹ 这个动作不可行，
        # 如实说出来，别发一个注定把布局搞成非法的动作。
        if out.get("insert_lambda") is None and int(n_insert) > 0:
            _tail_k_now = int(ranges[-1][1]) - int(ranges[-1][0])
            _tail_k_after = _tail_k_now + int(n_insert)
            if _tail_k_after > int(self.hi) and _tail_anchor is None:
                out["insert_lambda"] = (
                    f"插 {int(n_insert)} 个 λ 会让末窗从 K={_tail_k_now} 涨到 "
                    f"K={_tail_k_after} > 执行层上限 {self.hi}，而合法化只能靠拆末窗、"
                    "拆末窗又取不到 tail anchor（没有 window_idx > 0 的不可信窗口）"
                    " ⟹ 插完就是一个**拆不开的非法布局**。这不是「再试一次」能变的："
                    "要么先让 window 0 之后的某个窗口成为不可信窗口（anchor 才存在），"
                    "要么加大输入 λ 总数让末窗有余量。"
                )
        # 🔑🔑 [2026-09-15] **可行性 = 执行器的 dry-run，不是第二套闭式判据。**
        #
        # 上面 `feasible_repair_actions` 用的是闭式不等式
        # `m(lo−1)+1 ≤ T ≤ m(hi−1)+1` 且要求存在 m ≥ m0+1（更细），而执行器
        # `repartition_tail_from_anchor` 先前调的分窗器内部取 `n_windows =
        # min_windows`（最少窗口数，即最宽的窗）⟹ **判据要更细、执行器给更宽**。
        # 纸面复现见 `repartition_tail_from_anchor` 的参数注释。
        # 这正是本仓库反复复发的那个形状：同一个不变量两份实现
        # （`docs/STAGE2_CONTROLLER_DESIGN_2026-09-12.md` 把它记成最贵的 bug）。
        #
        # 现在改成**直接把执行器跑一遍**：`_dry_run_tail_repartition` 是纯函数
        # （不落盘、不采样，微秒级），切得出来才叫可行。执行器落地时调**同一个**
        # 方法、同样的入参，确定性保证拿到同一份 ranges —— 两边不再各切一遍。
        # 方向/缩跨度的硬断言在执行器内部，一并在这里就被验过。
        # ⚠️ 不往这个 dict 里塞 ranges/diag：它的契约是 `{动作名: None | 不可行理由}`，
        # 调用方按 `v is None` 取可行集，混进非动作键迟早出事。
        if out.get("split_tail_window") is None:
            try:
                self._dry_run_tail_repartition(view)
            except Exception as _e:  # noqa: BLE001 —— 切不出来就是不可行，不是崩
                out["split_tail_window"] = (
                    f"执行器 dry-run 切不出更细的尾段：{_e}"
                )
        # 同理，插 λ 有**跨 resume 的累计上限**。原来这个数只出现在一行诊断打印里，
        # `decide()` 和执行器都不看 ⟹ 自治循环无限插点：cyclod_ligand1/rep2 在
        # max_path_insertions=3 下插了 6 次，末窗 K=6→12。失败形状 ⑥（没有停止条件）。
        if out.get("insert_lambda") is None:
            _left = int(view.get("path_insertions_left") or 0)
            if _left <= 0:
                out["insert_lambda"] = (
                    f"插 λ 的跨 resume 累计预算已用尽"
                    f"（max_path_insertions={self.max_path_insertions}，"
                    f"版本链上已插 "
                    f"{int(((view.get('path') or {}).get('events') or {}).get('insert_lambda', 0))} 次）"
                    " ⟹ 继续插就是无限循环。λ 总数不够是**输入问题**，"
                    "应判 HALT_LAMBDA_BUDGET_INSUFFICIENT 由人工改输入。"
                )
        return out

    def rewindow_feasible(self, view: Dict[str, Any],
                          window_idx: Optional[int]) -> Optional[str]:
        """有界重窗（`IMMUTABLE_REWINDOW`）在这个窗口上可不可行。`None` = 可行。

        🔑🔑 [REWIND-01，2026-09-17] 它是**唯一对任意窗口都够得着**的缩跨度动作：
        不动 λ 表、不需要 tail anchor、不吃 `max_path_insertions`
        ⟹ `feasible()` 里挡住 `SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA` 的四个条件
        （取不到 anchor、dry-run 切不出更细、插完越过 `hi`、插点预算用尽）一个都不碰。
        失败窗口是 window 0（本方法的必然坏窗口）时，那两个动作在构造上永远不可行，
        而这一个可行 —— 这就是接它回来的全部理由。

        前提只有两条：
          · **切得开**：子窗划分走 `vanishing_rescue_ranges`（与执行器**同一份**），
            切不出两个各 ≥2 态的重叠子窗就是不可行（执行器那边是 fail-closed 抛错）；
          · **这个父窗还没建过子系综**：一个父窗只切**一层**。子窗再失败不再递归 ——
            不设深度上限就是一个新的无限循环，而子窗的失败另有出口（见死线 D3）。
        """
        if window_idx is None:
            return "没有目标窗口"
        _i = int(window_idx)
        _rw_view = view.get("immutable_rewindow") or {}
        _done = {int(x) for x in (_rw_view.get("parents_done") or [])}
        if _i in _done:
            return (f"窗口 {_i} 在**当前布局**下已经建过有界重窗的子系综 ⟹ 不再切第二层"
                    "（递归重窗没有停止条件）。子窗自己的失败由子窗那条路由处理。")
        # 🔑🔑 [2026-09-17] **放弃过的父窗可以再试一次，但只有一次。**
        # `parents_done` 现在只认 `SAMPLED`（见 `read()` 里那段：不过滤 status 会让
        # 一条 `ABANDONED_NO_PRODUCT` 把物理窗口永久隐身）。代价是"建了又废"的父窗
        # 重新变得可切 ⟹ **建→废→再建** 就是一个新的无限循环。
        # 这里按**总尝试次数**封顶：同一父窗在当前布局下最多两条台账条目
        # （一次正常 + 一次重试）。这与 `max_path_insertions` 同性质，是终身预算。
        # ⚠️ 用**总数**而不是"放弃数"：status 未来若新增别的非 SAMPLED 值，
        # 这道闸不需要跟着改。
        _abandoned = {int(x) for x in (_rw_view.get("parents_abandoned") or [])}
        if _i in _abandoned and int(_rw_view.get("attempts_by_parent", {}).get(
                str(_i), 0) or 0) >= 2:
            return (f"窗口 {_i} 在当前布局下已经建过 2 次有界重窗且都没产出可用子系综 "
                    "⟹ 不再重试（建→废→再建是无限循环）。这是**输入/预算问题**，"
                    "要继续需要人工看那两次为什么 ABANDONED。")
        ranges = ((view.get("path") or {}).get("window_ranges") or [])
        if not (0 <= _i < len(ranges)):
            return f"读不到窗口 {_i} 的 λ 区间（布局里只有 {len(ranges)} 个窗口）"
        _children = vanishing_rescue_ranges([_i], ranges)
        if len(_children) < 2:
            _a, _b = (int(x) for x in ranges[_i])
            return (f"窗口 {_i} 区间 ({_a}, {_b}) 只有 {_b - _a} 个态，"
                    "拆不出两个各自 ≥2 态的重叠子系综 ⟹ 有界重窗不适用。")
        return None

    def rewindow_children(self, view: Dict[str, Any],
                          window_idx: Optional[int]) -> List[Tuple[int, int]]:
        """这个窗口切出来的子系综区间。空 = 切不开。执行器用同一份实现。"""
        ranges = ((view.get("path") or {}).get("window_ranges") or [])
        if window_idx is None or not (0 <= int(window_idx) < len(ranges)):
            return []
        _c = vanishing_rescue_ranges([int(window_idx)], ranges)
        return _c if len(_c) >= 2 else []

    # -------------------------------------------------------------- 决策

    # 🔑🔑 [2026-09-16] **一个窗口修不动了 ≠ 整个 stage 没动作可做。**
    #
    # 真机（brd4_ligand2/rep1、cyclod_ligand2/rep2）：`earliest` 连吃 4 块帧撞上
    # 补帧配额 ⟹ `NO_FEASIBLE_ACTION` ⟹ 主循环 break，而末窗**一块都没批过**、
    # 预热预算还剩 37 万 / 87 万步，全程躺在 `blocked_by_upstream` 里一次都没被看过。
    # 两个 run 的整腿半程漂移（+8.01 / +11.14 kJ/mol）几乎全部来自那个末窗。
    #
    # 这些出口的语义是「**对这个窗口**没有可行动作」，不是「对这条 stage 没有」——
    # `plan()` 里 [审计 #30] 那段注释早就写明「满额是路由信号，不是终态」，但它只
    # 处理了"同一个动作的多个目标窗口"，没处理"换一个窗口重新决策"。
    #
    # 退役条件苛刻，因为 `earliest` 的排序有物理理由（上游重锚会作废下游的 warmup
    # lineage）：**只有当这个窗口已经没有任何动作可做时**，那条理由才自动失效 ——
    # 不会再有重锚，下游的 lineage 就此固定。所以退役只在终态出口上发生，且要求
    # 别处确实还有补帧预算，否则原样返回原来的终态。
    # [2026-09-17 P0] `SUPPORT_ATTRIBUTION_UNKNOWN` 也可退役：用户规格原话
    # 「若还有其他窗口可做，则绕过该单元继续调度，不能让它停掉整跑」——
    # 退役换窗正是这个机制。别处确实没活干时它仍然是真终态。
    # 🗑️ [2026-09-17，用户拍板] **`_RETIRABLE_EXITS` 已删除。**
    # 可退役性不再是一张要记得维护的出口码名单，而是 `plan()` 的一等语义字段
    # `halt_scope`（见模块级 `EXIT_SPECS`）。`_retirable_window()` 只看 scope。

    def _retirable_window(self, view, plan, retired):
        """这一轮的终态该不该退役、换个窗口重来。返回窗口号或 `None`。

        🔑🔑 [2026-09-17，用户拍板] **只看 `halt_scope`，不认 exit 字符串。**
        先前靠 `_RETIRABLE_EXITS` 这张手写名单，漏一个就是「停掉一个仍有活干的
        stage」——实证漏过两次（`SUPPORT_ATTRIBUTION_UNKNOWN`、
        `D3_REWINDOW_DEPTH_EXHAUSTED`）。现在语义由 `EXIT_SPECS` 一处声明、
        由 `plan()` 强制落到每个终态上，漏了会抛而不是静默停。
        `exit` 只负责解释**为什么**停。
        """
        if plan.get("halt_scope") != "TARGET_LOCAL":
            return None
        e = plan.get("blocking_window")
        if e is None or int(e) in {int(x) for x in retired}:
            return None
        recs = {int(w["window_idx"]): w for w in (view.get("windows") or [])}
        w = recs.get(int(e)) or {}
        # ⚠️ 身份不符 / 终态相 / 证据被布局作废 ⟹ **不退役**：这三种不是"这个窗口
        # 修不动"，而是"这条 stage 的证据不成立"。跳过它去采下游等于拿一条已知
        # 不成立的路径继续烧 GPU。
        if w.get("phase") in ("IDENTITY_MISMATCH", "TERMINAL"):
            return None
        if w.get("stale_layout_evidence_only"):
            return None
        # 别处还有补帧预算才谈得上"换一个窗口"；一个都没有时原样交出终态
        # （`HALT_FRAMES_ADMISSION_CAP` 的定义就是全窗满额 ⟹ 这里必然返回 None）。
        _skipped = {
            int(x if not isinstance(x, dict) else x.get("window_index", -1))
            for x in (view.get("skipped_windows") or [])
        } - {-1}
        _done = {int(e)} | {int(x) for x in retired}
        _replaced = {int(x) for x in
                     ((view.get("immutable_rewindow") or {}).get("parents_done") or [])}
        for idx, rec in recs.items():
            if idx in _done:
                continue
            if self._is_routable_candidate(view, rec, _skipped, _replaced):
                return int(e)
        return None

    @staticmethod
    def unrouted_window_signals(view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """本轮**不会被路由、而且不会自己成为 `earliest`** 的窗口信号。唯一实现。

        🔑🔑 [2026-09-17] **`_pick()` 对所有分支一视同仁，但分支分两类。**
        `_pick()` 只服务 `earliest`，目标窗口不在候选里就返空。对「挑哪个窗口干活」
        那类分支，押后无害 —— 上游解决后该窗口自己会成为 `earliest`。
        但对**路由态是 `ELIGIBLE` 的窗口**，押后 = **永远轮不到**：
        `earliest` 取的是第一个 `PROBLEM`，`ELIGIBLE` 永远不当选；而全窗合格时
        `earliest is None` ⟹ `_pick()` 恒返空 ⟹ **每一条分支都不触发**。

        实测（2026-09-17，造盘面逐条验的 15 个 `_pick` 调用点）：15 条里
        13 条的触发条件都会把窗口判成 `PROBLEM`（押后、会轮到），只有 2 条不会：
          · 生产步数未达标（自检 ELIGIBLE）⟹ **永远补不到目标步数**，
            而全窗合格时 run 直接判完成；
          · 疑似脱轨（单块边际 N_eff 塌，自检仍 ELIGIBLE）⟹ 早期预警丢掉。

        ⚠️ 本函数**不改路由**。放宽 `_pick()` 是另一件事（它的注释写明「刻意不放宽」：
        11 条分支靠它的返空维持各自未成文的前提）。这里只保证**不静默**：
        这两类信号恒常出现在 plan 里，谁都能看见它们没被处理、以及为什么。
        """
        out = []
        for w in (view.get("windows") or []):
            if Stage2RepairController._window_routing_state(w) != "ELIGIBLE":
                continue          # 非 ELIGIBLE ⟹ 会自己成为 earliest ⟹ 押后不丢
            idx = int(w["window_idx"])
            ps, tgt = w.get("production_steps"), w.get("production_steps_target")
            if ps is not None and tgt is not None and int(ps) < int(tgt):
                out.append({
                    "window_idx": idx, "signal": "PRODUCTION_BELOW_TARGET",
                    "production_steps": int(ps), "target": int(tgt),
                    "why_never_routed":
                        "自检判 ELIGIBLE ⟹ 路由态不是 PROBLEM ⟹ 永远不会成为 "
                        "`earliest`；而全窗合格时 `earliest is None`，`_pick()` 恒返空 "
                        "⟹ 补足生产步数那条分支一次都不触发。",
                })
            # ⚠️ [2026-09-17] 脱轨那一条**已经修了**（`_suspected` 在 `earliest is None`
            # 时也发 —— 探针 `probe_only=True` 零采样、不重锚，是 `_pick` 的合法例外）。
            # 这里只在**仍然发不出**时报：别处有 PROBLEM 窗口占着 `earliest`，
            # 而本窗口是 ELIGIBLE、永远排不上。那才是真的还丢着。
            if (w.get("derailment_status") not in (None, "OK", "NONE")
                    and any(Stage2RepairController._window_routing_state(o) == "PROBLEM"
                            for o in (view.get("windows") or []))):
                out.append({
                    "window_idx": idx, "signal": "DERAILMENT_SUSPECTED",
                    "derailment_status": w.get("derailment_status"),
                    "why_never_routed":
                        "本窗口自检仍 ELIGIBLE ⟹ 永远不会成为 `earliest`，而别处有 "
                        "PROBLEM 窗口占着它 ⟹ `PROBE_CANDIDATE_FK` 排不上。"
                        "（全窗合格时这条已由 `earliest is None` 的例外发出。）",
                })
            # 🔑🔑 [2026-09-17] **另外两条「能与 ELIGIBLE 共存」的信号，
            # 判定为「不该开例外、但必须可见」。**
            #
            # 与上面两条同属 `_pick` 的第四类（触发条件能与 `ELIGIBLE` 共存 ⟹
            # 该窗口永远不会成为 `earliest` ⟹ 分支一次都不触发），但**结论相反**：
            #   · 上面两条开了例外 —— 它们的动作是**零成本**（`probe_only=True`
            #     离线算候选）或**补足已承诺的配额**，不重锚上游、不改布局；
            #   · 下面两条**不开** —— `RECALIBRATE_FK` 是 `probe_only=False`、
            #     **开新段跑满** `n_steps_per_window`（成本表里"最贵的那个"）；
            #     `INSERT_LAMBDA` 改**全局 λ 表**、影响所有窗口。
            #     为一个**自检合格**的窗口花这个钱没有依据。
            # 同一把尺子（零成本/已承诺、不重锚上游）量出两个方向 —— 这比
            # 「两条都开」或「两条都不开」可信。
            # ⟹ 对这两条，"永远不触发"很可能**就是对的行为**；真正的缺陷只剩
            # **静默**：探针给了建议、没人知道它被丢了。所以只报告，不改路由。
            if idx in {int(x) for x in
                       ((view.get("fk_probe") or {}).get(
                           "recalibration_recommended_windows") or [])}:
                out.append({
                    "window_idx": idx, "signal": "FK_RECALIBRATION_RECOMMENDED",
                    "why_never_routed":
                        "f_k 探针建议重标定，但本窗口自检 ELIGIBLE ⟹ 永远不会成为 "
                        "`earliest` ⟹ `RECALIBRATE_FK` 一次都不触发。"
                        "**这不是待修的 bug**：重标定会开新段跑满 "
                        "`n_steps_per_window`，为一个自检合格的窗口花这个钱没有依据。"
                        "报出来只是为了让「探针的建议被丢了」这件事**不静默**。",
                })
            _h = [float(v) for _nm, v in
                  ((view.get("min_n_eff_over_g_history") or {}).get(idx) or [])]
            _st, _d = marginal_gain_stalled(_h)
            if _st:
                out.append({
                    "window_idx": idx, "signal": "MARGINAL_GAIN_STALLED",
                    "series": _h, "diagnosis": _d,
                    "why_never_routed":
                        "主验收量不随采样上升，但本窗口自检 ELIGIBLE ⟹ 永远不会成为 "
                        "`earliest` ⟹ 那条分支（插 λ 缩跨度）一次都不触发。"
                        "**这不是待修的 bug**：`INSERT_LAMBDA` 改全局 λ 表、影响所有"
                        "窗口，而一个自检合格的窗口**没有要补救的失败**。"
                        "报出来只是为了不静默。",
                })
        return out

    @staticmethod
    def local_validation_cap_hits(view: Dict[str, Any]) -> List[Dict[str, Any]]:
        """打满单周期验证批次上限、但**全局预热预算还有钱**的窗口。唯一实现。

        🔑🔑 [2026-09-17] **`LOCAL_VALIDATION_CAP` 是一个「提问」，不是失败。**
        引擎打这个信号时日志里写的是「交上层决定是否延长预算」「交顶层自治控制器
        重判」—— 它明确把一个决定**上交**给控制器。而控制器对它**有**答案
        （分支 3a：`PROVISIONAL_PRODUCTION`，不扩 15 批上限、用未验证的冻结 f_k
        采一个 +250k 诊断块）。

        真机 (2026-09-17)：窗口 4 打满 15/15 批、烧掉 17.5 万步、报"无结论"，
        而 `earliest` 是窗口 0 ⟹ `_pick()` 判 `0 not in [4]` ⟹ **分支 3a 整条不触发**
        ⟹ 那个提问既没被回答、也没被记下，只换来一句 `blocked_by_upstream`。
        下一轮同样、再下一轮同样。**不是拒绝，是连问题都没被看见。**

        本函数把"谁在提问"抽成共享谓词，`plan()` 据此**恒常**把未被回答的提问
        落进 `local_validation_cap_pending` —— 哪怕本轮的动作属于别的窗口。
        """
        try:
            import ibs_engine as _ie
            cap = int(_ie.IBS_LOCAL_MBAR_GATE_MAX_BATCHES)
        except Exception:      # noqa: BLE001 —— 读不到上限就谈不上"打满"
            return []
        out = []
        for w in (view.get("windows") or []):
            if w.get("phase") not in ("WARMUP_LEARN", "WARMUP_VALIDATE"):
                continue
            if (w.get("warmup_steps_left") or 0) <= 0:
                continue
            if (w.get("frozen_validation_batches") or 0) < cap:
                continue
            # 已经在用临时生产取证的窗口不再算"提问"——它的答案已经给过了。
            if str(w.get("bias_status") or "") == "provisional_production":
                continue
            out.append({
                "window_idx": int(w["window_idx"]),
                "frozen_validation_batches": w.get("frozen_validation_batches"),
                "batch_cap": cap,
                "warmup_steps_left": w.get("warmup_steps_left"),
                "bias_status": w.get("bias_status"),
            })
        return out

    @staticmethod
    def _window_routing_state(x: Dict[str, Any]) -> str:
        """一个窗口的**路由态**：`ELIGIBLE` / `PROBLEM` / `UNKNOWN`。唯一实现。

        [2026-09-17，用户拍板 #1] **从 `_decide_once` 的闭包提到类上。**
        `_is_routable_candidate`（退役判定）原来在旁边**又写了一份**「什么算未解决」
        （`self_verdict in {...} or solver_skip or skipped`），比这一份少了整整两态，
        于是真机那个「还在预热 / 没有 self-support 产物」的末窗两边判得不一样：
        `earliest` 排序认得它，退役判定不认 ⟹ `earliest` 无路可走时也不退役
        ⟹ 末窗全程躺在 `blocked_by_upstream` 里一次没被看过。
        同一个不变量两份实现，本仓最贵的那个形状。现在只有这一份。

        ⚠️ **三态不是两态。** 缺证据（`self_verdict is None`）= `UNKNOWN` =「不知道」，
        对它的正确动作是**去产出证据**、不是重标定 —— 但它**同样是「还有活可干」**，
        所以退役判定只排除 `ELIGIBLE`。
        """
        if x.get("phase") in ("IDENTITY_MISMATCH", "TERMINAL"):
            return "PROBLEM"
        if x.get("verdict") == "STATISTICALLY_REJECTED":
            return "PROBLEM"
        # 🔑 **"证据被我们自己作废" ≠ "还不知道"。**
        # 布局一变（插 λ / 拆末窗），旧证据描述的是另一个窗口几何、已被剔除。
        # 那是**确定的未解决**，不是 UNKNOWN —— 必须挡住下游，否则就会
        # 「跳过前面的窗口去跑后面的」，正是要防的那个失败模式。
        if x.get("stale_layout_evidence_only"):
            return "PROBLEM"
        # 🔑🔑 [DECORR-01] **求解器跳窗优先于自检通过。**
        # 求解器对"能否进入求解"有操作权威：它把这个窗口踢出了 MBAR 与协方差链，
        # 那这个窗口就是**未解决**，哪怕自检说 `ANALYSIS_ELIGIBLE`。
        # 实测 win3：自检 56 帧 `sufficient=True`，求解器报 9/7 帧并跳窗 ——
        # 先前这里只看自检，于是它被判 ELIGIBLE、earliest 越过它去处理下游，
        # 而它的帧一次都没补上。必须放在 `ANALYSIS_ELIGIBLE` **之前**。
        if x.get("solver_skip"):
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

    # 一个窗口「还有活可干」的**统一**判据。退役判定与 earliest 排序共用
    # `_window_routing_state`，这里只再加"预算花不花得出去"。
    @staticmethod
    def _is_routable_candidate(view, rec, skipped, replaced) -> bool:
        """这个窗口**既未解决、又确实还有自己的预算**可花。

        🔑🔑 [2026-09-17，用户拍板 #1] **「未解决」原来漏掉了最要命的那一形态。**

        先前判据是 `self_verdict ∈ {HARD_INSUFFICIENT, INSUFFICIENT_DATA}`
        或被求解器跳过。而真机 brd4_ligand2/rep1、cyclod_ligand2/rep2 的末窗是：
          · 还在 `WARMUP_LEARN` / `WARMUP_VALIDATE`；
          · **没有 self-support 产物**（那份是生产跑完才写的）⟹ `self_verdict is None`；
          · `has_convergence=False`；
          · 预热预算还剩 37 万 / 87 万步。
        ⟹ 三个条件一条都不命中 ⟹ **不算"未解决"** ⟹ 不构成退役理由 ⟹
        `earliest` 死在原地、这个末窗全程躺在 `blocked_by_upstream` 里一次没被看过，
        而整腿半程漂移（+8.01 / +11.14 kJ/mol）几乎全部来自它。

        ⚠️ **这不是 round-robin**：上游重锚 / rewindow 确实会作废下游的 warmup
        lineage，所以 `earliest` 的因果顺序**保留**。这里只回答一个问题：
        「earliest 已经无路可走时，别处到底还有没有活干」——答错会让整跑提前收摊。

        ⚠️ **预算必须是这个窗口自己的**，而且 `None` = 未知 ⟹ **不放行**（fail-closed）：
        先前只问生产块 headroom，于是预热期窗口（生产账为空）一律判成"没预算"。
        现在预热账与生产账任一有余量即可 —— 两者是两本账，互不代替。
        """
        idx = int(rec["window_idx"])
        # 这三种不是"还没修"，是"这条证据不成立" ⟹ 绕过去采下游等于拿一条已知
        # 不成立的路径继续烧 GPU（与 `_retirable_window` 开头那三道同一口径）。
        if rec.get("phase") in ("IDENTITY_MISMATCH", "TERMINAL"):
            return False
        if rec.get("stale_layout_evidence_only"):
            return False
        if idx in replaced:
            return False          # 已被子系综接管 ⟹ 修父窗没有意义
        # 「未解决」走**与 earliest 排序同一份**判据（`_window_routing_state`），
        # 不在这里另写。`ELIGIBLE` 之外的两态（`PROBLEM` / `UNKNOWN`）都是还有活可干：
        # 前者有明确问题，后者「不知道」—— 而对「不知道」的正确动作是去产出证据。
        # 被求解器跳过的窗口另加一条（它可能带着 ELIGIBLE 的自检却根本没进 MBAR）。
        if (Stage2RepairController._window_routing_state(rec) == "ELIGIBLE"
                and rec.get("solver_skip") is None and idx not in skipped):
            return False
        # 预算：这个窗口**花得出去**的余量。None（读不到）不算有 —— fail-closed。
        #
        # ⚠️ **"账上有钱"不等于"这笔钱它能花"。** 预热账只对**还在花它**的窗口算数：
        # 仍在 `WARMUP_*` 相，或 f_k 还没 verified（那笔钱正是用来验它的）。
        # 一个已经进 `PRODUCTION` 相、f_k 已 verified、生产配额却烧光的窗口，
        # 预热余额再多也推不动它一步 —— 把它算成"还有活干"会让退役循环空转一圈，
        # 而且会让「别处确实还有预算」这个退役前提变成假的。
        # （这条是实测出来的：`test_nothing_left_anywhere_still_terminates` 的盘面
        #   正是"生产烧光、预热账面还剩 50.5 万步"。）
        _wl = rec.get("warmup_steps_left")
        _warmup_spendable = (
            rec.get("phase") in ("WARMUP_LEARN", "WARMUP_VALIDATE")
            or str(rec.get("f_k_evidence_status") or "") != "verified"
        )
        if _warmup_spendable and _wl is not None and int(_wl) > 0:
            return True
        _hr = frames_growth_headroom(view, idx)
        return _hr is not None and float(_hr) > 1.0

    def decide(self, view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """**纯函数式判断：下一步做什么。** 不执行、不落盘。

        这一层只做一件事：`_decide_once` 给出「对 `earliest` 没有可行动作」的终态时，
        把那个窗口退役、换下一个还有预算的未解决窗口重新决策（见上面那段）。
        每轮至多退役一个窗口、退役过的不再参选 ⟹ 必然终止。
        """
        view = view or self.read()
        retired: List[int] = []
        while True:
            plan = self._decide_once(view, retired=tuple(retired))
            nxt = self._retirable_window(view, plan, retired)
            if nxt is None:
                if retired:
                    plan["retired_windows"] = list(retired)
                    plan["reason"] = (
                        f"（窗口 {retired} 本轮已无任何可行动作、**已退役出路由顺序**："
                        "它们不会再被重锚，下游窗口的 lineage 就此固定 ⟹ 继续修下一个"
                        "还有预算的窗口。退役**不等于合格**，它们的 verdict 原样保留、"
                        "质量门照判。）"
                    ) + str(plan.get("reason") or "")
                return plan
            retired.append(nxt)

    def _decide_once(
        self, view: Optional[Dict[str, Any]] = None, retired: Sequence[int] = (),
    ) -> Dict[str, Any]:
        """一轮决策。`retired` 里的窗口不参与 `earliest` 排序，其余逐字不变。

        优先级是刻意排的，理由写在每一条里。核心一条：**`UNMEASURED`（没测出来）
        永远不当 `FAIL`（测出来不合格）用** —— 前者加预算，后者换 Epoch；混起来
        就会退化成"再测一次直到碰巧通过"。
        """
        view = view or self.read()
        W = view["windows"]
        feas = self.feasible(view)

        # 🔑🔑 [2026-09-17] **本轮的路由上下文。** `earliest` / `blocked` 一算出来就
        # 存进这里，`plan()` 在调用点没传时从这里取。
        #
        # 先前的约定是「**必须显式传**，不许从闭包里取」，理由是闭包引用会在**全新
        # 运行**上炸 NameError（分支 0 在 `earliest` 被绑定**之前**就 return）。那个
        # 理由只对**裸闭包引用**成立；代价是同一对参数要在 **44 个**调用点各写一遍，
        # 而漏一个是静默的 —— 实测漏了 6 个（分支 2 refuted / 3a 验证不可达的两个出口 /
        # 3a PROVISIONAL_PRODUCTION / 5a-0 PROBE_CANDIDATE_FK / 边际停滞的 INSERT_LAMBDA）。
        #
        # 后果不是"少了一行诊断"：`decide()` 的**退役换窗**（2026-09-16 那条，专治
        # 「末窗全程躺在 `blocked_by_upstream` 里一次没被看过」）靠
        # `plan["earliest_unresolved_window"]` 找退役目标，`None` 就直接 `return None`
        # ⟹ 这六条路径上退役**永不触发**。而它们恰恰都通向 `plan()` 内部的两道改写闸
        # （换 Epoch 付不起 → `RUN_PRODUCTION` → 补帧准入 → `NO_FEASIBLE_ACTION` /
        # `HALT_FRAMES_ADMISSION_CAP`），也就是 `_RETIRABLE_EXITS` 的**全部**两个值。
        #
        # 用一个**字典**而不是裸闭包：分支 0 / 0a 在它被填之前 return，`.get()` 返回
        # `None`，行为与先前逐字相同 —— NameError 那条理由被保住，重复那 44 遍被删掉。
        # 显式传参仍然优先（调用点想覆盖就覆盖）。
        def _pw_or_last(u):
            """子窗的父窗号；**读不出来就排到最后**，不冒充 window 0。

            [2026-09-17] 先前全仓五处写 `int(u.get("parent_window") or 0)`。
            「未知不是零」在这里有两个后果：排序上插到队首、块账上记到 window 0
            （它是本方法的必然坏窗口、配额最金贵）。目前执行器两处都写
            `int(window_idx)` 所以不会触发，但那是**静默**错，出事时没有迹象。
            """
            _v = u.get("parent_window")
            return 1 << 30 if _v is None else int(_v)

        _ctx: Dict[str, Any] = {}

        def plan(action, reason, *, exit_=None, windows=None, missing=None,
                 blocked=None, earliest=None, unit_id=None, extra=None,
                 halt_scope=None, blocking_window=None, blocking_unit=None):
            """`blocked` / `earliest` 不传时从 `_ctx` 取（见它上面那段）。

            `halt_scope`：**停的是谁** —— `TARGET_LOCAL` / `STAGE_GLOBAL` /
            `PATH_INVALID`。终态出口不传时从 `EXIT_SPECS` 取默认；
            `NO_FEASIBLE_ACTION` 没有默认（它同时承载局部与全局两种语义），
            漏传直接抛 —— 见 `EXIT_SPECS` 里那条的 `why`。
            """
            if earliest is None:
                earliest = _ctx.get("earliest")
            blocked = list((blocked if blocked is not None
                            else _ctx.get("blocked")) or [])
            # 🔑🔑 [审计 #20，2026-09-14] **换 Epoch 的预算预检只在真要换 Epoch 时查。**
            #
            # 先前它是一道**无条件**的前置闸：`decide()` 一算出 `earliest` 就对它跑
            # `_epoch_validation_unaffordable`，不分接下来要发什么动作，而且对
            # `warmup_steps_left is None` 是 fail-closed ⟹
            #   · 账本读不到的窗口被**永久钉死在 `RUN_PRODUCTION`**，分支
            #     1c/1d/3/3a/4/5/5a/5b/6 全部不可达（#21/#22/#23 三条都被它盖住）；
            #   · `left` 介于 0 和 floor 之间时，还会在一份**尚未冻结/验证**的 f_k 上
            #     直接开生产。
            # 现在挪进 `plan()`：只有动作**真的是换 Epoch 那一族**才查，付不起就改发
            # `RUN_PRODUCTION`（保持原来的 `HALT_BUDGET` 路由语义与理由口径）。
            # ⚠️ 必须排在下面那道补帧准入**之前** —— 它改出来的 `RUN_PRODUCTION`
            # 同样要过块数上限，否则就成了一条绕过刹车的旁路。
            _EPOCH_ACTIONS = (
                "RECALIBRATE_FK", "RELEARN_FK_EPOCH", "PROBE_REANCHOR_EPOCH")
            if action in _EPOCH_ACTIONS and windows:
                _ep_ws = [int(x) for x in windows]
                _ep_broke = self._epoch_validation_unaffordable(
                    view, _ep_ws, new_epoch=(action == "RELEARN_FK_EPOCH"))
                if _ep_broke:
                    _orig = action
                    # 🔑🔑 [2026-09-15] **降级到补帧，只对「样本量类」失败成立。**
                    #
                    # 这条降级的原意是「换 Epoch 付不起 ⟹ 退而做一件便宜且有用的事」。
                    # 但对**偏斜类**窗口，补帧不是便宜且有用，是**便宜且无用**：
                    # 它拿的正是那份需要被换掉的冻结 f_k，采出来的帧权重剖面一模一样。
                    # 本仓 §5.1 实测：同分布 250k→1M 让 top1% 从 0.545 涨到 0.762、
                    # ESS 比值**反而更差**；重标定一次 rawESS 27.5→503。
                    #
                    # 真机后果（`采样问题_2026-09-15.md` 上半部分那条链的落点）：
                    #     warmup 账干 → 换 Epoch 付不起 → 降级补帧 → 生产预算判不了它
                    #     不可行 → 连补至 max_production_blocks=4 → NO_FEASIBLE_ACTION
                    # 三个 rep **一次都没重标定过**，各自把 1.25M 步砸在一份冻结 f_k 上。
                    #
                    # ⚠️ 归因用与别处**同一份**实现（`support_failure_is_skew`），
                    # 不在这里另写判据。`_is_skew` 是 `decide()` 里的闭包、此刻可能
                    # 还没绑定（`plan()` 的调用点早于它的定义），所以直接调模块级函数。
                    _ep_recs = [w for w in view["windows"]
                                if int(w["window_idx"]) in set(_ep_ws)]
                    # 🔑🔑 [2026-09-17，用户拍板 A] **降级到补帧要「样本量类」的**硬证据**，
                    # 「还没被证伪」不算。**
                    #
                    # 先前这里用的是 `not is_skew`，而 `is_skew` 只区分两态 ⟹
                    # `UNKNOWN`（射程说"也许够得着"）被算进了"样本量类"，于是一个
                    # **乐观上界**就足以让 `plan()` 把分支刚判出的「f_k 不对，换 Epoch」
                    # 静默改写成「补帧」——把一个**还没被证伪**的猜测升级成了**确定归因**。
                    # 而那个上界假定 η 与 g 恒定，两个假设实测都朝不利方向走
                    # （见 `support_failure_attribution` 的长注释）。
                    # 现在只有 `SAMPLE_SIZE`（`solver_eligibility` 族来源，或明确的
                    # `n_decorrelated < min_frames`）才授权降级。
                    _ep_attr = {
                        int(w["window_idx"]): support_failure_attribution(
                            w.get("self_verdict"), w.get("self_verdict_source"),
                            n_decorrelated=w.get("self_n_frames_decorrelated"),
                            min_frames=w.get("self_min_frames"),
                            min_n_eff_over_g=w.get("min_n_eff_over_g"),
                            n_eff_over_g_target=w.get("self_n_eff_over_g_eligible"),
                            frames_headroom=frames_growth_headroom(
                                view, int(w["window_idx"])))
                        for w in _ep_recs
                    }
                    _ep_skew = [i for i, a in _ep_attr.items()
                                if a == SUPPORT_FAILURE_STRUCTURAL]
                    _ep_unknown = [i for i, a in _ep_attr.items()
                                   if a == SUPPORT_FAILURE_UNKNOWN]
                    # 🔑 换不起 Epoch 时**优先走有界重窗**，不是退回补帧。
                    # 它不动 λ 表、不需要 tail anchor、不吃插点预算，而且子系综**各自
                    # 锁一份自己的 f_k** ⟹ 它正面回答了「这份 f_k 不对」这个诊断，
                    # 而补帧恰恰是拿那份要被换掉的 f_k 继续采。
                    _ep_rw = None
                    if _ep_skew or _ep_unknown:
                        _ep_tgt = (_ep_skew or _ep_unknown)[0]
                        if self.rewindow_feasible(view, _ep_tgt) is None:
                            _ep_rw = int(_ep_tgt)
                    if _ep_rw is not None:
                        _why_attr = (f"{_ep_skew} 的自检是**支撑/偏斜类**失败"
                                     if _ep_skew else
                                     f"{_ep_unknown} 的归因是 `UNKNOWN`"
                                     "（射程只说明「还没被证伪」，不是「帧不够」）")
                        action, exit_ = "IMMUTABLE_REWINDOW", None
                        windows = [_ep_rw]
                        reason = (
                            f"原动作 `{_orig}` 要换 Epoch，但窗口 {_ep_ws} **付不起新 "
                            f"Epoch 的最低验证额度**（{_ep_broke}）。而 {_why_attr} ⟹ "
                            "**不降级去补帧**（补帧用的正是那份需要被换掉的冻结 f_k，"
                            "采出来的帧权重剖面一模一样）。改走**有界重窗**："
                            f"在窗口 {_ep_rw} 内建重叠子系综，**各自锁一份自己的 f_k** ⟹ "
                            "它正面回答了「这份 f_k 不对」这个诊断，而且不花验证额度、"
                            "不动 λ 表。原动作与理由：" + reason
                        )
                    elif _ep_skew or _ep_unknown:
                        action, exit_ = "NO_ACTION", "HALT_BUDGET"
                        reason = (
                            f"原动作 `{_orig}` 要换 Epoch，但窗口 {_ep_ws} **付不起新 "
                            f"Epoch 的最低验证额度**（{_ep_broke}）。"
                            + (f"而其中 {_ep_skew} 的自检是**支撑/偏斜类**失败 ⟹ "
                               if _ep_skew else
                               f"而 {_ep_unknown} 的归因是 `UNKNOWN` —— 射程只说明"
                               "「还没被证伪」（它假定 η 与 g 恒定，两者实测都朝不利"
                               "方向走），**不足以断定「帧不够」** ⟹ ")
                            + "**不降级去补帧**：补帧用的正是那份需要被换掉的冻结 f_k，"
                            "采出来的帧权重剖面一模一样（§5.1 实测同分布 250k→1M 让 "
                            "top1% 0.545→0.762、ESS 比值反而更差）。"
                            "便宜且无用的动作不该顶替一个做不起的对症动作；"
                            f"有界重窗也不可行（{self.rewindow_feasible(view, (_ep_skew or _ep_unknown)[0])}）"
                            " ⟹ 如实停下。**要继续必须显式提高该窗口的预热/验证预算**，"
                            "或改走布局动作（缩跨度）。原动作与理由：" + reason
                        )
                    else:
                        action, exit_ = "RUN_PRODUCTION", "HALT_BUDGET"
                        reason = (
                            f"原动作 `{_orig}` 要换 Epoch，但窗口 {_ep_ws} **付不起新 "
                            f"Epoch 的最低验证额度**（{_ep_broke}）⟹ **不启动**。"
                            "启动之后才发现没预算验，正是 win2 连死三次的形状。"
                            "低支撑/没预算永远是「尚不可测」，不是 FAIL ⟹ 退而补生产帧"
                            "（用的是已冻结的 f_k，一步验证预算都不花，两本账互不代替）。"
                            "⚠️ 这条降级只对**样本量类**失败成立，而且要的是**硬证据**"
                            "（`solver_eligibility` 族来源，或明确的 "
                            "`n_decorrelated < min_frames`）—— 射程判据说的"
                            "「也许够得着」是 `UNKNOWN`，**不授权**这条降级"
                            "（2026-09-17 收紧；先前 `not is_skew` 把 UNKNOWN 也算了进来，"
                            "等于拿一个乐观上界覆盖掉分支刚做出的对症诊断）。"
                            "偏斜类与 UNKNOWN 走上面那两条。"
                            "原动作与理由：" + reason
                        )
            # 🔑🔑 [裁决 2] **只有生产预算「已知且确实耗尽」才终止。**
            # 花 GPU 的动作（除 ANALYZE / DONE / NO_ACTION 之外全都花）在这里统一
            # 被拦一道 —— 放在 `plan()` 里而不是逐个分支，是因为逐个分支必然漏。
            # ⚠️ 上限 unknown 时**不拦**：未知不是零，"预算耗尽"这个结论不成立。
            # ⚠️ 预热/验证预算（`warmup_steps_left`）**不参与**这道判断，它是另一本账。
            # 🔑🔑 [2026-09-14] **补帧准入：放在 `plan()` 里，不逐个分支挂。**
            # `decide()` 有 **16 个**发 `RUN_PRODUCTION` 的出口；逐个挂门必然漏 ——
            # 实测就漏了 15 个（其中两处的改动还因为同一个脚本里后面的 assert 抛错
            # 而整份写入被中止、悄悄丢掉）。这里一次挂住全部。
            if action == "RUN_PRODUCTION":
                # 🔑🔑 [CTL-13，2026-09-14] **准入要逐个目标窗口判，不能只看第一个。**
                # 先前是 `_tgt_w = int(windows[0])` —— 只拿第一个窗口的块账去问准入。
                # 而分支 9c 在「端点 σ 归因不到具体窗口」时发的是
                # `windows=sorted(全部窗口)`（那是有意的：归因不到就全窗各补一块）。
                # ⟹ win0 块数一满就把**整批窗口**的补帧一起毙掉；反过来 win0 还有
                # 额度时，其余已超额的窗口也照批。两个方向都错。
                # 现在：逐窗判，**只保留还能批的那些**；一个都不剩才是 NO_ACTION。
                _tgt_ws = []
                if unit_id:
                    # 🔑 [契约 B] 子窗查**子窗自己那本账** —— 先前拿父窗的账去问，
                    # 而父窗的步数已经冻结 ⟹ 去重后恒 1 行 ⟹ 准入恒放行。
                    # 🔑🔑 [2026-09-17] **`or 0` 会把「父窗未知」记到窗口 0 的块账上。**
                    # 「未知不是零」——这里尤其致命：块账决定补帧准入，把一个
                    # 未知父窗的子窗算进 window 0，会**替 window 0 消耗配额**
                    # （它是本方法的必然坏窗口，配额最金贵），同时让真正的父窗
                    # 一块都不记、准入恒放行。实测目前 `parent_window` 一直有值
                    # （执行器两处都写 `int(window_idx)`），但这不是不修的理由：
                    # 它是**静默**错，出事时没有任何迹象。
                    _u_pw = next(
                        (u.get("parent_window")
                         for u in (view.get("sampling_units") or [])
                         if str(u.get("unit_id")) == str(unit_id)), None)
                    if _u_pw is not None:
                        _tgt_ws = [int(_u_pw)]
                elif windows:
                    _tgt_ws = [int(x) for x in windows]
                _adm = {w: _frames_admission(w, unit_id=unit_id if unit_id else None)
                        for w in _tgt_ws}
                _ok_ws = [w for w in _tgt_ws if _adm[w][0]]
                if _tgt_ws and _ok_ws and len(_ok_ws) < len(_tgt_ws) and not unit_id:
                    # 部分窗口还能批 ⟹ **只补那几个**，别因为一个窗口满额就整批作废。
                    _dropped = [w for w in _tgt_ws if not _adm[w][0]]
                    windows = list(_ok_ws)
                    reason = (
                        f"（窗口 {_dropped} 已达补帧准入上限、本轮剔除："
                        + "；".join(f"w{w}: {_adm[w][1]}" for w in _dropped)
                        + f"）仍然补 {_ok_ws}。原理由：" + reason
                    )
                if _tgt_ws and not _ok_ws:
                    _adm_why = "；".join(f"w{w}: {_adm[w][1]}" for w in _tgt_ws)
                    # 🔑🔑 [审计 #30，2026-09-14] **满额是路由信号，不是终态。**
                    # 这道闸在 `plan()` 内部，而那时分支**已经锁定了单个窗口** ⟹
                    # 先前发 `NO_FEASIBLE_ACTION`（终态）等于「一个窗口满额就终止
                    # 整跑」，别的窗口和全部布局动作一个都没试过。
                    # 只有**所有**窗口都满额（按跨段累计那本判）才谈得上"没动作可做"。
                    _tot_all = view.get("production_blocks_total_by_window") or {}
                    _all_w = [int(w["window_idx"]) for w in view.get("windows") or []]
                    _all_full = bool(_all_w) and all(
                        len(_tot_all.get(int(i)) or []) >= int(
                            view.get("max_production_blocks_per_window") or 4)
                        for i in _all_w
                    )
                    # 两个都是终态（见 `TERMINAL_EXITS` 里那段注释）：这道闸只在
                    # 目标窗口**一个都批不了**时走到这里，而部分满额早就在上面被
                    # 「剔掉满额的、只补剩下的」处理掉了。`HALT_FRAMES_ADMISSION_CAP`
                    # 只是比笼统的 `NO_FEASIBLE_ACTION` 说得更具体：补帧配额用尽。
                    action = "NO_ACTION"
                    exit_ = ("HALT_FRAMES_ADMISSION_CAP" if _all_full
                             else "NO_FEASIBLE_ACTION")
                    # [2026-09-17] 全窗满额 ⟹ stage 级（`EXIT_SPECS` 给默认）；
                    # 否则只是**本轮点名的这些窗口**批不过 ⟹ 局部，允许换窗。
                    if not _all_full:
                        halt_scope = "TARGET_LOCAL"
                    reason = (
                        f"原动作是给 {_tgt_ws}{f'（子窗 {unit_id}）' if unit_id else ''}"
                        f" 补帧，但**不再自动补**：{_adm_why}"
                        "「又产生了帧」不等于「获得了有用证据」。"
                        + ("**所有**窗口的补帧块数配额都已用尽 ⟹ 没有可做的补帧动作了"
                           "（`HALT_FRAMES_ADMISSION_CAP`）。"
                           if _all_full else
                           "本轮点名的窗口一个都批不过（部分满额时上面已经剔掉满额的、"
                           "只补剩下的，走不到这里）。")
                        + "要对这个窗口继续补帧必须显式提高它的块数上限，"
                        "或改走布局动作。原动作与理由：" + reason
                    )

            # 🔑🔑 [CTL-12，2026-09-14] **no-op 台账是执行器对**每一个**动作记的，
            # 判读却只有三个动作在做 —— 差的那几个正是真机上把循环转死的那几个。**
            #
            # `_record_noop_action` 是通用的（执行前后各取一次盘面指纹，一样就记账），
            # 而 `decide()` 里 `_is_noop()` 只在 `CONTINUE_WARMUP` / `RECALIBRATE_FK` /
            # `PROBE_REANCHOR_EPOCH` 三个动作上被查过。`SPLIT_TAIL_WINDOW` /
            # `INSERT_LAMBDA` / `RUN_PRODUCTION` / `IMMUTABLE_REWINDOW` 一概不查。
            # 真机 cyclod_ligand1/rep2 与 cyclod_ligand2/rep1 都死在同一条路上：
            # `SPLIT_TAIL_WINDOW` 连发 4 次，执行器每次都打了「记为当前盘面上的
            # no-op」，控制器每次都看不见，最后由停滞保护给出 NO_FEASIBLE_ACTION ——
            # 白烧三轮，而且退出理由说成「推不动」而不是「这个动作在这个盘面上
            # 已经被证明什么也不做」。
            #
            # 这里统一挂一道：**已记账为 no-op 的动作不再发**，如实给出口。
            # ⚠️ 指纹带路径版本与生产步数 ⟹ 补过帧 / 换过段 / 插过 λ 之后记录自动
            # 失效、动作重新可选。所以这不是"永久封杀一个动作"。
            # ⚠️ 上面那三个已有判读点**更强**（它们会改发一个对症动作而不是停下），
            # 它们在调用 `plan()` 之前就换好了动作，不会被这道闸碰到。
            # ⚠️ `ANALYZE` 不挂：它是产出证据的动作，判它 no-op 的风险大于收益。
            _NOOP_GUARDED = (
                "RUN_PRODUCTION", "INSERT_LAMBDA", "SPLIT_TAIL_WINDOW",
                "IMMUTABLE_REWINDOW", "RELEARN_FK_EPOCH", "PROBE_CANDIDATE_FK",
                "PROVISIONAL_PRODUCTION",
            )
            # [审计 #8，契约 A] `and not unit_id` 删掉 —— 子窗动作先前**显式被排除**
            # 在这道闸之外，于是子窗那边一条 no-op 刹车都没有。
            if action in _NOOP_GUARDED and (windows or unit_id):
                if unit_id:
                    _dead = ([str(unit_id)]
                             if _is_noop(action, None, unit_id=unit_id) else [])
                    _n_tgt_noop = 1
                else:
                    _dead = [int(w) for w in windows if _is_noop(action, int(w))]
                    _n_tgt_noop = len(windows)
                if _dead and len(_dead) == _n_tgt_noop:
                    _dead_action = action
                    action, exit_ = "NO_ACTION", "NO_FEASIBLE_ACTION"
                    # 这个动作对**这些目标**是 no-op ⟹ 局部无路，别处照常可修。
                    halt_scope = "TARGET_LOCAL"
                    reason = (
                        f"原动作 `{_dead_action}` "
                        f"对 {_dead} 在**当前盘面**上已被执行器记账为 no-op"
                        "（跑过一次、盘面逐项未变）⟹ 再发一次仍然什么也不做。"
                        "盘面一变（补帧 / 换段 / 插 λ）这条记录自动失效、动作重新可选。"
                        "原动作与理由：" + reason
                    )

            # 🔑🔑 [2026-09-15 审计②③] **终态不许和自己的证据打架。**
            #
            # 两条 DONE 分支各自判一遍前提，而 `evidence_status` 在**下面**才算：
            #   · 0a（4331）查了 `_layout_trustworthy` / `stale` / `coverage`，
            #     但**没查逐窗 `STATISTICALLY_REJECTED` / `phase == TERMINAL`**
            #     ⟹ 一份 f_k 已被统计驳回的窗口仍可被宣布 DONE；
            #   · 分支 7（5603）**一道都没查** —— 而 0a 的注释写着「两条都不满足
            #     就往下走路由，由分支 7 兜底（那条路径上保护仍然成立）」，
            #     **那句话是假的**。真机形状：path v2 + stage 缓存不盖 path_version
            #     ⟹ 0a 跳过 ⟹ 分支 7 直接 DONE，发布一个描述**另一条布局**的 ΔG。
            #
            # 逐条去补每个分支 = 又一次"同一不变量 N 份实现"。改在**唯一的出口**：
            # `DONE` 的证据必须是 `CONVERGED`；不是就不是终态，如实降级并说清楚
            # 是哪一维在反对。`allow_untrusted` 只改 `trust_level`，不救这里 ——
            # 它的语义是"门没过但我放行"，不是"把反面证据改写成正面"。
            _ev_now = self._evidence_status(view, action, exit_)
            if action == "DONE" and _ev_now != "CONVERGED":
                _bad_w = sorted(
                    int(w["window_idx"]) for w in (view.get("windows") or [])
                    if w.get("verdict") == "STATISTICALLY_REJECTED"
                    or w.get("phase") == "TERMINAL"
                )
                action, exit_ = "NO_ACTION", "HALT_EVIDENCE_CONTRADICTS_DONE"
                reason = (
                    f"原动作 `DONE` 的证据是 `{_ev_now}`（不是 `CONVERGED`）⟹ "
                    "**这不是完成**。"
                    + (f"逐窗反面证据：窗口 {_bad_w} 的 f_k 被统计驳回或已进终态。"
                       if _bad_w else
                       "布局版本未核实 / 证据被布局变更作废 / 覆盖有缺口"
                       "（见 `missing_evidence`）。")
                    + "原理由：" + reason
                )

            _pb = view.get("production_budget") or {}
            # 🔑🔑 [审计 #34，2026-09-14] **只有真正产生生产帧的动作才走生产账。**
            # 先前 `CONTINUE_WARMUP` / `RELEARN_FK_EPOCH` / `INSERT_LAMBDA` /
            # `SPLIT_TAIL_WINDOW` 也被按一整块**生产**预算收费，付不起就发
            # `GLOBAL_BUDGET_EXHAUSTED`（终态）—— 正是这套双账要防的
            # 「拿 A 账本余额终止只花 B 账本的动作」，方向还反了。
            #   · `CONTINUE_WARMUP` / `RELEARN_FK_EPOCH` 花的是**预热账**
            #     （`warmup_steps_left`），由 `_epoch_validation_unaffordable` 那条路管；
            #   · `INSERT_LAMBDA` / `SPLIT_TAIL_WINDOW` 是**布局动作**，本身不直接花
            #     采样预算（它们引发的重采由下一轮的 `RUN_PRODUCTION` 计费）。
            _PRODUCTION_CHARGED = (
                "RUN_PRODUCTION", "PROBE_REANCHOR_EPOCH",
                "RECALIBRATE_FK", "IMMUTABLE_REWINDOW",
                # [2026-09-15] 临时生产花的是**生产账**（它采生产帧），
                # 不是预热账 —— 与上面 `_BLOCK_CHARGING_ACTIONS` 同一理由。
                "PROVISIONAL_PRODUCTION",
            )
            if action in _PRODUCTION_CHARGED and _pb.get("cap_known"):
                # 🔑 [2026-09-14] **按下一动作的完整成本准入，不是只问"已经耗尽"。**
                # 先前剩 100k 也照发 250k 块 —— 那等于允许超支一整块。
                _blk = int(_pb.get("production_block_steps") or 0)
                _n_tgt = max(1, len(windows or [1]))
                # 🔑🔑 [审计 #32] **`_NON_SAMPLING` 原来选错了动作对。**
                # 实际派发（`abfe_pipeline`）：
                #   · `RECALIBRATE_FK`      `probe_only=False`，**开新段跑满**
                #     `n_steps_per_window` ⟹ 最贵的那个，先前却被判零成本；
                #   · `PROBE_CANDIDATE_FK`  `probe_only=True`，离线算候选、零采样
                #     ⟹ 它才是真正的非变异探针（连这道闸都进不到，见上面的动作表）；
                #   · `PROBE_REANCHOR_EPOCH` 明确「候选 f_k + 独立 burn-in +
                #     **一个 +250k 块**」⟹ 按一块收。
                if action == "PROBE_REANCHOR_EPOCH":
                    _cost = _blk
                elif action == "IMMUTABLE_REWINDOW":
                    # 🔑 [审计 #44] 建新系综：每个子窗一个首块。子窗数从 model B 的
                    # 实际拆法算 —— 先前硬编码 2，而执行器按 `len(children) × base_unit`
                    # 收费，两边一旦不等就是「预留了 2 块、实际花 3 块」。
                    # 子窗数的权威是 `vanishing_rescue_ranges`（执行器
                    # `_build_vanishing_rescue_ranges` 转发到同一份）。
                    # [REWIND-01，2026-09-17] 这里原来自己写 `2 if (b-a) > 2 else 1`
                    # 复算了一遍 —— 那份实现搬到模块级之后直接调，两边不再各写一遍。
                    _n_children = None
                    try:
                        _n_children = len(self.rewindow_children(
                            view, int(windows[0]) if windows else None)) or None
                    except Exception:  # noqa: BLE001
                        _n_children = None
                    if not _n_children:
                        _n_children = 2      # 拿不到就保守取 model B 的最小拆法
                    _cost = int(_n_children) * int(
                        _pb.get("new_ensemble_reserve_steps") or _blk)
                else:
                    _cost = _blk * _n_tgt
                _left = _pb.get("stage_remaining_steps")
                # ⚠️ 余量**未知**（上限未知，或用量账不完整）⟹ 拦不拦都没有依据。
                # 先前 `or 0` 把 None 当成 0 ⟹ 一律拦死，等于"未知 = 耗尽"。
                if _left is not None and int(_left) < _cost:
                    action, exit_ = "NO_ACTION", "GLOBAL_BUDGET_EXHAUSTED"
                    reason = (
                        f"原动作需要 {_cost} 步生产预算，而 stage 只剩 {int(_left)} 步"
                        f"（已用 {_pb.get('stage_used_steps')} / 上限 "
                        f"{_pb.get('stage_cap_steps')}，范围 {_pb.get('cap_scope')}，"
                        f"来源 {_pb.get('cap_source')}）⟹ **不发一个付不起的动作**。"
                        "这与预热/验证预算是两本账，互不代替。原动作与理由：" + reason
                    )
            # 🔑🔑 [2026-09-17，用户拍板] **停的是谁，必须说清楚。**
            # 终态出口没有 scope 就抛 —— 静默按某个默认处理的两个方向都错：
            # 按局部处理会绕过一条已知不成立的路径继续烧 GPU；按全局处理会停掉
            # 一个仍有活干的 stage（`D3_REWINDOW_DEPTH_EXHAUSTED` 就是这么漏的）。
            _spec = EXIT_SPECS.get(str(exit_)) if exit_ else None
            if halt_scope is None and _spec is not None:
                halt_scope = _spec.default_scope
            if halt_scope is not None and halt_scope not in HALT_SCOPES:
                raise ValueError(
                    f"未知 halt_scope {halt_scope!r}；只接受 {HALT_SCOPES}")
            if exit_ and _spec is None:
                raise ValueError(
                    f"出口 {exit_!r} 不在唯一注册表 `EXIT_SPECS` 里 ⟹ 它的 terminal /"
                    " scope 语义没有任何地方声明过。先去那张表里加一条。")
            if _spec is not None and _spec.terminal and halt_scope is None:
                raise ValueError(
                    f"终态出口 {exit_!r} 缺 `halt_scope`：它在 `EXIT_SPECS` 里没有"
                    f"默认 scope（{_spec.why}），发出点必须显式传 "
                    "`halt_scope=\"TARGET_LOCAL\"` 或 `\"STAGE_GLOBAL\"` 或 "
                    "`\"PATH_INVALID\"`。漏传不再静默停整跑。")
            return {
                "protocol_version": STAGE2_CONTROLLER_PROTOCOL_VERSION,
                "action": action,
                "exit": exit_,
                # 停的是谁（`None` = 不是终态，没有"停"这回事）
                "halt_scope": halt_scope,
                "blocking_window": (
                    int(blocking_window) if blocking_window is not None
                    else (int(earliest) if earliest is not None else None)),
                "blocking_unit": (
                    str(blocking_unit) if blocking_unit is not None
                    else (str(unit_id) if unit_id else None)),
                "reason": reason,
                "windows": windows or [],
                # 🔑 [2026-09-14] **采样单元 ID。** 物理窗口不是唯一可调度的采样
                # 单元：immutable rewindow 的子窗有自己的目录、自己的 checkpoint、
                # 自己的冻结 f_k。给子窗补帧必须点名 `unit_id`，执行器据此续跑
                # **那一个**子窗的 checkpoint，而不是回头动父窗。
                # 普通窗口的动作不带它（None）。
                "unit_id": unit_id,
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
                # [2026-09-16] 本轮路由锁定的那个窗口。`blocked_by` 不能代替它：
                # 那个键在"没有下游被挡住"时是 None，而退役判定需要知道**这一轮
                # 到底在修谁**（见 `_retirable_window`）。
                "earliest_unresolved_window": (
                    int(earliest) if earliest is not None else None
                ),
                "terminal": bool(exit_ in self.TERMINAL_EXITS),
                "routing": bool(exit_ is not None and exit_ not in self.TERMINAL_EXITS),
                # ⚠️ **只是提示，不是动作。** 影子模式报"按 warmup 剖面谁最弱"，
                # 好让"提前配预算"这件事先被看见；**不给阈值、不给倍数**，
                # 剖面→预算的映射是未定项（PLAN P3）。
                "warmup_weakness_ranking": self._warmup_weakness_ranking(view),
                # 🔑🔑 [2026-09-15] **逐段质量门：报告，不是动作依据。**
                # 每条带 `drives_action: False`。`decide()` 的任何分支都**不**按
                # 它分岔（老板定案：这些未经标定的阈值不得再驱动自动补帧）。
                # 保留它是因为「实测值 / 阈值 / 最差窗口」对人工审查确实有用。
                "stage_quality_gates": stage_quality_gate_failures(
                    self._read_stage_result()),
                # 新契约的两维，原样透出（`converged` 已删除，不做兼容别名）。
                "stage_analysis_status": view.get("stage_analysis_status"),
                "stage_analysis_incomplete_reasons":
                    view.get("stage_analysis_incomplete_reasons") or [],
                "stage_precision_status": view.get("stage_precision_status"),
                "stage_precision_evidence": view.get("stage_precision_evidence") or {},
                # 🔑🔑 [2026-09-17] **被上交的提问必须有回应，哪怕答案是"押后"。**
                # `LOCAL_VALIDATION_CAP` 是引擎明确交给控制器的决定（见
                # `local_validation_cap_hits` 的长注释）。分支 3a 有答案，但它走
                # `_pick()` ⟹ 只对 `earliest` 生效 ⟹ 提问窗口不是 earliest 时
                # **整条分支不触发、提问被静默丢掉**。
                # 这里恒常落一份：本轮回答了谁、押后了谁、押后的理由是什么。
                # ⚠️ 这里**不改路由**：押后的理由是真的（上游可能重锚，那会让这一块
                # 取证落在即将作废的布局上 —— 分支 3a 自己的注释就警告了这一点）。
                # 押后不会变成永远：`earliest` 无路可走时会被退役，届时提问窗口
                # 自己成为 `earliest`，分支 3a 就触发了。
                "local_validation_cap_pending": [
                    {**h,
                     "answered_this_round": bool(
                         action == "PROVISIONAL_PRODUCTION"
                         and int(h["window_idx"]) in [int(x) for x in (windows or [])]),
                     "deferred_because": (
                         None if (action == "PROVISIONAL_PRODUCTION"
                                  and int(h["window_idx"]) in
                                  [int(x) for x in (windows or [])])
                         else (
                             f"本轮路由锁定在更早的未解决窗口 {earliest}"
                             "（上游重锚会作废下游 lineage，所以先修最早的）；"
                             "在它解决或被退役之前，对本窗口取证有落在"
                             "**即将作废的布局**上的风险 ⟹ 押后，不是拒绝。"
                             if earliest is not None and int(earliest) != int(h["window_idx"])
                             else "本轮的动作不是 PROVISIONAL_PRODUCTION（见 reason）。"
                         )),
                     }
                    for h in self.local_validation_cap_hits(view)
                ],
                # 🔑🔑 [2026-09-17] **路由态 ELIGIBLE 的窗口带着信号 ⟹ 永远轮不到。**
                # 见 `unrouted_window_signals` 的长注释：15 个 `_pick` 调用点里
                # 13 条押后无害（窗口会自己成为 earliest），2 条会**永久丢失**。
                # 这里恒常把它们摆出来 —— **不改路由，只保证不静默**。
                "unrouted_window_signals": self.unrouted_window_signals(view),
                # 出口专属的结构化诊断（目前只有 D3 用）。**不是**通用逃生舱：
                # 加新键之前先问"它能不能进上面某个既有字段"。
                **(dict(extra) if extra else {}),
                # 🔑 [2026-09-17，死线 D1] **终态窗口的失败信息提到 run 级。**
                # 它们原来只出现在 D1 那条分支的 `reason` 字符串里 ⟹ 换一条出口
                # （退役换窗、或 TERMINAL 窗口不是 `earliest`）就整个看不见了，
                # 而这是人工接手唯一需要的东西。D1 本身**不是控制器能修的死线**
                # （要人改输入 λ 表 / 升档预热预算 / 接受失败），所以对它能做的
                # 全部改进就是**报清楚**。与 `earliest` 是谁无关，恒常输出。
                "terminal_window_failures": [
                    {"window_idx": int(w["window_idx"]),
                     "bias_status": w.get("bias_status"),
                     "verdict": w.get("verdict"),
                     "last_failure_reason": w.get("last_failure_reason"),
                     "last_gate_error": w.get("last_gate_error")}
                    for w in (view.get("windows") or [])
                    if w.get("phase") == "TERMINAL"
                ],
            }

        # 0) 读不到任何窗口 —— 还没开跑，或者路径/目录给错了。别猜。
        if not W:
            return plan(
                "CONTINUE_WARMUP",
                f"{self.stage_dir} 下没有任何窗口产物 —— stage 还没开始，或 "
                "run_dir/stage_name 给错了。",
                missing=["任一窗口的 convergence/warmup_failure 产物"],
            )

        # 0a) 🔑🔑 [CTL-11，2026-09-14] **stage 级判据已经通过 ⟹ DONE，不许再修。**
        #
        # 先前 `DONE` 只在分支 7 判，排在「最早未解决窗口」路由**之后**；而
        # `earliest` 来自逐窗**自检**的 `self_verdict`。两者是同一个量的两份实现：
        #   · stage 级 `converged` 由 `solve_stage_integrated` 在**合并后的全部段**
        #     的帧上算（五条合取），是权威；
        #   · 自检 `min N_eff/g` 只看该窗口**一个段**的帧，对多段窗口系统性偏悲观
        #     （同 DECORR-01：自检是早期诊断，不是求解器资格）。
        # ⟹ 任一窗口自检 < 10 就永远轮不到分支 7。真机 cyclod_ligand2/rep2：
        # stage `converged=True`、无缺窗无跳窗，而 win0/win3 自检 4.37/4.81，
        # `decide()` 给出的是 `PROBE_REANCHOR_EPOCH` —— **一个已经跑完的 stage，
        # 每次 resume 都会重开一个 Epoch 烧 GPU。**
        #
        # 缺窗 / 跳窗仍然一票否决（缺窗口的和不是 ΔG，是另一个量），所以这里
        # 显式再查一遍，不只信 `converged` 一个布尔。
        # ⚠️⚠️ [2026-09-14 自查，这一条是本次改动自己引入的 regression 的补丁]
        # 把 DONE 提到路由之前，就**同时拆掉了原来由路由提供的保护**：
        # 分支 7 在 `1d-0`（产物与布局对不上）/ `6`（跳窗）/ `6b`（缺窗）之后，
        # 那几条会先拦住一个"布局已经变了、`converged` 却是旧布局结论"的盘面。
        # 而 `_read_stage_result()` 对**不带 `path_version` 的老产物**是放行的，
        # stage 缓存 `stage2_vanishing.json` 恰恰不盖版本号 ⟹
        #   布局 v1 收敛 → 后来插 λ 变 v2 → 中间产物因版本不符被跳过
        #   → 老缓存被放行 → 这里直接 DONE。**发布一个描述另一条布局的 ΔG。**
        # 所以提前判 DONE 必须额外要求：① 结果的版本**核实过**；
        # ② 没有任何窗口的证据被布局变更作废。两条都不满足就老老实实往下走路由，
        # 由分支 7 在路由之后兜底（那条路径上保护仍然成立）。
        # [2026-09-14] 版本前置抽成 `_layout_trustworthy()`，与 `_evidence_status`
        # 共用一份 —— 两处不同步正是 `DONE` + `INCONCLUSIVE` 那条自相矛盾的成因。
        # `skipped_windows` 现在只含**物理窗口**（审计 #58）：被跳的子窗不再以
        # 一个不存在的窗口号（10000+）永久封死 DONE；它们由 1c/1c-2 单独路由。
        # 🔑🔑 [2026-09-15 老板定案] **`ANALYSIS_COMPLETE` ≠ 可以判 `DONE`。**
        # 结构与身份检查通过只记「分析完整」；跨重复精度记 `UNMEASURED`，
        # **不得作为已验收结果发布**。所以这里分两个出口：
        #   · `precision_status == MEETS_CROSS_REPEAT_TARGET` ⟹ `DONE`
        #     （单次 run 里**永远不成立** —— 一个 run 无法自证精度，有意如此）；
        #   · 否则 ⟹ `NO_ACTION` + `ANALYSIS_COMPLETE_PRECISION_UNMEASURED`，
        #     **终态**（该停就停、不继续烧 GPU），但**不是 `DONE`**。
        # 🔑🔑 [2026-09-17] **已承诺的生产步数没跑完，不许判终态。**
        # 实测：全窗自检合格时 `earliest is None` ⟹ `_pick()` 恒返空 ⟹ 补足生产那条
        # 分支一次都不触发 ⟹ 一个跑了 10 万 / 目标 25 万步的窗口跟着整条 stage 一起
        # 被判 `terminal=True`。这里只**挡住终态**；真正的补帧由下面分支 5
        # （`short`）发，它在 `_frames_admission` 定义之后，会照常过块数上限与射程闸。
        # ⚠️ 补的是**已承诺的配额**（`n_steps_per_window_effective`，调用方配的目标），
        # 不是任何质量门 —— 与「未经标定的阈值不驱动补帧」不冲突：那条禁的是拿阈值
        # 要更多帧，这条只是把说好要跑的跑完。
        _below_target = [
            int(w["window_idx"]) for w in W
            if w.get("production_steps_target")
            and (w.get("production_steps") or 0) < int(w["production_steps_target"])
        ]

        if (view.get("stage_analysis_status") == "ANALYSIS_COMPLETE"
                and self._layout_trustworthy(view)
                and not view.get("stale_layout_evidence")
                and not self._coverage_incomplete(view)
                and not _below_target):          # [2026-09-17] 见上
            _common = (
                "stage 分析的**硬不变量**已通过（求解跑完、无缺窗、无跳窗、无跳子窗、"
                "布局版本核实过）⟹ `analysis_status = ANALYSIS_COMPLETE`。"
                "⚠️ 逐窗自检的 `min N_eff/g` 只看单段帧、对多段窗口系统性偏悲观，"
                "**它不是权威**，不得用它挡住一个已跑完的 stage —— 否则每次 resume "
                "都会在一个跑完的 stage 上重开修复动作（CTL-11）。"
            )
            if view.get("stage_precision_status") == "MEETS_CROSS_REPEAT_TARGET":
                return plan(
                    "DONE",
                    _common + "且 `precision_status = MEETS_CROSS_REPEAT_TARGET`"
                    "（跨重复精度已达标）⟹ 完成。"
                    "⚠️ 这仍不等于答案正确 —— STAGE2_ROOT_CAUSE_2026-08-28.md §2"
                    "「所有收敛门对该失效模式失明」仍然有效。",
                    exit_="DONE_UNTRUSTED" if self.allow_untrusted else "DONE",
                )
            return plan(
                "NO_ACTION",
                _common + f"但 `precision_status = "
                f"{view.get('stage_precision_status')!r}` —— **精度从未被测过**。"
                "跨重复精度只能由多次独立重复给出，**单次 run 无法自证**，"
                "所以这里不是 `DONE`，而是「分析完整 / 精度未测」这个终态："
                "这一跑没有别的事可做（继续补帧只会烧 GPU 而不提高可验证性），"
                "但结果**不得作为已验收结果发布**。"
                "逐段质量门的读数仍在 `stage_quality_gates` 里供人审查 —— "
                "它们是**未经标定**的阈值，已被明令不得驱动自动补帧。",
                exit_="ANALYSIS_COMPLETE_PRECISION_UNMEASURED",
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

        _window_state = self._window_routing_state
        _states = {int(x["window_idx"]): _window_state(x) for x in _order}
        # 🔑🔑 [CTL-03，2026-09-14] **被 rewindow 取代的父窗不参与 earliest 排序。**
        # 它已经退出求解覆盖（子系综接管了它那段 λ），它的旧 warmup / f_k 状态
        # 只是历史记录。先前它照样能当 `earliest`，于是分支 1e（父窗预热预算耗尽
        # ⟹ 发 INSERT_LAMBDA / NO_ACTION）会排在未完成子窗的路由**之前**，
        # 用一个已经不在覆盖里的窗口的旧状态，挡住子窗自己的补帧。
        _replaced_parents = {
            int(x) for x in
            ((view.get("immutable_rewindow") or {}).get("parents_done") or [])
        }
        # 🔑🔑 [2026-09-15 真机 brd4_ligand1/rep2] **被求解器跳掉的窗口优先于
        # "自检说支撑不足"的窗口。**
        #
        # 两者不是同一强度的信号：
        #   · `skipped_windows` 是**求解器的操作权威** —— 那个窗口真的没进 MBAR，
        #     整条路径因此**缺窗**，`analysis_status` 判 `ANALYSIS_INCOMPLETE`，
        #     交出去的和**不是 ΔG**（硬不变量，本仓不许放宽）；
        #   · `self_verdict` 只是逐窗自检的诊断，说的是"这个窗还没测够"。
        #
        # 先前 `earliest` 只按下标排序、两者一视同仁 ⟹ 一个**加帧治不好**的窗口
        # 会把真正让整条路径不成立的那个无限期挡在后面。真机：
        #     win1 self_verdict=INSUFFICIENT_DATA（n_decorr=888 够得离谱、
        #          min N_eff/g=8.68 ⟹ 是**偏斜**，同分布加帧治不了）
        #     win4 HARD_INSUFFICIENT，n_decorr=7 ⟹ 被求解器跳掉，skipped_windows=[4]
        # 6 轮全部路由到 win1（4 块帧烧光配额），win4 一次都没被看过；
        # 整条路径照常采完，最后死在缺窗那道身份门上。
        # 而 win4 恰恰有现成的对症动作（`INSERT_LAMBDA` 缩跨度），只是轮不到。
        #
        # ⚠️ 这**不放宽任何判据**：只改"先修哪一个"的顺序。被跳的窗口仍然要过
        # 同一套可行性与预算闸；自检类窗口一个都没被跳过，只是排在后面。
        # ⚠️ `_replaced_parents` 的排除照旧（父窗已退出求解覆盖，修它没有意义）。
        _skipped_now = {
            int(x if not isinstance(x, dict) else x.get("window_index", -1))
            for x in (view.get("skipped_windows") or [])
        } - {-1}
        # 🔑🔑 [2026-09-16] `_retired` = **本轮已经判定"对它没有任何动作可做"的窗口**。
        # 它们退出 `earliest` 排序（因此也不再挡住下游），但**不被当成合格**：
        # 窗口状态、`analysis_status`、质量门一个都不放宽。语义只有一句：
        # 「这个窗口修不动了，别再让它把还能修的窗口一起拖死。」
        # 谁进这个集合由 `decide()` 那层决定（只有终态 + 别处确实还有预算时才退役）。
        _retired = {int(x) for x in (retired or ())}
        earliest = next(
            (i for i in sorted(_states)
             if i in _skipped_now and i not in _replaced_parents
             and i not in _retired),
            None,
        )
        if earliest is None:
            earliest = next(
                (i for i in sorted(_states)
                 if _states[i] == "PROBLEM" and i not in _replaced_parents
                 and i not in _retired),
                None,
            )
        unknown = [i for i in sorted(_states) if _states[i] == "UNKNOWN"]
        # 🔑 [2026-09-17] **已退役的窗口不算"被上游挡住"。**
        # `blocked_by_upstream` 的语义是「它们的证据前提可能被 `earliest` 的重锚
        # 作废，所以先别动」。而**已退役**的窗口恰恰是「本轮已判定对它没有任何
        # 动作可做」——它不在等 `earliest`，它是被绕过去的那个。两件事混在一张
        # 表里会让人读成"下游还有 N 个在排队"，而其中几个其实已经放弃了。
        # 纯报告字段，不改任何路由。
        blocked = (
            [int(x["window_idx"]) for x in _order
             if int(x["window_idx"]) > earliest
             and int(x["window_idx"]) not in _retired]
            if earliest is not None else []
        )
        # 本轮路由上下文就位 ⟹ `plan()` 从这里取默认值（见 `_ctx` 的长注释）。
        _ctx.update(earliest=earliest, blocked=blocked)
        # 🔑🔑 [REWIND-01，2026-09-17] **λ 表上的两个动作都不可行 ≠ 没有对症动作。**
        #
        # `SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA` 是仅有的两个缩跨度动作，而它们都锚在
        # 尾部：拆末窗要 tail anchor（`first_untrusted_window` 在 `idx <= 0` 时恒为
        # None），插 λ 在末窗顶到 `hi` 且没有 anchor 时也不可行，再加一个跨 resume 的
        # `max_path_insertions`。**失败窗口是 window 0 时两个恒不可行** —— 而 window 0
        # （解耦端点）正是本方法的必然坏窗口。于是四条独立的分支（1e 预热预算耗尽、
        # 5a-2 累计 f_k + held-out REJECT、边际增益停滞、5b 偏斜类自检失败）全都汇到
        # 同一句 `NO_FEASIBLE_ACTION`：控制器**有**对症归因，却没有一个够得着的动作。
        #
        # `IMMUTABLE_REWINDOW` 恰好绕开全部四个封锁条件（见 `rewindow_feasible`），
        # 成本表 / 执行器 / 子窗调度 / no-op 台账早就齐了，唯独缺这一个发出点
        # （`docs/TODO.md` 的 REWIND-01：13 个声明动作里唯一发不出的那个）。
        # ⚠️ 接在**每条死线各自的兜底之前**，不是在链尾加第 33 条分支 ——
        # 否则又是「按书写顺序决定语义」。
        def _bounded_rewindow(win, why):
            """缩跨度都不可行时退到有界重窗。不可行返回 None，由原来的兜底接手。"""
            if self.rewindow_feasible(view, win) is not None:
                return None
            _c = self.rewindow_children(view, int(win))
            return plan(
                "IMMUTABLE_REWINDOW",
                why
                + f"⟹ 退到**固定 λ 表上的有界重窗**：在窗口 {int(win)} 内按中点建 "
                f"{len(_c)} 个重叠子系综 {_c}，各自预热、各自锁一份 f_k。"
                "它**不动 λ 表**（不需要 tail anchor、不吃插点预算、末窗一个态都不涨）"
                "⟹ 上面那些封锁条件一个都不碰；子系综是**新系综**、有自己的预热预算"
                "⟹ 也绕开「重进父窗预热被弹回」。原数据一个字节不改"
                "（独立目录 `<stage>_rewindow_<identity>/`）。"
                "**一个父窗只切一层**：子窗再失败不再递归。",
                windows=[int(win)], blocked=blocked, earliest=earliest,
            )

        # 🗑️ [2026-09-15] 这里原有 `_sidx_to_unit` / `_unit_by_id` / `_route_target`
        # 三样，**唯一**的用途是把 stage 质量门报的 worst_window（solver 索引）翻成
        # 调度身份，好让分支 9b/9c 据此发补帧 / 缩跨度动作。那两个分支已删除
        # （未经标定的阈值不得驱动自动补帧），这三样随之无人使用 ⟹ 一并删掉，
        # 不留"以后也许用得上"的死代码。子窗的路由仍由分支 1c 自己做。

        # 🔑🔑 [2026-09-14] **补帧准入：批下一块之前，先看上一块换来了什么。**
        #
        # 真机可以一路 25→50→…→150 万步：早期分支（求解器跳窗 ⟹ 补帧）走不到
        # 后面的边际收益刹车，而停滞保护把"多跑了一块"算成盘面进展、把重复计数
        # 清零。**"又产生了帧"不等于"获得了有用证据"。**
        #
        # 两条准入，缺一不可：
        #   ① **块数硬上限**（每单元，config 可调，缺省 4 块）——
        #      生产 cap 缺省是 unknown，拦不住，必须有这道；
        #   ② **上一块必须有实质增益** —— 判据量是**求解器侧**的去相关帧数
        #      （不是自检那份，两者差 2–8 倍；也不是步数）。没涨就停，并说清楚。
        _blocks = view.get("production_blocks_by_window") or {}
        # 🔑 [BUD-03] 硬上限数的是**跨全部采样段**的块（换段不重置配额），
        # 边际增益仍然只在**同段**内比（跨 f_k epoch 比数字没有意义，设计 §6.10）。
        # 先前两者共用同段那本 ⟹ 换一次 Epoch 配额清零，实测单窗烧到 150 万步。
        _blocks_total = view.get("production_blocks_total_by_window") or _blocks
        # 🔑🔑 [契约 B] **子窗（sampling unit）有自己的两本账。**
        # 先前块账只扫 `view["windows"]`，而 `sampling_units` 一个都不在里面，
        # `plan()` 又拿**父窗**的账去问准入（父窗步数已冻结 ⟹ 去重后恒 1 行）⟹
        # 子窗的块数硬上限与边际刹车**双双失效**（BUD-03 修过的「单窗烧 150 万步」
        # 在子窗这个新位置原样复发）。
        _blocks_u = view.get("production_blocks_by_unit") or {}
        _blocks_total_u = view.get("production_blocks_total_by_unit") or _blocks_u
        _max_blocks = int(view.get("max_production_blocks_per_window") or 4)

        def _frames_admission(w_idx, unit_id=None):
            """能不能再批一块帧。返回 `(True, None)` 或 `(False, 原因)`。

            带 `unit_id` ⟹ 查**子窗那本账**（父窗的账描述的不是它）。
            """
            if unit_id:
                _who = f"子窗 `{unit_id}`"
                rows = _blocks_u.get(str(unit_id)) or []
                rows_all = _blocks_total_u.get(str(unit_id)) or []
                _dim = "子系综 identity"
            else:
                _who = f"窗口 {int(w_idx)}"
                rows = _blocks.get(int(w_idx)) or []
                rows_all = _blocks_total.get(int(w_idx)) or []
                _dim = "段"
            if len(rows_all) >= _max_blocks:
                return False, (
                    f"{_who}在当前布局下**跨全部采样段**已经批过 {len(rows_all)} 块帧"
                    f"（上限 {_max_blocks}；换段/换 Epoch 不重置配额）⟹ **不再自动补**。"
                    f"逐块判据量（{_dim}, 求解器去相关帧数）="
                    f"{[(r.get('segment'), r.get('solver_n_decorrelated')) for r in rows_all]}。"
                )
            # [审计 #47] 边际增益刹车改用共享的 `marginal_gain_stalled()`：
            # 至少 3 个点、基准取前面各点的中位数、只有明显低于基准才算被证伪。
            # 旧判据 `max(fin[1:]) <= fin[0]` 把 `solver_n_decorrelated` 当成随采样
            # 单调增的量（分子 Kish ESS 小 N 偏高、分母 ĝ 在 N≫τ 前还在涨，两个偏差
            # 同向），而基准恰好取偏高最严重的 `fin[0]` ⟹ 对正常窗口误触发。
            _stalled, _diag = marginal_gain_stalled(
                [r.get("solver_n_decorrelated") for r in rows])
            if _stalled:
                return False, (
                    f"{_who}的补帧**没有带来实质增益**：求解器侧去相关帧数逐块 "
                    f"{_diag['series']}，基准（除末点外各点的中位数）="
                    f"{_diag['baseline_median_of_earlier_points']:.1f}，"
                    f"末点 {_diag['last']:.1f}（比值 "
                    f"{_diag['last_over_baseline']:.2f} < 刹车阈值 "
                    f"{_diag['stall_ratio']}）。"
                    f"同期 min N_eff/g 逐块="
                    f"{[r.get('min_n_eff_over_g') for r in rows]}"
                    "（去相关帧数是 N/ĝ 那一族：它不涨可能是分子没涨，也可能是分母 ĝ "
                    "还在随样本上涨 —— 两种都说明同分布再加一块治不了）⟹ "
                    "**停止自动补帧**。"
                )
            # 🔑🔑 [2026-09-17，DEAD_LINES §3.1+§3.2] **第三道：剩余配额的乐观上界
            # 还够不够得着门。**
            #
            # `n_eff_over_g_reachable_by_frames` 的 docstring 自己写着它是**上界**、
            # 「`True` 只表示值得一试」，并且「真正的刹车是事后的
            # `marginal_gain_stalled()` —— 两者一前一后，**缺一不可**」。
            # 而那个刹车的判据是 `末点 < 前面各点中位数 × 0.9`：**单调上升但渐近在
            # 门以下的序列永不触发**。真机 cyclod_ligand1_outer/rep1 win0 逐块
            # `4.459 / 4.553 / 5.287`（门 10）⟹ 判「还在涨」、刹车不响，按 +0.41/块
            # 要 11 块才够。⟹ **这一对少了一半**：进的时候乐观，出的时候没人管。
            #
            # 补的这一道**不引入任何新阈值、也不拟合斜率**（那个量单次噪声 34×，
            # 3 个点拟斜率同样在量噪声），也**不做 `÷1.44` 那种"已观测 g 惩罚"**
            # （`R_now` 本来就是用当前 g 算的 ⟹ 重复扣除；拿过去的比例用于未来
            # ⟹ 那仍是外推，只是不叫"斜率"）—— 它就是**同一个前向判据**，只是拿
            # **现在的读数**和**剩下的配额**再问一次：
            #     当前 ratio × 剩余 headroom < 门  ⟹ 连最乐观的情形都够不着 ⟹ 停。
            # `frames_growth_headroom` 本来就是 `(1+cap)/(1+used)`，用一块少一块，
            # 所以"剩余"是它的原义，不用新写。
            # 上面那三个真机读数代进去：used=1 ⟹ 4.459×2.5=11.1 ≥ 10 放行（此刻确实
            # 还可能够得着）；used=2 ⟹ 4.553×1.67=7.6 < 10 ⟹ **停**。少烧至少一块。
            #
            # ⚠️ **只在明确判 False 时拦**。`None` 是「判不了」（缺数 / 门缺失 /
            # 非有限），一律放行 —— 未知不是"够不着"，这与本文件别处的口径一致。
            # ⚠️ 归因用的判据与 `support_failure_is_skew` **同一个函数**，所以
            # 「加帧够不够得着」在准入侧与归因侧不会给出两个答案。
            _rec = None
            if unit_id:
                _rec = next((u for u in (view.get("sampling_units") or [])
                             if str(u.get("unit_id")) == str(unit_id)), None)
            else:
                _rec = next((w for w in (view.get("windows") or [])
                             if int(w["window_idx"]) == int(w_idx)), None)
            # ⚠️⚠️ [2026-09-17] **帧数没到下限时，射程闸不适用。**
            # 射程算的是 `N_eff/g` 这道**后一道**判据；而一个去相关帧数还没到
            # `min_frames` 下限的窗口，连被求解器尝试的资格都没有 —— 它那个 ratio
            # 是从 7 帧算出来的，**本来就没有意义**，拿它否决补帧等于用一个不可信
            # 的读数掐掉唯一的出路。实测：被求解器跳掉的 w3（7 帧 < 10、归因
            # `SAMPLE_SIZE` 硬证据）被这道闸拒绝补帧 ⟹ 退役 ⟹ 路由跑去修一个
            # 下标更小、却没被跳的窗口，正是「跳窗优先」这条要防的反面。
            if (_rec is not None
                    and support_failure_attribution(
                        _rec.get("self_verdict"), _rec.get("self_verdict_source"),
                        n_decorrelated=_rec.get("self_n_frames_decorrelated"),
                        min_frames=_rec.get("self_min_frames"),
                    ) == SUPPORT_FAILURE_SAMPLE_SIZE):
                return True, None
            if _rec is not None:
                _hr = frames_growth_headroom(
                    view, None if unit_id else int(w_idx),
                    unit_id=str(unit_id) if unit_id else None)
                _ratio = _rec.get("min_n_eff_over_g")
                # ⚠️ 这里原来还有一个 `or _rec.get("n_eff_over_g_target")` 候补别名，
                # 而那个键**窗口侧与子窗侧都没有任何人写**（唯一写侧的键名是
                # `self_n_eff_over_g_eligible`）⟹ 纯死分支，删掉，别留着假装有兜底。
                _target = _rec.get("self_n_eff_over_g_eligible")
                if n_eff_over_g_reachable_by_frames(
                        _ratio, _target, _hr) is False:
                    return False, (
                        f"{_who}把**剩余补帧配额全花掉**也够不着门："
                        f"当前 min N_eff/g={_ratio}，剩余帧数倍率={_hr:.3g}"
                        f"（已批 {len(rows_all)}/{_max_blocks} 块），"
                        f"最乐观射程 {float(_ratio) * float(_hr):.3g} < 门 {_target}。"
                        "⚠️ 这**不是「证明够不着」**，是「连一个**刻意乐观**的估计都"
                        "够不着」：真实关系是 "
                        "`R_future = R_now × H × (η_future/η_now) × (g_now/g_future)`"
                        "（`η = N_eff/N`），而这里只留 `R_now × H`、**假定 η 与 g 恒定** —— "
                        "两个假设实测都朝不利方向走（g 实测 12.25→17.63；η 见 §5.1："
                        "250k→1M 让 top1% 0.545→0.762）⟹ 它是**上界**。"
                        "上界都够不着就不是「再试一次」的事。"
                        "对症动作是缩跨度（插 λ / 有界重窗），不是更多帧。"
                    )
            return True, None

        _noop_led = view.get("noop_actions") or {}
        _by_idx_for_noop = {int(w["window_idx"]): w for w in W}
        _by_uid_for_noop = {str(u["unit_id"]): u
                            for u in (view.get("sampling_units") or [])}

        def _is_noop(action: str, w_idx, unit_id=None) -> bool:
            """这个动作在**当前盘面**对这个窗口/子窗已经被证明什么也不做。

            执行器写的账（`stage2_noop_actions.json`），指纹由
            `action_noop_fingerprint()` 算 —— 两侧同一份实现。补过帧/换过段
            之后指纹就变了，这条记录自动失效、动作重新可选。

            🔑🔑 [审计 #9，契约 A] **key 有两套命名空间，必须都查。**
            执行器给子窗动作写的是 `f"{action}:unit:{unit_id}"`，而这里先前**只查**
            `f"{action}:{idx}"` ⟹ 子窗的 no-op 账**全仓无人读**，写了等于没写；
            反过来执行器的通用 no-op 检测只要 `plan["unit_id"]` 非空就一律走 unit
            分支 ⟹ 那一轮连普通 key 也不写。两边一起漏 = 子窗的 no-op 刹车从来
            没生效过。带 `unit_id` 时**优先查 unit key，并用该 unit 自己的记录算指纹**
            （父窗的步数是冻结的，拿它算等于把"补过帧"这一维关掉）。
            """
            # 🔑 [2026-09-15] `action_noop_fingerprint` 返回 None = **这个盘面
            # 没有可比身份**（生产步数读不到）。None 不匹配任何东西 —— 包括
            # 台账里那条同样是 None/`?` 的旧记录。否则「从未生产过」的窗口
            # 指纹恒定，no-op 记录在结构上永不失效（真机 win4 死锁的第二环）。
            def _match(rec, record) -> bool:
                if not isinstance(rec, dict):
                    return False
                fp = action_noop_fingerprint(record, view.get("path_version"))
                return fp is not None and rec.get("fingerprint") == fp

            if unit_id:
                return _match(_noop_led.get(f"{action}:unit:{unit_id}"),
                              _by_uid_for_noop.get(str(unit_id)))
            return _match(_noop_led.get(f"{action}:{int(w_idx)}"),
                          _by_idx_for_noop.get(int(w_idx)))

        def _pick(items, key: Optional[str] = "window_idx"):
            """**只路由 earliest 这一个窗口**：它不在本分支的候选里就不触发。

            ⚠️ [审计 #22，2026-09-14] **这里刻意不放宽。** `earliest is None` 时
            返空看起来把一批分支（1b/2/3/3a/4/5/0b/5a-*/5b/6）都打掉了，但那些分支
            **没有一条写过自己「全窗合格时我不该跑」的前提** —— 它们靠的就是这个
            返空。全局放宽等于一次性拆掉 11 条分支的隐式前提，而它们各自该不该在
            「全窗合格」盘面上触发，没有任何一处有成文依据。
            #22 的真实失效面（全窗 UNKNOWN + 生产帧没攒够 ⟹ 只发得出空转的
            `ANALYZE`）改在**分支 5 里定点修**，理由写在那里。
            """
            if earliest is None:
                return []
            idxs = [int(i) if key is None else int(i[key]) for i in items]
            return [earliest] if earliest in idxs else []

        # 0b) **身份不一致**：盘上的 converged 是另一个系综的结论，不得当依据。
        # 🔑🔑 [2026-09-14 第三轮复核] **这一条以前物理上排在分支 5 之后。**
        # 编号写着 `0b`、意图是"最早"，位置却在 `5)（生产帧没攒够）` 后面 ——
        # 而 `IDENTITY_MISMATCH` 的窗口 `production_steps` 往往就是没到目标，
        # 于是分支 5 先命中、对着一个**另一个系综**的产物发 RUN_PRODUCTION，
        # `HALT_INVALID_INPUT` 永远轮不到。身份不合法时任何动作都不该发，
        # 所以它必须排在全部路由之前。
        _stale_id = _pick([w for w in W if w.get("phase") == "IDENTITY_MISMATCH"])
        if _stale_id:
            return plan(
                "CONTINUE_WARMUP",
                f"窗口 {_stale_id} 的 `stage_protocol_key` 在 ibs_state 与产物之间不一致 ⟹ "
                "盘上那个 `bias_status=converged` 是**另一个系综**的旧结论，"
                "不得据此声称已在生产。λ / Hamiltonian / box / 规范版本任一变化，"
                "旧 PASS 都不能继续粘住。",
                exit_="HALT_INVALID_INPUT", windows=_stale_id,
                blocked=blocked, earliest=earliest,
            )

        # 0c) 🔑🔑 [审计 #23，2026-09-14] **`phase == "TERMINAL"` 全仓没有任何分支处理。**
        #
        # `bias_status ∈ {failed, calibrated_validation_failed}` 映成 `TERMINAL`，
        # `_window_state()` 判它 PROBLEM ⟹ 它**当上 `earliest`** 并挡住所有下游窗口；
        # 而后面每一条分支的候选集合（warming / short / skipped / …）都不含它 ——
        # `_pick` 于是对每一条都返回 []，最后掉到 9d 以一个**与根因完全无关**的理由
        # （"归因不到任何一道具体的门"）退出。真正的根因（这个窗口的 f_k 已经被判死）
        # 一个字都没出现在结局里。
        # 这里显式分流：f_k 被驳回走已有的 refuted 路径（分支 2），其余情形如实发
        # `NO_ACTION` + `NO_FEASIBLE_ACTION`，并把 `last_failure_reason` /
        # `last_gate_error` 原样带出去 —— 它们就是人工接手所需要的全部信息。
        if earliest is not None and int(earliest) not in _replaced_parents:
            _tw = next((w for w in W if int(w["window_idx"]) == int(earliest)), {}) or {}
            if _tw.get("phase") == "TERMINAL":
                if _tw.get("verdict") == "STATISTICALLY_REJECTED":
                    pass          # 有统计功效的驳回 ⟹ 交给分支 2 的既有路径
                else:
                    return plan(
                        "NO_ACTION",
                        f"窗口 {earliest} 的 `bias_status` 是终态"
                        f"（{_tw.get('bias_status')!r} ⟹ phase=TERMINAL）：这份 f_k 的"
                        "预热/验证**已经判死**，不是「尚不可测」。"
                        f"失败原因 `{_tw.get('last_failure_reason')}`，"
                        f"最后一次门错误 `{_tw.get('last_gate_error')}`。"
                        "控制器没有能推动它的动作 —— 重解/换 Epoch 都要先重新进入这个"
                        "窗口，而它的预热在入口就会被弹回；加帧同理。"
                        "要继续需要人工判断（改输入 λ 表 / 显式升档预热预算 / "
                        "接受这个窗口失败）。**不是 DONE** —— 结果没有达标。",
                        exit_="NO_FEASIBLE_ACTION", windows=[int(earliest)],
                        blocked=blocked, earliest=earliest,
                        # [2026-09-17] phase=TERMINAL：这份 f_k 已判死 ⟹ 证据不成立，**不得**绕过去采下游
                        halt_scope="PATH_INVALID",
                    )

        # 1d-0) 🔑🔑 [2026-09-14 真机] **产物与当前布局对不上的窗口，只能重采。**
        #
        # 真机 `cyclod_ligand1/rep2` 17:57 崩在这上面：
        #     ValueError('窗口 3 状态数与 window_ranges 不符')  ← 炸出流水线
        # 实测盘面：路径已演化到 v2（win3 从 6 态涨到 7 态），`ibs_state` 里的
        # `lambdas_vdw` / `f_k` 已经是 7，而该窗口的**生产产物**
        # （`convergence.json` / `energies.npy`）还是**6 态的旧帧**。
        # 控制器知道证据过期（`stale_layout_evidence`），却仍然选了
        # `PROBE_REANCHOR_EPOCH` —— 那是"拿已有帧重解"那一族，而这些帧描述的是
        # **另一个窗口几何**，解出来必然维度不符。
        #
        # 与 1d（被求解器跳掉）、1e（预热预算耗尽）是**同一个形状**的第三例：
        # **一个在构造上不可能成功的动作被发了出去**。插过 λ 之后唯一能推动的
        # 只有重采。
        _stale_now = self.stale_layout_windows(view)
        if earliest is not None and int(earliest) in _stale_now:
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {earliest} 的生产产物描述的是**另一套 λ 布局**"
                "（插 λ / 拆窗之后，旧帧的态数与当前 `window_ranges` 不符）⟹ "
                "**任何拿已有帧重解的动作（重标定 / 换 Epoch / 累计残差）在构造上都会"
                "维度不符**，真机实测是直接 `ValueError` 炸出流水线。"
                "唯一能推动它的动作是**按新布局重采**。",
                windows=[int(earliest)], blocked=blocked, earliest=earliest,
            )

        # 1e) 🔑🔑 [2026-09-14 真机] **预热预算耗尽且 f_k 未判定的窗口，根本进不去。**
        #
        # 要给这种窗口补生产帧，流水线必须**重新进入**这个窗口；而入口处的 warmup
        # 门会先看它的累计预算 —— 已耗尽就抛 `LOCAL_VALIDATION_CAP`
        # （"累计预算已用尽（上限 555000、已耗 555000），本次可用 0 步…要继续必须
        # 显式升档"）。于是 `RUN_PRODUCTION` 在**构造上**推不动这个窗口。
        # 真机：win0 连发 40 轮、每轮都被路由信号弹回来，而路由那条 `continue`
        # 还绕过了 no-op 记账，连"这条路走不通"都留不下。
        #
        # 对症动作只有两个：**缩跨度**（布局动作，不需要重进预热），
        # 或**显式升档 warmup 预算**（调用方的决定，控制器不替它做）。
        if earliest is not None and int(earliest) not in _replaced_parents:
            _ew = next((w for w in W if int(w["window_idx"]) == int(earliest)), {}) or {}
            _left = _ew.get("warmup_steps_left")
            _unenterable = (
                _left is not None and int(_left) <= 0
                and str(_ew.get("f_k_evidence_status") or "") not in ("verified",)
            )
            if _unenterable:
                _can_ins = feas.get("insert_lambda") is None
                if _can_ins:
                    return plan(
                        "INSERT_LAMBDA",
                        f"窗口 {earliest} 的 warmup 累计预算已耗尽（剩 {_left} 步）"
                        f"且 f_k 仍是 `{_ew.get('f_k_evidence_status')}` ⟹ "
                        "**重新进入该窗口采样会在预热门上被直接弹回**"
                        "（LOCAL_VALIDATION_CAP），补帧类动作在构造上推不动它。"
                        "不需要重进预热的对症动作是**缩跨度**：插 λ。",
                        windows=[int(earliest)], blocked=blocked, earliest=earliest,
                    )
                _rw = _bounded_rewindow(
                    int(earliest),
                    f"窗口 {earliest} 的 warmup 累计预算已耗尽（剩 {_left} 步）且 f_k 仍是 "
                    f"`{_ew.get('f_k_evidence_status')}` ⟹ 补帧会在预热门上被弹回；"
                    f"而插 λ 也不可行（{feas.get('insert_lambda')}）",
                )
                if _rw is not None:
                    return _rw
                return plan(
                    "NO_ACTION",
                    f"窗口 {earliest} 的 warmup 累计预算已耗尽（剩 {_left} 步）且 f_k 仍是 "
                    f"`{_ew.get('f_k_evidence_status')}` ⟹ 补帧会在预热门上被弹回；"
                    f"而缩跨度也不可行（插 λ：{feas.get('insert_lambda')}；"
                    f"有界重窗：{self.rewindow_feasible(view, earliest)}）。"
                    "**要继续必须显式升档该窗口的 warmup 预算** —— 那是调用方的决定，"
                    "控制器不替它做，也不假装还有别的动作。",
                    exit_="NO_FEASIBLE_ACTION", windows=[int(earliest)],
                    blocked=blocked, earliest=earliest,
                    # [2026-09-17] 这个窗口进不去且缩不了跨度；别的窗口不受影响
                    halt_scope="TARGET_LOCAL",
                )

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
        #
        # 🔑🔑 [审计 #20] **这道闸已经挪进 `plan()`**（见那里的长注释）。
        # 留在这里的版本是**无条件**跑的：不分接下来要发什么动作，一律先判
        # "这个窗口付不付得起换 Epoch"，付不起就 return `RUN_PRODUCTION` ——
        # 于是账本读不到的窗口被永久钉死，后面所有分支（1c/1d/3/3a/4/5/5a/5b/6）
        # 一条都到不了。预算消费的**顺序**没变（仍在选定动作之后、返回之前），
        # 变的只是"只对真要花这本账的动作消费"。

        # 1c) 🔑🔑 [2026-09-14 裁决 1] **未完成的子窗优先于对父窗口的任何动作。**
        #     父窗口已经被子系综取代（它整个不在求解覆盖里），所以对它本身做什么
        #     都没有意义；能推动结果的只有把子窗补完。按因果顺序挑
        #     （父窗号、局部下标），并且**不越过更早的未解决物理窗口**。
        #     动作带 `unit_id` —— 执行器据此续跑**那一个子窗自己的 checkpoint**，
        #     绝不回头动父窗（子窗局部下标 0/1 与父窗索引同名不同义）。
        # 🔑 [CTL-02] **只有"缺帧"才走补帧这条分支。** 支撑/偏斜类失败的子窗
        # 由下面的 1c-2 单独处理 —— 拿加帧顶替它正是要修的那个错。
        _pending_units = sorted(
            [u for u in (view.get("sampling_units") or [])
             if u.get("needs_frames")],
            # [2026-09-17] 父窗未知 ⟹ 排到**最后**（大数），不冒充 window 0。
            # 先前 `or 0` 让它插到队首、并被当成 window 0 的子窗路由。
            key=lambda u: (_pw_or_last(u), int(u.get("local_index") or 0)),
        )
        if _pending_units:
            _u = _pending_units[0]
            _pw = _u.get("parent_window")
            if _pw is None:
                # 父窗未知 ⟹ 不知道它在因果顺序里排哪儿，路由它可能越过未解决的
                # 上游窗口。fail-closed：跳过，并由 `plan()` 的常设字段报出来。
                _pending_units = []
            elif earliest is None or int(_pw) <= int(earliest):
                _why = (
                    "还没有生产产物" if not _u.get("has_convergence")
                    else ("被求解器踢出协方差链"
                          f"（去相关后 {(_u.get('solver_skip') or {}).get('n_frames_after_decorrelation')} 帧）"
                          if _u.get("solver_skip")
                          else f"自检判帧数不足（verdict={_u.get('self_verdict')}）")
                )
                return plan(
                    "RUN_PRODUCTION",
                    f"父窗口 {_pw} 已被 immutable rewindow 的子系综取代，子窗 "
                    f"`{_u['unit_id']}`（局部下标 {_u.get('local_index')}，λ 区间 "
                    f"{_u.get('range')}）{_why} ⟹ **续跑这个子窗自己的 checkpoint**"
                    f"（已采 {_u.get('production_steps')} 步）。"
                    "语义是「尚不可测」（INSUFFICIENT_DATA ≠ FAIL）；"
                    "对父窗口本身做任何动作都推不动结果 —— 它整个不在求解覆盖里。",
                    windows=[_pw], unit_id=_u["unit_id"],
                    blocked=blocked, earliest=earliest,
                )

        # 1d) 🔑🔑 [2026-09-14 真机] **被求解器跳掉的窗口只能加帧，不能重解。**
        #
        # 「重标定 f_k」「换 Epoch 拿独立证据」都要**拿这个窗口已有的生产帧重解**。
        # 而求解器把它跳掉的意思恰恰是「这些帧去相关之后不够解」（9 帧 < 门 10）
        # ⟹ 那一整族动作在**构造上**就是 no-op。
        #
        # 真机 cyclod：win0 被跳（9 帧），f_k 探针按**节奏**（已采 ≥ 500k 步）把它
        # 列进建议重标定名单，而执行器按**相邻位移 > 0.5 kJ/mol** 判 —— 同一个决定
        # 两套判据，且 win0 **根本没有位移读数**（它解不出来，位移表里只有
        # w1/w2/w4/w5）。于是 `RECALIBRATE_FK[0]` 连发 4 次、停滞保护降级到
        # `PROBE_REANCHOR_EPOCH` 又是同一个 no-op，最后 NO_FEASIBLE_ACTION 退出。
        # 全程盘面一个字节没变，而它真正需要的动作（加帧）就在分支 6。
        #
        # 「低支撑永远不是 FAIL，是『尚不可测』，动作是加预算」——
        # 这条在这里就该生效，不该等 f_k 那一整族先空转一遍。
        _solver_skipped_now = (
            {int(x) for x in (view.get("skipped_windows") or [])}
            | {int(w["window_idx"]) for w in W if w.get("solver_skip")}
        )
        if (earliest is not None and int(earliest) in _solver_skipped_now
                and int(earliest) not in _replaced_parents):
            _sk = next((w.get("solver_skip") for w in W
                        if int(w["window_idx"]) == int(earliest)), None) or {}
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {earliest} 去相关帧数不足、被**求解器**踢出协方差链"
                + (f"（去相关后 {_sk.get('n_frames_after_decorrelation')} 帧 < "
                   f"{_sk.get('min_frames_per_window')}）"
                   if _sk.get("n_frames_after_decorrelation") is not None else "")
                + " ⟹ 它的帧不够解，**任何拿已有帧重解的动作（重标定 / 换 Epoch）"
                "在构造上都是 no-op**。对症动作是**加帧**："
                "低支撑永远是「尚不可测」（INSUFFICIENT_DATA ≠ FAIL），动作是加预算。",
                windows=[int(earliest)], blocked=blocked, earliest=earliest,
            )

        # 1c-2) 🔑🔑 [CTL-02] **支撑/偏斜类失败的子窗：加帧治不了，别拿它顶替。**
        #
        # 这条与上面 1c 是**两种科学问题**，不是同一件事的两个程度：
        #   · 1c   缺样本量   ⟹ 加帧（INSUFFICIENT_DATA，"尚不可测"）
        #   · 1c-2 支撑/偏斜 ⟹ 缩跨度；没有有界的缩跨度动作就**如实停下**
        # 先前两者都落进 `complete=False` 这一个布尔、都发 `RUN_PRODUCTION`，
        # 于是 top1%/raw ESS 失败的子窗被反复加帧（§5.1 实测越加越差）。
        # 🔑 [2026-09-17 P0] `support_failed` 现在**只**由 `STRUCTURAL` 置位
        # （见 `read()` 的第二遍归因）。`UNKNOWN` 不进这个选择器 —— D3 停机是
        # 「有界重窗这条路用尽」的结论，归因判不出来时说不出这句话。
        _skew_units = sorted(
            [u for u in (view.get("sampling_units") or [])
             if u.get("support_attribution") == SUPPORT_FAILURE_STRUCTURAL
             and not u.get("needs_frames")],
            key=lambda u: (_pw_or_last(u), int(u.get("local_index") or 0)),
        )
        if _skew_units:
            _su = _skew_units[0]
            _spw = _su.get("parent_window")
            if _spw is not None and (earliest is None or int(_spw) <= int(earliest)):
                # 🔑🔑 [2026-09-17，用户拍板 D3] **专门的终止码，而不是笼统的
                # `NO_FEASIBLE_ACTION`。**
                #
                # 这条死线（DEAD_LINES 的 D3）在 2026-09-17 之前**结构上不可达**
                # （子窗只由 `IMMUTABLE_REWINDOW` 产生，而它发不出来）。REWIND-01
                # 接通之后它第一次真正会被走到，所以它值一个说得清的名字：
                # 走到这里意味着「**有界重窗这条路已经用尽**」——不是"没动作可做"
                # 那种笼统无解，而是一个具体的、可以拿去做决定的结论。
                #
                # ⚠️ **不开放递归。** 现在的数据模型**只能表达"一次物理父窗替代"**：
                # `rewindow_feasible()` 只收物理 `window_idx`；执行器与合并器都没有
                # `parent_unit_id` / ancestry / 局部替代 / 递归预算语义。直接放开会
                # 产生重复区间覆盖、identity 冲突和预算失真。要开放必须先设计
                # 层级契约，而且门槛是：**至少三个独立 seed 稳定复现 D3，并证明
                # 二次切分优于停机。**
                # ⚠️ 也**不再补帧**、**不创建第二个 rewindow entry**。
                #
                # 诊断一次给全（unit / range / identity / 失败来源 / g / η / 已用块数）
                # —— 这条终态的全部价值就是让人不用再去翻盘面。
                _d3_blocks = len(
                    (view.get("production_blocks_total_by_unit") or {}).get(
                        str(_su.get("unit_id")), []) or [])
                _d3 = {
                    "unit_id": str(_su.get("unit_id")),
                    "parent_window": _spw,
                    "range": _su.get("range"),
                    "identity": _su.get("identity"),
                    "support_failure_source": _su.get("support_failure_source"),
                    "self_verdict": _su.get("self_verdict"),
                    "blocks_used": _d3_blocks,
                    # 瓶颈态观测（同 k*、同 frame set；读不到是 None = UNKNOWN）
                    "bottleneck_state": _su.get("bottleneck_state"),
                    "bottleneck_g": _su.get("bottleneck_g"),
                    "bottleneck_eta": _su.get("bottleneck_eta"),
                    "bottleneck_ratio": _su.get("bottleneck_ratio"),
                }
                return plan(
                    "NO_ACTION",
                    f"子窗 `{_su['unit_id']}`（父窗 {_spw}，λ 区间 {_su.get('range')}，"
                    f"identity {_su.get('identity')}）的失败是**支撑/偏斜类**"
                    f"（归因 {_su.get('support_failure_source')}，"
                    f"verdict={_su.get('self_verdict')}，已批 {_d3_blocks} 块；"
                    f"瓶颈态 k*={_su.get('bottleneck_state')}，"
                    f"g={_su.get('bottleneck_g')}，η={_su.get('bottleneck_eta')}，"
                    f"N_eff/g={_su.get('bottleneck_ratio')}）——**不是样本量不足**。"
                    "同分布加帧治不了偏斜（§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 "
                    "0.762、ESS 比值反而更差），而它已经是**固定 λ 表上的有界重窗**、"
                    "**有界重窗这条路到此为止**：当前数据模型只能表达「一次物理父窗"
                    "替代」（执行器/合并器没有 `parent_unit_id`、没有 ancestry、"
                    "没有递归预算语义），强行再切一层会产生重复区间覆盖、identity "
                    "冲突和预算失真 ⟹ **如实停下**，不递归、不补帧、不再建第二个"
                    "子系综。要继续需要显式换布局或升档预算；要开放二次切分，"
                    "先拿**至少三个独立 seed** 稳定复现这条终态、并证明二次切分优于停机。",
                    exit_="D3_REWINDOW_DEPTH_EXHAUSTED", windows=[_spw],
                    unit_id=_su["unit_id"], blocked=blocked, earliest=earliest,
                    extra={"rewindow_depth_exhausted": _d3},
                )

        # 1c-3) 🔑🔑 [2026-09-17 P0] **归因判不出来的子窗：两边都不授权。**
        #
        # `UNKNOWN` 不是"介于两者之间"，是「**还没判出来**」。它不得授权
        # `RUN_PRODUCTION`（可能是结构性的，加帧越加越差）、也不得授权
        # `D3 停机` / `IMMUTABLE_REWINDOW` / `SPLIT_TAIL_WINDOW` / `INSERT_LAMBDA`
        # （可能只是数据缺口，改布局是拿"没测出来"当"测出来是坏的"）。
        # ⚠️ **但它不该停掉整跑**：别处还有活干就绕过它继续调度。所以这里只在
        # 「没有 earliest、也没有别的可做单元」时才发终态。
        _unk_units = [u for u in (view.get("sampling_units") or [])
                      if u.get("attribution_unknown")
                      and not u.get("needs_frames")
                      and u.get("schedulable", True)]
        if _unk_units and earliest is None and not _pending_units:
            _uu = _unk_units[0]
            return plan(
                "NO_ACTION",
                f"子窗 `{_uu['unit_id']}`（父窗 {_uu.get('parent_window')}，"
                f"λ 区间 {_uu.get('range')}）自检判失败，但**归因判不出来**"
                f"（verdict={_uu.get('self_verdict')}，来源 "
                f"{_uu.get('support_failure_source')}；射程三输入 "
                f"ratio={_uu.get('min_n_eff_over_g')} / "
                f"target={_uu.get('self_n_eff_over_g_eligible')} / "
                f"headroom 缺任一 ⟹ 判不了）。"
                "`UNKNOWN` **两边都不授权**：补帧可能是"
                "越加越差（若其实是结构性），改布局则是拿「没测出来」当"
                "「测出来是坏的」（若其实只是数据缺口）。"
                "⟹ 停在诊断态。要往下走，先把缺的读数补出来"
                "（`n_eff_over_g_eligible_threshold` / 块账 / 自检来源），"
                "**不是**挑一个动作试试看。",
                exit_="SUPPORT_ATTRIBUTION_UNKNOWN",
                windows=([int(_uu["parent_window"])]
                         if _uu.get("parent_window") is not None else []),
                unit_id=str(_uu["unit_id"]),
                missing=[f"子窗 {_uu['unit_id']} 的支撑失败归因输入"],
                blocked=blocked, earliest=earliest,
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
                # 🔑🔑 [2026-09-14] **这里恒为路由，绝不发终态。**
                # 这四处（本处 / skipped+rescue / f_k 被驳回 / 探针建议重标定）
                # 形状完全一样：「新 Epoch 付不起验证额度 ⟹ 退而补生产帧」。
                # 而 `all_windows_budget_exhausted` 判的是
                # **所有窗口的 `warmup_steps_left` ≤ 0** —— 纯预热账本。
                # `RUN_PRODUCTION` 用的是**已冻结的 f_k**，一步验证预算都不花。
                # 先前在这里发 `GLOBAL_BUDGET_EXHAUSTED`（它在 `TERMINAL_EXITS` 里）
                # ⟹ 主循环在分发**之前**就 break，那个补帧动作**根本不执行**：
                # 拿 A 账本的余额，终止一个只花 B 账本的动作。
                # ⚠️ 控制器里目前**没有"生产补帧预算"这个量**
                # （`per_window_budget_remaining` 也是 `warmup_steps_left`），
                # 所以这里无法判定补帧可不可行 —— 唯一能判的地方是主循环的停滞
                # 保护（"补了一轮，盘上没变"），`GLOBAL_BUDGET_EXHAUSTED` 已改由
                # 它发出。造一个真正的双层预算是 PLAN 第 5 步 F，尚未实现。
                exit_="HALT_BUDGET",
                windows=_sel,
                blocked=blocked, earliest=earliest,
            )

        # 2) f_k 被**有统计功效地**驳回 ⟹ 终态。不许"再测一次"。
        refuted = _pick([w for w in W if w["verdict"] == "STATISTICALLY_REJECTED"])
        if refuted:
            # 🔑 [审计 #20] 这里原来又抄了一遍「付不起就退而补生产帧」的分岔。
            # 那道闸现在统一在 `plan()` 里（只对换 Epoch 类动作跑），这一处是
            # **死代码**：`refuted` 里的窗口必然已经被上面那道无条件前置闸截走。
            # 直接发 `RECALIBRATE_FK`，付不付得起由 `plan()` 判并按同样的口径改发
            # `RUN_PRODUCTION` + `HALT_BUDGET`。
            return plan(
                "RECALIBRATE_FK",
                f"窗口 {refuted} 的 f_k 证据是 refuted（有统计功效的否决）⟹ 必须换 "
                "Epoch 重标定，**不能**再加验证预算。"
                # [2026-09-14 更正] 原来这里写「抛 IBSFrozenCalibrationValidationError "
                # 而全仓库没有任何 except 捕获它、会直接炸穿整个 run」——**已经不成立**：
                # `abfe_pipeline._run_stage2_autonomous` 有
                # `except _ie_exc.IBSFrozenCalibrationValidationError`，会封存候选
                # （`seal_refuted_candidate`）、落 `stage2_fk_refuted.json`、再 `continue`
                # 回到顶层重判。留着旧话会让人以为这条路是断的。
                "出口 `HALT_FK_REFUTED` 是**路由**：这个动作要执行，不是终止。",
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
        # [2026-09-17] 判据搬到 `local_validation_cap_hits()`（与 `plan()` 共用一份）——
        # 先前这里和 plan 各判各的，正是"同一个不变量两份实现"。
        _cap_idx_now = {int(h["window_idx"]) for h in self.local_validation_cap_hits(view)}
        _cap_hit = [w for w in with_budget if int(w["window_idx"]) in _cap_idx_now]
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
                f"凑够 {int(_r['required_decorrelated_frames'])} 个去相关帧**从零算起**"
                f"需要约 {int(_r['projected_raw_frames_needed'])} 原始帧 = {_need} 步；"
                f"该窗口已攒 {_w.get('validation_sample_count')} 帧，"
                f"**还差约 {max(0, _need - int(_w.get('warmup_steps_spent') or 0))} 步**，"
                f"而剩余预算 {_w.get('warmup_steps_left')} 步。"
                # 🔑 [审计 #52，2026-09-14] **`projected_steps_needed` 是从零算起的
                # 总步数，不是缺口。** 先前把它与 `warmup_steps_left` 并排打印成
                # 「还需 X 步 / 剩余仅 Y 步」，读者会把 X 当缺口 ⟹ 报出去的差距被
                # 系统性夸大（已攒的那部分被重复计了一次）。
                # ⚠️ **判定本身不受影响**：判定用的是 `gcrit_budget`，它已经含
                # `frames_already`。这里只是把措辞改对，别顺手去改判据。
                "（判定用的是已含 `frames_already` 的 `gcrit_budget`，不是上面这个总数。）"
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
            # 🔑 [2026-09-14 梳理] 这里原来写的是 `action="DONE"` 配
            # `exit="NO_FEASIBLE_ACTION"` —— 自相矛盾：`plan()` 按 `action == "DONE"`
            # 算 `execution_status=COMPLETE`，于是一个"无路可走"的结局被记成
            # "执行完毕"。与 09-13 修过的那处是同一个毛病，这一处漏了。
            return plan(
                "NO_ACTION",
                _why + "处置：替代候选**已经用过一次**（一个窗口只给一次，"
                "否则就是反复试到偶然通过）⟹ NO_FEASIBLE_ACTION，"
                "留完整诊断终止。**不是 DONE** —— 结果没有达标。",
                exit_="NO_FEASIBLE_ACTION", windows=_un_sel,
                # [2026-09-17] 这个窗口的唯一一次替代候选已用掉；别处照常
                halt_scope="TARGET_LOCAL",
            )

        # 🔑🔑 [2026-09-15] **「只给一个 +250k 诊断块」在这里兑现。**
        # 已经用掉那一块的窗口（`bias_status == "provisional_production"`）不再
        # 发这个动作 —— 它的处置改由块后的 N_eff/g 判据接手（达到 10 则继续；
        # 边际停滞/下降或 far-end support 塌陷 ⟹ 关 Epoch 走 tail rewindow；
        # top1% 灾难性集中 ⟹ 停止同分布加帧），那些分支在下面本来就有。
        # ⚠️ 再发一次 = 「反复试到偶然通过」，正是 4818 那段明令禁止的。
        _cap_hit = [
            w for w in _cap_hit
            if str(w.get("bias_status") or "") != "provisional_production"
        ]
        _cap_sel = _pick(_cap_hit)
        if _cap_sel:
            _cap_hit = [w for w in _cap_hit if int(w["window_idx"]) in _cap_sel]
            _idx = _cap_sel
            return plan(
                "PROVISIONAL_PRODUCTION",
                f"窗口 {_idx} 的单周期验证**批次上限**已打满"
                f"（{[w.get('frozen_validation_batches') for w in _cap_hit]}/{_batch_cap} 批），"
                f"而全局预算**还有钱**（剩 {[w.get('warmup_steps_left') for w in _cap_hit]} 步）"
                " ⟹ 这是 `LOCAL_VALIDATION_CAP_EXHAUSTED`，**不是 HALT_BUDGET、"
                "更不是 F_K_REFUTED**（没有任何证据驳回这份 f_k，只是这一轮没测出来）。"
                "处置：**不扩大 15 批上限**；进 PROVISIONAL_PRODUCTION（**不是可信 PASS**）"
                "—— 执行器显式授权引擎用这份**未验证**的冻结 f_k 采一块，"
                "引擎把 `bias_status` 写成 `provisional_production`、证据保持 "
                "`indeterminate`、warmup_failure.json 留作证据（不删）；"
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
        if _wb_sel and all(_is_noop("CONTINUE_WARMUP", i) for i in _wb_sel):
            # 🔑🔑 [2026-09-14 真机] **续预热对"生产缓存已完整"的窗口是结构性 no-op。**
            # 窗口级 resume 缓存门在**走到预热之前**就 `continue` 掉整个窗口
            # （"已有有效缓存能量 … resume 模式下跳过重新采样"），所以
            # `CONTINUE_WARMUP` 一步也跑不了。真机 rep2：win4 连发 **40 次**、
            # `production_steps` 与 `warmup_steps_left` 一动不动，
            # `repeat_count` 在 1→2→3 之间循环（停滞保护每次降级到探针、探针又
            # 改了一点盘面把计数清零），永远走不到退出。
            # 它真正缺的是**生产帧**（端点 σ），那才是对症动作。
            return plan(
                "RUN_PRODUCTION",
                f"窗口 {_wb_sel} 仍在预热阶段且 warmup 预算有余，但**续预热在当前盘面"
                "上已被证明什么也没做**（窗口级 resume 缓存在走到预热之前就跳过了整个"
                "窗口）⟹ 换对症动作：补生产帧。"
                "语义仍是「尚不可测」（INSUFFICIENT_DATA ≠ FAIL）。",
                windows=list(_wb_sel), blocked=blocked, earliest=earliest,
            )
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

        # 4) 预热预算耗尽、仍未收敛 ⟹ 需要归因。
        # 🔑 [审计 #21，2026-09-14 复核] **这一段先前是死代码，现在可达了。**
        # 原因是 #20：换 Epoch 的预算预检当时是一道**无条件**前置闸，`stuck` 里的
        # 窗口必然先被它（或 1e）截走 ⟹ `HALT_NO_ATTRIBUTION` 在生产里从不执行。
        # #20 把那道闸挪进 `plan()`、只对换 Epoch 类动作跑之后，这条路打开了：
        # 1e 只处理「预热预算耗尽 **且** f_k 未判定（`f_k_evidence_status != verified`）
        # **且** 确实进不去」的窗口；「预算耗尽但 f_k 已 verified、仍卡在 WARMUP_*」
        # 的窗口落到这里做归因 —— 两条分支的前提互补、不重叠，不需要再改 1e。
        # 归因要的三个量
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
                    # [2026-09-17] 与上面 exit_ **同一个**条件：只有全窗预算耗尽且无可行结构动作才是终态（stage 级）；否则是路由、没有 scope
                    halt_scope=("STAGE_GLOBAL" if (view.get("all_windows_budget_exhausted") and not any(v is None for v in feas.values())) else None),
                )
            # 重解在当前盘面已被证明是 no-op ⟹ 直接走布局动作，别再空转一轮。
            # 🔑🔑 [审计 #14，2026-09-14] **三元式与它上面的注释正好相反。**
            # 注释（也就是 PLAN §3ter.4 的优先级）是：**先重标定 f_k，仍压不住才补 λ**。
            # 代码写的却是「插 λ 可行就插、不可行才重标定」—— 顺序整个反过来。
            # 而且 `_recal_is_noop` 为真时 `or` 会**短路**发一个 `INSERT_LAMBDA`，
            # 哪怕插 λ 此刻不可行；同一条理由文本还在打印「补 λ 当前不可行」，
            # exit 词也配着 λ 预算耗尽 —— 动作、理由、出口三样互相矛盾。
            # 按注释写的优先级重写：
            #   默认 RECALIBRATE_FK；只有「重标定在当前盘面已被记为 no-op」**且**
            #   「插 λ 可行」才改发 INSERT_LAMBDA；两者都不行就如实停下。
            # `HALT_LAMBDA_BUDGET_INSUFFICIENT` 只在**真的因为 λ 预算耗尽**而无路可走
            # 时才用（λ 总数不够是输入问题，交人工改输入）。
            _recal_is_noop = all(_is_noop("RECALIBRATE_FK", i) for i in idxs)
            _ins_ok = feas.get("insert_lambda") is None
            _why4 = (
                f"窗口 {idxs} 的 warmup 预算已耗尽且未收敛。按 PLAN §3ter.4，窗口级"
                "缺陷的对症动作依次是：**重标定 f_k → 仍压不住才补 λ 缩跨度**（model B）。"
            )
            if not _recal_is_noop:
                return plan(
                    "RECALIBRATE_FK", _why4 + "重标定尚未在本盘面上被证伪 ⟹ 先重标定。",
                    windows=idxs, blocked=blocked, earliest=earliest,
                )
            if _ins_ok:
                return plan(
                    "INSERT_LAMBDA",
                    _why4 + "重标定在**当前盘面**上已被执行器记账为 no-op"
                    "（跑过一次、盘面逐项未变）⟹ 按序进到下一档：补 λ 缩跨度。",
                    windows=idxs, blocked=blocked, earliest=earliest,
                )
            return plan(
                "NO_ACTION",
                _why4 + "重标定在当前盘面上已被记为 no-op，"
                f"而补 λ 也不可行（{feas.get('insert_lambda')}）⟹ 如实停下。"
                # 归因（λ 预算耗尽 vs 布局顶到上限）写进理由文本，**不进 exit 词**：
                # `NO_ACTION` 必须配终态出口，否则主循环不 break、执行器会收到一个
                # 它不认识的动作（`test_stage2_controller_vocabulary` 钉的就是这条）。
                # 先前这里配 `HALT_LAMBDA_BUDGET_INSUFFICIENT`（非终态）正是那个毛病。
                + ("归因：λ 总数不够是**输入问题**，应由人工改输入 λ 表。"
                   if "预算已用尽" in str(feas.get("insert_lambda") or "")
                   else "归因：布局本身顶到了可拆上限，不是预算问题。"),
                # 两个出口都在 `TERMINAL_EXITS` 里（`NO_ACTION` 必须配终态出口，
                # 否则主循环不 break、执行器会收到一个它不认识的动作）。
                # 只有**真的因为 λ 预算耗尽**才用那个更具体的词。
                exit_=("HALT_LAMBDA_BUDGET_INSUFFICIENT"
                       if "预算已用尽" in str(feas.get("insert_lambda") or "")
                       else "NO_FEASIBLE_ACTION"),
                windows=idxs, blocked=blocked, earliest=earliest,
                # [2026-09-17] 与 exit_ 同源：λ 总数不够是**输入问题**（stage 级，换窗也没用）；布局顶到可拆上限则只是这个窗口没路走
                halt_scope=("STAGE_GLOBAL" if "预算已用尽" in str(feas.get("insert_lambda") or "") else "TARGET_LOCAL"),
            )

        # 5) 生产帧没攒够 ⟹ 接着跑（同一个 f_k、接着原段，不改任何结构）。
        short = [
            w for w in W
            # ⚠️ **「未知不是零」这条通则在这一行是有意的例外，别按通则"修"它。**
            # 别处压扁 `None → 0` 都会造出错误决策（虚报余量 / 虚报耗尽 / 把未知
            # 固化成事实落盘）。**这里不会**：未知与 0 的**处置完全相同** ——
            # 两者都该去跑生产。而改成「未知就不判它短」会压制掉「真的一步没跑过」
            # 的窗口的正常生产（那正是 `production_steps=None` 最常见的成因：
            # 还没有 convergence 产物）。
            if w["production_steps_target"] and (w["production_steps"] or 0)
            < int(w["production_steps_target"])
        ]
        # 🔑 [2026-09-17] `earliest is None`（全窗自检合格）时 `_pick()` 恒返空，
        # 而"把已承诺的步数跑完"这件事**不重锚任何上游**、也不改布局 ⟹ 它是
        # `_pick` 的合法例外。没有这条例外，上面那道"不许判终态"的闸会让控制器
        # 卡在兜底上空转，而不是去把帧补上。
        _sh_sel = _pick(short) or (
            [int(w["window_idx"]) for w in short] if earliest is None else [])
        # 🔑🔑 [审计 #22，2026-09-14] **「生产帧没攒够」本来就不是 PROBLEM 态，
        # 所以它必须有自己的取窗口逻辑，不能借 `earliest`。**
        #
        # `earliest is None` 的意思是「没有任何窗口处于 PROBLEM 态」；而生产帧没
        # 攒够的窗口是 `UNKNOWN`（还没有自检产物）或 `ELIGIBLE`。先前这一条完全
        # 靠 `_pick(short)`，于是**全窗 UNKNOWN 的盘面**（老 run / 刚起步的 run，
        # 最常见）上它一次都触发不了，`decide()` 只发得出 6a 的 `ANALYZE` ——
        # 而窗口根本还没跑完，分析是空转，靠停滞保护退出。
        # 定点修：`earliest is None` 时按窗口序取 `short` 里最早的那一个。
        # （**不放宽 `_pick`** —— 那会一次性拆掉另外 10 条分支的隐式前提。）
        # ⚠️ 已被 rewindow 取代的父窗不参与（CTL-03：它整个不在求解覆盖里）。
        if not _sh_sel and earliest is None and short:
            _cand = sorted(int(w["window_idx"]) for w in short
                           if int(w["window_idx"]) not in _replaced_parents)
            _sh_sel = _cand[:1]
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
        # 5a-0) **SUSPECTED_DERAILMENT（单块）⟹ 只做非变异的候选计算，不切换。**
        #   老板定案：单块只产生"疑似"，触发廉价诊断；**连续两块**才允许关闭 Epoch。
        #   ⚠️ 而且在 **held-out 反事实验收**接好之前，单块**一律不得**升级成真重标定 ——
        #   否则就是"因一个震荡低块贸然切换到更差的 f_k"（win1 已经吃过这个亏）。
        #   held-out 验收目前**未实现**，所以这一档现在只能到"算候选 + 报告"为止。
        # 🔑🔑 [2026-09-17] **`earliest is None` 时也要发 —— 这是 `_pick` 的第二个
        # 合法例外（第一个是分支 5 的"补足已承诺步数"）。**
        #
        # 脱轨探针的目标窗口自检往往**仍然 ELIGIBLE**（脱轨是单块边际 N_eff 塌，
        # 不是整窗不合格）⟹ 路由态不是 PROBLEM ⟹ 它**永远不会**成为 `earliest`
        # ⟹ `_pick` 恒返空 ⟹ 这条分支一次都不触发。注意这不是"押后"：
        # `earliest` 只从 PROBLEM 里选，退役只会把窗口**移出**排序、永远不会把一个
        # ELIGIBLE 的加进来 —— 所以它是**永久**的丢失。
        #
        # 开这个例外的理由与分支 5 同一条：`PROBE_CANDIDATE_FK` 是 `probe_only=True`，
        # **离线算候选、零采样**（见 `_PRODUCTION_CHARGED` 那段：它是唯一真正的
        # 非变异探针），不重锚、不改布局、不花 GPU。零成本的取证没有理由被
        # 「先修最早的」挡住。
        _derailed = [w for w in W
                     if w.get("derailment_status") == "SUSPECTED_DERAILMENT"]
        _suspected = _pick(_derailed) or (
            [int(w["window_idx"]) for w in _derailed] if earliest is None else [])
        _confirmed = [
            w["window_idx"] for w in W
            if w.get("derailment_status") == "CONFIRMED_DERAILMENT"
        ]
        # 🔑 [2026-09-12] **探过一次就得往下走。**
        # 探针是**非变异**的，按定义不改盘 ⟹ "盘上状态没变"对它永远成立 ⟹
        # 主循环的停滞保护必然把它判成"推不动"，连探三次后 NO_FEASIBLE_ACTION
        # 退出（真机 16:46 就是这么把 win5 放弃的）。
        # 它已经给出结论了（`stage2_fk_recalibration_probe.json`），
        # 所以探过就得往下走，让后面的分支按那个结论选动作。
        #
        # 🔑🔑 [2026-09-15] **但"探过"必须是「在**这个盘面**上探过」，不是"探过一次"。**
        # 先前这里用 `_probed`（探针产物的 `windows` 列表）判，而那份列表含**全部**
        # 窗口（探针一次就把所有窗口都写进去），于是**第一次探针之后全仓永久不再重探**。
        # 致命之处在于探针的结论是**步数的函数**（固定节奏重锚：已采步数 ≥ cadence），
        # 每补一块帧都可能翻面 —— 只在第一轮算一次等于把这条判据钉死在初始步数上。
        # 真机 4W53 vanishing：第 5 轮探针在 250k 步判"未到节奏"，此后 win0 补到
        # 1.25M 步（早已超过 500k 的节奏）却再也没被重探，最后 NO_FEASIBLE_ACTION。
        # 换成 `_is_noop`：它的指纹**带生产步数与路径版本**，补过帧/换过段/插过 λ
        # 之后记录自动失效、动作重新可选，正好是这里要的语义；而"同一盘面上重复发"
        # 仍然被挡住，防停滞的原意一个字没丢。
        if _suspected and all(
            _is_noop("PROBE_CANDIDATE_FK", int(i)) for i in _suspected
        ):
            _suspected = []
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
        # 被求解器跳掉的窗口在上面 1d 就已经被路由走了（重解类动作对它们是 no-op），
        # 这里不会再看到它们。
        _recal = [
            i for i in _pick(
                [int(x) for x in
                 (_probe.get("recalibration_recommended_windows") or [])],
                key=None,
            )
            # 🔑 [2026-09-14 真机] 执行器已经在**这个盘面**上试过一次、什么也没做
            # （"重标定未产生新段：没有窗口超过 0.5 kJ/mol 相邻位移阈值"）⟹ 不再发。
            # 探针按**节奏**推荐、执行器按**位移**放行，而且两者读的还不是同一份
            # 数据（执行器从"有这个窗口数据的最新段"重解，真机那次是只含 win0 的
            # `vanishing_9`；探针读的是全量）。同一个决定两套判据 ⟹ 死循环。
            if not _is_noop("RECALIBRATE_FK", i)
        ]
        if _recal:
            # 🔑 [审计 #20] 同分支 2：这里原来又抄了一遍「付不起 ⟹ 退而补生产帧」。
            # 那道闸已统一到 `plan()`（只对换 Epoch 类动作跑，口径与理由文本不变），
            # 这一处是死代码 —— 无条件前置闸必然先把这些窗口截走。
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
        # ⚠️ [2026-09-14 真机] **「零额外采样就能算」有前提：这批帧求解器得能用。**
        #   窗口去相关后帧数低于求解器下限（`sufficient=False` / 被踢出协方差链）时，
        #   分析**根本跑不到**这个窗口 —— 它在生成任何诊断之前就被 `continue` 掉，
        #   于是 `cumulative_fk_residual_production` 永远不会出现。原来这里不看这一条，
        #   结果是对这种窗口无限返回 ANALYZE：真机 win0（9 帧 < 10）连发 4 次
        #   ANALYZE、盘上一个字节没变，靠停滞保护降级到 PROBE_REANCHOR_EPOCH
        #   （探针判"f_k 不是瓶颈"什么也没做），最后 NO_FEASIBLE_ACTION 退出——
        #   而它真正需要的动作（5b/6 的补采）就在下面几行，永远轮不到。
        #   这类窗口的正确动作是**先把帧补到求解器能用**，再谈 f_k 偏差证据。
        _unsolvable = set(int(x) for x in (view.get("skipped_windows") or []))
        _need_cum = [
            w["window_idx"] for w in W
            if earliest is not None and int(w["window_idx"]) == earliest
            and w.get("self_verdict") in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA")
            # `UNMEASURED` 是**已产出的证据**（"这个窗口结构上测不了"），不是"还没算"。
            # 拿它去跑 ANALYZE 是空转 —— 见 `cum_fk_verdict` 赋值处的长注释。
            and w.get("cum_fk_verdict") is None
            and w.get("self_sufficient") is not False
            and int(w["window_idx"]) not in _unsolvable
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
                # 🔑 [CTL-07，2026-09-14] **拆末窗只对末窗有意义。**
                # 这里先前只问「拆末窗在结构上可不可行」，不问「失败的是不是末窗」——
                # 与 9b 已经修过的是同一个毛病，这一处漏了：中间窗的 f_k 被 held-out
                # 驳回，却换来一次**与它无关**的尾段重分（烧 GPU、改布局、作废下游
                # 证据，元凶窗口一个字节没动）。判据与 9b 同一份。
                _tail_i = max((int(w["window_idx"]) for w in W), default=None)
                _is_tail = (_tail_i is not None and int(earliest) == int(_tail_i))
                _feas_split = feas.get("split_tail_window") is None and _is_tail
                # 🔑🔑 [审计 #13，2026-09-14] **这里原来完全不查插 λ 的可行性。**
                # 它是全仓最后一处「发布局动作却不问可不可行」的地方（其余每一处
                # ——`_no_gain` 分支、9b —— 都先问过 `feas.get("insert_lambda")`）。
                # 插点预算耗尽 / 末窗顶到可拆上限时，它发出的是一个**在构造上不可能
                # 成功**的动作：执行器要么抛错炸穿流水线，要么绕过 `max_path_insertions`
                # 无限插点（真机末窗被从 K=6 顶到 K=12 就是这个形状）。
                # 归因（f_k 救不了它、要缩跨度）是**对的**，所以不可行时**不改归因**，
                # 只改动作：如实给 NO_ACTION + NO_FEASIBLE_ACTION，并带上 `feas` 的原因。
                _feas_ins = feas.get("insert_lambda") is None
                if not _feas_split and not _feas_ins:
                    _rw = _bounded_rewindow(
                        int(earliest),
                        f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}，"
                        "但候选在 held-out 上**没有改善最差态**（或伤到了健康态）⟹ "
                        "**f_k 救不了它，对症动作是缩跨度**；而 λ 表上的两个缩跨度动作"
                        f"都不可行（拆末窗：{feas.get('split_tail_window')}"
                        f"{'（失败的不是末窗）' if not _is_tail else ''}；"
                        f"插 λ：{feas.get('insert_lambda')}）",
                    )
                    if _rw is not None:
                        return _rw
                    return plan(
                        "NO_ACTION",
                        f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}，"
                        "但候选在 held-out 上**没有改善最差态**（或伤到了健康态）⟹ "
                        "**f_k 救不了它，对症动作是缩跨度**；而三个缩跨度动作都不可行"
                        f"（拆末窗：{feas.get('split_tail_window')}"
                        f"{'（失败的不是末窗）' if not _is_tail else ''}；"
                        f"插 λ：{feas.get('insert_lambda')}；"
                        f"有界重窗：{self.rewindow_feasible(view, earliest)}）⟹ 如实停下。"
                        "要继续需要显式加插点预算或改输入 λ 表。**不是 DONE** —— "
                        "结果没有达标。",
                        exit_="NO_FEASIBLE_ACTION", windows=[earliest],
                        blocked=blocked, earliest=earliest,
                        # [2026-09-17] 这个窗口三个缩跨度动作都不可行；别处照常
                        halt_scope="TARGET_LOCAL",
                    )
                return plan(
                    "SPLIT_TAIL_WINDOW" if _feas_split else "INSERT_LAMBDA",
                    f"窗口 {earliest} 累计 f_k 偏差 span={_w0.get('cum_fk_span')}，"
                    "但候选在 held-out 上**没有改善最差态**（或伤到了健康态）⟹ "
                    "**f_k 救不了它，转布局动作**。"
                    + ("拆末窗。" if _feas_split
                       else (f"失败的不是末窗（末窗是 {_tail_i}）⟹ 在它自己身上插 λ 缩跨度。"
                             if not _is_tail
                             else f"拆窗不可行（{feas.get('split_tail_window')}）⟹ 插 λ 缩跨度。")),
                    windows=[earliest], blocked=blocked, earliest=earliest,
                )
            # None（还没算）或 UNMEASURED（算了但判不了）：都走有界探针 ——
            # 但探针**也是**"拿已有帧重解"那一族，执行器已经在这个盘面上试过且
            # 什么也没做时，再发一次仍然是 no-op（真机停滞保护降级过去、依旧空转）。
            if _is_noop("PROBE_REANCHOR_EPOCH", earliest) or _is_noop(
                    "RECALIBRATE_FK", earliest):
                return plan(
                    "RUN_PRODUCTION",
                    f"窗口 {earliest} 的 held-out 判不了，按序该开有界探针；"
                    "但执行器已经在**当前盘面**上试过重解/换 Epoch、什么也没做"
                    "（未产生新段）⟹ 再发一次仍是 no-op。先补一块帧换取新证据："
                    "盘面一变，那条路自动重新可选。",
                    windows=[earliest], blocked=blocked, earliest=earliest,
                )
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

        # ── 边际增长判据：同分布加帧已经被证伪就别再加 ──────────────────
        # cap 分支的原话：「达到 10 则继续；**边际停滞/下降**或 far-end support
        # 单调塌陷则关闭 Epoch」。这里只落**无歧义的那一半**：主验收量
        # `min N_eff/g` **没有上升**。不自造"停滞"阈值。
        # 真机 win4：3.44 → 2.29 → 2.01 → 1.62，g 15.1 → 168.8，
        # 每加一次帧/换一次 Epoch 都更糟，而控制器仍在返回 RUN_PRODUCTION。
        # 🔑 [审计 #47] 判据改用共享的 `marginal_gain_stalled()`：至少 3 个点、
        # 基准取**前面各点的中位数**（不是最偏高的 `h[0]`）、末点明显低于基准才算
        # 被证伪。理由见那三个模块级常量的长注释：`min N_eff/g` 不是随采样单调增的量
        # （分子 Kish ESS 小 N 偏高、分母 ĝ 在 N≫τ 前还在涨，两个偏差同向），
        # 旧判据 `max(h[1:]) <= h[0]` 因此对正常窗口误触发，而且与本文件
        # `_unreach` 那段自己否决「按 n_eff 外推提前判死」的论证直接矛盾。
        _hist = view.get("min_n_eff_over_g_history") or {}
        _no_gain = []
        _no_gain_diag: Dict[int, Any] = {}
        for w in W:
            h = [float(v) for _nm, v in (_hist.get(int(w["window_idx"])) or [])]
            _stall, _d = marginal_gain_stalled(h)
            if _stall:
                _no_gain.append(w)
                _no_gain_diag[int(w["window_idx"])] = _d
        _ng_sel = _pick(_no_gain)
        if _ng_sel:
            _h = _hist.get(int(_ng_sel[0])) or []
            # [审计 #47④] 理由里把三个量都带上，让人看得出是**分子没涨**还是
            # **分母 ĝ 在涨** —— `min N_eff/g` 是个比值，只报比值分不出这两种。
            _d = _no_gain_diag.get(int(_ng_sel[0])) or {}
            _why_ng = (
                f"窗口 {_ng_sel} 的主验收量 min N_eff/g **没有随采样上升**"
                f"（逐段 {[(nm, round(float(v), 2)) for nm, v in _h]}；"
                f"基准=前面各点的中位数 "
                f"{_d.get('baseline_median_of_earlier_points')}，末点 "
                f"{_d.get('last')}，比值 {_d.get('last_over_baseline')} < 刹车阈值 "
                f"{_d.get('stall_ratio')}；"
                f"同窗求解器去相关帧数 {_by_idx_for_noop.get(int(_ng_sel[0]), {}).get('solver_n_frames_decorrelated')}，"
                f"求解器侧 g="
                f"{((_by_idx_for_noop.get(int(_ng_sel[0]), {}).get('evidence_decorrelation') or {}).get('solver') or {}).get('statistical_inefficiency')}"
                f"）⟹ "
            )
            # 🔑🔑 [2026-09-14 第三轮复核] **这是唯一一处不查可行性就发 INSERT_LAMBDA 的分支。**
            # 其余每一处发布局动作的地方都先问 `feas.get("insert_lambda") is None`，
            # 只有这里直接发。插点预算耗尽 / 末窗顶到可拆上限时，它发出的是一个
            # **在构造上不可能成功**的动作 —— 执行器要么抛错，要么造出执行层非法的
            # 布局（真机末窗被顶到 K=10~12 就是这个形状）。
            # 归因（加帧已被证伪）是对的，所以不可行时不改归因、只改动作：
            # 如实给 NO_FEASIBLE_ACTION，别假装还有路走。
            if feas.get("insert_lambda") is not None:
                _rw = _bounded_rewindow(
                    int(_ng_sel[0]),
                    _why_ng
                    + "同分布加帧已被本窗口自己的数据证伪，**不得再加**；"
                    f"而插 λ 也不可行（{feas['insert_lambda']}）",
                )
                if _rw is not None:
                    return _rw
                return plan(
                    "NO_ACTION",
                    _why_ng
                    + "同分布加帧已被本窗口自己的数据证伪，**不得再加**；"
                    f"而对症的缩跨度动作也不可行（插 λ：{feas['insert_lambda']}；"
                    f"有界重窗：{self.rewindow_feasible(view, int(_ng_sel[0]))}）⟹ "
                    "如实停下。要继续需要显式加插点预算或改输入 λ 表。",
                    exit_="NO_FEASIBLE_ACTION", windows=_ng_sel,
                    blocked=blocked, earliest=earliest,
                    # [2026-09-17] 加帧被证伪 + 缩跨度不可行，局部
                    halt_scope="TARGET_LOCAL",
                )
            return plan(
                "INSERT_LAMBDA",
                _why_ng
                + "同分布加帧已被本窗口自己的数据证伪，**不得再加**。"
                "瓶颈不是帧数而是这个窗口的跨度：轨迹在不同时间块占据不同 λ 区域，"
                "连续采样再多也摊不平（窗口内不遍历）。"
                "对症动作是**缩小跨度**（model B 插 λ，溢出落末窗），"
                "不是更多帧、也不是再换一份 f_k —— 换 Epoch 同样在同一个跨度上采样。"
                f"⚠️ 这条只在**至少 {MARGINAL_GAIN_MIN_POINTS} 个可比点**时触发"
                "（两点分不出趋势与噪声）。",
                windows=_ng_sel,
            )

        # 5b) 窗口自检已经判出"帧数不够" ⟹ 补采。**这是 (6) 的提前版**：同一个量、
        #     同一个门槛，只是在该窗口刚跑完那一刻就知道了，不用等全部窗口跑完
        #     （实测浪费：5×250k=125 万步烧完才做第一次预算判断）。
        #     verdict 是 INSUFFICIENT_DATA（还没测够）⟹ 加预算，不是换 Epoch。
        # 🔑🔑 [2026-09-14] **物理窗口这一侧也要分"缺帧"和"偏斜"。**
        # `sufficient=False` 混了三项（帧数 / `N_eff/g` / top1%），先前一律补帧 ——
        # 与子窗那边（CTL-02）是同一个毛病，这一处漏了。归因口径复用同一份：
        #   · `top1pct_veto` 或 `HARD_INSUFFICIENT` ⟹ 权重塌缩，**加帧治不了**
        #   · 其余 ⟹ 样本量问题，补帧对症
        # 🔑 [审计 #46] 归因判定与子窗那一侧**共用同一份实现**
        # （`support_failure_is_skew`）。先前这里和 `_read_single_stage` 各写一遍，
        # 而且都只认 verdict、不看 `verdict_source` ⟹ `solver_eligibility`
        # （纯帧数不够，写侧被强制置成 `HARD_INSUFFICIENT`）被判成"加帧治不了"，
        # 于是长 τ 的解耦端窗口被送去插 λ —— 而插 λ 不缩短构象慢模态的 τ_int。
        def _attr_of(w):
            """物理窗口的**三态**归因。与子窗那一侧、与 O1 同一份实现。

            🔑🔑 [2026-09-17 P0] **三态必须贯穿到动作选择层。**
            `support_failure_is_skew()` 把三态压成布尔（`== STRUCTURAL`），于是
            `not is_skew` 会把 `UNKNOWN` 一起算成"样本量类"⟹ 补帧。
            那只是把错误从「错误授权 D3」换成「错误授权补帧」，语义仍然在下一层丢掉。
            """
            return support_failure_attribution(
                w.get("self_verdict"), w.get("self_verdict_source"),
                n_decorrelated=w.get("self_n_frames_decorrelated"),
                min_frames=w.get("self_min_frames"),
                min_n_eff_over_g=w.get("min_n_eff_over_g"),
                n_eff_over_g_target=w.get("self_n_eff_over_g_eligible"),
                frames_headroom=frames_growth_headroom(
                    view, int(w["window_idx"])))

        def _is_skew(w):
            # 帧数一并传进去：判据自己再验一次，不押在写侧的覆盖顺序上。
            # [2026-09-16] 再加「加帧的射程」：比值 × 剩余配额够得着门 ⟹ 这是
            # 样本量问题而不是偏斜，该补帧就补帧（见 `support_failure_is_skew`）。
            return support_failure_is_skew(
                w.get("self_verdict"), w.get("self_verdict_source"),
                n_decorrelated=w.get("self_n_frames_decorrelated"),
                min_frames=w.get("self_min_frames"),
                min_n_eff_over_g=w.get("min_n_eff_over_g"),
                n_eff_over_g_target=w.get("self_n_eff_over_g_eligible"),
                frames_headroom=frames_growth_headroom(
                    view, int(w["window_idx"])))

        _short_all = [w for w in W if w.get("self_sufficient") is False]
        _skew_sel = _pick([w for w in _short_all if _is_skew(w)])
        if _skew_sel:
            _sw = next(w for w in _short_all if int(w["window_idx"]) in _skew_sel)
            _can_split_skew = feas.get("split_tail_window") is None and (
                int(_sw["window_idx"]) == max((int(x["window_idx"]) for x in W),
                                              default=-1))
            _can_ins_skew = feas.get("insert_lambda") is None
            if _can_split_skew or _can_ins_skew:
                return plan(
                    "SPLIT_TAIL_WINDOW" if _can_split_skew else "INSERT_LAMBDA",
                    f"窗口 {_skew_sel} 的自检是**支撑/偏斜类**失败"
                    f"（verdict={_sw.get('self_verdict')}，归因 "
                    f"{_sw.get('self_verdict_source')}）—— **不是样本量不足**。"
                    "同分布加帧治不了偏斜（§5.1 实测 250k→1M 让 top1% 从 0.545 涨到 "
                    "0.762、ESS 比值反而更差）⟹ 对症动作是**缩跨度**。",
                    windows=list(_skew_sel), blocked=blocked, earliest=earliest,
                )
            # 🔑🔑 [2026-09-16 真机 cmet_ligand2/rep1 w4] **停下之前，先看证据是不是
            # 根本没做出来过。**
            #
            # 老板给的链是「判累计 f_k 偏差 → 生成候选 → held-out 验收 → 换 Epoch /
            # 缩跨度」，落在 5a-1（ANALYZE）+ 5a-2。但 5a-1 的 guard 里有一条
            # `self_sufficient is not False` —— 那是**自检侧**的量，而 CTL-11 已裁定
            # 逐窗自检「只看单段帧、对多段窗口系统性偏悲观，**它不是权威**」。
            # 于是一个偏斜类窗口必然 `sufficient=False` ⟹ 拿不到 ANALYZE ⟹
            # `cumulative_fk_residual_production` 永不出现 ⟹ 5a-2 的 guard（要求
            # `cum_fk_verdict ∈ {FAIL, UNMEASURED}`）也永不匹配 ⟹ **整条链对它结构上
            # 不可达**，只能掉到这里，布局动作一不可行就 NO_FEASIBLE_ACTION 收摊。
            #
            # 真机读数：w4 `skipped_windows=[]`、去相关 **69** 帧（min_frames=10）——
            # 求解器**确实**把它算进了 MBAR，证据完全做得出来，只是没人去做。
            #
            # 为什么补在**这里**而不是放宽 5a-1 的 guard：5a-1 排在边际增长判据和所有
            # 布局动作**之前**，放宽它等于把「先 ANALYZE」插到全仓每一条路由前面
            # （实测打断 `加帧被证伪 ⟹ 插 λ` 等既有优先级）。而这里是**死胡同本身**：
            # 已经确认没有任何布局动作可发，ANALYZE 零额外采样、只在现有帧上算，
            # 它要么产出证据让 5a-2 接手，要么算不出来 —— 后者由 `_unsolvable` 与
            # 停滞保护兜底，不会空转。
            if (earliest is not None and int(earliest) in set(int(i) for i in _skew_sel)
                    and _sw.get("cum_fk_verdict") is None
                    and int(earliest) not in set(
                        int(x) for x in (view.get("skipped_windows") or []))):
                return plan(
                    "ANALYZE",
                    f"窗口 {_skew_sel} 是**支撑/偏斜类**失败，加帧治不了；缩跨度也不可行"
                    f"（拆窗：{feas.get('split_tail_window')}；插 λ：{feas.get('insert_lambda')}）。"
                    "但**累计 f_k 偏差证据从未做出来过**（`cumulative_fk_residual_production`"
                    "缺失），而求解器并没有跳过这个窗口（`skipped_windows` 里没有它）"
                    "⟹ 证据算得出来，只是没算。先在**现有帧**上把它算出来（零额外采样），"
                    "再由 held-out 决定是换 f_k 还是缩跨度 —— 在证据缺失时直接判"
                    "`NO_FEASIBLE_ACTION` 是把「没查」说成「无路可走」。",
                    windows=list(_skew_sel), blocked=blocked, earliest=earliest,
                )
            _rw = _bounded_rewindow(
                int(_sw["window_idx"]),
                f"窗口 {_skew_sel} 是**支撑/偏斜类**失败，加帧治不了；而 λ 表上的缩跨度"
                f"也不可行（拆窗：{feas.get('split_tail_window')}；"
                f"插 λ：{feas.get('insert_lambda')}）",
            )
            if _rw is not None:
                return _rw
            return plan(
                "NO_ACTION",
                f"窗口 {_skew_sel} 是**支撑/偏斜类**失败，加帧治不了；而缩跨度也不可行"
                f"（拆窗：{feas.get('split_tail_window')}；插 λ：{feas.get('insert_lambda')}；"
                f"有界重窗：{self.rewindow_feasible(view, int(_sw['window_idx']))}）"
                "⟹ 如实停下，**不拿加帧顶替**。",
                exit_="NO_FEASIBLE_ACTION", windows=list(_skew_sel),
                blocked=blocked, earliest=earliest,
                # [2026-09-17] 偏斜类 + 缩跨度不可行，局部
                halt_scope="TARGET_LOCAL",
            )

        # 🔑🔑 [2026-09-17 P0] **`not _is_skew` ⟹ 改成只认 SAMPLE_SIZE 硬证据。**
        # 先前这里是 `not _is_skew(w)`，而 `is_skew` 只区分两态 ⟹ `UNKNOWN`
        # （归因判不出来：射程三输入缺项 / 老产物缺来源）被算成"样本量类"⟹ 补帧。
        # 这正是用户点名的那句：**只改归因函数不够，下一层会把语义重新丢掉** ——
        # 从「错误授权 D3 停机」换成「错误授权补帧」。
        # `plan()` 的 O1 已在同日因同一理由收紧成只认 SAMPLE_SIZE，这里照同一口径。
        # `UNKNOWN` 的窗口两边都不授权，落到下面的诊断态（见 1c-3 / 兜底）。
        short_self = _pick([w for w in _short_all
                            if _attr_of(w) == SUPPORT_FAILURE_SAMPLE_SIZE])
        if short_self:
            return plan(
                "RUN_PRODUCTION",
                # ⚠️ `sufficient=False` **不只是**"去相关帧数不足"：它同时含
                # `N_eff/g` 与 top1% 两个条件（见 `window_self_support_check`）。
                # 措辞写死成帧数会把一个偏斜问题说成样本量问题。
                f"窗口 {short_self} 的生产后自检判定支撑不足"
                f"（`sufficient=False`；该判据含去相关帧数、N_eff/g 与 top1% 三项，"
                f"逐项读数见自检产物）"
                f"（{[(w.get('self_n_frames_decorrelated'), w.get('self_min_frames')) for w in _short_all]}）。"
                "这不是终态 —— 语义是「这个窗口还需要加采样」（INSUFFICIENT_DATA ≠ FAIL）。"
                "它们的 checkpoint 此刻还热，补采不作废任何已有帧。",
                windows=short_self,
                blocked=blocked, earliest=earliest,
            )

        # 1c-4) 🔑🔑 [2026-09-17 P0] **物理窗口侧的 UNKNOWN 出口。**
        #
        # 三态收紧之后，`_skew_sel`（只收 STRUCTURAL）与 `short_self`（只收
        # SAMPLE_SIZE）都不收 `UNKNOWN` ⟹ 一个自检判失败、归因判不出来的**物理
        # 窗口**两条都不命中，一路掉穿到后面的兜底，并报出一句**假话**：
        # 「所有窗口的预热与生产都已完成」。实测：`earliest=1`、w1/w3 都是 PROBLEM，
        # 而 plan 的 `windows` 是空的。**窗口既没被处理、也没被报出来。**
        # 这正是用户规格那句「不能让它停掉整跑」的反面 —— 它没停跑，它把窗口弄丢了。
        #
        # ⚠️ 与子窗那条（1c-3）同一形状、同一出口。`UNKNOWN` 两边都不授权：
        # 补帧可能越加越差（若其实是结构性），改布局是拿「没测出来」当「测出来是
        # 坏的」（若其实只是数据缺口）。
        # ⚠️ 出口进了 `_RETIRABLE_EXITS`：别处还有活干时退役换窗继续调度，
        # 不因为一个判不出归因的窗口停掉整跑（用户规格原话）。
        # ⚠️⚠️ **只在「没有其他独立硬证据」时才拦。**（用户规格原话）
        # 被求解器跳掉 / 还没有产物 —— 这两类是**独立于归因的硬证据**，它们各自
        # 有更靠后的路由（缺窗、跳窗那几条）。UNKNOWN 分支放在它们前面，不排除
        # 就会**截胡**：实测一个被跳掉的窗口被判成"归因不出来"而退役，路由跑去
        # 修一个下标更小、却没被跳的窗口 —— 正是这条优先级要防的反面。
        # （与子窗那侧 `needs_frames_independent` 同一个道理，我在那边做对了、
        #  在这边漏了。）
        _skipped_idx_now = {
            int(x if not isinstance(x, dict) else x.get("window_index", -1))
            for x in (view.get("skipped_windows") or [])} - {-1}
        _unk_wins = [w for w in _short_all
                     if _attr_of(w) == SUPPORT_FAILURE_UNKNOWN
                     and w.get("solver_skip") is None
                     and int(w["window_idx"]) not in _skipped_idx_now
                     and w.get("has_convergence")]
        _unk_sel = _pick(_unk_wins)
        if _unk_sel:
            _uw = next(w for w in _unk_wins if int(w["window_idx"]) in _unk_sel)
            return plan(
                "NO_ACTION",
                f"窗口 {_unk_sel} 自检判支撑不足，但**归因判不出来**"
                f"（verdict={_uw.get('self_verdict')}，来源 "
                f"{_uw.get('self_verdict_source')}；射程三输入 "
                f"ratio={_uw.get('min_n_eff_over_g')} / "
                f"target={_uw.get('self_n_eff_over_g_eligible')} / "
                f"headroom={frames_growth_headroom(view, int(_uw['window_idx']))}"
                " —— 缺任一就判不了）。"
                "`UNKNOWN` **两边都不授权**：补帧可能是越加越差（若其实是结构性），"
                "改布局则是拿「没测出来」当「测出来是坏的」（若其实只是数据缺口）。"
                "⟹ 停在诊断态。要往下走，先把缺的读数补出来"
                "（写侧 `n_eff_over_g_eligible_threshold` / 块账 / 自检来源），"
                "**不是**挑一个动作试试看。",
                exit_="SUPPORT_ATTRIBUTION_UNKNOWN",
                windows=list(_unk_sel),
                missing=[f"窗口 {int(_uw['window_idx'])} 的支撑失败归因输入"],
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

        # 7) 分析的硬不变量已通过 ⟹ 终态。**注意它不是 `DONE`。**
        #    与分支 0a 同一套口径（那条在路由之前、这条在路由之后兜底）：
        #    `DONE` 的充要条件是 `precision_status == MEETS_CROSS_REPEAT_TARGET`，
        #    单次 run 里恒不成立 ⟹ 实际出口是「分析完整 / 精度未测」。
        if view["stage_analysis_status"] == "ANALYSIS_COMPLETE":
            if view.get("stage_precision_status") == "MEETS_CROSS_REPEAT_TARGET":
                return plan(
                    "DONE",
                    "分析的硬不变量全部通过，且跨重复精度已达标"
                    "（`precision_status = MEETS_CROSS_REPEAT_TARGET`）。"
                    "⚠️ 这不等于答案正确 —— STAGE2_ROOT_CAUSE_2026-08-28.md §2"
                    "「所有收敛门对该失效模式失明」仍然有效（该节被其自身的超越"
                    "声明明确标为仍然有效）。",
                    exit_="DONE_UNTRUSTED" if self.allow_untrusted else "DONE",
                    blocked=blocked, earliest=earliest,
                )
            return plan(
                "NO_ACTION",
                "分析的硬不变量全部通过（`analysis_status = ANALYSIS_COMPLETE`），"
                f"但 `precision_status = {view.get('stage_precision_status')!r}` —— "
                "**精度从未被测过**，只有跨重复才测得出来 ⟹ 这一跑到此为止，"
                "结果**不得作为已验收结果发布**。逐段质量门的读数在 "
                "`stage_quality_gates` 里供人审查（纯报告，不驱动动作）。",
                exit_="ANALYSIS_COMPLETE_PRECISION_UNMEASURED",
                blocked=blocked, earliest=earliest,
            )

        # 8) 窗口都跑完了、stage 分析还没跑 ⟹ 只差分析。
        if not view["has_stage_result"]:
            return plan(
                "ANALYZE",
                "所有窗口的预热与生产都已完成，但 stage 分析还没跑"
                "（stage2_*.json 不存在）⟹ 先跑分析。",
                # ANALYZE 无出口 —— 它是路由，不是终态。
                missing=["stage2_*.json"],
            )

        # 9) 分析没跑完整（`analysis_status = ANALYSIS_INCOMPLETE`）⟹ **如实停下**。
        #
        # 🔑🔑 [2026-09-15 老板定案] **原来的 9b / 9c 已删除，不得重建。**
        # 它们把逐段质量门（`min_overlap` / `target_support_gate` /
        # `min_decorrelated_samples` / `max_endpoint_uncertainty_kJ_mol`）映射成
        # 「缩跨度」「加采样」两类**命令**，由本分支直接发 `SPLIT_TAIL_WINDOW` /
        # `INSERT_LAMBDA` / `IMMUTABLE_REWINDOW` / `RUN_PRODUCTION` —— **烧 GPU**。
        # 那四个阈值全是**未经标定**的拟合值；`max_endpoint_uncertainty_kJ_mol ≤ 1.0`
        # 更是抄自一个**默认关闭**的 early-stop 启发式的默认形参，全仓无任何依据。
        # 而且它是**逐段**量，科学目标却是两腿合成的 `σ_bind`：实测各段
        # `[0.3×5, 2.0]` ⟹ σ_bind = 2.98 已达标，门却判未收敛并命令补帧；
        # 把那一段从 2.0 压到 1.0 要 **4× 采样量、精度提升为零**。
        # 更根本：MBAR 的 σ 是**渐近**估计，对「该采的构型一次都没采到」结构性失明
        # —— 4W53（+32 kJ/mol）与 decharging（−88 kJ/mol）两次大错 σ 都很小、
        # 都没报警。⟹ **未经标定的阈值不得驱动自动补帧。**
        #
        # ⚠️ **不在这里发明新的补帧依据去填这个空。** 走到分支 9 说明：分析的硬
        # 不变量没过（求解器报错 / 丢窗口之类），而所有**有对症动作**的失效形状
        # —— 缺窗（6b）、跳窗（6）、跳子窗与子窗缺帧（1c）、逐窗自检支撑不足
        # （5x 各支）—— 都在**上游**分支，它们一个都没命中。所以这里只剩一个
        # 诚实的终态：如实停下，把全部诊断一起交出去。
        _gate_failures = stage_quality_gate_failures(self._read_stage_result())
        _named = "、".join(
            f"{f['gate']}（实测 {f['value']}，门 {f['threshold']}，"
            f"分类 {f['report_category']}，**仅报告**）"
            for f in _gate_failures
        ) or "（读不出任何具体门的读数）"
        _why = "；".join(view.get("stage_analysis_incomplete_reasons") or []) \
            or "（写侧判了 ANALYSIS_INCOMPLETE 但没给理由）"

        # 🗑️ [2026-09-17，用户拍板 (a)] **这里原有一条「`allow_untrusted` ⟹ DONE_UNTRUSTED」
        # 的分支，是死代码，连同它那段「这是发布策略」的注释一起删掉。**
        #
        # 它**一次都没发出过** `DONE`：`plan()` 的 O4（`action == "DONE"` 且
        # `_evidence_status != CONVERGED` ⟹ 改写成 `NO_ACTION`/`HALT_EVIDENCE_CONTRADICTS_DONE`）
        # **没有 `allow_untrusted` 豁免**，而走到这里 `analysis_status` 必然是
        # `ANALYSIS_INCOMPLETE`（上游已排除 `ANALYSIS_COMPLETE` 与「没有 stage 结果」）
        # ⟹ `_evidence_status` 不可能是 `CONVERGED` ⟹ 必被改写。
        #
        # **不给 O4 开豁免**的理由不止"O4 的判断对"：`ANALYSIS_INCOMPLETE` 表示
        # 缺窗 / 跳窗 / 求解失败这类**硬不变量**没满足，不是"质量门偏低"。即使放开
        # O4，下游 `abfe_pipeline` 的 `analysis_status != ANALYSIS_COMPLETE ⟹ RuntimeError
        # （拒绝标记为 completed）`照样拦死 ⟹ 只会变成"控制器先报完成、再被下游打回"。
        #
        # ⚠️ **`allow_untrusted` 本身没有被削弱**：它仍然改 `trust_level`
        # （→ `OVERRIDDEN_UNTRUSTED`）、仍然在分析**完整**时给出 `DONE_UNTRUSTED`
        # （分支 0a 与分支 7 两处 `exit_="DONE_UNTRUSTED" if self.allow_untrusted else "DONE"`），
        # 也仍然管跳过 rescue / 不进指纹。死掉的只有"分析不完整也放行"这一条。

        # 9a) 唯一的出口：如实停下并报告。**不是 DONE**，也不发任何花 GPU 的动作。
        # ⚠️ 停之前先分清「没有对症动作」与「归因读不出数」（见 `_attribution_blind_spots`）：
        # 两者的措辞一样、处置相反，前者交人工，后者是证据缺失、该去补产物。
        _corrupt, _absent = self._attribution_blind_spots(view)
        _blind_note = (
            ("⚠️⚠️ **但先别当成「无路可走」：有归因读数是非有限值**"
             f"（{_corrupt}）。上游分支拿 NaN 做的每一个比较都是 False ⟹ "
             "「一条都没命中」**在这里不是结论，是读到了坏数**。"
             "该做的是查这些读数为什么算成 NaN/inf，不是交人工判定无路可走。"
             if _corrupt else "")
            + (f"（另有尚未算出的归因读数，属正常：{_absent}）" if _absent else "")
        )
        return plan(
            "NO_ACTION",
            f"分析未跑完整（`analysis_status = ANALYSIS_INCOMPLETE`）：{_why}。"
            "有对症动作的失效形状都在上游分支、一个都没命中 ⟹ 这里**没有**"
            "一个在科学上站得住的下一步动作。编一个出来会退化成「再跑一次直到"
            "碰巧通过」，回去跑 ANALYZE 则是空转（分析已经跑过）⟹ 交人工。"
            + _blind_note
            + "⚠️ 逐段质量门**不再**是动作依据（未经标定的阈值，2026-09-15 定案），"
            f"它们的读数只作报告：{_named}。"
            "完整条目见返回值的 `stage_quality_gates`（每条带 `drives_action: False`）。",
            exit_="NO_FEASIBLE_ACTION",
            missing=(
                [f"窗口 {r['window_idx']} 的归因读数是非有限值："
                 f"{'、'.join(r['non_finite'])}" for r in _corrupt]
                or (view.get("stage_analysis_incomplete_reasons") or
                    ["stage 分析的硬不变量读数"])
            ),
            blocked=blocked, earliest=earliest,
            # [2026-09-17] 分析未跑完整，且上游每一条有对症动作的分支都没命中 ⟹ 整条 stage 无路，换窗也没用
            halt_scope="STAGE_GLOBAL",
        )

    # 兜底分支要区分「没有对症动作」和「归因函数读不出数」时看的那几个读数 ——
    # 就是上游各分支真正据以分岔的量，不是随便挑的。
    _ATTRIBUTION_INPUTS = (
        ("phase", "窗口相 phase"),
        ("self_verdict", "逐窗自检结论 self_verdict"),
        ("cum_fk_verdict", "累计 f_k 残差判定 cum_fk_verdict"),
        ("min_n_eff_over_g", "主验收量 min N_eff/g"),
        ("warmup_steps_left", "预热预算余额 warmup_steps_left"),
    )

    @classmethod
    def _attribution_blind_spots(cls, view: Dict[str, Any]):
        """落到兜底时，逐窗把归因读数分成**坏数**和**没算**。返回 `(corrupt, absent)`。

        🔑🔑 [2026-09-17，死线 D8] **「没有对症动作」与「归因读不出数」是两件事，
        处置相反。** 兜底的措辞说的是前者（交人工），而真机出现过后者 ——
        `converged=False` + 零条归因，真相是某道门的读数是 NaN（审计 #59 已修
        `stage_quality_gate_failures` 的 fail-open，但**兜底本身仍然分不出这两种**）。

        两类要分开，别混成一个"读不出来"：
          · **corrupt**（非有限：NaN / inf）⟹ **真缺陷**。上游分支拿它做比较时
            每一个比较都是 False（`nan >= x`、`nan < x` 全假）⟹ 静默地一条都不命中，
            长得和"确实没有对症动作"一模一样。必须大声报。
          · **absent**（None）⟹ **正常**。很多窗口本来就还没有 `cum_fk_verdict`
            （证据没产出），那恰恰是上游 5a-1 发 ANALYZE 的依据。只作附注。

        ⚠️ 不做「全部读数都读不出来 ⟹ 全盲」那种判据：有产物的窗口 `phase` 恒可导出，
        全盲在构造上不可达，写了就是死支。
        ⚠️ 这**不新增任何动作**：D8 是设计上正确的兜底，给它配动作会退化成
        「再跑一次直到碰巧通过」。这里只改报告 —— 让停下来的理由说的是真相。
        """
        corrupt, absent = [], []
        for w in (view.get("windows") or []):
            _bad, _none = [], []
            for key, label in cls._ATTRIBUTION_INPUTS:
                v = w.get(key)
                if v is None:
                    _none.append(label)
                elif isinstance(v, float) and not np.isfinite(v):
                    _bad.append(f"{label}={v}")
            if _bad:
                corrupt.append({"window_idx": int(w["window_idx"]),
                                "non_finite": _bad})
            if _none:
                absent.append({"window_idx": int(w["window_idx"]),
                               "not_yet_computed": _none})
        return corrupt, absent

    @staticmethod
    def first_untrusted_window(view: Dict[str, Any]) -> Optional[int]:
        """**第一个不可信窗口**的下标 —— tail-only 重分的起点，由控制器自己算。

        [老板] 「当前从 `first_untrusted_window` 开始重分；**不是让用户手选 anchor**。」
        判据就是主验收量：verdict 不是 `ANALYSIS_ELIGIBLE` 的第一个窗口。
        （本例是 win3，也是第一个 `min(N_eff/g) < 10` 的 —— 人选的 anchor 恰好同解，
        但**机制必须是自动的**。）

        🔑🔑 [2026-09-14 第三轮复核] **「证据被我们自己作废」也是不可信，不是"还不知道"。**

        先前判据只有 `v is not None and v != ANALYSIS_ELIGIBLE` —— 那个
        `v is not None` 是对的（没跑过自检 ≠ 有问题，设计 §4 的三态），但它把
        **布局变更作废掉证据**的窗口一起漏掉了。设计 §4 同一节明写：插 λ / 拆窗
        之后旧证据描述的是另一个窗口几何、已被剔除，那是**确定的未解决**，
        必须判 PROBLEM —— `_window_state()` 就是这么写的，本函数却另写了一套。

        真机 cyclod_ligand1/rep3：win0–3 全 ANALYSIS_ELIGIBLE，win4 的产物被布局
        变更作废、一份证据都没有（`self_verdict=None`、`stale_layout_evidence_only`）
        ⟹ 本函数返回 None ⟹ anchor 取不到 ⟹ 末窗 K=11 > hi=8，
        `_legalize_tail_window` 在启动布局校验处 fail-closed 抛错，
        **decide() 一次都轮不到、整个 run 起不来**。
        判据与 `_window_state()` 对齐即可，不新发明规则。
        """
        for w in sorted(view.get("windows") or [], key=lambda x: int(x["window_idx"])):
            if w.get("stale_layout_evidence_only"):
                return int(w["window_idx"])
            v = w.get("self_verdict")
            if v is not None and v != "ANALYSIS_ELIGIBLE":
                return int(w["window_idx"])
        return None

    @staticmethod
    def stale_layout_windows(view: Dict[str, Any]) -> set:
        """产物描述的是**另一套 λ 布局**的窗口号。

        🔑 [2026-09-16] 这个集合有**两个**消费者（`decide()` 的 1d-0 分支、
        主循环停滞保护的降级闸），所以只能有一份实现 —— 它判的是
        「拿已有帧重解会不会维度不符」，而 `ibs_engine` 的 loader 对同一件事
        是 fail-closed 抛 `ValueError`。两边一旦分岔，就是一个在构造上不可能
        成功的动作被发出去、炸穿流水线（真机 7 个 run）。
        """
        return {
            int(x) for x in (view.get("stale_layout_evidence") or {})
        } | {
            int(w["window_idx"]) for w in (view.get("windows") or [])
            if w.get("stale_layout_evidence_only")
        }

    def tail_repartition_anchor(self, view: Dict[str, Any]) -> Optional[float]:
        """`first_untrusted_window` 的**首态** λ —— 它就是与前一窗共享的那个节点。

        冻结的是「anchor 之前」的窗口，所以从**不可信窗口自己的首态**切，
        才能把这个已知弱的窗口一起重分掉；若从它的**末态**切，会把它永久保留。
        """
        idx = self.first_untrusted_window(view)
        if idx is None or idx <= 0:
            return None
        path = view.get("path") or {}
        ranges = path.get("window_ranges")
        if not ranges or idx >= len(ranges):
            return None
        # 🔑🔑 [2026-09-14 真机] 这里原来读 `w["lambda_vdw_hi"]` —— 全仓**零处写**
        # 这个键（只有这一行读它）。于是 anchor 恒为 None，执行器每次都走
        # 「取不到 tail anchor，跳过本动作」：`SPLIT_TAIL_WINDOW` 自诞生起
        # **一次都没有真正执行过**。cyclod_ligand1/rep2 发了 4 次、版本链上
        # 一条 tail_repartition 都没有，最后靠通用停滞探测判 NO_FEASIBLE_ACTION。
        # anchor 的定义本来就是「不可信窗口自己的首态」= `lam[ranges[idx][0]]`，
        # 布局和 λ 表都在 `view["path"]` 里，不需要第二个字段来搬运它。
        lam = path.get("lambdas_vdw") or []
        start = int(ranges[idx][0])
        if start >= len(lam):
            return None
        return float(lam[start])

    def _pilot_for_partition(self):
        """(pilot_lambdas, metric_g)；读不到返回 (None, None)。

        与 `abfe_pipeline._load_pilot_for_path_evolution` 读的是同一份文件的
        同一批字段 —— 尾段重分要和生产用**同一个**分窗判据，就得拿到同一份 pilot。
        """
        diag = (self._json(os.path.join(
            self.path_checkpoint_dir, "preopt_dual_vanishing.json"
        )) or {}).get("path_diagnostics") or {}
        pl, mg = diag.get("pilot_lambdas"), diag.get("metric_g")
        return (list(pl) if pl else None, list(mg) if mg else None)

    def _dry_run_tail_repartition(self, view: Dict[str, Any]):
        """把 `SPLIT_TAIL_WINDOW` 的执行器跑一遍（纯计算，不落盘）。

        可行性与执行共用**这一次调用的结果**：`feasible()` 用它判可不可行，
        执行器用它返回的 ranges 落地。两边不再各切一遍。
        """
        path = view.get("path") or {}
        lam = [float(x) for x in (path.get("lambdas_vdw") or [])]
        ranges = [tuple(int(i) for i in r) for r in (path.get("window_ranges") or [])]
        anchor = self.tail_repartition_anchor(view)
        if not lam or not ranges or anchor is None:
            raise RuntimeError("读不到 λ 表 / 布局 / anchor")
        a_idx = int(ranges[int(self.first_untrusted_window(view))][0])
        m0 = sum(1 for a, _b in ranges if a >= a_idx)
        pl, mg = self._pilot_for_partition()
        new_ranges, diag = repartition_tail_from_anchor(
            lam, ranges, float(anchor),
            min_states_per_window=self.lo,
            max_states_per_window=self.hi,
            # 🔑 目标窗口数 = 原尾段窗口数 + 1。这就是"拆细"的定义，
            # 也是可行性判据问的那个 m ≥ m0+1 里最小的那个 m。
            n_windows=m0 + 1,
            pilot_lambdas=pl,
            metric_g=mg,
            partition_criterion=self.partition_criterion,
        )
        return [list(r) for r in new_ranges], diag

    @staticmethod
    def _epoch_validation_unaffordable(
        view: Dict[str, Any], windows: Sequence[int], *, new_epoch: bool = False
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

        🔑🔑 [审计 #33，2026-09-14] **两类动作的门槛不是同一个数。**

        `new_epoch=False`（`RECALIBRATE_FK` / `PROBE_REANCHOR_EPOCH`）：只**重解**
        已有帧、不重学 f_k ⟹ 要的就是首档冻结验证额度（50k）。

        `new_epoch=True`（`RELEARN_FK_EPOCH`）：执行器要从头 **LEARN**
        ⟹ 真实需求是 `relearn_epoch_required_steps()` = learn + burn-in + 首档，
        ≥ 140k、真机 cyclod_ligand1/rep2 win3 实测 **290k**。先前这里一律拿 50k
        当门槛 ⟹ **控制器批准、执行器拒绝并终止整个循环**（而不是换个动作），
        方向是危险那一侧：批准一个跑不完的 Epoch，把预算烧光再半路死掉。
        两个数都写进返回值，便于事后对账。
        """
        try:
            import ibs_engine as _ie
            floor_first_rung = int(_ie.FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0])
        except Exception:
            return None  # 拿不到阈值就不假装判断（fail-open：只影响诊断）
        by_idx = {int(w["window_idx"]): w for w in (view.get("windows") or [])}
        broke = []
        for i in windows:
            w = by_idx.get(int(i))
            if w is None:
                continue
            # 全新 Epoch 的门槛按**这个窗口自己**的账本自校准（learn 步数逐窗不同）。
            floor = (int(relearn_epoch_required_steps(w)) if new_epoch
                     else int(floor_first_rung))
            left = w.get("warmup_steps_left")
            # ⚠️ **预算未知 = 不可行**（fail-closed）。以前 `left is None` 会跳过检查，
            # 于是"账本读不到"的窗口能一路走到重标定 —— 而真实零预算窗口
            # （555k/555k）正是这样漏过去的。不知道有没有钱，就不许开新 Epoch。
            if left is None:
                broke.append({
                    "window_idx": int(i), "warmup_steps_left": None,
                    "min_epoch_validation_reservation": floor,
                    "first_rung_only_reservation": int(floor_first_rung),
                    "relearn_epoch_required_steps": int(
                        relearn_epoch_required_steps(w)),
                    "reason": "budget_unknown_fail_closed",
                })
            elif int(left) < floor:
                broke.append({
                    "window_idx": int(i),
                    "warmup_steps_left": int(left),
                    "min_epoch_validation_reservation": floor,
                    "first_rung_only_reservation": int(floor_first_rung),
                    "relearn_epoch_required_steps": int(
                        relearn_epoch_required_steps(w)),
                })
        if not broke:
            return None
        return {
            "requires_new_learning": bool(new_epoch),
            "windows": broke,
            "source": (
                "relearn_epoch_required_steps()（learn+burn-in+首档）"
                if new_epoch
                else "ibs_engine.FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0]（只重解）"
            ),
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

    @staticmethod
    def _coverage_incomplete(view) -> List[Any]:
        """**求解覆盖有没有缺口**：缺窗 / 跳窗 / 跳子窗，返回缺口清单（空 = 完整）。

        🔑🔑 [2026-09-14] **子窗被跳与物理窗口被跳是同一条硬约束，不是类比。**
        `skipped_windows` 拆成物理窗口 + `skipped_sampling_units` 之后，所有拿它
        当 DONE 闸的读点就只覆盖物理窗口了 —— 那是**漏网**。被求解器踢出协方差链的
        子窗意味着**它那段 λ 没进最终的 MBAR**，交出去的是缺了一段的和，而
        「缺窗口的和不是 ΔG，是另一个量」在本仓是硬约束。
        子窗缺了还**更严重**：子窗是 immutable rewindow 用来**接管父窗那段 λ** 的，
        父窗已经退出求解覆盖、不会自动顶回来 ⟹ 那段 λ 直接没人管。

        ⚠️ 抽成一份的理由与 `_layout_trustworthy` 完全相同：分支 0a 与
        `_evidence_status` 各写一份、不同步，已经造成过一次真 regression。

        ⚠️ **`production_rescue_targets` 维持只看物理窗口**（别"修"它）：
        production rescue 是按物理窗口目录组织的旧机制（`vanishing_rescue/<plan_id>`），
        对子窗**在构造上跑不动**。给子窗补帧的对症动作是带 `unit_id` 的
        `RUN_PRODUCTION`（走 `_topup_rewindow_child` 续跑它自己的 checkpoint），
        由分支 1c 接住 —— 两条路，不是一条路的两半。
        """
        out: List[Any] = []
        # 🔑 [2026-09-15 审计④] `out_of_range_windows` 一并算缺口：证据里混进了
        # 当前布局根本没有的窗口 ⟹ 这份"完成"描述的不是当前这条路径。
        for k in ("missing_windows", "skipped_windows", "skipped_sampling_units",
                  "out_of_range_windows"):
            for x in ((view or {}).get(k) or []):
                out.append((k, x))
        return out

    @staticmethod
    def _layout_trustworthy(view) -> bool:
        """这份 stage 结论（`analysis_status` / `precision_status`）描述的**是不是
        当前布局**。

        🔑🔑 [2026-09-14] **两处共用一份，别各写一遍。**
        分支 0a 与 `_evidence_status` 原来各自写了一份版本前置，而分支 7 一份都没有
        ⟹ 老产物上走出 `action=DONE` + `evidence_status=INCONCLUSIVE` 这种自相矛盾
        的结局（`tests/test_stage2_repair_controller.py::
        test_converged_stage_is_done_but_not_claimed_correct` 钉的就是它）。

        判据放宽的理由：版本核实只在**布局演化过**时才是必要条件。路径链从没演化过
        （当前版本 ≤ 1，没插过 λ 也没拆过窗）时根本不存在"另一条布局"，而 stage 缓存
        `stage2_vanishing.json` 历史上不盖 `path_version` ⟹ **所有既有 run 恒为
        False**，CTL-11 对它们完全是死的。
        「布局演化过 + 结果没盖版本号」仍然 fail-closed —— 那才是真风险。
        """
        _pv_now = int((view or {}).get("path_version") or 1)
        return bool((view or {}).get("stage_result_path_version_verified")
                    or _pv_now <= 1)

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
        # 🔑 [审计 #28] 先前只映了前两个，漏了后两个 —— 而它们同样是"预算耗尽"，
        # 于是被报成 `INCONCLUSIVE`（"结果差一点"），与三行之上自己写的规则
        # 「预算耗尽**永远**是还没测够」直接矛盾。
        # `HALT_FRAMES_ADMISSION_CAP`（审计 #30 新增）同理：块数配额用完 =
        # 还没测够，不是测出来不合格。
        if exit_ in ("HALT_BUDGET", "HALT_LOCAL_VALIDATION_CAP",
                     "GLOBAL_BUDGET_EXHAUSTED",
                     "HALT_VALIDATION_BUDGET_UNREACHABLE",
                     "HALT_FRAMES_ADMISSION_CAP"):
            return "INSUFFICIENT_DATA"
        # [审计 #58] 缺口判据与分支 0a 的 DONE 闸**共用一份**（见
        # `_coverage_incomplete` 的长注释）：缺窗 / 跳窗 / 跳子窗都是覆盖有缺口。
        if self._coverage_incomplete(view):
            return "INSUFFICIENT_DATA"
        # 🔑🔑 [2026-09-15 老板定案] **`CONVERGED` 的充要条件是跨重复精度达标。**
        # 先前这里是 `stage_converged is True ⟹ CONVERGED` —— 那个 `converged`
        # 只是四个**未经标定**阈值的合取，把「算出了数」误报成「精度已验收」。
        # 现在两维分开判：
        #   · `analysis_status == ANALYSIS_COMPLETE` 只说**分析跑完整了**
        #     （硬不变量），**不得**据此报 CONVERGED；
        #   · 只有 `precision_status == MEETS_CROSS_REPEAT_TARGET` 才是
        #     CONVERGED —— 单次 run 里恒不成立（一个 run 无法自证精度）。
        # ⚠️ 版本没核实过 / 有证据被布局变更作废时仍然一票否决（那份结论描述的
        #   可能是另一条布局；同分支 0a 的补丁，理由写在那里）。
        if (self._layout_trustworthy(view)
                and not view.get("stale_layout_evidence")
                and view.get("stage_analysis_status") == "ANALYSIS_COMPLETE"):
            if view.get("stage_precision_status") == "MEETS_CROSS_REPEAT_TARGET":
                return "CONVERGED"
            # 分析完整、精度未测 ⟹ **自己的词**。既不是 CONVERGED（没验收过），
            # 也不是 INSUFFICIENT_DATA（不缺数据，是这一跑测不了精度），
            # 更不是 INCONCLUSIVE（不是"差一点"）。下游拿它跟 "CONVERGED"
            # 比字符串会**明确地**不相等 —— 这正是要的 fail-loud。
            return "PRECISION_UNMEASURED"
        # 任一窗口的主验收量（N_eff/g）判不可分析 ⟹ 整条路径的证据就不够。
        if any(
            w.get("self_verdict") in ("HARD_INSUFFICIENT", "INSUFFICIENT_DATA")
            for w in view["windows"]
        ):
            return "INSUFFICIENT_DATA"
        # 🔑 [2026-09-14 新发现] **f_k 证据本身「还没测出来」也是 INSUFFICIENT_DATA。**
        # 先前这里只看**自检**的 `self_verdict`，于是一个还在预热/验证、
        # `f_k_evidence_status ∈ {calibrated, indeterminate, none}` 的窗口
        # （`_VERDICT` 把它们映成 `INSUFFICIENT_DATA`，字面意思就是"尚不可测"）
        # 会被报成 `INCONCLUSIVE`（"结果差一点"）。这与 `_VERDICT` 那张表自己的
        # 口径矛盾，而且 INCONCLUSIVE 容易被下游读成"测出来不合格"。
        # 先前它没暴露，只是因为这类窗口总会顺带拿到一个 `HALT_BUDGET` 出口。
        if any(w.get("verdict") == "INSUFFICIENT_DATA" for w in view["windows"]):
            return "INSUFFICIENT_DATA"
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
        # `**kwargs` 原样透传给 `__init__`，`effective_config` 也走这条路
        # （审计 #1：溶剂腿的 run_dir 底下没有 run_provenance.json）。
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
        """同一物理 stage 的所有段目录名，按**段号**升序（基准段在最前）。

        [审计 #63] 「什么算一个段」原来在三处各写一遍、规则互不一致
        （一处不过滤空壳 + 字符串序，一处过滤空壳 + 整数序，一处用
        `rsplit("_",1)[-1]` 会把 rewindow 目录误判成段）。现在段号解析统一走
        `segment_index_of_dir()`，排序一律按整数；**是否过滤空壳**由调用点决定，
        两个调用点的理由分别写在各自位置：
          · 本函数（合并视图）**要过滤** —— 空转循环每轮建一个新段目录，真机
            rep2 留下 39 个只有占位文件、一份 convergence 都没有的目录；把它们
            当采样段会让每次 read() 白扫一遍，也会让"证据来自哪个段"被搅浑。
          · `existing_segment_names`（写路径版本记录用）**不过滤** —— 它记的是
            「变更前盘上有哪些段目录」，那是一份事实清单，不是证据来源。
        """
        base = getattr(self, "_stage_base", self.stage_name)
        stage_dir = os.path.join(self.run_dir, base)
        names = []
        for d in glob.glob(stage_dir + "*"):
            if not os.path.isdir(d):
                continue
            n = segment_index_of_dir(d, stage_dir)
            if n is None:
                continue
            if n > 1 and not glob.glob(os.path.join(
                    d, f"dual_window_*_{self.stage_type}_convergence.json")):
                continue          # 空壳段不算段（理由见 docstring）
            names.append((int(n), os.path.basename(d)))
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
                # 见 __init__ 里 `_explicit_config` 的说明：不传这份，
                # 合并视图的 cap / 块大小 / 分窗判据全部退回默认。
                effective_config=getattr(self, "_explicit_config", None),
            )
            # 🔑🔑 [审计 #25] **段级子控制器必须知道自己的物理 stage 名。**
            # 不设 `_stage_base`，`_read_stage_result()` 就用 `self.stage_name`
            # （= `vanishing_3`）去格式化候选名 ⟹ 找 `stage2_vanishing_3_*.json`，
            # 而写侧硬编码写的是 `stage2_vanishing_*` ⟹ 两个候选**全 miss**，
            # 落到 mtime 兜底 glob，CTL-01 的「路径版本必须匹配当前布局」这层隔离
            # 对所有非基准段是关着的。段号只改窗口产物目录，stage 结果是全局的。
            sub._stage_base = getattr(self, "_stage_base", self.stage_name)
            views.append((nm, sub.read()))
        base_view = dict(views[0][1])

        # 🔑 [2026-09-12] **先按 λ 身份剔除过期证据，再谈段号。**
        # 原来只按"段号最大且有 convergence"取，等于拿**段号当新旧的代理**。
        # 那在"段只往前开、基准段从不重跑"时成立；但循环现在会往基准段写
        # （补采按证据来源路由）、布局变更时整段重跑 —— 基准段反而常常最新。
        # 真机后果：win4 在基准段已经 43.25 通过，却被 45 分钟前废弃 Epoch 里
        # 那份 1.62 盖住，于是循环接着去"修"一个已经好了的窗口。
        #
        # 判据不是 mtime（脆弱），是**语义身份**：这份证据是在哪套 λ 上产出的。
        # 对不上当前布局 ⟹ 它描述的是另一个窗口几何，直接丢弃。
        _path_now = self._read_path()
        _cur_lam = list(_path_now.get("lambdas_vdw") or [])
        _cur_rng = [tuple(r) for r in (_path_now.get("window_ranges") or [])]

        # 🔑 [2026-09-17] 判定搬进模块级 `classify_layout_evidence()`，与单段视图
        # **同一份**。先前它是本函数的一个闭包，于是单段视图整套判定都没有。
        _cls = classify_layout_evidence(
            [(nm, v.get("windows") or []) for nm, v in views], _path_now)
        unverifiable = _cls["unverifiable"]   # 有证据但核不了布局（缺 λ）
        out_of_range = _cls["out_of_range"]   # 当前布局里根本没有的窗口号

        merged: Dict[int, Dict[str, Any]] = {}
        provenance: Dict[int, str] = {}
        stale: Dict[int, List[str]] = _cls["stale"]
        stale_tpl: Dict[int, Dict[str, Any]] = {}
        for nm, v in views:          # 段号升序 ⟹ 后面的覆盖前面的
            for w in v.get("windows") or []:
                i = int(w["window_idx"])
                # 只有 `STALE` 不参与合并；`OUT_OF_RANGE` / `UNVERIFIABLE` 照旧合并
                # 并已被 helper 记账（判死会重现 win4 占位记录死锁）。
                if _cls["verdict_by"].get((nm, i)) == "STALE":
                    stale_tpl[i] = w          # 留作占位模板：键齐全
                    continue
                if i not in merged or w.get("has_convergence"):
                    merged[i] = dict(w, segment=nm)
                    provenance[i] = nm
        # 🔑🔑 [审计 #18，2026-09-14] **`stale` 表必须清除，不然 DONE 永远到不了。**
        #
        # 先前两张表互不排斥：窗口在新段里**重采成功并进了 `merged`** 之后，它在旧段
        # 留下的那条过期证据仍然躺在 `stale` 里。于是插过一次 λ 之后
        # `view["stale_layout_evidence"]` **恒非空**，而分支 0a 的 `DONE` 与
        # `_evidence_status` 的 `CONVERGED` 都要求它为空 ⟹ **两条收敛路径被永久封死**，
        # 一个已经修好的 stage 每次 resume 都会重新开修复动作烧 GPU。
        # `stale` 的语义只能是「这个窗口**当前没有任何有效证据**」——
        # 有了有效证据就不再是 stale，旧段那份只是历史。
        # （`stale_layout_evidence_only` 占位记录的语义不变：它恰恰是"清不掉"的那些。）
        for _i in list(stale):
            if _i in merged:
                del stale[_i]
                stale_tpl.pop(_i, None)
        # ⚠️ **证据全过期 ≠ 窗口不存在。** 直接把它从列表里剔掉，窗口就"消失"了，
        # 决策会越过它去处理下游 —— 正是要防的那个失败模式（跳过前面的窗口）。
        # 正确语义是「这个窗口目前**没有有效证据**」：留一条占位记录，
        # `self_verdict=None` ⟹ 三态分类判 UNKNOWN ⟹ 动作是**产出证据**，
        # 而且它仍然占着自己的位置，earliest 排序不会跳过它。
        for i, segs in stale.items():
            if i in merged:
                continue
            # 用被剔掉的那份当模板（键齐全），把**所有证据字段清空** ——
            # 留结构、不留结论。
            tpl = dict(stale_tpl.get(i) or {})
            for k in list(tpl):
                if k not in ("window_idx",):
                    tpl[k] = None
            tpl.update({
                "window_idx": int(i),
                "segment": None,
                "has_convergence": False,
                "has_warmup_failure": False,
                "stale_layout_evidence_only": True,
                "stale_segments": list(segs),
            })
            merged[i] = tpl
            provenance[i] = "(证据已被布局变更作废)"
        windows = [merged[i] for i in sorted(merged)]

        expected = base_view.get("n_windows_expected")
        missing = (
            [i for i in range(int(expected)) if i not in merged] if expected else []
        )
        # [审计 #58] 物理窗口与子窗分两个键；单段视图已经拆好，这里只合并。
        skipped = sorted({
            int(x) for _nm, v in views for x in (v.get("skipped_windows") or [])
        })
        skipped_units = sorted({
            str(x) for _nm, v in views
            for x in (v.get("skipped_sampling_units") or [])
        })
        # [2026-09-12] 同一窗口**跨段**的主验收量轨迹。cap 分支早就写明
        # 「块后立即用 N_eff/g 的边际增长判 —— 边际停滞/下降则停止同分布加帧」，
        # 但判这件事需要的历史此前根本没进视图，于是规则一直是死的。
        # ⚠️ **布局变更会作废这段历史。** 插 λ / 拆末窗之后，旧段描述的是**另一个
        # 窗口几何**，它们的 min N_eff/g 不能再拿来判"加帧有没有用"。真机 15:59
        # 实证：第一次插完还没产出任何新证据，判据读着旧轨迹又插了一次，
        # 两次把末窗顶到 K=6 越界炸掉。
        # 当前路径版本记录里的 `segments_before_change` 就是变更前已有的段。
        # [2026-09-17] 判定搬进 `layout_change_invalidated_segments()`，与单段视图
        # 同一份 —— 它是全局事实，两个构造器算出来必须一样。
        _stale_segs = set(self.layout_change_invalidated_segments())
        # ⚠️ [契约 B] `_production_blocks_*` 现在返回 `{"by_window", "by_unit"}`
        # **两本账**，必须在这里拆开放进两个视图键 —— 直接把整个双层字典塞进
        # `production_blocks_by_window`，消费侧的 `.get(int(w_idx))` 会恒取到
        # None，块数硬上限对**所有**窗口静默失效（比原 bug 更糟）。
        _agg_units = base_view.get("sampling_units") or []
        _agg_same = self._production_blocks_ledger(
            windows, (_path_now or {}).get("version"), units=_agg_units)
        # 🔑 [BUD-03] 跨段累计的块数（硬上限用），与上面那本同源不同过滤。
        _agg_all = self._production_blocks_total_by_window(
            windows, (_path_now or {}).get("version"), units=_agg_units)
        blocks_by_window = _agg_same["by_window"]
        blocks_by_unit = _agg_same["by_unit"]
        blocks_total_by_window = _agg_all["by_window"]
        blocks_total_by_unit = _agg_all["by_unit"]

        # 🔑🔑 [S2-C，2026-09-14] **边际增长判据需要两个可比点，而按段建史给不出。**
        #
        # 原来这里一个采样段只贡献**一个**点 ⟹ 单段窗口永远只有 1 个点，
        # `len(h) >= 2` 从不成立 ⟹「同分布加帧已被证伪就别再加」这道唯一的刹车
        # **对单段窗口从不触发**（而单段正是最常见的情形）。布局一变旧段被标 stale
        # 丢掉，多段窗口也会退回 1 个点。
        #
        # 补法是读**已经在写的**逐轮快照（`stage2_autonomous_history.json` 的
        # `snapshot`，主循环每轮决策时落的盘），它天然是**逐块**的。
        # 可比性由两个身份保证，缺一不可：
        #   · `path_version` 相同 —— 布局变了跨度就变了，前后不是同一个量；
        #   · `segment` 相同     —— 换段 = 换了一份 f_k，重加权口径不同。
        # 排序按 `production_steps`（采样量单调），去重也按它。
        support_history: Dict[int, List[Any]] = {}
        _hist_pts: Dict[int, Dict[Any, Any]] = {}

        def _add_pt(i, seg, steps, value, label):
            # ⚠️ 可比性由 (segment, production_steps) 共同决定：**换段 = 换 f_k**，
            # 不同段的 min N_eff/g 不是同一把尺子，不能放进同一条边际增长曲线。
            # 下面排序/比较只在**同一段**内部才有意义 —— 混段的点会让
            # "加帧有没有用"这个判据读出一个根本不存在的趋势。
            if value is None or seg is None:
                return
            key = (seg, int(steps or 0))
            _hist_pts.setdefault(int(i), {}).setdefault(
                key, (int(steps or 0), label, float(value))
            )

        for nm, vv in views:
            if nm in _stale_segs:
                continue          # 变更前的几何，不参与边际增长判定
            for w in vv.get("windows") or []:
                _add_pt(w["window_idx"], nm, w.get("production_steps"),
                        w.get("min_n_eff_over_g"), nm)

        try:
            _auto = self._json(os.path.join(
                self.path_checkpoint_dir, "stage2_autonomous_history.json")) or {}
            _cur_pv = (_path_now or {}).get("version")
            for _it in (_auto.get("iterations") or []):
                if _cur_pv is not None and _it.get("path_version") != _cur_pv:
                    continue      # 别的布局下的点，不可比
                for _sn in (_it.get("snapshot") or []):
                    _seg = _sn.get("segment")
                    if _seg in _stale_segs:
                        continue
                    _add_pt(_sn.get("window_idx"), _seg,
                            _sn.get("production_steps"),
                            _sn.get("min_n_eff_over_g"),
                            f"{_seg or 'base'}@{int(_sn.get('production_steps') or 0)}")
        except Exception:          # noqa: BLE001 —— 历史读不到只是少几个点，不阻断决策
            pass

        for i, pts in _hist_pts.items():
            # **按段分组，只在段内按采样量排序**；把各段首尾串起来会把
            # "换了一份 f_k" 伪装成 "加帧的效果"。
            # 🔑 [CTL-06，2026-09-14] 判据来源必须是**当前段**（这个窗口证据实际
            # 所在的那一段，也就是最新那份 f_k），**不是"点数最多的段"** ——
            # 旧段点数多时会拿旧 f_k 的趋势去裁决新段该不该继续加帧。
            # 当前段不足两点就不判（`len(h) >= 2` 自然不成立），不许拿别的段凑。
            by_seg: Dict[Any, List[Any]] = {}
            for (seg, steps), (_st, label, value) in pts.items():
                by_seg.setdefault(seg, []).append((steps, label, value))
            _cur_seg = (merged.get(int(i)) or {}).get("segment")
            best = by_seg.get(_cur_seg)
            if best is None:
                # 读不到当前段（证据全过期等）⟹ 宁可不判，也不拿别的段代替。
                best = []
            support_history[i] = [
                (label, value) for _steps, label, value in sorted(best, key=lambda t: t[0])
            ]
        # 跨段实际生产消耗：每个窗口在**每个段**里都烧过 GPU，只取胜出段会漏账。
        # 🔑 [CTL-04②，2026-09-14] **消耗未知不得记作零。**
        # 先前 `int(... or 0)` 把"读不到步数"和"确实是 0 步"压成同一个数 ——
        # 与"上限未知 ≠ 零"是同一个错、方向相反：一个会虚报余量、一个会虚报耗尽。
        # 读不到就**登记为未知**，账本据此把整个 stage 的用量标成 unknown，
        # 而不是悄悄少算一块。
        _per_win_all_seg: Dict[int, int] = {}
        _unknown_usage: List[Any] = []
        for _nm, _vv in views:
            for _w in (_vv.get("windows") or []):
                _i = int(_w["window_idx"])
                _ps = _w.get("production_steps")
                if _ps is None:
                    # 有这个窗口的证据、却读不到它烧了多少步 ⟹ 未知，不是 0。
                    if _w.get("has_convergence"):
                        _unknown_usage.append({"window_idx": _i, "segment": _nm})
                    continue
                _per_win_all_seg[_i] = _per_win_all_seg.get(_i, 0) + int(_ps)
        _unit_used_raw = (
            (base_view.get("_production_budget_inputs") or {}).get("unit_used") or {})
        for _uid, _v in _unit_used_raw.items():
            if _v is None:
                _unknown_usage.append({"unit_id": _uid})
        _all_seg_used = (
            sum(_per_win_all_seg.values())
            + sum(int(v) for v in _unit_used_raw.values() if v is not None)
        )

        base_view.update({
            "stage_name": getattr(self, "_stage_base", self.stage_name),
            "aggregated_segments": names,
            "min_n_eff_over_g_history": support_history,
            "production_blocks_by_window": blocks_by_window,
            "production_blocks_by_unit": blocks_by_unit,
            "production_blocks_total_by_window": blocks_total_by_window,
            "production_blocks_total_by_unit": blocks_total_by_unit,
            "max_production_blocks_per_window": int(self.max_blocks_per_window),
            "layout_change_invalidated_segments": sorted(_stale_segs),
            "window_provenance": provenance,
            # 布局变更后作废的逐窗证据（λ 对不上当前布局）。
            "stale_layout_evidence": {int(k): v for k, v in stale.items()},
            # 🔑 [2026-09-15 审计①] 有证据、但**无从核对**它描述的是不是当前布局
            # （那一段的 convergence/ibs_state 没有 λ 列表）。与 `stale` 不同：
            # stale 是"核对过、不匹配"，这个是"核不了"。合并照旧（判死会重现
            # win4 占位记录死锁），但必须**说出来**。
            "unverifiable_layout_evidence": {
                int(k): v for k, v in unverifiable.items()},
            # 🔑 [2026-09-15 审计④] 盘上多出来的窗口号：当前布局里**根本没有**
            # 这个下标（旧布局的遗留产物）。`missing_windows` 只算
            # `range(expected)`、从不查多 ⟹ 先前它一路进 merged 且无人报告。
            "out_of_range_windows": sorted(out_of_range),
            "windows": windows,
            "n_windows_found": len(windows),
            "missing_windows": missing,
            "skipped_windows": skipped,
            # [审计 #58] 两个视图必须给出**同一组键**，否则同一个不变量又变成两份
            # 实现（单段视图只传物理窗口、合并视图双传，下游读哪个全看运气）。
            "skipped_sampling_units": skipped_units,
            # 🔑🔑 [BUD-05，2026-09-14] **未知就写 None，不许写 0。**
            # 先前 `int(... or 0)` 把「账本读不到」写成「余量 0」= 耗尽 —— 而同一个
            # 量在 `_epoch_validation_unaffordable` 里是 fail-closed、在 `decide()`
            # 分支 1e 里是 `left is not None and <=0`（未知 = 有钱）。同一个未知，
            # 三套语义。实测 cyclod_ligand1/rep3 win4 的 ledger 确实读不到，
            # 在这里被显示成 0。规矩（TODO §1）：**未知不是零，两个方向都不是。**
            "per_window_budget_remaining": {
                int(w["window_idx"]): (
                    None if w.get("warmup_steps_left") is None
                    else int(w["warmup_steps_left"])
                )
                for w in windows
            },
            # 🔑 [BUD-05] 「全部耗尽」是个**断言**，账不完整就不成立：
            # 只要有一个窗口的余量未知，就不得宣称全局耗尽（否则是拿少算过的账
            # 去论证一个终止性结论）。
            "all_windows_budget_exhausted": bool(
                windows
                and all(w.get("warmup_steps_left") is not None for w in windows)
                and all(int(w["warmup_steps_left"]) <= 0 for w in windows)
            ),
            # 合并视图的生产账：按**合并后**的窗口 + 全部子窗重算。
            # 跨**所有段**的实际生产消耗（含子窗），不是只算胜出段。
            "production_budget": (lambda _inp, _per=_per_win_all_seg,
                                  _tot=_all_seg_used,
                                  _unknown=list(_unknown_usage): {
                "stage_cap_steps": _inp.get("cap"),
                "cap_source": _inp.get("cap_source"),
                "cap_known": _inp.get("cap") is not None,
                "stage_used_steps": _tot,
                # 有未知消耗时**余量不可知** —— 不许拿一个少算过的数当余量去准入。
                "usage_complete": not _unknown,
                "unknown_usage": _unknown,
                "stage_remaining_steps": (
                    None if (_inp.get("cap") is None or _unknown)
                    else int(_inp["cap"]) - _tot
                ),
                # ⚠️ 逐窗按**所有段**累加，不是只取胜出段 —— 旧段的帧是真烧过的
                # GPU，漏掉它账就永远比实际宽。
                "per_window_used_steps": _per,
                "per_sampling_unit_used_steps": dict(_inp.get("unit_used") or {}),
                "new_ensemble_reserve_steps": int(_inp.get("reserve") or 0),
                # 用量不完整时**不得**宣称耗尽（少算过的账不能拿来判终态）。
                "exhausted": bool(
                    _inp.get("cap") is not None and not _unknown
                    and _tot >= int(_inp["cap"])
                ),
                "production_block_steps": _inp.get("block"),
                "cap_scope": "production_steps_only",
                "note": (
                    "合并视图：逐窗生产步数按**所有采样段**累加（不是只取胜出段），"
                    "子窗另计。cap 只管生产步数，预热/验证走另一本账。"
                ),
            })(base_view.get("_production_budget_inputs") or {}),
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
        # [审计 #58] 子窗用 `unit_id` 打印。先前它们的 solver 索引（10000+）混在
        # `skipped_windows` 里被当窗口号打印出来，人读到一个不存在的"窗口 10000"。
        if view.get("skipped_sampling_units"):
            line += f"   ⚠️ 子窗被踢出协方差链 {view['skipped_sampling_units']}"
        out.append(line)
        out.append(f"stage  : analysis={view['stage_analysis_status']}  "
                   f"precision={view['stage_precision_status']}  "
                   f"ΔG={view['stage_total_delta_G']}  "
                   f"(stage 结果{'已' if view['has_stage_result'] else '**未**'}落盘)")
        if view.get("stage_analysis_incomplete_reasons"):
            out.append("         分析未完整：" +
                       "；".join(view["stage_analysis_incomplete_reasons"]))
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
                    # 🔑 [2026-09-14] **列名是 `g`，就必须读 g 那个键。**
                    # `join_lambda_two_sided_support` 现在落 `statistical_inefficiency_g`
                    # = g；`tau_int` 保留为别名但**值已改成真正的 τ_int=(g−1)/2**
                    # ⟹ 照旧读它会静默打印出**一半**的数字，表头却还写着 g。
                    # 旧产物没有新键时才回退到 `tau_int`（那时它还是 g）。
                    f"{f2(u.get('raw_ess'), 9)} "
                    f"{f2(u.get('statistical_inefficiency_g', u.get('tau_int')), 7, 2)} "
                    f"{f2(u.get('top1pct_weight'), 6, 3)}  "
                    f"{('w%d' % j.get('downstream_window', -1)):>5} "
                    f"{f2(d.get('raw_ess'), 9)} "
                    f"{f2(d.get('statistical_inefficiency_g', d.get('tau_int')), 7, 2)} "
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



# =============================================================================
# Stage-2 自治控制器：布局/证据侧的纯函数（S2-D，2026-09-12 从 abfe_pipeline 迁入）
# =============================================================================
# 为什么在这里：这四个是**决策同源**的东西 —— 新 Epoch 预留量、证据来源段解析、
# λ 路径版本读回、段名枚举。判据跟本模块的 `insert_lambda_in_failed_ibs_window` /
# `repartition_tail_from_anchor` / `feasible_repair_actions` 是同一套。跨文件放
# 就是在"同一个不变量的 N 份实现"这条老路上又走一步（设计文档 §6.1）。
#
# ⚠️ **末窗合法化（`_legalize_tail_window`）不在这里**，它留在 `abfe_pipeline`。
# 早先这段注释把它列进来过，是错的：它会 `append_version` 和
# `record_tail_repartition_version` —— **它写盘**。本模块提供的是它用到的那些
# 纯函数（插 λ / 尾段重分 / 版本记录），合法化这个**动作**本身是执行器的事。
#
# ⚠️ **执行器不在这里**：`_run_stage2_autonomous` 主循环留在 `abfe_pipeline`。
# 控制器只读、执行器写盘，这条边界（`test_controller_never_writes_anything`）
# 不因为收拢代码而模糊。`_solve_merged_segments_if_any` 同理留在 pipeline ——
# 它跑的是合并求解，那是求解不是决策。
#
# 这四个都不打日志（迁入前用 `self._log` 的那段是末窗合法化，已留在 pipeline）。

def action_noop_fingerprint(record, path_version) -> Optional[str]:
    """「这个窗口当前的盘面」指纹，用来记住某个动作在此盘面上是 **no-op**。

    返回 ``None`` = **这个盘面没有可比身份**（生产步数读不到）。写侧不得落账、
    读侧不得认作匹配 —— 见下面 `_steps is None` 那段。

    🔑 [2026-09-14 真机] 控制器会反复发一个执行器什么也不做的动作：
    真机两次都是 `RECALIBRATE_FK[0]` 连发 4 次、盘面一字节没变，靠停滞保护
    退出 —— 而退出前**没有**去试那个真正对症的动作。原因是执行器知道"我没做事"
    （日志里明写"重标定未产生新段"），控制器却无从得知。

    指纹只取**加帧/换段就会变**的量：路径版本 + 该窗口的生产步数 + 证据所在段。
    所以补过一块帧之后这条 no-op 记录自动失效、动作重新可选 —— 记的是
    「在这个盘面上没用」，不是「这个动作永远没用」。

    ⚠️ 两侧（执行器记账 / 控制器判读）必须用**同一份**实现，否则又是一个
    「同一个不变量的 N 份实现」。
    """
    w = record or {}
    # 🔑🔑 [审计 #64 后续，2026-09-14] **形参从 `window_record` 改名成 `record`。**
    # 本函数**同时**吃窗口记录和子窗（sampling unit）记录，而这两张表的字段集合
    # 不同（`identity` / `local_index` / `output_dir` 只在子窗上，`segment` /
    # `n_production_segments` 只在窗口记录上）。叫 `window_record` 是在说谎，
    # 而且会让 `tests/test_no_phantom_view_keys.py` 的扫描器（它按形参名认表）
    # 把子窗专属的 `identity` 报成"窗口记录上的幽灵键"—— 它报得对：
    # **按那个名字，那确实是个幽灵键。** 改名让名字与契约一致，扫描器的前提也恢复成立。
    #
    # ⚠️⚠️ **「哪些键在哪张表上」的权威名单在测试里，不在这里，也不许在这里复制一份。**
    #   `tests/test_audit_2026_09_14_controller_budget.py::
    #    test_the_window_table_and_the_unit_table_do_not_share_each_other_s_keys`
    # 它是**从真实 `read()` 打差集**建出来的（单段 + 合并两个构造各跑一遍），
    # 键集一漂就红。在源码里再写一份注释表 = 同一个不变量两份实现，而注释那一份
    # 会先过期 —— 这正是本仓最贵的那类 bug。要改名单，先跑一次 `read()` 打差集。
    #
    # 这一对表的字段差异是 2026-09-14 当天重复出错最多的地方，**四次**：
    #   · `segment` —— 子窗上**从来没有**；窗口记录上也只有**合并视图**才加
    #     （单段视图没有）⟹ 「窗口记录必有 segment」这个直觉是错的；
    #   · `solver_n_decorrelated` —— **两张表上都不存在**，真名
    #     `solver_n_frames_decorrelated`（它只是 history snapshot 的输出键名）；
    #   · `skipped_windows` —— 控制器视图与求解器结果**同名不同源**（元素一个是
    #     int 下标、一个是带 `window_index` 的 dict）；
    #   · `identity` —— 子窗专属，读到窗口记录上就是幽灵键（本函数的形参先前
    #     叫 `window_record`，那个名字本身在说谎，已改名 `record`）。
    # 🔑 [契约 A] **本函数要同时吃窗口记录和子窗（sampling unit）记录。**
    # 三种记录各自的"换段就变"那一维不同名，按可得性依次取：
    #   · 子窗记录 → `identity`（每个子系综自己锁一份 f_k，identity 变就是换了系综）
    #   · 合并视图的窗口记录 → `segment`（段名）
    #   · 单段视图的窗口记录 → `n_production_segments`（段**数**，开了新段就变）
    # 缺这一维的后果是把"换段"这一维悄悄关掉 ⟹ no-op 记录在换段之后仍然有效，
    # 而换段恰恰是"让一个先前没用的动作重新变得有用"的操作。
    seg = w.get("identity")
    if seg is None:
        seg = w.get("segment")
    if seg is None:
        seg = w.get("n_production_segments")
    # ⚠️ **未知步数不等于 0 步。** 先前 `int(... or 0)` 让两者给出同一个指纹 ⟹
    # 一个窗口从"读不到"变成"确实 0 步"（或反过来）指纹不变，no-op 记录不失效。
    #
    # 🔑🔑 [2026-09-15 真机 cyclod_ligand2/rep2] **先前写 `?` 的那版做反了。**
    # 原注释说「未知写成 `?`：它与任何真实步数都不同，记录因此自动失效」——
    # 这是错的：`?` 等于 `?`。**从未生产过**的窗口 `production_steps` 恒为 None
    # ⟹ 指纹恒为 `1|?|vanishing` ⟹ 挂在它上面的 no-op 记录**在结构上永不失效**。
    # 真机后果：win4 的 `RUN_PRODUCTION:4` 被一个 bug 误记成 no-op 之后，
    # 想让它失效必须先产帧，而产帧恰恰被这条记录挡着 —— 死锁，
    # 整条流水线停在 NO_FEASIBLE_ACTION。
    #
    # 现在回到注释本来声明的那个**保守方向**：**未知 ⟹ 没有可比身份 ⟹ 返回
    # None**。两侧都按「None 不匹配任何东西」处理（写侧不记、读侧不认），
    # 宁可多试一次，也绝不粘住一条无法失效的记录。
    _steps = w.get("production_steps")
    if _steps is None:
        return None
    return "|".join(str(x) for x in (
        int(path_version or 0),
        int(_steps),
        str(seg if seg is not None else ""),
    ))


def relearn_epoch_required_steps(window_record):
    """开一个全新 f_k Epoch 至少要预留多少步：LEARN + burn-in + **首档**验证。

    量从这个窗口**自己**上一轮的实际消耗估（自校准），估不到才退保守常量 ——
    跨体系拍一个固定数字是引入未验证阈值。
    """
    led = (window_record or {}).get("warmup_budget_ledger") or {}
    learn = int(led.get("learning_steps") or 0) or 80000
    burn = int(led.get("freeze_burn_in_steps") or 0) or 10000
    # 首档验证预留：不是整份验证预算，只要够走完第一档。
    # ⚠️ 这里原来读 `window_record["validation_attempt_budget_steps"]` —— 窗口记录里
    # **没有这个键**，于是恒取 50000 兜底。首档的权威值就在冻结验证阶梯里，直接用它，
    # 不再要求窗口记录搬运一个它没有的数。
    try:
        import ibs_engine as _ie
        first_rung = int(_ie.FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[0])
    except Exception:  # noqa: BLE001
        first_rung = 50000
    return int(learn + burn + first_rung)


def segment_dirs_for_evidence(segments, stage_dir: str, checkpoint_dir: str):
    """把证据来源的段名翻成 (输出目录, checkpoint 目录)；基准段返回 (None, None)。

    `segments` 是窗口记录里的 `segment` 值（stage 目录名，如 `vanishing` /
    `vanishing_3`）。多个窗口落在不同段时**fail-closed 抛错**，不静默回落到
    基准段 —— 回落正是本来那个 bug。
    """
    names = {str(x) for x in segments if x}
    suffixes = set()
    for nm in names:
        # [审计 #63] 段号解析统一走 `segment_index_of_dir()`（同 `_segment_stage_names`
        # / `existing_segment_names`）：完整后缀纯数字才算段号，rewindow 目录不会被
        # 误判。解析不出来仍然 **fail-closed 抛错**，绝不静默回落到基准段 ——
        # 回落正是本函数原来那个 bug。
        n = segment_index_of_dir(nm, stage_dir)
        if n is None:
            raise ValueError(
                f"无法解析证据来源段名 {nm!r}"
                f"（基准 {os.path.basename(os.path.normpath(stage_dir))!r}）"
            )
        suffixes.add(int(n))
    if len(suffixes) > 1:
        raise ValueError(
            f"补采的窗口跨越多个采样段 {sorted(suffixes)}，无法落在单一段里"
        )
    n = next(iter(suffixes)) if suffixes else 1
    # ⚠️ **基准 stage 目录隐含就是段 1**：新采样段从 **2** 起
    # （`_recalibrate_f_k_and_resample_segment` 默认 `segment_index=2`，
    # 段号分配是 `max([1] + existing) + 1`）。所以 `n <= 1` 覆盖基准段是**正确**的，
    # 不是"`vanishing_1` 被静默重定向"的漏洞 —— `vanishing_1` 这个目录名按约定
    # 根本不会被创建。
    if n <= 1:
        return None, None          # 基准段：走默认目录
    return (
        f"{stage_dir.rstrip(os.sep)}_{n}",
        os.path.join(checkpoint_dir, f"segment_{n}"),
    )



def windows_by_segment(window_indices, window_records, added_steps):
    """把要补采的窗口按**它们证据所在的段**分组：`{段名: {窗口: 新的目标步数}}`。

    目标步数为 ``None`` 表示「这个窗口盘上烧了多少步读不到 ⟹ 不给目标、按
    调用方原目标跑」。**窗口本身一定在分组里** —— 见下面那段长注释。

    多窗补采天然会跨段（真机 win0-3 在 `vanishing_2`、win4 在基准段），而
    `segment_dirs_for_evidence` 对跨段集合是 fail-closed 的 ⟹ 执行器必须
    **一段一次**，不能把整批丢进去。目标步数是"这个窗口已有的 + 一块"，
    所以分组和算步数是同一件事，放在一起才不会漂开。
    """
    by_seg: Dict[str, Dict[int, Optional[int]]] = {}
    for w in window_indices:
        rec = next(
            (x for x in (window_records or []) if int(x["window_idx"]) == int(w)),
            None,
        ) or {}
        # 🔑 [2026-09-14] **这是全仓唯一把「未知步数」写回盘的地方，最该修。**
        # 目标步数会被执行器当成 `_production_step_overrides` 落进 run —— 先前
        # `int(... or 0) + added` 把「读不到这个窗口烧了多少步」固化成
        # 「它烧过 0 步、目标就是一块」，未知从此变成事实，而这个窗口盘上其实
        # 可能已经有几十万步。fail-closed：读不到就**不给目标**，让调用方按原目标
        # 跑（绝不编一个比现有帧还小的目标去覆盖它）。
        #
        # 🔑🔑 [2026-09-15 真机 cyclod_ligand2/rep2] **「不给目标」≠「不给窗口」。**
        # 上面那条 fail-closed 原来写的是 `continue` —— 把窗口整个从分组里**删掉**，
        # 于是执行器的 `for _seg, overrides in ...` 空转，`run_once` **一次都没调**，
        # 1 秒返回、盘面当然逐项未变 ⟹ 通用 no-op 记账把一个**从没被执行过**的
        # 动作记成「执行过且没用」⟹ 下一轮 NO_FEASIBLE_ACTION ⟹
        # `_assert_stage_result_sane` 抛 ANALYSIS_INCOMPLETE 打死整条流水线。
        # 真机盘面：win4 从未生产（只有 warmup_failure.json），`production_steps`
        # 恒 None，no-op 台账里那条 `RUN_PRODUCTION:4` 的指纹 `1|?|vanishing`
        # 里的 `?` 就是它。`CONTINUE_WARMUP`（同一个函数、added_steps=0）被同一行
        # 静默吞掉，日志里「win4 连发 40 次、盘面一动不动」也是这个，不是窗口级
        # resume 缓存门。
        # 正解就是注释本来说的那句：**不给目标、让调用方按原目标跑** ⟹ 窗口留在
        # 分组里、值为 None。引擎侧 `if window_idx in production_step_overrides`
        # 本来就按「不在表里 = 用默认 n_steps_per_window」处理，调用方把 None
        # 过滤掉即可（见 `_run_stage2_autonomous` 的 RUN_PRODUCTION 分支）。
        _have = rec.get("production_steps")
        by_seg.setdefault(str(rec.get("segment") or ""), {})[int(w)] = (
            None if _have is None else int(_have) + int(added_steps)
        )
    return by_seg


def lambdas_from_version_record(record, fallback):
    """🔑 **落盘的路径版本是 λ 的唯一权威。**

    `lambda_path_versions._q()` 写盘时把 λ 量化到 `LAMBDA_DECIMALS` 位；
    如果继续拿内存里那份**未量化**的去采样，每插一个 λ 就永久制造一次错位。

    真机（2026-09-12 17:05）实测 win4：
        路径版本 0.31005333   vs   实际采样 0.310053335   差 5e-9
    后果是死锁 —— 采样侧的 resume 门用 `np.allclose(atol=1e-9)`（含默认
    rtol=1e-5 ⟹ 实际容差 ~3e-6）判"λ 匹配、跳过重采"，而分析侧
    `load_ibs_window_outputs_from_dir` 用**精确相等**判"λ 不匹配"直接抛
    ValueError。于是那个窗口**永远采不了也永远读不了**，每次启动必崩。

    🔑🔑 [审计 #4，2026-09-14] **原来的长度兜底是 fail-open，恰好在最要命的
    那一种情形下放行。**

    先前是 `return out if len(out) == len(fallback) else [float(x) for x in fallback]`
    —— 长度对不上就**静默退回内存里那份未量化的 λ**，也就是这个函数存在的全部
    理由所要防的那个东西。而长度对不上正是**唯一危险**的情形：
    `append_version` 是幂等的，事件 id 已在祖先链里时它**返回现有的当前版本**
    （见 `lambda_path_versions.append_version`）。所以「刚算出的 `new_l` 与返回
    记录的态数不同」= 这次插点其实没被应用、盘上的权威布局是另一套。此时拿
    `new_l` 去采样，采出来的帧与已发布的路径版本**描述的不是同一条布局**，正是
    上面那段死锁的成因。

    现在：长度不符 ⟹ **fail-closed 抛错**，把「盘上权威布局 ≠ 调用方以为的布局」
    这件事当场暴露，而不是挑一个看起来能跑的继续跑。
    记录里根本没有 λ（老记录 / 空 states）是**另一回事** —— 那是「无从核对」，
    保持原来的回退行为，不制造新的 fail。
    """
    states = (record or {}).get("states") or []
    out = [
        float(st.get("lambda_vdw")) for st in states
        if st.get("lambda_vdw") is not None
    ]
    if not out:
        # 记录里没有 λ（老格式 / 空 states）⟹ 无从核对，保持既有回退。
        return [float(x) for x in fallback]
    if len(out) != len(fallback):
        raise ValueError(
            f"路径版本记录里的 λ 态数（{len(out)}）与调用方要采样的那份"
            f"（{len(fallback)}）不符。`append_version` 幂等 ⟹ 这通常意味着"
            "**这次布局变更并没有被应用**（同一个 event id 已在祖先链里），"
            "盘上的权威布局是另一套。继续按调用方那份采样会产出一批与已发布路径"
            "版本对不上的帧（分析侧按精确相等判 λ，会直接拒绝，窗口从此既采不了"
            "也读不了）。请先核对 `path_current.json` 与 `path_versions/`，"
            "不要绕过这道检查。"
        )
    return out



def segment_index_of_dir(path: str, stage_dir: str):
    """`<stage_dir>_<N>` → N；基准段 → 1；不是采样段 → None。

    判据是**完整后缀**纯数字，不是 `rsplit("_", 1)[-1]`。immutable rewindow 的
    目录叫 `<stage>_rewindow_<sha256[:12]>`，那 12 位十六进制有 (10/16)**12
    ≈ 0.34% 的概率全是数字 —— 按末段判就会把子系综目录当成采样段合并，
    子窗的局部下标 0/1 被当成物理窗口 0/1，静默产出错误 ΔG。

    ⚠️ **别把那个后缀简化成裸 sha12。** 现在安全是**构造上**安全，不是概率上：
    完整后缀是 `rewindow_<sha12>`，含字面量 `rewindow_` ⟹ 即使 sha12 十二位
    全是数字，`"rewindow_0123456789ab".isdigit()` 仍是 False。去掉那个前缀
    就把构造保证降级成上面那个 0.34%。
    """
    base = os.path.basename(os.path.normpath(stage_dir))
    name = os.path.basename(os.path.normpath(path))
    if name == base:
        return 1
    if not name.startswith(base + "_"):
        return None
    suf = name[len(base) + 1:]
    return int(suf) if suf.isdigit() else None


def existing_segment_names(stage_dir: str):
    """盘上已有的采样段目录名（含基准段），按**段号**排序。

    [审计 #63] 先前按**名字**排序（字符串序 ⟹ `vanishing_10` 排在 `vanishing_2`
    之前），且用 `rsplit("_", 1)[-1]` 判段号 ⟹ rewindow 目录有 0.34% 概率被当成
    采样段。两处都改走 `segment_index_of_dir()`，与 `_segment_stage_names` /
    `segment_dirs_for_evidence` 同一份规则。
    """
    base = os.path.basename(os.path.normpath(stage_dir))
    found = []
    for d in glob.glob(stage_dir.rstrip(os.sep) + "_*"):
        n = segment_index_of_dir(d, stage_dir)
        if n is not None and n > 1 and os.path.isdir(d):
            found.append((n, os.path.basename(d)))
    return [base] + [nm for _n, nm in sorted(found)]


def repartition_tail_from_anchor(
    lambdas_var: Sequence[float],
    window_ranges: Sequence[Sequence[int]],
    anchor: float,
    *,
    min_states_per_window: int,
    max_states_per_window: int,
    anchor_atol: float = 1e-9,
    # 🔑🔑 [2026-09-15] 下面三个参数修的是**同一个 bug 的两半**。
    #
    # (1) **方向反了。** 控制器侧的可行性（`feasible_repair_actions` 的
    #     `split_tail_window`）问的是「存不存在 m ≥ 当前尾段窗口数 **+1** 的合法
    #     分窗」——要更**细**。而这里先前调 `vanishing_subdomain_ranges_from_lambdas`
    #     且不传窗口数，那个分窗器内部是 `n_windows = min_windows`，取**最少**窗口数
    #     ⟹ 拆窗动作可能反过来**合并**窗口、把跨度**扩大**。
    #     纸面复现（lo=4, hi=8）：[(0,4),(3,8),(7,12),(11,16)]、anchor=3、尾段 3 个窗
    #     ⟹ 尾段 13 态、总间隔 12、min_windows=ceil(12/7)=2 ⟹ [(3,10),(9,16)]，
    #     三个 5 态窗变成两个 7 态窗。可行性说"可以拆细"，执行器交付"合并了"。
    #     ⟹ `n_windows` 现在是**必须显式给**的目标窗口数（调用方传 m0+1）。
    #
    # (2) **判据被换掉了。** 生产分窗是 `stage2_window_partition=metric_integral`
    #     （按 ∫g dλ 均衡），而尾段重分调的是等状态数贪心、**根本不读 metric_g**。
    #     于是重分之后：冻结前缀按 ∫g 切、新尾段按边数切，同一条路径上两套判据。
    #     ⟹ 给了 `pilot_lambdas`/`metric_g` 就用 `partition_windows_by_metric_integral`，
    #     与生产同一个分窗器；`partition_criterion="metric_integral"` 而数据缺失时
    #     **fail-closed**，不静默退回等弧长（与 `_load_pilot_for_path_evolution`
    #     那几个调用点的口径一致）。
    n_windows: Optional[int] = None,
    pilot_lambdas: Optional[Sequence[float]] = None,
    metric_g: Optional[Sequence[float]] = None,
    partition_criterion: str = "arclength",
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
    _crit = str(partition_criterion).lower()
    _n_tail_before = sum(1 for a, _b in ranges if a >= a_idx)
    _n_target = int(n_windows) if n_windows is not None else None
    if _crit in ("metric_integral", "state_count"):
        if not pilot_lambdas or not metric_g:
            raise RuntimeError(
                f"partition_criterion={_crit} 需要 pilot_lambdas/metric_g，"
                "但调用方没给。拒绝静默退回等状态数贪心 —— 那会让冻结前缀按 ∫g 切、"
                "新尾段按边数切，同一条路径上两套分窗判据。"
            )
        _tail_ranges_local, _mi_diag = partition_windows_by_metric_integral(
            [float(x) for x in tail_lam],
            [float(x) for x in pilot_lambdas],
            [float(x) for x in metric_g],
            min_states_per_window=lo_n,
            max_states_per_window=hi_n,
            n_windows=_n_target,
            objective=_crit,
        )
        tail_ranges_local = [(int(x), int(y)) for x, y in _tail_ranges_local]
    else:
        _mi_diag = None
        if _n_target is not None:
            # 等状态数贪心没有"目标窗口数"这个入口（它内部恒取 min_windows）。
            # 直接用它的构造规则按 `_n_target` 铺：总间隔均分，小窗在前，
            # 与 `vanishing_subdomain_ranges_from_lambdas` 的分配逻辑逐字一致。
            _iv = int(tail_lam.size) - 1
            if not (_n_target * (lo_n - 1) <= _iv <= _n_target * (hi_n - 1)):
                raise RuntimeError(
                    f"尾段 {tail_lam.size} 个态切不出 {_n_target} 个窗口"
                    f"（每窗 ∈[{lo_n},{hi_n}]）"
                )
            _base, _extra = divmod(_iv, _n_target)
            _spans = [_base] * (_n_target - _extra) + [_base + 1] * _extra
            tail_ranges_local, _st = [], 0
            for _sp in _spans:
                tail_ranges_local.append((_st, _st + _sp + 1))
                _st += _sp
        else:
            tail_ranges_local = [
                (int(x), int(y)) for x, y in vanishing_subdomain_ranges_from_lambdas(
                    tail_lam,
                    min_states_per_window=lo_n,
                    max_states_per_window=hi_n,
                    # 🔑🔑 [审计 #50，2026-09-14] **尾段是子路径，不许套全路径的手工固定表。**
                    # `vanishing_subdomain_ranges_from_lambdas` 在 `size == 23` 时会
                    # 短路返回 `VANISHING_FIXED_WINDOW_RANGES` —— 那张表是**全路径**
                    # 手工调出来的，含专为 λ=1 耦合端做的 window-0 收窄，而尾段的起点
                    # 根本不是 λ=1；它还会**完全忽略** min/max_states_per_window，
                    # 而尾段重分恰恰要按调用方给的 lo/hi 切。更糟的是那张表的尺寸是
                    # 5,4,5,5,5,4，后置的 `validate_single_shared_boundary_ranges`
                    # 照样通过 ⟹ 替换发生了却没有任何人报告。
                    allow_fixed_23_state_table=False,
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

    # ---- 🔑🔑 [2026-09-15] 验收 9/10：**动作的方向必须被证明，不能假定** ----
    # 这两条是这次 bug 的直接回归钉子。先前这里的验收全是**结构性**的（覆盖 /
    # 共享边界 / [lo,hi]），而 `[7,7]` 这种「三窗合成两窗」在结构上完全合法 ——
    # 于是一个方向完全反了的结果一路通过、被 `record_tail_repartition_version`
    # 记成一次成功的拆窗。`SPLIT_TAIL_WINDOW` 的语义是**缩跨度**，缩没缩得由
    # 数字说了算。任何一条不满足就 fail-closed，宁可让控制器看见"这个动作做不到"。
    # ⚠️ 这两条**只在调用方显式要求拆细时**生效（`n_windows` 非空）。
    # 另一个调用点是"末窗 K > hi 的布局合法化"，它的目标是**变合法**、不是变细：
    # 例如 lo/hi=4/8、尾段 [9,4] ⟹ [7,6] 就是一次完全正确的合法化，窗口数没变。
    # 把"必须更细"套在它头上会把一个正确结果判成失败。
    _require_finer = _n_target is not None
    _n_tail_after = len(tail_ranges)
    if _require_finer and _n_tail_after <= _n_tail_before:
        raise RuntimeError(
            f"尾段重分没有把窗口切细：anchor 之后原有 {_n_tail_before} 个窗口、"
            f"重分后 {_n_tail_after} 个（sizes={[b - a for a, b in tail_ranges]}）。"
            "`SPLIT_TAIL_WINDOW` 的语义是缩跨度；窗口数没增加说明分窗器取的是"
            "**最少窗口数**（`vanishing_subdomain_ranges_from_lambdas` 内部的 "
            "`n_windows = min_windows`），方向与可行性判据相反。"
            "调用方必须显式给 `n_windows`（= 原尾段窗口数 + 1）。"
        )
    # 跨度的口径**跟分窗器的目标函数同一个**：metric_integral 时比 ∫g，
    # 否则比态数。比"失败窗口重分后的 span"是不成立的 —— 边界整体移动之后，
    # 原来那个窗口的下标已经不指向同一段 λ 了。
    _tail_old = [(a, b) for a, b in ranges if a >= a_idx]
    if _mi_diag is not None:
        _gc = metric_integral_cumulative(
            lam, [float(x) for x in pilot_lambdas], [float(x) for x in metric_g]
        )
        _span = lambda rs: max(abs(float(_gc[b - 1] - _gc[a])) for a, b in rs)
        _before, _after, _unit = _span(_tail_old), _span(tail_ranges), "∫g dλ"
    else:
        _span = lambda rs: max(b - a for a, b in rs)
        _before, _after, _unit = _span(_tail_old), _span(tail_ranges), "态数"
    if _require_finer and not (_after < _before):
        raise RuntimeError(
            f"尾段重分没有缩小跨度：最大窗口 {_unit} {_before:.4g} → {_after:.4g}"
            f"（尾段 sizes {[b - a for a, b in _tail_old]} → "
            f"{[b - a for a, b in tail_ranges]}）。拒绝把一次扩大跨度的重分"
            "记成成功的 `SPLIT_TAIL_WINDOW`。"
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
        "tail_windows_before": int(_n_tail_before),
        "tail_windows_after": int(_n_tail_after),
        "partition_criterion": _crit,
        "tail_max_window_span_before": float(_before),
        "tail_max_window_span_after": float(_after),
        "tail_span_unit": _unit,
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


def tail_overflow_exemption_active(path_events) -> bool:
    """末窗的「溢出槽豁免」还在不在。

    🔑🔑 [审计 #29，2026-09-14] **`feasible_repair_actions(tail_exempt_from_max=…)`
    先前是个死参数** —— 它的 docstring 明写「调用方应从路径版本链里数
    ``split_tail_window`` 事件来决定这个标志」，而**全仓没有任何调用方传过它**
    ⟹ 恒为 `True` ⟹「末窗被拆过之后豁免作废、此后插点无处可去 ⟹
    `HALT_LAMBDA_BUDGET_INSUFFICIENT`」那条分支**结构上不可达**。
    后果是插 λ 在溢出槽已经用尽之后仍被判成可行。

    规则（就是那句 docstring 说的）：末窗是 model B 的溢出槽，**在被拆过之前**
    豁免 ``max_states_per_window``；拆过之后两个孩子都受约束、溢出槽消失。

    ⚠️ 两种拆窗事件**都**算数：路径演化分支写 ``split_tail_window``，
    自治分支写 ``tail_repartition``（`record_tail_repartition_version`）。
    先前那句 docstring 只提了前者 —— 只数一种等于漏掉自治循环拆的每一次。

    ``path_events``：``{事件名: 次数}``（`Stage2RepairController._read_path()`
    的 `events`，或 `lambda_path_versions` 数出来的同形 dict）。
    """
    ev = dict(path_events or {})
    return not (int(ev.get("split_tail_window", 0) or 0)
                + int(ev.get("tail_repartition", 0) or 0))


def vanishing_rescue_ranges(
    failing_window_indices: Sequence[int],
    base_ranges: Sequence[Sequence[int]],
) -> List[Tuple[int, int]]:
    """把一个失败的系综换成若干**更小的重叠子系综**（λ 表一个态都不动）。

    🔑 [REWIND-01，2026-09-17] **这是有界重窗子系综划分的唯一权威。**
    先前它只活在 `abfe_pipeline.ABFEPipeline._build_vanishing_rescue_ranges`
    （静态、纯函数），而控制器那侧要判可行性 / 算成本却 import 不动 abfe_pipeline
    （循环依赖 + 它 import openmm）⟹ 成本表只好自己写一份
    `2 if (b-a) > 2 else 1`。同一个不变量两份实现，正是本仓最贵的那个形状。
    现在搬到这里，`abfe_pipeline` 那个静态方法转成一行转发。

    切法：按中点切左右两个**共享中点**的子窗（覆盖完整、接缝有共享节点，
    两条都是执行器的 fail-closed 断言）。K ≤ 2 切不出两个各 ≥2 态的子窗 ⟹
    原样返回，由执行器的 `len(children) < 2` 拒掉。
    """
    rescue_ranges: List[Tuple[int, int]] = []
    for window_idx in sorted(set(int(x) for x in failing_window_indices)):
        start, end = (int(x) for x in base_ranges[window_idx])
        n_states = end - start
        if n_states <= 2:
            rescue_ranges.append((start, end))
            continue
        midpoint = start + (n_states - 1) // 2
        left = (start, midpoint + 1)
        right = (midpoint, end)
        for child in (left, right):
            if child[1] - child[0] >= 2 and child not in rescue_ranges:
                rescue_ranges.append(child)
    return rescue_ranges


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
    # 🔑🔑 [审计 #24，2026-09-14] `SPLIT_TAIL_WINDOW` 这个**动作**实际做的是
    # 「从 anchor 起重分整个尾段」（`repartition_tail_from_anchor`），不是
    # 「把末窗一分为二」。判它需要重分起点 = anchor 所在窗口的**首态全局下标**。
    # 给不出就如实返回「判不了」，**不拿末窗那条冒充**（见函数体里的长注释）。
    tail_repartition_start_state: Optional[int] = None,
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

    # ---- 拆**末窗本身**：两子窗共享一个边界态 ⟹ p + q − 1 = K，两侧都落在 [lo, hi] ----
    # 这一条问的是「溢出槽长到能一分为二了没有」，是 model B 插点的收尾条件。
    split_ok = any(
        p + q - 1 == k_tail and lo_n <= p <= hi_n and lo_n <= q <= hi_n
        for p in range(lo_n, hi_n + 1)
        for q in range(lo_n, hi_n + 1)
    )
    out["split_last_window_in_two"] = None if split_ok else (
        f"末窗 K_tail={k_tail} 在 [{lo_n},{hi_n}] 下切不出两个合法子窗"
        f"（需 p+q−1={k_tail} 且两侧都在区间内 ⟹ 最小可拆 K = {2 * lo_n - 1}）。"
        "继续插 λ 让末窗长大即可到达；拆窗只是末窗溢出压不住时的收尾动作。"
    )

    # ---- `SPLIT_TAIL_WINDOW` 这个**动作**的可行性 ----
    # 🔑🔑 [审计 #24，2026-09-14] **判据问的问题和动作做的事不是一回事。**
    #
    # 控制器的 `SPLIT_TAIL_WINDOW` 落到执行器是 `repartition_tail_from_anchor`：
    # 从 `first_untrusted_window` 的**首态**起，把**整个尾段**重新分窗
    # （anchor 之前逐字冻结）。而先前这里给的可行性是「**末窗**能不能一分为二」
    # （K_tail ∈ [2lo−1, 2hi−1]）—— 两个完全不同的问题。
    #
    # 后果是双向的：
    #   · 标准 23 态布局末窗 K=4 < 7 ⟹ 这个动作**永久不可行**，
    #     而 `repartition_tail_from_anchor` 明明能处理中段重分 ⟹ 设计里那条
    #     「从第一个不可信窗口起重分」的主力修复路径**一次都没被选中过**；
    #   · 反过来 K_tail ∈ [7,9] 被放行时，实际改动范围远大于「拆末窗」——
    #     anchor 之后所有窗口都被重写、下游证据全部作废，而可行性检查对此一无所知。
    #
    # 现在：调用方给出重分起点（`tail_repartition_start_state`，即 anchor 所在窗口
    # 的首态全局下标）时，按**真正要做的那件事**判；给不出就如实说「判不了」，
    # **不再拿末窗那条冒充**。
    # 数学：相邻窗口共享一个边界态 ⟹ m 个窗口覆盖 T 个态需 Σk_i −(m−1) = T，
    # 每个 k_i ∈ [lo,hi] ⟹ m(lo−1)+1 ≤ T ≤ m(hi−1)+1。要「缩跨度」还得比现有
    # 窗口数**更细**，所以要求存在 m ≥ m0+1 满足它（m0 = 当前尾段窗口数）。
    _t_start = tail_repartition_start_state
    if _t_start is None:
        out["split_tail_window"] = (
            "判不了：`SPLIT_TAIL_WINDOW` 实际做的是**从 anchor 起重分整个尾段**"
            "（`repartition_tail_from_anchor`），而调用方没给重分起点 "
            "`tail_repartition_start_state`。**不拿「末窗能否一分为二」冒充** —— "
            f"那是另一个问题（它的答案在 `split_last_window_in_two`："
            f"{'可行' if split_ok else '不可行'}）。"
        )
    else:
        _t_start = int(_t_start)
        _starts = [a for a, _b in ranges]
        if _t_start not in _starts:
            out["split_tail_window"] = (
                f"重分起点 {_t_start} 不是任何现有窗口的首态（窗口起点 {_starts}）⟹ "
                "它不是一个共享边界态，`repartition_tail_from_anchor` 会 fail-closed 抛错。"
            )
        else:
            _t_states = int(n_states) - _t_start
            _m0 = sum(1 for a in _starts if a >= _t_start)
            _ok_ms = [m for m in range(_m0 + 1, _t_states + 2)
                      if m * (lo_n - 1) + 1 <= _t_states <= m * (hi_n - 1) + 1]
            out["split_tail_window"] = None if _ok_ms else (
                f"从态 {_t_start} 起的尾段有 {_t_states} 个态、现为 {_m0} 个窗口，"
                f"在 [{lo_n},{hi_n}] 下切不出**更细**的合法分窗"
                f"（需存在 m ≥ {_m0 + 1} 使 m·{lo_n - 1}+1 ≤ {_t_states} ≤ m·{hi_n - 1}+1）。"
                "缩跨度得另找动作（插 λ / 固定 λ 表上的有界重窗）。"
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
    if criterion not in ("arclength", "metric_integral", "state_count"):
        raise ValueError(f"未知分窗判据 {partition_criterion!r}")
    if criterion in ("metric_integral", "state_count") and pilot_metric_g is None:
        # 初始布局按 ∫g 定的度量，补救却按 ∫√g 定 n，等于第一次自动补救就换了判据。
        # 拿不到 metric_g 就明着拒绝，不静默退回等弧长。
        raise ValueError(
            f"partition_criterion={criterion!r} 必须同时提供 pilot_metric_g；"
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

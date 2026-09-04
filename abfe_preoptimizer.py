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
from typing import Any, Dict, List, Tuple, Optional
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

    with open(preopt_path, "r") as f:
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
    with open(preopt_path, "w") as f:
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
        integrator.step(sample_interval)
        state = context.getState(getEnergy=True, groups={1})
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    if remainder:
        integrator.step(remainder)
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
            self.context.getIntegrator().step(25000)

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
            self.context.getIntegrator().step(500)

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

    def _group1_energy_at(self, parameter_name: str, lam: float) -> float:
        self.context.setParameter(parameter_name, float(lam))
        state = self.context.getState(getEnergy=True, groups={1})
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
                e_hi = self._group1_energy_at(parameter_name, hi)
                e_lo = self._group1_energy_at(parameter_name, lo)
                return (e_hi - e_lo) / (hi - lo)
            e_0 = self._group1_energy_at(parameter_name, lam)
            if hi > lam:
                return (self._group1_energy_at(parameter_name, hi) - e_0) / (hi - lam)
            if lo < lam:
                return (e_0 - self._group1_energy_at(parameter_name, lo)) / (lam - lo)
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
                self.context.getIntegrator().step(batch_steps)
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
        self.context.getIntegrator().step(5000)

        variance_data = []
        lambdas = np.linspace(1.0, 0.0, n_states)
        print(f"[STAGE1] 线性采样点: {lambdas}")

        for i, lam in enumerate(lambdas):
            self.context.setParameter(self.param_coul, float(lam))
            if self.param_vdw is not None:  # ✅ 修复：增加 None 守卫，确保 Stage1 安全
                self.context.setParameter(self.param_vdw, 1.0)
            self.context.getIntegrator().step(500)
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
            for lam in new_lams:
                self.context.setParameter(self.param_vdw, float(lam))
                if self.param_coul is not None and self.param_coul in current_params:
                    self.context.setParameter(self.param_coul, 0.0)
                self.context.getIntegrator().step(500)
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
        self.context.getIntegrator().step(5000)

        # Probe a conventional grid for diagnostics.  Production lambda
        # placement keeps the v19 quadratic base so the lambda~0 tail cannot
        # collapse again; v20 additionally lets this metric insert two bridge
        # states into the longest remaining production edges.
        pilot_lambdas = human_vanishing_initial_lambdas(int(n_states))
        metric_g = []
        pilot_points = []
        for lam in pilot_lambdas:
            self.context.setParameter(self.param_vdw, float(lam))
            if self.param_coul is not None and self.param_coul in current_params:
                self.context.setParameter(self.param_coul, 0.0)
            self.context.getIntegrator().step(500)
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
            _timing = point_diag.get("timing_s", {})
            print(
                f"    [preopt λ={float(lam):.4f}] "
                + ", ".join(f"{k}={v:.1f}s" for k, v in _timing.items())
            )

        pilot_lambdas, metric_g, pilot_points = self._refine_pilot_grid_in_steep_segments(
            pilot_lambdas,
            metric_g,
            pilot_points,
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
from openmm import XmlSerializer


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
    new_sys = ensure_owned_system(XmlSerializer.deserialize(XmlSerializer.serialize(system)))
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
        # Dual-lambda pilot diagnostics sample force group 1.  Include the
        # native PME/NonbondedForce there so finite differences see the same
        # λ-dependent Coulomb energy that the offsets just installed control.
        nb.setForceGroup(1)

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


def _sample_group1_energy(context, lam_coul: float, lam_vdw: float) -> float:
    context.setParameter("lam_coul", float(lam_coul))
    context.setParameter("lam_vdw", float(lam_vdw))
    return context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


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
        e_plus = _sample_group1_energy(context, *plus)
        e_minus = _sample_group1_energy(context, *minus)
        return (e_plus - e_minus) / (2.0 * delta)
    if plus_ok:
        e_plus = _sample_group1_energy(context, *plus)
        e0 = _sample_group1_energy(context, lc, lv)
        step = _safe_lambda_delta(lc if axis == "coul" else lv, plus[0] if axis == "coul" else plus[1])
        return (e_plus - e0) / max(step, 1e-8)
    if minus_ok:
        e_minus = _sample_group1_energy(context, *minus)
        e0 = _sample_group1_energy(context, lc, lv)
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

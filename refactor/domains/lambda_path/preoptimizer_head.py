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



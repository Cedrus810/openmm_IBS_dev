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
    last_window_max_states: Optional[int] = None,
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
    # 末窗上界：不给就等于 `hi`（逐位保持旧行为）。给了就必须能被拆回 [lo, hi]
    # —— 见 `_cap` 上面那段推导：`last_hi ≤ 2*hi − 2`。超了 fail-closed，
    # 因为那种布局一旦落盘就**永远合法化不了**（拆出来必有子窗 > hi）。
    last_hi = hi if last_window_max_states is None else int(last_window_max_states)
    if last_hi < lo:
        raise ValueError(
            f"last_window_max_states={last_hi} 小于 min_states_per_window={lo}")
    if last_hi > 2 * hi - 2:
        raise ValueError(
            f"last_window_max_states={last_hi} 超过 2*max_states_per_window−2="
            f"{2 * hi - 2}：末窗长到 {last_hi + 1} 时要拆成两个共享一个边界态的子窗"
            f"（p+q−1={last_hi + 1}），而那样至少有一个子窗 > {hi} ⟹ 这个布局"
            "在构造上合法化不了。")
    # 约束**进 DP 的候选枚举**，不是解完再过滤。两者不等价：先解再滤只会把
    # 整条布局丢掉（然后回退到窗口数更多的那条），拿不到"在该约束下的最优"。
    # 🔑🔑🔑 [2026-09-18] **三个上界，不是一个。**
    #
    #   · 首窗   `first_hi`   —— 解耦端点，本方法的必然坏窗口，单独收紧；
    #   · 非末窗 `hi`         —— 真正要跑的窗口的上界；
    #   · 末窗   `last_hi`    —— **溢出槽**。model B 插 λ 时非末窗 ranges 逐字冻结、
    #     多出来的态一律落到末窗（`insert_lambda_in_failed_ibs_window` 的后置断言），
    #     所以末窗天然会比别的窗口大。
    #
    # 先前这三件事**共用一个 `hi`**，后果是要给末窗留溢出空间就必须把 `hi` 抬高，
    # 而那会把**每一个非末窗**一起抬上去。真机实测（45 个 run 的全部布局）里
    # `[4,6,5,4,6]`、`[4,7,5,4,5]`、`[4,8,4,4,5]`、`[4,6,6,4,5]` 的第二/三窗
    # 就是这么被抬到 6/7/8 的 —— 那些窗口并不需要溢出空间。
    #
    # `last_hi` 的上界是**算出来的，不是拍的**：末窗要拆成两个共享一个边界态的
    # 子窗时 `p + q − 1 = K`，两个都 ≤ `hi` ⟹ 可拆的最大 K 是 `2*hi − 1`。
    # 所以「长到 `last_hi`、超了就拆」要求 `last_hi + 1 ≤ 2*hi − 1`，即
    # **`last_hi ≤ 2*hi − 2`**。lo=4/hi=5 ⟹ `last_hi ≤ 8`，而 9 正好拆成 5+5。
    # 顺带消掉死区：死区是 `(hi, 2lo−1)`，而拆窗只在 K = last_hi+1 ≥ 2lo−1 时发生
    # ⟹ 永远落在可拆区间里，不再需要"先补一个 λ 把末窗喂到可拆"。
    # ⚠️ 末窗同时是首窗（单窗布局）时按**首窗**算：那是更紧的那个，保守优先。
    # ⚠️ `w_target` 是 `_solve()` 的形参（每个候选窗口数各解一次），所以这里
    # 必须把它当**入参**收进来，不能靠闭包 —— 闭包在定义点还没绑定，延迟求值
    # 会在第一次调用时 NameError。
    _cap = lambda a, w, w_target: (
        first_hi if a == 0 else (last_hi if w == w_target - 1 else hi))
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
                for size in range(lo, _cap(i, w, w_target) + 1):
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
                for size in range(lo, _cap(i, w, w_target) + 1):
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
                for size in range(lo, _cap(i, w, w_target) + 1):
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



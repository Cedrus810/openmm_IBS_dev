class IBSWindowManagerShadowCoul(IBSWindowManagerDualLambda):
    """Shadow-Coulomb IBS 去电荷窗口管理器 (实验性，尚未经物理验证)。

    复用 IBSWindowManagerDualLambda 的窗口调度/最小化/Boresch 爬坡/渐进预热/
    生产采样/断点续传/能量与 f_k 落盘逻辑，只把系统构造换成
    build_shadow_coul_ibs_system：CV 核函数是 erfc(alpha*r)/r（Ewald 实空间
    形式），物理上兼容 CustomNonbondedForce/CustomCVForce，因此可以合法放进
    IBS 偏置，不会像真实 PME 库仑那样被截断成 cutoff 相互作用。

    约束与前提（均由 build_shadow_coul_ibs_system 内部显式检查，不满足时会
    直接 raise 而不是静默算错）：
      - 配体必须电中性；带净电配体的 Shadow-Coulomb 去电荷路径尚不支持。
      - VdW 全程满强度、不参与炼金（属于 U_common），因此 self.lambdas_vdw
        恒为 1.0，仅用于日志；构建出的系统不含 lambda_shield 全局参数，
        基类里对它的 setParameter 调用会被存在性检查自动跳过。
      - 这条去电荷路径只覆盖"Shadow 满电荷 -> Shadow 去电荷"这一段，前面还
        需要一段独立的 Bridge 腿（ibs_engine.run_shadow_bridge_leg）把体系从
        "真实 PME 满电荷"搭桥到"Shadow 满电荷"端点，两段 ΔG 相加才是完整的
        去电荷自由能。

    这个子类的 `super().__init__(...)` 没有传 `pilot_lambdas`/
    `pilot_mean_dU_dlambda`（基类默认 `None`），对 Shadow-Coulomb 无意义——
    这两个字段只用于 `run_all_windows` 里给 vanishing 窗口的 f_k 热启动，
    不需要为这个子类专门补传。
    """

    def __init__(
        self,
        system_template,
        topology,
        perturbed_atom_indices,
        lambdas_shadow_coul: List[float],
        temperature,
        window_ranges: List[Tuple[int, int]],
        restraint_params: Optional[Dict] = None,
        prefix: str = "abfe_shadow",
        platform_name: str = "CUDA",
        output_dir: str = "./output",
        checkpoint_dir: str = "./checkpoints",
        coion_identity: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            system_template=system_template,
            topology=topology,
            perturbed_atom_indices=perturbed_atom_indices,
            lambdas_coul=list(lambdas_shadow_coul),
            lambdas_vdw=[1.0] * len(lambdas_shadow_coul),
            temperature=temperature,
            window_ranges=window_ranges,
            alchemical_params=None,
            potential_type="softcore",
            restraint_params=restraint_params,
            prefix=prefix,
            platform_name=platform_name,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            coion_identity=coion_identity,
        )
        self.lambdas_shadow_coul = self.lambdas_coul

    def _build_window_system(self, lc_win, lv_win, resolved_box, positions):
        # 🔑 [2026-09-10] serialize→deserialize 往返 → clone()。
        # 这是 2026-09-09 那批迁移漏掉的一半。clone() 走 SerializationNode，
        # 不生成中间字符串；原写法每个窗口都要在堆上开一份多兆字节的 str
        # （大体系上是几十 MB），紧接着就是 build_ibs_dual_system 里
        # 30k 原子 × getParticleParameters 的分配风暴。
        return build_shadow_coul_ibs_system(
            ensure_owned_system(XmlSerializer.clone(self.system_template)),
            self.topology,
            self.ligand_indices,
            lc_win,
            restraint_params=self.boresch,
            prefix=self.prefix,
            box_vectors=resolved_box,
            temperature=self.temperature,
        )


def _softmax_occupancy_per_state(
    u_kn_softcore: np.ndarray, f_frozen: np.ndarray, kt: float
) -> np.ndarray:
    """Per-state mean occupancy under a FIXED f_k, from raw softcore
    energies: p_k ∝ exp(-beta*(u_k - f_k)), softmax per frame then averaged
    over frames.

    [Candidate-first, Validate-or-Learn v1] Extracted so both
    ``_diagnose_local_mbar_situation`` and the VALIDATE-phase cheap
    early-exit occupancy probe (``run_all_windows``, before the full
    IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES-batch local-MBAR gate is run) read
    the exact same occupancy definition instead of two copies that could
    silently drift apart.
    """
    u = np.asarray(u_kn_softcore, dtype=np.float64)
    f = np.asarray(f_frozen, dtype=np.float64).ravel()
    beta = 1.0 / float(kt)
    logits = beta * (f[:, None] - u)
    logits -= np.max(logits, axis=0, keepdims=True)
    w = np.exp(logits)
    p = w / np.sum(w, axis=0, keepdims=True)
    return np.mean(p, axis=1)


def _diagnose_local_mbar_situation(
    u_kn_softcore: np.ndarray,
    f_frozen: np.ndarray,
    kt: float,
    gate_mbar: Dict[str, Any],
    global_start: int,
) -> Dict[str, Any]:
    """[IBS_BIAS_PROTOCOL_VERSION=29] 把一次冻结 local-MBAR loose gate 的"现场"
    量化成可读诊断，回答"哪个态/哪条边饿死、物理缺口多大、该在哪里插 λ"。

    - occupancy_per_state：当前冻结 f_k 下每个局部态的平均占据（softmax 与
      update_weights/gate 同式：p_k ∝ exp(-beta*(u_k - f_k))）。一个态≈1、其余
      ≈0 就是塌缩。
    - ess_ratio_per_state：local MBAR 对每个物理态的重加权有效样本比例（来自
      gate_mbar）。~0 = 该态与采样分布零重叠，ΔF 不可信。
    - adjacent_mean_delta_u_kJ_mol：相邻态原始 softcore 平均能量差（物理缺口，
      与 f_k 无关）。某条边极大 = 该边本身没有构象重叠，加权改不动。
    - starved_state / starved_edge：占据最低（或 ESS 最低）的态及其两侧边，给出
      global 态索引，方便直接决定在哪对 global 态之间插 λ 或拆窗。
    纯诊断，不参与任何控制流。
    """
    u = np.asarray(u_kn_softcore, dtype=np.float64)
    f = np.asarray(f_frozen, dtype=np.float64).ravel()
    K = u.shape[0]
    # 占据：p_k ∝ exp(-beta*(u_k - f_k))，逐帧 softmax 再对帧取平均。
    occ = _softmax_occupancy_per_state(u, f, kt)
    mean_u = np.mean(u, axis=1)
    adjacent_delta_u = np.diff(mean_u) if K > 1 else np.array([])
    ess_map = gate_mbar.get("ess_ratio_per_lambda") or {}
    ess_per_state = [float(ess_map.get(int(k), float("nan"))) for k in range(K)]
    # 最低占据态：occ 最小者（占据是最直接、总有定义的信号）。注意"最低占据"≠
    # "饿死"——只有当它真的低于健康下限 0.5/K 时才算塌陷/饿死。
    starved_k = int(np.argmin(occ)) if K > 0 else 0
    starved_edges = []
    if starved_k - 1 >= 0:
        starved_edges.append((starved_k - 1, starved_k))
    if starved_k + 1 <= K - 1:
        starved_edges.append((starved_k, starved_k + 1))
    is_endpoint = bool(starved_k == 0 or starved_k == K - 1)
    # 🔑 占据平坦度：区分"真塌陷（需插 λ/拆窗）"与"只是短 warmup 统计薄、
    # 占据其实平坦（交生产后 MBAR）"。用与生产入口占据兜底门相同的判据。
    target_p = 1.0 / float(K) if K > 0 else 0.0
    sum_sq_occ = float(np.sum(np.square(occ))) if K > 0 else 0.0
    coverage_ess = float(1.0 / sum_sq_occ) if sum_sq_occ > 0.0 else 0.0
    min_occ = float(np.min(occ)) if K > 0 else 0.0
    max_occ = float(np.max(occ)) if K > 0 else 0.0
    least_occupied_below_floor = bool(
        K > 0 and min_occ < IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION * target_p
    )
    occupancy_is_flat = bool(
        K > 0
        and max_occ <= IBS_LOCAL_MBAR_GATE_OCC_MAX_FRACTION * target_p
        and min_occ >= IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION * target_p
        and coverage_ess
        >= IBS_LOCAL_MBAR_GATE_OCC_MIN_COVERAGE_ESS_FRACTION * float(K)
    )
    # 塌陷（硬瓶颈）判据比"未平坦"严得多：coverage_ESS<0.5K 或某态<0.25/K。
    occupancy_collapsed = bool(
        K > 0
        and (
            coverage_ess
            < IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_COVERAGE_ESS_FRACTION * float(K)
            or min_occ < IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_MIN_FRACTION * target_p
        )
    )
    return {
        "n_states": int(K),
        "global_state_start": int(global_start),
        "occupancy_per_state": [float(x) for x in occ],
        "ess_ratio_per_state": ess_per_state,
        "mean_softcore_u_kJ_mol_per_state": [float(x) for x in mean_u],
        "adjacent_mean_delta_u_kJ_mol": [float(x) for x in adjacent_delta_u],
        "coverage_ess": coverage_ess,
        "occupancy_is_flat": occupancy_is_flat,
        "occupancy_collapsed": occupancy_collapsed,
        "least_occupied_below_floor": least_occupied_below_floor,
        "occupancy_floor": float(IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION * target_p),
        # 名字保留 starved_* 以兼容既有 gate_diag 消费方；语义是"最低占据态"。
        "starved_local_state": int(starved_k),
        "starved_global_state": int(global_start + starved_k),
        "starved_occupancy": float(occ[starved_k]) if K > 0 else float("nan"),
        "starved_ess_ratio": (
            float(ess_per_state[starved_k]) if K > 0 else float("nan")
        ),
        "starved_edges_global": [
            [int(global_start + a), int(global_start + b)] for a, b in starved_edges
        ],
        "starved_edge_mean_delta_u_kJ_mol": [
            float(adjacent_delta_u[min(a, b)]) for a, b in starved_edges
        ],
        "is_endpoint": is_endpoint,
    }


def _format_local_mbar_situation(situation: Dict[str, Any]) -> str:
    """把 _diagnose_local_mbar_situation 的 dict 压成一行可读中文摘要。"""
    if not situation:
        return "无可用现场诊断"
    if situation.get("error") or situation.get("starved_global_state") is None:
        return f"现场诊断不可用（{situation.get('error', 'incomplete')}）"
    occ = situation.get("occupancy_per_state") or []
    gstart = int(situation.get("global_state_start", 0))
    occ_str = ", ".join(
        f"s{gstart + k}={o:.2e}" for k, o in enumerate(occ)
    )
    edges = situation.get("starved_edges_global") or []
    edge_du = situation.get("starved_edge_mean_delta_u_kJ_mol") or []
    edge_str = "; ".join(
        f"边{e}原始Δu≈{du:+.1f} kJ/mol"
        for e, du in zip(edges, edge_du)
    ) or "n/a"
    # "最低占据态"用中性词；只有真的塌陷（occupancy_collapsed）才标"饿死"，避免把
    # 0.125=0.5/K 这种健康端点误报成塌陷。
    _collapsed = situation.get("occupancy_collapsed")
    _flat = situation.get("occupancy_is_flat")
    _label = "饿死态" if _collapsed else "最低占据态"
    _flat_tag = "，占据平坦" if _flat else ("，占据塌陷" if _collapsed else "，占据未平坦")
    return (
        f"占据[{occ_str}]（coverage_ESS={situation.get('coverage_ess', float('nan')):.2f}"
        f"{_flat_tag}）；{_label}=global s{situation.get('starved_global_state')}"
        f"（占据={situation.get('starved_occupancy'):.2e}, "
        f"ESS_ratio={situation.get('starved_ess_ratio'):.3f}"
        f"{'，端点' if situation.get('is_endpoint') else ''}）；{edge_str}"
    )


def _logsumexp_rows(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable log-sum-exp along ``axis`` (no scipy.special dep)."""
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    return np.squeeze(a_max, axis=axis) + np.log(
        np.sum(np.exp(a - a_max), axis=axis)
    )


def _ess_from_log_weights(log_w: np.ndarray) -> float:
    """(sum w)^2 / sum w^2 computed in log space; invariant to any constant
    shift of ``log_w`` (so it does not depend on the f_k gauge for a fixed
    state, nor on the arbitrary global energy offset)."""
    log_w = np.asarray(log_w, dtype=np.float64).ravel()
    if log_w.size == 0 or not np.all(np.isfinite(log_w)):
        return float("nan")
    return float(np.exp(2.0 * _logsumexp_rows(log_w) - _logsumexp_rows(2.0 * log_w)))



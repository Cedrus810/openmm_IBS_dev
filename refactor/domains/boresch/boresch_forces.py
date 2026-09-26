



# ============================================================================
# 2. Boresch 限制力 & 解析修正
# ============================================================================
THERMODYNAMIC_CYCLE_DOC = """
Thermodynamic cycle used by this ABFE workflow
=============================================

Complex leg:
  0. [P1-17] The Boresch restraint is switched on in the *fully coupled* complex
     by a sampled alchemical leg over lambda_boresch_scale: 0 -> 1
     (ibs_engine.run_boresch_attachment_leg).  This is the attachment term
     ΔG_attach = ΔG(A' -> A), where A' is the physical bound state (ligand
     coupled, NO restraint) and A is the restrained state the decoupling legs
     actually sample.  Because the Boresch potential is non-negative everywhere,
     ΔG_attach = -kT ln <exp(-beta U_rest)>_{A'} >= 0 is a strict bound; the
     implementation fails closed on a negative value.

     Omitting this term does not merely drop a small constant: it leaves the
     restrained-ensemble charging and vdW values with nowhere to go, which shows
     up as apparent per-term disagreement with reference implementations that do
     include it.  With the term present the cycle closes exactly for ANY
     restraint strength, so the restrained-ensemble charging/vdW values are
     correct as measured and must NOT be separately "corrected" - doing so
     double counts.
  1. A physical Boresch restraint is applied to keep the ligand in the binding
     pose during decoupling.
  2. The alchemical sampler computes the restrained complex-leg decoupling free
     energy, ΔG_decouple,restrained.
  3. The analytical Boresch term returned by calculate_boresch_analytical_correction
     is the standard-state release correction added to that leg:

       ΔG_complex = ΔG_attach + ΔG_decouple,restrained + ΔG_release_to_1M

     Only the total is invariant to the choice of Boresch anchors and force
     constants; the three terms individually are not, so comparing any single
     term against another implementation that used different anchors is
     meaningless.

     with V° = 1.6605 nm^3 and

       ΔG_release_to_1M = -RT ln[
         8π²V° / (r0² sinθA sinθB)
         * sqrt(Kr KθA KθB KφA KφB KφC) / (2πRT)^3
       ].

Solvent leg:
  No Boresch restraint is applied to the ligand in bulk solvent; therefore no
  Boresch analytical release term is added to the solvent leg.

PME/self correction:
  NonbondedForce.addParticleParameterOffset linearly scales ligand charges with
  lambda_coul; every getState(getEnergy=True) call at a given lambda recomputes
  the *complete* PME energy (reciprocal + self + real-space) from the actual
  scaled charges at that state, including the Ewald self-energy term. That
  self-energy term is a real, required part of U_k(x) at that lambda state (it
  exactly cancels the reciprocal-space sum's self-interaction double-count) and
  is therefore already correct in the offline u_kn without any further action.
  An earlier version of this workflow additionally added a manual +C*lambda^2
  correction on top of this, on the (incorrect) assumption that OpenMM's
  reported PME energy was missing this term. That manual correction has been
  revoked: apply_pme_self_correction is now always False in production, and is
  only recorded as an inert diagnostic (charge_square_sum_e2, applied=false) for
  auditing. See PHYSICS_DEFECTS.md for the full history. Charged ligand paths
  still disable ligand-only self correction outright unless a validated
  co-alchemical neutralization cycle is active.

LJ long-range/dispersion correction:
  Custom softcore VDW interaction-group forces do not automatically reproduce
  the original NonbondedForce dispersion correction, and OpenMM's native
  CustomNonbondedForce.setUseLongRangeCorrection cannot simply be enabled on
  these forces: the softcore expression bundles LJ and Coulomb into one
  CustomNonbondedForce, and OpenMM's analytic tail integral diverges (verified
  to crash the CUDA backend outright) once real nonzero charges are present in
  that combined expression. For the default ACE/dual_lambda path
  (_create_softcore_force), this is instead handled with a hand-derived
  analytic mean-field r^-6 dispersion tail correction (uniform-density
  approximation beyond the cutoff, attractive term only), precomputed once per
  window in ibs_engine.py::build_ibs_dual_system and added per-frame inside
  IBSSampler.collect_energies() before MBAR sees the energies -- i.e. it is
  folded into the sampled Hamiltonian itself, not added as a separate additive
  cycle term afterward. The BeutlerSoftcoreBuilder / --decoupling single_lambda
  (REMD) path does not yet have an equivalent correction; results from that
  path should not be treated as including this term. See AUDIT_STATUS.md for
  the full investigation and remaining gaps.

Binding free energy:
  Each leg's decoupling free energy (ΔG_complex, ΔG_solvent) is defined as the
  cost of the lambda:1->0 transformation (coupled -> decoupled), i.e. it is
  positive for a leg with net-favorable interactions. ΔG_complex already
  includes the Boresch restrained-decoupling free energy plus the analytical
  standard-state release term; ΔG_solvent has no restraint term (no restraint
  is applied in bulk solvent). Closing the thermodynamic cycle through the
  common "fully decoupled ligand at 1 M standard state" reference state (which
  is the same whether reached from the complex or the solvent leg) gives:

    ΔG_bind = ΔG_solvent - ΔG_complex + ΔG_APBS

  (ΔG_APBS defaults to 0 and is only added when supplied as an explicit
  external term; it does not replace a Lennard-Jones dispersion/tail
  correction). For a genuine binder, ΔG_complex > ΔG_solvent (interactions lost
  on decoupling are stronger in the pocket than in bulk solvent), so ΔG_bind is
  negative, as expected. Terms that are identical in both legs can cancel only
  when the Hamiltonians and correction conventions are documented and matched.
""".strip()


# ============================================================================
# P0-12a：配体构象诊断 + 跨腿构象一致性门（memtodolist §3.0 末条）
#
# §3.0 早就写着：「若配体本身亲脂，溶剂腿（纯水）里可能出现构象塌缩或自聚集；
# 记录溶剂腿配体回转半径与内部氢键随 λ 的变化。」**那条诊断一直没实现**，
# 于是 2026-08-04 那轮膜运行里它真的发生了、跑完了、汇总了，全程没有任何报警：
#
#     配体重原子最大内距（去电荷 12 个 replica、600 帧汇总）
#       膜运行 复合物腿  p5–p95 = 1.34–1.44 nm   （口袋撑着，伸展）
#       膜运行 溶剂腿    p5–p95 = 0.62–0.71 nm   ← 塌缩，且 12 个 replica 零涨落
#       可溶   复合物腿  p5–p95 = 1.34–1.44 nm
#       可溶   溶剂腿    p5–p95 = 0.73–1.39 nm   ← 健康：从塌缩到伸展都采到
#
# 后果不是"某个数偏了"，而是**两条腿在给不同的构象族做热力学循环**：
# 塌缩构象把极性基团聚拢 ⟹ 配体–水静电耦合强 3 倍
# （⟨U_lig-env⟩ = −569 ± 90 vs −190 ± 34 kJ/mol）⟹ 去电荷 191.05 vs 62.80 kJ/mol。
# ΔG_bind = ΔG_solv − ΔG_cplx 在这种情况下没有意义。
#
# 判据为什么取「两条腿的 [p5, p95] 必须重叠」而不是某个绝对阈值：
#   · 它是**物理要求**——循环有意义的前提是两条腿采的是同一个分子的同一个构象族，
#     溶剂腿的分布应当**覆盖**复合物腿采到的那些构象（通常还更宽）；
#   · 它不是为了让哪次运行变绿而调出来的数：实测可溶基线 1.34–1.44 vs 0.73–1.39
#     **有重叠**（1.34–1.39）而膜运行 1.34–1.44 vs 0.62–0.71 **完全不重叠**，
#     两者是被同一条判据分开的，没有留任何可调旋钮。
# ⚠️ 不要为了让某次运行通过而把百分位放宽成 [p0, p100] 或改成"均值差 < 某个 nm"。
#    不重叠的正解是修采样（双起点 / 加构象采样维度），不是放宽门。
# ============================================================================

LIGAND_CONFORMER_DIAGNOSTICS_VERSION = 1
# 判重叠用的百分位区间。取 p5/p95 而不是 min/max：端点单帧极值噪声大，
# 用它判"分布是否重叠"会把一次偶然的伸展当成充分采样。
LIGAND_CONFORMER_OVERLAP_PERCENTILES = (5.0, 95.0)
# 内部极性接触（§3.0 的"内部氢键"代理量）：N/O 重原子对，键路径隔 ≥ 4 键。
# 用重原子距离而不是显式 H 几何，是为了不依赖氢的命名/位置，换体系不会挂。
LIGAND_INTERNAL_POLAR_CONTACT_NM = 0.35
LIGAND_INTERNAL_POLAR_MIN_BOND_SEPARATION = 4


def ligand_conformer_metrics(
    xyz_nm: Any,
    heavy_local_indices: Sequence[int],
    masses_amu: Optional[Sequence[float]] = None,
    polar_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, np.ndarray]:
    """逐帧算配体构象度量。`xyz_nm` 形状 (n_frames, n_ligand_atoms, 3)。

    索引都是**配体局部**索引（0..n_ligand_atoms-1），不是全体系索引 ——
    调用方只需要把配体那几十个原子的坐标切出来，不用搬整条轨迹。

    返回三个逐帧数组：
      * `max_internal_heavy_distance_nm`：重原子间最大距离。**主判据**，
        对"塌缩 vs 伸展"最敏感（实测 0.66 vs 1.28 nm）。
      * `radius_of_gyration_nm`：§3.0 明确要求记录的那个量（质量加权）。
      * `internal_polar_contact_count`：§3.0 的"内部氢键"代理量。
    """
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"xyz_nm 形状非法：{xyz.shape}，期望 (n_frames, n_atoms, 3)")
    heavy = np.asarray(list(heavy_local_indices), dtype=int)
    if heavy.size < 2:
        raise ValueError(
            f"配体重原子少于 2 个（{heavy.size}），无法定义构象度量。"
        )

    h = xyz[:, heavy, :]
    d = np.linalg.norm(h[:, :, None, :] - h[:, None, :, :], axis=-1)
    max_internal = d.max(axis=(1, 2))

    if masses_amu is None:
        w = np.ones(heavy.size, dtype=np.float64)
    else:
        w = np.asarray([float(masses_amu[i]) for i in heavy], dtype=np.float64)
    com = (h * w[None, :, None]).sum(axis=1) / w.sum()
    rg = np.sqrt(
        ((np.linalg.norm(h - com[:, None, :], axis=-1) ** 2) * w[None, :]).sum(axis=1)
        / w.sum()
    )

    if polar_pairs:
        pairs = np.asarray(polar_pairs, dtype=int)
        pd = np.linalg.norm(xyz[:, pairs[:, 0], :] - xyz[:, pairs[:, 1], :], axis=-1)
        contacts = (pd <= LIGAND_INTERNAL_POLAR_CONTACT_NM).sum(axis=1)
    else:
        contacts = np.zeros(xyz.shape[0], dtype=int)

    return {
        "max_internal_heavy_distance_nm": max_internal,
        "radius_of_gyration_nm": rg,
        "internal_polar_contact_count": contacts.astype(float),
    }


def _series_summary(values: Any) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = LIGAND_CONFORMER_OVERLAP_PERCENTILES
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        f"p{lo:g}": float(np.percentile(v, lo)),
        "p50": float(np.percentile(v, 50.0)),
        f"p{hi:g}": float(np.percentile(v, hi)),
        "max": float(np.max(v)),
    }


def ligand_conformer_summary(
    metrics: Dict[str, Any],
    *,
    leg: str,
    source: str = "",
) -> Dict[str, Any]:
    """把逐帧度量汇总成可落盘、可跨腿比较的 summary。"""
    return {
        "protocol_version": LIGAND_CONFORMER_DIAGNOSTICS_VERSION,
        "leg": str(leg),
        "source": str(source),
        "overlap_percentiles": list(LIGAND_CONFORMER_OVERLAP_PERCENTILES),
        "observables": {
            name: _series_summary(values) for name, values in metrics.items()
        },
    }


def ligand_conformer_fingerprint(
    positions_nm: Any,
    ligand_indices: Sequence[int],
    heavy_indices: Optional[Sequence[int]] = None,
    decimals: int = 3,
) -> Dict[str, Any]:
    """[P0-12b] 配体**起始构象**的指纹，进溶剂腿缓存身份。

    用内部距离矩阵（排序后取整）而不是原始坐标：刚体平移/旋转不该让缓存失效，
    **构象**变了才该失效。而构象确实必须进身份 —— 实测同一个分子换一个起始构象，
    溶剂腿去电荷从 62.80 变成 191.05 kJ/mol（P0-12），旧口径里这两次却都判"缓存有效"。
    """
    import hashlib

    pos = np.asarray(positions_nm, dtype=np.float64)
    idx = np.asarray(sorted(int(i) for i in ligand_indices), dtype=int)
    heavy = (
        np.asarray(sorted(int(i) for i in heavy_indices), dtype=int)
        if heavy_indices is not None
        else idx
    )
    h = pos[heavy]
    d = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=-1)
    iu = np.triu_indices(len(heavy), k=1)
    rounded = np.round(np.sort(d[iu]), int(decimals))
    blob = ",".join(f"{v:.{int(decimals)}f}" for v in rounded)
    return {
        "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "n_heavy_atoms": int(len(heavy)),
        "decimals": int(decimals),
        "max_internal_heavy_distance_nm": float(d.max()),
        "radius_of_gyration_nm": float(
            np.sqrt(np.mean(np.linalg.norm(h - h.mean(axis=0), axis=-1) ** 2))
        ),
        "invariant_under": "rigid_translation_and_rotation",
    }


def evaluate_cross_leg_conformer_consistency(
    complex_summary: Optional[Dict[str, Any]],
    solvent_summary: Optional[Dict[str, Any]],
    observable: str = "max_internal_heavy_distance_nm",
) -> Dict[str, Any]:
    """[P0-12a] 两条腿的配体构象分布是否重叠；不重叠即不许汇总 ΔG_bind。

    缺任何一侧的 summary 时**不判**（记 `not_evaluated` + 原因），因为判不了门
    与"门过了"必须能区分开；这条路径留给 traditional / 后处理模式（它们没有
    replica 轨迹可读）。IBS 生产路径两条腿都会给出 summary。
    """
    lo, hi = LIGAND_CONFORMER_OVERLAP_PERCENTILES
    lo_key, hi_key = f"p{lo:g}", f"p{hi:g}"
    report: Dict[str, Any] = {
        "protocol_version": LIGAND_CONFORMER_DIAGNOSTICS_VERSION,
        "observable": observable,
        "overlap_percentiles": [lo, hi],
        "evaluated": False,
        "passed": None,
        "reason": "",
    }
    if not complex_summary or not solvent_summary:
        missing = [
            name
            for name, value in (("complex", complex_summary), ("solvent", solvent_summary))
            if not value
        ]
        report["reason"] = (
            f"not_evaluated_missing_conformer_summary_for_{'_and_'.join(missing)}"
        )
        return report
    try:
        c = complex_summary["observables"][observable]
        s = solvent_summary["observables"][observable]
    except (KeyError, TypeError):
        report["reason"] = f"not_evaluated_observable_{observable}_absent"
        return report

    c_lo, c_hi = float(c[lo_key]), float(c[hi_key])
    s_lo, s_hi = float(s[lo_key]), float(s[hi_key])
    overlap = min(c_hi, s_hi) - max(c_lo, s_lo)
    union = max(c_hi, s_hi) - min(c_lo, s_lo)
    report.update(
        {
            "evaluated": True,
            "complex_interval_nm": [c_lo, c_hi],
            "solvent_interval_nm": [s_lo, s_hi],
            "overlap_nm": float(overlap),
            "union_nm": float(union),
            "overlap_fraction_of_union": (
                float(overlap / union) if union > 0 else None
            ),
            "complex_mean_nm": float(c["mean"]),
            "solvent_mean_nm": float(s["mean"]),
            "complex_std_nm": float(c["std"]),
            "solvent_std_nm": float(s["std"]),
            "passed": bool(overlap > 0.0),
        }
    )
    if not report["passed"]:
        report["reason"] = (
            "cross_leg_conformer_ensembles_do_not_overlap: "
            f"复合物腿 {observable} 的 [p{lo:g}, p{hi:g}] = [{c_lo:.3f}, {c_hi:.3f}] nm，"
            f"溶剂腿 = [{s_lo:.3f}, {s_hi:.3f}] nm，两者不相交。"
            "两条腿采的不是同一个构象族 ⟹ ΔG_bind = ΔG_solv − ΔG_cplx 没有意义（§3.0）。"
            "[WARN] 严格说不重叠有两种读法：(1) 某条腿的构象系综**没收敛**（被困在一个 basin）；"
            "(2) 两相的构象偏好**真的**差这么多，那个差值本该是 ΔG_bind 的一部分。"
            "在当前每窗口 0.5 ns 的采样下这两者分辨不开 —— 看 "
            "`per_replica_mean_max_internal_heavy_distance_nm`：若各 replica 挤在同一个"
            "窄区间（实测 0.657–0.672 nm，σ=0.005）就是 (1)。所以门保守阻断，"
            "由**溶剂腿双起点验证**（折叠/伸展各跑一遍，ΔG 差 ≤ 2σ）来区分，"
            "**不是**放宽本判据的百分位区间。"
        )
    return report


def assert_cross_leg_conformer_consistency(report: Dict[str, Any]) -> None:
    """门：不重叠就 raise。判不了门（`evaluated=False`）不阻断，但会如实记录。"""
    if report.get("evaluated") and not report.get("passed"):
        raise ValueError(
            "配体构象跨腿一致性门未通过（P0-12a / §3.0）：\n    "
            + str(report.get("reason"))
        )

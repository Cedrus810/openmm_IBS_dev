# ---------------------------------------------------------------------------
# Boresch attachment 腿 (A′→A)
# ---------------------------------------------------------------------------
# 热力学循环里缺的那一项。当前 ΔG_complex = ΔG(A→B) + ΔG_release，其中
# A =「配体耦合 + 限制已打开」；但物理结合态是 A′ =「配体耦合 + 无限制」。
# 缺的正是 A′→A。补上之后，循环对**任意**限制强度都严格闭合——受约束系综里
# 测出来的 charging/vdW 就是对的，不需要（也不能）再单独修正。
#
# 与 stage1/stage2 的关键差别：
#   * 配体全程完全耦合，不碰 lambda_coul / lambda_vdw；
#   * 只有一维 λ（限制强度），没有软核端点奇异性；
#   * 因此不需要副本交换，独立窗口 + MBAR 就够——也就不必去动 REMDManager
#     里那些硬编码的 "lambda_coul"。
BORESCH_ATTACHMENT_PROTOCOL_VERSION = 1
BORESCH_ATTACHMENT_LAMBDA_NAME = "lambda_boresch_scale"
# attachment 腿的监控落盘间隔（步）。2 fs × 500 = 1 ps 一行，一条腿约 2400 行，
# 写盘开销可忽略。这条腿此前**一帧都不写**，2026-08-03 的 NaN 因此没有任何现场证据。
ATTACHMENT_MONITOR_INTERVAL = 500
# 头若干步用细粒度：2026-08-03 定位 MEM-15 时，NaN 发生在第一个 500 步分块内，
# 粗粒度只留下 step 0 一行、夹不住死亡时刻。根因修掉之后细粒度就只剩成本了 ——
# 每行要两次 `getState`（一次 `getForces` = GPU 同步 + 45354×3 个 float 下载，
# 一次取 Boresch 力组能量），25 步一行相当于每 50 fs 打断一次 GPU。
# 放宽到 250 步（0.5 ps）、只覆盖头 2500 步：既留住"起跑阶段出事能夹住"的能力，
# 又不再按 50 fs 停。要重新细化时把这两个数调小即可，逻辑不用动。
ATTACHMENT_MONITOR_FINE_INTERVAL = 250
ATTACHMENT_MONITOR_FINE_STEPS = 2500
BORESCH_ATTACHMENT_FORCE_GROUP = 3

# 升序排列——MBAR 的 delta_G = f[K-1] − f[0]，升序时它直接就是 ΔG(A′→A)，
# 全链路不出现任何负号。
#
# ⚠️ 2026-07-28 更正：初版默认表有 12 个态，理由写的是「有效力常数是 λ·k，
# λ→0 附近相邻态重叠极差，所以近 0 端必须加密」。**那个论证在这里不成立，
# 而且实测正好反过来。** 它的前提是限制势是该坐标的主要约束；但配体在口袋里，
# 蛋白本身已经把它按住了，U_Boresch 在所有 λ 下都被压在 1.6–5.6 kT，于是
# Δu_相邻 = Δλ·U_B/kT 处处很小。12 态实测每条边的 ⟨Δu⟩ 全部 ≤ 0.33 kT
# （λ→0 那几档 0.04/0.08/0.15 反而是全阶梯重叠最好的），而 BAR 的合理区间是
# 每边 1–2 kT——等于密了 3–6 倍。
#
# 用同一批样本做的对照（`attachment_rerun/20260728_130848`）：
#     [0, 1]                2 态  ΔG=5.4317   最大边 3.54 kT
#     [0, 0.25, 1]          3 态  ΔG=5.1329   最大边 1.73 kT
#     [0, 0.1, 0.35, 1]     4 态  ΔG=5.1949   最大边 1.41 kT
#     12 态链式 BAR                ΔG=5.3784   最大边 0.33 kT
#     12 态 MBAR                   ΔG=5.7440
# 全距 0.61 kJ/mol，而 12 态 MBAR 报出的 σ 只有 0.083——**估计量选择造成的
# 散布是报出误差棒的 7 倍**。多花 6 倍态数买不到任何精度。
#
# 本体系真正的瓶颈是**慢构象弛豫**，不是 λ 重叠：12 个密集态把预算切成了
# 12 段短平衡，每段都可能残留上一个态的记忆。省下的态数要投到每态更长采样、
# 多 seed 和双向验证上。
DEFAULT_BORESCH_ATTACHMENT_LAMBDAS = (0.0, 0.1, 0.35, 1.0)


def add_scalable_boresch_restraint(
    system: openmm.System,
    restraint_params: Optional[Dict],
    lam_name: str = BORESCH_ATTACHMENT_LAMBDA_NAME,
    force_group: int = BORESCH_ATTACHMENT_FORCE_GROUP,
) -> bool:
    """加一条**可扫 λ** 的 Boresch 力（注册全局参数）。

    与 `_add_physical_boresch_restraint` 的区别：那个传 `fixed_lam=1.0`，
    `LambdaDependentBoreschForce.__init__` 会把 1.0 直接编译进表达式字符串、
    **连全局参数都不注册**，所以生产采样期的限制强度根本改不了。
    attachment 腿要扫 λ，必须用 `fixed_lam=None` 这一路。
    """
    if not _has_valid_boresch_restraint(restraint_params):
        return False
    force = LambdaDependentBoreschForce(
        rec_idx=restraint_params["receptor_indices"],
        lig_idx=restraint_params["ligand_indices"],
        eq=restraint_params["equilibrium_values"],
        fc=restraint_params["force_constants"],
        lam_name=lam_name,
        fixed_lam=None,
        sign=1.0,
        use_pbc=True,
    )
    force.setForceGroup(force_group)
    system.addForce(force)
    return True


def _assert_force_group_is_free(system: openmm.System, force_group: int) -> None:
    """确认没有别的力占着这个力组。

    attachment 腿靠「只取该力组的能量」来拿 U_Boresch(x)，一旦有别的力混进来，
    拆出来的就不是限制势，整条腿的能量矩阵都会错——而且错得很隐蔽（数值仍然
    有限、MBAR 照样收敛）。所以这里 fail closed。
    """
    occupied = [
        f.__class__.__name__
        for f in system.getForces()
        if f.getForceGroup() == int(force_group)
    ]
    if occupied:
        raise RuntimeError(
            f"力组 {force_group} 已被 {occupied} 占用，无法用它单独取 Boresch 能量。"
            "请换一个空闲力组。"
        )


def adjacent_bar_chain(u_kn, n_k, kt):
    """相邻 BAR 链：逐边双向 BAR 再相加。`u_kn` 必须是**约化**势，返回 kJ/mol。

    适用条件（**硬前提，下面 fail closed**）：每个态自己都有样本。
    满足的腿：stage1 decharging（REMD demux 后每态有帧）、attachment 腿
    （顺序独立窗口）。**不满足的腿：stage2 IBS**——每个窗口只有 row 0（偏置混合
    采样分布）有样本，物理 λ 行 `n_k=0`（见
    `GlobalMBARAnalyzer.solve_stage_integrated` 里 `n_k_local[sampled_row] = n_frames`）。
    把偏置混合帧当端点帧喂给 BAR 会违反其前提，算出来的数看似正常实则无效，
    所以这里对任何零样本态直接抛错，而不是给出一个能被误引用的值。

    为什么它比去相关 MBAR 可信：2026-07-28 实测同一批 attachment 样本上
    TI 5.3867 / 相邻 BAR 5.3784 / 两端 BAR 5.4317 三者一致，而
    `TraditionalMBARAnalyzer` 的**去相关 MBAR** 给 5.7440——选帧把结果偏出
    0.37 kJ/mol（4.4σ）。stage1 上同样的模式：BAR 65.076 / FD-TI 65.126 /
    全帧 MBAR 65.003 一致，去相关 MBAR 64.411 偏低 0.6。BAR 逐边只用相邻两态
    自己的样本，不受全局选帧影响。

    误差按独立边方差相加——边之间其实有相关，这是保守近似而非严格值。
    """
    from pymbar import other_estimators

    n_k = np.asarray(n_k, dtype=int)
    u_kn = np.asarray(u_kn, dtype=np.float64)
    K = len(n_k)
    if K < 2:
        raise ValueError(f"相邻 BAR 至少需要 2 个态，收到 {K}")
    if u_kn.ndim != 2 or u_kn.shape[0] != K:
        raise ValueError(
            f"u_kn 形状 {u_kn.shape} 与态数 {K} 不符；相邻 BAR 需要 (K, N) 的约化势矩阵"
        )
    if int(np.sum(n_k)) != u_kn.shape[1]:
        raise ValueError(
            f"sum(n_k)={int(np.sum(n_k))} 与 u_kn 帧数 {u_kn.shape[1]} 不符"
        )
    zero_states = [int(k) for k in range(K) if n_k[k] <= 0]
    if zero_states:
        raise ValueError(
            f"相邻 BAR 前提不成立：态 {zero_states} 的 n_k=0。BAR 要求两个端点系综"
            "各自有样本。IBS stage2 每窗只有 row 0（偏置混合采样分布）有样本、物理 λ "
            "行 n_k=0，因此 stage2 **不得**使用 BAR（也不得使用线性势 TI：vdW "
            "softcore 对 λ 非线性且未落盘 ∂U/∂λ）；那两格必须写 not_applicable_reason。"
        )
    off = np.concatenate(([0], np.cumsum(n_k))).astype(int)
    total, var, edges = 0.0, 0.0, []
    for k in range(K - 1):
        a = slice(off[k], off[k + 1])
        b = slice(off[k + 1], off[k + 2])
        e = other_estimators.bar(
            u_kn[k + 1, a] - u_kn[k, a],
            u_kn[k, b] - u_kn[k + 1, b],
            compute_uncertainty=True,
        )
        d, s = float(e["Delta_f"]) * kt, float(e["dDelta_f"]) * kt
        total += d
        var += s ** 2
        edges.append({"edge": [int(k), int(k + 1)], "delta_G_kJ_mol": d, "error_kJ_mol": s})
    return total, float(np.sqrt(var)), edges


def _attachment_bar_chain(u_kn, n_k, kt):
    """attachment 腿的主估计量。就是 `adjacent_bar_chain`，保留旧名以免破坏调用方。

    （曾经是独立实现；2026-07-28 stage1 也要用同一个链式 BAR，与其抄第二份，
    直接复用——两者数学完全相同，连误差合并方式都一样。）
    """
    return adjacent_bar_chain(u_kn, n_k, kt)


def _attachment_ti(lam, mean_u_boresch):
    """梯形 TI：∫₀¹⟨∂U/∂λ⟩dλ。势对 λ 严格线性 ⟹ ∂U/∂λ = U_Boresch。

    TI 是三个口径里对稀有大能量帧最不敏感的（普通均值，没有指数放大），
    所以拿它做 BAR 的交叉检查最有分辨力。

    ⚠️ **只对 attachment 腿成立**：Boresch 势乘 λ_boresch 是严格线性的，所以
    ∂U/∂λ 就是 U_Boresch 本身。**不要拿它去做 stage2 的 vdW 腿**——softcore 对 λ
    非线性，线性势假设不成立；那条腿也从未落盘 ∂U/∂λ。需要在已有 λ 网格上估
    ∂U/∂λ 的场合用下面的 `reweighted_fd_ti`（它也只适用于每个态都有样本的腿）。
    """
    lam = np.asarray(lam, dtype=float)
    ub = np.asarray(mean_u_boresch, dtype=float)
    return float(np.trapezoid(ub, lam) if hasattr(np, "trapezoid") else np.trapz(ub, lam))


def reweighted_fd_ti(
    u_kn,
    n_k,
    lambdas: Sequence[float],
    kt: float,
    weights: Optional[np.ndarray] = None,
):
    """⟨∂U/∂λ⟩ 重加权有限差分 → 对 λ 梯形积分，返回 (kJ/mol, 逐态 ⟨∂U/∂λ⟩)。

    `u_kn` 是**约化**势（无量纲），所以差分出来要乘 kT 才是 kJ/mol。
    ∂U/∂λ 用**实际的（非均匀）λ 网格**做中心差分，端点用单边差分。
    **绝不用 state index 代替 λ**——λ 表通常非均匀，用序号会给出错误的梯度。

    `weights=None` 时退化为「每个态用自己的样本块求平均」（普通 TI）；
    给了 MBAR 的 `W_nk` 就是重加权 TI。

    适用条件与 `adjacent_bar_chain` 相同：每个态都要有样本。stage2 IBS 不适用。
    """
    lam = np.asarray(lambdas, dtype=float)
    n_k = np.asarray(n_k, dtype=int)
    u_kn = np.asarray(u_kn, dtype=np.float64)
    K = len(n_k)
    if lam.size != K:
        raise ValueError(f"λ 数 {lam.size} != 态数 {K}")
    if K < 2:
        raise ValueError("λ 表至少需要 2 个态")
    off = np.concatenate(([0], np.cumsum(n_k))).astype(int)

    dudl = np.empty(K, dtype=float)
    for k in range(K):
        lo = max(0, k - 1)
        hi = min(K - 1, k + 1)
        dlam = lam[hi] - lam[lo]
        if abs(dlam) < 1e-12:
            raise ValueError(f"λ[{lo}] 与 λ[{hi}] 重合，无法做差分")
        # (U_hi − U_lo)/(λ_hi − λ_lo)，逐帧算完再按权重平均。乘 kt 把约化势转 kJ/mol。
        per_frame = (u_kn[hi] - u_kn[lo]) * kt / dlam
        if weights is not None:
            w = np.asarray(weights[:, k], dtype=float)
            tot = float(np.sum(w))
            if not np.isfinite(tot) or tot <= 0.0:
                raise RuntimeError(f"态 {k} 的 MBAR 权重和非正，无法重加权")
            dudl[k] = float(np.dot(w / tot, per_frame))
        else:
            if off[k + 1] <= off[k]:
                raise ValueError(f"态 {k} 没有样本，无法做普通 TI 平均")
            dudl[k] = float(np.mean(per_frame[off[k] : off[k + 1]]))

    # ⚠️ 不要翻符号。`np.trapz(y, x)` 沿给定顺序算 ∫_{x[0]}^{x[-1]} y dx，
    # 所以 λ 降序时它**直接就是** ΔG(λ=1 → λ=0)，正是本仓库解耦腿的约定。
    # 数值核对：降序 λ + ∂U/∂λ<0（λ↓ 电荷关掉、U 升）两个负号相乘 → 正值 ✓；
    # 升序 λ（attachment）+ ∂U/∂λ=U_B>0 → 也是正值 ✓。
    # 曾在这里多加过一次 `if lam[0] > lam[-1]: integ = -integ`，会把 stage1 的
    # +65 变成 −65。
    integ = np.trapezoid(dudl, lam) if hasattr(np, "trapezoid") else np.trapz(dudl, lam)
    return float(integ), dudl.tolist()


# 门的绝对下限。σ 可以任意小（弱限制极限下所有边的 ΔG 都≈0，BAR 误差也≈0），
# 这时纯 z 判据会把 1e-5 kJ/mol 的数值噪声判成 7σ——实测就出现过
# 「漂移 +0.0000 kJ/mol = 7.1×2σ」。所以每道门都必须同时超过一个绝对量才失败。
ATTACHMENT_BAR_TI_ABS_TOL_KJ = 1.0
ATTACHMENT_SPLIT_HALF_ABS_TOL_KJ = 0.5


def attachment_bar_ti_gate(dg_bar: float, err: float, dg_ti: float) -> Tuple[bool, float, str]:
    """BAR 与 TI 一致性门。返回 ``(是否通过, 容差, 说明)``。

    两者对稀有大能量帧的敏感度不同：TI 是普通均值，BAR 走指数平均。
    一帧二面角反转（2k ≈ 360–470 kJ/mol）能把 BAR 拉走而 TI 几乎不动，
    分歧就是「估计量被少数帧支配」的信号。
    """
    tol = max(ATTACHMENT_BAR_TI_ABS_TOL_KJ, 3.0 * max(float(err), 0.0))
    diff = abs(float(dg_bar) - float(dg_ti))
    msg = (
        f"BAR({dg_bar:.4f}) 与 TI({dg_ti:.4f}) 差 {diff:.4f} kJ/mol，容差 {tol:.4f}"
    )
    return diff <= tol, tol, msg


def attachment_split_half_gate(drift: float, err: float) -> Tuple[bool, float, str]:
    """前后半程一致性门。返回 ``(是否通过, 容差, 说明)``。

    两个半程各自 SE≈√2σ，其差的 SE≈2σ，所以 z 判据的分母是 2σ。
    再叠一个绝对下限，避免 σ≈0 时把数值噪声判成大偏离。
    """
    tol = max(ATTACHMENT_SPLIT_HALF_ABS_TOL_KJ, 3.0 * 2.0 * max(float(err), 0.0))
    d = abs(float(drift))
    return d <= tol, tol, f"前后半程漂移 {drift:+.4f} kJ/mol，容差 {tol:.4f}"


def run_boresch_attachment_leg(
    system: openmm.System,
    topology,
    positions,
    box_vectors,
    restraint_params: Dict,
    *,
    temperature_k: float = 300.0,
    lambdas: Optional[Sequence[float]] = None,
    n_steps_per_state: int = 250_000,
    equil_steps_per_state: int = 50_000,
    steps_per_sample: int = 1_000,
    platform_name: str = "CUDA",
    seed: int = 0,
    n_seeds: int = 1,
    enforce_convergence_gates: bool = True,
    output_dir: Optional[str] = None,
    log=print,
) -> Dict:
    """测 ΔG(A′→A)：把 Boresch 限制从关到开的自由能代价。

    `enforce_convergence_gates=False` 只关掉 BAR/TI 与 split-half 两道**收敛**门
    （改为打警告），用于故意跑很短轨迹的单元测试；`ΔG ≥ 0`、力组占用、
    几何奇点、λ 阶梯方向这些**正确性**门任何时候都不能关。

    顺序独立窗口（λ 从 1 降到 0 链式推进），**主估计量是相邻 BAR**，TI 作交叉检查。

    2026-07-28 的两次更正，都写在这里免得后人重蹈：

    1. **主值曾接在去相关 MBAR 上，那是 bug。** 同一批样本 TI 5.3867 /
       相邻 BAR 5.3784 一致，而去相关 MBAR 给 5.7440。MBAR 现在降级为诊断。
    2. **不要改用 Hamiltonian REMD。** 试过，产出 38.6±110、零 round trip。
       根因是 Boresch 的 `k(1−cosΔ)` 项在反转时取 2k：单个二面角 359–469 kJ/mol
       （144–188 kT），三个全反转 1189 kJ/mol = 477 kT。交换一旦把副本送进翻转态，
       那一帧的天文 U_B 就支配了指数平均。顺序窗口从不离开原盆地，反而良态。
       **而且单盆地是必要的**：配套的解析释放项（`calculate_boresch_analytical_correction`）
       本身就假定单一简谐盆地，配一个会采到翻转的 attachment 腿反而不自洽。

    λ 升序（0 → 1），所有估计量都直接给 ΔG(A′→A)，全链路不出现负号。

    严格下界：Boresch 势处处 ≥ 0 ⟹ ΔG(A′→A) ≥ 0，算出负值直接 fail closed。
    """
    if not _has_valid_boresch_restraint(restraint_params):
        raise ValueError("attachment 腿需要完整的 3+3 Boresch 锚点")

    try:
        temperature_k = float(temperature_k)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("attachment 温度必须是正有限数") from exc
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("attachment 温度必须是正有限数")

    n_seeds = int(n_seeds)
    if n_seeds < 1:
        raise ValueError(f"n_seeds 必须 ≥1，收到 {n_seeds}")
    try:
        n_steps_per_state = int(n_steps_per_state)
        equil_steps_per_state = int(equil_steps_per_state)
        steps_per_sample = int(steps_per_sample)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "attachment 步数参数必须是整数"
        ) from exc
    if n_steps_per_state < 2:
        raise ValueError("attachment n_steps_per_state 必须至少为 2")
    if equil_steps_per_state < 0:
        raise ValueError("attachment equil_steps_per_state 不能为负")
    if steps_per_sample <= 0 or steps_per_sample > n_steps_per_state:
        raise ValueError(
            "attachment steps_per_sample 必须为正且不超过 n_steps_per_state"
        )
    if n_seeds > 1:
        common = dict(
            temperature_k=temperature_k, lambdas=lambdas,
            n_steps_per_state=n_steps_per_state,
            equil_steps_per_state=equil_steps_per_state,
            steps_per_sample=steps_per_sample,
            platform_name=platform_name, n_seeds=1, log=log,
            enforce_convergence_gates=enforce_convergence_gates,
        )
        runs = []
        for r in range(n_seeds):
            log(f"\n  ── seed {r + 1}/{n_seeds} ──")
            runs.append(run_boresch_attachment_leg(
                system, topology, positions, box_vectors, restraint_params,
                seed=int(seed) + 104729 * r,
                output_dir=(os.path.join(output_dir, f"seed{r}") if output_dir else None),
                **common,
            ))
        vals = np.array([r["attachment_delta_G_kJ_mol"] for r in runs], dtype=float)
        mean = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        log(f"\n  [OK] 跨 {n_seeds} 个 seed: ΔG(A′→A) = {mean:.4f} ± {sem:.4f} kJ/mol "
            f"= {mean / 4.184:.4f} ± {sem / 4.184:.4f} kcal/mol")
        log(f"     逐 seed: {[round(v, 4) for v in vals.tolist()]}")
        return {
            "stage": "boresch_attachment",
            "converged": bool(all(r.get("converged") is True for r in runs)),
            "protocol_version": int(BORESCH_ATTACHMENT_PROTOCOL_VERSION),
            "method": "sequential-windows-adjacent-BAR",
            "attachment_delta_G_kJ_mol": mean,
            "attachment_error_kJ_mol": sem,
            "uncertainty_source": "across_seed_sem",
            "n_seeds": n_seeds,
            "per_seed_delta_G_kJ_mol": vals.tolist(),
            "per_seed_spread_kJ_mol": float(np.max(vals) - np.min(vals)),
            "direction": "A_prime_to_A (restraint OFF -> ON), always >= 0",
            "lambdas": runs[0]["lambdas"],
            "per_seed": runs,
        }

    lam = np.asarray(
        list(lambdas) if lambdas is not None else list(DEFAULT_BORESCH_ATTACHMENT_LAMBDAS),
        dtype=np.float64,
    )
    if lam.ndim != 1 or lam.size < 2:
        raise ValueError(f"λ 阶梯至少 2 个态（两端双向 BAR），收到 {lam.size} 个")
    if not (np.all(np.diff(lam) > 0) and abs(lam[0]) < 1e-12 and abs(lam[-1] - 1.0) < 1e-12):
        raise ValueError(
            f"λ 阶梯必须严格升序且从 0 到 1（否则 ΔG 的符号含义就变了），收到 {lam.tolist()}"
        )

    kt = 0.008314462618 * float(temperature_k)
    K = int(lam.size)

    work = openmm.XmlSerializer.clone(system)
    work.thisown = 1
    _assert_force_group_is_free(work, BORESCH_ATTACHMENT_FORCE_GROUP)
    if not add_scalable_boresch_restraint(work, restraint_params):
        raise RuntimeError("Boresch 力注入失败")

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(int(seed) or 1)
    resolved_platform, platform_props = _build_platform_properties(platform_name)
    simulation = app.Simulation(
        topology,
        work,
        integrator,
        openmm.Platform.getPlatformByName(resolved_platform),
        platform_props,
    )
    simulation.context.setPositions(positions)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)

    safe, r0_chk, tha_chk, thb_chk = _check_boresch_geometry_safe(
        simulation.context, restraint_params
    )
    if not safe:
        raise RuntimeError(
            f"attachment 腿起始构象的 Boresch 锚点几何不安全（r0={r0_chk}, "
            f"thetaA={tha_chk}°, thetaB={thb_chk}°）：角度接近 0°/180°，"
            "力的解析梯度会发散。请换锚点或起始构象。"
        )
    log(f"  [OK] 锚点几何: r0={r0_chk:.4f} nm  θA={tha_chk:.2f}°  θB={thb_chk:.2f}°")

    # 🔑 [0831issue P2] `max(2, ...)` 这个下限会让**实际步数超过用户设定**：采样循环
    # 按 n_samples 次 × steps_per_sample 步推进，所以当
    # steps_per_sample > n_steps_per_state/2 时（短 pilot），下限把帧数抬到 2，实跑步数
    # 变成 2*steps_per_sample > n_steps_per_state。同时 split-half 被切成每半 1 帧，
    # 那个 z 检验退化成无意义的常量。生产配置 250000/1000 → 250 帧，不受影响。
    # 与其静默跑超预算，不如直接告诉调用方参数组合不成立。
    _n_samples_raw = int(n_steps_per_state) // int(steps_per_sample)
    if _n_samples_raw < 2:
        raise ValueError(
            f"attachment 腿参数组合不可用：n_steps_per_state={int(n_steps_per_state)} / "
            f"steps_per_sample={int(steps_per_sample)} 只能取到 {_n_samples_raw} 帧（< 2）。"
            "以前这里用 max(2, ...) 兜底，代价是实跑步数悄悄变成 "
            f"{2 * int(steps_per_sample)} 步（超过设定的 {int(n_steps_per_state)} 步），"
            "且 split-half 每半只有 1 帧、z 检验失效。请增大 n_steps_per_state 或"
            "减小 steps_per_sample。"
        )
    n_samples = _n_samples_raw
    u_boresch = np.zeros((K, n_samples), dtype=np.float64)
    u_base = np.zeros((K, n_samples), dtype=np.float64)

    order = list(range(K - 1, -1, -1))   # 从全强度端往下走
    log(f"  attachment 腿：{K} 个 λ 态 {np.round(lam, 4).tolist()}")
    log(
        f"  [WARN] 扫描顺序是**从全强度端往下**：第一个实际跑的态是 "
        f"λ={float(lam[order[0]]):.4f}，不是列表里的第一个 λ。"
    )
    log(f"  每态 平衡 {equil_steps_per_state} 步 + 生产 {n_steps_per_state} 步 → {n_samples} 帧")

    # ---- 起点体检（MEM-06）：把"跑了几步的无上下文 NaN"变成"起点就坏，坏在哪" ----
    #
    # 实测背景：memtest 膜体系 2026-08-02 在这里出 `Particle coordinate is NaN`。
    # 那次第一个跑的态是 λ=1.0（**全强度** Boresch 限制力，kr=946.7 kJ/mol/nm²），
    # 起点坐标刚经过两次 PBC 修复 + 两次锚点重推导（r0 0.705 → 0.767 nm、
    # θA 61.3° → 63.3°）。所以"限制力参数"与"起点坐标"都是嫌疑，
    # 而当时的日志两样都没留下 —— 下面把它们全部量出来再落盘。
    #
    # ⚠️ 这段**只读**：不最小化、不改坐标/速度、不改任何 global parameter
    # （λ 会在下面的循环里由 order[0] 重新设成同一个值）。所以对既有可溶生产路径
    # 的数值逐位无影响（§7.7）。
    simulation.context.setParameter(
        BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[order[0]])
    )
    log(
        "  Boresch 力常数: "
        + ", ".join(
            f"{name}={float(restraint_params['force_constants'][name]):.4g}"
            for name in ("kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC")
            if name in (restraint_params.get("force_constants") or {})
        )
    )
    try:
        _state = simulation.context.getState(getPositions=True)
        _measured = calc_boresch_from_last_frame(
            _state.getPositions(asNumpy=True),
            restraint_params["receptor_indices"],
            restraint_params["ligand_indices"],
            # BOR-01：运行中的 context state 没有成像保证，解耦配体漂过边界正是
            # 最可能触发的场景。
            box_vectors=_state.getPeriodicBoxVectors(),
        )
        _eq = restraint_params.get("equilibrium_values") or {}
        log(
            "  起点实测几何 vs 已提交平衡值: "
            + ", ".join(
                f"{key}: {float(_measured[key]):.4f} / {float(_eq[key]):.4f}"
                for key in ("r0", "thetaA0", "thetaB0", "phiA0", "phiB0", "phiC0")
                if key in _measured and key in _eq
            )
        )
    except Exception as exc:  # noqa: BLE001
        # 诊断失败不该顶替真正的失败，但也不许静默 —— 说出来，然后继续体检受力。
        log(f"  [WARN] 起点几何诊断未能完成（不阻断）：{type(exc).__name__}: {exc}")
    _group_energies = {}
    for _group in (0, BORESCH_ATTACHMENT_FORCE_GROUP):
        try:
            _group_energies[_group] = float(
                simulation.context.getState(getEnergy=True, groups={_group})
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
        except Exception:  # noqa: BLE001
            pass
    if _group_energies:
        log(
            "  逐 force group 能量 (kJ/mol): "
            + ", ".join(f"g{g}={e:.6g}" for g, e in sorted(_group_energies.items()))
        )
    # ---- 入口快照 + 在跑中监控（2026-08-03）----
    #
    # 为什么必须有：这条腿此前**一帧轨迹、一行监控都不写**，所以 2026-08-03 的
    # `Particle coordinate is NaN` 除了一个 traceback 什么证据都没留下。
    # 而离线忠实重放（同起点、同种子、同步数，走完整条 λ 序列 2.4 ns）
    # **不复现** —— 也就是说生产与重放之间有差异，但没有任何落盘的东西能拿来 diff。
    # §0.5.7 的教训原话是"离线再猜性价比很低，必须在跑中留下轨迹"。
    _diag_dir = output_dir or "."
    try:
        os.makedirs(_diag_dir, exist_ok=True)
        _forces = []
        for _i, _f in enumerate(work.getForces()):
            _entry = {
                "index": _i,
                "class": _f.__class__.__name__,
                "force_group": int(_f.getForceGroup()),
            }
            if hasattr(_f, "usesPeriodicBoundaryConditions"):
                try:
                    _entry["uses_pbc"] = bool(_f.usesPeriodicBoundaryConditions())
                except Exception:  # noqa: BLE001
                    pass
            for _attr in ("getNonbondedMethod", "getCutoffDistance", "getNumParticles"):
                if hasattr(_f, _attr):
                    try:
                        _v = getattr(_f, _attr)()
                        _entry[_attr[3:]] = (
                            float(_v.value_in_unit(unit.nanometer))
                            if hasattr(_v, "value_in_unit") else int(_v)
                        )
                    except Exception:  # noqa: BLE001
                        pass
            _forces.append(_entry)
        _pos_nm = np.asarray(
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        _box_nm = np.asarray(
            [v.value_in_unit(unit.nanometer)
             for v in simulation.context.getState().getPeriodicBoxVectors()],
            dtype=np.float64,
        )
        # 锚点的 raw 距离 vs minimum-image 距离：两者不等就说明锚点分处不同周期镜像，
        # 而 `CustomCompoundBondForce` 的 angle()/dihedral() 不做 minimum-image。
        _rec = [int(i) for i in restraint_params["receptor_indices"]]
        _lig = [int(i) for i in restraint_params["ligand_indices"]]
        _anchor_pairs = {}
        for _tag, _a, _b in (
            ("R0-L0", _rec[0], _lig[0]), ("R1-R0", _rec[1], _rec[0]),
            ("R2-R1", _rec[2], _rec[1]), ("L1-L0", _lig[1], _lig[0]),
            ("L2-L1", _lig[2], _lig[1]),
        ):
            _d = _pos_nm[_a] - _pos_nm[_b]
            # Use the same closest-lattice implementation as all production
            # geometry paths.  The former diagonal ``round`` approximation was
            # silently wrong for a triclinic box and made this diagnostic
            # disagree with the actual OpenMM periodic geometry.
            _mi = minimum_image_displacement_nm(_d, _box_nm)
            _anchor_pairs[_tag] = {
                "raw_nm": float(np.linalg.norm(_d)),
                "minimum_image_nm": float(np.linalg.norm(_mi)),
                "differs": bool(
                    abs(np.linalg.norm(_d) - np.linalg.norm(_mi)) > 1.0e-6
                ),
            }
        _snapshot = {
            "n_particles": int(work.getNumParticles()),
            "n_constraints": int(work.getNumConstraints()),
            "forces": _forces,
            "box_vectors_nm": _box_nm.tolist(),
            "positions_sha256": hashlib.sha256(
                np.ascontiguousarray(_pos_nm, dtype=np.float64).tobytes()
            ).hexdigest(),
            "positions_min_nm": _pos_nm.min(axis=0).tolist(),
            "positions_max_nm": _pos_nm.max(axis=0).tolist(),
            "anchor_pair_distances": _anchor_pairs,
            "lambdas": [float(v) for v in lam],
            "scan_order_k": [int(k) for k in order],
            "seed": int(seed),
            "equil_steps_per_state": int(equil_steps_per_state),
            "n_steps_per_state": int(n_steps_per_state),
            "steps_per_sample": int(steps_per_sample),
            "restraint": {
                "equilibrium_values": restraint_params.get("equilibrium_values"),
                "force_constants": restraint_params.get("force_constants"),
                "receptor_indices": _rec,
                "ligand_indices": _lig,
            },
        }
        _snap_path = os.path.join(_diag_dir, "stage0_attachment_inputs.json")
        with open(_snap_path, "w", encoding="utf-8") as _h:
            json.dump(_snapshot, _h, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        log(f"  attachment 腿入口快照已落盘: {_snap_path}")
        # ⚠️ **坐标本身**也要存，不能只存 SHA256。
        # 2026-08-03 的教训：只有哈希时，离线既无法复现也无法逐力分解 ——
        # 那次 NaN 发生在第一个 500 步分块内（< 1 ps），而用 rebalance 末帧做的
        # 离线重放（同种子、同步数）走完 2.4 ns 都不崩，两者 PE 差 31 MJ/mol。
        # 差异只能来自坐标，而坐标没落盘，于是每查一步都要重跑一次生产。
        # 1.1 MB 换"能离线确定性复现"，永远值。
        _start_path = os.path.join(_diag_dir, "stage0_attachment_start.npz")
        np.savez_compressed(
            _start_path, positions_nm=_pos_nm, box_vectors_nm=_box_nm
        )
        log(f"  attachment 腿起点坐标已落盘（可离线复现）: {_start_path}")
        _torn = [t for t, v in _anchor_pairs.items() if v["differs"]]
        if _torn:
            log(
                f"  [WARN] 锚点对 {_torn} 的 raw 距离与 minimum-image 距离不同 —— "
                "说明锚点分处不同周期镜像。`CustomCompoundBondForce` 的 "
                "angle()/dihedral() **不做** minimum-image，这会让被约束的几何量"
                "不是你以为的那个。"
            )
            raise RuntimeError(
                "attachment Boresch 锚点跨越周期边界；angle/dihedral 的 PBC 几何"
                "无法与 minimum-image 参考一致，拒绝在撕裂的锚点几何上采样。"
            )
    except RuntimeError:
        raise
    except Exception as _exc:  # noqa: BLE001
        log(f"  [WARN] 入口快照落盘失败（不阻断）: {type(_exc).__name__}: {_exc}")

    _monitor_path = os.path.join(_diag_dir, "stage0_attachment_monitor.csv")
    _monitor = None
    try:
        _monitor = open(_monitor_path, "w", encoding="utf-8", buffering=1)
        _monitor.write("cumulative_step,lambda_index_k,lambda,phase,"
                       "potential_kJ_mol,temperature_K,max_force_kJ_mol_nm,"
                       "max_force_atom_index,boresch_energy_kJ_mol\n")
        log(f"  attachment 腿监控已启用（每 {ATTACHMENT_MONITOR_INTERVAL} 步）: "
            f"{_monitor_path}")
    except Exception as _exc:  # noqa: BLE001
        log(f"  [WARN] 监控无法写入（不阻断）: {_exc}")
        _monitor = None

    _n_dof = 3 * work.getNumParticles() - work.getNumConstraints()

    def _monitor_row(cumulative_step, k_index, lam_value, phase, simulation_obj=simulation):
        """写一行监控。**崩之前的最后几行就是唯一的现场证据。**"""
        if _monitor is None:
            return
        try:
            st = simulation_obj.context.getState(getEnergy=True, getForces=True)
            pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            temp = 2.0 * ke / (_n_dof * 0.008314462618) if _n_dof > 0 else float("nan")
            fmag = np.linalg.norm(
                np.asarray(st.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer), dtype=float),
                axis=1,
            )
            worst = int(np.argmax(fmag)) if fmag.size else -1
            ub = simulation_obj.context.getState(
                getEnergy=True, groups={BORESCH_ATTACHMENT_FORCE_GROUP}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            _monitor.write(
                f"{int(cumulative_step)},{int(k_index)},{float(lam_value):.6f},"
                f"{phase},{pe:.6f},{temp:.4f},{float(fmag.max()):.6f},{worst},"
                f"{ub:.6f}\n"
            )
        except Exception:  # noqa: BLE001
            pass

    assert_starting_state_is_sane(
        simulation.context,
        topology,
        label=f"attachment 腿起点 λ={float(lam[order[0]]):.4f}",
        remediation=(
            "    这一步**没有**做最小化：起点坐标是上游交给本腿的（预平衡 → 带 Boresch\n"
            "    的 rebalance → PBC 修复 → 锚点重推导）。所以要分两种情况看：\n"
            "      * 上面 g0（基础力）已经异常 → 坏的是起点坐标本身，与 Boresch 无关，\n"
            "        查 PBC 修复是否撕开了分子（§0.5.7）以及拓扑有没有键；\n"
            "      * 只有 Boresch 力组异常 → 查上面那行「实测几何 vs 已提交平衡值」，\n"
            "        以及力常数是否过硬。`--boresch-source simple` 是纯几何涨落估算，\n"
            "        可与 `auto`（MACE 打分、偏好大 kr）对照。"
        ),
        log=log,
    )

    linearity_checked = False
    _cumulative = 0
    for k in order:
        simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[k]))
        simulation.context.setVelocitiesToTemperature(
            temperature_k * unit.kelvin, int(seed) + 7919 * k + 1
        )
        _monitor_row(_cumulative, k, lam[k], "state_start")
        # 平衡段分块步进，只为了能在崩之前留下监控行；总步数与分块前完全相同。
        #
        # 头 `ATTACHMENT_MONITOR_FINE_STEPS` 步用**细**粒度：2026-08-03 那次 NaN
        # 就发生在第一个 500 步分块内，粗粒度只留下了 step 0 一行，等于没夹住。
        _remaining = int(equil_steps_per_state)
        while _remaining > 0:
            _interval = (
                ATTACHMENT_MONITOR_FINE_INTERVAL
                if _cumulative < ATTACHMENT_MONITOR_FINE_STEPS
                else ATTACHMENT_MONITOR_INTERVAL
            )
            _chunk = min(_interval, _remaining)
            guarded_step(simulation, _chunk, f"attachment 腿 λ 态 {k} 预平衡")
            _remaining -= _chunk
            _cumulative += _chunk
            _monitor_row(_cumulative, k, lam[k], "equil")
        for s in range(n_samples):
            guarded_step(simulation, int(steps_per_sample), f"attachment 腿 λ 态 {k} 采样")
            _cumulative += int(steps_per_sample)
            if s % max(1, ATTACHMENT_MONITOR_INTERVAL // int(steps_per_sample)) == 0:
                _monitor_row(_cumulative, k, lam[k], "sample")
            total = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            total = total.value_in_unit(unit.kilojoule_per_mole)
            # U_B 一律在 λ=1 下直接量，不用「力组能量 / λ_k」反推：
            # λ=0 时无定义，λ=0.01 会把单精度舍入放大 100 倍。
            simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, 1.0)
            u_b = simulation.context.getState(
                getEnergy=True, groups={BORESCH_ATTACHMENT_FORCE_GROUP}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[k]))
            u_boresch[k, s] = u_b
            u_base[k, s] = total - float(lam[k]) * u_b

            if not linearity_checked:
                for probe in (0.35, 0.8):
                    simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, probe)
                    direct = simulation.context.getState(
                        getEnergy=True
                    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    predicted = u_base[k, s] + probe * u_b
                    if abs(direct - predicted) > 1.0e-4 * max(1.0, abs(direct)):
                        raise RuntimeError(
                            f"Boresch 势对 λ 的线性假设不成立：λ={probe} 直接测 {direct:.6f}，"
                            f"按 U_0+λ·U_B 预测 {predicted:.6f}。力组可能被别的力共用。"
                        )
                simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[k]))
                linearity_checked = True
                log("  [OK] U(λ) 对 λ 的线性性自检通过（λ=0.35 / 0.8 两点）")

        log(f"    λ={lam[k]:.4f}  ⟨U_Boresch⟩={float(np.mean(u_boresch[k])):9.3f} "
            f"± {float(np.std(u_boresch[k])):7.3f}  max={float(np.max(u_boresch[k])):9.3f} kJ/mol")

    if _monitor is not None:
        try:
            _monitor.close()
        except Exception:  # noqa: BLE001
            pass

    n_k = np.full(K, n_samples, dtype=int)
    u_kn = np.zeros((K, K * n_samples), dtype=np.float64)
    for k in range(K):
        cols = slice(k * n_samples, (k + 1) * n_samples)
        for j in range(K):
            u_kn[j, cols] = (u_base[k] + lam[j] * u_boresch[k]) / kt

    mean_ub = [float(np.mean(u_boresch[k])) for k in range(K)]
    max_ub = float(np.max(u_boresch))

    # ---- 主估计量：相邻 BAR ----
    dg, err, bar_edges = _attachment_bar_chain(u_kn, n_k, kt)
    # ---- 交叉检查：TI ----
    dg_ti = _attachment_ti(lam, mean_ub)
    # ---- 诊断：去相关 MBAR（历史主值，已降级）----
    try:
        an = TraditionalMBARAnalyzer(temperature=temperature_k)
        an._last_n_k = n_k
        _m = an.solve(u_kn, decorrelate=True)
        dg_mbar, err_mbar = float(_m["delta_G"]), float(_m["error"])
    except Exception as exc:
        dg_mbar, err_mbar = float("nan"), float("nan")
        log(f"  [WARN] MBAR 诊断失败（不影响主值）: {exc}")

    log(f"  BAR(主) {dg:.4f} ± {err:.4f} | TI {dg_ti:.4f} | MBAR(诊断) {dg_mbar:.4f}")
    if not np.isfinite(dg):
        raise RuntimeError(f"attachment 腿 ΔG 非有限值: {dg}")
    if dg < 0.0:
        raise RuntimeError(
            f"attachment 腿算出 ΔG(A′→A) = {dg:.4f} kJ/mol < 0，数学上不可能："
            "Boresch 势处处 ≥0 ⟹ ⟨exp(−βU_rest)⟩ ≤ 1 ⟹ ΔG ≥ 0。拒绝返回。"
        )

    # 二面角反转告警：单个二面角反转就要 2k ≈ 360–470 kJ/mol。一旦采到，
    # 指数平均会被那一帧支配（HREMD 那轮 38.6±110 就是这么来的）。
    fc = restraint_params.get("force_constants", {})
    flip_scale = 2.0 * min(
        [float(fc[k]) for k in ("kphiA", "kphiB", "kphiC") if k in fc] or [np.inf]
    )
    dihedral_flip_seen = bool(np.isfinite(flip_scale) and max_ub > 0.5 * flip_scale)
    if dihedral_flip_seen:
        log(f"  [WARN] 采到了 U_B={max_ub:.1f} kJ/mol 的帧（最软二面角反转标度 {flip_scale:.1f}），"
            "指数平均可能被单帧支配；BAR/TI 一致性门是唯一的拦截。")

    # ---- 门：只在真不一致时失败 ----
    ti_ok, ti_tol, ti_msg = attachment_bar_ti_gate(dg, err, dg_ti)
    if not ti_ok:
        if enforce_convergence_gates:
            raise RuntimeError(
                f"attachment 腿 {ti_msg}。两者对稀有大能量帧的敏感度不同，"
                "分歧说明估计量被少数帧支配（多半是二面角反转）。"
            )
        log(f"  [WARN] [门已关闭] {ti_msg} —— 超容差但未阻断")
    mbar_z = abs(dg_mbar - dg) / err if (err > 0 and np.isfinite(dg_mbar)) else float("nan")
    if np.isfinite(mbar_z) and mbar_z > 3.0:
        log(f"  [WARN] 去相关 MBAR({dg_mbar:.4f}) 偏离主值 {mbar_z:.1f}σ —— 只记录不失败，"
            "MBAR 已降级为诊断。")

    half = {}
    try:
        for label, (lo, hi) in (("first", (0.0, 0.5)), ("second", (0.5, 1.0))):
            cols, nk2 = [], []
            for k in range(K):
                a = k * n_samples
                s, e = a + int(lo * n_samples), a + int(hi * n_samples)
                cols.append(np.arange(s, e)); nk2.append(e - s)
            v, _, _ = _attachment_bar_chain(
                u_kn[:, np.concatenate(cols)], np.asarray(nk2, dtype=int), kt
            )
            half[label] = v
        half["drift"] = half["second"] - half["first"]
        half["drift_over_2sigma"] = abs(half["drift"]) / (2.0 * err) if err > 0 else None
        sh_ok, sh_tol, sh_msg = attachment_split_half_gate(half["drift"], err)
        half["tolerance_kJ_mol"] = sh_tol
        half["passed"] = bool(sh_ok)
        log(f"  前后半程 {half['first']:.4f} / {half['second']:.4f}，{sh_msg}")
        if not sh_ok:
            if enforce_convergence_gates:
                raise RuntimeError(f"attachment 腿{sh_msg}，系综未收敛，拒绝返回。")
            log(f"  [WARN] [门已关闭] {sh_msg} —— 超容差但未阻断")
    except RuntimeError:
        raise
    except Exception as exc:
        half = {"error": str(exc)}
        log(f"  [WARN] split-half 诊断失败（不影响主值）: {exc}")

    payload = {
        "stage": "boresch_attachment",
        # When convergence gates are deliberately disabled for a short pilot,
        # preserve their outcome in the result instead of claiming completion.
        # Production callers leave the gates enabled and will already have
        # raised on a failed check.
        "converged": bool(
            ti_ok and isinstance(half, dict) and half.get("passed") is True
        ),
        "protocol_version": int(BORESCH_ATTACHMENT_PROTOCOL_VERSION),
        "method": "sequential-windows-adjacent-BAR",
        "attachment_delta_G_kJ_mol": dg,
        "attachment_error_kJ_mol": err,
        "uncertainty_source": "adjacent_bar_independent_edge_variances",
        "n_seeds": 1,
        "direction": "A_prime_to_A (restraint OFF -> ON), always >= 0",
        "lambdas": lam.tolist(),
        "n_states": K,
        "n_samples_per_state": int(n_samples),
        "n_steps_per_state": int(n_steps_per_state),
        "equil_steps_per_state": int(equil_steps_per_state),
        "temperature_K": float(temperature_k),
        "boresch_params": {
            "receptor_indices": [int(i) for i in restraint_params["receptor_indices"]],
            "ligand_indices": [int(i) for i in restraint_params["ligand_indices"]],
            "equilibrium_values": dict(restraint_params["equilibrium_values"]),
            "force_constants": dict(restraint_params["force_constants"]),
        },
        "mean_u_boresch_per_state_kJ_mol": mean_ub,
        "max_u_boresch_kJ_mol": max_ub,
        "dihedral_flip_scale_kJ_mol": None if not np.isfinite(flip_scale) else float(flip_scale),
        "dihedral_flip_sampled": dihedral_flip_seen,
        "crosschecks": {
            "primary_estimator": "adjacent_bar",
            "adjacent_bar_kJ_mol": dg,
            "adjacent_bar_error_kJ_mol": err,
            "bar_edges": bar_edges,
            "thermodynamic_integration_kJ_mol": dg_ti,
            "bar_minus_ti_kJ_mol": float(dg - dg_ti),
            "bar_ti_tolerance_kJ_mol": float(ti_tol),
            "decorrelated_mbar_kJ_mol": dg_mbar,
            "decorrelated_mbar_error_kJ_mol": err_mbar,
            "decorrelated_mbar_z_vs_primary": None if not np.isfinite(mbar_z) else float(mbar_z),
            "mbar_note": "去相关 MBAR 已降级为诊断：2026-07-28 实测它的选帧把结果偏出 0.37 kJ/mol",
        },
        "split_half": half,
    }

    log(f"  [OK] ΔG(A′→A) = {dg:.4f} ± {err:.4f} kJ/mol "
        f"= {dg / 4.184:.4f} ± {err / 4.184:.4f} kcal/mol")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "attachment_u_kn.npy"), u_kn)
        np.save(os.path.join(output_dir, "attachment_u_kn.npy.n_k.npy"), n_k)
        with open(os.path.join(output_dir, "attachment_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # [P0-REMD-CUDA] 显式释放 attachment 腿的 CUDA Context。
    #
    # 本函数整段原来**没有任何 teardown**：`simulation` 是局部变量，只靠函数返回时的
    # 引用计数回收。这在"下一步立刻要建 12 个 replica Context"的场景下太脆——
    # 引用一旦被 payload / 闭包 / 异常 traceback 之一勾住，显存就一直挂着，而
    # Stage 1 实测开跑前卡上已被占 12197/16303 MiB（只剩 3646，12 个 Context 需 3804）。
    # 与预平衡那段同一口径：显式断链 + gc，并把释放前后的显存打出来，对不上账就看得见。
    #
    # ⚠️ 量级要诚实：这里只有**一个** Context ≈ 317 MiB，所以它**不足以**解释那
    # 12.2 GB 里对不上账的约 10 GB。修它是因为它本来就该修，不是因为它是元凶。
    _vram_before_release = _gpu_memory_mib()
    try:
        del simulation.context
    except Exception:  # noqa: BLE001
        pass
    try:
        del simulation, integrator
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    _vram_after_release = _gpu_memory_mib()
    if _vram_before_release and _vram_after_release:
        log(
            f"  attachment 腿 Context 已释放 | 显存 used "
            f"{_vram_before_release[0]} → {_vram_after_release[0]} MiB "
            f"(free {_vram_before_release[1]} → {_vram_after_release[1]} MiB)"
        )
    return payload



def _prepare_pme_mixed_alchemical_system(
    system_template: openmm.System,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors=None,
    lambda_coul_name: str = "lambda_coul",
    lambda_vdw_name: str = "lambda_vdw",
    restraint_params: Optional[Dict] = None,
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
) -> openmm.System:
    """构建同时支持 PME 去电荷与软核去 VDW 的联合炼金体系。"""
    mixed_sys = _prepare_pme_coulomb_leg_system(
        system_template,
        ligand_indices,
        lambda_name=lambda_coul_name,
        allow_charged_ligand=True,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        co_alchemical_ion_spec=co_alchemical_ion_spec,
    )
    mixed_sys.thisown = 1

    nb_force = next(
        (f for f in mixed_sys.getForces() if isinstance(f, openmm.NonbondedForce)),
        None,
    )
    if nb_force is None:
        raise RuntimeError("联合 PME/VDW 炼金体系未找到 NonbondedForce。")

    ligand_set = set(int(i) for i in ligand_indices)
    environment_indices = [
        idx for idx in range(mixed_sys.getNumParticles()) if idx not in ligand_set
    ]
    particle_params = [
        nb_force.getParticleParameters(i) for i in range(nb_force.getNumParticles())
    ]

    _restore_ligand_internal_nonbonded(
        mixed_sys,
        nb_force,
        ligand_indices,
        zero_original_exceptions=True,
    )

    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(int(idx))
        nb_force.setParticleParameters(
            int(idx),
            q,
            0.1 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )

    for exc_idx in range(nb_force.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb_force.getExceptionParameters(exc_idx)
        p1 = int(p1)
        p2 = int(p2)
        if (p1 in ligand_set) ^ (p2 in ligand_set):
            nb_force.setExceptionParameters(
                exc_idx,
                p1,
                p2,
                charge_prod,
                sig,
                0.0 * unit.kilojoule_per_mole,
            )

    sc_force = BeutlerSoftcoreBuilder.build(
        nb_force,
        list(ligand_indices),
        environment_indices,
        particle_params_override=particle_params,
    )
    sc_force.setForceGroup(1)
    mixed_sys.addForce(sc_force)
    _add_physical_boresch_restraint(mixed_sys, restraint_params, force_group=3)
    return mixed_sys


def _build_traditional_mbar_eval_context(
    system_xml: str,
    platform_name: str,
    cpu_threads: Optional[int] = None,
):
    """为离线 MBAR worker 构建独立的 OpenMM Context。

    `cpu_threads`：见 `_build_platform_properties`——离线 MBAR 常常是
    `n_workers` 个 CPU 进程同时跑，每个进程的 Context 如果不设线程上限，
    默认会各自尝试用满所有物理核，造成进程数×线程数的过度并行。调用方
    （见下方 `_mbar_worker_init`/`compute_u_kn`）按 `physical_cores //
    n_workers` 算好预算后传进来。
    """
    eval_sys = openmm.XmlSerializer.deserialize(system_xml)
    resolved_platform, props = _build_platform_properties(platform_name, cpu_threads=cpu_threads)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(eval_sys, integ, platform, props)
    return eval_sys, integ, ctx


# 🔑 [性能修复：worker 级 Context 复用] 只有被带 initializer 的 multiprocessing
# Pool 派生出的 worker 子进程才会调用 _mbar_worker_init 把这个填上；直接调用
# _compute_u_kn_chunk 的两条串行路径（n_workers==1 / 多进程失败后的单进程回退，
# 见 compute_u_kn）永远在主进程里跑，这个全局量在主进程里永远是 None，因此这两
# 条路径的行为跟本次改动前逐位一致（各自独立反序列化 System、建 Context）。
_MBAR_WORKER_CTX: Optional[Tuple[Any, Any, Any]] = None


def _mbar_worker_init(system_xml: str, platform_name: str, cpu_threads: Optional[int]) -> None:
    """`multiprocessing.Pool(initializer=...)` 钩子：worker 进程启动时只反
    序列化一次 System、建一次 Context，此后由这个 worker 通过
    `imap_unordered` 拉到的每个 chunk 任务都复用同一份，不再像之前那样
    `_compute_u_kn_chunk` 每次调用都重新反序列化+重新建 Context。只在一次
    `compute_u_kn` 调用内部临时创建的 Pool 生命周期内有效——Pool 退出时
    worker 进程被销毁，不会跨调用残留旧的 System/Context。
    """
    global _MBAR_WORKER_CTX
    _MBAR_WORKER_CTX = _build_traditional_mbar_eval_context(
        system_xml=system_xml,
        platform_name=platform_name,
        cpu_threads=cpu_threads,
    )


def _compute_u_kn_chunk(task: Dict) -> Tuple[int, np.ndarray]:
    """多进程 worker：重算一个帧块在所有 λ 态下的约化势。

    [性能修复] 如果这个进程是被带 `initializer=_mbar_worker_init` 的 Pool
    派生出来的，`_MBAR_WORKER_CTX` 在这个 worker 进程里已经被填好，直接复用
    （同一个 worker 处理的所有 chunk 共享一份 System/Context，不再每个 chunk
    都反序列化+重建）；否则（`n_workers==1` 的串行路径、或多进程失败后的
    单进程回退路径，两者都在主进程里跑，主进程从不调用
    `_mbar_worker_init`，`_MBAR_WORKER_CTX` 在那里恒为 None）保持原来的行为：
    从 `task["system_xml"]`/`task["platform_name"]` 独立建一份、用完即删。
    """
    frame_offset = int(task["frame_offset"])
    xyz_chunk = np.asarray(task["xyz"], dtype=np.float64)
    box_chunk = task.get("box_vectors")
    if box_chunk is not None:
        box_chunk = np.asarray(box_chunk, dtype=np.float64)

    reuse_worker_ctx = _MBAR_WORKER_CTX is not None
    if reuse_worker_ctx:
        eval_sys, integ, ctx = _MBAR_WORKER_CTX
    else:
        eval_sys, integ, ctx = _build_traditional_mbar_eval_context(
            system_xml=task["system_xml"],
            platform_name=str(task["platform_name"]),
            cpu_threads=task.get("cpu_threads"),
        )

    lambdas_coul = np.asarray(task["lambdas_coul"], dtype=float)
    lambdas_vdw = np.asarray(task["lambdas_vdw"], dtype=float)
    n_states = len(lambdas_coul)
    u_chunk = np.zeros((n_states, xyz_chunk.shape[0]), dtype=np.float64)
    kt = float(task["kt"])
    use_total_energy = bool(task.get("use_total_energy", False))
    apply_pme_self_correction = bool(task.get("apply_pme_self_correction", False))
    # 🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=2] 不再是一个标量 prefactor + 统一的
    # lambda_vdw**power 缩放——每个 λ 态的 switching+softcore-aware 积分结果不同，
    # 由生产者（compute_u_kn）逐态数值积分好、按 lambdas_vdw 的顺序传过来。
    lj_tail_lrc_coeff = task.get("lj_tail_lrc_coeff_kj_mol")
    if lj_tail_lrc_coeff is not None:
        lj_tail_lrc_coeff = np.asarray(lj_tail_lrc_coeff, dtype=np.float64)
    if lj_tail_lrc_coeff is not None and box_chunk is None:
        raise RuntimeError(
            "传统 Beutler LRC 已启用，但轨迹没有周期盒向量，无法计算 1/V 尾项。"
        )
    pme_self_prefactor_kj = None
    if use_total_energy and apply_pme_self_correction:
        lig_qsq = float(task.get("ligand_charge_square_sum", 0.0))
        if lig_qsq > 0.0:
            nb_force = next((f for f in eval_sys.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_force is None:
                raise RuntimeError(
                    "PME 自能修正被标记为启用，但 worker 进程里的 eval_sys 找不到 "
                    "NonbondedForce，无法拿到 Ewald alpha；拒绝静默跳过修正（那会产出一个"
                    "看起来正常、实际没做自能修正的 u_kn）。"
                )
            # 静态 PME 参数查询在 OpenMM 自动派生参数时通常为零，不能作为
            # Context 查询失败后的 fallback。优先读取 Context 的真实参数；若平台
            # 不支持该查询，则用同一 cutoff/tolerance 公式闭式派生 alpha。
            alpha_source = "context"
            try:
                alpha_ewald, _, _, _ = nb_force.getPMEParametersInContext(ctx)
                alpha_ewald = _value_in_inverse_nanometer(alpha_ewald)
            except Exception as exc:
                # A static PME-parameter query returns zero when OpenMM is
                # deriving parameters automatically, so it is not a valid fallback.
                # Derive the same alpha from cutoff/tolerance instead; this is
                # also independent of the worker platform implementation.
                alpha_source = (
                    "closed_form_after_context_query_failed"
                    f"({type(exc).__name__}: {exc})"
                )
                alpha_ewald = get_pme_alpha_for_system(eval_sys)
            if alpha_ewald <= 0.0:
                raise RuntimeError(
                    f"PME 自能修正被标记为启用（Σq_offset²={lig_qsq:.6f} e²），但无法拿到"
                    f"有效的 Ewald alpha（来源={alpha_source}，取值={alpha_ewald}）。拒绝"
                    "静默跳过修正——这会产出一个看起来正常、实际带着未修正自能伪项的 "
                    "u_kn。请检查该 NonbondedForce 的 nonbonded method/cutoff 设置，或"
                    "该 worker 平台是否支持 getPMEParametersInContext。"
                )
            pme_self_prefactor_kj = pme_self_correction_prefactor_kj(alpha_ewald, lig_qsq)
            if frame_offset == 0:
                print(
                    f"  [PME 自能修正 worker] alpha_ewald={alpha_ewald:.6f} /nm "
                    f"(来源={alpha_source})，Σq²={lig_qsq:.6f} e² → "
                    f"prefactor={pme_self_prefactor_kj:.4f} kJ/mol"
                )

    for local_f in range(xyz_chunk.shape[0]):
        ctx.setPositions(xyz_chunk[local_f] * unit.nanometer)
        if box_chunk is not None and len(box_chunk) > 0:
            ctx.setPeriodicBoxVectors(*box_chunk[local_f] * unit.nanometer)

        for k in range(n_states):
            REMDManager._try_set_context_parameter(ctx, "lambda_coul", lambdas_coul[k])
            REMDManager._try_set_context_parameter(ctx, "lambda_vdw", lambdas_vdw[k])
            if use_total_energy:
                e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                if apply_pme_self_correction and pme_self_prefactor_kj is not None:
                    # PME 总能量包含与坐标无关的自能项 -C*lambda^2。
                    # 它不会影响构型采样，但会严重污染 ΔG；这里在离线 u_kn 中解析去除。
                    e += pme_self_correction_energy_kj(lambdas_coul[k], pme_self_prefactor_kj)
            else:
                e = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if lj_tail_lrc_coeff is not None:
                volume_nm3 = _periodic_box_volume_nm3(box_chunk[local_f])
                e += float(lj_tail_lrc_coeff[k]) / volume_nm3
            u_chunk[k, local_f] = e / kt

    if not reuse_worker_ctx:
        # 复用 worker 缓存的 Context 时不能删——同一个 worker 处理的下一个
        # chunk 还要用它。ctx 上的 lambda 参数/positions/box 在下一个 chunk
        # 的循环开头就会被完整覆盖（见上面 for local_f/k 循环），不存在跨
        # chunk 的状态残留风险。
        del ctx, integ, eval_sys
    return frame_offset, u_chunk
def _split_platform_spec(platform_name: str) -> Tuple[str, Optional[str]]:
    spec = str(platform_name or "CPU").strip()
    if ":" not in spec:
        return spec, None
    base, device = spec.split(":", 1)
    base = base.strip() or "CPU"
    device = device.strip() or None
    if device is None:
        raise ValueError(
            f"平台设备索引不能为空：{platform_name!r}；请使用 CUDA、CUDA:N、"
            "OpenCL 或 OpenCL:N"
        )
    return base, device


# [P0-REMD-CUDA] 判 OOM 的下限：一个 45354 原子 PME Context 实测约 315–338 MiB
# （`memtest/probe_remd_context_capacity.py`）。留一倍余量当"还装得下一个"的门槛。


_REMD_CONTEXT_VRAM_FLOOR_MIB = 700.0


def _available_host_mib() -> Optional[float]:
    """/proc/meminfo 的 MemAvailable，MiB；拿不到就返回 None。

    用 MemAvailable 而不是 MemFree：page cache 是可回收的，MemFree 会把一台
    正常工作的机器误判成"没内存了"。
    """
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def _host_memory_mib() -> Optional[Tuple[int, int]]:
    """(当前 RSS, 进程峰值 RSS) MiB；拿不到就返回 None。

    ## 为什么加这个

    全仓在阶段边界打了 8 处**显存**（`_gpu_memory_mib`），**宿主内存一处都没有**。
    于是宿主侧内存耗尽只能表现成一句无法归因的
    `OpenMMException: std::bad_alloc` —— 2026-09-09 cyclod_ligand1/rep1 就是这样：
    崩在 Stage 1 第一个 replica 的 `XmlSerializer.deserializeSystem` 上，而那份
    XML 只有 7.4 MB、机器 93 GB、`ulimit -v unlimited`、当时只有这一个进程。
    崩点本身离线复现完全正常（8 个 replica 峰值 708 MiB），所以内存是在**进
    Stage 1 之前**被吃掉的，但日志里没有任何一个数能说是哪一段吃的。

    当前 RSS 和峰值都要：只有峰值是单调的，看不出某一段跑完有没有还回去。
    """
    try:
        import resource

        peak_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux/FreeBSD 的 ru_maxrss 是 KiB，macOS 是 bytes。
        peak = peak_raw / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
        with open("/proc/self/statm", "r", encoding="ascii") as fh:
            current = float(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024.0 / 1024.0
    except (ImportError, OSError, ValueError, IndexError, AttributeError):
        return None
    return int(current), int(peak)


def _gpu_memory_mib() -> Optional[Tuple[int, int, int]]:
    """(used, free, total) MiB；拿不到就返回 None。

    刻意用 `nvidia-smi` 子进程而不是 torch/pynvml：这条路径**不能**为了打个日志就
    在本进程里初始化 CUDA（ATT-04 的教训——`abfe_core` 模块级那行 torch 调用曾让
    每个 spawn 子进程在 import 期各抓一个 CUDA context）。诊断绝不能改变被诊断的状态。
    """
    try:
        import subprocess

        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip().splitlines()
        if not out:
            return None
        used, free, total = (int(v.strip()) for v in out[0].split(","))
        return used, free, total
    except Exception:
        # 拿不到显存读数不是错误——诊断失败不该让生产跑挂。
        return None


def _build_platform_properties(
    platform_name: str,
    cpu_threads: Optional[int] = None,
) -> Tuple[str, Dict[str, str]]:
    """`cpu_threads`：仅在 CPU 平台生效，对应 OpenMM CPU 平台的 "Threads"
    属性。默认 None 时保持原来的行为（不设置，OpenMM 自行决定，通常是用满
    所有核）——只有明确知道"这个 Context 只应该用几个线程"的调用方（比如
    离线 MBAR 多进程 worker pool，见 _mbar_worker_init）才会传非 None 值，
    其余调用点（生产/warmup 用的主 CUDA Context 等）不受影响。
    """
    base, device = _split_platform_spec(platform_name)
    upper = base.upper()
    props: Dict[str, str] = {}
    if device is not None:
        if upper not in {"CUDA", "OPENCL"}:
            raise ValueError(
                f"平台 {base!r} 不支持设备索引写法 {platform_name!r}；"
                "只有 CUDA:N/OpenCL:N 合法"
            )
        if not device.isascii() or not device.isdigit():
            raise ValueError(f"平台设备索引必须是非负整数：{platform_name!r}")
    if upper == "CUDA":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
    elif upper == "OPENCL":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
    elif upper == "CPU" and cpu_threads is not None:
        props["Threads"] = str(int(cpu_threads))
    return base, props


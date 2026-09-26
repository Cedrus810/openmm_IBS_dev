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

# ---------------------------------------------------------------------------
# R2：对真实 hybrid Hamiltonian 做 MBAR
# ---------------------------------------------------------------------------
#
# 计划 §6 明确禁止复用 `TraditionalMBARAnalyzer.compute_u_kn`——它按**单配体去耦**
# 和 LRC 假设重建评估系统（PME self correction、ligand_charge_square_sum、
# co-ion、Boresch……），那套假设对 hybrid Hamiltonian 全都不成立。
#
# 但同一句话的后半段也要照做：计划 §6 说**独立的 MBAR 数值求解部分**可以复用。
# 所以这里的分工是：
#
#   u_kn        —— RBFE 自己算，直接在 hybrid System 上换 λ 求能量（本节）
#   MBAR 求解   —— 复用 `ibs_engine.TraditionalMBARAnalyzer.solve()`
#                  （去相关子采样、overlap 诊断、多套 solver protocol 兜底，
#                    都是那边踩出来的，重写必错）
#
# u_kn 这边比 ABFE 简单得多，而且**结构上不可能算错评估体系**：hybrid System 只有
# 一个，换状态就是 `context.setParameter(λ)`，不重建任何东西。ABFE 那边之所以复杂，
# 正是因为它要为每个 λ 重新造一个"等效的去耦体系"。

#: 1 bar·nm³ 换算成 kJ/mol。
#: 1e5 Pa × 1e-27 m³ = 1e-22 J；× N_A(6.02214076e23 /mol) = 60.2214076 J/mol。
_KJ_PER_MOL_PER_BAR_NM3 = 0.0602214076

#: 气体常数 R，kJ/(mol·K)。与 `ibs_engine.TraditionalMBARAnalyzer` 用的是同一个值——
#: 两边必须一致，否则 u_kn 的 β 和求解器的 kT 对不上，ΔG 会差一个常数因子。
_R_KJ_PER_MOL_K = 0.008314462618


def compute_hybrid_u_kn(
    bundle: HybridSystemBundle,
    samples,
    *,
    temperature_kelvin: float,
    pressure_bar: Optional[float] = None,
    platform_name: str = "Reference",
) -> dict:
    """在**真实 hybrid Hamiltonian** 上求约化能量矩阵 u_kn。

    `samples` 是 `free_energy_engine.run_independent_windows` 的产物（鸭子类型：
    需要 `positions_by_state` / `box_vectors_by_state` / `state_parameters`）。

    ## 样本归属靠下标，不靠文件名

    计划 §7：「能量矩阵记录维度、状态顺序、样本状态索引、是否约化和是否包含 NPT
    所需项；**不按文件名猜**。」这里 `samples.positions_by_state[k]` 就是状态 k 的
    帧，`n_k[k]` 是它的帧数，返回值把这些一并带出去。

    进函数第一件事是核对 `samples.state_parameters` 与 `bundle.schedule` **逐位相同**
    ——样本是在哪个 λ 上采的、u_kn 又要在哪些 λ 上评估，这两件事必须由同一张表决定。
    对不上就报错，不做任何"看起来能对上"的迁就。

    ## NPT

    `pressure_bar` 给了就在约化势里加 βpV（V 取**每一帧自己的**盒体积）。
    不给就是 NVT。返回值里 `includes_pV` 如实记录，不让下游去猜。
    """
    import openmm
    from openmm import unit

    schedule_states = [bundle.schedule.state(i) for i in range(bundle.schedule.n_states)]
    sample_states = [dict(p) for p in samples.state_parameters]
    if len(sample_states) != len(schedule_states):
        raise RBFEHybridBuildError(
            f"样本有 {len(sample_states)} 个状态，λ 表有 {len(schedule_states)} 个。"
            "样本必须是在这张 λ 表上采的。"
        )
    for index, (sampled, expected) in enumerate(zip(sample_states, schedule_states)):
        if set(sampled) != set(expected) or any(
            abs(float(sampled[name]) - float(expected[name])) > 1e-12 for name in expected
        ):
            raise RBFEHybridBuildError(
                f"第 {index} 个状态的 λ 与 λ 表不一致：样本 {sampled} vs λ 表 {expected}。"
                "样本采样时用的 λ 与评估用的 λ 必须来自同一张表。"
            )

    sample_pressure = getattr(samples, "pressure_bar", None)
    if (sample_pressure is None) != (pressure_bar is None):
        raise RBFEHybridBuildError(
            f"系综不一致：样本是 pressure_bar={sample_pressure} 采的，"
            f"却要求按 pressure_bar={pressure_bar} 约化。"
            "NPT 样本按 NVT 约化会漏掉 βpV，反之会凭空多一项。"
        )

    beta = 1.0 / (_R_KJ_PER_MOL_K * float(temperature_kelvin))
    n_states = len(schedule_states)
    n_k = np.asarray([len(frames) for frames in samples.positions_by_state], dtype=int)
    if np.any(n_k <= 0):
        raise RBFEHybridBuildError(f"有状态一帧都没有：n_k={n_k.tolist()}")
    total = int(n_k.sum())

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(bundle.system, integrator, platform)

    u_kn = np.zeros((n_states, total), dtype=float)
    sample_state_index = np.zeros(total, dtype=int)
    column = 0
    try:
        for k in range(n_states):
            frames = samples.positions_by_state[k]
            boxes = samples.box_vectors_by_state[k]
            for frame_index in range(len(frames)):
                box = np.asarray(boxes[frame_index], dtype=float)
                context.setPeriodicBoxVectors(*[openmm.Vec3(*row) * unit.nanometer for row in box])
                context.setPositions(np.asarray(frames[frame_index], dtype=float) * unit.nanometer)
                pv_term = 0.0
                if pressure_bar is not None:
                    volume = abs(float(np.linalg.det(box)))
                    pv_term = float(pressure_bar) * volume * _KJ_PER_MOL_PER_BAR_NM3
                for evaluated in range(n_states):
                    for name, value in schedule_states[evaluated].items():
                        context.setParameter(name, float(value))
                    energy = context.getState(getEnergy=True).getPotentialEnergy()
                    energy = energy.value_in_unit(unit.kilojoule_per_mole)
                    if not math.isfinite(energy):
                        raise RBFEHybridBuildError(
                            f"状态 {k} 第 {frame_index} 帧在 λ 态 {evaluated} 上能量非有限：{energy}"
                        )
                    u_kn[evaluated, column] = beta * (energy + pv_term)
                sample_state_index[column] = k
                column += 1
    finally:
        del context, integrator

    return {
        "u_kn": u_kn,
        "n_k": n_k,
        "sample_state_index": sample_state_index,
        "n_states": int(n_states),
        "n_samples": total,
        "reduced": True,
        "includes_pV": pressure_bar is not None,
        "temperature_kelvin": float(temperature_kelvin),
        "pressure_bar": pressure_bar,
        "beta_mol_per_kJ": float(beta),
        "state_order": [dict(s) for s in schedule_states],
        "hybrid_fingerprint": bundle.fingerprint(),
        "evaluated_on": "hybrid_system_via_setParameter__no_system_rebuild",
    }


def analyze_leg(
    bundle: HybridSystemBundle,
    samples,
    *,
    phase: str,
    edge_id: str,
    ligand_a_name: str,
    ligand_b_name: str,
    temperature_kelvin: float,
    pressure_bar: Optional[float] = None,
    energy_unit: str = KJ_PER_MOL,
    platform_name: str = "Reference",
    decorrelate: bool = True,
) -> tuple:
    """R2：由样本算出这条腿的 ΔG(A→B)。返回 `(LegResult, 诊断字典)`。

    ## 方向约定的落点（计划 §3）

    `TraditionalMBARAnalyzer.solve()` 返回的 `delta_G` 是
    `(f[-1] − f[0]) · kT`，即**最后一个状态减第一个状态**。本 builder 的 λ 表被
    `HybridLambdaSchedule` 强制成端点 0→1，而 λ=0 对应 A、λ=1 对应 B，所以

        delta_G == G(B) − G(A) == ΔG(A→B)

    正是 `LegResult.delta_g` 要求的含义。**这条链不能靠记忆维持**——
    `test_leg_result_direction_matches_the_plan` 用一个符号已知的构造把它钉死。

    ## 与计划里那个签名的差别

    计划 §4 草案写的是 `analyze_leg(leg: PreparedLeg, artifacts: SamplingArtifacts)`。
    `PreparedLeg` / `SamplingArtifacts` 那两层还没有（前者要 `prepare_edge`，
    后者是落盘轨迹的路径），所以这里直接吃 `HybridSystemBundle` + 内存样本。
    接口形状变了，职责没变。
    """
    if phase not in RBFE_PHASES:
        raise ValueError(f"phase 必须是 {RBFE_PHASES} 之一：收到 {phase!r}")

    matrix = compute_hybrid_u_kn(
        bundle,
        samples,
        temperature_kelvin=temperature_kelvin,
        pressure_bar=pressure_bar,
        platform_name=platform_name,
    )

    # 🔑 复用 ABFE 的 MBAR 数值求解（计划 §6 允许的那一部分）。**惰性 import**：
    # `ibs_engine` 是两万多行、且会拉进 openmm/pymbar，放模块顶部会让 R0/R1a
    # 那两层「不 import openmm」的性质失效。这里只用它的 solve()，
    # 一行 ABFE 代码都不改。
    import ibs_engine

    analyzer = ibs_engine.TraditionalMBARAnalyzer(temperature=float(temperature_kelvin))
    analyzer._last_n_k = matrix["n_k"]
    solved = analyzer.solve(matrix["u_kn"], decorrelate=decorrelate)

    delta_g_kj = float(solved["delta_G"])
    stderr_kj = float(solved.get("error", float("nan")))
    if not math.isfinite(stderr_kj):
        raise RBFEHybridBuildError(
            "MBAR 没有给出不确定度（error 为 nan）。没有误差棒的 ΔG 不能进 LegResult——"
            "下游的 ΔΔG 误差传播会静默变成 nan，而 nan 一路传到最终报告里很难被发现。"
        )

    delta_g = convert_energy(delta_g_kj, KJ_PER_MOL, energy_unit)
    stderr = convert_energy(stderr_kj, KJ_PER_MOL, energy_unit)

    # MBAR 的 ESS 算不出来时（`compute_effective_sample_number()` 抛错），原来
    # 退回**原始帧数** `n_frames`——把"没测到"报成最乐观的那个数，任何下游
    # "N_eff 够不够"的门都会被这个假数字放行。未知只能往保守方向记，并把
    # 原因写进 diagnostics，不许伪装成一次测量。
    effective = solved.get("diagnostics", {}).get("effective_sample_number")
    n_effective_measured = isinstance(effective, (list, tuple)) and bool(effective)
    if n_effective_measured:
        n_effective = int(min(float(v) for v in effective))
    else:
        n_effective = 0

    fingerprint_payload = {
        "hybrid_fingerprint": matrix["hybrid_fingerprint"],
        "n_k": matrix["n_k"].tolist(),
        "reduced": matrix["reduced"],
        "includes_pV": matrix["includes_pV"],
        "temperature_kelvin": matrix["temperature_kelvin"],
        "pressure_bar": matrix["pressure_bar"],
        "state_order": matrix["state_order"],
        "sampler": dict(getattr(samples, "provenance", {}) or {}),
        "mbar_method": solved.get("method"),
        "decorrelate": bool(decorrelate),
    }
    artifacts_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()

    result = LegResult(
        phase=phase,
        edge_id=edge_id,
        ligand_a_name=ligand_a_name,
        ligand_b_name=ligand_b_name,
        delta_g=delta_g,
        stderr=stderr,
        energy_unit=energy_unit,
        uncertainty_method=str(solved.get("method", "mbar")),
        n_effective_samples=n_effective,
        quality_gate_passed=bool(solved.get("converged", False)),
        artifacts_fingerprint=artifacts_fingerprint,
    )
    diagnostics = {
        "u_kn_shape": list(matrix["u_kn"].shape),
        "n_k": matrix["n_k"].tolist(),
        #: False 时上面的 n_effective_samples=0 是"没测到"，不是"一个有效样本都没有"。
        "n_effective_samples_measured": bool(n_effective_measured),
        "includes_pV": matrix["includes_pV"],
        "min_overlap": solved.get("min_overlap"),
        "converged": solved.get("converged"),
        "mbar": {key: value for key, value in solved.items() if key != "diagnostics"},
        "mbar_diagnostics": solved.get("diagnostics", {}),
        "sampler_provenance": dict(getattr(samples, "provenance", {}) or {}),
    }
    return result, diagnostics


# === 未实现区（R1 起）=====================================================
#
# 计划 §4.1：「不提供『尚未实现但返回成功』的占位 sampler；接口准备完成与可运行
# 科学计算是两个验收阶段。」所以下面这些**抛错**，不返回假数据。
#
# ==========================================================================


def prepare_edge(spec: EdgeSpec):  # pragma: no cover - 未实现
    """从 `EdgeSpec` 的输入路径建出两个端点体系并冻结映射。**尚未实现。**

    缺的是「读参数化输入 → 建 System → 同一个盒子里溶剂化 A 和 B」这一段，
    它需要一份**可跑的**配体 B（改基团必然要重新给部分电荷与成键参数，
    §5.1 不允许悄悄重参数化），是 R3 的硬前提。

    映射与 hybrid builder 都已经可用，可以直接用：
        `map_atoms` / `validate_mapping` / `build_hybrid_system`
    """
    raise NotImplementedError(
        "prepare_edge 尚未实现：它要从 EdgeSpec 的输入路径建出 A、B 两个端点体系，"
        "前提是有一份可跑的配体 B（含部分电荷与成键参数）。"
        "映射用 map_atoms、hybrid 体系用 build_hybrid_system，两者都已可用。"
        "见 docs/design/PLAN_rbfe_interface_and_implementation.md §0。"
    )


def build_hybrid_leg(prepared, phase: str):  # pragma: no cover - 等 prepare_edge
    """按 `PreparedEdge` 建一条腿的 hybrid 体系。**这一层还没有**。

    ⚠ 别被这个 `NotImplementedError` 误导成「hybrid builder 没做」——做了，
    是 `build_hybrid_system()`（R1b，已实现并通过端点等价 + 有限差分力验收）。
    缺的只是它上面这层薄封装，因为 `PreparedEdge` 要等 `prepare_edge`
    （从 `EdgeSpec` 的输入路径建出两个 System）才有定义。

    现在就能用的调用方式：

        bundle = rbfe_core.build_hybrid_system(
            system_a, ligand_indices_a, system_b, ligand_indices_b, mapping, schedule
        )
    """
    raise NotImplementedError(
        "build_hybrid_leg 需要 prepare_edge 产出的 PreparedEdge，那一层尚未实现。"
        "hybrid builder 本身已经可用：直接调 rbfe_core.build_hybrid_system("
        "system_a, ligand_indices_a, system_b, ligand_indices_b, mapping, schedule)。"
    )



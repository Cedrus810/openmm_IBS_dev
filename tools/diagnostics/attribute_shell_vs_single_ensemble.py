#!/usr/bin/env python
"""归因实验：stage2 的 +45 kJ/mol 误差，是**壳**还是**单系综重加权**？

2026-09-02 已有 2×2 的两个角（见 docs/reference_data/ 与
docs/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md §2.9）：

              | 逐态独立采样        | 单混合分布
    ----------+---------------------+------------------
    无壳      | -6.581 ± 0.256 ✓    | ?
    有壳      | 本脚本 ←            | +38.72 ✗（生产）

本脚本补"有壳 + 逐态独立采样"那一角。**相对原参考算例只改一件事**：
采样哈密顿量里加上生产的 λ-WCA 防护壳（Group 4，rc=0.244 nm、
eps_wca=1.0 kJ/mol，幅度 4λ_s(1−λ_s)、λ_s = 该态所属窗口的 λ_vdw 均值，
与 `ibs_engine.py:4915` + `:13675` 逐字同形），而**目标态能量仍然无壳**
（`shell_amp` 求 u_kln 时置 0）—— 这正是生产的口径
（WCA_ACCOUNTING_VERSION=2：e_bias = groups{1,4}）。

软核指数已改成与生产一致的 m=n=2（`aces_softcore_params.power_lj=[2,2]`），
所以逐窗口 ΔF 与生产可比。

判读
----
* 结果仍 ≈ **−6.58** ⟹ **壳无罪**，病在单混合分布重加权
  （= STAGE2_ROOT_CAUSE_2026-08-28 §1 的空腔重组根因）。
* 结果朝 **+38.7** 移动 ⟹ **壳是主因**，`bias_to_signal_ratio` 那条相关
  （Spearman +0.886）是因果不是巧合。

⚠️ 前车之鉴：`measure_wca_shell_cost_at_decoupled_endpoint.py` 那个尝试是
**ill-posed** 的——它把 1/r¹² 排斥核线性缩放到恰好 0，最后一步吃掉 92.4% 的
答案（−92.9/−100.5），MBAR 报的 ±0.056 严重低估；而且它测的量生产从不单独
计算（壳对窗口内所有 k 相同，在窗口内 ΔF 里大部分抵消）。本脚本不缩放壳，
壳只在"采样有 / 目标无"这一个二值差异上出现，没有端点奇点。

用法
----
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    python tools/diagnostics/attribute_shell_vs_single_ensemble.py \
        --root /home/ruigengji/ABFE_IBS/4W53 --vdw-only \
        --platform CUDA --precision mixed \
        --lambda-vdw-json <生产真实 23 态 λ 表> \
        --window-ranges "[[0,5],[4,8],[7,12],[11,16],[15,20],[19,23]]" \
        --out <outdir>

`--no-shell` 是对照组，应复现 −6.58（用来确认本脚本没有引入别的改动）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

ONE_4PI_EPS0 = 138.935456  # kJ/mol * nm / e^2
KB = unit.MOLAR_GAS_CONSTANT_R


def build_alchemical_system(system, ligand_indices, softcore_alpha=0.5):
    """把 system 改成：配体↔环境 的 LJ 走 softcore CustomNonbondedForce（受
    lambda_vdw 控制），配体↔环境 的静电走 NonbondedForce 的 charge offset
    （受 lambda_elec 控制，PME 由 OpenMM 自己处理）。

    配体分子内非键先从 NonbondedForce 里整体剔除（全部加成 exception），
    再用一个 λ 无关的 CustomBondForce 原样加回来 —— 所以分子内那部分在任何
    λ 下都是满强度，绝不参与解耦。
    """
    lig = set(int(i) for i in ligand_indices)
    nb = None
    for f in system.getForces():
        if isinstance(f, mm.NonbondedForce):
            assert nb is None, "系统里有多个 NonbondedForce，需要人工确认"
            nb = f
    assert nb is not None, "没找到 NonbondedForce"

    n = system.getNumParticles()
    base = []
    for i in range(n):
        q, sig, eps = nb.getParticleParameters(i)
        base.append((q.value_in_unit(unit.elementary_charge),
                     sig.value_in_unit(unit.nanometer),
                     eps.value_in_unit(unit.kilojoule_per_mole)))

    # ---- 1. 收集配体分子内已有 exception，并把它们清零 -------------------
    intra = {}          # (i,j) -> (chargeProd, sigma, epsilon)  满强度值
    seen_pairs = set()
    for e in range(nb.getNumExceptions()):
        i, j, qq, sig, eps = nb.getExceptionParameters(e)
        if i in lig and j in lig:
            key = (min(i, j), max(i, j))
            seen_pairs.add(key)
            intra[key] = (qq.value_in_unit(unit.elementary_charge**2),
                          sig.value_in_unit(unit.nanometer),
                          eps.value_in_unit(unit.kilojoule_per_mole))
            nb.setExceptionParameters(e, i, j, 0.0, sig, 0.0)

    # ---- 2. 其余配体内部对：按组合规则算出满强度值，再加成清零的 exception --
    lig_sorted = sorted(lig)
    for a in range(len(lig_sorted)):
        for b in range(a + 1, len(lig_sorted)):
            i, j = lig_sorted[a], lig_sorted[b]
            key = (i, j)
            if key in seen_pairs:
                continue
            qi, si, ei = base[i]
            qj, sj, ej = base[j]
            sig = 0.5 * (si + sj)
            eps = float(np.sqrt(ei * ej))
            intra[key] = (qi * qj, sig, eps)
            nb.addException(i, j, 0.0, sig, 0.0)

    # ---- 3. 把配体分子内非键以 λ 无关的形式加回来 -------------------------
    intra_force = mm.CustomBondForce(
        f"{ONE_4PI_EPS0}*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)"
    )
    intra_force.addPerBondParameter("chargeProd")
    intra_force.addPerBondParameter("sigma")
    intra_force.addPerBondParameter("epsilon")
    for (i, j), (qq, sig, eps) in sorted(intra.items()):
        if qq == 0.0 and eps == 0.0:
            continue
        intra_force.addBond(i, j, [qq, sig, eps])
    intra_force.setName("LigandIntramolecularNonbonded_fixed")
    system.addForce(intra_force)

    # ---- 4. 配体静电：base charge 归零，用 lambda_elec offset 拉回 --------
    nb.addGlobalParameter("lambda_elec", 1.0)
    for i in lig_sorted:
        q, sig, eps = base[i]
        nb.setParticleParameters(i, 0.0, sig * unit.nanometer,
                                 0.0 * unit.kilojoule_per_mole)  # eps 也清零，LJ 交给下面
        if q != 0.0:
            nb.addParticleParameterOffset("lambda_elec", i, q, 0.0, 0.0)

    # ---- 5. 配体↔环境 LJ：softcore CustomNonbondedForce -------------------
    expr = (
        "lambda_vdw^2*4*epsilon*(1/den^2 - 1/den);"
        f"den = {softcore_alpha}*(1-lambda_vdw)^2 + (r/sigma)^6;"
        "sigma = 0.5*(sigma1+sigma2);"
        "epsilon = sqrt(epsilon1*epsilon2)"
    )
    cnb = mm.CustomNonbondedForce(expr)
    cnb.addGlobalParameter("lambda_vdw", 1.0)
    cnb.addPerParticleParameter("sigma")
    cnb.addPerParticleParameter("epsilon")
    for i in range(n):
        _, sig, eps = base[i]
        cnb.addParticle([sig, eps])
    cnb.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    cnb.setCutoffDistance(nb.getCutoffDistance())
    cnb.setUseSwitchingFunction(nb.getUseSwitchingFunction())
    if nb.getUseSwitchingFunction():
        cnb.setSwitchingDistance(nb.getSwitchingDistance())
    # interaction group 下 OpenMM 的解析长程修正不适用，显式关掉；
    # 15 原子配体的 LJ tail 量级 ~0.2 kcal/mol，对本次判据（差 11 kcal/mol）无关。
    cnb.setUseLongRangeCorrection(False)
    # OpenMM 要求同一体系里所有非键力的 exclusion 集合一致，所以把
    # NonbondedForce 的每一条 exception 都复制成 CustomNonbondedForce 的
    # exclusion。配体↔环境之间本来就没有 exception，所以这一步不会挖掉任何
    # 真正要解耦的相互作用。
    for e in range(nb.getNumExceptions()):
        i, j, _, _, _ = nb.getExceptionParameters(e)
        cnb.addExclusion(i, j)
    env = [i for i in range(n) if i not in lig]
    cnb.addInteractionGroup(lig_sorted, env)
    cnb.setName("LigandEnvironmentSoftcoreLJ")
    system.addForce(cnb)

    # ===== [2026-09-02 归因用] 生产的 λ-WCA 防护壳（Group 4 → 此处 group 7）=====
    # 与 ibs_engine.py:4915 逐字同形；rc/eps 取自那次运行 pipeline.log
    # （rc=0.244 nm, eps_wca=1.0 kJ/mol, lj_sigma_10th_percentile）。
    # 幅度用可变 global `shell_amp`，由外层按"该态所属窗口的 4λ_s(1-λ_s)"设置——
    # 生产就是这么设的（lambda_shield = 本窗口 λ_vdw 均值）。
    # 关键：**目标态能量不含壳**（与生产一致），所以 u_kln 求值时必须把 shell_amp
    # 置 0；壳只进采样哈密顿量。
    wca_expr = ("shell_amp*step(rc-r)*eps_wca*"
                "(((rc/max(r, 1e-6))^6)^2 - 2*((rc/max(r, 1e-6))^6) + 1)")
    wca = mm.CustomNonbondedForce(wca_expr)
    wca.addGlobalParameter("shell_amp", 0.0)
    wca.addGlobalParameter("rc", 0.244)
    wca.addGlobalParameter("eps_wca", 1.0)
    for _ in range(n): wca.addParticle([])
    wca.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    wca.setCutoffDistance(0.244 * unit.nanometer)
    wca.addInteractionGroup(lig_sorted, env)
    for e in range(nb.getNumExceptions()):
        i, j, _, _, _ = nb.getExceptionParameters(e)
        wca.addExclusion(i, j)
    wca.setForceGroup(7)
    wca.setName("ProductionLambdaWCAShield")
    system.addForce(wca)
    return system


def lambda_schedule():
    """先关静电（LJ 全开），再关 LJ。共 15 个态，端点共用一个。"""
    states = [(le, 1.0) for le in (1.0, 0.75, 0.5, 0.25, 0.0)]
    for lv in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        states.append((0.0, lv))
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None, help="输出目录，默认 <root>/output/hydration_reference")
    ap.add_argument("--timestep-fs", type=float, default=1.0,
                    help="积分步长 fs。默认 1.0：体系只约束了水（4193 条），配体 X-H "
                         "没有约束，2 fs 不稳。")
    ap.add_argument("--pre-equil-steps", type=int, default=200000,
                    help="进入 λ 循环之前，在全耦合态(λ=1,1)下的整体预平衡步数。"
                         "topology_solvent.cif 是建系坐标，不是平衡后的。")
    ap.add_argument("--equil-steps", type=int, default=100000, help="每个 λ 态预平衡步数")
    ap.add_argument("--prod-steps", type=int, default=500000, help="每个 λ 态生产步数")
    ap.add_argument("--sample-interval", type=int, default=1000, help="每多少步取一帧")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--lambda-vdw-json", default=None,
                    help="用生产管线自己的 λ_vdw 路径替换本脚本默认的均匀 LJ 梯子"
                         "（传 preopt_dual_vanishing.json 的路径）。静电段不变。"
                         "用途：把'λ 布点对不对'和'估计器/WCA 对不对'分开。")
    ap.add_argument("--vdw-only", action="store_true",
                    help="只跑 LJ 段（起点 = 已去电荷、LJ 全耦合）。静电段两套代码已经"
                         "对到 0.02 kJ/mol，没必要重跑。省掉 5 个 λ 态的 GPU 时间。")
    ap.add_argument("--window-ranges", default=None,
                    help='生产的 window_ranges JSON，例如 "[[0,5],[4,8],...]"。'
                         "给了就按各窗口 λ_vdw 均值开壳；不给则全程无壳。")
    ap.add_argument("--no-shell", action="store_true",
                    help="强制全程无壳（对照组，应复现原参考算例）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：极短步数 + CPU，只验证能跑通，结果无物理意义")
    args = ap.parse_args()

    if args.smoke:
        args.pre_equil_steps = 200
        args.equil_steps, args.prod_steps, args.sample_interval = 200, 400, 100
        args.platform, args.precision = "CPU", None

    root = args.root
    outdir = args.out or os.path.join(root, "output", "hydration_reference")
    os.makedirs(outdir, exist_ok=True)

    sys_xml = os.path.join(root, "output", "system_solvent.xml")
    cif = os.path.join(root, "output", "topology_solvent.cif")
    boxf = os.path.join(root, "output", "box_vectors_solvent.npy")
    ligf = os.path.join(root, "output", "ligand_indices_solvent.json")
    for f in (sys_xml, cif, boxf, ligf):
        if not os.path.exists(f):
            sys.exit(f"缺少输入文件: {f}")

    system = mm.XmlSerializer.deserialize(open(sys_xml).read())
    pdbx = app.PDBxFile(cif)
    box = np.load(boxf)
    ligand_indices = json.load(open(ligf))["ligand_indices"]

    print(f"体系原子数 = {system.getNumParticles()}, 约束 = {system.getNumConstraints()}")
    print(f"配体原子 = {len(ligand_indices)} (indices {ligand_indices[0]}..{ligand_indices[-1]})")
    for f in list(system.getForces()):
        if isinstance(f, mm.MonteCarloBarostat):
            sys.exit("系统里有 barostat；本脚本按 NVT 设计，请先确认")

    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) * unit.nanometer for row in box])
    build_alchemical_system(system, ligand_indices)

    T = args.temperature * unit.kelvin
    kT = (KB * T).value_in_unit(unit.kilojoule_per_mole)
    integrator = mm.LangevinMiddleIntegrator(
        T, 1.0 / unit.picosecond, args.timestep_fs * unit.femtosecond
    )
    integrator.setRandomNumberSeed(args.seed)

    plat = mm.Platform.getPlatformByName(args.platform)
    props = {}
    if args.platform == "CUDA":
        props = {"Precision": args.precision, "DeterministicForces": "false"}
    context = mm.Context(system, integrator, plat, props) if props else mm.Context(system, integrator, plat)
    context.setPositions(pdbx.positions)
    context.setPeriodicBoxVectors(*[mm.Vec3(*row) * unit.nanometer for row in box])

    ladder = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    if args.lambda_vdw_json:
        ladder = [float(x) for x in json.load(open(args.lambda_vdw_json))["lambdas_var"]]
        assert abs(ladder[0] - 1.0) < 1e-9 and abs(ladder[-1]) < 1e-9, \
            "生产 λ_vdw 端点必须是 1 和 0"
        print(f"[λ 路径] 用生产管线的 λ_vdw ({len(ladder)} 态) 替换默认梯子: "
              f"{args.lambda_vdw_json}")
    if args.vdw_only:
        # 起点就是"电荷已关、LJ 全耦合"，终点是完全去耦。得到的正是可以直接跟
        # 生产 stage2_vanishing 的 total_delta_G 对比的那一段。
        states = [(0.0, lv) for lv in ladder]
    else:
        states = [(le, 1.0) for le in (1.0, 0.75, 0.5, 0.25, 0.0)]
        states += [(0.0, lv) for lv in ladder[1:]]
    K = len(states)
    n_samples = args.prod_steps // args.sample_interval
    print(f"λ 态数 = {K}, 每态样本数 = {n_samples}")
    print("λ 表 (lambda_elec, lambda_vdw):")
    for k, (le, lv) in enumerate(states):
        print(f"  {k:2d}: elec={le:.2f} vdw={lv:.2f}")

    # ===== [2026-09-02 归因] 每个态所属"生产窗口"的壳幅度 =====
    # 生产把 lambda_shield 设成**本窗口 λ_vdw 均值**（ibs_engine.py:13675），
    # 壳幅度 = 4·λ_s·(1−λ_s)。这里按 --window-ranges 把态映射回窗口。
    _win_ranges = json.loads(args.window_ranges) if args.window_ranges else None
    shell_amp_for_state = [0.0] * K
    if _win_ranges is not None and not args.no_shell:
        for a, b in _win_ranges:
            lv_win = [states[i][1] for i in range(a, min(b, K))]
            if not lv_win:
                continue
            lam_s = sum(lv_win) / len(lv_win)
            amp = 4.0 * lam_s * (1.0 - lam_s)
            for i in range(a, min(b, K)):
                shell_amp_for_state[i] = amp
        print("[壳] 逐态采样壳幅度 4λ_s(1-λ_s):")
        for i in range(K):
            print(f"    态 {i:2d}  λ_vdw={states[i][1]:.4f}  shell_amp={shell_amp_for_state[i]:.4f}")
    else:
        print("[壳] --no-shell 或未给 window_ranges ⟹ 全程无壳（等价于原参考算例）")

    def set_state(k):
        le, lv = states[k]
        context.setParameter("lambda_elec", le)
        context.setParameter("lambda_vdw", lv)

    def set_sampling_shell(k):
        """采样时：壳按该态所属窗口开启。"""
        context.setParameter("shell_amp", float(shell_amp_for_state[k]))

    def clear_shell_for_target_energy():
        """求目标态能量时：壳必须为 0 —— 生产的 target_energies 不含壳
        （WCA_ACCOUNTING_VERSION=2：e_bias=groups{1,4}）。这一条是本脚本的
        全部要点，写错了就退化成'壳进了目标态'，测的就不是生产那个口径了。"""
        context.setParameter("shell_amp", 0.0)

    u_kln = np.zeros((K, K, n_samples), dtype=np.float64)
    N_k = np.zeros(K, dtype=np.int32)

    t0 = time.time()
    # 全耦合态下先整体平衡一次；之后按 λ 顺序推进，每个态都从上一个态的构型继续
    # （慢生长式启动），不再反复最小化——反复从热化构型最小化会引入偏置。
    set_state(0)
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=2000)
    context.setVelocitiesToTemperature(T, args.seed)
    integrator.step(args.pre_equil_steps)
    print(f"全耦合预平衡完成 ({args.pre_equil_steps} 步) | {time.time()-t0:.0f}s", flush=True)

    for k in range(K):
        set_state(k); set_sampling_shell(k)
        integrator.step(args.equil_steps)
        for s in range(n_samples):
            set_state(k); set_sampling_shell(k)
            integrator.step(args.sample_interval)
            clear_shell_for_target_energy()          # 目标态无壳
            for l in range(K):
                set_state(l)
                e = context.getState(getEnergy=True).getPotentialEnergy()
                u_kln[k, l, s] = e.value_in_unit(unit.kilojoule_per_mole) / kT
        N_k[k] = n_samples
        el, lv = states[k]
        print(f"  [态 {k:2d}] elec={el:.2f} vdw={lv:.2f} 完成 | 累计 {time.time()-t0:.0f}s", flush=True)

    from pymbar import MBAR
    u_kn = np.zeros((K, int(N_k.sum())))
    idx = 0
    for k in range(K):
        u_kn[:, idx:idx + N_k[k]] = u_kln[k, :, :N_k[k]]
        idx += N_k[k]
    mbar = MBAR(u_kn, N_k)
    res = mbar.compute_free_energy_differences()
    df, ddf = res["Delta_f"], res["dDelta_f"]

    dG_decouple_kT = float(df[0, K - 1])
    err_kT = float(ddf[0, K - 1])
    dG_decouple = dG_decouple_kT * kT
    err = err_kT * kT

    print("\n" + "=" * 70)
    if args.vdw_only:
        print("独立参考算例：LJ 段（去 VDW）—— 甲苯 / 纯水，只解耦配体↔环境")
        print("=" * 70)
        print(f"  ΔG_LJ (态0 -> 态{K-1})  = {dG_decouple:+8.3f} ± {err:.3f} kJ/mol")
        print(f"  本脚本默认梯子的同一段   =   -6.26 kJ/mol")
        print(f"  生产 stage2_vanishing    =  +35.61 kJ/mol")
        print("=" * 70)
        print("判读：")
        print("  ≈ -6  kJ/mol  -> λ 布点没问题，bug 在 IBS 估计器 / WCA 防护壳")
        print("  ≈ +35 kJ/mol  -> λ 布点本身就把这段算错了，查 softcore/端点")
    else:
        print("独立参考算例结果（甲苯 / 纯水，只解耦配体↔环境）")
        print("=" * 70)
        print(f"  ΔG_decouple (态0 -> 态{K-1}) = {dG_decouple:+8.3f} ± {err:.3f} kJ/mol")
        print(f"                              = {dG_decouple/4.184:+8.3f} ± {err/4.184:.3f} kcal/mol")
        print(f"  ΔG_hyd = -ΔG_decouple       = {-dG_decouple/4.184:+8.3f} kcal/mol")
        print(f"  实验参考 ΔG_hyd(toluene)    =    -0.89 kcal/mol")
        print(f"  生产管线溶剂腿              =   +10.24 kcal/mol (解耦口径)")
        print("=" * 70)
        print("判读：")
        print("  解耦 ≈ +0.9 kcal/mol  -> 建系/力场没问题，bug 在生产的自由能机器里")
        print("  解耦 ≈ +10  kcal/mol  -> 这个 Hamiltonian 本身就给这个数，跟 IBS 无关")

    payload = {
        "delta_G_decouple_kJ_mol": dG_decouple,
        "delta_G_decouple_kcal_mol": dG_decouple / 4.184,
        "error_kJ_mol": err,
        "delta_G_hydration_kcal_mol": -dG_decouple / 4.184,
        "experimental_hydration_kcal_mol": -0.89,
        "production_solvent_leg_decouple_kcal_mol": 10.24,
        "n_states": K,
        "lambda_schedule": states,
        "n_samples_per_state": int(n_samples),
        "pre_equil_steps": args.pre_equil_steps,
        "equil_steps": args.equil_steps,
        "prod_steps": args.prod_steps,
        "timestep_fs": args.timestep_fs,
        "temperature_K": args.temperature,
        "softcore_alpha": 0.5,
        "ligand_intramolecular_nonbonded": "kept at full strength, never decoupled",
        "long_range_dispersion_correction": "disabled for ligand-environment LJ (~0.2 kcal/mol scale)",
        "smoke": bool(args.smoke),
        "vdw_only": bool(args.vdw_only),
        "lambda_vdw_json": args.lambda_vdw_json,
        "reference_default_ladder_LJ_segment_kJ_mol": -6.26,
        "production_stage2_vanishing_kJ_mol": 35.61,
        "per_state_Delta_f_kT": df[0, :].tolist(),
        "wall_clock_s": time.time() - t0,
    }
    # 冒烟结果单独命名，绝不能跟正式结果混在同一个文件名里
    outf = os.path.join(
        outdir,
        "hydration_reference_SMOKE_meaningless.json" if args.smoke else
        "hydration_reference_%s%s_results.json" % (
            "vdwonly" if args.vdw_only else "full",
            "_prodlambda" if args.lambda_vdw_json else "",
        ),
    )
    with open(outf, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n结果已保存: {outf}")


if __name__ == "__main__":
    main()

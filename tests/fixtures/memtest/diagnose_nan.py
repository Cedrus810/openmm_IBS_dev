#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""定位膜体系预平衡的 NaN：逐步报告能量与受力，把责任落到具体一步/具体原子。

不猜，按顺序验证四件事：

  A. **原始 `.gro` 坐标**下的势能与最大受力 —— 输入本身有没有坏掉的地方。
  B. `center_system_rigidly()` 之后的势能 —— 它声称是**纯刚性平移**，
     对周期体系应当**能量完全不变**。若变了，就是这一步破坏了物理
     （wrap 撕分子 / box 处理错 / xyz 搞错），NaN 的责任在此。
  C. **最小化后**的势能、最大受力、以及受力最大的原子属于哪个残基/分子 ——
     若最小化后仍有极端受力，问题是局部结构（插入配体的钢丝、原子重叠）。
  D. **分力项**（bonded / LJ+Coulomb / 各 CustomForce）与**最近原子对距离** ——
     区分"键项爆炸"（分子被撕）与"非键爆炸"（原子重叠）。

  E. 短跑（**不加 barostat**）—— 区分"局部结构问题"与"barostat 问题"。
  F/G. **加膜 barostat**，对比"不初始化速度"（复现生产行为）与"初始化速度"，
     逐段报告**盒子三边与体积**。这是判断"盒子是不是塌了"的直接观测。

用法（在 memtest/ 里，openmm_dev 环境）：

    python diagnose_nan.py                                   # CPU，F/G 各 1000 步
    python diagnose_nan.py --platform=CUDA --steps=200000    # GPU 跑长，逼出 NaN
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))))

import numpy as np
import openmm
from openmm import app, unit

import abfe_core as core

KJ = unit.kilojoule_per_mole
NM = unit.nanometer


def _energy_and_forces(system, positions, box_vectors, platform_name="CPU"):
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(KJ)
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(KJ / NM), dtype=float
    )
    del context, integrator
    return energy, forces


def _report_extreme_forces(forces, topology, label, top_n=8):
    magnitudes = np.linalg.norm(forces, axis=1)
    finite = np.isfinite(magnitudes)
    n_bad = int(np.count_nonzero(~finite))
    print(f"    {label}: max|F| = ", end="")
    if n_bad:
        print(f"NaN/Inf（{n_bad} 个原子）")
    else:
        print(f"{magnitudes.max():.4g} kJ/mol/nm   中位数 {np.median(magnitudes):.4g}")
    atoms = list(topology.atoms())
    order = np.argsort(np.where(finite, magnitudes, np.inf))[::-1][:top_n]
    for index in order:
        atom = atoms[int(index)]
        value = magnitudes[int(index)]
        print(
            f"      idx={int(index):6d} {atom.residue.name:>6s}"
            f"{atom.residue.index:<6d} {atom.name:>6s}  "
            f"|F| = {'NaN' if not np.isfinite(value) else f'{value:.4g}'}"
        )


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {
        a.split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else "")
        for a in sys.argv[1:]
        if a.startswith("--")
    }
    barostat_platform = flags.get("--platform", "CPU")
    barostat_steps = int(flags.get("--steps", "1000"))

    config_path = argv[0] if argv else "abfe_config.json"
    config = {
        k: v
        for k, v in json.load(open(config_path, encoding="utf-8")).items()
        if not k.startswith("_")
    }
    gro_file, top_file = config["gro"], config["top"]
    print(f"配置 {config_path}\n坐标 {gro_file}\n拓扑 {top_file}\n")

    # ---- 载入（走生产用的唯一入口，含 [ pairs ] funct 2 等价转换）----
    gro = app.GromacsGroFile(gro_file)
    top = core.load_gromacs_topology_for_openmm(
        top_file, includeDir=None, periodicBoxVectors=gro.getPeriodicBoxVectors()
    )
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * NM,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    topology = top.topology
    box_vectors = gro.getPeriodicBoxVectors()
    positions = gro.positions
    box_nm = np.asarray(
        [[v.x, v.y, v.z] for v in box_vectors.value_in_unit(NM)]
        if hasattr(box_vectors, "value_in_unit")
        else [[v.x, v.y, v.z] for v in box_vectors],
        dtype=float,
    )
    print(f"原子数 {system.getNumParticles()}   盒 {np.diag(box_nm)}")
    print(f"力对象: {[type(f).__name__ for f in system.getForces()]}\n")

    ligand_indices = [
        int(a.index)
        for a in topology.atoms()
        if a.residue.name == config.get("ligand", "MOL")
    ]
    print(f"配体 {config.get('ligand')} 原子数 {len(ligand_indices)}\n")

    # ---- A. 原始坐标 ----
    print("[A] 原始 .gro 坐标")
    energy_raw, forces_raw = _energy_and_forces(system, positions, box_vectors)
    print(f"    势能 = {energy_raw:.6g} kJ/mol")
    _report_extreme_forces(forces_raw, topology, "受力")

    # ---- B. 刚性居中之后：能量必须不变 ----
    print("\n[B] center_system_rigidly() 之后（纯刚性平移 ⇒ 能量应完全不变）")
    import runabfe

    centered, centered_box = runabfe.center_system_rigidly(
        positions, box_vectors, ligand_indices
    )
    energy_centered, forces_centered = _energy_and_forces(
        system, centered, centered_box
    )
    delta = energy_centered - energy_raw
    print(f"    势能 = {energy_centered:.6g} kJ/mol   Δ = {delta:+.6g}")
    tolerance = max(1.0, abs(energy_raw) * 1e-6)
    if abs(delta) <= tolerance:
        print(f"    ✅ 能量在容差内不变（|Δ| ≤ {tolerance:.3g}）→ 这一步没破坏物理")
    else:
        print(
            f"    ❌ 能量变了 {delta:+.6g} kJ/mol（容差 {tolerance:.3g}）——"
            "刚性平移不该改变周期体系能量。责任在这一步：\n"
            "       可能是逐原子 wrap 撕开了分子、box 处理错、或 xyz 轴用错。"
        )
        _report_extreme_forces(forces_centered, topology, "受力")

    # ---- C. 最小化 ----
    print("\n[C] 最小化 1000 步之后")
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("CPU")
    )
    context.setPeriodicBoxVectors(*centered_box)
    context.setPositions(centered)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=1000)
    state = context.getState(getEnergy=True, getForces=True, getPositions=True)
    energy_min = state.getPotentialEnergy().value_in_unit(KJ)
    forces_min = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(KJ / NM), dtype=float
    )
    minimized = state.getPositions()
    print(f"    势能 = {energy_min:.6g} kJ/mol")
    _report_extreme_forces(forces_min, topology, "受力")

    # ---- D. 分力项 + 最近原子对 ----
    print("\n[D] 分力项（各 Force 单独求值）")
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)
    context.reinitialize(preserveState=True)
    for index, force in enumerate(system.getForces()):
        sub = context.getState(getEnergy=True, groups={index})
        print(
            f"    {type(force).__name__:32s} "
            f"{sub.getPotentialEnergy().value_in_unit(KJ):+.6g} kJ/mol"
        )

    print("\n[D2] 最小化后最近的非键原子对（检测重叠）")
    coords = np.asarray(minimized.value_in_unit(NM), dtype=float)
    heavy = np.asarray(
        [
            int(a.index)
            for a in topology.atoms()
            if (getattr(a.element, "symbol", "") or "").upper() != "H"
        ],
        dtype=int,
    )
    # 网格化找最近对（0.15 nm 内），避免 N² 爆炸
    cell = 0.2
    grid = {}
    for i in heavy:
        key = tuple((coords[i] // cell).astype(int))
        grid.setdefault(key, []).append(int(i))
    bonded = set()
    for a, b in topology.bonds():
        bonded.add((min(a.index, b.index), max(a.index, b.index)))
    closest = []
    for key, members in grid.items():
        neighbours = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbours += grid.get((key[0] + dx, key[1] + dy, key[2] + dz), [])
        for i in members:
            for j in neighbours:
                if j <= i:
                    continue
                if (i, j) in bonded:
                    continue
                delta_vec = coords[i] - coords[j]
                delta_vec -= np.diag(box_nm) * np.round(delta_vec / np.diag(box_nm))
                distance = float(np.linalg.norm(delta_vec))
                if distance < 0.15:
                    closest.append((distance, i, j))
    closest.sort()
    atoms = list(topology.atoms())
    if not closest:
        print("    ✅ 没有 < 0.15 nm 的非键重原子对")
    else:
        print(f"    ⚠️ {len(closest)} 对 < 0.15 nm（最近 8 对）：")
        for distance, i, j in closest[:8]:
            ai, aj = atoms[i], atoms[j]
            print(
                f"      {distance:.4f} nm  "
                f"{ai.residue.name}{ai.residue.index}/{ai.name}  ↔  "
                f"{aj.residue.name}{aj.residue.index}/{aj.name}"
            )

    # ---- E. 短跑，看能量往哪走 ----
    print("\n[E] Langevin 短跑（2 fs × 500 步，每 100 步报一次）")
    del context, integrator
    langevin = openmm.LangevinMiddleIntegrator(
        float(config.get("temperature", 300.0)) * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    simulation = app.Simulation(
        topology, system, langevin, openmm.Platform.getPlatformByName("CPU")
    )
    simulation.context.setPeriodicBoxVectors(*centered_box)
    simulation.context.setPositions(minimized)
    for step in range(5):
        try:
            simulation.step(100)
        except Exception as exc:
            print(f"    ❌ 第 {step * 100}–{(step + 1) * 100} 步炸了：{exc}")
            break
        e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(f"    {(step + 1) * 100:5d} 步: {e.value_in_unit(KJ):+.6g} kJ/mol")
    else:
        print("    ✅ 500 步无 NaN（说明 NaN 与 barostat 或更长时间尺度有关）")

    # ---- F/G. 带膜 barostat：对比"不初始化速度"与"初始化速度" ----
    # [E] 已证明无 barostat 时 500 步不炸，所以责任落在 barostat 上。
    # 生产代码的 `pre_equilibrate()` **从不调 setVelocitiesToTemperature**
    # （`ibs_engine` 里到处都调，唯独它没有），所以体系从 **0 K** 起跑，
    # 而膜 barostat 从第 0 步就以频率 25 做体积移动。冷体系下 |ΔE| 很小、
    # `P·ΔV` 项主导 → barostat 会接受大幅压缩 → 盒子（尤其 Z 自由的那一维）
    # 可能塌掉 → 原子挤在一起 → NaN。
    #
    # 下面两段把这个假设变成可观测的：**逐段报告盒子三边与体积**。
    # 若 [F] 的 z 明显收缩而 [G] 稳定，假设成立，修法就是初始化速度
    #（必要时再加一段 NVT 加热），而不是去动 barostat 参数。
    temperature = float(config.get("temperature", 300.0)) * unit.kelvin
    membrane_cfg = (config.get("membrane") or {})

    def _run_with_barostat(label, init_velocities, n_steps=None, report_every=None):
        n_steps = n_steps or barostat_steps
        report_every = report_every or max(100, n_steps // 20)
        print(f"\n[{label}] 膜 barostat + "
              f"{'初始化速度到 T' if init_velocities else '不初始化速度（复现生产行为）'}"
              f"  |  {barostat_platform}, {n_steps} 步")
        system_copy = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(system)
        )
        system_copy.addForce(
            openmm.MonteCarloMembraneBarostat(
                1.0 * unit.bar,
                float(membrane_cfg.get("surface_tension_bar_nm", 0.0))
                * unit.bar * unit.nanometer,
                temperature,
                openmm.MonteCarloMembraneBarostat.XYIsotropic,
                openmm.MonteCarloMembraneBarostat.ZFree,
                int(membrane_cfg.get("barostat_frequency", 25)),
            )
        )
        integrator_local = openmm.LangevinMiddleIntegrator(
            temperature, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
        )
        resolved_platform, platform_props = (
            ("CUDA", {"Precision": "mixed"})
            if barostat_platform.upper() == "CUDA"
            else (barostat_platform, {})
        )
        sim = app.Simulation(
            topology,
            system_copy,
            integrator_local,
            openmm.Platform.getPlatformByName(resolved_platform),
            platform_props,
        )
        sim.context.setPeriodicBoxVectors(*centered_box)
        sim.context.setPositions(minimized)
        if init_velocities:
            sim.context.setVelocitiesToTemperature(temperature, 20260730)

        def _box_line(step):
            st = sim.context.getState(getEnergy=True)
            vectors = st.getPeriodicBoxVectors().value_in_unit(NM)
            a, b, c = vectors[0][0], vectors[1][1], vectors[2][2]
            energy = st.getPotentialEnergy().value_in_unit(KJ)
            print(
                f"    {step:5d} 步: 盒 {a:.4f} × {b:.4f} × {c:.4f} nm  "
                f"体积 {a * b * c:9.2f} nm³  PE {energy:+.6g} kJ/mol"
            )
            return c

        c0 = _box_line(0)
        for chunk in range(n_steps // report_every):
            try:
                sim.step(report_every)
            except Exception as exc:
                print(
                    f"    ❌ 第 {chunk * report_every}–{(chunk + 1) * report_every} 步炸了：{exc}"
                )
                return False, None
            c_now = _box_line((chunk + 1) * report_every)
        # 统一符号约定：**正 = 膨胀，负 = 收缩**。
        # （早先这里返回的是 (c0-c_now)/c0，负值反而代表膨胀，汇总又打上"收缩幅度"
        #  的标签，把两个膨胀读成了收缩。约定写在一处，避免再错。）
        change_percent = (c_now - c0) / c0 * 100.0
        print(
            f"    Z 变化: {c0:.4f} → {c_now:.4f} nm（{change_percent:+.2f}%，"
            f"{'膨胀' if change_percent > 0 else '收缩'}）"
        )
        return True, change_percent

    ok_f, change_f = _run_with_barostat("F", init_velocities=False)
    ok_g, change_g = _run_with_barostat("G", init_velocities=True)

    print("\n结论（Z 变化：正 = 膨胀，负 = 收缩）：")
    if not ok_f and ok_g:
        print("  ✅ 不初始化速度会炸、初始化后不炸 → 责任在 0 K 冷启动 + barostat。")
        print("     修法 = pre_equilibrate() 最小化后 setVelocitiesToTemperature，")
        print("     **仅对膜体系生效**（可溶路径必须逐位不变，§7.7 / R7）。")
    elif ok_f and ok_g:
        print(f"  ⚠️ 两段都没炸。[F] {change_f:+.2f}%   [G] {change_g:+.2f}%")
        if change_f > 0 and change_g > 0:
            print("     两段都在【膨胀】→ 不是盒子被压塌。零表面张力的膜 barostat")
            print("     本来就会在近似定容下调整面积/厚度比（XY 收、Z 涨），属正常弛豫。")
            print("     ⇒ NaN 发生在本次步数之后。离线再猜性价比低，改用【在跑中监控】：")
            print("       生产预平衡现在会周期性记录盒体积/势能/温度，直接看它在哪一步崩。")
        else:
            print("     若 [F] 明显收缩而 [G] 稳定，支持【冷启动压塌】；否则加大 --steps。")
    elif not ok_f and not ok_g:
        print("  ❌ 两段都炸 → 不只是速度初始化，barostat 本身或膜结构有更深的毛病。")
        print("     下一步：barostat 频率调小 / 用 ZFixed 固定法向，分别定位是哪一维。")
    else:
        print("  ❓ 初始化速度反而炸了 —— 与预期相反，请把上面两段输出贴回来。")

    print(
        "\n判读指引（前几段）：\n"
        "  [B] 变了      → 我们的预处理破坏了物理，先修那一步。\n"
        "  [C] max|F| 极大 → 局部结构问题；看受力最大的残基是谁（配体？插入位点？）。\n"
        "  [D] 键项极大   → 分子被撕（PBC / 拓扑与坐标错位）；非键极大 → 原子重叠。\n"
        "  [E] 500 步就炸 → 局部问题；500 步不炸 → 查 barostat（该段刻意没加）。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

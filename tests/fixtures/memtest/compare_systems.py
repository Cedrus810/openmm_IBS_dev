#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比"诊断脚本构造的 System"与"生产落盘的 system_native.xml"。

## 为什么需要这一步

`diagnose_nan.py` 在 CUDA 上带膜 barostat 跑了 200 000 步（0.4 ns）**不炸**；
而生产跑在**第 5000 步之前**就 NaN（`pre_equilibration_monitor.csv` 与
`pre_equilibration.dcd` 都是 0 字节 —— 前者每 5000 步写、后者每 10000 步写）。

两边用的 `createSystem(...)` 参数逐字相同（PME / 1.0 nm / HBonds / rigidWater /
ewaldErrorTolerance=0.0005），所以差异不在那几个 kwarg 上。但生产还多做了
`save_native_system` → `load_native_system` 的 XML 往返、`topology.cif` 往返、
以及可能的缓存命中。**System 只要有一处不同，哈密顿量就不同。**

本脚本把两个 System 摆在一起逐项比，并在**同一组坐标**上算每个力的能量——
能量对不上就说明哈密顿量真的不同，那才是 NaN 的根。

用法（memtest/ 内）：

    python compare_systems.py [abfe_config.json]
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


def _describe(system, label):
    print(f"\n--- {label} ---")
    print(f"  粒子数        : {system.getNumParticles()}")
    print(f"  约束数        : {system.getNumConstraints()}")
    forces = list(system.getForces())
    print(f"  力对象 ({len(forces)}): {[type(f).__name__ for f in forces]}")
    info = {}
    for index, force in enumerate(forces):
        name = type(force).__name__
        detail = {}
        if isinstance(force, openmm.NonbondedForce):
            detail = {
                "particles": force.getNumParticles(),
                "exceptions": force.getNumExceptions(),
                "method": int(force.getNonbondedMethod()),
                "cutoff_nm": force.getCutoffDistance().value_in_unit(NM),
                "uses_dispersion_correction": force.getUseDispersionCorrection(),
                "switching": force.getUseSwitchingFunction(),
                "ewald_tol": force.getEwaldErrorTolerance(),
                "offsets": force.getNumParticleParameterOffsets(),
            }
        elif isinstance(force, openmm.HarmonicBondForce):
            detail = {"bonds": force.getNumBonds()}
        elif isinstance(force, openmm.HarmonicAngleForce):
            detail = {"angles": force.getNumAngles()}
        elif isinstance(force, openmm.PeriodicTorsionForce):
            detail = {"torsions": force.getNumTorsions()}
        elif isinstance(force, openmm.CustomTorsionForce):
            detail = {"torsions": force.getNumTorsions(), "expr": force.getEnergyFunction()[:60]}
        info[f"{index}:{name}"] = detail
        if detail:
            print(f"    [{index}] {name}: {detail}")
    box = system.getDefaultPeriodicBoxVectors()
    print(f"  默认盒        : {[round(v.x, 5) for v in box]} / "
          f"{[round(v.y, 5) for v in box]} / {[round(v.z, 5) for v in box]}")
    return info


def _per_force_energy(system, positions, box_vectors, label):
    work = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    for index, force in enumerate(work.getForces()):
        force.setForceGroup(index)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
        work, integrator, openmm.Platform.getPlatformByName("CPU")
    )
    context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    energies = {}
    for index, force in enumerate(work.getForces()):
        value = context.getState(
            getEnergy=True, groups={index}
        ).getPotentialEnergy().value_in_unit(KJ)
        energies[f"{index}:{type(force).__name__}"] = value
    total = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)
    energies["TOTAL"] = total
    del context, integrator
    print(f"\n--- {label} 逐力能量 ---")
    for key, value in energies.items():
        print(f"    {key:36s} {value:+.6g} kJ/mol")
    return energies


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "abfe_config.json"
    config = {
        k: v
        for k, v in json.load(open(config_path, encoding="utf-8")).items()
        if not k.startswith("_")
    }
    output_dir = config.get("output", "./output_membrane")
    native_xml = os.path.join(output_dir, "system_native.xml")
    if not os.path.isfile(native_xml):
        print(f"❌ 找不到 {native_xml} —— 先跑一次 runabfe 让它落盘。")
        return 1

    # ---- 诊断口径：直接从拓扑构建 ----
    gro = app.GromacsGroFile(config["gro"])
    top = core.load_gromacs_topology_for_openmm(
        config["top"], includeDir=None, periodicBoxVectors=gro.getPeriodicBoxVectors()
    )
    diagnostic = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * NM,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    diagnostic.setDefaultPeriodicBoxVectors(*gro.getPeriodicBoxVectors())

    # ---- 生产口径：读落盘的 XML ----
    with open(native_xml, encoding="utf-8") as handle:
        production = openmm.XmlSerializer.deserialize(handle.read())

    info_d = _describe(diagnostic, "诊断构造（diagnose_nan.py 口径）")
    info_p = _describe(production, f"生产落盘（{native_xml}）")

    print("\n=== 结构差异 ===")
    差异 = []
    if diagnostic.getNumParticles() != production.getNumParticles():
        差异.append(
            f"粒子数 {diagnostic.getNumParticles()} vs {production.getNumParticles()}"
        )
    if diagnostic.getNumConstraints() != production.getNumConstraints():
        差异.append(
            f"约束数 {diagnostic.getNumConstraints()} vs {production.getNumConstraints()}"
        )
    names_d = [type(f).__name__ for f in diagnostic.getForces()]
    names_p = [type(f).__name__ for f in production.getForces()]
    if names_d != names_p:
        差异.append(f"力对象列表不同:\n      诊断 {names_d}\n      生产 {names_p}")
    for key in sorted(set(info_d) | set(info_p)):
        if info_d.get(key) != info_p.get(key):
            差异.append(f"{key}: 诊断 {info_d.get(key)} vs 生产 {info_p.get(key)}")
    if 差异:
        print("  ❌ 发现差异：")
        for item in 差异:
            print(f"    - {item}")
    else:
        print("  ✅ 结构完全一致（粒子/约束/力对象/各力计数与非键设置）")

    # ---- 同一组坐标上的逐力能量 ----
    positions = gro.positions
    box_vectors = gro.getPeriodicBoxVectors()
    e_d = _per_force_energy(diagnostic, positions, box_vectors, "诊断构造")
    e_p = _per_force_energy(production, positions, box_vectors, "生产落盘")

    print("\n=== 能量差异（同一组坐标）===")
    worst = 0.0
    for key in sorted(set(e_d) | set(e_p)):
        a, b = e_d.get(key), e_p.get(key)
        if a is None or b is None:
            print(f"  ❌ {key}: 只有一侧有（诊断 {a} / 生产 {b}）")
            continue
        delta = b - a
        worst = max(worst, abs(delta))
        flag = "✅" if abs(delta) < max(1.0, abs(a) * 1e-6) else "❌"
        print(f"  {flag} {key:36s} Δ = {delta:+.6g} kJ/mol")

    # ---- 坐标来源对比：.gro vs topology.cif ----
    #
    # `load_native_system` 的坐标恢复顺序是：可信的预平衡 DCD 末帧 →
    # **`topology.cif` 的坐标** → 才回退到 `.gro`。
    # 本次 `pre_equilibration.dcd` 是 0 字节，所以缓存命中时生产用的是 **CIF 坐标**，
    # 而 `diagnose_nan.py` 一直用 `.gro`。这正是两边行为不同的一个具体差异。
    cif_path = os.path.join(output_dir, "topology.cif")
    print("\n=== 坐标来源对比：.gro vs topology.cif ===")
    if not os.path.isfile(cif_path):
        print(f"  （{cif_path} 不存在，跳过）")
    else:
        try:
            cif_file = app.PDBxFile(cif_path)
            gro_xyz = np.asarray(
                gro.positions.value_in_unit(NM), dtype=float
            )
            cif_xyz = np.asarray(
                cif_file.positions.value_in_unit(NM), dtype=float
            )
            print(f"  .gro 原子数 {gro_xyz.shape[0]}   CIF 原子数 {cif_xyz.shape[0]}")
            if gro_xyz.shape != cif_xyz.shape:
                print("  ❌ 原子数不同 —— CIF 缓存与输入不是同一个体系")
            else:
                delta = np.linalg.norm(cif_xyz - gro_xyz, axis=1)
                print(
                    f"  逐原子位移: max {delta.max():.6f} nm  "
                    f"中位数 {np.median(delta):.6f}  "
                    f"> 0.01 nm 的原子数 {int(np.count_nonzero(delta > 0.01))}"
                )
                if delta.max() < 1.0e-3:
                    print("  ✅ 仅舍入级差异（CIF 以 Å 有限位数写盘），坐标等价")
                else:
                    print("  ❌ 坐标**实质不同** —— 生产从 CIF 取坐标，等于换了一组构型！")
                    atoms = list(top.topology.atoms())
                    order = np.argsort(delta)[::-1][:10]
                    for index in order:
                        atom = atoms[int(index)]
                        print(
                            f"      idx={int(index):6d} {atom.residue.name:>6s}"
                            f"{atom.residue.index:<6d} {atom.name:>6s}  "
                            f"|Δ| = {delta[int(index)]:.4f} nm"
                        )
                    # 顺带比一下能量：CIF 坐标下的势能若远高于 .gro，就是它了。
                    e_gro = _per_force_energy(
                        production, gro.positions, box_vectors, "生产 System @ .gro 坐标"
                    )
                    e_cif = _per_force_energy(
                        production, cif_file.positions, box_vectors,
                        "生产 System @ CIF 坐标",
                    )
                    print(
                        f"\n  总势能: .gro {e_gro['TOTAL']:+.6g}  "
                        f"CIF {e_cif['TOTAL']:+.6g}  "
                        f"Δ = {e_cif['TOTAL'] - e_gro['TOTAL']:+.6g} kJ/mol"
                    )
        except Exception as exc:
            print(f"  ⚠️ CIF 读取失败: {type(exc).__name__}: {exc}")

    print("\n结论：")
    if 差异 or worst > 1.0:
        print("  ❌ 两个 System **不是**同一个哈密顿量 —— 这就是 NaN 的根。")
        print("     生产多做的是 save_native_system → load_native_system 的 XML 往返、")
        print("     topology.cif 往返、以及缓存命中；差异出在其中某一步。")
        print("     下一步：按上面列出的具体差异去查对应的落盘/读回代码。")
    else:
        print("  ✅ 两个 System 一致 ⇒ 差异不在 System，而在**别处**：")
        print("     候选（按可疑度）：")
        print("       1. 喂给 pre_equilibrate 的 positions 不是我诊断里那一组")
        print("          （缓存命中时 load_native_system 可能从别处取坐标）；")
        print("       2. topology 与 System 的原子顺序不一致（topology.cif 往返）；")
        print("       3. Simulation 的 platform properties 不同（如 CudaCompiler）。")
        print("     用 --dump-positions 落盘生产坐标再逐点比对。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

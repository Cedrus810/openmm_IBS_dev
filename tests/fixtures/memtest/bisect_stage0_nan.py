#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 0 attachment 腿 NaN 的三臂 bisect：Boresch 到底有没有参与？

## 为什么需要它

2026-08-03 的 100 ns 运行在 attachment 腿第一个态（λ_rest = 1.0）出
`Particle coordinate is NaN`。MEM-06 的起点体检显示起点是**健康**的：

    g0 = -539566 kJ/mol, g3(Boresch) = 1.42e-11 kJ/mol
    max|F| = 5292 kJ/mol/nm (idx=3353 GLU208/C), 中位数 802.7

三条证据都指向"不是 Boresch"：
  1. λ=1 时六个几何量与已提交平衡值逐位相同 → 正好在势能最小点，力也为零（g3 ≈ 0）；
  2. 受力最大的原子是 GLU208/C，与锚点（ASP82 N/CA/CB、MOL O8/C9/C10）不相干；
  3. **同一个全强度限制力刚在 rebalance 里跑完 100 ps 没事**
     （T 191→304 K、PE −509k、体积恒定 440.089 nm³）。

但起点体检有个**盲区**：限制力恰好在最小点，所以它看不到"跑起来之后才显现"的问题。
这个脚本就是补这个盲区——用**与生产同一条函数**跑短程，逐步监控，三臂对照：

    A  as_is        λ_rest = 1.0，2 fs           ← 忠实复现生产（同起点、同种子、同步数）
    B  no_restraint λ_rest = 0.0，2 fs           ← 限制力完全关掉
    C  small_dt     λ_rest = 1.0，0.5 fs         ← 只改积分步长

判读：
  * B 不崩、A 崩  → 与限制力有关（但注意 λ=1 与 λ=0 的差别不只是"有没有力"，
                    λ=0 时那一项整体乘 0，所以这一臂是干净的开关）
  * B 也崩        → **与 Boresch 无关**，问题在起点坐标 / 体系本身 / 积分设置
  * C 不崩、A 崩  → 数值稳定性（步长 / 约束），不是几何或参数选择问题
  * 三臂全活      → **本次没复现**。别把它读成"已排除"：λ 态还有 0.35 / 0.1 / 0.0
                    三个，且每态之后还有 250000 步采样段。用 --steps 250000 或
                    在 A 臂后继续扫下一个 λ 才算覆盖到生产走过的路径。

⚠️ 本脚本**只调生产函数**加载体系（`runabfe.load_native_system` +
`abfe_core.load_gromacs_topology_for_openmm` 链路），不自己搭 System ——
§0.5.7 的教训是"离线重建与生产路径不一致，白花几轮"。

用法（在 memtest/ 下）：

    mamba activate openmm_dev
    python bisect_stage0_nan.py                    # 三臂各 50000 步（100 ps，= 生产）
    python bisect_stage0_nan.py --arms A B         # 只跑指定臂
    python bisect_stage0_nan.py --steps 250000     # 跑到生产的采样段长度
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openmm
from openmm import app, unit

import abfe_core as core
import runabfe
from ibs_engine import (
    BORESCH_ATTACHMENT_FORCE_GROUP,
    BORESCH_ATTACHMENT_LAMBDA_NAME,
    add_scalable_boresch_restraint,
)

OUTPUT_DIR = "output_membrane_100ns"
CONFIG = "abfe_config.json"

ARMS = {
    # 名字: (lambda_rest, timestep_fs)
    "A": ("as_is           λ=1.0, 2 fs", 1.0, 2.0),
    "B": ("no_restraint    λ=0.0, 2 fs", 0.0, 2.0),
    "C": ("small_dt        λ=1.0, 0.5 fs", 1.0, 0.5),
}

# 生产在 attachment 腿第一个态上要跑的平衡步数
# （`abfe_pipeline` 的 `attachment_equil_steps_per_state`，默认 50_000 = 100 ps）。
# ⚠️ 默认必须等于它：本脚本第一版只跑 5000 步（10 ps），三臂全活但**什么都没证明**
# —— NaN 落在 10–100 ps 之间。测试跑得比生产短，等于没测。
PRODUCTION_EQUIL_STEPS = 50_000
# 生产的速度种子：`integrator.setRandomNumberSeed(seed)` +
# 每个 λ 态 `setVelocitiesToTemperature(T, seed + 7919*k + 1)`，
# 而第一个跑的态是 k = K-1 = 3（扫描顺序从全强度端往下）。
PRODUCTION_ATTACHMENT_SEED = 20260728
PRODUCTION_FIRST_STATE_K = 3


def _load():
    config = {k: v for k, v in json.load(open(CONFIG, encoding="utf-8")).items()
              if not str(k).startswith("_")}
    system, topology, positions, box_vectors, ligand = runabfe.load_native_system(
        OUTPUT_DIR,
        gro_file=config["gro"],
        top_file=config["top"],
        gmx_include_dir=runabfe.find_gmx_include_dir(config.get("gmx_path")),
        phase="complex",
        require_bonded_topology=True,
    )
    # 起点必须是 attachment 腿真正拿到的那一份：rebalance 末帧（带 Boresch 的
    # 100 ps 再平衡之后）。用 GRO 初始坐标是另一个体系状态，比出来的结论没用。
    import mdtraj as md
    reb = os.path.join(OUTPUT_DIR, "rebalance_traj.dcd")
    if not os.path.exists(reb):
        raise SystemExit(f"找不到 {reb}；本脚本要用 attachment 腿的真实起点。")
    traj = md.load(reb, top=md.Topology.from_openmm(topology))
    positions = traj.xyz[-1] * unit.nanometer
    box_vectors = traj.unitcell_vectors[-1] * unit.nanometer
    print(f"  起点 = rebalance 末帧（{len(traj)} 帧），盒长 "
          f"{np.diag(np.asarray(traj.unitcell_vectors[-1]))} nm")

    committed = os.path.join(OUTPUT_DIR, "checkpoints",
                             "boresch_equilibrium_committed.json")
    boresch = json.load(open(committed, encoding="utf-8"))
    return system, topology, positions, box_vectors, boresch, config


def _run_arm(label, lam, dt_fs, system, topology, positions, box_vectors,
             boresch, temperature_k, steps, report_every, seed):
    print(f"\n{'='*72}\n[{label}]  {steps} 步 × {dt_fs} fs = "
          f"{steps*dt_fs/1000:.1f} ps\n{'='*72}")
    work = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    work.thisown = 1
    if not add_scalable_boresch_restraint(work, boresch):
        raise SystemExit("Boresch 力注入失败")
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtosecond
    )
    integrator.setRandomNumberSeed(int(seed))
    sim = app.Simulation(topology, work, integrator,
                         openmm.Platform.getPlatformByName("CUDA"))
    sim.context.setPositions(positions)
    sim.context.setPeriodicBoxVectors(*box_vectors)
    sim.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam))
    # 与生产逐位相同的速度种子，这样 A 臂是**忠实复现**而不是"另一条随机轨迹"。
    velocity_seed = int(seed) + 7919 * PRODUCTION_FIRST_STATE_K + 1
    sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, velocity_seed)
    print(f"  integrator seed={int(seed)}, velocity seed={velocity_seed}")

    # OpenMM 的 Residue 只有 .index / .id（`resSeq` 是 mdtraj 的属性）。
    # 用与 `abfe_core.assert_starting_state_is_sane` 相同的标签口径，便于两边对照。
    atoms = list(topology.atoms())

    def _label(index):
        a = atoms[int(index)]
        return f"{a.residue.name}{a.residue.index}/{a.name}"

    def snapshot(step):
        st = sim.context.getState(getEnergy=True, getForces=True)
        pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f = np.asarray(st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer), dtype=float)
        mag = np.linalg.norm(f, axis=1)
        ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        n_dof = 3 * work.getNumParticles() - work.getNumConstraints()
        temp = 2 * ke / (n_dof * 0.008314462618)
        if not np.all(np.isfinite(mag)):
            print(f"  步 {step:7d}  ✗ 出现非有限受力（{int((~np.isfinite(mag)).sum())} 个原子）")
            return False
        worst = int(np.argmax(mag))
        gb = sim.context.getState(
            getEnergy=True, groups={BORESCH_ATTACHMENT_FORCE_GROUP}
        ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        print(f"  步 {step:7d}  PE={pe:14.1f}  T={temp:6.1f} K  "
              f"max|F|={mag.max():11.4g} @ {_label(worst)}"
              f"  E_Boresch={gb:12.4g}")
        return np.isfinite(pe) and abs(pe) < 1e12

    snapshot(0)
    done = 0
    while done < steps:
        chunk = min(report_every, steps - done)
        try:
            sim.step(chunk)
        except Exception as exc:
            print(f"  步 {done+chunk:7d}  ✗ {type(exc).__name__}: {exc}")
            print(f"\n  → [{label}] **崩了**，累计 {(done+chunk)*dt_fs/1000:.2f} ps")
            return False
        done += chunk
        if not snapshot(done):
            print(f"\n  → [{label}] **发散**，累计 {done*dt_fs/1000:.2f} ps")
            return False
    print(f"\n  → [{label}] 跑完 {steps*dt_fs/1000:.1f} ps **没崩**")
    return True


def run_full_sequence(system, topology, positions, box_vectors, boresch,
                      temperature_k, seed, report_every, equil_steps,
                      sample_steps, steps_per_sample):
    """D 臂：**忠实重放**生产 `run_boresch_attachment_leg` 的整条 λ 序列。

    A / B 臂只覆盖第一个 λ 态（λ=1.0）并且都活了 100 ps，而生产的 traceback
    落在 `simulation.step(int(equil_steps_per_state))` —— 某个态的**平衡段**。
    所以崩的是后面的态（k=2/1/0）。这一臂按生产的顺序与步数走完：

        order = [K-1 … 0]（从全强度端往下）
        每态：setParameter(λ_k) → setVelocitiesToTemperature(seed+7919k+1)
              → step(equil) → 250 × step(steps_per_sample)

    与生产唯一的差别是不做能量记账（`u_boresch`/`u_base`），因为那些 getState
    不影响动力学。总长 4 × (100 + 500) ps = 2.4 ns，约 6 分钟。
    """
    lambdas = [0.0, 0.1, 0.35, 1.0]
    order = list(range(len(lambdas) - 1, -1, -1))
    print(f"\n{'='*72}\n[D full_sequence] 重放生产整条 λ 序列 "
          f"{lambdas}，扫描顺序 k={order}\n{'='*72}")

    work = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    work.thisown = 1
    if not add_scalable_boresch_restraint(work, boresch):
        raise SystemExit("Boresch 力注入失败")
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(int(seed))
    sim = app.Simulation(topology, work, integrator,
                         openmm.Platform.getPlatformByName("CUDA"))
    sim.context.setPositions(positions)
    sim.context.setPeriodicBoxVectors(*box_vectors)
    atoms = list(topology.atoms())

    def report(tag, cumulative_steps):
        st = sim.context.getState(getEnergy=True, getForces=True)
        pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f = np.asarray(st.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer), dtype=float)
        mag = np.linalg.norm(f, axis=1)
        if not np.all(np.isfinite(mag)) or not np.isfinite(pe):
            print(f"    {tag}  ✗ 非有限（PE={pe}）")
            return False
        worst = int(np.argmax(mag))
        a = atoms[worst]
        print(f"    {tag}  t={cumulative_steps*2e-3:8.1f} ps  PE={pe:13.1f}  "
              f"max|F|={mag.max():10.4g} @ {a.residue.name}{a.residue.index}/{a.name}")
        return True

    cumulative = 0
    for k in order:
        lam = float(lambdas[k])
        velocity_seed = int(seed) + 7919 * k + 1
        sim.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, lam)
        sim.context.setVelocitiesToTemperature(
            temperature_k * unit.kelvin, velocity_seed
        )
        print(f"\n  ── 态 k={k}  λ={lam}  velocity_seed={velocity_seed} ──")
        for phase, total in (("平衡", equil_steps), ("采样", sample_steps)):
            done = 0
            while done < total:
                chunk = min(report_every, total - done)
                try:
                    sim.step(chunk)
                except Exception as exc:
                    print(f"    ✗ {phase}段 {done+chunk}/{total} 步时崩: "
                          f"{type(exc).__name__}: {exc}")
                    print(f"\n  → **崩在 k={k}（λ={lam}）的{phase}段**，"
                          f"该态内第 {done+chunk} 步，累计 "
                          f"{(cumulative+done+chunk)*2e-3:.1f} ps")
                    return False, k, lam, phase
                done += chunk
                if done % (report_every * 10) == 0 or done == total:
                    if not report(f"k={k} {phase} {done:6d}/{total}",
                                  cumulative + done):
                        print(f"\n  → **发散在 k={k}（λ={lam}）的{phase}段**")
                        return False, k, lam, phase
            cumulative += total
    print(f"\n  → D 臂走完整条序列 {cumulative*2e-3:.1f} ps **没崩**")
    return True, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=PRODUCTION_EQUIL_STEPS,
                    help=f"每臂步数（默认 {PRODUCTION_EQUIL_STEPS}，= 生产第一个态的平衡步数）")
    ap.add_argument("--seed", type=int, default=PRODUCTION_ATTACHMENT_SEED,
                    help="与生产一致的 attachment seed（默认 20260728）")
    ap.add_argument("--report-every", type=int, default=500)
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS) + ["D"])
    ap.add_argument("--sample-steps", type=int, default=250_000,
                    help="D 臂每态采样段步数（默认 250000，= 生产）")
    ap.add_argument("--steps-per-sample", type=int, default=1_000)
    args = ap.parse_args()

    print("加载体系（走生产函数）...")
    system, topology, positions, box_vectors, boresch, config = _load()
    temperature_k = float(config.get("temperature", 303.15))
    print(f"  原子数 {system.getNumParticles()}，约束 {system.getNumConstraints()}，"
          f"T = {temperature_k} K")
    print(f"  Boresch: kr={boresch['force_constants']['kr']:.1f}, "
          f"r0={boresch['equilibrium_values']['r0']:.4f} nm")
    if args.steps < PRODUCTION_EQUIL_STEPS:
        print(f"  ⚠️ 每臂只跑 {args.steps} 步 < 生产的 {PRODUCTION_EQUIL_STEPS} 步："
              "跑得比生产短，'没崩'不能作为证据。")

    results = {}
    failure = None
    for arm in args.arms:
        if arm == "D":
            ok, k, lam, phase = run_full_sequence(
                system, topology, positions, box_vectors, boresch,
                temperature_k, args.seed, args.report_every, args.steps,
                args.sample_steps, args.steps_per_sample,
            )
            results["D"] = ok
            if not ok:
                failure = (k, lam, phase)
            continue
        label, lam, dt_fs = ARMS[arm]
        results[arm] = _run_arm(
            label, lam, dt_fs, system, topology, positions, box_vectors,
            boresch, temperature_k, args.steps, args.report_every, args.seed,
        )

    print(f"\n{'='*72}\n结论\n{'='*72}")
    for arm in args.arms:
        name = "full_sequence   生产整条 λ 序列" if arm == "D" else ARMS[arm][0]
        print(f"  {arm}  {name:32s} {'存活' if results[arm] else '崩了'}")
    if results.get("B") is False:
        print("\n  → B（限制力关掉）也崩 ⇒ **与 Boresch 无关**。"
              "查起点坐标 / PBC 修复 / 体系本身 / 积分设置。")
    elif results.get("A") is False and results.get("B") is True:
        print("\n  → 只有带限制力时崩 ⇒ 与 Boresch 有关。"
              "下一步看 C：若 C 存活则是数值稳定性（步长/约束），"
              "不是锚点几何或参数选择。")
    if failure is not None:
        k, lam, phase = failure
        print(f"\n  → 生产序列崩在 **k={k}（λ={lam}）的{phase}段**。"
              "λ 越小限制力越弱，所以若 k<3 崩而 k=3 没崩，"
              "问题几乎不可能是限制力本身。")
    if results.get("A") is True and results.get("B") is True and "D" not in results:
        print("\n  → A/B 都活只覆盖了**第一个** λ 态。生产的 traceback 落在某个态的"
              "平衡段，所以还剩 k=2/1/0 三个态没测：加 --arms D 走完整条序列。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Boresch 开/关 对照：限制到底有没有改变 fully-charged 口袋系综。

**要回答的问题**（P1-18 第 1 项 / P1-17 的前置判别）：

复合物 charging 的真实端（λ_coul=1）全程带满强度 Boresch
（`ibs_engine.py:889` `fixed_lam=1.0`，力常数被编译进表达式，连全局参数都没有）。
所以我们测到的 171.08 kJ/mol 是**受约束系综**的配体-环境静电，不是物理结合态的。

实测对照：

    水中    180.96 kJ/mol
    口袋    171.08 kJ/mol      → 口袋比水差 9.88

`result.txt` 要成立（复合物去电荷 75.1 > 溶剂 68.1），这个符号必须反过来。

**判据**：从同一批 fully-charged 坐标出发，Boresch 开 / 关 各跑 N 条独立短轨迹，
比 ⟨ΔU⟩、氢键占据、配体 RMSD/质心位移。

- 关掉后口袋 ⟨ΔU⟩ 冲向或越过 181  → 限制就是原因，接着做 P1-17 的 A′→A 腿去定量
- 关掉也不动                        → 限制不是原因，只剩 `llfreeze` 口径那条线

**为什么这一步值得先做**：它不需要任何新的 alchemical 机制，几小时出结果，
而且直接决定要不要花力气去建那条腿。

⟨ΔU⟩ 的定义与 `diagnose_charging_linear_response.py` 完全一致——
`ΔU = U(λ_coul=0) − U(λ_coul=1)`，在同一帧坐标上求，λ 缩放走生产的
`ibs_engine.configure_pme_ligand_charge_offsets`（含 llfreeze L–L 冻结 +
PME 电荷 offset），所以数值可以和 171.08 / 180.96 直接比。
Boresch 势不依赖 λ_coul，在 U(0)−U(1) 里精确抵消，不需要特殊处理。

**NVT**：`output_lrc_fix/system_native.xml` 里本来就没有 barostat（已核），
脚本的 `_strip_barostats` 只是保险。⟨ΔU⟩ 是同一帧上两个 Hamiltonian 之差，
与系综是 NVT 还是 NPT 无关。

用法::

    python tools/diagnostics/diagnose_boresch_ensemble_effect.py --run-dir output_lrc_fix \\
        --n-replicas 3 --equil-ps 200 --production-ns 2.0
"""

from __future__ import annotations

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))


import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import openmm
from openmm import app, unit, XmlSerializer

import runabfe
from ibs_engine import configure_pme_ligand_charge_offsets

R_KJ = 0.008314462618
LAMBDA_NAME = "lambda_coul"
POCKET_CUTOFF_NM = 0.9

# 生产那轮在 λ=1 上测到的值，作为 ON 臂的对照锚点（采样长度不同，只看量级）。
REFERENCE_POCKET_DU_KJ = 171.08
REFERENCE_WATER_DU_KJ = 180.96


# ---------------------------------------------------------------------------
# 体系构建
# ---------------------------------------------------------------------------


def _strip_barostats(system: openmm.System) -> int:
    removed = 0
    for i in reversed(range(system.getNumForces())):
        f = system.getForce(i)
        if isinstance(
            f,
            (
                openmm.MonteCarloBarostat,
                openmm.MonteCarloAnisotropicBarostat,
                openmm.MonteCarloMembraneBarostat,
            ),
        ):
            system.removeForce(i)
            removed += 1
    return removed


def load_boresch(run_dir: str):
    """取生产实际用的那组 Boresch 参数，并归一化键名。

    优先 `checkpoints/boresch_equilibrium_committed.json`——那是「本腿后续 resume
    强制复用」的那一组，也就是生产采样真正用的。它的键已经是归一化的
    （`r0` / `thetaA0` / `kr`），而 `boresch_params.json` 带单位后缀
    （`r0_nm` / `thetaA0_rad` / `kr_kJ_mol_nm2`），直接喂给
    `LambdaDependentBoreschForce` 会 KeyError——归一化统一走仓库现成的
    `runabfe._sanitize_boresch_params`，不在这里另写一份映射表。
    """
    committed = os.path.join(run_dir, "checkpoints", "boresch_equilibrium_committed.json")
    if os.path.isfile(committed):
        with open(committed, encoding="utf-8") as fh:
            return runabfe._sanitize_boresch_params(json.load(fh)), committed
    fallback = os.path.join(run_dir, "boresch_params.json")
    if os.path.isfile(fallback):
        with open(fallback, encoding="utf-8") as fh:
            return runabfe._sanitize_boresch_params(json.load(fh)), fallback
    raise FileNotFoundError(f"{run_dir} 下找不到 Boresch 参数")


def build_arm(
    system_xml: str,
    ligand_indices: List[int],
    boresch_params: Optional[Dict],
    with_boresch: bool,
) -> openmm.System:
    """从同一份 XML 造一个臂的 System。两臂唯一的差别就是有没有 Boresch 力。"""
    from abfe_core import LambdaDependentBoreschForce, ensure_owned_system

    system = XmlSerializer.deserialize(system_xml)
    system = ensure_owned_system(system)
    _strip_barostats(system)

    if with_boresch:
        if not boresch_params:
            raise RuntimeError("ON 臂需要 Boresch 参数")
        rest = LambdaDependentBoreschForce(
            rec_idx=boresch_params["receptor_indices"],
            lig_idx=boresch_params["ligand_indices"],
            eq=boresch_params["equilibrium_values"],
            fc=boresch_params["force_constants"],
            fixed_lam=1.0,   # 与生产 ibs_engine.py:889 完全一致
            sign=1.0,
            use_pbc=True,
        )
        rest.setForceGroup(3)
        system.addForce(rest)

    # 生产去电荷 Hamiltonian：llfreeze L–L 冻结 + PME 电荷 offset。
    configure_pme_ligand_charge_offsets(system, ligand_indices, lambda_name=LAMBDA_NAME)
    return system


# ---------------------------------------------------------------------------
# 结构指标
# ---------------------------------------------------------------------------


def _min_image(delta: np.ndarray, box_diag: np.ndarray) -> np.ndarray:
    return delta - box_diag * np.round(delta / box_diag)


def gather_near(xyz_nm: np.ndarray, idx: np.ndarray, center: np.ndarray, box_diag: np.ndarray):
    """把 idx 这批原子各自搬到离 center 最近的周期镜像。

    DCD 是按整分子折叠过的，同一个口袋的不同残基完全可能落在不同镜像里，
    直接取平均会得到一个没有物理意义的质心。
    """
    return center + _min_image(xyz_nm[idx] - center, box_diag)


def pocket_frame(
    xyz_nm: np.ndarray, lig_idx: np.ndarray, pocket_idx: np.ndarray, box_diag: np.ndarray
):
    """返回 (口袋质心, 已归位到口袋附近的配体坐标)。"""
    lig_com_raw = xyz_nm[lig_idx].mean(axis=0)  # 配体整分子折叠，自身不会被撕开
    pocket_com = gather_near(xyz_nm, pocket_idx, lig_com_raw, box_diag).mean(axis=0)
    lig = gather_near(xyz_nm, lig_idx, pocket_com, box_diag)
    return pocket_com, lig


def structural_metrics(
    xyz_nm: np.ndarray,
    ref_lig_local: np.ndarray,
    ref_lig_heavy_local: np.ndarray,
    lig_idx: np.ndarray,
    heavy_mask: np.ndarray,
    pocket_idx: np.ndarray,
    box_diag: np.ndarray,
) -> Dict[str, float]:
    """配体质心位移 + 重原子 RMSD，全部在"相对口袋质心"的参照系里算。

    不做旋转对齐——口袋骨架在几 ns 内几乎不转，而我们要看的正是配体相对
    口袋的取向变化，旋转对齐反而会把它抹掉。
    """
    pocket_com, lig = pocket_frame(xyz_nm, lig_idx, pocket_idx, box_diag)
    lig_local = lig - pocket_com
    com_disp = float(np.linalg.norm(lig_local.mean(axis=0) - ref_lig_local.mean(axis=0)))
    rmsd = float(
        np.sqrt(np.mean(np.sum((lig_local[heavy_mask] - ref_lig_heavy_local) ** 2, axis=1)))
    )
    return {"com_displacement_nm": com_disp, "heavy_rmsd_nm": rmsd}


def count_ligand_protein_hbonds(
    xyz_nm: np.ndarray,
    donors: List,
    acceptors: np.ndarray,
    box_diag: np.ndarray,
    max_dist_nm: float = 0.25,
    min_angle_deg: float = 120.0,
) -> int:
    """几何氢键计数：H···A < 0.25 nm 且 D–H···A > 120°。

    只数跨越配体/蛋白边界的那些。自己实现而不用 mdtraj.baker_hubbard，
    是因为这里需要逐帧占据率而不是"出现频率超过阈值的三元组列表"，
    而且要显式带最小镜像。
    """
    if acceptors.size == 0 or not donors:
        return 0
    n = 0
    acc_xyz = xyz_nm[acceptors]
    for d_idx, h_idx in donors:
        h = xyz_nm[h_idx]
        d = xyz_nm[d_idx]
        delta = _min_image(acc_xyz - h, box_diag)
        dist = np.linalg.norm(delta, axis=1)
        close = np.nonzero(dist < max_dist_nm)[0]
        if close.size == 0:
            continue
        dh = _min_image(d - h, box_diag)
        dh_norm = np.linalg.norm(dh)
        if dh_norm < 1.0e-9:
            continue
        for c in close:
            if dist[c] < 1.0e-9:
                continue
            # D–H···A 角 = H 处 H→D 与 H→A 两射线的夹角。线性氢键时两者反向，
            # cos = −1 → 180°。第一版这里多写了一个负号（arccos(−cos)），
            # 完美氢键被判成 0° 从而全部落选，导致 2026-07-28 09:27 那轮
            # 六条轨迹的氢键数全是 0.00 —— 那是 bug，不是"没有氢键"。
            cos_a = float(np.dot(dh, delta[c]) / (dh_norm * dist[c]))
            angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            if angle > min_angle_deg:
                n += 1
    return n


def build_hbond_sets(topology: app.Topology, ligand_set: set):
    """返回 (跨界 donor 的 (D,H) 对, 跨界 acceptor 索引)。

    donor = 连着 H 的 N/O/S；acceptor = N/O/S。只保留配体↔蛋白跨界的组合，
    做法是分别建两套，再在计数时交叉配对。
    """
    heavy_syms = {"N", "O", "S"}
    bonded_h: Dict[int, List[int]] = {}
    for a1, a2 in topology.bonds():
        for heavy, h in ((a1, a2), (a2, a1)):
            if (
                heavy.element is not None
                and heavy.element.symbol in heavy_syms
                and h.element is not None
                and h.element.symbol == "H"
            ):
                bonded_h.setdefault(int(heavy.index), []).append(int(h.index))

    lig_donors, env_donors = [], []
    lig_acc, env_acc = [], []
    for atom in topology.atoms():
        idx = int(atom.index)
        if atom.element is None or atom.element.symbol not in heavy_syms:
            continue
        # 只把蛋白/配体算进来，水和离子排除——我们要的是定向氢键网络。
        res = atom.residue.name.upper()
        if res in {"HOH", "WAT", "TIP3", "NA", "NA+", "CL", "CL-", "SOD", "CLA"}:
            continue
        in_lig = idx in ligand_set
        (lig_acc if in_lig else env_acc).append(idx)
        for h in bonded_h.get(idx, []):
            (lig_donors if in_lig else env_donors).append((idx, h))
    return {
        "lig_donors": lig_donors,
        "env_donors": env_donors,
        "lig_acceptors": np.asarray(lig_acc, dtype=int),
        "env_acceptors": np.asarray(env_acc, dtype=int),
    }


# ---------------------------------------------------------------------------
# 单条轨迹
# ---------------------------------------------------------------------------


def run_replica(
    system: openmm.System,
    topology: app.Topology,
    positions,
    box_vectors,
    ligand_indices: List[int],
    hb: Dict,
    ref_lig_local: np.ndarray,
    ref_lig_heavy_local: np.ndarray,
    pocket_idx: np.ndarray,
    heavy_mask: np.ndarray,
    *,
    temperature_k: float,
    timestep_fs: float,
    equil_steps: int,
    production_steps: int,
    sample_interval: int,
    seed: int,
    platform_name: str,
    min_hbond_angle: float = 120.0,
) -> Dict:
    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(seed)
    platform = openmm.Platform.getPlatformByName(platform_name)
    sim = app.Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    sim.context.setPeriodicBoxVectors(*box_vectors)
    sim.context.setParameter(LAMBDA_NAME, 1.0)
    sim.minimizeEnergy(maxIterations=500)
    sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)
    sim.step(equil_steps)

    lig_idx = np.asarray(sorted(int(i) for i in ligand_indices), dtype=int)
    box_diag = np.array(
        [box_vectors[0][0].value_in_unit(unit.nanometer),
         box_vectors[1][1].value_in_unit(unit.nanometer),
         box_vectors[2][2].value_in_unit(unit.nanometer)],
        dtype=float,
    )

    dU, rmsd, com, hbonds = [], [], [], []
    n_samples = max(1, production_steps // sample_interval)
    for _ in range(n_samples):
        sim.step(sample_interval)
        state = sim.context.getState(getPositions=True, getEnergy=True)
        u1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        sim.context.setParameter(LAMBDA_NAME, 0.0)
        u0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        sim.context.setParameter(LAMBDA_NAME, 1.0)
        dU.append(u0 - u1)

        xyz = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        m = structural_metrics(
            xyz, ref_lig_local, ref_lig_heavy_local, lig_idx, heavy_mask, pocket_idx, box_diag
        )
        rmsd.append(m["heavy_rmsd_nm"])
        com.append(m["com_displacement_nm"])
        hbonds.append(
            count_ligand_protein_hbonds(
                xyz, hb["lig_donors"], hb["env_acceptors"], box_diag,
                min_angle_deg=min_hbond_angle,
            )
            + count_ligand_protein_hbonds(
                xyz, hb["env_donors"], hb["lig_acceptors"], box_diag,
                min_angle_deg=min_hbond_angle,
            )
        )

    del sim, integrator
    return {
        "seed": int(seed),
        "n_samples": len(dU),
        "mean_dU_kJ_mol": float(np.mean(dU)),
        "std_dU_kJ_mol": float(np.std(dU)),
        "mean_heavy_rmsd_nm": float(np.mean(rmsd)),
        "mean_com_displacement_nm": float(np.mean(com)),
        "mean_ligand_protein_hbonds": float(np.mean(hbonds)),
    }


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", default="output_lrc_fix", help="复合物缓存目录（只读）")
    ap.add_argument("--out-dir", default="boresch_ensemble_effect")
    ap.add_argument("--n-replicas", type=int, default=3)
    ap.add_argument("--equil-ps", type=float, default=200.0)
    ap.add_argument("--production-ns", type=float, default=2.0)
    ap.add_argument("--sample-interval-ps", type=float, default=10.0)
    ap.add_argument("--timestep-fs", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--seed0", type=int, default=20260728)
    ap.add_argument(
        "--start-from",
        choices=("rebalance", "pre-equil", "auto"),
        default="rebalance",
        help="起始构象：rebalance_traj.dcd 末帧（默认，与 stage1 REMD 起点一致）"
        " / 无约束预平衡末帧 / auto（有 rebalance 就用，没有就回退）",
    )
    ap.add_argument(
        "--min-hbond-angle",
        type=float,
        default=120.0,
        help="D–H···A 角下限（度）；180° 为线性氢键",
    )
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    log_fh = open(os.path.join(args.out_dir, "run.log"), "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    system, topology, positions, box_vectors, ligand_indices = runabfe.load_native_system(
        args.run_dir, phase="complex", prefer_equilibrated=True
    )
    # 🔑 起始构象必须是 stage1 REMD 真正的起点。
    # `prefer_equilibrated=True` 给的是 `pre_equilibration.dcd` 末帧——那是
    # **无约束**预平衡的结果；而生产的 λ=1 系综来自 rebalance（Boresch 全开、
    # 50000 步）之后的 stage1 REMD。用错起点会让 ON 臂锚不到 171.08。
    if args.start_from in ("rebalance", "auto"):
        traj_path = os.path.join(args.run_dir, "rebalance_traj.dcd")
        if os.path.isfile(traj_path) and os.path.getsize(traj_path) > 0:
            import mdtraj

            # mdtraj 的 DCD reader 不接受负 frame 索引（seek(-1) 直接 OSError），
            # 所以先问一下总帧数再按正索引取最后一帧。
            top_path = os.path.join(args.run_dir, "topology.cif")
            with mdtraj.formats.DCDTrajectoryFile(traj_path) as dcd:
                n_frames = len(dcd)
            if n_frames < 1:
                raise RuntimeError(f"{traj_path} 里没有帧")
            frame = mdtraj.load_frame(traj_path, n_frames - 1, top=top_path)
            log(f"  {traj_path} 共 {n_frames} 帧，取第 {n_frames - 1} 帧（末帧）")
            positions = (frame.xyz[0] * unit.nanometer)
            if frame.unitcell_vectors is not None:
                box_vectors = [
                    openmm.Vec3(*frame.unitcell_vectors[0][i]) * unit.nanometer for i in range(3)
                ]
            log(f"起始构象：{traj_path} 末帧（rebalance 之后，Boresch 全开 50000 步）")
        elif args.start_from == "rebalance":
            raise FileNotFoundError(f"要求从 rebalance 起步但找不到 {traj_path}")
        else:
            log("⚠️ 没有 rebalance_traj.dcd，回退到无约束预平衡末帧——ON 臂可能锚不到 171.08")
    else:
        log("⚠️ 按要求从无约束预平衡末帧起步——ON 臂可能锚不到 171.08")
    system_xml = XmlSerializer.serialize(system)
    boresch, boresch_src = load_boresch(args.run_dir)
    log(f"Boresch 参数来自 {boresch_src}")
    log(
        "  r0={r0:.4f} nm  thetaA0={thetaA0:.4f}  thetaB0={thetaB0:.4f} rad".format(
            **boresch["equilibrium_values"]
        )
    )
    log(
        "  kr={kr:.1f}  kthetaA={kthetaA:.1f}  kthetaB={kthetaB:.1f}".format(
            **boresch["force_constants"]
        )
    )

    ligand_set = set(int(i) for i in ligand_indices)
    lig_idx = np.asarray(sorted(ligand_set), dtype=int)
    atoms = list(topology.atoms())
    lig_heavy = np.asarray(
        [i for i in lig_idx if atoms[i].element is not None and atoms[i].element.symbol != "H"],
        dtype=int,
    )
    ref_xyz = np.asarray(positions.value_in_unit(unit.nanometer))
    ref_box_diag = np.array(
        [
            box_vectors[0][0].value_in_unit(unit.nanometer),
            box_vectors[1][1].value_in_unit(unit.nanometer),
            box_vectors[2][2].value_in_unit(unit.nanometer),
        ],
        dtype=float,
    )
    lig_com0 = ref_xyz[lig_idx].mean(axis=0)

    # 🔑 距离必须带最小镜像。`rebalance_traj.dcd` 是 OpenMM DCDReporter 写的、
    # 按分子折叠过的坐标——配体整体可能落在与口袋不同的周期镜像里，
    # 直接算欧氏距离会一个口袋原子都选不到（实测选出 0 个）。
    # 折叠是按整分子做的，所以配体自身不会被撕开，lig_com0 仍然有效。
    env_mask = np.array(
        [
            int(a.index) not in ligand_set
            and a.residue.name.upper()
            not in {"HOH", "WAT", "TIP3", "NA", "NA+", "CL", "CL-", "SOD", "CLA"}
            for a in atoms
        ]
    )
    env_idx = np.nonzero(env_mask)[0]
    d = np.linalg.norm(_min_image(ref_xyz[env_idx] - lig_com0, ref_box_diag), axis=1)
    pocket_idx = env_idx[d <= POCKET_CUTOFF_NM].astype(int)
    if pocket_idx.size < 10:
        raise RuntimeError(
            f"口袋原子只找到 {pocket_idx.size} 个（非水非离子的环境原子共 {env_idx.size} 个，"
            f"最近的距配体质心 {d.min():.3f} nm）；检查 POCKET_CUTOFF_NM 或起始构象"
        )
    hb = build_hbond_sets(topology, ligand_set)

    # 参照构象也要走同一套归位逻辑，否则 RMSD/位移的基准本身就是错的。
    heavy_mask = np.isin(lig_idx, lig_heavy)
    ref_pocket_com, ref_lig = pocket_frame(ref_xyz, lig_idx, pocket_idx, ref_box_diag)
    ref_lig_local = ref_lig - ref_pocket_com
    ref_lig_heavy_local = ref_lig_local[heavy_mask]

    log(f"配体 {lig_idx.size} 原子（重原子 {lig_heavy.size}），口袋 {pocket_idx.size} 原子")
    log(
        f"氢键候选：配体 donor {len(hb['lig_donors'])} / acceptor {hb['lig_acceptors'].size}；"
        f"环境 donor {len(hb['env_donors'])} / acceptor {hb['env_acceptors'].size}"
    )

    steps = lambda ps: int(round(ps * 1000.0 / args.timestep_fs))
    equil_steps = steps(args.equil_ps)
    production_steps = steps(args.production_ns * 1000.0)
    sample_interval = steps(args.sample_interval_ps)
    log(
        f"每条轨迹：平衡 {args.equil_ps} ps + 生产 {args.production_ns} ns，"
        f"每 {args.sample_interval_ps} ps 取样 → {production_steps // sample_interval} 帧"
    )

    results: Dict[str, List[Dict]] = {"boresch_on": [], "boresch_off": []}
    for arm, with_b in (("boresch_on", True), ("boresch_off", False)):
        arm_system = build_arm(system_xml, sorted(ligand_set), boresch, with_b)
        log(f"\n=== {arm} ===")
        for r in range(args.n_replicas):
            seed = args.seed0 + 1000 * (1 if with_b else 2) + r
            t0 = time.time()
            rep = run_replica(
                arm_system, topology, positions, box_vectors, sorted(ligand_set),
                hb, ref_lig_local, ref_lig_heavy_local, pocket_idx, heavy_mask,
                temperature_k=args.temperature,
                timestep_fs=args.timestep_fs,
                equil_steps=equil_steps,
                production_steps=production_steps,
                sample_interval=sample_interval,
                seed=seed,
                platform_name=args.platform,
                min_hbond_angle=args.min_hbond_angle,
            )
            rep["wall_seconds"] = round(time.time() - t0, 1)
            results[arm].append(rep)
            log(
                f"  rep{r} seed={seed}: ⟨ΔU⟩={rep['mean_dU_kJ_mol']:.2f} kJ/mol "
                f"(帧内 σ={rep['std_dU_kJ_mol']:.2f})  RMSD={rep['mean_heavy_rmsd_nm']*10:.2f} Å  "
                f"COM={rep['mean_com_displacement_nm']*10:.2f} Å  "
                f"氢键={rep['mean_ligand_protein_hbonds']:.2f}  [{rep['wall_seconds']/60:.1f} min]"
            )

    def agg(arm: str, key: str):
        vals = np.array([r[key] for r in results[arm]], dtype=float)
        se = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else float("nan")
        return float(np.mean(vals)), se

    log("\n" + "=" * 70)
    log("跨 replica 汇总（均值 ± 标准误）")
    log("=" * 70)
    summary = {}
    for key, label, scale, u in (
        ("mean_dU_kJ_mol", "⟨ΔU⟩ 配体-环境静电", 1.0, "kJ/mol"),
        ("mean_heavy_rmsd_nm", "配体重原子 RMSD", 10.0, "Å"),
        ("mean_com_displacement_nm", "配体质心位移", 10.0, "Å"),
        ("mean_ligand_protein_hbonds", "配体-蛋白氢键数", 1.0, "个"),
    ):
        on_m, on_s = agg("boresch_on", key)
        off_m, off_s = agg("boresch_off", key)
        summary[key] = {
            "boresch_on": {"mean": on_m, "sem": on_s},
            "boresch_off": {"mean": off_m, "sem": off_s},
            "off_minus_on": off_m - on_m,
        }
        log(
            f"  {label:22s}  ON {on_m*scale:8.2f} ± {on_s*scale:5.2f}   "
            f"OFF {off_m*scale:8.2f} ± {off_s*scale:5.2f}   "
            f"差 {(off_m-on_m)*scale:+8.2f} {u}"
        )

    du_on = summary["mean_dU_kJ_mol"]["boresch_on"]["mean"]
    du_off = summary["mean_dU_kJ_mol"]["boresch_off"]["mean"]
    log("")
    log(f"  对照锚点：生产 λ=1 口袋 {REFERENCE_POCKET_DU_KJ:.2f}，水中 {REFERENCE_WATER_DU_KJ:.2f} kJ/mol")
    log(f"  ON  臂 {du_on:.2f}（应与 171.08 同量级，否则体系/口径对不上，后面别信）")
    log(f"  OFF 臂 {du_off:.2f}")
    if du_off >= REFERENCE_WATER_DU_KJ:
        verdict = "关掉 Boresch 后口袋静电已越过水中值 → 限制就是原因，接着做 P1-17 的 A′→A 腿"
    elif du_off - du_on > 3.0:
        verdict = (
            f"关掉后口袋静电上升 {du_off - du_on:.2f} kJ/mol 但未越过 {REFERENCE_WATER_DU_KJ:.2f}"
            " → 限制是部分原因，还有别的"
        )
    else:
        verdict = "关掉也基本不动 → 限制不是原因，转去查 llfreeze 口径（P1-18 第 2 项）"
    log(f"\n  判读：{verdict}")

    out = {
        "run_dir": args.run_dir,
        "config": vars(args),
        "reference": {
            "pocket_dU_kJ_mol": REFERENCE_POCKET_DU_KJ,
            "water_dU_kJ_mol": REFERENCE_WATER_DU_KJ,
        },
        "per_replica": results,
        "summary": summary,
        "verdict": verdict,
    }
    out_path = os.path.join(args.out_dir, "boresch_ensemble_effect.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    log(f"\n结果已写入 {out_path}")
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

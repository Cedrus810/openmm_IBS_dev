#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABFE 计算命令行入口 (v3.0 - 完全重构版)
===========================================
核心设计：
  - 内置 GROMACS → OpenMM 原生缓存转换逻辑（无需 convert_to_native.py）
  - 所有模拟强制从 output 目录的缓存文件加载，避免 GROMACS 解析的副作用
  - 支持 JSON 配置文件，命令行参数优先级最高
  - 智能断点续传：自动检测各阶段状态
  - 修复多个已知 bug：配体索引提取、坐标类型转换、Boresch 参数清洗等
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import openmm
from openmm import app, unit, Vec3, XmlSerializer

# 确保只导入所需模块，避免循环依赖
# 删除 runabfe.py 中手写的 NumpyEncoder 类
# 修改导入语句：
from abfe_core import (
    ACESoftcorePotential, UnitFormatter, calculate_boresch_analytical_correction,
    calc_boresch_from_last_frame, GeometricRestraintEstimator, OrbBoreschEstimator,
    DEXPSurrogatePotential, LambdaDependentBoreschForce, ensure_owned_system,
    NumpyEncoder,  # ✅ 统一从 abfe_core 导入
)
from abfe_pipeline import ABFEPipeline, TraditionalABFEPipeline
from ibs_engine import solve_stage_integrated, generate_overlapping_windows # ✅ 保持从 ibs_engine 导入
from abfe_preoptimizer import DualLambdaPreOptimizer, build_aces_probe_system_dual_lambda

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("runabfe")

# ---------------------------------------------------------------------------
# 常量与预设
# ---------------------------------------------------------------------------
PRESET_CONFIGS = {
    "test": {
        "n_steps_per_window": 10000,
        "steps_per_update": 500,
        "stage1_n_states": 12,
        "stage2_n_states": 12,
    },
    "production": {
        "n_steps_per_window": 250000,
        "steps_per_update": 500,
        "stage1_n_states": 16,
        "stage2_n_states": 16,
    },
    "high_accuracy": {
        "n_steps_per_window": 500000,
        "steps_per_update": 500,
        "stage1_n_states": 24,
        "stage2_n_states": 24,
    },
}


class NumpyEncoder(json.JSONEncoder):
    """JSON 序列化支持 numpy 类型"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# 工具函数：GROMACS 力场路径探测
# ---------------------------------------------------------------------------
def find_gmx_include_dir(user_path: Optional[str] = None) -> Optional[str]:
    """智能查找 GROMACS 力场 include 目录"""
    if user_path and os.path.exists(user_path):
        return user_path

    env_path = os.environ.get("GMXDATA")
    if env_path and os.path.exists(env_path):
        top_path = os.path.join(env_path, "top")
        return top_path if os.path.exists(top_path) else env_path

    try:
        gmx_bin = (
            subprocess.check_output(["which", "gmx"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        gmx_root = os.path.abspath(os.path.join(os.path.dirname(gmx_bin), ".."))
        ff_path = os.path.join(gmx_root, "share/gromacs/top")
        if os.path.exists(ff_path):
            return ff_path
    except Exception:
        pass

    for path in [
        "/home/ruigengji/gmx25.1/share/gromacs/top",  
        "/home/ruigengji/gmx26.1_avx2/share/gromacs/top",      
        "/usr/local/gromacs/share/gromacs/top",
        "/usr/share/gromacs/top",
        "/opt/gromacs/share/gromacs/top",
    ]:
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# 状态检测函数
# ---------------------------------------------------------------------------
def system_cache_exists(output_dir: str) -> bool:
    """检查原生 OpenMM 缓存是否存在"""
    xml = os.path.join(output_dir, "system_native.xml")
    idx = os.path.join(output_dir, "ligand_indices.json")
    return os.path.isfile(xml) and os.path.isfile(idx)


def solvent_cache_exists(output_dir: str) -> bool:
    """检查溶剂腿原生缓存是否存在。"""
    xml = os.path.join(output_dir, "system_solvent.xml")
    idx = os.path.join(output_dir, "ligand_indices_solvent.json")
    top = os.path.join(output_dir, "topology_solvent.cif")
    return os.path.isfile(xml) and os.path.isfile(idx) and os.path.isfile(top)


def equilibrium_is_done(output_dir: str) -> bool:
    """判断预平衡是否完成（存在轨迹和 checkpoint）"""
    traj = os.path.join(output_dir, "pre_equilibration.dcd")
    chk = os.path.join(output_dir, "checkpoints", "pre_equil.chk")
    return os.path.isfile(traj) and os.path.getsize(traj) > 10000 and os.path.isfile(chk)


def boresch_params_ready(output_dir: str, source: str) -> bool:
    """检查 Boresch 参数文件是否已生成"""
    if source == "auto":
        return os.path.isfile(os.path.join(output_dir, "boresch_auto.json"))
    if source == "simple":
        return os.path.isfile(os.path.join(output_dir, "boresch_simple.json"))
    if source == "fluctuation":
        return os.path.isfile(os.path.join(output_dir, "boresch_fluctuation.json"))
    return False


def _cache_paths(output_dir: str, phase: str = "complex") -> Dict[str, str]:
    if phase == "solvent":
        return {
            "xml": os.path.join(output_dir, "system_solvent.xml"),
            "idx": os.path.join(output_dir, "ligand_indices_solvent.json"),
            "box": os.path.join(output_dir, "box_vectors_solvent.npy"),
            "top": os.path.join(output_dir, "topology_solvent.cif"),
            "runtime_dir": os.path.join(output_dir, "solvent_leg"),
        }
    return {
        "xml": os.path.join(output_dir, "system_native.xml"),
        "idx": os.path.join(output_dir, "ligand_indices.json"),
        "box": os.path.join(output_dir, "box_vectors.npy"),
        "top": os.path.join(output_dir, "topology.cif"),
        "runtime_dir": output_dir,
    }


def _get_residue_name_by_atom_index(topology: app.Topology, atom_index: int) -> str:
    atoms = list(topology.atoms())
    if atom_index < 0 or atom_index >= len(atoms):
        raise IndexError(f"原子索引越界: {atom_index}")
    return atoms[atom_index].residue.name


def resolve_ligand_ffxml(output_dir: str, ligand_resname: str, explicit_path: Optional[str] = None) -> Optional[str]:
    """定位配体力场 XML/FFXML，优先用户显式指定，其次尝试 output 下常见命名。"""
    candidates: List[Optional[str]] = [
        explicit_path,
        os.path.join(output_dir, "ligand.xml"),
        os.path.join(output_dir, "ligand.ffxml"),
        os.path.join(output_dir, f"{ligand_resname}.xml"),
        os.path.join(output_dir, f"{ligand_resname}.ffxml"),
        os.path.join(output_dir, "ffxml", "ligand.xml"),
        os.path.join(output_dir, "ffxml", "ligand.ffxml"),
        os.path.join(output_dir, "ffxml", f"{ligand_resname}.xml"),
        os.path.join(output_dir, "ffxml", f"{ligand_resname}.ffxml"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _build_stable_ligand_atom_names(gmx_top: app.GromacsTopFile, ligand_resname: str) -> Dict[int, str]:
    """基于 .top 的原始原子顺序生成稳定且唯一的配体原子名映射。"""
    lig_atoms = [a for a in gmx_top.topology.atoms() if a.residue.name == ligand_resname]
    if not lig_atoms:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    raw_names = [atom.name.strip() or atom.element.symbol for atom in lig_atoms]
    if len(set(raw_names)) == len(raw_names):
        return {atom.index: raw for atom, raw in zip(lig_atoms, raw_names)}

    elem_counter: Dict[str, int] = {}
    stable_names: Dict[int, str] = {}
    for atom in lig_atoms:
        elem = atom.element.symbol if atom.element is not None else "X"
        elem_counter[elem] = elem_counter.get(elem, 0) + 1
        stable_names[atom.index] = f"{elem}{elem_counter[elem]}"
    return stable_names


def _resolve_mdtraj_topology_input(output_dir: str, gro_file: Optional[str], top_file: Optional[str]) -> str:
    """为 mdtraj.load 解析最稳妥的拓扑输入，优先缓存的 mmCIF。"""
    cif_path = os.path.join(output_dir, "topology.cif")
    if os.path.isfile(cif_path):
        return cif_path
    if gro_file and os.path.isfile(gro_file):
        return gro_file
    if top_file and os.path.isfile(top_file):
        return top_file
    raise FileNotFoundError(
        "无法为轨迹加载解析拓扑：未找到 topology.cif 缓存，也未提供有效的 --gro/--top。"
    )


def generate_ligand_xml_from_top(
    top_file: str,
    ligand_resname: str,
    output_dir: str,
    gmx_include_dir: Optional[str] = None,
) -> str:
    """从 GROMACS .top 中抽取配体参数，生成可被 OpenMM ForceField 加载的 ligand_only.xml。"""
    log.info("🔹 从 GROMACS 拓扑抽取配体参数: 生成 ligand_only.xml")
    top = app.GromacsTopFile(top_file, includeDir=gmx_include_dir)
    lig_res = next((r for r in top.topology.residues() if r.name == ligand_resname), None)
    if lig_res is None:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    lig_idx = [a.index for a in lig_res.atoms()]
    stable_names = _build_stable_ligand_atom_names(top, ligand_resname)
    global_to_local = {g: l for l, g in enumerate(lig_idx)}
    atoms = list(top.topology.atoms())
    lig_set = set(lig_idx)
    bond_neighbors: Dict[int, set] = {idx: set() for idx in lig_idx}
    for bond in top.topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        if i in lig_set and j in lig_set:
            bond_neighbors[i].add(j)
            bond_neighbors[j].add(i)

    # 不再依赖 GromacsTopFile 的私有字段；直接用 createSystem() 后的真实 OpenMM 力参数。
    extracted_system = top.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    nb_force = next(
        (f for f in extracted_system.getForces() if isinstance(f, openmm.NonbondedForce)),
        None,
    )
    bond_force = next(
        (f for f in extracted_system.getForces() if isinstance(f, openmm.HarmonicBondForce)),
        None,
    )
    angle_force = next(
        (f for f in extracted_system.getForces() if isinstance(f, openmm.HarmonicAngleForce)),
        None,
    )
    torsion_force = next(
        (f for f in extracted_system.getForces() if isinstance(f, openmm.PeriodicTorsionForce)),
        None,
    )
    if nb_force is None:
        raise RuntimeError("GROMACS 拓扑构建出的 System 中未找到 NonbondedForce，无法抽取配体参数")

    def classify_torsion(p1: int, p2: int, p3: int, p4: int) -> Tuple[str, Tuple[int, int, int, int]]:
        atoms4 = (p1, p2, p3, p4)
        if (
            p2 in bond_neighbors.get(p1, ())
            and p3 in bond_neighbors.get(p2, ())
            and p4 in bond_neighbors.get(p3, ())
        ):
            return "Proper", atoms4

        for center in atoms4:
            outer = [x for x in atoms4 if x != center]
            if all(x in bond_neighbors.get(center, ()) for x in outer):
                return "Improper", (center, outer[0], outer[1], outer[2])

        return "Proper", atoms4

    unique_names: Dict[int, str] = {int(i): stable_names[int(i)] for i in lig_idx}

    xml_lines = ["<ForceField>", "  <AtomTypes>"]
    for i in lig_idx:
        elem = atoms[i].element
        mass = elem.mass.value_in_unit(unit.dalton) if hasattr(elem, "mass") else 12.0
        xml_lines.append(f'    <Type name="T{i}" class="T{i}" element="{elem.symbol}" mass="{mass:.6f}"/>')
    xml_lines += ["  </AtomTypes>", "  <Residues>", f'    <Residue name="{ligand_resname}">']
    for i in lig_idx:
        charge = nb_force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        xml_lines.append(f'      <Atom name="{unique_names[i]}" type="T{i}" charge="{charge:.8f}"/>')

    for bond in top.topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        if i in global_to_local and j in global_to_local:
            xml_lines.append(
                f'      <Bond from="{global_to_local[i]}" to="{global_to_local[j]}"/>'
            )
    xml_lines += ["    </Residue>", "  </Residues>"]

    xml_lines.append("  <HarmonicBondForce>")
    if bond_force is not None:
        for bidx in range(bond_force.getNumBonds()):
            p1, p2, length, k = bond_force.getBondParameters(bidx)
            p1 = int(p1)
            p2 = int(p2)
            if p1 in unique_names and p2 in unique_names:
                xml_lines.append(
                    f'    <Bond class1="T{p1}" class2="T{p2}" '
                    f'length="{length.value_in_unit(unit.nanometer):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2):.8f}"/>'
                )
    xml_lines.append("  </HarmonicBondForce>")

    xml_lines.append("  <HarmonicAngleForce>")
    if angle_force is not None:
        for aidx in range(angle_force.getNumAngles()):
            p1, p2, p3, angle, k = angle_force.getAngleParameters(aidx)
            p1, p2, p3 = int(p1), int(p2), int(p3)
            if all(x in unique_names for x in (p1, p2, p3)):
                xml_lines.append(
                    f'    <Angle class1="T{p1}" class2="T{p2}" class3="T{p3}" '
                    f'angle="{angle.value_in_unit(unit.radian):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2):.8f}"/>'
                )
    xml_lines.append("  </HarmonicAngleForce>")

    xml_lines.append("  <PeriodicTorsionForce>")
    if torsion_force is not None:
        for tidx in range(torsion_force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(tidx)
            p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)
            if all(x in unique_names for x in (p1, p2, p3, p4)):
                torsion_tag, ordered = classify_torsion(p1, p2, p3, p4)
                t1, t2, t3, t4 = ordered
                xml_lines.append(
                    f'    <{torsion_tag} class1="T{t1}" class2="T{t2}" class3="T{t3}" class4="T{t4}" '
                    f'periodicity="{int(periodicity)}" '
                    f'phase="{phase.value_in_unit(unit.radian):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole):.8f}"/>'
                )
    xml_lines.append("  </PeriodicTorsionForce>")

    xml_lines.append('  <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">')
    for i in lig_idx:
        charge, sigma, epsilon = nb_force.getParticleParameters(i)
        xml_lines.append(
            f'    <Atom type="T{i}" '
            f'charge="{charge.value_in_unit(unit.elementary_charge):.8f}" '
            f'sigma="{sigma.value_in_unit(unit.nanometer):.8f}" '
            f'epsilon="{epsilon.value_in_unit(unit.kilojoule_per_mole):.8f}"/>'
        )
    xml_lines.append("  </NonbondedForce>")
    xml_lines.append("</ForceField>")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ligand_only.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))
    log.info("✅ 已生成配体 XML: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 原生系统缓存读写
# ---------------------------------------------------------------------------
def diagnose_14_scaling(system: openmm.System) -> None:
    """打印 NonbondedForce 的 1-4 缩放因子，验证 GROMACS 继承正确性。"""
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb = force
            break
    else:
        log.warning("  ⚠️ 未找到 NonbondedForce，无法诊断 1-4 缩放")
        return

    # 收集所有异常对的参数
    charge_prods = []
    lj_scales = []
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sigma, eps = nb.getExceptionParameters(i)
        # 获取两个原子的原始参数
        q1, sig1, eps1 = nb.getParticleParameters(int(p1))
        q2, sig2, eps2 = nb.getParticleParameters(int(p2))
        q_prod_raw = (q1 * q2).value_in_unit(unit.elementary_charge**2)
        if abs(q_prod_raw) > 1e-10:
            charge_prods.append(cp.value_in_unit(unit.elementary_charge**2) / q_prod_raw)
        # LJ 缩放近似为 epsilon_14 / (sqrt(eps1*eps2))
        eps_cross = np.sqrt(eps1.value_in_unit(unit.kilojoule_per_mole) * eps2.value_in_unit(unit.kilojoule_per_mole))
        if eps_cross > 1e-10:
            lj_scales.append(eps.value_in_unit(unit.kilojoule_per_mole) / eps_cross)

    fudgeQQ = np.mean(charge_prods) if charge_prods else 0.0
    fudgeLJ = np.mean(lj_scales) if lj_scales else 0.0
    log.info("  🔍 1-4 缩放诊断: fudgeQQ=%.4f (应 ~0.8333), fudgeLJ=%.4f (应 ~0.5)", fudgeQQ, fudgeLJ)

def save_native_system(output_dir, system, topology, ligand_indices, positions, box_vectors):
    """持久化原生 System XML 与辅助数据至 output_dir（内置 convert 功能）"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. System XML (强制 Python 所有权)
    native_sys = ensure_owned_system(system)
    xml_path = os.path.join(output_dir, "system_native.xml")
    with open(xml_path, "w") as f:
        f.write(XmlSerializer.serialize(native_sys))
    log.info("  [缓存] System XML 已保存: %s", xml_path)

    # 2. Ligand indices
    lig_path = os.path.join(output_dir, "ligand_indices.json")
    with open(lig_path, "w") as f:
        json.dump({"ligand_indices": [int(i) for i in ligand_indices]}, f, indent=2)
    log.info("  [缓存] 配体索引已保存: %s", lig_path)

    # 3. Box vectors
    if box_vectors is not None:
        box_path = os.path.join(output_dir, "box_vectors.npy")
        if hasattr(box_vectors, 'value_in_unit'):
            box_nm = box_vectors.value_in_unit(unit.nanometer)
        else:
            box_nm = box_vectors
        np.save(box_path, np.asarray(box_nm, dtype=np.float64))
        log.info("  [缓存] 盒子向量已保存: %s", box_path)

    # 4. Topology 缓存 (🔑 改用 mmCIF 格式，彻底解决 PDB >99999 原子序列号溢出与截断问题)
    top_path = os.path.join(output_dir, "topology.cif")
    try:
        app.PDBxFile.writeFile(topology, positions, top_path)
        log.info("  [缓存] 参考拓扑已保存 (mmCIF): %s", top_path)
    except Exception as e:
        log.warning("  [缓存] 拓扑 mmCIF 保存失败: %s", e)

# ================= runabfe.py =================
# 完整替换 build_and_cache_solvent_leg 函数
def build_and_cache_solvent_leg(
    output_dir,
    topology,          # 注意：这个参数我们现在彻底不用了，避免 mmCIF 污染
    positions,
    ligand_indices,
    ligand_resname,
    ligand_ffxml: Optional[str] = None,
    top_file: Optional[str] = None,
    gmx_include_dir: Optional[str] = None,
):
    """
    🔑 终极纯净版：彻底抛弃 mmCIF，直接从原始 .top 提取配体并自动加水。
    """
    log.info("💧 正在构建溶剂相 (配体腿) 系统并生成缓存...")
    from openmm.app import Modeller, ForceField
    os.makedirs(output_dir, exist_ok=True)

    # 1. 提取配体坐标，计算回旋半径以决定水盒子大小
    if hasattr(positions, "value_in_unit"):
        pos_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    else:
        pos_nm = np.asarray(positions, dtype=np.float64)
    
    lig_coords = pos_nm[ligand_indices]
    center = lig_coords.mean(axis=0)
    max_r = np.max(np.linalg.norm(lig_coords - center, axis=1))
    box_size = max(max_r + 1.5, 3.5)  # nm (至少 3.5nm 盒子)

    # 2. 🔑 降维打击：彻底抛弃 mmCIF，直接从原始 .top 读取 100% 纯净的配体拓扑
    if not top_file or not os.path.isfile(top_file):
        log.error("❌ 构建溶剂相必须提供原始复合物 .top 文件以提取纯净配体键连接")
        return False
        
    log.info("  🧬 彻底抛弃 mmCIF，从原始 .top 重建 100% 纯净配体拓扑...")
    gmx_top = app.GromacsTopFile(top_file, includeDir=gmx_include_dir)
    stable_names = _build_stable_ligand_atom_names(gmx_top, ligand_resname)
    
    # 获取 .top 中的配体原子索引
    top_lig_indices = [a.index for a in gmx_top.topology.atoms() if a.residue.name == ligand_resname]
    if not top_lig_indices:
        log.error(f"❌ 在 .top 文件中未找到配体残基 {ligand_resname}")
        return False
        
    # 使用 .top 的绝对正确拓扑 + 全局坐标创建 Modeller
    modeller = Modeller(gmx_top.topology, positions)
    
    # 删掉所有环境原子（蛋白、水、离子），只留配体
    atoms_to_delete = [a for a in modeller.topology.atoms() if a.index not in top_lig_indices]
    modeller.delete(atoms_to_delete)
    log.info("  ✅ 纯净配体拓扑构建完成 (键连接 100% 忠于 .top，无假键)")

    # 3. 用 .top 的稳定名字映射重命名，确保拓扑与 XML 完全同序对齐
    lig_atoms_in_mod = [a for a in modeller.topology.atoms() if a.residue.name == ligand_resname]
    if len(lig_atoms_in_mod) != len(top_lig_indices):
        log.error("❌ 配体原子数不匹配：Modeller=%d, Topology=%d", len(lig_atoms_in_mod), len(top_lig_indices))
        return False
    for atom, top_idx in zip(lig_atoms_in_mod, top_lig_indices):
        atom.name = stable_names[int(top_idx)]
    log.info("  ✅ 配体原子名已按 .top 稳定映射重命名并对齐 XML")

    # 4. 复用/生成配体 XML + TIP3P 水
    ffxml_path = resolve_ligand_ffxml(output_dir, ligand_resname, explicit_path=ligand_ffxml)
    if ffxml_path is None:
        ffxml_path = generate_ligand_xml_from_top(
            top_file=top_file,
            ligand_resname=ligand_resname,
            output_dir=output_dir,
            gmx_include_dir=gmx_include_dir,
        )
    if ffxml_path is None:
        log.error("❌ 溶剂相构建失败：未找到或无法生成配体力场 XML")
        return False

    try:
        log.info("  🔍 使用 Amber14 + TIP3P + 配体 XML 构建溶剂相: %s", ffxml_path)
        ff = ForceField("amber14-all.xml", "amber14/tip3p.xml", ffxml_path)
        
        # 🔑 让 OpenMM 自动根据配体大小铺设水盒子
        modeller.addSolvent(ff, boxSize=Vec3(box_size, box_size, box_size) * unit.nanometer)
        
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
    except Exception as e:
        log.error("❌ 溶剂相构建失败 (配体 XML 未正确接入或模板不匹配): %s", e)
        log.error("💡 当前使用的配体 XML: %s", ffxml_path)
        return False

    # 5. 获取溶剂相中的新配体索引并保存缓存
    new_lig_indices = [atom.index for atom in modeller.topology.atoms() if atom.residue.name == ligand_resname]
    
    sol_xml = os.path.join(output_dir, "system_solvent.xml")
    sol_cif = os.path.join(output_dir, "topology_solvent.cif")
    sol_idx = os.path.join(output_dir, "ligand_indices_solvent.json")
    sol_box = os.path.join(output_dir, "box_vectors_solvent.npy")
    
    with open(sol_xml, "w") as f:
        f.write(XmlSerializer.serialize(ensure_owned_system(system)))
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, sol_cif)
    with open(sol_idx, "w") as f:
        json.dump({"ligand_indices": new_lig_indices}, f)
    
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    if box_vecs:
        np.save(sol_box, np.array([v.value_in_unit(unit.nanometer) for v in box_vecs]))
        
    log.info("✅ 溶剂相缓存已保存 (盒子大小: %.2f nm, 原子数: %d)", box_size, system.getNumParticles())
    return True

def load_native_system(
    output_dir,
    gro_file=None,
    top_file=None,
    gmx_include_dir=None,
    phase="complex",
    prefer_equilibrated: bool = True,
):
    """从 output_dir 的缓存文件加载系统（跳过 GROMACS 解析）"""
    paths = _cache_paths(output_dir, phase=phase)
    xml_path = paths["xml"]
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Native XML 缓存不存在: {xml_path}")
    log.info("♻️ 从原生缓存加载系统 (%s): %s", phase, output_dir)

    # 1. System
    with open(xml_path, "r") as f:
        xml_str = f.read()
    system = ensure_owned_system(XmlSerializer.deserialize(xml_str))
    log.info("  ✓ System 恢复 | 原子数: %d", system.getNumParticles())

    # 2. Ligand indices
    lig_path = paths["idx"]
    with open(lig_path, "r") as f:
        lig_data = json.load(f)
    ligand_indices = lig_data["ligand_indices"]
    log.info("  ✓ 配体索引恢复: %d 原子", len(ligand_indices))

    # 3. Box vectors
    box_path = paths["box"]
    box_vectors = None
    if os.path.exists(box_path):
        box_nm = np.load(box_path)
        box_vectors = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in box_nm] * unit.nanometer
        log.info("  ✓ 盒子向量恢复")
        try:
            system.setDefaultPeriodicBoxVectors(*box_vectors)
        except Exception as e:
            log.warning("  ⚠️ 设置默认盒子失败: %s", e)

    # 4. Topology 恢复 (🔑 优先 mmCIF 缓存，降级至原始 TOP 文件)
    top_path = paths["top"]
    topology = None
    cif = None
    if os.path.exists(top_path):
        try:
            cif = app.PDBxFile(top_path)
            topology = cif.topology
            n_cif = topology.getNumAtoms()
            n_sys = system.getNumParticles()
            if n_cif == n_sys:
                log.info("  ✓ 拓扑从 mmCIF 缓存恢复 (原子数校验通过: %d)", n_cif)
            else:
                log.warning("  ⚠️ mmCIF 拓扑原子数 (%d) 与 System (%d) 不匹配，已丢弃缓存", n_cif, n_sys)
                topology = None
        except Exception as e:
            log.warning("  ⚠️ mmCIF 拓扑加载失败: %s", e)

    # 降级方案：从原始 TOP 文件重建拓扑
    if topology is None and phase == "complex" and top_file and os.path.exists(top_file):
        try:
            inc_dir = gmx_include_dir or find_gmx_include_dir()
            top = app.GromacsTopFile(top_file, includeDir=inc_dir)
            topology = top.topology
            log.info("  ✓ 拓扑从原始 TOP 文件重建")
        except Exception as e:
            log.warning("  ⚠️ TOP 拓扑重建失败: %s", e)

    if topology is None:
        raise RuntimeError(
            "无法恢复拓扑：mmCIF 缓存损坏且无有效 TOP 文件。\n"
            "   💡 解决方案：请确保命令中包含 --top 参数，或删除 output 目录后重新运行。"
        )

    # 5. Positions (保持不变)
    positions = None
    equil_dcd = os.path.join(paths["runtime_dir"], "pre_equilibration.dcd")
    if prefer_equilibrated and os.path.exists(equil_dcd) and os.path.getsize(equil_dcd) > 212:
        try:
            import mdtraj as md
            md_top = md.Topology.from_openmm(topology)
            traj = md.load(equil_dcd, top=md_top)
            if len(traj) > 0:
                positions = traj.xyz[-1] * unit.nanometer
                if len(traj.unitcell_vectors) > 0:
                    box_vectors = traj.unitcell_vectors[-1] * unit.nanometer
                    system.setDefaultPeriodicBoxVectors(*box_vectors)
                log.info("  ✓ 坐标从预平衡 DCD 最后一帧恢复")
        except Exception as e:
            log.warning("  ⚠️ DCD 坐标加载失败: %s", e)

    if positions is None and cif is not None:
        try:
            positions = cif.positions
            log.info("  ✓ 坐标从 mmCIF 缓存恢复")
        except Exception as e:
            log.warning("  ⚠️ mmCIF 坐标恢复失败: %s", e)

    if positions is None and phase == "complex" and gro_file and os.path.exists(gro_file):
        gro = app.GromacsGroFile(gro_file)
        positions = gro.positions
        log.info("  ⚠️ 坐标回退到 GRO 初始值")

    if positions is None:
        raise RuntimeError("无法恢复坐标，请检查输入文件或预平衡结果")

    log.info("✅ 原生缓存加载完成")
    return system, topology, positions, box_vectors, ligand_indices

# ---------------------------------------------------------------------------
# GROMACS 系统构建（仅首次使用，之后都会被缓存替代）
# ---------------------------------------------------------------------------
def build_system_from_gromacs(
    gro_file: str,
    top_file: str,
    ligand_resname: str,
    gmx_include_dir: Optional[str] = None,
) -> Tuple[openmm.System, app.Topology, list, list, List[int]]:
    """从 GROMACS 文件构建 OpenMM 系统（仅用于首次创建缓存）"""
    if not os.path.exists(gro_file):
        raise FileNotFoundError(f"GRO 文件不存在: {gro_file}")
    if not os.path.exists(top_file):
        raise FileNotFoundError(f"TOP 文件不存在: {top_file}")

    log.info("📥 读取 GROMACS 文件: %s | %s", gro_file, top_file)

    # 解析坐标与盒子
    gro = app.GromacsGroFile(gro_file)
    positions = gro.positions
    box_vectors = gro.getPeriodicBoxVectors()

    # 解析拓扑
    include_dir = gmx_include_dir or find_gmx_include_dir()
    if include_dir:
        log.info("  🔍 使用 GROMACS include 目录: %s", include_dir)
    else:
        log.warning("  ⚠️ 未找到 include 目录，若拓扑含 #include 可能失败")

    try:
        top = app.GromacsTopFile(
            top_file,
            periodicBoxVectors=box_vectors,
            includeDir=include_dir,
        )
    except ValueError as e:
        if "Could not locate #include file" in str(e):
            log.error("拓扑解析失败: %s", e)
            log.error("请通过 --gmx-path 指定包含 .ff 文件夹的父目录")
            sys.exit(1)
        raise

    omm_top = top.topology

    # 创建 System（保留自定义电荷与力常数）
    log.info("  ⚙️ 构建 OpenMM System (保留自定义 RESP/Hessian 参数)...")
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    # 提取配体索引（修复：正确使用 atom.index）
    all_atoms = list(omm_top.atoms())
    ligand_info = [(atom.index, atom.residue.index) for atom in all_atoms if atom.residue.name == ligand_resname]
    ligand_indices = [idx for idx, _ in ligand_info]

    if not ligand_indices:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    # 防御性处理同名残基
    res_indices = set(res_idx for _, res_idx in ligand_info)
    if len(res_indices) > 1:
        target_res = min(res_indices)
        ligand_indices = [idx for idx, res_idx in ligand_info if res_idx == target_res]
        log.warning("⚠️ 发现多个同名残基，仅保留残基索引 %d", target_res)

    log.info("  ✅ 系统构建完成 | 原子数: %d | 配体: %d 原子", system.getNumParticles(), len(ligand_indices))

    return system, omm_top, positions, box_vectors, ligand_indices

def center_system_rigidly(
    positions: unit.Quantity,
    box_vectors: unit.Quantity,
    ligand_indices: List[int],
) -> Tuple[unit.Quantity, unit.Quantity]:
    """
    🔑 绝对刚性居中协议 (No-Wrap Guarantee)
    1. 彻底禁用原子级 PBC Wrap，杜绝跨盒分子被撕裂。
    2. 仅计算配体质心，将整个系统作为刚体平移至盒子几何中心。
    3. 配合 CMMotionRemover，确保生产采样期间体系死死钉在盒子中央。
    """
    import numpy as np
    from openmm import Vec3
    
    pos = np.array(positions.value_in_unit(unit.nanometer))
    box = np.array([v.value_in_unit(unit.nanometer) for v in box_vectors])
    
    # 计算配体质心与盒子中心
    lig_com = np.mean(pos[ligand_indices], axis=0)
    box_center = 0.5 * np.sum(box, axis=0)
    
    # 整体刚性平移
    shift = box_center - lig_com
    pos += shift
    
    log.info("  🔒 体系已刚性锁定至盒子中心 (No-Wrap, 仅整体平移)")
    new_pos = [Vec3(float(x), float(y), float(z)) for x, y, z in pos] * unit.nanometer
    return new_pos, box_vectors
# ---------------------------------------------------------------------------
# 配置管理
# ---------------------------------------------------------------------------
def _load_config(config_path: str) -> dict:
    """加载配置文件，支持 .json / .yaml / .yml 格式"""
    ext = os.path.splitext(config_path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "读取 .yaml 配置文件需要 pyyaml，请运行: pip install pyyaml\n"
                "或使用 .json 格式的配置文件。"
            )
    else:
        with open(config_path) as f:
            return json.load(f)


class RunConfig:
    """统一运行时配置，优先级：命令行 > 配置文件 > 预设"""

    def __init__(self, args: argparse.Namespace):
        argv = sys.argv[1:]

        def _flag_present(*flags: str) -> bool:
            for flag in flags:
                if flag in argv:
                    return True
                prefix = f"{flag}="
                if any(token.startswith(prefix) for token in argv):
                    return True
            return False

        # 1. 载入预设
        preset = PRESET_CONFIGS.get(args.preset, PRESET_CONFIGS["production"]).copy()

        # 2. 合并配置文件（支持 .json / .yaml / .yml）
        if hasattr(args, "config") and args.config and os.path.exists(args.config):
            file_conf = _load_config(args.config)
            for k, v in file_conf.items():
                if k.startswith("_"):
                    continue
                preset[k] = v
            log.info("📄 已合并配置文件: %s", args.config)

        # 3. 仅当命令行显式提供参数时才覆盖配置文件，避免 parser 默认值反向污染配置。
        if _flag_present("--resume"):
            preset["resume"] = bool(args.resume)
        if _flag_present("--reset"):
            preset["reset"] = bool(args.reset)
        if _flag_present("--n-steps-per-window"):
            preset["n_steps_per_window"] = args.n_steps_per_window
        if _flag_present("--steps-per-update"):
            preset["steps_per_update"] = args.steps_per_update
        if _flag_present("--n-states-per-stage"):
            preset["stage1_n_states"] = args.n_states_per_stage
            preset["stage2_n_states"] = args.n_states_per_stage
        if _flag_present("--temperature"):
            preset["temperature"] = args.temperature
        if _flag_present("--platform"):
            preset["platform"] = args.platform
        if _flag_present("--output"):
            preset["output"] = args.output
        if _flag_present("--mode"):
            preset["mode"] = args.mode
        if _flag_present("--decoupling"):
            preset["decoupling"] = args.decoupling
        if _flag_present("--potential"):
            preset["potential"] = args.potential
        if _flag_present("--dexp-params"):
            preset["dexp_params"] = args.dexp_params
        if _flag_present("--gro"):
            preset["gro"] = args.gro
        if _flag_present("--top"):
            preset["top"] = args.top
        if _flag_present("--ligand"):
            preset["ligand"] = args.ligand
        if _flag_present("--ligand-xml"):
            preset["ligand_xml"] = args.ligand_xml
        if _flag_present("--gmx-path"):
            preset["gmx_path"] = args.gmx_path
        if _flag_present("--torsion-params"):
            preset["torsion_params"] = args.torsion_params
        if _flag_present("--boresch", "--no-boresch"):
            preset["boresch"] = args.boresch
        if _flag_present("--boresch-source"):
            preset["boresch_source"] = args.boresch_source
        if _flag_present("--boresch-anchors"):
            preset["boresch_anchors"] = args.boresch_anchors
        if _flag_present("--boresch-orb"):
            preset["boresch_orb"] = args.boresch_orb
        if _flag_present("--boresch-batch"):
            preset["boresch_batch"] = args.boresch_batch
        if _flag_present("--boresch-select"):
            preset["boresch_select"] = args.boresch_select
        if _flag_present("--enable-early-stop"):
            preset["enable_early_stop"] = bool(args.enable_early_stop)
        if _flag_present("--disable-warmup"):
            preset["enable_gradual_warmup"] = False
        elif _flag_present("--enable-gradual-warmup"):
            preset["enable_gradual_warmup"] = True
        if _flag_present("--warmup-steps"):
            preset["warmup_steps"] = args.warmup_steps
        if _flag_present("--rebalance-steps"):
            preset["rebalance_steps"] = args.rebalance_steps
        if _flag_present("--skip-rebalance"):
            preset["skip_rebalance"] = bool(args.skip_rebalance)
        if _flag_present("--n-workers"):
            preset["n_workers"] = args.n_workers
        if _flag_present("--parallel-stages"):
            preset["parallel_stages"] = bool(args.parallel_stages)
        if _flag_present("--n-lambda"):
            preset["n_lambda"] = args.n_lambda

        defaults = {
            "resume": False,
            "reset": False,
            "temperature": 300.0,
            "platform": "CUDA",
            "output": "./output",
            "mode": "ibs",
            "decoupling": "dual_lambda",
            "potential": "softcore",
            "boresch_batch": 0,
            "boresch_select": 1,
            "enable_early_stop": False,
            "enable_gradual_warmup": False,
            "warmup_steps": 500000,
            "rebalance_steps": 50000,
            "skip_rebalance": False,
            "parallel_stages": False,
            "n_lambda": 12,
        }
        for key, value in defaults.items():
            preset.setdefault(key, value)

        # 复合物腿默认启用 Boresch；若用户未显式指定来源，则默认走自动估算。
        if preset.get("boresch") is None:
            preset["boresch"] = True
        if preset.get("boresch_source") in (None, "", "traditional") and preset.get("boresch", False):
            if getattr(args, "boresch_source", None) is None and not getattr(args, "boresch_anchors", None):
                preset["boresch_source"] = "auto"

        self.data = preset
        self.args = args  # 保留原始 args 供其它函数使用

    def __getattr__(self, item):
        # 优先从配置数据获取
        if item in self.data:
            return self.data[item]
        # 其次从命令行参数获取
        return getattr(self.args, item, None)

    def get(self, key, default=None):
        """兼容字典式获取"""
        if key in self.data:
            return self.data[key]
        return getattr(self.args, key, default)


# ---------------------------------------------------------------------------
# Boresch 参数统一管理
# ---------------------------------------------------------------------------
def _sanitize_boresch_params(params: Dict) -> Dict:
    """清洗 Boresch 参数字典，去除单位后缀，解包嵌套"""
    if params is None:
        return None

    # 尝试解包嵌套结构
    anchors = params.get("boresch_anchors", params)
    eq = anchors.get("equilibrium_values", params.get("equilibrium_values", {}))
    fc = anchors.get("force_constants", params.get("force_constants", {}))

    eq_mapping = {
        "r0_nm": "r0", "thetaA0_rad": "thetaA0", "thetaB0_rad": "thetaB0",
        "phiA0_rad": "phiA0", "phiB0_rad": "phiB0", "phiC0_rad": "phiC0"
    }
    fc_mapping = {
        "kr_kJ_mol_nm2": "kr", "kthetaA_kJ_mol_rad2": "kthetaA",
        "kthetaB_kJ_mol_rad2": "kthetaB", "kphiA_kJ_mol_rad2": "kphiA",
        "kphiB_kJ_mol_rad2": "kphiB", "kphiC_kJ_mol_rad2": "kphiC"
    }

    clean_eq = {eq_mapping.get(k, k): v for k, v in eq.items()}
    clean_fc = {fc_mapping.get(k, k): v for k, v in fc.items()}

    return {
        "receptor_indices": anchors.get("receptor_indices", params.get("receptor_indices", [])),
        "ligand_indices": anchors.get("ligand_indices", params.get("ligand_indices", [])),
        "equilibrium_values": clean_eq,
        "force_constants": clean_fc,
        "is_fallback": params.get("is_fallback", False),
    }


def _has_valid_boresch_anchors(params: Optional[Dict]) -> bool:
    """判定 Boresch 参数是否包含完整 3+3 锚点，避免键存在但值为空时静默失效。"""
    if not isinstance(params, dict):
        return False
    rec_idx = params.get("receptor_indices") or []
    lig_idx = params.get("ligand_indices") or []
    return len(rec_idx) == 3 and len(lig_idx) == 3


def _sanitize_boresch_params_strict(params: Dict) -> Dict:
    """严格清洗并校验 Boresch 参数，发现锚点缺失时立即报错。"""
    cleaned = _sanitize_boresch_params(params)
    if cleaned is None:
        return None
    if not _has_valid_boresch_anchors(cleaned):
        anchors = params.get("boresch_anchors", params) if isinstance(params, dict) else {}
        raise ValueError(
            "Boresch 参数缺失有效锚点：需要 3 个 receptor_indices 和 3 个 ligand_indices。"
            f" 当前解析结果 receptor_indices={anchors.get('receptor_indices', params.get('receptor_indices', [])) if isinstance(params, dict) else []},"
            f" ligand_indices={anchors.get('ligand_indices', params.get('ligand_indices', [])) if isinstance(params, dict) else []}"
        )
    return cleaned


def resolve_boresch_restraint(config: RunConfig, pipeline: ABFEPipeline) -> Optional[Dict]:
    """统一获取 Boresch 参数，支持传统文件、auto、simple、fluctuation 多种来源"""
    if not config.boresch:
        return None

    source = config.boresch_source
    output_dir = config.output

    # 传统文件来源
    if source in ("traditional", "orb_ml"):
        path = config.boresch_orb if source == "orb_ml" else config.boresch_anchors
        if not path or not os.path.exists(path):
            raise ValueError(f"Boresch 参数文件不存在: {path}")
        with open(path) as f:
            params = json.load(f)
        log.info("✅ 加载外部 Boresch 参数: %s", path)
        return _sanitize_boresch_params_strict(params)

    # 自动估算来源 (auto / simple / fluctuation)
    boresch_file = os.path.join(output_dir, f"boresch_{source}.json")
    if config.resume and not config.reset and os.path.exists(boresch_file):
        log.info("♻️ 从缓存加载 Boresch 参数: %s", boresch_file)
        with open(boresch_file) as f:
            params = json.load(f)
        return _sanitize_boresch_params_strict(params)

    # 需要预平衡生成轨迹
    if not config.reset and equilibrium_is_done(output_dir):
        log.info("♻️ 预平衡已完成，使用已有轨迹估算 Boresch")
    else:
        log.info("▶️ 执行预平衡以生成轨迹 (用于 Boresch 估算)")
        pipeline.pre_equilibrate(
            n_steps=config.get("n_equil_steps", 5_000_000),
            save_traj=True,
            resume=config.resume and not config.reset,
        )

    traj_file = os.path.join(output_dir, "pre_equilibration.dcd")
    if not os.path.exists(traj_file):
        raise RuntimeError("预平衡轨迹不存在，无法估算 Boresch 参数")
    traj_top = _resolve_mdtraj_topology_input(output_dir, config.gro, config.top)

    # 根据来源调用不同估算器
    if source == "auto":
        from abfe_core import OrbBoreschEstimator
        estimator = OrbBoreschEstimator(temperature=config.temperature)
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        # 自动估算（默认使用最后 5ns）
        candidates = estimator.estimate_multiple_anchors_from_trajectory(
            traj, config.ligand, n_candidates=1, output_path=boresch_file, use_last_ns=5.0
        )
        if not candidates:
            raise RuntimeError("自动 Boresch 估算失败，未找到合格候选")
        boresch = candidates[0]

    elif source == "simple":
        from abfe_core import OrbBoreschEstimator
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        estimator = OrbBoreschEstimator(temperature=config.temperature)
        boresch = estimator.estimate_from_trajectory(traj, config.ligand, output_path=boresch_file)

    elif source == "fluctuation":
        estimator = GeometricRestraintEstimator(temperature=config.temperature)
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        boresch = estimator.estimate_from_trajectory(traj, config.ligand, output_path=boresch_file)
    else:
        raise ValueError(f"未识别的 Boresch 来源: {source}")

    # 用最后一帧更新平衡几何量
    try:
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        protein_sel = traj.topology.select("protein and backbone")
        if len(protein_sel) > 0:
            traj.superpose(traj, 0, atom_indices=protein_sel)
        traj.center_coordinates()
        traj.image_molecules(inplace=True)
        last_frame_pos = traj.xyz[-1] * unit.nanometer
        new_eq = calc_boresch_from_last_frame(
            last_frame_pos, boresch["receptor_indices"], boresch["ligand_indices"]
        )
        boresch["equilibrium_values"] = new_eq
        log.info("✅ Boresch 平衡值已用最后一帧更新: r0=%.3f nm", new_eq.get("r0", 0))
    except Exception as e:
        log.warning("⚠️ 更新 Boresch 平衡值失败: %s", e)
    # 裁剪力常数到安全范围
    if "force_constants" in boresch:
        fc = boresch["force_constants"]
        fc["kr"] = float(np.clip(fc.get("kr", 500), 100.0, 2000.0))
        for key in ["kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC"]:
            if key in fc:
                fc[key] = float(np.clip(fc[key], 10.0, 200.0))
    # 保存
    with open(boresch_file, "w") as f:
        json.dump(boresch, f, indent=2, cls=NumpyEncoder)
    log.info("💾 Boresch 参数已保存: %s", boresch_file)
    return _sanitize_boresch_params_strict(boresch)


# ---------------------------------------------------------------------------
# 命令行解析
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ABFE 计算流程控制器 (v3.0 重构版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 首次运行（自动创建缓存并运行）
  python runabfe.py --gro complex.gro --top complex.top --ligand MOL \\
      --output ./output --boresch --boresch-source auto --preset production

  # 续跑（自动检测缓存）
  python runabfe.py --output ./output --ligand MOL --resume

  # 使用配置文件
  python runabfe.py --config params.json --output ./output --ligand MOL
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    prep_parser = subparsers.add_parser("prepare", help="生成预处理文件 (Boresch/DEXP等)")
    prep_parser.add_argument("--gro", required=True)
    prep_parser.add_argument("--top", required=True)
    prep_parser.add_argument("--ligand", required=True)
    prep_parser.add_argument("--ligand-xml", default=None, help="配体力场 XML/FFXML，用于溶剂腿构建")
    prep_parser.add_argument("--gmx-path", default=None)
    prep_parser.add_argument("--output-dir", default="./prep_output")
    prep_parser.add_argument("--save-boresch", default=None, help="保存 Boresch 文件")
    prep_parser.add_argument("--save-dexp", default=None, help="保存 DEXP 文件")
    prep_parser.add_argument("--fit-dexp", action="store_true")
    prep_parser.add_argument("--fit-frames", type=int, default=200, help="DEXP 拟合使用的最大帧数")
    prep_parser.add_argument("--fit-last-ns", type=float, default=None, help="仅使用轨迹最后多少 ns 做 DEXP 拟合")
    prep_parser.add_argument("--fit-env-radius", type=float, default=0.85, help="DEXP 环境筛选半径 (nm)")
    prep_parser.add_argument("--fit-env-max-atoms", type=int, default=0, help="DEXP 环境原子上限；<=0 表示不裁剪")
    prep_parser.add_argument("--fit-r-min", type=float, default=0.20, help="DEXP 拟合距离下限 (nm)")
    prep_parser.add_argument("--fit-r-max", type=float, default=0.45, help="DEXP 拟合距离上限 (nm)")
    prep_parser.add_argument("--temperature", type=float, default=300.0)
    prep_parser.add_argument("--platform", default="CUDA")
    prep_parser.add_argument("--n-steps", type=int, default=5_000_000)
    # 基本输入
    parser.add_argument("--gro", default=None, help="GROMACS 结构文件 (首次运行时必需)")
    parser.add_argument("--top", default=None, help="GROMACS 拓扑文件 (首次运行时必需)")
    parser.add_argument("--ligand", default=None, help="配体残基名称 (如 MOL)")
    parser.add_argument("--ligand-xml", default=None, help="配体力场 XML/FFXML，用于溶剂腿构建")
    parser.add_argument("--gmx-path", default=None, help="GROMACS 力场 include 目录")
    parser.add_argument("--output", default="./output", help="输出目录 (同时也是缓存目录)")
    parser.add_argument("--config", default=None, help="JSON 配置文件，可覆盖预设")

    # 运行模式
    parser.add_argument("--resume", action="store_true", help="从 Checkpoint 恢复运行")
    parser.add_argument("--reset", action="store_true", help="忽略所有缓存，强制重新开始")

    # 策略选择
    parser.add_argument("--mode", default="ibs", choices=["ibs", "traditional"],
                        help="采样引擎: ibs (默认) 或 traditional-REMD")
    parser.add_argument(
        "--decoupling",
        default="dual_lambda",
        choices=["dual_lambda", "single_lambda", "2d_diagonal", "2d_geodesic"],
    )
    parser.add_argument("--potential", default="softcore", choices=["softcore", "dexp"])
    parser.add_argument("--dexp-params", default=None, help="DEXP 参数文件 (JSON)")

    # Boresch 相关
    parser.add_argument("--boresch", action=argparse.BooleanOptionalAction, default=None,
                        help="是否启用 Boresch 限制力；复合物腿默认启用，可用 --no-boresch 显式关闭")
    parser.add_argument("--boresch-source", default=None,
                        choices=["traditional", "orb_ml", "simple", "auto", "fluctuation"])
    parser.add_argument("--boresch-anchors", default=None, help="传统 Boresch 锚点文件")
    parser.add_argument("--boresch-orb", default=None, help="Orb ML 预测 Boresch 文件")
    parser.add_argument("--boresch-batch", type=int, default=0, help="批量估算候选数量")
    parser.add_argument("--boresch-select", type=int, default=1, help="选择第 N 个候选")
    parser.add_argument("--skip-rebalance", action="store_true", help="跳过带 Boresch 的再平衡")
    parser.add_argument("--rebalance-steps", type=int, default=50000, help="再平衡步数")

    # 采样控制
    parser.add_argument("--preset", default="production", choices=PRESET_CONFIGS.keys())
    parser.add_argument("--n-steps-per-window", type=int, default=None)
    parser.add_argument("--steps-per-update", type=int, default=None)
    parser.add_argument("--n-states-per-stage", type=int, default=None, help="每阶段 λ 状态数")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU"])

    # 高级选项
    parser.add_argument("--torsion-params", default=None)
    parser.add_argument("--enable-early-stop", action="store_true")
    parser.add_argument("--enable-gradual-warmup", action="store_true")
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=500000)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true", help="仅分析已有 .npy")
    parser.add_argument("--parallel-stages", action="store_true", help="并行执行去电荷和去VDW阶段")
    parser.add_argument("--n-lambda", type=int, default=12, help="传统 REMD 模式的 λ 状态数")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 分析模式（简化版，调用原有后处理）
# ---------------------------------------------------------------------------
def run_post_analysis(args):
    output_dir = args.output
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")

    log.info("进入后处理分析模式...")
    temp = args.temperature * unit.kelvin
    kt = (unit.MOLAR_GAS_CONSTANT_R * temp).value_in_unit(unit.kilojoule_per_mole)

    # 加载 Boresch 参数（如果有）
    boresch_params = None
    for json_name in ["boresch_auto.json", "boresch_simple.json", "boresch_fluctuation.json"]:
        path = os.path.join(output_dir, json_name)
        if os.path.exists(path):
            with open(path) as f:
                boresch_params = _sanitize_boresch_params(json.load(f))
            break

    # 扫描 output_dir 下所有 stage 的 energies.npy
    stages = ["coul", "vdw"]  # 假设 decharging= coul, vanishing= vdw
    stage_name_map = {"coul": "decharging", "vdw": "vanishing"}
    total_dg = 0.0
    total_err_sq = 0.0
    for stage in stages:
        stage_name = stage_name_map[stage]
        stage_dir = os.path.join(output_dir, stage_name)
        if not os.path.exists(stage_dir):
            continue
        window_data = []

        stage_checkpoint = os.path.join(
            output_dir,
            "checkpoints",
            "stage1_decharging.json" if stage == "coul" else "stage2_vanishing.json",
        )
        if stage == "coul" and os.path.exists(stage_checkpoint):
            with open(stage_checkpoint) as f:
                cached_stage = json.load(f)
            dg = float(cached_stage.get("total_delta_G", 0.0))
            err = float(cached_stage.get("total_error", 0.0))
            total_dg += dg
            total_err_sq += err**2
            log.info("Stage %s: 使用 PME 去电荷 checkpoint ΔG = %.2f ± %.2f kJ/mol", stage, dg, err)
            continue
            
        # 🔑 修复：读取真实的窗口划分与 Lambda 索引
        preopt_file = os.path.join(output_dir, "checkpoints", f"preopt_dual_{stage_name}.json")
        window_ranges = []
        if os.path.exists(preopt_file):
            try:
                with open(preopt_file) as f:
                    window_ranges = json.load(f).get("window_ranges", [])
                log.info("  ✅ 从缓存恢复真实 window_ranges: %s", window_ranges)
            except Exception as e:
                log.warning("  ⚠️ 读取 preopt 缓存失败: %s", e)
                
        e_files = sorted(glob.glob(os.path.join(stage_dir, f"dual_window_*_{stage}_energies.npy")))
        for w_idx, e_file in enumerate(e_files):
            u_kn = np.load(e_file)
            bias_path = e_file.replace("_energies.npy", "_bias.npy")
            base_path = e_file.replace("_energies.npy", "_base.npy")
            if not os.path.exists(bias_path):
                raise RuntimeError(
                    f"缺少 IBS bias 能量文件: {bias_path}。"
                    "偏置采样必须在 MBAR 中补偿，不能用 0 代替。"
                )
            if not os.path.exists(base_path):
                raise RuntimeError(f"缺少 base 能量文件: {base_path}")
            bias = np.load(bias_path)
            base = np.load(base_path)
            
            # 🔑 修复：赋予真实的全局 lambda_indices
            if w_idx < len(window_ranges):
                start, end = window_ranges[w_idx]
                real_indices = list(range(start, end))
            else:
                log.warning("  ⚠️ 窗口 %d 缺失真实索引映射，降级使用局部索引（拼接可能失败）", w_idx)
                real_indices = list(range(u_kn.shape[0]))
                
            window_data.append({
                'u_kn': u_kn, 
                'bias_energies': bias, 
                'base_energies': base, 
                'lambda_indices': real_indices  # ✅ 传入真实全局索引
            })
        if window_data:
            from ibs_engine import solve_stage_integrated  # 已在顶部导入
            res = solve_stage_integrated(window_data, kt, stage_name=stage)
            dg = res.get("total_delta_G", 0.0)
            err = res.get("total_error", 999.0)
            total_dg += dg
            total_err_sq += err**2
            log.info("Stage %s: ΔG = %.2f ± %.2f kJ/mol", stage, dg, err)

    # Boresch 解析修正
    dg_boresch = 0.0
    if boresch_params and boresch_params.get("force_constants") and boresch_params.get("equilibrium_values"):
        try:
            dg_boresch = calculate_boresch_analytical_correction(
                eq=boresch_params["equilibrium_values"],
                fc=boresch_params["force_constants"],
                T=temp,
            )
            log.info("Boresch 修正: %.2f kJ/mol", dg_boresch)
        except Exception as e:
            log.warning("Boresch 修正计算失败: %s", e)

    final_dg = total_dg + dg_boresch
    final_err = np.sqrt(total_err_sq + 0.1**2)
    log.info("总 ΔG = %.2f ± %.2f kJ/mol", final_dg, final_err)

    result = {
        "total_delta_G_kJ_mol": float(final_dg),
        "total_error_kJ_mol": float(final_err),
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(output_dir, "final_results_postprocess.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    log.info("结果已保存: %s", out_path)

def run_prepare_command(args):
    """简化版预处理命令：生成系统缓存 + 可选 Boresch / DEXP 估算"""
    log.info("🔧 执行 prepare 子命令")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 构建系统（与 main 中首次构建一致）
    system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
        args.gro, args.top, args.ligand,
        find_gmx_include_dir(args.gmx_path)
    )
    diagnose_14_scaling(system)
    save_native_system(output_dir, system, topology, ligand_indices, positions, box_vectors)

    # 2. 从缓存重载（保证一致性）
    system, topology, positions, box_vectors, ligand_indices = load_native_system(
        output_dir, gro_file=args.gro
    )

    # 3. PBC 居中（与主流程完全一致）
    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    positions, box_vectors = center_system_rigidly(positions, box_vectors, ligand_indices)

    # 4. 初始化 Pipeline（只做预平衡估算用）
    pipeline = ABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=args.temperature,
        output_dir=output_dir,
        platform_name=args.platform,
    )

    # 5. 预平衡生成轨迹
    equil_data = pipeline.pre_equilibrate(
        n_steps=args.n_steps,
        save_traj=True,
        platform_name=args.platform,
    )
    traj_file = equil_data["trajectory_file"]

    # 6. Boresch 估算（默认 fluctuation 模式）
    if args.save_boresch:
        import mdtraj as md
        traj = md.load(traj_file, top=args.gro)
        estimator = GeometricRestraintEstimator(temperature=args.temperature)
        boresch = estimator.estimate_from_trajectory(
            traj, args.ligand,
            output_path=os.path.join(output_dir, args.save_boresch)
        )
        log.info("Boresch 参数已保存至 %s", args.save_boresch)

    # DEXP 拟合（若需要）
    if args.fit_dexp and args.save_dexp:
        log.info("🧪 启动 Orbv3 → DEXP 拟合...")
        dexp_json = pipeline.fit_dexp_parameters(
            ligand_resname=args.ligand,
            top_file=args.top,
            output_name=args.save_dexp,
            device="cuda" if args.platform.upper() == "CUDA" else "cpu",
            n_frames=args.fit_frames,
            env_radius_nm=args.fit_env_radius,
            env_max_atoms=(args.fit_env_max_atoms if args.fit_env_max_atoms > 0 else None),
            fit_last_ns=args.fit_last_ns,
            fit_r_min=args.fit_r_min,
            fit_r_max=args.fit_r_max,
            gmx_include_dir=find_gmx_include_dir(args.gmx_path),
        )
        log.info("DEXP 参数已保存至 %s", dexp_json)

    log.info("✅ prepare 完成，文件已输出至 %s", output_dir)

# ---------------------------------------------------------------------------
# 传统 ABFE-REMD 模式
# ---------------------------------------------------------------------------
def run_traditional_mode(config: RunConfig):
    """传统双阶段 λ-REMD：分别计算复合物腿与溶剂腿并汇总结合自由能。"""
    log.info("🔧 启动传统 ABFE-REMD 模式")
    output_dir = config.output
    os.makedirs(output_dir, exist_ok=True)

    if not config.gro or not config.top:
        raise ValueError("traditional 模式必须提供 gro/top 输入，或在配置文件中定义 gro/top。")

    system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
        config.gro,
        config.top,
        config.ligand,
        find_gmx_include_dir(config.gmx_path),
    )
    diagnose_14_scaling(system)

    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    positions, box_vectors = center_system_rigidly(positions, box_vectors, ligand_indices)

    ligand_resname = _get_residue_name_by_atom_index(topology, ligand_indices[0])
    if config.reset or not solvent_cache_exists(output_dir):
        log.info("💧 traditional 模式准备溶剂腿缓存...")
        if not build_and_cache_solvent_leg(
            output_dir,
            topology,
            positions,
            ligand_indices,
            ligand_resname,
            ligand_ffxml=getattr(config, "ligand_xml", None),
            top_file=config.top,
            gmx_include_dir=find_gmx_include_dir(config.gmx_path),
        ):
            raise RuntimeError("traditional 模式自动构建溶剂腿缓存失败。")

    dg_boresch = 0.0
    if config.boresch:
        boresch_pipeline = ABFEPipeline(
            system=system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            ligand_indices=ligand_indices,
            temperature=config.temperature,
            output_dir=output_dir,
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            platform_name=config.platform,
        )
        boresch_restraint = resolve_boresch_restraint(config, boresch_pipeline)
        if boresch_restraint:
            dg_boresch = boresch_pipeline.apply_boresch_correction(
                boresch_restraint,
                autoload_from_disk=False,
            ).get("delta_g_rest", 0.0)

    complex_pipeline = TraditionalABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=config.temperature,
        platform_name=config.platform,
        output_dir=os.path.join(output_dir, "traditional_complex"),
    )

    log.info("🔄 开始传统复合物腿 REMD + MBAR (n_lambda=%d, n_steps=%d)",
             config.n_lambda, config.n_steps_per_window or 500000)
    complex_results = complex_pipeline.run_full(
        n_lambda=config.n_lambda,
        n_steps_per_leg=config.n_steps_per_window or 500000,
        boresch_correction=0.0,
    )

    sys_solv, top_solv, pos_solv, box_solv, lig_idx_solv = load_native_system(
        output_dir,
        phase="solvent",
        prefer_equilibrated=not config.reset,
    )
    pos_solv, box_solv = center_system_rigidly(pos_solv, box_solv, lig_idx_solv)
    solvent_pipeline = TraditionalABFEPipeline(
        system=sys_solv,
        topology=top_solv,
        positions=pos_solv,
        box_vectors=box_solv,
        ligand_indices=lig_idx_solv,
        temperature=config.temperature,
        platform_name=config.platform,
        output_dir=os.path.join(output_dir, "traditional_solvent"),
    )
    log.info("🔄 开始传统溶剂腿 REMD + MBAR (n_lambda=%d, n_steps=%d)",
             config.n_lambda, config.n_steps_per_window or 500000)
    solvent_results = solvent_pipeline.run_full(
        n_lambda=config.n_lambda,
        n_steps_per_leg=config.n_steps_per_window or 500000,
        boresch_correction=0.0,
    )

    dg_complex = float(complex_results["delta_G_total_kJ_mol"])
    dg_solvent = float(solvent_results["delta_G_total_kJ_mol"])
    err_complex = float(complex_results["error_leg_kJ_mol"])
    err_solvent = float(solvent_results["error_leg_kJ_mol"])
    delta_g_bind = dg_complex - dg_solvent + float(dg_boresch)
    total_err_bind = float(np.sqrt(err_complex**2 + err_solvent**2))

    final = {
        "complex_leg": complex_results,
        "solvent_leg": solvent_results,
        "complex_delta_G_kJ_mol": dg_complex,
        "solvent_delta_G_kJ_mol": dg_solvent,
        "boresch_correction_kJ_mol": float(dg_boresch),
        "delta_G_bind_kJ_mol": float(delta_g_bind),
        "delta_G_bind_kcal_mol": float(delta_g_bind / 4.184),
        "total_error_kJ_mol": total_err_bind,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(output_dir, "final_binding_results_traditional.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, cls=NumpyEncoder)
    log.info("✅ 传统 ABFE 完成: ΔG_bind = %.2f ± %.2f kJ/mol", delta_g_bind, total_err_bind)
    log.info("💾 传统模式最终结果已保存: %s", out_path)
    return final

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()

    if args.command == "prepare":
        run_prepare_command(args)
        return
    # 创建配置对象
    config = RunConfig(args)
    if not config.ligand:
        log.error("未提供配体残基名称。请通过 --ligand 或配置文件中的 ligand 指定。")
        sys.exit(2)
    # 传统模式单独处理
    if config.mode == "traditional":
        run_traditional_mode(config)
        return
    # 分析模式单独处理
    if args.analyze_only:
        run_post_analysis(config)
        return

    # 准备输出目录
    output_dir = config.output
    os.makedirs(output_dir, exist_ok=True)

    # ----- 1. 系统加载：优先从缓存，否则 GROMACS 构建并立即落盘 -----
    if not config.reset and system_cache_exists(output_dir):
        system, topology, positions, box_vectors, ligand_indices = load_native_system(
            output_dir, 
            gro_file=config.gro, 
            top_file=config.top, 
            gmx_include_dir=find_gmx_include_dir(config.gmx_path)
        )
        log.info("♻️ 从缓存加载 System 完成")
    else:
        if not config.gro or not config.top:
            log.error("未提供 --gro/--top 且无缓存，无法构建系统")
            sys.exit(1)
        # 从 GROMACS 构建
        system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
            config.gro, config.top, config.ligand,
            find_gmx_include_dir(config.gmx_path)
        )
        diagnose_14_scaling(system)
        # 立即保存为原生缓存
        save_native_system(output_dir, system, topology, ligand_indices, positions, box_vectors)
        log.info("✅ 原生缓存已生成")
        # 重新从缓存加载，确保后续所有对象都来自落盘文件
        # 替换原有的 load_native_system 调用为：
        system, topology, positions, box_vectors, ligand_indices = load_native_system(
            output_dir, 
            gro_file=config.gro, 
            top_file=config.top, 
            gmx_include_dir=find_gmx_include_dir(config.gmx_path),
            prefer_equilibrated=False,
        )
        log.info("🔄 已从缓存重新加载 System (使用落盘后对象)")

    # 坐标安全处理：转换为纯 numpy 再转为 Vec3 列表（防止类型问题）
    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    # ========== PBC 确定性居中与包裹 ==========
    positions, box_vectors = center_system_rigidly(
        positions, box_vectors, ligand_indices
    )
    log.info("  ✅ 配体已居中，分子完整性修复完毕")    

    # ----- 1.5 自动构建溶剂腿缓存 -----
    ligand_resname = _get_residue_name_by_atom_index(topology, ligand_indices[0])
    if config.reset or not solvent_cache_exists(output_dir):
        log.info("💧 溶剂腿缓存缺失或要求重建，开始自动构建纯水配体体系...")
        if not build_and_cache_solvent_leg(
            output_dir,
            topology,
            positions,
            ligand_indices,
            ligand_resname,
            ligand_ffxml=getattr(config, "ligand_xml", None),
            top_file=config.top,
            gmx_include_dir=find_gmx_include_dir(config.gmx_path),
        ):
            log.error("❌ 自动构建溶剂腿失败，无法继续一键 ABFE")
            sys.exit(1)
    else:
        log.info("♻️ 检测到已有溶剂腿缓存，跳过重建")

    # ----- 2. 初始化 Pipeline -----
    pipeline = ABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=config.temperature,
        output_dir=output_dir,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        platform_name=config.platform,
    )

    # ----- 3. 加载可选参数 -----
    # 二面角修正
    torsion_params = None
    if config.torsion_params and os.path.exists(config.torsion_params):
        with open(config.torsion_params) as f:
            torsion_params = json.load(f)

    # DEXP 势能
    dexp_params = None
    if config.potential == "dexp" and config.dexp_params and os.path.exists(config.dexp_params):
        with open(config.dexp_params) as f:
            dexp_dict = json.load(f)
        potential_obj = DEXPSurrogatePotential.from_dict(dexp_dict)
        dexp_params = potential_obj.get_parameters_dict()
    else:
        dexp_params = None

    # ----- 4. Boresch 参数获取（可能触发预平衡估算） -----
    if config.boresch:
        log.info("🧷 复合物腿 Boresch 已启用 | source=%s", config.boresch_source)
    else:
        log.info("🧷 复合物腿 Boresch 已显式关闭")
    boresch_restraint = resolve_boresch_restraint(config, pipeline)

    # ----- 5. 带限制力再平衡（如果启用） -----
    if boresch_restraint and not config.skip_rebalance:
        log.info("🔧 执行带 Boresch 限制力的再平衡...")
        rebal_data = pipeline._rebalance_with_boresch(
            boresch_params=boresch_restraint,
            n_steps=config.rebalance_steps,
            resume=config.resume and not config.reset,
        )
        pipeline.positions = rebal_data["positions"]
        pipeline.box_vectors = rebal_data["box_vectors"]
        log.info("✓ 再平衡完成，坐标已更新")

    # ----- 6. 运行复合物腿主流程 -----
    log.info("🔄 启动复合物腿主采样流程 (%s)", config.decoupling)
    complex_results = pipeline.run_full_pipeline(
        decoupling_scheme=config.decoupling,
        potential_type=config.potential,
        dexp_params=dexp_params,
        boresch_params=boresch_restraint,
        torsion_params=torsion_params,
        resume=config.resume and not config.reset,
        run_equilibration=not equilibrium_is_done(output_dir) or config.reset,
        n_steps_per_window=config.n_steps_per_window,
        steps_per_update=config.steps_per_update,
        n_states_per_stage=config.get("stage1_n_states", 16),
        stage1_n_states=config.get("stage1_n_states", 16),
        stage2_n_states=config.get("stage2_n_states", config.get("stage1_n_states", 16)),
        enable_early_stop=config.enable_early_stop,
        enable_gradual_warmup=config.enable_gradual_warmup,
        warmup_steps=config.warmup_steps,
        n_workers=config.n_workers,
        parallel_stages=config.parallel_stages,
        allow_disk_boresch_autoload=True,
    )
    
    dg_complex = complex_results.get("total_delta_G_complex_kJ_mol", complex_results.get("decoupling_delta_G_kJ_mol", 0.0))
    err_complex = complex_results.get("total_error_kJ_mol", 0.0)
    dg_boresch = complex_results.get("boresch_correction_kJ_mol", 0.0)

    # ----- 7. 自动加载溶剂相缓存并运行溶剂腿 (ABFE 必选项) -----
    log.info("\n" + "="*70)
    log.info("💧 启动溶剂相 (Ligand-in-Water) 配体腿计算 (自动加载缓存)...")
    log.info("="*70)
    
    try:
        sys_solv, top_solv, pos_solv, box_solv, lig_idx_solv = load_native_system(output_dir, phase="solvent")
    except FileNotFoundError:
        log.error("❌ 未找到溶剂相缓存，自动构建步骤未成功完成")
        sys.exit(1)
        
    pos_solv, box_solv = center_system_rigidly(pos_solv, box_solv, lig_idx_solv)
    
    solvent_out_dir = os.path.join(output_dir, "solvent_leg")
    pipeline_solv = ABFEPipeline(
        system=sys_solv, topology=top_solv, positions=pos_solv, box_vectors=box_solv,
        ligand_indices=lig_idx_solv, temperature=config.temperature,
        output_dir=solvent_out_dir, checkpoint_dir=os.path.join(solvent_out_dir, "checkpoints"),
        platform_name=config.platform,
    )
    
    # 运行溶剂腿 (🔑 强制关闭 Boresch)
    solv_results = pipeline_solv.run_full_pipeline(
        decoupling_scheme=config.decoupling,
        potential_type=config.potential,
        dexp_params=dexp_params,
        boresch_params=None,  # 绝对不加 Boresch
        torsion_params=torsion_params,
        resume=config.resume and not config.reset,
        run_equilibration=not equilibrium_is_done(solvent_out_dir) or config.reset,
        n_steps_per_window=config.n_steps_per_window,
        steps_per_update=config.steps_per_update,
        n_states_per_stage=config.get("stage1_n_states", 16),
        stage1_n_states=config.get("stage1_n_states", 16),
        stage2_n_states=config.get("stage2_n_states", config.get("stage1_n_states", 16)),
        enable_early_stop=config.enable_early_stop,
        enable_gradual_warmup=config.enable_gradual_warmup,
        warmup_steps=config.warmup_steps,
        allow_disk_boresch_autoload=False,
    )
    
    dg_solvent = solv_results.get("total_delta_G_complex_kJ_mol", solv_results.get("decoupling_delta_G_kJ_mol", 0.0))
    err_solvent = solv_results.get("total_error_kJ_mol", 0.0)
    
    # ----- 8. 计算最终结合自由能 ΔG_bind -----
    delta_g_bind = dg_complex - dg_solvent
    total_err_bind = np.sqrt(err_complex**2 + err_solvent**2)
    
    log.info("\n" + "="*70)
    log.info("🎯 ABFE 最终结合自由能计算结果:")
    log.info("   复合物腿 (膜/蛋白+水) ΔG_cplx  = %.2f ± %.2f kJ/mol", dg_complex, err_complex)
    log.info("   溶剂腿   (纯水)       ΔG_solv  = %.2f ± %.2f kJ/mol", dg_solvent, err_solvent)
    log.info("   Boresch 解析修正      ΔG_rest  = %.2f kJ/mol", dg_boresch)
    log.info("   --------------------------------------------------------")
    log.info("   结合自由能 ΔG_bind           = %.2f ± %.2f kJ/mol", delta_g_bind, total_err_bind)
    log.info("                              = %.2f ± %.2f kcal/mol", delta_g_bind/4.184, total_err_bind/4.184)
    log.info("="*70)
    
    # 保存最终结合结果
    final_bind_result = {
        "complex_delta_G_kJ_mol": float(dg_complex),
        "solvent_delta_G_kJ_mol": float(dg_solvent),
        "boresch_correction_kJ_mol": float(dg_boresch),
        "delta_G_bind_kJ_mol": float(delta_g_bind),
        "delta_G_bind_kcal_mol": float(delta_g_bind / 4.184),
        "total_error_kJ_mol": float(total_err_bind),
        "timestamp": datetime.now().isoformat()
    }
    bind_out_path = os.path.join(output_dir, "final_binding_results.json")
    with open(bind_out_path, "w") as f:
        json.dump(final_bind_result, f, indent=2, cls=NumpyEncoder)
    log.info("💾 最终结合自由能结果已保存: %s", bind_out_path)

    # ----- 7. 输出最终结果 -----
    log.info("✅ ABFE 计算完成")
    log.info("结果已生成，见 %s", bind_out_path)


if __name__ == "__main__":
    main()

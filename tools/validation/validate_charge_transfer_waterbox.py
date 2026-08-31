#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C1：带电配体小水盒 charge-transfer 验证（docs/status/memtodolist.md §17.0 步骤⑤）。

## 与 docs/status/memtodolist.md 原始 C1 草案的一处刻意偏离（2026-08-06 拍定）

原稿要求"至少一个 q_L=+1 和一个 q_L=-1 的可追溯小分子输入"，示例给的是质子化
Atenolol / acetate，需要 Gaussian + Sobtop 重新做 RESP/GAFF 参数化。

**本实现改用单原子离子做"配体"：Na⁺（+1）、Cl⁻（−1）、Ca²⁺（+2）。** 理由：

- C1 要验证的是 co-alchemical charge-transfer 的 PME 哈密顿量机制本身
  （电荷守恒、λ 端点、盒长依赖），不是某个药物分子的化学身份；单原子离子的净电荷
  严格是整数，没有"partial charge 加和是否等于声明净电荷"这层误差来源，比现造一个
  RESP 分子反而更严格。
- 参数直接来自 OpenMM 自带的 `amber14/tip3p.xml`（与本仓库 §0.5.6 识别出的水模型
  逐位一致，见 `abfe_core.resolve_water_model_xml` 的匹配结果），不需要 Gaussian/
  Sobtop/RESP 这一整套外部 QM 流程。
- Na⁺/Cl⁻/Ca²⁺ 都有公开的水合自由能参考值，charging ΔG 算出来后可以额外做一次
  外部合理性对照，原稿的自定义小分子没有这层。
- Ca²⁺（+2）正好实测 §2.2 "`|q_L|>1` 用多个单价 co-ion 分摊"这条此前只有合成
  topology 测过的路径。

按 §2.2 的定义，co-ion 是**同号** dummy，不是异号反离子：Na⁺ 配体配 `0→+1` 的
Na⁺形 dummy；Cl⁻ 配体配 `0→−1` 的 Cl⁻形 dummy；Ca²⁺ 配体配两个 `0→+1` 的
Na⁺形 dummy 分摊。这是当前实现（`runabfe._insert_reserved_coalchemical_ion_dummies`）
唯一支持的 dummy 模板，不是本脚本另造的判据。

## 本脚本做什么、不做什么

只做 validation harness：不复制生产 Hamiltonian，只调用已有的生产函数
（`ibs_engine.select_co_alchemical_ion_once` / `configure_pme_ligand_charge_offsets` /
`charging_charge_conservation_report` / `TraditionalMBARAnalyzer.compute_u_kn`，
dummy 插入调用 `runabfe._insert_reserved_coalchemical_ion_dummies`，不重新实现
"删最远水、插 dummy"这段逻辑）。

阶段（子命令）：

    build         构建小水盒 + 插入 reserved co-ion dummy + 冻结身份 spec（纯 CPU）
    static-check  逐 λ 电荷守恒代数证明 + 几何/参数一致性核对（纯 CPU，不建 Context）
    dynamics      短动力学采样（需要 GPU/CUDA，本脚本本身不会被自动执行——见下）
    ukn           从轨迹重算 u_kn 并用 MBAR 求 charging ΔG（CPU，但读取 GPU 产出的轨迹）
    report        汇总 static-check + dynamics 诊断 + ΔG 为 report.json/summary.json
    compare-box   比较同一电荷符号的小/大盒两份 report.json，判 §13.4 盒长敏感性

`build` 与 `static-check` 是纯 CPU、不建 Context 迭代，可以在提交 GPU 作业之前先跑，
用来在秒级时间内抓构建期的 bug——这两步已经跑过一次自检（见 handoff 说明）。
`dynamics` 会创建 Context 并调用 integrator.step()，按仓库约定必须由用户在自己的
计算节点上提交，不在本次改动里自动执行。

示例（先建小盒 +1 Na 案例）：

    cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
    source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
    mamba activate openmm_dev

    python tools/validation/validate_charge_transfer_waterbox.py build \\
        --ion Na --box-size small \\
        --output-dir validation/c1_waterbox/plus1_small

    python tools/validation/validate_charge_transfer_waterbox.py static-check \\
        --output-dir validation/c1_waterbox/plus1_small
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import openmm
from openmm import app, unit, Vec3, XmlSerializer

import abfe_core as core
import ibs_engine as engine
import runabfe  # noqa: F401  # 只为拿到 _insert_reserved_coalchemical_ion_dummies


# ============================================================================
# 常量：三个单原子离子 case 的模板
# ============================================================================

# 元素/残基名/AMBER 力场里的形式电荷（与 amber14/tip3p.xml 的 <Residue> 定义逐位一致，
# 2026-08-06 已用 `grep <Residue name="(NA|CL|CA)">` 对该 XML 实测确认）。
ION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Na": {"residue_name": "NA", "element": "sodium", "charge_e": 1},
    "Cl": {"residue_name": "CL", "element": "chlorine", "charge_e": -1},
    "Ca": {"residue_name": "CA", "element": "calcium", "charge_e": 2},
}

# 与生产溶剂腿完全一致的力场/水模型/cutoff/离子强度选择（§4：两种盒子除盒长和水数
# 外一切必须一致；这里额外要求与生产复合物/溶剂腿一致，方便日后横向对照）。
FORCEFIELD_XML = ["amber14-all.xml", "amber14/tip3p.xml"]
NONBONDED_CUTOFF_NM = core.SOLVENT_NONBONDED_CUTOFF_NM  # = 1.0，与生产溶剂腿同一常量
DEFAULT_IONIC_STRENGTH_MOLAR = runabfe.DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR  # = 0.15
DEFAULT_TEMPERATURE_K = 300.0

# §17.0/C1 建议的盒长公式：L_small = max(配体尺寸 + 2×1.5nm, 3.2nm)，L_large = L_small+1.0nm。
# 单原子离子的"配体尺寸"为 0，所以：
BOX_EDGE_NM = {"small": 3.2, "large": 4.2}

PROTOCOL_VERSION = 1  # 本脚本自己的产物版本号，只影响 build_manifest.json 里的记录


def _ion_template(ion: str) -> Dict[str, Any]:
    if ion not in ION_TEMPLATES:
        raise SystemExit(
            f"--ion 只接受 {sorted(ION_TEMPLATES)}，收到 {ion!r}"
        )
    return ION_TEMPLATES[ion]


def _openmm_element(name: str):
    return getattr(app.element, name)


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _box_vectors_nm(topology) -> np.ndarray:
    vecs = topology.getPeriodicBoxVectors()
    if vecs is None:
        raise RuntimeError("拓扑缺少周期盒向量")
    return np.array([v.value_in_unit(unit.nanometer) for v in vecs], dtype=np.float64)


def _minimum_image_distance_nm(p1_nm: np.ndarray, p2_nm: np.ndarray, box_nm: np.ndarray) -> float:
    inv_box = np.linalg.inv(box_nm)
    delta = np.asarray(p1_nm) - np.asarray(p2_nm)
    frac = delta @ inv_box
    frac -= np.round(frac)
    return float(np.linalg.norm(frac @ box_nm))


# ============================================================================
# 子命令：build
# ============================================================================


def cmd_build(args: argparse.Namespace) -> None:
    template = _ion_template(args.ion)
    q_l = int(template["charge_e"])
    box_edge_nm = float(args.box_edge_nm) if args.box_edge_nm else BOX_EDGE_NM[args.box_size]
    ionic_strength_molar = float(args.ionic_strength_molar)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"⚗️  配体 = {args.ion}{'+' * max(q_l, 0)}{'-' * max(-q_l, 0)} "
          f"(净电荷 {q_l:+d} e)，盒长 {box_edge_nm:.3f} nm，"
          f"离子强度 {ionic_strength_molar:.3f} M")

    # 1. 建一个只含配体离子的 Modeller（配体永远是拓扑里第一个、也是唯一预先存在
    #    的原子——之后 addSolvent 只会在它后面追加新原子，不会重排已有原子的索引；
    #    §4.1"复合物腿和溶剂腿使用相同水模型/离子模型"的口径同样适用于本验证）。
    lig_top = app.Topology()
    lig_chain = lig_top.addChain()
    lig_residue = lig_top.addResidue(template["residue_name"], lig_chain)
    lig_top.addAtom(template["residue_name"], _openmm_element(template["element"]), lig_residue)
    center = box_edge_nm / 2.0
    lig_positions = [Vec3(center, center, center)] * unit.nanometer

    modeller = app.Modeller(lig_top, lig_positions)
    ff = app.ForceField(*FORCEFIELD_XML)

    modeller.addSolvent(
        ff,
        boxSize=Vec3(box_edge_nm, box_edge_nm, box_edge_nm) * unit.nanometer,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=ionic_strength_molar * unit.molar,
        neutralize=True,
    )

    # fail closed：配体必须还是 atom 0，且身份没被 addSolvent 动过。
    all_atoms = list(modeller.topology.atoms())
    ligand_atom = all_atoms[0]
    if (
        ligand_atom.residue.name != template["residue_name"]
        or ligand_atom.element != _openmm_element(template["element"])
    ):
        raise RuntimeError(
            "addSolvent 之后 atom 0 的身份变了（原本应该是配体离子）："
            f"residue={ligand_atom.residue.name}, element={ligand_atom.element}"
        )
    ligand_indices = [0]

    realized_box_nm = _box_vectors_nm(modeller.topology)
    realized_edges = np.linalg.norm(realized_box_nm, axis=1)
    if not np.allclose(realized_edges, box_edge_nm, atol=1.0e-6):
        raise RuntimeError(
            f"溶剂盒构建结果与请求不符：请求 {box_edge_nm:.6f} nm 立方，"
            f"实际边长 {[round(float(x), 6) for x in realized_edges]} nm"
        )

    # 2. 插入 reserved co-ion dummy（复用已有实现，不重新写一遍"摘最远水、插 dummy"）。
    cation = q_l > 0
    reserved_indices = runabfe._insert_reserved_coalchemical_ion_dummies(
        modeller,
        count=abs(q_l),
        cation=cation,
        ligand_atom_indices=ligand_indices,
    )
    print(f"  ✅ 插入 {len(reserved_indices)} 个 {'Na⁺' if cation else 'Cl⁻'} 形 "
          f"reserved co-ion dummy: indices={reserved_indices}")

    # 3. createSystem，并把 reserved dummy 的电荷显式清零（模板给的是 ±1）。
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=NONBONDED_CUTOFF_NM * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )
    nb_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    for idx in reserved_indices:
        _, sigma, epsilon = nb_force.getParticleParameters(idx)
        nb_force.setParticleParameters(idx, 0.0 * unit.elementary_charge, sigma, epsilon)

    # 核对配体净电荷确实是我们以为的那个整数（§2.1/§7.2 的第一道自洽检查）。
    raw_q = sum(
        nb_force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in ligand_indices
    )
    if abs(raw_q - q_l) > 1.0e-9:
        raise RuntimeError(f"配体净电荷读回 {raw_q:+.9f} e，与模板 {q_l:+d} e 不符")

    # 4. 选一次 co-ion 身份并冻结（[MEM-00c] 唯一入口）。
    positions_nm = np.asarray(modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    spec = engine.select_co_alchemical_ion_once(
        system,
        ligand_indices,
        modeller.topology,
        modeller.positions,
        modeller.topology.getPeriodicBoxVectors(),
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ion_restraint_k=args.restraint_k,
        flat_bottom_radius_nm=args.restraint_r0_nm,
    )
    if spec is None:
        raise RuntimeError("select_co_alchemical_ion_once 返回 None——配体净电荷被判成 0，不应该发生")

    ion_indices = [int(ion["atom_index"]) for ion in spec["ions"]]
    ligand_coion_distances_nm = [
        _minimum_image_distance_nm(positions_nm[ligand_indices[0]], positions_nm[i], realized_box_nm)
        for i in ion_indices
    ]
    print(f"  📏 配体↔co-ion minimum-image 距离: "
          f"{[round(d, 3) for d in ligand_coion_distances_nm]} nm "
          f"(§13.1 要求初始 ≥ 1.6 nm)")

    # 5. 落盘。刻意不用 mmCIF 存正式身份——本体系只有单原子离子和标准水残基，没有
    #    §0.5.7 那种"非标准多原子残基丢键"的风险，但为了不给未来读者留疑问，位置/
    #    盒矢量单独存 .npy，topology 的 mmCIF 只当人可读的几何快照，不是身份来源。
    sys_xml_path = os.path.join(output_dir, "system.xml")
    with open(sys_xml_path, "w") as fh:
        fh.write(XmlSerializer.serialize(core.ensure_owned_system(system)))

    top_cif_path = os.path.join(output_dir, "topology.cif")
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, top_cif_path)

    positions_path = os.path.join(output_dir, "positions_nm.npy")
    np.save(positions_path, positions_nm)
    box_path = os.path.join(output_dir, "box_vectors_nm.npy")
    np.save(box_path, realized_box_nm)

    ligand_idx_path = os.path.join(output_dir, "ligand_indices.json")
    with open(ligand_idx_path, "w") as fh:
        json.dump({"ligand_indices": ligand_indices}, fh)

    spec_path = os.path.join(output_dir, "coalchemical_ion_spec.json")
    with open(spec_path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, cls=core.NumpyEncoder)

    residue_names = [str(r.name).upper() for r in modeller.topology.residues()]
    na_count = sum(n in {"NA", "NA+", "SOD"} for n in residue_names)
    cl_count = sum(n in {"CL", "CL-", "CLA"} for n in residue_names)
    # ⚠️ `all_atoms` 是 addSolvent 之后、dummy 插入之前捕获的快照——dummy 插入会
    # delete()+add() 重新编号拓扑，用那份快照的 index 去查 reserved_indices（那是
    # 插入*之后*的新编号）对应的原子身份，两套编号不是一回事。必须在插入之后重新
    # 从当前 `modeller.topology` 取。
    current_atoms = list(modeller.topology.atoms())
    reserved_na = sum(
        1 for i in reserved_indices
        if current_atoms[i].residue.name.upper() in {"NA", "NA+", "SOD"}
    )
    reserved_cl = sum(
        1 for i in reserved_indices
        if current_atoms[i].residue.name.upper() in {"CL", "CL-", "CLA"}
    )
    # §4.3 三类电荷来源不能混算：na_count/cl_count（残基名总数）同时包含了
    # reserved dummy *和*配体本身（配体在这里就是一个 Na⁺/Cl⁻ 原子，与普通盐离子
    # 共用同一个残基名模板）——"普通盐"计数必须把这两者都减掉，不只是减 dummy。
    ligand_is_na = template["residue_name"] == "NA"
    ligand_is_cl = template["residue_name"] == "CL"
    ordinary_na_count = na_count - reserved_na - (1 if ligand_is_na else 0)
    ordinary_cl_count = cl_count - reserved_cl - (1 if ligand_is_cl else 0)

    # fail closed：addSolvent(neutralize=True) 保证的是"配体+普通盐离子"这个子系统
    # 总电荷为 0（reserved dummy 是之后才插入、且已清零，不贡献电荷）。用独立的残基名
    # 计数重新核一遍这句话，专门用来抓"普通离子计数算错了但没人发现"这类 bug
    # ——就是刚才 self-test 里因为用了插入前的 stale 原子列表而抓到的那个。
    physical_total_e = q_l + ordinary_na_count * 1 + ordinary_cl_count * (-1)
    if abs(physical_total_e) > 1.0e-9:
        raise RuntimeError(
            f"普通离子计数与配体净电荷不自洽：ligand={q_l:+d}, "
            f"ordinary_na={ordinary_na_count}(+1 each), "
            f"ordinary_cl={ordinary_cl_count}(-1 each) ⟹ 物理体系总电荷算出 "
            f"{physical_total_e:+.9f} e，应严格为 0（reserved dummy 已清零，不贡献电荷；"
            "addSolvent(neutralize=True) 保证的就是这个子系统电中性）。"
            "这说明按残基名数普通离子的逻辑本身有 bug，不是数值误差。"
        )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "case": f"{args.ion}_{args.box_size}",
        "ion": args.ion,
        "ligand_net_charge_e": q_l,
        "box_size_label": args.box_size,
        "box_edge_nm": box_edge_nm,
        "nonbonded_cutoff_nm": NONBONDED_CUTOFF_NM,
        "ionic_strength_molar": ionic_strength_molar,
        "forcefield_xml": FORCEFIELD_XML,
        "n_atoms": system.getNumParticles(),
        "n_water": sum(1 for r in modeller.topology.residues() if r.name in ("HOH", "WAT")),
        "na_count_total": int(na_count),
        "cl_count_total": int(cl_count),
        "ordinary_na_count": int(ordinary_na_count),
        "ordinary_cl_count": int(ordinary_cl_count),
        "reserved_coion_indices": [int(i) for i in reserved_indices],
        "reserved_coion_cation": cation,
        "ligand_indices": ligand_indices,
        "co_ion_indices": ion_indices,
        "ligand_coion_min_image_distances_nm": ligand_coion_distances_nm,
        "charge_treatment": core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        "coalchemical_ion_fingerprint": spec.get("fingerprint"),
        "system_xml_sha256": _sha256_file(sys_xml_path),
        "topology_cif_sha256": _sha256_file(top_cif_path),
    }
    manifest_path = os.path.join(output_dir, "build_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"✅ build 完成: {output_dir}")
    print(f"   原子数={manifest['n_atoms']}, 水={manifest['n_water']}, "
          f"普通 Na={manifest['ordinary_na_count']}, 普通 Cl={manifest['ordinary_cl_count']}, "
          f"reserved dummy={len(reserved_indices)}")


# ============================================================================
# 子命令：static-check（纯 CPU，不建 Context）
# ============================================================================


def _load_build_artifacts(output_dir: str):
    with open(os.path.join(output_dir, "system.xml")) as fh:
        system = XmlSerializer.deserialize(fh.read())
    topology = app.PDBxFile(os.path.join(output_dir, "topology.cif")).topology
    positions_nm = np.load(os.path.join(output_dir, "positions_nm.npy"))
    box_nm = np.load(os.path.join(output_dir, "box_vectors_nm.npy"))
    with open(os.path.join(output_dir, "ligand_indices.json")) as fh:
        ligand_indices = json.load(fh)["ligand_indices"]
    with open(os.path.join(output_dir, "coalchemical_ion_spec.json")) as fh:
        spec = json.load(fh)
    with open(os.path.join(output_dir, "build_manifest.json")) as fh:
        manifest = json.load(fh)
    return system, topology, positions_nm, box_nm, ligand_indices, spec, manifest


def cmd_static_check(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    system, topology, positions_nm, box_nm, ligand_indices, spec, manifest = (
        _load_build_artifacts(output_dir)
    )
    q_l = int(manifest["ligand_net_charge_e"])
    box_vectors = tuple(Vec3(*row) for row in box_nm) * unit.nanometer

    # 1. 只读核对身份没有漂（[MEM-00c] 六个消费点之一，这里是"离线复判"）。
    pinned = core.verify_co_alchemical_ion_identity(
        spec,
        system=system,
        topology=topology,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=q_l,
        context="validate_charge_transfer_waterbox.static-check",
    )
    print(f"✅ verify_co_alchemical_ion_identity 通过，co-ion indices={list(pinned)}")

    # 2. 装配 lambda_coul 的 particle-parameter-offset（就地修改 system 的 NonbondedForce）。
    info = engine.configure_pme_ligand_charge_offsets(
        system,
        ligand_indices,
        lambda_name="lambda_coul",
        allow_charged_ligand=True,
        topology=topology,
        positions=positions_nm * unit.nanometer,
        box_vectors=box_vectors,
        co_alchemical_ion_spec=spec,
    )
    print(f"✅ configure_pme_ligand_charge_offsets: mode={info['mode']}, "
          f"n_offsets={info['n_offsets']}")

    nb_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    lambdas = [round(x, 2) for x in np.arange(1.0, -0.001, -0.1)]
    report = engine.charging_charge_conservation_report(
        nb_force,
        "lambda_coul",
        ligand_indices=ligand_indices,
        co_ion_indices=list(pinned),
        ligand_net_charge_e=q_l,
        lambdas=lambdas,
    )
    print("✅ 电荷守恒代数证明（Σq_scale = "
          f"{report['scale_sum_e']:+.3e} e，容差 {report['tolerance_e']:g} e）")
    for lam in lambdas:
        key = f"{lam:.6g}"
        print(f"     λ_coul={lam:.2f}  总电荷={report['total_charge_by_lambda_e'][key]:+.6f} e"
              f"  配体={report['ligand_charge_by_lambda_e'][key]:+.6f} e"
              f"  co-ion={report['co_ion_charge_by_lambda_e'][key]:+.6f} e")

    # 3. reserved dummy 在 λ=1 端严格中性、mass/LJ 与配体电荷无关（§2.2/§7.3）。
    for ion in spec["ions"]:
        idx = int(ion["atom_index"])
        q1 = float(ion["charge_at_lambda1_e"])
        if abs(q1) > core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            raise SystemExit(f"co-ion {idx} 在 λ=1 端不是严格中性: {q1:+.9f} e")
    print(f"✅ {len(spec['ions'])} 个 co-ion 在 λ=1 端严格中性")

    # 4. restraint 力组核对（§6.4 末条：restraint 不得混进任何 λ 相关分解）。
    restraint_group = core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP
    for ion in spec["ions"]:
        fg = int(ion["restraint"]["force_group"])
        if fg != restraint_group:
            raise SystemExit(f"co-ion {ion['atom_index']} restraint force group={fg}，应为 {restraint_group}")
    print(f"✅ 全部 co-ion restraint 声明的 force group = {restraint_group}")

    # 5. 单点能量自检（CPU、Reference/CPU platform，只做一次 getState，不做任何
    #    integrator.step()——这是"truly tiny isolated API-verification"级别的自检，
    #    不是本仓库规矩里要交给用户的那种采样/生产运行）。
    check_system = XmlSerializer.deserialize(XmlSerializer.serialize(system))
    engine._inject_co_alchemical_ion_restraints(check_system, spec)
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(check_system, integrator, platform)
    context.setPositions(positions_nm * unit.nanometer)
    context.setPeriodicBoxVectors(*box_vectors)
    energies = {}
    for lam in (1.0, 0.5, 0.0):
        context.setParameter("lambda_coul", lam)
        state = context.getState(getEnergy=True, getForces=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        max_force = float(np.max(np.linalg.norm(forces, axis=1)))
        if not np.isfinite(pe) or not np.isfinite(max_force):
            raise SystemExit(f"λ_coul={lam}: 能量或力出现非有限值 (PE={pe}, max|F|={max_force})")
        energies[lam] = {"potential_kj_mol": pe, "max_force_kj_mol_nm": max_force}
        print(f"   λ_coul={lam:.2f}: PE={pe:.3f} kJ/mol, max|F|={max_force:.3f} kJ/mol/nm")
    del context, integrator, check_system
    print("✅ CPU 单点能量/力自检：全部有限")

    result = {
        "case": manifest["case"],
        "identity_fingerprint": pinned,
        "charge_conservation_report": report,
        "endpoint_single_point_energies_cpu": energies,
        "ligand_coion_min_image_distances_nm": manifest["ligand_coion_min_image_distances_nm"],
        "passed": bool(report["total_charge_is_lambda_independent"]),
    }
    out_path = os.path.join(output_dir, "static_check_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, cls=core.NumpyEncoder)
    print(f"✅ static-check 完成，报告: {out_path}")


# ============================================================================
# 子命令：dynamics（需要 GPU/CUDA——按仓库规矩由用户在自己的计算节点提交，
# 本脚本本身只是"写代码"，不会在这次改动里被自动执行）
# ============================================================================


def _water_oxygen_indices(topology) -> List[int]:
    return [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name in ("HOH", "WAT") and atom.element == app.element.oxygen
    ]


def _box_volume_nm3(box_nm: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(box_nm, dtype=np.float64))))


def cmd_dynamics(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    system, topology, positions_nm, box_nm, ligand_indices, spec, manifest = (
        _load_build_artifacts(output_dir)
    )
    q_l = int(manifest["ligand_net_charge_e"])
    ion_indices = [int(ion["atom_index"]) for ion in spec["ions"]]
    box_vectors = tuple(Vec3(*row) for row in box_nm) * unit.nanometer

    lambdas_coul = [float(x) for x in args.lambda_coul.split(",")]
    if abs(lambdas_coul[0] - 1.0) > 1.0e-9 or abs(lambdas_coul[-1] - 0.0) > 1.0e-9:
        print(
            "⚠️  λ 表两端不是 1.0/0.0——不阻断，但请确认这是故意的"
            "（生产惯例是从物理态 λ=1 开始，逐步走向 λ=0 全去电荷）。"
        )

    # 1. 装配"prepared" system：restraint + barostat + lambda_coul offset 一次做完，
    #    随后 dynamics 与 ukn 阶段共用同一个 system_prepared.xml——保证离线重算用的
    #    哈密顿量与真正采样时的哈密顿量逐位一致（§7.2"生产者与校验者共用同一份真相"）。
    engine._inject_co_alchemical_ion_restraints(system, spec)
    system.addForce(
        openmm.MonteCarloBarostat(
            args.pressure_bar * unit.bar,
            args.temperature_kelvin * unit.kelvin,
            int(args.barostat_frequency),
        )
    )
    info = engine.configure_pme_ligand_charge_offsets(
        system,
        ligand_indices,
        lambda_name="lambda_coul",
        allow_charged_ligand=True,
        topology=topology,
        positions=positions_nm * unit.nanometer,
        box_vectors=box_vectors,
        co_alchemical_ion_spec=spec,
    )
    print(
        f"✅ prepared system: mode={info['mode']}, n_offsets={info['n_offsets']}, "
        f"barostat=MonteCarloBarostat({args.pressure_bar} bar, {args.temperature_kelvin} K, "
        f"每 {args.barostat_frequency} 步)"
    )

    prepared_xml_path = os.path.join(output_dir, "system_prepared.xml")
    with open(prepared_xml_path, "w") as fh:
        fh.write(XmlSerializer.serialize(core.ensure_owned_system(system)))

    # 逐 λ 电荷账目直接查代数表，同一 λ 态内不会变，不必每帧重新算。
    nb_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    charge_by_lambda = engine.charging_charge_conservation_report(
        nb_force,
        "lambda_coul",
        ligand_indices=ligand_indices,
        co_ion_indices=ion_indices,
        ligand_net_charge_e=q_l,
        lambdas=lambdas_coul,
    )

    def _new_integrator() -> openmm.LangevinMiddleIntegrator:
        integ = openmm.LangevinMiddleIntegrator(
            args.temperature_kelvin * unit.kelvin,
            args.friction_per_ps / unit.picosecond,
            args.timestep_ps * unit.picosecond,
        )
        integ.setRandomNumberSeed(int(args.seed))
        return integ

    # ⚠️ 一个 Integrator 一旦被绑进某个 Context 就不能复用给另一个 Context——如果
    # 首选 platform 在 `app.Simulation()` 内部走到"已经建了 Context 再失败"这一步
    # （不只是 `getPlatformByName` 就失败的那种早期错误），复用同一个 integrator
    # 对象去建第二个 Simulation 会出问题。所以 except 分支里重新造一个全新
    # integrator，不复用上面那个。
    platform_name = args.platform
    fallback_reason = None
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        properties = {"Precision": args.precision} if platform_name == "CUDA" else {}
        simulation = app.Simulation(topology, system, _new_integrator(), platform, properties)
    except Exception as exc:  # noqa: BLE001
        fallback_reason = str(exc)
        print(f"⚠️  平台 {platform_name} 不可用（{exc}），回退 CPU")
        platform_name = "CPU"
        platform = openmm.Platform.getPlatformByName("CPU")
        simulation = app.Simulation(topology, system, _new_integrator(), platform)

    simulation.context.setPositions(positions_nm * unit.nanometer)
    simulation.context.setPeriodicBoxVectors(*box_vectors)

    print(f"⚙️  最小化（maxIterations={args.n_steps_minimize}）...")
    simulation.minimizeEnergy(maxIterations=int(args.n_steps_minimize))
    state0 = simulation.context.getState(getEnergy=True)
    pe0 = state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(pe0):
        raise RuntimeError(f"最小化后势能非有限: {pe0}——起点坐标坏，不要继续往下跑")
    print(f"   最小化后 PE = {pe0:.3f} kJ/mol")

    dynamics_dir = os.path.join(output_dir, "dynamics")
    os.makedirs(dynamics_dir, exist_ok=True)
    csv_path = os.path.join(dynamics_dir, "timeseries.csv")
    csv_fields = [
        "lambda_state_index", "lambda_coul", "step", "time_ps",
        "total_charge_e", "ligand_charge_e", "coion_charge_e",
        "potential_kJ_mol", "max_force_kJ_mol_nm",
        "ligand_coion_distance_nm", "coion_water_coordination",
        "restraint_energy_kJ_mol", "box_volume_nm3",
    ]
    water_o = _water_oxygen_indices(topology)
    restraint_group = {core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP}

    dcd_paths: List[str] = []
    wall_clock_by_state: List[float] = []
    # §0.5.7 的教训：DCDFile 的 boxFlag 只在**构造时**按 topology 判一次；构造前必须
    # 先给 topology 一份非 None 的盒矢量，之后才能靠 writeModel(periodicBoxVectors=)
    # 逐帧覆盖成 NPT 下真实变化的盒。
    topology.setPeriodicBoxVectors(box_vectors)

    with open(csv_path, "w") as csv_fh:
        csv_fh.write(",".join(csv_fields) + "\n")
        for state_idx, lam in enumerate(lambdas_coul):
            t_state_start = time.time()
            simulation.context.setParameter("lambda_coul", float(lam))
            key = f"{float(lam):.6g}"
            total_q = charge_by_lambda["total_charge_by_lambda_e"][key]
            lig_q = charge_by_lambda["ligand_charge_by_lambda_e"][key]
            coion_q = charge_by_lambda["co_ion_charge_by_lambda_e"][key]

            print(
                f"— λ_coul={lam:.2f}（态 {state_idx + 1}/{len(lambdas_coul)}）："
                f"平衡 {args.n_steps_equil} 步..."
            )
            simulation.step(int(args.n_steps_equil))

            dcd_path = os.path.join(
                dynamics_dir, f"traj_state{state_idx:02d}_lam{lam:.2f}.dcd"
            )
            dcd_paths.append(dcd_path)
            n_chunks = int(args.n_steps_sample) // int(args.save_interval_steps)
            with open(dcd_path, "wb") as dcd_fh:
                dcd = app.DCDFile(
                    dcd_fh,
                    topology,
                    dt=args.timestep_ps * unit.picosecond,
                    interval=int(args.save_interval_steps),
                )
                for chunk in range(n_chunks):
                    simulation.step(int(args.save_interval_steps))
                    state = simulation.context.getState(
                        getPositions=True, getEnergy=True, getForces=True
                    )
                    pos_quantity = state.getPositions(asNumpy=True)
                    pos_nm_frame = pos_quantity.value_in_unit(unit.nanometer)
                    box_frame_vecs = state.getPeriodicBoxVectors()
                    box_frame_nm = np.array(
                        [v.value_in_unit(unit.nanometer) for v in box_frame_vecs],
                        dtype=np.float64,
                    )
                    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    forces = state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer
                    )
                    max_force = float(np.max(np.linalg.norm(forces, axis=1)))
                    if not np.isfinite(pe) or not np.isfinite(max_force):
                        raise RuntimeError(
                            f"λ_coul={lam}, 态内第 {chunk} 个保存点：能量/力出现非有限值 "
                            f"(PE={pe}, max|F|={max_force})——立即停止，不带着坏构型继续采样。"
                        )
                    restraint_state = simulation.context.getState(
                        getEnergy=True, groups=restraint_group
                    )
                    restraint_e = restraint_state.getPotentialEnergy().value_in_unit(
                        unit.kilojoule_per_mole
                    )
                    coion_dist = min(
                        _minimum_image_distance_nm(
                            pos_nm_frame[ligand_indices[0]], pos_nm_frame[i], box_frame_nm
                        )
                        for i in ion_indices
                    )
                    coord = sum(
                        1
                        for o in water_o
                        if _minimum_image_distance_nm(
                            pos_nm_frame[ion_indices[0]], pos_nm_frame[o], box_frame_nm
                        )
                        <= 0.32
                    )
                    step_now = (
                        state_idx * (int(args.n_steps_equil) + int(args.n_steps_sample))
                        + int(args.n_steps_equil)
                        + (chunk + 1) * int(args.save_interval_steps)
                    )
                    time_ps = step_now * args.timestep_ps
                    dcd.writeModel(pos_quantity, periodicBoxVectors=box_frame_vecs)
                    row = [
                        state_idx, f"{lam:.6f}", step_now, f"{time_ps:.3f}",
                        f"{total_q:.6f}", f"{lig_q:.6f}", f"{coion_q:.6f}",
                        f"{pe:.4f}", f"{max_force:.4f}",
                        f"{coion_dist:.4f}", coord,
                        f"{restraint_e:.4f}", f"{_box_volume_nm3(box_frame_nm):.4f}",
                    ]
                    csv_fh.write(",".join(str(v) for v in row) + "\n")
            csv_fh.flush()
            dt_state = time.time() - t_state_start
            wall_clock_by_state.append(dt_state)
            print(
                f"   ✅ 态 {state_idx + 1} 采样完成（{n_chunks} 帧），"
                f"耗时 {dt_state / 60.0:.1f} min"
            )

    dyn_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "case": manifest["case"],
        "lambdas_coul": lambdas_coul,
        "platform_requested": args.platform,
        "platform_used": platform_name,
        "platform_fallback_reason": fallback_reason,
        "seed": int(args.seed),
        "temperature_kelvin": args.temperature_kelvin,
        "pressure_bar": args.pressure_bar,
        "barostat_frequency": int(args.barostat_frequency),
        "timestep_ps": args.timestep_ps,
        "friction_per_ps": args.friction_per_ps,
        "n_steps_minimize": int(args.n_steps_minimize),
        "n_steps_equil_per_state": int(args.n_steps_equil),
        "n_steps_sample_per_state": int(args.n_steps_sample),
        "save_interval_steps": int(args.save_interval_steps),
        "post_minimization_potential_kJ_mol": float(pe0),
        "dcd_paths": dcd_paths,
        "timeseries_csv": csv_path,
        "system_prepared_xml": prepared_xml_path,
        "system_prepared_xml_sha256": _sha256_file(prepared_xml_path),
        "wall_clock_seconds_by_state": wall_clock_by_state,
        "wall_clock_seconds_total": float(sum(wall_clock_by_state)),
    }
    dyn_manifest_path = os.path.join(output_dir, "dynamics_manifest.json")
    with open(dyn_manifest_path, "w", encoding="utf-8") as fh:
        json.dump(dyn_manifest, fh, indent=2)
    print(f"✅ dynamics 完成，manifest: {dyn_manifest_path}")


# ============================================================================
# 子命令：ukn（CPU，但读取 GPU 产出的轨迹——本身可以在没有 GPU 的机器上跑）
# ============================================================================


def cmd_ukn(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    _system, topology, _pos, _box, ligand_indices, spec, manifest = _load_build_artifacts(
        output_dir
    )
    dyn_manifest_path = os.path.join(output_dir, "dynamics_manifest.json")
    if not os.path.exists(dyn_manifest_path):
        raise SystemExit(f"缺少 {dyn_manifest_path}——请先跑 dynamics 子命令")
    with open(dyn_manifest_path) as fh:
        dyn_manifest = json.load(fh)

    with open(dyn_manifest["system_prepared_xml"]) as fh:
        prepared_system = XmlSerializer.deserialize(fh.read())

    lambdas_coul = dyn_manifest["lambdas_coul"]
    # C1 只验证 charging（配体 q→0，co-ion 0→q）；vdW 全程保持耦合，不在这条腿里
    # 一并做 vanishing——那是 memtodolist §17.0 更靠后、独立的验证范围。
    lambdas_vdw = [1.0] * len(lambdas_coul)
    dcd_paths = dyn_manifest["dcd_paths"]
    for path in dcd_paths:
        if not os.path.exists(path):
            raise SystemExit(f"轨迹文件不存在: {path}——dynamics 是否已经跑完？")

    analyzer = engine.TraditionalMBARAnalyzer(temperature=dyn_manifest["temperature_kelvin"])
    u_kn = analyzer.compute_u_kn(
        traj_files=dcd_paths,
        system_template=prepared_system,
        ligand_indices=ligand_indices,
        lambdas_coul=lambdas_coul,
        lambdas_vdw=lambdas_vdw,
        platform_name=args.platform,
        topology=topology,
        co_alchemical_ion_spec=spec,
    )
    n_k = np.asarray(analyzer._last_n_k, dtype=int)
    if not np.all(np.isfinite(u_kn)):
        raise RuntimeError("u_kn 含 NaN/Inf——重算能量失败，先查轨迹是否有坏构型")

    result = analyzer.solve(u_kn, decorrelate=True)
    dg_kj = float(result["delta_G"])
    err_kj = float(result["error"])
    dg_kcal = dg_kj / 4.184
    err_kcal = err_kj / 4.184
    print(
        f"✅ charging ΔG(λ_coul: 1→0) = {dg_kj:.4f} ± {err_kj:.4f} kJ/mol "
        f"= {dg_kcal:.4f} ± {err_kcal:.4f} kcal/mol  "
        f"(method={result.get('method')}, converged={result.get('converged')}, "
        f"min_overlap={result.get('min_overlap')})"
    )

    u_kn_path = os.path.join(output_dir, "u_kn.npz")
    np.savez(
        u_kn_path,
        u_kn=u_kn,
        n_k=n_k,
        lambdas_coul=np.asarray(lambdas_coul, dtype=float),
        lambdas_vdw=np.asarray(lambdas_vdw, dtype=float),
        temperature_kelvin=dyn_manifest["temperature_kelvin"],
        beta=analyzer.beta,
        coion_fingerprint=spec.get("fingerprint", ""),
        system_sha256=dyn_manifest["system_prepared_xml_sha256"],
    )

    dg_result = {
        "case": manifest["case"],
        "delta_G_charging_kJ_mol": dg_kj,
        "uncertainty_kJ_mol": err_kj,
        "delta_G_charging_kcal_mol": dg_kcal,
        "uncertainty_kcal_mol": err_kcal,
        "method": result.get("method"),
        "converged": result.get("converged"),
        "min_overlap": result.get("min_overlap"),
        "n_states": result.get("n_states"),
        "n_frames": result.get("n_frames"),
        "u_kn_path": u_kn_path,
    }
    dg_path = os.path.join(output_dir, "charging_delta_G.json")
    with open(dg_path, "w", encoding="utf-8") as fh:
        json.dump(dg_result, fh, indent=2, cls=core.NumpyEncoder)
    print(f"✅ ukn 完成: {u_kn_path}, {dg_path}")


# ============================================================================
# 子命令：report / compare-box（纯 CPU，汇总已有产物）
# ============================================================================


def cmd_report(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    with open(os.path.join(output_dir, "build_manifest.json")) as fh:
        build_manifest = json.load(fh)
    with open(os.path.join(output_dir, "static_check_report.json")) as fh:
        static_report = json.load(fh)

    dg_path = os.path.join(output_dir, "charging_delta_G.json")
    dg_result = None
    if os.path.exists(dg_path):
        with open(dg_path) as fh:
            dg_result = json.load(fh)

    dyn_manifest_path = os.path.join(output_dir, "dynamics_manifest.json")
    dyn_manifest = None
    if os.path.exists(dyn_manifest_path):
        with open(dyn_manifest_path) as fh:
            dyn_manifest = json.load(fh)

    checks = {
        # static-check/dynamics 任一处非有限值都已经在那一步 raise、不会走到这里；
        # 这两项存在于 checks 里只是让"跑到 report 就说明它们全过了"这句话写死可查。
        "charge_conservation_passed": bool(static_report["passed"]),
        "finite_energy_force_passed": True,
        "coion_geometry_initial_passed": all(
            d >= core.COION_LIGAND_MIN_IMAGE_INITIAL_NM
            for d in build_manifest["ligand_coion_min_image_distances_nm"]
        ),
        "restraint_force_group_passed": True,
        "endpoint_charge_passed": True,
        "dynamics_completed": dyn_manifest is not None,
        "ukn_completed": dg_result is not None,
    }
    passed = all(checks.values())

    report = {
        "case": build_manifest["case"],
        "ion": build_manifest["ion"],
        "ligand_net_charge_e": build_manifest["ligand_net_charge_e"],
        "box_size_label": build_manifest["box_size_label"],
        "box_edge_nm": build_manifest["box_edge_nm"],
        "build_manifest": build_manifest,
        "static_check_report": static_report,
        "dynamics_manifest": dyn_manifest,
        "charging_delta_G": dg_result,
        "checks": checks,
        "passed": passed,
        "failure_reasons": [k for k, v in checks.items() if not v],
    }
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, cls=core.NumpyEncoder)

    summary = {
        "case": build_manifest["case"],
        "passed": passed,
        **checks,
        "delta_G_charging_kJ_mol": dg_result["delta_G_charging_kJ_mol"] if dg_result else None,
        "uncertainty_kJ_mol": dg_result["uncertainty_kJ_mol"] if dg_result else None,
        "failure_reasons": report["failure_reasons"],
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"{'✅' if passed else '❌'} report 完成: {report_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_compare_box(args: argparse.Namespace) -> None:
    """§13.4/C1 硬验收：同一电荷符号的小/大盒 charging ΔG 差必须同时满足
    `|ΔΔG| ≤ 2σ_combined` 与 `|ΔΔG| ≤ 1.0 kcal/mol`（两条都要满足，不是任一条）。
    """
    with open(args.small_report) as fh:
        small = json.load(fh)
    with open(args.large_report) as fh:
        large = json.load(fh)
    if small.get("charging_delta_G") is None or large.get("charging_delta_G") is None:
        raise SystemExit("两份 report.json 都必须已经跑完 ukn（含 charging_delta_G）")

    dg_small = float(small["charging_delta_G"]["delta_G_charging_kcal_mol"])
    err_small = float(small["charging_delta_G"]["uncertainty_kcal_mol"])
    dg_large = float(large["charging_delta_G"]["delta_G_charging_kcal_mol"])
    err_large = float(large["charging_delta_G"]["uncertainty_kcal_mol"])

    ddg = dg_large - dg_small
    combined_sigma = float(np.sqrt(err_small ** 2 + err_large ** 2))
    threshold_2sigma = 2.0 * combined_sigma
    threshold_abs_kcal_mol = 1.0
    passed = (abs(ddg) <= threshold_2sigma) and (abs(ddg) <= threshold_abs_kcal_mol)

    result = {
        "small_case": small.get("case"),
        "large_case": large.get("case"),
        "small_box_edge_nm": small.get("box_edge_nm"),
        "large_box_edge_nm": large.get("box_edge_nm"),
        "delta_G_small_kcal_mol": dg_small,
        "uncertainty_small_kcal_mol": err_small,
        "delta_G_large_kcal_mol": dg_large,
        "uncertainty_large_kcal_mol": err_large,
        "delta_delta_G_kcal_mol": ddg,
        "combined_sigma_kcal_mol": combined_sigma,
        "threshold_2sigma_kcal_mol": threshold_2sigma,
        "threshold_abs_kcal_mol": threshold_abs_kcal_mol,
        "passed": passed,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"✅ compare-box 结果已写入 {args.output}")


# ============================================================================
# CLI
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_build = sub.add_parser("build", help="构建小水盒 + reserved co-ion dummy + 冻结身份（纯 CPU）")
    p_build.add_argument("--ion", required=True, choices=sorted(ION_TEMPLATES))
    p_build.add_argument("--box-size", choices=sorted(BOX_EDGE_NM), default="small")
    p_build.add_argument("--box-edge-nm", type=float, default=None, help="覆盖 --box-size 的默认盒长")
    p_build.add_argument("--ionic-strength-molar", type=float, default=DEFAULT_IONIC_STRENGTH_MOLAR)
    p_build.add_argument("--restraint-k", type=float, default=None, help="覆盖 §13.1 默认 k（kJ/mol/nm²）")
    p_build.add_argument("--restraint-r0-nm", type=float, default=None, help="覆盖 §13.1 默认平坦区半径")
    p_build.add_argument("--output-dir", required=True)
    p_build.set_defaults(func=cmd_build)

    p_check = sub.add_parser("static-check", help="逐 λ 电荷守恒 + 几何/参数一致性（纯 CPU）")
    p_check.add_argument("--output-dir", required=True)
    p_check.set_defaults(func=cmd_static_check)

    default_lambda_coul = "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0"

    p_dyn = sub.add_parser(
        "dynamics",
        help="逐 λ 短平衡+采样，写 DCD/timeseries.csv（需要 GPU/CUDA，交给用户在计算节点跑）",
    )
    p_dyn.add_argument("--output-dir", required=True)
    p_dyn.add_argument("--lambda-coul", default=default_lambda_coul, help="逗号分隔的 λ_coul 表，默认 1→0 共 11 点")
    p_dyn.add_argument("--n-steps-minimize", type=int, default=5000)
    p_dyn.add_argument("--n-steps-equil", type=int, default=20000, help="每个 λ 态的平衡步数（不落轨迹）")
    p_dyn.add_argument("--n-steps-sample", type=int, default=50000, help="每个 λ 态的采样步数（落轨迹）")
    p_dyn.add_argument("--save-interval-steps", type=int, default=500)
    p_dyn.add_argument("--temperature-kelvin", type=float, default=DEFAULT_TEMPERATURE_K)
    p_dyn.add_argument("--pressure-bar", type=float, default=1.0)
    p_dyn.add_argument("--barostat-frequency", type=int, default=25)
    p_dyn.add_argument("--friction-per-ps", type=float, default=1.0)
    p_dyn.add_argument("--timestep-ps", type=float, default=0.002)
    p_dyn.add_argument("--seed", type=int, default=2026)
    p_dyn.add_argument("--platform", default="CUDA", choices=["CUDA", "CPU", "OpenCL", "Reference"])
    p_dyn.add_argument("--precision", default="mixed", help="仅 --platform CUDA 时生效")
    p_dyn.set_defaults(func=cmd_dynamics)

    p_ukn = sub.add_parser("ukn", help="从 dynamics 的轨迹重算 u_kn 并用 MBAR 求 charging ΔG（CPU）")
    p_ukn.add_argument("--output-dir", required=True)
    p_ukn.add_argument("--platform", default="CPU", help="重算能量用的 platform（离线复判默认 CPU）")
    p_ukn.set_defaults(func=cmd_ukn)

    p_report = sub.add_parser("report", help="汇总 build/static-check/dynamics/ukn 为 report.json + summary.json")
    p_report.add_argument("--output-dir", required=True)
    p_report.set_defaults(func=cmd_report)

    p_cmp = sub.add_parser("compare-box", help="比较同一电荷符号的小/大盒两份 report.json，判 §13.4 盒长敏感性")
    p_cmp.add_argument("--small-report", required=True)
    p_cmp.add_argument("--large-report", required=True)
    p_cmp.add_argument("--output", default=None, help="可选：把比较结果另存一份 JSON")
    p_cmp.set_defaults(func=cmd_compare_box)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

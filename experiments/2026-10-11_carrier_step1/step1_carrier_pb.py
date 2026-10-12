#!/usr/bin/env python
"""第 1 步：carrier 在约束位点 vs 无限远充电之差的 PB 估计（只读运行目录，后处理）。

设计稿：docs/design/PROPOSAL_charge_transfer_carrier_soluble_2026-10-11.md §3.3–§3.4、§4 第 1 步。

每条腿、每帧算两项（carrier 电荷 q_X，配体电荷一律置 0 —— 拆成"先去配体电荷、
再给 carrier 充电"这条路径时，carrier 充电那一段配体是中性的）：

  E_int    = q_X · φ_env(r_X)：环境（蛋白，含 R_CUT 以内的周期像）在 carrier 处的
             PB 电势，连续介质盐浓度取模拟里的盐对数；
  E_desolv = E(carrier 带电，全部溶质作空腔) − E(carrier 单独在溶剂里)。

ΔU_X = E_int + E_desolv；线性响应 δ_leg = −½(⟨ΔU_X⟩_{λ=1} + ⟨ΔU_X⟩_{λ=0})，
ΔG_bind 修正 = δ_solv − δ_cplx（ΔG_bind = ΔG_solv − ΔG_cplx，两腿同为 耦合→去耦）。

周期项（NET/RIP/DSC）不在这里：它们都正比于盒的净电荷，而两条腿每个 λ 都是
电中性的（system XML 总电荷 0，且逐 λ 电荷守恒由 ibs_engine 校验）。

用法：
    python step1_carrier_pb.py <rep_dir> --leg complex|solvent [--stride 5] [--workers 20]
           [--max-frames N] [--out-dir DIR]
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from openmm import NonbondedForce, HarmonicBondForce, XmlSerializer, app

APBS = "/home/ruigengji/software/APBS-3.4.1.Linux/bin/apbs"
KT_KJ = 0.00831446261815324 * 300.0
EPS_SOLVENT_TIP3P = 97.0      # 与 apbs_correction.WATER_MODELS["tip3p"] 一致
PDIE = 1.0                    # 与 apbs_correction.ROCKLIN_SOLUTE_DIELECTRIC 一致
R_CUT_NM = 3.0                # 约 4 个 Debye 长度（0.15 M 时 κ⁻¹≈0.78 nm）
WATER_ION = {"HOH", "WAT", "NA", "CL", "K", "SOD", "CLA"}
WATERS_PER_LITER_M = 55.5


def _radius_a(sigma_nm):
    # AMBER 风格 PB 半径 = Rmin/2 = σ·2^(1/6)/2（PDB2PQR 的 AMBER 半径即此口径）
    return float(sigma_nm) * 2.0 ** (1.0 / 6.0) / 2.0 * 10.0


def _load_leg(rep_dir, leg):
    leg_dir = rep_dir if leg == "complex" else os.path.join(rep_dir, "solvent_leg")
    cif = os.path.join(rep_dir, "topology.cif" if leg == "complex" else "topology_solvent.cif")
    xml = os.path.join(rep_dir, "system_native.xml" if leg == "complex" else "system_solvent.xml")
    lig_json = os.path.join(rep_dir, "ligand_indices.json" if leg == "complex" else "ligand_indices_solvent.json")
    spec = json.load(open(os.path.join(leg_dir, "checkpoints", "coalchemical_ion_spec.json")))
    ion = spec["ions"][0]
    i_x = int(ion["atom_index"])
    q_x = float(ion["charge_at_lambda0_e"])
    lig = json.load(open(lig_json))
    lig = [int(i) for i in (lig["ligand_indices"] if isinstance(lig, dict) else lig)]
    top = app.PDBxFile(cif).topology
    system = XmlSerializer.deserialize(open(xml).read())
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    atoms = list(top.atoms())
    solute = [a.index for a in atoms if a.residue.name not in WATER_ION]   # 蛋白 + 配体
    lig_set = set(lig)
    q = np.zeros(len(atoms))
    r = np.zeros(len(atoms))
    for a in atoms:
        c, s, _e = nb.getParticleParameters(a.index)
        q[a.index] = c._value
        r[a.index] = _radius_a(s._value)
    for i in lig:
        q[i] = 0.0
    # 分子归属（拼回整分子、做周期像都按分子走）
    bonds = next(f for f in system.getForces() if isinstance(f, HarmonicBondForce))
    parent = list(range(len(atoms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in range(bonds.getNumBonds()):
        i, j, *_ = bonds.getBondParameters(k)
        parent[find(i)] = find(j)
    for k in range(system.getNumConstraints()):
        i, j, _d = system.getConstraintParameters(k)
        parent[find(i)] = find(j)
    mols = {}
    for i in solute:
        mols.setdefault(find(i), []).append(i)
    n_w = sum(1 for res in top.residues() if res.name in ("HOH", "WAT"))
    n_na = sum(1 for res in top.residues() if res.name in ("NA", "SOD", "K", "POT"))
    n_cl = sum(1 for res in top.residues() if res.name in ("CL", "CLA")) - 1   # 去掉 carrier
    salt_m = min(n_na, n_cl) / (n_w / WATERS_PER_LITER_M)
    q_prot = float(sum(q[i] for i in solute if i not in lig_set))
    return {
        "leg": leg, "leg_dir": leg_dir, "cif": cif, "i_x": i_x, "q_x": q_x, "lig": lig,
        "solute": solute, "mols": list(mols.values()), "q": q, "r": r, "top": top,
        "salt_M": salt_m, "q_protein_e": q_prot, "n_water": n_w, "n_na": n_na, "n_cl_excl_carrier": n_cl,
    }


def _min_image(v, box):
    inv = np.linalg.inv(box)
    f = v @ inv
    f -= np.round(f)
    return f @ box


def _make_whole(xyz, mol, box):
    ref = xyz[mol[0]]
    xyz[mol] = ref + _min_image(xyz[mol] - ref, box)


def _frame_atoms(L, xyz, box):
    """返回 (coords_nm, q, r, tag) —— 以 carrier 为中心，R_CUT 内的溶质整分子周期像 + carrier。"""
    xyz = xyz.copy()
    for mol in L["mols"]:
        _make_whole(xyz, mol, box)
    x_x = xyz[L["i_x"]]
    shifts = [np.array([a, b, c]) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    coords, qs, rs = [x_x[None, :]], [np.array([L["q_x"]])], [np.array([L["r"][L["i_x"]]])]
    tags = [np.array(["X"])]
    n_copies = 0
    for mol in L["mols"]:
        base = xyz[mol]
        # 先把整分子平移到离 carrier 最近的那个像，再枚举 ±1 盒
        c0 = base.mean(axis=0)
        base = base + (_min_image(c0 - x_x, box) - (c0 - x_x))
        for s in shifts:
            img = base + s @ box
            if np.min(np.linalg.norm(img - x_x, axis=1)) < R_CUT_NM:
                coords.append(img)
                qs.append(L["q"][mol])
                rs.append(L["r"][mol])
                tags.append(np.array(["S"] * len(mol)))
                n_copies += 1
    return np.vstack(coords), np.concatenate(qs), np.concatenate(rs), np.concatenate(tags), n_copies


def _write_pqr(path, coords_nm, q, r):
    with open(path, "w") as fh:
        for k, (p, c, rad) in enumerate(zip(coords_nm * 10.0, q, r)):
            fh.write(f"ATOM  {k % 100000:5d}  X   MOL A   1    {p[0]:9.4f} {p[1]:9.4f} {p[2]:9.4f} {c:9.5f} {rad:7.4f}\n")
        fh.write("END\n")


def _elec(name, mol, cglen, cgcent, fglen, fgcent, salt_m, energy, write_pot):
    lines = [f"elec name {name}", "  mg-auto", "  dime 129 129 129",
             "  cglen " + " ".join(f"{v:.3f}" for v in cglen),
             "  fglen " + " ".join(f"{v:.3f}" for v in fglen),
             "  cgcent " + " ".join(f"{v:.4f}" for v in cgcent),
             "  fgcent " + " ".join(f"{v:.4f}" for v in fgcent),
             f"  mol {mol}", "  lpbe", "  bcfl mdh"]
    if salt_m > 0:
        lines += [f"  ion charge 1 conc {salt_m:.5f} radius 2.0", f"  ion charge -1 conc {salt_m:.5f} radius 2.0"]
    lines += [f"  pdie {PDIE}", f"  sdie {EPS_SOLVENT_TIP3P}", "  chgm spl4", "  srfm smol", "  srad 1.4",
              "  swin 0.3", "  sdens 10.0", "  temp 300.0",
              f"  calcenergy {'total' if energy else 'no'}", "  calcforce no"]
    if write_pot:
        lines.append(f"  write pot dx {name}")
    lines.append("end")
    return "\n".join(lines)


def _read_dx_at(path, point_a):
    with open(path) as fh:
        txt = fh.read()
    counts = tuple(int(v) for v in re.search(r"gridpositions counts\s+(\d+)\s+(\d+)\s+(\d+)", txt).groups())
    origin = np.array([float(v) for v in re.search(r"origin\s+(\S+)\s+(\S+)\s+(\S+)", txt).groups()])
    deltas = [float(m.group(i + 1)) for i, m in enumerate(re.finditer(r"delta\s+(\S+)\s+(\S+)\s+(\S+)", txt))]
    h = np.array(deltas[:3])            # 对角网格：第 i 条 delta 取第 i 个分量
    start = re.search(r"data follows\s*", txt).end()
    end = txt.find("attribute", start)
    vals = np.array(txt[start:end].split(), dtype=float)[: counts[0] * counts[1] * counts[2]]
    grid = vals.reshape(counts)         # OpenDX：z 最快
    u = (np.asarray(point_a) - origin) / h
    i0 = np.floor(u).astype(int)
    t = u - i0
    acc = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (t[0] if dx else 1 - t[0]) * (t[1] if dy else 1 - t[1]) * (t[2] if dz else 1 - t[2])
                acc += w * grid[i0[0] + dx, i0[1] + dy, i0[2] + dz]
    return float(acc)


def _parse_print_energies(log):
    # print elecEnergy 结果行："  Global net ELEC energy = 1.234E+02 kJ/mol"
    return [float(v) for v in re.findall(r"Global net ELEC energy =\s*([-+0-9.Ee]+)\s*kJ/mol", log)]


def _job(args):
    L_small, label, coords, q, r, tags, n_copies, keep = args
    # APBS 的网格文件（每个 ~30 MB）与 io.mc 写在本机 scratch，读完即删；
    # 40 个进程同时往 NFS 写会把共享盘拖住（10-11 第一次全量就这样卡了 11 分钟）。
    work = os.path.join(L_small["scratch"], label)
    os.makedirs(work, exist_ok=True)
    os.makedirs(keep, exist_ok=True)
    is_x = tags == "X"
    x_pos_a = coords[is_x][0] * 10.0
    q_env = np.where(is_x, 0.0, q)
    q_xonly = np.where(is_x, q, 0.0)
    _write_pqr(os.path.join(work, "env.pqr"), coords, q_env, r)
    _write_pqr(os.path.join(work, "xcav.pqr"), coords, q_xonly, r)
    _write_pqr(os.path.join(work, "xbulk.pqr"), coords[is_x], q[is_x], r[is_x])
    span = coords.max(axis=0) * 10.0 - coords.min(axis=0) * 10.0
    cglen = np.maximum(span + 30.0, 60.0)
    cgcent = (coords.max(axis=0) + coords.min(axis=0)) * 5.0
    fglen = np.array([24.0, 24.0, 24.0])
    salt = L_small["salt_M"]
    blocks = ["read\n  mol pqr env.pqr\n  mol pqr xcav.pqr\n  mol pqr xbulk.pqr\nend"]
    has_env = bool(np.any(np.abs(q_env) > 0))
    if has_env:
        blocks.append(_elec("envpot", 1, cglen, cgcent, fglen, x_pos_a, salt, False, True))
        blocks.append(_elec("envpot0", 1, cglen, cgcent, fglen, x_pos_a, 0.0, False, True))
    blocks.append(_elec("xcav", 2, cglen, cgcent, fglen, x_pos_a, salt, True, False))
    blocks.append(_elec("xbulk", 3, cglen, cgcent, fglen, x_pos_a, salt, True, False))
    blocks.append("print elecEnergy xcav end")
    blocks.append("print elecEnergy xbulk end")
    blocks.append("quit\n")
    with open(os.path.join(work, "apbs.in"), "w") as fh:
        fh.write("\n\n".join(blocks))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    p = subprocess.run([APBS, "apbs.in"], cwd=work, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=env)
    with open(os.path.join(work, "apbs.log"), "w") as fh:
        fh.write(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"APBS failed: {work}")
    e_cav, e_bulk = _parse_print_energies(p.stdout)[-2:]
    out = {"label": label, "n_solute_copies": int(n_copies), "n_atoms": int(len(q)),
           "E_desolv_kJ": e_cav - e_bulk}
    qx = float(q[is_x][0])
    if has_env:
        for name, key in (("envpot", "E_int_kJ"), ("envpot0", "E_int_0M_kJ")):
            dx = os.path.join(work, f"{name}.dx")
            phi = _read_dx_at(dx, x_pos_a)        # kT/e
            out[key] = qx * phi * KT_KJ
            os.remove(dx)                         # 每个 ~30 MB，读完即删
    else:
        out["E_int_kJ"] = 0.0
        out["E_int_0M_kJ"] = 0.0
    for name in ("apbs.in", "apbs.log"):          # 只把小文件留在项目目录
        shutil.copyfile(os.path.join(work, name), os.path.join(keep, name))
    shutil.rmtree(work)
    return out


def _frames(L, state):
    import mdtraj as md
    dcd = os.path.join(L["leg_dir"], "decharging", f"decharging_rep{state}.dcd")
    t = md.load_dcd(dcd, top=md.Topology.from_openmm(L["top"]))
    return t.xyz.astype(float), t.unitcell_vectors.astype(float)


def _stats(v):
    v = np.asarray(v, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "sem_naive": float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None, "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rep_dir")
    ap.add_argument("--leg", choices=("complex", "solvent"), required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--scratch", required=True, help="本机本地盘上的临时目录（放 APBS 网格，跑完即删）")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = a.out_dir or os.path.join(here, a.leg)
    os.makedirs(out_dir, exist_ok=True)
    L = _load_leg(os.path.abspath(a.rep_dir), a.leg)
    meta = json.load(open(os.path.join(L["leg_dir"], "decharging", "decharging_sampling.meta.json")))
    n_states = len(meta["sampling_fingerprint"]["payload"]["lambdas_coul"])
    small = {"salt_M": L["salt_M"], "scratch": os.path.join(os.path.abspath(a.scratch), a.leg)}
    jobs = []
    for state in (0, n_states - 1):
        xyz, boxes = _frames(L, state)
        idx = list(range(0, len(xyz), a.stride))
        if a.max_frames:
            idx = idx[: a.max_frames]
        for f in idx:
            coords, q, r, tags, nc = _frame_atoms(L, xyz[f], boxes[f])
            label = f"state{state}_frame{f:04d}"
            jobs.append((small, label, coords, q, r, tags, nc, os.path.join(out_dir, label)))
    with Pool(a.workers) as pool:
        rows = pool.map(_job, jobs)
    by_state = {}
    for row in rows:
        by_state.setdefault(row["label"].split("_")[0], []).append(row)
    summary = {"leg": a.leg, "rep_dir": os.path.abspath(a.rep_dir), "salt_M": L["salt_M"],
               "q_protein_e": L["q_protein_e"], "q_carrier_e": L["q_x"], "pdie": PDIE,
               "sdie": EPS_SOLVENT_TIP3P, "r_cut_nm": R_CUT_NM, "stride": a.stride, "states": {}}
    du_means = []
    for st, rr in sorted(by_state.items()):
        du = [x["E_int_kJ"] + x["E_desolv_kJ"] for x in rr]
        du0 = [x["E_int_0M_kJ"] + x["E_desolv_kJ"] for x in rr]
        summary["states"][st] = {
            "E_int_kJ": _stats([x["E_int_kJ"] for x in rr]),
            "E_int_0M_kJ": _stats([x["E_int_0M_kJ"] for x in rr]),
            "E_desolv_kJ": _stats([x["E_desolv_kJ"] for x in rr]),
            "dU_X_kJ": _stats(du), "dU_X_0M_kJ": _stats(du0),
            "n_solute_copies": sorted({x["n_solute_copies"] for x in rr}),
        }
        du_means.append((np.mean(du), np.std(du, ddof=1) / math.sqrt(len(du)) if len(du) > 1 else 0.0,
                         np.mean(du0)))
    if len(du_means) == 2:
        delta = -0.5 * (du_means[0][0] + du_means[1][0])
        err = 0.5 * math.hypot(du_means[0][1], du_means[1][1])
        summary["delta_leg_kJ"] = float(delta)
        summary["delta_leg_sem_naive_kJ"] = float(err)
        summary["delta_leg_0M_kJ"] = float(-0.5 * (du_means[0][2] + du_means[1][2]))
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump({"summary": summary, "frames": rows}, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

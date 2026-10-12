#!/usr/bin/env python
"""第 0 步：rep1 里 carrier（charge-transfer 的 co-ion）离蛋白多近、撞不撞墙（只读）。

设计稿：docs/design/PROPOSAL_charge_transfer_carrier_soluble_2026-10-11.md §4 第 0 步。
只读运行目录，不写回；结果写到本脚本所在目录。

用法：
    python step0_carrier_geometry.py <rep_dir> [out_json]
"""
import glob
import json
import os
import sys

import mdtraj as md
import numpy as np
from openmm import app

KT_300 = 0.00831446261815324 * 300.0
CATIONIC_N = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
CONTACT_NM = 0.4


def _min_image(vec, box):
    """vec: (F,3)，box: (F,3,3) 盒矩阵（行向量）；三斜也成立的分数坐标取整。"""
    inv = np.linalg.inv(box)
    frac = np.einsum("fi,fij->fj", vec, inv)
    frac -= np.round(frac)
    return np.einsum("fj,fji->fi", frac, box)


def _min_dist(x_ion, xyz_targets, box):
    """x_ion: (F,3)；xyz_targets: (F,N,3) → (F,) 最小 minimum-image 距离。"""
    d = xyz_targets - x_ion[:, None, :]
    F, N, _ = d.shape
    d = _min_image(d.reshape(F * N, 3), np.repeat(box, N, axis=0)).reshape(F, N, 3)
    return np.linalg.norm(d, axis=2).min(axis=1)


def _summarize(x_ion, x_anchor, box, spec_r, prot_xyz=None, cat_xyz=None, lig_xyz=None):
    d0 = np.asarray(spec_r["reference_displacement_nm"], dtype=float)
    k = float(spec_r["k_kj_per_mol_nm2"])
    r0 = float(spec_r["flat_bottom_radius_nm"])
    dev = np.linalg.norm(_min_image(x_ion - x_anchor - d0, box), axis=1)
    e = 0.5 * k * np.clip(dev - r0, 0.0, None) ** 2
    out = {
        "n_frames": int(len(dev)),
        "well_offset_nm": {"median": float(np.median(dev)), "p95": float(np.percentile(dev, 95)),
                           "max": float(dev.max())},
        "frac_outside_flat_bottom": float(np.mean(dev > r0)),
        "restraint_energy_kT": {"mean": float(e.mean() / KT_300), "max": float(e.max() / KT_300)},
    }
    if prot_xyz is not None:
        dp = _min_dist(x_ion, prot_xyz, box)
        out["min_dist_protein_heavy_nm"] = {"min": float(dp.min()), "p05": float(np.percentile(dp, 5)),
                                            "median": float(np.median(dp))}
    if cat_xyz is not None:
        dc = _min_dist(x_ion, cat_xyz, box)
        out["min_dist_lys_arg_N_nm"] = {"min": float(dc.min()), "p05": float(np.percentile(dc, 5)),
                                        "median": float(np.median(dc))}
        out[f"frac_contact_lys_arg_N_lt_{CONTACT_NM}nm"] = float(np.mean(dc < CONTACT_NM))
    if lig_xyz is not None:
        dl = _min_dist(x_ion, lig_xyz, box)
        out["min_dist_ligand_nm"] = {"min": float(dl.min()), "median": float(np.median(dl))}
    return out


def _leg(rep_dir, leg_dir, cif, is_complex):
    spec = json.load(open(os.path.join(leg_dir, "checkpoints", "coalchemical_ion_spec.json")))
    ion = spec["ions"][0]
    i_ion = int(ion["atom_index"])
    r = ion["restraint"]
    i_anchor = int(r["anchor_atom_index"])
    lig = json.load(open(os.path.join(rep_dir, "ligand_indices.json" if is_complex else "ligand_indices_solvent.json")))
    lig = [int(i) for i in (lig if isinstance(lig, list) else lig.get("ligand_indices", lig))]
    top_mm = app.PDBxFile(cif).topology
    atoms = list(top_mm.atoms())
    skip = {"HOH", "WAT", "NA", "CL", "K", "SOD", "CLA"}
    prot_heavy = [a.index for a in atoms if a.residue.name not in skip and a.index not in set(lig)
                  and a.element is not None and a.element.symbol != "H"]
    cationic = [a.index for a in atoms if (a.residue.name, a.name) in CATIONIC_N]
    res = {"carrier_index": i_ion, "anchor_index": i_anchor, "n_protein_heavy": len(prot_heavy),
           "n_lys_arg_N": len(cationic), "decharging": {}, "vanishing": {}}

    # stage 1：REMD 每个 λ 态一个 DCD（rep{state}）；只取 λ_coul=1 与 0 两端。
    meta = json.load(open(os.path.join(leg_dir, "decharging", "decharging_sampling.meta.json")))
    lams = meta["sampling_fingerprint"]["payload"]["lambdas_coul"]
    md_top = md.Topology.from_openmm(top_mm)
    sel = sorted(set([i_ion, i_anchor] + prot_heavy + cationic + lig))
    pos = {a: j for j, a in enumerate(sel)}
    for s in (0, len(lams) - 1):
        dcd = os.path.join(leg_dir, "decharging", f"decharging_rep{s}.dcd")
        t = md.load_dcd(dcd, top=md_top, atom_indices=sel)
        box = t.unitcell_vectors.astype(float)
        X = t.xyz.astype(float)
        res["decharging"][f"state{s}_lam_coul={lams[s]:.3g}"] = _summarize(
            X[:, pos[i_ion]], X[:, pos[i_anchor]], box, r,
            prot_xyz=X[:, [pos[a] for a in prot_heavy]] if prot_heavy else None,
            cat_xyz=X[:, [pos[a] for a in cationic]] if cationic else None,
            lig_xyz=X[:, [pos[a] for a in lig]])

    # stage 2：carrier 全程带电；坐标旁车只存子集原子。
    for seg_dir in sorted(glob.glob(os.path.join(leg_dir, "vanishing*"))):
        for f in sorted(glob.glob(os.path.join(seg_dir, "dual_window_*_vdw_coords.npz"))):
            z = np.load(f)
            ai = {int(a): j for j, a in enumerate(z["atom_indices"])}
            if i_ion not in ai or i_anchor not in ai:
                continue
            X = z["xyz"].astype(float)
            box = z["box"].astype(float)
            ph = [ai[a] for a in prot_heavy if a in ai]
            ca = [ai[a] for a in cationic if a in ai]
            key = f"{os.path.basename(seg_dir)}/{os.path.basename(f).replace('_vdw_coords.npz', '')}"
            res["vanishing"][key] = _summarize(
                X[:, ai[i_ion]], X[:, ai[i_anchor]], box, r,
                prot_xyz=X[:, ph] if ph else None, cat_xyz=X[:, ca] if ca else None,
                lig_xyz=X[:, [ai[a] for a in lig if a in ai]])
    return res


def main():
    rep_dir = os.path.abspath(sys.argv[1])
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "step0_carrier_geometry.json")
    result = {
        "rep_dir": rep_dir,
        "complex": _leg(rep_dir, rep_dir, os.path.join(rep_dir, "topology.cif"), True),
        "solvent": _leg(rep_dir, os.path.join(rep_dir, "solvent_leg"),
                        os.path.join(rep_dir, "topology_solvent.cif"), False),
    }
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print(out)


if __name__ == "__main__":
    main()

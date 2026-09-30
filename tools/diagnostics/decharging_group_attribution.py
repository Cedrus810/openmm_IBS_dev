"""Where does the cross-repeat spread of the complex-leg decharging come from?

For each run, frames of the fully charged decharging state (decharging_rep0.dcd)
are split into ligand--group electrostatic interaction energies:
  * each protein residue, as a whole residue (made whole around the ligand
    centroid by minimum image of the residue centroid), vacuum Coulomb with no
    cutoff -- a cutoff would cut charged residues in half
  * all waters within --water-shell of the ligand (whole molecules)
  * ions within --water-shell
Mean per run; groups ranked by the spread of their per-run means.

The absolute numbers are vacuum Coulomb, not the PME cross energy the
pipeline integrates, so only differences between runs are meaningful.

Read-only on the run directories.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ONE_4PI_EPS0 = 138.935456


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="label=run_dir")
    p.add_argument("--state", type=int, default=0, help="decharging λ state (0 = fully charged)")
    p.add_argument("--water-shell", type=float, default=1.2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import mdtraj as md
    import openmm as mm
    from openmm import unit

    results = {}
    names = None
    for spec in args.runs:
        label, run = spec.split("=", 1)
        run = Path(run)
        sysx = mm.XmlSerializer.deserialize((run / "system_native.xml").read_text())
        nb = [f for f in sysx.getForces() if isinstance(f, mm.NonbondedForce)][0]
        q = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                      for i in range(nb.getNumParticles())])
        top = md.load_topology(str(run / "topology.cif"))
        lig = np.array(json.load(open(run / "ligand_indices.json"))["ligand_indices"])
        ligset = set(lig.tolist())
        res_groups, res_names = [], []
        for r in top.residues:
            idx = [a.index for a in r.atoms if a.index not in ligset]
            if not idx or r.name in ("HOH", "NA", "CL", "SOL", "Na+", "Cl-"):
                continue
            res_groups.append(np.array(idx))
            res_names.append(f"{r.name}{r.resSeq}")
        waters = np.array([[a.index for a in r.atoms] for r in top.residues if r.name in ("HOH", "SOL")])
        ions = np.array([a.index for a in top.atoms if a.residue.name in ("NA", "CL", "Na+", "Cl-")])
        if names is None:
            names = res_names
        elif names != res_names:
            raise SystemExit(f"{label}: residue list differs from the first run")

        t = md.load_dcd(str(run / "decharging" / f"decharging_rep{args.state}.dcd"), top=top)[::args.stride]
        E_res = np.zeros((len(t), len(res_groups)))
        E_wat = np.zeros(len(t))
        E_ion = np.zeros(len(t))
        n_wat = np.zeros(len(t))
        qL = q[lig]
        for f in range(len(t)):
            x = t.xyz[f].astype(np.float64)
            L = t.unitcell_lengths[f].astype(np.float64)
            lx = x[lig]
            lx = lx[0] + (lx - lx[0]) - L * np.round((lx - lx[0]) / L)
            lc = lx.mean(0)

            def coul(idx):
                y = x[idx]
                c = y.mean(0)
                y = y + (lc + (c - lc) - L * np.round((c - lc) / L)) - c
                d = np.sqrt(((lx[:, None, :] - y[None]) ** 2).sum(-1))
                return ONE_4PI_EPS0 * float((qL[:, None] * q[idx][None] / d).sum())

            for k, g in enumerate(res_groups):
                E_res[f, k] = coul(g)
            O = x[waters[:, 0]]
            dO = O - lc - L * np.round((O - lc) / L)
            dmin = np.sqrt(((lx[:, None, :] - (lc + dO)[None]) ** 2).sum(-1)).min(0)
            near = waters[dmin < args.water_shell]
            n_wat[f] = int((dmin < 0.35).sum())
            E_wat[f] = sum(coul(w) for w in near)
            if len(ions):
                I = x[ions]
                dI = np.sqrt(((I - lc - L * np.round((I - lc) / L)) ** 2).sum(-1))
                E_ion[f] = sum(coul(np.array([i])) for i in ions[dI < args.water_shell])
        results[label] = {"E_res": E_res.mean(0), "E_res_sd": E_res.std(0), "E_wat": E_wat.mean(),
                          "E_ion": E_ion.mean(), "n_wat_035": n_wat.mean(), "n_frames": len(t),
                          "E_res_half": (E_res[: len(t) // 2].mean(0), E_res[len(t) // 2:].mean(0)),
                          "E_wat_half": (E_wat[: len(t) // 2].mean(), E_wat[len(t) // 2:].mean())}
        print(f"{label}: {len(t)} frames, protein {E_res.mean(0).sum():8.1f}  water {E_wat.mean():8.1f}  "
              f"ions {E_ion.mean():6.1f}  waters<0.35nm {n_wat.mean():4.1f}", flush=True)

    labels = list(results)
    M = np.array([results[k]["E_res"] for k in labels])
    spread = M.std(0, ddof=1)
    order = np.argsort(-spread)[:args.top]
    print("\nresidues ranked by spread of per-run mean ligand Coulomb (kJ/mol):")
    print(f"{'residue':10s} {'SD':>6s}  " + " ".join(f"{k:>11s}" for k in labels))
    for i in order:
        print(f"{names[i]:10s} {spread[i]:6.2f}  " + " ".join(f"{M[j, i]:11.2f}" for j in range(len(labels))))
    tot_p = M.sum(1)
    tot_w = np.array([results[k]["E_wat"] for k in labels])
    tot_i = np.array([results[k]["E_ion"] for k in labels])
    print(f"\n{'protein':10s} {tot_p.std(ddof=1):6.2f}  " + " ".join(f"{v:11.2f}" for v in tot_p))
    print(f"{'water':10s} {tot_w.std(ddof=1):6.2f}  " + " ".join(f"{v:11.2f}" for v in tot_w))
    print(f"{'ions':10s} {tot_i.std(ddof=1):6.2f}  " + " ".join(f"{v:11.2f}" for v in tot_i))
    print(f"{'sum':10s} {(tot_p + tot_w + tot_i).std(ddof=1):6.2f}  " + " ".join(f"{v:11.2f}" for v in tot_p + tot_w + tot_i))
    print("\nwithin-run first/second half for the top residues (frozen state => halves agree, runs differ):")
    for i in order[:6]:
        print(f"{names[i]:10s} " + "  ".join(f"{results[k]['E_res_half'][0][i]:7.1f}/{results[k]['E_res_half'][1][i]:7.1f}"
                                            for k in labels))
    if args.out:
        json.dump({k: {"protein_total": float(results[k]["E_res"].sum()), "water": float(results[k]["E_wat"]),
                       "ions": float(results[k]["E_ion"]),
                       "top_residues": {names[i]: float(results[k]["E_res"][i]) for i in order}}
                   for k in labels}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

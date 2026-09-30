"""MD + pocket-water relocation MC on one endpoint state of a pipeline run.

  state holo : system_native.xml as is (ligand coupled)
  state apo  : every ligand nonbonded parameter zeroed (ghost ligand, lambda = 0)

Every --md-steps of Langevin MD, one MC phase of --attempts relocation
attempts (tools/pocket_water/relocation.py).  --attempts 0 gives the plain-MD
control with identical bookkeeping.  The pocket region R is the union of
--r-in spheres on the ligand heavy atoms of the start frame, the shell S the
union of --r-out spheres minus R; both follow the pocket-lining CA atoms.

Question it answers: does exact-Hamiltonian water exchange put more (or
fewer) waters into the pocket than MD alone reaches?

Read-only on the run directory; writes only to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.pocket_water.relocation import RelocationMC, min_dist, min_image  # noqa: E402
from tools.pocket_water.u0 import U0Engine  # noqa: E402

KB_KJ = 0.0083144626


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--state", choices=["holo", "apo"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ps", type=float, default=1000.0, help="total MD time")
    p.add_argument("--md-steps", type=int, default=500, help="MD steps (2 fs) between MC phases")
    p.add_argument("--attempts", type=int, default=10000, help="relocation attempts per MC phase (0 = plain MD)")
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--r-in", type=float, default=0.45)
    p.add_argument("--r-out", type=float, default=1.20)
    p.add_argument("--start", default=None, help="npz with 'pos' (nm) to start from instead of pre_equil")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--platform", default="CUDA")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - pc @ R.T


def contiguous(X, box):
    return X[0] + min_image(X - X[0], box)


def main():
    args = parse_args()
    import openmm as mm
    from openmm import unit
    from openmm.app import PDBxFile

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    beta = 1.0 / (KB_KJ * args.temperature)
    t0 = time.time()

    def log(*a):
        print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)

    top = PDBxFile(str(run / "topology.cif")).topology
    atoms = list(top.atoms())
    st = np.load(run / "checkpoints" / "pre_equil_final_state.npz")
    pos0 = st["positions_nm"].astype(np.float64)
    box0 = st["box_nm"].astype(np.float64)
    assert np.allclose(box0, np.diag(np.diag(box0))), "orthorhombic boxes only"
    box = np.diag(box0)
    lig = np.array(json.load(open(run / "ligand_indices.json"))["ligand_indices"])
    ligset = set(int(i) for i in lig)
    lig_heavy = np.array([i for i in lig if atoms[i].element.symbol != "H"])
    waters = np.array([[b.index for b in r.atoms()] for r in top.residues() if r.name == "HOH"])
    assert all(atoms[w[0]].element.symbol == "O" for w in waters)
    ca = np.array([a.index for a in atoms if a.name == "CA" and a.index not in ligset
                   and a.residue.name not in ("HOH", "NA", "CL")])

    # region centres: ligand heavy atoms of the start frame, in the lining-CA frame
    lig_pos = contiguous(pos0[lig_heavy], box)
    lc = lig_pos.mean(0)
    ca_loc = lc + min_image(pos0[ca] - lc, box)
    sel = np.linalg.norm(ca_loc - lc, axis=1) < 1.2
    lining, lining_ref = ca[sel], ca_loc[sel]
    centers_ref = lig_pos

    def centers_fn(p):
        R, t = kabsch(lining_ref, contiguous(p[lining], box))
        return centers_ref @ R.T + t

    system = mm.XmlSerializer.deserialize((run / "system_native.xml").read_text())
    for k in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(k), mm.MonteCarloBarostat):
            system.removeForce(k)
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    if args.state == "apo":
        for i in ligset:
            q, s, e = nb.getParticleParameters(i)
            nb.setParticleParameters(i, 0.0, s, 0.0)
        for k in range(nb.getNumExceptions()):
            i, j, q, s, e = nb.getExceptionParameters(k)
            if i in ligset or j in ligset:
                nb.setExceptionParameters(k, i, j, 0.0, s, 0.0)
    plat = mm.Platform.getPlatformByName(args.platform)
    props = {"Precision": "mixed"} if args.platform in ("CUDA", "OpenCL") else {}
    if args.device is not None and args.platform == "CUDA":
        props["DeviceIndex"] = str(args.device)
    integ = mm.LangevinMiddleIntegrator(args.temperature * unit.kelvin, 1.0 / unit.picosecond,
                                        0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(args.seed % 2**31)
    ctx = mm.Context(system, integ, plat, dict(props))
    ev = mm.Context(system, mm.VerletIntegrator(0.001), plat, dict(props))
    for c in (ctx, ev):
        c.setPeriodicBoxVectors(*box0)
    start = np.load(args.start)["pos"] if args.start else pos0
    ctx.setPositions(start)
    ctx.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed % 2**31)
    ev.setPositions(start)

    def exact(p):
        ev.setPositions(p)
        return ev.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    eng = U0Engine.from_openmm(nb, ctx, waters)
    mc = RelocationMC(eng, beta, args.r_in, args.r_out, centers_ref, centers_fn, exact, rng)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
    log(f"state={args.state} V_R={mc.V_R:.3f} nm3 (bulk-equiv {mc.V_R*33.4:.1f} waters) V_S={mc.V_S:.2f} nm3 "
        f"attempts/phase={args.attempts}")

    n_phases = int(round(args.ps * 500 / args.md_steps))
    trace = []
    O = waters[:, 0]
    t_mc = t_md = 0.0
    for ph in range(n_phases):
        a = time.time()
        integ.step(args.md_steps)
        s = ctx.getState(getPositions=True)
        p = np.asarray(s.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=np.float64)
        t_md += time.time() - a
        moved = []
        if args.attempts:
            a = time.time()
            p, moved = mc.run_phase(p, box, args.attempts, batch=args.batch)
            t_mc += time.time() - a
            if moved:
                ctx.setPositions(p)
                v = np.asarray(ctx.getState(getVelocities=True).getVelocities(asNumpy=True)
                               .value_in_unit(unit.nanometer / unit.picosecond))
                idx = waters[moved].reshape(-1)
                sd = np.sqrt(KB_KJ * args.temperature / masses[idx])
                v[idx] = rng.normal(size=(len(idx), 3)) * sd[:, None]
                ctx.setVelocities(v)
                ctx.applyVelocityConstraints(1e-6)
        d = min_dist(p[O], centers_fn(p), box)
        in_R = np.where(d < args.r_in)[0]
        trace.append({"t_ps": (ph + 1) * args.md_steps * 0.002, "N_R": int(len(in_R)),
                      "R": in_R.tolist(), "moved": moved})
        if (ph + 1) % 50 == 0 or ph == n_phases - 1:
            st_ = mc.stats
            n_recent = [x["N_R"] for x in trace[-50:]]
            log(f"t={trace[-1]['t_ps']:7.1f} ps  N_R mean(last50)={np.mean(n_recent):5.2f}  "
                f"attempts={st_['attempts']}  stage1 in/out={st_['s1_in']}/{st_['s1_out']}  "
                f"accepted in/out={st_['acc_in']}/{st_['acc_out']}  md {t_md:.0f}s  mc {t_mc:.0f}s")
            summary = {"args": vars(args), "V_R": mc.V_R, "V_S": mc.V_S,
                       "stats": {k: v for k, v in mc.stats.items() if k != "s2_log_alpha"},
                       "s2_alpha_mean": float(np.mean(np.exp(mc.stats["s2_log_alpha"]))) if mc.stats["s2_log_alpha"] else None,
                       "t_md_s": t_md, "t_mc_s": t_mc, "trace": trace}
            json.dump(summary, open(out / "sampler_summary.json", "w"))
    np.savez_compressed(out / "final_state.npz", pos=p.astype(np.float32))
    log("done")


if __name__ == "__main__":
    main()

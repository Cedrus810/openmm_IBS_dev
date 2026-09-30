"""Dry-cavity check: are there empty spots near the ligand where a water would be
as happy as the waters that are already there?

Blind spot of the S1 accessibility / occupancy counts: a buried cavity that is
dry from the start and never fills in either start is invisible to them.
This probe looks for such voids directly on frames saved by
pocket_water_s1_probe.py (s1_frames.npz: holo = ligand coupled, apo = ligand
nonbonded off).

Per frame:
  1. grid (--spacing) within --shell of ligand heavy atoms; a grid point is
     void if it is >= --clearance from every heavy atom (protein, water O,
     ions; ligand too in holo frames, not in apo frames where it is a ghost)
  2. connected void points form a cavity; its site is the point of maximum
     clearance; buriedness = protein heavy atoms within 0.8 nm
  3. a far bulk water is relocated onto the site with --orientations random
     orientations (+ <= --jitter nm position jitter); exact full-system
     OpenMM energy gives beta*u of a water at that site
  4. reference: beta*u of the real waters within --shell of the ligand

A void whose water energy is comparable to the real pocket waters is a spot
that should be wet but is dry.  beta*u carries the same constant per-water
offset for every water (Ewald self/exclusion, LRC), so only comparisons are
meaningful.

Read-only on the run directory and the S1 output; writes only to --out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

KB_KJ = 0.0083144626


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory the S1 frames came from")
    p.add_argument("--s1", required=True, help="pocket_water_s1_probe.py output dir (has s1_frames.npz)")
    p.add_argument("--out", required=True)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--platform", default="CUDA")
    p.add_argument("--shell", type=float, default=0.80)
    p.add_argument("--spacing", type=float, default=0.05)
    p.add_argument("--clearance", type=float, default=0.28)
    p.add_argument("--orientations", type=int, default=300)
    p.add_argument("--jitter", type=float, default=0.03)
    p.add_argument("--max-voids", type=int, default=20, help="per frame, largest clearance first")
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


def random_rotations(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def main():
    args = parse_args()
    import openmm as mm
    from openmm import unit
    from openmm.app import PDBxFile
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    beta = 1.0 / (KB_KJ * args.temperature)
    t_start = time.time()

    def log(*a):
        print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)

    top = PDBxFile(str(run / "topology.cif")).topology
    atoms = list(top.atoms())
    st = np.load(run / "checkpoints" / "pre_equil_final_state.npz")
    box0 = st["box_nm"].astype(np.float64)
    L = np.diag(box0)
    lig = np.array(json.load(open(run / "ligand_indices.json"))["ligand_indices"])
    ligset = set(int(i) for i in lig)
    heavy = np.array([a.index for a in atoms if a.element.symbol != "H"])
    lig_heavy = np.array([i for i in lig if atoms[i].element.symbol != "H"])
    nonlig_heavy = np.array([i for i in heavy if int(i) not in ligset])
    prot_heavy = np.array([a.index for a in atoms if a.element.symbol != "H" and a.index not in ligset
                           and a.residue.name not in ("HOH", "NA", "CL")])
    water_O = np.array([a.index for a in atoms if a.residue.name == "HOH" and a.element.symbol == "O"])
    water_res = {a.index: [b.index for b in a.residue.atoms()] for a in atoms
                 if a.residue.name == "HOH" and a.element.symbol == "O"}

    xml = (run / "system_native.xml").read_text()
    systems = {"holo": mm.XmlSerializer.deserialize(xml), "apo": mm.XmlSerializer.deserialize(xml)}
    nb_apo = [f for f in systems["apo"].getForces() if isinstance(f, mm.NonbondedForce)][0]
    for i in ligset:
        q, s, e = nb_apo.getParticleParameters(i)
        nb_apo.setParticleParameters(i, 0.0, s, 0.0)
    for k in range(nb_apo.getNumExceptions()):
        i, j, q, s, e = nb_apo.getExceptionParameters(k)
        if i in ligset or j in ligset:
            nb_apo.setExceptionParameters(k, i, j, 0.0, s, 0.0)
    for s_ in systems.values():
        for k in reversed(range(s_.getNumForces())):
            if isinstance(s_.getForce(k), mm.MonteCarloBarostat):
                s_.removeForce(k)
    nbf = {k: [f for f in s.getForces() if isinstance(f, mm.NonbondedForce)][0] for k, s in systems.items()}
    plat = mm.Platform.getPlatformByName(args.platform)
    props = {"Precision": "mixed"} if args.platform in ("CUDA", "OpenCL") else {}
    ctxs = {}
    for k, s in systems.items():
        integ = mm.VerletIntegrator(0.001)
        ctxs[k] = (mm.Context(s, integ, plat, props), integ)

    def energy(ctx):
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def water_bu(ctx, label, o, e_full):
        f = nbf[label]
        saved = [f.getParticleParameters(i) for i in water_res[o]]
        for i, (q, sg, ep) in zip(water_res[o], saved):
            f.setParticleParameters(i, 0.0, sg, 0.0)
        f.updateParametersInContext(ctx)
        e = energy(ctx)
        for i, (q, sg, ep) in zip(water_res[o], saved):
            f.setParticleParameters(i, q, sg, ep)
        f.updateParametersInContext(ctx)
        return beta * (e_full - e)

    frames = np.load(Path(args.s1) / "s1_frames.npz")
    results = []
    for key in sorted(frames.files, key=lambda k: int(k.split("_")[0][1:])):
        label = key.split("_", 1)[1]
        pos = frames[key].astype(np.float64)
        ctx = ctxs[label][0]
        ctx.setPeriodicBoxVectors(*box0)
        ctx.setPositions(pos)
        e0 = energy(ctx)

        wrapped = np.mod(pos, L)
        blockers = heavy if label == "holo" else nonlig_heavy
        tree = cKDTree(wrapped[blockers], boxsize=L)
        ptree = cKDTree(wrapped[prot_heavy], boxsize=L)
        lh = wrapped[lig_heavy]
        lh = lh[0] + (lh - lh[0]) - L * np.round((lh - lh[0]) / L)  # contiguous ligand
        lo, hi = lh.min(0) - args.shell, lh.max(0) + args.shell
        axes = [np.arange(lo[d], hi[d] + 1e-9, args.spacing) for d in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        dl = cKDTree(lh).query(grid)[0]
        grid = grid[dl < args.shell]
        clear = tree.query(np.mod(grid, L))[0]
        void = grid[clear >= args.clearance]
        vclear = clear[clear >= args.clearance]
        voids = []
        if len(void):
            pairs = cKDTree(void).query_pairs(args.spacing * 1.01, output_type="ndarray")
            n = len(void)
            adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) \
                else coo_matrix((n, n))
            ncomp, lab = connected_components(adj, directed=False)
            for c in range(ncomp):
                m = lab == c
                i = np.argmax(np.where(m, vclear, -1))
                site = void[i]
                voids.append({"site": site, "n_points": int(m.sum()), "clearance": float(vclear[i]),
                              "buried_prot_0.8": int(len(ptree.query_ball_point(np.mod(site, L), 0.8))),
                              "d_lig": float(cKDTree(lh).query(site)[0])})
        voids.sort(key=lambda v: -v["clearance"])
        voids = voids[:args.max_voids]

        # reference: real waters near the ligand
        dw = cKDTree(lh).query(pos[water_O] - L * np.floor((pos[water_O] - lh.mean(0) + L / 2) / L))[0]
        near_w = water_O[dw < args.shell]
        ref_bu = np.array([water_bu(ctx, label, int(o), e0) for o in near_w])
        ref_bur = np.array([len(ptree.query_ball_point(np.mod(pos[o], L), 0.8)) for o in near_w])

        # source: a far bulk water
        far = water_O[dw > 2.0]
        src = int(rng.choice(far))
        widx = np.array(water_res[src])
        geom = pos[widx] - pos[src]
        bu_src = water_bu(ctx, label, src, e0)
        work = pos.copy()
        for v in voids:
            rots = random_rotations(rng, args.orientations)
            jit = rng.normal(size=(args.orientations, 3))
            jit *= (args.jitter * rng.random(args.orientations) ** (1 / 3) / np.linalg.norm(jit, axis=1))[:, None]
            bu = np.empty(args.orientations)
            for k in range(args.orientations):
                work[widx] = v["site"] + jit[k] + geom @ rots[k].T
                ctx.setPositions(work)
                bu[k] = beta * (energy(ctx) - e0) + bu_src
            work[widx] = pos[widx]
            ctx.setPositions(pos)
            v["bu_min"] = float(bu.min())
            m = (-bu).max()
            v["bF_site"] = float(-(m + np.log(np.exp(-bu - m).mean())))
            v["site"] = v["site"].round(3).tolist()
        row = {"frame": key, "state": label, "n_voids": len(voids), "voids": voids,
               "ref_water_bu": np.sort(ref_bu).round(2).tolist(),
               "ref_buried_water_bu": np.sort(ref_bu[ref_bur >= 45]).round(2).tolist(),
               "bu_src": float(bu_src)}
        results.append(row)
        best = min((v["bu_min"] for v in voids), default=None)
        log(key, f"voids={len(voids)}", f"best bu_min={best}",
            f"ref waters n={len(ref_bu)} max(weakest)={ref_bu.max() if len(ref_bu) else None:.1f}")
        json.dump({"args": vars(args), "rows": results}, open(out / "void_summary.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()

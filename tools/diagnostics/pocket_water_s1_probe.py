"""S1 feasibility probe for the pocket-water relocation sampler.

See docs/design/PLAN_pocket_water_swap_sampler_2026-09-27.md (sections 6 and 8).

Per run directory it answers three questions, for two pocket-region definitions:

  region "lig"  : union of spheres (R 0.30 nm) on buried ligand heavy atoms
                  (the ligand's own volume; N_holo ~ 0 by construction)
  region "site" : union of small spheres (R 0.14 nm) on persistent holo
                  hydration sites within 0.5 nm of the ligand, clustered once
                  from pre_equilibration.dcd

  1. accessibility : plain MD at lambda=1 (holo) and lambda=0 (ghost ligand),
                     N(t) and water-identity exchange with a hysteresis buffer
                     (a water enters when its O is inside the region, leaves
                     when it is farther than --buffer beyond the region)
  2. P(N) overlap   : holo vs apo N histograms per region (unbiased MD, so
                     conditional on whatever the MD reaches)
  3. acceptance     : energy-biased source selection (p_i ~ exp(beta u_i),
                     JCTC 2026 SwapMC eq 1); single-trial acceptance
                     A = (V_in/V_out) Z_out(x) / Z_in(y) with
                     Z_in(y) ~ Z_in(x) + exp(beta u_y), and the probability
                     that a batch of --trials-in candidates accepts at least
                     once.  All energies are exact full-system OpenMM (PME).

Read-only on the run directory; writes only to --out.  Pocket regions follow
the pocket-lining CA atoms by Kabsch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

KB_KJ = 0.0083144626  # kJ/mol/K
RHO_WATER = 33.4  # nm^-3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory (e.g. runs/brd4_ligand2/rep1)")
    p.add_argument("--out", required=True)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--platform", default="CUDA")
    p.add_argument("--device", default=None, help="CUDA DeviceIndex")
    p.add_argument("--r-lig", type=float, default=0.30)
    p.add_argument("--buried-rel-sasa", type=float, default=0.25)
    p.add_argument("--r-site", type=float, default=0.14)
    p.add_argument("--site-shell", type=float, default=0.50, help="sites within this of ligand heavy atoms")
    p.add_argument("--site-min-occ", type=float, default=0.60)
    p.add_argument("--buffer", type=float, default=0.30, help="hysteresis buffer for exchange events (nm)")
    p.add_argument("--md-ps", type=float, default=1000.0, help="plain MD length per state (holo and apo)")
    p.add_argument("--md-skip-ps", type=float, default=100.0)
    p.add_argument("--frames-per-state", type=int, default=5, help="probe frames per state")
    p.add_argument("--sources-per-frame", type=int, default=2)
    p.add_argument("--trials-in", type=int, default=18000, help="pocket candidates per source per region")
    p.add_argument("--trials-out", type=int, default=2000, help="bulk candidates per source")
    p.add_argument("--bulk-sample", type=int, default=600, help="bulk waters sampled for Z_out")
    p.add_argument("--skip-probe", action="store_true", help="only accessibility + P(N)")
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


# ----------------------------------------------------------------- geometry

def min_image(d, box_diag):
    return d - box_diag * np.round(d / box_diag)


def kabsch(P, Q):
    """Rotation R, translation t minimising |(P @ R.T + t) - Q|."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - pc @ R.T


def make_contiguous(X, box_diag):
    return X[0] + min_image(X - X[0], box_diag)


def dist_to_centers(points, centers, box_diag):
    d = min_image(points[:, None, :] - centers[None, :, :], box_diag)
    return np.sqrt(np.einsum("ijk,ijk->ij", d, d)).min(1)


def union_volume(centers, r, rng, n=2_000_000):
    lo, hi = centers.min(0) - r, centers.max(0) + r
    pts = lo + (hi - lo) * rng.random((n, 3))
    d = pts[:, None, :] - centers[None, :, :]
    frac = (np.einsum("ijk,ijk->ij", d, d) < r * r).any(1).mean()
    return float(np.prod(hi - lo) * frac)


def uniform_in_union(centers, r, rng, n):
    lo, hi = centers.min(0) - r, centers.max(0) + r
    out, got = [], 0
    while got < n:
        pts = lo + (hi - lo) * rng.random((4 * n, 3))
        d = pts[:, None, :] - centers[None, :, :]
        sel = pts[(np.einsum("ijk,ijk->ij", d, d) < r * r).any(1)]
        out.append(sel)
        got += len(sel)
    return np.concatenate(out)[:n]


def uniform_outside(regions_c, box_diag, rng, n):
    out, got = [], 0
    while got < n:
        pts = box_diag * rng.random((2 * n, 3))
        keep = np.ones(len(pts), bool)
        for c, r in regions_c:
            keep &= dist_to_centers(pts, c, box_diag) >= r
        out.append(pts[keep])
        got += keep.sum()
    return np.concatenate(out)[:n]


def random_rotations(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def logsumexp(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -np.inf
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum()))


# ------------------------------------------------------------ exchange stats

def exchange_stats(core_sets, infl_sets, dt_ps):
    """Hysteresis identity exchange.  core_sets/infl_sets: per-frame sets of
    water O indices inside the region / inside region+buffer."""
    state = {}  # water -> (committed_in, t_enter); t_enter 0 = present at start
    for w in core_sets[0]:
        state[w] = (True, 0)
    initial = set(core_sets[0])
    entries = exits = 0
    visits = []
    ever = set(core_sets[0])
    for t in range(1, len(core_sets)):
        core, infl = core_sets[t], infl_sets[t]
        for w in core:
            if not state.get(w, (False, 0))[0]:
                state[w] = (True, t)
                entries += 1
                ever.add(w)
        for w, (is_in, t0) in list(state.items()):
            if is_in and w not in infl:
                state[w] = (False, 0)
                exits += 1
                if t0 > 0:
                    visits.append((t - t0) * dt_ps)
    still = sum(1 for w in initial if state.get(w, (False, 0))[0])
    T_ns = (len(core_sets) - 1) * dt_ps / 1000.0
    n_mean = float(np.mean([len(c) for c in core_sets]))
    return {
        "N_mean": n_mean,
        "entries": entries, "exits": exits,
        "entries_per_ns": entries / T_ns if T_ns else None,
        "turnover_per_water_per_ns": (entries / T_ns / n_mean) if T_ns and n_mean > 0 else None,
        "distinct_waters_visited": len(ever),
        "initial_waters": len(initial),
        "initial_still_inside_at_end": still,
        "complete_visit_ps_median": float(np.median(visits)) if visits else None,
        "complete_visits": len(visits),
    }


# ------------------------------------------------------------------- main

def main():
    args = parse_args()
    import mdtraj as md
    import openmm as mm
    from openmm import unit
    from openmm.app import PDBxFile

    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    beta = 1.0 / (KB_KJ * args.temperature)
    t_start = time.time()

    def log(*a):
        print(f"[{time.time() - t_start:7.0f}s]", *a, flush=True)

    top = PDBxFile(str(run / "topology.cif")).topology
    st = np.load(run / "checkpoints" / "pre_equil_final_state.npz")
    pos0 = st["positions_nm"].astype(np.float64)
    box0 = st["box_nm"].astype(np.float64)
    assert np.allclose(box0, np.diag(np.diag(box0))), "probe assumes a rectangular box"
    box_diag = np.diag(box0)
    lig = np.array(json.load(open(run / "ligand_indices.json"))["ligand_indices"])
    ligset = set(int(i) for i in lig)

    atoms = list(top.atoms())
    water_O = np.array([a.index for a in atoms if a.residue.name == "HOH" and a.element.symbol == "O"])
    water_res = {a.index: [b.index for b in a.residue.atoms()] for a in atoms
                 if a.residue.name == "HOH" and a.element.symbol == "O"}
    prot_ca = np.array([a.index for a in atoms if a.name == "CA" and a.index not in ligset
                        and a.residue.name not in ("HOH", "NA", "CL")])
    lig_heavy = np.array([i for i in lig if atoms[i].element.symbol != "H"])

    # ---- region "lig": buried ligand heavy atoms by SASA ratio
    mdt = md.Topology.from_openmm(top)
    lig_pos = make_contiguous(pos0[lig], box_diag)
    lc = lig_pos.mean(0)
    solute_all = np.array([a.index for a in atoms if a.residue.name not in ("HOH", "NA", "CL")
                           and a.index not in ligset])
    loc = lc + min_image(pos0[solute_all] - lc, box_diag)
    near = np.linalg.norm(loc - lc, axis=1) < 2.5
    solute = np.concatenate([lig, solute_all[near]])
    spos = np.concatenate([lig_pos, loc[near]])
    order = np.argsort(solute)
    solute, spos = solute[order], spos[order]
    sasa_c = md.shrake_rupley(md.Trajectory(spos[None], mdt.subset(solute)), mode="atom")[0]
    sasa_f = md.shrake_rupley(md.Trajectory(lig_pos[None], mdt.subset(lig)), mode="atom")[0]
    rel = sasa_c[np.searchsorted(solute, lig_heavy)] / np.maximum(sasa_f[np.searchsorted(lig, lig_heavy)], 1e-6)
    buried = lig_heavy[rel < args.buried_rel_sasa]
    lig_centers = lig_pos[np.searchsorted(lig, buried)]

    # lining CA in the ligand's image; this is the reference frame of every region
    ca_loc = lc + min_image(pos0[prot_ca] - lc, box_diag)
    sel = np.linalg.norm(ca_loc - lc, axis=1) < 1.2
    lining = prot_ca[sel]
    lining_ref = ca_loc[sel]

    def frame_transform(pos, bd):
        L = make_contiguous(pos[lining], bd)
        return kabsch(lining_ref, L)

    # ---- region "site": persistent holo hydration sites from pre_equilibration.dcd
    dcd = md.load_dcd(str(run / "pre_equilibration.dcd"), top=mdt)
    nf = len(dcd)
    pts, frame_of = [], []
    lig_heavy_ref = lig_pos[np.searchsorted(lig, lig_heavy)]
    for f in range(nf // 2, nf):
        xyz = dcd.xyz[f].astype(np.float64)
        bd = dcd.unitcell_lengths[f].astype(np.float64)
        R, t = frame_transform(xyz, bd)
        c = lc @ R.T + t
        wo = c + min_image(xyz[water_O] - c, bd)
        ref = (wo - t) @ R  # back into the reference frame
        dmin = np.sqrt(((ref[:, None, :] - lig_heavy_ref[None]) ** 2).sum(-1)).min(1)
        keep = dmin < args.site_shell
        pts.append(ref[keep])
        frame_of.append(np.full(keep.sum(), f))
    n_site_frames = nf - nf // 2
    pts = np.concatenate(pts)
    frame_of = np.concatenate(frame_of)
    del dcd
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    nbr_count = np.array([len(x) for x in tree.query_ball_point(pts, 0.10)])
    alive = np.ones(len(pts), bool)
    sites, site_occ = [], []
    for i in np.argsort(-nbr_count):
        if not alive[i]:
            continue
        nb = [j for j in tree.query_ball_point(pts[i], 0.10) if alive[j]]
        center = pts[nb].mean(0)
        inside = tree.query_ball_point(center, args.r_site)
        occ = len(set(frame_of[inside].tolist())) / n_site_frames
        if occ < args.site_min_occ:
            if nbr_count[i] < args.site_min_occ * n_site_frames * 0.3:
                break
            alive[i] = False
            continue
        sites.append(center)
        site_occ.append(occ)
        for j in tree.query_ball_point(center, 0.28):
            alive[j] = False
    site_centers = np.array(sites) if sites else np.zeros((0, 3))

    regions = {"lig": {"centers": lig_centers, "r": args.r_lig}}
    if len(site_centers):
        regions["site"] = {"centers": site_centers, "r": args.r_site}
    for name, rg in regions.items():
        rg["V"] = union_volume(rg["centers"], rg["r"], rng)
    region_info = {name: {"n_spheres": int(len(rg["centers"])), "r_nm": rg["r"], "V_nm3": rg["V"],
                          "bulk_equivalent_waters": rg["V"] * RHO_WATER}
                   for name, rg in regions.items()}
    region_info["site"] = dict(region_info.get("site", {}), occupancies=[round(o, 3) for o in site_occ],
                               min_occ=args.site_min_occ, shell_nm=args.site_shell)
    region_info["n_lining_ca"] = int(len(lining))
    region_info["lig_rel_sasa_threshold"] = args.buried_rel_sasa
    log("regions", json.dumps(region_info))

    def centers_now(pos, bd, name):
        R, t = frame_transform(pos, bd)
        return regions[name]["centers"] @ R.T + t

    # ---- systems
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
    for sys_ in systems.values():
        for k in reversed(range(sys_.getNumForces())):
            if isinstance(sys_.getForce(k), mm.MonteCarloBarostat):
                sys_.removeForce(k)
    nbf = {k: [f for f in s.getForces() if isinstance(f, mm.NonbondedForce)][0] for k, s in systems.items()}

    plat = mm.Platform.getPlatformByName(args.platform)
    props = {"Precision": "mixed"} if args.platform in ("CUDA", "OpenCL") else {}
    if args.device is not None and args.platform == "CUDA":
        props["DeviceIndex"] = str(args.device)

    def new_context(label):
        integ = mm.LangevinMiddleIntegrator(args.temperature * unit.kelvin, 1.0 / unit.picosecond,
                                            0.002 * unit.picoseconds)
        return mm.Context(systems[label], integ, plat, props), integ

    # ---- 1+2: plain MD per state, exchange + N traces, probe frames
    md_out = {}
    frames = []
    n_ps = int(args.md_ps)
    save_at = set(np.linspace(args.md_skip_ps, args.md_ps, args.frames_per_state).astype(int).tolist())
    for si, label in enumerate(("holo", "apo")):
        ctx, integ = new_context(label)
        ctx.setPeriodicBoxVectors(*box0)
        ctx.setPositions(pos0)
        ctx.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed + si)
        core = {n: [] for n in regions}
        infl = {n: [] for n in regions}
        n_sites = len(regions["site"]["centers"]) if "site" in regions else 0
        site_core = [[] for _ in range(n_sites)]
        site_infl = [[] for _ in range(n_sites)]
        t0 = time.time()
        for ps in range(0, n_ps + 1):
            if ps:
                integ.step(500)
            p = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            p = np.asarray(p, dtype=np.float64)
            for name, rg in regions.items():
                d = dist_to_centers(p[water_O], centers_now(p, box_diag, name), box_diag)
                core[name].append(set(water_O[d < rg["r"]].tolist()))
                infl[name].append(set(water_O[d < rg["r"] + args.buffer].tolist()))
            if n_sites:
                cs = centers_now(p, box_diag, "site")
                for k in range(n_sites):
                    d = dist_to_centers(p[water_O], cs[k:k + 1], box_diag)
                    site_core[k].append(set(water_O[d < args.r_site].tolist()))
                    site_infl[k].append(set(water_O[d < args.r_site + args.buffer].tolist()))
            if ps in save_at:
                frames.append((label, p.copy()))
            if ps and ps % 100 == 0:
                log(f"{label} md {ps} ps  " + "  ".join(f"N_{n}={len(core[n][-1])}" for n in regions))
        ns_per_day = (n_ps / 1000) / ((time.time() - t0) / 86400)
        md_out[label] = {"ns_per_day": ns_per_day}
        skip = int(args.md_skip_ps)
        for name in regions:
            ntr = [len(c) for c in core[name]]
            md_out[label][name] = {
                "N_trace_every_10ps": ntr[::10],
                "N_hist": {int(k): int(v) for k, v in zip(*np.unique(ntr[skip:], return_counts=True))},
                "exchange": exchange_stats(core[name][skip:], infl[name][skip:], 1.0),
            }
            log(label, name, json.dumps(md_out[label][name]["exchange"]))
        if n_sites:
            md_out[label]["per_site"] = []
            for k in range(n_sites):
                ex = exchange_stats(site_core[k][skip:], site_infl[k][skip:], 1.0)
                ex["occupancy"] = float(np.mean([len(c) > 0 for c in site_core[k][skip:]]))
                md_out[label]["per_site"].append(ex)
            log(label, "per_site", json.dumps([{"occ": round(e["occupancy"], 2), "entries": e["entries"],
                                                 "distinct": e["distinct_waters_visited"]}
                                                for e in md_out[label]["per_site"]]))
        del ctx, integ

    overlap = {}
    for name in regions:
        h, a = md_out["holo"][name]["N_hist"], md_out["apo"][name]["N_hist"]
        ks = sorted(set(h) | set(a))
        ph = np.array([h.get(k, 0) for k in ks], float)
        pa = np.array([a.get(k, 0) for k in ks], float)
        ph /= ph.sum()
        pa /= pa.sum()
        prod = ph * pa
        overlap[name] = {
            "N": ks, "P_holo": ph.round(4).tolist(), "P_apo": pa.round(4).tolist(),
            "bhattacharyya": float(np.sqrt(prod).sum()),
            "N_star_max_product": int(ks[int(np.argmax(prod))]) if prod.max() > 0 else None,
            "N_holo_mode": int(ks[int(np.argmax(ph))]), "N_apo_mode": int(ks[int(np.argmax(pa))]),
        }
        log("overlap", name, json.dumps(overlap[name]))

    summary = {"run": str(run), "regions": region_info, "md": md_out, "overlap": overlap,
               "args": vars(args)}
    json.dump(summary, open(out / "s1_summary.json", "w"), indent=1)
    np.savez_compressed(out / "s1_frames.npz", **{f"f{i}_{l}": p.astype(np.float32)
                                                  for i, (l, p) in enumerate(frames)})
    if args.skip_probe:
        log("done (probe skipped)")
        return

    # ---- 3: acceptance probe with energy-biased source selection
    ctxs = {k: new_context(k)[0] for k in ("holo", "apo")}

    def energy(ctx):
        return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def water_bu(ctx, label, o, e_full):
        """beta*u of water o (E_full - E with its nonbonded off).  Carries a
        constant per-water offset (Ewald self/exclusion, LRC) that is the same
        for every water and cancels in every ratio below."""
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

    rows = []
    t_eval = []
    for fi, (label, pos) in enumerate(frames):
        ctx = ctxs[label]
        ctx.setPeriodicBoxVectors(*box0)
        ctx.setPositions(pos)
        e0 = energy(ctx)
        cen = {n: centers_now(pos, box_diag, n) for n in regions}
        dist = {n: dist_to_centers(pos[water_O], cen[n], box_diag) for n in regions}
        any_in = np.zeros(len(water_O), bool)
        for n, rg in regions.items():
            any_in |= dist[n] < rg["r"]
        v_box = float(np.prod(box_diag))
        bulk_sample = rng.choice(water_O[~any_in], size=min(args.bulk_sample, int((~any_in).sum())), replace=False)
        bu_bulk = np.array([water_bu(ctx, label, int(o), e0) for o in bulk_sample])
        n_out = int((~any_in).sum())
        log_z_out = np.log(n_out) + logsumexp(bu_bulk) - np.log(len(bu_bulk))
        bu_pocket = {}
        for n, rg in regions.items():
            inside = water_O[dist[n] < rg["r"]]
            bu_pocket[n] = np.array([water_bu(ctx, label, int(o), e0) for o in inside])
        far = water_O[np.min([dist[n] for n in regions], axis=0) > 1.5]
        for si in range(args.sources_per_frame):
            o = int(rng.choice(far))
            widx = np.array(water_res[o])
            # the previous source's trial loop leaves the context at its last
            # trial placement; water_bu needs the frame's own coordinates
            ctx.setPositions(pos)
            rel_geom = pos[widx] - pos[o]
            bu_src = water_bu(ctx, label, o, e0)
            work = pos.copy()

            def pool(sites):
                rots = random_rotations(rng, len(sites))
                es = np.empty(len(sites))
                t1 = time.time()
                for k in range(len(sites)):
                    work[widx] = sites[k] + rel_geom @ rots[k].T
                    ctx.setPositions(work)
                    es[k] = energy(ctx)
                work[widx] = pos[widx]
                t_eval.append((time.time() - t1) / len(sites))
                return beta * (es - e0) + bu_src  # = beta*u of the moved water at the trial

            bu_new = pool(uniform_outside([(cen[n], regions[n]["r"]) for n in regions], box_diag, rng,
                                          args.trials_out))
            for n, rg in regions.items():
                bu_y = pool(uniform_in_union(cen[n], rg["r"], rng, args.trials_in))
                ln_v = np.log(rg["V"] / (v_box - rg["V"]))
                log_z_in = logsumexp(bu_pocket[n])
                la_in = np.minimum(0.0, ln_v + log_z_out - np.logaddexp(log_z_in, bu_y))
                a_in = np.exp(la_in)
                if len(bu_pocket[n]):
                    la_out = np.minimum(0.0, -ln_v + log_z_in - np.logaddexp(log_z_out, bu_new))
                    a_out = np.exp(la_out)
                else:
                    a_out = np.zeros(len(bu_new))
                # probability that one batch of all trials accepts at least once
                p_batch_in = float(-np.expm1(np.log1p(-np.clip(a_in, 0, 1 - 1e-16)).sum()))
                row = {"frame": fi, "state": label, "region": n, "source_O": o,
                       "N_in": int(len(bu_pocket[n])), "bu_src": float(bu_src),
                       "bu_pocket_max": float(bu_pocket[n].max()) if len(bu_pocket[n]) else None,
                       "bu_bulk_max": float(bu_bulk.max()), "log_z_out": float(log_z_out),
                       "log_z_in": float(log_z_in), "bu_y_min": float(bu_y.min()),
                       "acc_in_per_trial": float(a_in.mean()), "acc_out_per_trial": float(a_out.mean()),
                       "p_batch_in": p_batch_in, "K": int(len(bu_y))}
                rows.append(row)
                log("probe", json.dumps(row))
        summary["probe_rows"] = rows
        summary["t_eval_s"] = float(np.median(t_eval))
        json.dump(summary, open(out / "s1_summary.json", "w"), indent=1)

    agg = {}
    for n in regions:
        for label in ("holo", "apo"):
            rr = [r for r in rows if r["region"] == n and r["state"] == label]
            if rr:
                agg[f"{n}/{label}"] = {
                    "acc_in_per_trial_mean": float(np.mean([r["acc_in_per_trial"] for r in rr])),
                    "acc_out_per_trial_mean": float(np.mean([r["acc_out_per_trial"] for r in rr])),
                    "p_batch_in_mean": float(np.mean([r["p_batch_in"] for r in rr])),
                    "p_batch_in_median": float(np.median([r["p_batch_in"] for r in rr])),
                    "n_sources": len(rr),
                }
    summary["probe_aggregate"] = agg
    json.dump(summary, open(out / "s1_summary.json", "w"), indent=1)
    log("aggregate", json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()

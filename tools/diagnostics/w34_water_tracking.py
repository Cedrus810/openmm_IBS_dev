"""w3+w4 start-structure test with ghost-volume occupancy tracking.

Same experiment as abfe-benchmark/openmm_IBS/hysteresis_w34/w34_hysteresis.py
(independent fixed-λ states over λ_vdw 0.4 -> 0, production Hamiltonian from
ibs_engine.build_ibs_dual_system, coupled start vs λ=0-pre-equilibrated start),
imported from there unchanged, plus per saved frame:

  n_wat  : water O within --r-ghost of any ligand heavy atom (water inside the
           ghost-ligand volume)
  n_prot : protein heavy atoms within --r-ghost of any ligand heavy atom

Question: when the coupled start and the decoupled start disagree on ΔF, is
the disagreement carried by how much water sits in the ghost volume?

Read-only on the run directory; writes only to --out.  State files keep the
w34_hysteresis format (u_kn, state, lam, t_ps) so its analyze() works as is.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")
sys.path.insert(0, "/home/ruigengji/ABFE_IBS/ABFE_IBS")
import numpy as np
import openmm
import openmm.unit as u

W34 = "/home/ruigengji/abfe-benchmark/openmm_IBS/hysteresis_w34/w34_hysteresis.py"
_spec = importlib.util.spec_from_file_location("w34_hysteresis", W34)
w34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w34)


def ghost_counts(pos, box_diag, lig_heavy, water_O, prot_heavy, r):
    L = pos[lig_heavy]

    def count(idx):
        d = pos[idx][:, None, :] - L[None, :, :]
        d -= box_diag * np.round(d / box_diag)
        return int(((d * d).sum(-1) < r * r).any(1).sum())

    return count(water_O), count(prot_heavy)


def run_direction(args, ie, leg_data, direction, platform):
    native, topology, lig, rest, pos0, box0 = leg_data
    idx = list(range(args.first, args.last + 1))
    lams = [w34.lam_of(i) for i in idx]
    K = len(lams)
    out = os.path.join(args.out, direction)
    os.makedirs(out, exist_ok=True)
    json.dump({"lambda_indices": idx, "lambdas_vdw": lams, "args": vars(args)},
              open(os.path.join(out, "meta.json"), "w"), indent=1)

    atoms = list(topology.atoms())
    ligset = set(int(i) for i in lig)
    lig_heavy = np.array([i for i in lig if atoms[i].element.symbol != "H"])
    water_O = np.array([a.index for a in atoms if a.residue.name == "HOH" and a.element.symbol == "O"])
    prot_heavy = np.array([a.index for a in atoms if a.index not in ligset and a.element.symbol != "H"
                           and a.residue.name not in ("HOH", "NA", "CL")])

    ev_sys, ev_wrap, ev_cvf = w34.build(ie, native, topology, lig, rest, lams, box0, "hyst_eval")
    lrc = np.asarray(ev_wrap.lj_tail_lrc_coeff_kj_mol if ev_wrap.lj_tail_lrc_coeff_kj_mol is not None
                     else np.zeros(K))
    names = [ev_cvf.getCollectiveVariableName(i) for i in range(ev_cvf.getNumCollectiveVariables())]
    cv_cols = [names.index(f"cv_{k}_int") for k in range(K)]
    ev_ctx, _ = w34.make_ctx(ev_sys, platform, 1)

    def dyn_for(k):
        other = lams[k + 1] if k + 1 < K else lams[k - 1]
        s, _, cvf = w34.build(ie, native, topology, lig, rest, [lams[k], other], box0, f"hyst_dyn{k}")
        cvf.setEnergyFunction("cv_0_int")
        return s

    start_pos, start_box = pos0, box0
    if direction == "decoupled":
        start_pos, start_box = w34._decoupled_start(args, ie, leg_data, lams, out, platform, dyn_for)

    save_steps = int(round(args.save_ps * 500))
    n_frames = int(round(args.prod_ps / args.save_ps))
    for k in range(K):
        f = os.path.join(out, f"state_{idx[k]:02d}.npz")
        if os.path.exists(f):
            print(f"[{direction}] state {idx[k]} (λ={lams[k]:.2f}) done, skip", flush=True)
            continue
        s = dyn_for(k)
        ctx, integ = w34.make_ctx(s, platform, args.seed + 17 * k + (0 if direction == "coupled" else 5000))
        w34.set_state(ctx, start_pos, start_box)
        openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, 2000)
        ctx.setVelocitiesToTemperature(300 * u.kelvin, args.seed + k)
        box_diag = np.diag(np.asarray(start_box))
        V = float(np.linalg.det(np.asarray(start_box)))
        t0 = time.time()
        integ.step(int(args.eq_ps * 500))
        u_kn = np.empty((K, n_frames))
        n_wat = np.empty(n_frames, dtype=np.int32)
        n_prot = np.empty(n_frames, dtype=np.int32)
        for n in range(n_frames):
            integ.step(save_steps)
            st = ctx.getState(getPositions=True)
            p = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
            ev_ctx.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
            ev_ctx.setPositions(p)
            cv = np.asarray(ev_cvf.getCollectiveVariableValues(ev_ctx))
            u_kn[:, n] = (cv[cv_cols] + lrc / V) / w34.KT
            n_wat[n], n_prot[n] = ghost_counts(np.asarray(p), box_diag, lig_heavy, water_O, prot_heavy,
                                               args.r_ghost)
        np.savez(f, u_kn=u_kn, state=k, lam=lams[k],
                 t_ps=args.eq_ps + args.save_ps * (1 + np.arange(n_frames)),
                 n_wat=n_wat, n_prot=n_prot, r_ghost=args.r_ghost)
        print(f"[{direction}] state {idx[k]} λ={lams[k]:.2f}: {args.eq_ps:.0f}+{args.prod_ps:.0f} ps "
              f"in {time.time()-t0:.0f}s  n_wat mean {n_wat.mean():.2f} (last 25% {n_wat[-n_frames//4:].mean():.2f})"
              f"  n_prot mean {n_prot.mean():.2f}", flush=True)
        del ctx, integ


def occupancy_report(args):
    rows = {}
    for direction in ("coupled", "decoupled"):
        d = os.path.join(args.out, direction)
        if not os.path.isdir(d):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        for i, lam in zip(meta["lambda_indices"], meta["lambdas_vdw"]):
            f = os.path.join(d, f"state_{i:02d}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            w, pr = z["n_wat"], z["n_prot"]
            q = max(1, len(w) // 4)
            rows.setdefault(i, {"lam": lam})[direction] = {
                "n_wat_mean": float(w.mean()), "n_wat_first25": float(w[:q].mean()),
                "n_wat_last25": float(w[-q:].mean()), "n_wat_min": int(w.min()), "n_wat_max": int(w.max()),
                "n_prot_mean": float(pr.mean()),
            }
    print("\nghost-volume occupancy (r_ghost = %.2f nm): n_wat mean [first 25%% -> last 25%%], n_prot mean"
          % args.r_ghost)
    for i in sorted(rows):
        r = rows[i]
        line = f"  λ={r['lam']:.2f}"
        for direction in ("coupled", "decoupled"):
            if direction in r:
                x = r[direction]
                line += (f"   {direction:9s} {x['n_wat_mean']:5.2f} [{x['n_wat_first25']:5.2f} -> "
                         f"{x['n_wat_last25']:5.2f}]  prot {x['n_prot_mean']:4.1f}")
        print(line)
    json.dump(rows, open(os.path.join(args.out, "occupancy.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--leg", choices=["complex", "solvent"], default="complex")
    p.add_argument("--direction", choices=["coupled", "decoupled", "both"], default="both")
    p.add_argument("--first", type=int, default=15, help="first λ index of the 26-grid (15 = λ 0.4)")
    p.add_argument("--last", type=int, default=25, help="last λ index (25 = λ 0)")
    p.add_argument("--eq-ps", type=float, default=70.0)
    p.add_argument("--prod-ps", type=float, default=1000.0)
    p.add_argument("--decoupled-eq-ps", type=float, default=2000.0)
    p.add_argument("--decoupled-start", default=None)
    p.add_argument("--save-ps", type=float, default=1.0)
    p.add_argument("--r-ghost", type=float, default=0.30)
    p.add_argument("--block-ps", type=float, default=250.0)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--platform", default="CUDA")
    p.add_argument("--out", required=True)
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.analyze_only:
        import ibs_engine as ie
        platform = openmm.Platform.getPlatformByName(args.platform)
        leg_data = w34.load_leg(args.run_dir, args.leg)
        for d in (["coupled", "decoupled"] if args.direction == "both" else [args.direction]):
            run_direction(args, ie, leg_data, d, platform)
    occupancy_report(args)
    if all(os.path.isdir(os.path.join(args.out, d)) for d in ("coupled", "decoupled")):
        w34.analyze(args)


if __name__ == "__main__":
    main()

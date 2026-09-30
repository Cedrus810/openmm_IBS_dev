"""Which pocket degree of freedom decides the stage-2 segment "basin"?

Stage-2 (vdW) segments of the same window land in one of two discrete values
of the window ΔF and never switch within a segment.  For every
(run, window, segment) this computes

  * the segment's window ΔF from its own frames
    (F_k = -kT ln <exp(-(u_k - bias)/kT)>, ΔF = F_last - F_first)
  * per-segment means of candidate structural features from the coordinate
    sidecar (dual_window_*_vdw_coords.npz, non-water atoms):
      - min distance ligand heavy atoms <-> side-chain heavy atoms of every
        residue within --shell of the ligand
      - chi1 of the same residues (circular mean)

and ranks features by how well they separate the segment ΔF within each
window (|Pearson r| over all segments of all runs; per-window, so the
window-to-window offset does not enter).

Read-only on the run directories.
"""

import argparse
import glob
import json
import os

import numpy as np

KT = 0.0083144626 * 300.0


def lme(a):
    m = a.max()
    return m + np.log(np.exp(a - m).mean())


def seg_dF(u, b):
    F = np.array([-KT * lme(-(u[k] - b) / KT) for k in range(u.shape[0])])
    return F[-1] - F[0]


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


CHI1_4TH = {"SER": "OG", "THR": "OG1", "CYS": "SG", "VAL": "CG1", "ILE": "CG1", "LEU": "CG", "ASP": "CG",
            "ASN": "CG", "GLU": "CG", "GLN": "CG", "LYS": "CG", "ARG": "CG", "HIS": "CG", "HID": "CG",
            "HIE": "CG", "HIP": "CG", "PHE": "CG", "TYR": "CG", "TRP": "CG", "MET": "CG"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=run_dir")
    ap.add_argument("--shell", type=float, default=0.8)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import mdtraj as md

    rows = []
    feat_names = None
    for spec in args.runs:
        label, run = spec.split("=", 1)
        top = md.load_topology(os.path.join(run, "topology.cif"))
        lig = json.load(open(os.path.join(run, "ligand_indices.json")))["ligand_indices"]
        ligset = set(lig)
        lig_heavy = [i for i in lig if top.atom(i).element.symbol != "H"]
        segdirs = sorted([d for d in glob.glob(os.path.join(run, "vanishing*")) if os.path.isdir(d)],
                         key=lambda d: 1 if os.path.basename(d) == "vanishing" else int(d.rsplit("_", 1)[1]))
        # pocket residues from the first coords file of window 0
        z0 = np.load(os.path.join(segdirs[0], "dual_window_0_vdw_coords.npz"))
        pos_of = {int(a): k for k, a in enumerate(z0["atom_indices"])}
        if feat_names is None:
            x = z0["xyz"][0]
            L = np.diag(z0["box"][0])
            lh = x[[pos_of[i] for i in lig_heavy]]
            cand = []
            for r in top.residues:
                if not r.is_protein:
                    continue
                sc = [a.index for a in r.atoms if a.element.symbol != "H"
                      and a.name not in ("N", "CA", "C", "O") and a.index not in ligset]
                if not sc:
                    continue
                d = x[[pos_of[i] for i in sc]][:, None, :] - lh[None]
                d -= L * np.round(d / L)
                if np.sqrt((d * d).sum(-1)).min() < args.shell:
                    names = {a.name: a.index for a in r.atoms}
                    chi = None
                    if r.name in CHI1_4TH and all(k in names for k in ("N", "CA", "CB", CHI1_4TH[r.name])):
                        chi = [names["N"], names["CA"], names["CB"], names[CHI1_4TH[r.name]]]
                    cand.append((f"{r.name}{r.resSeq}", sc, chi))
            feat_names = [f"d:{c[0]}" for c in cand] + [f"chi1:{c[0]}" for c in cand if c[2]]
            cand_ref = cand
        for d in segdirs:
            seg = os.path.basename(d)
            for w in range(10):
                fe = os.path.join(d, f"dual_window_{w}_vdw_energies.npy")
                fc = os.path.join(d, f"dual_window_{w}_vdw_coords.npz")
                if not os.path.exists(fe) or not os.path.exists(fc):
                    continue
                u = np.load(fe)
                b = np.load(os.path.join(d, f"dual_window_{w}_vdw_bias.npy"))
                z = np.load(fc)
                pos_of = {int(a): k for k, a in enumerate(z["atom_indices"])}
                X = z["xyz"].astype(np.float64)
                L = np.array([np.diag(bx) for bx in z["box"]])[:, None, None, :]
                lh = X[:, [pos_of[i] for i in lig_heavy]]
                feats = []
                for name, sc, chi in cand_ref:
                    dd = X[:, [pos_of[i] for i in sc]][:, :, None, :] - lh[:, None]
                    dd -= L * np.round(dd / L)
                    feats.append(np.sqrt((dd * dd).sum(-1)).min((1, 2)).mean())
                for name, sc, chi in cand_ref:
                    if chi:
                        a = dihedral(*[X[:, pos_of[i]] for i in chi])
                        feats.append(float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))))
                rows.append({"run": label, "seg": seg, "window": w, "dF": float(seg_dF(u, b)),
                             "feats": [float(f) for f in feats]})
        print(f"{label}: {sum(1 for r in rows if r['run'] == label)} window-segments", flush=True)

    report = {}
    for w in sorted({r["window"] for r in rows}):
        rs = [r for r in rows if r["window"] == w]
        y = np.array([r["dF"] for r in rs])
        F = np.array([r["feats"] for r in rs])
        scores = []
        for j, name in enumerate(feat_names):
            f = F[:, j]
            if name.startswith("chi1:"):
                # use cos/sin of the angle; take the better-correlated one
                c = [np.corrcoef(np.cos(np.radians(f)), y)[0, 1], np.corrcoef(np.sin(np.radians(f)), y)[0, 1]]
                r_ = max(c, key=lambda v: abs(v) if np.isfinite(v) else -1)
            else:
                r_ = np.corrcoef(f, y)[0, 1] if f.std() > 0 else np.nan
            scores.append((abs(r_) if np.isfinite(r_) else 0.0, r_, name, j))
        scores.sort(reverse=True)
        print(f"\nwindow {w}: {len(rs)} segments, dF range {y.min():.1f}..{y.max():.1f} kJ")
        for s, r_, name, j in scores[:args.top]:
            print(f"  r={r_:+.2f}  {name:14s}  values(low dF → high dF): "
                  + " ".join(f"{F[k, j]:.2f}" for k in np.argsort(y)))
        report[w] = [{"feature": n, "r": float(r_)} for s, r_, n, j in scores[:args.top]]
    if args.out:
        json.dump({"rows": rows, "features": feat_names, "ranking": report}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

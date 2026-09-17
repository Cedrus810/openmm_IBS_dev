import json, glob, os
import numpy as np

ROOT = "/home/ruigengji/abfe-benchmark/openmm_IBS/runs"
SYS = ("brd4", "cmet", "jnk1", "p38")          # 只用蛋白体系

def integ_g(lam_a, lam_b, pl, g):
    """∫g dλ between two lambda values, trapezoid on the pilot grid."""
    o = np.argsort(pl); pl, g = np.asarray(pl)[o], np.asarray(g)[o]
    cum = np.concatenate(([0.0], np.cumsum(0.5*(g[:-1]+g[1:])*np.diff(pl))))
    return abs(float(np.interp(lam_b, pl, cum) - np.interp(lam_a, pl, cum)))

def _sc(v, agg):
    """逐态 list ⟹ 归约成标量；读不成数返回 None。"""
    vals = v if isinstance(v, (list, tuple)) else [v]
    out = [float(x) for x in vals
           if isinstance(x, (int, float)) and float(x) == float(x)]
    return agg(out) if out else None

rows = []
for d in sorted(glob.glob(f"{ROOT}/*/rep[0-9]")):
    name = os.path.basename(os.path.dirname(d))
    if not name.startswith(SYS):
        continue
    ck = os.path.join(d, "checkpoints")
    pre = f"{ck}/preopt_dual_vanishing.json"
    st = next((p for p in (f"{ck}/stage2_vanishing.json",
                           f"{ck}/stage2_vanishing_autonomous_inprogress.json")
               if os.path.exists(p)), None)
    if not (os.path.exists(pre) and st):
        continue
    P = json.load(open(pre)); S = json.load(open(st))
    pd = P.get("path_diagnostics") or {}
    pl, g = pd.get("pilot_lambdas"), pd.get("metric_g")
    edges = ((pd.get("subdomain_allocation") or {}).get("edge_free_energy_kJ_mol"))
    lam = P.get("lambdas_var")
    wods = ((S.get("diagnostics") or {}).get("window_overlap_diagnostics")) or []
    if not (pl and g and lam and wods):
        continue
    for w in wods:
        rng = w.get("window_range")
        if not rng or len(rng) != 2:
            continue
        a, b = int(rng[0]), int(rng[1])
        if b > len(lam):
            continue
        K = b - a
        df = (float(np.sum(np.abs(edges[a:b-1]))) if edges and b-1 <= len(edges) else None)
        mx = (float(np.max(np.abs(edges[a:b-1]))) if edges and b-1 <= len(edges) and b-1 > a else None)
        rows.append(dict(
            run=f"{name}/{os.path.basename(d)}", win=int(w.get("window_index", -1)),
            K=K,
            Ig=integ_g(lam[a], lam[b-1], pl, g),
            dF=df, maxedge=mx,
            raw_ess=_sc(w.get("raw_min_absolute_ess"), min),
            ess=_sc(w.get("absolute_ess"), min),
            ndec=_sc(w.get("n_frames_decorrelated"), min),
            top1=_sc(w.get("top1pct_raw_weight"), max),
        ))

json.dump(rows, open("/tmp/claude-1000/-home-ruigengji-ABFE-IBS-ABFE-IBS/94607659-87fe-4214-83a3-69ae1cc0ae2b/scratchpad/rows.json","w"))
print(f"{len(rows)} 个窗口，来自 {len({r['run'] for r in rows})} 个 rep")
print(f"{'run':28s} {'w':>2s} {'K':>2s} {'∫g':>7s} {'ΔF':>7s} {'maxE':>6s} {'rawESS':>7s} {'top1%':>6s}")
for r in rows:
    f = lambda v, p=1: ("  n/a" if v is None else f"{v:.{p}f}")
    print(f"{r['run']:28s} {r['win']:2d} {r['K']:2d} {f(r['Ig']):>7s} {f(r['dF']):>7s} "
          f"{f(r['maxedge']):>6s} {f(r['raw_ess']):>7s} {f(r['top1'],3):>6s}")

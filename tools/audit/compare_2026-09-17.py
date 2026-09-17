import json, glob, os
import numpy as np
ROOT="/home/ruigengji/abfe-benchmark/openmm_IBS/runs"
SYS=("brd4","cmet","jnk1","p38")

def cum(pl, f):
    o=np.argsort(pl); pl=np.asarray(pl)[o]; f=np.asarray(f)[o]
    return pl, np.concatenate(([0.0], np.cumsum(0.5*(f[:-1]+f[1:])*np.diff(pl))))

print(f"{'run':22s} {'w':>2s} {'K':>2s} {'∫g':>7s} {'L=∫√g':>7s} {'ΔF':>7s}")
for d in sorted(glob.glob(f"{ROOT}/*/rep[0-9]")):
    name=os.path.basename(os.path.dirname(d))
    if not name.startswith(SYS): continue
    ck=os.path.join(d,"checkpoints")
    pre=f"{ck}/preopt_dual_vanishing.json"
    if not os.path.exists(pre): continue
    P=json.load(open(pre)); pd=P.get("path_diagnostics") or {}
    pl,g=pd.get("pilot_lambdas"),pd.get("metric_g")
    lam=P.get("lambdas_var"); rngs=P.get("window_ranges")
    edges=((pd.get("subdomain_allocation") or {}).get("edge_free_energy_kJ_mol"))
    if not(pl and g and lam and rngs): continue
    g=np.asarray(g,dtype=float)
    plg,cg = cum(pl,g)
    pls,cs = cum(pl,np.sqrt(np.maximum(g,1e-12)))
    I=lambda a,b,p,c: abs(float(np.interp(lam[b-1],p,c)-np.interp(lam[a],p,c)))
    for i,(a,b) in enumerate(rngs):
        a,b=int(a),int(b)
        dF=(float(np.sum(np.abs(edges[a:b-1]))) if edges and b-1<=len(edges) else float('nan'))
        print(f"{name+'/'+os.path.basename(d):22s} {i:2d} {b-a:2d} "
              f"{I(a,b,plg,cg):7.1f} {I(a,b,pls,cs):7.2f} {dF:7.1f}")

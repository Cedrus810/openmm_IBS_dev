import json, glob, os
import numpy as np
ROOT="/home/ruigengji/abfe-benchmark/openmm_IBS/runs"
EXP={"brd4_ligand1":-9.60,"brd4_ligand2":-7.40,"cmet_ligand1":None,"cmet_ligand2":-7.34,
     "jnk1_ligand1":-8.52,"jnk1_ligand2":None,"p38_ligand1":-8.98,"p38_ligand2":-11.87}
G=["min_overlap","raw_min_overlap","min_absolute_ess","raw_min_absolute_ess",
   "max_top1pct_raw_weight","min_decorrelated_samples","max_endpoint_uncertainty_kJ_mol",
   "min_occupancy_normalized","split_half_max_window_z","max_common_mode_log_sigma_kT"]
rows=[]
for d in sorted(glob.glob(f"{ROOT}/*/rep[0-9]")):
    sysname=os.path.basename(os.path.dirname(d))
    fb=os.path.join(d,"final_binding_results.json")
    if not os.path.exists(fb) or EXP.get(sysname) is None: continue
    F=json.load(open(fb))
    dg=F.get("delta_G_bind_kcal_mol")
    if dg is None: continue
    r=dict(run=f"{sysname}/{os.path.basename(d)}", dG=float(dg),
           dev=float(dg)-EXP[sysname], err=F.get("total_error_kJ_mol"))
    # 两条腿的 stage2 门读数：取更差的一侧
    for leg in ("complex","solvent"):
        p=os.path.join(d,"checkpoints",f"stage2_vanishing.json")
        if os.path.exists(p):
            dd=(json.load(open(p)).get("diagnostics") or {})
            for k in G:
                v=dd.get(k)
                if isinstance(v,(int,float)): r.setdefault(k,float(v))
    rows.append(r)

print(f"n = {len(rows)} 个跑完且有实验值的 run\n")
hdr=f"{'run':22s} {'ΔG':>7s} {'偏差':>7s} " + " ".join(f"{k[:11]:>11s}" for k in G if any(k in r for r in rows))
print(hdr)
GG=[k for k in G if any(k in r for r in rows)]
for r in rows:
    print(f"{r['run']:22s} {r['dG']:7.2f} {r['dev']:7.2f} " +
          " ".join(("        n/a" if r.get(k) is None else f"{r[k]:11.3f}") for k in GG))

def spear(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean()
    return float(rx@ry/np.sqrt((rx@rx)*(ry@ry)))
print("\n每道门 vs |与实验的偏差| 的 Spearman ρ（门要有用 ⟹ |ρ| 明显大于 0）")
ad=[abs(r['dev']) for r in rows]
for k in GG:
    v=[r.get(k) for r in rows]
    ok=[i for i,x in enumerate(v) if x is not None]
    if len(ok)<5: print(f"  {k:36s} n={len(ok)} 太少"); continue
    print(f"  {k:36s} ρ = {spear([v[i] for i in ok],[ad[i] for i in ok]):+.3f}   (n={len(ok)})")
print(f"\n  {'报告的 total_error':36s} ρ = "
      f"{spear([r['err'] for r in rows],ad):+.3f}   ← σ 预测得了自己的误差吗")

import json, glob, os, itertools
import numpy as np
RUNS="/home/ruigengji/abfe-benchmark/openmm_IBS/runs"
EXP={"brd4_ligand1":-9.60,"brd4_ligand2":-7.40,"cmet_ligand2":-7.34,
     "jnk1_ligand1":-8.52,"p38_ligand1":-8.98,"p38_ligand2":-11.87}
G=["min_overlap","raw_min_overlap","min_absolute_ess","raw_min_absolute_ess",
   "max_top1pct_raw_weight","min_decorrelated_samples","max_endpoint_uncertainty_kJ_mol",
   "min_occupancy_normalized","max_common_mode_log_sigma_kT"]
rows=[]
for d in sorted(glob.glob(f"{RUNS}/*/rep[0-9]")):
    s=os.path.basename(os.path.dirname(d))
    fb=os.path.join(d,"final_binding_results.json"); st=os.path.join(d,"checkpoints","stage2_vanishing.json")
    if not(os.path.exists(fb) and os.path.exists(st) and s in EXP): continue
    F=json.load(open(fb)); dd=(json.load(open(st)).get("diagnostics") or {})
    if F.get("delta_G_bind_kcal_mol") is None: continue
    r={"dev":abs(float(F["delta_G_bind_kcal_mol"])-EXP[s])}
    for k in G:
        v=dd.get(k)
        if isinstance(v,(int,float)): r[k]=float(v)
    if all(k in r for k in G): rows.append(r)
def sp(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean(); d=np.sqrt((rx@rx)*(ry@ry))
    return float(rx@ry/d) if d>0 else float("nan")
n=len(rows); print(f"n = {n} 个 run（九道门读数齐全）\n")
print("门与门之间的 Spearman |ρ|（它们是独立证据吗）")
M=np.zeros((len(G),len(G)))
for i,a in enumerate(G):
    for j,b in enumerate(G):
        M[i,j]=sp([r[a] for r in rows],[r[b] for r in rows])
off=[abs(M[i,j]) for i in range(len(G)) for j in range(i+1,len(G))]
print(f"  两两 |ρ| 中位 = {np.median(off):.2f}   最大 = {max(off):.2f}   "
      f"|ρ|>0.7 的对数 = {sum(1 for x in off if x>0.7)}/{len(off)}")
ev=np.linalg.eigvalsh(np.nan_to_num(M,nan=0.0))[::-1]
ev=np.clip(ev,0,None); tot=ev.sum()
k=int(np.argmax(np.cumsum(ev)/tot>=0.9))+1
print(f"  秩相关矩阵前 1 个主成分解释 {ev[0]/tot:.0%}；解释 90% 需要 {k} 个主成分")
print(f"  ⟹ 九道门大致相当于 **{k} 个独立方向**，不是 9 份独立证据")
print(f"\n在 n={n} 下，Spearman |ρ| 要多大才显著（双侧 0.05，近似 1.96/sqrt(n-1)）："
      f" |ρ| > {1.96/np.sqrt(n-1):.2f}")
print(f"  实测最大 |ρ|(门, |偏差|) = "
      f"{max(abs(sp([r[k_] for r in rows],[r['dev'] for r in rows])) for k_ in G):.3f}")

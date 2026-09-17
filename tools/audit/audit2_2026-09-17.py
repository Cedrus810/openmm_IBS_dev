import json, glob, os, csv, collections
import numpy as np
RUNS="/home/ruigengji/abfe-benchmark/openmm_IBS/runs"
BASE="/home/ruigengji/abfe-benchmark/openmm_IBS"
PROT=("brd4","cmet","jnk1","p38")
EXP={"brd4_ligand1":-9.60,"brd4_ligand2":-7.40,"cmet_ligand2":-7.34,
     "jnk1_ligand1":-8.52,"p38_ligand1":-8.98,"p38_ligand2":-11.87}
def spear(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean(); d=np.sqrt((rx@rx)*(ry@ry))
    return float(rx@ry/d) if d>0 else float("nan")

res=[]
for f in sorted(glob.glob(f"{RUNS}/**/final_binding_results.json", recursive=True)):
    rel=os.path.relpath(f,RUNS); s=rel.split("/")[0]; who=rel.split("/")[1]
    if not s.startswith(PROT) or s not in EXP: continue
    F=json.load(open(f)); dg=F.get("delta_G_bind_kcal_mol"); e=F.get("total_error_kJ_mol")
    if dg is None: continue
    arch="_backup" in who or "_trash" in who
    rep=who if not arch else ("rep"+who.split("rep")[1][0] if "rep" in who else who)
    res.append(dict(sys=s,rep=rep,arch=arch,dG=float(dg),
                    sig=(float(e)/4.184 if e is not None else None),dev=float(dg)-EXP[s]))
act=[r for r in res if not r['arch']]
print(f"===== A) 现役 ΔG，蛋白体系 n = {len(act)} =====")
d=[r['dev'] for r in act]
print(f"ME = {np.mean(d):+.2f}   MAE = {np.mean(np.abs(d)):.2f}   "
      f"RMSE = {np.sqrt(np.mean(np.square(d))):.2f} kcal/mol   负偏差 {sum(1 for x in d if x<0)}/{len(d)}")

print(f"\n===== B1) 自报 σ vs 真实重复 SD（只用现役 rep，独立重复） =====")
by=collections.defaultdict(list)
for r in act: by[r['sys']].append(r)
for s,rs in sorted(by.items()):
    if len(rs)<3: print(f"  {s:16s} 只有 {len(rs)} 个重复，算不了"); continue
    dd=[x['dG'] for x in rs]; sg=[x['sig'] for x in rs if x['sig']]
    sd=float(np.std(dd,ddof=1)); ms=float(np.median(sg))
    print(f"  {s:16s} n={len(rs)}  重复SD={sd:.2f}  自报σ中位={ms:.2f}  "
          f"**低估 {sd/ms:.1f}×**   偏离实验 {np.mean([x['dev'] for x in rs]):+.2f}")

print(f"\n===== B2) 同一个 rep 的归档版 vs 现役版（应当是同一份计算） =====")
pair=collections.defaultdict(dict)
for r in res: pair[(r['sys'],r['rep'])]["A" if r['arch'] else "L"]=r
print(f"{'体系/rep':24s} {'归档':>8s} {'现役':>8s} {'差':>7s} {'自报σ':>7s} {'差/σ':>6s}")
gaps=[]
for (s,rp),v in sorted(pair.items()):
    if "A" not in v or "L" not in v: continue
    g=abs(v["A"]['dG']-v["L"]['dG']); sg=v["L"]['sig'] or float('nan')
    gaps.append(g/sg); print(f"{s+'/'+rp:24s} {v['A']['dG']:8.2f} {v['L']['dG']:8.2f} "
                             f"{g:7.2f} {sg:7.2f} {g/sg:6.1f}×")
if gaps: print(f"\n同一份计算重分析一次就差 {np.median(gaps):.1f}× 自报 σ（中位）")

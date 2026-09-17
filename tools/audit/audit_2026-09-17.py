import json, glob, os, csv, collections
import numpy as np
RUNS="/home/ruigengji/abfe-benchmark/openmm_IBS/runs"
BASE="/home/ruigengji/abfe-benchmark/openmm_IBS"
PROT=("brd4","cmet","jnk1","p38")

EXP={}
for r in csv.DictReader(open(f"{BASE}/results.csv")):
    try: EXP[r["system"]]=float(r["exp"])
    except (TypeError,ValueError): pass

def spear(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean()
    d=np.sqrt((rx@rx)*(ry@ry))
    return float(rx@ry/d) if d>0 else float("nan")

# ---------- A) 全部 ΔG（含归档） ----------
res=[]
for f in sorted(glob.glob(f"{RUNS}/**/final_binding_results.json", recursive=True)):
    rel=os.path.relpath(f,RUNS); sysname=rel.split("/")[0]
    if not sysname.startswith(PROT) or sysname not in EXP: continue
    F=json.load(open(f))
    dg=F.get("delta_G_bind_kcal_mol"); err=F.get("total_error_kJ_mol")
    if dg is None: continue
    res.append(dict(sys=sysname, who=rel.split("/")[1], dG=float(dg),
                    sig=float(err)/4.184 if err is not None else None,
                    dev=float(dg)-EXP[sysname],
                    arch=("_backup" in rel or "_trash" in rel)))
print(f"===== A) 全部 ΔG（含归档），蛋白体系 n = {len(res)} =====")
print(f"{'体系':16s} {'来源':38s} {'ΔG':>7s} {'σ':>6s} {'实验':>6s} {'偏差':>7s}")
for r in sorted(res,key=lambda x:(x['sys'],x['who'])):
    print(f"{r['sys']:16s} {r['who'][:38]:38s} {r['dG']:7.2f} "
          f"{(r['sig'] if r['sig'] is not None else float('nan')):6.2f} "
          f"{EXP[r['sys']]:6.2f} {r['dev']:7.2f}")
dev=[r['dev'] for r in res]
print(f"\nME = {np.mean(dev):+.2f}   MAE = {np.mean(np.abs(dev)):.2f}   "
      f"RMSE = {np.sqrt(np.mean(np.square(dev))):.2f} kcal/mol")
print(f"偏差为负的：{sum(1 for d in dev if d<0)}/{len(dev)}")

# ---------- B) 报告的 σ vs 真实重复离散 ----------
print(f"\n===== B) 自报 σ 对不对 =====")
by=collections.defaultdict(list)
for r in res: by[r['sys']].append(r)
print(f"{'体系':16s} {'n':>2s} {'ΔG 范围':>16s} {'重复SD':>7s} {'自报σ中位':>9s} {'低估倍数':>8s}")
ratios=[]
for s,rs in sorted(by.items()):
    if len(rs)<2: continue
    d=[x['dG'] for x in rs]; sg=[x['sig'] for x in rs if x['sig'] is not None]
    sd=float(np.std(d,ddof=1)); ms=float(np.median(sg)) if sg else float('nan')
    ratios.append(sd/ms if ms==ms and ms>0 else float('nan'))
    print(f"{s:16s} {len(rs):2d} {min(d):7.2f}~{max(d):<8.2f} {sd:7.2f} {ms:9.2f} {sd/ms:8.1f}×")
rr=[x for x in ratios if x==x]
if rr: print(f"\n低估倍数中位数 = {np.median(rr):.1f}×  ⟸ 自报 σ 比真实重复离散小这么多")

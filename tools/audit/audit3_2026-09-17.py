import json, glob, os, collections
import numpy as np
RUNS="/home/ruigengji/abfe-benchmark/openmm_IBS/runs"; PROT=("brd4","cmet","jnk1","p38")
def _sc(v,agg):
    vals=v if isinstance(v,(list,tuple)) else [v]
    o=[float(x) for x in vals if isinstance(x,(int,float)) and float(x)==float(x)]
    return agg(o) if o else None
def cum(pl,f):
    o=np.argsort(pl); pl=np.asarray(pl)[o]; f=np.asarray(f)[o]
    return pl,np.concatenate(([0.0],np.cumsum(0.5*(f[:-1]+f[1:])*np.diff(pl))))
def spear(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean(); d=np.sqrt((rx@rx)*(ry@ry))
    return float(rx@ry/d) if d>0 else float("nan")
def rank(v):
    v=np.asarray(v,float); r=np.argsort(np.argsort(v)).astype(float); return r-r.mean()
def partial(x,y,z):
    X,Y,Z=rank(x),rank(y),rank(z)
    rx=X-Z*(X@Z)/(Z@Z); ry=Y-Z*(Y@Z)/(Z@Z)
    return float(rx@ry/np.sqrt((rx@rx)*(ry@ry)))

R=[]
for st in sorted(glob.glob(f"{RUNS}/**/stage2_vanishing*.json", recursive=True)):
    rel=os.path.relpath(st,RUNS)
    if not rel.startswith(PROT): continue
    ck=os.path.dirname(st); pre=os.path.join(ck,"preopt_dual_vanishing.json")
    if not os.path.exists(pre): continue
    try: P=json.load(open(pre)); S=json.load(open(st))
    except (ValueError,OSError): continue
    pdg=P.get("path_diagnostics") or {}
    pl,g,lam=pdg.get("pilot_lambdas"),pdg.get("metric_g"),P.get("lambdas_var")
    ed=((pdg.get("subdomain_allocation") or {}).get("edge_free_energy_kJ_mol"))
    wods=((S.get("diagnostics") or {}).get("window_overlap_diagnostics")) or []
    if not(pl and g and lam and wods and ed): continue
    g=np.asarray(g,float); plg,cg=cum(pl,g); pls,cs=cum(pl,np.sqrt(np.maximum(g,1e-12)))
    I=lambda a,b,p,c: abs(float(np.interp(lam[b-1],p,c)-np.interp(lam[a],p,c)))
    for pos,w in enumerate(wods):
        rg=w.get("window_range")
        if not rg or len(rg)!=2: continue
        a,b=int(rg[0]),int(rg[1])
        if b>len(lam) or b-1>len(ed) or b-1<=a: continue
        ess=_sc(w.get("raw_min_absolute_ess"),min); t1=_sc(w.get("top1pct_raw_weight"),max)
        nd=_sc(w.get("n_frames_decorrelated"),min); si=_sc(w.get("statistical_inefficiency"),max)
        if ess is None or t1 is None: continue
        R.append(dict(src=rel,pos=pos,K=b-a,Ig=I(a,b,plg,cg),L=I(a,b,pls,cs),
                      dF=float(np.sum(np.abs(ed[a:b-1]))),ess=ess,top1=t1,nd=nd,g=si))
print(f"===== C) 窗口级审查：n = {len(R)} 个窗口 / {len({r['src'] for r in R})} 份 stage2 结果 =====\n")
K=[r['K'] for r in R]; L=[r['L'] for r in R]; Ig=[r['Ig'] for r in R]
dF=[r['dF'] for r in R]; ess=[r['ess'] for r in R]; t1=[r['top1'] for r in R]
pos=[r['pos'] for r in R]
print(f"{'预测量':>8s} {'ρ(·,rawESS)':>12s} {'ρ(·,top1%)':>11s}   {'期望':>6s}")
for v,lab in ((Ig,"∫g"),(L,"L=∫√g"),(dF,"ΔF"),(K,"K"),(pos,"窗口序号")):
    print(f"{lab:>8s} {spear(v,ess):12.3f} {spear(v,t1):11.3f}   {'负/正':>6s}")
print(f"\n控制窗口序号后：ρ(K,rawESS|pos) = {partial(K,ess,pos):.3f}   "
      f"ρ(K,top1%|pos) = {partial(K,t1,pos):.3f}")
print(f"控制 K 后：      ρ(L,rawESS|K)  = {partial(L,ess,K):.3f}   "
      f"ρ(∫g,rawESS|K) = {partial(Ig,ess,K):.3f}")
keep=[i for i in range(len(R)) if pos[i]>0]
print(f"剔除每份结果的 w0（n={len(keep)}）：ρ(K,rawESS) = "
      f"{spear([K[i] for i in keep],[ess[i] for i in keep]):.3f}   "
      f"ρ(K,top1%) = {spear([K[i] for i in keep],[t1[i] for i in keep]):.3f}")
# 同一 (src 家族, pos) 的重复离散
fam=collections.defaultdict(list)
for r in R: fam[(r['src'].split('/')[0], r['pos'])].append(r['ess'])
mult=[(k,v) for k,v in fam.items() if len(v)>=3]
if mult:
    sp=[max(v)/min(v) for _,v in mult if min(v)>0]
    print(f"\n同体系同窗位的 rawESS 跨度（max/min），{len(sp)} 组：中位 {np.median(sp):.1f}×  "
          f"最大 {max(sp):.1f}×")

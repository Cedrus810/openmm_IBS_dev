import json, glob, os
import numpy as np
ROOT="/home/ruigengji/abfe-benchmark/openmm_IBS/runs"; SYS=("brd4","cmet","jnk1","p38")

def _sc(v, agg):
    vals = v if isinstance(v,(list,tuple)) else [v]
    out=[float(x) for x in vals if isinstance(x,(int,float)) and float(x)==float(x)]
    return agg(out) if out else None
def cum(pl,f):
    o=np.argsort(pl); pl=np.asarray(pl)[o]; f=np.asarray(f)[o]
    return pl, np.concatenate(([0.0],np.cumsum(0.5*(f[:-1]+f[1:])*np.diff(pl))))

R=[]
for d in sorted(glob.glob(f"{ROOT}/*/rep[0-9]")):
    name=os.path.basename(os.path.dirname(d))
    if not name.startswith(SYS): continue
    ck=os.path.join(d,"checkpoints"); pre=f"{ck}/preopt_dual_vanishing.json"
    st=next((p for p in (f"{ck}/stage2_vanishing.json",
             f"{ck}/stage2_vanishing_autonomous_inprogress.json") if os.path.exists(p)),None)
    if not (os.path.exists(pre) and st): continue
    P=json.load(open(pre)); S=json.load(open(st)); pd=P.get("path_diagnostics") or {}
    pl,g=pd.get("pilot_lambdas"),pd.get("metric_g"); lam=P.get("lambdas_var")
    edges=((pd.get("subdomain_allocation") or {}).get("edge_free_energy_kJ_mol"))
    wods=((S.get("diagnostics") or {}).get("window_overlap_diagnostics")) or []
    if not(pl and g and lam and wods): continue
    g=np.asarray(g,float); plg,cg=cum(pl,g); pls,cs=cum(pl,np.sqrt(np.maximum(g,1e-12)))
    I=lambda a,b,p,c: abs(float(np.interp(lam[b-1],p,c)-np.interp(lam[a],p,c)))
    for w in wods:
        rg=w.get("window_range")
        if not rg or len(rg)!=2: continue
        a,b=int(rg[0]),int(rg[1])
        if b>len(lam) or b-1>len(edges or []): continue
        ess=_sc(w.get("raw_min_absolute_ess"),min); t1=_sc(w.get("top1pct_raw_weight"),max)
        if ess is None or t1 is None: continue
        R.append(dict(run=f"{name}/{os.path.basename(d)}", K=b-a,
                      Ig=I(a,b,plg,cg), L=I(a,b,pls,cs),
                      dF=float(np.sum(np.abs(edges[a:b-1]))),
                      maxE=float(np.max(np.abs(edges[a:b-1]))),
                      ess=ess, top1=t1))

def spearman(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean()
    return float(rx@ry/np.sqrt((rx@rx)*(ry@ry)))

print(f"n = {len(R)} 个窗口 / {len({r['run'] for r in R})} 个 rep（仅蛋白体系）\n")
print("Spearman ρ（预测量 vs 实测支撑）。窗口越难 ⟹ ESS 越低、top1% 越高")
print(f"{'预测量':>6s} {'ρ(·, rawESS)':>14s} {'ρ(·, top1%)':>13s}   {'期望符号':>10s}")
for k,lab in (("Ig","∫g"),("L","L=∫√g"),("dF","ΔF"),("maxE","max边ΔF"),("K","K")):
    print(f"{lab:>6s} {spearman([r[k] for r in R],[r['ess'] for r in R]):14.3f} "
          f"{spearman([r[k] for r in R],[r['top1'] for r in R]):13.3f}   {'负 / 正':>10s}")

# ---- L 是不是只是 K 的影子？控制 K 之后 L 还剩多少 ----
def rank(v): 
    v=np.asarray(v,float); r=np.argsort(np.argsort(v)).astype(float); return r-r.mean()
def partial(x,y,z):
    """ρ(x,y | z)：先把 z 从 x、y 里回归掉，再算秩相关。"""
    X,Y,Z=rank(x),rank(y),rank(z)
    rx=X-Z*(X@Z)/(Z@Z); ry=Y-Z*(Y@Z)/(Z@Z)
    return float(rx@ry/np.sqrt((rx@rx)*(ry@ry)))

K=[r['K'] for r in R]; L=[r['L'] for r in R]; Ig=[r['Ig'] for r in R]
dF=[r['dF'] for r in R]; ess=[r['ess'] for r in R]; t1=[r['top1'] for r in R]
print(f"\nρ(K, L) = {spearman(K,L):.3f}   ρ(K, ΔF) = {spearman(K,dF):.3f}"
      f"   ρ(K, ∫g) = {spearman(K,Ig):.3f}")
print("\n控制 K 之后的偏相关：")
for k,lab in ((L,"L=∫√g"),(Ig,"∫g"),(dF,"ΔF")):
    print(f"  ρ({lab:>6s}, rawESS | K) = {partial(k,ess,K):6.3f}    "
          f"ρ({lab:>6s}, top1% | K) = {partial(k,t1,K):6.3f}")
print("\n反过来，控制各量之后 K 还剩多少：")
for k,lab in ((L,"L"),(Ig,"∫g"),(dF,"ΔF")):
    print(f"  ρ(K, rawESS | {lab:>2s}) = {partial(K,ess,k):6.3f}    "
          f"ρ(K, top1% | {lab:>2s}) = {partial(K,t1,k):6.3f}")

# ==== K 与「窗口位置」的混淆：ρ(K,结果) 到底是"K 大难"还是"w0 难"？ ====
print("\n" + "="*64)
print("混淆检查：K 与窗口序号高度共线（w0 恒为最大窗）")
# 用逐窗口序号做协变量
import collections
runs=collections.defaultdict(list)
for i,r in enumerate(R): runs[r['run']].append(i)
widx=[]
for r_,idxs in runs.items():
    for _pos,i in enumerate(idxs): widx.append((i,_pos))
W=[0]*len(R)
for i,_pos in widx: W[i]=_pos
print(f"ρ(K, 窗口序号) = {spearman(K,W):.3f}")
print(f"ρ(窗口序号, rawESS) = {spearman(W,ess):.3f}   ρ(窗口序号, top1%) = {spearman(W,t1):.3f}")
print(f"\nρ(K, rawESS | 窗口序号) = {partial(K,ess,W):6.3f}"
      f"    ρ(K, top1% | 窗口序号) = {partial(K,t1,W):6.3f}")
print(f"ρ(窗口序号, rawESS | K) = {partial(W,ess,K):6.3f}"
      f"    ρ(窗口序号, top1% | K) = {partial(W,t1,K):6.3f}")

# w0 单独拿掉，看 K 在其余窗口里还有没有信号
keep=[i for i in range(len(R)) if W[i]>0]
if len(keep)>6:
    print(f"\n剔除每个 run 的 w0 之后（n={len(keep)}）：")
    print(f"  ρ(K, rawESS) = {spearman([K[i] for i in keep],[ess[i] for i in keep]):6.3f}"
          f"   ρ(K, top1%) = {spearman([K[i] for i in keep],[t1[i] for i in keep]):6.3f}")

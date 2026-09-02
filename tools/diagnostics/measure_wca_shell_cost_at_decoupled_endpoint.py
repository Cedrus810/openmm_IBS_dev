#!/usr/bin/env python
"""量 λ-WCA 防护壳在**完全解耦端**被漏掉的自由能（GPU，约 20 分钟）。

要回答的问题
------------
2026-09-02 的独立参考真值实测（docs/reference_data/）显示 4W53 甲苯溶剂腿
stage2 的 +45.3 kJ/mol 误差有 **76% 在解耦端 win3/4/5**，而 `bias_to_signal_ratio`
（= sd(力组{1,4}) / min_k sd(U_k_int)）与逐窗口误差的 Spearman = **+0.886**
（同批数据 ESS 与误差是 **−0.600 反相关**）。

机制假设：壳幅度是 `4λ_s(1−λ_s)`（λ→0 只线性衰减），物理配体↔环境 LJ 是
`λ^n_lj`（生产 n_lj=2，二次衰减），比值 `4(1−λ)/λ` 在 λ→0 **发散**。
生产 window 5 用 `lambda_shield = mean(λ_vdw) = 0.138` ⟹ 壳还开着
`4·0.138·0.862 = 0.476` 强度，而配体自己的 LJ 只剩 `λ²= 1.9%`。
目标态能量**不含**壳，所以生产是在「带壳系综」上重加权到「无壳目标」——
水贴近配体的那些构型系综里根本没有。

本脚本直接量那个缺失量：**ΔG(壳 0.476 → 0) 在 λ_vdw=0 处**。

判读
----
* **≈ −11 kJ/mol** ⟹ 正好对上 window 5 的误差 +11.02，机制闭环。
* 对不上 ⟹ 壳不是（唯一）原因，还有别的东西，回去查。

符号：应为**负**（去掉纯排斥壳，水能贴近，有利）。生产把这份有利自由能漏掉，
所以生产的 ΔF 偏**正** —— 与实测 6 个窗口全部偏正一致。

不改任何生产代码；只读 4W53 的建系产物。壳的表达式与参数
（rc=0.244 nm, eps_wca=1.0 kJ/mol，取自那次运行的 pipeline.log）与
`ibs_engine.py` 的 Group-4 逐字同形。

用法
----
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    python tools/diagnostics/measure_wca_shell_cost_at_decoupled_endpoint.py \
        [equil_steps] [prod_steps] [out.json]

默认 40000 / 200000 / shell_cost.json。`XLA_*` 两个环境变量是**必须**的：
pymbar4 的 JAX 后端会预占整卡 75% 显存，而 MBAR 是在 OpenMM Context 还活着时调的。

需要 `ROOT` 下的 output/{system_solvent.xml, topology_solvent.cif,
box_vectors_solvent.npy, ligand_indices_solvent.json}。
"""
import json, sys, time
import numpy as np
import openmm as mm, openmm.app as app, openmm.unit as u

ROOT="/home/ruigengji/ABFE_IBS/4W53"
RC_NM=0.244; EPS_WCA=1.0
SHELL_TOP=4*0.138*(1-0.138)          # 生产 window 5 的壳幅度
T=300.0; DT_FS=1.0
EQUIL=int(sys.argv[1]) if len(sys.argv)>1 else 40000
PROD=int(sys.argv[2]) if len(sys.argv)>2 else 200000
INTERVAL=500

sysxml=f"{ROOT}/output/system_solvent.xml"
system=mm.XmlSerializer.deserialize(open(sysxml).read())
box=np.load(f"{ROOT}/output/box_vectors_solvent.npy")
system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*r)*u.nanometer for r in box])
pdbx=app.PDBxFile(f"{ROOT}/output/topology_solvent.cif")
lig=json.load(open(f"{ROOT}/output/ligand_indices_solvent.json"))["ligand_indices"]
n=system.getNumParticles()

# 找 NonbondedForce，取 exception 与 cutoff
nb=[f for f in system.getForces() if isinstance(f,mm.NonbondedForce)][0]
base=[nb.getParticleParameters(i) for i in range(n)]

# 配体↔环境的静电与 LJ 全部关掉（λ_vdw=0、λ_elec=0 的完全解耦端点）：
# 直接把配体电荷与 epsilon 置零，并用 exception 断开配体内部? —— 不，配体内部保留。
# 做法与参考脚本一致：把配体↔环境从 NonbondedForce 里摘出来，本实验不再加回去
# （λ_vdw=0 ⟹ 生产的 softcore 项恒为 0），所以只剩「壳」这一项是可变的。
for i in lig:
    q,sig,eps=nb.getParticleParameters(i)
    nb.setParticleParameters(i,0.0*u.elementary_charge,sig,0.0*u.kilojoule_per_mole)
# 配体内部非键用显式 CustomBondForce 补回满强度（与参考脚本同口径）
ONE_4PI=138.935456
intra=mm.CustomBondForce(f"{ONE_4PI}*chargeProd/r + 4*epsilon*((sigma/r)^12-(sigma/r)^6)")
for pn in ("chargeProd","sigma","epsilon"): intra.addPerBondParameter(pn)
exc=set()
for e in range(nb.getNumExceptions()):
    i,j,_,_,_=nb.getExceptionParameters(e); exc.add((min(i,j),max(i,j)))
ls=sorted(lig)
for a in range(len(ls)):
    for b in range(a+1,len(ls)):
        i,j=ls[a],ls[b]
        if (i,j) in exc: continue
        qi,si,ei=base[i]; qj,sj,ej=base[j]
        intra.addBond(i,j,[qi.value_in_unit(u.elementary_charge)*qj.value_in_unit(u.elementary_charge),
                           0.5*(si.value_in_unit(u.nanometer)+sj.value_in_unit(u.nanometer)),
                           np.sqrt(ei.value_in_unit(u.kilojoule_per_mole)*ej.value_in_unit(u.kilojoule_per_mole))])
intra.setForceGroup(6); system.addForce(intra)

# λ-WCA 防护壳：与生产逐字同形，幅度用可变 global `shell_amp`
wca_expr=("shell_amp*step(rc-r)*eps_wca*"
          "(((rc/max(r, 1e-6))^6)^2 - 2*((rc/max(r, 1e-6))^6) + 1)")
wca=mm.CustomNonbondedForce(wca_expr)
wca.addGlobalParameter("shell_amp", SHELL_TOP)
wca.addGlobalParameter("rc", RC_NM)
wca.addGlobalParameter("eps_wca", EPS_WCA)
for _ in range(n): wca.addParticle([])
wca.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
wca.setCutoffDistance(RC_NM*u.nanometer)
env=[i for i in range(n) if i not in set(lig)]
wca.addInteractionGroup(ls, env)
for i,j in exc: wca.addExclusion(i,j)
wca.setForceGroup(7); system.addForce(wca)

integ=mm.LangevinMiddleIntegrator(T*u.kelvin, 1.0/u.picosecond, DT_FS*u.femtosecond)
integ.setRandomNumberSeed(20260902)
ctx=mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"), {"Precision":"mixed"})
ctx.setPositions(pdbx.positions)
ctx.setPeriodicBoxVectors(*[mm.Vec3(*r)*u.nanometer for r in box])
mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=2000)
ctx.setVelocitiesToTemperature(T*u.kelvin, 20260902)

ladder=[SHELL_TOP, 0.75*SHELL_TOP, 0.5*SHELL_TOP, 0.25*SHELL_TOP, 0.10*SHELL_TOP, 0.0]
K=len(ladder); ns=PROD//INTERVAL
KB=8.314462618e-3; kT=KB*T
print(f"壳幅度梯子（生产 window5 = {SHELL_TOP:.4f}）: {[round(x,4) for x in ladder]}")
print(f"每态 equil={EQUIL} prod={PROD} 样本={ns}")
u_kln=np.zeros((K,K,ns)); N_k=np.zeros(K,dtype=int)
t0=time.time()
print("全体系预平衡 (壳=生产强度) ...", flush=True)
ctx.setParameter("shell_amp", SHELL_TOP); integ.step(100000)
for k,amp in enumerate(ladder):
    ctx.setParameter("shell_amp", amp); integ.step(EQUIL)
    for s in range(ns):
        integ.step(INTERVAL)
        for l,amp2 in enumerate(ladder):
            ctx.setParameter("shell_amp", amp2)
            u_kln[k,l,s]=ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)/kT
        ctx.setParameter("shell_amp", amp)
    N_k[k]=ns
    print(f"  [态 {k}] shell_amp={amp:.4f} 完成 | 累计 {time.time()-t0:.0f}s", flush=True)

from pymbar import MBAR
u_kn=np.zeros((K,int(N_k.sum()))); idx=0
for k in range(K):
    u_kn[:,idx:idx+N_k[k]]=u_kln[k,:,:N_k[k]]; idx+=N_k[k]
res=MBAR(u_kn,N_k).compute_free_energy_differences()
df,ddf=res["Delta_f"],res["dDelta_f"]
dG=float(df[0,K-1])*kT; err=float(ddf[0,K-1])*kT
print("\n"+"="*66)
print(f"ΔG(壳 {SHELL_TOP:.4f} -> 0) 在 λ_vdw=0 处 = {dG:+8.3f} ± {err:.3f} kJ/mol")
print(f"生产 window 5 的误差                      = +11.023 kJ/mol")
print(f"生产 window 3+4+5 合计误差                = +34.397 kJ/mol")
print("="*66)
print("判读：ΔG 为负、量级 ~-10 ⟹ 壳把水挡在配体近程外，这份有利自由能被生产漏掉")
json.dump({"shell_amp_top":SHELL_TOP,"ladder":ladder,"dG_kJ_mol":dG,"err_kJ_mol":err,
           "per_state_Delta_f_kT":[float(x) for x in df[0,:]],"rc_nm":RC_NM,"eps_wca":EPS_WCA,
           "equil":EQUIL,"prod":PROD}, open(sys.argv[3] if len(sys.argv)>3 else "shell_cost.json","w"), indent=2)

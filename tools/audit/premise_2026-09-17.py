import json, glob, os
import numpy as np
ROOT="/home/ruigengji/abfe-benchmark/openmm_IBS/runs"; SYS=("brd4","cmet","jnk1","p38")
print("λ 布点是否等热力学长度 δ？（δ = 每条边的 ∫√g dλ）\n")
print(f"{'run':22s} {'n边':>3s} {'δ_min':>6s} {'δ_max':>6s} {'δ中位':>6s} {'max/中位':>8s}")
sp=[]
for d in sorted(glob.glob(f"{ROOT}/*/rep[0-9]")):
    name=os.path.basename(os.path.dirname(d))
    if not name.startswith(SYS): continue
    p=os.path.join(d,"checkpoints","preopt_dual_vanishing.json")
    if not os.path.exists(p): continue
    pd=(json.load(open(p)).get("path_diagnostics") or {})
    e=pd.get("optimized_edge_thermodynamic_lengths")
    if not e: continue
    e=np.abs(np.asarray(e,float)); med=float(np.median(e))
    sp.append(float(e.max()/med))
    print(f"{name+'/'+os.path.basename(d):22s} {e.size:3d} {e.min():6.3f} {e.max():6.3f} "
          f"{med:6.3f} {e.max()/med:8.2f}")
print(f"\n全体 max/中位 的中位数 = {np.median(sp):.2f}（等 δ 的理想值 = 1.00）")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参照真值臂的盒体积是不是偏离 1 bar 平衡？—— 零 GPU 实验 + GPU 复核输入准备。

## 背景（一句话）

`docs/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md` 把溶剂腿 −4.32 kJ/mol 残差里的
**−0.86（20%）** 归给"生产与真值不在同一密度"。那条归因立在一个**未直接检验过的
前提**上：43.950 nm³ 的**建系盒**才是偏离平衡的那个，42.63~42.75 的生产盒是对的。

之所以只是"前提"，是因为当时只看了生产自己的 NPT 预平衡曲线 —— 那是**生产臂的**
证据，拿它去判**真值臂**的盒对不对，是循环的。真值脚本
`4W53/toluene_hydration_reference.py` 从建系坐标 + 建系盒起步，显式拒绝恒压器
（第 208 行），全程 NVT ⟹ 它**永远不会**发现自己跑在错的密度上。

## 这个脚本做什么

1. **报三个体积**：建系盒 / 生产冻结盒（从 LRC 偏移反解）/ 生产 NPT 预平衡的平衡值。
2. **实验**：拿**真值脚本自己的那份输入**（同一个 `system_solvent.xml` +
   同一份建系坐标 + 同一个建系盒），加一个 1 bar 恒压器跑 CPU NPT，看 V 往哪走。
   这一步不依赖生产臂的任何数据 —— 前提被独立检验，循环被打断。
3. **准备 GPU 复核的输入**：把建系坐标按**分子质心**仿射缩到平衡盒，写出一个
   可以直接 `--root` 进去的目录，并打印那条 GPU 命令。

## 用法

    python tools/diagnostics/probe_reference_box_density.py \
        --case-root /home/ruigengji/ABFE_IBS/4W53 \
        --run-dir  /home/ruigengji/ABFE_IBS/4W53/output_v3_seed20260908

零 GPU（脚本自己强制 `CUDA_VISIBLE_DEVICES=""`）。默认 50 ps CPU NPT，约 10 分钟。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 这个脚本按定义不碰 GPU：生产作业可能正占着卡。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _volume(box) -> float:
    return float(abs(np.linalg.det(np.asarray(box, dtype=float))))


# ---------------------------------------------------------------------------
# 1. 三个体积
# ---------------------------------------------------------------------------


def report_volumes(case_root: Path, run_dir: Path) -> dict:
    build_box = np.load(case_root / "output" / "box_vectors_solvent.npy")
    v_build = _volume(build_box)

    v_prod = _solve_production_volume(run_dir)

    monitor = run_dir / "solvent_leg" / "pre_equilibration_convergence_monitor.csv"
    v_npt = None
    if monitor.is_file():
        rows = [line.split(",") for line in monitor.read_text().strip().splitlines()[1:]]
        header = monitor.read_text().splitlines()[0].split(",")
        col = header.index("volume_nm3")
        series = np.asarray([float(r[col]) for r in rows if r[col]], dtype=float)
        v_npt = (float(series.mean()), float(series.std(ddof=1)), len(series))

    print("=" * 78)
    print("三个盒体积")
    print("=" * 78)
    print(f"  建系盒（真值脚本硬编码用它，NVT，拒绝恒压器） V = {v_build:.5f} nm^3"
          f"   边长 {np.diag(build_box)[0]:.5f} nm")
    if v_prod is not None:
        print(f"  生产 stage2 冻结盒（从 LRC 偏移反解）        V = {v_prod:.5f} nm^3"
              f"   （比建系盒小 {100 * (v_build - v_prod) / v_prod:.2f}%）")
    if v_npt is not None:
        mean, sd, n = v_npt
        print(f"  生产 NPT 预平衡实测（1 bar, {n} 个快照）      V = {mean:.5f} ± {sd:.5f} nm^3")
    return {"v_build": v_build, "v_prod": v_prod, "v_npt": v_npt}


def _solve_production_volume(run_dir: Path):
    """从 `energies − sampling_states` 的逐态常数偏移反解生产实际用的 V。

    与 `attribute_stage2_solvent_leg_gap.py` 同一套判据：偏移 == lrc_coeff[k]/V，
    所有 λ 态必须给出同一个 V，否则这个读法本身不成立。
    """
    try:
        import openmm as mm

        import ibs_engine as ie

        solv = run_dir / "solvent_leg"
        vdir = solv / "vanishing"
        results = json.loads((solv / "final_results.json").read_text())
        n_win = len(results["stage_diagnostics"]["stage2"]["covariance_chain_segments"])

        lam, off = [], []
        for w in range(n_win):
            E = np.load(vdir / f"dual_window_{w}_vdw_energies.npy")
            S = np.load(vdir / f"dual_window_{w}_vdw_sampling_states.npy").T
            conv = json.loads((vdir / f"dual_window_{w}_vdw_convergence.json").read_text())
            lv = np.asarray(conv["lambdas_vdw"], dtype=float)
            take = slice(0, len(lv)) if w == 0 else slice(1, len(lv))
            lam.extend(lv[take])
            off.extend((E - S).mean(axis=1)[take])
        lam, off = np.asarray(lam), np.asarray(off)

        system = mm.XmlSerializer.deserialize((run_dir / "system_solvent.xml").read_text())
        lig = json.loads((run_dir / "ligand_indices_solvent.json").read_text())
        lig = set(lig["ligand_indices"] if isinstance(lig, dict) else lig)
        nb = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
        params = [nb.getParticleParameters(i) for i in range(nb.getNumParticles())]
        env = [i for i in range(nb.getNumParticles()) if i not in lig]
        sig, s6, s12 = ie._lj_tail_correction_sigma_resolved_moments(params, sorted(lig), env)
        sc = results["protocol_key"]["payload"]["stage2_protocol_key"]["payload"][
            "aces_softcore_params"]
        coeff = ie._lj_tail_lrc_coefficients_kj_mol(
            lam, sig, s6, s12, float(sc["alpha_lj"]),
            *(float(x) for x in sc["power_lj"]),
            ie.LJ_TAIL_LRC_R_SWITCH_NM, ie.LJ_TAIL_LRC_R_CUTOFF_NM)
        nz = off != 0.0
        return float((coeff[nz] / off[nz]).mean())
    except Exception as exc:  # 诊断辅助，缺产物不该拖垮主实验
        print(f"（生产盒反解跳过：{type(exc).__name__}: {exc}）")
        return None


# ---------------------------------------------------------------------------
# 2. 实验：真值臂自己的输入 + 1 bar 恒压器
# ---------------------------------------------------------------------------


def run_cpu_npt(case_root: Path, *, steps: int, report_every: int, seed: int,
                threads: int, temperature: float, timestep_fs: float):
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    os.environ["OPENMM_CPU_THREADS"] = str(threads)

    system = mm.XmlSerializer.deserialize(
        (case_root / "output" / "system_solvent.xml").read_text())
    box = np.load(case_root / "output" / "box_vectors_solvent.npy")
    positions = app.PDBxFile(str(case_root / "output" / "topology_solvent.cif")).positions

    # 真值脚本在这里 sys.exit；本实验反过来 —— 就是要装上它来问"平衡体积是多少"。
    assert not any(isinstance(f, mm.MonteCarloBarostat) for f in system.getForces()), \
        "system_solvent.xml 里已经有恒压器了，本实验的前提（真值臂是 NVT）不成立"

    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*r) * u.nanometer for r in box])
    barostat = mm.MonteCarloBarostat(1.0 * u.bar, temperature * u.kelvin, 25)
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(
        temperature * u.kelvin, 1.0 / u.picosecond, timestep_fs * u.femtosecond)
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CPU"))
    context.setPositions(positions)
    context.setPeriodicBoxVectors(*[mm.Vec3(*r) * u.nanometer for r in box])
    mm.LocalEnergyMinimizer.minimize(context, 10.0, 1000)
    context.setVelocitiesToTemperature(temperature * u.kelvin, seed)

    print()
    print("=" * 78)
    print(f"实验：真值臂输入 + 1 bar 恒压器，CPU NPT {steps * timestep_fs / 1000:.0f} ps"
          f"（{threads} 线程，seed={seed}）")
    print("=" * 78)
    print(f"{'ps':>8} {'V (nm^3)':>11} {'边长 (nm)':>10} {'相对建系盒':>11}")

    v0 = _volume(box)
    trace = []
    t_start = time.time()
    done = 0
    while done < steps:
        chunk = min(report_every, steps - done)
        integrator.step(chunk)
        done += chunk
        state = context.getState()
        v = state.getPeriodicBoxVolume().value_in_unit(u.nanometer ** 3)
        trace.append((done * timestep_fs / 1000.0, float(v)))
        print(f"{done * timestep_fs / 1000.0:>8.1f} {v:>11.4f} {v ** (1 / 3):>10.5f} "
              f"{100 * (v - v0) / v0:>+10.2f}%")

    elapsed = time.time() - t_start
    tail = np.asarray([v for t, v in trace if t >= trace[-1][0] * 0.5], dtype=float)
    print(f"\n耗时 {elapsed / 60:.1f} min。后半程平均 V = {tail.mean():.4f} ± "
          f"{tail.std(ddof=1) if len(tail) > 1 else float('nan'):.4f} nm^3")
    return float(tail.mean()), trace


# ---------------------------------------------------------------------------
# 3. 准备 GPU 复核的 root
# ---------------------------------------------------------------------------


def prepare_gpu_root(case_root: Path, out_root: Path, target_volume: float):
    """按**分子质心**仿射缩放建系坐标到目标盒，写出一个可 `--root` 的目录。

    为什么按质心而不是按原子：整体缩原子坐标会把水的 O–H 压掉 0.9%，虽然约束
    第一步就会拉回来，但那一步是在**已经错了的键长**上算力。按质心平移则内部
    几何一个字不动 —— 这也正是 `∂ΔA/∂V` 那次有限差分用的同一套缩放。
    """
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u

    import abfe_core as core

    src = case_root / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    dst = out_root / "output"
    dst.mkdir(exist_ok=True)

    system = mm.XmlSerializer.deserialize((src / "system_solvent.xml").read_text())
    pdbx = app.PDBxFile(str(src / "topology_solvent.cif"))
    box = np.load(src / "box_vectors_solvent.npy")
    xyz = np.asarray(pdbx.positions.value_in_unit(u.nanometer), dtype=float)

    scale = (target_volume / _volume(box)) ** (1.0 / 3.0)
    molecules, _bonds = core.system_molecule_grouping(system)
    masses = np.asarray(
        [system.getParticleMass(i).value_in_unit(u.dalton)
         for i in range(system.getNumParticles())], dtype=float)

    scaled = xyz.copy()
    for mol in molecules:
        idx = np.asarray(sorted(mol), dtype=int)
        w = masses[idx]
        com = (xyz[idx] * w[:, None]).sum(axis=0) / w.sum()
        scaled[idx] += (scale - 1.0) * com

    new_box = box * scale
    np.save(dst / "box_vectors_solvent.npy", new_box)
    # 让写出来的 CIF 自洽：真值脚本会用 npy 覆盖盒，但一份自相矛盾的 CIF
    # 迟早会被别的工具读到。
    # 必须是「一个 Quantity 包着一串 Vec3」，不是「一串 Quantity」——后者
    # 让 PDBxFile.writeHeader 的 `a*10` 拿到 Quantity 而不是 float 而炸。
    pdbx.topology.setPeriodicBoxVectors(
        u.Quantity([mm.Vec3(*row) for row in new_box], u.nanometer))
    with open(dst / "topology_solvent.cif", "w") as handle:
        app.PDBxFile.writeFile(
            pdbx.topology, scaled * u.nanometer, handle, keepIds=True)
    for name in ("system_solvent.xml", "ligand_indices_solvent.json"):
        link = dst / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((src / name).resolve())

    print()
    print("=" * 78)
    print("已准备 GPU 复核输入")
    print("=" * 78)
    print(f"  目录       {out_root}")
    print(f"  缩放因子   {scale:.6f}（按分子质心，内部几何不变）")
    print(f"  盒         {_volume(box):.5f} → {_volume(new_box):.5f} nm^3"
          f"   边长 {np.diag(box)[0]:.5f} → {np.diag(new_box)[0]:.5f} nm")
    print(f"  分子数     {len(molecules)}")
    return scale


def emit_lambda_subset(source: Path, out_path: Path, n_states: int) -> list:
    """从生产 λ 表等步长抽稀，写一份 `--lambda-vdw-json` 能直接吃的文件。

    **为什么等步长抽是对的**：生产那把梯子是
    `redistribute_vanishing_lambda_subdomains` 按**等热力学长度**布的
    —— λ 间隔大的地方正是度规小的地方。所以按索引等步长抽，等价于把每段的
    热力学长度整齐地放大同一个倍数，不会在某一段偷偷变稀。
    （反过来，按 λ 均匀抽会把端点奇异区抽秃。）

    端点 1.0 / 0.0 强制保留 —— 下游拿 `df[0, K-1]` 当 ΔG，端点动了数就不是那个数。
    """
    payload = json.loads(source.read_text())
    full = [float(x) for x in payload["lambdas_var"]]
    if not (abs(full[0] - 1.0) < 1e-9 and abs(full[-1]) < 1e-9):
        raise ValueError(f"{source} 的 λ 端点不是 1/0：{full[0]}, {full[-1]}")
    if not 2 <= n_states <= len(full):
        raise ValueError(f"n_states 必须在 2..{len(full)} 之间，给的是 {n_states}")

    idx = {int(round(i)) for i in np.linspace(0, len(full) - 1, n_states)}
    # 端点段（最后一个内部态 → λ=0）**不参与抽稀**。等热力学长度的论证在这里最弱：
    # 度规是从 dU/dλ 的方差估的，而 λ→0 的软核奇异区恰恰是它低估的地方
    # （见 4W53「端点段卡在 n_k 差 2 个样本」）。多留一个态是便宜的保险。
    idx.add(len(full) - 2)
    idx = sorted(idx)
    subset = [full[i] for i in idx]

    out_path.write_text(json.dumps({
        "lambdas_var": subset,
        "_provenance": {
            "source": str(source),
            "source_n_states": len(full),
            "selected_indices": idx,
            "note": "等索引步长抽稀 = 等热力学长度抽稀（生产梯子本身就是等热力学长度布的）",
        },
    }, indent=2), encoding="utf-8")

    gaps = np.diff(subset)
    print()
    print("=" * 78)
    print(f"稀疏 λ 表：{len(full)} → {len(subset)} 态"
          f"{'（要了 %d，端点段保护 +1）' % n_states if len(subset) != n_states else ''}")
    print("=" * 78)
    print(f"  写到     {out_path}")
    print(f"  取的索引 {idx}")
    print(f"  λ        {', '.join('%.4f' % x for x in subset)}")
    print(f"  最大间隔 {abs(gaps).max():.4f}（原表 "
          f"{abs(np.diff(full)).max():.4f}）—— 端点段最容易掉重叠，先看这个")
    print(f"  成本     动力学 ~{len(subset) / len(full):.0%}，"
          f"K×K 取能量 ~{(len(subset) / len(full)) ** 2:.0%}")
    return subset


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 🔑 不给默认值。个人路径当 argparse default 比**没有** default 更糟：忘了传参时
    # 它不报错，而是静默去读别人机器上的目录。同一条已经在 `gmx_path` 上付过一次代价
    # （2026-08-24 删掉两条本机路径）。`--gpu-root` 那种从 `--case-root` 派生的默认值
    # 是安全的，因为它不含任何机器相关的字面量。
    ap.add_argument("--case-root", type=Path, required=True,
                    help="真值脚本的 --root（含 output/ 的那个）")
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="要对照的生产运行目录")
    ap.add_argument("--steps", type=int, default=50000, help="CPU NPT 步数（1 fs）")
    ap.add_argument("--report-every", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--skip-npt", action="store_true", help="只报体积、只准备 GPU 输入")
    ap.add_argument("--gpu-root", type=Path, default=None,
                    help="准备 GPU 复核输入的目录（默认 <case-root>/reference_at_npt_box）")
    ap.add_argument("--lambda-subset", type=int, default=None, metavar="N",
                    help="从生产 λ 表等步长抽稀到 N 态，写一份稀疏 lambda json。"
                         "23 态跑满约 31 min；12 态约 16 min。端点 1/0 必定保留。")
    ap.add_argument("--dp-dv", type=float, default=4.07,
                    help="∂ΔA_LJ/∂V (kJ/mol/nm^3)，用来预报 GPU 那一跑会移动多少。"
                         "默认 4.07（68 bar）来自 2026-09-10 的配对重跑实测；"
                         "别用 09-09 那个 100 帧有限差分的 0.71，它错 5.7 倍。")
    ap.add_argument("--dp-dv-err", type=float, default=0.33)
    args = ap.parse_args()

    volumes = report_volumes(args.case_root, args.run_dir)

    v_target = volumes["v_prod"]
    if not args.skip_npt:
        v_relaxed, _trace = run_cpu_npt(
            args.case_root, steps=args.steps, report_every=args.report_every,
            seed=args.seed, threads=args.threads, temperature=args.temperature,
            timestep_fs=args.timestep_fs)
        v_build = volumes["v_build"]
        drop = 100 * (v_build - v_relaxed) / v_relaxed
        print()
        if drop > 1.0:
            print(f"⟹ **建系盒确实偏离 1 bar 平衡**：真值臂自己的输入在恒压器下把 V "
                  f"从 {v_build:.4f} 收缩到 {v_relaxed:.4f} nm^3（低 {drop:.2f}%）。")
            if volumes["v_npt"] is not None:
                print(f"   与生产 NPT 预平衡的 {volumes['v_npt'][0]:.4f} nm^3 一致 ⟹ "
                      f"错的是**真值臂**的密度，不是生产。")
        else:
            print(f"⟹ 建系盒未见明显收缩（{drop:+.2f}%）——"
                  f"密度归因的前提**不成立**，−0.86 那一项要撤。")
            return 1

    if v_target is None:
        print("\n（反解不出生产盒，跳过 GPU 输入准备）")
        return 0

    gpu_root = args.gpu_root or (args.case_root / "reference_at_npt_box")
    prepare_gpu_root(args.case_root, gpu_root, v_target)

    source_lam = (args.run_dir / "solvent_leg" / "checkpoints"
                  / "preopt_dual_vanishing.json")
    lam_json = source_lam
    n_lambda = len(json.loads(source_lam.read_text())["lambdas_var"])
    if args.lambda_subset:
        lam_json = gpu_root / f"lambdas_{args.lambda_subset}states.json"
        n_lambda = len(emit_lambda_subset(source_lam, lam_json, args.lambda_subset))

    dv = volumes["v_build"] - v_target
    print()
    print("=" * 78)
    print("GPU 复核：预报")
    print("=" * 78)
    print(f"  ΔV = {volumes['v_build']:.4f} − {v_target:.4f} = {dv:+.4f} nm^3")
    print(f"  按 ∂ΔA_LJ/∂V = {args.dp_dv:+.2f} ± {args.dp_dv_err:.2f} kJ/mol/nm^3，"
          f"真值挪到生产盒之后应当移动 {-dv * args.dp_dv:+.2f} ± {dv * args.dp_dv_err:.2f} kJ/mol")
    print(f"  即 ΔG_truth  −6.59 ± 0.32（建系盒）  →  预报 "
          f"{-6.594 - dv * args.dp_dv:+.2f} ± {(0.324 ** 2 + (dv * args.dp_dv_err) ** 2) ** 0.5:.2f}")
    print(f"  残差 −4.318 → 预报 {-4.318 + dv * args.dp_dv:+.2f}")
    print("  （4W53 已实测：−11.490 ± 0.342，残差 +0.59 = 0.72σ ⟹ 密度是 100% 不是 20%。")
    print("    别忘了配对对照臂——同 λ 表同 seed 只换回建系盒，否则分不清是盒还是布点。）")
    print()
    # 前台直连终端：**不接管道**。上一次挂就是 `| tee` —— tee 那头一堵，
    # 写端跟着阻塞，看起来像"跑到一半死了"，其实是 print 卡在 write。
    # 直连 tty 没有这个中间缓冲，卡没卡一眼就看得出来。
    # 结果本来就会落到 --out 的 JSON，log 文件不是必需品。
    print("命令（前台，不接管道 —— 卡没卡直接看得见）：")
    print(f"""
  cd /home/ruigengji/ABFE_IBS/4W53
  PYMBAR_DISABLE_JAX=1 python toluene_hydration_reference.py \\
      --root {gpu_root} \\
      --out  {gpu_root}/hydration_reference \\
      --vdw-only \\
      --lambda-vdw-json {lam_json} \\
      --platform CUDA --precision mixed \\
      --seed 20260828
""")
    print(f"  正常节奏：预平衡 ~30s，之后每个态 ~80s，{n_lambda} 态约 "
          f"{(30 + 80 * n_lambda) / 60:.0f} min。超过 3 分钟没新行就是真卡了。")
    same_ladder = not args.lambda_subset
    print(f"  跑完对比（λ 表{'与本次实跑相同，可直接逐窗口比' if same_ladder else '是实跑的子集：端点相同 ⟹ **总量**可比，逐窗口仍需插值'}）：")
    print(f"    python tools/diagnostics/attribute_stage2_solvent_leg_gap.py {args.run_dir}")
    print("    ——把 docs/reference_data/ 的真值临时换成新跑出来的那份，"
          "残差应从 −4.32 落到上面的预报值附近。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

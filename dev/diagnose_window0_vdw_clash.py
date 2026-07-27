"""结构诊断：vanishing leg window 0 (lambda_vdw=1.0 端点) 崩溃/不收敛的逐 pair 碰撞分解。

背景见 /home/kasuga/.claude/plans/ibs-in-compressed-sky.md。只读 + 新增，不修改任何
现有文件、不写入 checkpoints/，不会被 pipeline 的 resume/缓存逻辑捡到。

用法：
    python diagnose_window0_vdw_clash.py --output-dir ./output_lrc_fix --n-steps 20000 \
        --lambda-shield 1.0 --clash-threshold-kj-mol 20 --platform CUDA
    python diagnose_window0_vdw_clash.py --output-dir ./output_lrc_fix --n-steps 20000 \
        --lambda-shield 0.9843 --clash-threshold-kj-mol 20 --platform CUDA
    python diagnose_window0_vdw_clash.py --output-dir ./output_lrc_fix --skip-md \
        --lambda-shield 1.0
"""

import argparse
import collections
import csv
import json
import os

import numpy as np
import openmm
from openmm import app, unit, XmlSerializer

import runabfe
import abfe_pipeline
import ibs_engine
import dexp_experiment


DIAG_SUBDIR = "vanishing_diagnostics"


def _make_prefix(lambda_vdw, lambda_shield):
    return f"diag_window0_vdw_lam{lambda_vdw:.4f}_shield{lambda_shield:.4f}"


def _load_boresch_params(output_dir):
    for name in ("boresch_params.json", "boresch_auto.json", "boresch_simple.json", "boresch_fluctuation.json"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            print(f"  Boresch params loaded from {path}")
            return runabfe._sanitize_boresch_params(raw)
    raise FileNotFoundError(f"未在 {output_dir} 下找到任何 boresch_*.json")


def _apply_committed_boresch_equilibrium(output_dir, boresch_params):
    # 生产环境每条腿只在第一次采样时推导一次平衡几何量, 之后所有 resume 都强制复用这份
    # 落盘值 (abfe_pipeline.py:5874-5894), 不然拼接处会有 ~200 kJ/mol 的伪能量跳变。
    # 诊断跑必须复用同一份值, 否则跟真实 window 0 用的 Boresch 基准就不是同一个 Hamiltonian。
    committed_path = os.path.join(output_dir, "checkpoints", "boresch_equilibrium_committed.json")
    if not os.path.exists(committed_path):
        print(f"  警告: 未找到 {committed_path}, 将使用重新清洗的 boresch_params 里的平衡值, "
              "可能与真实 window 0 用的基准不完全一致。")
        return boresch_params
    with open(committed_path) as f:
        committed_eq = json.load(f)["equilibrium_values"]
    boresch_params = dict(boresch_params)
    boresch_params["equilibrium_values"] = committed_eq
    print(f"  Boresch 平衡值已用 {committed_path} 覆盖")
    return boresch_params


def _assert_lambda_vdw_in_schedule(output_dir, lambda_vdw):
    schedule_path = os.path.join(output_dir, "checkpoints", "preopt_dual_vanishing.json")
    with open(schedule_path) as f:
        schedule = json.load(f)
    lambdas_vdw = schedule["lambdas_var"]
    if not any(abs(float(v) - lambda_vdw) < 1e-6 for v in lambdas_vdw):
        raise RuntimeError(
            f"--lambda-vdw={lambda_vdw!r} 不在当前 schedule 的 lambdas_var 里 "
            f"({lambdas_vdw!r}) —— vanishing leg 的 schedule 可能已经变化, 拒绝在不确认"
            "该态确实存在于当前 window 排布中的情况下继续诊断。"
        )


def _load_pre_equil_state(output_dir, system, topology, platform_name, props):
    chk_path = os.path.join(output_dir, "checkpoints", "pre_equil.chk")
    if not os.path.exists(chk_path):
        raise FileNotFoundError(f"未找到 {chk_path}")
    # 只用它读坐标/速度/盒子, 用完即弃 —— 不能复用这个 Simulation 本身
    # (它绑定的是普通非 alchemical system, 不是我们要跑的诊断系统)。
    # OpenMM 的 checkpoint 是平台绑定的二进制格式, 必须用保存时同一种 Platform 加载
    # (pre_equil.chk 是在 CUDA 上存的, 用 Reference 加载会报
    # "Checkpoint was created with a different Platform"), 所以这里跟正式模拟用同一个
    # platform/props, 不能图省事换成 Reference。
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    throwaway_sim = app.Simulation(topology, system, integrator, platform, props)
    throwaway_sim.loadCheckpoint(chk_path)
    state = throwaway_sim.context.getState(getPositions=True, getVelocities=True)
    print(f"  从 {chk_path} 读取坐标/速度/盒子 (platform={platform_name})")
    return state.getPositions(), state.getVelocities(), state.getPeriodicBoxVectors()


def build_diagnostic_system(output_dir, platform_name, props, lambda_vdw):
    system, topology, _unused_positions, box_vectors, ligand_indices = runabfe.load_native_system(
        output_dir, phase="complex", prefer_equilibrated=False,
    )
    positions, velocities, chk_box_vectors = _load_pre_equil_state(
        output_dir, system, topology, platform_name, props
    )
    if chk_box_vectors is not None:
        box_vectors = chk_box_vectors

    boresch_params = _load_boresch_params(output_dir)
    boresch_params = _apply_committed_boresch_equilibrium(output_dir, boresch_params)
    if not abfe_pipeline._has_valid_boresch_restraint(boresch_params):
        raise RuntimeError("Boresch 参数缺少完整 3+3 锚点, 无法构建 window 0 的真实 Hamiltonian。")

    _assert_lambda_vdw_in_schedule(output_dir, lambda_vdw)

    alchemical_params = abfe_pipeline._resolve_alchemical_params("softcore", None, ligand_indices)

    # n_states=1: IBSBiasForce 的 log-sum-exp 表达式里 `for k in range(1, n_states)` 循环
    # 不执行, pivot=0, sum=exp(0)=1, log(1)=0, 整个表达式恒等于裸的 state-k CV 能量
    # (cv_0_int+cv_0_rest-f_0, f_0=0) —— 也就是零偏置、零 SGD 的纯单态物理系统,
    # 且复用与生产环境完全相同的 build_ibs_dual_system/_create_softcore_force 代码路径。
    diag_system, ibs_wrapper = ibs_engine.build_ibs_dual_system(
        system, topology, ligand_indices,
        lambdas_coul=[0.0], lambdas_vdw=[lambda_vdw],
        alchemical_params=alchemical_params,
        potential_type="softcore",
        restraint_params=boresch_params,
        temperature=300 * unit.kelvin,
        prefix="diagW0k0",
        box_vectors=box_vectors,
        reference_positions=positions,
    )
    return diag_system, topology, positions, velocities, box_vectors


def run_diagnostic_md(args):
    platform_name, props = ibs_engine._build_platform_properties(args.platform)
    diag_system, topology, positions, velocities, box_vectors = build_diagnostic_system(
        args.output_dir, platform_name, props, args.lambda_vdw
    )

    has_shield = ibs_engine._system_has_global_parameter(diag_system, "lambda_shield")
    print(f"  lambda_shield 参数{'存在' if has_shield else '不存在(该系统变体不含 WCA 防护壳)'}")

    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 2.0 / unit.picosecond, 0.002 * unit.picosecond)
    simulation = app.Simulation(topology, diag_system, integrator, platform, props)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocities(velocities)

    if has_shield:
        simulation.context.setParameter("lambda_shield", float(args.lambda_shield))

    if args.minimize_first:
        print("  --minimize-first: 执行能量极小化 (仅作对照用; 默认路径不做, "
              "因为 minimize 会抹掉我们想观察的碰撞信号)")
        simulation.minimizeEnergy(maxIterations=200)

    diag_dir = os.path.join(args.output_dir, DIAG_SUBDIR)
    os.makedirs(diag_dir, exist_ok=True)
    prefix = _make_prefix(args.lambda_vdw, args.lambda_shield)
    dcd_path, log_path, chk_path = abfe_pipeline.attach_simulation_reporters(
        simulation, prefix=prefix, output_dir=diag_dir,
        traj_interval=args.traj_interval, energy_interval=args.traj_interval, chk_interval=5000,
    )

    group1_energies = []
    steps_done = 0
    chunk = 500
    crashed_at_step = None
    while steps_done < args.n_steps:
        this_chunk = min(chunk, args.n_steps - steps_done)
        simulation.step(this_chunk)
        steps_done += this_chunk
        pe = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not np.isfinite(pe):
            crashed_at_step = steps_done
            print(f"  !! 势能非有限值 ({pe}) 发生在第 {steps_done} 步 —— 立即停止。"
                  "这本身就是一个诊断信号 (越早炸越像 pose 里预置的真实碰撞)。")
            break
        group1_pe = simulation.context.getState(getEnergy=True, groups={1}).getPotentialEnergy()
        group1_energies.append(group1_pe.value_in_unit(unit.kilojoule_per_mole))
        if steps_done % (chunk * 10) == 0:
            print(f"  step {steps_done}/{args.n_steps}  U_total={pe:.2f} kJ/mol  "
                  f"U_group1(lambda_vdw={args.lambda_vdw:.4f} CV)={group1_energies[-1]:.2f} kJ/mol")

    energy_npy_path = os.path.join(diag_dir, f"{prefix}_group1_energy.npy")
    np.save(energy_npy_path, np.asarray(group1_energies, dtype=float))

    summary = {
        "n_steps_requested": args.n_steps,
        "n_steps_completed": steps_done,
        "crashed_at_step": crashed_at_step,
        "lambda_vdw": args.lambda_vdw,
        "lambda_shield": args.lambda_shield,
        "traj_interval": args.traj_interval,
        "dcd_path": dcd_path,
        "energy_log_path": log_path,
        "checkpoint_path": chk_path,
        "group1_energy_npy": energy_npy_path,
    }
    summary_path = os.path.join(diag_dir, f"{prefix}_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  运行摘要已写入 {summary_path}")
    return summary


def analyze_clashes(args):
    output_dir = args.output_dir
    diag_dir = os.path.join(output_dir, DIAG_SUBDIR)
    prefix = _make_prefix(args.lambda_vdw, args.lambda_shield)
    dcd_path = os.path.join(diag_dir, f"{prefix}_traj.dcd")
    if not os.path.exists(dcd_path):
        raise FileNotFoundError(f"未找到 {dcd_path}; 请先跑 MD 阶段 (不要加 --skip-md)。")

    paths = runabfe._cache_paths(output_dir, phase="complex")
    traj, _frame_indices = dexp_experiment.load_analysis_traj(dcd_path, paths["top"], max_frames=0)
    lig_idx, env_idx = dexp_experiment.get_ligand_env_heavy_indices(traj.topology, args.ligand_resname)

    with open(paths["xml"], "r") as f:
        native_system = XmlSerializer.deserialize(f.read())
    nb_force = next(f for f in native_system.getForces() if isinstance(f, openmm.NonbondedForce))
    n_particles = native_system.getNumParticles()
    sigma_all = np.zeros(n_particles, dtype=float)
    eps_all = np.zeros(n_particles, dtype=float)
    for i in range(n_particles):
        _, sigma_i, epsilon_i = nb_force.getParticleParameters(i)
        sigma_all[i] = sigma_i.value_in_unit(unit.nanometer)
        eps_all[i] = epsilon_i.value_in_unit(unit.kilojoule_per_mole)
    sigma_lig, eps_lig = sigma_all[lig_idx], eps_all[lig_idx]
    sigma_env, eps_env = sigma_all[env_idx], eps_all[env_idx]
    sigma_ij = 0.5 * (sigma_lig[:, None] + sigma_env[None, :])
    eps_ij = np.sqrt(np.clip(eps_lig[:, None] * eps_env[None, :], 0.0, None))

    atoms = list(traj.topology.atoms)
    box = np.asarray(traj.unitcell_vectors, dtype=np.float64) if traj.unitcell_vectors is not None else None
    rows = []
    for f in range(len(traj)):
        box_vecs = box[f] if box is not None else None
        dists = dexp_experiment.compute_pairwise_distances_nm(
            np.asarray(traj.xyz[f], dtype=np.float64), lig_idx, env_idx, box_vecs,
        )
        sr6 = (sigma_ij / np.maximum(dists, 1e-6)) ** 6
        e_pair = 4.0 * eps_ij * (sr6 ** 2 - sr6)
        i_max, j_max = np.unravel_index(np.argmax(e_pair), e_pair.shape)
        lig_atom = atoms[lig_idx[i_max]]
        env_atom = atoms[env_idx[j_max]]
        rows.append({
            "frame_idx": f,
            "time_ps": float(traj.time[f]) if traj.time is not None else float("nan"),
            "e_max_kJ_mol": float(e_pair[i_max, j_max]),
            "r_min_nm": float(dists[i_max, j_max]),
            "lig_atom_idx": int(lig_idx[i_max]),
            "lig_atom_name": lig_atom.name,
            "env_atom_idx": int(env_idx[j_max]),
            "env_atom_name": env_atom.name,
            "env_resname": env_atom.residue.name,
            "env_resid": int(getattr(env_atom.residue, "resSeq", env_atom.residue.index)),
            "frame_total_LJ_kJ_mol": float(e_pair.sum()),
        })

    csv_path = os.path.join(diag_dir, f"{prefix}_pairwise_clash.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  逐帧最大碰撞 pair 表已写入 {csv_path} ({len(rows)} 帧)")

    group1_npy = os.path.join(diag_dir, f"{prefix}_group1_energy.npy")
    if os.path.exists(group1_npy):
        group1_energies = np.load(group1_npy)
        frame_totals = np.array([r["frame_total_LJ_kJ_mol"] for r in rows])
        n_compare = min(len(group1_energies), len(frame_totals))
        if n_compare > 0:
            diff = np.abs(group1_energies[:n_compare] - frame_totals[:n_compare])
            print(f"  逐 pair 求和 vs Group-1 总能量 交叉校验: max|diff|={diff.max():.3f} kJ/mol "
                  f"(应接近 0, 明显偏大说明分解代码本身有误)")

    threshold = args.clash_threshold_kj_mol
    flagged = [r for r in rows if r["e_max_kJ_mol"] > threshold]
    print(f"  {len(flagged)}/{len(rows)} 帧超过碰撞阈值 {threshold} kJ/mol")

    residue_hist = collections.Counter((r["env_resname"], r["env_resid"]) for r in flagged)
    ligatom_hist = collections.Counter(r["lig_atom_name"] for r in flagged)

    hist_path = os.path.join(diag_dir, f"{prefix}_clash_histogram.json")
    with open(hist_path, "w") as f:
        json.dump({
            "n_frames_total": len(rows),
            "n_frames_flagged": len(flagged),
            "clash_threshold_kj_mol": threshold,
            "residue_histogram": [
                {"env_resname": k[0], "env_resid": k[1], "count": v}
                for k, v in residue_hist.most_common(20)
            ],
            "ligand_atom_histogram": [
                {"lig_atom_name": k, "count": v} for k, v in ligatom_hist.most_common(20)
            ],
        }, f, indent=2)
    print(f"  碰撞直方图已写入 {hist_path}")

    if residue_hist:
        top_residue, top_count = residue_hist.most_common(1)[0]
        frac = top_count / max(len(flagged), 1)
        print(f"  最主要残基: {top_residue} ({top_count}/{len(flagged)} = {frac:.1%})")
        if frac > 0.7:
            print("  >>> 持久单一元凶：怀疑 pose 需要重新对接/拒绝。")
        else:
            print("  >>> 弥散分布：考虑 lambda=1 端点附近加密 λ 或延长该态预平衡, 而非否定 pose。")
    return csv_path, hist_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="./output_lrc_fix")
    parser.add_argument("--ligand-resname", default="MOL")
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--traj-interval", type=int, default=100)
    parser.add_argument(
        "--lambda-vdw", type=float, default=1.0,
        help="要单独拿出来跑的 window 0 态的 lambda_vdw 值 (必须是当前 schedule "
             "lambdas_var 里的一个值)。默认 1.0 = state 0(全耦合端点); 换成 window 0 "
             "里其他态的值(如 0.968649751561061, state 5)可以对比该态单独拿出来是否也一样正常。"
    )
    parser.add_argument(
        "--lambda-shield", type=float, default=1.0,
        help="1.0 = 防护壳完全关闭(纯 lambda=1 物理对照); 用真实窗口的 mean(lambda_vdw) "
             "(如 0.9843) 跑第二次对照, 排查 WCA shield 是否是碰撞放大的一部分原因。"
    )
    parser.add_argument("--clash-threshold-kj-mol", type=float, default=20.0)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--minimize-first", action="store_true")
    parser.add_argument("--skip-md", action="store_true", help="跳过 MD, 只分析已有轨迹。")
    parser.add_argument("--analyze-only", action="store_true", help="--skip-md 的别名。")
    args = parser.parse_args()

    if args.skip_md or args.analyze_only:
        analyze_clashes(args)
        return

    print("=== 阶段 1: window 0 (lambda_vdw=1.0) 单态诊断 MD ===")
    run_diagnostic_md(args)
    print("=== 阶段 2: 逐 pair 配体-环境 LJ 碰撞分解 ===")
    analyze_clashes(args)


if __name__ == "__main__":
    main()

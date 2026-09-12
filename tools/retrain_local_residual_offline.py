#!/usr/bin/env python
"""离线重训 R1 局部残差模型（换配体用）。**离线**——不在生产 run 内跑。

为什么必须离线：在同一个 run 里、用即将做 A/B 的那条臂自己的数据重拟 B_φ，
会让 `sampling_score_sha256` 变成 run-dependent（两臂不再共用同一把尺子），
而且拿缺构型的帧去拟合会把缺掉的态焊进模型、让所有收敛门更绿。
见 `local_residual/autofit.py` 末尾"为什么撤"和 EXP-030 §6.2。

四步链（`docs/RETRAIN_LOCAL_RESIDUAL.md`）：
    ① 帧 + ledger  →  ② 训练  →  ③ 导出 payload  →  ④ 部署 manifest
本脚本把 ① 换成"吃盘上已有的轨迹、按目标窗口的 λ 表重算标签"，②③④ 直接复用
仓里已有的三个生成器。

标签怎么来（这一步是全部风险所在）：
    target_u   = βU_k(x)，k 遍历目标窗口的 λ 表（`TraditionalMBARAnalyzer.compute_u_kn`）
    sampling_u = 这些帧**实际**被采样时那个哈密尔顿量的约化势能（额外算一个态）
    adjacent_gap_reduced        = diff(target_u, axis=1)
    log_importance_unnormalized = sampling_u[:, None] - target_u
与 `exp012_xed/mm_ledger.py:113-118` 同一套定义。

帧从哪来：默认 `<run-dir>/pre_equilibration.dcd`（名字是固定的），有多少帧用多少，
不抽稀、不设帧数下限。它是 λ=1 全耦合的单系综，正对上 window_0 λ 表所在的耦合端，
里面没有鬼影构型，所以不会踩"跨整条 λ 梯子采样再互相重加权 ⇒ r⁻¹² 爆炸"那条坑。
`build_dataset_v1` 要三个分区，单条轨迹按**时间**切三段。

⚠️ 已知口径差异：出厂那份 R1 的训练帧是 `sample-hard-window-scratch` 产的固定盒 NVT
窗口 scratch 轨迹（那条链现在在主线里是死的）；预平衡是 NPT。来源会写进 provenance。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _three_way_edges(n_frames: int) -> list[int]:
    """把 n 帧按时间切成三段的边界。相邻段首尾相接、不重叠、都非空。"""
    if n_frames < 3:
        raise ValueError(f"只有 {n_frames} 帧，切不出三个分区")
    return [round(i * n_frames / 3) for i in range(4)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="提供 system_native.xml / topology.cif / ligand_indices.json 的目录")
    parser.add_argument("--trajectory", action="append", default=None,
                        help="训练帧来源。默认 <run-dir>/pre_equilibration.dcd（名字是固定的），"
                             "整条全用、不抽稀、不设帧数下限；单条会按时间切成三段当 LORO 三折。"
                             "可重复给多条来覆盖默认。")
    parser.add_argument("--window-manifest", type=Path, default=None,
                        help="目标窗口的 manifest.json（取 lambdas_vdw/lambdas_coul）；"
                             "不给则用 --lambdas-vdw")
    parser.add_argument("--lambdas-vdw", type=float, nargs="+", default=None)
    parser.add_argument("--sampling-lambda-coul", type=float, default=1.0,
                        help="这些帧实际被采样时的 λ_coul（预平衡/rebalance 是 1.0）")
    parser.add_argument("--sampling-lambda-vdw", type=float, default=1.0)
    parser.add_argument("--max-frames-per-trajectory", type=int, default=None)
    parser.add_argument("--ligand-name", default="MOL")
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--output", required=True, type=Path,
                        help="产物目录（dataset / training / export / resources 都在这下面）")
    args = parser.parse_args(argv)

    import numpy as np

    from runabfe import load_native_system
    from local_residual.autofit import (
        _autofit_from_dataset, a_k_schedule, derived_capacities,
    )
    from local_residual.openmm_plugin import topology_atomic_numbers
    from local_residual.softlift_dataset import build_dataset_v1

    run_dir = args.run_dir.resolve()
    out = args.output.resolve()
    (out / "ledger").mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 加载体系: {run_dir}")
    system, topology, positions, box_vectors, ligand_indices = load_native_system(
        str(run_dir), prefer_equilibrated=False
    )
    atomic_numbers = topology_atomic_numbers(topology, system=system)
    n_ligand = len(ligand_indices)
    capacities = derived_capacities(n_ligand)
    print(f"      配体 {n_ligand} 原子；元素 {sorted(set(atomic_numbers))}；容量 {capacities}")

    if args.window_manifest is not None:
        manifest = json.loads(args.window_manifest.read_text(encoding="utf-8"))
        lambdas_vdw = [float(v) for v in manifest["lambdas_vdw"]]
        lambdas_coul = [float(v) for v in manifest["lambdas_coul"]]
    elif args.lambdas_vdw:
        lambdas_vdw = [float(v) for v in args.lambdas_vdw]
        lambdas_coul = [0.0] * len(lambdas_vdw)
    else:
        raise SystemExit("必须给 --window-manifest 或 --lambdas-vdw")
    print(f"[2/5] 目标窗口 λ_vdw = {[round(v, 4) for v in lambdas_vdw]}")

    # 轨迹按需抽稀，抽稀后的副本单独落盘（build_dataset_v1 要帧数与 ledger 对齐）
    import mdtraj

    sources = [Path(s) for s in (args.trajectory or [run_dir / "pre_equilibration.dcd"])]
    for source in sources:
        if not source.exists():
            raise SystemExit(f"找不到训练轨迹：{source}")

    loaded = [
        (source, mdtraj.load(str(source), top=str(run_dir / "topology.cif")))
        for source in sources
    ]
    if len(loaded) == 1:
        # `build_dataset_v1` 要三个分区（LORO 三折）。一条轨迹就按**时间**切三段，
        # 不隔帧交错：交错会把同一段相关时间的帧同时放进训练和 held-out，
        # held-out 会假绿。段与段本来就相关，所以 LORO 在这里只是过拟合的内部诊断，
        # 不是放行判据（见 local_residual/autofit.py 末尾）。
        source, traj = loaded[0]
        try:
            edges = _three_way_edges(traj.n_frames)
        except ValueError as exc:
            raise SystemExit(f"{source.name}: {exc}") from exc
        loaded = [
            (source, traj[edges[i]:edges[i + 1]]) for i in range(3)
        ]
        print(f"      {source.name}: {traj.n_frames} 帧 → 按时间切三段")

    traj_paths = []
    for index, (source, traj) in enumerate(loaded):
        if args.max_frames_per_trajectory and traj.n_frames > args.max_frames_per_trajectory:
            step = max(1, traj.n_frames // args.max_frames_per_trajectory)
            traj = traj[::step][: args.max_frames_per_trajectory]
        path = out / "ledger" / f"partition{index}.dcd"
        traj.save_dcd(str(path))
        traj_paths.append(str(path))
        print(f"      分区 {index}: {source.name} → {traj.n_frames} 帧")

    print(f"[3/5] 重算标签（{len(lambdas_vdw)} 个目标态 + 1 个采样态）")
    # 🔑 用**逐态软核 probe** 直接算 U^sc_k，不走 `TraditionalMBARAnalyzer.compute_u_kn`。
    #   后者是传统 REMD 腿的重加权入口，会给目标能量补 Beutler 软核缺的 1/V LRC 尾项，
    #   因此带一道"轨迹必须固定盒"的门（NPT 轨迹 relative_span > 1e-3 直接拒绝）。
    #   R1 的训练标签不需要那个尾项：它是 V 和 λ 的光滑函数，跟模型看得见的局部几何
    #   无关，模型既学不到也不该学它。这里评估的是同一个体系在不同 (λ_coul, λ_vdw)
    #   下的总势能——所有与 λ 无关的项（蛋白内部、水-水）在相邻 gap 里逐位抵消。
    import mdtraj as _md
    import openmm
    from openmm import unit

    from abfe_preoptimizer import ACESoftcorePotential, build_aces_probe_system_dual_lambda

    softcore = ACESoftcorePotential.from_dict(
        ACESoftcorePotential.optimize_alpha(n_ligand)
    )
    probe_system = build_aces_probe_system_dual_lambda(
        system, list(ligand_indices), softcore,
        fixed_lam_coul=0.0, fixed_lam_vdw=1.0,
        topology=topology, positions=positions, box_vectors=box_vectors,
    )
    for index in reversed(range(probe_system.getNumForces())):
        if "Barostat" in type(probe_system.getForce(index)).__name__:
            probe_system.removeForce(index)
    probe_integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    probe_context = openmm.Context(
        probe_system, probe_integrator,
        openmm.Platform.getPlatformByName(str(args.platform)),
    )
    kt = 0.008314462618 * float(args.temperature)
    states = [(c, v) for c, v in zip(lambdas_coul, lambdas_vdw)]
    states.append((float(args.sampling_lambda_coul), float(args.sampling_lambda_vdw)))

    runs = []
    for index, path in enumerate(traj_paths):
        traj = _md.load(path, top=str(run_dir / "topology.cif"))
        frames = traj.n_frames
        block = np.zeros((frames, len(states)), dtype=np.float64)
        for frame in range(frames):
            probe_context.setPeriodicBoxVectors(
                *(traj.unitcell_vectors[frame] * unit.nanometer)
            )
            probe_context.setPositions(traj.xyz[frame] * unit.nanometer)
            for state_index, (lam_c, lam_v) in enumerate(states):
                probe_context.setParameter("lam_coul", float(lam_c))
                probe_context.setParameter("lam_vdw", float(lam_v))
                energy = probe_context.getState(getEnergy=True).getPotentialEnergy()
                block[frame, state_index] = (
                    energy.value_in_unit(unit.kilojoule_per_mole) / kt
                )
        if not np.all(np.isfinite(block)):
            raise SystemExit(f"分区 {index} 的约化势能里有非有限值")
        target_u = block[:, : len(lambdas_vdw)]
        sampling_u = block[:, len(lambdas_vdw)]
        ledger = out / "ledger" / f"partition{index}.npz"
        report = out / "ledger" / f"partition{index}_report.json"
        np.savez(
            ledger,
            frame_index=np.arange(frames, dtype=np.int64),
            adjacent_gap_reduced=np.diff(target_u, axis=1),
            log_importance_unnormalized=sampling_u[:, None] - target_u,
        )
        report.write_text(json.dumps({"frame_count": int(frames)}), encoding="utf-8")
        worst = float(np.abs(sampling_u[:, None] - target_u).max())
        print(f"      分区 {index}: {frames} 帧, |log_importance| 最大 {worst:.4g}")
        runs.append({
            "run_id": f"offline_partition{index}",
            "trajectory_path": path,
            "ledger_path": str(ledger),
            "ledger_report_path": str(report),
        })
    del probe_context, probe_integrator
    n_k = np.asarray([json.loads((out / "ledger" / f"partition{i}_report.json").read_text())["frame_count"]
                      for i in range(len(traj_paths))], dtype=int)

    print("[4/5] 建数据集")
    delta_A, A_k = a_k_schedule(lambdas_vdw)
    dataset = out / "dataset" / "softlift_dataset_v1.npz"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    protocol_sha = __import__("hashlib").sha256(
        json.dumps({
            "chain": "tools.retrain_local_residual_offline", "version": 1,
            "lambdas_vdw": lambdas_vdw, "lambdas_coul": lambdas_coul,
            "sampling_state": [args.sampling_lambda_coul, args.sampling_lambda_vdw],
            "sources": [str(s) for s in sources], "capacities": capacities,
        }, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    build_dataset_v1(
        runs=runs, ligand_topology_indices=list(ligand_indices),
        topology_path=run_dir / "topology.cif", output_path=dataset,
        delta_A=delta_A, A_k_window=A_k, protocol_sha256=protocol_sha,
        atomic_numbers_override=atomic_numbers, **capacities,
    )

    print("[5/5] 训练 → 导出 → manifest")
    result = _autofit_from_dataset(
        dataset_path=dataset, first_trajectory=traj_paths[0], work_dir=out,
        resources_dir=out / "resources", manifest_path=out / "resources" / "manifest.json",
        topology_cif=run_dir / "topology.cif",
        ligand_indices_path=run_dir / "ligand_indices.json",
        system_xml=run_dir / "system_native.xml", ligand_name=args.ligand_name,
        protocol_sha=protocol_sha, vocabulary=tuple(sorted(set(atomic_numbers))),
        capacities=capacities, n_ligand=n_ligand, lambdas_vdw=lambdas_vdw,
        teacher=None, max_epochs=args.max_epochs, patience=30,
        train_seeds=args.train_seeds, log=print,
    )
    provenance = out / "retrain_provenance.json"
    provenance.write_text(json.dumps({
        "schema_version": "local-residual-offline-retrain-v1",
        "protocol_sha256": protocol_sha,
        "frame_sources": [str(s) for s in sources],
        "frames_per_partition": n_k.tolist(),
        "target_window_lambdas_vdw": lambdas_vdw,
        "target_window_lambdas_coul": lambdas_coul,
        "sampling_state": {"lambda_coul": args.sampling_lambda_coul,
                           "lambda_vdw": args.sampling_lambda_vdw},
        "ligand_atoms": n_ligand, "capacities": capacities,
        "manifest": str(result.manifest_path),
        "differs_from_factory_recipe": (
            "出厂 R1 用的是 hard_window0_run{1,2,3}（各 500 帧，sample-hard-window-scratch "
            "产的窗口 scratch 轨迹）；本次用的是盘上现成轨迹，来源不同。"
        ),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== 完成 ===\n  manifest   : {result.manifest_path}\n  provenance : {provenance}")
    print("\n下一步才是验收：上机 A/B，两臂各自标定并冻结自己的 f_k。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

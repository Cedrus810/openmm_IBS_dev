#!/usr/bin/env python
"""GPU 冒烟：把换配体自动重训链（`local_residual.autofit`）在真实体系上整条跑一遍。

四步，任何一步失败都当场停：

  1. 从一个已有 run 目录的原生缓存加载 System/topology/配体索引（不重跑 GROMACS）；
  2. 启动期元素覆盖门；
  3. `autofit_r1`：探针采样 → ledger → dataset → 训练 → 导出 → run 私有 manifest；
  4. **用生产 loader 回读产物** —— `build_outer_lambda_local_residual_runtime`
     会验插件源码 sha、payload/weights sha、配体化学指纹、原子类型索引，
     全过才算这份自动重训出来的模型真的能进生产路径。

默认参数是**冒烟规模**（几分钟），不是生产规模。真要用于生产，把 --probe-states /
--frames-per-state / --steps-between-frames / --train-seeds 调回默认生产值。

    python tools/validation/validate_local_residual_autofit.py \
        --run-dir /path/to/an/existing/output --platform CUDA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="含 system_native.xml / topology.cif / ligand_indices.json 的 run 目录")
    parser.add_argument("--output", type=Path, default=None,
                        help="产物写到哪（默认 <run-dir>/autofit_smoke）。不会碰 run-dir 里的既有结果。")
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--ligand-name", default="MOL")
    parser.add_argument("--teacher", default=None,
                        help="可选：MACE 模型路径/文件名，只用于元素覆盖判定")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260911)
    # 冒烟规模
    parser.add_argument("--probe-states", type=int, default=6)
    parser.add_argument("--frames-per-state", type=int, default=6)
    parser.add_argument("--steps-between-frames", type=int, default=500)
    parser.add_argument("--equilibration-steps", type=int, default=1000)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-epochs", type=int, default=200)
    args = parser.parse_args(argv)

    from runabfe import load_native_system
    from local_residual.autofit import autofit_r1, startup_element_gate
    from local_residual.openmm_plugin import build_outer_lambda_local_residual_runtime

    run_dir = args.run_dir.resolve()
    output_dir = (args.output or run_dir / "autofit_smoke").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] 从原生缓存加载: {run_dir}")
    system, topology, positions, box_vectors, ligand_indices = load_native_system(
        str(run_dir), prefer_equilibrated=False
    )
    print(f"      原子 {topology.getNumAtoms()}，配体 {len(ligand_indices)} 个原子")

    print("[2/4] 启动期元素覆盖门")
    vocabulary = startup_element_gate(
        topology=topology, system=system, autofit_enabled=True, teacher=args.teacher, log=print
    )
    print(f"      体系元素: {list(vocabulary)}")

    print("[3/4] autofit 整链（探针采样 → ledger → dataset → 训练 → 导出 → manifest）")
    started = time.time()
    result = autofit_r1(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature_kelvin=args.temperature,
        platform_name=args.platform,
        output_dir=output_dir,
        ligand_name=args.ligand_name,
        topology_cif=run_dir / "topology.cif",
        ligand_indices_path=run_dir / "ligand_indices.json",
        system_xml=run_dir / "system_native.xml",
        n_probe_states=args.probe_states,
        frames_per_state=args.frames_per_state,
        steps_between_frames=args.steps_between_frames,
        equilibration_steps=args.equilibration_steps,
        seed=args.seed,
        teacher=args.teacher,
        max_epochs=args.max_epochs,
        train_seeds=args.train_seeds,
        log=print,
    )
    print(f"      用时 {time.time() - started:.0f}s；复用既有产物={result.reused_existing}")

    print("[4/4] 用生产 loader 回读这份自动重训出来的模型")
    runtime = build_outer_lambda_local_residual_runtime(
        topology=topology,
        ligand_indices=ligand_indices,
        system=system,
        temperature_kelvin=args.temperature,
        potential_type="softcore",
        output_dir=str(output_dir),
        ligand_indices_path=str(run_dir / "ligand_indices.json"),
        leg_name="complex",
        platform_name=args.platform,
        resource_manifest=result.manifest_path,
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    print("\n=== PASS ===")
    print(f"  manifest        : {result.manifest_path}")
    print(f"  词表 / 容量     : {report['type_vocabulary']} / {report['capacities']}")
    print(f"  held-out 改善   : {report['mean_relative_improvement']:+.4f}"
          f"（离线信号，不是验收口径）")
    print(f"  sampling score  : {runtime.sampling_score_sha256}")
    print(f"  em_policy       : {runtime.em_policy}")
    print("\n下一步才是真验收：上机 A/B，两臂各自标定并冻结自己的 f_k。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

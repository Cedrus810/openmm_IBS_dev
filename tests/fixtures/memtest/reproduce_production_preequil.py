#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""走**完全相同的生产路径**复现预平衡 NaN，然后逐项替换以定位差异。

## 为什么需要它

`diagnose_nan.py` 是我手写的重建：自己 `createSystem`、自己用 `.gro` 坐标。
它在 CUDA 上带膜 barostat 跑 200 000 步（0.4 ns）**不炸**；
而生产在**第 5000 步之前**就 NaN（监控 CSV 与 DCD 都是 0 字节）。

差异必然在生产路径里。已知生产额外做的事：

  1. 命中 `system_cache_exists()` 时走 `load_native_system()`，
     **System 来自 `system_native.xml`**、**topology 来自 `topology.cif`**、
     **坐标优先级是 预平衡 DCD → CIF → .gro**（本次 DCD 是 0 字节，所以用 CIF）；
  2. `center_system_rigidly()` 用的是缓存里的 `ligand_indices.json`；
  3. `ABFEPipeline` 的 platform properties 含 `CudaCompiler=nvcc`。

所以本脚本**不自己搭**任何东西：直接调 runabfe 的函数，构造真正的
`ABFEPipeline`，调真正的 `pre_equilibrate()`，只把步数压小。
再用 `--from-gro` / `--no-cache` 逐项替换，看哪一项换掉之后就不炸了。

## 用法（memtest/ 内）

    # ① 原样复现（应当在很少的步数内 NaN）
    python reproduce_production_preequil.py --steps=20000

    # ② 只把坐标换成 .gro（其余仍走缓存）→ 若不炸，责任在 CIF 坐标
    python reproduce_production_preequil.py --steps=20000 --from-gro

    # ③ 完全绕开缓存，从 .top/.gro 重建 → 若不炸，责任在缓存往返
    python reproduce_production_preequil.py --steps=20000 --no-cache
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, ".."))))

import numpy as np
from openmm import app, unit

import runabfe
from abfe_pipeline import ABFEPipeline


def main() -> int:
    flags = {
        a.split("=", 1)[0]: (a.split("=", 1)[1] if "=" in a else "")
        for a in sys.argv[1:]
        if a.startswith("--")
    }
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    steps = int(flags.get("--steps", "20000"))
    from_gro = "--from-gro" in flags
    no_cache = "--no-cache" in flags

    config_path = argv[0] if argv else "abfe_config.json"
    config = {
        k: v
        for k, v in json.load(open(config_path, encoding="utf-8")).items()
        if not k.startswith("_")
    }
    output_dir = os.path.abspath(config.get("output", "./output_membrane"))
    include_dir = runabfe.find_gmx_include_dir(config.get("gmx_path"))

    print(f"配置 {config_path}")
    print(f"steps={steps}  from_gro={from_gro}  no_cache={no_cache}\n")

    cache_hit = (not no_cache) and runabfe.system_cache_exists(
        output_dir,
        gro_file=config["gro"],
        top_file=config["top"],
        ligand_resname=config["ligand"],
        gmx_include_dir=include_dir,
    )
    print(f"缓存命中: {cache_hit}")

    if cache_hit:
        system, topology, positions, box_vectors, ligand_indices = (
            runabfe.load_native_system(
                output_dir,
                gro_file=config["gro"],
                top_file=config["top"],
                gmx_include_dir=include_dir,
                prefer_equilibrated=False,
            )
        )
        source = "load_native_system（坐标优先 DCD → CIF → .gro）"
    else:
        top_for_openmm, conversion = runabfe.openmm_compatible_gromacs_top(
            config["top"], include_dir,
            compat_dir=os.path.join(output_dir, "gromacs_openmm_compat"),
        )
        if conversion:
            print(f"  （已做 [ pairs ] funct 2 等价转换）")
        system, topology, positions, box_vectors, ligand_indices = (
            runabfe.build_system_from_gromacs(
                config["gro"], top_for_openmm, config["ligand"], include_dir
            )
        )
        source = "build_system_from_gromacs（坐标来自 .gro）"
    print(f"坐标来源: {source}")

    if from_gro:
        gro = app.GromacsGroFile(config["gro"])
        old = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
        new = np.asarray(gro.positions.value_in_unit(unit.nanometer), dtype=float)
        if old.shape == new.shape:
            delta = np.linalg.norm(new - old, axis=1)
            print(
                f"  🔁 坐标强制换成 .gro：与原来逐原子位移 max {delta.max():.6f} nm, "
                f"中位数 {np.median(delta):.6f} nm"
            )
        else:
            print(f"  ⚠️ 形状不同 {old.shape} vs {new.shape}，直接采用 .gro")
        positions = gro.positions

    positions, box_vectors = runabfe.center_system_rigidly(
        positions, box_vectors, ligand_indices
    )

    run_dir = os.path.join(output_dir, "repro_preequil")
    pipeline = ABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=config["temperature"],
        output_dir=run_dir,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        platform_name=config["platform"],
        environment_type=config.get("system_type"),
        membrane=config.get("membrane"),
        dispersion_protocol=config.get("dispersion_protocol"),
        forcefield_family=config.get("forcefield_family"),
    )

    print(f"\n跑 pre_equilibrate({steps} 步)…（与生产同一函数）\n")
    try:
        pipeline.pre_equilibrate(n_steps=steps, save_traj=True, resume=False)
    except Exception as exc:
        print(f"\n❌ 炸了：{type(exc).__name__}: {exc}")
        monitor = os.path.join(run_dir, "pre_equilibration_monitor.csv")
        if os.path.isfile(monitor) and os.path.getsize(monitor) > 0:
            print(f"\n监控尾部（{monitor}）：")
            with open(monitor, encoding="utf-8") as handle:
                rows = handle.read().splitlines()
            for row in rows[:1] + rows[-12:]:
                print("   ", row)
        else:
            print("  监控文件为空 → NaN 发生在第一次落盘之前（< 5000 步）。")
        print(
            "\n下一步：\n"
            "  · 若本次是缓存命中，加 --from-gro 再跑；不炸 ⇒ 责任在 CIF 坐标。\n"
            "  · 若 --from-gro 仍炸，加 --no-cache 再跑；不炸 ⇒ 责任在缓存往返"
            "（System XML 或 topology.cif）。\n"
            "  · 若 --no-cache 也炸，而 diagnose_nan.py 不炸，"
            "差异只剩 platform properties 与 topology 来源。"
        )
        return 1

    print("\n✅ 没炸。")
    print("   ⇒ 本组合不是 NaN 的触发条件。把上一次炸掉的组合与这次对比，差异就是根因。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

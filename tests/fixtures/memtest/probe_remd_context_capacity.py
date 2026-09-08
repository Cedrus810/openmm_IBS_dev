#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""[P0-REMD-CUDA] 逐个创建 REMD replica Context，测出到底第几个断、断时剩多少显存。

## 为什么需要这个探针

生产日志只给一句 `No compatible CUDA device is available` —— 那是 OpenMM 在**所有**
设备都初始化失败时的通用文案，**真 OOM 也长这样**。于是"显存不够"和"其它初始化失败"
无法区分，只能靠猜。而已知事实是反直觉的：

| | 原子数 | 盒 | 12×CUDA Context |
|---|---|---|---|
| `output_lrc_fix`（可溶） | 73536 | ~735 nm³ | ✅ 成功过 |
| `memtest/output_membrane_100ns`（膜） | 45354 | ~440 nm³ | ❌ 失败 |

**更小的体系失败**，所以"体系大 → 爆显存"这条推理在这里不成立，必须实测。

本探针走**与生产完全相同**的 `_prepare_pme_coulomb_leg_system` 路径（不是裸
`system_native.xml`），因为 decharging 腿真正驻留显存的是那个 prepared System。
每建一个 Context 就打印一次 `nvidia-smi` 的剩余显存，失败时打印原始异常。

产出的两个数直接回答两件事：
1. 断点是不是显存（剩余显存单调下降到接近 0 ⟹ 是；第 1 个就失败 ⟹ 不是）；
2. **这张卡最多能开几个 λ 状态** —— 也就是把 λ 数减到多少才不会退 CPU。

## 用法

    ./tests/run_offline_tests.sh --version >/dev/null 2>&1   # 只为拿到 env 激活方式
    # 或直接：
    source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh && mamba activate openmm_dev
    cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
    python memtest/probe_remd_context_capacity.py memtest/output_membrane_100ns

    # 想对照可溶体系（那条已知成功的）：
    python memtest/probe_remd_context_capacity.py output_lrc_fix

预期耗时：每个 Context 几秒（首个含 kernel 编译，可能 30-60 s），12 个约 2-3 分钟。
⚠️ 需要 GPU 空闲。它只建 Context、不跑动力学，不写任何生产产物，
**不碰坐标、不碰任何指纹，不作废已有预平衡或已完成的腿。**
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _free_vram_mib():
    """返回 (used, free, total) MiB；拿不到就返回 None（探针不因此失败）。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip().splitlines()[0]
        return tuple(int(v.strip()) for v in out.split(","))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="含 system_native.xml / topology.cif / ligand_indices.json 的输出目录",
    )
    parser.add_argument("--n-contexts", type=int, default=12, help="最多尝试建几个（默认 12）")
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument(
        "--replay-preoptimizer",
        action="store_true",
        help=(
            "先复现生产在 Stage 1 前 17 秒做的那一步：建 λ 路径预优化的 probe Context "
            "+ LocalEnergyMinimizer.minimize，然后释放，再建 12 个 replica Context。"
            "这是探针（12/12 成功）与生产（失败）之间**唯一**剩下的差异。"
        ),
    )
    args = parser.parse_args()

    import json

    import openmm
    from openmm import app, unit

    import ibs_engine as ie

    out = args.output_dir
    system_path = os.path.join(out, "system_native.xml")
    topology_path = os.path.join(out, "topology.cif")
    ligand_path = os.path.join(out, "ligand_indices.json")
    for path in (system_path, topology_path, ligand_path):
        if not os.path.isfile(path):
            print(f"❌ 缺少 {path}")
            return 2

    print(f"── 读取 {out} ──")
    with open(system_path, encoding="utf-8") as handle:
        system_template = openmm.XmlSerializer.deserialize(handle.read())
    cif = app.PDBxFile(topology_path)
    topology = cif.topology
    positions = cif.positions
    with open(ligand_path, encoding="utf-8") as handle:
        ligand_indices = [int(i) for i in json.load(handle)["ligand_indices"]]

    box = system_template.getDefaultPeriodicBoxVectors()
    print(f"   原子数 {system_template.getNumParticles()} | 配体原子 {len(ligand_indices)}")
    print(f"   盒: {[round(v.value_in_unit(unit.nanometer)[i], 3) for i, v in enumerate(box)]} nm")
    baseline = _free_vram_mib()
    if baseline:
        print(f"   起始显存: used={baseline[0]} free={baseline[1]} total={baseline[2]} MiB")

    print("\n── 构建 decharging 腿的 prepared System（与生产同一路径）──")
    # [MEM-00c] 带电配体需要冻结的 co-ion spec；本探针只测显存容量，
    # 所以中性配体直接过，带电配体读已落盘的 spec（没有就明确报出来，不在这里重选）。
    spec = None
    spec_path = os.path.join(out, "checkpoints", "coalchemical_ion_spec.json")
    if os.path.isfile(spec_path):
        with open(spec_path, encoding="utf-8") as handle:
            spec = json.load(handle)
        print(f"   使用已冻结的 co-ion 身份: {spec_path}")

    prepared = ie._prepare_pme_coulomb_leg_system(
        system_template,
        ligand_indices,
        lambda_name="lambda_coul",
        allow_charged_ligand=True,
        topology=topology,
        positions=positions,
        box_vectors=box,
        co_alchemical_ion_spec=spec,
    )
    print(f"   ✓ prepared System: {prepared.getNumForces()} 个力")

    resolved_platform, props = ie._build_platform_properties(args.platform)
    platform = openmm.Platform.getPlatformByName(resolved_platform)

    if args.replay_preoptimizer:
        # 复现 `abfe_pipeline._optimize_dual_lambda_path`（约 2551-2567 行）在
        # Stage 1 前做的事。它 `del context, integrator, probe_sys` 之后就直接建
        # replica —— 若这一步留下了没释放的显存/Context，就会在这里显形。
        from abfe_preoptimizer import build_aces_probe_system_dual_lambda
        from abfe_core import ACESoftcorePotential

        print("\n── 复现 λ 路径预优化的 probe Context（生产在 Stage 1 前 17 秒做的）──")
        # 与 `abfe_pipeline.py:2484` 逐字一致的构造方式。
        softcore_obj = ACESoftcorePotential.from_dict(
            ACESoftcorePotential.optimize_alpha(len(ligand_indices))
        )
        probe_sys = build_aces_probe_system_dual_lambda(
            system_template,
            ligand_indices,
            softcore_obj,
            fixed_lam_coul=0.0,
            fixed_lam_vdw=1.0,
        )
        integ = openmm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
        )
        probe_ctx = openmm.Context(probe_sys, integ, platform, props)
        probe_ctx.setPositions(positions)
        probe_ctx.setPeriodicBoxVectors(*box)
        openmm.LocalEnergyMinimizer.minimize(probe_ctx, maxIterations=500)
        vram = _free_vram_mib()
        if vram:
            print(f"   probe Context 建成后: used={vram[0]} free={vram[1]} MiB")
        del probe_ctx, integ, probe_sys
        import gc

        gc.collect()
        vram = _free_vram_mib()
        if vram:
            print(f"   del + gc.collect() 之后: used={vram[0]} free={vram[1]} MiB")
            if baseline and vram[0] - baseline[0] > 50:
                print(
                    f"   ⚠️ 比起始多占了 {vram[0] - baseline[0]} MiB —— "
                    "预优化的显存没有真正释放，这就是生产与本探针的差异所在。"
                )
        # 生产在这一步之后**没有** gc.collect()，也没有重新读起始基线；
        # 下面的"平均每 Context"因此会把这段残留算进去，属实。

    print(f"\n── 逐个创建 Context (platform={resolved_platform}, props={props}) ──")

    contexts = []
    integrators = []
    failed_at = None
    try:
        for i in range(args.n_contexts):
            try:
                integ = openmm.LangevinMiddleIntegrator(
                    300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
                )
                ctx = openmm.Context(prepared, integ, platform, props)
                ctx.setPositions(positions)
                ctx.setPeriodicBoxVectors(*box)
                contexts.append(ctx)
                integrators.append(integ)
            except Exception as exc:  # noqa: BLE001
                failed_at = i
                vram = _free_vram_mib()
                print(f"\n❌ 第 {i + 1} 个 Context 创建失败（已成功 {len(contexts)} 个）")
                if vram:
                    print(f"   失败瞬间显存: used={vram[0]} free={vram[1]} total={vram[2]} MiB")
                print(f"   原始异常 [{type(exc).__name__}]: {exc}")
                break

            vram = _free_vram_mib()
            if vram:
                per_ctx = (
                    (vram[0] - baseline[0]) / len(contexts) if baseline else float("nan")
                )
                print(
                    f"   ✓ Context {i + 1:2d}/{args.n_contexts}  "
                    f"used={vram[0]:5d} free={vram[1]:5d} MiB  "
                    f"平均每 Context ≈ {per_ctx:.0f} MiB"
                )
            else:
                print(f"   ✓ Context {i + 1:2d}/{args.n_contexts}（拿不到 nvidia-smi 读数）")

        print("\n── 结论 ──")
        if failed_at is None:
            print(f"✅ {args.n_contexts} 个 Context 全部建成 —— 这张卡装得下。")
            print("   那么生产里的失败**不是**本探针能复现的容量问题：")
            print("   差别只剩'生产进程在此之前还建过别的 CUDA Context'，需要在生产路径就地打点。")
        else:
            vram = _free_vram_mib()
            print(f"⚠️ 上限是 {len(contexts)} 个 Context（第 {failed_at + 1} 个失败）。")
            if vram and baseline:
                per_ctx = (vram[0] - baseline[0]) / max(1, len(contexts))
                print(
                    f"   平均每 Context ≈ {per_ctx:.0f} MiB，"
                    f"总量 {vram[2]} MiB ⟹ 理论上限 ≈ {int(vram[2] / max(per_ctx, 1))} 个。"
                )
            print(f"   ⟹ Stage 1/2 的 λ 状态数应当 ≤ {len(contexts)}，否则会静默退 CPU。")
            print("   若失败瞬间 free 显存仍然很大，那**不是** OOM，别按 OOM 修。")
    finally:
        for ctx in contexts:
            del ctx
        contexts.clear()
        integrators.clear()

    return 0 if failed_at is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

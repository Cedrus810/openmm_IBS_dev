#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线复判 §9 膜质量门：吃一条已有的预平衡轨迹，输出与生产同源的判定。

## 为什么需要它

质量门原先只在 `ABFEPipeline._evaluate_membrane_quality_gate_after_equilibration`
里跑，于是"验证质量门"唯一的办法是**重烧一遍预平衡**（膜体系 10–100 ns，5 h 起）。
memtest 的门连续两次崩在同一个 `UnboundLocalError` 上（2026-07-31 / 08-02），
第二次是又烧掉 30 min GPU 之后才看到的。

本脚本把这一步变成**秒级、纯 CPU、不建 Context**：

    python tools/diagnostics/evaluate_membrane_quality_gate.py \
        --output-dir memtest/output_membrane \
        --config memtest/abfe_config.json \
        --membrane-input memtest/membrane_input.json

## 它凭什么与生产判得一样

不自己搭东西 —— 每一步都调生产函数，与 `runabfe.main()` 的膜分支逐项对齐：

| 这里 | 生产路径 |
| --- | --- |
| `runabfe.load_native_system(require_bonded_topology=True)` | `runabfe.py:3769/3830` |
| `abfe_core.parse_gromacs_topology` + `classify_system_composition` | `runabfe.py:3976` |
| `abfe_core.run_membrane_quality_gate` | `abfe_pipeline.py:1404` 走同一个函数 |

§0.5.7 的教训就是"离线重建与生产路径不一致，是白花几轮的直接原因"
（当时离线诊断用 `.top` 拓扑 + 不调 PBC 修复，所以怎么跑都不炸）。
所以这里刻意**不**接受"给我一个 DCD 和一个 gro 就判"这种便捷入口：
拓扑必须来自与那条轨迹同一次运行的输出目录。

⚠️ 默认用 `advisory` 模式判读，**这不是把门放松**：advisory 只影响"失败要不要
抛异常"，不影响任何阈值。要看生产资格请传 `--mode enforce`（失败即非零退出）。
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import abfe_core as core
import runabfe


def _load_json(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"{what} 不存在: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _strip_comments(config):
    """去掉 `_` 前缀的注释键，与 `memtest/check_membrane_protocol.py` 同一口径。"""
    return {k: v for k, v in config.items() if not str(k).startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="离线复判 §9 膜质量门（纯 CPU，不建 Context）",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="那次运行的输出目录（需含 system_native.xml / pre_equilibration.dcd）",
    )
    parser.add_argument("--config", required=True, help="该次运行用的 abfe_config.json")
    parser.add_argument(
        "--membrane-input",
        default=None,
        help="膜输入声明；默认取 config 的 membrane_input_declaration",
    )
    parser.add_argument(
        "--trajectory",
        default=None,
        help="要判的轨迹；默认 <output-dir>/pre_equilibration.dcd",
    )
    parser.add_argument(
        "--mode",
        default=core.MEMBRANE_QUALITY_GATE_MODE_ADVISORY,
        help=(
            "advisory（默认，失败只报告不抛）或 enforce（失败即非零退出）。"
            "阈值两者完全相同——模式不改判据。"
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help=(
            "报告落盘目录；默认**不落盘**，避免离线复判覆盖那次运行的"
            "membrane_quality_gate.json（生产记录不该被诊断脚本改写）"
        ),
    )
    args = parser.parse_args()

    config_dir = os.path.dirname(os.path.abspath(args.config)) or "."
    config = _strip_comments(_load_json(args.config, "配置"))

    def _resolve(path):
        """**config 里**的相对路径以配置文件所在目录为基准。

        生产用法是 `cd memtest && python ../runabfe.py --config abfe_config.json`，
        所以 config 里的 `gro`/`top`/`membrane_input_declaration` 都是相对
        `memtest/` 写的。命令行显式传进来的路径则按**当前工作目录**解析
        （那是 CLI 的常规约定）—— 两者混用会拼出 `memtest/memtest/...`。
        """
        if path is None or os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(config_dir, path))

    def _resolve_cli_or_config(cli_value, config_value):
        """命令行值按 CWD、config 值按 config 目录。"""
        if cli_value:
            return cli_value
        return _resolve(config_value)

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        raise SystemExit(f"输出目录不存在: {args.output_dir}")

    traj_path = args.trajectory or os.path.join(output_dir, "pre_equilibration.dcd")
    if not os.path.exists(traj_path):
        raise SystemExit(
            f"轨迹不存在: {traj_path}\n"
            "这个脚本判的是**已经跑出来的**轨迹，不会自己跑动力学。"
        )

    # ---- 环境类型 / 膜协议：只看 config，与 runabfe.py:3751 同一口径 ----
    barostat_protocol = core.resolve_membrane_protocol(
        config.get("system_type"), config.get("membrane")
    )
    if barostat_protocol["system_type"] != core.ENVIRONMENT_TYPE_MEMBRANE:
        raise SystemExit(
            f"config 的 system_type = {config.get('system_type')!r} 不是 membrane。"
            "§9 质量门只对膜体系有定义，可溶体系没有 APL / 膜厚 / 叶片这些量。"
        )
    normal_axis = (barostat_protocol["membrane"] or {}).get("normal_axis", "z")

    declaration_path = _resolve_cli_or_config(
        args.membrane_input, config.get("membrane_input_declaration")
    )
    if not declaration_path:
        raise SystemExit(
            "缺膜输入声明：§9 的口袋定义（pocket_atom_indices）不接受运行时推断，"
            "必须由声明显式给出。用 --membrane-input 指定。"
        )
    declared = _load_json(declaration_path, "膜输入声明")

    # ---- 拓扑：与生产同一入口，且膜体系必须带键（§0.5.7）----
    print(f"[1/3] 从输出目录加载 System/拓扑: {output_dir}")
    system, topology, positions, box_vectors, ligand_indices = runabfe.load_native_system(
        output_dir,
        gro_file=_resolve(config.get("gro")),
        top_file=_resolve(config.get("top")),
        gmx_include_dir=runabfe.find_gmx_include_dir(config.get("gmx_path")),
        phase="complex",
        require_bonded_topology=True,
    )

    # ---- 组成：一律从 `.top` 的 [ molecules ] + [ moleculetype ] 取，不靠残基名 ----
    print("[2/3] 解析体系组成（.top 的 [ molecules ] + [ moleculetype ]）")
    composition = core.classify_system_composition(
        core.parse_gromacs_topology(
            _resolve(config.get("top")),
            runabfe.find_gmx_include_dir(config.get("gmx_path")),
        ),
        ligand_molecule_name=declared.get("ligand_molecule_name"),
        declared_roles=declared.get("molecule_roles"),
    )
    if composition["n_atoms_total"] != topology.getNumAtoms():
        raise SystemExit(
            f".top 展开得到 {composition['n_atoms_total']} 个原子，"
            f"但拓扑有 {topology.getNumAtoms()} 个 —— 原子索引对不上，"
            "按索引的选择会静默错位。"
        )

    # `equilibration_length_ns` 与生产同一口径：上游时长不可考时传 None，
    # 提取器退回用轨迹自身跨度（`runabfe.py:4158-4161` 的注释即此意）。
    membrane_input_report = core.validate_membrane_input(
        topology,
        positions,
        box_vectors,
        declared=declared,
        normal_axis=normal_axis,
        composition=composition,
    )
    equil_ns = None
    if membrane_input_report["nominal_equilibration_precheck_applicable"]:
        equil_steps = int(config.get("n_equil_steps", 5_000_000))
        timestep_ps = float(config.get("timestep_ps", 0.002))
        equil_ns = float(membrane_input_report["upstream_equilibration_ns"]) + (
            equil_steps * timestep_ps / 1000.0
        )

    inputs = {
        key: declared.get(key)
        for key in (
            "pocket_atom_indices",
            "coion_atom_index",
            "literature_apl_nm2",
            "pure_lipid_reference_apl_nm2",  # 诊断专用，不判门（MEM-12）
            "ligand_resname",
        )
    }
    inputs["composition"] = composition
    inputs["equilibration_length_ns"] = equil_ns

    # 时间轴必须显式重建：mdtraj 读 DCD 给的 `traj.time` 是整数帧号
    # （实测 memtest 那条 10 ns / 500 帧的轨迹被当成 0.499 ns，差 20 倍）。
    frame_interval_ps = core.pre_equilibration_frame_interval_ps(
        timestep_ps=config.get("timestep_ps")
    )
    print(
        f"[3/3] 判定 §9 质量门（mode={args.mode}，轨迹 {traj_path}，"
        f"{frame_interval_ps:g} ps/帧）"
    )
    try:
        report = core.run_membrane_quality_gate(
            traj_path,
            topology,
            inputs,
            mode=args.mode,
            normal_axis=normal_axis,
            ligand_indices=ligand_indices,
            output_dir=args.report,
            frame_interval_ps=frame_interval_ps,
            log=print,
        )
    except RuntimeError as exc:
        # enforce 模式下门未过/未评估都会 raise —— 这是它的定义行为，不是脚本崩了。
        print(f"\n❌ [enforce] {exc}")
        return 2

    print("\n" + "=" * 70)
    if not report.get("evaluated"):
        print(f"⚠️ 未能评估: {report.get('blocked_reason')}")
        return 3
    print(f"protocol_version = {report['protocol_version']}  "
          f"thresholds_version = {report['thresholds_version']}  "
          f"tail_window = {report['tail_window_ns']} ns")
    print(f"passed = {report['passed']}")
    if report["failed_checks"]:
        print(f"failed_checks = {report['failed_checks']}")
        print(report["remediation"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

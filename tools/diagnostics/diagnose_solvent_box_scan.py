#!/usr/bin/env python
"""溶剂腿盒子尺寸扫描。

目的：判定 2026-07-28 那轮溶剂腿的 3.000 nm 盒（= 2×padding，溶质尺寸贡献为 0）
到底给 ΔG_solvent 带来了多大的有限尺寸偏差，从而决定生产默认 padding 该取多少。

背景数字（历史参考运行 `output_lrc_fix/`，在 Atenolol-rank11，不在本分支；
3.000 nm 盒，2574 粒子）：

    decharging  62.8865 ± 1.0191 kJ/mol
    vanishing  101.8247 ± 1.4651 kJ/mol
    总计       162.7831 ± 1.7847 kJ/mol   （已含 constraint 修正 -1.9280）

参考值 `result.txt` 反解出的溶剂腿去电荷是 ~68.1 kJ/mol，差 ~5.2 kJ/mol。
如果那 5.2 是有限尺寸偏差，decharging 会随盒子变大往 68 爬并收敛；
如果盒子加大后纹丝不动，就说明差异另有来源，不要为此改生产默认值。

每次调用只跑一个 padding，产物落在独立目录，互不干扰、可单独 --resume。
本脚本走的是生产代码路径（`build_and_cache_solvent_leg` + `ABFEPipeline`），
不是平行实现——否则扫描结果说明不了生产管线的问题。

用法::

    python tools/diagnostics/diagnose_solvent_box_scan.py --padding 1.5
    python tools/diagnostics/diagnose_solvent_box_scan.py --padding 2.4

`--padding` 与盒边的关系是 `盒边 = 配体最长轴 + 2×padding`（本体系配体最长轴
1.257 nm）。要复现 3.000 nm 基线作对照，用 `--padding 0.8715`。
"""

from __future__ import annotations

# 默认运行目录：统一由 tools/_run_dir.py 解析（ABFE_OUTPUT_DIR -> abfe_config.json
# 的 "output" -> ./output）。2026-08-31 前这里硬编码 output_lrc_fix，那是
# Atenolol-rank11 的验收基线目录，不在本工程区分支里。显式传参永远优先。
import sys as _abfe_rd_sys
from pathlib import Path as _AbfeRdPath

_ABFE_TOOLS_ROOT = _AbfeRdPath(__file__).resolve().parents[1]
if str(_ABFE_TOOLS_ROOT) not in _abfe_rd_sys.path:
    _abfe_rd_sys.path.insert(0, str(_ABFE_TOOLS_ROOT))
from _run_dir import DEFAULT_RUN_DIR  # noqa: E402


# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))


import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np

import runabfe
from abfe_core import solvent_box_edge_nm
from abfe_pipeline import ABFEPipeline

# 与 output_lrc_fix/final_binding_results.json 的 provenance.config 逐项对齐。
# 扫描只允许盒子这一个自由度变化，其余采样设置必须和生产那轮完全一致，
# 否则测出来的差值分不清是盒子还是采样协议造成的。
PRODUCTION_SAMPLING = dict(
    decoupling_scheme="dual_lambda",
    potential_type="softcore",
    system_type="solvent",
    n_steps_per_window=250_000,
    steps_per_update=500,
    n_states_per_stage=12,
    stage1_n_states=12,
    stage2_n_states=17,
    enable_early_stop=False,
    enable_gradual_warmup=True,
    warmup_steps=500_000,
    n_workers=None,
    parallel_stages=False,
    decharge_method="pme",
    allow_disk_boresch_autoload=False,
    enable_lambda_refine=False,
    refine_n_steps_per_window=30_000,
    refine_steps_per_update=500,
    refine_max_window_span_kJ=35.0,
    pilot_finite_difference_delta=0.01,
    pilot_n_steps_per_state=30_000,
    ibs_lse_log_residual_tolerance=0.5,
    min_bias_updates=12,
    max_bias_updates=50,
    required_consecutive_bias_updates=3,
    max_bias_warmup_steps=500_000,
)

N_EQUIL_STEPS = 5_000_000
TEMPERATURE_K = 300.0
IONIC_STRENGTH_M = 0.15


def _build_solvent_cache(scan_dir: str, args, log) -> Dict:
    """在 scan_dir 下用生产 builder 建一个指定 padding 的溶剂腿缓存。"""
    system, topology, positions, _box, ligand_indices = runabfe.load_native_system(
        args.source_dir, phase="complex", prefer_equilibrated=False
    )
    # load_native_system 返回的是 OpenMM app.Topology（只有 .atoms()），
    # 不是 abfe_pipeline 里那个有 .atom(i) 的 mdtraj Topology。
    ligand_resname = runabfe._get_residue_name_by_atom_index(
        topology, int(ligand_indices[0])
    )
    include_dir = runabfe.find_gmx_include_dir(args.gmx_path)

    pos_nm = np.asarray(positions.value_in_unit(runabfe.unit.nanometer), dtype=float)
    edge_nm, extent_nm = solvent_box_edge_nm(
        pos_nm[ligand_indices], padding_nm=args.padding
    )
    log(
        f"配体最长轴 {extent_nm:.4f} nm + 2×{args.padding:.4f} nm "
        f"→ 立方盒 {edge_nm:.4f} nm（体积 {edge_nm ** 3:.1f} nm³）"
    )

    identity = runabfe._ligand_parameter_identity(
        system,
        topology,
        ligand_indices,
        ligand_resname,
        args.top,
        None,
        include_dir,
        padding_nm=args.padding,
    )
    if not runabfe.solvent_cache_exists(
        scan_dir, ionic_strength_molar=IONIC_STRENGTH_M, expected_identity=identity
    ):
        ok = runabfe.build_and_cache_solvent_leg(
            scan_dir,
            topology,
            positions,
            ligand_indices,
            ligand_resname,
            ligand_ffxml=args.ligand_xml,
            top_file=args.top,
            gmx_include_dir=include_dir,
            ionic_strength_molar=IONIC_STRENGTH_M,
            cache_identity=identity,
            padding_nm=args.padding,
        )
        if not ok:
            raise RuntimeError("溶剂腿缓存构建失败")
    else:
        log("复用已有溶剂腿缓存（padding 与身份指纹均匹配）")

    with open(os.path.join(scan_dir, "solvent_cache_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    return {
        "padding_nm": args.padding,
        "box_edge_nm": manifest.get("box_edge_nm"),
        "ligand_longest_axis_nm": manifest.get("ligand_longest_axis_nm"),
        "na_count": manifest.get("na_count"),
        "cl_count": manifest.get("cl_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--padding",
        type=float,
        required=True,
        help="配体每一侧的溶剂厚度 nm；盒边 = 配体最长轴 + 2×padding",
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_RUN_DIR,
        help="提供复合物 System 缓存的目录（只读，用来取配体拓扑/坐标/参数身份）",
    )
    parser.add_argument("--scan-root", default="solvent_box_scan")
    parser.add_argument("--top", default="topol.top")
    parser.add_argument(
        "--ligand-xml",
        default=None,
        help="配体力场 XML；不给就在扫描目录里按 .top 重新生成（确定性的，只是多花点时间）",
    )
    parser.add_argument("--gmx-path", default=None)
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    scan_dir = os.path.join(args.scan_root, f"pad_{args.padding:.4f}")
    os.makedirs(scan_dir, exist_ok=True)
    log_path = os.path.join(scan_dir, "scan.log")
    log_fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log(f"=== 溶剂盒扫描 padding={args.padding} nm → {scan_dir} ===")
    box_info = _build_solvent_cache(scan_dir, args, log)

    sys_solv, top_solv, pos_solv, box_solv, lig_idx_solv = runabfe.load_native_system(
        scan_dir, phase="solvent", prefer_equilibrated=args.resume
    )
    pos_solv, box_solv = runabfe.center_system_rigidly(pos_solv, box_solv, lig_idx_solv)
    log(f"溶剂 System 粒子数 {sys_solv.getNumParticles()}")

    leg_dir = os.path.join(scan_dir, "solvent_leg")
    pipeline = ABFEPipeline(
        system=sys_solv,
        topology=top_solv,
        positions=pos_solv,
        box_vectors=box_solv,
        ligand_indices=lig_idx_solv,
        temperature=TEMPERATURE_K,
        output_dir=leg_dir,
        checkpoint_dir=os.path.join(leg_dir, "checkpoints"),
        platform_name=args.platform,
    )

    need_equil = not runabfe.equilibrium_is_done(
        leg_dir,
        expected_fingerprint=runabfe._pre_equilibration_fingerprint(
            pipeline.system,
            pipeline.ligand_indices,
            pipeline.temperature,
            pipeline.pressure,
            positions=pipeline.positions,
            box_vectors=pipeline.box_vectors,
            requested_steps=N_EQUIL_STEPS,
        ),
    )
    log(f"预平衡：{'需要' if need_equil else '已完成，跳过'}")

    t0 = time.time()
    results = pipeline.run_full_pipeline(
        boresch_params=None,  # 溶剂腿绝不加 Boresch
        dexp_params=None,
        torsion_params=None,
        resume=args.resume,
        run_equilibration=need_equil,
        **PRODUCTION_SAMPLING,
    )
    wall_s = time.time() - t0

    def _stage_total(name: str):
        path = os.path.join(leg_dir, "checkpoints", f"{name}.json")
        if not os.path.isfile(path):
            return None, None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("total_delta_G"), data.get("total_error")

    dg_dechg, err_dechg = _stage_total("stage1_decharging")
    dg_vdw, err_vdw = _stage_total("stage2_vanishing")

    summary = {
        **box_info,
        "n_particles": int(sys_solv.getNumParticles()),
        "decharging_kJ_mol": dg_dechg,
        "decharging_error_kJ_mol": err_dechg,
        "vanishing_kJ_mol": dg_vdw,
        "vanishing_error_kJ_mol": err_vdw,
        "constraint_correction_kJ_mol": results.get("constraint_correction_kJ_mol"),
        "total_delta_G_solvent_kJ_mol": results.get("total_delta_G_complex_kJ_mol"),
        "total_error_kJ_mol": results.get("total_error_kJ_mol"),
        "wall_seconds": round(wall_s, 1),
        "sampling_config": PRODUCTION_SAMPLING,
    }
    out_path = os.path.join(scan_dir, "box_scan_result.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log(
        f"盒边 {summary['box_edge_nm']} nm | 粒子 {summary['n_particles']} | "
        f"decharging {dg_dechg} | vanishing {dg_vdw} | "
        f"总计 {summary['total_delta_G_solvent_kJ_mol']} kJ/mol | "
        f"耗时 {wall_s / 3600:.2f} h"
    )
    log(f"结果已写入 {out_path}")
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单文件 DEXP 拟合与稳定性对比实验。

目标：
1. 读取 pre_equilibration.dcd 的后 500 帧做 Orbv3 -> DEXP 拟合。
2. 用拟合后的 DEXP 替身势跑 1 ns 稳定性测试。
3. 用原始势能再跑 1 ns 作为 baseline。
4. 导出非键项 lambda schedule 对比（同步线性 vs interaction-separation）。

典型用法：
    python dexp_experiment.py

如果需要显式指定输入：
    python dexp_experiment.py ^
        --traj output/pre_equilibration.dcd ^
        --traj-top output/topology.cif ^
        --gmx-top topol.top ^
        --ligand MOL ^
        --platform CUDA
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np

DEFAULT_PATHS = {
    "traj": "output/pre_equilibration.dcd",
    "traj_top": "output/topology.cif",
    "system_xml": "output/system_native.xml",
    "ligand_indices": "output/ligand_indices.json",
    "gmx_top": "topol.top",
}


def require_module(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception as exc:
        raise RuntimeError(
            f"当前 Python 环境缺少依赖 `{name}`，无法运行该实验脚本。"
        ) from exc


def require_openmm():
    try:
        import openmm  # type: ignore
        from openmm import app, unit, XmlSerializer  # type: ignore
        return openmm, app, unit, XmlSerializer
    except Exception as exc:
        raise RuntimeError(
            "当前 Python 环境缺少 `openmm`，无法执行拟合后的 1 ns 稳定性测试。"
        ) from exc


def load_abfe_symbols():
    try:
        from abfe_core import (  # type: ignore
            HAS_ORB,
            NumpyEncoder,
            Orbv3DEXPFittingPipeline,
            Orbv3SurrogateFitter,
            SurrogateSystemBuilder,
        )
    except Exception as exc:
        raise RuntimeError("无法导入项目内的 DEXP / Orb 辅助模块。") from exc
    if not HAS_ORB:
        raise RuntimeError(
            "当前环境未启用 Orb 相关依赖（例如 torch/openmmml），无法进行 DEXP 拟合。"
        )
    return {
        "NumpyEncoder": NumpyEncoder,
        "Orbv3DEXPFittingPipeline": Orbv3DEXPFittingPipeline,
        "Orbv3SurrogateFitter": Orbv3SurrogateFitter,
        "SurrogateSystemBuilder": SurrogateSystemBuilder,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DEXP 拟合 + 1ns 稳定性对比实验")
    parser.add_argument("--traj", default=DEFAULT_PATHS["traj"], help="预平衡轨迹 DCD")
    parser.add_argument("--traj-top", default=DEFAULT_PATHS["traj_top"], help="轨迹对应拓扑，推荐 output/topology.cif")
    parser.add_argument("--gmx-top", default=DEFAULT_PATHS["gmx_top"], help="GROMACS .top，用于 MM 参考能与系统语义")
    parser.add_argument("--system-xml", default=DEFAULT_PATHS["system_xml"], help="原始 OpenMM System XML")
    parser.add_argument("--ligand-indices", default=DEFAULT_PATHS["ligand_indices"], help="配体原子索引 JSON")
    parser.add_argument("--ligand", default="MOL", help="配体残基名")
    parser.add_argument("--gmx-include-dir", default=None, help="GROMACS include 目录")
    parser.add_argument("--output-dir", default="output/dexp_experiment", help="实验输出目录")
    parser.add_argument("--fit-frames", type=int, default=500, help="从末段时间窗中最多取多少帧参与拟合")
    parser.add_argument("--fit-last-ns", type=float, default=5.0, help="只使用轨迹最后多少 ns 做拟合")
    parser.add_argument("--fit-env-radius", type=float, default=0.45, help="环境筛选半径 (nm)")
    parser.add_argument("--fit-r-min", type=float, default=0.20, help="拟合距离下限 (nm)")
    parser.add_argument("--fit-r-max", type=float, default=0.80, help="拟合距离上限 (nm)")
    parser.add_argument("--temperature", type=float, default=300.0, help="温度 (K)")
    parser.add_argument("--device", default="cuda", help="Orb 设备，例如 cuda/cpu")
    parser.add_argument("--platform", default="CPU", help="OpenMM 平台，例如 CPU/CUDA")
    parser.add_argument("--sim-ns", type=float, default=1.0, help="每套体系模拟时长 (ns)")
    parser.add_argument("--dt-fs", type=float, default=2.0, help="积分步长 (fs)")
    parser.add_argument("--friction-ps", type=float, default=1.0, help="Langevin 摩擦系数 (1/ps)")
    parser.add_argument("--report-interval", type=int, default=1000, help="状态输出步频")
    parser.add_argument("--traj-interval", type=int, default=5000, help="DCD 输出步频")
    parser.add_argument("--schedule-states", type=int, default=16, help="导出的 lambda 状态数")
    parser.add_argument("--seed", type=int, default=20260526, help="随机种子")
    parser.add_argument("--minimize", action="store_true", help="在每次 1 ns 测试前先做一次最小化")
    parser.add_argument("--warmup-steps", type=int, default=50000, help="DEXP surrogate 慢启动步数")
    parser.add_argument("--warmup-stages", type=int, default=20, help="DEXP surrogate 慢启动分段数")
    parser.add_argument("--softstart-dt-fs", type=float, default=0.2, help="软启动初始步长 (fs)")
    parser.add_argument("--ramp-dt-fs", default="0.5,1.0,2.0", help="逐级升温步长列表 (fs, 逗号分隔)")
    parser.add_argument("--reuse-fit-labels", action="store_true", help="复用 output-dir 下已有的能量标注缓存，只重新拟合 DEXP 参数")
    return parser.parse_args()


def ensure_file(path: str, label: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_gmx_include_dir_from_runabfe(user_path: str | None = None) -> str | None:
    try:
        from runabfe import find_gmx_include_dir  # type: ignore
    except Exception:
        return None
    try:
        return find_gmx_include_dir(user_path)
    except Exception:
        return None


def _infer_gmx_include_dir_from_top(top_file: str) -> str | None:
    candidates: List[str] = []
    try:
        with open(top_file, "r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                match = re.search(r"Data prefix:\s*(.+?)\s*$", line)
                if match:
                    prefix = match.group(1).strip()
                    candidates.append(os.path.join(prefix, "share", "gromacs", "top"))
                    candidates.append(prefix)
                    break
    except Exception:
        return None

    include_re = re.compile(r'#include\s+"([^"]+)"')
    try:
        with open(top_file, "r", encoding="utf-8", errors="ignore") as handle:
            for _ in range(80):
                line = handle.readline()
                if not line:
                    break
                match = include_re.search(line)
                if not match:
                    continue
                include_rel = match.group(1)
                if ".ff/" in include_rel:
                    ff_dir = include_rel.split(".ff/", 1)[0] + ".ff"
                    top_dir = os.path.dirname(os.path.abspath(top_file))
                    candidates.append(os.path.join(top_dir, ff_dir))
    except Exception:
        pass

    for path in candidates:
        if path and os.path.exists(path):
            if os.path.basename(path) == "top":
                return path
            if os.path.isdir(path) and any(name.endswith(".ff") for name in os.listdir(path)):
                return path
    return None


def resolve_gmx_include_dir(user_path: str | None, top_file: str) -> str | None:
    for candidate in (
        user_path,
        _find_gmx_include_dir_from_runabfe(user_path),
        _infer_gmx_include_dir_from_top(top_file),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def load_ligand_indices(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "ligand_indices" in payload:
        return [int(x) for x in payload["ligand_indices"]]
    if isinstance(payload, list):
        return [int(x) for x in payload]
    raise ValueError(f"无法从 {path} 解析配体索引")


def select_tail_indices_from_time(traj, fit_frames: int, fit_last_ns: float) -> List[int]:
    import numpy as np

    n_frames_total = len(traj)
    if n_frames_total == 0:
        return []

    time_ps = getattr(traj, "time", None)
    if time_ps is None or len(time_ps) != n_frames_total:
        start = max(0, n_frames_total - fit_frames)
        return list(range(start, n_frames_total))

    time_ps = np.asarray(time_ps, dtype=float)
    last_time_ps = float(time_ps[-1])
    window_start_ps = last_time_ps - float(fit_last_ns) * 1000.0
    in_window = np.where(time_ps >= window_start_ps)[0]
    if len(in_window) == 0:
        start = max(0, n_frames_total - fit_frames)
        return list(range(start, n_frames_total))
    if len(in_window) <= fit_frames:
        return [int(idx) for idx in in_window.tolist()]

    sampled = np.linspace(in_window[0], in_window[-1], fit_frames, dtype=int)
    return [int(idx) for idx in sampled.tolist()]


def detect_suspicious_fit(fitted_params: Dict) -> Dict:
    bounds = {
        "alpha_vdw": (10.0, 30.0),
        "r0_vdw": (0.28, 0.45),
        "A_fit": (0.1, 5.0),
        "B_fit": (0.1, 3.0),
        "offset_c0": (-200.0, 200.0),
        "offset_c1": (-50.0, 50.0),
    }
    eps = 1.0e-6
    hits: List[str] = []
    for key, (lower, upper) in bounds.items():
        value = fitted_params.get(key)
        if value is None:
            continue
        if abs(float(value) - lower) < eps:
            hits.append(f"{key}=lower_bound({lower})")
        elif abs(float(value) - upper) < eps:
            hits.append(f"{key}=upper_bound({upper})")
    return {
        "suspicious_fit": bool(hits),
        "boundary_hits": hits,
    }


def summarize_delta_e(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "mean_abs": math.nan,
        }
    if len(values) == 1:
        val = float(values[0])
        return {
            "count": 1,
            "mean": val,
            "std": 0.0,
            "min": val,
            "max": val,
            "mean_abs": abs(val),
        }
    mean_val = float(statistics.fmean(values))
    return {
        "count": int(len(values)),
        "mean": mean_val,
        "std": float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean_abs": float(statistics.fmean(abs(v) for v in values)),
    }


def choose_delta_e_threshold(delta_e_values: Sequence[float], base_threshold: float = 500.0) -> Tuple[float, Dict]:
    stats = summarize_delta_e(delta_e_values)
    polluted = False
    reason = "default"
    threshold = float(base_threshold)
    center = float(stats["mean"]) if stats["count"] > 0 else 0.0
    if stats["count"] == 0:
        return threshold, {"polluted": False, "reason": "no_data", "stats": stats, "center": center}
    if abs(stats["mean"]) > 50.0 or stats["std"] > 200.0:
        polluted = True
        threshold = 200.0
        reason = "abs_mean_gt_50_or_std_gt_200"
    if abs(stats["mean"]) > 100.0 or stats["std"] > 350.0:
        polluted = True
        threshold = 100.0
        reason = "severe_pollution"
    if np.isfinite(stats["std"]):
        threshold = max(50.0, min(threshold, 4.0 * float(stats["std"]) + 20.0))
    return threshold, {"polluted": polluted, "reason": reason, "stats": stats, "center": center}


def fit_dexp_from_tail_frames(args: argparse.Namespace, output_dir: str) -> Dict:
    md = require_module("mdtraj")
    symbols = load_abfe_symbols()
    NumpyEncoder = symbols["NumpyEncoder"]
    Orbv3DEXPFittingPipeline = symbols["Orbv3DEXPFittingPipeline"]
    Orbv3SurrogateFitter = symbols["Orbv3SurrogateFitter"]
    _, _, unit, _ = require_openmm()
    import numpy as np

    print(
        f"[1/4] 载入轨迹并选取最后 {args.fit_last_ns:.2f} ns 内最多 {args.fit_frames} 帧做 DEXP 拟合"
    )
    args.gmx_include_dir = resolve_gmx_include_dir(args.gmx_include_dir, args.gmx_top)
    if not args.gmx_include_dir:
        raise RuntimeError(
            "无法定位 GROMACS include 目录。请显式传入 "
            "`--gmx-include-dir /path/to/gromacs/share/gromacs/top`。"
        )
    print(f"    GROMACS include 目录: {args.gmx_include_dir}")
    traj = md.load(args.traj, top=args.traj_top)
    if len(traj) == 0:
        raise RuntimeError("轨迹为空，无法进行 DEXP 拟合")

    fit_indices = select_tail_indices_from_time(traj, args.fit_frames, args.fit_last_ns)
    fit_traj = traj[fit_indices]
    if fit_traj.unitcell_vectors is not None:
        fit_traj = fit_traj.image_molecules(inplace=False)
    lig_idx = np.array(fit_traj.top.select(f"resname {args.ligand}"), dtype=int)
    if len(lig_idx) == 0:
        raise ValueError(f"未在轨迹拓扑中找到配体残基 `{args.ligand}`")

    ref_frame = fit_traj[-1]
    env_search_radius = max(float(args.fit_env_radius), float(args.fit_r_max), 0.85)
    if env_search_radius > float(args.fit_env_radius) + 1.0e-8:
        print(
            f"    环境搜索半径已自动扩展到 {env_search_radius:.2f} nm "
            f"(原始短程半径 {args.fit_env_radius:.2f} nm)"
        )
    raw_env = md.compute_neighbors(ref_frame, env_search_radius, lig_idx)[0]
    env_idx = np.setdiff1d(raw_env, lig_idx, assume_unique=True)
    if len(env_idx) == 0:
        raise RuntimeError("未找到配体附近环境原子，请增大 --fit-env-radius")

    all_nums = np.array([a.element.atomic_number for a in fit_traj.top.atoms], dtype=int)

    pipeline = Orbv3DEXPFittingPipeline(device=args.device)
    mm_contexts = build_mm_le_contexts_from_system_xml(
        args.system_xml,
        ligand_indices=lig_idx.tolist(),
        environment_indices=env_idx.tolist(),
    )

    fit_log_rows: List[Dict] = []
    raw_delta_e_values: List[float] = []
    raw_orb_values: List[float] = []
    raw_mm_coul_values: List[float] = []
    raw_mm_vdw_values: List[float] = []
    fit_log_path = os.path.join(output_dir, "fit_frame_diagnostics.csv")
    fit_label_meta_path = os.path.join(output_dir, "fit_label_cache_meta.json")
    print(f"    实际参与拟合帧数: {len(fit_indices)}")
    fit_xyz = np.asarray(fit_traj.xyz, dtype=np.float64)
    fit_time = np.asarray(getattr(traj, "time", np.arange(len(traj), dtype=float)), dtype=float)[fit_indices]
    fit_box = None
    if fit_traj.unitcell_vectors is not None:
        fit_box = np.asarray(fit_traj.unitcell_vectors, dtype=np.float64)
    reuse_labels = False
    cached_rows_by_frame: Dict[int, Dict[str, str]] = {}
    if args.reuse_fit_labels and os.path.isfile(fit_log_path) and os.path.isfile(fit_label_meta_path):
        try:
            with open(fit_label_meta_path, "r", encoding="utf-8") as handle:
                cache_meta = json.load(handle)
            frame_indices_cached = [int(x) for x in cache_meta.get("fit_indices", [])]
            env_idx_cached = [int(x) for x in cache_meta.get("env_indices", [])]
            lig_idx_cached = [int(x) for x in cache_meta.get("ligand_indices", [])]
            reuse_labels = (
                frame_indices_cached == [int(x) for x in fit_indices.tolist()]
                and env_idx_cached == [int(x) for x in env_idx.tolist()]
                and lig_idx_cached == [int(x) for x in lig_idx.tolist()]
                and abs(float(cache_meta.get("env_search_radius_nm", -1.0)) - float(env_search_radius)) < 1.0e-8
            )
            if reuse_labels:
                for row in read_csv_rows(fit_log_path):
                    cached_rows_by_frame[int(row["frame_index"])] = row
                print(f"    复用已有能量标注缓存: {fit_log_path}")
            else:
                print("    已检测到旧缓存，但当前 frame/env 选择已变化，回退为重新标注。")
        except Exception:
            reuse_labels = False

    for local_idx in range(len(fit_indices)):
        frame_id = int(fit_indices[local_idx])
        pos_nm = fit_xyz[local_idx].copy()

        box_vecs = fit_box[local_idx] if fit_box is not None else np.eye(3) * 3.0
        box_lens = np.linalg.norm(box_vecs, axis=1)
        delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
        delta -= box_lens * np.round(delta / box_lens)
        dists = np.linalg.norm(delta, axis=-1)
        valid_dists = dists[(dists >= args.fit_r_min) & (dists <= args.fit_r_max)]
        candidate_dists = dists[dists <= env_search_radius]

        if reuse_labels and frame_id in cached_rows_by_frame:
            cached_row = cached_rows_by_frame[frame_id]
            e_orb_int = float(cached_row["e_orb_int_kjmol"])
            e_mm_coul = float(cached_row["e_mm_coul_kjmol"])
            e_mm_vdw = float(cached_row.get("e_mm_vdw_kjmol", "0.0"))
            delta_e = float(cached_row.get("delta_e_res_kjmol", cached_row["delta_e_kjmol"]))
        else:
            e_orb_int = pipeline._compute_orb_decomposition(pos_nm, lig_idx, env_idx, all_nums)
            e_mm_coul = 0.0
            e_mm_vdw = 0.0
            for label, ctx in mm_contexts.items():
                ctx.setPositions(pos_nm * unit.nanometer)
                energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoules_per_mole
                )
                if label == "coul":
                    e_mm_coul = energy
                elif label == "vdw":
                    e_mm_vdw = energy
            delta_e = float(e_orb_int - e_mm_coul)

        raw_orb_values.append(float(e_orb_int))
        raw_mm_coul_values.append(float(e_mm_coul))
        raw_mm_vdw_values.append(float(e_mm_vdw))
        if np.isfinite(delta_e):
            raw_delta_e_values.append(delta_e)

        fit_log_rows.append(
            {
                "frame_index": frame_id,
                "time_ps": float(fit_time[local_idx]),
                "e_orb_int_kjmol": float(e_orb_int),
                "e_mm_coul_kjmol": float(e_mm_coul),
                "e_mm_vdw_kjmol": float(e_mm_vdw),
                "delta_e_kjmol": delta_e,
                "delta_e_res_kjmol": delta_e,
                "n_env_pairs": int(len(candidate_dists)),
                "n_valid_pairs": int(len(valid_dists)),
                "used_for_fit": 0,
            }
        )
        if (local_idx + 1) % 50 == 0 or local_idx == len(fit_indices) - 1:
            print(f"    已处理 {local_idx + 1}/{len(fit_indices)} 帧")

    delta_threshold, delta_diag = choose_delta_e_threshold(raw_delta_e_values)
    print(
        "    ΔE_res 诊断: "
        f"mean={delta_diag['stats']['mean']:.2f} kJ/mol | "
        f"std={delta_diag['stats']['std']:.2f} | "
        f"centered-threshold={delta_threshold:.1f} | "
        f"polluted={delta_diag['polluted']}"
    )
    orb_stats = summarize_delta_e(raw_orb_values)
    mm_coul_stats = summarize_delta_e(raw_mm_coul_values)
    mm_vdw_stats = summarize_delta_e(raw_mm_vdw_values)
    print(
        "    能量分量: "
        f"E_orb mean={orb_stats['mean']:.2f} | "
        f"E_mm_coul mean={mm_coul_stats['mean']:.2f} | "
        f"E_mm_vdw mean={mm_vdw_stats['mean']:.2f} | "
        f"mean(orb-coul)={delta_diag['stats']['mean']:.2f}"
    )

    for row_idx, row in enumerate(fit_log_rows):
        delta_e = float(row["delta_e_kjmol"])
        n_valid_pairs = int(row["n_valid_pairs"])
        centered_delta = delta_e - float(delta_diag["center"])
        use_frame = int(
            np.isfinite(delta_e)
            and abs(centered_delta) < delta_threshold
            and n_valid_pairs > 0
        )
        row["used_for_fit"] = use_frame
        row["delta_e_threshold_kjmol"] = float(delta_threshold)
        row["delta_e_center_kjmol"] = float(delta_diag["center"])
        row["delta_e_centered_kjmol"] = float(centered_delta)

    # Rebuild the distance list from accepted frames using the same minimum-image rule.
    rebuilt_dists_per_frame: List[np.ndarray] = []
    accepted_delta_e_final: List[float] = []
    for row_idx, row in enumerate(fit_log_rows):
        if not int(row["used_for_fit"]):
            continue
        pos_nm = fit_xyz[row_idx]
        box_vecs = fit_box[row_idx] if fit_box is not None else np.eye(3) * 3.0
        box_lens = np.linalg.norm(box_vecs, axis=1)
        delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
        delta -= box_lens * np.round(delta / box_lens)
        dists = np.linalg.norm(delta, axis=-1)
        valid_dists = dists[(dists >= args.fit_r_min) & (dists <= args.fit_r_max)]
        candidate_dists = dists[dists <= env_search_radius]
        if len(valid_dists) == 0 or len(candidate_dists) == 0:
            row["used_for_fit"] = 0
            continue
        rebuilt_dists_per_frame.append(valid_dists)
        accepted_delta_e_final.append(float(row["delta_e_kjmol"]))

    if len(accepted_delta_e_final) < 10:
        raise RuntimeError(
            f"有效拟合帧只有 {len(accepted_delta_e_final)} 帧，无法稳定拟合 DEXP。"
        )

    fitter = Orbv3SurrogateFitter(fitting_region=(args.fit_r_min, args.fit_r_max))
    fitted_params = fitter.fit_parameters(rebuilt_dists_per_frame, accepted_delta_e_final)
    fitted_params["fit_frames_requested"] = int(args.fit_frames)
    fitted_params["fit_last_ns_requested"] = float(args.fit_last_ns)
    fitted_params["fit_frames_total"] = int(len(fit_indices))
    fitted_params["fit_frames_used"] = int(len(accepted_delta_e_final))
    fitted_params["fit_frame_start"] = int(fit_indices[0])
    fitted_params["fit_frame_end"] = int(fit_indices[-1])
    fitted_params["fit_time_start_ps"] = float(traj.time[fit_indices[0]]) if getattr(traj, "time", None) is not None else None
    fitted_params["fit_time_end_ps"] = float(traj.time[fit_indices[-1]]) if getattr(traj, "time", None) is not None else None
    fitted_params["env_radius_nm"] = float(args.fit_env_radius)
    fitted_params["env_search_radius_nm"] = float(env_search_radius)
    fitted_params["fit_region_nm"] = [float(args.fit_r_min), float(args.fit_r_max)]
    fitted_params["traj_total_frames"] = int(len(traj))
    fitted_params["delta_e_filter_threshold_kjmol"] = float(delta_threshold)
    fitted_params["delta_e_res_filter_threshold_kjmol"] = float(delta_threshold)
    fitted_params["delta_e_polluted"] = bool(delta_diag["polluted"])
    fitted_params["delta_e_pollution_reason"] = str(delta_diag["reason"])
    fitted_params["delta_e_mean_kjmol"] = float(delta_diag["stats"]["mean"])
    fitted_params["delta_e_std_kjmol"] = float(delta_diag["stats"]["std"])
    fitted_params["delta_e_mean_abs_kjmol"] = float(delta_diag["stats"]["mean_abs"])
    fitted_params["delta_e_res_mean_kjmol"] = float(delta_diag["stats"]["mean"])
    fitted_params["delta_e_res_std_kjmol"] = float(delta_diag["stats"]["std"])
    fitted_params["e_orb_int_mean_kjmol"] = float(orb_stats["mean"])
    fitted_params["e_orb_int_std_kjmol"] = float(orb_stats["std"])
    fitted_params["e_mm_coul_mean_kjmol"] = float(mm_coul_stats["mean"])
    fitted_params["e_mm_coul_std_kjmol"] = float(mm_coul_stats["std"])
    fitted_params["e_mm_vdw_mean_kjmol"] = float(mm_vdw_stats["mean"])
    fitted_params["e_mm_vdw_std_kjmol"] = float(mm_vdw_stats["std"])
    fitted_params.update(detect_suspicious_fit(fitted_params))

    params_path = os.path.join(output_dir, "dexp_fitted_params.json")
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(fitted_params, handle, indent=2, cls=NumpyEncoder)

    with open(fit_log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fit_log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fit_log_rows)
    with open(fit_label_meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fit_indices": [int(x) for x in fit_indices],
                "ligand_indices": [int(x) for x in lig_idx],
                "env_indices": [int(x) for x in env_idx],
                "env_search_radius_nm": float(env_search_radius),
            },
            handle,
            indent=2,
        )

    print(f"    拟合完成，已保存参数: {params_path}")
    if fitted_params.get("suspicious_fit"):
        print("    ⚠️ 检测到拟合参数撞边界，当前 DEXP 参数很可能不可靠")
        print(f"    ⚠️ 边界命中: {', '.join(fitted_params.get('boundary_hits', []))}")
    print(f"    帧诊断已保存: {fit_log_path}")
    return fitted_params


def load_last_frame_positions(traj_path: str, traj_top: str):
    md = require_module("mdtraj")
    openmm, _, unit, _ = require_openmm()

    traj = md.load(traj_path, top=traj_top)
    if len(traj) == 0:
        raise RuntimeError("轨迹为空，无法提取最后一帧坐标")

    xyz = traj.xyz[-1]
    positions = [
        openmm.Vec3(float(x), float(y), float(z))
        for x, y, z in xyz
    ] * unit.nanometer

    box_vectors = None
    if traj.unitcell_vectors is not None:
        box_vectors = [
            openmm.Vec3(float(v[0]), float(v[1]), float(v[2]))
            for v in traj.unitcell_vectors[-1]
        ] * unit.nanometer
    return traj, positions, box_vectors


def load_cached_system(system_xml: str, topology_cif: str):
    _, app, _, XmlSerializer = require_openmm()
    with open(system_xml, "r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())
    pdbx = app.PDBxFile(topology_cif)
    return system, pdbx.topology


def build_mm_le_contexts_from_system_xml(
    system_xml: str,
    ligand_indices: Sequence[int],
    environment_indices: Sequence[int],
):
    openmm, _, unit, XmlSerializer = require_openmm()
    with open(system_xml, "r", encoding="utf-8") as handle:
        system = XmlSerializer.deserialize(handle.read())

    nb_force = next(
        (force for force in system.getForces() if isinstance(force, openmm.NonbondedForce)),
        None,
    )
    if nb_force is None:
        raise RuntimeError("system_native.xml 中未找到 NonbondedForce，无法构建 MM 参考 L-E 相互作用")

    n_particles = system.getNumParticles()
    force_defs = {
        "coul": (
            "138.935456*q1*q2/max(r, 0.05)",
            ("q",),
        ),
        "vdw": (
            "4*eps*((sigma/r)^12-(sigma/r)^6); "
            "eps=sqrt(epsilon1*epsilon2); sigma=0.5*(sigma1+sigma2)",
            ("sigma", "epsilon"),
        ),
    }
    cutoff_nm = 0.85
    switching_nm = cutoff_nm - 0.15
    contexts = {}
    for label, (expr, per_params) in force_defs.items():
        le_sys = openmm.System()
        for atom_idx in range(n_particles):
            le_sys.addParticle(system.getParticleMass(atom_idx))
        le_force = openmm.CustomNonbondedForce(expr)
        for param_name in per_params:
            le_force.addPerParticleParameter(param_name)
        for atom_idx in range(n_particles):
            q, sigma, epsilon = nb_force.getParticleParameters(atom_idx)
            payload = []
            for param_name in per_params:
                if param_name == "q":
                    payload.append(q.value_in_unit(unit.elementary_charge))
                elif param_name == "sigma":
                    payload.append(sigma.value_in_unit(unit.nanometer))
                elif param_name == "epsilon":
                    payload.append(epsilon.value_in_unit(unit.kilojoule_per_mole))
            le_force.addParticle(payload)
        le_force.addInteractionGroup(
            [int(idx) for idx in ligand_indices],
            [int(idx) for idx in environment_indices],
        )
        le_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        le_force.setCutoffDistance(cutoff_nm * unit.nanometer)
        le_force.setUseSwitchingFunction(True)
        le_force.setSwitchingDistance(switching_nm * unit.nanometer)
        for exc_idx in range(nb_force.getNumExceptions()):
            p1, p2, _, _, _ = nb_force.getExceptionParameters(exc_idx)
            le_force.addExclusion(int(p1), int(p2))
        le_sys.addForce(le_force)
        contexts[label] = openmm.Context(le_sys, openmm.VerletIntegrator(0.001))
    return contexts


def clone_system(system):
    openmm, _, _, XmlSerializer = require_openmm()
    return XmlSerializer.deserialize(XmlSerializer.serialize(system))


def strip_barostat(system):
    openmm, _, _, _ = require_openmm()
    new_system = clone_system(system)
    for idx in reversed(range(new_system.getNumForces())):
        if isinstance(new_system.getForce(idx), openmm.MonteCarloBarostat):
            new_system.removeForce(idx)
    return new_system


def select_platform(platform_name: str):
    openmm, _, _, _ = require_openmm()
    resolved = platform_name.upper()
    if resolved == "CUDA":
        return openmm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}
    if resolved == "OPENCL":
        return openmm.Platform.getPlatformByName("OpenCL"), {}
    return openmm.Platform.getPlatformByName("CPU"), {}


def format_platform_label(platform, properties: Dict[str, str]) -> str:
    platform_name = str(platform.getName())
    if platform_name.upper() in {"CUDA", "OPENCL"}:
        device_suffix = ""
        device_index = properties.get("DeviceIndex")
        if device_index not in (None, ""):
            device_suffix = f":{device_index}"
        return f"{platform_name}{device_suffix} (GPU)"
    return f"{platform_name} (CPU)"


def summarize_series(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "min": float(values[0]), "max": float(values[0])}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def compute_ligand_rmsd_metrics(dcd_path: str, top_path: str, ligand_resname: str) -> Dict[str, float]:
    md = require_module("mdtraj")
    traj = md.load(dcd_path, top=top_path)
    if len(traj) == 0:
        return {"ligand_rmsd_mean_A": math.nan, "ligand_rmsd_max_A": math.nan}

    lig_atoms = traj.top.select(f"resname {ligand_resname} and not element H")
    if len(lig_atoms) == 0:
        lig_atoms = traj.top.select(f"resname {ligand_resname}")
    if len(lig_atoms) == 0:
        return {"ligand_rmsd_mean_A": math.nan, "ligand_rmsd_max_A": math.nan}

    rmsd_nm = md.rmsd(traj, traj, 0, atom_indices=lig_atoms)
    rmsd_A = [float(x * 10.0) for x in rmsd_nm]
    return {
        "ligand_rmsd_mean_A": float(statistics.fmean(rmsd_A)),
        "ligand_rmsd_max_A": float(max(rmsd_A)),
    }


def read_state_csv(csv_path: str) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {
        "step": [],
        "potentialEnergy": [],
        "kineticEnergy": [],
        "totalEnergy": [],
        "temperature": [],
    }
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in list(columns.keys()):
                if key in row and row[key] not in (None, ""):
                    columns[key].append(float(row[key]))
    return columns


def parse_ramp_dt_schedule(args: argparse.Namespace) -> List[float]:
    values: List[float] = []
    for token in str(args.ramp_dt_fs).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        values = [0.5, 1.0, float(args.dt_fs)]
    if values[-1] != float(args.dt_fs):
        values.append(float(args.dt_fs))
    return values


def run_stability_simulation(
    label: str,
    system,
    topology,
    positions,
    box_vectors,
    args: argparse.Namespace,
    output_dir: str,
) -> Dict:
    openmm, app, unit, _ = require_openmm()
    sim_dir = ensure_dir(os.path.join(output_dir, label))
    csv_path = os.path.join(sim_dir, "state.csv")
    dcd_path = os.path.join(sim_dir, "traj.dcd")

    sim_system = strip_barostat(system)
    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.dt_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(args.seed)
    platform, properties = select_platform(args.platform)
    platform_label = format_platform_label(platform, properties)
    simulation = app.Simulation(topology, sim_system, integrator, platform, properties)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)

    for parameter_name in ("lam_coul", "lam_vdw"):
        try:
            simulation.context.setParameter(parameter_name, 1.0)
        except Exception:
            pass

    if args.minimize or label == "dexp_surrogate":
        print(f"  ↪ 阶段1 最小化: {label}")
        openmm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=500)

    def _apply_lambda(lam_coul: float | None = None, lam_vdw: float | None = None) -> None:
        if lam_coul is not None:
            try:
                simulation.context.setParameter("lam_coul", float(lam_coul))
            except Exception:
                pass
        if lam_vdw is not None:
            try:
                simulation.context.setParameter("lam_vdw", float(lam_vdw))
            except Exception:
                pass

    simulation.reporters.append(
        app.StateDataReporter(
            csv_path,
            args.report_interval,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            separator=",",
        )
    )
    simulation.reporters.append(
        app.DCDReporter(dcd_path, args.traj_interval, enforcePeriodicBox=False)
    )

    def _set_dt_fs(dt_fs: float) -> None:
        integrator.setStepSize(dt_fs * unit.femtosecond)

    def _run_dynamics_phase(
        steps: int,
        dt_fs: float,
        lam_coul_start: float | None = None,
        lam_coul_end: float | None = None,
        lam_vdw_start: float | None = None,
        lam_vdw_end: float | None = None,
        label_text: str = "",
    ) -> None:
        if steps <= 0:
            return
        _set_dt_fs(dt_fs)
        _apply_lambda(lam_coul=lam_coul_start, lam_vdw=lam_vdw_start)
        has_lambda_ramp = (
            steps > 1
            and any(value is not None for value in (lam_coul_start, lam_coul_end, lam_vdw_start, lam_vdw_end))
        )
        if has_lambda_ramp:
            chunk = max(1, steps // max(1, args.warmup_stages))
            completed = 0
            while completed < steps:
                this_chunk = min(chunk, steps - completed)
                frac = (completed + this_chunk) / steps
                if lam_coul_start is not None and lam_coul_end is not None:
                    lam_coul = lam_coul_start + (lam_coul_end - lam_coul_start) * frac
                else:
                    lam_coul = lam_coul_start
                if lam_vdw_start is not None and lam_vdw_end is not None:
                    lam_vdw = lam_vdw_start + (lam_vdw_end - lam_vdw_start) * frac
                else:
                    lam_vdw = lam_vdw_start
                _apply_lambda(lam_coul=lam_coul, lam_vdw=lam_vdw)
                simulation.step(this_chunk)
                completed += this_chunk
        else:
            simulation.step(steps)
        if label_text:
            print(f"  ↪ {label_text}: {steps} steps @ {dt_fs:.3f} fs")

    if label == "dexp_surrogate":
        ramp_schedule = parse_ramp_dt_schedule(args)
        total_warmup_steps = max(0, int(args.warmup_steps))
        print(
            f"  ↪ 三阶段启动: soft-start={args.softstart_dt_fs:.3f} fs | "
            f"ramp={','.join(f'{dt:.3f}' for dt in ramp_schedule)} fs | "
            f"steps={total_warmup_steps} | backend={platform_label}"
        )
        soft_steps = max(1, total_warmup_steps // 4) if total_warmup_steps > 0 else 0
        ramp_steps_total = max(0, total_warmup_steps - soft_steps)
        vdw_ramp_steps = max(0, int(round(ramp_steps_total * 0.65)))
        coul_ramp_steps = max(0, ramp_steps_total - vdw_ramp_steps)
        vdw_dt_schedule = ramp_schedule
        coul_dt_schedule = [float(ramp_schedule[-1])] if coul_ramp_steps > 0 else []
        per_vdw_ramp = max(1, vdw_ramp_steps // max(1, len(vdw_dt_schedule))) if vdw_ramp_steps > 0 else 0

        # 阶段2: 先只抬起 vdW 核，静电保持关闭，避免点电荷在近接触处先发散
        _run_dynamics_phase(
            soft_steps,
            float(args.softstart_dt_fs),
            lam_coul_start=0.0,
            lam_coul_end=0.0,
            lam_vdw_start=0.05,
            lam_vdw_end=0.25,
            label_text="阶段2 vdW 软启动",
        )

        # 阶段3: 继续把 vdW 拉满，此时 lam_coul 固定为 0
        if vdw_ramp_steps > 0:
            lam_ranges = []
            lam_current = 0.25
            for idx, _dt in enumerate(vdw_dt_schedule):
                lam_next = 1.0 if idx == len(vdw_dt_schedule) - 1 else min(1.0, lam_current + (0.75 / max(1, len(vdw_dt_schedule))))
                lam_ranges.append((lam_current, lam_next))
                lam_current = lam_next
            remaining = vdw_ramp_steps
            for idx, dt_fs in enumerate(vdw_dt_schedule):
                stage_steps = remaining if idx == len(vdw_dt_schedule) - 1 else min(per_vdw_ramp, remaining)
                lam_vdw_start, lam_vdw_end = lam_ranges[idx]
                _run_dynamics_phase(
                    stage_steps,
                    float(dt_fs),
                    lam_coul_start=0.0,
                    lam_coul_end=0.0,
                    lam_vdw_start=lam_vdw_start,
                    lam_vdw_end=lam_vdw_end,
                    label_text=f"阶段3 vdW-ramp[{idx+1}]",
                )
                remaining -= stage_steps

        # 阶段4: 在完整排斥核保护下，再把静电从 0 拉回 1
        if coul_ramp_steps > 0:
            _run_dynamics_phase(
                coul_ramp_steps,
                float(coul_dt_schedule[-1]),
                lam_coul_start=0.0,
                lam_coul_end=1.0,
                lam_vdw_start=1.0,
                lam_vdw_end=1.0,
                label_text="阶段4 Coulomb-ramp",
            )

        _apply_lambda(lam_coul=1.0, lam_vdw=1.0)
        _set_dt_fs(float(args.dt_fs))
    else:
        _set_dt_fs(float(args.dt_fs))

    n_steps = int(round(args.sim_ns * 1000.0 / (args.dt_fs / 1000.0)))
    print(f"[稳定性] {label}: 运行 {n_steps} 步 ({args.sim_ns:.3f} ns) | backend={platform_label}")
    simulation.step(n_steps)

    data = read_state_csv(csv_path)
    summary = {
        "label": label,
        "steps": int(n_steps),
        "dt_fs": float(args.dt_fs),
        "sim_ns": float(args.sim_ns),
        "potential_kjmol": summarize_series(data["potentialEnergy"]),
        "kinetic_kjmol": summarize_series(data["kineticEnergy"]),
        "total_kjmol": summarize_series(data["totalEnergy"]),
        "temperature_K": summarize_series(data["temperature"]),
    }
    summary.update(compute_ligand_rmsd_metrics(dcd_path, args.traj_top, args.ligand))
    with open(os.path.join(sim_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_interaction_separation_schedule(n_states: int) -> List[Tuple[int, str, float, float]]:
    if n_states < 2:
        raise ValueError("schedule 至少需要 2 个状态")
    rows: List[Tuple[int, str, float, float]] = []
    split = max(1, n_states // 2)
    for state in range(n_states):
        if state < split:
            frac = state / max(split - 1, 1)
            lam_coul = 1.0 - frac
            lam_vdw = 1.0
            stage = "decharge"
        else:
            frac = (state - split) / max((n_states - split) - 1, 1)
            lam_coul = 0.0
            lam_vdw = 1.0 - frac
            stage = "vdw"
        rows.append((state, stage, max(0.0, lam_coul), max(0.0, lam_vdw)))
    return rows


def write_schedule_comparison(output_dir: str, n_states: int) -> str:
    out_csv = os.path.join(output_dir, "lambda_schedule_comparison.csv")
    rows: List[Dict] = []
    for state in range(n_states):
        frac = state / max(n_states - 1, 1)
        lam = 1.0 - frac
        rows.append(
            {
                "schedule": "original_linear",
                "state": state,
                "stage": "coupled",
                "lambda_coul": lam,
                "lambda_vdw": lam,
            }
        )
    for state, stage, lam_coul, lam_vdw in build_interaction_separation_schedule(n_states):
        rows.append(
            {
                "schedule": "interaction_separation",
                "state": state,
                "stage": stage,
                "lambda_coul": lam_coul,
                "lambda_vdw": lam_vdw,
            }
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def write_comparison_report(output_dir: str, original_summary: Dict, dexp_summary: Dict, fitted_params: Dict, schedule_csv: str) -> str:
    report_path = os.path.join(output_dir, "comparison_report.md")
    lines = [
        "# DEXP Stability Comparison",
        "",
        "## Fitting",
        f"- Fit frames requested: {fitted_params.get('fit_frames_requested')}",
        f"- Fit frames used: {fitted_params.get('fit_frames_used')}",
        f"- Fitting success: {fitted_params.get('fitting_success')}",
        f"- Suspicious fit: {fitted_params.get('suspicious_fit')}",
        f"- Boundary hits: {', '.join(fitted_params.get('boundary_hits', [])) or 'none'}",
        f"- alpha_vdw: {fitted_params.get('alpha_vdw')}",
        f"- beta_vdw: {fitted_params.get('beta_vdw')}",
        f"- r0_vdw: {fitted_params.get('r0_vdw')}",
        f"- offset_c0 (force-disabled): {fitted_params.get('offset_c0')}",
        f"- offset_c1 (force-disabled): {fitted_params.get('offset_c1')}",
        f"- diagnostic_global_mu: {fitted_params.get('diagnostic_global_mu')}",
        f"- diagnostic_fit_c0: {fitted_params.get('diagnostic_fit_c0')}",
        f"- diagnostic_contact_mu: {fitted_params.get('diagnostic_contact_mu')}",
        f"- diagnostic_contact_slope: {fitted_params.get('diagnostic_contact_slope')}",
        "",
        "## Stability",
        f"- Original mean temperature (K): {original_summary['temperature_K']['mean']:.3f}",
        f"- DEXP mean temperature (K): {dexp_summary['temperature_K']['mean']:.3f}",
        f"- Original ligand RMSD mean (A): {original_summary.get('ligand_rmsd_mean_A', math.nan):.3f}",
        f"- DEXP ligand RMSD mean (A): {dexp_summary.get('ligand_rmsd_mean_A', math.nan):.3f}",
        f"- Original total energy std (kJ/mol): {original_summary['total_kjmol']['std']:.3f}",
        f"- DEXP total energy std (kJ/mol): {dexp_summary['total_kjmol']['std']:.3f}",
        "",
        "## Lambda Schedules",
        f"- CSV: {schedule_csv}",
        "- `original_linear`: Coulomb 与 VDW 同步线性缩放。",
        "- `interaction_separation`: 先去电荷，再去 VDW。",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def main() -> int:
    args = parse_args()
    args.traj = ensure_file(args.traj, "预平衡轨迹")
    args.traj_top = ensure_file(args.traj_top, "轨迹拓扑")
    args.gmx_top = ensure_file(args.gmx_top, "GROMACS 拓扑")
    args.system_xml = ensure_file(args.system_xml, "原始 system XML")
    args.ligand_indices = ensure_file(args.ligand_indices, "配体索引 JSON")
    output_dir = ensure_dir(args.output_dir)

    fitted_params = fit_dexp_from_tail_frames(args, output_dir)

    print("[2/4] 载入原始系统与最后一帧坐标")
    system, topology = load_cached_system(args.system_xml, args.traj_top)
    _, positions, box_vectors = load_last_frame_positions(args.traj, args.traj_top)
    ligand_indices = load_ligand_indices(args.ligand_indices)
    env_indices = [
        idx for idx in range(system.getNumParticles())
        if idx not in set(ligand_indices)
    ]

    print("[3/4] 构建 DEXP surrogate system 并执行 1 ns 稳定性测试")
    symbols = load_abfe_symbols()
    SurrogateSystemBuilder = symbols["SurrogateSystemBuilder"]
    surrogate_builder = SurrogateSystemBuilder(fitted_params)
    dexp_system = surrogate_builder.build_surrogate_system(
        original_system=system,
        ligand_indices=ligand_indices,
        environment_indices=env_indices,
        lambda_names=("lam_coul", "lam_vdw"),
        force_group=1,
        reference_positions=positions,
        box_vectors=box_vectors,
    )
    dexp_summary = run_stability_simulation(
        label="dexp_surrogate",
        system=dexp_system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        args=args,
        output_dir=output_dir,
    )

    print("[4/4] 执行原始势能 1 ns baseline，并导出 lambda schedule 对比")
    original_summary = run_stability_simulation(
        label="original_baseline",
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        args=args,
        output_dir=output_dir,
    )
    schedule_csv = write_schedule_comparison(output_dir, args.schedule_states)
    report_path = write_comparison_report(
        output_dir,
        original_summary=original_summary,
        dexp_summary=dexp_summary,
        fitted_params=fitted_params,
        schedule_csv=schedule_csv,
    )

    comparison_json = os.path.join(output_dir, "comparison_summary.json")
    with open(comparison_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fitted_params": fitted_params,
                "dexp_surrogate": dexp_summary,
                "original_baseline": original_summary,
                "lambda_schedule_csv": schedule_csv,
                "report_md": report_path,
            },
            handle,
            indent=2,
        )

    print("实验完成。")
    print(f"参数文件: {os.path.join(output_dir, 'dexp_fitted_params.json')}")
    print(f"对比汇总: {comparison_json}")
    print(f"对比报告: {report_path}")
    print(f"Schedule 对比: {schedule_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

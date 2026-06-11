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
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import math
import os
import re
import statistics
import struct
import zlib
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
            _select_env_indices_from_mdtraj_frame,
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
        "_select_env_indices_from_mdtraj_frame": _select_env_indices_from_mdtraj_frame,
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
    parser.add_argument("--ml-model", default="mace-off24-medium", help="OpenMM-ML 预训练模型名，例如 mace-off24-medium")
    parser.add_argument("--fit-frames", type=int, default=500, help="从末段时间窗中最多取多少帧参与拟合")
    parser.add_argument("--fit-last-ns", type=float, default=5.0, help="只使用轨迹最后多少 ns 做拟合")
    parser.add_argument("--fit-env-radius", type=float, default=0.50, help="环境筛选半径 (nm)")
    parser.add_argument("--fit-env-max-atoms", type=int, default=0, help="OpenMM-ML 环境原子上限；<=0 表示关闭最近邻裁剪")
    parser.add_argument("--fit-gpu-workers", type=int, default=1, help="OpenMM-ML worker 数；默认 1，按单 context 滚动标注以避免 CUDA 句柄分配失败")
    parser.add_argument("--fit-r-min", type=float, default=0.20, help="拟合距离下限 (nm)")
    parser.add_argument("--fit-r-max", type=float, default=0.45, help="拟合距离上限 (nm)")
    parser.add_argument("--fit-mm-ref-cutoff", type=float, default=0.85, help="MM 参考 L-E cutoff (nm)，独立于 DEXP 拟合距离窗")
    parser.add_argument("--fit-mm-ref-switch", type=float, default=0.70, help="MM 参考 L-E switching distance (nm)")
    parser.add_argument(
        "--fit-target-mode",
        choices=("ml_minus_mm_total", "qmmm_residual", "gaussian_replacement_residual", "ml_minus_mm_coul"),
        default="ml_minus_mm_total",
        help="DEXP 拟合目标；推荐 ml_minus_mm_total。",
    )
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
    parser.add_argument("--analysis-max-frames", type=int, default=200, help="后处理分析最多读取多少帧")
    parser.add_argument("--lambda-scan-points", type=int, default=11, help="lambda 单点扫描状态数")
    parser.add_argument("--rdf-r-max", type=float, default=1.2, help="L-E RDF 最大半径 (nm)")
    parser.add_argument("--rdf-bin-width", type=float, default=0.01, help="L-E RDF bin 宽度 (nm)")
    parser.add_argument("--pmf-bin-width", type=float, default=0.01, help="1D PMF 的 min-distance bin 宽度 (nm)")
    parser.add_argument("--analysis-r-min", type=float, default=0.20, help="后处理重点关注距离下限 (nm)，默认 0.20 = 2A")
    parser.add_argument("--analysis-r-max", type=float, default=0.65, help="后处理重点关注距离上限 (nm)，默认 0.65 = 6.5A")
    parser.add_argument("--lambda-window-values", default="1.0,0.75,0.5,0.25,0.0", help="后处理固定 lambda 窗口，逗号分隔")
    parser.add_argument("--lambda-window-ns", type=float, default=0.10, help="每个固定 lambda 窗口的短程重跑时长 (ns)")
    parser.add_argument("--postprocess-only", action="store_true", help="跳过拟合与动力学，只基于现有 output-dir 结果重跑后处理")
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
        "r0_vdw": (0.28, 0.40),
        "A_fit": (1.0e-5, 5.0),
        "B_fit": (1.0e-5, 5.0),
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


def validate_fit_for_dynamics(fitted_params: Dict) -> None:
    issues: List[str] = []
    if fitted_params.get("suspicious_fit"):
        issues.append(f"boundary_hits={', '.join(fitted_params.get('boundary_hits', []))}")
    fit_frames_used = fitted_params.get("fit_frames_used")
    fit_frames_total = fitted_params.get("fit_frames_total")
    if fit_frames_used is not None and fit_frames_total:
        if int(fit_frames_used) < max(50, int(0.25 * int(fit_frames_total))):
            issues.append(f"fit_frames_used={fit_frames_used}/{fit_frames_total}")
    final_cost = fitted_params.get("final_cost")
    if final_cost is not None and float(final_cost) > 1000.0:
        issues.append(f"final_cost={float(final_cost):.3f}")
    if issues:
        raise RuntimeError(
            "DEXP 拟合结果当前不适合直接做稳定性动力学，已阻止运行以避免 NaN。"
            f" 触发条件: {'; '.join(issues)}"
        )


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
    centered_values = [float(v) - center for v in delta_e_values]
    centered_stats = summarize_delta_e(centered_values)
    if centered_stats["std"] > 200.0:
        polluted = True
        threshold = 200.0
        reason = "centered_std_gt_200"
    if centered_stats["std"] > 350.0:
        polluted = True
        threshold = 100.0
        reason = "severe_centered_pollution"
    if np.isfinite(centered_stats["std"]):
        threshold = max(50.0, min(threshold, 4.0 * float(centered_stats["std"]) + 20.0))
    return threshold, {"polluted": polluted, "reason": reason, "stats": stats, "centered_stats": centered_stats, "center": center}


def fit_dexp_from_tail_frames(args: argparse.Namespace, output_dir: str) -> Dict:
    ensure_dir(output_dir)
    md = require_module("mdtraj")
    symbols = load_abfe_symbols()
    NumpyEncoder = symbols["NumpyEncoder"]
    Orbv3DEXPFittingPipeline = symbols["Orbv3DEXPFittingPipeline"]
    Orbv3SurrogateFitter = symbols["Orbv3SurrogateFitter"]
    select_env_indices = symbols["_select_env_indices_from_mdtraj_frame"]
    openmm, _, unit, _ = require_openmm()
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
    env_search_radius = float(args.fit_env_radius)
    env_max_atoms = int(args.fit_env_max_atoms) if int(args.fit_env_max_atoms) > 0 else None
    env_idx = select_env_indices(
        ref_frame, lig_idx, env_search_radius, max_env_atoms=env_max_atoms
    )
    if len(env_idx) == 0:
        raise RuntimeError("未找到配体附近环境原子，请增大 --fit-env-radius")
    if env_max_atoms is not None:
        print(f"    OpenMM-ML 环境原子上限: {env_max_atoms} | 实际选中: {len(env_idx)}")

    all_nums = np.array([a.element.atomic_number for a in fit_traj.top.atoms], dtype=int)
    pipeline = Orbv3DEXPFittingPipeline(model_name=args.ml_model, device=args.device)
    label_mode = getattr(pipeline, "label_mode", "orbv3_interaction")
    fit_target_mode = str(args.fit_target_mode)
    use_gaussian_replacement = fit_target_mode == "gaussian_replacement_residual"
    use_qmmm_total = fit_target_mode in ("qmmm_residual", "ml_minus_mm_total")

    mm_contexts = build_mm_le_contexts_from_system_xml(
        args.system_xml,
        ligand_indices=lig_idx.tolist(),
        environment_indices=env_idx.tolist(),
        cutoff_nm=float(args.fit_mm_ref_cutoff),
        switching_nm=float(args.fit_mm_ref_switch),
    )

    fit_log_rows: List[Dict] = []
    raw_delta_e_values: List[float] = []
    raw_gauss_coul_values: List[float] = []
    raw_delta_vs_mm_total_values: List[float] = []
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
                frame_indices_cached == [int(x) for x in fit_indices]
                and env_idx_cached == [int(x) for x in env_idx]
                and lig_idx_cached == [int(x) for x in lig_idx]
                and str(cache_meta.get("ml_model", "")) == str(args.ml_model)
                and str(cache_meta.get("label_mode", "")) == str(label_mode)
                and str(cache_meta.get("fit_target_mode", "")) == fit_target_mode
                and abs(float(cache_meta.get("env_search_radius_nm", -1.0)) - float(env_search_radius)) < 1.0e-8
                and cache_meta.get("env_max_atoms", None) == (int(env_max_atoms) if env_max_atoms is not None else None)
            )
            if reuse_labels:
                for row in read_csv_rows(fit_log_path):
                    cached_rows_by_frame[int(row["frame_index"])] = row
                print(f"    复用已有能量标注缓存: {fit_log_path}")
            else:
                print("    已检测到旧缓存，但当前 frame/env 选择已变化，回退为重新标注。")
        except Exception:
            reuse_labels = False

    gpu_workers = 1
    worker_pipelines: List[Orbv3DEXPFittingPipeline] = []
    if not reuse_labels and str(args.device).lower() == "cuda":
        gpu_workers = 1
        first_pos_nm = fit_xyz[0].copy()
        print(f"    OpenMM-ML 预建 GPU worker: {gpu_workers}")
        for wid in range(gpu_workers):
            worker = Orbv3DEXPFittingPipeline(model_name=args.ml_model, device=args.device)
            worker._cache_contexts = True
            worker._preflight_orb_backend(first_pos_nm, lig_idx, env_idx, all_nums)
            worker_pipelines.append(worker)

    if not reuse_labels and gpu_workers > 1:
        orb_energy_by_local_idx: Dict[int, float] = {}

        def _compute_orb_batch_with_prebuilt_pipeline(worker_id: int, batch_local_indices: List[int]) -> List[Tuple[int, float]]:
            worker = worker_pipelines[worker_id]
            results: List[Tuple[int, float]] = []
            for local_idx in batch_local_indices:
                pos_nm = fit_xyz[local_idx].copy()
                e_orb_int = worker._compute_orb_decomposition(pos_nm, lig_idx, env_idx, all_nums)
                results.append((local_idx, float(e_orb_int)))
            return results

        work_batches: List[List[int]] = [[] for _ in range(gpu_workers)]
        for idx, local_idx in enumerate(range(len(fit_indices))):
            work_batches[idx % gpu_workers].append(local_idx)
        work_batches = [batch for batch in work_batches if batch]

        with ThreadPoolExecutor(max_workers=len(work_batches)) as executor:
            future_map = {
                executor.submit(_compute_orb_batch_with_prebuilt_pipeline, worker_id, batch): worker_id
                for worker_id, batch in enumerate(work_batches)
            }
            completed = 0
            for future in as_completed(future_map):
                batch_results = future.result()
                for local_idx, e_orb_int in batch_results:
                    orb_energy_by_local_idx[local_idx] = e_orb_int
                    completed += 1
                print(f"    ORB 已完成 {completed}/{len(fit_indices)} 帧")
    else:
        orb_energy_by_local_idx = {}

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
            e_gauss_coul = float(cached_row.get("e_gauss_coul_kjmol", "0.0"))
            e_mm_coul = float(cached_row["e_mm_coul_kjmol"])
            e_mm_vdw = float(cached_row.get("e_mm_vdw_kjmol", "0.0"))
            delta_gauss_replacement = float(
                cached_row.get(
                    "delta_gaussian_replacement_kjmol",
                    float(e_orb_int - e_gauss_coul),
                )
            )
            delta_noncoul = float(e_orb_int - e_mm_coul)
            delta_vs_mm_total = float(
                cached_row.get(
                    "delta_vs_mm_total_kjmol",
                    float(e_orb_int - e_mm_coul - e_mm_vdw),
                )
            )
            delta_fit = float(
                cached_row.get(
                    "delta_fit_kjmol",
                    delta_gauss_replacement if use_gaussian_replacement else (
                        delta_vs_mm_total if use_qmmm_total else cached_row.get("delta_e_res_kjmol", cached_row["delta_e_kjmol"])
                    ),
                )
            )
        else:
            if local_idx in orb_energy_by_local_idx:
                e_orb_int = float(orb_energy_by_local_idx[local_idx])
            else:
                e_orb_int = pipeline._compute_orb_decomposition(pos_nm, lig_idx, env_idx, all_nums)
            e_gauss_coul = 0.0
            e_mm_coul = 0.0
            e_mm_vdw = 0.0
            for label, ctx in mm_contexts.items():
                if fit_box is not None:
                    ctx.setPeriodicBoxVectors(
                        *[openmm.Vec3(float(vec[0]), float(vec[1]), float(vec[2])) for vec in fit_box[local_idx]]
                    )
                ctx.setPositions(pos_nm * unit.nanometer)
                energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                    unit.kilojoules_per_mole
                )
                if label == "gauss_coul":
                    e_gauss_coul = energy
                elif label == "coul":
                    e_mm_coul = energy
                elif label == "vdw":
                    e_mm_vdw = energy
            delta_gauss_replacement = float(e_orb_int - e_gauss_coul)
            delta_noncoul = float(e_orb_int - e_mm_coul)
            delta_vs_mm_total = float(e_orb_int - e_mm_coul - e_mm_vdw)
            delta_fit = float(
                delta_gauss_replacement if use_gaussian_replacement else (
                    delta_vs_mm_total if use_qmmm_total else delta_noncoul
                )
            )

        raw_orb_values.append(float(e_orb_int))
        raw_gauss_coul_values.append(float(e_gauss_coul))
        raw_mm_coul_values.append(float(e_mm_coul))
        raw_mm_vdw_values.append(float(e_mm_vdw))
        if np.isfinite(delta_fit):
            raw_delta_e_values.append(delta_fit)
        if np.isfinite(delta_vs_mm_total):
            raw_delta_vs_mm_total_values.append(delta_vs_mm_total)

        fit_log_rows.append(
            {
                "frame_index": frame_id,
                "time_ps": float(fit_time[local_idx]),
                "e_orb_int_kjmol": float(e_orb_int),
                "e_gauss_coul_kjmol": float(e_gauss_coul),
                "e_mm_coul_kjmol": float(e_mm_coul),
                "e_mm_vdw_kjmol": float(e_mm_vdw),
                "e_mm_region_kjmol": float(e_mm_coul + e_mm_vdw),
                "e_qm_region_kjmol": float(e_orb_int),
                "delta_e_kjmol": float(delta_fit),
                "delta_e_res_kjmol": float(delta_noncoul),
                "delta_fit_kjmol": float(delta_fit),
                "delta_gaussian_replacement_kjmol": float(delta_gauss_replacement),
                "delta_vs_mm_total_kjmol": float(delta_vs_mm_total),
                "delta_qmmm_kjmol": float(delta_vs_mm_total),
                "n_env_pairs": int(len(candidate_dists)),
                "n_valid_pairs": int(len(valid_dists)),
                "used_for_fit": 0,
            }
        )
        if (local_idx + 1) % 50 == 0 or local_idx == len(fit_indices) - 1:
            print(f"    已处理 {local_idx + 1}/{len(fit_indices)} 帧")

    for worker in worker_pipelines:
        try:
            worker._clear_orb_context_cache()
        except Exception:
            pass
    try:
        pipeline._clear_orb_context_cache()
    except Exception:
        pass

    delta_threshold, delta_diag = choose_delta_e_threshold(raw_delta_e_values)
    if use_qmmm_total:
        delta_label = "ΔE_qmmm(region)"
        mean_label = "mean(qm-mm_region)"
    elif use_gaussian_replacement:
        delta_label = "ΔE_replace(region)"
        mean_label = "mean(qm-gauss_coul)"
    else:
        delta_label = "ΔE_res" if label_mode == "orbv3_interaction" else "ΔE_mace"
        mean_label = "mean(orb-coul)" if label_mode == "orbv3_interaction" else "mean(mace-coul)"
    ml_energy_label = "E_orb_int" if label_mode == "orbv3_interaction" else "E_mace_int"
    print(
        f"    {delta_label} 诊断: "
        f"mean={delta_diag['stats']['mean']:.2f} kJ/mol | "
        f"std={delta_diag['stats']['std']:.2f} | "
        f"centered-threshold={delta_threshold:.1f} | "
        f"polluted={delta_diag['polluted']}"
    )
    orb_stats = summarize_delta_e(raw_orb_values)
    gauss_coul_stats = summarize_delta_e(raw_gauss_coul_values)
    mm_coul_stats = summarize_delta_e(raw_mm_coul_values)
    mm_vdw_stats = summarize_delta_e(raw_mm_vdw_values)
    mm_total_delta_stats = summarize_delta_e(raw_delta_vs_mm_total_values)
    print(
        "    能量分量: "
        f"{ml_energy_label} mean={orb_stats['mean']:.2f} | "
        f"E_gauss_coul mean={gauss_coul_stats['mean']:.2f} | "
        f"E_mm_coul mean={mm_coul_stats['mean']:.2f} | "
        f"E_mm_vdw mean={mm_vdw_stats['mean']:.2f} | "
        f"{mean_label}={delta_diag['stats']['mean']:.2f} | "
        f"mean(ml-mm_total)={mm_total_delta_stats['mean']:.2f}"
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
        row["qm_mm_offset_kjmol"] = float(delta_diag["center"])
        row["delta_qmmm_centered_kjmol"] = float(centered_delta)

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
    fitted_params["env_max_atoms"] = int(env_max_atoms) if env_max_atoms is not None else None
    fitted_params["fit_region_nm"] = [float(args.fit_r_min), float(args.fit_r_max)]
    fitted_params["traj_total_frames"] = int(len(traj))
    fitted_params["ml_model"] = str(args.ml_model)
    fitted_params["label_mode"] = str(label_mode)
    fitted_params["fit_target_mode"] = fit_target_mode
    fitted_params["qm_reference_region_definition"] = "ligand + environment pocket"
    fitted_params["qm_mm_offset_kjmol"] = float(delta_diag["center"])
    fitted_params["delta_e_filter_threshold_kjmol"] = float(delta_threshold)
    fitted_params["delta_e_res_filter_threshold_kjmol"] = float(delta_threshold)
    fitted_params["delta_e_polluted"] = bool(delta_diag["polluted"])
    fitted_params["delta_e_pollution_reason"] = str(delta_diag["reason"])
    fitted_params["delta_e_mean_kjmol"] = float(delta_diag["stats"]["mean"])
    fitted_params["delta_e_std_kjmol"] = float(delta_diag["stats"]["std"])
    fitted_params["delta_e_mean_abs_kjmol"] = float(delta_diag["stats"]["mean_abs"])
    fitted_params["delta_e_res_mean_kjmol"] = float(delta_diag["stats"]["mean"])
    fitted_params["delta_e_res_std_kjmol"] = float(delta_diag["stats"]["std"])
    fitted_params["delta_e_weighted_center_kjmol"] = float(fitted_params.get("diagnostic_weighted_center", math.nan))
    fitted_params["delta_e_centered_std_kjmol"] = float(fitted_params.get("diagnostic_centered_std", math.nan))
    fitted_params["e_gauss_coul_mean_kjmol"] = float(gauss_coul_stats["mean"])
    fitted_params["e_gauss_coul_std_kjmol"] = float(gauss_coul_stats["std"])
    fitted_params["e_orb_int_mean_kjmol"] = float(orb_stats["mean"])
    fitted_params["e_orb_int_std_kjmol"] = float(orb_stats["std"])
    fitted_params["e_mm_coul_mean_kjmol"] = float(mm_coul_stats["mean"])
    fitted_params["e_mm_coul_std_kjmol"] = float(mm_coul_stats["std"])
    fitted_params["e_mm_vdw_mean_kjmol"] = float(mm_vdw_stats["mean"])
    fitted_params["e_mm_vdw_std_kjmol"] = float(mm_vdw_stats["std"])
    fitted_params["delta_vs_mm_total_mean_kjmol"] = float(mm_total_delta_stats["mean"])
    fitted_params["delta_vs_mm_total_std_kjmol"] = float(mm_total_delta_stats["std"])
    if use_qmmm_total:
        fitted_params["fit_target_definition"] = "delta_fit = (E_qm_region - E_mm_region) - <E_qm_region - E_mm_region>"
    elif use_gaussian_replacement:
        fitted_params["fit_target_definition"] = "delta_fit = (E_qm_region - E_gauss_coul_region) - <E_qm_region - E_gauss_coul_region>"
    else:
        fitted_params["fit_target_definition"] = "delta_fit = E_ml_interaction - E_mm_coul"
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
                "ml_model": str(args.ml_model),
                "label_mode": str(label_mode),
                "fit_target_mode": fit_target_mode,
                "env_search_radius_nm": float(env_search_radius),
                "env_max_atoms": int(env_max_atoms) if env_max_atoms is not None else None,
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
    cutoff_nm: float = 0.85,
    switching_nm: float = 0.70,
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
    lig_set = {int(idx) for idx in ligand_indices}
    sigma_gauss_nm = 0.10
    gamma_eff = 1.0 / max(math.sqrt(2.0) * sigma_gauss_nm, 1.0e-6)
    force_defs = {
        "gauss_coul": (
            f"active * 138.935456*q1*q2*erf({gamma_eff}*r_safe)/r_safe; "
            "active = abs(type1-type2); "
            "r_safe = max(r, 1e-6)",
            ("q", "type"),
        ),
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
                elif param_name == "type":
                    payload.append(1.0 if atom_idx in lig_set else 0.0)
                elif param_name == "sigma":
                    payload.append(sigma.value_in_unit(unit.nanometer))
                elif param_name == "epsilon":
                    payload.append(epsilon.value_in_unit(unit.kilojoule_per_mole))
            le_force.addParticle(payload)
        if label != "gauss_coul":
            le_force.addInteractionGroup(
                [int(idx) for idx in ligand_indices],
                [int(idx) for idx in environment_indices],
            )
        le_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        le_force.setCutoffDistance(cutoff_nm * unit.nanometer)
        le_force.setUseSwitchingFunction(False)
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
    alias_map = {
        "step": "step",
        "#step": "step",
        "potentialenergy": "potentialEnergy",
        "potential energy (kj/mole)": "potentialEnergy",
        "kineticenergy": "kineticEnergy",
        "kinetic energy (kj/mole)": "kineticEnergy",
        "totalenergy": "totalEnergy",
        "total energy (kj/mole)": "totalEnergy",
        "temperature": "temperature",
        "temperature (k)": "temperature",
    }

    def _normalize_header(text: str) -> str:
        cleaned = str(text).strip().strip('"').strip("'")
        cleaned = cleaned.lstrip("#").strip()
        return cleaned.lower()

    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            normalized_names = [_normalize_header(name) for name in reader.fieldnames]
            header_lookup = {
                raw_name: alias_map.get(norm_name)
                for raw_name, norm_name in zip(reader.fieldnames, normalized_names)
            }
        else:
            header_lookup = {}
        for row in reader:
            for raw_name, value in row.items():
                canonical = header_lookup.get(raw_name)
                if canonical is None or value in (None, ""):
                    continue
                columns[canonical].append(float(value))
    return columns


def select_analysis_frame_indices(n_frames: int, max_frames: int) -> List[int]:
    if n_frames <= 0:
        return []
    if max_frames <= 0 or n_frames <= max_frames:
        return list(range(n_frames))
    return [int(idx) for idx in np.unique(np.linspace(0, n_frames - 1, max_frames, dtype=int)).tolist()]


def load_analysis_traj(traj_path: str, top_path: str, max_frames: int):
    md = require_module("mdtraj")
    traj = md.load(traj_path, top=top_path)
    if len(traj) == 0:
        raise RuntimeError(f"轨迹为空，无法分析: {traj_path}")
    frame_indices = select_analysis_frame_indices(len(traj), max_frames)
    sliced = traj[frame_indices]
    if sliced.unitcell_vectors is not None:
        sliced = sliced.image_molecules(inplace=False)
    return sliced, frame_indices


def get_ligand_env_heavy_indices(traj_topology, ligand_resname: str) -> Tuple[np.ndarray, np.ndarray]:
    lig_heavy = np.array(
        traj_topology.select(f"resname {ligand_resname} and not element H"),
        dtype=int,
    )
    if len(lig_heavy) == 0:
        lig_heavy = np.array(traj_topology.select(f"resname {ligand_resname}"), dtype=int)
    if len(lig_heavy) == 0:
        raise ValueError(f"未在拓扑中找到配体 `{ligand_resname}` 的原子")

    env_heavy = np.array(
        traj_topology.select(f"not resname {ligand_resname} and not element H"),
        dtype=int,
    )
    if len(env_heavy) == 0:
        env_heavy = np.array(traj_topology.select(f"not resname {ligand_resname}"), dtype=int)
    if len(env_heavy) == 0:
        raise ValueError("未在拓扑中找到环境原子")
    return lig_heavy, env_heavy


def compute_pairwise_distances_nm(
    pos_nm: np.ndarray,
    lig_idx: np.ndarray,
    env_idx: np.ndarray,
    box_vecs_nm: np.ndarray | None,
) -> np.ndarray:
    delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
    if box_vecs_nm is not None:
        box_lens = np.linalg.norm(np.asarray(box_vecs_nm, dtype=np.float64), axis=1)
        delta -= box_lens * np.round(delta / box_lens)
    return np.linalg.norm(delta, axis=-1)


def compute_min_distance_series_nm(traj, lig_idx: np.ndarray, env_idx: np.ndarray) -> List[float]:
    out: List[float] = []
    box = np.asarray(traj.unitcell_vectors, dtype=np.float64) if traj.unitcell_vectors is not None else None
    for frame_idx in range(len(traj)):
        box_vecs = box[frame_idx] if box is not None else None
        dists = compute_pairwise_distances_nm(
            np.asarray(traj.xyz[frame_idx], dtype=np.float64),
            lig_idx,
            env_idx,
            box_vecs,
        )
        out.append(float(np.min(dists)))
    return out


def summarize_series_with_percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p05": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    arr = np.asarray(values, dtype=float)
    base = summarize_series([float(x) for x in arr.tolist()])
    return {
        "count": int(arr.size),
        "mean": float(base["mean"]),
        "std": float(base["std"]),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "max": float(np.max(arr)),
    }


def write_rows_csv(path: str, rows: List[Dict]) -> str:
    if not rows:
        raise ValueError(f"无数据可写入: {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def get_matplotlib_pyplot():
    import importlib

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg", force=True)
    return importlib.import_module("matplotlib.pyplot")


def parse_lambda_window_values(args: argparse.Namespace) -> List[float]:
    values: List[float] = []
    for token in str(args.lambda_window_values).split(","):
        token = token.strip()
        if not token:
            continue
        lam = float(token)
        values.append(min(max(lam, 0.0), 1.0))
    if not values:
        values = [1.0, 0.75, 0.5, 0.25, 0.0]
    values = sorted({round(v, 6) for v in values}, reverse=True)
    return [float(v) for v in values]


def build_context_for_system(system, args: argparse.Namespace):
    openmm, _, _, _ = require_openmm()
    integrator = openmm.VerletIntegrator(0.001)
    platform, properties = select_platform(args.platform)
    return openmm.Context(system, integrator, platform, properties)


def evaluate_context(
    context,
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray | None = None,
    lam_coul: float | None = None,
    lam_vdw: float | None = None,
    include_forces: bool = False,
) -> Dict[str, float]:
    openmm, _, unit, _ = require_openmm()
    if box_vectors_nm is not None:
        context.setPeriodicBoxVectors(
            *[
                openmm.Vec3(float(vec[0]), float(vec[1]), float(vec[2]))
                for vec in np.asarray(box_vectors_nm, dtype=np.float64)
            ]
        )
    context.setPositions(np.asarray(positions_nm, dtype=np.float64) * unit.nanometer)
    if lam_coul is not None:
        try:
            context.setParameter("lam_coul", float(lam_coul))
        except Exception:
            pass
    if lam_vdw is not None:
        try:
            context.setParameter("lam_vdw", float(lam_vdw))
        except Exception:
            pass

    state = context.getState(getEnergy=True, getForces=include_forces)
    result = {
        "potential_kjmol": float(
            state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        )
    }
    if include_forces:
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        norms = np.linalg.norm(np.asarray(forces, dtype=np.float64), axis=1)
        result["max_force_kjmol_per_nm"] = float(np.max(norms))
        result["mean_force_kjmol_per_nm"] = float(np.mean(norms))
    return result


def run_lambda_single_point_scan(
    args: argparse.Namespace,
    output_dir: str,
    dexp_system,
) -> Dict:
    print("[5/8] 执行 lambda=1→0 单点扫描")
    traj, sampled_indices = load_analysis_traj(args.traj, args.traj_top, args.analysis_max_frames)
    context = build_context_for_system(dexp_system, args)
    lambda_values = np.linspace(1.0, 0.0, max(2, int(args.lambda_scan_points)))
    box = np.asarray(traj.unitcell_vectors, dtype=np.float64) if traj.unitcell_vectors is not None else None
    rows: List[Dict] = []
    per_lambda_energy: Dict[float, List[float]] = {}
    per_lambda_force: Dict[float, List[float]] = {}

    for local_idx, frame_idx in enumerate(sampled_indices):
        pos_nm = np.asarray(traj.xyz[local_idx], dtype=np.float64)
        box_vecs = box[local_idx] if box is not None else None
        prev_energy = None
        for lam in lambda_values:
            metrics = evaluate_context(
                context,
                positions_nm=pos_nm,
                box_vectors_nm=box_vecs,
                lam_coul=float(lam),
                lam_vdw=float(lam),
                include_forces=True,
            )
            jump = math.nan if prev_energy is None else float(metrics["potential_kjmol"] - prev_energy)
            prev_energy = float(metrics["potential_kjmol"])
            row = {
                "frame_index": int(frame_idx),
                "lambda_value": float(lam),
                "potential_kjmol": float(metrics["potential_kjmol"]),
                "delta_from_prev_lambda_kjmol": jump,
                "max_force_kjmol_per_nm": float(metrics["max_force_kjmol_per_nm"]),
                "mean_force_kjmol_per_nm": float(metrics["mean_force_kjmol_per_nm"]),
                "is_finite": int(
                    np.isfinite(metrics["potential_kjmol"])
                    and np.isfinite(metrics["max_force_kjmol_per_nm"])
                ),
            }
            rows.append(row)
            lam_key = float(lam)
            per_lambda_energy.setdefault(lam_key, []).append(float(metrics["potential_kjmol"]))
            per_lambda_force.setdefault(lam_key, []).append(float(metrics["max_force_kjmol_per_nm"]))

    csv_path = write_rows_csv(os.path.join(output_dir, "lambda_single_point_scan.csv"), rows)
    summary = {
        "scan_csv": csv_path,
        "n_frames": int(len(sampled_indices)),
        "n_lambda": int(len(lambda_values)),
        "all_finite": bool(all(int(row["is_finite"]) for row in rows)),
        "max_abs_energy_jump_kjmol": float(
            max(
                (abs(float(row["delta_from_prev_lambda_kjmol"])) for row in rows if np.isfinite(row["delta_from_prev_lambda_kjmol"])),
                default=math.nan,
            )
        ),
        "max_force_kjmol_per_nm": float(
            max((max(vals) for vals in per_lambda_force.values()), default=math.nan)
        ),
        "per_lambda": [
            {
                "lambda_value": float(lam),
                "potential_mean_kjmol": float(statistics.fmean(per_lambda_energy[lam])),
                "potential_std_kjmol": float(statistics.stdev(per_lambda_energy[lam])) if len(per_lambda_energy[lam]) > 1 else 0.0,
                "max_force_mean_kjmol_per_nm": float(statistics.fmean(per_lambda_force[lam])),
                "max_force_max_kjmol_per_nm": float(max(per_lambda_force[lam])),
            }
            for lam in sorted(per_lambda_energy.keys(), reverse=True)
        ],
    }
    return summary


def compute_rdf(traj, lig_idx: np.ndarray, env_idx: np.ndarray, r_max_nm: float, bin_width_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    if bin_width_nm <= 0.0:
        raise ValueError("rdf bin 宽度必须 > 0")
    n_bins = max(1, int(math.ceil(r_max_nm / bin_width_nm)))
    edges = np.linspace(0.0, r_max_nm, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.float64)
    shell_factor = 4.0 * math.pi / 3.0
    rho_sum = 0.0
    n_frames_used = 0
    box = np.asarray(traj.unitcell_vectors, dtype=np.float64) if traj.unitcell_vectors is not None else None

    for frame_idx in range(len(traj)):
        box_vecs = box[frame_idx] if box is not None else None
        dists = compute_pairwise_distances_nm(
            np.asarray(traj.xyz[frame_idx], dtype=np.float64),
            lig_idx,
            env_idx,
            box_vecs,
        ).ravel()
        dists = dists[np.isfinite(dists)]
        dists = dists[dists <= r_max_nm]
        hist, _ = np.histogram(dists, bins=edges)
        counts += hist
        if box_vecs is not None:
            volume = abs(float(np.linalg.det(np.asarray(box_vecs, dtype=np.float64))))
            if volume > 1.0e-8:
                rho_sum += float(len(env_idx)) / volume
                n_frames_used += 1

    radii = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = shell_factor * (edges[1:] ** 3 - edges[:-1] ** 3)
    avg_density = rho_sum / max(n_frames_used, 1)
    denom = max(len(traj), 1) * max(len(lig_idx), 1) * avg_density * shell_volumes
    g_r = np.divide(counts, denom, out=np.zeros_like(counts), where=denom > 0.0)
    return radii, g_r


def build_1d_pmf(
    distance_nm: Sequence[float],
    temperature_k: float,
    bin_width_nm: float,
    edges_nm: np.ndarray | None = None,
    shift_kjmol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if not distance_nm:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    dist = np.asarray(distance_nm, dtype=float)
    if edges_nm is None:
        d_min = max(0.0, float(np.min(dist)) - bin_width_nm)
        d_max = float(np.max(dist)) + bin_width_nm
        n_bins = max(10, int(math.ceil((d_max - d_min) / max(bin_width_nm, 1.0e-6))))
        edges = np.linspace(d_min, d_max, n_bins + 1)
    else:
        edges = np.asarray(edges_nm, dtype=float)
    counts, _ = np.histogram(dist, bins=edges)
    prob = counts.astype(np.float64) / max(np.sum(counts), 1.0)
    pmf = np.full_like(prob, np.nan, dtype=np.float64)
    valid = prob > 0.0
    kbt = 0.00831446261815324 * float(temperature_k)
    pmf[valid] = -kbt * np.log(prob[valid])
    if np.any(valid):
        pmf[valid] += float(shift_kjmol)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, pmf


def choose_pmf_reference_region(
    centers_nm: np.ndarray,
    pmf_arrays: Sequence[np.ndarray],
    preferred_start_nm: float,
) -> Tuple[np.ndarray, float]:
    finite_all = np.logical_and.reduce([np.isfinite(arr) for arr in pmf_arrays])
    mask = finite_all & (centers_nm >= float(preferred_start_nm))
    if np.any(mask):
        return mask, float(preferred_start_nm)
    if np.any(finite_all):
        fallback_start = float(np.percentile(centers_nm[finite_all], 75.0))
        mask = finite_all & (centers_nm >= fallback_start)
        if np.any(mask):
            return mask, fallback_start
    return finite_all, float(preferred_start_nm)


def build_safe_histogram_edges(
    values: Sequence[float],
    bin_width_nm: float,
    lower_nm: float | None = None,
    upper_nm: float | None = None,
    min_bins: int = 10,
    force_full_range: bool = False,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        start = 0.0 if lower_nm is None else float(lower_nm)
        stop = start + max(float(bin_width_nm), 0.01)
        return np.linspace(start, stop, max(2, int(min_bins)) + 1)

    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    start = arr_min - float(bin_width_nm)
    stop = arr_max + float(bin_width_nm)

    if lower_nm is not None:
        start = float(lower_nm) if force_full_range else max(start, float(lower_nm))
    if upper_nm is not None:
        stop = float(upper_nm) if force_full_range else min(stop, float(upper_nm))

    if not np.isfinite(start):
        start = arr_min
    if not np.isfinite(stop):
        stop = arr_max + float(bin_width_nm)

    if stop <= start:
        center = 0.5 * (arr_min + arr_max)
        half_span = max(float(bin_width_nm), 0.01) * max(1, int(min_bins) // 2)
        start = center - half_span
        stop = center + half_span
        if lower_nm is not None:
            start = max(start, float(lower_nm))
        if upper_nm is not None:
            stop = min(stop, float(upper_nm))
        if stop <= start:
            stop = start + max(float(bin_width_nm), 0.01)

    n_bins = max(int(min_bins), int(math.ceil((stop - start) / max(float(bin_width_nm), 1.0e-6))))
    return np.linspace(start, stop, n_bins + 1)


def run_contact_and_pmf_analysis(
    args: argparse.Namespace,
    output_dir: str,
) -> Dict:
    print("[6/8] 分析 L-E min-distance / RDF / 1D PMF")
    analysis_r_min = float(args.analysis_r_min)
    analysis_r_max = float(args.analysis_r_max)
    if analysis_r_max <= analysis_r_min:
        raise ValueError("analysis-r-max 必须大于 analysis-r-min")
    original_traj, original_sampled = load_analysis_traj(
        os.path.join(output_dir, "original_baseline", "traj.dcd"),
        args.traj_top,
        args.analysis_max_frames,
    )
    dexp_traj, dexp_sampled = load_analysis_traj(
        os.path.join(output_dir, "dexp_surrogate", "traj.dcd"),
        args.traj_top,
        args.analysis_max_frames,
    )
    lig_heavy, env_heavy = get_ligand_env_heavy_indices(original_traj.top, args.ligand)

    original_min = compute_min_distance_series_nm(original_traj, lig_heavy, env_heavy)
    dexp_min = compute_min_distance_series_nm(dexp_traj, lig_heavy, env_heavy)
    min_rows = [
        {
            "ensemble": "original_baseline",
            "frame_index": int(frame_idx),
            "min_distance_nm": float(value),
        }
        for frame_idx, value in zip(original_sampled, original_min)
    ] + [
        {
            "ensemble": "dexp_surrogate",
            "frame_index": int(frame_idx),
            "min_distance_nm": float(value),
        }
        for frame_idx, value in zip(dexp_sampled, dexp_min)
    ]
    min_csv = write_rows_csv(os.path.join(output_dir, "le_min_distance_comparison.csv"), min_rows)

    rdf_r_original, rdf_g_original = compute_rdf(
        original_traj, lig_heavy, env_heavy, max(float(args.rdf_r_max), analysis_r_max), args.rdf_bin_width
    )
    rdf_r_dexp, rdf_g_dexp = compute_rdf(
        dexp_traj, lig_heavy, env_heavy, max(float(args.rdf_r_max), analysis_r_max), args.rdf_bin_width
    )
    rdf_rows: List[Dict] = []
    for radius, g_mm, g_dexp in zip(rdf_r_original, rdf_g_original, rdf_g_dexp):
        rdf_rows.append(
            {
                "r_nm": float(radius),
                "g_r_original": float(g_mm),
                "g_r_dexp": float(g_dexp),
                "delta_g_r": float(g_dexp - g_mm),
            }
        )
    rdf_csv = write_rows_csv(os.path.join(output_dir, "le_rdf_comparison.csv"), rdf_rows)

    pmf_edges = build_safe_histogram_edges(
        original_min + dexp_min,
        args.pmf_bin_width,
        lower_nm=analysis_r_min,
        upper_nm=analysis_r_max,
        min_bins=10,
        force_full_range=True,
    )
    pmf_r_original, pmf_original_raw = build_1d_pmf(
        original_min, args.temperature, args.pmf_bin_width, edges_nm=pmf_edges
    )
    pmf_r_dexp, pmf_dexp_raw = build_1d_pmf(
        dexp_min, args.temperature, args.pmf_bin_width, edges_nm=pmf_edges
    )
    n_pmf = min(len(pmf_r_original), len(pmf_r_dexp))
    ref_mask, ref_start_nm = choose_pmf_reference_region(
        pmf_r_original[:n_pmf],
        [pmf_original_raw[:n_pmf], pmf_dexp_raw[:n_pmf]],
        preferred_start_nm=max(0.50, analysis_r_max - 0.10),
    )
    original_ref = float(np.nanmean(pmf_original_raw[:n_pmf][ref_mask])) if n_pmf > 0 and np.any(ref_mask) else 0.0
    dexp_ref = float(np.nanmean(pmf_dexp_raw[:n_pmf][ref_mask])) if n_pmf > 0 and np.any(ref_mask) else 0.0
    pmf_original = pmf_original_raw.copy()
    pmf_dexp = pmf_dexp_raw.copy()
    if n_pmf > 0:
        finite_original = np.isfinite(pmf_original[:n_pmf])
        finite_dexp = np.isfinite(pmf_dexp[:n_pmf])
        pmf_original[:n_pmf] = np.where(
            finite_original,
            pmf_original[:n_pmf] - original_ref,
            pmf_original[:n_pmf],
        )
        pmf_dexp[:n_pmf] = np.where(
            finite_dexp,
            pmf_dexp[:n_pmf] - dexp_ref,
            pmf_dexp[:n_pmf],
        )
    pmf_rows: List[Dict] = []
    for idx in range(n_pmf):
        pmf_rows.append(
            {
                "distance_nm": float(pmf_r_original[idx]),
                "pmf_original_kjmol": float(pmf_original[idx]) if np.isfinite(pmf_original[idx]) else math.nan,
                "pmf_dexp_kjmol": float(pmf_dexp[idx]) if np.isfinite(pmf_dexp[idx]) else math.nan,
                "delta_pmf_kjmol": float(pmf_dexp[idx] - pmf_original[idx])
                if np.isfinite(pmf_original[idx]) and np.isfinite(pmf_dexp[idx])
                else math.nan,
                "analysis_r_min_nm": float(analysis_r_min),
                "analysis_r_max_nm": float(analysis_r_max),
                "pmf_reference_region_start_nm": float(ref_start_nm),
            }
        )
    pmf_csv = write_rows_csv(os.path.join(output_dir, "le_pmf_1d_comparison.csv"), pmf_rows)

    working_window_mask = (rdf_r_original >= analysis_r_min) & (rdf_r_original <= analysis_r_max)
    pmf_window_mask = (pmf_r_original[:n_pmf] >= analysis_r_min) & (pmf_r_original[:n_pmf] <= analysis_r_max)
    summary = {
        "min_distance_csv": min_csv,
        "rdf_csv": rdf_csv,
        "pmf_csv": pmf_csv,
        "analysis_r_min_nm": float(analysis_r_min),
        "analysis_r_max_nm": float(analysis_r_max),
        "ligand_heavy_atoms": int(len(lig_heavy)),
        "environment_heavy_atoms": int(len(env_heavy)),
        "original_min_distance_nm": summarize_series_with_percentiles(original_min),
        "dexp_min_distance_nm": summarize_series_with_percentiles(dexp_min),
        "rdf_working_window_peak_original": float(np.max(rdf_g_original[working_window_mask])) if np.any(working_window_mask) else math.nan,
        "rdf_working_window_peak_dexp": float(np.max(rdf_g_dexp[working_window_mask])) if np.any(working_window_mask) else math.nan,
        "pmf_reference_region_start_nm": float(ref_start_nm),
        "pmf_working_window_delta_max_kjmol": float(
            np.nanmax(np.abs((pmf_dexp[:n_pmf] - pmf_original[:n_pmf])[pmf_window_mask]))
        ) if n_pmf > 0 and np.any(pmf_window_mask) else math.nan,
    }
    return summary


def run_delta_u_analysis(
    args: argparse.Namespace,
    output_dir: str,
    original_system,
    dexp_system,
) -> Dict:
    print("[7/8] 统计 ΔU = U_DEXP - U_MM 分布")
    mm_context = build_context_for_system(original_system, args)
    dexp_context = build_context_for_system(dexp_system, args)
    rows: List[Dict] = []

    for ensemble, traj_path in (
        ("original_baseline", os.path.join(output_dir, "original_baseline", "traj.dcd")),
        ("dexp_surrogate", os.path.join(output_dir, "dexp_surrogate", "traj.dcd")),
    ):
        traj, sampled_indices = load_analysis_traj(traj_path, args.traj_top, args.analysis_max_frames)
        box = np.asarray(traj.unitcell_vectors, dtype=np.float64) if traj.unitcell_vectors is not None else None
        for local_idx, frame_idx in enumerate(sampled_indices):
            pos_nm = np.asarray(traj.xyz[local_idx], dtype=np.float64)
            box_vecs = box[local_idx] if box is not None else None
            u_mm = evaluate_context(mm_context, pos_nm, box_vecs, include_forces=False)["potential_kjmol"]
            u_dexp = evaluate_context(
                dexp_context,
                pos_nm,
                box_vecs,
                lam_coul=1.0,
                lam_vdw=1.0,
                include_forces=False,
            )["potential_kjmol"]
            rows.append(
                {
                    "ensemble": ensemble,
                    "frame_index": int(frame_idx),
                    "u_mm_kjmol": float(u_mm),
                    "u_dexp_kjmol": float(u_dexp),
                    "delta_u_kjmol": float(u_dexp - u_mm),
                }
            )

    csv_path = write_rows_csv(os.path.join(output_dir, "delta_u_distribution.csv"), rows)
    delta_by_ensemble: Dict[str, List[float]] = {}
    for row in rows:
        delta_by_ensemble.setdefault(str(row["ensemble"]), []).append(float(row["delta_u_kjmol"]))
    all_values = [float(row["delta_u_kjmol"]) for row in rows]
    return {
        "delta_u_csv": csv_path,
        "all_frames": summarize_series_with_percentiles(all_values),
        "by_ensemble": {
            label: summarize_series_with_percentiles(values)
            for label, values in delta_by_ensemble.items()
        },
    }


def save_postprocess_plots(output_dir: str) -> Dict[str, str]:
    plt = get_matplotlib_pyplot()
    pngs: Dict[str, str] = {}

    schedule_csv = os.path.join(output_dir, "lambda_schedule_comparison.csv")
    if os.path.isfile(schedule_csv):
        rows = read_csv_rows(schedule_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
        if rows and "schedule" in rows[0]:
            schedules = sorted({row["schedule"] for row in rows})
            for schedule in schedules:
                subset = [row for row in rows if row["schedule"] == schedule]
                subset.sort(key=lambda row: int(row["state"]))
                x = [int(row["state"]) for row in subset]
                y_c = [float(row["lambda_coul"]) for row in subset]
                y_v = [float(row["lambda_vdw"]) for row in subset]
                ax.plot(x, y_c, label=f"{schedule}: lam_coul")
                ax.plot(x, y_v, linestyle="--", label=f"{schedule}: lam_vdw")
            ax.set_xlabel("State")
        else:
            rows_sorted = sorted(rows, key=lambda row: float(row["lambda_value"]), reverse=True)
            x = list(range(len(rows_sorted)))
            y_c = [float(row["lam_coul"]) for row in rows_sorted]
            y_v = [float(row["lam_vdw"]) for row in rows_sorted]
            x_labels = [f"{float(row['lambda_value']):.2f}" for row in rows_sorted]
            ax.plot(x, y_c, marker="o", label="lam_coul")
            ax.plot(x, y_v, marker="o", linestyle="--", label="lam_vdw")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel("Lambda Window")
        ax.set_ylabel("Lambda")
        ax.set_title("Lambda Schedule Comparison")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = os.path.join(output_dir, "lambda_schedule_comparison.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pngs["lambda_schedule_png"] = path

    lambda_csv = os.path.join(output_dir, "lambda_single_point_scan.csv")
    if os.path.isfile(lambda_csv):
        rows = read_csv_rows(lambda_csv)
        grouped: Dict[float, Dict[str, List[float]]] = {}
        for row in rows:
            lam = float(row["lambda_value"])
            payload = grouped.setdefault(lam, {"potential": [], "force": []})
            payload["potential"].append(float(row["potential_kjmol"]))
            payload["force"].append(float(row["max_force_kjmol_per_nm"]))
        lambdas = sorted(grouped.keys(), reverse=True)
        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
        axes[0].plot(lambdas, [statistics.fmean(grouped[lam]["potential"]) for lam in lambdas], marker="o")
        axes[0].set_ylabel("Mean Potential (kJ/mol)")
        axes[0].set_title("Lambda Single-Point Scan")
        axes[0].grid(alpha=0.3)
        axes[1].plot(lambdas, [max(grouped[lam]["force"]) for lam in lambdas], marker="o", color="tab:red")
        axes[1].set_xlabel("Lambda")
        axes[1].set_ylabel("Max Force (kJ/mol/nm)")
        axes[1].grid(alpha=0.3)
        path = os.path.join(output_dir, "lambda_single_point_scan.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pngs["lambda_single_point_scan_png"] = path

    min_csv = os.path.join(output_dir, "le_min_distance_comparison.csv")
    if os.path.isfile(min_csv):
        rows = read_csv_rows(min_csv)
        grouped: Dict[str, List[float]] = {}
        for row in rows:
            grouped.setdefault(str(row["ensemble"]), []).append(float(row["min_distance_nm"]))
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, values in grouped.items():
            ax.hist(values, bins=30, alpha=0.5, density=True, label=label)
        ax.set_xlabel("Min L-E Distance (nm)")
        ax.set_ylabel("Density")
        ax.set_title("Min-Distance Distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(output_dir, "le_min_distance_comparison.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pngs["min_distance_png"] = path

    rdf_csv = os.path.join(output_dir, "le_rdf_comparison.csv")
    if os.path.isfile(rdf_csv):
        rows = read_csv_rows(rdf_csv)
        x = [float(row["r_nm"]) for row in rows]
        y_mm = [float(row["g_r_original"]) for row in rows]
        y_dexp = [float(row["g_r_dexp"]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y_mm, label="original_baseline")
        ax.plot(x, y_dexp, label="dexp_surrogate")
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("g(r)")
        ax.set_title("Ligand-Environment RDF")
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(output_dir, "le_rdf_comparison.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pngs["rdf_png"] = path

    pmf_csv = os.path.join(output_dir, "le_pmf_1d_comparison.csv")
    if os.path.isfile(pmf_csv):
        rows = read_csv_rows(pmf_csv)
        x = [float(row["distance_nm"]) for row in rows]
        y_mm = [float(row["pmf_original_kjmol"]) if row["pmf_original_kjmol"] not in ("", "nan", "NaN") else math.nan for row in rows]
        y_dexp = [float(row["pmf_dexp_kjmol"]) if row["pmf_dexp_kjmol"] not in ("", "nan", "NaN") else math.nan for row in rows]
        analysis_r_min = float(rows[0].get("pmf_reference_region_start_nm", 0.20)) - 0.10 if rows else 0.20
        analysis_r_max = max(x) if x else 0.65
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y_mm, label="original_baseline")
        ax.plot(x, y_dexp, label="dexp_surrogate")
        ax.set_xlabel("Min L-E Distance (nm)")
        ax.set_ylabel("Relative Free Energy (kJ/mol)")
        ax.set_title("1D Contact Free-Energy Profile")
        ax.set_xlim(max(0.20, analysis_r_min), min(0.65, analysis_r_max))
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(output_dir, "le_pmf_1d_comparison.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pngs["pmf_png"] = path

    delta_u_csv = os.path.join(output_dir, "delta_u_distribution.csv")
    if os.path.isfile(delta_u_csv):
        rows = read_csv_rows(delta_u_csv)
        grouped: Dict[str, List[float]] = {}
        for row in rows:
            grouped.setdefault(str(row["ensemble"]), []).append(float(row["delta_u_kjmol"]))
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, values in grouped.items():
            ax.hist(values, bins=30, alpha=0.5, density=True, label=label)
        ax.set_xlabel("ΔU = U_DEXP - U_MM (kJ/mol)")
        ax.set_ylabel("Density")
        ax.set_title("Delta-U Distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(output_dir, "delta_u_distribution.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pngs["delta_u_png"] = path

    return pngs


def run_lambda_window_contact_analysis(
    args: argparse.Namespace,
    output_dir: str,
) -> Dict:
    schedule_csv = os.path.join(output_dir, "lambda_schedule_comparison.csv")
    rows = read_csv_rows(schedule_csv)
    lambda_rows = [row for row in rows if int(float(row.get("used_for_postprocess", 0))) == 1]
    if not lambda_rows:
        return {}

    pmf_rows: List[Dict] = []
    rdf_rows: List[Dict] = []
    min_rows: List[Dict] = []
    summaries: List[Dict] = []

    for row in lambda_rows:
        lam = float(row["lambda_value"])
        window_dir = str(row["window_dir"])
        traj, sampled = load_analysis_traj(
            os.path.join(window_dir, "traj.dcd"),
            args.traj_top,
            args.analysis_max_frames,
        )
        lig_heavy, env_heavy = get_ligand_env_heavy_indices(traj.top, args.ligand)
        min_series = compute_min_distance_series_nm(traj, lig_heavy, env_heavy)
        for frame_idx, value in zip(sampled, min_series):
            min_rows.append(
                {
                    "lambda_value": float(lam),
                    "frame_index": int(frame_idx),
                    "min_distance_nm": float(value),
                }
            )

        rdf_r, rdf_g = compute_rdf(traj, lig_heavy, env_heavy, args.rdf_r_max, args.rdf_bin_width)
        for radius, g_val in zip(rdf_r, rdf_g):
            rdf_rows.append(
                {
                    "lambda_value": float(lam),
                    "r_nm": float(radius),
                    "g_r": float(g_val),
                }
            )

        pmf_min = np.asarray(min_series, dtype=float)
        pmf_edges = build_safe_histogram_edges(
            pmf_min,
            args.pmf_bin_width,
            lower_nm=float(args.analysis_r_min),
            upper_nm=float(args.analysis_r_max),
            min_bins=10,
            force_full_range=True,
        )
        pmf_r, pmf = build_1d_pmf(min_series, args.temperature, args.pmf_bin_width, edges_nm=pmf_edges)
        finite_mask = np.isfinite(pmf) & (pmf_r >= float(args.fit_r_max))
        if not np.any(finite_mask):
            finite_mask = np.isfinite(pmf)
        pmf_ref = float(np.nanmean(pmf[finite_mask])) if np.any(finite_mask) else 0.0
        pmf = np.where(np.isfinite(pmf), pmf - pmf_ref, pmf)
        for distance_nm, pmf_val in zip(pmf_r, pmf):
            pmf_rows.append(
                {
                    "lambda_value": float(lam),
                    "distance_nm": float(distance_nm),
                    "pmf_kjmol": float(pmf_val) if np.isfinite(pmf_val) else math.nan,
                }
            )

        summary = summarize_series_with_percentiles(min_series)
        summary["lambda_value"] = float(lam)
        summaries.append(summary)

    min_csv = write_rows_csv(os.path.join(output_dir, "lambda_window_min_distance.csv"), min_rows)
    rdf_csv = write_rows_csv(os.path.join(output_dir, "lambda_window_rdf.csv"), rdf_rows)
    pmf_csv = write_rows_csv(os.path.join(output_dir, "lambda_window_pmf.csv"), pmf_rows)

    plt = get_matplotlib_pyplot()
    pngs: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in sorted({float(row["lambda_value"]) for row in rdf_rows}, reverse=True):
        subset = [row for row in rdf_rows if abs(float(row["lambda_value"]) - lam) < 1.0e-8]
        subset.sort(key=lambda item: float(item["r_nm"]))
        ax.plot([float(item["r_nm"]) for item in subset], [float(item["g_r"]) for item in subset], label=f"λ={lam:.2f}")
    ax.set_xlim(float(args.analysis_r_min), float(args.analysis_r_max))
    ax.set_xlabel("r (nm)")
    ax.set_ylabel("g(r)")
    ax.set_title("Lambda-Resolved RDF")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(output_dir, "lambda_window_rdf.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    pngs["lambda_window_rdf_png"] = path

    fig, ax = plt.subplots(figsize=(7, 5))
    for lam in sorted({float(row["lambda_value"]) for row in pmf_rows}, reverse=True):
        subset = [row for row in pmf_rows if abs(float(row["lambda_value"]) - lam) < 1.0e-8]
        subset.sort(key=lambda item: float(item["distance_nm"]))
        ax.plot(
            [float(item["distance_nm"]) for item in subset],
            [float(item["pmf_kjmol"]) if str(item["pmf_kjmol"]).lower() != "nan" else math.nan for item in subset],
            label=f"λ={lam:.2f}",
        )
    ax.set_xlim(float(args.analysis_r_min), float(args.analysis_r_max))
    ax.set_xlabel("Min L-E Distance (nm)")
    ax.set_ylabel("Relative Free Energy (kJ/mol)")
    ax.set_title("Lambda-Resolved 1D PMF")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(output_dir, "lambda_window_pmf.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    pngs["lambda_window_pmf_png"] = path

    fig, ax = plt.subplots(figsize=(7, 5))
    lambdas = [float(item["lambda_value"]) for item in summaries]
    p05 = [float(item["p05"]) for item in summaries]
    p50 = [float(item["p50"]) for item in summaries]
    p95 = [float(item["p95"]) for item in summaries]
    ax.plot(lambdas, p50, marker="o", label="p50")
    ax.fill_between(lambdas, p05, p95, alpha=0.25, label="p05-p95")
    ax.invert_xaxis()
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Min L-E Distance (nm)")
    ax.set_title("Lambda-Resolved Min-Distance Summary")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(output_dir, "lambda_window_min_distance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    pngs["lambda_window_min_distance_png"] = path

    return {
        "lambda_window_min_distance_csv": min_csv,
        "lambda_window_rdf_csv": rdf_csv,
        "lambda_window_pmf_csv": pmf_csv,
        "lambda_window_summaries": summaries,
        **pngs,
    }


def run_postprocess_analysis(
    args: argparse.Namespace,
    output_dir: str,
    original_system,
    dexp_system,
    topology,
    positions,
    box_vectors,
) -> Dict:
    lambda_window_info = run_lambda_window_ensemble(
        args=args,
        system=dexp_system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        output_dir=output_dir,
    )
    lambda_scan_summary = run_lambda_single_point_scan(args, output_dir, dexp_system)
    contact_summary = run_contact_and_pmf_analysis(args, output_dir)
    delta_u_summary = run_delta_u_analysis(args, output_dir, original_system, dexp_system)
    lambda_window_summary = run_lambda_window_contact_analysis(args, output_dir)
    schedule_csv = lambda_window_info["schedule_csv"]
    plot_paths = save_postprocess_plots(output_dir)
    lambda_scan_summary.update({k: v for k, v in plot_paths.items() if k.startswith("lambda_")})
    contact_summary.update(
        {
            k: v
            for k, v in plot_paths.items()
            if k in {"min_distance_png", "rdf_png", "pmf_png"}
        }
    )
    delta_u_summary.update({k: v for k, v in plot_paths.items() if k == "delta_u_png"})
    return {
        "lambda_single_point_scan": lambda_scan_summary,
        "contact_diagnostics": contact_summary,
        "delta_u_distribution": delta_u_summary,
        "lambda_window_analysis": lambda_window_summary,
        "lambda_schedule_csv": schedule_csv,
        "plot_paths": plot_paths,
    }


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

    # 仅记录正式 1 ns production，避免把 surrogate warmup 混入 RDF/PMF/能量统计。
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


def run_fixed_lambda_window_simulation(
    system,
    topology,
    positions,
    box_vectors,
    args: argparse.Namespace,
    output_dir: str,
    lambda_value: float,
) -> Dict:
    openmm, app, unit, _ = require_openmm()
    label = f"lambda_{lambda_value:.2f}".replace(".", "p")
    sim_dir = ensure_dir(os.path.join(output_dir, "lambda_windows", label))
    csv_path = os.path.join(sim_dir, "state.csv")
    dcd_path = os.path.join(sim_dir, "traj.dcd")

    sim_system = strip_barostat(system)
    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.dt_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(args.seed + int(round(lambda_value * 1000.0)))
    platform, properties = select_platform(args.platform)
    simulation = app.Simulation(topology, sim_system, integrator, platform, properties)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(
        args.temperature * unit.kelvin,
        args.seed + int(round(lambda_value * 1000.0)),
    )

    for parameter_name in ("lam_coul", "lam_vdw"):
        try:
            simulation.context.setParameter(parameter_name, float(lambda_value))
        except Exception:
            pass

    if args.minimize:
        openmm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=250)

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

    n_steps = int(round(args.lambda_window_ns * 1000.0 / (args.dt_fs / 1000.0)))
    simulation.step(max(n_steps, 1))
    data = read_state_csv(csv_path)
    summary = {
        "label": label,
        "lambda_value": float(lambda_value),
        "steps": int(max(n_steps, 1)),
        "sim_ns": float(args.lambda_window_ns),
        "potential_kjmol": summarize_series(data["potentialEnergy"]),
        "kinetic_kjmol": summarize_series(data["kineticEnergy"]),
        "total_kjmol": summarize_series(data["totalEnergy"]),
        "temperature_K": summarize_series(data["temperature"]),
    }
    summary.update(compute_ligand_rmsd_metrics(dcd_path, args.traj_top, args.ligand))
    with open(os.path.join(sim_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def run_lambda_window_ensemble(
    args: argparse.Namespace,
    system,
    topology,
    positions,
    box_vectors,
    output_dir: str,
) -> Dict:
    lambda_values = parse_lambda_window_values(args)
    rows: List[Dict] = []
    summaries: List[Dict] = []
    for lam in lambda_values:
        summary = run_fixed_lambda_window_simulation(
            system=system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            args=args,
            output_dir=output_dir,
            lambda_value=float(lam),
        )
        summaries.append(summary)
        rows.append(
            {
                "lambda_value": float(lam),
                "lam_coul": float(lam),
                "lam_vdw": float(lam),
                "sim_ns": float(args.lambda_window_ns),
                "window_dir": os.path.join(output_dir, "lambda_windows", summary["label"]),
                "used_for_postprocess": 1,
            }
        )
    csv_path = write_rows_csv(os.path.join(output_dir, "lambda_schedule_comparison.csv"), rows)
    return {
        "lambda_values": [float(x) for x in lambda_values],
        "schedule_csv": csv_path,
        "window_summaries": summaries,
    }


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


def build_surrogate_activation_reference_schedule(n_states: int) -> List[Tuple[int, str, float, float]]:
    if n_states < 3:
        raise ValueError("schedule 至少需要 3 个状态")
    rows: List[Tuple[int, str, float, float]] = []
    phase_edges = np.linspace(0, n_states - 1, 4, dtype=int)
    phase_edges[-1] = n_states - 1
    for state in range(n_states):
        if state <= phase_edges[1]:
            frac = state / max(phase_edges[1], 1)
            lam_coul = 0.0
            lam_vdw = 0.05 + 0.20 * frac
            stage = "vdw_softstart"
        elif state <= phase_edges[2]:
            frac = (state - phase_edges[1]) / max(phase_edges[2] - phase_edges[1], 1)
            lam_coul = 0.0
            lam_vdw = 0.25 + 0.75 * frac
            stage = "vdw_ramp"
        else:
            frac = (state - phase_edges[2]) / max((n_states - 1) - phase_edges[2], 1)
            lam_coul = frac
            lam_vdw = 1.0
            stage = "coul_ramp"
        rows.append((state, stage, min(max(lam_coul, 0.0), 1.0), min(max(lam_vdw, 0.0), 1.0)))
    return rows


def write_schedule_comparison(output_dir: str, n_states: int) -> str:
    out_csv = os.path.join(output_dir, "lambda_schedule_comparison.csv")
    rows: List[Dict] = []
    for state in range(n_states):
        frac = state / max(n_states - 1, 1)
        lam = 1.0 - frac
        rows.append(
            {
                "schedule": "traditional_linear_decoupling",
                "state": state,
                "stage": "coupled",
                "lambda_coul": lam,
                "lambda_vdw": lam,
                "direction": "1_to_0",
                "used_by_current_stability_run": 0,
                "notes": "Reference traditional decoupling path only",
            }
        )
    for state, stage, lam_coul, lam_vdw in build_surrogate_activation_reference_schedule(n_states):
        rows.append(
            {
                "schedule": "current_surrogate_activation_warmup",
                "state": state,
                "stage": stage,
                "lambda_coul": lam_coul,
                "lambda_vdw": lam_vdw,
                "direction": "0_to_1",
                "used_by_current_stability_run": 1,
                "notes": "Reference of actual surrogate warmup path used before production",
            }
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def write_comparison_report(
    output_dir: str,
    original_summary: Dict,
    dexp_summary: Dict,
    fitted_params: Dict,
    schedule_csv: str,
    lambda_scan_summary: Dict,
    contact_summary: Dict,
    delta_u_summary: Dict,
    lambda_window_summary: Dict | None = None,
) -> str:
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
        f"- fit_target_definition: {fitted_params.get('fit_target_definition')}",
        f"- qm_mm_offset_kjmol (diagnostic only): {fitted_params.get('qm_mm_offset_kjmol')}",
        f"- diagnostic_global_mu: {fitted_params.get('diagnostic_global_mu')}",
        f"- diagnostic_fit_c0: {fitted_params.get('diagnostic_fit_c0')}",
        f"- diagnostic_weighted_center: {fitted_params.get('diagnostic_weighted_center')}",
        f"- diagnostic_centered_std: {fitted_params.get('diagnostic_centered_std')}",
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
        "## Lambda Single-Point Scan",
        f"- Scan CSV: {lambda_scan_summary.get('scan_csv')}",
        f"- All finite: {lambda_scan_summary.get('all_finite')}",
        f"- Max |ΔU(lambda_i)-ΔU(lambda_i-1)| (kJ/mol): {lambda_scan_summary.get('max_abs_energy_jump_kjmol', math.nan):.3f}",
        f"- Max force across scan (kJ/mol/nm): {lambda_scan_summary.get('max_force_kjmol_per_nm', math.nan):.3f}",
        "",
        "## Contact Diagnostics",
        f"- Min-distance CSV: {contact_summary.get('min_distance_csv')}",
        f"- RDF CSV: {contact_summary.get('rdf_csv')}",
        f"- PMF CSV: {contact_summary.get('pmf_csv')}",
        f"- PMF PNG: {contact_summary.get('pmf_png')}",
        f"- RDF PNG: {contact_summary.get('rdf_png')}",
        "- RDF / PMF 当前基于 production 轨迹的接触统计对比，属于几何/热力学 proxy，不是严格的传统 ABFE PMF。",
        f"- Analysis window (nm): {contact_summary.get('analysis_r_min_nm', math.nan):.2f} to {contact_summary.get('analysis_r_max_nm', math.nan):.2f}",
        f"- PMF reference-region start (nm): {contact_summary.get('pmf_reference_region_start_nm', math.nan):.3f}",
        f"- Original min-distance p05 / p50 / p95 (nm): "
        f"{contact_summary['original_min_distance_nm']['p05']:.3f} / "
        f"{contact_summary['original_min_distance_nm']['p50']:.3f} / "
        f"{contact_summary['original_min_distance_nm']['p95']:.3f}",
        f"- DEXP min-distance p05 / p50 / p95 (nm): "
        f"{contact_summary['dexp_min_distance_nm']['p05']:.3f} / "
        f"{contact_summary['dexp_min_distance_nm']['p50']:.3f} / "
        f"{contact_summary['dexp_min_distance_nm']['p95']:.3f}",
        f"- Working-window RDF peak original/dexp: "
        f"{contact_summary.get('rdf_working_window_peak_original', math.nan):.3f} / "
        f"{contact_summary.get('rdf_working_window_peak_dexp', math.nan):.3f}",
        f"- Working-window PMF max |Δ| (kJ/mol): {contact_summary.get('pmf_working_window_delta_max_kjmol', math.nan):.3f}",
        "",
        "## Delta-U Distribution",
        f"- CSV: {delta_u_summary.get('delta_u_csv')}",
        f"- PNG: {delta_u_summary.get('delta_u_png')}",
        f"- All-frame ΔU mean ± std (kJ/mol): "
        f"{delta_u_summary['all_frames']['mean']:.3f} ± {delta_u_summary['all_frames']['std']:.3f}",
        f"- All-frame ΔU p05 / p50 / p95 (kJ/mol): "
        f"{delta_u_summary['all_frames']['p05']:.3f} / "
        f"{delta_u_summary['all_frames']['p50']:.3f} / "
        f"{delta_u_summary['all_frames']['p95']:.3f}",
        "",
        "## Lambda Schedules",
        f"- CSV: {schedule_csv}",
        f"- PNG: {lambda_scan_summary.get('lambda_schedule_png')}",
        "- 当前 `lambda_schedule_comparison.csv` 记录的是后处理实际重跑的固定 lambda 窗口。",
        "- 当前脚本没有完成传统 ABFE vs surrogate-correction 的自由能数值对比；这里只有稳定性/几何/能量 proxy 对比。",
        "",
    ]
    if lambda_window_summary:
        lines.extend(
            [
                "## Lambda-Resolved Contact Analysis",
                f"- Window RDF CSV: {lambda_window_summary.get('lambda_window_rdf_csv')}",
                f"- Window PMF CSV: {lambda_window_summary.get('lambda_window_pmf_csv')}",
                f"- Window Min-distance CSV: {lambda_window_summary.get('lambda_window_min_distance_csv')}",
                f"- Window RDF PNG: {lambda_window_summary.get('lambda_window_rdf_png')}",
                f"- Window PMF PNG: {lambda_window_summary.get('lambda_window_pmf_png')}",
                f"- Window Min-distance PNG: {lambda_window_summary.get('lambda_window_min_distance_png')}",
                "",
            ]
        )
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
    system, topology = load_cached_system(args.system_xml, args.traj_top)
    _, positions, box_vectors = load_last_frame_positions(args.traj, args.traj_top)
    ligand_indices = load_ligand_indices(args.ligand_indices)
    env_indices = [
        idx for idx in range(system.getNumParticles())
        if idx not in set(ligand_indices)
    ]

    if args.postprocess_only:
        params_path = ensure_file(os.path.join(output_dir, "dexp_fitted_params.json"), "已拟合 DEXP 参数")
        with open(params_path, "r", encoding="utf-8") as handle:
            fitted_params = json.load(handle)
    else:
        fitted_params = fit_dexp_from_tail_frames(args, output_dir)
        print("[2/4] 载入原始系统与最后一帧坐标")

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

    if args.postprocess_only:
        original_summary_path = ensure_file(os.path.join(output_dir, "original_baseline", "summary.json"), "baseline summary")
        dexp_summary_path = ensure_file(os.path.join(output_dir, "dexp_surrogate", "summary.json"), "dexp summary")
        with open(original_summary_path, "r", encoding="utf-8") as handle:
            original_summary = json.load(handle)
        with open(dexp_summary_path, "r", encoding="utf-8") as handle:
            dexp_summary = json.load(handle)
    else:
        print("[3/4] 构建 DEXP surrogate system 并执行 1 ns 稳定性测试")
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

    print("[后处理] 生成 CSV / PNG 诊断产物")
    postprocess = run_postprocess_analysis(
        args,
        output_dir,
        system,
        dexp_system,
        topology,
        positions,
        box_vectors,
    )
    lambda_scan_summary = postprocess["lambda_single_point_scan"]
    contact_summary = postprocess["contact_diagnostics"]
    delta_u_summary = postprocess["delta_u_distribution"]
    lambda_window_summary = postprocess.get("lambda_window_analysis", {})
    schedule_csv = postprocess["lambda_schedule_csv"]

    report_path = write_comparison_report(
        output_dir,
        original_summary=original_summary,
        dexp_summary=dexp_summary,
        fitted_params=fitted_params,
        schedule_csv=schedule_csv,
        lambda_scan_summary=lambda_scan_summary,
        contact_summary=contact_summary,
        delta_u_summary=delta_u_summary,
        lambda_window_summary=lambda_window_summary,
    )

    comparison_json = os.path.join(output_dir, "comparison_summary.json")
    with open(comparison_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fitted_params": fitted_params,
                "dexp_surrogate": dexp_summary,
                "original_baseline": original_summary,
                "lambda_single_point_scan": lambda_scan_summary,
                "contact_diagnostics": contact_summary,
                "delta_u_distribution": delta_u_summary,
                "lambda_window_analysis": lambda_window_summary,
                "plot_paths": postprocess.get("plot_paths", {}),
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

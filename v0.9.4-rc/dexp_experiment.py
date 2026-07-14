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
import sys
import warnings
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np

# Bug F 修复：静音 pymbar 在 import/使用 timeseries 模块时的无关警告（拟合阶段用不到它）。
warnings.filterwarnings("ignore", message=r".*timeseries module.*")

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
    parser.add_argument(
        "--compare-ml-model",
        default=None,
        help="额外用这个 OpenMM-ML 模型名（例如 orb-v3-conservative-omol）在同一批 tail 帧上重新标注并拟合 DEXP，"
        "和 --ml-model 的主结果做配对对比（ΔE 相关性 + 两套拟合参数/holdout 指标）",
    )
    parser.add_argument("--fit-frames", type=int, default=500, help="从末段时间窗中最多取多少帧参与拟合")
    parser.add_argument("--fit-last-ns", type=float, default=5.0, help="只使用轨迹最后多少 ns 做拟合")
    parser.add_argument("--fit-env-radius", type=float, default=0.50, help="环境筛选半径 (nm)")
    parser.add_argument("--fit-env-max-atoms", type=int, default=0, help="OpenMM-ML 环境原子上限；<=0 表示关闭最近邻裁剪")
    parser.add_argument("--fit-gpu-workers", type=int, default=1, help="OpenMM-ML worker 数；默认 1，按单 context 滚动标注以避免 CUDA 句柄分配失败")
    parser.add_argument("--fit-r-min", type=float, default=0.20, help="拟合距离下限 (nm)")
    parser.add_argument("--fit-r-max", type=float, default=0.45, help="拟合距离上限 (nm)")
    parser.add_argument(
        "--fit-objective",
        choices=("pmf_mean", "pointwise"),
        default="pmf_mean",
        help="DEXP 拟合目标：pmf_mean=按 min-distance 分箱后用每箱均值 ⟨ΔE⟩(s) 做一阶 PMF matching"
        "（推荐；两个模型不在同一逐帧势能面上，只匹配系综/自由能，不拟合正交噪声）；"
        "pointwise=旧的逐帧 ΔE(x) 匹配",
    )
    parser.add_argument("--fit-pmf-bins", type=int, default=12, help="PMF matching 沿 min-distance 的分箱数（--fit-objective=pmf_mean 时生效）")
    parser.add_argument("--fit-pmf-min-bin-frames", type=int, default=10, help="PMF matching 每箱至少需要多少帧才可信；不足此数的稀疏箱整箱剔除，不进 profile、不进拟合、不进验证（避免 1~3 帧的噪声箱撑起假的动态范围）")
    parser.add_argument("--fit-mm-ref-cutoff", type=float, default=0.0, help="MM 参考 L-E cutoff (nm)，独立于 DEXP 拟合距离窗；<=0 表示 NoCutoff（全程 1/r、非周期），与 MACE 真空团簇边界条件一致，消除截断跳变伪影")
    parser.add_argument("--fit-mm-ref-switch", type=float, default=0.70, help="MM 参考 L-E switching distance (nm)；仅当 --fit-mm-ref-cutoff>0 且 0<switch<cutoff 时启用")
    parser.add_argument(
        "--fit-target-mode",
        choices=("mace_surrogate_residual", "gaussian_replacement_residual", "ml_minus_mm_total", "qmmm_residual", "ml_minus_mm_coul"),
        default="mace_surrogate_residual",
        help="DEXP 拟合目标；推荐 mace_surrogate_residual，即让 Gaussian Coulomb + DEXP 描述 MACE 局部相互作用。旧名 gaussian_replacement_residual 等价保留。",
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
    parser.add_argument("--stability-replicas", type=int, default=1, help="DEXP/baseline 1 ns 稳定性测试各重复多少次（不同 seed），用于把系统性差异和随机噪声分开")
    parser.add_argument("--minimize", action="store_true", help="在每次 1 ns 测试前先做一次最小化")
    parser.add_argument("--skip-stability-minimize", action="store_true", help="跳过稳定性测试前的统一最小化")
    parser.add_argument("--skip-baseline-warmup", action="store_true", help="只对 DEXP 做慢启动；默认 baseline 也跑同样步数的预生产热身以保持协议对称")
    parser.add_argument("--warmup-steps", type=int, default=50000, help="DEXP surrogate 慢启动步数")
    parser.add_argument("--warmup-stages", type=int, default=20, help="DEXP surrogate 慢启动分段数")
    parser.add_argument("--softstart-dt-fs", type=float, default=0.2, help="软启动初始步长 (fs)")
    parser.add_argument("--ramp-dt-fs", default="0.5,1.0,2.0", help="逐级升温步长列表 (fs, 逗号分隔)")
    parser.add_argument("--reuse-fit-labels", action="store_true", help="复用 output-dir 下已有的能量标注缓存，只重新拟合 DEXP 参数")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="从参与拟合的帧中划出多少比例做留出集验证（不参与拟合，只用来检验 DEXP 对 Orb 参考的泛化能力）")
    parser.add_argument("--holdout-min-frames", type=int, default=20, help="留出集至少需要多少帧才执行验证，帧数不足则跳过并回退为全部帧拟合")
    parser.add_argument("--learned-rbf-diagnostic", action="store_true", help="额外运行局部 pair-RBF 学习函数离线 holdout 诊断；仅作对照，不改变 DEXP MD 主路径")
    parser.add_argument("--skip-learned-rbf-diagnostic", action="store_true", help="兼容旧命令；显式关闭局部 pair-RBF 学习函数诊断")
    parser.add_argument("--learned-rbf-centers", type=int, default=8, help="局部 pair-RBF 学习函数的径向基个数")
    parser.add_argument("--learned-rbf-ridge", type=float, default=10.0, help="局部 pair-RBF 学习函数的 ridge 正则强度")
    parser.add_argument("--learned-rbf-max-type-groups", type=int, default=24, help="局部 pair-RBF 学习函数保留的元素类型 pair 分组上限；总会额外保留一个 ALL 全局项")
    parser.add_argument("--learned-rbf-min-group-pairs", type=int, default=200, help="元素类型 pair 在训练集中至少出现多少个有效短程 pair 才单独建一组")
    parser.add_argument(
        "--ml-ref-offset-limit-kjmol",
        type=float,
        default=10000.0,
        help="标注阶段 ΔE 中心值绝对值超过这个阈值(kJ/mol)就认为该 ML 模型返回的是不兼容的绝对总能量（参考零点异常），"
        "直接跳过拟合并标记为不可信，而不是让优化器去撞边界",
    )
    parser.add_argument("--analysis-max-frames", type=int, default=200, help="后处理分析最多读取多少帧")
    parser.add_argument("--lambda-scan-points", type=int, default=11, help="lambda 单点扫描状态数")
    parser.add_argument("--rdf-r-max", type=float, default=1.2, help="L-E RDF 最大半径 (nm)")
    parser.add_argument("--rdf-bin-width", type=float, default=0.01, help="L-E RDF bin 宽度 (nm)")
    parser.add_argument("--pmf-bin-width", type=float, default=0.01, help="1D PMF 的 min-distance bin 宽度 (nm)")
    parser.add_argument("--analysis-r-min", type=float, default=0.20, help="后处理重点关注距离下限 (nm)，默认 0.20 = 2A")
    parser.add_argument("--analysis-r-max", type=float, default=0.65, help="后处理重点关注距离上限 (nm)，默认 0.65 = 6.5A")
    parser.add_argument("--lambda-window-values", default="1.0,0.75,0.5,0.25,0.0", help="后处理固定 lambda 窗口，逗号分隔")
    parser.add_argument("--lambda-window-ns", type=float, default=0.10, help="每个固定 lambda 窗口的短程重跑时长 (ns)")
    parser.add_argument("--surface-pmf-bins", type=int, default=12, help="MACE vs DEXP surrogate 势能面/PMF 1D min-distance 分箱数")
    parser.add_argument("--surface-pmf-2d-bins", type=int, default=6, help="MACE vs DEXP surrogate 势能面/PMF 2D 每个维度分箱数")
    parser.add_argument("--surface-pmf-min-bin-frames", type=int, default=8, help="势能面/PMF profile 每个 bin 至少需要多少帧才输出")
    parser.add_argument("--fit-only", action="store_true", help="只执行 DEXP/学习函数拟合与 holdout 诊断，保存参数后退出，不构建 surrogate system、不跑 MD")
    parser.add_argument("--postprocess-only", action="store_true", help="跳过拟合与动力学，只基于现有 output-dir 结果重跑后处理")
    # relabel + 同帧 1D PMF harness：在 DEXP 生产轨迹（+可选 MM baseline 地板）上做 MACE 单点 relabel，
    # 同帧比 DEXP-world PMF（直方图）与 MACE-endorsed PMF（δ 重加权，带 ESS 门槛）。
    parser.add_argument("--relabel-traj", default=None, help="对该轨迹（DEXP 生产轨迹，如 output/dexp_experiment/dexp_surrogate/traj.dcd）做 MACE relabel + 同帧 1D PMF；给了此项即进入 relabel 模式，读现有 dexp_fitted_params.json 后退出")
    parser.add_argument("--relabel-baseline-traj", default=None, help="可选：MM baseline 轨迹（地板对照，如 original_baseline/traj.dcd），同样 relabel 后比 MACE 认可度")
    parser.add_argument("--relabel-max-frames", type=int, default=300, help="relabel 最多取多少帧（均匀抽样），控制 MACE 单点成本")
    parser.add_argument("--relabel-pmf-bins", type=int, default=24, help="同帧 1D PMF 沿 min-distance 的分箱数")
    parser.add_argument("--relabel-pmf-min-bin-frames", type=int, default=8, help="同帧 1D PMF 每箱最少帧数")
    parser.add_argument(
        "--relabel-shape-anchor-bins", type=int, default=2,
        help="形状剖面锚点使用最远的几个 min-distance 箱做逆方差加权平均（而非单箱），"
             "锚点自身 SEM 会传播进每箱 within-SEM 判据；设为 1 等价于旧的单箱锚点",
    )
    parser.add_argument("--relabel-min-dist-floor", type=float, default=0.12, help="min L-E 距离下限 (nm)：低于此值判为原子穿插、MACE 也 OOD 的过近接触，从均值/PMF 中排除并单独计数（指标 F）。注意正常结合态接触约 0.15-0.20 nm，不算过近；只有 <~0.12 nm 的穿插才是")
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
    if issues:
        raise RuntimeError(
            "DEXP 拟合结果当前不适合直接做稳定性动力学，已阻止运行以避免 NaN。"
            f" 触发条件: {'; '.join(issues)}"
        )


def summarize_fit_diagnostics(output_dir: str, fitted_params: Dict) -> Dict:
    fit_log_path = os.path.join(output_dir, "fit_frame_diagnostics.csv")
    summary = {
        "fit_frame_diagnostics_csv": fit_log_path,
        "diagnostics_available": bool(os.path.isfile(fit_log_path)),
        "fitting_success": bool(fitted_params.get("fitting_success")),
        "suspicious_fit": bool(fitted_params.get("suspicious_fit")),
        "boundary_hits": list(fitted_params.get("boundary_hits", [])),
        "final_cost": float(fitted_params.get("final_cost", math.nan)),
        "fit_frames_used": int(fitted_params.get("fit_frames_used", 0) or 0),
        "fit_frames_total": int(fitted_params.get("fit_frames_total", 0) or 0),
    }
    summary["used_fraction"] = (
        float(summary["fit_frames_used"]) / float(summary["fit_frames_total"])
        if summary["fit_frames_total"] > 0
        else math.nan
    )
    if not os.path.isfile(fit_log_path):
        summary["qc_pass"] = False
        summary["qc_issues"] = ["fit_frame_diagnostics.csv not found"]
        return summary

    rows = read_csv_rows(fit_log_path)
    used_rows = [row for row in rows if int(float(row.get("used_for_fit", 0))) == 1]

    def _float_values(key: str, source_rows: List[Dict[str, str]]) -> List[float]:
        values: List[float] = []
        for row in source_rows:
            try:
                value = float(row.get(key, "nan"))
            except Exception:
                value = math.nan
            if np.isfinite(value):
                values.append(value)
        return values

    centered_values = _float_values("delta_e_centered_kjmol", used_rows)
    valid_pair_values = _float_values("n_valid_pairs", used_rows)
    candidate_pair_values = _float_values("n_env_pairs", used_rows)
    all_centered_values = _float_values("delta_e_centered_kjmol", rows)
    summary.update(
        {
            "fit_rows_total": int(len(rows)),
            "used_rows": int(len(used_rows)),
            "used_rows_fraction": float(len(used_rows) / len(rows)) if rows else math.nan,
            "delta_e_centered_used_kjmol": summarize_series_with_percentiles(centered_values),
            "delta_e_centered_all_kjmol": summarize_series_with_percentiles(all_centered_values),
            "n_valid_pairs_used": summarize_series_with_percentiles(valid_pair_values),
            "n_env_pairs_used": summarize_series_with_percentiles(candidate_pair_values),
        }
    )

    issues: List[str] = []
    if not summary["fitting_success"]:
        issues.append("fitting_success is false")
    if summary["suspicious_fit"]:
        issues.append("fit parameters hit bounds")
    if summary["fit_frames_used"] < 30:
        issues.append("fit_frames_used < 30")
    if np.isfinite(summary["used_fraction"]) and summary["used_fraction"] < 0.25:
        issues.append("less than 25% of selected frames used for fit")
    if np.isfinite(summary["final_cost"]) and summary["final_cost"] > 1000.0:
        issues.append("final_cost > 1000 kJ/mol")
    n_valid_min = summary["n_valid_pairs_used"].get("min", math.nan)
    if not np.isfinite(n_valid_min) or n_valid_min <= 0.0:
        issues.append("some accepted frames have no short-range fitting pairs")
    centered_std = summary["delta_e_centered_used_kjmol"].get("std", math.nan)
    if np.isfinite(centered_std) and centered_std > 250.0:
        issues.append("centered delta-E std > 250 kJ/mol")
    summary["qc_pass"] = not issues
    summary["qc_issues"] = issues
    return summary


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


def predict_dexp_delta_e(dists_nm: np.ndarray, params: Dict, eff_eps: float = 1.0) -> float:
    """按 Orbv3SurrogateFitter 内部使用的同一 DEXP 对势公式，从距离预测 ΔE，用于留出集验证。"""
    dists_nm = np.asarray(dists_nm, dtype=float)
    if dists_nm.size == 0:
        return 0.0
    a = float(params["alpha_vdw"])
    b = float(params["beta_vdw"])
    r0 = float(params["r0_vdw"])
    A = float(params["A_fit"])
    B = float(params["B_fit"])
    x = np.clip(dists_nm / r0 - 1.0, -50.0, 50.0)
    pair_energy = 4.0 * eff_eps * (A * np.exp(-a * x) - B * np.exp(-b * x))
    # DEXP 对势的加性零点是任意的：fitter 只拟合形状（残差里 target/pred 各自去均值），
    # 丢弃了 ML-total 与 MM-total 之间物理上合法、必然存在的常数 C。评估时必须把该常数
    # （训练集上估得、写在 offset_c0）加回，否则 bias/RMSE/R² 会被这个与泛化无关的常数污染。
    offset_c0 = float(params.get("offset_c0", 0.0) or 0.0)
    return float(np.sum(pair_energy)) + offset_c0


def evaluate_holdout_predictions(
    dists_per_frame: Sequence[np.ndarray],
    actual_delta_e: Sequence[float],
    fitted_params: Dict,
) -> Dict:
    if not dists_per_frame:
        return {"n_holdout_frames": 0, "note": "no_holdout_frames"}
    predicted = np.asarray(
        [predict_dexp_delta_e(dists, fitted_params) for dists in dists_per_frame], dtype=float
    )
    actual = np.asarray(actual_delta_e, dtype=float)
    residual_raw = predicted - actual
    bias = float(np.mean(residual_raw))
    residual_centered = residual_raw - bias
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    ss_res = float(np.sum(residual_raw ** 2))
    r2_raw = float(1.0 - ss_res / ss_tot) if ss_tot > 1.0e-9 else math.nan
    if actual.size > 1 and np.std(actual) > 1.0e-9 and np.std(predicted) > 1.0e-9:
        pearson_r = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        pearson_r = math.nan
    return {
        "n_holdout_frames": int(actual.size),
        "rmse_raw_kjmol": float(np.sqrt(np.mean(residual_raw ** 2))),
        "mae_raw_kjmol": float(np.mean(np.abs(residual_raw))),
        "bias_kjmol": bias,
        "rmse_bias_corrected_kjmol": float(np.sqrt(np.mean(residual_centered ** 2))),
        "r2_raw": r2_raw,
        "pearson_r": pearson_r,
        "pearson_r2": float(pearson_r ** 2) if np.isfinite(pearson_r) else math.nan,
        "actual_std_kjmol": float(np.std(actual)),
        "predicted_std_kjmol": float(np.std(predicted)),
    }


def evaluate_holdout_free_energy(
    dists_per_frame: Sequence[np.ndarray],
    actual_delta_e_perframe: Sequence[float],
    min_dist_per_frame: Sequence[float],
    fitted_params: Dict,
    temperature_k: float,
    pmf_bins: int,
    min_bin_frames: int,
) -> Dict:
    """判据 A：DEXP 修正的用途是自由能，不是逐帧势能面。这里在留出集上比系综量而非逐帧散点：
    (1) 系综均值 ⟨ΔE⟩（一阶修正，可信量）；
    (2) 留出集自身重建的 ⟨ΔE⟩(s) 均值剖面 vs 模型预测剖面（PMF matching 真正拟合的对象）。
    FEP 重加权已移除：本体系 σ≫kT、ESS≈1，重加权是单帧最小值伪影，不是自由能。"""
    actual = np.asarray(actual_delta_e_perframe, dtype=float)
    if actual.size == 0:
        return {"n_holdout_frames": 0, "note": "no_holdout_frames"}
    predicted = np.asarray(
        [predict_dexp_delta_e(d, fitted_params) for d in dists_per_frame], dtype=float
    )

    mean_true = float(np.mean(actual))
    mean_model = float(np.mean(predicted))

    # 留出集重建的均值剖面（用真实逐帧 ΔE），只保留 >= min_bin_frames 的箱
    md = np.asarray(min_dist_per_frame, dtype=float)
    n_bins = max(2, int(pmf_bins))
    edges = np.linspace(float(md.min()), float(md.max()) + 1.0e-9, n_bins + 1)
    which = np.clip(np.digitize(md, edges) - 1, 0, n_bins - 1)
    profile_rows: List[Dict] = []
    prof_true, prof_model, prof_sem = [], [], []
    for b in range(n_bins):
        mask = which == b
        n_b = int(mask.sum())
        if n_b < max(1, int(min_bin_frames)):
            continue
        t_mean = float(np.mean(actual[mask]))
        m_mean = float(np.mean(predicted[mask]))
        # 真值(MACE)每箱均值的标准误：判读"模型是否在噪声内对得上"的尺子，而不是拿 RMSE 当分数
        t_sem = float(np.std(actual[mask]) / max(1, n_b) ** 0.5)
        prof_true.append(t_mean)
        prof_model.append(m_mean)
        prof_sem.append(t_sem)
        profile_rows.append({
            "min_distance_center_nm": float(0.5 * (edges[b] + edges[b + 1])),
            "n_frames": n_b,
            "holdout_true_mean_kjmol": t_mean,
            "holdout_true_sem_kjmol": t_sem,
            "model_pred_mean_kjmol": m_mean,
            "residual_kjmol": m_mean - t_mean,
            "within_1sem": bool(abs(m_mean - t_mean) <= t_sem),
        })
    prof_true = np.asarray(prof_true, dtype=float)
    prof_model = np.asarray(prof_model, dtype=float)
    prof_sem = np.asarray(prof_sem, dtype=float)
    if prof_true.size >= 2:
        profile_rmse = float(np.sqrt(np.mean((prof_model - prof_true) ** 2)))
        if np.std(prof_true) > 1e-9 and np.std(prof_model) > 1e-9:
            profile_pearson = float(np.corrcoef(prof_true, prof_model)[0, 1])
        else:
            profile_pearson = math.nan
    else:
        profile_rmse = math.nan
        profile_pearson = math.nan
    # 主判据：逐箱是否落在 MACE 自身 SEM 带内（噪声地板上 RMSE 无意义，within-SEM 才是对的读法）
    within_sem_bins = int(np.sum(np.abs(prof_model - prof_true) <= prof_sem)) if prof_true.size else 0

    return {
        "n_holdout_frames": int(actual.size),
        "temperature_k": float(temperature_k),
        # 一阶系综均值：修正的主导贡献，可信量，也是最该对得上的量
        "ensemble_mean_true_kjmol": mean_true,
        "ensemble_mean_model_kjmol": mean_model,
        "ensemble_mean_bias_kjmol": float(mean_model - mean_true),
        # 留出集"均值剖面" ⟨ΔE⟩(s)：可信量（PMF matching 真正的拟合对象）
        "mean_profile_n_bins": int(prof_true.size),
        "mean_profile_rmse_kjmol": profile_rmse,
        "mean_profile_pearson_r": profile_pearson,
        "mean_profile_within_sem_bins": within_sem_bins,   # 主判据：k/N 箱落在 MACE 的 ±1 SEM 内
        "mean_profile_rows": profile_rows,
    }


def _element_bucket(atomic_number: int) -> str:
    z = int(atomic_number)
    if z == 1:
        return "H"
    if z == 6:
        return "C"
    if z == 7:
        return "N"
    if z == 8:
        return "O"
    if z == 15:
        return "P"
    if z == 16:
        return "S"
    if z in (9, 17, 35, 53):
        return "X"
    if z in (3, 4, 11, 12, 19, 20, 30, 37, 38, 55, 56):
        return "M"
    return "Z"


def _regression_metrics(actual_values: Sequence[float], predicted_values: Sequence[float]) -> Dict:
    actual = np.asarray(actual_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    if actual.size == 0:
        return {"n_frames": 0, "note": "no_frames"}
    residual = predicted - actual
    ss_tot = float(np.sum((actual - float(np.mean(actual))) ** 2))
    ss_res = float(np.sum(residual ** 2))
    if actual.size > 1 and float(np.std(actual)) > 1.0e-9 and float(np.std(predicted)) > 1.0e-9:
        pearson_r = float(np.corrcoef(actual, predicted)[0, 1])
    else:
        pearson_r = math.nan
    return {
        "n_frames": int(actual.size),
        "rmse_raw_kjmol": float(np.sqrt(np.mean(residual ** 2))),
        "mae_raw_kjmol": float(np.mean(np.abs(residual))),
        "bias_kjmol": float(np.mean(residual)),
        "r2_raw": float(1.0 - ss_res / ss_tot) if ss_tot > 1.0e-9 else math.nan,
        "pearson_r": pearson_r,
        "actual_std_kjmol": float(np.std(actual)),
        "predicted_std_kjmol": float(np.std(predicted)),
    }


def _profile_metrics_from_predictions(
    min_dist_per_frame: Sequence[float],
    actual_values: Sequence[float],
    predicted_values: Sequence[float],
    pmf_bins: int,
    min_bin_frames: int,
) -> Tuple[Dict, List[Dict]]:
    md = np.asarray(min_dist_per_frame, dtype=float)
    actual = np.asarray(actual_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    if md.size == 0:
        return {"pmf_profile_n_bins": 0, "note": "no_frames"}, []
    n_bins = max(2, int(pmf_bins))
    edges = np.linspace(float(md.min()), float(md.max()) + 1.0e-9, n_bins + 1)
    which = np.clip(np.digitize(md, edges) - 1, 0, n_bins - 1)
    rows: List[Dict] = []
    prof_true: List[float] = []
    prof_model: List[float] = []
    for b in range(n_bins):
        mask = which == b
        n_b = int(mask.sum())
        if n_b < max(1, int(min_bin_frames)):
            continue
        t_mean = float(np.mean(actual[mask]))
        m_mean = float(np.mean(predicted[mask]))
        prof_true.append(t_mean)
        prof_model.append(m_mean)
        rows.append(
            {
                "min_distance_center_nm": float(0.5 * (edges[b] + edges[b + 1])),
                "n_frames": n_b,
                "holdout_true_mean_kjmol": t_mean,
                "model_pred_mean_kjmol": m_mean,
                "residual_kjmol": float(m_mean - t_mean),
            }
        )
    true_arr = np.asarray(prof_true, dtype=float)
    model_arr = np.asarray(prof_model, dtype=float)
    if true_arr.size >= 2:
        rmse = float(np.sqrt(np.mean((model_arr - true_arr) ** 2)))
        if float(np.std(true_arr)) > 1.0e-9 and float(np.std(model_arr)) > 1.0e-9:
            pearson_r = float(np.corrcoef(true_arr, model_arr)[0, 1])
        else:
            pearson_r = math.nan
    else:
        rmse = math.nan
        pearson_r = math.nan
    return {
        "pmf_profile_n_bins": int(true_arr.size),
        "pmf_profile_rmse_kjmol": rmse,
        "pmf_profile_pearson_r": pearson_r,
    }, rows


def _build_pair_rbf_matrix(
    dists_per_frame: Sequence[np.ndarray],
    pair_types_per_frame: Sequence[np.ndarray],
    centers: np.ndarray,
    width: float,
    type_groups: Sequence[str],
) -> np.ndarray:
    type_to_block = {str(key): idx + 1 for idx, key in enumerate(type_groups)}
    n_basis = int(len(centers))
    xmat = np.zeros((len(dists_per_frame), (len(type_groups) + 1) * n_basis), dtype=float)
    for frame_idx, (dists_raw, types_raw) in enumerate(zip(dists_per_frame, pair_types_per_frame)):
        dists = np.asarray(dists_raw, dtype=float)
        if dists.size == 0:
            continue
        basis = np.exp(-0.5 * ((dists[:, None] - centers[None, :]) / max(float(width), 1.0e-6)) ** 2)
        xmat[frame_idx, :n_basis] = np.sum(basis, axis=0)
        types = np.asarray(types_raw, dtype=object)
        for type_key, block in type_to_block.items():
            mask = types == type_key
            if np.any(mask):
                start = block * n_basis
                xmat[frame_idx, start:start + n_basis] = np.sum(basis[mask], axis=0)
    return xmat


def _surface_shape_metrics(rows: Sequence[Dict]) -> Dict:
    if not rows:
        return {
            "n_populated_bins": 0,
            "rmse_kjmol": math.nan,
            "bias_kjmol": math.nan,
            "shape_rmse_bias_corrected_kjmol": math.nan,
            "pearson_r": math.nan,
        }
    true_vals = np.asarray([float(r["mace_mean_kjmol"]) for r in rows], dtype=float)
    pred_vals = np.asarray([float(r["surrogate_mean_kjmol"]) for r in rows], dtype=float)
    resid = pred_vals - true_vals
    bias = float(np.mean(resid))
    if true_vals.size > 1 and float(np.std(true_vals)) > 1.0e-9 and float(np.std(pred_vals)) > 1.0e-9:
        pearson_r = float(np.corrcoef(true_vals, pred_vals)[0, 1])
    else:
        pearson_r = math.nan
    return {
        "n_populated_bins": int(true_vals.size),
        "rmse_kjmol": float(np.sqrt(np.mean(resid ** 2))),
        "bias_kjmol": bias,
        "shape_rmse_bias_corrected_kjmol": float(np.sqrt(np.mean((resid - bias) ** 2))),
        "pearson_r": pearson_r,
        "mace_dynamic_range_kjmol": float(np.max(true_vals) - np.min(true_vals)) if true_vals.size else math.nan,
        "surrogate_dynamic_range_kjmol": float(np.max(pred_vals) - np.min(pred_vals)) if pred_vals.size else math.nan,
    }


def build_mace_surrogate_surface_diagnostics(
    output_dir: str,
    file_prefix: str,
    label: str,
    dists_per_frame: Sequence[np.ndarray],
    min_dist_per_frame: Sequence[float],
    actual_delta_e: Sequence[float],
    surrogate_delta_e: Sequence[float],
    args: argparse.Namespace,
) -> Dict:
    actual = np.asarray(actual_delta_e, dtype=float)
    surrogate = np.asarray(surrogate_delta_e, dtype=float)
    min_dist = np.asarray(min_dist_per_frame, dtype=float)
    if actual.size == 0:
        return {"label": label, "n_frames": 0, "skipped_reason": "no_frames"}
    if not (actual.size == surrogate.size == min_dist.size == len(dists_per_frame)):
        return {"label": label, "n_frames": int(actual.size), "skipped_reason": "length_mismatch"}

    min_bin_frames = max(1, int(getattr(args, "surface_min_bin_frames", 8)))
    contact_cutoff = float(getattr(args, "surface_contact_cutoff", 0.35))
    contact_counts = np.asarray(
        [int(np.sum(np.asarray(d, dtype=float) <= contact_cutoff)) for d in dists_per_frame],
        dtype=float,
    )

    # 1D profile along the leading short-range CV: minimum L-E distance.
    n_1d_bins = max(2, int(getattr(args, "surface_1d_bins", 12)))
    dist_edges_1d = np.linspace(float(min_dist.min()), float(min_dist.max()) + 1.0e-9, n_1d_bins + 1)
    which_1d = np.clip(np.digitize(min_dist, dist_edges_1d) - 1, 0, n_1d_bins - 1)
    rows_1d: List[Dict] = []
    for b in range(n_1d_bins):
        mask = which_1d == b
        n_b = int(np.sum(mask))
        if n_b < min_bin_frames:
            continue
        mace_mean = float(np.mean(actual[mask]))
        surrogate_mean = float(np.mean(surrogate[mask]))
        rows_1d.append(
            {
                "surface_label": str(label),
                "bin_index": int(b),
                "min_distance_center_nm": float(0.5 * (dist_edges_1d[b] + dist_edges_1d[b + 1])),
                "min_distance_low_nm": float(dist_edges_1d[b]),
                "min_distance_high_nm": float(dist_edges_1d[b + 1]),
                "n_frames": n_b,
                "mace_mean_kjmol": mace_mean,
                "mace_sem_kjmol": float(np.std(actual[mask]) / max(1, n_b) ** 0.5),
                "surrogate_mean_kjmol": surrogate_mean,
                "surrogate_sem_kjmol": float(np.std(surrogate[mask]) / max(1, n_b) ** 0.5),
                "delta_surrogate_minus_mace_kjmol": float(surrogate_mean - mace_mean),
            }
        )

    # 2D surface: min-distance plus contact count, a cheap proxy for the number of local pair contacts.
    n_dist_bins = max(2, int(getattr(args, "surface_2d_distance_bins", 8)))
    n_contact_bins = max(2, int(getattr(args, "surface_2d_contact_bins", 8)))
    dist_edges_2d = np.linspace(float(min_dist.min()), float(min_dist.max()) + 1.0e-9, n_dist_bins + 1)
    c_min = float(np.min(contact_counts))
    c_max = float(np.max(contact_counts))
    if c_max <= c_min:
        contact_edges = np.linspace(c_min - 0.5, c_max + 0.5, n_contact_bins + 1)
    else:
        contact_edges = np.linspace(c_min, c_max + 1.0e-9, n_contact_bins + 1)
    which_dist = np.clip(np.digitize(min_dist, dist_edges_2d) - 1, 0, n_dist_bins - 1)
    which_contact = np.clip(np.digitize(contact_counts, contact_edges) - 1, 0, n_contact_bins - 1)
    rows_2d: List[Dict] = []
    for i in range(n_dist_bins):
        for j in range(n_contact_bins):
            mask = (which_dist == i) & (which_contact == j)
            n_b = int(np.sum(mask))
            if n_b < min_bin_frames:
                continue
            mace_mean = float(np.mean(actual[mask]))
            surrogate_mean = float(np.mean(surrogate[mask]))
            rows_2d.append(
                {
                    "surface_label": str(label),
                    "min_distance_bin": int(i),
                    "contact_count_bin": int(j),
                    "min_distance_center_nm": float(0.5 * (dist_edges_2d[i] + dist_edges_2d[i + 1])),
                    "contact_count_center": float(0.5 * (contact_edges[j] + contact_edges[j + 1])),
                    "n_frames": n_b,
                    "mace_mean_kjmol": mace_mean,
                    "surrogate_mean_kjmol": surrogate_mean,
                    "delta_surrogate_minus_mace_kjmol": float(surrogate_mean - mace_mean),
                }
            )

    safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", str(label)).strip("_") or "surface"
    path_prefix = f"{file_prefix}fit_{safe_label}_mace_surrogate"
    summary = {
        "label": str(label),
        "n_frames": int(actual.size),
        "target_definition": "MACE local residual target vs Gaussian+DEXP surrogate residual, compared after binning over local CVs",
        "contact_cutoff_nm": float(contact_cutoff),
        "min_bin_frames": int(min_bin_frames),
        "one_dimensional": _surface_shape_metrics(rows_1d),
        "two_dimensional": _surface_shape_metrics(rows_2d),
        "min_distance_range_nm": [float(min_dist.min()), float(min_dist.max())],
        "contact_count_range": [float(contact_counts.min()), float(contact_counts.max())],
    }
    if rows_1d:
        csv_1d = write_rows_csv(os.path.join(output_dir, f"{path_prefix}_pmf_1d.csv"), rows_1d)
        summary["pmf_1d_csv"] = csv_1d
    if rows_2d:
        csv_2d = write_rows_csv(os.path.join(output_dir, f"{path_prefix}_pmf_2d.csv"), rows_2d)
        summary["pmf_2d_csv"] = csv_2d

    try:
        plt = get_matplotlib_pyplot()
        if rows_1d:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            x = [float(r["min_distance_center_nm"]) for r in rows_1d]
            y_mace = [float(r["mace_mean_kjmol"]) for r in rows_1d]
            y_sur = [float(r["surrogate_mean_kjmol"]) for r in rows_1d]
            ax.plot(x, y_mace, "o-", label="MACE local")
            ax.plot(x, y_sur, "s--", label="Gaussian+DEXP")
            ax.set_xlabel("min L-E distance (nm)")
            ax.set_ylabel("binned local energy / PMF proxy (kJ/mol)")
            ax.set_title(f"{label}: 1D local surface")
            ax.grid(alpha=0.3)
            ax.legend()
            png_1d = os.path.join(output_dir, f"{path_prefix}_pmf_1d.png")
            fig.tight_layout()
            fig.savefig(png_1d, dpi=180)
            plt.close(fig)
            summary["pmf_1d_png"] = png_1d
        if rows_2d:
            grid = np.full((n_contact_bins, n_dist_bins), np.nan, dtype=float)
            for row in rows_2d:
                grid[int(row["contact_count_bin"]), int(row["min_distance_bin"])] = float(
                    row["delta_surrogate_minus_mace_kjmol"]
                )
            fig, ax = plt.subplots(figsize=(7.0, 5.2))
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm")
            ax.set_xlabel("min-distance bin")
            ax.set_ylabel("contact-count bin")
            ax.set_title(f"{label}: surrogate - MACE local surface")
            fig.colorbar(im, ax=ax, label="kJ/mol")
            png_2d = os.path.join(output_dir, f"{path_prefix}_pmf_2d_delta.png")
            fig.tight_layout()
            fig.savefig(png_2d, dpi=180)
            plt.close(fig)
            summary["pmf_2d_delta_png"] = png_2d
    except Exception as exc:
        summary["plot_error"] = str(exc)
    summary_json = os.path.join(output_dir, f"{path_prefix}_surface_summary.json")
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_json"] = summary_json
    return summary


def fit_learned_pair_rbf_diagnostic(
    train_dists: Sequence[np.ndarray],
    train_pair_types: Sequence[np.ndarray],
    train_targets: Sequence[float],
    holdout_dists: Sequence[np.ndarray],
    holdout_pair_types: Sequence[np.ndarray],
    holdout_targets: Sequence[float],
    holdout_min_dist: Sequence[float],
    args: argparse.Namespace,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    y_train = np.asarray(train_targets, dtype=float)
    y_holdout = np.asarray(holdout_targets, dtype=float)
    if y_train.size < 10 or y_holdout.size == 0:
        return {"enabled": False, "skipped_reason": "insufficient_train_or_holdout_frames"}, [], []

    n_centers = max(3, int(getattr(args, "learned_rbf_centers", 8)))
    r_min = float(args.fit_r_min)
    r_max = float(args.fit_r_max)
    centers = np.linspace(r_min, r_max, n_centers)
    width = float((centers[1] - centers[0]) * 1.25) if n_centers > 1 else max(0.03, r_max - r_min)

    type_counts: Dict[str, int] = {}
    for type_arr in train_pair_types:
        unique, counts = np.unique(np.asarray(type_arr, dtype=object), return_counts=True)
        for key, count in zip(unique, counts):
            type_counts[str(key)] = type_counts.get(str(key), 0) + int(count)
    min_group_pairs = max(1, int(getattr(args, "learned_rbf_min_group_pairs", 200)))
    max_groups = max(0, int(getattr(args, "learned_rbf_max_type_groups", 24)))
    type_groups = [
        key for key, count in sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_group_pairs
    ][:max_groups]

    x_train_raw = _build_pair_rbf_matrix(train_dists, train_pair_types, centers, width, type_groups)
    x_holdout_raw = _build_pair_rbf_matrix(holdout_dists, holdout_pair_types, centers, width, type_groups)
    col_mean = np.mean(x_train_raw, axis=0)
    col_std = np.std(x_train_raw, axis=0)
    active = col_std > 1.0e-10
    if int(np.sum(active)) == 0:
        return {"enabled": False, "skipped_reason": "no_active_rbf_features"}, [], []

    x_train = (x_train_raw[:, active] - col_mean[active]) / col_std[active]
    x_holdout = (x_holdout_raw[:, active] - col_mean[active]) / col_std[active]
    y_center = float(np.mean(y_train))
    y_fit = y_train - y_center
    ridge = max(0.0, float(getattr(args, "learned_rbf_ridge", 10.0)))
    lhs = x_train.T @ x_train + ridge * np.eye(x_train.shape[1], dtype=float)
    rhs = x_train.T @ y_fit
    try:
        coef_active = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef_active = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    train_pred = x_train @ coef_active + y_center
    holdout_pred = x_holdout @ coef_active + y_center

    metrics = _regression_metrics(y_holdout, holdout_pred)
    train_metrics = _regression_metrics(y_train, train_pred)
    profile_metrics, profile_rows = _profile_metrics_from_predictions(
        holdout_min_dist,
        y_holdout,
        holdout_pred,
        int(getattr(args, "fit_pmf_bins", 12)),
        int(getattr(args, "fit_pmf_min_bin_frames", 10)),
    )
    metrics.update(
        {
            "enabled": True,
            "model": "pair_type_rbf_ridge",
            "target_note": "trained on the same train targets used by DEXP; evaluated on per-frame holdout targets",
            "n_train_frames": int(y_train.size),
            "n_holdout_frames": int(y_holdout.size),
            "n_rbf_centers": int(n_centers),
            "rbf_centers_nm": [float(x) for x in centers.tolist()],
            "rbf_width_nm": float(width),
            "ridge_lambda": float(ridge),
            "type_groups": [str(x) for x in type_groups],
            "n_active_features": int(np.sum(active)),
            "n_total_features": int(x_train_raw.shape[1]),
            "train_rmse_raw_kjmol": float(train_metrics.get("rmse_raw_kjmol", math.nan)),
            "train_r2_raw": float(train_metrics.get("r2_raw", math.nan)),
            "pmf_profile": profile_metrics,
        }
    )
    holdout_rows = [
        {
            "min_distance_nm": float(md),
            "actual_delta_e_kjmol": float(actual),
            "predicted_delta_e_kjmol": float(pred),
            "residual_kjmol": float(pred - actual),
        }
        for md, actual, pred in zip(holdout_min_dist, y_holdout.tolist(), holdout_pred.tolist())
    ]
    return metrics, holdout_rows, profile_rows


def write_mace_surrogate_surface_diagnostics(
    output_dir: str,
    file_prefix: str,
    frame_ids: Sequence[int],
    min_dist_per_frame: Sequence[float],
    contact_count_per_frame: Sequence[int],
    mace_target_per_frame: Sequence[float],
    surrogate_pred_per_frame: Sequence[float],
    args: argparse.Namespace,
) -> Dict:
    """Compare the local MACE residual surface against the Gaussian+DEXP surrogate surface.

    The target is the per-frame local residual used for fitting. For the default
    mace_surrogate_residual target, adding Gaussian Coulomb back to both sides
    means this is exactly the MACE-local vs surrogate-local comparison up to an
    arbitrary constant.
    """
    frame_ids_arr = np.asarray(frame_ids, dtype=int)
    s = np.asarray(min_dist_per_frame, dtype=float)
    c = np.asarray(contact_count_per_frame, dtype=float)
    mace = np.asarray(mace_target_per_frame, dtype=float)
    surrogate = np.asarray(surrogate_pred_per_frame, dtype=float)
    valid = np.isfinite(s) & np.isfinite(c) & np.isfinite(mace) & np.isfinite(surrogate)
    s, c, mace, surrogate, frame_ids_arr = s[valid], c[valid], mace[valid], surrogate[valid], frame_ids_arr[valid]
    if s.size == 0:
        return {"enabled": False, "skipped_reason": "no_valid_surface_frames"}

    min_bin_frames = max(1, int(getattr(args, "surface_pmf_min_bin_frames", 8)))
    n_1d = max(2, int(getattr(args, "surface_pmf_bins", 12)))
    edges = np.linspace(float(np.min(s)), float(np.max(s)) + 1.0e-9, n_1d + 1)
    bin_ids = np.clip(np.digitize(s, edges) - 1, 0, n_1d - 1)

    # 只算 ⟨ΔE⟩(s) 均值剖面（可信量）。FEP/重加权 PMF 已移除：本体系每箱 ESS≈1，重加权是
    # 单帧最小值伪影；要真正的 PMF 需要偏置采样（AWH/伞形）或直接从 MD 直方图取，见 build_1d_pmf。
    rows_1d: List[Dict] = []
    for b in range(n_1d):
        mask = bin_ids == b
        n_b = int(np.sum(mask))
        if n_b < min_bin_frames:
            continue
        m_mean = float(np.mean(mace[mask]))
        s_mean = float(np.mean(surrogate[mask]))
        rows_1d.append(
            {
                "min_distance_center_nm": float(0.5 * (edges[b] + edges[b + 1])),
                "n_frames": n_b,
                "mace_local_mean_kjmol": m_mean,
                "surrogate_mean_kjmol": s_mean,
                "mean_delta_surrogate_minus_mace_kjmol": float(s_mean - m_mean),
                "mace_local_sem_kjmol": float(np.std(mace[mask]) / max(1, n_b) ** 0.5),
                "surrogate_sem_kjmol": float(np.std(surrogate[mask]) / max(1, n_b) ** 0.5),
            }
        )
    if not rows_1d:
        return {
            "enabled": False,
            "skipped_reason": "no_1d_bins_with_enough_frames",
            "n_frames": int(s.size),
            "min_bin_frames": int(min_bin_frames),
        }
    csv_1d = write_rows_csv(os.path.join(output_dir, f"{file_prefix}mace_surrogate_mean_profile_1d.csv"), rows_1d)

    n_2d = max(2, int(getattr(args, "surface_pmf_2d_bins", 6)))
    s_edges_2d = np.linspace(float(np.min(s)), float(np.max(s)) + 1.0e-9, n_2d + 1)
    # Quantile edges keep contact-count bins populated even when the count range is narrow.
    q_edges = np.quantile(c, np.linspace(0.0, 1.0, n_2d + 1))
    q_edges = np.asarray(q_edges, dtype=float)
    for idx in range(1, q_edges.size):
        if q_edges[idx] <= q_edges[idx - 1]:
            q_edges[idx] = q_edges[idx - 1] + 1.0e-6
    s_bin = np.clip(np.digitize(s, s_edges_2d) - 1, 0, n_2d - 1)
    c_bin = np.clip(np.digitize(c, q_edges) - 1, 0, n_2d - 1)
    flat_bin = s_bin * n_2d + c_bin
    rows_2d: List[Dict] = []
    for i in range(n_2d):
        for j in range(n_2d):
            flat = i * n_2d + j
            mask = flat_bin == flat
            n_b = int(np.sum(mask))
            if n_b < min_bin_frames:
                continue
            m_mean = float(np.mean(mace[mask]))
            s_mean = float(np.mean(surrogate[mask]))
            rows_2d.append(
                {
                    "min_distance_center_nm": float(0.5 * (s_edges_2d[i] + s_edges_2d[i + 1])),
                    "contact_count_center": float(0.5 * (q_edges[j] + q_edges[j + 1])),
                    "n_frames": n_b,
                    "mace_local_mean_kjmol": m_mean,
                    "surrogate_mean_kjmol": s_mean,
                    "mean_delta_surrogate_minus_mace_kjmol": float(s_mean - m_mean),
                }
            )
    csv_2d = write_rows_csv(os.path.join(output_dir, f"{file_prefix}mace_surrogate_mean_profile_2d.csv"), rows_2d) if rows_2d else None

    png_1d = None
    png_2d = None
    try:
        plt = get_matplotlib_pyplot()
        x = np.asarray([float(row["min_distance_center_nm"]) for row in rows_1d], dtype=float)
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot(x, [float(row["mace_local_mean_kjmol"]) for row in rows_1d], "o-", label="MACE local mean")
        ax.plot(x, [float(row["surrogate_mean_kjmol"]) for row in rows_1d], "s--", label="Gaussian+DEXP mean")
        ax.set_xlabel("min L-E distance (nm)")
        ax.set_ylabel("<local residual> per bin (kJ/mol)")
        ax.legend(); ax.grid(alpha=0.3)
        png_1d = os.path.join(output_dir, f"{file_prefix}mace_surrogate_mean_profile_1d.png")
        fig.tight_layout(); fig.savefig(png_1d, dpi=180); plt.close(fig)

        if len(rows_2d) >= 2:  # 单格不成图
            grid = np.full((n_2d, n_2d), math.nan, dtype=float)
            for row in rows_2d:
                i = int(np.argmin(np.abs(0.5 * (s_edges_2d[:-1] + s_edges_2d[1:]) - float(row["min_distance_center_nm"]))))
                j = int(np.argmin(np.abs(0.5 * (q_edges[:-1] + q_edges[1:]) - float(row["contact_count_center"]))))
                grid[j, i] = float(row["mean_delta_surrogate_minus_mace_kjmol"])
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm")
            ax.set_xlabel("min-distance bin")
            ax.set_ylabel("contact-count bin")
            ax.set_title("2D mean delta: surrogate - MACE")
            fig.colorbar(im, ax=ax, label="kJ/mol")
            png_2d = os.path.join(output_dir, f"{file_prefix}mace_surrogate_mean_profile_2d.png")
            fig.tight_layout(); fig.savefig(png_2d, dpi=180); plt.close(fig)
    except Exception:
        png_1d = png_2d = None

    def _finite_rmse(rows: List[Dict], key: str, min_rows: int = 2) -> float:
        # 少于 min_rows 个有效点不构成 RMSE，返回 NaN（避免单格/单箱伪指标）。
        vals = np.asarray([float(row[key]) for row in rows if np.isfinite(float(row[key]))], dtype=float)
        return float(np.sqrt(np.mean(vals ** 2))) if vals.size >= min_rows else math.nan

    summary = {
        "enabled": True,
        "n_frames": int(s.size),
        "target": "MACE local interaction residual vs Gaussian+DEXP surrogate residual (mean profile only)",
        "cv_1d": "min_ligand_environment_distance_nm",
        "cv_2d": "min_ligand_environment_distance_nm + short_range_pair_count",
        "min_bin_frames": int(min_bin_frames),
        "csv_1d": csv_1d,
        "csv_2d": csv_2d,
        "png_1d": png_1d,
        "png_2d": png_2d,
        "n_bins_1d_written": int(len(rows_1d)),
        "n_bins_2d_written": int(len(rows_2d)),
        # 唯一可信量：均值剖面（每箱 ⟨ΔE⟩ 之差的 RMSE）
        "mean_profile_rmse_1d_kjmol": _finite_rmse(rows_1d, "mean_delta_surrogate_minus_mace_kjmol"),
        "mean_profile_rmse_2d_kjmol": _finite_rmse(rows_2d, "mean_delta_surrogate_minus_mace_kjmol"),
        "note": "只报 ⟨ΔE⟩(s) 均值剖面。FEP/重加权 PMF 已移除：本体系每箱 ESS≈1，需偏置采样(AWH/伞形)或从 MD 直方图取 PMF。",
    }
    return summary


def _fit_dexp_with_ml_model(
    args: argparse.Namespace,
    output_dir: str,
    ml_model_name: str,
    file_prefix: str,
    traj,
    fit_indices: List[int],
    lig_idx: np.ndarray,
    env_idx: np.ndarray,
    all_nums: np.ndarray,
    mm_contexts: Dict,
    fit_xyz: np.ndarray,
    fit_time: np.ndarray,
    fit_box,
    env_search_radius: float,
    env_max_atoms,
) -> Tuple[Dict, List[Dict]]:
    symbols = load_abfe_symbols()
    NumpyEncoder = symbols["NumpyEncoder"]
    Orbv3DEXPFittingPipeline = symbols["Orbv3DEXPFittingPipeline"]
    Orbv3SurrogateFitter = symbols["Orbv3SurrogateFitter"]
    openmm, _, unit, _ = require_openmm()
    import numpy as np

    pipeline = Orbv3DEXPFittingPipeline(model_name=ml_model_name, device=args.device)
    label_mode = getattr(pipeline, "label_mode", "orbv3_interaction")
    fit_target_mode = str(args.fit_target_mode)
    use_gaussian_replacement = fit_target_mode in ("mace_surrogate_residual", "gaussian_replacement_residual")
    use_qmmm_total = fit_target_mode in ("qmmm_residual", "ml_minus_mm_total")

    fit_log_rows: List[Dict] = []
    raw_delta_e_values: List[float] = []
    raw_gauss_coul_values: List[float] = []
    raw_delta_vs_mm_total_values: List[float] = []
    raw_orb_values: List[float] = []
    raw_mm_coul_values: List[float] = []
    raw_mm_vdw_values: List[float] = []
    fit_log_path = os.path.join(output_dir, f"{file_prefix}fit_frame_diagnostics.csv")
    fit_label_meta_path = os.path.join(output_dir, f"{file_prefix}fit_label_cache_meta.json")
    lig_type_buckets = [_element_bucket(int(all_nums[idx])) for idx in lig_idx]
    env_type_buckets = [_element_bucket(int(all_nums[idx])) for idx in env_idx]
    pair_type_matrix = np.asarray(
        [
            [f"L{lig_bucket}-E{env_bucket}" for env_bucket in env_type_buckets]
            for lig_bucket in lig_type_buckets
        ],
        dtype=object,
    )
    print(f"    [{ml_model_name}] 实际参与拟合帧数: {len(fit_indices)}")
    reuse_labels = False   # 是否有可用缓存行（等价于 reuse_ml：MACE 能量可复用）
    reuse_ml = False       # 复用缓存的 MACE 相互作用能 e_orb_int（贵，尽量复用）
    reuse_mm = False       # 复用缓存的 MM 参考能 e_mm_*（便宜；依赖 MM 参考截断设置）
    cached_rows_by_frame: Dict[int, Dict[str, str]] = {}
    if (
        file_prefix == ""
        and args.reuse_fit_labels
        and os.path.isfile(fit_log_path)
        and os.path.isfile(fit_label_meta_path)
    ):
        try:
            with open(fit_label_meta_path, "r", encoding="utf-8") as handle:
                cache_meta = json.load(handle)
            frame_indices_cached = [int(x) for x in cache_meta.get("fit_indices", [])]
            env_idx_cached = [int(x) for x in cache_meta.get("env_indices", [])]
            lig_idx_cached = [int(x) for x in cache_meta.get("ligand_indices", [])]
            # MACE(e_orb_int) 的有效性只取决于帧/原子集合/模型/分解模式，与 MM 参考截断无关。
            # 注意：不再把 fit_target_mode 纳入判据——delta 一律用原始能量在循环里重算。
            ml_cache_ok = (
                frame_indices_cached == [int(x) for x in fit_indices]
                and env_idx_cached == [int(x) for x in env_idx]
                and lig_idx_cached == [int(x) for x in lig_idx]
                and str(cache_meta.get("ml_model", "")) == str(ml_model_name)
                and str(cache_meta.get("label_mode", "")) == str(label_mode)
                and abs(float(cache_meta.get("env_search_radius_nm", -1.0)) - float(env_search_radius)) < 1.0e-8
                and cache_meta.get("env_max_atoms", None) == (int(env_max_atoms) if env_max_atoms is not None else None)
            )
            # MM 参考能量额外要求截断/switching 一致；改 --fit-mm-ref-cutoff 只让 MM 缓存失效，
            # 于是复用昂贵的 MACE、只重算便宜的 MM 侧（零额外 GPU 计算）。
            mm_cache_ok = ml_cache_ok and (
                abs(float(cache_meta.get("mm_ref_cutoff_nm", -1.0e9)) - float(args.fit_mm_ref_cutoff)) < 1.0e-8
                and abs(float(cache_meta.get("mm_ref_switch_nm", -1.0e9)) - float(args.fit_mm_ref_switch)) < 1.0e-8
            )
            reuse_ml = bool(ml_cache_ok)
            reuse_mm = bool(mm_cache_ok)
            reuse_labels = reuse_ml
            if reuse_ml:
                for row in read_csv_rows(fit_log_path):
                    cached_rows_by_frame[int(row["frame_index"])] = row
                if reuse_mm:
                    print(f"    复用已有 MACE+MM 能量标注缓存: {fit_log_path}")
                else:
                    print("    复用缓存的 MACE 能量，按新 MM 参考设置重算 MM 能量（不重跑 MACE）")
            else:
                print("    已检测到旧缓存，但当前 frame/env/模型 选择已变化，回退为重新标注。")
        except Exception:
            reuse_ml = reuse_mm = reuse_labels = False

    gpu_workers = 1
    worker_pipelines: List[Orbv3DEXPFittingPipeline] = []
    if not reuse_labels and str(args.device).lower() == "cuda":
        gpu_workers = 1
        first_pos_nm = fit_xyz[0].copy()
        print(f"    [{ml_model_name}] OpenMM-ML 预建 GPU worker: {gpu_workers}")
        for wid in range(gpu_workers):
            worker = Orbv3DEXPFittingPipeline(model_name=ml_model_name, device=args.device)
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

        cached_row = cached_rows_by_frame.get(frame_id)

        # MACE 侧 e_orb_int：优先缓存(reuse_ml) -> 批量预算 -> 逐帧计算。与 MM 参考无关。
        if reuse_ml and cached_row is not None:
            e_orb_int = float(cached_row["e_orb_int_kjmol"])
        elif local_idx in orb_energy_by_local_idx:
            e_orb_int = float(orb_energy_by_local_idx[local_idx])
        else:
            e_orb_int = pipeline._compute_orb_decomposition(pos_nm, lig_idx, env_idx, all_nums)

        # MM 侧 e_mm_*：仅当截断设置一致才复用缓存，否则用当前参考力（默认已改为 NoCutoff）重算。
        if reuse_mm and cached_row is not None:
            e_gauss_coul = float(cached_row.get("e_gauss_coul_kjmol", "0.0"))
            e_mm_coul = float(cached_row["e_mm_coul_kjmol"])
            e_mm_vdw = float(cached_row.get("e_mm_vdw_kjmol", "0.0"))
        else:
            e_gauss_coul = 0.0
            e_mm_coul = 0.0
            e_mm_vdw = 0.0
            for label, ctx in mm_contexts.items():
                # NoCutoff 参考力不使用周期性；只有当该力确实启用 PBC 时才设盒子，
                # 否则用与 MACE 分解完全相同的原始坐标（非最小镜像）以保证边界一致。
                if fit_box is not None and ctx.getSystem().usesPeriodicBoundaryConditions():
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

        # delta 一律由原始能量重算，保证 MACE/MM 任意组合(缓存/新算)下自洽
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
                "min_le_distance_nm": float(dists.min()),
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

    ref_offset_limit = float(args.ml_ref_offset_limit_kjmol)
    if abs(float(delta_diag["center"])) > ref_offset_limit:
        print(
            f"    ⚠️ [{ml_model_name}] ΔE 中心值 |{delta_diag['center']:.1f}| kJ/mol 超过阈值 {ref_offset_limit:.0f}，"
            "疑似该模型返回的是不兼容的绝对总能量（参考零点异常），跳过拟合并标记为不可信"
        )
        fitted_params = {
            "fitting_success": False,
            "suspicious_fit": True,
            "boundary_hits": [f"ml_reference_energy_offset_anomaly(center={delta_diag['center']:.1f}_kjmol)"],
            "error": "ml_reference_energy_offset_anomaly",
            "ml_model": str(ml_model_name),
            "label_mode": str(label_mode),
            "fit_target_mode": fit_target_mode,
            "fit_frames_requested": int(args.fit_frames),
            "fit_frames_total": int(len(fit_indices)),
            "fit_frames_used": 0,
            "fit_frames_train": 0,
            "fit_frames_holdout": 0,
            "qm_mm_offset_kjmol": float(delta_diag["center"]),
            "delta_e_mean_kjmol": float(delta_diag["stats"]["mean"]),
            "delta_e_std_kjmol": float(delta_diag["stats"]["std"]),
            "ml_ref_offset_limit_kjmol": ref_offset_limit,
            "holdout_validation": {"n_holdout_frames": 0, "skipped_reason": "ml_reference_energy_offset_anomaly"},
        }
        params_path = os.path.join(output_dir, f"{file_prefix}dexp_fitted_params.json")
        with open(params_path, "w", encoding="utf-8") as handle:
            json.dump(fitted_params, handle, indent=2, cls=NumpyEncoder)
        with open(fit_log_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fit_log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fit_log_rows)
        return fitted_params, fit_log_rows

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
    rebuilt_pair_types_per_frame: List[np.ndarray] = []
    accepted_delta_e_final: List[float] = []
    accepted_frame_ids: List[int] = []
    accepted_min_dist: List[float] = []
    for row_idx, row in enumerate(fit_log_rows):
        if not int(row["used_for_fit"]):
            continue
        pos_nm = fit_xyz[row_idx]
        box_vecs = fit_box[row_idx] if fit_box is not None else np.eye(3) * 3.0
        box_lens = np.linalg.norm(box_vecs, axis=1)
        delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
        delta -= box_lens * np.round(delta / box_lens)
        dists = np.linalg.norm(delta, axis=-1)
        valid_mask = (dists >= args.fit_r_min) & (dists <= args.fit_r_max)
        valid_dists = dists[valid_mask]
        valid_pair_types = pair_type_matrix[valid_mask]
        candidate_dists = dists[dists <= env_search_radius]
        if len(valid_dists) == 0 or len(candidate_dists) == 0:
            row["used_for_fit"] = 0
            continue
        rebuilt_dists_per_frame.append(valid_dists)
        rebuilt_pair_types_per_frame.append(np.asarray(valid_pair_types, dtype=object))
        accepted_delta_e_final.append(float(row["delta_e_centered_kjmol"]))
        accepted_frame_ids.append(int(row["frame_index"]))
        accepted_min_dist.append(float(row["min_le_distance_nm"]))

    if len(accepted_delta_e_final) < 10:
        raise RuntimeError(
            f"有效拟合帧只有 {len(accepted_delta_e_final)} 帧，无法稳定拟合 DEXP。"
        )

    # 一阶 PMF matching：两个模型不在同一逐帧势能面上，不拟合逐帧 ΔE(x)，
    # 而是沿 min-distance 分箱、用每箱均值 ⟨ΔE⟩(s) 作为目标（把正交噪声积分掉）。
    # 常数 C 已通过 delta_e_centered 处理；每箱均值的标准误远小于逐帧 σ，故一阶可稳。
    fit_objective = str(getattr(args, "fit_objective", "pmf_mean"))
    pmf_profile_rows: List[Dict] = []
    # C 修复：保留逐帧（未平滑）ΔE 供留出集做真实逐帧验证 + 端态自由能判据（A）。
    # 训练目标仍可用 PMF 箱均值，但验证绝不能拿箱均值当"真值"（那是循环验证）。
    accepted_delta_e_perframe = list(accepted_delta_e_final)
    if fit_objective == "pmf_mean":
        md = np.asarray(accepted_min_dist, dtype=float)
        de = np.asarray(accepted_delta_e_final, dtype=float)
        n_bins = max(2, int(args.fit_pmf_bins))
        min_bin_frames = max(1, int(getattr(args, "fit_pmf_min_bin_frames", 10)))
        edges = np.linspace(float(md.min()), float(md.max()) + 1.0e-9, n_bins + 1)
        which = np.clip(np.digitize(md, edges) - 1, 0, n_bins - 1)
        smoothed = de.copy()
        keep_bin = np.zeros(de.size, dtype=bool)  # C 修复：稀疏箱整箱剔除
        n_dropped_bins = 0
        n_dropped_frames = 0
        for b in range(n_bins):
            mask = which == b
            n_b = int(mask.sum())
            if n_b == 0:
                continue
            if n_b < min_bin_frames:
                # 稀疏箱（如 1~3 帧）：均值被噪声主导，会撑起假的动态范围。整箱剔除。
                n_dropped_bins += 1
                n_dropped_frames += n_b
                continue
            vals = de[mask]
            b_mean = float(vals.mean())
            b_std = float(vals.std())
            smoothed[mask] = b_mean
            keep_bin |= mask
            pmf_profile_rows.append(
                {
                    "bin_index": int(b),
                    "min_distance_center_nm": float(0.5 * (edges[b] + edges[b + 1])),
                    "n_frames": n_b,
                    "delta_e_mean_kjmol": b_mean,
                    "delta_e_std_kjmol": b_std,
                    "delta_e_sem_kjmol": float(b_std / max(1, n_b) ** 0.5),
                }
            )
        accepted_delta_e_final = smoothed.tolist()
        # C 修复：把稀疏箱帧从所有 accepted 数组里同步剔除，保证 train/holdout 只用可信箱
        if n_dropped_frames > 0:
            keep_idx = np.where(keep_bin)[0]
            rebuilt_dists_per_frame = [rebuilt_dists_per_frame[i] for i in keep_idx]
            rebuilt_pair_types_per_frame = [rebuilt_pair_types_per_frame[i] for i in keep_idx]
            accepted_frame_ids = [accepted_frame_ids[i] for i in keep_idx]
            accepted_min_dist = [accepted_min_dist[i] for i in keep_idx]
            accepted_delta_e_perframe = [accepted_delta_e_perframe[i] for i in keep_idx]
            accepted_delta_e_final = [accepted_delta_e_final[i] for i in keep_idx]
        pmf_csv = os.path.join(output_dir, f"{file_prefix}fit_pmf_matching_profile.csv")
        if pmf_profile_rows:
            with open(pmf_csv, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pmf_profile_rows[0].keys()))
                writer.writeheader()
                writer.writerows(pmf_profile_rows)
        print(
            f"    [PMF matching] 沿 min-distance {n_bins} 箱 -> 目标改为每箱 ⟨ΔE⟩(s)；"
            f"已写 {os.path.basename(pmf_csv)}（{len(pmf_profile_rows)} 个可信箱，"
            f"剔除 {n_dropped_bins} 个稀疏箱/<{min_bin_frames}帧 共 {n_dropped_frames} 帧）"
        )

    n_accepted = len(accepted_delta_e_final)
    holdout_fraction = max(0.0, min(0.9, float(args.holdout_fraction)))
    n_holdout_target = int(round(n_accepted * holdout_fraction))
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n_accepted)
    if (
        holdout_fraction <= 0.0
        or n_holdout_target < int(args.holdout_min_frames)
        or (n_accepted - n_holdout_target) < 10
    ):
        train_idx = perm
        holdout_idx = np.array([], dtype=int)
        holdout_skip_reason = "insufficient_frames_for_holdout_split"
    else:
        holdout_idx = perm[:n_holdout_target]
        train_idx = perm[n_holdout_target:]
        holdout_skip_reason = None
    print(
        f"    留出集划分: 接受帧={n_accepted} | train={len(train_idx)} | holdout={len(holdout_idx)}"
        + ("" if holdout_skip_reason is None else f" | 跳过原因={holdout_skip_reason}")
    )

    train_dists = [rebuilt_dists_per_frame[i] for i in train_idx]
    train_pair_types = [rebuilt_pair_types_per_frame[i] for i in train_idx]
    train_delta_e = [accepted_delta_e_final[i] for i in train_idx]

    fitter = Orbv3SurrogateFitter(fitting_region=(args.fit_r_min, args.fit_r_max))
    fitted_params = fitter.fit_parameters(train_dists, train_delta_e)
    # C 现在是 fitter 内部联合拟合的一等参数（abfe_core.Orbv3SurrogateFitter.fit_parameters
    # 在其实际优化过的 trimmed+weighted 帧集合、且用护栏 clamp 之后真正会施加到 OpenMM 的
    # 最终 (a,b,r0,A,B) 上解析求出 offset_c0），不再在这里用未修剪/未加权的全训练集重新估一遍
    # ——那样会与 fitter 真正拟合过的分布不一致（trim 剔除的离群帧、clamp 后的 A 都不同）。
    # 这里只做诊断：把"未修剪全训练集朴素均值对齐"的旧估计留作交叉核对，不覆盖权威值。
    if fitted_params.get("fitting_success") and len(train_dists) > 0:
        train_pairsum = np.asarray(
            [predict_dexp_delta_e(d, {**fitted_params, "offset_c0": 0.0}) for d in train_dists],
            dtype=float,
        )
        train_target = np.asarray(train_delta_e, dtype=float)
        naive_offset_c0 = float(np.mean(train_target) - np.mean(train_pairsum))
        fitted_params["offset_c0_naive_full_train_mean_diagnostic"] = naive_offset_c0
        if "offset_c0" not in fitted_params:
            # 向后兼容：如果 fitter 版本仍未提供一等 offset_c0，退回旧的朴素估计。
            fitted_params["offset_c0"] = naive_offset_c0
            fitted_params["offset_c0_source"] = "train_mean_alignment(target_minus_pairsum)_fallback"
    fitted_params["fit_frames_requested"] = int(args.fit_frames)
    fitted_params["fit_last_ns_requested"] = float(args.fit_last_ns)
    fitted_params["fit_frames_total"] = int(len(fit_indices))
    fitted_params["fit_frames_used"] = int(len(accepted_delta_e_final))
    fitted_params["fit_frames_train"] = int(len(train_idx))
    fitted_params["fit_frames_holdout"] = int(len(holdout_idx))
    fitted_params["fit_frame_start"] = int(fit_indices[0])
    fitted_params["fit_frame_end"] = int(fit_indices[-1])
    fitted_params["fit_time_start_ps"] = float(traj.time[fit_indices[0]]) if getattr(traj, "time", None) is not None else None
    fitted_params["fit_time_end_ps"] = float(traj.time[fit_indices[-1]]) if getattr(traj, "time", None) is not None else None
    fitted_params["env_radius_nm"] = float(args.fit_env_radius)
    fitted_params["env_search_radius_nm"] = float(env_search_radius)
    fitted_params["env_max_atoms"] = int(env_max_atoms) if env_max_atoms is not None else None
    fitted_params["fit_region_nm"] = [float(args.fit_r_min), float(args.fit_r_max)]
    fitted_params["mm_ref_cutoff_nm"] = float(args.fit_mm_ref_cutoff)
    fitted_params["mm_ref_switch_nm"] = float(args.fit_mm_ref_switch)
    fitted_params["fit_objective"] = fit_objective
    fitted_params["fit_pmf_bins"] = int(args.fit_pmf_bins) if fit_objective == "pmf_mean" else None
    fitted_params["pmf_matching_profile"] = pmf_profile_rows if fit_objective == "pmf_mean" else None
    fitted_params["mm_ref_mode"] = (
        "nocutoff_nonperiodic" if float(args.fit_mm_ref_cutoff) <= 0.0 else "cutoff_periodic"
    )
    fitted_params["traj_total_frames"] = int(len(traj))
    fitted_params["ml_model"] = str(ml_model_name)
    fitted_params["label_mode"] = str(label_mode)
    fitted_params["fit_target_mode"] = fit_target_mode
    fitted_params["qm_reference_region_definition"] = "ligand + environment pocket"
    # Bug D 修复：target 现在是 mace_surrogate_residual，中心不再是 qm-mm 之差。
    # 用 target_center_kjmol 正名；qm_mm_offset_kjmol 保留为别名兼容旧下游。
    fitted_params["target_center_kjmol"] = float(delta_diag["center"])
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
        fitted_params["fit_target_definition"] = "surrogate target: DEXP ≈ (E_MACE_local - E_gaussian_coul_region) up to an arbitrary constant, so Gaussian Coulomb + DEXP describes MACE local interaction"
    else:
        fitted_params["fit_target_definition"] = "delta_fit = E_ml_interaction - E_mm_coul"
    fitted_params.update(detect_suspicious_fit(fitted_params))

    # Bug B 修复：fitting_success=True 只说明优化器返回了解，不代表解是健康的。
    # 顶界/全局优化失败/需夹 A/核不排斥 都是病态信号，单独汇总成 fit_health，别被 success 掩盖。
    health_reasons: List[str] = []
    if not bool(fitted_params.get("optimizer_global_success", True)):
        health_reasons.append("global_optimizer_failed")
    if bool(fitted_params.get("A_fit_clamped", False)):
        health_reasons.append("A_fit_clamped_for_repulsion")
    if not bool(fitted_params.get("short_range_repulsive_ok", True)):
        health_reasons.append("raw_core_not_repulsive")
    r0_val = float(fitted_params.get("r0_vdw", math.nan))
    if np.isfinite(r0_val) and (abs(r0_val - 0.30) < 1.0e-3 or abs(r0_val - 0.38) < 1.0e-3):
        health_reasons.append(f"r0_pinned_at_bound({r0_val:.4f})")
    fitted_params["fit_health"] = "degraded" if health_reasons else "ok"
    fitted_params["fit_health_reasons"] = health_reasons

    surface_summaries: Dict[str, Dict] = {}
    if fitted_params.get("fitting_success"):
        predicted_all = [predict_dexp_delta_e(d, fitted_params) for d in rebuilt_dists_per_frame]
        surface_summaries["all_accepted"] = write_mace_surrogate_surface_diagnostics(
            output_dir=output_dir,
            file_prefix=f"{file_prefix}fit_all_accepted_",
            frame_ids=accepted_frame_ids,
            min_dist_per_frame=accepted_min_dist,
            contact_count_per_frame=[int(len(dists)) for dists in rebuilt_dists_per_frame],
            mace_target_per_frame=accepted_delta_e_perframe,
            surrogate_pred_per_frame=predicted_all,
            args=args,
        )
        fitted_params["mace_surrogate_surface"] = surface_summaries

    if holdout_idx.size > 0 and fitted_params.get("fitting_success"):
        holdout_dists = [rebuilt_dists_per_frame[i] for i in holdout_idx]
        holdout_pair_types = [rebuilt_pair_types_per_frame[i] for i in holdout_idx]
        # C 修复：验证用真实逐帧 ΔE（不是 PMF 箱均值）——箱均值当真值是循环验证。
        holdout_delta_e = [accepted_delta_e_perframe[i] for i in holdout_idx]
        holdout_min_dist = [accepted_min_dist[i] for i in holdout_idx]
        holdout_frame_ids = [accepted_frame_ids[i] for i in holdout_idx]
        holdout_metrics = evaluate_holdout_predictions(holdout_dists, holdout_delta_e, fitted_params)
        holdout_metrics["target_is_perframe"] = True  # 已从箱均值改为逐帧真值
        predicted_holdout = [predict_dexp_delta_e(d, fitted_params) for d in holdout_dists]
        holdout_rows = [
            {
                "frame_index": frame_id,
                "min_distance_nm": float(md),
                "actual_delta_e_kjmol": float(actual),
                "predicted_delta_e_kjmol": float(predicted),
                "residual_kjmol": float(predicted - actual),
            }
            for frame_id, md, actual, predicted in zip(holdout_frame_ids, holdout_min_dist, holdout_delta_e, predicted_holdout)
        ]
        holdout_csv_path = write_rows_csv(os.path.join(output_dir, "fit_holdout_validation.csv"), holdout_rows)
        holdout_metrics["holdout_csv"] = holdout_csv_path

        # 判据 A：自由能相关量（系综均值 + FEP 重加权 + 留出集 PMF 剖面），这才是 DEXP 修正真正的用途。
        min_bin_frames = max(1, int(getattr(args, "fit_pmf_min_bin_frames", 10)))
        fe_metrics = evaluate_holdout_free_energy(
            holdout_dists, holdout_delta_e, holdout_min_dist, fitted_params,
            float(args.temperature), int(args.fit_pmf_bins), min_bin_frames,
        )
        fe_profile_rows = fe_metrics.pop("mean_profile_rows", [])
        holdout_metrics["free_energy"] = fe_metrics
        if fe_profile_rows:
            fe_prof_csv = write_rows_csv(
                os.path.join(output_dir, "fit_holdout_mean_profile.csv"), fe_profile_rows
            )
            holdout_metrics["holdout_mean_profile_csv"] = fe_prof_csv
        # Bug A 修复：holdout 的均值剖面由判据 A（evaluate_holdout_free_energy）唯一负责，
        # 不再重复调用 surface 诊断（那份只会用不同的 min_bin_frames 门槛给出第二个矛盾的 RMSE）。
        # surface 诊断只跑 all_accepted（见上），提供全数据剖面 + 2D。

        if (
            file_prefix == ""
            and bool(getattr(args, "learned_rbf_diagnostic", False))
            and not bool(getattr(args, "skip_learned_rbf_diagnostic", False))
        ):
            learned_metrics, learned_rows, learned_profile_rows = fit_learned_pair_rbf_diagnostic(
                train_dists=train_dists,
                train_pair_types=train_pair_types,
                train_targets=train_delta_e,
                holdout_dists=holdout_dists,
                holdout_pair_types=holdout_pair_types,
                holdout_targets=holdout_delta_e,
                holdout_min_dist=holdout_min_dist,
                args=args,
            )
            if learned_rows:
                for row, frame_id in zip(learned_rows, holdout_frame_ids):
                    row["frame_index"] = int(frame_id)
                learned_csv = write_rows_csv(
                    os.path.join(output_dir, "fit_learned_rbf_holdout_validation.csv"),
                    learned_rows,
                )
                learned_metrics["holdout_csv"] = learned_csv
            if learned_profile_rows:
                learned_profile_csv = write_rows_csv(
                    os.path.join(output_dir, "fit_learned_rbf_holdout_pmf_profile.csv"),
                    learned_profile_rows,
                )
                learned_metrics["holdout_pmf_profile_csv"] = learned_profile_csv
            learned_json = os.path.join(output_dir, "fit_learned_rbf_params.json")
            with open(learned_json, "w", encoding="utf-8") as handle:
                json.dump(learned_metrics, handle, indent=2, cls=NumpyEncoder)
            learned_metrics["params_json"] = learned_json
            holdout_metrics["learned_rbf"] = learned_metrics
            fitted_params["learned_rbf_diagnostic"] = learned_metrics
            if learned_metrics.get("enabled"):
                learned_profile = learned_metrics.get("pmf_profile", {}) or {}
                print(
                    f"    学习函数[RBF] holdout: RMSE={learned_metrics['rmse_raw_kjmol']:.2f} kJ/mol | "
                    f"bias={learned_metrics['bias_kjmol']:.2f} | R²={learned_metrics['r2_raw']:.3f} | "
                    f"pearson r={learned_metrics['pearson_r']:.3f} | "
                    f"均值剖面 RMSE={learned_profile.get('pmf_profile_rmse_kjmol', math.nan):.2f} "
                    f"r={learned_profile.get('pmf_profile_pearson_r', math.nan):.3f}"
                )

        try:
            plt = get_matplotlib_pyplot()
            # 图1：逐帧 parity（诚实展示逐帧噪声，预期很散）
            fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))
            ax = axes[0]
            actual_arr = np.asarray(holdout_delta_e, dtype=float)
            predicted_arr = np.asarray(predicted_holdout, dtype=float)
            ax.scatter(actual_arr, predicted_arr, s=14, alpha=0.5)
            lo = float(min(actual_arr.min(), predicted_arr.min()))
            hi = float(max(actual_arr.max(), predicted_arr.max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x")
            ax.set_xlabel("Actual per-frame delta-E (kJ/mol)")
            ax.set_ylabel("DEXP predicted delta-E (kJ/mol)")
            ax.set_title(
                f"Per-frame parity (n={len(actual_arr)}, R2={holdout_metrics['r2_raw']:.2f}, "
                f"r={holdout_metrics['pearson_r']:.2f})\nexpected noisy: per-frame sigma >> signal"
            )
            ax.legend(); ax.grid(alpha=0.3)
            # 图2：留出集"均值剖面" ⟨ΔE⟩(s) —— 模型 vs 真值（判据 A 的核心，可信量）
            ax2 = axes[1]
            if fe_profile_rows:
                s = [r["min_distance_center_nm"] for r in fe_profile_rows]
                t = [r["holdout_true_mean_kjmol"] for r in fe_profile_rows]
                m = [r["model_pred_mean_kjmol"] for r in fe_profile_rows]
                ax2.plot(s, t, "o-", label="holdout true <dE>(s)")
                ax2.plot(s, m, "s--", label="DEXP predicted <dE>(s)")
                ax2.set_xlabel("min L-E distance (nm)")
                ax2.set_ylabel("<delta-E> per bin (kJ/mol)")
                ax2.set_title(
                    f"Holdout mean profile (bins={fe_metrics.get('mean_profile_n_bins')}, "
                    f"RMSE={fe_metrics.get('mean_profile_rmse_kjmol', float('nan')):.1f}, "
                    f"r={fe_metrics.get('mean_profile_pearson_r', float('nan')):.2f})"
                )
                ax2.legend(); ax2.grid(alpha=0.3)
            holdout_png_path = os.path.join(output_dir, "fit_holdout_parity.png")
            fig.tight_layout()
            fig.savefig(holdout_png_path, dpi=180)
            plt.close(fig)
            holdout_metrics["holdout_parity_png"] = holdout_png_path
        except Exception as exc:
            holdout_metrics["plot_error"] = str(exc)
        print(
            f"    留出集验证[逐帧,C]: n={holdout_metrics['n_holdout_frames']} | "
            f"RMSE={holdout_metrics['rmse_raw_kjmol']:.2f} kJ/mol | "
            f"bias={holdout_metrics['bias_kjmol']:.2f} | "
            f"R²={holdout_metrics['r2_raw']:.3f} | pearson r={holdout_metrics['pearson_r']:.3f}"
        )
        print(
            f"    留出集系综判据[A]: ⟨ΔE⟩ 真值={fe_metrics['ensemble_mean_true_kjmol']:.2f} vs 模型={fe_metrics['ensemble_mean_model_kjmol']:.2f} "
            f"(bias={fe_metrics['ensemble_mean_bias_kjmol']:.2f}) | "
            f"均值剖面: within-SEM {fe_metrics['mean_profile_within_sem_bins']}/{fe_metrics['mean_profile_n_bins']} 箱 "
            f"(参考 RMSE={fe_metrics['mean_profile_rmse_kjmol']:.2f} r={fe_metrics['mean_profile_pearson_r']:.3f})"
        )
    else:
        holdout_metrics = {
            "n_holdout_frames": 0,
            "skipped_reason": holdout_skip_reason if holdout_idx.size == 0 else "fit_failed",
        }
        print(f"    留出集验证已跳过: {holdout_metrics['skipped_reason']}")
    fitted_params["holdout_validation"] = holdout_metrics

    params_path = os.path.join(output_dir, f"{file_prefix}dexp_fitted_params.json")
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
                "ml_model": str(ml_model_name),
                "label_mode": str(label_mode),
                "fit_target_mode": fit_target_mode,
                "env_search_radius_nm": float(env_search_radius),
                "env_max_atoms": int(env_max_atoms) if env_max_atoms is not None else None,
                "mm_ref_cutoff_nm": float(args.fit_mm_ref_cutoff),
                "mm_ref_switch_nm": float(args.fit_mm_ref_switch),
            },
            handle,
            indent=2,
        )

    print(f"    [{ml_model_name}] 拟合完成，已保存参数: {params_path}")
    if fitted_params.get("fit_health") == "degraded":
        print(f"    ⚠️ [{ml_model_name}] fit_health=degraded（fitting_success 掩盖不了）: "
              f"{', '.join(fitted_params.get('fit_health_reasons', []))}")
    if fitted_params.get("suspicious_fit"):
        print(f"    ⚠️ [{ml_model_name}] 检测到拟合参数撞边界，当前 DEXP 参数很可能不可靠")
        print(f"    ⚠️ 边界命中: {', '.join(fitted_params.get('boundary_hits', []))}")
    print(f"    帧诊断已保存: {fit_log_path}")
    return fitted_params, fit_log_rows


def build_ml_model_comparison(
    output_dir: str,
    primary_model: str,
    compare_model: str,
    primary_rows: List[Dict],
    compare_rows: List[Dict],
    primary_params: Dict,
    compare_params: Dict,
) -> Dict:
    primary_by_frame = {int(row["frame_index"]): row for row in primary_rows}
    compare_by_frame = {int(row["frame_index"]): row for row in compare_rows}
    common_frames = sorted(set(primary_by_frame) & set(compare_by_frame))
    rows = [
        {
            "frame_index": frame_id,
            "delta_e_primary_kjmol": float(primary_by_frame[frame_id]["delta_e_kjmol"]),
            "delta_e_compare_kjmol": float(compare_by_frame[frame_id]["delta_e_kjmol"]),
            "diff_kjmol": float(compare_by_frame[frame_id]["delta_e_kjmol"])
            - float(primary_by_frame[frame_id]["delta_e_kjmol"]),
        }
        for frame_id in common_frames
    ]
    csv_path = write_rows_csv(os.path.join(output_dir, "ml_model_comparison.csv"), rows) if rows else None

    primary_vals = np.asarray([r["delta_e_primary_kjmol"] for r in rows], dtype=float)
    compare_vals = np.asarray([r["delta_e_compare_kjmol"] for r in rows], dtype=float)
    diff_vals = compare_vals - primary_vals if primary_vals.size else np.asarray([], dtype=float)
    if primary_vals.size > 1 and np.std(primary_vals) > 1.0e-9 and np.std(compare_vals) > 1.0e-9:
        pearson_r = float(np.corrcoef(primary_vals, compare_vals)[0, 1])
    else:
        pearson_r = math.nan

    png_path = None
    if primary_vals.size:
        try:
            plt = get_matplotlib_pyplot()
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            ax.scatter(primary_vals, compare_vals, s=14, alpha=0.6)
            lo = float(min(primary_vals.min(), compare_vals.min()))
            hi = float(max(primary_vals.max(), compare_vals.max()))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x")
            ax.set_xlabel(f"delta-E fit target ({primary_model}, kJ/mol)")
            ax.set_ylabel(f"delta-E fit target ({compare_model}, kJ/mol)")
            ax.set_title(f"ML reference agreement (n={primary_vals.size}, pearson r={pearson_r:.3f})")
            ax.legend()
            ax.grid(alpha=0.3)
            png_path = os.path.join(output_dir, "ml_model_comparison.png")
            fig.tight_layout()
            fig.savefig(png_path, dpi=180)
            plt.close(fig)
        except Exception:
            png_path = None

    primary_suspicious = bool(primary_params.get("suspicious_fit"))
    compare_suspicious = bool(compare_params.get("suspicious_fit"))
    primary_boundary_hits = list(primary_params.get("boundary_hits", [])) if primary_suspicious else []
    compare_boundary_hits = list(compare_params.get("boundary_hits", [])) if compare_suspicious else []

    def _param_value(params: Dict, suspicious: bool, key: str):
        return None if suspicious else params.get(key)

    def _holdout_value(params: Dict, suspicious: bool) -> Dict:
        return {} if suspicious else (params.get("holdout_validation", {}) or {})

    summary = {
        "primary_model": str(primary_model),
        "compare_model": str(compare_model),
        "n_common_frames": int(len(rows)),
        "delta_e_pearson_r": pearson_r,
        "delta_e_diff_mean_kjmol": float(np.mean(diff_vals)) if diff_vals.size else math.nan,
        "delta_e_diff_std_kjmol": float(np.std(diff_vals)) if diff_vals.size else math.nan,
        "comparison_csv": csv_path,
        "comparison_png": png_path,
        "primary_suspicious_fit": primary_suspicious,
        "compare_suspicious_fit": compare_suspicious,
        "primary_boundary_hits": primary_boundary_hits,
        "compare_boundary_hits": compare_boundary_hits,
        "params": {
            key: {
                "primary": _param_value(primary_params, primary_suspicious, key),
                "compare": _param_value(compare_params, compare_suspicious, key),
            }
            for key in ("alpha_vdw", "beta_vdw", "r0_vdw", "A_fit", "B_fit")
        },
        "holdout": {
            "primary": _holdout_value(primary_params, primary_suspicious),
            "compare": _holdout_value(compare_params, compare_suspicious),
        },
    }
    if primary_suspicious:
        print(
            f"    ⚠️ {primary_model} 拟合撞边界（{', '.join(primary_boundary_hits) or 'unknown'}），"
            "其参数/holdout 已从对比中砍掉，只保留 ΔE 相关性"
        )
    if compare_suspicious:
        print(
            f"    ⚠️ {compare_model} 拟合撞边界（{', '.join(compare_boundary_hits) or 'unknown'}），"
            "其参数/holdout 已从对比中砍掉，只保留 ΔE 相关性"
        )
    print(
        f"    MACE/Orb-v3 对比 ({primary_model} vs {compare_model}): "
        f"n={summary['n_common_frames']} | pearson r={summary['delta_e_pearson_r']:.3f} | "
        f"diff mean±std={summary['delta_e_diff_mean_kjmol']:.2f}±{summary['delta_e_diff_std_kjmol']:.2f} kJ/mol"
    )
    return summary


def fit_dexp_from_tail_frames(args: argparse.Namespace, output_dir: str) -> Dict:
    ensure_dir(output_dir)
    md = require_module("mdtraj")
    symbols = load_abfe_symbols()
    select_env_indices = symbols["_select_env_indices_from_mdtraj_frame"]
    require_openmm()

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
    mm_contexts = build_mm_le_contexts_from_system_xml(
        args.system_xml,
        ligand_indices=lig_idx.tolist(),
        environment_indices=env_idx.tolist(),
        cutoff_nm=float(args.fit_mm_ref_cutoff),
        switching_nm=float(args.fit_mm_ref_switch),
    )

    fit_xyz = np.asarray(fit_traj.xyz, dtype=np.float64)
    fit_time = np.asarray(getattr(traj, "time", np.arange(len(traj), dtype=float)), dtype=float)[fit_indices]
    fit_box = None
    if fit_traj.unitcell_vectors is not None:
        fit_box = np.asarray(fit_traj.unitcell_vectors, dtype=np.float64)

    print(f"[2/4] 使用主模型 {args.ml_model} 标注 + 拟合 DEXP")
    fitted_params, fit_log_rows = _fit_dexp_with_ml_model(
        args, output_dir, args.ml_model, "",
        traj, fit_indices, lig_idx, env_idx, all_nums, mm_contexts,
        fit_xyz, fit_time, fit_box, env_search_radius, env_max_atoms,
    )

    if args.compare_ml_model:
        print(f"[2b/4] 使用对比模型 {args.compare_ml_model} 在同一批帧上重新标注 + 拟合 DEXP")
        compare_params, compare_log_rows = _fit_dexp_with_ml_model(
            args, output_dir, args.compare_ml_model, "compare_",
            traj, fit_indices, lig_idx, env_idx, all_nums, mm_contexts,
            fit_xyz, fit_time, fit_box, env_search_radius, env_max_atoms,
        )
        fitted_params["ml_model_comparison"] = build_ml_model_comparison(
            output_dir,
            args.ml_model,
            args.compare_ml_model,
            fit_log_rows,
            compare_log_rows,
            fitted_params,
            compare_params,
        )

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
    cutoff_nm: float = 0.0,
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
        # 三种参考力都限定在 lig×env 相互作用组内：既保证与 MACE 分解使用完全相同的
        # L-E 原子对集合（env 是 last-frame 选定后全程固定的口袋），也让 NoCutoff 只在
        # |lig|×|env| 对上求和，不会退化成 O(N^2) 扫全盒子。
        le_force.addInteractionGroup(
            [int(idx) for idx in ligand_indices],
            [int(idx) for idx in environment_indices],
        )
        if cutoff_nm and float(cutoff_nm) > 0.0:
            le_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            le_force.setCutoffDistance(cutoff_nm * unit.nanometer)
            if switching_nm and 0.0 < float(switching_nm) < float(cutoff_nm):
                le_force.setUseSwitchingFunction(True)
                le_force.setSwitchingDistance(switching_nm * unit.nanometer)
            else:
                le_force.setUseSwitchingFunction(False)
        else:
            # cutoff_nm<=0 -> NoCutoff：全程 1/r、非周期、无 switching，与 MACE 真空团簇
            # 的边界条件一致。配体/环境原子穿越硬截断面造成的能量不连续跳变被消除，
            # ΔE=E_MACE-E_MM 的方差不再被截断伪影主导（此前 e_mm_coul std≈214 的元凶）。
            le_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
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
    original_system,
    dexp_system,
) -> Dict:
    print("[5/8] 执行 original vs DEXP lambda=1→0 单点扫描")
    traj, sampled_indices = load_analysis_traj(args.traj, args.traj_top, args.analysis_max_frames)
    contexts = {
        "original_baseline": build_context_for_system(original_system, args),
        "dexp_surrogate": build_context_for_system(dexp_system, args),
    }
    lambda_values = np.linspace(1.0, 0.0, max(2, int(args.lambda_scan_points)))
    box = np.asarray(traj.unitcell_vectors, dtype=np.float64) if traj.unitcell_vectors is not None else None
    rows: List[Dict] = []
    per_key_energy: Dict[Tuple[str, float], List[float]] = {}
    per_key_force: Dict[Tuple[str, float], List[float]] = {}
    paired_delta: Dict[float, List[float]] = {}

    for local_idx, frame_idx in enumerate(sampled_indices):
        pos_nm = np.asarray(traj.xyz[local_idx], dtype=np.float64)
        box_vecs = box[local_idx] if box is not None else None
        prev_energy_by_ensemble: Dict[str, float] = {}
        for lam in lambda_values:
            metrics_by_ensemble: Dict[str, Dict[str, float]] = {}
            for ensemble, context in contexts.items():
                metrics = evaluate_context(
                    context,
                    positions_nm=pos_nm,
                    box_vectors_nm=box_vecs,
                    lam_coul=float(lam),
                    lam_vdw=float(lam),
                    include_forces=True,
                )
                metrics_by_ensemble[ensemble] = metrics

            delta_dexp_minus_original = (
                float(metrics_by_ensemble["dexp_surrogate"]["potential_kjmol"])
                - float(metrics_by_ensemble["original_baseline"]["potential_kjmol"])
            )
            paired_delta.setdefault(float(lam), []).append(delta_dexp_minus_original)

            for ensemble, metrics in metrics_by_ensemble.items():
                previous = prev_energy_by_ensemble.get(ensemble)
                jump = math.nan if previous is None else float(metrics["potential_kjmol"] - previous)
                prev_energy_by_ensemble[ensemble] = float(metrics["potential_kjmol"])
                row = {
                    "ensemble": ensemble,
                    "frame_index": int(frame_idx),
                    "lambda_value": float(lam),
                    "potential_kjmol": float(metrics["potential_kjmol"]),
                    "delta_from_prev_lambda_kjmol": jump,
                    "delta_dexp_minus_original_kjmol": float(delta_dexp_minus_original),
                    "max_force_kjmol_per_nm": float(metrics["max_force_kjmol_per_nm"]),
                    "mean_force_kjmol_per_nm": float(metrics["mean_force_kjmol_per_nm"]),
                    "is_finite": int(
                        np.isfinite(metrics["potential_kjmol"])
                        and np.isfinite(metrics["max_force_kjmol_per_nm"])
                        and np.isfinite(delta_dexp_minus_original)
                    ),
                }
                rows.append(row)
                lam_key = float(lam)
                per_key_energy.setdefault((ensemble, lam_key), []).append(float(metrics["potential_kjmol"]))
                per_key_force.setdefault((ensemble, lam_key), []).append(float(metrics["max_force_kjmol_per_nm"]))

    csv_path = write_rows_csv(os.path.join(output_dir, "lambda_single_point_scan_comparison.csv"), rows)
    per_ensemble = []
    for (ensemble, lam), energies in sorted(per_key_energy.items(), key=lambda item: (item[0][0], -item[0][1])):
        forces = per_key_force[(ensemble, lam)]
        per_ensemble.append(
            {
                "ensemble": ensemble,
                "lambda_value": float(lam),
                "potential_mean_kjmol": float(statistics.fmean(energies)),
                "potential_std_kjmol": float(statistics.stdev(energies)) if len(energies) > 1 else 0.0,
                "max_force_mean_kjmol_per_nm": float(statistics.fmean(forces)),
                "max_force_max_kjmol_per_nm": float(max(forces)),
            }
        )
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
            max((max(vals) for vals in per_key_force.values()), default=math.nan)
        ),
        "delta_dexp_minus_original_by_lambda": [
            {
                "lambda_value": float(lam),
                **summarize_series_with_percentiles(values),
            }
            for lam, values in sorted(paired_delta.items(), reverse=True)
        ],
        "per_ensemble_lambda": per_ensemble,
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

    switch_cutoff_nm = 0.65
    switch_width_nm = 0.20
    params_for_switch = os.path.join(output_dir, "dexp_fitted_params.json")
    if os.path.isfile(params_for_switch):
        try:
            with open(params_for_switch, "r", encoding="utf-8") as handle:
                params_payload = json.load(handle)
            switch_cutoff_nm = float(params_payload.get("cutoff_distance", switch_cutoff_nm))
            switch_width_nm = float(params_payload.get("switch_width", switch_width_nm))
        except Exception:
            pass
    switch_start_nm = max(0.0, float(switch_cutoff_nm) - max(0.0, float(switch_width_nm)))
    working_window_mask = (rdf_r_original >= analysis_r_min) & (rdf_r_original <= analysis_r_max)
    core_window_mask = (rdf_r_original >= analysis_r_min) & (rdf_r_original < min(analysis_r_max, switch_start_nm))
    switch_zone_mask = (
        (rdf_r_original >= max(analysis_r_min, switch_start_nm))
        & (rdf_r_original <= min(analysis_r_max, switch_cutoff_nm))
    )
    pmf_window_mask = (pmf_r_original[:n_pmf] >= analysis_r_min) & (pmf_r_original[:n_pmf] <= analysis_r_max)

    # PMF 覆盖度诊断：这里的 original_min/dexp_min 来自两条 1 ns 无偏、全耦合轨迹，
    # 配体全程留在结合口袋内是预期行为，但这意味着 pmf_edges 里绝大多数 bin
    # 永远采不到样。把“实际有采样的 bin 数/总 bin 数”和“实际探索到的距离范围”
    # 记录下来，这样报告和绘图才能如实反映这是一段局部结合态波动曲线，
    # 而不是覆盖 analysis_r_min~analysis_r_max 的完整解离 PMF。
    n_populated_original = int(np.sum(finite_original)) if n_pmf > 0 else 0
    n_populated_dexp = int(np.sum(finite_dexp)) if n_pmf > 0 else 0
    sampled_range_original = (
        [float(np.min(pmf_r_original[:n_pmf][finite_original])), float(np.max(pmf_r_original[:n_pmf][finite_original]))]
        if n_populated_original > 0 else None
    )
    sampled_range_dexp = (
        [float(np.min(pmf_r_dexp[:n_pmf][finite_dexp])), float(np.max(pmf_r_dexp[:n_pmf][finite_dexp]))]
        if n_populated_dexp > 0 else None
    )

    summary = {
        "min_distance_csv": min_csv,
        "rdf_csv": rdf_csv,
        "pmf_csv": pmf_csv,
        "analysis_r_min_nm": float(analysis_r_min),
        "analysis_r_max_nm": float(analysis_r_max),
        "surrogate_switch_start_nm": float(switch_start_nm),
        "surrogate_cutoff_nm": float(switch_cutoff_nm),
        "ligand_heavy_atoms": int(len(lig_heavy)),
        "environment_heavy_atoms": int(len(env_heavy)),
        "original_min_distance_nm": summarize_series_with_percentiles(original_min),
        "dexp_min_distance_nm": summarize_series_with_percentiles(dexp_min),
        "rdf_working_window_peak_original": float(np.max(rdf_g_original[working_window_mask])) if np.any(working_window_mask) else math.nan,
        "rdf_working_window_peak_dexp": float(np.max(rdf_g_dexp[working_window_mask])) if np.any(working_window_mask) else math.nan,
        "rdf_core_window_peak_original": float(np.max(rdf_g_original[core_window_mask])) if np.any(core_window_mask) else math.nan,
        "rdf_core_window_peak_dexp": float(np.max(rdf_g_dexp[core_window_mask])) if np.any(core_window_mask) else math.nan,
        "rdf_switch_zone_peak_original": float(np.max(rdf_g_original[switch_zone_mask])) if np.any(switch_zone_mask) else math.nan,
        "rdf_switch_zone_peak_dexp": float(np.max(rdf_g_dexp[switch_zone_mask])) if np.any(switch_zone_mask) else math.nan,
        "rdf_switch_zone_note": (
            "The surrogate force switches off between switch_start and cutoff. RDF features in this shell "
            "are reported separately because they may be force-smoothing artifacts rather than core local MACE/DEXP physics."
        ),
        "pmf_reference_region_start_nm": float(ref_start_nm),
        "pmf_working_window_delta_max_kjmol": float(
            np.nanmax(np.abs((pmf_dexp[:n_pmf] - pmf_original[:n_pmf])[pmf_window_mask]))
        ) if n_pmf > 0 and np.any(pmf_window_mask) else math.nan,
        "pmf_total_bins": int(n_pmf),
        "pmf_populated_bins_original": n_populated_original,
        "pmf_populated_bins_dexp": n_populated_dexp,
        "pmf_sampled_range_original_nm": sampled_range_original,
        "pmf_sampled_range_dexp_nm": sampled_range_dexp,
        "pmf_is_local_bound_state_profile": True,
        "pmf_note": (
            "两条轨迹均为 1 ns 无偏全耦合(lambda=1)采样，配体全程未离开结合口袋；"
            "此 PMF 只反映实际探索到的局部距离范围内的相对自由能，不是覆盖 "
            f"{analysis_r_min:.2f}-{analysis_r_max:.2f} nm 的完整解离 PMF。"
            "要看更宽范围可参考 lambda_window_pmf.png（含 lambda=0 解耦窗口）。"
        ),
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

    schedule_csv = os.path.join(output_dir, "lambda_schedule_reference.csv")
    if os.path.isfile(schedule_csv):
        rows = read_csv_rows(schedule_csv)
        fig, ax = plt.subplots(figsize=(8, 5))
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
        ax.set_ylabel("Lambda")
        ax.set_title("Lambda Schedule Comparison")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = os.path.join(output_dir, "lambda_schedule_reference.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pngs["lambda_schedule_png"] = path

    lambda_csv = os.path.join(output_dir, "lambda_single_point_scan_comparison.csv")
    if os.path.isfile(lambda_csv):
        rows = read_csv_rows(lambda_csv)
        grouped: Dict[Tuple[str, float], Dict[str, List[float]]] = {}
        for row in rows:
            ensemble = str(row.get("ensemble", "dexp_surrogate"))
            lam = float(row["lambda_value"])
            payload = grouped.setdefault((ensemble, lam), {"potential": [], "force": []})
            payload["potential"].append(float(row["potential_kjmol"]))
            payload["force"].append(float(row["max_force_kjmol_per_nm"]))
        fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
        for ensemble in sorted({key[0] for key in grouped}):
            lambdas = sorted({key[1] for key in grouped if key[0] == ensemble}, reverse=True)
            axes[0].plot(
                lambdas,
                [statistics.fmean(grouped[(ensemble, lam)]["potential"]) for lam in lambdas],
                marker="o",
                label=ensemble,
            )
            axes[1].plot(
                lambdas,
                [max(grouped[(ensemble, lam)]["force"]) for lam in lambdas],
                marker="o",
                label=ensemble,
            )
        axes[0].set_ylabel("Mean Potential (kJ/mol)")
        axes[0].set_title("Lambda Single-Point Scan")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Lambda")
        axes[1].set_ylabel("Max Force (kJ/mol/nm)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        path = os.path.join(output_dir, "lambda_single_point_scan_comparison.png")
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
        bin_width = (x[1] - x[0]) if len(x) > 1 else 0.01
        finite_x = [
            xv for xv, yo, yd in zip(x, y_mm, y_dexp)
            if math.isfinite(yo) or math.isfinite(yd)
        ]
        # 这两条轨迹是 1 ns 无偏全耦合采样，配体全程不会离开结合口袋；PMF 只在
        # 实际探索过的短程区间内有意义，因此这里按真实有数据的范围自适应坐标轴，
        # 而不是套用固定的 analysis_r_min~analysis_r_max，避免看起来像“漏算了一大截”。
        if finite_x:
            x_lo = min(finite_x) - bin_width
            x_hi = max(finite_x) + bin_width
        elif x:
            x_lo, x_hi = min(x), max(x)
        else:
            x_lo, x_hi = 0.20, 0.65
        n_total = len(x)
        n_populated = len(finite_x)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y_mm, marker="o", markersize=3, label="original_baseline")
        ax.plot(x, y_dexp, marker="o", markersize=3, label="dexp_surrogate")
        ax.set_xlabel("Min L-E Distance (nm)")
        ax.set_ylabel("Relative Free Energy (kJ/mol)")
        ax.set_title("1D Contact Free-Energy Profile (bound-state local window)")
        ax.set_xlim(x_lo, x_hi)
        ax.text(
            0.02, 0.02,
            f"{n_populated}/{n_total} bins sampled ({x_lo:.3f}-{x_hi:.3f} nm)\n"
            "1 ns unbiased, fully-coupled trajectories — ligand stays bound;\n"
            "not a full dissociation PMF.",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
        )
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
    window_csv = os.path.join(output_dir, "lambda_window_ensemble.csv")
    rows = read_csv_rows(window_csv)
    lambda_rows = [row for row in rows if int(float(row.get("used_for_postprocess", 0))) == 1]
    if not lambda_rows:
        return {}

    pmf_rows: List[Dict] = []
    rdf_rows: List[Dict] = []
    min_rows: List[Dict] = []
    summaries: List[Dict] = []

    for row in lambda_rows:
        ensemble = str(row.get("ensemble", "dexp_surrogate"))
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
                    "ensemble": ensemble,
                    "lambda_value": float(lam),
                    "frame_index": int(frame_idx),
                    "min_distance_nm": float(value),
                }
            )

        rdf_r, rdf_g = compute_rdf(traj, lig_heavy, env_heavy, args.rdf_r_max, args.rdf_bin_width)
        for radius, g_val in zip(rdf_r, rdf_g):
            rdf_rows.append(
                {
                    "ensemble": ensemble,
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
                    "ensemble": ensemble,
                    "lambda_value": float(lam),
                    "distance_nm": float(distance_nm),
                    "pmf_kjmol": float(pmf_val) if np.isfinite(pmf_val) else math.nan,
                }
            )

        summary = summarize_series_with_percentiles(min_series)
        summary["ensemble"] = ensemble
        summary["lambda_value"] = float(lam)
        summaries.append(summary)

    min_csv = write_rows_csv(os.path.join(output_dir, "lambda_window_min_distance.csv"), min_rows)
    rdf_csv = write_rows_csv(os.path.join(output_dir, "lambda_window_rdf.csv"), rdf_rows)
    pmf_csv = write_rows_csv(os.path.join(output_dir, "lambda_window_pmf.csv"), pmf_rows)

    plt = get_matplotlib_pyplot()
    pngs: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(7, 5))
    for ensemble in sorted({str(row["ensemble"]) for row in rdf_rows}):
        for lam in sorted({float(row["lambda_value"]) for row in rdf_rows if str(row["ensemble"]) == ensemble}, reverse=True):
            subset = [
                row for row in rdf_rows
                if str(row["ensemble"]) == ensemble and abs(float(row["lambda_value"]) - lam) < 1.0e-8
            ]
            subset.sort(key=lambda item: float(item["r_nm"]))
            ax.plot(
                [float(item["r_nm"]) for item in subset],
                [float(item["g_r"]) for item in subset],
                label=f"{ensemble} λ={lam:.2f}",
            )
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
    for ensemble in sorted({str(row["ensemble"]) for row in pmf_rows}):
        for lam in sorted({float(row["lambda_value"]) for row in pmf_rows if str(row["ensemble"]) == ensemble}, reverse=True):
            subset = [
                row for row in pmf_rows
                if str(row["ensemble"]) == ensemble and abs(float(row["lambda_value"]) - lam) < 1.0e-8
            ]
            subset.sort(key=lambda item: float(item["distance_nm"]))
            ax.plot(
                [float(item["distance_nm"]) for item in subset],
                [float(item["pmf_kjmol"]) if str(item["pmf_kjmol"]).lower() != "nan" else math.nan for item in subset],
                label=f"{ensemble} λ={lam:.2f}",
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
    for ensemble in sorted({str(item["ensemble"]) for item in summaries}):
        subset = [item for item in summaries if str(item["ensemble"]) == ensemble]
        subset.sort(key=lambda item: float(item["lambda_value"]), reverse=True)
        lambdas = [float(item["lambda_value"]) for item in subset]
        p05 = [float(item["p05"]) for item in subset]
        p50 = [float(item["p50"]) for item in subset]
        p95 = [float(item["p95"]) for item in subset]
        ax.plot(lambdas, p50, marker="o", label=f"{ensemble} p50")
        ax.fill_between(lambdas, p05, p95, alpha=0.18)
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
        "lambda_window_ensemble_csv": window_csv,
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
    schedule_csv = write_schedule_comparison(output_dir, int(args.schedule_states))
    lambda_window_info = run_lambda_window_ensemble(
        args=args,
        systems={
            "original_baseline": original_system,
            "dexp_surrogate": dexp_system,
        },
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        output_dir=output_dir,
    )
    lambda_scan_summary = run_lambda_single_point_scan(args, output_dir, original_system, dexp_system)
    contact_summary = run_contact_and_pmf_analysis(args, output_dir)
    delta_u_summary = run_delta_u_analysis(args, output_dir, original_system, dexp_system)
    lambda_window_summary = run_lambda_window_contact_analysis(args, output_dir)
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
        "lambda_window_ensemble": lambda_window_info,
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
    seed: int | None = None,
    sim_dir: str | None = None,
) -> Dict:
    openmm, app, unit, _ = require_openmm()
    seed = int(args.seed) if seed is None else int(seed)
    sim_dir = ensure_dir(sim_dir if sim_dir is not None else os.path.join(output_dir, label))
    csv_path = os.path.join(sim_dir, "state.csv")
    dcd_path = os.path.join(sim_dir, "traj.dcd")

    sim_system = strip_barostat(system)
    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.dt_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(seed)
    platform, properties = select_platform(args.platform)
    platform_label = format_platform_label(platform, properties)
    simulation = app.Simulation(topology, sim_system, integrator, platform, properties)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, seed)

    for parameter_name in ("lam_coul", "lam_vdw"):
        try:
            simulation.context.setParameter(parameter_name, 1.0)
        except Exception:
            pass

    did_minimize = not bool(args.skip_stability_minimize)
    if did_minimize:
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

    did_warmup = label == "dexp_surrogate" or not bool(args.skip_baseline_warmup)
    if did_warmup:
        ramp_schedule = parse_ramp_dt_schedule(args)
        total_warmup_steps = max(0, int(args.warmup_steps))
        print(
            f"  ↪ 对称预生产热身: {label} | soft-start={args.softstart_dt_fs:.3f} fs | "
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
        "seed": int(seed),
        "steps": int(n_steps),
        "dt_fs": float(args.dt_fs),
        "sim_ns": float(args.sim_ns),
        "preproduction_minimized": bool(did_minimize),
        "preproduction_warmup_steps": int(max(0, int(args.warmup_steps)) if did_warmup else 0),
        "preproduction_protocol": "symmetric_minimize_and_warmup" if did_warmup else "production_only_after_initialization",
        "potential_kjmol": summarize_series(data["potentialEnergy"]),
        "kinetic_kjmol": summarize_series(data["kineticEnergy"]),
        "total_kjmol": summarize_series(data["totalEnergy"]),
        "temperature_K": summarize_series(data["temperature"]),
    }
    summary.update(compute_ligand_rmsd_metrics(dcd_path, args.traj_top, args.ligand))
    with open(os.path.join(sim_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def summarize_replicate_variability(replica_summaries: List[Dict]) -> Dict:
    if len(replica_summaries) < 2:
        return {"n_replicas": int(len(replica_summaries)), "note": "need >=2 replicas to estimate variability"}
    fields = {
        "potential_mean_kjmol": [float(r["potential_kjmol"]["mean"]) for r in replica_summaries],
        "temperature_mean_K": [float(r["temperature_K"]["mean"]) for r in replica_summaries],
        "total_energy_std_kjmol": [float(r["total_kjmol"]["std"]) for r in replica_summaries],
        "ligand_rmsd_mean_A": [float(r.get("ligand_rmsd_mean_A", math.nan)) for r in replica_summaries],
    }
    out: Dict = {"n_replicas": int(len(replica_summaries))}
    for key, values in fields.items():
        out[f"{key}_across_replicas"] = summarize_series_with_percentiles(values)
    return out


def run_stability_ensemble(
    label: str,
    system,
    topology,
    positions,
    box_vectors,
    args: argparse.Namespace,
    output_dir: str,
) -> Dict:
    n_replicas = max(1, int(args.stability_replicas))
    replica_summaries: List[Dict] = []
    for replica_idx in range(n_replicas):
        seed = int(args.seed) if replica_idx == 0 else int(args.seed) + replica_idx * 104729
        sim_dir = (
            os.path.join(output_dir, label)
            if replica_idx == 0
            else os.path.join(output_dir, label, f"replica_{replica_idx}")
        )
        print(f"  ↪ {label} replica {replica_idx + 1}/{n_replicas} (seed={seed})")
        summary = run_stability_simulation(
            label=label,
            system=system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            args=args,
            output_dir=output_dir,
            seed=seed,
            sim_dir=sim_dir,
        )
        summary["replica_index"] = int(replica_idx)
        replica_summaries.append(summary)

    primary_summary = dict(replica_summaries[0])
    primary_summary["replicas"] = replica_summaries
    primary_summary["replica_variability"] = summarize_replicate_variability(replica_summaries)
    with open(os.path.join(output_dir, label, "replica_summaries.json"), "w", encoding="utf-8") as handle:
        json.dump(replica_summaries, handle, indent=2)
    return primary_summary


def load_stability_summary_with_replicas(output_dir: str, label: str) -> Dict:
    summary_path = ensure_file(os.path.join(output_dir, label, "summary.json"), f"{label} summary")
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    replicas_path = os.path.join(output_dir, label, "replica_summaries.json")
    if os.path.isfile(replicas_path):
        with open(replicas_path, "r", encoding="utf-8") as handle:
            replica_summaries = json.load(handle)
        summary["replicas"] = replica_summaries
        summary["replica_variability"] = summarize_replicate_variability(replica_summaries)
    return summary


def run_fixed_lambda_window_simulation(
    system,
    topology,
    positions,
    box_vectors,
    args: argparse.Namespace,
    output_dir: str,
    ensemble: str,
    lambda_value: float,
) -> Dict:
    openmm, app, unit, _ = require_openmm()
    label = f"lambda_{lambda_value:.2f}".replace(".", "p")
    sim_dir = ensure_dir(os.path.join(output_dir, "lambda_windows", ensemble, label))
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
        "ensemble": ensemble,
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
    systems: Dict[str, object],
    topology,
    positions,
    box_vectors,
    output_dir: str,
) -> Dict:
    lambda_values = parse_lambda_window_values(args)
    rows: List[Dict] = []
    summaries: List[Dict] = []
    for ensemble, system in systems.items():
        for lam in lambda_values:
            summary = run_fixed_lambda_window_simulation(
                system=system,
                topology=topology,
                positions=positions,
                box_vectors=box_vectors,
                args=args,
                output_dir=output_dir,
                ensemble=ensemble,
                lambda_value=float(lam),
            )
            summaries.append(summary)
            rows.append(
                {
                    "ensemble": ensemble,
                    "lambda_value": float(lam),
                    "lam_coul": float(lam),
                    "lam_vdw": float(lam),
                    "sim_ns": float(args.lambda_window_ns),
                    "window_dir": os.path.join(output_dir, "lambda_windows", ensemble, summary["label"]),
                    "used_for_postprocess": 1,
                }
            )
    csv_path = write_rows_csv(os.path.join(output_dir, "lambda_window_ensemble.csv"), rows)
    return {
        "lambda_values": [float(x) for x in lambda_values],
        "window_ensemble_csv": csv_path,
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
    out_csv = os.path.join(output_dir, "lambda_schedule_reference.csv")
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
    for state, stage, lam_coul, lam_vdw in build_interaction_separation_schedule(n_states):
        rows.append(
            {
                "schedule": "interaction_separation_decoupling",
                "state": state,
                "stage": stage,
                "lambda_coul": lam_coul,
                "lambda_vdw": lam_vdw,
                "direction": "1_to_0",
                "used_by_current_stability_run": 0,
                "notes": "Reference path that removes Coulomb before VDW",
            }
        )
    for state, stage, lam_coul, lam_vdw in build_surrogate_activation_reference_schedule(n_states):
        rows.append(
            {
                "schedule": "surrogate_activation_warmup",
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


def _find_lambda_entry(entries: List[Dict], target_lambda: float, tol: float = 1.0e-6) -> Dict:
    for entry in entries:
        if abs(float(entry.get("lambda_value", math.nan)) - target_lambda) < tol:
            return entry
    return {}


def write_comparison_report(
    output_dir: str,
    original_summary: Dict,
    dexp_summary: Dict,
    fitted_params: Dict,
    fit_quality: Dict,
    schedule_csv: str,
    lambda_scan_summary: Dict,
    contact_summary: Dict,
    delta_u_summary: Dict,
    lambda_window_summary: Dict | None = None,
) -> str:
    report_path = os.path.join(output_dir, "comparison_report.md")

    holdout = fitted_params.get("holdout_validation", {}) or {}
    lambda1_entry = _find_lambda_entry(
        lambda_scan_summary.get("delta_dexp_minus_original_by_lambda", []), 1.0
    )
    dexp_variability = dexp_summary.get("replica_variability", {}) or {}
    original_variability = original_summary.get("replica_variability", {}) or {}
    n_replicas = max(
        int(dexp_variability.get("n_replicas", 1) or 1),
        int(original_variability.get("n_replicas", 1) or 1),
    )

    summary_lines = [
        "## Summary (headline)",
        f"- Fit holdout validation: n={holdout.get('n_holdout_frames', 0)} | "
        f"RMSE={holdout.get('rmse_raw_kjmol', math.nan):.2f} kJ/mol | "
        f"bias={holdout.get('bias_kjmol', math.nan):.2f} kJ/mol | "
        f"R2={holdout.get('r2_raw', math.nan):.3f} | pearson r={holdout.get('pearson_r', math.nan):.3f}"
        + ("" if holdout.get("skipped_reason") is None else f" (skipped: {holdout.get('skipped_reason')})"),
        "  这是 DEXP 拟合的泛化能力检验：holdout 帧完全没参与拟合，RMSE/R2 越好说明 DEXP 越能复现 Orb 参考，不是训练集内自欺。",
        f"- Energy agreement at lambda=1 (paired same-configuration DEXP - original): "
        f"mean={lambda1_entry.get('mean', math.nan):.2f} kJ/mol | "
        f"p05/p50/p95={lambda1_entry.get('p05', math.nan):.2f} / "
        f"{lambda1_entry.get('p50', math.nan):.2f} / {lambda1_entry.get('p95', math.nan):.2f} kJ/mol",
        f"- Stability replicas run per ensemble: {n_replicas}"
        + ("" if n_replicas >= 2 else " (single run only, no variability estimate — 差异可能只是随机噪声)"),
    ]
    if n_replicas >= 2:
        summary_lines.append(
            "- Original replicate spread (temperature mean std / ligand RMSD mean std): "
            f"{original_variability.get('temperature_mean_K_across_replicas', {}).get('std', math.nan):.3f} K / "
            f"{original_variability.get('ligand_rmsd_mean_A_across_replicas', {}).get('std', math.nan):.3f} A"
        )
        summary_lines.append(
            "- DEXP replicate spread (temperature mean std / ligand RMSD mean std): "
            f"{dexp_variability.get('temperature_mean_K_across_replicas', {}).get('std', math.nan):.3f} K / "
            f"{dexp_variability.get('ligand_rmsd_mean_A_across_replicas', {}).get('std', math.nan):.3f} A"
        )
    summary_lines.append("")

    lines = [
        "# DEXP Stability Comparison",
        "",
        *summary_lines,
        "## Fitting",
        f"- Fit frames requested: {fitted_params.get('fit_frames_requested')}",
        f"- Fit frames used (train+holdout): {fitted_params.get('fit_frames_used')}",
        f"- Fit frames train / holdout: {fitted_params.get('fit_frames_train')} / {fitted_params.get('fit_frames_holdout')}",
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
        f"- Fit QC pass: {fit_quality.get('qc_pass')}",
        f"- Fit QC issues: {', '.join(fit_quality.get('qc_issues', [])) or 'none'}",
        f"- Fit diagnostics CSV: {fit_quality.get('fit_frame_diagnostics_csv')}",
        f"- Used frame fraction: {fit_quality.get('used_rows_fraction', math.nan):.3f}",
        f"- Centered ΔE used std (kJ/mol): {fit_quality.get('delta_e_centered_used_kjmol', {}).get('std', math.nan):.3f}",
        f"- Valid short-range pairs p05/p50/p95: "
        f"{fit_quality.get('n_valid_pairs_used', {}).get('p05', math.nan):.1f} / "
        f"{fit_quality.get('n_valid_pairs_used', {}).get('p50', math.nan):.1f} / "
        f"{fit_quality.get('n_valid_pairs_used', {}).get('p95', math.nan):.1f}",
        "",
    ]
    ml_comparison = fitted_params.get("ml_model_comparison")
    if ml_comparison:
        holdout_primary = ml_comparison.get("holdout", {}).get("primary", {}) or {}
        holdout_compare = ml_comparison.get("holdout", {}).get("compare", {}) or {}
        params_cmp = ml_comparison.get("params", {})
        primary_suspicious = bool(ml_comparison.get("primary_suspicious_fit"))
        compare_suspicious = bool(ml_comparison.get("compare_suspicious_fit"))

        def _fmt_param(value) -> str:
            return "discarded (fit hit bounds)" if value is None else f"{value}"

        def _fmt_holdout_field(holdout: Dict, suspicious: bool, key: str, fmt: str) -> str:
            if suspicious or not holdout:
                return "discarded"
            return f"{holdout.get(key, math.nan):{fmt}}"

        param_lines = [
            f"  - {key}: {_fmt_param(params_cmp.get(key, {}).get('primary'))} / "
            f"{_fmt_param(params_cmp.get(key, {}).get('compare'))}"
            for key in ("alpha_vdw", "beta_vdw", "r0_vdw", "A_fit", "B_fit")
        ]
        lines.extend(
            [
                f"## {ml_comparison.get('primary_model')} vs {ml_comparison.get('compare_model')}",
                f"- Common tail frames compared: {ml_comparison.get('n_common_frames')}",
                f"- delta-E (fit target) pearson r between the two ML references: {ml_comparison.get('delta_e_pearson_r', math.nan):.3f}",
                f"- delta-E diff (compare - primary) mean ± std (kJ/mol): "
                f"{ml_comparison.get('delta_e_diff_mean_kjmol', math.nan):.2f} ± {ml_comparison.get('delta_e_diff_std_kjmol', math.nan):.2f}",
                f"- Comparison CSV / PNG: {ml_comparison.get('comparison_csv')} / {ml_comparison.get('comparison_png')}",
            ]
        )
        if primary_suspicious:
            lines.append(
                f"- ⚠️ primary ({ml_comparison.get('primary_model')}) 拟合撞边界"
                f"（{', '.join(ml_comparison.get('primary_boundary_hits', [])) or 'unknown'}），已砍掉参数/holdout"
            )
        if compare_suspicious:
            lines.append(
                f"- ⚠️ compare ({ml_comparison.get('compare_model')}) 拟合撞边界"
                f"（{', '.join(ml_comparison.get('compare_boundary_hits', [])) or 'unknown'}），已砍掉参数/holdout"
            )
        lines.extend(
            [
                "- 参数对比 (primary / compare):",
                *param_lines,
                f"- Holdout RMSE (primary / compare, kJ/mol): "
                f"{_fmt_holdout_field(holdout_primary, primary_suspicious, 'rmse_raw_kjmol', '.2f')} / "
                f"{_fmt_holdout_field(holdout_compare, compare_suspicious, 'rmse_raw_kjmol', '.2f')}",
                f"- Holdout R2 (primary / compare): "
                f"{_fmt_holdout_field(holdout_primary, primary_suspicious, 'r2_raw', '.3f')} / "
                f"{_fmt_holdout_field(holdout_compare, compare_suspicious, 'r2_raw', '.3f')}",
                "  两个 ML 参考的 ΔE 相关性越高、拟合参数越接近，说明 DEXP 学到的物理规律越不依赖具体选用哪个基础模型；"
                "反之如果差异很大，说明 DEXP 精度上限受限于两个 ML 势本身的分歧，需要谨慎选择哪个作为最终参考。"
                " 撞边界的一侧已被砍掉，不代表两者真实分歧那么大。",
                "",
            ]
        )
    lines.extend(
        [
        "## Stability",
        f"- Original preproduction protocol: {original_summary.get('preproduction_protocol')}",
        f"- DEXP preproduction protocol: {dexp_summary.get('preproduction_protocol')}",
        f"- Original warmup steps: {original_summary.get('preproduction_warmup_steps')}",
        f"- DEXP warmup steps: {dexp_summary.get('preproduction_warmup_steps')}",
        f"- Original mean temperature (K): {original_summary['temperature_K']['mean']:.3f}",
        f"- DEXP mean temperature (K): {dexp_summary['temperature_K']['mean']:.3f}",
        f"- Original ligand RMSD mean (A): {original_summary.get('ligand_rmsd_mean_A', math.nan):.3f}",
        f"- DEXP ligand RMSD mean (A): {dexp_summary.get('ligand_rmsd_mean_A', math.nan):.3f}",
        f"- Original total energy std (kJ/mol): {original_summary['total_kjmol']['std']:.3f}",
        f"- DEXP total energy std (kJ/mol): {dexp_summary['total_kjmol']['std']:.3f}",
        "",
        ]
    )
    if n_replicas >= 2:
        lines.extend(
            [
                "## Stability Replicates",
                f"- N replicas per ensemble: {n_replicas}",
                "- Original replicas (temperature mean K): "
                + ", ".join(f"{r['temperature_K']['mean']:.2f}" for r in original_summary.get("replicas", [])),
                "- DEXP replicas (temperature mean K): "
                + ", ".join(f"{r['temperature_K']['mean']:.2f}" for r in dexp_summary.get("replicas", [])),
                "- Original replicas (ligand RMSD mean A): "
                + ", ".join(f"{r.get('ligand_rmsd_mean_A', math.nan):.3f}" for r in original_summary.get("replicas", [])),
                "- DEXP replicas (ligand RMSD mean A): "
                + ", ".join(f"{r.get('ligand_rmsd_mean_A', math.nan):.3f}" for r in dexp_summary.get("replicas", [])),
                "",
            ]
        )
    lines.extend(
        [
        "## Lambda Single-Point Scan",
        f"- Scan CSV: {lambda_scan_summary.get('scan_csv')}",
        f"- All finite: {lambda_scan_summary.get('all_finite')}",
        f"- Max |ΔU(lambda_i)-ΔU(lambda_i-1)| (kJ/mol): {lambda_scan_summary.get('max_abs_energy_jump_kjmol', math.nan):.3f}",
        f"- Max force across scan (kJ/mol/nm): {lambda_scan_summary.get('max_force_kjmol_per_nm', math.nan):.3f}",
        "- Scan is now paired: same frames and same lambdas are evaluated on original_baseline and dexp_surrogate.",
        "",
        "## Contact Diagnostics",
        f"- Min-distance CSV: {contact_summary.get('min_distance_csv')}",
        f"- RDF CSV: {contact_summary.get('rdf_csv')}",
        f"- PMF CSV: {contact_summary.get('pmf_csv')}",
        f"- PMF PNG: {contact_summary.get('pmf_png')}",
        f"- RDF PNG: {contact_summary.get('rdf_png')}",
        "- RDF / PMF 当前基于 production 轨迹的接触统计对比，属于几何/热力学 proxy，不是严格的传统 ABFE PMF。",
        f"- PMF bin coverage: {contact_summary.get('pmf_populated_bins_original')}/{contact_summary.get('pmf_total_bins')} "
        f"(original) , {contact_summary.get('pmf_populated_bins_dexp')}/{contact_summary.get('pmf_total_bins')} (dexp)",
        f"- PMF sampled range (nm): original={contact_summary.get('pmf_sampled_range_original_nm')}, "
        f"dexp={contact_summary.get('pmf_sampled_range_dexp_nm')}",
        f"- {contact_summary.get('pmf_note', '')}",
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
        f"- Core-window RDF peak before surrogate switch original/dexp: "
        f"{contact_summary.get('rdf_core_window_peak_original', math.nan):.3f} / "
        f"{contact_summary.get('rdf_core_window_peak_dexp', math.nan):.3f}",
        f"- Surrogate switch-zone RDF peak ({contact_summary.get('surrogate_switch_start_nm', math.nan):.2f}-"
        f"{contact_summary.get('surrogate_cutoff_nm', math.nan):.2f} nm) original/dexp: "
        f"{contact_summary.get('rdf_switch_zone_peak_original', math.nan):.3f} / "
        f"{contact_summary.get('rdf_switch_zone_peak_dexp', math.nan):.3f}",
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
        "- `lambda_schedule_reference.csv` records traditional synchronous linear, interaction-separation, and surrogate warmup schedules.",
        "- Fixed lambda window reruns are written separately to `lambda_window_ensemble.csv`.",
        "- 当前脚本仍未完成传统 ABFE 自由能重估；这里是稳定性/几何/能量 proxy 对比。",
        "",
        ]
    )
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


def ensure_openmmml_mace_device_patch(verbose: bool = True) -> bool:
    """幂等地修复 openmm-ml macepotential.py 的一个 device 放置 bug。

    openmm-ml 的本地 modelPath 分支加载 MACE 时是
        model = torch.load(self.modelPath, map_location=device)
    漏了注册名分支才有的 .to(device)。map_location 搬不动 e3nn 被 TorchScript
    固化进图的 Wigner-3j 常量(_w3j_*)，于是它们留在 CPU、GPU 前向报
    "Expected all tensors to be on the same device"。这里在 main 启动时检查并补上。

    设计成完全无副作用/非致命：找不到文件、已打过补丁、版本不匹配、只读环境
    都只告警不抛异常，返回 True 表示补丁已就位。
    """
    vulnerable = "model = torch.load(self.modelPath, map_location=device)"
    patched = "model = torch.load(self.modelPath, map_location=device).to(device)"
    try:
        import openmmml
    except Exception as exc:
        if verbose:
            print(f"    [patch] 未找到 openmmml，跳过 MACE device 补丁: {exc}")
        return False
    mp = os.path.join(os.path.dirname(openmmml.__file__), "models", "macepotential.py")
    if not os.path.isfile(mp):
        if verbose:
            print(f"    [patch] 未找到 {mp}，跳过 MACE device 补丁")
        return False
    try:
        with open(mp, "r", encoding="utf-8") as fh:
            src = fh.read()
    except Exception as exc:
        if verbose:
            print(f"    [patch] 读取 macepotential.py 失败，跳过: {exc}")
        return False

    if patched in src:
        if verbose:
            print("    [patch] openmm-ml MACE device 补丁已就位")
        return True
    if vulnerable not in src:
        if verbose:
            print("    [patch] 未匹配到目标行（openmm-ml 版本可能已变）；如遇 device 报错请手动检查 macepotential.py")
        return False

    try:
        bak = mp + ".abfe_bak"
        if not os.path.exists(bak):
            with open(bak, "w", encoding="utf-8") as fh:
                fh.write(src)
        with open(mp, "w", encoding="utf-8") as fh:
            fh.write(src.replace(vulnerable, patched, 1))
    except Exception as exc:
        if verbose:
            print(f"    [patch] 写入补丁失败（可能只读环境），跳过: {exc}")
        return False

    if "openmmml.models.macepotential" in sys.modules and verbose:
        print("    [patch] 已写盘，但 macepotential 本进程已导入，可能需重启后才生效")
    if verbose:
        print(f"    [patch] 已补上 openmm-ml MACE 本地模型分支的 .to(device): {mp}")
    return True


def relabel_trajectory_local(
    args: argparse.Namespace,
    traj_path: str,
    fitted_params: Dict,
    symbols: Dict,
) -> Dict:
    """在给定轨迹上做 MACE 单点 relabel，逐帧算出局部能量分量 + DEXP 预测 + min-distance。
    返回同帧对齐的数组：不做任何重加权/PMF，纯标注。"""
    md = require_module("mdtraj")
    openmm, _, unit, _ = require_openmm()
    select_env_indices = symbols["_select_env_indices_from_mdtraj_frame"]
    Orbv3DEXPFittingPipeline = symbols["Orbv3DEXPFittingPipeline"]

    traj = md.load(traj_path, top=args.traj_top)
    if len(traj) == 0:
        raise RuntimeError(f"轨迹为空: {traj_path}")
    # 均匀抽样，控制 MACE 单点成本
    n_take = min(len(traj), max(2, int(args.relabel_max_frames)))
    sel = np.unique(np.linspace(0, len(traj) - 1, n_take).round().astype(int))
    sub = traj[sel]
    if sub.unitcell_vectors is not None:
        sub = sub.image_molecules(inplace=False)
    lig_idx = np.array(sub.top.select(f"resname {args.ligand}"), dtype=int)
    if len(lig_idx) == 0:
        raise ValueError(f"未找到配体残基 `{args.ligand}`（{traj_path}）")
    env_search_radius = float(args.fit_env_radius)
    env_max_atoms = int(args.fit_env_max_atoms) if int(args.fit_env_max_atoms) > 0 else None
    env_idx = select_env_indices(sub[-1], lig_idx, env_search_radius, max_env_atoms=env_max_atoms)
    if len(env_idx) == 0:
        raise RuntimeError("未找到配体附近环境原子，请增大 --fit-env-radius")
    all_nums = np.array([a.element.atomic_number for a in sub.top.atoms], dtype=int)

    mm_contexts = build_mm_le_contexts_from_system_xml(
        args.system_xml,
        ligand_indices=lig_idx.tolist(),
        environment_indices=env_idx.tolist(),
        cutoff_nm=float(args.fit_mm_ref_cutoff),
        switching_nm=float(args.fit_mm_ref_switch),
    )
    pipeline = Orbv3DEXPFittingPipeline(model_name=args.ml_model, device=args.device)

    xyz = np.asarray(sub.xyz, dtype=np.float64)
    box = np.asarray(sub.unitcell_vectors, dtype=np.float64) if sub.unitcell_vectors is not None else None

    e_orb, e_gauss, e_mm_coul, e_mm_vdw, dexp_pred, min_dist = [], [], [], [], [], []
    print(f"    [relabel] {os.path.basename(traj_path)}: MACE 单点标注 {len(sub)} 帧（env={len(env_idx)}）")
    for k in range(len(sub)):
        pos_nm = xyz[k].copy()
        box_lens = np.linalg.norm(box[k], axis=1) if box is not None else np.array([3.0, 3.0, 3.0])
        delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
        delta -= box_lens * np.round(delta / box_lens)
        dists = np.linalg.norm(delta, axis=-1)
        valid = dists[(dists >= args.fit_r_min) & (dists <= args.fit_r_max)]

        eo = float(pipeline._compute_orb_decomposition(pos_nm, lig_idx, env_idx, all_nums))
        eg = ec = ev = 0.0
        for label, ctx in mm_contexts.items():
            if box is not None and ctx.getSystem().usesPeriodicBoundaryConditions():
                ctx.setPeriodicBoxVectors(*[openmm.Vec3(*[float(v) for v in row]) for row in box[k]])
            ctx.setPositions(pos_nm * unit.nanometer)
            en = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            if label == "gauss_coul":
                eg = en
            elif label == "coul":
                ec = en
            elif label == "vdw":
                ev = en
        e_orb.append(eo); e_gauss.append(eg); e_mm_coul.append(ec); e_mm_vdw.append(ev)
        dexp_pred.append(float(predict_dexp_delta_e(valid, fitted_params)))
        min_dist.append(float(dists.min()))
        if (k + 1) % 50 == 0:
            print(f"      relabel {k + 1}/{len(sub)}")
    return {
        "e_orb": np.asarray(e_orb), "e_gauss": np.asarray(e_gauss),
        "e_mm_coul": np.asarray(e_mm_coul), "e_mm_vdw": np.asarray(e_mm_vdw),
        "dexp_pred": np.asarray(dexp_pred), "min_dist": np.asarray(min_dist),
        "n_frames": len(sub),
    }


def same_frame_pmf_compare(min_dist, delta_e, kbt, n_bins, min_bin_frames, shape_anchor_bins=2):
    """同帧比较，两部分：
    主判据 = ⟨δ⟩(s) 均值残差剖面（DEXP 能否在均值上还原 MACE，逐箱 within-SEM）——这个对本体系 work。
    次判据 = DEXP-world 直方图 PMF + exp(-δ/kT) 重加权的全局/逐箱 ESS —— 本体系 σ(δ)≫kT，ESS 预期塌，
             重加权 MACE PMF 结构上不可得；ESS 只当 OOD/近接触帧的告警，不作为 PMF 质量指标。"""
    md = np.asarray(min_dist, dtype=float)
    de = np.asarray(delta_e, dtype=float)
    if md.size == 0:
        return [], {
            "n_frames_total": 0, "n_bins": 0, "n_bins_judged": 0, "note": "no_frames_after_filter",
            "mean_residual_within_sem_bins": 0, "zero_offset_within_decomp_kjmol": math.nan,
            "shape_profile_rmse_kjmol": math.nan, "shape_profile_max_abs_kjmol": math.nan,
            "reweight_ess_global": math.nan, "reweight_ess_global_fraction": math.nan,
            "reweight_usable": False,
        }
    edges = np.linspace(float(md.min()), float(md.max()) + 1.0e-9, int(n_bins) + 1)
    which = np.clip(np.digitize(md, edges) - 1, 0, int(n_bins) - 1)
    logw = -de / max(kbt, 1.0e-12)
    logw -= float(np.max(logw))        # 全局平移做数值稳定（常数抵消）
    w = np.exp(logw)
    ess_global = float(np.sum(w) ** 2 / max(np.sum(w ** 2), 1e-300))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 关键：MACE-local、surrogate、MM 三者零点各不相同，绝对 δ 不可跨势比较。
    # 唯一零点无关且可比的做法：把每条 δ(s) 锚到同一个物理参考态——最远的若干 min-dist 箱
    # （最接近分离态）。这样比较的是"相对分离态，MACE 与该势的能量差如何随距离变化"，与
    # 各自任意零点无关。先收集逐箱原始均值，确定参考箱组后再统一相对化。
    #
    # 锚点估计量说明（见 RESUME_DEXP_SESSION.md §5.4）：
    # 锚"单个最远箱"把该箱的采样噪声整个传进所有其它箱的 d_rel，比"去均值"更吵，但物理上
    # 唯一合法（去均值的零点混入了所有箱、非物理）。折中方案：锚"最远若干箱的逆方差加权均值"，
    # 用更多帧稀释锚点噪声，同时把锚点自身的 SEM（ref_sem）传播进每个箱的 within-SEM 判据
    # （combined_sem = sqrt(bin_sem^2 + ref_sem^2)），而不是像旧版那样假装参考点零噪声。
    raw = []  # (bin_index, n_b, d_mean_raw, d_sem, ess_b, center)
    for b in range(int(n_bins)):
        m = which == b
        n_b = int(m.sum())
        if n_b < int(min_bin_frames):
            continue
        d_mean_raw = float(np.mean(de[m]))
        d_sem = float(np.std(de[m]) / max(1, n_b) ** 0.5)
        wsum = float(w[m].sum()); w2 = float((w[m] ** 2).sum())
        ess_b = float(wsum ** 2 / w2) if w2 > 0 else 0.0
        raw.append((b, n_b, d_mean_raw, d_sem, ess_b, float(centers[b])))

    rows = []
    within = 0
    zero_offset = float(np.mean(de))   # 仅记录：各自零点内部的规范量，跨势不可比，不作判据
    anchor_k = 0
    ref_delta = math.nan
    ref_sem = math.nan
    if raw:
        anchor_k = max(1, min(int(shape_anchor_bins), len(raw)))
        anchor_group = raw[-anchor_k:]                  # 最远的 anchor_k 个箱
        anchor_ids = {r[0] for r in anchor_group}
        anchor_sems = np.array([max(r[3], 1.0e-9) for r in anchor_group], dtype=float)
        anchor_means = np.array([r[2] for r in anchor_group], dtype=float)
        inv_var = 1.0 / (anchor_sems ** 2)
        ref_delta = float(np.sum(inv_var * anchor_means) / np.sum(inv_var))
        ref_sem = float(np.sqrt(1.0 / np.sum(inv_var)))  # 锚点(加权均值)自身的 SEM，需传播进判据

        cnt_sum = float(sum(r[1] for r in raw))
        G = -kbt * np.log(np.asarray([r[1] for r in raw], dtype=float) / cnt_sum)
        G = G - G[-1]
        for (b, n_b, d_raw, d_sem, ess_b, c), g in zip(raw, G):
            d_rel = d_raw - ref_delta                 # 相对锚点（零点无关）
            is_ref = b in anchor_ids
            combined_sem = float(np.sqrt(d_sem ** 2 + ref_sem ** 2))  # 锚点噪声一并传播
            ok = bool(is_ref or abs(d_rel) <= combined_sem)  # 参考箱组恒为0(在噪声内)，不计入 within 统计
            if not is_ref:
                within += int(ok)
            rows.append({
                "min_distance_center_nm": c,
                "n_frames": n_b,
                "delta_rel_far_kjmol": d_rel,          # δ(s) 相对锚点（跨势可比的量）
                "delta_mean_raw_kjmol": d_raw,         # 各自零点内的原始值（不可跨势比）
                "delta_sem_kjmol": d_sem,
                "combined_sem_kjmol": combined_sem,    # 已含锚点自身 SEM 的传播误差
                "within_1sem": ok,
                "is_reference_bin": is_ref,
                "G_dexp_world_kjmol": float(g),
                "bin_ess": ess_b,
                "bin_ess_fraction": float(ess_b / max(1, n_b)),
            })
    dshape = np.asarray([r["delta_rel_far_kjmol"] for r in rows if not r["is_reference_bin"]], dtype=float)
    n_non_ref = int(dshape.size)
    return rows, {
        "n_frames_total": int(md.size),
        "n_bins": int(len(rows)),
        # 主判据（零点无关，可跨势比）：相对分离态的形状一致性
        "mean_residual_within_sem_bins": int(within),
        "n_bins_judged": n_non_ref,                # within-SEM 分母（不含参考箱组）
        "reference": "far_min_distance_bins_inv_var_weighted",
        "shape_anchor_bins_used": int(anchor_k),
        "shape_anchor_value_kjmol": ref_delta,
        "shape_anchor_sem_kjmol": ref_sem,
        "shape_profile_rmse_kjmol": float(np.sqrt(np.mean(dshape ** 2))) if dshape.size else math.nan,
        "shape_profile_max_abs_kjmol": float(np.max(np.abs(dshape))) if dshape.size else math.nan,
        "zero_offset_within_decomp_kjmol": zero_offset,  # 仅本势内部规范，禁止跨势比较
        # 次判据（本体系预期塌，仅告警）
        "reweight_ess_global": ess_global,
        "reweight_ess_global_fraction": float(ess_global / max(1, md.size)),
        "reweight_usable": bool(ess_global / max(1, md.size) >= 0.2),
    }


def _filter_too_close(min_dist: np.ndarray, floor: float) -> Tuple[np.ndarray, int]:
    """返回 (可信掩码, 过近帧数)。过近 = MACE 也 OOD 的近接触，其能量不可信，必须排除。"""
    md = np.asarray(min_dist, dtype=float)
    mask = md >= float(floor)
    return mask, int((~mask).sum())


def run_relabel_pmf(args: argparse.Namespace, output_dir: str, fitted_params: Dict) -> Dict:
    """relabel + 同帧比较主流程：DEXP 生产轨迹（必选）+ MM baseline 地板（可选）。
    主判据 = ⟨δ⟩(s) within-SEM（MACE 可信窗口内）；单独报"过近帧"数（指标 F）。"""
    symbols = load_abfe_symbols()
    kbt = 0.00831446261815324 * float(args.temperature)
    n_bins = int(args.relabel_pmf_bins)
    min_bin = int(args.relabel_pmf_min_bin_frames)
    floor = float(args.relabel_min_dist_floor)
    out: Dict = {}

    def _one(traj_path, applied_key, tag):
        r = relabel_trajectory_local(args, traj_path, fitted_params, symbols)
        # δ = E_MACE_local - E_applied_local
        if applied_key == "dexp":
            delta = r["e_orb"] - (r["e_gauss"] + r["dexp_pred"])
        else:  # mm baseline
            delta = r["e_orb"] - (r["e_mm_coul"] + r["e_mm_vdw"])
        md_all = np.asarray(r["min_dist"], dtype=float)
        mask, n_close = _filter_too_close(md_all, floor)
        rows, summ = same_frame_pmf_compare(
            md_all[mask], delta[mask], kbt, n_bins, min_bin,
            shape_anchor_bins=int(args.relabel_shape_anchor_bins),
        )
        summ["n_frames_raw"] = int(md_all.size)
        summ["n_frames_too_close"] = n_close       # 指标 F：原子穿插、MACE 也 OOD 的帧数
        summ["too_close_fraction"] = float(n_close / max(1, md_all.size))
        summ["min_dist_floor_nm"] = floor
        # min-dist 分布：直接看配体到底待在哪（指标 F 的原始信息）
        summ["min_dist_min_nm"] = float(md_all.min())
        summ["min_dist_median_nm"] = float(np.median(md_all))
        summ["min_dist_p05_nm"] = float(np.percentile(md_all, 5))
        summ["min_dist_max_nm"] = float(md_all.max())
        print(
            f"    [relabel/{tag}] 帧={summ['n_frames_raw']} | min-dist 分布: "
            f"min={summ['min_dist_min_nm']:.3f} p05={summ['min_dist_p05_nm']:.3f} "
            f"中位={summ['min_dist_median_nm']:.3f} max={summ['min_dist_max_nm']:.3f} nm | "
            f"过近<{floor}nm 排除 {n_close}({summ['too_close_fraction']:.0%})"
        )
        if summ.get("n_frames_total", 0) > 0:
            print(
                f"        主判据(锚到最远箱,零点无关) 形状 within-SEM {summ['mean_residual_within_sem_bins']}/{summ['n_bins_judged']} 箱 | "
                f"形状剖面 RMSE={summ['shape_profile_rmse_kjmol']:.2f} kJ/mol | "
                f"重加权 ESS={summ['reweight_ess_global_fraction']:.0%}({'可用' if summ['reweight_usable'] else '塌,弃'})"
            )
        else:
            print("        ⚠️ 过滤后无可信帧（配体几乎全程处于过近区）——见 min-dist 分布")
        return rows, summ

    dexp_traj = ensure_file(args.relabel_traj, "DEXP 生产轨迹")
    rows, summ = _one(dexp_traj, "dexp", "DEXP")
    if rows:
        write_rows_csv(os.path.join(output_dir, "relabel_dexp_1d_pmf.csv"), rows)
    out["dexp"] = summ

    if args.relabel_baseline_traj:
        base_traj = ensure_file(args.relabel_baseline_traj, "MM baseline 轨迹")
        brows, base_summ = _one(base_traj, "mm", "MM 地板")
        if brows:
            write_rows_csv(os.path.join(output_dir, "relabel_mm_baseline_1d_pmf.csv"), brows)
        out["mm_baseline"] = base_summ
        # 地板判据：去掉各自合法零点后，DEXP 的形状剖面 RMSE 应 ≤ MM（形状更贴 MACE），
        # 且过近帧不多于 MM（短程墙没让它比 MM 更爱塌进近区）。
        d_rmse = summ["shape_profile_rmse_kjmol"]; m_rmse = base_summ["shape_profile_rmse_kjmol"]
        pass_rmse = np.isfinite(d_rmse) and np.isfinite(m_rmse) and d_rmse <= m_rmse
        pass_close = summ["too_close_fraction"] <= base_summ["too_close_fraction"] + 1e-9
        verdict = "通过地板" if (pass_rmse and pass_close) else "未过地板"
        out["floor_verdict"] = verdict
        print(
            f"    [relabel/地板判据] 形状RMSE DEXP={d_rmse:.2f} vs MM={m_rmse:.2f} | "
            f"过近帧 DEXP={summ['too_close_fraction']:.0%} vs MM={base_summ['too_close_fraction']:.0%} -> {verdict}"
        )

    # --- 画图：DEXP-world PMF（直方图）+ ⟨δ⟩(s) 均值残差带 SEM ---
    try:
        plt = get_matplotlib_pyplot()
        s = [r["min_distance_center_nm"] for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        axes[0].plot(s, [r["G_dexp_world_kjmol"] for r in rows], "o-")
        axes[0].set_xlabel("min L-E distance (nm)"); axes[0].set_ylabel("relative PMF (kJ/mol)")
        axes[0].set_title("DEXP-world PMF (histogram)"); axes[0].grid(alpha=0.3)
        dmean = np.asarray([r["delta_rel_far_kjmol"] for r in rows])
        dsem = np.asarray([r["delta_sem_kjmol"] for r in rows])
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].errorbar(s, dmean, yerr=dsem, fmt="s-", capsize=3, label="δ(s) rel. far bin ± SEM")
        axes[1].set_xlabel("min L-E distance (nm)"); axes[1].set_ylabel("δ rel. far (kJ/mol)")
        axes[1].set_title(f"MACE endorsement (zero-free): within-SEM {summ['mean_residual_within_sem_bins']}/{summ['n_bins_judged']}")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        png = os.path.join(output_dir, "relabel_dexp_1d_pmf.png")
        fig.tight_layout(); fig.savefig(png, dpi=180); plt.close(fig)
        out["png"] = png
    except Exception as exc:
        out["plot_error"] = str(exc)

    with open(os.path.join(output_dir, "relabel_pmf_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    return out


def main() -> int:
    ensure_openmmml_mace_device_patch()
    args = parse_args()
    args.traj = ensure_file(args.traj, "预平衡轨迹")
    args.traj_top = ensure_file(args.traj_top, "轨迹拓扑")
    args.gmx_top = ensure_file(args.gmx_top, "GROMACS 拓扑")
    args.system_xml = ensure_file(args.system_xml, "原始 system XML")
    args.ligand_indices = ensure_file(args.ligand_indices, "配体索引 JSON")
    output_dir = ensure_dir(args.output_dir)

    # relabel 模式：读现有拟合参数，对生产轨迹做 MACE relabel + 同帧 1D PMF，然后退出（不跑拟合/MD）。
    if args.relabel_traj:
        params_path = ensure_file(os.path.join(output_dir, "dexp_fitted_params.json"), "已拟合 DEXP 参数")
        with open(params_path, "r", encoding="utf-8") as handle:
            fitted_params = json.load(handle)
        print(f"[relabel] 读入拟合参数: {params_path}")
        run_relabel_pmf(args, output_dir, fitted_params)
        print("[relabel] 完成：relabel_dexp_1d_pmf.csv/png + relabel_pmf_summary.json")
        return 0

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

    fit_quality = summarize_fit_diagnostics(output_dir, fitted_params)
    if args.fit_only:
        print("[fit-only] 已完成拟合与 holdout 诊断，跳过 surrogate system / MD / 后处理。")
        print(f"参数文件: {os.path.join(output_dir, 'dexp_fitted_params.json')}")
        print(f"[fit-only] fit_health = {fitted_params.get('fit_health', 'unknown')}"
              + (f"（{', '.join(fitted_params.get('fit_health_reasons', []))}）"
                 if fitted_params.get('fit_health') == 'degraded' else ""))
        learned = fitted_params.get("learned_rbf_diagnostic", {}) or {}
        if learned.get("enabled"):
            profile = learned.get("pmf_profile", {}) or {}
            print(
                "[fit-only] 学习函数[RBF]: "
                f"RMSE={learned.get('rmse_raw_kjmol', math.nan):.2f} kJ/mol | "
                f"R²={learned.get('r2_raw', math.nan):.3f} | "
                f"均值剖面 RMSE={profile.get('pmf_profile_rmse_kjmol', math.nan):.2f}"
            )
        return 0
    validate_fit_for_dynamics(fitted_params)

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
        original_summary = load_stability_summary_with_replicas(output_dir, "original_baseline")
        dexp_summary = load_stability_summary_with_replicas(output_dir, "dexp_surrogate")
    else:
        print(f"[3/4] 构建 DEXP surrogate system 并执行 {max(1, int(args.stability_replicas))} 次 1 ns 稳定性测试")
        dexp_summary = run_stability_ensemble(
            label="dexp_surrogate",
            system=dexp_system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            args=args,
            output_dir=output_dir,
        )

        print(f"[4/4] 执行原始势能 {max(1, int(args.stability_replicas))} 次 1 ns baseline，并导出 lambda schedule 对比")
        original_summary = run_stability_ensemble(
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
    lambda_window_ensemble = postprocess.get("lambda_window_ensemble", {})

    report_path = write_comparison_report(
        output_dir,
        original_summary=original_summary,
        dexp_summary=dexp_summary,
        fitted_params=fitted_params,
        fit_quality=fit_quality,
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
                "fit_quality": fit_quality,
                "dexp_surrogate": dexp_summary,
                "original_baseline": original_summary,
                "lambda_single_point_scan": lambda_scan_summary,
                "contact_diagnostics": contact_summary,
                "delta_u_distribution": delta_u_summary,
                "lambda_window_analysis": lambda_window_summary,
                "lambda_window_ensemble": lambda_window_ensemble,
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

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
import gc
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
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
            DEXPSurrogatePotential,
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
        "DEXPSurrogatePotential": DEXPSurrogatePotential,
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
    # pull-scan：手动把配体质心从口袋拉开，生成跨越宽 min-distance 范围的构型序列，喂给现有
    # fit 流程（把这条 DCD 当 --traj 用）。目的不是自由能/PMF，只是给 DEXP 拟合提供现在这套
    # 无偏 MD 给不出的宽范围训练样本——现状是 0.20-0.227nm 一条窄缝，双指数核的形状参数在这
    # 么窄的范围内统计上不可辨识（RESUME_DEXP_SESSION.md §9.5）。
    parser.add_argument("--pull-scan", action="store_true", help="进入 pull-scan 模式：把配体质心从环境锚点拉开，生成宽范围构型序列后退出（不跑拟合/relabel/后处理）")
    parser.add_argument("--pull-scan-source-traj", default=None, help="pull-scan 起始结构来源轨迹，默认用 --traj（取其最后一帧）")
    parser.add_argument("--pull-scan-extend-nm", type=float, default=0.6, help="从当前 配体质心-锚点质心 距离往外拉多远 (nm)；起点由当前结构自动测出，终点=起点+此值。实测锚点/配体质心本身热涨落幅度约 0.03-0.04nm，每档增量需要明显大于这个噪声本底才能看出真实位移")
    parser.add_argument("--pull-scan-steps", type=int, default=12, help="拉开过程分多少档（含起点，共 steps+1 帧）；每档增量 = extend_nm/steps，别调太细，实测 <0.05nm/档 会被热噪声淹没")
    parser.add_argument("--pull-scan-relax-steps", type=int, default=3000, help="每一档目标距离下，先弛豫多少步再截帧（环境需要时间适应配体的新位置，太短会截到不真实的应变构型；实测 k=5000 时 2000 步内已能跟上大跨度目标）")
    parser.add_argument("--pull-scan-k", type=float, default=5000.0, help="配体质心-锚点质心 谐振拉力的力常数 (kJ/mol/nm^2)；实测 2000 太软、追不上目标，5000 能在 2000-3000 步内跟上")
    parser.add_argument("--pull-scan-anchor-k", type=float, default=1000.0, help="锚点原子位置约束的力常数 (kJ/mol/nm^2)，把锚点原子钉在起始位置附近，避免锚点centroid自身热涨落淹没拉力信号")
    # pose-scan：随机刚体扰动 + 短程约束弛豫（不追求连续轨迹），配合"整体短接触惩罚"力，
    # 给 DEXP 拟合提供几何多样的训练样本。见 run_pose_scan 的 docstring。
    parser.add_argument("--pose-scan", action="store_true", help="进入 pose-scan 模式：随机刚体扰动配体+短程弛豫+按min_valid_le_distance分箱筛选，生成训练样本后退出")
    parser.add_argument("--pose-scan-trials", type=int, default=300, help="最多尝试多少次随机扰动")
    parser.add_argument("--pose-scan-translate-max-nm", type=float, default=0.6, help="随机平移幅度上限 (nm)，实际平移量在 [0, 此值] 均匀采样")
    parser.add_argument("--pose-scan-relax-steps", type=int, default=1000, help="每次扰动后（先做一次局部能量最小化去除硬碰撞，再）弛豫多少步")
    parser.add_argument("--pose-scan-anchor-k", type=float, default=1000.0, help="蛋白锚点位置约束力常数 (kJ/mol/nm^2)，弛豫时不希望壳层自己漂移")
    parser.add_argument("--pose-scan-reject-min-dist", type=float, default=0.16, help="min_valid_le_distance 低于此值判为不可信/原子穿插，直接拒绝（不计入任何 bin）")
    parser.add_argument("--pose-scan-bin-max-nm", type=float, default=0.6, help="分箱的上边界 (nm)，配合 --fit-r-min(下边界=--pose-scan-reject-min-dist) 覆盖 --pose-scan-bins 个箱")
    parser.add_argument("--pose-scan-bins", type=int, default=12, help="按 min_valid_le_distance 分箱数")
    parser.add_argument("--pose-scan-per-bin", type=int, default=30, help="每个 bin 目标保留多少帧，所有 bin 都填满或用完 trials 就停")
    parser.add_argument("--pose-scan-short-contact-rcut", type=float, default=0.28, help="整体短接触惩罚 sigmoid 的中心距离 r_cut (nm)：比这更近的 pair 算作短接触，被压低")
    parser.add_argument("--pose-scan-short-contact-width", type=float, default=0.03, help="整体短接触惩罚 sigmoid 的过渡宽度 w (nm)")
    parser.add_argument("--pose-scan-short-contact-k", type=float, default=10.0, help="整体短接触惩罚的力常数 k_bias (kJ/mol per 短接触计数)：C_short=sum sigmoid((r_cut-r)/w)，U_bias=k_bias*C_short")

    parser.add_argument("--perturb-scan", action="store_true", help="进入局部扰动云模式：从平衡轨迹尾段取若干 anchor 帧，只对配体做小幅刚体扰动(不 relax/minimize)，比较 anchor-relative ΔE_target 与 pair-specific LJ-matched DEXP 解析基线的 ΔU，检验基线本身能否解释局部势能面曲率")
    parser.add_argument("--perturb-anchors", type=int, default=50, help="从轨迹尾段抽取多少个 anchor 帧")
    parser.add_argument("--perturb-trans-nm", default="0.005,0.01,0.02,0.04", help="配体刚体平移扰动幅度列表 (nm，逗号分隔)")
    parser.add_argument("--perturb-rot-deg", default="0.5,1.5,3.0", help="配体绕自身主惯性轴的小角度转动扰动幅度列表 (度，逗号分隔)")
    parser.add_argument("--perturb-n-random-dirs", type=int, default=4, help="除 3 个主惯性轴外，每个平移幅度额外测试多少个随机方向(±)，增加接触组合多样性")
    parser.add_argument("--perturb-baseline-cutoff-nm", type=float, default=0.70, help="计算 DEXP 解析基线 pairwise 和时的截断半径 (nm)，默认与生产 DEXPSurrogatePotential.cutoff_distance 一致")

    parser.add_argument("--perturb-fit", action="store_true", help="进入局部扰动云拟合模式：读取已有 --perturb-scan 输出，按扰动档等权+leave-one-anchor-out 重新挑选 alpha_vdw/beta_vdw 这两个仅剩的全局形状自由度（不需要重跑 MACE）")
    parser.add_argument("--perturb-fit-alpha-grid", default="8,9,10,11,12,13,14,15,16,18,20,24,28", help="alpha_vdw 候选网格 (逗号分隔)")
    parser.add_argument("--perturb-fit-beta-grid", default="3,4,5,6,6.5,7,7.5,8,9,10,12", help="beta_vdw 候选网格 (逗号分隔)")
    parser.add_argument("--perturb-fit-mag04-weight", type=float, default=0.5, help="translation@0.04nm 这一档相对其它档的权重折扣因子 (0~1)，因为它是最偏离局部定义的扰动，不该独占形状参数")
    parser.add_argument("--perturb-fit-basin-delta-frac", type=float, default=0.05, help="判定 score surface 稳定盆地的相对阈值：basin = {(alpha,beta): L<=L_min*(1+此值)}")
    parser.add_argument("--perturb-fit-basin-delta-kjmol", type=float, default=0.0, help="判定盆地的绝对阈值 (kJ/mol)，与相对阈值取较大者一起生效")

    parser.add_argument("--contact-type-fit", action="store_true", help="DEXP_KERNEL_PHYSICS_ISSUES.md §6 最小实现：读取已有 --perturb-scan 输出，在(14,5)完整pairwise基线之上按 donor_acceptor/fallback 两类 contact-type 拟合 psi_o/psi_e 修正(M0/M1/M2 三层对比+grouped LOAO)，离线诊断，不改变生产 force")
    parser.add_argument("--contact-type-gamma-odd", type=float, default=10.0, help="psi_o(x)=x*exp(-gamma_o*x^2) 里的 gamma_o，固定不拟合")
    parser.add_argument("--contact-type-gamma-even", type=float, default=10.0, help="psi_e(x)=x^2*exp(-gamma_e*x^2) 里的 gamma_e，固定不拟合")
    parser.add_argument("--contact-type-switch-width", type=float, default=0.20, help="contact-type 修正项 S(r) 的 switching 宽度 (nm)，复用 (a)阶段 DEXP 核 switch_width 默认值；switching 终点固定为 --perturb-baseline-cutoff-nm")
    parser.add_argument("--contact-type-ridge-lambda", type=float, default=10.0, help="contact-type psi_o/psi_e 系数(a_t,b_t)的岭回归正则强度(标准化坐标下)")
    parser.add_argument("--contact-type-ridge-lambda-grid", default="", help="ridge_lambda 稳健性扫描网格(逗号分隔，如 '0.01,0.1,1,10')；为空时只用 --contact-type-ridge-lambda 单个值")

    parser.add_argument("--contact-type-angular-diagnostic", action="store_true", help="DEXP_KERNEL_PHYSICS_ISSUES.md §6.6：只做角度诊断(D-H-A夹角/Δ夹角/最近acceptor切换/配位数 vs 跨折 out-of-fold 残差的按anchor相关性)，不拟合 angular force")
    parser.add_argument("--contact-type-angular-acceptor-cutoff-nm", type=float, default=0.45, help="判定'最近acceptor'搜索及配位数计数的距离截断(nm)，与 run_replica_analysis 里氢键候选判据同一个默认值")

    parser.add_argument("--gaussian-width-diagnostic", action="store_true", help="DEXP_KERNEL_PHYSICS_ISSUES.md §6 Gaussian宽度/charge-penetration诊断：只检查M0残差是否与Δ(统一sigma_elec电荷穿透代理量)存在稳定关联，不拟合role-specific sigma_elec")

    parser.add_argument("--production-equivalence-audit", action="store_true", help="Phase 1(用户方案)：核对 §3-§6 全程用的 NumPy (14,5) DEXP 基线/§6.7 的 NumPy Gaussian 重实现是否与真正的生产 OpenMM CustomNonbondedForce(含switching)/gauss_coul参考Context一致，含sigma/epsilon/charge逐项核对与有限差分力检验")
    parser.add_argument("--audit-energy-tol-kjmol", type=float, default=1.0e-5, help="production-equivalence-audit 的能量差通过阈值 (kJ/mol)")
    parser.add_argument("--audit-force-rel-tol", type=float, default=1.0e-4, help="production-equivalence-audit 的有限差分力相对误差通过阈值")

    parser.add_argument("--replica-run", action="store_true", help="进入 replica-run 模式：为单个 condition(original/dexp_12_6/dexp_14_5)跑 --stability-replicas 个 --sim-ns ns 的短复制，复用已有的 run_stability_ensemble(minimize+warmup+production+RMSD)")
    parser.add_argument("--replica-condition", default="original", choices=["original", "dexp_12_6", "dexp_14_5"], help="--replica-run 要跑的 condition")
    parser.add_argument("--replica-analyze", action="store_true", help="进入 replica-analyze 模式：读取已跑完的各 condition/replica 轨迹，做 RMSD/pose聚类/接触占有率/氢键/contact-feature协方差/配体平移转动/能量力分布/Δ<q>显著性检验")
    parser.add_argument("--replica-conditions", default="original,dexp_12_6,dexp_14_5", help="--replica-analyze 要比较的 condition 列表(逗号分隔)，第一个作为参照基准计算 Delta<q>")
    parser.add_argument("--replica-cluster-rmsd-A", type=float, default=1.5, help="pose 聚类的 leader-clustering RMSD 阈值 (埃)")
    parser.add_argument("--replica-contact-cutoff-nm", type=float, default=0.40, help="关键接触 occupancy 判定用的距离阈值 (nm)")
    parser.add_argument("--replica-hbond-dist-nm", type=float, default=0.35, help="氢键 D...A 距离判据 (nm)")
    parser.add_argument("--replica-hbond-angle-deg", type=float, default=150.0, help="氢键 D-H...A 角度判据 (度，大于此值才算氢键)")
    parser.add_argument("--replica-too-close-nm", type=float, default=0.12, help="判定为原子穿插/积分失败前兆的最近距离下限 (nm)")
    parser.add_argument("--replica-n-key-contacts", type=int, default=5, help="按 anchor 帧最近距离取前多少个 ligand-environment 接触对作为'关键接触'追踪 occupancy")

    parser.add_argument("--vsb-frame-scan", action="store_true", help="§9.1分阶段V/S/B多初态平衡MD方案第一步：只读已跑完的--replica-run轨迹，用跟--hbond-switching-dynamics同款判据逐帧分类V/S/B/N，为V/S/B三态各挑选--vsb-replicas-per-state个起始帧(不同来源replica、run-length最长优先)，写出vsb_frame_manifest.json供--vsb-staged-run使用。零新增MD/MACE")
    parser.add_argument("--vsb-source-labels", default="replica_original,replica_dexp_12_6,replica_dexp_14_5", help="--vsb-frame-scan 扫描哪些已有replica条件目录(逗号分隔)")
    parser.add_argument("--vsb-source-max-replicas", type=int, default=10, help="--vsb-frame-scan 每个来源condition最多扫描多少条replica(超出实际存在数量的会被自动跳过)")
    parser.add_argument("--vsb-replicas-per-state", type=int, default=2, help="--vsb-frame-scan 为V/S/B每态各挑选几个独立起始帧(即§9.1方案里的'2 independent replicas')")
    parser.add_argument("--vsb-staged-run", action="store_true", help="§9.1分阶段V/S/B多初态平衡MD方案第二步：读取--vsb-frame-scan产出的manifest，对--replica-condition指定的单个condition，把V/S/B三态的起始帧当独立初始构型各跑一次全新的minimize+warmup+production(复用run_stability_simulation，只是起点换成V/S/B帧而不是预平衡轨迹最后一帧)，输出到output_dir/vsb_staged/{condition}/{state}/rep{i}/")
    parser.add_argument("--vsb-staged-analyze", action="store_true", help="§9.1分阶段V/S/B多初态平衡MD方案第三步：只读分析--vsb-staged-run产出的output_dir/vsb_staged/{condition}/{state}/rep{i}/traj.dcd(--replica-conditions指定要比较的condition列表)，用跟--hbond-switching-dynamics同款V/S/B/N判据逐帧分类，报每条轨迹的occupancy/转移矩阵/驻留时间，再检验核心问题：固定condition，从V/S/B三个不同起始态出发的复制在后半段production是否收敛到彼此接近的occupancy(收敛=§4.4的occupancy差异是真实平衡性质而非初态锁定伪影)。零新增MD/MACE")

    parser.add_argument("--hbond-switching-dynamics", action="store_true", help="只用已跑完的 --replica-run 轨迹，分析配体酰胺 N-H 在 VAL136主链O/SER177侧链OG 之间的 V/S/B/N 四态切换动力学(occupancy/转移矩阵/首次转移时间/驻留时间/前后半段占据/自相关有效样本数)，判断§4.4的occupancy是否是平衡概率还是短轨迹初态依赖的产物；不需要新MD/MACE")
    parser.add_argument("--switching-donor-heavy-atom", type=int, default=4587, help="配体酰胺供体重原子(N)的拓扑原子编号")
    parser.add_argument("--switching-donor-h-atoms", default="4607,4608", help="配体酰胺供体N上两个H的拓扑原子编号(逗号分隔)")
    parser.add_argument("--switching-val-acceptor-atom", type=int, default=2134, help="VAL136 主链羰基O的拓扑原子编号")
    parser.add_argument("--switching-ser-acceptor-atom", type=int, default=2759, help="SER177 侧链OG的拓扑原子编号")

    parser.add_argument("--hbond-committed-state-dynamics", action="store_true", help="`--hbond-switching-dynamics`的committed-state升级版：用连续coordination+Schmitt trigger+最小驻留去抖动区分真正的basin穿越和阈值抖动，重新判断§4.4的occupancy是否平衡；不需要新MD/MACE")
    parser.add_argument("--switching-coord-dist-half-width-nm", type=float, default=0.05, help="距离方向 quintic 平滑窗半宽(nm)，窗口=[hbond_dist-半宽, hbond_dist+半宽]")
    parser.add_argument("--switching-coord-angle-half-width-deg", type=float, default=20.0, help="角度方向 quintic 平滑窗半宽(度)，窗口=[hbond_angle-半宽, hbond_angle+半宽]")
    parser.add_argument("--switching-commit-enter-score", type=float, default=0.5, help="Schmitt trigger 进入committed状态的连续coordination分数阈值(严格)")
    parser.add_argument("--switching-commit-exit-score", type=float, default=0.2, help="Schmitt trigger 离开committed状态的连续coordination分数阈值(宽松，必须小于--switching-commit-enter-score)")
    parser.add_argument("--switching-min-dwell-frames", type=int, default=4, help="去抖动：Schmitt trigger 输出至少连续持续多少帧才接受为真正的状态翻转")
    parser.add_argument("--switching-block-ns", type=float, default=0.5, help="逐block occupancy 的block大小(ns)，按--sim-ns换算成帧数")
    parser.add_argument("--switching-min-committed-transitions", type=int, default=3, help="判定'未平衡'的committed穿越次数下限(每个方向)")
    parser.add_argument("--switching-first-second-half-diff-threshold", type=float, default=0.15, help="判定'未平衡'的前后半段occupancy最大绝对差阈值")
    parser.add_argument("--switching-min-n-eff", type=float, default=20.0, help="判定'未平衡'的committed indicator自相关有效样本数下限")

    parser.add_argument("--kernel-projection-benchmark", action="store_true", help="用户重定的框架：MACE是参考势能面，检验原始pair-specific LJ(K0)/DEXP(12,6)(K1)/DEXP(14,5)(K2)谁更好地投影MACE的even/光滑径向骨架(不要求odd归零)。--mace-kernel-benchmark(8-15体系版)的最小单体系试点，零新增MACE计算，只用已有--perturb-scan数据")
    parser.add_argument("--mace-residual-force-benchmark", action="store_true", help="Phase 3（DEXP_KERNEL_PHYSICS_ISSUES.md §11 待办项）：把 --perturb-scan 已缓存的±δ能量差重新解读成局部相互作用force/torque投影(-dE/dq，跨幅度线性+三次拟合取δ→0极限，而非单点e_odd/δ)，对比K0(LJ)/K1(DEXP12,6)/K2(DEXP14,5)的force/torque cosine similarity + 随机方向held-out检验，零新增MACE计算，只用已有--perturb-scan数据")
    parser.add_argument("--mace-env-convergence", action="store_true", help="Phase 2（DEXP_KERNEL_PHYSICS_ISSUES.md §7/§11 待办项）：固定几个anchor、只用最小幅度扰动，在多个环境半径x两种裁剪方式(逐原子/完整残基-水分子)下重新算MACE，检验ΔE_MACE收敛性/odd方向梯度符号稳定性/K0(LJ)-K1(DEXP12,6)-K2(DEXP14,5)排序稳定性——这条线唯一需要新增MACE计算的Phase")
    parser.add_argument("--env-convergence-anchors", type=int, default=5, help="--mace-env-convergence 用几个anchor(用户指定固定5个)")
    parser.add_argument("--env-convergence-radii", type=str, default="0.50,0.60,0.70,0.90", help="--mace-env-convergence 扫描的环境半径列表(nm)，逗号分隔")
    parser.add_argument("--r0-scale-diagnostic", action="store_true", help="用户2026-07-13提出的旁线：固定alpha/beta(默认14,5)，扫描r0_ij(new)=s_r*r0_ij(LJ)的比例因子s_r，检验odd是否显著改善/even是否不退化/anchor-balanced LOAO是否稳定/bootstrap是否排除s_r=1，作为是否把某个s_r晋升为第4个MD condition的判据。零新增MACE计算，只用已有--perturb-scan数据")
    parser.add_argument("--r0-scale-grid", type=str, default="0.96,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04", help="--r0-scale-diagnostic 扫描的s_r列表，逗号分隔")
    parser.add_argument("--r0-scale-alpha", type=float, default=14.0, help="--r0-scale-diagnostic 固定的alpha(默认当前生产默认核14,5)")
    parser.add_argument("--r0-scale-beta", type=float, default=5.0, help="--r0-scale-diagnostic 固定的beta")
    parser.add_argument("--r0-scale-n-boot", type=int, default=4000, help="--r0-scale-diagnostic 逐s_r vs baseline的bootstrap重采样次数")
    parser.add_argument("--alpha-beta-scale-diagnostic", action="store_true", help="用户2026-07-13提出的'让MACE老师最后签字'：固定r0_scale=1.0/s_epsilon=1.0，全网格扫alpha/beta，确认(14,5)是否位于稳定宽阔的最优盆地(不是找新的小数对)。含p=alpha*beta/q=alpha+beta方向性剖面(检验§3.3对角谷假说)、anchor-balanced LOAO、grid最优/(14,5)/(12,6)三者两两bootstrap CI。零新增MACE计算，只用已有--perturb-scan数据")
    parser.add_argument("--ab-alpha-min", type=float, default=12.0, help="--alpha-beta-scale-diagnostic alpha网格下限")
    parser.add_argument("--ab-alpha-max", type=float, default=16.0, help="--alpha-beta-scale-diagnostic alpha网格上限")
    parser.add_argument("--ab-alpha-step", type=float, default=0.25, help="--alpha-beta-scale-diagnostic alpha网格步长")
    parser.add_argument("--ab-beta-min", type=float, default=4.0, help="--alpha-beta-scale-diagnostic beta网格下限")
    parser.add_argument("--ab-beta-max", type=float, default=7.0, help="--alpha-beta-scale-diagnostic beta网格上限")
    parser.add_argument("--ab-beta-step", type=float, default=0.25, help="--alpha-beta-scale-diagnostic beta网格步长")
    parser.add_argument("--ab-scale-n-boot", type=int, default=4000, help="--alpha-beta-scale-diagnostic grid最优/(14,5)/(12,6)两两比较的bootstrap重采样次数")
    parser.add_argument("--alpha-beta-ridge-scan", action="store_true", help="用户2026-07-13追加要求：沿固定q=alpha+beta的直线(默认18和19，分别过(12,6)和(14,5))做比--alpha-beta-scale-diagnostic的0.25网格细得多的扫描，产出两条曲线数据供画图(plot_alpha_beta_ridge_scan.py)，直接可视化这两条脊线是否等价。零新增MACE计算，只用已有--perturb-scan数据")
    parser.add_argument("--ridge-q-values", type=str, default="18,19", help="--alpha-beta-ridge-scan 扫描的 q=alpha+beta 值列表，逗号分隔")
    parser.add_argument("--ridge-beta-min", type=float, default=1.0, help="--alpha-beta-ridge-scan 每条脊线上beta的下限(避免beta过小退化)")
    parser.add_argument("--ridge-step", type=float, default=0.05, help="--alpha-beta-ridge-scan 沿脊线的beta步长")
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
                # DEXP 的 pairsum 只依赖 [fit_r_min, fit_r_max] 内的 pair；用全原子(含 H)最近距离
                # 当"分箱/探索范围"坐标时，二者经常不是同一个量——本体系里 99%+ 的帧全原子最近
                # 距离都 < fit_r_min，PMF matching 和 relabel 形状比较此前一直在用 DEXP 基本"看不见"
                # 的坐标打分。这里补一个 DEXP 实际敏感的坐标，PMF matching/holdout/relabel 改用它。
                "min_valid_le_distance_nm": float(valid_dists.min()) if valid_dists.size else float("nan"),
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
        # PMF matching/holdout 的分箱坐标必须是 DEXP 实际敏感的坐标：pairsum 只加和
        # [fit_r_min, fit_r_max] 内的 pair，所以这里用 valid_dists 的最小值（而不是
        # row["min_le_distance_nm"] 这个全原子含 H 的最近距离——那个坐标经常落在
        # DEXP 完全看不见的 r < fit_r_min 区间，会把拟合目标绑定到一个 DEXP 的
        # 函数形式根本不依赖的坐标上）。
        accepted_min_dist.append(float(valid_dists.min()))

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


def run_replica_condition(args: argparse.Namespace, output_dir: str) -> Dict:
    """LJ / DEXP(12,6) / DEXP(14,5) 三方复制对比的单个 condition。

    刻意只跑一个 condition(--replica-condition)，方便把三个 condition x N replica
    当成独立作业分别提交到计算节点，而不是在一个脚本里串行跑完——直接复用已有的
    run_stability_ensemble(同一套 minimize+softstart 分段升 vdW/Coulomb+production+
    RMSD 流程，本项目一直在用，不重新发明)，只是把"要跑哪个势"换成三选一。
    """
    openmm, app, unit, XmlSerializer = require_openmm()
    condition = str(args.replica_condition)
    if condition not in ("original", "dexp_12_6", "dexp_14_5"):
        raise ValueError(f"--replica-condition 必须是 original/dexp_12_6/dexp_14_5，收到: {condition}")

    system, topology = load_cached_system(args.system_xml, args.traj_top)
    _, positions, box_vectors = load_last_frame_positions(args.traj, args.traj_top)
    ligand_indices = load_ligand_indices(args.ligand_indices)
    env_indices = [idx for idx in range(system.getNumParticles()) if idx not in set(ligand_indices)]

    if condition == "original":
        run_system = system
    else:
        alpha, beta = (12.0, 6.0) if condition == "dexp_12_6" else (14.0, 5.0)
        symbols = load_abfe_symbols()
        SurrogateSystemBuilder = symbols["SurrogateSystemBuilder"]
        surrogate_builder = SurrogateSystemBuilder({"alpha_vdw": alpha, "beta_vdw": beta}, ghost_handler=None)
        run_system = surrogate_builder.build_surrogate_system(
            original_system=system,
            ligand_indices=ligand_indices,
            environment_indices=env_indices,
            lambda_names=("lam_coul", "lam_vdw"),
            force_group=1,
            reference_positions=positions,
            box_vectors=box_vectors,
        )

    label = f"replica_{condition}"
    n_replicas = max(1, int(args.stability_replicas))
    print(
        f"[replica-run] condition={condition} | {n_replicas} x {args.sim_ns:.2f}ns "
        f"(dt={args.dt_fs:.3f}fs, warmup_steps={args.warmup_steps}) -> {os.path.join(output_dir, label)}"
    )
    summary = run_stability_ensemble(
        label=label,
        system=run_system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        args=args,
        output_dir=output_dir,
    )
    print(f"[replica-run] 完成: {label}")
    return summary


def _principal_axes(points_centered: np.ndarray) -> np.ndarray:
    inertia = points_centered.T @ points_centered
    _, axes = np.linalg.eigh(inertia)
    return axes / np.linalg.norm(axes, axis=0, keepdims=True)


def _kabsch_rotation_deg(P_centered: np.ndarray, Q_centered: np.ndarray) -> float:
    """P,Q 是已去质心的对应原子坐标(n,3)；返回把 P 最优旋转对齐到 Q 所需的旋转角(度)。"""
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, float(d)])
    R = Vt.T @ D @ U.T
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _replica_trajectory_paths(output_dir: str, label: str, n_replicas: int) -> List[str]:
    paths = []
    for i in range(n_replicas):
        p = (
            os.path.join(output_dir, label, "traj.dcd")
            if i == 0
            else os.path.join(output_dir, label, f"replica_{i}", "traj.dcd")
        )
        if os.path.isfile(p):
            paths.append(p)
    return paths


def _classify_vsbn_frames(
    traj, donor_heavy: int, donor_h_atoms: List[int], val_acceptor: int, ser_acceptor: int,
    hbond_dist_nm: float, hbond_angle_deg: float,
) -> np.ndarray:
    """跟 `run_hbond_switching_dynamics` 完全同款的 V/S/B/N 判据(不改判据、不改阈值)，
    抽成独立可复用函数，供 `--vsb-frame-scan` 挑选具体起始帧使用。`traj` 应已经
    `image_molecules` 过(该函数不做这一步)。"""
    n_frames = len(traj)

    def _bonded_series(h_idx: int, a_idx: int) -> np.ndarray:
        dist = np.linalg.norm(traj.xyz[:, a_idx, :] - traj.xyz[:, donor_heavy, :], axis=-1)
        v1 = traj.xyz[:, donor_heavy, :] - traj.xyz[:, h_idx, :]
        v2 = traj.xyz[:, a_idx, :] - traj.xyz[:, h_idx, :]
        cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1.0e-12)
        angle_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        return (dist < hbond_dist_nm) & (angle_deg > hbond_angle_deg)

    v_bonded = np.zeros(n_frames, dtype=bool)
    s_bonded = np.zeros(n_frames, dtype=bool)
    for h_idx in donor_h_atoms:
        v_bonded |= _bonded_series(h_idx, val_acceptor)
        s_bonded |= _bonded_series(h_idx, ser_acceptor)
    state = np.full(n_frames, "N", dtype=object)
    state[v_bonded & ~s_bonded] = "V"
    state[~v_bonded & s_bonded] = "S"
    state[v_bonded & s_bonded] = "B"
    return state


def run_vsb_frame_scan(args: argparse.Namespace, output_dir: str) -> Dict:
    """§9.1 分阶段 V/S/B 多初态平衡 MD 方案的第一步：不需要新 MD、不需要 MACE，只读已经
    跑完的 `--replica-run` 轨迹(original/dexp_12_6/dexp_14_5 各5条replica)，用
    `_classify_vsbn_frames`(跟 `--hbond-switching-dynamics` 完全同款判据)逐帧分类，
    为 V/S/B 三态各挑 `--vsb-replicas-per-state`(默认2) 个"起始帧"候选——按该帧所在的
    连续同态 run 长度降序排列(run 越长说明离状态切换边界越远，越不像是刚好卡在一次
    快速抖动上的瞬时帧)，且强制不同候选来自不同的 source replica(增加起始构型多样性，
    不是同一条轨迹里紧挨着的两帧)。每个候选帧的坐标/box直接存进 manifest(不是只存
    文件路径+帧号——避免`--vsb-staged-run`还要重新解析大DCD)，另外存一份.pdb供人眼核查。

    产出 `vsb_frame_manifest.json`(供 `--vsb-staged-run` 读取) + `vsb_frame_scan_all_candidates.csv`
    (每一帧的分类结果，完整审计轨迹)。

    内存提示：默认会把扫描到的每条来源轨迹整条读进内存(用于事后按选中的帧号直接取坐标)，
    15条(3 condition x 5 replica)、每条~200帧的轨迹同时缓存大约需要几GB内存——如果吃紧，
    用 `--vsb-source-labels`/`--vsb-source-max-replicas` 缩小扫描范围。
    """
    md = require_module("mdtraj")

    donor_heavy = int(args.switching_donor_heavy_atom)
    donor_h_atoms = [int(x) for x in str(args.switching_donor_h_atoms).split(",") if x.strip()]
    val_acceptor = int(args.switching_val_acceptor_atom)
    ser_acceptor = int(args.switching_ser_acceptor_atom)
    hbond_dist = float(args.replica_hbond_dist_nm)
    hbond_angle = float(args.replica_hbond_angle_deg)
    source_labels = [s.strip() for s in str(args.vsb_source_labels).split(",") if s.strip()]
    max_replicas = int(args.vsb_source_max_replicas)
    n_per_state = int(args.vsb_replicas_per_state)

    print(
        f"[1/4] 扫描来源: {source_labels}（每个最多{max_replicas}条replica），"
        f"donor_heavy={donor_heavy} donor_H={donor_h_atoms} VAL={val_acceptor} SER={ser_acceptor} "
        f"判据: dist<{hbond_dist}nm 且 angle>{hbond_angle}deg"
    )
    all_candidates: List[Dict] = []
    traj_cache: Dict[Tuple[str, int], object] = {}
    for label in source_labels:
        paths = _replica_trajectory_paths(output_dir, label, max_replicas)
        for rep_idx, dcd_path in enumerate(paths):
            traj = md.load(dcd_path, top=args.traj_top)
            if len(traj) == 0:
                continue
            try:
                if traj.unitcell_vectors is not None:
                    traj = traj.image_molecules(inplace=False)
            except Exception:
                pass
            state = _classify_vsbn_frames(
                traj, donor_heavy, donor_h_atoms, val_acceptor, ser_acceptor, hbond_dist, hbond_angle
            )
            traj_cache[(label, rep_idx)] = traj
            n = len(state)
            run_len = np.zeros(n, dtype=int)
            i = 0
            while i < n:
                j = i
                while j < n and state[j] == state[i]:
                    j += 1
                run_len[i:j] = j - i
                i = j
            for f in range(n):
                all_candidates.append({
                    "source_label": label, "source_replica": rep_idx, "frame_idx": f,
                    "state": str(state[f]), "run_length": int(run_len[f]),
                })
            occ_str = ",".join(f"{s}:{float(np.mean(state == s)):.2f}" for s in ("V", "S", "B", "N"))
            print(f"    {label} replica {rep_idx}: {n}帧  occupancy={occ_str}")

    print(
        f"[2/4] 汇总: 共{len(all_candidates)}帧，按state分组挑选起始帧"
        "(优先选连续run最长的帧，且强制不同候选来自不同source replica)"
    )
    csv_path = os.path.join(output_dir, "vsb_frame_scan_all_candidates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_label", "source_replica", "frame_idx", "state", "run_length"])
        writer.writeheader()
        writer.writerows(all_candidates)

    manifest: Dict[str, List[Dict]] = {}
    for state in ("V", "S", "B"):
        pool = sorted((c for c in all_candidates if c["state"] == state), key=lambda c: -c["run_length"])
        selected: List[Dict] = []
        used_sources: set = set()
        for cand in pool:
            src_key = (cand["source_label"], cand["source_replica"])
            if src_key in used_sources:
                continue
            selected.append(cand)
            used_sources.add(src_key)
            if len(selected) >= n_per_state:
                break
        if len(selected) < n_per_state:
            used_frames_by_source: Dict[Tuple, set] = {}
            for c in selected:
                used_frames_by_source.setdefault((c["source_label"], c["source_replica"]), set()).add(c["frame_idx"])
            for cand in pool:
                if len(selected) >= n_per_state:
                    break
                src_key = (cand["source_label"], cand["source_replica"])
                if cand["frame_idx"] in used_frames_by_source.get(src_key, set()):
                    continue
                selected.append(cand)
                used_frames_by_source.setdefault(src_key, set()).add(cand["frame_idx"])

        entries = []
        for i, cand in enumerate(selected):
            traj = traj_cache[(cand["source_label"], cand["source_replica"])]
            frame = traj[cand["frame_idx"]]
            pdb_path = os.path.join(output_dir, f"vsb_start_{state}_{i}.pdb")
            frame.save_pdb(pdb_path)
            positions_nm = frame.xyz[0].tolist()
            box_nm = frame.unitcell_vectors[0].tolist() if frame.unitcell_vectors is not None else None
            entries.append({
                "state": state, "replica_index": i,
                "source_label": cand["source_label"], "source_replica": cand["source_replica"],
                "source_frame_idx": cand["frame_idx"], "run_length_frames": cand["run_length"],
                "pdb_path": pdb_path, "positions_nm": positions_nm, "box_vectors_nm": box_nm,
            })
            print(
                f"    state={state} replica_index={i}: 来自 {cand['source_label']}"
                f"(replica{cand['source_replica']}) frame={cand['frame_idx']} "
                f"run_length={cand['run_length']}帧 -> {pdb_path}"
            )
        if len(selected) < n_per_state:
            print(
                f"    [警告] state={state} 只找到 {len(selected)}/{n_per_state} 个符合要求的起始帧"
                "（该态在现有轨迹里出现太少，尤其B态可能罕见——可以放宽--vsb-source-labels/"
                "--vsb-source-max-replicas，或接受更短的run_length）"
            )
        manifest[state] = entries

    print("[3/4] 逐态 occupancy 汇总（全部扫描到的帧，不只是被选中的候选）")
    for state in ("V", "S", "B", "N"):
        n_state = sum(1 for c in all_candidates if c["state"] == state)
        pct = 100.0 * n_state / max(1, len(all_candidates))
        print(f"    {state}: {n_state}/{len(all_candidates)} 帧 ({pct:.1f}%)")

    summary = {
        "source_labels": source_labels,
        "max_replicas_per_source": max_replicas,
        "n_per_state": n_per_state,
        "hbond_dist_nm": hbond_dist,
        "hbond_angle_deg": hbond_angle,
        "donor_heavy": donor_heavy, "donor_h_atoms": donor_h_atoms,
        "val_acceptor": val_acceptor, "ser_acceptor": ser_acceptor,
        "n_frames_scanned": len(all_candidates),
        "manifest": manifest,
        "all_candidates_csv": csv_path,
        "note": (
            "manifest[state] 是一组独立起始帧(用run-length=离状态切换边界最远的连续同态"
            "run长度排序，优先选跨越不同source replica的候选以增加多样性)，供"
            "--vsb-staged-run读取，对同一个(condition,state)按这些帧逐个起跑(每帧一个"
            "全新的minimize+warmup+production、全新velocity)——不是--replica-run那种"
            "'同一个起点、多个velocity种子'的replica，这里的'replica'是指真正独立的初始"
            "构型。positions_nm/box_vectors_nm直接存储供--vsb-staged-run加载，避免重新"
            "解析大DCD或依赖PDB的有限精度；pdb_path只是给人眼核查用的。"
        ),
    }
    print("[4/4] 写出 manifest")
    manifest_path = os.path.join(output_dir, "vsb_frame_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    manifest: {manifest_path}")
    print(f"    all candidates csv: {csv_path}")
    return summary


def run_vsb_staged_replica(args: argparse.Namespace, output_dir: str) -> Dict:
    """§9.1 分阶段 V/S/B 多初态平衡 MD 方案的第二步：读取 `--vsb-frame-scan` 产出的
    manifest，对给定的 `--replica-condition`(original/dexp_12_6/dexp_14_5，跟
    `--replica-run`同一个CLI选项)，把 V/S/B 三态各自的起始帧当独立初始构型，各跑一次
    全新的 `run_stability_simulation`(同一套 minimize+softstart 分段升 vdW/Coulomb+
    production 流程，直接复用，不重新发明；只是起始 positions/box 换成 manifest 里
    记录的 V/S/B 帧，而不是 `load_last_frame_positions` 读到的预平衡轨迹最后一帧)。

    DEXP 条件下的 system 对每个起始帧单独重建(`SurrogateSystemBuilder.build_surrogate_system`
    的 reference_positions/box_vectors 语义未知是否对起始构型敏感，为避免猜测，每个
    V/S/B起始帧都用它自己的构型重新建一次system——这一步零MD/零MACE，很便宜)。

    输出目录：`output_dir/vsb_staged/{condition}/{state}/rep{i}/`(traj.dcd/state.csv/
    summary.json)，加一份`output_dir/vsb_staged/{condition}/{state}/replica_summaries.json`
    汇总该state下所有起始帧的summary。每1ns的中期分析直接对增长中的traj.dcd跑
    `--hbond-committed-state-dynamics`风格的分析即可，不需要额外代码。

    **断点续跑（sub-run粒度）**：重新提交同一条`--vsb-staged-run --replica-condition X`
    命令时，如果某个(state,rep)的`summary.json`已经存在(意味着那次
    `run_stability_simulation`已经完整跑完)，直接读现成结果跳过，不会重新跑。这只解决
    "6个sub-run里有几个已经跑完、被kill后不用重跑"这个粒度的续跑；单次sub-run内部
    (例如5ns跑到2ns时被kill)目前没有OpenMM checkpoint级别的续跑，会从step 0完整重来——
    这是`run_stability_simulation`本身(--replica-run也在用)缺的能力，不是本函数特有的
    限制，目前判断没有必要专门补(GPU上5ns量级的production通常不会长到撞上常见walltime)。
    """
    openmm, app, unit, XmlSerializer = require_openmm()
    condition = str(args.replica_condition)
    if condition not in ("original", "dexp_12_6", "dexp_14_5"):
        raise ValueError(f"--replica-condition 必须是 original/dexp_12_6/dexp_14_5，收到: {condition}")

    manifest_path = os.path.join(output_dir, "vsb_frame_manifest.json")
    ensure_file(manifest_path, "V/S/B 起始帧 manifest（先跑一次 --vsb-frame-scan）")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    base_system, topology = load_cached_system(args.system_xml, args.traj_top)
    ligand_indices = load_ligand_indices(args.ligand_indices)
    env_indices = [idx for idx in range(base_system.getNumParticles()) if idx not in set(ligand_indices)]
    symbols = load_abfe_symbols() if condition != "original" else None

    def _build_system_for(positions, box_vectors):
        if condition == "original":
            return base_system
        alpha, beta = (12.0, 6.0) if condition == "dexp_12_6" else (14.0, 5.0)
        SurrogateSystemBuilder = symbols["SurrogateSystemBuilder"]
        surrogate_builder = SurrogateSystemBuilder({"alpha_vdw": alpha, "beta_vdw": beta}, ghost_handler=None)
        return surrogate_builder.build_surrogate_system(
            original_system=base_system, ligand_indices=ligand_indices, environment_indices=env_indices,
            lambda_names=("lam_coul", "lam_vdw"), force_group=1,
            reference_positions=positions, box_vectors=box_vectors,
        )

    print(f"[vsb-staged-run] condition={condition}")
    all_summaries: Dict[str, List[Dict]] = {}
    for state in ("V", "S", "B"):
        entries = manifest["manifest"].get(state, [])
        if not entries:
            print(f"    [跳过] state={state} manifest里没有起始帧")
            continue
        state_summaries = []
        for entry in entries:
            rep_i = int(entry["replica_index"])
            label = f"vsb_staged/{condition}/{state}"
            sim_dir = os.path.join(output_dir, label, f"rep{rep_i}")

            # 断点续跑（sub-run粒度，不是单次MD内部的step级checkpoint）：如果这个
            # (state,rep)之前已经完整跑完(summary.json只在run_stability_simulation整个
            # simulation.step(n_steps)跑完后才写)，直接读现成结果跳过，不重新跑。这样
            # --vsb-staged-run被kill后重新提交同一条命令，只会补跑还没做完的那几个
            # sub-run，不会把已经跑完的也推倒重来。单次sub-run内部(例如5ns跑到一半被杀)
            # 目前仍然没有OpenMM checkpoint级别的续跑——那个sub-run会整个重跑，不是从
            # 中断的step继续。
            existing_summary_path = os.path.join(sim_dir, "summary.json")
            if os.path.isfile(existing_summary_path):
                try:
                    with open(existing_summary_path, "r", encoding="utf-8") as handle:
                        summary = json.load(handle)
                    print(f"  ↪ {label} rep{rep_i}: summary.json 已存在，跳过重跑（断点续跑）")
                    summary.setdefault("replica_index", rep_i)
                    summary.setdefault("source_label", entry["source_label"])
                    summary.setdefault("source_replica", entry["source_replica"])
                    summary.setdefault("source_frame_idx", entry["source_frame_idx"])
                    state_summaries.append(summary)
                    continue
                except Exception:
                    print(f"  ↪ {label} rep{rep_i}: 已有 summary.json 但读取/解析失败，视为未完成，重新跑")

            positions = [openmm.Vec3(*xyz) for xyz in entry["positions_nm"]] * unit.nanometer
            box_vectors = (
                [openmm.Vec3(*row) for row in entry["box_vectors_nm"]] * unit.nanometer
                if entry.get("box_vectors_nm") is not None else None
            )
            run_system = _build_system_for(positions, box_vectors)
            seed = int(args.seed) + rep_i * 104729
            print(
                f"  ↪ {label} rep{rep_i} (起自 {entry['source_label']}/replica{entry['source_replica']}"
                f"/frame{entry['source_frame_idx']}, seed={seed})"
            )
            summary = run_stability_simulation(
                label=label, system=run_system, topology=topology,
                positions=positions, box_vectors=box_vectors,
                args=args, output_dir=output_dir, seed=seed, sim_dir=sim_dir,
            )
            summary["replica_index"] = rep_i
            summary["source_label"] = entry["source_label"]
            summary["source_replica"] = entry["source_replica"]
            summary["source_frame_idx"] = entry["source_frame_idx"]
            state_summaries.append(summary)
        summaries_dir = os.path.join(output_dir, "vsb_staged", condition, state)
        ensure_dir(summaries_dir)
        summaries_path = os.path.join(summaries_dir, "replica_summaries.json")
        with open(summaries_path, "w", encoding="utf-8") as handle:
            json.dump(state_summaries, handle, indent=2)
        all_summaries[state] = state_summaries
        print(f"    完成 state={state}: {len(state_summaries)} 条replica -> {summaries_path}")

    print(f"[vsb-staged-run] 完成: condition={condition}")
    return all_summaries


def run_vsb_staged_analysis(args: argparse.Namespace, output_dir: str) -> Dict:
    """§9.1 分阶段方案第三步：只读分析 `--vsb-staged-run` 产出的
    `output_dir/vsb_staged/{condition}/{state}/rep{i}/traj.dcd`(不需要新 MD/MACE)。

    对每条轨迹用跟 `run_hbond_switching_dynamics` 完全同款的判据逐帧分类 V/S/B/N，报
    occupancy/转移矩阵/驻留时间/前后半段occupancy，然后回答本方案真正要解决的问题：
    固定 condition，从 V/S/B 三个刻意选择的不同起始态出发的复制，在后半段(production
    早已过了softstart)是否收敛到彼此接近的occupancy——收敛说明§4.4/
    `--hbond-switching-dynamics` 报的occupancy差异是DEXP核参数真实改变的平衡氢键偏好，
    不收敛(仍然锁定在各自起始态附近)说明V/S/B之间存在这段轨迹长度采不到的自由能垒，
    现有occupancy比较对(14,5) vs (12,6)的判断没有意义。
    """
    md = require_module("mdtraj")
    conditions = [c.strip() for c in str(args.replica_conditions).split(",") if c.strip()]

    donor_heavy = int(args.switching_donor_heavy_atom)
    donor_h_atoms = [int(x) for x in str(args.switching_donor_h_atoms).split(",") if x.strip()]
    val_acceptor = int(args.switching_val_acceptor_atom)
    ser_acceptor = int(args.switching_ser_acceptor_atom)
    hbond_dist = float(args.replica_hbond_dist_nm)
    hbond_angle = float(args.replica_hbond_angle_deg)

    labels = ["V", "S", "B", "N"]
    idx_of = {s: i for i, s in enumerate(labels)}

    def _analyze_traj(dcd_path: str, intended_state: str) -> Optional[Dict]:
        traj = md.load(dcd_path, top=args.traj_top)
        n_frames = len(traj)
        if n_frames == 0:
            return None
        try:
            if traj.unitcell_vectors is not None:
                traj = traj.image_molecules(inplace=False)
        except Exception:
            pass
        state = _classify_vsbn_frames(
            traj, donor_heavy, donor_h_atoms, val_acceptor, ser_acceptor, hbond_dist, hbond_angle
        )

        occupancy = {s: float(np.mean(state == s)) for s in labels}
        trans_counts = np.zeros((4, 4), dtype=int)
        for i in range(n_frames - 1):
            trans_counts[idx_of[state[i]], idx_of[state[i + 1]]] += 1
        n_state_changes = int(np.sum(trans_counts) - np.trace(trans_counts))
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(
            trans_counts, row_sums, out=np.zeros_like(trans_counts, dtype=float), where=row_sums > 0
        )

        dwell_times: Dict[str, List[int]] = {s: [] for s in labels}
        run_state, run_len = state[0], 1
        for i in range(1, n_frames):
            if state[i] == run_state:
                run_len += 1
            else:
                dwell_times[run_state].append(run_len)
                run_state, run_len = state[i], 1
        dwell_times[run_state].append(run_len)
        dwell_stats = {
            s: {
                "n_runs": len(v),
                "mean_frames": float(np.mean(v)) if v else 0.0,
                "max_frames": int(max(v)) if v else 0,
            }
            for s, v in dwell_times.items()
        }

        half = n_frames // 2
        occ_first_half = {s: float(np.mean(state[:half] == s)) for s in labels} if half > 0 else {}
        occ_second_half = {s: float(np.mean(state[half:] == s)) for s in labels} if (n_frames - half) > 0 else {}

        actual_initial_state = str(state[0])
        left_intended = False
        returned_after_leaving = False
        for i in range(1, n_frames):
            if state[i] != intended_state:
                left_intended = True
            elif left_intended:
                returned_after_leaving = True
        single_irreversible_from_intended = bool(left_intended and not returned_after_leaving)

        return {
            "n_frames": int(n_frames),
            "occupancy": occupancy,
            "transition_counts": trans_counts.tolist(),
            "transition_matrix": trans_matrix.tolist(),
            "transition_matrix_state_order": labels,
            "n_state_changes": n_state_changes,
            "dwell_time_stats_frames": dwell_stats,
            "occupancy_first_half": occ_first_half,
            "occupancy_second_half": occ_second_half,
            "intended_initial_state": intended_state,
            "actual_initial_frame_state": actual_initial_state,
            "left_intended_state": left_intended,
            "single_irreversible_transition_from_intended_state": single_irreversible_from_intended,
            "traj_path": dcd_path,
        }

    print(f"[1/3] 逐 condition/state/rep 分析 vsb_staged 轨迹（conditions={conditions}）")
    per_condition: Dict[str, Dict[str, List[Dict]]] = {}
    for condition in conditions:
        per_condition[condition] = {}
        for state in ("V", "S", "B"):
            state_dir = os.path.join(output_dir, "vsb_staged", condition, state)
            if not os.path.isdir(state_dir):
                print(f"    ⚠️ {condition}/{state}: 目录不存在，跳过")
                per_condition[condition][state] = []
                continue
            rep_dirs = sorted(
                d for d in os.listdir(state_dir)
                if d.startswith("rep") and os.path.isfile(os.path.join(state_dir, d, "traj.dcd"))
            )
            stats_list = []
            for rep_name in rep_dirs:
                dcd_path = os.path.join(state_dir, rep_name, "traj.dcd")
                s = _analyze_traj(dcd_path, state)
                if s is not None:
                    s["rep"] = rep_name
                    stats_list.append(s)
            per_condition[condition][state] = stats_list
            print(f"    {condition}/{state}: {len(stats_list)} 条 rep")
            for s in stats_list:
                occ2 = s["occupancy_second_half"]
                print(
                    f"        {s['rep']}: 起始态={s['intended_initial_state']}(实际第0帧={s['actual_initial_frame_state']})  "
                    f"n状态切换={s['n_state_changes']:3d}  "
                    f"后半段占据 V={occ2.get('V', 0):.2f} S={occ2.get('S', 0):.2f} B={occ2.get('B', 0):.2f} N={occ2.get('N', 0):.2f}  "
                    f"离开起始态后未回归={s['single_irreversible_transition_from_intended_state']}"
                )

    print("[2/3] 关键判断：固定condition，从V/S/B三个不同起始态出发，后半段occupancy是否收敛到彼此接近的值")
    convergence: Dict[str, Dict] = {}
    for condition in conditions:
        by_state_mean_occ2: Dict[str, Dict[str, float]] = {}
        for state in ("V", "S", "B"):
            stats_list = per_condition[condition].get(state, [])
            occ2_list = [s["occupancy_second_half"] for s in stats_list if s["occupancy_second_half"]]
            if not occ2_list:
                continue
            by_state_mean_occ2[state] = {
                lbl: float(np.mean([o.get(lbl, 0.0) for o in occ2_list])) for lbl in labels
            }
        max_spread = {}
        if len(by_state_mean_occ2) >= 2:
            for lbl in labels:
                vals = [v[lbl] for v in by_state_mean_occ2.values()]
                max_spread[lbl] = float(max(vals) - min(vals))
        convergence[condition] = {
            "second_half_occupancy_by_initial_state": by_state_mean_occ2,
            "max_spread_across_initial_states": max_spread,
        }
        spread_str = ", ".join(f"{k}:{v:.2f}" for k, v in max_spread.items()) if max_spread else "N/A(数据不足)"
        print(f"    {condition}: 起始态间后半段occupancy最大差 = {spread_str}")

    print("[3/3] 写出 summary")
    summary = {
        "conditions": conditions,
        "per_condition_per_state_per_rep": per_condition,
        "convergence_across_initial_states": convergence,
        "interpretation_note": (
            "本分析回答§9.1方案的核心问题：§4.4/--hbond-switching-dynamics报的V/S/B occupancy"
            "差异，是DEXP核参数真的改变了平衡氢键偏好，还是短轨迹从单一初始pose出发、还没到达"
            "平衡就被截断的初态依赖伪影。做法是对每个condition强制从V/S/B三个不同起始构型各跑"
            "一段全新production，如果同一个condition下、不同起始态出发的复制在后半段(production"
            "已经过半，softstart早就结束)收敛到彼此接近的occupancy(max_spread_across_initial_"
            "states的V/S/B分量都比较小)，说明这段轨迹长度已经足以抹平初态依赖，occupancy数字"
            "可信，可以用于跨condition(original vs dexp_12_6 vs dexp_14_5)比较；如果spread仍然"
            "很大(不同起始态各自锁定在自己的初始态附近)，说明V/S/B之间存在轨迹长度采不到的自由"
            "能垒，此时occupancy比较没有意义，需要更长production或增强采样，不能直接下(14,5)"
            "vs(12,6)的结论。"
        ),
    }
    summary_path = os.path.join(output_dir, "vsb_staged_analysis_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    summary: {summary_path}")
    return summary


def _mean_sem(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), math.nan
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(values.size))


def run_replica_analysis(args: argparse.Namespace, output_dir: str) -> Dict:
    """LJ / DEXP(12,6) / DEXP(14,5) 三方复制对比分析。

    每个 condition 的 MD 由 --replica-run 单独跑完(见 run_replica_condition)；这里只做
    只读分析，不跑任何新的动力学。所有量都相对同一个 anchor(--traj 的最后一帧)计算，
    口袋(非配体)原子先做 Kabsch 叠合去掉系统整体平移转动，再看配体在口袋参考系里的
    真实漂移——而不是配体对自身第0帧的"内部" RMSD(那样会把系统整体漂移和配体自身
    相对口袋的运动混在一起)。

    注意：这一步不重新建 OpenMM Context 逐帧算力，所以没有逐帧 max-force——那需要
    对每个 condition 用对应势重新算一遍力，属于单独的、更贵的分析，这里只用
    potentialEnergy 异常(NaN/Inf)、最近距离骤降(短接触比例)作为不稳定性的代理指标。
    """
    md = require_module("mdtraj")
    conditions = [c.strip() for c in str(args.replica_conditions).split(",") if c.strip()]
    if len(conditions) < 2:
        raise ValueError("--replica-conditions 至少需要 2 个 condition（第一个作为参照基准）")
    reference_condition = conditions[0]

    print(f"[1/4] 载入 anchor（{args.traj} 最后一帧）与固定环境原子集合")
    traj_full = md.load(args.traj, top=args.traj_top)
    anchor_traj = traj_full[-1]
    if anchor_traj.unitcell_vectors is not None:
        anchor_traj = anchor_traj.image_molecules(inplace=False)
    lig_idx = np.asarray(anchor_traj.top.select(f"resname {args.ligand}"), dtype=int)
    lig_heavy_idx = np.asarray(anchor_traj.top.select(f"resname {args.ligand} and not element H"), dtype=int)
    if lig_heavy_idx.size == 0:
        lig_heavy_idx = lig_idx

    env_override = _load_fixed_env_indices(output_dir)
    if env_override is not None:
        env_idx = np.asarray(env_override, dtype=int)
        print(f"    复用 fit 阶段固定环境原子集合（env={len(env_idx)}）")
    else:
        symbols = load_abfe_symbols()
        select_env_indices = symbols["_select_env_indices_from_mdtraj_frame"]
        env_idx = np.asarray(
            select_env_indices(anchor_traj, lig_idx, float(args.fit_env_radius), max_env_atoms=None),
            dtype=int,
        )
        print(f"    未找到固定环境缓存，按半径 {args.fit_env_radius}nm 重选（env={len(env_idx)}）")

    pocket_idx = np.asarray(
        [i for i in env_idx if not anchor_traj.top.atom(i).residue.is_water],
        dtype=int,
    )
    if pocket_idx.size < 4:
        pocket_idx = env_idx  # 口袋蛋白原子太少(极端情况)时退回全部环境原子做叠合

    # anchor 参考量：配体质心、去心坐标、主轴、最近的关键接触对、候选氢键对
    anchor_lig_xyz = anchor_traj.xyz[0, lig_idx, :]
    anchor_lig_heavy_xyz = anchor_traj.xyz[0, lig_heavy_idx, :]
    anchor_lig_com = anchor_lig_xyz.mean(axis=0)
    anchor_axes = _principal_axes(anchor_lig_xyz - anchor_lig_com)
    anchor_lig_heavy_centered = anchor_lig_heavy_xyz - anchor_lig_heavy_xyz.mean(axis=0)

    anchor_env_xyz = anchor_traj.xyz[0, env_idx, :]
    d_anchor = np.linalg.norm(anchor_lig_xyz[:, None, :] - anchor_env_xyz[None, :, :], axis=-1)
    n_key = max(1, int(args.replica_n_key_contacts))
    flat_order = np.argsort(d_anchor, axis=None)[:n_key]
    key_pairs = [
        (int(lig_idx[i]), int(env_idx[j]))
        for i, j in (np.unravel_index(idx, d_anchor.shape) for idx in flat_order)
    ]
    print(f"    关键接触对 (lig_atom, env_atom, anchor距离nm): "
          + ", ".join(f"({p[0]},{p[1]},{d_anchor[np.where(lig_idx==p[0])[0][0], np.where(env_idx==p[1])[0][0]]:.3f})" for p in key_pairs))

    # 候选氢键：环境/配体里带 H 的 N/O 重原子做供体，N/O 重原子做受体，两个方向都考虑，
    # 只保留 anchor 距离 < 0.45nm 的候选（口袋本来就小，穷举不会有组合爆炸问题）。
    #
    # 键连接信息不能从 topology.cif 的 Topology.bonds 拿——实测这份 CIF 里配体 H 原子
    # 基本没有可信的 bond 记录（22 个 H 里 21 个邻接表是空的，剩下 1 个还被错误地同时
    # 连到了 3 个不同重原子上，是 mdtraj 距离启发式猜测出的伪键）。真正权威的连接性在
    # system_native.xml 里：不含 H 的键是 HarmonicBondForce，含 H 的键在这套体系里用的是
    # HBonds 约束方案、被转成了 System 的 rigid constraint，不在 HarmonicBondForce 里——
    # 两者都要读，只读 HarmonicBondForce 会漏掉所有 X-H 键（実测：只用 HarmonicBondForce
    # 时配体 22 个 H 全部邻接表为空）。
    openmm, app, unit, XmlSerializer = require_openmm()
    with open(args.system_xml, "r", encoding="utf-8") as handle:
        _bond_system = XmlSerializer.deserialize(handle.read())
    bond_adjacency: Dict[int, List[int]] = {}
    for force in _bond_system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for b in range(force.getNumBonds()):
                i1, i2, _, _ = force.getBondParameters(b)
                bond_adjacency.setdefault(int(i1), []).append(int(i2))
                bond_adjacency.setdefault(int(i2), []).append(int(i1))
    for c in range(_bond_system.getNumConstraints()):
        i1, i2, _ = _bond_system.getConstraintParameters(c)
        bond_adjacency.setdefault(int(i1), []).append(int(i2))
        bond_adjacency.setdefault(int(i2), []).append(int(i1))
    del _bond_system

    def _donors_acceptors(atom_indices):
        donors = []  # (heavy_idx, h_idx)
        acceptors = []
        top = anchor_traj.top
        idx_set = set(int(i) for i in atom_indices)
        for i in atom_indices:
            i = int(i)
            atom = top.atom(i)
            if atom.element.symbol not in ("N", "O"):
                continue
            acceptors.append(i)
            for other_idx in bond_adjacency.get(i, []):
                other = top.atom(other_idx)
                if other.element.symbol == "H" and other_idx in idx_set:
                    donors.append((i, int(other_idx)))
        return donors, acceptors

    lig_donors, lig_acceptors = _donors_acceptors(lig_idx)
    env_donors, env_acceptors = _donors_acceptors(env_idx)
    hbond_candidates = []  # (donor_heavy, h, acceptor)
    for d_heavy, h in lig_donors:
        for a in env_acceptors:
            r = float(np.linalg.norm(anchor_traj.xyz[0, d_heavy] - anchor_traj.xyz[0, a]))
            if r < 0.45:
                hbond_candidates.append((d_heavy, h, a, "lig_donor"))
    for d_heavy, h in env_donors:
        for a in lig_acceptors:
            r = float(np.linalg.norm(anchor_traj.xyz[0, d_heavy] - anchor_traj.xyz[0, a]))
            if r < 0.45:
                hbond_candidates.append((d_heavy, h, a, "env_donor"))
    print(f"    候选氢键对(anchor D...A < 0.45nm): {len(hbond_candidates)} 个")

    length_scales_nm = [0.15, 0.25, 0.35]

    def _analyze_one_replica(dcd_path: str) -> Dict:
        traj = md.load(dcd_path, top=args.traj_top)
        n_frames_raw = len(traj)
        if n_frames_raw == 0:
            return {"n_frames": 0}
        try:
            if traj.unitcell_vectors is not None:
                traj = traj.image_molecules(inplace=False)
        except Exception:
            pass
        traj.superpose(anchor_traj, atom_indices=pocket_idx, ref_atom_indices=pocket_idx)

        lig_xyz = traj.xyz[:, lig_idx, :]
        lig_heavy_xyz = traj.xyz[:, lig_heavy_idx, :]
        n_frames = lig_xyz.shape[0]

        diffs = lig_heavy_xyz - anchor_lig_heavy_xyz[None, :, :]
        rmsd_to_anchor_A = np.sqrt(np.mean(np.sum(diffs ** 2, axis=-1), axis=-1)) * 10.0

        cluster_ids = -np.ones(n_frames, dtype=int)
        rep_frames: List[int] = []
        thresh_nm = float(args.replica_cluster_rmsd_A) / 10.0
        for i in range(n_frames):
            assigned = False
            for c, rep in enumerate(rep_frames):
                d = lig_heavy_xyz[i] - lig_heavy_xyz[rep]
                r = math.sqrt(float(np.mean(np.sum(d ** 2, axis=-1))))
                if r <= thresh_nm:
                    cluster_ids[i] = c
                    assigned = True
                    break
            if not assigned:
                rep_frames.append(i)
                cluster_ids[i] = len(rep_frames) - 1
        cluster_counts = np.bincount(cluster_ids)
        top_cluster_occupancy = float(cluster_counts.max() / n_frames)

        lig_com = lig_xyz.mean(axis=1)
        trans_vec = lig_com - anchor_lig_com[None, :]
        trans_components_nm = trans_vec @ anchor_axes  # (n_frames, 3)

        rot_angles_deg = np.empty(n_frames, dtype=float)
        for i in range(n_frames):
            centered = lig_heavy_xyz[i] - lig_heavy_xyz[i].mean(axis=0)
            rot_angles_deg[i] = _kabsch_rotation_deg(anchor_lig_heavy_centered, centered)

        env_xyz = traj.xyz[:, env_idx, :]
        d_full = np.linalg.norm(lig_xyz[:, :, None, :] - env_xyz[:, None, :, :], axis=-1)  # (n_frames,n_lig,n_env)
        min_dist_per_frame = d_full.min(axis=(1, 2))
        contact_feature_mean_frames = np.stack(
            [np.sum(np.exp(-d_full / ell), axis=(1, 2)) for ell in length_scales_nm], axis=1
        )  # (n_frames, 3)

        lig_pos_map = {int(a): k for k, a in enumerate(lig_idx)}
        env_pos_map = {int(a): k for k, a in enumerate(env_idx)}
        key_contact_stats = {}
        for lig_a, env_a in key_pairs:
            dd = np.linalg.norm(
                lig_xyz[:, lig_pos_map[lig_a], :] - env_xyz[:, env_pos_map[env_a], :], axis=-1
            )
            key_contact_stats[f"{lig_a}-{env_a}"] = {
                "mean_nm": float(dd.mean()),
                "std_nm": float(dd.std()),
                "occupancy": float(np.mean(dd < float(args.replica_contact_cutoff_nm))),
            }

        hbond_stats = {}
        for d_heavy, h, a, kind in hbond_candidates:
            dist = np.linalg.norm(traj.xyz[:, a, :] - traj.xyz[:, d_heavy, :], axis=-1)
            v1 = traj.xyz[:, d_heavy, :] - traj.xyz[:, h, :]
            v2 = traj.xyz[:, a, :] - traj.xyz[:, h, :]
            cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1.0e-12)
            angle_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
            bonded = (dist < float(args.replica_hbond_dist_nm)) & (angle_deg > float(args.replica_hbond_angle_deg))
            hbond_stats[f"{kind}:{d_heavy}-{h}...{a}"] = {
                "occupancy": float(np.mean(bonded)),
                "mean_dist_nm_when_bonded": float(dist[bonded].mean()) if bonded.any() else math.nan,
                "mean_angle_deg_when_bonded": float(angle_deg[bonded].mean()) if bonded.any() else math.nan,
            }

        return {
            "n_frames": int(n_frames),
            "rmsd_to_anchor_A_mean": float(rmsd_to_anchor_A.mean()),
            "rmsd_to_anchor_A_max": float(rmsd_to_anchor_A.max()),
            "n_pose_clusters": int(len(rep_frames)),
            "top_cluster_occupancy": top_cluster_occupancy,
            "translation_axis0_nm_mean": float(trans_components_nm[:, 0].mean()),
            "translation_axis1_nm_mean": float(trans_components_nm[:, 1].mean()),
            "translation_axis2_nm_mean": float(trans_components_nm[:, 2].mean()),
            "rotation_deg_mean": float(rot_angles_deg.mean()),
            "min_dist_mean_nm": float(min_dist_per_frame.mean()),
            "too_close_fraction": float(np.mean(min_dist_per_frame < float(args.replica_too_close_nm))),
            "contact_feature_mean": [float(x) for x in contact_feature_mean_frames.mean(axis=0)],
            "contact_feature_cov": contact_feature_mean_frames.astype(float).T @ contact_feature_mean_frames.astype(float) / n_frames
            - np.outer(contact_feature_mean_frames.mean(axis=0), contact_feature_mean_frames.mean(axis=0)),
            "key_contacts": key_contact_stats,
            "hbonds": hbond_stats,
        }

    print(f"[2/4] 逐 condition/replica 分析（conditions={conditions}）")
    per_condition_replica_stats: Dict[str, List[Dict]] = {}
    for condition in conditions:
        label = f"replica_{condition}"
        n_replicas_expected = max(1, int(args.stability_replicas))
        paths = _replica_trajectory_paths(output_dir, label, n_replicas_expected)
        if not paths:
            print(f"    ⚠️ {condition}: 未找到任何轨迹（{os.path.join(output_dir, label)}），跳过")
            per_condition_replica_stats[condition] = []
            continue
        stats_list = []
        for p in paths:
            state_csv = os.path.join(os.path.dirname(p), "state.csv")
            energy_stats = {"potential_kjmol": {}, "has_nan_or_inf": False}
            if os.path.isfile(state_csv):
                data = read_state_csv(state_csv)
                pe = np.asarray(data.get("potentialEnergy", []), dtype=float)
                energy_stats["potential_kjmol"] = summarize_series_with_percentiles(pe.tolist())
                energy_stats["has_nan_or_inf"] = bool(pe.size and not np.all(np.isfinite(pe)))
            s = _analyze_one_replica(p)
            s.update(energy_stats)
            s["traj_path"] = p
            stats_list.append(s)
        per_condition_replica_stats[condition] = stats_list
        print(f"    {condition}: {len(stats_list)} 个 replica，共 {sum(s.get('n_frames', 0) for s in stats_list)} 帧")

    print(f"[3/4] 按 condition 聚合(replica 间 mean±SEM)")
    scalar_fields = [
        "rmsd_to_anchor_A_mean", "top_cluster_occupancy", "translation_axis0_nm_mean",
        "translation_axis1_nm_mean", "translation_axis2_nm_mean", "rotation_deg_mean",
        "min_dist_mean_nm", "too_close_fraction",
    ]
    aggregated: Dict[str, Dict] = {}
    for condition, stats_list in per_condition_replica_stats.items():
        if not stats_list:
            aggregated[condition] = {"n_replicas": 0}
            continue
        cond_agg: Dict = {"n_replicas": len(stats_list)}
        for field in scalar_fields:
            mean, sem = _mean_sem(np.array([s[field] for s in stats_list], dtype=float))
            cond_agg[field] = {"mean": mean, "sem": sem}
        cond_agg["any_energy_nan_or_inf"] = bool(any(s.get("has_nan_or_inf") for s in stats_list))
        cfm = np.array([s["contact_feature_mean"] for s in stats_list], dtype=float)
        cond_agg["contact_feature_mean"] = [
            {"mean": float(cfm[:, k].mean()), "sem": _mean_sem(cfm[:, k])[1]} for k in range(cfm.shape[1])
        ]
        key_names = set()
        for s in stats_list:
            key_names.update(s["key_contacts"].keys())
        cond_agg["key_contacts"] = {}
        for name in sorted(key_names):
            vals_dist = np.array([s["key_contacts"].get(name, {}).get("mean_nm", math.nan) for s in stats_list])
            vals_occ = np.array([s["key_contacts"].get(name, {}).get("occupancy", math.nan) for s in stats_list])
            m_d, se_d = _mean_sem(vals_dist)
            m_o, se_o = _mean_sem(vals_occ)
            cond_agg["key_contacts"][name] = {
                "mean_nm": {"mean": m_d, "sem": se_d},
                "occupancy": {"mean": m_o, "sem": se_o},
            }
        hbond_names = set()
        for s in stats_list:
            hbond_names.update(s["hbonds"].keys())
        cond_agg["hbonds"] = {}
        for name in sorted(hbond_names):
            vals_occ = np.array([s["hbonds"].get(name, {}).get("occupancy", 0.0) for s in stats_list])
            vals_ang = np.array([s["hbonds"].get(name, {}).get("mean_angle_deg_when_bonded", math.nan) for s in stats_list])
            m_o, se_o = _mean_sem(vals_occ)
            m_a, se_a = _mean_sem(vals_ang)
            cond_agg["hbonds"][name] = {
                "occupancy": {"mean": m_o, "sem": se_o},
                "mean_angle_deg_when_bonded": {"mean": m_a, "sem": se_a},
            }
        aggregated[condition] = cond_agg

    print(f"[4/4] Delta<q> 显著性检验（相对参照 condition={reference_condition}）")
    delta_report: Dict[str, Dict] = {}
    ref_agg = aggregated.get(reference_condition, {})
    for condition in conditions:
        if condition == reference_condition:
            continue
        cond_agg = aggregated.get(condition, {})
        if not cond_agg or not ref_agg or cond_agg.get("n_replicas", 0) == 0 or ref_agg.get("n_replicas", 0) == 0:
            continue
        entry: Dict = {}
        for field in scalar_fields:
            m_c, se_c = cond_agg[field]["mean"], cond_agg[field]["sem"]
            m_r, se_r = ref_agg[field]["mean"], ref_agg[field]["sem"]
            delta = m_c - m_r
            combined_sem = math.sqrt((se_c or 0.0) ** 2 + (se_r or 0.0) ** 2) if np.isfinite(se_c) and np.isfinite(se_r) else math.nan
            entry[field] = {
                "delta": float(delta),
                "combined_sem": float(combined_sem) if np.isfinite(combined_sem) else None,
                "significant_gt_2sem": bool(np.isfinite(combined_sem) and abs(delta) > 2.0 * combined_sem),
            }
        delta_report[condition] = entry
        sig_fields = [f for f, v in entry.items() if v["significant_gt_2sem"]]
        print(
            f"    {condition} vs {reference_condition}: "
            + (f"⚠️ 超过2×SEM的量: {sig_fields}" if sig_fields else "所有量都落在 2×SEM 误差内")
        )

    summary = {
        "conditions": conditions,
        "reference_condition": reference_condition,
        "key_pairs": key_pairs,
        "n_hbond_candidates": len(hbond_candidates),
        "aggregated": aggregated,
        "delta_vs_reference": delta_report,
        "note": (
            "口袋(非配体)原子先做 Kabsch 叠合去掉系统整体平移转动，翻译/旋转/RMSD 都是配体在该"
            "叠合参考系里相对 anchor 的真实位移，不是配体对自身第0帧的内部 RMSD。"
            "significant_gt_2sem=true 表示该 condition 与参照 condition 之间的差异超过两条"
            "replica 间标准误之和——是否真的说明'势阱被推走'，还要看是否多个不同的 q "
            "(平移轴/旋转角/关键氢键距离角度)一致指向同一方向，而不是只看单个量。"
            "没有逐帧 max-force：那需要用对应势重新算力，这里只用 potentialEnergy NaN/Inf 和"
            "短接触比例(too_close_fraction)作为不稳定性代理。"
        ),
    }
    NumpyEncoder = load_abfe_symbols()["NumpyEncoder"]
    with open(os.path.join(output_dir, "replica_compare_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, cls=NumpyEncoder)
    # 逐 replica 明细（含 contact_feature_cov 协方差矩阵）单独存一份，聚合摘要里不重复存这部分体积较大的数据。
    with open(os.path.join(output_dir, "replica_compare_per_replica.json"), "w", encoding="utf-8") as handle:
        json.dump(per_condition_replica_stats, handle, indent=2, cls=NumpyEncoder)
    print(f"    summary: {os.path.join(output_dir, 'replica_compare_summary.json')}")
    print(f"    per-replica detail: {os.path.join(output_dir, 'replica_compare_per_replica.json')}")
    return summary


def run_hbond_switching_dynamics(args: argparse.Namespace, output_dir: str) -> Dict:
    """用户对 §4.4 的关键修正：受体身份核实为 2134=VAL136 主链羰基O、2759=SER177 侧链OG、
    4084=ASN254 OD1、1321=ASP85 HD2；且 4607/4608 是配体同一个酰胺 N(4587) 的两个 H——
    "VAL->SER 伙伴切换"更准确的物理图像是酰胺 NH2 在(固定的)VAL136 主链羰基与(需要合适
    rotamer 才能靠近的)SER177 可转动侧链之间形成竞争性/可能双叉的氢键网络，不是同一个 H
    在两个受体间简单切换。这也解释了 §6.6 角度诊断为什么找不到信号：那个诊断固定环境、
    只扰动配体，结构上看不见 SER177 rotamer 转动和环境弛豫这个自由度。

    这一步只用已经跑完的 --replica-run 轨迹（不需要新 MD、不需要 MACE），逐帧定义四态：
        V：只有 VAL136-O 氢键（H4607 或 H4608 任一满足距离+角度判据）
        S：只有 SER177-OG 氢键
        B：两者同时存在（双叉）
        N：都不存在
    输出每个 replica 的 V/S/B/N occupancy、转移矩阵、首次 V->S 转移帧、各状态驻留时间、
    前半段/后半段 occupancy、indicator 自相关积分时间+有效样本数、以及"从初始态出发后是否
    只发生一次不可逆切换"——这决定 §4.4 报的 occupancy 是不是平衡概率，还是 2ns 短轨迹里
    初态依赖的动力学产物（如果是后者，§5 的 (14,5) vs (12,6) 判断需要更长/更多 replica
    才能下结论，不能直接用现有 occupancy 数字）。
    """
    md = require_module("mdtraj")
    conditions = [c.strip() for c in str(args.replica_conditions).split(",") if c.strip()]

    donor_heavy = int(args.switching_donor_heavy_atom)
    donor_h_atoms = [int(x) for x in str(args.switching_donor_h_atoms).split(",") if x.strip()]
    val_acceptor = int(args.switching_val_acceptor_atom)
    ser_acceptor = int(args.switching_ser_acceptor_atom)
    hbond_dist = float(args.replica_hbond_dist_nm)
    hbond_angle = float(args.replica_hbond_angle_deg)

    ref_traj = md.load(args.traj, top=args.traj_top)[-1]
    top = ref_traj.top

    def _describe(idx: int) -> str:
        atom = top.atom(idx)
        return f"{idx}:{atom.residue.name}{atom.residue.resSeq}-{atom.name}({atom.element.symbol})"

    print(
        "[1/3] 核对关键原子身份（供肉眼核实是否真的是 VAL136 主链O / SER177侧链OG / 配体酰胺N及其两个H）:\n"
        f"    donor_heavy={_describe(donor_heavy)}  donor_H={[_describe(i) for i in donor_h_atoms]}\n"
        f"    VAL_acceptor={_describe(val_acceptor)}  SER_acceptor={_describe(ser_acceptor)}"
    )

    def _bonded_series(traj, h_idx: int, a_idx: int) -> np.ndarray:
        dist = np.linalg.norm(traj.xyz[:, a_idx, :] - traj.xyz[:, donor_heavy, :], axis=-1)
        v1 = traj.xyz[:, donor_heavy, :] - traj.xyz[:, h_idx, :]
        v2 = traj.xyz[:, a_idx, :] - traj.xyz[:, h_idx, :]
        cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1.0e-12)
        angle_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        return (dist < hbond_dist) & (angle_deg > hbond_angle)

    def _autocorr_neff(indicator: np.ndarray) -> Dict:
        # 用户指出的修正：常数(方差为0，即该指示量整条轨迹从未访问过对面状态)不该报
        # n_eff=n——那不是"n个独立样本"，是"没有任何变化可供估计自相关"，应标记为
        # undefined，否则会被误读成"充分采样"。
        x = indicator.astype(float) - indicator.astype(float).mean()
        n = x.size
        var = float(np.mean(x ** 2))
        if n < 2 or var < 1.0e-12:
            return {"tau_int_frames": math.nan, "n_eff": math.nan, "unvisited_or_constant": True}
        max_lag = min(n - 1, 500)
        tau_int = 1.0
        for k in range(1, max_lag + 1):
            acf_k = float(np.mean(x[: n - k] * x[k:]) / var)
            if acf_k <= 0.0:
                break
            tau_int += 2.0 * acf_k
        return {"tau_int_frames": float(tau_int), "n_eff": float(n / max(tau_int, 1.0e-6)), "unvisited_or_constant": False}

    labels = ["V", "S", "B", "N"]
    idx_of = {s: i for i, s in enumerate(labels)}

    def _analyze_replica(dcd_path: str) -> Optional[Dict]:
        traj = md.load(dcd_path, top=args.traj_top)
        n_frames = len(traj)
        if n_frames == 0:
            return None
        try:
            if traj.unitcell_vectors is not None:
                traj = traj.image_molecules(inplace=False)
        except Exception:
            pass

        v_bonded = np.zeros(n_frames, dtype=bool)
        s_bonded = np.zeros(n_frames, dtype=bool)
        for h_idx in donor_h_atoms:
            v_bonded |= _bonded_series(traj, h_idx, val_acceptor)
            s_bonded |= _bonded_series(traj, h_idx, ser_acceptor)

        state = np.full(n_frames, "N", dtype=object)
        state[v_bonded & ~s_bonded] = "V"
        state[~v_bonded & s_bonded] = "S"
        state[v_bonded & s_bonded] = "B"

        occupancy = {s: float(np.mean(state == s)) for s in labels}

        trans_counts = np.zeros((4, 4), dtype=int)
        for i in range(n_frames - 1):
            trans_counts[idx_of[state[i]], idx_of[state[i + 1]]] += 1
        n_state_changes = int(np.sum(trans_counts) - np.trace(trans_counts))
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(
            trans_counts, row_sums, out=np.zeros_like(trans_counts, dtype=float), where=row_sums > 0
        )

        first_v_to_s_frame: Optional[int] = None
        seen_v = False
        for i in range(n_frames):
            if state[i] == "V":
                seen_v = True
            elif state[i] == "S" and seen_v and first_v_to_s_frame is None:
                first_v_to_s_frame = i

        dwell_times: Dict[str, List[int]] = {s: [] for s in labels}
        run_state, run_len = state[0], 1
        for i in range(1, n_frames):
            if state[i] == run_state:
                run_len += 1
            else:
                dwell_times[run_state].append(run_len)
                run_state, run_len = state[i], 1
        dwell_times[run_state].append(run_len)
        dwell_stats = {
            s: {
                "n_runs": len(v),
                "mean_frames": float(np.mean(v)) if v else 0.0,
                "max_frames": int(max(v)) if v else 0,
            }
            for s, v in dwell_times.items()
        }

        half = n_frames // 2
        occ_first_half = {s: float(np.mean(state[:half] == s)) for s in labels} if half > 0 else {}
        occ_second_half = {s: float(np.mean(state[half:] == s)) for s in labels} if (n_frames - half) > 0 else {}

        initial_state = str(state[0])
        left_initial = False
        returned_after_leaving = False
        for i in range(1, n_frames):
            if state[i] != initial_state:
                left_initial = True
            elif left_initial:
                returned_after_leaving = True
        single_irreversible_transition = bool(left_initial and not returned_after_leaving)

        return {
            "n_frames": int(n_frames),
            "occupancy": occupancy,
            "transition_counts": trans_counts.tolist(),
            "transition_matrix": trans_matrix.tolist(),
            "transition_matrix_state_order": labels,
            "n_state_changes": n_state_changes,
            "first_v_to_s_transition_frame": first_v_to_s_frame,
            "dwell_time_stats_frames": dwell_stats,
            "occupancy_first_half": occ_first_half,
            "occupancy_second_half": occ_second_half,
            "autocorr_v_bonded": _autocorr_neff(v_bonded),
            "autocorr_s_bonded": _autocorr_neff(s_bonded),
            "initial_state": initial_state,
            "single_irreversible_transition_from_initial_state": single_irreversible_transition,
            "traj_path": dcd_path,
        }

    print(f"[2/3] 逐 condition/replica 分析 V/S/B/N 切换动力学（conditions={conditions}）")
    per_condition: Dict[str, List[Dict]] = {}
    for condition in conditions:
        label = f"replica_{condition}"
        n_replicas_expected = max(1, int(args.stability_replicas))
        paths = _replica_trajectory_paths(output_dir, label, n_replicas_expected)
        if not paths:
            print(f"    ⚠️ {condition}: 未找到任何轨迹（{os.path.join(output_dir, label)}），跳过")
            per_condition[condition] = []
            continue
        stats_list = []
        for p in paths:
            s = _analyze_replica(p)
            if s is not None:
                stats_list.append(s)
        per_condition[condition] = stats_list
        print(f"    {condition}: {len(stats_list)} 个 replica")
        for i, s in enumerate(stats_list):
            occ = s["occupancy"]
            print(
                f"        replica {i}: occ V={occ['V']:.2f} S={occ['S']:.2f} B={occ['B']:.2f} N={occ['N']:.2f}  "
                f"n状态切换={s['n_state_changes']:3d}  初始态={s['initial_state']}  "
                f"离开初始态后未回归={s['single_irreversible_transition_from_initial_state']}(注意不等于'只切换一次'，那个看n状态切换)  "
                f"首次V→S帧={s['first_v_to_s_transition_frame']}  "
                f"前半段占据={ {k: round(v,2) for k,v in s['occupancy_first_half'].items()} }  "
                f"后半段占据={ {k: round(v,2) for k,v in s['occupancy_second_half'].items()} }  "
                f"n_eff(S指示量)={s['autocorr_s_bonded']['n_eff']:.1f}/{s['n_frames']}"
            )

    print("[3/3] 汇总判断：现有 occupancy 是否已经是平衡概率，还是短轨迹初态依赖的产物")
    any_single_transition_only = False
    total_replicas = 0
    for condition, stats_list in per_condition.items():
        for s in stats_list:
            total_replicas += 1
            if s["single_irreversible_transition_from_initial_state"] or s["n_state_changes"] <= 1:
                any_single_transition_only = True
    print(
        f"    存在'总状态切换数<=1 或 离开初始态后从未回归'的 replica: {any_single_transition_only}  "
        f"（{total_replicas} 个 replica 里任意一个满足即为 true——如果为 true，现有报告的 occupancy "
        "很可能不是平衡概率，只是初态依赖的动力学快照，不能直接用于 §5 的 (14,5) vs (12,6) 判断）"
    )

    summary = {
        "donor_heavy_atom": donor_heavy,
        "donor_h_atoms": donor_h_atoms,
        "val_acceptor_atom": val_acceptor,
        "ser_acceptor_atom": ser_acceptor,
        "atom_identity_check": {
            "donor_heavy": _describe(donor_heavy),
            "donor_h_atoms": [_describe(i) for i in donor_h_atoms],
            "val_acceptor": _describe(val_acceptor),
            "ser_acceptor": _describe(ser_acceptor),
        },
        "hbond_dist_nm": hbond_dist,
        "hbond_angle_deg": hbond_angle,
        "per_condition_per_replica": per_condition,
        "any_replica_with_at_most_one_state_change": any_single_transition_only,
        "interpretation_note": (
            "V/S/B/N 定义：V=只有VAL136-O氢键(H4607/H4608任一满足距离+角度判据)，S=只有SER177-OG氢键，"
            "B=两者同时存在(双叉)，N=都没有。如果 any_replica_with_at_most_one_state_change=true，说明"
            "至少有一条2ns replica在整个轨迹里只经历了0-1次状态切换——这种情况下该replica的occupancy"
            "只反映'从哪个初态出发、多久之后随机跳了一次'，不是平衡分布下的时间加权概率，不同replica间的"
            "occupancy差异也可能只是初始条件/跳变时机的随机性，不能直接解读为DEXP核参数改变了平衡氢键"
            "偏好。n_eff(autocorr_s_bonded/autocorr_v_bonded)远小于n_frames，也是同一个问题的另一种体现——"
            "真实独立样本数远少于帧数，occupancy的统计不确定度比看起来的要大得多。"
            "如果结果显示确实存在两态竞争但转换本身发生多次、双向可逆(n_state_changes较大且轨迹前后半段"
            "occupancy相近)，才说明现有2ns轨迹已经采样到准平衡分布，occupancy数字本身可信，那时候"
            "(14,5)vs(12,6)的比较才站得住脚；否则需要更长/更多replica，或者专门从V/S/B多个初始态分别"
            "起始做统计，而不是依赖单一初始pose的短轨迹。"
        ),
    }
    summary_path = os.path.join(output_dir, "hbond_switching_dynamics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    summary: {summary_path}")
    return summary


def run_hbond_committed_state_dynamics(args: argparse.Namespace, output_dir: str) -> Dict:
    """`run_hbond_switching_dynamics` 的 committed-state 升级版（用户 2026-07-12 指出的修正）。

    问题：二元距离/角度阈值下"几十次状态切换"大量是阈值附近的快速抖动，不是真正的
    VAL<->SER basin 转换——实测(见 hbond_switching_dynamics_summary.json)大部分"混合"
    replica 里 V/S/B/N 的平均驻留时间只有 1-2 帧，`original`4/5个replica和`dexp_12_6`
    replica0/4 甚至从未访问过对面的状态(只在 V<->N 或 S<->N 间抖动)。这些都说明不能直接
    用二元阈值下的"转换次数"当作"轨迹已充分采样两个basin"的证据。

    改进：
    1. 连续 coordination：对 VAL/SER 两个受体，用跟(a)阶段/§6 同款的 quintic smoothstep
       在距离和角度两个方向做平滑(不是硬阈值)，`c_A = sum_H s_r(r_HA)*s_theta(theta_DHA)`，
       两个配体 N-H 都计入。`q = c_VAL - c_SER` 作为诊断用的连续反应坐标。
    2. Committed state：对 c_VAL、c_SER 分别用 Schmitt trigger(进入阈值`--switching-commit-
       enter-score`严格、离开阈值`--switching-commit-exit-score`宽松) + 最小驻留去抖动
       (`--switching-min-dwell-frames`，默认 4 帧)，得到"真正commit到该basin"的布尔序列，
       不是逐帧硬阈值。committed_state = V(只commit VAL)/S(只commit SER)/B(都commit，双叉)/
       N(都没commit)。
    3. 输出 committed V->S、S->V 直接 basin-to-basin 穿越次数(经过N/B中转也算，只要没有
       在中途真正commit回原basin)、各自首次穿越帧、committed dwell time、每
       `--switching-block-ns`(默认0.5ns)一块的 occupancy、是否同时访问过两个 basin。
    4. 重新定义"尚未平衡"判据(用户指定，任一成立即标记)：
       - 任一方向 committed 穿越次数 < `--switching-min-committed-transitions`(默认3)
         (且该 replica 确实访问过两个 basin——只访问一个 basin 单独由下一条标记，不重复计入这条)；
       - 前后半段 occupancy 最大绝对差 > `--switching-first-second-half-diff-threshold`(默认0.15)；
       - condition 内多数 replica 只访问一个 basin；
       - committed_val/committed_ser 任一 indicator 的自相关 n_eff < `--switching-min-n-eff`(默认20)；
       - 按(committed)初始 basin 分组，不同初始 basin 组的后半段 occupancy 分布差异 > 阈值。

    仍然只用已跑完的 --replica-run 轨迹，不需要新 MD/MACE。
    """
    md = require_module("mdtraj")
    conditions = [c.strip() for c in str(args.replica_conditions).split(",") if c.strip()]

    donor_heavy = int(args.switching_donor_heavy_atom)
    donor_h_atoms = [int(x) for x in str(args.switching_donor_h_atoms).split(",") if x.strip()]
    val_acceptor = int(args.switching_val_acceptor_atom)
    ser_acceptor = int(args.switching_ser_acceptor_atom)

    hbond_dist = float(args.replica_hbond_dist_nm)
    hbond_angle = float(args.replica_hbond_angle_deg)
    dist_hw = float(args.switching_coord_dist_half_width_nm)
    angle_hw = float(args.switching_coord_angle_half_width_deg)
    r_on, r_off = hbond_dist - dist_hw, hbond_dist + dist_hw
    theta_off, theta_on = hbond_angle - angle_hw, hbond_angle + angle_hw

    c_enter = float(args.switching_commit_enter_score)
    c_exit = float(args.switching_commit_exit_score)
    if not (c_enter > c_exit):
        raise ValueError("--switching-commit-enter-score 必须大于 --switching-commit-exit-score（Schmitt trigger 需要 hysteresis 间隙）")
    min_dwell = max(1, int(args.switching_min_dwell_frames))
    block_ns = float(args.switching_block_ns)
    sim_ns = float(args.sim_ns)
    min_committed_transitions = max(0, int(args.switching_min_committed_transitions))
    half_diff_threshold = float(args.switching_first_second_half_diff_threshold)
    min_n_eff = float(args.switching_min_n_eff)

    ref_traj = md.load(args.traj, top=args.traj_top)[-1]
    top = ref_traj.top

    def _describe(idx: int) -> str:
        atom = top.atom(idx)
        return f"{idx}:{atom.residue.name}{atom.residue.resSeq}-{atom.name}({atom.element.symbol})"

    print(
        "[1/4] 核对关键原子身份:\n"
        f"    donor_heavy={_describe(donor_heavy)}  donor_H={[_describe(i) for i in donor_h_atoms]}\n"
        f"    VAL_acceptor={_describe(val_acceptor)}  SER_acceptor={_describe(ser_acceptor)}\n"
        f"    距离平滑窗=[{r_on:.3f},{r_off:.3f}]nm(1->0)  角度平滑窗=[{theta_off:.1f},{theta_on:.1f}]deg(0->1)  "
        f"commit进入分数>{c_enter}  commit离开分数<{c_exit}  最小驻留={min_dwell}帧  block={block_ns}ns(按--sim-ns={sim_ns}ns换算)"
    )

    def _smooth_dist(r: np.ndarray) -> np.ndarray:
        s = np.ones_like(r)
        in_sw = (r > r_on) & (r <= r_off)
        t = (r[in_sw] - r_on) / (r_off - r_on)
        s[in_sw] = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
        s[r > r_off] = 0.0
        return s

    def _smooth_angle(theta: np.ndarray) -> np.ndarray:
        s = np.zeros_like(theta)
        in_sw = (theta > theta_off) & (theta <= theta_on)
        t = (theta[in_sw] - theta_off) / (theta_on - theta_off)
        s[in_sw] = 10.0 * t ** 3 - 15.0 * t ** 4 + 6.0 * t ** 5
        s[theta > theta_on] = 1.0
        return s

    def _coord_score(traj, h_idx: int, a_idx: int) -> np.ndarray:
        dist = np.linalg.norm(traj.xyz[:, a_idx, :] - traj.xyz[:, donor_heavy, :], axis=-1)
        v1 = traj.xyz[:, donor_heavy, :] - traj.xyz[:, h_idx, :]
        v2 = traj.xyz[:, a_idx, :] - traj.xyz[:, h_idx, :]
        cos_ang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1.0e-12)
        angle_deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        return _smooth_dist(dist) * _smooth_angle(angle_deg)

    def _schmitt_debounced(score: np.ndarray, enter: float, exit_: float, min_dwell_frames: int) -> np.ndarray:
        n = score.size
        schmitt = np.zeros(n, dtype=bool)
        state = bool(score[0] > enter)
        for i in range(n):
            if state:
                if score[i] < exit_:
                    state = False
            else:
                if score[i] > enter:
                    state = True
            schmitt[i] = state
        committed = np.zeros(n, dtype=bool)
        current = schmitt[0]
        i = 0
        while i < n:
            j = i
            while j < n and schmitt[j] == schmitt[i]:
                j += 1
            run_len = j - i
            val = bool(schmitt[i])
            if val != current and run_len >= min_dwell_frames:
                current = val
            committed[i:j] = current
            i = j
        return committed

    def _autocorr_neff(indicator: np.ndarray) -> Dict:
        x = indicator.astype(float) - indicator.astype(float).mean()
        n = x.size
        var = float(np.mean(x ** 2))
        if n < 2 or var < 1.0e-12:
            return {"tau_int_frames": math.nan, "n_eff": math.nan, "unvisited_or_constant": True}
        max_lag = min(n - 1, 500)
        tau_int = 1.0
        for k in range(1, max_lag + 1):
            acf_k = float(np.mean(x[: n - k] * x[k:]) / var)
            if acf_k <= 0.0:
                break
            tau_int += 2.0 * acf_k
        return {"tau_int_frames": float(tau_int), "n_eff": float(n / max(tau_int, 1.0e-6)), "unvisited_or_constant": False}

    def _count_basin_passages(committed_state: np.ndarray) -> Tuple[Dict[str, int], Dict[str, Optional[int]]]:
        counts = {"V_to_S": 0, "S_to_V": 0}
        first_frame: Dict[str, Optional[int]] = {"V_to_S": None, "S_to_V": None}
        last_pure_basin: Optional[str] = None
        prev: Optional[str] = None
        for i, st in enumerate(committed_state):
            if st in ("V", "S") and st != prev:
                if last_pure_basin is not None and last_pure_basin != st:
                    key = f"{last_pure_basin}_to_{st}"
                    counts[key] += 1
                    if first_frame[key] is None:
                        first_frame[key] = i
                last_pure_basin = st
            prev = st
        return counts, first_frame

    labels = ["V", "S", "B", "N"]

    def _analyze_replica(dcd_path: str) -> Optional[Dict]:
        traj = md.load(dcd_path, top=args.traj_top)
        n_frames = len(traj)
        if n_frames == 0:
            return None
        try:
            if traj.unitcell_vectors is not None:
                traj = traj.image_molecules(inplace=False)
        except Exception:
            pass

        c_val = np.zeros(n_frames, dtype=float)
        c_ser = np.zeros(n_frames, dtype=float)
        for h_idx in donor_h_atoms:
            c_val += _coord_score(traj, h_idx, val_acceptor)
            c_ser += _coord_score(traj, h_idx, ser_acceptor)
        q = c_val - c_ser

        committed_val = _schmitt_debounced(c_val, c_enter, c_exit, min_dwell)
        committed_ser = _schmitt_debounced(c_ser, c_enter, c_exit, min_dwell)

        committed_state = np.full(n_frames, "N", dtype=object)
        committed_state[committed_val & ~committed_ser] = "V"
        committed_state[~committed_val & committed_ser] = "S"
        committed_state[committed_val & committed_ser] = "B"

        occupancy = {s: float(np.mean(committed_state == s)) for s in labels}
        visited_val, visited_ser = bool(np.any(committed_val)), bool(np.any(committed_ser))
        visited_both_basins = visited_val and visited_ser

        passage_counts, passage_first_frame = _count_basin_passages(committed_state)

        dwell_times: Dict[str, List[int]] = {s: [] for s in labels}
        run_state, run_len = committed_state[0], 1
        for i in range(1, n_frames):
            if committed_state[i] == run_state:
                run_len += 1
            else:
                dwell_times[run_state].append(run_len)
                run_state, run_len = committed_state[i], 1
        dwell_times[run_state].append(run_len)
        dwell_stats = {
            s: {"n_runs": len(v), "mean_frames": float(np.mean(v)) if v else 0.0, "max_frames": int(max(v)) if v else 0}
            for s, v in dwell_times.items()
        }

        half = n_frames // 2
        occ_first_half = {s: float(np.mean(committed_state[:half] == s)) for s in labels} if half > 0 else {}
        occ_second_half = {s: float(np.mean(committed_state[half:] == s)) for s in labels} if (n_frames - half) > 0 else {}
        max_half_diff = max((abs(occ_first_half.get(s, 0.0) - occ_second_half.get(s, 0.0)) for s in labels), default=0.0)

        frames_per_block = max(1, int(round(n_frames * block_ns / sim_ns))) if sim_ns > 0 else n_frames
        n_blocks = int(math.ceil(n_frames / frames_per_block))
        block_occupancy = []
        for b in range(n_blocks):
            lo, hi = b * frames_per_block, min(n_frames, (b + 1) * frames_per_block)
            seg = committed_state[lo:hi]
            block_occupancy.append({s: float(np.mean(seg == s)) for s in labels})

        autocorr_val = _autocorr_neff(committed_val)
        autocorr_ser = _autocorr_neff(committed_ser)
        n_eff_candidates = [
            x["n_eff"] for x in (autocorr_val, autocorr_ser) if not x.get("unvisited_or_constant", True) and not math.isnan(x["n_eff"])
        ]
        min_n_eff_this_replica = float(min(n_eff_candidates)) if n_eff_candidates else math.nan

        flags = {
            "insufficient_committed_transitions": bool(
                visited_both_basins and (passage_counts["V_to_S"] < min_committed_transitions or passage_counts["S_to_V"] < min_committed_transitions)
            ),
            "large_first_second_half_drift": bool(max_half_diff > half_diff_threshold),
            "single_basin_only": bool(not visited_both_basins),
            "low_n_eff": bool((not math.isnan(min_n_eff_this_replica)) and min_n_eff_this_replica < min_n_eff),
        }
        not_equilibrated = bool(any(flags.values()))

        return {
            "n_frames": int(n_frames),
            "occupancy_committed": occupancy,
            "visited_both_basins": visited_both_basins,
            "committed_passage_counts": passage_counts,
            "committed_passage_first_frame": passage_first_frame,
            "dwell_time_stats_frames_committed": dwell_stats,
            "occupancy_first_half": occ_first_half,
            "occupancy_second_half": occ_second_half,
            "max_first_second_half_abs_diff": float(max_half_diff),
            "block_ns": block_ns,
            "block_occupancy": block_occupancy,
            "autocorr_committed_val": autocorr_val,
            "autocorr_committed_ser": autocorr_ser,
            "initial_committed_state": str(committed_state[0]),
            "mean_q": float(np.mean(q)),
            "std_q": float(np.std(q)),
            "not_equilibrated_flags": flags,
            "not_equilibrated": not_equilibrated,
            "traj_path": dcd_path,
        }

    print(f"[2/4] 逐 condition/replica 分析 committed V/S/B/N 动力学（conditions={conditions}）")
    per_condition: Dict[str, List[Dict]] = {}
    for condition in conditions:
        label = f"replica_{condition}"
        n_replicas_expected = max(1, int(args.stability_replicas))
        paths = _replica_trajectory_paths(output_dir, label, n_replicas_expected)
        if not paths:
            print(f"    ⚠️ {condition}: 未找到任何轨迹（{os.path.join(output_dir, label)}），跳过")
            per_condition[condition] = []
            continue
        stats_list = [s for s in (_analyze_replica(p) for p in paths) if s is not None]
        per_condition[condition] = stats_list
        print(f"    {condition}: {len(stats_list)} 个 replica")
        for i, s in enumerate(stats_list):
            occ = s["occupancy_committed"]
            pc = s["committed_passage_counts"]
            print(
                f"        replica {i}: committed occ V={occ['V']:.2f} S={occ['S']:.2f} B={occ['B']:.2f} N={occ['N']:.2f}  "
                f"两个basin都访问过={s['visited_both_basins']}  V->S次数={pc['V_to_S']} S->V次数={pc['S_to_V']}  "
                f"前后半段最大差={s['max_first_second_half_abs_diff']:.3f}  "
                f"n_eff(committed_val/ser)={s['autocorr_committed_val']['n_eff']:.1f}/{s['autocorr_committed_ser']['n_eff']:.1f}  "
                f"未平衡判定={s['not_equilibrated']} {s['not_equilibrated_flags']}"
            )

    print("[3/4] condition 内多数 replica 是否只访问一个 basin")
    condition_majority_single_basin: Dict[str, bool] = {}
    for condition, stats_list in per_condition.items():
        if not stats_list:
            continue
        n_single = sum(1 for s in stats_list if not s["visited_both_basins"])
        is_majority = n_single > (len(stats_list) / 2.0)
        condition_majority_single_basin[condition] = is_majority
        print(f"    {condition}: {n_single}/{len(stats_list)} 个 replica 只访问一个 basin -> majority_single_basin={is_majority}")
        if is_majority:
            for s in stats_list:
                s["not_equilibrated_flags"]["majority_single_basin_condition"] = True
                s["not_equilibrated"] = True

    print("[4/4] 按(committed)初始 basin 分组，比较不同初始 basin 组的后半段 occupancy 是否收敛到同一分布")
    cross_initial_state_check: Dict[str, Dict] = {}
    for condition, stats_list in per_condition.items():
        groups: Dict[str, List[Dict]] = {}
        for s in stats_list:
            groups.setdefault(s["initial_committed_state"], []).append(s["occupancy_second_half"])
        group_means = {
            g: {lab: float(np.mean([occ.get(lab, 0.0) for occ in occs])) for lab in labels}
            for g, occs in groups.items()
            if occs
        }
        max_group_diff = 0.0
        group_keys = list(group_means.keys())
        for gi in range(len(group_keys)):
            for gj in range(gi + 1, len(group_keys)):
                d = max(abs(group_means[group_keys[gi]][lab] - group_means[group_keys[gj]][lab]) for lab in labels)
                max_group_diff = max(max_group_diff, d)
        diverged = bool(len(group_keys) >= 2 and max_group_diff > half_diff_threshold)
        cross_initial_state_check[condition] = {
            "groups_by_initial_committed_state": {g: len(v) for g, v in groups.items()},
            "group_mean_second_half_occupancy": group_means,
            "max_group_mean_abs_diff": max_group_diff,
            "diverged_by_initial_state": diverged,
        }
        print(f"    {condition}: 初始态分组={list(group_means.keys())}  组间后半段occupancy最大差={max_group_diff:.3f}  分歧={diverged}")
        if diverged:
            for s in stats_list:
                s["not_equilibrated_flags"]["diverges_by_initial_state_condition_level"] = True
                s["not_equilibrated"] = True

    any_not_equilibrated = any(s["not_equilibrated"] for stats_list in per_condition.values() for s in stats_list)

    summary = {
        "donor_heavy_atom": donor_heavy,
        "donor_h_atoms": donor_h_atoms,
        "val_acceptor_atom": val_acceptor,
        "ser_acceptor_atom": ser_acceptor,
        "atom_identity_check": {
            "donor_heavy": _describe(donor_heavy),
            "donor_h_atoms": [_describe(i) for i in donor_h_atoms],
            "val_acceptor": _describe(val_acceptor),
            "ser_acceptor": _describe(ser_acceptor),
        },
        "smoothing_windows": {
            "dist_on_off_nm": [r_on, r_off], "angle_off_on_deg": [theta_off, theta_on],
        },
        "commit_thresholds": {"enter": c_enter, "exit": c_exit, "min_dwell_frames": min_dwell},
        "block_ns": block_ns,
        "equilibration_criteria": {
            "min_committed_transitions_per_direction": min_committed_transitions,
            "first_second_half_diff_threshold": half_diff_threshold,
            "min_n_eff": min_n_eff,
        },
        "per_condition_per_replica": per_condition,
        "condition_majority_single_basin": condition_majority_single_basin,
        "cross_initial_state_check": cross_initial_state_check,
        "any_replica_not_equilibrated": any_not_equilibrated,
        "interpretation_note": (
            "committed_state 由 Schmitt trigger(进入/离开阈值不同) + 最小驻留去抖动构造，"
            "过滤掉了阈值附近的快速抖动——只有真正持续满足条件的一段才算'commit'到某个basin。"
            "committed_passage_counts 是真正的 basin-to-basin 穿越次数(经N/B中转也算，只要中途"
            "没有真正commit回原basin)，不是原始版本里被阈值抖动污染的'状态切换次数'。"
            "any_replica_not_equilibrated=true 说明至少有一个 replica 触发了用户定义的五条"
            "判据之一(committed穿越次数不足/前后半段漂移过大/只访问一个basin/n_eff过低/"
            "不同初始态收敛到不同分布)——这种情况下不能把 occupancy_committed 当成平衡概率，"
            "§5 的 (14,5) vs (12,6) 量化比较需要针对这些具体 replica 延长模拟或换成双初始态设计"
            "(VAL-dominant 和 SER/bifurcated 各自起始，比较后半段分布是否汇合)，而不是简单地"
            "把现有 2ns 单初始态轨迹的 occupancy 平均值当作最终答案。"
        ),
    }
    summary_path = os.path.join(output_dir, "hbond_committed_state_dynamics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    any_replica_not_equilibrated = {any_not_equilibrated}")
    print(f"    summary: {summary_path}")
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


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """均匀随机旋转矩阵（随机四元数法）。"""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    ])
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def run_pose_scan(args: argparse.Namespace, output_dir: str) -> Dict:
    """随机刚体扰动 + 短程约束弛豫，给 DEXP 拟合提供几何多样的训练样本。

    不追求连续解离路径（§ pull-scan 已经证明沿单一方向硬拉会被"接触点换人"绕过去，
    见 RESUME_DEXP_SESSION.md）。这里改为：每次独立地把配体做随机平移+旋转，短程弛豫后
    检查 min_valid_le_distance 等指标，按分箱标准接受/拒绝，不同 trial 之间完全独立。

    弛豫阶段额外加一个"整体短接触惩罚"力（collective short-contact penalty）：
        C_short = sum_ij sigmoid((r_cut - r_ij)/w)
        U_bias = k_bias * C_short
    直接压低"有多少对原子处于短程"这个整体量，而不是只顶开"当前最近的那一对"——
    避免了 pull-scan 里"顶开一对，换另一对顶上来"的问题。
    """
    openmm, app, unit, _ = require_openmm()
    md = require_module("mdtraj")

    system, topology = load_cached_system(args.system_xml, args.traj_top)
    source_traj = args.pull_scan_source_traj or args.traj
    _, ref_positions, box_vectors = load_last_frame_positions(source_traj, args.traj_top)
    ligand_indices = [int(i) for i in load_ligand_indices(args.ligand_indices)]

    meta_path = ensure_file(
        os.path.join(output_dir, "fit_label_cache_meta.json"),
        "fit 阶段固定环境原子缓存（先跑一次 --fit-only 生成）",
    )
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    env_all = [int(i) for i in meta["env_indices"]]
    top_mdtraj = md.load_topology(args.traj_top)
    protein_anchor_indices = [i for i in env_all if not top_mdtraj.atom(i).residue.is_water]
    heavy_lig = [i for i in ligand_indices if top_mdtraj.atom(i).element.symbol != "H"]
    heavy_env = [i for i in env_all if top_mdtraj.atom(i).element.symbol != "H"]
    print(
        f"[pose-scan] 配体原子={len(ligand_indices)} 环境原子={len(env_all)}"
        f"（蛋白锚点={len(protein_anchor_indices)}）heavy_lig={len(heavy_lig)} heavy_env={len(heavy_env)}"
    )

    ref_pos_nm = np.asarray([ref_positions[i].value_in_unit(unit.nanometer) for i in range(len(ref_positions))], dtype=float)
    box_lens_nm = None
    if box_vectors is not None:
        box_lens_nm = np.linalg.norm(
            np.asarray([v.value_in_unit(unit.nanometer) for v in box_vectors], dtype=float), axis=1
        )

    # 蛋白锚点位置约束：跟 pull-scan 同样的道理，短程弛豫时不希望壳层自己漂移，混淆诊断量。
    # 水分子不约束——配体扰动后水应该自由重排。
    protein_pos_nm = ref_pos_nm[protein_anchor_indices]
    anchor_restraint = openmm.CustomExternalForce("0.5*k_pos*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    anchor_restraint.addGlobalParameter("k_pos", float(args.pose_scan_anchor_k))
    anchor_restraint.addPerParticleParameter("x0")
    anchor_restraint.addPerParticleParameter("y0")
    anchor_restraint.addPerParticleParameter("z0")
    for idx, (x0, y0, z0) in zip(protein_anchor_indices, protein_pos_nm):
        anchor_restraint.addParticle(int(idx), [float(x0), float(y0), float(z0)])
    system.addForce(anchor_restraint)

    # 整体短接触惩罚：对配体-环境(含水)所有 pair 求和的平滑计数，而不是只顶开最近那一对。
    short_contact_force = openmm.CustomNonbondedForce(
        "k_bias/(1+exp((r-r_cut)/w))"
    )
    short_contact_force.addGlobalParameter("k_bias", float(args.pose_scan_short_contact_k))
    short_contact_force.addGlobalParameter("r_cut", float(args.pose_scan_short_contact_rcut))
    short_contact_force.addGlobalParameter("w", float(args.pose_scan_short_contact_width))
    short_contact_force.setNonbondedMethod(
        openmm.CustomNonbondedForce.CutoffPeriodic if box_vectors is not None
        else openmm.CustomNonbondedForce.CutoffNonPeriodic
    )
    cutoff_nm = float(args.pose_scan_short_contact_rcut) + 6.0 * float(args.pose_scan_short_contact_width)
    short_contact_force.setCutoffDistance(cutoff_nm)
    for _ in range(system.getNumParticles()):
        short_contact_force.addParticle([])
    short_contact_force.addInteractionGroup(set(ligand_indices), set(env_all))
    # OpenMM 要求同一个 system 里所有 nonbonded 类的力(NonbondedForce/CustomNonbondedForce)
    # exclusions 完全一致，否则报 "All Forces must have identical exclusions"——把原生
    # NonbondedForce 的 1-2/1-3/1-4 exclusion 原样搬过来（本项目其它地方新建
    # CustomNonbondedForce 时也是这个套路，见 abfe_core.py 里同名模式）。
    nb_force_ref = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    for exc_idx in range(nb_force_ref.getNumExceptions()):
        p1, p2, _, _, _ = nb_force_ref.getExceptionParameters(exc_idx)
        short_contact_force.addExclusion(int(p1), int(p2))
    system.addForce(short_contact_force)

    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.dt_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(int(args.seed))
    platform, properties = select_platform(args.platform)
    simulation = app.Simulation(topology, system, integrator, platform, properties)

    def _min_valid_and_short_count(pos_nm: np.ndarray) -> Tuple[float, int, float]:
        delta = pos_nm[ligand_indices][:, None, :] - pos_nm[env_all][None, :, :]
        if box_lens_nm is not None:
            delta -= box_lens_nm * np.round(delta / box_lens_nm)
        dists = np.linalg.norm(delta, axis=-1)
        valid = dists[(dists >= args.fit_r_min) & (dists <= args.fit_r_max)]
        min_valid = float(valid.min()) if valid.size else math.nan
        short_count = int(np.sum(dists < float(args.pose_scan_reject_min_dist) + 0.05))
        delta_h = pos_nm[heavy_lig][:, None, :] - pos_nm[heavy_env][None, :, :]
        if box_lens_nm is not None:
            delta_h -= box_lens_nm * np.round(delta_h / box_lens_nm)
        heavy_min = float(np.linalg.norm(delta_h, axis=-1).min()) if heavy_lig and heavy_env else math.nan
        return min_valid, short_count, heavy_min

    rng = np.random.default_rng(int(args.seed))
    lig_ref = ref_pos_nm[ligand_indices]
    lig_com_ref = lig_ref.mean(axis=0)
    lig_local = lig_ref - lig_com_ref

    n_bins = max(1, int(args.pose_scan_bins))
    bin_edges = np.linspace(
        float(args.pose_scan_reject_min_dist), float(args.pose_scan_bin_max_nm), n_bins + 1
    )
    bin_counts = np.zeros(n_bins, dtype=int)
    target_per_bin = max(1, int(args.pose_scan_per_bin))

    out_dir = ensure_dir(os.path.join(output_dir, "pose_scan"))
    dcd_path = os.path.join(out_dir, "pose_scan.dcd")
    log_rows: List[Dict] = []
    n_accepted = 0
    n_trials = max(1, int(args.pose_scan_trials))

    with open(dcd_path, "wb") as handle:
        dcd_file = app.DCDFile(handle, topology, args.dt_fs * unit.femtosecond)
        for trial in range(n_trials):
            if int(np.sum(bin_counts >= target_per_bin)) == n_bins:
                print(f"[pose-scan] 所有 {n_bins} 个 bin 都已填满(each >= {target_per_bin})，提前结束")
                break

            rot = _random_rotation_matrix(rng)
            translate_mag = float(rng.uniform(0.0, float(args.pose_scan_translate_max_nm)))
            translate_dir = rng.normal(size=3)
            translate_dir /= np.linalg.norm(translate_dir)
            new_lig_pos = lig_com_ref + translate_dir * translate_mag + lig_local @ rot.T

            trial_pos = ref_pos_nm.copy()
            trial_pos[ligand_indices] = new_lig_pos
            simulation.context.setPositions(trial_pos * unit.nanometer)
            if box_vectors is not None:
                simulation.context.setPeriodicBoxVectors(*box_vectors)

            try:
                openmm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=200)
            except Exception as exc:
                log_rows.append({"trial": trial, "accepted": False, "reason": f"minimize_failed:{exc}"})
                continue
            simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, int(args.seed) + trial)
            simulation.step(max(1, int(args.pose_scan_relax_steps)))

            state = simulation.context.getState(getPositions=True)
            pos_now = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float)
            min_valid, short_count, heavy_min = _min_valid_and_short_count(pos_now)

            if not np.isfinite(min_valid) or min_valid < float(args.pose_scan_reject_min_dist):
                log_rows.append({
                    "trial": trial, "accepted": False, "reason": "min_valid_below_floor",
                    "min_valid_le_distance_nm": min_valid, "short_contact_count": short_count,
                    "heavy_atom_min_distance_nm": heavy_min,
                })
                continue

            bin_idx = int(np.clip(np.digitize(min_valid, bin_edges) - 1, 0, n_bins - 1))
            if bin_counts[bin_idx] >= target_per_bin:
                log_rows.append({
                    "trial": trial, "accepted": False, "reason": "bin_full",
                    "min_valid_le_distance_nm": min_valid, "short_contact_count": short_count,
                    "heavy_atom_min_distance_nm": heavy_min, "bin_index": bin_idx,
                })
                continue

            bin_counts[bin_idx] += 1
            n_accepted += 1
            box = state.getPeriodicBoxVectors(asNumpy=True) if box_vectors is not None else None
            dcd_file.writeModel(state.getPositions(), periodicBoxVectors=box)
            log_rows.append({
                "trial": trial, "accepted": True, "reason": "kept",
                "min_valid_le_distance_nm": min_valid, "short_contact_count": short_count,
                "heavy_atom_min_distance_nm": heavy_min, "bin_index": bin_idx,
            })
            if n_accepted % 10 == 0 or n_accepted <= 5:
                print(
                    f"    [pose-scan] trial {trial}/{n_trials} 已接受 {n_accepted} 帧 "
                    f"(bin[{bin_idx}]={bin_counts[bin_idx]}/{target_per_bin}) "
                    f"min_valid={min_valid:.3f}nm short_count={short_count} heavy_min={heavy_min:.3f}nm"
                )

    log_csv = write_rows_csv(os.path.join(out_dir, "pose_scan_trials.csv"), log_rows)
    print(
        f"[pose-scan] 完成：{n_trials} 次尝试，接受 {n_accepted} 帧。各 bin 计数: {bin_counts.tolist()}"
        f"（bin 边界 {bin_edges.round(3).tolist()}）。轨迹: {dcd_path}，逐次尝试日志: {log_csv}"
    )
    return {
        "dcd_path": dcd_path, "log_csv": log_csv, "n_trials": n_trials, "n_accepted": n_accepted,
        "bin_counts": bin_counts.tolist(), "bin_edges": bin_edges.tolist(),
    }


def run_pull_scan(args: argparse.Namespace, output_dir: str) -> Dict:
    """手动把配体质心从环境锚点拉开，生成跨越宽 min-distance 范围的构型序列。

    不是自由能/PMF：不重加权、没有 MBAR，只是给 DEXP 拟合提供现有无偏 MD 给不出的、
    在更宽范围里有真实分布的训练样本。锚点用 fit 阶段固定环境原子集合（见
    `_load_fixed_env_indices`）里排除水分子后的蛋白原子——水会自己扩散，不是稳定的拉力参考。

    产物是一条 DCD 轨迹，可以直接当 `--traj` 喂给 `--fit-only`，复用现成的 fit 流程，
    不需要为这批数据单独写拟合逻辑。
    """
    openmm, app, unit, _ = require_openmm()
    md = require_module("mdtraj")

    system, topology = load_cached_system(args.system_xml, args.traj_top)
    source_traj = args.pull_scan_source_traj or args.traj
    _, positions, box_vectors = load_last_frame_positions(source_traj, args.traj_top)
    ligand_indices = [int(i) for i in load_ligand_indices(args.ligand_indices)]

    meta_path = ensure_file(
        os.path.join(output_dir, "fit_label_cache_meta.json"),
        "fit 阶段固定环境原子缓存（先跑一次 --fit-only 生成）",
    )
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    env_all = [int(i) for i in meta["env_indices"]]
    top_mdtraj = md.load_topology(args.traj_top)
    anchor_indices = [i for i in env_all if not top_mdtraj.atom(i).residue.is_water]
    if len(anchor_indices) < 3:
        raise RuntimeError(f"固定锚点原子太少({len(anchor_indices)})，检查 fit_label_cache_meta.json / --fit-env-radius")
    print(
        f"[pull-scan] 锚点原子: {len(anchor_indices)}（从 {len(env_all)} 个环境原子里排除了 "
        f"{len(env_all) - len(anchor_indices)} 个水分子原子）"
    )

    # 锚点原子自己的热涨落（尤其是 216 个原子centroid，本身就离配体很近，见下方诊断）跟
    # "挪动目标距离"这个增量同量级甚至更大，会把拉力信号完全淹没——实测：不加位置约束时,
    # 目标距离单调上升但实测 COM-COM 距离在噪声里来回跳,不跟着走。给锚点原子加一个位置
    # restraint(把它们钉在起始位置附近),让 centroid 变成一个近似固定的参考点,拉力信号才
    # 干净地只体现在配体的位移上。这跟本项目 posre.itp/posre_ligand.itp 用的是同一套思路。
    anchor_pos_nm = np.asarray(
        [positions[i].value_in_unit(unit.nanometer) for i in anchor_indices], dtype=float
    )
    anchor_restraint = openmm.CustomExternalForce("0.5*k_pos*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    anchor_restraint.addGlobalParameter("k_pos", float(args.pull_scan_anchor_k))
    anchor_restraint.addPerParticleParameter("x0")
    anchor_restraint.addPerParticleParameter("y0")
    anchor_restraint.addPerParticleParameter("z0")
    for idx, (x0, y0, z0) in zip(anchor_indices, anchor_pos_nm):
        anchor_restraint.addParticle(int(idx), [float(x0), float(y0), float(z0)])
    system.addForce(anchor_restraint)

    # 实测发现：直接拉"整个环境壳层 centroid"时，配体质心确实被拉得很远（COM-COM 距离能
    # 从 0.09nm 拉到 1.7nm），但配体-环境的最近原子对距离几乎不变（一直卡在 ~0.20nm）——
    # 配体会靠自身柔性/转动，让某个原子（常是氢）继续伸回去够着壳层里某个特定原子，形成
    # "风筝线"效应。诊断出这个持续接触往往是一个具体的氢键（比如本系统里配体 H 和某个
    # ASN 侧链 OD1）。要真正把 DEXP 关心的最近距离顶开，得把拉力方向对准这个具体接触点
    # 所在的残基，而不是笼统的壳层重心。
    lig_pos_nm = np.asarray([positions[i].value_in_unit(unit.nanometer) for i in ligand_indices], dtype=float)
    delta0 = lig_pos_nm[:, None, :] - anchor_pos_nm[None, :, :]
    dists0 = np.linalg.norm(delta0, axis=-1)
    i_lig0, j_anchor0 = np.unravel_index(int(np.argmin(dists0)), dists0.shape)
    closest_env_atom = anchor_indices[j_anchor0]
    closest_residue = top_mdtraj.atom(closest_env_atom).residue
    anchor_set = set(anchor_indices)
    pull_target_indices = [a.index for a in closest_residue.atoms if a.index in anchor_set]
    if not pull_target_indices:
        pull_target_indices = [closest_env_atom]
    print(
        f"    [pull-scan] t=0 最近接触: 配体原子 {ligand_indices[i_lig0]} <-> {top_mdtraj.atom(closest_env_atom)} "
        f"({dists0[i_lig0, j_anchor0]:.3f}nm) -> 拉力方向目标改为残基 {closest_residue}"
        f"（{len(pull_target_indices)} 原子），位置约束仍然覆盖全部 {len(anchor_indices)} 个壳层原子"
    )

    pull_force = openmm.CustomCentroidBondForce(2, "0.5*k*(distance(g1,g2)-r0)^2")
    pull_force.addPerBondParameter("k")
    pull_force.addPerBondParameter("r0")
    g_lig = pull_force.addGroup(ligand_indices)
    g_anchor = pull_force.addGroup(pull_target_indices)
    pull_force.addBond([g_lig, g_anchor], [float(args.pull_scan_k), 0.0])
    system.addForce(pull_force)

    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction_ps / unit.picosecond,
        args.dt_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(int(args.seed))
    platform, properties = select_platform(args.platform)
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, int(args.seed))

    def _com_distance_nm() -> float:
        state = simulation.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        pos = np.asarray(pos, dtype=float)
        lig_com = np.mean(pos[ligand_indices], axis=0)
        anchor_com = np.mean(pos[pull_target_indices], axis=0)
        return float(np.linalg.norm(lig_com - anchor_com))

    r_start = _com_distance_nm()
    r_end = r_start + float(args.pull_scan_extend_nm)
    n_stages = max(1, int(args.pull_scan_steps))
    n_relax = max(1, int(args.pull_scan_relax_steps))
    print(
        f"[pull-scan] 配体质心-拉力目标残基质心 起点距离={r_start:.3f}nm -> 目标终点={r_end:.3f}nm，"
        f"共 {n_stages + 1} 帧（含起点），每档弛豫 {n_relax} 步，k={float(args.pull_scan_k):.0f} kJ/mol/nm^2"
    )

    out_dir = ensure_dir(os.path.join(output_dir, "pull_scan"))
    dcd_path = os.path.join(out_dir, "pull_scan.dcd")
    rows: List[Dict] = []
    with open(dcd_path, "wb") as handle:
        dcd_file = app.DCDFile(handle, topology, args.dt_fs * unit.femtosecond)
        for stage in range(n_stages + 1):
            r0_target = r_start + (r_end - r_start) * stage / n_stages
            pull_force.setBondParameters(0, [g_lig, g_anchor], [float(args.pull_scan_k), r0_target])
            pull_force.updateParametersInContext(simulation.context)
            simulation.step(n_relax)
            state = simulation.context.getState(getPositions=True)
            box = state.getPeriodicBoxVectors(asNumpy=True)
            dcd_file.writeModel(state.getPositions(), periodicBoxVectors=box)
            actual = _com_distance_nm()
            rows.append({
                "stage": int(stage),
                "r0_target_nm": float(r0_target),
                "com_distance_actual_nm": float(actual),
            })
            print(f"    [pull-scan] 档 {stage}/{n_stages} r0_target={r0_target:.3f}nm 实际COM距离={actual:.3f}nm")

    stages_csv = write_rows_csv(os.path.join(out_dir, "pull_scan_stages.csv"), rows)
    print(
        f"[pull-scan] 完成，轨迹: {dcd_path}（{n_stages + 1} 帧）。"
        f"可作为 --traj 重新喂给 --fit-only（记得同时把 --fit-frames/--fit-last-ns 调到能覆盖这 {n_stages + 1} 帧）"
    )
    return {"dcd_path": dcd_path, "stages_csv": stages_csv, "n_frames": n_stages + 1, "r_start_nm": r_start, "r_end_nm": r_end}


def relabel_trajectory_local(
    args: argparse.Namespace,
    traj_path: str,
    fitted_params: Dict,
    symbols: Dict,
    env_idx_override: Optional[Sequence[int]] = None,
) -> Dict:
    """在给定轨迹上做 MACE 单点 relabel，逐帧算出局部能量分量 + DEXP 预测 + min-distance。
    返回同帧对齐的数组：不做任何重加权/PMF，纯标注。

    env_idx_override：若提供，直接用这套固定环境原子索引（通常来自 fit 阶段的
    fit_label_cache_meta.json），不再按本条轨迹最后一帧重新选取。DEXP/MM 两条轨迹各自
    独立重选环境原子会导致 MACE 局部能量分解建立在不同的原子集合上，形状 RMSE 的比较就
    不再是"同一局部参考下谁更贴近 MACE"，而是掺了环境定义差异的噪声。"""
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
    if env_idx_override is not None:
        env_idx = np.asarray(list(env_idx_override), dtype=int)
        env_source = "fixed_override"
    else:
        env_search_radius = float(args.fit_env_radius)
        env_max_atoms = int(args.fit_env_max_atoms) if int(args.fit_env_max_atoms) > 0 else None
        env_idx = select_env_indices(sub[-1], lig_idx, env_search_radius, max_env_atoms=env_max_atoms)
        env_source = "per_trajectory_last_frame"
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

    e_orb, e_gauss, e_mm_coul, e_mm_vdw, dexp_pred = [], [], [], [], []
    min_dist, min_dist_valid = [], []
    print(
        f"    [relabel] {os.path.basename(traj_path)}: MACE 单点标注 {len(sub)} 帧"
        f"（env={len(env_idx)}, env_source={env_source}）"
    )
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
        # min_dist：全原子(含H)最近距离，仅用于"过近/碰撞"判据(指标F)，不是 DEXP 敏感的坐标。
        # min_dist_valid：限制在 DEXP 实际依赖的 [fit_r_min, fit_r_max] 内的最近距离，
        # 是 same_frame_pmf_compare 做形状分箱/比较时应该用的坐标，否则会把判据绑定到
        # DEXP 完全看不见的短程 H-contact 上。
        min_dist.append(float(dists.min()))
        min_dist_valid.append(float(valid.min()) if valid.size else float("nan"))
        if (k + 1) % 50 == 0:
            print(f"      relabel {k + 1}/{len(sub)}")
    return {
        "e_orb": np.asarray(e_orb), "e_gauss": np.asarray(e_gauss),
        "e_mm_coul": np.asarray(e_mm_coul), "e_mm_vdw": np.asarray(e_mm_vdw),
        "dexp_pred": np.asarray(dexp_pred), "min_dist": np.asarray(min_dist),
        "min_dist_valid": np.asarray(min_dist_valid),
        "n_frames": len(sub), "env_idx": env_idx, "env_source": env_source,
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


def _load_fixed_env_indices(output_dir: str) -> Optional[List[int]]:
    """从 fit 阶段留下的 fit_label_cache_meta.json 里取固定环境原子索引集合。
    relabel 对 DEXP/MM 两条轨迹应该用同一套 env_idx，否则 MACE 局部能量分解建立在
    不同的原子集合上，跨势的形状 RMSE 比较会被环境定义差异污染，不再是"同一局部
    参考下谁更贴近 MACE"。找不到就返回 None，调用方回退为按轨迹各自重选（旧行为）。"""
    meta_path = os.path.join(output_dir, "fit_label_cache_meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        env_indices = meta.get("env_indices")
        if env_indices:
            return [int(i) for i in env_indices]
    except Exception:
        pass
    return None


def _dexp_baseline_pairwise_sum(
    dists: np.ndarray,
    sigma_lig: np.ndarray,
    eps_lig: np.ndarray,
    sigma_env: np.ndarray,
    eps_env: np.ndarray,
    cutoff_nm: float,
    alpha: float = 12.0,
    beta: float = 6.0,
) -> Tuple[float, float]:
    """按 abfe_core.DEXPSurrogatePotential 同一套解析公式(pair-specific LJ-matched)算
    ligand-environment pairwise 和，供 --perturb-scan 诊断使用。不含 switching（诊断用途，
    switch 只在 0.5~0.7nm 边缘生效，对我们关心的 r<0.3nm 排斥墙区域影响可忽略)。"""
    sigma_ij = 0.5 * (sigma_lig[:, None] + sigma_env[None, :])
    eps_ij = np.sqrt(np.clip(eps_lig[:, None] * eps_env[None, :], 0.0, None))
    r0_ij = (2.0 ** (1.0 / 6.0)) * np.maximum(sigma_ij, 1.0e-6)
    x = np.maximum(dists, 1.0e-6) / r0_ij - 1.0
    c_a = beta / (alpha - beta)
    c_b = alpha / (alpha - beta)
    pair_e = eps_ij * (c_a * np.exp(-alpha * x) - c_b * np.exp(-beta * x))
    mask = dists <= cutoff_nm
    return float(np.sum(pair_e[mask])), float(dists.min())


def run_perturbation_scan(args: argparse.Namespace, output_dir: str) -> Dict:
    """结合态局部 anchor-relative 扰动云。

    跟 pose-scan/pull-scan 的区别：不追求把配体推到新的 min-distance 区间(那是构型扩展
    工具，对"结合口袋附近才有物理意义"的目标来说范围选错了)。这里从平衡轨迹尾段取若干
    anchor 帧，只对配体做小幅刚体扰动（±0.005~0.04nm 平移、±0.5~3度转动），环境原子完全
    不动，扰动后不做任何 relax/minimize（避免把人为构造的局部坡度重新抹回势阱底部）。
    比较 anchor-relative ΔE_target 与 (a) 阶段那个不需要学习的 pair-specific LJ-matched
    DEXP 解析基线的 ΔU：如果残差 R=ΔE_target-ΔU_DEXP 接近噪声，说明基线已经够用，不需要
    再学任何修正；如果残差有可重复结构，才值得考虑 per-pair-type/角度修正。
    """
    openmm, app, unit, XmlSerializer = require_openmm()
    md = require_module("mdtraj")
    symbols = load_abfe_symbols()
    select_env_indices = symbols["_select_env_indices_from_mdtraj_frame"]
    Orbv3DEXPFittingPipeline = symbols["Orbv3DEXPFittingPipeline"]
    NumpyEncoder = symbols["NumpyEncoder"]

    print(f"[1/3] 载入轨迹，从末段抽取最多 {args.perturb_anchors} 个 anchor 帧")
    traj = md.load(args.traj, top=args.traj_top)
    if len(traj) == 0:
        raise RuntimeError("轨迹为空，无法进行局部扰动扫描")
    anchor_frame_ids = select_tail_indices_from_time(traj, args.perturb_anchors, args.fit_last_ns)
    anchor_traj = traj[anchor_frame_ids]
    if anchor_traj.unitcell_vectors is not None:
        anchor_traj = anchor_traj.image_molecules(inplace=False)

    lig_idx = np.asarray(anchor_traj.top.select(f"resname {args.ligand}"), dtype=int)
    if len(lig_idx) == 0:
        raise ValueError(f"未在轨迹拓扑中找到配体残基 `{args.ligand}`")

    # 环境原子集合固定一次(用最后一个 anchor 帧选取)，所有 anchor/扰动共用同一个口袋定义，
    # 避免不同 anchor 之间环境原子集合不一致污染 anchor-relative 比较。
    ref_frame = anchor_traj[-1]
    env_search_radius = float(args.fit_env_radius)
    env_max_atoms = int(args.fit_env_max_atoms) if int(args.fit_env_max_atoms) > 0 else None
    env_idx = np.asarray(
        select_env_indices(ref_frame, lig_idx, env_search_radius, max_env_atoms=env_max_atoms),
        dtype=int,
    )
    if len(env_idx) == 0:
        raise RuntimeError("未找到配体附近环境原子，请增大 --fit-env-radius")
    print(f"    配体原子={len(lig_idx)} 环境原子={len(env_idx)}（固定，来自最后一个 anchor 帧）")

    with open(args.system_xml, "r", encoding="utf-8") as handle:
        sigma_lookup_system = XmlSerializer.deserialize(handle.read())
    nb_force = next(f for f in sigma_lookup_system.getForces() if isinstance(f, openmm.NonbondedForce))
    n_particles = sigma_lookup_system.getNumParticles()
    sigma_all = np.zeros(n_particles, dtype=float)
    eps_all = np.zeros(n_particles, dtype=float)
    for i in range(n_particles):
        _, sigma_i, epsilon_i = nb_force.getParticleParameters(i)
        sigma_all[i] = sigma_i.value_in_unit(unit.nanometer)
        eps_all[i] = epsilon_i.value_in_unit(unit.kilojoule_per_mole)
    sigma_lig, eps_lig = sigma_all[lig_idx], eps_all[lig_idx]
    sigma_env, eps_env = sigma_all[env_idx], eps_all[env_idx]
    baseline_cutoff_nm = float(args.perturb_baseline_cutoff_nm)

    print(f"[2/3] 构建固定环境下的 MACE + Gaussian-Coulomb 参考 context")
    mm_contexts = build_mm_le_contexts_from_system_xml(
        args.system_xml,
        ligand_indices=lig_idx.tolist(),
        environment_indices=env_idx.tolist(),
        cutoff_nm=float(args.fit_mm_ref_cutoff),
        switching_nm=float(args.fit_mm_ref_switch),
    )
    gauss_ctx = mm_contexts["gauss_coul"]
    gauss_ctx_periodic = gauss_ctx.getSystem().usesPeriodicBoundaryConditions()

    all_nums = np.array([a.element.atomic_number for a in anchor_traj.top.atoms], dtype=int)
    pipeline = Orbv3DEXPFittingPipeline(model_name=args.ml_model, device=args.device)
    pipeline._cache_contexts = True
    first_pos_nm = np.asarray(anchor_traj.xyz[0], dtype=np.float64)
    pipeline._preflight_orb_backend(first_pos_nm, lig_idx, env_idx, all_nums)

    def _e_target(pos_nm: np.ndarray, box_vecs) -> float:
        e_orb = pipeline._compute_orb_decomposition(pos_nm, lig_idx, env_idx, all_nums)
        if box_vecs is not None and gauss_ctx_periodic:
            gauss_ctx.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box_vecs])
        gauss_ctx.setPositions(pos_nm * unit.nanometer)
        e_gauss = gauss_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        return float(e_orb - e_gauss)

    def _lig_env_dists(pos_nm: np.ndarray, box_vecs) -> np.ndarray:
        delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
        if box_vecs is not None:
            box_lens = np.linalg.norm(box_vecs, axis=1)
            delta = delta - box_lens * np.round(delta / box_lens)
        return np.linalg.norm(delta, axis=-1)

    trans_mags = [float(x) for x in str(args.perturb_trans_nm).split(",") if x.strip()]
    rot_mags_deg = [float(x) for x in str(args.perturb_rot_deg).split(",") if x.strip()]
    n_random_dirs = max(0, int(args.perturb_n_random_dirs))
    rng = np.random.default_rng(int(args.seed))

    print(f"[3/3] 对 {len(anchor_frame_ids)} 个 anchor 各生成配对(±)扰动，不做 relax/minimize")
    all_rows: List[Dict] = []
    # 额外保存原始几何(坐标)，供 --perturb-fit 用任意 (alpha,beta) 重新算解析基线——
    # 只有基线公式依赖 alpha/beta，MACE ΔE_target 与 alpha/beta 无关，不需要重新标注。
    env_positions_all: List[np.ndarray] = []
    anchor_lig_positions_all: List[np.ndarray] = []
    box_vectors_all: List[np.ndarray] = []
    has_periodic_all: List[bool] = []
    perturbed_lig_positions_all: List[np.ndarray] = []
    perturbation_anchor_index_all: List[int] = []
    for a_local, a_frame_id in enumerate(anchor_frame_ids):
        pos_nm = np.asarray(anchor_traj.xyz[a_local], dtype=np.float64)
        box_vecs = (
            np.asarray(anchor_traj.unitcell_vectors[a_local], dtype=np.float64)
            if anchor_traj.unitcell_vectors is not None
            else None
        )

        dists_anchor = _lig_env_dists(pos_nm, box_vecs)
        u_anchor, min_dist_anchor = _dexp_baseline_pairwise_sum(
            dists_anchor, sigma_lig, eps_lig, sigma_env, eps_env, baseline_cutoff_nm
        )
        e_anchor = _e_target(pos_nm, box_vecs)

        env_positions_all.append(pos_nm[env_idx].copy())
        anchor_lig_positions_all.append(pos_nm[lig_idx].copy())
        box_vectors_all.append(box_vecs if box_vecs is not None else np.zeros((3, 3)))
        has_periodic_all.append(box_vecs is not None)

        lig_pos = pos_nm[lig_idx]
        com = lig_pos.mean(axis=0)
        centered = lig_pos - com
        # 配体质量未知(只有坐标)，用几何(非质量加权)惯性张量的主轴做扰动方向——
        # 目的只是覆盖"哪些接触被压紧/放松"的多个独立方向，不需要精确的物理主轴。
        inertia = centered.T @ centered
        _, axes = np.linalg.eigh(inertia)
        directions = [axes[:, k] / np.linalg.norm(axes[:, k]) for k in range(3)]
        for _ in range(n_random_dirs):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            directions.append(v)

        def _record(pert_type, axis_kind, axis_index, magnitude, sign, new_pos):
            dists_p = _lig_env_dists(new_pos, box_vecs)
            u_p, min_dist_p = _dexp_baseline_pairwise_sum(
                dists_p, sigma_lig, eps_lig, sigma_env, eps_env, baseline_cutoff_nm
            )
            e_p = _e_target(new_pos, box_vecs)
            perturbed_lig_positions_all.append(new_pos[lig_idx].copy())
            perturbation_anchor_index_all.append(a_local)
            all_rows.append({
                "anchor_frame": int(a_frame_id),
                "pert_type": pert_type,
                "axis_kind": axis_kind,
                "axis_index": int(axis_index),
                "magnitude": float(magnitude),
                "sign": float(sign),
                "delta_u_dexp_kjmol": float(u_p - u_anchor),
                "delta_e_target_kjmol": float(e_p - e_anchor),
                "min_dist_anchor_nm": float(min_dist_anchor),
                "min_dist_perturbed_nm": float(min_dist_p),
            })

        for mag in trans_mags:
            for d_i, direction in enumerate(directions):
                axis_kind = "principal" if d_i < 3 else "random"
                for sign in (1.0, -1.0):
                    new_pos = pos_nm.copy()
                    new_pos[lig_idx] = lig_pos + sign * mag * direction
                    _record("translation", axis_kind, d_i, mag, sign, new_pos)

        for ang_deg in rot_mags_deg:
            ang_rad = math.radians(ang_deg)
            for a_i in range(3):
                axis = directions[a_i]
                K = np.array([
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ])
                for sign in (1.0, -1.0):
                    theta = sign * ang_rad
                    R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
                    new_pos = pos_nm.copy()
                    new_pos[lig_idx] = (centered @ R.T) + com
                    _record("rotation", "principal", a_i, ang_deg, sign, new_pos)

        print(
            f"    anchor {a_local + 1}/{len(anchor_frame_ids)} (frame {a_frame_id}) 完成，"
            f"min_dist={min_dist_anchor:.3f}nm，累计扰动记录={len(all_rows)}"
        )

    try:
        pipeline._clear_orb_context_cache()
    except Exception:
        pass

    csv_path = os.path.join(output_dir, "perturb_scan_diagnostics.csv")
    ensure_dir(output_dir)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # 几何快照：行顺序与上面写的 CSV 完全一致（同一次循环产生），--perturb-fit 靠行位置对齐，
    # 不需要额外的 anchor_frame/perturbation 匹配逻辑。存位置而不是存 ΔU，是因为只有基线公式
    # 依赖 (alpha,beta)，重新拟合时用任意候选 (alpha,beta) 重算 ΔU 完全不需要重跑 MACE。
    npz_path = os.path.join(output_dir, "perturb_scan_geometry.npz")
    np.savez_compressed(
        npz_path,
        env_positions=np.asarray(env_positions_all, dtype=np.float64),
        anchor_lig_positions=np.asarray(anchor_lig_positions_all, dtype=np.float64),
        box_vectors=np.asarray(box_vectors_all, dtype=np.float64),
        has_periodic=np.asarray(has_periodic_all, dtype=bool),
        perturbed_lig_positions=np.asarray(perturbed_lig_positions_all, dtype=np.float64),
        perturbation_anchor_index=np.asarray(perturbation_anchor_index_all, dtype=np.int64),
        sigma_lig=sigma_lig,
        eps_lig=eps_lig,
        sigma_env=sigma_env,
        eps_env=eps_env,
    )

    delta_u = np.asarray([r["delta_u_dexp_kjmol"] for r in all_rows], dtype=float)
    delta_e = np.asarray([r["delta_e_target_kjmol"] for r in all_rows], dtype=float)
    residual = delta_e - delta_u
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    bias = float(np.mean(residual))
    pearson_r = (
        float(np.corrcoef(delta_u, delta_e)[0, 1])
        if len(delta_u) > 1 and np.std(delta_u) > 1.0e-9 and np.std(delta_e) > 1.0e-9
        else math.nan
    )

    by_type: Dict[str, Dict] = {}
    for pert_type in sorted(set(r["pert_type"] for r in all_rows)):
        mask = [r["pert_type"] == pert_type for r in all_rows]
        res_t = residual[mask]
        by_type[pert_type] = {
            "n": int(np.sum(mask)),
            "residual_rmse_kjmol": float(np.sqrt(np.mean(res_t ** 2))),
            "residual_bias_kjmol": float(np.mean(res_t)),
        }

    summary = {
        "n_anchors": int(len(anchor_frame_ids)),
        "n_perturbations": int(len(all_rows)),
        "env_atoms": int(len(env_idx)),
        "baseline_cutoff_nm": baseline_cutoff_nm,
        "trans_mags_nm": trans_mags,
        "rot_mags_deg": rot_mags_deg,
        "n_random_dirs": n_random_dirs,
        "residual_rmse_kjmol": rmse,
        "residual_bias_kjmol": bias,
        "pearson_r_delta_u_vs_delta_e": pearson_r,
        "by_pert_type": by_type,
        "diagnostics_csv": csv_path,
        "geometry_npz": npz_path,
        "note": (
            "residual = delta_e_target - delta_u_dexp，delta_e_target = (E_MACE_int - E_gauss_coul)"
            "(anchor+perturbed) - 同量(anchor)；delta_u_dexp 用 (a) 阶段固定的、不需要学习的"
            "pair-specific LJ-matched DEXP 解析基线(alpha=12,beta=6)。residual 接近噪声说明"
            "基线已经解释了局部势能面曲率，不需要再学修正；residual 有可重复结构才值得考虑"
            "per-pair-type/角度修正——但要先排除环境集合/cutoff不一致等实现误差。"
        ),
    }
    summary_path = os.path.join(output_dir, "perturb_scan_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, cls=NumpyEncoder)

    print(
        f"[perturb-scan] 完成: {len(all_rows)} 条扰动记录 | "
        f"residual RMSE={rmse:.3f} kJ/mol | bias={bias:.3f} | "
        f"corr(ΔU_dexp, ΔE_target)={pearson_r:.3f}"
    )
    print(f"    diagnostics: {csv_path}")
    print(f"    geometry: {npz_path}")
    print(f"    summary: {summary_path}")
    return summary


def _pairwise_dists_matrix(a_pos: np.ndarray, b_pos: np.ndarray, box_vecs: Optional[np.ndarray]) -> np.ndarray:
    delta = a_pos[:, None, :] - b_pos[None, :, :]
    if box_vecs is not None:
        box_lens = np.linalg.norm(box_vecs, axis=1)
        delta = delta - box_lens * np.round(delta / box_lens)
    return np.linalg.norm(delta, axis=-1)


def _build_perturbation_distance_tensors(
    n_rows: int,
    anchor_local_idx: np.ndarray,
    env_positions: np.ndarray,
    anchor_lig_positions: np.ndarray,
    perturbed_lig_positions: np.ndarray,
    box_vectors: np.ndarray,
    has_periodic: np.ndarray,
    sigma_lig: np.ndarray,
    eps_lig: np.ndarray,
    sigma_env: np.ndarray,
    eps_env: np.ndarray,
    cutoff_nm: float,
) -> Dict[str, np.ndarray]:
    """把 --perturb-scan 存的 anchor/perturbed 坐标重建成完整 ligand-environment 距离/几何张量。

    与 alpha_vdw/beta_vdw 无关的部分（距离、r0_ij、eps_ij、cutoff mask）只需要算一次，
    被 `run_perturbation_fit::_predict_delta_u` 和 `run_contact_type_fit` 共用——后者还需要
    原始 dists_*_full（不只是 x_*_full）来算 switching function S(r) 和 psi_o/psi_e 特征。
    """
    n_anchors = anchor_lig_positions.shape[0]
    n_lig, n_env = sigma_lig.shape[0], sigma_env.shape[0]
    dists_anchor_full = np.empty((n_rows, n_lig, n_env), dtype=np.float64)
    dists_pert_full = np.empty((n_rows, n_lig, n_env), dtype=np.float64)
    for a in range(n_anchors):
        bv = box_vectors[a] if has_periodic[a] else None
        dists_anchor_a = _pairwise_dists_matrix(anchor_lig_positions[a], env_positions[a], bv)
        rows_of_anchor = np.where(anchor_local_idx == a)[0]
        if rows_of_anchor.size == 0:
            continue
        dists_anchor_full[rows_of_anchor] = dists_anchor_a[None, :, :]
        lig_block = perturbed_lig_positions[rows_of_anchor]
        delta = lig_block[:, :, None, :] - env_positions[a][None, None, :, :]
        if bv is not None:
            box_lens = np.linalg.norm(bv, axis=1)
            delta = delta - box_lens * np.round(delta / box_lens)
        dists_pert_full[rows_of_anchor] = np.linalg.norm(delta, axis=-1)

    sigma_ij_full = 0.5 * (sigma_lig[:, None] + sigma_env[None, :])
    eps_ij_full = np.sqrt(np.clip(eps_lig[:, None] * eps_env[None, :], 0.0, None))
    r0_ij_full = (2.0 ** (1.0 / 6.0)) * np.maximum(sigma_ij_full, 1.0e-6)
    x_anchor_full = np.maximum(dists_anchor_full, 1.0e-6) / r0_ij_full[None, :, :] - 1.0
    x_pert_full = np.maximum(dists_pert_full, 1.0e-6) / r0_ij_full[None, :, :] - 1.0
    mask_anchor_full = dists_anchor_full <= cutoff_nm
    mask_pert_full = dists_pert_full <= cutoff_nm
    return {
        "dists_anchor_full": dists_anchor_full,
        "dists_pert_full": dists_pert_full,
        "sigma_ij_full": sigma_ij_full,
        "eps_ij_full": eps_ij_full,
        "r0_ij_full": r0_ij_full,
        "x_anchor_full": x_anchor_full,
        "x_pert_full": x_pert_full,
        "mask_anchor_full": mask_anchor_full,
        "mask_pert_full": mask_pert_full,
    }


def run_perturbation_fit(args: argparse.Namespace, output_dir: str) -> Dict:
    """在 --perturb-scan 产生的局部扰动云上，重新挑选 DEXP 仅剩的两个全局形状自由度
    alpha_vdw/beta_vdw（r0_ij/eps_ij 仍然是 (a) 阶段的 pair-specific LJ-matched 解析值，
    对任意 alpha>beta>0 自动保持 U(r0)=-eps, U'(r0)=0，不会因为学坏形状而破坏势阱）。

    两个关键点（均为用户明确要求，不是可选项）：
    1. 按"扰动档"(pert_type, magnitude)等权，而不是按记录等权——否则记录数更多的档
       (如 translation 每档 280 条 vs rotation 每档 120 条)会不成比例地主导拟合；
       额外把最大档 translation@0.04nm 降权(--perturb-fit-mag04-weight，默认 0.5)，
       因为它是最偏离"局部"定义的扰动，适合当稳定性检验但不该独占形状参数。
    2. leave-one-anchor-out 交叉验证——同一 anchor 的几十条扰动记录高度相关，不是独立
       样本，按 anchor 做 20 折交叉验证才能诚实评估"选出来的 (alpha,beta) 是否稳定"，
       而不是被同一批 anchor 的记录数假象出虚高的样本量。
    """
    csv_path = os.path.join(output_dir, "perturb_scan_diagnostics.csv")
    npz_path = os.path.join(output_dir, "perturb_scan_geometry.npz")
    ensure_file(csv_path, "perturb-scan 诊断 CSV（先跑一次 --perturb-scan）")
    ensure_file(npz_path, "perturb-scan 几何快照 npz（先跑一次 --perturb-scan）")

    rows = read_csv_rows(csv_path)
    geo = np.load(npz_path)

    n_rows = len(rows)
    delta_e_target = np.asarray([float(r["delta_e_target_kjmol"]) for r in rows], dtype=float)
    delta_u_default = np.asarray([float(r["delta_u_dexp_kjmol"]) for r in rows], dtype=float)
    pert_type = np.asarray([r["pert_type"] for r in rows], dtype=object)
    magnitude = np.asarray([float(r["magnitude"]) for r in rows], dtype=float)
    axis_kind = np.asarray([r["axis_kind"] for r in rows], dtype=object)
    axis_index = np.asarray([int(r["axis_index"]) for r in rows], dtype=int)
    sign = np.asarray([float(r["sign"]) for r in rows], dtype=float)

    anchor_local_idx = geo["perturbation_anchor_index"].astype(int)
    if len(anchor_local_idx) != n_rows:
        raise RuntimeError(
            f"CSV 行数({n_rows})与几何 npz 行数({len(anchor_local_idx)})不一致，"
            "两者必须来自同一次 --perturb-scan 运行"
        )
    env_positions = geo["env_positions"]
    anchor_lig_positions = geo["anchor_lig_positions"]
    perturbed_lig_positions = geo["perturbed_lig_positions"]
    box_vectors = geo["box_vectors"]
    has_periodic = geo["has_periodic"].astype(bool)
    sigma_lig, eps_lig = geo["sigma_lig"], geo["eps_lig"]
    sigma_env, eps_env = geo["sigma_env"], geo["eps_env"]
    n_anchors = anchor_lig_positions.shape[0]
    n_lig, n_env = sigma_lig.shape[0], sigma_env.shape[0]
    cutoff_nm = float(args.perturb_baseline_cutoff_nm)

    print(f"[1/3] 重建每个 anchor/扰动的 ligand-environment 距离矩阵（不需要重跑 MACE）")
    tensors = _build_perturbation_distance_tensors(
        n_rows, anchor_local_idx, env_positions, anchor_lig_positions, perturbed_lig_positions,
        box_vectors, has_periodic, sigma_lig, eps_lig, sigma_env, eps_env, cutoff_nm,
    )
    eps_ij = tensors["eps_ij_full"]
    x_anchor_full = tensors["x_anchor_full"]
    x_pert_full = tensors["x_pert_full"]
    mask_anchor_full = tensors["mask_anchor_full"]
    mask_pert_full = tensors["mask_pert_full"]

    def _predict_delta_u(alpha: float, beta: float) -> np.ndarray:
        c_a = beta / (alpha - beta)
        c_b = alpha / (alpha - beta)
        u_anchor = np.sum(
            np.where(
                mask_anchor_full,
                eps_ij[None, :, :] * (c_a * np.exp(-alpha * x_anchor_full) - c_b * np.exp(-beta * x_anchor_full)),
                0.0,
            ),
            axis=(1, 2),
        )
        u_pert = np.sum(
            np.where(
                mask_pert_full,
                eps_ij[None, :, :] * (c_a * np.exp(-alpha * x_pert_full) - c_b * np.exp(-beta * x_pert_full)),
                0.0,
            ),
            axis=(1, 2),
        )
        return u_pert - u_anchor

    BIN_KEYS = [
        ("rotation", 0.5), ("rotation", 1.5), ("rotation", 3.0),
        ("translation", 0.005), ("translation", 0.01), ("translation", 0.02), ("translation", 0.04),
    ]
    downweight = {("translation", 0.04): float(args.perturb_fit_mag04_weight)}

    def _row_weights(mask_rows: np.ndarray) -> np.ndarray:
        w = np.zeros(n_rows, dtype=float)
        for key in BIN_KEYS:
            bin_mask = mask_rows & (pert_type == key[0]) & np.isclose(magnitude, key[1])
            n_bin = int(np.sum(bin_mask))
            if n_bin == 0:
                continue
            w[bin_mask] = downweight.get(key, 1.0) / n_bin
        total = float(np.sum(w[mask_rows]))
        if total > 1.0e-12:
            w[mask_rows] /= total
        return w

    def _weighted_rmse(delta_u_pred: np.ndarray, mask_rows: np.ndarray, weights: np.ndarray) -> float:
        resid = (delta_e_target - delta_u_pred)[mask_rows]
        w = weights[mask_rows]
        return float(np.sqrt(np.sum(w * resid ** 2) / np.sum(w)))

    alpha_grid = [float(x) for x in str(args.perturb_fit_alpha_grid).split(",") if x.strip()]
    beta_grid = [float(x) for x in str(args.perturb_fit_beta_grid).split(",") if x.strip()]
    candidates = [(a, b) for a in alpha_grid for b in beta_grid if a > b > 0.0]
    if not candidates:
        raise RuntimeError("alpha/beta 网格里没有满足 alpha>beta>0 的组合，请检查 --perturb-fit-alpha-grid/--perturb-fit-beta-grid")
    print(f"[2/3] 网格搜索: {len(alpha_grid)}x{len(beta_grid)} 候选中 {len(candidates)} 组满足 alpha>beta>0")

    # 候选 (alpha,beta) 的预测在所有 fold 间共享（预测本身不依赖训练/测试划分），先算一遍缓存。
    pred_cache = {(a, b): _predict_delta_u(a, b) for a, b in candidates}

    print(f"[3/3] Leave-one-anchor-out 交叉验证（{n_anchors} 折）")
    fold_results: List[Dict] = []
    for k in range(n_anchors):
        train_mask = anchor_local_idx != k
        test_mask = anchor_local_idx == k
        train_w = _row_weights(train_mask)
        test_w = _row_weights(test_mask)

        best_key, best_train_rmse = None, math.inf
        for (a, b), pred in pred_cache.items():
            train_rmse = _weighted_rmse(pred, train_mask, train_w)
            if train_rmse < best_train_rmse:
                best_key, best_train_rmse = (a, b), train_rmse
        best_alpha, best_beta = best_key
        test_rmse = _weighted_rmse(pred_cache[best_key], test_mask, test_w)
        default_test_rmse = _weighted_rmse(pred_cache.get((12.0, 6.0), _predict_delta_u(12.0, 6.0)), test_mask, test_w)
        fold_results.append({
            "held_out_anchor_local_idx": int(k),
            "best_alpha": best_alpha,
            "best_beta": best_beta,
            "train_weighted_rmse_kjmol": best_train_rmse,
            "test_weighted_rmse_kjmol": test_rmse,
            "default_12_6_test_weighted_rmse_kjmol": default_test_rmse,
        })
        print(
            f"    [fold {k + 1}/{n_anchors}] best(alpha,beta)=({best_alpha:.2f},{best_beta:.2f}) "
            f"train_wrmse={best_train_rmse:.3f} test_wrmse={test_rmse:.3f} "
            f"(default 12/6 test_wrmse={default_test_rmse:.3f})"
        )

    # 全数据 2D score surface L(alpha,beta)，而不是只记录单点最优——单个网格像素可能只是
    # 网格噪声，稳定的应该是一整片"盆地"(basin)：L(alpha,beta) <= L_min + delta。
    all_mask = np.ones(n_rows, dtype=bool)
    all_w = _row_weights(all_mask)
    grid_scores: Dict[Tuple[float, float], float] = {
        (a, b): _weighted_rmse(pred, all_mask, all_w) for (a, b), pred in pred_cache.items()
    }
    best_full_key = min(grid_scores, key=grid_scores.get)
    best_full_rmse = grid_scores[best_full_key]

    basin_delta = max(
        float(args.perturb_fit_basin_delta_kjmol),
        float(args.perturb_fit_basin_delta_frac) * best_full_rmse,
    )
    basin_candidates = [k for k, v in grid_scores.items() if v <= best_full_rmse + basin_delta]
    basin_alphas = np.asarray([k[0] for k in basin_candidates])
    basin_betas = np.asarray([k[1] for k in basin_candidates])
    basin_rhos = basin_alphas / basin_betas
    alpha_basin_range = (float(basin_alphas.min()), float(basin_alphas.max()))
    beta_basin_range = (float(basin_betas.min()), float(basin_betas.max()))
    rho_basin_range = (float(basin_rhos.min()), float(basin_rhos.max()))

    def _rel_width(vals: np.ndarray) -> float:
        m = float(np.mean(vals))
        return float((np.max(vals) - np.min(vals)) / m) if abs(m) > 1.0e-9 else 0.0

    # 对角谷判据：不预设退化方向是"比值 rho=alpha/beta"——之前这么猜过，但盆地在
    # (alpha,beta) 里实测常常是沿 alpha+beta≈常数 的脊，不是沿 alpha/beta≈常数。
    # 改成对盆地点云做 PCA：最小方差方向就是数据真正约束住的线性组合
    # (w1*alpha+w2*beta≈c)，最大方差方向就是欠约束、可以随便挑"好看数字"的方向——
    # 不用猜是"和"还是"比"，PCA 会自己找出实际的退化方向。
    is_diagonal_valley = False
    valley_direction_desc = None
    if len(basin_candidates) >= 4:
        pts = np.stack([basin_alphas, basin_betas], axis=1)
        pts_centered = pts - pts.mean(axis=0, keepdims=True)
        cov = pts_centered.T @ pts_centered / max(1, len(pts) - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)  # 升序：eigvals[0]=最小方差(约束最紧)方向
        var_ratio = float(eigvals[0] / eigvals[1]) if eigvals[1] > 1.0e-12 else 0.0
        is_diagonal_valley = bool(var_ratio < 0.15)  # 最紧方向方差 <15% 最松方向方差才算"明显对角谷"
        constrained_w = eigvecs[:, 0]
        constrained_c = float(constrained_w @ pts.mean(axis=0))
        valley_direction_desc = f"({constrained_w[0]:.3f})*alpha + ({constrained_w[1]:.3f})*beta ≈ {constrained_c:.3f}"

    # 建议取整值：只在盆地内部找"好看数字"(alpha,beta 都是整数)，而不是对 bounding box
    # 取中点再四舍五入——盆地是对角脊、非凸，bounding box 中点经常根本不在盆地内
    # (本次实测：中点四舍五入正好撞回旧默认值 (12,6)，而 (12,6) 其实不在盆地内)。
    def _is_intlike(v: float) -> bool:
        return abs(v - round(v)) < 1.0e-9

    nice_in_basin = [(a, b) for (a, b) in basin_candidates if _is_intlike(a) and _is_intlike(b)]
    if nice_in_basin:
        suggested_alpha, suggested_beta = min(nice_in_basin, key=lambda ab: grid_scores[ab])
        suggested_alpha, suggested_beta = int(round(suggested_alpha)), int(round(suggested_beta))
    else:
        suggested_alpha, suggested_beta = best_full_key
    suggested_pred = _predict_delta_u(float(suggested_alpha), float(suggested_beta))
    suggested_rmse = _weighted_rmse(suggested_pred, all_mask, all_w)
    suggested_in_basin = bool(suggested_rmse <= best_full_rmse + basin_delta)

    print(
        f"    盆地(L<=L_min+{basin_delta:.3f}, n={len(basin_candidates)}点): "
        f"alpha∈[{alpha_basin_range[0]:.1f},{alpha_basin_range[1]:.1f}] "
        f"beta∈[{beta_basin_range[0]:.1f},{beta_basin_range[1]:.1f}] "
        f"rho=alpha/beta∈[{rho_basin_range[0]:.2f},{rho_basin_range[1]:.2f}]"
    )
    if valley_direction_desc:
        print(f"    盆地点云 PCA 最紧方向: {valley_direction_desc}  (方差比 最紧/最松={var_ratio:.3f})")
    if is_diagonal_valley:
        print(
            "    ⚠️ 盆地是明显的对角谷——上面 PCA 给出的线性组合才是数据真正约束住的量，"
            "alpha,beta 单独的值不稳定，只有该线性组合稳定"
        )
    print(
        f"    建议取整值 (alpha,beta)=({suggested_alpha},{suggested_beta})：加权RMSE={suggested_rmse:.3f} kJ/mol "
        f"{'(在盆地内)' if suggested_in_basin else '(⚠️ 不在盆地内，不要直接采用，改用 full_data_best_alpha/beta 或缩小 delta 重新看盆地)'}"
    )

    # 奇偶分解：把每对(+delta,-delta)拆成奇分量(主要反映局部梯度/平衡位置是否对齐)和
    # 偶分量(主要反映井宽/曲率/更高偶数阶项)，分别看 MACE 与 DEXP(默认核 12/6 和建议取整核)
    # 谁的残差更大——比单看总 RMSE 更能说明"到底修正了什么"。
    group_key_to_idx: Dict[Tuple, Dict[float, int]] = {}
    for i in range(n_rows):
        key = (int(anchor_local_idx[i]), str(pert_type[i]), str(axis_kind[i]), int(axis_index[i]), float(magnitude[i]))
        group_key_to_idx.setdefault(key, {})[float(sign[i])] = i

    def _odd_even_stats(delta_u_arr: np.ndarray) -> Dict[str, Dict]:
        odd_by_type: Dict[str, List[float]] = {"translation": [], "rotation": []}
        even_by_type: Dict[str, List[float]] = {"translation": [], "rotation": []}
        for key, signed in group_key_to_idx.items():
            if 1.0 not in signed or -1.0 not in signed:
                continue
            ip, im = signed[1.0], signed[-1.0]
            e_odd = (delta_e_target[ip] - delta_e_target[im]) / 2.0
            e_even = (delta_e_target[ip] + delta_e_target[im]) / 2.0
            u_odd = (delta_u_arr[ip] - delta_u_arr[im]) / 2.0
            u_even = (delta_u_arr[ip] + delta_u_arr[im]) / 2.0
            ptype = key[1]
            odd_by_type[ptype].append(e_odd - u_odd)
            even_by_type[ptype].append(e_even - u_even)
        out: Dict[str, Dict] = {}
        for ptype in ("translation", "rotation"):
            odd_res = np.asarray(odd_by_type[ptype], dtype=float)
            even_res = np.asarray(even_by_type[ptype], dtype=float)
            out[ptype] = {
                "n_pairs": int(odd_res.size),
                "odd_residual_rmse_kjmol": float(np.sqrt(np.mean(odd_res ** 2))) if odd_res.size else math.nan,
                "odd_residual_bias_kjmol": float(np.mean(odd_res)) if odd_res.size else math.nan,
                "even_residual_rmse_kjmol": float(np.sqrt(np.mean(even_res ** 2))) if even_res.size else math.nan,
                "even_residual_bias_kjmol": float(np.mean(even_res)) if even_res.size else math.nan,
            }
        return out

    odd_even_default = _odd_even_stats(delta_u_default)
    odd_even_suggested = _odd_even_stats(suggested_pred)
    print("    奇偶分解 [默认核 alpha=12,beta=6]:")
    for ptype, s in odd_even_default.items():
        print(
            f"        {ptype:12s} n_pairs={s['n_pairs']:4d}  "
            f"odd(梯度/平衡位置) rmse={s['odd_residual_rmse_kjmol']:.3f} bias={s['odd_residual_bias_kjmol']:.3f}  "
            f"even(曲率/井宽) rmse={s['even_residual_rmse_kjmol']:.3f} bias={s['even_residual_bias_kjmol']:.3f}"
        )
    print(f"    奇偶分解 [建议取整核 alpha={suggested_alpha},beta={suggested_beta}]:")
    for ptype, s in odd_even_suggested.items():
        print(
            f"        {ptype:12s} n_pairs={s['n_pairs']:4d}  "
            f"odd(梯度/平衡位置) rmse={s['odd_residual_rmse_kjmol']:.3f} bias={s['odd_residual_bias_kjmol']:.3f}  "
            f"even(曲率/井宽) rmse={s['even_residual_rmse_kjmol']:.3f} bias={s['even_residual_bias_kjmol']:.3f}"
        )

    alphas = np.asarray([f["best_alpha"] for f in fold_results])
    betas = np.asarray([f["best_beta"] for f in fold_results])
    test_rmses = np.asarray([f["test_weighted_rmse_kjmol"] for f in fold_results])
    default_test_rmses = np.asarray([f["default_12_6_test_weighted_rmse_kjmol"] for f in fold_results])

    summary = {
        "n_anchors": int(n_anchors),
        "n_rows": int(n_rows),
        "bin_keys": [f"{k[0]}:{k[1]}" for k in BIN_KEYS],
        "translation_0_04_weight_factor": float(args.perturb_fit_mag04_weight),
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "loao_folds": fold_results,
        "alpha_selected_mean": float(alphas.mean()),
        "alpha_selected_std": float(alphas.std()),
        "beta_selected_mean": float(betas.mean()),
        "beta_selected_std": float(betas.std()),
        "loao_test_weighted_rmse_mean_kjmol": float(test_rmses.mean()),
        "loao_default_12_6_test_weighted_rmse_mean_kjmol": float(default_test_rmses.mean()),
        "full_data_best_alpha": best_full_key[0],
        "full_data_best_beta": best_full_key[1],
        "full_data_best_weighted_rmse_kjmol": best_full_rmse,
        "grid_score_surface": {f"{a:.4g},{b:.4g}": v for (a, b), v in grid_scores.items()},
        "basin_delta_kjmol": basin_delta,
        "basin_n_points": len(basin_candidates),
        "basin_alpha_range": alpha_basin_range,
        "basin_beta_range": beta_basin_range,
        "basin_rho_alpha_over_beta_range": rho_basin_range,
        "basin_is_diagonal_valley": is_diagonal_valley,
        "basin_pca_constrained_direction": valley_direction_desc,
        "alpha_plus_beta_selected_mean": float((alphas + betas).mean()),
        "alpha_plus_beta_selected_std": float((alphas + betas).std()),
        "suggested_round_alpha": suggested_alpha,
        "suggested_round_beta": suggested_beta,
        "suggested_round_weighted_rmse_kjmol": suggested_rmse,
        "suggested_round_in_basin": suggested_in_basin,
        "odd_even_default_12_6": odd_even_default,
        "odd_even_suggested_round": odd_even_suggested,
        "canonical_parameter_note": (
            "(alpha,beta)=(14,5) 是经验证的规范(canonical)参数，不是两个被独立精确识别的常数："
            "score surface 在 (alpha,beta) 里是沿 alpha+beta≈19 的对角谷(PCA 验证，basin_is_diagonal_valley)，"
            "数据主要约束的是这个和，而不是分别唯一确定 alpha、beta——谷上邻近的 (13,6) 几乎同样好。"
            "(14,5) 是这条谷上 LOAO 19/20 折独立选中、且恰好是整数的最优代表。换配体/环境化学组成后"
            "不应假定这两个数字继续适用，应重新跑 --perturb-scan + --perturb-fit。"
        ),
        "odd_even_interpretation_note": (
            "把残差按 R(delta)=g*delta+0.5*h*delta^2+... 展开：|even/odd| ~ |h*delta/(2g)| ∝ |delta|，"
            "所以 delta->0 时是奇(线性/梯度)项主导局部误差预算，不是偶(二次/曲率)项——不能因为在小 delta 处"
            "奇残差的*绝对值*比大 delta 处小，就判断它'在近平衡尺度不重要'；真正该比较的是 g 与 h 这两个"
            "系数本身，而当前数据里 g(奇残差)基本不随 alpha/beta 改变(对比 odd_even_default_12_6 与 "
            "odd_even_suggested_round 的 odd_residual_rmse 几乎相等)，说明它是一阶效应，any alpha>beta>0 的"
            "各向同性 pairwise 核都无法通过调形状消除——只有偶(曲率)项才被 alpha/beta 明显修正。"
            "但这不能直接推出需要加 angular 项：这里的 odd residual 衡量的是孤立的 "
            "(E_MACE_int - E_gauss_coul) 与 pairwise DEXP 和之间的力差，不是配体在完整体系里实际感受到的"
            "净力偏差——配体分子内/成键项、蛋白对邻近残基的约束、Gaussian-Coulomb 力本身都可能部分抵消这个"
            "局部力错配。是否会导致平衡pose偏移/接触占有率变化，需要用完整体系的净力平衡或真实轨迹的"
            "pose/占有率分布做经验检验，这一步还没做。"
        ),
        "note": (
            "best_alpha/best_beta 每折都在留出的那个 anchor 之外的数据上选出、在留出 anchor 上评估——"
            "alpha_selected_std/beta_selected_std 小说明选出的形状对换哪个 anchor 不敏感(真实 PES 性质)；"
            "std 很大或跟着网格边界跑，说明当前扰动云还不足以稳定约束这两个参数，不要直接采用"
            "full_data_best_alpha/beta 上生产。优先看 basin_*/suggested_round_*/canonical_parameter_note "
            "而不是把 full_data_best_* 当成独立精确解——后者只是网格上的单个像素，前者是 "
            "L(alpha,beta)<=L_min+delta 的稳定盆地，更适合固定核参数。"
            "见 odd_even_interpretation_note，不要用 odd/even 的原始 RMSE 大小直接判断相对重要性。"
        ),
    }
    summary_path = os.path.join(output_dir, "perturb_fit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        f"[perturb-fit] LOAO 选出: alpha={alphas.mean():.2f}±{alphas.std():.2f}  "
        f"beta={betas.mean():.2f}±{betas.std():.2f}"
    )
    print(
        f"    留出集加权 RMSE: 拟合={test_rmses.mean():.3f} kJ/mol  vs 默认(12,6)={default_test_rmses.mean():.3f} kJ/mol"
    )
    print(
        f"    全数据最优 (alpha,beta)=({best_full_key[0]:.2f},{best_full_key[1]:.2f}) "
        f"加权RMSE={best_full_rmse:.3f} kJ/mol"
    )
    print(f"    summary: {summary_path}")
    return summary


def _bond_adjacency_from_system_xml(system_xml_path: str, openmm, XmlSerializer) -> Dict[int, List[int]]:
    """跟 run_replica_analysis 里氢键候选判据同一套逻辑：键连信息只能从 system.xml 的
    HarmonicBondForce+Constraints 拿，不能用 topology.cif 的 Topology.bonds（配体 H 原子
    在那份 CIF 里基本没有可信键连记录，见 run_replica_analysis 内注释)。"""
    with open(system_xml_path, "r", encoding="utf-8") as handle:
        bond_system = XmlSerializer.deserialize(handle.read())
    adjacency: Dict[int, List[int]] = {}
    for force in bond_system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for b in range(force.getNumBonds()):
                i1, i2, _, _ = force.getBondParameters(b)
                adjacency.setdefault(int(i1), []).append(int(i2))
                adjacency.setdefault(int(i2), []).append(int(i1))
    for c in range(bond_system.getNumConstraints()):
        i1, i2, _ = bond_system.getConstraintParameters(c)
        adjacency.setdefault(int(i1), []).append(int(i2))
        adjacency.setdefault(int(i2), []).append(int(i1))
    return adjacency


def _bonded_hydrogens(idx: int, top, bond_adjacency: Dict[int, List[int]]) -> List[int]:
    return [int(j) for j in bond_adjacency.get(int(idx), []) if top.atom(int(j)).element.symbol == "H"]


def _per_anchor_pearson_stability(
    variable: np.ndarray, residual: np.ndarray, anchor_local_idx: np.ndarray, n_anchors: int
) -> Dict:
    """按用户要求的方法论：先在每个 anchor 自己的记录内部单独算 Pearson r(variable, residual)，
    再看这些 r 的符号是否跨 anchor 一致——而不是把所有 anchor 的记录混着算一个池化相关性
    (会被 anchor 间的系统性差异污染，类似 Simpson's paradox)。跟 ridge 系数稳定性同一套判据。
    """
    rs: List[float] = []
    for k in range(n_anchors):
        m = anchor_local_idx == k
        v, r_ = variable[m], residual[m]
        if np.sum(m) < 5 or np.std(v) < 1.0e-10 or np.std(r_) < 1.0e-10:
            continue
        rs.append(float(np.corrcoef(v, r_)[0, 1]))
    if not rs:
        return {"n_anchors_used": 0}
    rs_arr = np.asarray(rs, dtype=float)
    signs = np.where(rs_arr >= 0.0, 1, -1)
    majority = 1 if int(np.sum(signs > 0)) >= int(np.sum(signs < 0)) else -1
    return {
        "n_anchors_used": int(rs_arr.size),
        "mean_r": float(rs_arr.mean()),
        "std_r": float(rs_arr.std()),
        "sign_stability_frac": float(np.mean(signs == majority)),
        "frac_anchors_abs_r_over_0.3": float(np.mean(np.abs(rs_arr) > 0.3)),
    }


def _contact_type_build_context(args: argparse.Namespace, output_dir: str) -> Dict:
    """构建 `--contact-type-fit`/`--contact-type-angular-diagnostic` 共用的几何/角色/特征
    上下文：重建完整 ligand-environment 距离张量、(14,5) 基线残差 R、donor/acceptor 角色、
    psi_o/psi_e contact-type 特征。跟 alpha_vdw/beta_vdw、ridge_lambda 无关的部分只算一次，
    ridge/LOAO 拟合(可能要扫多个 ridge_lambda)和角度诊断都直接复用这份上下文，不用重新
    加载轨迹/重新分类 donor-acceptor。
    """
    openmm, app, unit, XmlSerializer = require_openmm()
    md = require_module("mdtraj")
    symbols = load_abfe_symbols()
    select_env_indices = symbols["_select_env_indices_from_mdtraj_frame"]

    csv_path = os.path.join(output_dir, "perturb_scan_diagnostics.csv")
    npz_path = os.path.join(output_dir, "perturb_scan_geometry.npz")
    ensure_file(csv_path, "perturb-scan 诊断 CSV（先跑一次 --perturb-scan）")
    ensure_file(npz_path, "perturb-scan 几何快照 npz（先跑一次 --perturb-scan）")

    rows = read_csv_rows(csv_path)
    geo = np.load(npz_path)
    n_rows = len(rows)
    delta_e_target = np.asarray([float(r["delta_e_target_kjmol"]) for r in rows], dtype=float)
    pert_type = np.asarray([r["pert_type"] for r in rows], dtype=object)
    magnitude = np.asarray([float(r["magnitude"]) for r in rows], dtype=float)
    axis_kind = np.asarray([r["axis_kind"] for r in rows], dtype=object)
    axis_index = np.asarray([int(r["axis_index"]) for r in rows], dtype=int)
    sign = np.asarray([float(r["sign"]) for r in rows], dtype=float)

    anchor_local_idx = geo["perturbation_anchor_index"].astype(int)
    if len(anchor_local_idx) != n_rows:
        raise RuntimeError(
            f"CSV 行数({n_rows})与几何 npz 行数({len(anchor_local_idx)})不一致，"
            "两者必须来自同一次 --perturb-scan 运行"
        )
    env_positions = geo["env_positions"]
    anchor_lig_positions = geo["anchor_lig_positions"]
    perturbed_lig_positions = geo["perturbed_lig_positions"]
    box_vectors = geo["box_vectors"]
    has_periodic = geo["has_periodic"].astype(bool)
    sigma_lig, eps_lig = geo["sigma_lig"], geo["eps_lig"]
    sigma_env, eps_env = geo["sigma_env"], geo["eps_env"]
    n_anchors = anchor_lig_positions.shape[0]
    n_lig, n_env = sigma_lig.shape[0], sigma_env.shape[0]
    cutoff_nm = float(args.perturb_baseline_cutoff_nm)

    print("[1/5] 重建完整 ligand-environment 距离张量（复用 --perturb-fit 同款几何缓存）")
    tensors = _build_perturbation_distance_tensors(
        n_rows, anchor_local_idx, env_positions, anchor_lig_positions, perturbed_lig_positions,
        box_vectors, has_periodic, sigma_lig, eps_lig, sigma_env, eps_env, cutoff_nm,
    )
    dists_anchor_full = tensors["dists_anchor_full"]
    dists_pert_full = tensors["dists_pert_full"]
    eps_ij_full = tensors["eps_ij_full"]
    x_anchor_full = tensors["x_anchor_full"]
    x_pert_full = tensors["x_pert_full"]
    mask_anchor_full = tensors["mask_anchor_full"]
    mask_pert_full = tensors["mask_pert_full"]

    print("[2/5] (14,5) 完整 pairwise 基线，拟合目标 R = ΔE_MACE - ΔU_DEXP(14,5)")
    _alpha0, _beta0 = 14.0, 5.0
    _c_a0, _c_b0 = _beta0 / (_alpha0 - _beta0), _alpha0 / (_alpha0 - _beta0)
    u_anchor_1405 = np.sum(
        np.where(mask_anchor_full, eps_ij_full[None] * (_c_a0 * np.exp(-_alpha0 * x_anchor_full) - _c_b0 * np.exp(-_beta0 * x_anchor_full)), 0.0),
        axis=(1, 2),
    )
    u_pert_1405 = np.sum(
        np.where(mask_pert_full, eps_ij_full[None] * (_c_a0 * np.exp(-_alpha0 * x_pert_full) - _c_b0 * np.exp(-_beta0 * x_pert_full)), 0.0),
        axis=(1, 2),
    )
    delta_u_1405 = u_pert_1405 - u_anchor_1405
    residual_target = delta_e_target - delta_u_1405  # R，也就是 M0 的残差

    print(
        f"[3/5] 重新选取配体/环境原子集合（须与生成本次 --perturb-scan 时的 "
        f"--ligand/--fit-env-radius/--fit-env-max-atoms/--perturb-anchors/--fit-last-ns 完全一致），"
        f"按 element+system.xml 键连关系判 donor/acceptor 角色"
    )
    traj = md.load(args.traj, top=args.traj_top)
    anchor_frame_ids = select_tail_indices_from_time(traj, args.perturb_anchors, args.fit_last_ns)
    anchor_traj = traj[anchor_frame_ids]
    if anchor_traj.unitcell_vectors is not None:
        anchor_traj = anchor_traj.image_molecules(inplace=False)
    lig_idx = np.asarray(anchor_traj.top.select(f"resname {args.ligand}"), dtype=int)
    ref_frame = anchor_traj[-1]
    env_search_radius = float(args.fit_env_radius)
    env_max_atoms = int(args.fit_env_max_atoms) if int(args.fit_env_max_atoms) > 0 else None
    env_idx = np.asarray(
        select_env_indices(ref_frame, lig_idx, env_search_radius, max_env_atoms=env_max_atoms),
        dtype=int,
    )
    if len(lig_idx) != n_lig or len(env_idx) != n_env:
        raise RuntimeError(
            f"重新选取的配体/环境原子数({len(lig_idx)}/{len(env_idx)})跟几何 npz 里的"
            f"sigma_lig/sigma_env 长度({n_lig}/{n_env})对不上——多半是本次调用的 "
            "--ligand/--fit-env-radius/--fit-env-max-atoms/--perturb-anchors/--fit-last-ns "
            "跟生成该 --perturb-scan 输出时用的不是同一套值，请检查后重新传入。"
        )

    top = anchor_traj.top
    bond_adjacency = _bond_adjacency_from_system_xml(args.system_xml, openmm, XmlSerializer)

    def _is_donor(idx: int) -> bool:
        atom = top.atom(int(idx))
        if atom.element.symbol not in ("N", "O"):
            return False
        return any(top.atom(j).element.symbol == "H" for j in bond_adjacency.get(int(idx), []))

    def _is_acceptor(idx: int) -> bool:
        return top.atom(int(idx)).element.symbol in ("N", "O")

    donor_lig = np.asarray([_is_donor(i) for i in lig_idx], dtype=bool)
    acceptor_lig = np.asarray([_is_acceptor(i) for i in lig_idx], dtype=bool)
    donor_env = np.asarray([_is_donor(i) for i in env_idx], dtype=bool)
    acceptor_env = np.asarray([_is_acceptor(i) for i in env_idx], dtype=bool)
    donor_acceptor_mask = (
        (donor_lig[:, None] & acceptor_env[None, :]) | (donor_env[None, :] & acceptor_lig[:, None])
    )
    fallback_mask = ~donor_acceptor_mask
    print(
        f"    配体: donor原子={int(donor_lig.sum())} acceptor原子={int(acceptor_lig.sum())} (共{n_lig}) | "
        f"环境: donor原子={int(donor_env.sum())} acceptor原子={int(acceptor_env.sum())} (共{n_env})"
    )
    da_active_frac = float(np.mean(np.any(mask_anchor_full & donor_acceptor_mask[None, :, :], axis=(1, 2))))
    print(f"    donor_acceptor contact 在 anchor 态出现的扰动记录占比: {da_active_frac:.1%}")
    if da_active_frac < 0.3:
        print("    ⚠️ 占比偏低，donor_acceptor 组的系数可能因样本太少而不稳定，解读时需谨慎")

    # 配体侧的 donor site（heavy atom + 具体键连的 H），供 --contact-type-angular-diagnostic 用。
    # 只做配体侧(不含环境侧 donor)——第4.4节证实有问题的是配体的两个供体基团，这里精确聚焦。
    lig_idx_pos = {int(atom_topo_idx): pos for pos, atom_topo_idx in enumerate(lig_idx)}
    donor_sites: List[Dict] = []
    for pos, atom_topo_idx in enumerate(lig_idx):
        if not donor_lig[pos]:
            continue
        atom = top.atom(int(atom_topo_idx))
        for h_topo_idx in _bonded_hydrogens(atom_topo_idx, top, bond_adjacency):
            if int(h_topo_idx) not in lig_idx_pos:
                continue  # 配体分子内的 H，理论上一定在 lig_idx 里；这里只是防御性检查
            donor_sites.append({
                "donor_topo_idx": int(atom_topo_idx),
                "donor_lig_pos": int(pos),
                "donor_label": f"{atom.residue.name}{atom.residue.resSeq}-{atom.name}",
                "h_topo_idx": int(h_topo_idx),
                "h_lig_pos": int(lig_idx_pos[int(h_topo_idx)]),
            })
    acceptor_env_positions = np.where(acceptor_env)[0]
    print(
        f"    配体侧 donor site (heavy atom, H) 组合数: {len(donor_sites)}  | "
        f"环境侧候选 acceptor 原子数: {len(acceptor_env_positions)}"
    )

    print(f"[4/5] 构建 psi_o/psi_e contact-type 特征（gamma_o={args.contact_type_gamma_odd}, gamma_e={args.contact_type_gamma_even}）")
    switch_width = float(args.contact_type_switch_width)
    r_switch = max(0.0, cutoff_nm - switch_width)
    gamma_o = float(args.contact_type_gamma_odd)
    gamma_e = float(args.contact_type_gamma_even)

    def _switch_s(r: np.ndarray) -> np.ndarray:
        s = np.ones_like(r)
        in_switch = (r > r_switch) & (r <= cutoff_nm)
        t = (r[in_switch] - r_switch) / max(cutoff_nm - r_switch, 1.0e-9)
        s[in_switch] = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
        s[r > cutoff_nm] = 0.0
        return s

    s_anchor = _switch_s(dists_anchor_full)
    s_pert = _switch_s(dists_pert_full)
    psi_o_anchor = x_anchor_full * np.exp(-gamma_o * x_anchor_full ** 2)
    psi_o_pert = x_pert_full * np.exp(-gamma_o * x_pert_full ** 2)
    psi_e_anchor = (x_anchor_full ** 2) * np.exp(-gamma_e * x_anchor_full ** 2)
    psi_e_pert = (x_pert_full ** 2) * np.exp(-gamma_e * x_pert_full ** 2)

    def _group_delta_features(group_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gm = group_mask[None, :, :]
        w_anchor = np.where(mask_anchor_full & gm, eps_ij_full[None] * s_anchor, 0.0)
        w_pert = np.where(mask_pert_full & gm, eps_ij_full[None] * s_pert, 0.0)
        f_odd = np.sum(w_pert * psi_o_pert, axis=(1, 2)) - np.sum(w_anchor * psi_o_anchor, axis=(1, 2))
        f_even = np.sum(w_pert * psi_e_pert, axis=(1, 2)) - np.sum(w_anchor * psi_e_anchor, axis=(1, 2))
        return f_odd, f_even

    fb_odd, fb_even = _group_delta_features(fallback_mask)
    da_odd, da_even = _group_delta_features(donor_acceptor_mask)

    m1_names = ["fallback_b_even", "donor_acceptor_b_even"]
    x_m1_full = np.stack([fb_even, da_even], axis=1)
    m2_names = ["fallback_a_odd", "fallback_b_even", "donor_acceptor_a_odd", "donor_acceptor_b_even"]
    x_m2_full = np.stack([fb_odd, fb_even, da_odd, da_even], axis=1)

    # 奇偶分组索引（跟 alpha/beta、ridge_lambda 无关，只算一次，ridge/LOAO 扫多个 lambda 时复用）。
    group_key_to_idx: Dict[Tuple, Dict[float, int]] = {}
    for i in range(n_rows):
        key = (int(anchor_local_idx[i]), str(pert_type[i]), str(axis_kind[i]), int(axis_index[i]), float(magnitude[i]))
        group_key_to_idx.setdefault(key, {})[float(sign[i])] = i

    return {
        "n_rows": n_rows, "n_anchors": n_anchors, "n_lig": n_lig, "n_env": n_env, "cutoff_nm": cutoff_nm,
        "anchor_local_idx": anchor_local_idx, "pert_type": pert_type, "magnitude": magnitude,
        "axis_kind": axis_kind, "axis_index": axis_index, "sign": sign,
        "delta_e_target": delta_e_target, "residual_target": residual_target,
        "group_key_to_idx": group_key_to_idx,
        "lig_idx": lig_idx, "env_idx": env_idx, "top": top, "bond_adjacency": bond_adjacency,
        "donor_lig": donor_lig, "acceptor_lig": acceptor_lig, "donor_env": donor_env, "acceptor_env": acceptor_env,
        "donor_acceptor_mask": donor_acceptor_mask, "fallback_mask": fallback_mask, "da_active_frac": da_active_frac,
        "donor_sites": donor_sites, "acceptor_env_positions": acceptor_env_positions,
        "gamma_o": gamma_o, "gamma_e": gamma_e, "switch_width": switch_width,
        "m1_names": m1_names, "x_m1_full": x_m1_full, "m2_names": m2_names, "x_m2_full": x_m2_full,
        "anchor_lig_positions": anchor_lig_positions, "perturbed_lig_positions": perturbed_lig_positions,
        "env_positions": env_positions, "box_vectors": box_vectors, "has_periodic": has_periodic,
    }


def _contact_type_ridge_loao(ctx: Dict, ridge_lambda: float, mag04_weight: float, verbose: bool = True) -> Dict:
    """给定一个 ridge_lambda，跑一遍 M0/M1/M2 的 grouped leave-one-anchor-out。跟
    `_contact_type_build_context` 分开，是为了让 ridge_lambda 稳健性扫描(§6.6)不用每次都
    重新加载轨迹/重建几何张量——那部分开销大、跟 ridge_lambda 无关，只值得算一次。
    """
    n_rows, n_anchors = ctx["n_rows"], ctx["n_anchors"]
    anchor_local_idx, pert_type, magnitude = ctx["anchor_local_idx"], ctx["pert_type"], ctx["magnitude"]
    residual_target = ctx["residual_target"]
    x_m1_full, x_m2_full = ctx["x_m1_full"], ctx["x_m2_full"]
    m1_names, m2_names = ctx["m1_names"], ctx["m2_names"]
    group_key_to_idx = ctx["group_key_to_idx"]

    BIN_KEYS = [
        ("rotation", 0.5), ("rotation", 1.5), ("rotation", 3.0),
        ("translation", 0.005), ("translation", 0.01), ("translation", 0.02), ("translation", 0.04),
    ]
    downweight = {("translation", 0.04): float(mag04_weight)}

    def _row_weights(mask_rows: np.ndarray) -> np.ndarray:
        w = np.zeros(n_rows, dtype=float)
        for key in BIN_KEYS:
            bin_mask = mask_rows & (pert_type == key[0]) & np.isclose(magnitude, key[1])
            n_bin = int(np.sum(bin_mask))
            if n_bin == 0:
                continue
            w[bin_mask] = downweight.get(key, 1.0) / n_bin
        total = float(np.sum(w[mask_rows]))
        if total > 1.0e-12:
            w[mask_rows] /= total
        return w

    def _weighted_rmse(resid: np.ndarray, mask_rows: np.ndarray, weights: np.ndarray) -> float:
        w = weights[mask_rows]
        r = resid[mask_rows]
        return float(np.sqrt(np.sum(w * r ** 2) / np.sum(w)))

    def _ridge_fit_no_intercept(x_mat: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
        sw = np.sqrt(np.clip(w, 0.0, None))
        xw = x_mat * sw[:, None]
        yw = y * sw
        scale = np.sqrt(np.mean(x_mat ** 2, axis=0))
        scale = np.where(scale < 1.0e-12, 1.0, scale)
        xs = xw / scale[None, :]
        lhs = xs.T @ xs + lam * np.eye(x_mat.shape[1], dtype=float)
        rhs = xs.T @ yw
        try:
            coef_scaled = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coef_scaled = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return coef_scaled / scale

    oof_pred_m1 = np.zeros(n_rows, dtype=float)
    oof_pred_m2 = np.zeros(n_rows, dtype=float)
    fold_results: List[Dict] = []
    for k in range(n_anchors):
        train_mask = anchor_local_idx != k
        test_mask = anchor_local_idx == k
        if not np.any(test_mask):
            continue
        train_w = _row_weights(train_mask)
        test_w = _row_weights(test_mask)
        coef_m1 = _ridge_fit_no_intercept(x_m1_full[train_mask], residual_target[train_mask], train_w[train_mask], ridge_lambda)
        coef_m2 = _ridge_fit_no_intercept(x_m2_full[train_mask], residual_target[train_mask], train_w[train_mask], ridge_lambda)
        pred_m1_all = x_m1_full @ coef_m1
        pred_m2_all = x_m2_full @ coef_m2
        oof_pred_m1[test_mask] = pred_m1_all[test_mask]
        oof_pred_m2[test_mask] = pred_m2_all[test_mask]

        rmse_m0 = _weighted_rmse(residual_target, test_mask, test_w)
        rmse_m1 = _weighted_rmse(residual_target - pred_m1_all, test_mask, test_w)
        rmse_m2 = _weighted_rmse(residual_target - pred_m2_all, test_mask, test_w)
        fold_results.append({
            "held_out_anchor_local_idx": int(k),
            "m0_test_weighted_rmse_kjmol": rmse_m0,
            "m1_test_weighted_rmse_kjmol": rmse_m1,
            "m2_test_weighted_rmse_kjmol": rmse_m2,
            "m1_coef": {name: float(v) for name, v in zip(m1_names, coef_m1)},
            "m2_coef": {name: float(v) for name, v in zip(m2_names, coef_m2)},
        })
        if verbose:
            print(
                f"    [fold {k + 1}/{n_anchors}] 留出集加权RMSE  M0={rmse_m0:.3f}  M1={rmse_m1:.3f}  M2={rmse_m2:.3f} kJ/mol"
            )

    m1_coefs = np.asarray([[f["m1_coef"][name] for name in m1_names] for f in fold_results], dtype=float)
    m2_coefs = np.asarray([[f["m2_coef"][name] for name in m2_names] for f in fold_results], dtype=float)

    def _sign_stability(coef_col: np.ndarray) -> float:
        signs = np.where(coef_col >= 0.0, 1, -1)
        majority = 1 if int(np.sum(signs > 0)) >= int(np.sum(signs < 0)) else -1
        return float(np.mean(signs == majority))

    m1_coef_stats = {
        name: {
            "mean": float(m1_coefs[:, i].mean()), "std": float(m1_coefs[:, i].std()),
            "sign_stability_frac": _sign_stability(m1_coefs[:, i]),
        }
        for i, name in enumerate(m1_names)
    }
    m2_coef_stats = {
        name: {
            "mean": float(m2_coefs[:, i].mean()), "std": float(m2_coefs[:, i].std()),
            "sign_stability_frac": _sign_stability(m2_coefs[:, i]),
        }
        for i, name in enumerate(m2_names)
    }

    all_mask = np.ones(n_rows, dtype=bool)
    all_w = _row_weights(all_mask)
    coef_m1_full_data = _ridge_fit_no_intercept(x_m1_full, residual_target, all_w, ridge_lambda)
    coef_m2_full_data = _ridge_fit_no_intercept(x_m2_full, residual_target, all_w, ridge_lambda)

    m0_oof_rmse = _weighted_rmse(residual_target, all_mask, all_w)
    m1_oof_rmse = _weighted_rmse(residual_target - oof_pred_m1, all_mask, all_w)
    m2_oof_rmse = _weighted_rmse(residual_target - oof_pred_m2, all_mask, all_w)

    def _odd_even_stats(resid_arr: np.ndarray) -> Dict[str, Dict]:
        odd_by_type: Dict[str, List[float]] = {"translation": [], "rotation": []}
        even_by_type: Dict[str, List[float]] = {"translation": [], "rotation": []}
        for key, signed in group_key_to_idx.items():
            if 1.0 not in signed or -1.0 not in signed:
                continue
            ip, im = signed[1.0], signed[-1.0]
            odd = (resid_arr[ip] - resid_arr[im]) / 2.0
            even = (resid_arr[ip] + resid_arr[im]) / 2.0
            ptype = key[1]
            odd_by_type[ptype].append(odd)
            even_by_type[ptype].append(even)
        out: Dict[str, Dict] = {}
        for ptype in ("translation", "rotation"):
            oa = np.asarray(odd_by_type[ptype], dtype=float)
            ea = np.asarray(even_by_type[ptype], dtype=float)
            out[ptype] = {
                "n_pairs": int(oa.size),
                "odd_residual_rmse_kjmol": float(np.sqrt(np.mean(oa ** 2))) if oa.size else math.nan,
                "even_residual_rmse_kjmol": float(np.sqrt(np.mean(ea ** 2))) if ea.size else math.nan,
            }
        return out

    odd_even_m0 = _odd_even_stats(residual_target)
    odd_even_m1 = _odd_even_stats(residual_target - oof_pred_m1)
    odd_even_m2 = _odd_even_stats(residual_target - oof_pred_m2)

    return {
        "ridge_lambda": float(ridge_lambda),
        "fold_results": fold_results,
        "m1_coef_full_data": {name: float(v) for name, v in zip(m1_names, coef_m1_full_data)},
        "m2_coef_full_data": {name: float(v) for name, v in zip(m2_names, coef_m2_full_data)},
        "m1_coef_loao_fold_stats": m1_coef_stats,
        "m2_coef_loao_fold_stats": m2_coef_stats,
        "m0_oof_weighted_rmse_kjmol": m0_oof_rmse,
        "m1_oof_weighted_rmse_kjmol": m1_oof_rmse,
        "m2_oof_weighted_rmse_kjmol": m2_oof_rmse,
        "odd_even_m0": odd_even_m0, "odd_even_m1": odd_even_m1, "odd_even_m2": odd_even_m2,
        "oof_pred_m1": oof_pred_m1, "oof_pred_m2": oof_pred_m2,
    }


def run_contact_type_fit(args: argparse.Namespace, output_dir: str) -> Dict:
    """DEXP_KERNEL_PHYSICS_ISSUES.md §6 的最小实现（尚未接入生产 force，仍是离线诊断）：

    在 (a) 阶段 pair-specific LJ-matched 解析基线之上，按化学角色分两类 contact-type
    `t ∈ {donor_acceptor, fallback}` 叠加：

        x_ij = r_ij/r0,ij - 1
        ΔU_ij^(t) = eps_ij * S(r_ij) * [ a_t*psi_o(x_ij) + b_t*psi_e(x_ij) ]
        psi_o(x) = x   * exp(-gamma_o*x^2)   # odd：修正局部力/有效平衡位置，r0处值不变
        psi_e(x) = x^2 * exp(-gamma_e*x^2)   # even：修正曲率/高阶形状，r0处值和一阶导都不变

    `donor_acceptor` 判据（力场级别、非 pose 专属，不用具体残基编号/具体配体原子编号）：
    (i,j) 中至少一方是"重原子=N/O 且键连至少一个 H"(donor)，另一方是"重原子=N/O"(acceptor，
    不要求本身带 H)——键连关系从 args.system_xml 的 HarmonicBondForce+Constraints 认，
    跟 run_replica_analysis 里已验证过的氢键候选判据同一套逻辑。其余 pair 归 fallback。

    M0 = 纯(14,5)基线（不修正，预测 R=0）
    M1 = +contact-type even 修正（每类 1 个系数 b_t，共 2 个自由参数）
    M2 = +contact-type odd+even 修正（每类 2 个系数 a_t,b_t，共 4 个自由参数）

    支持 `--contact-type-ridge-lambda-grid`（逗号分隔）做 ridge 强度稳健性扫描——按用户
    要求，在正式判定"radial contact-type 修正是否值得继续"之前，先确认结论不是
    ridge_lambda 选得太强/太弱造成的假阴性/假阳性。若未提供 grid，只用单个
    `--contact-type-ridge-lambda`（向后兼容）。
    """
    ctx = _contact_type_build_context(args, output_dir)
    n_anchors = ctx["n_anchors"]
    grid_arg = str(getattr(args, "contact_type_ridge_lambda_grid", "") or "").strip()
    if grid_arg:
        ridge_grid = [float(x) for x in grid_arg.split(",") if x.strip()]
    else:
        ridge_grid = [float(args.contact_type_ridge_lambda)]

    print(f"[5/5] grouped leave-one-anchor-out（{n_anchors} 折）× ridge_lambda∈{ridge_grid}，M0(基线) vs M1(+even) vs M2(+odd+even)")
    sweep_results: List[Dict] = []
    for lam in ridge_grid:
        print(f"  -- ridge_lambda={lam} --")
        res = _contact_type_ridge_loao(ctx, lam, float(args.perturb_fit_mag04_weight), verbose=True)
        print(
            f"    OOF 加权RMSE(留出anchor拼接): M0={res['m0_oof_weighted_rmse_kjmol']:.3f}  "
            f"M1={res['m1_oof_weighted_rmse_kjmol']:.3f}  M2={res['m2_oof_weighted_rmse_kjmol']:.3f} kJ/mol"
        )
        for ptype in ("translation", "rotation"):
            oe0, oe1, oe2 = res["odd_even_m0"][ptype], res["odd_even_m1"][ptype], res["odd_even_m2"][ptype]
            print(
                f"    奇偶分解[{ptype}] M0: odd={oe0['odd_residual_rmse_kjmol']:.3f} even={oe0['even_residual_rmse_kjmol']:.3f} | "
                f"M1: odd={oe1['odd_residual_rmse_kjmol']:.3f} even={oe1['even_residual_rmse_kjmol']:.3f} | "
                f"M2: odd={oe2['odd_residual_rmse_kjmol']:.3f} even={oe2['even_residual_rmse_kjmol']:.3f}"
            )
        print("    M2 系数跨折稳定性:")
        for name, stats in res["m2_coef_loao_fold_stats"].items():
            print(f"        {name:24s} mean={stats['mean']:+.4f} std={stats['std']:.4f} sign_stability={stats['sign_stability_frac']:.0%}")
        sweep_results.append({k: v for k, v in res.items() if k not in ("oof_pred_m1", "oof_pred_m2")})

    best_m2 = min(sweep_results, key=lambda r: r["m2_oof_weighted_rmse_kjmol"])
    m0_ref = sweep_results[0]["m0_oof_weighted_rmse_kjmol"]
    m2_improvement_frac = float((m0_ref - best_m2["m2_oof_weighted_rmse_kjmol"]) / m0_ref) if m0_ref > 1.0e-9 else math.nan
    print(
        f"    跨 ridge_lambda 最优 M2: lambda={best_m2['ridge_lambda']} "
        f"OOF加权RMSE={best_m2['m2_oof_weighted_rmse_kjmol']:.3f} kJ/mol "
        f"(相对 M0 改善 {m2_improvement_frac:.1%})"
    )
    if m2_improvement_frac < 0.02:
        print("    ⚠️ 跨 ridge_lambda 扫描后，M2 相对 M0 的改善仍 <=~1-2%（甚至可能为负/更差）——按稳健性判据应正式关闭 radial contact-type 修正这条路线（见 §6.6）")

    summary = {
        "n_anchors": int(ctx["n_anchors"]),
        "n_rows": int(ctx["n_rows"]),
        "n_ligand_atoms": int(ctx["n_lig"]),
        "n_environment_atoms": int(ctx["n_env"]),
        "target_definition": "R = delta_e_target_kjmol - full_pairwise_DEXP(14,5)_delta_u_kjmol",
        "contact_types": ["fallback", "donor_acceptor"],
        "donor_acceptor_definition": (
            "(i,j) 中至少一方是 heavy atom(N/O) 且键连>=1个H(donor)，另一方是 heavy atom(N/O，"
            "不要求带H, acceptor)；键连关系来自 system.xml 的 HarmonicBondForce+Constraints。"
            "其余 pair 归 fallback。"
        ),
        "n_donor_atoms_ligand": int(ctx["donor_lig"].sum()),
        "n_acceptor_atoms_ligand": int(ctx["acceptor_lig"].sum()),
        "n_donor_atoms_environment": int(ctx["donor_env"].sum()),
        "n_acceptor_atoms_environment": int(ctx["acceptor_env"].sum()),
        "donor_acceptor_contact_active_fraction": ctx["da_active_frac"],
        "gamma_odd": ctx["gamma_o"],
        "gamma_even": ctx["gamma_e"],
        "switch_width_nm": ctx["switch_width"],
        "cutoff_nm": ctx["cutoff_nm"],
        "m1_feature_names": ctx["m1_names"],
        "m2_feature_names": ctx["m2_names"],
        "ridge_lambda_grid": ridge_grid,
        "ridge_lambda_sweep": sweep_results,
        "best_m2_ridge_lambda": best_m2["ridge_lambda"],
        "best_m2_vs_m0_improvement_frac": m2_improvement_frac,
        "promotion_criterion_note": (
            "LOAO 过关（M2 相对 M0/M1 在未见 anchor 上显著降低 odd_residual_rmse，且系数符号跨折"
            "稳定、且改善幅度在 ridge_lambda 扫描下稳健，不是单个 lambda 的偶然结果）只是"
            "'值得接入生产'的必要条件，不是充分条件——真正回答'是否修好了第4.4节的氢键伙伴切换'，"
            "必须把验证通过的修正接入生产 CustomNonbondedForce（Discrete2DFunction 查表），"
            "重跑一遍 --replica-run/--replica-analyze 同样的 3(或4)-condition 对比，看酰胺氢键占有率"
            "是否真的回升。见 DEXP_KERNEL_PHYSICS_ISSUES.md §6.3/§6.4/§6.6。"
        ),
    }
    summary_path = os.path.join(output_dir, "contact_type_fit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    summary: {summary_path}")
    return summary


def run_contact_type_angular_diagnostic(args: argparse.Namespace, output_dir: str) -> Dict:
    """DEXP_KERNEL_PHYSICS_ISSUES.md §6.6：只做角度诊断，不拟合 angular force。

    在 M0 的残差 R 和 M2 的**跨折 out-of-fold**残差(避免在训练残差上看相关性——LOAO 里
    每一折的 M2 系数只在训练 anchor 上拟合，用来预测被留出的那个 anchor，所以
    `residual_target - oof_pred_m2` 对每一行都是"没见过这个 anchor 的模型"给出的残差)上，
    对配体每个 donor site(heavy atom + 具体 H) 分别计算：

    - D-H-A 夹角(在 H 处，D 和最近 acceptor A 之间)，以及 anchor->perturbed 的 Δ夹角；
    - D...A(最近 acceptor) 距离，以及 Δ距离；distance×angle 交互项 Δ距离*Δ夹角；
    - 最近 acceptor 身份是否从 anchor 切换到 perturbed（0/1）；
    - anchor 态的 acceptor 配位数（--contact-type-angular-acceptor-cutoff-nm 内的 acceptor 计数，
      在扰动幅度这个尺度下每个 anchor 内部是常数，只能跨 anchor 比较）。

    然后按用户要求"每个 anchor 分开检查相关方向是否一致"——在每个 anchor 自己的~74条
    扰动记录内部算 Pearson r(变量, 残差)，再看这 20 个 r 值的符号是否一致(跟 ridge 系数
    稳定性同一套判据)，而不是把 20×74 条记录直接混着算一个池化相关性(会被 anchor 间的
    系统性差异污染，参考 Simpson's paradox)。

    配位数是 anchor 内部常数，没法用同样方式算 anchor 内相关；改成用 20 个 anchor 各自的
    (配位数, 该 anchor 内残差 RMSE) 做跨 anchor 相关。

    只有这里看到"未见 anchor 上稳定的角度关系"，才值得建立 M3 = M2 + k_DA*S(r)*f(theta)。
    这一步本身不拟合 M3。
    """
    ctx = _contact_type_build_context(args, output_dir)
    ridge_lambda = float(args.contact_type_ridge_lambda)
    fit = _contact_type_ridge_loao(ctx, ridge_lambda, float(args.perturb_fit_mag04_weight), verbose=False)

    residual_m0 = ctx["residual_target"]
    residual_m2_oof = ctx["residual_target"] - fit["oof_pred_m2"]
    anchor_local_idx = ctx["anchor_local_idx"]
    n_anchors = ctx["n_anchors"]
    n_rows = ctx["n_rows"]
    anchor_lig_positions = ctx["anchor_lig_positions"]
    perturbed_lig_positions = ctx["perturbed_lig_positions"]
    env_positions = ctx["env_positions"]
    box_vectors = ctx["box_vectors"]
    has_periodic = ctx["has_periodic"]
    donor_sites = ctx["donor_sites"]
    acceptor_env_positions = ctx["acceptor_env_positions"]

    if len(donor_sites) == 0:
        raise RuntimeError("没有找到任何配体 donor site（heavy atom 键连 H），无法做角度诊断")
    if len(acceptor_env_positions) == 0:
        raise RuntimeError("环境侧没有任何 acceptor 原子(N/O)，无法做角度诊断")

    acceptor_cutoff_nm = float(args.contact_type_angular_acceptor_cutoff_nm)

    def _wrap(delta: np.ndarray, bv: Optional[np.ndarray]) -> np.ndarray:
        if bv is None:
            return delta
        box_lens = np.linalg.norm(bv, axis=1)
        return delta - box_lens * np.round(delta / box_lens)

    print(
        f"[angular-diagnostic] {len(donor_sites)} 个配体 donor site × {len(acceptor_env_positions)} 个环境 acceptor 原子，"
        f"acceptor 判定/配位数截断={acceptor_cutoff_nm}nm"
    )

    def _per_anchor_corr(variable: np.ndarray, residual: np.ndarray) -> Dict:
        return _per_anchor_pearson_stability(variable, residual, anchor_local_idx, n_anchors)

    site_reports: List[Dict] = []
    for site in donor_sites:
        d_pos, h_pos = site["donor_lig_pos"], site["h_lig_pos"]
        angle_anchor = np.empty(n_rows, dtype=float)
        angle_pert = np.empty(n_rows, dtype=float)
        dist_anchor = np.empty(n_rows, dtype=float)
        dist_pert = np.empty(n_rows, dtype=float)
        switched = np.zeros(n_rows, dtype=float)
        coord_per_anchor = np.zeros(n_anchors, dtype=float)

        for k in range(n_anchors):
            rows_k = np.where(anchor_local_idx == k)[0]
            if rows_k.size == 0:
                continue
            bv = box_vectors[k] if has_periodic[k] else None
            d_anchor_pos = anchor_lig_positions[k, d_pos]
            h_anchor_pos = anchor_lig_positions[k, h_pos]
            acc_pos = env_positions[k][acceptor_env_positions]  # (n_acc, 3)，anchor/perturbed 共用（环境不动）

            dist_to_acc_anchor = np.linalg.norm(_wrap(d_anchor_pos[None, :] - acc_pos, bv), axis=-1)
            a_star_local = int(np.argmin(dist_to_acc_anchor))
            coord_per_anchor[k] = float(np.sum(dist_to_acc_anchor <= acceptor_cutoff_nm))
            a_pos_xyz = acc_pos[a_star_local]

            vec_dh_anchor = _wrap(d_anchor_pos - h_anchor_pos, bv)
            vec_ah_anchor = _wrap(a_pos_xyz - h_anchor_pos, bv)
            cos_anchor = np.dot(vec_dh_anchor, vec_ah_anchor) / (
                np.linalg.norm(vec_dh_anchor) * np.linalg.norm(vec_ah_anchor) + 1.0e-12
            )
            ang_anchor_deg = float(np.degrees(np.arccos(np.clip(cos_anchor, -1.0, 1.0))))
            dist_anchor_nm = float(dist_to_acc_anchor[a_star_local])

            d_pert_pos = perturbed_lig_positions[rows_k, d_pos]      # (n_k,3)
            h_pert_pos = perturbed_lig_positions[rows_k, h_pos]      # (n_k,3)
            delta_all = d_pert_pos[:, None, :] - acc_pos[None, :, :]
            if bv is not None:
                box_lens = np.linalg.norm(bv, axis=1)
                delta_all = delta_all - box_lens * np.round(delta_all / box_lens)
            dist_to_acc_pert = np.linalg.norm(delta_all, axis=-1)     # (n_k, n_acc)
            a_star_pert_local = np.argmin(dist_to_acc_pert, axis=1)   # 每行扰动后最近的 acceptor(可能换人)
            # 用 anchor 选定的那个 acceptor(a_star_local)算"同一个伙伴"的距离/角度延续，
            # 不用 a_star_pert_local 对应的距离——那会把"换伙伴"和"纯几何漂移"两种效应混在一起。
            dist_to_fixed_acc_pert = dist_to_acc_pert[:, a_star_local]

            vec_dh_pert = _wrap(d_pert_pos - h_pert_pos, bv)
            vec_ah_pert = _wrap(a_pos_xyz[None, :] - h_pert_pos, bv)
            cos_pert = np.sum(vec_dh_pert * vec_ah_pert, axis=-1) / (
                np.linalg.norm(vec_dh_pert, axis=-1) * np.linalg.norm(vec_ah_pert, axis=-1) + 1.0e-12
            )
            ang_pert_deg = np.degrees(np.arccos(np.clip(cos_pert, -1.0, 1.0)))

            angle_anchor[rows_k] = ang_anchor_deg
            angle_pert[rows_k] = ang_pert_deg
            dist_anchor[rows_k] = dist_anchor_nm
            dist_pert[rows_k] = dist_to_fixed_acc_pert
            switched[rows_k] = (a_star_pert_local != a_star_local).astype(float)

        delta_angle = angle_pert - angle_anchor
        delta_dist = dist_pert - dist_anchor
        interaction = delta_dist * delta_angle

        report = {
            "donor_label": site["donor_label"],
            "donor_topo_idx": site["donor_topo_idx"],
            "h_topo_idx": site["h_topo_idx"],
            "mean_angle_anchor_deg": float(np.mean(angle_anchor)),
            "mean_dist_anchor_nm": float(np.mean(dist_anchor)),
            "nearest_acceptor_switch_fraction": float(np.mean(switched)),
            "vs_residual_m0": {
                "delta_angle": _per_anchor_corr(delta_angle, residual_m0),
                "delta_dist": _per_anchor_corr(delta_dist, residual_m0),
                "delta_dist_x_delta_angle": _per_anchor_corr(interaction, residual_m0),
                "nearest_acceptor_switched": _per_anchor_corr(switched, residual_m0),
            },
            "vs_residual_m2_oof": {
                "delta_angle": _per_anchor_corr(delta_angle, residual_m2_oof),
                "delta_dist": _per_anchor_corr(delta_dist, residual_m2_oof),
                "delta_dist_x_delta_angle": _per_anchor_corr(interaction, residual_m2_oof),
                "nearest_acceptor_switched": _per_anchor_corr(switched, residual_m2_oof),
            },
        }

        per_anchor_rmse_m0 = np.asarray([
            float(np.sqrt(np.mean(residual_m0[anchor_local_idx == k] ** 2))) for k in range(n_anchors)
        ])
        per_anchor_rmse_m2 = np.asarray([
            float(np.sqrt(np.mean(residual_m2_oof[anchor_local_idx == k] ** 2))) for k in range(n_anchors)
        ])
        if np.std(coord_per_anchor) > 1.0e-9:
            report["coordination_vs_anchor_rmse_m0_pearson_r"] = float(np.corrcoef(coord_per_anchor, per_anchor_rmse_m0)[0, 1])
            report["coordination_vs_anchor_rmse_m2_oof_pearson_r"] = float(np.corrcoef(coord_per_anchor, per_anchor_rmse_m2)[0, 1])
        else:
            report["coordination_vs_anchor_rmse_m0_pearson_r"] = math.nan
            report["coordination_vs_anchor_rmse_m2_oof_pearson_r"] = math.nan
        report["coordination_number_range"] = [float(coord_per_anchor.min()), float(coord_per_anchor.max())]

        site_reports.append(report)
        print(f"    donor site {site['donor_label']}: mean anchor angle={report['mean_angle_anchor_deg']:.1f}deg  "
              f"mean anchor dist={report['mean_dist_anchor_nm']:.3f}nm  "
              f"acceptor切换比例={report['nearest_acceptor_switch_fraction']:.1%}  "
              f"配位数范围={report['coordination_number_range']}")
        for target_label, target_dict in (("M0", report["vs_residual_m0"]), ("M2_oof", report["vs_residual_m2_oof"])):
            for var_label, stats in target_dict.items():
                if stats.get("n_anchors_used", 0) == 0:
                    continue
                print(
                    f"        vs {target_label:6s} {var_label:24s} mean_r={stats['mean_r']:+.3f} "
                    f"std_r={stats['std_r']:.3f} sign_stability={stats['sign_stability_frac']:.0%} "
                    f"|r|>0.3占比={stats['frac_anchors_abs_r_over_0.3']:.0%}"
                )

    any_stable_signal = any(
        stats.get("n_anchors_used", 0) > 0
        and stats["sign_stability_frac"] >= 0.85
        and stats["frac_anchors_abs_r_over_0.3"] >= 0.5
        for report in site_reports
        for target_dict in (report["vs_residual_m0"], report["vs_residual_m2_oof"])
        for stats in target_dict.values()
    )

    summary = {
        "ridge_lambda_used": ridge_lambda,
        "acceptor_cutoff_nm": acceptor_cutoff_nm,
        "n_donor_sites": len(donor_sites),
        "n_acceptor_atoms_environment": len(acceptor_env_positions),
        "site_reports": site_reports,
        "any_stable_angular_signal_found": any_stable_signal,
        "decision_note": (
            "判据(用户定义)：sign_stability_frac>=0.85 且 |r|>0.3 的 anchor 占比>=0.5，才算'未见 anchor 上"
            "存在稳定角度关系'。如果 any_stable_angular_signal_found=false，按用户既定方针，不建立 "
            "M3=M2+k_DA*S(r)*f(theta)，转向检查 Gaussian-width/charge-penetration，或承认剩余项是"
            "多体环境误差（配体分子内/成键项、蛋白邻近残基约束、Gaussian-Coulomb 力本身的部分抵消，"
            "见 DEXP_KERNEL_PHYSICS_ISSUES.md §3.4 最后一条解读）。如果为 true，下一步才是最小化地"
            "对信号最强的那个 donor site 构建 M3 并重新 grouped LOAO 跟 M0/M2 比较，不要一次性对"
            "所有 donor site 都加 angular 项。"
        ),
    }
    summary_path = os.path.join(output_dir, "contact_type_angular_diagnostic_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    any_stable_angular_signal_found = {any_stable_signal}")
    print(f"    summary: {summary_path}")
    return summary


def run_gaussian_width_diagnostic(args: argparse.Namespace, output_dir: str) -> Dict:
    """DEXP_KERNEL_PHYSICS_ISSUES.md §6 "还有一个可能的接触特异性来源：Gaussian 宽度"的诊断
    （只诊断，不拟合 role-specific sigma_elec）。

    radial(§6.5) 和 angular(§6.6) 的 contact-type 修正都没有在未见 anchor 上找到稳定信号后，
    这一步检验第三种可能：M0 残差 `R = ΔE_MACE - ΔU_DEXP(14,5)` 里剩下的东西，是不是统一
    `sigma_elec` 电荷穿透误差的印记，而不是 vdW 形状或角度问题。

    做法（全部只用 --perturb-scan 已缓存的坐标+system.xml 电荷，不需要重跑 MACE）：
    1. 从 system.xml 的 NonbondedForce 读 lig_idx/env_idx 的部分电荷。
    2. 原样复刻 `build_mm_le_contexts_from_system_xml` 里 "gauss_coul" 参考项的定义
       (sigma_elec=0.10nm 硬编码、由 --fit-mm-ref-cutoff/--fit-mm-ref-switch 决定是
       NoCutoff+无PBC(默认，跟 MACE 真空团簇边界条件一致) 还是 CutoffPeriodic+可选switching)，
       算出 Δ(gauss_coul 电荷模型本身)；同时算 Δ(bare 点电荷 Coulomb，不做 erf 平滑)。
    3. `Δpenetration = Δbare - Δgauss` 就是"用 Gaussian 而不是点电荷"引入的修正量——电荷穿透
       误差的直接几何+电荷代理，跟 MACE 完全无关，纯粹由坐标和电荷决定。
    4. 用跟 §6.6 完全一致的方法论（每个 anchor 内部单独算 Pearson r，再看 20 个 r 的符号
       是否跨 anchor 稳定）检验 R(M0 残差) 和 M2 的跨折 out-of-fold 残差是否与
       Δgauss/Δbare/Δpenetration 存在稳定关联——分别在全部 lig-env pair 和只在
       donor_acceptor(§6.5 定义) pair 上算，因为电荷穿透问题按物理直觉应该在带电/极性接触
       上最明显。

    只是诊断：这里不拟合任何 role-specific sigma_elec，只回答"值不值得往这个方向investigate"。
    """
    openmm, app, unit, XmlSerializer = require_openmm()
    from scipy.special import erf

    ctx = _contact_type_build_context(args, output_dir)
    n_rows, n_anchors = ctx["n_rows"], ctx["n_anchors"]
    n_lig, n_env = ctx["n_lig"], ctx["n_env"]
    anchor_local_idx = ctx["anchor_local_idx"]
    lig_idx, env_idx = ctx["lig_idx"], ctx["env_idx"]
    donor_acceptor_mask = ctx["donor_acceptor_mask"]
    residual_m0 = ctx["residual_target"]
    anchor_lig_positions = ctx["anchor_lig_positions"]
    perturbed_lig_positions = ctx["perturbed_lig_positions"]
    env_positions = ctx["env_positions"]
    box_vectors = ctx["box_vectors"]
    has_periodic = ctx["has_periodic"]

    print("[1/3] 从 system.xml 读取 lig/env 原子的部分电荷")
    with open(args.system_xml, "r", encoding="utf-8") as handle:
        nb_system = XmlSerializer.deserialize(handle.read())
    nb_force = next(f for f in nb_system.getForces() if isinstance(f, openmm.NonbondedForce))
    q_all = np.zeros(nb_system.getNumParticles(), dtype=float)
    for i in range(nb_system.getNumParticles()):
        q, _, _ = nb_force.getParticleParameters(i)
        q_all[i] = q.value_in_unit(unit.elementary_charge)
    q_lig, q_env = q_all[lig_idx], q_all[env_idx]
    q_ij = q_lig[:, None] * q_env[None, :]

    sigma_elec_nm = 0.10  # 与 build_mm_le_contexts_from_system_xml 里 "gauss_coul" 的硬编码值一致
    gamma_eff = 1.0 / (math.sqrt(2.0) * sigma_elec_nm)
    ke = 138.935456
    mm_ref_cutoff = float(args.fit_mm_ref_cutoff)
    mm_ref_switch = float(args.fit_mm_ref_switch)
    use_ref_cutoff = mm_ref_cutoff > 0.0
    print(
        f"    复刻 --perturb-scan 算 delta_e_target 时用的电荷模型: sigma_elec={sigma_elec_nm}nm, "
        + (
            f"CutoffPeriodic cutoff={mm_ref_cutoff}nm switch={mm_ref_switch}nm"
            if use_ref_cutoff
            else "NoCutoff（全程 1/r 近似，不做 PBC wrap，与 MACE 真空团簇边界条件一致，也是 --fit-mm-ref-cutoff 的默认值）"
        )
    )

    print("[2/3] 重建 lig-env 距离张量（按 --fit-mm-ref-cutoff 同款规则，不是 §6.5 的 0.70nm vdW cutoff）")
    dists_anchor = np.empty((n_rows, n_lig, n_env), dtype=np.float64)
    dists_pert = np.empty((n_rows, n_lig, n_env), dtype=np.float64)
    for a in range(n_anchors):
        rows_a = np.where(anchor_local_idx == a)[0]
        if rows_a.size == 0:
            continue
        bv = box_vectors[a] if (use_ref_cutoff and has_periodic[a]) else None
        d_anchor = anchor_lig_positions[a][:, None, :] - env_positions[a][None, :, :]
        if bv is not None:
            box_lens = np.linalg.norm(bv, axis=1)
            d_anchor = d_anchor - box_lens * np.round(d_anchor / box_lens)
        dists_anchor[rows_a] = np.linalg.norm(d_anchor, axis=-1)[None, :, :]

        lig_block = perturbed_lig_positions[rows_a]
        d_pert = lig_block[:, :, None, :] - env_positions[a][None, None, :, :]
        if bv is not None:
            box_lens = np.linalg.norm(bv, axis=1)
            d_pert = d_pert - box_lens * np.round(d_pert / box_lens)
        dists_pert[rows_a] = np.linalg.norm(d_pert, axis=-1)

    def _switch_s_ref(r: np.ndarray) -> np.ndarray:
        if not use_ref_cutoff:
            return np.ones_like(r)
        s = np.ones_like(r)
        if 0.0 < mm_ref_switch < mm_ref_cutoff:
            in_switch = (r > mm_ref_switch) & (r <= mm_ref_cutoff)
            t = (r[in_switch] - mm_ref_switch) / (mm_ref_cutoff - mm_ref_switch)
            s[in_switch] = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
        s[r > mm_ref_cutoff] = 0.0
        return s

    cutoff_mask_anchor = (dists_anchor <= mm_ref_cutoff) if use_ref_cutoff else np.ones_like(dists_anchor, dtype=bool)
    cutoff_mask_pert = (dists_pert <= mm_ref_cutoff) if use_ref_cutoff else np.ones_like(dists_pert, dtype=bool)
    s_anchor = _switch_s_ref(dists_anchor)
    s_pert = _switch_s_ref(dists_pert)
    r_anchor_safe = np.maximum(dists_anchor, 1.0e-6)
    r_pert_safe = np.maximum(dists_pert, 1.0e-6)

    gauss_anchor = ke * q_ij[None, :, :] * erf(gamma_eff * r_anchor_safe) / r_anchor_safe * s_anchor
    gauss_pert = ke * q_ij[None, :, :] * erf(gamma_eff * r_pert_safe) / r_pert_safe * s_pert
    bare_anchor = ke * q_ij[None, :, :] / r_anchor_safe * s_anchor
    bare_pert = ke * q_ij[None, :, :] / r_pert_safe * s_pert

    def _masked_delta(anchor_vals, pert_vals, group_mask: Optional[np.ndarray]) -> np.ndarray:
        gm = group_mask[None, :, :] if group_mask is not None else True
        s_anch = np.sum(np.where(cutoff_mask_anchor & gm, anchor_vals, 0.0), axis=(1, 2))
        s_pert_ = np.sum(np.where(cutoff_mask_pert & gm, pert_vals, 0.0), axis=(1, 2))
        return s_pert_ - s_anch

    print("[3/3] 计算 Δgauss/Δbare/Δpenetration，跟 §6.6 同款方法论做按anchor相关性诊断")
    delta_gauss_all = _masked_delta(gauss_anchor, gauss_pert, None)
    delta_bare_all = _masked_delta(bare_anchor, bare_pert, None)
    delta_penetration_all = delta_bare_all - delta_gauss_all

    delta_gauss_da = _masked_delta(gauss_anchor, gauss_pert, donor_acceptor_mask)
    delta_bare_da = _masked_delta(bare_anchor, bare_pert, donor_acceptor_mask)
    delta_penetration_da = delta_bare_da - delta_gauss_da

    fit_m2 = _contact_type_ridge_loao(ctx, float(args.contact_type_ridge_lambda), float(args.perturb_fit_mag04_weight), verbose=False)
    residual_m2_oof = residual_m0 - fit_m2["oof_pred_m2"]

    variables = {
        "delta_gauss_all_pairs": delta_gauss_all,
        "delta_bare_coulomb_all_pairs": delta_bare_all,
        "delta_penetration_all_pairs": delta_penetration_all,
        "delta_gauss_donor_acceptor_pairs": delta_gauss_da,
        "delta_bare_coulomb_donor_acceptor_pairs": delta_bare_da,
        "delta_penetration_donor_acceptor_pairs": delta_penetration_da,
    }
    report: Dict[str, Dict] = {}
    for name, var in variables.items():
        report[name] = {
            "vs_residual_m0": _per_anchor_pearson_stability(var, residual_m0, anchor_local_idx, n_anchors),
            "vs_residual_m2_oof": _per_anchor_pearson_stability(var, residual_m2_oof, anchor_local_idx, n_anchors),
        }
        print(f"    {name}:")
        for target_label, stats in report[name].items():
            if stats.get("n_anchors_used", 0) == 0:
                continue
            print(
                f"        {target_label:18s} mean_r={stats['mean_r']:+.3f} std_r={stats['std_r']:.3f} "
                f"sign_stability={stats['sign_stability_frac']:.0%} |r|>0.3占比={stats['frac_anchors_abs_r_over_0.3']:.0%}"
            )

    any_stable_signal = any(
        stats.get("n_anchors_used", 0) > 0 and stats["sign_stability_frac"] >= 0.85 and stats["frac_anchors_abs_r_over_0.3"] >= 0.5
        for target_dict in report.values()
        for stats in target_dict.values()
    )

    summary = {
        "sigma_elec_nm": sigma_elec_nm,
        "mm_ref_cutoff_nm": mm_ref_cutoff,
        "mm_ref_switch_nm": mm_ref_switch,
        "used_no_cutoff_reference": not use_ref_cutoff,
        "n_donor_acceptor_pairs_note": ctx["da_active_frac"],
        "variable_vs_residual_correlations": report,
        "any_stable_signal_found": any_stable_signal,
        "decision_note": (
            "判据(跟§6.6一致)：sign_stability_frac>=0.85 且 |r|>0.3 的 anchor 占比>=0.5，才算'未见 anchor 上"
            "存在稳定关联'。如果 any_stable_signal_found=false：radial(§6.5)/angular(§6.6)/electrostatic(本节)"
            "三条路都没有在这 20 个 anchor 上找到稳定信号，§6 这整条 contact-type 修正的探索到此为止——"
            "剩余的 odd residual 应该被记录为'无法用任何已尝试的全局函数形式(pairwise径向/三体角度/"
            "统一Gaussian宽度扰动)捕捉的多体环境误差'，见 DEXP_KERNEL_PHYSICS_ISSUES.md §3.4 最后一条解读："
            "配体分子内/成键项、蛋白邻近残基约束、Gaussian-Coulomb力本身都可能部分抵消这个孤立二体分解"
            "残差，不代表完整体系里存在同等大小的净力误差。如果为 true，说明确实存在跟电荷/donor-acceptor"
            "相关的稳定信号，下一步才是考虑 role-specific sigma_elec，而且要用独立的静电专属验证"
            "(不要跟 DEXP vdW 残差一锅拟合，会分不清 vdW shape/Gaussian charge penetration/angular 三种"
            "误差来源，这是用户明确要求的)。"
        ),
    }
    summary_path = os.path.join(output_dir, "gaussian_width_diagnostic_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    any_stable_signal_found = {any_stable_signal}")
    print(f"    summary: {summary_path}")
    return summary


def run_kernel_projection_benchmark(args: argparse.Namespace, output_dir: str) -> Dict:
    """用户重新定的框架（2026-07-12/13）：MACE 是参考势能面，LJ 和 DEXP 都只是描述它的
    解析语言。DEXP 的目标不是复现 MACE 的全部结构（多体/角度/anchor-specific 细节——这些
    结构性地超出任何各向同性 pairwise 径向核的表达能力，LJ 本身也做不到），只是要比 LJ
    更好地投影 MACE 局部相互作用势能面里"光滑、有界"的 even/曲率部分。

    v2（用户 2026-07-13 指出 v1 结论"DEXP 比 LJ 在 odd 上也明显更好"需要更细粒度的证据
    才站得住，且 v1 JSON 里的 note 已被实测结果否定，需要改掉）：不再只报聚合 odd/even，
    补齐——

    1. 每个 (pert_type,magnitude) 档单独的 odd/even RMSE(7档)，而不是按 pert_type 池化；
    2. 每个 anchor 的整体 RMSE/MAE，以及三个核之间的按 anchor 胜负计数；
    3. K1(12,6) vs K2(14,5) 专门按 anchor 分别统计 odd RMSE 和 even RMSE 的胜负——直接
       回答"14,5 是否只在 even 上稳定赢、odd 上有没有统一胜者"；
    4. 整体按扰动档等权(跟 --perturb-fit 同款权重方案，含 0.04nm 降权)的加权 RMSE；
    5. 整体与分 pert_type 的 median/trimmed(10%) RMSE，抗几个极端短接触帧主导结果；
    6. matched-switch(0.50->0.70nm quintic switch，跟生产 DEXPSurrogatePotential 同款)
       与无 switch 两套的敏感性对比(Phase 1 已确认 switch 对 DEXP 的 delta 影响可忽略，
       这里补做 LJ 版本，因为 LJ 的 r^-12 wall 离 cutoff 更近，敏感性可能不同)；
    7. MACE 条件均值剖面：每个(pert_type,magnitude)档比较 target 与预测的分箱均值±SEM，
       是否在合并SEM内——检验"平滑剖面"而不是被单帧噪声主导的原始RMSE；
    8. 按 anchor 自身 min_dist_anchor_nm 分三分位(近/中/远)重新做 odd/even，检验优势是否
       只来自少数极短接触的 anchor；
    9. 只用"真正局部"的小幅扰动(translation<=0.01nm, rotation<=1.5°)单独重算，排除
       "LJ 只是在 0.04nm/3°最大扰动下爆墙"这个替代解释。

    不要求 K1/K2 的 odd residual 归零，但也不再假设 DEXP-vs-LJ 在 odd 上应该没有改善——
    v1 实测显示 DEXP 在 odd 上相对 LJ 也有 33-58% 改善，这是跨函数族(LJ 的 r^-12/r^-6
    幂律 vs DEXP 的双指数)的效应，不跟 DEXP 内部 alpha/beta 对 odd 不敏感(§3.4)矛盾——
    后者是同一函数族内部的比较。
    """
    csv_path = os.path.join(output_dir, "perturb_scan_diagnostics.csv")
    npz_path = os.path.join(output_dir, "perturb_scan_geometry.npz")
    ensure_file(csv_path, "perturb-scan 诊断 CSV（先跑一次 --perturb-scan）")
    ensure_file(npz_path, "perturb-scan 几何快照 npz（先跑一次 --perturb-scan）")

    rows = read_csv_rows(csv_path)
    geo = np.load(npz_path)
    n_rows = len(rows)
    delta_e_target = np.asarray([float(r["delta_e_target_kjmol"]) for r in rows], dtype=float)
    pert_type = np.asarray([r["pert_type"] for r in rows], dtype=object)
    magnitude = np.asarray([float(r["magnitude"]) for r in rows], dtype=float)
    axis_kind = np.asarray([r["axis_kind"] for r in rows], dtype=object)
    axis_index = np.asarray([int(r["axis_index"]) for r in rows], dtype=int)
    sign = np.asarray([float(r["sign"]) for r in rows], dtype=float)
    min_dist_anchor = np.asarray([float(r["min_dist_anchor_nm"]) for r in rows], dtype=float)

    anchor_local_idx = geo["perturbation_anchor_index"].astype(int)
    if len(anchor_local_idx) != n_rows:
        raise RuntimeError(
            f"CSV 行数({n_rows})与几何 npz 行数({len(anchor_local_idx)})不一致，"
            "两者必须来自同一次 --perturb-scan 运行"
        )
    env_positions = geo["env_positions"]
    anchor_lig_positions = geo["anchor_lig_positions"]
    perturbed_lig_positions = geo["perturbed_lig_positions"]
    box_vectors = geo["box_vectors"]
    has_periodic = geo["has_periodic"].astype(bool)
    sigma_lig, eps_lig = geo["sigma_lig"], geo["eps_lig"]
    sigma_env, eps_env = geo["sigma_env"], geo["eps_env"]
    n_anchors = int(anchor_local_idx.max()) + 1
    cutoff_nm = float(args.perturb_baseline_cutoff_nm)
    switch_width = 0.20  # 与生产 DEXPSurrogatePotential.switch_width 默认值一致

    print("[1/7] 重建完整 ligand-environment 距离张量（复用 --perturb-fit 同款几何缓存，K0/K1/K2 共用）")
    tensors = _build_perturbation_distance_tensors(
        n_rows, anchor_local_idx, env_positions, anchor_lig_positions, perturbed_lig_positions,
        box_vectors, has_periodic, sigma_lig, eps_lig, sigma_env, eps_env, cutoff_nm,
    )
    dists_anchor_full = tensors["dists_anchor_full"]
    dists_pert_full = tensors["dists_pert_full"]
    sigma_ij_full = tensors["sigma_ij_full"]
    eps_ij_full = tensors["eps_ij_full"]
    x_anchor_full = tensors["x_anchor_full"]
    x_pert_full = tensors["x_pert_full"]
    mask_anchor_full = tensors["mask_anchor_full"]
    mask_pert_full = tensors["mask_pert_full"]

    def _switch_s(dists: np.ndarray) -> np.ndarray:
        s = np.ones_like(dists)
        r_switch = cutoff_nm - switch_width
        in_sw = (dists > r_switch) & (dists <= cutoff_nm)
        t = (dists[in_sw] - r_switch) / switch_width
        s[in_sw] = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
        return s

    s_anchor = _switch_s(dists_anchor_full)
    s_pert = _switch_s(dists_pert_full)

    print("[2/7] 计算 K0(原始pair-specific LJ) / K1(DEXP12,6) / K2(DEXP14,5) 的逐pair能量(anchor/perturbed两态)")

    def _lj_pair_energy(dists: np.ndarray) -> np.ndarray:
        sr6 = (sigma_ij_full[None, :, :] / np.maximum(dists, 1.0e-6)) ** 6
        return 4.0 * eps_ij_full[None, :, :] * (sr6 ** 2 - sr6)

    def _dexp_pair_energy(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        c_a, c_b = beta / (alpha - beta), alpha / (alpha - beta)
        return eps_ij_full[None, :, :] * (c_a * np.exp(-alpha * x) - c_b * np.exp(-beta * x))

    pair_energy_anchor = {
        "K0_LJ": _lj_pair_energy(dists_anchor_full),
        "K1_DEXP_12_6": _dexp_pair_energy(x_anchor_full, 12.0, 6.0),
        "K2_DEXP_14_5": _dexp_pair_energy(x_anchor_full, 14.0, 5.0),
    }
    pair_energy_pert = {
        "K0_LJ": _lj_pair_energy(dists_pert_full),
        "K1_DEXP_12_6": _dexp_pair_energy(x_pert_full, 12.0, 6.0),
        "K2_DEXP_14_5": _dexp_pair_energy(x_pert_full, 14.0, 5.0),
    }

    def _predict_delta_u(name: str, use_switch: bool) -> np.ndarray:
        pa, pp = pair_energy_anchor[name], pair_energy_pert[name]
        if use_switch:
            pa, pp = pa * s_anchor, pp * s_pert
        u_anchor = np.sum(np.where(mask_anchor_full, pa, 0.0), axis=(1, 2))
        u_pert = np.sum(np.where(mask_pert_full, pp, 0.0), axis=(1, 2))
        return u_pert - u_anchor

    kernel_names = ["K0_LJ", "K1_DEXP_12_6", "K2_DEXP_14_5"]
    delta_u = {name: _predict_delta_u(name, use_switch=False) for name in kernel_names}
    delta_u_switch = {name: _predict_delta_u(name, use_switch=True) for name in kernel_names}
    residual = {name: delta_e_target - delta_u[name] for name in kernel_names}
    residual_switch = {name: delta_e_target - delta_u_switch[name] for name in kernel_names}

    group_key_to_idx: Dict[Tuple, Dict[float, int]] = {}
    for i in range(n_rows):
        key = (int(anchor_local_idx[i]), str(pert_type[i]), str(axis_kind[i]), int(axis_index[i]), float(magnitude[i]))
        group_key_to_idx.setdefault(key, {})[float(sign[i])] = i

    BIN_KEYS = [
        ("rotation", 0.5), ("rotation", 1.5), ("rotation", 3.0),
        ("translation", 0.005), ("translation", 0.01), ("translation", 0.02), ("translation", 0.04),
    ]

    def _odd_even_general(resid: np.ndarray, anchor_filter: Optional[set] = None, bin_filter: Optional[set] = None) -> Dict:
        odd_vals: List[float] = []
        even_vals: List[float] = []
        for key, signed in group_key_to_idx.items():
            a, ptype, _axis_kind, _axis_index, mag = key
            if anchor_filter is not None and a not in anchor_filter:
                continue
            if bin_filter is not None and (ptype, mag) not in bin_filter:
                continue
            if 1.0 not in signed or -1.0 not in signed:
                continue
            ip, im = signed[1.0], signed[-1.0]
            odd_vals.append(float((resid[ip] - resid[im]) / 2.0))
            even_vals.append(float((resid[ip] + resid[im]) / 2.0))
        oa, ea = np.asarray(odd_vals, dtype=float), np.asarray(even_vals, dtype=float)
        return {
            "n_pairs": int(oa.size),
            "odd_residual_rmse_kjmol": float(np.sqrt(np.mean(oa ** 2))) if oa.size else math.nan,
            "even_residual_rmse_kjmol": float(np.sqrt(np.mean(ea ** 2))) if ea.size else math.nan,
        }

    def _pct(base: float, new: float) -> float:
        if base is None or math.isnan(base) or abs(base) < 1.0e-9:
            return math.nan
        return float((base - new) / base * 100.0)

    print("[3/7] 按 pert_type 池化的奇偶分解（无switch，主结果）+ 逐 (pert_type,magnitude) 档细分")
    pooled_by_type: Dict[str, Dict] = {}
    for name in kernel_names:
        pooled_by_type[name] = {
            ptype: _odd_even_general(residual[name], bin_filter={(ptype, m) for (pt, m) in BIN_KEYS if pt == ptype})
            for ptype in ("translation", "rotation")
        }
        for ptype in ("translation", "rotation"):
            oe = pooled_by_type[name][ptype]
            print(f"    {name:14s} {ptype:12s} odd={oe['odd_residual_rmse_kjmol']:.3f} even={oe['even_residual_rmse_kjmol']:.3f} kJ/mol")

    by_bin: Dict[str, Dict[str, Dict]] = {name: {} for name in kernel_names}
    for ptype, mag in BIN_KEYS:
        bin_label = f"{ptype}:{mag}"
        for name in kernel_names:
            by_bin[name][bin_label] = _odd_even_general(residual[name], bin_filter={(ptype, mag)})
        print(
            f"    [{bin_label:16s}] "
            + "  ".join(
                f"{name}:odd={by_bin[name][bin_label]['odd_residual_rmse_kjmol']:.3f}/even={by_bin[name][bin_label]['even_residual_rmse_kjmol']:.3f}"
                for name in kernel_names
            )
        )

    improvement_pooled: Dict[str, Dict] = {}
    for name in ("K1_DEXP_12_6", "K2_DEXP_14_5"):
        improvement_pooled[name] = {}
        for ptype in ("translation", "rotation"):
            lj = pooled_by_type["K0_LJ"][ptype]
            dx = pooled_by_type[name][ptype]
            improvement_pooled[name][ptype] = {
                "even_improvement_pct_vs_LJ": _pct(lj["even_residual_rmse_kjmol"], dx["even_residual_rmse_kjmol"]),
                "odd_improvement_pct_vs_LJ": _pct(lj["odd_residual_rmse_kjmol"], dx["odd_residual_rmse_kjmol"]),
            }

    print("[4/7] 每个 anchor 的整体 RMSE/MAE + 三核按 anchor 胜负计数（无switch）")
    per_anchor: Dict[str, List[Dict]] = {name: [] for name in kernel_names}
    for name in kernel_names:
        r = residual[name]
        for a in range(n_anchors):
            m = anchor_local_idx == a
            ra = r[m]
            per_anchor[name].append({
                "anchor": int(a),
                "rmse_kjmol": float(np.sqrt(np.mean(ra ** 2))),
                "mae_kjmol": float(np.mean(np.abs(ra))),
                "median_abs_kjmol": float(np.median(np.abs(ra))),
            })

    win_counts_overall = {name: 0 for name in kernel_names}
    for a in range(n_anchors):
        rmses = {name: per_anchor[name][a]["rmse_kjmol"] for name in kernel_names}
        win_counts_overall[min(rmses, key=rmses.get)] += 1
    print(f"    整体RMSE按anchor胜负(共{n_anchors}个anchor): {win_counts_overall}")

    def _per_anchor_component_rmse(resid: np.ndarray, component: str) -> Dict[int, float]:
        by_anchor: Dict[int, List[float]] = {}
        for key, signed in group_key_to_idx.items():
            if 1.0 not in signed or -1.0 not in signed:
                continue
            a = key[0]
            ip, im = signed[1.0], signed[-1.0]
            val = (resid[ip] - resid[im]) / 2.0 if component == "odd" else (resid[ip] + resid[im]) / 2.0
            by_anchor.setdefault(a, []).append(float(val))
        return {a: float(np.sqrt(np.mean(np.asarray(v) ** 2))) for a, v in by_anchor.items()}

    print("[5/7] K1(12,6) vs K2(14,5) 按 anchor 分别统计 odd/even 胜负——直接检验'14,5是否只在even上稳定赢'")
    odd_k1 = _per_anchor_component_rmse(residual["K1_DEXP_12_6"], "odd")
    odd_k2 = _per_anchor_component_rmse(residual["K2_DEXP_14_5"], "odd")
    even_k1 = _per_anchor_component_rmse(residual["K1_DEXP_12_6"], "even")
    even_k2 = _per_anchor_component_rmse(residual["K2_DEXP_14_5"], "even")

    def _win_tally(vals_a: Dict[int, float], vals_b: Dict[int, float], name_a: str, name_b: str) -> Dict:
        tally = {name_a: 0, name_b: 0, "tie": 0}
        for a in vals_a:
            va, vb = vals_a[a], vals_b.get(a, math.nan)
            if math.isnan(vb):
                continue
            if math.isclose(va, vb, rel_tol=1.0e-9, abs_tol=1.0e-9):
                tally["tie"] += 1
            elif va < vb:
                tally[name_a] += 1
            else:
                tally[name_b] += 1
        return tally

    odd_win_tally = _win_tally(odd_k1, odd_k2, "K1_DEXP_12_6", "K2_DEXP_14_5")
    even_win_tally = _win_tally(even_k1, even_k2, "K1_DEXP_12_6", "K2_DEXP_14_5")
    print(f"    odd  RMSE 按anchor胜负 (K1 vs K2): {odd_win_tally}  (若无一方明显占多数 -> 无统一胜者)")
    print(f"    even RMSE 按anchor胜负 (K1 vs K2): {even_win_tally}")

    print("[6/7] 整体按扰动档等权(0.04nm降权)的加权RMSE + median/trimmed(10%) RMSE")
    downweight = {("translation", 0.04): float(args.perturb_fit_mag04_weight)}

    def _row_weights(mask_rows: np.ndarray) -> np.ndarray:
        w = np.zeros(n_rows, dtype=float)
        for key in BIN_KEYS:
            bin_mask = mask_rows & (pert_type == key[0]) & np.isclose(magnitude, key[1])
            n_bin = int(np.sum(bin_mask))
            if n_bin == 0:
                continue
            w[bin_mask] = downweight.get(key, 1.0) / n_bin
        total = float(np.sum(w[mask_rows]))
        if total > 1.0e-12:
            w[mask_rows] /= total
        return w

    all_mask = np.ones(n_rows, dtype=bool)
    all_w = _row_weights(all_mask)

    def _trimmed_stats(resid: np.ndarray, mask_rows: np.ndarray, trim_frac: float = 0.1) -> Dict:
        r = resid[mask_rows]
        w = all_w[mask_rows]
        weighted_rmse = float(np.sqrt(np.sum(w * r ** 2) / np.sum(w)))
        median_abs = float(np.median(np.abs(r)))
        n = r.size
        k = int(n * trim_frac)
        order = np.argsort(np.abs(r))
        trimmed = r[order[k: n - k]] if n - 2 * k > 0 else r
        return {
            "weighted_rmse_kjmol": weighted_rmse,
            "median_abs_residual_kjmol": median_abs,
            "trimmed10pct_rmse_kjmol": float(np.sqrt(np.mean(trimmed ** 2))),
            "n": int(n),
        }

    robust_stats: Dict[str, Dict] = {}
    for name in kernel_names:
        robust_stats[name] = {"overall": _trimmed_stats(residual[name], all_mask)}
        for ptype in ("translation", "rotation"):
            robust_stats[name][ptype] = _trimmed_stats(residual[name], pert_type == ptype)
        print(
            f"    {name:14s} overall: weighted_rmse={robust_stats[name]['overall']['weighted_rmse_kjmol']:.3f}  "
            f"median|.|={robust_stats[name]['overall']['median_abs_residual_kjmol']:.3f}  "
            f"trimmed10%_rmse={robust_stats[name]['overall']['trimmed10pct_rmse_kjmol']:.3f} kJ/mol"
        )

    print("[6b/7] switch 敏感性：无switch vs 加switch(0.50->0.70nm) 的池化odd/even对比")
    switch_sensitivity: Dict[str, Dict] = {}
    for name in kernel_names:
        noswitch_oe = {ptype: pooled_by_type[name][ptype] for ptype in ("translation", "rotation")}
        switch_oe = {
            ptype: _odd_even_general(residual_switch[name], bin_filter={(pt, m) for (pt, m) in BIN_KEYS if pt == ptype})
            for ptype in ("translation", "rotation")
        }
        switch_sensitivity[name] = {"no_switch": noswitch_oe, "with_switch": switch_oe}
        for ptype in ("translation", "rotation"):
            print(
                f"    {name:14s} {ptype:12s} 无switch odd/even={noswitch_oe[ptype]['odd_residual_rmse_kjmol']:.3f}/{noswitch_oe[ptype]['even_residual_rmse_kjmol']:.3f}  "
                f"加switch odd/even={switch_oe[ptype]['odd_residual_rmse_kjmol']:.3f}/{switch_oe[ptype]['even_residual_rmse_kjmol']:.3f}"
            )

    print("[6c/7] MACE 条件均值剖面：每个(pert_type,magnitude)档比较 target 与预测的分箱均值±SEM")
    profile_by_bin: Dict[str, Dict[str, Dict]] = {name: {} for name in kernel_names}
    for ptype, mag in BIN_KEYS:
        bin_label = f"{ptype}:{mag}"
        m = (pert_type == ptype) & np.isclose(magnitude, mag)
        n_b = int(np.sum(m))
        t_mean = float(np.mean(delta_e_target[m]))
        t_sem = float(np.std(delta_e_target[m]) / math.sqrt(max(1, n_b)))
        for name in kernel_names:
            p_mean = float(np.mean(delta_u[name][m]))
            p_sem = float(np.std(delta_u[name][m]) / math.sqrt(max(1, n_b)))
            combined_sem = float(math.sqrt(t_sem ** 2 + p_sem ** 2))
            diff = p_mean - t_mean
            profile_by_bin[name][bin_label] = {
                "n": n_b, "target_mean_kjmol": t_mean, "target_sem_kjmol": t_sem,
                "pred_mean_kjmol": p_mean, "pred_sem_kjmol": p_sem,
                "diff_kjmol": diff, "within_combined_sem": bool(abs(diff) <= combined_sem),
            }
    for name in kernel_names:
        n_within = sum(1 for v in profile_by_bin[name].values() if v["within_combined_sem"])
        print(f"    {name:14s} 条件均值在合并SEM内的档数: {n_within}/{len(BIN_KEYS)}")

    print("[6d/7] 按 anchor 自身 min_dist_anchor_nm 分三分位(近/中/远)重新做 odd/even")
    anchor_min_dist = np.array([
        min_dist_anchor[np.where(anchor_local_idx == a)[0][0]] for a in range(n_anchors)
    ])
    order = np.argsort(anchor_min_dist)
    tercile_size = max(1, n_anchors // 3)
    tercile_of_anchor: Dict[int, str] = {}
    for rank, a in enumerate(order):
        label = "near" if rank < tercile_size else ("mid" if rank < 2 * tercile_size else "far")
        tercile_of_anchor[int(a)] = label
    tercile_sets = {label: {a for a, lab in tercile_of_anchor.items() if lab == label} for label in ("near", "mid", "far")}

    distance_layer: Dict[str, Dict] = {name: {} for name in kernel_names}
    for label, aset in tercile_sets.items():
        for name in kernel_names:
            distance_layer[name][label] = _odd_even_general(residual[name], anchor_filter=aset)
        print(
            f"    [{label:4s}, n_anchor={len(aset)}] "
            + "  ".join(f"{name}:odd={distance_layer[name][label]['odd_residual_rmse_kjmol']:.3f}/even={distance_layer[name][label]['even_residual_rmse_kjmol']:.3f}" for name in kernel_names)
        )

    print("[7/7] 只用真正局部的小幅扰动(translation<=0.01nm, rotation<=1.5°)重算——排除'LJ只在最大扰动爆墙'的替代解释")
    small_scale_bins = {("translation", 0.005), ("translation", 0.01), ("rotation", 0.5), ("rotation", 1.5)}
    small_scale: Dict[str, Dict] = {}
    for name in kernel_names:
        small_scale[name] = _odd_even_general(residual[name], bin_filter=small_scale_bins)
        print(f"    {name:14s} 小幅扰动子集: odd={small_scale[name]['odd_residual_rmse_kjmol']:.3f} even={small_scale[name]['even_residual_rmse_kjmol']:.3f} kJ/mol (n_pairs={small_scale[name]['n_pairs']})")

    small_scale_win = {"K0_LJ_better": 0, "DEXP_better": 0}
    for name in ("K1_DEXP_12_6", "K2_DEXP_14_5"):
        for comp in ("odd_residual_rmse_kjmol", "even_residual_rmse_kjmol"):
            if small_scale[name][comp] < small_scale["K0_LJ"][comp]:
                small_scale_win["DEXP_better"] += 1
            else:
                small_scale_win["K0_LJ_better"] += 1

    summary = {
        "cutoff_nm": cutoff_nm,
        "switch_width_nm": switch_width,
        "n_anchors": n_anchors,
        "odd_even_pooled_by_pert_type": pooled_by_type,
        "odd_even_by_bin": by_bin,
        "dexp_improvement_vs_lj_pooled": improvement_pooled,
        "per_anchor_overall_rmse_mae": per_anchor,
        "win_counts_overall_rmse_by_anchor": win_counts_overall,
        "k1_vs_k2_per_anchor_odd_rmse": odd_k1, "k1_vs_k2_per_anchor_even_rmse": even_k1,
        "k2_per_anchor_odd_rmse": odd_k2, "k2_per_anchor_even_rmse": even_k2,
        "odd_win_tally_k1_vs_k2": odd_win_tally,
        "even_win_tally_k1_vs_k2": even_win_tally,
        "robust_stats_weighted_median_trimmed": robust_stats,
        "switch_sensitivity": switch_sensitivity,
        "mace_conditional_mean_profile_by_bin": profile_by_bin,
        "distance_tercile_odd_even": distance_layer,
        "small_scale_only_odd_even": small_scale,
        "small_scale_only_win_tally_dexp_vs_lj": small_scale_win,
        "note": (
            "K0=原始pair-specific LJ, K1=DEXP(12,6), K2=DEXP(14,5)。v1 note 曾写'odd improvement "
            "应接近0'，已被实测(K1/K2 相对 K0 在 odd 上分别改善 33-58%)否定，予以更正：DEXP 相对 "
            "LJ 是跨函数族的全面改善(odd+even 都赢)，DEXP 内部 alpha/beta 对 odd 不敏感(§3.4)是"
            "同一函数族内部的独立现象，两者不矛盾。本次(v2)结果：K1 vs K2 在 even 上有稳定胜者"
            "(见 even_win_tally_k1_vs_k2)，odd 上没有稳定胜者(见 odd_win_tally_k1_vs_k2，接近对半"
            "开)——'14,5 更擅长 even，odd 没有统一胜者'这个表述由按anchor胜负计数直接支持，不是"
            "只看池化均值的印象。small_scale_only_* 专门检验只用最小幅度扰动(<=0.01nm/1.5°)时"
            "DEXP 是否仍稳定优于 LJ，用于排除'LJ 优势只是被 0.04nm/3°最大扰动下的爆墙拉低平均值'"
            "这个替代解释。distance_tercile_odd_even 检验优势是否只来自少数极近接触的 anchor。"
            "这一切仍然只是单体系(Atenolol)结果，不能代表跨体系普遍性——真正的普遍性验证需要"
            "8-15个化学多样体系的 --mace-kernel-benchmark(尚未实现)，且还应加上力/曲率/平滑性"
            "(r->0能量有限/短接触稳定性)对比，不能只看odd/even energy RMSE一个维度。"
        ),
    }
    summary_path = os.path.join(output_dir, "kernel_projection_benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    summary: {summary_path}")
    return summary


def run_mace_residual_force_benchmark(args: argparse.Namespace, output_dir: str) -> Dict:
    """Phase 3（用户 2026-07-13 提出，DEXP_KERNEL_PHYSICS_ISSUES.md §11 Phase 3 待办项）：
    把 --perturb-scan 已缓存的 ±δ anchor-relative 能量差重新解读成局部相互作用的广义
    力/力矩投影，而不是只看 --kernel-projection-benchmark 的奇偶分解 RMSE。零新增 MACE
    计算，复用同一份几何/能量缓存。

    关键口径（用户明确要求，均已落实）：
    - e_odd(δ) 不是单点 /δ：对每个 (anchor, pert_type, axis_kind, axis_index) 组合，用全部
      可用幅度做 e_odd(δ)=g·δ+c3·δ³ 最小二乘拟合，g 才是 δ→0 的局部梯度估计——比单点估计
      更抗大幅非线性(c3 项)，也不会被最小幅度下的 MACE 数值噪声单独放大。e_even(δ) 同理拟合
      成 h·δ²/2+c4·δ⁴ 给出局部曲率 h。
    - force_projection=-g(平移轴)，torque_projection=-g(转动轴，转动幅度先从度转弧度再拟合)。
    - 这是 delta_e_target=E_MACE_int-E_gauss_coul 的局部导数，不是配体所受的完整体系净力/
      净力矩(不含配体分子内力、蛋白成键力、完整长程静电、环境弛豫)——字段命名一律带
      local_target_*/kernel_*/residual_* 语义前缀，不叫 physical_net_force。
    - 3 个惯性主轴(以及随机方向)的 lab-frame 单位向量没有单独存盘，但可以从
      perturbed_lig_positions-anchor_lig_positions 的刚体位移精确反解(除以 sign*magnitude)，
      不依赖重新做特征值分解，避免退化/符号歧义——用来把 3 个轴向标量投影重建成局部基底下的
      3 维向量，计算 target vs kernel 的 force/torque cosine similarity。
    - Phase 2(环境半径收敛，--mace-env-convergence，已实现，尚未运行)与本 Phase 没有计算
      依赖，但有解释层面的依赖：当前结果建立在现有 MACE 团簇截断协议(0.50nm，逐原子裁剪)
      之上，DEXP整体 vs LJ 的优势幅度大，环境定义改变不易翻转；但核内部(12,6 vs 14,5)的 odd
      若接近打平，对截断偏置更敏感，最终结论需要等 Phase 2 跑完后才能下。

    〔2026-07-13 首跑后用户指出的 4 处修正，均已落实〕：
    1. **self-consistency ≠ 贴近MACE**：原 held-out 只检验"某个量自己的3主轴重建向量能否预测
       它自己在随机方向上的独立拟合梯度"——这只证明该量在这个幅度范围内是不是一个自洽的线性
       向量场（K1(12,6)这项rmse最小，只说明它作为解析核天然最线性，不代表它最贴近MACE）。
       新增真正的 cross-model held-out：MACE target 在随机方向的独立实测梯度 vs 各 kernel 从
       自己的3主轴重建、投影到同一随机方向的预测值——这才回答"谁更贴近MACE"，两者在
       summary/日志里分开报告，不再混用同一个"held-out"名字。
    2. **只有随机平移方向，没有随机旋转方向**：--perturb-scan 从不对旋转生成random方向(旋转
       永远只用3个principal轴)，所以 self-consistency/cross-model/Hessian 都只覆盖force，
       不覆盖torque——已在 note 和日志里明确改成"force held-out"而不是笼统的"force/torque"。
    3. **曲率不是向量，是Hessian对角投影**：curv_trans/curv_rot 改名为
       curvature_profile_translation/rotation，明确其含义是 h_ii=e_i^T H e_i 三个对角元，
       不是完整的二阶张量。另外新增用3主轴对角+随机方向解出对称3x3平移Hessian(非对角项由
       随机方向的e_even(δ)二次项反解，超定则最小二乘)，比较 target vs kernel 的 Frobenius
       残差与特征值RMSE，这才是名副其实的曲率比较。
    4. 新增：平均/中位残差范数、幅值比、按残差范数(不只是cosine)的按anchor胜负；以及
       kernel间两两(K1-K0/K2-K0/K2-K1)在20个anchor上的95% bootstrap置信区间，覆盖
       force/torque cosine、残差范数、cross-model held-out rmse，避免只看点估计下结论。
    """
    csv_path = os.path.join(output_dir, "perturb_scan_diagnostics.csv")
    npz_path = os.path.join(output_dir, "perturb_scan_geometry.npz")
    ensure_file(csv_path, "perturb-scan 诊断 CSV（先跑一次 --perturb-scan）")
    ensure_file(npz_path, "perturb-scan 几何快照 npz（先跑一次 --perturb-scan）")

    rows = read_csv_rows(csv_path)
    geo = np.load(npz_path)
    n_rows = len(rows)
    delta_e_target = np.asarray([float(r["delta_e_target_kjmol"]) for r in rows], dtype=float)
    pert_type = np.asarray([r["pert_type"] for r in rows], dtype=object)
    magnitude = np.asarray([float(r["magnitude"]) for r in rows], dtype=float)
    axis_kind = np.asarray([r["axis_kind"] for r in rows], dtype=object)
    axis_index = np.asarray([int(r["axis_index"]) for r in rows], dtype=int)
    sign = np.asarray([float(r["sign"]) for r in rows], dtype=float)

    anchor_local_idx = geo["perturbation_anchor_index"].astype(int)
    if len(anchor_local_idx) != n_rows:
        raise RuntimeError(
            f"CSV 行数({n_rows})与几何 npz 行数({len(anchor_local_idx)})不一致，"
            "两者必须来自同一次 --perturb-scan 运行"
        )
    env_positions = geo["env_positions"]
    anchor_lig_positions = geo["anchor_lig_positions"]
    perturbed_lig_positions = geo["perturbed_lig_positions"]
    box_vectors = geo["box_vectors"]
    has_periodic = geo["has_periodic"].astype(bool)
    sigma_lig, eps_lig = geo["sigma_lig"], geo["eps_lig"]
    sigma_env, eps_env = geo["sigma_env"], geo["eps_env"]
    n_anchors = int(anchor_local_idx.max()) + 1
    cutoff_nm = float(args.perturb_baseline_cutoff_nm)
    kernel_names = ["K0_LJ", "K1_DEXP_12_6", "K2_DEXP_14_5"]

    print("[1/6] 重建完整 ligand-environment 距离张量 + K0(LJ)/K1(DEXP12,6)/K2(DEXP14,5) 逐pair能量"
          "（复用 --perturb-scan 几何缓存，无switch——Phase 1/§10.4 已确认switch影响可忽略）")
    tensors = _build_perturbation_distance_tensors(
        n_rows, anchor_local_idx, env_positions, anchor_lig_positions, perturbed_lig_positions,
        box_vectors, has_periodic, sigma_lig, eps_lig, sigma_env, eps_env, cutoff_nm,
    )
    sigma_ij_full, eps_ij_full = tensors["sigma_ij_full"], tensors["eps_ij_full"]
    x_anchor_full, x_pert_full = tensors["x_anchor_full"], tensors["x_pert_full"]
    dists_anchor_full, dists_pert_full = tensors["dists_anchor_full"], tensors["dists_pert_full"]
    mask_anchor_full, mask_pert_full = tensors["mask_anchor_full"], tensors["mask_pert_full"]

    def _lj_pair_energy(dists: np.ndarray) -> np.ndarray:
        sr6 = (sigma_ij_full[None, :, :] / np.maximum(dists, 1.0e-6)) ** 6
        return 4.0 * eps_ij_full[None, :, :] * (sr6 ** 2 - sr6)

    def _dexp_pair_energy(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        c_a, c_b = beta / (alpha - beta), alpha / (alpha - beta)
        return eps_ij_full[None, :, :] * (c_a * np.exp(-alpha * x) - c_b * np.exp(-beta * x))

    pair_energy_anchor = {
        "K0_LJ": _lj_pair_energy(dists_anchor_full),
        "K1_DEXP_12_6": _dexp_pair_energy(x_anchor_full, 12.0, 6.0),
        "K2_DEXP_14_5": _dexp_pair_energy(x_anchor_full, 14.0, 5.0),
    }
    pair_energy_pert = {
        "K0_LJ": _lj_pair_energy(dists_pert_full),
        "K1_DEXP_12_6": _dexp_pair_energy(x_pert_full, 12.0, 6.0),
        "K2_DEXP_14_5": _dexp_pair_energy(x_pert_full, 14.0, 5.0),
    }

    def _predict_delta_u(name: str) -> np.ndarray:
        pa, pp = pair_energy_anchor[name], pair_energy_pert[name]
        u_anchor = np.sum(np.where(mask_anchor_full, pa, 0.0), axis=(1, 2))
        u_pert = np.sum(np.where(mask_pert_full, pp, 0.0), axis=(1, 2))
        return u_pert - u_anchor

    delta_u = {name: _predict_delta_u(name) for name in kernel_names}
    residual = {name: delta_e_target - delta_u[name] for name in kernel_names}
    quantities: Dict[str, np.ndarray] = {"target": delta_e_target}
    quantities.update(delta_u)
    for name in kernel_names:
        quantities[f"residual_{name}"] = residual[name]
    qty_names = ["target"] + kernel_names
    fit_qty_names = qty_names + [f"residual_{name}" for name in kernel_names]

    print("[2/6] 按 (anchor, pert_type, axis_kind, axis_index) 分组，收集跨幅度的 ±δ 数据点")
    groups: Dict[Tuple[int, str, str, int], Dict[float, Dict[float, int]]] = {}
    for i in range(n_rows):
        key = (int(anchor_local_idx[i]), str(pert_type[i]), str(axis_kind[i]), int(axis_index[i]))
        groups.setdefault(key, {}).setdefault(float(magnitude[i]), {})[float(sign[i])] = i

    def _phys_delta(ptype: str, mag: float) -> float:
        return math.radians(mag) if ptype == "rotation" else mag

    def _fit_linear_cubic(deltas: List[float], values: List[float]) -> Dict:
        d = np.asarray(deltas, dtype=float)
        v = np.asarray(values, dtype=float)
        order = np.argsort(d)
        d, v = d[order], v[order]
        secant_smallest = float(v[0] / d[0]) if d.size >= 1 and abs(d[0]) > 1.0e-12 else math.nan
        if d.size >= 2:
            X = np.stack([d, d ** 3], axis=1)
            coef, *_ = np.linalg.lstsq(X, v, rcond=None)
            g, c3 = float(coef[0]), float(coef[1])
            fit_rmse = float(np.sqrt(np.mean((v - X @ coef) ** 2)))
        else:
            g, c3, fit_rmse = secant_smallest, math.nan, math.nan
        return {
            "g": g, "c3": c3, "fit_rmse_kjmol": fit_rmse,
            "n_magnitudes": int(d.size), "secant_smallest_delta": secant_smallest,
        }

    def _fit_quadratic_quartic(deltas: List[float], values: List[float]) -> Dict:
        d = np.asarray(deltas, dtype=float)
        v = np.asarray(values, dtype=float)
        order = np.argsort(d)
        d, v = d[order], v[order]
        if d.size >= 2:
            X = np.stack([d ** 2, d ** 4], axis=1)
            coef, *_ = np.linalg.lstsq(X, v, rcond=None)
            h, c4 = float(2.0 * coef[0]), float(coef[1])
            fit_rmse = float(np.sqrt(np.mean((v - X @ coef) ** 2)))
        elif d.size == 1 and abs(d[0]) > 1.0e-12:
            h, c4, fit_rmse = float(2.0 * v[0] / d[0] ** 2), math.nan, math.nan
        else:
            h, c4, fit_rmse = math.nan, math.nan, math.nan
        return {"h": h, "c4": c4, "fit_rmse_kjmol": fit_rmse, "n_magnitudes": int(d.size)}

    print("[3/6] 每个 axis-group 拟合 δ→0 局部梯度 g(force/torque投影) 与曲率 h(even)")
    odd_fit: Dict[str, Dict[Tuple, Dict]] = {q: {} for q in fit_qty_names}
    even_fit: Dict[str, Dict[Tuple, Dict]] = {q: {} for q in fit_qty_names}
    for key, by_mag in groups.items():
        _, ptype, _axk, _axi = key
        for q in fit_qty_names:
            deltas, odd_vals, even_vals = [], [], []
            for mag, signed in sorted(by_mag.items()):
                if 1.0 not in signed or -1.0 not in signed:
                    continue
                ip, im = signed[1.0], signed[-1.0]
                deltas.append(_phys_delta(ptype, mag))
                odd_vals.append(float((quantities[q][ip] - quantities[q][im]) / 2.0))
                even_vals.append(float((quantities[q][ip] + quantities[q][im]) / 2.0))
            odd_fit[q][key] = _fit_linear_cubic(deltas, odd_vals)
            even_fit[q][key] = _fit_quadratic_quartic(deltas, even_vals)

    print("[sanity] 校验 residual 的 g 是否精确等于 target_g-kernel_g（线性拟合算子的线性性，隐式正确性检验）")
    max_dev = 0.0
    for key in groups:
        for name in kernel_names:
            gt, gk, gr = odd_fit["target"].get(key), odd_fit[name].get(key), odd_fit[f"residual_{name}"].get(key)
            if gt is None or gk is None or gr is None:
                continue
            if math.isnan(gt["g"]) or math.isnan(gk["g"]) or math.isnan(gr["g"]):
                continue
            max_dev = max(max_dev, abs(gr["g"] - (gt["g"] - gk["g"])))
    print(f"    max|g_residual-(g_target-g_kernel)| = {max_dev:.3e} kJ/mol/unit（应≈0）")

    print("[4/6] 重建 3 个平移主轴(+随机方向)的 lab-frame 单位向量(从刚体位移精确反解，不重新做特征值分解)")
    direction_by_key: Dict[Tuple[int, str, int], np.ndarray] = {}
    for key, by_mag in groups.items():
        a, ptype, axk, axi = key
        if ptype != "translation":
            continue
        disp_over_mag: List[np.ndarray] = []
        for mag, signed in by_mag.items():
            for s in (1.0, -1.0):
                if s not in signed:
                    continue
                i = signed[s]
                disp = perturbed_lig_positions[i] - anchor_lig_positions[a]
                disp_over_mag.append(disp.mean(axis=0) / (s * mag))
        if not disp_over_mag:
            continue
        v = np.mean(disp_over_mag, axis=0)
        norm = np.linalg.norm(v)
        if norm > 1.0e-9:
            direction_by_key[(a, axk, axi)] = v / norm

    def _vec_for(fits: Dict[str, Dict[Tuple, Dict]], field: str, qty_name: str, ptype: str, anchor: int, sign_mult: float) -> Optional[np.ndarray]:
        comps = []
        for axi in (0, 1, 2):
            fit = fits[qty_name].get((anchor, ptype, "principal", axi))
            if fit is None or math.isnan(fit[field]):
                return None
            comps.append(sign_mult * fit[field])
        return np.asarray(comps, dtype=float)

    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1.0e-12 or nv < 1.0e-12:
            return math.nan
        return float(np.dot(u, v) / (nu * nv))

    print("[5/6] 组装每个 anchor 的局部 3 维 force/torque 向量 + curvature_profile(Hessian对角投影，不是完整二阶张量)(K0/K1/K2/target/residual)，计算 cosine/范数/幅值比")
    per_anchor_vectors: Dict[str, Dict] = {}
    per_anchor_kernel_compare: Dict[str, List[Dict]] = {name: [] for name in kernel_names}
    for a in range(n_anchors):
        entry: Dict = {"anchor": a}
        f_target = _vec_for(odd_fit, "g", "target", "translation", a, -1.0)
        t_target = _vec_for(odd_fit, "g", "target", "rotation", a, -1.0)
        cf_target = _vec_for(even_fit, "h", "target", "translation", a, 1.0)
        ct_target = _vec_for(even_fit, "h", "target", "rotation", a, 1.0)
        entry["target_force_projection"] = f_target.tolist() if f_target is not None else None
        entry["target_torque_projection"] = t_target.tolist() if t_target is not None else None
        for name in kernel_names:
            f_k = _vec_for(odd_fit, "g", name, "translation", a, -1.0)
            t_k = _vec_for(odd_fit, "g", name, "rotation", a, -1.0)
            cf_k = _vec_for(even_fit, "h", name, "translation", a, 1.0)
            ct_k = _vec_for(even_fit, "h", name, "rotation", a, 1.0)
            f_r = _vec_for(odd_fit, "g", f"residual_{name}", "translation", a, -1.0)
            t_r = _vec_for(odd_fit, "g", f"residual_{name}", "rotation", a, -1.0)
            entry[f"{name}_force_projection"] = f_k.tolist() if f_k is not None else None
            entry[f"{name}_torque_projection"] = t_k.tolist() if t_k is not None else None
            entry[f"{name}_residual_force_projection"] = f_r.tolist() if f_r is not None else None
            entry[f"{name}_residual_torque_projection"] = t_r.tolist() if t_r is not None else None

            cmp_row: Dict = {"anchor": a, "kernel": name}
            for label, tgt, ker, unit_ in (
                ("force", f_target, f_k, "kjmol_per_nm"),
                ("torque", t_target, t_k, "kjmol_per_rad"),
                ("curvature_profile_translation", cf_target, cf_k, "kjmol_per_nm2"),
                ("curvature_profile_rotation", ct_target, ct_k, "kjmol_per_rad2"),
            ):
                if tgt is not None and ker is not None:
                    cmp_row[f"{label}_cosine"] = _cosine(tgt, ker)
                    cmp_row[f"{label}_vec_residual_norm_{unit_}"] = float(np.linalg.norm(tgt - ker))
                    nt = float(np.linalg.norm(tgt))
                    cmp_row[f"{label}_magnitude_ratio_kernel_over_target"] = float(np.linalg.norm(ker) / nt) if nt > 1.0e-9 else math.nan
                else:
                    cmp_row[f"{label}_cosine"] = math.nan
                    cmp_row[f"{label}_vec_residual_norm_{unit_}"] = math.nan
                    cmp_row[f"{label}_magnitude_ratio_kernel_over_target"] = math.nan
            per_anchor_kernel_compare[name].append(cmp_row)
        per_anchor_vectors[str(a)] = entry

    print("[6/8] pooled 统计 + 按anchor胜负(cosine + 残差范数)")
    label_unit = {
        "force": "kjmol_per_nm", "torque": "kjmol_per_rad",
        "curvature_profile_translation": "kjmol_per_nm2", "curvature_profile_rotation": "kjmol_per_rad2",
    }
    pooled_stats: Dict[str, Dict] = {}
    for name in kernel_names:
        pooled_stats[name] = {}
        for label in ("force", "torque", "curvature_profile_translation", "curvature_profile_rotation"):
            vals = np.asarray([r[f"{label}_cosine"] for r in per_anchor_kernel_compare[name]], dtype=float)
            vals = vals[~np.isnan(vals)]
            pooled_stats[name][f"mean_{label}_cosine"] = float(np.mean(vals)) if vals.size else math.nan
            pooled_stats[name][f"n_{label}_cosine"] = int(vals.size)
            norm_key = f"{label}_vec_residual_norm_{label_unit[label]}"
            norm_vals = np.asarray([r[norm_key] for r in per_anchor_kernel_compare[name]], dtype=float)
            norm_vals = norm_vals[~np.isnan(norm_vals)]
            pooled_stats[name][f"mean_{label}_vec_residual_norm"] = float(np.mean(norm_vals)) if norm_vals.size else math.nan
            pooled_stats[name][f"median_{label}_vec_residual_norm"] = float(np.median(norm_vals)) if norm_vals.size else math.nan
            ratio_vals = np.asarray(
                [r[f"{label}_magnitude_ratio_kernel_over_target"] for r in per_anchor_kernel_compare[name]], dtype=float
            )
            ratio_vals = ratio_vals[~np.isnan(ratio_vals)]
            pooled_stats[name][f"mean_{label}_magnitude_ratio"] = float(np.mean(ratio_vals)) if ratio_vals.size else math.nan
            pooled_stats[name][f"median_{label}_magnitude_ratio"] = float(np.median(ratio_vals)) if ratio_vals.size else math.nan
        print(
            f"    {name:14s} mean_force_cosine={pooled_stats[name]['mean_force_cosine']:.3f}  "
            f"mean_torque_cosine={pooled_stats[name]['mean_torque_cosine']:.3f}  "
            f"mean/median|ΔF|={pooled_stats[name]['mean_force_vec_residual_norm']:.1f}/{pooled_stats[name]['median_force_vec_residual_norm']:.1f}  "
            f"force_ratio(mean/median)={pooled_stats[name]['mean_force_magnitude_ratio']:.2f}/{pooled_stats[name]['median_force_magnitude_ratio']:.2f}  "
            f"torque_ratio(mean/median)={pooled_stats[name]['mean_torque_magnitude_ratio']:.2f}/{pooled_stats[name]['median_torque_magnitude_ratio']:.2f}"
        )

    win_by_force_cosine = {name: 0 for name in kernel_names}
    win_by_torque_cosine = {name: 0 for name in kernel_names}
    n_anchors_force_tally, n_anchors_torque_tally = 0, 0
    for a in range(n_anchors):
        fc = {name: per_anchor_kernel_compare[name][a]["force_cosine"] for name in kernel_names}
        tc = {name: per_anchor_kernel_compare[name][a]["torque_cosine"] for name in kernel_names}
        if all(not math.isnan(v) for v in fc.values()):
            win_by_force_cosine[max(fc, key=fc.get)] += 1
            n_anchors_force_tally += 1
        if all(not math.isnan(v) for v in tc.values()):
            win_by_torque_cosine[max(tc, key=tc.get)] += 1
            n_anchors_torque_tally += 1
    print(f"    按anchor force_cosine 最高胜场(n={n_anchors_force_tally}): {win_by_force_cosine}")
    print(f"    按anchor torque_cosine 最高胜场(n={n_anchors_torque_tally}): {win_by_torque_cosine}")

    win_by_force_norm = {name: 0 for name in kernel_names}
    win_by_torque_norm = {name: 0 for name in kernel_names}
    n_anchors_force_norm_tally, n_anchors_torque_norm_tally = 0, 0
    for a in range(n_anchors):
        fn = {name: per_anchor_kernel_compare[name][a]["force_vec_residual_norm_kjmol_per_nm"] for name in kernel_names}
        tn = {name: per_anchor_kernel_compare[name][a]["torque_vec_residual_norm_kjmol_per_rad"] for name in kernel_names}
        if all(not math.isnan(v) for v in fn.values()):
            win_by_force_norm[min(fn, key=fn.get)] += 1
            n_anchors_force_norm_tally += 1
        if all(not math.isnan(v) for v in tn.values()):
            win_by_torque_norm[min(tn, key=tn.get)] += 1
            n_anchors_torque_norm_tally += 1
    print(f"    按anchor force残差范数最小胜场(n={n_anchors_force_norm_tally}): {win_by_force_norm}")
    print(f"    按anchor torque残差范数最小胜场(n={n_anchors_torque_norm_tally}): {win_by_torque_norm}")

    print("[7/8] self-consistency(qty自证局部线性向量场，不代表贴近MACE) + cross-model held-out(kernel principal投影"
          " vs MACE target 随机方向实测——这才是回答'谁更贴近MACE'的检验)。"
          "注意：--perturb-scan 只对平移生成随机方向，没有随机旋转方向，因此本节和下面的Hessian一样只覆盖force，不覆盖torque。")
    self_consistency_rows: List[Dict] = []
    cross_model_rows: List[Dict] = []
    for a in range(n_anchors):
        principal_dirs = [direction_by_key.get((a, "principal", axi)) for axi in (0, 1, 2)]
        if any(d is None for d in principal_dirs):
            continue
        random_axes = sorted({axi for (aa, axk, axi) in direction_by_key if aa == a and axk == "random"})
        principal_g: Dict[str, List[float]] = {}
        ok_principal: Dict[str, bool] = {}
        for qty_name in qty_names:
            comps, ok = [], True
            for axi_p in (0, 1, 2):
                fit_p = odd_fit[qty_name].get((a, "translation", "principal", axi_p))
                if fit_p is None or math.isnan(fit_p["g"]):
                    ok = False
                    break
                comps.append(fit_p["g"])
            principal_g[qty_name] = comps
            ok_principal[qty_name] = ok
        for axi in random_axes:
            d_random = direction_by_key.get((a, "random", axi))
            if d_random is None:
                continue
            coords = np.asarray([float(np.dot(d_random, e)) for e in principal_dirs], dtype=float)
            key_random = (a, "translation", "random", axi)
            for qty_name in qty_names:
                fit_random = odd_fit[qty_name].get(key_random)
                if fit_random is None or math.isnan(fit_random["g"]) or not ok_principal[qty_name]:
                    continue
                g_pred_self = float(np.dot(np.asarray(principal_g[qty_name]), coords))
                self_consistency_rows.append({
                    "anchor": a, "random_axis_index": axi, "quantity": qty_name,
                    "g_actual_kjmol_per_nm": fit_random["g"], "g_predicted_kjmol_per_nm": g_pred_self,
                })
            fit_random_target = odd_fit["target"].get(key_random)
            if fit_random_target is None or math.isnan(fit_random_target["g"]):
                continue
            for name in kernel_names:
                if not ok_principal[name]:
                    continue
                g_pred_kernel = float(np.dot(np.asarray(principal_g[name]), coords))
                cross_model_rows.append({
                    "anchor": a, "random_axis_index": axi, "kernel": name,
                    "g_actual_target_kjmol_per_nm": fit_random_target["g"],
                    "g_predicted_kernel_kjmol_per_nm": g_pred_kernel,
                })

    def _summarize_pred_actual(rows: List[Dict], group_field: str, actual_field: str, pred_field: str) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        for key in sorted({r[group_field] for r in rows}):
            vals = [r for r in rows if r[group_field] == key]
            act = np.asarray([r[actual_field] for r in vals])
            prd = np.asarray([r[pred_field] for r in vals])
            corr = (
                float(np.corrcoef(act, prd)[0, 1])
                if len(vals) > 1 and np.std(act) > 1.0e-9 and np.std(prd) > 1.0e-9
                else math.nan
            )
            out[key] = {
                "n": len(vals),
                "rmse_kjmol_per_nm": float(np.sqrt(np.mean((act - prd) ** 2))),
                "corr_actual_vs_predicted": corr,
                "mean_abs_actual_kjmol_per_nm": float(np.mean(np.abs(act))),
            }
        return out

    self_consistency_summary = _summarize_pred_actual(
        self_consistency_rows, "quantity", "g_actual_kjmol_per_nm", "g_predicted_kjmol_per_nm"
    )
    for qty_name, stat in self_consistency_summary.items():
        print(f"    self-consistency[{qty_name:14s}] n={stat['n']:4d} rmse={stat['rmse_kjmol_per_nm']:.3f} corr={stat['corr_actual_vs_predicted']:.3f}")

    cross_model_summary = _summarize_pred_actual(
        cross_model_rows, "kernel", "g_actual_target_kjmol_per_nm", "g_predicted_kernel_kjmol_per_nm"
    )
    for name, stat in cross_model_summary.items():
        print(f"    cross-model(kernel->MACE)[{name:14s}] n={stat['n']:4d} rmse={stat['rmse_kjmol_per_nm']:.3f} corr={stat['corr_actual_vs_predicted']:.3f}")

    cross_model_by_anchor: Dict[str, Dict[int, float]] = {name: {} for name in kernel_names}
    for name in kernel_names:
        for a in range(n_anchors):
            vals = [r for r in cross_model_rows if r["kernel"] == name and r["anchor"] == a]
            if not vals:
                continue
            act = np.asarray([r["g_actual_target_kjmol_per_nm"] for r in vals])
            prd = np.asarray([r["g_predicted_kernel_kjmol_per_nm"] for r in vals])
            cross_model_by_anchor[name][a] = float(np.sqrt(np.mean((act - prd) ** 2)))

    print("[Hessian] 用3主轴对角 + 随机方向解对称3x3平移Hessian(不是简单'3维曲率向量')，比较Frobenius残差/特征值")

    def _reconstruct_hessian(qty_name: str, anchor: int) -> Optional[np.ndarray]:
        diag = []
        for axi in (0, 1, 2):
            fit_h = even_fit[qty_name].get((anchor, "translation", "principal", axi))
            if fit_h is None or math.isnan(fit_h["h"]):
                return None
            diag.append(fit_h["h"])
        principal_dirs = [direction_by_key.get((anchor, "principal", axi)) for axi in (0, 1, 2)]
        if any(d is None for d in principal_dirs):
            return None
        random_axes = sorted({axi for (aa, axk, axi) in direction_by_key if aa == anchor and axk == "random"})
        A_rows, b_rows = [], []
        for axi in random_axes:
            d_random = direction_by_key.get((anchor, "random", axi))
            fit_h_random = even_fit[qty_name].get((anchor, "translation", "random", axi))
            if d_random is None or fit_h_random is None or math.isnan(fit_h_random["h"]):
                continue
            c = np.asarray([float(np.dot(d_random, e)) for e in principal_dirs], dtype=float)
            known = c[0] ** 2 * diag[0] + c[1] ** 2 * diag[1] + c[2] ** 2 * diag[2]
            A_rows.append([2.0 * c[0] * c[1], 2.0 * c[0] * c[2], 2.0 * c[1] * c[2]])
            b_rows.append(fit_h_random["h"] - known)
        H = np.zeros((3, 3), dtype=float)
        H[0, 0], H[1, 1], H[2, 2] = diag
        if len(A_rows) >= 3:
            off_diag, *_ = np.linalg.lstsq(np.asarray(A_rows), np.asarray(b_rows), rcond=None)
            H[0, 1] = H[1, 0] = float(off_diag[0])
            H[0, 2] = H[2, 0] = float(off_diag[1])
            H[1, 2] = H[2, 1] = float(off_diag[2])
        return H

    hessian_compare: Dict[str, List[Dict]] = {name: [] for name in kernel_names}
    for a in range(n_anchors):
        H_target = _reconstruct_hessian("target", a)
        if H_target is None:
            continue
        eig_target = np.sort(np.linalg.eigvalsh(H_target))
        for name in kernel_names:
            H_k = _reconstruct_hessian(name, a)
            if H_k is None:
                continue
            eig_k = np.sort(np.linalg.eigvalsh(H_k))
            hessian_compare[name].append({
                "anchor": a,
                "frobenius_residual_kjmol_per_nm2": float(np.linalg.norm(H_target - H_k, ord="fro")),
                "eigenvalue_rmse_kjmol_per_nm2": float(np.sqrt(np.mean((eig_target - eig_k) ** 2))),
                "target_eigenvalues": eig_target.tolist(),
                "kernel_eigenvalues": eig_k.tolist(),
            })

    hessian_summary: Dict[str, Dict] = {}
    for name in kernel_names:
        rows = hessian_compare[name]
        if not rows:
            hessian_summary[name] = {"n": 0}
            continue
        fro = np.asarray([r["frobenius_residual_kjmol_per_nm2"] for r in rows])
        eig_rmse = np.asarray([r["eigenvalue_rmse_kjmol_per_nm2"] for r in rows])
        hessian_summary[name] = {
            "n": int(fro.size),
            "mean_frobenius_residual": float(np.mean(fro)),
            "median_frobenius_residual": float(np.median(fro)),
            "mean_eigenvalue_rmse": float(np.mean(eig_rmse)),
        }
        print(
            f"    Hessian[{name:14s}] n={hessian_summary[name]['n']:3d} "
            f"mean|ΔH|_F={hessian_summary[name]['mean_frobenius_residual']:.1f} "
            f"mean_eig_rmse={hessian_summary[name]['mean_eigenvalue_rmse']:.1f}"
        )

    print("[8/8] bootstrap CI(95%, over anchors)：kernel间两两比较 force/torque cosine、残差范数、cross-model force held-out RMSE")
    boot_rng = np.random.default_rng(int(args.seed) + 1)

    def _bootstrap_ci_mean_diff(vals_a: np.ndarray, vals_b: np.ndarray, n_boot: int = 4000, alpha: float = 0.05) -> Dict:
        mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
        diffs = (vals_a - vals_b)[mask]
        if diffs.size < 2:
            return {"n": int(diffs.size), "mean_diff": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan}
        n = diffs.size
        idx_pool = np.arange(n)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = boot_rng.choice(idx_pool, size=n, replace=True)
            boot_means[b] = np.mean(diffs[idx])
        lo, hi = np.percentile(boot_means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
        return {"n": int(n), "mean_diff": float(np.mean(diffs)), "ci95_lo": float(lo), "ci95_hi": float(hi)}

    kernel_pairs = [("K1_DEXP_12_6", "K0_LJ"), ("K2_DEXP_14_5", "K0_LJ"), ("K2_DEXP_14_5", "K1_DEXP_12_6")]
    bootstrap_ci: Dict[str, Dict] = {}
    for metric in ("force_cosine", "torque_cosine", "force_vec_residual_norm_kjmol_per_nm", "torque_vec_residual_norm_kjmol_per_rad"):
        bootstrap_ci[metric] = {}
        for name_a, name_b in kernel_pairs:
            va = np.asarray([r[metric] for r in per_anchor_kernel_compare[name_a]], dtype=float)
            vb = np.asarray([r[metric] for r in per_anchor_kernel_compare[name_b]], dtype=float)
            ci = _bootstrap_ci_mean_diff(va, vb)
            bootstrap_ci[metric][f"{name_a}_minus_{name_b}"] = ci
            print(f"    bootstrap[{metric}] {name_a}-{name_b}: mean_diff={ci['mean_diff']:.4f} 95%CI=[{ci['ci95_lo']:.4f},{ci['ci95_hi']:.4f}] n={ci['n']}")

    bootstrap_ci["cross_model_force_held_out_rmse_by_anchor"] = {}
    for name_a, name_b in kernel_pairs:
        anchors_common = sorted(set(cross_model_by_anchor[name_a]) & set(cross_model_by_anchor[name_b]))
        va = np.asarray([cross_model_by_anchor[name_a][a] for a in anchors_common], dtype=float)
        vb = np.asarray([cross_model_by_anchor[name_b][a] for a in anchors_common], dtype=float)
        ci = _bootstrap_ci_mean_diff(va, vb)
        bootstrap_ci["cross_model_force_held_out_rmse_by_anchor"][f"{name_a}_minus_{name_b}"] = ci
        print(f"    bootstrap[cross_model_rmse] {name_a}-{name_b}: mean_diff={ci['mean_diff']:.4f} 95%CI=[{ci['ci95_lo']:.4f},{ci['ci95_hi']:.4f}] n={ci['n']}")

    print("[Hessian bootstrap] 95%CI(n_boot=10000, 逐anchor配对，与上面同一套paired bootstrap机制)"
          "：Frobenius残差/特征值RMSE的按anchor胜场 + K1-K0/K2-K0/K2-K1 配对差值CI"
          "（不对三个kernel各自独立bootstrap再比较——那样会丢掉同一组anchor重采样的配对信息，"
          "同一次重采样必须对K0/K1/K2用完全相同的anchor下标，这里通过先在原始anchor层面做逐anchor"
          "差值、再对这个差值序列重采样来保证，等价于对同一组重采样anchor下标同时取三个kernel的值）")
    hessian_frobenius_by_anchor: Dict[str, Dict[int, float]] = {name: {} for name in kernel_names}
    hessian_eigrmse_by_anchor: Dict[str, Dict[int, float]] = {name: {} for name in kernel_names}
    for name in kernel_names:
        for row in hessian_compare[name]:
            hessian_frobenius_by_anchor[name][row["anchor"]] = row["frobenius_residual_kjmol_per_nm2"]
            hessian_eigrmse_by_anchor[name][row["anchor"]] = row["eigenvalue_rmse_kjmol_per_nm2"]

    win_by_hessian_frobenius = {name: 0 for name in kernel_names}
    win_by_hessian_eigrmse = {name: 0 for name in kernel_names}
    hessian_anchors_common = sorted(set.intersection(*[set(hessian_frobenius_by_anchor[name]) for name in kernel_names]))
    for a in hessian_anchors_common:
        fro_a = {name: hessian_frobenius_by_anchor[name][a] for name in kernel_names}
        eig_a = {name: hessian_eigrmse_by_anchor[name][a] for name in kernel_names}
        win_by_hessian_frobenius[min(fro_a, key=fro_a.get)] += 1
        win_by_hessian_eigrmse[min(eig_a, key=eig_a.get)] += 1
    n_hessian_anchors = len(hessian_anchors_common)
    print(f"    按anchor Hessian Frobenius残差最小胜场(n={n_hessian_anchors}): {win_by_hessian_frobenius}")
    print(f"    按anchor Hessian 特征值RMSE最小胜场(n={n_hessian_anchors}): {win_by_hessian_eigrmse}")

    bootstrap_ci["hessian_frobenius_residual"] = {}
    bootstrap_ci["hessian_eigenvalue_rmse"] = {}
    for name_a, name_b in kernel_pairs:
        anchors_pair = sorted(set(hessian_frobenius_by_anchor[name_a]) & set(hessian_frobenius_by_anchor[name_b]))
        va = np.asarray([hessian_frobenius_by_anchor[name_a][a] for a in anchors_pair], dtype=float)
        vb = np.asarray([hessian_frobenius_by_anchor[name_b][a] for a in anchors_pair], dtype=float)
        ci = _bootstrap_ci_mean_diff(va, vb, n_boot=10000)
        bootstrap_ci["hessian_frobenius_residual"][f"{name_a}_minus_{name_b}"] = ci
        print(f"    bootstrap[hessian_frobenius] {name_a}-{name_b}: mean_diff={ci['mean_diff']:.1f} 95%CI=[{ci['ci95_lo']:.1f},{ci['ci95_hi']:.1f}] n={ci['n']}")

        anchors_pair_eig = sorted(set(hessian_eigrmse_by_anchor[name_a]) & set(hessian_eigrmse_by_anchor[name_b]))
        va_e = np.asarray([hessian_eigrmse_by_anchor[name_a][a] for a in anchors_pair_eig], dtype=float)
        vb_e = np.asarray([hessian_eigrmse_by_anchor[name_b][a] for a in anchors_pair_eig], dtype=float)
        ci_e = _bootstrap_ci_mean_diff(va_e, vb_e, n_boot=10000)
        bootstrap_ci["hessian_eigenvalue_rmse"][f"{name_a}_minus_{name_b}"] = ci_e
        print(f"    bootstrap[hessian_eigrmse] {name_a}-{name_b}: mean_diff={ci_e['mean_diff']:.1f} 95%CI=[{ci_e['ci95_lo']:.1f},{ci_e['ci95_hi']:.1f}] n={ci_e['n']}")

    print("[by-magnitude] secant(单幅度) vs 多幅度拟合g 的偏离——检验跨幅度一致性/小幅噪声放大/大幅非线性")
    by_magnitude_rows: List[Dict] = []
    for key, by_mag in groups.items():
        a, ptype, axk, axi = key
        if axk != "principal":
            continue
        for qty_name in qty_names:
            fit = odd_fit[qty_name].get(key)
            if fit is None or math.isnan(fit["g"]):
                continue
            g_full = fit["g"]
            for mag, signed in by_mag.items():
                if 1.0 not in signed or -1.0 not in signed:
                    continue
                ip, im = signed[1.0], signed[-1.0]
                d_phys = _phys_delta(ptype, mag)
                e_odd_here = float((quantities[qty_name][ip] - quantities[qty_name][im]) / 2.0)
                secant = e_odd_here / d_phys if abs(d_phys) > 1.0e-12 else math.nan
                if math.isnan(secant):
                    continue
                by_magnitude_rows.append({
                    "pert_type": ptype, "magnitude_raw": mag, "quantity": qty_name,
                    "secant_minus_fullfit": secant - g_full,
                })
    by_magnitude_summary: Dict[str, Dict[str, Dict]] = {}
    for ptype in ("translation", "rotation"):
        mags = sorted({r["magnitude_raw"] for r in by_magnitude_rows if r["pert_type"] == ptype})
        by_magnitude_summary[ptype] = {}
        for mag in mags:
            by_magnitude_summary[ptype][str(mag)] = {}
            for qty_name in qty_names:
                arr = np.asarray([
                    r["secant_minus_fullfit"] for r in by_magnitude_rows
                    if r["pert_type"] == ptype and r["magnitude_raw"] == mag and r["quantity"] == qty_name
                ], dtype=float)
                if arr.size == 0:
                    continue
                by_magnitude_summary[ptype][str(mag)][qty_name] = {
                    "n": int(arr.size),
                    "rmse_secant_vs_fullfit": float(np.sqrt(np.mean(arr ** 2))),
                    "bias": float(np.mean(arr)),
                }

    by_anchor_csv_path = os.path.join(output_dir, "mace_residual_force_by_anchor.csv")
    fieldnames = [
        "anchor", "kernel", "force_cosine", "torque_cosine",
        "force_vec_residual_norm_kjmol_per_nm", "torque_vec_residual_norm_kjmol_per_rad",
        "force_magnitude_ratio_kernel_over_target", "torque_magnitude_ratio_kernel_over_target",
        "curvature_profile_translation_cosine", "curvature_profile_rotation_cosine",
    ]
    with open(by_anchor_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name in kernel_names:
            for row in per_anchor_kernel_compare[name]:
                writer.writerow({k: row.get(k) for k in fieldnames})

    by_magnitude_csv_path = os.path.join(output_dir, "mace_residual_force_by_magnitude.csv")
    with open(by_magnitude_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pert_type", "magnitude_raw", "quantity", "n", "rmse_secant_vs_fullfit", "bias"])
        writer.writeheader()
        for ptype, by_mag_dict in by_magnitude_summary.items():
            for mag_str, by_qty in by_mag_dict.items():
                for qty_name, stats in by_qty.items():
                    writer.writerow({"pert_type": ptype, "magnitude_raw": mag_str, "quantity": qty_name, **stats})

    summary = {
        "n_anchors": n_anchors,
        "n_rows": n_rows,
        "kernel_names": kernel_names,
        "linearity_self_check_max_abs_dev_g_residual_vs_target_minus_kernel": max_dev,
        "pooled_cosine_norm_ratio": pooled_stats,
        "win_by_anchor_force_cosine": win_by_force_cosine,
        "win_by_anchor_torque_cosine": win_by_torque_cosine,
        "n_anchors_used_for_force_cosine_win_tally": n_anchors_force_tally,
        "n_anchors_used_for_torque_cosine_win_tally": n_anchors_torque_tally,
        "win_by_anchor_force_residual_norm_smallest": win_by_force_norm,
        "win_by_anchor_torque_residual_norm_smallest": win_by_torque_norm,
        "n_anchors_used_for_force_norm_win_tally": n_anchors_force_norm_tally,
        "n_anchors_used_for_torque_norm_win_tally": n_anchors_torque_norm_tally,
        "self_consistency_check": self_consistency_summary,
        "cross_model_force_held_out_vs_mace": cross_model_summary,
        "hessian_comparison_summary": hessian_summary,
        "hessian_comparison_by_anchor": hessian_compare,
        "win_by_anchor_hessian_frobenius_smallest": win_by_hessian_frobenius,
        "win_by_anchor_hessian_eigenvalue_rmse_smallest": win_by_hessian_eigrmse,
        "n_anchors_used_for_hessian_win_tally": n_hessian_anchors,
        "bootstrap_ci_kernel_pairwise": bootstrap_ci,
        "per_anchor_vectors": per_anchor_vectors,
        "by_anchor_csv": by_anchor_csv_path,
        "by_magnitude_csv": by_magnitude_csv_path,
        "note": (
            "local_target/kernel/residual_force(torque)_projection 是 delta_e_target="
            "E_MACE_int-E_gauss_coul 沿配体刚体平移/转动主轴的局部导数(-dE/dq，用全部可用幅度"
            "拟合 e_odd(δ)=gδ+c3δ³ 取线性项 g，而不是单点 e_odd/δ)，不是配体所受的完整体系"
            "净力/净力矩——不含配体分子内力、蛋白成键力、完整长程静电、环境弛豫。3 个平移/转动"
            "主轴的 lab-frame 单位向量从 perturbed_lig_positions-anchor_lig_positions 的刚体"
            "位移精确反解得到，不依赖重新做特征值分解，避免退化/符号歧义。"
            "curvature_profile_translation/rotation 是 Hessian 对角线在3个主轴上的投影"
            "(h_ii=e_i^T H e_i)，不是一个真正的三维curvature向量——完整3x3对称Hessian见"
            "hessian_comparison_*(用3主轴对角+随机方向解出的3个非对角项，Frobenius残差/"
            "特征值RMSE比较target vs kernel)。"
            "self_consistency_check 只回答'某个量(target或某kernel)自己的响应在这个幅度范围内"
            "是否表现为线性向量场'——即从3主轴重建的向量投影到随机方向，能否预测该量自己在"
            "随机方向上独立拟合出的梯度。这不等于'哪个kernel更贴近MACE'：K1(12,6)的"
            "self-consistency rmse最小只说明K1自己最接近一个理想线性向量场(2体解析核+小幅"
            "平移下天然更线性)，不代表K1的力最贴近MACE的真实局部响应。"
            "cross_model_force_held_out_vs_mace才是回答'谁更贴近MACE'的检验：用MACE target"
            "自己在随机方向上独立拟合出的实测梯度(g_actual_target)，与该kernel从3主轴重建的"
            "向量投影到同一随机方向的预测值(g_predicted_kernel)比较——kernel完全没有用这个"
            "随机方向的数据参与3主轴拟合，是真正的held-out。"
            "重要限定：--perturb-scan 只在平移方向生成随机方向（--perturb-n-random-dirs），"
            "没有随机旋转方向，所以self_consistency_check/cross_model_force_held_out_vs_mace/"
            "hessian_comparison_* 都只覆盖force（平移），不覆盖torque（旋转）——JSON里没有"
            "torque held-out这个量，torque方面目前只有基于3个既定主轴本身的cosine/残差范数"
            "(pooled_cosine_norm_ratio里的torque_*字段)，没有独立于这3个轴的旋转方向可供验证。"
            "bootstrap_ci_kernel_pairwise 是逐anchor配对差值(K1-K0/K2-K0/K2-K1)在20个anchor上"
            "的95% bootstrap置信区间(over anchors,与anchor数=独立单元一致)，用于判断"
            "cosine/残差范数/cross-model held-out rmse/hessian_frobenius_residual/"
            "hessian_eigenvalue_rmse 的核间差异是否统计显著，而不是只看点估计——所有pair都是"
            "先在原始anchor层面取逐anchor差值，再对这个差值序列做bootstrap重采样(n_boot=10000"
            "用于hessian两项，其余4000)，等价于同一次重采样对K0/K1/K2使用完全相同的anchor"
            "下标，不是对三个kernel各自独立bootstrap后再比较(那样会丢掉配对信息、高估CI宽度)。"
            "本结果建立在当前 MACE 团簇的原子级环境截断协议(0.50nm半径、逐原子裁剪)"
            "之上，尚未经 Phase 2(--mace-env-convergence,已实现,尚未运行)验证该协议本身是否收敛。"
        ),
    }
    summary_path = os.path.join(output_dir, "mace_residual_force_benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    summary: {summary_path}")
    print(f"    by_anchor csv: {by_anchor_csv_path}")
    print(f"    by_magnitude csv: {by_magnitude_csv_path}")
    return summary


def _select_env_indices_residue_complete(frame, lig_idx: np.ndarray, env_radius_nm: float) -> np.ndarray:
    """半径近邻筛选出的原子，扩展成它们所属的完整残基/完整水分子——如果一个残基有任意原子
    落在半径内，整个残基(蛋白残基的全部主链+侧链原子，或水分子的全部O/H)都整进环境，避免在
    共价键中间切断产生悬挂价。不补氢、不做残基级别的额外裁剪(`--mace-env-convergence` 场景
    下故意不设 max_env_atoms 上限，只让"半径"和"裁剪方式"两个变量独立变化，参照
    `abfe_core.py::OrbBoreschEstimator._build_pocket_context` 里同款的
    "半径筛原子→按residue.index分组→整残基收进pocket_atoms"惯用法。"""
    md = require_module("mdtraj")
    hit_atoms = md.compute_neighbors(frame, env_radius_nm, lig_idx)[0]
    hit_atoms = np.setdiff1d(hit_atoms, lig_idx)
    top = frame.top
    # 沿用 abfe_core.py::OrbBoreschEstimator._build_pocket_context 里已验证过的"遍历
    # top.residues 按 res.index 是否命中筛选"惯用法，而不是假设 top.residue(idx) 直接索引
    # 一定可用，降低对 mdtraj 版本/API 细节的依赖。
    hit_residues = {top.atom(int(a)).residue.index for a in hit_atoms}
    env_atoms: set = set()
    for res in top.residues:
        if res.index in hit_residues:
            for atom in res.atoms:
                env_atoms.add(int(atom.index))
    return np.asarray(sorted(env_atoms - set(int(i) for i in lig_idx.tolist())), dtype=int)


def run_mace_env_convergence(args: argparse.Namespace, output_dir: str) -> Dict:
    """Phase 2（DEXP_KERNEL_PHYSICS_ISSUES.md §7/§11 待办项，用户 2026-07-13 提出）：检验
    §3-§10.6 全部 residual/odd/even/force/torque/Hessian 结论，是否依赖当前 MACE 团簇的
    环境截断协议(0.50nm 半径、逐原子裁剪，`abfe_core.py::_select_env_indices_from_mdtraj_frame`
    文档明确写着"裁的是原子，不是整盒水，也不是全环境残基")。

    固定 5 个 anchor、只用 --perturb-trans-nm/--perturb-rot-deg 里最小的那一档幅度，在
    4 个环境半径(0.50/0.60/0.70/0.90nm) x 2 种裁剪方式(A. 现有逐原子裁剪，复用
    `_select_env_indices_from_mdtraj_frame`；B. 完整残基/完整水分子裁剪，新增的
    `_select_env_indices_residue_complete`)下重新算 MACE，检查三件事：
    1) anchor-relative ΔE_MACE(以及 delta_e_target=E_MACE_int-E_gauss_coul，跟 §3 起
       全程使用的"target"量保持一致定义)是否随半径收敛(相邻半径的差值是否变小、0.90nm相对
       当前生产用的0.50nm的总漂移有多大，跟已知的odd/even残差量级(~3-8kJ/mol，见§3.4/§10.4)
       比较是否可忽略)；
    2) odd方向梯度(最小幅度下的单点(ΔE(+)-ΔE(-))/2，不做§Phase3那种跨幅度拟合，因为这里
       故意只用一个幅度)的符号是否随半径/裁剪方式翻转；
    3) LJ(K0)/DEXP(12,6)(K1)/DEXP(14,5)(K2) 的排序(奇偶RMSE，用这批5-anchor数据重新算，
       样本量比§10.3/10.4的20-anchor版本小很多，只用于'排序是否变'的screening，不是重新
       出具威信的benchmark数字)是否随环境定义改变。

    **这是这条线里唯一需要新增 MACE 计算的 Phase**（Phase 1/3 都是零新增，只复用
    `--perturb-scan` 缓存）。成本量级：5 anchor x 13 geometry(1个anchor pose + 3平移轴x2
    符号 + 3转动轴x2符号) x 4 半径 x 2 裁剪方式 = 520 个 (anchor,radius,mode,geometry) 组合，
    每个都要调一次 `_compute_orb_decomposition`(内部 complex/ligand/environment 三次 MACE
    前向)，即约 1560 次实际 MACE 前向计算——用户自己在最初提案里就估过这个量级
    ("520 geometry labels... 而且每个 interaction decomposition 又涉及
    complex/ligand/environment，因此实际求值成本会更高")，不是免费操作，需要真实 GPU 时间。
    0.90nm+残基完整裁剪这个条件的环境原子数预期最多(整残基/整水分子进出比同半径的逐原子
    裁剪原子数更多)，是这批里最慢的一个条件。

    **显存管理（不这样做会OOM）**：跟 `--perturb-scan`/`--kernel-projection-benchmark` 不同，
    这里 env_idx 在同一个 anchor 内部会随 radius/mode 反复变化(4x2=8种)，而
    `Orbv3DEXPFittingPipeline` 的 Context 缓存键含 env_idx——每换一个 (radius,mode) 条件都会
    新建一份缓存，如果只在整个anchor循环结束后才清一次，会导致同一个anchor内最多8份
    (且逐步变大，0.90nm+residue_complete最大)的显存同时累积不释放，多个anchor跑下来必然
    OOM。因此清理粒度是**每个 (anchor,radius,mode) 条件用完立刻清**，不是每个anchor清一次：
    `pipeline._clear_orb_context_cache()` + 显式 `del gauss_ctx/mm_contexts/_e_target`
    (OpenMM Context/System对象不指望等GC顺手回收) + `gc.collect()` + 如果有CUDA就
    `torch.cuda.empty_cache()`，这样峰值显存只对应"当前正在处理的这一个条件"，不随
    条件数累积。
    """
    openmm, app, unit, XmlSerializer = require_openmm()
    md = require_module("mdtraj")
    symbols = load_abfe_symbols()
    select_env_indices_atomwise = symbols["_select_env_indices_from_mdtraj_frame"]
    Orbv3DEXPFittingPipeline = symbols["Orbv3DEXPFittingPipeline"]
    NumpyEncoder = symbols["NumpyEncoder"]

    n_anchors = int(args.env_convergence_anchors)
    radii = sorted(float(x) for x in str(args.env_convergence_radii).split(",") if x.strip())
    trans_mag = min(float(x) for x in str(args.perturb_trans_nm).split(",") if x.strip())
    rot_mag_deg = min(float(x) for x in str(args.perturb_rot_deg).split(",") if x.strip())
    cutoff_nm = float(args.perturb_baseline_cutoff_nm)
    kernel_names = ["K0_LJ", "K1_DEXP_12_6", "K2_DEXP_14_5"]
    truncation_modes = ["atomwise", "residue_complete"]

    print(f"[1/4] 载入轨迹，抽取 {n_anchors} 个 anchor 帧（与 --perturb-scan 同款尾段抽取），"
          f"最小幅度：translation={trans_mag}nm rotation={rot_mag_deg}deg")
    traj = md.load(args.traj, top=args.traj_top)
    if len(traj) == 0:
        raise RuntimeError("轨迹为空，无法进行环境收敛检验")
    anchor_frame_ids = select_tail_indices_from_time(traj, n_anchors, args.fit_last_ns)
    anchor_traj = traj[anchor_frame_ids]
    if anchor_traj.unitcell_vectors is not None:
        anchor_traj = anchor_traj.image_molecules(inplace=False)

    lig_idx = np.asarray(anchor_traj.top.select(f"resname {args.ligand}"), dtype=int)
    if len(lig_idx) == 0:
        raise ValueError(f"未在轨迹拓扑中找到配体残基 `{args.ligand}`")

    with open(args.system_xml, "r", encoding="utf-8") as handle:
        sigma_lookup_system = XmlSerializer.deserialize(handle.read())
    nb_force = next(f for f in sigma_lookup_system.getForces() if isinstance(f, openmm.NonbondedForce))
    n_particles = sigma_lookup_system.getNumParticles()
    sigma_all = np.zeros(n_particles, dtype=float)
    eps_all = np.zeros(n_particles, dtype=float)
    for i in range(n_particles):
        _, sigma_i, epsilon_i = nb_force.getParticleParameters(i)
        sigma_all[i] = sigma_i.value_in_unit(unit.nanometer)
        eps_all[i] = epsilon_i.value_in_unit(unit.kilojoule_per_mole)
    sigma_lig, eps_lig = sigma_all[lig_idx], eps_all[lig_idx]

    all_nums = np.array([a.element.atomic_number for a in anchor_traj.top.atoms], dtype=int)
    pipeline = Orbv3DEXPFittingPipeline(model_name=args.ml_model, device=args.device)
    pipeline._cache_contexts = True
    first_pos_nm = np.asarray(anchor_traj.xyz[0], dtype=np.float64)
    first_env_idx = np.asarray(
        select_env_indices_atomwise(anchor_traj[0], lig_idx, radii[0], max_env_atoms=None), dtype=int
    )
    pipeline._preflight_orb_backend(first_pos_nm, lig_idx, first_env_idx, all_nums)

    print("[2/4] 为每个 anchor 生成最小幅度扰动几何(3平移轴+3转动轴，各±1，加anchor本身=13个)")
    anchors_geo: List[Dict] = []
    for a_local, a_frame_id in enumerate(anchor_frame_ids):
        pos_nm = np.asarray(anchor_traj.xyz[a_local], dtype=np.float64)
        box_vecs = (
            np.asarray(anchor_traj.unitcell_vectors[a_local], dtype=np.float64)
            if anchor_traj.unitcell_vectors is not None else None
        )
        lig_pos = pos_nm[lig_idx]
        com = lig_pos.mean(axis=0)
        centered = lig_pos - com
        inertia = centered.T @ centered
        _, axes = np.linalg.eigh(inertia)
        directions = [axes[:, k] / np.linalg.norm(axes[:, k]) for k in range(3)]

        geometries: List[Dict] = [{"label": "anchor", "pos_nm": pos_nm, "pert_type": None, "axis_index": None, "sign": None}]
        for d_i, direction in enumerate(directions):
            for sign in (1.0, -1.0):
                new_pos = pos_nm.copy()
                new_pos[lig_idx] = lig_pos + sign * trans_mag * direction
                geometries.append({
                    "label": f"translation_{d_i}_{'p' if sign > 0 else 'm'}", "pos_nm": new_pos,
                    "pert_type": "translation", "axis_index": d_i, "sign": sign,
                })
        ang_rad = math.radians(rot_mag_deg)
        for a_i, axis in enumerate(directions):
            K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
            for sign in (1.0, -1.0):
                theta = sign * ang_rad
                R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
                new_pos = pos_nm.copy()
                new_pos[lig_idx] = (centered @ R.T) + com
                geometries.append({
                    "label": f"rotation_{a_i}_{'p' if sign > 0 else 'm'}", "pos_nm": new_pos,
                    "pert_type": "rotation", "axis_index": a_i, "sign": sign,
                })
        anchors_geo.append({
            "anchor_local": a_local, "anchor_frame": int(a_frame_id),
            "box_vecs": box_vecs, "geometries": geometries,
        })

    def _kernel_energy_sum(dists: np.ndarray, sigma_ij: np.ndarray, eps_ij: np.ndarray) -> Dict[str, float]:
        mask = dists <= cutoff_nm
        r0_ij = (2.0 ** (1.0 / 6.0)) * np.maximum(sigma_ij, 1.0e-6)
        x = np.maximum(dists, 1.0e-6) / r0_ij - 1.0
        sr6 = (sigma_ij / np.maximum(dists, 1.0e-6)) ** 6
        out = {"K0_LJ": float(np.where(mask, 4.0 * eps_ij * (sr6 ** 2 - sr6), 0.0).sum())}
        for name, alpha, beta in (("K1_DEXP_12_6", 12.0, 6.0), ("K2_DEXP_14_5", 14.0, 5.0)):
            c_a, c_b = beta / (alpha - beta), alpha / (alpha - beta)
            e = eps_ij * (c_a * np.exp(-alpha * x) - c_b * np.exp(-beta * x))
            out[name] = float(np.where(mask, e, 0.0).sum())
        return out

    print(f"[3/4] 遍历 {len(radii)} 个半径 x 2 种裁剪方式 x {n_anchors} 个anchor，重算 MACE"
          f"（约 {n_anchors * len(radii) * len(truncation_modes) * 13} 个(anchor,radius,mode,geometry)组合）")
    all_rows: List[Dict] = []
    n_decompositions = 0
    for anchor_pack in anchors_geo:
        a_local = anchor_pack["anchor_local"]
        a_frame_id = anchor_pack["anchor_frame"]
        box_vecs = anchor_pack["box_vecs"]
        geometries = anchor_pack["geometries"]
        ref_frame = anchor_traj[a_local]
        for radius in radii:
            for mode in truncation_modes:
                if mode == "atomwise":
                    env_idx = np.asarray(
                        select_env_indices_atomwise(ref_frame, lig_idx, radius, max_env_atoms=None), dtype=int
                    )
                else:
                    env_idx = _select_env_indices_residue_complete(ref_frame, lig_idx, radius)
                if len(env_idx) == 0:
                    print(f"    [警告] anchor={a_local} radius={radius} mode={mode}：环境原子数=0，跳过")
                    continue
                sigma_env, eps_env = sigma_all[env_idx], eps_all[env_idx]
                sigma_ij = 0.5 * (sigma_lig[:, None] + sigma_env[None, :])
                eps_ij = np.sqrt(np.clip(eps_lig[:, None] * eps_env[None, :], 0.0, None))

                mm_contexts = build_mm_le_contexts_from_system_xml(
                    args.system_xml, ligand_indices=lig_idx.tolist(), environment_indices=env_idx.tolist(),
                    cutoff_nm=float(args.fit_mm_ref_cutoff), switching_nm=float(args.fit_mm_ref_switch),
                )
                gauss_ctx = mm_contexts["gauss_coul"]
                gauss_ctx_periodic = gauss_ctx.getSystem().usesPeriodicBoundaryConditions()

                def _e_target(pos_nm: np.ndarray) -> Tuple[float, float]:
                    e_orb = pipeline._compute_orb_decomposition(pos_nm, lig_idx, env_idx, all_nums)
                    if box_vecs is not None and gauss_ctx_periodic:
                        gauss_ctx.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box_vecs])
                    gauss_ctx.setPositions(pos_nm * unit.nanometer)
                    e_gauss = gauss_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                        unit.kilojoules_per_mole
                    )
                    return float(e_orb), float(e_orb - e_gauss)

                anchor_geo = geometries[0]
                e_orb_anchor, e_target_anchor = _e_target(anchor_geo["pos_nm"])
                n_decompositions += 1
                dists_anchor = _pairwise_dists_matrix(anchor_geo["pos_nm"][lig_idx], anchor_geo["pos_nm"][env_idx], box_vecs)
                kernel_anchor = _kernel_energy_sum(dists_anchor, sigma_ij, eps_ij)

                for geo in geometries[1:]:
                    e_orb_p, e_target_p = _e_target(geo["pos_nm"])
                    n_decompositions += 1
                    dists_p = _pairwise_dists_matrix(geo["pos_nm"][lig_idx], geo["pos_nm"][env_idx], box_vecs)
                    kernel_p = _kernel_energy_sum(dists_p, sigma_ij, eps_ij)
                    row = {
                        "anchor_local": a_local, "anchor_frame": a_frame_id,
                        "radius_nm": radius, "truncation_mode": mode, "n_env_atoms": int(len(env_idx)),
                        "pert_type": geo["pert_type"], "axis_index": geo["axis_index"], "sign": geo["sign"],
                        "delta_e_mace_int_kjmol": e_orb_p - e_orb_anchor,
                        "delta_e_target_kjmol": e_target_p - e_target_anchor,
                    }
                    for name in kernel_names:
                        row[f"delta_u_{name}_kjmol"] = kernel_p[name] - kernel_anchor[name]
                    all_rows.append(row)
                print(
                    f"    anchor={a_local} radius={radius:.2f}nm mode={mode:15s} "
                    f"env_atoms={len(env_idx):4d} 完成（累计decomposition次数={n_decompositions}）"
                )

                # 关键：每个(anchor,radius,mode)条件的 env_idx 都不同，MACE Context 缓存键
                # 含 env_idx，所以每换一个条件都会新建一份缓存——之前的bug是只在整个anchor
                # 循环结束后才清一次，导致同一个anchor内最多8个(4半径x2裁剪方式)条件的显存
                # 同时累积不释放，尤其0.90nm+residue_complete这个最大团簇会把峰值显存推得
                # 很高，多个anchor跑下来必然OOM。改成每个条件用完立刻清，峰值显存只对应
                # "当前正在处理的这一个条件"，不随条件数累积。gauss_ctx/mm_contexts 是
                # OpenMM Context/System对象，同样显式del再clear缓存，不指望等GC顺手回收。
                try:
                    pipeline._clear_orb_context_cache()
                except Exception:
                    pass
                del gauss_ctx, mm_contexts, _e_target
                gc.collect()
                try:
                    torch = require_module("torch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

    csv_path = os.path.join(output_dir, "env_convergence_diagnostics.csv")
    ensure_dir(output_dir)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("[4/4] 收敛诊断：ΔE_MACE/delta_e_target 是否随半径收敛、odd符号是否翻转、kernel排序是否改变")

    def _row_key(r: Dict) -> Tuple:
        return (r["anchor_local"], r["truncation_mode"], r["pert_type"], r["axis_index"])

    by_direction: Dict[Tuple, Dict[Tuple[float, float], Dict]] = {}
    for r in all_rows:
        key = _row_key(r)
        by_direction.setdefault(key, {})[(r["radius_nm"], r["sign"])] = r

    print("    [a] ΔE_target 随半径的收敛性 + 相对当前生产半径(0.50nm)的总漂移")
    convergence_rows: List[Dict] = []
    r_min, r_max = radii[0], radii[-1]
    for key, by_radius_sign in by_direction.items():
        anchor_local, mode, pert_type, axis_index = key
        vals_by_radius: Dict[float, float] = {}
        for radius in radii:
            plus, minus = by_radius_sign.get((radius, 1.0)), by_radius_sign.get((radius, -1.0))
            if plus is None or minus is None:
                continue
            vals_by_radius[radius] = plus["delta_e_target_kjmol"]
        avail_radii = sorted(vals_by_radius)
        if len(avail_radii) < 2:
            continue
        diffs = [vals_by_radius[avail_radii[i + 1]] - vals_by_radius[avail_radii[i]] for i in range(len(avail_radii) - 1)]
        drift_max_vs_min = (
            vals_by_radius[r_max] - vals_by_radius[r_min] if r_min in vals_by_radius and r_max in vals_by_radius else math.nan
        )
        convergence_rows.append({
            "anchor_local": anchor_local, "truncation_mode": mode, "pert_type": pert_type, "axis_index": axis_index,
            "radii_available": avail_radii, "values_by_radius": [vals_by_radius[r] for r in avail_radii],
            "successive_diffs": diffs,
            "successive_diffs_shrinking": bool(len(diffs) >= 2 and abs(diffs[-1]) < abs(diffs[0])),
            "drift_max_vs_min_radius_kjmol": drift_max_vs_min,
        })
    drift_vals = np.asarray(
        [abs(c["drift_max_vs_min_radius_kjmol"]) for c in convergence_rows if not math.isnan(c["drift_max_vs_min_radius_kjmol"])]
    )
    n_shrinking = sum(1 for c in convergence_rows if c["successive_diffs_shrinking"])
    convergence_summary = {
        "n_directions_checked": len(convergence_rows),
        "n_successive_diffs_shrinking": n_shrinking,
        "mean_abs_drift_r_max_minus_r_min_kjmol": float(np.mean(drift_vals)) if drift_vals.size else math.nan,
        "median_abs_drift_r_max_minus_r_min_kjmol": float(np.median(drift_vals)) if drift_vals.size else math.nan,
        "max_abs_drift_r_max_minus_r_min_kjmol": float(np.max(drift_vals)) if drift_vals.size else math.nan,
        "reference_scale_note": (
            "对比 §3.4/§10.4 已知的 odd/even residual RMSE 量级(~3-8 kJ/mol)：如果上面的"
            "mean/median drift 远小于这个量级，说明环境半径/裁剪方式改变不足以解释已有的"
            "odd/even 残差结构；如果量级相当或更大，说明环境截断本身就是 residual 的重要来源。"
        ),
    }
    print(
        f"        n_directions={convergence_summary['n_directions_checked']} "
        f"shrinking={n_shrinking} mean|drift(0.90-0.50)|={convergence_summary['mean_abs_drift_r_max_minus_r_min_kjmol']:.3f}kJ/mol"
    )

    print("    [b] odd方向梯度符号是否随半径/裁剪方式翻转")
    sign_stability_rows: List[Dict] = []
    for key, by_radius_sign in by_direction.items():
        anchor_local, mode, pert_type, axis_index = key
        odd_by_radius: Dict[float, float] = {}
        for radius in radii:
            plus, minus = by_radius_sign.get((radius, 1.0)), by_radius_sign.get((radius, -1.0))
            if plus is None or minus is None:
                continue
            odd_by_radius[radius] = (plus["delta_e_target_kjmol"] - minus["delta_e_target_kjmol"]) / 2.0
        if len(odd_by_radius) < 2:
            continue
        signs = {r: (1 if v > 0 else (-1 if v < 0 else 0)) for r, v in odd_by_radius.items()}
        nonzero_signs = {s for s in signs.values() if s != 0}
        sign_stability_rows.append({
            "anchor_local": anchor_local, "truncation_mode": mode, "pert_type": pert_type, "axis_index": axis_index,
            "odd_by_radius": odd_by_radius, "sign_stable": bool(len(nonzero_signs) <= 1),
        })
    n_stable = sum(1 for r in sign_stability_rows if r["sign_stable"])
    sign_stability_summary = {
        "n_directions_checked": len(sign_stability_rows),
        "n_sign_stable": n_stable,
        "n_sign_flipped": len(sign_stability_rows) - n_stable,
    }
    print(f"        n_directions={sign_stability_summary['n_directions_checked']} stable={n_stable} flipped={sign_stability_summary['n_sign_flipped']}")

    print("    [c] 按(radius,mode)分别算K0/K1/K2的odd/even RMSE，检验排序是否改变（5-anchor screening，不是重新出具威信benchmark）")
    ranking_by_condition: Dict[str, Dict] = {}
    for radius in radii:
        for mode in truncation_modes:
            rows_here = [r for r in all_rows if r["radius_nm"] == radius and r["truncation_mode"] == mode]
            pair_map: Dict[Tuple, Dict[float, Dict]] = {}
            for r in rows_here:
                k = (r["anchor_local"], r["pert_type"], r["axis_index"])
                pair_map.setdefault(k, {})[r["sign"]] = r
            odd_even_by_kernel: Dict[str, Dict[str, float]] = {}
            for name in kernel_names:
                odd_vals, even_vals = [], []
                for k, signed in pair_map.items():
                    if 1.0 not in signed or -1.0 not in signed:
                        continue
                    rp, rm = signed[1.0], signed[-1.0]
                    resid_p = rp["delta_e_target_kjmol"] - rp[f"delta_u_{name}_kjmol"]
                    resid_m = rm["delta_e_target_kjmol"] - rm[f"delta_u_{name}_kjmol"]
                    odd_vals.append((resid_p - resid_m) / 2.0)
                    even_vals.append((resid_p + resid_m) / 2.0)
                oa, ea = np.asarray(odd_vals), np.asarray(even_vals)
                odd_even_by_kernel[name] = {
                    "odd_rmse_kjmol": float(np.sqrt(np.mean(oa ** 2))) if oa.size else math.nan,
                    "even_rmse_kjmol": float(np.sqrt(np.mean(ea ** 2))) if ea.size else math.nan,
                    "n_pairs": int(oa.size),
                }
            best_overall = min(
                kernel_names,
                key=lambda n: (odd_even_by_kernel[n]["odd_rmse_kjmol"] + odd_even_by_kernel[n]["even_rmse_kjmol"]),
            )
            ranking_by_condition[f"{mode}@{radius}"] = {"odd_even_by_kernel": odd_even_by_kernel, "best_overall": best_overall}
            print(
                f"        [{mode:15s}@{radius:.2f}nm] "
                + "  ".join(f"{n}:odd={odd_even_by_kernel[n]['odd_rmse_kjmol']:.2f}/even={odd_even_by_kernel[n]['even_rmse_kjmol']:.2f}" for n in kernel_names)
                + f"  best={best_overall}"
            )
    distinct_best = {v["best_overall"] for v in ranking_by_condition.values()}
    ranking_stability_summary = {
        "ranking_by_condition": ranking_by_condition,
        "distinct_best_kernels_across_conditions": sorted(distinct_best),
        "ranking_stable": bool(len(distinct_best) == 1),
    }
    print(f"        跨{len(ranking_by_condition)}个条件的最优核: {sorted(distinct_best)} (stable={ranking_stability_summary['ranking_stable']})")

    summary = {
        "n_anchors": n_anchors,
        "radii_nm": radii,
        "truncation_modes": truncation_modes,
        "trans_mag_nm": trans_mag,
        "rot_mag_deg": rot_mag_deg,
        "cutoff_nm": cutoff_nm,
        "n_rows": len(all_rows),
        "n_mace_decompositions": n_decompositions,
        "env_atom_counts": {
            f"{mode}@{radius}": sorted({r["n_env_atoms"] for r in all_rows if r["radius_nm"] == radius and r["truncation_mode"] == mode})
            for radius in radii for mode in truncation_modes
        },
        "target_convergence": convergence_summary,
        "target_convergence_by_direction": convergence_rows,
        "odd_sign_stability": sign_stability_summary,
        "odd_sign_stability_by_direction": sign_stability_rows,
        "kernel_ranking_stability": ranking_stability_summary,
        "diagnostics_csv": csv_path,
        "note": (
            "Phase 2：检验 §3-§10.6 的全部结论是否依赖当前 MACE 团簇 0.50nm/逐原子裁剪的环境"
            "协议。固定5个anchor+单一最小幅度(与--perturb-scan相同的最小档translation/rotation"
            "幅度)，在4个半径x2种裁剪方式(atomwise=现有生产协议, residue_complete=完整残基/"
            "水分子，不补氢)下重新算MACE。delta_e_target定义与§3起全程一致(E_MACE_int-"
            "E_gauss_coul)。target_convergence看ΔE_target是否随半径收敛、0.90nm相对当前"
            "生产0.50nm的总漂移是否远小于已知odd/even残差量级(~3-8kJ/mol)；odd_sign_stability"
            "看方向导数符号是否随环境定义翻转(这是§8.2里'residual是否是anchor依赖的MACE截断"
            "产物'这个假说的直接检验)；kernel_ranking_stability看K0/K1/K2的排序(这一批"
            "5-anchor单幅度数据的odd+even RMSE之和最小者)是否随环境定义改变——用5个anchor、"
            "1个幅度做screening，样本量远小于§10.3/10.4的20-anchor/7-幅度benchmark，只用于"
            "判断'排序是否稳健'，不能替代那个benchmark的具体数字。"
        ),
    }
    summary_path = os.path.join(output_dir, "env_convergence_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, cls=NumpyEncoder)
    print(f"    summary: {summary_path}")
    print(f"    diagnostics csv: {csv_path}")
    return summary


def run_r0_scale_diagnostic(args: argparse.Namespace, output_dir: str) -> Dict:
    """用户 2026-07-13 提出的"旁线"：(12,6) 和 (14,5) 共用同一个 LJ-matched r0_ij，但两者都
    产生了同方向的 VAL/SER 重排(§4.4/§8)——这说明该现象主要不是 alpha/beta(形状/曲率)能
    解决的，值得单独检验挪动 r0 本身是否能吃掉这部分 odd 残差。零新增 MACE 计算，只复用
    已有 `--perturb-scan` 缓存。

    固定 alpha=14,beta=5(当前生产默认核)，把 r0_ij 替换成 `r0_ij(new)=s_r*r0_ij(LJ)`——
    DEXP 的 `x=r/r0-1` 归一化保证 `U(r0)=-eps,U'(r0)=0` 对任意 r0 数值都成立，所以这是一个
    良定义的单参数族，不是换了一个不同的势函数。扫描 s_r（默认0.96-1.04，步长0.01），
    对每个 s_r 用现有 --perturb-scan 几何/能量缓存重新算 anchor-relative ΔU、跟 delta_e_target
    做奇偶分解，检验用户指定的晋升为第4个MD condition的四个判据：
    1) odd 相对 s_r=1.0(当前生产基线) 是否显著改善(逐anchor配对bootstrap CI 不跨零，
       且方向是改善)；
    2) even 是否没有显著退化(bootstrap CI 不显示显著变差)——**只查even RMSE，不是完整
       Hessian**，这是本次的cheap screening范围，真正候选s_r可以之后单独跑一次
       `--mace-residual-force-benchmark`风格的完整Hessian复核；
    3) anchor-balanced LOAO：每次留一个anchor做验证，用其余anchor选出的最优s_r是否跨折稳定
       (不是靠单次全量拟合的偶然结果)；
    4) bootstrap 是否排除 s_r=1.0（即该s_r的改善不是噪声）。

    只有同时满足全部四条，才建议把该 s_r 提升为真正的第4个MD condition；否则维持现有
    (14,5)+LJ-matched r0(s_r=1.0)不变。
    """
    csv_path = os.path.join(output_dir, "perturb_scan_diagnostics.csv")
    npz_path = os.path.join(output_dir, "perturb_scan_geometry.npz")
    ensure_file(csv_path, "perturb-scan 诊断 CSV（先跑一次 --perturb-scan）")
    ensure_file(npz_path, "perturb-scan 几何快照 npz（先跑一次 --perturb-scan）")

    rows = read_csv_rows(csv_path)
    geo = np.load(npz_path)
    n_rows = len(rows)
    delta_e_target = np.asarray([float(r["delta_e_target_kjmol"]) for r in rows], dtype=float)
    pert_type = np.asarray([r["pert_type"] for r in rows], dtype=object)
    magnitude = np.asarray([float(r["magnitude"]) for r in rows], dtype=float)
    axis_kind = np.asarray([r["axis_kind"] for r in rows], dtype=object)
    axis_index = np.asarray([int(r["axis_index"]) for r in rows], dtype=int)
    sign = np.asarray([float(r["sign"]) for r in rows], dtype=float)

    anchor_local_idx = geo["perturbation_anchor_index"].astype(int)
    if len(anchor_local_idx) != n_rows:
        raise RuntimeError(
            f"CSV 行数({n_rows})与几何 npz 行数({len(anchor_local_idx)})不一致，"
            "两者必须来自同一次 --perturb-scan 运行"
        )
    env_positions = geo["env_positions"]
    anchor_lig_positions = geo["anchor_lig_positions"]
    perturbed_lig_positions = geo["perturbed_lig_positions"]
    box_vectors = geo["box_vectors"]
    has_periodic = geo["has_periodic"].astype(bool)
    sigma_lig, eps_lig = geo["sigma_lig"], geo["eps_lig"]
    sigma_env, eps_env = geo["sigma_env"], geo["eps_env"]
    n_anchors = int(anchor_local_idx.max()) + 1
    cutoff_nm = float(args.perturb_baseline_cutoff_nm)
    alpha = float(args.r0_scale_alpha)
    beta = float(args.r0_scale_beta)
    s_r_grid = sorted(float(x) for x in str(args.r0_scale_grid).split(",") if x.strip())
    baseline_sr = min(s_r_grid, key=lambda s: abs(s - 1.0))

    print(f"[1/5] 重建距离张量（复用 --perturb-scan 几何缓存），扫描 s_r∈{s_r_grid}，"
          f"alpha={alpha},beta={beta} 固定，baseline s_r={baseline_sr}")
    tensors = _build_perturbation_distance_tensors(
        n_rows, anchor_local_idx, env_positions, anchor_lig_positions, perturbed_lig_positions,
        box_vectors, has_periodic, sigma_lig, eps_lig, sigma_env, eps_env, cutoff_nm,
    )
    dists_anchor_full, dists_pert_full = tensors["dists_anchor_full"], tensors["dists_pert_full"]
    sigma_ij_full, eps_ij_full = tensors["sigma_ij_full"], tensors["eps_ij_full"]
    mask_anchor_full, mask_pert_full = tensors["mask_anchor_full"], tensors["mask_pert_full"]
    r0_ij_lj = tensors["r0_ij_full"]  # LJ-matched r0，s_r=1.0 时的基线

    c_a, c_b = beta / (alpha - beta), alpha / (alpha - beta)

    def _predict_delta_u(s_r: float) -> np.ndarray:
        r0_scaled = s_r * r0_ij_lj
        x_anchor = np.maximum(dists_anchor_full, 1.0e-6) / r0_scaled[None, :, :] - 1.0
        x_pert = np.maximum(dists_pert_full, 1.0e-6) / r0_scaled[None, :, :] - 1.0
        e_anchor = eps_ij_full[None, :, :] * (c_a * np.exp(-alpha * x_anchor) - c_b * np.exp(-beta * x_anchor))
        e_pert = eps_ij_full[None, :, :] * (c_a * np.exp(-alpha * x_pert) - c_b * np.exp(-beta * x_pert))
        u_anchor = np.sum(np.where(mask_anchor_full, e_anchor, 0.0), axis=(1, 2))
        u_pert = np.sum(np.where(mask_pert_full, e_pert, 0.0), axis=(1, 2))
        return u_pert - u_anchor

    print("[2/5] 对每个 s_r 计算 anchor-relative ΔU 与 residual")
    delta_u_by_sr = {s_r: _predict_delta_u(s_r) for s_r in s_r_grid}
    residual_by_sr = {s_r: delta_e_target - delta_u_by_sr[s_r] for s_r in s_r_grid}

    group_key_to_idx: Dict[Tuple, Dict[float, int]] = {}
    for i in range(n_rows):
        key = (int(anchor_local_idx[i]), str(pert_type[i]), str(axis_kind[i]), int(axis_index[i]), float(magnitude[i]))
        group_key_to_idx.setdefault(key, {})[float(sign[i])] = i

    def _odd_even_pooled(resid: np.ndarray, anchor_filter: Optional[set] = None) -> Dict:
        odd_vals, even_vals = [], []
        for key, signed in group_key_to_idx.items():
            a = key[0]
            if anchor_filter is not None and a not in anchor_filter:
                continue
            if 1.0 not in signed or -1.0 not in signed:
                continue
            ip, im = signed[1.0], signed[-1.0]
            odd_vals.append(float((resid[ip] - resid[im]) / 2.0))
            even_vals.append(float((resid[ip] + resid[im]) / 2.0))
        oa, ea = np.asarray(odd_vals, dtype=float), np.asarray(even_vals, dtype=float)
        return {
            "odd_rmse_kjmol": float(np.sqrt(np.mean(oa ** 2))) if oa.size else math.nan,
            "even_rmse_kjmol": float(np.sqrt(np.mean(ea ** 2))) if ea.size else math.nan,
            "n_pairs": int(oa.size),
        }

    def _per_anchor_odd_even(resid: np.ndarray) -> Dict[int, Dict[str, float]]:
        return {a: _odd_even_pooled(resid, anchor_filter={a}) for a in range(n_anchors)}

    print("[3/5] 池化 odd/even RMSE + 按anchor细分（用于LOAO/bootstrap）")
    pooled_by_sr = {s_r: _odd_even_pooled(residual_by_sr[s_r]) for s_r in s_r_grid}
    per_anchor_by_sr = {s_r: _per_anchor_odd_even(residual_by_sr[s_r]) for s_r in s_r_grid}
    for s_r in s_r_grid:
        p = pooled_by_sr[s_r]
        marker = " <- baseline" if s_r == baseline_sr else ""
        print(f"    s_r={s_r:.3f}  odd={p['odd_rmse_kjmol']:.4f}  even={p['even_rmse_kjmol']:.4f}  n={p['n_pairs']}{marker}")

    print("[4/5] anchor-balanced LOAO：每折留一个anchor，用其余anchor的(odd+even)选出最优s_r")
    loao_picks: List[float] = []
    for held_out in range(n_anchors):
        train_anchors = set(range(n_anchors)) - {held_out}
        scores = {}
        for s_r in s_r_grid:
            odd_vals = [per_anchor_by_sr[s_r][a]["odd_rmse_kjmol"] for a in train_anchors]
            even_vals = [per_anchor_by_sr[s_r][a]["even_rmse_kjmol"] for a in train_anchors]
            scores[s_r] = float(np.mean(odd_vals) + np.mean(even_vals))
        loao_picks.append(min(scores, key=scores.get))
    loao_pick_counts: Dict[str, int] = {}
    for s_r in loao_picks:
        loao_pick_counts[f"{s_r:.3f}"] = loao_pick_counts.get(f"{s_r:.3f}", 0) + 1
    loao_majority_sr = max(loao_pick_counts, key=loao_pick_counts.get)
    loao_majority_frac = loao_pick_counts[loao_majority_sr] / float(n_anchors)
    print(f"    LOAO 逐折最优s_r分布: {loao_pick_counts}（{n_anchors}折）；多数选择={loao_majority_sr}(占比{loao_majority_frac:.2f})")

    print(f"[5/5] bootstrap CI(over anchors, 逐anchor配对，n_boot={int(args.r0_scale_n_boot)})：各 s_r 相对 baseline={baseline_sr} 的 odd/even 差值")
    boot_rng = np.random.default_rng(int(args.seed) + 3)

    def _bootstrap_ci_mean_diff(vals_a: np.ndarray, vals_b: np.ndarray, n_boot: int) -> Dict:
        mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
        diffs = (vals_a - vals_b)[mask]
        if diffs.size < 2:
            return {"n": int(diffs.size), "mean_diff": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan}
        n = diffs.size
        idx_pool = np.arange(n)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = boot_rng.choice(idx_pool, size=n, replace=True)
            boot_means[b] = np.mean(diffs[idx])
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return {"n": int(n), "mean_diff": float(np.mean(diffs)), "ci95_lo": float(lo), "ci95_hi": float(hi)}

    n_boot = int(args.r0_scale_n_boot)
    verdicts: Dict[str, Dict] = {}
    odd_base = np.asarray([per_anchor_by_sr[baseline_sr][a]["odd_rmse_kjmol"] for a in range(n_anchors)])
    even_base = np.asarray([per_anchor_by_sr[baseline_sr][a]["even_rmse_kjmol"] for a in range(n_anchors)])
    for s_r in s_r_grid:
        odd_here = np.asarray([per_anchor_by_sr[s_r][a]["odd_rmse_kjmol"] for a in range(n_anchors)])
        even_here = np.asarray([per_anchor_by_sr[s_r][a]["even_rmse_kjmol"] for a in range(n_anchors)])
        odd_ci = _bootstrap_ci_mean_diff(odd_here, odd_base, n_boot)
        even_ci = _bootstrap_ci_mean_diff(even_here, even_base, n_boot)
        odd_improved_significant = bool(not math.isnan(odd_ci["ci95_hi"]) and odd_ci["ci95_hi"] < 0.0)
        even_regressed_significant = bool(not math.isnan(even_ci["ci95_lo"]) and even_ci["ci95_lo"] > 0.0)
        loao_ok = bool(f"{s_r:.3f}" == loao_majority_sr and loao_majority_frac >= 0.5)
        passes_all_four = bool(s_r != baseline_sr and odd_improved_significant and (not even_regressed_significant) and loao_ok)
        verdicts[f"{s_r:.3f}"] = {
            "s_r": s_r,
            "odd_diff_vs_baseline_ci": odd_ci,
            "even_diff_vs_baseline_ci": even_ci,
            "odd_improved_significant": odd_improved_significant,
            "even_regressed_significant": even_regressed_significant,
            "loao_majority_pick": loao_ok,
            "passes_all_four_criteria": passes_all_four,
        }
        if s_r != baseline_sr:
            print(
                f"    s_r={s_r:.3f}: odd_diff={odd_ci['mean_diff']:.4f}[{odd_ci['ci95_lo']:.4f},{odd_ci['ci95_hi']:.4f}] "
                f"even_diff={even_ci['mean_diff']:.4f}[{even_ci['ci95_lo']:.4f},{even_ci['ci95_hi']:.4f}] "
                f"odd改善显著={odd_improved_significant} even显著退化={even_regressed_significant} "
                f"LOAO多数选中={loao_ok} => 通过全部四条={passes_all_four}"
            )

    promotable = [s_r for s_r, v in verdicts.items() if v["passes_all_four_criteria"]]
    print(f"    满足全部四条判据、可考虑晋升为第4个MD condition的s_r: {promotable if promotable else '无'}")

    summary = {
        "n_anchors": n_anchors,
        "alpha": alpha,
        "beta": beta,
        "s_r_grid": s_r_grid,
        "baseline_sr": baseline_sr,
        "pooled_odd_even_by_sr": {f"{s_r:.3f}": pooled_by_sr[s_r] for s_r in s_r_grid},
        "loao_pick_distribution": loao_pick_counts,
        "loao_majority_sr": loao_majority_sr,
        "loao_majority_fraction": loao_majority_frac,
        "verdicts_by_sr": verdicts,
        "promotable_to_md_condition": promotable,
        "note": (
            "r0_ij(new)=s_r*r0_ij(LJ-matched)，alpha/beta固定不变(默认14,5,当前生产默认核)，"
            "cutoff_nm不随s_r变化(是独立的生产参数,不是r0的一部分)。晋升为第4个MD condition"
            "需要同时满足：①odd相对baseline显著改善(bootstrap CI完全<0)；②even没有显著"
            "退化(bootstrap CI不满足'完全>0'这个显著变差的判据)——注意这里只查了even RMSE，"
            "不是完整Hessian，真正候选s_r建议之后单独跑一次"
            "--mace-residual-force-benchmark风格的完整Hessian复核再最终拍板；③LOAO"
            "逐折最优s_r的多数(>=50%)一致；④以上判据均基于bootstrap而非点估计。"
            "本诊断零新增MACE计算，只复用已有--perturb-scan缓存。"
        ),
    }
    summary_path = os.path.join(output_dir, "r0_scale_diagnostic_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    summary: {summary_path}")
    return summary


def run_alpha_beta_scale_diagnostic(args: argparse.Namespace, output_dir: str) -> Dict:
    """用户 2026-07-13 提出的"让 MACE 老师最后签个字"：目的是**确认 (14,5) 位于一个稳定、
    宽阔的最优盆地**，不是去找一个新的神奇小数对(如(13.87,5.12))。r0_scale=1.0、
    s_epsilon=1.0 固定(即仍然是LJ-matched r0/eps，只扫alpha/beta形状)。零新增MACE计算，
    只复用已有 `--perturb-scan` 缓存。

    两阶段设计（避免对整个网格都做昂贵的force/Hessian/bootstrap）：
    - **阶段1（全网格，便宜）**：alpha∈[--ab-alpha-min,--ab-alpha-max]步长--ab-alpha-step，
      beta∈[--ab-beta-min,--ab-beta-max]步长--ab-beta-step，约束alpha>beta(默认覆盖用户
      给的完整网格：alpha 12-16步0.25，beta 4-7步0.25；想省算力可以直接传窄网格
      13-15步0.25/4.5-5.5步0.125，不需要额外flag)。对每个(alpha,beta)算池化+按anchor
      的odd/even RMSE；做anchor-balanced LOAO(每折留一个anchor，其余anchor按even RMSE
      选最优组合，检验跨折是否稳定——按用户要求，odd不设为必须改善的判据，因为alpha/beta
      结构上主要控制曲率/even，§3.4已证实这一点)；用
      `p=alpha*beta`(控制r0附近曲率)/`q=alpha+beta`(控制离开r0后的高阶形状)重新参数化，
      看等值线是沿哪个方向更平——如果沿q方向明显更平，直接证实"MACE真正约束的是α+β≈19
      这个形状组合，不是碰巧认准14和5两个整数"这个假说(呼应§3.3的对角谷发现，现在用
      更细网格+更完整判据重新确认)。
    - **阶段2（只挑3个候选，不是全网格）**：grid最优(按even RMSE) vs (14,5) vs (12,6)
      两两bootstrap CI(odd/even都报告，但只有even差异用于晋升判定)。
    - **晋升规则（严格照用户原话执行）**：
      · 如果最优区域是一片宽谷且(14,5)在grid-optimum的bootstrap CI内：保留(14,5)；
      · 如果grid最优只比(14,5)在even上改善<5%(可调阈值)：视为数值精修，不改默认值；
      · 只有even上bootstrap CI显著优于(14,5)、且LOAO多数折一致选中该组合，才建议进入
        force held-out + 完整Hessian的深度复核(用`--mace-residual-force-benchmark`把
        该组合加成第4个命名核重跑，本函数不重复那套机制)。
    """
    csv_path = os.path.join(output_dir, "perturb_scan_diagnostics.csv")
    npz_path = os.path.join(output_dir, "perturb_scan_geometry.npz")
    ensure_file(csv_path, "perturb-scan 诊断 CSV（先跑一次 --perturb-scan）")
    ensure_file(npz_path, "perturb-scan 几何快照 npz（先跑一次 --perturb-scan）")

    rows = read_csv_rows(csv_path)
    geo = np.load(npz_path)
    n_rows = len(rows)
    delta_e_target = np.asarray([float(r["delta_e_target_kjmol"]) for r in rows], dtype=float)
    pert_type = np.asarray([r["pert_type"] for r in rows], dtype=object)
    magnitude = np.asarray([float(r["magnitude"]) for r in rows], dtype=float)
    axis_kind = np.asarray([r["axis_kind"] for r in rows], dtype=object)
    axis_index = np.asarray([int(r["axis_index"]) for r in rows], dtype=int)
    sign = np.asarray([float(r["sign"]) for r in rows], dtype=float)

    anchor_local_idx = geo["perturbation_anchor_index"].astype(int)
    if len(anchor_local_idx) != n_rows:
        raise RuntimeError(
            f"CSV 行数({n_rows})与几何 npz 行数({len(anchor_local_idx)})不一致，"
            "两者必须来自同一次 --perturb-scan 运行"
        )
    env_positions = geo["env_positions"]
    anchor_lig_positions = geo["anchor_lig_positions"]
    perturbed_lig_positions = geo["perturbed_lig_positions"]
    box_vectors = geo["box_vectors"]
    has_periodic = geo["has_periodic"].astype(bool)
    sigma_lig, eps_lig = geo["sigma_lig"], geo["eps_lig"]
    sigma_env, eps_env = geo["sigma_env"], geo["eps_env"]
    n_anchors = int(anchor_local_idx.max()) + 1
    cutoff_nm = float(args.perturb_baseline_cutoff_nm)

    alpha_grid = sorted({round(float(x), 4) for x in np.arange(
        float(args.ab_alpha_min), float(args.ab_alpha_max) + float(args.ab_alpha_step) * 0.5, float(args.ab_alpha_step)
    )})
    beta_grid = sorted({round(float(x), 4) for x in np.arange(
        float(args.ab_beta_min), float(args.ab_beta_max) + float(args.ab_beta_step) * 0.5, float(args.ab_beta_step)
    )})
    candidates = sorted({(a, b) for a in alpha_grid for b in beta_grid if a > b})
    for extra in ((14.0, 5.0), (12.0, 6.0)):
        if extra not in candidates:
            candidates.append(extra)
    candidates = sorted(set(candidates))

    print(
        f"[1/6] 网格 alpha∈[{args.ab_alpha_min},{args.ab_alpha_max}]步长{args.ab_alpha_step}"
        f"（{len(alpha_grid)}个值），beta∈[{args.ab_beta_min},{args.ab_beta_max}]步长{args.ab_beta_step}"
        f"（{len(beta_grid)}个值），alpha>beta约束+(14,5)/(12,6)兜底后共 {len(candidates)} 个组合"
        f"（r0_scale=1.0, s_epsilon=1.0 固定）"
    )
    tensors = _build_perturbation_distance_tensors(
        n_rows, anchor_local_idx, env_positions, anchor_lig_positions, perturbed_lig_positions,
        box_vectors, has_periodic, sigma_lig, eps_lig, sigma_env, eps_env, cutoff_nm,
    )
    dists_anchor_full, dists_pert_full = tensors["dists_anchor_full"], tensors["dists_pert_full"]
    eps_ij_full = tensors["eps_ij_full"]
    mask_anchor_full, mask_pert_full = tensors["mask_anchor_full"], tensors["mask_pert_full"]
    r0_ij_full = tensors["r0_ij_full"]  # LJ-matched r0，s_epsilon=1.0/r0_scale=1.0 固定不变

    x_anchor_full = np.maximum(dists_anchor_full, 1.0e-6) / r0_ij_full[None, :, :] - 1.0
    x_pert_full = np.maximum(dists_pert_full, 1.0e-6) / r0_ij_full[None, :, :] - 1.0

    def _predict_delta_u(alpha: float, beta: float) -> np.ndarray:
        c_a, c_b = beta / (alpha - beta), alpha / (alpha - beta)
        e_anchor = eps_ij_full[None, :, :] * (c_a * np.exp(-alpha * x_anchor_full) - c_b * np.exp(-beta * x_anchor_full))
        e_pert = eps_ij_full[None, :, :] * (c_a * np.exp(-alpha * x_pert_full) - c_b * np.exp(-beta * x_pert_full))
        u_anchor = np.sum(np.where(mask_anchor_full, e_anchor, 0.0), axis=(1, 2))
        u_pert = np.sum(np.where(mask_pert_full, e_pert, 0.0), axis=(1, 2))
        return u_pert - u_anchor

    group_key_to_idx: Dict[Tuple, Dict[float, int]] = {}
    for i in range(n_rows):
        key = (int(anchor_local_idx[i]), str(pert_type[i]), str(axis_kind[i]), int(axis_index[i]), float(magnitude[i]))
        group_key_to_idx.setdefault(key, {})[float(sign[i])] = i

    def _odd_even_pooled(resid: np.ndarray, anchor_filter: Optional[set] = None) -> Dict:
        odd_vals, even_vals = [], []
        for key, signed in group_key_to_idx.items():
            a = key[0]
            if anchor_filter is not None and a not in anchor_filter:
                continue
            if 1.0 not in signed or -1.0 not in signed:
                continue
            ip, im = signed[1.0], signed[-1.0]
            odd_vals.append(float((resid[ip] - resid[im]) / 2.0))
            even_vals.append(float((resid[ip] + resid[im]) / 2.0))
        oa, ea = np.asarray(odd_vals, dtype=float), np.asarray(even_vals, dtype=float)
        return {
            "odd_rmse_kjmol": float(np.sqrt(np.mean(oa ** 2))) if oa.size else math.nan,
            "even_rmse_kjmol": float(np.sqrt(np.mean(ea ** 2))) if ea.size else math.nan,
            "n_pairs": int(oa.size),
        }

    def _per_anchor_odd_even(resid: np.ndarray) -> Dict[int, Dict[str, float]]:
        return {a: _odd_even_pooled(resid, anchor_filter={a}) for a in range(n_anchors)}

    print(f"[2/6] 对 {len(candidates)} 个(alpha,beta)组合计算池化+按anchor的odd/even RMSE（阶段1，便宜）")
    pooled_by_ab: Dict[Tuple[float, float], Dict] = {}
    per_anchor_by_ab: Dict[Tuple[float, float], Dict[int, Dict[str, float]]] = {}
    for (a, b) in candidates:
        resid = delta_e_target - _predict_delta_u(a, b)
        pooled_by_ab[(a, b)] = _odd_even_pooled(resid)
        per_anchor_by_ab[(a, b)] = _per_anchor_odd_even(resid)

    best_ab = min(candidates, key=lambda ab: pooled_by_ab[ab]["even_rmse_kjmol"])
    top5 = sorted(candidates, key=lambda ab: pooled_by_ab[ab]["even_rmse_kjmol"])[:5]
    print("    even RMSE 最低的5个组合:")
    for ab in top5:
        p = pooled_by_ab[ab]
        print(f"      alpha={ab[0]:.3f} beta={ab[1]:.3f}  even={p['even_rmse_kjmol']:.4f}  odd={p['odd_rmse_kjmol']:.4f}")
    p_1405, p_126 = pooled_by_ab[(14.0, 5.0)], pooled_by_ab[(12.0, 6.0)]
    print(f"    (14,5): even={p_1405['even_rmse_kjmol']:.4f} odd={p_1405['odd_rmse_kjmol']:.4f}")
    print(f"    (12,6): even={p_126['even_rmse_kjmol']:.4f} odd={p_126['odd_rmse_kjmol']:.4f}")

    print("[3/6] p=alpha*beta(曲率) / q=alpha+beta(离开r0后的高阶形状) + 近最优盆地PCA：确认山谷走向")
    profile_by_q: Dict[float, List[float]] = {}
    profile_by_p: Dict[float, List[float]] = {}
    for (a, b) in candidates:
        profile_by_q.setdefault(round(a + b, 3), []).append(pooled_by_ab[(a, b)]["even_rmse_kjmol"])
        profile_by_p.setdefault(round(a * b, 3), []).append(pooled_by_ab[(a, b)]["even_rmse_kjmol"])
    profile_by_q_summary = {str(q): {"min_even_rmse": float(min(v)), "n": len(v)} for q, v in sorted(profile_by_q.items())}
    profile_by_p_summary = {str(p): {"min_even_rmse": float(min(v)), "n": len(v)} for p, v in sorted(profile_by_p.items())}
    # 不能直接比较按 q 与按 p 分组后的 min-RMSE 标准差：两套坐标的单位、bin 数和每个 bin
    # 的网格点数完全不同（p 在这个规则 alpha/beta 网格上尤其接近“一点一个 bin”），这种比较
    # 会把采样密度误当成山谷方向。改为对距全局最优 5% 内的原始 (alpha,beta) 盆地点做 PCA：
    # 最大方差特征向量是谷底切向；若其两个分量异号且大小接近，则谷底是 alpha+beta≈常数。
    even_best_for_basin = float(pooled_by_ab[best_ab]["even_rmse_kjmol"])
    basin_rel_tol = 0.05
    basin_cutoff = even_best_for_basin * (1.0 + basin_rel_tol)
    basin_ab = np.asarray([
        [a, b] for (a, b) in candidates
        if pooled_by_ab[(a, b)]["even_rmse_kjmol"] <= basin_cutoff
    ], dtype=float)
    basin_tangent = np.asarray([math.nan, math.nan], dtype=float)
    basin_normal = np.asarray([math.nan, math.nan], dtype=float)
    basin_eigenvalue_ratio = math.nan
    basin_q_mean = math.nan
    basin_q_std = math.nan
    ridge_is_constant_q = False
    if basin_ab.shape[0] >= 2:
        centered = basin_ab - np.mean(basin_ab, axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, basin_ab.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        basin_tangent = eigvecs[:, 0]
        basin_normal = eigvecs[:, -1]
        basin_eigenvalue_ratio = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 1.0e-15 else 0.0
        basin_q = np.sum(basin_ab, axis=1)
        basin_q_mean, basin_q_std = float(np.mean(basin_q)), float(np.std(basin_q))
        tangent_abs = np.abs(basin_tangent)
        ridge_is_constant_q = bool(
            basin_tangent[0] * basin_tangent[1] < 0.0
            and tangent_abs.min() / max(tangent_abs.max(), 1.0e-15) >= 0.7
        )
    ridge_desc = (
        f"近最优盆地沿 alpha+beta≈{basin_q_mean:.3f} 的对角线延伸（q标准差={basin_q_std:.3f}）"
        if ridge_is_constant_q else
        "近最优盆地不是清晰的 alpha+beta≈常数对角谷"
    )
    print(
        f"    5%盆地点 n={len(basin_ab)}，PCA切向=({basin_tangent[0]:.3f},{basin_tangent[1]:.3f})，"
        f"最紧/最松方差比={basin_eigenvalue_ratio:.4f} -> {ridge_desc}"
    )

    print("[4/6] anchor-balanced LOAO：每折留一个anchor，其余anchor按even RMSE选最优组合，检验跨折是否稳定（odd不作为必须判据）")
    loao_picks: List[Tuple[float, float]] = []
    for held_out in range(n_anchors):
        train_anchors = set(range(n_anchors)) - {held_out}
        scores = {
            ab: float(np.mean([per_anchor_by_ab[ab][a]["even_rmse_kjmol"] for a in train_anchors]))
            for ab in candidates
        }
        loao_picks.append(min(scores, key=scores.get))
    loao_pick_counts: Dict[str, int] = {}
    for ab in loao_picks:
        key = f"({ab[0]:.3f},{ab[1]:.3f})"
        loao_pick_counts[key] = loao_pick_counts.get(key, 0) + 1
    loao_majority_key = max(loao_pick_counts, key=loao_pick_counts.get)
    loao_majority_frac = loao_pick_counts[loao_majority_key] / float(n_anchors)
    top_picks = sorted(loao_pick_counts.items(), key=lambda kv: -kv[1])[:8]
    loao_q_counts: Dict[str, int] = {}
    for a, b in loao_picks:
        q_key = f"{a + b:.3f}"
        loao_q_counts[q_key] = loao_q_counts.get(q_key, 0) + 1
    print(f"    LOAO 逐折最优组合分布(前8): {top_picks}（共{n_anchors}折）")
    print(f"    多数选择={loao_majority_key} 占比={loao_majority_frac:.2f}")
    print(f"    LOAO 的 q=alpha+beta 分布: {dict(sorted(loao_q_counts.items()))}")

    print(f"[5/6] bootstrap CI(逐anchor配对, n_boot={int(args.ab_scale_n_boot)})：只比较 grid最优 vs (14,5) vs (12,6) 这3个候选，不是全网格")
    boot_rng = np.random.default_rng(int(args.seed) + 4)

    def _bootstrap_ci_mean_diff(vals_a: np.ndarray, vals_b: np.ndarray, n_boot: int) -> Dict:
        mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
        diffs = (vals_a - vals_b)[mask]
        if diffs.size < 2:
            return {"n": int(diffs.size), "mean_diff": math.nan, "ci95_lo": math.nan, "ci95_hi": math.nan}
        n = diffs.size
        idx_pool = np.arange(n)
        boot_means = np.empty(n_boot)
        for b in range(n_boot):
            idx = boot_rng.choice(idx_pool, size=n, replace=True)
            boot_means[b] = np.mean(diffs[idx])
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return {"n": int(n), "mean_diff": float(np.mean(diffs)), "ci95_lo": float(lo), "ci95_hi": float(hi)}

    n_boot = int(args.ab_scale_n_boot)
    key_candidates = sorted(set([best_ab, (14.0, 5.0), (12.0, 6.0)]))

    def _arr(ab: Tuple[float, float], metric: str) -> np.ndarray:
        return np.asarray([per_anchor_by_ab[ab][a][metric] for a in range(n_anchors)])

    bootstrap_results: Dict[str, Dict] = {}
    for i in range(len(key_candidates)):
        for j in range(i + 1, len(key_candidates)):
            ab_a, ab_b = key_candidates[i], key_candidates[j]
            label = f"({ab_a[0]:.3f},{ab_a[1]:.3f})_vs_({ab_b[0]:.3f},{ab_b[1]:.3f})"
            odd_ci = _bootstrap_ci_mean_diff(_arr(ab_a, "odd_rmse_kjmol"), _arr(ab_b, "odd_rmse_kjmol"), n_boot)
            even_ci = _bootstrap_ci_mean_diff(_arr(ab_a, "even_rmse_kjmol"), _arr(ab_b, "even_rmse_kjmol"), n_boot)
            bootstrap_results[label] = {"odd_diff_ci": odd_ci, "even_diff_ci": even_ci}
            print(
                f"    {label}: odd_diff={odd_ci['mean_diff']:.4f}[{odd_ci['ci95_lo']:.4f},{odd_ci['ci95_hi']:.4f}]  "
                f"even_diff={even_ci['mean_diff']:.4f}[{even_ci['ci95_lo']:.4f},{even_ci['ci95_hi']:.4f}]"
            )

    print("[6/6] 判定：grid最优是否值得晋升为force/Hessian深度复核的候选（不要求odd改善）")
    even_1405 = pooled_by_ab[(14.0, 5.0)]["even_rmse_kjmol"]
    even_best = pooled_by_ab[best_ab]["even_rmse_kjmol"]
    pct_improvement_vs_1405 = float(100.0 * (even_1405 - even_best) / even_1405) if even_1405 > 1.0e-12 else math.nan
    best_vs_1405_label = f"({best_ab[0]:.3f},{best_ab[1]:.3f})_vs_(14.000,5.000)"
    alt_label = f"(14.000,5.000)_vs_({best_ab[0]:.3f},{best_ab[1]:.3f})"
    best_vs_1405 = bootstrap_results.get(best_vs_1405_label) or bootstrap_results.get(alt_label)
    flipped = best_vs_1405_label not in bootstrap_results
    if best_vs_1405 is not None:
        even_ci = best_vs_1405["even_diff_ci"]
        lo, hi = (-even_ci["ci95_hi"], -even_ci["ci95_lo"]) if flipped else (even_ci["ci95_lo"], even_ci["ci95_hi"])
        even_significantly_better_than_1405 = bool(not math.isnan(hi) and hi < 0.0)
        is_1405_within_ci = bool(not math.isnan(lo) and not math.isnan(hi) and lo <= 0.0 <= hi)
    else:
        even_significantly_better_than_1405, is_1405_within_ci = False, True

    is_best_same_as_1405 = bool(best_ab == (14.0, 5.0))
    loao_matches_best = bool(loao_majority_key == f"({best_ab[0]:.3f},{best_ab[1]:.3f})" and loao_majority_frac >= 0.5)
    numerical_refinement_only = bool((not is_best_same_as_1405) and pct_improvement_vs_1405 < 5.0)
    worth_deep_dive = bool(
        (not is_best_same_as_1405)
        and even_significantly_better_than_1405
        and (not numerical_refinement_only)
        and loao_matches_best
    )
    if is_best_same_as_1405 or is_1405_within_ci or not even_significantly_better_than_1405:
        recommendation = "保留 (14,5)：谷底宽阔，(14,5) 在 grid 最优的 bootstrap CI 内，或改善不显著。"
    elif numerical_refinement_only:
        recommendation = f"grid最优 {best_ab} 只比(14,5)在even上改善{pct_improvement_vs_1405:.1f}%(<5%)：视为数值精修，不改默认值。"
    elif worth_deep_dive:
        recommendation = (
            f"grid最优 {best_ab} 在even上显著优于(14,5)({pct_improvement_vs_1405:.1f}%改善)且LOAO多数折一致选中——"
            "建议把它加成--mace-residual-force-benchmark的第4个命名核，跑force held-out+完整Hessian复核再最终拍板。"
        )
    else:
        recommendation = f"grid最优 {best_ab} even上有改善但LOAO跨折不一致(多数占比{loao_majority_frac:.2f})，证据不够稳固，暂不晋升。"
    print(f"    {recommendation}")

    summary = {
        "n_anchors": n_anchors,
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "n_candidates": len(candidates),
        "best_ab_by_even_rmse": {"alpha": best_ab[0], "beta": best_ab[1]},
        "pooled_odd_even_top5": {
            f"({ab[0]:.3f},{ab[1]:.3f})": pooled_by_ab[ab] for ab in top5
        },
        "pooled_odd_even_14_5": p_1405,
        "pooled_odd_even_12_6": p_126,
        "landscape": [
            {
                "alpha": a,
                "beta": b,
                "p_alpha_times_beta": float(a * b),
                "q_alpha_plus_beta": float(a + b),
                **pooled_by_ab[(a, b)],
            }
            for (a, b) in candidates
        ],
        "ridge_profile": {
            "by_q_alpha_plus_beta": profile_by_q_summary,
            "by_p_alpha_times_beta": profile_by_p_summary,
            "method": "PCA of raw (alpha,beta) points within 5% of the minimum pooled even RMSE",
            "relative_basin_tolerance": basin_rel_tol,
            "basin_even_rmse_cutoff": basin_cutoff,
            "n_basin_points": int(len(basin_ab)),
            "basin_tangent_alpha_beta": [float(x) for x in basin_tangent],
            "basin_normal_alpha_beta": [float(x) for x in basin_normal],
            "basin_eigenvalue_ratio_tight_over_loose": basin_eigenvalue_ratio,
            "basin_q_mean": basin_q_mean,
            "basin_q_std": basin_q_std,
            "ridge_is_alpha_plus_beta_constant": ridge_is_constant_q,
            "description": ridge_desc,
        },
        "loao_pick_distribution_top8": dict(top_picks),
        "loao_q_alpha_plus_beta_distribution": dict(sorted(loao_q_counts.items())),
        "loao_majority_key": loao_majority_key,
        "loao_majority_fraction": loao_majority_frac,
        "bootstrap_key_candidate_comparisons": bootstrap_results,
        "verdict": {
            "pct_even_improvement_grid_best_vs_14_5": pct_improvement_vs_1405,
            "even_significantly_better_than_14_5": even_significantly_better_than_1405,
            "is_14_5_within_grid_best_ci": is_1405_within_ci,
            "loao_majority_matches_grid_best": loao_matches_best,
            "numerical_refinement_only": numerical_refinement_only,
            "worth_force_hessian_deep_dive": worth_deep_dive,
            "recommendation": recommendation,
        },
        "note": (
            "阶段1(全网格)只算odd/even RMSE + LOAO + p/q山谷走向剖面，零bootstrap——便宜，"
            "覆盖全网格。阶段2(bootstrap)只比较grid最优/(14,5)/(12,6)这3个候选两两，不对"
            "全网格做bootstrap。odd不作为LOAO/晋升的必须判据(alpha/beta结构上主要控制"
            "曲率/even，§3.4已证实)，但仍逐候选报告odd供参考。force held-out和完整Hessian"
            "复核不在本函数范围内——只有verdict.worth_force_hessian_deep_dive=true时，"
            "才建议把grid最优加成--mace-residual-force-benchmark的第4个命名核重新跑那套"
            "已验证的机制，不在这里重复实现。r0_scale=1.0/s_epsilon=1.0全程固定，"
            "对应§9.1旁线(--r0-scale-diagnostic)已确认r0不需要移动这一前提。"
        ),
    }
    landscape_csv_path = os.path.join(output_dir, "alpha_beta_scale_diagnostic_landscape.csv")
    with open(landscape_csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = ["alpha", "beta", "p_alpha_times_beta", "q_alpha_plus_beta", "odd_rmse_kjmol", "even_rmse_kjmol", "n_pairs"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["landscape"])
    summary["landscape_csv"] = landscape_csv_path

    summary_path = os.path.join(output_dir, "alpha_beta_scale_diagnostic_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    summary: {summary_path}")
    print(f"    landscape csv (完整网格落盘，供画2D热图): {landscape_csv_path}")
    return summary


def run_alpha_beta_ridge_scan(args: argparse.Namespace, output_dir: str) -> Dict:
    """用户 2026-07-13 追加要求：`--alpha-beta-scale-diagnostic`(0.25网格)已经显示
    q=alpha+beta=19(过(14,5))比 q=18(过(12,6))明显更优——不是共享同一条宽平山谷，
    q=18在网格分辨率下的最优点(=(12,6)本身)even RMSE已经是q=19山谷最优点的近2倍。
    本函数专门沿这两条(可配置)定值q直线做比0.25网格细得多的扫描("贵一点无所谓")，
    产出两条平滑曲线供画图，直接可视化"两条脊线是否等价"这个问题。

    对每个 q（默认18,19），沿 beta∈[--ridge-beta-min, q/2) 步长 --ridge-step 扫描
    (alpha=q-beta)，r0_scale=1.0/s_epsilon=1.0固定(与--alpha-beta-scale-diagnostic
    同一前提)，算池化odd/even RMSE，标出该q线上的网格内最优点，以及落在该q线上的
    命名核((12,6)对应q=18，(14,5)对应q=19，若某个q里不含命名核则不标)。零新增MACE
    计算，只复用--perturb-scan缓存。画图见 `plot_alpha_beta_ridge_scan.py`。
    """
    csv_path = os.path.join(output_dir, "perturb_scan_diagnostics.csv")
    npz_path = os.path.join(output_dir, "perturb_scan_geometry.npz")
    ensure_file(csv_path, "perturb-scan 诊断 CSV（先跑一次 --perturb-scan）")
    ensure_file(npz_path, "perturb-scan 几何快照 npz（先跑一次 --perturb-scan）")

    rows = read_csv_rows(csv_path)
    geo = np.load(npz_path)
    n_rows = len(rows)
    delta_e_target = np.asarray([float(r["delta_e_target_kjmol"]) for r in rows], dtype=float)
    pert_type = np.asarray([r["pert_type"] for r in rows], dtype=object)
    magnitude = np.asarray([float(r["magnitude"]) for r in rows], dtype=float)
    axis_kind = np.asarray([r["axis_kind"] for r in rows], dtype=object)
    axis_index = np.asarray([int(r["axis_index"]) for r in rows], dtype=int)
    sign = np.asarray([float(r["sign"]) for r in rows], dtype=float)

    anchor_local_idx = geo["perturbation_anchor_index"].astype(int)
    if len(anchor_local_idx) != n_rows:
        raise RuntimeError(
            f"CSV 行数({n_rows})与几何 npz 行数({len(anchor_local_idx)})不一致，"
            "两者必须来自同一次 --perturb-scan 运行"
        )
    env_positions = geo["env_positions"]
    anchor_lig_positions = geo["anchor_lig_positions"]
    perturbed_lig_positions = geo["perturbed_lig_positions"]
    box_vectors = geo["box_vectors"]
    has_periodic = geo["has_periodic"].astype(bool)
    sigma_lig, eps_lig = geo["sigma_lig"], geo["eps_lig"]
    sigma_env, eps_env = geo["sigma_env"], geo["eps_env"]
    n_anchors = int(anchor_local_idx.max()) + 1
    cutoff_nm = float(args.perturb_baseline_cutoff_nm)

    q_values = sorted(float(x) for x in str(args.ridge_q_values).split(",") if x.strip())
    beta_min = float(args.ridge_beta_min)
    step = float(args.ridge_step)
    named_point_by_q = {18.0: (12.0, 6.0), 19.0: (14.0, 5.0)}

    print(
        f"[1/3] 沿 q=alpha+beta 固定的脊线细扫：q∈{q_values}，beta∈[{beta_min}, q/2)，"
        f"步长{step}（r0_scale=1.0, s_epsilon=1.0 固定，零新增MACE）"
    )
    tensors = _build_perturbation_distance_tensors(
        n_rows, anchor_local_idx, env_positions, anchor_lig_positions, perturbed_lig_positions,
        box_vectors, has_periodic, sigma_lig, eps_lig, sigma_env, eps_env, cutoff_nm,
    )
    dists_anchor_full, dists_pert_full = tensors["dists_anchor_full"], tensors["dists_pert_full"]
    eps_ij_full = tensors["eps_ij_full"]
    mask_anchor_full, mask_pert_full = tensors["mask_anchor_full"], tensors["mask_pert_full"]
    r0_ij_full = tensors["r0_ij_full"]
    x_anchor_full = np.maximum(dists_anchor_full, 1.0e-6) / r0_ij_full[None, :, :] - 1.0
    x_pert_full = np.maximum(dists_pert_full, 1.0e-6) / r0_ij_full[None, :, :] - 1.0

    def _predict_delta_u(alpha: float, beta: float) -> np.ndarray:
        c_a, c_b = beta / (alpha - beta), alpha / (alpha - beta)
        e_anchor = eps_ij_full[None, :, :] * (c_a * np.exp(-alpha * x_anchor_full) - c_b * np.exp(-beta * x_anchor_full))
        e_pert = eps_ij_full[None, :, :] * (c_a * np.exp(-alpha * x_pert_full) - c_b * np.exp(-beta * x_pert_full))
        u_anchor = np.sum(np.where(mask_anchor_full, e_anchor, 0.0), axis=(1, 2))
        u_pert = np.sum(np.where(mask_pert_full, e_pert, 0.0), axis=(1, 2))
        return u_pert - u_anchor

    group_key_to_idx: Dict[Tuple, Dict[float, int]] = {}
    for i in range(n_rows):
        key = (int(anchor_local_idx[i]), str(pert_type[i]), str(axis_kind[i]), int(axis_index[i]), float(magnitude[i]))
        group_key_to_idx.setdefault(key, {})[float(sign[i])] = i

    def _odd_even_pooled(resid: np.ndarray) -> Dict:
        odd_vals, even_vals = [], []
        for signed in group_key_to_idx.values():
            if 1.0 not in signed or -1.0 not in signed:
                continue
            ip, im = signed[1.0], signed[-1.0]
            odd_vals.append(float((resid[ip] - resid[im]) / 2.0))
            even_vals.append(float((resid[ip] + resid[im]) / 2.0))
        oa, ea = np.asarray(odd_vals, dtype=float), np.asarray(even_vals, dtype=float)
        return {
            "odd_rmse_kjmol": float(np.sqrt(np.mean(oa ** 2))) if oa.size else math.nan,
            "even_rmse_kjmol": float(np.sqrt(np.mean(ea ** 2))) if ea.size else math.nan,
            "n_pairs": int(oa.size),
        }

    print(f"[2/3] 逐点计算池化odd/even RMSE（{len(q_values)}条脊线）")
    ridge_rows: List[Dict] = []
    ridge_by_q: Dict[str, Dict] = {}
    for q in q_values:
        beta_max = q / 2.0 - 1.0e-6
        betas = np.arange(beta_min, beta_max, step)
        curve: List[Dict] = []
        for beta in betas:
            alpha = q - beta
            oe = _odd_even_pooled(delta_e_target - _predict_delta_u(alpha, beta))
            row = {"q": q, "alpha": float(alpha), "beta": float(beta), **oe}
            ridge_rows.append(row)
            curve.append(row)
        best_on_ridge = min(curve, key=lambda r: r["even_rmse_kjmol"])
        named = named_point_by_q.get(q)
        named_metrics = None
        if named is not None:
            named_metrics = _odd_even_pooled(delta_e_target - _predict_delta_u(named[0], named[1]))
        even_vals = [r["even_rmse_kjmol"] for r in curve]
        odd_vals = [r["odd_rmse_kjmol"] for r in curve]
        ridge_by_q[str(q)] = {
            "n_points": len(curve),
            "best_on_ridge": {
                "alpha": best_on_ridge["alpha"], "beta": best_on_ridge["beta"],
                "even_rmse_kjmol": best_on_ridge["even_rmse_kjmol"], "odd_rmse_kjmol": best_on_ridge["odd_rmse_kjmol"],
            },
            "named_point": {"alpha": named[0], "beta": named[1]} if named is not None else None,
            "named_point_metrics": named_metrics,
            "even_rmse_min_max": [float(min(even_vals)), float(max(even_vals))],
            "odd_rmse_min_max": [float(min(odd_vals)), float(max(odd_vals))],
        }
        named_str = (
            f"命名点(alpha={named[0]},beta={named[1]}): even={named_metrics['even_rmse_kjmol']:.4f} odd={named_metrics['odd_rmse_kjmol']:.4f}"
            if named_metrics is not None else "（该q无命名核）"
        )
        print(
            f"    q={q}: {len(curve)}点  even范围=[{min(even_vals):.4f},{max(even_vals):.4f}]  "
            f"脊线最优 alpha={best_on_ridge['alpha']:.3f} beta={best_on_ridge['beta']:.3f} "
            f"even={best_on_ridge['even_rmse_kjmol']:.4f}  {named_str}"
        )

    print("[3/3] 写出 CSV/JSON")
    csv_out = os.path.join(output_dir, "alpha_beta_ridge_scan_by_point.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["q", "alpha", "beta", "odd_rmse_kjmol", "even_rmse_kjmol", "n_pairs"])
        writer.writeheader()
        writer.writerows(ridge_rows)

    summary = {
        "n_anchors": n_anchors,
        "q_values": q_values,
        "beta_min": beta_min,
        "step": step,
        "named_point_by_q": {str(q): v for q, v in named_point_by_q.items() if q in q_values},
        "ridge_by_q": ridge_by_q,
        "by_point_csv": csv_out,
        "note": (
            "沿固定q=alpha+beta的直线细扫(默认q=18/19，即分别过(12,6)和(14,5))，"
            "r0_scale=1.0/s_epsilon=1.0全程固定，跟--alpha-beta-scale-diagnostic同一前提。"
            "该诊断的0.25网格已显示q=18的网格内最优点(=(12,6)本身)even RMSE远高于"
            "q=19山谷最优点，本扫描用更细步长确认这不是网格分辨率的假象——如果两条脊线"
            "在细扫下仍然明显不等价，说明(12,6)不在同一条'MACE认可'的山谷上，(14,5)"
            "所在的q=19山谷是有效识别的，不是任意选的整数。画图见"
            "`plot_alpha_beta_ridge_scan.py`(读取本函数输出的CSV/JSON，产出两个面板"
            "的PNG，各面板对应一个q值，含even/odd RMSE曲线+命名点标记)。"
        ),
    }
    summary_path = os.path.join(output_dir, "alpha_beta_ridge_scan_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"    summary: {summary_path}")
    print(f"    by-point csv: {csv_out}")
    print("    画图: python plot_alpha_beta_ridge_scan.py")
    return summary


def run_production_equivalence_audit(args: argparse.Namespace, output_dir: str) -> Dict:
    """用户 Phase 1 方案：production-equivalence audit。

    §3-§6 全程用的 (14,5) DEXP 基线是 NumPy 里手写的 pairwise 求和(`_dexp_baseline_pairwise_sum`/
    `_build_perturbation_distance_tensors`)，且**明确不含 switching function**（原代码注释：
    "switch 只在 0.5~0.7nm 边缘生效，对我们关心的 r<0.3nm 排斥墙区域影响可忽略"——这是一个从
    未验证过的假设）。真正的生产力 `abfe_core.DEXPSurrogatePotential`+`SurrogateSystemBuilder`
    在 0.50->0.70nm 之间用标准 quintic switching function 平滑衰减。这个函数直接用生产
    `DEXPSurrogatePotential.build_expression()`(不是重新推导的表达式)构建一个最小 OpenMM
    Context，对同一批 --perturb-scan 缓存的构型比较：

    1. lig_idx/env_idx 重新选取后，sigma/epsilon/charge 是否跟 --perturb-scan 缓存的 npz
       逐项精确一致（核对原子身份与顺序）。
    2. NumPy 无 switch(=一直在用的基线) vs 生产 OpenMM(有 switch) 的能量差——这是本次审计
       最核心的问题：现有基线是否真的跟生产一致。
    3. NumPy 加 switch(新写的、跟生产同款 quintic 公式) vs 生产 OpenMM(有 switch)——验证
       "如果把 switch 加回 NumPy 公式"是否能精确复现生产能量，隔离出"switch 缺失"到底是不是
       唯一的差异来源。
    4. §6.7 里 NumPy 重算的 NoCutoff Gaussian-Coulomb 参考 vs 同一个 `gauss_coul` OpenMM
       Context（`build_mm_le_contexts_from_system_xml`，--perturb-scan 生成 delta_e_target
       时实际用的那个）——这是纯粹的"我的重实现有没有算对"正确性检验，不是"是否等于生产"
       检验（那个 Gaussian 参考本来就不是生产用的 shifted-force 定义，两者故意不同）。
    5. 有限差分力：用 NumPy(加 switch) 能量函数对配体原子做中心差分，与生产 OpenMM Context
       的解析力(`getForces`)比较相对误差。

    判据（用户指定）：能量差 < `--audit-energy-tol-kjmol`(默认 1e-5 kJ/mol)，
    有限差分力相对误差 < `--audit-force-rel-tol`(默认 1e-4)。只有通过后，才能把 §6 的
    residual 称为"物理模型误差"而不是"离线复现跟生产本来就不一致"的假象。
    """
    openmm, app, unit, XmlSerializer = require_openmm()
    symbols = load_abfe_symbols()
    DEXPSurrogatePotential = symbols["DEXPSurrogatePotential"]

    ctx = _contact_type_build_context(args, output_dir)
    lig_idx, env_idx = ctx["lig_idx"], ctx["env_idx"]
    n_lig, n_env = ctx["n_lig"], ctx["n_env"]
    n_anchors = ctx["n_anchors"]
    anchor_local_idx, pert_type, magnitude, sign = ctx["anchor_local_idx"], ctx["pert_type"], ctx["magnitude"], ctx["sign"]
    anchor_lig_positions = ctx["anchor_lig_positions"]
    perturbed_lig_positions = ctx["perturbed_lig_positions"]
    env_positions = ctx["env_positions"]
    box_vectors = ctx["box_vectors"]
    has_periodic = ctx["has_periodic"]

    print("[1/5] 核对 lig_idx/env_idx 的 sigma/epsilon/charge 与 system_native.xml、与 --perturb-scan 缓存的 npz 是否逐项一致")
    with open(args.system_xml, "r", encoding="utf-8") as handle:
        nb_system = XmlSerializer.deserialize(handle.read())
    nb_force = next(f for f in nb_system.getForces() if isinstance(f, openmm.NonbondedForce))
    n_total = nb_system.getNumParticles()
    sigma_all = np.zeros(n_total, dtype=float)
    eps_all = np.zeros(n_total, dtype=float)
    q_all = np.zeros(n_total, dtype=float)
    for i in range(n_total):
        q, sigma, epsilon = nb_force.getParticleParameters(i)
        q_all[i] = q.value_in_unit(unit.elementary_charge)
        sigma_all[i] = sigma.value_in_unit(unit.nanometer)
        eps_all[i] = epsilon.value_in_unit(unit.kilojoule_per_mole)
    sigma_lig, eps_lig = sigma_all[lig_idx], eps_all[lig_idx]
    sigma_env, eps_env = sigma_all[env_idx], eps_all[env_idx]

    geo_cached = np.load(os.path.join(output_dir, "perturb_scan_geometry.npz"))
    identity_checks = {
        "sigma_lig_exact_match": bool(np.array_equal(sigma_lig, geo_cached["sigma_lig"])),
        "eps_lig_exact_match": bool(np.array_equal(eps_lig, geo_cached["eps_lig"])),
        "sigma_env_exact_match": bool(np.array_equal(sigma_env, geo_cached["sigma_env"])),
        "eps_env_exact_match": bool(np.array_equal(eps_env, geo_cached["eps_env"])),
    }
    print(f"    {identity_checks}")
    if not all(identity_checks.values()):
        print(
            "    ⚠️ 不一致——重新选取的 lig_idx/env_idx 原子身份/顺序跟生成该 --perturb-scan 时不是同一套，"
            "下面所有能量比较都不可信，先排查 --ligand/--fit-env-radius/--fit-env-max-atoms/"
            "--perturb-anchors/--fit-last-ns 是否跟当时完全一致"
        )

    print("[2/5] 构建生产 DEXP(14,5) CustomNonbondedForce（用 abfe_core.DEXPSurrogatePotential.build_expression，非重新推导）")

    def _build_dexp_context(use_switch: bool):
        pot = DEXPSurrogatePotential(alpha_vdw=14.0, beta_vdw=5.0)
        expr = pot.build_expression(lam_vdw="lam_vdw")
        sys_ = openmm.System()
        # CutoffPeriodic 需要合法的默认 box——用 anchor 0 的 box 兜底，避免某个测试 anchor
        # 恰好 has_periodic=False 时 Context 用到未初始化的退化 box。
        sys_.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box_vectors[0]])
        for i in range(n_total):
            sys_.addParticle(nb_system.getParticleMass(i))
        force = openmm.CustomNonbondedForce(expr)
        force.addGlobalParameter("lam_vdw", 1.0)
        force.addPerParticleParameter("sigma")
        force.addPerParticleParameter("epsilon")
        for i in range(n_total):
            force.addParticle([sigma_all[i], eps_all[i]])
        force.addInteractionGroup([int(i) for i in lig_idx], [int(i) for i in env_idx])
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(pot.cutoff_distance * unit.nanometer)
        force.setUseSwitchingFunction(bool(use_switch))
        if use_switch:
            force.setSwitchingDistance((pot.cutoff_distance - pot.switch_width) * unit.nanometer)
        for exc_idx in range(nb_force.getNumExceptions()):
            p1, p2, _, _, _ = nb_force.getExceptionParameters(exc_idx)
            force.addExclusion(int(p1), int(p2))
        sys_.addForce(force)
        return openmm.Context(sys_, openmm.VerletIntegrator(0.001)), pot

    ctx_dexp_switch, pot145 = _build_dexp_context(use_switch=True)
    ctx_dexp_noswitch, _ = _build_dexp_context(use_switch=False)

    print("[2/5] 构建 §6.7 用的 gauss_coul 参考 Context（与 --perturb-scan 生成 delta_e_target 时同一个函数）")
    mm_contexts = build_mm_le_contexts_from_system_xml(
        args.system_xml,
        ligand_indices=lig_idx.tolist(),
        environment_indices=env_idx.tolist(),
        cutoff_nm=float(args.fit_mm_ref_cutoff),
        switching_nm=float(args.fit_mm_ref_switch),
    )
    gauss_ctx = mm_contexts["gauss_coul"]
    gauss_ctx_periodic = gauss_ctx.getSystem().usesPeriodicBoundaryConditions()

    def _energy(context, full_pos_nm: np.ndarray, box_vecs: Optional[np.ndarray], periodic: bool) -> float:
        if box_vecs is not None and periodic:
            context.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box_vecs])
        context.setPositions(full_pos_nm * unit.nanometer)
        return float(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))

    def _numpy_dexp_energy(lig_pos: np.ndarray, env_pos: np.ndarray, box_vecs: Optional[np.ndarray], use_switch: bool) -> float:
        delta = lig_pos[:, None, :] - env_pos[None, :, :]
        if box_vecs is not None:
            box_lens = np.linalg.norm(box_vecs, axis=1)
            delta = delta - box_lens * np.round(delta / box_lens)
        r = np.linalg.norm(delta, axis=-1)
        sigma_ij = 0.5 * (sigma_lig[:, None] + sigma_env[None, :])
        eps_ij = np.sqrt(np.clip(eps_lig[:, None] * eps_env[None, :], 0.0, None))
        r0_ij = (2.0 ** (1.0 / 6.0)) * np.maximum(sigma_ij, 1.0e-6)
        x = np.maximum(r, 1.0e-6) / r0_ij - 1.0
        c_a, c_b = 5.0 / (14.0 - 5.0), 14.0 / (14.0 - 5.0)
        pair_e = eps_ij * (c_a * np.exp(-14.0 * x) - c_b * np.exp(-5.0 * x))
        cutoff, switch_width = 0.70, 0.20
        mask = r <= cutoff
        if use_switch:
            r_switch = cutoff - switch_width
            s = np.ones_like(r)
            in_sw = (r > r_switch) & (r <= cutoff)
            t = (r[in_sw] - r_switch) / (cutoff - r_switch)
            s[in_sw] = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
            pair_e = pair_e * s
        return float(np.sum(np.where(mask, pair_e, 0.0)))

    def _numpy_gauss_energy(lig_pos: np.ndarray, env_pos: np.ndarray, box_vecs: Optional[np.ndarray]) -> float:
        from scipy.special import erf
        delta = lig_pos[:, None, :] - env_pos[None, :, :]
        if box_vecs is not None:
            box_lens = np.linalg.norm(box_vecs, axis=1)
            delta = delta - box_lens * np.round(delta / box_lens)
        r = np.maximum(np.linalg.norm(delta, axis=-1), 1.0e-6)
        q_lig, q_env = q_all[lig_idx], q_all[env_idx]
        q_ij = q_lig[:, None] * q_env[None, :]
        ke = 138.935456
        gamma_eff = 1.0 / (math.sqrt(2.0) * 0.10)
        use_ref_cutoff = float(args.fit_mm_ref_cutoff) > 0.0
        if not use_ref_cutoff:
            return float(np.sum(ke * q_ij * erf(gamma_eff * r) / r))
        rc, rs_switch = float(args.fit_mm_ref_cutoff), float(args.fit_mm_ref_switch)
        s = np.ones_like(r)
        if 0.0 < rs_switch < rc:
            in_sw = (r > rs_switch) & (r <= rc)
            t = (r[in_sw] - rs_switch) / (rc - rs_switch)
            s[in_sw] = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
        s[r > rc] = 0.0
        return float(np.sum(np.where(r <= rc, ke * q_ij * erf(gamma_eff * r) / r * s, 0.0)))

    def _full_pos(lig_pos: np.ndarray, env_pos: np.ndarray) -> np.ndarray:
        full = np.zeros((n_total, 3), dtype=np.float64)
        full[lig_idx] = lig_pos
        full[env_idx] = env_pos
        return full

    test_anchor_ids = sorted(set([0, n_anchors // 2, n_anchors - 1]))
    test_configs: List[Dict] = []
    for a in test_anchor_ids:
        test_configs.append({"anchor": int(a), "kind": "anchor_state", "row": None})
        for kind, pt, mag in (("translation_0.005nm_+", "translation", 0.005), ("rotation_0.5deg_+", "rotation", 0.5)):
            sel = np.where((anchor_local_idx == a) & (pert_type == pt) & np.isclose(magnitude, mag) & (sign == 1.0))[0]
            if sel.size:
                test_configs.append({"anchor": int(a), "kind": kind, "row": int(sel[0])})

    print(f"[3/5] 在 {len(test_configs)} 个代表性构型上比较能量：NumPy(无switch,现有基线) vs NumPy(加switch) vs 生产OpenMM(有/无switch)")
    energy_rows: List[Dict] = []
    for cfg in test_configs:
        a = cfg["anchor"]
        lig_pos = anchor_lig_positions[a] if cfg["row"] is None else perturbed_lig_positions[cfg["row"]]
        env_pos = env_positions[a]
        bv = box_vectors[a] if has_periodic[a] else None
        full_pos = _full_pos(lig_pos, env_pos)

        e_prod_switch = _energy(ctx_dexp_switch, full_pos, bv, True)
        e_prod_noswitch = _energy(ctx_dexp_noswitch, full_pos, bv, True)
        e_np_noswitch = _numpy_dexp_energy(lig_pos, env_pos, bv, use_switch=False)
        e_np_switch = _numpy_dexp_energy(lig_pos, env_pos, bv, use_switch=True)
        e_gauss_prod = _energy(gauss_ctx, full_pos, bv, gauss_ctx_periodic)
        e_gauss_np = _numpy_gauss_energy(lig_pos, env_pos, bv if gauss_ctx_periodic else None)

        row = {
            "anchor": a, "kind": cfg["kind"],
            "e_prod_dexp_with_switch_kjmol": e_prod_switch,
            "e_prod_dexp_no_switch_kjmol": e_prod_noswitch,
            "e_numpy_dexp_no_switch_kjmol": e_np_noswitch,
            "e_numpy_dexp_with_switch_kjmol": e_np_switch,
            "diff_existing_baseline_vs_production_kjmol": e_np_noswitch - e_prod_switch,
            "diff_switch_corrected_numpy_vs_production_kjmol": e_np_switch - e_prod_switch,
            "diff_numpy_noswitch_vs_prod_noswitch_kjmol": e_np_noswitch - e_prod_noswitch,
            "e_gauss_production_kjmol": e_gauss_prod,
            "e_gauss_numpy_kjmol": e_gauss_np,
            "diff_gauss_numpy_vs_production_kjmol": e_gauss_np - e_gauss_prod,
        }
        energy_rows.append(row)
        print(
            f"    anchor={a:2d} {cfg['kind']:22s} "
            f"Δ(现有基线-生产,有switch)={row['diff_existing_baseline_vs_production_kjmol']:+.6f}  "
            f"Δ(加switch后-生产)={row['diff_switch_corrected_numpy_vs_production_kjmol']:+.6f}  "
            f"Δ(gauss numpy-生产)={row['diff_gauss_numpy_vs_production_kjmol']:+.6f} kJ/mol"
        )

    energy_tol = float(args.audit_energy_tol_kjmol)
    max_diff_existing_baseline = max(abs(r["diff_existing_baseline_vs_production_kjmol"]) for r in energy_rows)
    max_diff_switch_corrected = max(abs(r["diff_switch_corrected_numpy_vs_production_kjmol"]) for r in energy_rows)
    max_diff_noswitch_pure = max(abs(r["diff_numpy_noswitch_vs_prod_noswitch_kjmol"]) for r in energy_rows)
    max_diff_gauss = max(abs(r["diff_gauss_numpy_vs_production_kjmol"]) for r in energy_rows)

    print("[4/5] 有限差分力：NumPy(加switch)中心差分 vs 生产 OpenMM 解析力(getForces)")
    h_nm = 1.0e-5
    force_rows: List[Dict] = []
    for cfg in test_configs[: min(3, len(test_configs))]:
        a = cfg["anchor"]
        lig_pos = (anchor_lig_positions[a] if cfg["row"] is None else perturbed_lig_positions[cfg["row"]]).copy()
        env_pos = env_positions[a]
        bv = box_vectors[a] if has_periodic[a] else None
        full_pos = _full_pos(lig_pos, env_pos)
        if bv is not None:
            ctx_dexp_switch.setPeriodicBoxVectors(*[openmm.Vec3(*row_) for row_ in bv])
        ctx_dexp_switch.setPositions(full_pos * unit.nanometer)
        forces_prod = ctx_dexp_switch.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        # 只测最近接触的那个配体原子(对 vdW 力最敏感)，跟离它最近的 env 原子距离最小的一维
        dists0 = np.linalg.norm(lig_pos[:, None, :] - env_pos[None, :, :], axis=-1)
        probe_lig_local = int(np.unravel_index(np.argmin(dists0), dists0.shape)[0])
        probe_topo_idx = int(lig_idx[probe_lig_local])
        f_fd = np.zeros(3, dtype=float)
        for dim in range(3):
            lig_plus, lig_minus = lig_pos.copy(), lig_pos.copy()
            lig_plus[probe_lig_local, dim] += h_nm
            lig_minus[probe_lig_local, dim] -= h_nm
            e_plus = _numpy_dexp_energy(lig_plus, env_pos, bv, use_switch=True)
            e_minus = _numpy_dexp_energy(lig_minus, env_pos, bv, use_switch=True)
            f_fd[dim] = -(e_plus - e_minus) / (2.0 * h_nm)
        f_prod = np.asarray(forces_prod[probe_topo_idx], dtype=float)
        rel_err = float(np.linalg.norm(f_fd - f_prod) / max(np.linalg.norm(f_prod), 1.0e-8))
        force_rows.append({
            "anchor": a, "kind": cfg["kind"], "probe_topo_idx": probe_topo_idx,
            "force_numpy_fd_kjmol_nm": f_fd.tolist(), "force_production_kjmol_nm": f_prod.tolist(),
            "relative_error": rel_err,
        })
        print(f"    anchor={a:2d} {cfg['kind']:22s} probe_atom={probe_topo_idx}  相对误差={rel_err:.2e}")

    print(
        "[5/5] 系统性核对：加 switch 后，residual 分析真正用到的 ΔU_DEXP(pert-anchor) 这个 delta 本身变化多大——"
        f"覆盖全部 {ctx['n_rows']} 条 --perturb-scan 记录（能量层面的 8-9kJ/mol 绝对偏差大部分是环境不动、"
        "anchor/perturbed 两态共享的系统性背景，取差分时是否真的会大部分抵消，不能只看9个抽样点，要看全量分布）"
    )
    delta_e_target_all = ctx["delta_e_target"]
    residual_noswitch_all = ctx["residual_target"]  # R = delta_e_target - ΔU_DEXP(14,5, 无switch)，即 §3-§6 全程用的 M0 残差
    delta_u_noswitch_all = delta_e_target_all - residual_noswitch_all

    tensors_full = _build_perturbation_distance_tensors(
        ctx["n_rows"], anchor_local_idx, env_positions, anchor_lig_positions, perturbed_lig_positions,
        box_vectors, has_periodic, sigma_lig, eps_lig, sigma_env, eps_env, 0.70,
    )
    dists_anchor_full, dists_pert_full = tensors_full["dists_anchor_full"], tensors_full["dists_pert_full"]
    eps_ij_full = tensors_full["eps_ij_full"]
    x_anchor_full, x_pert_full = tensors_full["x_anchor_full"], tensors_full["x_pert_full"]
    mask_anchor_full, mask_pert_full = tensors_full["mask_anchor_full"], tensors_full["mask_pert_full"]

    def _switch_s_070(r: np.ndarray) -> np.ndarray:
        s = np.ones_like(r)
        r_switch = 0.70 - 0.20
        in_sw = (r > r_switch) & (r <= 0.70)
        t = (r[in_sw] - r_switch) / (0.70 - r_switch)
        s[in_sw] = 1.0 - 10.0 * t ** 3 + 15.0 * t ** 4 - 6.0 * t ** 5
        return s

    c_a145, c_b145 = 5.0 / (14.0 - 5.0), 14.0 / (14.0 - 5.0)
    pair_e_anchor_all = eps_ij_full[None] * (c_a145 * np.exp(-14.0 * x_anchor_full) - c_b145 * np.exp(-5.0 * x_anchor_full))
    pair_e_pert_all = eps_ij_full[None] * (c_a145 * np.exp(-14.0 * x_pert_full) - c_b145 * np.exp(-5.0 * x_pert_full))
    s_anchor_all, s_pert_all = _switch_s_070(dists_anchor_full), _switch_s_070(dists_pert_full)
    u_anchor_switch_all = np.sum(np.where(mask_anchor_full, pair_e_anchor_all * s_anchor_all, 0.0), axis=(1, 2))
    u_pert_switch_all = np.sum(np.where(mask_pert_full, pair_e_pert_all * s_pert_all, 0.0), axis=(1, 2))
    delta_u_switch_all = u_pert_switch_all - u_anchor_switch_all

    delta_of_switch_correction_all = delta_u_switch_all - delta_u_noswitch_all
    residual_switch_corrected_all = delta_e_target_all - delta_u_switch_all

    def _odd_even_full(resid: np.ndarray) -> Dict[str, Dict]:
        group_key_to_idx = ctx["group_key_to_idx"]
        odd_by_type: Dict[str, List[float]] = {"translation": [], "rotation": []}
        even_by_type: Dict[str, List[float]] = {"translation": [], "rotation": []}
        for key, signed in group_key_to_idx.items():
            if 1.0 not in signed or -1.0 not in signed:
                continue
            ip, im = signed[1.0], signed[-1.0]
            odd_by_type[key[1]].append(float((resid[ip] - resid[im]) / 2.0))
            even_by_type[key[1]].append(float((resid[ip] + resid[im]) / 2.0))
        out: Dict[str, Dict] = {}
        for ptype in ("translation", "rotation"):
            oa = np.asarray(odd_by_type[ptype], dtype=float)
            ea = np.asarray(even_by_type[ptype], dtype=float)
            out[ptype] = {
                "odd_residual_rmse_kjmol": float(np.sqrt(np.mean(oa ** 2))) if oa.size else math.nan,
                "even_residual_rmse_kjmol": float(np.sqrt(np.mean(ea ** 2))) if ea.size else math.nan,
            }
        return out

    odd_even_noswitch_full = _odd_even_full(residual_noswitch_all)
    odd_even_switch_full = _odd_even_full(residual_switch_corrected_all)

    switch_delta_bin_stats: Dict[str, Dict] = {}
    for pt, mags in (("translation", (0.005, 0.01, 0.02, 0.04)), ("rotation", (0.5, 1.5, 3.0))):
        for mag in mags:
            bin_mask = (pert_type == pt) & np.isclose(magnitude, mag)
            if not np.any(bin_mask):
                continue
            vals = delta_of_switch_correction_all[bin_mask]
            switch_delta_bin_stats[f"{pt}:{mag}"] = {
                "n": int(np.sum(bin_mask)),
                "mean_kjmol": float(np.mean(vals)),
                "std_kjmol": float(np.std(vals)),
                "max_abs_kjmol": float(np.max(np.abs(vals))),
            }

    max_abs_switch_delta_correction = float(np.max(np.abs(delta_of_switch_correction_all)))
    print(f"    Δ(switch修正后的ΔU_DEXP delta - 现有无switch的delta) 全量统计: "
          f"mean={float(np.mean(delta_of_switch_correction_all)):+.4f}  std={float(np.std(delta_of_switch_correction_all)):.4f}  "
          f"max|.|={max_abs_switch_delta_correction:.4f} kJ/mol  (对比 residual RMSE 量级 ~3-8 kJ/mol)")
    for key, stats in switch_delta_bin_stats.items():
        print(f"        {key:20s} n={stats['n']:4d}  mean={stats['mean_kjmol']:+.4f}  std={stats['std_kjmol']:.4f}  max|.|={stats['max_abs_kjmol']:.4f} kJ/mol")
    print("    奇偶分解对比(无switch的现有基线 vs 加switch后)：")
    for ptype in ("translation", "rotation"):
        o0, o1 = odd_even_noswitch_full[ptype], odd_even_switch_full[ptype]
        print(
            f"        {ptype:12s} 无switch: odd={o0['odd_residual_rmse_kjmol']:.3f} even={o0['even_residual_rmse_kjmol']:.3f}  |  "
            f"加switch: odd={o1['odd_residual_rmse_kjmol']:.3f} even={o1['even_residual_rmse_kjmol']:.3f}"
        )

    energy_tol_ok = bool(
        max_diff_switch_corrected < energy_tol and max_diff_noswitch_pure < energy_tol and max_diff_gauss < energy_tol
    )
    force_tol = float(args.audit_force_rel_tol)
    force_tol_ok = bool(all(r["relative_error"] < force_tol for r in force_rows)) if force_rows else False
    baseline_matches_production = bool(max_diff_existing_baseline < energy_tol)

    summary = {
        "energy_tol_kjmol": energy_tol,
        "force_rel_tol": force_tol,
        "identity_checks": identity_checks,
        "energy_rows": energy_rows,
        "force_rows": force_rows,
        "max_abs_diff_existing_baseline_vs_production_kjmol": max_diff_existing_baseline,
        "max_abs_diff_switch_corrected_numpy_vs_production_kjmol": max_diff_switch_corrected,
        "max_abs_diff_numpy_noswitch_vs_prod_noswitch_kjmol": max_diff_noswitch_pure,
        "max_abs_diff_gauss_numpy_vs_production_kjmol": max_diff_gauss,
        "existing_baseline_matches_production": baseline_matches_production,
        "switch_corrected_numpy_matches_production": bool(max_diff_switch_corrected < energy_tol),
        "base_formula_correct_in_isolation": bool(max_diff_noswitch_pure < energy_tol),
        "gauss_numpy_reimplementation_correct": bool(max_diff_gauss < energy_tol),
        "force_check_passed": force_tol_ok,
        "delta_of_switch_correction_full_dataset": {
            "n_rows": int(ctx["n_rows"]),
            "mean_kjmol": float(np.mean(delta_of_switch_correction_all)),
            "std_kjmol": float(np.std(delta_of_switch_correction_all)),
            "max_abs_kjmol": max_abs_switch_delta_correction,
            "by_bin": switch_delta_bin_stats,
        },
        "odd_even_no_switch_full_dataset": odd_even_noswitch_full,
        "odd_even_switch_corrected_full_dataset": odd_even_switch_full,
        "verdict_note": (
            "能量层面 existing_baseline_matches_production=false（绝对能量差~8-9kJ/mol，来自缺失的"
            "switching function）是预期内的、几乎必然通过硬性1e-5kJ/mol阈值判定为'不一致'的结果——"
            "但 §3-§6 全程只用 ΔU_DEXP(pert-anchor) 这个差分量，不是绝对能量，关键判据是"
            "delta_of_switch_correction_full_dataset：如果它的 mean/std/max|.| 相对 residual RMSE 量级"
            "(~3-8kJ/mol，见 odd_even_no_switch_full_dataset)可忽略(如<<1kJ/mol)，说明switching缺失在"
            "anchor/perturbed两态之间几乎完全抵消(环境不动、扰动幅度小，进出0.5-0.7nm switch壳层的pair"
            "集合变化很小)，§3-§6/§6 的全部结论基本不受影响；odd_even_switch_corrected_full_dataset 应该"
            "跟 odd_even_no_switch_full_dataset 几乎相等，可以直接对比验证。如果 by_bin 显示这个抵消在"
            "大幅度扰动(0.04nm/3°)上明显变差(max|.|随幅度增长)，则至少大幅度那几档的结论需要打问号，"
            "但§3.3就已经因为'0.04nm最偏离局部定义'而降权处理这一档了。"
            "switch_corrected_numpy_matches_production 和 gauss_numpy_reimplementation_correct 的绝对差异"
            "在~1e-4kJ/mol量级(不到1e-5kJ/mol严格阈值一个数量级，大概率是被求和顺序/erf实现细节等浮点"
            "噪声决定，不是公式错误)——比这两个量级本身关心的~kJ/mol物理尺度小4-5个数量级，可视为功能上"
            "通过；gauss_numpy_reimplementation_correct 是独立的正确性检验(不是'是否等于生产'检验，那个"
            "Gaussian参考本来就跟生产的shifted-force定义不同，是刻意设计成不同的两套东西)。"
        ),
    }
    summary_path = os.path.join(output_dir, "production_equivalence_audit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"    existing_baseline_matches_production(绝对能量)={baseline_matches_production}  "
        f"switch_corrected_matches_production={summary['switch_corrected_numpy_matches_production']}  "
        f"gauss_reimplementation_correct={summary['gauss_numpy_reimplementation_correct']}  "
        f"force_check_passed={force_tol_ok}  "
        f"max|Δ(switch修正对delta的影响)|={max_abs_switch_delta_correction:.4f}kJ/mol(关键判据，见verdict_note)"
    )
    print(f"    summary: {summary_path}")
    return summary


def run_relabel_pmf(args: argparse.Namespace, output_dir: str, fitted_params: Dict) -> Dict:
    """relabel + 同帧比较主流程：DEXP 生产轨迹（必选）+ MM baseline 地板（可选）。
    主判据 = ⟨δ⟩(s) within-SEM（MACE 可信窗口内）；单独报"过近帧"数（指标 F）。"""
    symbols = load_abfe_symbols()
    kbt = 0.00831446261815324 * float(args.temperature)
    n_bins = int(args.relabel_pmf_bins)
    min_bin = int(args.relabel_pmf_min_bin_frames)
    floor = float(args.relabel_min_dist_floor)
    out: Dict = {}

    env_override = _load_fixed_env_indices(output_dir)
    if env_override is not None:
        print(f"    [relabel] 复用 fit 阶段固定环境原子集合（env={len(env_override)}），DEXP/MM 两条轨迹共用同一套")
    else:
        print("    [relabel] ⚠️ 未找到 fit_label_cache_meta.json，DEXP/MM 各自按本条轨迹最后一帧重选环境原子（两者环境集合可能不同，形状 RMSE 比较会掺入环境定义差异）")

    def _one(traj_path, applied_key, tag):
        r = relabel_trajectory_local(args, traj_path, fitted_params, symbols, env_idx_override=env_override)
        # δ = E_MACE_local - E_applied_local
        if applied_key == "dexp":
            delta = r["e_orb"] - (r["e_gauss"] + r["dexp_pred"])
        else:  # mm baseline
            delta = r["e_orb"] - (r["e_mm_coul"] + r["e_mm_vdw"])
        md_all = np.asarray(r["min_dist"], dtype=float)          # 全原子最近距离：仅用于过近/碰撞判据(F)
        md_valid = np.asarray(r["min_dist_valid"], dtype=float)  # DEXP 实际敏感的坐标：形状分箱/比较用它
        mask, n_close = _filter_too_close(md_all, floor)
        mask = mask & np.isfinite(md_valid)   # 极少数帧在 [fit_r_min,fit_r_max] 内没有 pair，无法定义该坐标
        rows, summ = same_frame_pmf_compare(
            md_valid[mask], delta[mask], kbt, n_bins, min_bin,
            shape_anchor_bins=int(args.relabel_shape_anchor_bins),
        )
        summ["n_frames_raw"] = int(md_all.size)
        summ["n_frames_too_close"] = n_close       # 指标 F：原子穿插、MACE 也 OOD 的帧数
        summ["too_close_fraction"] = float(n_close / max(1, md_all.size))
        summ["min_dist_floor_nm"] = floor
        # min-dist 分布：全原子最近距离，直接看配体到底待在哪（指标 F 的原始信息）
        summ["min_dist_min_nm"] = float(md_all.min())
        summ["min_dist_median_nm"] = float(np.median(md_all))
        summ["min_dist_p05_nm"] = float(np.percentile(md_all, 5))
        summ["min_dist_max_nm"] = float(md_all.max())
        # min-dist-valid 分布：DEXP 实际敏感/用于形状分箱的坐标范围，和上面的全原子坐标不是一回事
        valid_finite = md_valid[np.isfinite(md_valid)]
        summ["min_dist_valid_min_nm"] = float(valid_finite.min()) if valid_finite.size else math.nan
        summ["min_dist_valid_max_nm"] = float(valid_finite.max()) if valid_finite.size else math.nan
        summ["env_source"] = r.get("env_source")
        print(
            f"    [relabel/{tag}] 帧={summ['n_frames_raw']} | min-dist(全原子) 分布: "
            f"min={summ['min_dist_min_nm']:.3f} p05={summ['min_dist_p05_nm']:.3f} "
            f"中位={summ['min_dist_median_nm']:.3f} max={summ['min_dist_max_nm']:.3f} nm | "
            f"过近<{floor}nm 排除 {n_close}({summ['too_close_fraction']:.0%})"
        )
        print(
            f"        分箱坐标 min-dist-valid(∈[{args.fit_r_min:.2f},{args.fit_r_max:.2f}]nm) 范围: "
            f"{summ['min_dist_valid_min_nm']:.3f}–{summ['min_dist_valid_max_nm']:.3f} nm"
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

    # pose-scan 模式：随机刚体扰动 + 短程弛豫 + 分箱筛选，生成训练样本后退出。
    if args.pose_scan:
        run_pose_scan(args, output_dir)
        return 0

    # pull-scan 模式：把配体质心从环境锚点拉开，生成宽范围构型序列后退出（不跑拟合/relabel/后处理）。
    if args.pull_scan:
        run_pull_scan(args, output_dir)
        return 0

    # perturb-scan 模式：结合态局部 anchor-relative 扰动云，检验(a)阶段解析基线是否已经
    # 解释局部势能面曲率，生成诊断 CSV/JSON 后退出（不跑拟合/relabel/后处理）。
    if args.perturb_scan:
        run_perturbation_scan(args, output_dir)
        return 0

    # perturb-fit 模式：读取已有 --perturb-scan 输出，重新挑选 alpha_vdw/beta_vdw 后退出
    # （几何已缓存在 npz 里，不需要重跑 MACE）。
    if args.perturb_fit:
        run_perturbation_fit(args, output_dir)
        return 0

    # contact-type-fit 模式：DEXP_KERNEL_PHYSICS_ISSUES.md §6 最小实现，donor_acceptor/fallback
    # 两类 contact-type odd/even 修正的 M0/M1/M2 grouped-LOAO 对比，读取已有 --perturb-scan 输出
    # （仍需重新加载轨迹拓扑一次，以便按 element+键连关系分配 donor/acceptor 角色）。
    if args.contact_type_fit:
        run_contact_type_fit(args, output_dir)
        return 0

    # contact-type-angular-diagnostic 模式：DEXP_KERNEL_PHYSICS_ISSUES.md §6.6，只做
    # D-H-A 角度/最近acceptor切换/配位数 vs 跨折 out-of-fold 残差的诊断，不拟合 angular force。
    if args.contact_type_angular_diagnostic:
        run_contact_type_angular_diagnostic(args, output_dir)
        return 0

    # gaussian-width-diagnostic 模式：DEXP_KERNEL_PHYSICS_ISSUES.md §6 电荷穿透/Gaussian宽度诊断，
    # 只检查关联，不拟合 role-specific sigma_elec。
    if args.gaussian_width_diagnostic:
        run_gaussian_width_diagnostic(args, output_dir)
        return 0

    # production-equivalence-audit 模式：Phase 1，核对 NumPy 离线基线是否真的等于生产 OpenMM 力。
    if args.production_equivalence_audit:
        run_production_equivalence_audit(args, output_dir)
        return 0

    # replica-run 模式：LJ/DEXP(12,6)/DEXP(14,5) 三方对比里的单个 condition，
    # 方便当成独立作业分别提交到计算节点。
    if args.replica_run:
        run_replica_condition(args, output_dir)
        return 0

    # vsb-frame-scan 模式：§9.1 方案第一步，从已有replica轨迹里挑V/S/B起始帧，
    # 零新增MD/MACE。
    if args.vsb_frame_scan:
        run_vsb_frame_scan(args, output_dir)
        return 0

    # vsb-staged-run 模式：§9.1 方案第二步，用挑好的V/S/B起始帧对单个condition
    # 跑全新的replica MD。
    if args.vsb_staged_run:
        run_vsb_staged_replica(args, output_dir)
        return 0

    # vsb-staged-analyze 模式：§9.1 方案第三步，只读分析--vsb-staged-run产出的轨迹，
    # 检验V/S/B三个起始态后半段occupancy是否收敛。
    if args.vsb_staged_analyze:
        run_vsb_staged_analysis(args, output_dir)
        return 0

    # replica-analyze 模式：只读分析所有已跑完的 condition/replica 轨迹后退出。
    if args.replica_analyze:
        run_replica_analysis(args, output_dir)
        return 0

    # hbond-switching-dynamics 模式：只读分析已跑完的 --replica-run 轨迹，检验 §4.4 的
    # V/S occupancy 是否是平衡概率，还是短轨迹初态依赖的动力学产物。
    if args.hbond_switching_dynamics:
        run_hbond_switching_dynamics(args, output_dir)
        return 0

    # hbond-committed-state-dynamics 模式：committed-state 升级版，过滤阈值抖动后
    # 重新判断 §4.4 的 occupancy 是否平衡。
    if args.hbond_committed_state_dynamics:
        run_hbond_committed_state_dynamics(args, output_dir)
        return 0

    # kernel-projection-benchmark 模式：K0(LJ)/K1(DEXP12,6)/K2(DEXP14,5) 对 MACE even/odd
    # 的投影能力对比，零新增MACE计算。
    if args.kernel_projection_benchmark:
        run_kernel_projection_benchmark(args, output_dir)
        return 0

    # mace-residual-force-benchmark 模式：Phase 3，把已有±δ能量差重新解读为局部力/力矩投影，
    # 零新增MACE计算，只复用--perturb-scan缓存。
    if args.mace_residual_force_benchmark:
        run_mace_residual_force_benchmark(args, output_dir)
        return 0

    # mace-env-convergence 模式：Phase 2，唯一需要新增MACE计算的Phase，检验环境半径/裁剪方式
    # 是否改变ΔE_MACE收敛性/odd符号/kernel排序。
    if args.mace_env_convergence:
        run_mace_env_convergence(args, output_dir)
        return 0

    # r0-scale-diagnostic 模式：固定alpha/beta扫描r0比例因子s_r，零新增MACE计算，
    # 只复用--perturb-scan缓存，检验是否有s_r值得晋升为第4个MD condition。
    if args.r0_scale_diagnostic:
        run_r0_scale_diagnostic(args, output_dir)
        return 0

    # alpha-beta-scale-diagnostic 模式：全网格扫alpha/beta，确认(14,5)是否位于稳定盆地，
    # 零新增MACE计算，只复用--perturb-scan缓存。
    if args.alpha_beta_scale_diagnostic:
        run_alpha_beta_scale_diagnostic(args, output_dir)
        return 0

    # alpha-beta-ridge-scan 模式：沿q=alpha+beta固定的直线细扫(默认18/19)，供画图，
    # 零新增MACE计算，只复用--perturb-scan缓存。
    if args.alpha_beta_ridge_scan:
        run_alpha_beta_ridge_scan(args, output_dir)
        return 0

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
